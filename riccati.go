package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// RiccatiWorkspace is reusable scratch storage for Care and Dare, sized by
// NewRiccatiWorkspace for a state dimension n and m columns of B. It serves
// any problem with at most n states and m inputs; a smaller one returns
// ErrDimensionMismatch. Kalman, Kalmd and Lqe solve the dual problem, so
// their workspace needs m = number of measured outputs.
//
// A workspace is not safe for concurrent use, and results computed with it
// share its storage until the next call that reuses it.
type RiccatiWorkspace struct {
	n, m        int
	rChol       []float64
	aWork       []float64
	qWork       []float64
	rinvBt      []float64
	rinvSt      []float64
	g           []float64
	h           []float64
	wr          []float64
	wi          []float64
	vs          []float64
	bwork       []bool
	work        []float64
	u11         []float64
	u21         []float64
	ipiv        []int
	xData       []float64
	eig         []complex128
	kData       []float64
	z           []float64
	beta        []float64
	pencilH     []float64
	pencilJ     []float64
	pencilInput []float64
	tau         []float64
	btx         []float64
	rbar        []float64
	iwork       []int
	scaleT      []float64
	scaledA     []float64
	scaledE     []float64
	scaledQ     []float64
	scaledB     []float64
	scaledS     []float64
	scaleProxy  []float64
}

// NewRiccatiWorkspace returns a workspace for problems with up to n states
// and m inputs.
func NewRiccatiWorkspace(n, m int) *RiccatiWorkspace {
	nn := 2 * n
	ws := &RiccatiWorkspace{
		n:           n,
		m:           m,
		rChol:       make([]float64, m*m),
		aWork:       make([]float64, n*n),
		qWork:       make([]float64, n*n),
		rinvBt:      make([]float64, m*n),
		rinvSt:      make([]float64, m*n),
		g:           make([]float64, n*n),
		h:           make([]float64, nn*nn),
		wr:          make([]float64, nn),
		wi:          make([]float64, nn),
		vs:          make([]float64, nn*nn),
		bwork:       make([]bool, nn),
		work:        make([]float64, nn*50),
		u11:         make([]float64, n*n),
		u21:         make([]float64, n*n),
		ipiv:        make([]int, n),
		xData:       make([]float64, n*n),
		eig:         make([]complex128, n),
		kData:       make([]float64, m*n),
		z:           make([]float64, nn*nn),
		beta:        make([]float64, nn),
		pencilH:     make([]float64, (nn+m)*nn),
		pencilJ:     make([]float64, (nn+m)*nn),
		pencilInput: make([]float64, (nn+m)*m),
		tau:         make([]float64, m),
		btx:         make([]float64, m*n),
		rbar:        make([]float64, m*m),
		iwork:       make([]int, n),
	}
	scale := make([]float64, n+3*n*n+2*n*m+(1+m)*n)
	ws.scaleT, scale = scale[:n:n], scale[n:]
	ws.scaledA, scale = scale[:n*n:n*n], scale[n*n:]
	ws.scaledE, scale = scale[:n*n:n*n], scale[n*n:]
	ws.scaledQ, scale = scale[:n*n:n*n], scale[n*n:]
	ws.scaledB, scale = scale[:n*m:n*m], scale[n*m:]
	ws.scaledS, ws.scaleProxy = scale[:n*m:n*m], scale[n*m:]
	return ws
}

// RiccatiOpts holds the optional arguments of Care and Dare; nil means none.
type RiccatiOpts struct {
	// S is the optional cross-term matrix. It is read during the call.
	S *mat.Dense
	// E is the optional nonsingular descriptor matrix of E x' = Ax + Bu
	// (E x[k+1] = Ax[k] + Bu[k]), as in MATLAB icare/idare. nil means I.
	// It is read during the call.
	E *mat.Dense
	// Workspace supplies reusable scratch storage. Results may share its storage
	// until the next call that reuses the same workspace, and it must not be
	// shared across goroutines.
	Workspace *RiccatiWorkspace
	// NoScaling disables the default power-of-two state and cost scaling,
	// as the MATLAB icare/idare 'noscaling' option. A pencil solve (Dare,
	// or Care with E) that fails, or leaves a Riccati residual above 1e-12
	// of its terms, is still retried with cost scaling.
	NoScaling bool
}

// RiccatiResult is the stabilizing solution of Care or Dare.
type RiccatiResult struct {
	// X is the stabilizing solution, K the state-feedback gain and Eig the
	// closed-loop eigenvalues. They are caller-owned unless a workspace was
	// supplied.
	X   *mat.Dense
	K   *mat.Dense
	Eig []complex128
	// Rcnd is the reciprocal 1-norm condition estimate of U11, where
	// [U11; U21] spans the stable invariant subspace and X = U21·U11⁻¹. A
	// tiny Rcnd means X may be inaccurate. U11 is that of the state- and
	// cost-scaled problem actually solved.
	Rcnd float64
}

// riccatiArgs rejects nil or non-finite Riccati data before it reaches
// LAPACK.
func riccatiArgs(op string, A, B, Q, R *mat.Dense, opts *RiccatiOpts) error {
	for _, a := range []struct {
		name string
		m    *mat.Dense
	}{{"A", A}, {"B", B}, {"Q", Q}, {"R", R}} {
		if err := requireFiniteDense(op, a.name, a.m); err != nil {
			return err
		}
	}
	if opts == nil {
		return nil
	}
	if opts.S != nil {
		if err := requireFiniteDense(op, "S", opts.S); err != nil {
			return err
		}
	}
	if opts.E != nil {
		if err := requireFiniteDense(op, "E", opts.E); err != nil {
			return err
		}
	}
	return nil
}

func (ws *RiccatiWorkspace) fits(n, m int) error {
	if ws.n < n || ws.m < m {
		return fmt.Errorf("workspace sized for n=%d m=%d, need n=%d m=%d: %w", ws.n, ws.m, n, m, ErrDimensionMismatch)
	}
	return nil
}

// Care solves the continuous algebraic Riccati equation:
//
//	A'X + XA - (XB+S)*R⁻¹*(B'X+S') + Q = 0
//
// When opts is nil or opts.S is nil, the cross-term is zero:
//
//	A'X + XA - XB*R⁻¹*B'X + Q = 0
//
// A is n×n, B is n×m, Q is n×n symmetric, R is m×m symmetric positive definite.
//
// With opts.E, Care solves the generalized equation of MATLAB icare
// (https://www.mathworks.com/help/control/ref/icare.html)
//
//	A'XE + E'XA - (E'XB+S)*R⁻¹*(B'XE+S') + Q = 0,  K = R⁻¹*(B'XE+S')
//
// from the stable deflating subspace of the extended pencil, and Eig holds
// the generalized eigenvalues of (A-BK, E). X is the generalized solution;
// E'XE solves the Care of the explicit model (E⁻¹A, E⁻¹B). A singular E
// returns ErrDescriptorSingular. Nil or NaN/Inf arguments return
// ErrInvalidArgument.
//
// As MATLAB icare, Care first balances the states by an exact power-of-two
// diagonal scaling, so X, K and Eig do not depend on the units of x. When
// ‖X‖ is far from the scale of the Hamiltonian (for example a cost Q, R
// tiny against A), Care solves again with Q, R, S divided by a power of
// two near ‖X‖, which keeps the relative accuracy of X. opts.NoScaling
// disables both.
func Care(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	if err := riccatiArgs("Care", A, B, Q, R, opts); err != nil {
		return nil, err
	}
	res, err := care(A, B, Q, R, opts)
	if err != nil {
		return nil, fmt.Errorf("Care: %w", err)
	}
	return res, nil
}

func care(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	problem, err := newRiccatiProblem(A, B, Q, R, opts)
	if err != nil {
		return nil, err
	}
	n, m := problem.n, problem.m
	if err := problem.ws.fits(n, m); err != nil {
		return nil, err
	}
	problem.scale(opts)
	A, B, Q, S, ws := problem.A, problem.B, problem.Q, problem.S, problem.ws

	// Cholesky factor R
	rChol := ws.rChol[:m*m]
	rRaw := R.RawMatrix()
	copyStrided(rChol, m, rRaw.Data, rRaw.Stride, m, m)
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
	}
	if problem.E != nil {
		return problem.descriptorCare()
	}

	// Working copies of A and Q for cross-term transformation
	aWork := ws.aWork[:n*n]
	aRaw := A.RawMatrix()
	copyStrided(aWork, n, aRaw.Data, aRaw.Stride, n, n)
	qWork := ws.qWork[:n*n]
	qRaw := Q.RawMatrix()
	copyStrided(qWork, n, qRaw.Data, qRaw.Stride, n, n)

	// Compute R⁻¹*B' (m×n): solve R*W = B' via Dpotrs
	rinvBt := ws.rinvBt[:m*n]
	bRaw := B.RawMatrix()
	for i := range n {
		for j := range m {
			rinvBt[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rChol, m, rinvBt, n)

	if S != nil {
		// Abar = A - B*R^-1*S'
		// Z = R^-1*S' (m x n): solve R*Z = S'
		rinvSt := ws.rinvSt[:m*n]
		sRaw := S.RawMatrix()
		for i := range n {
			for j := range m {
				rinvSt[j*n+i] = sRaw.Data[i*sRaw.Stride+j]
			}
		}
		impl.Dpotrs(blas.Upper, m, n, rChol, m, rinvSt, n)

		// aWork -= B * Z
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: aWork, Stride: n})

		// qWork -= S * Z
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: sRaw.Data, Stride: sRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n})
		symmetrize(qWork, n, n)
	}

	// G = B * R⁻¹ * B' (n×n symmetric)
	g := ws.g[:n*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: m, Cols: n, Data: rinvBt, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: g, Stride: n})
	symmetrize(g, n, n)

	nn := 2 * n
	wr := ws.wr[:nn]
	wi := ws.wi[:nn]
	vs := ws.vs[:nn*nn]
	bwork := ws.bwork[:nn]
	selctg := func(wr, wi float64) bool { return wr < 0 }
	var norms riccatiNorms
	var prev *riccatiPass
	var X *mat.Dense
	var xData []float64
	var rcnd float64
	for {
		// Hamiltonian of the cost-scaled problem: H = [[A, -σG], [-Q/σ, -A']]
		sigma, isigma := problem.sigma, 1/problem.sigma
		h := ws.h[:nn*nn]
		for i := range n {
			for j := range n {
				h[i*nn+j] = aWork[i*n+j]
				h[i*nn+n+j] = -sigma * g[i*n+j]
				h[(n+i)*nn+j] = -isigma * qWork[i*n+j]
				h[(n+i)*nn+n+j] = -aWork[j*n+i]
			}
		}

		var workQuery [1]float64
		impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
			nn, h, nn, wr, wi, vs, nn, workQuery[:], -1, bwork)
		lwork := int(workQuery[0])
		if len(ws.work) < lwork {
			ws.work = make([]float64, lwork)
		}
		sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
			nn, h, nn, wr, wi, vs, nn, ws.work, lwork, bwork)
		err := ErrSchurFailed
		if ok {
			err = ErrNoStabilizing
			if sdim == n {
				X, xData, rcnd, err = problem.stabilizingSolution(vs)
			}
		}
		if err != nil || (prev != nil && !finiteSlice(xData)) {
			if prev == nil {
				return nil, err
			}
			X, xData, rcnd = problem.restorePass(prev)
			break
		}
		if prev == nil && problem.costScaling {
			norms = riccatiNorms{
				a:     impl.Dlange(lapack.Frobenius, n, n, aWork, n, nil),
				q:     impl.Dlange(lapack.Frobenius, n, n, qWork, n, nil),
				g:     impl.Dlange(lapack.Frobenius, n, n, g, n, nil),
				kappa: 1,
			}
		}
		if !problem.rescaleCost(true, norms, xData, &prev, rcnd) {
			if problem.keepBest(true, prev, xData) {
				X, xData, rcnd = problem.restorePass(prev)
			}
			break
		}
	}

	// Closed-loop eigenvalues
	eig := ws.eig[:n]
	for i := range n {
		eig[i] = complex(wr[i], wi[i])
	}

	// Gain K = R⁻¹ * (B'X + S')
	// kData = B'*X (m×n)
	kData := ws.kData[:m*n]
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n})
	if S != nil {
		sRaw := S.RawMatrix()
		for j := range m {
			row := kData[j*n:]
			for i := range n {
				row[i] += sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rChol, m, kData, n)
	K := mat.NewDense(m, n, kData)

	return problem.result(X, K, eig, rcnd), nil
}

// Dare solves the discrete algebraic Riccati equation:
//
//	A'XA - X - (A'XB+S)*(R+B'XB)⁻¹*(B'XA+S') + Q = 0
//
// When opts is nil or opts.S is nil, the cross-term is zero:
//
//	A'XA - X - A'XB*(R+B'XB)⁻¹*B'XA + Q = 0
//
// A is n×n, B is n×m, Q is n×n symmetric, R is m×m symmetric positive definite.
//
// With opts.E, Dare solves the generalized equation of MATLAB idare
// (https://www.mathworks.com/help/control/ref/idare.html)
//
//	A'XA - E'XE - (A'XB+S)*(R+B'XB)⁻¹*(B'XA+S') + Q = 0
//
// with the same K formula, and Eig holds the generalized eigenvalues of
// (A-BK, E). E'XE solves the Dare of the explicit model (E⁻¹A, E⁻¹B). A
// singular E returns ErrDescriptorSingular. Nil or NaN/Inf arguments return
// ErrInvalidArgument.
//
// Dare always deflates the extended pencil, as MATLAB idare does, so A may
// be singular or have near-zero eigenvalues. A solution with R+B'XB not
// positive definite returns ErrNoStabilizing.
//
// Dare scales the states and the cost as Care does (MATLAB idare);
// opts.NoScaling disables it. An unscaled pencil that fails QZ or the
// stable split is solved again cost-scaled before an error is returned.
func Dare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	if err := riccatiArgs("Dare", A, B, Q, R, opts); err != nil {
		return nil, err
	}
	res, err := dare(A, B, Q, R, opts)
	if err != nil {
		return nil, fmt.Errorf("Dare: %w", err)
	}
	return res, nil
}

func dare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	problem, err := newRiccatiProblem(A, B, Q, R, opts)
	if err != nil {
		return nil, err
	}
	n, m := problem.n, problem.m
	if err := problem.ws.fits(n, m); err != nil {
		return nil, err
	}
	problem.scale(opts)
	A, B, S, ws := problem.A, problem.B, problem.S, problem.ws

	// Cholesky factor R
	rChol := ws.rChol[:m*m]
	rRaw := R.RawMatrix()
	copyStrided(rChol, m, rRaw.Data, rRaw.Stride, m, m)
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
	}

	aRaw := A.RawMatrix()
	bRaw := B.RawMatrix()

	subspace, X, xData, rcnd, err := problem.pencilSolution(false)
	if err != nil {
		return nil, err
	}

	// Closed-loop eigenvalues
	eig := ws.eig[:n]
	for i := range n {
		eig[i] = complex(subspace.alphaR[i]/subspace.beta[i], subspace.alphaI[i]/subspace.beta[i])
	}

	// Gain K = (R + B'XB)⁻¹ * (B'XA + S')
	btx := ws.btx[:m*n]
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n})

	rbar := ws.rbar[:m*m]
	copyStrided(rbar, m, rRaw.Data, rRaw.Stride, m, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
		blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		1, blas64.General{Rows: m, Cols: m, Data: rbar, Stride: m})

	if !impl.Dpotrf(blas.Upper, m, rbar, m) {
		return nil, ErrNoStabilizing
	}

	// BtXA = BtX * A (m×n)
	kData := ws.kData[:m*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: aRaw.Data, Stride: aRaw.Stride},
		0, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n})
	if S != nil {
		sRaw := S.RawMatrix()
		for j := range m {
			row := kData[j*n:]
			for i := range n {
				row[i] += sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rbar, m, kData, n)
	K := mat.NewDense(m, n, kData)
	if err := problem.rescued(false, K); err != nil {
		return nil, err
	}
	return problem.result(X, K, eig, rcnd), nil
}

// riccatiNorms are the Frobenius norms of the pencil blocks that cost
// scaling balances: a of A, Ā = A − BR⁻¹Sᵀ (and E), q of Q̄ = Q − SR⁻¹Sᵀ,
// g of G = BR⁻¹Bᵀ. kappa is the error amplification of the given pencil's
// input columns [B; −S; R] in its QR, max(‖B‖, ‖S‖, ‖R‖)/min(‖B‖, ‖R‖)
// (1 for the Hamiltonian, which forms G).
type riccatiNorms struct{ a, q, g, kappa float64 }

// cost is the predicted error amplification eps·cost/sep of X, of norm x,
// for the cost scale sigma (see rescaleCost).
func (norms riccatiNorms) cost(x, sigma float64) float64 {
	return max(norms.a, sigma*norms.g, norms.q/sigma) * (x/sigma + sigma/x)
}

// candidates returns the cost scales near x and √(q/g), 0 when undefined.
func (norms riccatiNorms) candidates(x float64) [2]float64 {
	var c [2]float64
	if x > 0 && !math.IsInf(x, 0) {
		c[0] = pow2Near(x)
	}
	if norms.q > 0 && norms.g > 0 {
		c[1] = pow2Near(math.Sqrt(norms.q) / math.Sqrt(norms.g))
	}
	return c
}

// firstSigma returns the cost scale for a retry without a solution:
// √(q/g) (SLICOT SB02MD SCAL='G'), which minimizes max(σg, q/σ) and, when
// q·g ≤ a², is the geometric middle of the σ range where the pencil is
// balanced; 1 when undefined.
func (norms riccatiNorms) firstSigma() float64 {
	if c := norms.candidates(0)[1]; c > 0 {
		return c
	}
	return 1
}

// riccatiPass is a saved solution of an earlier cost scale.
type riccatiPass struct {
	x, wr, wi, beta  []float64
	rcnd, sigma, res float64
	solves           int
}

// rescaleCost picks the cost scale σ for another solve from the solution
// xData of this one, saving it in *prev. The problem (Q/σ, R/σ, S/σ) has
// the solution X/σ and the same K. The Schur method's relative error in X
// is about eps·max(a, σg, q/σ)·(‖X‖/σ + σ/‖X‖)·kappa/sep, smallest near
// σ = ‖X‖ (SLICOT SB02MD SCAL='G' uses σ = √(q/g), right only when
// X ≈ √(q/g)), so X keeps its relative accuracy when ‖X‖ is far below (or
// above) the Hamiltonian scale. A zero first X is below accuracy and
// estimated by q/a. Later solves use the balanced pencil (kappa 1). It
// reports false, leaving σ, when the predicted gain is under 1024 or after
// three solves. The estimate can overrate the gain (an ill-conditioned R),
// so *prev keeps the saved solution of smallest Riccati residual, which
// keepBest weighs against the last.
func (problem *riccatiProblem) rescaleCost(continuous bool, norms riccatiNorms, xData []float64, prev **riccatiPass, rcnd float64) bool {
	if !problem.costScaling || (*prev != nil && (*prev).solves >= 2) {
		return false
	}
	n := problem.n
	x := impl.Dlange(lapack.Frobenius, n, n, xData, n, nil)
	if x == 0 && *prev == nil {
		x = norms.q / norms.a
	}
	if !(x > 0) || math.IsInf(x, 0) {
		return false
	}
	cost := func(sigma float64) float64 { return norms.cost(x, sigma) }
	cur := problem.sigma
	best, bestCost := cur, cost(cur)
	curCost := bestCost
	if !problem.balanced {
		curCost *= norms.kappa
	}
	for _, c := range norms.candidates(x) {
		if c > 0 {
			if v := cost(c); v < bestCost {
				best, bestCost = c, v
			}
		}
	}
	if !(curCost >= 1024*bestCost) {
		return false
	}
	problem.savePass(prev, xData, rcnd, problem.residual(continuous, xData))
	problem.sigma, problem.balanced = best, true
	return true
}

// savePass counts a solve in *prev and saves its solution xData there
// unless the saved one has a smaller Riccati residual than res.
func (problem *riccatiProblem) savePass(prev **riccatiPass, xData []float64, rcnd, res float64) {
	if *prev == nil {
		*prev = &riccatiPass{}
	}
	p, ws, n := *prev, problem.ws, problem.n
	p.solves++
	if p.solves > 1 && !(res < p.res) {
		return
	}
	p.res = res
	p.x = append(p.x[:0], xData...)
	p.wr = append(p.wr[:0], ws.wr[:n]...)
	p.wi = append(p.wi[:0], ws.wi[:n]...)
	p.beta = append(p.beta[:0], ws.beta[:n]...)
	p.rcnd, p.sigma = rcnd, problem.sigma
}

// keepBest reports whether the saved pass prev has a smaller Riccati
// residual than xData, the last solution.
func (problem *riccatiProblem) keepBest(continuous bool, prev *riccatiPass, xData []float64) bool {
	return prev != nil && !(problem.residual(continuous, xData) < prev.res)
}

// residual returns the Riccati residual of residualTerms.
func (problem *riccatiProblem) residual(continuous bool, xData []float64) float64 {
	res, _ := problem.residualTerms(continuous, xData)
	return res
}

// residualTerms returns ‖Ric(X)‖_F of the state-scaled problem
//
//	continuous: AᵀXE + EᵀXA − (EᵀXB + S)R⁻¹(BᵀXE + Sᵀ) + Q
//	discrete:   AᵀXA − EᵀXE − (AᵀXB + S)(R + BᵀXB)⁻¹(BᵀXA + Sᵀ) + Q
//
// and the sum of the Frobenius norms of its four terms.
func (problem *riccatiProblem) residualTerms(continuous bool, xData []float64) (res, scale float64) {
	n := problem.n
	X := mat.NewDense(n, n, xData)
	var E mat.Matrix = eyeDense(n)
	if problem.E != nil {
		E = problem.E
	}
	A, B := problem.A, problem.B
	var xe, xa, ric, t, m, w mat.Dense
	xe.Mul(X, E)
	xa.Mul(X, A)
	r := mat.DenseCopyOf(problem.R)
	if continuous {
		ric.Mul(A.T(), &xe)
		t.Mul(E.T(), &xa)
		scale = mat.Norm(&ric, 2) + mat.Norm(&t, 2)
		ric.Add(&ric, &t)
		m.Mul(xe.T(), B)
	} else {
		ric.Mul(A.T(), &xa)
		t.Mul(E.T(), &xe)
		scale = mat.Norm(&ric, 2) + mat.Norm(&t, 2)
		ric.Sub(&ric, &t)
		m.Mul(xa.T(), B)
		var bxb, xb mat.Dense
		xb.Mul(X, B)
		bxb.Mul(B.T(), &xb)
		r.Add(r, &bxb)
	}
	if problem.S != nil {
		m.Add(&m, problem.S)
	}
	if err := w.Solve(r, m.T()); err != nil {
		return math.Inf(1), math.Inf(1)
	}
	var mw mat.Dense
	mw.Mul(&m, &w)
	scale += mat.Norm(&mw, 2) + mat.Norm(problem.Q, 2)
	ric.Sub(&ric, &mw)
	ric.Add(&ric, problem.Q)
	return mat.Norm(&ric, 2), scale
}

// restorePass restores the solution saved by rescaleCost.
func (problem *riccatiProblem) restorePass(p *riccatiPass) (*mat.Dense, []float64, float64) {
	n, ws := problem.n, problem.ws
	xData := ws.xData[:n*n]
	copy(xData, p.x)
	copy(ws.wr[:n], p.wr)
	copy(ws.wi[:n], p.wi)
	copy(ws.beta[:n], p.beta)
	problem.sigma, problem.balanced = p.sigma, false
	return mat.NewDense(n, n, xData), xData, p.rcnd
}

// pow2Near returns the power of two nearest v within [2⁻⁵⁰⁰, 2⁵⁰⁰].
func pow2Near(v float64) float64 {
	frac, e := math.Frexp(v)
	if frac < math.Sqrt2/2 {
		e--
	}
	return math.Ldexp(1, min(max(e, -500), 500))
}

func finiteSlice(v []float64) bool {
	for _, x := range v {
		if !isFinite(x) {
			return false
		}
	}
	return true
}

// pencilNorms returns the riccatiNorms of the extended pencil, with
// ‖U⁻ᵀBᵀ‖_F² ≥ ‖G‖₂ for g (R = UᵀU in ws.rChol, U⁻ᵀBᵀ in ws.rinvBt), and
// with S forms Ā in ws.g and Q̄ in ws.qWork.
func (problem *riccatiProblem) pencilNorms() riccatiNorms {
	n, m, ws := problem.n, problem.m, problem.ws
	aRaw := problem.A.RawMatrix()
	norms := riccatiNorms{a: impl.Dlange(lapack.Frobenius, n, n, aRaw.Data, aRaw.Stride, nil)}
	if problem.E != nil {
		eRaw := problem.E.RawMatrix()
		norms.a = max(norms.a, impl.Dlange(lapack.Frobenius, n, n, eRaw.Data, eRaw.Stride, nil))
	} else {
		norms.a = max(norms.a, math.Sqrt(float64(n)))
	}
	rChol := ws.rChol[:m*m]
	upper := blas64.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit, N: m, Data: rChol, Stride: m}
	w := ws.rinvBt[:m*n]
	bRaw := problem.B.RawMatrix()
	for i := range n {
		for j := range m {
			w[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
		}
	}
	blas64.Trsm(blas.Left, blas.Trans, 1, upper, blas64.General{Rows: m, Cols: n, Data: w, Stride: n})
	gn := impl.Dlange(lapack.Frobenius, m, n, w, n, nil)
	norms.g = gn * gn
	rRaw := problem.R.RawMatrix()
	bn := impl.Dlange(lapack.Frobenius, n, m, bRaw.Data, bRaw.Stride, nil)
	rn := impl.Dlange(lapack.Frobenius, m, m, rRaw.Data, rRaw.Stride, nil)
	norms.kappa = max(bn, rn) / min(bn, rn)
	qRaw := problem.Q.RawMatrix()
	if !(norms.kappa >= 1) || math.IsInf(norms.kappa, 0) {
		norms.kappa = 1
	}
	if problem.S == nil {
		norms.q = impl.Dlange(lapack.Frobenius, n, n, qRaw.Data, qRaw.Stride, nil)
		return norms
	}
	v := ws.rinvSt[:m*n]
	sRaw := problem.S.RawMatrix()
	for i := range n {
		for j := range m {
			v[j*n+i] = sRaw.Data[i*sRaw.Stride+j]
		}
	}
	blas64.Trsm(blas.Left, blas.Trans, 1, upper, blas64.General{Rows: m, Cols: n, Data: v, Stride: n})
	qbar := ws.qWork[:n*n]
	copyStrided(qbar, n, qRaw.Data, qRaw.Stride, n, n)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		-1, blas64.General{Rows: m, Cols: n, Data: v, Stride: n},
		blas64.General{Rows: m, Cols: n, Data: v, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: qbar, Stride: n})
	norms.q = impl.Dlange(lapack.Frobenius, n, n, qbar, n, nil)
	blas64.Trsm(blas.Left, blas.NoTrans, 1, upper, blas64.General{Rows: m, Cols: n, Data: v, Stride: n})
	abar := ws.g[:n*n]
	copyStrided(abar, n, aRaw.Data, aRaw.Stride, n, n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		-1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: m, Cols: n, Data: v, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: abar, Stride: n})
	norms.a = max(norms.a, impl.Dlange(lapack.Frobenius, n, n, abar, n, nil))
	norms.kappa = max(norms.kappa, impl.Dlange(lapack.Frobenius, n, m, sRaw.Data, sRaw.Stride, nil)/min(bn, rn))
	return norms
}

// riccatiPencil is the data of the extended pencil of
// generalizedStableSubspace: A, Q and S (S.Data nil for none), with Q and S
// divided by σ = 1/isigma and the input row [S' (B') R] multiplied by mu.
// A non-nil bt (m×n) replaces B and R by B̃ = btᵀ and I, the inputs ũ = Uu
// of R = UᵀU, which leaves the deflation of R well-conditioned.
type riccatiPencil struct {
	a, q, s    blas64.General
	bt         []float64
	isigma, mu float64
}

// pencil returns the pencil data: the given A, Q, S at first, else for
// the cost scale σ Ā, Q̄ of pencilNorms and no S, so that S/σ does not
// swamp B in the input columns, the normalized inputs B̃ = BU⁻¹ of
// pencilNorms, and the power of two mu ≈ ‖B̃‖/‖I‖ balancing μI against B̃.
func (problem *riccatiProblem) pencil() riccatiPencil {
	d := riccatiPencil{a: problem.A.RawMatrix(), q: problem.Q.RawMatrix(), isigma: 1, mu: 1}
	if !problem.balanced {
		if problem.S != nil {
			d.s = problem.S.RawMatrix()
		}
		return d
	}
	n, ws := problem.n, problem.ws
	d.isigma = 1 / problem.sigma
	if problem.S != nil {
		d.a = blas64.General{Rows: n, Cols: n, Data: ws.g[:n*n], Stride: n}
		d.q = blas64.General{Rows: n, Cols: n, Data: ws.qWork[:n*n], Stride: n}
	}
	m := problem.m
	d.bt = ws.rinvBt[:m*n]
	if bn := impl.Dlange(lapack.Frobenius, m, n, d.bt, n, nil); bn > 0 {
		d.mu = pow2Near(bn / math.Sqrt(float64(m)))
	}
	return d
}

// pencilSolution deflates the extended pencil and returns its stabilizing
// solution, solving again with a cost scale when rescaleCost predicts a
// more accurate X; a failed later solve keeps the earlier one.
//
// The given pencil can fail QZ or misplace eigenvalues at extreme cost
// scales, so a failed first solve is rescued, also under NoScaling, by
// solves from the cost scale of firstSigma. The rescued X is kept only
// with a Riccati residual within rescueTol of its terms (and, see
// rescued, a stable closed loop), else the first error is returned. Under
// NoScaling a first X with a residual above noScalingTol of its terms is
// rescued too, keeping the solution of smallest residual.
func (problem *riccatiProblem) pencilSolution(continuous bool) (subspace riccatiSubspace, X *mat.Dense, xData []float64, rcnd float64, err error) {
	var norms riccatiNorms
	var prev *riccatiPass
	rescue := func() {
		problem.costScaling = true
		norms = problem.pencilNorms()
		problem.sigma, problem.balanced = norms.firstSigma(), true
	}
	for {
		subspace, err = problem.generalizedStableSubspace(continuous, problem.pencil())
		if err == nil {
			X, xData, rcnd, err = problem.stabilizingSolution(subspace.vectors)
		}
		if err == nil && !finiteSlice(xData) {
			err = ErrNoStabilizing
		}
		if err != nil && prev == nil {
			if problem.rescueErr != nil {
				return riccatiSubspace{}, nil, nil, 0, problem.rescueErr
			}
			problem.rescueErr = err
			rescue()
			continue
		}
		if err == nil && prev == nil && problem.rescueErr == nil {
			if !problem.costScaling {
				res, scale := problem.residualTerms(continuous, xData)
				if res <= noScalingTol*scale {
					return subspace, X, xData, rcnd, nil
				}
				problem.savePass(&prev, xData, rcnd, res)
				rescue()
				continue
			}
			norms = problem.pencilNorms()
		}
		if err == nil && problem.rescaleCost(continuous, norms, xData, &prev, rcnd) {
			continue
		}
		if err != nil || problem.keepBest(continuous, prev, xData) {
			subspace, X, xData, rcnd = problem.restoredSolution(prev)
		}
		if problem.rescueErr != nil {
			if res, scale := problem.residualTerms(continuous, xData); !(res <= rescueTol*scale) {
				return riccatiSubspace{}, nil, nil, 0, problem.rescueErr
			}
		}
		return subspace, X, xData, rcnd, nil
	}
}

// rescueTol and noScalingTol bound the Riccati residual of pencilSolution
// relative to the sum of its terms: a rescued solution above rescueTol is
// rejected, and an unscaled one above noScalingTol is rescued.
const (
	rescueTol    = 1e-6
	noScalingTol = 1e-12
)

// rescued returns problem.rescueErr when pencilSolution rescued a failed
// first solve and A − BK (with E: E⁻¹(A − BK)) is not stable, nil
// otherwise.
func (problem *riccatiProblem) rescued(continuous bool, K *mat.Dense) error {
	if problem.rescueErr == nil {
		return nil
	}
	var bk, ea mat.Dense
	bk.Mul(problem.B, K)
	acl := mat.NewDense(problem.n, problem.n, nil)
	acl.Sub(problem.A, &bk)
	if problem.E != nil {
		var lu mat.LU
		lu.Factorize(problem.E)
		if err := lu.SolveTo(&ea, false, acl); err != nil {
			if _, ok := err.(mat.Condition); !ok {
				return problem.rescueErr
			}
		}
		acl = &ea
	}
	var eig mat.Eigen
	if !eig.Factorize(acl, mat.EigenNone) {
		return problem.rescueErr
	}
	for _, v := range eig.Values(nil) {
		if continuous && !(real(v) < 0) || !continuous && !(cmplx.Abs(v) < 1) {
			return problem.rescueErr
		}
	}
	return nil
}

// restoredSolution restores the saved pass prev as pencilSolution returns
// it.
func (problem *riccatiProblem) restoredSolution(prev *riccatiPass) (riccatiSubspace, *mat.Dense, []float64, float64) {
	X, xData, rcnd := problem.restorePass(prev)
	ws, nn := problem.ws, 2*problem.n
	return riccatiSubspace{alphaR: ws.wr[:nn], alphaI: ws.wi[:nn], beta: ws.beta[:nn]}, X, xData, rcnd
}

// stabilizingSolution returns X = σ·U21*(E*U11)⁻¹ from the stable subspace
// basis [U11; U21] of a pencil with 2n columns and cost scale σ.
func (problem *riccatiProblem) stabilizingSolution(vs []float64) (X *mat.Dense, xData []float64, rcnd float64, err error) {
	n, ws := problem.n, problem.ws
	u11 := ws.u11[:n*n]
	u21 := ws.u21[:n*n]
	copyStrided(u11, n, vs, 2*n, n, n)
	copyBlock(u21, n, 0, 0, vs, 2*n, n, 0, n, n)
	if problem.E != nil {
		eRaw := problem.E.RawMatrix()
		eu11 := ws.aWork[:n*n]
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			1, blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: u11, Stride: n},
			0, blas64.General{Rows: n, Cols: n, Data: eu11, Stride: n})
		u11 = eu11
	}

	work := ws.work
	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, u11, n, work[:n])
	ipiv := ws.ipiv[:n]
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, nil, 0, ErrNoStabilizing
	}
	rcnd = impl.Dgecon(lapack.MaxColumnSum, n, u11, n, anorm, work[:4*n], ws.iwork[:n])

	xData = ws.xData[:n*n]
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)
	if problem.sigma != 1 {
		for i := range xData {
			xData[i] *= problem.sigma
		}
	}
	symmetrize(xData, n, n)
	return mat.NewDense(n, n, xData), xData, rcnd, nil
}

// descriptorCare solves the generalized continuous Riccati equation; R is
// already Cholesky-factored in ws.rChol.
func (problem *riccatiProblem) descriptorCare() (*RiccatiResult, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	subspace, X, xData, rcnd, err := problem.pencilSolution(true)
	if err != nil {
		return nil, err
	}
	eig := ws.eig[:n]
	for i := range n {
		eig[i] = complex(subspace.alphaR[i]/subspace.beta[i], subspace.alphaI[i]/subspace.beta[i])
	}

	bRaw := problem.B.RawMatrix()
	eRaw := problem.E.RawMatrix()
	btx := ws.btx[:m*n]
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n})
	kData := ws.kData[:m*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
		0, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n})
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		for j := range m {
			row := kData[j*n:]
			for i := range n {
				row[i] += sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	impl.Dpotrs(blas.Upper, m, n, ws.rChol[:m*m], m, kData, n)
	K := mat.NewDense(m, n, kData)
	if err := problem.rescued(true, K); err != nil {
		return nil, err
	}
	return problem.result(X, K, eig, rcnd), nil
}

type riccatiSubspace struct {
	vectors []float64
	alphaR  []float64
	alphaI  []float64
	beta    []float64
}

// generalizedStableSubspace deflates the extended pencil
//
//	continuous: [A 0 B; -Q -A' -S; S' B' R] - λ[E 0 0; 0 E' 0; 0 0 0]
//	discrete:   [A 0 B; -Q E' -S; S' 0 R]   - λ[E 0 0; 0 A' 0; 0 -B' 0]
//
// to 2n×2n by a QR of its input columns and orders the stable eigenvalues
// first. E nil means I. With the cost scale of d it deflates the
// equivalent pencil diag(I, I/σ, μ)·(H − λJ)·diag(I, σI, I), and with d.bt
// that of the inputs ũ = Uu, diag(I, I, U⁻ᵀ)·(H − λJ)·diag(I, I, U⁻¹).
func (problem *riccatiProblem) generalizedStableSubspace(continuous bool, d riccatiPencil) (riccatiSubspace, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	nn := 2 * n
	rows := nn + m
	isigma, mu := d.isigma, d.mu
	muSigma := mu / isigma

	hLeft := ws.pencilH[:rows*nn]
	jLeft := ws.pencilJ[:rows*nn]
	input := ws.pencilInput[:rows*m]
	clear(hLeft)
	clear(jLeft)
	clear(input)

	aRaw, qRaw, sRaw := d.a, d.q, d.s
	bRaw := problem.B.RawMatrix()
	rRaw := problem.R.RawMatrix()
	hasS := sRaw.Data != nil
	b := func(i, j int) float64 { return bRaw.Data[i*bRaw.Stride+j] }
	if d.bt != nil {
		b = func(i, j int) float64 { return d.bt[j*n+i] }
	}

	var eRaw blas64.General
	if problem.E != nil {
		eRaw = problem.E.RawMatrix()
	}
	e := func(i, j int) float64 {
		if problem.E != nil {
			return eRaw.Data[i*eRaw.Stride+j]
		}
		if i == j {
			return 1
		}
		return 0
	}
	for i := range n {
		for j := range n {
			hLeft[i*nn+j] = aRaw.Data[i*aRaw.Stride+j]
			hLeft[(n+i)*nn+j] = -isigma * qRaw.Data[i*qRaw.Stride+j]
			jLeft[i*nn+j] = e(i, j)
			if continuous {
				hLeft[(n+i)*nn+n+j] = -aRaw.Data[j*aRaw.Stride+i]
				jLeft[(n+i)*nn+n+j] = e(j, i)
			} else {
				hLeft[(n+i)*nn+n+j] = e(j, i)
				jLeft[(n+i)*nn+n+j] = aRaw.Data[j*aRaw.Stride+i]
			}
		}

		for j := range m {
			bij := b(i, j)
			input[i*m+j] = bij
			if hasS {
				input[(n+i)*m+j] = -isigma * sRaw.Data[i*sRaw.Stride+j]
				hLeft[(nn+j)*nn+i] = mu * sRaw.Data[i*sRaw.Stride+j]
			}
			if continuous {
				hLeft[(nn+j)*nn+n+i] = muSigma * bij
			} else {
				jLeft[(nn+j)*nn+n+i] = -muSigma * bij
			}
		}
	}
	for i := range m {
		if d.bt != nil {
			input[(nn+i)*m+i] = mu
			continue
		}
		for j := range m {
			input[(nn+i)*m+j] = mu * rRaw.Data[i*rRaw.Stride+j]
		}
	}

	if m != 0 {
		tau := ws.tau[:m]
		var qrQuery, applyQuery [1]float64
		impl.Dgeqrf(rows, m, nil, m, nil, qrQuery[:], -1)
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, m, nil, m, nil, nil, nn, applyQuery[:], -1)
		lwork := max(int(qrQuery[0]), int(applyQuery[0]))
		if len(ws.work) < lwork {
			ws.work = make([]float64, lwork)
		}
		impl.Dgeqrf(rows, m, input, m, tau, ws.work, lwork)
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, m, input, m, tau, hLeft, nn, ws.work, lwork)
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, m, input, m, tau, jLeft, nn, ws.work, lwork)
	}

	h := ws.h[:nn*nn]
	j := ws.z[:nn*nn]
	copy(h, hLeft[m*nn:])
	copy(j, jLeft[m*nn:])

	alphaR := ws.wr[:nn]
	alphaI := ws.wi[:nn]
	beta := ws.beta[:nn]
	vectors := ws.vs[:nn*nn]
	bwork := ws.bwork[:nn]
	stable := func(alphaR, alphaI, beta float64) bool {
		return math.Hypot(alphaR, alphaI) < math.Abs(beta)
	}
	if continuous {
		stable = func(alphaR, _, beta float64) bool {
			return alphaR*beta < 0
		}
	}

	var workQuery [1]float64
	impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, stable,
		nn, h, nn, j, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, workQuery[:], -1, bwork)
	lwork := int(workQuery[0])
	if len(ws.work) < lwork {
		ws.work = make([]float64, lwork)
	}

	sdim, ok := impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, stable,
		nn, h, nn, j, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, ws.work, lwork, bwork)
	if !ok {
		return riccatiSubspace{}, ErrSchurFailed
	}
	if sdim != n {
		return riccatiSubspace{}, ErrNoStabilizing
	}
	for i := range n {
		if beta[i] == 0 {
			return riccatiSubspace{}, ErrNoStabilizing
		}
	}

	return riccatiSubspace{
		vectors: vectors,
		alphaR:  alphaR,
		alphaI:  alphaI,
		beta:    beta,
	}, nil
}
