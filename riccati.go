package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

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
	costV       []float64
	gainRes     []float64
	eLU         []float64
	eIpiv       []int
	eRes        []float64
	iwork       []int
	scaleT      []float64
	scaledA     []float64
	scaledE     []float64
	scaledQ     []float64
	scaledB     []float64
	scaledS     []float64
	scaleProxy  []float64
	lyap        *LyapunovWorkspace
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
		costV:       make([]float64, m*n),
		gainRes:     make([]float64, 2*m*n),
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
	// of its terms, is still retried with cost scaling; so is a plain Care
	// solve that fails or leaves a residual above 1e-6 of its terms. A
	// solve in either mode that finds no stabilizing solution, or one no
	// Newton step certifies (Accuracy 1), is retried once in the other,
	// and Care's then state-scaled ignoring R.
	NoScaling bool
	// plainScaling balances Care's state scaling with B in place of its
	// cost proxy (scale): otherScaling's last resort.
	plainScaling bool
	// reducedQ marks Q as Dare's Q − S₂R₂₂⁻¹S₂ᵀ of reduceInputs, which
	// may be indefinite as Q − SR⁻¹Sᵀ of the given Dare may.
	reducedQ bool
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
	// cost-scaled problem actually solved or, for Dare's pole-placement
	// fallback, of the orthonormal basis of span[I; X̂/σ], σ ≈ ‖X̂‖.
	Rcnd float64
	// Accuracy is the Report of MATLAB icare/idare for X: 0 when a Newton
	// step certifies X (to about 1e-8 relative), 1 when X is a stabilizing
	// Hamiltonian or pencil solution (A − BK stable in float64 arithmetic)
	// that no Newton step certifies, so X may be inaccurate and, near the
	// stability boundary, A − BK only rounded to stable; check it before
	// use. A non-finite X or no stabilizing solution (Report 2, 3) is an
	// error, not a result, as is an X with an unstabilizable mode of (A, B)
	// on the boundary to within rounding (PBH).
	Accuracy int

	step float64
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
// diagonal scaling, so X, K and Eig do not depend on the units of x; the
// balance is that of the Hamiltonian, with BR⁻¹Bᵀ weighed as Q. When
// ‖X‖ is far from the scale of the Hamiltonian (for example a cost Q, R
// tiny against A), Care solves again with Q, R, S divided by a power of
// two near ‖X‖, which keeps the relative accuracy of X. opts.NoScaling
// disables both. A failed first solve (Schur failure, no stabilizing
// subspace) is retried, also under NoScaling, at the cost scale
// √(‖Q̄‖/‖G‖); the retry's X is kept only with a Riccati residual within
// 1e-6 of its terms and a stable A − BK, else the first error is returned.
// A solution with a residual above 1e-6 of its terms is likewise re-solved
// at that cost scale, and the more accurate of the two is kept.
//
// Care checks X by a Newton–Kleinman step from its gain (one Lyapunov
// solve of the double-double Riccati residual; with E, for E⁻¹(A − BK)
// and EᵀΔE), as Dare does
// with Hewer's step: the Hamiltonian forms BR⁻¹Bᵀ, which under cheap
// control (R ≪ BᵀXB) swamps A and loses digits the Riccati residual does
// not show. An X the step moves by more than 1e-8 relative is refined by
// Newton's method, and an iterate is kept only once Newton has converged
// (a step within 1e-10); when that fails (or A − BK is unstable, or the
// Hamiltonian solve fails, or E is given) Care solves the extended pencil
// of MATLAB icare, verified the same way at each of its cost scales. A
// certified X has Accuracy 0. With none certified, Care returns, as
// MATLAB icare with Report 1, the stabilizing solution (A − BK stable in
// float64) of smallest first Newton step (or, without one, a stabilizing
// X whose Lyapunov equation is singular) with Accuracy 1, unrefined:
// under cheap control with weak actuation Newton has no quadratic region
// (its first step from an X accurate to 1e-11 can be 1e-6), and such an
// X may be accurate or off by far more than its step; when (A, B) has a
// mode on the boundary (to its eigenvalue's rounding error) that fails
// the PBH rank test, no stabilizing solution exists and Care returns
// ErrNoStabilizing instead, in either mode. An X whose U11 has
// rcnd below eps is not finite (icare Report 2) and is never returned.
// A solve that finds no stabilizing solution, or only an Accuracy 1 one,
// is retried with NoScaling flipped and then state-scaled balancing B in
// place of BR⁻¹Bᵀ: the first certified X is returned, else the Accuracy
// 1 X of smallest first Newton step; with none Care returns
// ErrNoStabilizing (or the Hamiltonian's error) of the first.
func Care(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	if err := riccatiArgs("Care", A, B, Q, R, opts); err != nil {
		return nil, err
	}
	res, err := otherScaling(care, true, A, B, Q, R, opts)
	if err != nil {
		return nil, fmt.Errorf("Care: %w", err)
	}
	return res, nil
}

// otherScaling solves by solve and, when that finds no stabilizing
// solution (ErrNoStabilizing or ErrSchurFailed) or one no Newton step
// certifies (Accuracy 1), once more with opts.NoScaling flipped, and then,
// when continuous (Care), state-scaled with B in place of its cost proxy
// (opts.plainScaling).
// State scaling can push a cheap-control pencil (R ≪ BᵀXB) to where no
// cost scale deflates it while the unscaled one solves (w17 #8), and the
// reverse; each solution passes the same Newton verification. The first
// certified result is kept; else of Accuracy 1 results the one of
// smallest first Newton step (the first on a tie), and with all failing
// the first error. An unstabilizable boundary mode
// (errBoundaryUnstabilizable) in any mode overrides every Accuracy 1
// result but not a certified one: its rounding band is a heuristic that
// a non-normal block can inflate past a mode inside the boundary (seen
// unscaled only), while a Newton-certified X is stable. A certified X
// with a closed-loop mode within rounding of the boundary
// (boundaryCertified) is a boundary verdict when both the unscaled and
// the state-scaled problem have an unstabilizable boundary mode: an
// exactly uncontrollable Jordan block at 1 left the closed loop at
// 1 − 3e-16 ± 1e-8i with X ≈ 1e49 that Newton could not move (MPWOJM).
// A kept Accuracy 1 result lives in opts.Workspace, so later solves run
// without it.
func otherScaling(solve func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error), continuous bool, A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	res, err := solve(A, B, Q, R, opts)
	if err == nil && res.Accuracy == 0 && boundaryCertified(continuous, A, B, Q, R, opts, res) {
		res, err = nil, errBoundaryUnstabilizable
	}
	if err == nil && res.Accuracy == 0 ||
		err != nil && !errors.Is(err, ErrNoStabilizing) && !errors.Is(err, ErrSchurFailed) {
		return res, err
	}
	var boundary error
	if errors.Is(err, errBoundaryUnstabilizable) {
		boundary = err
	}
	var base RiccatiOpts
	if opts != nil {
		base = *opts
	}
	flipped, plain := base, base
	flipped.NoScaling = !base.NoScaling
	plain.NoScaling, plain.plainScaling = false, true
	retries := []RiccatiOpts{flipped, plain}
	if !continuous {
		retries = retries[:1]
	}
	for _, o := range retries {
		if res != nil {
			o.Workspace = nil
		}
		r, e := solve(A, B, Q, R, &o)
		if e == nil && r.Accuracy == 0 && boundaryCertified(continuous, A, B, Q, R, opts, r) {
			r, e = nil, errBoundaryUnstabilizable
		}
		switch {
		case errors.Is(e, errBoundaryUnstabilizable):
			if boundary == nil {
				boundary = e
			}
		case e != nil:
		case r.Accuracy == 0:
			return r, nil
		case res == nil || r.step < res.step:
			res = r
		}
	}
	if boundary != nil {
		return nil, boundary
	}
	if res != nil {
		return res, nil
	}
	return nil, err
}

// boundaryCertified reports a closed-loop eigenvalue of the certified
// res within the rounding of a well-conditioned eigenvalue, n·eps·‖A‖_F,
// of the stability boundary, where the certificate shows no stability,
// with boundaryUnstabilizable true both unscaled and state-scaled as
// solve scales (Care with its cost proxy). Wider bands (√eps, 10·n·eps·‖A‖_F)
// rejected oracle-exact X of stable uncontrollable modes at 1 − 1e-9 and
// 1 − 1e-11 that boundaryUnstabilizable's κ-weighted band covers.
func boundaryCertified(continuous bool, A, B, Q, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult) bool {
	if len(res.Eig) == 0 {
		return false
	}
	n, _ := A.Dims()
	band := float64(n) * eps() * denseNorm(A)
	near := false
	for _, l := range res.Eig {
		if continuous && -real(l) <= band || !continuous && 1-cmplx.Abs(l) <= band {
			near = true
			break
		}
	}
	if !near {
		return false
	}
	var base RiccatiOpts
	if opts != nil {
		base = *opts
	}
	base.Workspace, base.NoScaling, base.plainScaling = nil, true, false
	problem, err := newRiccatiProblem(A, B, Q, R, &base)
	if err != nil || !problem.boundaryUnstabilizable(continuous) {
		return false
	}
	base.NoScaling = false
	problem.scale(&base, continuous)
	return problem.boundaryUnstabilizable(continuous)
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
	problem.scale(opts, opts == nil || !opts.plainScaling)
	A, B, Q, S, ws := problem.A, problem.B, problem.Q, problem.S, problem.ws

	// Cholesky factor R
	rChol := ws.rChol[:m*m]
	rRaw := R.RawMatrix()
	copyStrided(rChol, m, rRaw.Data, rRaw.Stride, m, m)
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
	}
	if problem.E != nil {
		X, K, rcnd, accuracy, err := problem.pencilRiccati(true)
		if err != nil {
			return nil, err
		}
		return problem.result(X, K, ws.eig[:n], rcnd, accuracy), nil
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

	eig := ws.eig[:n]
	costScaling := problem.costScaling
	X, rcnd, err := problem.hamiltonianSolution(aWork, qWork, g, eig)
	var K *mat.Dense
	if err == nil {
		K, err = problem.acceptRiccati(true, X, eig, rcnd)
	}
	accuracy := 0
	if err != nil {
		first := err
		problem.costScaling, problem.balanced, problem.sigma = costScaling, false, 1
		problem.rescueErr, problem.resolved, problem.candidates = nil, false, problem.candidates[:0]
		if X, K, rcnd, accuracy, err = problem.pencilRiccati(true); err != nil {
			if errors.Is(err, errBoundaryUnstabilizable) {
				return nil, err
			}
			return nil, first
		}
	}
	return problem.result(X, K, eig, rcnd, accuracy), nil
}

// hamiltonianSolution returns the stabilizing solution X of care's
// Hamiltonian [[Ā, −G], [−Q̄, −Āᵀ]] (n×n blocks a, q, g) and rcnd, with the
// closed-loop eigenvalues in eig, cost-scaled and re-solved as Care
// describes.
func (problem *riccatiProblem) hamiltonianSolution(aWork, qWork, g []float64, eig []complex128) (*mat.Dense, float64, error) {
	n, ws := problem.n, problem.ws
	nn := 2 * n
	wr := ws.wr[:nn]
	wi := ws.wi[:nn]
	vs := ws.vs[:nn*nn]
	bwork := ws.bwork[:nn]
	selctg := func(wr, wi float64) bool { return wr < 0 }
	var norms riccatiNorms
	var prev, orig *riccatiPass
	var X *mat.Dense
	var xData []float64
	var rcnd float64
solve:
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
		if err == nil && !finiteSlice(xData) {
			err = ErrNoStabilizing
		}
		if err != nil {
			switch {
			case prev != nil:
				X, xData, rcnd = problem.restorePass(prev)
			case orig != nil:
				X, xData, rcnd = problem.restorePass(orig)
				break solve
			case problem.rescueErr != nil:
				return nil, 0, problem.rescueErr
			default:
				problem.rescueErr, problem.costScaling, problem.resolved = err, true, true
				norms = hamiltonianNorms(n, aWork, qWork, g)
				problem.sigma, problem.balanced = norms.firstSigma(), true
				continue
			}
		} else {
			if prev == nil && problem.costScaling && problem.rescueErr == nil && orig == nil {
				norms = hamiltonianNorms(n, aWork, qWork, g)
			}
			if problem.rescaleCost(true, norms, xData, &prev, rcnd) {
				problem.resolved = true
				continue
			}
			if problem.keepBest(true, prev, xData) {
				X, xData, rcnd = problem.restorePass(prev)
			}
		}
		res, scale := hamiltonianResidual(n, aWork, qWork, g, xData, ws.h[:nn*nn])
		if problem.rescueErr != nil {
			if !(res <= rescueTol*scale) {
				return nil, 0, problem.rescueErr
			}
			break
		}
		if orig != nil {
			if !(res <= rescueTol*scale && res < orig.res) {
				X, xData, rcnd = problem.restorePass(orig)
			}
			break
		}
		if res <= rescueTol*scale {
			break
		}
		problem.savePass(&orig, xData, rcnd, res)
		prev, problem.costScaling, problem.resolved = nil, true, true
		norms = hamiltonianNorms(n, aWork, qWork, g)
		problem.sigma, problem.balanced = norms.firstSigma(), true
	}

	for i := range n {
		eig[i] = complex(wr[i], wi[i])
	}
	return X, rcnd, nil
}

// hamiltonianResidual returns ‖ĀᵀX + XĀ − XGX + Q̄‖_F of care's
// Hamiltonian blocks and the sum of the Frobenius norms of its terms, using
// work (≥ 3n²) as scratch.
func hamiltonianResidual(n int, a, q, g, x, work []float64) (res, scale float64) {
	gen := func(d []float64) blas64.General { return blas64.General{Rows: n, Cols: n, Data: d, Stride: n} }
	xa, gx, xgx := work[:n*n], work[n*n:2*n*n], work[2*n*n:3*n*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(x), gen(a), 0, gen(xa))
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(g), gen(x), 0, gen(gx))
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(x), gen(gx), 0, gen(xgx))
	scale = 2*impl.Dlange(lapack.Frobenius, n, n, xa, n, nil) +
		impl.Dlange(lapack.Frobenius, n, n, xgx, n, nil) +
		impl.Dlange(lapack.Frobenius, n, n, q, n, nil)
	for i := range n {
		for j := range n {
			gx[i*n+j] = xa[i*n+j] + xa[j*n+i] - xgx[i*n+j] + q[i*n+j]
		}
	}
	return impl.Dlange(lapack.Frobenius, n, n, gx, n, nil), scale
}

// hamiltonianNorms returns the riccatiNorms of care's Hamiltonian blocks
// Ā, Q̄ and G (n×n, row-major).
func hamiltonianNorms(n int, a, q, g []float64) riccatiNorms {
	return riccatiNorms{
		a:     impl.Dlange(lapack.Frobenius, n, n, a, n, nil),
		q:     impl.Dlange(lapack.Frobenius, n, n, q, n, nil),
		g:     impl.Dlange(lapack.Frobenius, n, n, g, n, nil),
		kappa: 1,
	}
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
// Dare checks the pencil's X by a Newton (Hewer) step from its gain, a
// Stein solve (with E, for E⁻¹(A − BK) and EᵀΔE) of the double-double
// Riccati residual that is blind
// to the rounding of (R+B'XB)⁻¹ that hides lost digits from the residual
// itself. An X the step moves by more than 1e-8 relative is refined by
// Newton's method until a step is within 1e-10; if that does not
// converge, the other cost scales' solutions are tried. A certified X has
// Accuracy 0; with none certified Dare returns the stabilizing solution of
// smallest first step with Accuracy 1 (MATLAB idare Report 1, as Care,
// including its PBH check of boundary modes, and its retry in the other
// NoScaling mode). With no stabilizing pencil solution in either mode and
// no unstabilizable boundary mode, Dare runs Hewer's
// iteration from a pole-placement gain and returns its X only when the
// same Newton check certifies it (Accuracy 0; Rcnd then that of span[I; X/σ] at σ ≈ ‖X‖),
// else ErrNoStabilizing: under cheap control of a strongly unstable plant
// no cost scale of the pencil may resolve its split. It runs them too
// over an Accuracy 1 pencil solution, which a certified placed X
// replaces: under cheap control with m = n the pencil's X can be off by
// 4e-3 where Hewer from a placed gain certifies it to 1e-10. Not with B
// of condition above rotationCond in the given inputs, where Newton's
// R + BᵀXB has lost its smallest eigenvalue and certified placed X off by
// 2e-6 over pencil X accurate to 1e-7.
//
// Dare scales the states and the cost as Care does (MATLAB idare), its
// state scaling balanced with B (the cost proxy of Care brought no
// certification on cheap-control sweeps and cost 10-20% time);
// opts.NoScaling disables it. An unscaled pencil that fails QZ or the
// stable split is solved again cost-scaled before an error is returned.
//
// With m ≥ 2 and B of condition above rotationCond, Dare solves in the
// inputs w = Vᵀu of B's right singular vectors V (reduceInputs): X, Eig
// and Rcnd are unchanged, and R + (BV)ᵀX(BV) keeps on its diagonal the
// smallest eigenvalue that R + BᵀXB loses to rounding under cheap
// control, where Newton certified X off by 5e-7 (or the pencil's X by
// 1e-5) on well-conditioned problems with nearly collinear inputs. With
// a cross term S and rank(B) = r < m (always when m > n), Dare instead
// eliminates the inputs B does not see, as their optimal value
// w₂ = −R₂₂⁻¹(R₂₁w₁ + S₂ᵀx) is explicit, and solves the problem of r
// inputs: R + BᵀXB loses their R₂₂, and with it a cost xᵀS₂R₂₂⁻¹S₂ᵀx
// comparable to Q, so Newton certified a wrong X ≈ Q. Its X is returned
// once the same Newton check certifies it in the m inputs w, with K
// and Eig from there. Either K is returned only with A − BK in the given
// inputs stable; should that, or the problem in w, fail, the given one
// is solved.
func Dare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	if err := riccatiArgs("Dare", A, B, Q, R, opts); err != nil {
		return nil, err
	}
	red, ill := reduceInputs(A, B, Q, R, opts)
	if red != nil {
		var o RiccatiOpts
		if opts != nil {
			o = *opts
		}
		o.S, o.reducedQ = red.s, !red.rotated
		if res, err := dareSolve(A, red.b, red.q, red.r, &o, true); err == nil {
			ok := true
			if red.rotated {
				res.K = red.gain(res.K, o.Workspace)
			} else {
				res, ok = red.certify(A, Q, opts, res)
			}
			if ok && closedLoopStable(A, B, res.K, o.E, res.Eig) {
				return res, nil
			}
		}
	}
	return dareSolve(A, B, Q, R, opts, !ill)
}

// closedLoopStable reports whether the eigenvalues of A − BK (with E, of
// E⁻¹(A − BK)), A − BK formed in double-double, lie inside the unit
// circle, and writes them to eig when they do. Dare checks the K of the
// inputs u it returns: under cheap control ‖K‖ ≈ 1e13 rounds by 1e-3 in
// every direction of u, which BV₁ = O(1) can carry to an unstable A − BK
// although that of the inputs w is stable.
func closedLoopStable(A, B, K, E *mat.Dense, eig []complex128) bool {
	n, m := B.Dims()
	aRaw, bRaw, kRaw := A.RawMatrix(), B.RawMatrix(), K.RawMatrix()
	hi, lo := make([]float64, n*n), make([]float64, n*n)
	copyStrided(hi, n, aRaw.Data, aRaw.Stride, n, n)
	ddMulAcc(n, n, m, -1, bRaw.Data, bRaw.Stride, 1, kRaw.Data, kRaw.Stride, 1, hi, lo, n)
	for i := range hi {
		hi[i] += lo[i]
	}
	acl := mat.NewDense(n, n, hi)
	if E != nil {
		var lu mat.LU
		lu.Factorize(E)
		var f mat.Dense
		if lu.SolveTo(&f, false, acl) != nil {
			return false
		}
		acl = &f
	}
	var ed mat.Eigen
	if !ed.Factorize(acl, mat.EigenNone) {
		return false
	}
	vals := ed.Values(nil)
	for _, v := range vals {
		if !(cmplx.Abs(v) < 1) {
			return false
		}
	}
	copy(eig, vals)
	return true
}

// certify returns the X of res, the solution of the reduced Dare, once
// verifyRiccati certifies it as that of the given one in the inputs w,
// whose R + (BV)ᵀX(BV) keeps R₂₂ on its diagonal, with the gain and
// Eig of A − BK there: BV₂ is zero only to rounding, and under cheap
// control the K₂ ≈ R₂₂⁻¹S₂ᵀ it multiplies can move X, or destabilize
// A − BK, well beyond the rounding of B.
func (red *reducedInputs) certify(A, Q *mat.Dense, opts *RiccatiOpts, res *RiccatiResult) (*RiccatiResult, bool) {
	var o RiccatiOpts
	if opts != nil {
		o = *opts
	}
	o.S, o.Workspace = red.sw, nil
	problem, err := newRiccatiProblem(A, red.bw, Q, red.rw, &o)
	if err != nil {
		return nil, false
	}
	problem.scale(&o, false)
	n := problem.n
	X := mat.NewDense(n, n, problem.ws.xData[:n*n])
	xr := res.X.RawMatrix()
	for i := range n {
		for j := range n {
			v := xr.Data[i*xr.Stride+j]
			if t := problem.t; t != nil {
				v /= t[i] * t[j]
			}
			X.Set(i, j, v)
		}
	}
	eig := problem.ws.eig[:n]
	K, err := problem.acceptRiccati(false, X, eig, res.Rcnd)
	if err != nil {
		return nil, false
	}
	out := problem.result(X, K, eig, res.Rcnd, 0)
	out.K = red.gain(out.K, nil)
	return out, true
}

// dareSolve is Dare without its input reduction; swapPoor lets placedDare
// replace an Accuracy 1 pencil X, which Dare declines on a B of condition
// above rotationCond in the inputs u, where Newton's R + BᵀXB has lost
// its smallest eigenvalue and certifies a wrong X.
func dareSolve(A, B, Q, R *mat.Dense, opts *RiccatiOpts, swapPoor bool) (*RiccatiResult, error) {
	res, err := otherScaling(dare, false, A, B, Q, R, opts)
	poor := swapPoor && err == nil && res.Accuracy != 0
	if poor || (errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrSchurFailed)) && !errors.Is(err, errBoundaryUnstabilizable) {
		var base RiccatiOpts
		if opts != nil {
			base = *opts
		}
		if poor {
			base.Workspace = nil
		}
		flipped := base
		flipped.NoScaling = !base.NoScaling
		for _, o := range []RiccatiOpts{base, flipped} {
			if r, e := placedDare(A, B, Q, R, &o); e == nil && !boundaryCertified(false, A, B, Q, R, opts, r) {
				return r, nil
			}
		}
	}
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
	problem.scale(opts, false)
	ws := problem.ws

	// Cholesky factor R
	rChol := ws.rChol[:m*m]
	rRaw := R.RawMatrix()
	copyStrided(rChol, m, rRaw.Data, rRaw.Stride, m, m)
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
	}

	X, K, rcnd, accuracy, err := problem.pencilRiccati(false)
	if err != nil {
		return nil, err
	}
	return problem.result(X, K, ws.eig[:n], rcnd, accuracy), nil
}

// reducedInputs is the Dare of reduceInputs in the inputs w = Vᵀu, V = [V₁ V₂]
// the right singular vectors of B with BV₂ = 0 to rounding: B₁ = BV₁ (b),
// Q − S₂R₂₂⁻¹S₂ᵀ (q), R₁₁ − R₁₂R₂₂⁻¹R₂₁ (r) and S₁ − S₂R₂₂⁻¹R₂₁ (s) for
// VᵀRV = [R₁₁ R₁₂; R₂₁ R₂₂] and SV = [S₁ S₂]. With V₂ empty (rotated) it
// is the given Dare in the inputs w: BV, Q, VᵀRV and SV.
type reducedInputs struct {
	b, q, r, s *mat.Dense
	// bw, rw and sw are BV, VᵀRV and SV (nil without S).
	bw, rw, sw *mat.Dense
	v          *mat.Dense
	rotated    bool
}

// rotationCond is the condition of B above which Dare solves in the
// inputs of B's right singular vectors: R + BᵀXB then loses its smallest
// eigenvalue, ≈ R₂₂ + σ²_min·xᵀXx, to the rounding eps·σ²_max·‖X‖ of its
// entries, a relative error of up to eps·cond(B)², while (BV)ᵀX(BV) keeps
// it on its diagonal.
const rotationCond = 1e3

// reduceInputs returns the reducedInputs of a Dare with a cross term S and
// rank(B) < m, rank as placedHewer counts it, or, with rank(B) = m ≥ 2 and
// cond(B) above rotationCond, the rotated one; nil when there are none
// or R₂₂ is not positive definite. ill reports m ≥ 2 and cond(B) above
// rotationCond.
func reduceInputs(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (red *reducedInputs, ill bool) {
	n, m := B.Dims()
	if m < 2 || n == 0 {
		return nil, ill
	}
	var S *mat.Dense
	if opts != nil {
		S = opts.S
		if opts.Workspace != nil && opts.Workspace.fits(n, m) != nil {
			return nil, ill
		}
	}
	if an, _ := A.Dims(); an != n {
		return nil, ill
	}
	if S != nil {
		if sr, sc := S.Dims(); sr != n || sc != m {
			return nil, ill
		}
	}
	if rr, rc := R.Dims(); rr != m || rc != m {
		return nil, ill
	}
	if qr, qc := Q.Dims(); qr != n || qc != n {
		return nil, ill
	}
	var svd mat.SVD
	if !svd.Factorize(B, mat.SVDFullV) {
		return nil, ill
	}
	sv := svd.Values(nil)
	r := 0
	for _, s := range sv {
		if s > float64(max(n, m))*eps()*sv[0] {
			r++
		}
	}
	ill = len(sv) < m || !(sv[m-1] >= sv[0]/rotationCond)
	if r == 0 || !ill || r < m && S == nil {
		return nil, ill
	}
	red = &reducedInputs{v: &mat.Dense{}}
	svd.VTo(red.v)
	var rv, rt mat.Dense
	rv.Mul(R, red.v)
	rt.Mul(red.v.T(), &rv)
	symmetrize(rt.RawMatrix().Data, m, rt.RawMatrix().Stride)
	red.bw, red.rw = &mat.Dense{}, &rt
	red.bw.Mul(B, red.v)
	var sv2 mat.Dense
	if S != nil {
		sv2.Mul(S, red.v)
		red.sw = &sv2
	}
	if r == m {
		red.b, red.q, red.r, red.s, red.rotated = red.bw, Q, red.rw, red.sw, true
		return red, ill
	}
	red.b = mat.DenseCopyOf(red.bw.Slice(0, n, 0, r))
	r22 := mat.NewSymDense(m-r, nil)
	for i := range m - r {
		for j := i; j < m-r; j++ {
			r22.SetSym(i, j, rt.At(r+i, r+j))
		}
	}
	var r22c mat.Cholesky
	if !r22c.Factorize(r22) {
		return nil, ill
	}
	s2 := sv2.Slice(0, n, r, m)
	var r21, s2t mat.Dense
	if r22c.SolveTo(&r21, rt.Slice(r, m, 0, r)) != nil || r22c.SolveTo(&s2t, s2.T()) != nil {
		return nil, ill
	}
	var t mat.Dense
	t.Mul(rt.Slice(0, r, r, m), &r21)
	red.r = &mat.Dense{}
	red.r.Sub(rt.Slice(0, r, 0, r), &t)
	symmetrize(red.r.RawMatrix().Data, r, r)
	t.Reset()
	t.Mul(s2, &r21)
	red.s = &mat.Dense{}
	red.s.Sub(sv2.Slice(0, n, 0, r), &t)
	t.Reset()
	t.Mul(s2, &s2t)
	red.q = &mat.Dense{}
	red.q.Sub(Q, &t)
	symmetrize(red.q.RawMatrix().Data, n, n)
	return red, ill
}

// gain returns the gain VK of the inputs u from the gain K of the inputs
// w, in ws.kData when ws is given.
func (red *reducedInputs) gain(K *mat.Dense, ws *RiccatiWorkspace) *mat.Dense {
	m, _ := red.v.Dims()
	_, n := K.Dims()
	var out *mat.Dense
	if ws != nil {
		out = mat.NewDense(m, n, ws.kData[:m*n])
	} else {
		out = mat.NewDense(m, n, nil)
	}
	out.Mul(red.v, mat.DenseCopyOf(K))
	return out
}

// placedDare solves the Dare by placedHewer, state-scaled as dare.
func placedDare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	problem, err := newRiccatiProblem(A, B, Q, R, opts)
	if err != nil {
		return nil, err
	}
	if err := problem.ws.fits(problem.n, problem.m); err != nil {
		return nil, err
	}
	problem.scale(opts, false)
	eig := problem.ws.eig[:problem.n]
	X, K, rcnd, err := problem.placedHewer(eig)
	if err != nil {
		return nil, err
	}
	return problem.result(X, K, eig, rcnd, 0), nil
}

// pencilRiccati solves the Riccati equation from the extended pencil
// (pencilSolution) and returns X, K, rcnd and the Accuracy of X with Eig
// in ws.eig. A solution acceptRiccati rejects is replaced by the first
// accepted one of the other passes' solutions and then of lastResort;
// with none accepted, by problem.poor (Accuracy 1).
func (problem *riccatiProblem) pencilRiccati(continuous bool) (*mat.Dense, *mat.Dense, float64, int, error) {
	n := problem.n
	problem.collecting = true
	subspace, X, _, rcnd, err := problem.pencilSolution(continuous)
	eig := problem.ws.eig[:n]
	var K *mat.Dense
	if err == nil {
		for i := range n {
			eig[i] = complex(subspace.alphaR[i]/subspace.beta[i], subspace.alphaI[i]/subspace.beta[i])
		}
		K, err = problem.acceptRiccati(continuous, X, eig, rcnd)
	}
	if err != nil {
		for i := range problem.candidates {
			c := &problem.candidates[i]
			X, _, rcnd = problem.restorePass(c)
			for j := range n {
				eig[j] = complex(c.wr[j]/c.beta[j], c.wi[j]/c.beta[j])
			}
			if k, cerr := problem.acceptRiccati(continuous, X, eig, rcnd); cerr == nil {
				K, err = k, nil
				break
			}
		}
	}
	if err != nil {
		if x, k, rc, cerr := problem.lastResort(continuous, eig); cerr == nil {
			X, K, rcnd, err = x, k, rc, nil
		}
	}
	if err != nil {
		// Before poor: a pass with no Accuracy 1 X must not let the
		// other scaling mode return one.
		if problem.boundaryUnstabilizable(continuous) {
			return nil, nil, 0, 0, errBoundaryUnstabilizable
		}
		if problem.poor == nil {
			return nil, nil, 0, 0, err
		}
		X, K, rcnd = problem.poorSolution(eig)
		return X, K, rcnd, 1, nil
	}
	return X, K, rcnd, 0, nil
}

// lastResort solves the cost-balanced pencil at the cost scales of
// riccatiNorms.candidates not yet tried and returns the first solution
// verifyRiccati accepts.
func (problem *riccatiProblem) lastResort(continuous bool, eig []complex128) (*mat.Dense, *mat.Dense, float64, error) {
	problem.resolved = true
	tried := func(sigma float64) bool {
		for _, c := range problem.candidates {
			if c.sigma == sigma {
				return true
			}
		}
		return false
	}
	norms := problem.pencilNorms()
	rRaw := problem.R.RawMatrix()
	rn := impl.Dlange(lapack.Frobenius, problem.m, problem.m, rRaw.Data, rRaw.Stride, nil)
	for _, sigma := range []float64{norms.firstSigma(), pow2Near(rn), pow2Near(norms.q), pow2Near(math.Sqrt(rn) * math.Sqrt(norms.q))} {
		if !(sigma > 0) || math.IsInf(sigma, 0) || tried(sigma) {
			continue
		}
		problem.pencilNorms()
		problem.sigma, problem.balanced = sigma, true
		subspace, err := problem.generalizedStableSubspace(continuous, problem.pencil())
		if err != nil {
			continue
		}
		X, xData, rcnd, err := problem.stabilizingSolution(subspace.vectors)
		if err != nil || !finiteSlice(xData) {
			continue
		}
		for i := range eig {
			eig[i] = complex(subspace.alphaR[i]/subspace.beta[i], subspace.alphaI[i]/subspace.beta[i])
		}
		if K, err := problem.acceptRiccati(continuous, X, eig, rcnd); err == nil {
			return X, K, rcnd, nil
		}
		problem.candidates = append(problem.candidates, riccatiPass{sigma: sigma})
	}
	return nil, nil, 0, ErrNoStabilizing
}

// placedHewer is Dare's last source of a stabilizing solution, tried by
// Dare (placedDare, as given and then with NoScaling flipped) only
// when neither scaling mode's pencil gave a certified X nor a boundary
// verdict: Hewer's
// iteration, which from any stabilizing gain converges to the stabilizing
// solution, started from a pole-placement gain. Under cheap control of a
// strongly unstable plant (ρ(A) ≈ 500, Q/R ≈ 1e26, ‖X‖ ≈ 1e30) every cost
// scale of the pencil misplaces its eigenvalues or yields an X whose gain
// destabilizes, while Hewer from a placed gain reaches the oracle X to
// 1e-10. Placement (KNV, then Varga's Schur method) only supplies
// a K0 with E⁻¹(A − BK0) stable in float64, so its pole accuracy does not
// matter; the first iterate X0 solves the Stein equation of the cost of
// K0, A_K0ᵀX0A_K0 − EᵀX0E + Q_K0 = 0, and the result is the one
// verifyRiccati certifies from it (Accuracy 0), else an error. Rcnd is
// that of U11 for the orthonormal basis of span[I; X̂/σ], σ the power of
// two nearest ‖X̂‖ (the subspace a pencil at that cost scale deflates).
// It declines a cross term S with rank(B) < m: inputs in the null space
// of B lower the cost by about xᵀSR⁻¹Sᵀx, comparable to Q under cheap
// control, which R + BᵀXB (singular in float64) loses, so Newton then
// certifies a wrong X (≈ Q).
func (problem *riccatiProblem) placedHewer(eig []complex128) (*mat.Dense, *mat.Dense, float64, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	A, B := problem.A, problem.B
	var eLU mat.LU
	if problem.E != nil {
		eLU.Factorize(problem.E)
		var ea, eb mat.Dense
		if err := eLU.SolveTo(&ea, false, A); err != nil {
			return nil, nil, 0, ErrNoStabilizing
		}
		if err := eLU.SolveTo(&eb, false, B); err != nil {
			return nil, nil, 0, ErrNoStabilizing
		}
		A, B = &ea, &eb
	}
	var svd mat.SVD
	if !svd.Factorize(B, mat.SVDFull) {
		return nil, nil, 0, ErrNoStabilizing
	}
	sv := svd.Values(nil)
	r := 0
	for _, s := range sv {
		if s > float64(max(n, m))*eps()*sv[0] {
			r++
		}
	}
	if r == 0 || (problem.S != nil && r < m) {
		return nil, nil, 0, ErrNoStabilizing
	}
	nf := float64(n)
	poleSets := [][]complex128{make([]complex128, n), make([]complex128, n)}
	for i := range n {
		poleSets[0][i] = complex(0.1+0.8*float64(i)/nf, 0)
		poleSets[1][i] = complex(-0.6+1.2*(float64(i)+0.5)/nf, 0)
	}
	xData := ws.xData[:n*n]
	try := func(K0 *mat.Dense) (*mat.Dense, *mat.Dense, float64, error) {
		if K0 == nil || !finiteSlice(K0.RawMatrix().Data) {
			return nil, nil, 0, ErrNoStabilizing
		}
		if _, ok := problem.stableGain(false, K0); !ok {
			return nil, nil, 0, ErrNoStabilizing
		}
		var bk, acl, rk, qk mat.Dense
		bk.Mul(problem.B, K0)
		acl.Sub(problem.A, &bk)
		rk.Mul(problem.R, K0)
		qk.Mul(K0.T(), &rk)
		qk.Add(&qk, problem.Q)
		if problem.S != nil {
			var sk mat.Dense
			sk.Mul(problem.S, K0)
			qk.Sub(&qk, &sk)
			qk.Sub(&qk, sk.T())
		}
		f := mat.DenseCopyOf(acl.T())
		if problem.E != nil {
			var ft mat.Dense
			if err := eLU.SolveTo(&ft, false, &acl); err != nil {
				return nil, nil, 0, ErrNoStabilizing
			}
			f = mat.DenseCopyOf(ft.T())
		}
		symmetrize(qk.RawMatrix().Data, n, n)
		Y, err := dlyap(f, &qk, nil)
		if err != nil {
			return nil, nil, 0, ErrNoStabilizing
		}
		X0 := Y
		if problem.E != nil {
			var w, x mat.Dense
			if err := eLU.SolveTo(&w, true, Y); err != nil {
				return nil, nil, 0, ErrNoStabilizing
			}
			if err := eLU.SolveTo(&x, true, w.T()); err != nil {
				return nil, nil, 0, ErrNoStabilizing
			}
			X0 = &x
		}
		copyStrided(xData, n, X0.RawMatrix().Data, X0.RawMatrix().Stride, n, n)
		symmetrize(xData, n, n)
		if !finiteSlice(xData) {
			return nil, nil, 0, ErrNoStabilizing
		}
		X := mat.NewDense(n, n, xData)
		K, err := problem.acceptRiccati(false, X, eig, 0)
		if err != nil {
			return nil, nil, 0, err
		}
		return X, K, placedRcnd(xData, n), nil
	}
	for _, poles := range poleSets {
		if r >= 2 {
			if X, K, rcnd, err := try(placeKNV(A, &svd, sv[:r], poles)); err == nil {
				return X, K, rcnd, nil
			}
		}
		if K0, err := placeSchur(A, B, poles); err == nil {
			if X, K, rcnd, err := try(K0); err == nil {
				return X, K, rcnd, nil
			}
		}
	}
	return nil, nil, 0, ErrNoStabilizing
}

// placedRcnd returns the reciprocal 2-norm condition of U11 for the
// orthonormal basis of span[I; X/σ], σ the power of two nearest ‖X‖_F:
// √((1 + λmin²)/(1 + λmax²)) over the eigenvalues λ of X/σ ordered by
// magnitude.
func placedRcnd(xData []float64, n int) float64 {
	sigma := pow2Near(impl.Dlange(lapack.Frobenius, n, n, xData, n, nil))
	var es mat.EigenSym
	if !es.Factorize(mat.NewSymDense(n, slices.Clone(xData)), false) {
		return 0
	}
	lo, hi := math.Inf(1), 0.0
	for _, v := range es.Values(nil) {
		a := math.Abs(v / sigma)
		lo, hi = min(lo, a), max(hi, a)
	}
	return math.Sqrt((1 + lo*lo) / (1 + hi*hi))
}

// acceptRiccati returns the gain of the solution X (with rcnd), which it
// verifies (verifyRiccati); X and eig may be refined in place.
func (problem *riccatiProblem) acceptRiccati(continuous bool, X *mat.Dense, eig []complex128, rcnd float64) (*mat.Dense, error) {
	K, err := problem.gain(continuous, X.RawMatrix().Data)
	if err != nil {
		return nil, err
	}
	return problem.verifyRiccati(continuous, X, K, eig, rcnd)
}

// riccatiPoor is a stabilizing solution no Newton step certified: X, its
// gain K, the eigenvalues of A − BK, rcnd and its first Newton step.
type riccatiPoor struct {
	x, k       []float64
	eig        []complex128
	rcnd, step float64
}

// keepPoor saves X (xData, gain K) in problem.poor with the eigenvalues
// eig of A − BK (nil: those of the last newtonCost), unless the saved one
// has a smaller first Newton step or rcnd is below poorRcnd.
func (problem *riccatiProblem) keepPoor(xData []float64, K *mat.Dense, eig []complex128, rcnd, step float64) bool {
	if !(rcnd >= poorRcnd) {
		return false
	}
	p := problem.poor
	if p == nil {
		p = &riccatiPoor{}
		problem.poor = p
	} else if !(step < p.step) {
		return false
	}
	n, m := problem.n, problem.m
	p.x = append(p.x[:0], xData...)
	p.k = append(p.k[:0], K.RawMatrix().Data[:m*n]...)
	if eig != nil {
		p.eig = append(p.eig[:0], eig...)
	} else {
		p.eig = append(p.eig[:0], make([]complex128, n)...)
		problem.closedLoopEig(p.eig)
	}
	p.rcnd, p.step = rcnd, step
	return true
}

// errBoundaryUnstabilizable is the ErrNoStabilizing of
// boundaryUnstabilizable: structural, so it holds in either scaling mode
// over any Accuracy 1 X, though not over a certified one (otherScaling).
var errBoundaryUnstabilizable = fmt.Errorf("%w: a mode on or beyond the stability boundary is unstabilizable", ErrNoStabilizing)

// boundaryUnstabilizable reports a finite eigenvalue λ of (A, E) on or
// beyond the stability boundary, or within rounding of it, at which
// [A − λE, B] has rank below n by the PBH test: no stabilizing solution
// exists (icare/idare Report 3), and an Accuracy 1 X would be stable only
// to rounding, its exact closed loop unstable (K72KOR). Within rounding
// is 10× the first-order bound τ = (n·eps·(|y|ᵀ|A||x| + |λ|·|y|ᵀ|E||x|)
// + |y|ᵀ|(A − λE)x|)/|yᴴEx| of λ's error (x, y its right and left
// eigenvectors): componentwise rounding plus the residual, which carries
// QZ's backward error, so an exact mode at 0 or 1 of a dense A, computed
// eps·‖A‖·κ off, is tested whatever the rounding (zero1 under -race), while a mode at −3e-16 that is an exact diagonal entry
// of a triangular A is known to relative eps and stays stable (BWUCB5),
// as does a lightly damped mode at −3e-9 ± i. PBH rank is σmin ≤
// n·eps·‖[‖A‖ + |λ|‖E‖, ‖B‖]‖ + τ‖E‖. A mode in a Jordan block with a
// controllable one splits by √(eps·κ)·‖A‖ beyond first order: x ⊥ y
// there, and τ, past the split, would swallow a stable mode at −1e-5. λ is
// tested alone only when τ is below its distance to every other
// eigenvalue; the mean of each nearest-first cluster of λ within
// max(√eps·(‖A‖ + |λ|‖E‖), τ) is tested too, to 10·n·eps·(‖A‖ +
// |mean|‖E‖) times the cluster condition 1/min(pl, pr) (Dtgsen; a Jordan
// block at 0 with an ill-conditioned P is off by more than eps·‖A‖,
// MPWOJM) when it is within √eps of the boundary, and only when that
// band is below the mean's distance to the eigenvalues left out.
func (problem *riccatiProblem) boundaryUnstabilizable(continuous bool) bool {
	n, m := problem.n, problem.m
	aRaw, bRaw := problem.A.RawMatrix(), problem.B.RawMatrix()
	a := make([]float64, n*n)
	copyStrided(a, n, aRaw.Data, aRaw.Stride, n, n)
	e := make([]float64, n*n)
	for i := range n {
		e[i*n+i] = 1
	}
	if problem.E != nil {
		eRaw := problem.E.RawMatrix()
		copyStrided(e, n, eRaw.Data, eRaw.Stride, n, n)
	}
	an, en := frobenius(a), frobenius(e)
	bn := impl.Dlange(lapack.Frobenius, n, m, bRaw.Data, bRaw.Stride, nil)
	ac, ec := slices.Clone(a), slices.Clone(e)
	wr, wi, beta := make([]float64, n), make([]float64, n), make([]float64, n)
	vl, vr := make([]float64, n*n), make([]float64, n*n)
	work := make([]float64, max(1, 8*n))
	if !impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n, ac, n, ec, n, wr, wi, beta, vl, n, vr, n, work, len(work)) {
		return false
	}
	nf, ep := float64(n), eps()
	inBand := func(lam complex128, tol float64) bool {
		if continuous {
			return real(lam) >= -tol
		}
		return cmplx.Abs(lam) >= 1-tol
	}
	g := mat.NewDense(2*n, 2*(n+m), nil)
	var svd mat.SVD
	pbhDeficient := func(lam complex128, tol float64) bool {
		lr, li := real(lam), imag(lam)
		for i := range n {
			for j := range n {
				re, im := a[i*n+j]-lr*e[i*n+j], -li*e[i*n+j]
				g.Set(i, j, re)
				g.Set(i, n+m+j, -im)
				g.Set(n+i, j, im)
				g.Set(n+i, n+m+j, re)
			}
			for j := range m {
				b := bRaw.Data[i*bRaw.Stride+j]
				g.Set(i, n+j, b)
				g.Set(n+i, n+m+n+j, b)
			}
		}
		if !svd.Factorize(g, mat.SVDNone) {
			return false
		}
		sv := svd.Values(nil)
		return sv[2*n-1] <= nf*ep*math.Hypot(an+cmplx.Abs(lam)*en, bn)+tol*en
	}
	var sa, se, sar, sai, sbeta []float64
	clusterCond := func(lam complex128, rho float64) float64 {
		if sa == nil {
			sa, se = slices.Clone(a), slices.Clone(e)
			sar, sai, sbeta = make([]float64, n), make([]float64, n), make([]float64, n)
			var query [1]float64
			impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
				n, sa, n, se, n, sar, sai, sbeta, nil, 1, nil, 1, query[:], -1, nil)
			w := make([]float64, int(query[0]))
			if _, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
				n, sa, n, se, n, sar, sai, sbeta, nil, 1, nil, 1, w, len(w), nil); !ok {
				sbeta = nil
			}
		}
		if sbeta == nil {
			return 1
		}
		sel := make([]bool, n)
		for j := 0; j < n; j++ {
			in := func(k int) bool {
				return sbeta[k] != 0 && cmplx.Abs(complex(sar[k]/sbeta[k], sai[k]/sbeta[k])-lam) <= rho
			}
			if sai[j] != 0 && j+1 < n {
				sel[j] = in(j) || in(j+1)
				sel[j+1] = sel[j]
				j++
				continue
			}
			sel[j] = in(j)
		}
		ta, te := slices.Clone(sa), slices.Clone(se)
		ar, ai, b := slices.Clone(sar), slices.Clone(sai), slices.Clone(sbeta)
		lw := max(4*n+16, n*n)
		_, pl, pr, _, ok := impl.Dtgsen(1, false, false, sel, n, ta, n, te, n, ar, ai, b,
			nil, 1, nil, 1, make([]float64, lw), lw, make([]int, n+6), n+6)
		if !ok || !(pl > 0) || !(pr > 0) {
			return 1
		}
		return 1 / min(pl, pr)
	}
	lams := make([]complex128, n)
	finite := make([]bool, n)
	for k := range n {
		if finite[k] = math.Abs(beta[k]) > 0; finite[k] {
			lams[k] = complex(wr[k]/beta[k], wi[k]/beta[k])
		}
	}
	x, y := make([]complex128, n), make([]complex128, n)
	ds := make([]float64, 0, n)
	for k := 0; k < n; k++ {
		kk := k
		if wi[k] != 0 {
			k++
		}
		if !finite[kk] {
			continue
		}
		lam := lams[kk]
		for i := range n {
			if imag(lam) == 0 {
				x[i], y[i] = complex(vr[i*n+kk], 0), complex(vl[i*n+kk], 0)
			} else {
				x[i], y[i] = complex(vr[i*n+kk], vr[i*n+kk+1]), complex(vl[i*n+kk], vl[i*n+kk+1])
			}
		}
		var ya, ye, yr float64
		var yex complex128
		for i := range n {
			var ra, re float64
			var ax, ex complex128
			for j := range n {
				ra += math.Abs(a[i*n+j]) * cmplx.Abs(x[j])
				re += math.Abs(e[i*n+j]) * cmplx.Abs(x[j])
				ax += complex(a[i*n+j], 0) * x[j]
				ex += complex(e[i*n+j], 0) * x[j]
			}
			ay := cmplx.Abs(y[i])
			ya += ay * ra
			ye += ay * re
			yr += ay * cmplx.Abs(ax-lam*ex)
			yex += cmplx.Conj(y[i]) * ex
		}
		mod := cmplx.Abs(lam)
		tol := 10 * (nf*ep*(ya+mod*ye) + yr) / cmplx.Abs(yex)
		rho := math.Sqrt(ep) * (an + mod*en)
		ds = ds[:0]
		for j := range n {
			if finite[j] {
				ds = append(ds, cmplx.Abs(lams[j]-lam))
			}
		}
		slices.Sort(ds)
		gap := math.Inf(1)
		if len(ds) > 1 {
			gap = ds[1]
		}
		if tol < gap && inBand(lam, tol) && pbhDeficient(lam, tol) {
			return true
		}
		for c := 1; c < len(ds) && ds[c] <= max(rho, tol); c++ {
			r := ds[c]
			if c+1 < len(ds) {
				if ds[c+1] == r {
					continue
				}
				r = (r + ds[c+1]) / 2
			}
			var mean complex128
			for j := range n {
				if finite[j] && cmplx.Abs(lams[j]-lam) <= r {
					mean += lams[j]
				}
			}
			mean /= complex(float64(c+1), 0)
			sep := math.Inf(1)
			for j := range n {
				if finite[j] && cmplx.Abs(lams[j]-lam) > r {
					sep = min(sep, cmplx.Abs(lams[j]-mean))
				}
			}
			scale := an + cmplx.Abs(mean)*en
			mtol := 10 * nf * ep * scale
			if !inBand(mean, mtol) && inBand(mean, math.Sqrt(ep)*scale) {
				mtol *= clusterCond(lam, r)
			}
			if mtol < sep && inBand(mean, mtol) && pbhDeficient(mean, mtol) {
				return true
			}
		}
	}
	return false
}

// poorSolution restores problem.poor to ws.xData, ws.kData and eig.
func (problem *riccatiProblem) poorSolution(eig []complex128) (X, K *mat.Dense, rcnd float64) {
	n, m, ws, p := problem.n, problem.m, problem.ws, problem.poor
	xData, kData := ws.xData[:n*n], ws.kData[:m*n]
	copy(xData, p.x)
	copy(kData, p.k)
	copy(eig, p.eig)
	return mat.NewDense(n, n, xData), mat.NewDense(m, n, kData), p.rcnd
}

// gain returns the gain of X: careGain or dareGain.
func (problem *riccatiProblem) gain(continuous bool, xData []float64) (*mat.Dense, error) {
	if continuous {
		return problem.careGain(xData), nil
	}
	return problem.dareGain(xData)
}

// careGain returns K = R⁻¹(BᵀXE + Sᵀ) (E = I when absent) in ws.kData;
// R is Cholesky-factored in ws.rChol.
func (problem *riccatiProblem) careGain(xData []float64) *mat.Dense {
	n, m, ws := problem.n, problem.m, problem.ws
	bRaw := problem.B.RawMatrix()
	kData := ws.kData[:m*n]
	btx := kData
	if problem.E != nil {
		btx = ws.btx[:m*n]
	}
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n})
	if problem.E != nil {
		eRaw := problem.E.RawMatrix()
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
			blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
			0, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n})
	}
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
	return mat.NewDense(m, n, kData)
}

// dareGain returns K = (R + BᵀXB)⁻¹(BᵀXA + Sᵀ) in ws.kData, or
// ErrNoStabilizing when R + BᵀXB is not positive definite.
func (problem *riccatiProblem) dareGain(xData []float64) (*mat.Dense, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	aRaw, bRaw, rRaw := problem.A.RawMatrix(), problem.B.RawMatrix(), problem.R.RawMatrix()
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

	kData := ws.kData[:m*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: aRaw.Data, Stride: aRaw.Stride},
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
	impl.Dpotrs(blas.Upper, m, n, rbar, m, kData, n)
	return mat.NewDense(m, n, kData), nil
}

// verifyRiccati returns the gain of the solution X (gain K) once a Newton
// step certifies it. A step is ‖X₁ − X‖/‖X₁‖ (stepNorms) for the Newton
// iterate X₁ = X + Δ of newtonCost (Newton–Kleinman for Care, Hewer for
// Dare), whose Δ solves the Lyapunov (Stein) equation of A − BK for the
// double-double Riccati residual of X: to first order the forward error
// of X. X is kept when its step is within newtonTol with A − BK stable.
//
// The relative Riccati residual is no certificate: under cheap control
// the rounded stabilizing X has a residual of 1e-6 of its terms while an
// X wrong by 5e-7 has 3e-11 (Dare w15 #16), and against the rounding of
// the gain form both sit at 1e-17. The step errs the safe way for the
// pencil's X (it overstates the error where Newton has no quadratic
// region, cheap control with ‖X‖‖BR⁻¹Bᵀ‖‖L_K⁻¹‖ ≫ 1), but not for Newton
// iterates, whose error can lie where the step does not see it: a step
// of 3e-9 from an iterate wrong by 2.4e-7, or Newton–Kleinman creeping
// from an X accurate to 1e-11 to 1e-6 and stalling there. So a larger
// first step is iterated, up to newtonSteps solves, and an iterate is kept
// only once its step is within newtonRefineTol, where Newton has
// converged to its rounding level; the iterate and its K then replace X
// and K. Eig is that of A − BK for the returned K. A first solve whose
// Lyapunov equation is singular is kept with a residual within rescueTol
// of its terms and a stable A − BK as a candidate of problem.poor, as is
// a stabilizing X whose first step exceeds newtonTol (keepPoor, before
// any iterate overwrites it). From a stabilizing K, Newton's iterates
// stay stabilizing and converge in exact arithmetic, so when they break
// down (an iterate unstable or unsolvable) or newtonSteps solves pass
// without convergence or a stall within newtonGlobalTol, rounding rules
// the iteration and the first step is no error estimate (75DTNG: an
// unscaled non-normal pencil X off by 5e8, first step 0.5): the
// candidate it displaced is restored, else, as a singular first solve,
// it is kept with step +Inf only with a residual within rescueTol of its
// terms (3XCWZW: a Hamiltonian pair on the axis to rounding left an X of
// residual 0.9 of its terms, first step 1, that no stabilizing solution
// underlies). Otherwise verifyRiccati returns
// problem.rescueErr, or ErrNoStabilizing, and the caller falls back to
// problem.poor with Accuracy 1. X and eig are updated in place.
func (problem *riccatiProblem) verifyRiccati(continuous bool, X, K *mat.Dense, eig []complex128, rcnd float64) (*mat.Dense, error) {
	fail := problem.rescueErr
	if fail == nil {
		fail = ErrNoStabilizing
	}
	xData := X.RawMatrix().Data
	bestStep, stalls := math.Inf(1), 0
	kept, prevPoor := false, (*riccatiPoor)(nil)
	defer func() {
		if !kept {
			return
		}
		if prevPoor != nil {
			problem.poor = prevPoor
			return
		}
		if res, scale := problem.residualTerms(continuous, problem.poor.x); res <= rescueTol*scale {
			problem.poor.step = math.Inf(1)
		} else {
			problem.poor = nil
		}
	}()
	for it := range newtonSteps {
		x1, err := problem.newtonCost(continuous, K, xData)
		if err != nil {
			if it == 0 && !errors.Is(err, ErrNoStabilizing) && !problem.resolved {
				if res, scale := problem.residualTerms(continuous, xData); res <= rescueTol*scale {
					if vals, ok := problem.stableGain(continuous, K); ok {
						problem.keepPoor(xData, K, vals, rcnd, math.Inf(1))
					}
				}
			}
			return nil, fail
		}
		d, x1n := problem.stepNorms(x1, xData)
		step := 0.0
		if d != 0 {
			step = math.Sqrt(d / x1n)
		}
		if it == 0 {
			if step <= newtonTol {
				problem.closedLoopEig(eig)
				return K, nil
			}
			if p := problem.poor; p != nil && step < p.step {
				prevPoor = &riccatiPoor{slices.Clone(p.x), slices.Clone(p.k), slices.Clone(p.eig), p.rcnd, p.step}
			}
			kept = problem.keepPoor(xData, K, nil, rcnd, step)
		}
		if it > 0 && step <= newtonRefineTol {
			kept = false
			prev := append([]float64(nil), xData...)
			copy(xData, x1)
			if K1, err := problem.closedLoopGain(continuous, xData, eig); err == nil {
				return K1, nil
			}
			copy(xData, prev)
			if K, err = problem.closedLoopGain(continuous, xData, eig); err != nil {
				return nil, fail
			}
			return K, nil
		}
		if step < bestStep {
			bestStep, stalls = step, 0
		} else if stalls++; stalls == 3 && step <= newtonGlobalTol {
			kept = false
			break
		}
		copy(xData, x1)
		if K, err = problem.gain(continuous, xData); err != nil {
			break
		}
	}
	return nil, fail
}

// stepNorms returns ‖X₁ − X‖² and ‖X₁‖² (Frobenius) of x1 and xData in
// the caller's coordinates, X = T·X̂·T, so that a step measures the
// relative error of the returned X rather than that of the state-scaled
// iterate, where an error in the small entries of X̂ can hide.
func (problem *riccatiProblem) stepNorms(x1, xData []float64) (d, x1n float64) {
	t, n := problem.t, problem.n
	for i, v := range x1 {
		e := v - xData[i]
		if t != nil {
			w := t[i/n] * t[i%n]
			v, e = v*w, e*w
		}
		d += e * e
		x1n += v * v
	}
	return d, x1n
}

// closedLoopGain returns the gain K of xData with the eigenvalues of
// A − BK in eig, or an error when A − BK is not stable or its cost does
// not solve.
func (problem *riccatiProblem) closedLoopGain(continuous bool, xData []float64, eig []complex128) (*mat.Dense, error) {
	K, err := problem.gain(continuous, xData)
	if err != nil {
		return nil, err
	}
	if _, err := problem.newtonCost(continuous, K, xData); err != nil {
		return nil, err
	}
	problem.closedLoopEig(eig)
	return K, nil
}

// closedLoopEig copies the eigenvalues of A − BK of the last newtonCost
// to eig.
func (problem *riccatiProblem) closedLoopEig(eig []complex128) {
	lw := problem.ws.lyap
	for i := range eig {
		eig[i] = complex(lw.wr[i], lw.wi[i])
	}
}

// newtonCost returns the Newton iterate X₁ = X + Δ of the iterate X
// (xData) of gain K of verifyRiccati, with the eigenvalues of A − BK
// (with E, of (A − BK, E)) in ws.lyap.wr, ws.lyap.wi; ErrNoStabilizing
// when they are not stable, or the error of the Lyapunov solve. With
// A_K = A − BK (E = I when absent),
//
//	continuous: A_KᵀΔE + EᵀΔA_K + Ric(X) = 0
//	discrete:   A_KᵀΔA_K − EᵀΔE + Ric(X) = 0
//
// solved with E as FᵀY + YF + Ric(X) = 0 or FᵀYF − Y + Ric(X) = 0 for
// F = E⁻¹A_K and Y = EᵀΔE,
// for the Riccati residual Ric(X) of riccatiResidual, formed in
// double-double arithmetic: in float64 its terms (‖A_K‖²‖X‖, KᵀRK under
// an ill-conditioned R, A − BK under cheap control) round it to a noise
// that the solve maps to a step of the size of the error it should
// measure, while Δ's own rounding is relative to the small ‖Δ‖.
func (problem *riccatiProblem) newtonCost(continuous bool, K *mat.Dense, xData []float64) ([]float64, error) {
	n, ws := problem.n, problem.ws
	if ws.lyap == nil {
		ws.lyap = &LyapunovWorkspace{}
	}
	aclT := ws.aWork[:n*n]
	res := ws.z[:n*n]
	problem.riccatiResidual(continuous, K.RawMatrix().Data, xData, aclT, res)
	symmetrize(res, n, n)

	var eLU []float64
	var eIpiv []int
	if problem.E != nil {
		eLU, eIpiv = problem.descriptorLU()
		f := ws.h[:n*n]
		for i := range n {
			for j := range n {
				f[j*n+i] = aclT[i*n+j]
			}
		}
		impl.Dgetrs(blas.NoTrans, n, n, eLU, n, eIpiv, f, n)
		for i := range n {
			for j := range n {
				aclT[j*n+i] = f[i*n+j]
			}
		}
	}

	solve, stable := dlyap, func(wr, wi float64) bool { return math.Hypot(wr, wi) < 1 }
	if continuous {
		solve, stable = lyap, func(wr, _ float64) bool { return wr < 0 }
	}
	D, err := solve(mat.NewDense(n, n, aclT), mat.NewDense(n, n, res), &LyapunovOpts{Workspace: ws.lyap})
	if err != nil && !errors.Is(err, ErrSingularEquation) {
		return nil, err
	}
	lw := ws.lyap
	for i := range n {
		if !stable(lw.wr[i], lw.wi[i]) {
			return nil, ErrNoStabilizing
		}
	}
	if err != nil {
		return nil, err
	}
	x1 := D.RawMatrix().Data[:n*n]
	if eLU != nil {
		impl.Dgetrs(blas.Trans, n, n, eLU, n, eIpiv, x1, n)
		for i := range n {
			for j := i + 1; j < n; j++ {
				x1[i*n+j], x1[j*n+i] = x1[j*n+i], x1[i*n+j]
			}
		}
		impl.Dgetrs(blas.Trans, n, n, eLU, n, eIpiv, x1, n)
		symmetrize(x1, n, n)
	}
	for i, v := range xData {
		x1[i] += v
	}
	if !finiteSlice(x1) {
		return nil, ErrSingularEquation
	}
	return x1, nil
}

// descriptorLU returns the LU factors of E (row-major, with pivots) in
// ws.eLU and ws.eIpiv, factored on first use.
func (problem *riccatiProblem) descriptorLU() ([]float64, []int) {
	n, ws := problem.n, problem.ws
	if !problem.eFactored {
		if len(ws.eLU) < n*n {
			ws.eLU, ws.eIpiv = make([]float64, n*n), make([]int, n)
		}
		eRaw := problem.E.RawMatrix()
		copyStrided(ws.eLU[:n*n], n, eRaw.Data, eRaw.Stride, n, n)
		impl.Dgetrf(n, n, ws.eLU[:n*n], n, ws.eIpiv[:n])
		problem.eFactored = true
	}
	return ws.eLU[:n*n], ws.eIpiv[:n]
}

// riccatiResidual writes (A − BK)ᵀ to aclT and the Riccati residual of
// xData to res, rounded from double-double sums of exact products
// (ddMulAcc). It forms the cost residual of xData under the gain kData,
//
//	continuous: A_KᵀXE + EᵀXA_K + Q_K
//	discrete:   A_KᵀXA_K − EᵀXE + Q_K
//
// (E = I when absent) with Q_K = Q + KᵀRK − SK − KᵀSᵀ, and subtracts its
// excess over the K-free residual, GᵀW⁻¹G with G = BᵀXE + Sᵀ − RK, W = R
// (continuous) or G = BᵀXA_K + Sᵀ − RK, W = R + BᵀXB (discrete), so that
// the rounding of the gain drops out. It uses ws.h, ws.btx, ws.costV,
// ws.gainRes, ws.rbar and, with E, ws.eRes as scratch.
func (problem *riccatiProblem) riccatiResidual(continuous bool, kData, xData, aclT, res []float64) {
	n, m, ws := problem.n, problem.m, problem.ws
	aRaw, bRaw, qRaw, rRaw := problem.A.RawMatrix(), problem.B.RawMatrix(), problem.Q.RawMatrix(), problem.R.RawMatrix()
	nn := n * n
	acH, acL := ws.h[:nn], ws.h[nn:2*nn]
	yH, yL := ws.h[2*nn:3*nn], ws.h[3*nn:4*nn]
	copyStrided(acH, n, aRaw.Data, aRaw.Stride, n, n)
	clear(acL)
	ddMulAcc(n, n, m, -1, bRaw.Data, bRaw.Stride, 1, kData, n, 1, acH, acL, n)
	ddNormalize(acH, acL)
	for i := range n {
		for j := range n {
			aclT[j*n+i] = acH[i*n+j]
		}
	}

	clear(yH)
	clear(yL)
	ddMulAcc(n, n, n, 1, xData, n, 1, acH, n, 1, yH, yL, n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: acL, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: yL, Stride: n})

	var eRaw blas64.General
	var xeH, xeL []float64
	if problem.E != nil {
		eRaw = problem.E.RawMatrix()
		if len(ws.eRes) < 4*nn {
			ws.eRes = make([]float64, 4*nn)
		}
		xeH, xeL = ws.eRes[2*nn:3*nn], ws.eRes[3*nn:4*nn]
		clear(xeH)
		clear(xeL)
		ddMulAcc(n, n, n, 1, xData, n, 1, eRaw.Data, eRaw.Stride, 1, xeH, xeL, n)
	}

	resL := ws.z[nn : 2*nn]
	copyStrided(res, n, qRaw.Data, qRaw.Stride, n, n)
	clear(resL)
	if continuous {
		zH, zL := yH, yL
		if problem.E != nil {
			zH, zL = ws.eRes[:nn], ws.eRes[nn:2*nn]
			clear(zH)
			clear(zL)
			ddMulAcc(n, n, n, 1, eRaw.Data, 1, eRaw.Stride, yH, n, 1, zH, zL, n)
			blas64.Gemm(blas.Trans, blas.NoTrans, 1,
				blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
				blas64.General{Rows: n, Cols: n, Data: yL, Stride: n},
				1, blas64.General{Rows: n, Cols: n, Data: zL, Stride: n})
		}
		for i := range n {
			for j := range n {
				ddAdd(&res[i*n+j], &resL[i*n+j], zH[i*n+j], zL[i*n+j])
				ddAdd(&res[i*n+j], &resL[i*n+j], zH[j*n+i], zL[j*n+i])
			}
		}
	} else {
		if problem.E == nil {
			for i, v := range xData {
				ddAdd(&res[i], &resL[i], -v, 0)
			}
		} else {
			ddMulAcc(n, n, n, -1, eRaw.Data, 1, eRaw.Stride, xeH, n, 1, res, resL, n)
			blas64.Gemm(blas.Trans, blas.NoTrans, -1,
				blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
				blas64.General{Rows: n, Cols: n, Data: xeL, Stride: n},
				1, blas64.General{Rows: n, Cols: n, Data: resL, Stride: n})
		}
		ddMulAcc(n, n, n, 1, acH, 1, n, yH, n, 1, res, resL, n)
		ac := blas64.General{Rows: n, Cols: n, Data: acH, Stride: n}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, ac,
			blas64.General{Rows: n, Cols: n, Data: yL, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: resL, Stride: n})
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: n, Data: acL, Stride: n},
			blas64.General{Rows: n, Cols: n, Data: yH, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: resL, Stride: n})
	}

	rkH, rkL := ws.btx[:m*n], ws.costV[:m*n]
	clear(rkH)
	clear(rkL)
	ddMulAcc(m, n, m, 1, rRaw.Data, rRaw.Stride, 1, kData, n, 1, rkH, rkL, n)
	ddMulAcc(n, n, m, 1, kData, 1, n, rkH, n, 1, res, resL, n)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n},
		blas64.General{Rows: m, Cols: n, Data: rkL, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: resL, Stride: n})
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		ddMulAcc(n, n, m, -1, sRaw.Data, sRaw.Stride, 1, kData, n, 1, res, resL, n)
		ddMulAcc(n, n, m, -1, kData, 1, n, sRaw.Data, 1, sRaw.Stride, res, resL, n)
	}
	for i := range res {
		res[i] += resL[i]
	}

	gH, gL := ws.gainRes[:m*n], ws.gainRes[m*n:2*m*n]
	for i := range gH {
		gH[i], gL[i] = -rkH[i], -rkL[i]
	}
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		for i := range m {
			for j := range n {
				ddAdd(&gH[i*n+j], &gL[i*n+j], sRaw.Data[j*sRaw.Stride+i], 0)
			}
		}
	}
	w := ws.rbar[:m*m]
	if continuous {
		if problem.E == nil {
			ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, xData, n, 1, gH, gL, n)
		} else {
			ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, xeH, n, 1, gH, gL, n)
			blas64.Gemm(blas.Trans, blas.NoTrans, 1,
				blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
				blas64.General{Rows: n, Cols: n, Data: xeL, Stride: n},
				1, blas64.General{Rows: m, Cols: n, Data: gL, Stride: n})
		}
		copy(w, ws.rChol[:m*m])
	} else {
		ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, yH, n, 1, gH, gL, n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: yL, Stride: n},
			1, blas64.General{Rows: m, Cols: n, Data: gL, Stride: n})
		bx := ws.btx[:m*n]
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
			0, blas64.General{Rows: m, Cols: n, Data: bx, Stride: n})
		copyStrided(w, m, rRaw.Data, rRaw.Stride, m, m)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: m, Cols: n, Data: bx, Stride: n},
			blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			1, blas64.General{Rows: m, Cols: m, Data: w, Stride: m})
		if !impl.Dpotrf(blas.Upper, m, w, m) {
			return
		}
	}
	for i := range gH {
		gH[i] += gL[i]
	}
	copy(gL, gH)
	impl.Dpotrs(blas.Upper, m, n, w, m, gL, n)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		-1, blas64.General{Rows: m, Cols: n, Data: gH, Stride: n},
		blas64.General{Rows: m, Cols: n, Data: gL, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: res, Stride: n})
}

// ddMulAcc adds alpha·A·B to the double-double matrix (hi, lo) (r×c,
// stride ldc), with A r×k at a[i*ai + p*ap] and B k×c at b[p*bp + j*bj],
// alpha ±1. Products are exact (FMA) and sums compensated (TwoSum).
func ddMulAcc(r, c, k int, alpha float64, a []float64, ai, ap int, b []float64, bp, bj int, hi, lo []float64, ldc int) {
	for i := range r {
		h, l := hi[i*ldc:i*ldc+c], lo[i*ldc:i*ldc+c]
		for p := range k {
			x := alpha * a[i*ai+p*ap]
			if x == 0 {
				continue
			}
			bRow := b[p*bp:]
			for j := range h {
				y := bRow[j*bj]
				// The conversion stops Go fusing x*y into the sum
				// below (arm64 FMA), which would lose pe.
				pr := float64(x * y)
				pe := math.FMA(x, y, -pr)
				sh := h[j]
				s := sh + pr
				bb := s - sh
				l[j] += (sh - (s - bb)) + (pr - bb) + pe
				h[j] = s
			}
		}
	}
}

// ddAdd adds the double-double (h, l) to (*hi, *lo).
func ddAdd(hi, lo *float64, h, l float64) {
	s := *hi + h
	bb := s - *hi
	*lo += (*hi - (s - bb)) + (h - bb) + l
	*hi = s
}

// ddNormalize rounds each hi[i] + lo[i] to hi[i] and leaves the rest in
// lo[i].
func ddNormalize(hi, lo []float64) {
	for i, h := range hi {
		s := h + lo[i]
		lo[i] -= s - h
		hi[i] = s
	}
}

// newtonTol bounds the first Newton step that keeps the given X in
// verifyRiccati, newtonRefineTol the step that ends an iteration, and
// newtonSteps its solves. Three steps without a new smallest one end it
// once within newtonGlobalTol; larger steps are Newton–Kleinman's global
// phase, which need not decrease.
//
// poorRcnd = eps bounds the rcnd of U11 of an Accuracy 1 X from below:
// under it U11 is numerically singular and X not finite (MATLAB
// icare/idare Report 2), its K so large that a float64-stable A − BK
// says nothing (Kalman fuzz e5d054af185f91b9 under FMA rounding: rcnd
// 1.5e-17, three exact closed-loop roots outside the unit circle).
// weakBStall's X has rcnd 8e-15.
const (
	poorRcnd        = 0x1p-52
	newtonTol       = 1e-8
	newtonRefineTol = 1e-10
	newtonGlobalTol = 1e-5
	newtonSteps     = 30
)

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
		if _, ok := err.(mat.Condition); !ok || !finiteSlice(w.RawMatrix().Data) {
			return math.Inf(1), math.Inf(1)
		}
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
// verifyRiccati, a stable closed loop), else the first error is returned. Under
// NoScaling a first X with a residual above noScalingTol of its terms is
// rescued too, keeping the solution of smallest residual.
func (problem *riccatiProblem) pencilSolution(continuous bool) (subspace riccatiSubspace, X *mat.Dense, xData []float64, rcnd float64, err error) {
	var norms riccatiNorms
	var prev *riccatiPass
	rescue := func() {
		problem.resolved, problem.costScaling = true, true
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
				problem.keepCandidate(xData, rcnd)
				rescue()
				continue
			}
			norms = problem.pencilNorms()
		}
		if err == nil && problem.rescaleCost(continuous, norms, xData, &prev, rcnd) {
			problem.resolved = true
			problem.keepCandidate(xData, rcnd)
			continue
		}
		if err != nil || problem.keepBest(continuous, prev, xData) {
			if err == nil {
				problem.keepCandidate(xData, rcnd)
			}
			subspace, X, xData, rcnd = problem.restoredSolution(prev)
		}
		if problem.rescueErr != nil {
			if res, scale := problem.residualTerms(continuous, xData); !(res <= rescueTol*scale) {
				problem.keepCandidate(xData, rcnd)
				return riccatiSubspace{}, nil, nil, 0, problem.rescueErr
			}
		}
		return subspace, X, xData, rcnd, nil
	}
}

// keepCandidate saves the solution xData of a pencil solve in
// problem.candidates when collecting is set and another solve follows or
// replaces it.
func (problem *riccatiProblem) keepCandidate(xData []float64, rcnd float64) {
	if !problem.collecting {
		return
	}
	ws, n := problem.ws, problem.n
	problem.candidates = append(problem.candidates, riccatiPass{
		x:    append([]float64(nil), xData...),
		wr:   append([]float64(nil), ws.wr[:n]...),
		wi:   append([]float64(nil), ws.wi[:n]...),
		beta: append([]float64(nil), ws.beta[:n]...),
		rcnd: rcnd, sigma: problem.sigma,
	})
}

// rescueTol and noScalingTol bound the Riccati residual of pencilSolution
// relative to the sum of its terms: a rescued solution above rescueTol is
// rejected, and an unscaled one above noScalingTol is rescued.
const (
	rescueTol    = 1e-6
	noScalingTol = 1e-12
)

// stableGain returns the eigenvalues of A − BK (with E: E⁻¹(A − BK)) and
// reports whether they are stable.
func (problem *riccatiProblem) stableGain(continuous bool, K *mat.Dense) ([]complex128, bool) {
	var bk, ea mat.Dense
	bk.Mul(problem.B, K)
	acl := mat.NewDense(problem.n, problem.n, nil)
	acl.Sub(problem.A, &bk)
	if problem.E != nil {
		var lu mat.LU
		lu.Factorize(problem.E)
		if err := lu.SolveTo(&ea, false, acl); err != nil {
			if _, ok := err.(mat.Condition); !ok {
				return nil, false
			}
		}
		acl = &ea
	}
	var eig mat.Eigen
	if !eig.Factorize(acl, mat.EigenNone) {
		return nil, false
	}
	vals := eig.Values(nil)
	for _, v := range vals {
		if continuous && !(real(v) < 0) || !continuous && !(cmplx.Abs(v) < 1) {
			return nil, false
		}
	}
	return vals, true
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

	sdim, ok := stableDeflatingQZ(nn, h, j, hLeft[m*nn:], jLeft[m*nn:], stable, alphaR, alphaI, beta, vectors, &ws.work, bwork)
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

// stableDeflatingQZ writes to vectors the right Schur vectors Z of the
// generalized Schur form of the nn×nn pencil (h, j), a copy of (h0, j0),
// ordered so that the sdim eigenvalues selected by stable lead, with the
// eigenvalues in alphaR, alphaI, beta; h and j are overwritten. A reorder
// that Dgges rejects (a borderline swap, Netlib DTGEX2 parity) is retried
// on the reciprocal pencil (j, h), whose deflating subspaces are the same,
// and kept only when its eigenvalues match the first run's one to one
// (qzRetryConsistent); ok is false otherwise.
func stableDeflatingQZ(nn int, h, j, h0, j0 []float64, stable func(ar, ai, b float64) bool,
	alphaR, alphaI, beta, vectors []float64, work *[]float64, bwork []bool) (sdim int, ok bool) {
	var workQuery [1]float64
	impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, stable,
		nn, h, nn, j, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, workQuery[:], -1, bwork)
	lwork := int(workQuery[0])
	if len(*work) < lwork {
		*work = make([]float64, lwork)
	}
	sdim, ok = impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, stable,
		nn, h, nn, j, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, *work, lwork, bwork)
	if ok {
		return sdim, true
	}
	first := make([]float64, 3*nn)
	copy(first, alphaR)
	copy(first[nn:], alphaI)
	copy(first[2*nn:], beta)
	copy(h, h0)
	copy(j, j0)
	reciprocal := func(ar, ai, b float64) bool { return stable(qzReciprocal(ar, ai, b)) }
	sdim, ok = impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, reciprocal,
		nn, j, nn, h, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, *work, lwork, bwork)
	for i := range nn {
		alphaR[i], alphaI[i], beta[i] = qzReciprocal(alphaR[i], alphaI[i], beta[i])
	}
	return sdim, ok && qzRetryConsistent(first, alphaR, alphaI, beta, stable)
}
