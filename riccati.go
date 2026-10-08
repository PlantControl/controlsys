package controlsys

import (
	"fmt"
	"math"

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
	// NoScaling disables the default power-of-two state scaling, as the
	// MATLAB icare/idare 'noscaling' option.
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
	// tiny Rcnd means X may be inaccurate. Unless opts.NoScaling is set, U11
	// is that of the state-scaled problem.
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
// diagonal scaling, so X, K and Eig do not depend on the units of x;
// opts.NoScaling disables it.
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

	// Form 2n×2n Hamiltonian: H = [[A, -G], [-Q, -A']]
	nn := 2 * n
	h := ws.h[:nn*nn]
	for i := range n {
		for j := range n {
			h[i*nn+j] = aWork[i*n+j]
			h[i*nn+n+j] = -g[i*n+j]
			h[(n+i)*nn+j] = -qWork[i*n+j]
			h[(n+i)*nn+n+j] = -aWork[j*n+i]
		}
	}

	// Schur decomposition with sorting: Re(λ) < 0 to top-left
	wr := ws.wr[:nn]
	wi := ws.wi[:nn]
	vs := ws.vs[:nn*nn]
	bwork := ws.bwork[:nn]

	selctg := func(wr, wi float64) bool { return wr < 0 }

	var workQuery [1]float64
	impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, h, nn, wr, wi, vs, nn, workQuery[:], -1, bwork)
	lwork := int(workQuery[0])
	work := ws.work
	if len(work) < lwork {
		work = make([]float64, lwork)
		ws.work = work
	}

	sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
		nn, h, nn, wr, wi, vs, nn, work, lwork, bwork)
	if !ok {
		return nil, ErrSchurFailed
	}
	if sdim != n {
		return nil, ErrNoStabilizing
	}

	X, xData, rcnd, err := problem.stabilizingSolution(vs)
	if err != nil {
		return nil, err
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
// Dare scales the states as Care does (MATLAB idare); opts.NoScaling
// disables it.
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

	subspace, err := problem.discreteStableSubspace()
	if err != nil {
		return nil, err
	}
	X, xData, rcnd, err := problem.stabilizingSolution(subspace.vectors)
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
		return nil, ErrSingularR
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

	return problem.result(X, K, eig, rcnd), nil
}

// stabilizingSolution returns X = U21*(E*U11)⁻¹ from the stable subspace
// basis [U11; U21] of a pencil with 2n columns.
func (problem riccatiProblem) stabilizingSolution(vs []float64) (X *mat.Dense, xData []float64, rcnd float64, err error) {
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
	symmetrize(xData, n, n)
	return mat.NewDense(n, n, xData), xData, rcnd, nil
}

// descriptorCare solves the generalized continuous Riccati equation; R is
// already Cholesky-factored in ws.rChol.
func (problem riccatiProblem) descriptorCare() (*RiccatiResult, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	subspace, err := problem.generalizedStableSubspace(true)
	if err != nil {
		return nil, err
	}
	X, xData, rcnd, err := problem.stabilizingSolution(subspace.vectors)
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
	return problem.result(X, mat.NewDense(m, n, kData), eig, rcnd), nil
}

type riccatiSubspace struct {
	vectors []float64
	alphaR  []float64
	alphaI  []float64
	beta    []float64
}

func (problem riccatiProblem) discreteStableSubspace() (riccatiSubspace, error) {
	if problem.E != nil {
		return problem.generalizedStableSubspace(false)
	}
	if subspace, suitable, err := problem.regularDiscreteStableSubspace(); suitable || err != nil {
		return subspace, err
	}
	return problem.generalizedStableSubspace(false)
}

// regularDiscreteStableSubspace orders the symplectic matrix built from
// A⁻ᵀ. suitable false (nil error) means A is too ill-conditioned for that
// form and the caller uses the extended pencil instead; a non-nil error is
// a failure of the suitable path.
func (problem riccatiProblem) regularDiscreteStableSubspace() (subspace riccatiSubspace, suitable bool, err error) {
	n, m, ws := problem.n, problem.m, problem.ws
	nn := 2 * n
	aRaw := problem.A.RawMatrix()
	bRaw := problem.B.RawMatrix()

	aWork := ws.aWork[:n*n]
	copyStrided(aWork, n, aRaw.Data, aRaw.Stride, n, n)
	qWork := ws.qWork[:n*n]
	qRaw := problem.Q.RawMatrix()
	copyStrided(qWork, n, qRaw.Data, qRaw.Stride, n, n)

	rinvBt := ws.rinvBt[:m*n]
	for i := range n {
		for j := range m {
			rinvBt[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
		}
	}
	impl.Dpotrs(blas.Upper, m, n, ws.rChol[:m*m], m, rinvBt, n)

	if problem.S != nil {
		rinvSt := ws.rinvSt[:m*n]
		sRaw := problem.S.RawMatrix()
		for i := range n {
			for j := range m {
				rinvSt[j*n+i] = sRaw.Data[i*sRaw.Stride+j]
			}
		}
		impl.Dpotrs(blas.Upper, m, n, ws.rChol[:m*m], m, rinvSt, n)

		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: aWork, Stride: n})
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: sRaw.Data, Stride: sRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n})
		symmetrize(qWork, n, n)
	}

	g := ws.g[:n*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: m, Cols: n, Data: rinvBt, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: g, Stride: n})
	symmetrize(g, n, n)

	scratch := ws.pencilH
	ait := scratch[:n*n]
	aitq := scratch[n*n : 2*n*n]
	gait := scratch[2*n*n : 3*n*n]
	gaitq := scratch[3*n*n : 4*n*n]
	for i := range n {
		for j := range n {
			ait[i*n+j] = aWork[j*n+i]
		}
	}
	work := ws.work
	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, ait, n, work[:n])
	ipiv := ws.ipiv[:n]
	if !impl.Dgetrf(n, n, ait, n, ipiv) {
		return riccatiSubspace{}, false, nil
	}
	rcnd := impl.Dgecon(lapack.MaxColumnSum, n, ait, n, anorm, work[:4*n], ws.iwork[:n])
	if rcnd < math.Sqrt(eps()) {
		return riccatiSubspace{}, false, nil
	}
	var inverseQuery [1]float64
	impl.Dgetri(n, ait, n, ipiv, inverseQuery[:], -1)
	lwork := int(inverseQuery[0])
	if len(ws.work) < lwork {
		ws.work = make([]float64, lwork)
	}
	impl.Dgetri(n, ait, n, ipiv, ws.work, lwork)

	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: ait, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: aitq, Stride: n})
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: g, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: ait, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: gait, Stride: n})
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: gait, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: gaitq, Stride: n})

	z := ws.z[:nn*nn]
	for i := range n {
		for j := range n {
			z[i*nn+j] = aWork[i*n+j] + gaitq[i*n+j]
			z[i*nn+n+j] = -gait[i*n+j]
			z[(n+i)*nn+j] = -aitq[i*n+j]
			z[(n+i)*nn+n+j] = ait[i*n+j]
		}
	}

	alphaR := ws.wr[:nn]
	alphaI := ws.wi[:nn]
	beta := ws.beta[:nn]
	vectors := ws.vs[:nn*nn]
	bwork := ws.bwork[:nn]
	insideUnitCircle := func(real, imag float64) bool {
		return math.Hypot(real, imag) < 1
	}

	var workQuery [1]float64
	impl.Dgees(lapack.SchurHess, lapack.SortSelected, insideUnitCircle,
		nn, z, nn, alphaR, alphaI, vectors, nn, workQuery[:], -1, bwork)
	lwork = int(workQuery[0])
	if len(ws.work) < lwork {
		ws.work = make([]float64, lwork)
	}
	sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, insideUnitCircle,
		nn, z, nn, alphaR, alphaI, vectors, nn, ws.work, lwork, bwork)
	if !ok {
		return riccatiSubspace{}, true, ErrSchurFailed
	}
	if sdim != n {
		return riccatiSubspace{}, true, ErrNoStabilizing
	}
	for i := range nn {
		beta[i] = 1
	}
	return riccatiSubspace{
		vectors: vectors,
		alphaR:  alphaR,
		alphaI:  alphaI,
		beta:    beta,
	}, true, nil
}

// generalizedStableSubspace deflates the extended pencil
//
//	continuous: [A 0 B; -Q -A' -S; S' B' R] - λ[E 0 0; 0 E' 0; 0 0 0]
//	discrete:   [A 0 B; -Q E' -S; S' 0 R]   - λ[E 0 0; 0 A' 0; 0 -B' 0]
//
// to 2n×2n by a QR of its input columns and orders the stable eigenvalues
// first. E nil means I.
func (problem riccatiProblem) generalizedStableSubspace(continuous bool) (riccatiSubspace, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	nn := 2 * n
	rows := nn + m

	hLeft := ws.pencilH[:rows*nn]
	jLeft := ws.pencilJ[:rows*nn]
	input := ws.pencilInput[:rows*m]
	clear(hLeft)
	clear(jLeft)
	clear(input)

	aRaw := problem.A.RawMatrix()
	bRaw := problem.B.RawMatrix()
	qRaw := problem.Q.RawMatrix()
	rRaw := problem.R.RawMatrix()
	var sRaw blas64.General
	if problem.S != nil {
		sRaw = problem.S.RawMatrix()
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
			hLeft[(n+i)*nn+j] = -qRaw.Data[i*qRaw.Stride+j]
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
			input[i*m+j] = bRaw.Data[i*bRaw.Stride+j]
			if problem.S != nil {
				input[(n+i)*m+j] = -sRaw.Data[i*sRaw.Stride+j]
				hLeft[(nn+j)*nn+i] = sRaw.Data[i*sRaw.Stride+j]
			}
			if continuous {
				hLeft[(nn+j)*nn+n+i] = bRaw.Data[i*bRaw.Stride+j]
			} else {
				jLeft[(nn+j)*nn+n+i] = -bRaw.Data[i*bRaw.Stride+j]
			}
		}
	}
	for i := range m {
		for j := range m {
			input[(nn+i)*m+j] = rRaw.Data[i*rRaw.Stride+j]
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
