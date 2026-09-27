package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

type RiccatiWorkspace struct {
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
}

func NewRiccatiWorkspace(n, m int) *RiccatiWorkspace {
	nn := 2 * n
	return &RiccatiWorkspace{
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
}

type RiccatiOpts struct {
	// S is the optional cross-term matrix. It is read during the call.
	S *mat.Dense
	// Workspace supplies reusable scratch storage. Results may share its storage
	// until the next call that reuses the same workspace.
	Workspace *RiccatiWorkspace
}

type RiccatiResult struct {
	// X, K, and Eig are caller-owned unless a workspace was supplied.
	X    *mat.Dense
	K    *mat.Dense
	Eig  []complex128
	Rcnd float64
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
func Care(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	problem, err := newRiccatiProblem(A, B, Q, R, opts)
	if err != nil {
		return nil, err
	}
	n, m := problem.n, problem.m
	if n == 0 {
		return &RiccatiResult{X: &mat.Dense{}, K: &mat.Dense{}, Eig: nil}, nil
	}
	S, ws := problem.S, problem.ws

	// Cholesky factor R
	rChol := ws.rChol[:m*m]
	rRaw := R.RawMatrix()
	copyStrided(rChol, m, rRaw.Data, rRaw.Stride, m, m)
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
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

	// Extract U11 = vs[0:n, 0:n], U21 = vs[n:2n, 0:n]
	u11 := ws.u11[:n*n]
	u21 := ws.u21[:n*n]
	copyStrided(u11, n, vs, nn, n, n)
	copyBlock(u21, n, 0, 0, vs, nn, n, 0, n, n)

	// X = U21 * U11⁻¹; since X is symmetric: X = (U11')⁻¹ * U21'
	// Solve via DGETRS(Trans) instead of explicit inverse
	ipiv := ws.ipiv[:n]
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, ErrNoStabilizing
	}

	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, u11, n, work[:n])
	iwork := ws.iwork[:n]
	rcnd := impl.Dgecon(lapack.MaxColumnSum, n, u11, n, anorm, work[:4*n], iwork)

	xData := ws.xData[:n*n]
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)

	symmetrize(xData, n, n)
	X := mat.NewDense(n, n, xData)

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

	return &RiccatiResult{X: X, K: K, Eig: eig, Rcnd: rcnd}, nil
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
func Dare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	problem, err := newRiccatiProblem(A, B, Q, R, opts)
	if err != nil {
		return nil, err
	}
	n, m := problem.n, problem.m
	if n == 0 {
		return &RiccatiResult{X: &mat.Dense{}, K: &mat.Dense{}, Eig: nil}, nil
	}
	S, ws := problem.S, problem.ws

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
	vs := subspace.vectors

	// Extract U11 = vs[0:n, 0:n], U21 = vs[n:2n, 0:n]
	u11 := ws.u11[:n*n]
	u21 := ws.u21[:n*n]
	copyStrided(u11, n, vs, 2*n, n, n)
	copyBlock(u21, n, 0, 0, vs, 2*n, n, 0, n, n)

	work := ws.work
	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, u11, n, work[:n])
	ipiv := ws.ipiv[:n]
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, ErrNoStabilizing
	}

	iwork2 := ws.iwork[:n]
	rcnd := impl.Dgecon(lapack.MaxColumnSum, n, u11, n, anorm, work[:4*n], iwork2)

	xData := ws.xData[:n*n]
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)

	symmetrize(xData, n, n)
	X := mat.NewDense(n, n, xData)

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

	return &RiccatiResult{X: X, K: K, Eig: eig, Rcnd: rcnd}, nil
}

type discreteRiccatiSubspace struct {
	vectors []float64
	alphaR  []float64
	alphaI  []float64
	beta    []float64
}

func (problem riccatiProblem) discreteStableSubspace() (discreteRiccatiSubspace, error) {
	if subspace, suitable, err := problem.regularDiscreteStableSubspace(); suitable || err != nil {
		return subspace, err
	}
	return problem.generalizedDiscreteStableSubspace()
}

func (problem riccatiProblem) regularDiscreteStableSubspace() (subspace discreteRiccatiSubspace, suitable bool, err error) {
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
		return discreteRiccatiSubspace{}, false, nil
	}
	rcnd := impl.Dgecon(lapack.MaxColumnSum, n, ait, n, anorm, work[:4*n], ws.iwork[:n])
	if rcnd < math.Sqrt(eps()) {
		return discreteRiccatiSubspace{}, false, nil
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
		return discreteRiccatiSubspace{}, true, ErrSchurFailed
	}
	if sdim != n {
		return discreteRiccatiSubspace{}, true, ErrNoStabilizing
	}
	for i := range nn {
		beta[i] = 1
	}
	return discreteRiccatiSubspace{
		vectors: vectors,
		alphaR:  alphaR,
		alphaI:  alphaI,
		beta:    beta,
	}, true, nil
}

func (problem riccatiProblem) generalizedDiscreteStableSubspace() (discreteRiccatiSubspace, error) {
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

	for i := range n {
		for j := range n {
			hLeft[i*nn+j] = aRaw.Data[i*aRaw.Stride+j]
			hLeft[(n+i)*nn+j] = -qRaw.Data[i*qRaw.Stride+j]
			jLeft[(n+i)*nn+n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
		hLeft[(n+i)*nn+n+i] = 1
		jLeft[i*nn+i] = 1

		for j := range m {
			input[i*m+j] = bRaw.Data[i*bRaw.Stride+j]
			if problem.S != nil {
				input[(n+i)*m+j] = -sRaw.Data[i*sRaw.Stride+j]
				hLeft[(nn+j)*nn+i] = sRaw.Data[i*sRaw.Stride+j]
			}
			jLeft[(nn+j)*nn+n+i] = -bRaw.Data[i*bRaw.Stride+j]
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
	insideUnitCircle := func(alphaR, alphaI, beta float64) bool {
		return math.Hypot(alphaR, alphaI) < math.Abs(beta)
	}

	var workQuery [1]float64
	impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, insideUnitCircle,
		nn, h, nn, j, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, workQuery[:], -1, bwork)
	lwork := int(workQuery[0])
	if len(ws.work) < lwork {
		ws.work = make([]float64, lwork)
	}

	sdim, ok := impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, insideUnitCircle,
		nn, h, nn, j, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, ws.work, lwork, bwork)
	if !ok {
		return discreteRiccatiSubspace{}, ErrSchurFailed
	}
	if sdim != n {
		return discreteRiccatiSubspace{}, ErrNoStabilizing
	}
	for i := range n {
		if beta[i] == 0 {
			return discreteRiccatiSubspace{}, ErrNoStabilizing
		}
	}

	return discreteRiccatiSubspace{
		vectors: vectors,
		alphaR:  alphaR,
		alphaI:  alphaI,
		beta:    beta,
	}, nil
}
