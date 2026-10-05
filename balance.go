package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// StateProjection selects how Balred and Modred eliminate states, as the
// MATLAB balredOptions StateProjection option and the modred method argument.
// The zero value is MatchDC, the MATLAB default for both.
type StateProjection int

const (
	// MatchDC residualizes the eliminated states (singular perturbation) so
	// the reduced model keeps the DC gain: their derivatives (continuous) or
	// increments (discrete) are set to zero.
	MatchDC StateProjection = iota
	// Truncate discards the eliminated states without altering the others.
	Truncate
)

func (m StateProjection) validate(op string) error {
	if m != MatchDC && m != Truncate {
		return fmt.Errorf("%s: unknown state projection %d: %w", op, m, ErrInvalidArgument)
	}
	return nil
}

// BalredOptions mirrors MATLAB balredOptions. The zero value selects MatchDC.
type BalredOptions struct {
	StateProjection StateProjection
}

// BalrealResult holds the outputs of MATLAB [sysb,g,TL,TR] = balreal(sys):
// Sys.A = TL·A·TR, Sys.B = TL·B, Sys.C = C·TR, TL·TR = I, so the balanced
// state is x̄ = TL·x and x = TR·x̄. HSV holds the Hankel singular values in
// decreasing order.
type BalrealResult struct {
	Sys *System
	HSV []float64
	TL  *mat.Dense
	TR  *mat.Dense
}

// Balreal computes the balanced realization of a stable, minimal LTI system,
// as MATLAB balreal; see
// https://www.mathworks.com/help/control/ref/dynamicsystem.balreal.html.
// The returned system has equal controllability and observability gramians,
// both equal to diag(σ₁, σ₂, …, σₙ) where σᵢ are the Hankel singular values.
//
// Unlike MATLAB, which balances the stable part and reports Inf for unstable
// modes, an unstable model returns ErrUnstable; a non-minimal one returns
// ErrNotMinimal. A model with no states returns ErrDimensionMismatch, since
// TL and TR cannot be formed.
func Balreal(sys *System) (*BalrealResult, error) {
	if err := requireFiniteSystem("Balreal", sys); err != nil {
		return nil, err
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Balreal"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Balreal"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 {
		return nil, fmt.Errorf("Balreal: system has no states: %w", ErrDimensionMismatch)
	}

	stable, err := sys.IsStable()
	if err != nil {
		return nil, fmt.Errorf("Balreal: %w", err)
	}
	if !stable {
		return nil, fmt.Errorf("Balreal: %w", ErrUnstable)
	}
	if m == 0 || p == 0 {
		return nil, fmt.Errorf("Balreal: model has no inputs or no outputs, so all Hankel singular values are zero: %w", ErrNotMinimal)
	}

	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()

	bufSize := 14*n*n + n*m + p*n + n
	ws := make([]float64, bufSize)
	wi := 0

	alloc := func(size int) []float64 {
		res := ws[wi : wi+size : wi+size]
		wi += size
		return res
	}

	aData := alloc(n * n)
	at := alloc(n * n)
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)
	for i := range n {
		for j := range n {
			at[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}

	bbt := alloc(n * n)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: bbt})
	symmetrize(bbt, n, n)

	ctc := alloc(n * n)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ctc})
	symmetrize(ctc, n, n)

	A := mat.NewDense(n, n, aData)
	At := mat.NewDense(n, n, at)

	var Wc, Wo *mat.Dense
	if sys.IsContinuous() {
		Wc, err = Lyap(A, mat.NewDense(n, n, bbt), nil)
		if err != nil {
			return nil, err
		}
		Wo, err = Lyap(At, mat.NewDense(n, n, ctc), nil)
	} else {
		Wc, err = DLyap(A, mat.NewDense(n, n, bbt), nil)
		if err != nil {
			return nil, err
		}
		Wo, err = DLyap(At, mat.NewDense(n, n, ctc), nil)
	}
	if err != nil {
		return nil, err
	}

	wcRaw := Wc.RawMatrix()
	lc := alloc(n * n)
	copyStrided(lc, n, wcRaw.Data, wcRaw.Stride, n, n)
	if !impl.Dpotrf(blas.Lower, n, lc, n) {
		return nil, fmt.Errorf("Balreal: controllability gramian: %w", ErrNotMinimal)
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lc[i*n+j] = 0
		}
	}

	woRaw := Wo.RawMatrix()
	lo := alloc(n * n)
	copyStrided(lo, n, woRaw.Data, woRaw.Stride, n, n)
	if !impl.Dpotrf(blas.Lower, n, lo, n) {
		return nil, fmt.Errorf("Balreal: observability gramian: %w", ErrNotMinimal)
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lo[i*n+j] = 0
		}
	}

	mData := alloc(n * n)
	lcGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: lc}
	loGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: lo}
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, loGen, lcGen,
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: mData})

	s := alloc(n)
	uData := alloc(n * n)
	vtData := alloc(n * n)
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDAll, lapack.SVDAll, n, n, mData, n, s,
		uData, n, vtData, n, wq, -1)
	svdWork := make([]float64, int(wq[0]))
	if !impl.Dgesvd(lapack.SVDAll, lapack.SVDAll, n, n, mData, n, s,
		uData, n, vtData, n, svdWork, len(svdWork)) {
		return nil, fmt.Errorf("Balreal: SVD of gramian factors did not converge: %w", ErrSchurFailed)
	}

	// V = Vt' — scale columns of Vt' by 1/sqrt(σ) → vScaled
	// T = Lc * vScaled
	vScaled := alloc(n * n)
	for i := range n {
		invSqrt := 1.0 / math.Sqrt(s[i])
		for j := range n {
			vScaled[j*n+i] = vtData[i*n+j] * invSqrt
		}
	}

	tData := alloc(n * n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, lcGen,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: vScaled},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tData})
	T := mat.NewDense(n, n, tData)

	// Tinv = Σ^{-1/2} * U' * Lo'
	// Scale columns of U by 1/sqrt(σ) → uScaled, then Tinv = uScaled' * Lo'
	for j := range n {
		invSqrt := 1.0 / math.Sqrt(s[j])
		for i := range n {
			uData[i*n+j] *= invSqrt
		}
	}

	tinvData := alloc(n * n)
	blas64.Gemm(blas.Trans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: uData},
		loGen,
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tinvData})
	Tinv := mat.NewDense(n, n, tinvData)

	tGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: tData}
	tiGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: tinvData}

	// Ab = Tinv * A * T
	tmp := alloc(n * n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tiGen,
		blas64.General{Rows: n, Cols: n, Stride: aRaw.Stride, Data: aRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tmp})
	abData := alloc(n * n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: tmp},
		tGen,
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: abData})

	// Bb = Tinv * B
	bbData := alloc(n * m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tiGen,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bbData})

	// Cb = C * T
	cbData := alloc(p * n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		tGen,
		0, blas64.General{Rows: p, Cols: n, Stride: n, Data: cbData})

	Ab := mat.NewDense(n, n, abData)
	Bb := mat.NewDense(n, m, bbData)
	Cb := mat.NewDense(p, n, cbData)
	Db := denseCopy(sys.D)

	balSys, err := policy.result(Ab, Bb, Cb, Db)
	if err != nil {
		return nil, err
	}

	return &BalrealResult{Sys: balSys, HSV: s, TL: Tinv, TR: T}, nil
}

// Balred reduces sys to order states by balanced truncation, as MATLAB
// [rsys,info] = balred(sys,order,opts); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.balred.html.
// hsv holds the Hankel singular values (MATLAB info.HSV). order must be in
// [0, n]; order n returns the balanced realization and order 0 a static gain.
// opts.StateProjection selects MatchDC (the zero value, MATLAB default) or
// Truncate. sys must satisfy the requirements of Balreal.
func Balred(sys *System, order int, opts BalredOptions) (*System, []float64, error) {
	if err := requireFiniteSystem("Balred", sys); err != nil {
		return nil, nil, err
	}
	if err := opts.StateProjection.validate("Balred"); err != nil {
		return nil, nil, err
	}
	br, err := Balreal(sys)
	if err != nil {
		return nil, nil, err
	}
	n, _, _ := br.Sys.Dims()
	if order < 0 || order > n {
		return nil, nil, fmt.Errorf("Balred: order %d outside [0,%d]: %w", order, n, ErrInvalidOrder)
	}
	if order == n {
		return br.Sys, br.HSV, nil
	}
	elim := make([]int, 0, n-order)
	for i := order; i < n; i++ {
		elim = append(elim, i)
	}
	red, err := Modred(br.Sys, elim, opts.StateProjection)
	if err != nil {
		return nil, nil, fmt.Errorf("Balred: %w", err)
	}
	propagateIONames(red, sys)
	return red, br.HSV, nil
}

// Modred eliminates the states with 0-based indices elim, as MATLAB
// modred(sys,elim,method) with 1-based indices; see
// https://www.mathworks.com/help/control/ref/ss.modred.html. method MatchDC
// (the zero value, MATLAB default) residualizes them and keeps the DC gain;
// Truncate discards them. Indices out of range or repeated return
// ErrInvalidArgument; an empty elim returns a copy of sys, and eliminating
// every state returns a static gain. MatchDC returns ErrSingularA22 when the
// eliminated block (A22, or A22 - I for discrete models) is singular.
func Modred(sys *System, elim []int, method StateProjection) (*System, error) {
	if err := requireSystem("Modred", sys); err != nil {
		return nil, err
	}
	if err := method.validate("Modred"); err != nil {
		return nil, err
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Modred"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Modred"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 || len(elim) == 0 {
		return policy.zeroOrderCopy(), nil
	}
	if m == 0 || p == 0 {
		return withZeroIOPadding("Modred", sys, func(padded *System) (*System, error) {
			return Modred(padded, elim, method)
		})
	}

	elimSet := make(map[int]bool, len(elim))
	for _, idx := range elim {
		if idx < 0 || idx >= n {
			return nil, fmt.Errorf("Modred: state index %d outside [0,%d): %w", idx, n, ErrInvalidArgument)
		}
		if elimSet[idx] {
			return nil, fmt.Errorf("Modred: duplicate state index %d: %w", idx, ErrInvalidArgument)
		}
		elimSet[idx] = true
	}

	keep := make([]int, 0, n-len(elim))
	for i := range n {
		if !elimSet[i] {
			keep = append(keep, i)
		}
	}
	r := len(keep)

	if r == 0 {
		d := denseCopy(sys.D)
		if method == MatchDC {
			var err error
			if d, err = residualizedGain(sys); err != nil {
				return nil, err
			}
		}
		g, err := NewGain(d, sys.Dt)
		if err != nil {
			return nil, err
		}
		propagateIONames(g, sys)
		return g, nil
	}
	if r == n {
		return sys.Copy(), nil
	}

	perm := make([]int, n)
	copy(perm, keep)
	ei := 0
	for i := range n {
		if elimSet[i] {
			perm[r+ei] = i
			ei++
		}
	}

	aRaw := sys.A.RawMatrix()
	ap := make([]float64, n*n)
	for i := range n {
		srcRow := aRaw.Data[perm[i]*aRaw.Stride:]
		dstOff := i * n
		for j := range n {
			ap[dstOff+j] = srcRow[perm[j]]
		}
	}

	bRaw := sys.B.RawMatrix()
	bp := make([]float64, n*m)
	for i := range n {
		copy(bp[i*m:i*m+m], bRaw.Data[perm[i]*bRaw.Stride:perm[i]*bRaw.Stride+m])
	}

	cRaw := sys.C.RawMatrix()
	cp := make([]float64, p*n)
	for i := range p {
		srcRow := cRaw.Data[i*cRaw.Stride:]
		dstOff := i * n
		for j := range n {
			cp[dstOff+j] = srcRow[perm[j]]
		}
	}

	Ap := mat.NewDense(n, n, ap)
	Bp := mat.NewDense(n, m, bp)
	Cp := mat.NewDense(p, n, cp)

	A11 := extractSubmatrix(Ap, 0, r, 0, r)
	B1 := extractSubmatrix(Bp, 0, r, 0, m)
	C1 := extractSubmatrix(Cp, 0, p, 0, r)

	if method == Truncate {
		red, err := policy.result(A11, B1, C1, denseCopy(sys.D))
		if err != nil {
			return nil, err
		}
		return red, nil
	}

	red, err := singularPerturbation(Ap, Bp, Cp, sys.D, A11, B1, C1, n, m, p, r, sys.Dt)
	if err != nil {
		return nil, fmt.Errorf("Modred: %w", err)
	}
	propagateIONames(red, sys)
	return red, nil
}

// singularPerturbation computes the reduced system via residualization.
// Ar = A11 - A12*inv(M)*A21, etc., with M = A22 (continuous) or A22-I
// (discrete, x2[k+1] = x2[k]). Uses Gemm beta=-1 to fuse multiply-subtract.
func singularPerturbation(
	Ab, Bb, Cb, Db *mat.Dense,
	A11, B1, C1 *mat.Dense,
	n, m, p, r int,
	dt float64,
) (*System, error) {
	A12 := extractSubmatrix(Ab, 0, r, r, n)
	A21 := extractSubmatrix(Ab, r, n, 0, r)
	A22 := extractSubmatrix(Ab, r, n, r, n)
	B2 := extractSubmatrix(Bb, r, n, 0, m)
	C2 := extractSubmatrix(Cb, 0, p, r, n)

	n2 := n - r
	slab := make([]float64, n2*n2+n2*r+n2*m+r*r+r*m+p*r+p*m)
	off := 0
	take := func(sz int) []float64 { s := slab[off : off+sz : off+sz]; off += sz; return s }

	a22Data := take(n2 * n2)
	a22Raw := A22.RawMatrix()
	copyStrided(a22Data, n2, a22Raw.Data, a22Raw.Stride, n2, n2)

	if dt > 0 {
		for i := range n2 {
			a22Data[i*n2+i]--
		}
	}

	ipiv := make([]int, n2)
	if !impl.Dgetrf(n2, n2, a22Data, n2, ipiv) {
		return nil, ErrSingularA22
	}

	rhs1 := take(n2 * r)
	a21Raw := A21.RawMatrix()
	copyStrided(rhs1, r, a21Raw.Data, a21Raw.Stride, n2, r)
	impl.Dgetrs(blas.NoTrans, n2, r, a22Data, n2, ipiv, rhs1, r)

	rhs2 := take(n2 * m)
	b2Raw := B2.RawMatrix()
	copyStrided(rhs2, m, b2Raw.Data, b2Raw.Stride, n2, m)
	impl.Dgetrs(blas.NoTrans, n2, m, a22Data, n2, ipiv, rhs2, m)

	invA22_A21 := blas64.General{Rows: n2, Cols: r, Stride: r, Data: rhs1}
	invA22_B2 := blas64.General{Rows: n2, Cols: m, Stride: m, Data: rhs2}
	a12Raw := A12.RawMatrix()
	a12Gen := blas64.General{Rows: r, Cols: n2, Stride: a12Raw.Stride, Data: a12Raw.Data}
	c2Raw := C2.RawMatrix()
	c2Gen := blas64.General{Rows: p, Cols: n2, Stride: c2Raw.Stride, Data: c2Raw.Data}

	arData := take(r * r)
	a11Raw := A11.RawMatrix()
	copyStrided(arData, r, a11Raw.Data, a11Raw.Stride, r, r)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, a12Gen, invA22_A21,
		1, blas64.General{Rows: r, Cols: r, Stride: r, Data: arData})

	brData := take(r * m)
	b1Raw := B1.RawMatrix()
	copyStrided(brData, m, b1Raw.Data, b1Raw.Stride, r, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, a12Gen, invA22_B2,
		1, blas64.General{Rows: r, Cols: m, Stride: m, Data: brData})

	crData := take(p * r)
	c1Raw := C1.RawMatrix()
	copyStrided(crData, r, c1Raw.Data, c1Raw.Stride, p, r)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, c2Gen, invA22_A21,
		1, blas64.General{Rows: p, Cols: r, Stride: r, Data: crData})

	drData := take(p * m)
	dRaw := Db.RawMatrix()
	copyStrided(drData, m, dRaw.Data, dRaw.Stride, p, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, c2Gen, invA22_B2,
		1, blas64.General{Rows: p, Cols: m, Stride: m, Data: drData})

	red, err := newNoCopy(
		mat.NewDense(r, r, arData),
		mat.NewDense(r, m, brData),
		mat.NewDense(p, r, crData),
		mat.NewDense(p, m, drData),
		dt,
	)
	return red, err
}

// residualizedGain returns D - C·M⁻¹·B with M = A (continuous) or A - I
// (discrete): the static gain left when every state is residualized.
func residualizedGain(sys *System) (*mat.Dense, error) {
	n, m, p := sys.Dims()
	M := mat.DenseCopyOf(sys.A)
	if sys.IsDiscrete() {
		for i := range n {
			M.Set(i, i, M.At(i, i)-1)
		}
	}
	var lu mat.LU
	lu.Factorize(M)
	if lu.Det() == 0 {
		return nil, fmt.Errorf("Modred: %w", ErrSingularA22)
	}
	var x mat.Dense
	if err := lu.SolveTo(&x, false, sys.B); err != nil {
		return nil, fmt.Errorf("Modred: %w", ErrSingularA22)
	}
	g := mat.NewDense(p, m, nil)
	g.Mul(sys.C, &x)
	g.Sub(sys.D, g)
	return g, nil
}
