package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

type BalredMethod int

func (m BalredMethod) validate(op string) error {
	if m != Truncate && m != SingularPerturbation {
		return fmt.Errorf("%s: unknown method %d: %w", op, m, ErrInvalidArgument)
	}
	return nil
}

const (
	Truncate BalredMethod = iota
	SingularPerturbation
)

type BalrealResult struct {
	Sys  *System
	HSV  []float64
	T    *mat.Dense
	Tinv *mat.Dense
}

// Balreal computes the balanced realization of a stable, minimal LTI system.
//
// The returned system has equal controllability and observability gramians,
// both equal to diag(σ₁, σ₂, …, σₙ) where σᵢ are the Hankel singular values.
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
		return &BalrealResult{Sys: policy.zeroOrderCopy()}, nil
	}

	stable, err := sys.IsStable()
	if err != nil {
		return nil, err
	}
	if !stable {
		return nil, ErrUnstable
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
		res := ws[wi : wi+size]
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
		return nil, ErrNotMinimal
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
		return nil, ErrNotMinimal
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

	return &BalrealResult{
		Sys:  balSys,
		HSV:  s,
		T:    T,
		Tinv: Tinv,
	}, nil
}

// Balred computes a reduced-order model via balanced truncation.
//
// If order is 0, the truncation order is automatically selected based on
// the largest relative gap in the Hankel singular values.
// method selects between Truncate (default) and SingularPerturbation (DC-gain matching).
func Balred(sys *System, order int, method BalredMethod) (*System, []float64, error) {
	if err := requireFiniteSystem("Balred", sys); err != nil {
		return nil, nil, err
	}
	if err := method.validate("Balred"); err != nil {
		return nil, nil, err
	}
	policy := newRealizationTransformPolicy(sys)
	br, err := Balreal(sys)
	if err != nil {
		return nil, nil, err
	}

	n, m, p := br.Sys.Dims()
	hsv := br.HSV

	r := order
	if r == 0 {
		r = autoSelectOrder(hsv)
	}
	if r < 1 || r > n {
		return nil, nil, ErrInvalidOrder
	}
	if r == n {
		return policy.zeroOrderCopy(), hsv, nil
	}

	Ab := br.Sys.A
	Bb := br.Sys.B
	Cb := br.Sys.C
	Db := br.Sys.D

	A11 := extractSubmatrix(Ab, 0, r, 0, r)
	B1 := extractSubmatrix(Bb, 0, r, 0, m)
	C1 := extractSubmatrix(Cb, 0, p, 0, r)

	if method == Truncate {
		red, err := policy.result(A11, B1, C1, denseCopy(Db))
		if err != nil {
			return nil, hsv, err
		}
		return red, hsv, nil
	}

	red, hsvOut, err := singularPerturbation(Ab, Bb, Cb, Db, A11, B1, C1, n, m, p, r, hsv, sys.Dt)
	if err != nil {
		return nil, hsvOut, err
	}
	propagateIONames(red, policy.sys)
	return red, hsvOut, nil
}

// Modred reduces a system by eliminating specified states.
//
// elim contains 0-based indices of states to remove.
// method selects Truncate or SingularPerturbation (DC-gain matching).
func Modred(sys *System, elim []int, method BalredMethod) (*System, error) {
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
			return nil, ErrInvalidOrder
		}
		elimSet[idx] = true
	}
	if len(elimSet) != len(elim) {
		return nil, ErrInvalidOrder
	}

	keep := make([]int, 0, n-len(elim))
	for i := range n {
		if !elimSet[i] {
			keep = append(keep, i)
		}
	}
	r := len(keep)

	if r == 0 {
		g, err := NewGain(denseCopy(sys.D), sys.Dt)
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

	red, _, err := singularPerturbation(Ap, Bp, Cp, sys.D, A11, B1, C1, n, m, p, r, nil, sys.Dt)
	if err != nil {
		return nil, err
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
	hsv []float64, dt float64,
) (*System, []float64, error) {
	A12 := extractSubmatrix(Ab, 0, r, r, n)
	A21 := extractSubmatrix(Ab, r, n, 0, r)
	A22 := extractSubmatrix(Ab, r, n, r, n)
	B2 := extractSubmatrix(Bb, r, n, 0, m)
	C2 := extractSubmatrix(Cb, 0, p, r, n)

	n2 := n - r
	slab := make([]float64, n2*n2+n2*r+n2*m+r*r+r*m+p*r+p*m)
	off := 0
	take := func(sz int) []float64 { s := slab[off : off+sz]; off += sz; return s }

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
		return nil, hsv, ErrSingularA22
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
	return red, hsv, err
}

func autoSelectOrder(hsv []float64) int {
	if len(hsv) <= 1 {
		return len(hsv)
	}
	bestIdx := 0
	bestRatio := 0.0
	for i := 0; i < len(hsv)-1; i++ {
		if hsv[i+1] > 0 {
			ratio := hsv[i] / hsv[i+1]
			if ratio > bestRatio {
				bestRatio = ratio
				bestIdx = i
			}
		} else if hsv[i] > 0 {
			bestIdx = i
			break
		}
	}
	return bestIdx + 1
}
