package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// Lqr solves the continuous-time linear-quadratic regulator problem.
// It computes the optimal gain K such that u = -Kx minimizes
// J = integral(x'Qx + u'Ru) for the system dx/dt = Ax + Bu.
//
// A is n×n, B is n×m, Q is n×n symmetric PSD, R is m×m symmetric PD.
// Returns gain K (m×n), Riccati solution X, and closed-loop eigenvalues.
func Lqr(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	return Care(A, B, Q, R, opts)
}

// Dlqr solves the discrete-time linear-quadratic regulator problem.
// It computes the optimal gain K such that u[k] = -Kx[k] minimizes
// J = sum(x'Qx + u'Ru) for the system x[k+1] = Ax[k] + Bu[k].
//
// A is n×n, B is n×m, Q is n×n symmetric PSD, R is m×m symmetric PD.
// Returns gain K (m×n), Riccati solution X, and closed-loop eigenvalues.
func Dlqr(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	return Dare(A, B, Q, R, opts)
}

// Lqi computes the linear-quadratic regulator with integral action.
// The system is augmented with integral states: xi = -C*x, giving
// Aaug = [A, 0; -C, 0] and Baug = [B; 0].
//
// A is n×n, B is n×m, C is p×n. Q is (n+p)×(n+p), R is m×m.
// Returns gain K of size m×(n+p) where K = [Kx, Ki].
func Lqi(A, B, C, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	na, nac := A.Dims()
	if na != nac {
		return nil, ErrDimensionMismatch
	}
	nb, m := B.Dims()
	if nb != na {
		return nil, ErrDimensionMismatch
	}
	p, cn := C.Dims()
	if cn != na {
		return nil, ErrDimensionMismatch
	}
	n := na
	aug := n + p

	aaugData := make([]float64, aug*aug)
	aRaw := A.RawMatrix()
	cRaw := C.RawMatrix()
	copyStrided(aaugData, aug, aRaw.Data, aRaw.Stride, n, n)
	for i := range p {
		for j := range n {
			aaugData[(n+i)*aug+j] = -cRaw.Data[i*cRaw.Stride+j]
		}
	}
	Aaug := mat.NewDense(aug, aug, aaugData)

	baugData := make([]float64, aug*m)
	bRaw := B.RawMatrix()
	copyStrided(baugData, m, bRaw.Data, bRaw.Stride, n, m)
	Baug := mat.NewDense(aug, m, baugData)

	return Lqr(Aaug, Baug, Q, R, opts)
}

// Lqrd computes the discrete-time LQR gain from a continuous-time plant,
// matching MATLAB lqrd. It discretizes (A, B) using zero-order hold with
// sample time dt and discretizes the continuous cost
//
//	J = integral(x'Qx + u'Ru + 2x'Nu)
//
// for piecewise-constant u into the equivalent sampled-data weights
//
//	[Qd Nd; Nd' Rd] = integral_0^dt [Φ(τ) Γ(τ); 0 I]' [Q N; N' R] [Φ(τ) Γ(τ); 0 I] dτ
//
// (Van Loan's method), then solves the discrete LQR problem with cross term Nd.
//
// A is n×n, B is n×m, Q is n×n, R is m×m, dt > 0. opts.S, when set, is the
// continuous cross weight N (n×m); opts.Workspace is used for the discrete
// Riccati solve.
func Lqrd(A, B, Q, R *mat.Dense, dt float64, opts *RiccatiOpts) (*RiccatiResult, error) {
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}
	na, nac := A.Dims()
	if na != nac {
		return nil, ErrDimensionMismatch
	}
	nb, m := B.Dims()
	if nb != na {
		return nil, ErrDimensionMismatch
	}
	n := na
	if qr, qc := Q.Dims(); qr != n || qc != n {
		return nil, ErrDimensionMismatch
	}
	if rr, rc := R.Dims(); rr != m || rc != m {
		return nil, ErrDimensionMismatch
	}
	var N *mat.Dense
	if opts != nil && opts.S != nil {
		N = opts.S
		if sr, sc := N.Dims(); sr != n || sc != m {
			return nil, ErrDimensionMismatch
		}
	}

	if n == 0 {
		return Dlqr(A, B, Q, R, opts)
	}

	nm := n + m
	h := 2 * nm
	H := mat.NewDense(h, h, nil)
	hRaw := H.RawMatrix()
	aRaw := A.RawMatrix()
	bRaw := B.RawMatrix()
	qRaw := Q.RawMatrix()
	rRaw := R.RawMatrix()
	for i := range n {
		for j := range n {
			hRaw.Data[j*hRaw.Stride+i] = -aRaw.Data[i*aRaw.Stride+j] * dt
			hRaw.Data[(nm+i)*hRaw.Stride+nm+j] = aRaw.Data[i*aRaw.Stride+j] * dt
			hRaw.Data[i*hRaw.Stride+nm+j] = qRaw.Data[i*qRaw.Stride+j] * dt
		}
		for j := range m {
			hRaw.Data[(n+j)*hRaw.Stride+i] = -bRaw.Data[i*bRaw.Stride+j] * dt
			hRaw.Data[(nm+i)*hRaw.Stride+nm+n+j] = bRaw.Data[i*bRaw.Stride+j] * dt
		}
	}
	for i := range m {
		for j := range m {
			hRaw.Data[(n+i)*hRaw.Stride+nm+n+j] = rRaw.Data[i*rRaw.Stride+j] * dt
		}
	}
	if N != nil {
		sRaw := N.RawMatrix()
		for i := range n {
			for j := range m {
				v := sRaw.Data[i*sRaw.Stride+j] * dt
				hRaw.Data[i*hRaw.Stride+nm+n+j] = v
				hRaw.Data[(n+j)*hRaw.Stride+nm+i] = v
			}
		}
	}

	var eH mat.Dense
	eH.Exp(H)
	ehRaw := eH.RawMatrix()

	phiData := make([]float64, nm*nm)
	f12Data := make([]float64, nm*nm)
	copyBlock(phiData, nm, 0, 0, ehRaw.Data, ehRaw.Stride, nm, nm, nm, nm)
	copyBlock(f12Data, nm, 0, 0, ehRaw.Data, ehRaw.Stride, 0, nm, nm, nm)
	Phi := mat.NewDense(nm, nm, phiData)
	W := mat.NewDense(nm, nm, nil)
	W.Mul(Phi.T(), mat.NewDense(nm, nm, f12Data))
	wRaw := W.RawMatrix()
	symmetrize(wRaw.Data, nm, wRaw.Stride)

	adData := make([]float64, n*n)
	bdData := make([]float64, n*m)
	qdData := make([]float64, n*n)
	ndData := make([]float64, n*m)
	rdData := make([]float64, m*m)
	copyBlock(adData, n, 0, 0, phiData, nm, 0, 0, n, n)
	copyBlock(bdData, m, 0, 0, phiData, nm, 0, n, n, m)
	copyBlock(qdData, n, 0, 0, wRaw.Data, wRaw.Stride, 0, 0, n, n)
	copyBlock(ndData, m, 0, 0, wRaw.Data, wRaw.Stride, 0, n, n, m)
	copyBlock(rdData, m, 0, 0, wRaw.Data, wRaw.Stride, n, n, m, m)

	dopts := &RiccatiOpts{S: mat.NewDense(n, m, ndData)}
	if opts != nil {
		dopts.Workspace = opts.Workspace
	}
	return Dlqr(mat.NewDense(n, n, adData), mat.NewDense(n, m, bdData), mat.NewDense(n, n, qdData), mat.NewDense(m, m, rdData), dopts)
}

// Acker computes SISO pole placement using Ackermann's formula.
// Given single-input system (A, B) and desired closed-loop poles,
// returns gain K (1×n) such that eig(A - B*K) = poles.
//
// Only valid for single-input systems (m=1). Numerically fragile for n > 10.
func Acker(A, B *mat.Dense, poles []complex128) (*mat.Dense, error) {
	na, nac := A.Dims()
	if na != nac {
		return nil, ErrDimensionMismatch
	}
	nb, m := B.Dims()
	if nb != na {
		return nil, ErrDimensionMismatch
	}
	if m != 1 {
		return nil, ErrNotSISO
	}
	n := na
	if n == 0 {
		return mat.NewDense(1, 0, nil), nil
	}
	if len(poles) != n {
		return nil, ErrPoleCount
	}
	if err := validatePoles(poles); err != nil {
		return nil, err
	}

	p := polyFromComplexRoots(sortConjugatePairs(poles))

	Cm, err := Ctrb(A, B)
	if err != nil {
		return nil, err
	}

	var lu mat.LU
	lu.Factorize(Cm)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("Acker: %w", ErrUncontrollable)
	}

	paData := make([]float64, n*n)
	for i := range n {
		paData[i*n+i] = p[0]
	}
	tmpData := make([]float64, n*n)
	aRaw := A.RawMatrix()
	aGen := blas64.General{Rows: n, Cols: n, Data: aRaw.Data, Stride: aRaw.Stride}
	for k := 1; k <= n; k++ {
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			1, blas64.General{Rows: n, Cols: n, Data: paData, Stride: n}, aGen,
			0, blas64.General{Rows: n, Cols: n, Data: tmpData, Stride: n})
		copy(paData, tmpData)
		for i := range n {
			paData[i*n+i] += p[k]
		}
	}
	pA := mat.NewDense(n, n, paData)

	// K = last_row(Cm^{-1}) * p(A)
	// Solve Cm' * y = e_n for y, then K = y' * p(A)
	en := mat.NewVecDense(n, nil)
	en.SetVec(n-1, 1)
	var y mat.VecDense
	if err := lu.SolveVecTo(&y, true, en); err != nil {
		return nil, fmt.Errorf("Acker: %w", ErrUncontrollable)
	}

	kData := make([]float64, n)
	yData := y.RawVector()
	pARaw := pA.RawMatrix()
	for j := range n {
		var s float64
		for i := range n {
			s += yData.Data[i*yData.Inc] * pARaw.Data[i*pARaw.Stride+j]
		}
		kData[j] = s
	}

	return mat.NewDense(1, n, kData), nil
}

// Place computes state feedback gain F (m×n) via Schur-based pole placement
// such that eig(A - B*F) equals the desired poles.
//
// Uses Varga's method: Schur decomposition with iterative minimum-norm
// eigenvalue assignment. Works for both SISO and MIMO systems.
//
// Poles must come in conjugate pairs. len(poles) must equal n. Unlike
// MATLAB place, pole multiplicity may exceed rank(B); the closed loop is then
// defective and its repeated eigenvalues are correspondingly sensitive.
func Place(A, B *mat.Dense, poles []complex128) (*mat.Dense, error) {
	na, nac := A.Dims()
	if na != nac {
		return nil, ErrDimensionMismatch
	}
	nb, m := B.Dims()
	if nb != na {
		return nil, ErrDimensionMismatch
	}
	n := na
	if n == 0 {
		return &mat.Dense{}, nil
	}
	if len(poles) != n {
		return nil, ErrPoleCount
	}
	if err := validatePoles(poles); err != nil {
		return nil, err
	}

	t := make([]float64, n*n)
	aRaw := A.RawMatrix()
	copyStrided(t, n, aRaw.Data, aRaw.Stride, n, n)

	z := make([]float64, n*n)
	wr := make([]float64, n)
	wi := make([]float64, n)
	bwork := make([]bool, n)

	workQuery := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		n, t, n, wr, wi, z, n, workQuery, -1, bwork)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		n, t, n, wr, wi, z, n, work, lwork, bwork)
	if !ok {
		return nil, ErrSchurFailed
	}

	bRaw := B.RawMatrix()
	bhat := make([]float64, n*m)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: n, Data: z, Stride: n},
		blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		0, blas64.General{Rows: n, Cols: m, Data: bhat, Stride: m})

	fData := make([]float64, m*n)

	pool := make([]complex128, len(poles))
	copy(pool, poles)

	trexcWork := make([]float64, n)
	fBuf := make([]float64, m*2)
	bGen := blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride}
	zGen := blas64.General{Rows: n, Cols: n, Data: z, Stride: n}
	bhatGen := blas64.General{Rows: n, Cols: m, Data: bhat, Stride: m}
	updateBhat := func() {
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, zGen, bGen, 0, bhatGen)
	}
	moveBlock := func(from, to int) error {
		if from == to {
			return nil
		}
		if _, _, ok := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, from, to, trexcWork); !ok {
			return ErrSchurFailed
		}
		updateBhat()
		return nil
	}

	nDone := 0
	for nDone < n {
		k := n - 1
		if k > nDone && t[k*n+k-1] != 0 {
			k--
		}

		if k == n-1 {
			if pidx := closestPlacePole(pool, complex(t[k*n+k], 0), false); pidx >= 0 {
				if err := placeAssign1x1(t, z, bhat, fData, fBuf, real(pool[pidx]), n, m, k); err != nil {
					return nil, err
				}
				pool = removePoolEntry(pool, pidx)
				if err := moveBlock(k, nDone); err != nil {
					return nil, err
				}
				nDone++
				continue
			}
			j := nDone
			for j < n-1 && t[(j+1)*n+j] != 0 {
				j += 2
			}
			if j >= n-1 {
				return nil, ErrSchurFailed
			}
			if err := moveBlock(j, n-1); err != nil {
				return nil, err
			}
			k = n - 2
		}

		var tr, det float64
		if pidx := closestPlacePole(pool, schurBlock2x2Eig(t, n, k), true); pidx >= 0 {
			p := pool[pidx]
			tr, det = 2*real(p), real(p)*real(p)+imag(p)*imag(p)
			conj := cmplx.Conj(p)
			pool = removePoolEntry(pool, pidx)
			pool = removePoolEntry(pool, slices.Index(pool, conj))
		} else {
			cur := schurBlock2x2Eig(t, n, k)
			i1 := closestPlacePole(pool, cur, false)
			r1 := real(pool[i1])
			pool = removePoolEntry(pool, i1)
			i2 := closestPlacePole(pool, cur, false)
			r2 := real(pool[i2])
			pool = removePoolEntry(pool, i2)
			tr, det = r1+r2, r1*r2
		}
		if err := placeAssign2x2(t, z, bhat, fData, fBuf, tr, det, n, m, k); err != nil {
			return nil, err
		}
		standardizeSchur2x2(t, z, n, k)
		updateBhat()
		if t[(k+1)*n+k] != 0 {
			if err := moveBlock(k, nDone); err != nil {
				return nil, err
			}
		} else {
			if err := moveBlock(k, nDone); err != nil {
				return nil, err
			}
			if err := moveBlock(k+1, nDone+1); err != nil {
				return nil, err
			}
		}
		nDone += 2
	}

	return mat.NewDense(m, n, fData), nil
}

func placeAssign1x1(t, z, bhat, fData, fBuf []float64, desired float64, n, m, k int) error {
	curEig := t[k*n+k]

	bkNorm2 := blas64.Dot(
		blas64.Vector{N: m, Data: bhat[k*m:], Inc: 1},
		blas64.Vector{N: m, Data: bhat[k*m:], Inc: 1})
	if bkNorm2 < eps()*eps() {
		return fmt.Errorf("Place: mode %d: %w", k, ErrUncontrollable)
	}

	delta := curEig - desired
	scale := delta / bkNorm2
	f := fBuf[:m]
	for j := range m {
		f[j] = scale * bhat[k*m+j]
	}

	// T[:,k] -= Bhat * f  (rank-1 update on column k)
	bhatGen := blas64.General{Rows: n, Cols: m, Data: bhat, Stride: m}
	for i := range n {
		var s float64
		for j := range m {
			s += bhatGen.Data[i*bhatGen.Stride+j] * f[j]
		}
		t[i*n+k] -= s
	}

	// F_original += f outer Z[:,k]
	zk := z[k:] // Z[col,k] = z[col*n+k], stride = n
	for j := range m {
		fj := f[j]
		row := fData[j*n:]
		for col := range n {
			row[col] += fj * zk[col*n]
		}
	}
	return nil
}

func placeAssign2x2(t, z, bhat, fData, fBuf []float64, desiredTrace, desiredDet float64, n, m, k int) error {
	for iter := range 10 {
		t11 := t[k*n+k]
		t12 := t[k*n+k+1]
		t21 := t[(k+1)*n+k]
		t22 := t[(k+1)*n+k+1]

		traceNow := t11 + t22
		detNow := t11*t22 - t12*t21
		r1 := traceNow - desiredTrace
		r2 := detNow - desiredDet

		if math.Abs(r1) < eps()*math.Abs(desiredTrace+traceNow) &&
			math.Abs(r2) < eps()*math.Abs(desiredDet+detNow) {
			break
		}

		cc00, cc01, cc11 := 0.0, 0.0, 0.0
		for j := range m {
			b0 := bhat[k*m+j]
			b1 := bhat[(k+1)*m+j]
			c1f := b0*t22 - b1*t12
			c1s := b1*t11 - b0*t21

			cc00 += b0*b0 + b1*b1
			cc01 += b0*c1f + b1*c1s
			cc11 += c1f*c1f + c1s*c1s
		}

		det := cc00*cc11 - cc01*cc01
		if iter == 0 && math.Abs(det) < eps()*eps()*(cc00*cc11+1) {
			return fmt.Errorf("Place: mode %d: %w", k, ErrUncontrollable)
		}
		if math.Abs(det) < eps()*eps() {
			break
		}

		a0 := (cc11*r1 - cc01*r2) / det
		a1 := (-cc01*r1 + cc00*r2) / det

		fBlock := fBuf[:m*2]
		for j := range m {
			b0 := bhat[k*m+j]
			b1 := bhat[(k+1)*m+j]
			c1f := b0*t22 - b1*t12
			c1s := b1*t11 - b0*t21

			fBlock[j*2] = b0*a0 + c1f*a1
			fBlock[j*2+1] = b1*a0 + c1s*a1
		}

		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1,
			blas64.General{Rows: n, Cols: m, Data: bhat, Stride: m},
			blas64.General{Rows: m, Cols: 2, Data: fBlock, Stride: 2},
			1, blas64.General{Rows: n, Cols: 2, Data: t[k:], Stride: n})

		for j := range m {
			for col := range n {
				fData[j*n+col] += fBlock[j*2]*z[col*n+k] + fBlock[j*2+1]*z[col*n+k+1]
			}
		}
	}
	t11, t12, t21, t22 := t[k*n+k], t[k*n+k+1], t[(k+1)*n+k], t[(k+1)*n+k+1]
	scale := 1 + math.Abs(desiredTrace) + math.Abs(desiredDet)
	if math.Abs(t11+t22-desiredTrace) > 1e-8*scale || math.Abs(t11*t22-t12*t21-desiredDet) > 1e-8*scale {
		return fmt.Errorf("Place: mode %d: assignment did not converge: %w", k, ErrUncontrollable)
	}
	return nil
}

// standardizeSchur2x2 rotates the 2×2 block at k into standard Schur form,
// splitting it into two 1×1 blocks when its eigenvalues are real.
func standardizeSchur2x2(t, z []float64, n, k int) {
	k1 := k + 1
	var cs, sn float64
	t[k*n+k], t[k*n+k1], t[k1*n+k], t[k1*n+k1], _, _, _, _, cs, sn = impl.Dlanv2(t[k*n+k], t[k*n+k1], t[k1*n+k], t[k1*n+k1])
	if n-k-2 > 0 {
		blas64.Rot(blas64.Vector{N: n - k - 2, Data: t[k*n+k+2:], Inc: 1},
			blas64.Vector{N: n - k - 2, Data: t[k1*n+k+2:], Inc: 1}, cs, sn)
	}
	if k > 0 {
		blas64.Rot(blas64.Vector{N: k, Data: t[k:], Inc: n},
			blas64.Vector{N: k, Data: t[k1:], Inc: n}, cs, sn)
	}
	blas64.Rot(blas64.Vector{N: n, Data: z[k:], Inc: n},
		blas64.Vector{N: n, Data: z[k1:], Inc: n}, cs, sn)
}

func validatePoles(poles []complex128) error {
	counts := make(map[complex128]int)
	for _, p := range poles {
		counts[p]++
	}
	for p, c := range counts {
		if imag(p) == 0 {
			continue
		}
		conj := cmplx.Conj(p)
		if counts[conj] != c {
			return ErrConjugatePairs
		}
	}
	return nil
}

func polyFromComplexRoots(roots []complex128) Poly {
	n := len(roots)
	if n == 0 {
		return Poly{1}
	}

	sz := n + 1
	buf := make([]float64, 2*sz)
	a, b := buf[:sz], buf[sz:]
	for k := range a {
		a[k] = 0
	}
	a[0] = 1
	deg := 0

	i := 0
	for i < n {
		if imag(roots[i]) == 0 {
			u := real(roots[i])
			b[0] = a[0]
			for k := 1; k <= deg; k++ {
				b[k] = a[k] - u*a[k-1]
			}
			b[deg+1] = -u * a[deg]
			deg++
			a, b = b, a
			i++
		} else {
			ar, ai := real(roots[i]), imag(roots[i])
			c1 := -2 * ar
			c0 := ar*ar + ai*ai
			b[0] = a[0]
			b[1] = c1 * a[0]
			if deg >= 1 {
				b[1] += a[1]
			}
			for k := 2; k <= deg; k++ {
				b[k] = a[k] + c1*a[k-1] + c0*a[k-2]
			}
			b[deg+1] = c1 * a[deg]
			if deg >= 1 {
				b[deg+1] += c0 * a[deg-1]
			}
			b[deg+2] = c0 * a[deg]
			deg += 2
			a, b = b, a
			i += 2
		}
	}

	result := make(Poly, deg+1)
	copy(result, a[:deg+1])
	return result
}

// closestPlacePole returns the index of the pool entry nearest target among
// complex (wantComplex) or real entries, or -1 if there is none.
func closestPlacePole(pool []complex128, target complex128, wantComplex bool) int {
	best := -1
	bestDist := math.Inf(1)
	for i, p := range pool {
		if (imag(p) != 0) != wantComplex {
			continue
		}
		if d := cmplx.Abs(p - target); d < bestDist {
			bestDist = d
			best = i
		}
	}
	return best
}

func removePoolEntry(pool []complex128, idx int) []complex128 {
	return append(pool[:idx], pool[idx+1:]...)
}

func schurBlock2x2Eig(t []float64, n, k int) complex128 {
	ev, _ := schur2x2Eigenvalues(t[k*n+k], t[k*n+k+1], t[(k+1)*n+k], t[(k+1)*n+k+1])
	return ev
}
