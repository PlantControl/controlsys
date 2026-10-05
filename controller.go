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
//
// opts.E, when set, is a nonsingular descriptor matrix for E dx/dt = Ax + Bu.
// As MATLAB lqr for descriptor models
// (https://www.mathworks.com/help/control/ref/lti.lqr.html), X then solves the
// Riccati equation of the explicit model dx/dt = E⁻¹Ax + E⁻¹Bu (E'X_gE for
// the Care solution X_g with E); K is unchanged and Eig holds the
// generalized eigenvalues of (A-BK, E).
func Lqr(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	res, err := Care(A, B, Q, R, opts)
	return explicitRiccatiSolution(res, opts, err)
}

// Dlqr solves the discrete-time linear-quadratic regulator problem.
// It computes the optimal gain K such that u[k] = -Kx[k] minimizes
// J = sum(x'Qx + u'Ru) for the system x[k+1] = Ax[k] + Bu[k].
//
// A is n×n, B is n×m, Q is n×n symmetric PSD, R is m×m symmetric PD.
// Returns gain K (m×n), Riccati solution X, and closed-loop eigenvalues.
// opts.E is handled as in Lqr: X solves the Riccati equation of the explicit
// model x[k+1] = E⁻¹Ax[k] + E⁻¹Bu[k].
func Dlqr(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	res, err := Dare(A, B, Q, R, opts)
	return explicitRiccatiSolution(res, opts, err)
}

// explicitRiccatiSolution maps the generalized solution X of a Care or Dare
// solved with opts.E to E'XE, the solution for the explicit model.
func explicitRiccatiSolution(res *RiccatiResult, opts *RiccatiOpts, err error) (*RiccatiResult, error) {
	if err != nil || opts == nil || isIdentityDescriptor(opts.E) {
		return res, err
	}
	n, _ := res.X.Dims()
	if n == 0 {
		return res, nil
	}
	xe := mulDims(n, n, res.X, opts.E)
	res.X = mulDims(n, n, opts.E.T(), xe)
	symmetrize(res.X.RawMatrix().Data, n, n)
	return res, nil
}

// Lqi computes the linear-quadratic regulator with integral action, matching
// MATLAB lqi(sys,Q,R,N) (https://www.mathworks.com/help/control/ref/ss.lqi.html).
// The plant x' = Ax + Bu, y = Cx + Du is augmented with the integral xi of
// the tracking error r - y:
//
//	continuous: xi' = r - y                 Aa = [A 0; -C 0],     Ba = [B; -D]
//	discrete:   xi[n+1] = xi[n] + Ts(r - y)  Aa = [A 0; -Ts*C I],  Ba = [B; -Ts*D]
//
// and u = -K[x; xi] minimizes the cost on z = [x; xi] with weights Q
// ((n+p)×(n+p)), R (m×m) and cross term opts.S = N ((n+p)×m), solved with
// Care or Dare by the plant's time domain. K is m×(n+p), [Kx Ki]; X and Eig
// are the augmented Riccati solution and closed-loop eigenvalues.
//
// Descriptor plants E x' = Ax + Bu with nonsingular E are supported as in
// MATLAB: the augmented descriptor is blkdiag(E, I), K is the gain of the
// explicit model (E⁻¹A, E⁻¹B, C, D) and X solves its Riccati equation.
//
// Plants without states, inputs or outputs return ErrDimensionMismatch,
// asymmetric Q or R ErrNotSymmetric, singular E ErrDescriptorSingular,
// opts.E ErrOptionUnsupported and plants with delays ErrDelayUnsupported.
func Lqi(sys *System, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	policy, err := newControllerObserverPolicy(sys, "Lqi")
	if err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if m == 0 || p == 0 {
		return nil, fmt.Errorf("Lqi: model needs inputs and outputs: %w", ErrDimensionMismatch)
	}
	if err := policy.rejectOptsE(opts); err != nil {
		return nil, err
	}
	if Q == nil || R == nil {
		return nil, fmt.Errorf("Lqi: nil weight: %w", ErrDimensionMismatch)
	}
	if qr, qc := Q.Dims(); qr != n+p || qc != n+p {
		return nil, fmt.Errorf("Lqi: Q is %dx%d, want %dx%d: %w", qr, qc, n+p, n+p, ErrDimensionMismatch)
	}
	if rr, rc := R.Dims(); rr != m || rc != m {
		return nil, fmt.Errorf("Lqi: R is %dx%d, want %dx%d: %w", rr, rc, m, m, ErrDimensionMismatch)
	}
	if !isSymmetric(Q, eps()*denseNorm(Q)) || !isSymmetric(R, eps()*denseNorm(R)) {
		return nil, fmt.Errorf("Lqi: %w", ErrNotSymmetric)
	}
	Aa, Ba := lqiAugmentation(sys)
	if Ea := augmentedDescriptor(sys.E, p); Ea != nil {
		o := RiccatiOpts{E: Ea}
		if opts != nil {
			o.S, o.Workspace = opts.S, opts.Workspace
		}
		opts = &o
	}
	if sys.IsContinuous() {
		return Lqr(Aa, Ba, Q, R, opts)
	}
	return Dlqr(Aa, Ba, Q, R, opts)
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
// Riccati solve. opts.E is rejected with ErrOptionUnsupported.
func Lqrd(A, B, Q, R *mat.Dense, dt float64, opts *RiccatiOpts) (*RiccatiResult, error) {
	if opts != nil && opts.E != nil {
		return nil, fmt.Errorf("Lqrd: opts.E: %w", ErrOptionUnsupported)
	}
	if dt <= 0 || newTimeDomain(dt).validateSampleTime() != nil {
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

// Place computes the state-feedback gain K (m×n) such that eig(A − B·K)
// equals the desired poles, following MATLAB place.
//
// When rank(B) ≥ 2 it uses the Kautsky–Nichols–Van Dooren robust assignment
// (method 0; MATLAB place cites the same paper): each closed-loop eigenvector is
// chosen within its admissible subspace to keep the eigenvector matrix well
// conditioned, so the poles are insensitive to perturbations in A and B and a
// repeated pole is non-defective. When rank(B) = 1 the gain is unique and is
// computed by Varga's Schur method, which is also the fallback when the robust
// eigenvector matrix has 1-norm condition number above 1/√ε (an
// uncontrollable mode makes it singular). A mode that no input can move
// returns ErrUncontrollable.
//
// Poles must come in conjugate pairs and len(poles) must equal n. As in MATLAB,
// a pole repeated (exactly) more than rank(B) times returns
// ErrPoleMultiplicity; use Acker for repeated single-input poles.
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

	var svd mat.SVD
	if !svd.Factorize(B, mat.SVDFull) {
		return nil, ErrSchurFailed
	}
	sv := svd.Values(nil)
	r := 0
	for _, s := range sv {
		if s > float64(max(n, m))*eps()*sv[0] {
			r++
		}
	}
	if r == 0 {
		return nil, fmt.Errorf("Place: %w", ErrUncontrollable)
	}
	if maxPoleMultiplicity(poles) > r {
		return nil, ErrPoleMultiplicity
	}
	if r >= 2 {
		if K := placeKNV(A, &svd, sv[:r], poles); K != nil {
			return K, nil
		}
	}
	return placeSchur(A, B, poles)
}

// placeSchur is Varga's Schur-based pole placement: deflating assignment of
// 1×1 blocks (minimum-norm gain) and 2×2 blocks (closed form; with two
// independent input directions the block is made normal).
func placeSchur(A, B *mat.Dense, poles []complex128) (*mat.Dense, error) {
	n, m := B.Dims()

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

		var p1, p2 complex128
		if pidx := closestPlacePole(pool, schurBlock2x2Eig(t, n, k), true); pidx >= 0 {
			p1 = pool[pidx]
			p2 = cmplx.Conj(p1)
			pool = removePoolEntry(pool, pidx)
			pool = removePoolEntry(pool, slices.Index(pool, p2))
		} else {
			cur := schurBlock2x2Eig(t, n, k)
			i1 := closestPlacePole(pool, cur, false)
			p1 = pool[i1]
			pool = removePoolEntry(pool, i1)
			i2 := closestPlacePole(pool, cur, false)
			p2 = pool[i2]
			pool = removePoolEntry(pool, i2)
		}
		if err := placeAssign2x2(t, z, bhat, fData, fBuf, p1, p2, n, m, k); err != nil {
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

// placeAssign2x2 assigns the eigenvalues p1, p2 (a conjugate pair or two
// reals) to the trailing 2×2 Schur block at k in closed form. When B2 (the
// block's rows of Bhat) has two well-conditioned input directions, the block
// is driven to a normal matrix M via F2 = B2⁺(T2−M), giving perfectly
// conditioned block eigenvalues and non-defective repeated poles. Otherwise a
// single input direction b = B2·w is used, which fixes F2 uniquely. Targets enter only through
// (re, im) or (r1, r2), never through (trace, det), so small imaginary parts
// survive at any scale.
func placeAssign2x2(t, z, bhat, fData, fBuf []float64, p1, p2 complex128, n, m, k int) error {
	k1 := k + 1
	t11, t12, t21, t22 := t[k*n+k], t[k*n+k1], t[k1*n+k], t[k1*n+k1]
	b0 := bhat[k*m : k*m+m]
	b1 := bhat[k1*m : k1*m+m]
	tNorm := math.Abs(t11) + math.Abs(t12) + math.Abs(t21) + math.Abs(t22)

	fBlock := fBuf[:2*m]
	if !placeMultiInput2x2(fBlock, b0, b1, t11, t12, t21, t22, p1, p2, m) &&
		!placeSingleInput2x2(fBlock, b0, b1, t11, t12, t21, t22, tNorm, p1, p2, m) {
		return fmt.Errorf("Place: mode %d: %w", k, ErrUncontrollable)
	}

	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1,
		blas64.General{Rows: n, Cols: m, Data: bhat, Stride: m},
		blas64.General{Rows: m, Cols: 2, Data: fBlock, Stride: 2},
		1, blas64.General{Rows: n, Cols: 2, Data: t[k:], Stride: n})

	for j := range m {
		for col := range n {
			fData[j*n+col] += fBlock[j*2]*z[col*n+k] + fBlock[j*2+1]*z[col*n+k1]
		}
	}
	return nil
}

// placeSingleInput2x2 writes into f (m×2) the gain w·g that assigns p1, p2
// through the dominant input direction b = B2·w, reporting false when the
// block is uncontrollable from b. In the rotated basis where
// b = [0; β] only the second row of the block changes, and its new entries
// follow from the factored characteristic polynomial.
func placeSingleInput2x2(f, b0, b1 []float64, t11, t12, t21, t22, tNorm float64, p1, p2 complex128, m int) bool {
	g00, g01, g11 := 0.0, 0.0, 0.0
	for j := range m {
		g00 += b0[j] * b0[j]
		g01 += b0[j] * b1[j]
		g11 += b1[j] * b1[j]
	}
	_, _, u0, u1 := impl.Dlaev2(g00, g01, g11)
	var v0, v1, wn float64
	for j := range m {
		w := u0*b0[j] + u1*b1[j]
		wn += w * w
		v0 += b0[j] * w
		v1 += b1[j] * w
	}
	wn = math.Sqrt(wn)
	sigma := math.Hypot(v0, v1)
	if sigma == 0 {
		return false
	}
	beta := sigma / wn
	c, s := v1/sigma, v0/sigma
	// Q = [c -s; s c] maps b to [0; |b|]; A' = Q T2 Qᵀ.
	a11 := c*(c*t11-s*t21) - s*(c*t12-s*t22)
	a12 := s*(c*t11-s*t21) + c*(c*t12-s*t22)
	a21 := c*(s*t11+c*t21) - s*(s*t12+c*t22)
	a22 := s*(s*t11+c*t21) + c*(s*t12+c*t22)
	if math.Abs(a12) <= eps()*tNorm {
		return false
	}
	x2 := real(p1) + real(p2) - a11
	var x1 float64
	if imag(p1) != 0 {
		d := a11 - real(p1)
		x1 = -(d*d + imag(p1)*imag(p1)) / a12
	} else {
		x1 = -(a11 - real(p1)) * (a11 - real(p2)) / a12
	}
	g0 := (a21 - x1) / beta
	g1 := (a22 - x2) / beta
	h0 := g0*c + g1*s
	h1 := -g0*s + g1*c
	for j := range m {
		w := (u0*b0[j] + u1*b1[j]) / wn
		f[j*2] = w * h0
		f[j*2+1] = w * h1
	}
	return true
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

// placeMultiInput2x2 writes into f (m×2) the gain B2⁺(T2−M) that turns the
// 2×2 block into the normal matrix M with eigenvalues p1, p2, using an LQ
// factorization of B2 = L·Q. It reports false, leaving f unspecified, when
// L is too ill-conditioned (l11/l00 < √ε) for B2·B2⁺ ≈ I to hold.
func placeMultiInput2x2(f, b0, b1 []float64, t11, t12, t21, t22 float64, p1, p2 complex128, m int) bool {
	if m < 2 {
		return false
	}
	var m11, m12, m21, m22 float64
	if im := imag(p1); im != 0 {
		im = math.Abs(im)
		if t12 < t21 {
			im = -im
		}
		m11, m12, m21, m22 = real(p1), im, -im, real(p1)
	} else {
		r1, r2 := real(p1), real(p2)
		if math.Abs(t11-r1)+math.Abs(t22-r2) > math.Abs(t11-r2)+math.Abs(t22-r1) {
			r1, r2 = r2, r1
		}
		m11, m22 = r1, r2
	}
	d00, d01, d10, d11 := t11-m11, t12-m12, t21-m21, t22-m22

	ra, rb := b0, b1
	n0 := blas64.Nrm2(blas64.Vector{N: m, Data: b0, Inc: 1})
	n1 := blas64.Nrm2(blas64.Vector{N: m, Data: b1, Inc: 1})
	l00 := n0
	if n1 > n0 {
		ra, rb, l00 = b1, b0, n1
		d00, d01, d10, d11 = d10, d11, d00, d01
	}
	if l00 == 0 {
		return false
	}
	var l10 float64
	for j := range m {
		l10 += ra[j] / l00 * rb[j]
	}
	var corr float64
	for j := range m {
		corr += ra[j] / l00 * (rb[j] - l10*ra[j]/l00)
	}
	l10 += corr
	var l11 float64
	for j := range m {
		r := rb[j] - l10*ra[j]/l00
		l11 += r * r
	}
	l11 = math.Sqrt(l11)
	if l11 < math.Sqrt(eps())*l00 {
		return false
	}

	y00, y01 := d00/l00, d01/l00
	y10, y11 := (d10-l10*y00)/l11, (d11-l10*y01)/l11
	for j := range m {
		q0 := ra[j] / l00
		q1 := (rb[j] - l10*q0) / l11
		f[j*2] = q0*y00 + q1*y10
		f[j*2+1] = q0*y01 + q1*y11
	}
	return true
}

const placeKNVMaxSweeps = 5

func maxPoleMultiplicity(poles []complex128) int {
	counts := make(map[complex128]int, len(poles))
	best := 0
	for _, p := range poles {
		counts[p]++
		best = max(best, counts[p])
	}
	return best
}

// placeKNV is the Kautsky–Nichols–Van Dooren method 0 for B = U·Σ·Vᵀ of
// rank r = len(sv) ≥ 2. In an orthogonal basis W where Wᵀ·B = [Σ_r·V_rᵀ; 0]
// and T = Wᵀ·A·W has T[p][q] = 0 for q < p−r, the admissible eigenvectors
// for λ span null(T[r:,:] − λ·[0 I]). X starts from fixed pseudo-random
// admissible vectors (distinct ones for a repeated pole); each sweep then
// replaces every eigenvector, or conjugate pair jointly, by the admissible
// unit vector maximizing |det X| with the others fixed, read off the rows of
// X⁻¹. X is kept in real form (Re x, Im x for a pair). It returns nil when
// κ₁(X) ≥ 1/√ε.
func placeKNV(A *mat.Dense, svd *mat.SVD, sv []float64, poles []complex128) *mat.Dense {
	n, _ := A.Dims()
	r := len(sv)
	var uMat, vMat mat.Dense
	svd.UTo(&uMat)
	svd.VTo(&vMat)
	m, _ := vMat.Dims()

	w := make([]float64, n*n)
	uRaw := uMat.RawMatrix()
	copyStrided(w, n, uRaw.Data, uRaw.Stride, n, n)
	aRaw := A.RawMatrix()
	tmp := make([]float64, n*n)
	at := make([]float64, n*n)
	wGen := blas64.General{Rows: n, Cols: n, Data: w, Stride: n}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Data: aRaw.Data, Stride: aRaw.Stride}, wGen,
		0, blas64.General{Rows: n, Cols: n, Data: tmp, Stride: n})
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, wGen,
		blas64.General{Rows: n, Cols: n, Data: tmp, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: at, Stride: n})
	placeBandReduce(at, w, n, r)

	lam := make([]complex128, 0, n)
	for _, p := range poles {
		if imag(p) == 0 {
			lam = append(lam, p)
		}
	}
	for _, p := range poles {
		if imag(p) > 0 {
			lam = append(lam, p, cmplx.Conj(p))
		}
	}

	basisOf := make(map[complex128][]complex128)
	basis := make([][]complex128, n)
	occ := make([]int, n)
	seen := make(map[complex128]int)
	for _, p := range lam {
		if imag(p) >= 0 {
			seen[p]++
		}
	}
	slab := make([]complex128, len(seen)*n*r)
	clear(seen)
	nb := newPlaceNullBasis(n, r)
	for j := range n {
		p := lam[j]
		if imag(p) < 0 {
			continue
		}
		s, ok := basisOf[p]
		if !ok {
			s, slab = slab[:n*r:n*r], slab[n*r:]
			nb.compute(s, at, p)
			basisOf[p] = s
		}
		basis[j] = s
		occ[j] = seen[p]
		seen[p]++
	}

	x := make([]float64, n*n)
	col := make([]complex128, n)
	c := make([]complex128, r)
	h := make([]complex128, r)
	seed := uint64(0x9e3779b97f4a7c15)
	next := func() float64 {
		seed ^= seed << 13
		seed ^= seed >> 7
		seed ^= seed << 17
		return float64(seed>>11)/(1<<53) - 0.5
	}
	for j := range n {
		if basis[j] == nil {
			continue
		}
		for k := range c {
			c[k] = complex(next(), 0)
			if imag(lam[j]) != 0 {
				c[k] += complex(0, next())
			}
		}
		c[occ[j]%r] += 2
		placeKNVCombine(col, basis[j], c, n, r)
		placeKNVSetCols(x, col, n, j, imag(lam[j]) != 0)
	}

	ws := newPlaceKNVWork(n)
	xinv := ws.inverse(x, eps())
	if xinv == nil {
		return nil
	}
	for range placeKNVMaxSweeps {
		gain := 0.0
		for j := range n {
			s := basis[j]
			if s == nil {
				continue
			}
			y1 := xinv[j*n : (j+1)*n]
			if imag(lam[j]) == 0 {
				for k := range r {
					var acc float64
					for i := range n {
						acc += real(s[i*r+k]) * y1[i]
					}
					c[k] = complex(acc, 0)
				}
				if placeKNVCombine(col, s, c, n, r) {
					gain += ws.replace(x, xinv, col, j, false)
				}
				continue
			}
			y2 := xinv[(j+1)*n : (j+2)*n]
			for k := range r {
				var acc complex128
				for i := range n {
					acc += cmplx.Conj(s[i*r+k]) * complex(y1[i], y2[i])
				}
				c[k] = acc
				acc = 0
				for i := range n {
					acc += cmplx.Conj(s[i*r+k]) * complex(y1[i], -y2[i])
				}
				h[k] = acc
			}
			if placeKNVPairCoeffs(c, h) && placeKNVCombine(col, s, c, n, r) {
				gain += ws.replace(x, xinv, col, j, true)
			}
		}
		if gain < 1e-3 {
			break
		}
	}

	xinv = ws.inverse(x, math.Sqrt(eps()))
	if xinv == nil {
		return nil
	}

	y := make([]float64, r*n)
	for i := range r {
		for j := 0; j < n; j++ {
			re, im := real(lam[j]), imag(lam[j])
			if im == 0 {
				y[i*n+j] = x[i*n+j] * re
				continue
			}
			xr, xi := x[i*n+j], x[i*n+j+1]
			y[i*n+j] = re*xr - im*xi
			y[i*n+j+1] = im*xr + re*xi
			j++
		}
	}
	d := make([]float64, r*n)
	for i := range r {
		copy(d[i*n:(i+1)*n], at[i*n:(i+1)*n])
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1,
		blas64.General{Rows: r, Cols: n, Data: y, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: xinv, Stride: n},
		1, blas64.General{Rows: r, Cols: n, Data: d, Stride: n})
	for i := range r {
		blas64.Scal(1/sv[i], blas64.Vector{N: n, Data: d[i*n:], Inc: 1})
	}
	vRaw := vMat.RawMatrix()
	kz := make([]float64, m*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: m, Cols: r, Data: vRaw.Data, Stride: vRaw.Stride},
		blas64.General{Rows: r, Cols: n, Data: d, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: kz, Stride: n})
	k := make([]float64, m*n)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: m, Cols: n, Data: kz, Stride: n}, wGen,
		0, blas64.General{Rows: m, Cols: n, Data: k, Stride: n})
	return mat.NewDense(m, n, k)
}

// placeBandReduce applies Householder similarities on coordinates r..n−1 so
// that t[p][q] = 0 for q < p−r, accumulating them into the columns of w.
func placeBandReduce(t, w []float64, n, r int) {
	v := make([]float64, n)
	work := make([]float64, n)
	for q := 0; q+r+1 < n; q++ {
		p := q + r
		l := n - p
		beta, tau := impl.Dlarfg(l, t[p*n+q], t[(p+1)*n+q:], n)
		v[0] = 1
		for i := 1; i < l; i++ {
			v[i] = t[(p+i)*n+q]
			t[(p+i)*n+q] = 0
		}
		t[p*n+q] = beta
		impl.Dlarf(blas.Left, l, n-q-1, v[:l], 1, tau, t[p*n+q+1:], n, work)
		impl.Dlarf(blas.Right, n, l, v[:l], 1, tau, t[p:], n, work)
		impl.Dlarf(blas.Right, n, l, v[:l], 1, tau, w[p:], n, work)
	}
}

type placeGivens struct {
	c, d   int
	ga, gb complex128
}

type placeGivensReal struct {
	c, d   int
	ga, gb float64
}

type placeNullBasis struct {
	n, r  int
	mw    []complex128
	mwr   []float64
	sr    []float64
	rots  []placeGivens
	rotsR []placeGivensReal
}

func newPlaceNullBasis(n, r int) *placeNullBasis {
	return &placeNullBasis{n: n, r: r}
}

// compute writes into s an orthonormal basis (n×r, row-major) of
// null(t[r:,:] − λ·[0 I]). Column rotations reduce the banded pencil row by
// row from the bottom to [0 R]; the first r columns of their product span
// the null space.
func (b *placeNullBasis) compute(s []complex128, t []float64, lam complex128) {
	if imag(lam) == 0 {
		b.computeReal(s, t, real(lam))
		return
	}
	n, r := b.n, b.r
	if b.mw == nil {
		b.mw = make([]complex128, (n-r)*n)
		b.rots = make([]placeGivens, 0, (n-r)*r)
	}
	mw := b.mw
	for i := range n - r {
		for j := range n {
			mw[i*n+j] = complex(t[(r+i)*n+j], 0)
		}
		mw[i*n+r+i] -= lam
	}
	b.rots = b.rots[:0]
	for i := n - r - 1; i >= 0; i-- {
		d := r + i
		for c := i; c < d; c++ {
			a, bb := mw[i*n+c], mw[i*n+d]
			h := math.Hypot(cmplx.Abs(a), cmplx.Abs(bb))
			if h == 0 {
				continue
			}
			ga, gb := a/complex(h, 0), bb/complex(h, 0)
			cga, cgb := cmplx.Conj(ga), cmplx.Conj(gb)
			for row := 0; row <= i; row++ {
				xc, xd := mw[row*n+c], mw[row*n+d]
				mw[row*n+c] = xc*gb - xd*ga
				mw[row*n+d] = xc*cga + xd*cgb
			}
			b.rots = append(b.rots, placeGivens{c, d, ga, gb})
		}
	}
	clear(s)
	for k := range r {
		s[k*r+k] = 1
	}
	for _, g := range slices.Backward(b.rots) {

		cga, cgb := cmplx.Conj(g.ga), cmplx.Conj(g.gb)
		for k := range r {
			vc, vd := s[g.c*r+k], s[g.d*r+k]
			s[g.c*r+k] = g.gb*vc + cga*vd
			s[g.d*r+k] = -g.ga*vc + cgb*vd
		}
	}
}

func (b *placeNullBasis) computeReal(s []complex128, t []float64, lam float64) {
	n, r := b.n, b.r
	if b.mwr == nil {
		b.mwr = make([]float64, (n-r)*n)
		b.sr = make([]float64, n*r)
		b.rotsR = make([]placeGivensReal, 0, (n-r)*r)
	}
	mw := b.mwr
	copy(mw, t[r*n:])
	for i := range n - r {
		mw[i*n+r+i] -= lam
	}
	b.rotsR = b.rotsR[:0]
	for i := n - r - 1; i >= 0; i-- {
		d := r + i
		for c := i; c < d; c++ {
			a, bb := mw[i*n+c], mw[i*n+d]
			if a == 0 {
				continue
			}
			h := math.Hypot(a, bb)
			ga, gb := a/h, bb/h
			for row := 0; row <= i; row++ {
				xc, xd := mw[row*n+c], mw[row*n+d]
				mw[row*n+c] = xc*gb - xd*ga
				mw[row*n+d] = xc*ga + xd*gb
			}
			b.rotsR = append(b.rotsR, placeGivensReal{c, d, ga, gb})
		}
	}
	sr := b.sr
	clear(sr)
	for k := range r {
		sr[k*r+k] = 1
	}
	for _, g := range slices.Backward(b.rotsR) {

		vc, vd := sr[g.c*r:g.c*r+r], sr[g.d*r:g.d*r+r]
		for k := range r {
			vc[k], vd[k] = g.gb*vc[k]+g.ga*vd[k], -g.ga*vc[k]+g.gb*vd[k]
		}
	}
	for i, v := range sr {
		s[i] = complex(v, 0)
	}
}

// placeKNVCombine writes the unit vector S·c into col, reporting false when
// S·c vanishes.
func placeKNVCombine(col, s, c []complex128, n, r int) bool {
	var nrm float64
	for i := range n {
		var acc complex128
		for k := range r {
			acc += s[i*r+k] * c[k]
		}
		col[i] = acc
		nrm += real(acc)*real(acc) + imag(acc)*imag(acc)
	}
	if nrm == 0 {
		return false
	}
	inv := complex(1/math.Sqrt(nrm), 0)
	for i := range col {
		col[i] *= inv
	}
	return true
}

// placeKNVSetCols stores the eigenvector col in real form: column j for a
// real pole, or Re col, Im col in columns j, j+1 for a conjugate pair.
func placeKNVSetCols(x []float64, col []complex128, n, j int, pair bool) {
	for i := range n {
		x[i*n+j] = real(col[i])
		if pair {
			x[i*n+j+1] = imag(col[i])
		}
	}
}

// placeKNVPairCoeffs turns g = Sᴴv, h = Sᴴv̄ (v ⟂ all columns but the pair)
// into c maximizing |det X| for the pair [S·c, conj(S·c)]: the top
// eigenvector of g·gᴴ − h·hᴴ, written as c = α·g + β·h. It reports false
// when g and h vanish.
func placeKNVPairCoeffs(g, h []complex128) bool {
	var gg, hh float64
	var gh complex128
	for k := range g {
		gg += real(g[k])*real(g[k]) + imag(g[k])*imag(g[k])
		hh += real(h[k])*real(h[k]) + imag(h[k])*imag(h[k])
		gh += cmplx.Conj(g[k]) * h[k]
	}
	d := gg - hh
	mu := (d + math.Copysign(math.Sqrt(d*d+4*math.Max(gg*hh-real(gh*cmplx.Conj(gh)), 0)), d)) / 2
	if mu == 0 {
		return false
	}
	a1, b1 := gh, complex(mu-gg, 0)
	a2, b2 := complex(hh+mu, 0), -cmplx.Conj(gh)
	alpha, beta := a1, b1
	if cmplx.Abs(a2)+cmplx.Abs(b2) > cmplx.Abs(a1)+cmplx.Abs(b1) {
		alpha, beta = a2, b2
	}
	for k := range g {
		g[k] = alpha*g[k] + beta*h[k]
	}
	return true
}

type placeKNVWork struct {
	n    int
	ipiv []int
	iw   []int
	work []float64
	lu   []float64
	inv  []float64
	xc   []float64
	u    []float64
	rows []float64
}

func newPlaceKNVWork(n int) *placeKNVWork {
	return &placeKNVWork{
		n:    n,
		ipiv: make([]int, n),
		iw:   make([]int, n),
		work: make([]float64, 4*n),
		lu:   make([]float64, n*n),
		inv:  make([]float64, n*n),
		xc:   make([]float64, 2*n),
		u:    make([]float64, 2*n),
		rows: make([]float64, 2*n),
	}
}

// inverse returns x⁻¹ in reused storage, or nil when κ₁(x) ≥ 1/√ε.
func (w *placeKNVWork) inverse(x []float64, minRcond float64) []float64 {
	n := w.n
	copy(w.lu, x)
	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, w.lu, n, w.work)
	if !impl.Dgetrf(n, n, w.lu, n, w.ipiv) {
		return nil
	}
	if rcond := impl.Dgecon(lapack.MaxColumnSum, n, w.lu, n, anorm, w.work, w.iw); !(rcond > minRcond) {
		return nil
	}
	clear(w.inv)
	for i := range n {
		w.inv[i*n+i] = 1
	}
	impl.Dgetrs(blas.NoTrans, n, n, w.lu, n, w.ipiv, w.inv, n)
	return w.inv
}

// replace stores the eigenvector col at column j (and j+1 for a pair) of
// the real-form x and updates xinv by the Sherman–Morrison–Woodbury formula,
// returning log|det X_new / det X_old|. It leaves x unchanged and returns 0
// when the update would make x numerically singular.
func (w *placeKNVWork) replace(x, xinv []float64, col []complex128, j int, pair bool) float64 {
	n := w.n
	p := 1
	if pair {
		p = 2
	}
	xr, xi := w.xc[:n], w.xc[n:]
	u0, u1 := w.u[:n], w.u[n:]
	for i := range n {
		xr[i], xi[i] = real(col[i]), imag(col[i])
	}
	xinvGen := blas64.General{Rows: n, Cols: n, Data: xinv, Stride: n}
	blas64.Gemv(blas.NoTrans, 1, xinvGen, blas64.Vector{N: n, Data: xr, Inc: 1}, 0, blas64.Vector{N: n, Data: u0, Inc: 1})
	un := math.Abs(u0[blas64.Iamax(blas64.Vector{N: n, Data: u0, Inc: 1})])
	var i00, i01, i10, i11, det float64
	if pair {
		blas64.Gemv(blas.NoTrans, 1, xinvGen, blas64.Vector{N: n, Data: xi, Inc: 1}, 0, blas64.Vector{N: n, Data: u1, Inc: 1})
		un = max(un, math.Abs(u1[blas64.Iamax(blas64.Vector{N: n, Data: u1, Inc: 1})]))
		d00, d01, d10, d11 := u0[j], u1[j], u0[j+1], u1[j+1]
		det = d00*d11 - d01*d10
		i00, i01, i10, i11 = d11/det, -d01/det, -d10/det, d00/det
	} else {
		det = u0[j]
		i00 = 1 / det
	}
	if !(math.Abs(det) > math.Sqrt(eps())*math.Pow(un, float64(p))) {
		return 0
	}
	r0, r1 := w.rows[:n], w.rows[n:]
	copy(r0, xinv[j*n:(j+1)*n])
	u0[j]--
	if pair {
		copy(r1, xinv[(j+1)*n:(j+2)*n])
		u1[j+1]--
		for i := range n {
			a, b := u0[i], u1[i]
			u0[i], u1[i] = a*i00+b*i10, a*i01+b*i11
		}
		blas64.Ger(-1, blas64.Vector{N: n, Data: u1, Inc: 1}, blas64.Vector{N: n, Data: r1, Inc: 1}, xinvGen)
	} else {
		blas64.Scal(i00, blas64.Vector{N: n, Data: u0, Inc: 1})
	}
	blas64.Ger(-1, blas64.Vector{N: n, Data: u0, Inc: 1}, blas64.Vector{N: n, Data: r0, Inc: 1}, xinvGen)
	placeKNVSetCols(x, col, n, j, pair)
	return math.Log(math.Abs(det))
}
