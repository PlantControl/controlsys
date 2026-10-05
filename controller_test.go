package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func matchEigenvalues(achieved, desired []complex128, _ float64) float64 {
	if len(achieved) != len(desired) {
		return math.Inf(1)
	}
	used := make([]bool, len(desired))
	maxDist := 0.0
	for _, a := range achieved {
		bestDist := math.Inf(1)
		bestIdx := -1
		for j, d := range desired {
			if used[j] {
				continue
			}
			dist := cmplx.Abs(a - d)
			if dist < bestDist {
				bestDist = dist
				bestIdx = j
			}
		}
		if bestIdx >= 0 {
			used[bestIdx] = true
			if bestDist > maxDist {
				maxDist = bestDist
			}
		}
	}
	return maxDist
}

func closedLoopEig(A, B, K *mat.Dense) []complex128 {
	n, _ := A.Dims()
	ACL := mat.NewDense(n, n, nil)
	ACL.Mul(B, K)
	ACL.Sub(A, ACL)
	var eig mat.Eigen
	ok := eig.Factorize(ACL, mat.EigenNone)
	if !ok {
		return nil
	}
	return eig.Values(nil)
}

// --- LQR Tests ---

func TestLqr_DoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Lqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(res.K.At(0, 0)-1) > 1e-6 || math.Abs(res.K.At(0, 1)-math.Sqrt(3)) > 1e-6 {
		t.Errorf("K = [%v, %v], want [1, sqrt(3)]", res.K.At(0, 0), res.K.At(0, 1))
	}
	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("non-stable eigenvalue: %v", e)
		}
	}
}

func TestLqr_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 0})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Lqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r := careResidual(A, B, Q, R, res.X); r > 1e-10 {
		t.Errorf("residual = %e", r)
	}
}

func TestLqr_MIMO(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 1, 0,
		0, -2, 1,
		0, 0, -3,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		1, 1,
	})
	Q := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	res, err := Lqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("non-stable eigenvalue: %v", e)
		}
	}
}

func TestLqr_CrossTerm(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{2, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	S := mat.NewDense(2, 1, []float64{0, 0.5})

	res, err := Lqr(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("non-stable eigenvalue: %v", e)
		}
	}
}

// --- Dlqr Tests ---

func TestDlqr_DoubleIntegrator(t *testing.T) {
	dt := 0.1
	A := mat.NewDense(2, 2, []float64{1, dt, 0, 1})
	B := mat.NewDense(2, 1, []float64{0.5 * dt * dt, dt})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Dlqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1 {
			t.Errorf("non-stable eigenvalue: %v (|e|=%v)", e, cmplx.Abs(e))
		}
	}
}

func TestDlqr_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.2, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{0.1, 0.05})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Dlqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r := dareResidual(A, B, Q, R, res.X); r > 1e-10 {
		t.Errorf("residual = %e", r)
	}
}

// --- Lqi Tests ---

func TestLqi_SISO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	n, _ := A.Dims()
	p, _ := C.Dims()
	aug := n + p
	Q := mat.NewDense(aug, aug, nil)
	for i := range aug {
		Q.Set(i, i, 1)
	}
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Lqi(A, B, C, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	kr, kc := res.K.Dims()
	if kr != 1 || kc != aug {
		t.Errorf("K dims = %dx%d, want 1x%d", kr, kc, aug)
	}
	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("non-stable eigenvalue: %v", e)
		}
	}
}

func TestLqi_MIMO(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 1, 0,
		0, -2, 1,
		0, 0, -3,
	})
	B := mat.NewDense(3, 2, []float64{1, 0, 0, 1, 0, 0})
	C := mat.NewDense(2, 3, []float64{1, 0, 0, 0, 1, 0})
	n, _ := A.Dims()
	p, _ := C.Dims()
	_, m := B.Dims()
	aug := n + p
	Q := mat.NewDense(aug, aug, nil)
	for i := range aug {
		Q.Set(i, i, 1)
	}
	R := mat.NewDense(m, m, nil)
	for i := range m {
		R.Set(i, i, 1)
	}

	res, err := Lqi(A, B, C, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	kr, kc := res.K.Dims()
	if kr != m || kc != aug {
		t.Errorf("K dims = %dx%d, want %dx%d", kr, kc, m, aug)
	}
}

func TestLqi_DimError(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(2, 1, nil)
	C := mat.NewDense(1, 3, nil) // wrong cols
	Q := mat.NewDense(3, 3, nil)
	R := mat.NewDense(1, 1, []float64{1})
	_, err := Lqi(A, B, C, Q, R, nil)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

// --- Lqrd Tests ---

func TestLqrd_DoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	dt := 0.01

	res, err := Lqrd(A, B, Q, R, dt, nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1 {
			t.Errorf("non-stable eigenvalue: %v (|e|=%v)", e, cmplx.Abs(e))
		}
	}
}

// lqrdOracle discretizes the continuous cost by composite Simpson quadrature of
// [Φ Γ; 0 I]' [Q N; N' R] [Φ Γ; 0 I] and solves the discrete Riccati equation by
// value iteration, independently of the Van Loan block exponential and Dare.
func lqrdOracle(t *testing.T, A, B, Q, R, N *mat.Dense, dt float64) (K, X *mat.Dense) {
	t.Helper()
	n, m := B.Dims()
	nm := n + m
	M := mat.NewDense(nm, nm, nil)
	M.Slice(0, n, 0, n).(*mat.Dense).Copy(A)
	M.Slice(0, n, n, nm).(*mat.Dense).Copy(B)
	Wc := mat.NewDense(nm, nm, nil)
	Wc.Slice(0, n, 0, n).(*mat.Dense).Copy(Q)
	Wc.Slice(n, nm, n, nm).(*mat.Dense).Copy(R)
	if N != nil {
		Wc.Slice(0, n, n, nm).(*mat.Dense).Copy(N)
		Wc.Slice(n, nm, 0, n).(*mat.Dense).Copy(N.T())
	}
	const steps = 2000
	W := mat.NewDense(nm, nm, nil)
	var Phi mat.Dense
	for k := 0; k <= steps; k++ {
		var Ms mat.Dense
		Ms.Scale(dt*float64(k)/steps, M)
		Phi.Exp(&Ms)
		w := 2.0
		if k == 0 || k == steps {
			w = 1
		} else if k%2 == 1 {
			w = 4
		}
		var term, tmp mat.Dense
		tmp.Mul(Phi.T(), Wc)
		term.Mul(&tmp, &Phi)
		term.Scale(w*dt/steps/3, &term)
		W.Add(W, &term)
	}
	Ad := mat.DenseCopyOf(Phi.Slice(0, n, 0, n))
	Bd := mat.DenseCopyOf(Phi.Slice(0, n, n, nm))
	Qd := mat.DenseCopyOf(W.Slice(0, n, 0, n))
	Nd := mat.DenseCopyOf(W.Slice(0, n, n, nm))
	Rd := mat.DenseCopyOf(W.Slice(n, nm, n, nm))

	X = mat.DenseCopyOf(Qd)
	for range 20000 {
		var BtX, BtXB, BtXA, G, AtX, AtXA mat.Dense
		BtX.Mul(Bd.T(), X)
		BtXA.Mul(&BtX, Ad)
		BtXB.Mul(&BtX, Bd)
		BtXB.Add(&BtXB, Rd)
		BtXA.Add(&BtXA, Nd.T())
		var inv mat.Dense
		if err := inv.Inverse(&BtXB); err != nil {
			t.Fatal(err)
		}
		K = mat.NewDense(m, n, nil)
		K.Mul(&inv, &BtXA)
		AtX.Mul(Ad.T(), X)
		AtXA.Mul(&AtX, Ad)
		G.Mul(BtXA.T(), K)
		next := mat.DenseCopyOf(Qd)
		next.Add(next, &AtXA)
		next.Sub(next, &G)
		X = mat.NewDense(n, n, nil)
		X.Add(next, next.T())
		X.Scale(0.5, X)
	}
	return K, X
}

func TestLqrd_DiscretizesCostMATLAB(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.2, 1, -0.3, -1.5, -0.4, 0.6, 0.3, -0.8, -0.9})
	B := mat.NewDense(3, 2, []float64{0, 1, 1, 0.3, -0.4, 0.7})
	Q := mat.NewDense(3, 3, []float64{4, 0.5, 0.1, 0.5, 1, -0.2, 0.1, -0.2, 2})
	R := mat.NewDense(2, 2, []float64{0.2, 0.05, 0.05, 0.4})
	N := mat.NewDense(3, 2, []float64{0.1, -0.05, 0.02, 0.1, -0.03, 0.04})
	dt := 0.5
	for _, cross := range []*mat.Dense{nil, N} {
		var opts *RiccatiOpts
		if cross != nil {
			opts = &RiccatiOpts{S: cross}
		}
		res, err := Lqrd(A, B, Q, R, dt, opts)
		if err != nil {
			t.Fatal(err)
		}
		K, X := lqrdOracle(t, A, B, Q, R, cross, dt)
		if !mat.EqualApprox(res.K, K, 1e-9) {
			t.Errorf("N=%v: K=%v want %v", cross != nil, mat.Formatted(res.K), mat.Formatted(K))
		}
		if !mat.EqualApprox(res.X, X, 1e-9) {
			t.Errorf("N=%v: X=%v want %v", cross != nil, mat.Formatted(res.X), mat.Formatted(X))
		}
	}
}

func TestLqrd_CrossWeightDims(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	_, err := Lqrd(A, B, eye(2), eye(1), 0.1, &RiccatiOpts{S: mat.NewDense(1, 2, nil)})
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("err=%v, want ErrDimensionMismatch", err)
	}
}

func TestLqrd_InvalidDt(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	Q := mat.NewDense(1, 1, []float64{1})
	R := mat.NewDense(1, 1, []float64{1})
	_, err := Lqrd(A, B, Q, R, -1, nil)
	if !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("expected ErrInvalidSampleTime, got %v", err)
	}
}

// --- Acker Tests ---

func TestAcker_DoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-1, -2}

	K, err := Acker(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, K)
	if d := matchEigenvalues(eigs, poles, 1e-10); d > 1e-10 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestAcker_3x3(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-6, -11, -6,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})
	poles := []complex128{-3, -4, -5}

	K, err := Acker(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, K)
	if d := matchEigenvalues(eigs, poles, 1e-8); d > 1e-8 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestAcker_ComplexPoles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-1 + 2i, -1 - 2i}

	K, err := Acker(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, K)
	if d := matchEigenvalues(eigs, poles, 1e-10); d > 1e-10 {
		t.Errorf("eigenvalue mismatch: max distance %e", d)
	}
}

func TestAcker_NotSISO(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(2, 2, nil)
	_, err := Acker(A, B, []complex128{-1, -2})
	if !errors.Is(err, ErrNotSISO) {
		t.Errorf("expected ErrNotSISO, got %v", err)
	}
}

func TestAcker_PoleCountError(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(2, 1, nil)
	_, err := Acker(A, B, []complex128{-1})
	if !errors.Is(err, ErrPoleCount) {
		t.Errorf("expected ErrPoleCount, got %v", err)
	}
}

func TestAcker_ConjugatePairError(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(2, 1, nil)
	_, err := Acker(A, B, []complex128{-1 + 2i, -1 + 3i})
	if !errors.Is(err, ErrConjugatePairs) {
		t.Errorf("expected ErrConjugatePairs, got %v", err)
	}
}

// --- Place Tests ---

func TestPlace_SISO_2x2(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-1, -2}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-8); d > 1e-8 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestPlace_SISO_3x3(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-6, -11, -6,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})
	poles := []complex128{-3, -1 + 1i, -1 - 1i}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-8); d > 1e-8 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestPlace_MIMO_3x2(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 1, 0,
		0, -2, 1,
		0, 0, -3,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		1, 1,
	})
	poles := []complex128{-5, -6, -7}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-6); d > 1e-6 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestPlace_MIMO_4x2_MixedPoles(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		0, 1, 0, 0,
		0, 0, 1, 0,
		0, 0, 0, 1,
		-1, -2, -3, -4,
	})
	B := mat.NewDense(4, 2, []float64{
		0, 0,
		1, 0,
		0, 0,
		0, 1,
	})
	poles := []complex128{-2, -3, -1 + 2i, -1 - 2i}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-6); d > 1e-6 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestPlace_AllComplexPoles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -1, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-2 + 3i, -2 - 3i}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-8); d > 1e-8 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestPlace_AllRealPoles(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		0, 0, 0,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})
	poles := []complex128{-1, -2, -3}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-8); d > 1e-8 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestPlace_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 3, 0, -2})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-5, -6}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-8); d > 1e-8 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func TestPlace_PoleCountError(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(2, 1, nil)
	_, err := Place(A, B, []complex128{-1})
	if !errors.Is(err, ErrPoleCount) {
		t.Errorf("expected ErrPoleCount, got %v", err)
	}
}

func TestPlace_ConjugatePairError(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(2, 1, nil)
	_, err := Place(A, B, []complex128{-1 + 2i, -1 + 3i})
	if !errors.Is(err, ErrConjugatePairs) {
		t.Errorf("expected ErrConjugatePairs, got %v", err)
	}
}

func TestPlace_Empty(t *testing.T) {
	F, err := Place(&mat.Dense{}, &mat.Dense{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	r, c := F.Dims()
	if r != 0 || c != 0 {
		t.Errorf("expected empty, got %dx%d", r, c)
	}
}

func TestPlace_AckerConsistency(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-1, -2}

	kAcker, err := Acker(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	fPlace, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}

	eigsAcker := closedLoopEig(A, B, kAcker)
	eigsPlace := closedLoopEig(A, B, fPlace)

	sort.Slice(eigsAcker, func(i, j int) bool { return real(eigsAcker[i]) < real(eigsAcker[j]) })
	sort.Slice(eigsPlace, func(i, j int) bool { return real(eigsPlace[i]) < real(eigsPlace[j]) })

	for i := range eigsAcker {
		if cmplx.Abs(eigsAcker[i]-eigsPlace[i]) > 1e-6 {
			t.Errorf("eigenvalue %d: acker=%v, place=%v", i, eigsAcker[i], eigsPlace[i])
		}
	}
}

// Aircraft lateral model (n=4, m=2) with mixed real/complex desired poles.
func TestPlace_Aircraft(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		-6.8, 0.0, -207.0, 0.0,
		1.0, 0.0, 0.0, 0.0,
		43.2, 0.0, 0.0, -4.2,
		0.0, 0.0, 1.0, 0.0,
	})
	B := mat.NewDense(4, 2, []float64{
		5.64, 0.0,
		0.0, 0.0,
		0.0, 1.18,
		0.0, 0.0,
	})
	poles := []complex128{-0.5 + 0.15i, -0.5 - 0.15i, -2.0, -0.4}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-6); d > 1e-6 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

// Unstable plant stabilization (Re(eig(A)) > 0).
func TestPlace_UnstablePlant(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		1.0, 1.0, 0.0,
		0.0, 2.0, 1.0,
		0.0, 0.0, 3.0,
	})
	B := mat.NewDense(3, 2, []float64{
		1.0, 0.0,
		0.0, 1.0,
		1.0, 1.0,
	})
	poles := []complex128{-1.0, -2.0, -3.0}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-6); d > 1e-6 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

// MIMO 4x3 pole placement with well-conditioned system.
func TestPlace_MIMO_4x3(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		-1.0, 0.5, 0.0, 0.0,
		0.0, -2.0, 0.5, 0.0,
		0.0, 0.0, -3.0, 0.5,
		0.0, 0.0, 0.0, -4.0,
	})
	B := mat.NewDense(4, 3, []float64{
		1.0, 0.0, 0.0,
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		1.0, 1.0, 1.0,
	})
	poles := []complex128{-5.0, -6.0, -7.0, -8.0}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-6); d > 1e-6 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

// Harmonic oscillator with complex conjugate pole assignment.
func TestPlace_OscillatorComplexPoles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.0, 1.0, -1.0, 0.0})
	B := mat.NewDense(2, 1, []float64{0.0, 1.0})
	poles := []complex128{-1.0 + 1.0i, -1.0 - 1.0i}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-10); d > 1e-10 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

// Discrete-time pole placement inside unit circle.
func TestPlace_Discrete(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.8, 0.3, 0.0, 0.9})
	B := mat.NewDense(2, 1, []float64{1.0, 0.5})
	poles := []complex128{0.3, 0.4}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-8); d > 1e-8 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
	for _, e := range eigs {
		if cmplx.Abs(e) >= 1 {
			t.Errorf("eigenvalue outside unit circle: %v", e)
		}
	}
}

// SISO companion form with closely spaced real poles.
func TestPlace_CompanionForm(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0.0, 1.0, 0.0,
		0.0, 0.0, 1.0,
		-6.0, -11.0, -6.0,
	})
	B := mat.NewDense(3, 1, []float64{0.0, 0.0, 1.0})
	poles := []complex128{-1.0, -1.5, -2.0}

	F, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	eigs := closedLoopEig(A, B, F)
	if d := matchEigenvalues(eigs, poles, 1e-8); d > 1e-8 {
		t.Errorf("eigenvalue mismatch: max distance %e, eigs=%v", d, eigs)
	}
}

func monicPolyFromRoots(roots []complex128) []complex128 {
	c := []complex128{1}
	for _, r := range roots {
		next := make([]complex128, len(c)+1)
		for i, v := range c {
			next[i] += v
			next[i+1] -= r * v
		}
		c = next
	}
	return c
}

// assertCharPoly compares det(sI-(A-BK)) with prod(s-p_i); coefficients stay
// well-conditioned for repeated poles where eigenvalues split by eps^(1/k).
func assertCharPoly(t *testing.T, label string, A, B, K *mat.Dense, poles []complex128) {
	t.Helper()
	eigs := closedLoopEig(A, B, K)
	if eigs == nil {
		t.Fatalf("%s: eig failed", label)
	}
	got := monicPolyFromRoots(eigs)
	want := monicPolyFromRoots(poles)
	for i := range want {
		if cmplx.Abs(got[i]-want[i]) > 1e-9*(1+cmplx.Abs(want[i])) {
			t.Errorf("%s: coef %d = %v, want %v (eig=%v)", label, i, got[i], want[i], eigs)
		}
	}
}

func TestPlace_OscillatorDistinctRealPoles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -1, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	K, err := Place(A, B, []complex128{-1, -2})
	if err != nil {
		t.Fatal(err)
	}
	// s^2 + k2 s + 1 + k1 = (s+1)(s+2)
	want := []float64{1, 3}
	for j, w := range want {
		if math.Abs(K.At(0, j)-w) > 1e-12 {
			t.Errorf("K[%d] = %v, want %v", j, K.At(0, j), w)
		}
	}
}

func TestPlace_ComplexOpenLoopContract(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		0.3, 1.0, -0.4, 0.2,
		-1.1, -0.2, 0.5, 0.0,
		0.7, 0.1, -1.5, 0.9,
		0.0, -0.6, 0.3, 0.4,
	})
	inputs := map[string]*mat.Dense{
		"MIMO": mat.NewDense(4, 2, []float64{1, 0, 0.2, 0.5, 0, 1, 0.3, 0}),
		"SISO": mat.NewDense(4, 1, []float64{0, 0, 0, 1}),
	}
	sets := [][]complex128{
		{-1, -2, -3, -4},
		{-1, -2, -1 + 2i, -1 - 2i},
		{-1 + 2i, -1, -1 - 2i, -2},
		{-1, -1, -2, -3},
		{-1, -1, -2 + 1i, -2 - 1i},
		{-2, -2, -2, -3},
		{-1 + 1i, -1 - 1i, -1 + 1i, -1 - 1i},
		{-2 + 1i, -2 - 1i, -3 + 0.5i, -3 - 0.5i},
		{0.5, -0.3, 0.2 + 0.4i, 0.2 - 0.4i},
		{0.1, 0.2, 0.3, 0.4},
	}
	for name, B := range inputs {
		for _, p := range sets {
			K, err := Place(A, B, p)
			if err != nil {
				t.Errorf("%s %v: %v", name, p, err)
				continue
			}
			assertCharPoly(t, name, A, B, K, p)
		}
	}
}

func TestPlace_RepeatedPolesComplexOpenLoop(t *testing.T) {
	A3 := mat.NewDense(3, 3, []float64{0.3, 1.0, -0.4, -1.1, -0.2, 0.5, 0.7, 0.1, -1.5})
	B3 := mat.NewDense(3, 2, []float64{1, 0, 0.2, 0.5, 0, 1})
	for _, p := range [][]complex128{{-1, -1, -3}, {-3, -1, -1}, {-1, -3, -1}, {-1, -2, -3}} {
		K, err := Place(A3, B3, p)
		if err != nil {
			t.Fatalf("%v: %v", p, err)
		}
		assertCharPoly(t, "3x3", A3, B3, K, p)
	}
	A2 := mat.NewDense(2, 2, []float64{0.3, 1.0, -1.1, -0.2})
	B2 := mat.NewDense(2, 2, []float64{1, 0, 0.2, 0.5})
	for _, p := range [][]complex128{{-1, -1}, {-1, -2}} {
		K, err := Place(A2, B2, p)
		if err != nil {
			t.Fatalf("%v: %v", p, err)
		}
		assertCharPoly(t, "2x2", A2, B2, K, p)
	}
}

func TestAcker_NonAdjacentConjugates(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		0.3, 1.0, -0.4, 0.2,
		-1.1, -0.2, 0.5, 0.0,
		0.7, 0.1, -1.5, 0.9,
		0.0, -0.6, 0.3, 0.4,
	})
	B := mat.NewDense(4, 1, []float64{0, 0, 0, 1})
	p := []complex128{-1 + 1i, -2, -1 - 1i, -3}
	K, err := Acker(A, B, p)
	if err != nil {
		t.Fatal(err)
	}
	assertCharPoly(t, "Acker", A, B, K, p)
}

func TestValidatePoles(t *testing.T) {
	if err := validatePoles([]complex128{-1, -2}); err != nil {
		t.Errorf("real poles should be valid: %v", err)
	}
	if err := validatePoles([]complex128{-1 + 2i, -1 - 2i}); err != nil {
		t.Errorf("conjugate pair should be valid: %v", err)
	}
	if err := validatePoles([]complex128{-1 + 2i, -1 + 3i}); err == nil {
		t.Error("unpaired complex poles should fail")
	}
}

func TestPlaceScaledComplexBlockMultiInput(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-30, 2, -1,
		0, -10, 1,
		0, -1, -10,
	})
	B := mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, -1, 2})
	want := []complex128{-20, complex(-15, 1), complex(-15, -1)}
	F, err := Place(A, B, want)
	if err != nil {
		t.Fatal(err)
	}
	var bf, cl mat.Dense
	bf.Mul(B, F)
	cl.Sub(A, &bf)
	var eig mat.Eigen
	if !eig.Factorize(&cl, mat.EigenNone) {
		t.Fatal("eigen failed")
	}
	got := eig.Values(nil)
	for _, w := range want {
		found := false
		for _, g := range got {
			if cmplx.Abs(g-w) < 1e-9 {
				found = true
			}
		}
		if !found {
			t.Fatalf("closed-loop poles = %v, want %v", got, want)
		}
	}
}
