package controlsys

import (
	"errors"
	"fmt"
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

func lqiTestWeights(n, m, p int) (Q, R, N *mat.Dense) {
	Q = mat.NewDense(n+p, n+p, nil)
	for i := range n + p {
		Q.Set(i, i, 1+0.2*float64(i))
	}
	Q.Set(0, n, 0.1)
	Q.Set(n, 0, 0.1)
	Q.Set(1, 2, -0.15)
	Q.Set(2, 1, -0.15)
	R = mat.NewDense(m, m, nil)
	for i := range m {
		R.Set(i, i, 1.5+0.5*float64(i))
	}
	R.Set(0, m-1, 0.2)
	R.Set(m-1, 0, 0.2)
	N = mat.NewDense(n+p, m, nil)
	N.Set(0, 0, 0.1)
	N.Set(n+p-1, m-1, -0.05)
	return Q, R, N
}

func TestLqi_AugmentedRiccatiOracle(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := obsTestPlant(t, dt, false)
		n, m, p := sys.Dims()
		Q, R, N := lqiTestWeights(n, m, p)
		res, err := Lqi(sys, Q, R, &RiccatiOpts{S: N})
		if err != nil {
			t.Fatal(err)
		}
		Aa, Ba, _, _, _ := lqgServoAugmented(sys, mat.NewDense(n+m, n+m, nil), eye(p))
		lqrResidual(t, fmt.Sprintf("dt=%v", dt), dt > 0, Aa, Ba, Q, N, R, res.X, res.K)

		var Acl mat.Dense
		Acl.Mul(Ba, res.K)
		Acl.Sub(Aa, &Acl)
		if !complexSetsApprox(res.Eig, eigValues(t, &Acl), 1e-9) {
			t.Errorf("dt=%v: Eig %v, want eig(Aa-Ba*K)", dt, res.Eig)
		}
		for _, e := range res.Eig {
			if (dt == 0 && real(e) >= 0) || (dt > 0 && cmplx.Abs(e) >= 1) {
				t.Errorf("dt=%v: unstable closed-loop eigenvalue %v", dt, e)
			}
		}

		h := 1.0
		if dt > 0 {
			h = dt
		}
		Bcl := mat.NewDense(n+p, p, nil)
		for i := range p {
			Bcl.Set(n+i, i, h)
		}
		Ccl := mat.NewDense(p, n+p, nil)
		setBlock(Ccl, 0, 0, sys.C)
		var DK mat.Dense
		DK.Mul(sys.D, res.K)
		Ccl.Sub(Ccl, &DK)
		var M, X, dc mat.Dense
		if dt > 0 {
			M.Sub(eye(n+p), &Acl)
		} else {
			M.Scale(-1, &Acl)
		}
		if err := X.Solve(&M, Bcl); err != nil {
			t.Fatal(err)
		}
		dc.Mul(Ccl, &X)
		dc.Sub(&dc, eye(p))
		assertSmall(t, fmt.Sprintf("dt=%v servo DC gain - I", dt), &dc, 1e-9)
	}
}

func TestLqi_MatchesLqgServo(t *testing.T) {
	QI := mat.NewDense(2, 2, []float64{2, 0.3, 0.3, 1})
	for _, dt := range []float64{0, 0.1} {
		sys := obsTestPlant(t, dt, false)
		n, m, p := sys.Dims()
		QXU, QWV := lqgCrossWeights(n, m, p)
		lqg, err := Lqg(sys, QXU, QWV, &LqgOpts{QI: QI})
		if err != nil {
			t.Fatal(err)
		}
		_, _, Qa, Na, R := lqgServoAugmented(sys, QXU, QI)
		res, err := Lqi(sys, Qa, R, &RiccatiOpts{S: Na})
		if err != nil {
			t.Fatal(err)
		}
		assertMatEqual(t, "Kx", subDense(res.K, 0, 0, m, n), lqg.K, 1e-10)
		assertMatEqual(t, "Ki", subDense(res.K, 0, n, m, p), lqg.Ki, 1e-10)
		assertMatEqual(t, "X", res.X, lqg.Xc, 1e-10)
	}
}

// TestLqi_MatlabLqgDocExample checks [Kx Ki] against the servo controller of
// the MATLAB lqg doc example, whose output matrix is -[Kx Ki].
func TestLqi_MatlabLqgDocExample(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, 1, 0, 0}),
		mat.NewDense(3, 2, []float64{0.3, 1, 0, 1, -0.3, 0.9}),
		mat.NewDense(1, 3, []float64{1.9, 1.3, 1}),
		mat.NewDense(1, 2, []float64{0.53, -0.61}), 0)
	if err != nil {
		t.Fatal(err)
	}
	Q := eye(4)
	for i := range 3 {
		Q.Set(i, i, 0.1)
	}
	res, err := Lqi(sys, Q, mat.NewDense(2, 2, []float64{1, 0, 0, 2}), nil)
	if err != nil {
		t.Fatal(err)
	}
	assertLqgNear(t, "K", res.K, mat.NewDense(2, 4, []float64{
		0.5388, 0.4173, 0.2481, -0.5578,
		1.492, 1.388, 1.131, -0.5869}), 1e-3)
}

func TestLqi_Errors(t *testing.T) {
	csys := obsTestPlant(t, 0, false)
	desc := obsTestPlant(t, 0, true)
	delayed := obsTestPlant(t, 0, false)
	if err := delayed.SetInputDelay([]float64{0.2, 0}); err != nil {
		t.Fatal(err)
	}
	gain, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	noOut, err := NewFromSlices(2, 2, 0, []float64{-1, 0.5, 0, -2}, []float64{1, 0, 0, 1}, nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		name string
		sys  *System
		q, r *mat.Dense
		want error
	}{
		{"Q dims", csys, eye(3), eye(2), ErrDimensionMismatch},
		{"R dims", csys, eye(5), eye(3), ErrDimensionMismatch},
		{"nil R", csys, eye(5), nil, ErrDimensionMismatch},
		{"no states", gain, eye(1), eye(1), ErrDimensionMismatch},
		{"no outputs", noOut, eye(2), eye(2), ErrDimensionMismatch},
		{"descriptor", desc, eye(5), eye(2), ErrDescriptorRiccati},
		{"delay", delayed, eye(5), eye(2), ErrDelayUnsupported},
	}
	for _, tc := range cases {
		if _, err := Lqi(tc.sys, tc.q, tc.r, nil); !errors.Is(err, tc.want) {
			t.Errorf("%s: err = %v, want %v", tc.name, err, tc.want)
		}
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
		rankB := 2
		if name == "SISO" {
			rankB = 1
		}
		for _, p := range sets {
			K, err := Place(A, B, p)
			if maxPoleMultiplicity(p) > rankB {
				if !errors.Is(err, ErrPoleMultiplicity) {
					t.Errorf("%s %v: err = %v, want ErrPoleMultiplicity (MATLAB place)", name, p, err)
				}
				continue
			}
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

// assertPlacedEig checks eig(A-BK) against poles within a backward-error
// bound scaled by the eigenvector condition number, and that the condition
// number does not exceed maxKappa (0 disables).
func assertPlacedEig(t *testing.T, label string, A, B, K *mat.Dense, poles []complex128, maxKappa float64) {
	t.Helper()
	n, _ := A.Dims()
	acl := mat.NewDense(n, n, nil)
	acl.Mul(B, K)
	acl.Sub(A, acl)
	var eig mat.Eigen
	if !eig.Factorize(acl, mat.EigenRight) {
		t.Fatalf("%s: eig failed", label)
	}
	var v mat.CDense
	eig.VectorsTo(&v)
	re := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			c := v.At(i, j)
			re.Set(i, j, real(c))
			re.Set(i+n, j+n, real(c))
			re.Set(i, j+n, -imag(c))
			re.Set(i+n, j, imag(c))
		}
	}
	kappa := mat.Cond(re, 2)
	scale := mat.Norm(A, 2) + mat.Norm(B, 2)*mat.Norm(K, 2)
	tol := 100 * eps() * kappa * scale
	if maxKappa > 0 && !(kappa <= maxKappa) {
		t.Errorf("%s: eigenvector condition %.3g > %.3g", label, kappa, maxKappa)
	}
	got := eig.Values(nil)
	if d := matchEigenvalues(got, poles, 0); !(d <= tol) {
		t.Errorf("%s: max eig error %.3g > tol %.3g (kappa=%.3g)\n got %v\nwant %v", label, d, tol, kappa, got, poles)
	}
}

func placeScaledSystem(s, aim float64) *mat.Dense {
	return mat.NewDense(3, 3, []float64{
		-3 * s, 2, -1,
		0, -s, aim,
		0, -aim, -s,
	})
}

func TestPlace_ScaledComplexPairAccuracy(t *testing.T) {
	inputs := map[string]*mat.Dense{
		"MIMO": mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, -1, 2}),
		"SISO": mat.NewDense(3, 1, []float64{1, 0.5, -1}),
	}
	for name, B := range inputs {
		for _, s := range []float64{1e-4, 1e-2, 1, 1e2, 1e3, 1e4, 1e6} {
			A := placeScaledSystem(s, 1)
			p := []complex128{complex(-2*s, 0), complex(-1.5*s, 1), complex(-1.5*s, -1)}
			K, err := Place(A, B, p)
			if err != nil {
				t.Errorf("%s s=%g: %v", name, s, err)
				continue
			}
			maxKappa := 0.0
			if name == "MIMO" {
				maxKappa = 100
			}
			assertPlacedEig(t, fmt.Sprintf("%s s=%g", name, s), A, B, K, p, maxKappa)
		}
	}
}

func TestPlace_NearScalarBlockMIMO(t *testing.T) {
	B := mat.NewDense(3, 2, []float64{1, 0.3, 0.5, 1, -1, 2})
	for _, s := range []float64{1e-4, 1e-2, 1, 1e2, 1e4, 1e6} {
		for _, aim := range []float64{0, 1e-6} {
			A := placeScaledSystem(s, aim)
			for _, p := range [][]complex128{
				{complex(-2*s, 0), complex(-1.5*s, s), complex(-1.5*s, -s)},
				{complex(-2*s, 0), complex(-1.5*s, 1), complex(-1.5*s, -1)},
				{complex(-2*s, 0), complex(-0.5*s, 0), complex(-4*s, 0)},
			} {
				K, err := Place(A, B, p)
				if err != nil {
					t.Errorf("s=%g aim=%g %v: %v", s, aim, p, err)
					continue
				}
				assertPlacedEig(t, fmt.Sprintf("s=%g aim=%g %v", s, aim, p), A, B, K, p, 0)
			}
		}
	}
}

func TestPlace_FullRankBlockNormalClosedLoop(t *testing.T) {
	B := mat.NewDense(2, 2, []float64{1, 0.3, -0.7, 2})
	for _, s := range []float64{1e-4, 1e-2, 1, 1e2, 1e4, 1e6} {
		systems := map[string]*mat.Dense{
			"scalar":       mat.NewDense(2, 2, []float64{-s, 0, 0, -s}),
			"near-scalar":  mat.NewDense(2, 2, []float64{-s, 1e-6, -1e-6, -s}),
			"oscillator":   mat.NewDense(2, 2, []float64{-s, 1, -1, -s}),
			"nonsymmetric": mat.NewDense(2, 2, []float64{0.3 * s, s, -1.1 * s, -0.2 * s}),
		}
		for name, A := range systems {
			for _, p := range [][]complex128{
				{complex(-1.5*s, 1), complex(-1.5*s, -1)},
				{complex(-1.5*s, s), complex(-1.5*s, -s)},
				{complex(-0.5*s, 0), complex(-4*s, 0)},
			} {
				K, err := Place(A, B, p)
				if err != nil {
					t.Errorf("%s s=%g %v: %v", name, s, p, err)
					continue
				}
				assertPlacedEig(t, fmt.Sprintf("%s s=%g %v", name, s, p), A, B, K, p, 10)
			}
			K, err := Place(A, B, []complex128{complex(-2*s, 0), complex(-2*s, 0)})
			if err != nil {
				t.Errorf("%s s=%g double pole: %v", name, s, err)
				continue
			}
			acl := mat.NewDense(2, 2, nil)
			acl.Mul(B, K)
			acl.Sub(A, acl)
			acl.Sub(acl, mat.NewDense(2, 2, []float64{-2 * s, 0, 0, -2 * s}))
			if r := mat.Norm(acl, 2); r > 100*eps()*(mat.Norm(A, 2)+mat.Norm(B, 2)*mat.Norm(K, 2)) {
				t.Errorf("%s s=%g double pole: A-BK not -2s·I (rank(B)=2), residual %.3g", name, s, r)
			}
		}
	}
}

func TestPlace_ScaledNonSymmetric(t *testing.T) {
	A0 := []float64{
		0.3, 1.0, -0.4, 0.2,
		-1.1, -0.2, 0.5, 0.0,
		0.7, 0.1, -1.5, 0.9,
		0.0, -0.6, 0.3, 0.4,
	}
	inputs := map[string]*mat.Dense{
		"MIMO": mat.NewDense(4, 2, []float64{1, 0, 0.2, 0.5, 0, 1, 0.3, 0}),
		"SISO": mat.NewDense(4, 1, []float64{0, 0, 0, 1}),
	}
	for name, B := range inputs {
		for _, s := range []float64{1e-4, 1e-2, 1, 1e2, 1e4, 1e6} {
			a := make([]float64, len(A0))
			for i, v := range A0 {
				a[i] = s * v
			}
			A := mat.NewDense(4, 4, a)
			p := []complex128{complex(-2*s, 0), complex(-3*s, 0), complex(-s, 1e-3*s), complex(-s, -1e-3*s)}
			K, err := Place(A, B, p)
			if err != nil {
				t.Errorf("%s s=%g: %v", name, s, err)
				continue
			}
			assertPlacedEig(t, name, A, B, K, p, 0)
		}
	}
}

func TestPlace_UncontrollableModeMIMO(t *testing.T) {
	A := placeScaledSystem(1, 0)
	B := mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, -1, 2})
	_, err := Place(A, B, []complex128{-2, -1.5 + 1i, -1.5 - 1i})
	if !errors.Is(err, ErrUncontrollable) {
		t.Fatalf("err = %v, want ErrUncontrollable", err)
	}
}

func TestPlace_UncontrollableBlockSISO(t *testing.T) {
	for _, th := range []float64{0, 0.3, 1.1, 2.5} {
		c, s := math.Cos(th), math.Sin(th)
		R := mat.NewDense(2, 2, []float64{c, -s, s, c})
		// A = R·[-1 2; 0 -3]·Rᵀ, b = R·e1: eigenvalue -3 is uncontrollable.
		var A mat.Dense
		A.Product(R, mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), R.T())
		B := mat.NewDense(2, 1, []float64{c, s})
		for _, p := range [][]complex128{{-1 + 1i, -1 - 1i}, {-2, -4}} {
			K, err := Place(&A, B, p)
			if !errors.Is(err, ErrUncontrollable) {
				t.Errorf("th=%g %v: err = %v, want ErrUncontrollable (K=%v)", th, p, err, K)
			}
		}
	}
}

func TestPlace_IllConditionedInputsNoSilentError(t *testing.T) {
	B := mat.NewDense(2, 2, []float64{1, 1, 1, 1 + 1e-10})
	for _, aim := range []float64{0, 1e-6, 1e-2} {
		A := mat.NewDense(2, 2, []float64{-1, aim, -aim, -1})
		for _, p := range [][]complex128{{-1.5 + 1i, -1.5 - 1i}, {-2, -3}} {
			K, err := Place(A, B, p)
			if err != nil {
				if !errors.Is(err, ErrUncontrollable) {
					t.Errorf("aim=%g %v: err = %v", aim, p, err)
				}
				continue
			}
			assertPlacedEig(t, fmt.Sprintf("aim=%g %v", aim, p), A, B, K, p, 0)
		}
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

// placeKappa returns the 2-norm condition number of the unit-column
// eigenvector matrix of A−BK.
func placeKappa(t *testing.T, A, B, K *mat.Dense) float64 {
	t.Helper()
	n, _ := A.Dims()
	acl := mat.NewDense(n, n, nil)
	acl.Mul(B, K)
	acl.Sub(A, acl)
	var eig mat.Eigen
	if !eig.Factorize(acl, mat.EigenRight) {
		t.Fatal("eig failed")
	}
	var v mat.CDense
	eig.VectorsTo(&v)
	re := mat.NewDense(2*n, 2*n, nil)
	for j := range n {
		var nrm float64
		for i := range n {
			nrm += real(v.At(i, j))*real(v.At(i, j)) + imag(v.At(i, j))*imag(v.At(i, j))
		}
		nrm = math.Sqrt(nrm)
		for i := range n {
			c := v.At(i, j) / complex(nrm, 0)
			re.Set(i, j, real(c))
			re.Set(i+n, j+n, real(c))
			re.Set(i, j+n, -imag(c))
			re.Set(i+n, j, imag(c))
		}
	}
	return mat.Cond(re, 2)
}

// eigenspaceDim returns dim null(A−BK−λI) from the singular values of its
// real embedding, counting those below tol·‖A−BK‖.
func eigenspaceDim(A, B, K *mat.Dense, lam complex128, tol float64) int {
	n, _ := A.Dims()
	acl := mat.NewDense(n, n, nil)
	acl.Mul(B, K)
	acl.Sub(A, acl)
	scale := mat.Norm(acl, 2) + cmplx.Abs(lam)
	re := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			a := acl.At(i, j)
			re.Set(i, j, a)
			re.Set(i+n, j+n, a)
		}
		re.Set(i, i, re.At(i, i)-real(lam))
		re.Set(i+n, i+n, re.At(i+n, i+n)-real(lam))
		re.Set(i, i+n, imag(lam))
		re.Set(i+n, i, -imag(lam))
	}
	var svd mat.SVD
	svd.Factorize(re, mat.SVDNone)
	cnt := 0
	for _, s := range svd.Values(nil) {
		if s <= tol*scale {
			cnt++
		}
	}
	return cnt / 2
}

func TestPlace_RepeatedPolesNonDefective(t *testing.T) {
	A3 := mat.NewDense(3, 3, []float64{0.3, 1.0, -0.4, -1.1, -0.2, 0.5, 0.7, 0.1, -1.5})
	B3 := mat.NewDense(3, 2, []float64{1, 0, 0.2, 0.5, 0, 1})
	A4 := mat.NewDense(4, 4, []float64{
		0.3, 1.0, -0.4, 0.2,
		-1.1, -0.2, 0.5, 0.0,
		0.7, 0.1, -1.5, 0.9,
		0.0, -0.6, 0.3, 0.4,
	})
	B42 := mat.NewDense(4, 2, []float64{1, 0, 0.2, 0.5, 0, 1, 0.3, 0})
	B43 := mat.NewDense(4, 3, []float64{1, 0, 0.4, 0.2, 0.5, 0, 0, 1, -0.7, 0.3, 0, 1})
	cases := []struct {
		name  string
		A, B  *mat.Dense
		poles []complex128
	}{
		{"3x2", A3, B3, []complex128{-1, -1, -3}},
		{"3x2 perm", A3, B3, []complex128{-1, -3, -1}},
		{"4x2 two doubles", A4, B42, []complex128{-1, -1, -2, -2}},
		{"4x2 double pair", A4, B42, []complex128{-1 + 1i, -1 - 1i, -1 + 1i, -1 - 1i}},
		{"4x3 triple", A4, B43, []complex128{-2, -2, -2, -5}},
		{"4x3 double+pair", A4, B43, []complex128{-2, -2, -1 + 3i, -1 - 3i}},
	}
	for _, c := range cases {
		K, err := Place(c.A, c.B, c.poles)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		assertPlacedEig(t, c.name, c.A, c.B, K, c.poles, 1e3)
		mult := map[complex128]int{}
		for _, p := range c.poles {
			mult[p]++
		}
		for p, k := range mult {
			if d := eigenspaceDim(c.A, c.B, K, p, 1e-10); d != k {
				t.Errorf("%s: dim null(A-BK-(%v)I) = %d, want %d (non-defective)", c.name, p, d, k)
			}
		}
	}
}

func TestPlace_MultiplicityExceedsRankB(t *testing.T) {
	A4 := mat.NewDense(4, 4, []float64{
		0.3, 1.0, -0.4, 0.2,
		-1.1, -0.2, 0.5, 0.0,
		0.7, 0.1, -1.5, 0.9,
		0.0, -0.6, 0.3, 0.4,
	})
	cases := []struct {
		name  string
		B     *mat.Dense
		poles []complex128
	}{
		{"MIMO triple", mat.NewDense(4, 2, []float64{1, 0, 0.2, 0.5, 0, 1, 0.3, 0}), []complex128{-2, -2, -2, -3}},
		{"rank-1 B double", mat.NewDense(4, 2, []float64{1, 2, 0.5, 1, 0, 0, -1, -2}), []complex128{-1, -1, -2, -3}},
		{"SISO double", mat.NewDense(4, 1, []float64{0, 0, 0, 1}), []complex128{-1, -1, -2, -3}},
		{"SISO double pair", mat.NewDense(4, 1, []float64{0, 0, 0, 1}), []complex128{-1 + 1i, -1 - 1i, -1 + 1i, -1 - 1i}},
	}
	for _, c := range cases {
		if _, err := Place(A4, c.B, c.poles); !errors.Is(err, ErrPoleMultiplicity) {
			t.Errorf("%s: err = %v, want ErrPoleMultiplicity", c.name, err)
		}
	}
	// python-control statefbk_test: triple pole with rank(B) = 2.
	Ap := mat.NewDense(4, 4, []float64{
		1.380, -0.2077, 6.715, -5.676,
		-0.5814, -4.290, 0, 0.6750,
		1.067, 4.273, -6.654, 5.893,
		0.0480, 4.273, 1.343, -2.104,
	})
	Bp := mat.NewDense(4, 2, []float64{0, 5.679, 1.136, 1.136, 0, 0, -3.146, 0})
	if _, err := Place(Ap, Bp, []complex128{-0.5, -0.5, -0.5, -8.6659}); !errors.Is(err, ErrPoleMultiplicity) {
		t.Errorf("python-control triple: err = %v, want ErrPoleMultiplicity", err)
	}
	if _, err := Place(Ap, Bp, []complex128{-0.5, -0.5, -2, -8.6659}); err != nil {
		t.Errorf("python-control double: %v", err)
	}
}

// TestPlace_RobustEigenvectorConditioning checks κ(V) of A−BK against
// scipy.signal.place_poles (method="YT", an independent robust assignment);
// a random search over admissible eigenvectors finds no lower κ.
func TestPlace_RobustEigenvectorConditioning(t *testing.T) {
	cases := []struct {
		b       []float64
		s, im   float64
		imScale bool
		scipy   float64
	}{
		{[]float64{1, 0.3, 0.5, 1, -1, 2}, 1e-4, 1, true, 22074.64},
		{[]float64{1, 0.3, 0.5, 1, -1, 2}, 1e-2, 1, true, 221.068},
		{[]float64{1, 0.3, 0.5, 1, -1, 2}, 1, 1, true, 3.6004},
		{[]float64{1, 0, 0.5, 1, -1, 2}, 1e-4, 1, false, 2.6661},
		{[]float64{1, 0, 0.5, 1, -1, 2}, 1e-2, 1, false, 2.6783},
		{[]float64{1, 0, 0.5, 1, -1, 2}, 1, 1, false, 4.5716},
		{[]float64{1, 0, 0.5, 1, -1, 2}, 1e2, 1, false, 4.0201},
		{[]float64{1, 0, 0.5, 1, -1, 2}, 1e4, 1, false, 4.0481},
	}
	for _, c := range cases {
		A := placeScaledSystem(c.s, 1)
		B := mat.NewDense(3, 2, c.b)
		im := c.im
		if c.imScale {
			im *= c.s
		}
		p := []complex128{complex(-2*c.s, 0), complex(-1.5*c.s, im), complex(-1.5*c.s, -im)}
		label := fmt.Sprintf("B=%v s=%g", c.b, c.s)
		K, err := Place(A, B, p)
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		assertPlacedEig(t, label, A, B, K, p, 1.01*c.scipy)
		Kold, err := placeSchur(A, B, p)
		if err != nil {
			t.Fatalf("%s schur: %v", label, err)
		}
		if kNew, kOld := placeKappa(t, A, B, K), placeKappa(t, A, B, Kold); !(kNew < kOld) {
			t.Errorf("%s: kappa %.4g not below Schur %.4g", label, kNew, kOld)
		}
	}
}

func TestPlace_RobustBeatsSchurConditioning(t *testing.T) {
	rng := newPlaceRNG(7)
	var sumNew, sumOld float64
	cnt := 0
	for _, dim := range [][2]int{{4, 2}, {6, 2}, {6, 3}, {8, 3}, {10, 4}} {
		n, m := dim[0], dim[1]
		for trial := range 8 {
			a := make([]float64, n*n)
			for i := range a {
				a[i] = rng()
			}
			b := make([]float64, n*m)
			for i := range b {
				b[i] = rng()
			}
			A, B := mat.NewDense(n, n, a), mat.NewDense(n, m, b)
			p := make([]complex128, 0, n)
			for len(p) < n {
				if len(p)+2 <= n && trial%2 == 0 {
					re, im := -0.5-2*math.Abs(rng()), 0.2+math.Abs(rng())
					p = append(p, complex(re, im), complex(re, -im))
				} else {
					p = append(p, complex(-0.5-2*math.Abs(rng()), 0))
				}
			}
			label := fmt.Sprintf("n=%d m=%d trial=%d", n, m, trial)
			K, err := Place(A, B, p)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			Kold, err := placeSchur(A, B, p)
			if err != nil {
				t.Fatalf("%s schur: %v", label, err)
			}
			kNew, kOld := placeKappa(t, A, B, K), placeKappa(t, A, B, Kold)
			assertPlacedEig(t, label, A, B, K, p, 0)
			if kNew > 2*kOld {
				t.Errorf("%s: kappa %.3g worse than Schur %.3g", label, kNew, kOld)
			}
			if testing.Verbose() {
				t.Logf("%s: kappa robust %.3g, Schur %.3g", label, kNew, kOld)
			}
			sumNew += math.Log10(kNew)
			sumOld += math.Log10(kOld)
			cnt++
		}
	}
	if gm := sumNew / float64(cnt); gm > sumOld/float64(cnt)-1 {
		t.Errorf("geometric-mean log10 kappa: robust %.2f, Schur %.2f; want ≥1 decade improvement", gm, sumOld/float64(cnt))
	}
}

func newPlaceRNG(seed uint64) func() float64 {
	s := seed
	return func() float64 {
		s ^= s << 13
		s ^= s >> 7
		s ^= s << 17
		return float64(s>>11)/float64(1<<53)*2 - 1
	}
}
