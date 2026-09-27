package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestStateSpace_RejectsImproperTF(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0}}}, // s (degree 1)
		Den: [][]float64{{1}},        // 1 (degree 0)
	}
	_, err := tf.StateSpace(nil)
	if !errors.Is(err, ErrImproperTF) {
		t.Errorf("expected ErrImproperTF, got %v", err)
	}
}

func TestStateSpace_AcceptsProperTF(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1}}},
		Den: [][]float64{{1, 1}},
	}
	_, err := tf.StateSpace(nil)
	if err != nil {
		t.Errorf("proper TF should be accepted, got %v", err)
	}
}

func TestStateSpace_AcceptsBiproperTF(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 1}}},
		Den: [][]float64{{1, 2}},
	}
	_, err := tf.StateSpace(nil)
	if err != nil {
		t.Errorf("biproper TF (equal degree) should be accepted, got %v", err)
	}
}

func TestCare_RejectsNegativeDefiniteQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{-1, 0, 0, -1})
	R := mat.NewDense(1, 1, []float64{1})

	_, err := Care(A, B, Q, R, nil)
	if !errors.Is(err, ErrNotPSD) {
		t.Errorf("expected ErrNotPSD for negative definite Q, got %v", err)
	}
}

func TestCare_RejectsIndefiniteQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, -1})
	R := mat.NewDense(1, 1, []float64{1})

	_, err := Care(A, B, Q, R, nil)
	if !errors.Is(err, ErrNotPSD) {
		t.Errorf("expected ErrNotPSD for indefinite Q, got %v", err)
	}
}

func TestCare_AcceptsPSDQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 0}) // PSD but singular
	R := mat.NewDense(1, 1, []float64{1})

	_, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Errorf("PSD Q should be accepted, got %v", err)
	}
}

func TestDare_RejectsNegativeDefiniteQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{-1, 0, 0, -1})
	R := mat.NewDense(1, 1, []float64{1})

	_, err := Dare(A, B, Q, R, nil)
	if !errors.Is(err, ErrNotPSD) {
		t.Errorf("expected ErrNotPSD for negative definite Q, got %v", err)
	}
}

func TestCare_RejectsNonPDR(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{-1})

	_, err := Care(A, B, Q, R, nil)
	if !errors.Is(err, ErrSingularR) {
		t.Errorf("expected ErrSingularR for non-PD R, got %v", err)
	}
}

func TestKalman_RejectsDescriptor(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 2})

	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	_, err := Kalman(sys, Qn, Rn, nil)
	if !errors.Is(err, ErrDescriptorRiccati) {
		t.Errorf("expected ErrDescriptorRiccati, got %v", err)
	}
}

func TestLqg_RejectsDescriptor(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	sys.E = mat.NewDense(2, 2, []float64{1, 0, 0, 2})

	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	_, err := Lqg(sys, Q, R, Qn, Rn, nil)
	if !errors.Is(err, ErrDescriptorRiccati) {
		t.Errorf("expected ErrDescriptorRiccati, got %v", err)
	}
}

func TestCare_SymmetryCheckScaleAware(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	t.Run("LargeScale", func(t *testing.T) {
		Q := mat.NewDense(2, 2, []float64{1e8, 1e-8, 0, 1e8})
		res, err := Care(A, B, Q, R, nil)
		if err != nil {
			t.Fatalf("CARE should accept near-symmetric Q at large scale: %v", err)
		}
		r := careResidual(A, B, Q, R, res.X)
		relTol := 1e-4 * mat.Norm(Q, 1)
		if r > relTol {
			t.Errorf("residual = %e, relTol = %e", r, relTol)
		}
	})

	t.Run("SmallScale", func(t *testing.T) {
		Q := mat.NewDense(2, 2, []float64{1e-8, 1e-24, 0, 1e-8})
		res, err := Care(A, B, Q, R, nil)
		if err != nil {
			t.Fatalf("CARE should accept near-symmetric Q at small scale: %v", err)
		}
		if r := careResidual(A, B, Q, R, res.X); r > 1e-6 {
			t.Errorf("residual = %e", r)
		}
	})
}

func TestCare_GeneralizedWithCrossTerm(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-2, -1, -1, -1})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 4})
	Q := mat.NewDense(2, 2, []float64{0, 0, 0, 1})
	R := mat.NewDense(2, 2, []float64{2, 0, 0, 1})
	S := mat.NewDense(2, 2, []float64{0, 0, 0, 0})

	res, err := Care(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}

	n := 2
	X := res.X
	var atx, xa, rinv_bts, xb_rinv_bts, residual mat.Dense

	atx.Mul(A.T(), X)
	xa.Mul(X, A)

	var xb mat.Dense
	xb.Mul(X, B)
	var xbPlusS mat.Dense
	xbPlusS.Add(&xb, S)

	var btx mat.Dense
	btx.Mul(B.T(), X)
	var btxPlusSt mat.Dense
	btxPlusSt.Add(&btx, S.T())

	var luR mat.LU
	luR.Factorize(R)
	luR.SolveTo(&rinv_bts, false, &btxPlusSt)

	xb_rinv_bts.Mul(&xbPlusS, &rinv_bts)

	residual.Add(&atx, &xa)
	residual.Sub(&residual, &xb_rinv_bts)
	residual.Add(&residual, Q)

	rNorm := denseNorm(&residual) / float64(n)
	if rNorm > 1e-10 {
		t.Errorf("CARE residual = %e", rNorm)
	}
	checkSymmetric(t, X, 1e-10)
}

func TestDare_GeneralizedWithCrossTerm(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-0.6, 0, -0.1, -0.4})
	B := mat.NewDense(2, 2, []float64{2, 1, 0, 1})
	Q := mat.NewDense(2, 2, []float64{2, 1, 1, 1})
	R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	S := mat.NewDense(2, 2, []float64{0, 0, 0, 0})

	res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}

	if r := dareResidualWithCrossTerm(A, B, Q, R, S, res.X); r > 1e-9 {
		t.Errorf("DARE residual = %e", r)
	}
	checkSymmetric(t, res.X, 1e-10)

	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1.0 {
			t.Errorf("unstable closed-loop eigenvalue: %v (|λ|=%v)", e, cmplx.Abs(e))
		}
	}
}

func TestDare_IllConditionedA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1.0, 0.001, 0, 0.999})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	if r := dareResidual(A, B, Q, R, res.X); r > 1e-9 {
		t.Errorf("DARE residual = %e", r)
	}

	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1.0 {
			t.Errorf("unstable closed-loop eigenvalue: %v (|λ|=%v)", e, cmplx.Abs(e))
		}
	}
}

func TestDare_SingularA_SciPyOracle(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0.2})
	B := mat.NewDense(2, 1, []float64{0.2, 1})
	Q := mat.NewDense(2, 2, []float64{2, 0.1, 0.1, 1})
	R := mat.NewDense(1, 1, []float64{1.5})
	S := mat.NewDense(2, 1, []float64{0.1, 0.2})
	wantX := mat.NewDense(2, 2, []float64{
		1.9977280744658075, 0.07198755827337247,
		0.07198755827337247, 2.7928485672615015,
	})
	wantK := mat.NewDense(1, 2, []float64{0.022719255341908287, 0.2801244172662588})

	res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	if !mat.EqualApprox(res.X, wantX, 2e-12) {
		t.Errorf("X differs from SciPy oracle:\ngot:\n%v\nwant:\n%v", mat.Formatted(res.X), mat.Formatted(wantX))
	}
	if !mat.EqualApprox(res.K, wantK, 2e-12) {
		t.Errorf("K differs from SciPy oracle:\ngot:\n%v\nwant:\n%v", mat.Formatted(res.K), mat.Formatted(wantK))
	}
	if residual := dareResidualWithCrossTerm(A, B, Q, R, S, res.X); residual > 2e-14 {
		t.Errorf("residual = %e", residual)
	}
	checkSymmetric(t, res.X, 1e-13)
	for _, eig := range res.Eig {
		if cmplx.Abs(eig) >= 1 {
			t.Errorf("non-stable closed-loop eigenvalue: %v", eig)
		}
	}
}

func TestDare_IllConditionedNonSymmetricA_SciPyOracle(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1.05, 0.4, 0, 1e-12})
	B := mat.NewDense(2, 1, []float64{1, 0.3})
	Q := mat.NewDense(2, 2, []float64{2, 0.25, 0.25, 1})
	R := mat.NewDense(1, 1, []float64{0.75})
	wantX := mat.NewDense(2, 2, []float64{
		2.64626274734184, 0.49619533232058405,
		0.49619533232058405, 1.0937886980268443,
	})
	wantK := mat.NewDense(1, 2, []float64{0.7738796586414393, 0.2948112985302894})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	if !mat.EqualApprox(res.X, wantX, 2e-12) {
		t.Errorf("X differs from SciPy oracle:\ngot:\n%v\nwant:\n%v", mat.Formatted(res.X), mat.Formatted(wantX))
	}
	if !mat.EqualApprox(res.K, wantK, 2e-12) {
		t.Errorf("K differs from SciPy oracle:\ngot:\n%v\nwant:\n%v", mat.Formatted(res.K), mat.Formatted(wantK))
	}
	if residual := dareResidual(A, B, Q, R, res.X); residual > 1e-13 {
		t.Errorf("residual = %e", residual)
	}
	checkSymmetric(t, res.X, 1e-13)
	for _, eig := range res.Eig {
		if cmplx.Abs(eig) >= 1 {
			t.Errorf("non-stable closed-loop eigenvalue: %v", eig)
		}
	}
}

func TestCare_MATLABValidated_3x3(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1.017, -0.224, 0.043,
		-0.310, -0.516, -0.119,
		-1.453, 1.800, -1.492,
	})
	B := mat.NewDense(3, 2, []float64{
		0.313, -0.165,
		-0.865, 0.628,
		-0.030, 1.093,
	})
	Q := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	if r := careResidual(A, B, Q, R, res.X); r > 1e-10 {
		t.Errorf("CARE residual = %e", r)
	}
	checkSymmetric(t, res.X, 1e-10)

	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("non-stable closed-loop eigenvalue: %v", e)
		}
	}
}

func TestLqr_ClosedLoopStability(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, 3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Lqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("LQR produced non-stable eigenvalue: %v", e)
		}
	}

	var bk, acl mat.Dense
	bk.Mul(B, res.K)
	acl.Sub(A, &bk)

	var eig mat.Eigen
	eig.Factorize(&acl, mat.EigenNone)
	vals := eig.Values(nil)
	for _, v := range vals {
		if real(v) >= 0 {
			t.Errorf("closed-loop eigenvalue from A-BK: %v", v)
		}
	}
}

func TestDlqr_ClosedLoopStability(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1.1, 0.1, 0, 0.95})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Dlqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1.0 {
			t.Errorf("DLQR produced unstable eigenvalue: %v (|λ|=%v)", e, cmplx.Abs(e))
		}
	}

	var bk, acl mat.Dense
	bk.Mul(B, res.K)
	acl.Sub(A, &bk)

	var eig mat.Eigen
	eig.Factorize(&acl, mat.EigenNone)
	vals := eig.Values(nil)
	for _, v := range vals {
		if cmplx.Abs(v) >= 1.0 {
			t.Errorf("closed-loop eigenvalue from A-BK: %v (|λ|=%v)", v, cmplx.Abs(v))
		}
	}
}

func TestCare_NearZeroQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	eps := 1e-14
	Q := mat.NewDense(2, 2, []float64{eps, 0, 0, eps})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	xNorm := mat.Norm(res.X, 1)
	if xNorm > 1e-6 {
		t.Errorf("expected near-zero X for near-zero Q on stable plant, got norm = %e", xNorm)
	}
}

func TestDare_LargeSystem_5x5(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{
		0.5, 0.1, 0.1, 0.1, 0.1,
		0.1, 0.5, 0.1, 0.1, 0.1,
		0.1, 0.1, 0.5, 0.1, 0.1,
		0.1, 0.1, 0.1, 0.5, 0.1,
		0.1, 0.1, 0.1, 0.1, 0.5,
	})
	B := mat.NewDense(5, 5, []float64{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1,
	})
	Q := mat.NewDense(5, 5, []float64{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1,
	})
	R := mat.NewDense(5, 5, []float64{
		1, 0, 0, 0, 0,
		0, 1, 0, 0, 0,
		0, 0, 1, 0, 0,
		0, 0, 0, 1, 0,
		0, 0, 0, 0, 1,
	})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	if r := dareResidual(A, B, Q, R, res.X); r > 1e-9 {
		t.Errorf("DARE residual = %e", r)
	}
	checkSymmetric(t, res.X, 1e-10)

	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1.0 {
			t.Errorf("unstable closed-loop eigenvalue: %v (|λ|=%v)", e, cmplx.Abs(e))
		}
	}

	kr, kc := res.K.Dims()
	if kr != 5 || kc != 5 {
		t.Errorf("K dims = %d×%d, want 5×5", kr, kc)
	}
}

func TestLqe_ClosedLoopStability(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, 3})
	G := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	Qn := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	Rn := mat.NewDense(1, 1, []float64{1})

	res, err := Lqe(A, G, C, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("LQE produced non-stable eigenvalue: %v", e)
		}
	}

	var lc, alc mat.Dense
	lc.Mul(res.K, C)
	alc.Sub(A, &lc)

	var eig mat.Eigen
	eig.Factorize(&alc, mat.EigenNone)
	vals := eig.Values(nil)
	for _, v := range vals {
		if real(v) >= 0 {
			t.Errorf("observer eigenvalue from A-LC: %v", v)
		}
	}
}

func TestKalman_ClosedLoopStability(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, 3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})

	res, err := Kalman(sys, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("Kalman produced non-stable eigenvalue: %v", e)
		}
	}
}

func TestCare_NearZeroQ_UnstablePlant(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 2, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	eps := 1e-14
	Q := mat.NewDense(2, 2, []float64{eps, 0, 0, eps})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	if r := careResidual(A, B, Q, R, res.X); r > 1e-10 {
		t.Errorf("CARE residual = %e", r)
	}

	xNorm := math.Abs(mat.Norm(res.X, 1))
	if xNorm < 1e-10 {
		t.Error("expected non-zero X for near-zero Q on unstable plant")
	}
}
