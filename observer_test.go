package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// --- Lqe Tests ---

func TestLqe_Scalar(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	G := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})

	res, err := Lqe(A, G, C, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Dual CARE: Care(-1, 1, 1, 1) -> X = sqrt(2)-1, K=X
	// L = K' = sqrt(2)-1
	want := math.Sqrt(2) - 1
	if got := res.K.At(0, 0); math.Abs(got-want) > 1e-10 {
		t.Errorf("L = %v, want %v", got, want)
	}
}

func TestLqe_DoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	G := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	Qn := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	Rn := mat.NewDense(1, 1, []float64{1})

	res, err := Lqe(A, G, C, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	lr, lc := res.K.Dims()
	if lr != 2 || lc != 1 {
		t.Fatalf("L dims = %dx%d, want 2x1", lr, lc)
	}

	// Verify A - L*C is stable
	ALC := mat.NewDense(2, 2, nil)
	ALC.Mul(res.K, C)
	ALC.Sub(A, ALC)
	var eig mat.Eigen
	eig.Factorize(ALC, mat.EigenNone)
	for _, e := range eig.Values(nil) {
		if real(e) >= 0 {
			t.Errorf("non-stable estimator eigenvalue: %v", e)
		}
	}
}

func TestLqe_NonSquareG(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	G := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 1})
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})

	res, err := Lqe(A, G, C, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}
	lr, lc := res.K.Dims()
	if lr != 2 || lc != 1 {
		t.Fatalf("L dims = %dx%d, want 2x1", lr, lc)
	}
}

func TestLqe_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	G := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	Qn := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	Rn := mat.NewDense(1, 1, []float64{1})

	res, err := Lqe(A, G, C, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	ALC := mat.NewDense(2, 2, nil)
	ALC.Mul(res.K, C)
	ALC.Sub(A, ALC)
	var eig mat.Eigen
	eig.Factorize(ALC, mat.EigenNone)
	for _, e := range eig.Values(nil) {
		if real(e) >= 0 {
			t.Errorf("non-stable estimator eigenvalue: %v", e)
		}
	}
}

func TestLqe_DimErrors(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	G := mat.NewDense(3, 2, nil) // wrong rows
	C := mat.NewDense(1, 2, nil)
	Qn := mat.NewDense(2, 2, nil)
	Rn := mat.NewDense(1, 1, []float64{1})
	_, err := Lqe(A, G, C, Qn, Rn, nil)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestLqe_Empty(t *testing.T) {
	res, err := Lqe(&mat.Dense{}, &mat.Dense{}, &mat.Dense{}, &mat.Dense{}, &mat.Dense{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	r, c := res.X.Dims()
	if r != 0 || c != 0 {
		t.Errorf("expected empty, got %dx%d", r, c)
	}
}

// --- Kalman Tests ---

func TestKalman_Continuous(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})

	res, err := Kalman(sys, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Compare with direct Lqe(A, B, C, Qn, Rn)
	res2, err := Lqe(A, B, C, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}
	lr, lc := res.K.Dims()
	for i := range lr {
		for j := range lc {
			if math.Abs(res.K.At(i, j)-res2.K.At(i, j)) > 1e-10 {
				t.Errorf("L(%d,%d): Kalman=%v, Lqe=%v", i, j, res.K.At(i, j), res2.K.At(i, j))
			}
		}
	}
}

func TestKalman_Discrete(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0.1, 0, 1})
	B := mat.NewDense(2, 1, []float64{0.005, 0.1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0.1)
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})

	res, err := Kalman(sys, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Verify estimator eigenvalues inside unit circle
	ALC := mat.NewDense(2, 2, nil)
	ALC.Mul(res.K, C)
	ALC.Sub(A, ALC)
	var eig mat.Eigen
	eig.Factorize(ALC, mat.EigenNone)
	for _, e := range eig.Values(nil) {
		if cmplx.Abs(e) >= 1 {
			t.Errorf("non-stable estimator eigenvalue: %v (|e|=%v)", e, cmplx.Abs(e))
		}
	}
}

func TestKalman_NoStates(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{1})
	sys, _ := NewGain(D, 0)
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	_, err := Kalman(sys, Qn, Rn, nil)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

// --- Kalmd Tests ---

func TestKalmd_DoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	dt := 0.01

	res, err := Kalmd(sys, Qn, Rn, dt, nil)
	if err != nil {
		t.Fatal(err)
	}

	lr, lc := res.K.Dims()
	if lr != 2 || lc != 1 {
		t.Fatalf("L dims = %dx%d, want 2x1", lr, lc)
	}

	// Estimator eigenvalues inside unit circle
	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1 {
			t.Errorf("non-stable eigenvalue: %v", e)
		}
	}
}

func TestKalmd_WrongDomain(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0.1, 0, 1})
	B := mat.NewDense(2, 1, []float64{0.005, 0.1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0.1)
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	_, err := Kalmd(sys, Qn, Rn, 0.1, nil)
	if !errors.Is(err, ErrWrongDomain) {
		t.Errorf("expected ErrWrongDomain, got %v", err)
	}
}

func TestKalmd_InvalidDt(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	_, err := Kalmd(sys, Qn, Rn, -1, nil)
	if !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("expected ErrInvalidSampleTime, got %v", err)
	}
}

// --- Estim Tests ---

func TestEstim_Dims(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)
	L := mat.NewDense(2, 1, []float64{1, 2})

	est, err := Estim(sys, L)
	if err != nil {
		t.Fatal(err)
	}

	n, mIn, pOut := est.Dims()
	if n != 2 || mIn != 2 || pOut != 3 {
		t.Errorf("dims = (%d, %d, %d), want (2, 2, 3)", n, mIn, pOut)
	}
}

func TestEstim_Values(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	L := mat.NewDense(2, 1, []float64{3, 4})

	est, err := Estim(sys, L)
	if err != nil {
		t.Fatal(err)
	}

	// Ae = A - L*C = [0,1;-2,-3] - [3;4]*[1,0] = [-3,1;-6,-3]
	wantAe := []float64{-3, 1, -6, -3}
	for i := range 2 {
		for j := range 2 {
			if math.Abs(est.A.At(i, j)-wantAe[i*2+j]) > 1e-10 {
				t.Errorf("Ae(%d,%d) = %v, want %v", i, j, est.A.At(i, j), wantAe[i*2+j])
			}
		}
	}

	// Be = [B-LD, L] = [[0;1]-[0;0], [3;4]] = [0,3; 1,4]
	wantBe := []float64{0, 3, 1, 4}
	for i := range 2 {
		for j := range 2 {
			if math.Abs(est.B.At(i, j)-wantBe[i*2+j]) > 1e-10 {
				t.Errorf("Be(%d,%d) = %v, want %v", i, j, est.B.At(i, j), wantBe[i*2+j])
			}
		}
	}

	// Ce = [C; I] = [1,0; 1,0; 0,1]
	wantCe := []float64{1, 0, 1, 0, 0, 1}
	for i := range 3 {
		for j := range 2 {
			if math.Abs(est.C.At(i, j)-wantCe[i*2+j]) > 1e-10 {
				t.Errorf("Ce(%d,%d) = %v, want %v", i, j, est.C.At(i, j), wantCe[i*2+j])
			}
		}
	}
}

func TestEstim_StableClosedLoop(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)

	G := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	Qn := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	Rn := mat.NewDense(1, 1, []float64{1})
	res, _ := Lqe(A, G, C, Qn, Rn, nil)
	L := res.K

	est, err := Estim(sys, L)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := est.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("estimator should be stable")
	}
}

func TestEstim_DimError(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(2, 1, nil)
	C := mat.NewDense(1, 2, nil)
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)
	L := mat.NewDense(3, 1, nil) // wrong rows
	_, err := Estim(sys, L)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestEstim_Discrete(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0.1, 0, 1})
	B := mat.NewDense(2, 1, []float64{0.005, 0.1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0.1)
	L := mat.NewDense(2, 1, []float64{0.5, 0.3})

	est, err := Estim(sys, L)
	if err != nil {
		t.Fatal(err)
	}
	if est.Dt != 0.1 {
		t.Errorf("Dt = %v, want 0.1", est.Dt)
	}
}

// --- Reg Tests ---

func TestReg_Dims(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	B := mat.NewDense(2, 1, []float64{1, 0})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)
	K := mat.NewDense(1, 2, []float64{1, 2})
	L := mat.NewDense(2, 1, []float64{3, 4})

	reg, err := Reg(sys, K, L)
	if err != nil {
		t.Fatal(err)
	}

	n, mIn, pOut := reg.Dims()
	if n != 2 || mIn != 1 || pOut != 1 {
		t.Errorf("dims = (%d, %d, %d), want (2, 1, 1)", n, mIn, pOut)
	}
}

func TestReg_Values(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)
	K := mat.NewDense(1, 2, []float64{1, 2})
	L := mat.NewDense(2, 1, []float64{3, 4})

	reg, err := Reg(sys, K, L)
	if err != nil {
		t.Fatal(err)
	}

	// Ar = A - B*K - L*C + L*D*K
	// BK = [0,1]*[1,2] = [0,0;1,2] -> nope, B is 2x1, K is 1x2 -> BK is 2x2
	// BK = [0;1]*[1,2] = [0,0;1,2]
	// LC = [3;4]*[1,0] = [3,0;4,0]
	// LDK = [3;4]*0*[1,2] = 0
	// Ar = [0,1;-2,-3] - [0,0;1,2] - [3,0;4,0] + 0 = [-3,1;-7,-5]
	wantAr := []float64{-3, 1, -7, -5}
	for i := range 2 {
		for j := range 2 {
			if math.Abs(reg.A.At(i, j)-wantAr[i*2+j]) > 1e-10 {
				t.Errorf("Ar(%d,%d) = %v, want %v", i, j, reg.A.At(i, j), wantAr[i*2+j])
			}
		}
	}

	// Cr = -K = [-1, -2]
	if math.Abs(reg.C.At(0, 0)+1) > 1e-10 || math.Abs(reg.C.At(0, 1)+2) > 1e-10 {
		t.Errorf("Cr = [%v, %v], want [-1, -2]", reg.C.At(0, 0), reg.C.At(0, 1))
	}
}

func TestReg_DimError(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(2, 1, nil)
	C := mat.NewDense(1, 2, nil)
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)
	K := mat.NewDense(1, 3, nil) // wrong cols
	L := mat.NewDense(2, 1, nil)
	_, err := Reg(sys, K, L)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestReg_LQG_Integration(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)

	// LQR gain
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	lqrRes, err := Lqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	K := lqrRes.K

	// Kalman gain
	Qn := mat.NewDense(1, 1, []float64{1})
	Rn := mat.NewDense(1, 1, []float64{1})
	kalRes, err := Kalman(sys, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}
	L := kalRes.K

	// Form regulator
	reg, err := Reg(sys, K, L)
	if err != nil {
		t.Fatal(err)
	}

	// Close loop
	cl, err := Feedback(sys, reg, -1)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		poles, _ := cl.Poles()
		t.Errorf("closed-loop should be stable, poles=%v", poles)
	}
}
