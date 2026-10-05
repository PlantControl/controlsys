package controlsys

import (
	"errors"
	"fmt"
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

func obsTestPlant(t *testing.T, dt float64, withE bool) *System {
	t.Helper()
	A := mat.NewDense(3, 3, []float64{-0.8, 0.6, 0.1, -0.4, -0.3, 0.9, 0.2, -0.7, -1.1})
	B := mat.NewDense(3, 2, []float64{1, 0.2, 0.3, -0.5, -0.1, 0.8})
	C := mat.NewDense(2, 3, []float64{1, 0.4, -0.2, 0.1, -0.6, 0.9})
	D := mat.NewDense(2, 2, []float64{0.3, -0.1, 0.2, 0.5})
	if dt > 0 {
		A.Scale(0.4, A)
	}
	var E *mat.Dense
	if withE {
		E = mat.NewDense(3, 3, []float64{2, 0.3, -0.1, 0.4, 1.5, 0.2, -0.3, 0.1, 1.2})
	}
	sys, err := NewDescriptor(A, B, C, D, E, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func obsTestGain() *mat.Dense {
	return mat.NewDense(3, 2, []float64{0.4, 0.1, -0.2, 0.3, 0.5, 0.2})
}

func obsSolveComplex(M [][]complex128, Rhs [][]complex128) [][]complex128 {
	n := len(M)
	k := len(Rhs[0])
	a := make([][]complex128, n)
	for i := range n {
		a[i] = append(append([]complex128{}, M[i]...), Rhs[i]...)
	}
	for c := range n {
		piv := c
		for r := c + 1; r < n; r++ {
			if cmplx.Abs(a[r][c]) > cmplx.Abs(a[piv][c]) {
				piv = r
			}
		}
		a[c], a[piv] = a[piv], a[c]
		for r := range n {
			if r == c {
				continue
			}
			f := a[r][c] / a[c][c]
			for j := c; j < n+k; j++ {
				a[r][j] -= f * a[c][j]
			}
		}
	}
	X := make([][]complex128, n)
	for i := range n {
		X[i] = make([]complex128, k)
		for j := range k {
			X[i][j] = a[i][n+j] / a[i][i]
		}
	}
	return X
}

// obsEval returns C(sE-A)^{-1}B+D with each input column scaled by its input delay factor.
func obsEval(sys *System, s complex128) [][]complex128 {
	n, m, p := sys.Dims()
	M := make([][]complex128, n)
	Bc := make([][]complex128, n)
	for i := range n {
		M[i] = make([]complex128, n)
		Bc[i] = make([]complex128, m)
		for j := range n {
			e := 0.0
			if sys.E != nil {
				e = sys.E.At(i, j)
			} else if i == j {
				e = 1
			}
			M[i][j] = s*complex(e, 0) - complex(sys.A.At(i, j), 0)
		}
		for j := range m {
			Bc[i][j] = complex(sys.B.At(i, j), 0)
		}
	}
	X := obsSolveComplex(M, Bc)
	G := make([][]complex128, p)
	for i := range p {
		G[i] = make([]complex128, m)
		for j := range m {
			v := complex(sys.D.At(i, j), 0)
			for k := range n {
				v += complex(sys.C.At(i, k), 0) * X[k][j]
			}
			if sys.InputDelay != nil {
				if sys.Dt > 0 {
					v *= cmplx.Pow(s, complex(-sys.InputDelay[j], 0))
				} else {
					v *= cmplx.Exp(-s * complex(sys.InputDelay[j], 0))
				}
			}
			G[i][j] = v
		}
	}
	return G
}

func obsStateResponse(sys *System, s complex128) [][]complex128 {
	n, m, _ := sys.Dims()
	ident := mat.NewDense(n, n, nil)
	for i := range n {
		ident.Set(i, i, 1)
	}
	st := &System{A: sys.A, B: sys.B, C: ident, D: mat.NewDense(n, m, nil), E: sys.E, InputDelay: sys.InputDelay, Dt: sys.Dt}
	return obsEval(st, s)
}

func TestEstim_DescriptorInputDelayOracle(t *testing.T) {
	for _, tc := range []struct {
		name  string
		dt    float64
		delay []float64
		pts   []complex128
	}{
		{"continuous", 0, []float64{0.4, 0.9}, []complex128{complex(0, 0.3), complex(0.2, 1.7), complex(-0.1, 4)}},
		{"discrete", 0.1, []float64{2, 0}, []complex128{cmplx.Exp(complex(0, 0.4)), cmplx.Exp(complex(0, 2.1)), complex(0.6, -0.9)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys := obsTestPlant(t, tc.dt, true)
			if err := sys.SetInputDelay(tc.delay); err != nil {
				t.Fatal(err)
			}
			est, err := Estim(sys, obsTestGain())
			if err != nil {
				t.Fatal(err)
			}
			if est.E == nil || !mat.EqualApprox(est.E, sys.E, 0) {
				t.Fatalf("Estim dropped E: got %v", est.E)
			}
			for _, s := range tc.pts {
				G := obsEval(sys, s)
				Xu := obsStateResponse(sys, s)
				T := obsEval(est, s)
				want := append(G, Xu...)
				for i := range want {
					for j := range 2 {
						got := T[i][j] + T[i][2]*G[0][j] + T[i][3]*G[1][j]
						if cmplx.Abs(got-want[i][j]) > 1e-9 {
							t.Errorf("s=%v row %d col %d: Tu+Ty*G=%v want %v", s, i, j, got, want[i][j])
						}
					}
				}
			}
		})
	}
}

func TestEstim_RejectsUnsupportedDelays(t *testing.T) {
	L := obsTestGain()
	out := obsTestPlant(t, 0, false)
	if err := out.SetOutputDelay([]float64{0, 0.3}); err != nil {
		t.Fatal(err)
	}
	io := obsTestPlant(t, 0, false)
	if err := io.SetDelay(mat.NewDense(2, 2, []float64{0, 0.2, 0, 0})); err != nil {
		t.Fatal(err)
	}
	lft := obsTestPlant(t, 0, false)
	if err := lft.SetInternalDelay([]float64{0.5},
		mat.NewDense(3, 1, []float64{0.1, 0, 0.2}), mat.NewDense(1, 3, []float64{0.3, 0, 0}),
		mat.NewDense(2, 1, nil), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*System{"output": out, "iodelay": io, "internal": lft} {
		if _, err := Estim(sys, L); !errors.Is(err, ErrDelayUnsupported) {
			t.Errorf("%s: err=%v, want ErrDelayUnsupported", name, err)
		}
	}
}

func TestReg_DescriptorClosedLoopOracle(t *testing.T) {
	K := mat.NewDense(2, 3, []float64{0.4, -0.2, 0.1, 0.3, 0.5, -0.6})
	L := obsTestGain()
	for _, dt := range []float64{0, 0.1} {
		sys := obsTestPlant(t, dt, true)
		reg, err := Reg(sys, K, L)
		if err != nil {
			t.Fatal(err)
		}
		if reg.E == nil || !mat.EqualApprox(reg.E, sys.E, 0) {
			t.Fatalf("dt=%v: Reg dropped E", dt)
		}
		n := 3
		Ecl := mat.NewDense(2*n, 2*n, nil)
		setBlock(Ecl, 0, 0, sys.E)
		setBlock(Ecl, n, n, reg.E)
		Acl := mat.NewDense(2*n, 2*n, nil)
		setBlock(Acl, 0, 0, sys.A)
		setBlock(Acl, 0, n, mulDense(sys.B, reg.C))
		setBlock(Acl, n, 0, mulDense(reg.B, sys.C))
		var a22 mat.Dense
		a22.Add(reg.A, mulDense(reg.B, mulDense(sys.D, reg.C)))
		setBlock(Acl, n, n, &a22)
		var M mat.Dense
		if err := M.Solve(Ecl, Acl); err != nil {
			t.Fatal(err)
		}
		var got mat.Eigen
		got.Factorize(&M, mat.EigenNone)

		eigOf := func(X *mat.Dense) []complex128 {
			var Y mat.Dense
			if err := Y.Solve(sys.E, X); err != nil {
				t.Fatal(err)
			}
			var e mat.Eigen
			e.Factorize(&Y, mat.EigenNone)
			return e.Values(nil)
		}
		var ABK, ALC mat.Dense
		ABK.Sub(sys.A, mulDense(sys.B, K))
		ALC.Sub(sys.A, mulDense(L, sys.C))
		want := append(eigOf(&ABK), eigOf(&ALC)...)
		if g := got.Values(nil); !complexSetsApprox(g, want, 1e-9) {
			t.Errorf("dt=%v: closed-loop eig %v, want %v", dt, g, want)
		}
	}
}

func TestReg_RejectsDelays(t *testing.T) {
	K := mat.NewDense(2, 3, []float64{0.4, -0.2, 0.1, 0.3, 0.5, -0.6})
	sys := obsTestPlant(t, 0, false)
	if err := sys.SetInputDelay([]float64{0.2, 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := Reg(sys, K, obsTestGain()); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("err=%v, want ErrDelayUnsupported", err)
	}
}

func TestKalmd_Opts(t *testing.T) {
	sys := obsTestPlant(t, 0, false)
	sys.D.Zero()
	Qn := mat.NewDense(2, 2, []float64{1, 0.2, 0.2, 0.5})
	Rn := mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2})
	if _, err := Kalmd(sys, Qn, Rn, 0.1, &RiccatiOpts{S: mat.NewDense(3, 2, nil)}); !errors.Is(err, ErrOptionUnsupported) {
		t.Errorf("S: err=%v, want ErrOptionUnsupported", err)
	}
	ref, err := Kalmd(sys, Qn, Rn, 0.1, nil)
	if err != nil {
		t.Fatal(err)
	}
	ws := NewRiccatiWorkspace(3, 2)
	got, err := Kalmd(sys, Qn, Rn, 0.1, &RiccatiOpts{Workspace: ws})
	if err != nil {
		t.Fatal(err)
	}
	if &got.X.RawMatrix().Data[0] != &ws.xData[0] {
		t.Error("Kalmd ignored opts.Workspace")
	}
	if !mat.EqualApprox(got.K, ref.K, 1e-12) || !mat.EqualApprox(got.X, ref.X, 1e-12) {
		t.Errorf("Workspace result differs: K=%v want %v", mat.Formatted(got.K), mat.Formatted(ref.K))
	}
}

func estMul(ms ...mat.Matrix) *mat.Dense {
	out := mat.DenseCopyOf(ms[0])
	for _, m := range ms[1:] {
		var t mat.Dense
		t.Mul(out, m)
		out = &t
	}
	return out
}

func estInv(t *testing.T, m mat.Matrix) *mat.Dense {
	t.Helper()
	var inv mat.Dense
	if err := inv.Inverse(m); err != nil {
		t.Fatal(err)
	}
	return &inv
}

// estMATLABNoise forms Qbar = G Qn G', Rbar = Rn + H N + N'H' + H Qn H', Nbar = G (Qn H' + N)
// per MATLAB kalman with G = B, H = D.
func estMATLABNoise(sys *System, Qn, Rn, Nn *mat.Dense) (Qbar, Rbar, Nbar *mat.Dense) {
	G, H := sys.B, sys.D
	_, m := G.Dims()
	p, _ := H.Dims()
	if Nn == nil {
		Nn = mat.NewDense(m, p, nil)
	}
	Qbar = estMul(G, Qn, G.T())
	Rbar = mat.DenseCopyOf(Rn)
	Rbar.Add(Rbar, estMul(H, Nn))
	Rbar.Add(Rbar, estMul(Nn.T(), H.T()))
	Rbar.Add(Rbar, estMul(H, Qn, H.T()))
	QHt := estMul(Qn, H.T())
	QHt.Add(QHt, Nn)
	Nbar = estMul(G, QHt)
	return
}

func TestKalman_NoiseFeedthroughMATLAB(t *testing.T) {
	Qn := mat.NewDense(2, 2, []float64{1, 0.2, 0.2, 0.7})
	Rn := mat.NewDense(2, 2, []float64{0.5, 0.05, 0.05, 0.3})
	Nn := mat.NewDense(2, 2, []float64{0.1, -0.05, 0.04, 0.08})
	for _, dt := range []float64{0, 0.1} {
		for _, N := range []*mat.Dense{nil, Nn} {
			sys := obsTestPlant(t, dt, false)
			var opts *RiccatiOpts
			if N != nil {
				opts = &RiccatiOpts{S: N}
			}
			res, err := Kalman(sys, Qn, Rn, opts)
			if err != nil {
				t.Fatal(err)
			}
			A, C := sys.A, sys.C
			Qbar, Rbar, Nbar := estMATLABNoise(sys, Qn, Rn, N)
			var Lwant, Pwant *mat.Dense
			if dt > 0 {
				P := mat.DenseCopyOf(Qbar)
				for range 5000 {
					APCt := estMul(A, P, C.T())
					APCt.Add(APCt, Nbar)
					S := estMul(C, P, C.T())
					S.Add(S, Rbar)
					L := estMul(APCt, estInv(t, S))
					next := estMul(A, P, A.T())
					next.Add(next, Qbar)
					next.Sub(next, estMul(L, S, L.T()))
					P, Lwant = next, L
				}
				Pwant = P
			} else {
				P := res.X
				L := estMul(P, C.T())
				L.Add(L, Nbar)
				Lwant = estMul(L, estInv(t, Rbar))
				resid := estMul(A, P)
				resid.Add(resid, estMul(P, A.T()))
				resid.Add(resid, Qbar)
				resid.Sub(resid, estMul(Lwant, Rbar, Lwant.T()))
				if mat.Norm(resid, 1) > 1e-9 {
					t.Errorf("dt=%v N=%v: filter CARE residual %.3g", dt, N != nil, mat.Norm(resid, 1))
				}
				var Acl mat.Dense
				Acl.Sub(A, estMul(Lwant, C))
				var eig mat.Eigen
				eig.Factorize(&Acl, mat.EigenNone)
				for _, e := range eig.Values(nil) {
					if real(e) >= 0 {
						t.Errorf("dt=%v N=%v: A-LC not Hurwitz: %v", dt, N != nil, e)
					}
				}
				Pwant = P
			}
			if !mat.EqualApprox(res.K, Lwant, 1e-9) {
				t.Errorf("dt=%v N=%v: L=%v want %v", dt, N != nil, mat.Formatted(res.K), mat.Formatted(Lwant))
			}
			if !mat.EqualApprox(res.X, Pwant, 1e-9) {
				t.Errorf("dt=%v N=%v: P=%v want %v", dt, N != nil, mat.Formatted(res.X), mat.Formatted(Pwant))
			}
		}
	}
}

func TestKalman_CrossCovarianceDims(t *testing.T) {
	sys := obsTestPlant(t, 0, false)
	_, err := Kalman(sys, eye(2), eye(2), &RiccatiOpts{S: mat.NewDense(3, 2, nil)})
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("err=%v, want ErrDimensionMismatch", err)
	}
}

func TestLqe_CrossCovarianceMapsThroughG(t *testing.T) {
	sys := obsTestPlant(t, 0, false)
	A, C := sys.A, sys.C
	G := mat.NewDense(3, 1, []float64{0.4, -1, 0.6})
	Qn := mat.NewDense(1, 1, []float64{0.8})
	Rn := mat.NewDense(2, 2, []float64{0.5, 0.05, 0.05, 0.3})
	N := mat.NewDense(1, 2, []float64{0.2, -0.1})
	res, err := Lqe(A, G, C, Qn, Rn, &RiccatiOpts{S: N})
	if err != nil {
		t.Fatal(err)
	}
	P := res.X
	Nbar := estMul(G, N)
	L := estMul(P, C.T())
	L.Add(L, Nbar)
	L = estMul(L, estInv(t, Rn))
	resid := estMul(A, P)
	resid.Add(resid, estMul(P, A.T()))
	resid.Add(resid, estMul(G, Qn, G.T()))
	resid.Sub(resid, estMul(L, Rn, L.T()))
	if mat.Norm(resid, 1) > 1e-9 {
		t.Errorf("filter CARE residual %.3g", mat.Norm(resid, 1))
	}
	if !mat.EqualApprox(res.K, L, 1e-9) {
		t.Errorf("L=%v want %v", mat.Formatted(res.K), mat.Formatted(L))
	}
}

func TestKalmd_RejectsNoiseFeedthrough(t *testing.T) {
	sys := obsTestPlant(t, 0, false)
	_, err := Kalmd(sys, eye(2), eye(2), 0.1, nil)
	if !errors.Is(err, ErrNoiseFeedthrough) {
		t.Errorf("err=%v, want ErrNoiseFeedthrough", err)
	}
}

func TestEstimatorDesignRejectsDelays(t *testing.T) {
	setters := map[string]func(*System) error{
		"input":  func(s *System) error { return s.SetInputDelay([]float64{0.2, 0}) },
		"output": func(s *System) error { return s.SetOutputDelay([]float64{0, 0.3}) },
		"io":     func(s *System) error { return s.SetDelay(mat.NewDense(2, 2, []float64{0, 0.1, 0, 0})) },
		"internal": func(s *System) error {
			return s.SetInternalDelay([]float64{0.4},
				mat.NewDense(3, 1, []float64{0.1, 0, 0.2}), mat.NewDense(1, 3, []float64{0.3, 0, -0.1}),
				mat.NewDense(2, 1, nil), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil))
		},
	}
	for name, set := range setters {
		designs := map[string]func(*System) error{
			"Kalman": func(s *System) error { _, err := Kalman(s, eye(2), eye(2), nil); return err },
			"Kalmd": func(s *System) error {
				s.D.Zero()
				_, err := Kalmd(s, eye(2), eye(2), 0.1, nil)
				return err
			},
			"Lqg": func(s *System) error { _, err := Lqg(s, eye(5), eye(5), nil); return err },
		}
		for dname, design := range designs {
			sys := obsTestPlant(t, 0, false)
			if err := set(sys); err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			if !sys.HasDelay() {
				t.Fatalf("%s: setter produced no delay", name)
			}
			if err := design(sys); !errors.Is(err, ErrDelayUnsupported) {
				t.Errorf("%s/%s: err=%v, want ErrDelayUnsupported", dname, name, err)
			}
		}
	}
}

func assertSameFrequencyResponse(t *testing.T, label string, got, want *System, tol float64) {
	t.Helper()
	for _, w := range []float64{0.05, 0.7, 2.3} {
		s := complex(0.1, w)
		if want.IsDiscrete() {
			s = cmplx.Exp(complex(0, w*want.Dt))
		}
		g, err := got.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		h, err := want.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		for i := range h {
			for j := range h[i] {
				if cmplx.Abs(g[i][j]-h[i][j]) > tol*(1+cmplx.Abs(h[i][j])) {
					t.Errorf("%s: H(%v)[%d,%d] = %v, want %v", label, s, i, j, g[i][j], h[i][j])
				}
			}
		}
	}
}

func TestKalman_DescriptorMatchesExplicitTwin(t *testing.T) {
	Qn := mat.NewDense(2, 2, []float64{1.2, 0.3, 0.3, 0.8})
	Rn := mat.NewDense(2, 2, []float64{0.9, -0.1, -0.1, 1.1})
	Nn := mat.NewDense(2, 2, []float64{0.1, 0.05, -0.02, 0.08})
	for _, dt := range []float64{0, 0.1} {
		sys := obsTestPlant(t, dt, true)
		twin := descriptorTwin(t, sys)
		got, err := Kalman(sys, Qn, Rn, &RiccatiOpts{S: Nn})
		if err != nil {
			t.Fatalf("dt=%v: %v", dt, err)
		}
		want, err := Kalman(twin, Qn, Rn, &RiccatiOpts{S: Nn})
		if err != nil {
			t.Fatal(err)
		}
		assertMatNear(t, fmt.Sprintf("dt=%v P", dt), got.X, want.X, 1e-9)
		assertMatNear(t, fmt.Sprintf("dt=%v L = E*Lbar", dt), got.K, mulDense(sys.E, want.K), 1e-9)
		assertEigSetNear(t, fmt.Sprintf("dt=%v Eig", dt), got.Eig, want.Eig, 1e-9)

		est, err := Estim(sys, got.K)
		if err != nil {
			t.Fatal(err)
		}
		estTwin, err := Estim(twin, want.K)
		if err != nil {
			t.Fatal(err)
		}
		assertSameFrequencyResponse(t, fmt.Sprintf("dt=%v Estim", dt), est, estTwin, 1e-9)
	}
	if _, err := Kalman(obsTestPlant(t, 0, false), Qn, Rn, &RiccatiOpts{E: eye(3)}); !errors.Is(err, ErrOptionUnsupported) {
		t.Errorf("opts.E: err = %v, want ErrOptionUnsupported", err)
	}
}

func TestKalmd_DescriptorMatchesExplicitTwin(t *testing.T) {
	sys := obsTestPlant(t, 0, true)
	sys.D = mat.NewDense(2, 2, nil)
	Qn := mat.NewDense(2, 2, []float64{1.2, 0.3, 0.3, 0.8})
	Rn := mat.NewDense(2, 2, []float64{0.9, -0.1, -0.1, 1.1})
	got, err := Kalmd(sys, Qn, Rn, 0.1, nil)
	if err != nil {
		t.Fatal(err)
	}
	want, err := Kalmd(descriptorTwin(t, sys), Qn, Rn, 0.1, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertMatNear(t, "P", got.X, want.X, 1e-12)
	assertMatNear(t, "L", got.K, want.K, 1e-12)
}
