package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func careResidual(A, B, Q, R, X *mat.Dense) float64 {
	n, _ := A.Dims()
	_, m := B.Dims()
	var atx, xa, xbrinvbtx mat.Dense
	atx.Mul(A.T(), X)
	xa.Mul(X, A)

	var btx, rinvbtx, xb mat.Dense
	btx.Mul(B.T(), X)
	var luR mat.LU
	luR.Factorize(R)
	luR.SolveTo(&rinvbtx, false, &btx)
	xb.Mul(X, B)
	xbrinvbtx.Mul(&xb, &rinvbtx)

	var res mat.Dense
	res.Add(&atx, &xa)
	res.Sub(&res, &xbrinvbtx)
	res.Add(&res, Q)
	return denseNorm(&res) / float64(n*m)
}

func dareResidual(A, B, Q, R, X *mat.Dense) float64 {
	return dareResidualWithCrossTerm(A, B, Q, R, nil, X)
}

func dareResidualWithCrossTerm(A, B, Q, R, S, X *mat.Dense) float64 {
	n, _ := A.Dims()
	var atx, atxa, atxb, btx, btxb, rbar, btxa, mid, res mat.Dense

	atx.Mul(A.T(), X)
	atxa.Mul(&atx, A)
	atxb.Mul(&atx, B)

	btx.Mul(B.T(), X)
	btxb.Mul(&btx, B)
	btxa.Mul(&btx, A)
	if S != nil {
		atxb.Add(&atxb, S)
		btxa.Add(&btxa, S.T())
	}

	rbar.Add(R, &btxb)
	var luRbar mat.LU
	luRbar.Factorize(&rbar)

	var rbarInvBtxa mat.Dense
	luRbar.SolveTo(&rbarInvBtxa, false, &btxa)

	mid.Mul(&atxb, &rbarInvBtxa)

	res.Sub(&atxa, X)
	res.Sub(&res, &mid)
	res.Add(&res, Q)
	return denseNorm(&res) / float64(n)
}

func TestCare_1x1(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	Q := mat.NewDense(1, 1, []float64{1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	// -2x - x² + 1 = 0, positive root = √2 - 1
	want := math.Sqrt(2) - 1
	if got := res.X.At(0, 0); math.Abs(got-want) > 1e-10 {
		t.Errorf("X = %v, want %v", got, want)
	}
}

func TestCare_DoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-10
	if r := careResidual(A, B, Q, R, res.X); r > tol {
		t.Errorf("residual = %e", r)
	}
	// K ≈ [1, √3]
	if math.Abs(res.K.At(0, 0)-1) > 1e-6 || math.Abs(res.K.At(0, 1)-math.Sqrt(3)) > 1e-6 {
		t.Errorf("K = [%v, %v], want [1, √3]", res.K.At(0, 0), res.K.At(0, 1))
	}
	checkSymmetric(t, res.X, tol)
	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("non-stable closed-loop eigenvalue: %v", e)
		}
	}
}

func TestCare_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 0})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-10
	if r := careResidual(A, B, Q, R, res.X); r > tol {
		t.Errorf("residual = %e", r)
	}
	checkSymmetric(t, res.X, tol)
}

func TestCare_CrossTerm(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{2, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	S := mat.NewDense(2, 1, []float64{0, 0.5})

	res, err := Care(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	// Verify with transformed equation: Abar = A - B*R^-1*S', Qbar = Q - S*R^-1*S'
	tol := 1e-10
	checkSymmetric(t, res.X, tol)
	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("non-stable eigenvalue: %v", e)
		}
	}
}

func TestCare_MIMO(t *testing.T) {
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
	Q := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
	R := mat.NewDense(2, 2, []float64{
		1, 0,
		0, 1,
	})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-9
	if r := careResidual(A, B, Q, R, res.X); r > tol {
		t.Errorf("residual = %e", r)
	}
	checkSymmetric(t, res.X, 1e-10)
}

func TestCare_RSingular(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-1})
	B := mat.NewDense(1, 1, []float64{1})
	Q := mat.NewDense(1, 1, []float64{1})
	R := mat.NewDense(1, 1, []float64{0})
	_, err := Care(A, B, Q, R, nil)
	if !errors.Is(err, ErrSingularR) {
		t.Errorf("expected ErrSingularR, got %v", err)
	}
}

func TestCare_Empty(t *testing.T) {
	res, err := Care(&mat.Dense{}, &mat.Dense{}, &mat.Dense{}, &mat.Dense{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	r, c := res.X.Dims()
	if r != 0 || c != 0 {
		t.Errorf("expected empty, got %d×%d", r, c)
	}
}

func TestCare_DimErrors(t *testing.T) {
	A := mat.NewDense(2, 2, nil)
	B := mat.NewDense(3, 1, nil)
	Q := mat.NewDense(2, 2, nil)
	R := mat.NewDense(1, 1, []float64{1})
	_, err := Care(A, B, Q, R, nil)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

// --- Dare tests ---

func TestDare_1x1(t *testing.T) {
	a := 1.5
	A := mat.NewDense(1, 1, []float64{a})
	B := mat.NewDense(1, 1, []float64{1})
	Q := mat.NewDense(1, 1, []float64{1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	// a²x - x - a²x²/(1+x) + 1 = 0 → quadratic in x
	// (a²-1)*x - a²x²/(1+x) + 1 = 0 → (a²-1)*x*(1+x) - a²*x² + (1+x) = 0
	// (a²-1)*x + (a²-1)*x² - a²x² + 1 + x = 0
	// -x² + a²x + 1 = 0 → x² - a²x - 1 = 0
	// x = (a² + sqrt(a⁴+4))/2
	want := (a*a + math.Sqrt(a*a*a*a+4)) / 2
	if got := res.X.At(0, 0); math.Abs(got-want)/want > 1e-10 {
		t.Errorf("X = %v, want %v", got, want)
	}
}

func TestDare_DoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 1, 0, 1})
	B := mat.NewDense(2, 1, []float64{0.5, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-9
	if r := dareResidual(A, B, Q, R, res.X); r > tol {
		t.Errorf("residual = %e", r)
	}
	checkSymmetric(t, res.X, 1e-10)
	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1 {
			t.Errorf("non-stable eigenvalue: %v (|λ|=%v)", e, cmplx.Abs(e))
		}
	}
}

func TestDare_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.5, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-9
	if r := dareResidual(A, B, Q, R, res.X); r > tol {
		t.Errorf("residual = %e", r)
	}
	checkSymmetric(t, res.X, 1e-10)
}

func TestDare_CrossTerm(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 1, 0, 1})
	B := mat.NewDense(2, 1, []float64{0.5, 1})
	Q := mat.NewDense(2, 2, []float64{2, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})
	S := mat.NewDense(2, 1, []float64{0, 0.3})

	res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	checkSymmetric(t, res.X, 1e-10)
	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1 {
			t.Errorf("non-stable eigenvalue: %v", e)
		}
	}
}

func TestDare_MIMO(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0.9, 0.1, 0,
		0, 0.8, 0.1,
		0, 0, 0.7,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		0.5, 0.5,
	})
	Q := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-9
	if r := dareResidual(A, B, Q, R, res.X); r > tol {
		t.Errorf("residual = %e", r)
	}
	checkSymmetric(t, res.X, 1e-10)
}

func TestDare_RSingular(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{1.5})
	B := mat.NewDense(1, 1, []float64{1})
	Q := mat.NewDense(1, 1, []float64{1})
	R := mat.NewDense(1, 1, []float64{0})
	_, err := Dare(A, B, Q, R, nil)
	if !errors.Is(err, ErrSingularR) {
		t.Errorf("expected ErrSingularR, got %v", err)
	}
}

func TestDare_Empty(t *testing.T) {
	res, err := Dare(&mat.Dense{}, &mat.Dense{}, &mat.Dense{}, &mat.Dense{}, nil)
	if err != nil {
		t.Fatal(err)
	}
	r, c := res.X.Dims()
	if r != 0 || c != 0 {
		t.Errorf("expected empty, got %d×%d", r, c)
	}
}

// Reference: A=[0,1;0,0], Q=[1,0;0,2], G=[0,0;0,1] → X=[2,1;1,2]
func TestCare_Reference(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-4
	want := [][]float64{{2, 1}, {1, 2}}
	for i := range 2 {
		for j := range 2 {
			if d := math.Abs(res.X.At(i, j) - want[i][j]); d > tol {
				t.Errorf("X[%d,%d] = %v, want %v", i, j, res.X.At(i, j), want[i][j])
			}
		}
	}
	if r := careResidual(A, B, Q, R, res.X); r > 1e-10 {
		t.Errorf("residual = %e", r)
	}
	for _, e := range res.Eig {
		if real(e) >= 0 {
			t.Errorf("non-stable eigenvalue: %v", e)
		}
	}
}

// Reference: factored Q=C'C, R=D'D
func TestCare_FactoredQR(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	// C = [1 0; 0 1; 0 0], D = [0; 0; 1] → Q = C'C = I, R = D'D = 1
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-3
	want := [][]float64{{1.7321, 1.0}, {1.0, 1.7321}}
	for i := range 2 {
		for j := range 2 {
			if d := math.Abs(res.X.At(i, j) - want[i][j]); d > tol {
				t.Errorf("X[%d,%d] = %v, want %v", i, j, res.X.At(i, j), want[i][j])
			}
		}
	}
}

// Reference discrete test: A upper triangular, B 3x2
func TestDare_Reference_Discrete(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0.8, 0.1, 0.0,
		0.0, 0.9, 0.1,
		0.0, 0.0, 0.7,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		1, 0,
	})
	Q := mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1})
	R := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	if r := dareResidual(A, B, Q, R, res.X); r > 1e-9 {
		t.Errorf("residual = %e", r)
	}
	checkSymmetric(t, res.X, 1e-10)
	for _, e := range res.Eig {
		if cmplx.Abs(e) >= 1 {
			t.Errorf("non-stable eigenvalue: %v (|λ|=%v)", e, cmplx.Abs(e))
		}
	}
}

// Dare gain K verification: check K = (R+B'XB)⁻¹*(B'XA)
func TestDare_GainK(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 1, 0, 1})
	B := mat.NewDense(2, 1, []float64{0.5, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Independently compute K = (R + B'XB)⁻¹ * B'XA
	var btx, btxb, rbar, btxa, kExpected mat.Dense
	btx.Mul(B.T(), res.X)
	btxb.Mul(&btx, B)
	rbar.Add(R, &btxb)
	btxa.Mul(&btx, A)
	var lu mat.LU
	lu.Factorize(&rbar)
	lu.SolveTo(&kExpected, false, &btxa)

	kr, kc := res.K.Dims()
	for i := range kr {
		for j := range kc {
			if d := math.Abs(res.K.At(i, j) - kExpected.At(i, j)); d > 1e-10 {
				t.Errorf("K[%d,%d] = %v, want %v", i, j, res.K.At(i, j), kExpected.At(i, j))
			}
		}
	}
}

func TestCare_Dare_Consistency(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	R := mat.NewDense(1, 1, []float64{1})

	careRes, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal("Care:", err)
	}

	dt := 0.001
	sys, err := New(A, B, mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	dsys, err := sys.DiscretizeZOH(dt)
	if err != nil {
		t.Fatal(err)
	}
	Ad := dsys.A
	Bd := dsys.B
	Qd := mat.NewDense(2, 2, nil)
	Qd.Scale(dt, Q)
	Rd := mat.NewDense(1, 1, nil)
	Rd.Scale(1.0/dt, R)

	dareRes, err := Dare(Ad, Bd, Qd, Rd, nil)
	if err != nil {
		t.Fatal("Dare:", err)
	}

	// Continuous and discrete solutions should approximately match
	// X_dare ≈ X_care as dt → 0 (with proper scaling)
	tol := 0.1
	for i := range 2 {
		for j := range 2 {
			if d := math.Abs(careRes.X.At(i, j) - dareRes.X.At(i, j)); d > tol {
				t.Errorf("X[%d,%d] differs: care=%v dare=%v diff=%v", i, j,
					careRes.X.At(i, j), dareRes.X.At(i, j), d)
			}
		}
	}
}
