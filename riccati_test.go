package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"strings"
	"testing"
	"time"

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
	if !errors.Is(err, ErrDimensionMismatch) || res != nil {
		t.Errorf("zero states: res = %v, err = %v, want nil, ErrDimensionMismatch", res, err)
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
	if !errors.Is(err, ErrDimensionMismatch) || res != nil {
		t.Errorf("zero states: res = %v, err = %v, want nil, ErrDimensionMismatch", res, err)
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

func descriptorRiccatiData(discrete bool) (A, B, Q, R, S, E *mat.Dense) {
	A = mat.NewDense(4, 4, []float64{
		-0.8, 0.6, 0.1, 0.3,
		-0.4, 0.2, 0.9, -0.5,
		0.2, -0.7, -1.1, 0.4,
		0.5, 0.1, -0.3, 0.6,
	})
	if discrete {
		A.Scale(0.7, A)
	}
	B = mat.NewDense(4, 2, []float64{1, 0.2, 0.3, -0.5, -0.1, 0.8, 0.4, 0.1})
	Q = mat.NewDense(4, 4, []float64{
		2, 0.3, -0.2, 0.1,
		0.3, 1.5, 0.4, 0,
		-0.2, 0.4, 1.2, 0.2,
		0.1, 0, 0.2, 1,
	})
	R = mat.NewDense(2, 2, []float64{1.5, 0.2, 0.2, 0.8})
	S = mat.NewDense(4, 2, []float64{0.1, -0.05, 0.02, 0.1, -0.08, 0.03, 0.05, 0.04})
	E = mat.NewDense(4, 4, []float64{
		2, 0.3, -0.1, 0.2,
		0.4, 1.5, 0.2, -0.3,
		-0.3, 0.1, 1.2, 0.5,
		0.2, -0.4, 0.1, 1.8,
	})
	return
}

func explicitTwin(t *testing.T, E, A, B *mat.Dense) (Ab, Bb *mat.Dense) {
	t.Helper()
	var lu mat.LU
	lu.Factorize(E)
	Ab, Bb = new(mat.Dense), new(mat.Dense)
	if err := lu.SolveTo(Ab, false, A); err != nil {
		t.Fatal(err)
	}
	if err := lu.SolveTo(Bb, false, B); err != nil {
		t.Fatal(err)
	}
	return Ab, Bb
}

func assertEigSetNear(t *testing.T, label string, got, want []complex128, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d eigenvalues, want %d", label, len(got), len(want))
	}
	used := make([]bool, len(want))
	for _, g := range got {
		best, bestDist := -1, math.Inf(1)
		for j, w := range want {
			if d := cmplx.Abs(g - w); !used[j] && d < bestDist {
				best, bestDist = j, d
			}
		}
		if bestDist > tol*(1+cmplx.Abs(g)) {
			t.Errorf("%s: eigenvalue %v has no match in %v", label, g, want)
			continue
		}
		used[best] = true
	}
}

func TestCare_Descriptor(t *testing.T) {
	A, B, Q, R, S, E := descriptorRiccatiData(false)
	res, err := Care(A, B, Q, R, &RiccatiOpts{S: S, E: E})
	if err != nil {
		t.Fatal(err)
	}
	X := res.X

	var atxe, etxa, etxb, gain, cross, resid mat.Dense
	atxe.Mul(A.T(), mulDense(X, E))
	etxa.Mul(E.T(), mulDense(X, A))
	etxb.Mul(E.T(), mulDense(X, B))
	etxb.Add(&etxb, S)
	var lu mat.LU
	lu.Factorize(R)
	if err := lu.SolveTo(&gain, false, etxb.T()); err != nil {
		t.Fatal(err)
	}
	cross.Mul(&etxb, &gain)
	resid.Add(&atxe, &etxa)
	resid.Sub(&resid, &cross)
	resid.Add(&resid, Q)
	scale := denseNorm(&atxe) + denseNorm(&cross) + denseNorm(Q)
	if r := denseNorm(&resid); r > 1e-10*scale {
		t.Errorf("generalized CARE residual %g > 1e-10*%g", r, scale)
	}
	assertMatNear(t, "K = R⁻¹(B'XE+S')", res.K, &gain, 1e-10)

	Ab, Bb := explicitTwin(t, E, A, B)
	std, err := Care(Ab, Bb, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	var etxeStd mat.Dense
	etxeStd.Mul(E.T(), mulDense(X, E))
	assertMatNear(t, "E'XE vs explicit X", &etxeStd, std.X, 1e-9)
	assertMatNear(t, "K vs explicit K", res.K, std.K, 1e-9)

	var acl mat.Dense
	acl.Mul(B, res.K)
	acl.Sub(A, &acl)
	poles, err := generalizedPoles(&acl, E, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range poles {
		if real(p) >= 0 {
			t.Errorf("closed-loop generalized eigenvalue %v not stable", p)
		}
	}
	assertEigSetNear(t, "Eig", res.Eig, poles, 1e-9)
	assertEigSetNear(t, "Eig vs explicit", res.Eig, std.Eig, 1e-9)
}

func TestDare_Descriptor(t *testing.T) {
	A, B, Q, R, S, E := descriptorRiccatiData(true)
	res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, E: E})
	if err != nil {
		t.Fatal(err)
	}
	X := res.X

	var atxa, etxe, atxbS, rbar, gain, cross, resid mat.Dense
	atxa.Mul(A.T(), mulDense(X, A))
	etxe.Mul(E.T(), mulDense(X, E))
	atxbS.Mul(A.T(), mulDense(X, B))
	atxbS.Add(&atxbS, S)
	rbar.Mul(B.T(), mulDense(X, B))
	rbar.Add(&rbar, R)
	var lu mat.LU
	lu.Factorize(&rbar)
	if err := lu.SolveTo(&gain, false, atxbS.T()); err != nil {
		t.Fatal(err)
	}
	cross.Mul(&atxbS, &gain)
	resid.Sub(&atxa, &etxe)
	resid.Sub(&resid, &cross)
	resid.Add(&resid, Q)
	scale := denseNorm(&atxa) + denseNorm(&etxe) + denseNorm(&cross) + denseNorm(Q)
	if r := denseNorm(&resid); r > 1e-10*scale {
		t.Errorf("generalized DARE residual %g > 1e-10*%g", r, scale)
	}
	assertMatNear(t, "K = (R+B'XB)⁻¹(B'XA+S')", res.K, &gain, 1e-10)

	Ab, Bb := explicitTwin(t, E, A, B)
	std, err := Dare(Ab, Bb, Q, R, &RiccatiOpts{S: S})
	if err != nil {
		t.Fatal(err)
	}
	assertMatNear(t, "E'XE vs explicit X", &etxe, std.X, 1e-9)
	assertMatNear(t, "K vs explicit K", res.K, std.K, 1e-9)

	var acl mat.Dense
	acl.Mul(B, res.K)
	acl.Sub(A, &acl)
	poles, err := generalizedPoles(&acl, E, 4)
	if err != nil {
		t.Fatal(err)
	}
	for _, p := range poles {
		if cmplx.Abs(p) >= 1 {
			t.Errorf("closed-loop generalized eigenvalue %v not inside unit circle", p)
		}
	}
	assertEigSetNear(t, "Eig", res.Eig, poles, 1e-9)
	assertEigSetNear(t, "Eig vs explicit", res.Eig, std.Eig, 1e-9)
}

func TestRiccati_DescriptorValidation(t *testing.T) {
	A, B, Q, R, _, _ := descriptorRiccatiData(false)
	singular := mat.NewDense(4, 4, []float64{1, 2, 0, 0, 2, 4, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1})
	for name, solve := range map[string]func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error){"Care": Care, "Dare": Dare} {
		if _, err := solve(A, B, Q, R, &RiccatiOpts{E: singular}); !errors.Is(err, ErrDescriptorSingular) {
			t.Errorf("%s singular E: err = %v, want ErrDescriptorSingular", name, err)
		}
		if _, err := solve(A, B, Q, R, &RiccatiOpts{E: eye(3)}); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s 3x3 E: err = %v, want ErrDimensionMismatch", name, err)
		}
		Ad := mat.DenseCopyOf(A)
		Ad.Scale(0.5, Ad)
		want, err := solve(Ad, B, Q, R, nil)
		if err != nil {
			t.Fatal(err)
		}
		got, err := solve(Ad, B, Q, R, &RiccatiOpts{E: eye(4)})
		if err != nil {
			t.Fatal(err)
		}
		if !mat.Equal(got.X, want.X) || !mat.Equal(got.K, want.K) {
			t.Errorf("%s E = I differs from E = nil", name)
		}
	}
}

func riccatiTestProblem() (A, B, Q, R *mat.Dense) {
	A = mat.NewDense(3, 3, []float64{
		0.3, 2, -1,
		-4, -0.5, 7,
		0.2, -3, 1.5,
	})
	B = mat.NewDense(3, 2, []float64{
		1, 0.5,
		0, 2,
		-1, 0.3,
	})
	L := mat.NewDense(3, 3, []float64{
		20, 0, 0,
		0.15, 1, 0,
		0.05, 0.2, 0.1,
	})
	Q = new(mat.Dense)
	Q.Mul(L, L.T())
	R = mat.NewDense(2, 2, []float64{2, 0.5, 0.5, 1})
	return
}

func TestRiccati_UndersizedWorkspace(t *testing.T) {
	A, B, Q, R := riccatiTestProblem()
	for _, ws := range []*RiccatiWorkspace{NewRiccatiWorkspace(1, 1), NewRiccatiWorkspace(3, 1), NewRiccatiWorkspace(2, 2), {}} {
		for name, solve := range map[string]func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error){"Care": Care, "Dare": Dare} {
			var err error
			func() {
				defer func() {
					if r := recover(); r != nil {
						t.Fatalf("%s panicked: %v", name, r)
					}
				}()
				_, err = solve(A, B, Q, R, &RiccatiOpts{Workspace: ws})
			}()
			if !errors.Is(err, ErrDimensionMismatch) || !strings.HasPrefix(err.Error(), name+": ") {
				t.Fatalf("%s undersized workspace: %v", name, err)
			}
		}
	}
	ws := NewRiccatiWorkspace(4, 3)
	got, err := Care(A, B, Q, R, &RiccatiOpts{Workspace: ws})
	if err != nil {
		t.Fatal(err)
	}
	if res := careResidual(A, B, Q, R, got.X); res > 1e-9 {
		t.Fatalf("oversized workspace residual %g", res)
	}
}

func TestCare_RcondOfU11(t *testing.T) {
	A, B, Q, R := riccatiTestProblem()
	Q.Scale(1e8, Q)
	res, err := Care(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	var es mat.EigenSym
	if !es.Factorize(mat.NewSymDense(3, res.X.RawMatrix().Data), false) {
		t.Fatal("eigsym")
	}
	vals := es.Values(nil)
	lo, hi := math.Inf(1), 0.0
	for _, v := range vals {
		lo = math.Min(lo, 1+v*v)
		hi = math.Max(hi, 1+v*v)
	}
	// The stable-subspace basis is [U11; U21] = [I; X](I+X²)^(-1/2)·Q with Q
	// orthogonal, so cond2(U11) is basis independent; cond1 is within n of it.
	cond2 := math.Sqrt(hi / lo)
	n := 3.0
	if res.Rcnd < 1/(n*cond2) || res.Rcnd > n/cond2 {
		t.Fatalf("Rcnd = %g, want within [%g, %g] (cond2(U11)=%g)", res.Rcnd, 1/(n*cond2), n/cond2, cond2)
	}
}

func TestRiccati_ErrorPrefixes(t *testing.T) {
	A, B, Q, R := riccatiTestProblem()
	bad := mat.NewDense(2, 2, []float64{1, 0, 0, -1})
	for name, solve := range map[string]func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error){"Care": Care, "Dare": Dare} {
		_, err := solve(A, B, Q, bad, nil)
		if !errors.Is(err, ErrSingularR) || !strings.HasPrefix(err.Error(), name+": ") {
			t.Fatalf("%s indefinite R: %v", name, err)
		}
		_, err = solve(A, B, Q, mat.NewDense(3, 3, nil), nil)
		if !errors.Is(err, ErrDimensionMismatch) || !strings.HasPrefix(err.Error(), name+": ") {
			t.Fatalf("%s R size: %v", name, err)
		}
		An := mat.DenseCopyOf(A)
		An.Set(1, 2, math.NaN())
		finishesWithin(t, 5*time.Second, func() { _, err = solve(An, B, Q, R, nil) })
		if !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), name+": ") {
			t.Fatalf("%s NaN A: %v", name, err)
		}
		_, err = solve(nil, B, Q, R, nil)
		if !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("%s nil A: %v", name, err)
		}
	}
}
