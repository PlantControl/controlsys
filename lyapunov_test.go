package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func lyapResidual(A, X, Q *mat.Dense) float64 {
	n, _ := A.Dims()
	var ax, xat, r mat.Dense
	ax.Mul(A, X)
	xat.Mul(X, A.T())
	r.Add(&ax, &xat)
	r.Add(&r, Q)
	return denseNorm(&r) / float64(n)
}

func dlyapResidual(A, X, Q *mat.Dense) float64 {
	n, _ := A.Dims()
	var axat, r mat.Dense
	axat.Mul(A, X)
	axat.Mul(&axat, A.T())
	r.Sub(&axat, X)
	r.Add(&r, Q)
	return denseNorm(&r) / float64(n)
}

func TestLyap_1x1(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-2})
	Q := mat.NewDense(1, 1, []float64{4})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	// A*X + X*A' + Q = 0 => -2x + -2x + 4 = 0 => x = 1
	if got := X.At(0, 0); math.Abs(got-1) > 1e-12 {
		t.Errorf("got %v, want 1", got)
	}
}

func TestLyap_2x2_Diagonal(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	// diag: x_ii = q_ii / (-2*a_ii)
	if got := X.At(0, 0); math.Abs(got-0.5) > 1e-12 {
		t.Errorf("X[0,0] = %v, want 0.5", got)
	}
	if got := X.At(1, 1); math.Abs(got-0.25) > 1e-12 {
		t.Errorf("X[1,1] = %v, want 0.25", got)
	}
}

func TestLyap_2x2_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-10
	if res := lyapResidual(A, X, Q); res > tol {
		t.Errorf("residual = %e", res)
	}
}

func TestLyap_3x3_ComplexEigs(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 1, 0,
		-1, -1, 0,
		0, 0, -2,
	})
	Q := mat.NewDense(3, 3, []float64{
		2, 1, 0,
		1, 3, 1,
		0, 1, 2,
	})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-10
	if res := lyapResidual(A, X, Q); res > tol {
		t.Errorf("residual = %e", res)
	}
	checkSymmetric(t, X, tol)
}

func TestLyap_10x10(t *testing.T) {
	n := 10
	aData := make([]float64, n*n)
	for i := range n {
		aData[i*n+i] = -float64(i+1) * 0.5
		if i+1 < n {
			aData[i*n+i+1] = 0.3
		}
	}
	A := mat.NewDense(n, n, aData)

	qData := make([]float64, n*n)
	for i := range n {
		qData[i*n+i] = 1
		if i+1 < n {
			qData[i*n+i+1] = 0.1
			qData[(i+1)*n+i] = 0.1
		}
	}
	Q := mat.NewDense(n, n, qData)

	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := float64(n) * 1e-10
	if res := lyapResidual(A, X, Q); res > tol {
		t.Errorf("residual = %e", res)
	}
	checkSymmetric(t, X, 1e-10)
}

func TestLyap_Empty(t *testing.T) {
	if _, err := Lyap(&mat.Dense{}, &mat.Dense{}, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("err = %v, want ErrDimensionMismatch", err)
	}
}

func TestLyap_Singular(t *testing.T) {
	// A has eigenvalues +1, -1: sum = 0
	A := mat.NewDense(2, 2, []float64{0, 1, 1, 0})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	_, err := Lyap(A, Q, nil)
	if !errors.Is(err, ErrSingularEquation) {
		t.Errorf("expected ErrSingularEquation, got %v", err)
	}
}

func TestLyap_NotSymmetricQ(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	Q := mat.NewDense(2, 2, []float64{1, 0.5, 0, 1})
	_, err := Lyap(A, Q, nil)
	if !errors.Is(err, ErrNotSymmetric) {
		t.Errorf("expected ErrNotSymmetric, got %v", err)
	}
}

func TestLyap_DimMismatch(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	Q := mat.NewDense(3, 3, nil)
	_, err := Lyap(A, Q, nil)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

// --- DLyap tests ---

func TestDLyap_1x1(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{0.5})
	Q := mat.NewDense(1, 1, []float64{1})
	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	// 0.25*x - x + 1 = 0 => x = 1/0.75 = 4/3
	want := 4.0 / 3.0
	if got := X.At(0, 0); math.Abs(got-want) > 1e-12 {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestDLyap_2x2_Diagonal(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0, 0, 0.8})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	want0 := 1.0 / (1 - 0.25) // 4/3
	want1 := 1.0 / (1 - 0.64) // 25/9
	if got := X.At(0, 0); math.Abs(got-want0) > 1e-12 {
		t.Errorf("X[0,0] = %v, want %v", got, want0)
	}
	if got := X.At(1, 1); math.Abs(got-want1) > 1e-12 {
		t.Errorf("X[1,1] = %v, want %v", got, want1)
	}
}

func TestDLyap_2x2_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0.3, 0, 0.8})
	Q := mat.NewDense(2, 2, []float64{1, 0.2, 0.2, 1})
	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-10
	if res := dlyapResidual(A, X, Q); res > tol {
		t.Errorf("residual = %e", res)
	}
	checkSymmetric(t, X, tol)
}

func TestDLyap_3x3_Mixed(t *testing.T) {
	// A with complex pair + real eigenvalue → mixed 2×2 and 1×1 Schur blocks
	A := mat.NewDense(3, 3, []float64{
		0.5, 0.6, 0,
		-0.6, 0.5, 0,
		0, 0, 0.3,
	})
	Q := mat.NewDense(3, 3, []float64{
		2, 0.5, 0.1,
		0.5, 3, 0.2,
		0.1, 0.2, 1,
	})
	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-10
	if res := dlyapResidual(A, X, Q); res > tol {
		t.Errorf("residual = %e", res)
	}
	checkSymmetric(t, X, tol)
}

func TestDLyap_10x10(t *testing.T) {
	n := 10
	aData := make([]float64, n*n)
	for i := range n {
		aData[i*n+i] = 0.3 + 0.05*float64(i)
		if i+1 < n {
			aData[i*n+i+1] = 0.1
		}
	}
	A := mat.NewDense(n, n, aData)

	qData := make([]float64, n*n)
	for i := range n {
		qData[i*n+i] = 1
		if i+1 < n {
			qData[i*n+i+1] = 0.1
			qData[(i+1)*n+i] = 0.1
		}
	}
	Q := mat.NewDense(n, n, qData)

	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := float64(n) * 1e-10
	if res := dlyapResidual(A, X, Q); res > tol {
		t.Errorf("residual = %e", res)
	}
	checkSymmetric(t, X, 1e-10)
}

func TestDLyap_Empty(t *testing.T) {
	if _, err := DLyap(&mat.Dense{}, &mat.Dense{}, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("err = %v, want ErrDimensionMismatch", err)
	}
}

func TestDLyap_Singular(t *testing.T) {
	// A=diag(2, 0.5): eigenvalues product = 1
	A := mat.NewDense(2, 2, []float64{2, 0, 0, 0.5})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	_, err := DLyap(A, Q, nil)
	if !errors.Is(err, ErrSingularEquation) {
		t.Errorf("expected ErrSingularEquation, got %v", err)
	}
}

func TestDLyap_AllBlockCombos(t *testing.T) {
	// 5×5 system: 2×2 complex pair, 1×1 real, 2×2 complex pair
	// This tests all block combination paths in the solver
	A := mat.NewDense(5, 5, []float64{
		0.4, 0.5, 0.1, 0, 0,
		-0.5, 0.4, 0, 0.1, 0,
		0, 0, 0.3, 0, 0,
		0, 0, 0, 0.6, 0.3,
		0, 0, 0, -0.3, 0.6,
	})
	qData := make([]float64, 25)
	for i := range 5 {
		qData[i*5+i] = 1
	}
	Q := mat.NewDense(5, 5, qData)

	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	tol := 1e-10
	if res := dlyapResidual(A, X, Q); res > tol {
		t.Errorf("residual = %e", res)
	}
	checkSymmetric(t, X, tol)
}

// Reference discrete example: A'*X*A - X = C
// Our DLyap solves A*X*A' - X + Q = 0, so use A' and Q = -C
func TestDLyap_Reference(t *testing.T) {
	At := mat.NewDense(3, 3, []float64{
		3, 1, 0,
		1, 3, 0,
		1, 0, 3,
	})
	negC := mat.NewDense(3, 3, []float64{
		-25, -24, -15,
		-24, -32, -8,
		-15, -8, -40,
	})
	X, err := DLyap(At, negC, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := [][]float64{
		{2, 1, 1},
		{1, 3, 0},
		{1, 0, 4},
	}
	tol := 1e-4
	for i := range 3 {
		for j := range 3 {
			if d := math.Abs(X.At(i, j) - want[i][j]); d > tol {
				t.Errorf("X[%d,%d] = %v, want %v", i, j, X.At(i, j), want[i][j])
			}
		}
	}
	if res := dlyapResidual(At, X, negC); res > 1e-10 {
		t.Errorf("residual = %e", res)
	}
}

// Reference continuous diagonal: A = diag(-1,-2,-3), C symmetric
func TestLyap_Reference_Continuous(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 0, 0,
		0, -2, 0,
		0, 0, -3,
	})
	Q := mat.NewDense(3, 3, []float64{
		2, 0.5, 0.3,
		0.5, 4, 0.2,
		0.3, 0.2, 6,
	})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res := lyapResidual(A, X, Q); res > 1e-11 {
		t.Errorf("residual = %e", res)
	}
	checkSymmetric(t, X, 1e-14)
	// For diagonal A: x_ii = q_ii / (-2*a_ii)
	if d := math.Abs(X.At(0, 0) - 1.0); d > 1e-12 {
		t.Errorf("X[0,0] = %v, want 1.0", X.At(0, 0))
	}
	if d := math.Abs(X.At(1, 1) - 1.0); d > 1e-12 {
		t.Errorf("X[1,1] = %v, want 1.0", X.At(1, 1))
	}
	if d := math.Abs(X.At(2, 2) - 1.0); d > 1e-12 {
		t.Errorf("X[2,2] = %v, want 1.0", X.At(2, 2))
	}
}

func checkSymmetric(t *testing.T, X *mat.Dense, tol float64) {
	t.Helper()
	r, c := X.Dims()
	for i := range r {
		for j := i + 1; j < c; j++ {
			if d := math.Abs(X.At(i, j) - X.At(j, i)); d > tol {
				t.Errorf("X[%d,%d]-X[%d,%d] = %e", i, j, j, i, d)
			}
		}
	}
}

func TestDLyapUnitCircleComplexPairIsSingular(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.6, -1.28, 0.5, 0.6})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	X, err := DLyap(A, Q, nil)
	if !errors.Is(err, ErrSingularEquation) {
		t.Fatalf("X=%v err=%v, want ErrSingularEquation (λ=0.6±0.8i on the unit circle)", X, err)
	}
}
