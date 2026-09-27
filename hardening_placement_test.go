package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func sortEigenvalues(eigs []complex128) {
	sort.Slice(eigs, func(i, j int) bool {
		ri, rj := real(eigs[i]), real(eigs[j])
		if ri != rj {
			return ri < rj
		}
		return imag(eigs[i]) < imag(eigs[j])
	})
}

func assertEigenvaluesMatch(t *testing.T, label string, A, B, K *mat.Dense, poles []complex128, tol float64) {
	t.Helper()
	eigs := closedLoopEig(A, B, K)
	if eigs == nil {
		t.Fatalf("%s: eigenvalue decomposition failed", label)
	}
	sortEigenvalues(eigs)
	sorted := make([]complex128, len(poles))
	copy(sorted, poles)
	sortEigenvalues(sorted)
	if len(eigs) != len(sorted) {
		t.Fatalf("%s: got %d eigenvalues, want %d", label, len(eigs), len(sorted))
	}
	for i := range eigs {
		if cmplx.Abs(eigs[i]-sorted[i]) > tol {
			t.Errorf("%s: eig[%d] = %v, want %v", label, i, eigs[i], sorted[i])
		}
	}
}

func TestAcker_ReturnShape(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-1 + 1i, -1 - 1i}

	K, err := Acker(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	r, c := K.Dims()
	if r != 1 || c != 2 {
		t.Errorf("K dims = (%d,%d), want (1,2)", r, c)
	}
}

func TestAcker_MIMO_Rejection(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	poles := []complex128{-1, -2}

	_, err := Acker(A, B, poles)
	if err == nil {
		t.Fatal("expected error for MIMO system, got nil")
	}
}

func TestPlace_ScipyExample(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{
		1.380, -0.2077, 6.715, -5.676,
		-0.5814, -4.290, 0, 0.6750,
		1.067, 4.273, -6.654, 5.893,
		0.0480, 4.273, 1.343, -2.104,
	})
	B := mat.NewDense(4, 2, []float64{
		0, 5.679,
		1.136, 1.136,
		0, 0,
		-3.146, 0,
	})
	poles := []complex128{-0.5 + 1i, -0.5 - 1i, -5.0566, -8.6659}

	K, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	assertEigenvaluesMatch(t, "scipy_example", A, B, K, poles, 1e-4)
}

func TestPlace_ComplexConjugateOnUnstable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 100, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-20 + 10i, -20 - 10i}

	K, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	assertEigenvaluesMatch(t, "unstable_complex", A, B, K, poles, 1e-8)
}

func TestAcker_ComplexConjugatePoles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-2 + 3i, -2 - 3i}

	K, err := Acker(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	assertEigenvaluesMatch(t, "acker_complex_conj", A, B, K, poles, 1e-10)
}

func TestPlace_RealPoles(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	poles := []complex128{-1, -2}

	K, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	assertEigenvaluesMatch(t, "real_poles", A, B, K, poles, 1e-8)
}

func TestCtrb_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, -2, 3, -4})
	B := mat.NewDense(2, 1, []float64{5, 7})

	got, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}

	want := mat.NewDense(2, 2, []float64{
		5, 1*5 + (-2)*7,
		7, 3*5 + (-4)*7,
	})
	assertMatNearT(t, "Ctrb_NonSymA", got, want, 1e-10)
}

func TestCtrb_Obsv_Duality_Hardening(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, -2, 3, -4})
	B := mat.NewDense(2, 2, []float64{5, 6, 7, 8})

	wc, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}

	wo, err := Obsv(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T()))
	if err != nil {
		t.Fatal(err)
	}

	woT := mat.DenseCopyOf(wo.T())
	assertMatNearT(t, "duality_hardening", wc, woT, 1e-10)
}

func TestPlace_3x1_System(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-6, -11, -6,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})
	poles := []complex128{-10, -20, -30}

	K, err := Place(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	assertEigenvaluesMatch(t, "3x1_system", A, B, K, poles, 1e-6)
}

func TestAcker_SingleState(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{2})
	B := mat.NewDense(1, 1, []float64{1})
	poles := []complex128{-5}

	K, err := Acker(A, B, poles)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(K.At(0, 0)-7) > 1e-10 {
		t.Errorf("K = %v, want [[7]]", K.At(0, 0))
	}
}
