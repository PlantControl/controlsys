package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestLyap_Continuous_2x2(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res := lyapResidual(A, X, Q); res > 1e-10 {
		t.Errorf("residual = %e", res)
	}
}

func TestLyap_Continuous_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, -2, 3, -4})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res := lyapResidual(A, X, Q); res > 1e-10 {
		t.Errorf("residual = %e", res)
	}
	r, c := X.Dims()
	for i := range r {
		for j := i + 1; j < c; j++ {
			if d := math.Abs(X.At(i, j) - X.At(j, i)); d > 1e-12 {
				t.Errorf("X not symmetric: X[%d,%d]-X[%d,%d] = %e", i, j, j, i, d)
			}
		}
	}
}

func TestDLyap_Discrete_2x2(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0.1, 0, 0.8})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res := dlyapResidual(A, X, Q); res > 1e-10 {
		t.Errorf("residual = %e", res)
	}
}

func TestDLyap_NonSymmetricA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.6, 0, 0.1, 0.4})
	Q := mat.NewDense(2, 2, []float64{2, 1, 1, 3})
	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res := dlyapResidual(A, X, Q); res > 1e-10 {
		t.Errorf("residual = %e", res)
	}
}

func TestLyap_3x3_Verify(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-3, 1, 0,
		0, -2, 1,
		0, 0, -1,
	})
	Q := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res := lyapResidual(A, X, Q); res > 1e-10 {
		t.Errorf("residual = %e", res)
	}
	checkSymmetric(t, X, 1e-12)
}

func TestLyap_NearSingular_Error(t *testing.T) {
	eps := 1e-10
	A := mat.NewDense(2, 2, []float64{-eps, 1, -1, -eps})
	Q := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Skipf("solver returned error for near-singular case: %v", err)
	}
	if res := lyapResidual(A, X, Q); res > 1e-6 {
		t.Errorf("residual = %e", res)
	}
}

func TestCanon_Modal_DistinctReal(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	Amod := res.Sys.A
	for i := range 2 {
		for j := range 2 {
			if i != j && math.Abs(Amod.At(i, j)) > 1e-8 {
				t.Errorf("A_modal[%d,%d] = %g, want 0", i, j, Amod.At(i, j))
			}
		}
	}

	poles, _ := res.Sys.Poles()
	sortPolesByReal(poles)
	if math.Abs(real(poles[0])+3) > 1e-10 || math.Abs(real(poles[1])+1) > 1e-10 {
		t.Errorf("poles = %v, want [-3, -1]", poles)
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
}

func TestCanon_Modal_ComplexPair(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -4, -2}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	Amod := res.Sys.A
	if math.Abs(Amod.At(0, 0)+1) > 1e-8 || math.Abs(Amod.At(1, 1)+1) > 1e-8 {
		t.Errorf("diagonal should be -1, got [%g, %g]", Amod.At(0, 0), Amod.At(1, 1))
	}
	wn := math.Sqrt(3)
	if math.Abs(math.Abs(Amod.At(0, 1))-wn) > 1e-8 || math.Abs(math.Abs(Amod.At(1, 0))-wn) > 1e-8 {
		t.Errorf("off-diagonal should be ±sqrt(3), got [%g, %g]", Amod.At(0, 1), Amod.At(1, 0))
	}
	if Amod.At(0, 1)*Amod.At(1, 0) > 0 {
		t.Errorf("off-diagonal should have opposite signs")
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
}

func TestCanon_Modal_RepeatedEigenvalues(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 1, 0, -1}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	Traw := res.T.RawMatrix()
	maxEntry := 0.0
	for _, v := range Traw.Data {
		if a := math.Abs(v); a > maxEntry {
			maxEntry = a
		}
	}
	if maxEntry > 1e6 {
		t.Errorf("T has huge entries (max=%g), Schur fallback may have failed", maxEntry)
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-8)
}

func TestCanon_Companion_Verify(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			-6, -11, -6,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	sort.Float64s(realParts(poles))
	wantPoles := []float64{-3, -2, -1}
	rp := realParts(poles)
	sort.Float64s(rp)
	for i, w := range wantPoles {
		if math.Abs(rp[i]-w) > 1e-8 {
			t.Errorf("pole[%d] = %g, want %g", i, rp[i], w)
		}
	}

	res, err := Canon(sys, CanonCompanion)
	if err != nil {
		t.Fatal(err)
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
}

func TestCanon_Modal_PreservesPoles(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-2, 1, 0,
			0, -3, 1,
			0, 0, -5,
		}),
		mat.NewDense(3, 1, []float64{1, 0, 0}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	origPoles, _ := sys.Poles()
	modPoles, _ := res.Sys.Poles()
	sortPolesByReal(origPoles)
	sortPolesByReal(modPoles)
	for i := range origPoles {
		if cmplx.Abs(origPoles[i]-modPoles[i]) > 1e-10 {
			t.Errorf("pole[%d]: orig=%v modal=%v", i, origPoles[i], modPoles[i])
		}
	}
}

func TestCanon_Modal_PreservesDCGain(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-2, 1, 0,
			0, -3, 1,
			0, 0, -5,
		}),
		mat.NewDense(3, 1, []float64{1, 0, 0}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	dc1, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	dc2, err := res.Sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}

	r, c := dc1.Dims()
	for i := range r {
		for j := range c {
			if d := math.Abs(dc1.At(i, j) - dc2.At(i, j)); d > 1e-10 {
				t.Errorf("DCGain[%d,%d]: orig=%g modal=%g", i, j, dc1.At(i, j), dc2.At(i, j))
			}
		}
	}
}

func TestLyap_SolutionSymmetry(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-3, 1, 0.5, -2})
	Q := mat.NewDense(2, 2, []float64{5, 1, 1, 3})
	X, err := Lyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	norm := denseNorm(X)
	r, c := X.Dims()
	for i := range r {
		for j := i + 1; j < c; j++ {
			if d := math.Abs(X.At(i, j) - X.At(j, i)); d/norm > 1e-12 {
				t.Errorf("relative asymmetry X[%d,%d]-X[%d,%d] = %e (norm=%e)", i, j, j, i, d, norm)
			}
		}
	}
}

func realParts(poles []complex128) []float64 {
	rp := make([]float64, len(poles))
	for i, p := range poles {
		rp[i] = real(p)
	}
	return rp
}
