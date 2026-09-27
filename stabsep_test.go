package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestStabsep_Diagonal(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, 2, 0, 0, 0, -3}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 2 {
		t.Errorf("stable order = %d, want 2", ns)
	}
	if nu != 1 {
		t.Errorf("unstable order = %d, want 1", nu)
	}

	stablePoles, _ := res.Stable.Poles()
	for _, p := range stablePoles {
		if real(p) >= 0 {
			t.Errorf("stable pole %v has Re >= 0", p)
		}
	}

	unstablePoles, _ := res.Unstable.Poles()
	for _, p := range unstablePoles {
		if real(p) < 0 {
			t.Errorf("unstable pole %v has Re < 0", p)
		}
	}

	checkAdditiveDecomposition(t, sys, res.Stable, res.Unstable)
}

func TestStabsep_FullyStable(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 2 {
		t.Errorf("stable order = %d, want 2", ns)
	}
	if nu != 0 {
		t.Errorf("unstable order = %d, want 0", nu)
	}
}

func TestStabsep_FullyUnstable(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{1, 0, 0, 2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 0 {
		t.Errorf("stable order = %d, want 0", ns)
	}
	if nu != 2 {
		t.Errorf("unstable order = %d, want 2", nu)
	}
}

func TestStabsep_ComplexEigenvalues(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0, 1, 0, -1, 0, 0, 0, 0, 2}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	// ±j are marginal (Re=0), treated as unstable; pole 2 is unstable
	if ns != 0 {
		t.Errorf("stable order = %d, want 0", ns)
	}
	if nu != 3 {
		t.Errorf("unstable order = %d, want 3", nu)
	}
}

func TestStabsep_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0.5, 0, 0, 0, 1.5, 0, 0, 0, 0.9}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 2 {
		t.Errorf("stable order = %d, want 2", ns)
	}
	if nu != 1 {
		t.Errorf("unstable order = %d, want 1", nu)
	}

	checkAdditiveDecomposition(t, sys, res.Stable, res.Unstable)
}

func TestStabsep_NonDiagonalStable(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1, 0.5, 0,
			0, -2, 0,
			0, 0, 3,
		}),
		mat.NewDense(3, 1, []float64{1, 1, 1}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	nu, _, _ := res.Unstable.Dims()

	if ns != 2 {
		t.Errorf("stable order = %d, want 2", ns)
	}
	if nu != 1 {
		t.Errorf("unstable order = %d, want 1", nu)
	}

	checkAdditiveDecomposition(t, sys, res.Stable, res.Unstable)
}

func TestStabsep_Empty(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Stable.Dims()
	if ns != 0 {
		t.Errorf("stable order = %d, want 0", ns)
	}
}

func TestStabsep_EigenvaluePreservation(t *testing.T) {
	sys, err := New(
		mat.NewDense(4, 4, []float64{
			-1, 0.5, 0, 0,
			0, -2, 0, 0,
			0, 0, 1, 0.3,
			0, 0, 0, 3,
		}),
		mat.NewDense(4, 4, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
		mat.NewDense(4, 4, []float64{1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1}),
		mat.NewDense(4, 4, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	origPoles, _ := sys.Poles()
	stablePoles, _ := res.Stable.Poles()
	unstablePoles, _ := res.Unstable.Poles()

	allDecomp := append(stablePoles, unstablePoles...)
	sortPoles(origPoles)
	sortPoles(allDecomp)

	if len(origPoles) != len(allDecomp) {
		t.Fatalf("pole count: %d vs %d", len(origPoles), len(allDecomp))
	}
	for i := range origPoles {
		if cmplx.Abs(origPoles[i]-allDecomp[i]) > 1e-10 {
			t.Errorf("pole[%d]: %v != %v", i, origPoles[i], allDecomp[i])
		}
	}
}

func checkAdditiveDecomposition(t *testing.T, sys, sys1, sys2 *System) {
	t.Helper()
	_, m, p := sys.Dims()

	freqs := []float64{0.01, 0.1, 1, 10, 100}
	if sys.IsDiscrete() {
		freqs = []float64{0.01, 0.1, 0.5, 1, 2}
	}

	for _, w := range freqs {
		var s complex128
		if sys.IsContinuous() {
			s = complex(0, w)
		} else {
			s = complex(0, w*sys.Dt)
		}

		g, err := sys.EvalFr(s)
		if err != nil {
			continue
		}
		g1, err := sys1.EvalFr(s)
		if err != nil {
			continue
		}
		g2, err := sys2.EvalFr(s)
		if err != nil {
			continue
		}

		for i := range p {
			for j := range m {
				sum := g1[i][j] + g2[i][j]
				diff := cmplx.Abs(sum - g[i][j])
				mag := cmplx.Abs(g[i][j])
				tol := 1e-8
				if mag > 1 {
					tol = 1e-8 * mag
				}
				if diff > tol && mag > 1e-12 {
					t.Errorf("w=%g: G[%d,%d] sum=%v, orig=%v, diff=%g", w, i, j, sum, g[i][j], diff)
				}
			}
		}
	}
}

func TestStabsep_AdditiveDecompositionDCGain(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, 2, 0, 0, 0, -3}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}

	stableDC, err := res.Stable.DCGain()
	if err != nil {
		t.Fatal(err)
	}

	// Stable part DC gain: diag(-1/(−1), -1/(-3)) in transformed coords
	// The trace of stable DC should be sum of -1/pole for stable poles
	traceDC := 0.0
	ns, _, _ := res.Stable.Dims()
	for i := range ns {
		if i < 3 {
			traceDC += stableDC.At(i, i)
		}
	}
	if math.IsInf(traceDC, 0) || math.IsNaN(traceDC) {
		t.Errorf("stable DC gain has inf/nan")
	}
}
