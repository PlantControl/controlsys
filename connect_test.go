package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func evalTF(sys *System, s complex128) [][]complex128 {
	tf, err := sys.TransferFunction(nil)
	if err != nil {
		panic(err)
	}
	return tf.TF.Eval(s)
}

func assertTFClose(t *testing.T, sys *System, s complex128, want complex128, tol float64) {
	t.Helper()
	h := evalTF(sys, s)
	got := h[0][0]
	if cmplx.Abs(got-want) > tol {
		t.Errorf("H(%v) = %v, want %v (diff=%v)", s, got, want, cmplx.Abs(got-want))
	}
}

func TestSeries_SISO(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := NewGain(mat.NewDense(1, 1, []float64{2}), 0)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 1 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (1,1,1)", n, m, p)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		want := 2.0 / s
		assertTFClose(t, result, s, want, 1e-10)
	}
}

func TestSeries_MIMO(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(2, 1, []float64{1, -1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 3 {
		t.Errorf("n = %d, want 3", n)
	}
	if m != 2 || p != 2 {
		t.Errorf("m,p = %d,%d, want 2,2", m, p)
	}

	s := complex(0, 1.0)
	h1 := evalTF(sys1, s)
	h2 := evalTF(sys2, s)
	hc := evalTF(result, s)

	for i := range 2 {
		for j := range 2 {
			want := h2[i][0]*h1[0][j] + h2[i][1]*h1[1][j]
			if cmplx.Abs(hc[i][j]-want) > 1e-8 {
				t.Errorf("H[%d][%d] at s=j: got %v, want %v", i, j, hc[i][j], want)
			}
		}
	}
}

func TestSeries_DimMismatch(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(2, 1, []float64{0, 0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_, err := Series(sys1, sys2)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestSeries_DomainMismatch(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.01,
	)
	_, err := Series(sys1, sys2)
	if !errors.Is(err, ErrDomainMismatch) {
		t.Errorf("expected ErrDomainMismatch, got %v", err)
	}
}

func TestSeries_GainSystems(t *testing.T) {
	g1, _ := NewGain(mat.NewDense(1, 1, []float64{3}), 0)
	g2, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	result, err := Series(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := result.Dims()
	if n != 0 {
		t.Errorf("n = %d, want 0", n)
	}
	if result.D.At(0, 0) != 15 {
		t.Errorf("D = %v, want 15", result.D.At(0, 0))
	}
}

func TestSeries_Roundtrip(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	for _, omega := range []float64{0.1, 0.5, 1.0, 5.0, 10.0} {
		s := complex(0, omega)
		h1 := evalTF(sys1, s)[0][0]
		h2 := evalTF(sys2, s)[0][0]
		hc := evalTF(result, s)[0][0]
		want := h2 * h1
		if cmplx.Abs(hc-want) > 1e-10 {
			t.Errorf("w=%v: got %v, want %v", omega, hc, want)
		}
	}
}

func TestParallel_SISO(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := result.Dims()
	if n != 2 {
		t.Fatalf("n = %d, want 2", n)
	}

	for _, omega := range []float64{0.1, 1.0, 10.0} {
		s := complex(0, omega)
		h1 := evalTF(sys1, s)[0][0]
		h2 := evalTF(sys2, s)[0][0]
		hc := evalTF(result, s)[0][0]
		want := h1 + h2
		if cmplx.Abs(hc-want) > 1e-10 {
			t.Errorf("w=%v: got %v, want %v", omega, hc, want)
		}
	}
}

func TestParallel_DimMismatch(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	_, err := Parallel(sys1, sys2)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestSeriesDelay_SISO(t *testing.T) {
	sys1, _ := NewFromSlices(1, 1, 1, []float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys1.Delay = mat.NewDense(1, 1, []float64{2})
	sys2, _ := NewFromSlices(1, 1, 1, []float64{0.8}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys2.Delay = mat.NewDense(1, 1, []float64{3})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.Delay == nil {
		t.Fatal("series should preserve SISO delay")
	}
	if result.Delay.At(0, 0) != 5 {
		t.Errorf("delay = %v, want 5 (2+3)", result.Delay.At(0, 0))
	}
}

func TestSeriesDelay_MIMO_DifferentPaths(t *testing.T) {
	// Paths through different intermediate channels have different total delays
	sys1, _ := NewFromSlices(1, 2, 2,
		[]float64{0.5}, []float64{1, 0}, []float64{1, 1}, []float64{0, 0, 0, 0}, 1.0)
	sys1.Delay = mat.NewDense(2, 2, []float64{
		2, 3,
		2, 3,
	})
	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 1.0)
	sys2.Delay = mat.NewDense(1, 2, []float64{1, 4})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	// d1[0,0]+d2[0,0]=3 vs d1[1,0]+d2[0,1]=6 → different → nil
	if result.Delay != nil {
		t.Error("different path delays should yield nil delay")
	}
}

func TestSeriesDelay_MIMO_NonUniform(t *testing.T) {
	// Different path delays → should return nil
	sys1, _ := NewFromSlices(1, 2, 2,
		[]float64{0.5}, []float64{1, 0}, []float64{1, 1}, []float64{0, 0, 0, 0}, 1.0)
	sys1.Delay = mat.NewDense(2, 2, []float64{
		2, 3,
		5, 3,
	})
	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 1.0)
	sys2.Delay = mat.NewDense(1, 2, []float64{1, 1})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	// d1[0,0]+d2[0,0]=3 vs d1[1,0]+d2[0,1]=6 → non-uniform → nil
	if result.Delay != nil {
		t.Error("non-uniform path delays should yield nil delay")
	}
}

func TestSeriesDelay_MIMO_ConsistentPaths(t *testing.T) {
	// Delays chosen so all paths sum to same value per (i,j)
	sys1, _ := NewFromSlices(1, 1, 2,
		[]float64{0.5}, []float64{1}, []float64{1, 1}, []float64{0, 0}, 1.0)
	sys1.Delay = mat.NewDense(2, 1, []float64{3, 5})

	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 1.0)
	sys2.Delay = mat.NewDense(1, 2, []float64{7, 5})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	// Path through k=0: d1[0,0]+d2[0,0] = 3+7 = 10
	// Path through k=1: d1[1,0]+d2[0,1] = 5+5 = 10 ✓
	if result.Delay == nil {
		t.Fatal("consistent paths should preserve delay")
	}
	if result.Delay.At(0, 0) != 10 {
		t.Errorf("delay = %v, want 10", result.Delay.At(0, 0))
	}
}

func TestFeedback_Integrator(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	result, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	poles, err := result.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if len(poles) != 1 {
		t.Fatalf("expected 1 pole, got %d", len(poles))
	}
	if math.Abs(real(poles[0])-(-1)) > 1e-10 || math.Abs(imag(poles[0])) > 1e-10 {
		t.Errorf("pole = %v, want -1", poles[0])
	}
}

func TestFeedback_SingularM(t *testing.T) {
	plant, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{-1}), 0)

	_, err := Feedback(plant, controller, -1)
	if !errors.Is(err, ErrSingularTransform) {
		t.Errorf("expected ErrSingularTransform, got %v", err)
	}
}

func TestFeedback_ZeroD(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	controller, _ := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 3 || m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (3,1,1)", n, m, p)
	}
	stable, err := result.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Log("closed-loop system may not be stable with this controller")
	}
}

func TestFeedback_WithDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{0.5})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Error("closed-loop should have internal delays")
	}
	if len(cl.LFT.Tau) != 1 || cl.LFT.Tau[0] != 0.5 {
		t.Errorf("InternalDelay = %v, want [0.5]", cl.LFT.Tau)
	}
}

func TestAppend_Basic(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (2,2,2)", n, m, p)
	}

	if result.A.At(0, 1) != 0 || result.A.At(1, 0) != 0 {
		t.Error("A should be block-diagonal")
	}
	if result.B.At(0, 1) != 0 || result.B.At(1, 0) != 0 {
		t.Error("B should be block-diagonal")
	}
	if result.C.At(0, 1) != 0 || result.C.At(1, 0) != 0 {
		t.Error("C should be block-diagonal")
	}
}

func TestAppend_WithInputOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{2}
	sys1.OutputDelay = []float64{1}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	sys2.InputDelay = []float64{0, 3}
	sys2.OutputDelay = []float64{4}

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if n != 2 || m != 3 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (2,3,2)", n, m, p)
	}
	wantIn := []float64{2, 0, 3}
	wantOut := []float64{1, 4}
	for i, w := range wantIn {
		if result.InputDelay[i] != w {
			t.Errorf("InputDelay[%d] = %v, want %v", i, result.InputDelay[i], w)
		}
	}
	for i, w := range wantOut {
		if result.OutputDelay[i] != w {
			t.Errorf("OutputDelay[%d] = %v, want %v", i, result.OutputDelay[i], w)
		}
	}
}

func TestAppend_WithMixedNilDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{5}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	wantIn := []float64{5, 0, 0}
	if len(result.InputDelay) != 3 {
		t.Fatalf("InputDelay length = %d, want 3", len(result.InputDelay))
	}
	for i, w := range wantIn {
		if result.InputDelay[i] != w {
			t.Errorf("InputDelay[%d] = %v, want %v", i, result.InputDelay[i], w)
		}
	}
	if result.OutputDelay != nil {
		t.Errorf("OutputDelay should be nil, got %v", result.OutputDelay)
	}
}

func TestSeries_WithInputOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{2}
	sys1.OutputDelay = []float64{1}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.InputDelay = []float64{3}
	sys2.OutputDelay = []float64{4}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InputDelay) != 1 || result.InputDelay[0] != 2 {
		t.Errorf("InputDelay = %v, want [2]", result.InputDelay)
	}
	if len(result.OutputDelay) != 1 || result.OutputDelay[0] != 4 {
		t.Errorf("OutputDelay = %v, want [4]", result.OutputDelay)
	}
	td := mustTotalDelay(t, result)
	if td == nil {
		t.Fatal("TotalDelay should not be nil")
	}
	if td.At(0, 0) != 2+1+3+4 {
		t.Errorf("TotalDelay = %v, want %v", td.At(0, 0), 2+1+3+4)
	}
}

func TestSeries_WithOnlyInputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{5}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InputDelay) != 1 || result.InputDelay[0] != 5 {
		t.Errorf("InputDelay = %v, want [5]", result.InputDelay)
	}
	if result.OutputDelay != nil {
		t.Errorf("OutputDelay should be nil, got %v", result.OutputDelay)
	}
}

func TestSeries_WithOnlyOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.OutputDelay = []float64{7}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.InputDelay != nil {
		t.Errorf("InputDelay should be nil, got %v", result.InputDelay)
	}
	if len(result.OutputDelay) != 1 || result.OutputDelay[0] != 7 {
		t.Errorf("OutputDelay = %v, want [7]", result.OutputDelay)
	}
}

func TestSeries_IntermediateDelayUniform(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	sys1.OutputDelay = []float64{1, 1}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	sys2.InputDelay = []float64{3, 3}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	td := mustTotalDelay(t, result)
	if td == nil {
		t.Fatal("TotalDelay should not be nil")
	}
	for j := range 2 {
		if td.At(0, j) != 4 {
			t.Errorf("TotalDelay[0][%d] = %v, want 4", j, td.At(0, j))
		}
	}
}

func TestSeries_IntermediateDelayNonUniform(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(2, 1, []float64{0, 0}),
		0,
	)
	sys1.OutputDelay = []float64{1, 3}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)
	sys2.InputDelay = []float64{2, 4}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("non-uniform intermediate delays should produce InternalDelay")
	}
	if len(result.LFT.Tau) != 4 {
		t.Errorf("InternalDelay count = %d, want 4 (2 output + 2 input)", len(result.LFT.Tau))
	}
}

func TestParallel_WithMatchingInputOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{2}
	sys1.OutputDelay = []float64{3}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.InputDelay = []float64{2}
	sys2.OutputDelay = []float64{3}

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.InputDelay) != 1 || result.InputDelay[0] != 2 {
		t.Errorf("InputDelay = %v, want [2]", result.InputDelay)
	}
	if len(result.OutputDelay) != 1 || result.OutputDelay[0] != 3 {
		t.Errorf("OutputDelay = %v, want [3]", result.OutputDelay)
	}
}

func TestParallel_WithMismatchedInputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{1}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.InputDelay = []float64{3}

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("mismatched input delays should produce InternalDelay")
	}
	if len(result.LFT.Tau) != 1 {
		t.Errorf("InternalDelay count = %d, want 1 (only difference goes internal)", len(result.LFT.Tau))
	}
	if result.InputDelay == nil || result.InputDelay[0] != 1 {
		t.Errorf("common InputDelay should stay external: got %v, want [1]", result.InputDelay)
	}
}

func TestParallel_WithMismatchedOutputDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.OutputDelay = []float64{2}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.OutputDelay = []float64{5}

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("mismatched output delays should produce InternalDelay")
	}
	if len(result.LFT.Tau) != 1 {
		t.Errorf("InternalDelay count = %d, want 1 (only difference goes internal)", len(result.LFT.Tau))
	}
	if result.OutputDelay == nil || result.OutputDelay[0] != 2 {
		t.Errorf("common OutputDelay should stay external: got %v, want [2]", result.OutputDelay)
	}
}

func TestAppend_WithDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.Delay = mat.NewDense(1, 1, []float64{0.5})

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.Delay = mat.NewDense(1, 1, []float64{1.0})

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.Delay == nil {
		t.Fatal("expected delay matrix")
	}
	r, c := result.Delay.Dims()
	if r != 2 || c != 2 {
		t.Fatalf("delay dims = %dx%d, want 2x2", r, c)
	}
	if result.Delay.At(0, 0) != 0.5 {
		t.Errorf("delay[0][0] = %v, want 0.5", result.Delay.At(0, 0))
	}
	if result.Delay.At(1, 1) != 1.0 {
		t.Errorf("delay[1][1] = %v, want 1.0", result.Delay.At(1, 1))
	}
	if result.Delay.At(0, 1) != 0 || result.Delay.At(1, 0) != 0 {
		t.Error("off-diagonal delays should be 0")
	}
}

func TestFeedbackApprox_DiscreteInputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{3})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0.1)

	cl, err := Feedback(plant, controller, -1, WithApproximatedDelays())
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}

	plantAbs, _ := plant.AbsorbDelay()
	clManual, _ := Feedback(plantAbs, controller, -1)

	nCl, _, _ := cl.Dims()
	nManual, _, _ := clManual.Dims()
	if nCl != nManual {
		t.Errorf("state count %d != manual %d", nCl, nManual)
	}

	steps := 15
	u := mat.NewDense(1, steps, nil)
	for k := range steps {
		u.Set(0, k, 1)
	}
	clResp, _ := cl.Simulate(u, nil, nil)
	manualResp, _ := clManual.Simulate(u, nil, nil)
	if !matEqual(clResp.Y, manualResp.Y, 1e-10) {
		t.Error("Feedback != manual absorb+feedback")
	}
}

func TestFeedbackApprox_DiscreteOutputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetOutputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0.1)

	cl, err := Feedback(plant, controller, -1, WithApproximatedDelays())
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}
}

func TestFeedbackApprox_DiscreteIODelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{0.8, 0.1, 0, 0.9}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, 0.3}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{4})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.3}), 0.1)

	cl, err := Feedback(plant, controller, -1, WithApproximatedDelays())
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}
}

func TestFeedbackApprox_DiscreteControllerDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	controller, _ := New(
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = controller.SetInputDelay([]float64{2})

	cl, err := Feedback(plant, controller, -1, WithApproximatedDelays())
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}
}

func TestFeedbackApprox_NoDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0.1)

	cl, err := Feedback(plant, controller, -1, WithApproximatedDelays())
	if err != nil {
		t.Fatal(err)
	}

	clDirect, _ := Feedback(plant, controller, -1)
	nCl, _, _ := cl.Dims()
	nDirect, _, _ := clDirect.Dims()
	if nCl != nDirect {
		t.Errorf("state count %d != direct %d", nCl, nDirect)
	}
}

func TestFeedbackApprox_ContinuousPade(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{0.3})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0)

	cl, err := Feedback(plant, controller, -1, WithPadeOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("closed-loop should have no delay")
	}
	if !cl.IsContinuous() {
		t.Error("closed-loop should be continuous")
	}

	padeApprox, _ := PadeDelay(0.3, 3)
	plantNoDelay := plant.Copy()
	plantNoDelay.Delay = nil
	plantApprox, _ := Series(padeApprox, plantNoDelay)
	clManual, _ := Feedback(plantApprox, controller, -1)

	nCl, _, _ := cl.Dims()
	nManual, _, _ := clManual.Dims()
	if nCl != nManual {
		t.Errorf("state count %d != manual %d", nCl, nManual)
	}

	stable, _ := cl.IsStable()
	if !stable {
		t.Error("closed-loop should be stable")
	}
}

func TestFeedbackApprox_ContinuousInputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = plant.SetInputDelay([]float64{0.5})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	cl, err := Feedback(plant, controller, -1, WithPadeOrder(5))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
	nCl, _, _ := cl.Dims()
	if nCl != 1+5 {
		t.Errorf("n = %d, want 6 (1 plant + 5 Pade)", nCl)
	}
}

func TestFeedbackApprox_ContinuousOutputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = plant.SetOutputDelay([]float64{0.4})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	cl, err := Feedback(plant, controller, -1, WithPadeOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
	nCl, _, _ := cl.Dims()
	if nCl != 1+3 {
		t.Errorf("n = %d, want 4 (1 plant + 3 Pade)", nCl)
	}
}

func TestFeedbackApprox_SingularAlgebraicLoop(t *testing.T) {
	plant, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0.1)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0.1)

	_, err := Feedback(plant, controller, 1, WithApproximatedDelays())
	if !errors.Is(err, ErrSingularTransform) {
		t.Errorf("expected ErrSingularTransform, got %v", err)
	}
}

func TestFeedbackApprox_DomainMismatch(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0.1)

	_, err := Feedback(plant, controller, -1, WithApproximatedDelays())
	if !errors.Is(err, ErrDomainMismatch) {
		t.Errorf("expected ErrDomainMismatch, got %v", err)
	}
}

func TestFeedbackApprox_PositiveFeedback(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.3}), 0.1)

	cl, err := Feedback(plant, controller, 1, WithApproximatedDelays())
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
}

func TestFeedbackApprox_WellPosednessCheck(t *testing.T) {
	// Odd-order Pade has D=(-1)^N=-1. With negative feedback and unit gains:
	// I - (-1)*(-1)*1 = I - 1 = 0 → singular.
	plant, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	plant.InputDelay = []float64{0.5}
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	_, err := Feedback(plant, controller, -1, WithPadeOrder(3))
	if err == nil {
		t.Fatal("expected singular algebraic loop error for odd-order Pade with unit gains and negative feedback")
	}
	if !errors.Is(err, ErrAlgebraicLoop) {
		t.Fatalf("err = %v, want ErrAlgebraicLoop", err)
	}

	// Even-order Pade has D=1: I - (-1)*1*1 = 2 → not singular.
	cl, err := Feedback(plant, controller, -1, WithPadeOrder(2))
	if err != nil {
		t.Fatalf("even-order Pade should work: %v", err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
}

func TestFeedbackApprox_DefaultPadeOrder(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{0.2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)

	cl, err := Feedback(plant, controller, -1, WithApproximatedDelays())
	if err != nil {
		t.Fatal(err)
	}
	nCl, _, _ := cl.Dims()
	if nCl != 1+5 {
		t.Errorf("n = %d, want 6 (1 plant + 5 default Pade order)", nCl)
	}
}

func TestFeedbackApprox_WithThiranOrder(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{3})

	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0.1)

	cl, err := Feedback(plant, controller, -1, WithThiranOrder(2))
	if err != nil {
		t.Fatal(err)
	}

	nCl, m, p := cl.Dims()
	if m != 1 || p != 1 {
		t.Errorf("m=%d p=%d, want 1,1", m, p)
	}
	if nCl < 1 {
		t.Errorf("n=%d, want ≥ 1", nCl)
	}
}

func TestFeedbackApprox_DiscreteWithThiranOrder(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{2})

	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.1}), 0.1)

	cl, err := Feedback(plant, controller, -1, WithThiranOrder(3))
	if err != nil {
		t.Fatal(err)
	}

	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("closed-loop should be stable")
	}
}

func TestFeedbackApprox_DiscreteFractionalDelayRequiresThiranOrder(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.7}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	plant.InputDelay = []float64{3.5}
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.2}), 0.1)

	if _, err := Feedback(plant, controller, -1, WithApproximatedDelays()); !errors.Is(err, ErrFractionalDelay) {
		t.Fatalf("Feedback fractional delay error = %v, want ErrFractionalDelay", err)
	}

	cl, err := Feedback(plant, controller, -1, WithThiranOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Fatal("Thiran safe feedback should absorb external delays")
	}
	n, m, p := cl.Dims()
	if n != 4 || m != 1 || p != 1 {
		t.Fatalf("closed-loop dims n=%d m=%d p=%d, want 4,1,1", n, m, p)
	}
}

func TestFeedbackApprox_DiscreteThiranRejectsResidualIODelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{0.7, 0.2, -0.1, 0.6}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0.1, -0.2, 1}),
		mat.NewDense(2, 2, nil),
		0.1,
	)
	plant.Delay = mat.NewDense(2, 2, []float64{
		1.5, 4.0,
		2.0, 1.5,
	})
	controller, _ := NewGain(mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2}), 0.1)

	if _, err := Feedback(plant, controller, -1, WithThiranOrder(2)); !errors.Is(err, ErrFeedbackDelay) {
		t.Fatalf("Feedback residual IODelay error = %v, want ErrFeedbackDelay", err)
	}
}

func TestFeedbackApprox_ContinuousMIMO(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, nil),
		0,
	)
	_ = plant.SetInputDelay([]float64{0.2, 0.5})
	controller, _ := New(
		mat.NewDense(1, 1, []float64{-3}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(2, 1, []float64{0.5, 0.3}),
		mat.NewDense(2, 2, nil),
		0,
	)

	cl, err := Feedback(plant, controller, -1, WithPadeOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasDelay() {
		t.Error("should have no delay")
	}
	nCl, _, _ := cl.Dims()
	if nCl != 2+3+3+1 {
		t.Errorf("n = %d, want 9 (2 plant + 3+3 Pade + 1 controller)", nCl)
	}
}

func TestSeriesLFT_IncompatibleIODelay(t *testing.T) {
	sys1, _ := NewFromSlices(1, 2, 2,
		[]float64{0.5}, []float64{1, 0}, []float64{1, 1}, []float64{0, 0, 0, 0}, 1.0)
	sys1.Delay = mat.NewDense(2, 2, []float64{2, 3, 5, 3})

	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 1.0)
	sys2.Delay = mat.NewDense(1, 2, []float64{1, 1})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("incompatible IO delays should produce InternalDelay")
	}
	n, m, p := result.Dims()
	if m != 2 || p != 1 {
		t.Errorf("dims m=%d p=%d, want m=2 p=1", m, p)
	}
	_ = n
}

func TestSeriesLFT_WithExistingInternalDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("should preserve InternalDelay from sys1")
	}
	if len(result.LFT.Tau) != 1 {
		t.Errorf("InternalDelay count = %d, want 1", len(result.LFT.Tau))
	}
	if result.LFT.Tau[0] != 0.5 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", result.LFT.Tau[0])
	}
	n, m, p := result.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (2,1,1)", n, m, p)
	}
}

func TestSeriesLFT_BothWithInternalDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys2.SetInternalDelay(
		[]float64{1.5, 2.0},
		mat.NewDense(1, 2, []float64{0.1, 0.2}),
		mat.NewDense(2, 1, []float64{0.3, 0.4}),
		mat.NewDense(1, 2, []float64{0.5, 0.6}),
		mat.NewDense(2, 1, []float64{0.7, 0.8}),
		mat.NewDense(2, 2, nil),
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.LFT.Tau) != 3 {
		t.Errorf("InternalDelay count = %d, want 3 (1+2)", len(result.LFT.Tau))
	}
	if result.LFT.Tau[0] != 0.5 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", result.LFT.Tau[0])
	}
	if result.LFT.Tau[1] != 1.5 {
		t.Errorf("InternalDelay[1] = %v, want 1.5", result.LFT.Tau[1])
	}
	if result.LFT.Tau[2] != 2.0 {
		t.Errorf("InternalDelay[2] = %v, want 2.0", result.LFT.Tau[2])
	}
}

func TestParallelLFT_MismatchedIODelay(t *testing.T) {
	sys1, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys1.Delay = mat.NewDense(1, 1, []float64{2})

	sys2, _ := NewFromSlices(1, 1, 1,
		[]float64{0.8}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys2.Delay = mat.NewDense(1, 1, []float64{5})

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if !result.HasInternalDelay() {
		t.Fatal("mismatched IO delays should produce InternalDelay")
	}
	if len(result.LFT.Tau) != 2 {
		t.Errorf("InternalDelay count = %d, want 2", len(result.LFT.Tau))
	}
	_, m, p := result.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims m=%d p=%d, want m=1 p=1", m, p)
	}
}

func TestParallelLFT_WithExistingInternalDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.3},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys2.SetInternalDelay(
		[]float64{0.7},
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{0.6}),
		mat.NewDense(1, 1, []float64{0.7}),
		mat.NewDense(1, 1, []float64{0.8}),
		mat.NewDense(1, 1, []float64{0}),
	)

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.LFT.Tau) != 2 {
		t.Errorf("InternalDelay count = %d, want 2 (1+1)", len(result.LFT.Tau))
	}
	if result.LFT.Tau[0] != 0.3 {
		t.Errorf("InternalDelay[0] = %v, want 0.3", result.LFT.Tau[0])
	}
	if result.LFT.Tau[1] != 0.7 {
		t.Errorf("InternalDelay[1] = %v, want 0.7", result.LFT.Tau[1])
	}
}

func TestAppend_WithInternalDelay(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys2.SetInternalDelay(
		[]float64{1.0, 2.0},
		mat.NewDense(1, 2, []float64{0.5, 0.6}),
		mat.NewDense(2, 1, []float64{0.7, 0.8}),
		mat.NewDense(1, 2, []float64{0.9, 1.0}),
		mat.NewDense(2, 1, []float64{1.1, 1.2}),
		mat.NewDense(2, 2, nil),
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.LFT.Tau) != 3 {
		t.Fatalf("InternalDelay count = %d, want 3 (1+2)", len(result.LFT.Tau))
	}
	if result.LFT.Tau[0] != 0.5 {
		t.Errorf("InternalDelay[0] = %v, want 0.5", result.LFT.Tau[0])
	}
	if result.LFT.Tau[1] != 1.0 {
		t.Errorf("InternalDelay[1] = %v, want 1.0", result.LFT.Tau[1])
	}
	if result.LFT.Tau[2] != 2.0 {
		t.Errorf("InternalDelay[2] = %v, want 2.0", result.LFT.Tau[2])
	}

	n, m, p := result.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Errorf("dims = (%d,%d,%d), want (2,2,2)", n, m, p)
	}

	r, c := result.LFT.B2.Dims()
	if r != 2 || c != 3 {
		t.Errorf("B2 dims = %dx%d, want 2x3", r, c)
	}
	if result.LFT.B2.At(0, 0) != 0.1 {
		t.Errorf("B2[0,0] = %v, want 0.1", result.LFT.B2.At(0, 0))
	}
	if result.LFT.B2.At(1, 1) != 0.5 {
		t.Errorf("B2[1,1] = %v, want 0.5", result.LFT.B2.At(1, 1))
	}
	if result.LFT.B2.At(0, 1) != 0 {
		t.Errorf("B2[0,1] = %v, want 0 (block-diagonal)", result.LFT.B2.At(0, 1))
	}

	r, c = result.LFT.D22.Dims()
	if r != 3 || c != 3 {
		t.Errorf("D22 dims = %dx%d, want 3x3", r, c)
	}
}

func TestSeriesLFT_Roundtrip(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys2.SetInternalDelay(
		[]float64{1.5},
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{0.6}),
		mat.NewDense(1, 1, []float64{0.7}),
		mat.NewDense(1, 1, []float64{0.8}),
		mat.NewDense(1, 1, []float64{0}),
	)

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	H, tau, err := result.GetDelayModel()
	if err != nil {
		t.Fatal(err)
	}
	if len(tau) != 2 {
		t.Fatalf("tau length = %d, want 2", len(tau))
	}

	rebuilt, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	if len(rebuilt.LFT.Tau) != 2 {
		t.Fatalf("rebuilt InternalDelay = %d, want 2", len(rebuilt.LFT.Tau))
	}
	for i, v := range result.LFT.Tau {
		if math.Abs(rebuilt.LFT.Tau[i]-v) > 1e-12 {
			t.Errorf("InternalDelay[%d] = %v, want %v", i, rebuilt.LFT.Tau[i], v)
		}
	}
}

func TestSeriesLFT_IncompatibleMIMOIODelay(t *testing.T) {
	sys1, _ := NewFromSlices(1, 1, 2,
		[]float64{0.5}, []float64{1}, []float64{1, 1}, []float64{0, 0}, 0)
	sys1.Delay = mat.NewDense(2, 1, []float64{1, 3})

	sys2, _ := NewFromSlices(1, 2, 1,
		[]float64{0.8}, []float64{1, 1}, []float64{1}, []float64{0, 0}, 0)
	sys2.Delay = mat.NewDense(1, 2, []float64{2, 4})

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	if !result.HasInternalDelay() {
		t.Fatal("incompatible MIMO IO delays should produce InternalDelay")
	}

	_, m, p := result.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims m=%d p=%d, want m=1 p=1", m, p)
	}

	H, tau, err := result.GetDelayModel()
	if err != nil {
		t.Fatal(err)
	}
	if len(tau) == 0 {
		t.Fatal("should have non-empty tau")
	}
	_, mH, pH := H.Dims()
	if mH != m+len(tau) {
		t.Errorf("H inputs = %d, want %d", mH, m+len(tau))
	}
	if pH != p+len(tau) {
		t.Errorf("H outputs = %d, want %d", pH, p+len(tau))
	}
}

func TestParallelLFT_MatchingDelaysNoLFT(t *testing.T) {
	sys1, _ := NewFromSlices(1, 1, 1,
		[]float64{0.5}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys1.Delay = mat.NewDense(1, 1, []float64{3})

	sys2, _ := NewFromSlices(1, 1, 1,
		[]float64{0.8}, []float64{1}, []float64{1}, []float64{0}, 1.0)
	sys2.Delay = mat.NewDense(1, 1, []float64{3})

	result, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.HasInternalDelay() {
		t.Error("matching delays should not produce InternalDelay")
	}
	if result.Delay == nil || result.Delay.At(0, 0) != 3 {
		t.Errorf("matching delays should preserve IODelay=3")
	}
}

func TestAppend_MixedInternalAndIO(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = sys1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	result, err := Append(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if len(result.LFT.Tau) != 1 {
		t.Errorf("InternalDelay count = %d, want 1", len(result.LFT.Tau))
	}
	r, c := result.LFT.B2.Dims()
	if r != 2 || c != 1 {
		t.Errorf("B2 dims = %dx%d, want 2x1", r, c)
	}
	if result.LFT.B2.At(1, 0) != 0 {
		t.Errorf("B2[1,0] = %v, want 0 (block-diagonal)", result.LFT.B2.At(1, 0))
	}
}

func TestFeedback_LFT_InputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{3})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.8}), 0.1)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.InputDelay != nil || !cl.HasInternalDelay() || len(cl.LFT.Tau) != 1 || cl.LFT.Tau[0] != 3 {
		t.Errorf("plant InputDelay must become internal: InputDelay=%v internal=%v", cl.InputDelay, cl.LFT)
	}
	_, m, p := cl.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims m=%d p=%d, want 1,1", m, p)
	}
	assertResponseOracle(t, "feedback", cl, func(s complex128) [][]complex128 {
		return closedLoopOracle(t, plant, controller, -1, s)
	})
}

func TestFeedback_LFT_OutputDelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetOutputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0.1)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Error("should have internal delays")
	}
	if len(cl.LFT.Tau) != 1 || cl.LFT.Tau[0] != 2 {
		t.Errorf("InternalDelay = %v, want [2]", cl.LFT.Tau)
	}
}

func TestFeedback_LFT_BothDelayed(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{2})
	controller, _ := New(
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = controller.SetOutputDelay([]float64{1})

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() || len(cl.LFT.Tau) != 2 {
		t.Fatalf("plant InputDelay and controller OutputDelay must both become internal: %v", cl.LFT)
	}
	if cl.InputDelay != nil {
		t.Errorf("plant InputDelay left external: %v", cl.InputDelay)
	}
	assertResponseOracle(t, "feedback", cl, func(s complex128) [][]complex128 {
		return closedLoopOracle(t, plant, controller, -1, s)
	})
}

func TestFeedback_LFT_NoDelay_Unchanged(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.HasInternalDelay() {
		t.Error("delay-free feedback should not have internal delays")
	}
	poles, _ := cl.Poles()
	if len(poles) != 1 || math.Abs(real(poles[0])+1) > 1e-10 {
		t.Errorf("pole = %v, want -1", poles[0])
	}
}

func TestFeedback_LFT_IODelay(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	plant.Delay = mat.NewDense(1, 1, []float64{0.3})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Error("should have internal delays from IODelay")
	}
	if len(cl.LFT.Tau) != 1 || cl.LFT.Tau[0] != 0.3 {
		t.Errorf("InternalDelay = %v, want [0.3]", cl.LFT.Tau)
	}
	_, m, p := cl.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims m=%d p=%d, want 1,1", m, p)
	}
}

func TestFeedback_LFT_PositiveFeedback(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.3}), 0.1)

	cl, err := Feedback(plant, controller, 1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.InputDelay != nil || !cl.HasInternalDelay() {
		t.Errorf("plant InputDelay must become internal: InputDelay=%v internal=%v", cl.InputDelay, cl.LFT)
	}
	assertResponseOracle(t, "positive feedback", cl, func(s complex128) [][]complex128 {
		return closedLoopOracle(t, plant, controller, 1, s)
	})
}

func TestFeedbackMovesInputDelayInsideLoop(t *testing.T) {
	plant, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	_ = plant.SetInputDelay([]float64{5})
	_ = plant.SetOutputDelay([]float64{2})
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0.1)

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if cl.InputDelay != nil || cl.OutputDelay != nil {
		t.Errorf("plant delays left external: in=%v out=%v", cl.InputDelay, cl.OutputDelay)
	}
	if !cl.HasInternalDelay() || !slices.Contains(cl.LFT.Tau, 5) || !slices.Contains(cl.LFT.Tau, 2) {
		t.Fatalf("internal delays should contain 5 and 2, got %v", cl.LFT)
	}
	assertResponseOracle(t, "feedback", cl, func(s complex128) [][]complex128 {
		return closedLoopOracle(t, plant, controller, -1, s)
	})
}

func TestSeriesKeepsExternalDelays(t *testing.T) {
	sys1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys1.InputDelay = []float64{1.5}
	sys1.OutputDelay = []float64{0.5}

	sys2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sys2.InputDelay = []float64{0.3}
	sys2.OutputDelay = []float64{0.7}

	result, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}
	if result.InputDelay == nil || math.Abs(result.InputDelay[0]-1.5) > 1e-12 {
		t.Errorf("sys1 InputDelay should stay external: got %v, want [1.5]", result.InputDelay)
	}
	if result.OutputDelay == nil || math.Abs(result.OutputDelay[0]-0.7) > 1e-12 {
		t.Errorf("sys2 OutputDelay should stay external: got %v, want [0.7]", result.OutputDelay)
	}
}

func TestFeedback_DelayInFeedbackPath(t *testing.T) {
	plant, _ := New(
		mat.NewDense(1, 1, []float64{-1.0 / 3.0}),
		mat.NewDense(1, 1, []float64{1.5 / 3.0}),
		mat.NewDense(1, 1, []float64{1.0}),
		mat.NewDense(1, 1, []float64{0.0}),
		0,
	)
	_ = plant.SetInputDelay([]float64{0.5})

	ctrl, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.25}),
		mat.NewDense(1, 1, []float64{0.5}),
		0,
	)

	L, _ := Series(ctrl, plant)

	I, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	S, _ := Feedback(I, L, -1)
	T, _ := Feedback(L, nil, -1)

	dt := 0.05
	nSteps := 1001
	u := mat.NewDense(1, nSteps, nil)
	for j := range nSteps {
		u.Set(0, j, 1.0)
	}

	Sd, _ := S.C2D(dt, C2DOptions{})
	sResp, _ := Sd.Simulate(u, nil, nil)
	sFinal := sResp.Y.At(0, nSteps-1)

	Td, _ := T.C2D(dt, C2DOptions{})
	tResp, _ := Td.Simulate(u, nil, nil)
	tFinal := tResp.Y.At(0, nSteps-1)

	if math.Abs(sFinal) > 0.05 {
		t.Errorf("S(inf) = %.4f, want ~0", sFinal)
	}
	if math.Abs(tFinal-1.0) > 0.05 {
		t.Errorf("T(inf) = %.4f, want ~1", tFinal)
	}
}

func TestBlkDiag_Empty(t *testing.T) {
	_, err := BlkDiag()
	if err == nil {
		t.Fatal("expected error for 0 args")
	}
	if !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("got %v, want ErrInvalidArgument", err)
	}
}

func TestBlkDiag_Single(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	got, err := BlkDiag(sys)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := got.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (2,1,1)", n, m, p)
	}
	for _, s := range []complex128{0.1i, 1i, 10i} {
		h1 := evalTF(sys, s)
		h2 := evalTF(got, s)
		if cmplx.Abs(h1[0][0]-h2[0][0]) > 1e-10 {
			t.Errorf("TF mismatch at s=%v: %v vs %v", s, h1[0][0], h2[0][0])
		}
	}
}

func TestBlkDiag_Two_MatchesAppend(t *testing.T) {
	G1, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.3, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	G2, _ := New(
		mat.NewDense(1, 1, []float64{-3}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	blk, err := BlkDiag(G1, G2)
	if err != nil {
		t.Fatal(err)
	}
	app, err := Append(G1, G2)
	if err != nil {
		t.Fatal(err)
	}

	nb, mb, pb := blk.Dims()
	na, ma, pa := app.Dims()
	if nb != na || mb != ma || pb != pa {
		t.Fatalf("dim mismatch: blkdiag(%d,%d,%d) vs append(%d,%d,%d)", nb, mb, pb, na, ma, pa)
	}

	for _, s := range []complex128{0.1i, 1i, 10i} {
		hb := evalTF(blk, s)
		ha := evalTF(app, s)
		for i := range hb {
			for j := range hb[i] {
				if cmplx.Abs(hb[i][j]-ha[i][j]) > 1e-10 {
					t.Errorf("TF[%d][%d] mismatch at s=%v: %v vs %v", i, j, s, hb[i][j], ha[i][j])
				}
			}
		}
	}
}

func TestBlkDiag_Three_SISO(t *testing.T) {
	G1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	G2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	G3, _ := New(
		mat.NewDense(1, 1, []float64{-3}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	blk, err := BlkDiag(G1, G2, G3)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := blk.Dims()
	if n != 3 || m != 3 || p != 3 {
		t.Fatalf("dims = (%d,%d,%d), want (3,3,3)", n, m, p)
	}

	for _, s := range []complex128{0.1i, 1i, 10i} {
		h := evalTF(blk, s)
		h1 := evalTF(G1, s)
		h2 := evalTF(G2, s)
		h3 := evalTF(G3, s)

		if cmplx.Abs(h[0][0]-h1[0][0]) > 1e-10 {
			t.Errorf("(0,0) mismatch at s=%v", s)
		}
		if cmplx.Abs(h[1][1]-h2[0][0]) > 1e-10 {
			t.Errorf("(1,1) mismatch at s=%v", s)
		}
		if cmplx.Abs(h[2][2]-h3[0][0]) > 1e-10 {
			t.Errorf("(2,2) mismatch at s=%v", s)
		}
		if cmplx.Abs(h[0][1]) > 1e-10 || cmplx.Abs(h[1][0]) > 1e-10 {
			t.Errorf("off-diagonal nonzero at s=%v", s)
		}
	}
}

func TestBlkDiag_AllGain(t *testing.T) {
	G1, _ := NewGain(mat.NewDense(1, 2, []float64{1, 2}), 0)
	G2, _ := NewGain(mat.NewDense(2, 1, []float64{3, 4}), 0)

	blk, err := BlkDiag(G1, G2)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := blk.Dims()
	if n != 0 {
		t.Fatalf("expected n=0, got %d", n)
	}
	if m != 3 || p != 3 {
		t.Fatalf("dims = (%d,%d,%d), want (0,3,3)", n, m, p)
	}

	wantD := [][]float64{
		{1, 2, 0},
		{0, 0, 3},
		{0, 0, 4},
	}
	for i := range wantD {
		for j := range wantD[i] {
			if math.Abs(blk.D.At(i, j)-wantD[i][j]) > 1e-10 {
				t.Errorf("D(%d,%d) = %v, want %v", i, j, blk.D.At(i, j), wantD[i][j])
			}
		}
	}
}

func TestBlkDiag_DomainMismatch(t *testing.T) {
	G1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	G2, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.01,
	)
	_, err := BlkDiag(G1, G2)
	if err == nil {
		t.Fatal("expected error for domain mismatch")
	}
	if !errors.Is(err, ErrDomainMismatch) {
		t.Errorf("got %v, want ErrDomainMismatch", err)
	}
}

func TestBlkDiag_NonSymmetricA(t *testing.T) {
	G1, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.7, -0.3, -2}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{0.2, 1}),
		mat.NewDense(1, 1, []float64{0.1}),
		0,
	)
	G2, _ := New(
		mat.NewDense(2, 2, []float64{-3, 1.2, -0.8, -4}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0.3}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	blk, err := BlkDiag(G1, G2)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := blk.Dims()
	if n != 4 || m != 2 || p != 2 {
		t.Fatalf("dims = (%d,%d,%d), want (4,2,2)", n, m, p)
	}

	if math.Abs(blk.A.At(0, 2)) > 1e-15 || math.Abs(blk.A.At(2, 0)) > 1e-15 {
		t.Error("off-diagonal blocks of A should be zero")
	}

	for _, s := range []complex128{0.1i, 1i, 10i} {
		hblk := evalTF(blk, s)
		h1 := evalTF(G1, s)
		h2 := evalTF(G2, s)
		if cmplx.Abs(hblk[0][0]-h1[0][0]) > 1e-10 {
			t.Errorf("(0,0) mismatch at s=%v", s)
		}
		if cmplx.Abs(hblk[1][1]-h2[0][0]) > 1e-10 {
			t.Errorf("(1,1) mismatch at s=%v", s)
		}
		if cmplx.Abs(hblk[0][1]) > 1e-10 || cmplx.Abs(hblk[1][0]) > 1e-10 {
			t.Errorf("off-diagonal nonzero at s=%v", s)
		}
	}
}

func TestConnect_SeriesEquivalence(t *testing.T) {
	G1, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	G2, _ := New(
		mat.NewDense(1, 1, []float64{-3}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	aug, err := BlkDiag(G1, G2)
	if err != nil {
		t.Fatal(err)
	}

	Q := mat.NewDense(2, 2, nil)
	Q.Set(1, 0, 1)

	conn, err := Connect(aug, Q, []int{0}, []int{1})
	if err != nil {
		t.Fatal(err)
	}

	ser, err := Series(G1, G2)
	if err != nil {
		t.Fatal(err)
	}

	for _, s := range []complex128{0.1i, 1i, 10i} {
		hc := evalTF(conn, s)
		hs := evalTF(ser, s)
		if cmplx.Abs(hc[0][0]-hs[0][0]) > 1e-10 {
			t.Errorf("series mismatch at s=%v: connect=%v series=%v", s, hc[0][0], hs[0][0])
		}
	}
}

func TestConnect_FeedbackEquivalence(t *testing.T) {
	P, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	K, _ := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	aug, err := BlkDiag(P, K)
	if err != nil {
		t.Fatal(err)
	}

	Q := mat.NewDense(2, 2, nil)
	Q.Set(0, 1, -1)
	Q.Set(1, 0, 1)

	conn, err := Connect(aug, Q, []int{0}, []int{0})
	if err != nil {
		t.Fatal(err)
	}

	fb, err := Feedback(P, K, -1)
	if err != nil {
		t.Fatal(err)
	}

	for _, s := range []complex128{0.1i, 1i, 10i} {
		hc := evalTF(conn, s)
		hf := evalTF(fb, s)
		if cmplx.Abs(hc[0][0]-hf[0][0]) > 1e-10 {
			t.Errorf("feedback mismatch at s=%v: connect=%v feedback=%v", s, hc[0][0], hf[0][0])
		}
	}
}

func TestConnect_AlgebraicLoop(t *testing.T) {
	G, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)

	aug, _ := BlkDiag(G)

	Q := mat.NewDense(1, 1, []float64{1})

	_, err := Connect(aug, Q, []int{0}, []int{0})
	if err == nil {
		t.Fatal("expected algebraic loop error")
	}
	if !errors.Is(err, ErrAlgebraicLoop) {
		t.Errorf("got %v, want ErrAlgebraicLoop", err)
	}
	var diagnostic *AlgebraicLoopError
	if !errors.As(err, &diagnostic) {
		t.Fatalf("err = %T, want *AlgebraicLoopError", err)
	}
	if len(diagnostic.Signals) != 0 {
		t.Errorf("signals = %v, want none for indexed connection", diagnostic.Signals)
	}
	if !math.IsInf(diagnostic.Condition, 1) {
		t.Errorf("condition = %g, want +Inf", diagnostic.Condition)
	}
}

func TestConnect_ZeroQ(t *testing.T) {
	G, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.3, -0.5, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)

	Q := mat.NewDense(2, 2, nil)

	conn, err := Connect(G, Q, []int{0}, []int{1})
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := conn.Dims()
	if m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (_,1,1)", n, m, p)
	}

	for _, s := range []complex128{0.1i, 1i, 10i} {
		hc := evalTF(conn, s)
		hg := evalTF(G, s)
		if cmplx.Abs(hc[0][0]-hg[1][0]) > 1e-10 {
			t.Errorf("ZeroQ mismatch at s=%v: got %v, want G[1][0]=%v", s, hc[0][0], hg[1][0])
		}
	}
}

func TestConnect_QDimMismatch(t *testing.T) {
	G, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{0, 0}),
		0,
	)

	Q := mat.NewDense(3, 3, nil)
	_, err := Connect(G, Q, []int{0}, []int{0})
	if err == nil {
		t.Fatal("expected error for Q dim mismatch")
	}
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("got %v, want ErrDimensionMismatch", err)
	}
}

func TestConnect_IndexOutOfRange(t *testing.T) {
	G, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	Q := mat.NewDense(1, 1, nil)

	_, err := Connect(G, Q, []int{5}, []int{0})
	if err == nil {
		t.Fatal("expected error for input index out of range")
	}

	_, err = Connect(G, Q, []int{0}, []int{5})
	if err == nil {
		t.Fatal("expected error for output index out of range")
	}

	_, err = Connect(G, Q, []int{-1}, []int{0})
	if err == nil {
		t.Fatal("expected error for negative input index")
	}
}

func TestConnect_DuplicateIndex(t *testing.T) {
	G, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.3, -0.5, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	Q := mat.NewDense(2, 2, nil)

	_, err := Connect(G, Q, []int{0, 0}, []int{0})
	if err == nil {
		t.Fatal("expected error for duplicate input index")
	}

	_, err = Connect(G, Q, []int{0}, []int{1, 1})
	if err == nil {
		t.Fatal("expected error for duplicate output index")
	}
}

func TestConnect_AllGain(t *testing.T) {
	G1, _ := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	G2, _ := NewGain(mat.NewDense(1, 1, []float64{3}), 0)

	aug, _ := BlkDiag(G1, G2)
	Q := mat.NewDense(2, 2, nil)
	Q.Set(1, 0, 1)

	conn, err := Connect(aug, Q, []int{0}, []int{1})
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := conn.Dims()
	if n != 0 {
		t.Fatalf("expected n=0 for gain-only, got %d", n)
	}
	if m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (0,1,1)", n, m, p)
	}

	if math.Abs(conn.D.At(0, 0)-6) > 1e-10 {
		t.Errorf("series gain: got %v, want 6", conn.D.At(0, 0))
	}
}

func TestConnect_WithDelay(t *testing.T) {
	G, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = G.SetDelay(mat.NewDense(1, 1, []float64{0.5}))

	Q := mat.NewDense(1, 1, nil)
	result, err := Connect(G, Q, []int{0}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := result.Dims()
	if m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (_,1,1)", n, m, p)
	}
	if !result.HasInternalDelay() && !result.HasDelay() {
		if result.OutputDelay == nil && result.InputDelay == nil {
			t.Error("expected delay to be preserved")
		}
	}
}

func TestConnect_WithDelay_Feedback(t *testing.T) {
	P, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	_ = P.SetOutputDelay([]float64{0.3})

	K, _ := NewGain(mat.NewDense(1, 1, []float64{2}), 0)

	aug, _ := BlkDiag(P, K)
	_, m, p := aug.Dims()
	Q := mat.NewDense(m, p, nil)
	Q.Set(0, 1, -1)
	Q.Set(1, 0, 1)

	result, err := Connect(aug, Q, []int{0}, []int{0})
	if err != nil {
		t.Fatal(err)
	}
	_, mr, pr := result.Dims()
	if mr != 1 || pr != 1 {
		t.Fatalf("dims = (_,%d,%d), want (_,1,1)", mr, pr)
	}
}

func TestConnect_NonSymmetricA(t *testing.T) {
	G1, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.7, -0.3, -2}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{0.2, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	G2, _ := New(
		mat.NewDense(2, 2, []float64{-3, 1.2, -0.8, -4}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0.3}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	aug, _ := BlkDiag(G1, G2)
	Q := mat.NewDense(2, 2, nil)
	Q.Set(1, 0, 1)
	conn, err := Connect(aug, Q, []int{0}, []int{1})
	if err != nil {
		t.Fatal(err)
	}

	ser, _ := Series(G1, G2)
	for _, s := range []complex128{0.1i, 1i, 10i} {
		hc := evalTF(conn, s)
		hs := evalTF(ser, s)
		if cmplx.Abs(hc[0][0]-hs[0][0]) > 1e-10 {
			t.Errorf("non-sym series mismatch at s=%v: %v vs %v", s, hc[0][0], hs[0][0])
		}
	}
}

func TestConnect_MIMOThreeBlocks(t *testing.T) {
	P, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, -0.3, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	K, _ := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{2}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	S, _ := New(
		mat.NewDense(1, 1, []float64{-10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	aug, err := BlkDiag(P, K, S)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := aug.Dims()
	if n != 4 || m != 3 || p != 3 {
		t.Fatalf("aug dims = (%d,%d,%d), want (4,3,3)", n, m, p)
	}

	Q := mat.NewDense(3, 3, nil)
	Q.Set(0, 2, -1)
	Q.Set(1, 0, 0)
	Q.Set(2, 0, 1)

	conn, err := Connect(aug, Q, []int{0, 1}, []int{0})
	if err != nil {
		t.Fatal(err)
	}

	cn, cm, cp := conn.Dims()
	if cm != 2 || cp != 1 {
		t.Fatalf("conn dims = (%d,%d,%d), want (%d,2,1)", cn, cm, cp, cn)
	}

	for _, s := range []complex128{0.1i, 1i, 10i} {
		h := evalTF(conn, s)
		if cmplx.IsNaN(h[0][0]) || cmplx.IsInf(h[0][0]) {
			t.Errorf("TF is NaN/Inf at s=%v", s)
		}
	}

	hdc := evalTF(conn, 0)
	if cmplx.Abs(hdc[0][0]) < 1e-15 && cmplx.Abs(hdc[0][1]) < 1e-15 {
		t.Error("DC gain should not be all zero")
	}
}

func TestBlkDiag_ThreeWithInternalDelay(t *testing.T) {
	mkLFT := func(a, tau float64) *System {
		s, _ := New(
			mat.NewDense(1, 1, []float64{a}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{0}), 0)
		_ = s.SetInternalDelay(
			[]float64{tau},
			mat.NewDense(1, 1, []float64{0.1}),
			mat.NewDense(1, 1, []float64{0.2}),
			mat.NewDense(1, 1, []float64{0.3}),
			mat.NewDense(1, 1, []float64{0.4}),
			mat.NewDense(1, 1, []float64{0}),
		)
		return s
	}

	s1 := mkLFT(-1, 0.5)
	s2 := mkLFT(-2, 1.0)
	s3 := mkLFT(-3, 1.5)

	result, err := BlkDiag(s1, s2, s3)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := result.Dims()
	if n != 3 || m != 3 || p != 3 {
		t.Errorf("dims = (%d,%d,%d), want (3,3,3)", n, m, p)
	}
	if len(result.LFT.Tau) != 3 {
		t.Fatalf("InternalDelay = %v, want 3 entries", result.LFT.Tau)
	}
	wantTau := []float64{0.5, 1.0, 1.5}
	for i, w := range wantTau {
		if result.LFT.Tau[i] != w {
			t.Errorf("tau[%d] = %v, want %v", i, result.LFT.Tau[i], w)
		}
	}

	r, c := result.LFT.B2.Dims()
	if r != 3 || c != 3 {
		t.Errorf("B2 dims = (%d,%d), want (3,3)", r, c)
	}
	for i := range 3 {
		if result.LFT.B2.At(i, i) != 0.1 {
			t.Errorf("B2[%d,%d] = %v, want 0.1", i, i, result.LFT.B2.At(i, i))
		}
	}

	r, c = result.LFT.D12.Dims()
	if r != 3 || c != 3 {
		t.Errorf("D12 dims = (%d,%d), want (3,3)", r, c)
	}
	r, c = result.LFT.D21.Dims()
	if r != 3 || c != 3 {
		t.Errorf("D21 dims = (%d,%d), want (3,3)", r, c)
	}
}

func TestAppend_MixedLFTAndPlain(t *testing.T) {
	s1, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	_ = s1.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0.4}),
		mat.NewDense(1, 1, []float64{0}),
	)

	s2, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	result, err := Append(s1, s2)
	if err != nil {
		t.Fatal(err)
	}

	if len(result.LFT.Tau) != 1 {
		t.Errorf("InternalDelay = %v, want [0.5]", result.LFT.Tau)
	}
	n, m, p := result.Dims()
	if n != 2 || m != 2 || p != 2 {
		t.Errorf("dims = (%d,%d,%d), want (2,2,2)", n, m, p)
	}
}

func blockDiagOracle(t *testing.T, s complex128, parts ...*System) [][]complex128 {
	t.Helper()
	var p, m int
	for _, s := range parts {
		_, mi, pi := s.Dims()
		p += pi
		m += mi
	}
	out := cmatOf(nil, p, m)
	r0, c0 := 0, 0
	for _, sys := range parts {
		G := evalDelaySystem(t, sys, s, exactDelayFactor(sys.Dt))
		for i := range G {
			copy(out[r0+i][c0:], G[i])
		}
		r0 += len(G)
		c0 += len(G[0])
	}
	return out
}

func TestBlkDiagWithInternalDelayKeepsExternalDelaysOnce(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		mk := map[string]func() *System{
			"lft":      func() *System { return feedbackDelayPlant(t, dt, "lft") },
			"mixed":    func() *System { return feedbackDelayPlant(t, dt, "mixed") },
			"k-in":     func() *System { return feedbackDelayController(t, dt, "in") },
			"k-out":    func() *System { return feedbackDelayController(t, dt, "out") },
			"k-iod":    func() *System { return feedbackDelayController(t, dt, "iod") },
			"gain-iod": func() *System { return feedbackDelayController(t, dt, "gain-iod") },
		}
		for _, pair := range [][2]string{{"lft", "k-in"}, {"k-out", "lft"}, {"mixed", "lft"}, {"lft", "k-iod"}, {"gain-iod", "lft"}, {"lft", "lft"}} {
			a, b := mk[pair[0]](), mk[pair[1]]()
			r, err := BlkDiag(a, b)
			if err != nil {
				t.Fatalf("dt=%v %v: %v", dt, pair, err)
			}
			assertResponseOracle(t, fmt.Sprintf("dt=%v BlkDiag%v", dt, pair), r, func(s complex128) [][]complex128 {
				return blockDiagOracle(t, s, a, b)
			})
		}
	}
}

// connectOracle evaluates outputs of y = G*u with u = Q*y + r on the
// selected input channels.
func connectOracle(t *testing.T, sys *System, Q *mat.Dense, inputs, outputs []int, s complex128) [][]complex128 {
	t.Helper()
	G := evalDelaySystem(t, sys, s, exactDelayFactor(sys.Dt))
	p := len(G)
	GQ := cmul(G, cmatOf(Q, len(G[0]), p))
	loop := make([][]complex128, p)
	for i := range p {
		loop[i] = make([]complex128, p)
		for j := range p {
			loop[i][j] = -GQ[i][j]
		}
		loop[i][i] += 1
	}
	T := csolve(t, loop, G)
	out := make([][]complex128, len(outputs))
	for a, i := range outputs {
		out[a] = make([]complex128, len(inputs))
		for b, j := range inputs {
			out[a][b] = T[i][j]
		}
	}
	return out
}

func TestConnectKeepsLoopDelaysInsideLoop(t *testing.T) {
	full := mat.NewDense(4, 4, []float64{
		0, 0, -1, 0,
		0, 0, 0, -1,
		1, 0, 0, 0,
		0, 1, 0, 0,
	})
	partial := mat.NewDense(4, 4, []float64{
		0, 0, -1, 0,
		0, 0, 0, 0,
		1, 0, 0, 0,
		0, 0, 0, 0,
	})
	for _, dt := range []float64{0, 0.1} {
		sc := 1.0
		if dt > 0 {
			sc = 10
		}
		for _, pk := range []string{"in", "out", "iod", "mixed", "lft"} {
			for _, kk := range []string{"none", "in", "out"} {
				for qName, Q := range map[string]*mat.Dense{"full": full, "partial": partial} {
					P := feedbackDelayPlant(t, dt, pk)
					K := feedbackDelayController(t, dt, kk)
					aug, err := BlkDiag(P, K)
					if err != nil {
						t.Fatal(err)
					}
					inputs, outputs := []int{0, 1}, []int{0, 1}
					label := fmt.Sprintf("dt=%v P=%s K=%s Q=%s", dt, pk, kk, qName)
					res, err := Connect(aug, Q, inputs, outputs)
					if err != nil {
						t.Fatalf("%s: %v", label, err)
					}
					assertResponseOracle(t, label, res, func(s complex128) [][]complex128 {
						return connectOracle(t, aug, Q, inputs, outputs, s)
					})
					if qName == "partial" && pk == "in" {
						if res.InputDelay == nil || res.InputDelay[0] != 0 || res.InputDelay[1] != 0.2*sc {
							t.Errorf("%s: open-loop input delay should stay external, got %v", label, res.InputDelay)
						}
					}
				}
			}
		}
	}
}

func zeroStateDelayBlock(t *testing.T, dt float64, kind int) *System {
	t.Helper()
	sc := 1.0
	if dt > 0 {
		sc = 10
	}
	var H *System
	var tau []float64
	var err error
	switch kind {
	case 0:
		H, err = NewGain(mat.NewDense(3, 3, []float64{
			0.7, -0.2, 1,
			0.4, 1.1, 0.3,
			0.5, -0.6, 0.25,
		}), dt)
		tau = []float64{0.3 * sc}
	default:
		H, err = NewGain(mat.NewDense(4, 4, []float64{
			-0.4, 0.8, 0, 0.5,
			0.2, 0.1, 0.9, 0,
			1.3, 0, -0.35, 0.1,
			0.6, -0.15, 0.2, 0.3,
		}), dt)
		tau = []float64{0.2 * sc, 0.5 * sc}
	}
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := sys.Dims(); n != 0 || !sys.HasInternalDelay() {
		t.Fatalf("fixture kind %d: n=%d internal=%v", kind, n, sys.HasInternalDelay())
	}
	return sys
}

func assertLFTShape(t *testing.T, label string, sys *System) {
	t.Helper()
	n, m, p := sys.Dims()
	if sys.LFT == nil {
		return
	}
	N := len(sys.LFT.Tau)
	chk := func(name string, d *mat.Dense, r, c int) {
		var gr, gc int
		if d != nil {
			gr, gc = d.Dims()
		}
		if r == 0 || c == 0 {
			r, c = 0, 0
		}
		if gr != r || gc != c {
			t.Errorf("%s: %s %dx%d, want %dx%d", label, name, gr, gc, r, c)
		}
	}
	chk("B2", sys.LFT.B2, n, N)
	chk("C2", sys.LFT.C2, N, n)
	chk("D12", sys.LFT.D12, p, N)
	chk("D21", sys.LFT.D21, N, m)
	chk("D22", sys.LFT.D22, N, N)
}

func TestZeroStateInternalDelayInterconnectionsHaveNoPhantomStates(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		z0 := zeroStateDelayBlock(t, dt, 0)
		z1 := zeroStateDelayBlock(t, dt, 1)
		gain, err := NewGain(mat.NewDense(2, 2, []float64{0.3, -1.2, 0.7, 0.45}), dt)
		if err != nil {
			t.Fatal(err)
		}
		dyn := feedbackDelayPlant(t, dt, "lft")
		nDyn, _, _ := dyn.Dims()
		z0Src, z1Src := z0.Copy(), z1.Copy()

		check := func(label string, got *System, wantN int, ref func(s complex128) [][]complex128) {
			t.Helper()
			label = fmt.Sprintf("dt=%v %s", dt, label)
			if n, _, _ := got.Dims(); n != wantN {
				t.Errorf("%s: n=%d, want %d", label, n, wantN)
			}
			assertLFTShape(t, label, got)
			assertResponseOracle(t, label, got, ref)
		}
		eval := func(sys *System, s complex128) [][]complex128 {
			return evalDelaySystem(t, sys, s, exactDelayFactor(dt))
		}

		for _, pr := range []struct {
			name string
			a, b *System
			n    int
		}{
			{"z0,z1", z0, z1, 0},
			{"z1,gain", z1, gain, 0},
			{"gain,z0", gain, z0, 0},
			{"z0,dyn", z0, dyn, nDyn},
			{"dyn,z1", dyn, z1, nDyn},
			{"gain-iod,z0", feedbackDelayController(t, dt, "gain-iod"), z0, 0},
		} {
			a, b := pr.a, pr.b
			r, err := Append(a, b)
			if err != nil {
				t.Fatalf("Append %s: %v", pr.name, err)
			}
			check("Append "+pr.name, r, pr.n, func(s complex128) [][]complex128 { return blockDiagOracle(t, s, a, b) })
			r, err = BlkDiag(a, b)
			if err != nil {
				t.Fatalf("BlkDiag %s: %v", pr.name, err)
			}
			check("BlkDiag "+pr.name, r, pr.n, func(s complex128) [][]complex128 { return blockDiagOracle(t, s, a, b) })
			r, err = Series(a, b)
			if err != nil {
				t.Fatalf("Series %s: %v", pr.name, err)
			}
			check("Series "+pr.name, r, pr.n, func(s complex128) [][]complex128 { return cmul(eval(b, s), eval(a, s)) })
			r, err = Parallel(a, b)
			if err != nil {
				t.Fatalf("Parallel %s: %v", pr.name, err)
			}
			check("Parallel "+pr.name, r, pr.n, func(s complex128) [][]complex128 {
				g1, g2 := eval(a, s), eval(b, s)
				for i := range g1 {
					for j := range g1[i] {
						g1[i][j] += g2[i][j]
					}
				}
				return g1
			})
			r, err = Feedback(a, b, -1)
			if err != nil {
				t.Fatalf("Feedback %s: %v", pr.name, err)
			}
			check("Feedback "+pr.name, r, pr.n, func(s complex128) [][]complex128 { return closedLoopOracle(t, a, b, -1, s) })
		}

		r, err := Append(z0, z1)
		if err != nil {
			t.Fatal(err)
		}
		r, err = Append(r, z0)
		if err != nil {
			t.Fatal(err)
		}
		check("Append chain", r, 0, func(s complex128) [][]complex128 { return blockDiagOracle(t, s, z0, z1, z0) })
		if len(r.LFT.Tau) != 4 {
			t.Errorf("dt=%v Append chain: %d internal delays, want 4", dt, len(r.LFT.Tau))
		}

		aug, err := Append(z0, z1)
		if err != nil {
			t.Fatal(err)
		}
		Q := mat.NewDense(4, 4, []float64{
			0, 0, 0.3, 0,
			0, 0, 0, -0.5,
			0.4, 0, 0, 0,
			0, 0.2, 0, 0,
		})
		in, out := []int{0, 1, 3}, []int{0, 2}
		r, err = Connect(aug, Q, in, out)
		if err != nil {
			t.Fatal(err)
		}
		check("Connect", r, 0, func(s complex128) [][]complex128 { return connectOracle(t, aug, Q, in, out, s) })

		same := func(a, b *System) bool {
			return mat.Equal(a.D, b.D) && slices.Equal(a.LFT.Tau, b.LFT.Tau) && mat.Equal(a.LFT.D12, b.LFT.D12) &&
				mat.Equal(a.LFT.D21, b.LFT.D21) && mat.Equal(a.LFT.D22, b.LFT.D22)
		}
		if !same(z0, z0Src) || !same(z1, z1Src) {
			t.Errorf("dt=%v: sources mutated", dt)
		}
	}
}

func TestAppendEmptyGainKeepsDims(t *testing.T) {
	empty, err := NewGain(&mat.Dense{}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	auto, err := New(mat.NewDense(1, 1, []float64{0.5}), nil, mat.NewDense(1, 1, []float64{2}), nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	for _, tc := range []struct {
		name    string
		a, b    *System
		n, m, p int
	}{
		{"gain+gain", empty, empty, 0, 0, 0},
		{"gain+autonomous", empty, auto, 1, 0, 1},
		{"autonomous+gain", auto, empty, 1, 0, 1},
	} {
		got, err := Append(tc.a, tc.b.Copy())
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		if n, m, p := got.Dims(); n != tc.n || m != tc.m || p != tc.p {
			t.Errorf("%s: Dims = (%d,%d,%d), want (%d,%d,%d)", tc.name, n, m, p, tc.n, tc.m, tc.p)
		}
		if err := got.Validate(); err != nil {
			t.Errorf("%s: Validate: %v", tc.name, err)
		}
	}
}

func TestInterconnectNilArgs(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	q := mat.NewDense(1, 1, nil)
	for name, call := range map[string]func() error{
		"Series(sys,nil)":   func() error { _, err := Series(sys, nil); return err },
		"Series(nil,sys)":   func() error { _, err := Series(nil, sys); return err },
		"Parallel(sys,nil)": func() error { _, err := Parallel(sys, nil); return err },
		"Parallel(nil,sys)": func() error { _, err := Parallel(nil, sys); return err },
		"Append(sys,nil)":   func() error { _, err := Append(sys, nil); return err },
		"Append(nil,sys)":   func() error { _, err := Append(nil, sys); return err },
		"BlkDiag(nil)":      func() error { _, err := BlkDiag(nil); return err },
		"BlkDiag(sys,nil)":  func() error { _, err := BlkDiag(sys, nil); return err },
		"Feedback(nil)":     func() error { _, err := Feedback(nil, sys, -1); return err },
		"Connect(nil)":      func() error { _, err := Connect(nil, q, []int{0}, []int{0}); return err },
		"Connect(sys,nilQ)": func() error { _, err := Connect(sys, nil, []int{0}, []int{0}); return err },
		"Augstate(nil)":     func() error { _, err := Augstate(nil); return err },
	} {
		if err := call(); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("%s: err = %v, want ErrInvalidArgument", name, err)
		}
	}
}

func TestInterconnectInvalidSampleTimeGain(t *testing.T) {
	g := &System{A: &mat.Dense{}, B: &mat.Dense{}, C: &mat.Dense{}, D: mat.NewDense(1, 1, []float64{2}), Dt: -1}
	if _, err := Append(g, g); !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("Append: err = %v, want ErrInvalidSampleTime", err)
	}
	if _, err := BlkDiag(g, g); !errors.Is(err, ErrInvalidSampleTime) {
		t.Errorf("BlkDiag: err = %v, want ErrInvalidSampleTime", err)
	}
}

func TestFeedbackRejectsSign(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, sign := range []float64{0, 2.5, math.NaN(), -0.5} {
		if _, err := Feedback(sys, nil, sign); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("sign %g: err = %v, want ErrInvalidArgument", sign, err)
		}
		if _, err := Feedback(sys, nil, sign, WithApproximatedDelays()); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("approximated sign %g: err = %v, want ErrInvalidArgument", sign, err)
		}
	}
}

func TestSeriesLFTZeroPortOperands(t *testing.T) {
	p, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	p.InputDelay = []float64{0.5}
	cl, err := Feedback(p, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !cl.HasInternalDelay() {
		t.Fatal("closed loop has no internal delay")
	}
	a2 := mat.NewDense(2, 2, []float64{-2, 1, 0, -3})
	noOut, err := New(a2, mat.NewDense(2, 1, []float64{1, 2}), &mat.Dense{}, &mat.Dense{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	noOutPad, err := New(a2, mat.NewDense(2, 1, []float64{1, 2}), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	noIn, err := New(a2, &mat.Dense{}, mat.NewDense(1, 2, []float64{1, -1}), &mat.Dense{}, 0)
	if err != nil {
		t.Fatal(err)
	}
	noInPad, err := New(a2, mat.NewDense(2, 1, nil), mat.NewDense(1, 2, []float64{1, -1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	check := func(name string, got, ref *System, wantM, wantP int) {
		t.Helper()
		n, m, pp := got.Dims()
		rn, _, _ := ref.Dims()
		if n != rn || m != wantM || pp != wantP {
			t.Fatalf("%s: dims (%d,%d,%d), want (%d,%d,%d)", name, n, m, pp, rn, wantM, wantP)
		}
		if !mat.EqualApprox(got.A, ref.A, 1e-12) {
			t.Errorf("%s: A = %v, want %v", name, mat.Formatted(got.A), mat.Formatted(ref.A))
		}
		if !slices.Equal(got.LFT.Tau, ref.LFT.Tau) || !mat.EqualApprox(got.LFT.D22, ref.LFT.D22, 1e-12) ||
			!mat.EqualApprox(got.LFT.B2, ref.LFT.B2, 1e-12) || !mat.EqualApprox(got.LFT.C2, ref.LFT.C2, 1e-12) {
			t.Errorf("%s: LFT mismatch", name)
		}
		if err := got.Validate(); err != nil {
			t.Errorf("%s: Validate: %v", name, err)
		}
	}
	got, err := Series(cl, noOut)
	if err != nil {
		t.Fatalf("Series(cl, noOut): %v", err)
	}
	ref, err := Series(cl, noOutPad)
	if err != nil {
		t.Fatal(err)
	}
	check("Series(cl, noOut)", got, ref, 1, 0)
	if !mat.EqualApprox(got.B, ref.B, 1e-12) || !mat.EqualApprox(got.LFT.D21, ref.LFT.D21, 1e-12) {
		t.Errorf("Series(cl, noOut): B/D21 mismatch")
	}

	got, err = Series(noIn, cl)
	if err != nil {
		t.Fatalf("Series(noIn, cl): %v", err)
	}
	ref, err = Series(noInPad, cl)
	if err != nil {
		t.Fatal(err)
	}
	check("Series(noIn, cl)", got, ref, 0, 1)
	if !mat.EqualApprox(got.C, ref.C, 1e-12) || !mat.EqualApprox(got.LFT.D12, ref.LFT.D12, 1e-12) {
		t.Errorf("Series(noIn, cl): C/D12 mismatch")
	}
}
