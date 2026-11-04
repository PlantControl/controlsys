package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// emptyIOInternalDelaySystem builds an n-state system with no inputs or
// outputs and one internal delay; n may be zero.
func emptyIOInternalDelaySystem(t *testing.T, n int) *System {
	t.Helper()
	var A, B2, C2 *mat.Dense
	if n > 0 {
		A = mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2})
		B2 = mat.NewDense(2, 1, []float64{0.4, -0.7})
		C2 = mat.NewDense(1, 2, []float64{0.5, 1.1})
	}
	sys, err := New(A, nil, nil, nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{3}, B2, C2, nil, nil, mat.NewDense(1, 1, []float64{0.25})); err != nil {
		t.Fatal(err)
	}
	return sys
}

func assertSystemDims(t *testing.T, label string, sys *System, n, m, p int) {
	t.Helper()
	if err := sys.Validate(); err != nil {
		t.Fatalf("%s: Validate: %v", label, err)
	}
	if gn, gm, gp := sys.Dims(); gn != n || gm != m || gp != p {
		t.Fatalf("%s: dims (%d,%d,%d), want (%d,%d,%d)", label, gn, gm, gp, n, m, p)
	}
}

func assertInternalDelayModelEqual(t *testing.T, label string, got, want *System) {
	t.Helper()
	gn, gm, gp := got.Dims()
	assertSystemDims(t, label, got, gn, gm, gp)
	assertSystemDims(t, label, want, gn, gm, gp)
	if gn > 0 {
		assertMatNear(t, label+" A", got.A, want.A, 1e-14)
	}
	gl, wl := got.LFT, want.LFT
	if gl == nil || wl == nil {
		t.Fatalf("%s: LFT got %v want %v", label, gl, wl)
	}
	if len(gl.Tau) != len(wl.Tau) {
		t.Fatalf("%s: tau %v, want %v", label, gl.Tau, wl.Tau)
	}
	for i := range gl.Tau {
		if gl.Tau[i] != wl.Tau[i] {
			t.Fatalf("%s: tau %v, want %v", label, gl.Tau, wl.Tau)
		}
	}
	for _, b := range []struct {
		name      string
		got, want *mat.Dense
	}{
		{"B2", gl.B2, wl.B2}, {"C2", gl.C2, wl.C2}, {"D12", gl.D12, wl.D12},
		{"D21", gl.D21, wl.D21}, {"D22", gl.D22, wl.D22},
	} {
		if b.got == nil {
			t.Fatalf("%s: %s is nil", label, b.name)
		}
		assertMatNear(t, label+" "+b.name, b.got, b.want, 1e-14)
	}
}

// 3HZ6Q3: Copy kept empty internal-delay blocks as nil, which Series then
// dereferenced.
func TestCopyInternalDelayEmptyBlocksNonNil(t *testing.T) {
	for _, n := range []int{0, 2} {
		sys := emptyIOInternalDelaySystem(t, n)
		assertInternalDelayModelEqual(t, "Copy", sys.Copy(), sys)
		assertInternalDelayModelEqual(t, "Copy.Copy", sys.Copy().Copy(), sys)
	}
}

// 3HZ6Q3
func TestSeriesInternalDelayEmptyIO(t *testing.T) {
	for _, n := range []int{0, 2} {
		sys := emptyIOInternalDelaySystem(t, n)
		want, err := Append(sys, sys)
		if err != nil {
			t.Fatal(err)
		}
		got, err := Series(sys, sys.Copy())
		if err != nil {
			t.Fatalf("n=%d: %v", n, err)
		}
		assertSystemDims(t, "Series", got, 2*n, 0, 0)
		assertInternalDelayModelEqual(t, "Series", got, want)
	}
}

// HN6Z3U: with zero loop channels feedback leaves both models open, so the
// result is their block-diagonal append, internal delays plant first.
func TestFeedbackInternalDelayEmptyLoop(t *testing.T) {
	for _, n := range []int{0, 2} {
		for _, sign := range []float64{-1, 1} {
			sys := emptyIOInternalDelaySystem(t, n)
			want, err := Append(sys, sys)
			if err != nil {
				t.Fatal(err)
			}
			got, err := Feedback(sys, sys.Copy(), sign)
			if err != nil {
				t.Fatalf("n=%d sign=%g: %v", n, sign, err)
			}
			assertSystemDims(t, "Feedback", got, 2*n, 0, 0)
			assertInternalDelayModelEqual(t, "Feedback", got, want)
		}
	}
}

// KM7GOG: a zero-input model must augment like the same model with one
// all-zero input column, minus that column.
func TestAugmentInternalDelayOutputsNoInputs(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2})
	C := mat.NewDense(2, 2, []float64{1, 0.5, -0.4, 2})
	B2 := mat.NewDense(2, 1, []float64{0.4, -0.7})
	C2 := mat.NewDense(1, 2, []float64{0.5, 1.1})
	D12 := mat.NewDense(2, 1, []float64{0.3, -0.6})
	D22 := mat.NewDense(1, 1, []float64{0.25})

	sys, err := New(A, nil, C, nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{1}, B2, C2, D12, nil, D22); err != nil {
		t.Fatal(err)
	}
	ref, err := New(A, mat.NewDense(2, 1, nil), C, mat.NewDense(2, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := ref.SetInternalDelay([]float64{1}, B2, C2, D12, mat.NewDense(1, 1, nil), D22); err != nil {
		t.Fatal(err)
	}

	got, err := sys.AugmentInternalDelayOutputs("w")
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.AugmentInternalDelayOutputs("w")
	if err != nil {
		t.Fatal(err)
	}
	assertSystemDims(t, "Augment", got, 2, 0, 3)
	assertMatNear(t, "A", got.A, want.A, 0)
	assertMatNear(t, "C", got.C, want.C, 0)
	assertMatNear(t, "B2", got.LFT.B2, want.LFT.B2, 0)
	assertMatNear(t, "C2", got.LFT.C2, want.LFT.C2, 0)
	assertMatNear(t, "D12", got.LFT.D12, want.LFT.D12, 0)
	assertMatNear(t, "D22", got.LFT.D22, want.LFT.D22, 0)
	if len(got.OutputName) != 3 || got.OutputName[2] != want.OutputName[2] {
		t.Fatalf("OutputName %v, want %v", got.OutputName, want.OutputName)
	}
}

// KM7GOG: a static model must augment like the same model with one
// decoupled zero state, minus that state.
func TestAugmentInternalDelayOutputsNoStates(t *testing.T) {
	D := mat.NewDense(2, 2, []float64{1, 0.5, -0.4, 2})
	D12 := mat.NewDense(2, 1, []float64{0.3, -0.6})
	D21 := mat.NewDense(1, 2, []float64{0.7, -0.2})
	D22 := mat.NewDense(1, 1, []float64{0.25})

	sys, err := New(nil, nil, nil, D, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{1}, nil, nil, D12, D21, D22); err != nil {
		t.Fatal(err)
	}
	ref, err := New(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 2, nil), mat.NewDense(2, 1, nil), D, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := ref.SetInternalDelay([]float64{1}, mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil), D12, D21, D22); err != nil {
		t.Fatal(err)
	}

	got, err := sys.AugmentInternalDelayOutputs("w")
	if err != nil {
		t.Fatal(err)
	}
	want, err := ref.AugmentInternalDelayOutputs("w")
	if err != nil {
		t.Fatal(err)
	}
	assertSystemDims(t, "Augment", got, 0, 2, 3)
	assertMatNear(t, "D", got.D, want.D, 0)
	assertMatNear(t, "D12", got.LFT.D12, want.LFT.D12, 0)
	assertMatNear(t, "D21", got.LFT.D21, want.LFT.D21, 0)
	assertMatNear(t, "D22", got.LFT.D22, want.LFT.D22, 0)
}

// KM7GOG: a model without states or inputs cannot hold the appended
// outputs (gonum has no p×0 D), so it reports ErrDimensionMismatch, as New
// does for such a static gain.
func TestAugmentInternalDelayOutputsEmptyIO(t *testing.T) {
	sys := emptyIOInternalDelaySystem(t, 2)
	got, err := sys.AugmentInternalDelayOutputs("w")
	if err != nil {
		t.Fatal(err)
	}
	assertSystemDims(t, "Augment", got, 2, 0, 1)
	assertMatNear(t, "C", got.C, sys.LFT.C2, 0)
	assertMatNear(t, "D12", got.LFT.D12, sys.LFT.D22, 0)
	assertMatNear(t, "B2", got.LFT.B2, sys.LFT.B2, 0)

	if _, err := emptyIOInternalDelaySystem(t, 0).AugmentInternalDelayOutputs("w"); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("n=0: err %v, want ErrDimensionMismatch", err)
	}
}

// HN6Z3U: a plant without inputs feeding a controller without outputs closes
// no loop, so the states and internal delays match the open cascade.
func TestFeedbackInternalDelayOneSidedLoop(t *testing.T) {
	plant, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.3, -0.2, -2}), nil,
		mat.NewDense(2, 2, []float64{1, 0.5, -0.4, 2}), nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := plant.SetInternalDelay([]float64{2}, mat.NewDense(2, 1, []float64{0.4, -0.7}),
		mat.NewDense(1, 2, []float64{0.5, 1.1}), mat.NewDense(2, 1, []float64{0.3, -0.6}), nil,
		mat.NewDense(1, 1, []float64{0.25})); err != nil {
		t.Fatal(err)
	}
	ctrl, err := New(
		mat.NewDense(2, 2, []float64{-0.5, 0.1, 0, -1.5}),
		mat.NewDense(2, 2, []float64{1, -0.3, 0.2, 0.8}), nil, nil, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := ctrl.SetInternalDelay([]float64{1}, mat.NewDense(2, 1, []float64{0.6, 0.1}),
		mat.NewDense(1, 2, []float64{-0.4, 0.9}), nil, mat.NewDense(1, 2, []float64{0.2, -0.5}),
		mat.NewDense(1, 1, []float64{0.1})); err != nil {
		t.Fatal(err)
	}
	cascade, err := Series(plant, ctrl)
	if err != nil {
		t.Fatal(err)
	}
	for _, sign := range []float64{-1, 1} {
		got, err := Feedback(plant, ctrl, sign)
		if err != nil {
			t.Fatalf("sign=%g: %v", sign, err)
		}
		assertSystemDims(t, "Feedback", got, 4, 0, 2)
		assertMatNear(t, "A", got.A, cascade.A, 1e-14)
		assertMatNear(t, "B2", got.LFT.B2, cascade.LFT.B2, 1e-14)
		assertMatNear(t, "C2", got.LFT.C2, cascade.LFT.C2, 1e-14)
		assertMatNear(t, "D22", got.LFT.D22, cascade.LFT.D22, 1e-14)
		assertMatNear(t, "C", got.C, mat.NewDense(2, 4, []float64{1, 0.5, 0, 0, -0.4, 2, 0, 0}), 0)
		assertMatNear(t, "D12", got.LFT.D12, mat.NewDense(2, 2, []float64{0.3, 0, -0.6, 0}), 0)
	}
}
