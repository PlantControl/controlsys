package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestModelArrayPreservesShapeMetadataAndVoidEntries(t *testing.T) {
	sys1 := modelArrayTestSystem(t, 0, []float64{-1, 2, -3, -4})
	sys2 := modelArrayTestSystem(t, 0, []float64{-2, 1, -4, -5})
	if err := sys1.SetInputName("command"); err != nil {
		t.Fatal(err)
	}
	if err := sys1.SetOutputName("position"); err != nil {
		t.Fatal(err)
	}
	if err := sys2.SetInputName("command"); err != nil {
		t.Fatal(err)
	}
	if err := sys2.SetOutputName("position"); err != nil {
		t.Fatal(err)
	}

	arr, err := NewModelArray([]int{2, 2}, []*System{sys1, nil, sys2, nil})
	if err != nil {
		t.Fatalf("NewModelArray: %v", err)
	}

	if got, want := arr.Shape(), []int{2, 2}; !sameInts(got, want) {
		t.Fatalf("Shape() = %v, want %v", got, want)
	}
	if got := arr.Len(); got != 4 {
		t.Fatalf("Len() = %d, want 4", got)
	}
	if p, m := arr.IOSize(); m != 1 || p != 1 {
		t.Fatalf("IOSize() = (%d, %d), want (1, 1)", p, m)
	}
	if _, err := arr.Model(0, 1); !errors.Is(err, ErrVoidModel) {
		t.Fatalf("void Model(0,1) err = %v, want ErrVoidModel", err)
	}
	if void, err := arr.IsVoid(0, 1); err != nil || !void {
		t.Fatalf("IsVoid(0,1) = %v, %v; want true", void, err)
	}
	if void, err := arr.IsVoid(1, 0); err != nil || void {
		t.Fatalf("IsVoid(1,0) = %v, %v; want false", void, err)
	}
	got, err := arr.Model(1, 0)
	if err != nil {
		t.Fatalf("Model(1,0): %v", err)
	}
	got.A.Set(0, 0, 99)
	if sys2.A.At(0, 0) == 99 {
		t.Fatal("Model returned aliased system")
	}
	if got := arr.InputName(); !sameStrings(got, []string{"command"}) {
		t.Fatalf("InputName() = %v", got)
	}
	if got := arr.OutputName(); !sameStrings(got, []string{"position"}) {
		t.Fatalf("OutputName() = %v", got)
	}
}

func TestModelArraySelectAndStack(t *testing.T) {
	a := modelArrayTestSystem(t, 0.1, []float64{0.8, 0.2, -0.1, 0.6})
	b := modelArrayTestSystem(t, 0.1, []float64{0.7, 0.1, -0.2, 0.5})
	c := modelArrayTestSystem(t, 0.1, []float64{0.6, 0.3, -0.3, 0.4})

	left, err := NewModelArray([]int{2}, []*System{a, nil})
	if err != nil {
		t.Fatalf("left array: %v", err)
	}
	right, err := NewModelArray([]int{2}, []*System{b, c})
	if err != nil {
		t.Fatalf("right array: %v", err)
	}
	selected, err := left.SelectFlat(1, 0)
	if err != nil {
		t.Fatalf("SelectFlat: %v", err)
	}
	if _, err := selected.ModelFlat(0); !errors.Is(err, ErrVoidModel) {
		t.Fatalf("selected void err = %v, want ErrVoidModel", err)
	}
	if _, err := selected.ModelFlat(1); err != nil {
		t.Fatalf("selected model: %v", err)
	}
	if _, err := left.SelectFlat(1); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("all-void selection err = %v, want ErrInvalidArgument", err)
	}

	stacked, err := StackModelArrays(left, right)
	if err != nil {
		t.Fatalf("StackModelArrays: %v", err)
	}
	if got, want := stacked.Shape(), []int{2, 2}; !sameInts(got, want) {
		t.Fatalf("stacked Shape() = %v, want %v", got, want)
	}
	if _, err := stacked.Model(0, 1); !errors.Is(err, ErrVoidModel) {
		t.Fatalf("stacked void err = %v, want ErrVoidModel", err)
	}
	if got, err := stacked.Model(1, 0); err != nil || got.A.At(0, 0) != b.A.At(0, 0) {
		t.Fatalf("stacked Model(1,0) = (%v, %v), want copied b", got, err)
	}

	incompatible, err := NewModelArray([]int{1}, []*System{c})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := StackModelArrays(left, incompatible); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("StackModelArrays shape mismatch err = %v, want ErrDimensionMismatch", err)
	}

	incompatible, err = NewModelArray([]int{2}, []*System{b, c})
	if err != nil {
		t.Fatal(err)
	}
	incompatible.dt = 0.2
	if _, err := StackModelArrays(left, incompatible); !errors.Is(err, ErrDomainMismatch) {
		t.Fatalf("StackModelArrays sample-time mismatch err = %v, want ErrDomainMismatch", err)
	}
	if _, err := StackModelArrays(); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("StackModelArrays empty err = %v, want ErrInvalidArgument", err)
	}
	if _, err := StackModelArrays(left, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("StackModelArrays nil err = %v, want ErrInvalidArgument", err)
	}
}

func TestConcatModelArraysFlattensCompatibleArrays(t *testing.T) {
	a := modelArrayTestSystem(t, 0, []float64{-1, 2, -3, -4})
	b := modelArrayTestSystem(t, 0, []float64{-2, 1, -4, -5})
	left, err := NewModelArray([]int{1, 2}, []*System{a, nil})
	if err != nil {
		t.Fatal(err)
	}
	right, err := NewModelArray([]int{1}, []*System{b})
	if err != nil {
		t.Fatal(err)
	}

	concatenated, err := ConcatModelArrays(left, right)
	if err != nil {
		t.Fatalf("ConcatModelArrays: %v", err)
	}
	if got, want := concatenated.Shape(), []int{3}; !sameInts(got, want) {
		t.Fatalf("concatenated Shape() = %v, want %v", got, want)
	}
	if _, err := concatenated.ModelFlat(2); err != nil {
		t.Fatalf("concatenated final model: %v", err)
	}
}

func TestModelArrayRejectsInvalidCompatibility(t *testing.T) {
	sys := modelArrayTestSystem(t, 0, []float64{-1, 2, -3, -4})
	badDt := modelArrayTestSystem(t, 0.1, []float64{0.8, 0.2, -0.1, 0.6})
	badDims, err := NewGain(mat.NewDense(2, 1, []float64{1, 2}), 0)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := NewModelArray([]int{2}, []*System{sys, badDt}); !errors.Is(err, ErrDomainMismatch) {
		t.Fatalf("mixed sample time err = %v, want ErrDomainMismatch", err)
	}
	if _, err := NewModelArray([]int{2}, []*System{nil, nil}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("all-void err = %v, want ErrInvalidArgument", err)
	}
	renamed := sys.Copy()
	renamed.InputName = []string{"other"}
	named := sys.Copy()
	named.InputName = []string{"u"}
	if _, err := NewModelArray([]int{2}, []*System{named, renamed}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("signal-name mismatch err = %v, want ErrInvalidArgument", err)
	}
	if _, err := NewModelArray([]int{2}, []*System{sys, badDims}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("mixed dimensions err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := NewModelArray([]int{2, 2}, []*System{sys}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("shape/product mismatch err = %v, want ErrDimensionMismatch", err)
	}
}

func TestModelArrayBatchResponsesPreserveResultShape(t *testing.T) {
	sys1 := modelArrayTestSystem(t, 0, []float64{-1, 2, -3, -4})
	sys2 := modelArrayTestSystem(t, 0, []float64{-2, 1, -4, -5})
	arr, err := NewModelArray([]int{3}, []*System{sys1, nil, sys2})
	if err != nil {
		t.Fatalf("NewModelArray: %v", err)
	}

	fresp, err := arr.FreqResponse([]float64{0.5, 1.0})
	if err != nil {
		t.Fatalf("FreqResponse: %v", err)
	}
	if got, want := fresp.Shape, []int{3}; !sameInts(got, want) {
		t.Fatalf("freq shape = %v, want %v", got, want)
	}
	if _, err := fresp.ResponseFlat(1); !errors.Is(err, ErrVoidModel) {
		t.Fatalf("void freq response err = %v, want ErrVoidModel", err)
	}
	if _, err := fresp.ResponseFlat(3); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("out-of-range freq response err = %v, want ErrInvalidArgument", err)
	}
	first, err := fresp.ResponseFlat(0)
	if err != nil {
		t.Fatal(err)
	}
	want, err := sys1.FreqResponse([]float64{0.5, 1.0})
	if err != nil {
		t.Fatal(err)
	}
	if got := first.At(0, 0, 0); got != want.At(0, 0, 0) {
		t.Fatalf("first frequency response = %v, want %v", got, want.At(0, 0, 0))
	}

	step, err := arr.Step(0.2)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if got, want := step.Shape, []int{3}; !sameInts(got, want) {
		t.Fatalf("step shape = %v, want %v", got, want)
	}
	if _, err := step.ResponseFlat(1); !errors.Is(err, ErrVoidModel) {
		t.Fatalf("void step response err = %v, want ErrVoidModel", err)
	}
	stepFirst, err := step.ResponseFlat(0)
	if err != nil || len(stepFirst.T) == 0 {
		t.Fatalf("first step response = %v, %v; want samples", stepFirst, err)
	}
}

func modelArrayTestSystem(t *testing.T, dt float64, a []float64) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(2, 2, a),
		mat.NewDense(2, 1, []float64{1, -2}),
		mat.NewDense(1, 2, []float64{3, -1}),
		mat.NewDense(1, 1, []float64{0.25}),
		dt,
	)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	return sys
}

func sameInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

func sameStrings(a, b []string) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
