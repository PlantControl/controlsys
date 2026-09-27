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
	if _, m, p := arr.Dims(); m != 1 || p != 1 {
		t.Fatalf("Dims() = (_, %d, %d), want (_, 1, 1)", m, p)
	}
	if got, ok, err := arr.Model(0, 1); err != nil || ok || got != nil {
		t.Fatalf("void Model(0,1) = (%v, %v, %v), want nil,false,nil", got, ok, err)
	}
	got, ok, err := arr.Model(1, 0)
	if err != nil || !ok {
		t.Fatalf("Model(1,0) ok=%v err=%v, want ok", ok, err)
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
	if got, ok, err := selected.ModelFlat(0); err != nil || ok || got != nil {
		t.Fatalf("selected void = (%v, %v, %v), want nil,false,nil", got, ok, err)
	}
	if _, ok, err := selected.ModelFlat(1); err != nil || !ok {
		t.Fatalf("selected model ok=%v err=%v, want ok", ok, err)
	}

	stacked, err := StackModelArrays(left, right)
	if err != nil {
		t.Fatalf("StackModelArrays: %v", err)
	}
	if got, want := stacked.Shape(), []int{2, 2}; !sameInts(got, want) {
		t.Fatalf("stacked Shape() = %v, want %v", got, want)
	}
	if got, ok, err := stacked.Model(0, 1); err != nil || ok || got != nil {
		t.Fatalf("stacked void = (%v, %v, %v), want nil,false,nil", got, ok, err)
	}
	if got, ok, err := stacked.Model(1, 0); err != nil || !ok || got.A.At(0, 0) != b.A.At(0, 0) {
		t.Fatalf("stacked Model(1,0) = (%v, %v, %v), want copied b", got, ok, err)
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
	incompatible.models[0].Dt = 0.2
	if _, err := StackModelArrays(left, incompatible); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("StackModelArrays header mismatch err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := StackModelArrays(); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("StackModelArrays empty err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := StackModelArrays(left, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("StackModelArrays nil err = %v, want ErrDimensionMismatch", err)
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
	if _, ok, err := concatenated.ModelFlat(2); err != nil || !ok {
		t.Fatalf("concatenated final model ok=%v err=%v, want ok", ok, err)
	}
}

func TestModelArrayRejectsInvalidCompatibility(t *testing.T) {
	sys := modelArrayTestSystem(t, 0, []float64{-1, 2, -3, -4})
	badDt := modelArrayTestSystem(t, 0.1, []float64{0.8, 0.2, -0.1, 0.6})
	badDims, err := NewGain(mat.NewDense(2, 1, []float64{1, 2}), 0)
	if err != nil {
		t.Fatal(err)
	}

	if _, err := NewModelArray([]int{2}, []*System{sys, badDt}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("mixed sample time err = %v, want ErrDimensionMismatch", err)
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
	if got := len(fresp.Responses); got != 3 {
		t.Fatalf("len(freq responses) = %d, want 3", got)
	}
	if fresp.Responses[1] != nil || !fresp.Void[1] {
		t.Fatalf("void freq response = (%v, %v), want nil,true", fresp.Responses[1], fresp.Void[1])
	}
	if got := fresp.Responses[0].At(0, 0, 0); got == 0 {
		t.Fatal("first frequency response is zero; expected evaluated model")
	}

	step, err := arr.Step(0.2)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if got, want := step.Shape, []int{3}; !sameInts(got, want) {
		t.Fatalf("step shape = %v, want %v", got, want)
	}
	if step.Responses[1] != nil || !step.Void[1] {
		t.Fatalf("void step response = (%v, %v), want nil,true", step.Responses[1], step.Void[1])
	}
	if got := len(step.Responses[0].T); got == 0 {
		t.Fatal("first step response has no time samples")
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
