package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSminreal_DecoupledState(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -2, 0, 0, 0, -3}),
		mat.NewDense(3, 1, []float64{1, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	red, err := Sminreal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := red.Dims()
	if n != 2 {
		t.Fatalf("order = %d, want 2", n)
	}

	poles, _ := red.Poles()
	sortPolesDecomp(poles)
	wantPoles := []complex128{-3, -1}
	for i, p := range poles {
		if cmplx.Abs(p-wantPoles[i]) > 1e-10 {
			t.Errorf("pole[%d] = %v, want %v", i, p, wantPoles[i])
		}
	}

	dc, _ := red.DCGain()
	want := 1.0 + 1.0/3.0
	if math.Abs(dc.At(0, 0)-want) > 1e-10 {
		t.Errorf("dcgain = %g, want %g", dc.At(0, 0), want)
	}
}

func TestSminreal_CoupledThroughA(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -2, 0, 0, 0, -3}),
		mat.NewDense(3, 1, []float64{0, 1, 1}),
		mat.NewDense(1, 3, []float64{1, 0, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	red, err := Sminreal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := red.Dims()
	if n != 3 {
		t.Errorf("order = %d, want 3 (no reduction)", n)
	}
}

func TestSminreal_UncontrollableUncoupled(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -2, 0, 0, 0, -3}),
		mat.NewDense(3, 1, []float64{1, 1, 0}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	red, err := Sminreal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := red.Dims()
	if n != 2 {
		t.Fatalf("order = %d, want 2", n)
	}

	poles, _ := red.Poles()
	sortPolesDecomp(poles)
	wantPoles := []complex128{-2, -1}
	for i, p := range poles {
		if cmplx.Abs(p-wantPoles[i]) > 1e-10 {
			t.Errorf("pole[%d] = %v, want %v", i, p, wantPoles[i])
		}
	}
}

func TestSminreal_NoReduction(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	red, err := Sminreal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := red.Dims()
	if n != 2 {
		t.Errorf("order = %d, want 2", n)
	}
}

func TestSminreal_Empty(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	red, err := Sminreal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := red.Dims()
	if n != 0 {
		t.Errorf("order = %d, want 0", n)
	}
}

func TestSminreal_AllRemoved(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{0, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{3}), 0)
	if err != nil {
		t.Fatal(err)
	}

	red, err := Sminreal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := red.Dims()
	if n != 0 {
		t.Errorf("order = %d, want 0", n)
	}
	if red.D.At(0, 0) != 3 {
		t.Errorf("D = %g, want 3", red.D.At(0, 0))
	}
}

func TestSminreal_MIMO(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -2, 0, 0, 0, -3}),
		mat.NewDense(3, 2, []float64{1, 0, 0, 0, 0, 1}),
		mat.NewDense(2, 3, []float64{1, 0, 0, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	red, err := Sminreal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := red.Dims()
	if n != 2 {
		t.Errorf("order = %d, want 2 (state 1 removed)", n)
	}
}

func sortPolesDecomp(poles []complex128) {
	sort.Slice(poles, func(i, j int) bool {
		if real(poles[i]) != real(poles[j]) {
			return real(poles[i]) < real(poles[j])
		}
		return imag(poles[i]) < imag(poles[j])
	})
}
