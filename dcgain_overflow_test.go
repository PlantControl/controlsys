package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A3QHIT: the explicit DC solve succeeded but X = A⁻¹B overflowed, and C·X
// gave NaN with a nil error; the exact limit decides, else ErrOverflow.
func TestDCGain_OverflowedSolveUsesExactLimit(t *testing.T) {
	for _, c := range []struct {
		name string
		a    []float64
		b    []float64
		c    []float64
		dt   float64
		want float64
	}{
		{"cancel", []float64{-1e-300, 0, 0, -1e-300}, []float64{1e200, 1e200}, []float64{1, -1}, 0, 0},
		{"discreteCancel", []float64{0.5, 0, 0, 0.5}, []float64{1e308, 1e308}, []float64{1, -1}, 1, 0},
	} {
		sys, err := New(mat.NewDense(2, 2, c.a), mat.NewDense(2, 1, c.b), mat.NewDense(1, 2, c.c), mat.NewDense(1, 1, nil), c.dt)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if got := g.At(0, 0); got != c.want {
			t.Errorf("%s: DCGain = %v, want %v", c.name, got, c.want)
		}
	}
}

func TestDCGain_FiniteBeyondRangeIsErrOverflow(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-1e-300}), mat.NewDense(1, 1, []float64{1e200}), mat.NewDense(1, 1, []float64{1e10}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	g, err := sys.DCGain()
	if !errors.Is(err, ErrOverflow) {
		var v float64
		if g != nil {
			v = g.At(0, 0)
		}
		t.Fatalf("DCGain = %v, %v; want ErrOverflow (exact 1e510)", v, err)
	}
	if g != nil {
		t.Fatal("non-nil gain with error")
	}
}
