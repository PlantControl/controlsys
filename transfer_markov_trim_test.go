package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// EHBRDP: an entry with resolved C·B has nothing to trim, so trim returns
// nil instead of a one-candidate slice per entry.
func TestMarkovTrimNothingToTrim(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 2, 0, 0.5, -3, 1, 0, 0.25, -2})
	B := mat.NewDense(3, 1, []float64{1, 0.5, -1})
	C := mat.NewDense(1, 3, []float64{1, 0, 0})
	sys, err := New(A, B, C, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	mt := newMarkovTrim(sys)
	num := []float64{1, 0.5, 2}
	if got := mt.trim(0, 0, 3, num, nil); got != nil {
		t.Fatalf("trim = %v, want nil", got)
	}
	if !raceEnabled {
		if a := testing.AllocsPerRun(100, func() { mt.trim(0, 0, 3, num, nil) }); a > 0 {
			t.Fatalf("trim allocs = %v, want 0", a)
		}
	}

	C0 := mat.NewDense(1, 3, []float64{0, 0, 1})
	B0 := mat.NewDense(3, 1, []float64{1, 0, 0})
	sys0, err := New(A, B0, C0, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	mt0 := newMarkovTrim(sys0)
	cands := mt0.trim(0, 0, 3, []float64{1e-17, 0.5, 2}, nil)
	if len(cands) == 0 || len(cands[len(cands)-1]) != 2 {
		t.Fatalf("trim of negligible C·B = %v, want candidates ending in [0.5 2]", cands)
	}
}
