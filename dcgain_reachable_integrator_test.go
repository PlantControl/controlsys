package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// 2TUAD6 guards (w12 #16, w12f #31): a reachable, observable exact
// integrator gives ±Inf (MATLAB dcgain) even with input entries far below
// ‖A‖·ε or a large coupling into a fast stable mode. Both returned finite
// values on base 2e4a07d and pass on f86231f; the oracle is the exact
// rational limit on the float model.
func TestDCGain_ReachableObservableIntegratorInfinite(t *testing.T) {
	tiny := &System{
		A: mat.NewDense(2, 2, []float64{0, 0, 1, -1.3258457598967608e-08}),
		B: mat.NewDense(2, 1, []float64{7.29653485561532e-14, 1.0337733234010072e-13}),
		C: mat.NewDense(1, 2, []float64{0, 1}),
		D: mat.NewDense(1, 1, nil),
	}
	models := []*System{tiny}
	for _, a := range []float64{-1, -10, -250, -1000, 7, 250, 2.5e4} {
		for _, k := range []float64{10, 1e3, 1e4, 33000, 1e6} {
			sys, err := NewFromSlices(2, 1, 1, []float64{a, k, 0, 0}, []float64{1, 1}, []float64{1, 1}, []float64{0}, 0)
			if err != nil {
				t.Fatal(err)
			}
			models = append(models, sys)
		}
	}
	for _, sys := range models {
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("A=%v: %v", mat.Formatted(sys.A), err)
		}
		if got, want := g.At(0, 0), dcGainExactLimit(t, sys)[0][0]; got != want {
			t.Errorf("A=%v: DCGain = %v, exact limit %v", mat.Formatted(sys.A, mat.Squeeze()), got, want)
		}
	}
}
