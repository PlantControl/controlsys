package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// An upper-triangular A whose stiff mode leads (as the standard form of a
// rank-deficient E with its near-infinite pole) and whose B row of that
// mode is 1e12 keeps its order in the modal hold: swapped slow-first, the
// slow inputs became b_s − X·b_f, cancelling to ε·1e12 absolute (EJDS3U
// R1). Slow-first and interleaved forms keep today's order; all three
// hold DC within the descriptor bar.
func TestC2DStiffHoldFastLeadingOrder(t *testing.T) {
	const lam = -3e12
	for _, tc := range []struct {
		name string
		a    []float64
		b, c []float64
	}{
		{"fast first", []float64{
			lam, 0.2 * lam, -0.1 * lam,
			0, -1, 0.5,
			0, 0, -0.3}, []float64{-lam, 1, -0.5}, []float64{0.3, 1, -0.7}},
		{"slow first", []float64{
			-0.3, 0.5, 0.2,
			0, -1, 0.4,
			0, 0, lam}, []float64{-0.5, 1, -lam}, []float64{-0.7, 1, 0.3}},
		{"interleaved", []float64{
			-1, 0.4, 0.5,
			0, lam, 0.1 * lam,
			0, 0, -0.3}, []float64{1, -lam, -0.5}, []float64{1, 0.3, -0.7}},
	} {
		sys, err := New(mat.NewDense(3, 3, tc.a), mat.NewDense(3, 1, tc.b), mat.NewDense(1, 3, tc.c), mat.NewDense(1, 1, []float64{0.4}), 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, dt := range []float64{1e-2, 1} {
			for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
				out, err := sys.C2D(dt, C2DOptions{Method: m})
				if err != nil {
					t.Fatalf("%s dt=%g %s: %v", tc.name, dt, m, err)
				}
				bars, ok := descriptorDCBars(t, sys, out)
				if !ok || !(bars <= 1) {
					t.Errorf("%s dt=%g %s: DC %.3g bars off G(0) (ok %v)", tc.name, dt, m, bars, ok)
				}
			}
		}
	}
}
