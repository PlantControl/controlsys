package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// An upper-triangular A whose stiff mode leads and whose B row of that mode
// is 1e15 (the standard form of stfr/45, a rank-deficient E with its
// near-infinite pole kept) keeps its order in the modal hold: swapped
// slow-first, the slow inputs became b_s − X·b_f, cancelling to ε·1e15
// absolute, 5e10 descriptor bars at DC (EJDS3U R1). Slow-first and
// interleaved forms keep today's order: slow-first holds DC to 1e-10
// relative (the explicit form's own descriptor bar, with ‖A‖ ~ 1e15, would
// not tell); the interleaved form is still swapped (1.8e-5 here) and only
// runs.
func TestC2DStiffHoldFastLeadingOrder(t *testing.T) {
	const lam = -3e12
	for _, tc := range []struct {
		name  string
		a     []float64
		b, c  []float64
		loose bool
	}{
		{"fast first (stfr/45)", []float64{
			-1.9486202866615216e+14, -2.1022453927409644e+14, -7.082096004462772e+14, 4.303792122621075e+11,
			0, 1.5537522208719299, 1.20133122486868, -1.8293835018126303,
			0, 0, -2.32201795171109, -0.9882340534510028,
			0, 0, 0, -0.7345007216279068},
			[]float64{2.1916722254020552e+15, -10.19979710322142, 2.5075922790185228, -3.6371289847345825},
			[]float64{-0.7133888798079264, 0.17292946916213467, -0.5896386127533522, -1.3423937437825788}, false},
		{"slow first", []float64{
			-0.3, 0.5, 0.2,
			0, -1, 0.4,
			0, 0, lam}, []float64{-0.5, 1, -lam}, []float64{-0.7, 1, 0.3}, false},
		{"interleaved", []float64{
			-1, 0.4, 0.5,
			0, lam, 0.1 * lam,
			0, 0, -0.3}, []float64{1, -lam, -0.5}, []float64{1, 0.3, -0.7}, true},
	} {
		n := len(tc.b)
		sys, err := New(mat.NewDense(n, n, tc.a), mat.NewDense(n, 1, tc.b), mat.NewDense(1, n, tc.c), mat.NewDense(1, 1, []float64{-0.057485455287890286}), 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, dt := range []float64{1e-2, 1} {
			for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
				out, err := sys.C2D(dt, C2DOptions{Method: m})
				if err != nil {
					t.Fatalf("%s dt=%g %s: %v", tc.name, dt, m, err)
				}
				g0, _, ok0 := exactDCGainAtNonsingular(nil, sys.A, sys.B, sys.C, sys.D, 0)
				h, _, ok := exactDCGainAtNonsingular(nil, out.A, out.B, out.C, out.D, 1)
				if !ok0 || !ok {
					t.Fatalf("%s dt=%g %s: exact DC gain singular", tc.name, dt, m)
				}
				if rel := math.Abs(h.At(0, 0)-g0.At(0, 0)) / math.Abs(g0.At(0, 0)); !tc.loose && !(rel <= 1e-10) {
					t.Errorf("%s dt=%g %s: DC %.3g relative off G(0)", tc.name, dt, m, rel)
				}
			}
		}
	}
}
