package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// ZHTUBP: with wc = 1 the 241-point search grid over [wc/100, 100·wc]
// samples 10^0.4, where this plant has an undamped mode (eigenvalues 0, 0,
// ±jω), so the response there is not finite and TunePID(I) failed with
// ErrInvalidArgument. ω is the grid sample as the search computes it, so the
// mode is exact on every CPU (QS7HIC: a decoded mode rounds differently from
// the grid). A grid sample on a jω-axis mode moves off it; the plant is
// valid, so the result is a stable design or unattainable, as MATLAB
// pidtune, whose I loop at wc = 1 is unstable. Oracle: closed-loop poles of
// the realization for any design.
func TestTunePIDGridSampleOnImaginaryAxisModeIsNotFatal(t *testing.T) {
	const points, k = 241, 144
	lo, hi := 0.01, 100.0
	w := math.Exp(math.Log(lo) + float64(k)*math.Log(hi/lo)/float64(points-1))
	if math.Abs(w-math.Pow(10, .4)) > 1e-14 {
		t.Fatalf("grid sample %d is %v, not 10^0.4", k, w)
	}
	sys, err := NewFromSlices(4, 1, 1,
		[]float64{0, w, 0.5, -1.25, -w, 0, 0.75, 0.3, 0, 0, 0, 1, 0, 0, 0, 0},
		[]float64{0.2, -0.4, 0.6, 1}, []float64{1, 0.5, -0.7, 0.9}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if g, err := sys.EvalFr(complex(0, w)); err == nil && !cmplx.IsInf(g[0][0]) && !cmplx.IsNaN(g[0][0]) {
		t.Fatalf("response at the grid sample jω = %vj is %v, want non-finite", w, g[0][0])
	}
	for _, wc := range []float64{1, 0} {
		r, err := TunePID(context.Background(), sys, PidtuneI, PIDTuningOptions{CrossoverFrequency: wc})
		pidTuningCheckStableOrUnattainable(t, "I", sys, nil, nil, r, err)
	}
}

func fzHasMode(eig []complex128, m complex128) bool {
	for _, e := range eig {
		if cmplx.Abs(e-m) < 1e-12*cmplx.Abs(m) {
			return true
		}
	}
	return false
}
