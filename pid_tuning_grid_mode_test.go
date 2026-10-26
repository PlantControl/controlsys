package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// ZHTUBP: with wc = 1 the 241-point search grid over [wc/100, 100·wc]
// samples 10^0.4 exactly, where this realization has an undamped mode
// (eigenvalues 0, 0, ±2.5118864315095806j), so the response there is not
// finite and TunePID(I) failed with ErrInvalidArgument. A grid sample on a
// jω-axis mode moves off it; the plant is valid, so the result is a
// stable design or unattainable, as MATLAB pidtune, whose I loop at wc = 1
// is unstable. Oracle: closed-loop poles of the realization for any design.
func TestTunePIDGridSampleOnImaginaryAxisModeIsNotFatal(t *testing.T) {
	s := fzDecode(t, 0x80, [fzNVals]float64{-57.8, -1.5, 0.4, -378, 0, 0, 6.2, -63, -1.5, -1, 0.7, 1.1, -10, -10, -10, -10})
	sys := s.System(t, fzSysOpts{})
	if !fzHasMode(s.Eig, complex(0, math.Pow(10, .4))) {
		t.Fatalf("eigenvalues %v lack the jω mode at 10^0.4", s.Eig)
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
