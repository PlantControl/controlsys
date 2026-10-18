package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

func TestTunePIDOneGainInfeasiblePhaseMarginUnattainable(t *testing.T) {
	horner := func(p []float64, s complex128) complex128 {
		var v complex128
		for _, c := range p {
			v = v*s + complex(c, 0)
		}
		return v
	}
	for _, tc := range []struct {
		name     string
		family   PidtuneType
		num, den []float64
		wc, pm   float64
	}{
		{"I integrator pm45 auto", PidtuneI, []float64{1}, []float64{1, 0}, 0, 45},
		{"I integrator pm75 auto", PidtuneI, []float64{1}, []float64{1, 0}, 0, 75},
		{"I integrator pm45 wc1", PidtuneI, []float64{1}, []float64{1, 0}, 1, 45},
		{"I integrator pm75 wc1", PidtuneI, []float64{1}, []float64{1, 0}, 1, 75},
		{"I resonant pm45", PidtuneI, []float64{1}, []float64{1, .2, 1}, 1, 45},
		{"I resonant pm75", PidtuneI, []float64{1}, []float64{1, .2, 1}, 1, 75},
		{"I negative lag2 pm45", PidtuneI, []float64{-1}, []float64{1, 2, 1}, 1, 45},
		{"P double integrator pm45", PidtuneP, []float64{1}, []float64{1, 0, 0}, 1, 45},
	} {
		for _, maxEval := range []int{0, 300} {
			_, err := TunePID(context.Background(), makePlant(t, tc.num, tc.den), tc.family, PIDTuningOptions{CrossoverFrequency: tc.wc, PhaseMargin: tc.pm, MaxEvaluations: maxEval})
			te, ok := errors.AsType[*PIDTuningError](err)
			if !ok || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s max %d: err %v, want *PIDTuningError", tc.name, maxEval, err)
				continue
			}
			wc := te.Evidence.RequestedCrossover
			s := complex(0, wc)
			l := horner(tc.num, s) / horner(tc.den, s)
			if tc.family == PidtuneI {
				l /= s
			}
			best := math.Inf(1)
			for _, sign := range []complex128{1, -1} {
				pm := 180 + cmplx.Phase(sign*l)*180/math.Pi
				best = math.Min(best, math.Abs(math.Remainder(pm-tc.pm, 360)))
			}
			if !(wc > 0) || best <= 3 {
				t.Errorf("%s max %d: wc %g reaches PM within %g° of %g; target was attainable", tc.name, maxEval, wc, best, tc.pm)
			}
		}
	}
}
