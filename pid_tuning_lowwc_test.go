package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// DJZXQB: an explicit crossover two decades over the HXK42S plant's exact
// double pole at z = 1 put the band floor wc/100 where the point solve
// cannot resolve |G| ~ 1e13, and TunePID failed with ErrInvalidArgument.
// MATLAB pidtune(G,'pid',0.0035) returns a design. Designs must be
// Schur-stable exactly, meet their reported margin at their crossover, and
// at 0.0035, the ergo case, keep |S| <= 4, all checked on the factored
// plant, which is exact here. Explicit crossovers do not bound |S|: the
// 0.001 PID design keeps |L| = 1.13 at -178°.
func TestTunePIDExplicitLowCrossoverUnitPoles(t *testing.T) {
	num := []float64{-0.14994524256120972}
	den := []float64{1, -2.9872237308563525, 2.974447461712705, -0.9872237308563525}
	const r3, dt = 0.9872237308563525, 0.018062614213725733
	for _, wc := range []float64{0.0035, 0.001} {
		for _, fam := range []PidtuneType{PidtunePID, PidtunePIDF, PidtunePD, PidtunePI, PidtuneP} {
			name := fmt.Sprintf("wc %g %s", wc, fam)
			r, err := TunePID(context.Background(), makeDiscretePlant(t, num, den, dt), fam, PIDTuningOptions{CrossoverFrequency: wc})
			if fam == PidtunePI || fam == PidtuneP {
				if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
					t.Errorf("%s: got %v, want target unattainable", name, err)
				}
				continue
			}
			if !pidCheckZPoleResult(t, name, num, den, dt, r, err) {
				t.Errorf("%s: no design: %v", name, err)
				continue
			}
			k := r.Controller
			loop := func(w float64) complex128 {
				z := cmplx.Exp(complex(0, w*dt))
				g := complex(num[0], 0) / ((z - 1) * (z - 1) * (z - r3))
				integ := complex(dt, 0) / (z - 1)
				deriv := 1 / (complex(k.Tf, 0) + complex(dt, 0)/(z-1))
				return g * (complex(k.Kp, 0) + complex(k.Ki, 0)*integ + complex(k.Kd, 0)*deriv)
			}
			l := loop(wc)
			pm := 180 + cmplx.Phase(l)*180/math.Pi
			if r.Evidence.AchievedCrossover != wc || math.Abs(cmplx.Abs(l)-1) > 1e-3 || math.Abs(pm-r.Evidence.AchievedPhaseMargin) > .5 || pm < 56 {
				t.Errorf("%s: crossover %g, |L| = %.6g, margin %.4g, reported %.4g", name, r.Evidence.AchievedCrossover, cmplx.Abs(l), pm, r.Evidence.AchievedPhaseMargin)
			}
			if wc != 0.0035 {
				continue
			}
			peak := 0.
			for w := wc / 100; w < math.Pi/dt; w *= 1.001 {
				peak = math.Max(peak, 1/cmplx.Abs(1+loop(w)))
			}
			if peak > pidTuningMaxSensitivity*1.01 {
				t.Errorf("%s: sampled |S| peak %.4g", name, peak)
			}
		}
	}
}
