package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"slices"
	"strings"
	"testing"
)

// e^{-s}/(s²+0.05s+1): the delay-capped default wc = π/3 sits past the ζ=0.025
// resonance (|G| = 20), where no PI/PID/PIDF meets |S| <= 4, so the call
// failed. MATLAB pidtune returns a stable low-bandwidth design; the default
// crossover now backs off. In the second plant the back-off reaches 2⁻⁸·wc0,
// whose own band [wc/100, 100wc] ends below the ζ=0.0035 resonance at 1.3:
// |S| there was unchecked and reached 7.7. Oracles: dense margins, dense |S|
// over the default and backed-off bands, and the winding of 1+L(jω), +90°
// for a stable loop with one integrator.
func TestTunePIDBacksOffResonantDelayedDefault(t *testing.T) {
	for _, c := range []struct {
		num, den []float64
		tau      float64
		fams     []PidtuneType
	}{
		{[]float64{1}, []float64{1, .05, 1}, 1, []PidtuneType{PidtunePI, PidtunePID, PidtunePIDF}},
		{[]float64{2.229880306748357}, []float64{1, .009089419529549078, 1.6808028078517465}, .0415713306402339, []PidtuneType{PidtunePI}},
	} {
		plant := makePlant(t, c.num, c.den)
		if err := plant.SetInputDelay([]float64{c.tau}); err != nil {
			t.Fatal(err)
		}
		wc0, err := findCrossoverFreq(plant, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		g := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s) * cmplx.Exp(-s*complex(c.tau, 0))
		}
		for _, fam := range c.fams {
			r, err := TunePID(context.Background(), plant, fam, PIDTuningOptions{})
			if err != nil {
				t.Errorf("%v %s: %v", c.den, fam, err)
				continue
			}
			k := r.Controller
			if k.Ki <= 0 || k.Kp < 0 || k.Kd < 0 {
				t.Errorf("%v %s: %+v gains not of the plant's DC sign", c.den, fam, *k)
			}
			wc := r.Evidence.AchievedCrossover
			if wc >= wc0 || wc < wc0/1024 {
				t.Errorf("%v %s: crossover %g not backed off from %g", c.den, fam, wc, wc0)
			}
			if !slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "lowered to") }) {
				t.Errorf("%v %s: warnings %q lack the back-off", c.den, fam, r.Evidence.Warnings)
			}
			L := func(w float64) complex128 {
				s := complex(0, w)
				return g(w) * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
			}
			margins := pidLoopMargins(L, wc/1e3, wc0*1e3)
			if len(margins) == 0 {
				t.Errorf("%v %s: no crossover", c.den, fam)
			}
			for _, pm := range margins {
				if pm < 57-1e-6 {
					t.Errorf("%v %s: %+v margin %g below the 60° floor", c.den, fam, *k, pm)
				}
			}
			if peak, at := pidDenseSensitivity(*k, g, wc/1e3, wc0*100); peak > 4+1e-3 {
				t.Errorf("%v %s: %+v |S| %g at ω=%g", c.den, fam, *k, peak, at)
			}
			if wind := pidWindingDeg(L, 1e-6, 1e4); math.Abs(wind-90) > 1 {
				t.Errorf("%v %s: %+v arg(1+L) changes by %g°, want 90°", c.den, fam, *k, wind)
			}
		}
	}
}
