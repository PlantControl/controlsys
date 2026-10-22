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
// resonance (|G| = 20), whose phase drop leaves no same-sign PI/PID/PIDF
// there, so the call failed. MATLAB pidtune returns a stable lower-bandwidth
// design; the default crossover now backs off. In the second plant the
// back-off reaches 2⁻⁸·wc0,
// whose own band [wc/100, 100wc] ends below the ζ=0.0035 resonance at 1.3:
// |S| there was unchecked and reached 7.7. The third, with an RHP zero, has
// PIDF designs only from 2⁻⁹·wc0 down; the scan down from wc0 spent the
// budget first and failed, and the deep design is now kept. The last two,
// TestTunePIDBackOffProbesShallowFirst's, back off one or two depths. The
// two after, from a seeded sweep, have designs only at depths 13-18 and
// 12-17 (PID at 12 alone) while both deepest fail, so the deep gate lost
// them; the mid-depth probes find them.
// Oracles: dense
// margins, dense |S| over the default and backed-off bands, and the winding of 1+L(jω), +90°
// for a stable loop with one integrator.
func TestTunePIDBacksOffResonantDelayedDefault(t *testing.T) {
	skipSlowUnderRace(t)
	for _, c := range []struct {
		num, den []float64
		tau      float64
		fams     []PidtuneType
	}{
		{[]float64{1}, []float64{1, .05, 1}, 1, []PidtuneType{PidtunePI, PidtunePID, PidtunePIDF}},
		{[]float64{2.229880306748357}, []float64{1, .009089419529549078, 1.6808028078517465}, .0415713306402339, []PidtuneType{PidtunePI}},
		{[]float64{3.1016, -.99088}, []float64{1, 1.02767, .71308, .45281, .113657}, .03776, []PidtuneType{PidtunePIDF}},
		{[]float64{4469.208567469051, 3872.5108510086206}, []float64{1, .13029988431555797, 635.262873587796}, .0008295413703164025, []PidtuneType{PidtunePI}},
		{[]float64{50939.635002137285}, []float64{1, 74.35099627689756, 84.57397936219536, 6173.81323963725}, .01232428206353334, []PidtuneType{PidtunePID}},
		{[]float64{-829.9233529498596, 385.9534649577797}, []float64{1, 1.3979863130162498, 88.69018768710332, 59.61767784915426}, .06820806116630618, []PidtuneType{PidtunePID}},
		{[]float64{.7432501908577885, .11503292699378517}, []float64{1, .04811033390987361, 17.226160634569503, .6708668130621463}, .45827350160174746, []PidtuneType{PidtunePI, PidtunePID}},
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
			dc := math.Copysign(1, c.num[len(c.num)-1]/c.den[len(c.den)-1])
			if dc*k.Ki <= 0 || dc*k.Kp < 0 || dc*k.Kd < 0 {
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
			if wind := pidWindingDeg(L, 1e-6, math.Max(1e4, wc0*1e4)); math.Abs(wind-90) > 1 {
				t.Errorf("%v %s: %+v arg(1+L) changes by %g°, want 90°", c.den, fam, *k, wind)
			}
		}
	}
}

// The back-off bisected the half-octave depth, taking success as monotone.
// On e^{-s}/(s²+0.05s+1) a PID crossing at wc0/2 (Ki ≈ 0.55, Kd ≈ 0.6)
// meets 60°±3° with failures from wc0/2^1.5 to wc0/2^4.5, so bisection
// returned wc0/32; and PIDF, seeded at Tf = 0.1/wc, found no design there.
// The scan down from wc0 keeps the shallowest design, and PIDF a minimal-Tf
// seed. Oracle: |L(jω)| = 1 at the reported crossover, from the plant's
// transfer function; TestTunePIDBacksOffResonantDelayedDefault vets the
// margins, |S| and stability of the same designs.
func TestTunePIDBackOffKeepsShallowDesign(t *testing.T) {
	plant := pidPlantDelayed(t, []float64{1}, []float64{1, .05, 1}, 1)
	wc0, err := findCrossoverFreq(plant, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	wc := map[PidtuneType]float64{}
	for _, fam := range []PidtuneType{PidtunePID, PidtunePIDF} {
		r, err := TunePID(context.Background(), plant, fam, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("%s: %v", fam, err)
		}
		k, w := r.Controller, r.Evidence.AchievedCrossover
		s := complex(0, w)
		L := cmplx.Exp(-s) / (s*s + .05*s + 1) * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
		if math.Abs(cmplx.Abs(L)-1) > 1e-6 {
			t.Errorf("%s: |L(j%g)| = %g", fam, w, cmplx.Abs(L))
		}
		if w <= wc0/4 {
			t.Errorf("%s: crossover %g <= wc0/4 = %g", fam, w, wc0/4)
		}
		wc[fam] = w
	}
	if wc[PidtunePIDF] < wc[PidtunePID]/2 {
		t.Errorf("PIDF crossover %g below half PID's %g", wc[PidtunePIDF], wc[PidtunePID])
	}
}
