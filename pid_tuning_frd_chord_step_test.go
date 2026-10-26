package controlsys

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"
)

// XXVU4O: resonance sweep cases PCG(2525, 84) and (2525, 237), PIDFs
// whose crossing lies in a step no fit follows beside a ζ 0.01 mode,
// where the cubic's gate fails as the neighbouring step turns, were
// certified with the chord's 48.0° against the plant's 44.8° and 46.4°.
// The cubic and quadratics through the neighbouring samples read 40.9°
// and 45.8° there, so the step is unresolved: each design is unchanged
// but uncertified, nor are its margins certified. On 219 and 38 the
// polynomial readings stay within 1.5° below the reported margin, which
// is within 1.5° of the plant's, and the designs stay certified.
// Oracles: companion roots and dense margin sweeps of the exact plant.
func TestTunePIDFRDChordStepCertification(t *testing.T) {
	for _, c := range []struct {
		seed      uint64
		k         PID2
		certified bool
	}{
		{84, PID2{Kp: 0.6108507400559753, Ki: 7.71140935680998, Kd: 0.37148212359619165, Tf: 0.011249053674357416}, false},
		{237, PID2{Kp: 0.2756510432802599, Ki: 0.24877371414799967, Kd: 0.2568817871989368, Tf: 0.0007662945678520511}, false},
		{219, PID2{Kp: -0.0005223280762033192, Ki: 0.00015101532366830803}, true},
		{38, PID2{Kp: -0.00010248146531481524, Kd: -0.0010378000601871783}, true},
	} {
		p := pidFRDResonanceSweepCase(rand.New(rand.NewPCG(2525, c.seed)))
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, p.num, p.den, 0, p.lo, p.hi, p.n), p.num, p.den)
		r, err := TunePIDFRD(context.Background(), frd, p.family, p.o)
		if err != nil {
			t.Errorf("case %d: %v", c.seed, err)
			continue
		}
		k := *r.Controller
		for _, g := range [][2]float64{{k.Kp, c.k.Kp}, {k.Ki, c.k.Ki}, {k.Kd, c.k.Kd}, {k.Tf, c.k.Tf}} {
			if math.Abs(g[0]-g[1]) > 1e-6*math.Abs(g[1]) {
				t.Errorf("case %d: %+v, base gave %+v", c.seed, k, c.k)
				break
			}
		}
		if !pidFRDStrictStable(p.num, p.den, k) {
			t.Errorf("case %d: %+v unstable", c.seed, k)
		}
		pm, least := r.Evidence.AchievedPhaseMargin, pidFRDDenseMargin(p.num, p.den, k, p.lo/1e3, p.hi*1e3)
		certified := !pidHasWarning(r, "not certified")
		switch {
		case certified != c.certified:
			t.Errorf("case %d: certified %v, achieved margin %g, dense %g, warnings %q", c.seed, certified, pm, least, r.Evidence.Warnings)
		case !c.certified && (!pidHasWarning(r, "nor its margins") || !(pm > least+1.5)):
			t.Errorf("case %d: achieved margin %g, dense %g, warnings %q; want the chord's over-read warned", c.seed, pm, least, r.Evidence.Warnings)
		case c.certified && !(math.Abs(pm-least) <= 1.5):
			t.Errorf("case %d: achieved margin %g, dense %g", c.seed, pm, least)
		}
	}
}
