package controlsys

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"
)

// pidFRDTopCutSweepCase is the top-cut family of
// TestTunePIDFRDCertificationBar: a strict sweep plant whose band stops
// 3-30 times lower.
func pidFRDTopCutSweepCase(seed uint64) pidFRDStrictCase {
	c := pidFRDStrictSweepCase(rand.New(rand.NewPCG(9108, seed)))
	c.hi = math.Max(c.hi/math.Pow(10, 0.5+rand.New(rand.NewPCG(9109, seed)).Float64()), 30*c.lo)
	return c
}

// QHMGQL: on top-cut seed 402 a pair at 75.7 rad/s, ζ 0.07, 14 times the
// data's top, bends the top half decade by β 0.0052, inside the TopFlat
// bound, so the Markov end anchored and the ideal PID with
// |L(∞)| = 0.83 came back certified at a 63° margin while the plant's is
// 13.6° and |S| 4.41. The design is unchanged but uncertified. Seeds 490
// (β 0.0020, |L(∞)| 0.60) and 448 (β 1.4e-5, |L(∞)| 1.47) stay certified
// and accurate. Oracles: companion roots, dense margin and |S| sweeps of
// the exact plant.
func TestTunePIDFRDIdealDerivativeEndBeyondTopZone(t *testing.T) {
	for _, tc := range []struct {
		seed      uint64
		k         PID2
		certified bool
	}{
		{402, PID2{Kp: -0.2984063906390367, Ki: -0.7139901589236518, Kd: -22.826706625891763}, false},
		{490, PID2{Kp: 0.17407353397826925, Kd: -0.4649628434304586}, true},
		{448, PID2{Kp: -0.20546367619978415, Ki: -0.08024351905981994, Kd: -4.515687947761089}, true},
	} {
		c := pidFRDTopCutSweepCase(tc.seed)
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
		if p, _ := pidTuningFRDStrictPlant(frd, nil); p.markov == 0 || !p.markovAnchored {
			t.Errorf("%d: markov %g anchored %v", tc.seed, p.markov, p.markovAnchored)
		}
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			t.Fatalf("%d: %v", tc.seed, err)
		}
		k := *r.Controller
		near := func(a, b float64) bool { return math.Abs(a-b) <= 1e-6*math.Max(1, math.Abs(b)) }
		if !near(k.Kp, tc.k.Kp) || !near(k.Ki, tc.k.Ki) || !near(k.Kd, tc.k.Kd) || k.Tf != 0 {
			t.Errorf("%d: design %+v changed from %+v", tc.seed, k, tc.k)
		}
		if !pidFRDStrictStable(c.num, c.den, k) {
			t.Errorf("%d: %+v unstable", tc.seed, k)
		}
		pm, least := r.Evidence.AchievedPhaseMargin, pidFRDDenseMarginAny(c.num, c.den, c.dt, k, c.lo*1e-4, c.hi*1e5)
		peak := pidFRDStrictPeak(c.num, c.den, c.dt, k, c.lo)
		if got := !pidHasWarning(r, "not certified"); got != tc.certified {
			t.Errorf("%d: certified %v, want %v; warnings %q", tc.seed, got, tc.certified, r.Evidence.Warnings)
		}
		if tc.certified {
			if pm > least+1.5 || !(peak <= 4*1.05) {
				t.Errorf("%d: certified margin %g, dense %g, dense |S| %g", tc.seed, pm, least, peak)
			}
			continue
		}
		if !pidHasWarning(r, "loop's end above the band is established") {
			t.Errorf("%d: warnings %q", tc.seed, r.Evidence.Warnings)
		}
		if !(least < 20) || !(peak > 4.2) || !(pm > least+1.5) {
			t.Errorf("%d: dense margin %g, |S| %g, reported %g; want the hidden 13.6°, 4.41 the warning covers", tc.seed, least, peak, pm)
		}
	}
}
