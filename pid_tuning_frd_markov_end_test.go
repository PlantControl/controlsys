package controlsys

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"
)

// NHV6F5: on a plant whose zero and pole at 34.1 and 37.6 rad/s, 2.8 and
// 3.1 times the data's top, nearly cancel in |G|, the strict count took
// the top sample's lim sG, −0.0407, for the Markov end, certifying an ideal
// PD whose true L(∞) = Kd·(−0.0444) leaves |S| at 4.34. The design is
// unchanged but uncertified, as the strict count certifies no ideal
// derivative's end above the band (QHMGQL). Oracles: companion roots and
// a dense |S| sweep of the exact plant.
func TestTunePIDFRDMarkovEndAboveBand(t *testing.T) {
	c := pidFRDStrictSweepCase(rand.New(rand.NewPCG(9103, 189)))
	frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
	if p, _ := pidTuningFRDStrictPlant(frd, nil); p.markov == 0 {
		t.Errorf("no Markov end read")
	}
	r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	if math.Abs(k.Kp+1.7026602355238596) > 1e-6 || math.Abs(k.Kd-17.341292797023367) > 1e-5 || k.Ki != 0 || k.Tf != 0 {
		t.Errorf("design %+v changed", k)
	}
	if !pidFRDStrictStable(c.num, c.den, k) {
		t.Errorf("%+v unstable", k)
	}
	if s := pidFRDStrictPeak(c.num, c.den, 0, k, c.lo); !(s > 4.2) {
		t.Errorf("dense |S| %g, want the 4.34 the warning covers", s)
	}
	if !pidHasWarning(r, "loop's end above the band is established") {
		t.Errorf("certified; warnings %q", r.Evidence.Warnings)
	}
}
