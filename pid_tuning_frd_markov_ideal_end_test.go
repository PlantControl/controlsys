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

// pidFRDHiddenPairSweepCase is a plant K·ω_n²·p/((s+p)(s²+2ζω_n·s+ω_n²)),
// a third holding an extra real zero and a third an extra real pole in the
// band, sampled at 40-200 points on a band whose top stops 10-63 times
// below a pair of ζ 0.01-0.32, with a PID, PD or PI family.
func pidFRDHiddenPairSweepCase(seed uint64) pidFRDStrictCase {
	r := rand.New(rand.NewPCG(7702, seed))
	p := math.Pow(10, -2+2*r.Float64())
	wn := math.Pow(10, -1+3*r.Float64())
	z := math.Pow(10, -2+1.5*r.Float64())
	hi := wn / math.Pow(10, 1+0.8*r.Float64())
	lo := math.Min(p, hi) * math.Pow(10, -1.5-r.Float64())
	mul := func(a, b []float64) []float64 {
		o := make([]float64, len(a)+len(b)-1)
		for i, x := range a {
			for j, y := range b {
				o[i+j] += x * y
			}
		}
		return o
	}
	num, den := []float64{wn * wn * p}, mul([]float64{1, p}, []float64{1, 2 * z * wn, wn * wn})
	a := lo * math.Pow(hi/lo, r.Float64())
	switch r.IntN(3) {
	case 1:
		num = mul(num, []float64{1 / a, 1})
	case 2:
		den = mul(den, []float64{1 / a, 1})
	}
	k := math.Pow(10, -1+2*r.Float64())
	if r.IntN(2) == 0 {
		k = -k
	}
	for i := range num {
		num[i] *= k
	}
	fam := []PidtuneType{PidtunePID, PidtunePD, PidtunePI}[r.IntN(3)]
	return pidFRDStrictCase{num: num, den: den, lo: lo, hi: hi, n: 40 + r.IntN(161), family: fam}
}

// QHMGQL: an ideal derivative (Tf = 0) on a plant showing lim sG = m at
// the data's top ends L at Kd·m, but a pole pair of damping ζ anywhere
// above the band lifts |L| there by about 1/(2ζ), through 1 where no
// sample reads a crossing, and the data fix neither ζ nor how far up the
// pair sits. Top-cut seed 402 (ζ 0.07, 14 times the top, |L(∞)| 0.83) was
// certified at a 63° margin while the plant's is 13.6° and |S| 4.41; the
// review's hidden-pair plants were certified through either end of the
// earlier |L(∞)| ≤ ½ or flat-top gate: ζ 0.010 59 times up at |L(∞)| 1.00
// (63° against 1.75°), ζ 0.22 19.5 times up at 1.00 (63° against 35.8°)
// and ζ 0.012 11.7 times up at 0.36 (63° against 4.48°, |S| 12.9). All
// designs are unchanged and uncertified, including top-cut 490 and 448,
// accurate as their pairs damp the lift, a limit of band-limited data.
// Oracles: companion roots, dense margin and |S| sweeps of the exact
// plant.
func TestTunePIDFRDIdealDerivativeEndBeyondTopZone(t *testing.T) {
	hidden := func(num, den []float64, lo, hi float64, n int) pidFRDStrictCase {
		return pidFRDStrictCase{num: num, den: den, lo: lo, hi: hi, n: n, family: PidtunePID}
	}
	for _, tc := range []struct {
		name   string
		c      pidFRDStrictCase
		k      PID2
		hidden bool
	}{
		{"top-cut 402", pidFRDTopCutSweepCase(402), PID2{Kp: -0.2984063906390367, Ki: -0.7139901589236518, Kd: -22.826706625891763}, true},
		{"top-cut 490", pidFRDTopCutSweepCase(490), PID2{Kp: 0.17407353397826925, Kd: -0.4649628434304586}, false},
		{"top-cut 448", pidFRDTopCutSweepCase(448), PID2{Kp: -0.20546367619978415, Ki: -0.08024351905981994, Kd: -4.515687947761089}, false},
		{"pair 59x up", hidden([]float64{-550.1119417785624}, []float64{1, 1.0153244784814242, 2102.8635973724276, 121.97224753528222}, 4.6856718009253943e-4, 0.7812983258361484, 68), PID2{Kp: 0.035442561417648906, Ki: -0.005821987835373217, Kd: -3.8373406340883274}, true},
		{"pair 19.5x up", hidden([]float64{-23.090911445803705}, []float64{1, 16.260224805862975, 1388.571001908289, 176.4504292976721}, 1.157992832408082e-3, 1.909333811235378, 161), PID2{Kp: 0.9490632378673998, Ki: -0.5134888009205247, Kd: -60.17213993630066}, true},
		{"pair 11.7x up", hidden([]float64{0.7236342486824499}, []float64{1, 0.18074167240644537, 47.468058118134735, 0.6972259671414005}, 3.0536650423351774e-4, 0.5912649637147409, 51), PID2{Kp: 0.3479479036742572, Ki: 0.021170115670324303, Kd: 23.584544743868392}, true},
	} {
		c := tc.c
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, c.dt, c.lo, c.hi, c.n), c.num, c.den)
		if p, _ := pidTuningFRDStrictPlant(frd, nil); p.markov == 0 {
			t.Errorf("%s: no Markov end read", tc.name)
		}
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		k := *r.Controller
		near := func(a, b float64) bool { return math.Abs(a-b) <= 1e-6*math.Max(1, math.Abs(b)) }
		if !near(k.Kp, tc.k.Kp) || !near(k.Ki, tc.k.Ki) || !near(k.Kd, tc.k.Kd) || k.Tf != 0 {
			t.Errorf("%s: design %+v changed from %+v", tc.name, k, tc.k)
		}
		if !pidFRDStrictStable(c.num, c.den, k) {
			t.Errorf("%s: %+v unstable", tc.name, k)
		}
		if !pidHasWarning(r, "loop's end above the band is established") {
			t.Errorf("%s: certified; warnings %q", tc.name, r.Evidence.Warnings)
		}
		pm, least := r.Evidence.AchievedPhaseMargin, pidFRDDenseMarginAny(c.num, c.den, c.dt, k, c.lo*1e-4, c.hi*1e5)
		peak := pidFRDStrictPeak(c.num, c.den, c.dt, k, c.lo)
		if hit := pm > least+1.5 || !(peak <= 4*1.05); hit != tc.hidden {
			t.Errorf("%s: reported margin %g, dense %g, dense |S| %g; want hidden crossing %v", tc.name, pm, least, peak, tc.hidden)
		}
	}
}
