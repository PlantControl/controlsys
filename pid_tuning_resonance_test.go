package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// The crossover grid kept only the first point near a mode: each added
// point became the grid's end, so the next failed the in-band test.
func TestPIDTuningCrossGridAddsEveryModePoint(t *testing.T) {
	m := complex(-.01857, 3.2572)
	p := pidTuningPlant{modes: []complex128{m}, hasCrossFloor: true, crossLow: .0325, crossHigh: 325}
	grid := pidTuningCrossGrid(p, []float64{.0325, 1, 3.2535, 325})
	for k := -8.; k <= 8; k++ {
		w := imag(m) + k*math.Abs(real(m))/2
		found := false
		for _, g := range grid {
			found = found || math.Abs(g-w) < 1e-12
		}
		if !found {
			t.Errorf("grid %v misses %g", grid, w)
		}
	}
}

// Outside the floor the margin check scanned the 241-point grid only, so a
// crossover pair at a lightly damped resonance between samples was stepped
// over. Base's design for the delayed plant crosses 0 dB again just above
// the 1.021 rad/s mode with a 36.25° margin (dense sweep); the check must
// report it.
func TestPIDTuningCrossingsFindsResonancePair(t *testing.T) {
	num := pidPoly(-.0503, []float64{1, -9.178})
	den := pidPoly(1, []float64{1, .06089, 1.043}, []float64{1, .0589}, []float64{1, 6.552})
	plant := pidPlantDelayed(t, num, den, .0301)
	eval, err := newSISOEval(plant)
	if err != nil {
		t.Fatal(err)
	}
	wc := 1.0136734096365496
	p := pidTuningPlant{low: wc / 100, high: wc * 100, at: eval.at, modes: pidTuningModes(plant), crossLow: wc / 100, crossHigh: wc * 100}
	omega := make([]float64, 241)
	for k := range omega {
		omega[k] = p.low * math.Pow(1e4, float64(k)/240)
	}
	omega[120] = wc
	c := PID2{Kp: .4687458702389815, Kd: .78355447784191, B: 1, C: 1}
	crossings, margins := pidTuningCrossings(p, c, pidTuningCrossGrid(p, omega), wc)
	if len(crossings) != 2 || math.Abs(margins[1]-36.25) > .01 {
		t.Fatalf("crossings %v margins %v, want a second at 36.25°", crossings, margins)
	}
}

// End to end, every crossing in a dense sweep of the evidence band meets
// PM − 3° and is reported. The first plant's ζ = 0.0057 mode at 3.257 rad/s
// hid a 37° crossing at the moved crossover. The second has sound PD designs
// only in a 1% pocket at its resonance, off every crossover the search
// tries, so it may fail; it must not return the 36° design. The third's
// second crossover lies between wc and the next grid sample, where
// |L(wc)| = 1 hides the sign change. The fourth's seeds all cross 0 dB at
// a resonance with negative margin; the search must still reach the sound
// design from the least violating one.
func TestTunePIDMarginCheckFindsResonanceCrossings(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		tau      float64
		family   PidtuneType
		mayFail  bool
	}{
		{"0.05372(s+0.7974)/((s+0.779)(s+0.101)(s²+0.03714s+10.61))", pidPoly(.05372, []float64{1, .7974}), pidPoly(1, []float64{1, .779}, []float64{1, .101}, []float64{1, .03714, 10.61}), 0, PidtunePDF, false},
		{"-0.0503(s-9.178)/((s²+0.06089s+1.043)(s+0.0589)(s+6.552))e^-0.0301s", pidPoly(-.0503, []float64{1, -9.178}), pidPoly(1, []float64{1, .06089, 1.043}, []float64{1, .0589}, []float64{1, 6.552}), .0301, PidtunePD, true},
		{"sweep 42: partner 0.12% above wc", []float64{9.941801896667057, 75.47424183387096}, []float64{1, .6967863957610382, .6499124770289462, .4126294043259971, .10128779613762952, .05905400275971327}, 0, PidtunePD, false},
		{"sweep 109: every seed crosses at the 6.17 rad/s mode", []float64{-8.347729847190841}, []float64{1, .7307546373686498, 61.81362857880511, 31.504618973121026, 902.1672488522788, 215.17111864467765}, 0, PidtunePID, false},
	}
	for _, c := range cases {
		r, err := TunePID(context.Background(), pidPlantDelayed(t, c.num, c.den, c.tau), c.family, PIDTuningOptions{})
		if c.mayFail && errors.Is(err, ErrPIDTuningTargetUnattainable) {
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		k := r.Controller
		L := func(w float64) complex128 {
			s := complex(0, w)
			g := pidPolyAt(c.num, s) / pidPolyAt(c.den, s) * cmplx.Exp(-s*complex(c.tau, 0))
			return g * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
		}
		band := r.Evidence.FrequencyBand
		margins := pidLoopMargins(L, band[0], band[1])
		for _, pm := range margins {
			if pm < 57 {
				t.Errorf("%s: %+v independent margin %g below 57° (dense %v)", c.name, k, pm, margins)
			}
		}
		if len(margins) != len(r.Evidence.GainCrossovers) {
			t.Errorf("%s: evidence crossovers %v, dense margins %v", c.name, r.Evidence.GainCrossovers, margins)
		}
	}
}

// Searching under the refined crossing check changed the search path: an
// intermediate incumbent with a real −11° crossing pair at the 0.51 rad/s
// mode was rejected, and the search never reached base's design, which
// meets the floor at its only crossing. The search now runs on the grid
// scan and the refined check vets its design. Oracle: dense margins over
// the evidence band.
func TestTunePIDRefinedCheckKeepsSoundDesign(t *testing.T) {
	num, den := []float64{.112, 1.487, -6.146}, []float64{1, 1.630, .398, .424, .0311}
	r, err := TunePID(context.Background(), makePlant(t, num, den), PidtunePID, PIDTuningOptions{CrossoverFrequency: .053647})
	if err != nil {
		t.Fatal(err)
	}
	k := r.Controller
	L := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s) * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s)
	}
	band := r.Evidence.FrequencyBand
	margins := pidLoopMargins(L, band[0], band[1])
	for _, pm := range margins {
		if pm > 180 {
			pm -= 360
		}
		if pm < 57 {
			t.Errorf("%+v: dense margin %g below 57° (%v)", *k, pm, margins)
		}
	}
	if len(margins) != len(r.Evidence.GainCrossovers) {
		t.Errorf("evidence crossovers %v, dense margins %v", r.Evidence.GainCrossovers, margins)
	}
}
