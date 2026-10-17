package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// pidDenseSensitivity sweeps |1/(1+C·G)| over [lo, hi] on a 1e-5 log grid.
func pidDenseSensitivity(c PID2, g func(float64) complex128, lo, hi float64) (peak, at float64) {
	for w := lo; w <= hi; w *= 1 + 1e-5 {
		s := complex(0, w)
		k := complex(c.Kp, 0) + complex(c.Ki, 0)/s + complex(c.Kd, 0)*s/(1+complex(c.Tf, 0)*s)
		if v := 1 / cmplx.Abs(1+k*g(w)); v > peak {
			peak, at = v, w
		}
	}
	return peak, at
}

// e^{-s}/(s²+0.05s+1) under PD: the ζ=0.025 resonance puts a narrow |S| peak
// at ω≈1.019 that a 1% grid read as 3.89 against a dense 4.11–4.20.
func TestPidTuningSensitivityPeakResonance(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, .05, 1})
	if err := plant.SetInputDelay([]float64{1}); err != nil {
		t.Fatal(err)
	}
	eval, err := newSISOEval(plant)
	if err != nil {
		t.Fatal(err)
	}
	g := func(w float64) complex128 {
		s := complex(0, w)
		return cmplx.Exp(-s) / (s*s + .05*s + 1)
	}
	modes := pidTuningModes(plant)
	if len(modes) != 1 || cmplx.Abs(modes[0]-complex(-.025, math.Sqrt(1-.025*.025))) > 1e-12 {
		t.Fatalf("modes %v", modes)
	}
	wc := .9947247662799361
	p := pidTuningPlant{low: wc / 100, high: wc * 100, at: eval.at, tau: 1, modes: modes}
	for _, c := range []PID2{
		{Kp: .049499, Kd: .0116488, B: 1, C: 1},
		{Kp: .049378529, Kd: .012154619, B: 1, C: 1},
		{Kp: .049101765, Kd: .011696142, Tf: .033673440, B: 1, C: 1},
		{Kp: .04920622, Kd: .01284144, B: 1, C: 1},
	} {
		want, at := pidDenseSensitivity(c, g, p.low, p.high)
		if got := pidTuningSensitivityPeak(p, c); math.Abs(got-want) > 1e-3 {
			t.Errorf("%+v: peak %g, dense %g at ω=%g", c, got, want, at)
		}
	}
}

// Discrete modes map by s = ln(z)/dt: ZOH 1/(s²+0.05s+1) at dt=0.1 with
// 10 samples of delay keeps the narrow resonance peak.
func TestPidTuningSensitivityPeakDiscreteResonance(t *testing.T) {
	cont := makePlant(t, []float64{1}, []float64{1, .05, 1})
	d, err := cont.C2D(.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if err := d.SetInputDelay([]float64{10}); err != nil {
		t.Fatal(err)
	}
	eval, err := newSISOEval(d)
	if err != nil {
		t.Fatal(err)
	}
	model, err := d.AbsorbDelay(AbsorbAll)
	if err != nil {
		t.Fatal(err)
	}
	modes := pidTuningModes(model)
	if !slices.ContainsFunc(modes, func(m complex128) bool { return cmplx.Abs(m-complex(-.025, math.Sqrt(1-.025*.025))) < 1e-9 }) {
		t.Fatalf("modes %v miss the resonance", modes)
	}
	p := pidTuningPlant{dt: .1, low: .01, high: .99 * math.Pi / .1, at: eval.at, tau: 1, modes: modes}
	for _, kd := range []float64{.0116, .0122, .0128} {
		c := PID2{Kp: .0494, Kd: kd, Tf: .06, Dt: .1, B: 1, C: 1}
		want := 0.
		for w := p.low; w <= p.high; w *= 1 + 1e-5 {
			want = math.Max(want, 1/cmplx.Abs(1+eval.at(w)*pidTuningFeedback(c, w)))
		}
		if got := pidTuningSensitivityPeak(p, c); math.Abs(got-want) > 1e-3 {
			t.Errorf("Kd=%g: peak %g, dense %g", kd, got, want)
		}
	}
}

// A near-neutral P loop on e^{-2.1s}/(s+10) peaks between delay-grid
// samples; refinement must recover the dense peak.
func TestPidTuningSensitivityPeakDelayRipple(t *testing.T) {
	plant := delayedFirstOrder(t, 2.1)
	eval, err := newSISOEval(plant)
	if err != nil {
		t.Fatal(err)
	}
	g := func(w float64) complex128 {
		s := complex(0, w)
		return cmplx.Exp(-2.1*s) / (s + 10)
	}
	p := pidTuningPlant{low: .005, high: 50, at: eval.at, tau: 2.1}
	for kp := 5.; kp < 9.6; kp += .25 {
		c := PID2{Kp: kp, B: 1, C: 1}
		want, at := pidDenseSensitivity(c, g, p.low, p.high)
		if got := pidTuningSensitivityPeak(p, c); math.Abs(got-want) > 1e-3*want {
			t.Errorf("Kp=%g: peak %g, dense %g at ω=%g", kp, got, want, at)
		}
	}
}

// The default PD and PDF designs on e^{-s}/(s²+0.05s+1) must keep the dense
// |S| <= 4; the resonance peak used to slip between grid samples.
func TestTunePIDResonantDelayBoundsSensitivity(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, .05, 1})
	if err := plant.SetInputDelay([]float64{1}); err != nil {
		t.Fatal(err)
	}
	g := func(w float64) complex128 {
		s := complex(0, w)
		return cmplx.Exp(-s) / (s*s + .05*s + 1)
	}
	for _, family := range []PidtuneType{PidtunePD, PidtunePDF} {
		r, err := TunePID(context.Background(), plant, family, PIDTuningOptions{})
		if err != nil {
			t.Errorf("%s: %v", family, err)
			continue
		}
		c := r.Controller
		if c.Kp <= 0 || c.Kd <= 0 {
			t.Errorf("%s: %+v gains must share the DC sign", family, c)
		}
		if peak, at := pidDenseSensitivity(*c, g, 1e-3, 1e3); peak > pidTuningMaxSensitivity+1e-3 {
			t.Errorf("%s: %+v dense |S|=%g at ω=%g", family, c, peak, at)
		}
		if stable, _ := pidDelayLoopCheckPlant(t, c, 1, func(s complex128) complex128 { return 1 / (s*s + .05*s + 1) }, 2); !stable {
			t.Errorf("%s: %+v independently unstable", family, c)
		}
	}
}
