package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"strings"
	"testing"
)

func delayedFirstOrder(t *testing.T, tau float64) *System {
	t.Helper()
	p := makePlant(t, []float64{1}, []float64{1, 10})
	if err := p.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	return p
}

// pidDelayLoopCheck independently checks C(s)·e^{-τs}/(s+10) for same-sign
// gains: 1+L(jω) starts at arg 0 (or −π/2 with an integrator) and, with
// |L| < 1 beyond the grid, each closed-loop RHP root lowers its final
// unwrapped arg by π, so stability is a final arg in (−π/2, π/2). It also
// returns the sampled sensitivity peak, bounded by 1/(1−|L|) beyond the grid.
func pidDelayLoopCheck(t *testing.T, c *PID2, tau float64) (stable bool, peak float64) {
	t.Helper()
	L := func(w float64) complex128 {
		s := complex(0, w)
		k := complex(c.Kp, 0) + complex(c.Kd, 0)*s/(1+complex(c.Tf, 0)*s)
		if c.Ki != 0 {
			k += complex(c.Ki, 0) / s
		}
		return k * cmplx.Exp(-s*complex(tau, 0)) / (s + 10)
	}
	const wEnd = 1e4
	hf := math.Abs(c.Kp)/wEnd + math.Abs(c.Ki)/(wEnd*wEnd)
	if c.Tf > 0 {
		hf += math.Abs(c.Kd) / c.Tf / wEnd
	} else {
		hf += math.Abs(c.Kd)
	}
	if hf >= 1 {
		return false, math.Inf(1)
	}
	w := 1e-5
	f := 1 + L(w)
	phase := cmplx.Phase(f)
	for w < wEnd {
		w = math.Min(w*1.002, w+.01/tau)
		next := 1 + L(w)
		step := cmplx.Phase(next / f)
		if math.Abs(step) > math.Pi/4 {
			t.Fatalf("grid too coarse at ω=%g", w)
		}
		phase += step
		f = next
		peak = math.Max(peak, 1/cmplx.Abs(f))
	}
	return math.Abs(phase) < math.Pi/2, math.Max(peak, 1/(1-hf))
}

func TestTunePIDDefaultCrossoverHonoursDelay(t *testing.T) {
	for _, tau := range []float64{2.1, 0.3} {
		r, err := TunePID(context.Background(), delayedFirstOrder(t, tau), PidtunePID, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("τ=%g: %v", tau, err)
		}
		c := r.Controller
		want := (math.Pi - math.Pi/3) / (2 * tau)
		if math.Abs(r.Evidence.RequestedCrossover-want) > 1e-12*want {
			t.Errorf("τ=%g: crossover %g, want delay cap %g", tau, r.Evidence.RequestedCrossover, want)
		}
		if c.Kp <= 0 || c.Ki <= 0 || c.Kd <= 0 || c.Tf != 0 {
			t.Errorf("τ=%g: gains %+v, want positive ideal PID", tau, c)
		}
		if r.Evidence.Stability != PIDStabilityNyquist {
			t.Errorf("τ=%g: stability evidence %q", tau, r.Evidence.Stability)
		}
		if pm := r.Evidence.AchievedPhaseMargin; pm < 57-1e-9 || pm > 63 {
			t.Errorf("τ=%g: phase margin %g", tau, pm)
		}
		stable, peak := pidDelayLoopCheck(t, c, tau)
		if !stable || peak > 3 {
			t.Errorf("τ=%g: independent check stable=%v |S|max=%g", tau, stable, peak)
		}
	}
}

func TestFindCrossoverFreqDelayCap(t *testing.T) {
	p := delayedFirstOrder(t, 2.1)
	if wc, err := findCrossoverFreq(p, 0, 45); err != nil || math.Abs(wc-.75*math.Pi/4.2) > 1e-12 {
		t.Errorf("continuous cap %g, %v; want %g", wc, err, .75*math.Pi/4.2)
	}
	if wc, err := findCrossoverFreq(p, 7, 45); err != nil || wc != 7 {
		t.Errorf("explicit crossover changed to %g, %v", wc, err)
	}
	free := makePlant(t, []float64{1}, []float64{1, 10})
	mag, err := findCrossoverFreq(free, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	if wc, err := findCrossoverFreq(delayedFirstOrder(t, .05), 0, 60); err != nil || wc != mag {
		t.Errorf("cap bound a small delay: %g, %v; want %g", wc, err, mag)
	}
	d, err := NewFromSlices(1, 1, 1, []float64{0.9}, []float64{1}, []float64{0.1}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.SetInputDelay([]float64{30}); err != nil {
		t.Fatal(err)
	}
	if wc, err := findCrossoverFreq(d, 0, 60); err != nil || math.Abs(wc-(2*math.Pi/3)/6) > 1e-12 {
		t.Errorf("discrete cap %g, %v; want %g", wc, err, (2*math.Pi/3)/6)
	}
}

func TestPidtuneDefaultCrossoverHonoursDelay(t *testing.T) {
	const tau = 2.1
	p := delayedFirstOrder(t, tau)
	pid, err := Pidtune(p, PidtunePI, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pid.Kp <= 0 || pid.Ki <= 0 {
		t.Fatalf("gains %+v", pid)
	}
	if stable, _ := pidDelayLoopCheck(t, &PID2{Kp: pid.Kp, Ki: pid.Ki}, tau); !stable {
		t.Fatalf("Pidtune default design %+v unstable", pid)
	}
}

func TestTunePIDRejectsSignInconsistentDerivative(t *testing.T) {
	for _, family := range []PidtuneType{PidtunePD, PidtunePDF} {
		_, err := TunePID(context.Background(), makePlant(t, []float64{1}, []float64{1, 10}), family, PIDTuningOptions{})
		var te *PIDTuningError
		if !errors.As(err, &te) || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Fatalf("%s: err %v, want PIDTuningError", family, err)
		}
		if !slices.ContainsFunc(te.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "sign") }) {
			t.Errorf("%s: warnings %q do not name the sign rule", family, te.Evidence.Warnings)
		}
	}
	_, err := TunePID(context.Background(), delayedFirstOrder(t, .05), PidtunePD, PIDTuningOptions{CrossoverFrequency: 10})
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Errorf("explicit crossover needing negative Kd: err %v", err)
	}
	if pidTuningSignConsistent(PID2{Kp: 10.25, Kd: -.974}, 1) {
		t.Error("accepted negative Kd on positive-DC-gain plant")
	}
	if !pidTuningSignConsistent(PID2{Kp: -1, Kd: 2}, 0) {
		t.Error("sign rule applied without a known DC sign")
	}
	r, err := TunePID(context.Background(), makePlant(t, []float64{-1}, []float64{1, 1}), PidtunePI, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	if r.Controller.Kp >= 0 || r.Controller.Ki >= 0 {
		t.Errorf("negative-DC plant gains %+v, want negative", r.Controller)
	}
}

func TestTunePIDDelayedPDAtCapIsWellDamped(t *testing.T) {
	const tau = .05
	wc := (math.Pi - math.Pi/3) / (2 * tau)
	r, err := TunePID(context.Background(), delayedFirstOrder(t, tau), PidtunePD, PIDTuningOptions{CrossoverFrequency: wc})
	if err != nil {
		t.Fatal(err)
	}
	c := r.Controller
	if c.Kp <= 0 || c.Kd <= 0 || r.Evidence.Stability != PIDStabilityNyquist {
		t.Fatalf("design %+v evidence %q", c, r.Evidence.Stability)
	}
	if stable, peak := pidDelayLoopCheck(t, c, tau); !stable || peak > 3 {
		t.Fatalf("independent check stable=%v |S|max=%g", stable, peak)
	}
}

// Kd·CB = ρ puts a neutral root chain at Re s = ln ρ/τ: stable for ρ < 1,
// but barely damped as ρ → 1, which sampled margins do not reveal.
func TestTunePIDDelayRejectsNearNeutralDerivative(t *testing.T) {
	const tau = .05
	p := delayedFirstOrder(t, tau)
	eval, err := newSISOEval(p)
	if err != nil {
		t.Fatal(err)
	}
	stable, ok, err := pidTuningDelayStability(p, eval, 10, true)
	if err != nil || !ok {
		t.Fatalf("out of scope: %v", err)
	}
	for _, tc := range []struct {
		c    PID2
		want bool
	}{
		{PID2{Kp: 1, Kd: .3}, true},
		{PID2{Kp: 1, Kd: .8}, false},
		{PID2{Kp: 10.25, Kd: -.974}, false},
		{PID2{Kp: 1, Kd: 1.2}, false},
	} {
		got, err := stable(&tc.c)
		if err != nil || got != tc.want {
			t.Errorf("%+v: stable=%v err=%v, want %v", tc.c, got, err, tc.want)
		}
		if ind, _ := pidDelayLoopCheck(t, &tc.c, tau); tc.want && !ind {
			t.Errorf("%+v accepted but independently unstable", tc.c)
		}
	}
}

// With D ≠ 0, |Kd·s·G(s)| grows without bound, so any ideal derivative makes
// the delayed loop unstable; the screen must not fall back to samples.
func TestTunePIDDelayFeedthroughRejectsIdealDerivative(t *testing.T) {
	p := makePlant(t, []float64{1, 1}, []float64{1, 2})
	if err := p.SetInputDelay([]float64{.2}); err != nil {
		t.Fatal(err)
	}
	eval, err := newSISOEval(p)
	if err != nil {
		t.Fatal(err)
	}
	stable, ok, err := pidTuningDelayStability(p, eval, 1, true)
	if err != nil || !ok {
		t.Fatalf("fell back to sampled screen: %v", err)
	}
	if got, err := stable(&PID2{Kp: .1, Kd: .01}); err != nil || got {
		t.Errorf("ideal derivative accepted: %v, %v", got, err)
	}
	if got, err := stable(&PID2{Kp: .1}); err != nil || !got {
		t.Errorf("small P rejected: %v, %v", got, err)
	}
	r, err := TunePID(context.Background(), p, PidtunePID, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
	var te *PIDTuningError
	if err == nil && (r.Controller.Kd != 0 || r.Evidence.Stability != PIDStabilityNyquist) {
		t.Fatalf("design %+v evidence %q", r.Controller, r.Evidence.Stability)
	} else if err != nil && !errors.As(err, &te) {
		t.Fatal(err)
	}
}

// Same-sign PD only adds lead, so a default crossover where the plant phase
// exceeds −180°+PM moves to where a positive-Kd design meets the margin.
func TestTunePIDDefaultPDMovesCrossover(t *testing.T) {
	moved := func(r *PIDTuningResult) bool {
		return slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "moved") })
	}
	for _, family := range []PidtuneType{PidtunePD, PidtunePDF} {
		r, err := TunePID(context.Background(), makePlant(t, []float64{1}, []float64{1, 3, 3, 1}), family, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("%s 1/(s+1)³: %v", family, err)
		}
		c := r.Controller
		if c.Kp <= 0 || c.Kd <= 0 || r.Evidence.Stability != PIDStabilityClosedLoopPoles || !moved(r) {
			t.Errorf("%s 1/(s+1)³: %+v evidence %q warnings %q", family, c, r.Evidence.Stability, r.Evidence.Warnings)
		}
		if pm := r.Evidence.AchievedPhaseMargin; pm < 57-1e-9 || pm > 63+1e-9 {
			t.Errorf("%s 1/(s+1)³: phase margin %g", family, pm)
		}
		if ph := -3 * math.Atan(r.Evidence.AchievedCrossover) * 180 / math.Pi; ph > -120 {
			t.Errorf("%s 1/(s+1)³: plant phase %g° at crossover leaves no lead", family, ph)
		}

		const tau = .05
		r, err = TunePID(context.Background(), delayedFirstOrder(t, tau), family, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("%s delayed: %v", family, err)
		}
		c = r.Controller
		if c.Kp <= 0 || c.Kd <= 0 || r.Evidence.Stability != PIDStabilityNyquist || !moved(r) {
			t.Errorf("%s delayed: %+v evidence %q", family, c, r.Evidence.Stability)
		}
		if stable, peak := pidDelayLoopCheck(t, c, tau); !stable || peak > 3 {
			t.Errorf("%s delayed: independent check stable=%v |S|max=%g", family, stable, peak)
		}
	}
	d, err := NewFromSlices(1, 1, 1, []float64{0.9}, []float64{1}, []float64{0.1}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.SetInputDelay([]float64{30}); err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), d, PidtunePD, PIDTuningOptions{})
	if err != nil {
		t.Fatalf("discrete delayed PD: %v", err)
	}
	if c := r.Controller; c.Kp <= 0 || c.Kd <= 0 || r.Evidence.AchievedCrossover >= math.Pi/d.Dt || !moved(r) {
		t.Errorf("discrete delayed PD %+v at %g, warnings %q", c, r.Evidence.AchievedCrossover, r.Evidence.Warnings)
	}
	neg := makePlant(t, []float64{-1}, []float64{1, 1})
	if err := neg.SetInputDelay([]float64{1}); err != nil {
		t.Fatal(err)
	}
	r, err = TunePID(context.Background(), neg, PidtunePD, PIDTuningOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if c := r.Controller; c.Kp >= 0 || c.Kd >= 0 || r.Evidence.Stability != PIDStabilityNyquist {
		t.Errorf("negative-DC delayed PD %+v evidence %q", c, r.Evidence.Stability)
	}
}

// The ½ tail cap guards the Kd·CB neutral chain only; P on a delayed
// biproper plant needs just limsup |L| < 1.
func TestTunePIDDelayTailCapIdealDerivativeOnly(t *testing.T) {
	p := makePlant(t, []float64{.5, 1}, []float64{1, 1})
	if err := p.SetInputDelay([]float64{1}); err != nil {
		t.Fatal(err)
	}
	eval, err := newSISOEval(p)
	if err != nil {
		t.Fatal(err)
	}
	stable, ok, err := pidTuningDelayStability(p, eval, 1, false)
	if err != nil || !ok {
		t.Fatalf("out of scope: %v", err)
	}
	for _, tc := range []struct {
		kp   float64
		want bool
	}{{1.4, true}, {1.6, true}, {1.8, false}, {2.1, false}} {
		if got, err := stable(&PID2{Kp: tc.kp}); err != nil || got != tc.want {
			t.Errorf("Kp=%g (tail %g): stable=%v err=%v, want %v", tc.kp, tc.kp/2, got, err, tc.want)
		}
	}
	for _, family := range []PidtuneType{PidtuneP, PidtunePI} {
		r, err := TunePID(context.Background(), p, family, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("%s: %v", family, err)
		}
		if c := r.Controller; c.Kp <= .5 || c.Ki < 0 || r.Evidence.Stability != PIDStabilityNyquist {
			t.Errorf("%s: %+v evidence %q", family, c, r.Evidence.Stability)
		}
	}
}

// G = −0.4k·s/((s+1)(s+3)) has DC gain exactly 0; rounding must not pick a
// gain sign that varies with the positive scale k.
func TestPidTuningDCSignIgnoresRoundingZero(t *testing.T) {
	sign := func(d, k float64) float64 {
		p, err := NewFromSlices(2, 1, 1, []float64{-1, 2, 0, -3}, []float64{1, 1}, []float64{.1 * k, -.5 * k}, []float64{d}, 0)
		if err != nil {
			t.Fatal(err)
		}
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		return pidTuningDCSign(p, eval.at, 10)
	}
	for _, k := range []float64{1, 3, 1.3, .9} {
		if s := sign(0, k); s != 0 {
			t.Errorf("k=%g: sign %g, want 0", k, s)
		}
	}
	if s := sign(1e-6, 1); s != 1 {
		t.Errorf("small genuine DC gain: sign %g, want 1", s)
	}
}
