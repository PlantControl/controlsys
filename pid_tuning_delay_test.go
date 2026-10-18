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
	return pidDelayLoopCheckPole(t, c, tau, 10)
}

// pidDelayLoopCheckPole is pidDelayLoopCheck for C(s)·e^{-τs}/(s+a), a > 0.
func pidDelayLoopCheckPole(t *testing.T, c *PID2, tau, a float64) (stable bool, peak float64) {
	t.Helper()
	return pidDelayLoopCheckPlant(t, c, tau, func(s complex128) complex128 { return 1 / (s + complex(a, 0)) }, 1)
}

// pidDelayLoopCheckPlant is pidDelayLoopCheck for C(s)·e^{-τs}·G(s), G
// stable with |G(jω)| <= g/ω^r for ω >= 1e4 (r = rel, g = 1).
func pidDelayLoopCheckPlant(t *testing.T, c *PID2, tau float64, g func(complex128) complex128, rel int) (stable bool, peak float64) {
	t.Helper()
	L := func(w float64) complex128 {
		s := complex(0, w)
		k := complex(c.Kp, 0) + complex(c.Kd, 0)*s/(1+complex(c.Tf, 0)*s)
		if c.Ki != 0 {
			k += complex(c.Ki, 0) / s
		}
		return k * cmplx.Exp(-s*complex(tau, 0)) * g(s)
	}
	const wEnd = 1e4
	gEnd := math.Pow(wEnd, -float64(rel))
	hf := (math.Abs(c.Kp) + math.Abs(c.Ki)/wEnd) * gEnd
	if c.Tf > 0 {
		hf += math.Abs(c.Kd) / c.Tf * gEnd
	} else {
		hf += math.Abs(c.Kd) * wEnd * gEnd
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
	// 1/(s+1)³ at ω=1.52 has phase −170°: same-sign PD reaches PM < 100°,
	// so PM 120° needs Kp < 0 at any band at or above it.
	for _, family := range []PidtuneType{PidtunePD, PidtunePDF} {
		_, err := TunePID(context.Background(), makePlant(t, []float64{1}, []float64{1, 3, 3, 1}), family, PIDTuningOptions{CrossoverFrequency: 1.52, PhaseMargin: 120})
		var te *PIDTuningError
		if !errors.As(err, &te) || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Fatalf("%s: err %v, want PIDTuningError", family, err)
		}
		if !slices.ContainsFunc(te.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "sign") }) {
			t.Errorf("%s: warnings %q do not name the sign rule", family, te.Evidence.Warnings)
		}
	}
	// At ω=10 e^{-0.05s}/(s+10) has phase −73.6°: PM 60° would need negative
	// Kd, so the margin floor takes the nearest same-sign design above it.
	r, err := TunePID(context.Background(), delayedFirstOrder(t, .05), PidtunePD, PIDTuningOptions{CrossoverFrequency: 10})
	if err != nil {
		t.Fatalf("explicit crossover with margin floor: %v", err)
	}
	if c := r.Controller; c.Kp <= 0 || c.Kd < 0 || r.Evidence.AchievedPhaseMargin < 57 {
		t.Fatalf("design %+v margin %g", c, r.Evidence.AchievedPhaseMargin)
	}
	if stable, _ := pidDelayLoopCheck(t, r.Controller, .05); !stable {
		t.Fatalf("design %+v independently unstable", r.Controller)
	}
	if pidTuningSignConsistent(PID2{Kp: 10.25, Kd: -.974}, 1) {
		t.Error("accepted negative Kd on positive-DC-gain plant")
	}
	if !pidTuningSignConsistent(PID2{Kp: -1, Kd: 2}, 0) {
		t.Error("sign rule applied without a known DC sign")
	}
	r, err = TunePID(context.Background(), makePlant(t, []float64{-1}, []float64{1, 1}), PidtunePI, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
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
	stable, ok, err := pidTuningDelayStability(p, eval, 10, true, pidTuningMaxTail)
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
	stable, ok, err := pidTuningDelayStability(p, eval, 1, true, pidTuningMaxTail)
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
	if err := d.SetInputDelay([]float64{3}); err != nil {
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

// The ideal-derivative tail cap guards the Kd·CB neutral chain only; P on a delayed
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
	stable, ok, err := pidTuningDelayStability(p, eval, 1, false, pidTuningMaxTail)
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

func pidSensitivityRejected(err error) bool {
	var te *PIDTuningError
	return errors.As(err, &te) && slices.ContainsFunc(te.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "sensitivity peak") })
}

// A delay-capped default crossover fixes only where |L| = 1; on a plant whose
// |G| is flat below the cap, P lands at DC loop gain ≈ 1 and moved-crossover
// PD leaves |L| ≈ 1 where the delay phase passes −180°. Such designs have
// |S| peaks of 10–100 despite healthy margins at wc and must be rejected.
func TestTunePIDDefaultDelayBoundsSensitivity(t *testing.T) {
	for _, tc := range []struct {
		a, tau float64
		family PidtuneType
		ok     bool
	}{
		{10, 2.1, PidtuneP, false},
		{1, 5, PidtuneP, false},
		{1, 5, PidtunePDF, false},
		{1, 5, PidtunePI, true},
		{1, 5, PidtunePID, true},
		{1, 1, PidtunePDF, true},
		{10, 2.1, PidtunePI, true},
	} {
		p := makePlant(t, []float64{1}, []float64{1, tc.a})
		if err := p.SetInputDelay([]float64{tc.tau}); err != nil {
			t.Fatal(err)
		}
		r, err := TunePID(context.Background(), p, tc.family, PIDTuningOptions{})
		if !tc.ok {
			if !pidSensitivityRejected(err) {
				t.Errorf("1/(s+%g) τ=%g %s: err %v, want sensitivity rejection", tc.a, tc.tau, tc.family, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("1/(s+%g) τ=%g %s: %v", tc.a, tc.tau, tc.family, err)
			continue
		}
		if stable, peak := pidDelayLoopCheckPole(t, r.Controller, tc.tau, tc.a); !stable || peak > pidTuningMaxSensitivity*1.01 {
			t.Errorf("1/(s+%g) τ=%g %s: %+v independent stable=%v |S|max=%g", tc.a, tc.tau, tc.family, r.Controller, stable, peak)
		}
	}
	p := makePlant(t, []float64{1}, []float64{1, 10})
	if err := p.SetInputDelay([]float64{2.1}); err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), p, PidtuneP, PIDTuningOptions{CrossoverFrequency: .4987})
	if err != nil {
		t.Fatalf("explicit crossover is the caller's choice: %v", err)
	}
	if _, peak := pidDelayLoopCheckPole(t, r.Controller, 2.1, 10); peak < 50 {
		t.Errorf("explicit P %+v |S|max=%g; fragile reference design changed", r.Controller, peak)
	}
	eval, err := newSISOEval(p)
	if err != nil {
		t.Fatal(err)
	}
	pp := pidTuningPlant{low: .004987, high: 49.87, at: eval.at, tau: 2.1}
	if peak := pidTuningSensitivityPeak(pp, *r.Controller); peak < 50 {
		t.Errorf("sampled |S| peak %g missed the near-neutral loop", peak)
	}
}

// 0.1/(z−0.9) with 30 samples of delay: the moved PD crossover meets the
// margin at wc only, with |S| ≈ 9 at the delay's −180° crossing.
func TestTunePIDDiscreteMovedCrossoverBoundsSensitivity(t *testing.T) {
	d, err := NewFromSlices(1, 1, 1, []float64{0.9}, []float64{1}, []float64{0.1}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := d.SetInputDelay([]float64{30}); err != nil {
		t.Fatal(err)
	}
	if _, err := TunePID(context.Background(), d, PidtunePD, PIDTuningOptions{}); !pidSensitivityRejected(err) {
		t.Fatalf("err %v, want sensitivity rejection", err)
	}
}

// An integrator with delay needs Kd·CB above ½; the tail cap must still admit
// a strongly stable ideal-derivative PID (f7f60ac tuned Kd ≈ 0.69 here).
func TestTunePIDDelayedIntegratorPID(t *testing.T) {
	const tau = 1.0
	p := makePlant(t, []float64{1}, []float64{1, 0})
	if err := p.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), p, PidtunePID, PIDTuningOptions{})
	if err != nil {
		t.Fatal(err)
	}
	c := r.Controller
	if c.Kp <= 0 || c.Ki < 0 || c.Kd < 0 || c.Kd > pidTuningMaxTail || r.Evidence.Stability != PIDStabilityNyquist {
		t.Fatalf("%+v evidence %q", c, r.Evidence.Stability)
	}
	minRet := math.Inf(1)
	for w := 1e-3; w < 1e3; w = math.Min(w*1.0005, w+1e-3) {
		s := complex(0, w)
		l := (complex(c.Kp, 0) + complex(c.Ki, 0)/s + complex(c.Kd, 0)*s) * cmplx.Exp(-s*complex(tau, 0)) / s
		minRet = math.Min(minRet, cmplx.Abs(1+l))
	}
	if minRet < 1/(pidTuningMaxSensitivity+.5) {
		t.Errorf("%+v: min|1+L| = %g", c, minRet)
	}
}

// 1/(s+1)² with a 3 s input delay: the PID/PIDF seeds meet PM 60° at the
// delay-capped wc but break |S| <= 4, and the search used to chase the
// best-scoring such design. A brute force at wc finds Kp=1.03 Ki=0.257
// Kd=0.85 with |S|=3.43, so feasibility must lead the search.
func TestTunePIDDelaySearchIsSensitivityAware(t *testing.T) {
	const tau = 3.0
	p := makePlant(t, []float64{1}, []float64{1, 2, 1})
	if err := p.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	g := func(s complex128) complex128 { return 1 / ((s + 1) * (s + 1)) }
	wc := (math.Pi - math.Pi/3) / (2 * tau)
	for _, family := range []PidtuneType{PidtunePID, PidtunePIDF} {
		r, err := TunePID(context.Background(), p, family, PIDTuningOptions{})
		if err != nil {
			t.Errorf("%s: %v", family, err)
			continue
		}
		c := r.Controller
		if math.Abs(r.Evidence.RequestedCrossover-wc) > 1e-12*wc || r.Evidence.Evaluations > 4096 {
			t.Errorf("%s: wc %g evaluations %d", family, r.Evidence.RequestedCrossover, r.Evidence.Evaluations)
		}
		if c.Kp <= 0 || c.Ki <= 0 || c.Kd <= 0 {
			t.Errorf("%s: %+v gains must share the DC sign", family, c)
		}
		if stable, peak := pidDelayLoopCheckPlant(t, c, tau, g, 2); !stable || peak > pidTuningMaxSensitivity*1.001 {
			t.Errorf("%s: %+v independent stable=%v |S|max=%g", family, c, stable, peak)
		}
		s := complex(0, wc)
		k := complex(c.Kp, 0) + complex(c.Ki, 0)/s + complex(c.Kd, 0)*s/(1+complex(c.Tf, 0)*s)
		l := k * cmplx.Exp(-s*complex(tau, 0)) * g(s)
		if pm := 180 + cmplx.Phase(l)*180/math.Pi; math.Abs(cmplx.Abs(l)-1) > 1e-9 || math.Abs(pm-60) > 3+1e-9 {
			t.Errorf("%s: |L(jwc)|=%g PM=%g", family, cmplx.Abs(l), pm)
		}
		loop := func(w float64) complex128 {
			s := complex(0, w)
			k := complex(c.Kp, 0) + complex(c.Ki, 0)/s + complex(c.Kd, 0)*s/(1+complex(c.Tf, 0)*s)
			return k * cmplx.Exp(-s*complex(tau, 0)) * g(s)
		}
		prev := cmplx.Abs(loop(1e-4)) - 1
		for w := 1e-4; w < 1e3; {
			w = math.Min(w*1.001, w+1e-3)
			m := cmplx.Abs(loop(w)) - 1
			if pm := 180 + cmplx.Phase(loop(w))*180/math.Pi; m*prev <= 0 && pm < 57-.05 {
				t.Errorf("%s: crossing %g PM %g", family, w, pm)
			}
			prev = m
		}
		again, err := TunePID(context.Background(), p, family, PIDTuningOptions{})
		if err != nil || *again.Controller != *c {
			t.Errorf("%s: rerun %+v %v, want %+v", family, again.Controller, err, c)
		}
	}
}

// 2.47e^{-0.0279s}/((s+0.125)(s+2.7)(s+6.4)): the default PID crossover is
// 0.07 rad/s, and an ideal derivative large enough to cross 0 dB again
// beyond the 100·wc band (Kd ≈ 137: 0.35° at 17.8 rad/s) must be rejected.
func TestTunePIDDefaultDelayBoundsSensitivityPastBand(t *testing.T) {
	num := []float64{2.472272636569423}
	den := []float64{1, 9.186133786090226, 18.145256089844757, 2.168002510114956}
	const tau = 0.027893978083026242
	p := makePlant(t, num, den)
	if err := p.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	g := func(s complex128) complex128 {
		return complex(num[0], 0) / (((s+complex(den[1], 0))*s+complex(den[2], 0))*s + complex(den[3], 0))
	}
	fragile := &PID2{Kp: 0.0684988011781491, Ki: 0.7383139487048626, Kd: 137.17716633336596, B: 1, C: 1}
	// |G| <= 1/ω² for ω >= 1e4 satisfies the oracle's tail precondition.
	if stable, peak := pidDelayLoopCheckPlant(t, fragile, tau, g, 2); !stable || peak < 100 {
		t.Fatalf("reference design stable=%v |S|max=%g; repro changed", stable, peak)
	}
	r, err := TunePID(context.Background(), p, PidtunePID, PIDTuningOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if stable, peak := pidDelayLoopCheckPlant(t, r.Controller, tau, g, 2); !stable || peak > pidTuningMaxSensitivity {
		t.Errorf("%+v independent stable=%v |S|max=%g, want stable with |S| <= %d", r.Controller, stable, peak, pidTuningMaxSensitivity)
	}
	if math.Abs(r.Evidence.AchievedPhaseMargin-60) > 3 {
		t.Errorf("achieved margin %g, want 60±3", r.Evidence.AchievedPhaseMargin)
	}
	eval, err := newSISOEval(p)
	if err != nil {
		t.Fatal(err)
	}
	wc := r.Evidence.AchievedCrossover
	pp := pidTuningPlant{low: wc / 100, high: wc * 100, at: eval.at, tau: tau, maxSensitivity: pidTuningMaxSensitivity}
	if peak := pidTuningSensitivityPeak(pp, *fragile); peak > 2 {
		t.Fatalf("in-band |S| peak %g; the repro's peak should lie past the band", peak)
	}
	pp.loopTail = pidTuningLoopTail(p)
	if peak := pidTuningSensitivityPeak(pp, *fragile); peak < 100 {
		t.Errorf("tail-extended |S| peak %g missed the crossing at 17.8 rad/s", peak)
	}
}
