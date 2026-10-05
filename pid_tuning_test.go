package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestTunePIDAnalyticTargetsAndFixedWeights(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1})
	b, c := .35, .2
	result, err := TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60, Weights: &PIDTuningWeights{FixedB: &b, FixedC: &c}})
	if err != nil {
		t.Fatal(err)
	}
	if result.Controller.B != b || result.Controller.C != c {
		t.Fatal("changed fixed weights")
	}
	if result.Evidence.Stability != "closed-loop-poles" {
		t.Fatalf("evidence %+v", result.Evidence)
	}
	pid := result.Controller
	h := (complex(pid.Kp, 0) + complex(pid.Ki, 0)/1i) / (1 + 1i)
	if math.Abs(cmplx.Abs(h)-1) > 1e-12 || math.Abs(180+cmplx.Phase(h)*180/math.Pi-60) > 3.000001 {
		t.Fatalf("independent loop target %v", h)
	}
	if 1+pid.Kp <= 0 || pid.Ki <= 0 {
		t.Fatal("independent second-order Routh check failed")
	}
	if result.Evidence.Objective > result.Evidence.SeedObjective+1e-12 {
		t.Fatal("search degraded seed")
	}
}

func TestTunePIDFocusChangesDesign(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 3, 1})
	var results []*PIDTuningResult
	for _, focus := range []PIDDesignFocus{PIDFocusTracking, PIDFocusBalanced, PIDFocusRejection} {
		r, err := TunePID(context.Background(), p, PidtunePIDF, PIDTuningOptions{CrossoverFrequency: .7, PhaseMargin: 60, Focus: focus, Weights: &PIDTuningWeights{}})
		if err != nil {
			t.Fatalf("%s: %v", focus, err)
		}
		results = append(results, r)
		t.Logf("%s controller %+v objective %+v", focus, r.Controller, r.Evidence.Components)
	}
	a, b := results[0], results[2]
	if math.Abs(a.Controller.Kp-b.Controller.Kp)+math.Abs(a.Controller.Ki-b.Controller.Ki)+math.Abs(a.Controller.Kd-b.Controller.Kd) < 1e-7 {
		t.Fatal("focus had no effect")
	}
	if a.Evidence.Components.Tracking > b.Evidence.Components.Tracking*1.01 {
		t.Fatal("tracking focus did not favor tracking")
	}
	if b.Evidence.Components.Rejection > a.Evidence.Components.Rejection*1.01 {
		t.Fatal("rejection focus did not favor rejection")
	}
}

func TestTunePIDDiscreteBasisMatchesRealization(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1})
	sampled, err := p.C2D(.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	for _, formula := range []PIDFormula{ForwardEuler, BackwardEuler, Trapezoidal} {
		r, err := TunePID(context.Background(), sampled, PidtunePIDF, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60, IFormula: formula, DFormula: formula, Weights: &PIDTuningWeights{}})
		if err != nil {
			t.Fatal(err)
		}
		realized, err := r.Controller.System()
		if err != nil {
			t.Fatal(err)
		}
		w := []float64{.1, 1, 3}
		response, err := realized.FreqResponse(w)
		if err != nil {
			t.Fatal(err)
		}
		for k, freq := range w {
			if cmplx.Abs(response.At(k, 0, 0)-pidTuningReference(*r.Controller, freq)) > 1e-9 || cmplx.Abs(response.At(k, 0, 1)+pidTuningFeedback(*r.Controller, freq)) > 1e-9 {
				t.Fatalf("formula %v reference/feedback mismatch", formula)
			}
		}
	}
}

func TestTunePIDCancellationBoundsAndUnattainable(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1})
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := TunePID(ctx, p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1}); !errors.Is(err, context.Canceled) {
		t.Fatal(err)
	}
	// A P controller cannot add phase: this plant at wc=1 has PM=135, not 60.
	r, err := TunePID(context.Background(), p, PidtuneP, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60, MaxEvaluations: 7})
	var tuneErr *PIDTuningError
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) || !errors.As(err, &tuneErr) || r != nil {
		t.Fatalf("result %+v err %v", r, err)
	}
	if e := tuneErr.Evidence; e.Evaluations == 0 || e.Evaluations > 7 || e.RequestedCrossover != 1 || e.Termination != PIDTerminationEvaluationLimit || !strings.HasPrefix(err.Error(), "TunePID: ") {
		t.Fatalf("evidence %+v err %v", e, err)
	}
	for _, o := range []PIDTuningOptions{{CrossoverFrequency: math.NaN()}, {CrossoverFrequency: -1}, {CrossoverFrequency: 1, PhaseMargin: 180}, {CrossoverFrequency: 1, Focus: "unknown"}, {CrossoverFrequency: 1, MaxEvaluations: 4097}} {
		if _, err := TunePID(context.Background(), p, PidtunePI, o); !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "TunePID: ") {
			t.Fatalf("%+v: err %v, want ErrInvalidArgument", o, err)
		}
	}
	if _, err := TunePID(nil, p, PidtunePI, PIDTuningOptions{}); !errors.Is(err, ErrInvalidArgument) { //nolint:staticcheck
		t.Fatalf("nil ctx err %v", err)
	}
	if _, err := TunePID(context.Background(), nil, PidtunePI, PIDTuningOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil plant err %v", err)
	}
	known := 1
	if _, err := TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1, UnstablePoles: &known}); !errors.Is(err, ErrOptionUnsupported) {
		t.Fatalf("UnstablePoles on System path err %v, want ErrOptionUnsupported", err)
	}
}

func TestTunePIDPlantClasses(t *testing.T) {
	for _, tc := range []struct {
		name     string
		num, den []float64
		wc, pm   float64
		family   PidtuneType
	}{
		{"integrator", []float64{1}, []float64{1, 0}, 1, 60, PidtunePI},
		{"unstable", []float64{1}, []float64{1, -1}, 2, 60, PidtunePDF},
		{"NMP", []float64{-1, 1}, []float64{1, 2, 1}, .2, 60, PidtunePI},
		{"negative gain", []float64{-1}, []float64{1, 1}, 1, 60, PidtunePI},
		{"PDF", []float64{1}, []float64{1, 3, 3, 1}, 1, 60, PidtunePDF},
	} {
		t.Run(tc.name, func(t *testing.T) {
			r, err := TunePID(context.Background(), makePlant(t, tc.num, tc.den), tc.family, PIDTuningOptions{CrossoverFrequency: tc.wc, PhaseMargin: tc.pm})
			if err != nil {
				t.Fatal(err)
			}
			if r.Controller == nil {
				t.Fatal("no controller")
			}
		})
	}
	p := makePlant(t, []float64{1}, []float64{1, 1})
	if err := p.SetInputDelay([]float64{.2}); err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	if r.Evidence.Stability != "nyquist-encirclement" || slices.ContainsFunc(r.Evidence.Warnings, isDelayWarning) {
		t.Fatalf("delay stability evidence = %q %v", r.Evidence.Stability, r.Evidence.Warnings)
	}
	dm, err := DiskMargin(pidTuningOpenLoop(t, p, r.Controller))
	if err != nil {
		t.Fatal(err)
	}
	if dm.Alpha <= 0 {
		t.Fatalf("tuned delayed loop unstable: %+v", dm)
	}
}

func isDelayWarning(w string) bool { return strings.Contains(w, "delay") }

func pidTuningOpenLoop(t *testing.T, plant *System, c *PID2) *System {
	t.Helper()
	cs, err := mustPID(t, c.Kp, c.Ki, c.Kd, c.Tf, 0).System()
	if err != nil {
		t.Fatal(err)
	}
	l, err := Series(cs, plant)
	if err != nil {
		t.Fatal(err)
	}
	return l
}

// Closed-form boundaries for C(s)·e^{-τs}·G(s): the critical gain is
// 1/|L(jω)| at the first ω where arg L(jω) = -π.
func TestTunePIDDelayNyquistBoundary(t *testing.T) {
	const tau = 0.5
	first := makePlant(t, []float64{1}, []float64{1, 1})
	second := makePlant(t, []float64{1}, []float64{1, 3, 2})
	for _, p := range []*System{first, second} {
		if err := p.SetInputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
	}
	for _, tc := range []struct {
		name  string
		plant *System
		c     PID2
		g     func(s complex128) complex128
	}{
		{"P", first, PID2{Kp: 1}, func(s complex128) complex128 { return 1 / (s + 1) }},
		{"I", first, PID2{Ki: 1}, func(s complex128) complex128 { return 1 / (s * (s + 1)) }},
		{"PI", first, PID2{Kp: 1, Ki: 2}, func(s complex128) complex128 { return (1 + 2/s) / (s + 1) }},
		{"PDideal", second, PID2{Kp: 1, Kd: 0.5}, func(s complex128) complex128 { return (1 + 0.5*s) / ((s + 1) * (s + 2)) }},
		{"PIDF", second, PID2{Kp: 1, Ki: 0.5, Kd: 0.5, Tf: 0.1}, func(s complex128) complex128 {
			return (1 + 0.5/s + 0.5*s/(0.1*s+1)) / ((s + 1) * (s + 2))
		}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			eval, err := newSISOEval(tc.plant)
			if err != nil {
				t.Fatal(err)
			}
			stable, ok, err := pidTuningDelayStability(tc.plant, eval, 1, tc.c.Kd != 0 && tc.c.Tf == 0)
			if err != nil || !ok {
				t.Fatal("plant outside Nyquist scope")
			}
			L := func(w float64) complex128 {
				s := complex(0, w)
				return tc.g(s) * cmplx.Exp(-s*complex(tau, 0))
			}
			phase := func(w float64) float64 {
				ph := cmplx.Phase(L(w))
				if ph > 0 {
					ph -= 2 * math.Pi
				}
				return ph + math.Pi
			}
			lo := 1e-3
			hi := lo
			for phase(hi) > 0 {
				lo, hi = hi, hi*1.01
			}
			wc := bisectRoot(phase, lo, hi)
			kc := 1 / cmplx.Abs(L(wc))
			for _, f := range []struct {
				scale float64
				want  bool
			}{{0.97 * kc, true}, {1.03 * kc, false}} {
				c := tc.c
				c.Kp *= f.scale
				c.Ki *= f.scale
				c.Kd *= f.scale
				if got, err := stable(&c); err != nil || got != f.want {
					t.Fatalf("gain %g (kc=%g at w=%g): stable=%v, want %v", f.scale, kc, wc, got, f.want)
				}
			}
		})
	}
}

func TestTunePIDDiscreteDelayUsesClosedLoopPoles(t *testing.T) {
	p, err := NewFromSlices(1, 1, 1, []float64{0.9}, []float64{1}, []float64{0.1}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetInputDelay([]float64{3}); err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 0.5, PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	if r.Evidence.Stability != "closed-loop-poles" || slices.ContainsFunc(r.Evidence.Warnings, isDelayWarning) {
		t.Fatalf("evidence = %q %v", r.Evidence.Stability, r.Evidence.Warnings)
	}
}

func TestTunePIDIdealDerivativeFamiliesRemainIdeal(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 3, 1})
	for _, family := range []PidtuneType{PidtunePD, PidtunePID} {
		r, err := TunePID(context.Background(), p, family, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
		if err != nil {
			t.Fatalf("%s: %v", family, err)
		}
		if r.Controller.Tf != 0 {
			t.Fatal("introduced unrequested derivative filter")
		}
		if r.Controller.Kd == 0 {
			t.Fatal("fixture did not exercise derivative")
		}
		// For PD, characteristic s³+3s²+(3+Kd)s+(1+Kp) has
		// positive coefficients and 3*(3+Kd) > 1+Kp (Routh-Hurwitz).
		if family == PidtunePD && (r.Controller.Kp <= -1 || r.Controller.Kd <= -3 || 3*(3+r.Controller.Kd) <= 1+r.Controller.Kp) {
			t.Fatal("independent cubic stability check failed")
		}
	}
}

func TestTunePIDGainCrossingEvidenceIndependentMargin(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1})
	r, err := TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	controller := mustPID(t, r.Controller.Kp, r.Controller.Ki, 0, 0, 0)
	loop := pidOpenLoop(t, p, controller)
	margin, err := Margin(loop)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(margin.PhaseMargin-r.Evidence.AchievedPhaseMargin) > 1e-5 {
		t.Fatalf("independent margin %g evidence %g", margin.PhaseMargin, r.Evidence.AchievedPhaseMargin)
	}
	if len(r.Evidence.GainCrossovers) == 0 || len(r.Evidence.GainCrossovers) != len(r.Evidence.PhaseMargins) {
		t.Fatal("missing crossing evidence")
	}
}

func TestTunePIDProportionalAndIntegralFamilies(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1})
	for _, test := range []struct {
		family PidtuneType
		pm     float64
	}{{PidtuneP, 135}, {PidtuneI, 45}} {
		r, err := TunePID(context.Background(), p, test.family, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: test.pm})
		if err != nil {
			t.Fatal(err)
		}
		if test.family == PidtuneP && r.Controller.Ki != 0 || test.family == PidtuneI && r.Controller.Kp != 0 || r.Controller.Kd != 0 {
			t.Fatal("introduced absent controller term")
		}
	}
}

func TestTunePIDIdealDerivativeRejectsHiddenUnstableMode(t *testing.T) {
	visible := makePlant(t, []float64{1}, []float64{1, 3, 3, 1})
	n, _, _ := visible.Dims()
	for _, hidden := range []string{"uncontrollable", "unobservable"} {
		A := mat.NewDense(n+1, n+1, nil)
		A.Slice(0, n, 0, n).(*mat.Dense).Copy(visible.A)
		A.Set(n, n, .5)
		B := mat.NewDense(n+1, 1, nil)
		B.Slice(0, n, 0, 1).(*mat.Dense).Copy(visible.B)
		C := mat.NewDense(1, n+1, nil)
		C.Slice(0, 1, 0, n).(*mat.Dense).Copy(visible.C)
		if hidden == "uncontrollable" {
			C.Set(0, n, 1)
		} else {
			B.Set(n, 0, 1)
		}
		p, err := New(A, B, C, mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		for _, family := range []PidtuneType{PidtunePD, PidtunePID} {
			r, err := TunePID(context.Background(), p, family, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
			var tuneErr *PIDTuningError
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) || !errors.As(err, &tuneErr) || r != nil {
				t.Fatalf("%s %s: hidden unstable mode certified: %v", hidden, family, err)
			}
		}
	}
}

func TestTunePIDOneGainFamiliesTakeThePlantsMargin(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 3, 1})
	for _, family := range []PidtuneType{PidtuneP, PidtuneI} {
		for _, wc := range []float64{0.185, 0} {
			result, err := TunePID(context.Background(), p, family, PIDTuningOptions{CrossoverFrequency: wc})
			if err != nil {
				t.Fatalf("%s at crossover %g: %v", family, wc, err)
			}
			e := result.Evidence
			if e.RequestedPhaseMargin <= 0 || e.RequestedPhaseMargin >= 180 {
				t.Fatalf("%s at %g: evidence %+v", family, wc, e)
			}
			if wc > 0 && math.Abs(e.AchievedCrossover-wc) > 1e-6*wc {
				t.Fatalf("%s crossover %g, want %g", family, e.AchievedCrossover, wc)
			}
			if math.Abs(e.AchievedPhaseMargin-e.RequestedPhaseMargin) > 3 {
				t.Fatalf("%s margin %g, want the plant's %g", family, e.AchievedPhaseMargin, e.RequestedPhaseMargin)
			}
			c := result.Controller
			if family == PidtuneP && (c.Ki != 0 || c.Kp <= 0) || family == PidtuneI && (c.Kp != 0 || c.Ki <= 0) {
				t.Fatalf("%s gains %+v", family, c)
			}
		}
	}
	h := 1 / cmplx.Pow(complex(1, 0.185), 3)
	margin := 180 + cmplx.Phase(h)*180/math.Pi
	result, err := TunePID(context.Background(), p, PidtuneP, PIDTuningOptions{CrossoverFrequency: 0.185})
	if err != nil || math.Abs(result.Evidence.RequestedPhaseMargin-margin) > 1e-9 {
		t.Fatalf("P margin %v, want %g: %v", result, margin, err)
	}
	if _, err := TunePID(context.Background(), p, PidtuneP, PIDTuningOptions{CrossoverFrequency: 0.185, PhaseMargin: 42}); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("an explicit unreachable margin was not refused: %v", err)
	}
}

func TestTunePIDOneGainAcceptsSeveralCrossovers(t *testing.T) {
	// A zero pair (ζ 0.5) over a pole pair (ζ 0.1) at 5 rad/s lifts |L| above 1 there, so
	// the P loop crosses 0 dB three times with margins below the one at wc.
	p := makePlant(t, []float64{1, 5, 25}, []float64{1, 2, 26, 25})
	result, err := TunePID(context.Background(), p, PidtuneP, PIDTuningOptions{CrossoverFrequency: 1})
	if err != nil {
		t.Fatal(err)
	}
	e := result.Evidence
	if len(e.GainCrossovers) < 3 {
		t.Fatalf("expected several gain crossovers: %+v", e)
	}
	lowest := math.Inf(1)
	for _, margin := range e.PhaseMargins {
		lowest = math.Min(lowest, margin)
	}
	if lowest >= e.RequestedPhaseMargin-3 || math.Abs(e.AchievedPhaseMargin-lowest) > 1e-9 {
		t.Fatalf("achieved margin %g should be the lowest crossover margin %g, below the local %g", e.AchievedPhaseMargin, lowest, e.RequestedPhaseMargin)
	}
	// Sampled data has no stability certificate: the requested-margin floor stays.
	omega := make([]float64, 400)
	for k := range omega {
		omega[k] = math.Pow(10, -2+4*float64(k)/float64(len(omega)-1))
	}
	frd, err := p.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := TunePIDFRD(context.Background(), frd, PidtuneP, PIDTuningOptions{CrossoverFrequency: 1}); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("sampled one-gain design accepted below its local margin without a stability certificate: %v", err)
	}
}

func TestTunePIDDelayIdealDerivativeOutOfScopeFallsBack(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 2})
	if err := p.SetInputDelay([]float64{.2}); err != nil {
		t.Fatal(err)
	}
	lft, err := p.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	for family, want := range map[PidtuneType]PIDStabilityEvidence{PidtunePID: PIDStabilitySampledOnly, PidtunePIDF: PIDStabilityNyquist} {
		r, err := TunePID(context.Background(), lft, family, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
		if err != nil {
			t.Fatalf("%s: %v", family, err)
		}
		if r.Evidence.Stability != want {
			t.Fatalf("%s: stability %q, want %q", family, r.Evidence.Stability, want)
		}
	}
}

func TestTunePIDDiscreteDelayPIDClosedLoopStable(t *testing.T) {
	p, err := NewFromSlices(2, 1, 1, []float64{0.9, 0.2, 0, 0.7}, []float64{0, 1}, []float64{0.1, 0.05}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), p, PidtunePID, PIDTuningOptions{CrossoverFrequency: 0.5, PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	if r.Evidence.Stability != "closed-loop-poles" {
		t.Fatalf("stability %q", r.Evidence.Stability)
	}
	c := r.Controller
	if c.IFormula != ForwardEuler || c.DFormula != ForwardEuler || c.Tf != 0 {
		t.Fatalf("unexpected controller form %+v", c)
	}
	// P(z) = (0.05z-0.025)/((z-0.9)(z-0.7)z²); forward Euler PID
	// C(z) = (Kp(z-1)h + Ki h² + Kd(z-1)²)/((z-1)h).
	h := c.Dt
	den := polyMulTest([]float64{1, -1.6, 0.63, 0, 0}, []float64{h, -h})
	num := polyMulTest([]float64{0.05, -0.025}, []float64{c.Kd, c.Kp*h - 2*c.Kd, c.Kd - c.Kp*h + c.Ki*h*h})
	char := slices.Clone(den)
	for i := range num {
		char[len(char)-len(num)+i] += num[i]
	}
	n := len(char) - 1
	comp := mat.NewDense(n, n, nil)
	for j := range n {
		comp.Set(0, j, -char[j+1]/char[0])
	}
	for i := 1; i < n; i++ {
		comp.Set(i, i-1, 1)
	}
	var eig mat.Eigen
	if !eig.Factorize(comp, mat.EigenNone) {
		t.Fatal("eigen failed")
	}
	for _, z := range eig.Values(nil) {
		if cmplx.Abs(z) >= 1 {
			t.Fatalf("closed-loop pole %v outside unit circle (gains %+v)", z, c)
		}
	}
}

func polyMulTest(a, b []float64) []float64 {
	out := make([]float64, len(a)+len(b)-1)
	for i, x := range a {
		for j, y := range b {
			out[i+j] += x * y
		}
	}
	return out
}

func TestTunePIDStabilityCheckFailureAborts(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 1})
	eval, err := newSISOEval(plant)
	if err != nil {
		t.Fatal(err)
	}
	boom := errors.New("eigen failed")
	p := pidTuningPlant{low: .01, high: 100, at: eval.at, stability: PIDStabilityClosedLoopPoles,
		stable: func(*PID2) (bool, error) { return false, boom }}
	r, err := tunePID(context.Background(), "TunePID", p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
	var tuneErr *PIDTuningError
	if r != nil || !errors.Is(err, boom) || errors.As(err, &tuneErr) || !strings.HasPrefix(err.Error(), "TunePID: ") {
		t.Fatalf("r=%v err=%v, want aborted with the stability error", r, err)
	}
}

// pidTuningBasis must match a 200-bit evaluation of the discrete integrator
// and filtered derivative at the exact on-circle point, including small ωT
// where a rounded e^{jωT} − 1 loses ε/(ωT) relative accuracy.
func TestPidTuningBasisDiscreteSmallOmegaT(t *testing.T) {
	const dt, tf = 0.1, 0.05
	formulas := []PIDFormula{ForwardEuler, BackwardEuler, Trapezoidal}
	for _, fi := range formulas {
		for _, fd := range formulas {
			c := PID2{Dt: dt, Tf: tf, IFormula: fi, DFormula: fd}
			for _, wT := range []float64{1e-7, 1e-4, 0.3, 3} {
				w := wT / dt
				z := frExactPoint(w, dt)
				basis := func(f PIDFormula) frBig {
					return frBigC(dt).mul(frBigC(1).quo(z.sub(frBigC(1))).add(frBigC(complex(pidFormulaWeight(f), 0))))
				}
				d := frBigC(1).quo(basis(fd))
				wantI := basis(fi).complex()
				wantD := d.quo(frBigC(1).add(frBigC(tf).mul(d))).complex()
				gotI, gotD := pidTuningBasis(c, w)
				for _, r := range []struct {
					name      string
					got, want complex128
				}{{"integral", gotI, wantI}, {"derivative", gotD, wantD}} {
					if e := cmplx.Abs(r.got-r.want) / cmplx.Abs(r.want); !(e <= 1e-14) {
						t.Errorf("I=%v D=%v ωT=%g %s: %v, exact %v, rel err %.3g", fi, fd, wT, r.name, r.got, r.want, e)
					}
				}
			}
		}
	}
}
