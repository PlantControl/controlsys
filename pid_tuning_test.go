package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
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
	if result.Evidence.Stability != "closed-loop-poles" || !result.Evidence.Feasible {
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
	sampled, err := p.DiscretizeZOH(.1)
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
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) || r.Evidence.Evaluations > 7 || r.Evidence.Feasible {
		t.Fatalf("result %+v err %v", r, err)
	}
	for _, o := range []PIDTuningOptions{{CrossoverFrequency: math.NaN()}, {CrossoverFrequency: -1}, {CrossoverFrequency: 1, PhaseMargin: 180}, {CrossoverFrequency: 1, Focus: "unknown"}, {CrossoverFrequency: 1, MaxEvaluations: 4097}} {
		if _, err := TunePID(context.Background(), p, PidtunePI, o); err == nil {
			t.Fatalf("accepted %+v", o)
		}
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
			if !r.Evidence.Feasible {
				t.Fatal("not feasible")
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
	if r.Evidence.Stability != "sampled-frequency-only" || len(r.Evidence.Warnings) == 0 {
		t.Fatal("delay stability overclaimed")
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
	controller := NewPID(r.Controller.Kp, r.Controller.Ki, 0)
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
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) || r == nil || r.Evidence.Feasible {
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
			if !e.Feasible || e.RequestedPhaseMargin <= 0 || e.RequestedPhaseMargin >= 180 {
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
	if len(e.GainCrossovers) < 3 || !e.Feasible {
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
