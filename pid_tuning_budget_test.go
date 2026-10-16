package controlsys

import (
	"context"
	"errors"
	"math"
	"sync/atomic"
	"testing"
)

// pidCountingContext counts Err calls. TunePID checks the context once on
// entry, once per search (±3° attempt, moved crossover or floor band) and
// once per counted candidate evaluation, so Err calls − 1 − Evaluations is
// the number of searches run.
type pidCountingContext struct {
	context.Context
	errs atomic.Int64
}

func (c *pidCountingContext) Err() error {
	c.errs.Add(1)
	return c.Context.Err()
}

func pidTuneCounted(t *testing.T, plant *System, family PidtuneType, o PIDTuningOptions) (r *PIDTuningResult, ev PIDTuningEvidence, searches int, err error) {
	t.Helper()
	ctx := &pidCountingContext{Context: context.Background()}
	r, err = TunePID(ctx, plant, family, o)
	var te *PIDTuningError
	switch {
	case err == nil:
		ev = r.Evidence
	case errors.As(err, &te):
		ev = te.Evidence
	default:
		t.Fatalf("%s: %v", family, err)
	}
	return r, ev, int(ctx.errs.Load()) - 1 - ev.Evaluations, err
}

func pidDelayedLag(t *testing.T, a, tau float64) *System {
	t.Helper()
	p := makePlant(t, []float64{1}, []float64{1, a})
	if tau > 0 {
		if err := p.SetInputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

// MaxEvaluations caps the ±3° attempt, the moved crossovers and the floor
// bands together, and Evidence.Evaluations counts all of them.
func TestTunePIDEvaluationBudgetCoversWholeCall(t *testing.T) {
	cases := []struct {
		name   string
		plant  *System
		family PidtuneType
	}{
		{"1/(s+10)e^-0.3s", pidDelayedLag(t, 10, .3), PidtunePDF},
		{"1/(s+1) floor", pidDelayedLag(t, 1, 0), PidtunePD},
		{"1/(s+1)^3 ladder", makePlant(t, []float64{1}, []float64{1, 3, 3, 1}), PidtunePD},
	}
	for _, c := range cases {
		for _, limit := range []int{60, 400} {
			_, ev, searches, _ := pidTuneCounted(t, c.plant, c.family, PIDTuningOptions{MaxEvaluations: limit})
			bound := limit
			if bound == 0 {
				bound = 4096
			}
			// One ±3° attempt, four moved crossovers and 20 floor bands at most.
			if ev.Evaluations > bound || searches < 1 || searches > 25 {
				t.Errorf("%s limit %d: %d evaluations reported, %d searches inferred", c.name, limit, ev.Evaluations, searches)
			}
			if ev.Evaluations == bound && ev.Termination != PIDTerminationEvaluationLimit {
				t.Errorf("%s limit %d: termination %q at the cap", c.name, limit, ev.Termination)
			}
		}
	}
}

// On 1/(s(s+1)) a PI controller adds phase lead at wc only with Ki < 0, and
// s³ + s² + Kp s + Ki then fails Routh. Every candidate meeting PM 75°±3° is
// unstable, as is every candidate above, so the floor is not searched.
func TestTunePIDFloorSkippedWhenBandUnstable(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 1, 0})
	_, ev, searches, err := pidTuneCounted(t, p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 75})
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("err %v", err)
	}
	if searches != 1 || ev.Evaluations == 0 {
		t.Fatalf("%d searches, %d evaluations; want only the ±3° search", searches, ev.Evaluations)
	}
}

// A P controller has no phase of its own: on 1/(s+1) at wc = 1 only the
// 135° band is reachable, so the floor skips straight to it and fits a
// 40-evaluation budget. Kp = √2 gives |L(j)| = 1 and PM = 180° − 45°.
func TestTunePIDFloorSkipsUnreachableBands(t *testing.T) {
	r, ev, searches, err := pidTuneCounted(t, pidDelayedLag(t, 1, 0), PidtuneP, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 45, MaxEvaluations: 40})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(r.Controller.Kp-math.Sqrt2) > 1e-9 || math.Abs(ev.AchievedPhaseMargin-135) > 1e-6 {
		t.Fatalf("Kp %g margin %g, want √2 and 135", r.Controller.Kp, ev.AchievedPhaseMargin)
	}
	if ev.Evaluations > 40 || searches != 2 {
		t.Fatalf("%d evaluations, %d searches", ev.Evaluations, searches)
	}
}

// 1/(s+10)e^{-0.3s} PDF: the floor bands above the default margin miss the
// |S| bound by more as the margin rises, so the floor stops after the first
// bands rather than searching all of them.
func TestTunePIDFloorStopsWhenSensitivityWorsens(t *testing.T) {
	_, ev, searches, err := pidTuneCounted(t, pidDelayedLag(t, 10, .3), PidtunePDF, PIDTuningOptions{})
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("err %v", err)
	}
	if searches > 8 || ev.Evaluations >= 2000 {
		t.Fatalf("%d searches, %d evaluations", searches, ev.Evaluations)
	}
}
