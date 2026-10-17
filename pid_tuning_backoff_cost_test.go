package controlsys

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
)

// The back-off ran on every failing default PI/PID/PIDF call on a stable
// delayed plant, each octave with a full budget. These plants fail on sign
// alone, with no lightly damped mode and no |S| rejection, and took 10-20×
// the evaluations to fail anyway; MaxEvaluations 100 still spent 2200. Base
// 15ca52c fails them within 266, 200 and 130 evaluations. A design found
// without backing off would also do.
func TestTunePIDBackOffNeedsResonanceOrSensitivity(t *testing.T) {
	for _, c := range []struct {
		num, den []float64
		tau      float64
		fam      PidtuneType
		o        PIDTuningOptions
		maxEvals int
	}{
		{pidPoly(53.3, []float64{1, -.182}), []float64{1, 57.6, 2336}, .0555, PidtunePIDF, PIDTuningOptions{}, 600},
		{pidPoly(53.3, []float64{1, -.182}), []float64{1, 57.6, 2336}, .0555, PidtunePIDF, PIDTuningOptions{MaxEvaluations: 100}, 200},
		{pidPoly(-1, []float64{1.85, 3.02, .28}), []float64{1, 30.2, 1149, 28005}, 2.363, PidtunePI, PIDTuningOptions{PhaseMargin: 45}, 300},
	} {
		ev, err := pidTuningCallEvidence(TunePID(context.Background(), pidPlantDelayed(t, c.num, c.den, c.tau), c.fam, c.o))
		if err != nil {
			t.Errorf("%v %s: %v", c.den, c.fam, err)
			continue
		}
		if ev.Evaluations > c.maxEvals {
			t.Errorf("%v %s %+v: %d evaluations, want <= %d", c.den, c.fam, c.o, ev.Evaluations, c.maxEvals)
		}
		if slices.ContainsFunc(ev.Warnings, func(w string) bool { return strings.Contains(w, "also tried") || strings.Contains(w, "lowered to") }) {
			t.Errorf("%v %s: backed off: %q", c.den, c.fam, ev.Warnings)
		}
	}
}

// The back-off's searches share one MaxEvaluations budget: with 100, a
// resonant plant's call spends at most 100 on the default crossover's ±3°
// search, 100 on its floor and 100 on the back-off.
func TestTunePIDBackOffSharesBudget(t *testing.T) {
	plant := pidPlantDelayed(t, []float64{1}, []float64{1, .05, 1}, 1)
	for _, fam := range []PidtuneType{PidtunePI, PidtunePID, PidtunePIDF} {
		ev, err := pidTuningCallEvidence(TunePID(context.Background(), plant, fam, PIDTuningOptions{MaxEvaluations: 100}))
		if err != nil {
			t.Fatalf("%s: %v", fam, err)
		}
		if ev.Evaluations > 300 {
			t.Errorf("%s: %d evaluations, want <= 300", fam, ev.Evaluations)
		}
		if !slices.ContainsFunc(ev.Warnings, func(w string) bool { return strings.Contains(w, "also tried") || strings.Contains(w, "lowered to") }) {
			t.Errorf("%s: no back-off in %q", fam, ev.Warnings)
		}
	}
}

// pidTuningCallEvidence returns the evidence of a design or of an
// unattainable target, and any other error.
func pidTuningCallEvidence(r *PIDTuningResult, err error) (*PIDTuningEvidence, error) {
	if err == nil {
		return &r.Evidence, nil
	}
	if te, ok := errors.AsType[*PIDTuningError](err); ok {
		return &te.Evidence, nil
	}
	return nil, err
}

// An RHP zero and a ζ=0.025 mode below the default crossover leave no
// same-sign design at wc0, and the bisected back-off spent its whole budget
// even though it found one: 4161-4229 evaluations. The scan down stops at
// its first design; the deep gate adds two depths.
func TestTunePIDBackOffStopsAtFirstDesign(t *testing.T) {
	plant := pidPlantDelayed(t, []float64{2.4655092866840786, -3.982218498257481}, []float64{1, 1.0125481422685352, 1.0543664790048073, .9556591577782928}, .4563737532946457)
	for _, fam := range []PidtuneType{PidtunePI, PidtunePID, PidtunePIDF} {
		r, err := TunePID(context.Background(), plant, fam, PIDTuningOptions{})
		if err != nil {
			t.Errorf("%s: %v", fam, err)
			continue
		}
		if r.Evidence.Evaluations > 3900 {
			t.Errorf("%s: %d evaluations, want <= 3900", fam, r.Evidence.Evaluations)
		}
	}
}

// An RHP zero and a ζ=0.03 mode near wc0 leave no same-sign PID or PIDF at
// any depth, and the scan down from wc0 tried all twenty: 4197 and 4229
// evaluations against base 15ca52c's 202 and 133. After the two shallowest
// depths the back-off tries the two deepest and, both failing, stops. Oracle
// for "no design": a full scan of every depth, each with its own budget,
// found none.
func TestTunePIDBackOffFailsFastWhenDeepestFail(t *testing.T) {
	plant := pidPlantDelayed(t, []float64{1.0794, -.89075}, []float64{1, 18.247, 107.37, 1711.56}, .01834)
	for _, fam := range []PidtuneType{PidtunePID, PidtunePIDF} {
		_, err := TunePID(context.Background(), plant, fam, PIDTuningOptions{PhaseMargin: 68.08})
		te, ok := errors.AsType[*PIDTuningError](err)
		if !ok {
			t.Errorf("%s: %v, want unattainable", fam, err)
			continue
		}
		if te.Evidence.Evaluations > 600 {
			t.Errorf("%s: %d evaluations, want <= 600", fam, te.Evidence.Evaluations)
		}
		if !slices.ContainsFunc(te.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "also tried") }) {
			t.Errorf("%s: no back-off in %q", fam, te.Evidence.Warnings)
		}
	}
}
