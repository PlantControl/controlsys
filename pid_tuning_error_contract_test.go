package controlsys

import (
	"context"
	"errors"
	"strings"
	"testing"
)

// TunePID and TunePIDFRD return (res, nil) only for a certified design; an
// uncertified stability, margin or sensitivity peak returns the full design
// with ErrAccuracyUnverified (U5XPQB), and every other error a nil result.
func pidTuningContract(t *testing.T, name string, r *PIDTuningResult, err error) {
	t.Helper()
	if (r != nil) != (err == nil || errors.Is(err, ErrAccuracyUnverified)) {
		t.Fatalf("%s: result %v with err %v", name, r != nil, err)
	}
}

func pidTuningWantUnverified(t *testing.T, name, prefix, reason string, r *PIDTuningResult, err error) {
	t.Helper()
	pidTuningContract(t, name, r, err)
	if !errors.Is(err, ErrAccuracyUnverified) || !strings.HasPrefix(err.Error(), prefix+reason+" not certified: ") {
		t.Fatalf("%s: err %v, want %s%s not certified", name, err, prefix, reason)
	}
	if r == nil || r.Controller == nil || r.Evidence.Evaluations == 0 || len(r.Evidence.Warnings) == 0 {
		t.Fatalf("%s: incomplete result %+v", name, r)
	}
}

func TestTunePIDErrorContractCertified(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 2})
	for _, f := range []PidtuneType{PidtuneP, PidtunePI, PidtunePD, PidtunePID, PidtunePIDF} {
		r, err := TunePID(context.Background(), p, f, PIDTuningOptions{CrossoverFrequency: 1})
		pidTuningContract(t, string(f), r, err)
		if err != nil || r.Evidence.Stability != PIDStabilityClosedLoopPoles {
			t.Fatalf("%s: err %v, stability %q", f, err, r.Evidence.Stability)
		}
	}
	d := makePlant(t, []float64{1}, []float64{1, 3, 2})
	if err := d.SetInputDelay([]float64{.2}); err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), d, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1})
	pidTuningContract(t, "delayed PI", r, err)
	if err != nil || r.Evidence.Stability != PIDStabilityNyquist {
		t.Fatalf("delayed PI: err %v, stability %v", err, r)
	}
	frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, []float64{1}, []float64{1, 3, 2}, 0, .01, 100, 120), []float64{1}, []float64{1, 3, 2})
	r, err = TunePIDFRD(context.Background(), frd, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1})
	pidTuningContract(t, "FRD PI", r, err)
	if err != nil || pidHasWarning(r, "not certified") || r.Evidence.Stability != PIDStabilitySampledOnly {
		t.Fatalf("FRD PI: err %v, evidence %+v", err, r)
	}
}

func TestTunePIDErrorContractUncertifiedStability(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 2})
	if err := p.SetInputDelay([]float64{.2}); err != nil {
		t.Fatal(err)
	}
	lft, err := p.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), lft, PidtunePID, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 60})
	pidTuningWantUnverified(t, "LFT PID", "TunePID: ", "stability", r, err)
	if r.Evidence.Stability != PIDStabilitySampledOnly || !pidHasWarning(r, "do not certify global closed-loop stability") {
		t.Fatalf("LFT PID: evidence %+v", r.Evidence)
	}

	noDC := pidFRDStrictData(t, []float64{1}, []float64{1, 3, 2}, 0, .01, 100, 120)
	r, err = TunePIDFRD(context.Background(), noDC, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1})
	pidTuningWantUnverified(t, "FRD without DC", "TunePIDFRD: ", "stability", r, err)
	if !pidHasWarning(r, "stability is not certified") {
		t.Fatalf("FRD without DC: warnings %q", r.Evidence.Warnings)
	}
}

func TestTunePIDFRDErrorContractUncertifiedPeak(t *testing.T) {
	num, den := []float64{3.2276235464735703}, []float64{1, -0.1353352832366127}
	frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, num, den, 1, .1, 3, 235), num, den)
	r, err := TunePIDFRD(context.Background(), frd, PidtuneP, PIDTuningOptions{DFormula: BackwardEuler})
	pidTuningWantUnverified(t, "short of Nyquist P", "TunePIDFRD: ", "sensitivity peak", r, err)
	if !pidHasWarning(r, "sensitivity peak is not certified") || pidHasWarning(r, "stability is not certified") {
		t.Fatalf("warnings %q", r.Evidence.Warnings)
	}
}

// A search stopped by its evaluation budget with a certified design keeps
// a nil error: Termination records the budget, not the certificate.
func TestTunePIDErrorContractBudgetLimitCertified(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 2})
	r, err := TunePID(context.Background(), p, PidtunePID, PIDTuningOptions{CrossoverFrequency: 1, MaxEvaluations: 8})
	pidTuningContract(t, "budget", r, err)
	if err != nil || r.Evidence.Termination != PIDTerminationEvaluationLimit {
		t.Fatalf("err %v, result %+v", err, r)
	}
}

func TestTunePIDErrorContractOtherErrors(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 2})
	r, err := TunePID(context.Background(), p, PidtuneP, PIDTuningOptions{CrossoverFrequency: 1, PhaseMargin: 150, MaxEvaluations: 7})
	pidTuningContract(t, "unattainable", r, err)
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) || errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("unattainable: %v", err)
	}
	r, err = TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{CrossoverFrequency: -1})
	pidTuningContract(t, "invalid", r, err)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("invalid: %v", err)
	}
	one := 1
	r, err = TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{UnstablePoles: &one})
	pidTuningContract(t, "unsupported", r, err)
	if !errors.Is(err, ErrOptionUnsupported) {
		t.Fatalf("unsupported: %v", err)
	}
	r, err = TunePIDFRD(context.Background(), nil, PidtunePI, PIDTuningOptions{})
	pidTuningContract(t, "FRD invalid", r, err)
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("FRD invalid: %v", err)
	}
}

// pidTuneFRDKept runs TunePIDFRD for tests that inspect uncertified designs
// too: it checks that ErrAccuracyUnverified comes exactly with a "not
// certified" warning and returns it as nil.
func pidTuneFRDKept(t testing.TB, ctx context.Context, frd *FRD, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
	t.Helper()
	r, err := TunePIDFRD(ctx, frd, family, o)
	if (r != nil) != (err == nil || errors.Is(err, ErrAccuracyUnverified)) {
		t.Fatalf("TunePIDFRD: result %v with err %v", r != nil, err)
	}
	if r != nil && errors.Is(err, ErrAccuracyUnverified) != pidHasWarning(r, "not certified") {
		t.Fatalf("TunePIDFRD: err %v with warnings %q", err, r.Evidence.Warnings)
	}
	if errors.Is(err, ErrAccuracyUnverified) {
		err = nil
	}
	return r, err
}

// pidTuneKept is pidTuneFRDKept for TunePID, whose uncertified designs are
// those checked only at sampled frequencies.
func pidTuneKept(t testing.TB, ctx context.Context, plant *System, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
	t.Helper()
	r, err := TunePID(ctx, plant, family, o)
	if (r != nil) != (err == nil || errors.Is(err, ErrAccuracyUnverified)) {
		t.Fatalf("TunePID: result %v with err %v", r != nil, err)
	}
	if r != nil && errors.Is(err, ErrAccuracyUnverified) != (r.Evidence.Stability == PIDStabilitySampledOnly) {
		t.Fatalf("TunePID: err %v with stability %q", err, r.Evidence.Stability)
	}
	if errors.Is(err, ErrAccuracyUnverified) {
		err = nil
	}
	return r, err
}
