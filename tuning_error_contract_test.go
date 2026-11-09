package controlsys

import (
	"context"
	"errors"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Systune and Looptune return (res, nil) only when every hard goal is met; a
// missed hard goal returns the full result with ErrHardGoalsNotMet, a soft
// shortfall alone stays nil, and every other error has a nil result (GAV3BD).
func systuneContract(t *testing.T, name string, r *SystuneResult, err error) {
	t.Helper()
	if (r != nil) != (err == nil || errors.Is(err, ErrHardGoalsNotMet)) {
		t.Fatalf("%s: result %v with err %v", name, r != nil, err)
	}
	if r != nil && r.Pass != (err == nil) {
		t.Fatalf("%s: Pass %v with err %v", name, r.Pass, err)
	}
}

func systuneWantHardMiss(t *testing.T, name, prefix string, failing []string, nsoft int, r *SystuneResult, err error) {
	t.Helper()
	systuneContract(t, name, r, err)
	if !errors.Is(err, ErrHardGoalsNotMet) || errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("%s: err %v, want ErrHardGoalsNotMet", name, err)
	}
	want := prefix + "hard goals \"" + strings.Join(failing, "\", \"") + "\" not met (HardScore "
	if !strings.HasPrefix(err.Error(), want) {
		t.Fatalf("%s: err %q, want prefix %q", name, err, want)
	}
	if r.Controller == nil || r.ClosedLoop == nil || len(r.Parameters) == 0 || r.Iterations == 0 || !(r.HardScore > 0) || math.IsInf(r.HardScore, 0) {
		t.Fatalf("%s: incomplete result %+v", name, r)
	}
	var got []string
	for _, g := range r.Goals[nsoft:] {
		if !g.Pass {
			got = append(got, g.GoalName)
		}
	}
	if strings.Join(got, ",") != strings.Join(failing, ",") {
		t.Fatalf("%s: failing goals %q, want %q", name, got, failing)
	}
}

func systuneContractLoop(t *testing.T) *GeneralizedClosedLoop {
	t.Helper()
	return mustOK(NewGeneralizedClosedLoop("loop", makeSISO(-2, 1, 1, 0), tuningGain(t, 0.1, 5), "u"))
}

func TestSystuneErrorContractHardGoalsMet(t *testing.T) {
	soft := []TuningGoal{mustOK(NewTrackingGoal("u", "u", 1, 0, 0))}
	hard := []TuningGoal{mustOK(NewPolesGoal("", 3, 0, math.Inf(1)))}
	r, err := Systune(context.Background(), systuneContractLoop(t), soft, hard, &SystuneOptions{GridPoints: 9})
	systuneContract(t, "mixed", r, err)
	if err != nil || r.HardScore != 0 {
		t.Fatalf("mixed: err %v, result %+v", err, r)
	}
	r, err = Systune(context.Background(), systuneContractLoop(t), nil, hard, &SystuneOptions{GridPoints: 9})
	systuneContract(t, "hard only", r, err)
	if err != nil {
		t.Fatalf("hard only: %v", err)
	}
}

func TestSystuneErrorContractHardGoalMissed(t *testing.T) {
	soft := []TuningGoal{mustOK(NewTrackingGoal("u", "u", 1, 0, 0))}
	hard := []TuningGoal{
		mustOK(NewPolesGoal("", 3, 0, math.Inf(1))).WithName("decay3"),
		mustOK(NewPolesGoal("", 100, 0, math.Inf(1))).WithName("decay100"),
	}
	r, err := Systune(context.Background(), systuneContractLoop(t), soft, hard, &SystuneOptions{GridPoints: 9})
	systuneWantHardMiss(t, "decay100", "Systune: ", []string{"decay100"}, len(soft), r, err)
	if r.Parameters["K"] != 5 || len(r.Goals) != 3 {
		t.Fatalf("decay100: result %+v", r)
	}
}

// A soft-goal shortfall is reported by Score, not by the error.
func TestSystuneErrorContractSoftShortfall(t *testing.T) {
	soft := []TuningGoal{mustOK(NewPolesGoal("", 100, 0, math.Inf(1)))}
	r, err := Systune(context.Background(), systuneContractLoop(t), soft, nil, &SystuneOptions{GridPoints: 9})
	systuneContract(t, "soft only", r, err)
	if err != nil || !r.Pass || !(r.Score > 0) || r.Goals[0].Pass {
		t.Fatalf("soft only: err %v, result %+v", err, r)
	}
	hard := []TuningGoal{mustOK(NewPolesGoal("", 3, 0, math.Inf(1)))}
	r, err = Systune(context.Background(), systuneContractLoop(t), soft, hard, &SystuneOptions{GridPoints: 9})
	systuneContract(t, "soft miss, hard met", r, err)
	if err != nil || !(r.Score > 0) || r.HardScore != 0 {
		t.Fatalf("soft miss, hard met: err %v, result %+v", err, r)
	}
}

func looptuneContractPlant(t *testing.T) *System {
	t.Helper()
	return mustOK(New(mat.NewDense(2, 2, []float64{0, 1, 0, -1}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), 0))
}

func TestLooptuneErrorContract(t *testing.T) {
	plant := looptuneContractPlant(t)
	r, err := Looptune(context.Background(), plant, tuningGain(t, 0.1, 5), []float64{0.5, 2}, nil, &LooptuneOptions{GridPoints: 50})
	systuneContract(t, "met", r, err)
	if err != nil {
		t.Fatalf("met: %v", err)
	}
	r, err = Looptune(context.Background(), plant, tuningGain(t, 0.1, 5), []float64{50}, nil, &LooptuneOptions{GridPoints: 9})
	systuneWantHardMiss(t, "crossover 50", "Looptune: ", []string{"LoopShape", "Margins"}, 0, r, err)
	req := mustOK(NewPolesGoal("", 100, 0, math.Inf(1))).WithName("decay100")
	r, err = Looptune(context.Background(), plant, tuningGain(t, 0.1, 5), []float64{0.5, 2}, []TuningGoal{req}, &LooptuneOptions{GridPoints: 50})
	systuneWantHardMiss(t, "user req", "Looptune: ", []string{"decay100"}, 0, r, err)
}

func TestSystuneErrorContractOtherErrors(t *testing.T) {
	loop := systuneContractLoop(t)
	goals := []TuningGoal{mustOK(NewPolesGoal("", 100, 0, math.Inf(1)))}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	for _, tc := range []struct {
		name string
		run  func() (*SystuneResult, error)
		want error
	}{
		{"canceled", func() (*SystuneResult, error) { return Systune(ctx, loop, nil, goals, nil) }, context.Canceled},
		{"no goals", func() (*SystuneResult, error) { return Systune(context.Background(), loop, nil, nil, nil) }, ErrInvalidArgument},
		{"nil model", func() (*SystuneResult, error) { return Systune(context.Background(), nil, nil, goals, nil) }, ErrInvalidArgument},
		{"grid too large", func() (*SystuneResult, error) {
			return Systune(context.Background(), loop, nil, goals, &SystuneOptions{GridPoints: 10, MaxEvaluations: 5})
		}, ErrInvalidArgument},
		{"all candidates fail", func() (*SystuneResult, error) {
			unstable := mustOK(NewGeneralizedClosedLoop("loop", makeSISO(10, 1, 1, 0), tuningGain(t, 0, 1), "u"))
			return Systune(context.Background(), unstable, nil, []TuningGoal{mustOK(NewOvershootGoal("u", "u", 50))}, nil)
		}, nil},
		{"Looptune nil C0", func() (*SystuneResult, error) {
			return Looptune(context.Background(), looptuneContractPlant(t), nil, []float64{1}, nil, nil)
		}, ErrInvalidArgument},
		{"Looptune negative margin", func() (*SystuneResult, error) {
			return Looptune(context.Background(), looptuneContractPlant(t), tuningGain(t, 0.1, 5), []float64{1}, nil, &LooptuneOptions{GainMarginDB: -1})
		}, ErrInvalidArgument},
		{"Looptune bad wc", func() (*SystuneResult, error) {
			return Looptune(context.Background(), looptuneContractPlant(t), tuningGain(t, 0.1, 5), []float64{2, 1}, nil, nil)
		}, ErrInvalidArgument},
		{"Looptune canceled", func() (*SystuneResult, error) {
			return Looptune(ctx, looptuneContractPlant(t), tuningGain(t, 0.1, 5), []float64{1}, nil, nil)
		}, context.Canceled},
	} {
		r, err := tc.run()
		systuneContract(t, tc.name, r, err)
		if err == nil || r != nil || errors.Is(err, ErrHardGoalsNotMet) || (tc.want != nil && !errors.Is(err, tc.want)) {
			t.Errorf("%s: res %v, err %v, want nil result with %v", tc.name, r != nil, err, tc.want)
		}
		if !strings.HasPrefix(err.Error(), "Systune: ") && !strings.HasPrefix(err.Error(), "Looptune: ") {
			t.Errorf("%s: err %q lacks op prefix", tc.name, err)
		}
	}
}

// GridTune treats every goal as soft: a missed goal sets Pass false with a
// nil error.
func TestGridTuneErrorContractMissedGoal(t *testing.T) {
	goals := []TuningGoal{mustOK(NewPolesGoal("", 100, 0, math.Inf(1)))}
	r, err := GridTune(context.Background(), systuneContractLoop(t), goals, &SystuneOptions{GridPoints: 9})
	if err != nil || r == nil || r.Pass || !(r.Score > 0) {
		t.Fatalf("GridTune: err %v, result %+v", err, r)
	}
}
