package controlsys

import (
	"context"
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSystuneTunesSISOTunableGain(t *testing.T) {
	plant := makeSISO(-2, 1, 1, 0)
	k, _ := newBoundedReal("K", 0.1, 0.1, 5)
	controller := mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k}}))
	closed, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		t.Fatal(err)
	}

	result, err := Systune(context.Background(), closed, []TuningGoal{mustOK(NewTrackingGoal("track", 0.4))}, nil, &SystuneOptions{GridPoints: 9})
	if err != nil {
		t.Fatalf("Systune: %v", err)
	}
	if !result.Pass {
		t.Fatalf("expected tuned result to pass, got %#v", result)
	}
	if result.Method != "cartesian-grid" {
		t.Fatalf("method = %q, want cartesian-grid", result.Method)
	}
	if result.Parameters["K"] <= 0.1 {
		t.Fatalf("K was not increased: %#v", result.Parameters)
	}
	if len(result.Goals) != 1 || !result.Goals[0].Pass {
		t.Fatalf("goal diagnostics = %#v", result.Goals)
	}
}

func TestGridTuneLimitsCartesianSearch(t *testing.T) {
	plant := benchSysNonSym(2, 2, 2)
	k1, _ := newBoundedReal("K1", 0.1, 0.1, 2)
	k2, _ := newBoundedReal("K2", 0.1, 0.1, 2)
	controller := mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k1, fixedReal(t, "z12_limit", 0)}, {fixedReal(t, "z21_limit", 0), k2}}))
	closed, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		t.Fatal(err)
	}
	_, err = GridTune(context.Background(), closed, []TuningGoal{mustOK(NewWeightedGainGoal("bounded", 10))}, &SystuneOptions{GridPoints: 5, MaxEvaluations: 24})
	if !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("GridTune error = %v, want ErrInvalidArgument", err)
	}
}

func TestSystuneTunesSmallMIMOTunableGain(t *testing.T) {
	plant := benchSysNonSym(2, 2, 2)
	k1, _ := newBoundedReal("K1", 0.1, 0.1, 2)
	k2, _ := newBoundedReal("K2", 0.1, 0.1, 2)
	controller := mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k1, fixedReal(t, "z12", 0)}, {fixedReal(t, "z21", 0), k2}}))
	closed, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		t.Fatal(err)
	}

	result, err := Systune(context.Background(), closed, nil, []TuningGoal{mustOK(NewWeightedGainGoal("bounded", 10))}, &SystuneOptions{GridPoints: 5})
	if err != nil {
		t.Fatalf("Systune: %v", err)
	}
	if !result.Pass {
		t.Fatalf("expected MIMO tuning pass, got %#v", result)
	}
	if _, ok := result.Parameters["K1"]; !ok {
		t.Fatalf("missing K1 in parameters: %#v", result.Parameters)
	}
}

func TestSystuneUsesTunableBlockInterface(t *testing.T) {
	plant := makeSISO(-2, 1, 1, 0)
	k, _ := newBoundedReal("K", 0.1, 0.1, 5)
	controller := wrappedTunableGain{gain: mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k}}))}
	closed, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		t.Fatal(err)
	}

	result, err := Systune(context.Background(), closed, []TuningGoal{mustOK(NewTrackingGoal("track", 0.4))}, nil, &SystuneOptions{GridPoints: 9})
	if err != nil {
		t.Fatalf("Systune: %v", err)
	}
	if !result.Pass {
		t.Fatalf("expected wrapped tunable block to tune, got %#v", result)
	}
}

func TestSystuneUnsupportedControllerFailsClearly(t *testing.T) {
	plant := makeSISO(-1, 1, 1, 0)
	fixed := makeSISO(-2, 1, 1, 0)
	closed, err := NewGeneralizedClosedLoop("loop", plant, fixedBlockT(t, fixed), "u")
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Systune(context.Background(), closed, []TuningGoal{mustOK(NewWeightedGainGoal("gain", 1))}, nil, nil); err == nil {
		t.Fatal("expected unsupported fixed controller to fail")
	}
}

type wrappedTunableGain struct {
	gain *TunableGain
}

func (w wrappedTunableGain) CurrentSystem() (*System, error) {
	return w.gain.CurrentSystem()
}

func (w wrappedTunableGain) FreeParameters() []*TunableReal {
	return w.gain.FreeParameters()
}

func (w wrappedTunableGain) SampleBlock(values map[string]float64) (TunableBlock, error) {
	sampled, err := w.gain.Sample(values)
	if err != nil {
		return nil, err
	}
	return wrappedTunableGain{gain: sampled}, nil
}

func tuningGain(t *testing.T, lo, hi float64) *TunableGain {
	t.Helper()
	k := mustOK(newBoundedReal("K", lo, lo, hi))
	return mustOK(tunableGainWith("Kblock", [][]*TunableReal{{k}}))
}

func TestGridTuneContextAndBounds(t *testing.T) {
	closed := mustOK(NewGeneralizedClosedLoop("loop", makeSISO(-2, 1, 1, 0), tuningGain(t, 0.1, 5), "u"))
	goals := []TuningGoal{mustOK(NewTrackingGoal("track", 0.4))}
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := GridTune(ctx, closed, goals, nil); !errors.Is(err, context.Canceled) {
		t.Errorf("canceled ctx: err = %v, want context.Canceled", err)
	}
	if _, err := GridTune(context.Background(), closed, goals, &SystuneOptions{GridPoints: -1}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("negative GridPoints: err = %v, want ErrInvalidArgument", err)
	}
	free := mustOK(NewTunableReal("K", 1))
	unbounded := mustOK(NewGeneralizedClosedLoop("loop", makeSISO(-2, 1, 1, 0), mustOK(tunableGainWith("Kblock", [][]*TunableReal{{free}})), "u"))
	if _, err := GridTune(context.Background(), unbounded, goals, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("unbounded free parameter: err = %v, want ErrInvalidArgument", err)
	}
}

func TestGridTuneSkipsFailingCandidates(t *testing.T) {
	goals := []TuningGoal{mustOK(NewOvershootGoal("os", 50))}
	// Plant 1/(s-1): the closed loop 1-K is unstable for K < 1, where the
	// overshoot goal cannot be evaluated; those grid points are skipped.
	closed := mustOK(NewGeneralizedClosedLoop("loop", makeSISO(1, 1, 1, 0), tuningGain(t, 0, 4), "u"))
	res, err := GridTune(context.Background(), closed, goals, &SystuneOptions{GridPoints: 5})
	if err != nil {
		t.Fatalf("GridTune: %v", err)
	}
	if k := res.Parameters["K"]; k <= 1 || res.Controller == nil || res.ClosedLoop == nil {
		t.Errorf("best K = %g controller=%v, want a stabilizing K > 1", k, res.Controller)
	}
	allUnstable := mustOK(NewGeneralizedClosedLoop("loop", makeSISO(10, 1, 1, 0), tuningGain(t, 0, 1), "u"))
	if _, err := GridTune(context.Background(), allUnstable, goals, nil); err == nil {
		t.Error("every candidate unstable: want the first candidate error")
	}
}

func TestSystuneSoftAndHardGoals(t *testing.T) {
	closed := mustOK(NewGeneralizedClosedLoop("loop", makeSISO(-2, 1, 1, 0), tuningGain(t, 0.1, 5), "u"))
	soft := []TuningGoal{mustOK(NewTrackingGoal("track", 0))}
	hard := []TuningGoal{mustOK(NewPoleGoal("slow", -3))}
	res, err := Systune(context.Background(), closed, soft, hard, &SystuneOptions{GridPoints: 9})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Pass || res.HardScore != 0 || len(res.Goals) != 2 {
		t.Fatalf("result = %+v, want hard goals met", res)
	}
	impossible := []TuningGoal{mustOK(NewPoleGoal("impossible", -100))}
	res, err = Systune(context.Background(), closed, soft, impossible, &SystuneOptions{GridPoints: 9})
	if err != nil {
		t.Fatal(err)
	}
	if res.Pass || res.HardScore <= 0 || res.Parameters["K"] != 5 {
		t.Errorf("infeasible hard goal: Pass=%v HardScore=%g K=%g, want false, >0, 5", res.Pass, res.HardScore, res.Parameters["K"])
	}
}

func TestLooptuneCrossoverBand(t *testing.T) {
	plant, err := New(mat.NewDense(2, 2, []float64{0, 1, 0, -1}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Looptune(context.Background(), plant, tuningGain(t, 0.1, 5), []float64{1}, nil, &LooptuneOptions{GridPoints: 50})
	if err != nil {
		t.Fatal(err)
	}
	if !res.Pass {
		t.Fatalf("Looptune did not meet crossover/margins: %+v", res.Goals)
	}
	K := res.Parameters["K"]
	// L = K/(s(s+1)); |L(jw)| = 1 at w² = (-1 + sqrt(1+4K²))/2.
	wc := math.Sqrt((-1 + math.Sqrt(1+4*K*K)) / 2)
	if wc < 0.5 || wc > 2 {
		t.Errorf("crossover %g outside [0.5, 2] for K=%g", wc, K)
	}
	if pm := 180 - 90 - math.Atan(wc)*180/math.Pi; pm < 45 {
		t.Errorf("phase margin %g < 45 for K=%g", pm, K)
	}
	for _, wc := range [][]float64{nil, {2, 1}, {-1}, {1, 2, 3}} {
		if _, err := Looptune(context.Background(), plant, tuningGain(t, 0.1, 5), wc, nil, nil); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("wc=%v: err = %v, want ErrInvalidArgument", wc, err)
		}
	}
}
