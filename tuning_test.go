package controlsys

import (
	"errors"
	"testing"
)

func TestSystuneTunesSISOTunableGain(t *testing.T) {
	plant := makeSISO(-2, 1, 1, 0)
	k, _ := NewTunableReal("K", 0.1, TunableBounds{Lower: 0.1, Upper: 5})
	controller := NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0)
	closed, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		t.Fatal(err)
	}

	result, err := Systune(closed, []TuningGoal{NewTrackingGoal("track", 0.4)}, &SystuneOptions{GridPoints: 9})
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
	k1, _ := NewTunableReal("K1", 0.1, TunableBounds{Lower: 0.1, Upper: 2})
	k2, _ := NewTunableReal("K2", 0.1, TunableBounds{Lower: 0.1, Upper: 2})
	controller := NewTunableGain("Kblock", [][]*TunableReal{{k1, fixedReal(t, "z12_limit", 0)}, {fixedReal(t, "z21_limit", 0), k2}}, 0)
	closed, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		t.Fatal(err)
	}
	_, err = GridTune(closed, []TuningGoal{NewWeightedGainGoal("bounded", 10)}, &SystuneOptions{GridPoints: 5, MaxEvaluations: 24})
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("GridTune error = %v, want ErrDimensionMismatch", err)
	}
}

func TestSystuneTunesSmallMIMOTunableGain(t *testing.T) {
	plant := benchSysNonSym(2, 2, 2)
	k1, _ := NewTunableReal("K1", 0.1, TunableBounds{Lower: 0.1, Upper: 2})
	k2, _ := NewTunableReal("K2", 0.1, TunableBounds{Lower: 0.1, Upper: 2})
	controller := NewTunableGain("Kblock", [][]*TunableReal{{k1, fixedReal(t, "z12", 0)}, {fixedReal(t, "z21", 0), k2}}, 0)
	closed, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		t.Fatal(err)
	}

	result, err := Looptune(closed, []TuningGoal{NewWeightedGainGoal("bounded", 10)}, &SystuneOptions{GridPoints: 5})
	if err != nil {
		t.Fatalf("Looptune: %v", err)
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
	k, _ := NewTunableReal("K", 0.1, TunableBounds{Lower: 0.1, Upper: 5})
	controller := wrappedTunableGain{gain: NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0)}
	closed, err := NewGeneralizedClosedLoop("loop", plant, controller, "u")
	if err != nil {
		t.Fatal(err)
	}

	result, err := Systune(closed, []TuningGoal{NewTrackingGoal("track", 0.4)}, &SystuneOptions{GridPoints: 9})
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
	if _, err := Systune(closed, []TuningGoal{NewWeightedGainGoal("gain", 1)}, nil); err == nil {
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
