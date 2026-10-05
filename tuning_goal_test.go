package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestTuningGoalsEvaluatePassFailFamilies(t *testing.T) {
	sys := makeSISO(-2, 2, 1, 0)
	goals := []TuningGoal{
		NewTrackingGoal("track", 1.2),
		NewRejectionGoal("reject", 1.2),
		NewSensitivityGoal("sens", 1.2),
		NewWeightedGainGoal("gain", 1.2),
		NewLoopShapeGoal("loop", 0.5, 2.0),
		NewMarginGoal("margin", 0, 0),
		NewPoleGoal("poles", 0),
		NewOvershootGoal("overshoot", 5),
	}
	for _, goal := range goals {
		result, err := goal.Evaluate(sys)
		if err != nil {
			t.Fatalf("%s Evaluate: %v", goal.Name(), err)
		}
		if result.GoalName != goal.Name() || len(result.Diagnostics) == 0 {
			t.Fatalf("%s result missing diagnostics: %#v", goal.Name(), result)
		}
	}
}

func TestTuningGoalsKnownFailuresAndGeneralizedCurrentValue(t *testing.T) {
	sys := makeSISO(-1, 1, 2, 0)
	failGoal := NewWeightedGainGoal("too_small", 0.5)
	res, err := failGoal.Evaluate(sys)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res.Pass {
		t.Fatalf("expected weighted gain failure, got %#v", res)
	}

	k, _ := NewTunableReal("K", 0.25, TunableBounds{Lower: 0, Upper: 1})
	gm, err := NewGeneralizedModel("gain", NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0))
	if err != nil {
		t.Fatal(err)
	}
	passGoal := NewWeightedGainGoal("small_gain", 0.5)
	gres, err := passGoal.Evaluate(gm)
	if err != nil {
		t.Fatalf("Evaluate generalized: %v", err)
	}
	if !gres.Pass {
		t.Fatalf("expected generalized gain pass, got %#v", gres)
	}
}

func TestTuningGoalValidation(t *testing.T) {
	if _, err := NewTuningGoal(TuningGoalSpec{Name: "", Type: TuningGoalWeightedGain, Max: 1}); err == nil {
		t.Fatal("empty goal name should fail")
	}
	if _, err := NewTuningGoal(TuningGoalSpec{Name: "bad", Type: TuningGoalWeightedGain, Max: -1}); err == nil {
		t.Fatal("negative max should fail")
	}
	if _, err := NewTuningGoal(TuningGoalSpec{Name: "pole_spec", Type: TuningGoalPole, Min: -2, Max: -0.5}); err != nil {
		t.Fatalf("negative pole bounds should pass: %v", err)
	}
	goal := NewPoleGoal("stable_fast", -0.5)
	pass, err := goal.Evaluate(makeSISO(-1, 1, 1, 0))
	if err != nil {
		t.Fatalf("Evaluate stable pole goal: %v", err)
	}
	if !pass.Pass {
		t.Fatalf("expected pole goal pass, got %#v", pass)
	}
	fail, err := goal.Evaluate(makeSISO(-0.2, 1, 1, 0))
	if err != nil {
		t.Fatalf("Evaluate slow pole goal: %v", err)
	}
	if fail.Pass {
		t.Fatalf("expected pole goal failure, got %#v", fail)
	}
}

func TestTuningGoalUsesMaximumSingularValueForMIMO(t *testing.T) {
	sys, err := NewGain(mat.NewDense(2, 2, []float64{1, 1, 1, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}
	result, err := NewWeightedGainGoal("sigma_max", 1.5).Evaluate(sys)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if result.Pass || math.Abs(result.Value-2) > 1e-12 {
		t.Fatalf("result = %#v, want sigma_max=2 failure", result)
	}
}

func TestTuningGoalHonorsFrequencyGridAndDynamicWeights(t *testing.T) {
	lowpass := makeSISO(-1, 1, 1, 0)
	bandGoal, err := NewTuningGoal(TuningGoalSpec{
		Name:  "high_frequency",
		Type:  TuningGoalWeightedGain,
		Max:   0.1,
		Omega: []float64{100},
	})
	if err != nil {
		t.Fatal(err)
	}
	bandResult, err := bandGoal.Evaluate(lowpass)
	if err != nil {
		t.Fatalf("band Evaluate: %v", err)
	}
	if !bandResult.Pass {
		t.Fatalf("high-frequency result = %#v, want pass", bandResult)
	}

	weight, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	weightedGoal, err := NewTuningGoal(TuningGoalSpec{
		Name:         "weighted",
		Type:         TuningGoalWeightedGain,
		Max:          1.5,
		Omega:        []float64{0},
		OutputWeight: weight,
	})
	if err != nil {
		t.Fatal(err)
	}
	weightedResult, err := weightedGoal.Evaluate(lowpass)
	if err != nil {
		t.Fatalf("weighted Evaluate: %v", err)
	}
	if weightedResult.Pass || math.Abs(weightedResult.Value-2) > 1e-12 {
		t.Fatalf("weighted result = %#v, want gain=2 failure", weightedResult)
	}
}

func TestLoopShapeChecksBothSampledEnvelopeBounds(t *testing.T) {
	goal, err := NewTuningGoal(TuningGoalSpec{
		Name:  "envelope",
		Type:  TuningGoalLoopShape,
		Min:   0.5,
		Max:   2,
		Omega: []float64{0, 100},
	})
	if err != nil {
		t.Fatal(err)
	}
	result, err := goal.Evaluate(makeSISO(-1, 1, 1, 0))
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if result.Pass || result.Diagnostics["sampled_min_gain"] >= 0.5 || result.Violation <= 0 {
		t.Fatalf("result = %#v, want lower-envelope violation", result)
	}
}

func TestTuningGoalRoutesGeneralizedLoopResponses(t *testing.T) {
	plant, _ := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	controller, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	loop, err := NewGeneralizedClosedLoop("loop", plant, fixedBlockT(t, controller), "output")
	if err != nil {
		t.Fatal(err)
	}
	sensitivity, err := NewSensitivityGoal("sensitivity", 0.4).Evaluate(loop)
	if err != nil {
		t.Fatalf("sensitivity Evaluate: %v", err)
	}
	closedLoop, err := NewWeightedGainGoal("closed_loop", 0.4).Evaluate(loop)
	if err != nil {
		t.Fatalf("closed-loop Evaluate: %v", err)
	}
	if !sensitivity.Pass || closedLoop.Pass {
		t.Fatalf("sensitivity=%#v closed-loop=%#v, want distinct routed responses", sensitivity, closedLoop)
	}
}

func TestTuningGoalTargetsNamedRectangularLoopBreak(t *testing.T) {
	plant, _ := NewGain(mat.NewDense(1, 2, []float64{1, 2}), 0)
	controller, _ := NewGain(mat.NewDense(2, 1, []float64{3, 4}), 0)
	loop, err := NewGeneralizedClosedLoop("loop", plant, fixedBlockT(t, controller), "plant_output")
	if err != nil {
		t.Fatal(err)
	}
	if err := loop.InsertAnalysisPoint("plant_input", AnalysisPointPlantInput); err != nil {
		t.Fatal(err)
	}
	outputGoal, err := NewTuningGoal(TuningGoalSpec{
		Name:          "output_sensitivity",
		Type:          TuningGoalSensitivity,
		Max:           0.2,
		AnalysisPoint: "plant_output",
		Omega:         []float64{0},
	})
	if err != nil {
		t.Fatal(err)
	}
	inputGoal, err := NewTuningGoal(TuningGoalSpec{
		Name:          "input_sensitivity",
		Type:          TuningGoalSensitivity,
		Max:           0.2,
		AnalysisPoint: "plant_input",
		Omega:         []float64{0},
	})
	if err != nil {
		t.Fatal(err)
	}
	outputResult, err := outputGoal.Evaluate(loop)
	if err != nil {
		t.Fatalf("output Evaluate: %v", err)
	}
	inputResult, err := inputGoal.Evaluate(loop)
	if err != nil {
		t.Fatalf("input Evaluate: %v", err)
	}
	if !outputResult.Pass || inputResult.Pass {
		t.Fatalf("output=%#v input=%#v, want distinct point constraints", outputResult, inputResult)
	}
}

func TestTuningGoalRejectsInvalidFrequencyAndWeightDimensions(t *testing.T) {
	if _, err := NewTuningGoal(TuningGoalSpec{
		Name:  "grid",
		Type:  TuningGoalWeightedGain,
		Max:   1,
		Omega: []float64{1, 1},
	}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("frequency validation error = %v", err)
	}
	if _, err := NewTuningGoal(TuningGoalSpec{
		Name:  "poles",
		Type:  TuningGoalPole,
		Max:   -0.1,
		Omega: []float64{1},
	}); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("unused frequency validation error = %v", err)
	}
	badWeight, _ := NewGain(mat.NewDense(2, 2, nil), 0)
	goal, err := NewTuningGoal(TuningGoalSpec{
		Name:         "weight",
		Type:         TuningGoalWeightedGain,
		Max:          1,
		OutputWeight: badWeight,
	})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := goal.Evaluate(makeSISO(-1, 1, 1, 0)); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("weight dimension error = %v, want ErrDimensionMismatch", err)
	}
}
