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
		mustOK(NewTrackingGoal("track", 1.2)),
		mustOK(NewRejectionGoal("reject", 1.2)),
		mustOK(NewSensitivityGoal("sens", 1.2)),
		mustOK(NewWeightedGainGoal("gain", 1.2)),
		mustOK(NewLoopShapeGoal("loop", 0.5, 2.0)),
		mustOK(NewMarginGoal("margin", 0, 0)),
		mustOK(NewPoleGoal("poles", 0)),
		mustOK(NewOvershootGoal("overshoot", 5)),
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
	failGoal := mustOK(NewWeightedGainGoal("too_small", 0.5))
	res, err := failGoal.Evaluate(sys)
	if err != nil {
		t.Fatalf("Evaluate: %v", err)
	}
	if res.Pass {
		t.Fatalf("expected weighted gain failure, got %#v", res)
	}

	k, _ := newBoundedReal("K", 0.25, 0, 1)
	gm, err := NewGeneralizedModel("gain", mustOK(NewTunableGain("Kblock", [][]*TunableReal{{k}}, 0)))
	if err != nil {
		t.Fatal(err)
	}
	passGoal := mustOK(NewWeightedGainGoal("small_gain", 0.5))
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
	if _, err := NewTuningGoal(TuningGoalSpec{Name: "pole_spec", Type: TuningGoalPole, Max: -0.5}); err != nil {
		t.Fatalf("negative pole bound should pass: %v", err)
	}
	if _, err := NewTuningGoal(TuningGoalSpec{Name: "pole_spec", Type: TuningGoalPole, Min: -2, Max: -0.5}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("unused Min on a pole goal: err = %v, want ErrInvalidArgument", err)
	}
	goal := mustOK(NewPoleGoal("stable_fast", -0.5))
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
	result, err := mustOK(NewWeightedGainGoal("sigma_max", 1.5)).Evaluate(sys)
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
	loop, err := NewGeneralizedClosedLoop("loop", plant, controller, "output")
	if err != nil {
		t.Fatal(err)
	}
	sensitivity, err := mustOK(NewSensitivityGoal("sensitivity", 0.4)).Evaluate(loop)
	if err != nil {
		t.Fatalf("sensitivity Evaluate: %v", err)
	}
	closedLoop, err := mustOK(NewWeightedGainGoal("closed_loop", 0.4)).Evaluate(loop)
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
	loop, err := NewGeneralizedClosedLoop("loop", plant, controller, "plant_output")
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
	}); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("frequency validation error = %v", err)
	}
	if _, err := NewTuningGoal(TuningGoalSpec{
		Name:  "poles",
		Type:  TuningGoalPole,
		Max:   -0.1,
		Omega: []float64{1},
	}); !errors.Is(err, ErrInvalidArgument) {
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

func TestTuningGoalSpecRejectsNaNAndMisusedFields(t *testing.T) {
	for _, spec := range []TuningGoalSpec{
		{Name: "x", Type: TuningGoalTracking, Max: math.NaN()},
		{Name: "x", Type: TuningGoalLoopShape, Min: math.NaN(), Max: 1},
		{Name: "m", Type: TuningGoalMargin, Min: 6, Max: 45},
		{Name: "m", Type: TuningGoalMargin, GainMarginDB: math.NaN()},
		{Name: "w", Type: TuningGoalWeightedGain, Max: 1, PhaseMarginDeg: 30},
		{Name: "c", Type: TuningGoalCrossover, Min: 0, Max: 1},
		{Name: "t", Type: TuningGoalType(99)},
	} {
		if _, err := NewTuningGoal(spec); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("%+v: err = %v, want ErrInvalidArgument", spec, err)
		}
	}
	if _, err := NewTrackingGoal("", 1); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("NewTrackingGoal empty name: err = %v, want ErrInvalidArgument", err)
	}
	if got := TuningGoalMargin.String(); got != "margin" {
		t.Errorf("String = %q", got)
	}
}

func TestTuningGoalMarginUsesDedicatedFields(t *testing.T) {
	// L = 2/(s+1)^3: GM = 20·log10(4) dB at w = sqrt(3).
	loop, err := New(mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1}), mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{2, 0, 0}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	m, err := Margin(loop)
	if err != nil {
		t.Fatal(err)
	}
	gm := 20 * math.Log10(4)
	if math.Abs(m.GainMargin-gm) > 1e-9 {
		t.Fatalf("oracle GM %g, Margin %g", gm, m.GainMargin)
	}
	goal := mustOK(NewMarginGoal("m", 6, 30))
	res, err := goal.Evaluate(loop)
	if err != nil {
		t.Fatal(err)
	}
	want := math.Min(gm/6, m.PhaseMargin/30)
	if !res.Pass || res.Limit != 1 || math.Abs(res.Value-want) > 1e-12 {
		t.Errorf("result = %+v, want pass, Value %g, Limit 1", res, want)
	}
	if res.Diagnostics["gain_margin_db"] != m.GainMargin || res.Diagnostics["phase_margin_deg"] != m.PhaseMargin {
		t.Errorf("diagnostics = %v", res.Diagnostics)
	}
}

func TestTuningGoalPoleDiscreteDecayRate(t *testing.T) {
	goal := mustOK(NewPoleGoal("decay", -5))
	disc, err := New(mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	res, err := goal.Evaluate(disc)
	if err != nil {
		t.Fatal(err)
	}
	if want := math.Log(0.5) / 0.1; math.Abs(res.Value-want) > 1e-12 || !res.Pass {
		t.Errorf("discrete pole value = %g pass=%v, want %g, true", res.Value, res.Pass, want)
	}
	res, err = goal.Evaluate(makeSISO(-6.93, 1, 1, 0))
	if err != nil {
		t.Fatal(err)
	}
	if res.Value != -6.93 || !res.Pass {
		t.Errorf("continuous pole value = %g pass=%v", res.Value, res.Pass)
	}
	if _, err := goal.Evaluate(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil model: err = %v, want ErrInvalidArgument", err)
	}
}
