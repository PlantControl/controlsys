package controlsys

import (
	"fmt"
	"math"
)

type TuningGoalType int

const (
	TuningGoalTracking TuningGoalType = iota
	TuningGoalRejection
	TuningGoalSensitivity
	TuningGoalWeightedGain
	TuningGoalLoopShape
	TuningGoalMargin
	TuningGoalPole
	TuningGoalOvershoot
)

type TuningGoalSpec struct {
	Name          string
	Type          TuningGoalType
	Max           float64
	Min           float64
	AnalysisPoint string
	Omega         []float64
	InputWeight   *System
	OutputWeight  *System
}

type TuningGoal struct {
	spec TuningGoalSpec
}

type TuningGoalResult struct {
	GoalName    string
	Pass        bool
	Value       float64
	Limit       float64
	Violation   float64
	Diagnostics map[string]float64
}

func NewTuningGoal(spec TuningGoalSpec) (TuningGoal, error) {
	if spec.Name == "" {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: name is empty: %w", ErrDimensionMismatch)
	}
	if spec.Type != TuningGoalPole && (spec.Max < 0 || spec.Min < 0) {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: negative bound: %w", ErrDimensionMismatch)
	}
	if spec.Type < TuningGoalTracking || spec.Type > TuningGoalOvershoot {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: unsupported goal type %d: %w", spec.Type, ErrDimensionMismatch)
	}
	if spec.Type == TuningGoalLoopShape && spec.Min > spec.Max {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: loop-shape minimum %g exceeds maximum %g: %w", spec.Min, spec.Max, ErrDimensionMismatch)
	}
	if spec.Omega != nil && !tuningGoalUsesFrequencyGrid(spec.Type) {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: goal type %d does not use a frequency grid: %w", spec.Type, ErrDimensionMismatch)
	}
	if err := validateTuningGoalFrequencyGrid(spec.Omega); err != nil {
		return TuningGoal{}, err
	}
	if (spec.InputWeight != nil || spec.OutputWeight != nil) && spec.Type != TuningGoalWeightedGain {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: weights require a weighted-gain goal: %w", ErrDimensionMismatch)
	}
	spec.Omega = copyFloatSlice(spec.Omega)
	if spec.InputWeight != nil {
		spec.InputWeight = spec.InputWeight.Copy()
	}
	if spec.OutputWeight != nil {
		spec.OutputWeight = spec.OutputWeight.Copy()
	}
	return TuningGoal{spec: spec}, nil
}

// NewTrackingGoal returns a tracking goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewTrackingGoal(name string, maxError float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalTracking, Max: maxError})
}

// NewRejectionGoal returns a rejection goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewRejectionGoal(name string, maxGain float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalRejection, Max: maxGain})
}

// NewSensitivityGoal returns a sensitivity goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewSensitivityGoal(name string, maxGain float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalSensitivity, Max: maxGain})
}

// NewWeightedGainGoal returns a weighted-gain goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewWeightedGainGoal(name string, maxGain float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalWeightedGain, Max: maxGain})
}

// NewLoopShapeGoal returns a loop-shape goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewLoopShapeGoal(name string, minGain, maxGain float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalLoopShape, Min: minGain, Max: maxGain})
}

// NewMarginGoal returns a stability-margin goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewMarginGoal(name string, minGainMarginDB, minPhaseMarginDeg float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalMargin, Min: minGainMarginDB, Max: minPhaseMarginDeg})
}

// NewPoleGoal returns a pole-location goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewPoleGoal(name string, maxRealPart float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalPole, Max: maxRealPart})
}

// NewOvershootGoal returns an overshoot goal and panics if the goal is invalid.
// Use NewTuningGoal for error-returning construction.
func NewOvershootGoal(name string, maxPercent float64) TuningGoal {
	return mustTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalOvershoot, Max: maxPercent})
}

func mustTuningGoal(spec TuningGoalSpec) TuningGoal {
	goal, err := NewTuningGoal(spec)
	if err != nil {
		panic(err)
	}
	return goal
}

func (g TuningGoal) Name() string {
	return g.spec.Name
}

func (g TuningGoal) Evaluate(model any) (TuningGoalResult, error) {
	sys, err := tuningGoalSystem(model, g.spec)
	if err != nil {
		return TuningGoalResult{}, err
	}
	return g.evaluateSystem(sys)
}

func (g TuningGoal) evaluateSystem(sys *System) (TuningGoalResult, error) {
	switch g.spec.Type {
	case TuningGoalTracking:
		return g.evaluateTracking(sys)
	case TuningGoalRejection, TuningGoalSensitivity, TuningGoalWeightedGain:
		return g.evaluateMaxGain(sys)
	case TuningGoalLoopShape:
		return g.evaluateLoopShape(sys)
	case TuningGoalMargin:
		return g.evaluateMargin(sys)
	case TuningGoalPole:
		return g.evaluatePole(sys)
	case TuningGoalOvershoot:
		return g.evaluateOvershoot(sys)
	default:
		return TuningGoalResult{}, fmt.Errorf("TuningGoal.Evaluate: unsupported goal type: %w", ErrDimensionMismatch)
	}
}

func tuningGoalSystem(model any, spec TuningGoalSpec) (*System, error) {
	switch v := model.(type) {
	case *System:
		return v.Copy(), nil
	case *GeneralizedModel:
		return v.CurrentSystem()
	case *GeneralizedClosedLoop:
		point := spec.AnalysisPoint
		if point == "" {
			point = v.primaryAnalysisPointName()
		}
		switch tuningGoalResponseForType(spec.Type) {
		case tuningGoalSensitivityResponse:
			return v.Sensitivity(point)
		case tuningGoalOpenLoopResponse:
			return v.OpenLoop(point)
		default:
			return v.ClosedLoop(point)
		}
	default:
		return nil, fmt.Errorf("TuningGoal.Evaluate: unsupported model %T: %w", model, ErrDimensionMismatch)
	}
}

type tuningGoalResponse uint8

const (
	tuningGoalClosedLoopResponse tuningGoalResponse = iota
	tuningGoalSensitivityResponse
	tuningGoalOpenLoopResponse
)

func tuningGoalResponseForType(goalType TuningGoalType) tuningGoalResponse {
	switch goalType {
	case TuningGoalRejection, TuningGoalSensitivity:
		return tuningGoalSensitivityResponse
	case TuningGoalLoopShape, TuningGoalMargin:
		return tuningGoalOpenLoopResponse
	default:
		return tuningGoalClosedLoopResponse
	}
}

func tuningGoalUsesFrequencyGrid(goalType TuningGoalType) bool {
	switch goalType {
	case TuningGoalRejection, TuningGoalSensitivity, TuningGoalWeightedGain, TuningGoalLoopShape:
		return true
	default:
		return false
	}
}

func firstAnalysisPointName(points map[string]AnalysisPoint) string {
	for name := range points {
		return name
	}
	return ""
}

func (g TuningGoal) evaluateTracking(sys *System) (TuningGoalResult, error) {
	dc, err := sys.DCGain()
	if err != nil {
		return TuningGoalResult{}, err
	}
	errVal := maxDCErrorFromOne(dc)
	return g.scalarResult(errVal, g.spec.Max, errVal <= g.spec.Max, map[string]float64{"dc_error": errVal}), nil
}

func (g TuningGoal) evaluateMaxGain(sys *System) (TuningGoalResult, error) {
	value, err := maxFrequencyGain(sys, g.spec.Omega, g.spec.OutputWeight, g.spec.InputWeight)
	if err != nil {
		return TuningGoalResult{}, err
	}
	return g.scalarResult(value, g.spec.Max, value <= g.spec.Max, map[string]float64{"max_gain": value}), nil
}

func (g TuningGoal) evaluateLoopShape(sys *System) (TuningGoalResult, error) {
	minimum, maximum, err := frequencyGainRange(sys, g.spec.Omega, nil, nil)
	if err != nil {
		return TuningGoalResult{}, err
	}
	pass := minimum >= g.spec.Min && maximum <= g.spec.Max
	result := g.scalarResult(maximum, g.spec.Max, pass, map[string]float64{
		"sampled_min_gain":  minimum,
		"sampled_max_gain":  maximum,
		"required_min_gain": g.spec.Min,
	})
	result.Violation = math.Max(normalizedLowerViolation(minimum, g.spec.Min), normalizedUpperViolation(maximum, g.spec.Max))
	return result, nil
}

func (g TuningGoal) evaluateMargin(sys *System) (TuningGoalResult, error) {
	margin, err := Margin(sys)
	if err != nil {
		return TuningGoalResult{}, err
	}
	pass := margin.GainMargin >= g.spec.Min && margin.PhaseMargin >= g.spec.Max
	diag := map[string]float64{"gain_margin_db": margin.GainMargin, "phase_margin_deg": margin.PhaseMargin}
	result := g.scalarResult(math.Min(margin.GainMargin, margin.PhaseMargin), g.spec.Max, pass, diag)
	result.Violation = math.Max(normalizedLowerViolation(margin.GainMargin, g.spec.Min), normalizedLowerViolation(margin.PhaseMargin, g.spec.Max))
	return result, nil
}

func (g TuningGoal) evaluatePole(sys *System) (TuningGoalResult, error) {
	poles, err := sys.Poles()
	if err != nil {
		return TuningGoalResult{}, err
	}
	maxReal := math.Inf(-1)
	for _, p := range poles {
		if real(p) > maxReal {
			maxReal = real(p)
		}
	}
	if len(poles) == 0 {
		maxReal = math.Inf(-1)
	}
	return g.scalarResult(maxReal, g.spec.Max, maxReal <= g.spec.Max, map[string]float64{"max_real_pole": maxReal}), nil
}

func (g TuningGoal) evaluateOvershoot(sys *System) (TuningGoalResult, error) {
	info, err := StepInfoForSystem(sys, 0, nil)
	if err != nil {
		return TuningGoalResult{}, err
	}
	maxOvershoot := 0.0
	for _, metric := range info.Metrics {
		if metric.Overshoot > maxOvershoot {
			maxOvershoot = metric.Overshoot
		}
	}
	return g.scalarResult(maxOvershoot, g.spec.Max, maxOvershoot <= g.spec.Max, map[string]float64{"overshoot_percent": maxOvershoot}), nil
}

func (g TuningGoal) scalarResult(value, limit float64, pass bool, diag map[string]float64) TuningGoalResult {
	return TuningGoalResult{
		GoalName:    g.spec.Name,
		Pass:        pass,
		Value:       value,
		Limit:       limit,
		Violation:   normalizedUpperViolation(value, limit),
		Diagnostics: diag,
	}
}

func normalizedUpperViolation(value, limit float64) float64 {
	if value <= limit {
		return 0
	}
	if limit == 0 {
		return value - limit
	}
	return (value - limit) / math.Abs(limit)
}

func normalizedLowerViolation(value, limit float64) float64 {
	if value >= limit {
		return 0
	}
	if limit == 0 {
		return limit - value
	}
	return (limit - value) / math.Abs(limit)
}

func maxDCErrorFromOne(dc interface {
	Dims() (int, int)
	At(int, int) float64
}) float64 {
	r, c := dc.Dims()
	maxErr := 0.0
	for i := range r {
		for j := range c {
			want := 0.0
			if i == j {
				want = 1
			}
			if err := math.Abs(dc.At(i, j) - want); err > maxErr {
				maxErr = err
			}
		}
	}
	return maxErr
}

func validateTuningGoalFrequencyGrid(omega []float64) error {
	if omega != nil && len(omega) == 0 {
		return fmt.Errorf("NewTuningGoal: frequency grid is empty: %w", ErrDimensionMismatch)
	}
	for i, w := range omega {
		if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
			return fmt.Errorf("NewTuningGoal: invalid frequency omega[%d]=%g: %w", i, w, ErrDimensionMismatch)
		}
		if i > 0 && w <= omega[i-1] {
			return fmt.Errorf("NewTuningGoal: frequencies must be strictly increasing: %w", ErrDimensionMismatch)
		}
	}
	return nil
}

func maxFrequencyGain(sys *System, omega []float64, outputWeight, inputWeight *System) (float64, error) {
	_, maximum, err := frequencyGainRange(sys, omega, outputWeight, inputWeight)
	return maximum, err
}

func frequencyGainRange(sys *System, omega []float64, outputWeight, inputWeight *System) (float64, float64, error) {
	if omega == nil {
		omega = logspace(-2, 2, 80)
	}
	if len(omega) == 0 {
		return 0, 0, fmt.Errorf("frequency gain: empty grid: %w", ErrDimensionMismatch)
	}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return 0, 0, err
	}
	var outputResponse, inputResponse *FreqResponseMatrix
	if outputWeight != nil {
		if err := domainMatch(sys, outputWeight); err != nil {
			return 0, 0, fmt.Errorf("output weight: %w", err)
		}
		outputResponse, err = outputWeight.FreqResponse(omega)
		if err != nil {
			return 0, 0, err
		}
	}
	if inputWeight != nil {
		if err := domainMatch(sys, inputWeight); err != nil {
			return 0, 0, fmt.Errorf("input weight: %w", err)
		}
		inputResponse, err = inputWeight.FreqResponse(omega)
		if err != nil {
			return 0, 0, err
		}
	}
	maxGain := 0.0
	minGain := math.Inf(1)
	var outputProduct, inputProduct []complex128
	weightedRows := resp.P
	weightedCols := resp.M
	if outputResponse != nil {
		outputProduct = make([]complex128, outputResponse.P*resp.M)
		weightedRows = outputResponse.P
	}
	if inputResponse != nil {
		inputProduct = make([]complex128, weightedRows*inputResponse.M)
		weightedCols = inputResponse.M
	}
	var singularValues *complexSVDWorkspace
	if weightedRows > 1 && weightedCols > 1 && (weightedRows != 2 || weightedCols != 2) {
		singularValues = newComplexSVDWorkspace(weightedRows, weightedCols)
	}
	for k := range omega {
		gain := complexResponseAt(resp, k)
		if outputResponse != nil {
			gain, err = multiplyComplexMatricesInto(outputProduct, complexResponseAt(outputResponse, k), gain)
			if err != nil {
				return 0, 0, fmt.Errorf("output weight: %w", err)
			}
		}
		if inputResponse != nil {
			gain, err = multiplyComplexMatricesInto(inputProduct, gain, complexResponseAt(inputResponse, k))
			if err != nil {
				return 0, 0, fmt.Errorf("input weight: %w", err)
			}
		}
		sigma, err := singularValues.maximumFromFlat(gain.data, 0, gain.rows, gain.cols)
		if err != nil {
			return 0, 0, err
		}
		if sigma > maxGain {
			maxGain = sigma
		}
		if sigma < minGain {
			minGain = sigma
		}
	}
	return minGain, maxGain, nil
}

type complexMatrix struct {
	rows int
	cols int
	data []complex128
}

func complexResponseAt(response *FreqResponseMatrix, frequency int) complexMatrix {
	blockSize := response.P * response.M
	base := frequency * blockSize
	return complexMatrix{rows: response.P, cols: response.M, data: response.Data[base : base+blockSize]}
}

func multiplyComplexMatricesInto(dst []complex128, a, b complexMatrix) (complexMatrix, error) {
	if a.cols != b.rows {
		return complexMatrix{}, fmt.Errorf("matrix dimensions %dx%d and %dx%d: %w", a.rows, a.cols, b.rows, b.cols, ErrDimensionMismatch)
	}
	if len(dst) != a.rows*b.cols {
		dst = make([]complex128, a.rows*b.cols)
	}
	result := complexMatrix{rows: a.rows, cols: b.cols, data: dst}
	for i := range result.data {
		result.data[i] = 0
	}
	for i := range a.rows {
		for k := range a.cols {
			aik := a.data[i*a.cols+k]
			for j := range b.cols {
				result.data[i*result.cols+j] += aik * b.data[k*b.cols+j]
			}
		}
	}
	return result, nil
}
