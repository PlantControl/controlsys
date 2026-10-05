package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
)

// TuningGoalType selects what a TuningGoal measures, mirroring MATLAB's
// TuningGoal classes.
type TuningGoalType int

const (
	// TuningGoalTracking bounds the DC tracking error max|T(0) - I| of the
	// closed loop by Max (TuningGoal.Tracking).
	TuningGoalTracking TuningGoalType = iota
	// TuningGoalRejection bounds the peak sensitivity gain by Max
	// (TuningGoal.Rejection).
	TuningGoalRejection
	// TuningGoalSensitivity bounds the peak sensitivity gain by Max
	// (TuningGoal.Sensitivity).
	TuningGoalSensitivity
	// TuningGoalWeightedGain bounds the peak weighted closed-loop gain
	// ‖OutputWeight·T·InputWeight‖ by Max (TuningGoal.WeightedGain).
	TuningGoalWeightedGain
	// TuningGoalLoopShape keeps the open-loop gain within [Min, Max] on Omega
	// (TuningGoal.LoopShape).
	TuningGoalLoopShape
	// TuningGoalMargin requires gain margin ≥ GainMarginDB and phase margin
	// ≥ PhaseMarginDeg of the open loop (TuningGoal.Margins).
	TuningGoalMargin
	// TuningGoalPole bounds the largest closed-loop pole real part by Max;
	// discrete poles z map to log|z|/Ts (TuningGoal.Poles decay rate).
	TuningGoalPole
	// TuningGoalOvershoot bounds the closed-loop step overshoot percentage by
	// Max (TuningGoal.Overshoot).
	TuningGoalOvershoot
	// TuningGoalCrossover places the open-loop gain crossover in [Min, Max]
	// rad/s: every singular value ≥ 1 at Min and ≤ 1 at Max (looptune's wc).
	TuningGoalCrossover
)

// String returns the goal type name.
func (t TuningGoalType) String() string {
	switch t {
	case TuningGoalTracking:
		return "tracking"
	case TuningGoalRejection:
		return "rejection"
	case TuningGoalSensitivity:
		return "sensitivity"
	case TuningGoalWeightedGain:
		return "weighted-gain"
	case TuningGoalLoopShape:
		return "loop-shape"
	case TuningGoalMargin:
		return "margin"
	case TuningGoalPole:
		return "pole"
	case TuningGoalOvershoot:
		return "overshoot"
	case TuningGoalCrossover:
		return "crossover"
	default:
		return fmt.Sprintf("TuningGoalType(%d)", int(t))
	}
}

// TuningGoalSpec describes a tuning goal; see TuningGoalType for which fields
// each type uses. Unused fields must be zero.
type TuningGoalSpec struct {
	Name string
	Type TuningGoalType
	// Max is the upper bound of the measured quantity (tracking error, gain,
	// pole real part, overshoot %, crossover band upper edge).
	Max float64
	// Min is the lower gain bound of a loop-shape goal or the crossover band
	// lower edge.
	Min float64
	// GainMarginDB and PhaseMarginDeg are the margin goal's requirements.
	GainMarginDB   float64
	PhaseMarginDeg float64
	// AnalysisPoint selects the loop opening; empty uses the model's primary
	// analysis point.
	AnalysisPoint string
	// Omega is an optional strictly increasing frequency grid (rad/s) for
	// gain goals; nil uses 80 points from 0.01 to 100.
	Omega        []float64
	InputWeight  *System
	OutputWeight *System
}

// TuningGoal is a validated tuning requirement, as MATLAB's TuningGoal
// objects. Construct it with NewTuningGoal or a typed constructor.
type TuningGoal struct {
	spec TuningGoalSpec
}

// TuningGoalResult is the evaluation of one goal. Value is the measured
// quantity in the goal's units and Limit its bound; for a margin goal both
// are dimensionless: Value = min(GM/GainMarginDB, PM/PhaseMarginDeg) with
// Limit 1, and for a crossover goal Value is the violation in decades with
// Limit 0. Violation is the normalized amount by which the goal is missed
// (0 when met).
type TuningGoalResult struct {
	GoalName    string
	Pass        bool
	Value       float64
	Limit       float64
	Violation   float64
	Diagnostics map[string]float64
}

// TuningGoalModel is a model a TuningGoal can be evaluated on: *System,
// *GeneralizedModel or *GeneralizedClosedLoop. For a closed loop the goal
// type selects the response (closed loop, sensitivity or open loop).
type TuningGoalModel interface {
	tuningGoalSystem(spec TuningGoalSpec) (*System, error)
}

// NewTuningGoal validates spec and returns the goal. A NaN or negative bound
// (other than a pole goal's Max), an empty name, an unknown type or a field
// the type does not use returns ErrInvalidArgument.
func NewTuningGoal(spec TuningGoalSpec) (TuningGoal, error) {
	if err := validateTuningGoalSpec(spec); err != nil {
		return TuningGoal{}, fmt.Errorf("NewTuningGoal: %w", err)
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

func validateTuningGoalSpec(spec TuningGoalSpec) error {
	if spec.Name == "" {
		return fmt.Errorf("name is empty: %w", ErrInvalidArgument)
	}
	if spec.Type < TuningGoalTracking || spec.Type > TuningGoalCrossover {
		return fmt.Errorf("unsupported goal type %v: %w", spec.Type, ErrInvalidArgument)
	}
	for _, v := range []struct {
		name string
		x    float64
	}{{"Max", spec.Max}, {"Min", spec.Min}, {"GainMarginDB", spec.GainMarginDB}, {"PhaseMarginDeg", spec.PhaseMarginDeg}} {
		if math.IsNaN(v.x) {
			return fmt.Errorf("%s is NaN: %w", v.name, ErrInvalidArgument)
		}
		if v.x < 0 && !(spec.Type == TuningGoalPole && v.name == "Max") {
			return fmt.Errorf("%s is %g, want >= 0: %w", v.name, v.x, ErrInvalidArgument)
		}
	}
	if spec.Type == TuningGoalMargin {
		if spec.Min != 0 || spec.Max != 0 {
			return fmt.Errorf("margin goal uses GainMarginDB and PhaseMarginDeg, not Min/Max: %w", ErrInvalidArgument)
		}
	} else if spec.GainMarginDB != 0 || spec.PhaseMarginDeg != 0 {
		return fmt.Errorf("%v goal does not use margin fields: %w", spec.Type, ErrInvalidArgument)
	}
	if spec.Type != TuningGoalLoopShape && spec.Type != TuningGoalCrossover && spec.Min != 0 {
		return fmt.Errorf("%v goal does not use Min: %w", spec.Type, ErrInvalidArgument)
	}
	if (spec.Type == TuningGoalLoopShape || spec.Type == TuningGoalCrossover) && spec.Min > spec.Max {
		return fmt.Errorf("%v minimum %g exceeds maximum %g: %w", spec.Type, spec.Min, spec.Max, ErrInvalidArgument)
	}
	if spec.Type == TuningGoalCrossover && !(spec.Min > 0) {
		return fmt.Errorf("crossover band lower edge %g must be positive: %w", spec.Min, ErrInvalidArgument)
	}
	if spec.Omega != nil && !tuningGoalUsesFrequencyGrid(spec.Type) {
		return fmt.Errorf("%v goal does not use a frequency grid: %w", spec.Type, ErrInvalidArgument)
	}
	if err := validateTuningGoalFrequencyGrid(spec.Omega); err != nil {
		return err
	}
	if (spec.InputWeight != nil || spec.OutputWeight != nil) && spec.Type != TuningGoalWeightedGain {
		return fmt.Errorf("weights require a weighted-gain goal: %w", ErrInvalidArgument)
	}
	return nil
}

// NewTrackingGoal returns a DC tracking-error goal, MATLAB
// TuningGoal.Tracking without channel selection or response time.
func NewTrackingGoal(name string, maxError float64) (TuningGoal, error) {
	return NewTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalTracking, Max: maxError})
}

// NewRejectionGoal returns a peak sensitivity-gain goal (TuningGoal.Rejection
// with a constant attenuation profile).
func NewRejectionGoal(name string, maxGain float64) (TuningGoal, error) {
	return NewTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalRejection, Max: maxGain})
}

// NewSensitivityGoal returns a peak sensitivity-gain goal
// (TuningGoal.Sensitivity with a constant profile).
func NewSensitivityGoal(name string, maxGain float64) (TuningGoal, error) {
	return NewTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalSensitivity, Max: maxGain})
}

// NewWeightedGainGoal returns a peak closed-loop gain goal
// (TuningGoal.WeightedGain without weights; set them through NewTuningGoal).
func NewWeightedGainGoal(name string, maxGain float64) (TuningGoal, error) {
	return NewTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalWeightedGain, Max: maxGain})
}

// NewLoopShapeGoal returns an open-loop gain band goal (TuningGoal.LoopShape
// with constant bounds).
func NewLoopShapeGoal(name string, minGain, maxGain float64) (TuningGoal, error) {
	return NewTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalLoopShape, Min: minGain, Max: maxGain})
}

// NewMarginGoal returns a stability-margin goal, MATLAB
// TuningGoal.Margins(location, gm, pm) at the primary analysis point.
func NewMarginGoal(name string, minGainMarginDB, minPhaseMarginDeg float64) (TuningGoal, error) {
	return NewTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalMargin, GainMarginDB: minGainMarginDB, PhaseMarginDeg: minPhaseMarginDeg})
}

// NewPoleGoal returns a closed-loop pole goal bounding the largest pole real
// part (TuningGoal.Poles with mindecay = -maxRealPart).
func NewPoleGoal(name string, maxRealPart float64) (TuningGoal, error) {
	return NewTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalPole, Max: maxRealPart})
}

// NewOvershootGoal returns a step-overshoot goal in percent
// (TuningGoal.Overshoot without channel selection).
func NewOvershootGoal(name string, maxPercent float64) (TuningGoal, error) {
	return NewTuningGoal(TuningGoalSpec{Name: name, Type: TuningGoalOvershoot, Max: maxPercent})
}

// Name returns the goal name.
func (g TuningGoal) Name() string {
	return g.spec.Name
}

// Evaluate measures the goal on model.
func (g TuningGoal) Evaluate(model TuningGoalModel) (TuningGoalResult, error) {
	if model == nil {
		return TuningGoalResult{}, fmt.Errorf("TuningGoal.Evaluate: model is nil: %w", ErrInvalidArgument)
	}
	sys, err := model.tuningGoalSystem(g.spec)
	if err != nil {
		return TuningGoalResult{}, fmt.Errorf("TuningGoal.Evaluate: %w", err)
	}
	res, err := g.evaluateSystem(sys)
	if err != nil {
		return TuningGoalResult{}, fmt.Errorf("TuningGoal.Evaluate: %w", err)
	}
	return res, nil
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
	case TuningGoalCrossover:
		return g.evaluateCrossover(sys)
	default:
		return TuningGoalResult{}, fmt.Errorf("unsupported goal type %v: %w", g.spec.Type, ErrInvalidArgument)
	}
}

func (sys *System) tuningGoalSystem(TuningGoalSpec) (*System, error) {
	if err := requireSystem("TuningGoal.Evaluate", sys); err != nil {
		return nil, err
	}
	return sys.Copy(), nil
}

func (g *GeneralizedModel) tuningGoalSystem(TuningGoalSpec) (*System, error) {
	return g.CurrentSystem()
}

func (g *GeneralizedClosedLoop) tuningGoalSystem(spec TuningGoalSpec) (*System, error) {
	return tuningGoalSystem(g, spec)
}

func tuningGoalSystem(v *GeneralizedClosedLoop, spec TuningGoalSpec) (*System, error) {
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
	case TuningGoalLoopShape, TuningGoalMargin, TuningGoalCrossover:
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
	first := ""
	for name := range points {
		if first == "" || name < first {
			first = name
		}
	}
	return first
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
	gm, pm := g.spec.GainMarginDB, g.spec.PhaseMarginDeg
	pass := margin.GainMargin >= gm && margin.PhaseMargin >= pm
	diag := map[string]float64{"gain_margin_db": margin.GainMargin, "phase_margin_deg": margin.PhaseMargin}
	ratio := math.Min(marginRatio(margin.GainMargin, gm), marginRatio(margin.PhaseMargin, pm))
	result := g.scalarResult(ratio, 1, pass, diag)
	result.Violation = math.Max(normalizedLowerViolation(margin.GainMargin, gm), normalizedLowerViolation(margin.PhaseMargin, pm))
	return result, nil
}

func marginRatio(value, required float64) float64 {
	if required == 0 {
		if value >= 0 {
			return math.Inf(1)
		}
		return math.Inf(-1)
	}
	return value / required
}

// evaluateCrossover measures, in decades, how far σ_min(L(j·Min)) falls
// below 1 plus how far σ_max(L(j·Max)) rises above 1.
func (g TuningGoal) evaluateCrossover(sys *System) (TuningGoalResult, error) {
	omega := []float64{g.spec.Min, g.spec.Max}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return TuningGoalResult{}, err
	}
	k := min(resp.P, resp.M)
	if k == 0 {
		return TuningGoalResult{}, fmt.Errorf("crossover goal needs a loop with inputs and outputs: %w", ErrDimensionMismatch)
	}
	ws := newComplexSVDWorkspace(resp.P, resp.M)
	sv := make([]float64, k)
	if err := ws.singularValuesFromFlat(sv, resp.Data, 0, resp.P, resp.M); err != nil {
		return TuningGoalResult{}, err
	}
	low := sv[k-1]
	high, err := ws.maximumFromFlat(resp.Data, resp.P*resp.M, resp.P, resp.M)
	if err != nil {
		return TuningGoalResult{}, err
	}
	violation := math.Max(0, -math.Log10(low)) + math.Max(0, math.Log10(high))
	result := g.scalarResult(violation, 0, violation == 0, map[string]float64{"min_gain_at_wcmin": low, "max_gain_at_wcmax": high})
	result.Violation = violation
	return result, nil
}

func (g TuningGoal) evaluatePole(sys *System) (TuningGoalResult, error) {
	poles, err := sys.Poles()
	if err != nil {
		return TuningGoalResult{}, err
	}
	maxReal := math.Inf(-1)
	for _, p := range poles {
		r := real(p)
		if sys.IsDiscrete() {
			r = math.Log(cmplx.Abs(p)) / sys.Dt
		}
		maxReal = math.Max(maxReal, r)
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
		return fmt.Errorf("frequency grid is empty: %w", ErrInvalidArgument)
	}
	for i, w := range omega {
		if math.IsNaN(w) || math.IsInf(w, 0) || w < 0 {
			return fmt.Errorf("invalid frequency omega[%d]=%g: %w", i, w, ErrInvalidArgument)
		}
		if i > 0 && w <= omega[i-1] {
			return fmt.Errorf("frequencies must be strictly increasing: %w", ErrInvalidArgument)
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
