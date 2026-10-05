package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
)

// TuningGoalType names the MATLAB TuningGoal class a TuningGoal mirrors.
type TuningGoalType int

const (
	// TuningGoalTracking is TuningGoal.Tracking: the relative tracking error
	// T − I from input to output stays below a maximum error profile.
	TuningGoalTracking TuningGoalType = iota
	// TuningGoalGain is TuningGoal.Gain: the gain from input to output stays
	// below a constant.
	TuningGoalGain
	// TuningGoalWeightedGain is TuningGoal.WeightedGain: ‖WL·T·WR‖∞ < 1.
	TuningGoalWeightedGain
	// TuningGoalRejection is TuningGoal.Rejection: the disturbance
	// attenuation 1/|S| at a location exceeds an attenuation profile.
	TuningGoalRejection
	// TuningGoalSensitivity is TuningGoal.Sensitivity: the sensitivity S at a
	// location stays below a profile.
	TuningGoalSensitivity
	// TuningGoalLoopShape is TuningGoal.LoopShape: the open loop L at a
	// location follows a target gain profile within a crossover tolerance.
	TuningGoalLoopShape
	// TuningGoalMargins is TuningGoal.Margins: disk-based gain and phase
	// margins of the loop at a location.
	TuningGoalMargins
	// TuningGoalPoles is TuningGoal.Poles: closed-loop poles meet a minimum
	// decay rate, minimum damping and maximum natural frequency.
	TuningGoalPoles
	// TuningGoalOvershoot is TuningGoal.Overshoot: the step-response
	// overshoot from input to output stays below a percentage.
	TuningGoalOvershoot
)

// String returns the MATLAB class name without the TuningGoal prefix.
func (t TuningGoalType) String() string {
	switch t {
	case TuningGoalTracking:
		return "Tracking"
	case TuningGoalGain:
		return "Gain"
	case TuningGoalWeightedGain:
		return "WeightedGain"
	case TuningGoalRejection:
		return "Rejection"
	case TuningGoalSensitivity:
		return "Sensitivity"
	case TuningGoalLoopShape:
		return "LoopShape"
	case TuningGoalMargins:
		return "Margins"
	case TuningGoalPoles:
		return "Poles"
	case TuningGoalOvershoot:
		return "Overshoot"
	default:
		return fmt.Sprintf("TuningGoalType(%d)", int(t))
	}
}

// TuningGoal is a tuning requirement, as MATLAB's TuningGoal objects;
// construct it with the New*Goal function named after the MATLAB class.
//
// Signal names map onto a GeneralizedClosedLoop, which plays the role of the
// MATLAB genss model T = feedback(G*C, 1): its inputs are the controller's
// InputName and its outputs the plant's OutputName at the primary analysis
// point. An inputname or outputname is either an analysis point name, which
// selects every channel of the loop closed at that point, or a channel name
// of that closed loop; a vector signal name such as "y" selects every channel
// "y(1)", "y(2)", ... as in MATLAB. When
// one of them is an analysis point the loop is closed there, otherwise at the
// primary point; two different analysis points return ErrOptionUnsupported.
// A location must be an analysis point name. Evaluated on a *System or
// GeneralizedModel, the model is the response the goal measures (closed loop
// for input/output goals, sensitivity for Sensitivity and Rejection, open
// loop for LoopShape and Margins) and inputname and outputname must be its
// channel names. An unknown name returns ErrSignalNotFound at evaluation.
type TuningGoal struct {
	typ                   TuningGoalType
	name                  string
	input, output         string
	location              string
	focus                 [2]float64
	responseTime, dcError float64
	peakError, gainValue  float64
	wl, wr, profile       *System
	wc, crossTol          float64
	gainMargin            float64
	phaseMargin           float64
	minDecay, minDamping  float64
	maxFreq, maxPercent   float64
}

// TuningGoalResult is the evaluation of one goal. Value is MATLAB's
// normalized goal value f (evalGoal): the goal is met when f ≤ Limit = 1.
// Violation is max(0, f − 1). Diagnostics holds the measured physical
// quantities. An unstable loop gives f = +Inf for Poles and Margins.
type TuningGoalResult struct {
	GoalName    string
	Pass        bool
	Value       float64
	Limit       float64
	Violation   float64
	Diagnostics map[string]float64
}

// TuningGoalModel is a model a TuningGoal can be evaluated on: *System,
// *GeneralizedModel or *GeneralizedClosedLoop.
type TuningGoalModel interface {
	tuningGoalResponse(g TuningGoal) (*System, error)
}

const tuningGoalGridPoints = 100

// NewTrackingGoal returns MATLAB
// TuningGoal.Tracking(inputname, outputname, responsetime, dcerror, peakerror):
// the relative error σmax(T(jω) − I) from inputname to outputname stays below
// MaxError(s) = (peakerror·s + ωc·dcerror)/(s + ωc), ωc = 2/responsetime. A
// zero dcerror or peakerror selects the MATLAB defaults 0.001 and 1. Empty
// names, a non-positive or non-finite responsetime or a negative or
// non-finite error returns ErrInvalidArgument. MATLAB regularizes 1/MaxError
// outside −20..60 dB; here the profile is used as is on the frequency grid.
// See https://www.mathworks.com/help/control/ref/tuninggoal.tracking.html.
func NewTrackingGoal(inputname, outputname string, responsetime, dcerror, peakerror float64) (TuningGoal, error) {
	const op = "NewTrackingGoal"
	if err := requireGoalIO(op, inputname, outputname); err != nil {
		return TuningGoal{}, err
	}
	if !(responsetime > 0) || math.IsInf(responsetime, 0) {
		return TuningGoal{}, fmt.Errorf("%s: responsetime %g must be positive and finite: %w", op, responsetime, ErrInvalidArgument)
	}
	if dcerror == 0 {
		dcerror = 0.001
	}
	if peakerror == 0 {
		peakerror = 1
	}
	if err := requirePositiveFinite(op, "dcerror", dcerror); err != nil {
		return TuningGoal{}, err
	}
	if err := requirePositiveFinite(op, "peakerror", peakerror); err != nil {
		return TuningGoal{}, err
	}
	return TuningGoal{typ: TuningGoalTracking, input: inputname, output: outputname, responseTime: responsetime, dcError: dcerror, peakError: peakerror}, nil
}

// NewGainGoal returns MATLAB TuningGoal.Gain(inputname, outputname,
// gainvalue): the largest singular value of the closed loop from inputname
// to outputname stays below gainvalue, f = max σmax(T)/gainvalue. A
// non-positive or non-finite gainvalue or an empty name returns
// ErrInvalidArgument. See
// https://www.mathworks.com/help/control/ref/tuninggoal.gain.html.
func NewGainGoal(inputname, outputname string, gainvalue float64) (TuningGoal, error) {
	const op = "NewGainGoal"
	if err := requireGoalIO(op, inputname, outputname); err != nil {
		return TuningGoal{}, err
	}
	if err := requirePositiveFinite(op, "gainvalue", gainvalue); err != nil {
		return TuningGoal{}, err
	}
	return TuningGoal{typ: TuningGoalGain, input: inputname, output: outputname, gainValue: gainvalue}, nil
}

// NewWeightedGainGoal returns MATLAB TuningGoal.WeightedGain(inputname,
// outputname, WL, WR): f = max σmax(WL·T·WR) over the frequency grid. A nil
// WL or WR means no weighting (MATLAB's scalar 1); the weights are copied,
// and their sizes and domains are checked at evaluation. See
// https://www.mathworks.com/help/control/ref/tuninggoal.weightedgain.html.
func NewWeightedGainGoal(inputname, outputname string, WL, WR *System) (TuningGoal, error) {
	const op = "NewWeightedGainGoal"
	if err := requireGoalIO(op, inputname, outputname); err != nil {
		return TuningGoal{}, err
	}
	g := TuningGoal{typ: TuningGoalWeightedGain, input: inputname, output: outputname}
	for _, w := range []struct {
		name string
		sys  *System
		dst  **System
	}{{"WL", WL, &g.wl}, {"WR", WR, &g.wr}} {
		if w.sys == nil {
			continue
		}
		if err := requireFiniteSystem(op, w.sys); err != nil {
			return TuningGoal{}, fmt.Errorf("%s: %w", w.name, err)
		}
		*w.dst = w.sys.Copy()
	}
	return g, nil
}

// NewRejectionGoal returns MATLAB TuningGoal.Rejection(distloc, attfact):
// the attenuation 1/σmax(S) of a disturbance entering at the analysis point
// distloc exceeds |attfact(jω)|, f = max |attfact|·σmax(S). attfact is a
// SISO model (use NewGain for a constant). See
// https://www.mathworks.com/help/control/ref/tuninggoal.rejection.html.
func NewRejectionGoal(distloc string, attfact *System) (TuningGoal, error) {
	return newProfileGoal("NewRejectionGoal", TuningGoalRejection, distloc, "attfact", attfact)
}

// NewSensitivityGoal returns MATLAB TuningGoal.Sensitivity(location,
// maxsens): σmax(S) at location stays below |maxsens(jω)|,
// f = max σmax(S)/|maxsens|. maxsens is a SISO model (use NewGain for a
// constant). See
// https://www.mathworks.com/help/control/ref/tuninggoal.sensitivity.html.
func NewSensitivityGoal(location string, maxsens *System) (TuningGoal, error) {
	return newProfileGoal("NewSensitivityGoal", TuningGoalSensitivity, location, "maxsens", maxsens)
}

// NewLoopShapeGoal returns MATLAB TuningGoal.LoopShape(location, loopgain,
// crosstol): the open loop L at location follows the SISO target loopgain,
// with gain crossover within crosstol decades of the target's (0 selects the
// MATLAB default 0.1). As MATLAB, it constrains S = (I+L)⁻¹ and T = I − S:
// f = max over ω of max(σmax(S)·|loopgain|, σmax(T)/|loopgain|)/10^crosstol.
// See https://www.mathworks.com/help/control/ref/tuninggoal.loopshape.html.
func NewLoopShapeGoal(location string, loopgain *System, crosstol float64) (TuningGoal, error) {
	const op = "NewLoopShapeGoal"
	g, err := newProfileGoal(op, TuningGoalLoopShape, location, "loopgain", loopgain)
	if err != nil {
		return TuningGoal{}, err
	}
	if g.crossTol, err = loopShapeCrossTol(op, crosstol); err != nil {
		return TuningGoal{}, err
	}
	return g, nil
}

// NewLoopShapeGoalWc returns MATLAB TuningGoal.LoopShape(location, wc) for a
// single wc, the target loop gain wc/s with the default crosstol 0.1, or
// TuningGoal.LoopShape(location, wcrange) for wc = [wc1, wc2], the target
// sqrt(wc1·wc2)/s with crosstol = log10(wc2/wc1)/2. Frequencies must be
// positive, finite and ordered (ErrInvalidArgument).
func NewLoopShapeGoalWc(location string, wc []float64) (TuningGoal, error) {
	const op = "NewLoopShapeGoalWc"
	if err := requireGoalLocation(op, location); err != nil {
		return TuningGoal{}, err
	}
	var lo, hi float64
	switch len(wc) {
	case 1:
		lo, hi = wc[0], wc[0]
	case 2:
		lo, hi = wc[0], wc[1]
	default:
		return TuningGoal{}, fmt.Errorf("%s: wc has %d elements, want 1 or 2: %w", op, len(wc), ErrInvalidArgument)
	}
	if !(lo > 0) || !(hi >= lo) || math.IsInf(hi, 0) {
		return TuningGoal{}, fmt.Errorf("%s: crossover band [%g, %g] must be positive, finite and ordered: %w", op, lo, hi, ErrInvalidArgument)
	}
	crossTol := 0.1
	if len(wc) == 2 {
		crossTol = math.Log10(hi/lo) / 2
	}
	return TuningGoal{typ: TuningGoalLoopShape, location: location, wc: math.Sqrt(lo * hi), crossTol: crossTol}, nil
}

// NewMarginsGoal returns MATLAB TuningGoal.Margins(location, gainmargin,
// phasemargin) with gainmargin in dB and phasemargin in degrees. As MATLAB,
// the margins are disk-based (DiskMargin, skew 0): the requirement is the
// disk size α = max(2(g−1)/(g+1), 2·tan(phasemargin/2)), g = 10^(gainmargin/20),
// and f = α/αmax of the SISO loop at location, +Inf for an unstable loop. A
// negative or non-finite gainmargin or a phasemargin outside [0, 90) returns
// ErrInvalidArgument. See
// https://www.mathworks.com/help/control/ref/tuninggoal.margins.html.
func NewMarginsGoal(location string, gainmargin, phasemargin float64) (TuningGoal, error) {
	const op = "NewMarginsGoal"
	if err := requireGoalLocation(op, location); err != nil {
		return TuningGoal{}, err
	}
	if !(gainmargin >= 0) || math.IsInf(gainmargin, 0) {
		return TuningGoal{}, fmt.Errorf("%s: gainmargin %g dB must be nonnegative and finite: %w", op, gainmargin, ErrInvalidArgument)
	}
	if !(phasemargin >= 0) || !(phasemargin < 90) {
		return TuningGoal{}, fmt.Errorf("%s: phasemargin %g must be in [0, 90) degrees: %w", op, phasemargin, ErrInvalidArgument)
	}
	return TuningGoal{typ: TuningGoalMargins, location: location, gainMargin: gainmargin, phaseMargin: phasemargin}, nil
}

// NewPolesGoal returns MATLAB TuningGoal.Poles(location, mindecay,
// mindamping, maxfreq); an empty location is the MATLAB form without one,
// constraining every closed-loop pole, and otherwise the poles of the
// sensitivity at location. Each pole s (s = log(z)/Ts in discrete time) must
// satisfy Re(s) < −mindecay, Re(s) < −mindamping·|s| and |s| < maxfreq; set
// mindecay = 0, mindamping = 0 or maxfreq = +Inf to drop a constraint
// (stability is always required). f is the worst ratio required/achieved,
// so f = 1.1 means about 10% short, and +Inf for an unstable pole. A negative
// or non-finite mindecay, mindamping outside [0, 1] or a non-positive maxfreq
// returns ErrInvalidArgument. See
// https://www.mathworks.com/help/control/ref/tuninggoal.poles.html.
func NewPolesGoal(location string, mindecay, mindamping, maxfreq float64) (TuningGoal, error) {
	const op = "NewPolesGoal"
	if !(mindecay >= 0) || math.IsInf(mindecay, 0) {
		return TuningGoal{}, fmt.Errorf("%s: mindecay %g must be nonnegative and finite: %w", op, mindecay, ErrInvalidArgument)
	}
	if !(mindamping >= 0 && mindamping <= 1) {
		return TuningGoal{}, fmt.Errorf("%s: mindamping %g must be in [0, 1]: %w", op, mindamping, ErrInvalidArgument)
	}
	if !(maxfreq > 0) {
		return TuningGoal{}, fmt.Errorf("%s: maxfreq %g must be positive: %w", op, maxfreq, ErrInvalidArgument)
	}
	return TuningGoal{typ: TuningGoalPoles, location: location, minDecay: mindecay, minDamping: mindamping, maxFreq: maxfreq}, nil
}

// NewOvershootGoal returns MATLAB TuningGoal.Overshoot(inputname,
// outputname, maxpercent): f = overshoot/maxpercent for the largest step
// overshoot (percent) from inputname to outputname. MATLAB tunes a peak-gain
// proxy; here the overshoot is measured from the step response
// (StepInfoForSystem). A non-positive or non-finite maxpercent returns
// ErrInvalidArgument. See
// https://www.mathworks.com/help/control/ref/tuninggoal.overshoot.html.
func NewOvershootGoal(inputname, outputname string, maxpercent float64) (TuningGoal, error) {
	const op = "NewOvershootGoal"
	if err := requireGoalIO(op, inputname, outputname); err != nil {
		return TuningGoal{}, err
	}
	if err := requirePositiveFinite(op, "maxpercent", maxpercent); err != nil {
		return TuningGoal{}, err
	}
	return TuningGoal{typ: TuningGoalOvershoot, input: inputname, output: outputname, maxPercent: maxpercent}, nil
}

func newProfileGoal(op string, typ TuningGoalType, location, argName string, profile *System) (TuningGoal, error) {
	if err := requireGoalLocation(op, location); err != nil {
		return TuningGoal{}, err
	}
	if err := requireFiniteSystem(op, profile); err != nil {
		return TuningGoal{}, fmt.Errorf("%s: %w", argName, err)
	}
	if _, m, p := profile.Dims(); m != 1 || p != 1 {
		return TuningGoal{}, fmt.Errorf("%s: %s is %d×%d, want SISO: %w", op, argName, p, m, ErrNotSISO)
	}
	return TuningGoal{typ: typ, location: location, profile: profile.Copy()}, nil
}

func loopShapeCrossTol(op string, crosstol float64) (float64, error) {
	if crosstol == 0 {
		return 0.1, nil
	}
	if !(crosstol > 0) || math.IsInf(crosstol, 0) {
		return 0, fmt.Errorf("%s: crosstol %g must be positive and finite: %w", op, crosstol, ErrInvalidArgument)
	}
	return crosstol, nil
}

func requireGoalIO(op, inputname, outputname string) error {
	if inputname == "" || outputname == "" {
		return fmt.Errorf("%s: inputname %q and outputname %q must be non-empty: %w", op, inputname, outputname, ErrInvalidArgument)
	}
	return nil
}

func requireGoalLocation(op, location string) error {
	if location == "" {
		return fmt.Errorf("%s: location is empty: %w", op, ErrInvalidArgument)
	}
	return nil
}

func requirePositiveFinite(op, name string, v float64) error {
	if !(v > 0) || math.IsInf(v, 0) {
		return fmt.Errorf("%s: %s %g must be positive and finite: %w", op, name, v, ErrInvalidArgument)
	}
	return nil
}

// Type returns the MATLAB class the goal mirrors.
func (g TuningGoal) Type() TuningGoalType { return g.typ }

// Name returns the goal name, as MATLAB Req.Name; it defaults to the class
// name (MATLAB's default is empty).
func (g TuningGoal) Name() string {
	if g.name == "" {
		return g.typ.String()
	}
	return g.name
}

// WithName returns a copy of g named name, as setting MATLAB Req.Name.
func (g TuningGoal) WithName(name string) TuningGoal {
	g.name = name
	return g
}

// WithFocus returns a copy of g enforced on [wmin, wmax] rad/s, as MATLAB's
// Focus property; frequency-domain goals sample 100 log-spaced frequencies
// there, clipped at the Nyquist frequency of a discrete response. Without a
// focus they sample a decade beyond the measured response's poles and delay
// corners (ending at the Nyquist frequency in discrete time), as
// DefaultFrequencyGrid does from poles and zeros, since a log grid cannot
// cover MATLAB's default [0, Inf]. A band that is not
// 0 < wmin < wmax < Inf returns ErrInvalidArgument, and a Poles, Margins or
// Overshoot goal ErrOptionUnsupported.
func (g TuningGoal) WithFocus(wmin, wmax float64) (TuningGoal, error) {
	switch g.typ {
	case TuningGoalPoles, TuningGoalMargins, TuningGoalOvershoot:
		return TuningGoal{}, fmt.Errorf("TuningGoal.WithFocus: %v goal is not evaluated on a frequency grid: %w", g.typ, ErrOptionUnsupported)
	}
	if !(wmin > 0) || !(wmax > wmin) || math.IsInf(wmax, 0) {
		return TuningGoal{}, fmt.Errorf("TuningGoal.WithFocus: band [%g, %g] must satisfy 0 < wmin < wmax < Inf: %w", wmin, wmax, ErrInvalidArgument)
	}
	g.focus = [2]float64{wmin, wmax}
	return g, nil
}

// Evaluate measures the goal on model, as MATLAB evalGoal.
func (g TuningGoal) Evaluate(model TuningGoalModel) (TuningGoalResult, error) {
	const op = "TuningGoal.Evaluate"
	if model == nil {
		return TuningGoalResult{}, fmt.Errorf("%s: model is nil: %w", op, ErrInvalidArgument)
	}
	if err := g.valid(); err != nil {
		return TuningGoalResult{}, fmt.Errorf("%s: %w", op, err)
	}
	sys, err := model.tuningGoalResponse(g)
	if err != nil {
		return TuningGoalResult{}, fmt.Errorf("%s: %w", op, err)
	}
	res, err := g.evaluateSystem(sys)
	if err != nil {
		return TuningGoalResult{}, fmt.Errorf("%s: goal %q: %w", op, g.Name(), err)
	}
	return res, nil
}

func (g TuningGoal) valid() error {
	if g.typ < TuningGoalTracking || g.typ > TuningGoalOvershoot || (g.typ == TuningGoalPoles && g.maxFreq == 0) || (g.typ != TuningGoalPoles && g.input == "" && g.location == "") {
		return fmt.Errorf("goal was not built by a New*Goal constructor: %w", ErrInvalidArgument)
	}
	return nil
}

func (g TuningGoal) isIOGoal() bool {
	switch g.typ {
	case TuningGoalTracking, TuningGoalGain, TuningGoalWeightedGain, TuningGoalOvershoot:
		return true
	default:
		return false
	}
}

func (sys *System) tuningGoalResponse(g TuningGoal) (*System, error) {
	if err := requireSystem("TuningGoal.Evaluate", sys); err != nil {
		return nil, err
	}
	if !g.isIOGoal() {
		return sys.Copy(), nil
	}
	return selectGoalChannels(sys, g.input, g.output, false, false)
}

func (m *GeneralizedModel) tuningGoalResponse(g TuningGoal) (*System, error) {
	sys, err := m.CurrentSystem()
	if err != nil {
		return nil, err
	}
	return sys.tuningGoalResponse(g)
}

func (m *GeneralizedClosedLoop) tuningGoalResponse(g TuningGoal) (*System, error) {
	key, err := g.responseKey(m)
	if err != nil {
		return nil, err
	}
	sys, err := m.goalResponse(key)
	if err != nil {
		return nil, err
	}
	return g.selectResponse(m, sys)
}

type tuningGoalResponse uint8

const (
	tuningGoalClosedLoopResponse tuningGoalResponse = iota
	tuningGoalSensitivityResponse
	tuningGoalOpenLoopResponse
)

type tuningGoalResponseKey struct {
	point    string
	response tuningGoalResponse
}

// responseKey resolves the analysis point and response the goal measures on
// a closed loop.
func (g TuningGoal) responseKey(m *GeneralizedClosedLoop) (tuningGoalResponseKey, error) {
	if m == nil {
		return tuningGoalResponseKey{}, fmt.Errorf("nil model: %w", ErrInvalidArgument)
	}
	switch g.typ {
	case TuningGoalRejection, TuningGoalSensitivity:
		return tuningGoalResponseKey{point: g.location, response: tuningGoalSensitivityResponse}, nil
	case TuningGoalLoopShape, TuningGoalMargins:
		return tuningGoalResponseKey{point: g.location, response: tuningGoalOpenLoopResponse}, nil
	case TuningGoalPoles:
		if g.location != "" {
			return tuningGoalResponseKey{point: g.location, response: tuningGoalSensitivityResponse}, nil
		}
		return tuningGoalResponseKey{point: m.primaryAnalysisPointName(), response: tuningGoalClosedLoopResponse}, nil
	}
	point := m.primaryAnalysisPointName()
	inAP, outAP := m.hasAnalysisPoint(g.input), m.hasAnalysisPoint(g.output)
	switch {
	case inAP && outAP && g.input != g.output:
		return tuningGoalResponseKey{}, fmt.Errorf("inputname %q and outputname %q are different analysis points: %w", g.input, g.output, ErrOptionUnsupported)
	case inAP:
		point = g.input
	case outAP:
		point = g.output
	}
	return tuningGoalResponseKey{point: point, response: tuningGoalClosedLoopResponse}, nil
}

func (g TuningGoal) selectResponse(m *GeneralizedClosedLoop, sys *System) (*System, error) {
	if !g.isIOGoal() {
		return sys, nil
	}
	return selectGoalChannels(sys, g.input, g.output, m.hasAnalysisPoint(g.input), m.hasAnalysisPoint(g.output))
}

func (m *GeneralizedClosedLoop) hasAnalysisPoint(name string) bool {
	_, ok := m.analysisPoints[name]
	return ok
}

func (m *GeneralizedClosedLoop) goalResponse(key tuningGoalResponseKey) (*System, error) {
	switch key.response {
	case tuningGoalSensitivityResponse:
		return m.Sensitivity(key.point)
	case tuningGoalOpenLoopResponse:
		return m.OpenLoop(key.point)
	default:
		return m.ClosedLoop(key.point)
	}
}

// selectGoalChannels keeps the input channel named input and output channel
// named output; a side flagged all keeps every channel.
func selectGoalChannels(sys *System, input, output string, allInputs, allOutputs bool) (*System, error) {
	if allInputs && allOutputs {
		return sys, nil
	}
	_, m, p := sys.Dims()
	inputs, outputs := identityIndices(m), identityIndices(p)
	var err error
	if !allInputs {
		if inputs, err = lookupGoalSignal(sys.InputName, input); err != nil {
			return nil, fmt.Errorf("inputname: %w", err)
		}
	}
	if !allOutputs {
		if outputs, err = lookupGoalSignal(sys.OutputName, output); err != nil {
			return nil, fmt.Errorf("outputname: %w", err)
		}
	}
	return sys.SelectByIndex(inputs, outputs)
}

// lookupGoalSignal returns the channel named name, or every channel of the
// vector signal name(1), name(2), ... as MATLAB resolves a vector signal name.
func lookupGoalSignal(names []string, name string) ([]int, error) {
	if i, err := lookupSignalIndex(names, name); err == nil {
		return []int{i}, nil
	}
	var out []int
	for k := 1; ; k++ {
		i, err := lookupSignalIndex(names, fmt.Sprintf("%s(%d)", name, k))
		if err != nil {
			break
		}
		out = append(out, i)
	}
	if len(out) == 0 {
		return nil, fmt.Errorf("signal %q: %w", name, ErrSignalNotFound)
	}
	return out, nil
}

func (g TuningGoal) evaluateSystem(sys *System) (TuningGoalResult, error) {
	switch g.typ {
	case TuningGoalTracking:
		return g.evaluateTracking(sys)
	case TuningGoalGain:
		return g.evaluateGain(sys)
	case TuningGoalWeightedGain:
		return g.evaluateWeightedGain(sys)
	case TuningGoalRejection, TuningGoalSensitivity:
		return g.evaluateSensitivity(sys)
	case TuningGoalLoopShape:
		return g.evaluateLoopShape(sys)
	case TuningGoalMargins:
		return g.evaluateMargins(sys)
	case TuningGoalPoles:
		return g.evaluatePoles(sys)
	case TuningGoalOvershoot:
		return g.evaluateOvershoot(sys)
	default:
		return TuningGoalResult{}, fmt.Errorf("unsupported goal type %v: %w", g.typ, ErrInvalidArgument)
	}
}

// frequencyGrid returns the focus band grid, or the response's default grid.
func (g TuningGoal) frequencyGrid(sys *System) ([]float64, error) {
	if g.focus[1] == 0 {
		poles, err := sys.Poles()
		if err != nil {
			return nil, err
		}
		lo, hi := autoFreqRange(sys, poles, systemDelays(sys))
		omega := logspace(math.Log10(lo), math.Log10(hi), tuningGoalGridPoints)
		omega[len(omega)-1] = hi
		return omega, nil
	}
	lo, hi := g.focus[0], g.focus[1]
	if sys.IsDiscrete() {
		hi = math.Min(hi, math.Pi/sys.Dt)
		if lo >= hi {
			return nil, fmt.Errorf("focus [%g, %g] lies above the Nyquist frequency %g: %w", g.focus[0], g.focus[1], hi, ErrInvalidArgument)
		}
	}
	omega := logspace(math.Log10(lo), math.Log10(hi), tuningGoalGridPoints)
	omega[0], omega[len(omega)-1] = lo, hi
	return omega, nil
}

// profileMagnitude evaluates |G| along omega, at jω for a continuous G and
// e^{jωTs} for a discrete one, independently of the measured response.
func profileMagnitude(profile *System, omega []float64) ([]float64, error) {
	out := make([]float64, len(omega))
	for k, w := range omega {
		s := complex(0, w)
		if profile.IsDiscrete() {
			s = cmplx.Exp(complex(0, w*profile.Dt))
		}
		h, err := profile.EvalFr(s)
		if err != nil {
			return nil, err
		}
		out[k] = cmplx.Abs(h[0][0])
	}
	return out, nil
}

func (g TuningGoal) result(f float64, diag map[string]float64) TuningGoalResult {
	return TuningGoalResult{
		GoalName:    g.Name(),
		Pass:        f <= 1,
		Value:       f,
		Limit:       1,
		Violation:   math.Max(0, f-1),
		Diagnostics: diag,
	}
}

func (g TuningGoal) evaluateTracking(sys *System) (TuningGoalResult, error) {
	omega, err := g.frequencyGrid(sys)
	if err != nil {
		return TuningGoalResult{}, err
	}
	wc := 2 / g.responseTime
	scale := make([]float64, len(omega))
	for k, w := range omega {
		s := complex(0, w)
		scale[k] = cmplx.Abs((s + complex(wc, 0)) / (complex(g.peakError, 0)*s + complex(wc*g.dcError, 0)))
	}
	f, at, err := gainSweep{minusIdentity: true, scale: scale}.peak(sys, omega)
	if err != nil {
		return TuningGoalResult{}, err
	}
	return g.result(f, map[string]float64{"peak_frequency": at}), nil
}

func (g TuningGoal) evaluateGain(sys *System) (TuningGoalResult, error) {
	omega, err := g.frequencyGrid(sys)
	if err != nil {
		return TuningGoalResult{}, err
	}
	peak, at, err := gainSweep{}.peak(sys, omega)
	if err != nil {
		return TuningGoalResult{}, err
	}
	return g.result(peak/g.gainValue, map[string]float64{"peak_gain": peak, "peak_frequency": at}), nil
}

func (g TuningGoal) evaluateWeightedGain(sys *System) (TuningGoalResult, error) {
	omega, err := g.frequencyGrid(sys)
	if err != nil {
		return TuningGoalResult{}, err
	}
	f, at, err := gainSweep{outputWeight: g.wl, inputWeight: g.wr}.peak(sys, omega)
	if err != nil {
		return TuningGoalResult{}, err
	}
	return g.result(f, map[string]float64{"peak_frequency": at}), nil
}

func (g TuningGoal) evaluateSensitivity(sys *System) (TuningGoalResult, error) {
	omega, err := g.frequencyGrid(sys)
	if err != nil {
		return TuningGoalResult{}, err
	}
	scale, err := profileMagnitude(g.profile, omega)
	if err != nil {
		return TuningGoalResult{}, err
	}
	if g.typ == TuningGoalSensitivity {
		for k := range scale {
			scale[k] = 1 / scale[k]
		}
	}
	f, at, err := gainSweep{scale: scale}.peak(sys, omega)
	if err != nil {
		return TuningGoalResult{}, err
	}
	return g.result(f, map[string]float64{"peak_frequency": at}), nil
}

func (g TuningGoal) evaluateLoopShape(loop *System) (TuningGoalResult, error) {
	_, m, p := loop.Dims()
	if m != p {
		return TuningGoalResult{}, fmt.Errorf("loop is %d×%d, want square: %w", p, m, ErrDimensionMismatch)
	}
	eye, err := makeIdentityGain(p, loop.Dt)
	if err != nil {
		return TuningGoalResult{}, err
	}
	S, err := Feedback(eye, loop, -1)
	if err != nil {
		return TuningGoalResult{}, err
	}
	T, err := Feedback(loop, eye, -1)
	if err != nil {
		return TuningGoalResult{}, err
	}
	omega, err := g.frequencyGrid(loop)
	if err != nil {
		return TuningGoalResult{}, err
	}
	var target []float64
	if g.profile != nil {
		if target, err = profileMagnitude(g.profile, omega); err != nil {
			return TuningGoalResult{}, err
		}
	} else {
		target = make([]float64, len(omega))
		for k, w := range omega {
			target[k] = g.wc / w
		}
	}
	tol := math.Pow(10, g.crossTol)
	sScale, tScale := make([]float64, len(omega)), make([]float64, len(omega))
	for k, gk := range target {
		sScale[k], tScale[k] = gk/tol, 1/(gk*tol)
	}
	fs, _, err := gainSweep{scale: sScale}.peak(S, omega)
	if err != nil {
		return TuningGoalResult{}, err
	}
	ft, _, err := gainSweep{scale: tScale}.peak(T, omega)
	if err != nil {
		return TuningGoalResult{}, err
	}
	return g.result(math.Max(fs, ft), map[string]float64{"low_frequency": fs, "high_frequency": ft}), nil
}

func (g TuningGoal) evaluateMargins(loop *System) (TuningGoalResult, error) {
	dm, err := DiskMargin(loop)
	if err != nil {
		return TuningGoalResult{}, err
	}
	gain := math.Pow(10, g.gainMargin/20)
	required := math.Max(2*(gain-1)/(gain+1), 2*math.Tan(g.phaseMargin*math.Pi/360))
	f := math.Inf(1)
	if dm.Alpha > 0 {
		f = required / dm.Alpha
	}
	return g.result(f, map[string]float64{
		"disk_margin":           dm.Alpha,
		"disk_gain_margin_db":   dm.GainMarginDB[1],
		"disk_phase_margin_deg": dm.PhaseMargin,
		"required_disk_margin":  required,
	}), nil
}

func (g TuningGoal) evaluatePoles(sys *System) (TuningGoalResult, error) {
	poles, err := sys.Poles()
	if err != nil {
		return TuningGoalResult{}, err
	}
	minDecay, minDamping, maxFreq := math.Inf(1), math.Inf(1), 0.0
	for _, p := range poles {
		if sys.IsDiscrete() && p == 0 {
			// z = 0 is s = −∞: infinitely fast decay, damping 1, unbounded frequency.
			maxFreq = math.Inf(1)
			minDamping = math.Min(minDamping, 1)
			continue
		}
		s := p
		if sys.IsDiscrete() {
			s = cmplx.Log(p) / complex(sys.Dt, 0)
		}
		decay := -real(s)
		damping := -1.0
		if mag := cmplx.Abs(s); mag > 0 {
			damping = decay / mag
		}
		minDecay = math.Min(minDecay, decay)
		minDamping = math.Min(minDamping, damping)
		maxFreq = math.Max(maxFreq, cmplx.Abs(s))
	}
	f := 0.0
	if len(poles) > 0 {
		if !(minDecay > 0) {
			f = math.Inf(1)
		} else {
			f = math.Max(g.minDecay/minDecay, g.minDamping/minDamping)
			if !math.IsInf(g.maxFreq, 1) {
				f = math.Max(f, maxFreq/g.maxFreq)
			}
		}
	}
	return g.result(f, map[string]float64{"min_decay": minDecay, "min_damping": minDamping, "max_frequency": maxFreq}), nil
}

func (g TuningGoal) evaluateOvershoot(sys *System) (TuningGoalResult, error) {
	info, err := StepInfoForSystem(sys, 0, nil)
	if err != nil {
		return TuningGoalResult{}, err
	}
	overshoot := 0.0
	for _, metric := range info.Metrics {
		overshoot = math.Max(overshoot, metric.Overshoot)
	}
	return g.result(overshoot/g.maxPercent, map[string]float64{"overshoot_percent": overshoot}), nil
}

// gainSweep measures max over the grid of scale[k]·σmax(WL·(H − I?)·WR).
type gainSweep struct {
	outputWeight, inputWeight *System
	minusIdentity             bool
	scale                     []float64
}

func (s gainSweep) peak(sys *System, omega []float64) (peak, at float64, err error) {
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return 0, 0, err
	}
	if s.minusIdentity && resp.P != resp.M {
		return 0, 0, fmt.Errorf("tracking needs as many inputs as outputs, have %d×%d: %w", resp.P, resp.M, ErrDimensionMismatch)
	}
	var outputResponse, inputResponse *FreqResponseMatrix
	if s.outputWeight != nil {
		if err := domainMatch(sys, s.outputWeight); err != nil {
			return 0, 0, fmt.Errorf("WL: %w", err)
		}
		if outputResponse, err = s.outputWeight.FreqResponse(omega); err != nil {
			return 0, 0, err
		}
	}
	if s.inputWeight != nil {
		if err := domainMatch(sys, s.inputWeight); err != nil {
			return 0, 0, fmt.Errorf("WR: %w", err)
		}
		if inputResponse, err = s.inputWeight.FreqResponse(omega); err != nil {
			return 0, 0, err
		}
	}
	rows, cols := resp.P, resp.M
	var outputProduct, inputProduct, shifted []complex128
	if outputResponse != nil {
		outputProduct = make([]complex128, outputResponse.P*resp.M)
		rows = outputResponse.P
	}
	if inputResponse != nil {
		inputProduct = make([]complex128, rows*inputResponse.M)
		cols = inputResponse.M
	}
	if s.minusIdentity {
		shifted = make([]complex128, resp.P*resp.M)
	}
	var svd *complexSVDWorkspace
	if rows > 1 && cols > 1 && (rows != 2 || cols != 2) {
		svd = newComplexSVDWorkspace(rows, cols)
	}
	peak = math.Inf(-1)
	for k, w := range omega {
		gain := complexResponseAt(resp, k)
		if s.minusIdentity {
			copy(shifted, gain.data)
			for i := range gain.rows {
				shifted[i*gain.cols+i]--
			}
			gain.data = shifted
		}
		if outputResponse != nil {
			if gain, err = multiplyComplexMatricesInto(outputProduct, complexResponseAt(outputResponse, k), gain); err != nil {
				return 0, 0, fmt.Errorf("WL: %w", err)
			}
		}
		if inputResponse != nil {
			if gain, err = multiplyComplexMatricesInto(inputProduct, gain, complexResponseAt(inputResponse, k)); err != nil {
				return 0, 0, fmt.Errorf("WR: %w", err)
			}
		}
		sigma, err := svd.maximumFromFlat(gain.data, 0, gain.rows, gain.cols)
		if err != nil {
			return 0, 0, err
		}
		if s.scale != nil {
			sigma *= s.scale[k]
		}
		if sigma > peak {
			peak, at = sigma, w
		}
	}
	return peak, at, nil
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
