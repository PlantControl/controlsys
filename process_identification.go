package controlsys

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sort"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v1/gonum/optimize"
)

// ProcessStructure selects a low-order process model, as the MATLAB procest
// type string ('P1D', 'P2DUZI', ...): one to three real or paired poles, an
// optional zero, integrator and transport delay.
type ProcessStructure struct {
	Poles           int  `json:"poles"`
	UnderdampedPair bool `json:"underdampedPair"`
	Zero            bool `json:"zero"`
	Integrator      bool `json:"integrator"`
	Delay           bool `json:"delay"`
}

// ProcessParameters are the physical parameters of a ProcessStructure:
// static gain, real time constants, an underdamped pair (NaturalFrequency,
// Damping), zero time and delay, in the data's time unit.
type ProcessParameters struct {
	Gain             float64   `json:"gain"`
	TimeConstants    []float64 `json:"timeConstants"`
	NaturalFrequency float64   `json:"naturalFrequency,omitempty"`
	Damping          float64   `json:"damping,omitempty"`
	ZeroTime         float64   `json:"zeroTime,omitempty"`
	Delay            float64   `json:"delay,omitempty"`
}

// ProcessFitData is a uniformly sampled SISO record. The first
// TrainingSamples (at least 20) are fitted; the rest (at least 3, after any
// ValidationInitializationSamples) are held out for validation.
type ProcessFitData struct {
	Input           []float64
	Output          []float64
	SampleTime      float64
	TrainingSamples int
}

// ProcessBounds limits each parameter. A Min/Max pair left at 0/0 selects
// a default derived from the sample time and record length.
type ProcessBounds struct {
	MinTimeConstant float64 `json:"minTimeConstant"`
	MaxTimeConstant float64 `json:"maxTimeConstant"`
	MinDelay        float64 `json:"minDelay"`
	MaxDelay        float64 `json:"maxDelay"`
	MinZero         float64 `json:"minZero"`
	MaxZero         float64 `json:"maxZero"`
	MinGain         float64 `json:"minGain"`
	MaxGain         float64 `json:"maxGain"`
	MinFrequency    float64 `json:"minFrequency"`
	MaxFrequency    float64 `json:"maxFrequency"`
	MinDamping      float64 `json:"minDamping"`
	MaxDamping      float64 `json:"maxDamping"`
}

// ProcessValidationInit selects how the held-out validation run starts.
type ProcessValidationInit string

const (
	// ProcessValidationContinuation continues the free run from the end of
	// training (the zero value "" means this).
	ProcessValidationContinuation ProcessValidationInit = "continuation"
	// ProcessValidationZero restarts validation from rest.
	ProcessValidationZero ProcessValidationInit = "zero"
)

// ProcessFitTermination names why a process fit stopped.
type ProcessFitTermination string

const (
	ProcessFitConverged       ProcessFitTermination = "converged"
	ProcessFitEvaluationLimit ProcessFitTermination = "evaluation-limit"
	ProcessFitCanceled        ProcessFitTermination = "canceled"
	ProcessFitManual          ProcessFitTermination = "manual-evaluation"
)

// ProcessFitOptions configures FitProcess and EvaluateProcess. Zero values
// select defaults: 2000 evaluations, 4 starts, default bounds and
// continuation validation.
type ProcessFitOptions struct {
	ValidationInitialCondition      ProcessValidationInit `json:"validationInitialCondition"`
	ValidationInitializationSamples int                   `json:"validationInitializationSamples,omitempty"`

	Structure            ProcessStructure   `json:"structure"`
	Bounds               ProcessBounds      `json:"bounds"`
	Initial              *ProcessParameters `json:"initial,omitempty"`
	EstimateInitialState bool               `json:"estimateInitialState"`
	EstimateOffset       bool               `json:"estimateOffset"`
	MaxEvaluations       int                `json:"maxEvaluations"`
	Starts               int                `json:"starts"`
}

// ProcessFitResult is a fitted or evaluated process model with its training
// and validation diagnostics. InitialState is in unit-gain model
// coordinates.
type ProcessFitResult struct {
	Parameters          ProcessParameters     `json:"parameters"`
	Structure           ProcessStructure      `json:"structure"`
	Bounds              ProcessBounds         `json:"bounds"`
	InitialState        []float64             `json:"initialState,omitempty"`
	Offset              float64               `json:"offset"`
	Predicted           []float64             `json:"predicted"`
	Residuals           []float64             `json:"residuals"`
	TrainingNRMSE       float64               `json:"trainingNrmse"`
	ValidationNRMSE     float64               `json:"validationNrmse"`
	ValidationMeanNRMSE float64               `json:"validationMeanNrmse"`
	Condition           float64               `json:"condition"`
	Evaluations         int                   `json:"evaluations"`
	Termination         ProcessFitTermination `json:"termination"`
	ActiveBounds        []string              `json:"activeBounds,omitempty"`
	Diagnostics         []string              `json:"diagnostics,omitempty"`
	System              *System               `json:"-"`

	residualLagOne float64
	residualLagOK  bool
}

// ResidualLagOne returns the lag-one Pearson correlation of the validation
// residuals. ok is false when it is undefined: constant (for example zero)
// residuals have no variance.
func (r *ProcessFitResult) ResidualLagOne() (rho float64, ok bool) {
	return r.residualLagOne, r.residualLagOK
}

type processFitResultJSON ProcessFitResult

type processFitResultWire struct {
	*processFitResultJSON
	ResidualLagOne *float64 `json:"residualLagOne,omitempty"`
}

// MarshalJSON encodes ResidualLagOne as "residualLagOne", omitted when
// undefined.
func (r ProcessFitResult) MarshalJSON() ([]byte, error) {
	w := processFitResultWire{processFitResultJSON: (*processFitResultJSON)(&r)}
	if r.residualLagOK {
		w.ResidualLagOne = &r.residualLagOne
	}
	return json.Marshal(w)
}

// UnmarshalJSON decodes the encoding of MarshalJSON.
func (r *ProcessFitResult) UnmarshalJSON(b []byte) error {
	w := processFitResultWire{processFitResultJSON: (*processFitResultJSON)(r)}
	if err := json.Unmarshal(b, &w); err != nil {
		return err
	}
	r.residualLagOne, r.residualLagOK = 0, w.ResidualLagOne != nil
	if r.residualLagOK {
		r.residualLagOne = *w.ResidualLagOne
	}
	return nil
}

// ProcessFitCanceledError reports a FitProcess run stopped by its context.
// Best is the best valid fit found before cancellation (Termination
// ProcessFitCanceled); Err is the context error it unwraps to.
type ProcessFitCanceledError struct {
	Best *ProcessFitResult
	Err  error
}

func (e *ProcessFitCanceledError) Error() string { return "FitProcess: " + e.Err.Error() }

func (e *ProcessFitCanceledError) Unwrap() error { return e.Err }

// System realizes the physical process definition, retaining transport delay
// exactly. TimeConstants are stable real poles; a damped pair consumes two of
// Structure.Poles. The optional integrator is in addition to those poles.
func (p ProcessParameters) System(s ProcessStructure) (*System, error) {
	unit, err := processSystem(s, p)
	if err != nil {
		return nil, fmt.Errorf("ProcessParameters.System: %w", err)
	}
	unit.C.Scale(p.Gain, unit.C)
	unit.D.Scale(p.Gain, unit.D)
	if p.Delay > 0 {
		if err = unit.SetInputDelay([]float64{p.Delay}); err != nil {
			return nil, fmt.Errorf("ProcessParameters.System: %w", err)
		}
	}
	return unit, nil
}

func processSystem(s ProcessStructure, p ProcessParameters) (*System, error) {
	realPoles := s.Poles
	if s.UnderdampedPair {
		realPoles -= 2
	}
	if s.Poles < 1 || s.Poles > 3 || realPoles < 0 || len(p.TimeConstants) != realPoles || !processFinite(p.Gain) {
		return nil, fmt.Errorf("invalid process structure: %w", ErrProcessData)
	}
	den := []float64{1}
	multiply := func(a, b []float64) []float64 {
		v := make([]float64, len(a)+len(b)-1)
		for i, x := range a {
			for j, y := range b {
				v[i+j] += x * y
			}
		}
		return v
	}
	for _, tau := range p.TimeConstants {
		if !processFinite(tau) || tau <= 0 {
			return nil, fmt.Errorf("time constants must be positive: %w", ErrProcessData)
		}
		den = multiply(den, []float64{tau, 1})
	}
	if s.UnderdampedPair {
		if !processFinite(p.NaturalFrequency) || p.NaturalFrequency <= 0 || !processFinite(p.Damping) || p.Damping <= 0 || p.Damping >= 1 {
			return nil, fmt.Errorf("pair requires positive frequency and damping in (0,1): %w", ErrProcessData)
		}
		den = multiply(den, []float64{1 / (p.NaturalFrequency * p.NaturalFrequency), 2 * p.Damping / p.NaturalFrequency, 1})
	}
	if s.Integrator {
		den = append(den, 0)
	}
	if !processFinite(p.Delay) || p.Delay < 0 || (!s.Delay && p.Delay != 0) || !processFinite(p.ZeroTime) || (!s.Zero && p.ZeroTime != 0) {
		return nil, fmt.Errorf("inactive or invalid zero/delay: %w", ErrProcessData)
	}
	num := []float64{1}
	if s.Zero {
		num = []float64{p.ZeroTime, 1}
	}
	n := len(den) - 1
	scale := den[0]
	for i := range den {
		den[i] /= scale
	}
	padded := make([]float64, n+1)
	for i, v := range num {
		padded[n+1-len(num)+i] = v / scale
	}
	a := mat.NewDense(n, n, nil)
	b := mat.NewDense(n, 1, nil)
	c := mat.NewDense(1, n, nil)
	d := mat.NewDense(1, 1, []float64{padded[0]})
	for j := range n {
		a.Set(0, j, -den[j+1])
		c.Set(0, j, padded[j+1]-padded[0]*den[j+1])
		if j > 0 {
			a.Set(j, j-1, 1)
		}
	}
	b.Set(0, 0, 1)
	return New(a, b, c, d, 0)
}

// FitProcess estimates a process model from a SISO record, the analogue of
// MATLAB procest (System Identification Toolbox). It fits only training
// outputs. Validation is a free run continued from training with measured
// inputs and no output-based correction. An exhausted evaluation budget
// returns the best valid result with Termination ProcessFitEvaluationLimit;
// cancellation returns a *ProcessFitCanceledError carrying it (or the bare
// context error when no valid candidate was found yet). No valid candidate
// returns ErrProcessFit with the last candidate's failure. EstimateInitialState
// also assumes the input was held at its first sample before the record,
// which fills the delay line of a delayed model.
func FitProcess(ctx context.Context, data ProcessFitData, options ProcessFitOptions) (*ProcessFitResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("FitProcess: nil context: %w", ErrInvalidArgument)
	}
	options, err := validateProcessFit(data, options)
	if err != nil {
		return nil, fmt.Errorf("FitProcess: %w", err)
	}
	coordinates := processCoordinates(options.Structure, options.Bounds)
	bestValue := math.Inf(1)
	var best *ProcessFitResult
	evals := 0
	termination := ProcessFitEvaluationLimit
	var lastErr error
	objective := func(x []float64) float64 {
		if ctx.Err() != nil || evals >= options.MaxEvaluations {
			return math.Inf(1)
		}
		evals++
		for _, v := range x {
			if v < 0 || v > 1 || !processFinite(v) {
				return 1e100
			}
		}
		p := decodeProcessPoint(x, coordinates, options.Structure)
		result, value, err := evaluateProcessFit(data, options, p, nil)
		if err != nil {
			lastErr = err
			return 1e100
		}
		if !processFinite(value) {
			return 1e100
		}
		if value < bestValue {
			bestValue = value
			best = result
		}
		return value
	}
	for start := 0; start < options.Starts && evals < options.MaxEvaluations; start++ {
		if ctx.Err() != nil {
			termination = ProcessFitCanceled
			break
		}
		x := make([]float64, len(coordinates))
		for j := range x {
			x[j] = math.Mod(.38196601125*float64((start+1)*(j+2)), 1)
			x[j] = .05 + .9*x[j]
		}
		if start == 0 {
			p := defaultProcessParameters(data, options)
			if options.Initial != nil {
				p = *options.Initial
			}
			x = encodeProcessPoint(p, coordinates)
		}
		budget := (options.MaxEvaluations - evals) / (options.Starts - start)
		budget = max(budget, len(x)+2)
		previousBest := bestValue
		result, fitErr := optimize.Minimize(optimize.Problem{Func: objective, Status: func() (optimize.Status, error) {
			if ctx.Err() != nil {
				return optimize.Failure, ctx.Err()
			}
			return optimize.NotTerminated, nil
		}}, x, &optimize.Settings{FuncEvaluations: budget, Converger: &optimize.FunctionConverge{Absolute: 1e-14, Relative: 1e-9, Iterations: 40}}, &optimize.NelderMead{SimplexSize: .06})
		if ctx.Err() != nil {
			termination = ProcessFitCanceled
			break
		}
		if bestValue < previousBest {
			termination = ProcessFitEvaluationLimit
		}
		if fitErr == nil && result != nil && !result.Status.Early() && result.F <= bestValue+1e-12*math.Max(1, bestValue) {
			termination = ProcessFitConverged
		}
		if bestValue < 1e-20 {
			termination = ProcessFitConverged
			break
		}
	}
	if best == nil {
		if err := ctx.Err(); err != nil {
			return nil, fmt.Errorf("FitProcess: %w", err)
		}
		if lastErr != nil {
			return nil, fmt.Errorf("FitProcess: no valid candidate (last: %v): %w", lastErr, ErrProcessFit)
		}
		return nil, fmt.Errorf("FitProcess: no valid candidate: %w", ErrProcessFit)
	}
	best.Evaluations = evals
	best.Termination = termination
	if termination != ProcessFitConverged {
		best.Diagnostics = append(best.Diagnostics, "Best valid approximate fit retained; convergence was not established.")
	}
	best.Diagnostics = append(best.Diagnostics, "Fit quality does not establish stability robustness or parameter uncertainty.")
	for i, tau := range best.Parameters.TimeConstants {
		if i > 0 && (tau-best.Parameters.TimeConstants[i-1])/tau < .05 {
			best.Diagnostics = append(best.Diagnostics, "Closely spaced poles weaken individual time-constant identifiability.")
		}
		if options.Structure.Zero && best.Parameters.ZeroTime > 0 && math.Abs(best.Parameters.ZeroTime/tau-1) < .05 {
			best.Diagnostics = append(best.Diagnostics, "Near pole-zero cancellation weakens physical parameter identifiability.")
		}
	}

	x := encodeProcessPoint(best.Parameters, coordinates)
	for i, v := range x {
		if v < 1e-3 || v > 1-1e-3 {
			best.ActiveBounds = append(best.ActiveBounds, coordinates[i].name)
		}
	}
	if err := ctx.Err(); err != nil {
		best.Termination = ProcessFitCanceled
		return nil, &ProcessFitCanceledError{Best: best, Err: err}
	}
	return best, nil
}

type processCoordinate struct {
	name     string
	min, max float64
	log      bool
}

func processCoordinates(s ProcessStructure, b ProcessBounds) []processCoordinate {
	var result []processCoordinate
	realPoles := s.Poles
	if s.UnderdampedPair {
		realPoles -= 2
	}
	for i := 0; i < realPoles; i++ {
		result = append(result, processCoordinate{fmt.Sprintf("tau%d", i+1), b.MinTimeConstant, b.MaxTimeConstant, true})
	}
	if s.UnderdampedPair {
		result = append(result, processCoordinate{"frequency", b.MinFrequency, b.MaxFrequency, true}, processCoordinate{"damping", b.MinDamping, b.MaxDamping, false})
	}
	if s.Zero {
		result = append(result, processCoordinate{"zero", b.MinZero, b.MaxZero, false})
	}
	if s.Delay {
		result = append(result, processCoordinate{"delay", b.MinDelay, b.MaxDelay, false})
	}
	return result
}
func decodeProcessPoint(x []float64, c []processCoordinate, s ProcessStructure) ProcessParameters {
	p := ProcessParameters{Gain: 1}
	for i, b := range c {
		v := b.min + x[i]*(b.max-b.min)
		if b.log {
			v = math.Exp(math.Log(b.min) + x[i]*math.Log(b.max/b.min))
		}
		switch b.name {
		case "frequency":
			p.NaturalFrequency = v
		case "damping":
			p.Damping = v
		case "zero":
			p.ZeroTime = v
		case "delay":
			p.Delay = v
		default:
			p.TimeConstants = append(p.TimeConstants, v)
		}
	}
	sort.Float64s(p.TimeConstants)
	return p
}
func encodeProcessPoint(p ProcessParameters, c []processCoordinate) []float64 {
	x := make([]float64, len(c))
	tau := 0
	for i, b := range c {
		var v float64
		switch b.name {
		case "frequency":
			v = p.NaturalFrequency
		case "damping":
			v = p.Damping
		case "zero":
			v = p.ZeroTime
		case "delay":
			v = p.Delay
		default:
			if tau < len(p.TimeConstants) {
				v = p.TimeConstants[tau]
			}
			tau++
		}
		if b.log {
			x[i] = math.Log(v/b.min) / math.Log(b.max/b.min)
		} else {
			x[i] = (v - b.min) / (b.max - b.min)
		}
	}
	return x
}
func defaultProcessParameters(d ProcessFitData, o ProcessFitOptions) ProcessParameters {
	p := ProcessParameters{Gain: 1, NaturalFrequency: 1 / math.Max(d.SampleTime, float64(d.TrainingSamples)*d.SampleTime/20), Damping: .5}
	n := o.Structure.Poles
	if o.Structure.UnderdampedPair {
		n -= 2
	}
	for i := 0; i < n; i++ {
		tau := float64(d.TrainingSamples) * d.SampleTime * float64(i+1) / 40
		p.TimeConstants = append(p.TimeConstants, math.Max(o.Bounds.MinTimeConstant, math.Min(o.Bounds.MaxTimeConstant, tau)))
	}
	p.NaturalFrequency = math.Max(o.Bounds.MinFrequency, math.Min(o.Bounds.MaxFrequency, p.NaturalFrequency))
	if o.Structure.Delay {
		p.Delay = (o.Bounds.MaxDelay-o.Bounds.MinDelay)*.1 + o.Bounds.MinDelay
	}
	return p
}
func validateProcessFit(d ProcessFitData, o ProcessFitOptions) (ProcessFitOptions, error) {
	n := len(d.Input)
	if n != len(d.Output) || n > 10000 || d.TrainingSamples < 20 || n-d.TrainingSamples < processMinHeldOut || !processFinite(d.SampleTime) || d.SampleTime <= 0 {
		return o, fmt.Errorf("require aligned data, 20 training samples, %d held-out samples and positive sample time (10000 maximum): %w", processMinHeldOut, ErrProcessData)
	}
	if o.ValidationInitialCondition == "" {
		o.ValidationInitialCondition = ProcessValidationContinuation
	}
	if o.ValidationInitialCondition != ProcessValidationContinuation && o.ValidationInitialCondition != ProcessValidationZero {
		return o, fmt.Errorf("validation initialization %q must be continuation or zero: %w", o.ValidationInitialCondition, ErrProcessData)
	}
	if o.ValidationInitializationSamples < 0 || n-d.TrainingSamples-o.ValidationInitializationSamples < processMinHeldOut {
		return o, fmt.Errorf("validation initialization prefix leaves fewer than %d held-out samples: %w", processMinHeldOut, ErrProcessData)
	}
	lo, hi := d.Input[0], d.Input[0]
	for i := range d.Input {
		if !processFinite(d.Input[i]) || !processFinite(d.Output[i]) {
			return o, fmt.Errorf("nonfinite sample %d: %w", i, ErrProcessData)
		}
		if i < d.TrainingSamples {
			lo = math.Min(lo, d.Input[i])
			hi = math.Max(hi, d.Input[i])
		}
	}
	if hi-lo <= 1e-12*math.Max(1, math.Max(math.Abs(lo), math.Abs(hi))) {
		return o, fmt.Errorf("training input is constant: %w", ErrProcessExcitation)
	}
	if o.Structure.Poles < 1 || o.Structure.Poles > 3 || (o.Structure.UnderdampedPair && o.Structure.Poles < 2) {
		return o, fmt.Errorf("choose one to three poles and a valid pair structure: %w", ErrProcessData)
	}
	if o.MaxEvaluations == 0 {
		o.MaxEvaluations = 2000
	}
	if o.Starts == 0 {
		o.Starts = 4
	}
	if o.MaxEvaluations < 20 || o.MaxEvaluations > 2000 || o.Starts < 1 || o.Starts > 8 {
		return o, fmt.Errorf("budget requires 20..2000 evaluations and 1..8 starts: %w", ErrProcessData)
	}
	duration := float64(d.TrainingSamples-1) * d.SampleTime
	b := &o.Bounds
	if b.MinTimeConstant == 0 && b.MaxTimeConstant == 0 {
		b.MinTimeConstant = d.SampleTime / 10
		b.MaxTimeConstant = duration * 10
	}
	if b.MinDelay == 0 && b.MaxDelay == 0 {
		b.MaxDelay = duration / 4
	}
	if b.MinZero == 0 && b.MaxZero == 0 {
		b.MinZero = -duration
		b.MaxZero = duration
	}
	if b.MinGain == 0 && b.MaxGain == 0 {
		b.MinGain = -1e6
		b.MaxGain = 1e6
	}
	if b.MinFrequency == 0 && b.MaxFrequency == 0 {
		b.MinFrequency = 1 / (duration * 10)
		b.MaxFrequency = 10 / d.SampleTime
	}
	if b.MinDamping == 0 && b.MaxDamping == 0 {
		b.MinDamping = .01
		b.MaxDamping = .999
	}
	for _, value := range []float64{b.MinTimeConstant, b.MaxTimeConstant, b.MinDelay, b.MaxDelay, b.MinZero, b.MaxZero, b.MinGain, b.MaxGain, b.MinFrequency, b.MaxFrequency, b.MinDamping, b.MaxDamping} {
		if !processFinite(value) {
			return o, fmt.Errorf("all bounds must be finite: %w", ErrProcessData)
		}
	}
	if !processFinite(b.MinGain) || !processFinite(b.MaxGain) || b.MinGain >= b.MaxGain {
		return o, fmt.Errorf("invalid gain bounds: %w", ErrProcessData)
	}
	for _, c := range processCoordinates(o.Structure, *b) {
		if !processFinite(c.min) || !processFinite(c.max) || c.min >= c.max || (c.log && c.min <= 0) {
			return o, fmt.Errorf("invalid %s bounds: %w", c.name, ErrProcessData)
		}
	}
	if b.MinDelay < 0 || b.MinDamping <= 0 || b.MaxDamping >= 1 {
		return o, fmt.Errorf("invalid delay/damping bounds: %w", ErrProcessData)
	}
	if o.Initial != nil {
		if _, err := processSystem(o.Structure, *o.Initial); err != nil {
			return o, err
		}
		for _, v := range encodeProcessPoint(*o.Initial, processCoordinates(o.Structure, *b)) {
			if !processFinite(v) || v < 0 || v > 1 {
				return o, fmt.Errorf("initial parameters outside bounds: %w", ErrProcessData)
			}
		}
	}
	return o, nil
}
func processFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }

// processMinHeldOut keeps at least two lag-one residual pairs for
// ResidualLagOne.
const processMinHeldOut = 3

func evaluateProcessFit(d ProcessFitData, o ProcessFitOptions, p ProcessParameters, fixedGain *float64) (*ProcessFitResult, float64, error) {
	system, err := processSystem(o.Structure, p)
	if err != nil {
		return nil, 0, err
	}
	forced, initial, err := processResponseBasis(system, d.Input, d.SampleTime, p.Delay, o.EstimateInitialState)
	if err != nil {
		return nil, 0, err
	}
	if o.ValidationInitialCondition == ProcessValidationZero {
		separate, _, err := processResponseBasis(system, d.Input[d.TrainingSamples:], d.SampleTime, p.Delay, false)
		if err != nil {
			return nil, 0, err
		}
		copy(forced[d.TrainingSamples:], separate)
		for _, column := range initial {
			clear(column[d.TrainingSamples:])
		}
	}
	startColumn := 1
	if fixedGain != nil {
		startColumn = 0
	}
	columns := startColumn + len(initial)
	if o.EstimateOffset {
		columns++
	}
	if d.TrainingSamples < 10*max(1, columns) {
		return nil, 0, fmt.Errorf("%d training samples for %d profiled coefficients: %w", d.TrainingSamples, columns, ErrProcessExcitation)
	}
	coefficients := mat.NewDense(max(1, columns), 1, nil)
	condition := 1.0
	if columns > 0 {
		design := mat.NewDense(d.TrainingSamples, columns, nil)
		target := mat.NewDense(d.TrainingSamples, 1, append([]float64(nil), d.Output[:d.TrainingSamples]...))
		for i := 0; i < d.TrainingSamples; i++ {
			if fixedGain == nil {
				design.Set(i, 0, forced[i])
			} else {
				target.Set(i, 0, target.At(i, 0)-*fixedGain*forced[i])
			}
			for j := range initial {
				design.Set(i, j+startColumn, initial[j][i])
			}
			if o.EstimateOffset {
				design.Set(i, columns-1, 1)
			}
		}
		var qr mat.QR
		qr.Factorize(design)
		condition = qr.Cond()
		if !processFinite(condition) || condition > 1e12 {
			return nil, 0, fmt.Errorf("profiling design condition %g: %w", condition, ErrProcessExcitation)
		}
		if err = qr.SolveTo(coefficients, false, target); err != nil {
			return nil, 0, err
		}
	}
	if fixedGain == nil {
		p.Gain = coefficients.At(0, 0)
	} else {
		p.Gain = *fixedGain
	}
	if p.Gain < o.Bounds.MinGain || p.Gain > o.Bounds.MaxGain {
		return nil, 0, fmt.Errorf("gain %g outside bounds [%g, %g]: %w", p.Gain, o.Bounds.MinGain, o.Bounds.MaxGain, ErrProcessFit)
	}
	if p.Gain == 0 && len(initial) > 0 {
		return nil, 0, fmt.Errorf("zero gain makes the initial state unidentifiable: %w", ErrProcessFit)
	}

	result := &ProcessFitResult{Parameters: p, Structure: o.Structure, Bounds: o.Bounds, Condition: condition, Predicted: make([]float64, len(d.Input)), Residuals: make([]float64, len(d.Input))}
	if o.EstimateOffset {
		result.Offset = coefficients.At(columns-1, 0)
	}
	for j := range initial {
		result.InitialState = append(result.InitialState, coefficients.At(j+startColumn, 0)/p.Gain)
	}
	for i := range d.Input {
		v := p.Gain*forced[i] + result.Offset
		for j := range initial {
			v += coefficients.At(j+startColumn, 0) * initial[j][i]
		}
		result.Predicted[i] = v
		result.Residuals[i] = d.Output[i] - v
		if !processFinite(v) {
			return nil, 0, fmt.Errorf("nonfinite prediction at sample %d: %w", i, ErrProcessFit)
		}
	}
	var sse float64
	for _, r := range result.Residuals[:d.TrainingSamples] {
		sse += r * r
	}
	value := sse / float64(d.TrainingSamples)
	result.TrainingNRMSE = processNRMSE(d.Output[:d.TrainingSamples], result.Residuals[:d.TrainingSamples])
	validationStart := d.TrainingSamples + o.ValidationInitializationSamples
	result.ValidationNRMSE = processNRMSE(d.Output[validationStart:], result.Residuals[validationStart:])
	mean := 0.0
	for _, v := range d.Output[:d.TrainingSamples] {
		mean += v
	}
	mean /= float64(d.TrainingSamples)
	baseline := make([]float64, len(d.Output)-validationStart)
	for i := range baseline {
		baseline[i] = d.Output[validationStart+i] - mean
	}
	result.ValidationMeanNRMSE = processNRMSE(d.Output[validationStart:], baseline)
	result.residualLagOne, result.residualLagOK = processResidualLagOne(result.Residuals[validationStart:])
	result.System, err = p.System(o.Structure)
	if err != nil {
		return nil, 0, err
	}
	return result, value, nil
}

// processResidualLagOne is the Pearson correlation of (r[i], r[i-1]) pairs;
// ok is false with fewer than two pairs or zero variance.
func processResidualLagOne(r []float64) (float64, bool) {
	pairs := len(r) - 1
	if pairs < 2 {
		return 0, false
	}
	var current, lagged float64
	for i := 1; i < len(r); i++ {
		current += r[i]
		lagged += r[i-1]
	}
	current /= float64(pairs)
	lagged /= float64(pairs)
	var cross, currentPower, laggedPower float64
	for i := 1; i < len(r); i++ {
		a, b := r[i]-current, r[i-1]-lagged
		cross += a * b
		currentPower += a * a
		laggedPower += b * b
	}
	if currentPower == 0 || laggedPower == 0 {
		return 0, false
	}
	return math.Max(-1, math.Min(1, cross/math.Sqrt(currentPower*laggedPower))), true
}

func processNRMSE(values, residuals []float64) float64 {
	mean := 0.0
	for _, v := range values {
		mean += v
	}
	mean /= float64(len(values))
	var spread, sse, energy float64
	for i, v := range values {
		spread += (v - mean) * (v - mean)
		energy += v * v
		sse += residuals[i] * residuals[i]
	}
	return math.Sqrt(sse / math.Max(spread, math.Max(energy*1e-12, 1e-24)))
}

// processResponseBasis splits each sample interval at the delayed hold switch.
// Fractional delay therefore stays exact for piecewise-constant measured input.
// With an estimated initial state the input is held at its first sample before
// the record, so the delay line starts consistent with that state; otherwise
// the model starts at rest with zero input.
func processResponseBasis(system *System, input []float64, dt, delay float64, estimateInitial bool) ([]float64, [][]float64, error) {
	n, _, _ := system.Dims()
	if delay/dt >= float64(len(input)) {
		return nil, nil, fmt.Errorf("delay %g spans the whole record: %w", delay, ErrProcessExcitation)
	}
	whole := int(math.Floor(delay / dt))
	fraction := delay - float64(whole)*dt
	if fraction < dt*1e-12 {
		fraction = 0
	}
	full, err := system.C2D(dt, C2DOptions{})
	if err != nil {
		return nil, nil, err
	}
	first, last := full, full
	if fraction > 0 {
		first, err = system.C2D(fraction, C2DOptions{})
		if err != nil {
			return nil, nil, err
		}
		last, err = system.C2D(dt-fraction, C2DOptions{})
		if err != nil {
			return nil, nil, err
		}
	}
	state := make([]float64, n)
	next := make([]float64, n)
	forced := make([]float64, len(input))
	var basis [][]float64
	var initialStates [][]float64
	if estimateInitial {
		basis = make([][]float64, n)
		initialStates = make([][]float64, n)
		for j := range n {
			basis[j] = make([]float64, len(input))
			initialStates[j] = make([]float64, n)
			initialStates[j][j] = 1
		}
	}
	at := func(i int) float64 {
		switch {
		case i >= len(input), i < 0 && !estimateInitial:
			return 0
		case i < 0:
			return input[0]
		}
		return input[i]
	}
	advance := func(target, current []float64, model *System, u float64) {
		for i := range n {
			v := model.B.At(i, 0) * u
			for j := range n {
				v += model.A.At(i, j) * current[j]
			}
			target[i] = v
		}
	}
	for k := range input {
		index := k - whole
		if fraction > 0 {
			index--
		}
		v := system.D.At(0, 0) * at(index)
		for j := range n {
			v += system.C.At(0, j) * state[j]
		}
		forced[k] = v
		for j := range basis {
			for i := range n {
				basis[j][k] += system.C.At(0, i) * initialStates[j][i]
			}
			advance(next, initialStates[j], full, 0)
			copy(initialStates[j], next)
		}
		if fraction > 0 {
			advance(next, state, first, at(k-whole-1))
			copy(state, next)
			advance(next, state, last, at(k-whole))
		} else {
			advance(next, state, full, at(k-whole))
		}
		copy(state, next)
	}
	return forced, basis, nil
}

// EvaluateProcess evaluates manually edited physical parameters without
// optimizing them. Optional initial-state/offset profiling remains training-only.
// Errors follow FitProcess.
func EvaluateProcess(ctx context.Context, data ProcessFitData, options ProcessFitOptions, parameters ProcessParameters) (*ProcessFitResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("EvaluateProcess: nil context: %w", ErrInvalidArgument)
	}
	options.Initial = &parameters
	options, err := validateProcessFit(data, options)
	if err != nil {
		return nil, fmt.Errorf("EvaluateProcess: %w", err)
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("EvaluateProcess: %w", err)
	}
	if parameters.Gain < options.Bounds.MinGain || parameters.Gain > options.Bounds.MaxGain {
		return nil, fmt.Errorf("EvaluateProcess: edited gain %g is outside bounds: %w", parameters.Gain, ErrProcessData)
	}
	result, _, err := evaluateProcessFit(data, options, parameters, &parameters.Gain)
	if err != nil {
		return nil, fmt.Errorf("EvaluateProcess: %w", err)
	}
	result.Termination = ProcessFitManual
	result.Evaluations = 1
	return result, nil
}
