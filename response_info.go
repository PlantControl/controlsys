package controlsys

import (
	"errors"
	"fmt"
	"math"
)

// StepInfoOptions holds the optional arguments of StepInfo, as MATLAB
// stepinfo name-value pairs. Zero values select the defaults:
// RiseTimeLimits [0.1 0.9], SettlingThreshold 0.02 and SteadyStateValue the
// last sample of each row.
type StepInfoOptions struct {
	RiseTimeLimits    [2]float64
	SettlingThreshold float64
	SteadyStateValue  []float64
}

// StepInfoResult holds one StepMetric per response row, in the row order of
// TimeResponse.Y (input-major for Step: row input*p+output).
type StepInfoResult struct {
	Metrics    []StepMetric
	OutputName []string
}

// StepMetric holds the step-response characteristics of one row, as MATLAB
// stepinfo. Overshoot and Undershoot are percentages of the step size; rise
// and settling times, which may not exist within the sampled horizon, are
// reported by RiseTime and SettlingTime.
type StepMetric struct {
	Overshoot        float64
	Undershoot       float64
	Peak             float64
	PeakTime         float64
	SteadyStateValue float64

	riseTime     float64
	risen        bool
	settlingTime float64
	settled      bool
}

// RiseTime returns the time to go from the lower to the upper rise-time
// limit. ok is false when the response does not cross both limits within
// the horizon or the step size is zero (MATLAB reports NaN).
func (m StepMetric) RiseTime() (t float64, ok bool) { return m.riseTime, m.risen }

// SettlingTime returns the time after which the response stays within the
// settling band around SteadyStateValue. ok is false when it has not
// settled by the last sample (MATLAB reports NaN); lengthen tFinal.
func (m StepMetric) SettlingTime() (t float64, ok bool) { return m.settlingTime, m.settled }

// StepInfo computes the step-response characteristics of every row of resp,
// as MATLAB stepinfo (https://www.mathworks.com/help/control/ref/dynamicsystem.stepinfo.html).
// resp.T must be strictly increasing with at least 2 samples.

func StepInfo(resp *TimeResponse, opts *StepInfoOptions) (*StepInfoResult, error) {
	if resp == nil || resp.Y == nil {
		return nil, fmt.Errorf("StepInfo: response must not be nil: %w", ErrInvalidArgument)
	}
	rows, cols := resp.Y.Dims()
	if cols != len(resp.T) {
		return nil, fmt.Errorf("StepInfo: response has %d samples but time vector has %d: %w", cols, len(resp.T), ErrDimensionMismatch)
	}
	if cols < 2 {
		return nil, fmt.Errorf("StepInfo: need at least 2 response samples: %w", ErrDimensionMismatch)
	}
	if err := validateStepInfoTime(resp.T); err != nil {
		return nil, err
	}

	cfg := defaultStepInfoOptions(opts)
	if err := cfg.validate(rows); err != nil {
		return nil, err
	}
	metrics := make([]StepMetric, rows)
	for row := range rows {
		metrics[row] = stepMetricForRow(resp, row, cfg)
	}
	return &StepInfoResult{Metrics: metrics, OutputName: copyStringSlice(resp.OutputName)}, nil
}

// StepInfoForSystem simulates the step response of a stable sys and returns
// its step metrics. Unstable models, including continuous models with
// internal delays that IsStable decides are unstable, return ErrUnstable.
// When IsStable cannot decide a delay model (ErrDelayUnsupported), the
// stability gate is skipped and the metrics come from the simulated
// response, as MATLAB recommends assessing such models with step.
func StepInfoForSystem(sys *System, tFinal float64, opts *StepInfoOptions) (*StepInfoResult, error) {
	if err := requireSystem("StepInfoForSystem", sys); err != nil {
		return nil, err
	}
	stable, err := sys.IsStable()
	switch {
	case errors.Is(err, ErrDelayUnsupported):
	case err != nil:
		return nil, fmt.Errorf("StepInfoForSystem: %w", err)
	case !stable:
		return nil, fmt.Errorf("StepInfoForSystem: model is unstable: %w", ErrUnstable)
	}
	resp, err := Step(sys, tFinal)
	if err != nil {
		return nil, fmt.Errorf("StepInfoForSystem: %w", err)
	}
	if opts == nil || opts.SteadyStateValue == nil {
		gain, err := sys.DCGain()
		if err != nil {
			return nil, fmt.Errorf("StepInfoForSystem: %w", err)
		}
		cfg := StepInfoOptions{}
		if opts != nil {
			cfg = *opts
		}
		_, m, p := sys.Dims()
		cfg.SteadyStateValue = make([]float64, m*p)
		for input := range m {
			for output := range p {
				cfg.SteadyStateValue[input*p+output] = gain.At(output, input)
			}
		}
		opts = &cfg
	}
	info, err := StepInfo(resp, opts)
	if err != nil {
		return nil, fmt.Errorf("StepInfoForSystem: %w", err)
	}
	return info, nil
}

func defaultStepInfoOptions(opts *StepInfoOptions) StepInfoOptions {
	cfg := StepInfoOptions{
		RiseTimeLimits:    [2]float64{0.1, 0.9},
		SettlingThreshold: 0.02,
	}
	if opts != nil {
		cfg = *opts
	}
	if cfg.RiseTimeLimits == [2]float64{} {
		cfg.RiseTimeLimits = [2]float64{0.1, 0.9}
	}
	if cfg.SettlingThreshold == 0 {
		cfg.SettlingThreshold = 0.02
	}
	return cfg
}

func (cfg StepInfoOptions) validate(rows int) error {
	lo, hi := cfg.RiseTimeLimits[0], cfg.RiseTimeLimits[1]
	if !(lo >= 0 && lo < hi && hi <= 1) {
		return fmt.Errorf("StepInfo: rise-time limits must satisfy 0 <= lo < hi <= 1, got [%g %g]: %w", lo, hi, ErrInvalidArgument)
	}
	if !(cfg.SettlingThreshold > 0 && cfg.SettlingThreshold < 1) {
		return fmt.Errorf("StepInfo: settling threshold must be in (0, 1), got %g: %w", cfg.SettlingThreshold, ErrInvalidArgument)
	}
	if cfg.SteadyStateValue == nil {
		return nil
	}
	if len(cfg.SteadyStateValue) != rows {
		return fmt.Errorf("StepInfo: steady-state values length %d does not match response rows %d: %w", len(cfg.SteadyStateValue), rows, ErrDimensionMismatch)
	}
	for row, v := range cfg.SteadyStateValue {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return fmt.Errorf("StepInfo: steady-state value %d must be finite, got %g: %w", row, v, ErrInvalidArgument)
		}
	}
	return nil
}

func validateStepInfoTime(t []float64) error {
	for k := 1; k < len(t); k++ {
		if t[k] <= t[k-1] {
			return fmt.Errorf("StepInfo: time vector must be strictly increasing at index %d: %w", k, ErrInvalidArgument)
		}
	}
	return nil
}

func stepMetricForRow(resp *TimeResponse, row int, cfg StepInfoOptions) StepMetric {
	_, cols := resp.Y.Dims()
	initial := resp.Y.At(row, 0)
	final := resp.Y.At(row, cols-1)
	if cfg.SteadyStateValue != nil {
		final = cfg.SteadyStateValue[row]
	}
	delta := final - initial
	scale := math.Abs(delta)
	if scale == 0 {
		scale = math.Max(math.Abs(final), 1)
	}
	direction := 1.0
	if delta < 0 {
		direction = -1
	}

	peak, peakTime := directionalPeak(resp, row, direction)
	var rise float64
	var risen bool
	if delta != 0 {
		lo := initial + delta*cfg.RiseTimeLimits[0]
		hi := initial + delta*cfg.RiseTimeLimits[1]
		tLo, okLo := crossingTime(resp, row, lo, direction)
		tHi, okHi := crossingTime(resp, row, hi, direction)
		if okLo && okHi {
			rise, risen = tHi-tLo, true
		}
	}

	settling, settled := settlingTime(resp, row, final, cfg.SettlingThreshold*scale)
	overshoot := 0.0
	if beyond := direction * (peak - final); beyond > 0 && delta != 0 {
		overshoot = 100 * beyond / math.Abs(delta)
	}
	undershoot := rowUndershoot(resp, row, initial, direction, scale)

	return StepMetric{
		Overshoot:        overshoot,
		Undershoot:       undershoot,
		Peak:             peak,
		PeakTime:         peakTime,
		SteadyStateValue: final,
		riseTime:         rise,
		risen:            risen,
		settlingTime:     settling,
		settled:          settled,
	}
}

func directionalPeak(resp *TimeResponse, row int, direction float64) (float64, float64) {
	_, cols := resp.Y.Dims()
	peak := resp.Y.At(row, 0)
	peakTime := resp.T[0]
	best := direction * peak
	for k := 1; k < cols; k++ {
		y := resp.Y.At(row, k)
		score := direction * y
		if score > best {
			best = score
			peak = y
			peakTime = resp.T[k]
		}
	}
	return peak, peakTime
}

func crossingTime(resp *TimeResponse, row int, level, direction float64) (float64, bool) {
	_, cols := resp.Y.Dims()
	prev := resp.Y.At(row, 0)
	for k := 1; k < cols; k++ {
		curr := resp.Y.At(row, k)
		if direction*(curr-level) >= 0 && direction*(prev-level) < 0 {
			if curr == prev {
				return resp.T[k], true
			}
			alpha := (level - prev) / (curr - prev)
			return resp.T[k-1] + alpha*(resp.T[k]-resp.T[k-1]), true
		}
		prev = curr
	}
	return 0, false
}

func settlingTime(resp *TimeResponse, row int, final, band float64) (float64, bool) {
	_, cols := resp.Y.Dims()
	lastOutside := -1
	for k := range cols {
		if math.Abs(resp.Y.At(row, k)-final) > band {
			lastOutside = k
		}
	}
	if lastOutside == -1 {
		return resp.T[0], true
	}
	if lastOutside == cols-1 {
		return 0, false
	}
	return resp.T[lastOutside+1], true
}

func rowUndershoot(resp *TimeResponse, row int, initial, direction, scale float64) float64 {
	_, cols := resp.Y.Dims()
	worst := 0.0
	for k := range cols {
		opposite := direction * (initial - resp.Y.At(row, k))
		if opposite > worst {
			worst = opposite
		}
	}
	if worst == 0 {
		return 0
	}
	return 100 * worst / scale
}
