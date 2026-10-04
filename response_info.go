package controlsys

import (
	"errors"
	"fmt"
	"math"
)

type StepInfoOptions struct {
	RiseTimeLimits    [2]float64
	SettlingThreshold float64
	SteadyStateValue  []float64
}

type StepInfoResult struct {
	Metrics    []StepMetric
	OutputName []string
}

type StepMetric struct {
	RiseTime         float64
	SettlingTime     float64
	Overshoot        float64
	Undershoot       float64
	Peak             float64
	PeakTime         float64
	SteadyStateValue float64
	Settled          bool
}

func StepInfo(resp *TimeResponse, opts *StepInfoOptions) (*StepInfoResult, error) {
	if resp == nil || resp.Y == nil {
		return nil, fmt.Errorf("StepInfo: response must not be nil: %w", ErrDimensionMismatch)
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
	if cfg.SteadyStateValue != nil && len(cfg.SteadyStateValue) != rows {
		return nil, fmt.Errorf("StepInfo: steady-state values length %d does not match response rows %d: %w", len(cfg.SteadyStateValue), rows, ErrDimensionMismatch)
	}
	metrics := make([]StepMetric, rows)
	for row := range rows {
		metrics[row] = stepMetricForRow(resp, row, cfg)
	}
	return &StepInfoResult{Metrics: metrics, OutputName: copyStringSlice(resp.OutputName)}, nil
}

// StepInfoForSystem simulates the step response of a stable sys and returns
// its step metrics. Unstable models return ErrUnstable. Continuous models with
// internal delays have no finite pole test, so the stability gate is skipped
// and the metrics come from the simulated response, as MATLAB recommends
// assessing such models with step.
func StepInfoForSystem(sys *System, tFinal float64, opts *StepInfoOptions) (*StepInfoResult, error) {
	if sys == nil {
		return nil, fmt.Errorf("StepInfoForSystem: system must not be nil: %w", ErrDimensionMismatch)
	}
	stable, err := sys.IsStable()
	switch {
	case errors.Is(err, ErrContinuousInternalDelay):
	case err != nil:
		return nil, fmt.Errorf("StepInfoForSystem: %w", err)
	case !stable:
		return nil, fmt.Errorf("StepInfoForSystem: model is unstable: %w", ErrUnstable)
	}
	resp, err := Step(sys, tFinal)
	if err != nil {
		return nil, err
	}
	return StepInfo(resp, opts)
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

func validateStepInfoTime(t []float64) error {
	for k := 1; k < len(t); k++ {
		if t[k] <= t[k-1] {
			return fmt.Errorf("StepInfo: time vector must be strictly increasing at index %d: %w", k, ErrDimensionMismatch)
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
	rise := math.NaN()
	if delta != 0 {
		lo := initial + delta*cfg.RiseTimeLimits[0]
		hi := initial + delta*cfg.RiseTimeLimits[1]
		tLo := crossingTime(resp, row, lo, direction)
		tHi := crossingTime(resp, row, hi, direction)
		if !math.IsNaN(tLo) && !math.IsNaN(tHi) {
			rise = tHi - tLo
		}
	}

	settling, settled := settlingTime(resp, row, final, cfg.SettlingThreshold*scale)
	overshoot := 0.0
	if beyond := direction * (peak - final); beyond > 0 && delta != 0 {
		overshoot = 100 * beyond / math.Abs(delta)
	}
	undershoot := rowUndershoot(resp, row, initial, direction, scale)

	return StepMetric{
		RiseTime:         rise,
		SettlingTime:     settling,
		Overshoot:        overshoot,
		Undershoot:       undershoot,
		Peak:             peak,
		PeakTime:         peakTime,
		SteadyStateValue: final,
		Settled:          settled,
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
		if score >= best {
			best = score
			peak = y
			peakTime = resp.T[k]
		}
	}
	return peak, peakTime
}

func crossingTime(resp *TimeResponse, row int, level, direction float64) float64 {
	_, cols := resp.Y.Dims()
	prev := resp.Y.At(row, 0)
	for k := 1; k < cols; k++ {
		curr := resp.Y.At(row, k)
		if direction*(curr-level) >= 0 && direction*(prev-level) < 0 {
			if curr == prev {
				return resp.T[k]
			}
			alpha := (level - prev) / (curr - prev)
			return resp.T[k-1] + alpha*(resp.T[k]-resp.T[k-1])
		}
		prev = curr
	}
	return math.NaN()
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
		return math.NaN(), false
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
