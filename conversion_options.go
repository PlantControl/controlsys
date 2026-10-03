package controlsys

import (
	"errors"
	"fmt"
	"math"
)

var ErrInvalidConversionOptions = errors.New("controlsys: invalid conversion options")

// D2COptions selects the inverse discretization method. Zero values select ZOH.
// PrewarpFrequency is in rad/s, applies only to Tustin, and must be below Nyquist.
type D2COptions struct {
	Method           C2DMethod
	PrewarpFrequency float64
}

func validateConversionSampleTime(dt float64) error {
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return ErrInvalidSampleTime
	}
	return nil
}

func validatePrewarp(dt float64, method C2DMethod, frequency float64) error {
	if frequency < 0 || math.IsNaN(frequency) || math.IsInf(frequency, 0) {
		return fmt.Errorf("prewarp frequency must be finite and nonnegative: %w", ErrInvalidConversionOptions)
	}
	if frequency == 0 {
		return nil
	}
	if method != C2DMethodTustin {
		return fmt.Errorf("prewarp frequency requires Tustin: %w", ErrInvalidConversionOptions)
	}
	if frequency*dt >= math.Pi {
		return fmt.Errorf("prewarp frequency must be below Nyquist: %w", ErrInvalidConversionOptions)
	}
	return nil
}

func tustinBeta(dt, prewarp float64) (float64, error) {
	if err := validateConversionSampleTime(dt); err != nil {
		return 0, err
	}
	if err := validatePrewarp(dt, C2DMethodTustin, prewarp); err != nil {
		return 0, err
	}
	beta := 2 / dt
	if x := prewarp * (dt / 2); prewarp > 0 && x > 0 {
		beta = prewarp / math.Tan(x)
	}
	if beta <= 0 || math.IsInf(beta, 0) || math.IsNaN(beta) || math.IsInf(2*beta, 0) {
		return 0, fmt.Errorf("Tustin scaling overflows: %w", ErrOverflow)
	}
	return beta, nil
}

func normalizeC2DOptions(dt float64, opts C2DOptions) (C2DOptions, error) {
	if err := validateConversionSampleTime(dt); err != nil {
		return opts, err
	}
	if opts.Method == "" {
		opts.Method = C2DMethodZOH
	}
	switch opts.Method {
	case C2DMethodZOH, C2DMethodTustin, C2DMethodFOH, C2DMethodImpulse, C2DMethodMatched, C2DMethodLeastSquares:
	default:
		return opts, fmt.Errorf("unsupported conversion method %q: %w", opts.Method, ErrInvalidConversionOptions)
	}
	if err := validatePrewarp(dt, opts.Method, opts.PrewarpFrequency); err != nil {
		return opts, err
	}
	if opts.ThiranOrder < 0 {
		return opts, fmt.Errorf("negative Thiran order: %w", ErrInvalidConversionOptions)
	}
	if opts.ThiranOrder > 0 && opts.Method != C2DMethodTustin && opts.Method != C2DMethodMatched {
		return opts, fmt.Errorf("Thiran order requires Tustin or matched: %w", ErrInvalidConversionOptions)
	}
	if opts.FitOrder < 0 || (opts.FitOrder != 0 && opts.Method != C2DMethodLeastSquares) {
		return opts, fmt.Errorf("fit order requires least-squares and must be nonnegative: %w", ErrInvalidConversionOptions)
	}
	if opts.DelayModeling == "" {
		opts.DelayModeling = C2DDelayModelingInternal
	}
	switch opts.DelayModeling {
	case C2DDelayModelingState, C2DDelayModelingInternal:
	default:
		return opts, fmt.Errorf("unsupported delay modeling %q: %w", opts.DelayModeling, ErrInvalidConversionOptions)
	}
	return opts, nil
}
