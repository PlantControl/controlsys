package controlsys

import (
	"fmt"
	"math/cmplx"
)

type ModsepResult struct {
	Slow *System
	Fast *System
}

// Modsep splits sys = Slow + Fast around cutoff (MATLAB freqsep). Slow holds
// the modes with natural frequency below cutoff; Fast holds the rest, the
// feedthrough D and, for singular-E descriptors, the infinite modes.
// A part without states on a model with no inputs or no outputs would be a
// p×0 or 0×m static gain, which cannot be stored: ErrDimensionMismatch.
// See https://www.mathworks.com/help/control/ref/dynamicsystem.freqsep.html.
func Modsep(sys *System, cutoff float64) (*ModsepResult, error) {
	if err := requireSystem("Modsep", sys); err != nil {
		return nil, err
	}
	if !(cutoff > 0) {
		return nil, fmt.Errorf("Modsep: cutoff must be positive, got %g: %w", cutoff, ErrInvalidArgument)
	}

	isSlow := func(ev complex128) bool {
		return cmplx.Abs(ev) < cutoff
	}

	slow, fast, err := decomposeByEigenvalues(sys, isSlow, false)
	if err != nil {
		return nil, err
	}

	return &ModsepResult{Slow: slow, Fast: fast}, nil
}
