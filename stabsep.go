package controlsys

import "fmt"

// StabsepResult is the result of Stabsep: sys = Stable + Unstable.
type StabsepResult struct {
	Stable   *System
	Unstable *System
}

// Stabsep splits sys = Stable + Unstable (MATLAB stabsep). Stable holds the
// modes strictly inside the stability boundary, is always proper and carries
// the feedthrough D; Unstable holds the remaining finite modes and, for
// singular-E descriptors, the infinite (nondynamic/improper) modes.
// A part without states on a model with no inputs or no outputs would be a
// p×0 or 0×m static gain, which cannot be stored: ErrDimensionMismatch.
// See https://www.mathworks.com/help/control/ref/dynamicsystem.stabsep.html.
func Stabsep(sys *System) (*StabsepResult, error) {
	if err := requireFiniteSystem("Stabsep", sys); err != nil {
		return nil, err
	}
	isStable := func(ev complex128) bool {
		return poleInsideStabilityBoundary(ev, sys.IsContinuous(), poleStabilityTolerance(ev))
	}

	stable, unstable, err := decomposeByEigenvalues("Stabsep", sys, isStable, true)
	if err != nil {
		return nil, fmt.Errorf("Stabsep: %w", err)
	}

	return &StabsepResult{Stable: stable, Unstable: unstable}, nil
}
