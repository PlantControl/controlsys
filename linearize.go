package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// NonlinearModel is a continuous model x' = F(x, u), y = H(x, u) with N
// states, M inputs and P outputs.
type NonlinearModel struct {
	F func(x, u *mat.VecDense) *mat.VecDense
	H func(x, u *mat.VecDense) *mat.VecDense
	N int
	M int
	P int
}

// Linearize returns the continuous state-space model of model about the
// operating point (x0, u0) from central finite differences of F and H, the
// numerical counterpart of Simulink linearize. A nil model, F, H, x0 or u0
// returns ErrInvalidArgument; vectors of the wrong length, or F/H results of
// the wrong length, ErrDimensionMismatch.
func Linearize(model *NonlinearModel, x0, u0 *mat.VecDense) (*System, error) {
	if model == nil {
		return nil, fmt.Errorf("Linearize: model is nil: %w", ErrInvalidArgument)
	}
	if model.F == nil || model.H == nil {
		return nil, fmt.Errorf("Linearize: F or H is nil: %w", ErrInvalidArgument)
	}
	contract := newLocalApproximationContract("Linearize", model.N, model.M, model.P)
	if err := contract.validateOperatingPoint(x0, u0); err != nil {
		return nil, err
	}
	A, B, C, D, err := finiteDifferenceLocalModel(contract, model.F, model.H, x0, u0)
	if err != nil {
		return nil, err
	}

	return New(A, B, C, D, 0)
}
