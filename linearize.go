package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

type NonlinearModel struct {
	F func(x, u *mat.VecDense) *mat.VecDense
	H func(x, u *mat.VecDense) *mat.VecDense
	N int
	M int
	P int
}

func Linearize(model *NonlinearModel, x0, u0 *mat.VecDense) (*System, error) {
	if model == nil {
		return nil, fmt.Errorf("controlsys: nil model: %w", ErrDimensionMismatch)
	}
	if model.F == nil || model.H == nil {
		return nil, fmt.Errorf("controlsys: nil F or H function: %w", ErrDimensionMismatch)
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
