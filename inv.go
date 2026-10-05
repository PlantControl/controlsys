package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// Inv returns the inverse model of a square system, like MATLAB inv(sys):
// for D nonsingular the realization (A - B D⁻¹ C, B D⁻¹, -D⁻¹ C, D⁻¹), with
// input and output names swapped. A non-square model returns
// ErrDimensionMismatch, a singular D ErrSingularTransform, a delayed model
// ErrDelayUnsupported (use Pade or AbsorbDelay first) and a descriptor model
// ErrDescriptorUnsupported.
// See https://www.mathworks.com/help/control/ref/dynamicsystem.inv.html.
func Inv(sys *System) (*System, error) {
	if err := requireSystem("Inv", sys); err != nil {
		return nil, err
	}
	if err := newDescriptorPolicy(sys).requireStandard("Inv"); err != nil {
		return nil, err
	}
	if sys.HasDelay() {
		return nil, fmt.Errorf("Inv: use Pade or AbsorbDelay first: %w", ErrDelayUnsupported)
	}
	n, m, p := sys.Dims()
	if m != p {
		return nil, fmt.Errorf("Inv: system must be square (p=%d, m=%d): %w", p, m, ErrDimensionMismatch)
	}

	if m == 0 {
		return sys.Copy(), nil
	}

	var luD mat.LU
	luD.Factorize(sys.D)
	if luNearSingular(&luD) {
		return nil, fmt.Errorf("Inv: D matrix is singular: %w", ErrSingularTransform)
	}

	Dinv := mat.NewDense(m, m, nil)
	ones := make([]float64, m)
	for i := range ones {
		ones[i] = 1
	}
	eye := mat.NewDiagDense(m, ones)
	if err := luD.SolveTo(Dinv, false, eye); err != nil {
		return nil, fmt.Errorf("Inv: %w", ErrSingularTransform)
	}

	if n == 0 {
		result, err := NewGain(Dinv, sys.Dt)
		if err != nil {
			return nil, err
		}
		result.InputName = copyStringSlice(sys.OutputName)
		result.OutputName = copyStringSlice(sys.InputName)
		return result, nil
	}

	var DinvC, BDinvC, Ainv, BDinv, DinvCneg mat.Dense

	DinvC.Mul(Dinv, sys.C)
	BDinvC.Mul(sys.B, &DinvC)

	Ainv.Sub(sys.A, &BDinvC)
	BDinv.Mul(sys.B, Dinv)
	DinvCneg.Scale(-1, &DinvC)

	result, err := newNoCopy(
		mat.DenseCopyOf(&Ainv),
		mat.DenseCopyOf(&BDinv),
		mat.DenseCopyOf(&DinvCneg),
		Dinv,
		sys.Dt,
	)
	if err != nil {
		return nil, err
	}

	result.InputName = copyStringSlice(sys.OutputName)
	result.OutputName = copyStringSlice(sys.InputName)
	result.StateName = copyStringSlice(sys.StateName)

	return result, nil
}
