package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

type localApproximationContract struct {
	context string
	n       int
	m       int
	p       int
}

func newLocalApproximationContract(context string, n, m, p int) localApproximationContract {
	return localApproximationContract{context: context, n: n, m: m, p: p}
}

func (c localApproximationContract) validateOperatingPoint(x0, u0 *mat.VecDense) error {
	if x0 == nil || u0 == nil {
		return fmt.Errorf("%s: x0 or u0 is nil: %w", c.context, ErrInvalidArgument)
	}
	if x0.Len() != c.n {
		return fmt.Errorf("%s: x0 length %d != N %d: %w", c.context, x0.Len(), c.n, ErrDimensionMismatch)
	}
	if u0.Len() != c.m {
		return fmt.Errorf("%s: u0 length %d != M %d: %w", c.context, u0.Len(), c.m, ErrDimensionMismatch)
	}
	return nil
}

func (c localApproximationContract) validateStateResult(label string, v *mat.VecDense) error {
	return validateVecResult(c.context+": "+label, v, c.n)
}

func (c localApproximationContract) validateMeasurementResult(label string, v *mat.VecDense) error {
	return validateVecResult(c.context+": "+label, v, c.p)
}

func (c localApproximationContract) validateStateJacobian(label string, m *mat.Dense) error {
	return validateDenseResult(c.context+": "+label, m, c.n, c.n)
}

func (c localApproximationContract) validateMeasurementJacobian(label string, m *mat.Dense) error {
	return validateDenseResult(c.context+": "+label, m, c.p, c.n)
}

func validateVecResult(context string, v *mat.VecDense, want int) error {
	if v == nil {
		return fmt.Errorf("%s returned nil vector: %w", context, ErrDimensionMismatch)
	}
	if v.Len() != want {
		return fmt.Errorf("%s returned length %d, want %d: %w", context, v.Len(), want, ErrDimensionMismatch)
	}
	return nil
}

func finiteDifferenceLocalModel(
	contract localApproximationContract,
	state func(x, u *mat.VecDense) *mat.VecDense,
	measurement func(x, u *mat.VecDense) *mat.VecDense,
	x0, u0 *mat.VecDense,
) (A, B, C, D *mat.Dense, err error) {
	sqrtEps := math.Sqrt(eps())
	n, m, p := contract.n, contract.m, contract.p

	aData := make([]float64, n*n)
	bData := make([]float64, n*m)
	cData := make([]float64, p*n)
	dData := make([]float64, p*m)

	xp := mat.NewVecDense(n, nil)
	xm := mat.NewVecDense(n, nil)
	for j := range n {
		h := sqrtEps * math.Max(math.Abs(x0.AtVec(j)), 1.0)
		inv2h := 1.0 / (2.0 * h)

		xp.CopyVec(x0)
		xm.CopyVec(x0)
		xp.SetVec(j, x0.AtVec(j)+h)
		xm.SetVec(j, x0.AtVec(j)-h)

		fp := state(xp, u0)
		fm := state(xm, u0)
		if err := contract.validateStateResult("F(x+h,u)", fp); err != nil {
			return nil, nil, nil, nil, err
		}
		if err := contract.validateStateResult("F(x-h,u)", fm); err != nil {
			return nil, nil, nil, nil, err
		}
		for i := range n {
			aData[i*n+j] = (fp.AtVec(i) - fm.AtVec(i)) * inv2h
		}

		hp := measurement(xp, u0)
		hm := measurement(xm, u0)
		if err := contract.validateMeasurementResult("H(x+h,u)", hp); err != nil {
			return nil, nil, nil, nil, err
		}
		if err := contract.validateMeasurementResult("H(x-h,u)", hm); err != nil {
			return nil, nil, nil, nil, err
		}
		for i := range p {
			cData[i*n+j] = (hp.AtVec(i) - hm.AtVec(i)) * inv2h
		}
	}

	up := mat.NewVecDense(m, nil)
	um := mat.NewVecDense(m, nil)
	for j := range m {
		h := sqrtEps * math.Max(math.Abs(u0.AtVec(j)), 1.0)
		inv2h := 1.0 / (2.0 * h)

		up.CopyVec(u0)
		um.CopyVec(u0)
		up.SetVec(j, u0.AtVec(j)+h)
		um.SetVec(j, u0.AtVec(j)-h)

		fp := state(x0, up)
		fm := state(x0, um)
		if err := contract.validateStateResult("F(x,u+h)", fp); err != nil {
			return nil, nil, nil, nil, err
		}
		if err := contract.validateStateResult("F(x,u-h)", fm); err != nil {
			return nil, nil, nil, nil, err
		}
		for i := range n {
			bData[i*m+j] = (fp.AtVec(i) - fm.AtVec(i)) * inv2h
		}

		hp := measurement(x0, up)
		hm := measurement(x0, um)
		if err := contract.validateMeasurementResult("H(x,u+h)", hp); err != nil {
			return nil, nil, nil, nil, err
		}
		if err := contract.validateMeasurementResult("H(x,u-h)", hm); err != nil {
			return nil, nil, nil, nil, err
		}
		for i := range p {
			dData[i*m+j] = (hp.AtVec(i) - hm.AtVec(i)) * inv2h
		}
	}

	return mat.NewDense(n, n, aData),
		mat.NewDense(n, m, bData),
		mat.NewDense(p, n, cData),
		mat.NewDense(p, m, dData),
		nil
}

func validateDenseResult(context string, m *mat.Dense, wantR, wantC int) error {
	if m == nil {
		return fmt.Errorf("%s returned nil matrix: %w", context, ErrDimensionMismatch)
	}
	r, c := m.Dims()
	if r != wantR || c != wantC {
		return fmt.Errorf("%s returned %dx%d, want %dx%d: %w", context, r, c, wantR, wantC, ErrDimensionMismatch)
	}
	return nil
}
