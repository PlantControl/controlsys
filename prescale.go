package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// PrescaleResult holds the output of [Prescale].
//
// StateScale is the diagonal of MATLAB's info.SR (TR = diag(StateScale),
// TL = TR⁻¹ for explicit models). InputScale and OutputScale are suggested
// normalizations (reciprocal of the largest magnitude in each column of
// [Bs; D] and each row of [Cs D]); they are not applied to Sys.
type PrescaleResult struct {
	Sys  *System
	Info struct {
		StateScale  []float64
		InputScale  []float64
		OutputScale []float64
	}
}

// Prescale scales the state vector of sys to improve the accuracy of
// frequency-domain computations, matching MATLAB prescale for explicit
// state-space models: with T = diag(Info.StateScale),
//
//	As = T⁻¹AT, Bs = T⁻¹B, Cs = CT, Ds = D.
//
// The state order is preserved, so Sys has the same response, state names,
// and metadata as sys. Descriptor and delayed models are rejected.
func Prescale(sys *System) (*PrescaleResult, error) {
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Prescale"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Prescale"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p

	if n == 0 {
		result := &PrescaleResult{Sys: policy.zeroOrderCopy()}
		result.Info.InputScale = ones(m)
		result.Info.OutputScale = ones(p)
		return result, nil
	}

	aRaw := sys.A.RawMatrix()
	aData := make([]float64, n*n)
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)

	scale := make([]float64, n)
	impl.Dgebal(lapack.Scale, n, aData, n, scale)

	stateScale := make([]float64, n)
	copy(stateScale, scale)

	dsData := make([]float64, n)
	dsInvData := make([]float64, n)
	for i := range n {
		dsData[i] = scale[i]
		dsInvData[i] = 1.0 / scale[i]
	}
	Ds := mat.NewDiagDense(n, dsData)
	DsInv := mat.NewDiagDense(n, dsInvData)

	Ab := mat.NewDense(n, n, aData)

	Bb := mat.NewDense(n, m, nil)
	Bb.Mul(DsInv, sys.B)

	Cb := mat.NewDense(p, n, nil)
	Cb.Mul(sys.C, Ds)

	Db := denseCopy(sys.D)

	inputScale := make([]float64, m)
	for j := range m {
		maxVal := 0.0
		for i := range n {
			if v := math.Abs(Bb.At(i, j)); v > maxVal {
				maxVal = v
			}
		}
		for i := range p {
			if v := math.Abs(Db.At(i, j)); v > maxVal {
				maxVal = v
			}
		}
		if maxVal > 0 {
			inputScale[j] = 1.0 / maxVal
		} else {
			inputScale[j] = 1.0
		}
	}

	outputScale := make([]float64, p)
	for i := range p {
		maxVal := 0.0
		for j := range n {
			if v := math.Abs(Cb.At(i, j)); v > maxVal {
				maxVal = v
			}
		}
		for j := range m {
			if v := math.Abs(Db.At(i, j)); v > maxVal {
				maxVal = v
			}
		}
		if maxVal > 0 {
			outputScale[i] = 1.0 / maxVal
		} else {
			outputScale[i] = 1.0
		}
	}

	scaled, err := policy.result(Ab, Bb, Cb, Db)
	if err != nil {
		return nil, err
	}

	propagateNames(scaled, sys)

	result := &PrescaleResult{Sys: scaled}
	result.Info.StateScale = stateScale
	result.Info.InputScale = inputScale
	result.Info.OutputScale = outputScale
	return result, nil
}

func ones(n int) []float64 {
	s := make([]float64, n)
	for i := range s {
		s[i] = 1.0
	}
	return s
}
