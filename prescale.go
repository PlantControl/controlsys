package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

type PrescaleResult struct {
	Sys  *System
	Info struct {
		StateScale  []float64
		InputScale  []float64
		OutputScale []float64
	}
}

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
	impl.Dgebal(lapack.PermuteScale, n, aData, n, scale)

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
