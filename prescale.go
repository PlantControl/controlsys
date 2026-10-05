package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// PrescaleResult holds the output of [Prescale].
//
// LeftScale and StateScale are the diagonals of MATLAB's info.SL and info.SR:
// TL = diag(LeftScale), TR = diag(StateScale), and TL = TR⁻¹ for explicit
// models. InputScale and OutputScale are suggested normalizations
// (reciprocal of the largest magnitude in each column of [Bs; D] and each row
// of [Cs D]); they are not applied to Sys.
type PrescaleResult struct {
	Sys  *System
	Info PrescaleInfo
}

// PrescaleInfo is the scaling information of Prescale, as MATLAB prescale's
// info output; see PrescaleResult.
type PrescaleInfo struct {
	LeftScale   []float64
	StateScale  []float64
	InputScale  []float64
	OutputScale []float64
}

// Prescale scales the state vector of sys to improve the accuracy of
// frequency-domain computations, as MATLAB prescale: with
// TL = diag(Info.LeftScale) and TR = diag(Info.StateScale),
//
//	As = TL·A·TR, Bs = TL·B, Cs = C·TR, Ds = D, Es = TL·E·TR.
//
// Explicit models use TL = TR⁻¹ from LAPACK balancing of A (Dgebal);
// descriptor models use the left and right scalings of the pencil (A, E)
// from Dggbal. Unlike MATLAB, the scaling is not tuned to a frequency band.
// The state order is preserved, so Sys has the same response, state names,
// and metadata as sys. I/O delays carry over and internal-delay channels
// scale like B and C (B2s = TL·B2, C2s = C2·TR).
//
// A nil model or NaN/Inf entries return ErrInvalidArgument.
func Prescale(sys *System) (*PrescaleResult, error) {
	if err := requireFiniteSystem("Prescale", sys); err != nil {
		return nil, err
	}
	policy := newRealizationTransformPolicy(sys)
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

	left := make([]float64, n)
	right := make([]float64, n)
	var Es *mat.Dense
	if sys.IsDescriptor() {
		eRaw := sys.E.RawMatrix()
		eData := make([]float64, n*n)
		copyStrided(eData, n, eRaw.Data, eRaw.Stride, n, n)
		impl.Dggbal(lapack.Scale, n, aData, n, eData, n, left, right, make([]float64, 6*n))
		Es = mat.NewDense(n, n, eData)
	} else {
		impl.Dgebal(lapack.Scale, n, aData, n, right)
		for i, v := range right {
			left[i] = 1 / v
		}
		if sys.E != nil {
			Es = mat.DenseCopyOf(sys.E)
		}
	}
	TL := mat.NewDiagDense(n, left)
	TR := mat.NewDiagDense(n, right)

	Ab := mat.NewDense(n, n, aData)

	Bb := newDense(n, m)
	if m > 0 {
		Bb.Mul(TL, sys.B)
	}

	Cb := newDense(p, n)
	if p > 0 {
		Cb.Mul(sys.C, TR)
	}

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

	var scaled *System
	var err error
	if nd := sys.internalDelayCount(); nd > 0 {
		B2 := mat.NewDense(n, nd, nil)
		B2.Mul(TL, sys.LFT.B2)
		C2 := mat.NewDense(nd, n, nil)
		C2.Mul(sys.LFT.C2, TR)
		scaled, err = policy.resultWithInternalDelay(Ab, Bb, Cb, Db, B2, C2)
	} else {
		scaled, err = policy.result(Ab, Bb, Cb, Db)
	}
	if err != nil {
		return nil, err
	}
	scaled.E = Es

	propagateNames(scaled, sys)

	result := &PrescaleResult{Sys: scaled}
	result.Info.LeftScale = left
	result.Info.StateScale = right
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
