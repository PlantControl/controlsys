package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// delayTopologyTol is the relative size, against the largest delay involved,
// below which a delay derived by arithmetic on other delays is roundoff.
// Cancelling sums of delays bounded by M carry rounding and decimal
// representation error of a few eps*M (0.1*3 - 0.1 - 0.2 ~ 5.6e-17); 1e-12
// leaves a margin of >1000x while staying far below any meaningful delay
// ratio. Left unsnapped, such residues become Padé banks with ~tau^-order
// coefficients.
const delayTopologyTol = 1e-12

func delayRoundoffBound(scale float64) float64 {
	return delayTopologyTol * scale
}

func snapDelayRoundoff(delays []float64, scale float64) {
	bound := delayRoundoffBound(scale)
	for i, v := range delays {
		if math.Abs(v) <= bound {
			delays[i] = 0
		}
	}
}

func delaysEqual(a, b, scale float64) bool {
	return math.Abs(a-b) <= delayRoundoffBound(scale)
}

type delayTopology struct {
	sys *System
	p   int
	m   int
}

type delayTopologyDecomposition struct {
	inputDelay  []float64
	outputDelay []float64
	residual    *mat.Dense
}

func newDelayTopology(sys *System) delayTopology {
	_, m, p := sys.Dims()
	return delayTopology{sys: sys, p: p, m: m}
}

func (dt delayTopology) totalExternal(includeDelayMatrix bool) *mat.Dense {
	return effectiveIODelayMatrix(dt.sys, dt.p, dt.m, includeDelayMatrix)
}

func decomposedDelayMatrix(delay *mat.Dense) delayTopologyDecomposition {
	if delay == nil {
		return delayTopologyDecomposition{}
	}
	inputDelay, outputDelay, residual := decomposeIODelay(delay)
	return delayTopologyDecomposition{
		inputDelay:  inputDelay,
		outputDelay: outputDelay,
		residual:    residual,
	}
}

func (d delayTopologyDecomposition) hasResidual() bool {
	return delayMatrixHasNonzero(d.residual)
}

func delayMatrixHasNonzero(m *mat.Dense) bool {
	if m == nil {
		return false
	}
	raw := m.RawMatrix()
	for i := 0; i < raw.Rows; i++ {
		for _, v := range raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols] {
			if v != 0 {
				return true
			}
		}
	}
	return false
}

func delaySliceHasNonzero(s []float64) bool {
	for _, v := range s {
		if v != 0 {
			return true
		}
	}
	return false
}
