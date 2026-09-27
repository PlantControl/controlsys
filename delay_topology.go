package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

const delayTopologyTol = 1e-12

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

func (dt delayTopology) decomposedExternal() delayTopologyDecomposition {
	total := dt.totalExternal(true)
	return decomposedDelayMatrix(total)
}

func decomposedDelayMatrix(delay *mat.Dense) delayTopologyDecomposition {
	if delay == nil {
		return delayTopologyDecomposition{}
	}
	inputDelay, outputDelay, residual := DecomposeIODelay(delay)
	return delayTopologyDecomposition{
		inputDelay:  inputDelay,
		outputDelay: outputDelay,
		residual:    residual,
	}
}

func (dt delayTopology) decomposableExternal(context string) (inputDelay, outputDelay []float64, err error) {
	decomp := dt.decomposedExternal()
	if decomp.hasResidual() {
		return nil, nil, &delayTopologyResidualError{context: context}
	}
	return decomp.inputDelay, decomp.outputDelay, nil
}

func (d delayTopologyDecomposition) hasDelay() bool {
	return delaySliceHasNonzero(d.inputDelay) || delaySliceHasNonzero(d.outputDelay) || d.hasResidual()
}

func (d delayTopologyDecomposition) hasResidual() bool {
	return delayMatrixHasNonzeroTol(d.residual, delayTopologyTol)
}

type delayTopologyResidualError struct {
	context string
}

func (e *delayTopologyResidualError) Error() string {
	return e.context + ": non-decomposable IODelay residual: " + ErrFeedbackDelay.Error()
}

func (e *delayTopologyResidualError) Unwrap() error {
	return ErrFeedbackDelay
}

func delayMatrixHasNonzero(m *mat.Dense) bool {
	return delayMatrixHasNonzeroTol(m, 0)
}

func delayMatrixHasNonzeroTol(m *mat.Dense, tol float64) bool {
	if m == nil {
		return false
	}
	raw := m.RawMatrix()
	for i := 0; i < raw.Rows; i++ {
		for j := 0; j < raw.Cols; j++ {
			if math.Abs(raw.Data[i*raw.Stride+j]) > tol {
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
