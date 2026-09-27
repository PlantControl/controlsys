package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type referenceSource string

const (
	ReferenceMATLAB        referenceSource = "MATLAB"
	ReferencePythonControl referenceSource = "python-control"
)

type referenceTolerance float64

const (
	RefTolExact referenceTolerance = 1e-12
	RefTolTight referenceTolerance = 1e-10
	RefTolLoose referenceTolerance = 1e-6
)

type referenceFixture struct {
	t      *testing.T
	source referenceSource
	caseID string
	tol    float64
}

func matlabReference(t *testing.T, caseID string, tol referenceTolerance) referenceFixture {
	t.Helper()
	return newReferenceFixture(t, ReferenceMATLAB, caseID, tol)
}

func pythonControlReference(t *testing.T, caseID string, tol referenceTolerance) referenceFixture {
	t.Helper()
	return newReferenceFixture(t, ReferencePythonControl, caseID, tol)
}

func newReferenceFixture(t *testing.T, source referenceSource, caseID string, tol referenceTolerance) referenceFixture {
	t.Helper()
	return referenceFixture{
		t:      t,
		source: source,
		caseID: caseID,
		tol:    float64(tol),
	}
}

func (f referenceFixture) Dense(r, c int, data []float64) *mat.Dense {
	f.t.Helper()
	return mat.NewDense(r, c, data)
}

func (f referenceFixture) StateSpace(n, m, p int, a, b, c, d []float64, dt float64) *System {
	f.t.Helper()
	sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
	if err != nil {
		f.t.Fatalf("%s reference %q state-space model: %v", f.source, f.caseID, err)
	}
	return sys
}

func (f referenceFixture) AssertDense(label string, got, want *mat.Dense) {
	f.t.Helper()
	gr, gc := got.Dims()
	wr, wc := want.Dims()
	if gr != wr || gc != wc {
		f.t.Fatalf("%s reference %q %s dims = %dx%d, want %dx%d", f.source, f.caseID, label, gr, gc, wr, wc)
	}
	for i := range gr {
		for j := range gc {
			if diff := math.Abs(got.At(i, j) - want.At(i, j)); diff > f.tol {
				f.t.Fatalf("%s reference %q %s[%d,%d] = %g, want %g (diff %g > tol %g)",
					f.source, f.caseID, label, i, j, got.At(i, j), want.At(i, j), diff, f.tol)
			}
		}
	}
}

func (f referenceFixture) AssertScalar(label string, got, want float64) {
	f.t.Helper()
	if diff := math.Abs(got - want); diff > f.tol {
		f.t.Fatalf("%s reference %q %s = %g, want %g (diff %g > tol %g)",
			f.source, f.caseID, label, got, want, diff, f.tol)
	}
}
