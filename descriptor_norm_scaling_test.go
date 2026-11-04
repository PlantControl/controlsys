package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestDescriptorNormsLargeEInverseB(t *testing.T) {
	A0 := mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2})
	B0 := mat.NewDense(2, 1, []float64{0, 1})
	C0 := mat.NewDense(1, 2, []float64{1, 1})
	D := mat.NewDense(1, 1, nil)
	base, err := New(A0, B0, C0, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	h2Base, err := H2Norm(base)
	if err != nil {
		t.Fatal(err)
	}
	hinfBase, wBase, err := HinfNorm(base)
	if err != nil {
		t.Fatal(err)
	}

	const eE, eB, eC = -660, 500, -160
	gain := eB + eC - eE
	E := mat.NewDense(2, 2, []float64{math.Ldexp(1, eE), 0, 0, math.Ldexp(1, eE)})
	sys, err := NewDescriptor(scaledDense(A0, eE), scaledDense(B0, eB), scaledDense(C0, eC), D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	h2, err := H2Norm(sys)
	if err != nil {
		t.Fatalf("H2Norm: %v", err)
	}
	if want := math.Ldexp(h2Base, gain); math.Abs(h2-want) > 1e-12*want {
		t.Errorf("H2Norm = %v, want %v", h2, want)
	}
	hinf, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("HinfNorm: %v", err)
	}
	if want := math.Ldexp(hinfBase, gain); math.Abs(hinf-want) > 1e-9*want || math.Abs(w-wBase) > 1e-6*(1+wBase) {
		t.Errorf("HinfNorm = %v at %v, want %v at %v", hinf, w, want, wBase)
	}
}

func TestDescriptorNormsGenuineOverflow(t *testing.T) {
	E := mat.NewDense(2, 2, []float64{math.Ldexp(1, -1000), 0, 0, math.Ldexp(1, -1000)})
	A := scaledDense(mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}), -1000)
	sys, err := NewDescriptor(A, mat.NewDense(2, 1, []float64{0, math.Ldexp(1, 1000)}),
		mat.NewDense(1, 2, []float64{math.Ldexp(1, 1000), 0}), mat.NewDense(1, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if h, err := H2Norm(sys); !errors.Is(err, ErrOverflow) {
		t.Errorf("H2Norm = %v, %v; want ErrOverflow", h, err)
	}
	if h, _, err := HinfNorm(sys); !errors.Is(err, ErrOverflow) {
		t.Errorf("HinfNorm = %v, %v; want ErrOverflow", h, err)
	}
}
