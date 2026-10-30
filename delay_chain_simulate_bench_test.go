package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func benchDelayChainSystem(b *testing.B) *System {
	A := mat.NewDense(3, 3, []float64{-1.2, 0.4, 0.1, -0.3, -0.8, 0.5, 0.2, -0.6, -1.5})
	sys, err := New(A, mat.NewDense(3, 1, []float64{1, 0.3, -0.2}), mat.NewDense(1, 3, []float64{0.7, -0.4, 1}), mat.NewDense(1, 1, []float64{0.1}), 0)
	if err != nil {
		b.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{0.3},
		mat.NewDense(3, 1, []float64{0.4, -0.2, 0.3}), mat.NewDense(1, 3, []float64{0.5, 0.1, -0.3}),
		mat.NewDense(1, 1, []float64{0.2}), mat.NewDense(1, 1, []float64{0.1}), mat.NewDense(1, 1, []float64{0.25})); err != nil {
		b.Fatal(err)
	}
	return sys
}

func BenchmarkStepInternalDelayChain(b *testing.B) {
	sys := benchDelayChainSystem(b)
	for b.Loop() {
		if _, err := Step(sys, 12); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkInitialInternalDelayChain(b *testing.B) {
	sys := benchDelayChainSystem(b)
	x0 := mat.NewVecDense(3, []float64{1, -0.5, 0.25})
	for b.Loop() {
		if _, err := Initial(sys, x0, 12); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLsimInternalDelayChain(b *testing.B) {
	sys := benchDelayChainSystem(b)
	const steps = 20001
	t := make([]float64, steps)
	u := mat.NewDense(steps, 1, nil)
	for k := range steps {
		t[k] = float64(k) * 6e-4
		if k >= steps/3 {
			u.Set(k, 0, 1)
		}
	}
	x0 := mat.NewVecDense(3, []float64{1, -0.5, 0.25})
	for b.Loop() {
		if _, err := Lsim(sys, u, t, x0); err != nil {
			b.Fatal(err)
		}
	}
}
