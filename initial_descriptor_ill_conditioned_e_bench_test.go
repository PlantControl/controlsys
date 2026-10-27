package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func illConditionedEBenchModel(b *testing.B) *System {
	c := stepFoldLossCases[2]
	sys, err := NewDescriptor(mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, c.m, c.b), mat.NewDense(c.p, c.n, c.c),
		mat.NewDense(c.p, c.m, c.d), mat.NewDense(c.n, c.n, c.e), 0)
	if err != nil {
		b.Fatal(err)
	}
	return sys
}

func BenchmarkInitialIllConditionedE(b *testing.B) {
	sys := illConditionedEBenchModel(b)
	x0 := mat.NewVecDense(4, []float64{1, 1, 1, 1})
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Initial(sys, x0, 5); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLsimInitialStateIllConditionedE(b *testing.B) {
	sys := illConditionedEBenchModel(b)
	x0 := mat.NewVecDense(4, []float64{1, 1, 1, 1})
	t := makeTimeVector(501, 0.01)
	u := mat.NewDense(len(t), 2, nil)
	for k := range t {
		u.Set(k, 0, 1)
	}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Lsim(sys, u, t, x0); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkC2DInputDelayIllConditionedE(b *testing.B) {
	sys := illConditionedEBenchModel(b)
	sys.InputDelay = []float64{0.2, 0}
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.C2D(0.1, C2DOptions{}); err != nil {
			b.Fatal(err)
		}
	}
}
