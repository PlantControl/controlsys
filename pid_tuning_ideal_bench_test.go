package controlsys

import (
	"context"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func benchmarkIdealHighOrderPlant(b *testing.B, lag float64) *System {
	b.Helper()
	num, den := Poly{0.3, 0.1}, Poly{1, -1.2, 0.5}
	for i := range 16 {
		p := 0.1 + 0.03*float64(i)
		num, den = num.Mul(Poly{1, -(p + 0.005)}), den.Mul(Poly{1, -p})
	}
	res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: 0.1}).StateSpace()
	if err != nil {
		b.Fatal(err)
	}
	if lag > 0 {
		if err := res.Sys.SetInputDelay([]float64{lag}); err != nil {
			b.Fatal(err)
		}
	}
	return res.Sys
}

func BenchmarkPIDStabilityScreenIdealHighOrder(b *testing.B) {
	g, err := New(mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, 0.4}), mat.NewDense(2, 1, []float64{1, 0.5}), mat.NewDense(1, 2, []float64{0.3, 1}), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		b.Fatal(err)
	}
	if err := g.SetInputDelay([]float64{20}); err != nil {
		b.Fatal(err)
	}
	k, err := NewGain(mat.NewDense(1, 1, []float64{0.2}), 0.1)
	if err != nil {
		b.Fatal(err)
	}
	internal, err := Feedback(g, k, -1)
	if err != nil {
		b.Fatal(err)
	}
	c := &PID2{Kp: 0.05, Ki: 0.02, Kd: 0.005, B: 1, C: 1, Dt: 0.1}
	for _, bc := range []struct {
		name  string
		plant *System
	}{
		{"n18", benchmarkIdealHighOrderPlant(b, 0)},
		{"n18lag15", benchmarkIdealHighOrderPlant(b, 15)},
		{"n18lag60", benchmarkIdealHighOrderPlant(b, 60)},
		{"internal22", internal},
	} {
		b.Run(bc.name, func(b *testing.B) {
			screen, err := newPIDStabilityScreen(bc.plant, nil, 1)
			if err != nil {
				b.Fatal(err)
			}
			for b.Loop() {
				if _, err := screen.closedLoopPoles(c); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}

func BenchmarkTunePIDIdealHighOrder(b *testing.B) {
	for _, bc := range []struct {
		name string
		lag  float64
	}{{"n18", 0}, {"n18lag15", 15}} {
		plant := benchmarkIdealHighOrderPlant(b, bc.lag)
		b.Run(bc.name, func(b *testing.B) {
			for b.Loop() {
				_, _ = TunePID(context.Background(), plant, PidtunePID, PIDTuningOptions{})
			}
		})
	}
}
