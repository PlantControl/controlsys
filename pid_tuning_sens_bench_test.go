package controlsys

import (
	"context"
	"testing"
)

func BenchmarkTunePIDSensitivityScan(b *testing.B) {
	for _, c := range []struct {
		name     string
		num, den []float64
		dt       float64
		lag      int
		family   PidtuneType
		o        PIDTuningOptions
	}{
		{"SlowPolePID", []float64{1}, []float64{1, 1.01, .01}, 0, 0, PidtunePID, PIDTuningOptions{}},
		{"NMPPD", []float64{0.25735368672434666, -0.09959483477301487, 0.0011398317458989433}, []float64{1, 96.41485396485868, 3538.780004777385, 53732.01273661257, 13120.99411576446}, 0, 0, PidtunePD, PIDTuningOptions{PhaseMargin: 45}},
		{"DelayedDiscretePI", []float64{0.1}, []float64{1, -0.9}, 0.1, 3, PidtunePI, PIDTuningOptions{}},
		{"DelayedDiscreteP", []float64{0.019742253670921476}, []float64{1, -0.14539455905586565}, 0.10463990694240013, 2, PidtuneP, PIDTuningOptions{}},
		{"DiscretePID", []float64{0.01, 0.009}, []float64{1, -1.8, 0.81}, 0.01, 0, PidtunePID, PIDTuningOptions{}},
	} {
		b.Run(c.name, func(b *testing.B) {
			res, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: c.dt}).StateSpace()
			if err != nil {
				b.Fatal(err)
			}
			if c.lag > 0 {
				if err := res.Sys.SetInputDelay([]float64{float64(c.lag)}); err != nil {
					b.Fatal(err)
				}
			}
			b.ReportAllocs()
			for b.Loop() {
				_, _ = TunePID(context.Background(), res.Sys, c.family, c.o)
			}
		})
	}
}
