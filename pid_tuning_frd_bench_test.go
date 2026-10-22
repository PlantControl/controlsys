package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// benchTunePIDFRDData samples num/den, continuous or at dt, at n log
// points on [lo, hi].
func benchTunePIDFRDData(b *testing.B, num, den []float64, dt, lo, hi float64, n int) *FRD {
	b.Helper()
	omega := make([]float64, n)
	values := make([][][]complex128, n)
	for k := range omega {
		omega[k] = math.Exp(math.Log(lo) + float64(k)*math.Log(hi/lo)/float64(n-1))
		x := complex(0, omega[k])
		if dt > 0 {
			x = cmplx.Exp(complex(0, omega[k]*dt))
		}
		values[k] = [][]complex128{{pidPolyAt(num, x) / pidPolyAt(den, x)}}
	}
	frd, err := NewFRD(values, omega, dt)
	if err != nil {
		b.Fatal(err)
	}
	return frd
}

func BenchmarkTunePIDFRD(b *testing.B) {
	for _, c := range []struct {
		name       string
		num, den   []float64
		dt, lo, hi float64
		n          int
		family     PidtuneType
		o          PIDTuningOptions
	}{
		{"DCPI", []float64{1}, []float64{1, 3, 3, 1}, 0, 1e-3, 1e2, 200, PidtunePI, PIDTuningOptions{}},
		{"DCPIDF", []float64{0.5, 1}, []float64{1, 1.2, 2.5, 1}, 0, 1e-3, 1e3, 300, PidtunePIDF, PIDTuningOptions{PhaseMargin: 45}},
		{"DiscretePID", []float64{3.220831505570417}, []float64{1, 0.2825729648756968}, 0.1, 0.2449532382066765, 24.49532382066765, 117, PidtunePID, PIDTuningOptions{}},
		{"NMPPDF", []float64{-0.5218348847732306, 0.01176025018024663}, []float64{1, 0.03931173190809955, 0.0050731315734235775}, 0, 1e-3, 1e2, 100, PidtunePDF, PIDTuningOptions{PhaseMargin: 30}},
		{"LowRotatedPI", []float64{-1.6397335053538995, 0.022678154196576948}, []float64{1, 0.05699814057448621, 0.0010336161545953615, 5.660971928156559e-06}, 0, 1e-3, 1e3, 100, PidtunePI, PIDTuningOptions{PhaseMargin: 45}},
	} {
		frd := benchTunePIDFRDData(b, c.num, c.den, c.dt, c.lo, c.hi, c.n)
		b.Run(c.name, func(b *testing.B) {
			for b.Loop() {
				_, _ = TunePIDFRD(context.Background(), frd, c.family, c.o)
			}
		})
	}
}
