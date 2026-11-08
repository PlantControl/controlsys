package controlsys

import (
	"context"
	"math"
	"testing"
)

// DXONUM: tuning is scale invariant (MATLAB pidtune), so k·G for tiny k must
// design C/k rather than fail the absolute |G(wc)| < 1e-15 zero-gain test.
func TestTunePID_LowGainPlantScalesDesign(t *testing.T) {
	ref, err := TunePID(context.Background(), makePlant(t, Poly{1}, Poly{1, 1}), PidtunePI, PIDTuningOptions{CrossoverFrequency: 1})
	if err != nil {
		t.Fatal(err)
	}
	for _, k := range []float64{1e-14, 1e-16, 1e-18, 1e-30} {
		for _, wc := range []float64{1, 0} {
			r, err := TunePID(context.Background(), makePlant(t, Poly{k}, Poly{1, 1}), PidtunePI, PIDTuningOptions{CrossoverFrequency: wc})
			if err != nil {
				t.Fatalf("k=%g wc=%g: %v", k, wc, err)
			}
			if wc != 1 {
				continue
			}
			for _, g := range [][2]float64{{r.Controller.Kp * k, ref.Controller.Kp}, {r.Controller.Ki * k, ref.Controller.Ki}} {
				if math.Abs(g[0]-g[1]) > 1e-9*math.Abs(g[1]) {
					t.Errorf("k=%g: k·gain %g, want %g", k, g[0], g[1])
				}
			}
		}
	}
}

func TestTunePIDFRD_LowGainPlantDesigns(t *testing.T) {
	const n, k = 200, 1e-16
	omega := make([]float64, n)
	resp := make([][][]complex128, n)
	for i := range n {
		omega[i] = math.Pow(10, -2+4*float64(i)/(n-1))
		resp[i] = [][]complex128{{complex(k, 0) / complex(1, omega[i])}}
	}
	data, err := NewFRD(resp, omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := pidTuneFRDKept(t, context.Background(), data, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1}); err != nil {
		t.Fatal(err)
	}
}
