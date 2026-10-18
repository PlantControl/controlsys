package controlsys

import (
	"math"
	"testing"
)

// FPMBHI: the integrator cluster radius 100·ε^(1/m)·scale reaches scale at
// m = 8, so with eight or more roots every root counted as an integrator
// and the pole fallback found no slowest pole: G = k·∏(s+0.5+0.1i)·(s+3) /
// (s·∏(s+1+0.2i)), i < n, has no gain crossover for k = 500 and returned
// "no nonintegrating pole or zero" for n ≥ 7. Oracle: the slowest
// nonintegrating pole is 1 by construction, so the crossover is 0.5; the
// PI design is checked by closed-loop roots and the signed margin at every
// crossover of the directly evaluated loop.
func TestPidtuneIntegratorClusterManyRoots(t *testing.T) {
	for n := 3; n <= 10; n++ {
		num, den := Poly{500}, Poly{1, 0}
		for i := range n {
			num = num.Mul(Poly{1, 0.5 + 0.1*float64(i)})
			den = den.Mul(Poly{1, 1 + 0.2*float64(i)})
		}
		num = num.Mul(Poly{1, 3})
		p := pidtuneBackoffPlant(t, num, den, 0, 0)
		wc, err := plantCrossoverFreq(p)
		if err != nil || math.Abs(wc-0.5) > 1e-6 {
			t.Errorf("n=%d: crossover %g, %v; want 0.5", n, wc, err)
			continue
		}
		pid, err := Pidtune(p, PidtunePI, 0, nil)
		if err != nil {
			t.Errorf("n=%d: %v", n, err)
			continue
		}
		if stable, ok := pidtuneOracleStable(t, pid, num, den, 0, 0); !ok || !stable {
			t.Errorf("n=%d: %+v unstable", n, pid)
		}
		pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, den, 0, 0), 1e-6, 1e4)
		if len(pms) == 0 || ms > pidTuningMaxSensitivity {
			t.Errorf("n=%d: %+v crossovers %v, |S| peak %.3g", n, pid, pms, ms)
		}
		for _, pm := range pms {
			if pm <= 0 {
				t.Errorf("n=%d: %+v phase margin %.4g° at a crossover", n, pid, pm)
			}
		}
	}
}

// FPMBHI, pole fallback without an integrator: G = 0.5·s·∏(s+0.5+0.1i) /
// ∏(s+1+0.2i), nine poles, stays below 0 dB with G(0) = 0, so the crossover
// is half the slowest pole, 0.5; all nine poles fell in the cluster.
func TestPidtunePoleFallbackClusterManyRoots(t *testing.T) {
	num, den := Poly{0.5, 0}, Poly{1}
	for i := range 9 {
		if i < 8 {
			num = num.Mul(Poly{1, 0.5 + 0.1*float64(i)})
		}
		den = den.Mul(Poly{1, 1 + 0.2*float64(i)})
	}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	wc, err := plantCrossoverFreq(p)
	if err != nil || math.Abs(wc-0.5) > 1e-6 {
		t.Fatalf("crossover %g, %v; want 0.5", wc, err)
	}
}

// A rounded integrator ring still counts as one cluster: the roots of s³
// from a balanced realization split by about ε^(1/3), their mean does not.
func TestPidtuneClusterSizeRing(t *testing.T) {
	d := 1e-5
	ring := []complex128{complex(d, 0), complex(-d/2, d*math.Sqrt(3)/2), complex(-d/2, -d*math.Sqrt(3)/2), -2}
	if m := pidtuneClusterSize(ring, 0, 2); m != 3 {
		t.Errorf("ring cluster size %d, want 3", m)
	}
	spread := []complex128{0, -1, -1.2, -1.4, -1.6, -1.8, -2, -2.2}
	if m := pidtuneClusterSize(spread, 0, 2.2); m != 1 {
		t.Errorf("spread cluster size %d, want 1", m)
	}
}
