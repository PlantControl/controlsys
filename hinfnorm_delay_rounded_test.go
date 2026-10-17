package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestHinfNormRoundedMultipleSweep sweeps the OUWA5P seeds τ = (k0·h, k1·h)
// with fl(k·h) ≠ k·h for some delay: each stable model has the finite norm
// of the dense ω oracle and the one-phase circle oracle, or a finite peak
// past the oracle range that G attains at the reported ω; IsStable agrees
// with the dyadic twin, exactly commensurate.
func TestHinfNormRoundedMultipleSweep(t *testing.T) {
	seeds := 200
	if testing.Short() {
		seeds = 40
	}
	rounded := 0
	for _, k := range [][2]float64{{1, 3}, {3, 1}, {2, 3}} {
		for i := range seeds {
			rng := rand.New(rand.NewPCG(uint64(i+1000), 7))
			u := func(a float64) float64 { return a * (2*rng.Float64() - 1) }
			d22 := []float64{u(.8), u(.8), u(.8), u(.8)}
			d12 := []float64{u(1), u(1)}
			d21 := []float64{u(1), u(1)}
			h := 0.02 + 0.3*rng.Float64()
			tau := []float64{k[0] * h, k[1] * h}
			if math.FMA(k[0], h, -tau[0]) == 0 && math.FMA(k[1], h, -tau[1]) == 0 {
				continue
			}
			rounded++
			sys, gain, gInf := feedthroughDifferenceLoop(t, d22, d12, d21, tau, k)
			hd := math.Ldexp(math.Round(math.Ldexp(h, 30)), -30)
			twin, _, _ := feedthroughDifferenceLoop(t, d22, d12, d21, []float64{k[0] * hd, k[1] * hd}, k)
			st, err := sys.IsStable()
			stD, errD := twin.IsStable()
			if err != nil || errD != nil || st != stD {
				t.Fatalf("k=%v seed %d: IsStable = %v, %v; dyadic twin %v, %v", k, i, st, err, stD, errD)
			}
			if !st {
				continue
			}
			got, w, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("k=%v seed %d: HinfNorm: %v", k, i, err)
			}
			want, _ := oraclePeak(gain, 80, 2e-3)
			circle, _ := oraclePeak(gInf, 2*math.Pi, 2*math.Pi/(1<<14))
			want = max(want, circle)
			if math.Abs(got-want) <= 1e-8*want {
				continue
			}
			if !(got > want && w > 80 && !math.IsInf(w, 1) && math.Abs(gain(w)-got) <= 1e-8*got) {
				t.Fatalf("k=%v seed %d: HinfNorm = %.12g @%g, want %.12g", k, i, got, w, want)
			}
		}
	}
	if rounded < seeds {
		t.Fatalf("only %d models with fl(k·h) ≠ k·h", rounded)
	}
}
