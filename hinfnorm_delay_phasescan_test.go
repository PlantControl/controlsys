package controlsys

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"
)

// TestDelayLoopGridMatchesSortedUnion checks the merged grid against the
// sorted, deduplicated union of its log, linear and scale points.
func TestDelayLoopGridMatchesSortedUnion(t *testing.T) {
	for _, tc := range []struct {
		tau, R, w1 float64
		scales     []float64
	}{
		{0, 50, 50, []float64{1, 3}},
		{0.67, 2e3, 2e3, []float64{1.88, 0.2, 40}},
		{2.5, 1e4, 300, []float64{0.01, 1e3, 7e3}},
		{0.1, 1e5, 1e5, nil},
	} {
		l := &delayLoop{tau: tc.tau, scales: tc.scales}
		got := l.grid(tc.R, tc.w1)
		lo := 1.0
		for _, s := range tc.scales {
			lo = min(lo, s)
		}
		lo *= 1e-3
		nlog := int(64*math.Log10(tc.R/lo)) + 2
		var want []float64
		for i := range nlog {
			want = append(want, lo*math.Pow(tc.R/lo, float64(i)/float64(nlog-1)))
		}
		if tc.tau > 0 {
			step := math.Pi / (8 * tc.tau)
			for w, prev := step, 0.0; w < tc.R && prev < tc.w1; prev, w = w, w+step {
				want = append(want, w)
			}
		}
		for _, s := range tc.scales {
			for _, d := range []float64{-0.3, -0.1, -1e-2, -1e-3, 0, 1e-3, 1e-2, 0.1, 0.3} {
				if w := s * (1 + d); w > 0 && w < tc.R {
					want = append(want, w)
				}
			}
		}
		want = append(want, tc.R)
		slices.Sort(want)
		want = slices.Compact(want)
		if !slices.Equal(got, want) {
			t.Fatalf("tau=%g R=%g: grid of %d points differs from sorted union of %d", tc.tau, tc.R, len(got), len(want))
		}
	}
}

// TestSampleAtNearest checks sampleAt against the circle sample of least
// angular distance, for φ of either sign and far beyond 2π.
func TestSampleAtNearest(t *testing.T) {
	c := limitAtInfCases[0]
	sys, _, _ := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, c.k * c.h}, [2]float64{1, c.k})
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
	x.sampleCircle()
	k := len(x.byS)
	step := 2 * math.Pi / float64(k)
	rng := rand.New(rand.NewPCG(1, 2))
	for range 2000 {
		phi := (rng.Float64() - 0.5) * math.Pow(10, 8*rng.Float64())
		got := x.sampleAt(phi)
		d := func(i int) float64 { return math.Abs(math.Remainder(phi-float64(i)*step, 2*math.Pi)) }
		best := 0
		for i := range k {
			if d(i) < d(best) {
				best = i
			}
		}
		if got != best && d(got)-d(best) > 1e-6*step {
			t.Fatalf("φ=%g: sampleAt %d at distance %g, nearest %d at %g", phi, got, d(got), best, d(best))
		}
	}
}
