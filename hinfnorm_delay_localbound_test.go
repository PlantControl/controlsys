package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"
)

// TestHighFrequencyGainLocalBounds checks localBound and quadBound against
// σ_max(G∞) sampled inside cells around random centres, for random
// non-symmetric D blocks with ‖D22‖_F = 0.7 and delay classes with k_i > 1.
func TestHighFrequencyGainLocalBounds(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	r2 := math.Sqrt2
	for _, tc := range []struct {
		p, m int
		tau  []float64
	}{
		{1, 1, []float64{1, 2, r2}},
		{1, 2, []float64{1, 3, r2, 2 * r2}},
		{2, 1, []float64{0.7, r2, math.Sqrt(3)}},
		{2, 2, []float64{1, 2, r2}},
		{3, 2, []float64{1, 5, r2, 3}},
	} {
		nd := len(tc.tau)
		p, m := tc.p+nd, tc.m+nd
		d := make([]float64, p*m)
		for i := range d {
			d[i] = 2*rng.Float64() - 1
		}
		var f float64
		for i := range nd {
			for j := range nd {
				f += d[(tc.p+i)*m+tc.m+j] * d[(tc.p+i)*m+tc.m+j]
			}
		}
		for i := range nd {
			for j := range nd {
				d[(tc.p+i)*m+tc.m+j] *= 0.7 / math.Sqrt(f)
			}
		}
		H, err := NewFromSlices(1, m, p, []float64{-1}, make([]float64, m), make([]float64, p), d, 0)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := SetDelayModel(H, tc.tau)
		if err != nil {
			t.Fatal(err)
		}
		e, err := newDelayLFT(sys)
		if err != nil {
			t.Fatal(err)
		}
		hf := newHighFrequencyGain(e)
		vector := tc.p == 1 || tc.m == 1
		c, y := make([]float64, hf.dim), make([]float64, hf.dim)
		tighter := 0
		for range 300 {
			for i := range c {
				c[i] = 2 * math.Pi * rng.Float64()
			}
			r := []float64{0.5, 0.1, 0.01}[rng.IntN(3)]
			v := hf.at(c)
			hf.local()
			lb, qb := v+hf.localBound(r), hf.quadBound(v, r)
			if math.IsInf(qb, 1) && r < 0.1 {
				t.Fatalf("%v: quadBound(%g) = %g, vector %v", tc.tau, r, qb, vector)
			}
			if qb < v+hf.lip*r {
				tighter++
			}
			for k := range 200 {
				for i := range y {
					u := 2*rng.Float64() - 1
					if k < 1<<hf.dim {
						u = float64(2*(k>>i&1) - 1)
					}
					y[i] = c[i] + u*r/hf.scale[i]
				}
				g := hf.at(y)
				if g > lb*(1+1e-12) || g > qb*(1+1e-12) {
					t.Fatalf("%v p=%d m=%d r=%g: σ(G∞) = %.15g above localBound %.15g or quadBound %.15g",
						tc.tau, tc.p, tc.m, r, g, lb, qb)
				}
			}
		}
		if vector && tighter < 100 {
			t.Errorf("%v: quadBound below v + lip·r in %d of 300 cells", tc.tau, tighter)
		}
	}
}

// TestHinfNormMixedClassNearLimit checks mixed-class delay sets whose finite
// peak lies within 1% (case a) and 0.02% (case b) of sup |G∞|, which the
// Lipschitz branch and bound alone could not separate within its sample
// budget: both are now certified, against a dense ω sweep and a random
// high-frequency sweep of the closed-form gain.
func TestHinfNormMixedClassNearLimit(t *testing.T) {
	for _, tc := range []struct {
		name     string
		tau, c   []float64
		d        []float64
		kr       float64
		maxProbe int
	}{
		{"a", []float64{1.3651970106171443, 2.7303940212342885, 1.9306801277259718, 5.792040383177914},
			[]float64{-0.10491059438649941, -0.6435138647708629, 0.8394097515189132, 0.24102980972400334},
			[]float64{0.19016676315365136, 0.37900150540340577, 0.3448308224580521, -0.2765731844900951},
			0.27347162580459977, 1 << 17},
		{"b", []float64{1.3363534443650815, 4.009060333095245, 2.31463206250999},
			[]float64{-0.6623250852096314, -0.6259897961718044, 0.035388776813513045},
			[]float64{0.3536518283537796, -0.3158520518498577, 0.20814462646252024},
			0.05284290654396216, 1 << 17},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, gain := delayResonanceLoop(t, tc.tau, tc.kr, tc.c, tc.d)
			got, w, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if g := gain(w); math.Abs(g-got) > 1e-9*got {
				t.Errorf("|G(j%g)| = %.12g, HinfNorm %.12g", w, g, got)
			}
			peak, _ := oraclePeak(gain, 100, 2e-3)
			if math.Abs(peak-got) > 1e-6*got {
				t.Errorf("oracle peak %.12g, HinfNorm %.12g", peak, got)
			}
			rng := rand.New(rand.NewPCG(1, 2))
			hfMax := 0.0
			for range 200000 {
				hfMax = max(hfMax, gain(1e4+1e6*rng.Float64()))
			}
			if hfMax > got*(1+1e-6) || hfMax < got*0.98 {
				t.Errorf("high-frequency sweep max %.12g, HinfNorm %.12g", hfMax, got)
			}
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			hf := newHighFrequencyGain(e)
			if _, ok := hf.levelBelow(hf.estimate(), got); !ok || hf.samples > tc.maxProbe {
				t.Errorf("levelBelow ok = %v after %d samples", ok, hf.samples)
			}
		})
	}
}
