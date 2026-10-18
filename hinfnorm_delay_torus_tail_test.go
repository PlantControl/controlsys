package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"
)

// torusRidgeSeed is the 2×2 lftCase of seed i with τ = (h, √2·h) whose D12
// and D22 decouple the phase of the second delay from G∞, so σ_max(G∞)
// peaks on a ridge of the torus. Seeds 115 and 153 have a finite peak just
// above the ridge, past the grid end, where tailGain closes the tail only
// near ω ≈ 10⁶.
func torusRidgeSeed(i int) lftCase {
	rng := rand.New(rand.NewPCG(uint64(i+5000), 11))
	u := func(a float64) float64 { return a * (2*rng.Float64() - 1) }
	v := func(n int, a float64) []float64 {
		s := make([]float64, n)
		for k := range s {
			s[k] = u(a)
		}
		return s
	}
	a := v(4, 1)
	a[0] -= 2
	a[3] -= 2
	h := 0.05 + 0.4*rng.Float64()
	c := lftCase{n: 2, a: a, b1: v(4, 1), b2: v(4, 1), c1: v(4, 1), c2: v(4, 0.3), d11: v(4, 1), d12: v(4, 2), d21: v(4, 2), d22: v(4, 0.6)}
	c.h, c.k = h, math.Sqrt2
	c.d12 = []float64{c.d12[0], 0, c.d12[2], 0}
	c.d22 = []float64{c.d22[0], 0, c.d22[2], c.d22[3]}
	return c
}

// ridgeSup is sup σ_max(G∞) of a torusRidgeSeed by a dense scan of the first
// phase refined by golden section, once G∞ is checked independent of the
// second.
func ridgeSup(t *testing.T, c lftCase) float64 {
	t.Helper()
	g := func(p float64) float64 { return lftGInf2(c, p, 0) }
	const n = 1 << 14
	step := 2 * math.Pi / n
	best := 0.0
	for i := range n {
		p := float64(i) * step
		v := g(p)
		for _, q := range []float64{1, 2.5, 4} {
			if w := lftGInf2(c, p, q); math.Abs(w-v) > 1e-12*v {
				t.Fatalf("G∞(%g, %g) = %g, G∞(%g, 0) = %g: not a ridge", p, q, w, p, v)
			}
		}
		if v >= g(p-step) && v >= g(p+step) {
			_, v = goldenMax(p-step, p+step, p, v, g)
		}
		best = max(best, v)
	}
	return best
}

// lftSigmaPhases is σ_max(G) of c at s = jw with the delays at phases p, q
// in place of w·τ: lftCase.sigma with h = p/w and k = q/p.
func lftSigmaPhases(c lftCase, w, p, q float64) float64 {
	c.h, c.k = p/w, q/p
	return c.sigma(w)
}

// TestTorusAtFrequency checks the torus gain of atFrequency against the
// closed-form response with free phases.
func TestTorusAtFrequency(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 9))
	cases := append([]lftCase{torusRidgeSeed(115), torusRidgeSeed(153), torusRidgeCase}, torusPeakCases...)
	for _, c := range cases {
		e, err := newDelayLFT(c.system(t))
		if err != nil {
			t.Fatal(err)
		}
		hf := newHighFrequencyGain(e)
		if hf.dim != 2 {
			t.Fatalf("dim = %d, want 2", hf.dim)
		}
		for range 20 {
			w := e.normA * (1.1 + 50*rng.Float64())
			f := hf.atFrequency(w)
			if f == nil {
				t.Fatalf("atFrequency(%g) = nil", w)
			}
			theta := []float64{0.1 + 6*rng.Float64(), 0.1 + 6*rng.Float64()}
			got, want := f.at(theta), lftSigmaPhases(c, w, theta[0], theta[1])
			if math.Abs(got-want) > 1e-10*want {
				t.Fatalf("%s: w=%g θ=%v: %.15g, want %.15g", c.name, w, theta, got, want)
			}
		}
	}
}

// TestTorusTailBelow checks torusTailBelow against a dense scan of the
// closed-form response past R: a certified level is never below a sample,
// and the level just under the largest sample is refused.
func TestTorusTailBelow(t *testing.T) {
	for _, i := range []int{115, 153} {
		c := torusRidgeSeed(i)
		e, err := newDelayLFT(c.system(t))
		if err != nil {
			t.Fatal(err)
		}
		hf := newHighFrequencyGain(e)
		lim := ridgeSup(t, c) * (1 + 1e-7)
		if !hf.below(lim, 1<<21) {
			t.Fatalf("seed %d: below(%g) failed", i, lim)
		}
		for _, R := range []float64{500, 2000, 8000} {
			top := 0.0
			for w := R; w < 4*R; w += math.Pi / (32 * c.k * c.h) {
				top = max(top, c.sigma(w))
			}
			if ok, _ := e.torusTailBelow(hf, lim, top*(1-1e-9), R, 1<<18); ok {
				t.Errorf("seed %d R=%g: certified %g below sample %g", i, R, top*(1-1e-9), top)
			}
			certified := false
			for _, d := range []float64{1e-7, 1e-6, 1e-5, 1e-4, 1e-3} {
				level := lim * (1 + d)
				if ok, _ := e.torusTailBelow(hf, lim, level, R, 1<<18); ok {
					certified = true
					if top > level {
						t.Errorf("seed %d R=%g: certified %g below sample %g", i, R, level, top)
					}
				}
			}
			if !certified {
				t.Errorf("seed %d R=%g: no level up to lim·(1+1e-3) certified", i, R)
			}
		}
	}
}

// TestHinfNormTorusRidgeTail pins the finite peak of the ridge seeds, just
// above the ridge and past the grid end, against a dense scan of the
// closed-form response refined by golden section, within a bounded number
// of samples of G: the tail used to be closed by tailGain only near
// ω ≈ 10⁶, and the certificate over 1.5M grid points gave up.
func TestHinfNormTorusRidgeTail(t *testing.T) {
	for _, i := range []int{115, 153} {
		c := torusRidgeSeed(i)
		e, err := newDelayLFT(c.system(t))
		if err != nil {
			t.Fatal(err)
		}
		l, res, err := e.stability()
		if err != nil || !res.stable {
			t.Fatalf("seed %d: stability = %v, %v", i, res.stable, err)
		}
		got, w, err := e.peakOrZero(l, res)
		if err != nil {
			t.Fatalf("seed %d: %v", i, err)
		}
		want, wWant := oraclePeak(c.sigma, 1500, 1e-2)
		if lim := ridgeSup(t, c); !(want > lim*(1+1e-7)) || !(wWant > 100) {
			t.Fatalf("seed %d: oracle peak %g at %g not past the grid above the torus sup %g", i, want, wWant, lim)
		}
		if math.Abs(got-want) > 1e-8*want || math.Abs(w-wWant) > 1e-3*wWant {
			t.Errorf("seed %d: peak %.12g at %g, oracle %.12g at %g", i, got, w, want, wWant)
		}
		const maxCalls = 1 << 17
		if e.sigmaCalls > maxCalls {
			t.Errorf("seed %d: %d samples of G, want <= %d", i, e.sigmaCalls, maxCalls)
		}
	}
}
