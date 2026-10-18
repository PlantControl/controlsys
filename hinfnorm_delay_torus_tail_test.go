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

// ridgeArgSup is the phase θ0* of the first delay at which a torusRidgeSeed
// attains its ridge sup, by the scan of ridgeSup.
func ridgeArgSup(c lftCase) float64 {
	g := func(p float64) float64 { return lftGInf2(c, p, 0) }
	const n = 1 << 12
	step := 2 * math.Pi / n
	best, arg := 0.0, 0.0
	for i := range n {
		p := float64(i) * step
		v := g(p)
		if v >= g(p-step) && v >= g(p+step) {
			p, v = goldenMax(p-step, p+step, p, v, g)
		}
		if v > best {
			best, arg = v, p
		}
	}
	return arg
}

// ridgeCrossingOracle is the finite peak of a torusRidgeSeed up to wMax: a
// dense scan to ω = 1500 and, past ω = 1000, a golden-section search about
// every crossing ω·h = ±θ0* (mod 2π) of the ridge, the only place where G,
// within O(1/ω) of G∞, can come near its sup once ω is large.
func ridgeCrossingOracle(c lftCase, wMax float64) (float64, float64) {
	th := ridgeArgSup(c)
	best, wb := oraclePeak(c.sigma, 1500, 1e-2)
	for _, t0 := range []float64{th, -th} {
		for n := math.Ceil((1000*c.h - t0) / (2 * math.Pi)); ; n++ {
			wc := (t0 + 2*math.Pi*n) / c.h
			if wc > wMax {
				break
			}
			d := 0.02 / c.h
			if w, v := goldenMax(wc-d, wc+d, wc, c.sigma(wc), c.sigma); v > best {
				best, wb = v, w
			}
		}
	}
	return best, wb
}

// TestFlatClasses checks that the second class of a torusRidgeSeed, which
// D12 and D22 decouple, is flat and that below certifies the ridge sup
// (dense oracle) to 1e-9 on the torus of the first, which the 2-D cells
// could not within 1<<20 samples, while it refuses a level under it. The
// classes of torusRiseSeed models are not flat.
func TestFlatClasses(t *testing.T) {
	for _, i := range []int{115, 153, 232} {
		c := torusRidgeSeed(i)
		e, err := newDelayLFT(c.system(t))
		if err != nil {
			t.Fatal(err)
		}
		hf := newHighFrequencyGain(e)
		if len(hf.flat) != 2 || hf.flat[0] || !hf.flat[1] || hf.ridge == nil || hf.ridge.dim != 1 {
			t.Fatalf("seed %d: flat %v, ridge %v", i, hf.flat, hf.ridge)
		}
		sup := ridgeSup(t, c)
		if !hf.below(sup*(1+1e-9), hf.samples+1<<16) {
			t.Errorf("seed %d: below(sup·(1+1e-9)) failed", i)
		}
		if hf.below(sup*(1-1e-9), hf.samples+1<<16) {
			t.Errorf("seed %d: below(sup·(1−1e-9)) certified", i)
		}
	}
	for _, s := range []struct {
		kind string
		i    int
	}{{"irr3", 4}, {"mixed", 25}, {"irr2", 42}} {
		e, err := newDelayLFT(torusRiseSeed(s.kind, s.i).system(t))
		if err != nil {
			t.Fatal(err)
		}
		if hf := newHighFrequencyGain(e); hf.flat != nil || hf.ridge != nil {
			t.Errorf("%s %d: flat %v", s.kind, s.i, hf.flat)
		}
	}
}

// TestHinfNormTorusRidgeCrossing pins KPL4UB: torusRidgeSeed 232 peaks
// 2.6e-8 above its ridge sup S* at ω ≈ 3183, where a crossing of the ridge
// meets a phase of the second delay at which G rises above it. below could
// not certify S*·(1 + 1e-9) on the 2-D torus, the passage scan found
// nothing above S*·(1 + 1e-7), and torusFarPassages refused the flat
// direction, so peak gave up at 8015ead. The oracles are ridgeSup and
// ridgeCrossingOracle to 2e5.
func TestHinfNormTorusRidgeCrossing(t *testing.T) {
	c := torusRidgeSeed(232)
	e, err := newDelayLFT(c.system(t))
	if err != nil {
		t.Fatal(err)
	}
	l, res, err := e.stability()
	if err != nil || !res.stable {
		t.Fatalf("stability = %v, %v", res.stable, err)
	}
	got, w, err := e.peakOrZero(l, res)
	if err != nil {
		t.Fatal(err)
	}
	want, wWant := ridgeCrossingOracle(c, 2e5)
	if sup := ridgeSup(t, c); !(want > sup*(1+1e-8)) {
		t.Fatalf("oracle peak %.15g at %g not above the ridge sup %.15g", want, wWant, sup)
	}
	if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-6*wWant {
		t.Errorf("peak %.15g at %g, oracle %.15g at %g", got, w, want, wWant)
	}
	const maxCalls = 1 << 19
	if e.sigmaCalls > maxCalls {
		t.Errorf("%d samples of G, want <= %d", e.sigmaCalls, maxCalls)
	}
}
