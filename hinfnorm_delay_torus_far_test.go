package controlsys

import (
	"cmp"
	"math"
	"math/big"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// thetaGain is σ_max(G) of z at s = jw, w = ∞ dropping the states, with
// class c at phase th[c].
func (z torusRiseCase) thetaGain(w float64, th []float64) float64 {
	ph := make([]float64, z.nd)
	for i := range ph {
		ph[i] = float64(z.k[i]) * th[z.cls[i]]
	}
	return z.gain(w, ph)
}

// torusArgSup is torusSup with its maximiser.
func (z torusRiseCase) torusArgSup() ([]float64, float64) {
	n := map[int]int{2: 360, 3: 64}[z.dim]
	step := 2 * math.Pi / float64(n)
	type cell struct {
		v  float64
		th []float64
	}
	var cells []cell
	th := make([]float64, z.dim)
	total := int(math.Pow(float64(n), float64(z.dim)))
	for idx := range total {
		r := idx
		for c := range z.dim {
			th[c] = float64(r%n) * step
			r /= n
		}
		cells = append(cells, cell{z.thetaGain(math.Inf(1), th), slices.Clone(th)})
	}
	slices.SortFunc(cells, func(a, b cell) int { return cmp.Compare(b.v, a.v) })
	best, arg := 0.0, []float64(nil)
	for _, c := range cells[:16] {
		x, v := c.th, c.v
		for s := step; s > 1e-13; {
			moved := false
			for k := range z.dim {
				for _, sg := range []float64{1, -1} {
					y := slices.Clone(x)
					y[k] += sg * s
					if f := z.thetaGain(math.Inf(1), y); f > v {
						x, v, moved = y, f, true
					}
				}
			}
			if !moved {
				s /= 2
			}
		}
		if v > best {
			best, arg = v, x
		}
	}
	return arg, best
}

// passageOracle returns the largest finite sample of σ_max(G(jω)) of z up
// to wMax: a dense scan to w0 = 5000, then every crossing ω_n of the class 0
// phase through θ* or −θ*, θ* the torus maximiser, where every other class
// is within δ = 2·√(c/(λ·ω)) of it, sampled at 81 points across ±δ/h_0, the
// largest refined by golden section. c is the measured rise
// (sup_θ σ_max(G(jω, θ)) − S*)·ω near ±θ* at ω = 10⁴, 10⁵ from the exact
// response, and λ the least eigenvalue of −½∇²S at θ*.
func passageOracle(z torusRiseCase, th []float64, sup, wMax float64) (float64, float64) {
	neg := make([]float64, z.dim)
	for c := range th {
		neg[c] = -th[c]
	}
	rise := func(w float64, t0 []float64) float64 {
		_, v := torusAscent(func(x []float64) float64 { return z.thetaGain(w, x) }, t0, z.thetaGain(w, t0))
		return (v - sup) * w
	}
	c := max(rise(1e4, th), rise(1e4, neg), rise(1e5, th), rise(1e5, neg), 1e-6)
	const hs = 1e-4
	hess := mat.NewSymDense(z.dim, nil)
	at := func(i, j int, di, dj float64) float64 {
		y := slices.Clone(th)
		y[i] += di
		y[j] += dj
		return z.thetaGain(math.Inf(1), y)
	}
	for i := range z.dim {
		hess.SetSym(i, i, -(at(i, i, hs, 0)-2*sup+at(i, i, -hs, 0))/(2*hs*hs))
		for j := range i {
			hess.SetSym(i, j, -(at(i, j, hs, hs)-at(i, j, hs, -hs)-at(i, j, -hs, hs)+at(i, j, -hs, -hs))/(8*hs*hs))
		}
	}
	var es mat.EigenSym
	es.Factorize(hess, false)
	lam := slices.Min(es.Values(nil))
	h := make([]float64, z.dim)
	for i, cl := range z.cls {
		h[cl] = z.tau[i] / float64(z.k[i])
	}
	const w0 = 5000
	fin, wf := denseScanMax(z.sigma, w0, math.Pi/(16*slices.Max(h)))
	for _, tg := range [][]float64{th, neg} {
		for n := math.Ceil((w0*h[0] - tg[0]) / (2 * math.Pi)); ; n++ {
			wc := (tg[0] + 2*math.Pi*n) / h[0]
			if wc > wMax {
				break
			}
			d := 2 * math.Sqrt(c/(lam*wc))
			near := true
			for k := 1; k < z.dim; k++ {
				x := wc*h[k] - tg[k]
				x -= 2 * math.Pi * math.Round(x/(2*math.Pi))
				if math.Abs(x) > d*(1+h[k]/h[0]) {
					near = false
					break
				}
			}
			if !near {
				continue
			}
			half := d / h[0]
			for i := -40; i <= 40; i++ {
				w := wc + half*float64(i)/40
				if v := z.sigma(w); v > fin {
					fin, wf = v, w
				}
			}
		}
	}
	dw := 1e-3 / slices.Max(h)
	wf, fin = goldenMax(wf-dw, wf+dw, wf, fin, z.sigma)
	return fin, wf
}

// TestHinfNormTorusFarPassages pins ADKXZX for maximisers θ* of σ_max(G∞)
// with a = Re⟨g0, g1⟩ ≠ 0, whose passages rise above S* = sup σ_max(G∞) by
// O(1/ω) only where they come close to θ*: "mixed" 35, whose passage
// window was narrower than the grid step and held no sample, and 172, past
// the old cap of the passage scan; "irr3" 9 and 45, three classes, whose
// finite peak lies past the grid span (8.5e5, 1.9e6); and "irr3" 1 and 23,
// whose passages stay below S* up to the horizon of the expansion, so the
// norm is S* at ω = +Inf. All gave up at 66bac00. The oracles are the dense
// torus grid for S* and passageOracle for the finite peaks.
func TestHinfNormTorusFarPassages(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	for _, c := range []struct {
		kind   string
		i      int
		wMax   float64
		finite bool
	}{
		{"mixed", 35, 2e5, true},
		{"mixed", 172, 2e5, true},
		{"irr3", 9, 1e6, true},
		{"irr3", 45, 2.5e6, true},
		{"irr3", 1, 1e7, false},
		{"irr3", 23, 1e7, false},
	} {
		z := torusRiseSeed(c.kind, c.i)
		th, sup := z.torusArgSup()
		fin, wf := passageOracle(z, th, sup, c.wMax)
		if lo, wl := denseScanMax(z.sigma, 400, 1e-3); lo > fin {
			fin, wf = lo, wl
		}
		got, w, err := HinfNorm(z.system(t))
		if err != nil {
			t.Errorf("%s %d: %v", c.kind, c.i, err)
			continue
		}
		if c.finite {
			if !(fin > sup*(1+1e-9)) {
				t.Fatalf("%s %d: oracle peak %.15g at %g not above torus sup %.15g", c.kind, c.i, fin, wf, sup)
			}
			if math.Abs(got-fin) > 2e-9*fin || math.IsInf(w, 1) {
				t.Errorf("%s %d: norm %.15g at %g, want finite peak %.15g at %g", c.kind, c.i, got, w, fin, wf)
			}
			continue
		}
		if math.Abs(got-sup) > 1e-9*sup || !math.IsInf(w, 1) {
			t.Errorf("%s %d: norm %.15g at %g, want torus sup %.15g at +Inf", c.kind, c.i, got, w, sup)
		}
		if fin > sup*(1+peakCertTol) {
			t.Errorf("%s %d: finite sample %.15g at %g above torus sup %.15g", c.kind, c.i, fin, wf, sup)
		}
	}
}

// TestNearCrossings checks nearCrossings against a scan of every n with
// ‖n·α + γ‖ computed in exact rational arithmetic, over ranges up to 10⁸
// past n0 ≈ 10⁷…10⁸ and widths eps from about one solution per range to
// thousands.
func TestNearCrossings(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	dist := func(n int64, a, g float64) float64 {
		v := new(big.Float).SetPrec(256).SetInt64(n)
		v.Mul(v, new(big.Float).SetPrec(256).SetFloat64(a))
		v.Add(v, new(big.Float).SetPrec(256).SetFloat64(g))
		i, _ := v.Int(nil)
		f := new(big.Float).SetPrec(256).Sub(v, new(big.Float).SetPrec(256).SetInt(i))
		x, _ := f.Float64()
		return math.Min(x, 1-x)
	}
	total := 0
	for trial := range 40 {
		alpha := 0.3 + 3*rng.Float64()
		gamma := rng.Float64() - 0.5
		n0 := int64(1e7 * (1 + 9*rng.Float64()))
		width := int64(1e5)
		if trial%4 == 3 {
			width = 64
		}
		eps := math.Pow(10, -1-3*rng.Float64()) * 1e3 / float64(width)
		n1 := n0 + width
		var want []int64
		for n := n0; n <= n1; n++ {
			v := float64(n)*alpha + gamma
			if math.Abs(v-math.Round(v)) > 2*eps+1e-6 {
				continue
			}
			if dist(n, alpha, gamma) <= eps*(1-1e-9) {
				want = append(want, n)
			}
		}
		total += len(want)
		var got []int64
		if !nearCrossings(alpha, gamma, n0, n1, eps, func(n int64) bool {
			got = append(got, n)
			return true
		}) {
			t.Fatalf("trial %d: gave up", trial)
		}
		if !slices.IsSorted(got) {
			t.Errorf("trial %d: not ascending", trial)
		}
		for _, n := range want {
			if _, ok := slices.BinarySearch(got, n); !ok {
				t.Errorf("trial %d α=%g γ=%g eps=%g: missed n=%d (dist %g)", trial, alpha, gamma, eps, n, dist(n, alpha, gamma))
			}
		}
		for _, n := range got {
			if d := dist(n, alpha, gamma); d > eps*(1+1e-6) {
				t.Errorf("trial %d: n=%d at %g > eps %g", trial, n, d, eps)
			}
		}
	}
	if total < 200 {
		t.Fatalf("only %d solutions over all trials", total)
	}
}

// TestHinfNormTorusFarAfterFind pins BEFIOK: the passage scan finds a
// sample above S* = sup σ_max(G∞) within the grid span, but the tail cannot
// be closed at that peak inside the span — for "mixed" 155 a higher passage
// peak lies past it (1.8e6, span end 5e5) — so peak gave up with "delay
// grid exceeds" at 8015ead. KPL4UB "mixed" 44 and "irr2" 42 gave up the
// same way, their peak a narrow resonance below ω = 400 (S* within ~1e-5).
// The oracles are the dense torus grid for S* and, for the finite peak,
// passageOracle with a fine scan to ω = 400.
func TestHinfNormTorusFarAfterFind(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	for _, c := range []struct {
		kind string
		i    int
		wMax float64
	}{
		{"mixed", 155, 4e6},
		{"irr3", 73, 1e6},
		{"mixed", 44, 1e6},
		{"irr2", 42, 1e6},
	} {
		z := torusRiseSeed(c.kind, c.i)
		th, sup := z.torusArgSup()
		fin, wf := passageOracle(z, th, sup, c.wMax)
		if lo, wl := denseScanMax(z.sigma, 400, 1e-3); lo > fin {
			fin, wf = lo, wl
		}
		if !(fin > sup*(1+1e-9)) {
			t.Fatalf("%s %d: oracle peak %.15g at %g not above torus sup %.15g", c.kind, c.i, fin, wf, sup)
		}
		got, w, err := HinfNorm(z.system(t))
		if err != nil {
			t.Errorf("%s %d: %v", c.kind, c.i, err)
			continue
		}
		if math.Abs(got-fin) > 2e-9*fin || math.IsInf(w, 1) {
			t.Errorf("%s %d: norm %.15g at %g, want finite peak %.15g at %g", c.kind, c.i, got, w, fin, wf)
		}
	}
}
