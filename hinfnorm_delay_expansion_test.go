package controlsys

import (
	"math"
	"testing"
)

type expansionCase struct {
	name          string
	d22, d12, d21 []float64
	h             float64
}

var (
	// sup |G∞| is attained off φ = 0, π, where a = Re⟨g0, g1⟩ ≠ 0: the
	// finite peaks exceed it by about |a|/(lim·ω) and the highest lies
	// several periods past the grid end.
	expansionFinite = []expansionCase{
		{"near-90",
			[]float64{0.7649551391810011, -0.4053354642068646, -0.5986288790758891, -0.735255719141454},
			[]float64{-0.14804120333685122, 0.8436249853687099}, []float64{-0.40213426719214174, -0.5844303786493328},
			0.3686756259842685},
		{"near-410",
			[]float64{-0.47675042569597786, -0.7643529185038457, 0.6634435446928469, -0.4697636633641407},
			[]float64{-0.477317332593425, 0.9680775762045155}, []float64{0.9231522571301278, 0.678449994147619},
			0.11037823685884307},
	}
	// sup |G∞| is attained at φ = π with κ < 0: every passage stays below.
	expansionLimit = expansionCase{"limit",
		[]float64{-0.6008003587828393, 0.2739929683353646, -0.6159222999747516, -0.37189479027245975},
		[]float64{-0.6331344630555211, 0.728526179491567}, []float64{0.24127665933031084, 0.7586790465022555},
		0.4851105119841035}
)

func (c expansionCase) build(t testing.TB) (*System, func(float64) float64, float64) {
	t.Helper()
	sys, gain, gInf := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, 2 * c.h}, [2]float64{1, 2})
	const k = 1 << 18
	lim := 0.0
	for i := range k {
		lim = max(lim, gInf(2*math.Pi*float64(i)/k))
	}
	return sys, gain, lim
}

// passageMax returns the largest |G| over the passage of φ = ωh through phi
// nearest wc, on a dense grid refined by golden-section search.
func passageMax(gain func(float64) float64, h, phi, wc float64) (float64, float64) {
	k := math.Round((wc*h - phi) / (2 * math.Pi))
	w0, dw := (phi+2*math.Pi*k)/h, 0.5/h
	best, bw := 0.0, 0.0
	for j := -400; j <= 400; j++ {
		w := w0 + dw*float64(j)/400
		if g := gain(w); g > best {
			best, bw = g, w
		}
	}
	bw, best = goldenMax(bw-dw/400, bw+dw/400, bw, best, gain)
	return best, bw
}

func TestHighFrequencyExpansionMatchesResponse(t *testing.T) {
	for _, c := range append([]expansionCase{expansionLimit}, expansionFinite...) {
		t.Run(c.name, func(t *testing.T) {
			sys, gain, _ := c.build(t)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
			if x == nil {
				t.Fatal("no expansion")
			}
			var errs []float64
			for _, w := range []float64{1000.37, 10000.37} {
				err := math.Abs(gain(w) - x.model(w*c.h, 1/w))
				first := math.Abs(gain(w) - x.model(w*c.h, 0))
				if !(first > 1e3*err) {
					t.Fatalf("ω = %g: |G − model| = %g, |G − G∞| = %g", w, err, first)
				}
				errs = append(errs, err)
			}
			if r := errs[0] / errs[1]; !(r > 300) {
				t.Fatalf("remainder ratio over a decade %g, want about 1e3 (O(1/ω³))", r)
			}
		})
	}
}

func TestHinfNormFinitePeakPastLimitScan(t *testing.T) {
	for _, c := range expansionFinite {
		t.Run(c.name, func(t *testing.T) {
			sys, gain, lim := c.build(t)
			want, wWant := oraclePeak(gain, 600, 1e-3)
			if !(want > lim*(1+1e-5)) {
				t.Fatalf("oracle peak %.12g must exceed sup |G∞| %.12g", want, lim)
			}
			got, w, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-3 {
				t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
			}
		})
	}
}

func TestHinfNormLimitBelowEveryPassage(t *testing.T) {
	c := expansionLimit
	sys, gain, lim := c.build(t)
	var prev float64
	for _, wc := range []float64{1e3, 1e4, 1e5} {
		m, w := passageMax(gain, c.h, math.Pi, wc)
		excess := (m - lim) * w * w
		if !(excess < 0) {
			t.Fatalf("passage at %g peaks at %.15g above sup |G∞| %.15g", w, m, lim)
		}
		if wc > 1e3 && math.Abs(excess-prev) > 0.02*math.Abs(prev) {
			t.Fatalf("(max − lim)·ω² = %g at %g, %g a decade earlier: not in the 1/ω² regime", excess, w, prev)
		}
		prev = excess
	}
	if want, _ := oraclePeak(gain, 3000, 2e-3); !(want < lim) {
		t.Fatalf("oracle peak %.12g reaches sup |G∞| %.12g", want, lim)
	}
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
	finite, W, ok := x.limitVerdict(lim, 1e-10, 10, 1e6)
	if !ok || finite || !(W < 1e4) {
		t.Fatalf("limitVerdict = %v, %g, %v; want +Inf certified from a finite horizon", finite, W, ok)
	}
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("HinfNorm: %v", err)
	}
	if !math.IsInf(w, 1) || math.Abs(got-lim) > 1e-9*lim {
		t.Fatalf("HinfNorm = (%.15g, %g), want (%.15g, +Inf)", got, w, lim)
	}
}

// BenchmarkHinfNorm_DelayLimitAtInf times delay models whose peak is the
// ω → ∞ limit, decided by limitVerdict: "limit" and "flat" with k = 2 ("flat"
// with κ small, so the expansion must reach far), "k20" with k = 20.
func BenchmarkHinfNorm_DelayLimitAtInf(b *testing.B) {
	for _, c := range []struct {
		expansionCase
		k float64
	}{
		{expansionLimit, 2},
		{expansionCase{"flat",
			[]float64{0.5888561770149723, -0.10453534772349099, 0.26009762791254687, -0.4187778122618928},
			[]float64{-1.3972940735741266, 0.09637692295337041}, []float64{-1.601893769323554, 0.40406439147179873},
			0.3547077050852388}, 2},
		{expansionCase{"k20",
			[]float64{-0.6267854986926139, -0.22753476407416448, 0.7604859326403655, 0.3445225407726179},
			[]float64{2.0994008736286496, -0.7771379653940502}, []float64{2.8964548762636153, -1.852398922038503},
			0.23582769791420977}, 20},
	} {
		sys, _, _ := feedthroughDifferenceLoop(b, c.d22, c.d12, c.d21, []float64{c.h, c.k * c.h}, [2]float64{1, c.k})
		b.Run(c.name, func(b *testing.B) {
			b.ReportAllocs()
			for b.Loop() {
				if _, w, err := HinfNorm(sys); err != nil || !math.IsInf(w, 1) {
					b.Fatal(w, err)
				}
			}
		})
	}
}
