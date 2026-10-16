package controlsys

import (
	"math"
	"math/cmplx"
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

// checkExpansionOrder checks the expansion of sys against gain at equal
// phases φ = ωh, where the remainder scales as ω^−(N+1).
func checkExpansionOrder(t *testing.T, sys *System, gain func(float64) float64, h float64) {
	t.Helper()
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
	if x == nil {
		t.Fatal("no expansion")
	}
	var errs, ws []float64
	for _, wc := range []float64{50, 400} {
		w := (1.234 + 2*math.Pi*math.Round(wc*h/(2*math.Pi))) / h
		err := math.Abs(gain(w) - x.model(w*h, 1/w))
		first := math.Abs(gain(w) - x.model(w*h, 0))
		if !(first > 1e3*err) {
			t.Fatalf("ω = %g: |G − model| = %g, |G − G∞| = %g", w, err, first)
		}
		errs, ws = append(errs, err), append(ws, w)
	}
	want := math.Pow(ws[1]/ws[0], expansionOrder+1)
	if r := errs[0] / errs[1]; !(r > want/2 && r < 2*want) {
		t.Fatalf("remainder ratio %g (%g, %g), want about %g (O(1/ω^(N+1)))", r, errs[0], errs[1], want)
	}
}

func TestHighFrequencyExpansionMatchesResponse(t *testing.T) {
	for _, c := range append([]expansionCase{expansionLimit}, expansionFinite...) {
		t.Run(c.name, func(t *testing.T) {
			sys, gain, _ := c.build(t)
			checkExpansionOrder(t, sys, gain, c.h)
		})
	}
}

// TestHighFrequencyExpansionLoopDynamics checks the expansion of a 1×2 gain
// whose delay loop has dynamics (C2, B2 ≠ 0, so every H22,k ≠ 0) against
// the response solved directly from the LFT equations
//
//	(jωI − A)·x − B2·Δ·z = B1,  −C2·x + (I − D22·Δ)·z = D21,  G = C1·x + D11 + D12·Δ·z.
func TestHighFrequencyExpansionLoopDynamics(t *testing.T) {
	const n, nu, nd = 2, 2, 2
	a := []float64{-1, 2, -0.7, -3}
	b1 := []float64{1, 0.2, 0.5, -1}
	b2 := []float64{0.3, -0.4, 0.6, 1}
	c1 := []float64{1, -0.4}
	c2 := []float64{0.5, 0.2, -0.3, 0.8}
	d11 := []float64{0.1, 0.05}
	d12 := []float64{-0.6, 0.7}
	d21 := []float64{0.25, 0.75, -0.5, 0.3}
	d22 := []float64{-0.4, 0.3, -0.35, -0.2}
	h := 0.37
	B := make([]float64, 0, n*(nu+nd))
	for i := range n {
		B = append(append(B, b1[i*nu:(i+1)*nu]...), b2[i*nd:(i+1)*nd]...)
	}
	C := append(append([]float64{}, c1...), c2...)
	D := append(append([]float64{}, d11...), d12...)
	for i := range nd {
		D = append(append(D, d21[i*nu:(i+1)*nu]...), d22[i*nd:(i+1)*nd]...)
	}
	H, err := NewFromSlices(n, nu+nd, 1+nd, a, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, []float64{h, 2 * h})
	if err != nil {
		t.Fatal(err)
	}
	gain := func(w float64) float64 {
		s := complex(0, w)
		dl := []complex128{cmplx.Exp(-s * complex(h, 0)), cmplx.Exp(-s * complex(2*h, 0))}
		const k = n + nd
		var norm2 float64
		for j := range nu {
			var m [k][k + 1]complex128
			for r := range n {
				for c := range n {
					m[r][c] = complex(-a[r*n+c], 0)
				}
				m[r][r] += s
				for c := range nd {
					m[r][n+c] = complex(-b2[r*nd+c], 0) * dl[c]
				}
				m[r][k] = complex(b1[r*nu+j], 0)
			}
			for r := range nd {
				for c := range n {
					m[n+r][c] = complex(-c2[r*n+c], 0)
				}
				for c := range nd {
					m[n+r][n+c] = complex(-d22[r*nd+c], 0) * dl[c]
				}
				m[n+r][n+r]++
				m[n+r][k] = complex(d21[r*nu+j], 0)
			}
			for c := range k {
				piv := c
				for r := c + 1; r < k; r++ {
					if cmplx.Abs(m[r][c]) > cmplx.Abs(m[piv][c]) {
						piv = r
					}
				}
				m[c], m[piv] = m[piv], m[c]
				for r := range k {
					if r != c {
						f := m[r][c] / m[c][c]
						for q := c; q <= k; q++ {
							m[r][q] -= f * m[c][q]
						}
					}
				}
			}
			g := complex(d11[j], 0)
			for c := range n {
				g += complex(c1[c], 0) * m[c][k] / m[c][c]
			}
			for c := range nd {
				g += complex(d12[c], 0) * dl[c] * m[n+c][k] / m[n+c][n+c]
			}
			norm2 += real(g)*real(g) + imag(g)*imag(g)
		}
		return math.Sqrt(norm2)
	}
	checkExpansionOrder(t, sys, gain, h)
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

// limitAtInfCases are delay models whose peak is the ω → ∞ limit, decided by
// limitVerdict: "limit" and "flat" with k = 2 ("flat" with κ small, so the
// expansion must reach far), "k20" with k = 20.
var limitAtInfCases = []struct {
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
}

// TestModelMaxMatchesFullScan checks the pruned modelMax against the model
// evaluated directly at every sample with every local maximum refined.
func TestModelMaxMatchesFullScan(t *testing.T) {
	for _, c := range limitAtInfCases {
		t.Run(c.name, func(t *testing.T) {
			sys, _, _ := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, c.k * c.h}, [2]float64{1, c.k})
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			x := newHighFrequencyExpansion(e, newHighFrequencyGain(e))
			n := 1024 * int(c.k)
			step := 2 * math.Pi / float64(n)
			for _, tt := range []float64{0.2, 0.05, 1e-2, 1e-4, 1e-8} {
				f := func(phi float64) float64 { return x.model(phi, tt) }
				v := make([]float64, n)
				for i := range v {
					v[i] = f(float64(i) * step)
				}
				want := 0.0
				for i := range v {
					if v[i] >= v[(i+n-1)%n] && v[i] >= v[(i+1)%n] {
						_, g := goldenMax(float64(i-1)*step, float64(i+1)*step, float64(i)*step, v[i], f)
						want = max(want, g)
					}
				}
				if got := x.modelMax(tt); math.Abs(got-want) > 1e-12*want {
					t.Errorf("t = %g: modelMax = %.16g, full scan %.16g", tt, got, want)
				}
			}
		})
	}
}

func BenchmarkHinfNorm_DelayLimitAtInf(b *testing.B) {
	for _, c := range limitAtInfCases {
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
