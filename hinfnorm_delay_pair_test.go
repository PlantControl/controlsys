package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// topGramEig returns the largest eigenvalue of G^H·G for the p×m flat G by
// mat.EigenSym on its real embedding, independent of gramPair.
func topGramEig(t *testing.T, g []complex128, p, m int) float64 {
	t.Helper()
	b := mat.NewSymDense(2*m, nil)
	for i := range m {
		for j := range m {
			var s complex128
			for r := range p {
				s += complex(real(g[r*m+i]), -imag(g[r*m+i])) * g[r*m+j]
			}
			b.SetSym(i, j, real(s))
			b.SetSym(m+i, m+j, real(s))
			b.SetSym(m+i, j, imag(s))
		}
	}
	var e mat.EigenSym
	if !e.Factorize(b, false) {
		t.Fatal("EigenSym failed")
	}
	v := e.Values(nil)
	return v[len(v)-1]
}

// TestGramPairExpansion checks λ0, λ1, λ2 of gramPair against central
// differences in t of the top eigenvalue of G(t)^H·G(t), G(t) = g0 + t·g1 +
// t²·g2, from an independent eigensolver, for tall, wide and square gains.
func TestGramPairExpansion(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 5))
	for _, sz := range [][2]int{{2, 2}, {3, 2}, {2, 3}, {4, 3}, {3, 5}} {
		p, m := sz[0], sz[1]
		gp := newGramPair(p, m)
		for range 20 {
			g := make([][]complex128, 3)
			for k := range g {
				g[k] = make([]complex128, p*m)
				for i := range g[k] {
					g[k][i] = complex(rng.NormFloat64(), rng.NormFloat64())
				}
			}
			l0, l1, l2, gap, ok := gp.expand(g[0], g[1], g[2])
			if !ok || !(gap > 1e-3) {
				t.Fatalf("%dx%d: ok = %v, gap = %g", p, m, ok, gap)
			}
			at := func(tt float64) float64 {
				gt := make([]complex128, p*m)
				for i := range gt {
					gt[i] = g[0][i] + complex(tt, 0)*g[1][i] + complex(tt*tt, 0)*g[2][i]
				}
				return topGramEig(t, gt, p, m)
			}
			const h = 1e-3
			e0, ep, em, ep2, em2 := at(0), at(h), at(-h), at(2*h), at(-2*h)
			d1 := (-ep2 + 8*ep - 8*em + em2) / (12 * h)
			d2 := (-ep2 + 16*ep - 30*e0 + 16*em - em2) / (12 * h * h) / 2
			scale := e0 + math.Abs(d1) + math.Abs(d2)
			if math.Abs(l0-e0) > 1e-12*scale || math.Abs(l1-d1) > 1e-7*scale || math.Abs(l2-d2) > 1e-5*scale {
				t.Errorf("%dx%d: (λ0, λ1, λ2) = (%.12g, %.12g, %.12g), differences (%.12g, %.12g, %.12g)", p, m, l0, l1, l2, e0, d1, d2)
			}
		}
	}
}

// TestGramPairRepeated checks that a repeated σ_max(g0), here g0 = 2·I
// plus a smaller block, is reported as such.
func TestGramPairRepeated(t *testing.T) {
	g0 := []complex128{2, 0, 0, 0, 2, 0, 0, 0, 1}
	g1 := []complex128{0, 1i, 0, 1, 0, 0, 0, 0, 0}
	if _, _, _, gap, ok := newGramPair(3, 3).expand(g0, g1, make([]complex128, 9)); ok || gap > repeatedGap {
		t.Fatalf("ok = %v, gap = %g; want repeated", ok, gap)
	}
}

// matrixWraps are the constant U, V, d of the 2×2 gains U·diag(G, d)·V.
var matrixWraps = []struct {
	name string
	d    float64
	u, v []float64
}{
	{"diag", 0.01, []float64{1, 0, 0, 1}, []float64{1, 0, 0, 1}},
	{"coupled", 0.3, []float64{1, 0.4, -0.3, 0.8}, []float64{0.7, -0.2, 0.5, 1.1}},
}

// TestLimitVerdictMatrixKappa checks that limitVerdict decides +Inf for 2×2
// gains built from limitAtInfCases with a = 0 at a maximiser φ* ∈ {0, π},
// from κ of gramPair, where it was undecided. The oracle: the passage maxima
// of the closed-form σ_max through φ* stay below sup σ_max(G∞), with
// (max − lim)·ω² settling to κ/(2·lim) < 0.
func TestLimitVerdictMatrixKappa(t *testing.T) {
	for _, c := range limitAtInfCases[:3] {
		for _, m := range matrixWraps {
			t.Run(c.name+"/"+m.name, func(t *testing.T) {
				sys, sigma := c.matrixGain(t, m.d, m.u, m.v)
				e, err := newDelayLFT(sys)
				if err != nil {
					t.Fatal(err)
				}
				hf := newHighFrequencyGain(e)
				x := newHighFrequencyExpansion(e, hf)
				if x == nil || x.vector {
					t.Fatal("no matrix expansion")
				}
				lim := hf.onePhaseSup()
				phi, far := 0.0, 0.0
				for _, p := range []float64{0, math.Pi} {
					if v, _ := passageMax(sigma, c.h, p, 1e5); v > far {
						phi, far = p, v
					}
				}
				if math.Abs(far-lim) > 1e-6*lim {
					t.Fatalf("passage at 1e5 through φ = %g: %.15g, sup σ_max(G∞) %.15g", phi, far, lim)
				}
				var prev float64
				for _, wc := range []float64{1e3, 1e4, 1e5} {
					v, w := passageMax(sigma, c.h, phi, wc)
					excess := (v - lim) * w * w
					if !(excess < 0) {
						t.Fatalf("passage at %g peaks at %.15g above %.15g", w, v, lim)
					}
					if wc > 1e3 && math.Abs(excess-prev) > 0.05*math.Abs(prev) {
						t.Fatalf("(max − lim)·ω² = %g at %g, %g a decade earlier", excess, w, prev)
					}
					prev = excess
				}
				finite, W, ok := x.limitVerdict(lim, 1e-10, 10, 1e6)
				if !ok || finite || !(W > 0) {
					t.Fatalf("limitVerdict = %v, %g, %v; want +Inf from a finite horizon", finite, W, ok)
				}
				got, w, err := HinfNorm(sys)
				if err != nil {
					t.Fatalf("HinfNorm: %v", err)
				}
				if !math.IsInf(w, 1) || math.Abs(got-lim) > 1e-9*lim {
					t.Fatalf("HinfNorm = (%.15g, %g), want (%.15g, +Inf)", got, w, lim)
				}
			})
		}
	}
}

// TestLimitVerdictMatrixKappaPositive checks κ > 0 at φ* = 0 for 2×2 gains
// built from G(s) = kr/(s² + 2ζω0·s + ω0²) + c·z/(1 − d·z), z = e^{−sτ},
// kr < 0, c, d > 0: sup |G∞| = c/(1 − d) at z = 1, a = 0 and a′ = 0
// there, and G − G∞ = −kr·t² + O(t³) is real, so κ = −2·lim·kr and, for
// the diagonal wrap, the passage maxima through φ = 0 exceed lim with
// (max − lim)·ω² → −kr.
func TestLimitVerdictMatrixKappaPositive(t *testing.T) {
	const kr, cc, dd, tau = -0.3, 0.8, 0.4, 0.37
	const w0, zeta = 1.3, 0.05
	sys, _ := delayResonanceLoop(t, []float64{tau}, kr, []float64{cc}, []float64{dd})
	resp := func(w float64) complex128 {
		s := complex(0, w)
		z := cmplx.Exp(-s * complex(tau, 0))
		return complex(kr, 0)/(s*s+complex(2*zeta*w0, 0)*s+complex(w0*w0, 0)) + complex(cc, 0)*z/(1-complex(dd, 0)*z)
	}
	lim := cc / (1 - dd)
	for _, m := range matrixWraps {
		t.Run(m.name, func(t *testing.T) {
			ms, sigma := wrapMatrixGain(t, sys, resp, m.d, m.u, m.v)
			e, err := newDelayLFT(ms)
			if err != nil {
				t.Fatal(err)
			}
			hf := newHighFrequencyGain(e)
			x := newHighFrequencyExpansion(e, hf)
			if x == nil || x.vector {
				t.Fatal("no matrix expansion")
			}
			limM := hf.onePhaseSup()
			if m.name == "diag" {
				if math.Abs(limM-lim) > 1e-12*lim {
					t.Fatalf("sup σ_max(G∞) = %.15g, want %.15g", limM, lim)
				}
				for _, wc := range []float64{1e3, 1e4} {
					v, w := passageMax(sigma, tau, 0, wc)
					if r := (v - lim) * w * w; math.Abs(r+kr) > 0.02*math.Abs(kr) {
						t.Fatalf("(max − lim)·ω² = %g at %g, want %g", r, w, -kr)
					}
				}
			}
			finite, _, ok := x.limitVerdict(limM, 1e-10, 10, 1e6)
			if !ok || !finite {
				t.Fatalf("limitVerdict = %v, %v; want finite peaks above the limit", finite, ok)
			}
			want, wWant := oraclePeak(sigma, 200, 1e-3)
			got, w, err := HinfNorm(ms)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-3 {
				t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
			}
		})
	}
}

// wrapMatrixGain returns U·diag(G, d)·V for the SISO sys with response
// resp, and σ_max of its response in closed form.
func wrapMatrixGain(t testing.TB, sys *System, resp func(float64) complex128, d float64, u, v []float64) (*System, func(float64) float64) {
	t.Helper()
	gain := func(r, c int, x []float64) *System {
		g, err := NewGain(mat.NewDense(r, c, x), 0)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	ap, err := Append(sys, gain(1, 1, []float64{d}))
	if err != nil {
		t.Fatal(err)
	}
	if ap, err = Series(gain(2, 2, v), ap); err != nil {
		t.Fatal(err)
	}
	if ap, err = Series(ap, gain(2, 2, u)); err != nil {
		t.Fatal(err)
	}
	sigma := func(w float64) float64 {
		g := [2]complex128{resp(w), complex(d, 0)}
		var m [2][2]complex128
		for i := range 2 {
			for j := range 2 {
				for k := range 2 {
					m[i][j] += complex(u[i*2+k]*v[k*2+j], 0) * g[k]
				}
			}
		}
		return oracleSigmaMax2(m)
	}
	return ap, sigma
}

func BenchmarkHinfNorm_DelayMatrixLimitAtInf(b *testing.B) {
	for _, c := range limitAtInfCases[:3] {
		sys, resp, _ := feedthroughDifferenceResponse(b, c.d22, c.d12, c.d21, []float64{c.h, 2 * c.h}, [2]float64{1, 2})
		for _, m := range matrixWraps {
			ms, _ := wrapMatrixGain(b, sys, resp, m.d, m.u, m.v)
			b.Run(c.name+"/"+m.name, func(b *testing.B) {
				b.ReportAllocs()
				for b.Loop() {
					if _, w, err := HinfNorm(ms); err != nil || !math.IsInf(w, 1) {
						b.Fatal(w, err)
					}
				}
			})
		}
	}
}
