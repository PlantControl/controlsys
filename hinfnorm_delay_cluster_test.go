package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// randUnitary returns an n×n unitary flat by Gram–Schmidt on a complex
// Gaussian matrix.
func randUnitary(rng *rand.Rand, n int) []complex128 {
	u := make([]complex128, n*n)
	for j := range n {
		for i := range n {
			u[i*n+j] = complex(rng.NormFloat64(), rng.NormFloat64())
		}
		for range 2 {
			for l := range j {
				var d complex128
				for i := range n {
					d += cmplx.Conj(u[i*n+l]) * u[i*n+j]
				}
				for i := range n {
					u[i*n+j] -= d * u[i*n+l]
				}
			}
		}
		var s float64
		for i := range n {
			s += real(u[i*n+j])*real(u[i*n+j]) + imag(u[i*n+j])*imag(u[i*n+j])
		}
		for i := range n {
			u[i*n+j] /= complex(math.Sqrt(s), 0)
		}
	}
	return u
}

// usw returns U·X·W^H for the p×p U, p×m X and m×m W.
func usw(u, x, w []complex128, p, m int) []complex128 {
	out := make([]complex128, p*m)
	for i := range p {
		for j := range m {
			var s complex128
			for a := range p {
				for b := range m {
					s += u[i*p+a] * x[a*m+b] * cmplx.Conj(w[j*m+b])
				}
			}
			out[i*m+j] = s
		}
	}
	return out
}

// TestGramPairClusterPencil checks clusterPencil for g0 = U·S·W^H with
// σ_max(g0) exactly double, against one-sided differences in t > 0 of the
// top eigenvalue of G_s(t)^H·G_s(t) along paths, G_s = g0 + t·(g1 + s·d0)
// + t²·(g2 + s·d1 + s²/2·dd0), from an independent eigensolver: λ0 + t·μ1
// + t²·λ_max(K(s)). The top-left 2×2 block of W^H·d0 (in U's frame) is
// anti-Hermitian, as at a maximiser of σ_max(g0(φ)); that of g1 is random
// ("split", μ1 simple) or zero ("flat", μ1 = 0 on the whole cluster, the
// resolvent outside it still active).
func TestGramPairClusterPencil(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 13))
	cplx := func() complex128 { return complex(rng.NormFloat64(), rng.NormFloat64()) }
	for _, sz := range [][2]int{{3, 3}, {4, 3}, {3, 4}} {
		for _, flat := range []bool{false, true} {
			p, m := sz[0], sz[1]
			u, w := randUnitary(rng, p), randUnitary(rng, m)
			sv := make([]complex128, p*m)
			sv[0], sv[m+1], sv[2*m+2] = 1.5, 1.5, 0.7
			rand4 := func(zeroTop, antiTop bool) []complex128 {
				x := make([]complex128, p*m)
				for i := range x {
					x[i] = cplx()
				}
				if zeroTop {
					x[0], x[1], x[m], x[m+1] = 0, 0, 0, 0
				}
				if antiTop {
					x[0], x[m+1] = complex(0, imag(x[0])), complex(0, imag(x[m+1]))
					x[m] = -cmplx.Conj(x[1])
				}
				return usw(u, x, w, p, m)
			}
			g0 := usw(u, sv, w, p, m)
			g1, g2 := rand4(flat, false), rand4(false, false)
			d0, d1, dd0 := rand4(false, true), rand4(false, false), rand4(false, false)
			q, r, mu1, _, ok := newGramPair(p, m).clusterPencil(g0, g1, g2, d0, d1, dd0, 1e-9)
			if !ok {
				t.Fatalf("%dx%d flat=%v: clusterPencil failed", p, m, flat)
			}
			if wantR := map[bool]int{false: 1, true: 2}[flat]; r != wantR {
				t.Fatalf("%dx%d flat=%v: r = %d, want %d", p, m, flat, r, wantR)
			}
			for _, s := range []float64{-0.8, 0, 0.5, 1.7} {
				at := func(tt float64) float64 {
					g := make([]complex128, p*m)
					for i := range g {
						g1s := g1[i] + complex(s, 0)*d0[i]
						g2s := g2[i] + complex(s, 0)*d1[i] + complex(s*s/2, 0)*dd0[i]
						g[i] = g0[i] + complex(tt, 0)*g1s + complex(tt*tt, 0)*g2s
					}
					return topGramEig(t, g, p, m)
				}
				l0 := at(0)
				if math.Abs(l0-2.25) > 1e-12 {
					t.Fatalf("λ0 = %.15g, want 2.25", l0)
				}
				const h = 2e-4
				f := func(tt float64) float64 { return (at(tt) - l0 - tt*mu1) / (tt * tt) }
				want := (8*f(h) - 6*f(2*h) + f(4*h)) / 3
				kmat := make([]complex128, r*r)
				for i := range kmat {
					kmat[i] = q[0][i] + complex(s, 0)*q[1][i] + complex(s*s, 0)*q[2][i]
				}
				vals, _, _, _ := hermTop(kmat, r, 0, 0)
				if math.Abs(vals[0]-want) > 1e-5*(1+math.Abs(want)) {
					t.Errorf("%dx%d flat=%v s=%g: λ_max(K) = %.10g, one-sided difference %.10g", p, m, flat, s, vals[0], want)
				}
			}
			if flat && math.Abs(mu1) > 1e-12 {
				t.Errorf("%dx%d flat: μ1 = %g, want 0", p, m, mu1)
			}
		}
	}
}

// clusterGain is a 2×2 gain U·diag(G1, G2)·V with orthogonal U, V and
// σ_max(G1∞) = σ_max(G2∞) at φ*, so σ_max(G∞(φ*)) is double; sigma is
// max(|G1|, |G2|) in closed form.
type clusterGain struct {
	name   string
	build  func(t *testing.T) (sys1, sys2 *System, g1, g2 func(float64) float64, h float64)
	phi    float64
	finite bool
}

// rotations are the orthogonal U, V of the cluster gains.
var rotations = [][2][]float64{
	{{1, 0, 0, 1}, {1, 0, 0, 1}},
	{{0.6, -0.8, 0.8, 0.6}, {math.Cos(1.1), math.Sin(1.1), -math.Sin(1.1), math.Cos(1.1)}},
}

func (c clusterGain) system(t *testing.T, u, v []float64) (*System, func(float64) float64, float64) {
	t.Helper()
	s1, s2, g1, g2, h := c.build(t)
	ap, err := Append(s1, s2)
	if err != nil {
		t.Fatal(err)
	}
	gain := func(x []float64) *System {
		g, err := NewGain(mat.NewDense(2, 2, x), 0)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}
	if ap, err = Series(gain(v), ap); err != nil {
		t.Fatal(err)
	}
	if ap, err = Series(ap, gain(u)); err != nil {
		t.Fatal(err)
	}
	return ap, func(w float64) float64 { return math.Max(g1(w), g2(w)) }, h
}

// resonancePair is G_i = kr_i/(s² + 2ζω0·s + ω0²) + c_i·z/(1 − d_i·z),
// z = e^{−sτ}, c_i/(1 − d_i) = 4/3 at φ* = 0; with d1 ≠ d2 the branches
// cross there only. κ_i = −2·lim·kr_i (a = a′ = 0).
func resonancePair(kr1, kr2, c2, d2 float64) func(t *testing.T) (*System, *System, func(float64) float64, func(float64) float64, float64) {
	return func(t *testing.T) (*System, *System, func(float64) float64, func(float64) float64, float64) {
		const tau = 0.37
		s1, g1 := delayResonanceLoop(t, []float64{tau}, kr1, []float64{0.8}, []float64{0.4})
		s2, g2 := delayResonanceLoop(t, []float64{tau}, kr2, []float64{c2}, []float64{d2})
		return s1, s2, g1, g2, tau
	}
}

// farChannel is G = c·z/(1 − d·z) + kr/(s + p)² + k4/(s + p)⁴, z = e^{−sτ}:
// at φ = 0, |G| − 4/3 ≈ (−kr/ω² + k4/ω⁴)·(4/3)/|G∞| for c/(1 − d) = 4/3, so
// with kr, k4 < 0 the passages through φ = 0 cross lim at ω ≈ √(k4/kr)
// and the excess peaks near √(2·k4/kr).
func farChannel(t *testing.T, tau, c, d, kr, k4, p float64) (*System, func(float64) float64) {
	t.Helper()
	ds, _ := delayResonanceLoop(t, []float64{tau}, 0, []float64{c}, []float64{d})
	a := []float64{-p, 0, 0, 0, 1, -p, 0, 0, 0, 1, -p, 0, 0, 0, 1, -p}
	r, err := NewFromSlices(4, 1, 1, a, []float64{1, 0, 0, 0}, []float64{0, kr, 0, k4}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := Parallel(ds, r)
	if err != nil {
		t.Fatal(err)
	}
	return sys, func(w float64) float64 {
		s := complex(0, w)
		z := cmplx.Exp(-s * complex(tau, 0))
		q := s + complex(p, 0)
		q2 := q * q
		return cmplx.Abs(complex(c, 0)*z/(1-complex(d, 0)*z) + complex(kr, 0)/q2 + complex(k4, 0)/(q2*q2))
	}
}

// farPair is diag(G1, G2) of farChannel with kr = −10⁻³, k4 = −10: the
// finite peak, about 6·10⁻⁸ above lim at ω ≈ 102, lies past the span
// scanned when limitVerdict is undecided.
func farPair(d2 float64) func(t *testing.T) (*System, *System, func(float64) float64, func(float64) float64, float64) {
	return func(t *testing.T) (*System, *System, func(float64) float64, func(float64) float64, float64) {
		const tau = 0.37
		s1, g1 := farChannel(t, tau, 0.8, 0.4, -1e-3, -10, 3)
		s2, g2 := farChannel(t, tau, 4.0/3*(1-d2), d2, -1.3e-3, -7, 3)
		return s1, s2, g1, g2, tau
	}
}

// limitPair is diag(G, G) for the expansionLimit model, sup σ_max(G∞) at
// φ* = π with a′ ≠ 0 and κ = c + a′²/α < 0.
func limitPair(t *testing.T) (*System, *System, func(float64) float64, func(float64) float64, float64) {
	c := expansionLimit
	sys, gain, _ := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, 2 * c.h}, [2]float64{1, 2})
	return sys, sys, gain, gain, c.h
}

var clusterGains = []clusterGain{
	{"equal/rise", resonancePair(-0.3, 0.3, 0.8, 0.4), 0, true},
	{"equal/fall", resonancePair(0.3, 0.5, 0.8, 0.4), 0, false},
	{"cross/rise1", resonancePair(-0.3, 0.3, 0.4, 0.7), 0, true},
	{"cross/rise2", resonancePair(0.3, -0.3, 0.4, 0.7), 0, true},
	{"cross/fall", resonancePair(0.3, 0.2, 0.4, 0.7), 0, false},
	{"limit", limitPair, math.Pi, false},
}

// TestLimitVerdictMatrixCluster checks limitVerdict where σ_max(G∞(φ*)) is
// exactly repeated and a = 0 (6QNG6U, IBRQNM), undecided before the
// degenerate expansion. The oracle: the passage maxima of the closed-form
// σ_max through φ*, (max − lim)·ω² settling to κ/(2·lim), whose sign gives
// finite peaks above lim or the limit at +Inf; clusterKappa matches it.
func TestLimitVerdictMatrixCluster(t *testing.T) {
	for _, c := range clusterGains {
		for ri, rot := range rotations {
			t.Run(c.name+map[int]string{0: "/diag", 1: "/rot"}[ri], func(t *testing.T) {
				sys, sigma, h := c.system(t, rot[0], rot[1])
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
				if !x.at(c.phi) {
					t.Fatal("singular at φ*")
				}
				if _, _, _, gap, ok := newGramPair(2, 2).expand(x.g[0], x.g[1], x.g[2]); ok || gap > repeatedGap {
					t.Fatalf("σ_max(G∞(φ*)) not repeated: gap %g", gap)
				}
				var excess float64
				for i, wc := range []float64{1e3, 1e4} {
					v, w := passageMax(sigma, h, c.phi, wc)
					e := (v - lim) * w * w
					if i > 0 && math.Abs(e-excess) > 0.02*math.Abs(excess) {
						t.Fatalf("(max − lim)·ω² = %g at %g, %g a decade earlier", e, w, excess)
					}
					excess = e
				}
				if (excess > 0) != c.finite {
					t.Fatalf("oracle (max − lim)·ω² = %g, case wants finite = %v", excess, c.finite)
				}
				finite, W, ok := x.limitVerdict(lim, 1e-10, 10, 1e6)
				if !ok || finite != c.finite || (!finite && !(W > 0)) {
					t.Fatalf("limitVerdict = %v, %g, %v; want finite = %v", finite, W, ok, c.finite)
				}
				checkClusterKappa(t, x, c.phi, lim, excess)
			})
		}
	}
}

// TestHinfNormMatrixClusterLimit checks HinfNorm on 2×2 cluster gains:
// resonances with kr < 0 small enough that σ_max stays below lim = 4/3 up
// to the passages through φ* = 0, which rise above it, so the peak is
// finite; farPair, whose finite peak lies past the span scanned when the
// cluster leaves limitVerdict undecided, where HinfNorm returned lim at
// +Inf before the degenerate expansion; and diag(G, G) of expansionLimit,
// lim at +Inf. Oracle: dense ω scan of the closed-form σ_max with golden
// refinement, and the passage at 1e5 for lim.
func TestHinfNormMatrixClusterLimit(t *testing.T) {
	for _, c := range []clusterGain{
		{"equal/rise", resonancePair(-0.01, -0.02, 0.8, 0.4), 0, true},
		{"cross/rise", resonancePair(-0.02, -0.01, 0.4, 0.7), 0, true},
		{"far/equal", farPair(0.4), 0, true},
		{"far/cross", farPair(0.7), 0, true},
		{"limit", limitPair, math.Pi, false},
	} {
		for ri, rot := range rotations {
			t.Run(c.name+map[int]string{0: "/diag", 1: "/rot"}[ri], func(t *testing.T) {
				sys, sigma, h := c.system(t, rot[0], rot[1])
				lim, _ := passageMax(sigma, h, c.phi, 1e5)
				got, w, err := HinfNorm(sys)
				if err != nil {
					t.Fatalf("HinfNorm: %v", err)
				}
				want, wWant := oraclePeak(sigma, 200, 1e-3)
				if !c.finite {
					if !(want < lim) {
						t.Fatalf("oracle peak %.15g not below lim %.15g", want, lim)
					}
					if !math.IsInf(w, 1) || math.Abs(got-lim) > 1e-6*lim {
						t.Fatalf("HinfNorm = (%.15g, %g), want (%.15g, +Inf)", got, w, lim)
					}
					return
				}
				if !(want > lim) || wWant < 1 {
					t.Fatalf("oracle peak %.15g at %g, want a passage above lim %.15g", want, wWant, lim)
				}
				if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-3 {
					t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
				}
			})
		}
	}
}

// TestClusterKappaSimple checks that clusterKappa, with a one-dimensional
// cluster, gives the κ = c + a′²/α of a simple σ_max: the 2×2 gains of
// TestLimitVerdictMatrixKappa, against the passage maxima through φ*,
// (max − lim)·ω² settling to κ/(2·lim).
func TestClusterKappaSimple(t *testing.T) {
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
				lim := hf.onePhaseSup()
				phi, far := 0.0, 0.0
				for _, p := range []float64{0, math.Pi} {
					if v, _ := passageMax(sigma, c.h, p, 1e5); v > far {
						phi, far = p, v
					}
				}
				v, w := passageMax(sigma, c.h, phi, 1e4)
				excess := (v - lim) * w * w
				checkClusterKappa(t, x, phi, lim, excess)
			})
		}
	}
}

// checkClusterKappa checks that the bounds of clusterKappa at φ are tight
// and hold κ/(2·lim) = excess, the passage oracle, to 2%.
func checkClusterKappa(t *testing.T, x *highFrequencyExpansion, phi, lim, excess float64) {
	t.Helper()
	lo, hi, _, _, ok := x.clusterKappa(phi, 1e-9)
	tol := 0.02 * math.Abs(excess) * 2 * lim
	if !ok || !(hi-lo <= 1e-3*math.Abs(lo+hi)) || !(lo-tol <= 2*lim*excess && 2*lim*excess <= hi+tol) {
		t.Fatalf("clusterKappa in [%g, %g] (%v), want κ = %g", lo, hi, ok, 2*lim*excess)
	}
}

// TestPencilSupNarrowBranch checks pencilSup on the decoupled pencil
// diag(−1 + 2s·b − 10⁻³·s², −1 + 2s·a′ − 10⁴·s²), a′ = √(1.5·10⁴): the
// flat branch sets |s| <= sB ~ 10², the sharp one peaks at
// −1 + a′²/10⁴ = 0.5 at s = a′/10⁴ ≈ 0.012, narrower than a uniform
// sample spacing over [−sB, sB]. The exact sup is max of the two
// branch maxima, −1 + b²/10⁻³ and 0.5.
func TestPencilSupNarrowBranch(t *testing.T) {
	ap := math.Sqrt(1.5e4)
	for _, b := range []float64{0, 0.01, 0.04} {
		q := [3][]complex128{{-1, 0, 0, -1}, {complex(2*b, 0), 0, 0, complex(2*ap, 0)}, {-1e-3, 0, 0, -1e4}}
		want := max(-1+b*b/1e-3, 0.5)
		lo, hi, ok := pencilSup(q, 2)
		if !ok || !(lo <= want+1e-12) || !(want <= hi+1e-12) || hi-lo > 1e-6 {
			t.Errorf("b = %g: pencilSup in [%.12g, %.12g] (%v), want %.12g", b, lo, hi, ok, want)
		}
		// Rotate by a unitary: the bounds must not change.
		c, s := complex(math.Cos(0.7), 0), complex(0, math.Sin(0.7))
		u := []complex128{c, -s, -s, c}
		var qr [3][]complex128
		for k := range q {
			qr[k] = usw(u, q[k], u, 2, 2)
		}
		// Rounding in the rotation moves ν = 1e-3 by ~eps·‖Q2‖ ≈ 2e-12, so the
		// rotated pencil's sup b²/ν − 1 differs from want by ~(want+1)·2e-9.
		tol := 64 * 0x1p-52 * 1e4 / 1e-3 * (want + 1)
		lo2, hi2, ok := pencilSup(qr, 2)
		if !ok || !(lo2 <= want+tol) || !(want <= hi2+tol) || hi2-lo2 > 1e-6 {
			t.Errorf("b = %g rotated: pencilSup in [%.12g, %.12g] (%v), want %.12g", b, lo2, hi2, ok, want)
		}
	}
}

// TestLimitVerdictMatrixClusterSharp checks diag(G, G) and U·diag(G, G)
// of the sharp resonance model of TestHinfNormSharpResonanceLimit, where
// c and a′²/α nearly cancel in κ < 0 and the peak has width about 1 − D22
// in φ, so φ-derivatives need small steps: limitVerdict is +Inf or
// undecided, never finite, and HinfNorm is sup σ_max(G∞) at +Inf, as for
// the 1×2 model.
func TestLimitVerdictMatrixClusterSharp(t *testing.T) {
	for _, d22 := range []float64{0.999, 0.99999} {
		for ri, rot := range rotations {
			t.Run(fmt.Sprint(d22)+map[int]string{0: "/diag", 1: "/rot"}[ri], func(t *testing.T) {
				g, _, gInf := sharpResonanceModel(t, d22)
				sys, err := Append(g, g)
				if err != nil {
					t.Fatal(err)
				}
				u, err := NewGain(mat.NewDense(2, 2, rot[0]), 0)
				if err != nil {
					t.Fatal(err)
				}
				if sys, err = Series(sys, u); err != nil {
					t.Fatal(err)
				}
				e, err := newDelayLFT(sys)
				if err != nil {
					t.Fatal(err)
				}
				hf := newHighFrequencyGain(e)
				x := newHighFrequencyExpansion(e, hf)
				lim := hf.onePhaseSup()
				if math.Abs(lim-gInf(0)) > 1e-12*lim {
					t.Fatalf("sup σ_max(G∞) = %.15g, |G∞(0)| = %.15g", lim, gInf(0))
				}
				finite, W, ok := x.limitVerdict(lim, 1e-10, 10, 1e6)
				t.Logf("limitVerdict = %v, %g, %v", finite, W, ok)
				if ok && (finite || !(W > 0)) {
					t.Fatalf("limitVerdict = %v, %g, %v; want +Inf or undecided", finite, W, ok)
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
