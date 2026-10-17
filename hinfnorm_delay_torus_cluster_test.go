package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// sisoTorusKappa is κ = c + ∇aᵀ·α⁻¹·∇a of a vector gain at θ, from the
// closed-form a and c of coefficientsAt and central differences of a and
// S² at step 1e-4, independent of the cluster pencil.
func sisoTorusKappa(t *testing.T, x *highFrequencyExpansion, theta []float64) float64 {
	t.Helper()
	s2, _, c, _, ok := x.coefficientsAt(theta)
	if !ok {
		t.Fatal("vector coefficients not exact")
	}
	nd := len(theta)
	th := make([]float64, nd)
	at := func(i, j int, di, dj float64) (float64, float64) {
		copy(th, theta)
		th[i] += di
		th[j] += dj
		s, a, _, _, _ := x.coefficientsAt(th)
		return s, a
	}
	const h = 1e-4
	grad := mat.NewVecDense(nd, nil)
	alpha := mat.NewSymDense(nd, nil)
	for i := range nd {
		sp, ap := at(i, i, h, 0)
		sm, am := at(i, i, -h, 0)
		grad.SetVec(i, (ap-am)/(2*h))
		alpha.SetSym(i, i, -(sp-2*s2+sm)/(2*h*h))
		for j := range i {
			pp, _ := at(i, j, h, h)
			pm, _ := at(i, j, h, -h)
			mp, _ := at(i, j, -h, h)
			mm, _ := at(i, j, -h, -h)
			alpha.SetSym(i, j, -(pp-pm-mp+mm)/(8*h*h))
		}
	}
	var ch mat.Cholesky
	y := mat.NewVecDense(nd, nil)
	if !ch.Factorize(alpha) || ch.SolveVecTo(y, grad) != nil {
		t.Fatal("α not positive definite")
	}
	return c + mat.Dot(grad, y)
}

// torusClusterModel returns diag(G1, G2) of the torus models of
// TestHinfNormTorusFixedPointLimit, τ = (h, √2·h), G1 = G and G2 = G +
// kr/(s + p)², rotated by U on the left; kr = 0 is U·diag(G, G). The
// strictly proper term leaves G∞ unchanged, so σ_max(G∞) is double on the
// whole torus, and at a fixed point θ*, where G∞ is real, it shifts c of G2
// by −2·kr·G∞(θ*). sigma is max(|G1|, |G2|) in closed form; g1 is G.
func torusClusterModel(t *testing.T, c expansionCase, kr, p float64, u []float64) (sys, g1 *System, sigma func(float64) float64) {
	t.Helper()
	g, resp, _ := feedthroughDifferenceResponse(t, c.d22, c.d12, c.d21, []float64{c.h, math.Sqrt2 * c.h}, [2]float64{1, 1})
	g2 := g
	if kr != 0 {
		r, err := NewFromSlices(2, 1, 1, []float64{-p, 0, 1, -p}, []float64{1, 0}, []float64{0, kr}, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		if g2, err = Parallel(g, r); err != nil {
			t.Fatal(err)
		}
	}
	sys, err := Append(g, g2)
	if err != nil {
		t.Fatal(err)
	}
	ug, err := NewGain(mat.NewDense(2, 2, u), 0)
	if err != nil {
		t.Fatal(err)
	}
	if sys, err = Series(sys, ug); err != nil {
		t.Fatal(err)
	}
	return sys, g, func(w float64) float64 {
		z := resp(w)
		q := complex(p, w)
		return math.Max(cmplx.Abs(z), cmplx.Abs(z+complex(kr, 0)/(q*q)))
	}
}

// torusTop returns the expansion of sys and its maximisers of σ_max(G∞)
// within 1e-9 of the largest, snapped to the fixed points.
func torusTop(t *testing.T, sys *System) (*highFrequencyExpansion, []torusMaximum) {
	t.Helper()
	e, err := newDelayLFT(sys)
	if err != nil {
		t.Fatal(err)
	}
	hf := newHighFrequencyGain(e)
	if hf.dim != 2 {
		t.Fatalf("dim = %d, want 2", hf.dim)
	}
	ms := torusMaxima(hf)
	snapFixedPoints(hf, ms)
	n := 0
	for n < len(ms) && ms[n].value >= ms[0].value*(1-1e-9) {
		n++
	}
	return newTorusExpansion(e, hf), ms[:n]
}

// TestHinfNormTorusClusterFixedPoint pins M2FKRH: U·diag(G, G) for the
// torus models of TestHinfNormTorusFixedPointLimit, whose σ_max(G∞) is
// double at its maximiser θ* ∈ {0, π}², used to stay undecided (fallsPast
// false, errTorusUnsupported). The cluster κ of clusterKappaAt brackets
// the κ of the SISO G from its closed-form coefficients (the oracle for
// the degenerate expansion), it is negative, and HinfNorm reports the torus
// sup of |G∞| from an independent grid search at ω = +Inf, above every
// finite peak of a dense ω scan.
func TestHinfNormTorusClusterFixedPoint(t *testing.T) {
	for _, c := range expansionFinite {
		for ri, rot := range rotations {
			t.Run(c.name+map[int]string{0: "/diag", 1: "/rot"}[ri], func(t *testing.T) {
				sys, g, sigma := torusClusterModel(t, c, 0, 1, rot[0])
				x, ms := torusTop(t, sys)
				xs, msS := torusTop(t, g)
				for _, m := range ms {
					if !m.fixed {
						t.Fatalf("maximiser %v off the fixed points", m.theta)
					}
					if x.risesPast(m.theta) || !x.fallsPast(m.theta) {
						t.Fatalf("at %v want falls (κ < 0)", m.theta)
					}
					lo, hi, mu1, _, ok := x.clusterKappaAt(m.theta, 1e-9)
					want := sisoTorusKappa(t, xs, msS[0].theta)
					if !ok || math.Abs(mu1) > 1e-9 || !(want < 0) || !(lo-1e-5*math.Abs(want) <= want && want <= hi+1e-5*math.Abs(want)) || hi-lo > 1e-6*math.Abs(want) {
						t.Fatalf("clusterKappaAt = [%.10g, %.10g], μ1 %g (%v); SISO κ %.10g", lo, hi, mu1, ok, want)
					}
				}
				sup := torusSupOf(func(p, q float64) float64 { return gInfTorus(c.d22, c.d12, c.d21, p, q) })
				if fin, w := oraclePeak(sigma, 10000, 2e-3); !(fin < sup*(1-1e-8)) {
					t.Fatalf("finite peak %.15g at %g not below torus sup %.15g", fin, w, sup)
				}
				got, w, err := HinfNorm(sys)
				if err != nil || !math.IsInf(w, 1) || math.Abs(got-sup) > 1e-9*sup {
					t.Fatalf("HinfNorm = %.15g @%g, %v; want %.15g @+Inf", got, w, err, sup)
				}
			})
		}
	}
}

// TestTorusClusterRisesPast checks risesPast on U·diag(G, G + kr/(s + p)²)
// with kr chosen so that κ of G2 at θ* turns positive: the cluster rises,
// as the SISO G2 does by its closed-form coefficients, where it used to
// stay undecided. The oracle: a dense ω scan of the closed-form σ_max finds
// a finite peak above the torus sup of |G∞| past the grid range.
func TestTorusClusterRisesPast(t *testing.T) {
	for _, tc := range []struct {
		c     expansionCase
		kr, p float64
		wMin  float64
		wMax  float64
	}{
		{expansionFinite[0], -365, 100, 100, 400},
		{expansionFinite[1], 347, 30, 200, 500},
	} {
		for ri, rot := range rotations {
			t.Run(tc.c.name+map[int]string{0: "/diag", 1: "/rot"}[ri], func(t *testing.T) {
				sys, _, sigma := torusClusterModel(t, tc.c, tc.kr, tc.p, rot[0])
				x, ms := torusTop(t, sys)
				g2 := torusClusterModelSISO(t, tc.c, tc.kr, tc.p)
				xs, msS := torusTop(t, g2)
				if k := sisoTorusKappa(t, xs, msS[0].theta); !(k > 0) || !xs.risesPast(msS[0].theta) {
					t.Fatalf("SISO G2: κ = %g, want a rise", k)
				}
				for _, m := range ms {
					if !m.fixed || !x.risesPast(m.theta) || x.fallsPast(m.theta) {
						t.Fatalf("at %v (fixed %v) want rises (κ > 0)", m.theta, m.fixed)
					}
				}
				sup := torusSupOf(func(p, q float64) float64 { return gInfTorus(tc.c.d22, tc.c.d12, tc.c.d21, p, q) })
				if lo, _ := oraclePeak(sigma, 60, 1e-3); !(lo < sup) {
					t.Fatalf("grid-range peak %.15g not below torus sup %.15g", lo, sup)
				}
				fin, w := oraclePeak(sigma, tc.wMax, 1e-3)
				if !(fin > sup*(1+1e-6)) || !(w > tc.wMin) {
					t.Fatalf("finite peak %.15g at %g, want above torus sup %.15g past %g", fin, w, sup, tc.wMin)
				}
			})
		}
	}
}

// torusClusterModelSISO is G2 = G + kr/(s + p)² of torusClusterModel.
func torusClusterModelSISO(t *testing.T, c expansionCase, kr, p float64) *System {
	t.Helper()
	g, _, _ := feedthroughDifferenceResponse(t, c.d22, c.d12, c.d21, []float64{c.h, math.Sqrt2 * c.h}, [2]float64{1, 1})
	r, err := NewFromSlices(2, 1, 1, []float64{-p, 0, 1, -p}, []float64{1, 0}, []float64{0, kr}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	g2, err := Parallel(g, r)
	if err != nil {
		t.Fatal(err)
	}
	return g2
}

// TestPencilSupN checks pencilSupN over s ∈ R²: decoupled branches
// −1 + 2·b_kᵀ·s − sᵀ·A_k·s, coupled A_k, sup max_k (−1 + b_kᵀ·A_k⁻¹·b_k)
// in closed form, one sharp and narrow; the same under a unitary rotation;
// r = 1; and a pencil not concave in s, refused.
func TestPencilSupN(t *testing.T) {
	type branch struct {
		b [2]float64
		a [3]float64 // A11, A12, A22
	}
	sup := func(br branch) float64 {
		det := br.a[0]*br.a[2] - br.a[1]*br.a[1]
		y0 := (br.a[2]*br.b[0] - br.a[1]*br.b[1]) / det
		y1 := (br.a[0]*br.b[1] - br.a[1]*br.b[0]) / det
		return -1 + br.b[0]*y0 + br.b[1]*y1
	}
	pencil := func(brs []branch) clusterQuad {
		r := len(brs)
		kq := clusterQuad{r: r, q0: make([]complex128, r*r), q1: [][]complex128{make([]complex128, r*r), make([]complex128, r*r)}}
		kq.q2 = [][][]complex128{{make([]complex128, r*r), make([]complex128, r*r)}, {nil, make([]complex128, r*r)}}
		kq.q2[1][0] = kq.q2[0][1]
		for k, br := range brs {
			d := k*r + k
			kq.q0[d] = -1
			kq.q1[0][d], kq.q1[1][d] = complex(2*br.b[0], 0), complex(2*br.b[1], 0)
			kq.q2[0][0][d], kq.q2[0][1][d], kq.q2[1][1][d] = complex(-br.a[0], 0), complex(-br.a[1], 0), complex(-br.a[2], 0)
		}
		return kq
	}
	rotate := func(kq clusterQuad, u []complex128) clusterQuad {
		r := kq.r
		out := clusterQuad{r: r, q0: usw(u, kq.q0, u, r, r), q1: make([][]complex128, 2), q2: [][][]complex128{make([][]complex128, 2), make([][]complex128, 2)}}
		for i := range 2 {
			out.q1[i] = usw(u, kq.q1[i], u, r, r)
			for j := range 2 {
				out.q2[i][j] = usw(u, kq.q2[i][j], u, r, r)
			}
		}
		return out
	}
	rng := rand.New(rand.NewPCG(5, 7))
	for _, brs := range [][]branch{
		{{[2]float64{0.3, -0.2}, [3]float64{1, 0.4, 2}}, {[2]float64{0.5, 0.1}, [3]float64{0.5, -0.2, 3}}},
		{{[2]float64{0.01, 0.02}, [3]float64{1e-3, 0, 2e-3}}, {[2]float64{122, -40}, [3]float64{1e4, 2e3, 1e4}}},
		{{[2]float64{0.7, 0.7}, [3]float64{1, 0.9, 1}}},
	} {
		want := math.Inf(-1)
		for _, br := range brs {
			want = max(want, sup(br))
		}
		kq := pencil(brs)
		lo, hi, ok := pencilSupN(kq)
		tol := 1e-8 * (1 + math.Abs(want))
		if !ok || !(lo <= want+tol) || !(want <= hi+tol) || hi-lo > 1e-6*(1+math.Abs(want)) {
			t.Errorf("r=%d: pencilSupN in [%.12g, %.12g] (%v), want %.12g", kq.r, lo, hi, ok, want)
		}
		if kq.r == 1 {
			continue
		}
		u := randUnitary(rng, kq.r)
		lo, hi, ok = pencilSupN(rotate(kq, u))
		// Rounding in the rotation moves Q2 by ~eps·‖Q2‖, the sup by that over ν.
		tol = 1e-6 * (1 + math.Abs(want))
		if !ok || !(lo <= want+tol) || !(want <= hi+tol) || hi-lo > 1e-6*(1+math.Abs(want)) {
			t.Errorf("r=%d rotated: pencilSupN in [%.12g, %.12g] (%v), want %.12g", kq.r, lo, hi, ok, want)
		}
	}
	saddle := pencil([]branch{{[2]float64{0.1, 0.1}, [3]float64{1, 0, -1}}, {[2]float64{0.1, 0.1}, [3]float64{1, 0, 1}}})
	if _, _, ok := pencilSupN(saddle); ok {
		t.Error("pencil not concave in s accepted")
	}
}

// TestGramPairClusterPencilN checks clusterPencilN over s ∈ R² for g0 =
// U·S·W^H with σ_max(g0) exactly double, against one-sided differences in
// t > 0 of the top eigenvalue of G_s(t)^H·G_s(t) from an independent
// eigensolver, G_s = g0 + t·(g1 + Σ s_i·d0_i) + t²·(g2 + Σ s_i·d1_i +
// ½·Σ_ij s_i·s_j·dd0_ij): λ0 + t·μ1 + t²·λ_max(K(s)). Each d0_i has an
// anti-Hermitian top 2×2 block in U's frame, as at a maximiser.
func TestGramPairClusterPencilN(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 19))
	cplx := func() complex128 { return complex(rng.NormFloat64(), rng.NormFloat64()) }
	for _, sz := range [][2]int{{3, 3}, {4, 3}, {3, 4}} {
		for _, flat := range []bool{false, true} {
			p, m := sz[0], sz[1]
			u, w := randUnitary(rng, p), randUnitary(rng, m)
			sv := make([]complex128, p*m)
			sv[0], sv[m+1], sv[2*m+2] = 1.5, 1.5, 0.7
			rnd := func(zeroTop, antiTop bool) []complex128 {
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
			g1, g2 := rnd(flat, false), rnd(false, false)
			d0 := [][]complex128{rnd(false, true), rnd(false, true)}
			d1 := [][]complex128{rnd(false, false), rnd(false, false)}
			dd0 := [][][]complex128{{rnd(false, false)}, {rnd(false, false), rnd(false, false)}}
			kq, mu1, _, ok := newGramPair(p, m).clusterPencilN(g0, g1, g2, d0, d1, dd0, 1e-9)
			if !ok {
				t.Fatalf("%dx%d flat=%v: clusterPencilN failed", p, m, flat)
			}
			r := kq.r
			for _, s := range [][2]float64{{0, 0}, {-0.8, 0.3}, {0.5, -1.1}, {1.7, 0.9}} {
				at := func(tt float64) float64 {
					g := make([]complex128, p*m)
					for i := range g {
						g1s := g1[i] + complex(s[0], 0)*d0[0][i] + complex(s[1], 0)*d0[1][i]
						g2s := g2[i] + complex(s[0], 0)*d1[0][i] + complex(s[1], 0)*d1[1][i] +
							complex(s[0]*s[0]/2, 0)*dd0[0][0][i] + complex(s[1]*s[1]/2, 0)*dd0[1][1][i] + complex(s[0]*s[1], 0)*dd0[1][0][i]
						g[i] = g0[i] + complex(tt, 0)*g1s + complex(tt*tt, 0)*g2s
					}
					return topGramEig(t, g, p, m)
				}
				l0 := at(0)
				const h = 2e-4
				f := func(tt float64) float64 { return (at(tt) - l0 - tt*mu1) / (tt * tt) }
				want := (8*f(h) - 6*f(2*h) + f(4*h)) / 3
				k := make([]complex128, r*r)
				for i := range k {
					k[i] = kq.q0[i]
					for a := range 2 {
						k[i] += complex(s[a], 0) * kq.q1[a][i]
						for b := range 2 {
							k[i] += complex(s[a]*s[b], 0) * kq.q2[a][b][i]
						}
					}
				}
				vals, _, _, _ := hermTop(k, r, 0, 0)
				if math.Abs(vals[0]-want) > 1e-5*(1+math.Abs(want)) {
					t.Errorf("%dx%d flat=%v s=%v: λ_max(K) = %.10g, one-sided difference %.10g", p, m, flat, s, vals[0], want)
				}
			}
		}
	}
}
