package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"strings"
	"testing"
)

// torusPeakCases are 2×2 gains with delays τ = (h, √2·h), two classes on a
// 2-torus, whose sup σ_max(G∞) over the torus lies above the Nyquist grid
// peak and is attained off the fixed points {0, π}², where a ≠ 0, so a
// passage of θ = ω·(h, √2·h) near a maximiser rises above it at a finite ω
// past the grid end.
var torusPeakCases = []lftCase{
	{name: "200", n: 2, k: math.Sqrt2, h: 0.18091879264790087,
		a:   []float64{-2.547988322076141, -0.4898449095946593, 0.5763421279314911, -1.7166277951918834},
		b1:  []float64{-0.46427994388238636, -0.6866885295215206, 0.1419841860721045, -0.9697332935996243},
		b2:  []float64{0.10658399043778544, -0.18905916747339324, 0.5199359770456558, -0.2798201294234439},
		c1:  []float64{-0.5500911998027607, 0.6945654476990228, 0.2475940233955265, 0.8437091858784171},
		c2:  []float64{-0.02510755679641208, 0.3135257518291674, 0.2101947443043065, -0.1819385423066361},
		d11: []float64{0.9293629138802137, 0.3831667784108652, -0.7971328967258045, 0.2626104000954108},
		d12: []float64{1.7221623970351856, 0.67224133766363, 0.9123323222143744, 1.216777469420597},
		d21: []float64{-1.3251581467135365, -0.9428011224315824, -1.2265786980721058, -1.5338497282757704},
		d22: []float64{-0.5821509210377749, 0.23611845014763766, -0.46975206550423176, -0.31198187847585396}},
	{name: "252", n: 2, k: math.Sqrt2, h: 0.4216195058674098,
		a:   []float64{-2.058442307229665, -0.7094143417081069, -0.281364420103257, -1.7607298847563198},
		b1:  []float64{0.3908172941138195, 0.11304550907635114, -0.9107620404625512, -0.7023309640014817},
		b2:  []float64{0.8644953330360685, -0.20666019186395945, -0.24152078889190332, -0.8219082637440094},
		c1:  []float64{-0.7595947547614257, 0.8025931733748843, 0.5331245367584572, -0.09699461395883491},
		c2:  []float64{-0.06389493517249678, 0.1739124871389044, -0.08253079480525394, 0.10536537834796211},
		d11: []float64{-0.6541365320179204, -0.8957279947457346, -0.5798404957516128, -0.7974704091670843},
		d12: []float64{1.6487612377319962, 1.3961980208600773, 0.5530341405905443, -0.6065163877507498},
		d21: []float64{-1.2657586161382999, 1.7240916183905566, 1.9769351132802382, -1.610348382969137},
		d22: []float64{-0.17312573115567081, 0.21125826455027705, 0.22386599303355206, -0.2228358788762669}},
}

// torusSup returns sup σ_max(G∞) over the 2-torus of c by a 512² grid with
// each of the largest cells refined by coordinate golden-section search.
func torusSup(c lftCase) float64 {
	g := func(p, q float64) float64 { return lftGInf2(c, p, q) }
	const n = 512
	type cell struct{ p, q, v float64 }
	var top []cell
	for i := range n {
		for j := range n {
			p, q := 2*math.Pi*float64(i)/n, 2*math.Pi*float64(j)/n
			v := g(p, q)
			top = append(top, cell{p, q, v})
		}
	}
	best := 0.0
	for _, cl := range top {
		best = max(best, cl.v)
	}
	out := best
	const gr = 0.6180339887498949
	for _, cl := range top {
		if cl.v < best*(1-1e-3) {
			continue
		}
		p, q := cl.p, cl.q
		for range 40 {
			lo, hi := p-0.02, p+0.02
			for hi-lo > 1e-12 {
				x1, x2 := hi-gr*(hi-lo), lo+gr*(hi-lo)
				if g(x1, q) > g(x2, q) {
					hi = x2
				} else {
					lo = x1
				}
			}
			p = (lo + hi) / 2
			lo, hi = q-0.02, q+0.02
			for hi-lo > 1e-12 {
				x1, x2 := hi-gr*(hi-lo), lo+gr*(hi-lo)
				if g(p, x1) > g(p, x2) {
					hi = x2
				} else {
					lo = x1
				}
			}
			q = (lo + hi) / 2
		}
		out = max(out, g(p, q))
	}
	return out
}

// lftGInf2 is lftCase.gInf with independent phases p, q of the two delays.
func lftGInf2(c lftCase, p, q float64) float64 {
	dl := [2]complex128{complex(math.Cos(p), -math.Sin(p)), complex(math.Cos(q), -math.Sin(q))}
	m := [4]complex128{1 - complex(c.d22[0], 0)*dl[0], -complex(c.d22[1], 0) * dl[1], -complex(c.d22[2], 0) * dl[0], 1 - complex(c.d22[3], 0)*dl[1]}
	det := m[0]*m[3] - m[1]*m[2]
	var g [4]complex128
	for j := range 2 {
		d0, d1 := complex(c.d21[j], 0), complex(c.d21[2+j], 0)
		z0 := (m[3]*d0 - m[1]*d1) / det
		z1 := (m[0]*d1 - m[2]*d0) / det
		for i := range 2 {
			g[i*2+j] = complex(c.d11[i*2+j], 0) + complex(c.d12[i*2], 0)*dl[0]*z0 + complex(c.d12[i*2+1], 0)*dl[1]*z1
		}
	}
	return sigma2x2(g)
}

// TestHinfNormTorusFinitePeakPastLimit checks torusPeakCases against a dense
// scan of the closed-form response: the finite peak, past the grid end, lies
// above the torus sup from an independent grid search, where the limit used
// to be left unsupported.
func TestHinfNormTorusFinitePeakPastLimit(t *testing.T) {
	for _, c := range torusPeakCases {
		t.Run(c.name, func(t *testing.T) {
			sys := c.system(t)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			if hf := newHighFrequencyGain(e); hf.dim != 2 {
				t.Fatalf("dim = %d, want 2", hf.dim)
			}
			lim := torusSup(c)
			want, wWant := oraclePeak(c.sigma, 1500, 1e-3)
			if !(want > lim*(1+1e-6)) || !(wWant > 100) {
				t.Fatalf("oracle peak %.12g at %g, want above torus sup %.12g past the grid", want, wWant, lim)
			}
			if lo, _ := oraclePeak(c.sigma, 60, 1e-3); !(lo < lim) {
				t.Fatalf("grid-range peak %.12g not below torus sup %.12g", lo, lim)
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

// TestHinfNormTorusFixedPointLimitUnsupported pins that a torus sup at a
// fixed point θ* ∈ {0, π}² with κ < 0, approached from below, is still not
// reported at ω = +Inf: the passages near θ* depend on how well √2 is
// approximated by the ratios of floating-point delays.
func TestHinfNormTorusFixedPointLimitUnsupported(t *testing.T) {
	for _, c := range expansionFinite {
		sys, _, _ := feedthroughDifferenceLoop(t, c.d22, c.d12, c.d21, []float64{c.h, math.Sqrt2 * c.h}, [2]float64{1, 1})
		e, err := newDelayLFT(sys)
		if err != nil {
			t.Fatal(err)
		}
		hf := newHighFrequencyGain(e)
		ms := torusMaxima(hf)
		x := newTorusExpansion(e, hf)
		for _, m := range ms {
			for _, th := range m.theta {
				if r := math.Remainder(th, math.Pi); math.Abs(r) > 1e-6 {
					t.Fatalf("%s: maximiser %v off the fixed points", c.name, m.theta)
				}
			}
			if x.risesPast(m.theta) {
				t.Fatalf("%s: risesPast at %v, want κ < 0", c.name, m.theta)
			}
		}
		if _, _, err := HinfNorm(sys); !errors.Is(err, errDelayPeakUnsupported) {
			t.Fatalf("%s: HinfNorm err = %v, want errDelayPeakUnsupported", c.name, err)
		}
	}
}

// TestQuadBoundMatrix checks the vertex bound of quadBound for matrix G∞
// against σ_max(G∞) sampled in the cell, on the 2-torus of torusPeakCases
// and the circle of slightMatrixPeakCases, and that near a maximiser it is
// second order in r where localBound is first order.
func TestQuadBoundMatrix(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	for _, c := range []lftCase{torusPeakCases[0], torusPeakCases[1], slightMatrixPeakCases[0]} {
		e, err := newDelayLFT(c.system(t))
		if err != nil {
			t.Fatal(err)
		}
		hf := newHighFrequencyGain(e)
		nd := hf.dim
		th := make([]float64, nd)
		probe := make([]float64, nd)
		for trial := range 40 {
			for i := range th {
				th[i] = 2 * math.Pi * rng.Float64()
			}
			if trial == 0 {
				copy(th, torusMaxima(hf)[0].theta)
			}
			for _, r := range []float64{0.3, 0.05, 0.005} {
				v := hf.at(th)
				hf.local()
				b, lb := hf.quadBound(v, r), hf.localBound(r)
				if trial == 0 && r == 0.005 && !(b-v < 0.1*lb) {
					t.Fatalf("%s: at the maximiser quadBound − v = %g, localBound %g", c.name, b-v, lb)
				}
				for range 300 {
					for i := range probe {
						probe[i] = th[i] + r/hf.scale[i]*(2*rng.Float64()-1)
					}
					if g := hf.at(probe); !(g <= b*(1+1e-12)) {
						t.Fatalf("%s: σ_max(G∞(%v)) = %.15g above quadBound %.15g within r = %g of %v", c.name, probe, g, b, r, th)
					}
				}
			}
		}
	}
}

// TestHinfNormRoundedMultipleDelays pins OUWA5P: τ = (h, fl(3h)) with
// fl(3h) ≠ 3h is two classes, whose torus sup lies above every finite peak
// (dense 2-D oracle) and is approached only once ω·|fl(3h) − 3h| ~ 1, so
// HinfNorm reports the torus case as unsupported; the same model with a
// dyadic h, exactly commensurate, matches the finite and limit oracles.
func TestHinfNormRoundedMultipleDelays(t *testing.T) {
	rng := rand.New(rand.NewPCG(1008, 7))
	u := func(a float64) float64 { return a * (2*rng.Float64() - 1) }
	d22 := []float64{u(.8), u(.8), u(.8), u(.8)}
	d12 := []float64{u(1), u(1)}
	d21 := []float64{u(1), u(1)}
	h := 0.02 + 0.3*rng.Float64()
	k := [2]float64{1, 3}
	if exactProductsEqual(3*h, 1, h, 3) {
		t.Fatal("fl(3h) must differ from 3h")
	}
	sys, gain, _ := feedthroughDifferenceLoop(t, d22, d12, d21, []float64{h, 3 * h}, k)
	if st, err := sys.IsStable(); err != nil || !st {
		t.Fatalf("IsStable = %v, %v", st, err)
	}
	fin, _ := oraclePeak(gain, 80, 1e-3)
	torus := 0.0
	const n = 512
	for a := range n {
		for b := range n {
			torus = max(torus, gInfTorus(d22, d12, d21, 2*math.Pi*float64(a)/n, 2*math.Pi*float64(b)/n))
		}
	}
	if !(torus > 1.5*fin) {
		t.Fatalf("torus sup %g not above finite peak %g", torus, fin)
	}
	_, _, err := HinfNorm(sys)
	if !errors.Is(err, errDelayPeakUnsupported) || !strings.Contains(err.Error(), "not exactly commensurate") {
		t.Fatalf("HinfNorm err = %v, want torus errDelayPeakUnsupported", err)
	}

	hd := math.Ldexp(math.Round(math.Ldexp(h, 30)), -30)
	sys, gain, gInf := feedthroughDifferenceLoop(t, d22, d12, d21, []float64{hd, 3 * hd}, k)
	got, w, err := HinfNorm(sys)
	if err != nil {
		t.Fatalf("dyadic HinfNorm: %v", err)
	}
	want, _ := oraclePeak(gain, 80, 1e-3)
	for j := range 1 << 16 {
		want = max(want, gInf(2*math.Pi*float64(j)/(1<<16)))
	}
	if math.Abs(got-want) > 1e-8*want {
		t.Fatalf("dyadic HinfNorm = %.12g @%g, want %.12g", got, w, want)
	}
}

// gInfTorus is |G∞| of feedthroughDifferenceLoop with independent phases.
func gInfTorus(d22, d12, d21 []float64, t1, t2 float64) float64 {
	c := func(x float64) complex128 { return complex(x, 0) }
	e1, e2 := cmplx.Exp(complex(0, -t1)), cmplx.Exp(complex(0, -t2))
	m := [4]complex128{1 - c(d22[0])*e1, -c(d22[1]) * e2, -c(d22[2]) * e1, 1 - c(d22[3])*e2}
	det := m[0]*m[3] - m[1]*m[2]
	v0 := (m[3]*c(d21[0]) - m[1]*c(d21[1])) / det
	v1 := (m[0]*c(d21[1]) - m[2]*c(d21[0])) / det
	return cmplx.Abs(0.1 + c(d12[0])*e1*v0 + c(d12[1])*e2*v1)
}

// TestHinfNormTorusGiveUpCost pins that the search of torusPastLimit for a
// finite peak above the limit, on the OUWA5P models τ = (k0·h, k1·h) with
// fl(k·h) ≠ k·h that stay unsupported, samples G only near passages of the
// phases past a maximiser and gives up within a bounded number of samples
// instead of scanning the whole grid budget.
func TestHinfNormTorusGiveUpCost(t *testing.T) {
	const maxCalls = 1 << 16
	seen := 0
	for _, k := range [][2]float64{{1, 3}, {2, 3}} {
		for i := range 30 {
			rng := rand.New(rand.NewPCG(uint64(i+1000), 7))
			u := func(a float64) float64 { return a * (2*rng.Float64() - 1) }
			d22 := []float64{u(.8), u(.8), u(.8), u(.8)}
			d12 := []float64{u(1), u(1)}
			d21 := []float64{u(1), u(1)}
			h := 0.02 + 0.3*rng.Float64()
			sys, _, _ := feedthroughDifferenceLoop(t, d22, d12, d21, []float64{k[0] * h, k[1] * h}, k)
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			l, res, err := e.stability()
			if err != nil || !res.stable {
				continue
			}
			_, _, err = e.peakOrZero(l, res)
			if !errors.Is(err, errDelayPeakUnsupported) || !strings.Contains(err.Error(), "not exactly commensurate") {
				continue
			}
			seen++
			if e.sigmaCalls > maxCalls {
				t.Errorf("k=%v seed %d: %d samples of G before giving up, want <= %d", k, i, e.sigmaCalls, maxCalls)
			}
		}
	}
	if seen < 4 {
		t.Fatalf("only %d torus-unsupported models", seen)
	}
}
