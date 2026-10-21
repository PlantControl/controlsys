package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// neutralChainRoot refines a root of χ(s) = den + num·e^{-sτ} from the
// asymptotic guess s_m = (ln|d| + j·(2m+1)π)/τ (d > 0) by Newton.
func neutralChainRoot(q quasiLoop, m int) (complex128, bool) {
	d := q.num[0] / q.den[0]
	s := complex(math.Log(math.Abs(d))/q.tau, float64(2*m+1)*math.Pi/q.tau)
	if d < 0 {
		s = complex(math.Log(-d)/q.tau, float64(2*m)*math.Pi/q.tau)
	}
	dnum, dden := polyDerivCoeffs(q.num), polyDerivCoeffs(q.den)
	for range 60 {
		e := cmplx.Exp(-s * complex(q.tau, 0))
		f := polyEval(q.den, s) + polyEval(q.num, s)*e
		df := polyEval(dden, s) + (polyEval(dnum, s)-complex(q.tau, 0)*polyEval(q.num, s))*e
		step := f / df
		s -= step
		if cmplx.Abs(step) <= 1e-13*cmplx.Abs(s) {
			return s, true
		}
	}
	return s, false
}

func polyDerivCoeffs(p []float64) []float64 {
	n := len(p) - 1
	if n <= 0 {
		return []float64{0}
	}
	d := make([]float64, n)
	for i := range n {
		d[i] = p[i] * float64(n-i)
	}
	return d
}

func TestNyquistNeutralDelayLoopUnsupported(t *testing.T) {
	k, a1, a0 := 1.8749, 2.6223, 0.25411
	cases := []struct{ d, tau float64 }{
		{1.49706, 0.71947},
		{1.269, 2.713},
		{1.427, 2.950},
		{1.0144, 2.225},
		{-1.2, 1},
		{1, 1},
	}
	for _, c := range cases {
		q := quasiLoop{num: []float64{c.d, c.d*a1 + 0.3*k, c.d*a0 + k}, den: []float64{1, a1, a0}, tau: c.tau}
		if math.Abs(c.d) > 1 {
			for _, m := range []int{1000, 10000, 100000} {
				s, ok := neutralChainRoot(q, m)
				if !ok || !(real(s) > 0) {
					t.Fatalf("d=%g tau=%g: chain root %d = %v (converged %v), want Re > 0", c.d, c.tau, m, s, ok)
				}
			}
		}
		sys := q.system(t)
		lft, err := sys.PullDelaysToLFT()
		if err != nil {
			t.Fatal(err)
		}
		for _, m := range []*System{sys, lft} {
			if res, err := m.Nyquist(nil, 0); !errors.Is(err, ErrDelayUnsupported) {
				n := -1
				if res != nil {
					n = res.RHPZerosCL
				}
				t.Errorf("d=%g tau=%g lft=%v: RHPZerosCL %d, err %v; want ErrDelayUnsupported", c.d, c.tau, m.LFT != nil, n, err)
			}
		}
	}
}

func TestNyquistSubunitFeedthroughDelayLoopCounts(t *testing.T) {
	k, a1, a0 := 1.8749, 2.6223, 0.25411
	d := 0.3
	q := quasiLoop{num: []float64{d, d*a1 + 0.3*k, d*a0 + k}, den: []float64{1, a1, a0}, tau: 1}
	res, err := q.system(t).Nyquist(nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	want, ok := sweepOracle(q)
	if !ok {
		t.Fatal("oracle undecided")
	}
	if res.RHPZerosCL != want {
		t.Errorf("RHPZerosCL = %d, want %d", res.RHPZerosCL, want)
	}
}

type neutralBranch struct {
	d, tau float64
	a, c   float64 // rational part c/(s − a) when c != 0
}

func (b neutralBranch) system(t *testing.T) *System {
	t.Helper()
	var sys *System
	var err error
	if b.c != 0 {
		sys, err = NewFromSlices(1, 1, 1, []float64{b.a}, []float64{1}, []float64{b.c}, []float64{b.d}, 0)
	} else {
		sys, err = NewGain(mat.NewDense(1, 1, []float64{b.d}), 0)
	}
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{b.tau}); err != nil {
		t.Fatal(err)
	}
	return sys
}

func neutralParallel(t *testing.T, bs []neutralBranch) *System {
	t.Helper()
	L := bs[0].system(t)
	for _, b := range bs[1:] {
		var err error
		if L, err = Parallel(L, b.system(t)); err != nil {
			t.Fatal(err)
		}
	}
	return L
}

// neutralChi is Π(s − a)·(1 + L(s)) over the rational branches, entire.
func neutralChi(bs []neutralBranch, s complex128) complex128 {
	den := complex(1, 0)
	for _, b := range bs {
		if b.c != 0 {
			den *= s - complex(b.a, 0)
		}
	}
	f := den
	for _, b := range bs {
		g := den * complex(b.d, 0)
		if b.c != 0 {
			g += den / (s - complex(b.a, 0)) * complex(b.c, 0)
		}
		f += g * cmplx.Exp(-s*complex(b.tau, 0))
	}
	return f
}

// rectRoots counts the zeros of f in (0, x)×(−y, y) by the argument
// principle, bisecting wherever arg f moves by more than 0.2.
func rectRoots(f func(complex128) complex128, x, y float64) (int, bool) {
	corners := []complex128{complex(1e-9, -y), complex(x, -y), complex(x, y), complex(1e-9, y), complex(1e-9, -y)}
	total, ok := 0.0, true
	var walk func(a, b, fa, fb complex128, depth int)
	walk = func(a, b, fa, fb complex128, depth int) {
		if d := cmplx.Phase(fb / fa); math.Abs(d) <= 0.2 {
			total += d
			return
		}
		if depth > 40 {
			ok = false
			return
		}
		m := (a + b) / 2
		fm := f(m)
		walk(a, m, fa, fm, depth+1)
		walk(m, b, fm, fb, depth+1)
	}
	const n = 20000
	for i := range 4 {
		a, b := corners[i], corners[i+1]
		pa, fa := a, f(a)
		for k := 1; k <= n; k++ {
			p := a + (b-a)*complex(float64(k)/n, 0)
			fp := f(p)
			walk(pa, p, fa, fp, 0)
			pa, fa = p, fp
		}
	}
	z := total / (2 * math.Pi)
	return int(math.Round(z)), ok && math.Abs(z-math.Round(z)) < 1e-6
}

// Stable difference operators (all roots of 1 + Σd_k·z^{τ_k} outside the
// unit disk) with Σ|d_k| >= 1: χ has finitely many RHP roots, counted; the
// tail guard returned ErrDelayUnsupported for them.
func TestNyquistNeutralStableDifferenceCounts(t *testing.T) {
	cases := [][]neutralBranch{
		{{d: 0.6, tau: 1}, {d: 0.6, tau: 2}},
		{{d: -0.6, tau: 1}, {d: 0.5, tau: 2}},
		{{d: 0.796, tau: 0.966}, {d: 0.364, tau: 1.932}},
		{{d: 0.6, tau: 1, a: -1, c: 0.2}, {d: 0.6, tau: 2}},
		{{d: 0.6, tau: 1, a: 0.5, c: 0.2}, {d: 0.6, tau: 2}},
		{{d: 0.6, tau: 1, a: 0.5, c: 1.5}, {d: 0.6, tau: 2}},
		{{d: 0.6, tau: 1, a: 0, c: 0.2}, {d: 0.6, tau: 2}},
		{{d: 0.6, tau: 1, a: 0, c: -0.2}, {d: 0.6, tau: 2}},
		{{d: 0.6, tau: 1}, {d: 0.6, tau: 2}, {d: 0.3, tau: 3}},
		{{d: 0.5, tau: 1}, {d: 0.5, tau: 2}, {d: 0.5, tau: 3}},
	}
	for _, bs := range cases {
		want, ok := rectRoots(func(s complex128) complex128 { return neutralChi(bs, s) }, 20, 400)
		if !ok {
			t.Fatalf("%v: oracle undecided", bs)
		}
		L := neutralParallel(t, bs)
		for _, n := range []int{50, 500, 3000} {
			res, err := L.Nyquist(nil, n)
			if err != nil {
				t.Errorf("%v n=%d: %v, want RHPZerosCL %d", bs, n, err, want)
				continue
			}
			if res.RHPZerosCL != want {
				t.Errorf("%v n=%d: RHPZerosCL %d, want %d", bs, n, res.RHPZerosCL, want)
			}
		}
	}
}

// Seeded sweep of d1·e^{−s} + d2·e^{−ks}: Z = 0 when every root of
// 1 + d1·z + d2·z^k lies outside the unit disk, else infinitely many RHP
// roots and ErrDelayUnsupported.
func TestNyquistNeutralTwoGainSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(15, 41))
	finite, infinite := 0, 0
	for range 200 {
		k := 2 + rng.IntN(2)
		d1, d2 := 2*rng.Float64()-1, 2*rng.Float64()-1
		if math.Abs(d1)+math.Abs(d2) < 1 {
			continue
		}
		comp := mat.NewDense(k, k, nil)
		for i := range k - 1 {
			comp.Set(i+1, i, 1)
		}
		comp.Set(0, k-1, -1/d2)
		comp.Set(1, k-1, comp.At(1, k-1)-d1/d2)
		var eig mat.Eigen
		if !eig.Factorize(comp, mat.EigenNone) {
			t.Fatal("eigen")
		}
		rmin := math.Inf(1)
		for _, z := range eig.Values(nil) {
			rmin = min(rmin, cmplx.Abs(z))
		}
		if math.Abs(rmin-1) < 1e-3 {
			continue
		}
		L := neutralParallel(t, []neutralBranch{{d: d1, tau: 1}, {d: d2, tau: float64(k)}})
		res, err := L.Nyquist(nil, 200)
		if rmin > 1 {
			finite++
			if err != nil || res.RHPZerosCL != 0 {
				t.Errorf("d1=%g d2=%g k=%d |z|min=%g: %v, want RHPZerosCL 0", d1, d2, k, rmin, err)
			}
		} else {
			infinite++
			if !errors.Is(err, ErrDelayUnsupported) {
				t.Errorf("d1=%g d2=%g k=%d |z|min=%g: err %v, want ErrDelayUnsupported", d1, d2, k, rmin, err)
			}
		}
	}
	if finite < 10 || infinite < 10 {
		t.Fatalf("sweep covered %d finite, %d infinite cases", finite, infinite)
	}
}

// TestNyquistDelayRootsBelowResolution: 33894/(s+18.92)·e^{−5.23s} has
// about 5.6e4 RHP roots, those near |s| ~ k within 2π/(kτ²) ~ 7e-6 of the
// axis, inside the on-contour tolerance; it returned ErrSingularTransform as
// if the closed loop had axis poles. A delayed loop with a true axis root at
// ωτ ~ 2 still does.
func TestNyquistDelayRootsBelowResolution(t *testing.T) {
	loop := func(k, a, tau float64) *System {
		sys, err := NewFromSlices(1, 1, 1, []float64{-a}, []float64{1}, []float64{k}, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetInputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
		return sys
	}
	_, err := loop(33894, 18.92, 5.23).Nyquist(nil, 0)
	if !errors.Is(err, ErrDelayUnsupported) || errors.Is(err, ErrSingularTransform) {
		t.Errorf("ωτ ~ 1.8e5: err %v, want ErrDelayUnsupported", err)
	}
	_, err = loop(2, 1, 2*math.Pi/(3*math.Sqrt(3))).Nyquist(nil, 0)
	if !errors.Is(err, ErrSingularTransform) {
		t.Errorf("axis root at ω = √3: err %v, want ErrSingularTransform", err)
	}
}
