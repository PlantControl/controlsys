package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand"
	"testing"
)

type sisoDelayLoop struct {
	a, b1, b2, c1, c2, d11, d12, d21, d22, h float64
}

func randomSISODelayLoop(rng *rand.Rand) sisoDelayLoop {
	p := sisoDelayLoop{
		a: -1 - 3*rng.Float64(), b1: 0.3 * rng.NormFloat64(), b2: rng.NormFloat64(),
		c1: rng.NormFloat64(), c2: rng.NormFloat64(),
		d11: rng.NormFloat64(), d12: rng.NormFloat64(), d21: rng.NormFloat64(),
		d22: 0.2 + 0.7*rng.Float64(),
	}
	if rng.NormFloat64() < 0 {
		p.d22 = -p.d22
	}
	p.h = 0.1 + rng.Float64()
	return p
}

func (p sisoDelayLoop) siso(t *testing.T) *System {
	t.Helper()
	H, err := NewFromSlices(1, 2, 2, []float64{p.a}, []float64{p.b1, p.b2}, []float64{p.c1, p.c2},
		[]float64{p.d11, p.d12, p.d21, p.d22}, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, []float64{p.h})
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// replicate returns the block-diagonal of p repeated once per scale, channel
// i's input scaled by ks[i]: closed-loop roots are those of p, each with
// multiplicity len(ks).
func (p sisoDelayLoop) replicate(t *testing.T, ks ...float64) *System {
	t.Helper()
	n := len(ks)
	a := make([]float64, n*n)
	b := make([]float64, n*2*n)
	c := make([]float64, 2*n*n)
	d := make([]float64, 4*n*n)
	tau := make([]float64, n)
	for i, k := range ks {
		a[i*n+i] = p.a
		b[i*2*n+i], b[i*2*n+n+i] = k*p.b1, p.b2
		c[i*n+i], c[(n+i)*n+i] = p.c1, p.c2
		d[i*2*n+i], d[i*2*n+n+i] = k*p.d11, p.d12
		d[(n+i)*2*n+i], d[(n+i)*2*n+n+i] = k*p.d21, p.d22
		tau[i] = p.h
	}
	H, err := NewFromSlices(n, 2*n, 2*n, a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// TestHinfNormDelayDuplicatedLightlyDamped: the loop has a closed-loop root
// pair at −0.0027828 ± 2.8446939j (Newton on χ below), so 1+L of the
// block-diagonal duplicate, det = (1+L₁)², turns by 2π between two grid samples
// that the endpoint phase step alone saw as −36°; the 4-fold replica by 4π.
func TestHinfNormDelayDuplicatedLightlyDamped(t *testing.T) {
	p := sisoDelayLoop{a: -1.4666973867074775, b1: 1, b2: -2.2697439281287144, c1: 1, c2: 1,
		d12: 1, d21: 1, d22: -0.44819513831474195, h: 0.8637187571421795}
	chi := func(s complex128) complex128 {
		e := cmplx.Exp(-s * complex(p.h, 0))
		return (s-complex(p.a, 0))*(1-complex(p.d22, 0)*e) - complex(p.b2*p.c2, 0)*e
	}
	s := complex(0, 2.84)
	for range 50 {
		const d = 1e-7
		s -= chi(s) * complex(d, 0) / (chi(s+d) - chi(s))
	}
	if math.Abs(real(s)+0.0027828) > 1e-6 || math.Abs(imag(s)-2.8446939) > 1e-6 {
		t.Fatalf("oracle root %v", s)
	}

	want, wantW, err := HinfNorm(p.siso(t))
	if err != nil || math.IsInf(want, 1) {
		t.Fatalf("SISO HinfNorm = %g, %v", want, err)
	}
	if math.Abs(wantW-imag(s)) > 1e-3 {
		t.Fatalf("SISO peak at %g, root at %g", wantW, imag(s))
	}
	for _, ks := range [][]float64{{1, 1}, {1, 1 - 1e-9}, {1, 1, 1, 1}, {1, 1 - 1e-9, 1 + 1e-9, 1}} {
		got, w, err := HinfNorm(p.replicate(t, ks...))
		if err != nil {
			t.Fatalf("ks=%v: %v", ks, err)
		}
		if math.Abs(got-want) > 1e-6*want || math.Abs(w-wantW) > 1e-4 {
			t.Errorf("ks=%v: HinfNorm = %g at %g, want %g at %g", ks, got, w, want, wantW)
		}
	}
}

// TestHinfNormDelayDuplicateSweep: replicating a channel 2 or 4 times raises
// every closed-loop root to that multiplicity, so the replica is stable
// exactly when the channel is and has its norm.
func TestHinfNormDelayDuplicateSweep(t *testing.T) {
	skipSlowUnderRace(t)
	const seeds = 300
	unstable := 0
	for seed := range seeds {
		p := randomSISODelayLoop(rand.New(rand.NewSource(int64(seed))))
		want, _, err := HinfNorm(p.siso(t))
		if err != nil {
			continue
		}
		if math.IsInf(want, 1) {
			unstable++
		}
		for _, ks := range [][]float64{{1, 1}, {1, 1, 1, 1}} {
			got, _, err := HinfNorm(p.replicate(t, ks...))
			switch {
			case err != nil:
				t.Errorf("seed %d, %d copies: %v (SISO %g)", seed, len(ks), err, want)
			case math.IsInf(want, 1) != math.IsInf(got, 1) || !math.IsInf(want, 1) && math.Abs(got-want) > 1e-6*want:
				t.Errorf("seed %d, %d copies: %g, SISO %g", seed, len(ks), got, want)
			}
		}
	}
	if unstable == 0 || unstable == seeds {
		t.Fatalf("%d of %d unstable; sweep covers one side only", unstable, seeds)
	}
}

// rhpRoots counts the roots of χ(s) = (s−a)(1−d22·e^{−sh}) − b2c2·e^{−sh} in
// Re s > 0. They solve e^{sh} = q(s) = d22 + b2c2/(s−a), with |q| >= 1 only
// for |s−a| <= |b2c2|/(1−|d22|), so the branches s = (Log q(s) + 2πjk)/h
// for |k| up to that radius hold them all, Log taken about arg d22 so that
// the branches do not jump across its cut; each branch root is found by
// fixed-point iteration and polished by Newton on χ.
func (p sisoDelayLoop) rhpRoots(t *testing.T) int {
	t.Helper()
	bc := complex(p.b2*p.c2, 0)
	h := complex(p.h, 0)
	q := func(s complex128) complex128 { return complex(p.d22, 0) + bc/(s-complex(p.a, 0)) }
	chi := func(s complex128) (complex128, complex128) {
		e := cmplx.Exp(-s * h)
		sa := s - complex(p.a, 0)
		return sa*(1-complex(p.d22, 0)*e) - bc*e, 1 - complex(p.d22, 0)*e + h*e*(complex(p.d22, 0)*sa+bc)
	}
	radius := math.Abs(p.b2*p.c2)/(1-math.Abs(p.d22)) + math.Abs(p.a) + 1
	K := int(radius*p.h/(2*math.Pi)) + 2
	logD := cmplx.Log(complex(p.d22, 0))
	n := 0
	for k := -K; k <= K; k++ {
		jk := complex(0, 2*math.Pi*float64(k))
		s := (logD + jk) / h
		for range 200 {
			s = (cmplx.Log(q(s)/complex(p.d22, 0)) + logD + jk) / h
		}
		for range 20 {
			v, d := chi(s)
			s -= v / d
		}
		if v, _ := chi(s); cmplx.Abs(v) > 1e-8*(1+cmplx.Abs(s)) {
			t.Fatalf("branch %d root %v not converged: χ = %v", k, s, v)
		}
		if real(s) > 0 {
			n++
		}
	}
	return n
}

// TestDelayNyquistNearNeutralBudget: |d22| within 1e-6 of 1 puts a dip
// of |1+L| down to about 1−|d22| in every delay period. The refined grid must
// not bisect the flanks of these resolved simple zeros, which took these
// loops (close to the point budget already) past it. The root counts are
// checked against the branch-root oracle rhpRoots.
func TestDelayNyquistNearNeutralBudget(t *testing.T) {
	for _, c := range []struct {
		eps   float64
		seed  int64
		roots int
	}{{1e-6, 15, 0}, {1e-6, 19, 80}} {
		p := randomSISODelayLoop(rand.New(rand.NewSource(c.seed)))
		p.d22 = math.Copysign(1-c.eps, p.d22)
		if got := p.rhpRoots(t); got != c.roots {
			t.Fatalf("eps=%g seed=%d: oracle %d RHP roots, want %d", c.eps, c.seed, got, c.roots)
		}
		e, err := newDelayLFT(p.siso(t))
		if err != nil {
			t.Fatal(err)
		}
		_, res, err := e.stability()
		if err != nil {
			t.Errorf("eps=%g seed=%d: %v", c.eps, c.seed, err)
			continue
		}
		if res.roots != c.roots || res.stable != (c.roots == 0) {
			t.Errorf("eps=%g seed=%d: %d RHP roots (stable %v), oracle %d", c.eps, c.seed, res.roots, res.stable, c.roots)
		}
	}
}

// nearNeutralSISODelayLoop draws randomSISODelayLoop, for odd seeds with a
// slow pole, a larger loop gain and a longer delay, and sets |d22| within
// eps of 1.
func nearNeutralSISODelayLoop(seed int64, eps float64) sisoDelayLoop {
	rng := rand.New(rand.NewSource(seed))
	p := randomSISODelayLoop(rng)
	if seed%2 == 1 {
		p.a = -0.05 - 3*rng.Float64()
		p.b2 *= 1 + 3*rng.Float64()
		p.h = 0.1 + 3*rng.Float64()
	}
	p.d22 = math.Copysign(1-eps*rng.Float64(), p.d22)
	return p
}

// rectRoots counts the roots of χ(s) = (s−a)(1−d22·e^{−sh}) − b2c2·e^{−sh}
// in Re s > 0 by the argument principle on the rectangle [0, R]×[−R, R],
// R past |s−a| <= |b2c2|/(1−|d22|) that holds them all, bisecting each
// boundary piece until arg χ moves by less than π/16 and log|χ| by less than
// 0.2.
func (p sisoDelayLoop) rectRoots() int {
	bc := complex(p.b2*p.c2, 0)
	h := complex(p.h, 0)
	chi := func(s complex128) complex128 {
		e := cmplx.Exp(-s * h)
		return (s-complex(p.a, 0))*(1-complex(p.d22, 0)*e) - bc*e
	}
	R := math.Abs(p.b2*p.c2)/(1-math.Abs(p.d22)) + math.Abs(p.a) + 1
	corners := []complex128{complex(0, -R), complex(R, -R), complex(R, R), complex(0, R), complex(0, -R)}
	turn := 0.0
	var walk func(a, b, fa, fb complex128, depth int)
	walk = func(a, b, fa, fb complex128, depth int) {
		d := wrapPi(cmplx.Phase(fb) - cmplx.Phase(fa))
		if math.Abs(d) < math.Pi/16 && math.Abs(math.Log(cmplx.Abs(fb)/cmplx.Abs(fa))) < 0.2 || depth > 60 {
			turn += d
			return
		}
		m := (a + b) / 2
		fm := chi(m)
		walk(a, m, fa, fm, depth+1)
		walk(m, b, fm, fb, depth+1)
	}
	for i := range len(corners) - 1 {
		a, b := corners[i], corners[i+1]
		n := int(cmplx.Abs(b-a)*p.h*16) + 16
		for k := range n {
			s0 := a + (b-a)*complex(float64(k)/float64(n), 0)
			s1 := a + (b-a)*complex(float64(k+1)/float64(n), 0)
			walk(s0, s1, chi(s0), chi(s1), 0)
		}
	}
	return int(math.Round(turn / (2 * math.Pi)))
}

// pair returns the block-diagonal of p and q.
func (p sisoDelayLoop) pair(t *testing.T, q sisoDelayLoop) *System {
	t.Helper()
	a := make([]float64, 4)
	b := make([]float64, 8)
	c := make([]float64, 8)
	d := make([]float64, 16)
	tau := make([]float64, 2)
	for i, p := range []sisoDelayLoop{p, q} {
		a[i*2+i] = p.a
		b[i*4+i], b[i*4+2+i] = p.b1, p.b2
		c[i*2+i], c[(2+i)*2+i] = p.c1, p.c2
		d[i*4+i], d[i*4+2+i] = p.d11, p.d12
		d[(2+i)*4+i], d[(2+i)*4+2+i] = p.d21, p.d22
		tau[i] = p.h
	}
	H, err := NewFromSlices(2, 4, 4, a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := SetDelayModel(H, tau)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// TestHinfNormDelayNearNeutralReplica: with |d22| near 1 the normalized loop
// det(I − H22Δ)/det(I − D22Δ) of k identical channels has poles of order k
// near the axis, where its phase turns by about kπ within one grid step and a
// 2π turn passes for a small one. The count went wrong both ways: too few
// (finite norm of an unstable replica) and negative (+Inf for a stable one).
// Two channels whose d22 differ by 1e-7 split each double root into a close
// pair.
func TestHinfNormDelayNearNeutralReplica(t *testing.T) {
	skipSlowUnderRace(t)
	cases := []struct {
		name  string
		p     sisoDelayLoop
		roots int
		pair  bool
	}{
		{name: "false stable", p: sisoDelayLoop{a: -3.4495080813388497, b1: -0.057854272810938155, b2: 0.16092035055548803,
			c1: 0.6025740677484148, c2: -1.0552115640699384, d11: 0.38036538276901166, d12: -2.4122147266495095,
			d21: 0.9918311278676921, d22: -0.9993353290172704, h: 0.10841451899761129}, roots: 2},
		{name: "false unstable", p: sisoDelayLoop{a: -1.7122239672475017, b1: -0.4806591033435851, b2: 1.1423007290835272,
			c1: 0.20220161351732413, c2: 0.6884135686976489, d11: -0.04704061087516162, d12: -0.9007179545656091,
			d21: 0.1134943999550504, d22: -0.9940874638769287, h: 1.9262380841867917}, roots: 0},
		{name: "eps 0.01 seed 10", p: nearNeutralSISODelayLoop(10, 0.01), roots: 0},
		{name: "eps 0.01 seed 7", p: nearNeutralSISODelayLoop(7, 0.01), roots: 20},
		{name: "pair eps 0.03 seed 84", p: nearNeutralSISODelayLoop(84, 0.03), roots: 0, pair: true},
		{name: "pair eps 0.03 seed 7", p: nearNeutralSISODelayLoop(7, 0.03), roots: 12, pair: true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.p.rectRoots(); got != c.roots {
				t.Fatalf("oracle %d RHP roots, want %d", got, c.roots)
			}
			want, _, err := HinfNorm(c.p.siso(t))
			if err != nil || math.IsInf(want, 1) != (c.roots > 0) {
				t.Fatalf("SISO HinfNorm = %g, %v; oracle %d roots", want, err, c.roots)
			}
			sys, n := c.p.replicate(t, 1, 1, 1, 1), 4
			if c.pair {
				q := c.p
				q.d22 *= 1 + 1e-7
				if got := q.rectRoots(); got != c.roots {
					t.Fatalf("perturbed oracle %d RHP roots", got)
				}
				wq, _, err := HinfNorm(q.siso(t))
				if err != nil {
					t.Fatal(err)
				}
				want = max(want, wq)
				sys, n = c.p.pair(t, q), 2
			}
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			_, res, err := e.stability()
			if err != nil || res.roots != n*c.roots {
				t.Errorf("%d RHP roots, %v; oracle %d", res.roots, err, n*c.roots)
			}
			got, _, err := HinfNorm(sys)
			switch {
			case err != nil:
				t.Errorf("HinfNorm: %v", err)
			case math.IsInf(want, 1) != math.IsInf(got, 1) || !math.IsInf(want, 1) && math.Abs(got-want) > 1e-6*want:
				t.Errorf("HinfNorm = %g, SISO %g", got, want)
			}
		})
	}
}

// TestDelayNyquistNegativeCount: 1+L = (s+1)/(s−1) winds once
// counterclockwise, which with no open-loop RHP pole on record gives −1
// closed-loop roots; that count is impossible and must not read as unstable.
func TestDelayNyquistNegativeCount(t *testing.T) {
	sys, err := NewFromSlices(1, 1, 1, []float64{1}, []float64{1}, []float64{2}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	l, err := delayLoopFromSystem(sys, "delay loop")
	if err != nil {
		t.Fatal(err)
	}
	l.rhp = 0
	if res, err := l.nyquist(0.5); !errors.Is(err, errDelayLoopUnsupported) {
		t.Fatalf("roots %d, stable %v, err %v", res.roots, res.stable, err)
	}
}
