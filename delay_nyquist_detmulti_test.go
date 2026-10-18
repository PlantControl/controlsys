package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDelayNyquistDetMultipleAxisPole: with two or more delay channels the
// det(I − H22Δ) certifier had only the Taylor model of H22, whose reach next
// to a double or triple pole of H22 at the origin falls below the bisection
// tolerance, so IsStable and HinfNorm returned ErrDelayUnsupported at the
// indentation ω ≈ 1e-7. Verdicts are pinned against the rectangle
// argument-principle oracle on χ, the H∞ norm against the SISO parts.
func TestDelayNyquistDetMultipleAxisPole(t *testing.T) {
	s2 := quasiLoop{num: []float64{1, 0.5}, den: []float64{1, 0, 0}, tau: 0.1}
	s3 := quasiLoop{num: []float64{0.2, 0.3, 0.05}, den: []float64{1, 0, 0, 0}, tau: 0.05}
	fast := quasiLoop{num: []float64{0.99517, 1.45104}, den: []float64{1, 0, 0}, tau: 0.0047}
	hot := quasiLoop{num: []float64{30, 15}, den: []float64{1, 0, 0}, tau: 0.1}
	for _, c := range []struct {
		name string
		a, b quasiLoop
	}{
		{"s2+s3", s2, s3},
		{"s2+fast", s2, fast},
		{"s3+s3", s3, s3},
		{"s2+hot", s2, hot},
	} {
		t.Run(c.name, func(t *testing.T) {
			want := oracleRHPRoots(t, func(s complex128) complex128 { return c.a.chi(s) * c.b.chi(s) }, 200, c.a.den, c.a.num, c.b.den, c.b.num)
			ca, err := Feedback(c.a.system(t), nil, -1)
			if err != nil {
				t.Fatal(err)
			}
			cb, err := Feedback(c.b.system(t), nil, -1)
			if err != nil {
				t.Fatal(err)
			}
			sys, err := Append(ca, cb)
			if err != nil {
				t.Fatal(err)
			}
			if len(sys.LFT.Tau) != 2 {
				t.Fatalf("%d delay channels, want 2", len(sys.LFT.Tau))
			}
			stable, err := sys.IsStable()
			if err != nil || stable != (want == 0) {
				t.Fatalf("IsStable = %v, %v; oracle %d RHP roots", stable, err, want)
			}
			got, _, err := HinfNorm(sys)
			if want != 0 {
				if err != nil || !math.IsInf(got, 1) {
					t.Fatalf("HinfNorm = %g, %v; unstable", got, err)
				}
				return
			}
			na, _, errA := HinfNorm(ca)
			nb, _, errB := HinfNorm(cb)
			if errA != nil || errB != nil {
				t.Fatalf("SISO HinfNorm: %v, %v", errA, errB)
			}
			if ref := max(na, nb); err != nil || math.Abs(got-ref) > 1e-6*ref {
				t.Fatalf("HinfNorm = %g, %v; SISO parts %g, %g", got, err, na, nb)
			}
		})
	}
}

// TestDelayNyquistDetCoupledDoubleIntegrator: P = (s+0.5)/s² under
// K = k1·e^{−sτ1} + k2·e^{−sτ2} couples both delay channels through the
// double integrator, χ = s² + (s+0.5)(k1·e^{−sτ1} + k2·e^{−sτ2}). The H∞
// norm is checked against a dense sweep of the exact response.
func TestDelayNyquistDetCoupledDoubleIntegrator(t *testing.T) {
	const tau1, tau2 = 0.1, 0.25
	for _, c := range []struct{ k1, k2 float64 }{{0.6, 0.4}, {0.3, 0.9}, {8, 6}, {-0.2, 0.7}} {
		kAt := func(s complex128) complex128 {
			return complex(c.k1, 0)*cmplx.Exp(-s*tau1) + complex(c.k2, 0)*cmplx.Exp(-s*tau2)
		}
		want := oracleRHPRoots(t, func(s complex128) complex128 { return s*s + (s+0.5)*kAt(s) }, 200, []float64{1, 0, 0}, []float64{1, 0.5})
		gain := func(k, tau float64) *System {
			g, err := NewGain(mat.NewDense(1, 1, []float64{k}), 0)
			if err != nil {
				t.Fatal(err)
			}
			if err := g.SetInputDelay([]float64{tau}); err != nil {
				t.Fatal(err)
			}
			return g
		}
		k, err := Parallel(gain(c.k1, tau1), gain(c.k2, tau2))
		if err != nil {
			t.Fatal(err)
		}
		cl, err := Feedback(makePlant(t, []float64{1, 0.5}, []float64{1, 0, 0}), k, -1)
		if err != nil {
			t.Fatal(err)
		}
		if cl.LFT == nil || len(cl.LFT.Tau) < 2 {
			t.Fatalf("k=%g,%g: want ≥2 delay channels", c.k1, c.k2)
		}
		stable, err := cl.IsStable()
		if err != nil || stable != (want == 0) {
			t.Fatalf("k=%g,%g: IsStable = %v, %v; oracle %d RHP roots", c.k1, c.k2, stable, err, want)
		}
		got, _, err := HinfNorm(cl)
		if want != 0 {
			if err != nil || !math.IsInf(got, 1) {
				t.Fatalf("k=%g,%g: HinfNorm = %g, %v; unstable", c.k1, c.k2, got, err)
			}
			continue
		}
		ref := 0.0
		for i := range 200001 {
			s := complex(0, 1e-3*math.Pow(1e7, float64(i)/200000))
			pv := (s + 0.5) / (s * s)
			ref = max(ref, cmplx.Abs(pv/(1+pv*kAt(s))))
		}
		if err != nil || got < ref*(1-1e-9) || got > ref*(1+1e-4) {
			t.Fatalf("k=%g,%g: HinfNorm = %g, %v; sweep %g", c.k1, c.k2, got, err, ref)
		}
	}
}

// TestDetCertifierChiPhaseBoundsSamples checks the det M phase bound (kind 1)
// against dense samples of det(I − H22Δ), and of its ratio to det(I − D22Δ):
// an accepted half-span must hold no zero and its sampled phase variation
// must not exceed the bound. The plants hold a double integrator coupling
// two channels, a lightly damped mode at 2 rad/s and random coupled delays.
func TestDetCertifierChiPhaseBoundsSamples(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	random := func(n, nd int, shift float64) *System {
		k := 1 + nd
		a := make([]float64, n*n)
		for i := range a {
			a[i] = rng.NormFloat64()
		}
		for i := range n {
			a[i*n+i] -= shift
		}
		b, c, d := make([]float64, n*k), make([]float64, k*n), make([]float64, k*k)
		for i := range b {
			b[i], c[i] = rng.NormFloat64(), rng.NormFloat64()
		}
		for i := range d {
			d[i] = 0.25 * rng.NormFloat64()
		}
		H, err := NewFromSlices(n, k, k, a, b, c, d, 0)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := SetDelayModel(H, []float64{0.3, 0.7, 1.1}[:nd])
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	resonant := func() *System {
		H, err := NewFromSlices(2, 3, 3,
			[]float64{-0.05, 2, -2, -0.05},
			[]float64{1, 0.5, -0.3, 0, 0.8, 0.4},
			[]float64{1, 0, 0.4, -0.6, 0.2, 0.9},
			[]float64{0, 0, 0, 0, 0.1, 0.2, 0, -0.1, 0.15}, 0)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := SetDelayModel(H, []float64{0.2, 0.45})
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	rotating := func() *System {
		H, err := NewFromSlices(2, 3, 3,
			[]float64{-1, 0, 0, -2},
			[]float64{1, 10, 0, 0, 0, 0.1},
			[]float64{1, 0, 1, 1, 0, 0},
			make([]float64, 9), 0)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := SetDelayModel(H, []float64{20, 3})
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	neutral := func() *System {
		H, err := NewFromSlices(2, 3, 3,
			[]float64{-1, 0.5, 0, -2},
			[]float64{1, 0.2, 0, 0, 0, 0.1},
			[]float64{1, 0, 1, 0.3, 0, 0.2},
			[]float64{0, 0, 0, 0, 0.95, 0.1, 0, 0, -0.6}, 0)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := SetDelayModel(H, []float64{4, 7})
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	integrator := func() *System {
		g := func(k, tau float64) *System {
			s, err := NewGain(mat.NewDense(1, 1, []float64{k}), 0)
			if err != nil {
				t.Fatal(err)
			}
			if err := s.SetInputDelay([]float64{tau}); err != nil {
				t.Fatal(err)
			}
			return s
		}
		k, err := Parallel(g(0.6, 0.1), g(0.4, 0.25))
		if err != nil {
			t.Fatal(err)
		}
		cl, err := Feedback(makePlant(t, []float64{1, 0.5}, []float64{1, 0, 0}), k, -1)
		if err != nil {
			t.Fatal(err)
		}
		return cl
	}
	for _, c := range []struct {
		name string
		sys  *System
		ws   []float64
	}{
		{"integrator", integrator(), []float64{1e-7, 3e-6, 1e-4, 0.01, 0.4, 2, 9}},
		{"resonant", resonant(), []float64{0.05, 1.7, 1.96, 2.04, 2.3, 6}},
		{"rotating", rotating(), []float64{0.02, 0.05, 0.1, 0.2, 0.3}},
		{"neutral", neutral(), []float64{0.1, 0.8, 1.5, 1.5708, 1.6, 3, 10}},
		{"random2", random(3, 2, 4), []float64{0.05, 0.9, 2.3, 7, 31}},
		{"random3", random(3, 3, 4), []float64{0.05, 0.9, 2.3, 7, 31}},
	} {
		e, err := newDelayLFT(c.sys)
		if err != nil {
			t.Fatal(err)
		}
		for _, normalized := range []bool{false, true} {
			cert, err := newDetCertifier(e, normalized)
			if err != nil {
				t.Fatal(err)
			}
			if cert.kinds() != 2 {
				t.Fatalf("%s: %d kinds", c.name, cert.kinds())
			}
			f := e.char
			if normalized {
				f = e.normalizedChar
			}
			accepted := 0
			for _, w := range c.ws {
				m, ok := cert.model(1, w)
				if !ok {
					t.Fatalf("%s normalized=%v: no det M model at ω=%g", c.name, normalized, w)
				}
				for _, rel := range []float64{1e-3, 1e-2, 0.05, 0.2, 1, -0.05, -0.5} {
					h := rel * w
					bound := cert.phase(1, &m, h)
					if math.IsInf(bound, 1) {
						continue
					}
					accepted++
					lo, hi, phi := 0.0, 0.0, 0.0
					prev := f(w)
					for k := 1; k <= 4000; k++ {
						v := f(w + h*float64(k)/4000)
						if v == 0 || !finite(v) {
							t.Fatalf("%s normalized=%v ω=%g h=%g: accepted span holds a zero or pole", c.name, normalized, w, h)
						}
						phi += cmplx.Phase(v / prev)
						lo, hi, prev = min(lo, phi), max(hi, phi), v
					}
					if hi-lo > bound+1e-9 {
						t.Fatalf("%s normalized=%v ω=%g h=%g: sampled phase variation %g > bound %g", c.name, normalized, w, h, hi-lo, bound)
					}
				}
			}
			if accepted == 0 {
				t.Fatalf("%s normalized=%v: no half-span accepted", c.name, normalized)
			}
		}
	}
}

// TestDelayNyquistDetSplitDelayDoubleIntegrator: a double-integrator loop
// with its delay split between input and output closes into two delay
// channels; IsStable and HinfNorm errored at the indentation ω ≈ 1e-7
// (BRT7G3). Verdicts are pinned against the argument-principle oracle on χ,
// the H∞ norm of L/(1+L) against a dense sweep.
func TestDelayNyquistDetSplitDelayDoubleIntegrator(t *testing.T) {
	for _, c := range []struct {
		num, den []float64
		in, out  float64
	}{
		{[]float64{1, 0.5}, []float64{1, 0, 0}, 0.05, 0.05},
		{[]float64{1, 0.5}, []float64{1, 0, 0}, 0.02, 0.13},
		{[]float64{0.3, 0.1}, []float64{1, 0, 0}, 0.4, 0.3},
		{[]float64{4, 2}, []float64{1, 0, 0}, 0.1, 0.2},
		{[]float64{2.19702}, []float64{1, 0.39245, 0, 0}, 0.00813, 0.00813},
		{[]float64{0.05}, []float64{1, 0.39245, 0, 0}, 0.3, 0.1},
	} {
		q := quasiLoop{num: c.num, den: c.den}
		sys := q.system(t)
		if err := sys.SetInputDelay([]float64{c.in}); err != nil {
			t.Fatal(err)
		}
		if err := sys.SetOutputDelay([]float64{c.out}); err != nil {
			t.Fatal(err)
		}
		cl, err := Feedback(sys, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		if cl.LFT == nil || len(cl.LFT.Tau) != 2 {
			t.Fatalf("%v/%v: want 2 delay channels", c.num, c.den)
		}
		q.tau = c.in + c.out
		want := oracleRHPRoots(t, q.chi, 200, q.den, q.num)
		stable, err := cl.IsStable()
		if err != nil || stable != (want == 0) {
			t.Fatalf("%v/%v τ=%g+%g: IsStable = %v, %v; oracle %d RHP roots", c.num, c.den, c.in, c.out, stable, err, want)
		}
		got, _, err := HinfNorm(cl)
		if want != 0 {
			if err != nil || !math.IsInf(got, 1) {
				t.Fatalf("%v/%v: HinfNorm = %g, %v; unstable", c.num, c.den, got, err)
			}
			continue
		}
		ref := 0.0
		for i := range 200001 {
			l := q.L(1e-3 * math.Pow(1e7, float64(i)/200000))
			ref = max(ref, cmplx.Abs(l/(1+l)))
		}
		if err != nil || got < ref*(1-1e-9) || got > ref*(1+1e-4) {
			t.Fatalf("%v/%v τ=%g+%g: HinfNorm = %g, %v; sweep %g", c.num, c.den, c.in, c.out, got, err, ref)
		}
	}
}
