package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// closeZeroPairLoop returns L = (Q − D)/D·e^{−τs} with D = (s+1)⁴ and
// Q = ((s+δ)² + (ω₀−ε)²)((s+δ)² + (ω₀+ε)²), so that for small τ the
// closed-loop roots sit near −δ ± j(ω₀ ± ε): two zeros of 1+L within one
// sampling interval of the axis, which turn its phase by 2π between two
// grid points whose wrapped phase step is small.
func closeZeroPairLoop(w0, eps, delta, tau float64) quasiLoop {
	q1 := []float64{1, 2 * delta, delta*delta + (w0-eps)*(w0-eps)}
	q2 := []float64{1, 2 * delta, delta*delta + (w0+eps)*(w0+eps)}
	d := polyMul(polyMul([]float64{1, 1}, []float64{1, 1}), polyMul([]float64{1, 1}, []float64{1, 1}))
	num := polyAdd(polyMul(q1, q2), polyScale(d, -1))[1:]
	return quasiLoop{name: fmt.Sprintf("w0=%g/eps=%g/delta=%g/tau=%g", w0, eps, delta, tau), num: num, den: d, tau: tau, box: 200}
}

// TestDelayNyquistCloseZeroPair: a pair of closed-loop roots close to the
// axis between two grid samples turns arg(1+L) by 2π, which the endpoint
// phase step alone missed (stable loops reported with 2 RHP roots). The
// certified spans resolve it; the counts are pinned against the rectangle
// argument-principle oracle, which for τ → 0 agrees with the roots of Q.
func TestDelayNyquistCloseZeroPair(t *testing.T) {
	for _, q := range []quasiLoop{
		closeZeroPairLoop(10, 0.02, 0.005, 1e-8),
		closeZeroPairLoop(30, 0.1, 0.02, 1e-8),
		closeZeroPairLoop(10, 0.05, 0.01, 1e-8),
		closeZeroPairLoop(10, 0.05, -0.01, 1e-8),
		closeZeroPairLoop(10, 0.05, 0.01, 1e-4),
	} {
		t.Run(q.name, func(t *testing.T) {
			want := oracleRHPRoots(t, q.chi, q.box)
			sys := q.system(t)
			for _, frac := range []float64{0.5, 1e-3} {
				l, err := delayLoopFromSystem(sys, "delay loop")
				if err != nil {
					t.Fatal(err)
				}
				res, err := l.nyquist(frac)
				if err != nil || res.roots != want || res.stable != (want == 0) {
					t.Fatalf("frac=%g: roots=%d stable=%v err=%v, oracle %d", frac, res.roots, res.stable, err, want)
				}
			}
			cl, err := Feedback(sys, nil, -1)
			if err != nil {
				t.Fatal(err)
			}
			stable, err := cl.IsStable()
			if err != nil || stable != (want == 0) {
				t.Fatalf("IsStable(feedback) = %v, %v; oracle %d RHP roots", stable, err, want)
			}
			dm, err := DiskMargin(sys)
			if err != nil {
				t.Fatal(err)
			}
			if (dm.Alpha > 0) != (want == 0) {
				t.Fatalf("DiskMargin alpha=%g, oracle %d RHP roots", dm.Alpha, want)
			}
		})
	}
}

// TestStadiumPhaseBoundsSampledPhase checks stadiumPhase against dense
// samples of f(ω) = (jω − z₁)(jω − z₂), whose exact Taylor remainder is
// t²: an accepted half-span must contain no zero and its sampled phase
// variation must not exceed the bound.
func TestStadiumPhaseBoundsSampledPhase(t *testing.T) {
	z1, z2 := complex(-0.01, 9.95), complex(-0.02, 10.05)
	f := func(w float64) complex128 { s := complex(0, w); return (s - z1) * (s - z2) }
	df := func(w float64) complex128 { s := complex(0, w); return complex(0, 1) * (2*s - z1 - z2) }
	accepted := 0
	for a := 9.0; a < 11; a += 0.0137 {
		for _, h := range []float64{1e-3, 1e-2, 0.05, 0.2, -0.05, -0.2} {
			bound := stadiumPhase(f(a), df(a), h, h*h)
			if math.IsInf(bound, 1) {
				continue
			}
			accepted++
			lo, hi, phi := 0.0, 0.0, 0.0
			prev := f(a)
			for k := 1; k <= 2000; k++ {
				v := f(a + h*float64(k)/2000)
				if v == 0 {
					t.Fatalf("a=%g h=%g: accepted span holds a zero", a, h)
				}
				phi += cmplx.Phase(v / prev)
				lo, hi, prev = min(lo, phi), max(hi, phi), v
			}
			if hi-lo > bound+1e-9 {
				t.Fatalf("a=%g h=%g: sampled phase variation %g > bound %g", a, h, hi-lo, bound)
			}
		}
	}
	if accepted == 0 {
		t.Fatal("no half-span accepted")
	}
}
