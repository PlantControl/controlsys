package controlsys

import (
	"math/cmplx"
	"testing"
)

// TestDelayNyquistPoleLimitBesideCrowd: den = s²(s²+0.464s-6.2e-8), poles
// 0, 0, 1.338e-7 (RHP), -0.464. The double solve fails at ω ~ 4e-7 and the
// pole limit, which skips every pole within 1e-6 of jω, took ω = 3.93e-7
// for a pole and returned +Inf without an error. A walk from a tight
// double-double cluster at the origin then reported an axis root there.
// Oracles: L from num/den directly and the argument principle on χ.
func TestDelayNyquistPoleLimitBesideCrowd(t *testing.T) {
	q := quasiLoop{num: []float64{1.9188317375876882, 1.5083785090172463, 0.32156836056367194, 0.015658860719559062}, den: []float64{1, 0.46404121934763415, -6.209067747296065e-08, 0, 0}, tau: 0.001181949427158492}
	p := makePlant(t, q.num, q.den)
	if err := p.SetInputDelay([]float64{q.tau}); err != nil {
		t.Fatal(err)
	}
	newLoop := func() *delayLoop {
		l, err := delayLoopFromSystem(p, "delay loop")
		if err != nil {
			t.Fatal(err)
		}
		return l
	}
	l := newLoop()
	for _, w := range []float64{3.9298633264503444e-07, 1.5e-7, 2e-7, 3e-7, 3.93e-7, 5e-7, 8e-7} {
		got, want := l.at(w), q.L(w)
		if l.failed() || !(cmplx.Abs(got-want) <= 1e-6*cmplx.Abs(want)) {
			t.Errorf("L(j%g) = %v (failed %v), want %v", w, got, l.failed(), want)
		}
	}
	want := oracleRHPRoots(t, q.chi, 200, q.den, q.num)
	for _, g := range []float64{1.2e-8, 1.83e-8, 2.5e-8} {
		l := newLoop()
		a, ok := l.verifyCluster(0, g, 2, true)
		if !ok {
			t.Fatalf("gap %g: dd cluster of the double integrator does not verify", g)
		}
		l.clusters = []axisPole{a}
		if res, err := l.nyquist(0.5); err != nil || res.axisRoot || res.roots != want {
			t.Errorf("gap %g: nyquist roots %d, axis root %v at %g, %v; oracle %d", g, res.roots, res.axisRoot, res.axisW, err, want)
		}
	}
}
