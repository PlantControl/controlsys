package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// TestDelayNyquistDoubleIntegratorTinyPoles: s² with real poles at 6.3e-3,
// 7.5e-6 and -1.65e-3. The companion pencil sI-A is singular to working
// precision for |s| <= 1e-4 (det ~ 1e-16), so L was unevaluable at every
// indentation radius and IsStable, HinfNorm and DiskMargin errored. The
// double-double fallback evaluates the exact realization there.
func TestDelayNyquistDoubleIntegratorTinyPoles(t *testing.T) {
	q := quasiLoop{
		num: []float64{5.220393384215734, 9.189246525763267, 6.390061731529178, 1.4385802578468079, 0.06731719072338148},
		den: []float64{1, -0.0046923109514164976, -1.0433945828807056e-05, 7.829332736549319e-11, 0, 0},
		tau: 0.0029979187523625347, box: 200,
	}
	if want := oracleRHPRoots(t, q.chi, q.box, q.den, q.num); want != 0 {
		t.Fatalf("oracle %d RHP roots, want 0", want)
	}
	var supT, supS float64
	for i := range 200001 {
		l := q.L(math.Pow(10, -8+13*float64(i)/200000))
		supT = max(supT, cmplx.Abs(l/(1+l)))
		supS = max(supS, cmplx.Abs(1/(1+l)-0.5))
	}
	tf := &TransferFunc{Num: [][][]float64{{q.num}}, Den: [][]float64{q.den}}
	r, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	p := r.Sys
	if err := p.SetInputDelay([]float64{q.tau}); err != nil {
		t.Fatal(err)
	}
	cl, err := Feedback(p, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	if stable, err := cl.IsStable(); err != nil || !stable {
		t.Fatalf("IsStable = %v, %v; oracle 0 RHP roots", stable, err)
	}
	norm, _, err := HinfNorm(cl)
	if err != nil || !(norm >= supT*(1-1e-9) && norm <= supT*(1+1e-3)) {
		t.Fatalf("HinfNorm = %g, %v; dense sup|T| %g", norm, err, supT)
	}
	dm, err := DiskMargin(p)
	if err != nil || !(dm.Alpha <= 1/supS*(1+1e-9) && dm.Alpha >= 1/supS*(1-1e-3)) {
		t.Fatalf("DiskMargin = %+v, %v; dense alpha %g", dm, err, 1/supS)
	}
}

// TestDelayNyquistMultipleIntegrator: near a double or triple pole of L at
// the origin the Taylor model of 1+L reaches less than the bisection
// tolerance, so the indentation start ω ≈ 1e-7 was never certified: the
// det path had no S model, and the descriptor S model rounded at an
// absolute eps floor above |S| ~ ω²/K. IsStable, DiskMargin and the loop's
// nyquist errored where the counts are pinned against the rectangle
// argument-principle oracle.
func TestDelayNyquistMultipleIntegrator(t *testing.T) {
	for _, q := range []quasiLoop{
		{name: "lead/s2/fast", num: []float64{0.99517, 1.45104}, den: []float64{1, 0, 0}, tau: 0.0047, box: 200},
		{name: "lead/s2", num: []float64{1, 0.5}, den: []float64{1, 0, 0}, tau: 0.1, box: 200},
		{name: "s2(s+a)", num: []float64{2.19702}, den: []float64{1, 0.39245, 0, 0}, tau: 0.00813, box: 200},
		{name: "s3", num: []float64{0.2, 0.3, 0.05}, den: []float64{1, 0, 0, 0}, tau: 0.05, box: 200},
	} {
		t.Run(q.name, func(t *testing.T) {
			want := oracleRHPRoots(t, q.chi, q.box, q.den, q.num)
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

// TestTunePIDIntegratingDelayedPlant: every PID/PIDF candidate on an
// integrating delayed plant was rejected as not certified stable, the ideal
// derivative loop through hidden modes at the plant poles. The returned
// controller must close a loop with no RHP roots per the oracle.
func TestTunePIDIntegratingDelayedPlant(t *testing.T) {
	for _, c := range []struct {
		num, den []float64
		tau      float64
		fam      PidtuneType
	}{
		{[]float64{1}, []float64{1, 0, 0}, 0.05, PidtunePIDF},
		{[]float64{1}, []float64{1, 0, 0}, 0.05, PidtunePID},
		{[]float64{2.273}, []float64{1, 0, 0}, 0.0681, PidtunePID},
		{[]float64{3.6257}, []float64{1, 0}, 0.0108, PidtunePIDF},
	} {
		p := makePlant(t, c.num, c.den)
		if err := p.SetInputDelay([]float64{c.tau}); err != nil {
			t.Fatal(err)
		}
		r, err := TunePID(context.Background(), p, c.fam, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("%v/%v τ=%g: %v", c.num, c.den, c.tau, err)
		}
		k := r.Controller
		numC, denC := []float64{k.Kd, k.Kp, k.Ki}, []float64{1, 0}
		if k.Tf != 0 {
			numC = []float64{k.Kp*k.Tf + k.Kd, k.Kp + k.Ki*k.Tf, k.Ki}
			denC = []float64{k.Tf, 1, 0}
		}
		q := quasiLoop{num: polyMul(numC, c.num), den: polyMul(denC, c.den), tau: c.tau}
		if n := oracleRHPRoots(t, q.chi, 200, q.den, q.num); n != 0 {
			t.Fatalf("%v/%v τ=%g: %+v closes %d RHP roots", c.num, c.den, c.tau, *k, n)
		}
	}
}
