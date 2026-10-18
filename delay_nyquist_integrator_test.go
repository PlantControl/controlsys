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

// TestDelayNyquistIntegratorNearOriginPoles: multiple integrators with real
// poles closer to the origin than indentLimit's 1e-6 cutoff. With the dd
// fallback, indent evaluated L at an enlarged radius enclosing them, saw |L|
// fall by more decades than mult and took it for an axis root at ω=0:
// IsStable false, HinfNorm +Inf, DiskMargin 0, although χ(0) = num(0) ≠ 0.
// An off-axis pole inside the indentation now leaves the count to
// clusterWinding, which errors here like the base did. With exact, the pole
// sits outside the indentation and the values must match the oracles.
func TestDelayNyquistIntegratorNearOriginPoles(t *testing.T) {
	for _, c := range []struct {
		q     quasiLoop
		exact bool
	}{
		{q: quasiLoop{ // s³, poles 1.31e-9, -2.72e-9, -0.1286
			num: []float64{13.222200959921482, 63.741266377137514, 129.45539425818563, 104.87423932251659, 34.825187773001446, 3.8595098872161198},
			den: []float64{1, 0.12864478027547946, 1.8146789689495635e-10, -4.594790547398718e-19, 0, 0, 0},
			tau: 0.003604611258568011, box: 200,
		}},
		{q: quasiLoop{ // s², poles 1.34e-7, -0.464
			num: []float64{1.9188317375876882, 1.5083785090172463, 0.32156836056367194, 0.015658860719559062},
			den: []float64{1, 0.46404121934763415, -6.209067747296065e-08, 0, 0},
			tau: 0.001181949427158492, box: 200,
		}, exact: true},
	} {
		q := c.q
		if want := oracleRHPRoots(t, q.chi, q.box, q.num, q.den); want != 0 {
			t.Fatalf("%v: oracle %d RHP roots, want 0", q.den, want)
		}
		var supT, supS float64
		for i := range 200001 {
			l := q.L(math.Pow(10, -8+13*float64(i)/200000))
			supT = max(supT, cmplx.Abs(l/(1+l)))
			supS = max(supS, cmplx.Abs(1/(1+l)-0.5))
		}
		p := makePlant(t, q.num, q.den)
		if err := p.SetInputDelay([]float64{q.tau}); err != nil {
			t.Fatal(err)
		}
		if !c.exact {
			l, err := delayLoopFromSystem(p, "delay loop")
			if err != nil {
				t.Fatal(err)
			}
			if w, root, err := l.indent(); err == nil {
				t.Fatalf("%v: indent = %g, %v, nil; want errDelayLoopUnsupported", q.den, w, root)
			}
		}
		cl, err := Feedback(p, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		if stable, err := cl.IsStable(); (err == nil && !stable) || (c.exact && err != nil) {
			t.Fatalf("%v: IsStable = %v, %v; oracle 0 RHP roots", q.den, stable, err)
		}
		norm, _, err := HinfNorm(cl)
		if (err == nil && !(norm >= supT*(1-1e-9) && norm <= supT*(1+1e-3))) || (c.exact && err != nil) {
			t.Fatalf("%v: HinfNorm = %g, %v; dense sup|T| %g", q.den, norm, err, supT)
		}
		dm, err := DiskMargin(p)
		if (err == nil && !(dm.Alpha <= 1/supS*(1+1e-9) && dm.Alpha >= 1/supS*(1-1e-3))) || (c.exact && err != nil) {
			t.Fatalf("%v: DiskMargin = %+v, %v; dense alpha %g", q.den, dm, err, 1/supS)
		}
	}
}
