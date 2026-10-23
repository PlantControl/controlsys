package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// A Tf step on a seed with Kd = 0 is a null move; its score differed from
// the incumbent's by rounding, so a 1e-14 change in one plant coefficient
// steered the PIDF search into one basin or another (DKOLCU: J 1.14 or
// 1.61). Skipping the move made the design deterministic but stranded every
// such search at the worse basin, as Kd steps at the seed's Tf left the
// phase band or crossed 0 dB again below the floor (J 1.61, 10.3 against
// base 1.30, 1.15 and 1.085 against 1.10 and 1.004). The Tf axis now steps
// Kd with it from Kd = 0. Designs must not depend on rounding of the
// coefficients and must reach the better basin. Oracles on the raw
// coefficients: closed-loop roots (Padé-10 for the delay), the winding of
// 1 + L, and a dense sweep of every 0 dB crossing and of |S|.
func TestTunePIDNullTfMoveRoundingInvariant(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		tau, dt  float64
		pm, maxJ float64
	}{
		{"DKOLCU delayed", []float64{1.7707309809714888, 14.680167506003471}, []float64{1, 14.361343001258692, 49.96072517014042, 38.982637653846695}, 0.019603320901217147, 0, 30, 1.2},
		{"#1449", []float64{48.06191588347829, 0.13735368706155338, 0.002249656527236773}, []float64{1, 0.32198253945344046, 0.18444876027906507, 0.05272355937125802}, 0, 0, 45, 1.31},
		{"#70 discrete", []float64{-0.3047465759638495}, []float64{1, -2.4260740657587038e-12}, 0, 1.13124, 60, 1.09},
		{"#723 discrete", []float64{2.702596709175337}, []float64{1, -0.062364934839474126}, 0, 0.00106787, 60, 1.03},
	}
	for _, c := range cases {
		var ref PID2
		for i, rel := range []float64{0, 1e-14, -1e-14} {
			den := append([]float64(nil), c.den...)
			den[1] *= 1 + rel
			if rel != 0 && den[1] == c.den[1] {
				den[1] += rel
			}
			var p *System
			if c.dt > 0 {
				p = makeDiscretePlant(t, c.num, den, c.dt)
			} else {
				p = pidPlantDelayed(t, c.num, den, c.tau)
			}
			r, err := TunePID(context.Background(), p, PidtunePIDF, PIDTuningOptions{PhaseMargin: c.pm})
			if err != nil {
				t.Fatalf("%s den[1]·(1%+g): %v", c.name, rel, err)
			}
			k := *r.Controller
			if i > 0 {
				for _, g := range [][2]float64{{k.Kp, ref.Kp}, {k.Ki, ref.Ki}, {k.Kd, ref.Kd}, {k.Tf, ref.Tf}} {
					if math.Abs(g[0]-g[1]) > 1e-6*math.Max(math.Abs(g[1]), 1e-3) {
						t.Fatalf("%s den[1]·(1%+g): %+v, unperturbed %+v", c.name, rel, k, ref)
					}
				}
				continue
			}
			ref = k
			if r.Evidence.Objective > c.maxJ {
				t.Errorf("%s: %+v J %g, want <= %g", c.name, k, r.Evidence.Objective, c.maxJ)
			}
			pidNullMoveOracle(t, c.name, c.num, c.den, c.tau, c.dt, p, k, r.Evidence.AchievedCrossover, c.pm-3)
		}
	}
}

// pidNullMoveOracle checks k on the raw coefficients: a stable closed loop,
// every 0 dB crossing at least floor from −1 and |S| <= 4.
func pidNullMoveOracle(t *testing.T, name string, num, den []float64, tau, dt float64, p *System, k PID2, wc, floor float64) {
	t.Helper()
	var L func(float64) complex128
	lo, hi := wc/1e4, wc*1e4
	switch {
	case dt > 0:
		L = func(w float64) complex128 {
			z := cmplx.Exp(complex(0, w*dt))
			c := complex(k.Kp, 0) + complex(k.Ki*dt, 0)/(z-1) + complex(k.Kd, 0)*(z-1)/(complex(k.Tf, 0)*(z-1)+complex(dt, 0))
			return c * pidPolyAt(num, z) / pidPolyAt(den, z)
		}
		hi = math.Pi / dt
		// (z−1)(Tf(z−1)+dt)·den + [Kp(z−1)(Tf(z−1)+dt) + Ki·dt(Tf(z−1)+dt) + Kd(z−1)²]·num.
		zm1, f := []float64{1, -1}, []float64{k.Tf, dt - k.Tf}
		cn := Poly(polyMulTest(zm1, f)).Scale(k.Kp).Add(Poly(f).Scale(k.Ki * dt)).Add(Poly(polyMulTest(zm1, zm1)).Scale(k.Kd))
		roots, err := Poly(polyMulTest(polyMulTest(zm1, f), den)).Add(Poly(polyMulTest(cn, num))).Roots()
		if err != nil {
			t.Fatal(err)
		}
		for _, q := range roots {
			if cmplx.Abs(q) >= 1 {
				t.Errorf("%s: %+v closed-loop root %v", name, k, q)
			}
		}
	case tau > 0:
		L = func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*complex(tau, 0)) * pidTuningFeedback(k, w)
		}
		pp, err := p.Pade(10)
		if err != nil {
			t.Fatal(err)
		}
		cs, err := mustPID(t, k.Kp, k.Ki, k.Kd, k.Tf, 0).System()
		if err != nil {
			t.Fatal(err)
		}
		l, err := Series(cs, pp)
		if err != nil {
			t.Fatal(err)
		}
		cl, err := Feedback(l, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		poles, err := cl.Poles()
		if err != nil {
			t.Fatal(err)
		}
		for _, q := range poles {
			if real(q) >= 0 {
				t.Errorf("%s: %+v closed-loop root %v", name, k, q)
			}
		}
	default:
		L = func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(num, s) / pidPolyAt(den, s) * pidTuningFeedback(k, w)
		}
		if re := pidClosedLoopMaxRe(num, den, k); re >= 0 {
			t.Errorf("%s: %+v closed-loop root at Re %g", name, k, re)
		}
	}
	if dt == 0 {
		if wind := pidWindingDeg(L, 1e-7*wc, 1e12*wc); math.Abs(wind-90) > 1 {
			t.Errorf("%s: %+v arg(1+L) changes by %g°, want 90° for a stable loop", name, k, wind)
		}
	}
	for _, m := range pidLoopMargins(L, lo, hi) {
		if math.Min(m, 360-m) < floor-1e-6 {
			t.Errorf("%s: %+v crossing %g° from −1, floor %g", name, k, math.Min(m, 360-m), floor)
		}
	}
	const n = 1000000
	peak := 0.
	for i := 0; i <= n; i++ {
		peak = math.Max(peak, 1/cmplx.Abs(1+L(lo*math.Pow(hi/lo, float64(i)/n))))
	}
	if peak > 4*(1+1e-4) {
		t.Errorf("%s: %+v max|S| %g", name, k, peak)
	}
}
