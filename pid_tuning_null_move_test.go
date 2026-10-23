package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// A Tf step on a seed with Kd = 0 is a null move; its score differed from
// the incumbent's by rounding, so a 1e-14 change in one plant coefficient
// let it steer the PIDF floor search into another basin (J 1.14 against
// 1.61; DKOLCU). Designs must not depend on rounding of the coefficients.
// Oracles: Padé-10 closed-loop roots and a dense exact sweep of L.
func TestTunePIDNullTfMoveRoundingInvariant(t *testing.T) {
	num := []float64{1.7707309809714888, 14.680167506003471}
	const tau = 0.019603320901217147
	var ref PID2
	for i, rel := range []float64{0, 1e-14, -1e-14} {
		den := []float64{1, 14.361343001258692, 49.96072517014042, 38.982637653846695}
		den[2] *= 1 + rel
		p := pidPlantDelayed(t, num, den, tau)
		r, err := TunePID(context.Background(), p, PidtunePIDF, PIDTuningOptions{PhaseMargin: 30})
		if err != nil {
			t.Fatalf("den[2]·(1%+g): %v", rel, err)
		}
		k := *r.Controller
		if i == 0 {
			ref = k
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
					t.Errorf("%+v: closed-loop root %v", k, q)
				}
			}
			L := func(w float64) complex128 {
				s := complex(0, w)
				return pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*tau) * pidTuningFeedback(k, w)
			}
			wc := r.Evidence.AchievedCrossover
			margin, peak := pidDenseLoop(L, wc/1e3, wc*1e4)
			if margin < 27-1e-6 || peak > 4 {
				t.Errorf("%+v: dense margin %g, max|S| %g", k, margin, peak)
			}
			continue
		}
		for _, g := range [][2]float64{{k.Kp, ref.Kp}, {k.Ki, ref.Ki}, {k.Kd, ref.Kd}, {k.Tf, ref.Tf}} {
			if math.Abs(g[0]-g[1]) > 1e-6*math.Max(math.Abs(g[1]), 1e-3) {
				t.Fatalf("den[2]·(1%+g): %+v, unperturbed %+v", rel, k, ref)
			}
		}
	}
}
