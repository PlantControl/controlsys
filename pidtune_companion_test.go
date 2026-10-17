package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// The companion realization of −31.29(s+0.09685)/(s(s²+14.84s+367.9)
// (s²+7.963s+36.18))·e^{−0.5042s} has ‖A‖₂ ≈ 1.4e4 against poles of
// modulus ≤ 19.2, so a Markov tail bound in that realization ends the
// linear delay grid near 1.4e4 rad/s (≈ 18k points) although |L| < 0.3
// beyond ~50 rad/s; the balanced realization certifies the same verdicts
// within 4096 points. Oracle: closed-loop roots with the delay replaced by
// Padé approximants of orders 6..10, which must all agree.
func TestPIDTuningDelayStabilityCompanionGridBudget(t *testing.T) {
	const tau = 0.5042
	num := Poly{-31.29, -31.29 * 0.09685}
	den := Poly{1, 0}.Mul(Poly{1, 14.84, 367.9}).Mul(Poly{1, 7.963, 36.18})
	q := makePlant(t, num, den)
	if err := q.SetInputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	if na := spectralNorm(q.A); na < 1e4 {
		t.Fatalf("‖A‖ = %g, want companion-scale ≥ 1e4", na)
	}
	eval, err := newSISOEval(q)
	if err != nil {
		t.Fatal(err)
	}
	for _, ideal := range []bool{true, false} {
		stable, ok, err := pidTuningDelayStabilityBudget(q, eval, 2.2767e-4, ideal, pidTuningMaxTail, 4096)
		if err != nil || !ok {
			t.Fatalf("out of scope: %v", err)
		}
		cands := []PID2{
			{Kp: -1, Kd: 8, Tf: 0.01},
			{Kp: -1, Ki: -0.05, Kd: 8, Tf: 0.01},
			{Kp: 1, Kd: 8, Tf: 0.01},
		}
		if ideal {
			cands = []PID2{
				{Kp: -1, Kd: 8},
				{Kp: -0.5, Kd: 2},
				{Kp: -1, Ki: -0.05, Kd: 8},
				{Kp: -0.98, Kd: 864},
				{Kp: 1, Kd: 8},
			}
		}
		stableSeen := false
		for _, c := range cands {
			want := padeClosedLoopStable(t, num, den, c, tau)
			stableSeen = stableSeen || want
			got, err := stable(&c)
			if err != nil || got != want {
				t.Errorf("%+v: stable=%v err=%v, want %v", c, got, err, want)
			}
		}
		if !stableSeen {
			t.Errorf("ideal=%v: no stable candidate exercised", ideal)
		}
	}
}

// padeClosedLoopStable reports whether 1 + C·num/den·e^{−τs} has all roots
// in the open LHP with the delay replaced by Padé approximants of orders
// 6..10, failing the test when they disagree.
func padeClosedLoopStable(t *testing.T, num, den Poly, c PID2, tau float64) bool {
	t.Helper()
	// C = (Kp·s(Tf·s+1) + Ki(Tf·s+1) + Kd·s²) / (s(Tf·s+1))
	cn := Poly{c.Kp*c.Tf + c.Kd, c.Kp + c.Ki*c.Tf, c.Ki}
	cd := Poly{c.Tf, 1, 0}
	var verdict []bool
	for order := 6; order <= 10; order++ {
		qk := make([]float64, order+1)
		qk[0] = 1
		for k := 1; k <= order; k++ {
			qk[k] = qk[k-1] * float64(order-k+1) / float64(k*(2*order-k+1))
		}
		pn, pd := make(Poly, order+1), make(Poly, order+1)
		for k := range order + 1 {
			v := qk[k] * math.Pow(tau, float64(k))
			pd[order-k] = v
			pn[order-k] = v * math.Pow(-1, float64(k))
		}
		chi := den.Mul(cd).Mul(pd).Add(num.Mul(cn).Mul(pn))
		roots, err := chi.Roots()
		if err != nil {
			t.Fatal(err)
		}
		ok := true
		for _, r := range roots {
			if c.Ki == 0 && cmplx.Abs(r) < 1e-12 {
				continue
			}
			if real(r) >= -1e-9 {
				ok = false
			}
		}
		verdict = append(verdict, ok)
	}
	for _, v := range verdict[1:] {
		if v != verdict[0] {
			t.Fatalf("%+v: Padé verdicts disagree: %v", c, verdict)
		}
	}
	return verdict[0]
}
