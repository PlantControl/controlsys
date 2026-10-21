package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// pidGainsOneSign reports whether the nonzero gains of k share one sign.
func pidGainsOneSign(k PID2) bool {
	s := 0.
	for _, g := range []float64{k.Kp, k.Ki, k.Kd} {
		if g == 0 {
			continue
		}
		if s != 0 && math.Copysign(1, g) != s {
			return false
		}
		s = math.Copysign(1, g)
	}
	return true
}

// pidtuneResidueSign is the sign of the residue N(r)/D'(r) of num/den at
// its simple pole r nearest s = 0 (z = 1), evaluated at that point.
func pidtuneResidueSign(num, den Poly, dt float64) float64 {
	at := 0.
	if dt > 0 {
		at = 1
	}
	d := make(Poly, len(den)-1)
	for i := range d {
		d[i] = den[i] * float64(len(den)-1-i)
	}
	return math.Copysign(1, real(num.Eval(complex(at, 0)))/real(d.Eval(complex(at, 0))))
}

// 6KBA3T: on a plant stable but for an integrator, TunePID accepted gains of
// any sign, as pidTuningDCSign gives such a plant none, so PD on
// 8.2/(z−1) returned Kp 0.0344, Kd −0.0019 (Kp 0.0273, Kd −0.0036 at a 30°
// margin) and on IP3XIC k=10 Kp 0.79, Kd −1.23e5, where MATLAB pidtune and
// Pidtune give same-sign gains. Every gain now takes the sign of the
// integrator's residue. Oracle: that sign from the coefficients, exact
// Schur–Cohn or Routh stability, the reported margin at the reported
// crossover and |S| <= 4 over the band, the loop evaluated directly.
func TestTunePIDIntegratorResidueSign(t *testing.T) {
	skipSlowUnderRace(t)
	ip3 := Poly{1, -5.388323209156541, 12.045405202918547, -14.292531743733242, 9.488502158799133, -3.339413745774066, 0.486361336946173}
	for _, c := range []struct {
		name     string
		num, den Poly
		dt, pm   float64
		fam      PidtuneType
	}{
		{"8.2/(z-1)", Poly{8.2}, Poly{1, -1}, .05, 0, PidtunePD},
		{"8.2/(z-1) pm30", Poly{8.2}, Poly{1, -1}, .05, 30, PidtunePD},
		{"-8.2/(z-1)", Poly{-8.2}, Poly{1, -1}, .05, 0, PidtunePD},
		{"8.2/(z-1) PDF", Poly{8.2}, Poly{1, -1}, .05, 0, PidtunePDF},
		{"68.7/(z-1) pm45", Poly{68.7}, Poly{1, -1}, .05, 45, PidtunePD},
		{"IP3XIC k=10", Poly{2.761969998249568e-15, 2.6332895798497864e-15, 6.196985260193668e-16}, ip3, .01, 0, PidtunePD},
		{"IP3XIC k=100", Poly{2.761969998249568e-14, 2.6332895798497864e-14, 6.196985260193668e-15}, ip3, .01, 0, PidtunePDF},
		{"integrator lag", Poly{-0.17044565115204277}, Poly{1, 15.942187409886401, 8.53689970019146, 0}, 0, 0, PidtunePD},
		{"integrator lag PIDF", Poly{-0.17044565115204277}, Poly{1, 15.942187409886401, 8.53689970019146, 0}, 0, 30, PidtunePIDF},
		{"-2/(s(0.1s+1))", Poly{-2}, Poly{.1, 1, 0}, 0, 45, PidtunePD},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		r, err := TunePID(context.Background(), p, c.fam, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := r.Controller
		want := pidtuneResidueSign(c.num, c.den, c.dt)
		for _, g := range []float64{k.Kp, k.Ki, k.Kd} {
			if g != 0 && math.Copysign(1, g) != want {
				t.Errorf("%s: %+v, want every gain of sign %g", c.name, *k, want)
				break
			}
		}
		pid := &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt, IFormula: k.IFormula, DFormula: k.DFormula}
		if c.dt > 0 && !pidtuneExactSchurStable(t, pid, c.num, c.den, c.dt) || c.dt == 0 && !pidtuneExactHurwitz(pid, c.num, c.den) {
			t.Errorf("%s: %+v unstable", c.name, *k)
		}
		L := pidtuneUndelayedLoop(k, c.num, c.den, c.dt)
		ev := r.Evidence
		l := L(ev.AchievedCrossover)
		if pm := 180 + cmplx.Phase(l)*180/math.Pi; math.Abs(cmplx.Abs(l)-1) > 1e-3 || pm < ev.AchievedPhaseMargin-.5 {
			t.Errorf("%s: |L(%.4g)| %.6g, margin %.4g, reported %.4g", c.name, ev.AchievedCrossover, cmplx.Abs(l), pm, ev.AchievedPhaseMargin)
		}
		if _, ms := pidtuneSignedMargins(L, ev.FrequencyBand[0], ev.FrequencyBand[1]); ms > pidTuningMaxSensitivity*1.01 {
			t.Errorf("%s: %+v |S| %.4g over band %v", c.name, *k, ms, ev.FrequencyBand)
		}
	}
}

// 6KBA3T: the residue sign applies only where every other pole is stable.
// On 1/(s(s−1)) the residue at s = 0 is −1, yet s² + (Kd−1)s + Kp is
// Hurwitz only for Kp > 0, Kd > 1, so the sign stays free there, as on a
// plant without an integrator. A pole rounded off z = 1 (YVAIMU) still
// counts as the integrator.
func TestTunePIDIntegratorSignScope(t *testing.T) {
	for _, c := range []struct {
		name     string
		num, den Poly
		dt       float64
		want     float64
	}{
		{"8.2/(z-1)", Poly{8.2}, Poly{1, -1}, .05, 1},
		{"-8.2/(z-1)", Poly{-8.2}, Poly{1, -1}, .05, -1},
		{"integrator lag", Poly{-0.17044565115204277}, Poly{1, 15.942187409886401, 8.53689970019146, 0}, 0, -1},
		{"1/s^2", Poly{1}, Poly{1, 0, 0}, 0, 1},
		{"YVAIMU", Poly{0.1, 0.05}, Poly{1, -1}.Mul(Poly{1, -0.999}).Mul(Poly{1, -0.999}), .1, 1},
		{"1/(s(s-1))", Poly{1}, Poly{1, -1, 0}, 0, 0},
		{"0.1/((z-1)(z-1.1))", Poly{0.1}, Poly{1, -2.1, 1.1}, .05, 0},
		{"1/(s+1)", Poly{1}, Poly{1, 1}, 0, 0},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		if got := pidTuningIntegratorSign(p, 1); got != c.want {
			t.Errorf("%s: sign %g, want %g", c.name, got, c.want)
		}
	}
}

// 6KBA3T: with the residue sign known, the floor's wrong-sign unstable
// candidates are rejected before the stability check, so on this delayed
// integrating plant the floor goes on to the 90.78°±3° band, where Pidtune
// reports 93° achievable, instead of ending unattainable. Oracle: the loop
// evaluated directly has one crossover, with at least the reported margin,
// and |S| about 1, with gains of the residue's sign.
func TestTunePIDIntegratorResidueSignDelayedFloor(t *testing.T) {
	num := pidPoly(-31.29, []float64{1, .09685})
	den := pidPoly(1, []float64{1, 0}, []float64{1, 14.84, 367.9}, []float64{1, 7.963, 36.18})
	const tau = .5042
	r, err := TunePID(context.Background(),pidPlantDelayed(t, num, den, tau), PidtunePD, PIDTuningOptions{PhaseMargin: 36.78})
	if err != nil {
		t.Fatal(err)
	}
	k := r.Controller
	if k.Kp >= 0 || k.Kd > 0 || k.Ki != 0 {
		t.Fatalf("%+v, want Kp < 0, Kd <= 0", *k)
	}
	L := func(w float64) complex128 {
		s := complex(0, w)
		g := pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*tau)
		return g * (complex(k.Kp, 0) + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
	}
	margins := pidLoopMargins(L, r.Evidence.AchievedCrossover*1e-3, 1e4)
	if len(margins) != 1 || margins[0] < r.Evidence.AchievedPhaseMargin-.5 || margins[0] < 36.78 {
		t.Fatalf("%+v: margins %v, reported %g", *k, margins, r.Evidence.AchievedPhaseMargin)
	}
	if _, ms := pidtuneSignedMargins(L, r.Evidence.AchievedCrossover*1e-3, 1e3); ms > 1.1 {
		t.Errorf("%+v: |S| %g", *k, ms)
	}
}
