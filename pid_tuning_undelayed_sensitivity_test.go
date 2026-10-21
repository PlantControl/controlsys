package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"strings"
	"testing"
)

// SIVH4J: a default-crossover TunePID design on an undelayed plant bounded
// |S| only on its one-gain path, so PD on k/(z−1) at a 30° margin returned
// |S| 6.2 and PI on 78/(z−0.24) |S| 6.8 inside the reported band. Every
// family now keeps |S| <= 4 there, as on a delayed plant, or fails with the
// delayed path's error. Oracle: the loop evaluated directly from the
// coefficients, closed-loop stability by exact Schur–Cohn or Routh, and
// the reported margin at the reported crossover.
func TestTunePIDUndelayedSensitivityBound(t *testing.T) {
	skipSlowUnderRace(t)
	for _, c := range []struct {
		name     string
		num, den Poly
		dt, pm   float64
		fam      PidtuneType
	}{
		{"integrator", Poly{8.2}, Poly{1, -1}, .05, 30, PidtunePD},
		{"integrator k69", Poly{68.7}, Poly{1, -1}, .05, 30, PidtunePD},
		{"first order", Poly{78.16154939112381}, Poly{1, -0.24048014891552025}, .05, 0, PidtunePI},
		{"triple near z=1", Poly{25.736941264909365, -12.025583802840814}, Poly{1, -2.9940148684730055, 2.9906760883456167, -0.9966552039257759}, .05, 0, PidtunePID},
		{"resonant", Poly{-0.0100124494824873, 0.02492664308083451}, Poly{1, -2.304046118930773, 2.2422214660257955, -0.9381753470950226}, .05, 30, PidtunePD},
		{"integrator lag", Poly{-0.17044565115204277}, Poly{1, 15.942187409886401, 8.53689970019146, 0}, 0, 30, PidtunePD},
		{"nmp zero PD", Poly{0.021519322555811803, -0.001627398735015024}, Poly{1, 0.24582060311366233, 0.01283516034153513}, 0, 30, PidtunePD},
		{"nmp zero PIDF", Poly{0.021519322555811803, -0.001627398735015024}, Poly{1, 0.24582060311366233, 0.01283516034153513}, 0, 30, PidtunePIDF},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		r, err := TunePID(context.Background(), p, c.fam, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := r.Controller
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
		pms, ms := pidtuneSignedMargins(L, ev.FrequencyBand[0], ev.FrequencyBand[1])
		for _, m := range pms {
			if math.Abs(m) < ev.AchievedPhaseMargin-.5 {
				t.Errorf("%s: crossover margin %.4g below reported %.4g", c.name, m, ev.AchievedPhaseMargin)
			}
		}
		if ms > pidTuningMaxSensitivity*1.01 {
			t.Errorf("%s: %+v |S| %.4g over band %v", c.name, *k, ms, ev.FrequencyBand)
		}
	}
}

// SIVH4J: where no undelayed design keeps |S| <= 4, TunePID fails as on a
// delayed plant: P on this plant at a 30° margin returned |S| 23, and PI on
// the continuous one |S| 9.9.
func TestTunePIDUndelayedSensitivityUnattainable(t *testing.T) {
	skipSlowUnderRace(t)
	for _, c := range []struct {
		name     string
		num, den Poly
		dt       float64
		fam      PidtuneType
	}{
		{"discrete P", Poly{0.2575090968157481, -0.2861993151515084, 0.07878949005677396}, Poly{1, -1.1528379367345725, 0.41958793173297826, -0.049000874014013845}, .05, PidtuneP},
		{"continuous PI", Poly{21.08078684237252, -19.832145026736544, -24.127473451619323, -6.775484440478227, -0.5685377599277538}, Poly{1, 27.732308948783388, 311.1931073410274, 1797.863806649219, 5404.942439434647, 6700.26711372442}, 0, PidtunePI},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		r, err := TunePID(context.Background(), p, c.fam, PIDTuningOptions{PhaseMargin: 30})
		if err == nil {
			_, ms := pidtuneSignedMargins(pidtuneUndelayedLoop(r.Controller, c.num, c.den, c.dt), r.Evidence.FrequencyBand[0], r.Evidence.FrequencyBand[1])
			t.Errorf("%s: %+v |S| %.4g, want error", c.name, *r.Controller, ms)
			continue
		}
		te, ok := errors.AsType[*PIDTuningError](err)
		if !ok || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Errorf("%s: %v, want unattainable *PIDTuningError", c.name, err)
			continue
		}
		want := fmt.Sprintf("sensitivity peak above %d", pidTuningMaxSensitivity)
		if !slices.ContainsFunc(te.Evidence.Warnings, func(w string) bool { return strings.Contains(w, want) }) {
			t.Errorf("%s: warnings %q lack %q", c.name, te.Evidence.Warnings, want)
		}
	}
}

// pidtuneUndelayedLoop returns L(jω) of the forward-Euler PID k on num/den,
// the plant evaluated directly from its coefficients, a discrete one in
// 256-bit arithmetic.
func pidtuneUndelayedLoop(k *PID2, num, den Poly, dt float64) func(float64) complex128 {
	return func(w float64) complex128 {
		if dt > 0 {
			h := math.Sin(w * dt / 2)
			zm1 := complex(-2*h*h, math.Sin(w*dt))
			c := complex(k.Kp, 0) + complex(k.Ki*dt, 0)/zm1
			if k.Kd != 0 {
				c += complex(k.Kd, 0) / (complex(k.Tf, 0) + complex(dt, 0)/zm1)
			}
			return c * pidtuneBigEval(num, 1+zm1) / pidtuneBigEval(den, 1+zm1)
		}
		s := complex(0, w)
		c := complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(complex(k.Tf, 0)*s+1)
		return c * num.Eval(s) / den.Eval(s)
	}
}
