package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// TunePIDFRD on data whose low end shows a flat real DC gain takes the
// plant to have no unstable poles, MATLAB's NumUnstablePoles default for
// frd. A stable plant's closed-loop characteristic polynomial then has the
// sign of 1 + L(0⁺) at s = 0 and of 1 + L(∞) at +∞; opposite signs leave a
// real root in the right half plane. Base returned such designs with
// positive reported margins: Ki opposite the DC gain (PI, PID, PIDF, I),
// or Kp·G(0) < −1 without Ki (PD, PDF), or 1 + Kd·CB < 0 under an ideal
// derivative on relative-degree-one plants, including 1 + Kd·CB just below
// 0 where L(∞) sits next to −1. The fixed search rejects them: a design of
// the right DC sign or an error. Oracle: closed-loop roots of the exact
// characteristic polynomial.
func TestTunePIDFRDStablePlantDCSign(t *testing.T) {
	for i, c := range []struct {
		num, den []float64
		n        int
		family   PidtuneType
		pm       float64
		mustTune bool
	}{
		{[]float64{0.0029356119296049623, 0.008229341811485028}, []float64{1, 0.16531968867170563, 0.059760631262763785, 0.00972338931651889}, 600, PidtunePI, 45, false},
		{[]float64{0.0029356119296049623, 0.008229341811485028}, []float64{1, 0.16531968867170563, 0.059760631262763785, 0.00972338931651889}, 600, PidtunePD, 45, false},
		{[]float64{0.0029356119296049623, 0.008229341811485028}, []float64{1, 0.16531968867170563, 0.059760631262763785, 0.00972338931651889}, 600, PidtunePDF, 45, false},
		{[]float64{0.0029356119296049623, 0.008229341811485028}, []float64{1, 0.16531968867170563, 0.059760631262763785, 0.00972338931651889}, 600, PidtunePID, 45, false},
		{[]float64{0.0029356119296049623, 0.008229341811485028}, []float64{1, 0.16531968867170563, 0.059760631262763785, 0.00972338931651889}, 600, PidtunePIDF, 45, false},
		{[]float64{58.087085573679886, 49.12653285412201}, []float64{1, 0.28773369250817876, 0.011748548943198453}, 600, PidtuneI, 0, false},
		{[]float64{0.16032995067304562}, []float64{1, 0.1797803514992294, 0.1118103357420428}, 600, PidtunePID, 0, true},
		{[]float64{0.577097277112868, 0.4380131583522059}, []float64{1, 0.5364586949033443, 0.08069147255344487, 0.0034546435700922268}, 100, PidtunePIDF, 0, true},
		{[]float64{20.876293292202146}, []float64{1, 0.10946545138033435, 0.056242258056678665}, 600, PidtunePID, 0, true},
		{[]float64{5.998024318626204}, []float64{1, 3.2754824697813687}, 600, PidtunePD, 0, false},
		{[]float64{1.7146661983250227}, []float64{1, 2.927179851338206}, 300, PidtunePD, 45, false},
		{[]float64{-0.4683988494422861}, []float64{1, 0.7032911471041147}, 600, PidtunePD, 60, false},
		{[]float64{-0.4683988494422861}, []float64{1, 0.7032911471041147}, 100, PidtunePD, 60, false},
		{[]float64{0.016084374962596817, 0.009220509586651611}, []float64{1, 1.2321898832719733, 0.257977790095751}, 100, PidtunePD, 60, false},
	} {
		name := fmt.Sprintf("case %d %s", i, c.family)
		frd := pidExactFRD(t, c.num, c.den, 1e-3, 1e3, c.n)
		r, err := TunePIDFRD(context.Background(), frd, c.family, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			if c.mustTune || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		k, ev := *r.Controller, r.Evidence
		if re := pidClosedLoopMaxRe(c.num, c.den, k); re >= 0 {
			t.Errorf("%s: %+v closed-loop root at Re %g, reported margin %g", name, k, re, ev.AchievedPhaseMargin)
		}
		dc := c.num[len(c.num)-1] / c.den[len(c.den)-1]
		if k.Ki*dc < 0 {
			t.Errorf("%s: Ki %g against DC gain %g", name, k.Ki, dc)
		}
		pm := c.pm
		if pm == 0 {
			pm = 60
		}
		// IJCIOS: case 3's chord read 42.0° at a crossing whose fit and the
		// exact plant put at 41.55°, hence the extra half degree.
		if ev.AchievedPhaseMargin < pm-3.5 {
			t.Errorf("%s: achieved margin %g", name, ev.AchievedPhaseMargin)
		}
	}
}

// Designs the DC test must keep: mixed-sign gains that close a stable,
// robust loop. Base returned each; requiring every gain of the DC sign, as
// TunePID does, lost these and about a thousand like them in a 300-plant
// sweep, since TunePIDFRD cannot move the crossover. The ideal-derivative
// PID has 1 + Kd·CB < 0 with Ki < 0 < G(0): both ends of the
// characteristic polynomial negative. Oracles: closed-loop roots and a
// dense |S| sweep of the exact plant.
func TestTunePIDFRDDCSignKeepsMixedSignDesigns(t *testing.T) {
	for i, c := range []struct {
		num, den []float64
		lo, hi   float64
		family   PidtuneType
		pm       float64
	}{
		{[]float64{5.998024318626204}, []float64{1, 3.2754824697813687}, 1e-3, 1e3, PidtunePI, 60},
		{[]float64{41.849356476107744}, []float64{1, 0.13764735513337695}, 1e-3, 1e3, PidtunePD, 60},
		{[]float64{777.739219099569, 24.483829600920053}, []float64{1, 3.8460921567336412, 21.003676318296215}, 1e-3, 1e3, PidtunePID, 45},
		{[]float64{1.475767261587179, -2.761797130009405}, []float64{1, 2.8234906807454356, 6.053557929541098, 12.422570998833214}, .07531, 73.01, PidtunePIDF, 0},
		{[]float64{1.475767261587179, -2.761797130009405}, []float64{1, 2.8234906807454356, 6.053557929541098, 12.422570998833214}, .07531, 73.01, PidtunePDF, 0},
	} {
		name := fmt.Sprintf("case %d %s", i, c.family)
		frd := pidExactFRD(t, c.num, c.den, c.lo, c.hi, 600)
		r, err := TunePIDFRD(context.Background(), frd, c.family, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		k := *r.Controller
		if re := pidClosedLoopMaxRe(c.num, c.den, k); re >= 0 {
			t.Errorf("%s: %+v closed-loop root at Re %g", name, k, re)
		}
		exact := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s)
		}
		if peak, at := pidDenseSensitivity(k, exact, c.lo, c.hi); peak > 4 {
			t.Errorf("%s: %+v |S| %g at ω=%g", name, k, peak, at)
		}
	}
}

// The DC test applies only without known unstable poles and only to data
// whose low end is a flat real gain.
func TestTunePIDFRDDCSignScope(t *testing.T) {
	num, den := []float64{58.087085573679886, 49.12653285412201}, []float64{1, 0.28773369250817876, 0.011748548943198453}
	frd := pidExactFRD(t, num, den, 1e-3, 1e3, 600)
	one := 1
	r, err := TunePIDFRD(context.Background(), frd, PidtuneI, PIDTuningOptions{UnstablePoles: &one})
	if err != nil {
		t.Fatalf("one unstable pole: %v", err)
	}
	if r.Controller.Ki >= 0 {
		t.Errorf("one unstable pole: Ki %g, want the unconstrained negative design", r.Controller.Ki)
	}
	build := func(g func(complex128) complex128, lo, hi float64) *FRD {
		omega := make([]float64, 200)
		values := make([][][]complex128, len(omega))
		for k := range omega {
			omega[k] = lo * math.Pow(hi/lo, float64(k)/float64(len(omega)-1))
			values[k] = [][]complex128{{g(complex(0, omega[k]))}}
		}
		f, err := NewFRD(values, omega, 0)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	for i, c := range []struct {
		g      func(complex128) complex128
		lo, hi float64
		want   float64
	}{
		{func(s complex128) complex128 { return 2 / (s + 1) }, 1e-3, 1e2, 2},
		{func(s complex128) complex128 { return -3 / (s*s + s + 1) }, 1e-3, 1e2, -3},
		{func(s complex128) complex128 { return 1 / (s * (s + 1)) }, 1e-3, 1e2, 0},
		{func(s complex128) complex128 { return s / (s + 1) }, 1e-3, 1e2, 0},
		{func(s complex128) complex128 { return 1 / cmplx.Pow(s+1, 3) }, 3, 1e2, 0},
		{func(s complex128) complex128 { return 1 / ((s + 1) * (s + 1)) }, 10, 1e3, 0},
	} {
		f := build(c.g, c.lo, c.hi)
		p := func(w float64) complex128 { return c.g(complex(0, w)) }
		got := pidTuningFRDDCGain(f, p)
		if c.want == 0 && got != 0 || c.want != 0 && math.Abs(got-c.want) > 1e-2*math.Abs(c.want) {
			t.Errorf("case %d: DC gain %g, want %g", i, got, c.want)
		}
	}
}
