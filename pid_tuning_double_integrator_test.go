package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// WDBBJB: free P on double integrators with a lead zero. The phase walk
// from wc0·1e-6 started below −120°, so it never saw the phase rise through
// it, or its floor sample failed at the double pole and the whole call
// returned that error. MATLAB moves wc to the 60° point, where
// atan(w/z)−atan(w/p) = 60°. Oracle: that closed form, Kp = ±1/|G(jwc)|,
// companion roots of den + Kp·num, and dense margins and |S|.
func TestTunePIDFreePDoubleIntegratorMovesToLeadPoint(t *testing.T) {
	r3 := math.Sqrt(3)
	cases := []struct {
		name     string
		num, den []float64
		wc       float64
	}{
		{"3600(s+0.1)/(s²(s+10))", []float64{3600, 360}, []float64{1, 10, 0, 0}, (9.9 + math.Sqrt(9.9*9.9-12)) / (2 * r3)},
		{"−(0.0917s+0.1721)/s²", []float64{-.0917, -.1721}, []float64{1, 0, 0}, .1721 / .0917 * r3},
		{"DSRJ7M −(0.0320s+0.0522)/s²", []float64{-0.032012382534863985, -0.052223340965159556}, []float64{1, 0, 0}, 0.052223340965159556 / 0.032012382534863985 * r3},
	}
	for _, c := range cases {
		r, err := TunePID(context.Background(), makePlant(t, c.num, c.den), PidtuneP, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		g := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s)
		}
		kp := math.Copysign(1/cmplx.Abs(g(c.wc)), c.num[0])
		ctl := *r.Controller
		if ctl.Ki != 0 || ctl.Kd != 0 || math.Abs(ctl.Kp-kp) > 1e-6*math.Abs(kp) {
			t.Fatalf("%s: gains %+v, want Kp %g", c.name, ctl, kp)
		}
		if math.Abs(r.Evidence.AchievedCrossover-c.wc) > 1e-8*c.wc {
			t.Fatalf("%s: crossover %g, want %g", c.name, r.Evidence.AchievedCrossover, c.wc)
		}
		if re := pidClosedLoopMaxRe(c.num, c.den, ctl); re >= 0 {
			t.Fatalf("%s: closed-loop max Re %g", c.name, re)
		}
		L := func(w float64) complex128 { return complex(ctl.Kp, 0) * g(w) }
		margins := pidLoopMargins(L, c.wc*1e-4, c.wc*1e4)
		if len(margins) != 1 || math.Abs(margins[0]-60) > 1e-6 {
			t.Fatalf("%s: dense margins %v", c.name, margins)
		}
		if peak, at := pidDenseSensitivity(ctl, g, c.wc/1e2, c.wc*1e2); peak > 2 {
			t.Fatalf("%s: |S| %g at ω=%g", c.name, peak, at)
		}
	}
}

// Seeded free P on double integrators, bare or with a lag, under a lead
// zero or none: each call returns a stable design with dense |S| ≤ 4 or
// target unattainable. Before the fix a share returned the floor sample's
// nonfinite-response or singular-matrix error.
func TestTunePIDFreePDoubleIntegratorSweep(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(53, 59))
	designs := 0
	for trial := range 300 {
		k := (2*rng.Float64() - 1) * math.Pow(10, 2*rng.Float64()-1)
		num := []float64{k}
		if trial%3 != 0 {
			num = []float64{k, k * math.Pow(10, 2*rng.Float64()-1.5)}
		}
		den := []float64{1, 0, 0}
		if trial%2 == 1 {
			den = []float64{1, math.Pow(10, 2*rng.Float64()-1), 0, 0}
		}
		r, err := TunePID(context.Background(), makePlant(t, num, den), PidtuneP, PIDTuningOptions{})
		if errors.Is(err, ErrPIDTuningTargetUnattainable) {
			continue
		}
		if err != nil {
			t.Errorf("trial %d %v/%v: %v", trial, num, den, err)
			continue
		}
		designs++
		c := *r.Controller
		if re := pidClosedLoopMaxRe(num, den, c); re >= 0 {
			t.Errorf("trial %d %v/%v: Kp %g closed-loop max Re %g", trial, num, den, c.Kp, re)
		}
		g := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(num, s) / pidPolyAt(den, s)
		}
		wc := r.Evidence.AchievedCrossover
		L := func(w float64) complex128 { return complex(c.Kp, 0) * g(w) }
		for w := wc / 1e3; w <= wc*1e3; w *= 1.0005 {
			if s := 1 / cmplx.Abs(1+L(w)); s > 4 {
				t.Errorf("trial %d %v/%v: Kp %g |S| %g at ω=%g", trial, num, den, c.Kp, s, w)
				break
			}
		}
	}
	if designs < 100 {
		t.Fatalf("designs %d", designs)
	}
	t.Logf("designs %d", designs)
}
