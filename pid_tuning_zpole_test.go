package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

func makeDiscretePlant(t *testing.T, num, den []float64, dt float64) *System {
	t.Helper()
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}
	res, err := tf.StateSpace()
	if err != nil {
		t.Fatalf("StateSpace: %v", err)
	}
	return res.Sys
}

// pidCheckZPoleResult accepts a design or target unattainable; a design's
// closed loop must be stable by the exact Schur–Cohn (discrete) or
// companion-root (continuous) test on num/den.
func pidCheckZPoleResult(t *testing.T, name string, num, den []float64, dt float64, r *PIDTuningResult, err error) bool {
	t.Helper()
	if errors.Is(err, ErrPIDTuningTargetUnattainable) {
		return false
	}
	if err != nil {
		t.Errorf("%s: %v", name, err)
		return false
	}
	c := r.Controller
	if dt > 0 {
		pid := &PID{Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf, Dt: c.Dt, IFormula: c.IFormula, DFormula: c.DFormula}
		if !pidtuneExactSchurStable(t, pid, num, den, dt) {
			t.Errorf("%s: unstable design %+v", name, *c)
		}
	} else if re := pidClosedLoopMaxRe(num, den, *c); re >= 0 {
		t.Errorf("%s: closed-loop max Re %g for %+v", name, re, *c)
	}
	return true
}

// KMOSVP: plants with rounded multiple poles at z = 1 or s = 0. The floor's
// crossover grid reached under the band to the rounded poles' spread, and
// the PI phase move walked down to wc0/1e4, where the point solve is
// singular; a free P scan's band met +Inf there. One failed sample aborted
// the call with a solver error. MATLAB pidtune returns a design or reports
// the target unattainable.
func TestTunePIDUnitPoleClusterNoSolverError(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		dt       float64
		fam      PidtuneType
	}{
		{"k/(z-1)^3 PI", []float64{-0.015511622455509733}, []float64{1, -3, 3, -1}, 0.19047714028975662, PidtunePI},
		{"(z-1)^3(z-0.72) PID", []float64{-2.3897487927970316, -1.1649076634460165, 1.1571601778487086}, []float64{1, -3.7187190663923624, 5.156157199177088, -3.156157199177087, 0.7187190663923623}, 0.016181320226854916, PidtunePID},
		{"antipalindromic z=1 PID", []float64{-86.58188789608674, -86.54930732135502}, []float64{1, -1.418487078684577, -0.7445387639462693, 0.7445387639462693, 1.418487078684577, -1}, 0.020779089557500845, PidtunePID},
		{"(z-1)^3 lags PI move", []float64{-2.0709513520973, 2.0019010642668236}, []float64{1, -4.863096899894288, 9.4547589026839, -9.185695308685977, 4.459501508897403, -0.8654682030010385}, 0.06496452636120702, PidtunePI},
		{"s^3(s+2.17) P", []float64{0.054868801144111294, 0.005987777486767363}, []float64{1, 2.168636082879029, 0, 0, 0}, 0, PidtuneP},
		{"s^3(s+2.17) PID", []float64{0.054868801144111294, 0.005987777486767363}, []float64{1, 2.168636082879029, 0, 0, 0}, 0, PidtunePID},
		{"s^3(s+2.17) PD", []float64{0.054868801144111294, 0.005987777486767363}, []float64{1, 2.168636082879029, 0, 0, 0}, 0, PidtunePD},
	}
	for _, c := range cases {
		r, err := TunePID(context.Background(), makeDiscretePlant(t, c.num, c.den, c.dt), c.fam, PIDTuningOptions{})
		pidCheckZPoleResult(t, c.name, c.num, c.den, c.dt, r, err)
	}
}

// Seeded discrete plants with a double or triple pole at z = 1, lags and a
// zero: every call returns a design, closed-loop stable by the exact
// Schur–Cohn test, or target unattainable. Before the fix about a third
// returned a singular-matrix error.
func TestTunePIDUnitPoleClusterSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(71, 79))
	fams := []PidtuneType{PidtunePI, PidtunePID, PidtunePIDF, PidtunePD}
	designs := 0
	for trial := range 60 {
		dt := math.Pow(10, -2+1.5*rng.Float64())
		den := Poly{1}
		for range 2 + rng.IntN(2) {
			den = den.Mul(Poly{1, -1})
		}
		for range rng.IntN(3) {
			den = den.Mul(Poly{1, -math.Exp(-math.Pow(10, 2*rng.Float64()-1) * dt)})
		}
		num := Poly{(2*rng.Float64() - 1) * math.Pow(10, 2*rng.Float64()-1)}
		if rng.IntN(2) == 0 {
			num = num.Mul(Poly{1, -math.Exp(-math.Pow(10, 2*rng.Float64()-1) * dt)})
		}
		// |G| > 1 at Nyquist leaves no gain crossover, which 6RFH5N covers.
		if cmplx.Abs(num.Eval(-1)/den.Eval(-1)) >= 1 {
			continue
		}
		fam := fams[trial%len(fams)]
		r, err := TunePID(context.Background(), makeDiscretePlant(t, num, den, dt), fam, PIDTuningOptions{})
		if pidCheckZPoleResult(t, fmt.Sprintf("trial %d %s", trial, fam), num, den, dt, r, err) {
			designs++
		}
	}
	t.Logf("%d designs", designs)
}
