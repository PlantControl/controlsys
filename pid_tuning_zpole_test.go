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
	t.Parallel()
	skipSlowUnderRace(t)
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
	t.Parallel()
	skipSlowUnderRace(t)
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

// HXK42S: undelayed integrating plants whose gain crossover sits above a
// lag, where the loop phase is out of reach for PID or PD gains. TunePID
// tried only that crossover and reported the target unattainable, while
// explicit crossovers one to three decades lower have designs and MATLAB
// pidtune returns one. Each design must be closed-loop stable by the exact
// test, meet its reported margin at its reported crossover on the raw
// num/den, and keep |S| <= 4 sampled independently up to Nyquist or 1e3·wc.
func TestTunePIDUndelayedIntegratorBackOff(t *testing.T) {
	skipSlowUnderRace(t)
	cases := []struct {
		name     string
		num, den []float64
		dt       float64
	}{
		{"k/((z-1)^2(z-0.987))", []float64{-0.14994524256120972}, []float64{1, -2.9872237308563525, 2.974447461712705, -0.9872237308563525}, 0.018062614213725733},
		{"k/((z-1)^2 lags)", []float64{2.976756686381004}, []float64{1, -3.737722851489804, 5.228826969943391, -3.2444853854173696, 0.7533812669637826}, 0.041200685092028046},
		{"(0.18s+1.6)/(s^2(s+0.64))", []float64{0.1813581630182136, 1.6005930866011284}, []float64{1, 0.6385258302218032, 0, 0}, 0},
	}
	for _, c := range cases {
		for _, fam := range []PidtuneType{PidtunePID, PidtunePIDF, PidtunePD, PidtunePDF} {
			name := fmt.Sprintf("%s %s", c.name, fam)
			r, err := TunePID(context.Background(), makeDiscretePlant(t, c.num, c.den, c.dt), fam, PIDTuningOptions{})
			if !pidCheckZPoleResult(t, name, c.num, c.den, c.dt, r, err) {
				t.Errorf("%s: no design: %v", name, err)
				continue
			}
			pidCheckUndelayedMarginAndS(t, name, c.num, c.den, c.dt, r)
		}
	}
}

// A free P scan's crossovers at or above Nyquist (8·wc0 on a discrete plant
// whose gain crossover is near Nyquist) failed the call with an
// invalid-argument error; they are skipped.
func TestTunePIDFreePScanSkipsNyquist(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		dt       float64
	}{
		{"k/((z-1)^2(z-0.987))", []float64{-0.14994524256120972}, []float64{1, -2.9872237308563525, 2.974447461712705, -0.9872237308563525}, 0.018062614213725733},
		{"(z-0.97)/((z-1)^2 lags)", []float64{-2.046618214878878, 2.0175518549605798}, []float64{1, -3.4280233577799715, 4.316862995517887, -2.3496559176958596, 0.4608162799579439}, 0.07421327254084888},
	}
	for _, c := range cases {
		r, err := TunePID(context.Background(), makeDiscretePlant(t, c.num, c.den, c.dt), PidtuneP, PIDTuningOptions{})
		pidCheckZPoleResult(t, c.name, c.num, c.den, c.dt, r, err)
	}
}

// pidCheckUndelayedMarginAndS checks r on the raw num/den: margin within
// 0.5° of the reported one and >= 56° at the reported crossover, and sampled
// |S| <= 4 from 1e-3·wc up to Nyquist or 1e3·wc.
func pidCheckUndelayedMarginAndS(t *testing.T, name string, num, den []float64, dt float64, r *PIDTuningResult) {
	t.Helper()
	k := r.Controller
	loop := func(w float64) complex128 {
		x := complex(0, w)
		if dt > 0 {
			x = cmplx.Exp(complex(0, w*dt))
		}
		g := Poly(num).Eval(x) / Poly(den).Eval(x)
		var integ, deriv complex128
		if dt > 0 {
			integ = complex(dt, 0) / (x - 1)
			deriv = 1 / (complex(k.Tf, 0) + complex(dt, 0)/(x-1))
		} else {
			integ = 1 / x
			deriv = x / (complex(k.Tf, 0)*x + 1)
		}
		return g * (complex(k.Kp, 0) + complex(k.Ki, 0)*integ + complex(k.Kd, 0)*deriv)
	}
	wc := r.Evidence.AchievedCrossover
	l := loop(wc)
	pm := 180 + cmplx.Phase(l)*180/math.Pi
	if math.Abs(cmplx.Abs(l)-1) > 1e-3 || math.Abs(math.Remainder(pm-r.Evidence.AchievedPhaseMargin, 360)) > .5 || pm < 56 {
		t.Errorf("%s: |L(%.4g)| = %.6g, margin %.4g, reported %.4g", name, wc, cmplx.Abs(l), pm, r.Evidence.AchievedPhaseMargin)
	}
	hi := 1e3 * wc
	if dt > 0 {
		hi = math.Pi / dt
	}
	peak := 0.
	for w := wc * 1e-3; w < hi; w *= 1.001 {
		peak = math.Max(peak, 1/cmplx.Abs(1+loop(w)))
	}
	if peak > pidTuningMaxSensitivity*1.01 {
		t.Errorf("%s: sampled |S| peak %.4g at wc %.4g", name, peak, wc)
	}
}
