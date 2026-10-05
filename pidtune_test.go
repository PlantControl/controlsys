package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func pidOpenLoop(t *testing.T, plant *System, pid *PID) *System {
	t.Helper()
	csys, err := pid.System()
	if err != nil {
		t.Fatalf("PID.System: %v", err)
	}
	ol, err := Series(csys, plant)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	return ol
}

func pidClosedLoop(t *testing.T, plant *System, pid *PID) *System {
	t.Helper()
	csys, err := pid.System()
	if err != nil {
		t.Fatalf("PID.System: %v", err)
	}
	ol, err := Series(csys, plant)
	if err != nil {
		t.Fatalf("Series: %v", err)
	}
	cl, err := Feedback(ol, nil, -1)
	if err != nil {
		t.Fatalf("Feedback: %v", err)
	}
	return cl
}

func assertStable(t *testing.T, sys *System, label string) {
	t.Helper()
	stable, err := sys.IsStable()
	if err != nil {
		t.Fatalf("%s: IsStable error: %v", label, err)
	}
	if !stable {
		poles, _ := sys.Poles()
		t.Fatalf("%s: closed loop is not stable, poles: %v", label, poles)
	}
}

func assertPhaseMargin(t *testing.T, ol *System, target, tol float64, label string) {
	t.Helper()
	mr, err := Margin(ol)
	if err != nil {
		t.Fatalf("%s: Margin error: %v", label, err)
	}
	if math.IsInf(mr.PhaseMargin, 1) {
		return
	}
	if math.Abs(mr.PhaseMargin-target) > tol {
		t.Errorf("%s: phase margin = %.1f°, want %.0f° ± %.0f°", label, mr.PhaseMargin, target, tol)
	}
}

func assertFiniteGains(t *testing.T, pid *PID, label string) {
	t.Helper()
	if math.IsNaN(pid.Kp) || math.IsInf(pid.Kp, 0) {
		t.Fatalf("%s: Kp = %g", label, pid.Kp)
	}
	if math.IsNaN(pid.Ki) || math.IsInf(pid.Ki, 0) {
		t.Fatalf("%s: Ki = %g", label, pid.Ki)
	}
	if math.IsNaN(pid.Kd) || math.IsInf(pid.Kd, 0) {
		t.Fatalf("%s: Kd = %g", label, pid.Kd)
	}
}

func makePlant(t *testing.T, num, den []float64) *System {
	t.Helper()
	tf := &TransferFunc{
		Num: [][][]float64{{num}},
		Den: [][]float64{den},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatalf("StateSpace: %v", err)
	}
	return res.Sys
}

func TestPidtune_PI_FirstOrder(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 1})
	pid, err := Pidtune(plant, PidtunePI)
	if err != nil {
		t.Fatal(err)
	}
	assertFiniteGains(t, pid, "PI/1st")
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "PI/1st")
	ol := pidOpenLoop(t, plant, pid)
	assertPhaseMargin(t, ol, 60, 5, "PI/1st")
}

func TestPidtune_PID_SecondOrder(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 2, 1})
	pid, err := Pidtune(plant, PidtunePID)
	if err != nil {
		t.Fatal(err)
	}
	assertFiniteGains(t, pid, "PID/2nd")
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "PID/2nd")
	ol := pidOpenLoop(t, plant, pid)
	assertPhaseMargin(t, ol, 60, 5, "PID/2nd")
}

func TestPidtune_PI_Integrator(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 0})
	pid, err := Pidtune(plant, PidtunePI)
	if err != nil {
		t.Fatal(err)
	}
	assertFiniteGains(t, pid, "PI/int")
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "PI/int")
}

func TestPidtune_PI_Unstable(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, -1})
	pid, err := Pidtune(plant, PidtunePI)
	if err != nil {
		t.Fatal(err)
	}
	assertFiniteGains(t, pid, "PI/unstable")
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "PI/unstable")
}

func TestPidtune_PI_CustomCrossover(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 1})
	pid, err := Pidtune(plant, PidtunePI, PidtuneOptions{CrossoverFrequency: 10})
	if err != nil {
		t.Fatal(err)
	}
	assertFiniteGains(t, pid, "PI/wc10")

	ol := pidOpenLoop(t, plant, pid)
	mr, err := AllMargin(ol)
	if err != nil {
		t.Fatal(err)
	}
	if len(mr.GainCrossFreqs) == 0 {
		t.Fatal("no gain crossover found")
	}
	gotWc := mr.GainCrossFreqs[0]
	if math.Abs(gotWc-10)/10 > 0.1 {
		t.Errorf("crossover = %.2f, want ≈10", gotWc)
	}
}

func TestPidtune_P(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 1})
	pid, err := Pidtune(plant, PidtuneP)
	if err != nil {
		t.Fatal(err)
	}
	if pid.Kp <= 0 {
		t.Fatalf("Kp = %g, want > 0", pid.Kp)
	}
	if pid.Ki != 0 || pid.Kd != 0 {
		t.Fatalf("P type should have Ki=Kd=0, got Ki=%g Kd=%g", pid.Ki, pid.Kd)
	}
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "P")
}

func TestPidtune_I(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 1})
	pid, err := Pidtune(plant, PidtuneI)
	if err != nil {
		t.Fatal(err)
	}
	if pid.Ki <= 0 {
		t.Fatalf("Ki = %g, want > 0", pid.Ki)
	}
	if pid.Kp != 0 || pid.Kd != 0 {
		t.Fatalf("I type should have Kp=Kd=0, got Kp=%g Kd=%g", pid.Kp, pid.Kd)
	}
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "I")
}

func TestPidtune_PD(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 2, 1})
	pid, err := Pidtune(plant, PidtunePD)
	if err != nil {
		t.Fatal(err)
	}
	assertFiniteGains(t, pid, "PD")
	if pid.Ki != 0 {
		t.Fatalf("PD type should have Ki=0, got %g", pid.Ki)
	}
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "PD")
}

func TestPidtune_PIDF(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 2, 1})
	pid, err := Pidtune(plant, PidtunePIDF)
	if err != nil {
		t.Fatal(err)
	}
	assertFiniteGains(t, pid, "PIDF")
	if pid.Tf <= 0 {
		t.Fatalf("PIDF should have Tf > 0, got %g", pid.Tf)
	}
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "PIDF")
	ol := pidOpenLoop(t, plant, pid)
	assertPhaseMargin(t, ol, 60, 5, "PIDF")
}

func TestPidtune_InvalidType(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 1})
	_, err := Pidtune(plant, PidtuneType("FOO"))
	if err == nil {
		t.Fatal("expected error for invalid type")
	}
}

func TestPidtune_MIMO_Rejected(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1}, {0}}, {{0}, {1}}},
		Den: [][]float64{{1, 1}, {1, 1}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	_, err = Pidtune(res.Sys, PidtunePI)
	if err == nil {
		t.Fatal("expected error for MIMO plant")
	}
}

func TestPidtune_CustomPM(t *testing.T) {
	plant := makePlant(t, []float64{1}, []float64{1, 1})
	pid, err := Pidtune(plant, PidtunePI, PidtuneOptions{PhaseMargin: 45})
	if err != nil {
		t.Fatal(err)
	}
	cl := pidClosedLoop(t, plant, pid)
	assertStable(t, cl, "PI/pm45")
	ol := pidOpenLoop(t, plant, pid)
	assertPhaseMargin(t, ol, 45, 5, "PI/pm45")
}

func pidtuneControllerOracle(pid *PID, w float64) complex128 {
	if pid.Dt == 0 {
		s := complex(0, w)
		return complex(pid.Kp, 0) + complex(pid.Ki, 0)/s + complex(pid.Kd, 0)*s/(complex(pid.Tf, 0)*s+1)
	}
	z := cmplx.Exp(complex(0, w*pid.Dt))
	integ := func(f PIDFormula) complex128 {
		ts := complex(pid.Dt, 0)
		switch f {
		case BackwardEuler:
			return ts * z / (z - 1)
		case Trapezoidal:
			return ts / 2 * (z + 1) / (z - 1)
		}
		return ts / (z - 1)
	}
	return complex(pid.Kp, 0) + complex(pid.Ki, 0)*integ(pid.IFormula) + complex(pid.Kd, 0)/(complex(pid.Tf, 0)+integ(pid.DFormula))
}

func TestPidtuneMeetsTargetForContinuousAndDiscretePlants(t *testing.T) {
	cont, err := New(
		mat.NewDense(3, 3, []float64{-1, 0.5, 0, 0, -2, 1, 0.3, 0, -3}),
		mat.NewDense(3, 1, []float64{0, 0.4, 1}),
		mat.NewDense(1, 3, []float64{6, -1, 0.5}),
		mat.NewDense(1, 1, []float64{0.05}), 0)
	if err != nil {
		t.Fatal(err)
	}
	disc, err := cont.DiscretizeZOH(0.1)
	if err != nil {
		t.Fatal(err)
	}
	cases := []struct {
		types  []PidtuneType
		wc, pm float64
	}{
		{[]PidtuneType{PidtunePI, PidtunePID}, 0.8, 60},
		{[]PidtuneType{PidtunePDF, PidtunePID, PidtunePIDF}, 2.5, 60},
	}
	for _, plant := range []*System{cont, disc} {
		for _, tc := range cases {
			for _, typ := range tc.types {
				pid, err := Pidtune(plant, typ, PidtuneOptions{CrossoverFrequency: tc.wc, PhaseMargin: tc.pm})
				if err != nil {
					t.Fatalf("dt=%g %s: %v", plant.Dt, typ, err)
				}
				s := complex(0, tc.wc)
				if plant.Dt > 0 {
					s = cmplx.Exp(complex(0, tc.wc*plant.Dt))
				}
				c := pidtuneControllerOracle(pid, tc.wc)
				if plant.Dt > 0 || pid.Tf > 0 || pid.Kd == 0 {
					csys, err := pid.System()
					if err != nil {
						t.Fatalf("dt=%g %s: System: %v", plant.Dt, typ, err)
					}
					if got := evalSS(csys, s); cmplx.Abs(got-c) > 1e-9*cmplx.Abs(c) {
						t.Fatalf("dt=%g %s: realized C=%v, oracle %v", plant.Dt, typ, got, c)
					}
				}
				l := c * evalSS(plant, s)
				pm := 180 + cmplx.Phase(l)*180/math.Pi
				if math.Abs(cmplx.Abs(l)-1) > 1e-9 || math.Abs(pm-tc.pm) > 1e-7 {
					t.Errorf("dt=%g %s: |L(jwc)|=%.12g PM=%.12g, want 1 and %g", plant.Dt, typ, cmplx.Abs(l), pm, tc.pm)
				}
				if typ == PidtunePID && tc.wc == 2.5 && pid.Kd <= 0 {
					t.Errorf("dt=%g PID lead case: Kd=%g, want derivative action", plant.Dt, pid.Kd)
				}
			}
		}
	}
}

func TestPidtuneRejectsDiscreteCrossoverAboveNyquist(t *testing.T) {
	plant, err := New(mat.NewDense(1, 1, []float64{0.9}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Pidtune(plant, PidtunePI, PidtuneOptions{CrossoverFrequency: math.Pi / 0.1}); err == nil {
		t.Fatal("expected error for crossover at Nyquist")
	}
}

func TestFindCrossoverFreq_NarrowResonance(t *testing.T) {
	plant, want := narrowResonance(t)
	wc, err := findCrossoverFreq(plant, 0)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(wc-want[0]) > 1e-12 {
		t.Errorf("crossover %.15g, want %.15g", wc, want[0])
	}
}
