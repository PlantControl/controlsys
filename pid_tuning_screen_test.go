package controlsys

import (
	"errors"
	"math"
	"testing"
)

// LHSHE7: double integrator with a 2-sample input delay, whose absorbed
// I loop has order 11.
func lhshe7Plant(t *testing.T) (p *System, num, den Poly) {
	t.Helper()
	num = Poly{-7.936740333052962e-05, -3.4050281418075133e-06}
	den = Poly{1, -6.3637121030117, 17.308167433139545, -26.077259170711883, 23.501906414802853, -12.668252277306863, 3.781355001756451, -0.4823028457659459, 9.754709754416954e-05}
	p, _ = negatedPlant(t, num, den, 0.1)
	if err := p.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	return p, num, den.Mul(Poly{1, 0, 0})
}

// TAUEF7, 5BV36Q, LHSHE7: designs the screen accepted although the closed
// loop is unstable. A high-gain loop's slow pole beside a fast one carries
// float errors ~ε‖A‖ far above its margin; the delayed I loop's order
// exceeded the exact test's cap. Oracle: exact Routh / Schur–Cohn.
func TestPIDStabilityScreenRejectsUnstable(t *testing.T) {
	type tc struct {
		name     string
		num, den Poly
		dt       float64
		c        PID2
	}
	lp, lnum, lden := lhshe7Plant(t)
	for _, c := range []tc{
		{"TAUEF7", Poly{0.3027244474952484, 2.0990193390429395e-12}, Poly{1, 0.6645618433030911, -0.4408574211719467}, 0, PID2{Kp: 1.6683290649636432e+11}},
		{"5BV36Q", Poly{0.26164438226170905, -4.191600328850453e-09}, Poly{1, -2.200740191646332, -12.33524019209764}, 0, PID2{Kp: 2.924461515036254e+09, Kd: 0.19292965714392324, Tf: 1.298735885615212e-10}},
		{"LHSHE7", lnum, lden, 0.1, PID2{Ki: 7.851150253966971e-11, Dt: 0.1}},
	} {
		p := lp
		if c.dt == 0 {
			p, _ = negatedPlant(t, c.num, c.den, 0)
		}
		pid := &PID{Kp: c.c.Kp, Ki: c.c.Ki, Kd: c.c.Kd, Tf: c.c.Tf, Dt: c.c.Dt}
		if c.dt == 0 && pidtuneExactHurwitz(pid, c.num, c.den) || c.dt != 0 && pidtuneExactSchur(pid, c.num, c.den, c.dt) {
			t.Fatalf("%s: oracle says stable", c.name)
		}
		screen, err := newPIDStabilityScreen(p, nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		if got, err := screen.closedLoopPoles(&c.c); err != nil || got {
			t.Errorf("%s: screen %v, %v", c.name, got, err)
		}
		if loop := newPidtuneLoop(screen.model); loop != nil {
			if got, err := loop.stable(screen, &c.c); err != nil || got {
				t.Errorf("%s: pidtuneLoop %v, %v", c.name, got, err)
			}
		}
	}
}

// Every Pidtune design on the TAUEF7, 5BV36Q and LHSHE7 plants must be
// stable by the exact test or fail with ErrPIDTuningTargetUnattainable.
func TestPidtuneBadlyScaledLoopsStable(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
	}{
		{Poly{0.3027244474952484, 2.0990193390429395e-12}, Poly{1, 0.6645618433030911, -0.4408574211719467}},
		{Poly{0.26164438226170905, -4.191600328850453e-09}, Poly{1, -2.200740191646332, -12.33524019209764}},
	} {
		p, _ := negatedPlant(t, c.num, c.den, 0)
		for _, f := range pidtuneAllTypes {
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
					t.Errorf("%v %s: %v", c.num, f, err)
				}
				continue
			}
			if pid.Kd != 0 && pid.Tf == 0 {
				continue
			}
			if !pidtuneExactHurwitz(pid, c.num, c.den) {
				t.Errorf("%v %s: %+v unstable", c.num, f, pid)
			}
		}
	}
	p, num, den := lhshe7Plant(t)
	pidtuneCheckDesigns(t, "LHSHE7", p, num, den, 0.1, nil)
}

// A P loop on 0.3(s+z)/(s²+as−b) has the slow root ≈ −(Kz−b)/(a+K), which
// crosses zero at K = b/z; with z ≤ 1e-8 its size is below the float
// eigenvalue error ε·K, so both sides need the exact test. Gains within 30%
// of b/z, whose root lies within 1e-10 of the axis, are skipped. Oracle:
// exact Routh.
func TestPIDStabilityScreenHighGainMatchesExact(t *testing.T) {
	const a, b = 0.66, 0.44
	var stable, unstable int
	for _, z := range []float64{1e-8, 1e-9} {
		num, den := Poly{0.3, 0.3 * z}, Poly{1, a, -b}
		p, _ := negatedPlant(t, num, den, 0)
		screen, err := newPIDStabilityScreen(p, nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		loop := newPidtuneLoop(p)
		if loop == nil {
			t.Fatal("no pidtuneLoop")
		}
		for i := range 400 {
			f := math.Pow(10, -1+3*float64(i)/400)
			if math.Abs(f-1) < 0.3 {
				continue
			}
			c := PID2{Kp: f * b / (0.3 * z)}
			want := pidtuneExactHurwitz(&PID{Kp: c.Kp}, num, den)
			got, err := screen.closedLoopPoles(&c)
			if err != nil {
				t.Fatal(err)
			}
			fast, err := loop.stable(screen, &c)
			if err != nil {
				t.Fatal(err)
			}
			if got != want || fast != want {
				t.Errorf("z=%g Kp=%g: screen %v, pidtuneLoop %v, exact %v", z, c.Kp, got, fast, want)
			}
			if want {
				stable++
			} else {
				unstable++
			}
		}
	}
	if stable == 0 || unstable == 0 {
		t.Fatalf("sweep one-sided: %d stable, %d unstable", stable, unstable)
	}
}

// LHSHE7 plant: the screen must match the exact roots for I and PI gains
// of either sign around the I design, but for roots within
// poleStabilityTolerance of the circle, which it rejects as IsStable does.
func TestPIDStabilityScreenDelayedClusterMatchesExact(t *testing.T) {
	p, num, den := lhshe7Plant(t)
	screen, err := newPIDStabilityScreen(p, nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	loop := newPidtuneLoop(screen.model)
	var stable, unstable int
	for _, kp := range []float64{0, -1e-7, -2.4e-7, 1e-7} {
		for _, ki := range []float64{-1e-9, -1e-10, -1e-11, 1e-12, 1e-11, 7.85e-11, 1e-9} {
			pid := &PID{Kp: kp, Ki: ki, Dt: 0.1}
			c := PID2{Kp: kp, Ki: ki, Dt: 0.1}
			want := pidtuneExactSchur(pid, num, den, 0.1)
			got, err := screen.closedLoopPoles(&c)
			if err != nil {
				t.Fatal(err)
			}
			fast := got
			if loop != nil {
				if fast, err = loop.stable(screen, &c); err != nil {
					t.Fatal(err)
				}
			}
			if want {
				stable++
			} else {
				unstable++
			}
			if got == want && fast == want {
				continue
			}
			if want && !got && !fast && !pidtuneExactSchurStableRadius(t, pid, num, den, 0.1, 1-1e-10) {
				continue
			}
			t.Errorf("Kp=%g Ki=%g: screen %v, pidtuneLoop %v, exact %v", kp, ki, got, fast, want)
		}
	}
	if stable == 0 || unstable == 0 {
		t.Fatalf("sweep one-sided: %d stable, %d unstable", stable, unstable)
	}
}
