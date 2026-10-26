package controlsys

import (
	"context"
	"errors"
	"fmt"
	"testing"
)

// pidTuningCheckStableOrUnattainable passes a TunePID result that is
// ErrPIDTuningTargetUnattainable or a design whose closed loop on plant is
// stable with a crossover and |S| <= 4.5 per the num/den oracle; nil num
// checks closed-loop poles of the realization instead.
func pidTuningCheckStableOrUnattainable(t *testing.T, name string, plant *System, num, den Poly, r *PIDTuningResult, err error) {
	t.Helper()
	if err != nil {
		if !errors.Is(err, ErrPIDTuningTargetUnattainable) || errors.Is(err, ErrInvalidArgument) {
			t.Errorf("%s: %v; want a design or ErrPIDTuningTargetUnattainable", name, err)
		}
		return
	}
	c := *r.Controller
	if num != nil {
		pidtuneCheckRobust(t, name, mustPID(t, c.Kp, c.Ki, c.Kd, c.Tf, 0), num, den, 0, 4.5)
		return
	}
	cl, e := Feedback(pidTuningOpenLoop(t, plant, &c), nil, -1)
	if e != nil {
		t.Fatalf("%s: Feedback: %v", name, e)
	}
	if stable, e := cl.IsStable(); e != nil || !stable {
		t.Errorf("%s: %+v closed loop stable %v, %v", name, c, stable, e)
	}
}

// ZHTUBP: a crossover TunePID moves off a failed default can land where
// |G| < 1e-15, far past the plant's dynamics: the free P walk on a
// high-pass k·s³/den reached ω ≈ 3.5e-7 (|G| ≈ 3.9e-21), the PDF raise on
// an unstable double integrator ω ≈ 1.7e3 (|G| ≈ 3.7e-16). That probe
// failed the whole call with ErrInvalidArgument; as MATLAB pidtune, a
// valid plant without an admissible design is unattainable. Oracle:
// closed-loop roots from num/den for any design returned.
func TestTunePIDMovedCrossoverAtZeroGainIsNotInvalid(t *testing.T) {
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
	}{
		{PidtuneP, Poly{-0.24371488599761515, 0, 0, 0}, Poly{1, 5.3999877579765405, 7.271135289167493, 2.7160209747907498}},
		{PidtunePDF, Poly{8482.156439500475}, Poly{1, 24.348694175227177, 281.9286578100816, -78.85833235018102, 22.51048305894404, 0, 0}},
	} {
		p := makePlant(t, c.num, c.den)
		r, err := TunePID(context.Background(), p, c.f, PIDTuningOptions{})
		pidTuningCheckStableOrUnattainable(t, fmt.Sprintf("%s %v/%v", c.f, c.num, c.den), p, c.num, c.den, r, err)
	}
}
