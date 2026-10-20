package controlsys

import (
	"context"
	"errors"
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// 5LB3LS (QNALQV): a first-order discrete plant with its pole at 3.7e-44
// and a 20-sample input delay. Its absorbed 21-state model's
// TransferFunction collapsed to 0/(z−3e-33), so the ideal-derivative
// screen tested den(c)·(z−a) alone and certified the forward-Euler PD
// TunePID returned (Kd/Dt ≈ 0.097, Tf = 0), which has 19 roots outside
// the unit circle. The screen now tests the undelayed plant's
// characteristic shifted by z^20, and TunePID PD reports the target
// unattainable: with |G| ≈ 58 a gain crossover of Kp + Kd(z−1)/Dt puts
// |L| > 1 at high frequency against the 20-sample delay. Oracle: exact
// Schur–Cohn on (D·z + CB)/((z−a)·z^20).
func TestPIDStabilityScreenIdealDelayAbsorbed(t *testing.T) {
	const a, b, c, d = 3.720075976020836e-44, -0.9667981925794609, -0.8568887533689471, 57.7675665396686
	const lag = 20
	num, den := Poly{d, c*b - d*a}, Poly{1, -a}
	for _, dt := range []float64{100, 1, 0.1} {
		p, err := New(mat.NewDense(1, 1, []float64{a}), mat.NewDense(1, 1, []float64{b}), mat.NewDense(1, 1, []float64{c}), mat.NewDense(1, 1, []float64{d}), dt)
		if err != nil {
			t.Fatal(err)
		}
		if err := p.SetInputDelay([]float64{lag}); err != nil {
			t.Fatal(err)
		}
		screen, err := newPIDStabilityScreen(p, nil, 1)
		if err != nil {
			t.Fatal(err)
		}
		for _, k := range []PID2{
			{Kp: 0.008945254131813402, Kd: 0.9728762924556705 * dt / 10},
			{Kp: 0.008945254131813402 / 100, Kd: 0.9728762924556705 * dt / 1e3},
			{Kp: 0.001, Ki: 1e-6 / dt, Kd: 1e-5 * dt},
		} {
			k.B, k.C, k.Dt = 1, 1, dt
			want := pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt, 1, lag)
			if got, err := screen.closedLoopPoles(&k); err != nil || got != want {
				t.Errorf("dt %g %+v: screen stable %v %v, oracle %v", dt, k, got, err, want)
			}
			if got, ok := pidTuningIdealExact(screen.rational, &k, screen.lag); !ok || got != want {
				t.Errorf("dt %g %+v: exact stable %v %v, oracle %v", dt, k, got, ok, want)
			}
		}
		for _, f := range []PidtuneType{PidtunePD, PidtunePID} {
			name := fmt.Sprintf("dt %g %s", dt, f)
			if k, err := Pidtune(p, f, 0, nil); err == nil && k.Kd != 0 && k.Tf == 0 && !pidtuneExactSchurStableLag(t, k, num, den, dt, 1, lag) {
				t.Errorf("%s: Pidtune %+v unstable", name, *k)
			}
			r, err := TunePID(context.Background(), p, f, PIDTuningOptions{})
			if f == PidtunePD && errors.Is(err, ErrPIDTuningTargetUnattainable) {
				continue
			}
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			k := r.Controller
			if r.Evidence.Stability == PIDStabilitySampledOnly {
				continue
			}
			if !pidtuneExactSchurStableLag(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt, IFormula: k.IFormula, DFormula: k.DFormula}, num, den, dt, 1, lag) {
				t.Errorf("%s: %+v certified %s, unstable", name, *k, r.Evidence.Stability)
			}
		}
	}
}

// 5LB3LS (IP3XIC k=1): TunePID returned Kp 105572.5, Ki −38.475,
// Kd 483.4 (Tf = 0) on this near-unit-pole plant and certified it by
// closed-loop poles; a negative Ki on a plant of DC gain +0.208 puts a real
// root above z = 1. Fixed by the exact retest of 632b44a; pinned here.
// Oracle: exact Schur–Cohn.
func TestPIDStabilityScreenIdealIP3XICUnit(t *testing.T) {
	const dt = 0.01
	num := Poly{2.761969998249568e-16, 2.6332895798497864e-16, 6.196985260193668e-17}
	den := Poly{1, -5.388323209156541, 12.045405202918547, -14.292531743733242, 9.488502158799133, -3.339413745774066, 0.486361336946173}
	screen, err := newPIDStabilityScreen(pidtuneBackoffPlant(t, num, den, dt, 0), nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	k := PID2{Kp: 105572.5, Ki: -38.475, Kd: 483.4, B: 1, C: 1, Dt: dt}
	if pidtuneExactSchurStable(t, &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Dt: dt}, num, den, dt) {
		t.Fatal("oracle stable")
	}
	if got, err := screen.closedLoopPoles(&k); err != nil || got {
		t.Errorf("screen stable %v %v, oracle unstable", got, err)
	}
}
