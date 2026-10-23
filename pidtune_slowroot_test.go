package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// 7ZE2YK: the delayed Nyquist screen merged the PI integrator with a plant
// pair at ω0 < 1e-6 into one indentation, which enclosed the slow real
// closed-loop root that a slow RHP zero leaves, and certified the loop
// stable. Oracle: P(s) = s·D(s) + (Kp·s + Ki)·N(s)·e^{−sτ} (D(s) + Kp·N(s)·e^{−sτ}
// for P) has P(+∞) > 0 for monic D, so P(0) < 0 is a real RHP root.
func TestPidTuningDelayScreenSlowRealRoot(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	rejected := 0
	for trial := range 300 {
		z := math.Pow(10, -9+6*rng.Float64())
		w0 := math.Pow(10, -8+7*rng.Float64())
		zeta := []float64{0, 1e-12, 0.01, 0.5}[trial%4]
		num := Poly{0.5 + rng.Float64(), -(0.5 + rng.Float64()) * z}
		den := Poly{1, 2 * zeta * w0, w0 * w0}
		if trial%3 == 0 {
			den = den.Mul(Poly{1, 0.3 + rng.Float64()})
		}
		tau := math.Pow(10, -3+2*rng.Float64())
		kp, ki := 0.1+rng.Float64(), 0.05+0.5*rng.Float64()
		if trial%2 == 1 {
			kp, ki = 1e-3+rng.Float64(), 0
		}
		p0 := ki * num[len(num)-1]
		if ki == 0 {
			p0 = den[len(den)-1] + kp*num[len(num)-1]
		}
		if p0 >= 0 {
			continue
		}
		screen := pidtuneSlowRootScreen(t, pidtuneBackoffPlant(t, num, den, 0, tau), 1)
		stable, err := screen(&PID2{Kp: kp, Ki: ki, B: 1, C: 1})
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if stable {
			t.Errorf("trial %d: %v/%v τ %g, Kp %g Ki %g: P(0) = %g < 0 certified stable", trial, num, den, tau, kp, ki, p0)
		}
		rejected++
	}
	if rejected < 100 {
		t.Fatalf("only %d designs with P(0) < 0", rejected)
	}

	a := []float64{-8.589e-24, 2.1544346900318827e-07, -2.1544346900318824e-07, -2.903e-23}
	b := []float64{-0.9667981925794609, -0.01238866346289056}
	c := []float64{-0.8568887533689471, -0.21079579943077972}
	p, err := New(mat.NewDense(2, 2, a), mat.NewDense(2, 1, b), mat.NewDense(1, 2, c), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := p.SetInputDelay([]float64{0.0031622776601683785}); err != nil {
		t.Fatal(err)
	}
	n0 := c[0]*(a[1]*b[1]-a[3]*b[0]) + c[1]*(a[2]*b[0]-a[0]*b[1])
	const kp, ki = 0.8673363870904787, 0.41363219216174363
	if !(ki*n0 < 0) {
		t.Fatalf("oracle: Ki·N(0) = %g", ki*n0)
	}
	stable, err := pidtuneSlowRootScreen(t, p, 0.83)(&PID2{Kp: kp, Ki: ki, B: 1, C: 1})
	if err != nil || stable {
		t.Errorf("6RPB54 PI: stable %v, %v; want unstable", stable, err)
	}
}

func pidtuneSlowRootScreen(t *testing.T, p *System, wc float64) func(*PID2) (bool, error) {
	t.Helper()
	eval, err := newSISOEval(p)
	if err != nil {
		t.Fatal(err)
	}
	screen, ok, err := pidTuningDelayStability(p, eval, wc, false, 1)
	if err != nil || !ok {
		t.Fatalf("screen: ok %v, %v", ok, err)
	}
	return screen
}
