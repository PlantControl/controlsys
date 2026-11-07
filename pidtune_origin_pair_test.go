package controlsys

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// XCAAKT/4PHRFF: the Nyquist count of a PI loop on a plant with an
// undamped pair at ω0 < 1e-6, judged without the real-axis sign
// certificate of the PID screen. addAxisPole counted the pair at the origin
// as one pole beside the integrator, so the indentation passed a slow real
// closed-loop root and stableClosedLoop certified the loop stable. Oracle:
// P(s) = s·D(s) + (Kp·s + Ki)·N(s)·e^{−sτ} has P(+∞) > 0 for monic D, so
// P(0) = Ki·N(0) < 0 is a real RHP root.
func TestDelayLoop_PIOriginAxisPairSlowRealRoot(t *testing.T) {
	check := func(name string, p *System, kp, ki float64) {
		t.Helper()
		pi, err := NewFromSlices(1, 1, 1, []float64{0}, []float64{1}, []float64{ki}, []float64{kp}, 0)
		if err != nil {
			t.Fatal(err)
		}
		loop, err := Series(pi, p)
		if err != nil {
			t.Fatal(err)
		}
		l, err := delayLoopFromSystem(loop, "loop")
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		stable, err := l.stableClosedLoop()
		if err == nil && stable {
			t.Errorf("%s: Kp %g Ki %g: P(0) < 0 certified stable", name, kp, ki)
		}
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
	check("6RPB54", p, kp, ki)

	rng := rand.New(rand.NewPCG(3, 2))
	for trial := range 100 {
		z := math.Pow(10, -9+6*rng.Float64())
		w0 := math.Pow(10, -8+2*rng.Float64())
		num := Poly{0.5 + rng.Float64(), -(0.5 + rng.Float64()) * z}
		den := Poly{1, 0, w0 * w0}
		tau := math.Pow(10, -3+2*rng.Float64())
		kp, ki := 0.1+rng.Float64(), 0.05+0.5*rng.Float64()
		if !(ki*num[1] < 0) {
			t.Fatalf("trial %d: oracle Ki·N(0) = %g", trial, ki*num[1])
		}
		check(fmt.Sprintf("trial %d", trial), pidtuneBackoffPlant(t, num, den, 0, tau), kp, ki)
	}
}
