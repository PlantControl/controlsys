package controlsys_test

import (
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v2/controlsys/v2"
)

// tinyDelays are internal delays far below any response sample interval:
// tiny normals, the smallest normal and denormals whose dt/τ overflows int
// or float64.
var tinyDelays = []float64{1e-300, 0x1p-1022, 1e-310, 5e-324, 1e-9}

// tinyDelayLoop is x' = −x + u − x(t−τ), y = x; as τ → 0 it tends to
// x' = −2x + u.
func tinyDelayLoop(t *testing.T, tau float64) *controlsys.System {
	t.Helper()
	H, err := controlsys.New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 2, []float64{1, -1}),
		mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(2, 2, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys, err := controlsys.SetDelayModel(H, []float64{tau})
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func tinyDelayCheck(t *testing.T, label string, resp *controlsys.TimeResponse, want func(float64) float64, tol float64) {
	t.Helper()
	if resp == nil || resp.Y == nil {
		t.Fatalf("%s: nil response", label)
	}
	for k, tk := range resp.T {
		got := resp.Y.At(0, k)
		if math.IsNaN(got) || math.Abs(got-want(tk)) > tol {
			t.Fatalf("%s: y(%g)=%g, want %g ± %g", label, tk, got, want(tk), tol)
		}
	}
}

func TestTinyInternalDelayResponses(t *testing.T) {
	const tFinal, tol = 3.0, 5e-3
	step := func(tk float64) float64 { return (1 - math.Exp(-2*tk)) / 2 }
	decay := func(tk float64) float64 { return math.Exp(-2 * tk) }
	for _, tau := range tinyDelays {
		label := fmt.Sprintf("tau=%g", tau)
		sys := tinyDelayLoop(t, tau)

		resp, err := controlsys.Step(sys, tFinal)
		if err != nil {
			t.Fatalf("%s Step: %v", label, err)
		}
		tinyDelayCheck(t, label+" Step", resp, step, tol)

		resp, err = controlsys.Impulse(sys, tFinal)
		if err != nil {
			t.Fatalf("%s Impulse: %v", label, err)
		}
		tinyDelayCheck(t, label+" Impulse", resp, decay, 2*tol)

		resp, err = controlsys.Initial(sys, mat.NewVecDense(1, []float64{1}), tFinal)
		if err != nil {
			t.Fatalf("%s Initial: %v", label, err)
		}
		tinyDelayCheck(t, label+" Initial", resp, decay, 2*tol)

		const steps = 61
		tm := make([]float64, steps)
		u := mat.NewDense(steps, 1, nil)
		for k := range tm {
			tm[k] = 0.05 * float64(k)
			u.Set(k, 0, 1)
		}
		resp, err = controlsys.Lsim(sys, u, tm, nil)
		if err != nil {
			t.Fatalf("%s Lsim: %v", label, err)
		}
		tinyDelayCheck(t, label+" Lsim", resp, step, 2*tol)
	}
}
