package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func prd103Model(t *testing.T, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			0, 1, 0.25,
			-2, -3, 0.5,
			0.75, -0.5, -1.5,
		},
		[]float64{
			1, -0.25,
			0.5, 1,
			-0.5, 0.75,
		},
		[]float64{
			1, 0.25, -0.5,
			0.2, 1, 0.4,
		},
		[]float64{
			0.05, -0.03,
			0.02, 0.04,
		},
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"reference", "load"}
	sys.OutputName = []string{"temperature", "pressure"}
	sys.StateName = []string{"x-temp", "x-flow", "x-energy"}
	return sys
}

func TestPRD103InterconnectionTopologyPublicDelayWorkflows(t *testing.T) {
	plant := prd103Model(t, 0.1)
	if err := plant.SetInputDelay([]float64{2, 0}); err != nil {
		t.Fatal(err)
	}
	if err := plant.SetOutputDelay([]float64{0, 3}); err != nil {
		t.Fatal(err)
	}

	controller, err := NewGain(mat.NewDense(2, 2, []float64{0.15, -0.02, 0.04, 0.12}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	controller.OutputName = []string{"reference", "load"}
	controller.InputName = []string{"temperature", "pressure"}
	if err := controller.SetInputDelay([]float64{0, 1}); err != nil {
		t.Fatal(err)
	}

	closed, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}
	if closed.InputDelay != nil || !closed.HasInternalDelay() {
		t.Fatalf("plant InputDelay must move inside the loop: InputDelay=%v internal=%v", closed.InputDelay, closed.HasInternalDelay())
	}
	assertResponseOracle(t, "closed", closed, func(s complex128) [][]complex128 {
		return closedLoopOracle(t, plant, controller, -1, s)
	})
	if !stringSlicesEqual(closed.InputName, plant.InputName) || !stringSlicesEqual(closed.OutputName, plant.OutputName) {
		t.Fatalf("closed names input=%v output=%v", closed.InputName, closed.OutputName)
	}

	M, err := Append(plant, controller)
	if err != nil {
		t.Fatal(err)
	}
	Q := mat.NewDense(4, 4, []float64{
		0, 0, 0.1, 0,
		0, 0, 0, 0.1,
		-1, 0, 0, 0,
		0, -1, 0, 0,
	})
	connected, err := Connect(M, Q, []int{0, 1}, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !stringSlicesEqual(connected.InputName, plant.InputName) || !stringSlicesEqual(connected.OutputName, plant.OutputName) {
		t.Fatalf("connected names input=%v output=%v", connected.InputName, connected.OutputName)
	}

	omega := []float64{0.2, 1.0, 2.5}
	if _, err := closed.FreqResponse(omega); err != nil {
		t.Fatalf("closed delayed frequency response failed: %v", err)
	}
	if _, err := connected.FreqResponse(omega); err != nil {
		t.Fatalf("connected delayed frequency response failed: %v", err)
	}
}

func TestPRD103DelayConversionPolicyPublicWorkflows(t *testing.T) {
	cont := prd103Model(t, 0)
	cont.InputDelay = []float64{0.25, 0}
	cont.OutputDelay = []float64{0, 0.35}
	if err := cont.SetDelay(mat.NewDense(2, 2, []float64{
		0, 0.25,
		0.35, 0.60,
	})); err != nil {
		t.Fatal(err)
	}

	disc, err := cont.DiscretizeWithOpts(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	if disc.Delay != nil {
		t.Fatalf("DiscretizeWithOpts kept residual IODelay: %v", disc.Delay)
	}
	for _, delays := range [][]float64{disc.InputDelay, disc.OutputDelay} {
		for _, d := range delays {
			if math.Abs(d-math.Round(d)) > 1e-12 {
				t.Fatalf("external delay remained fractional: input=%v output=%v", disc.InputDelay, disc.OutputDelay)
			}
		}
	}

	controller, err := NewGain(mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.1}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	discPlant := cont.Copy()
	discPlant.Dt = 0.1
	discPlant.InputDelay = []float64{2.5, 0}
	discPlant.OutputDelay = []float64{0, 3.5}
	discPlant.Delay = mat.NewDense(2, 2, []float64{0, 2.5, 3.5, 6.0})
	if _, err := Feedback(discPlant, controller, -1, WithApproximatedDelays()); !errors.Is(err, ErrFractionalDelay) {
		t.Fatalf("Feedback fractional delay error = %v, want ErrFractionalDelay", err)
	}
	closed, err := Feedback(discPlant, controller, -1, WithThiranOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	if closed.HasDelay() {
		t.Fatalf("Feedback kept external delay: input=%v output=%v io=%v", closed.InputDelay, closed.OutputDelay, closed.Delay)
	}
}

func TestPRD103FrequencyAnalysisSampledResponseParity(t *testing.T) {
	sys := prd103Model(t, 0.1)
	omega := []float64{0.2, 0.8, 1.7, 2.9}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}
	frm := frd.FreqResponse()
	for k := range omega {
		for i := 0; i < resp.P; i++ {
			for j := 0; j < resp.M; j++ {
				assertComplexApprox(t, frm.At(k, i, j), resp.At(k, i, j), 1e-12)
			}
		}
	}

	sysSigma, err := sys.Sigma(omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	frdSigma, err := frd.Sigma()
	if err != nil {
		t.Fatal(err)
	}
	if sysSigma.NSV() != frdSigma.NSV() {
		t.Fatalf("FRD singular value count = %d, want %d", frdSigma.NSV(), sysSigma.NSV())
	}
	for k := range omega {
		for i := 0; i < sysSigma.NSV(); i++ {
			if d := math.Abs(sysSigma.At(k, i) - frdSigma.At(k, i)); d > 1e-10 {
				t.Fatalf("sigma[%d,%d] differs by %g", k, i, d)
			}
		}
	}

	delayed, err := New(
		mat.NewDense(1, 1, []float64{0.8}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := delayed.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	if _, err := Margin(delayed); err != nil {
		t.Fatalf("delayed SISO Margin failed: %v", err)
	}
	if _, err := Pidtune(delayed, PidtunePI); err != nil {
		t.Fatalf("delayed SISO Pidtune failed: %v", err)
	}
}

func TestPRD103TimeResponsePlanningPublicWorkflows(t *testing.T) {
	gain, err := NewGain(mat.NewDense(2, 2, []float64{2, -1, 0.5, 1.5}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	gain.OutputName = []string{"temperature", "pressure"}
	step, err := Step(gain, 0.3)
	if err != nil {
		t.Fatal(err)
	}
	if !stringSlicesEqual(step.OutputName, gain.OutputName) {
		t.Fatalf("Step OutputName = %v, want %v", step.OutputName, gain.OutputName)
	}
	if r, c := step.Y.Dims(); r != 4 || c != 3 {
		t.Fatalf("Step dims = %dx%d, want 4x3", r, c)
	}

	cont := prd103Model(t, 0)
	disc, err := cont.DiscretizeZOH(0.1)
	if err != nil {
		t.Fatal(err)
	}
	if err := disc.SetDelay(mat.NewDense(2, 2, []float64{1, 2, 0, 1})); err != nil {
		t.Fatal(err)
	}
	tvec := []float64{0, 0.1, 0.2, 0.3}
	uLsim := mat.NewDense(len(tvec), 2, []float64{
		1, 0,
		0.5, -0.5,
		0, -1,
		-0.5, -0.25,
	})
	uSim := mat.NewDense(2, len(tvec), []float64{
		1, 0.5, 0, -0.5,
		0, -0.5, -1, -0.25,
	})
	lsim, err := Lsim(disc, uLsim, tvec, nil)
	if err != nil {
		t.Fatal(err)
	}
	sim, err := disc.Simulate(uSim, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertDenseApprox(t, lsim.Y, sim.Y, 1e-12)
	if _, err := Lsim(disc, uLsim, []float64{0, 0.1, 0.25, 0.3}, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("non-uniform time grid error = %v, want ErrDimensionMismatch", err)
	}
}
