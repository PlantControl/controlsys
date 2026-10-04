package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestStepInfoFirstOrderResponse(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(sys, 8)
	if err != nil {
		t.Fatal(err)
	}
	info, err := StepInfo(resp, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(info.Metrics) != 1 {
		t.Fatalf("got %d metrics, want 1", len(info.Metrics))
	}

	got := info.Metrics[0]
	if math.Abs(got.SteadyStateValue-1) > 1e-3 {
		t.Fatalf("steady-state value = %g, want near 1", got.SteadyStateValue)
	}
	if math.Abs(got.RiseTime-2.2) > 0.08 {
		t.Errorf("rise time = %g, want near 2.2", got.RiseTime)
	}
	if math.Abs(got.SettlingTime-3.9) > 0.15 {
		t.Errorf("settling time = %g, want near 3.9", got.SettlingTime)
	}
	if math.Abs(got.Peak-1) > 1e-3 {
		t.Errorf("peak = %g, want near 1", got.Peak)
	}
	if math.Abs(got.PeakTime-8) > 1e-12 {
		t.Errorf("peak time = %g, want final sample time 8", got.PeakTime)
	}
	if got.Overshoot > 1e-9 {
		t.Errorf("overshoot = %g, want 0", got.Overshoot)
	}
}

func TestStepInfoUnderdampedResponseUsesKnownSteadyState(t *testing.T) {
	wn := 2.0
	zeta := 0.25
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -wn * wn, -2 * zeta * wn}),
		mat.NewDense(2, 1, []float64{0, wn * wn}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(sys, 10)
	if err != nil {
		t.Fatal(err)
	}
	info, err := StepInfo(resp, &StepInfoOptions{SteadyStateValue: []float64{1}})
	if err != nil {
		t.Fatal(err)
	}
	got := info.Metrics[0]

	wantOvershoot := 100 * math.Exp(-zeta*math.Pi/math.Sqrt(1-zeta*zeta))
	if math.Abs(got.Overshoot-wantOvershoot) > 1.0 {
		t.Errorf("overshoot = %g, want near %g", got.Overshoot, wantOvershoot)
	}
	wantPeakTime := math.Pi / (wn * math.Sqrt(1-zeta*zeta))
	if math.Abs(got.PeakTime-wantPeakTime) > 0.08 {
		t.Errorf("peak time = %g, want near %g", got.PeakTime, wantPeakTime)
	}
	if got.Peak <= got.SteadyStateValue {
		t.Errorf("peak = %g, want above steady-state %g", got.Peak, got.SteadyStateValue)
	}
}

func TestStepInfoUnsettledResponseWithKnownSteadyState(t *testing.T) {
	resp := &TimeResponse{
		T: []float64{0, 1, 2, 3},
		Y: mat.NewDense(1, 4, []float64{0, 0.8, 1.4, 1.2}),
	}

	info, err := StepInfo(resp, &StepInfoOptions{SteadyStateValue: []float64{1}})
	if err != nil {
		t.Fatal(err)
	}
	got := info.Metrics[0]
	if got.Settled {
		t.Fatal("response reported settled, want unsettled")
	}
	if !math.IsNaN(got.SettlingTime) {
		t.Errorf("settling time = %g, want NaN", got.SettlingTime)
	}
}

func TestStepInfoDiscreteNonzeroFinalValue(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2}),
		1,
	)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(sys, 12)
	if err != nil {
		t.Fatal(err)
	}
	info, err := StepInfo(resp, &StepInfoOptions{SteadyStateValue: []float64{4}})
	if err != nil {
		t.Fatal(err)
	}
	got := info.Metrics[0]
	if math.Abs(got.SteadyStateValue-4) > 1e-12 {
		t.Fatalf("steady-state value = %g, want 4", got.SteadyStateValue)
	}
	if got.RiseTime <= 0 {
		t.Errorf("rise time = %g, want positive", got.RiseTime)
	}
	if !got.Settled {
		t.Error("discrete response did not settle")
	}
}

func TestStepInfoMIMONonSymmetricRows(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := Step(sys, 5)
	if err != nil {
		t.Fatal(err)
	}
	info, err := StepInfo(resp, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(info.Metrics) != 4 {
		t.Fatalf("got %d metrics, want 4", len(info.Metrics))
	}
	for i, metric := range info.Metrics {
		if math.IsNaN(metric.SettlingTime) {
			t.Fatalf("metric %d settling time is NaN", i)
		}
	}
}

func TestStepInfoForSystemRejectsUnstableModel(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = StepInfoForSystem(sys, 3, nil)
	if err == nil {
		t.Fatal("expected unstable model error")
	}
}

func TestStepInfoForSystem_ContinuousInternalDelaySimulates(t *testing.T) {
	sys := scalarDDE(t, -2, 0.5)
	info, err := StepInfoForSystem(sys, 40, nil)
	if err != nil {
		t.Fatal(err)
	}
	// x' = -x - 2x(t-τ) + u settles at 1/3.
	if m := info.Metrics[0]; !m.Settled || math.Abs(m.SteadyStateValue-1.0/3) > 1e-3 {
		t.Fatalf("metrics = %+v, want settled at 1/3", m)
	}
}
