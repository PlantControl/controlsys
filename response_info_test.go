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
	if rise, ok := got.RiseTime(); !ok || math.Abs(rise-2.2) > 0.08 {
		t.Errorf("rise time = %g, %v, want near 2.2", rise, ok)
	}
	if settling, ok := got.SettlingTime(); !ok || math.Abs(settling-3.9) > 0.15 {
		t.Errorf("settling time = %g, %v, want near 3.9", settling, ok)
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
	if settling, ok := got.SettlingTime(); ok {
		t.Fatalf("settling time = %g, want not settled", settling)
	}
	if rise, ok := got.RiseTime(); !ok || math.Abs(rise-(1+0.1/0.6-0.125)) > 1e-12 {
		t.Errorf("rise time = %g, %v, want interpolated 0.1→0.9 crossing", rise, ok)
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
	if rise, ok := got.RiseTime(); !ok || rise <= 0 {
		t.Errorf("rise time = %g, %v, want positive", rise, ok)
	}
	if _, ok := got.SettlingTime(); !ok {
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
		if _, ok := metric.SettlingTime(); !ok {
			t.Fatalf("metric %d did not settle", i)
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
	if m := info.Metrics[0]; !m.settled || math.Abs(m.SteadyStateValue-1.0/3) > 1e-3 {
		t.Fatalf("metrics = %+v, want settled at 1/3", m)
	}
}

func TestStepInfoPeakTimeIsFirstMaximum(t *testing.T) {
	resp := &TimeResponse{T: []float64{0, 1, 2, 3, 4}, Y: mat.NewDense(1, 5, []float64{0, 1, 1, 1, 1})}
	info, err := StepInfo(resp, nil)
	if err != nil {
		t.Fatal(err)
	}
	if m := info.Metrics[0]; m.PeakTime != 1 || m.Peak != 1 {
		t.Fatalf("Peak=%g PeakTime=%g, want 1 at t=1", m.Peak, m.PeakTime)
	}
}

func TestStepInfoForSystemUsesDCGainAsSteadyState(t *testing.T) {
	slow, _ := New(mat.NewDense(1, 1, []float64{-0.1}), mat.NewDense(1, 1, []float64{0.1}), mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
	info, err := StepInfoForSystem(slow, 5, nil)
	if err != nil {
		t.Fatal(err)
	}
	m := info.Metrics[0]
	if settling, ok := m.SettlingTime(); ok || m.SteadyStateValue != 1 {
		t.Fatalf("tau=10 at t=5: SettlingTime=%g, %v SteadyState=%g, want unsettled toward DC gain 1", settling, ok, m.SteadyStateValue)
	}

	a := mat.NewDense(2, 2, []float64{-1, 0.4, -0.3, -2})
	b := mat.NewDense(2, 3, []float64{1, 0, 0.5, 0, 1, -1})
	c := mat.NewDense(2, 2, []float64{1, 0.2, 0, 1})
	d := mat.NewDense(2, 3, []float64{0, 0.1, 0, 0.3, 0, 0})
	sys, _ := New(a, b, c, d, 0)
	info, err = StepInfoForSystem(sys, 20, nil)
	if err != nil {
		t.Fatal(err)
	}
	var aInv, ca, cab mat.Dense
	if err := aInv.Inverse(a); err != nil {
		t.Fatal(err)
	}
	ca.Mul(c, &aInv)
	cab.Mul(&ca, b)
	for input := range 3 {
		for output := range 2 {
			want := d.At(output, input) - cab.At(output, input)
			if got := info.Metrics[input*2+output].SteadyStateValue; math.Abs(got-want) > 1e-12 {
				t.Errorf("row u%d->y%d: steady state %g, want %g", input, output, got, want)
			}
		}
	}
}

func TestStepInfoRejectsInvalidOptions(t *testing.T) {
	resp := &TimeResponse{T: []float64{0, 1, 2, 3}, Y: mat.NewDense(1, 4, []float64{0, 0.5, 0.9, 1})}
	for _, opts := range []*StepInfoOptions{
		{RiseTimeLimits: [2]float64{0.9, 0.1}},
		{RiseTimeLimits: [2]float64{0.1, 0}},
		{RiseTimeLimits: [2]float64{-1, 2}},
		{RiseTimeLimits: [2]float64{math.NaN(), 0.9}},
		{SettlingThreshold: -0.02},
		{SettlingThreshold: math.NaN()},
		{SettlingThreshold: 1},
		{SteadyStateValue: []float64{math.NaN()}},
		{SteadyStateValue: []float64{math.Inf(1)}},
	} {
		if _, err := StepInfo(resp, opts); err == nil {
			t.Errorf("opts %+v: want error", *opts)
		}
	}
	if _, err := StepInfo(resp, &StepInfoOptions{RiseTimeLimits: [2]float64{0, 0.5}, SettlingThreshold: 0.05}); err != nil {
		t.Errorf("valid options rejected: %v", err)
	}
}

func TestStepInfoUndefinedRiseAndSentinels(t *testing.T) {
	resp := &TimeResponse{T: []float64{0, 1, 2}, Y: mat.NewDense(2, 3, []float64{1, 1, 1, 0, 0.05, 0.08})}
	info, err := StepInfo(resp, &StepInfoOptions{SteadyStateValue: []float64{1, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if rise, ok := info.Metrics[0].RiseTime(); ok {
		t.Errorf("zero step rise = %g, want undefined", rise)
	}
	if rise, ok := info.Metrics[1].RiseTime(); ok {
		t.Errorf("uncrossed rise = %g, want undefined", rise)
	}
	if _, ok := info.Metrics[1].SettlingTime(); ok {
		t.Error("unsettled row reported settled")
	}

	for name, call := range map[string]func() error{
		"nil response": func() error { _, err := StepInfo(nil, nil); return err },
		"limits": func() error {
			_, err := StepInfo(resp, &StepInfoOptions{RiseTimeLimits: [2]float64{0.9, 0.1}})
			return err
		},
		"threshold": func() error { _, err := StepInfo(resp, &StepInfoOptions{SettlingThreshold: 2}); return err },
		"NaN ss": func() error {
			_, err := StepInfo(resp, &StepInfoOptions{SteadyStateValue: []float64{math.NaN(), 1}})
			return err
		},
		"time order": func() error {
			_, err := StepInfo(&TimeResponse{T: []float64{0, 0, 1}, Y: resp.Y}, nil)
			return err
		},
	} {
		wantErr(t, name, call(), ErrInvalidArgument, "StepInfo")
	}
	_, err = StepInfoForSystem(nil, 1, nil)
	wantErr(t, "StepInfoForSystem nil", err, ErrInvalidArgument, "StepInfoForSystem")
}
