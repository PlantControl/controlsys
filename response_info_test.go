package controlsys

import (
	"errors"
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

func TestStepInfoForSystem_ContinuousInternalDelayUnstable(t *testing.T) {
	_, err := StepInfoForSystem(scalarDDE(t, -2, 2), 40, nil)
	if !errors.Is(err, ErrUnstable) {
		t.Fatalf("err = %v, want ErrUnstable", err)
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
	info, err := StepInfo(resp, &StepInfoOptions{SteadyStateValue: []float64{1, 1}, InitialValue: []float64{1, 0}})
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

func sampledStepInfoResponse(dt, tf float64, f func(float64) float64) *TimeResponse {
	n := int(math.Round(tf/dt)) + 1
	T := make([]float64, n)
	y := make([]float64, n)
	for k := range n {
		T[k] = float64(k) * dt
		y[k] = f(T[k])
	}
	return &TimeResponse{T: T, Y: mat.NewDense(1, n, y)}
}

func checkSettlingWithin(t *testing.T, m StepMetric, want, step float64) {
	t.Helper()
	st, ok := m.SettlingTime()
	if !ok || st < want-1e-12 || st > want+step+1e-12 {
		t.Errorf("settling = %g, %v, want first sample after %g", st, ok, want)
	}
}

func TestStepInfoFeedthroughUsesZeroInitialValue(t *testing.T) {
	const dt = 1e-3
	resp := sampledStepInfoResponse(dt, 20, func(t float64) float64 { return 2 - math.Exp(-t) })
	ss := []float64{2}
	info, err := StepInfo(resp, &StepInfoOptions{SteadyStateValue: ss})
	if err != nil {
		t.Fatal(err)
	}
	m := info.Metrics[0]
	if rt, ok := m.RiseTime(); !ok || math.Abs(rt-math.Log(5)) > 1e-5 {
		t.Errorf("rise = %g, %v, want ln5 (10%% level met at t=0)", rt, ok)
	}
	checkSettlingWithin(t, m, math.Log(25), dt)
	if m.Overshoot != 0 || m.Undershoot != 0 {
		t.Errorf("overshoot %g undershoot %g, want 0", m.Overshoot, m.Undershoot)
	}

	info, err = StepInfo(resp, &StepInfoOptions{SteadyStateValue: ss, InitialValue: []float64{1}})
	if err != nil {
		t.Fatal(err)
	}
	m = info.Metrics[0]
	if rt, ok := m.RiseTime(); !ok || math.Abs(rt-math.Log(9)) > 1e-5 {
		t.Errorf("yinit=1 rise = %g, %v, want ln9", rt, ok)
	}
	checkSettlingWithin(t, m, math.Log(50), dt)
}

func TestStepInfoForSystemFeedthroughMatchesMATLAB(t *testing.T) {
	sys, err := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	info, err := StepInfoForSystem(sys, 20, nil)
	if err != nil {
		t.Fatal(err)
	}
	m := info.Metrics[0]
	if rt, ok := m.RiseTime(); !ok || math.Abs(rt-math.Log(5)) > 0.01 {
		t.Errorf("rise = %g, %v, want ln5", rt, ok)
	}
	checkSettlingWithin(t, m, math.Log(25), 0.1)

	// (1-s)/(s+1): y = 1 - 2e^{-t} starts at -1, a 100% undershoot of the 0->1 step.
	sys, err = NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{2}, []float64{-1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	info, err = StepInfoForSystem(sys, 20, nil)
	if err != nil {
		t.Fatal(err)
	}
	m = info.Metrics[0]
	if math.Abs(m.Undershoot-100) > 1e-9 {
		t.Errorf("undershoot = %g, want 100", m.Undershoot)
	}
	if rt, ok := m.RiseTime(); !ok || math.Abs(rt-math.Log(9)) > 0.01 {
		t.Errorf("rise = %g, %v, want ln20-ln(2/0.9) = ln9", rt, ok)
	}
	checkSettlingWithin(t, m, math.Log(100), 0.1)
}

func TestStepInfoRiseWhenInitialSampleExceedsBothLimits(t *testing.T) {
	resp := &TimeResponse{T: []float64{0, 1, 2}, Y: mat.NewDense(1, 3, []float64{1, 1, 1})}
	info, err := StepInfo(resp, nil)
	if err != nil {
		t.Fatal(err)
	}
	m := info.Metrics[0]
	if rt, ok := m.RiseTime(); !ok || rt != 0 {
		t.Errorf("rise = %g, %v, want 0", rt, ok)
	}
	if st, ok := m.SettlingTime(); !ok || st != 0 {
		t.Errorf("settling = %g, %v, want 0", st, ok)
	}
}

func TestStepInfoRejectsInvalidInitialValue(t *testing.T) {
	resp := &TimeResponse{T: []float64{0, 1}, Y: mat.NewDense(1, 2, []float64{0, 1})}
	if _, err := StepInfo(resp, &StepInfoOptions{InitialValue: []float64{0, 0}}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("length mismatch err = %v", err)
	}
	if _, err := StepInfo(resp, &StepInfoOptions{InitialValue: []float64{math.NaN()}}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("NaN err = %v", err)
	}
}
