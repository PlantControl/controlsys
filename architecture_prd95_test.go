package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func prd95MIMOModel(t *testing.T, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			0, 1, 0,
			-2, -3, 0.5,
			0.25, -0.75, -1.5,
		},
		[]float64{
			1, 0,
			0, 1,
			0.5, -0.25,
		},
		[]float64{
			1, 0.5, 0,
			0, 1, -0.5,
		},
		[]float64{
			0.05, -0.02,
			0.01, 0.04,
		},
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputName = []string{"u-left", "u-right"}
	sys.OutputName = []string{"y-top", "y-bottom"}
	sys.StateName = []string{"x-a", "x-b", "x-c"}
	return sys
}

func TestPRD95DelayBankPublicWorkflowsShareRules(t *testing.T) {
	plant := prd95MIMOModel(t, 0)
	plant.InputDelay = []float64{0.35, 0}
	plant.OutputDelay = []float64{0, 0.45}
	if err := plant.SetDelay(mat.NewDense(2, 2, []float64{
		0, 0.35,
		0.45, 0.80,
	})); err != nil {
		t.Fatal(err)
	}

	disc, err := plant.C2D(0.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}
	if disc.Delay != nil {
		t.Fatalf("discretized model kept input-output delay: %v", disc.Delay)
	}
	for _, delays := range [][]float64{disc.InputDelay, disc.OutputDelay} {
		for _, d := range delays {
			if math.Abs(d-math.Round(d)) > 1e-12 {
				t.Fatalf("discretized model kept fractional external delay: input=%v output=%v", disc.InputDelay, disc.OutputDelay)
			}
		}
	}
	if n, m, p := disc.Dims(); n <= 3 || m != 2 || p != 2 {
		t.Fatalf("discretized dims = %d,%d,%d, want added delay states and 2x2 channels", n, m, p)
	}

	controller, err := NewGain(mat.NewDense(2, 2, []float64{0.1, 0.02, -0.03, 0.08}), 0)
	if err != nil {
		t.Fatal(err)
	}
	controller.Dt = 0.1
	discPlant := plant.Copy()
	discPlant.Dt = 0.1
	discPlant.InputDelay = []float64{3.5, 0}
	discPlant.OutputDelay = []float64{0, 4.5}
	discPlant.Delay = mat.NewDense(2, 2, []float64{
		0, 3.5,
		4.5, 8.0,
	})

	for _, opt := range []FeedbackOption{WithApproximatedDelays(), WithThiranOrder(3)} {
		if _, err := Feedback(discPlant, controller, -1, opt); !errors.Is(err, ErrFractionalDelay) {
			t.Fatalf("Feedback fractional delay error = %v, want ErrFractionalDelay", err)
		}
	}
}

func TestPRD95DelayBankKeepsIntegerDelayExactWithThiranOrder(t *testing.T) {
	sys := prd95MIMOModel(t, 0.1)
	sys.InputDelay = []float64{2, 0}
	sys.OutputDelay = []float64{0, 3}

	controller, err := NewGain(mat.NewDense(2, 2, []float64{0.2, 0, 0, 0.1}), 0.1)
	if err != nil {
		t.Fatal(err)
	}

	closed, err := Feedback(sys, controller, -1, WithThiranOrder(3))
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := closed.Dims()
	if m != 2 || p != 2 {
		t.Fatalf("closed dims = %d,%d,%d, want 2x2 channels", n, m, p)
	}
	if n >= 3+2*3 {
		t.Fatalf("closed state count = %d, want exact integer delay states rather than one Thiran block per delayed channel", n)
	}
}

func TestPRD95LFTDelayWorkflowPreservesExternalDelayAndMetadata(t *testing.T) {
	M, err := NewGain(mat.NewDense(2, 2, []float64{
		0.2, 0.5,
		1.0, 0.1,
	}), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	M.InputName = []string{"reference", "uncertain-return"}
	M.OutputName = []string{"controlled", "uncertain-drive"}
	if err := M.SetInputDelay([]float64{2, 0}); err != nil {
		t.Fatal(err)
	}
	if err := M.SetOutputDelay([]float64{3, 0}); err != nil {
		t.Fatal(err)
	}

	delta, err := New(
		mat.NewDense(1, 1, []float64{0.8}),
		mat.NewDense(1, 1, []float64{0.2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := delta.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}

	result, err := LFT(M, delta)
	if err != nil {
		t.Fatal(err)
	}
	if !floatSlicesEqual(result.InputDelay, []float64{2}) {
		t.Fatalf("InputDelay = %v, want [2]", result.InputDelay)
	}
	if !floatSlicesEqual(result.OutputDelay, []float64{3}) {
		t.Fatalf("OutputDelay = %v, want [3]", result.OutputDelay)
	}
	if !stringSlicesEqual(result.InputName, []string{"reference"}) {
		t.Fatalf("InputName = %v", result.InputName)
	}
	if !stringSlicesEqual(result.OutputName, []string{"controlled"}) {
		t.Fatalf("OutputName = %v", result.OutputName)
	}
	if !result.HasInternalDelay() {
		t.Fatal("expected Delta delay to be represented as internal delay")
	}

	omega := []float64{0.2, 1.1, 2.7}
	resp, err := result.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	if resp.NFreq != len(omega) || resp.P != 1 || resp.M != 1 {
		t.Fatalf("FreqResponse dims = %d,%d,%d", resp.NFreq, resp.P, resp.M)
	}
}

func TestPRD95TimeDomainResponsePreparationPublicContracts(t *testing.T) {
	cont := prd95MIMOModel(t, 0)
	cont.OutputName = []string{"temperature", "pressure"}
	tFinal := 0.5

	step, err := Step(cont, tFinal)
	if err != nil {
		t.Fatal(err)
	}
	if !stringSlicesEqual(step.OutputName, cont.OutputName) {
		t.Fatalf("Step OutputName = %v, want %v", step.OutputName, cont.OutputName)
	}
	imp, err := Impulse(cont, tFinal)
	if err != nil {
		t.Fatal(err)
	}
	if !stringSlicesEqual(imp.OutputName, cont.OutputName) {
		t.Fatalf("Impulse OutputName = %v, want %v", imp.OutputName, cont.OutputName)
	}
	initResp, err := Initial(cont, mat.NewVecDense(3, []float64{1, -1, 0.5}), tFinal)
	if err != nil {
		t.Fatal(err)
	}
	if !stringSlicesEqual(initResp.OutputName, cont.OutputName) {
		t.Fatalf("Initial OutputName = %v, want %v", initResp.OutputName, cont.OutputName)
	}

	disc, err := cont.C2D(0.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	tvec := []float64{0, 0.1, 0.2, 0.3, 0.4}
	uLsim := mat.NewDense(len(tvec), 2, []float64{
		1, -1,
		0.5, -0.5,
		0.25, -0.25,
		0, 0,
		-0.25, 0.25,
	})
	uSim := mat.NewDense(2, len(tvec), []float64{
		1, 0.5, 0.25, 0, -0.25,
		-1, -0.5, -0.25, 0, 0.25,
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

	if _, err := Lsim(disc, uLsim, []float64{0, 0.1, 0.25, 0.3, 0.4}, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("non-uniform Lsim err = %v, want ErrInvalidArgument", err)
	}
	otherDt := disc.Copy()
	otherDt.Dt = 0.2
	if _, err := Lsim(otherDt, uLsim, tvec, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("Dt mismatch Lsim err = %v, want ErrInvalidArgument", err)
	}
}

func TestPRD95FrequencyResponseLayoutPublicWorkflowsAgree(t *testing.T) {
	sys := prd95MIMOModel(t, 0.1)
	omega := []float64{0.2, 1.0, 2.4}
	resp, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}
	bode, err := sys.Bode(omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	for k := range omega {
		for i := 0; i < resp.P; i++ {
			for j := 0; j < resp.M; j++ {
				h := resp.At(k, i, j)
				assertComplexApprox(t, frd.At(k, i, j), h, 1e-12)
				wantMag := 20 * math.Log10(cmplx.Abs(h))
				if math.Abs(bode.MagDBAt(k, i, j)-wantMag) > 1e-10 {
					t.Fatalf("Bode mag[%d,%d,%d] = %v, want %v", k, i, j, bode.MagDBAt(k, i, j), wantMag)
				}
			}
		}
	}

	dt := 0.05
	steps := 512
	u := mat.NewDense(2, steps, nil)
	for k := range steps {
		u.Set(0, k, math.Sin(0.17*float64(k)))
		u.Set(1, k, math.Cos(0.11*float64(k)))
	}
	disc, err := sys.D2D(dt, D2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	sim, err := disc.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}
	est, err := FreqRespEst(u, sim.Y, dt, &FreqRespEstOpts{NFFT: 128})
	if err != nil {
		t.Fatal(err)
	}
	for f := range est.Omega {
		for i := 0; i < est.H.P; i++ {
			for j := 0; j < est.H.M; j++ {
				coh, _ := est.CoherenceAt(f, i, j)
				if coh < 0 || coh > 1+1e-9 {
					t.Fatalf("coherence[%d,%d,%d] = %v, want [0,1]", f, i, j, coh)
				}
			}
		}
	}
}
