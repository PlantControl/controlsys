package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestFreqResponse_FirstOrderLowpass(t *testing.T) {
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

	resp, err := sys.FreqResponse([]float64{1.0})
	if err != nil {
		t.Fatal(err)
	}

	mag := cmplx.Abs(resp.At(0, 0, 0))
	want := 1.0 / math.Sqrt(2)
	if math.Abs(mag-want) > 1e-10 {
		t.Fatalf("|H(j)| = %v, want %v", mag, want)
	}
}

func TestFreqResponse_Integrator(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 1.0, 10.0}
	resp, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		mag := cmplx.Abs(resp.At(k, 0, 0))
		if math.Abs(mag-1.0/w) > 1e-10 {
			t.Errorf("w=%v: |H|=%v, want %v", w, mag, 1.0/w)
		}
		ph := cmplx.Phase(resp.At(k, 0, 0)) * 180 / math.Pi
		if math.Abs(ph-(-90)) > 0.1 {
			t.Errorf("w=%v: phase=%v, want -90", w, ph)
		}
	}
}

func TestFreqResponse_Gain(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{3.5})
	sys, err := NewGain(D, 0)
	if err != nil {
		t.Fatal(err)
	}

	resp, err := sys.FreqResponse([]float64{0.1, 1.0, 100.0})
	if err != nil {
		t.Fatal(err)
	}

	for k := 0; k < resp.NFreq; k++ {
		mag := cmplx.Abs(resp.At(k, 0, 0))
		if math.Abs(mag-3.5) > 1e-10 {
			t.Errorf("freq %d: |H|=%v, want 3.5", k, mag)
		}
	}
}

func TestFreqResponse_Discrete(t *testing.T) {
	sysc, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dt := 0.001
	sysd, err := sysc.Discretize(dt)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.01, 0.1, 1.0}
	respC, err := sysc.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respD, err := sysd.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		magC := cmplx.Abs(respC.At(k, 0, 0))
		magD := cmplx.Abs(respD.At(k, 0, 0))
		relErr := math.Abs(magC-magD) / magC
		if relErr > 0.01 {
			t.Errorf("w=%v: continuous=%v, discrete=%v, relErr=%v", w, magC, magD, relErr)
		}
	}
}

func TestBode_FirstOrderLowpass(t *testing.T) {
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

	bode, err := sys.Bode([]float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	magDB := bode.MagDBAt(0, 0, 0)
	if math.Abs(magDB-(-3.0103)) > 0.01 {
		t.Errorf("mag at w=1: %v dB, want ~-3.01 dB", magDB)
	}

	bode2, err := sys.Bode([]float64{10.0, 100.0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	slope := (bode2.MagDBAt(1, 0, 0) - bode2.MagDBAt(0, 0, 0)) / (math.Log10(100.0) - math.Log10(10.0))
	if math.Abs(slope-(-20)) > 1 {
		t.Errorf("slope = %v dB/decade, want ~-20", slope)
	}
}

func TestBode_AutoFrequency(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-10}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode(nil, 50)
	if err != nil {
		t.Fatal(err)
	}

	if len(bode.Omega) != 50 {
		t.Fatalf("len(omega) = %d, want 50", len(bode.Omega))
	}

	wMin := bode.Omega[0]
	wMax := bode.Omega[len(bode.Omega)-1]
	if wMin > 10 || wMax < 10 {
		t.Errorf("auto range [%v, %v] doesn't cover pole at 10", wMin, wMax)
	}
}

func TestBode_PhaseUnwrap(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-1000, -110, -11,
		},
		[]float64{0, 0, 1000},
		[]float64{1, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode(nil, 500)
	if err != nil {
		t.Fatal(err)
	}

	for k := 1; k < len(bode.Omega); k++ {
		diff := math.Abs(bode.PhaseAt(k, 0, 0) - bode.PhaseAt(k-1, 0, 0))
		if diff > 180 {
			t.Errorf("phase jump at w=%v: %v -> %v (diff=%v)",
				bode.Omega[k], bode.PhaseAt(k-1, 0, 0), bode.PhaseAt(k, 0, 0), diff)
			break
		}
	}
}

func TestBode_MIMO(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode([]float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	mag00 := bode.MagDBAt(0, 0, 0)
	mag11 := bode.MagDBAt(0, 1, 1)

	want00 := 20 * math.Log10(1.0/math.Sqrt(2))
	want11 := 20 * math.Log10(1.0/math.Sqrt(5))

	if math.Abs(mag00-want00) > 0.1 {
		t.Errorf("H(0,0) mag = %v dB, want %v", mag00, want00)
	}
	if math.Abs(mag11-want11) > 0.1 {
		t.Errorf("H(1,1) mag = %v dB, want %v", mag11, want11)
	}
}

func TestEvalFr_MatchesFreqResponse(t *testing.T) {
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

	w := 5.0
	evalResult, err := sys.EvalFr(complex(0, w))
	if err != nil {
		t.Fatal(err)
	}

	freqResult, err := sys.FreqResponse([]float64{w})
	if err != nil {
		t.Fatal(err)
	}

	if cmplx.Abs(evalResult[0][0]-freqResult.At(0, 0, 0)) > 1e-15 {
		t.Fatalf("EvalFr=%v != FreqResponse=%v", evalResult[0][0], freqResult.At(0, 0, 0))
	}
}

func TestFrequencyEvaluatorSweepKernelParity(t *testing.T) {
	sys, err := New(
		mat.NewDense(4, 4, []float64{
			-0.8, 0.4, 0, 0.1,
			-0.2, -1.1, 0.3, 0,
			0.1, 0, -1.5, 0.2,
			0, -0.1, 0.25, -2,
		}),
		mat.NewDense(4, 2, []float64{1, 0.2, 0, 1, 0.4, -0.3, 0.1, 0.5}),
		mat.NewDense(2, 4, []float64{1, 0, 0.3, -0.2, 0.1, 0.7, 0, 0.4}),
		mat.NewDense(2, 2, []float64{0.1, 0, -0.05, 0.2}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.Delay = mat.NewDense(2, 2, []float64{0.02, 0.04, 0.01, 0.03})
	sys.InputDelay = []float64{0.05, 0.02}
	sys.OutputDelay = []float64{0.01, 0.03}
	discrete := sys.Copy()
	discrete.Dt = 0.1
	discrete.Delay = mat.NewDense(2, 2, []float64{1, 2, 3, 1})
	discrete.InputDelay = []float64{2, 1}
	discrete.OutputDelay = []float64{1, 3}

	siso := benchDenseSys(24, 1, 1)
	siso.InputDelay = []float64{0.05}
	siso.OutputDelay = []float64{0.02}
	discreteSISO := siso.Copy()
	discreteSISO.Dt = 0.1
	discreteSISO.InputDelay = []float64{2}
	discreteSISO.OutputDelay = []float64{1}

	tests := []struct {
		name      string
		system    *System
		omega     []float64
		wantDense bool
	}{
		{name: "ShortSweep", system: siso, omega: logspace(-2, 2, 2), wantDense: true},
		{name: "SmallModel", system: sys, omega: logspace(-2, 2, 100), wantDense: true},
		{name: "DiscreteSmallModel", system: discrete, omega: logspace(-2, 1, 100), wantDense: true},
		{name: "Hessenberg", system: siso, omega: logspace(-2, 2, 100), wantDense: false},
		{name: "DiscreteHessenberg", system: discreteSISO, omega: logspace(-2, 1, 100), wantDense: false},
		{name: "CoupledMIMO", system: benchDenseSys(30, 2, 2), omega: logspace(-2, 2, 100), wantDense: false},
		{name: "UpperHessenbergA", system: benchSys(40, 1, 1), omega: logspace(-2, 2, 100), wantDense: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			evaluator := newFrequencyEvaluator(test.system)
			if got := evaluator.useDenseSweep(len(test.omega)); got != test.wantDense {
				t.Fatalf("useDenseSweep = %t, want %t", got, test.wantDense)
			}
			response, err := test.system.FreqResponse(test.omega)
			if err != nil {
				t.Fatal(err)
			}
			for k, w := range test.omega {
				want, err := test.system.EvalFr(evaluator.sAt(w))
				if err != nil {
					t.Fatal(err)
				}
				for i := range response.P {
					for j := range response.M {
						if diff := cmplx.Abs(response.At(k, i, j) - want[i][j]); diff > 1e-12*cmplx.Abs(want[i][j]) {
							t.Fatalf("w=%g output=%d input=%d diff=%g", w, i, j, diff)
						}
					}
				}
			}
		})
	}
}

func TestFreqResponse_ShortSweepFallsBackAtPole(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, nil),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	response, err := sys.FreqResponse([]float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if got := response.At(1, 0, 0); cmplx.Abs(got+1i) > 1e-12 {
		t.Fatalf("response at w=1 = %v, want -1i", got)
	}
}

func TestFreqResponse_Empty(t *testing.T) {
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

	resp, err := sys.FreqResponse([]float64{})
	if err != nil {
		t.Fatal(err)
	}
	if resp != nil {
		t.Fatalf("expected nil, got %v", resp)
	}
}

func TestFreqResponse_InternalDelay_SISO(t *testing.T) {
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

	tau := 0.5
	err = sys.SetInternalDelay(
		[]float64{tau},
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 0.5, 1.0, 5.0, 10.0}
	respLFT, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respAbs, err := absorbed.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		magLFT := cmplx.Abs(respLFT.At(k, 0, 0))
		magAbs := cmplx.Abs(respAbs.At(k, 0, 0))
		relErr := math.Abs(magLFT-magAbs) / math.Max(magAbs, 1e-15)
		if relErr > 0.05 {
			t.Errorf("w=%v: LFT mag=%v, absorbed mag=%v, relErr=%v", w, magLFT, magAbs, relErr)
		}
	}
}

func TestFreqResponse_InternalDelay_D22Zero(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	err = sys.SetInternalDelay(
		[]float64{0.3},
		mat.NewDense(2, 1, []float64{0.5, 1}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 1.0, 10.0}
	respLFT, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respAbs, err := absorbed.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		magLFT := cmplx.Abs(respLFT.At(k, 0, 0))
		magAbs := cmplx.Abs(respAbs.At(k, 0, 0))
		relErr := math.Abs(magLFT-magAbs) / math.Max(magAbs, 1e-15)
		if relErr > 0.05 {
			t.Errorf("w=%v: LFT mag=%v, absorbed mag=%v, relErr=%v", w, magLFT, magAbs, relErr)
		}
	}
}

func TestFreqResponse_InternalDelay_MIMO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	err = sys.SetInternalDelay(
		[]float64{0.2, 0.5},
		mat.NewDense(2, 2, []float64{0.5, 0, 0, 0.3}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 1.0, 5.0}
	respLFT, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respAbs, err := absorbed.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		for i := range 2 {
			for j := range 2 {
				magLFT := cmplx.Abs(respLFT.At(k, i, j))
				magAbs := cmplx.Abs(respAbs.At(k, i, j))
				absErr := math.Abs(magLFT - magAbs)
				if absErr > 1e-6 && absErr/math.Max(magAbs, 1e-15) > 0.05 {
					t.Errorf("w=%v [%d,%d]: LFT=%v, absorbed=%v, absErr=%v",
						w, i, j, magLFT, magAbs, absErr)
				}
			}
		}
	}
}

func TestFreqResponse_InternalDelay_Discrete(t *testing.T) {
	dt := 1.0
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{0.1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}

	err = sys.SetInternalDelay(
		[]float64{2},
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	absorbed, err := sys.AbsorbDelay(AbsorbInternal)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 0.5, 1.0}
	respLFT, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respAbs, err := absorbed.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		magLFT := cmplx.Abs(respLFT.At(k, 0, 0))
		magAbs := cmplx.Abs(respAbs.At(k, 0, 0))
		relErr := math.Abs(magLFT-magAbs) / math.Max(magAbs, 1e-15)
		if relErr > 1e-10 {
			t.Errorf("w=%v: LFT=%v, absorbed=%v, relErr=%v", w, magLFT, magAbs, relErr)
		}
	}
}

func TestBode_InternalDelay(t *testing.T) {
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

	err = sys.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	bode, err := sys.Bode([]float64{0.1, 1.0, 10.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	if len(bode.Omega) != 3 {
		t.Fatalf("expected 3 freq points, got %d", len(bode.Omega))
	}

	for k := range bode.Omega {
		mag := bode.MagDBAt(k, 0, 0)
		if math.IsNaN(mag) || math.IsInf(mag, 0) {
			t.Errorf("w=%v: got NaN/Inf magnitude", bode.Omega[k])
		}
	}
}

func TestEvalFr_InternalDelay(t *testing.T) {
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

	err = sys.SetInternalDelay(
		[]float64{0.5},
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}

	w := 1.0
	evalRes, err := sys.EvalFr(complex(0, w))
	if err != nil {
		t.Fatal(err)
	}

	freqRes, err := sys.FreqResponse([]float64{w})
	if err != nil {
		t.Fatal(err)
	}

	diff := cmplx.Abs(evalRes[0][0] - freqRes.At(0, 0, 0))
	if diff > 1e-14 {
		t.Errorf("EvalFr=%v != FreqResponse=%v (diff=%v)", evalRes[0][0], freqRes.At(0, 0, 0), diff)
	}
}

func TestFreqResponse_IODelay_InputDelay(t *testing.T) {
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

	tau := 0.5
	err = sys.SetInputDelay([]float64{tau})
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{1.0}
	resp, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	s := complex(0, 1.0)
	wantNoDelay := 1 / (s + 1)
	want := wantNoDelay * cmplx.Exp(-s*complex(tau, 0))

	got := resp.At(0, 0, 0)
	if cmplx.Abs(got-want) > 1e-10 {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestFreqResponse_IODelay_OutputDelay(t *testing.T) {
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

	tau := 0.3
	err = sys.SetOutputDelay([]float64{tau})
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{2.0}
	resp, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	s := complex(0, 2.0)
	wantNoDelay := 1 / (s + 1)
	want := wantNoDelay * cmplx.Exp(-s*complex(tau, 0))

	got := resp.At(0, 0, 0)
	if cmplx.Abs(got-want) > 1e-10 {
		t.Errorf("got %v, want %v", got, want)
	}
}

func TestBode_SecondOrder(t *testing.T) {
	wn := 10.0
	zeta := 0.1
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

	omega := logspace(math.Log10(1), math.Log10(100), 500)
	bode, err := sys.Bode(omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	peakDB := math.Inf(-1)
	peakW := 0.0
	for k, w := range bode.Omega {
		if bode.MagDBAt(k, 0, 0) > peakDB {
			peakDB = bode.MagDBAt(k, 0, 0)
			peakW = w
		}
	}

	expectedPeak := wn * math.Sqrt(1-2*zeta*zeta)
	if math.Abs(peakW-expectedPeak)/expectedPeak > 0.05 {
		t.Errorf("resonance at w=%v, want ~%v", peakW, expectedPeak)
	}

	if peakDB < 10 {
		t.Errorf("peak mag = %v dB, expected significant resonance", peakDB)
	}
}

func TestEvalFr_IODelay_Continuous(t *testing.T) {
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
	sys.InputDelay = []float64{2.0}

	w := 5.0
	s := complex(0, w)
	val, err := sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}

	// Transfer function is 1/(s+1) * e^{-2s}
	tf := 1.0 / (s + 1)
	expected := tf * cmplx.Exp(-s*complex(2.0, 0))

	if cmplx.Abs(val[0][0]-expected) > 1e-10 {
		t.Errorf("EvalFr with I/O delay mismatch. got: %v, want: %v", val[0][0], expected)
	}
}

func TestEvalFr_IODelay_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.OutputDelay = []float64{3.0}

	w := 2.0
	s := cmplx.Exp(complex(0, w*sys.Dt))
	val, err := sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}

	// Transfer function is 1/(z-0.5) / z^3
	tf := 1.0 / (s - 0.5)
	expected := tf / (s * s * s)

	if cmplx.Abs(val[0][0]-expected) > 1e-10 {
		t.Errorf("EvalFr with I/O delay discrete mismatch. got: %v, want: %v", val[0][0], expected)
	}
}

func TestFreqResponse_MixedInternalAndIODelay(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(1, 2, []float64{1, 0.3}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	err = sys.SetInternalDelay(
		[]float64{0.3},
		mat.NewDense(2, 1, []float64{0.5, 0}),
		mat.NewDense(1, 2, []float64{0, 0.4}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
	)
	if err != nil {
		t.Fatal(err)
	}
	err = sys.SetInputDelay([]float64{0.2})
	if err != nil {
		t.Fatal(err)
	}

	merged, err := sys.PullDelaysToLFT()
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 0.5, 1.0, 5.0, 10.0}
	respOrig, err := sys.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}
	respMerged, err := merged.FreqResponse(freqs)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range freqs {
		got := respOrig.At(0, 0, k)
		want := respMerged.At(0, 0, k)
		if cmplx.Abs(got-want) > 1e-10 {
			t.Errorf("w=%v: direct=%v, merged=%v, diff=%v", w, got, want, cmplx.Abs(got-want))
		}
	}
}

func TestFreqResponse_SingularLFT(t *testing.T) {
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
	err = sys.SetInternalDelay(
		[]float64{1.0},
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
	)
	if err != nil {
		t.Fatal(err)
	}

	// I - D22·Δ is singular at w = 0, but the loop is decoupled from u and y
	// (D12 = D21 = B2 = C2 = 0), so G = 1/(s+1) stays finite there.
	resp, err := sys.FreqResponse([]float64{0.0})
	if err != nil {
		t.Fatal(err)
	}
	if g := resp.At(0, 0, 0); cmplx.Abs(g-1) > 1e-12 {
		t.Fatalf("G(0) = %v, want 1", g)
	}
}

func TestNichols_FirstOrderLowpass(t *testing.T) {
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

	r, err := sys.Nichols([]float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	magDB := r.MagDBAt(0, 0, 0)
	if math.Abs(magDB-(-3.0103)) > 0.01 {
		t.Errorf("mag = %v dB, want ~-3.01", magDB)
	}

	phase := r.PhaseAt(0, 0, 0)
	if math.Abs(phase-(-45)) > 0.5 {
		t.Errorf("phase = %v, want ~-45", phase)
	}
}

func TestNichols_PhaseRange(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-1000, -110, -11,
		},
		[]float64{0, 0, 1000},
		[]float64{1, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nichols(nil, 500)
	if err != nil {
		t.Fatal(err)
	}

	for k := range r.Omega {
		ph := r.PhaseAt(k, 0, 0)
		if ph > 0 || ph <= -360 {
			t.Errorf("w=%v: phase=%v outside (-360, 0]", r.Omega[k], ph)
			break
		}
	}
}

func TestNichols_MatchesBode(t *testing.T) {
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

	omega := logspace(-2, 2, 100)
	bode, err := sys.Bode(omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	nichols, err := sys.Nichols(omega, 0)
	if err != nil {
		t.Fatal(err)
	}

	for k := range omega {
		if bode.MagDBAt(k, 0, 0) != nichols.MagDBAt(k, 0, 0) {
			t.Errorf("w=%v: magDB mismatch", omega[k])
			break
		}
	}
}

func TestNichols_AutoFreq(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-10}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nichols(nil, 100)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Omega) != 100 {
		t.Fatalf("len(omega) = %d, want 100", len(r.Omega))
	}
}

func TestNichols_MIMO(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Nichols([]float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	mag00 := r.MagDBAt(0, 0, 0)
	mag11 := r.MagDBAt(0, 1, 1)
	want00 := 20 * math.Log10(1.0/math.Sqrt(2))
	want11 := 20 * math.Log10(1.0/math.Sqrt(5))

	if math.Abs(mag00-want00) > 0.1 {
		t.Errorf("H(0,0) mag = %v dB, want %v", mag00, want00)
	}
	if math.Abs(mag11-want11) > 0.1 {
		t.Errorf("H(1,1) mag = %v dB, want %v", mag11, want11)
	}
}

func TestNichols_Discrete(t *testing.T) {
	sysc, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	sysd, _ := sysc.Discretize(0.01)

	r, err := sysd.Nichols([]float64{0.1, 1.0, 10.0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for k := range r.Omega {
		if math.IsNaN(r.MagDBAt(k, 0, 0)) || math.IsInf(r.MagDBAt(k, 0, 0), 0) {
			t.Errorf("w=%v: NaN/Inf magnitude", r.Omega[k])
		}
	}
}

func TestSigma_SISO(t *testing.T) {
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

	freqs := []float64{0.01, 1.0, 10.0}
	r, err := sys.Sigma(freqs, 0)
	if err != nil {
		t.Fatal(err)
	}

	if r.NSV() != 1 {
		t.Fatalf("NSV = %d, want 1", r.NSV())
	}

	resp, _ := sys.FreqResponse(freqs)
	for k, w := range freqs {
		got := r.At(k, 0)
		want := cmplx.Abs(resp.At(k, 0, 0))
		if math.Abs(got-want) > 1e-10 {
			t.Errorf("w=%v: sigma=%v, want %v", w, got, want)
		}
	}
}

func TestSigma_DiagonalMIMO(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 1},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Sigma([]float64{0.001}, 0)
	if err != nil {
		t.Fatal(err)
	}

	if r.NSV() != 2 {
		t.Fatalf("NSV = %d, want 2", r.NSV())
	}

	sv0 := r.At(0, 0)
	sv1 := r.At(0, 1)
	if sv0 < sv1 {
		t.Errorf("SVs not descending: %v < %v", sv0, sv1)
	}

	if math.Abs(sv0-1.0) > 0.01 {
		t.Errorf("sv0 = %v, want ~1.0", sv0)
	}
	if math.Abs(sv1-0.5) > 0.01 {
		t.Errorf("sv1 = %v, want ~0.5", sv1)
	}
}

func TestSigma_NonSquare(t *testing.T) {
	sys, err := NewFromSlices(2, 3, 2,
		[]float64{-1, 0, 0, -2},
		[]float64{1, 0, 0, 0, 1, 0},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Sigma([]float64{1.0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if r.NSV() != 2 {
		t.Errorf("NSV = %d, want 2 (min(p=2, m=3))", r.NSV())
	}
	for i := 0; i < r.NSV(); i++ {
		if r.At(0, i) < 0 {
			t.Errorf("negative sv[%d] = %v", i, r.At(0, i))
		}
	}
}

func TestSigmaMatchesFRDForCoupledMIMOResponse(t *testing.T) {
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			0, 1, 0,
			-2, -3, 1,
			0.5, -0.25, -1.5,
		},
		[]float64{
			1, 0,
			0, 1,
			1, -1,
		},
		[]float64{
			1, 0.5, -0.25,
			0.2, 1, 0.75,
		},
		[]float64{
			0.1, -0.2,
			0.05, 0.3,
		},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := []float64{0.2, 1.3, 4.0}
	fromSystem, err := sys.Sigma(omega, 0)
	if err != nil {
		t.Fatal(err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		t.Fatal(err)
	}
	fromFRD, err := frd.Sigma()
	if err != nil {
		t.Fatal(err)
	}

	if fromSystem.NSV() != fromFRD.NSV() {
		t.Fatalf("NSV System=%d FRD=%d", fromSystem.NSV(), fromFRD.NSV())
	}
	for k := range omega {
		for sv := 0; sv < fromSystem.NSV(); sv++ {
			got := fromSystem.At(k, sv)
			want := fromFRD.At(k, sv)
			if math.Abs(got-want) > 1e-10*math.Max(1, math.Abs(want)) {
				t.Fatalf("omega[%d] sv[%d]: System.Sigma=%g FRD.Sigma=%g", k, sv, got, want)
			}
		}
	}
}

func TestSigmaClampsGramEigenvalueRelativeToScale(t *testing.T) {
	if got := nonnegativeGramEigenvalue(-1e3, 1e16); got != 0 {
		t.Fatalf("relative tiny negative eigenvalue = %g, want 0", got)
	}
	if got := nonnegativeGramEigenvalue(-1e3, 1); got != -1e3 {
		t.Fatalf("large negative eigenvalue = %g, want preserved negative", got)
	}
}

func TestSigma_MatchesHinfNorm(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{-1, 0.3, -0.5, -2},
		[]float64{1, 0, 0.5, 1},
		[]float64{1, 0.2, 0, 1},
		[]float64{0, 0, 0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	hinfNorm, _, err := HinfNorm(sys)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Sigma(nil, 500)
	if err != nil {
		t.Fatal(err)
	}

	peakSV := 0.0
	for k := range r.Omega {
		if r.At(k, 0) > peakSV {
			peakSV = r.At(k, 0)
		}
	}

	relErr := math.Abs(peakSV-hinfNorm) / hinfNorm
	if relErr > 0.05 {
		t.Errorf("peak sigma = %v, HinfNorm = %v, relErr = %v", peakSV, hinfNorm, relErr)
	}
}

func TestSigma_PureGain(t *testing.T) {
	D := mat.NewDense(2, 2, []float64{3, 0, 0, 4})
	sys, err := NewGain(D, 0)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Sigma([]float64{0.1, 1.0, 10.0}, 0)
	if err != nil {
		t.Fatal(err)
	}

	for k := range r.Omega {
		if math.Abs(r.At(k, 0)-4.0) > 1e-10 {
			t.Errorf("sv_max = %v, want 4.0", r.At(k, 0))
		}
		if math.Abs(r.At(k, 1)-3.0) > 1e-10 {
			t.Errorf("sv_min = %v, want 3.0", r.At(k, 1))
		}
	}
}

func TestSigma_SISO_MatchesBodeMag(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-5}),
		mat.NewDense(1, 1, []float64{5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	omega := logspace(-1, 2, 50)
	bode, _ := sys.Bode(omega, 0)
	sigma, _ := sys.Sigma(omega, 0)

	for k := range omega {
		svFromBode := math.Pow(10, bode.MagDBAt(k, 0, 0)/20)
		if math.Abs(sigma.At(k, 0)-svFromBode) > 1e-10 {
			t.Errorf("w=%v: sigma=%v, bode_lin=%v", omega[k], sigma.At(k, 0), svFromBode)
			break
		}
	}
}

func TestSigma_AutoFreq(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-10}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := sys.Sigma(nil, 50)
	if err != nil {
		t.Fatal(err)
	}
	if len(r.Omega) != 50 {
		t.Fatalf("len(omega) = %d, want 50", len(r.Omega))
	}
}

func TestAutoBodeFreqs_GridCoversDynamics(t *testing.T) {
	scaled := func(k float64) *System {
		a := []float64{-1, 2, 0.5, -0.3, -2, 1, 0.2, -0.4, -3}
		for i := range a {
			a[i] *= k
		}
		sys, _ := New(mat.NewDense(3, 3, a), mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, -1, 2}),
			mat.NewDense(2, 3, []float64{1, -1, 0.3, 0, 2, 1}), mat.NewDense(2, 2, []float64{0.1, 0, 0.2, -0.3}), 0)
		return sys
	}
	eigFreqs := func(sys *System) []float64 {
		var eig mat.Eigen
		eig.Factorize(sys.A, mat.EigenNone)
		var out []float64
		for _, p := range eig.Values(nil) {
			wn := cmplx.Abs(p)
			if sys.IsDiscrete() {
				wn = cmplx.Abs(cmplx.Log(p)) / sys.Dt
			}
			if wn > 0 && !math.IsInf(wn, 0) {
				out = append(out, wn)
			}
		}
		return out
	}

	leadLag, _ := New(mat.NewDense(2, 2, []float64{0, 1, -2, -3}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 27.1}), mat.NewDense(1, 1, []float64{1}), 0)
	delayed := scaled(1)
	delayed.InputDelay = []float64{1e-4, 0}
	zPole, _ := New(mat.NewDense(2, 2, []float64{1, 1, 0, 0}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{0.3, 0}), mat.NewDense(1, 1, nil), 0.1)
	pureDelay, _ := New(mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0.1)
	discMIMO, _ := New(mat.NewDense(3, 3, []float64{0.9, 0.2, 0, -0.1, 0.5, 0.3, 0, 0, 0}),
		mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, 0, 1}), mat.NewDense(2, 3, []float64{1, -1, 0.3, 0, 2, 1}),
		mat.NewDense(2, 2, []float64{0.1, 0, 0.2, -0.3}), 1e-3)
	discDelay, _ := New(mat.NewDense(1, 1, []float64{0.999}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0.01)
	discDelay.InputDelay = []float64{1000}

	cases := []struct {
		name     string
		sys      *System
		features []float64
	}{
		{"scaled 1e-5", scaled(1e-5), nil},
		{"scaled 1", scaled(1), nil},
		{"scaled 1e5", scaled(1e5), nil},
		{"zeros 0.1, 30", leadLag, []float64{0.1, 30}},
		{"input delay 1e-4", delayed, []float64{1e4}},
		{"discrete z=0 pole", zPole, nil},
		{"discrete pure delay", pureDelay, nil},
		{"discrete MIMO z=0 pole", discMIMO, nil},
		{"discrete 10s delay", discDelay, []float64{0.1}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.sys
			features := append(eigFreqs(sys), tc.features...)
			bode, err := sys.Bode(nil, 50)
			if err != nil {
				t.Fatal(err)
			}
			w := bode.Omega
			for k, v := range w {
				if math.IsNaN(v) || math.IsInf(v, 0) || v <= 0 {
					t.Fatalf("omega[%d] = %g", k, v)
				}
				if k > 0 && v <= w[k-1] {
					t.Fatalf("omega not increasing at %d: %g <= %g", k, v, w[k-1])
				}
			}
			wMin, wMax := w[0], w[len(w)-1]
			nyq := math.Pi / sys.Dt
			if sys.IsDiscrete() && wMax != nyq {
				t.Errorf("last omega = %g, want π/dt = %g", wMax, nyq)
			}
			for _, wn := range features {
				if wMin > wn/10*(1+1e-12) {
					t.Errorf("wMin = %g above feature %g/10", wMin, wn)
				}
				if sys.IsContinuous() && wMax < wn*10*(1-1e-12) {
					t.Errorf("wMax = %g below 10×feature %g", wMax, wn)
				}
			}
			sg, err := sys.Sigma(nil, 50)
			if err != nil {
				t.Fatal(err)
			}
			nc, err := sys.Nichols(nil, 50)
			if err != nil {
				t.Fatal(err)
			}
			for k := range w {
				if sg.Omega[k] != w[k] || nc.Omega[k] != w[k] {
					t.Fatalf("Sigma/Nichols grid differs from Bode at %d", k)
				}
			}
		})
	}
}
