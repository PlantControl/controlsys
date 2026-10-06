package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"strings"
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
	sysd, err := sysc.C2D(dt, C2DOptions{Method: C2DMethodTustin})
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

	siso := benchDenseSys(t, 24, 1, 1)
	siso.InputDelay = []float64{0.05}
	siso.OutputDelay = []float64{0.02}
	discreteSISO := siso.Copy()
	discreteSISO.Dt = 0.1
	discreteSISO.InputDelay = []float64{2}
	discreteSISO.OutputDelay = []float64{1}

	type sweepCase struct {
		name      string
		system    *System
		omega     []float64
		wantDense bool
	}
	tests := []sweepCase{
		{name: "ShortSweep", system: siso, omega: logspace(-2, 2, 2), wantDense: true},
		{name: "SmallModel", system: sys, omega: logspace(-2, 2, 100), wantDense: true},
		{name: "DiscreteSmallModel", system: discrete, omega: logspace(-2, 1, 100), wantDense: true},
		{name: "Hessenberg", system: siso, omega: logspace(-2, 2, 100), wantDense: false},
		{name: "DiscreteHessenberg", system: discreteSISO, omega: logspace(-2, 1, 100), wantDense: false},
		{name: "CoupledMIMO", system: benchDenseSys(t, 30, 2, 2), omega: logspace(-2, 2, 100), wantDense: false},
		{name: "UpperHessenbergA", system: benchSys(t, 40, 1, 1), omega: logspace(-2, 2, 100), wantDense: true},
		{name: "SISOBelowGridCrossover", system: siso, omega: logspace(-2, 2, lastDenseGrid[[2]int{24, 1}]), wantDense: true},
		{name: "SISOAtGridCrossover", system: siso, omega: logspace(-2, 2, lastDenseGrid[[2]int{24, 1}]+1), wantDense: false},
		{name: "ThreeQuarterCLongGrid", system: benchDenseSys(t, 12, 1, 1), omega: logspace(-2, 2, 200), wantDense: threeQuarterCLongGridDense},
	}
	for _, size := range [][2]int{{16, 1}, {44, 4}, {80, 4}} {
		sys := benchDenseSys(t, size[0], size[1], size[1])
		nw := lastDenseGrid[size]
		name := fmt.Sprintf("n=%d/m=%d", size[0], size[1])
		tests = append(tests,
			sweepCase{name: name + "/LastDense", system: sys, omega: logspace(-2, 2, nw), wantDense: true},
			sweepCase{name: name + "/FirstHessenberg", system: sys, omega: logspace(-2, 2, nw+1), wantDense: false},
		)
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
				want, err := evalFrAtPoint(test.system, evaluator.pointAt(w))
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

func TestUseDenseSweepUnboundedGrid(t *testing.T) {
	if !newFrequencyEvaluator(benchDenseSys(t, 8, 1, 1)).useDenseSweep(math.MaxInt) {
		t.Error("n = c/2: want dense for any grid")
	}
	if newFrequencyEvaluator(benchDenseSys(t, 80, 4, 4)).useDenseSweep(math.MaxInt) {
		t.Error("n = 2c: want Hessenberg for an unbounded grid")
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
	if !errors.Is(err, ErrInvalidArgument) || resp != nil {
		t.Fatalf("FreqResponse(empty) = %v, %v; want nil, ErrInvalidArgument", resp, err)
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
	sysd, _ := sysc.C2D(0.01, C2DOptions{Method: C2DMethodTustin})

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

func TestDefaultFrequencyGrid(t *testing.T) {
	mimo, err := New(
		mat.NewDense(3, 3, []float64{-1, 2, 0, -0.5, -3, 1, 0.2, 0, -10}),
		mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, 1}),
		mat.NewDense(2, 3, []float64{1, 0, 1, 0, 1, -1}),
		mat.NewDense(2, 2, []float64{0.1, 0, 0.2, 0.3}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	lag, err := NewWithDelay(
		mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0.1}), 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	dmimo, err := New(
		mat.NewDense(2, 2, []float64{0.5, 0.2, -0.1, 0.8}),
		mat.NewDense(2, 2, []float64{1, 0, 0.5, 1}),
		mat.NewDense(2, 2, []float64{1, 1, 0, 1}),
		mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	dInternal, err := New(
		mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	if err := dInternal.SetInternalDelay([]float64{2},
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.5}), mat.NewDense(1, 1, []float64{0.3}),
		mat.NewDense(1, 1, []float64{0})); err != nil {
		t.Fatal(err)
	}
	pureDelay, err := New(
		mat.NewDense(2, 2, []float64{0, 1, 0, 0}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0}), 0.1,
	)
	if err != nil {
		t.Fatal(err)
	}
	gain, err := NewGain(mat.NewDense(2, 1, []float64{2, -1}), 0)
	if err != nil {
		t.Fatal(err)
	}

	cases := []struct {
		name       string
		sys        *System
		wMin, wMax float64
	}{
		{"continuous MIMO", mimo, math.NaN(), math.NaN()},
		{"continuous IO delay", lag, 0.1, 100},
		{"discrete MIMO", dmimo, math.NaN(), math.Pi / 0.1},
		{"discrete internal delay", dInternal, math.Pi / 0.1 / 100, math.Pi / 0.1},
		{"discrete poles at z=0", pureDelay, 0.01, math.Pi / 0.1},
		{"static gain", gain, 0.01, 100},
	}
	for _, tc := range cases {
		for _, n := range []int{0, 1, 7, 200} {
			omega, err := tc.sys.DefaultFrequencyGrid(n)
			if err != nil {
				t.Fatalf("%s n=%d: %v", tc.name, n, err)
			}
			want := n
			if n <= 0 {
				want = 200
			}
			if len(omega) != want {
				t.Fatalf("%s n=%d: len %d, want %d", tc.name, n, len(omega), want)
			}
			for k := 1; k < len(omega); k++ {
				if !(omega[k] > omega[k-1]) {
					t.Fatalf("%s n=%d: not increasing at %d", tc.name, n, k)
				}
			}
			if !math.IsNaN(tc.wMin) && math.Abs(omega[0]-tc.wMin) > 1e-12*tc.wMin {
				t.Errorf("%s n=%d: first %v, want %v", tc.name, n, omega[0], tc.wMin)
			}
			if n != 1 && !math.IsNaN(tc.wMax) {
				last := omega[len(omega)-1]
				if tc.sys.IsDiscrete() && last != tc.wMax {
					t.Errorf("%s n=%d: last %v, want exactly %v", tc.name, n, last, tc.wMax)
				} else if math.Abs(last-tc.wMax) > 1e-12*tc.wMax {
					t.Errorf("%s n=%d: last %v, want %v", tc.name, n, last, tc.wMax)
				}
			}

			bode, err := tc.sys.Bode(nil, n)
			if err != nil {
				t.Fatalf("%s n=%d Bode: %v", tc.name, n, err)
			}
			sigma, err := tc.sys.Sigma(nil, n)
			if err != nil {
				t.Fatalf("%s n=%d Sigma: %v", tc.name, n, err)
			}
			nichols, err := tc.sys.Nichols(nil, n)
			if err != nil {
				t.Fatalf("%s n=%d Nichols: %v", tc.name, n, err)
			}
			for name, got := range map[string][]float64{"Bode": bode.Omega, "Sigma": sigma.Omega, "Nichols": nichols.Omega} {
				if len(got) != len(omega) {
					t.Fatalf("%s n=%d %s: len %d, want %d", tc.name, n, name, len(got), len(omega))
				}
				for k := range omega {
					if got[k] != omega[k] {
						t.Fatalf("%s n=%d %s: omega[%d]=%v, grid %v", tc.name, n, name, k, got[k], omega[k])
					}
				}
			}
		}
	}

	a, _ := lag.DefaultFrequencyGrid(5)
	a[0] = -1
	b, _ := lag.DefaultFrequencyGrid(5)
	if b[0] == -1 {
		t.Error("DefaultFrequencyGrid returned a shared slice")
	}
}

func TestDefaultFrequencyGridInvalid(t *testing.T) {
	var nilSys *System
	if _, err := nilSys.DefaultFrequencyGrid(10); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil system: err = %v, want ErrInvalidArgument", err)
	}
	if _, err := nilSys.Bode(nil, 10); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil system Bode: err = %v, want ErrInvalidArgument", err)
	}
}

func idiomMIMO(t *testing.T, dt float64) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(2, 2, []float64{-0.5, 0.3, -0.2, -0.4}),
		mat.NewDense(2, 2, []float64{1, 0.5, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0.2, 1}),
		mat.NewDense(2, 2, []float64{0.1, 0, 0, 0.2}),
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestFrequencyOpsRejectEmptyOmega(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := idiomMIMO(t, dt)
		empty := []float64{}
		ops := map[string]func() (any, error){
			"FreqResponse":          func() (any, error) { return sys.FreqResponse(empty) },
			"FreqResponsePointwise": func() (any, error) { return sys.FreqResponsePointwise(empty) },
			"Bode":                  func() (any, error) { return sys.Bode(empty, 0) },
			"Nichols":               func() (any, error) { return sys.Nichols(empty, 0) },
			"Sigma":                 func() (any, error) { return sys.Sigma(empty, 0) },
			"FRD":                   func() (any, error) { return sys.FRD(empty) },
		}
		for name, op := range ops {
			_, err := op()
			if !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("dt=%g %s(empty): err = %v, want ErrInvalidArgument", dt, name, err)
			}
		}
		if b, err := sys.Bode(nil, 7); err != nil || len(b.Omega) != 7 {
			t.Errorf("dt=%g Bode(nil): err=%v", dt, err)
		}
		if r, err := sys.Nichols(nil, 7); err != nil || len(r.Omega) != 7 {
			t.Errorf("dt=%g Nichols(nil): err=%v", dt, err)
		}
		if r, err := sys.Sigma(nil, 7); err != nil || len(r.Omega) != 7 {
			t.Errorf("dt=%g Sigma(nil): err=%v", dt, err)
		}
	}
}

func TestSigmaRejectsModelWithoutSingularValues(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		nil, nil, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := sys.Sigma([]float64{1}, 0); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Sigma on 0x1 model: err = %v, want ErrDimensionMismatch", err)
	}
}

func TestFrequencyResultsOwnOmega(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := idiomMIMO(t, dt)
		w := []float64{1, 2}
		resp, err := sys.FreqResponse(w)
		if err != nil {
			t.Fatal(err)
		}
		pw, err := sys.FreqResponsePointwise(w)
		if err != nil {
			t.Fatal(err)
		}
		b, err := sys.Bode(w, 0)
		if err != nil {
			t.Fatal(err)
		}
		nic, err := sys.Nichols(w, 0)
		if err != nil {
			t.Fatal(err)
		}
		sg, err := sys.Sigma(w, 0)
		if err != nil {
			t.Fatal(err)
		}
		w[0] = 99
		for name, got := range map[string][]float64{
			"FreqResponse": resp.Omega, "FreqResponsePointwise": pw.Omega,
			"Bode": b.Omega, "Nichols": nic.Omega, "Sigma": sg.Omega,
		} {
			if got[0] != 1 {
				t.Errorf("dt=%g %s.Omega aliases input: %v", dt, name, got)
			}
		}
	}
}

func TestFreqResponseInternalDelayOwnsOmega(t *testing.T) {
	sys := idiomMIMO(t, 0)
	if err := sys.SetInternalDelay([]float64{0.3},
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(2, 1, nil), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	w := []float64{1, 2}
	resp, err := sys.FreqResponse(w)
	if err != nil {
		t.Fatal(err)
	}
	w[0] = 99
	if resp.Omega[0] != 1 {
		t.Errorf("internal-delay FreqResponse.Omega aliases input: %v", resp.Omega)
	}
}

// frBig is a complex number in frBigPrec-bit floating point.
type frBig struct{ re, im *big.Float }

const frBigPrec = 200

func frBigF(v float64) *big.Float { return new(big.Float).SetPrec(frBigPrec).SetFloat64(v) }

func frBigC(v complex128) frBig { return frBig{frBigF(real(v)), frBigF(imag(v))} }

func (a frBig) add(b frBig) frBig {
	return frBig{frBigF(0).Add(a.re, b.re), frBigF(0).Add(a.im, b.im)}
}

func (a frBig) sub(b frBig) frBig {
	return frBig{frBigF(0).Sub(a.re, b.re), frBigF(0).Sub(a.im, b.im)}
}

func (a frBig) mul(b frBig) frBig {
	rr, ii := frBigF(0).Mul(a.re, b.re), frBigF(0).Mul(a.im, b.im)
	ri, ir := frBigF(0).Mul(a.re, b.im), frBigF(0).Mul(a.im, b.re)
	return frBig{rr.Sub(rr, ii), ri.Add(ri, ir)}
}

func (a frBig) quo(b frBig) frBig {
	den := frBigF(0).Mul(b.re, b.re)
	den.Add(den, frBigF(0).Mul(b.im, b.im))
	q := a.mul(frBig{b.re, frBigF(0).Neg(b.im)})
	return frBig{q.re.Quo(q.re, den), q.im.Quo(q.im, den)}
}

func (a frBig) abs2() *big.Float {
	r := frBigF(0).Mul(a.re, a.re)
	return r.Add(r, frBigF(0).Mul(a.im, a.im))
}

func (a frBig) complex() complex128 {
	re, _ := a.re.Float64()
	im, _ := a.im.Float64()
	return complex(re, im)
}

// frExactPoint is the evaluation point at w in frBigPrec bits: jω, or for
// discrete models z = h/conj(h) with h = cos(ωT/2) + j·sin(ωT/2) rounded to
// float64 as timeDomain.frequencyPoint rounds it, which lies on the unit
// circle.
func frExactPoint(w, dt float64) frBig {
	if dt == 0 {
		return frBigC(complex(0, w))
	}
	sn, cs := math.Sincos(w * dt / 2)
	return frBigC(complex(cs, sn)).quo(frBigC(complex(cs, -sn)))
}

// frExactResponse is G(z) = C(zE − A)⁻¹B + D of the stored realization with
// only the final G rounded: Gaussian elimination in frBigPrec bits.
func frExactResponse(sys *System, z frBig) []complex128 {
	gb := frExactResponseBig(sys, z)
	g := make([]complex128, len(gb))
	for i, v := range gb {
		g[i] = v.complex()
	}
	return g
}

// frExactResponseBig is frExactResponse unrounded, row-major p×m.
func frExactResponseBig(sys *System, z frBig) []frBig {
	n, m, p := sys.Dims()
	M := make([][]frBig, n)
	for i := range n {
		M[i] = make([]frBig, n+m)
		for j := range n {
			e := 0.0
			if sys.E != nil {
				e = sys.E.At(i, j)
			} else if i == j {
				e = 1
			}
			M[i][j] = z.mul(frBigC(complex(e, 0))).sub(frBigC(complex(sys.A.At(i, j), 0)))
		}
		for j := range m {
			M[i][n+j] = frBigC(complex(sys.B.At(i, j), 0))
		}
	}
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if M[i][k].abs2().Cmp(M[piv][k].abs2()) > 0 {
				piv = i
			}
		}
		M[k], M[piv] = M[piv], M[k]
		for i := k + 1; i < n; i++ {
			f := M[i][k].quo(M[k][k])
			for j := k; j < n+m; j++ {
				M[i][j] = M[i][j].sub(f.mul(M[k][j]))
			}
		}
	}
	X := make([][]frBig, n)
	for i := n - 1; i >= 0; i-- {
		X[i] = make([]frBig, m)
		for j := range m {
			acc := M[i][n+j]
			for k := i + 1; k < n; k++ {
				acc = acc.sub(M[i][k].mul(X[k][j]))
			}
			X[i][j] = acc.quo(M[i][i])
		}
	}
	G := make([]frBig, p*m)
	for i := range p {
		for j := range m {
			acc := frBigC(complex(sys.D.At(i, j), 0))
			for k := range n {
				acc = acc.add(frBigC(complex(sys.C.At(i, k), 0)).mul(X[k][j]))
			}
			G[i*m+j] = acc
		}
	}
	return G
}

// lightlyDampedDiscreteSys is a 3×2 discrete model (Dt = 0.1) with D ≠ 0, a
// pole pair at radius 1−gap and angle θ for each theta, extra well-damped
// pairs, a real pole at −(1−gap) and a stable real pole, coupled upper
// triangularly so A is non-symmetric and upper Hessenberg. With extra > 0
// the first state also feeds the last, stable one, so A is not upper
// Hessenberg while its poles stay those of the diagonal blocks. With
// descriptor set, the model is (E, EA, EB, C) for a power-of-two diagonal E,
// which keeps the poles exact.
func lightlyDampedDiscreteSys(t *testing.T, rng *rand.Rand, thetas []float64, gap float64, extra int, descriptor bool) *System {
	t.Helper()
	n := 2*(len(thetas)+extra) + 2
	m, p := 2, 3
	A := mat.NewDense(n, n, nil)
	for k := range len(thetas) + extra {
		r, th := 1-gap, 0.0
		if k < len(thetas) {
			th = thetas[k]
		} else {
			r, th = 0.8, 3*rng.Float64()
		}
		c, s := r*math.Cos(th), r*math.Sin(th)
		A.Set(2*k, 2*k, c)
		A.Set(2*k, 2*k+1, s*1.5)
		A.Set(2*k+1, 2*k, -s/1.5)
		A.Set(2*k+1, 2*k+1, c)
	}
	A.Set(n-2, n-2, -(1 - gap))
	A.Set(n-1, n-1, 0.4)
	if extra > 0 {
		A.Set(n-1, 0, 0.7)
	}
	for i := range n {
		for j := i + 1; j < n-1; j++ {
			if j > i+1 || i%2 == 1 {
				A.Set(i, j, 0.3*rng.NormFloat64())
			}
		}
	}
	B := mat.NewDense(n, m, nil)
	C := mat.NewDense(p, n, nil)
	D := mat.NewDense(p, m, nil)
	for i := range n {
		for j := range m {
			B.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			C.Set(j, i, rng.NormFloat64())
		}
	}
	for i := range p {
		for j := range m {
			D.Set(i, j, 0.5+rng.Float64())
		}
	}
	var sys *System
	var err error
	if descriptor {
		E := mat.NewDense(n, n, nil)
		for i := range n {
			E.Set(i, i, math.Pow(2, float64(rng.IntN(5)-2)))
		}
		A.Mul(E, mat.DenseCopyOf(A))
		B.Mul(E, mat.DenseCopyOf(B))
		sys, err = NewDescriptor(A, B, C, D, E, 0.1)
	} else {
		sys, err = New(A, B, C, D, 0.1)
	}
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// Next to a pole at distance 1e-9 from the unit circle, a z = e^{jωT}
// rounded up to ε off the circle moves |G| by about ε/1e-9 ≈ 1e-7 relative.
// The response is held, over frequencies a few ulps apart around each
// peak, to a 200-bit evaluation exactly on the circle. |G| (Frobenius) is
// the measure: a frequency perturbation of one ulp moves the phase of G by
// ε/1e-9 as well, but leaves |G| flat at the peak. The exact point removes
// the radial error; the rounded pencil entries and GEPP's rounding, relative
// to O(1) pivoted rows, would still leave ε/1e-9, which refinement against
// the exact pencil removes: for the upper Hessenberg n = 8 model, the
// coupled n = 10 model whose coupling row GEPP pivots into the resonant
// rows, and the n = 32 model a long sweep evaluates by the Hessenberg
// kernel, at small ωT, mid-band and near Nyquist.
func TestFreqResponseDiscreteLightlyDampedPeaks(t *testing.T) {
	const (
		dt  = 0.1
		gap = 1e-9
	)
	rng := rand.New(rand.NewPCG(7, 9))
	thetas := []float64{0.003, 0.03, 1.4, 3.11}
	frob := func(g []complex128) float64 {
		v := 0.0
		for _, x := range g {
			v += real(x)*real(x) + imag(x)*imag(x)
		}
		return math.Sqrt(v)
	}
	var omega []float64
	for _, th := range append(thetas, math.Pi) {
		w0 := th / dt
		for k := -6; k <= 6; k++ {
			if w := w0 * (1 + float64(k)*0x1p-52); w <= math.Pi/dt {
				omega = append(omega, w)
			}
		}
	}
	for _, descriptor := range []bool{false, true} {
		for _, tc := range []struct {
			extra     int
			tol       float64
			pointwise bool
		}{{0, 1e-13, true}, {1, 1e-13, true}, {12, 1e-13, true}} {
			sys := lightlyDampedDiscreteSys(t, rng, thetas, gap, tc.extra, descriptor)
			n, _, _ := sys.Dims()
			resps := map[string]*FreqResponseMatrix{}
			var err error
			if resps["FreqResponse"], err = sys.FreqResponse(omega); err != nil {
				t.Fatal(err)
			}
			if tc.pointwise {
				if resps["FreqResponsePointwise"], err = sys.FreqResponsePointwise(omega); err != nil {
					t.Fatal(err)
				}
			}
			for k, w := range omega {
				want := frob(frExactResponse(sys, frExactPoint(w, dt)))
				for name, resp := range resps {
					got := frob(resp.Data[k*6 : (k+1)*6])
					if e := math.Abs(got-want) / want; !(e <= tc.tol) {
						t.Errorf("descriptor=%v n=%d %s ω=%v (ωT=%.4g): |G| = %.17g, exact %.17g, rel err %.3g",
							descriptor, n, name, w, w*dt, got, want, e)
					}
				}
			}
		}
	}
}

func TestComplexSolveErrorsHaveNoDoublePrefix(t *testing.T) {
	err := cInvertInto(make([]complex128, 4), make([]complex128, 8), []complex128{1, 2, 2, 4}, 2)
	if !errors.Is(err, ErrSingularTransform) || strings.Count(err.Error(), "controlsys:") != 1 {
		t.Errorf("cInvertInto singular: err = %v", err)
	}
}

// Next to a pole of a discrete delay loop within 1e-9 of the unit circle,
// I − H22·Δ is nearly singular, and a rounded Δ = z^{-k} (k-fold rounding,
// and no complex128 lies on the circle) moves G by about kε/1e-9. The
// response must match a 200-bit closed loop of the exact H(z) and Δ =
// z^{-k} at the exact on-circle point. Delay 1 (3 samples) closes a static
// loop with gain 1 − 1e-9, delay 2 (5 samples) a dynamic one through a
// non-symmetric 3-state plant; H22 is lower triangular so the poles of the
// first loop are exact. The descriptor variant is (E, EA, EB, EB2) for a
// power-of-two diagonal E.
func TestFreqResponseDiscreteInternalDelayLightlyDamped(t *testing.T) {
	const dt, a = 0.1, 1 - 1e-9
	var omega []float64
	for _, th := range []float64{1e-7, 2 * math.Pi / 3, 4 * math.Pi / 3, 3} {
		for k := -3; k <= 3; k++ {
			omega = append(omega, th/dt*(1+float64(k)*0x1p-52))
		}
	}
	for _, descriptor := range []bool{false, true} {
		A := mat.NewDense(3, 3, []float64{0.5, 0.2, -0.1, -0.3, 0.4, 0.25, 0.1, -0.2, -0.6})
		B := mat.NewDense(3, 4, []float64{1, 0.3, 0, 0.3, -0.4, 0.8, 0, -0.2, 0.2, -0.5, 0, 0.4})
		C := mat.NewDense(4, 3, []float64{0.7, -0.2, 0.4, 0.1, 0.9, -0.3, 0, 0, 0, 0.5, 0.1, -0.3})
		D := mat.NewDense(4, 4, []float64{0.6, 0.1, 0.8, -0.3, -0.2, 0.9, 0.2, 0.5, 0.4, -0.6, a, 0, 0.3, 0.2, 0.35, 0.3})
		var E *mat.Dense
		if descriptor {
			E = mat.NewDense(3, 3, []float64{4, 0, 0, 0, 0.5, 0, 0, 0, 2})
			A.Mul(E, mat.DenseCopyOf(A))
			B.Mul(E, mat.DenseCopyOf(B))
		}
		plant, err := NewDescriptor(A, B, C, D, E, dt)
		if err != nil {
			t.Fatal(err)
		}
		sys, err := NewDescriptor(A, B.Slice(0, 3, 0, 2).(*mat.Dense), C.Slice(0, 2, 0, 3).(*mat.Dense), D.Slice(0, 2, 0, 2).(*mat.Dense), E, dt)
		if err != nil {
			t.Fatal(err)
		}
		sys.LFT = &LFTDelay{
			Tau: []float64{3, 5},
			B2:  mat.DenseCopyOf(B.Slice(0, 3, 2, 4)),
			C2:  mat.DenseCopyOf(C.Slice(2, 4, 0, 3)),
			D12: mat.DenseCopyOf(D.Slice(0, 2, 2, 4)),
			D21: mat.DenseCopyOf(D.Slice(2, 4, 0, 2)),
			D22: mat.DenseCopyOf(D.Slice(2, 4, 2, 4)),
		}
		resps := map[string]*FreqResponseMatrix{}
		if resps["FreqResponse"], err = sys.FreqResponse(omega); err != nil {
			t.Fatal(err)
		}
		if resps["FreqResponsePointwise"], err = sys.FreqResponsePointwise(omega); err != nil {
			t.Fatal(err)
		}
		for k, w := range omega {
			z := frExactPoint(w, dt)
			h := frExactResponseBig(plant, z)
			d := []frBig{frBigC(1), frBigC(1)}
			for j, tau := range []int{3, 5} {
				for range tau {
					d[j] = d[j].quo(z)
				}
			}
			// I − H22·Δ is lower triangular: x = (I − H22Δ)⁻¹H21 by substitution.
			l00 := frBigC(1).sub(h[2*4+2].mul(d[0]))
			l10 := frBigC(0).sub(h[3*4+2].mul(d[0]))
			l11 := frBigC(1).sub(h[3*4+3].mul(d[1]))
			var norm, diff float64
			for j := range 2 {
				x0 := h[2*4+j].quo(l00)
				x1 := h[3*4+j].sub(l10.mul(x0)).quo(l11)
				for i := range 2 {
					g := h[i*4+j].add(h[i*4+2].mul(d[0]).mul(x0)).add(h[i*4+3].mul(d[1]).mul(x1)).complex()
					norm = max(norm, cmplx.Abs(g))
					for _, resp := range resps {
						diff = max(diff, cmplx.Abs(resp.Data[k*4+i*2+j]-g))
					}
				}
			}
			if e := diff / norm; !(e <= 1e-13) {
				t.Errorf("descriptor=%v ω=%v (ωT=%.4g): max entry error %.3g relative to max |G|", descriptor, w, w*dt, e)
			}
		}
	}
}

// The absorbed fallback is bounded: a long delay is not absorbed.
func TestLFTAbsorbedSolverStateCap(t *testing.T) {
	sys := makeSISO(0.5, 1, 1, 0)
	sys.Dt = 0.1
	sys.LFT = &LFTDelay{Tau: []float64{maxAbsorbedStates},
		B2: mat.NewDense(1, 1, []float64{0}), C2: mat.NewDense(1, 1, []float64{0}),
		D12: mat.NewDense(1, 1, []float64{1}), D21: mat.NewDense(1, 1, []float64{1}), D22: mat.NewDense(1, 1, []float64{0.5})}
	if newLFTWorkspace(sys, 1, 1, 1, 1).absorbedSolver(sys) != nil {
		t.Errorf("absorbed %d+1 states, cap %d", maxAbsorbedStates, maxAbsorbedStates)
	}
	sys.LFT.Tau[0] = maxAbsorbedStates - 1
	if newLFTWorkspace(sys, 1, 1, 1, 1).absorbedSolver(sys) == nil {
		t.Errorf("did not absorb %d states", maxAbsorbedStates)
	}
}
