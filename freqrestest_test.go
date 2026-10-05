package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/dsp/window"
	"plantcontrol.org/v1/gonum/mat"
)

func TestFreqRespEst_KnownSISO(t *testing.T) {
	// Discrete first-order lowpass: A=0.9, B=0.1, C=1, D=0, dt=0.01
	dt := 0.01
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

	N := 4096
	rng := rand.New(rand.NewSource(42))
	uData := make([]float64, N)
	for i := range uData {
		uData[i] = rng.NormFloat64()
	}
	u := mat.NewDense(1, N, uData)

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	est, err := FreqRespEst(u, resp.Y, dt, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Compare at mid frequencies (skip DC and near-Nyquist)
	nFreq := len(est.Omega)
	skipLow := nFreq / 10
	skipHigh := nFreq * 9 / 10

	for f := skipLow; f < skipHigh; f++ {
		w := est.Omega[f]
		z := cmplx.Exp(complex(0, w*dt))
		hTrue := complex(0.1, 0) / (z - 0.9)

		hEst := est.H.At(f, 0, 0)
		relErr := cmplx.Abs(hEst-hTrue) / cmplx.Abs(hTrue)
		if relErr > 0.15 {
			t.Errorf("w=%.2f: relErr=%.3f, est=%v, true=%v", w, relErr, hEst, hTrue)
		}
	}
}

func TestFreqRespEst_Coherence(t *testing.T) {
	dt := 0.01
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

	N := 4096
	rng := rand.New(rand.NewSource(123))
	uData := make([]float64, N)
	for i := range uData {
		uData[i] = rng.NormFloat64()
	}
	u := mat.NewDense(1, N, uData)

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	// No noise: coherence should be near 1
	est, err := FreqRespEst(u, resp.Y, dt, nil)
	if err != nil {
		t.Fatal(err)
	}

	nFreq := len(est.Omega)
	for f := nFreq / 10; f < nFreq*9/10; f++ {
		coh := est.CoherenceAt(f, 0, 0)
		if coh < 0.95 {
			t.Errorf("w=%.2f: coherence=%.3f, want >0.95", est.Omega[f], coh)
			break
		}
	}
}

func TestFreqRespEst_EdgeCases(t *testing.T) {
	_, err := FreqRespEst(nil, nil, 0.01, nil)
	if err == nil {
		t.Error("expected error for nil input")
	}

	u := mat.NewDense(1, 10, nil)
	y := mat.NewDense(1, 5, nil)
	_, err = FreqRespEst(u, y, 0.01, nil)
	if err == nil {
		t.Error("expected error for mismatched lengths")
	}

	y2 := mat.NewDense(1, 10, nil)
	_, err = FreqRespEst(u, y2, 0, nil)
	if err == nil {
		t.Error("expected error for dt=0")
	}

	_, err = FreqRespEst(u, y2, -1, nil)
	if err == nil {
		t.Error("expected error for dt<0")
	}

	if _, err := FreqRespEst(&mat.Dense{}, &mat.Dense{}, 0.01, nil); !errors.Is(err, ErrInsufficientData) {
		t.Fatalf("empty matrices: got %v, want ErrInsufficientData", err)
	}
}

func TestFreqRespEst_WindowOptions(t *testing.T) {
	dt := 0.01
	N := 512
	rng := rand.New(rand.NewSource(99))
	uData := make([]float64, N)
	yData := make([]float64, N)
	for i := range N {
		uData[i] = rng.NormFloat64()
		yData[i] = 0.5 * uData[i]
	}
	u := mat.NewDense(1, N, uData)
	y := mat.NewDense(1, N, yData)

	for _, wf := range []struct {
		name string
		fn   func([]float64) []float64
	}{
		{"Rectangular", window.Rectangular},
		{"Blackman", window.Blackman},
		{"Hamming", window.Hamming},
	} {
		t.Run(wf.name, func(t *testing.T) {
			est, err := FreqRespEst(u, y, dt, &FreqRespEstOpts{Window: wf.fn})
			if err != nil {
				t.Fatal(err)
			}
			for f := range est.Omega {
				h := est.H.At(f, 0, 0)
				if math.IsNaN(real(h)) || math.IsInf(real(h), 0) {
					t.Errorf("NaN/Inf at f=%d", f)
					break
				}
			}
		})
	}
}

func TestFreqRespEst_FFTMethod(t *testing.T) {
	dt := 0.01
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

	N := 1024
	rng := rand.New(rand.NewSource(55))
	uData := make([]float64, N)
	for i := range uData {
		uData[i] = rng.NormFloat64()
	}
	u := mat.NewDense(1, N, uData)
	resp, _ := sys.Simulate(u, nil, nil)

	est, err := FreqRespEst(u, resp.Y, dt, &FreqRespEstOpts{Method: FreqRespEstFFT, NFFT: N})
	if err != nil {
		t.Fatal(err)
	}

	if est.Coherence != nil {
		t.Error("expected nil coherence for fft method")
	}

	// Verify estimate at a few mid-range frequencies
	nFreq := len(est.Omega)
	for f := nFreq / 10; f < nFreq/2; f += nFreq / 20 {
		w := est.Omega[f]
		z := cmplx.Exp(complex(0, w*dt))
		hTrue := complex(0.1, 0) / (z - 0.9)
		hEst := est.H.At(f, 0, 0)
		relErr := cmplx.Abs(hEst-hTrue) / cmplx.Abs(hTrue)
		if relErr > 0.2 {
			t.Errorf("w=%.2f: relErr=%.3f", w, relErr)
		}
	}
}

func TestFreqRespEst_MIMO(t *testing.T) {
	dt := 0.01
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{0.8, 0, 0, 0.6},
		[]float64{0.2, 0, 0, 0.4},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}

	N := 4096
	rng := rand.New(rand.NewSource(77))
	uData := make([]float64, 2*N)
	for i := range uData {
		uData[i] = rng.NormFloat64()
	}
	u := mat.NewDense(2, N, uData)

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	est, err := FreqRespEst(u, resp.Y, dt, nil)
	if err != nil {
		t.Fatal(err)
	}

	// Verify diagonal dominance: off-diag should be much smaller than diagonal
	nFreq := len(est.Omega)
	offDiagSum, diagSum := 0.0, 0.0
	for f := nFreq / 5; f < nFreq*4/5; f++ {
		diagSum += cmplx.Abs(est.H.At(f, 0, 0)) + cmplx.Abs(est.H.At(f, 1, 1))
		offDiagSum += cmplx.Abs(est.H.At(f, 0, 1)) + cmplx.Abs(est.H.At(f, 1, 0))
	}
	ratio := offDiagSum / diagSum
	if ratio > 0.3 {
		t.Errorf("off-diagonal/diagonal ratio = %.3f, want < 0.3", ratio)
	}
}

func TestFreqRespEstResultFRD(t *testing.T) {
	dt := 0.01
	sys, err := NewFromSlices(2, 2, 2,
		[]float64{0.8, 0.1, -0.2, 0.6},
		[]float64{0.2, 0.1, 0.05, 0.4},
		[]float64{1, 0, 0, 1},
		[]float64{0, 0, 0, 0},
		dt,
	)
	if err != nil {
		t.Fatal(err)
	}

	N := 2048
	rng := rand.New(rand.NewSource(91))
	uData := make([]float64, 2*N)
	for i := range uData {
		uData[i] = rng.NormFloat64()
	}
	u := mat.NewDense(2, N, uData)

	resp, err := sys.Simulate(u, nil, nil)
	if err != nil {
		t.Fatal(err)
	}

	est, err := FreqRespEst(u, resp.Y, dt, &FreqRespEstOpts{NFFT: 512})
	if err != nil {
		t.Fatal(err)
	}
	frd, err := est.FRD()
	if err != nil {
		t.Fatal(err)
	}

	if frd.Dt != dt {
		t.Fatalf("FRD sample time = %v, want %v", frd.Dt, dt)
	}
	if frd.NumFrequencies() != len(est.Omega) {
		t.Fatalf("NumFrequencies = %d, want %d", frd.NumFrequencies(), len(est.Omega))
	}
	p, m := frd.Dims()
	if p != est.H.P || m != est.H.M {
		t.Fatalf("FRD dims = (%d,%d), want (%d,%d)", p, m, est.H.P, est.H.M)
	}
	for k := range est.Omega {
		for i := range p {
			for j := range m {
				if frd.At(k, i, j) != est.H.At(k, i, j) {
					t.Fatalf("FRD[%d,%d,%d] = %v, want %v", k, i, j, frd.At(k, i, j), est.H.At(k, i, j))
				}
			}
		}
	}
}

func TestFreqRespEstRejectsUnknownMethod(t *testing.T) {
	u := mat.NewDense(1, 16, []float64{1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0})
	y := mat.NewDense(1, 16, []float64{0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0, 0.5, 0})

	_, err := FreqRespEst(u, y, 0.1, &FreqRespEstOpts{Method: FreqRespEstMethod("bogus")})
	if err == nil {
		t.Fatal("expected unknown method error")
	}
}

func TestFreqRespEst_ShortData(t *testing.T) {
	uData := []float64{1, 2, 3, 4, 5}
	yData := []float64{0.5, 1, 1.5, 2, 2.5}
	u := mat.NewDense(1, 5, uData)
	y := mat.NewDense(1, 5, yData)

	est, err := FreqRespEst(u, y, 0.01, nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(est.Omega) == 0 {
		t.Error("expected non-empty result for short data")
	}
}

func TestFreqRespEstResultOmegaNotShared(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	N := 512
	u := mat.NewDense(2, N, nil)
	y := mat.NewDense(1, N, nil)
	for k := range N {
		u.Set(0, k, rng.NormFloat64())
		u.Set(1, k, rng.NormFloat64())
		y.Set(0, k, u.At(0, k)+0.5*u.At(1, k))
	}
	for _, method := range []FreqRespEstMethod{FreqRespEstH1, FreqRespEstFFT} {
		in, out := u, y
		if method == FreqRespEstFFT {
			in, out = u.Slice(0, 1, 0, N).(*mat.Dense), y
		}
		est, err := FreqRespEst(in, out, 0.1, &FreqRespEstOpts{NFFT: 64, Method: method})
		if err != nil {
			t.Fatalf("%s: %v", method, err)
		}
		est.Omega[1] = -7
		if est.H.Omega[1] == -7 {
			t.Errorf("%s: Omega and H.Omega share a backing array", method)
		}
	}
}

func TestFreqRespEstRejectsNonFiniteDt(t *testing.T) {
	u := mat.NewDense(1, 64, nil)
	y := mat.NewDense(1, 64, nil)
	for k := range 64 {
		u.Set(0, k, math.Sin(0.3*float64(k)))
		y.Set(0, k, math.Cos(0.3*float64(k)))
	}
	for _, dt := range []float64{math.NaN(), math.Inf(1), 0, -0.1} {
		_, err := FreqRespEst(u, y, dt, nil)
		if !errors.Is(err, ErrInvalidSampleTime) || !strings.HasPrefix(err.Error(), "FreqRespEst: ") {
			t.Errorf("dt=%g: err = %v, want FreqRespEst: ... ErrInvalidSampleTime", dt, err)
		}
	}
	if _, err := FreqRespEst(nil, y, 0.1, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil input: err = %v, want ErrInvalidArgument", err)
	}
	bad := mat.DenseCopyOf(y)
	bad.Set(0, 3, math.Inf(-1))
	if _, err := FreqRespEst(u, bad, 0.1, nil); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Inf output: err = %v, want ErrInvalidArgument", err)
	}
}
