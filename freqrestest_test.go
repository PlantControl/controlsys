package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand"
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
		coh, ok := est.CoherenceAt(f, 0, 0)
		if !ok || coh < 0.95 {
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

	if _, ok := est.Coherence(); ok {
		t.Error("fft method reported coherence")
	}
	if _, ok := est.CoherenceAt(0, 0, 0); ok {
		t.Error("fft method reported CoherenceAt")
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

func welchSISOOracle(u, y []float64, nfft, noverlap int, win []float64) (omegaIdx []int, h []complex128) {
	hop := nfft - noverlap
	nSeg := (len(u) - noverlap) / hop
	nFreq := nfft/2 + 1
	suu := make([]float64, nFreq)
	syu := make([]complex128, nFreq)
	for s := range nSeg {
		for f := range nFreq {
			var U, Y complex128
			for k := range nfft {
				e := cmplx.Exp(complex(0, -2*math.Pi*float64(f*k)/float64(nfft)))
				U += complex(u[s*hop+k]*win[k], 0) * e
				Y += complex(y[s*hop+k]*win[k], 0) * e
			}
			suu[f] += real(U)*real(U) + imag(U)*imag(U)
			syu[f] += cmplx.Conj(U) * Y
		}
	}
	for f := range nFreq {
		if suu[f] > 1e-12 {
			omegaIdx = append(omegaIdx, f)
			h = append(h, syu[f]/complex(suu[f], 0))
		}
	}
	return omegaIdx, h
}

func TestFreqRespEstOptionsExplicit(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	N := 24
	u := make([]float64, N)
	y := make([]float64, N)
	for k := range N {
		u[k] = rng.NormFloat64()
		y[k] = 0.7 * u[k]
		if k > 0 {
			y[k] += 0.3*u[k-1] - 0.2*y[k-1]
		}
	}
	U, Y := mat.NewDense(1, N, u), mat.NewDense(1, N, y)
	dt := 0.5
	half := func(w []float64) []float64 {
		out := make([]float64, len(w))
		for i := range out {
			out[i] = 1 + 0.5*float64(i%2)
		}
		return out
	}
	zero, three := 0, 3
	for _, c := range []struct {
		name     string
		noverlap *int
		nov      int
		win      func([]float64) []float64
		w        []float64
	}{
		{"zero overlap", &zero, 0, window.Rectangular, []float64{1, 1, 1, 1, 1, 1, 1, 1}},
		{"overlap 3", &three, 3, window.Rectangular, []float64{1, 1, 1, 1, 1, 1, 1, 1}},
		{"returned window", &zero, 0, half, half(make([]float64, 8))},
	} {
		est, err := FreqRespEst(U, Y, dt, &FreqRespEstOpts{NFFT: 8, NOverlap: c.noverlap, Window: c.win})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		idx, want := welchSISOOracle(u, y, 8, c.nov, c.w)
		if len(est.Omega) != len(idx) {
			t.Fatalf("%s: %d bins, want %d", c.name, len(est.Omega), len(idx))
		}
		for k, f := range idx {
			if w := 2 * math.Pi * float64(f) / (8 * dt); math.Abs(est.Omega[k]-w) > 1e-12 {
				t.Errorf("%s: Omega[%d] = %g, want %g", c.name, k, est.Omega[k], w)
			}
			if got := est.H.At(k, 0, 0); cmplx.Abs(got-want[k]) > 1e-9*cmplx.Abs(want[k]) {
				t.Errorf("%s: H[%d] = %v, want %v", c.name, k, got, want[k])
			}
		}
	}

	bad := []struct {
		name string
		opts *FreqRespEstOpts
		want error
	}{
		{"NFFT > N", &FreqRespEstOpts{NFFT: N + 1}, ErrInvalidArgument},
		{"negative NFFT", &FreqRespEstOpts{NFFT: -4}, ErrInvalidArgument},
		{"NOverlap >= NFFT", &FreqRespEstOpts{NFFT: 8, NOverlap: new(8)}, ErrInvalidArgument},
		{"negative NOverlap", &FreqRespEstOpts{NFFT: 8, NOverlap: new(-1)}, ErrInvalidArgument},
		{"unknown method", &FreqRespEstOpts{Method: "bogus"}, ErrInvalidArgument},
		{"short window", &FreqRespEstOpts{NFFT: 8, Window: func([]float64) []float64 { return []float64{1} }}, ErrDimensionMismatch},
	}
	for _, c := range bad {
		if _, err := FreqRespEst(U, Y, dt, c.opts); !errors.Is(err, c.want) {
			t.Errorf("%s: err = %v, want %v", c.name, err, c.want)
		}
	}
}

func TestFreqRespEstUnidentifiable(t *testing.T) {
	rng := rand.New(rand.NewSource(8))
	N := 64
	u := mat.NewDense(2, N, nil)
	y := mat.NewDense(1, N, nil)
	for k := range N {
		u.Set(0, k, rng.NormFloat64())
		u.Set(1, k, rng.NormFloat64())
		y.Set(0, k, u.At(0, k)-u.At(1, k))
	}
	if _, err := FreqRespEst(u, y, 0.1, &FreqRespEstOpts{NFFT: 64}); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("one segment, two inputs: err = %v, want ErrInsufficientData", err)
	}
	if _, err := FreqRespEst(u, y, 0.1, &FreqRespEstOpts{Method: FreqRespEstH2}); !errors.Is(err, ErrOptionUnsupported) {
		t.Errorf("h2 MIMO: err = %v, want ErrOptionUnsupported", err)
	}

	dc := mat.NewDense(1, 16, nil)
	out := mat.NewDense(1, 16, nil)
	for k := range 16 {
		dc.Set(0, k, 2)
		out.Set(0, k, 3)
	}
	est, err := FreqRespEst(dc, out, 0.1, &FreqRespEstOpts{NFFT: 16, Method: FreqRespEstFFT, Window: window.Rectangular})
	if err != nil {
		t.Fatal(err)
	}
	if len(est.Omega) != 1 || est.Omega[0] != 0 || cmplx.Abs(est.H.At(0, 0, 0)-1.5) > 1e-12 {
		t.Errorf("constant input: Omega=%v H=%v, want only DC with H=1.5", est.Omega, est.H.Data)
	}
	if _, err := FreqRespEst(mat.NewDense(1, 16, nil), out, 0.1, &FreqRespEstOpts{NFFT: 16}); !errors.Is(err, ErrInsufficientData) {
		t.Errorf("zero input: err = %v, want ErrInsufficientData", err)
	}
}
