package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// --- 1. Pade Approximation Tests ---

func TestPade_MATLAB_T1_8_Order2(t *testing.T) {
	sys, err := PadeDelay(1.8, 2)
	if err != nil {
		t.Fatal(err)
	}
	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	wantNum := []float64{1, -3.3333, 3.7037}
	wantDen := []float64{1, 3.3333, 3.7037}

	num := make([]float64, len(tfRes.TF.Num[0][0]))
	copy(num, tfRes.TF.Num[0][0])
	den := make([]float64, len(tfRes.TF.Den[0]))
	copy(den, tfRes.TF.Den[0])

	scale := den[0]
	for i := range den {
		den[i] /= scale
	}
	numScale := num[0]
	for i := range num {
		num[i] /= numScale
	}

	for i, w := range wantDen {
		if math.Abs(den[i]-w) > 1e-3 {
			t.Errorf("den[%d] = %v, want %v", i, den[i], w)
		}
	}
	for i, w := range wantNum {
		if math.Abs(num[i]-w) > 1e-3 {
			t.Errorf("num[%d] = %v, want %v", i, num[i], w)
		}
	}
}

func TestPade_MATLAB_PureDelay_Order3(t *testing.T) {
	sys, err := PadeDelay(0.1, 3)
	if err != nil {
		t.Fatal(err)
	}
	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	wantNum := []float64{-1, 120, -6000, 120000}
	wantDen := []float64{1, 120, 6000, 120000}

	num := make([]float64, len(tfRes.TF.Num[0][0]))
	copy(num, tfRes.TF.Num[0][0])
	den := make([]float64, len(tfRes.TF.Den[0]))
	copy(den, tfRes.TF.Den[0])

	denScale := den[0] / wantDen[0]
	for i := range den {
		den[i] /= denScale
	}
	numScale := denScale
	for i := range num {
		num[i] /= numScale
	}

	for i, w := range wantDen {
		if math.Abs(den[i]-w) > 1e-2 {
			t.Errorf("den[%d] = %v, want %v", i, den[i], w)
		}
	}
	for i, w := range wantNum {
		if math.Abs(num[i]-w) > 1e-2 {
			t.Errorf("num[%d] = %v, want %v", i, num[i], w)
		}
	}
}

func TestPade_PythonControl_CoeffTable(t *testing.T) {
	tests := []struct {
		order int
		den   []float64
		num   []float64
	}{
		{1, []float64{1, 2}, []float64{-1, 2}},
		{2, []float64{1, 6, 12}, []float64{1, -6, 12}},
		{3, []float64{1, 12, 60, 120}, []float64{-1, 12, -60, 120}},
		{4, []float64{1, 20, 180, 840, 1680}, []float64{1, -20, 180, -840, 1680}},
		{5, []float64{1, 30, 420, 3360, 15120, 30240}, []float64{-1, 30, -420, 3360, -15120, 30240}},
	}

	for _, tc := range tests {
		t.Run("order"+string(rune('0'+tc.order)), func(t *testing.T) {
			sys, err := PadeDelay(1.0, tc.order)
			if err != nil {
				t.Fatal(err)
			}
			tfRes, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}

			num := make([]float64, len(tfRes.TF.Num[0][0]))
			copy(num, tfRes.TF.Num[0][0])
			den := make([]float64, len(tfRes.TF.Den[0]))
			copy(den, tfRes.TF.Den[0])

			denScale := den[0]
			for i := range den {
				den[i] /= denScale
			}
			numScale := num[0] / tc.num[0]
			for i := range num {
				num[i] /= numScale
			}

			for i, w := range tc.den {
				if math.Abs(den[i]-w) > 1e-6 {
					t.Errorf("den[%d] = %v, want %v", i, den[i], w)
				}
			}
			for i, w := range tc.num {
				if math.Abs(num[i]-w) > 1e-6 {
					t.Errorf("num[%d] = %v, want %v", i, num[i], w)
				}
			}
		})
	}
}

// --- 2. Thiran Approximation Tests ---

func TestThiran_MATLAB_Fractional(t *testing.T) {
	sys, err := thiranDelay(2.4, 2, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	if !sys.IsDiscrete() {
		t.Fatal("expected discrete")
	}

	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []float64{0.01, 0.1, 0.5, 1.0} {
		z := cmplx.Exp(complex(0, w))
		h := tfRes.TF.Eval(z)[0][0]
		mag := cmplx.Abs(h)
		if math.Abs(mag-1) > 1e-6 {
			t.Errorf("w=%v: |H| = %v, want 1 (allpass)", w, mag)
		}
	}

	dw := 1e-6
	z1 := cmplx.Exp(complex(0, dw))
	z2 := cmplx.Exp(complex(0, 2*dw))
	h1 := tfRes.TF.Eval(z1)[0][0]
	h2 := tfRes.TF.Eval(z2)[0][0]
	groupDelay := -(cmplx.Phase(h2) - cmplx.Phase(h1)) / dw
	if math.Abs(groupDelay-2.4) > 0.05 {
		t.Errorf("group delay at DC = %v, want ~2.4", groupDelay)
	}
}

func TestThiran_MATLAB_Integer(t *testing.T) {
	sys, err := thiranDelay(2.5, 5, 0.5)
	if err != nil {
		t.Fatal(err)
	}

	if !sys.IsDiscrete() {
		t.Fatal("expected discrete system")
	}

	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []float64{0.1, 0.5, 1.0, 2.0, 3.0}
	for _, w := range freqs {
		z := cmplx.Exp(complex(0, w*0.5))
		h := tfRes.TF.Eval(z)[0][0]
		mag := cmplx.Abs(h)
		if math.Abs(mag-1) > 1e-8 {
			t.Errorf("w=%v: |H(e^jw)| = %v, want 1", w, mag)
		}

		expected := cmplx.Pow(z, -5)
		if cmplx.Abs(h-expected) > 1e-8 {
			t.Errorf("w=%v: H(z)=%v, want z^(-5)=%v", w, h, expected)
		}
	}
}

// --- 3. C2D with Delay Tests ---

func TestC2D_MATLAB_ZOH_IntegerDelay(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -20, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, -2}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{1.0}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
	if err != nil {
		t.Fatal(err)
	}

	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}

	if len(disc.InputDelay) != 1 || disc.InputDelay[0] != 10 {
		t.Errorf("InputDelay = %v, want [10]", disc.InputDelay)
	}

	poles, err := disc.Poles()
	if err != nil {
		t.Fatal(err)
	}

	s1 := complex(-1.5, math.Sqrt(20-2.25))
	s2 := complex(-1.5, -math.Sqrt(20-2.25))
	wantP1 := cmplx.Exp(s1 * complex(dt, 0))
	wantP2 := cmplx.Exp(s2 * complex(dt, 0))

	matched := 0
	for _, p := range poles {
		if cmplx.Abs(p-wantP1) < 0.01 || cmplx.Abs(p-wantP2) < 0.01 {
			matched++
		}
	}
	if matched < 2 {
		t.Errorf("poles = %v, want near %v and %v", poles, wantP1, wantP2)
	}
}

func TestC2D_MATLAB_ZOH_FractionalDelay(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -10, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{10, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{0.25}

	dt := 0.1
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH, DelayModeling: C2DDelayModelingState})
	if err != nil {
		t.Fatal(err)
	}

	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}

	n, _, _ := disc.Dims()
	if n <= 2 {
		t.Errorf("expected augmented state dim > 2, got %d", n)
	}
}

func TestC2D_MATLAB_Tustin_Thiran(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -4}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{2.7}

	dt := 1.0
	disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}

	if !disc.IsDiscrete() {
		t.Fatal("expected discrete")
	}
	if disc.Dt != dt {
		t.Errorf("Dt = %v, want %v", disc.Dt, dt)
	}

	n, _, _ := disc.Dims()
	if n < 2 {
		t.Fatalf("state dim = %d, want >= 2", n)
	}

	for _, w := range []float64{0.1, 0.3, 0.5} {
		z := cmplx.Exp(complex(0, w*dt))
		resp, err := disc.FreqResponse([]float64{w})
		if err != nil {
			t.Fatal(err)
		}
		hDisc := resp.At(0, 0, 0)

		s := complex(0, w)
		hPlant := (s + 1 + 1) / (s*s + 4*s + 2)
		hDelay := cmplx.Exp(-s * complex(2.7, 0))
		hCont := hPlant * hDelay
		_ = z

		magRatio := cmplx.Abs(hDisc) / cmplx.Abs(hCont)
		if magRatio < 0.5 || magRatio > 2.0 {
			t.Errorf("w=%v: |H_disc|/|H_cont| = %v, want ~1", w, magRatio)
		}
	}
}

// --- 4. Feedback with Delay Tests ---

func TestFeedback_MATLAB_CreatesInternalDelay(t *testing.T) {
	G, err := New(
		mat.NewDense(1, 1, []float64{-10}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	G.InputDelay = []float64{2.1}

	C, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{2.3}),
		mat.NewDense(1, 1, []float64{0.5}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	L, err := Series(C, G)
	if err != nil {
		t.Fatal(err)
	}

	T, err := Feedback(L, nil, -1)
	if err != nil {
		t.Fatal(err)
	}

	if !T.HasInternalDelay() {
		t.Fatal("expected internal delay")
	}

	found := false
	for _, d := range T.LFT.Tau {
		if math.Abs(d-2.1) < 1e-10 {
			found = true
		}
	}
	if !found {
		t.Errorf("InternalDelay = %v, want to contain 2.1", T.LFT.Tau)
	}
}

func TestFeedback_MATLAB_PID_DeadTime(t *testing.T) {
	P, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -1, -0.3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{3, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	P.InputDelay = []float64{2.6}

	C, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0.06}),
		mat.NewDense(1, 1, []float64{0.06}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	T, err := Feedback(P, C, -1, WithPadeOrder(5))
	if err != nil {
		t.Fatal(err)
	}

	stable, err := T.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("closed-loop should be stable")
	}
}

// --- 5. Parallel Creates InternalDelay ---

func TestParallel_MATLAB_CreatesInternalDelay(t *testing.T) {
	H1, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	H2, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{5}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	H2.InputDelay = []float64{3.4}

	Hp, err := Parallel(H1, H2)
	if err != nil {
		t.Fatal(err)
	}

	if !Hp.HasInternalDelay() {
		t.Fatal("expected internal delay from parallel with mismatched delays")
	}
	found := false
	for _, d := range Hp.LFT.Tau {
		if math.Abs(d-3.4) < 1e-10 {
			found = true
		}
	}
	if !found {
		t.Errorf("InternalDelay = %v, want to contain 3.4", Hp.LFT.Tau)
	}

	resp, err := Hp.FreqResponse([]float64{0.01})
	if err != nil {
		t.Fatal(err)
	}
	h0 := resp.At(0, 0, 0)
	wantMag := 5.5
	if math.Abs(cmplx.Abs(h0)-wantMag) > 0.1 {
		t.Errorf("|H(j*0.01)| = %v, want ~%v", cmplx.Abs(h0), wantMag)
	}
}

// --- 6. AllMargin with Delay ---

func TestAllMargin_MATLAB_tf25(t *testing.T) {
	// G(s) = 25/(s^3 + 10s^2 + 10s + 10) in controllable canonical form
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			0, 1, 0,
			0, 0, 1,
			-10, -10, -10,
		}),
		mat.NewDense(3, 1, []float64{0, 0, 1}),
		mat.NewDense(1, 3, []float64{25, 0, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	r, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantGM := 20 * math.Log10(3.6)
	if math.Abs(r.GainMargin-wantGM) > 0.3 {
		t.Errorf("GM = %v dB, want ~%v dB", r.GainMargin, wantGM)
	}

	wantWp := 3.1623
	if math.Abs(r.WpFreq-wantWp) > 0.05 {
		t.Errorf("WpFreq = %v, want ~%v", r.WpFreq, wantWp)
	}

	wantPM := 29.1104
	if math.Abs(r.PhaseMargin-wantPM) > 1.0 {
		t.Errorf("PM = %v deg, want ~%v deg", r.PhaseMargin, wantPM)
	}

	wantWg := 1.7844
	if math.Abs(r.WgFreq-wantWg) > 0.05 {
		t.Errorf("WgFreq = %v, want ~%v", r.WgFreq, wantWg)
	}
}

// --- 7. Frequency Response with Delay ---

func TestFreqResponse_MATLAB_WithDelay(t *testing.T) {
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
	sys.InputDelay = []float64{0.5}

	omegas := []float64{0.01, 1.0, 10.0}
	resp, err := sys.FreqResponse(omegas)
	if err != nil {
		t.Fatal(err)
	}

	for k, w := range omegas {
		h := resp.At(k, 0, 0)

		wantMag := 1.0 / math.Sqrt(1+w*w)
		gotMag := cmplx.Abs(h)
		if math.Abs(gotMag-wantMag) > 1e-6 {
			t.Errorf("w=%v: |H| = %v, want %v", w, gotMag, wantMag)
		}

		wantH := complex(1, 0) / complex(1, w) * cmplx.Exp(complex(0, -0.5*w))
		if cmplx.Abs(h-wantH) > 1e-6 {
			t.Errorf("w=%v: H = %v, want %v", w, h, wantH)
		}
	}
}
