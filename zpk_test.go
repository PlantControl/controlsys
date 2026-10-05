package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestZPKCopyIsIndependent(t *testing.T) {
	z := &ZPK{
		Zeros:      [][][]complex128{{{1 + 2i}}},
		Poles:      [][][]complex128{{{-3 + 4i}}},
		Gain:       [][]float64{{5}},
		Dt:         0.1,
		InputName:  []string{"u"},
		OutputName: []string{"y"},
	}

	cp := z.Copy()
	z.Zeros[0][0][0] = 99
	z.Poles[0][0][0] = 99
	z.Gain[0][0] = 99
	z.InputName[0] = "mutated"
	z.OutputName[0] = "mutated"

	if cp.Zeros[0][0][0] != 1+2i || cp.Poles[0][0][0] != -3+4i || cp.Gain[0][0] != 5 {
		t.Fatalf("Copy aliases model storage: %+v", cp)
	}
	if cp.InputName[0] != "u" || cp.OutputName[0] != "y" {
		t.Fatalf("Copy aliases names: %+v", cp)
	}
}

// --- Phase 0: Poly.Roots tests ---

func TestPolyRootsQuadratic(t *testing.T) {
	p := Poly{1, -3, 2} // (s-1)(s-2)
	roots, err := p.Roots()
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 2 {
		t.Fatalf("got %d roots, want 2", len(roots))
	}
	matchRoots(t, roots, []complex128{1, 2}, 1e-12)
}

func TestPolyRootsComplex(t *testing.T) {
	p := Poly{1, 0, 1} // s^2+1, roots at +/-j
	roots, err := p.Roots()
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 2 {
		t.Fatalf("got %d roots, want 2", len(roots))
	}
	matchRoots(t, roots, []complex128{-1i, 1i}, 1e-12)
}

func TestPolyRootsLinear(t *testing.T) {
	p := Poly{2, -6}
	roots, err := p.Roots()
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 1 {
		t.Fatalf("got %d roots, want 1", len(roots))
	}
	if cmplx.Abs(roots[0]-3) > 1e-14 {
		t.Errorf("root = %v, want 3", roots[0])
	}
}

func TestPolyRootsConstant(t *testing.T) {
	roots, err := Poly{5}.Roots()
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 0 {
		t.Errorf("constant poly should have no roots, got %v", roots)
	}
}

func TestPolyRootsEmpty(t *testing.T) {
	roots, err := Poly{}.Roots()
	if err != nil {
		t.Fatal(err)
	}
	if roots != nil {
		t.Errorf("empty poly roots = %v, want nil", roots)
	}
}

func TestPolyRootsHighOrder(t *testing.T) {
	// (s-1)(s-2)(s-3)(s-4)(s-5)
	p := Poly{1, -1}
	for i := 2; i <= 5; i++ {
		p = p.Mul(Poly{1, float64(-i)})
	}
	roots, err := p.Roots()
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 5 {
		t.Fatalf("got %d roots, want 5", len(roots))
	}
	matchRoots(t, roots, []complex128{1, 2, 3, 4, 5}, 1e-10)
}

func TestPolyRootsRepeated(t *testing.T) {
	p := Poly{1, -2, 1} // (s-1)^2
	roots, err := p.Roots()
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 2 {
		t.Fatalf("got %d roots, want 2", len(roots))
	}
	for _, r := range roots {
		if cmplx.Abs(r-1) > 1e-6 {
			t.Errorf("root = %v, want 1", r)
		}
	}
}

func TestPolyRootsLeadingZeros(t *testing.T) {
	p := Poly{0, 0, 1, -3, 2} // leading zeros stripped → (s-1)(s-2)
	roots, err := p.Roots()
	if err != nil {
		t.Fatal(err)
	}
	if len(roots) != 2 {
		t.Fatalf("got %d roots, want 2", len(roots))
	}
	matchRoots(t, roots, []complex128{1, 2}, 1e-12)
}

func TestPolyRootsRoundtrip(t *testing.T) {
	want := []complex128{-1, -2, complex(-0.5, 1), complex(-0.5, -1)}
	p := polyFromComplexRoots(want)
	roots, err := p.Roots()
	if err != nil {
		t.Fatal(err)
	}
	matchRoots(t, roots, want, 1e-10)
}

// --- Phase 1: ZPK struct tests ---

func TestNewZPK_SISO(t *testing.T) {
	z, err := NewZPK([]complex128{-1}, []complex128{-2, -3}, 2.0, 0)
	if err != nil {
		t.Fatal(err)
	}
	p, m := z.Dims()
	if p != 1 || m != 1 {
		t.Errorf("Dims = (%d,%d), want (1,1)", p, m)
	}
	if !z.IsContinuous() {
		t.Error("expected continuous")
	}
}

func TestNewZPK_Discrete(t *testing.T) {
	z, err := NewZPK(nil, []complex128{0.5}, 1.0, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if !z.IsDiscrete() {
		t.Error("expected discrete")
	}
}

func TestNewZPK_RejectUnpairedPole(t *testing.T) {
	_, err := NewZPK(nil, []complex128{complex(0, 1)}, 1.0, 0)
	if err != ErrConjugatePairs {
		t.Errorf("got err = %v, want ErrConjugatePairs", err)
	}
}

func TestNewZPK_RejectUnpairedZero(t *testing.T) {
	_, err := NewZPK([]complex128{complex(0, 1)}, nil, 1.0, 0)
	if err != ErrConjugatePairs {
		t.Errorf("got err = %v, want ErrConjugatePairs", err)
	}
}

func TestNewZPK_RejectNegativeDt(t *testing.T) {
	_, err := NewZPK(nil, nil, 1.0, -1)
	if err == nil {
		t.Error("expected error for negative dt")
	}
}

func TestNewZPK_PureGain(t *testing.T) {
	z, err := NewZPK(nil, nil, 5.0, 0)
	if err != nil {
		t.Fatal(err)
	}
	if z.Gain[0][0] != 5.0 {
		t.Errorf("gain = %v, want 5", z.Gain[0][0])
	}
}

func TestNewZPKMIMO(t *testing.T) {
	zeros := [][][]complex128{
		{{-1}, {-3}},
		{{}, {-5}},
	}
	poles := [][][]complex128{
		{{-2, -4}, {-2, -4}},
		{{-6}, {-6, -7}},
	}
	gain := [][]float64{{2, 1}, {0, 3}}
	z, err := NewZPKMIMO(zeros, poles, gain, 0)
	if err != nil {
		t.Fatal(err)
	}
	p, m := z.Dims()
	if p != 2 || m != 2 {
		t.Errorf("Dims = (%d,%d), want (2,2)", p, m)
	}
}

func TestNewZPKMIMO_DimensionMismatch(t *testing.T) {
	zeros := [][][]complex128{{{-1}}}
	poles := [][][]complex128{{{-2}}, {{-3}}}
	gain := [][]float64{{1}}
	_, err := NewZPKMIMO(zeros, poles, gain, 0)
	if err != ErrDimensionMismatch {
		t.Errorf("got err = %v, want ErrDimensionMismatch", err)
	}
}

func TestZPKTransferFunctionRejectsMalformedRawModel(t *testing.T) {
	tests := []struct {
		name string
		z    *ZPK
	}{
		{name: "missing zeros", z: &ZPK{Poles: [][][]complex128{{nil}}, Gain: [][]float64{{1}}}},
		{name: "missing poles", z: &ZPK{Zeros: [][][]complex128{{nil}}, Gain: [][]float64{{1}}}},
		{name: "ragged gain", z: &ZPK{
			Zeros: [][][]complex128{{nil}, {nil}},
			Poles: [][][]complex128{{nil}, {nil}},
			Gain:  [][]float64{{1}, {1, 2}},
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.z.TransferFunction(); !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("got %v, want ErrDimensionMismatch", err)
			}
		})
	}
}

// --- Phase 2: ZPK.Eval tests ---

func TestZPKEval_SISO(t *testing.T) {
	// H(s) = 2(s+1)/((s+2)(s+3))
	z, _ := NewZPK([]complex128{-1}, []complex128{-2, -3}, 2.0, 0)

	freqs := []complex128{1i, 2i, complex(0.5, 1)}
	for _, s := range freqs {
		got := z.Eval(s)[0][0]
		num := 2.0 * (s + 1)
		den := (s + 2) * (s + 3)
		want := num / den
		if cmplx.Abs(got-want) > 1e-12 {
			t.Errorf("Eval(%v) = %v, want %v", s, got, want)
		}
	}
}

func TestZPKEval_PureGain(t *testing.T) {
	z, _ := NewZPK(nil, nil, 5.0, 0)
	got := z.Eval(1i)[0][0]
	if cmplx.Abs(got-5) > 1e-14 {
		t.Errorf("Eval(i) = %v, want 5", got)
	}
}

func TestZPKEval_Integrator(t *testing.T) {
	z, _ := NewZPK(nil, []complex128{0}, 1.0, 0)
	got := z.Eval(1i)[0][0]
	want := 1.0 / 1i
	if cmplx.Abs(got-want) > 1e-14 {
		t.Errorf("Eval(i) = %v, want %v", got, want)
	}
}

func TestZPKEval_ZeroGain(t *testing.T) {
	z, _ := NewZPK(nil, []complex128{-1}, 0.0, 0)
	got := z.Eval(1i)[0][0]
	if got != 0 {
		t.Errorf("Eval(i) = %v, want 0", got)
	}
}

func TestZPKFreqResponse_Continuous(t *testing.T) {
	z, _ := NewZPK([]complex128{-1}, []complex128{-2, -3}, 2.0, 0)
	omega := []float64{0.1, 1.0, 10.0}
	fr, err := z.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k, w := range omega {
		s := complex(0, w)
		got := fr.At(k, 0, 0)
		want := 2 * (s + 1) / ((s + 2) * (s + 3))
		if cmplx.Abs(got-want) > 1e-12 {
			t.Errorf("w=%g: got %v, want %v", w, got, want)
		}
	}
}

func TestZPKFreqResponse_Discrete(t *testing.T) {
	dt := 0.1
	z, _ := NewZPK(nil, []complex128{0.5}, 1.0, dt)
	omega := []float64{0.1, 1.0}
	fr, err := z.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k, w := range omega {
		s := cmplx.Exp(complex(0, w*dt))
		want := 1.0 / (s - 0.5)
		got := fr.At(k, 0, 0)
		if cmplx.Abs(got-want) > 1e-12 {
			t.Errorf("w=%g: got %v, want %v", w, got, want)
		}
	}
}

// --- Phase 3: ZPK.TransferFunction tests ---

func TestZPKToTF_SISO(t *testing.T) {
	z, _ := NewZPK([]complex128{-1, -2}, []complex128{-3, -4}, 3.0, 0)
	tf, err := z.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}
	// num = 3*(s+1)*(s+2) = 3s^2+9s+6
	wantNum := Poly{3, 9, 6}
	// den = (s+3)*(s+4) = s^2+7s+12
	wantDen := Poly{1, 7, 12}
	if !Poly(tf.Num[0][0]).Equal(wantNum, 1e-10) {
		t.Errorf("num = %v, want %v", tf.Num[0][0], wantNum)
	}
	if !Poly(tf.Den[0]).Equal(wantDen, 1e-10) {
		t.Errorf("den = %v, want %v", tf.Den[0], wantDen)
	}
}

func TestZPKToTF_PureGain(t *testing.T) {
	z, _ := NewZPK(nil, nil, 5.0, 0)
	tf, err := z.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}
	if len(tf.Num[0][0]) != 1 || tf.Num[0][0][0] != 5.0 {
		t.Errorf("num = %v, want [5]", tf.Num[0][0])
	}
	if len(tf.Den[0]) != 1 || tf.Den[0][0] != 1.0 {
		t.Errorf("den = %v, want [1]", tf.Den[0])
	}
}

func TestZPKToTF_ComplexConjugateZeros(t *testing.T) {
	z, _ := NewZPK(
		[]complex128{complex(-1, 2), complex(-1, -2)},
		[]complex128{-3},
		1.0, 0,
	)
	tf, err := z.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}
	// (s-(-1+2j))(s-(-1-2j)) = (s+1-2j)(s+1+2j) = (s+1)^2+4 = s^2+2s+5
	wantNum := Poly{1, 2, 5}
	if !Poly(tf.Num[0][0]).Equal(wantNum, 1e-10) {
		t.Errorf("num = %v, want %v", tf.Num[0][0], wantNum)
	}
}

func TestZPKToTF_FrequencyEquivalence(t *testing.T) {
	z, _ := NewZPK([]complex128{-1}, []complex128{-2, -3}, 2.0, 0)
	tf, err := z.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}
	freqs := []complex128{0.1i, 1i, 5i, complex(0.5, 1)}
	for _, s := range freqs {
		zpkVal := z.Eval(s)[0][0]
		tfVal := tf.Eval(s)[0][0]
		if cmplx.Abs(zpkVal-tfVal) > 1e-10 {
			t.Errorf("at s=%v: ZPK=%v, TF=%v", s, zpkVal, tfVal)
		}
	}
}

func TestZPKToTF_MIMO_SharedPoles(t *testing.T) {
	zeros := [][][]complex128{
		{{-1}, {-2}},
	}
	poles := [][][]complex128{
		{{-3, -4}, {-3, -4}},
	}
	gain := [][]float64{{2, 1}}
	z, _ := NewZPKMIMO(zeros, poles, gain, 0)
	tf, err := z.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}
	wantDen := Poly{1, 7, 12}
	if !Poly(tf.Den[0]).Equal(wantDen, 1e-10) {
		t.Errorf("den = %v, want %v", tf.Den[0], wantDen)
	}
}

func TestZPKToTF_MIMO_DifferentPoles(t *testing.T) {
	// Channel (0,0): poles at -1,-2; channel (0,1): poles at -1,-3
	// Common denom should have poles at -1,-2,-3
	zeros := [][][]complex128{
		{{}, {}},
	}
	poles := [][][]complex128{
		{{-1, -2}, {-1, -3}},
	}
	gain := [][]float64{{1, 1}}
	z, _ := NewZPKMIMO(zeros, poles, gain, 0)
	tf, err := z.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}

	// Common denom = (s+1)(s+2)(s+3) = s^3+6s^2+11s+6
	wantDen := Poly{1, 6, 11, 6}
	if !Poly(tf.Den[0]).Equal(wantDen, 1e-10) {
		t.Errorf("den = %v, want %v", tf.Den[0], wantDen)
	}

	// Channel (0,0) missing pole -3, so num = 1*(s+3) = s+3
	wantNum0 := Poly{1, 3}
	if !Poly(tf.Num[0][0]).Equal(wantNum0, 1e-10) {
		t.Errorf("num[0][0] = %v, want %v", tf.Num[0][0], wantNum0)
	}

	// Channel (0,1) missing pole -2, so num = 1*(s+2) = s+2
	wantNum1 := Poly{1, 2}
	if !Poly(tf.Num[0][1]).Equal(wantNum1, 1e-10) {
		t.Errorf("num[0][1] = %v, want %v", tf.Num[0][1], wantNum1)
	}
}

func TestZPKToTF_ZeroGainChannel(t *testing.T) {
	zeros := [][][]complex128{
		{{-1}, {}},
	}
	poles := [][][]complex128{
		{{-2}, {-3}},
	}
	gain := [][]float64{{2, 0}}
	z, _ := NewZPKMIMO(zeros, poles, gain, 0)
	tf, err := z.TransferFunction()
	if err != nil {
		t.Fatal(err)
	}
	if len(tf.Num[0][1]) != 1 || tf.Num[0][1][0] != 0 {
		t.Errorf("zero gain channel num = %v, want [0]", tf.Num[0][1])
	}
}

// --- Phase 4: TransferFunc.ZPK tests ---

func TestTFToZPK_Known(t *testing.T) {
	// s/(s^2+3s+2) = s/((s+1)(s+2))
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0}}},
		Den: [][]float64{{1, 3, 2}},
	}
	z, err := tf.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	matchRoots(t, z.Zeros[0][0], []complex128{0}, 1e-12)
	matchRoots(t, z.Poles[0][0], []complex128{-1, -2}, 1e-12)
	if math.Abs(z.Gain[0][0]-1) > 1e-14 {
		t.Errorf("gain = %v, want 1", z.Gain[0][0])
	}
}

func TestTFToZPK_PureGain(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{5}}},
		Den: [][]float64{{1}},
	}
	z, err := tf.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	if len(z.Zeros[0][0]) != 0 {
		t.Errorf("zeros = %v, want empty", z.Zeros[0][0])
	}
	if len(z.Poles[0][0]) != 0 {
		t.Errorf("poles = %v, want empty", z.Poles[0][0])
	}
	if z.Gain[0][0] != 5 {
		t.Errorf("gain = %v, want 5", z.Gain[0][0])
	}
}

func TestTFToZPK_FrequencyRoundtrip(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{2, 3}}},
		Den: [][]float64{{1, 5, 6}},
	}
	z, err := tf.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	freqs := []complex128{0.1i, 1i, 5i, complex(0.5, 1)}
	for _, s := range freqs {
		tfVal := tf.Eval(s)[0][0]
		zpkVal := z.Eval(s)[0][0]
		if cmplx.Abs(tfVal-zpkVal) > 1e-10 {
			t.Errorf("at s=%v: TF=%v, ZPK=%v", s, tfVal, zpkVal)
		}
	}
}

func TestTFToZPK_RepeatedRoots(t *testing.T) {
	// (s+1)^2 / (s+2)^3
	num := Poly{1, 1}.Mul(Poly{1, 1})
	den := Poly{1, 2}.Mul(Poly{1, 2}).Mul(Poly{1, 2})
	tf := &TransferFunc{
		Num: [][][]float64{{[]float64(num)}},
		Den: [][]float64{[]float64(den)},
	}
	z, err := tf.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	if len(z.Zeros[0][0]) != 2 {
		t.Fatalf("got %d zeros, want 2", len(z.Zeros[0][0]))
	}
	if len(z.Poles[0][0]) != 3 {
		t.Fatalf("got %d poles, want 3", len(z.Poles[0][0]))
	}
}

func TestTFToZPK_ZeroChannel(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0}, {0}}},
		Den: [][]float64{{1, 1}},
	}
	z, err := tf.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	if z.Gain[0][1] != 0 {
		t.Errorf("gain[0][1] = %v, want 0", z.Gain[0][1])
	}
}

func TestTFToZPK_MIMO_SharedDen(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1}, {2, 1}}},
		Den: [][]float64{{1, 3, 2}},
	}
	z, err := tf.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	// Both channels should share the same poles (from shared Den)
	matchRoots(t, z.Poles[0][0], z.Poles[0][1], 1e-12)
}

// --- Phase 5: System.ZPKModel tests ---

func TestSystemZPK_SISO(t *testing.T) {
	// Companion form for s/(s^2+3s+2)
	sys, _ := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	matchRoots(t, res.ZPK.Poles[0][0], []complex128{-1, -2}, 1e-8)
}

func TestSystemZPK_NonSymmetricA(t *testing.T) {
	// H(s) = (s+4)/((s+2)(s+3))
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-2, 1, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	matchRoots(t, res.ZPK.Poles[0][0], []complex128{-2, -3}, 1e-8)
	matchRoots(t, res.ZPK.Zeros[0][0], []complex128{-4}, 1e-8)
}

func TestSystemZPK_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{3}), 0)
	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.ZPK.Poles[0][0]) != 0 {
		t.Errorf("poles = %v, want empty", res.ZPK.Poles[0][0])
	}
	if math.Abs(res.ZPK.Gain[0][0]-3) > 1e-10 {
		t.Errorf("gain = %v, want 3", res.ZPK.Gain[0][0])
	}
}

func TestSystemZPK_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	if !res.ZPK.IsDiscrete() {
		t.Error("expected discrete")
	}
}

func TestSystemZPK_FrequencyMatch(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 1, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	res, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	omega := []float64{0.1, 1, 10}
	ssFR, err := sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	zpkFR, err := res.ZPK.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range omega {
		ssVal := ssFR.At(k, 0, 0)
		zpkVal := zpkFR.At(k, 0, 0)
		if cmplx.Abs(ssVal-zpkVal) > 1e-8 {
			t.Errorf("w=%g: SS=%v, ZPK=%v", omega[k], ssVal, zpkVal)
		}
	}
}

// --- Phase 6: ZPK.StateSpace tests ---

func TestZPKToSS_FrequencyRoundtrip(t *testing.T) {
	z, _ := NewZPK([]complex128{-1}, []complex128{-2, -3}, 2.0, 0)
	ssRes, err := z.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	omega := []float64{0.1, 1, 10}
	ssFR, err := ssRes.Sys.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	zpkFR, err := z.FreqResponse(omega)
	if err != nil {
		t.Fatal(err)
	}
	for k := range omega {
		ssVal := ssFR.At(k, 0, 0)
		zpkVal := zpkFR.At(k, 0, 0)
		if cmplx.Abs(ssVal-zpkVal) > 1e-8 {
			t.Errorf("w=%g: SS=%v, ZPK=%v", omega[k], ssVal, zpkVal)
		}
	}
}

func TestZPKToSS_PureGain(t *testing.T) {
	z, _ := NewZPK(nil, nil, 5.0, 0)
	ssRes, err := z.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := ssRes.Sys.Dims()
	if n != 0 {
		t.Errorf("state dim = %d, want 0", n)
	}
}

func TestZPKToSS_RoundtripZPK(t *testing.T) {
	z, _ := NewZPK([]complex128{-1}, []complex128{-2, -3}, 2.0, 0)
	ssRes, err := z.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	zpkRes, err := ssRes.Sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	matchRoots(t, zpkRes.ZPK.Poles[0][0], []complex128{-2, -3}, 1e-8)
	matchRoots(t, zpkRes.ZPK.Zeros[0][0], []complex128{-1}, 1e-8)
	if math.Abs(zpkRes.ZPK.Gain[0][0]-2) > 1e-8 {
		t.Errorf("gain = %v, want 2", zpkRes.ZPK.Gain[0][0])
	}
}

// --- Integration ---

func TestZPK_FullRoundtrip_SS_TF_ZPK(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 1, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	// SS→TF→ZPK→TF→SS→ZPK: compare frequency response at each step
	freqs := []complex128{0.1i, 1i, 5i, complex(0.5, 1)}

	tfRes, _ := sys.TransferFunction(nil)
	zpk1, _ := tfRes.TF.ZPK()
	tf2, _ := zpk1.TransferFunction()
	ssRes, _ := tf2.StateSpace(nil)
	zpk2, _ := ssRes.Sys.ZPKModel(nil)

	for _, s := range freqs {
		v1 := tfRes.TF.Eval(s)[0][0]
		v2 := zpk1.Eval(s)[0][0]
		v3 := tf2.Eval(s)[0][0]
		v4 := zpk2.ZPK.Eval(s)[0][0]

		if cmplx.Abs(v1-v2) > 1e-8 {
			t.Errorf("TF→ZPK mismatch at s=%v: %v vs %v", s, v1, v2)
		}
		if cmplx.Abs(v2-v3) > 1e-8 {
			t.Errorf("ZPK→TF mismatch at s=%v: %v vs %v", s, v2, v3)
		}
		if cmplx.Abs(v3-v4) > 1e-8 {
			t.Errorf("SS→ZPK mismatch at s=%v: %v vs %v", s, v3, v4)
		}
	}
}

// --- Helpers ---

func matchRoots(t *testing.T, got, want []complex128, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("got %d roots %v, want %d roots %v", len(got), got, len(want), want)
		return
	}
	used := make([]bool, len(want))
	for _, g := range got {
		found := false
		for k, w := range want {
			if !used[k] && cmplx.Abs(g-w) < tol {
				used[k] = true
				found = true
				break
			}
		}
		if !found {
			t.Errorf("unexpected root %v; got=%v want=%v", g, got, want)
			return
		}
	}
}

func TestZPKRejectsDelays(t *testing.T) {
	tf := &TransferFunc{
		Num:   [][][]float64{{{2}}},
		Den:   [][]float64{{1, 1}},
		Delay: [][]float64{{0.5}},
	}
	if _, err := tf.ZPK(); !errors.Is(err, ErrDelayNotRepresentable) {
		t.Errorf("TF.ZPK err = %v, want ErrDelayNotRepresentable", err)
	}
	for _, dt := range []float64{0, 0.1} {
		for name, mk := range map[string]func(*testing.T, float64) *System{"io": fieldIODelay, "lft": fieldLFT} {
			if _, err := mk(t, dt).ZPKModel(nil); !errors.Is(err, ErrDelayNotRepresentable) {
				t.Errorf("%s/dt=%g: ZPKModel err = %v, want ErrDelayNotRepresentable", name, dt, err)
			}
		}
	}
	tf.Delay = [][]float64{{0}}
	if _, err := tf.ZPK(); err != nil {
		t.Errorf("zero delay: %v", err)
	}
}

func TestNewZPKRejectsNonFiniteSampleTime(t *testing.T) {
	for _, dt := range []float64{math.NaN(), math.Inf(1)} {
		if _, err := NewZPK(nil, []complex128{-1}, 1, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("NewZPK dt=%v err = %v", dt, err)
		}
		if _, err := NewZPKMIMO([][][]complex128{{nil}}, [][][]complex128{{{-1}}}, [][]float64{{1}}, dt); !errors.Is(err, ErrInvalidSampleTime) {
			t.Errorf("NewZPKMIMO dt=%v err = %v", dt, err)
		}
	}
}
