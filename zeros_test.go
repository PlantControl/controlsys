package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestZeros_SISO_Known(t *testing.T) {
	// H(s) = (s+1)(s+2)/((s+3)(s+4)) → zeros at -1, -2
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 3, 2}}},
		Den: [][]float64{{1, 7, 12}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 2 {
		t.Fatalf("expected 2 zeros, got %d", len(zeros))
	}
	want := []complex128{-1, -2}
	assertZerosMatch(t, zeros, want, 1e-10)
}

func TestZeros_SISO_Integrator(t *testing.T) {
	// H(s) = 1/s → no zeros
	tf := &TransferFunc{
		Num: [][][]float64{{{1}}},
		Den: [][]float64{{1, 0}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 0 {
		t.Errorf("expected no zeros, got %v", zeros)
	}
}

func TestZeros_SISO_RepeatedZeros(t *testing.T) {
	// H(s) = (s+1)²/((s+2)³) → zeros at -1 (multiplicity 2)
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 2, 1}}},
		Den: [][]float64{{1, 6, 12, 8}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 2 {
		t.Fatalf("expected 2 zeros, got %d", len(zeros))
	}
	for _, z := range zeros {
		if cmplx.Abs(z-(-1)) > 1e-6 {
			t.Errorf("expected zero at -1, got %v", z)
		}
	}
}

func TestZeros_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 0 {
		t.Errorf("expected no zeros for gain system, got %v", zeros)
	}
}

func TestZeros_NoInputsOrOutputs(t *testing.T) {
	sys, _ := NewGain(&mat.Dense{}, 0)
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 0 {
		t.Errorf("expected no zeros, got %v", zeros)
	}
}

func TestZeros_MIMO_InvertibleD(t *testing.T) {
	// 2×2 system with invertible D
	// zeros = eig(A - B*D⁻¹*C)
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	C := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	D := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	// A - B*D⁻¹*C = A - I*I⁻¹*I = A - I = [-1 1; -2 -4]
	// eig([-1 1; -2 -4]) = (-5±sqrt(25-4*(-1)(-4+2)))/2... let's compute:
	// char poly: λ²+5λ+6 = (λ+2)(λ+3) → zeros at -2, -3
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	want := []complex128{-2, -3}
	assertZerosMatch(t, zeros, want, 1e-10)
}

func TestZeros_Discrete(t *testing.T) {
	// H(z) = (z-0.5)/(z-0.9) → zero at 0.5
	tf := &TransferFunc{
		Num: [][][]float64{{{1, -0.5}}},
		Den: [][]float64{{1, -0.9}},
		Dt:  0.1,
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 1 {
		t.Fatalf("expected 1 zero, got %d", len(zeros))
	}
	if cmplx.Abs(zeros[0]-0.5) > 1e-10 {
		t.Errorf("expected zero at 0.5, got %v", zeros[0])
	}
}

func TestZeros_SeriesRoundtrip(t *testing.T) {
	// Two SISO systems in series: zeros of series = union of zeros
	// G1: (s+1)/(s+3), G2: (s+2)/(s+4)
	tf1 := &TransferFunc{
		Num: [][][]float64{{{1, 1}}},
		Den: [][]float64{{1, 3}},
	}
	tf2 := &TransferFunc{
		Num: [][][]float64{{{1, 2}}},
		Den: [][]float64{{1, 4}},
	}
	r1, _ := tf1.StateSpace(nil)
	r2, _ := tf2.StateSpace(nil)

	z1, err := r1.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	z2, err := r2.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(z1) != 1 || len(z2) != 1 {
		t.Fatalf("expected 1 zero each, got %d and %d", len(z1), len(z2))
	}
	if cmplx.Abs(z1[0]-(-1)) > 1e-10 {
		t.Errorf("G1 zero: expected -1, got %v", z1[0])
	}
	if cmplx.Abs(z2[0]-(-2)) > 1e-10 {
		t.Errorf("G2 zero: expected -2, got %v", z2[0])
	}
}

func TestZeros_SISO_ComplexZeros(t *testing.T) {
	// H(s) = (s²+1)/((s+1)(s+2)) → zeros at ±j
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0, 1}}},
		Den: [][]float64{{1, 3, 2}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(zeros) != 2 {
		t.Fatalf("expected 2 zeros, got %d", len(zeros))
	}
	want := []complex128{complex(0, -1), complex(0, 1)}
	assertZerosMatch(t, zeros, want, 1e-10)
}

func TestZeros_SISO_HighOrder(t *testing.T) {
	// H(s) = (s+1)(s+2)(s+3)/((s+4)(s+5)(s+6)(s+7)) → zeros at -1,-2,-3
	num := Poly{1, 1}.Mul(Poly{1, 2}).Mul(Poly{1, 3})
	den := Poly{1, 4}.Mul(Poly{1, 5}).Mul(Poly{1, 6}).Mul(Poly{1, 7})
	tf := &TransferFunc{
		Num: [][][]float64{{[]float64(num)}},
		Den: [][]float64{[]float64(den)},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := res.Sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	want := []complex128{-1, -2, -3}
	assertZerosMatch(t, zeros, want, 1e-8)
}

func TestZeros_MIMO_9x2(t *testing.T) {
	// Reference: 9-state, 2-input, 2-output
	A := mat.NewDense(9, 9, []float64{
		-3.93, -0.00315, 0, 0, 0, 4.03e-5, 0, 0, 0,
		368, -3.05, 3.03, 0, 0, -3.77e-3, 0, 0, 0,
		27.4, 0.0787, -5.96e-2, 0, 0, -2.81e-4, 0, 0, 0,
		-0.0647, -5.2e-5, 0, -0.255, -3.35e-6, 3.6e-7, 6.33e-5, 1.94e-4, 0,
		3850, 17.3, -12.8, -12600, -2.91, -0.105, 12.7, 43.1, 0,
		22400, 18, 0, -35.6, -1.04e-4, -0.414, 90, 56.9, 0,
		0, 0, 2.34e-3, 0, 0, 2.22e-4, -0.203, 0, 0,
		0, 0, 0, -1.27, -1.00e-3, 7.86e-5, 0, -7.17e-2, 0,
		-2.2, -0.00177, 0, -8.44, -1.11e-4, 1.38e-5, 1.49e-3, 6.02e-3, -1e-10,
	})
	B := mat.NewDense(9, 2, []float64{
		0, 0,
		0, 0,
		1.56, 0,
		0, -5.13e-6,
		8.28, -1.55,
		0, 1.78,
		2.33, 0,
		0, -2.45e-2,
		0, 2.94e-5,
	})
	C := mat.NewDense(2, 9, []float64{
		0, 0, 0, 0, 0, 1, 0, 0, 0,
		0, 0, 0, 0, 0, 0, 0, 0, 1,
	})
	D := mat.NewDense(2, 2, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	want := []complex128{
		-2.64128629e+01,
		complex(-2.93193619, -0.419522621),
		-9.52183370e-03,
		complex(-2.93193619, 0.419522621),
		1.69789270e-01,
		5.46527700e-01,
	}
	if len(zeros) != len(want) {
		t.Fatalf("expected %d zeros, got %d: %v", len(want), len(zeros), zeros)
	}
	assertZerosMatch(t, zeros, want, 1e-5)
}

func TestZeros_MIMO_4x3_NonSquare(t *testing.T) {
	// Reference: 4-state, 3-input, 1-output (non-square)
	A := mat.NewDense(4, 4, []float64{
		-6.5, 0.5, 6.5, -6.5,
		-0.5, -5.5, -5.5, 5.5,
		-0.5, 0.5, 0.5, -6.5,
		-0.5, 0.5, -5.5, -0.5,
	})
	B := mat.NewDense(4, 3, []float64{
		0, 1, 0,
		2, 1, 2,
		3, 4, 3,
		3, 2, 3,
	})
	C := mat.NewDense(1, 4, []float64{1, 1, 0, 0})
	D := mat.NewDense(1, 3, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	want := []complex128{-6, -7}
	assertZerosMatch(t, zeros, want, 1e-8)
}

func TestZeros_Reinschke(t *testing.T) {
	// Reinschke 1988: 6-state, 2-input, 3-output
	A := mat.NewDense(6, 6, []float64{
		0, 0, 1, 0, 0, 0,
		2, 0, 0, 3, 4, 0,
		0, 0, 5, 0, 0, 6,
		0, 7, 0, 0, 0, 0,
		0, 0, 0, 8, 9, 0,
		0, 0, 0, 0, 0, 0,
	})
	B := mat.NewDense(6, 2, []float64{
		0, 0,
		0, 0,
		0, 0,
		0, 0,
		10, 0,
		0, 11,
	})
	C := mat.NewDense(3, 6, []float64{
		0, 12, 0, 0, 13, 0,
		14, 0, 0, 0, 0, 0,
		15, 0, 16, 0, 0, 0,
	})
	D := mat.NewDense(3, 2, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	zeros, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}

	want := []complex128{-6.78662791, 3.09432022}
	assertZerosMatch(t, zeros, want, 1e-6)
}

func TestZeros_StaircaseExample(t *testing.T) {
	// Staircase zeros doc example: N=6, M=2, P=3
	// Expected: nu=2, rank=2, zeros at 2.0 and -1.0
	A := mat.NewDense(6, 6, []float64{
		1, 0, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 3, 0, 0, 0,
		0, 0, 0, -4, 0, 0,
		0, 0, 0, 0, -1, 0,
		0, 0, 0, 0, 0, 3,
	})
	B := mat.NewDense(6, 2, []float64{
		0, -1,
		-1, 0,
		1, -1,
		0, 0,
		0, 1,
		-1, -1,
	})
	C := mat.NewDense(3, 6, []float64{
		1, 0, 0, 1, 0, 0,
		0, 1, 0, 1, 0, 1,
		0, 0, 1, 0, 0, 1,
	})
	D := mat.NewDense(3, 2, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.ZerosDetail()
	if err != nil {
		t.Fatal(err)
	}
	if res.Rank != 2 {
		t.Errorf("rank: got %d, want 2", res.Rank)
	}
	want := []complex128{-1, 2}
	assertZerosMatch(t, res.Zeros, want, 1e-10)
}

func TestZeros_StaircaseNonSymmetric(t *testing.T) {
	A := mat.NewDense(6, 6, []float64{
		1, 0.1, 0, 0, 0, 0,
		0, 1, 0, 0, 0, 0,
		0, 0, 3, 0.2, 0, 0,
		0, 0, 0, -4, 0, 0,
		0, 0, 0, 0, -1, 0.15,
		0, 0, 0, 0, 0, 3,
	})
	B := mat.NewDense(6, 2, []float64{
		0, -1,
		-1, 0,
		1, -1,
		0, 0,
		0, 1,
		-1, -1,
	})
	C := mat.NewDense(3, 6, []float64{
		1, 0, 0, 1, 0, 0,
		0, 1, 0, 1, 0, 1,
		0, 0, 1, 0, 0, 1,
	})
	D := mat.NewDense(3, 2, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.ZerosDetail()
	if err != nil {
		t.Fatal(err)
	}
	if res.Rank != 2 {
		t.Errorf("rank: got %d, want 2", res.Rank)
	}
	assertZerosMatch(t, res.Zeros, []complex128{-1}, 1e-10)
}

func assertZerosMatch(t *testing.T, got, want []complex128, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("expected %d zeros, got %d: %v", len(want), len(got), got)
	}
	used := make([]bool, len(want))
	for _, g := range got {
		matched := false
		for j, w := range want {
			if !used[j] && cmplx.Abs(g-w) < tol {
				used[j] = true
				matched = true
				break
			}
		}
		if !matched {
			t.Errorf("unexpected zero %v, want %v", g, want)
		}
	}
}
