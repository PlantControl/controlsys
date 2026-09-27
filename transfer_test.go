package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestTransferFuncDims(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{
			{{1, 0}, {2}},
			{{3}, {4, 1}},
		},
		Den: [][]float64{
			{1, 3, 2},
			{1, 1},
		},
	}
	p, m := tf.Dims()
	if p != 2 || m != 2 {
		t.Fatalf("Dims() = (%d,%d), want (2,2)", p, m)
	}
}

func TestTransferFuncCopyIsIndependent(t *testing.T) {
	tf := &TransferFunc{
		Num:        [][][]float64{{{1, 2}}},
		Den:        [][]float64{{1, 3}},
		Delay:      [][]float64{{0.5}},
		Dt:         0.1,
		InputName:  []string{"u"},
		OutputName: []string{"y"},
	}

	cp := tf.Copy()
	tf.Num[0][0][0] = 99
	tf.Den[0][0] = 99
	tf.Delay[0][0] = 99
	tf.InputName[0] = "mutated"
	tf.OutputName[0] = "mutated"

	if cp.Num[0][0][0] != 1 || cp.Den[0][0] != 1 || cp.Delay[0][0] != 0.5 {
		t.Fatalf("Copy aliases polynomial or delay storage: %+v", cp)
	}
	if cp.InputName[0] != "u" || cp.OutputName[0] != "y" {
		t.Fatalf("Copy aliases names: %+v", cp)
	}
}

func TestTransferFuncStateSpaceRejectsMalformedRawModel(t *testing.T) {
	tests := []struct {
		name string
		tf   *TransferFunc
	}{
		{name: "missing numerator rows", tf: &TransferFunc{Den: [][]float64{{1}}}},
		{name: "missing denominator", tf: &TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{}}}},
		{name: "ragged numerator rows", tf: &TransferFunc{
			Num: [][][]float64{{{1}}, {{1}, {2}}},
			Den: [][]float64{{1}, {1}},
		}},
		{name: "missing channel numerator", tf: &TransferFunc{
			Num: [][][]float64{{nil}},
			Den: [][]float64{{1}},
		}},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			if _, err := tc.tf.StateSpace(nil); !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("got %v, want ErrDimensionMismatch", err)
			}
		})
	}
}

func TestTransferFuncEval(t *testing.T) {
	// T(s) = s / (s^2 + 3s + 2) = s / ((s+1)(s+2))
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0}}},
		Den: [][]float64{{1, 3, 2}},
	}
	s := 1i
	got := tf.Eval(s)
	num := complex(0, 1)
	den := complex(-1, 0) + complex(0, 3) + complex(2, 0)
	want := num / den
	if cmplx.Abs(got[0][0]-want) > 1e-12 {
		t.Fatalf("Eval(1i) = %v, want %v", got[0][0], want)
	}
}

func TestEvalMultiConsistency(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0}}},
		Den: [][]float64{{1, 3, 2}},
	}
	freqs := []complex128{1i, 2i, complex(1, 1)}
	multi := tf.EvalMulti(freqs)
	for k, s := range freqs {
		single := tf.Eval(s)
		if cmplx.Abs(multi[k][0][0]-single[0][0]) > 1e-15 {
			t.Fatalf("EvalMulti[%d] != Eval at s=%v", k, s)
		}
	}
}

func TestTransferFunctionSISOKnown(t *testing.T) {
	// s / (s^2 + 3s + 2): controllable canonical form
	sys, err := NewFromSlices(2, 1, 1,
		[]float64{0, -2, 1, -3},
		[]float64{0, 1},
		[]float64{0, 1},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MinimalOrder != 2 {
		t.Fatalf("MinimalOrder = %d, want 2", res.MinimalOrder)
	}
	if res.RowDegrees[0] != 2 {
		t.Fatalf("RowDegrees[0] = %d, want 2", res.RowDegrees[0])
	}

	freqs := []complex128{1i, 2i, complex(0.5, 1), complex(-0.3, 2.7)}
	for _, s := range freqs {
		tfVal := res.TF.Eval(s)[0][0]
		ssVal := evalSS(sys, s)
		if cmplx.Abs(tfVal-ssVal) > 1e-8 {
			t.Errorf("at s=%v: TF=%v, SS=%v", s, tfVal, ssVal)
		}
	}
}

func TestTransferFunctionRoundtrip(t *testing.T) {
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			-1, 0, 0,
			0, -2, 0,
			0, 0, -3,
		},
		[]float64{
			1, 0,
			0, 1,
			1, 1,
		},
		[]float64{
			1, 0, 1,
			0, 1, 1,
		},
		[]float64{
			0, 0,
			0, 0,
		},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{1i, 0.5i, complex(1, 2)}
	for _, s := range freqs {
		tfMat := res.TF.Eval(s)
		for i := range 2 {
			for j := range 2 {
				ssVal := evalSSij(sys, s, i, j)
				if cmplx.Abs(tfMat[i][j]-ssVal) > 1e-6 {
					t.Errorf("at s=%v [%d][%d]: TF=%v, SS=%v", s, i, j, tfMat[i][j], ssVal)
				}
			}
		}
	}
}

func TestTransferFunctionPureGain(t *testing.T) {
	D := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	sys, err := NewGain(D, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.MinimalOrder != 0 {
		t.Fatalf("MinimalOrder = %d, want 0", res.MinimalOrder)
	}
	for i := range 2 {
		if len(res.TF.Den[i]) != 1 || res.TF.Den[i][0] != 1 {
			t.Errorf("Den[%d] = %v, want [1]", i, res.TF.Den[i])
		}
		for j := range 2 {
			want := D.At(i, j)
			if len(res.TF.Num[i][j]) != 1 || math.Abs(res.TF.Num[i][j][0]-want) > 1e-15 {
				t.Errorf("Num[%d][%d] = %v, want [%v]", i, j, res.TF.Num[i][j], want)
			}
		}
	}
}

func TestStateSpaceSISOCompanion(t *testing.T) {
	// T(s) = s / (s^2 + 3s + 2)
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0}}},
		Den: [][]float64{{1, 3, 2}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := res.Sys.Dims()
	if n != 2 || m != 1 || p != 1 {
		t.Fatalf("dims = (%d,%d,%d), want (2,1,1)", n, m, p)
	}

	// Verify frequency response matches
	freqs := []complex128{1i, 2i, complex(0.5, 1)}
	for _, s := range freqs {
		ssVal := evalSS(res.Sys, s)
		tfVal := tf.Eval(s)[0][0]
		if cmplx.Abs(ssVal-tfVal) > 1e-10 {
			t.Errorf("at s=%v: SS=%v, TF=%v", s, ssVal, tfVal)
		}
	}
}

func TestStateSpacePureGain(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{5}}, {{3}}},
		Den: [][]float64{{1}, {1}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := res.Sys.Dims()
	if n != 0 {
		t.Fatalf("n = %d, want 0", n)
	}
	if math.Abs(res.Sys.D.At(0, 0)-5) > 1e-15 {
		t.Errorf("D(0,0) = %v, want 5", res.Sys.D.At(0, 0))
	}
	if math.Abs(res.Sys.D.At(1, 0)-3) > 1e-15 {
		t.Errorf("D(1,0) = %v, want 3", res.Sys.D.At(1, 0))
	}
}

func TestStateSpaceNonMonic(t *testing.T) {
	// T(s) = 2s / (2s^2 + 6s + 4) = s / (s^2 + 3s + 2)
	tf := &TransferFunc{
		Num: [][][]float64{{{2, 0}}},
		Den: [][]float64{{2, 6, 4}},
	}
	res, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{1i, 2i}
	for _, s := range freqs {
		ssVal := evalSS(res.Sys, s)
		tfVal := tf.Eval(s)[0][0]
		if cmplx.Abs(ssVal-tfVal) > 1e-10 {
			t.Errorf("at s=%v: SS=%v, TF=%v", s, ssVal, tfVal)
		}
	}
}

func TestStateSpaceSingularDenom(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1}}},
		Den: [][]float64{{0, 1}},
	}
	_, err := tf.StateSpace(nil)
	if err == nil {
		t.Fatal("expected error for near-zero leading coeff")
	}
}

func TestTFSSRoundtripFrequency(t *testing.T) {
	// Start with TF, go to SS, back to TF, compare at frequencies
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 1}}},
		Den: [][]float64{{1, 2, 1}},
	}
	ssRes, err := tf.StateSpace(nil)
	if err != nil {
		t.Fatal(err)
	}
	tfRes, err := ssRes.Sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{1i, 0.1i, 10i, complex(0.5, 1)}
	for _, s := range freqs {
		orig := tf.Eval(s)[0][0]
		rt := tfRes.TF.Eval(s)[0][0]
		if cmplx.Abs(orig-rt) > 1e-6 {
			t.Errorf("at s=%v: orig=%v, roundtrip=%v", s, orig, rt)
		}
	}
}

func TestTransferFunctionWithFeedthrough(t *testing.T) {
	// System with D != 0
	sys, err := NewFromSlices(1, 1, 1,
		[]float64{-1},
		[]float64{1},
		[]float64{1},
		[]float64{2},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{1i, 2i, complex(1, 1)}
	for _, s := range freqs {
		tfVal := res.TF.Eval(s)[0][0]
		ssVal := evalSS(sys, s)
		if cmplx.Abs(tfVal-ssVal) > 1e-10 {
			t.Errorf("at s=%v: TF=%v, SS=%v", s, tfVal, ssVal)
		}
	}
}

func TestTransferFunctionNonSymmetricA(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-6, -11, -6,
		},
		[]float64{0, 0, 1},
		[]float64{1, 0, 0},
		[]float64{0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{1i, 2i, complex(0.5, 1), complex(-0.3, 2.7)}
	for _, s := range freqs {
		tfVal := res.TF.Eval(s)[0][0]
		ssVal := evalSS(sys, s)
		if cmplx.Abs(tfVal-ssVal) > 1e-8 {
			t.Errorf("at s=%v: TF=%v, SS=%v", s, tfVal, ssVal)
		}
	}
}

func TestTransferFunctionNonSymmetricSIMO(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 2,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-6, -11, -6,
		},
		[]float64{0, 0, 1},
		[]float64{
			1, 0, 0,
			0, 1, 0,
		},
		[]float64{0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{1i, 2i, complex(0.5, 1)}
	for _, s := range freqs {
		tfMat := res.TF.Eval(s)
		for i := range 2 {
			ssVal := evalSSij(sys, s, i, 0)
			if cmplx.Abs(tfMat[i][0]-ssVal) > 1e-8 {
				t.Errorf("at s=%v [%d][0]: TF=%v, SS=%v", s, i, tfMat[i][0], ssVal)
			}
		}
	}
}

func TestTransferFunctionNonSymmetricMISO(t *testing.T) {
	sys, err := NewFromSlices(3, 2, 1,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-6, -11, -6,
		},
		[]float64{
			1, 0,
			0, 1,
			0, 0,
		},
		[]float64{1, 0, 0},
		[]float64{0, 0},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{1i, 2i, complex(0.5, 1)}
	for _, s := range freqs {
		tfMat := res.TF.Eval(s)
		for j := range 2 {
			ssVal := evalSSij(sys, s, 0, j)
			if cmplx.Abs(tfMat[0][j]-ssVal) > 1e-8 {
				t.Errorf("at s=%v [0][%d]: TF=%v, SS=%v", s, j, tfMat[0][j], ssVal)
			}
		}
	}
}

func TestTransferFunctionNonSymmetricMIMO(t *testing.T) {
	sys, err := NewFromSlices(3, 2, 2,
		[]float64{
			0, 1, 0,
			0, 0, 1,
			-6, -11, -6,
		},
		[]float64{
			1, 0,
			0, 1,
			1, 1,
		},
		[]float64{
			1, 0, 0,
			0, 1, 0,
		},
		[]float64{
			0, 0,
			0, 0,
		},
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	freqs := []complex128{1i, 2i, complex(0.5, 1)}
	for _, s := range freqs {
		tfMat := res.TF.Eval(s)
		for i := range 2 {
			for j := range 2 {
				ssVal := evalSSij(sys, s, i, j)
				if cmplx.Abs(tfMat[i][j]-ssVal) > 1e-8 {
					t.Errorf("at s=%v [%d][%d]: TF=%v, SS=%v", s, i, j, tfMat[i][j], ssVal)
				}
			}
		}
	}
}

// evalSS evaluates C*(sI-A)^{-1}*B + D for SISO at complex s
func evalSS(sys *System, s complex128) complex128 {
	n, _, _ := sys.Dims()
	if n == 0 {
		return complex(sys.D.At(0, 0), 0)
	}

	// Build sI - A as complex matrix, solve
	sIA := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			sIA[i*n+j] = -complex(sys.A.At(i, j), 0)
		}
		sIA[i*n+i] += s
	}

	// Solve (sI - A) * x = B for x, then y = C*x + D
	bVec := make([]complex128, n)
	for i := range n {
		bVec[i] = complex(sys.B.At(i, 0), 0)
	}

	x := complexSolve(sIA, bVec, n)

	result := complex(sys.D.At(0, 0), 0)
	for i := range n {
		result += complex(sys.C.At(0, i), 0) * x[i]
	}
	return result
}

// evalSSij evaluates (C*(sI-A)^{-1}*B + D)[i][j]
func evalSSij(sys *System, s complex128, oi, ij int) complex128 {
	n, _, _ := sys.Dims()
	if n == 0 {
		return complex(sys.D.At(oi, ij), 0)
	}

	sIA := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			sIA[i*n+j] = -complex(sys.A.At(i, j), 0)
		}
		sIA[i*n+i] += s
	}

	// Solve for column ij of B
	bVec := make([]complex128, n)
	for i := range n {
		bVec[i] = complex(sys.B.At(i, ij), 0)
	}
	x := complexSolve(sIA, bVec, n)

	result := complex(sys.D.At(oi, ij), 0)
	for i := range n {
		result += complex(sys.C.At(oi, i), 0) * x[i]
	}
	return result
}

// complexSolve solves Ax=b via Gaussian elimination for complex matrices
func complexSolve(A []complex128, b []complex128, n int) []complex128 {
	a := make([]complex128, n*n)
	copy(a, A)
	x := make([]complex128, n)
	copy(x, b)

	for k := range n {
		// Partial pivoting
		maxVal := cmplx.Abs(a[k*n+k])
		maxRow := k
		for i := k + 1; i < n; i++ {
			if v := cmplx.Abs(a[i*n+k]); v > maxVal {
				maxVal = v
				maxRow = i
			}
		}
		if maxRow != k {
			for j := range n {
				a[k*n+j], a[maxRow*n+j] = a[maxRow*n+j], a[k*n+j]
			}
			x[k], x[maxRow] = x[maxRow], x[k]
		}

		pivot := a[k*n+k]
		for i := k + 1; i < n; i++ {
			factor := a[i*n+k] / pivot
			for j := k + 1; j < n; j++ {
				a[i*n+j] -= factor * a[k*n+j]
			}
			x[i] -= factor * x[k]
		}
	}

	// Back substitution
	for k := n - 1; k >= 0; k-- {
		for j := k + 1; j < n; j++ {
			x[k] -= a[k*n+j] * x[j]
		}
		x[k] /= a[k*n+k]
	}

	return x
}
