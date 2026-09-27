package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestLoopsens_FirstOrder(t *testing.T) {
	P, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	C, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	res, err := Loopsens(P, C)
	if err != nil {
		t.Fatal(err)
	}

	sDC, _ := res.So.DCGain()
	if math.Abs(sDC.At(0, 0)-2.0/3.0) > 1e-8 {
		t.Errorf("dcgain(S) = %g, want 2/3", sDC.At(0, 0))
	}

	tDC, _ := res.To.DCGain()
	if math.Abs(tDC.At(0, 0)-1.0/3.0) > 1e-8 {
		t.Errorf("dcgain(T) = %g, want 1/3", tDC.At(0, 0))
	}

	for _, w := range []float64{0.01, 0.1, 1, 10, 100} {
		sResp, _ := res.So.EvalFr(complex(0, w))
		tResp, _ := res.To.EvalFr(complex(0, w))
		sum := sResp[0][0] + tResp[0][0]
		if cmplx.Abs(sum-1) > 1e-6 {
			t.Errorf("w=%g: So+To = %v, want 1", w, sum)
		}
		siResp, _ := res.Si.EvalFr(complex(0, w))
		tiResp, _ := res.Ti.EvalFr(complex(0, w))
		sumI := siResp[0][0] + tiResp[0][0]
		if cmplx.Abs(sumI-1) > 1e-6 {
			t.Errorf("w=%g: Si+Ti = %v, want 1", w, sumI)
		}
	}
}

func TestLoopsens_Integrator(t *testing.T) {
	P, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	C, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)

	res, err := Loopsens(P, C)
	if err != nil {
		t.Fatal(err)
	}

	tDC, _ := res.To.DCGain()
	if math.Abs(tDC.At(0, 0)-1.0) > 1e-8 {
		t.Errorf("dcgain(T) = %g, want 1", tDC.At(0, 0))
	}
}

func TestLoopsens_NilError(t *testing.T) {
	C, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	_, err := Loopsens(nil, C)
	if err == nil {
		t.Error("expected error for nil P")
	}
}

func TestLoopsens_MIMO_SiNotEqualSo(t *testing.T) {
	// Non-symmetric MIMO: P and C such that P*C ≠ C*P
	P, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 2, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}), 0)
	C, _ := New(
		mat.NewDense(1, 1, []float64{-3}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(2, 1, []float64{1, 0.5}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0)

	res, err := Loopsens(P, C)
	if err != nil {
		t.Fatal(err)
	}

	w := 1.0
	s := complex(0, w)
	soResp, _ := res.So.EvalFr(s)
	siResp, _ := res.Si.EvalFr(s)

	differ := false
	for i := range soResp {
		for j := range soResp[i] {
			if cmplx.Abs(soResp[i][j]-siResp[i][j]) > 1e-6 {
				differ = true
			}
		}
	}
	if !differ {
		t.Error("Si should differ from So for non-commutative P*C")
	}

	// Still verify So+To=I and Si+Ti=I
	toResp, _ := res.To.EvalFr(s)
	tiResp, _ := res.Ti.EvalFr(s)
	n := len(soResp)
	for i := range n {
		for j := range n {
			eye := complex(0, 0)
			if i == j {
				eye = 1
			}
			sumO := soResp[i][j] + toResp[i][j]
			if cmplx.Abs(sumO-eye) > 1e-6 {
				t.Errorf("(So+To)[%d][%d] = %v, want %v", i, j, sumO, eye)
			}
			sumI := siResp[i][j] + tiResp[i][j]
			if cmplx.Abs(sumI-eye) > 1e-6 {
				t.Errorf("(Si+Ti)[%d][%d] = %v, want %v", i, j, sumI, eye)
			}
		}
	}
}
