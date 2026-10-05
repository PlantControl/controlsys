package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestFeedback_NegativeFeedback_StabilizesUnstable(t *testing.T) {
	plant, err := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	poles, err := plant.Poles()
	if err != nil {
		t.Fatal(err)
	}
	if real(poles[0]) <= 0 {
		t.Fatal("plant should be unstable")
	}

	controller, err := NewGain(mat.NewDense(1, 1, []float64{5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		clPoles, _ := cl.Poles()
		t.Errorf("closed-loop should be stable, poles = %v", clPoles)
	}

	clPoles, _ := cl.Poles()
	if math.Abs(real(clPoles[0])-(-4)) > 1e-8 {
		t.Errorf("CL pole = %v, want -4", clPoles[0])
	}
}

func TestFeedback_UnitFeedback_DCGain(t *testing.T) {
	plant, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{10}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	controller, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	if err != nil {
		t.Fatal(err)
	}

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	dc, err := cl.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	want := 10.0 / 11.0
	if math.Abs(dc.At(0, 0)-want) > 1e-6 {
		t.Errorf("DC gain = %v, want %v", dc.At(0, 0), want)
	}
}

func TestFeedback_HighGain_Bounded(t *testing.T) {
	plant, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{10.314}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	controller, err := NewGain(mat.NewDense(1, 1, []float64{500}), 0)
	if err != nil {
		t.Fatal(err)
	}

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Fatal("closed-loop should be stable")
	}

	resp, err := Step(cl, 5.0)
	if err != nil {
		t.Fatal(err)
	}

	_, cols := resp.Y.Dims()
	for k := range cols {
		if math.Abs(resp.Y.At(0, k)) > 100 {
			t.Fatalf("step response unbounded at t=%v: y=%v", resp.T[k], resp.Y.At(0, k))
		}
	}
}

func TestSeries_Cascade(t *testing.T) {
	sys1, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	sys2, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	cascade, err := Series(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	poles, err := cascade.Poles()
	if err != nil {
		t.Fatal(err)
	}
	matchRoots(t, poles, []complex128{-1, -2}, 1e-8)

	dc, err := cascade.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dc.At(0, 0)-0.5) > 1e-10 {
		t.Errorf("DC gain = %v, want 0.5", dc.At(0, 0))
	}
}

func TestParallel_Addition(t *testing.T) {
	sys1, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	sys2, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	par, err := Parallel(sys1, sys2)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := par.Dims()
	if n != 2 {
		t.Errorf("states = %d, want 2", n)
	}

	dc, err := par.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dc.At(0, 0)-1.5) > 1e-10 {
		t.Errorf("DC gain = %v, want 1.5", dc.At(0, 0))
	}
}

func TestFeedback_StaticGain(t *testing.T) {
	plant, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}

	controller, err := NewGain(mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	dc, err := cl.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dc.At(0, 0)-1.0) > 1e-10 {
		t.Errorf("DC gain = %v, want 1.0", dc.At(0, 0))
	}
}

func TestSS2TF_StrictlyProper(t *testing.T) {
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

	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	num := tfRes.TF.Num[0][0]
	den := tfRes.TF.Den[0]
	if !Poly(num).Equal(Poly{1}, 1e-10) {
		t.Errorf("num = %v, want [1]", num)
	}
	if !Poly(den).Equal(Poly{1, 1}, 1e-10) {
		t.Errorf("den = %v, want [1 1]", den)
	}

	ssRes, err := tfRes.TF.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	recoveredPoles, err := ssRes.Sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	matchRoots(t, recoveredPoles, []complex128{-1}, 1e-8)
}

func TestSS2TF_WithFeedthrough(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	dc, err := sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dc.At(0, 0)-2.5) > 1e-10 {
		t.Errorf("DC gain = %v, want 2.5", dc.At(0, 0))
	}

	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}

	s := complex(0, 1.0)
	tfVal := mustEval(t, tfRes.TF, s)[0][0]
	want := (complex(1, 0)*s + 5) / (s + 2)
	if cmplx.Abs(tfVal-want) > 1e-8 {
		t.Errorf("TF(j) = %v, want %v", tfVal, want)
	}
}

func TestFeedback_MIMO_2x2(t *testing.T) {
	plant, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	controller, err := NewGain(mat.NewDense(2, 2, []float64{1, 0, 0, 1}), 0)
	if err != nil {
		t.Fatal(err)
	}

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("closed-loop should be stable")
	}

	n, _, _ := cl.Dims()
	if n != 2 {
		t.Errorf("states = %d, want 2", n)
	}

	olPoles, _ := plant.Poles()
	clPoles, _ := cl.Poles()
	for i, clp := range clPoles {
		if real(clp) >= real(olPoles[i]) {
			t.Errorf("CL pole %v not more negative than OL pole %v", clp, olPoles[i])
		}
	}
}

func TestTF2SS_StaticGain(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{23}}},
		Den: [][]float64{{46}},
	}

	ssRes, err := tf.StateSpace()
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := ssRes.Sys.Dims()
	if n != 0 {
		t.Errorf("states = %d, want 0", n)
	}

	dc, err := ssRes.Sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dc.At(0, 0)-0.5) > 1e-10 {
		t.Errorf("DC gain = %v, want 0.5", dc.At(0, 0))
	}
}

func TestFeedback_Preserves_Discrete(t *testing.T) {
	plant, err := New(
		mat.NewDense(1, 1, []float64{0.9}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)
	if err != nil {
		t.Fatal(err)
	}

	controller, err := NewGain(mat.NewDense(1, 1, []float64{1}), 0.1)
	if err != nil {
		t.Fatal(err)
	}

	cl, err := Feedback(plant, controller, -1)
	if err != nil {
		t.Fatal(err)
	}

	if cl.Dt != 0.1 {
		t.Errorf("Dt = %v, want 0.1", cl.Dt)
	}
}
