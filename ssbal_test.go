package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSsbal_PoorlyConditioned(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{1000, 1, 0, 0.001}),
		mat.NewDense(2, 1, []float64{1000, 0.001}),
		mat.NewDense(1, 2, []float64{1, 1000}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-8)

	for _, w := range []float64{0.01, 0.1, 1, 10} {
		g1, _ := sys.EvalFr(complex(0, w))
		g2, _ := res.Sys.EvalFr(complex(0, w))
		diff := math.Abs(real(g1[0][0]) - real(g2[0][0]))
		relDiff := diff
		if math.Abs(real(g1[0][0])) > 1e-10 {
			relDiff = diff / math.Abs(real(g1[0][0]))
		}
		if relDiff > 1e-6 {
			t.Errorf("w=%g: freq response mismatch rel=%g", w, relDiff)
		}
	}
}

func TestSsbal_AlreadyBalanced(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	for i := range n {
		d := res.T.At(i, i)
		if math.Abs(d-1) > 1 {
			t.Errorf("T[%d,%d] = %g, expected near 1 for already balanced system", i, i, d)
		}
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
}

func TestSsbal_Empty(t *testing.T) {
	sys, err := New(nil, nil, nil, mat.NewDense(1, 1, []float64{3}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	dc, _ := res.Sys.DCGain()
	if dc.At(0, 0) != 3 {
		t.Errorf("dcgain = %g, want 3", dc.At(0, 0))
	}
}

func TestSsbal_TransformIsDiagonal(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1, 100, 0,
			0.01, -2, 50,
			0, 0.02, -3,
		}),
		mat.NewDense(3, 1, []float64{100, 1, 0.01}),
		mat.NewDense(1, 3, []float64{0.01, 1, 100}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Ssbal(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := sys.Dims()
	for i := range n {
		for j := range n {
			if i != j && res.T.At(i, j) != 0 {
				t.Errorf("T[%d,%d] = %g, want 0 (T should be diagonal)", i, j, res.T.At(i, j))
			}
		}
		if res.T.At(i, i) <= 0 {
			t.Errorf("T[%d,%d] = %g, want positive", i, i, res.T.At(i, i))
		}
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-8)
}
