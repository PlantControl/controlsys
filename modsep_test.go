package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestModsep_ClearSeparation(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -10, 0, 0, 0, -100}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 5)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 1 {
		t.Errorf("slow order = %d, want 1", ns)
	}
	if nf != 2 {
		t.Errorf("fast order = %d, want 2", nf)
	}

	slowPoles, _ := res.Slow.Poles()
	for _, p := range slowPoles {
		if cmplx.Abs(p) >= 5 {
			t.Errorf("slow pole %v has |λ| >= 5", p)
		}
	}

	fastPoles, _ := res.Fast.Poles()
	for _, p := range fastPoles {
		if cmplx.Abs(p) < 5 {
			t.Errorf("fast pole %v has |λ| < 5", p)
		}
	}

	checkAdditiveDecomposition(t, sys, res.Slow, res.Fast)
}

func TestModsep_DifferentCutoff(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -10, 0, 0, 0, -100}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 50)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 2 {
		t.Errorf("slow order = %d, want 2", ns)
	}
	if nf != 1 {
		t.Errorf("fast order = %d, want 1", nf)
	}

	checkAdditiveDecomposition(t, sys, res.Slow, res.Fast)
}

func TestModsep_AllFast(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -10, 0, 0, 0, -100}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 0 {
		t.Errorf("slow order = %d, want 0", ns)
	}
	if nf != 3 {
		t.Errorf("fast order = %d, want 3", nf)
	}
}

func TestModsep_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0.1, 0, 0, 0, 0.5, 0, 0, 0, 0.99}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 0.3)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 1 {
		t.Errorf("slow order = %d, want 1", ns)
	}
	if nf != 2 {
		t.Errorf("fast order = %d, want 2", nf)
	}

	checkAdditiveDecomposition(t, sys, res.Slow, res.Fast)
}

func TestModsep_InvalidCutoff(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Modsep(sys, -1)
	if err == nil {
		t.Error("expected error for negative cutoff")
	}
}

func TestModsep_Empty(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	res, err := Modsep(sys, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	if ns != 0 {
		t.Errorf("slow order = %d, want 0", ns)
	}
}

func TestModsep_NonDiagonal(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1, 0.5, 0,
			0, -10, 0,
			0, 0, -100,
		}),
		mat.NewDense(3, 1, []float64{1, 1, 1}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 5)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 1 {
		t.Errorf("slow order = %d, want 1", ns)
	}
	if nf != 2 {
		t.Errorf("fast order = %d, want 2", nf)
	}

	checkAdditiveDecomposition(t, sys, res.Slow, res.Fast)
}
