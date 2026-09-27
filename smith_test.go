package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSmithPredictor_FOPDT(t *testing.T) {
	plant, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	piA := mat.NewDense(1, 1, []float64{0})
	piB := mat.NewDense(1, 1, []float64{1})
	piC := mat.NewDense(1, 1, []float64{2})
	piD := mat.NewDense(1, 1, []float64{2})
	ctrl, err := New(piA, piB, piC, piD, 0)
	if err != nil {
		t.Fatal(err)
	}

	smith, err := SmithPredictor(ctrl, plant, 1.0, 3)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := smith.Dims()
	if m != 1 || p != 1 {
		t.Errorf("dims = (%d,%d,%d), want (_, 1, 1)", n, m, p)
	}
	if !smith.IsContinuous() {
		t.Error("smith predictor should be continuous")
	}

	cl, err := Feedback(smith, plant, -1)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		t.Error("closed-loop with delay-free plant should be stable")
	}
}

func TestSmithPredictor_DimensionCheck(t *testing.T) {
	ctrl, _ := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	model, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}),
		0,
	)

	_, err := SmithPredictor(ctrl, model, 1.0, 2)
	if err == nil {
		t.Error("expected dimension mismatch error")
	}
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got: %v", err)
	}
}

func TestSmithPredictor_InvalidDelay(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)

	_, err := SmithPredictor(sys, sys, 0, 2)
	if err == nil {
		t.Error("expected error for delay=0")
	}

	_, err = SmithPredictor(sys, sys, -1, 2)
	if err == nil {
		t.Error("expected error for delay<0")
	}
}

func TestSmithPredictor_DiscreteError(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0.1,
	)

	_, err := SmithPredictor(sys, sys, 1.0, 2)
	if err == nil {
		t.Error("expected error for discrete system")
	}
	if !errors.Is(err, ErrWrongDomain) {
		t.Errorf("expected ErrWrongDomain, got: %v", err)
	}
}

func TestSmithPredictor_Stability(t *testing.T) {
	plant, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	ctrl, err := New(
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{3}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	plantDelayed := plant.Copy()
	_ = plantDelayed.SetDelay(mat.NewDense(1, 1, []float64{2.0}))
	plantPade, err := plantDelayed.Pade(5)
	if err != nil {
		t.Fatal(err)
	}

	clNoSmith, err := Feedback(ctrl, plantPade, -1)
	if err != nil {
		t.Fatal(err)
	}
	stableNoSmith, err := clNoSmith.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if stableNoSmith {
		t.Skip("standard feedback is already stable; cannot test Smith stabilization advantage")
	}

	smith, err := SmithPredictor(ctrl, plant, 2.0, 5)
	if err != nil {
		t.Fatal(err)
	}

	clSmith, err := Feedback(smith, plant, -1)
	if err != nil {
		t.Fatal(err)
	}
	stableSmith, err := clSmith.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stableSmith {
		t.Error("Smith predictor should stabilize the closed-loop with delay-free plant")
	}
}
