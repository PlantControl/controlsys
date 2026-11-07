package controlsys

import (
	"errors"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// HOVKXP: a single-input multi-output model was rejected with a message
// claiming it was not single-input; least squares is SISO-only, as in MATLAB.
func TestC2DLeastSquares_SingleInputMultiOutputErrorNamesSISO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	_, err = sys.C2D(0.1, C2DOptions{Method: C2DMethodLeastSquares})
	if !errors.Is(err, ErrNotSISO) {
		t.Fatalf("err = %v, want ErrNotSISO", err)
	}
	if msg := err.Error(); !strings.Contains(msg, "2×1") || !strings.Contains(msg, "want SISO") {
		t.Fatalf("err = %q, want dims 2×1 and SISO requirement", msg)
	}
}
