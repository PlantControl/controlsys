package controlsys

import (
	"errors"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// 6N7XUA: Lsim reported every input-signal failure as a size mismatch.
func TestLsim_InputSignalErrorKeepsCause(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, -0.5, -3})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 0.3})
	D := mat.NewDense(1, 1, []float64{0.1})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	tg := []float64{0, 0.1, 0.2}
	planner := newTimeResponsePlanner(sys)

	for _, v := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		u := mat.NewDense(3, 1, []float64{0, v, 1})
		_, _, err := planner.lsim(u, tg)
		if !errors.Is(err, ErrInvalidArgument) || errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("u with %v: err = %v, want ErrInvalidArgument only", v, err)
		}
		if err != nil && strings.Contains(err.Error(), "must be") {
			t.Errorf("u with %v: err = %q reports a size mismatch", v, err)
		}
	}

	_, _, err = planner.lsim(mat.NewDense(2, 1, []float64{0, 1}), tg)
	if !errors.Is(err, ErrDimensionMismatch) || !strings.Contains(err.Error(), "u must be 3×1, got 2×1") {
		t.Errorf("wrong-size u: err = %v, want size mismatch", err)
	}

	if _, _, err := planner.lsim(mat.NewDense(3, 1, []float64{0, 1, 2}), tg); err != nil {
		t.Errorf("finite u: err = %v", err)
	}
}
