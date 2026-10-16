package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestBuildDelayBankAlwaysReturnsModelOrError(t *testing.T) {
	if _, err := buildContinuousPadeDelayBank(nil, 3); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("no channels: err = %v, want ErrDimensionMismatch", err)
	}
	bank, err := buildContinuousPadeDelayBank([]float64{0.2, 0}, 3)
	if err != nil {
		t.Fatal(err)
	}
	if _, m, p := bank.Dims(); m != 2 || p != 2 {
		t.Errorf("bank dims %dx%d, want 2x2", p, m)
	}
}

// MATLAB c2d honours FractDelayApproxOrder only for tustin and matched; the
// other methods reject it, so no ThiranOrder reaches their delay handling.
func TestC2DThiranOrderRejectedForNonTustinMatched(t *testing.T) {
	const dt = 0.1
	sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.InputDelay = []float64{0.35}
	for _, method := range []C2DMethod{"", C2DMethodZOH, C2DMethodFOH, C2DMethodImpulse, C2DMethodLeastSquares} {
		for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
			opts := C2DOptions{Method: method, ThiranOrder: 1, DelayModeling: modeling}
			if out, err := sys.C2D(dt, opts); !errors.Is(err, ErrInvalidConversionOptions) || out != nil {
				t.Errorf("C2D %q %s: %v, %v; want nil, ErrInvalidConversionOptions", method, modeling, out, err)
			}
			if out, _, err := sys.C2DMap(dt, opts); !errors.Is(err, ErrInvalidConversionOptions) || out != nil {
				t.Errorf("C2DMap %q %s: %v, %v; want nil, ErrInvalidConversionOptions", method, modeling, out, err)
			}
		}
	}
	if out, _, err := sys.C2DFit(dt, C2DOptions{Method: C2DMethodLeastSquares, ThiranOrder: 1}); !errors.Is(err, ErrInvalidConversionOptions) || out != nil {
		t.Errorf("C2DFit: %v, %v; want nil, ErrInvalidConversionOptions", out, err)
	}
}
