package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestCovarNoOutputsErrors(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}), nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Covar(sys, mat.NewDense(1, 1, []float64{1})); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("Covar on 0x1 model: err = %v, want ErrDimensionMismatch", err)
	}
}
