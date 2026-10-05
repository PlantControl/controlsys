package controlsys

import (
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPzmapGuardsInput(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, math.NaN(), -3}), mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{1, 1}), nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	finishesWithin(t, 5*time.Second, func() { _, err = Pzmap(sys) })
	if !errors.Is(err, ErrInvalidArgument) || !strings.HasPrefix(err.Error(), "Pzmap: ") {
		t.Fatalf("NaN err = %v", err)
	}
	if _, err := Pzmap(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil err = %v", err)
	}
}
