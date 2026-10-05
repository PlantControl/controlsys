package controlsys

import (
	"errors"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSynthesisRejectsInvalidPlant(t *testing.T) {
	plant := func() *System {
		P, err := New(
			mat.NewDense(2, 2, []float64{-1, 1, 0, -2}),
			mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
			mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
			mat.NewDense(2, 2, []float64{0, 1, 1, 0}),
			0,
		)
		if err != nil {
			t.Fatal(err)
		}
		return P
	}
	syn := map[string]func(*System, int, int) error{
		"H2Syn":   func(P *System, nm, nc int) error { _, err := H2Syn(P, nm, nc); return err },
		"HinfSyn": func(P *System, nm, nc int) error { _, err := HinfSyn(P, nm, nc); return err },
	}
	for op, run := range syn {
		if err := run(plant(), 1, 1); err != nil {
			t.Fatalf("%s delay-free plant: %v", op, err)
		}
		inDelay := plant()
		inDelay.InputDelay = []float64{0, 5}
		outDelay := plant()
		outDelay.OutputDelay = []float64{0.5, 0}
		nanD := plant()
		nanD.D.Set(0, 0, math.NaN())
		cases := []struct {
			name string
			P    *System
			nm   int
			want error
		}{
			{"input delay", inDelay, 1, ErrDelayUnsupported},
			{"output delay", outDelay, 1, ErrDelayUnsupported},
			{"nil", nil, 1, ErrInvalidArgument},
			{"NaN D", nanD, 1, ErrInvalidArgument},
			{"bad partition", plant(), 3, ErrInvalidPartition},
		}
		for _, tc := range cases {
			err := run(tc.P, tc.nm, 1)
			if !errors.Is(err, tc.want) {
				t.Errorf("%s %s: error = %v, want %v", op, tc.name, err, tc.want)
				continue
			}
			if !strings.HasPrefix(err.Error(), op+": ") {
				t.Errorf("%s %s: error %q lacks %q prefix", op, tc.name, err, op+": ")
			}
		}
	}
}

// I + D22·Dk that cancels to rounding noise is scale-free well conditioned,
// so a plain inversion accepted it and returned a garbage controller.
func TestNewControllerRejectsCancelledLoopShift(t *testing.T) {
	D22 := mat.NewDense(2, 2, []float64{0.6, 0.2, -0.3, 0.4})
	D := mat.NewDense(3, 3, nil)
	D.Set(0, 0, 1)
	D.Slice(1, 3, 1, 3).(*mat.Dense).Copy(D22)
	D.Set(1, 0, 1)
	D.Set(0, 1, 1)
	P, err := New(
		mat.NewDense(2, 2, []float64{-1, 0.3, 0.2, -2}),
		mat.NewDense(2, 3, []float64{1, 0, 0.5, 0, 1, 1}),
		mat.NewDense(3, 2, []float64{1, 0, 0.4, 1, 0, 1}),
		D, 0,
	)
	if err != nil {
		t.Fatal(err)
	}
	gp, err := partitionGeneralizedPlant("test", P, 2, 2)
	if err != nil {
		t.Fatal(err)
	}
	var Dk mat.Dense
	if err := Dk.Inverse(D22); err != nil {
		t.Fatal(err)
	}
	Dk.Scale(-1, &Dk)
	Dk.Add(&Dk, mat.NewDense(2, 2, []float64{3e-15, -2e-15, 1e-15, 4e-15}))
	n := 2
	_, err = gp.newController(mat.NewDense(n, n, []float64{-1, 0, 0, -1}), mat.NewDense(n, 2, []float64{1, 0, 0, 1}), mat.NewDense(2, n, []float64{1, 0, 0, 1}), &Dk)
	if !errors.Is(err, ErrAlgebraicLoop) {
		t.Fatalf("err = %v, want ErrAlgebraicLoop", err)
	}
}
