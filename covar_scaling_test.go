package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func scaledDense(m *mat.Dense, e int) *mat.Dense {
	var out mat.Dense
	out.Apply(func(_, _ int, v float64) float64 { return math.Ldexp(v, e) }, m)
	return &out
}

func TestCovarExtremeMagnitudeScaling(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-2, 1, 0.5, 0, -3, 1, 0.25, -1, -4})
	B := mat.NewDense(3, 2, []float64{1, 0, -0.5, 2, 0.3, 1})
	C := mat.NewDense(2, 3, []float64{1, -1, 0.5, 0, 2, 1})
	D := mat.NewDense(2, 2, nil)
	W := mat.NewDense(2, 2, []float64{2, 0.5, 0.5, 1})
	Ad := mat.NewDense(3, 3, []float64{0.5, 0.2, 0, -0.1, 0.4, 0.3, 0.05, 0, -0.6})

	for _, dt := range []float64{0, 0.1} {
		a := A
		if dt > 0 {
			a = Ad
		}
		base, err := New(a, B, C, D, dt)
		if err != nil {
			t.Fatal(err)
		}
		P0, err := Covar(base, W)
		if err != nil {
			t.Fatal(err)
		}
		for _, sc := range []struct{ eB, eC, eW int }{
			{600, -600, 0},
			{520, 0, -600},
			{-600, 600, 0},
			{0, 0, 1000},
			{300, -500, 400},
		} {
			sys, err := New(a, scaledDense(B, sc.eB), scaledDense(C, sc.eC), D, dt)
			if err != nil {
				t.Fatal(err)
			}
			P, err := Covar(sys, scaledDense(W, sc.eW))
			if err != nil {
				t.Errorf("dt=%g %+v: %v", dt, sc, err)
				continue
			}
			want := scaledDense(P0, 2*sc.eB+2*sc.eC+sc.eW)
			if !mat.EqualApprox(P, want, 0) {
				t.Errorf("dt=%g %+v: P = %v, want %v", dt, sc, mat.Formatted(P), mat.Formatted(want))
			}
		}
	}
}

func TestCovarLargeInputStronglyStable(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-1e10}), mat.NewDense(1, 1, []float64{1e155}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	P, err := Covar(sys, mat.NewDense(1, 1, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	if got, want := P.At(0, 0), 5e299; math.Abs(got-want) > 1e-14*want {
		t.Errorf("P = %v, want %v", got, want)
	}
}

func TestCovarGenuineOverflow(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-1e-3}), mat.NewDense(1, 1, []float64{1e155}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if P, err := Covar(sys, mat.NewDense(1, 1, []float64{1})); !errors.Is(err, ErrOverflow) {
		t.Fatalf("P = %v, %v; want ErrOverflow", P, err)
	}
}
