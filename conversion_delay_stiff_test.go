package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// firstOrderZOHState is x(t) of x' = a·x + u, x(0)=0, u held at u[k] on
// [k·dt, (k+1)·dt), u = 0 for t < 0.
func firstOrderZOHState(a, dt float64, u []float64, t float64) float64 {
	if t <= 0 {
		return 0
	}
	x, s := 0., 0.
	for k := 0; s < t; k++ {
		end := math.Min(float64(k+1)*dt, t)
		uk := 0.
		if k < len(u) {
			uk = u[k]
		}
		x = math.Exp(a*(end-s))*x + uk*math.Expm1(a*(end-s))/a
		s = end
	}
	return x
}

func TestC2DZOHFractionalOutputDelayStiffStep(t *testing.T) {
	const tau = 4.7471708
	for _, a := range []float64{-1e5, -1e6, -4e6} {
		sys, err := NewFromSlices(1, 1, 1, []float64{a}, []float64{1}, []float64{-a}, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetOutputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
		r, err := Step(sys, 20)
		if err != nil {
			t.Fatal(err)
		}
		for k, tk := range r.T {
			want := 0.
			if tk >= tau {
				want = -math.Expm1(a * (tk - tau))
			}
			if got := r.Y.At(0, k); !(math.Abs(got-want) <= 1e-9) {
				t.Fatalf("a=%g k=%d t=%g: y=%g want %g", a, k, tk, got, want)
			}
		}
	}
}

func TestC2DZOHFractionalOutputDelayLongResidual(t *testing.T) {
	const a, dt, tau = -1.58, 100., 50.
	sys, err := NewFromSlices(1, 1, 1, []float64{a}, []float64{1}, []float64{1}, []float64{0.3}, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetOutputDelay([]float64{tau}); err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay([]float64{20}); err != nil {
		t.Fatal(err)
	}
	N := 12
	T := make([]float64, N)
	uv := make([]float64, N)
	for k := range N {
		T[k] = float64(k) * dt
		uv[k] = math.Sin(0.7*float64(k)) + 0.2*float64(k%3)
	}
	r, err := Lsim(sys, mat.NewDense(N, 1, uv), T, nil)
	if err != nil {
		t.Fatal(err)
	}
	for k := range N {
		s := T[k] - tau - 20
		want := firstOrderZOHState(a, dt, uv, s)
		if s >= 0 {
			want += 0.3 * uv[int(math.Floor(s/dt))]
		}
		if got := r.Y.At(0, k); !(math.Abs(got-want) <= 1e-12) {
			t.Fatalf("k=%d: y=%g want %g", k, got, want)
		}
	}
	for _, input := range []float64{0, 20, 49.999, 50, 70} {
		sys.InputDelay = []float64{input}
		disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
		if err != nil {
			t.Fatal(err)
		}
		if c := disc.C.At(0, 0); !(math.Abs(c) <= 1) {
			t.Fatalf("input delay %g: |C|=%g > 1 (backward e^{-Aρ})", input, c)
		}
		resp, err := disc.Simulate(mat.NewDense(1, N, uv), nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		for k := range N {
			s := float64(k)*dt - tau - input
			want := firstOrderZOHState(a, dt, uv, s)
			if s >= 0 {
				want += 0.3 * uv[int(math.Floor(s/dt))]
			}
			if got := resp.Y.At(0, k); !(math.Abs(got-want) <= 1e-12) {
				t.Fatalf("input delay %g k=%d: y=%g want %g", input, k, got, want)
			}
		}
	}
}
