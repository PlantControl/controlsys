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

// zohChannelState is x(t) of x' = A·x + b·u, x(0)=0, u held at u[k] on
// [k·dt, (k+1)·dt), u = 0 for t < 0, by exact segment exponentials.
func zohChannelState(a *mat.Dense, b []float64, dt float64, u []float64, t float64) []float64 {
	n, _ := a.Dims()
	x := make([]float64, n)
	if t <= 0 {
		return x
	}
	col := mat.NewDense(n, 1, b)
	for k, s := 0, 0.; s < t; k++ {
		end := math.Min(float64(k+1)*dt, t)
		uk := 0.
		if k < len(u) {
			uk = u[k]
		}
		ad, bd := conversionZOHKernel(a, col, end-s)
		next := make([]float64, n)
		for i := range n {
			for l := range n {
				next[i] += ad.At(i, l) * x[l]
			}
			next[i] += bd.At(i, 0) * uk
		}
		x, s = next, end
	}
	return x
}

func TestC2DZOHFractionalOutputDelayMIMO(t *testing.T) {
	const dt = 0.5
	a := mat.NewDense(3, 3, []float64{-0.4, 2, 0, -1.5, -0.3, 0.7, 0.2, 0, -30})
	b := mat.NewDense(3, 2, []float64{1, 0, 0.5, -1, 0, 2})
	c := mat.NewDense(2, 3, []float64{1, 0, 0.3, -0.2, 1, 1})
	d := mat.NewDense(2, 2, []float64{0.1, 0, -0.4, 0.25})
	sys, err := New(a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	const N = 30
	uv := [2][]float64{make([]float64, N), make([]float64, N)}
	for k := range N {
		uv[0][k] = math.Sin(0.9*float64(k)) + 0.3
		uv[1][k] = math.Cos(0.4*float64(k)) - 0.2*float64(k%2)
	}
	u := mat.NewDense(2, N, append(append([]float64{}, uv[0]...), uv[1]...))
	for _, delays := range [][2][2]float64{
		{{0, 0.2}, {1.35, 0.5}},
		{{0.3, 0.45}, {0.1, 2.05}},
		{{0.25, 0.4}, {0.25, 0.3}},
		{{1.7, 0}, {0.45, 0.45}},
	} {
		in, out := delays[0], delays[1]
		sys.InputDelay = []float64{in[0], in[1]}
		sys.OutputDelay = []float64{out[0], out[1]}
		disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodZOH})
		if err != nil {
			t.Fatal(err)
		}
		resp, err := disc.Simulate(u, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i := range 2 {
			for k := range N {
				want := 0.
				for j := range 2 {
					s := float64(k)*dt - out[i] - in[j]
					x := zohChannelState(a, []float64{b.At(0, j), b.At(1, j), b.At(2, j)}, dt, uv[j], s)
					for l := range 3 {
						want += c.At(i, l) * x[l]
					}
					if s >= 0 {
						want += d.At(i, j) * uv[j][int(math.Floor(s/dt+1e-12))]
					}
				}
				if got := resp.Y.At(i, k); !(math.Abs(got-want) <= 1e-12*(1+math.Abs(want))) {
					t.Fatalf("delays %v y%d[%d]=%.17g want %.17g", delays, i, k, got, want)
				}
			}
		}
	}
}
