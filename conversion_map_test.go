package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestConversionInitialStateAnalytic(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{3}), mat.NewDense(1, 1, []float64{4}), mat.NewDense(1, 1, []float64{5}), 0)
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		t.Run(string(method), func(t *testing.T) {
			disc, G, err := sys.C2DMap(.1, C2DOptions{Method: method})
			if err != nil {
				t.Fatal(err)
			}
			xd := applyInitialMap(t, G, []float64{2}, []float64{7})
			want := 2.0
			if method == C2DMethodFOH {
				want -= 3 * (math.Exp(-.2) - 1 + .2) / .4 * 7
			}
			if method == C2DMethodTustin {
				want = (22*2 - 3*7) / 20.0
			}
			if math.Abs(xd[0]-want) > 1e-12 {
				t.Fatalf("state %g want %g", xd[0], want)
			}
			y := disc.C.At(0, 0)*xd[0] + disc.D.At(0, 0)*7
			if math.Abs(y-43) > 1e-11 {
				t.Fatalf("initial output %g want 43", y)
			}
			_, Gr, err := disc.D2CMap(D2COptions{Method: method})
			if err != nil {
				t.Fatal(err)
			}
			xc := applyInitialMap(t, Gr, xd, []float64{7})
			if math.Abs(xc[0]-2) > 1e-11 {
				t.Fatalf("reverse state %g want 2", xc[0])
			}
		})
	}
}

// applyInitialMap returns G·[parts...], treating an empty part as zeros of
// the remaining width when it is the last one.
func applyInitialMap(t *testing.T, G *mat.Dense, parts ...[]float64) []float64 {
	t.Helper()
	rows, cols := G.Dims()
	v := make([]float64, cols)
	off := 0
	for _, part := range parts {
		copy(v[off:], part)
		off += len(part)
	}
	if off > cols {
		t.Fatalf("map has %d columns, got %d values", cols, off)
	}
	out := make([]float64, rows)
	for i := range rows {
		for j := range cols {
			out[i] += G.At(i, j) * v[j]
		}
	}
	return out
}

func TestConversionMapUnavailable(t *testing.T) {
	sys := makeTestSystem()
	var nilSys *System
	if _, _, err := nilSys.C2DMap(.1, C2DOptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil C2DMap: %v", err)
	}
	if _, _, err := nilSys.D2CMap(D2COptions{}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil D2CMap: %v", err)
	}
	disc, G, err := sys.C2DMap(.1, C2DOptions{Method: C2DMethodMatched})
	if !errors.Is(err, ErrOptionUnsupported) || disc != nil || G != nil {
		t.Errorf("matched: sys=%v G=%v err=%v, want nil, nil, ErrOptionUnsupported", disc, G, err)
	}
	if _, err := sys.C2D(.1, C2DOptions{Method: C2DMethodMatched}); err != nil {
		t.Errorf("matched C2D: %v", err)
	}
	gain, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, _, err := gain.C2DMap(.1, C2DOptions{}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("static gain: %v, want ErrDimensionMismatch", err)
	}
}

func TestD2DInitialStateOutput(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{3}), mat.NewDense(1, 1, []float64{4}), mat.NewDense(1, 1, []float64{5}), 0)
	disc, err := sys.C2D(.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	cont, G1, err := disc.D2CMap(D2COptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	resampled, G2, err := cont.C2DMap(.25, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	direct, err := disc.D2D(.25, D2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	assertMatClose(t, "D2D A", direct.A, resampled.A, 1e-12)
	x := applyInitialMap(t, G2, applyInitialMap(t, G1, []float64{2}, []float64{7}), []float64{7})
	want := disc.C.At(0, 0)*2 + disc.D.At(0, 0)*7
	got := resampled.C.At(0, 0)*x[0] + resampled.D.At(0, 0)*7
	if math.Abs(got-want) > 1e-11 {
		t.Fatalf("output %g want %g", got, want)
	}
}

func TestImpulseInvariantSamplesIncludingZero(t *testing.T) {
	sys, _ := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 2, []float64{1, 2, 3, 4}), mat.NewDense(2, 2, []float64{2, 1, 1, -2}), mat.NewDense(2, 2, []float64{9, 8, 7, 6}), 0)
	disc, err := sys.C2D(.1, C2DOptions{Method: C2DMethodImpulse})
	if err != nil {
		t.Fatal(err)
	}
	for k := range 8 {
		e1, e3 := math.Exp(-float64(k)*.1), math.Exp(-float64(k)*.3)
		exp := mat.NewDense(2, 2, []float64{e1, e1 - e3, 0, e3})
		var expected, tmp mat.Dense
		tmp.Mul(exp, sys.B)
		expected.Mul(sys.C, &tmp)
		expected.Scale(.1, &expected)
		got := disc.D
		if k > 0 {
			var power mat.Dense
			power.Pow(disc.A, k-1)
			var response mat.Dense
			tmp.Mul(&power, disc.B)
			response.Mul(disc.C, &tmp)
			got = &response
		}
		assertMatClose(t, "sample", got, &expected, 1e-12)
	}
	assertMatClose(t, "source unchanged", sys.D, mat.NewDense(2, 2, []float64{9, 8, 7, 6}), 0)
}

func TestTustinInternalDelayInitialMap(t *testing.T) {
	sys := makeLFTSystem(t)
	disc, G, err := sys.C2DMap(.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	n, m, p := sys.Dims()
	x := make([]float64, n)
	u := make([]float64, m)
	w := make([]float64, sys.internalDelayCount())
	for i := range x {
		x[i] = float64(i + 1)
	}
	for i := range u {
		u[i] = .3
	}
	for i := range w {
		w[i] = .7
	}
	xd := applyInitialMap(t, G, x, u, w)
	for i := range p {
		yc, yd := 0., 0.
		for j := range n {
			yc += sys.C.At(i, j) * x[j]
			yd += disc.C.At(i, j) * xd[j]
		}
		for j := range m {
			yc += sys.D.At(i, j) * u[j]
			yd += disc.D.At(i, j) * u[j]
		}
		for j := range len(w) {
			yc += sys.LFT.D12.At(i, j) * w[j]
			yd += disc.LFT.D12.At(i, j) * w[j]
		}
		if math.Abs(yc-yd) > 1e-11 {
			t.Fatalf("initial internal output %g want %g", yd, yc)
		}
	}
}

func TestFOHInternalDelayInitialMapAndApproximation(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{-1, .5, 0, -2}), mat.NewDense(2, 1, []float64{1, .3}), mat.NewDense(1, 2, []float64{2, -.4}), mat.NewDense(1, 1, []float64{.2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	sys.LFT = &LFTDelay{Tau: []float64{.2}, B2: mat.NewDense(2, 1, []float64{.7, -.2}), C2: mat.NewDense(1, 2, []float64{.1, .3}), D12: mat.NewDense(1, 1, []float64{.4}), D21: mat.NewDense(1, 1, []float64{.2}), D22: mat.NewDense(1, 1, nil)}
	x, u, w := []float64{1, 2}, []float64{.3}, []float64{.7}
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		disc, G, err := sys.C2DMap(.1, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		xd := applyInitialMap(t, G, x, u, w)
		want := append([]float64(nil), x...)
		if method == C2DMethodFOH {
			const pieces = 512
			for k := 0; k <= pieces; k++ {
				at := .1 * float64(k) / pieces
				remaining := .1 - at
				e1, e2 := math.Exp(-remaining), math.Exp(-2*remaining)
				v1, v2 := 1*.3+.7*.7, .3*.3-.2*.7
				integrand := []float64{e1*v1 + .5*(e1-e2)*v2, e2 * v2}
				weight := 2.0
				if k == 0 || k == pieces {
					weight = 1
				} else if k%2 == 1 {
					weight = 4
				}
				for i := range want {
					want[i] -= weight * (.1 / pieces) / 3 * (at / .1) * integrand[i]
				}
			}
		}
		for i := range want {
			if math.Abs(xd[i]-want[i]) > 1e-12 {
				t.Fatalf("%s state[%d]=%g want %g", method, i, xd[i], want[i])
			}
		}
		yc := 2*x[0] - .4*x[1] + .2*u[0] + .4*w[0]
		yd := disc.C.At(0, 0)*xd[0] + disc.C.At(0, 1)*xd[1] + disc.D.At(0, 0)*u[0] + disc.LFT.D12.At(0, 0)*w[0]
		if math.Abs(yc-yd) > 1e-12 {
			t.Fatalf("%s initial output %g want %g", method, yd, yc)
		}
	}
}

func TestThiranMemoryRefusesUnavailableNonzeroInitialMap(t *testing.T) {
	for _, internal := range []bool{false, true} {
		sys := makeTestSystem()
		if internal {
			sys.LFT = &LFTDelay{Tau: []float64{.02}, B2: mat.NewDense(2, 1, []float64{.1, .2}), C2: mat.NewDense(1, 2, []float64{.2, .3}), D12: mat.NewDense(1, 1, nil), D21: mat.NewDense(1, 1, nil), D22: mat.NewDense(1, 1, nil)}
		} else {
			sys.InputDelay = []float64{.02}
		}
		for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
			opts := C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: modeling}
			if _, _, err := sys.C2DMap(.1, opts); !errors.Is(err, ErrOptionUnsupported) {
				t.Fatalf("internal=%v mode=%s: err = %v, want ErrOptionUnsupported", internal, modeling, err)
			}
			if _, err := sys.C2D(.1, opts); err != nil {
				t.Fatal(err)
			}
		}
	}
	sys := makeTestSystem()
	sys.InputDelay = []float64{.2}
	if _, _, err := sys.C2DMap(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3}); err != nil {
		t.Fatalf("integer delay has unchanged rational coordinates: %v", err)
	}
}

func TestConversionInitialStateMIMOIndependentResponse(t *testing.T) {
	const dt, steps = .1, 25
	a := [2][2]float64{{-1, .7}, {-.4, -2.5}}
	b := [2][2]float64{{1, .3}, {-.2, .8}}
	c := [2][2]float64{{1, -.6}, {.4, 1.2}}
	d := [2][2]float64{{.2, 0}, {-.1, .5}}
	sys, _ := New(mat.NewDense(2, 2, []float64{a[0][0], a[0][1], a[1][0], a[1][1]}), mat.NewDense(2, 2, []float64{b[0][0], b[0][1], b[1][0], b[1][1]}), mat.NewDense(2, 2, []float64{c[0][0], c[0][1], c[1][0], c[1][1]}), mat.NewDense(2, 2, []float64{d[0][0], d[0][1], d[1][0], d[1][1]}), 0)
	x0 := []float64{1.5, -.8}
	input := func(k int) [2]float64 {
		return [2]float64{math.Sin(.6*float64(k)) + .3, math.Cos(.9*float64(k)) - .2}
	}
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		t.Run(string(method), func(t *testing.T) {
			disc, G, err := sys.C2DMap(dt, C2DOptions{Method: method})
			if err != nil {
				t.Fatal(err)
			}
			u0 := input(0)
			xd := applyInitialMap(t, G, x0, u0[:])
			u := mat.NewDense(2, steps, nil)
			for k := range steps {
				v := input(k)
				u.Set(0, k, v[0])
				u.Set(1, k, v[1])
			}
			resp, err := disc.Simulate(u, mat.NewVecDense(2, xd), nil)
			if err != nil {
				t.Fatal(err)
			}
			x := x0
			deriv := func(x [2]float64, w [2]float64) [2]float64 {
				var dx [2]float64
				for i := range 2 {
					dx[i] = a[i][0]*x[0] + a[i][1]*x[1] + b[i][0]*w[0] + b[i][1]*w[1]
				}
				return dx
			}
			state := [2]float64{x[0], x[1]}
			for k := range steps - 1 {
				uk := input(k)
				for i := range 2 {
					want := c[i][0]*state[0] + c[i][1]*state[1] + d[i][0]*uk[0] + d[i][1]*uk[1]
					if got := resp.Y.At(i, k); math.Abs(got-want) > 1e-9 {
						t.Fatalf("sample%d y%d got%.12g want%.12g", k, i, got, want)
					}
				}
				next := input(k + 1)
				hold := func(s float64) [2]float64 {
					if method == C2DMethodZOH {
						return uk
					}
					return [2]float64{uk[0] + (next[0]-uk[0])*s/dt, uk[1] + (next[1]-uk[1])*s/dt}
				}
				const sub = 200
				h := dt / sub
				for j := range sub {
					s := float64(j) * h
					k1 := deriv(state, hold(s))
					k2 := deriv([2]float64{state[0] + h/2*k1[0], state[1] + h/2*k1[1]}, hold(s+h/2))
					k3 := deriv([2]float64{state[0] + h/2*k2[0], state[1] + h/2*k2[1]}, hold(s+h/2))
					k4 := deriv([2]float64{state[0] + h*k3[0], state[1] + h*k3[1]}, hold(s+h))
					for i := range 2 {
						state[i] += h / 6 * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])
					}
				}
			}
		})
	}
	t.Run("tustin", func(t *testing.T) {
		disc, G, err := sys.C2DMap(dt, C2DOptions{Method: C2DMethodTustin})
		if err != nil {
			t.Fatal(err)
		}
		u0 := []float64{.4, -.7}
		xd := applyInitialMap(t, G, x0, u0)
		for i := range 2 {
			yc := c[i][0]*x0[0] + c[i][1]*x0[1] + d[i][0]*u0[0] + d[i][1]*u0[1]
			yd := disc.C.At(i, 0)*xd[0] + disc.C.At(i, 1)*xd[1] + disc.D.At(i, 0)*u0[0] + disc.D.At(i, 1)*u0[1]
			if math.Abs(yc-yd) > 1e-11 {
				t.Fatalf("initial output%d %g want %g", i, yd, yc)
			}
		}
		_, Gr, err := disc.D2CMap(D2COptions{Method: C2DMethodTustin})
		if err != nil {
			t.Fatal(err)
		}
		xc := applyInitialMap(t, Gr, xd, u0)
		if math.Abs(xc[0]-x0[0]) > 1e-11 || math.Abs(xc[1]-x0[1]) > 1e-11 {
			t.Fatalf("roundtrip %v want %v", xc, x0)
		}
	})
}
