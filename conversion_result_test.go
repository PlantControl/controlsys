package controlsys

import (
	"errors"
	"math"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestConversionInitialStateAnalytic(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{3}), mat.NewDense(1, 1, []float64{4}), mat.NewDense(1, 1, []float64{5}), 0)
	for _, method := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin} {
		t.Run(string(method), func(t *testing.T) {
			result, err := sys.DiscretizeWithResult(.1, C2DOptions{Method: method})
			if err != nil {
				t.Fatal(err)
			}
			xd, err := result.MapInitialState([]float64{2}, []float64{7}, nil)
			if err != nil {
				t.Fatal(err)
			}
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
			y := result.System.C.At(0, 0)*xd[0] + result.System.D.At(0, 0)*7
			if math.Abs(y-43) > 1e-11 {
				t.Fatalf("initial output %g want 43", y)
			}
			reverse, err := result.System.D2CWithResult(D2COptions{Method: method})
			if err != nil {
				t.Fatal(err)
			}
			xc, err := reverse.MapInitialState(xd, []float64{7}, nil)
			if err != nil {
				t.Fatal(err)
			}
			if math.Abs(xc[0]-2) > 1e-11 {
				t.Fatalf("reverse state %g want 2", xc[0])
			}
		})
	}
}

func TestConversionMappingValidation(t *testing.T) {
	sys := makeTestSystem()
	result, err := sys.DiscretizeWithResult(.1, C2DOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if _, err = result.MapInitialState([]float64{1}, nil, nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Fatalf("dimension: %v", err)
	}
	if _, err = result.MapInitialState(nil, []float64{math.NaN()}, nil); !errors.Is(err, ErrInvalidConversionOptions) {
		t.Fatalf("finite: %v", err)
	}
	result.InitialStateMap = nil
	if _, err = result.MapInitialState([]float64{1, 0}, nil, nil); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("unavailable map: %v", err)
	}
}

func TestD2DInitialStateOutput(t *testing.T) {
	sys, _ := New(mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{3}), mat.NewDense(1, 1, []float64{4}), mat.NewDense(1, 1, []float64{5}), 0)
	disc, err := sys.DiscretizeWithOpts(.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	result, err := disc.D2DWithResult(.25, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	x, err := result.MapInitialState([]float64{2}, []float64{7}, nil)
	if err != nil {
		t.Fatal(err)
	}
	want := disc.C.At(0, 0)*2 + disc.D.At(0, 0)*7
	got := result.System.C.At(0, 0)*x[0] + result.System.D.At(0, 0)*7
	if math.Abs(got-want) > 1e-11 {
		t.Fatalf("output %g want %g", got, want)
	}
}

func TestImpulseInvariantSamplesIncludingZero(t *testing.T) {
	sys, _ := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 2, []float64{1, 2, 3, 4}), mat.NewDense(2, 2, []float64{2, 1, 1, -2}), mat.NewDense(2, 2, []float64{9, 8, 7, 6}), 0)
	disc, err := sys.DiscretizeImpulse(.1)
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
	result, err := sys.DiscretizeWithResult(.1, C2DOptions{Method: C2DMethodTustin})
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
	xd, err := result.MapInitialState(x, u, w)
	if err != nil {
		t.Fatal(err)
	}
	for i := range p {
		yc, yd := 0., 0.
		for j := range n {
			yc += sys.C.At(i, j) * x[j]
			yd += result.System.C.At(i, j) * xd[j]
		}
		for j := range m {
			yc += sys.D.At(i, j) * u[j]
			yd += result.System.D.At(i, j) * u[j]
		}
		for j := range len(w) {
			yc += sys.LFT.D12.At(i, j) * w[j]
			yd += result.System.LFT.D12.At(i, j) * w[j]
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
		result, err := sys.DiscretizeWithResult(.1, C2DOptions{Method: method})
		if err != nil {
			t.Fatal(err)
		}
		if !result.Approximate {
			t.Fatal("internal hold conversion must report approximation")
		}
		xd, err := result.MapInitialState(x, u, w)
		if err != nil {
			t.Fatal(err)
		}
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
		yd := result.System.C.At(0, 0)*xd[0] + result.System.C.At(0, 1)*xd[1] + result.System.D.At(0, 0)*u[0] + result.System.LFT.D12.At(0, 0)*w[0]
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
			result, err := sys.DiscretizeWithResult(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3, DelayModeling: modeling})
			if err != nil {
				t.Fatal(err)
			}
			if result.InitialStateMap != nil {
				t.Fatalf("internal=%v mode=%s advertised unavailable map", internal, modeling)
			}
			if _, err := result.MapInitialState([]float64{1, 0}, nil, nil); !errors.Is(err, ErrSingularTransform) {
				t.Fatalf("nonzero map %v", err)
			}
			if _, err := result.MapInitialState(nil, []float64{1}, nil); !errors.Is(err, ErrSingularTransform) {
				t.Fatalf("nonzero input %v", err)
			}
			if _, err := result.MapInitialState(nil, nil, nil); err != nil {
				t.Fatal(err)
			}
		}
	}
	sys := makeTestSystem()
	sys.InputDelay = []float64{.2}
	result, err := sys.DiscretizeWithResult(.1, C2DOptions{Method: C2DMethodTustin, ThiranOrder: 3})
	if err != nil {
		t.Fatal(err)
	}
	if result.InitialStateMap == nil {
		t.Fatal("integer delay has unchanged rational coordinates")
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
			result, err := sys.DiscretizeWithResult(dt, C2DOptions{Method: method})
			if err != nil {
				t.Fatal(err)
			}
			u0 := input(0)
			xd, err := result.MapInitialState(x0, u0[:], nil)
			if err != nil {
				t.Fatal(err)
			}
			u := mat.NewDense(2, steps, nil)
			for k := range steps {
				v := input(k)
				u.Set(0, k, v[0])
				u.Set(1, k, v[1])
			}
			resp, err := result.System.Simulate(u, mat.NewVecDense(2, xd), nil)
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
		result, err := sys.DiscretizeWithResult(dt, C2DOptions{Method: C2DMethodTustin})
		if err != nil {
			t.Fatal(err)
		}
		u0 := []float64{.4, -.7}
		xd, err := result.MapInitialState(x0, u0, nil)
		if err != nil {
			t.Fatal(err)
		}
		for i := range 2 {
			yc := c[i][0]*x0[0] + c[i][1]*x0[1] + d[i][0]*u0[0] + d[i][1]*u0[1]
			yd := result.System.C.At(i, 0)*xd[0] + result.System.C.At(i, 1)*xd[1] + result.System.D.At(i, 0)*u0[0] + result.System.D.At(i, 1)*u0[1]
			if math.Abs(yc-yd) > 1e-11 {
				t.Fatalf("initial output%d %g want %g", i, yd, yc)
			}
		}
		reverse, err := result.System.D2CWithResult(D2COptions{Method: C2DMethodTustin})
		if err != nil {
			t.Fatal(err)
		}
		xc, err := reverse.MapInitialState(xd, u0, nil)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(xc[0]-x0[0]) > 1e-11 || math.Abs(xc[1]-x0[1]) > 1e-11 {
			t.Fatalf("roundtrip %v want %v", xc, x0)
		}
	})
}

func TestConversionResultReportsDelayApproximation(t *testing.T) {
	scalar := func(v float64) *mat.Dense { return mat.NewDense(1, 1, []float64{v}) }
	cases := []struct {
		method C2DMethod
		thiran int
		tau    float64
		want   string
	}{
		{C2DMethodTustin, 0, .24, "rounded to the nearest sample"},
		{C2DMethodMatched, 0, .24, "rounded to the nearest sample"},
		{C2DMethodTustin, 3, .24, "Thiran filters approximate"},
		{C2DMethodTustin, 0, .2, ""},
		{C2DMethodZOH, 0, .24, ""},
	}
	for _, tc := range cases {
		sys, _ := New(scalar(-1), scalar(1), scalar(1), scalar(0), 0)
		sys.InputDelay = []float64{tc.tau}
		result, err := sys.DiscretizeWithResult(.1, C2DOptions{Method: tc.method, ThiranOrder: tc.thiran})
		if err != nil {
			t.Fatal(err)
		}
		found := ""
		for _, w := range result.Warnings {
			if strings.Contains(w, "rounded to the nearest sample") || strings.Contains(w, "Thiran filters approximate") {
				found = w
			}
		}
		if tc.want == "" && found != "" || tc.want != "" && !strings.Contains(found, tc.want) {
			t.Fatalf("%s thiran%d tau%g warnings %q", tc.method, tc.thiran, tc.tau, result.Warnings)
		}
		if tc.want != "" && !result.Approximate {
			t.Fatalf("%s tau%g not approximate", tc.method, tc.tau)
		}
	}
	sys, _ := New(scalar(-1), scalar(1), scalar(1), scalar(0), 0)
	if err := sys.SetInternalDelay([]float64{.15}, scalar(.2), scalar(1), scalar(0), scalar(0), scalar(0)); err != nil {
		t.Fatal(err)
	}
	result, err := sys.DiscretizeWithResult(.1, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(strings.Join(result.Warnings, " "), "rounded to the nearest sample") {
		t.Fatalf("internal rounding unreported: %q", result.Warnings)
	}
}
