package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestParallelEqualTotalDelaySplitDifferently(t *testing.T) {
	mk := func(t *testing.T, n, m, p int, a, b, c, d []float64, dt float64, in, out []float64, del *mat.Dense) *System {
		t.Helper()
		sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
		if err != nil {
			t.Fatal(err)
		}
		sys.InputDelay = in
		sys.OutputDelay = out
		sys.Delay = del
		return sys
	}
	siso := func(t *testing.T, a, dt float64, in, out []float64, del *mat.Dense) *System {
		return mk(t, 1, 1, 1, []float64{a}, []float64{1}, []float64{1}, []float64{0}, dt, in, out, del)
	}
	mimo := func(t *testing.T, a []float64, in, out []float64, del *mat.Dense) *System {
		return mk(t, 2, 2, 2, a, []float64{1, 0.3, -0.2, 1}, []float64{1, 0.5, 0.4, -1}, []float64{0.1, 0, 0.2, -0.3}, 0, in, out, del)
	}

	cases := []struct {
		name   string
		g1, g2 func(t *testing.T) *System
		s      complex128
	}{
		{"discrete_in3out2_vs_in2out3",
			func(t *testing.T) *System { return siso(t, 0.5, 0.1, []float64{3}, []float64{2}, nil) },
			func(t *testing.T) *System { return siso(t, -0.3, 0.1, []float64{2}, []float64{3}, nil) },
			0.3 + 1.1i},
		{"continuous_in2out05_vs_in05out2",
			func(t *testing.T) *System { return siso(t, -1, 0, []float64{2}, []float64{0.5}, nil) },
			func(t *testing.T) *System { return siso(t, -2, 0, []float64{0.5}, []float64{2}, nil) },
			0.1 + 0.7i},
		{"continuous_in2_vs_out2",
			func(t *testing.T) *System { return siso(t, -1, 0, []float64{2}, nil, nil) },
			func(t *testing.T) *System { return siso(t, -2, 0, nil, []float64{2}, nil) },
			0.1 + 0.7i},
		{"continuous_in2out05_vs_in1out15",
			func(t *testing.T) *System { return siso(t, -1, 0, []float64{2}, []float64{0.5}, nil) },
			func(t *testing.T) *System { return siso(t, -2, 0, []float64{1}, []float64{1.5}, nil) },
			0.1 + 0.7i},
		{"continuous_iodelay_vs_in",
			func(t *testing.T) *System { return siso(t, -1, 0, nil, nil, mat.NewDense(1, 1, []float64{1.2})) },
			func(t *testing.T) *System { return siso(t, -2, 0, []float64{1.2}, nil, nil) },
			0.2 - 0.9i},
		{"continuous_mimo_mixed",
			func(t *testing.T) *System {
				return mimo(t, []float64{-1, 0.4, -0.7, -2}, []float64{1, 2}, []float64{0.5, 0}, nil)
			},
			func(t *testing.T) *System {
				return mimo(t, []float64{-3, 1, 0.2, -0.5}, []float64{0.5, 1}, []float64{1, 0},
					mat.NewDense(2, 2, []float64{0, 0.5, 0.5, 1}))
			},
			0.3 + 0.8i},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			g1, g2 := tc.g1(t), tc.g2(t)
			r, err := Parallel(g1, g2)
			if err != nil {
				t.Fatal(err)
			}
			if r.HasInternalDelay() {
				t.Errorf("equal total delays should not need internal delays")
			}
			got, err := r.EvalFr(tc.s)
			if err != nil {
				t.Fatal(err)
			}
			h1, err := g1.EvalFr(tc.s)
			if err != nil {
				t.Fatal(err)
			}
			h2, err := g2.EvalFr(tc.s)
			if err != nil {
				t.Fatal(err)
			}
			for i := range got {
				for j := range got[i] {
					want := h1[i][j] + h2[i][j]
					if cmplx.Abs(got[i][j]-want) > 1e-12*(1+cmplx.Abs(want)) {
						t.Errorf("H[%d][%d] = %v, want %v", i, j, got[i][j], want)
					}
				}
			}
		})
	}
}

func TestParallelEqualTotalDelayClosedForm(t *testing.T) {
	g1, _ := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1}, []float64{1}, []float64{0}, 0)
	g1.InputDelay = []float64{2}
	g1.OutputDelay = []float64{0.5}
	g2, _ := NewFromSlices(1, 1, 1, []float64{-2}, []float64{1}, []float64{1}, []float64{0}, 0)
	g2.InputDelay = []float64{0.5}
	g2.OutputDelay = []float64{2}
	r, err := Parallel(g1, g2)
	if err != nil {
		t.Fatal(err)
	}
	s := 0.1 + 0.7i
	got, err := r.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	want := cmplx.Exp(-2.5*s) * (1/(s+1) + 1/(s+2))
	if cmplx.Abs(got[0][0]-want) > 1e-12 {
		t.Fatalf("H = %v, want %v", got[0][0], want)
	}
}
