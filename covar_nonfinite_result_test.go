package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestCovarNonFiniteResultReturnsErrOverflow(t *testing.T) {
	triA := []float64{-1, 2, 0, -3}
	oscA := []float64{-.1, 5, -5, -.1}
	ones := []float64{1, 1, 1, 1}
	I := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	inputDelay := func(tau float64) func(*System) error {
		return func(s *System) error { return s.SetInputDelay([]float64{0, tau}) }
	}
	for _, tc := range []struct {
		name  string
		build func() (*System, error)
		delay func(*System) error
		w     *mat.Dense
	}{
		{
			name: "delay overflows exponent",
			build: func() (*System, error) {
				return New(mat.NewDense(2, 2, triA), mat.NewDense(2, 2, ones), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 2, nil), 0)
			},
			delay: inputDelay(1e308),
			w:     I,
		},
		{
			name: "delay overflows exponent beside feedthrough",
			build: func() (*System, error) {
				return New(mat.NewDense(2, 2, triA), mat.NewDense(2, 2, ones), mat.NewDense(2, 2, []float64{1, 1, 1, -1}), mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
			},
			delay: inputDelay(1e308),
			w:     I,
		},
		{
			name: "continuous solution overflows",
			build: func() (*System, error) {
				return New(mat.NewDense(2, 2, oscA), mat.NewDense(2, 2, []float64{1, 0, 0, 1}), mat.NewDense(1, 2, []float64{1, 2}), mat.NewDense(1, 2, nil), 0)
			},
			w: mat.NewDense(2, 2, []float64{1e308, 0, 0, 1e308}),
		},
		{
			name: "continuous output scale overflows",
			build: func() (*System, error) {
				return New(mat.NewDense(2, 2, triA), mat.NewDense(2, 2, ones), mat.NewDense(1, 2, []float64{1e155, 1e155}), mat.NewDense(1, 2, nil), 0)
			},
			w: I,
		},
		{
			name: "discrete feedthrough overflows",
			build: func() (*System, error) {
				return New(mat.NewDense(2, 2, []float64{.5, .2, 0, -.3}), mat.NewDense(2, 2, ones), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 2, []float64{1e155, -1e155}), 1)
			},
			w: I,
		},
		{
			name: "discrete gain overflows behind delay",
			build: func() (*System, error) {
				return NewGain(mat.NewDense(2, 2, []float64{1e155, 1e155, 1e155, -1e155}), 1)
			},
			delay: inputDelay(1),
			w:     I,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := tc.build()
			if err != nil {
				t.Fatal(err)
			}
			if tc.delay != nil {
				if err := tc.delay(sys); err != nil {
					t.Fatal(err)
				}
			}
			P, err := Covar(sys, tc.w)
			if !errors.Is(err, ErrOverflow) {
				t.Fatalf("P=%v err=%v, want ErrOverflow", P, err)
			}
			if P != nil {
				t.Errorf("P=%v with error, want nil", P)
			}
		})
	}
}

func TestCovarFiniteAndFeedthroughValuesSurviveOverflowGuard(t *testing.T) {
	I := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	build := func(t *testing.T, d []float64, tau float64) *System {
		t.Helper()
		sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 2, []float64{1, 1, 1, 1}), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 2, d), 0)
		if err != nil {
			t.Fatal(err)
		}
		if err := sys.SetInputDelay([]float64{0, tau}); err != nil {
			t.Fatal(err)
		}
		return sys
	}
	t.Run("largest delay below exponent overflow", func(t *testing.T) {
		P, err := Covar(build(t, []float64{0, 0}, 1e307), I)
		if err != nil {
			t.Fatal(err)
		}
		if got := P.At(0, 0); math.Abs(got-4) > 1e-12 {
			t.Errorf("P=%g, want 4", got)
		}
	})
	t.Run("feedthrough entry stays infinite over overflowed delay", func(t *testing.T) {
		P, err := Covar(build(t, []float64{1, 0}, 1e308), I)
		if err != nil {
			t.Fatal(err)
		}
		if got := P.At(0, 0); !math.IsInf(got, 1) {
			t.Errorf("P=%g, want +Inf", got)
		}
	})
	t.Run("feedthrough entry stays infinite over overflowed output scale", func(t *testing.T) {
		sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 2, []float64{1, 1, 1, 1}), mat.NewDense(1, 2, []float64{1e155, 1e155}), mat.NewDense(1, 2, []float64{1, 0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		P, err := Covar(sys, I)
		if err != nil {
			t.Fatal(err)
		}
		if got := P.At(0, 0); !math.IsInf(got, 1) {
			t.Errorf("P=%g, want +Inf", got)
		}
	})
	t.Run("feedthrough entry stays infinite", func(t *testing.T) {
		P, err := Covar(build(t, []float64{1, 0}, 1e307), I)
		if err != nil {
			t.Fatal(err)
		}
		if got := P.At(0, 0); !math.IsInf(got, 1) {
			t.Errorf("P=%g, want +Inf", got)
		}
	})
}
