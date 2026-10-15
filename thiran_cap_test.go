package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestThiranDelayRejectsHugeDelay(t *testing.T) {
	const dt = 0.1
	for _, tc := range []struct {
		name    string
		tau, dt float64
		want    error
	}{
		{"integer 1e9", 1e9 * dt, dt, ErrInvalidArgument},
		{"integer 1e19", 1e19 * dt, dt, ErrInvalidArgument},
		{"fractional 1e9", (1e9 + 0.5) * dt, dt, ErrInvalidArgument},
		{"cap+1 integer", (MaxAbsorbedDelayStates + 1) * 1.0, 1, ErrInvalidArgument},
		{"cap fractional order cap+1", MaxAbsorbedDelayStates + 0.5, 1, ErrInvalidArgument},
		{"samples overflow to Inf", 1, 1e-320, ErrInvalidArgument},
		{"NaN tau", math.NaN(), dt, ErrNegativeDelay},
		{"Inf tau", math.Inf(1), dt, ErrNegativeDelay},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := ThiranDelay(tc.tau, tc.dt)
			if !errors.Is(err, tc.want) || sys != nil {
				t.Fatalf("ThiranDelay(%g, %g) = %v, %v; want nil, %v", tc.tau, tc.dt, sys, err, tc.want)
			}
		})
	}
}

func TestThiranDelayIntegerBelowCapIsExactShift(t *testing.T) {
	const d = 7
	sys, err := ThiranDelay(d*0.5, 0.5)
	if err != nil {
		t.Fatal(err)
	}
	if n, _, _ := sys.Dims(); n != d {
		t.Fatalf("states = %d, want %d", n, d)
	}
	tfRes, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0.1, 0.7, 2.3} {
		z := cmplx.Exp(complex(0, w))
		got := mustEval(t, tfRes.TF, z)[0][0]
		if want := cmplx.Pow(z, -d); cmplx.Abs(got-want) > 1e-12 {
			t.Errorf("w=%g: H = %v, want z^-%d = %v", w, got, d, want)
		}
	}
}

func TestFixedOrderThiranRejectsHugeDelay(t *testing.T) {
	for _, D := range []float64{1e9, 1e9 + 0.4, 1e19, MaxAbsorbedDelayStates + 1.4} {
		sys, err := thiranDelay(D, 3, 1)
		if !errors.Is(err, ErrInvalidArgument) || sys != nil {
			t.Errorf("thiranDelay(%g, 3, 1) = %v, %v; want nil, ErrInvalidArgument", D, sys, err)
		}
	}
}

// MATLAB c2d with ThiranOrder keeps whole samples of a delay as a delay and
// models only the fractional part with an order-N Thiran filter, so huge
// delays cost no extra states.
func TestC2DThiranHugeDelayKeepsWholeSamplesAsDelay(t *testing.T) {
	const dt, order = 0.1, 3
	for _, tc := range []struct {
		samples, whole float64
		extra          int
	}{
		{1e9 + 0.5, 1e9 + 1 - order, order},
		{1e19, 1e19, 0},
	} {
		for _, side := range []string{"input", "output"} {
			for _, modeling := range []C2DDelayModeling{C2DDelayModelingInternal, C2DDelayModelingState} {
				sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
					mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
				if err != nil {
					t.Fatal(err)
				}
				if side == "input" {
					err = sys.SetInputDelay([]float64{tc.samples * dt})
				} else {
					err = sys.SetOutputDelay([]float64{tc.samples * dt})
				}
				if err != nil {
					t.Fatal(err)
				}
				disc, err := sys.C2D(dt, C2DOptions{Method: C2DMethodTustin, ThiranOrder: order, DelayModeling: modeling})
				if err != nil {
					t.Fatalf("%s %g samples %q: %v", side, tc.samples, modeling, err)
				}
				n, _, _ := disc.Dims()
				if got := n + disc.internalDelayCount(); got != 1+tc.extra {
					t.Errorf("%s %g samples %q: states+internal = %d, want %d", side, tc.samples, modeling, got, 1+tc.extra)
				}
				kept := disc.InputDelay
				if side == "output" {
					kept = disc.OutputDelay
				}
				if len(kept) != 1 || math.Abs(kept[0]-tc.whole) > 1e-6*tc.whole {
					t.Errorf("%s %g samples %q: kept delay %v, want [%g]", side, tc.samples, modeling, kept, tc.whole)
				}
			}
		}
	}
}
