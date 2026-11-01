package controlsys

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func wantC2DOverflow(t *testing.T, name string, d *System, err error) {
	t.Helper()
	if !errors.Is(err, ErrOverflow) || d != nil {
		t.Errorf("%s = %v, %v; want nil, ErrOverflow", name, d, err)
	}
}

func explicitOverflowingHold(t *testing.T) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(2, 2, []float64{1e4, 1, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// e^{1e4·0.1} overflows on every arch, with or without delays, so this is the
// same symptom as the rounding-dependent stiff descriptor below (4MJ46B).
func TestC2DExplicitExponentialOverflowReturnsError(t *testing.T) {
	const dt = 0.1
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodImpulse} {
		d, err := explicitOverflowingHold(t).C2D(dt, C2DOptions{Method: m})
		wantC2DOverflow(t, "C2D "+string(m), d, err)

		d, g, err := explicitOverflowingHold(t).C2DMap(dt, C2DOptions{Method: m})
		wantC2DOverflow(t, "C2DMap "+string(m), d, err)
		if g != nil {
			t.Errorf("C2DMap %s: map %v with error", m, g)
		}
	}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		sys := explicitOverflowingHold(t)
		err := sys.SetInternalDelay([]float64{0.25},
			mat.NewDense(2, 1, []float64{0.3, -0.2}),
			mat.NewDense(1, 2, []float64{0.4, -0.3}),
			mat.NewDense(1, 1, []float64{0}),
			mat.NewDense(1, 1, []float64{0}),
			mat.NewDense(1, 1, []float64{0}))
		if err != nil {
			t.Fatal(err)
		}
		d, err := sys.C2D(dt, C2DOptions{Method: m})
		wantC2DOverflow(t, "C2D internal delay "+string(m), d, err)
	}
}

// Resampling to a longer period reaches the same conversion through d2c:
// e^{ln(1e5)/0.01} overflows.
func TestD2DExponentialOverflowReturnsError(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{1e5, 1, 0, 0.5}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0.01)
	if err != nil {
		t.Fatal(err)
	}
	d, err := sys.D2D(1, D2DOptions{})
	wantC2DOverflow(t, "D2D", d, err)
}

// F=[-1,c,1;0,-2,1;0,0,-1], W(1,1,1e-13), stream 777 at c=3e10 is the
// undetermined pencil of 7AAOBX, whose split fails on rounding; which seeds
// overflow depends on the arch, so the seeds are swept and only the contract
// is asserted: value without Inf/NaN in any matrix, or ErrOverflow (4MJ46B).
// A fractional input delay reaches FOH overflow in the LFT matrices alone.
func TestC2DStiffDescriptorValueIsFiniteOrError(t *testing.T) {
	const c, dt = 3e10, 0.1
	for _, tc := range []struct {
		name  string
		delay []float64
		m     C2DMethod
	}{
		{"zoh", nil, C2DMethodZOH},
		{"foh", nil, C2DMethodFOH},
		{"impulse", nil, C2DMethodImpulse},
		{"foh input delay", []float64{0.25}, C2DMethodFOH},
	} {
		t.Run(tc.name, func(t *testing.T) {
			overflowed := 0
			for seed := range uint64(60) {
				sys := stiffRotatedDescriptor(t, rand.New(rand.NewPCG(seed, 777)),
					[]float64{-1, c, 1, 0, -2, 1, 0, 0, -1},
					[]float64{1, 1, 1e-13}, []float64{1, 1.3, 1.6}, []float64{1, .8, .6})
				sys.InputDelay = tc.delay
				d, err := sys.C2D(dt, C2DOptions{Method: tc.m})
				switch {
				case errors.Is(err, ErrOverflow):
					overflowed++
					if d != nil {
						t.Errorf("seed %d: value with error %v", seed, err)
					}
				case err != nil:
					t.Errorf("seed %d: %v", seed, err)
				default:
					if nf := requireFiniteSystem("C2D", d); nf != nil {
						t.Errorf("seed %d: nil error with %v", seed, nf)
					}
				}
			}
			t.Logf("%d of 60 seeds overflow", overflowed)
			if overflowed == 0 {
				t.Error("no seed overflows; the sweep no longer reaches the case")
			}
		})
	}
}

// C2DMap FOH of the same pencil at c=1e10 can have a finite result and NaN in
// the Γ1 block of G, on seeds that depend on the arch (4MJ46B).
func TestC2DMapStiffDescriptorFOHMapIsFiniteOrError(t *testing.T) {
	const c, dt = 1e10, 0.1
	for seed := range uint64(100) {
		sys := stiffRotatedDescriptor(t, rand.New(rand.NewPCG(seed, 777)),
			[]float64{-1, c, 1, 0, -2, 1, 0, 0, -1},
			[]float64{1, 1, 1e-13}, []float64{1, 1.3, 1.6}, []float64{1, .8, .6})
		d, g, err := sys.C2DMap(dt, C2DOptions{Method: C2DMethodFOH})
		if err != nil {
			if !errors.Is(err, ErrOverflow) || d != nil || g != nil {
				t.Errorf("seed %d: %v, %v, %v; want nil, nil, ErrOverflow", seed, d, g, err)
			}
			continue
		}
		if nf := requireFiniteDense("C2DMap", "G", g); nf != nil {
			t.Errorf("seed %d: nil error with %v", seed, nf)
		}
	}
}

// Lsim keeps the non-finite samples of a diverging plant as its response, as
// Step does on the stiff unstable models of TestPolesFullRankEKeepsStiffModeOfExplicitModel,
// on the plain path and through the rational part of held delay feedback (4MJ46B).
func TestLsimDivergenceStaysResponseWhereC2DOverflows(t *testing.T) {
	delayed := explicitOverflowingHold(t)
	err := delayed.SetInternalDelay([]float64{0.25},
		mat.NewDense(2, 1, []float64{0.3, -0.2}),
		mat.NewDense(1, 2, []float64{0.4, -0.3}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0}))
	if err != nil {
		t.Fatal(err)
	}
	delayed.InputDelay = []float64{0.05}
	const steps = 5
	grid := make([]float64, steps)
	u := mat.NewDense(steps, 1, nil)
	for i := range steps {
		grid[i] = 0.1 * float64(i)
		u.Set(i, 0, 1)
	}
	for name, sys := range map[string]*System{"explicit": explicitOverflowingHold(t), "held delay feedback": delayed} {
		r, err := Lsim(sys, u, grid, nil)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		rows, cols := r.Y.Dims()
		diverged := false
		for i := range rows {
			for j := range cols {
				v := r.Y.At(i, j)
				diverged = diverged || math.IsNaN(v) || math.IsInf(v, 0)
			}
		}
		if !diverged {
			t.Errorf("%s: finite response %v of a diverging plant", name, mat.Formatted(r.Y))
		}
	}
}
