package controlsys

import (
	"errors"
	"math"
	"strings"
	"testing"
	"time"

	"plantcontrol.org/v1/gonum/mat"
)

// finishesWithin fails the test instead of hanging when f does not return.
func finishesWithin(t *testing.T, d time.Duration, f func()) {
	t.Helper()
	done := make(chan struct{})
	go func() {
		defer close(done)
		f()
	}()
	select {
	case <-done:
	case <-time.After(d):
		t.Fatalf("did not return within %v", d)
	}
}

func TestRequireSystem(t *testing.T) {
	valid, err := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, []float64{0.5}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := requireSystem("Op", valid); err != nil {
		t.Fatalf("valid system: %v", err)
	}

	err = requireSystem("Op", nil)
	if !errors.Is(err, ErrInvalidArgument) || err.Error() != "Op: system is nil: controlsys: invalid argument" {
		t.Fatalf("nil system: %v", err)
	}

	bad := valid.Copy()
	bad.B = mat.NewDense(3, 1, nil)
	err = requireSystem("Op", bad)
	if !errors.Is(err, ErrDimensionMismatch) || !strings.HasPrefix(err.Error(), "Op: ") {
		t.Fatalf("invalid system: %v", err)
	}

	withNaN := valid.Copy()
	withNaN.A.Set(0, 1, math.NaN())
	if err := requireSystem("Op", withNaN); err != nil {
		t.Fatalf("requireSystem must not check finiteness: %v", err)
	}
}

func TestRequireFiniteSystem(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0.5})
	E := mat.NewDense(2, 2, []float64{1, 0.5, 0, 2})
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := requireFiniteSystem("Op", sys); err != nil {
		t.Fatalf("finite system: %v", err)
	}
	if err := requireFiniteSystem("Op", nil); !errors.Is(err, ErrInvalidArgument) {
		t.Fatalf("nil system: %v", err)
	}

	cases := []struct {
		name string
		set  func(s *System)
		want string
	}{
		{"A", func(s *System) { s.A.Set(1, 0, math.NaN()) }, "Op: A[1,0] is NaN: controlsys: invalid argument"},
		{"B", func(s *System) { s.B.Set(1, 0, math.Inf(1)) }, "Op: B[1,0] is +Inf: controlsys: invalid argument"},
		{"C", func(s *System) { s.C.Set(0, 1, math.Inf(-1)) }, "Op: C[0,1] is -Inf: controlsys: invalid argument"},
		{"D", func(s *System) { s.D.Set(0, 0, math.NaN()) }, "Op: D[0,0] is NaN: controlsys: invalid argument"},
		{"E", func(s *System) { s.E.Set(0, 1, math.NaN()) }, "Op: E[0,1] is NaN: controlsys: invalid argument"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := sys.Copy()
			tc.set(s)
			err := requireFiniteSystem("Op", s)
			if !errors.Is(err, ErrInvalidArgument) || err.Error() != tc.want {
				t.Fatalf("got %v, want %q", err, tc.want)
			}
		})
	}
}

func TestRequireFiniteSystemInternalDelay(t *testing.T) {
	sys, err := New(mat.NewDense(1, 1, []float64{-1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInternalDelay([]float64{0.5},
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), mat.NewDense(1, 1, []float64{0}),
		mat.NewDense(1, 1, []float64{0})); err != nil {
		t.Fatal(err)
	}
	if err := requireFiniteSystem("Op", sys); err != nil {
		t.Fatalf("finite system: %v", err)
	}
	sys.LFT.D22.Set(0, 0, math.NaN())
	if err := requireFiniteSystem("Op", sys); !errors.Is(err, ErrInvalidArgument) || !strings.Contains(err.Error(), "D22[0,0]") {
		t.Fatalf("NaN D22: %v", err)
	}
}

func TestRequireFiniteDense(t *testing.T) {
	if err := requireFiniteDense("Op", "Q", nil); !errors.Is(err, ErrInvalidArgument) || err.Error() != "Op: Q is nil: controlsys: invalid argument" {
		t.Fatalf("nil: %v", err)
	}
	if err := requireFiniteDense("Op", "Q", &mat.Dense{}); err != nil {
		t.Fatalf("empty: %v", err)
	}

	m := mat.NewDense(3, 3, []float64{1, 2, math.NaN(), 4, 5, 6, 7, 8, 9})
	view := m.Slice(0, 2, 0, 2).(*mat.Dense)
	if err := requireFiniteDense("Op", "Q", view); err != nil {
		t.Fatalf("view excluding NaN column must pass (stride-aware scan): %v", err)
	}
	if err := requireFiniteDense("Op", "Q", m); !errors.Is(err, ErrInvalidArgument) || err.Error() != "Op: Q[0,2] is NaN: controlsys: invalid argument" {
		t.Fatalf("full: %v", err)
	}
	m.Set(1, 1, math.Inf(1))
	if err := requireFiniteDense("Op", "Q", m.Slice(1, 3, 1, 3).(*mat.Dense)); err == nil || err.Error() != "Op: Q[0,0] is +Inf: controlsys: invalid argument" {
		t.Fatalf("view with Inf: %v", err)
	}
}

func TestRequireFinite(t *testing.T) {
	if err := requireFinite("Op", "x"); err != nil {
		t.Fatalf("no values: %v", err)
	}
	if err := requireFinite("Op", "x", 1, -2, 0); err != nil {
		t.Fatalf("finite: %v", err)
	}
	if err := requireFinite("Op", "tFinal", math.NaN()); !errors.Is(err, ErrInvalidArgument) || err.Error() != "Op: tFinal is NaN: controlsys: invalid argument" {
		t.Fatalf("scalar: %v", err)
	}
	if err := requireFinite("Op", "omega", 1, 2, math.Inf(1)); !errors.Is(err, ErrInvalidArgument) || err.Error() != "Op: omega[2] is +Inf: controlsys: invalid argument" {
		t.Fatalf("slice: %v", err)
	}
}
