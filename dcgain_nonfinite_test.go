package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

func shiftModel(t *testing.T, n int, dt float64, set func(a, b, c, d []float64)) *System {
	t.Helper()
	a := make([]float64, n*n)
	for i := 1; i < n; i++ {
		a[i*n+i-1] = 1
	}
	b := make([]float64, n)
	c := make([]float64, n)
	if n > 0 {
		b[0], c[n-1] = 1, 1
	}
	d := []float64{1}
	set(a, b, c, d)
	return poleSys(t, n, 1, 1, a, b, c, d, dt)
}

func TestDCGainNonFiniteModelErrors(t *testing.T) {
	nan, inf := math.NaN(), math.Inf(1)
	cases := []struct {
		name string
		n    int
		dt   float64
		set  func(a, b, c, d []float64)
	}{
		{"A NaN n10", 10, 0, func(a, _, _, _ []float64) { a[99] = nan }},
		{"B NaN n3", 3, 0, func(a, b, _, _ []float64) { a[8] = -1; b[0] = nan }},
		{"C Inf n3", 3, 0, func(a, _, c, _ []float64) { a[8] = -1; c[2] = inf }},
		{"D NaN n3", 3, 0, func(a, _, _, d []float64) { a[8] = -1; d[0] = nan }},
		{"A Inf discrete", 3, 0.1, func(a, _, _, _ []float64) { a[8] = -inf }},
		{"static D NaN", 0, 0, func(_, _, _, d []float64) { d[0] = nan }},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := shiftModel(t, tc.n, tc.dt, tc.set)
			for _, f := range []func() error{
				func() error { _, err := sys.DCGain(); return err },
				func() error { _, err := sys.integratingDCGain(); return err },
			} {
				if err := f(); !errors.Is(err, ErrInvalidArgument) {
					t.Fatalf("err = %v, want ErrInvalidArgument", err)
				}
			}
		})
	}
}

func TestEvalWithPoleLimitNonFiniteRealization(t *testing.T) {
	for _, set := range []func(a, b, c, d []float64){
		func(_, b, _, _ []float64) { b[3] = math.NaN() },
		func(_, _, c, _ []float64) { c[2] = math.Inf(-1) },
		func(_, _, _, d []float64) { d[0] = math.NaN() },
	} {
		sys := shiftModel(t, 10, 0, set)
		dst := make([]complex128, 1)
		err := evalWithPoleLimit(newBalancedDense(sys, 10, 1, 1).evalInto, sys, pointAt(1i), dst)
		if !errors.Is(err, ErrInvalidArgument) {
			t.Fatalf("err = %v dst = %v, want ErrInvalidArgument", err, dst)
		}
	}
	sys := shiftModel(t, 10, 0, func(a, _, _, _ []float64) { a[99] = -1 })
	dst := make([]complex128, 1)
	if err := evalWithPoleLimit(newBalancedDense(sys, 10, 1, 1).evalInto, sys, pointAt(1i), dst); err != nil || cmplx.IsNaN(dst[0]) {
		t.Fatalf("finite model: err = %v dst = %v", err, dst)
	}
}
