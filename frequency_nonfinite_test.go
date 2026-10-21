package controlsys

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func nonFiniteShiftModel(t *testing.T, n int, bad float64) *System {
	t.Helper()
	a := make([]float64, n*n)
	for i := 1; i < n; i++ {
		a[i*n+i-1] = 1
	}
	a[n*n-1] = bad
	b := make([]float64, n)
	b[0] = 1
	c := make([]float64, n)
	c[n-1] = 1
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{1}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestFrequencyNonFiniteModelErrors(t *testing.T) {
	for _, n := range []int{2, 3, 5, 10, 40} {
		for _, bad := range []float64{math.NaN(), math.Inf(1)} {
			sys := nonFiniteShiftModel(t, n, bad)
			label := fmt.Sprintf("n=%d A[n-1,n-1]=%g", n, bad)
			for _, s := range []complex128{complex(0, 1e-36), complex(0, 1)} {
				if h, err := sys.EvalFr(s); !errors.Is(err, ErrInvalidArgument) {
					t.Errorf("%s: EvalFr(%v) = %v, %v; want ErrInvalidArgument", label, s, h, err)
				}
			}
			omega := []float64{1e-36, 1, 2, 3, 4, 5, 6, 7}
			if r, err := sys.FreqResponse(omega); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("%s: FreqResponse = %v, %v; want ErrInvalidArgument", label, r, err)
			}
			if r, err := sys.FreqResponsePointwise(omega); !errors.Is(err, ErrInvalidArgument) {
				t.Errorf("%s: FreqResponsePointwise = %v, %v; want ErrInvalidArgument", label, r, err)
			}
		}
	}

	ok, err := NewFromSlices(2, 1, 1, []float64{-1, 2, 0, -3}, []float64{1, 1}, []float64{1, 0}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, name := range []string{"B", "C", "D"} {
		bad := ok.Copy()
		switch name {
		case "B":
			bad.B.Set(1, 0, math.NaN())
		case "C":
			bad.C.Set(0, 1, math.NaN())
		case "D":
			bad.D.Set(0, 0, math.NaN())
		}
		if h, err := bad.EvalFr(complex(0, 1)); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("NaN in %s: EvalFr = %v, %v; want ErrInvalidArgument", name, h, err)
		}
	}
	e := mat.NewDense(2, 2, []float64{1, 0, 0, math.NaN()})
	if desc, err := NewDescriptor(ok.A, ok.B, ok.C, ok.D, e, 0); err == nil {
		if h, err := desc.EvalFr(complex(0, 1)); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("NaN in E: EvalFr = %v, %v; want ErrInvalidArgument", h, err)
		}
	}
	if h, err := ok.EvalFr(complex(math.NaN(), 0)); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("EvalFr(NaN) = %v, %v; want ErrInvalidArgument", h, err)
	}
}

func TestBalancedDenseNonFinitePencilNoPanic(t *testing.T) {
	sys := nonFiniteShiftModel(t, 10, math.NaN())
	bd := newBalancedDense(sys, 10, 1, 1)
	dst := make([]complex128, 1)
	pt := pointAt(complex(0, 1e-36))
	if err := bd.evalInto(pt, dst); !errors.Is(err, errNonFinitePencil) {
		t.Fatalf("evalInto = %v; want errNonFinitePencil", err)
	}
	if err := evalWithPoleLimit(bd.evalInto, sys, pt, dst); err == nil {
		t.Fatalf("evalWithPoleLimit = %v, nil error", dst)
	}
}

func TestEvalFrInfinityKeepsFeedthrough(t *testing.T) {
	sys, err := NewFromSlices(2, 1, 1, []float64{-1, 2, 0, -3}, []float64{1, 1}, []float64{1, 0}, []float64{2.5}, 0)
	if err != nil {
		t.Fatal(err)
	}
	h, err := sys.EvalFr(complex(math.Inf(1), 0))
	if err != nil || h[0][0] != 2.5 {
		t.Fatalf("EvalFr(Inf) = %v, %v; want D = 2.5", h, err)
	}
}
