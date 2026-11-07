package controlsys

import (
	"fmt"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// ECC62L: skipping the diagonal in balancing leaves mid-scale responses
// unchanged. The 1e300 analogue still underflows the balanced state and is
// not asserted here.
func TestEvalFr_BalanceDiagonalMidScale(t *testing.T) {
	for _, k := range []float64{1e3, 1e100} {
		t.Run(fmt.Sprint(k), func(t *testing.T) {
			sys, err := New(
				mat.NewDense(2, 2, []float64{-k, k, 0, -2 * k}),
				mat.NewDense(2, 1, []float64{1, 1}),
				mat.NewDense(1, 2, []float64{1, 1}),
				mat.NewDense(1, 1, []float64{0}), 0)
			if err != nil {
				t.Fatal(err)
			}
			omega := []float64{1, k}
			fr, err := sys.FreqResponse(omega)
			if err != nil {
				t.Fatalf("FreqResponse: %v", err)
			}
			for i, w := range omega {
				s := complex(0, w)
				want := bigSISOResponse(sys, s)
				g, err := sys.EvalFr(s)
				if err != nil {
					t.Errorf("EvalFr(%gi): %v", w, err)
					continue
				}
				for name, got := range map[string]complex128{"EvalFr": g[0][0], "FreqResponse": fr.At(i, 0, 0)} {
					if rel := cmplx.Abs(got-want) / cmplx.Abs(want); !(rel <= 1e-12) {
						t.Errorf("%s(%gi) = %v, want %v (rel %.3g)", name, w, got, want, rel)
					}
				}
			}
		})
	}
}

// ECC62L: balancing scaled the invariant diagonal by 1/f then f, so a
// diagonal above ~5.5e157 overflowed at f = 2⁻⁵⁰⁰. The diagonal of D⁻¹AD and D⁻¹ED is invariant and must stay
// bit-identical through balancing.
func TestBalancedRealization_DiagonalInvariant(t *testing.T) {
	const k = 1e300
	tests := []struct {
		name string
		n    int
		a, e []float64
		b, c []float64
	}{
		{"explicit huge", 2, []float64{-k, k, 0, -2 * k}, nil, []float64{1, 1}, []float64{1, 1}},
		{"descriptor huge", 2, []float64{-1, 1, 0, -2}, []float64{k, k, 0, 2 * k}, []float64{1, 1}, []float64{1, 1}},
		{"graded mid", 3, []float64{-1e3, 1e3, 1e-2, 3, -2e3, 1e6, 1e-6, 7, -3e3}, []float64{1, 1e-8, 0, 2, 3, 0, 1e9, 0, 5}, []float64{1, 1e-9, 1e9}, []float64{1e9, 1, 1e-9}},
	}
	for _, tc := range tests {
		br := balancedRealization{n: tc.n, m: 1, p: 1, a: append([]float64(nil), tc.a...), b: append([]float64(nil), tc.b...), c: append([]float64(nil), tc.c...)}
		if tc.e != nil {
			br.e = append([]float64(nil), tc.e...)
		}
		br.balance(nil)
		if !br.finite() {
			t.Errorf("%s: non-finite balanced realization a=%v e=%v", tc.name, br.a, br.e)
		}
		for i := range tc.n {
			d := i*tc.n + i
			if br.a[d] != tc.a[d] || tc.e != nil && br.e[d] != tc.e[d] {
				t.Errorf("%s: diagonal %d changed: a=%v e=%v", tc.name, i, br.a, br.e)
			}
		}
	}
}
