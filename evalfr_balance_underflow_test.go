package controlsys

import (
	"errors"
	"fmt"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// hugeDiagonalModel is the upper bidiagonal A with diagonal -k, -2k, …
// and superdiagonal k, B = bScale·1 with B₀ = 0 where coupled, and
// C = 1ᵀ/bScale.
func hugeDiagonalModel(t *testing.T, n int, k, bScale float64, coupled bool) *System {
	t.Helper()
	a := make([]float64, n*n)
	b := make([]float64, n)
	c := make([]float64, n)
	for i := range n {
		a[i*n+i] = -float64(i+1) * k
		if i+1 < n {
			a[i*n+i+1] = k
		}
		b[i], c[i] = bScale, 1/bScale
	}
	if coupled {
		b[0] = 0
	}
	sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, 1, b), mat.NewDense(1, n, c), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// ECC62L: with the diagonal skipped, balancing a 1e300 model is finite but
// leaves D⁻¹x below the normal range (x̂₀ ≈ 1.7e-400 at 1i), so the
// response takes the verified double-double solve of the unbalanced
// realization. Coupled has B̂₀ = 0: x̂₀ is driven only through â₀₁ and
// underflows the same way. Mid scales do not trigger it and stay on the
// double path.
func TestEvalFr_HugeDiagonalBalanceNoOverflow(t *testing.T) {
	for _, coupled := range []bool{false, true} {
		for _, n := range []int{2, 3} {
			for _, k := range []float64{1e3, 1e100, 1e300} {
				t.Run(fmt.Sprintf("coupled=%v/n=%d/k=%g", coupled, n, k), func(t *testing.T) {
					sys := hugeDiagonalModel(t, n, k, 1, coupled)
					omega := []float64{1, k}
					fr, err := sys.FreqResponse(omega)
					if err != nil {
						t.Fatalf("FreqResponse: %v", err)
					}
					for i, w := range omega {
						s := complex(0, w)
						want := bigSISOResponse(sys, s)
						got := map[string]complex128{"FreqResponse": fr.At(i, 0, 0)}
						if g, err := sys.EvalFr(s); err != nil {
							t.Errorf("EvalFr(%gi): %v", w, err)
						} else {
							got["EvalFr"] = g[0][0]
						}
						dst := make([]complex128, 1)
						if err := newBalancedDense(sys, n, 1, 1).evalInto(pointAt(s), dst); err != nil {
							t.Errorf("dense(%gi): %v", w, err)
						} else {
							got["dense"] = dst[0]
						}
						if err := newHessenbergSweep(sys, n, 1, 1).evalInto(pointAt(s), dst); err != nil {
							t.Errorf("hessenberg(%gi): %v", w, err)
						} else {
							got["hessenberg"] = dst[0]
						}
						for name, g := range got {
							if rel := cmplx.Abs(g-want) / cmplx.Abs(want); !(rel <= 1e-12) {
								t.Errorf("%s(%gi) = %v, want %v (rel %.3g)", name, w, g, want, rel)
							}
						}
					}
				})
			}
		}
	}
}

// ECC62L: where the unbalanced states underflow too, the fallback cannot
// verify and the point fails instead of returning the lost-state value.
func TestEvalFr_StateUnderflowUnverifiedFails(t *testing.T) {
	sys := hugeDiagonalModel(t, 2, 1e300, 1e-30, false)
	want := bigSISOResponse(sys, 1i)
	g, err := sys.EvalFr(1i)
	if !errors.Is(err, ErrOverflow) {
		t.Errorf("EvalFr(1i) = %v, %v; want ErrOverflow (oracle %v not resolvable: B/σ ≈ 3.7e-331)", g, err, want)
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
