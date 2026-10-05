package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestSchur2x2EigenvaluesBadlyScaled(t *testing.T) {
	tests := []struct {
		name       string
		a, b, c, d float64
		want       [2]complex128
	}{
		{"standardized large real tiny imag", 1e8, 1, -1, 1e8, [2]complex128{complex(1e8, 1), complex(1e8, -1)}},
		{"standardized partial cancellation", -1e8, 3.7, -3, -1e8, [2]complex128{complex(-1e8, math.Sqrt(11.1)), complex(-1e8, -math.Sqrt(11.1))}},
		{"standardized near repeated", 0.5, 1e-9, -1e-9, 0.5, [2]complex128{complex(0.5, 1e-9), complex(0.5, -1e-9)}},
		{"nonstandard complex", 1e8 + 1, 3, -2, 1e8 - 1, [2]complex128{complex(1e8, math.Sqrt(5)), complex(1e8, -math.Sqrt(5))}},
		{"nonstandard real", 1, 2, 3, 4, [2]complex128{complex((5+math.Sqrt(33))/2, 0), complex((5-math.Sqrt(33))/2, 0)}},
		{"near repeated real", 1, 1e-10, 1e-10, 1, [2]complex128{complex(1+1e-10, 0), complex(1-1e-10, 0)}},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			tol := 4 * eps() * max(math.Abs(tc.a), math.Abs(tc.b), math.Abs(tc.c), math.Abs(tc.d))
			got := schurEigenvaluesRaw([]float64{tc.a, tc.b, tc.c, tc.d}, 2)
			for i := range 2 {
				if cmplx.Abs(got[i]-tc.want[i]) > tol {
					t.Fatalf("schurEigenvaluesRaw = %v, want %v", got, tc.want)
				}
			}
			tt := []float64{
				7, 2, -3,
				0, tc.a, tc.b,
				0, tc.c, tc.d,
			}
			if ev := schurBlock2x2Eig(tt, 3, 1); cmplx.Abs(ev-tc.want[0]) > tol {
				t.Fatalf("schurBlock2x2Eig = %v, want %v", ev, tc.want[0])
			}
		})
	}
}

// badlyScaledPairSystem has poles other and re±im·i, with the pair sitting in a
// non-symmetric upper block-triangular A and MIMO B, C, D.
func badlyScaledPairSystem(t *testing.T, re, im, other, dt float64) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			other, 2, -1,
			0, re, im,
			0, -im, re,
		}),
		mat.NewDense(3, 2, []float64{1, 0, 0.5, 1, -1, 2}),
		mat.NewDense(2, 3, []float64{1, -2, 0.5, 0, 1, 3}),
		mat.NewDense(2, 2, []float64{0.1, 0, -0.2, 0.3}),
		dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func assertPolesMatch(t *testing.T, label string, sys *System, want []complex128, tol float64) {
	t.Helper()
	got, err := sys.Poles()
	if err != nil {
		t.Fatalf("%s: Poles: %v", label, err)
	}
	if len(got) != len(want) {
		t.Fatalf("%s: poles = %v, want %v", label, got, want)
	}
	used := make([]bool, len(got))
	for _, w := range want {
		found := false
		for i, g := range got {
			if !used[i] && cmplx.Abs(g-w) <= tol {
				used[i], found = true, true
				break
			}
		}
		if !found {
			t.Fatalf("%s: poles = %v, want %v (tol %g)", label, got, want, tol)
		}
	}
}
