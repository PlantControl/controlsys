package controlsys

import (
	"testing"
)

func BenchmarkRootLocus(b *testing.B) {
	mk := func(roots []complex128, num []float64, dd float64, seed int) (*System, error) {
		den := rlswPolyFromRoots(roots)
		d := make([]float64, len(den))
		for i, x := range den {
			d[i] = rlswF(x)
		}
		n := len(d) - 1
		a, bb, c := rlpRotatedCompanion(d, num, seed)
		return NewFromSlices(n, 1, 1, a, bb, c, []float64{dd}, 0)
	}
	cases := []struct {
		name  string
		roots []complex128
		num   []float64
		d     float64
	}{
		{"n4", []complex128{-1, -2, complex(-3, 1), complex(-3, -1)}, []float64{1, 5}, 0},
		{"n4neg", []complex128{-1, -2, complex(-3, 1), complex(-3, -1)}, []float64{-1, -5}, 0},
		{"n4dneg", []complex128{-1, -2, complex(-3, 1), complex(-3, -1)}, []float64{1, 5}, -0.5},
		{"n8", []complex128{-1, -2, -3, -4, complex(-1, 2), complex(-1, -2), complex(-0.5, 3), complex(-0.5, -3)}, []float64{1, 3, 2}, 0},
		{"n12", []complex128{-1, -2, -3, -4, -5, -6, complex(-1, 2), complex(-1, -2), complex(-0.5, 3), complex(-0.5, -3), complex(-2, 1), complex(-2, -1)}, []float64{1, 3, 2, 7}, 0},
		{"n12d", []complex128{-1, -2, -3, -4, -5, -6, complex(-1, 2), complex(-1, -2), complex(-0.5, 3), complex(-0.5, -3), complex(-2, 1), complex(-2, -1)}, []float64{1, 3, 2, 7}, 0.25},
	}
	for _, tc := range cases {
		sys, err := mk(tc.roots, tc.num, tc.d, 1)
		if err != nil {
			b.Fatal(err)
		}
		b.Run(tc.name, func(b *testing.B) {
			for b.Loop() {
				if _, err := RootLocus(sys, nil); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
