package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"
)

// singularBlockTriangularSystem returns an upper-triangular model whose
// states 0–2 form the dense rank-1 integer block u·vᵀ, u = [1 −2 3],
// v = [2 1 −1], so x0·I − A is exactly singular there (shifted by x0 for
// discrete models) and DCGain takes the exact rational limit.
func singularBlockTriangularSystem(t testing.TB, rng *rand.Rand, n, m, p int, dt float64) *System {
	t.Helper()
	x0 := 0.0
	if dt > 0 {
		x0 = 1
	}
	u, v := []float64{1, -2, 3}, []float64{2, 1, -1}
	a := make([]float64, n*n)
	for i := range n {
		for j := i; j < n; j++ {
			a[i*n+j] = rng.NormFloat64()
		}
		if i >= 3 {
			a[i*n+i] = x0 - 0.5 - rng.Float64()
		}
	}
	for i := range 3 {
		for j := range 3 {
			a[i*n+j] = u[i] * v[j]
			if i == j {
				a[i*n+j] += x0
			}
		}
	}
	b, c, d := make([]float64, n*m), make([]float64, p*n), make([]float64, p*m)
	for i := range b {
		b[i] = rng.NormFloat64()
	}
	for i := range c {
		c[i] = rng.NormFloat64()
	}
	for i := range d {
		if rng.IntN(2) == 0 {
			d[i] = rng.NormFloat64()
		}
	}
	sys, err := NewFromSlices(n, m, p, a, b, c, d, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// FSLFI7: DCGain on an exactly singular dense block equals the exact
// σ → 0+ limit bit for bit, Inf entries by sign, across orders up to
// exactDetMaxOrder and both domains.
func TestDCGain_ExactlySingularBlockTriangularMatchesExactLimit(t *testing.T) {
	rng := rand.New(rand.NewPCG(30, 7))
	for _, tc := range []struct{ n, m, p int }{{3, 1, 1}, {4, 2, 3}, {7, 3, 2}, {10, 2, 2}, {16, 4, 4}} {
		for _, dt := range []float64{0, 0.1} {
			sys := singularBlockTriangularSystem(t, rng, tc.n, tc.m, tc.p, dt)
			exact := dcGainExactLimit(t, sys)
			g, err := sys.DCGain()
			if err != nil {
				t.Fatalf("n=%d dt=%v: %v", tc.n, dt, err)
			}
			for i, row := range exact {
				for j, w := range row {
					if got := g.At(i, j); got != w && !(math.IsNaN(got) && math.IsNaN(w)) {
						t.Errorf("n=%d dt=%v: gain[%d,%d] = %v, exact limit %v", tc.n, dt, i, j, got, w)
					}
				}
			}
		}
	}
}

func BenchmarkDCGain_ExactlySingularBlock_N16_M4_P4(b *testing.B) {
	sys := singularBlockTriangularSystem(b, rand.New(rand.NewPCG(1, 2)), 16, 4, 4, 0)
	b.ResetTimer()
	for b.Loop() {
		if _, err := sys.DCGain(); err != nil {
			b.Fatal(err)
		}
	}
}
