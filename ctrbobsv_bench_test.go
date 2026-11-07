package controlsys

import (
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func BenchmarkIsStabilizable(b *testing.B) {
	rng := rand.New(rand.NewPCG(1, 2))
	const n, m = 10, 2
	gen := func(nu int) (*mat.Dense, *mat.Dense) {
		A := mat.NewDense(n, n, nil)
		B := mat.NewDense(n, m, nil)
		for i := range n {
			for j := range n {
				if i < n-nu || j >= n-nu {
					A.Set(i, j, rng.NormFloat64())
				}
			}
			if i < n-nu {
				for j := range m {
					B.Set(i, j, rng.NormFloat64())
				}
			}
		}
		return A, B
	}
	for _, bc := range []struct {
		name string
		nu   int
	}{{"controllable", 0}, {"uncontrollable", 3}} {
		A, B := gen(bc.nu)
		b.Run(bc.name, func(b *testing.B) {
			for b.Loop() {
				if _, err := IsStabilizable(A, B, true); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
	r := roundingCoupledRepros[1]
	A := mat.NewDense(r.n, r.n, r.a)
	B := mat.NewDense(r.n, 1, r.b)
	b.Run("rounding-coupled", func(b *testing.B) {
		for b.Loop() {
			if _, err := IsStabilizable(A, B, true); err != nil {
				b.Fatal(err)
			}
		}
	})
	A, B = weakChainEndSystem(1e-9, true, false)
	b.Run("weak-chain-end", func(b *testing.B) {
		for b.Loop() {
			if _, err := IsStabilizable(A, B, true); err != nil {
				b.Fatal(err)
			}
		}
	})
	A, B = gen(0)
	B.Scale(1e-12, B)
	b.Run("tiny-input", func(b *testing.B) {
		for b.Loop() {
			if _, err := IsStabilizable(A, B, true); err != nil {
				b.Fatal(err)
			}
		}
	})
}

// BenchmarkIsStabilizableRandom: generic A = N(0, 1/n), B = N(0, 1) with one
// input, whose staircase amplification bound fires the PBH pass.
func BenchmarkIsStabilizableRandom(b *testing.B) {
	for _, n := range []int{25, 35} {
		rng := rand.New(rand.NewPCG(uint64(n), 5))
		A := mat.NewDense(n, n, nil)
		B := mat.NewDense(n, 1, nil)
		s := 1 / math.Sqrt(float64(n))
		for i := range n {
			for j := range n {
				A.Set(i, j, s*rng.NormFloat64())
			}
			B.Set(i, 0, rng.NormFloat64())
		}
		b.Run(fmt.Sprintf("n=%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := IsStabilizable(A, B, true); err != nil {
					b.Fatal(err)
				}
			}
		})
		C := mat.DenseCopyOf(B.T())
		b.Run(fmt.Sprintf("detectable/n=%d", n), func(b *testing.B) {
			for b.Loop() {
				if _, err := IsDetectable(A, C, false); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
