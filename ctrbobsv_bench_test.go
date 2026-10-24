package controlsys

import (
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
