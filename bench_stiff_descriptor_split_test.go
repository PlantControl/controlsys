package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// benchStiffClusterDescriptor is the 5×5 pencil of the SCPYGG repro: E of
// cond 1e9 and a near-defective pair (δ) tuned to the nominal stiff cut;
// δ = 0.5 leaves the pair well separated.
func benchStiffClusterDescriptor(b *testing.B, delta float64) *System {
	b.Helper()
	rng := rand.New(rand.NewPCG(0, 0xc2d))
	u, v := randomOrthogonal(rng, 5), randomOrthogonal(rng, 5)
	build := func(kap float64) *System {
		A0 := mat.NewDense(5, 5, []float64{
			-1, 0.7, 0.3, -0.2, 0.1,
			0, -2.5, 0.4, 0.2, -0.3,
			0, 0, -kap, 1, 0.2,
			0, 0, 0, -kap * (1 + delta), 0.3,
			0, 0, 0, 0, -1})
		E0 := mat.NewDense(5, 5, nil)
		for i, d := range []float64{1, 1, 1e-4, 1e-4, 1e-9} {
			E0.Set(i, i, d)
		}
		var a, e, bb, c, tmp mat.Dense
		tmp.Mul(u, A0)
		a.Mul(&tmp, v.T())
		tmp.Reset()
		tmp.Mul(u, E0)
		e.Mul(&tmp, v.T())
		bb.Mul(u, mat.NewDense(5, 1, []float64{1, -0.5, 0.8, 1.2, 0.6}))
		c.Mul(mat.NewDense(1, 5, []float64{1, 0.7, -0.6, 0.9, 0.4}), v.T())
		sys, err := NewDescriptor(&a, &bb, &c, mat.NewDense(1, 1, nil), &e, 0)
		if err != nil {
			b.Fatal(err)
		}
		return sys
	}
	kap := 0.3
	for range 80 {
		eq := equilibrateDescriptor(build(kap))
		kap = 1 / qzLargeRatio * mat.Norm(eq.A, 2) / mat.Norm(eq.E, 2) * 1e-4 / (1 + delta/2)
	}
	return build(kap)
}

// benchStiffDenseDescriptor is a dense n = 20 descriptor with E singular
// values spread over 1 … 1e-10.
func benchStiffDenseDescriptor(b *testing.B) *System {
	b.Helper()
	const n = 20
	rng := rand.New(rand.NewPCG(20, 0x5f))
	F := mat.NewDense(n, n, nil)
	W := make([]float64, n)
	for i := range n {
		F.Set(i, i, -1-rng.Float64())
		for j := i + 1; j < n; j++ {
			F.Set(i, j, 0.3*rng.NormFloat64())
		}
		W[i] = math.Pow(10, -10*float64(i)/float64(n-1))
	}
	bv, cv := make([]float64, n), make([]float64, n)
	for i := range n {
		bv[i], cv[i] = rng.NormFloat64(), rng.NormFloat64()
	}
	U, V := randomOrthogonal(rng, n), randomOrthogonal(rng, n)
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, mat.NewDiagDense(n, W))
	E.Mul(&T, V)
	B.Mul(U, mat.NewDense(n, 1, bv))
	C.Mul(mat.NewDense(1, n, cv), V)
	sys, err := NewDescriptor(&A, &B, &C, mat.NewDense(1, 1, nil), &E, 0)
	if err != nil {
		b.Fatal(err)
	}
	return sys
}

type benchStiffCase struct {
	name string
	sys  *System
}

func benchStiffDescriptorCases(b *testing.B) []benchStiffCase {
	return []benchStiffCase{
		{"Cluster", benchStiffClusterDescriptor(b, 1e-6)},
		{"Apart", benchStiffClusterDescriptor(b, 0.5)},
		{"DenseN20", benchStiffDenseDescriptor(b)},
	}
}

func BenchmarkC2DStiffDescriptor(b *testing.B) {
	for _, bc := range benchStiffDescriptorCases(b) {
		name, sys := bc.name, bc.sys
		for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodTustin} {
			b.Run(name+"/"+string(m), func(b *testing.B) {
				for b.Loop() {
					sys.C2D(0.01, C2DOptions{Method: m})
				}
			})
		}
	}
}

func BenchmarkStepStiffDescriptor(b *testing.B) {
	for _, bc := range benchStiffDescriptorCases(b) {
		name, sys := bc.name, bc.sys
		b.Run(name, func(b *testing.B) {
			for b.Loop() {
				Step(sys, 5)
			}
		})
	}
}

func BenchmarkImpulseStiffDescriptor(b *testing.B) {
	for _, bc := range benchStiffDescriptorCases(b) {
		name, sys := bc.name, bc.sys
		b.Run(name, func(b *testing.B) {
			for b.Loop() {
				Impulse(sys, 5)
			}
		})
	}
}

func BenchmarkTransferFunctionStiffDescriptor(b *testing.B) {
	for _, bc := range benchStiffDescriptorCases(b) {
		name, sys := bc.name, bc.sys
		b.Run(name, func(b *testing.B) {
			for b.Loop() {
				tfVerdicts.Lock()
				tfVerdicts.m = nil
				tfVerdicts.Unlock()
				chainRegimes.Lock()
				chainRegimes.m = nil
				chainRegimes.Unlock()
				sys.TransferFunction(nil)
			}
		})
	}
}
