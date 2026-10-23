package controlsys

import (
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// benchSingularDescriptor returns U·(sE₀−A₀)·V with E₀ = blkdiag(I_nf, N)
// and A₀ = blkdiag(F, I), N nilpotent with the given chain lengths (index 1
// when all are 1), U, V random.
func benchSingularDescriptor(b *testing.B, nf int, chains []int) *System {
	b.Helper()
	rng := rand.New(rand.NewPCG(1, 2))
	n := nf
	for _, k := range chains {
		n += k
	}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	bb, cc := mat.NewDense(n, 2, nil), mat.NewDense(2, n, nil)
	for i := range nf {
		W.Set(i, i, 1)
		for j := range nf {
			F.Set(i, j, rng.NormFloat64())
		}
	}
	o := nf
	for _, k := range chains {
		for i := range k {
			F.Set(o+i, o+i, 1)
			if i+1 < k {
				W.Set(o+i, o+i+1, 1)
			}
		}
		bb.Set(o, 0, 1)
		o += k
	}
	for i := range n {
		cc.Set(0, i, rng.NormFloat64())
		cc.Set(1, i, rng.NormFloat64())
		if i < nf {
			bb.Set(i, 0, rng.NormFloat64())
			bb.Set(i, 1, rng.NormFloat64())
		}
	}
	U, V := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			U.Set(i, j, rng.NormFloat64())
			V.Set(i, j, rng.NormFloat64())
		}
	}
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, bb)
	C.Mul(cc, V)
	sys, err := NewDescriptor(&A, &B, &C, mat.NewDense(2, 2, nil), &E, 0)
	if err != nil {
		b.Fatal(err)
	}
	return sys
}

func benchDescriptorPoles(b *testing.B, sys *System) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.Poles(); err != nil {
			b.Fatal(err)
		}
	}
}

func benchDescriptorTF(b *testing.B, sys *System) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.TransferFunction(nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkDescriptorPoles_Index1_N10(b *testing.B) {
	benchDescriptorPoles(b, benchSingularDescriptor(b, 7, []int{1, 1, 1}))
}

func BenchmarkDescriptorPoles_Index1_N40(b *testing.B) {
	benchDescriptorPoles(b, benchSingularDescriptor(b, 30, []int{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}))
}

func BenchmarkDescriptorPoles_Index2_N10(b *testing.B) {
	benchDescriptorPoles(b, benchSingularDescriptor(b, 6, []int{2, 2}))
}

func BenchmarkDescriptorTF_Index1_N10(b *testing.B) {
	benchDescriptorTF(b, benchSingularDescriptor(b, 7, []int{1, 1, 1}))
}

func BenchmarkDescriptorTF_Index2_N10(b *testing.B) {
	benchDescriptorTF(b, benchSingularDescriptor(b, 6, []int{2, 2}))
}
