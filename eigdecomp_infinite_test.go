package controlsys

import (
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// weierstrassDescriptor is the SISO descriptor A = U·blkdiag(H, I_d)·V,
// E = U·blkdiag(J, N)·V, B = 1, C = 1ᵀ, with H, J random r×r and N = 0 or,
// with chain set, the d×d nilpotent shift (one infinite Jordan chain of
// length d). U, V are random orthogonal (orth) or random general. finite
// is the explicit model (J⁻¹H, J⁻¹·B_f, C_f) of its finite modes.
type weierstrassDescriptor struct {
	sys    *System
	finite *System
}

func newWeierstrassDescriptor(seed uint64, r, d int, chain, orth bool) weierstrassDescriptor {
	rng := rand.New(rand.NewPCG(seed, 91))
	n := r + d
	F := mat.NewDense(n, n, nil)
	W := mat.NewDense(n, n, nil)
	H := F.Slice(0, r, 0, r).(*mat.Dense)
	J := W.Slice(0, r, 0, r).(*mat.Dense)
	for i := range r {
		for j := range r {
			J.Set(i, j, rng.NormFloat64())
			H.Set(i, j, rng.NormFloat64())
		}
	}
	for i := r; i < n; i++ {
		F.Set(i, i, 1)
		if chain && i+1 < n {
			W.Set(i, i+1, 1)
		}
	}
	randMat := func() *mat.Dense {
		M := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				M.Set(i, j, rng.NormFloat64())
			}
		}
		if !orth {
			return M
		}
		var qr mat.QR
		qr.Factorize(M)
		var Q mat.Dense
		qr.QTo(&Q)
		return &Q
	}
	U, V := randMat(), randMat()
	var T, A, E mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B, C := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		B.Set(i, 0, 1)
		C.Set(0, i, 1)
	}
	sys, err := NewDescriptor(&A, B, C, mat.NewDense(1, 1, nil), &E, 0)
	if err != nil {
		panic(err)
	}

	var Ui, Vi mat.Dense
	if err := Ui.Inverse(U); err != nil {
		panic(err)
	}
	if err := Vi.Inverse(V); err != nil {
		panic(err)
	}
	var Bf, Cf, Af, JB mat.Dense
	Bf.Mul(Ui.Slice(0, r, 0, n), B)
	Cf.Mul(C, Vi.Slice(0, n, 0, r))
	var lu mat.LU
	lu.Factorize(J)
	if err := lu.SolveTo(&Af, false, H); err != nil {
		panic(err)
	}
	if err := lu.SolveTo(&JB, false, &Bf); err != nil {
		panic(err)
	}
	finite, err := New(&Af, &JB, &Cf, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		panic(err)
	}
	return weierstrassDescriptor{sys: sys, finite: finite}
}

func BenchmarkStabsep_SingularE_N20(b *testing.B) {
	sys := newWeierstrassDescriptor(1, 16, 4, false, true).sys
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Stabsep(sys); err != nil {
			b.Fatal(err)
		}
	}
}
