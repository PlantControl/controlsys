package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// unimodularSimilarity returns an integer T with integer inverse built from
// k random row additions, so T·A₀·T⁻¹ is exact in float64 for small A₀.
func unimodularSimilarity(rng *rand.Rand, n, k int) (T, Tinv *mat.Dense) {
	T, Tinv = eye(n), eye(n)
	for range k {
		i, j := rng.IntN(n), rng.IntN(n-1)
		if j >= i {
			j++
		}
		c := float64(rng.IntN(3) + 1)
		if rng.IntN(2) == 0 {
			c = -c
		}
		for col := range n {
			T.Set(i, col, T.At(i, col)+c*T.At(j, col))
		}
		for row := range n {
			Tinv.Set(row, j, Tinv.At(row, j)-c*Tinv.At(row, i))
		}
	}
	return T, Tinv
}

// jordanBoundaryPair returns (A, C) = (T·A₀·T⁻¹, C₀·T⁻¹) with T random
// unimodular and A₀ block upper triangular: a Jordan pair at mode (or at ±i
// in real rotation form when cplx) whose eigenvector C₀ does not see and
// whose generalized eigenvector it does, then the stable diagonal.
func jordanBoundaryPair(rng *rand.Rand, n int, mode float64, cplx bool, stable []float64) (A, C *mat.Dense) {
	A0 := mat.NewDense(n, n, nil)
	C0 := mat.NewDense(1, n, nil)
	k := 2
	if cplx {
		k = 4
		for _, b := range []int{0, 2} {
			A0.Set(b, b+1, -1)
			A0.Set(b+1, b, 1)
		}
		A0.Set(0, 2, 1)
		A0.Set(1, 3, 1)
		C0.Set(0, 2, 1)
		C0.Set(0, 3, float64(rng.IntN(5)-2))
	} else {
		A0.Set(0, 0, mode)
		A0.Set(1, 1, mode)
		A0.Set(0, 1, 1)
		C0.Set(0, 1, 1)
	}
	for i, v := range stable {
		A0.Set(k+i, k+i, v)
	}
	for i := range n {
		for j := max(i+1, k); j < n; j++ {
			A0.Set(i, j, float64(rng.IntN(7)-3))
		}
	}
	for j := k; j < n; j++ {
		C0.Set(0, j, float64(rng.IntN(5)-2))
	}
	C0.Set(0, k-1, 1)
	T, Tinv := unimodularSimilarity(rng, n, 14)
	A, C = new(mat.Dense), new(mat.Dense)
	A.Product(T, A0, Tinv)
	C.Mul(C0, Tinv)
	return A, C
}

// TestIsDetectable_BoundaryJordanUnobservable (F77544): an exactly
// unobservable mode on the stability boundary whose Jordan partner is
// observable, under exact integer similarities: undetectable by
// construction. The staircase isolates the mode with its eigenvalue
// √(eps·κ) off the boundary, on either side by rounding (FMA under -race
// flipped FuzzKalman 0xa7), or its amplified Krylov coupling exceeds the
// rank floor; base returned true for 495 of the 1500 trials. PBH rank
// at the boundary point decides.
func TestIsDetectable_BoundaryJordanUnobservable(t *testing.T) {
	rng := rand.New(rand.NewPCG(77544, 1))
	const n = 5
	for _, tc := range []struct {
		name       string
		continuous bool
		// mode is the boundary eigenvalue: real, or ±i as a 2×2 rotation.
		mode   float64
		cplx   bool
		stable []float64
	}{
		{"z=1", false, 1, false, []float64{0.5, -0.5, 0.25}},
		{"z=-1", false, -1, false, []float64{0.5, -0.5, 0.25}},
		{"z=±i", false, 0, true, []float64{0.5}},
		{"s=0", true, 0, false, []float64{-1, -2, -0.5}},
		{"s=±i", true, 0, true, []float64{-1}},
	} {
		for trial := range 300 {
			A, C := jordanBoundaryPair(rng, n, tc.mode, tc.cplx, tc.stable)
			ok, err := IsDetectable(A, C, tc.continuous)
			if err != nil {
				t.Fatalf("%s trial %d: %v", tc.name, trial, err)
			}
			if ok {
				t.Errorf("%s trial %d: IsDetectable true, unobservable Jordan mode on boundary", tc.name, trial)
			}
		}
	}
}

// TestIsStabilizable_StableUncontrollableNearBoundary: well-conditioned
// uncontrollable modes resolvably inside the boundary stay stabilizable,
// their mirrors outside it do not.
func TestIsStabilizable_StableUncontrollableNearBoundary(t *testing.T) {
	c, s := math.Cos(0.3), math.Sin(0.3)
	R := mat.NewDense(3, 3, []float64{c, -s, 0, s, c, 0, 0, 0, 1})
	for _, tc := range []struct {
		name       string
		lam, scale float64
		continuous bool
	}{
		{"z=0.99", 0.99, 1e3, false},
		{"z=1-1e-6", 1 - 1e-6, 1e3, false},
		{"z=-(1-1e-6)", -(1 - 1e-6), 10, false},
		{"s=-1e-6", -1e-6, 1e3, true},
		{"s=-1e-8", -1e-8, 10, true},
	} {
		A0 := mat.NewDense(3, 3, []float64{tc.lam, 0, 0, 0, -0.5, 1, 0, 0.25, tc.scale})
		B0 := mat.NewDense(3, 1, []float64{0, 1, 0})
		var A, B mat.Dense
		A.Product(R.T(), A0, R)
		B.Mul(R.T(), B0)
		ok, err := IsStabilizable(&A, &B, tc.continuous)
		if err != nil || !ok {
			t.Errorf("%s: IsStabilizable = %v, %v, want true", tc.name, ok, err)
		}
		if tc.continuous {
			A0.Set(0, 0, -tc.lam)
		} else {
			A0.Set(0, 0, 1/tc.lam)
		}
		A.Product(R.T(), A0, R)
		if ok, err := IsStabilizable(&A, &B, tc.continuous); err != nil || ok {
			t.Errorf("%s mirrored: IsStabilizable = %v, %v, want false", tc.name, ok, err)
		}
	}
}

// TestIsStabilizable_TinyInputScaleInvariant: a companion form with B = s·e₁
// is controllable for every s ≠ 0, poles within 1e-4 of z = 1 included
// (pidtuneIdealNearUnit d); σmin([A − I, B]) ≤ ‖B‖ must not read as rank
// loss when B is tiny.
func TestIsStabilizable_TinyInputScaleInvariant(t *testing.T) {
	den := []float64{-4.580963310835395, 8.325983494925048, -7.4921701917364585, 3.3302431420460463, -0.5830931343992402}
	const n = 5
	A := mat.NewDense(n, n, nil)
	for i := 1; i < n; i++ {
		A.Set(i, i-1, 1)
	}
	for i := range n {
		A.Set(i, n-1, -den[n-1-i])
	}
	for _, s := range []float64{8.43892612359764e-14, 1, 1e14} {
		B := mat.NewDense(n, 1, nil)
		B.Set(0, 0, s)
		if ok, err := IsStabilizable(A, B, false); err != nil || !ok {
			t.Errorf("s=%g: IsStabilizable = %v, %v, want true", s, ok, err)
		}
		var At, Bt mat.Dense
		At.CloneFrom(A.T())
		Bt.CloneFrom(B.T())
		if ok, err := IsDetectable(&At, &Bt, false); err != nil || !ok {
			t.Errorf("s=%g: IsDetectable = %v, %v, want true", s, ok, err)
		}
	}
}
