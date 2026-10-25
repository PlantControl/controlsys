package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestIsStabilizable_AmplifiedRoundingCases: exactly uncontrollable
// unstable modes the staircase counted controllable with coupling far above
// rounding, amplified by nearly parallel inputs (MinCoupling 5e-6, unstable
// real mode at 1.38, PBH σmin 1.7e-16) or by a dominating unstable
// eigenvalue (Q·diag(−1, −2, 100)·Qᵀ, B = Q·[1; 1; 0], MinCoupling 3.5e-13,
// PBH σmin at 100 7.5e-15).
func TestIsStabilizable_AmplifiedRoundingCases(t *testing.T) {
	cases := []struct {
		name string
		n, m int
		a, b []float64
	}{
		{"nearly parallel inputs", 4, 2,
			[]float64{-1.6117933783199974, 0.48359582334550888, 0.44937364920158951, -0.51447634974100864, 1.1678863890269564, 0.82703574685249792, -0.2611807475204676, -0.46124691741871388, 0.9729290197519127, -0.70935587192456262, -1.0383262881189503, 0.30275810984091633, 1.57794535064403, -1.3169609384891536, 0.64356499433223746, -0.87441757639230011},
			[]float64{0.19033953569052783, 0.024083864433425729, -0.26762466725641804, -0.033862834548092392, -0.39630877049851104, -0.050145371362450557, -0.6409332310140462, -0.081097965223341506}},
		{"dominant unstable mode", 3, 1,
			[]float64{-1, 0, 0, 0, 31.474066937881602, -47.89406717227244, 0, -47.89406717227244, 66.52593306211838},
			[]float64{1, 0.8196480178454795, 0.5728674601004813}},
	}
	for _, c := range cases {
		A := mat.NewDense(c.n, c.n, c.a)
		B := mat.NewDense(c.n, c.m, c.b)
		if ok, err := IsStabilizable(A, B, true); err != nil || ok {
			t.Errorf("%s: IsStabilizable = %v, %v, want false", c.name, ok, err)
		}
		if ok, err := IsDetectable(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T()), true); err != nil || ok {
			t.Errorf("%s: IsDetectable = %v, %v, want false", c.name, ok, err)
		}
	}
}

// nearlyParallelInputSystem: A₀ = [A11 A12; 0 A22], two inputs
// B₀ = [b, αb + c·e_k; 0], so the stable state k is reached only through
// c ∈ 10^U(−12, −8). A22 is a boundary rotation (kind 0), an unstable
// rotation (1) or a real unstable mode (2), lam its eigenvalue. d ≠ 0
// drives A22's last state through both inputs, making it controllable.
func nearlyParallelInputSystem(rng *rand.Rand, kind int, continuous bool, d float64) (A0, B0 *mat.Dense, lam complex128) {
	nc := 2 + rng.IntN(4)
	nu := 2
	if kind == 2 {
		nu = 1
	}
	n := nc + nu
	A0 = mat.NewDense(n, n, nil)
	B0 = mat.NewDense(n, 2, nil)
	k := nc - 1
	for i := range nc {
		s := -0.2 - rng.Float64()
		if !continuous {
			s = 0.9 * (2*rng.Float64() - 1)
		}
		A0.Set(i, i, s)
		if i != k {
			for j := i + 1; j < n; j++ {
				A0.Set(i, j, rng.NormFloat64())
			}
		}
	}
	for j := nc; j < n; j++ {
		A0.Set(k, j, rng.NormFloat64())
	}
	alpha := rng.NormFloat64()
	for i := range k {
		b := rng.NormFloat64()
		B0.Set(i, 0, b)
		B0.Set(i, 1, alpha*b)
	}
	B0.Set(k, 1, math.Pow(10, -12+4*rng.Float64()))
	if kind == 2 {
		l := 0.3 + 1.5*rng.Float64()
		if !continuous {
			l = 1.2 + 0.5*rng.Float64()
		}
		A0.Set(nc, nc, l)
		lam = complex(l, 0)
	} else {
		w := 0.5 + rng.Float64()
		re, im := 0.0, w
		if kind == 1 {
			re = 0.3 + rng.Float64()
		}
		if !continuous {
			r := 1.0
			if kind == 1 {
				r = 1.2 + 0.5*rng.Float64()
			}
			re, im = r*math.Cos(w), r*math.Sin(w)
		}
		A0.Set(nc, nc, re)
		A0.Set(nc+1, nc+1, re)
		A0.Set(nc, nc+1, -im)
		A0.Set(nc+1, nc, im)
		lam = complex(re, im)
	}
	if d != 0 {
		B0.Set(n-1, 0, d*rng.NormFloat64())
		B0.Set(n-1, 1, d*rng.NormFloat64())
	}
	return A0, B0, lam
}

// TestIsStabilizable_NearlyParallelInputsSweep: nearly parallel inputs
// amplify the rounding coupling of an exactly uncontrollable unstable or
// boundary mode to 1e-7..1e-4, above the weak-stair gate, so base counted
// it controllable (every uncontrollable draw wrong). Controllable draws,
// d = 1 or d ∈ 10^U(−8, −3), are asserted when MATLAB's ctrbf (CtrbF, tol 0)
// gives nc = n and the exact (A₀, B₀)'s distance to uncontrollability, B₀
// scaled to ‖A₀‖_F, exceeds 100·n²·eps·‖[A₀, B₀]‖_F.
func TestIsStabilizable_NearlyParallelInputsSweep(t *testing.T) {
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(0x4b59, 23))
	var checked, misses [3]int
	for trial := range 900 {
		continuous := trial%2 == 0
		kind := trial % 3
		v := (trial / 3) % 3
		d := 0.0
		switch v {
		case 1:
			d = 1
		case 2:
			d = math.Pow(10, -8+5*rng.Float64())
		}
		A0, B0, lam := nearlyParallelInputSystem(rng, kind, continuous, d)
		n, _ := A0.Dims()
		want := d != 0
		if want {
			an := frobNorm(A0)
			bs := mat.DenseCopyOf(B0)
			bs.Scale(an/frobNorm(B0), bs)
			if pbhDistanceNear(A0, bs, lam) < 100*float64(n*n)*eps()*math.Sqrt2*an {
				continue
			}
		}
		Q := stabSweepOrthogonal(rng, n)
		var A, B mat.Dense
		A.Product(Q, A0, Q.T())
		B.Mul(Q, B0)
		if want {
			sf, err := CtrbF(&A, &B, mat.NewDense(1, n, nil), 0)
			if err != nil {
				t.Fatal(err)
			}
			if sumInts(sf.K) != n {
				continue
			}
		}
		checked[v]++
		got, err := IsStabilizable(&A, &B, continuous)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		dg, err := IsDetectable(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T()), continuous)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if got != want || dg != want {
			misses[v]++
			if misses[v] <= 5 {
				t.Errorf("trial %d (n=%d kind=%d cont=%v d=%g): IsStabilizable = %v, IsDetectable = %v, want %v", trial, n, kind, continuous, d, got, dg, want)
			}
		}
	}
	t.Logf("checked %v", checked)
	if misses != [3]int{} || checked[0] < 300 || checked[1] < 290 || checked[2] < 250 {
		t.Errorf("checked %v (uncontrollable, d = 1, weak d): misses %v", checked, misses)
	}
}

// TestIsStabilizable_DominantUnstableModeSweep: A₀ = [A11 a; 0 λ], B₀ =
// [B1; 0], λ ∈ 10^U(1, 4) exactly uncontrollable and dominating a
// nonnormal stable A11, under random orthogonal similarity. Each staircase
// step multiplies the rounding in λ's direction, reaching couplings of
// 1e-6..1e-4·‖A‖_F (above the weak-stair gate) while PBH σmin at λ is
// 1e-3..5e-2 of n²·eps·‖[A, B]‖_F. Base counted 72% of draws stabilizable,
// the weak-stair gate alone 13%.
func TestIsStabilizable_DominantUnstableModeSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(991, 7))
	var misses int
	for trial := range 1000 {
		continuous := trial%2 == 0
		n := 3 + rng.IntN(5)
		m := 1 + rng.IntN(2)
		A0 := mat.NewDense(n, n, nil)
		B0 := mat.NewDense(n, m, nil)
		for i := range n - 1 {
			A0.Set(i, i, -0.2-3*rng.Float64())
			if !continuous {
				A0.Set(i, i, 0.9*(2*rng.Float64()-1))
			}
			for j := i + 1; j < n; j++ {
				A0.Set(i, j, rng.NormFloat64()*math.Pow(10, 2*rng.Float64()))
			}
			for j := range m {
				B0.Set(i, j, rng.NormFloat64())
			}
		}
		A0.Set(n-1, n-1, math.Pow(10, 1+3*rng.Float64()))
		Q := stabSweepOrthogonal(rng, n)
		var A, B mat.Dense
		A.Product(Q, A0, Q.T())
		B.Mul(Q, B0)
		got, err := IsStabilizable(&A, &B, continuous)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		dg, err := IsDetectable(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T()), continuous)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if got || dg {
			misses++
			if misses <= 5 {
				t.Errorf("trial %d (n=%d m=%d cont=%v λ=%.4g): IsStabilizable = %v, IsDetectable = %v, want false", trial, n, m, continuous, A0.At(n-1, n-1), got, dg)
			}
		}
	}
	if misses > 0 {
		t.Errorf("%d of 1000 uncontrollable draws stabilizable", misses)
	}
}

// TestIsStabilizable_AmplifiedGateControllable: controllable random models
// whose staircase amplification bound fires the PBH pass (inputs with σ
// ratios down to 1e-14, a dominant diagonal entry) stay stabilizable. Each is
// asserted only when the realified σmin([A − λI, B]) at every computed λ on
// or outside the boundary, B scaled to ‖A‖_F, exceeds 1e3·n²·eps·‖[A, B]‖_F.
func TestIsStabilizable_AmplifiedGateControllable(t *testing.T) {
	rng := rand.New(rand.NewPCG(77, 23))
	var checked, fired int
	for trial := range 3000 {
		n := 2 + rng.IntN(9)
		m := 2 + rng.IntN(3)
		continuous := trial%2 == 0
		A := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				A.Set(i, j, rng.NormFloat64())
			}
		}
		if rng.IntN(3) == 0 {
			A.Set(0, 0, A.At(0, 0)+math.Pow(10, 4*rng.Float64()))
		}
		U := stabSweepOrthogonal(rng, n)
		V := stabSweepOrthogonal(rng, m)
		S := mat.NewDense(n, m, nil)
		S.Set(0, 0, 1)
		for i := 1; i < min(n, m); i++ {
			S.Set(i, i, math.Pow(10, -14*rng.Float64()))
		}
		var B, us mat.Dense
		us.Mul(U, S)
		B.Mul(&us, V.T())
		an := frobNorm(A)
		st, err := controllabilityStaircase(A, &B, nil, 0, false)
		if err != nil {
			t.Fatal(err)
		}
		if st.NCont != n || st.MinCoupling > float64(n)*eps()*an*st.Amplification {
			continue
		}
		fired++
		bs := mat.DenseCopyOf(&B)
		bs.Scale(an/frobNorm(&B), bs)
		floor := float64(n*n) * eps() * math.Sqrt2 * an
		var e mat.Eigen
		if !e.Factorize(A, mat.EigenNone) {
			t.Fatal("eig failed")
		}
		ok := true
		for _, v := range e.Values(nil) {
			if poleOnOrOutsideStabilityBoundary(v, continuous, poleStabilityTolerance(v)) && realifiedPBHSigma(A, bs, v) < 1e3*floor {
				ok = false
			}
		}
		if !ok {
			continue
		}
		checked++
		got, err := IsStabilizable(A, &B, continuous)
		if err != nil || !got {
			t.Errorf("trial %d (n=%d m=%d cont=%v): IsStabilizable = %v, %v, want true", trial, n, m, continuous, got, err)
		}
	}
	t.Logf("fired %d checked %d", fired, checked)
	if checked < 200 {
		t.Errorf("checked %d of %d fired, want ≥ 200", checked, fired)
	}
}
