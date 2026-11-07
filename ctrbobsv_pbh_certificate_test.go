package controlsys

import (
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestPBHSolverExceeds_CertifiedLowerBound: the Cholesky certificate that
// lets the weak-gate PBH pass skip the SVD (JAZP5L) never claims
// σmin([A − λI, B]) > τ when the SVD gives σmin ≤ τ, over random,
// near-uncontrollable and exactly uncontrollable models at their
// eigenvalues and nearby points, with τ around σmin; and it certifies
// generic models, so the pass stays cheap.
func TestPBHSolverExceeds_CertifiedLowerBound(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 9))
	var certified, generic, genericCertified int
	for trial := range 600 {
		n := 2 + rng.IntN(30)
		m := 1 + rng.IntN(3)
		nu := 0
		eps0 := 0.0
		if trial%3 != 0 {
			nu = 1 + rng.IntN(min(3, n-1))
			if trial%3 == 2 {
				eps0 = math.Pow(10, -12+10*rng.Float64())
			}
		}
		A0 := mat.NewDense(n, n, nil)
		B0 := mat.NewDense(n, m, nil)
		s := 1 / math.Sqrt(float64(n))
		for i := range n {
			for j := range n {
				A0.Set(i, j, s*rng.NormFloat64())
				if i >= n-nu && j < n-nu {
					A0.Set(i, j, eps0*A0.At(i, j))
				}
			}
			for j := range m {
				B0.Set(i, j, rng.NormFloat64())
				if i >= n-nu {
					B0.Set(i, j, eps0*B0.At(i, j))
				}
			}
		}
		Q := stabSweepOrthogonal(rng, n)
		var A, B mat.Dense
		A.Product(Q, A0, Q.T())
		B.Mul(Q, B0)
		var e mat.Eigen
		if !e.Factorize(&A, mat.EigenNone) {
			t.Fatal("eig failed")
		}
		p := newPBHSolver(&A, &B)
		for _, lam := range e.Values(nil) {
			for _, z := range []complex128{lam, lam + complex(1e-3*rng.NormFloat64(), 1e-3*rng.NormFloat64())} {
				if imag(z) < 0 {
					z = complex(real(z), -imag(z))
				}
				sv := p.sigmaMin(z)
				for _, f := range []float64{0.5, 0.9, 0.99, 1.01, 1.1, 2} {
					tau := f * sv
					if p.exceeds(z, tau) {
						certified++
						if sv <= tau {
							t.Errorf("trial %d (n=%d m=%d nu=%d ε=%g) λ=%v: exceeds(%g) with SVD σmin %g", trial, n, m, nu, eps0, z, tau, sv)
						}
					}
				}
				if nu == 0 && z == lam {
					generic++
					if p.exceeds(z, 1e-3*sv) {
						genericCertified++
					}
				}
			}
		}
	}
	if certified == 0 || genericCertified < generic*9/10 {
		t.Errorf("certified %d; generic %d of %d at τ = 1e-3·σmin, want ≥ 90%%", certified, genericCertified, generic)
	}
}

// TestPBHSolverExceeds_WeakGateCasesNotSkipped: on the weak-gate inputs of
// 4KYHPC (nearly parallel inputs, dominant unstable mode, weak chain end
// with c = 0), the certificate does not skip the uncontrollable unstable
// mode at the PBH pass's τ = 2·(floor + band), so the SVD still decides
// it (JAZP5L).
func TestPBHSolverExceeds_WeakGateCasesNotSkipped(t *testing.T) {
	type sys struct {
		name string
		A, B *mat.Dense
	}
	var cases []sys
	for _, c := range []struct {
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
	} {
		cases = append(cases, sys{c.name, mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, c.m, c.b)})
	}
	for _, single := range []bool{true, false} {
		A, B := weakChainEndSystem(0, true, single)
		cases = append(cases, sys{"weak chain end", A, B})
	}
	for _, c := range cases {
		n, _ := c.A.Dims()
		an := frobNorm(c.A)
		B := mat.DenseCopyOf(c.B)
		B.Scale(an/frobNorm(B), B)
		floor := float64(n*n) * eps() * math.Hypot(an, frobNorm(B))
		band := float64(n) * math.Sqrt(eps()) * an
		p := newPBHSolver(c.A, B)
		var e mat.Eigen
		if !e.Factorize(c.A, mat.EigenNone) {
			t.Fatal("eig failed")
		}
		hit := false
		for _, lam := range e.Values(nil) {
			if imag(lam) < 0 || !poleOnOrOutsideStabilityBoundary(lam, true, poleStabilityTolerance(lam)) {
				continue
			}
			if p.sigmaMin(lam) <= floor+band {
				hit = true
				if p.exceeds(lam, 2*(floor+band)) {
					t.Errorf("%s: λ=%v certified above 2·(floor+band) with SVD σmin %g", c.name, lam, p.sigmaMin(lam))
				}
			}
		}
		if !hit {
			t.Errorf("%s: no unstable λ with σmin ≤ floor+band", c.name)
		}
	}
}
