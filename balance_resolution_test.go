package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// resolventSigmaMin returns σmin(x·I − A) from the real 2n×2n embedding.
func resolventSigmaMin(A *mat.Dense, x complex128) float64 {
	n, _ := A.Dims()
	M := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			re := -A.At(i, j)
			if i == j {
				re += real(x)
				M.Set(i, n+j, -imag(x))
				M.Set(n+i, j, imag(x))
			}
			M.Set(i, j, re)
			M.Set(n+i, n+j, re)
		}
	}
	var svd mat.SVD
	if !svd.Factorize(M, mat.SVDNone) {
		return 0
	}
	v := svd.Values(nil)
	return v[len(v)-1]
}

// requireBalancedSimilarity checks that br is a certified similarity of
// the literal c: ‖TL·TR − I‖ exact within the certificate, and the exact
// frequency responses at fixed points within τ_cert·‖C‖·‖B‖/σmin(xI − A)
// with an 8× margin.
func requireBalancedSimilarity(t *testing.T, c hsvCase, br *BalrealResult) {
	t.Helper()
	if f, tau := exactCertNorm(br.TL, br.TR), certTau(br.HSV); f > tau {
		t.Fatalf("%s: exact ‖TL·TR − I‖ = %.3g above τ = %.3g", c.name, f, tau)
	}
	sys := c.sys()
	raw := func(m *mat.Dense) []float64 { return mat.DenseCopyOf(m).RawMatrix().Data }
	scale := mat.Norm(sys.C, 2) * mat.Norm(sys.B, 2) * certTau(br.HSV)
	for _, x := range []complex128{1i, 0.3 + 2i, 0.1} {
		g0, ok0 := fzExactResponse(c.n, c.m, c.p, c.a, c.b, c.c, make([]float64, c.m*c.p), x)
		g1, ok1 := fzExactResponse(c.n, c.m, c.p, raw(br.Sys.A), raw(br.Sys.B), raw(br.Sys.C), raw(br.Sys.D), x)
		if !ok0 || !ok1 {
			t.Fatalf("%s: exact response at %v failed", c.name, x)
		}
		bound := scale / resolventSigmaMin(sys.A, x)
		for k := range g0 {
			if d := cmplx.Abs(g0[k] - g1[k]); d > bound/8 {
				t.Fatalf("%s: |G − G_bal| = %.3g at %v above %.3g/8", c.name, d, x, bound)
			}
		}
	}
}

func TestBalreal_UnresolvedSigma1Unverified(t *testing.T) {
	for _, c := range balrealResolution[:2] {
		br, err := Balreal(c.sys())
		if !errors.Is(err, ErrAccuracyUnverified) || br == nil {
			t.Fatalf("%s: Balreal = %v, err %v, want the result with ErrAccuracyUnverified", c.name, br != nil, err)
		}
		requireBalancedSimilarity(t, c, br)
	}
}

func TestBalred_FollowsBalrealResolution(t *testing.T) {
	for _, c := range balrealResolution[:2] {
		for _, order := range []int{1, c.n} {
			red, hsv, err := Balred(c.sys(), order, BalredOptions{})
			if !errors.Is(err, ErrAccuracyUnverified) || red == nil || len(hsv) != c.n {
				t.Fatalf("%s: Balred(%d) model %v, %d HSV, err %v, want the reduced model with ErrAccuracyUnverified", c.name, order, red != nil, len(hsv), err)
			}
			if nr, _, _ := red.Dims(); nr != order {
				t.Fatalf("%s: Balred(%d) has %d states", c.name, order, nr)
			}
		}
	}
}

func TestBalreal_ResolvedSigma1StaysNil(t *testing.T) {
	c := balrealResolution[2]
	br, err := Balreal(c.sys())
	if err != nil {
		t.Fatalf("%s: Balreal err %v, want nil", c.name, err)
	}
	if v := hsvContractViolation(br.HSV, c.ref, c.T); v > 1.0/8 {
		t.Fatalf("%s: Balreal HSV %v off the 320-bit %v by %.3g of the contract bar", c.name, br.HSV, c.ref, v)
	}
	if c.ref[0] < 8*1.6e-12*math.Sqrt(float64(c.n)/4)*c.T {
		t.Fatalf("%s: σ₁/T = %.3g not clear of the resolution floor; the literal no longer tests it", c.name, c.ref[0]/c.T)
	}
	requireBalancedSimilarity(t, c, br)
	red, _, err := Balred(c.sys(), 1, BalredOptions{})
	if err != nil || red == nil {
		t.Fatalf("%s: Balred(1) err %v, want nil", c.name, err)
	}
}

// rlrep/23 as stored on arm64 (D5BGNA): the balancing transformation fails
// its certificate, Balred reduces the certified head, and σ₁/T ≈ 10⁻¹⁷ is
// below the double-double resolution.
func TestBalred_CertifiedHeadUnresolvedUnverified(t *testing.T) {
	c := hsvBelowResolution[0]
	if _, err := Balreal(c.sys()); !errors.Is(err, ErrSingularTransform) {
		t.Fatalf("%s: Balreal err %v, want ErrSingularTransform (certified-head path)", c.name, err)
	}
	for _, order := range []int{1, 2} {
		red, hsv, err := Balred(c.sys(), order, BalredOptions{})
		if !errors.Is(err, ErrAccuracyUnverified) || red == nil || len(hsv) != c.n {
			t.Fatalf("%s: Balred(%d) model %v, %d HSV, err %v, want the reduced head with ErrAccuracyUnverified", c.name, order, red != nil, len(hsv), err)
		}
		if nr, _, _ := red.Dims(); nr != order {
			t.Fatalf("%s: Balred(%d) has %d states", c.name, order, nr)
		}
	}
}
