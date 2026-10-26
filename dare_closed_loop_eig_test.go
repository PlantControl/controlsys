package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// LRUQLU: with an open-loop pair on the unit circle (c2d of ±i), a
// repeated unstable mode and Q ≈ 1e-12, or A ≈ I with rank-1 Q, Dare's Eig
// once sat 2e-4–9e-4 from eig(A − BK) because the Newton refinement moved
// K without recomputing Eig. These are the FuzzRiccati inputs of those
// reports (also in testdata/fuzz/FuzzRiccati), built as the fuzz target
// builds them; Eig must be the spectrum of A − BK of the returned K.
func TestDare_EigMatchesClosedLoopOfReturnedGain(t *testing.T) {
	for _, tc := range []struct {
		kind uint8
		vals [fzNVals]float64
	}{
		{0x9b, [fzNVals]float64{376, -63, 11, -2, 0, 0, 0, -79, -48, -10, 0.37, 0, 100, 50, -20, -10}},
		{0x93, [fzNVals]float64{0, 97, 0.5, -2, -37, 120, 0, 55, 0, -10, 0.37, 0, 1, -10, 44, -10}},
		{0x47, [fzNVals]float64{-70, -2.5, -35, 0, 0, 0, -7, 0, 0, -10, 0.37, -63, -10, -10, -10, -10}},
	} {
		s := fzDecode(t, tc.kind, tc.vals)
		n, m, p := s.N, s.M, s.P
		q := mqGram(s.C, p, n, true)
		r := mqGram(s.D, p, m, true)
		rr := math.Pow(10, fzClamp(s.Vals[13], -3, 3))
		for i := range m {
			r[i*m+i] += rr
		}
		A, B := mqDense(n, n, s.A), mqDense(n, m, s.B)
		res, err := Dare(A, B, mqDense(n, n, q), mqDense(m, m, r), nil)
		if err != nil {
			t.Errorf("kind %#x: %v", tc.kind, err)
			continue
		}
		var acl mat.Dense
		acl.Mul(B, res.K)
		acl.Sub(A, &acl)
		var eig mat.Eigen
		if !eig.Factorize(&acl, mat.EigenNone) {
			t.Fatalf("kind %#x: eig(A − BK) failed", tc.kind)
		}
		vals := eig.Values(nil)
		tol := 1e-7 * (1 + mat.Norm(&acl, 2))
		used := make([]bool, len(vals))
		for _, e := range res.Eig {
			best, bi := math.Inf(1), -1
			for i, v := range vals {
				if d := cmplx.Abs(e - v); !used[i] && d < best {
					best, bi = d, i
				}
			}
			if !(best <= tol) {
				t.Errorf("kind %#x: Eig %v is %.3g from eig(A − BK) %v", tc.kind, e, best, vals)
				continue
			}
			used[bi] = true
		}
	}
}
