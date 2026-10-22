package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// exactBoundaryUncontrollable returns integer A = P·J·P⁻¹, B = P·B̃ and
// Q = MMᵀ + I, P unimodular, J lower block triangular with a leading
// boundary block Jb (0, 1 or [0 ω; −ω 0]) whose rows of B̃ are zero: Jb's
// modes are exactly uncontrollable (left eigenvector [w; 0] of J), so no
// stabilizing solution exists. The trailing 3×3 block is random and may
// share Jb's modes (a Jordan block across the two parts).
func exactBoundaryUncontrollable(rng *rand.Rand, jb [][]int64) (A, B, Q *mat.Dense) {
	const ns = 3
	nb := len(jb)
	n := nb + ns
	J := make([][]int64, n)
	for i := range J {
		J[i] = make([]int64, n)
	}
	for i := range nb {
		copy(J[i], jb[i])
	}
	bt := make([]int64, n)
	for i := nb; i < n; i++ {
		for k := range n {
			if k < nb {
				J[i][k] = int64(rng.IntN(5) - 2)
			} else {
				J[i][k] = int64(rng.IntN(7) - 3)
			}
		}
		if bt[i] = int64(rng.IntN(7) - 3); bt[i] == 0 {
			bt[i] = 1
		}
	}
	P, Pi := make([][]int64, n), make([][]int64, n)
	for i := range n {
		P[i], Pi[i] = make([]int64, n), make([]int64, n)
		P[i][i], Pi[i][i] = 1, 1
	}
	for range 2 * n {
		a, b := rng.IntN(n), rng.IntN(n)
		if a == b {
			continue
		}
		c := int64(rng.IntN(5) - 2)
		for k := range n {
			P[a][k] += c * P[b][k]
			Pi[k][b] -= c * Pi[k][a]
		}
	}
	A, B, Q = mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(n, n, nil)
	M := make([][]int64, n)
	for i := range M {
		M[i] = make([]int64, n)
		for k := range n {
			M[i][k] = int64(rng.IntN(7) - 3)
		}
	}
	for r := range n {
		var bs int64
		for k := range n {
			bs += P[r][k] * bt[k]
		}
		B.Set(r, 0, float64(bs))
		for c := range n {
			var s, q int64
			for k := range n {
				q += M[r][k] * M[c][k]
				for l := range n {
					s += P[r][k] * J[k][l] * Pi[l][c]
				}
			}
			if r == c {
				q++
			}
			A.Set(r, c, float64(s))
			Q.Set(r, c, float64(q))
		}
	}
	return A, B, Q
}

// TestRiccati_ExactBoundaryUncontrollable (K72KOR review): (A, B) of exact
// integers with an exactly uncontrollable mode at 0 or ±iω (Care) or 1
// (Dare) in a dense realization has no stabilizing solution (MATLAB
// icare/idare Report 3). Its computed mode is off by eps·‖A‖·κ, which a
// band of n·eps·|λ| missed (Accuracy 1 X on about a quarter of these, its
// exact closed loop not stable); the componentwise eigenvalue error band
// and the Jordan cluster mean catch them. The two review repros are named.
func TestRiccati_ExactBoundaryUncontrollable(t *testing.T) {
	type tcase struct {
		name       string
		continuous bool
		A, B, Q    *mat.Dense
		r          float64
	}
	cases := []tcase{
		{"care0", true,
			mat.NewDense(4, 4, []float64{-1, 2, -1, 2, -6, 8, -4, 8, -6, 9, -5, 8, 1, -1, 0, -2}),
			mat.NewDense(4, 1, []float64{2, 3, 4, -1}),
			mat.NewDense(4, 4, []float64{13, 4, -7, -6, 4, 6, -2, -3, -7, -2, 4, 3, -6, -3, 3, 5}), 1e-11},
		{"dare1", false,
			mat.NewDense(5, 5, []float64{3, 8, 18, -8, 4, 8, 12, 38, -16, -4, 3, 4, 14, -6, -2, 15, 23, 73, -31, -6, -1, 4, -2, -2, 2}),
			mat.NewDense(5, 1, []float64{4, 5, 1, 9, 1}),
			mat.NewDense(5, 5, []float64{14, 1, -3, 11, 4, 1, 9, 10, 4, 6, -3, 10, 17, 3, 0, 11, 4, 3, 14, 2, 4, 6, 0, 2, 13}), 1e-7},
	}
	kinds := []struct {
		name       string
		continuous bool
	}{{"zero", true}, {"one", false}, {"imag", true}}
	for kind, k := range kinds {
		for seed := range 50 {
			rng := rand.New(rand.NewPCG(uint64(seed), uint64(kind)))
			jb := [][]int64{{0}}
			switch kind {
			case 1:
				jb = [][]int64{{1}}
			case 2:
				w := int64(1 + rng.IntN(4))
				jb = [][]int64{{0, w}, {-w, 0}}
			}
			A, B, Q := exactBoundaryUncontrollable(rng, jb)
			cases = append(cases, tcase{fmt.Sprintf("%s%d", k.name, seed), k.continuous, A, B, Q, math.Pow(10, -12+12*rng.Float64())})
		}
	}
	for _, tc := range cases {
		for _, ns := range []bool{false, true} {
			R := mat.NewDense(1, 1, []float64{tc.r})
			opts := &RiccatiOpts{NoScaling: ns}
			var res *RiccatiResult
			var err error
			if tc.continuous {
				res, err = Care(tc.A, tc.B, tc.Q, R, opts)
			} else {
				res, err = Dare(tc.A, tc.B, tc.Q, R, opts)
			}
			if err == nil {
				t.Errorf("%s NoScaling %v: Accuracy %d X, want ErrNoStabilizing", tc.name, ns, res.Accuracy)
			} else if !errors.Is(err, ErrNoStabilizing) && !errors.Is(err, ErrSchurFailed) {
				t.Errorf("%s NoScaling %v: %v", tc.name, ns, err)
			}
		}
	}
}
