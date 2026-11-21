package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDLyap_NonNormalComplexBlock: A has the well-separated complex pair
// 0.9 ± 0.1i (|λ|² = 0.82) in a 2×2 block with |t12/t21| = 1e10. The 3×3
// system of the block's Stein equation was so badly scaled that Dgetc2's
// perturbed-pivot test fired and DLyap returned ErrSingularEquation. The
// oracle is the 320-bit Kronecker solve of (I − A⊗A)·vec X = vec Q.
func TestDLyap_NonNormalComplexBlock(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0.9, 1e4, 0.3,
		-1e-6, 0.9, -0.2,
		0, 0, 0.5,
	})
	Q := mat.NewDense(3, 3, []float64{2, 0.5, 0.1, 0.5, 1, -0.3, 0.1, -0.3, 3})
	X, err := DLyap(A, Q, nil)
	if err != nil {
		t.Fatal(err)
	}
	if e := relErr(X, dlyapBig(A, Q)); !(e <= 1e-9) {
		t.Errorf("X rel err %g against 320-bit Kronecker solve", e)
	}
}

// TestDLyap_NonNormalKroneckerBlock (75DTNG): A is already in real Schur
// form, a real mode at 1 − δ coupled to the pair 1 − δ ± iε whose block
// has t₁₂ = 1e4, t₂₁ = −ε²/t₁₂. The off-diagonal Stein block of the pair
// against the real mode (and, in "pairs", against a second pair) was
// solved unbalanced, its Kronecker pivot ((2δ)² + ε²)/t₁₂ below
// eps·t₁₂, so DLyap returned ErrSingularEquation though the equation is
// well posed (separation ~2δ once balanced). The oracle is the 320-bit
// Kronecker solve.
func TestDLyap_NonNormalKroneckerBlock(t *testing.T) {
	const d, t12, eps2 = 1e-5, 1e4, 1e-10
	cases := []struct {
		name string
		n    int
		a    []float64
	}{
		{"real", 3, []float64{
			1 - d, 0.3, -0.2,
			0, 1 - d, t12,
			0, -eps2 / t12, 1 - d,
		}},
		{"pairs", 4, []float64{
			1 - d, t12, 0.3, -0.2,
			-eps2 / t12, 1 - d, 0.1, 0.4,
			0, 0, 1 - d, t12,
			0, 0, -eps2 / t12, 1 - d,
		}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			A := mat.NewDense(tc.n, tc.n, tc.a)
			Q := mat.NewDense(tc.n, tc.n, nil)
			for i := range tc.n {
				for j := range tc.n {
					Q.Set(i, j, 0.1/float64(1+i+j))
				}
				Q.Set(i, i, float64(1+i))
			}
			X, err := DLyap(A, Q, nil)
			if err != nil {
				t.Fatal(err)
			}
			if e := relErr(X, dlyapBig(A, Q)); !(e <= 1e-8) {
				t.Errorf("X rel err %g against 320-bit Kronecker solve", e)
			}
		})
	}
}

// dlyapBig solves A X Aᵀ − X + Q = 0 by the 320-bit Kronecker system.
func dlyapBig(A, Q *mat.Dense) *mat.Dense {
	a, q := hbFrom(A), hbFrom(Q)
	n := a.r
	M := newHbMat(n*n, n*n)
	rhs := newHbMat(n*n, 1)
	p := newHbMat(1, 1).d[0]
	for i := range n {
		for j := range n {
			row := i*n + j
			rhs.d[row].Set(q.at(i, j))
			for k := range n {
				for l := range n {
					c := M.at(row, k*n+l)
					c.Sub(c, p.Mul(a.at(i, k), a.at(j, l)))
					if k == i && l == j {
						c.Add(c, newHbMat(1, 1).d[0].SetInt64(1))
					}
				}
			}
		}
	}
	xv := hbSolve(M, rhs)
	X := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			v, _ := xv.d[i*n+j].Float64()
			X.Set(i, j, v)
		}
	}
	return X
}
