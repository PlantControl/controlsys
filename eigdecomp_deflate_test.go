package controlsys

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDeflateInfinite_Structure: deflateInfinite returns orthogonal Q, Z
// with Qᵀ(A, E)Z = (a, e), zero lower-left blocks, nf = r finite modes and
// a nilpotent trailing pencil: (A∞⁻¹N)^d = 0 for the d infinite modes.
func TestDeflateInfinite_Structure(t *testing.T) {
	for _, chains := range append(higherIndexChains, []int{1}, []int{1, 1, 1}) {
		for _, orth := range []bool{true, false} {
			for seed := range uint64(20) {
				r := 2 + int(seed%5)
				w := newWeierstrassChains(seed, r, chains, orth)
				label := fmt.Sprintf("chains %v orth %v seed %d", chains, orth, seed)
				A, E := w.sys.A, w.sys.E
				n, _ := A.Dims()
				a, e, q, z, nf, ok := deflateInfinite(A, E, true)
				if !ok || nf != r {
					t.Fatalf("%s: nf %d ok %v, want %d", label, nf, ok, r)
				}
				scale := mat.Norm(A, 2) + mat.Norm(E, 2)
				for _, c := range []struct {
					m, got *mat.Dense
				}{{A, a}, {E, e}} {
					var qm, qmz, diff mat.Dense
					qm.Mul(q.T(), c.m)
					qmz.Mul(&qm, z)
					diff.Sub(&qmz, c.got)
					if d := mat.Norm(&diff, 2); d > 1e-12*scale {
						t.Fatalf("%s: ‖QᵀMZ − M̃‖ = %g", label, d)
					}
					for i := nf; i < n; i++ {
						for j := range nf {
							if c.got.At(i, j) != 0 {
								t.Fatalf("%s: lower-left (%d,%d) = %g", label, i, j, c.got.At(i, j))
							}
						}
					}
				}
				for _, m := range []*mat.Dense{q, z} {
					var g mat.Dense
					g.Mul(m.T(), m)
					for i := range n {
						g.Set(i, i, g.At(i, i)-1)
					}
					if d := mat.Norm(&g, 2); d > 1e-13*float64(n) {
						t.Fatalf("%s: ‖QᵀQ − I‖ = %g", label, d)
					}
				}
				d := n - nf
				var nilp mat.Dense
				if err := nilp.Solve(a.Slice(nf, n, nf, n), e.Slice(nf, n, nf, n)); err != nil {
					t.Fatalf("%s: A∞ singular: %v", label, err)
				}
				p := mat.DenseCopyOf(&nilp)
				for range d - 1 {
					p.Mul(p, &nilp)
				}
				if v := mat.Norm(p, 2); v > 1e-8 {
					t.Fatalf("%s: ‖(A∞⁻¹N)^%d‖ = %g", label, d, v)
				}
			}
		}
	}
}
