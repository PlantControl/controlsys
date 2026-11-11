package controlsys

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestBoundedResponseMatchesEvalFr checks the GEPP response of
// boundedResponse against EvalFr on random discrete MIMO models at the
// unit-circle probes: two evaluations of the same data, each within the
// first-order rounding bound, differ by at most twice it.
func TestBoundedResponseMatchesEvalFr(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 3))
	pts := gapProbePoints(nil, true)
	for c := range 100 {
		n, m, p := 1+rng.IntN(7), 1+rng.IntN(3), 1+rng.IntN(3)
		A, B, C, D := mat.NewDense(n, n, nil), mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil)
		for _, x := range []*mat.Dense{A, B, C, D} {
			raw := x.RawMatrix()
			for i := range raw.Data {
				raw.Data[i] = rng.NormFloat64()
			}
		}
		sys, err := New(A, B, C, D, 0.1)
		if err != nil {
			t.Fatal(err)
		}
		br := newBoundedResponse(sys, pts)
		for k, z := range pts {
			want, err := sys.EvalFr(z)
			if err != nil || !br.ok[k] {
				t.Fatalf("case %d z %v: EvalFr err %v, ok %v", c, z, err, br.ok[k])
			}
			for i := range p {
				for j := range m {
					got, bound := br.g[k*p*m+i*m+j], br.bound[k*p*m+i*m+j]
					if d := cmplx.Abs(got - want[i][j]); !(d <= 2*bound) {
						t.Errorf("case %d z %v (%d,%d): |Δ| %.3g above twice the bound %.3g", c, z, i, j, d, bound)
					}
				}
			}
		}
	}
}
