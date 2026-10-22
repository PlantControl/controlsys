package controlsys

import (
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDare_BlockedQZSizes guards U6SBBW: from n = 65 (2n > 128, gonum's
// blocked QR inside Dgges) the Dlarft prevlastv bug of gonum before v0.20.8 broke
// the pencil: wrong X (X error 1.5 at n = 70) before the Hewer check, and
// ErrNoStabilizing at n = 69 after it. E = I skips the Hewer refinement, so
// the pencil solution itself is checked, against Hewer's iteration from
// K = 0 (A is stable).
func TestDare_BlockedQZSizes(t *testing.T) {
	skipSlowUnderRace(t)
	const m = 5
	for _, n := range []int{65, 69, 70, 130} {
		t.Run(fmt.Sprintf("n=%d", n), func(t *testing.T) {
			A, B, Q := benchDiscreteA(n), benchB(n, m), benchSymPD(n)
			R, E := eye(m), eye(n)
			ref := hewerDare(t, A, B, Q, R, nil, mat.NewDense(m, n, nil))
			for _, opts := range []*RiccatiOpts{nil, {E: E}, {E: E, NoScaling: true}} {
				res, err := Dare(A, B, Q, R, opts)
				if err != nil {
					t.Fatalf("opts %+v: %v", opts, err)
				}
				if r := dareResidual(A, B, Q, R, res.X) / mat.Norm(Q, 2); r > 1e-12 {
					t.Errorf("opts %+v: relative residual %g", opts, r)
				}
				if e := relErr(res.X, ref); e > 1e-9 {
					t.Errorf("opts %+v: X rel err %g against Hewer", opts, e)
				}
			}
		})
	}
}
