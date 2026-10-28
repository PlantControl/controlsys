package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Impulse of a full-rank E with LU cond(E) 2.2e4 that equilibration brings
// under the stiff routing gate, poles -7.66 and -3.03e5, follows the exact
// pencil response, as does Impulse of its fold E⁻¹A, both also with an
// off-grid input delay. The step form (A, A·B, C, C·B) left 7.5e-4 in the
// tail from cancelling C·B ≈ 2.4e4 (task 3QI2IZ, gen 63).
func TestImpulseMildlyIllConditionedEFollowsPencil(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-2.2062384691401213, -0.47652924174522016, -0.23113165364696203, -7.404262819239519})
	B := mat.NewDense(2, 1, []float64{-0.27935629609262136, -1.6300299154881284})
	C := mat.NewDense(1, 2, []float64{-0.78297168402742789, -0.51186958517324233})
	D := mat.NewDense(1, 1, []float64{0})
	E := mat.NewDense(2, 2, []float64{0.28987755215798522, 0.047284014476774876, 0.15866734913819372, 0.025905467357583058})
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := stiffContinuousDescriptorParts(sys); ok {
		t.Fatal("model is routed stiff; it must exercise the fold path")
	}
	modes, ok := newStepOracleModes(sys, []complex128{-7.66474, -303096})
	if !ok {
		t.Fatal("pencil oracle did not converge")
	}
	ex, err := sys.ToExplicit()
	if err != nil {
		t.Fatal(err)
	}
	const tol = 1e-9
	for _, tau := range []float64{0, 0.1234567} {
		for _, src := range []*System{sys, ex} {
			model := src.Copy()
			if tau > 0 {
				model.InputDelay = []float64{tau}
			}
			resp, err := Impulse(model, 5)
			if err != nil {
				t.Fatal(err)
			}
			if len(resp.T) != 100001 {
				t.Fatalf("descriptor %v tau %g: %d samples, want 100001", src.IsDescriptor(), tau, len(resp.T))
			}
			worst, at := 0.0, 0
			for s, ts := range resp.T {
				want := 0.0
				if ts >= tau {
					want = modes.impulse(0, 0, ts-tau)
				}
				if e := stepOracleRel(resp.Y.At(0, s), want); e > worst {
					worst, at = e, s
				}
			}
			if worst > tol {
				t.Errorf("descriptor %v tau %g: worst rel error %.3g at t %g (got %.17g), want <= %g",
					src.IsDescriptor(), tau, worst, resp.T[at], resp.Y.At(0, at), tol)
			}
		}
	}
}
