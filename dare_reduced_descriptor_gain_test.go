package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDare_ReducedInputsNearSingularERefinedGainNotCertifiedWrong: with
// rank-deficient B (sv 0.505, 2.2e-11; reduced, not rotated inputs), a
// cross term, cond(E) 1.9e12 and a mode A00/E00 = -1.000000001 beside
// z = -1, the double-double refined K was exact while the gain of the
// rounded X it was measured against was 1.67e-6 off; Dare and Dlqr then
// returned the ddGain of X one Newton step on, 1.42 off the 320-bit gain
// (band 3.9e-7), at Accuracy 0, as gainMiss measured that gain against
// itself (UVLXIF, rv46Gen bnd 170). K stays within its band at any
// Accuracy.
func TestDare_ReducedInputsNearSingularERefinedGainNotCertifiedWrong(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-0.9872637343184305, 0, -0.6278046315935414, -0.19676804875153303})
	B := mat.NewDense(2, 3, []float64{-2.6334484791438702e-11, -5.1241294161283313e-11, 8.280981856583433e-11, -0.03342141858635045, -0.3065818785752709, 0.39934679132305667})
	Q := mat.NewDense(2, 2, []float64{1.6864118423120826, 0.0027189509686436075, 0.0027189509686436075, 0.5357480026609325})
	R := mat.NewDense(3, 3, []float64{9.42759067245151, -0.8098928281836238, 3.2455209595569476, -0.8098928281836238, 3.7853903401138664, 0.035383414548192195, 3.2455209595569476, 0.035383414548192195, 2.2016380836235556})
	S := mat.NewDense(2, 3, []float64{0.20914529128088966, 0.27954023252164134, -0.006142349002590308, 0.19269752140896834, 0.25755640784980444, -0.005659297520776421})
	E := mat.NewDense(2, 2, []float64{0.9872637333311667, 0, 0, 5.06877910221949e-13})
	ref := riccatiExactReference(false, A, B, Q, R, S, E, riccatiExactPrec)
	sens, ok := riccatiExactSensitivity(false, A, B, Q, R, S, E, ref, "ABQRSE")
	if !ref.OK || !ok {
		t.Fatal("reference not certified")
	}
	for _, ns := range []bool{false, true} {
		for _, c := range []struct {
			name  string
			solve func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error)
			x     *mat.Dense
			xBand float64
		}{{"Dare", Dare, ref.X, sens.X}, {"Dlqr", Dlqr, ref.Y, sens.Y}} {
			res, err := c.solve(A, B, Q, R, &RiccatiOpts{S: S, E: E, NoScaling: ns})
			if err != nil {
				t.Errorf("%s NoScaling %v: %v", c.name, ns, err)
				continue
			}
			eX, eK := relErr(res.X, c.x), relErr(res.K, ref.K)
			t.Logf("%s NoScaling %v: Accuracy %d, X err %.3g (band %.3g), K err %.3g (band %.3g)", c.name, ns, res.Accuracy, eX, c.xBand, eK, sens.KBand())
			if riccatiCertifiedWrong(res.Accuracy, eX, c.xBand, eK, sens.KBand(), gainFloor) {
				t.Errorf("%s NoScaling %v: certified-wrong: X err %g, K err %g", c.name, ns, eX, eK)
			}
			if eK > max(gainFloor, 4*sens.KBand()) {
				t.Errorf("%s NoScaling %v: K err %g outside its band", c.name, ns, eK)
			}
		}
	}
}
