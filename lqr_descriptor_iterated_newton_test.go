package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestLqrDescriptorIteratedNewtonGain: with a triangular E of last pivot
// 6.4e-12 and cheap control, the first Newton step of X exceeded
// newtonTol and the certified X was an iterate whose gain was then
// unstable, so the previous one was kept with no Newton step to bound
// EᵀXE and K: Lqr's EᵀXE was off the 320-bit oracle by 3.5e-4 and K by
// 1e-7, Care with S's K by 5e-8, at Accuracy 0, against eps-sensitivities
// of 2e-10 and 6e-14 (r2b seed 686, 5AFLTD). The step of the kept
// iterate now triggers the double-double refinement.
func TestLqrDescriptorIteratedNewtonGain(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-0.753551731888285, 0.8925697625035272, 0.32725113980572257, 0.2629316660170884})
	B := mat.NewDense(2, 2, []float64{-1.0265499495660801, 2.3069521901486887, -0.47228660560251035, 1.3783625520928942})
	E := mat.NewDense(2, 2, []float64{1.1172280692512373, 1.8101667016318157, 0, -6.4062077733861815e-12})
	Q := mat.NewDense(2, 2, []float64{235.7757109058383, -251.60823096115024, -251.60823096115024, 268.5041523609717})
	R := mat.NewDense(2, 2, []float64{7.388081871008216e-06, -1.1674576044424052e-07, -1.1674576044424052e-07, 7.388081871008216e-06})
	S := mat.NewDense(2, 2, []float64{0.002685903059931384, 0.0164770942666965, -0.0028662648514393486, -0.017583552010879907})
	for _, ns := range []bool{false, true} {
		label := "Lqr"
		if ns {
			label += " NoScaling"
		}
		res, err := Lqr(A, B, Q, R, &RiccatiOpts{E: E, NoScaling: ns})
		if err != nil {
			t.Fatalf("%s: %v", label, err)
		}
		Y, _, Ko, ok := nearSingularOracle(true, A, B, Q, R, nil, E, res.K)
		if !ok {
			t.Fatalf("%s: oracle did not converge", label)
		}
		if res.Accuracy == 0 {
			if e := relErr(res.X, Y); e > descriptorGainTol {
				t.Errorf("%s: Accuracy 0, EᵀXE off the oracle by %.3g", label, e)
			}
			if e := relErr(res.K, Ko); e > descriptorGainTol {
				t.Errorf("%s: Accuracy 0, K off the oracle by %.3g", label, e)
			}
		}
		res, err = Care(A, B, Q, R, &RiccatiOpts{E: E, S: S, NoScaling: ns})
		if err != nil {
			t.Fatalf("Care with S %v: %v", ns, err)
		}
		_, _, Ko, ok = nearSingularOracle(true, A, B, Q, R, S, E, res.K)
		if !ok {
			t.Fatalf("Care with S %v: oracle did not converge", ns)
		}
		if e := relErr(res.K, Ko); res.Accuracy == 0 && e > descriptorGainTol {
			t.Errorf("Care with S NoScaling=%v: Accuracy 0, K off the oracle by %.3g", ns, e)
		}
	}
}
