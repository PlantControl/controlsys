package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDareNewtonStepBoundsOneNormError pins a cheap-control Dare (R 8e-12,
// Q ~1e13, rv39 sweep seed 4005) whose first Newton step, 9.37e-9 in the
// Frobenius norm, was X's error in that norm; its 1-norm error was
// 1.01e-8, certified at Accuracy 0 on both arches (RBZR52). A step within
// newtonTol must bound the 1-norm error too: certified, X is within
// newtonTol of the 320-bit X in both norms.
func TestDareNewtonStepBoundsOneNormError(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.24906063957311242, 0.81941012112809408, -1.0616883071258587, 0.045838555698169407, -1.4216129925197165, -1.2736408520637319, 1.4476545227729678, 1.1285393745838641, -0.53671293322296032})
	B := mat.NewDense(3, 1, []float64{-0.22953966783489399, -0.077892151944638055, 0.75322833766504305})
	Q := mat.NewDense(3, 3, []float64{13388432603586.211, -3397686320779.3979, -4187029235569.4956, -3397686320779.3979, 2587200787602.0396, 1222371667741.2676, -4187029235569.4956, 1222371667741.2676, 4042706014924.833})
	R := mat.NewDense(1, 1, []float64{8.2084357346462581e-12})
	ref := riccatiExactReference(false, A, B, Q, R, nil, nil, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("320-bit reference failed")
	}
	for _, ns := range []bool{false, true} {
		res, err := Dare(A, B, Q, R, &RiccatiOpts{NoScaling: ns})
		if err != nil {
			t.Fatalf("NoScaling=%v: %v", ns, err)
		}
		var d mat.Dense
		d.Sub(res.X, ref.X)
		e1, eF := relErr(res.X, ref.X), mat.Norm(&d, 2)/mat.Norm(ref.X, 2)
		if e1 > newtonTol || eF > newtonTol {
			t.Errorf("NoScaling=%v: certified X %.3g (1-norm), %.3g (Frobenius) off the 320-bit X", ns, e1, eF)
		}
		if k := relErr(res.K, ref.K); k > newtonTol {
			t.Errorf("NoScaling=%v: certified K %.3g off the 320-bit K", ns, k)
		}
	}
}
