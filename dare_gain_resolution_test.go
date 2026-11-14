package controlsys

import (
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDareGainUnresolvedByRoundedXTakesHewerGain: zw46 reg seed 1515
// with cross term and E, Q or A scaled by 1 + k·2⁻⁵² (BBN22I; arm64
// draws native, amd64 draws under Rosetta). ‖X‖ is 2e11 and the gain of
// the 320-bit X rounded to float64 was up to 9.4e-7 off the 320-bit gain,
// while gainMissOf resolved the gain of X to 7.9e-8: the pencil K, 3e-8
// from the gain of its X, was accepted at gainOfX's gainTol accept with
// K 1.03e-6 to 1.19e-6 off at Accuracy 0. The bound of gainResolved
// (9.5e-6) withholds that accept, and the Hewer gain is exact.
func TestDareGainUnresolvedByRoundedXTakesHewerGain(t *testing.T) {
	armA := []float64{-0.36349575993785965, 0.029113424431950728, 0.24440218644099648, -1.685321490796345, -1.2494714930894988, -1.922130108127909, 0.22152146737969602, 0.08737613848261773, -0.07666773220635387}
	armB := []float64{-0.807984248974277, -0.1292819253667961, -0.803666193680655}
	armE := []float64{0.6913104660660118, 0.26142706333163596, -0.17898212615796477, -1.0173956770967822, 0.2664449125821802, -0.5809858174512162, 1.559594885770189, 1.1776100260883529, 0.3207239387185603}
	amdA := []float64{-0.3634957599378593, 0.029113424431950904, 0.2444021864409967, -1.685321490796345, -1.2494714930894988, -1.9221301081279087, 0.22152146737969614, 0.08737613848261777, -0.07666773220635374}
	amdB := []float64{-0.8079842489742768, -0.12928192536679597, -0.8036661936806547}
	amdE := []float64{0.6913104660660119, 0.2614270633316359, -0.1789821261579646, -1.0173956770967822, 0.26644491258218006, -0.5809858174512161, 1.5595948857701885, 1.1776100260883526, 0.3207239387185603}
	q := []float64{1.2155183294782106, 0.6594080757134002, 0.571419113069147, 0.6594080757134002, 3.01755002170249, 1.7483356459148793, 0.571419113069147, 1.7483356459148793, 2.51504423577922}
	r := []float64{3.5035434746171528}
	s := []float64{-0.7345471444540268, -2.247448373499815, -1.9475572161662924}
	for _, tc := range []struct {
		name, arch string
		A, B, Q, E []float64
	}{
		{"arm64 Q-29", "arm64", armA, armB, []float64{1.2155183294782028, 0.659408075713396, 0.5714191130691434, 0.659408075713396, 3.0175500217024704, 1.748335645914868, 0.5714191130691434, 1.748335645914868, 2.515044235779204}, armE},
		{"arm64 Q+220", "arm64", armA, armB, []float64{1.2155183294782699, 0.6594080757134324, 0.5714191130691749, 0.6594080757134324, 3.0175500217026374, 1.7483356459149648, 0.5714191130691749, 1.7483356459149648, 2.515044235779343}, armE},
		{"amd64 Q-232", "amd64", amdA, amdB, []float64{1.215518329478148, 0.6594080757133662, 0.5714191130691176, 0.6594080757133662, 3.0175500217023346, 1.7483356459147892, 0.5714191130691176, 1.7483356459147892, 2.5150442357790905}, amdE},
		{"amd64 A-138", "amd64", []float64{-0.36349575993784816, 0.029113424431950013, 0.2444021864409892, -1.6853214907962932, -1.2494714930894606, -1.92213010812785, 0.22152146737968934, 0.08737613848261509, -0.0766677322063514}, amdB, q, amdE},
	} {
		t.Run(tc.name, func(t *testing.T) {
			A, B := mat.NewDense(3, 3, tc.A), mat.NewDense(3, 1, tc.B)
			Q, R := mat.NewDense(3, 3, tc.Q), mat.NewDense(1, 1, r)
			S, E := mat.NewDense(3, 1, s), mat.NewDense(3, 3, tc.E)
			ref := riccatiExactReference(false, A, B, Q, R, S, E, riccatiExactPrec)
			sens, ok := riccatiExactSensitivity(false, A, B, Q, R, S, E, ref, "ABQRSE")
			if !ref.OK || !ok {
				t.Fatal("reference not certified")
			}
			kTol := max(gainFloor, 4*sens.K)
			if gainResolved(A, B, R, &RiccatiResult{X: ref.X, K: ref.K}, 0) {
				t.Error("gainResolved certifies the 320-bit gain from the rounded X")
			}
			native := runtime.GOARCH == tc.arch
			for _, solve := range []struct {
				name string
				f    func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error)
			}{{"Dare", Dare}, {"Dlqr", Dlqr}} {
				res, err := acceptUnverified(solve.f(A, B, Q, R, &RiccatiOpts{S: S, E: E}))
				if err != nil {
					t.Errorf("%s: %v", solve.name, err)
					continue
				}
				kErr := relErr(res.K, ref.K)
				t.Logf("%s on %s: Accuracy %d, K err %.3g (tol %.3g)", solve.name, runtime.GOARCH, res.accuracy, kErr, kTol)
				switch {
				case native && res.accuracy != 0:
					t.Errorf("%s: Accuracy %d, want 0", solve.name, res.accuracy)
				case native && !(kErr <= 1e-12):
					t.Errorf("%s: K %.3g off the 320-bit gain, want the exact Hewer gain", solve.name, kErr)
				case res.accuracy == 0 && !(kErr <= kTol):
					t.Errorf("%s: Accuracy 0 with K %.3g off the 320-bit gain, want within %.3g", solve.name, kErr, kTol)
				}
			}
		})
	}
}

// TestGainResolvedWellConditionedGain: on a non-symmetric A with cross
// term, ‖X‖ of order one, the rounding of X moves the gain by eps-order
// only, so gainResolved certifies the exact gain and holds a K off it by
// more than gainFloor.
func TestGainResolvedWellConditionedGain(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.9, 0.4, -0.2, -0.3, 0.7, 0.5, 0.1, -0.6, 0.8})
	B := mat.NewDense(3, 2, []float64{1, 0.2, -0.4, 1, 0.3, -0.5})
	Q := mat.NewDense(3, 3, []float64{2, 0.3, 0.1, 0.3, 1.5, -0.2, 0.1, -0.2, 1})
	R := mat.NewDense(2, 2, []float64{1, 0.1, 0.1, 2})
	S := mat.NewDense(3, 2, []float64{0.1, 0, -0.2, 0.1, 0, 0.3})
	ref := riccatiExactReference(false, A, B, Q, R, S, nil, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("reference not certified")
	}
	if !gainResolved(A, B, R, &RiccatiResult{X: ref.X, K: ref.K}, 0) {
		t.Error("exact gain not resolved")
	}
	if gainResolved(A, B, R, &RiccatiResult{X: ref.X, K: ref.K}, 2*gainFloor) {
		t.Error("K 2·gainFloor off its gain resolved")
	}
}
