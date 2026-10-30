package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// dareRotatedCoarseResolutionCase is zw46Gen("reg", 323, true) (wave-55
// zw46 sweep, 3RYNDV): n 3, m 3, S given, cond(E) 4.1.
var dareRotatedCoarseResolutionCase = dareRotatedInputsCase{
	name: "zw46 reg 323 cross",
	A:    mat.NewDense(3, 3, []float64{0.18911509287905978, -0.84815825847446047, -1.5727806734736527, -0.32337365661218853, 0.51867563725962529, 0.8372646959261848, 0.049994863520804678, 0.35816533240616066, -1.0118633748377237}),
	B:    mat.NewDense(3, 3, []float64{0.36198606787216142, -2.7361194165997684, -1.3916668565232166, 0.034370123326956631, 0.2788676042237796, 0.22269664537698045, -0.011247940005177839, 0.65030635866559383, 0.41562273760149082}),
	Q:    mat.NewDense(3, 3, []float64{2.5706863114504426, -2.2212623951184218, 1.2120276366914928, -2.2212623951184218, 2.1235744176969451, -1.5436794847643021, 1.2120276366914928, -1.5436794847643021, 6.0839713063803629}),
	R:    mat.NewDense(3, 3, []float64{5.8224599303140838, -1.165449286466228, -2.2514798850578148, -1.165449286466228, 6.6779886283905512, 3.8010145526973202, -2.2514798850578148, 3.8010145526973202, 7.9881844980849257}),
	S:    mat.NewDense(3, 3, []float64{-1.8497067790748263, 0.95180345160823965, -0.075448977267828649, 1.1643794957516236, -0.59915465282155644, 0.047494685698254768, -0.506693237767465, 0.26072909396744887, -0.020667863150288786}),
	E:    mat.NewDense(3, 3, []float64{1.1805871623026634, 0.37393987465133616, 0.43494494007720524, -0.64583843231843019, 0.63419626732436896, -0.68512347784522198, -1.044489012728296, 0.76905296592106176, 0.29911454101549767}),
}

// TestDareRotatedInputsCoarseGainResolutionNotCertifiedUnrefined: rotated
// inputs whose gain of X resolves K only to 8.7e-6, beyond gainFloor,
// while the pencil K is within gainFloor of it (2.6e-7). Kept unrefined
// at Accuracy 0, K was 1.25e-6 off the exact-E 320-bit gain (band
// max(gainFloor, 4·1.2e-10); 3RYNDV) natively, 9.2e-7 on amd64, where
// ddNewtonGain was 5.2e-14 off and the Hewer gain exact. Dare and Dlqr
// must return K within band at Accuracy 0, else Accuracy 1, with
// E⁻¹(A − BK) stable.
func TestDareRotatedInputsCoarseGainResolutionNotCertifiedUnrefined(t *testing.T) {
	c := dareRotatedCoarseResolutionCase
	ref := riccatiExactReference(false, c.A, c.B, c.Q, c.R, c.S, c.E, riccatiExactPrec)
	if !ref.OK {
		t.Fatalf("%s: exact-E oracle did not certify", c.name)
	}
	const kBand = 1.2e-10
	for _, ns := range []bool{false, true} {
		opts := &RiccatiOpts{E: c.E, S: c.S, NoScaling: ns}
		red, _ := reduceInputs(c.A, c.B, c.Q, c.R, opts)
		if red == nil || !red.rotated {
			t.Fatalf("%s NoScaling %v: inputs not rotated", c.name, ns)
		}
		if p := red.solve(c.A, c.B, c.Q, opts); p != nil {
			d := gainMissIn(c.A, c.B, c.R, opts, red, p)
			t.Logf("%s NoScaling %v: pencil K err %.3g, gain miss %.3g, resolution %.3g", c.name, ns, relErr(p.K, ref.K), d.d, d.tol)
		}
		for _, api := range []string{"Dare", "Dlqr"} {
			solve := Dare
			if api == "Dlqr" {
				solve = Dlqr
			}
			res, err := solve(c.A, c.B, c.Q, c.R, opts)
			if err != nil {
				t.Fatalf("%s %s NoScaling %v: %v", c.name, api, ns, err)
			}
			eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
			if api == "Dlqr" {
				eX = relErr(res.X, ref.Y)
			}
			t.Logf("%s %s NoScaling %v: Accuracy %d, X err %.3g, K err %.3g", c.name, api, ns, res.Accuracy, eX, eK)
			if riccatiCertifiedWrong(res.Accuracy, eX, 0, eK, kBand, gainFloor) {
				t.Errorf("%s %s NoScaling %v: certified-wrong: Accuracy 0 with X off by %.3g, K by %.3g", c.name, api, ns, eX, eK)
			}
			if dareRotatedInputsUnstable(c.A, c.B, c.E, res.K) {
				t.Errorf("%s %s NoScaling %v: E⁻¹(A − BK) unstable in %d bits", c.name, api, ns, riccatiExactPrec)
			}
		}
	}
}
