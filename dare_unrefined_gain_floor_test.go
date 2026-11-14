package controlsys

import (
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestDareUnrefinedGainWithinFloorTakesHewerGain: an unrefined pencil K of
// the given inputs within gainFloor of the gain of its X, refinement
// missing gainTol, was certified as it was (GFPGEU; zw46 reg seed 1515
// with cross term and E, Q·(1 − 38·2⁻⁵²), generated per arch): K 1.04e-6
// to 1.46e-6 off the 320-bit gain at Accuracy 0 while the Hewer gain of
// its X was exact. On the generating arch Dare and Dlqr return Accuracy 0
// with the exact K; elsewhere neither certifies a K beyond its band,
// max(gainFloor, 4× its move under eps perturbations of the data, the band
// of the sweeps that found it). KBand, which adds the gain's
// move under rounding of Y or X (KofY 7.5e-6), would not tell the pencil
// K from the exact one. Under gonum 30740af (HF5PTS) the race build of the
// generating arch exited at the gainTol accept with gainMissIn 5.0e-8 for
// a K 4.49e-7 off; gainResolved now withholds that accept (BBN22I).
func TestDareUnrefinedGainWithinFloorTakesHewerGain(t *testing.T) {
	for _, tc := range []struct {
		name, arch       string
		A, B, Q, R, S, E []float64
	}{
		{
			name: "arm64 draw", arch: "arm64",
			A: []float64{-0.36349575993785965, 0.029113424431950728, 0.24440218644099648, -1.685321490796345, -1.2494714930894988, -1.922130108127909, 0.22152146737969602, 0.08737613848261773, -0.07666773220635387},
			B: []float64{-0.807984248974277, -0.1292819253667961, -0.803666193680655},
			Q: []float64{1.2155183294782004, 0.6594080757133947, 0.5714191130691423, 0.6594080757133947, 3.0175500217024647, 1.7483356459148647, 0.5714191130691423, 1.7483356459148647, 2.515044235779199},
			R: []float64{3.5035434746171528},
			S: []float64{-0.7345471444540268, -2.247448373499815, -1.9475572161662924},
			E: []float64{0.6913104660660118, 0.26142706333163596, -0.17898212615796477, -1.0173956770967822, 0.2664449125821802, -0.5809858174512162, 1.559594885770189, 1.1776100260883529, 0.3207239387185603},
		},
		{
			name: "amd64 draw", arch: "amd64",
			A: []float64{-0.3634957599378593, 0.029113424431950904, 0.2444021864409967, -1.685321490796345, -1.2494714930894988, -1.9221301081279087, 0.22152146737969614, 0.08737613848261777, -0.07666773220635374},
			B: []float64{-0.8079842489742768, -0.12928192536679597, -0.8036661936806547},
			Q: []float64{1.2155183294782004, 0.6594080757133947, 0.5714191130691423, 0.6594080757133947, 3.0175500217024647, 1.7483356459148647, 0.5714191130691423, 1.7483356459148647, 2.515044235779199},
			R: []float64{3.5035434746171528},
			S: []float64{-0.7345471444540268, -2.247448373499815, -1.9475572161662924},
			E: []float64{0.6913104660660119, 0.2614270633316359, -0.1789821261579646, -1.0173956770967822, 0.26644491258218006, -0.5809858174512161, 1.5595948857701885, 1.1776100260883526, 0.3207239387185603},
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			A, B := mat.NewDense(3, 3, tc.A), mat.NewDense(3, 1, tc.B)
			Q, R := mat.NewDense(3, 3, tc.Q), mat.NewDense(1, 1, tc.R)
			S, E := mat.NewDense(3, 1, tc.S), mat.NewDense(3, 3, tc.E)
			ref := riccatiExactReference(false, A, B, Q, R, S, E, riccatiExactPrec)
			sens, ok := riccatiExactSensitivity(false, A, B, Q, R, S, E, ref, "ABQRSE")
			if !ref.OK || !ok {
				t.Fatal("reference not certified")
			}
			kTol := max(gainFloor, 4*sens.K)
			native := runtime.GOARCH == tc.arch
			for _, solve := range []struct {
				name string
				f    func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error)
			}{{"Dare", Dare}, {"Dlqr", Dlqr}} {
				res, err := acceptUnverified(solve.f(A, B, Q, R, &RiccatiOpts{S: S, E: E}))
				if err != nil {
					if native {
						t.Errorf("%s: %v", solve.name, err)
					} else {
						t.Logf("%s on %s: %v", solve.name, runtime.GOARCH, err)
					}
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
