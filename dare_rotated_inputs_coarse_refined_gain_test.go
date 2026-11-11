package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// dareOracleCase is a descriptor Dare with its 320-bit stabilizing X and
// K (doubling on the explicit model).
type dareOracleCase struct {
	name          string
	A, B, E, Q, R *mat.Dense
	wantX, wantK  *mat.Dense
}

// checkDareOracleCase fails on a Dare or Dlqr K, or Dare X, certified at
// Accuracy 0 beyond newtonTol (X) or gainFloor (K) of the 320-bit
// solution, or on an error.
func checkDareOracleCase(t *testing.T, c dareOracleCase) {
	t.Helper()
	for _, api := range []string{"Dare", "Dlqr"} {
		solve := Dare
		if api == "Dlqr" {
			solve = Dlqr
		}
		res, err := acceptUnverified(solve(c.A, c.B, c.Q, c.R, &RiccatiOpts{E: c.E}))
		if err != nil {
			t.Errorf("%s %s: %v", c.name, api, err)
			continue
		}
		eX, eK := relErr(res.X, c.wantX), relErr(res.K, c.wantK)
		if api == "Dlqr" {
			eX = 0
		}
		t.Logf("%s %s: Accuracy %d, X err %.3g, K err %.3g", c.name, api, res.accuracy, eX, eK)
		if riccatiCertifiedWrong(res.accuracy, eX, 0, eK, 0, gainFloor) {
			t.Errorf("%s %s: certified-wrong: Accuracy 0 with X off by %.3g, K by %.3g", c.name, api, eX, eK)
		}
	}
}

// TestDareRotatedInputsCoarseRefinedGainTakesHewer pins a descriptor Dare
// with rotated inputs (zrw "reg" sweep seed 28010, n 2, m 2, generated
// on each arch; NFET2G) whose refined K was 1.8e-7 from the gain of its X
// at a resolution of 6e-5, beyond gainFloor, and refinement converged in
// X: kept at Accuracy 0, K was 1.9e-6 off the 320-bit gain natively and
// 1.4e-6 on amd64 with X 2e-11 off, the Hewer gain exact. K must be
// within gainFloor at Accuracy 0, else Accuracy 1.
func TestDareRotatedInputsCoarseRefinedGainTakesHewer(t *testing.T) {
	for _, c := range []dareOracleCase{
		{
			name:  "zrw reg 28010 arm64",
			A:     mat.NewDense(2, 2, []float64{-1.9035698840258437, -1.8210127778056342, -1.4772570999201942, 1.5733384804061752}),
			B:     mat.NewDense(2, 2, []float64{-0.9612638423560205, 1.4924269932054508, -0.373609744244537, 0.5800518107761155}),
			E:     mat.NewDense(2, 2, []float64{1.0439388404453342, -0.5929821668374519, 0.01484132766615123, 0.9787358365977411}),
			Q:     mat.NewDense(2, 2, []float64{1.548931167546964, -0.5396178749950582, -0.5396178749950582, 0.5626743753616846}),
			R:     mat.NewDense(2, 2, []float64{0.9752581214041519, 0.43915757748138134, 0.43915757748138134, 1.3269521055325098}),
			wantX: mat.NewDense(2, 2, []float64{1.6285952458478687e+11, -4.190260480635617e+11, -4.190260480635617e+11, 1.0781244106375276e+12}),
			wantK: mat.NewDense(2, 2, []float64{223666.56330382306, -691887.9264425898, 64207.92759059273, -198623.9445927277}),
		},
		{
			name:  "zrw reg 28010 amd64",
			A:     mat.NewDense(2, 2, []float64{-1.9035698840258437, -1.8210127778056342, -1.4772570999201942, 1.573338480406175}),
			B:     mat.NewDense(2, 2, []float64{-0.9612638423560205, 1.4924269932054508, -0.373609744244537, 0.5800518107761155}),
			E:     mat.NewDense(2, 2, []float64{1.0439388404453342, -0.592982166837452, 0.014841327666151205, 0.9787358365977411}),
			Q:     mat.NewDense(2, 2, []float64{1.548931167546964, -0.539617874995058, -0.539617874995058, 0.5626743753616845}),
			R:     mat.NewDense(2, 2, []float64{0.9752581214041519, 0.4391575774813814, 0.4391575774813814, 1.3269521055325098}),
			wantX: mat.NewDense(2, 2, []float64{1.6285952458696155e+11, -4.19026048069157e+11, -4.19026048069157e+11, 1.0781244106519238e+12}),
			wantK: mat.NewDense(2, 2, []float64{223666.56330609854, -691887.9264496288, 64207.92759214168, -198623.9445975192}),
		},
	} {
		checkDareOracleCase(t, c)
	}
}
