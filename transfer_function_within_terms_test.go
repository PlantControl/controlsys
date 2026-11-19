package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestTransferFunctionExplicitCancellationWithinTerms: an explicit model's
// TransferFunction carries the within-terms claim (SL352H, user decision
// 2026-11-18). The frozen model is the E⁻¹ fold of seed 199 of
// TestTransferFunctionBadlyScaledDiagonalEStaysExact: non-symmetric A whose
// rows span 1e-6..1e6, so ss→tf cancels the denominator's constant term
// (1.7e3 against coefficients near 6e11) and the TF misses the response at
// s = 0 by 2.4e-5, 1.95× the threshold. Base returned it with nil.
func TestTransferFunctionExplicitCancellationWithinTerms(t *testing.T) {
	a := mat.NewDense(6, 6, []float64{
		-2.1526395573943165e-06, -1.6098102592741983e-07, 1.024187842785151e-06, 8.554534076887711e-07, 1.5968979345070293e-06, 6.980546990031616e-07,
		-1.2186441756691468, 1.383274645855612, -0.1781166681523655, -1.6584758680694145, -1.514016320795996, -1.3835385171315928,
		0.7454617967349538, 0.8249511188117254, -1.5331568453433537, -0.3354773247379212, -0.11770012767961613, 0.9352781717674695,
		-0.03178332463999581, 0.06849066153601797, -0.02386892811350999, -0.05774464709923803, 0.012611823850270121, -0.038910639857539196,
		636487.1613737826, 307727.44956641813, -305997.62464879156, -51128.43174224292, 557620.2286870152, -32646.9990874342,
		33272.12074610738, 221871.68802671094, 153573.13904404672, 140149.38067890162, 210763.12867068683, -47918.109564502454,
	})
	b := mat.NewDense(6, 1, []float64{8.414884089738574e-07, 3.615669355867621, -1.706035551377568, 0.009281780791814468, 694956.8244158281, -34223.756628359646})
	c := mat.NewDense(1, 6, []float64{-1.5472765647609799, -0.8102724218481375, 0.9620129068061639, 1.3209050785395942, -0.13558472630240903, -1.2751911458719805})
	d := mat.NewDense(1, 1, []float64{-0.4739316006621122})
	sys, err := New(a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if res == nil {
		t.Fatalf("no value: %v", err)
	}
	w, at := withinTermsWorst(t, identityPencil(sys), res.TF, math.Inf(1))
	switch {
	case err == nil && w > 1:
		t.Fatalf("nil TF outside terms at %v: %.3g× the threshold", at, w)
	case err != nil && !errors.Is(err, ErrAccuracyUnverified):
		t.Fatal(err)
	}
}

// TestTransferFunctionFoldUnderflowingEinvBUnverified: the E⁻¹ fold of a
// descriptor at 2^±600 underflows E⁻¹B to 0, so its TF is ≡ 0 against a
// nonzero response; base returned it with nil (SL352H; frozen ovf/14 of
// the 5CFETF census). The fold's residual E·B̄ − B = −B is no rounding of
// B, the pencil certificate refutes it, and it comes back unverified.
func TestTransferFunctionFoldUnderflowingEinvBUnverified(t *testing.T) {
	a := mat.NewDense(2, 2, []float64{-1.3076812206422687e+179, 1.217251857475639e+178, 1.1162048192073216e+179, -8.790741366365397e+178})
	b := mat.NewDense(2, 2, []float64{-7.025645444127569e-166, 1.5071368202610764e-166, -2.169155599180122e-166, 1.6285480104683778e-166})
	c := mat.NewDense(1, 2, []float64{-1.212546455738041e+293, -1.5385790206657246e+294})
	e := mat.NewDense(2, 2, []float64{1.3338663855694357e+179, -3.1383411901755512e+177, -1.1312463736812494e+177, 2.027122678256875e+179})
	sys, err := NewDescriptor(a, b, c, mat.NewDense(1, 2, nil), e, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if res == nil {
		t.Fatalf("no value: %v", err)
	}
	w, at := withinTermsWorst(t, sys, res.TF, math.Inf(1))
	switch {
	case err == nil && w > 1:
		t.Fatalf("nil TF outside terms at %v: %.3g× the threshold", at, w)
	case err != nil && !errors.Is(err, ErrAccuracyUnverified):
		t.Fatal(err)
	}
}
