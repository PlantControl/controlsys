package controlsys

import (
	"errors"
	"math"
	"testing"
)

// The cut of the stiff splits moves to the widest gap > 10 within 1e±3 of
// the nominal cut, at √lo·√hi (SCPYGG ruling, frozen constants).
func TestStiffSplitGapCut(t *testing.T) {
	inf := math.Inf(1)
	for _, tc := range []struct {
		name string
		mods []float64
		thr  float64
		want float64
	}{
		{"no gap keeps nominal", []float64{1, 2, 4, 8, 16}, 4, 4},
		{"ratio exactly 10 keeps nominal", []float64{1, 10}, 3, 3},
		{"just above 10 moves", []float64{1, 10.5}, 3, math.Sqrt(10.5)},
		{"widest wins", []float64{1, 100, 1e5}, 1e3, math.Sqrt(100) * math.Sqrt(1e5)},
		{"tie takes the lowest", []float64{1, 100, 1e4}, 100, 10},
		{"unsorted input", []float64{1e4, 1, 3}, 30, math.Sqrt(3) * math.Sqrt(1e4)},
		{"hi at the lower window edge qualifies", []float64{0.0001, 0.002}, 2, math.Sqrt(0.0001) * math.Sqrt(0.002)},
		{"hi below the window is skipped", []float64{0.0001, 0.0019}, 2, 2},
		{"lo at the upper window edge qualifies", []float64{2000, 1e5}, 2, math.Sqrt(2000) * math.Sqrt(1e5)},
		{"lo above the window is skipped", []float64{2001, 1e5}, 2, 2},
		{"zero modulus is skipped", []float64{0, 1, 2}, 1, 1},
		{"zero then a gap", []float64{0, 1, 50}, 5, math.Sqrt(50)},
		{"infinite modulus is skipped", []float64{1, 2, inf}, 1.5, 1.5},
		{"NaN modulus is skipped", []float64{1, math.NaN()}, 1, 1},
		{"no moduli", nil, 7, 7},
		{"overflow-safe midpoint", []float64{1e300, 1e305}, 1e302, math.Sqrt(1e300) * math.Sqrt(1e305)},
	} {
		got := stiffCut(tc.mods, tc.thr)
		if got != tc.want || math.IsInf(got, 0) {
			t.Errorf("%s: stiffCut(%v, %g) = %g, want %g", tc.name, tc.mods, tc.thr, got, tc.want)
		}
	}
}

// ZOH and FOH from C2D, C2DMap and D2D take the parts at ζ″ ≤ 0.5; every
// other consumer at ζ″ ≤ 1; an uncertified split errors at any limit
// (SCPYGG).
func TestStiffSplitHoldHeadroomBoundary(t *testing.T) {
	for _, tc := range []struct {
		method C2DMethod
		public bool
		want   float64
	}{
		{"", true, 0.5}, {C2DMethodZOH, true, 0.5}, {C2DMethodFOH, true, 0.5},
		{C2DMethodTustin, true, 1}, {C2DMethodImpulse, true, 1}, {C2DMethodMatched, true, 1},
		{C2DMethodLeastSquares, true, 1}, {"", false, 1}, {C2DMethodZOH, false, 1}, {C2DMethodFOH, false, 1},
	} {
		if got := stiffSplitLimit(tc.method, tc.public); got != tc.want {
			t.Errorf("stiffSplitLimit(%q, %v) = %g, want %g", tc.method, tc.public, got, tc.want)
		}
	}
	above := math.Nextafter(stiffHoldZeta, 1)
	for _, tc := range []struct {
		sp    stiffSplit
		limit float64
		ok    bool
	}{
		{stiffSplit{zeta: stiffHoldZeta}, stiffHoldZeta, true},
		{stiffSplit{zeta: above}, stiffHoldZeta, false},
		{stiffSplit{zeta: above}, 1, true},
		{stiffSplit{zeta: 1}, 1, true},
		{stiffSplit{zeta: math.Nextafter(1, 2)}, 1, false},
		{stiffSplit{zeta: math.NaN()}, 1, false},
		{stiffSplit{}, stiffHoldZeta, true},
		{stiffSplit{err: errStiffInseparable()}, 1, false},
	} {
		err := tc.sp.within(tc.limit)
		if (err == nil) != tc.ok || err != nil && !errors.Is(err, ErrSchurFailed) {
			t.Errorf("within(%g) of zeta %g, err %v = %v, want ok %v", tc.limit, tc.sp.zeta, tc.sp.err, err, tc.ok)
		}
	}
	var d descriptorBudget
	d.theta = 1
	d.zS = 0.5
	if z := d.zeta(stiffHoldZeta); z != 0.5 {
		t.Errorf("zeta of a dyadic plain budget = %g, want 0.5", z)
	}
	d.zS = above
	if z := d.zeta(stiffHoldZeta); !(z > stiffHoldZeta) || !d.accepted() {
		t.Errorf("zeta %g, accepted %v; want above the hold limit and accepted", z, d.accepted())
	}
	d.failed = true
	if z := d.zeta(stiffHoldZeta); !math.IsInf(z, 1) || d.accepted() {
		t.Errorf("failed budget: zeta %g, accepted %v", z, d.accepted())
	}
}
