package controlsys

import (
	"math"
	"runtime"
	"testing"
)

// Pinned wave-20 review-3 cases of the rlocus track. Expected sets come
// from the exact big.Rat oracle on the float coefficients (rlpExactOracle)
// or the sweep oracles (rlswCase.oracles).

// TestRootLocus_DoublePoleBesideSimplePoleExact: an exact double real pole
// beside a simple pole 2⁻¹¹ relative apart keeps its K = 0 breakaway (base
// correct). R272: controllable companion, eig resolves the double pair from
// the simple pole. Observable companion of (13/2³⁸)((s+3·2⁻¹⁶)²+(25·2⁻¹⁷)²)
// over (s+15·2⁻¹⁷)²(s+15·2⁻¹⁷(1+2⁻¹¹))(s+15·2⁻¹⁵)(s−3·2⁻¹⁴).
func TestRootLocus_DoublePoleBesideSimplePoleExact(t *testing.T) {
	a := make([]float64, 16)
	for i := range 3 {
		a[i*4+i+1] = 1
	}
	copy(a[12:], []float64{5.5084024941437495e-21, 1.7213335791466111e-15, 1.0912870607171499e-10, -1.0494142770767212e-05})
	rlqOracleCheck(t, "R272", 4, a, []float64{0, 0, 0, 1}, []float64{1.0799670077492329e-19, -1.3322676295501878e-14, 0, 0}, 0, 2)

	q := math.Ldexp(15, -17)
	poles := []complex128{complex(-q, 0), complex(-q, 0), complex(-q*(1+math.Ldexp(1, -11)), 0), complex(-4*q, 0), complex(math.Ldexp(3, -14), 0)}
	zeros := []complex128{complex(-math.Ldexp(3, -16), math.Ldexp(25, -17)), complex(-math.Ldexp(3, -16), -math.Ldexp(25, -17))}
	num := rlsdNum(5, rlswPolyFromRoots(poles), zeros, math.Ldexp(13, -38), 0)
	c := rlsdCase(nil, poles, zeros, num, 0, 1)
	rlqOracleCheck(t, "observable 2^-11", 5, c.a, c.b, c.c, 0, 3)
}

// TestRootLocus_NoSpuriousBreakBesideNearTriple (case 2095): a dyadic
// similarity of 448/((s+3.625)²(s+3.625885009765625)(s+5.625)) whose eig
// cannot order its three cluster poles (u·κ far above their split): a
// cluster pair is no double pole, so N'D − ND' root −3.62559 (K < 0) is no
// breakaway. amd64 eig splits the cluster into a complex pair, where base
// is wrong too ([]), so the pin runs on arm64.
func TestRootLocus_NoSpuriousBreakBesideNearTriple(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("amd64 eig: precision limit, base also wrong")
	}
	a := []float64{0, 1, -2.25, 9, 0, 0, 1, -2.25, -1072.0438921451569, -1613.7440816164017, 2421.6315648257732, -9751.529799342155, -268.0109730362892, -403.4360204041004, 605.4078912064433, -2438.132449835539}
	rlqOracleCheck(t, "2095", 4, a, []float64{0, 0, 4, 1}, []float64{448, 224, -392, 1568}, 0, 2)
}

// TestRootLocus_MultipleZeroBreakOnce: a triple zero is one break point.
// Companion of −1024s³(s−512)/((s+3072)³(s−1024)²) (w17 #60): the triple
// zero at 0 is split by rounding past √u·σ but within u^(1/3)·σ, base gave
// [409.15]. Third sweep 850/325/856 (D ≠ 0, triple zero −0.03125, −1.831e-4,
// −9.52e-3: base listed it twice or the converted numerator lost it) and 2
// (triple zero at 0, amd64).
func TestRootLocus_MultipleZeroBreakOnce(t *testing.T) {
	den := rlswPolyFromRoots([]complex128{-3072, -3072, -3072, 1024, 1024})
	df := make([]float64, len(den))
	for i, x := range den {
		df[i] = rlswF(x)
	}
	a, b, c := rlpCompanion(df, []float64{-1024, 524288, 0, 0, 0})
	rlqOracleCheck(t, "w17 #60", 5, a, b, c, 0, 3)

	cases := rlsdGenerate()
	for _, i := range []int{2, 325, 850, 856} {
		sys, err := cases[i].system()
		if err != nil {
			t.Fatal(err)
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		oi, of := cases[i].oracles()
		if !oi.accepts(res.Breakaway) && !of.accepts(res.Breakaway) {
			t.Errorf("sweep %d: Breakaway = %v, oracles %v / %v", i, res.Breakaway, oi.required, of.required)
		}
	}
}
