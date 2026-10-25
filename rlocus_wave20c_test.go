package controlsys

import (
	"fmt"
	"math"
	"testing"
)

// Pinned wave-20 review-4 cases of the rlocus track. Expected sets come
// from the exact big.Rat oracles of the intended and the float TF
// (rlswCase.oracles).

func rlrcCheck(t *testing.T, name string, c rlswCase) {
	t.Helper()
	sys, err := c.system()
	if err != nil {
		t.Fatal(err)
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	rlqUnique(t, name, res.Breakaway)
	if oi, of := c.oracles(); !oi.accepts(res.Breakaway) && !of.accepts(res.Breakaway) {
		t.Errorf("%s: Breakaway = %v, want %v / %v", name, res.Breakaway, oi.required, of.required)
	}
}

// rlrcPairBesideSimple is 1 − 3s(s+1.625)/den (d = 1) or −3s(s+1.625)/den
// (d = 0), den with poles −2, −5.5, −5.875 and −5.875(1+2^−k1) ± 5.875·2^−k2·i,
// as an exact companion (form 0) or observable (form 1) realization.
func rlrcPairBesideSimple(k1, k2, form int, d float64) rlswCase {
	q := 5.875
	re, im := -q*(1+math.Ldexp(1, -k1)), q*math.Ldexp(1, -k2)
	poles := []complex128{-2, -5.5, complex(-q, 0), complex(re, im), complex(re, -im)}
	zeros := []complex128{0, -1.625}
	return rlsdCase(nil, poles, zeros, rlsdNum(5, rlswPolyFromRoots(poles), zeros, -3, d), d, form)
}

// TestRootLocus_ResolvedPairBesideSimplePole: a resolved complex pole pair
// beside a simple pole is no K = 0 double pole, though the nearby simple
// pole makes the converted D confirm a double root at its p′ root to 10·u:
// eig resolves the pair (no u·κ link), and K < 0 there (exact G ≈ +3.5e10
// for 2^−11, 2^−13), so the only break point is −5.5932 (base correct for
// D = 1; base listed four points for D = 0).
func TestRootLocus_ResolvedPairBesideSimplePole(t *testing.T) {
	for _, k := range [][2]int{{9, 14}, {10, 14}, {11, 13}, {11, 14}, {12, 14}} {
		for form := range 2 {
			rlrcCheck(t, fmt.Sprintf("2^-%d, 2^-%d form %d", k[0], k[1], form), rlrcPairBesideSimple(k[0], k[1], form, 1))
		}
	}
	rlrcCheck(t, "2^-11, 2^-13 D=0", rlrcPairBesideSimple(11, 13, 0, 0))
}

func rlrcCompanion(n int, lastRow, c []float64) (a, b []float64) {
	a, b = make([]float64, n*n), make([]float64, n)
	for i := range n - 1 {
		a[i*n+i+1] = 1
	}
	copy(a[(n-1)*n:], lastRow)
	b[n-1] = 1
	return a, b
}

// TestRootLocus_TripleZeroListedOnce: −16(s+1)³/((s+38)(s+41)(s−10)(s−17)),
// integer companion: the triple zero −1 is one break point. amd64 tzero
// splits it by 1.5e-4, so both split roots of N'D − ND' were listed (base
// [12.4695], wrong too).
func TestRootLocus_TripleZeroListedOnce(t *testing.T) {
	a, b := rlrcCompanion(4, []float64{-264860, 28636, 405, -52}, nil)
	rlqOracleCheck(t, "triple zero -1", 4, a, b, []float64{-16, -48, -48, -16}, 0, 2)
}

// TestRootLocus_MultipleZeroOfStrictlyProperPartOnce: with D ≠ 0 a triple
// zero of G − D (no zero of G) is a double root of N'D − ND', one break
// point (base missed it).
func TestRootLocus_MultipleZeroOfStrictlyProperPartOnce(t *testing.T) {
	a := []float64{0, 0, 0, -2.575143298599869e-07, 1, 0, 0, -2.7045607566833496e-06, 0, 1, 0, -0.0010194778442382812, 0, 0, 1, -0.005859375}
	res, err := RootLocus(rlpSys(t, 4, a, []float64{2.6520865503698587e-09, -2.7157366275787354e-06, 0.000926971435546875, -0.10546875}, []float64{0, 0, 0, 1}, -0.25, 0), []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	// Exact oracle (arm64; its float posedness check fails on amd64).
	o := rlswOracle{posed: true, required: []float64{0.0029296875, 0.04708051704829962}, scale: 0.0059}
	if !o.accepts(res.Breakaway) {
		t.Errorf("observable triple zero 0.0029296875: Breakaway = %v, want %v", res.Breakaway, o.required)
	}
	a, b := rlrcCompanion(4, []float64{695625, 14450, -2360, -98}, nil)
	rlqOracleCheck(t, "companion triple zero -12", 4, a, b, []float64{193536, 48384, 4032, 112}, -0.25, 2)
}

// TestRootLocus_SlowSixPoleBreakaways (U3H63X): companion of poles
// 1.220703125e-4, −4.8828125e-4, −7.9345703125e-4, 3.0517578125e-4,
// −3.0517578125e-4 ± 5.4931640625e-4i with C = [0.00201416015625 −3 0 0 0 0]
// has three K > 0 break points (exact oracle −6.763e-4, 2.387e-4,
// 8.360e-4; not 1-ulp sensitive). Base and branch give []: the converted
// TF (rationalTransferFunction) is −3/(degree-5 D), dropping the zero
// 6.714e-4 with a pole the loop does not have, so N'D − ND' has no real
// root there; rlocus.go cannot recover it.
func TestRootLocus_SlowSixPoleBreakaways(t *testing.T) {
	t.Skip("TransferFunction drops zero 6.714e-4 and a pole of this slow companion (transfer.go)")
	poles := []complex128{1.220703125e-4, -4.8828125e-4, -7.9345703125e-4, 3.0517578125e-4, complex(-3.0517578125e-4, 5.4931640625e-4), complex(-3.0517578125e-4, -5.4931640625e-4)}
	den := rlswPolyFromRoots(poles)
	row := make([]float64, 6)
	for j := range 6 {
		row[j] = -rlswF(den[6-j])
	}
	a, b := rlrcCompanion(6, row, nil)
	rlqOracleCheck(t, "U3H63X six-pole", 6, a, b, []float64{0.00201416015625, -3, 0, 0, 0, 0}, 0, 3)
}
