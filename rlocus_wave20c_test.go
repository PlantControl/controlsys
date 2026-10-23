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
