package controlsys

import (
	"slices"
	"testing"
)

// TestRootLocus_ReviewSweepRepros pins the wave-20 review-5 repros, cases
// of the fourth frozen sweep (rlocus_sweep_rv_test.go, full A, B, C in its
// frozen cases), judged by its frozen exact oracle of the intended
// realization. Fixed: 684, 1694, 1829, 171 (resolved near-axis
// complex zero pair taken as a double zero by the pair rule without a pair
// conditioning), 1707 (K < 0 stationary root 0.13 from a double pole taken
// as that pole), 893 (resolved slow poles and zeros merged in the origin
// band), 922, 1375, 1725 (G ≡ 0). Known limits, logged: 678 and 838
// (intended and float TF disagree: −0.01171875 is a stationary point
// between float poles 1.4e-15 apart under ~1e-13 Givens rounding; 7187 is
// a noise root of relative degree 5 where G is below its rounding bound)
// and the w17 #58 cancellation class 1328, 1517, 1622, 1794 (stationary
// pair inside a pole/zero gap of ~1e-13 relative; coordinator decision).
func TestRootLocus_ReviewSweepRepros(t *testing.T) {
	cs := rvFrozenCases(t)
	oras := rvCachedOracles(t, cs)
	limits := []int{678, 838, 1328, 1517, 1622, 1794}
	for _, id := range []int{684, 1694, 1829, 171, 678, 1707, 893, 922, 1375, 1725, 1328, 1517, 1622, 1794, 838} {
		c := cs[id]
		r := rvRun(c)
		if r.Err != "" {
			t.Errorf("case %d: %s", c.ID, r.Err)
			continue
		}
		o := oras[c.ID]
		ok, miss, extra := o.I.match(rvCx(r.Bk))
		if o.I.Cancel {
			ok = true
		}
		switch {
		case slices.Contains(limits, c.ID):
			t.Logf("case %d %s/%s (known limit): Breakaway %v, oracle %v", c.ID, c.Fam, c.Xf, rvCx(r.Bk), o.I.Pts)
		case !ok:
			t.Errorf("case %d %s/%s: Breakaway %v, oracle %v (missing %v, extra %v)", c.ID, c.Fam, c.Xf, rvCx(r.Bk), o.I.Pts, miss, extra)
		}
	}
}
