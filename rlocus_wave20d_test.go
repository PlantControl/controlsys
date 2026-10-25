package controlsys

import (
	"testing"
)

// TestRootLocus_ReviewSweepRepros pins the wave-20 review-5 repros, cases
// of the fourth frozen sweep (rlocus_sweep_rv_test.go, full A, B, C in its
// frozen cases), judged as the sweep judges Breakaway: by the exact oracle
// of the float realization (rvScoreCase). Fixed: 684, 1694, 1829, 171
// (resolved near-axis complex zero pair taken as a double zero by the pair
// rule without a pair conditioning), 1707 (K < 0 stationary root 0.13 from
// a double pole taken as that pole), 893 (resolved slow poles and zeros
// merged in the origin band), 922, 1375, 1725 (G ≡ 0), 678 (semisimple
// double pole −0.01171875 split 2.8e-15 by eig), 838 (noise root of
// relative degree 5) and the w17 #58 cancellation class 1328, 1517, 1622,
// 1794 (stationary pair inside a pole/zero gap of ~1e-13 relative), which
// the exact path resolves.
func TestRootLocus_ReviewSweepRepros(t *testing.T) {
	cs := rvSweep4.frozenCases(t)
	oras := rvSweep4.cachedOracles(t, cs)
	for _, id := range []int{684, 1694, 1829, 171, 678, 1707, 893, 922, 1375, 1725, 1328, 1517, 1622, 1794, 838} {
		c := cs[id]
		r := rvRun(c)
		if r.Err != "" {
			t.Errorf("case %d: %s", c.ID, r.Err)
			continue
		}
		o := oras[c.ID]
		if s := rvScoreCase(c, o, r); !s.Bk {
			t.Errorf("case %d %s/%s: Breakaway %v, float-realization oracle %v (missing %v, extra %v)", c.ID, c.Fam, c.Xf, rvCx(r.Bk), o.A.Pts, s.Miss, s.Extra)
		}
	}
}
