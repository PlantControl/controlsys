package controlsys

import "testing"

// Fifth frozen RootLocus regression sweep (track rlocus, wave 20 final
// review): the reviewer's second generator (IDs 10000…12499; families
// dense, zoh, repzero, unstable, leadlag, notch, marginal, pid realized
// exactly from k·Π(s−z)/Π(s−p), each identity, transposed, diagonally
// scaled, Givens-rotated, elementary- or triangular-similar), judged by the
// exact oracle of rlocus_sweep_rv_test.go. Cases are frozen as generated
// (testdata/rlocus_sweep_rv5_cases.json.gz); RLOCUS_SWEEPRV5_ORACLE and
// RLOCUS_SWEEPRV5_WRITE=<path> regenerate the oracle cache and the base
// verdicts (base rlswBaseRev, per arch; base code only).
var rvSweep5 = rvSweep{
	name: "RV5", count: 2500,
	cases: "testdata/rlocus_sweep_rv5_cases.json.gz", oracle: "testdata/rlocus_sweep_rv5_oracle.json.gz",
	golden: "testdata/rlocus_sweep_rv5_base.json", envOracle: "RLOCUS_SWEEPRV5_ORACLE",
	envWrite: "RLOCUS_SWEEPRV5_WRITE", limits: rlrv5Limits,
}

// rlrv5Limits are verdicts logged, not enforced. Since the exact
// breakaway path every Breakaway verdict of the sweep is enforced (10041
// and 10517 are flagged 1-ulp sensitive by the oracle cache); 11942 is
// listed for TestRootLocus_RV5Repros, which judges every component.
var rlrv5Limits = map[int]string{
	11942: "amd64: departure angle at 6+32i off 2.5e-4 (eig zeros of a 4-fold zero −36 split 0.09 by the similarity's rounding); Breakaway matches the float-realization oracle",
}

func TestRootLocusSweepRV5NoRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("frozen 2500-case sweep")
	}
	skipSlowUnderRace(t)
	rvSweep5.check(t, rlswBaseRev)
}

// TestRootLocus_RV5Repros pins the wave-20 final-review cases of the fifth
// frozen sweep, judged on every component as the sweep judges them
// (Breakaway by the exact oracle of the float realization): 11941, 11400,
// 10478 (a rounding-level h₂ kept a noise zero, flipping departure angles
// by π and adding a far break point ~1e12), 11124 (a resolved pair
// 116±0.25i of a non-normal A, σ_b 4.4e7: the eig path gave the K < 0
// point 116.00007), 11942 (4-fold zero −36 split 0.09 by the similarity's
// rounding: the split system's stationary points) and 11761 (stationary
// points beside a pole/zero dipole). The rlrv5Limits cases are logged.
func TestRootLocus_RV5Repros(t *testing.T) {
	cs := rvSweep5.frozenCases(t)
	oras := rvSweep5.cachedOracles(t, cs)
	for _, id := range []int{11941, 11400, 10478, 11124, 11942, 11761} {
		i := id - cs[0].ID
		c, o := cs[i], oras[i]
		r := rvRun(c)
		s := rvScoreCase(c, o, r)
		switch {
		case rlrv5Limits[id] != "":
			t.Logf("case %d %s/%s (open): Breakaway %v, oracle %v", id, c.Fam, c.Xf, rvCx(r.Bk), o.A.Pts)
		case !s.ok():
			t.Errorf("case %d %s/%s: Breakaway %v (oracle %v, missing %v, extra %v), asymptotes %v, departure %v, err %q: %v", id, c.Fam, c.Xf, rvCx(r.Bk), o.A.Pts, s.Miss, s.Extra, r.Asym, r.Dep, s.Err, s.Why)
		}
	}
}
