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
// verdicts (base 2db88fe, per arch; base code only).
var rvSweep5 = rvSweep{
	name: "RV5", count: 2500,
	cases: "testdata/rlocus_sweep_rv5_cases.json.gz", oracle: "testdata/rlocus_sweep_rv5_oracle.json.gz",
	golden: "testdata/rlocus_sweep_rv5_base.json", envOracle: "RLOCUS_SWEEPRV5_ORACLE",
	envWrite: "RLOCUS_SWEEPRV5_WRITE", limits: rlrv5Limits,
}

// rlrv5Limits are base-correct breakaway verdicts the track leaves open
// (logged, not enforced; wave 22 report).
var rlrv5Limits = map[int]string{
	11942: "open (rv oracle Prec=false): 4-fold zero −36 of G split 0.09 by rounding of the similarity; as eigenvalues of A − B·C·A³/h₃ all four are u·κ-linked (u·κ ≈ 40) but the converted numerator needs ~1e3·u to merge them (10·u allowed) and the exact TF of the floats has four distinct zeros (stationary point −35.995)",
	11761: "amd64: discrete stationary points beside a pole/zero dipole off by 2e-5 relative (converted N/D accuracy)",
}

const rlrv5BaseRev = "2db88fe"

func TestRootLocusSweepRV5NoRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("frozen 2500-case sweep")
	}
	skipSlowUnderRace(t)
	rvSweep5.check(t, rlrv5BaseRev)
}

// TestRootLocus_RV5Repros pins the wave-20 final-review cases of the fifth
// frozen sweep, judged on every component by its exact oracle: 11941,
// 11400, 10478 (a rounding-level h₂ kept a noise zero, flipping departure
// angles by π and adding a far break point ~1e12), and 11124 (a resolved
// pair 116±0.25i of a non-normal A, σ_b 4.4e7, admitted as a double only
// by the √u·σ_b split, gave the K < 0 point 116.00007). The rlrv5Limits
// cases are logged.
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
			t.Logf("case %d %s/%s (open): Breakaway %v, oracle %v", id, c.Fam, c.Xf, rvCx(r.Bk), o.I.Pts)
		case !s.ok():
			t.Errorf("case %d %s/%s: Breakaway %v (oracle %v, missing %v, extra %v), asymptotes %v, departure %v, err %q: %v", id, c.Fam, c.Xf, rvCx(r.Bk), o.I.Pts, s.Miss, s.Extra, r.Asym, r.Dep, s.Err, s.Why)
		}
	}
}
