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

// rlrv5Limits are base-correct breakaway verdicts the track leaves open
// (logged, not enforced; wave 22 report).
var rlrv5Limits = map[int]string{
	11942: "open (rv oracle Prec=false): 4-fold zero −36 of G split 0.09 by rounding of the similarity; as eigenvalues of A − B·C·A³/h₃ all four are u·κ-linked (u·κ ≈ 40) but the converted numerator needs ~1e3·u to merge them (10·u allowed) and the exact TF of the floats has four distinct zeros (stationary point −35.995)",
	11761: "amd64: discrete stationary points beside a pole/zero dipole off by 2e-5 relative (converted N/D accuracy)",
}

func TestRootLocusSweepRV5NoRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("frozen 2500-case sweep")
	}
	skipSlowUnderRace(t)
	rvSweep5.check(t, rlswBaseRev)
}
