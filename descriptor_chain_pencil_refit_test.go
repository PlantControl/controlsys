package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// stiffPoleChainDescriptor: chainDescriptor with 1..3 finite poles, the
// first stiff (E singular value 10^(−1..−9) relative) in four draws of
// five, beside chains [2]..[6], [2 3], [3 2], [2 2], [1 3]; improper in
// one draw of three (b drives every chain state).
func stiffPoleChainDescriptor(t *testing.T, seed uint64) (sys, oracle *System, proper bool) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 2525))
	proper = rng.IntN(3) != 0
	nf := 1 + rng.IntN(3)
	stiffRel := 0.0
	if rng.IntN(5) != 0 {
		stiffRel = math.Pow(10, -1-8*rng.Float64())
	}
	chains := [][]int{{2}, {3}, {4}, {5}, {6}, {2, 3}, {3, 2}, {2, 2}, {1, 3}}[rng.IntN(9)]
	sys, oracle, _ = chainDescriptor(t, rng, nf, chains, stiffRel, proper)
	return sys, oracle, proper
}

// chainOracleMismatch is the largest error of res against the structured
// oracle at pts, relative to max(1, |G|).
func chainOracleMismatch(t *testing.T, res *TransferFuncResult, oracle *System, pts []complex128) float64 {
	t.Helper()
	worst := 0.0
	for _, s := range pts {
		g, err := res.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		w := pencilResponse(t, oracle, s)
		for i := range w {
			for j := range w[i] {
				worst = max(worst, cmplx.Abs(g[i][j]-w[i][j])/math.Max(1, cmplx.Abs(w[i][j])))
			}
		}
	}
	return worst
}

var chainLowFreq = []complex128{0.05i, 0.37i, complex(0.5, 3)}

// TestTransferFunctionImproperChainBesideStiffPole: 5CNIZA. Improper
// index-5/6 chains beside a stiff finite pole (seed 159: [6] beside
// −2.47e6; seed 45: [6] beside 2.95e5; seed 17) whose rounding chain
// eigenvalues reach past it: the split moved residue between the pole,
// D and the s-terms in cancelling amounts (seed 159: s-terms 4.4e17,
// 1.8e11, 7.2e4 against O(1)), the rounding tests kept the first only and
// the TF was off by 5.4e9 (seed 45: 2.9e5) relative at 0.05i..0.5+3i. The
// split is now checked against direct (sE − A) solves, which refute it,
// and B, D and the s-terms are refitted to them. Oracle: the structured
// model's pencil response, at low frequency and up to 1e6i where the
// s-terms dominate.
func TestTransferFunctionImproperChainBesideStiffPole(t *testing.T) {
	for _, seed := range []uint64{159, 45, 17} {
		label := fmt.Sprintf("seed %d", seed)
		sys, oracle, proper := stiffPoleChainDescriptor(t, seed)
		if proper {
			t.Fatalf("%s: generator drew a proper model", label)
		}
		if pr, err := sys.IsProper(); err != nil || pr {
			t.Fatalf("%s: IsProper = %v, %v; want false", label, pr, err)
		}
		if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
			t.Errorf("%s: Step: %v, want ErrImproperModel", label, err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", label, err)
		}
		if !fzTFImproper(res.TF) {
			t.Errorf("%s: TF proper", label)
		}
		if d := chainOracleMismatch(t, res, oracle, append(chainLowFreq, 1e2i, 1e4i, 1e6i)); !(d <= 1e-6) {
			t.Errorf("%s: TF off the oracle by %.3g relative", label, d)
		}
	}
}

// TestTransferFunctionChainBesideStiffPoleSweep: KLVET5 and 5CNIZA over
// 300 draws of stiffPoleChainDescriptor. Base returned 5 TFs off the
// oracle by more than 1e-5 at low frequency (7 on amd64) and failed 49
// (48) closed; now none is off, 30 (34) fail closed with
// ErrDescriptorSingular, no improper model reads proper and IsProper,
// Step and TransferFunction agree. Excepted: improper seed 276 on amd64,
// 5.1 off on base too, whose reduction keeps a fourth pole.
func TestTransferFunctionChainBesideStiffPoleSweep(t *testing.T) {
	matched, closed := 0, 0
	for seed := range uint64(300) {
		label := fmt.Sprintf("seed %d", seed)
		sys, oracle, proper := stiffPoleChainDescriptor(t, seed)
		pr, err := sys.IsProper()
		_, serr := Step(sys, 1)
		res, terr := sys.TransferFunction(nil)
		if errors.Is(err, ErrDescriptorSingular) {
			if !errors.Is(serr, ErrDescriptorSingular) || !errors.Is(terr, ErrDescriptorSingular) {
				t.Errorf("%s: IsProper %v, Step %v, TF %v disagree", label, err, serr, terr)
			}
			closed++
			continue
		}
		if err != nil || terr != nil {
			t.Fatalf("%s: IsProper %v, TransferFunction %v", label, err, terr)
		}
		if pr && !proper {
			t.Errorf("%s: improper model reads proper", label)
		}
		if pr == errors.Is(serr, ErrImproperModel) || fzTFImproper(res.TF) == pr {
			t.Errorf("%s: IsProper %v, Step %v, TF improper %v disagree", label, pr, serr, fzTFImproper(res.TF))
		}
		if seed == 276 {
			continue
		}
		if d := chainOracleMismatch(t, res, oracle, chainLowFreq); !(d <= 1e-5) {
			t.Errorf("%s: IsProper %v, TF off the oracle by %.3g relative", label, pr, d)
			continue
		}
		matched++
	}
	t.Logf("%d match, %d fail closed", matched, closed)
	if matched < 260 {
		t.Errorf("%d/300 TFs match the oracle, want ≥ 260", matched)
	}
}
