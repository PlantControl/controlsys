package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// chainSmearOraclePoles returns the finite poles of the structured oracle,
// the generalized eigenvalues of its leading nf×nf block (F11, W11).
func chainSmearOraclePoles(t *testing.T, oracle *System, nf int) []complex128 {
	t.Helper()
	f := denseCopy(extractBlock(oracle.A, 0, 0, nf, nf))
	w := denseCopy(extractBlock(oracle.E, 0, 0, nf, nf))
	p, err := generalizedPoles(f, w, nf)
	if err != nil || len(p) != nf {
		t.Fatalf("oracle poles %v: %v", p, err)
	}
	return p
}

// stiffPoleChainStream is stiffPoleChainDescriptor drawn from stream
// (2525 there) instead, with the finite count nf.
func stiffPoleChainStream(t *testing.T, seed, stream uint64) (sys, oracle *System, nf int, proper bool) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, stream))
	proper = rng.IntN(3) != 0
	nf = 1 + rng.IntN(3)
	stiffRel := 0.0
	if rng.IntN(5) != 0 {
		stiffRel = math.Pow(10, -1-8*rng.Float64())
	}
	chains := [][]int{{2}, {3}, {4}, {5}, {6}, {2, 3}, {3, 2}, {2, 2}, {1, 3}}[rng.IntN(9)]
	sys, oracle, _ = chainDescriptor(t, rng, nf, chains, stiffRel, proper)
	return sys, oracle, nf, proper
}

// indexThreeChainSeed is the index-3 chain beside nf poles, one of E
// singular value 1e-2..1e-9 relative, of PCG(seed, 5); proper for even
// seeds.
func indexThreeChainSeed(t *testing.T, seed uint64) (sys, oracle *System, nf int, proper bool) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 5))
	nf = 1 + rng.IntN(3)
	sr := math.Pow(10, -2-7*rng.Float64())
	proper = seed%2 == 0
	sys, oracle, _ = chainDescriptor(t, rng, nf, []int{3}, sr, proper)
	return sys, oracle, nf, proper
}

// chainSmearFlat returns x row-major without stride.
func chainSmearFlat(x *mat.Dense) []float64 {
	r, c := x.Dims()
	raw := x.RawMatrix()
	out := make([]float64, 0, r*c)
	for i := range r {
		out = append(out, raw.Data[i*raw.Stride:i*raw.Stride+c]...)
	}
	return out
}

type chainSmearCase struct {
	label       string
	sys, oracle *System
	nf          int
}

// TestPolesBesideChainSmear: KLVET5. Beside an index ≥ 2 chain whose
// rounding the staircase leaks into the deflated E blocks above its
// rank tolerance, deflateInfinite stopped at an ambiguous level and Poles
// fell back to QZ of the whole pencil, which keeps the chain members it
// smears to |λ| ~ ε^(−1/k) as finite poles: refuted seed 460 gave 9 poles
// for 3 (+93147, −0.196, −0.210), largePole [5] seed 31 and the stiff
// chains below reported IsStable false for a stable model (a smeared
// member with positive real part), stiff 20 also lost its stiff pole.
// leakedRank decides those levels from the eigenvalues of the leading
// block against ε-perturbed copies deflated with the same ranks: a leaked
// zero smears the members left into a ring that moves at first order, a
// finite pole stays. A pole large in A is moved in the block by the
// staircase rounding (+93147 by 1.2e-4) and counts as staying when QZ of
// the whole pencil keeps it data-stably; the rescued staircase stands
// only when its poles carry every such QZ pole and are moved onto QZ's
// values. Oracle: the generalized eigenvalues of the structured finite
// block (F11, W11), stability from their real parts. Base fails every
// case natively (extra smeared poles, IsStable flips, stiff 20 lost
// −7.03e6).
func TestPolesBesideChainSmear(t *testing.T) {
	var cases []chainSmearCase
	for _, seed := range []uint64{1, 3, 460} {
		sys, oracle := refutedChainDescriptor(t, seed)
		cases = append(cases, chainSmearCase{fmt.Sprintf("refuted %d", seed), sys, oracle, 3})
	}
	for _, seed := range []uint64{13, 31, 35, 62} {
		sys, oracle := largePoleChainDescriptor(t, seed, []int{5}, 0)
		n, _, _ := sys.Dims()
		cases = append(cases, chainSmearCase{fmt.Sprintf("largePole [5] %d", seed), sys, oracle, n - 5})
	}
	for _, seed := range []uint64{20, 102, 1426} {
		sys, oracle, nf, _ := stiffPoleChainStream(t, seed, 2525)
		cases = append(cases, chainSmearCase{fmt.Sprintf("stiffPole %d", seed), sys, oracle, nf})
	}
	for _, seed := range []uint64{68, 152} {
		sys, oracle, nf, _ := stiffPoleChainStream(t, seed, 31)
		cases = append(cases, chainSmearCase{fmt.Sprintf("stiffPole stream 31 %d", seed), sys, oracle, nf})
	}
	for _, seed := range []uint64{123, 176} {
		sys, oracle, nf, _ := indexThreeChainSeed(t, seed)
		cases = append(cases, chainSmearCase{fmt.Sprintf("index-3 %d", seed), sys, oracle, nf})
	}
	for _, tc := range cases {
		want := chainSmearOraclePoles(t, tc.oracle, tc.nf)
		got, err := tc.sys.Poles()
		if err != nil {
			t.Fatalf("%s: Poles: %v", tc.label, err)
		}
		if len(got) != len(want) {
			t.Errorf("%s: Poles = %v, want %v", tc.label, got, want)
			continue
		}
		// 1e-4: a staircase that needs no rescue keeps its leak in the
		// pole values (refuted 3 on amd64: 1.6e-5), as on base.
		requirePolesContain(t, tc.label, got, want, 1e-4)
		wantStable := true
		for _, w := range want {
			wantStable = wantStable && real(w) < 0
		}
		if st, err := tc.sys.IsStable(); err != nil || st != wantStable {
			t.Errorf("%s: IsStable = %v, %v; want %v", tc.label, st, err, wantStable)
		}
	}
}

// TestIsProperChainSmearSplitFailsClosed: KLVET5 (amd64 stiffPole 1426,
// stream 31 seed 68, w25 #12 PCG(125, 77) chain [3] 1e-6; native index-3
// 276, stiffPole 382 and 537). When the staircase cannot resolve the
// chain, the reduction takes the QZ split, which keeps the chain members
// QZ smears over a pole small in E as finite poles: 4 and 5 poles for 2
// (TF 1.73 and 0.0646 relative off at 0.05..3, Step overflowing), a
// spurious 1.96e10 beside −1.64e6 (TF right, Step NaN), or two wrong poles
// for one (TF right at the data scale, Step 14..21% off). The rescued
// staircase counts the finite poles; a split with another count drops the
// poles none of them matches when the rest still matches the pencil
// samples (dropChainPoles), else IsProper, TransferFunction and Step fail
// closed with ErrDescriptorSingular, as MATLAB cannot be matched there.
// Oracle: the structured finite block (F11, W11) and its pencil
// response where the float data determine it.
func TestIsProperChainSmearSplitFailsClosed(t *testing.T) {
	var cases []chainSmearCase
	for _, c := range []struct{ seed, stream uint64 }{{1426, 2525}, {382, 2525}, {537, 2525}, {68, 31}} {
		sys, oracle, nf, proper := stiffPoleChainStream(t, c.seed, c.stream)
		if !proper {
			t.Fatalf("stiffPole %d/%d: improper draw", c.seed, c.stream)
		}
		cases = append(cases, chainSmearCase{fmt.Sprintf("stiffPole %d stream %d", c.seed, c.stream), sys, oracle, nf})
	}
	sys, oracle, nf, _ := indexThreeChainSeed(t, 276)
	cases = append(cases, chainSmearCase{"index-3 276", sys, oracle, nf})
	for _, sr := range []float64{1e-5, 1e-6} {
		rng := rand.New(rand.NewPCG(125, 77))
		nf := 1 + rng.IntN(3)
		sys, oracle, _ := chainDescriptor(t, rng, nf, []int{3}, sr, true)
		cases = append(cases, chainSmearCase{fmt.Sprintf("w25 chain [3] %g", sr), sys, oracle, nf})
	}
	for _, tc := range cases {
		pr, perr := tc.sys.IsProper()
		res, terr := tc.sys.TransferFunction(nil)
		_, serr := Step(tc.sys, 1)
		closed := 0
		for _, e := range []error{perr, terr, serr} {
			if errors.Is(e, ErrDescriptorSingular) {
				closed++
			}
		}
		if closed == 3 {
			continue
		}
		if closed != 0 || perr != nil || terr != nil || serr != nil || !pr {
			t.Errorf("%s: IsProper %v %v, TF %v, Step %v", tc.label, pr, perr, terr, serr)
			continue
		}
		red, _, err := tc.sys.weierstrassReduceParts()
		if err != nil {
			t.Fatalf("%s: reduction: %v", tc.label, err)
		}
		want := chainSmearOraclePoles(t, tc.oracle, tc.nf)
		var eig mat.Eigen
		if !eig.Factorize(red.A, mat.EigenNone) {
			t.Fatalf("%s: eig", tc.label)
		}
		if got := eig.Values(nil); len(got) != len(want) {
			t.Errorf("%s: reduced poles %v, want %v", tc.label, got, want)
		} else {
			requirePolesContain(t, tc.label, got, want, 1e-3)
		}
		n, m, p := tc.sys.Dims()
		a, e, b, c, d := chainSmearFlat(tc.sys.A), chainSmearFlat(tc.sys.E), chainSmearFlat(tc.sys.B), chainSmearFlat(tc.sys.C), chainSmearFlat(tc.sys.D)
		for _, s := range []complex128{0.05i, 0.37i, complex(0.5, 3)} {
			w := pencilResponse(t, tc.oracle, s)[0][0]
			x, ok := fzExactDescResponse(n, m, p, a, e, b, c, d, s)
			if !ok || cmplx.Abs(x[0]-w) > 1e-9*math.Max(1, cmplx.Abs(w)) {
				continue
			}
			g, err := res.TF.Eval(s)
			if err != nil {
				t.Fatal(err)
			}
			if diff := cmplx.Abs(g[0][0] - w); diff > 1e-6*math.Max(1, cmplx.Abs(w)) {
				t.Errorf("%s: G(%v) = %v, oracle %v", tc.label, s, g[0][0], w)
			}
		}
	}
}
