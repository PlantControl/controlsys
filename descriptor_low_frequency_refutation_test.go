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

// largePoleChainDescriptor returns U·(F, W, b, c, D)·V and its structured
// oracle: one finite pole F₀/W₀ with |F₀| in 1e2..1e6, up to two O(1)
// poles, infinite chains of the given lengths (F = I, W = shift), SISO,
// Gaussian U, V; the chain kernel state is driven by a Gaussian b and
// the next chain state by a Gaussian times bscale (0: proper).
func largePoleChainDescriptor(t *testing.T, seed uint64, chains []int, bscale float64) (sys, oracle *System) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 2828))
	nf := 1 + rng.IntN(3)
	f, w := make([]float64, nf), make([]float64, nf)
	for i := range nf {
		f[i], w[i] = rng.NormFloat64(), 0.5+rng.Float64()
	}
	f[0] = math.Copysign(math.Pow(10, 2+4*rng.Float64()), rng.NormFloat64())
	n := nf
	for _, k := range chains {
		n += k
	}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	b, c, D := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil), mat.NewDense(1, 1, nil)
	for i := range nf {
		F.Set(i, i, f[i])
		W.Set(i, i, w[i])
		b.Set(i, 0, rng.NormFloat64())
		c.Set(0, i, rng.NormFloat64())
	}
	o := nf
	for _, k := range chains {
		for i := range k {
			F.Set(o+i, o+i, 1)
			if i+1 < k {
				W.Set(o+i, o+i+1, 1)
			}
			c.Set(0, o+i, rng.NormFloat64())
			v := rng.NormFloat64()
			switch i {
			case 0:
				b.Set(o+i, 0, v)
			case 1:
				b.Set(o+i, 0, v*bscale)
			}
		}
		o += k
	}
	D.Set(0, 0, rng.NormFloat64())
	gauss := func() *mat.Dense {
		g := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				g.Set(i, j, rng.NormFloat64())
			}
		}
		return g
	}
	U, V := gauss(), gauss()
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	var err error
	if sys, err = NewDescriptor(&A, &B, &C, D, &E, 0); err != nil {
		t.Fatal(err)
	}
	if oracle, err = NewDescriptor(F, b, c, D, W, 0); err != nil {
		t.Fatal(err)
	}
	return sys, oracle
}

// checkChainDescriptorReading requires IsProper, Step and TransferFunction
// of sys to agree, to fail closed with ErrDescriptorSingular together or
// to return values, and a returned transfer function to match the oracle
// at low-frequency points where the data determine the response (the
// direct pencil solve of sys agrees with the oracle to det there).
func checkChainDescriptorReading(t *testing.T, label string, sys, oracle *System) (proper bool, closed bool) {
	t.Helper()
	pr, perr := sys.IsProper()
	_, serr := Step(sys, 1)
	res, terr := sys.TransferFunction(nil)
	if perr != nil {
		if !errors.Is(perr, ErrDescriptorSingular) || !errors.Is(serr, ErrDescriptorSingular) || !errors.Is(terr, ErrDescriptorSingular) {
			t.Fatalf("%s: IsProper err %v, Step err %v, TF err %v: want all ErrDescriptorSingular", label, perr, serr, terr)
		}
		return false, true
	}
	if pr == errors.Is(serr, ErrImproperModel) || serr != nil && !errors.Is(serr, ErrImproperModel) {
		t.Fatalf("%s: IsProper %v but Step err %v", label, pr, serr)
	}
	if terr != nil {
		t.Fatalf("%s: TransferFunction: %v", label, terr)
	}
	if fzTFImproper(res.TF) == pr {
		t.Fatalf("%s: IsProper %v but TF improper %v", label, pr, fzTFImproper(res.TF))
	}
	for _, s := range []complex128{0.05i, 0.37i, complex(0.5, 3)} {
		want := pencilResponse(t, oracle, s)
		direct := pencilResponse(t, sys, s)
		got, err := res.TF.Eval(s)
		if err != nil {
			t.Fatalf("%s: TF.Eval(%v): %v", label, s, err)
		}
		sc := math.Max(1, cmplx.Abs(want[0][0]))
		det := cmplx.Abs(direct[0][0]-want[0][0]) / sc
		if det > 1e-7 {
			continue
		}
		if d := cmplx.Abs(got[0][0]-want[0][0]) / sc; d > math.Max(1e-6, 1e3*det) {
			t.Fatalf("%s: TF(%v) = %v, oracle %v (rel %.2g, data determine it to %.2g)", label, s, got[0][0], want[0][0], d, det)
		}
	}
	return pr, false
}

// TestDescriptorChainBesideLargePoleRefutedAtLowFrequency: beside an
// index-5 or index-6 chain the Weierstrass split from a large-A finite
// pole is ill-conditioned (|c₂| up to 1e25 for seed 35) and the proper
// reduction misallocates a residue of 1e18, so its TF is up to 3e5
// relative off at 0.05i, where the data determine the response to 1e-10.
// No direct pencil solve in pencilSamples' range is determined by the
// data, so only the low-frequency points can refute it; the reading
// fails closed instead (E664VD; base read IsProper true with the wrong TF
// for the improper chains and the wrong TF for their proper twins).
// Skipped under -race: without FMA fusion the split rounds differently
// and several seeds (89, 60) keep base's wrong TF unrefuted.
func TestDescriptorChainBesideLargePoleRefutedAtLowFrequency(t *testing.T) {
	if raceEnabled {
		t.Skip("rounding-regime dependent: no FMA fusion under -race")
	}
	for _, tc := range []struct {
		seed  uint64
		chain int
	}{{35, 6}, {89, 6}, {163, 6}, {60, 5}, {10, 5}, {60, 4}} {
		for _, bscale := range []float64{1e-8, 1e-4, 0} {
			sys, oracle := largePoleChainDescriptor(t, tc.seed, []int{tc.chain}, bscale)
			label := fmt.Sprintf("seed %d chain %d bscale %g", tc.seed, tc.chain, bscale)
			pr, closed := checkChainDescriptorReading(t, label, sys, oracle)
			if !closed && bscale != 0 && pr {
				t.Errorf("%s: improper chain read proper", label)
			}
		}
	}
}

// TestDescriptorChainBesideLargePoleReadingsKept: proper and improper
// chains beside a large-A pole that the split reads correctly keep their
// reading and a transfer function matching the oracle. Excepted under
// -race (no FMA fusion): proper seed 6 chain 6, which fails closed there,
// as on base.
func TestDescriptorChainBesideLargePoleReadingsKept(t *testing.T) {
	for _, tc := range []struct {
		chain  int
		bscale float64
		seeds  []uint64
	}{
		{4, 0, []uint64{2, 3, 4, 5, 7}},
		{6, 0, []uint64{3, 6, 11, 12, 15}},
		{4, 1e-4, []uint64{2, 3, 4, 5, 7}},
		{6, 1e-4, []uint64{3, 4, 5, 7, 12}},
	} {
		for _, seed := range tc.seeds {
			if raceEnabled && seed == 6 && tc.chain == 6 && tc.bscale == 0 {
				continue
			}
			sys, oracle := largePoleChainDescriptor(t, seed, []int{tc.chain}, tc.bscale)
			label := fmt.Sprintf("seed %d chain %d bscale %g", seed, tc.chain, tc.bscale)
			pr, closed := checkChainDescriptorReading(t, label, sys, oracle)
			switch {
			case closed:
				t.Errorf("%s: failed closed", label)
			case pr != (tc.bscale == 0):
				t.Errorf("%s: IsProper %v", label, pr)
			}
		}
	}
}

// TestDescriptorImproperChainBesideLargePoleRefutedAtLowFrequency: the
// improper split beside an index-6 chain and a large-A pole kept a
// misallocated residue unchecked when no pencilSamples point in the range
// was usable (pointEval fails there) or refitPolyFromPencil found no refit:
// base read IsProper false with a TF 0.4 to 93 relative off the oracle at
// low frequency, where the data determine it to 1e-10, for the improper
// chains and their proper twins. The points under the range refute it, as
// they do the proper split (QNIQ7V). Skipped under -race like
// TestDescriptorChainBesideLargePoleRefutedAtLowFrequency.
func TestDescriptorImproperChainBesideLargePoleRefutedAtLowFrequency(t *testing.T) {
	if raceEnabled {
		t.Skip("rounding-regime dependent: no FMA fusion under -race")
	}
	for _, tc := range []struct {
		firstTail bool
		seed      uint64
	}{{false, 8}, {false, 30}, {false, 164}, {false, 192}, {true, 29}, {true, 89}} {
		for _, bscale := range []float64{1e-8, 0} {
			var sys, oracle *System
			if tc.firstTail {
				sys, oracle = firstTailChainDescriptor(t, tc.seed, []int{6}, bscale, false)
			} else {
				sys, oracle = largePoleChainDescriptor(t, tc.seed, []int{6}, bscale)
			}
			label := fmt.Sprintf("firstTail %v seed %d bscale %g", tc.firstTail, tc.seed, bscale)
			pr, closed := checkChainDescriptorReading(t, label, sys, oracle)
			if !closed && pr != (bscale == 0) {
				t.Errorf("%s: IsProper %v", label, pr)
			}
		}
	}
}

// TestDescriptorImproperChainBesideLargePoleReadingsKept: improper splits
// beside index-5 and index-6 chains and a large-A pole that match the
// points under the range keep their reading and an oracle-matching TF.
func TestDescriptorImproperChainBesideLargePoleReadingsKept(t *testing.T) {
	for _, chain := range []int{5, 6} {
		for _, seed := range []uint64{3, 4, 5, 7, 12} {
			sys, oracle := firstTailChainDescriptor(t, seed, []int{chain}, 1e-4, false)
			label := fmt.Sprintf("chain %d seed %d", chain, seed)
			pr, closed := checkChainDescriptorReading(t, label, sys, oracle)
			switch {
			case closed:
				t.Errorf("%s: failed closed", label)
			case pr:
				t.Errorf("%s: improper chain read proper", label)
			}
		}
	}
}

// TestReductionRefutedBelowBlindSamples: with no pencil point usable in
// or under the range (every direct solve of a singular pencil fails) the
// reduction counts as refuted, not as matching zero samples; an exactly
// zero response (C = 0) is determined by the data and refutes nothing.
func TestReductionRefutedBelowBlindSamples(t *testing.T) {
	red, err := New(mat.NewDense(1, 1, []float64{-2}), mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	b, d := mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 1, nil)
	singular, err := NewDescriptor(mat.NewDense(2, 2, []float64{-2, 0, 0, 0}), b,
		mat.NewDense(1, 2, []float64{1, 1}), d, mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if !reductionRefutedBelow(singular, red, nil) {
		t.Error("singular pencil: no usable sample read as not refuted")
	}
	zero, err := NewDescriptor(mat.NewDense(2, 2, []float64{-2, 0, 0, 1}), b,
		mat.NewDense(1, 2, nil), d, mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	zeroRed := red.Copy()
	zeroRed.C = mat.NewDense(1, 1, nil)
	if reductionRefutedBelow(zero, zeroRed, nil) {
		t.Error("zero response: matching reduction refuted")
	}
}
