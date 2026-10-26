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
func TestDescriptorChainBesideLargePoleRefutedAtLowFrequency(t *testing.T) {
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
// reading and a transfer function matching the oracle.
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
