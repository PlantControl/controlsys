package controlsys

import (
	"errors"
	"fmt"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// withSmallestSingularValue returns sys with E's least singular value set
// to rel·σ₁(E).
func withSmallestSingularValue(t *testing.T, sys *System, rel float64) *System {
	t.Helper()
	var svd mat.SVD
	if !svd.Factorize(sys.E, mat.SVDFull) {
		t.Fatal("SVD failed")
	}
	sv := svd.Values(nil)
	n := len(sv)
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	var d mat.Dense
	d.Outer(rel*sv[0]-sv[n-1], u.ColView(n-1), v.ColView(n-1))
	out := sys.Copy()
	out.E.Add(out.E, &d)
	return out
}

// TestDescriptorNearSingularEReducedNotInverted: an E whose least singular
// value is a few ε·σ₁ (4ε jitterDescriptor, PCG(seed, 99 + copy), of
// U·W·V with singular W) has rank below n by MATLAB's rank, but its LU condition estimate stays under 1/ε. Base
// inverted it: E⁻¹A carried the infinite chain as poles near 1/ε, so
// IsProper read true for improper chains and TransferFunction was 0.5
// relative off at 0.5+3i, where the data determine the response to 2e-12
// (seed 47, chain 3, bscale 1e-4, jitter copies 1 and 4). The reduction now
// splits the chain as for exactly singular E, so IsProper, Step and the
// transfer function agree with the structured oracle or fail closed.
func TestDescriptorNearSingularEReducedNotInverted(t *testing.T) {
	type tc struct {
		label       string
		sys, oracle *System
		improper    bool
	}
	var cases []tc
	for _, c := range []struct {
		seed   uint64
		chain  int
		bscale float64
		cps    []int
	}{
		{47, 3, 1e-4, []int{1, 4}},
		{47, 3, 1e-2, []int{1, 4}},
		{46, 2, 1e-2, []int{0, 2}},
		{46, 2, 0, []int{0, 2}},
	} {
		sys, oracle := largePoleChainDescriptor(t, c.seed, []int{c.chain}, c.bscale)
		for _, cp := range c.cps {
			cases = append(cases, tc{fmt.Sprintf("seed %d chain %d bscale %g jitter %d", c.seed, c.chain, c.bscale, cp),
				jitterDescriptor(sys, rand.New(rand.NewPCG(c.seed, 99+uint64(cp))), 4), oracle, c.bscale != 0})
		}
		for _, rel := range []float64{1.5, 3} {
			cases = append(cases, tc{fmt.Sprintf("seed %d chain %d bscale %g σmin %gε", c.seed, c.chain, c.bscale, rel),
				withSmallestSingularValue(t, sys, rel*eps()), oracle, c.bscale != 0})
		}
	}
	checked := 0
	for _, c := range cases {
		var lu mat.LU
		lu.Factorize(c.sys.E)
		if luNearSingular(&lu) {
			continue
		}
		checked++
		pr, closed := checkChainDescriptorReading(t, c.label, c.sys, c.oracle)
		if !closed && pr == c.improper {
			t.Errorf("%s: IsProper %v, oracle improper %v", c.label, pr, c.improper)
		}
		if _, err := c.sys.ToExplicit(); !errors.Is(err, ErrDescriptorSingular) {
			t.Errorf("%s: ToExplicit err %v, want ErrDescriptorSingular", c.label, err)
		}
	}
	if checked < len(cases)/2 {
		t.Fatalf("only %d of %d cases have E with LU condition under 1/ε", checked, len(cases))
	}
}

// TestDescriptorWellConditionedEStaysExplicit: an E with least singular
// value well above n·ε·σ₁ is inverted as before.
func TestDescriptorWellConditionedEStaysExplicit(t *testing.T) {
	sys, _ := largePoleChainDescriptor(t, 47, []int{3}, 1e-4)
	near := withSmallestSingularValue(t, sys, 1e-9)
	if _, err := near.ToExplicit(); err != nil {
		t.Fatalf("ToExplicit: %v", err)
	}
	if pr, err := near.IsProper(); err != nil || !pr {
		t.Fatalf("IsProper = %v, %v; want true", pr, err)
	}
}
