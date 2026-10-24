package controlsys

import (
	"math/cmplx"
	"slices"
	"testing"
)

// TestTransferFunctionSISONumeratorCappedAtRelativeDegree checks that a SISO
// TF with D = 0 has numerator degree at most n − r, r from
// sisoRelativeDegree, over the frozen RootLocus sweep realizations: the
// Markov parameters h_1 … h_(r−1) vanish within the rounding of A, B, C, so
// higher coefficients are noise, and capSISOZeros caps Zeros at the same
// n − r (tzero parity). Base 23be053 kept them on 63 of these cases
// (arm64; BVLIUF).
func TestTransferFunctionSISONumeratorCappedAtRelativeDegree(t *testing.T) {
	cases := slices.Concat(rlswGenerate(), rlsxGenerate(), rlsdGenerate())
	bad := 0
	for i := range cases {
		c := &cases[i]
		if c.d != 0 {
			continue
		}
		sys, err := NewFromSlices(c.n, 1, 1, c.a, c.b, c.c, []float64{c.d}, c.dt)
		if err != nil {
			t.Fatal(err)
		}
		r, ok := sisoRelativeDegree(sys)
		if !ok {
			continue
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		num, den := res.TF.Num[0][0], res.TF.Den[0]
		if len(num) > len(den)-r {
			if bad++; bad <= 5 {
				t.Errorf("case %d: num %v of degree %d, den degree %d, relative degree %d\n%s", i, num, len(num)-1, len(den)-1, r, c)
			}
		}
	}
	if bad > 0 {
		t.Errorf("%d cases keep noise numerator coefficients", bad)
	}
}

// TestTransferFunctionSISONoiseNumeratorRootLocus pins frozen-sweep cases
// whose TF kept a rounding-level leading numerator coefficient (a far zero
// ~1e10 that Zeros drops), giving RootLocus a far spurious break point or
// the wrong real-axis segments. Each intended model has len(zeros) = n − r
// finite zeros; the TF must have those and the exact breakaway set.
func TestTransferFunctionSISONoiseNumeratorRootLocus(t *testing.T) {
	cases := rlswGenerate()
	for _, i := range []int{36, 142, 184, 274, 529, 725, 968, 1323} {
		c := &cases[i]
		sys, err := c.system()
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		num := res.TF.Num[0][0]
		if len(num)-1 != len(c.zeros) {
			t.Errorf("case %d: num %v, want degree %d (zeros %v)", i, num, len(c.zeros), c.zeros)
			continue
		}
		roots, err := Poly(num).Roots()
		if err != nil {
			t.Fatal(err)
		}
		for _, z := range c.zeros {
			if !slices.ContainsFunc(roots, func(x complex128) bool { return cmplx.Abs(x-z) <= 1e-6*max(cmplx.Abs(z), 1) }) {
				t.Errorf("case %d: num roots %v miss intended zero %v", i, roots, z)
			}
		}
		oi, of := c.oracles()
		ok, posed, _, err := rlswVerdict(c, oi, of)
		if err != nil {
			t.Fatal(err)
		}
		if posed&rlswBreak != 0 && ok&rlswBreak == 0 {
			rl, _ := RootLocus(sys, []float64{0, 1})
			t.Errorf("case %d: Breakaway %v, exact %v\n%s", i, rl.Breakaway, oi.required, c)
		}
	}
}
