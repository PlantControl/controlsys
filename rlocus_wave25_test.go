package controlsys

import (
	"math"
	"math/big"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestRootLocus_ExactTrimArchIndependent pins XGE724: the exact break-point
// path took its numerator trim from the eig Zeros() count (dbl 1472,
// ext 1044: no Markov parameter resolved; rv5 11934: h₄ resolved but
// overridden by a zero count of 0) or from float Markov parameters
// (FMA-fused on arm64), so arm64 and amd64 cut the exact numerator to
// different degrees, both below the data's exact one. The trim now comes
// from the exact hₖ alone: the same bits on every arch, matching the
// float-realization oracle at the intended degree. sw 433 (rounding-level
// h₁ at ρ 0.10 beside the exact h₂ at 0.54, none resolved) keeps the
// numerator of the data whole, its noise zero's far break point included.
func TestRootLocus_ExactTrimArchIndependent(t *testing.T) {
	hex := func(v ...uint64) []complex128 {
		out := make([]complex128, len(v))
		for i, x := range v {
			out[i] = complex(math.Float64frombits(x), 0)
		}
		return out
	}
	type tc struct {
		name    string
		n, L    int
		a, b, c []float64
		d       float64
		spDeg   int
		want    []complex128
	}
	var cases []tc
	sw := func(name string, c rlswCase, spDeg int, want []complex128) {
		if spDeg < 0 {
			spDeg = len(rlswRatSub(c.num, fzRatPolyMul([]*big.Rat{fzRat(c.d)}, c.den))) - 1
		}
		cases = append(cases, tc{name, c.n, c.L, c.a, c.b, c.c, c.d, spDeg, want})
	}
	sw("dbl 1472", rlsdGenerate()[1472], -1, hex(0xc056c1f8372678a6, 0xc040002aaa7b7f11, 0xc040000000000000, 0x4059ca1254cbc286))
	sw("ext 1044", rlsxGenerate()[1044], -1, hex(0xc04e67d6ed48a665, 0xc049fee0803f0b55, 0xc04300caaf25594b, 0xc043000000000000, 0))
	c433 := rlswGenerate()[433]
	sw("sw 433", c433, c433.n-1, hex(0x42d64f71c6a6d299))
	cs := rvSweep5.frozenCases(t)
	rv := cs[11934-cs[0].ID]
	cases = append(cases, tc{"rv5 11934", rv.N, 0, rv.A, rv.B, rv.C, rv.D, rvSPDegree(rv.A0, rv.B0, rv.C0, rv.N, 0), []complex128{-43.85270224035004, -17.770361115252246}})

	for _, c := range cases {
		var sys *System
		var err error
		if c.L > 0 {
			sys, err = NewFromSlices(c.n, 1, 1, c.a, c.b, c.c, []float64{c.d}, 1)
			if err == nil {
				err = sys.SetInputDelay([]float64{float64(c.L)})
			}
		} else {
			sys, err = New(mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, 1, c.b), mat.NewDense(1, c.n, c.c), mat.NewDense(1, 1, []float64{c.d}), 0)
		}
		if err != nil {
			t.Fatal(err)
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if !slices.Equal(res.Breakaway, c.want) {
			t.Errorf("%s: Breakaway = %v, want %v", c.name, res.Breakaway, c.want)
		}
		rlpFloatBreak(t, c.name, res.Breakaway, rlpFloatOracle(c.n, c.a, c.b, c.c, c.d, c.L, c.spDeg))
	}
}
