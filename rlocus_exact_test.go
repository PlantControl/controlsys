package controlsys

import (
	"math"
	"math/big"
	"slices"
	"strconv"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// rlpFloatOracle returns the exact break-point oracle (rvBreakOracle) of
// the float realization (A, B, C, D) with L delay poles at 0, the
// numerator of C(sI−A)⁻¹B cut to the intended degree spDeg (RootLocus
// trims the rounding-noise rest by the relative degree): the c = 0
// answer, exact for the data as given.
func rlpFloatOracle(n int, a, b, c []float64, d float64, L, spDeg int) rvOracle {
	num, den := rvTrimmedTF(a, b, c, d, n, L, spDeg)
	return rvBreakOracle(num, den)
}

// rlpFloatBreak fails when got is not the break set of o.
func rlpFloatBreak(t *testing.T, name string, got []complex128, o rvOracle) {
	t.Helper()
	if o.Cancel {
		t.Fatalf("%s: float TF has a common factor", name)
	}
	if ok, miss, extra := o.match(got); !ok {
		t.Errorf("%s: Breakaway = %v, float-realization oracle %v (missing %v, extra %v)", name, got, o.Pts, miss, extra)
	}
}

// TestRootLocus_ExactResolvedPairs pins the wave-22 B1 repros (J7OYOY): a
// resolved complex pole pair p ± σi (41968, 41106, 40765) or zero pair
// (z1–z3) of an exact or rounded realization, which eig or the converted
// TF could not tell from a double, so the eig path listed the K < 0 point
// at the pair's real part. The exact break set of the floats has none of
// them; expected values are those of the exact oracle (rvBreakOracle) and
// of the reports.
func TestRootLocus_ExactResolvedPairs(t *testing.T) {
	for _, c := range []struct {
		name    string
		n       int
		a, b, c []float64
		want    []float64
	}{
		{"41968", 2, []float64{-17640.000001026783, -7056.000000410713, 43681.00000256696, 17472.000001026783}, []float64{104, -260}, []float64{2.5, 1}, nil},
		{"41106", 2, []float64{10677.932276982869, -16293.51609339278, 7034.5151569723, -10733.932276982869}, []float64{-13.381913165633694, -8.770655621299904}, []float64{8.770655621299904, -13.381913165633694}, nil},
		{"40765", 3, []float64{2060.452070248197, 9378.968002511258, -2672.7249612548258, 844.6448471513086, 3849.362243862674, -1097.3297356811763, 4629.703867219536, 21073.342660288235, -6007.8143141108685}, []float64{571.1828038580064, 250.95023877213245, 1306.4793319211428}, []float64{0.2067053821216362, 0.940916767295054, -0.2682322166052647}, []float64{-0.71090}},
		{"z1", 3, []float64{-295.375, 36.234375, -75.46875, -892.125, 104.578125, -234.15625, 649.4375, -76.8984375, 169.296875}, []float64{261.5000000026266, 810.5000000078799, -624.7500000065666}, []float64{1, -0.125, 0.25}, []float64{-7.15111}},
		{"z2", 3, []float64{-703.078125, -182.046875, -1493.2265625, -489.47265625, -127.921875, -1042.052734375, 390.0625, 101.15625, 828.75}, []float64{2209.187634713948, 1101.375072285533, -1175.125072285533}, []float64{9.625, 2.5, 20.4375}, nil},
		{"z3", 3, []float64{740.0471469311776, 2386.034332743993, -252.9268581833637, 56.94956678871047, 182.10933708545218, -19.52173492933446, 2737.928380593127, 8813.353487387623, -936.28148401663}, []float64{836.080256377855, -27.571319725482912, 2183.2940924414193}, []float64{2.3664502535765193, 7.599122070015483, -0.8092242435899555}, []float64{-1.22221}},
	} {
		res, err := RootLocus(rlpSys(t, c.n, c.a, c.b, c.c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpBreak(t, c.name, res.Breakaway, c.want, 1e-5)
		rlpFloatBreak(t, c.name, res.Breakaway, rlpFloatOracle(c.n, c.a, c.b, c.c, 0, 0, c.n-1))
	}
}

// TestRootLocus_ExactMultiplePoles: an exact m-fold pole of a companion
// realization is one K = 0 break point at its exact value, whatever eig's
// eps^(1/m) split (base gave −0.99999 for 1/(s+1)⁴, PAOH7X w17 #63, and
// nothing for m = 3); −0.5 is the K > 0 stationary point of 1/(s(s+1)).
func TestRootLocus_ExactMultiplePoles(t *testing.T) {
	for m := 2; m <= 7; m++ {
		for _, p := range []float64{-1, -0.375, 0, 3} {
			roots := make([]complex128, m)
			for i := range roots {
				roots[i] = complex(p, 0)
			}
			dr := rlswPolyFromRoots(roots)
			den := make([]float64, len(dr))
			for i, x := range dr {
				den[i] = rlswF(x)
			}
			for _, k := range []float64{1, -2.5} {
				a, b, c := rlpCompanion(den, []float64{k})
				res, err := RootLocus(rlpSys(t, m, a, b, c, 0, 0), []float64{0, 1})
				if err != nil {
					t.Fatal(err)
				}
				if len(res.Breakaway) != 1 || res.Breakaway[0] != complex(p, 0) {
					t.Errorf("k/(s−%g)^%d, k=%g: Breakaway = %v, want [%g]", p, m, k, res.Breakaway, p)
				}
			}
		}
	}
	a, b, c := rlpCompanion([]float64{1, 1, 0}, []float64{1})
	res, err := RootLocus(rlpSys(t, 2, a, b, c, 0, 0), nil)
	if err != nil {
		t.Fatal(err)
	}
	if len(res.Breakaway) != 1 || res.Breakaway[0] != -0.5 {
		t.Errorf("1/(s(s+1)): Breakaway = %v, want [-0.5]", res.Breakaway)
	}
}

// TestRootLocus_ExactScalingInvariant: a power-of-two diagonal similarity
// changes no bit of the transfer function, so the exact break set is
// bit-identical; the eig path's depends on the realization's rounding.
func TestRootLocus_ExactScalingInvariant(t *testing.T) {
	cases := rlswGenerate()
	for i := 0; i < len(cases); i += 37 {
		c := cases[i]
		if c.L > 0 || c.n > rlExactMaxStates {
			continue
		}
		sys, err := c.system()
		if err != nil {
			t.Fatal(err)
		}
		want, err := RootLocus(sys, []float64{0, 0.5})
		if err != nil {
			t.Fatal(err)
		}
		a, b, cc := slices.Clone(c.a), slices.Clone(c.b), slices.Clone(c.c)
		for j := range c.n {
			e := (i+3*j)%11 - 5
			for k := range c.n {
				a[j*c.n+k] = math.Ldexp(a[j*c.n+k], e)
				a[k*c.n+j] = math.Ldexp(a[k*c.n+j], -e)
			}
			b[j], cc[j] = math.Ldexp(b[j], e), math.Ldexp(cc[j], -e)
		}
		got, err := RootLocus(rlpSys(t, c.n, a, b, cc, c.d, c.dt), []float64{0, 0.5})
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(got.Breakaway, want.Breakaway) {
			t.Errorf("case %d: scaled Breakaway %v, unscaled %v", i, got.Breakaway, want.Breakaway)
		}
	}
}

// TestRootLocus_ExactGateFallback: beyond rlExactMaxStates the eig path
// decides (here the textbook −0.5 of 1/(s(s+1)) padded with 12 fast
// uncoupled states), sorted like the exact path.
func TestRootLocus_ExactGateFallback(t *testing.T) {
	n := rlExactMaxStates + 2
	a := make([]float64, n*n)
	a[1], a[n+1] = 1, -1
	for i := 2; i < n; i++ {
		a[i*n+i] = -float64(1000 + i)
	}
	b, c := make([]float64, n), make([]float64, n)
	b[1], c[0] = 1, 1
	sys := rlpSys(t, n, a, b, c, 0, 0)
	if _, ok := systemExactBreakaway(sys, nil, 0); ok {
		t.Fatalf("exact path ran at n = %d", n)
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	rlpBreak(t, "padded 1/(s(s+1))", res.Breakaway, []float64{-0.5}, 1e-9)
}

// TestRootLocus_BVLIUFVerdicts pins the 44 frozen-sweep Breakaway verdicts
// 03edce8 got right only through TF noise and main (eb8d9e1) got wrong
// (BVLIUF): an exact multiple root of N'D − ND' listed twice, a negative
// numerator lead without K < 0 sign handling, and the arch-only cases dbl
// 1413 (amd64), 1452 (arm64), ext 32 (arm64), 903 (amd64); judged by the
// float-realization oracle on each arch, with no point listed twice.
func TestRootLocus_BVLIUFVerdicts(t *testing.T) {
	for _, s := range []struct {
		name  string
		cases []rlswCase
		ids   []int
	}{
		{"sw", rlswGenerate(), []int{17, 138, 166, 613, 977, 1389, 1436, 1518, 1590, 1648, 1709, 1745, 1754, 38, 74, 105, 194, 334, 372, 595, 958, 1187}},
		{"dbl", rlsdGenerate(), []int{2, 125, 325, 491, 602, 608, 856, 107, 1413, 1452}},
		{"ext", rlsxGenerate(), []int{27, 1420, 1751, 1762, 307, 364, 858, 1081, 1399, 1683, 32, 903}},
	} {
		for _, i := range s.ids {
			c := &s.cases[i]
			oi, of := c.oracles()
			ok, posed, dup, err := rlswVerdict(c, oi, of)
			if err != nil {
				t.Fatal(err)
			}
			if posed&rlswBreak == 0 || ok&rlswBreak == 0 || dup {
				sys, _ := c.system()
				res, _ := RootLocus(sys, []float64{0, 1})
				t.Errorf("%s %d: Breakaway %v (posed %v, dup %v), float-realization oracle %v", s.name, i, res.Breakaway, posed&rlswBreak != 0, dup, of.required)
			}
		}
	}
}

// TestRootLocus_ExactRootBesideDyadicRoot: a root of N'D − ND' next to an
// exact dyadic one, which root isolation splits off as an interval end
// (dbl sweep 59: stationary points −6.40869140625e-4 and −6.40973e-4;
// 1136: the double pole 0.01611328125 beside the stationary point
// 0.0161139), is found and classified.
func TestRootLocus_ExactRootBesideDyadicRoot(t *testing.T) {
	p := xpMul([]*big.Int{big.NewInt(1), big.NewInt(-21)}, []*big.Int{big.NewInt(1000), big.NewInt(-21003)})
	if ivs := xpIsolatePositive(p); len(ivs) != 2 {
		t.Fatalf("isolating intervals %v, want 2", ivs)
	}
	cs := rlsdGenerate()
	for _, i := range []int{59, 1136} {
		c := &cs[i]
		sys, err := c.system()
		if err != nil {
			t.Fatal(err)
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpFloatBreak(t, "dbl "+strconv.Itoa(i), res.Breakaway, rlpFloatOracle(c.n, c.a, c.b, c.c, c.d, c.L, c.n))
	}
}

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
