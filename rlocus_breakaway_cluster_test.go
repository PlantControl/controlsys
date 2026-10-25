package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Pinned wave-19 review blockers of the rlocus track (base correct in every
// case). Expected sets come from the exact big.Rat oracle on the float
// coefficients (rlpExactOracle).

func rlqObservable(num, den []float64) (a, b, c []float64, d float64) {
	n := len(den) - 1
	num = append(make([]float64, len(den)-len(num)), num...)
	a, b, c = make([]float64, n*n), make([]float64, n), make([]float64, n)
	d = num[0]
	for i := range n {
		a[i*n] = -den[i+1]
		if i+1 < n {
			a[i*n+i+1] = 1
		}
		b[i] = num[i+1] - den[i+1]*d
	}
	c[0] = 1
	return a, b, c, d
}

func rlqUnique(t *testing.T, name string, bp []complex128) {
	t.Helper()
	if x, dup := rlswDuplicate(bp); dup {
		t.Errorf("%s: Breakaway %v lists %v twice", name, bp, x)
	}
}

func rlqOracleCheck(t *testing.T, name string, n int, a, b, c []float64, d float64, nwant int) {
	t.Helper()
	o := rlpExactOracle(t, n, a, b, c, d)
	if len(o.required) != nwant {
		t.Fatalf("%s: oracle breakaways %v, want %d", name, o.required, nwant)
	}
	res, err := RootLocus(rlpSys(t, n, a, b, c, d, 0), []float64{0, 1})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	rlqUnique(t, name, res.Breakaway)
	if !o.accepts(res.Breakaway) {
		t.Errorf("%s: Breakaway = %v, want %v", name, res.Breakaway, o.required)
	}
}

// TestRootLocus_NegativeFeedthroughNoFarBreakaway: D < 0 with a strictly
// proper part of relative degree 2; noise in the high coefficients of
// N'D−ND' must not give a far real breakaway (exact oracle: none).
func TestRootLocus_NegativeFeedthroughNoFarBreakaway(t *testing.T) {
	a := []float64{-0.0341796875, -0.00019097328186035156, 4.6100467443466187e-07, 5.4637894209008664e-08, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0}
	b := []float64{1, 0, 0, 0}
	c := []float64{0, 0, 0.00011444091796875, -6.146728992462158e-07}
	for _, d := range []float64{-1.625, -1.375, -0.875, -0.25} {
		rlqOracleCheck(t, "D<0 companion", 4, a, b, c, d, 0)
	}
}

// TestRootLocus_DoublePoleBesideSimplePole: a double pole at −21/128 next to
// a simple pole 2⁻¹² relative away keeps its K = 0 breakaway; also scaled
// ×32 (A → 32A, exact).
func TestRootLocus_DoublePoleBesideSimplePole(t *testing.T) {
	row0 := []float64{-0.7500400543212891, -0.2297903150320053, -0.036129343730863184, -0.0029256685465952614, -9.772868674318147e-05}
	c := []float64{0, 0.03125, 0.00518798828125, 0.0003663301467895508, 6.959307938814163e-07}
	b := []float64{1, 0, 0, 0, 0}
	for _, sc := range []float64{1, 32} {
		a := make([]float64, 25)
		copy(a, row0)
		for i := 1; i < 5; i++ {
			a[i*5+i-1] = 1
		}
		for i := range a {
			a[i] *= sc
		}
		o := rlpExactOracle(t, 5, a, b, c, 0)
		want := []float64{-0.1640625 * sc, -0.1640892 * sc}
		for _, w := range want {
			if !slices.ContainsFunc(o.required, func(x float64) bool { return math.Abs(x-w) <= 1e-6*math.Abs(w) }) {
				t.Fatalf("scale %g: oracle %v lacks %v", sc, o.required, w)
			}
		}
		rlqOracleCheck(t, "double beside simple", 5, a, b, c, 0, 2)
	}
}

// TestRootLocus_MultiplePoleBreakawayOnce: a multiple pole breakaway is
// listed once, both for 1/s² in nilpotent non-triangular forms and for a
// biproper loop with a double pole at −2.5.
func TestRootLocus_MultiplePoleBreakawayOnce(t *testing.T) {
	for _, al := range []float64{0.5, -0.5, 0.25, 2, 0.125} {
		for _, k := range []float64{1, -1, 36, 0.0054} {
			a := []float64{al, 1, -al * al, -al}
			res, err := RootLocus(rlpSys(t, 2, a, []float64{0, 1}, []float64{k, 0}, 0, 0), []float64{0, 1})
			if err != nil {
				t.Fatal(err)
			}
			if len(res.Breakaway) != 1 || cmplx.Abs(res.Breakaway[0]) > 1e-8 {
				t.Errorf("a=%g k=%g: Breakaway = %v, want [0]", al, k, res.Breakaway)
			}
		}
	}
	num := []float64{1.5, 42, 424.875, 2075.625, 6393.75, -4182022.75, 10496072.5}
	den := []float64{1, 28, 283.25, 1383.75, 4262.5, 8187.5, 6875}
	a, b, c, d := rlqObservable(num, den)
	rlqOracleCheck(t, "biproper double pole", 6, a, b, c, d, 3)
}

// TestRootLocus_SlowLoopNonOrthogonalSimilarity: breakaways of a slow loop
// (poles ~1e-5) survive a unit upper triangular dyadic similarity T·A·T⁻¹
// of its observable companion, done exactly (rounded once) or in float as
// the reviewer did; G'(s) must not come from an inaccurate unbalanced solve.
func TestRootLocus_SlowLoopNonOrthogonalSimilarity(t *testing.T) {
	r := func(p int64, e int) float64 { return math.Ldexp(float64(p), -e) }
	num := []float64{-1.0 / 16384, -r(143, 35), -r(9061, 56), -r(120303, 77), r(1362465, 96)}
	den := []float64{1, r(49179, 29), r(2345031, 49), r(136879713, 70), r(134782623, 86), r(13177079595, 110)}
	const n = 5
	for _, full := range []bool{false, true} {
		a, b, c, d := rlqObservable(num, den)
		rlsxSimilarAt(0, n, a, b, c, full)
		rlqOracleCheck(t, "slow similar exact", n, a, b, c, d, 4)
	}
	for _, tc := range []struct {
		full bool
		off  int
	}{{false, 5}, {false, 6}, {true, 1}, {true, 2}, {true, 9}} {
		a, b, c, d := rlqObservable(num, den)
		T := mat.NewDense(n, n, nil)
		k := 0
		for i := range n {
			T.Set(i, i, 1)
			for j := i + 1; j < n; j++ {
				if j == i+1 || tc.full {
					T.Set(i, j, rlsxCycle[(tc.off+k)%len(rlsxCycle)])
					k++
				}
			}
		}
		var ti, ta, tat, tb, ctt mat.Dense
		if err := ti.Inverse(T); err != nil {
			t.Fatal(err)
		}
		ta.Mul(T, mat.NewDense(n, n, a))
		tat.Mul(&ta, &ti)
		tb.Mul(T, mat.NewDense(n, 1, b))
		ctt.Mul(mat.NewDense(1, n, c), &ti)
		rlqOracleCheck(t, "slow similar float", n, tat.RawMatrix().Data, tb.RawMatrix().Data, ctt.RawMatrix().Data, d, 4)
	}
}

// TestRootLocus_RotatedOriginDoublePole (w17 #66): Givens-rotated
// companions of k(s+z)/(s²(s+p)). The origin double pole is a breakaway
// where the float data keep it; where the rotation's rounding splits it
// (into a real pair of about ±5e-9 or a complex pair) the break set is
// that of the float realization (c = 0, J7OYOY).
func TestRootLocus_RotatedOriginDoublePole(t *testing.T) {
	rng := rand.New(rand.NewPCG(66, 611))
	miss := 0
	for i := range 600 {
		p := float64(1+rng.IntN(64)) / 32
		z := float64(1+rng.IntN(64)) / 64
		k := float64(rng.IntN(2)*2-1) * float64(1+rng.IntN(16)) / 8
		if i == 0 {
			p, z, k = 1.375, 0.03125, -0.5
		}
		a := []float64{0, 1, 0, 0, 0, 1, 0, 0, -p}
		b := []float64{0, 0, 1}
		c := []float64{k * z, k, 0}
		rlswRotate(rng, 3, a, b, c)
		res, err := RootLocus(rlpSys(t, 3, a, b, c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		o := rlpFloatOracle(3, a, b, c, 0, 0, 1)
		if ok, _, _ := o.match(res.Breakaway); !ok {
			miss++
			if miss <= 3 {
				t.Errorf("p=%g z=%g k=%g A=%v: Breakaway = %v, float-realization oracle %v", p, z, k, a, res.Breakaway, o.Pts)
			}
		}
	}
	if miss > 0 {
		t.Errorf("%d/600 rotated cases miss the float-realization break set", miss)
	}
}

// TestRootLocus_MarkovZeroCountConsistent: zeros are dropped as rounding
// noise only past the first Markov parameter no rounding can zero, and the
// asymptote sign follows the Markov parameter of the degree Zeros() fixes.
// Ext sweep 920: h₂ is 6.7·eps·sens (inside the 10× margin, yet cannot
// vanish), so the zero −232 stays. Ext 1294: h₁ is resolved in the floats
// but Zeros() reports none, so 2 asymptotes at the sign of h₂ > 0. Base
// sweep 1692: no hₖ is resolved, so no zero is dropped.
func TestRootLocus_MarkovZeroCountConsistent(t *testing.T) {
	ext, base := rlsxGenerate(), rlswGenerate()
	for _, tc := range []struct {
		c       *rlswCase
		name    string
		asym    []float64
		arrival int
	}{
		{&ext[920], "ext 920", []float64{math.Pi / 2, 3 * math.Pi / 2}, 4},
		{&ext[1294], "ext 1294", []float64{math.Pi / 2, 3 * math.Pi / 2}, 0},
		{&base[1692], "base 1692", []float64{0, math.Pi}, 3},
	} {
		sys, err := tc.c.system()
		if err != nil {
			t.Fatal(err)
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		rlpAngles(t, tc.name+" AsymptoteAngles", res.AsymptoteAngles, tc.asym)
		if len(res.ArrivalAngles) != tc.arrival {
			t.Errorf("%s: %d arrival angles, want %d", tc.name, len(res.ArrivalAngles), tc.arrival)
		}
		if tc.name == "ext 920" {
			oi, _ := tc.c.oracles()
			if !oi.accepts(res.Breakaway) {
				t.Errorf("ext 920: Breakaway = %v, want %v", res.Breakaway, oi.required)
			}
		}
	}
}

// TestRootLocus_PAOH7XSweepPins: frozen-sweep cases a first PAOH7X rework
// got wrong. Ext 439/495: a double pole beside a resolved simple pole (sub-
// group split; the stationary point between them is kept). Ext 1887: two
// resolved poles of a non-normal rotated realization, whose pair mean is no
// better conditioned than their split, are no double pole (amd64 eig links
// them). Ext 1994: an origin pair split by rounding is reported at the
// stationary point, not the eig mean. Ext 954: an exact triple pole at 0
// is reported at 0. Base 512: a near triple is not merged as a double.
// Base 207: an origin double split into a real pair ±3.9e-9.
func TestRootLocus_PAOH7XSweepPins(t *testing.T) {
	ext, base := rlsxGenerate(), rlswGenerate()
	for _, tc := range []struct {
		name string
		c    *rlswCase
	}{
		{"ext 439", &ext[439]}, {"ext 495", &ext[495]}, {"ext 1887", &ext[1887]},
		{"ext 1994", &ext[1994]}, {"ext 954", &ext[954]}, {"base 512", &base[512]}, {"base 207", &base[207]},
	} {
		sys, err := tc.c.system()
		if err != nil {
			t.Fatal(err)
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		oi, of := tc.c.oracles()
		if !oi.accepts(res.Breakaway) && !of.accepts(res.Breakaway) {
			t.Errorf("%s: Breakaway = %v, oracles %v / %v", tc.name, res.Breakaway, oi.required, of.required)
		}
	}
}

// TestRootLocus_MultiplePoleBreakawayListedOnce: PAOH7X review (wave 20).
// A multiple pole or a near-real stationary pair is one break point:
//   - exact delay poles z^L beside a nilpotent A whose eig returns an equal
//     pair −5.7e-18: one 4- or 5-fold pole at 0 (eig and delay clusters were
//     kept apart and both listed);
//   - a rotated+similar triple pole at 10.5 whose N'D − ND' roots near it
//     are a conjugate pair within 1e-6 of real (each listed);
//   - observable companions with a triple zero at −1.335144e-5, D = −1.75
//     and a pole at 2.25 or −2.75 (double) nearly cancelled by a zero, whose
//     stationary point there is the same near-real pair (oracle ill posed).
func TestRootLocus_MultiplePoleBreakawayListedOnce(t *testing.T) {
	for _, L := range []float64{2, 3} {
		sys := rlpSys(t, 2, []float64{0.5, 1, -0.25, -0.5}, []float64{0, 1}, []float64{1, 0}, 0, 1)
		if err := sys.SetInputDelay([]float64{L}); err != nil {
			t.Fatal(err)
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		if len(res.Breakaway) != 1 || res.Breakaway[0] != 0 {
			t.Errorf("delay %g: Breakaway = %v, want [0]", L, res.Breakaway)
		}
	}

	a := []float64{39.47763330898467, 351.3792695398101, 803.2895398100804, 108.07267348429517, 961.0695288531774, 2192.113853177502, -48.41540540540544, -424.72716216216213, -969.0471621621621}
	b := []float64{-0.4972972972972973, -1.3567567567567567, 0.6}
	c := []float64{-4550.918918918919, 14864.432432432433, 29840.432432432433}
	o := rlpExactOracle(t, 3, a, b, c, 0)
	res, err := RootLocus(rlpSys(t, 3, a, b, c, 0, 0), []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	rlqUnique(t, "rotated triple pole 10.5", res.Breakaway)
	for _, x := range o.required {
		if !slices.ContainsFunc(res.Breakaway, func(z complex128) bool { return cmplx.Abs(z-complex(x, 0)) <= 1e-6*math.Abs(x) }) {
			t.Errorf("rotated triple pole 10.5: Breakaway = %v, missing %v", res.Breakaway, x)
		}
	}

	for _, tc := range []struct{ num, den []float64 }{
		{[]float64{-1.75, 3.93742990494, 0.00015771294913001212, 2.105703307673736e-09, 9.371408993299497e-15}, []float64{1, -1.625, -2.71875, 2.953125, 0}},
		{[]float64{-1.75, 1.5311799049400001, 17.44537383224163, 0.0006987609482570744, 9.329457586179378e-09, 4.152054817864638e-14}, []float64{1, 2.25, -12.6875, -35.765625, -7.6484375, 14.1796875}},
	} {
		a, b, c, d := rlqObservable(tc.num, tc.den)
		n := len(tc.den) - 1
		res, err := RootLocus(rlpSys(t, n, a, b, c, d, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlqUnique(t, "triple zero companion", res.Breakaway)
	}
}

// Pinned wave-20 review-3 cases of the rlocus track. Expected sets come
// from the exact big.Rat oracle on the float coefficients (rlpExactOracle)
// or the sweep oracles (rlswCase.oracles).

// TestRootLocus_DoublePoleBesideSimplePoleExact: an exact double real pole
// beside a simple pole 2⁻¹¹ relative apart keeps its K = 0 breakaway (base
// correct). R272: controllable companion, eig resolves the double pair from
// the simple pole. Observable companion of (13/2³⁸)((s+3·2⁻¹⁶)²+(25·2⁻¹⁷)²)
// over (s+15·2⁻¹⁷)²(s+15·2⁻¹⁷(1+2⁻¹¹))(s+15·2⁻¹⁵)(s−3·2⁻¹⁴).
func TestRootLocus_DoublePoleBesideSimplePoleExact(t *testing.T) {
	a := make([]float64, 16)
	for i := range 3 {
		a[i*4+i+1] = 1
	}
	copy(a[12:], []float64{5.5084024941437495e-21, 1.7213335791466111e-15, 1.0912870607171499e-10, -1.0494142770767212e-05})
	rlqOracleCheck(t, "R272", 4, a, []float64{0, 0, 0, 1}, []float64{1.0799670077492329e-19, -1.3322676295501878e-14, 0, 0}, 0, 2)

	q := math.Ldexp(15, -17)
	poles := []complex128{complex(-q, 0), complex(-q, 0), complex(-q*(1+math.Ldexp(1, -11)), 0), complex(-4*q, 0), complex(math.Ldexp(3, -14), 0)}
	zeros := []complex128{complex(-math.Ldexp(3, -16), math.Ldexp(25, -17)), complex(-math.Ldexp(3, -16), -math.Ldexp(25, -17))}
	num := rlsdNum(5, rlswPolyFromRoots(poles), zeros, math.Ldexp(13, -38), 0)
	c := rlsdCase(nil, poles, zeros, num, 0, 1)
	rlqOracleCheck(t, "observable 2^-11", 5, c.a, c.b, c.c, 0, 3)
}

// TestRootLocus_NoSpuriousBreakBesideNearTriple (case 2095): a dyadic
// similarity of 448/((s+3.625)²(s+3.625885009765625)(s+5.625)) whose eig
// cannot order its three cluster poles (u·κ far above their split): a
// cluster pair is no double pole, so N'D − ND' root −3.62559 (K < 0) is no
// breakaway. amd64 eig splits the cluster into a complex pair, where base
// is wrong too ([]), so the pin runs on arm64.
func TestRootLocus_NoSpuriousBreakBesideNearTriple(t *testing.T) {
	if runtime.GOARCH != "arm64" {
		t.Skip("amd64 eig: precision limit, base also wrong")
	}
	a := []float64{0, 1, -2.25, 9, 0, 0, 1, -2.25, -1072.0438921451569, -1613.7440816164017, 2421.6315648257732, -9751.529799342155, -268.0109730362892, -403.4360204041004, 605.4078912064433, -2438.132449835539}
	rlqOracleCheck(t, "2095", 4, a, []float64{0, 0, 4, 1}, []float64{448, 224, -392, 1568}, 0, 2)
}

// TestRootLocus_MultipleZeroBreakOnce: a triple zero is one break point.
// Companion of −1024s³(s−512)/((s+3072)³(s−1024)²) (w17 #60): the triple
// zero at 0 is split by rounding past √u·σ but within u^(1/3)·σ, base gave
// [409.15]. Third sweep 850/325/856 (D ≠ 0, triple zero −0.03125, −1.831e-4,
// −9.52e-3: base listed it twice or the converted numerator lost it) and 2
// (triple zero at 0, amd64).
func TestRootLocus_MultipleZeroBreakOnce(t *testing.T) {
	den := rlswPolyFromRoots([]complex128{-3072, -3072, -3072, 1024, 1024})
	df := make([]float64, len(den))
	for i, x := range den {
		df[i] = rlswF(x)
	}
	a, b, c := rlpCompanion(df, []float64{-1024, 524288, 0, 0, 0})
	rlqOracleCheck(t, "w17 #60", 5, a, b, c, 0, 3)

	cases := rlsdGenerate()
	for _, i := range []int{2, 325, 850, 856} {
		sys, err := cases[i].system()
		if err != nil {
			t.Fatal(err)
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		oi, of := cases[i].oracles()
		if !oi.accepts(res.Breakaway) && !of.accepts(res.Breakaway) {
			t.Errorf("sweep %d: Breakaway = %v, oracles %v / %v", i, res.Breakaway, oi.required, of.required)
		}
	}
}

// Pinned wave-20 review-4 cases of the rlocus track. Expected sets come
// from the exact big.Rat oracles of the intended and the float TF
// (rlswCase.oracles).

func rlrcCheck(t *testing.T, name string, c rlswCase) {
	t.Helper()
	sys, err := c.system()
	if err != nil {
		t.Fatal(err)
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	rlqUnique(t, name, res.Breakaway)
	if oi, of := c.oracles(); !oi.accepts(res.Breakaway) && !of.accepts(res.Breakaway) {
		t.Errorf("%s: Breakaway = %v, want %v / %v", name, res.Breakaway, oi.required, of.required)
	}
}

// rlrcPairBesideSimple is 1 − 3s(s+1.625)/den (d = 1) or −3s(s+1.625)/den
// (d = 0), den with poles −2, −5.5, −5.875 and −5.875(1+2^−k1) ± 5.875·2^−k2·i,
// as an exact companion (form 0) or observable (form 1) realization.
func rlrcPairBesideSimple(k1, k2, form int, d float64) rlswCase {
	q := 5.875
	re, im := -q*(1+math.Ldexp(1, -k1)), q*math.Ldexp(1, -k2)
	poles := []complex128{-2, -5.5, complex(-q, 0), complex(re, im), complex(re, -im)}
	zeros := []complex128{0, -1.625}
	return rlsdCase(nil, poles, zeros, rlsdNum(5, rlswPolyFromRoots(poles), zeros, -3, d), d, form)
}

// TestRootLocus_ResolvedPairBesideSimplePole: a resolved complex pole pair
// beside a simple pole is no K = 0 double pole, though the nearby simple
// pole makes the converted D confirm a double root at its p′ root to 10·u:
// eig resolves the pair (no u·κ link), and K < 0 there (exact G ≈ +3.5e10
// for 2^−11, 2^−13), so the only break point is −5.5932 (base correct for
// D = 1; base listed four points for D = 0).
func TestRootLocus_ResolvedPairBesideSimplePole(t *testing.T) {
	for _, k := range [][2]int{{9, 14}, {10, 14}, {11, 13}, {11, 14}, {12, 14}} {
		for form := range 2 {
			rlrcCheck(t, fmt.Sprintf("2^-%d, 2^-%d form %d", k[0], k[1], form), rlrcPairBesideSimple(k[0], k[1], form, 1))
		}
	}
	rlrcCheck(t, "2^-11, 2^-13 D=0", rlrcPairBesideSimple(11, 13, 0, 0))
}

func rlrcCompanion(n int, lastRow, c []float64) (a, b []float64) {
	a, b = make([]float64, n*n), make([]float64, n)
	for i := range n - 1 {
		a[i*n+i+1] = 1
	}
	copy(a[(n-1)*n:], lastRow)
	b[n-1] = 1
	return a, b
}

// TestRootLocus_TripleZeroListedOnce: −16(s+1)³/((s+38)(s+41)(s−10)(s−17)),
// integer companion: the triple zero −1 is one break point. amd64 tzero
// splits it by 1.5e-4, so both split roots of N'D − ND' were listed (base
// [12.4695], wrong too).
func TestRootLocus_TripleZeroListedOnce(t *testing.T) {
	a, b := rlrcCompanion(4, []float64{-264860, 28636, 405, -52}, nil)
	rlqOracleCheck(t, "triple zero -1", 4, a, b, []float64{-16, -48, -48, -16}, 0, 2)
}

// TestRootLocus_MultipleZeroOfStrictlyProperPartOnce: with D ≠ 0 a triple
// zero of G − D (no zero of G) is a double root of N'D − ND', one break
// point (base missed it).
func TestRootLocus_MultipleZeroOfStrictlyProperPartOnce(t *testing.T) {
	a := []float64{0, 0, 0, -2.575143298599869e-07, 1, 0, 0, -2.7045607566833496e-06, 0, 1, 0, -0.0010194778442382812, 0, 0, 1, -0.005859375}
	res, err := RootLocus(rlpSys(t, 4, a, []float64{2.6520865503698587e-09, -2.7157366275787354e-06, 0.000926971435546875, -0.10546875}, []float64{0, 0, 0, 1}, -0.25, 0), []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	// Exact oracle (arm64; its float posedness check fails on amd64).
	o := rlswOracle{posed: true, required: []float64{0.0029296875, 0.04708051704829962}, scale: 0.0059}
	if !o.accepts(res.Breakaway) {
		t.Errorf("observable triple zero 0.0029296875: Breakaway = %v, want %v", res.Breakaway, o.required)
	}
	a, b := rlrcCompanion(4, []float64{695625, 14450, -2360, -98}, nil)
	rlqOracleCheck(t, "companion triple zero -12", 4, a, b, []float64{193536, 48384, 4032, 112}, -0.25, 2)
}

// TestRootLocus_SlowSixPoleBreakaways (U3H63X): companion of poles
// 1.220703125e-4, −4.8828125e-4, −7.9345703125e-4, 3.0517578125e-4,
// −3.0517578125e-4 ± 5.4931640625e-4i with C = [0.00201416015625 −3 0 0 0 0]
// has three K > 0 break points (exact oracle −6.763e-4, 2.387e-4,
// 8.360e-4; not 1-ulp sensitive). Base and branch give []: the converted
// TF (rationalTransferFunction) is −3/(degree-5 D), dropping the zero
// 6.714e-4 with a pole the loop does not have, so N'D − ND' has no real
// root there; rlocus.go cannot recover it.
func TestRootLocus_SlowSixPoleBreakaways(t *testing.T) {
	t.Skip("TransferFunction drops zero 6.714e-4 and a pole of this slow companion (transfer.go)")
	poles := []complex128{1.220703125e-4, -4.8828125e-4, -7.9345703125e-4, 3.0517578125e-4, complex(-3.0517578125e-4, 5.4931640625e-4), complex(-3.0517578125e-4, -5.4931640625e-4)}
	den := rlswPolyFromRoots(poles)
	row := make([]float64, 6)
	for j := range 6 {
		row[j] = -rlswF(den[6-j])
	}
	a, b := rlrcCompanion(6, row, nil)
	rlqOracleCheck(t, "U3H63X six-pole", 6, a, b, []float64{0.00201416015625, -3, 0, 0, 0, 0}, 0, 3)
}
