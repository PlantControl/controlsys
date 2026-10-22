package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
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
	for i, x := range bp {
		for _, y := range bp[i+1:] {
			if cmplx.Abs(x-y) <= 1e-6*math.Max(cmplx.Abs(x), 1e-300) || x == y {
				t.Errorf("%s: Breakaway %v lists %v twice", name, bp, x)
				return
			}
		}
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
// companions of k(s+z)/(s²(s+p)) keep the origin breakaway also when
// rounding splits the double pole into a real pair of about ±5e-9.
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
		if !slices.ContainsFunc(res.Breakaway, func(x complex128) bool { return cmplx.Abs(x) <= 1e-6*p }) {
			miss++
			if miss <= 3 {
				t.Errorf("p=%g z=%g k=%g A=%v: Breakaway = %v lacks 0", p, z, k, a, res.Breakaway)
			}
		}
	}
	if miss > 0 {
		t.Errorf("%d/600 rotated cases miss the origin breakaway", miss)
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
