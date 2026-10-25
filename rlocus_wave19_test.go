package controlsys

import (
	"math"
	"math/big"
	"testing"
)

// Pinned RootLocus review findings of wave 19 (B1, B2). Expected values are
// exact big.Rat oracles on the given float coefficients (rlpExactOracle,
// newRlswOracle) or the closed-form cancelled transfer function.

// rlpBreakOnce is rlpBreak plus a count check: no point is reported twice.
func rlpBreakOnce(t *testing.T, name string, got []complex128, want []float64, tol float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s: Breakaway = %v, want %v (each once)", name, got, want)
		return
	}
	rlpBreak(t, name, got, want, tol)
}

// TestRootLocus_FeedthroughExactDoubleZero (B1): G = 2(s−⅛)²(s+9/64)(s+½)
// (s−1/16)(s+5/32)/((s+13/16)(s+3/64)((s−9/64)²+(15/32)²)(s+21/64)(s+9/16))
// in a controllable companion with exact dyadic coefficients and D = 2. N
// has an exact double root at ⅛ (K = ∞), a break point; base ca89d74
// reported it. Every root scaled by f = 2^k (exact) keeps it at f/8.
func TestRootLocus_FeedthroughExactDoubleZero(t *testing.T) {
	a0 := []float64{-1.46875, -0.7353515625, -0.33377838134765625, -0.18951576948165894, -0.044135548174381256, -0.0016835869755595922}
	c0 := []float64{-1.96875, -1.5654296875, -0.7042388916015625, -0.37680375576019287, -0.0879172831773758, -0.003388631623238325}
	want := []float64{-0.6317008224543738, -0.1486497594375233, 0.125}
	for _, f := range []float64{0.125, 0.25, 0.5, 1, 2, 4, 8, 16} {
		const n = 6
		a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
		b[0] = 1
		for i := 1; i < n; i++ {
			a[i*n+i-1] = 1
		}
		g := 1.0
		for j := range n {
			g *= f
			a[j], c[j] = a0[j]*g, c0[j]*g
		}
		o := rlpExactOracle(t, n, a, b, c, 2)
		res, err := RootLocus(rlpSys(t, n, a, b, c, 2, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		name := "f=" + fmtG(f)
		if !o.accepts(res.Breakaway) {
			t.Errorf("%s: Breakaway = %v, exact oracle %v", name, res.Breakaway, o.required)
		}
		w := make([]float64, len(want))
		for i, x := range want {
			w[i] = x * f
		}
		rlpBreakOnce(t, name, res.Breakaway, w, 1e-6)
	}
}

// TestRootLocus_ZeroCancelsMultiplePole (B2): a zero at a multiple pole
// lowers its net multiplicity (minimal-TF semantics of w17 #58 / w18 #28),
// so a net simple pole there is no break point and a net double pole is
// reported once.
func TestRootLocus_ZeroCancelsMultiplePole(t *testing.T) {
	r := func(p, q int64) *big.Rat { return big.NewRat(p, q) }
	z := new(big.Rat)
	for _, tc := range []struct {
		name     string
		n        int
		a, b, c  []float64
		dt       float64
		L        int
		num, den []*big.Rat
		want     []float64
	}{
		// z/((z−½)(z−⅘))·z⁻² = 1/(z(z−½)(z−⅘)): N'D − ND' roots ⅕ (K < 0), ⅔.
		{"discrete L=2", 2, []float64{1.3, -0.4, 1, 0}, []float64{1, 0}, []float64{1, 0}, 0.1, 2,
			[]*big.Rat{r(1, 1)}, []*big.Rat{r(1, 1), r(-13, 10), r(2, 5), z}, []float64{2.0 / 3}},
		// L = 3 leaves a net double pole at 0.
		{"discrete L=3", 2, []float64{1.3, -0.4, 1, 0}, []float64{1, 0}, []float64{1, 0}, 0.1, 3,
			[]*big.Rat{r(1, 1)}, []*big.Rat{r(1, 1), r(-13, 10), r(2, 5), z, z}, nil},
		// s/(s²(s+1)) = 1/(s(s+1)).
		{"continuous s/(s²(s+1))", 3, []float64{-1, 0, 0, 1, 0, 0, 0, 1, 0}, []float64{1, 0, 0}, []float64{0, 1, 0}, 0, 0,
			[]*big.Rat{r(1, 1)}, []*big.Rat{r(1, 1), r(1, 1), z}, []float64{-0.5}},
		// s²/(s²(s+1)) = 1/(s+1).
		{"continuous s²/(s²(s+1))", 3, []float64{-1, 0, 0, 1, 0, 0, 0, 1, 0}, []float64{1, 0, 0}, []float64{1, 0, 0}, 0, 0,
			[]*big.Rat{r(1, 1)}, []*big.Rat{r(1, 1), r(1, 1)}, nil},
		// Companion of (s+1)/((s+1)³(s+3)) = 1/((s+1)²(s+3)): net double pole at −1.
		{"companion (s+1)/((s+1)³(s+3))", 4, []float64{0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, -3, -10, -12, -6}, []float64{0, 0, 0, 1}, []float64{1, 1, 0, 0}, 0, 0,
			[]*big.Rat{r(1, 1)}, []*big.Rat{r(1, 1), r(5, 1), r(7, 1), r(3, 1)}, []float64{-1}},
	} {
		sys := rlpSys(t, tc.n, tc.a, tc.b, tc.c, 0, tc.dt)
		if tc.L > 0 {
			if err := sys.SetInputDelay([]float64{float64(tc.L)}); err != nil {
				t.Fatal(err)
			}
		}
		o := newRlswOracle(tc.num, tc.den)
		if !o.posed {
			t.Fatalf("%s: oracle ill posed", tc.name)
		}
		want := tc.want
		if want == nil {
			want = o.required
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		if !o.accepts(res.Breakaway) {
			t.Errorf("%s: Breakaway = %v, exact oracle %v", tc.name, res.Breakaway, o.required)
		}
		rlpBreakOnce(t, tc.name, res.Breakaway, want, 1e-9)
	}
}

// TestRootLocus_BalancedRotatedSlowLoop (w17 #65 geometry): six poles near
// 1e-4 with a double pole at −1.5σ (−3.66e-4) and a zero at −6σ (−1.46e-3),
// σ = 2^−12, gain σ⁵: the companion balanced by exact power-of-two scaling
// and then rotated. Every Markov parameter is small in absolute terms, so
// a pessimistic significance bound trims the numerator and loses every
// breakaway; the result must match the exact oracle of the intended TF or
// of the float realization (rotation rounding splits the double pole).
func TestRootLocus_BalancedRotatedSlowLoop(t *testing.T) {
	sg := math.Ldexp(1, -12)
	poles := []complex128{complex(-1.5*sg, 0), complex(-1.5*sg, 0), complex(-0.5*sg, 0), complex(-sg, 0), complex(-0.25*sg, 0.5*sg), complex(-0.25*sg, -0.5*sg)}
	const n = 6
	den := rlswPolyFromRoots(poles)
	k := math.Pow(sg, 5)
	num := []*big.Rat{fzRat(k), fzRat(6 * sg * k)}
	for seed := range 6 {
		a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
		for i := range n - 1 {
			a[i*n+i+1] = sg
		}
		for j := range n {
			a[(n-1)*n+j] = -rlswF(den[n-j]) / math.Pow(sg, float64(n-1-j))
			if m := len(num) - 1 - j; m >= 0 {
				c[j] = rlswF(num[m]) * math.Pow(sg, float64(j))
			}
		}
		b[n-1] = math.Pow(sg, -(n - 1))
		rlpRotate(n, a, b, c, seed)
		cs := rlswCase{n: n, a: a, b: b, c: c, num: num, den: den}
		oInt, oFlt := cs.oracles()
		if !oInt.posed {
			t.Fatal("intended oracle ill posed")
		}
		res, err := RootLocus(rlpSys(t, n, a, b, c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		if !oInt.accepts(res.Breakaway) && !oFlt.accepts(res.Breakaway) {
			t.Errorf("seed %d: Breakaway = %v, exact oracles %v / %v", seed, res.Breakaway, oInt.required, oFlt.required)
		}
	}
}
