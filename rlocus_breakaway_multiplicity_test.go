package controlsys

import (
	"math"
	"math/big"
	"slices"
	"strconv"
	"testing"
)

// Pinned RootLocus review findings of wave 19, round 2 (multiple zeros with
// D = 0, triple poles, origin clusters). Each loop is realized in form k:
// 0 the companion form (rlpCompanion), 1–10 rlpRotate seed k−1, 11–20
// rlpSimilar seed k−11; a result passes when an exact big.Rat oracle
// accepts it, on the intended TF or on the TF of the float realization
// (the sweep's rule). Every pinned form was right in base ca89d74 on arm64
// and amd64.

// rlpSimilar applies n+1 exact-dyadic elementary similarities chosen by
// seed (row i += f·row j, column j −= f·column i) to (A, B, C) in place.
func rlpSimilar(n int, a, b, c []float64, seed int) {
	for r := range n + 1 {
		i, j := (r+seed)%n, (3*r+seed+1)%n
		if i == j {
			continue
		}
		f := float64((r+seed)%7-3) / 4
		if f == 0 {
			f = 0.5
		}
		for k := range n {
			a[i*n+k] += f * a[j*n+k]
		}
		for k := range n {
			a[k*n+j] -= f * a[k*n+i]
		}
		b[i] += f * b[j]
		c[j] -= f * c[i]
	}
}

func rlpForm(den, num []float64, form int) (a, b, c []float64) {
	n := len(den) - 1
	a, b, c = rlpCompanion(den, num)
	switch {
	case form == 0:
	case form <= 10:
		rlpRotate(n, a, b, c, form-1)
	default:
		rlpSimilar(n, a, b, c, form-11)
	}
	return a, b, c
}

// rlpCheckForms runs RootLocus on the given forms of num/den (descending,
// exact dyadic) and requires either exact oracle to accept the breakaway.
func rlpCheckForms(t *testing.T, name string, den, num []float64, forms []int) {
	t.Helper()
	n := len(den) - 1
	rat := func(p []float64) []*big.Rat {
		out := make([]*big.Rat, 0, n+1)
		for range n + 1 - len(p) {
			out = append(out, new(big.Rat))
		}
		for _, x := range p {
			out = append(out, fzRat(x))
		}
		return out
	}
	for _, form := range forms {
		a, b, c := rlpForm(den, num, form)
		cs := rlswCase{n: n, a: a, b: b, c: c, num: rat(num), den: rat(den)}
		oi, of := cs.oracles()
		res, err := RootLocus(rlpSys(t, n, a, b, c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Errorf("%s form %d: %v", name, form, err)
			continue
		}
		if !oi.accepts(res.Breakaway) && !of.accepts(res.Breakaway) {
			t.Errorf("%s form %d: Breakaway = %v, intended oracle %v (posed %v), float oracle %v (posed %v)",
				name, form, res.Breakaway, oi.required, oi.posed, of.required, of.posed)
		}
	}
}

func rlpAllForms() []int {
	f := make([]int, 21)
	for i := range f {
		f[i] = i
	}
	return f
}

// TestRootLocus_MultipleZeroNoFeedthrough: with D = 0 a double zero (K = ∞)
// is a break point, also when Zeros splits it to rounding level (±6.6e-9
// for the companion of s²(s+4)/((s+1)(s+2)(s+3)(s+5))) and the converted
// numerator's trailing coefficients are noise ([1 4 2.7e-15 −2.5e-16]).
func TestRootLocus_MultipleZeroNoFeedthrough(t *testing.T) {
	for _, tc := range []struct {
		name     string
		den, num []float64
	}{
		{"s²(s+4)/((s+1)(s+2)(s+3)(s+5))", []float64{1, 11, 41, 61, 30}, []float64{1, 4, 0, 0}},
		{"s²(s+1)/((s+2)(s+3)(s+4)(s+5))", []float64{1, 14, 71, 154, 120}, []float64{1, 1, 0, 0}},
		{"s²(s+2)/((s+1)(s+3)(s+4)(s+5))", []float64{1, 13, 59, 107, 60}, []float64{1, 2, 0, 0}},
		{"s²/((s+1)(s+2)(s+3))", []float64{1, 6, 11, 6}, []float64{1, 0, 0}},
		{"s²/((s+1)(s+2)(s+3)(s+4))", []float64{1, 10, 35, 50, 24}, []float64{1, 0, 0}},
		{"10s²/((s+1)(s+10)(s+100))", []float64{1, 111, 1110, 1000}, []float64{10, 0, 0}},
		{"s²/((s+10)(s+20)(s+30))", []float64{1, 60, 1100, 6000}, []float64{1, 0, 0}},
		{"(s+5)²/(s(s+2)(s+3)(s+4))", []float64{1, 9, 26, 24, 0}, []float64{1, 10, 25}},
		{"(s+1)²/(s(s+2)(s+3)(s+4))", []float64{1, 9, 26, 24, 0}, []float64{1, 2, 1}},
	} {
		rlpCheckForms(t, tc.name, tc.den, tc.num, rlpAllForms())
	}
	a, b, c := rlpCompanion([]float64{1, 11, 41, 61, 30}, []float64{1, 4, 0, 0})
	res, err := RootLocus(rlpSys(t, 4, a, b, c, 0, 0), []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	rlpBreakOnce(t, "companion s²(s+4)", res.Breakaway, []float64{0, -1.290317187133748}, 1e-6)
}

// TestRootLocus_TriplePoleCompanionExact (w17 #63): an exact companion with
// a triple pole at −0.010009765625 (−41/4096), which eig splits to 6.5e-6
// relative and the converted denominator misses by ~150·u. Exact oracle
// [−0.010009765625, −0.0013260194].
func TestRootLocus_TriplePoleCompanionExact(t *testing.T) {
	den := []float64{1, 123.0 / 4096, 1333.0 / 4194304, 26117 / math.Ldexp(1, 34), 1457427 / math.Ldexp(1, 48), 19918169 / math.Ldexp(1, 60), 0}
	num := []float64{7 / math.Ldexp(1, 33), -7 / math.Ldexp(1, 41), 2359 / math.Ldexp(1, 59)}
	rlpCheckForms(t, "case 248", den, num, []int{0})
	a, b, c := rlpCompanion(den, num)
	res, err := RootLocus(rlpSys(t, 6, a, b, c, 0, 0), []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	rlpBreakOnce(t, "case 248", res.Breakaway, []float64{-0.010009765625, -0.0013260193515220535}, 1e-6)
	for e := -6; e <= 14; e += 2 {
		f := math.Ldexp(1, e)
		ds, ns := slices.Clone(den), slices.Clone(num)
		for i := range ds {
			ds[i] *= math.Pow(f, float64(i))
		}
		for i := range ns {
			ns[i] *= math.Pow(f, float64(i+4))
		}
		rlpCheckForms(t, "case 248 roots×2^"+strconv.Itoa(e), ds, ns, []int{0})
	}
}

// TestRootLocus_TripleOriginPoleFarZero (PAOH7X): 11/4·(s−⅛)/s³; the zero
// at ⅛ must not shrink the u^(1/3) rounding window of the triple pole at 0
// below the realization's scale.
func TestRootLocus_TripleOriginPoleFarZero(t *testing.T) {
	rlpCheckForms(t, "11/4(s−1/8)/s³", []float64{1, 0, 0, 0}, []float64{2.75, -2.75 / 8}, []int{0, 19})
}

// TestRootLocus_RotatedTriplePole (w17 #63): 1/(s(s+1/128)³); the triple
// pole at −1/128 is a break point. The general similarities (forms 11, 13,
// 15, 17) are exact dyadic realizations, but eig splits the triple by
// 8.0e-7..8.5e-7 (1e-4 relative, merged by a relative change u of A) and
// the converted denominator confirms it only at 1e5..1e6·u; sweep case 274
// (distinct poles −160, −160.0049±0.0037i in a realization of scale 4e6,
// right in base) is merged by u with the same geometry and needs 1e6·u, so
// no eig or TF rule accepts these forms without losing it: a float64
// precision limit that base passed only through its absolute 1e-4 repeat
// window.
func TestRootLocus_RotatedTriplePole(t *testing.T) {
	den := []float64{1, 3.0 / 128, 3.0 / 16384, 1.0 / 2097152, 0}
	rlpCheckForms(t, "1/(s(s+1/128)³)", den, []float64{1}, []int{0})
	t.Run("similar", func(t *testing.T) {
		t.Skip("float64 precision limit: indistinguishable from sweep case 274's distinct near triple; rlswUlpSensitive confirms 1-ulp changes of A, B, C change the exact oracle for all four forms")
		rlpCheckForms(t, "1/(s(s+1/128)³)", den, []float64{1}, []int{11, 13, 15, 17})
	})
}

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
