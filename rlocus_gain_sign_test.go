package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"strconv"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Pinned RootLocus cases of tasks LP755Y, JHKPWD and PAOH7X (waves 16–18).
// Expected values come from exact big.Rat oracles on the given float
// coefficients (newRlswOracle) or from the closed-form transfer function.

func rlpSys(t *testing.T, n int, a, b, c []float64, d, dt float64) *System {
	t.Helper()
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{d}, dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// rlpExactOracle returns the exact breakaway oracle of sys's float
// coefficients (delay-free, numerator untrimmed).
func rlpExactOracle(t *testing.T, n int, a, b, c []float64, d float64) rlswOracle {
	t.Helper()
	cs := rlswCase{n: n, a: a, b: b, c: c, d: d}
	_, o := cs.oracles()
	if !o.posed {
		t.Fatalf("exact oracle ill posed for A=%v", a)
	}
	return o
}

func rlpRotatedCompanion(den, num []float64, seed int) (a, b, c []float64) {
	n := len(den) - 1
	a, b, c = make([]float64, n*n), make([]float64, n), make([]float64, n)
	for i := 0; i < n-1; i++ {
		a[i*n+i+1] = 1
	}
	for j := range n {
		a[(n-1)*n+j] = -den[n-j]
		if k := len(num) - 1 - j; k >= 0 {
			c[j] = num[k]
		}
	}
	b[n-1] = 1
	rlpRotate(n, a, b, c, seed)
	return a, b, c
}

// rlpRotate applies n+1 exact-rational Givens similarities chosen by seed
// to (A, B, C) in place.
func rlpRotate(n int, a, b, c []float64, seed int) {
	for r := range n + 1 {
		i, j := (r+seed)%n, (2*r+seed+1)%n
		if i == j {
			continue
		}
		p, q := float64(1+(r+seed)%5), float64(2+(3*r+seed)%5)
		h := p*p + q*q
		cs, sn := (p*p-q*q)/h, 2*p*q/h
		rot := func(x, y float64) (float64, float64) {
			return float64(cs*x) - float64(sn*y), float64(sn*x) + float64(cs*y)
		}
		for k := range n {
			a[i*n+k], a[j*n+k] = rot(a[i*n+k], a[j*n+k])
		}
		for k := range n {
			a[k*n+i], a[k*n+j] = rot(a[k*n+i], a[k*n+j])
		}
		b[i], b[j] = rot(b[i], b[j])
		c[i], c[j] = rot(c[i], c[j])
	}
}

func rlpAngles(t *testing.T, name string, got, want []float64) {
	t.Helper()
	if len(got) != len(want) {
		t.Errorf("%s = %v, want %v", name, got, want)
		return
	}
	for i := range got {
		if math.Abs(math.Remainder(got[i]-want[i], 2*math.Pi)) > 1e-9 {
			t.Errorf("%s = %v, want %v", name, got, want)
			return
		}
	}
}

func rlpBreak(t *testing.T, name string, got []complex128, want []float64, tol float64) {
	t.Helper()
	ok := true
	for _, z := range got {
		if imag(z) != 0 || !slices.ContainsFunc(want, func(w float64) bool { return math.Abs(real(z)-w) <= tol*math.Max(1, math.Abs(w)) }) {
			ok = false
		}
	}
	for _, w := range want {
		if !slices.ContainsFunc(got, func(z complex128) bool { return math.Abs(real(z)-w) <= tol*math.Max(1, math.Abs(w)) }) {
			ok = false
		}
	}
	if !ok {
		t.Errorf("%s: Breakaway = %v, want %v", name, got, want)
	}
}

// TestRootLocus_NegativeGainRules (O5FYRC-1): G = −1/((s+1)(s+2)) obeys the
// 0° rule under K > 0. Both branches stay real (−1 → +∞, −2 → −∞), so the
// stationary point −1.5 (K = −0.25) is no breakaway, the asymptotes are
// 0 and π and the departure angles are 0 (from −1) and π (from −2).
func TestRootLocus_NegativeGainRules(t *testing.T) {
	a, b := []float64{0, 1, -2, -3}, []float64{0, 1}
	for _, sgn := range []float64{-1, 1} {
		sys := rlpSys(t, 2, a, b, []float64{sgn, 0}, 0, 0)
		res, err := RootLocus(sys, []float64{0, 0.1, 0.25, 0.3, 1, 10})
		if err != nil {
			t.Fatal(err)
		}
		poles, err := sys.Poles()
		if err != nil {
			t.Fatal(err)
		}
		dep := make([]float64, len(poles))
		phase0 := math.Pi
		wantBreak := []float64{-1.5}
		asym := []float64{math.Pi / 2, 3 * math.Pi / 2}
		if sgn < 0 {
			phase0, wantBreak, asym = 0, nil, []float64{0, math.Pi}
		}
		for i, p := range poles {
			dep[i] = phase0 - cmplx.Phase(p-poles[1-i])
		}
		rlpBreak(t, "sign "+fmtSign(sgn), res.Breakaway, wantBreak, 1e-9)
		rlpAngles(t, "AsymptoteAngles", res.AsymptoteAngles, asym)
		rlpAngles(t, "DepartureAngles", res.DepartureAngles, dep)
		if c, ok := res.AsymptoteCentroid(); !ok || math.Abs(c+1.5) > 1e-12 {
			t.Errorf("centroid = %v,%v, want -1.5", c, ok)
		}
	}
}

func fmtSign(s float64) string {
	if s < 0 {
		return "negative"
	}
	return "positive"
}

// TestRootLocus_NegativeGainRotatedNoFarBreakaway (w18 #29): −1/((s−1)(s+1.5))
// under an orthogonal similarity. The numerator's rounding-noise leading
// coefficient must not produce a far "breakaway" near 1/eps that the 0°
// rule would put on the locus; the stationary point −0.25 needs K < 0.
func TestRootLocus_NegativeGainRotatedNoFarBreakaway(t *testing.T) {
	a := []float64{0.11523214264757396, 0.9713130155603278, 1.4713130155603276, -0.615232142647574}
	b := []float64{0.04657729683901443, 0.9989146887593404}
	c := []float64{-0.9989146887593405, 0.04657729683901452}
	res, err := RootLocus(rlpSys(t, 2, a, b, c, 0, 0), []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	rlpBreak(t, "rotated -1/((s-1)(s+1.5))", res.Breakaway, nil, 1e-9)
	rlpAngles(t, "AsymptoteAngles", res.AsymptoteAngles, []float64{0, math.Pi})
}

// TestRootLocus_NoiseZerosRelativeDegree (w17 #64): Zeros() of this
// delay-absorbed loop returns rounding-noise zeros (±7e6 with an output
// delay, four near 3.6e3 with an input delay), though the loop is
// 2.4375·z⁻³/((z−0.45)(z−1.05)(z+0.65)((z−0.3)²+0.01)), relative degree 8.
// The stationary points with K = −1/G > 0 are −0.5445 and 0.9381, and the
// triple pole at 0 (K = 0); an exact big.Rat oracle on the float
// coefficients agrees.
func TestRootLocus_NoiseZerosRelativeDegree(t *testing.T) {
	a := []float64{
		0.019797815752731235, 0.1279571753021788, 0.3811493659662884, 0.139536029114894, -0.24264502599501941,
		-0.024690736661808866, 0.3390417987388423, 0.07713472609337782, 0.02058320045505041, 0.16867929301626863,
		0.40147694836136044, 0.11464617555258745, 0.49445973150623634, -0.12279478026090808, 0.5966980275659939,
		0.1751612048549504, 0.04429700567025503, -0.11719819109801555, 0.3690601749614305, 0.18574256570034794,
		-0.21799075604346427, 0.06290279396295813, 0.6168424099007837, 0.2142588609269953, 0.2276404790407601,
	}
	b := []float64{-0.26141391328956, -0.5418493331123063, -1.6696096552907065, 0.5915593669800094, 0.7074766289020779}
	c := []float64{-142.41775793762778, 47.204252982690086, -16.13755010077696, -144.21002232543742, 66.02760815553987}
	want := []float64{-0.5444920469, 0, 0.9380893887}
	cs := rlswCase{n: 5, L: 3, a: a, b: b, c: c}
	_, o := cs.oracles()
	var near []float64
	for _, x := range o.required {
		if math.Abs(x) < 1e3 {
			near = append(near, x)
		}
	}
	slices.Sort(near)
	if !o.posed || len(near) != 3 || math.Abs(near[0]-want[0]) > 1e-9 || near[1] != 0 || math.Abs(near[2]-want[2]) > 1e-9 {
		t.Fatalf("exact oracle %+v disagrees with %v (beyond the sub-rounding far pair)", o, want)
	}
	for _, output := range []bool{true, false} {
		sys := rlpSys(t, 5, a, b, c, 0, 0.1)
		var err error
		if output {
			err = sys.SetOutputDelay([]float64{3})
		} else {
			err = sys.SetInputDelay([]float64{3})
		}
		if err != nil {
			t.Fatal(err)
		}
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpBreak(t, map[bool]string{true: "output delay", false: "input delay"}[output], res.Breakaway, want, 1e-6)
		asym := make([]float64, 8)
		for k := range asym {
			asym[k] = float64(2*k+1) * math.Pi / 8
		}
		rlpAngles(t, "AsymptoteAngles", res.AsymptoteAngles, asym)
		if len(res.ArrivalAngles) != 0 {
			t.Errorf("ArrivalAngles = %v, want none", res.ArrivalAngles)
		}
	}
}

// TestRootLocus_RotatedHighRelDegreeNegativeGain (w18 #33): G = −4.5/((s−0.375)
// ((s+3.625)²+1)((s+4.875)²+4.75²)((s−3.125)²+1.25²)), no zeros, as a
// rotated companion form: 7 asymptotes at 2kπ/7 and no real breakaway
// (exact oracle on the intended TF; the float coefficients add only
// sub-rounding far stationary points beyond 400).
func TestRootLocus_RotatedHighRelDegreeNegativeGain(t *testing.T) {
	roots := []complex128{0.375, complex(-3.625, 1), complex(-3.625, -1), complex(-4.875, 4.75), complex(-4.875, -4.75), complex(3.125, 1.25), complex(3.125, -1.25)}
	denR := rlswPolyFromRoots(roots)
	den := make([]float64, len(denR))
	for i, x := range denR {
		den[i] = rlswF(x)
	}
	for seed := range 4 {
		a, b, c := rlpRotatedCompanion(den, []float64{-4.5}, seed)
		o := newRlswOracle([]*big.Rat{big.NewRat(-9, 2)}, denR)
		if !o.posed || len(o.required) != 0 {
			t.Fatalf("intended oracle %+v", o)
		}
		res, err := RootLocus(rlpSys(t, 7, a, b, c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		if !o.accepts(res.Breakaway) {
			t.Errorf("seed %d: Breakaway = %v, exact oracle %v", seed, res.Breakaway, o.required)
		}
		asym := make([]float64, 7)
		for k := range asym {
			asym[k] = float64(2*k) * math.Pi / 7
		}
		rlpAngles(t, "AsymptoteAngles", res.AsymptoteAngles, asym)
	}
}

// TestRootLocus_OverflowLoopNoBreakawayError (LP755Y-1): a discrete loop
// whose N'D − ND' overflows (eigenvalue ≈ 2.2e109) must not fail with an
// undocumented ErrInvalidArgument from the breakaway polynomial.
func TestRootLocus_OverflowLoopNoBreakawayError(t *testing.T) {
	s := fzDecode(t, 0x4b, [fzNVals]float64{0.3, -0.2, 31, -70, -55, -50, 48, -88, 0, -2, 55, -95, -10, -10, 20, 0})
	n := s.N
	b := make([]float64, n)
	for i := range n {
		b[i] = s.B[i*s.M]
	}
	sys := rlpSys(t, n, s.A, b, s.C[:n], s.D[0], s.Dt)
	if s.In != nil {
		if err := sys.SetInputDelay(s.In[:1]); err != nil {
			t.Fatal(err)
		}
	}
	if s.Out != nil {
		if err := sys.SetOutputDelay(s.Out[:1]); err != nil {
			t.Fatal(err)
		}
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if errors.Is(err, ErrInvalidArgument) || (err == nil && res == nil) {
		t.Fatalf("RootLocus = %v, %v\n%s", res, err, s)
	}
	if err != nil {
		t.Logf("documented error: %v", err)
		return
	}
	for _, z := range res.Breakaway {
		if cmplx.IsNaN(z) || cmplx.IsInf(z) {
			t.Errorf("Breakaway %v not finite", z)
		}
	}
}

// TestRootLocus_NearMultiplePolesNoBreakaway pins the wave-18 blocker: poles
// 5e-5 apart at unit scale (eig resolves them to ~1e-7) are distinct, and
// the stationary points −1.0000333 ± 2.357e-5i are complex, so there is no
// real breakaway (base correct; the eps^(1/m) cluster window merged them).
// Case 169 is the same at the 1e-5 scale under rotation.
func TestRootLocus_NearMultiplePolesNoBreakaway(t *testing.T) {
	a := []float64{0, 1, 0, 0, 0, 1, -1.0001000050000002, -3.000200005000001, -3.0000999999999998}
	res, err := RootLocus(rlpSys(t, 3, a, []float64{0, 0, 1}, []float64{1, 0, 0}, 0, 0), []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	rlpBreak(t, "companion 1/((s+1)((s+1.00005)^2+2.5e-9))", res.Breakaway, nil, 0)

	roots := []complex128{-1.296875e-5, complex(-2.421875e-5, 1.421875e-5), complex(-2.421875e-5, -1.421875e-5)}
	denR := rlswPolyFromRoots(roots)
	den := make([]float64, len(denR))
	for i, x := range denR {
		den[i] = rlswF(x)
	}
	for seed := range 4 {
		a, b, c := rlpRotatedCompanion(den, []float64{2.671875}, seed)
		o := rlpExactOracle(t, 3, a, b, c, 0)
		if len(o.required) != 0 {
			t.Fatalf("seed %d: exact oracle has breakaways %v", seed, o.required)
		}
		res, err := RootLocus(rlpSys(t, 3, a, b, c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpBreak(t, "rotated case 169", res.Breakaway, nil, 0)
	}
}

// TestRootLocus_DoubleIntegratorRotated (w17 #66): the double pole of
// ±k/s² at the origin is a breakaway (K = ∓s²/k = 0) for either gain sign
// when the float data hold it (the nilpotent [1 1; −1 −1]); a rotation
// whose rounding splits it gives the float realization's break set
// (c = 0, J7OYOY). The asymptotes follow the gain sign either way.
func TestRootLocus_DoubleIntegratorRotated(t *testing.T) {
	rotate := func(th float64, k float64) *System {
		Q := mat.NewDense(2, 2, []float64{math.Cos(th), -math.Sin(th), math.Sin(th), math.Cos(th)})
		var qa, qaq, qb, cq mat.Dense
		qa.Mul(Q, mat.NewDense(2, 2, []float64{0, 1, 0, 0}))
		qaq.Mul(&qa, Q.T())
		qb.Mul(Q, mat.NewDense(2, 1, []float64{0, 1}))
		cq.Mul(mat.NewDense(1, 2, []float64{k, 0}), Q.T())
		sys, err := New(&qaq, &qb, &cq, mat.NewDense(1, 1, []float64{0}), 0)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	type tc struct {
		sys *System
		neg bool
	}
	cases := []tc{{rlpSys(t, 2, []float64{1, 1, -1, -1}, []float64{0, 1}, []float64{1, 0}, 0, 0), false}}
	for _, th := range []float64{0.8, 1.27, 1.87, 3.27} {
		for _, k := range []float64{-1, -2.494, 2.494} {
			cases = append(cases, tc{rotate(th, k), k < 0})
		}
	}
	for i, c := range cases {
		res, err := RootLocus(c.sys, []float64{0, 0.5, 2, 10})
		if err != nil {
			t.Fatal(err)
		}
		if i == 0 && (len(res.Breakaway) != 1 || res.Breakaway[0] != 0) {
			t.Errorf("A=%v: Breakaway = %v, want [0]", mat.Formatted(c.sys.A), res.Breakaway)
		}
		a := c.sys.A.RawMatrix().Data
		rlpFloatBreak(t, fmtG(a[0]), res.Breakaway, rlpFloatOracle(2, a, []float64{c.sys.B.At(0, 0), c.sys.B.At(1, 0)}, c.sys.C.RawMatrix().Data, 0, 0, 0))
		want := []float64{math.Pi / 2, 3 * math.Pi / 2}
		if c.neg {
			want = []float64{0, math.Pi}
		}
		rlpAngles(t, "AsymptoteAngles", res.AsymptoteAngles, want)
	}
}

// TestRootLocus_NoiseMarkovKeepsSign (wave 22 B2, reviewer case 11295
// class): k(s+12)(s+48)/((s+3.5)(s+12)(s+48)(s+72)((s+8)²+56²)) under a
// scaling/Givens/triangular similarity with entries ~1e13. Every exact hₖ
// is within 10·eps·sensₖ (h₄…h₆ at ρ 4…8), the rounding-level h₁…h₃ at
// ρ < 0.1; Zeros() returns two noise zeros. With one resolution threshold
// the relative degree is 4, the noise zeros are dropped and the gain sign
// is that of h₄; a noise h₂ had set relative degree 2 and the 0° rule,
// flipping every departure angle by π. Departure angles are checked at the
// poles away from the cancelling pairs, against the intended loop, to the
// 1e-2 the computed poles allow (they are 3e-3 off).
func TestRootLocus_NoiseMarkovKeepsSign(t *testing.T) {
	poles := []complex128{complex(-8, 56), complex(-8, -56), -72, -48, -12, -3.5}
	zeros := []complex128{-12, -48}
	for i, c := range []struct{ a, b, c []float64 }{
		{
			[]float64{-74963132.37725948, -9511.384684225211, 948165.0132592607, 1565098.0212562585, 460169014.6742613, -487.6001984110312, 37817023158.67332, 4798255.8547275225, -478325512.87315845, -789552716.5538807, -232143751619.16232, 245982.11324746237, -2260786023.649144, -286850.36756792275, 28595365.491656695, 47201226.099917084, 13878071165.65308, -14705.358237394117, -149613200.7230736, -18983.049116422902, 1892370.0755045891, 3123660.216335954, 918416291.7625471, -973.164012832899, -15626221.359052839, -1982.6680103696904, 197646.98175599411, 326247.9550113115, 95923192.05832767, -101.64128199636716, -8836528096590.037, -1121185844.503799, 111768098178.31207, 184491115049.68314, 54243951188997.086, -57477492.74378835},
			[]float64{-0.005564648973724355, 2.807226133546673, -0.16782223535867788, -0.011106058891801872, -0.0011599626905148934, -655.9514650395555},
			[]float64{-17520.069511167392, -0.27699507960125047, 54.46596491030627, 63.20878084119821, 90587.22980221029, -0.02775351409986515},
		},
		{
			[]float64{-72504523.04631251, -20283080.464146562, 61390825.59079451, 16313685.979787026, -0.6927383234371569, -16509649.086809227, 138700311.13803843, 38801301.26168309, -117439936.60339868, -31207891.73835765, 1.3252002143935115, 31582768.170062594, 67552301.43556345, 18897708.328706913, -57197693.131861374, -15199423.15534257, 0.6454226597029359, 15382003.164266203, -758622993.1246791, -212224183.48302314, 642339123.9318135, 170691932.0563096, -7.24800768021009, -172742324.45608312, 14973083.279266283, 4190336.6020686924, -12675319.674412185, -3366899.5722265346, 0.1430463646188881, 3409305.4792098426, -350414488.310972, -98028169.33862987, 296701974.82297105, 78844075.30230603, -3.3478143365102535, -79791168.78286518},
			[]float64{0.0010333238219211232, -0.001976736242270303, -0.0009627453641796911, 0.010811782136865036, -0.00021337525469430907, 0.004994054784714517},
			[]float64{106951.56459352338, 108451.20594420865, -65178.31847534501, 7066.97020602911, 0, -7066.97020602911},
		},
	} {
		name := "case " + strconv.Itoa(i)
		sys := rlpSys(t, 6, c.a, c.b, c.c, 0, 0)
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpAngles(t, name+" AsymptoteAngles", res.AsymptoteAngles, []float64{math.Pi / 4, 3 * math.Pi / 4, 5 * math.Pi / 4, 7 * math.Pi / 4})
		got, err := sys.Poles()
		if err != nil {
			t.Fatal(err)
		}
		for k, p := range got {
			j, _ := rvNearest(p, poles)
			if q := poles[j]; real(q) == -12 || real(q) == -48 {
				continue
			}
			want := math.Pi
			for l, q := range poles {
				if l != j {
					want -= cmplx.Phase(poles[j] - q)
				}
			}
			for _, z := range zeros {
				want += cmplx.Phase(poles[j] - z)
			}
			if d := math.Abs(math.Remainder(res.DepartureAngles[k]-want, 2*math.Pi)); d > 1e-2 {
				t.Errorf("%s: departure at %v = %g, want %g", name, poles[j], res.DepartureAngles[k], want)
			}
		}
	}
}

// TestRootLocus_SplitDoubleZeroNoFeedthrough (wave 22, reviewer case 10677
// class): 3(s+3)(s+2.5)²/(s(s+36)(s²+0.5625)), D = 0, under a scaling/
// Givens/triangular similarity whose rounding splits the double zero into
// two real zeros ~9e-5 apart. The float data hold no double zero, so −2.5
// (K = ∞ of the intended loop) is no break point: the break set is that of
// the float realization (c = 0, J7OYOY).
func TestRootLocus_SplitDoubleZeroNoFeedthrough(t *testing.T) {
	for i, m := range []struct{ a, b, c []float64 }{
		{
			[]float64{163.81420983322218, 74.64311883533684, -18.657679132180508, 104.08724455861936, -472.64224358957824, -215.64402982158288, 64.8635449321895, -254.73496123334695, -141.16136208281264, -64.32148825943844, 16.077280882322196, -89.69348206133829, 60.12041991450314, 27.632977608583705, -16.15089553712487, -0.24746089396148463},
			[]float64{-0.16756204546463738, 0.009101100712632315, 0.14439063482537975, 0.341252489940445},
			[]float64{5160.933543272735, 208.4169900883478, 5390.671604349607, 256.45896951229673},
		},
		{
			[]float64{0.15390810036605915, -0.31384178289046555, -0.512144401732552, -0.3095796931570512, 16.57185812997175, 6.543536321209192, -22.181593872621626, 4.336700561913234, -8.454250686839965, 69.9742851310155, 70.01580658577112, 65.34358255525224, -2.9969293057060007, -122.63553821294721, -93.25032335157478, -112.71325100734637},
			[]float64{0, 0, -126.50252275153427, 209.27586745612555},
			[]float64{-5.825280442286043, 3.3362212538934637, 5.326103880181568, 3.2338443295263697},
		},
	} {
		res, err := RootLocus(rlpSys(t, 4, m.a, m.b, m.c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpFloatBreak(t, "case "+strconv.Itoa(i), res.Breakaway, rlpFloatOracle(4, m.a, m.b, m.c, 0, 0, 3))
	}
}

func rlpRats(x ...float64) []*big.Rat {
	out := make([]*big.Rat, len(x))
	for i, v := range x {
		out[i] = fzRat(v)
	}
	return out
}

// TestRootLocus_ResolvedNonNormalPairKSign (wave 22 B1 class, reviewer
// cases 30987, 31290 and a minimal dyadic repro): a complex pole pair of
// relative split ~1e-4 realized by a non-normal A (σ_b/|λ| ~ 2e4) is
// u·κ-linked by eig and split by at most √u·σ_b, so only the converted D
// can tell it from a double; D needs 6e5·u…4.5e6·u to merge it, which the
// old √u tolerance admitted, listing the K < 0 point at the pair's real
// part. The exact oracle of the floats rejects it (robust to 1-ulp
// changes for 31290; rlswUlpSensitive flags the other two, pinned as
// given). Grid: G = 32/((s+3584)² + σ²) under exact I + v·e₀e₁ᵀ
// similarities of the companion [0 1; −q −7168] (the minimal repro is
// σ = 0.4375, v = −2), asserted only where the oracle is posed and robust
// to 1-ulp changes; every point is 1-ulp sensitive at entries ~1e7…1e10
// (eig splits most of them into two real roots), so they are logged as
// precision limits.
func TestRootLocus_ResolvedNonNormalPairKSign(t *testing.T) {
	type sysData struct {
		name    string
		n       int
		a, b, c []float64
	}
	cs := []sysData{
		{"minimal", 2, []float64{25690112.3828125, 51394561.765625, -12845056.19140625, -25697280.3828125}, []float64{-2, 1}, []float64{32, 64}},
		{"30987", 2, []float64{-2.56829453828125e7, -1.284147244140625e7, 5.1351556765625e7, 2.56757773828125e7}, []float64{32, -64}, []float64{2, 1}},
		{"31290", 3, []float64{45942.01278480432, -7700.285188740199, -46254.37380409317, 32263.2672003399, -5410.057865434646, -32487.91250383978, 40346.07038640868, -6761.950570224694, -40619.45491936967}, []float64{0.04153337941018198, 0.029169793742546878, 0.036473984259841205}, []float64{2898791.329147607, -505488.26534131414, -2896245.4694443424}},
	}
	const p = 3584.0
	for _, sg := range []float64{0.4375, 0.25, 0.125, 1.0 / 16, 1.0 / 64} {
		for _, v := range []float64{-2, -8, -64, -1024} {
			q := p*p + sg*sg
			cs = append(cs, sysData{
				"grid σ=" + strconv.FormatFloat(sg, 'g', -1, 64) + " v=" + strconv.FormatFloat(v, 'g', -1, 64), 2,
				[]float64{-v * q, v*v*q + 1 - 2*p*v, -q, v*q - 2*p},
				[]float64{v, 1},
				[]float64{32, -32 * v},
			})
		}
	}
	for i, c := range cs {
		res, err := RootLocus(rlpSys(t, c.n, c.a, c.b, c.c, 0, 0), nil)
		if err != nil {
			t.Fatal(err)
		}
		rc := rlswCase{n: c.n, a: c.a, b: c.b, c: c.c}
		_, o := rc.oracles()
		if !o.posed || i >= 3 && rlswUlpSensitive(&rc, i) {
			t.Logf("%s: oracle unposed or 1-ulp sensitive (precision limit): Breakaway %v, oracle %v", c.name, res.Breakaway, o.required)
			continue
		}
		if !o.accepts(res.Breakaway) {
			t.Errorf("%s: Breakaway = %v, want %v", c.name, res.Breakaway, o.required)
		}
	}
}
