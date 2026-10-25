package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
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
