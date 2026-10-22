package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"
)

// Pinned RootLocus breakaway cases of task JHKPWD (waves 16–18). Expected
// values come from exact big.Rat evaluation of G and G' on the given float
// coefficients (rlpExactG), the exact TF oracle (newRlswOracle) or the
// closed-form transfer function.

// rlpExactG returns, exactly over big.Rat, the signs of C(sI−A)⁻¹B and of
// G'(s) = −C(sI−A)⁻²B for the explicit SISO loop (A row-major n×n).
func rlpExactG(t *testing.T, n int, a, b, c []float64, s float64) (g, dg int) {
	t.Helper()
	solve := func(tr bool, rhs []float64) []*big.Rat {
		m := make([][]*big.Rat, n)
		for i := range n {
			m[i] = make([]*big.Rat, n)
			for j := range n {
				v := a[i*n+j]
				if tr {
					v = a[j*n+i]
				}
				m[i][j] = new(big.Rat).Neg(fzRat(v))
				if i == j {
					m[i][j].Add(m[i][j], fzRat(s))
				}
			}
		}
		x := make([]*big.Rat, n)
		for i := range n {
			x[i] = fzRat(rhs[i])
		}
		for k := range n {
			p := k
			for p < n && m[p][k].Sign() == 0 {
				p++
			}
			if p == n {
				t.Fatalf("sI−A singular at s=%v", s)
			}
			m[k], m[p] = m[p], m[k]
			x[k], x[p] = x[p], x[k]
			for i := k + 1; i < n; i++ {
				f := new(big.Rat).Quo(m[i][k], m[k][k])
				for j := k; j < n; j++ {
					m[i][j].Sub(m[i][j], new(big.Rat).Mul(f, m[k][j]))
				}
				x[i].Sub(x[i], new(big.Rat).Mul(f, x[k]))
			}
		}
		for k := n - 1; k >= 0; k-- {
			for j := k + 1; j < n; j++ {
				x[k].Sub(x[k], new(big.Rat).Mul(m[k][j], x[j]))
			}
			x[k].Quo(x[k], m[k][k])
		}
		return x
	}
	x, y := solve(false, b), solve(true, c)
	var gv, dv big.Rat
	for i := range n {
		gv.Add(&gv, new(big.Rat).Mul(fzRat(c[i]), x[i]))
		dv.Sub(&dv, new(big.Rat).Mul(y[i], x[i]))
	}
	return gv.Sign(), dv.Sign()
}

// TestRootLocus_DelayedLoopBreakaway (O5FYRC-4): an output delay of 2 on a
// near-integrator pair. The exact G' of the delay-absorbed realization
// changes sign in (0.66666652, 0.66666653), and the double delay pole is a
// K = 0 break point at z = 0; no far spurious point from noise in the
// delay-absorbed transfer function.
func TestRootLocus_DelayedLoopBreakaway(t *testing.T) {
	sys := rlpSys(t, 2, []float64{0.9999999032225667, 0, 0, 0.9999614835646372},
		[]float64{-0.9667981925794609, -0.01238866346289056}, []float64{-856.888753368947, -210.7957994307797}, 0, 1)
	if err := sys.SetOutputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	abs, err := sys.AbsorbDelay()
	if err != nil {
		t.Fatal(err)
	}
	n, _, _ := abs.Dims()
	a, b, c := abs.A.RawMatrix().Data, abs.B.RawMatrix().Data, abs.C.RawMatrix().Data
	lo, hi := 0.66666652, 0.66666653
	if _, d := rlpExactG(t, n, a, b, c, lo); d <= 0 {
		t.Fatalf("exact G'(%v) sign %d", lo, d)
	}
	if _, d := rlpExactG(t, n, a, b, c, hi); d >= 0 {
		t.Fatalf("exact G'(%v) sign %d", hi, d)
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	rlpBreak(t, "output delay 2", res.Breakaway, []float64{0, (lo + hi) / 2}, 1e-8)
}

// TestRootLocus_LongDelayTriplePole (O5FYRC-3): a discrete triple pole at
// z = 1 (A = I up to 1e-16) with a 40-sample output delay. The 40-fold
// delay pole at 0 is a K = 0 break point. A is the identity to rounding,
// so the plant acts as CB/(z−1) with CB < 0, and the stationary point
// 40/41 of CB·z⁻⁴⁰/(z−1) has G > 0 (exact), K = −1/G < 0: not on the
// locus. Points within 1e-3 of the near-cancelling cluster at z = 1 are a
// precision limit and not judged. The breakaway step must not fail.
func TestRootLocus_LongDelayTriplePole(t *testing.T) {
	s := fzDecode(t, 'k', [fzNVals]float64{-75.8, -100.5, -79.5, -52, 81, 0, -47.714285714285715, 0, 81, -2, 0.7, 1, -135, 75, -4, 0})
	n := s.N
	b := make([]float64, n)
	for i := range n {
		b[i] = s.B[i*s.M]
	}
	if g, _ := rlpExactG(t, n, s.A, b, s.C[:n], 40.0/41); g <= 0 {
		t.Fatalf("exact G(40/41) sign %d, want > 0", g)
	}
	sys := rlpSys(t, n, s.A, b, s.C[:n], s.D[0], s.Dt)
	if err := sys.SetOutputDelay(s.Out[:1]); err != nil {
		t.Fatal(err)
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatalf("RootLocus: %v\n%s", err, s)
	}
	near0 := 0
	for _, z := range res.Breakaway {
		x := real(z)
		switch {
		case imag(z) != 0 || !isFinite(x):
			t.Errorf("Breakaway %v not real and finite", z)
		case math.Abs(x) <= 1e-8:
			near0++
		case math.Abs(x-1) > 1e-3:
			t.Errorf("Breakaway %v off the locus (40/41 needs K < 0)", z)
		}
	}
	if near0 == 0 {
		t.Errorf("Breakaway = %v, want the 40-fold delay pole at 0", res.Breakaway)
	}
}

// TestRootLocus_DelayPolesExact: the z^L factor of an absorbed delay is
// exact, so its multiple pole is reported once and exactly at z = 0, not as
// L−1 rounding-split copies from an L-fold root cluster of N'D − ND'.
func TestRootLocus_DelayPolesExact(t *testing.T) {
	sys := rlpSys(t, 2, []float64{0.9999999032225667, 0, 0, 0.9999614835646372},
		[]float64{-0.9667981925794609, -0.01238866346289056}, []float64{-856.888753368947, -210.7957994307797}, 0, 1)
	if err := sys.SetOutputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	s := fzDecode(t, 'k', [fzNVals]float64{-75.8, -100.5, -79.5, -52, 81, 0, -47.714285714285715, 0, 81, -2, 0.7, 1, -135, 75, -4, 0})
	b := make([]float64, s.N)
	for i := range s.N {
		b[i] = s.B[i*s.M]
	}
	long := rlpSys(t, s.N, s.A, b, s.C[:s.N], s.D[0], s.Dt)
	if err := long.SetOutputDelay(s.Out[:1]); err != nil {
		t.Fatal(err)
	}
	in := rlpSys(t, 2, []float64{0, 1, 0, 0.5}, []float64{0, 1}, []float64{0, 1}, 0, 1)
	if err := in.SetInputDelay([]float64{5}); err != nil {
		t.Fatal(err)
	}
	for name, sys := range map[string]*System{"output delay 2": sys, "output delay 40": long, "input delay 5": in} {
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		var near []complex128
		for _, z := range res.Breakaway {
			if cmplx.Abs(z) <= 1e-8 {
				near = append(near, z)
			}
		}
		if len(near) != 1 || near[0] != 0 {
			t.Errorf("%s: Breakaway = %v, want z = 0 once and exact", name, res.Breakaway)
		}
	}
}

// TestRootLocus_FiveStateFeedthroughBreakaway (w17 #56): exact halves and
// integers, D = 0.5. The exact oracle on the float coefficients and the
// exact sign change of G' in (237.7532, 237.7533) both put the breakaway
// at 237.75326; the reported 237.80914 is not a stationary point.
func TestRootLocus_FiveStateFeedthroughBreakaway(t *testing.T) {
	a := []float64{0, 0, 128, -256, 512, 128, 128, 384, -128, -512, 0, 0, 0, -384, -512, 512, 0, -512, 128, -128, -512, -128, -256, -512, -512}
	b := []float64{128, -512, 384, 128, -256}
	c := []float64{0.5, 1, 1.5, -0.5, 0}
	_, d1 := rlpExactG(t, 5, a, b, c, 237.7532)
	_, d2 := rlpExactG(t, 5, a, b, c, 237.7533)
	_, d3 := rlpExactG(t, 5, a, b, c, 237.80914)
	if d1 <= 0 || d2 >= 0 || d3 != d2 {
		t.Fatalf("exact G' signs %d %d %d", d1, d2, d3)
	}
	o := rlpExactOracle(t, 5, a, b, c, 0.5)
	res, err := RootLocus(rlpSys(t, 5, a, b, c, 0.5, 0), []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !o.accepts(res.Breakaway) {
		t.Errorf("Breakaway = %v, exact oracle %v", res.Breakaway, o.required)
	}
	rlpBreak(t, "5-state", res.Breakaway, []float64{237.75326}, 1e-6)
}

// TestRootLocus_TextbookNonCanonicalBreakaway (w17 #61): 4/(s(s+1)) and
// 3/(s(s−1)) in non-canonical realizations; the numerator's noise leading
// coefficient must not move or drop the breakaway −1/2 (1/2).
func TestRootLocus_TextbookNonCanonicalBreakaway(t *testing.T) {
	for _, tc := range []struct {
		a, b, c []float64
		want    float64
	}{
		{[]float64{-1, 2, 0, 0}, []float64{2, 2}, []float64{2, -2}, -0.5},
		{[]float64{1, 0, -1, 0}, []float64{-1, -2}, []float64{-2, 1}, 0.5},
	} {
		res, err := RootLocus(rlpSys(t, 2, tc.a, tc.b, tc.c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpBreak(t, "textbook", res.Breakaway, []float64{tc.want}, 1e-12)
	}
}

// TestRootLocus_DelayDecouplingZero (w18 #28, owner question w17 #58):
// x1 is unobservable, so the plant is 1/(z−0.5) and with a 2-sample input
// delay G = 1/(z²(z−0.5)): N'D − ND' = −(3z² − z) gives the double pole at
// 0 (K = 0) and 1/3 (K = 1/27). The minimal and non-minimal forms agree.
func TestRootLocus_DelayDecouplingZero(t *testing.T) {
	sys := rlpSys(t, 2, []float64{0, 1, 0, 0.5}, []float64{0, 1}, []float64{0, 1}, 0, 1)
	if err := sys.SetInputDelay([]float64{2}); err != nil {
		t.Fatal(err)
	}
	o := newRlswOracle([]*big.Rat{big.NewRat(1, 1)}, []*big.Rat{big.NewRat(1, 1), big.NewRat(-1, 2), new(big.Rat), new(big.Rat)})
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !o.accepts(res.Breakaway) {
		t.Errorf("Breakaway = %v, exact oracle %v", res.Breakaway, o.required)
	}
	rlpBreak(t, "delayed 1/(z-0.5)", res.Breakaway, []float64{0, 1.0 / 3}, 1e-12)
}

// TestRootLocus_SweepCase482 (w18 #32): rotated 4-state loop at the 1e-3
// scale whose numerator carries rounding-noise leading coefficients; the
// breakaways must match an exact oracle (intended or float TF).
func TestRootLocus_SweepCase482(t *testing.T) {
	cs := &rlswGenerate()[482]
	oInt, oFlt := cs.oracles()
	sys, err := cs.system()
	if err != nil {
		t.Fatal(err)
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	if !oInt.accepts(res.Breakaway) && !oFlt.accepts(res.Breakaway) {
		t.Errorf("Breakaway = %v, exact oracles %v / %v\n%s", res.Breakaway, oInt.required, oFlt.required, cs)
	}
}

// TestRootLocus_DelayedHugePoleNoError: discrete loops with a pole at
// 4.5e86 and 1 or 21 delay samples, whose exact-delay N'D − ND' spans 300
// decades so its companion Schur iteration fails; RootLocus must not lose
// the branches to the optional breakaway step (base ca89d74 succeeds).
func TestRootLocus_DelayedHugePoleNoError(t *testing.T) {
	for _, v := range [][fzNVals]float64{
		{0, 2, 0.3, -13, 0, 0, 6.5, 0.2, 47.95, -10, 39, -64, -80, 0, -10, -64},
		{0, 2, 0.3, -13, 0, 0, 6.5, 0.2, 47.95, -10, 39, 0, 2, 0, -10, -64},
	} {
		s := fzDecode(t, 'a', v)
		sys := rlpSys(t, s.N, s.A, s.B[:s.N], s.C[:s.N], s.D[0], s.Dt)
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
		if err != nil {
			t.Fatalf("RootLocus: %v\n%s", err, s)
		}
		for _, z := range res.Breakaway {
			if imag(z) != 0 || !isFinite(real(z)) {
				t.Errorf("Breakaway %v not real and finite", z)
			}
		}
	}
}
