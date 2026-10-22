package controlsys

import (
	"math"
	"slices"
	"strconv"
	"testing"
)

// Pinned RootLocus cases of task PAOH7X (waves 16–18): breakaway
// classification must not depend on absolute imaginary-part or repeat
// cut-offs. Expected values are closed form or exact big.Rat oracles on
// the given float coefficients; blocker pins of earlier attempts are
// TestRootLocus_NearMultiplePolesNoBreakaway and
// TestRootLocus_DoubleIntegratorRotated.

func rlpCompanion(den, num []float64) (a, b, c []float64) {
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
	return a, b, c
}

// TestRootLocus_SlowOscillatorNoBreakaway (O5FYRC-2): ±w²/(s²+w²) from the
// normal realization [0 w; −w 0]. For +w² the K > 0 branches stay on the
// imaginary axis (s = ±jw·√(1+K)): no breakaway at any w. For −w² they
// meet at s = 0 at K = 1. Base reported 0 for +w² at w = 1e-7.
func TestRootLocus_SlowOscillatorNoBreakaway(t *testing.T) {
	for _, w := range []float64{1e-9, 1e-7, 1e-5, 1e-3, 1, 1e3} {
		for _, sg := range []float64{1, -1} {
			sys := rlpSys(t, 2, []float64{0, w, -w, 0}, []float64{0, 1}, []float64{sg * w, 0}, 0, 0)
			res, err := RootLocus(sys, []float64{0, 1})
			if err != nil {
				t.Fatal(err)
			}
			var want []float64
			if sg < 0 {
				want = []float64{0}
			}
			rlpBreak(t, "osc w="+fmtG(w)+" sign "+fmtSign(sg), res.Breakaway, want, 1e-9*w)
		}
	}
}

// TestRootLocus_CloseRealPolesGainSign (O5FYRC-2): ±1/((s+p)(s+p(1+δ)))
// with δ = 5e-5 and 1e-6 (eig resolves the poles to ~1e-10 relative). The
// stationary point s = −p(1+δ/2) needs K = ∓(s+p)(s+p(1+δ)) = ±p²δ²/4: a
// breakaway for +1, off the K > 0 locus for −1. Base merged the poles
// (within 1e-4) and reported it for both signs.
func TestRootLocus_CloseRealPolesGainSign(t *testing.T) {
	for _, p := range []float64{1e-6, 1, 1e4} {
		for _, d := range []float64{5e-5, 1e-6} {
			q := p * (1 + d)
			for _, sg := range []float64{1, -1} {
				a, b, c := rlpCompanion([]float64{1, p + q, p * q}, []float64{sg})
				res, err := RootLocus(rlpSys(t, 2, a, b, c, 0, 0), []float64{0, 1})
				if err != nil {
					t.Fatal(err)
				}
				var want []float64
				if sg > 0 {
					want = []float64{-(p + q) / 2}
				}
				rlpBreak(t, "close p="+fmtG(p)+" d="+fmtG(d)+" sign "+fmtSign(sg), res.Breakaway, want, 1e-9)
			}
		}
	}
}

// TestRootLocus_TriplePoleBreakaway (w17 #63): ±1/(s+p)³ in companion form;
// the triple pole is a K = 0 breakaway under either sign. eig splits it by
// ~eps^(1/3)·p.
func TestRootLocus_TriplePoleBreakaway(t *testing.T) {
	for _, p := range []float64{1e-3, 0.5, 1, 1e3} {
		for _, sg := range []float64{1, -1} {
			a, b, c := rlpCompanion([]float64{1, 3 * p, 3 * p * p, p * p * p}, []float64{sg})
			res, err := RootLocus(rlpSys(t, 3, a, b, c, 0, 0), []float64{0, 1})
			if err != nil {
				t.Fatal(err)
			}
			rlpBreak(t, "triple p="+fmtG(p)+" sign "+fmtSign(sg), res.Breakaway, []float64{-p}, 1e-6)
		}
	}
}

// TestRootLocus_ScaledDoublePoleBreakaway (w18 #37/#38): k(s+1024r)/
// (s(s+1536r)(s+2048r)²(s+3072r)), r = 1/2, 1, 2, in companion form. eig
// splits the double pole by ~1e-7 relative (1.9e-4 at r = 1), beyond the
// absolute 1e-4 repeat window base used; the exact oracle on the float
// coefficients keeps it.
func TestRootLocus_ScaledDoublePoleBreakaway(t *testing.T) {
	for _, r := range []float64{0.5, 1, 2} {
		roots := []complex128{0, complex(-1536*r, 0), complex(-2048*r, 0), complex(-2048*r, 0), complex(-3072*r, 0)}
		dr := rlswPolyFromRoots(roots)
		den := make([]float64, len(dr))
		for i, x := range dr {
			den[i] = rlswF(x)
		}
		k := 2.2518e15 * math.Pow(r, 4)
		a, b, c := rlpCompanion(den, []float64{k, k * 1024 * r})
		o := rlpExactOracle(t, 5, a, b, c, 0)
		res, err := RootLocus(rlpSys(t, 5, a, b, c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpBreak(t, "w18 #37 r="+fmtG(r), res.Breakaway, o.required, 1e-6)
	}
}

// TestRootLocus_RotatedDoublePoleBreakaway: ±1/((s+p)²(s+2p)) under an
// orthogonal similarity at p = 2^−12 … 2^3 (larger p makes the rotated
// companion itself ill-posed: entries ~p³ and an order-reduced TF). eig splits the double pole by
// up to ~√eps·‖A‖, beyond base's absolute 1e-4 window at large p and into
// a complex pair; it is a K = 0 breakaway under either sign (closed form;
// the exact oracle on the rotated floats is too close to judge).
func TestRootLocus_RotatedDoublePoleBreakaway(t *testing.T) {
	for _, e := range []int{-12, -6, 0, 3} {
		p := math.Ldexp(1, e)
		for _, sg := range []float64{1, -1} {
			for seed := range 3 {
				a, b, c := rlpRotatedCompanion([]float64{1, 4 * p, 5 * p * p, 2 * p * p * p}, []float64{sg * p * p * p}, seed)
				res, err := RootLocus(rlpSys(t, 3, a, b, c, 0, 0), []float64{0, 1})
				if err != nil {
					t.Fatal(err)
				}
				name := "rotated (s+p)^2(s+2p) p=" + fmtG(p) + " sign " + fmtSign(sg) + " seed " + strconv.Itoa(seed)
				if !slices.ContainsFunc(res.Breakaway, func(z complex128) bool { return math.Abs(real(z)+p) <= 1e-6*p }) {
					t.Errorf("%s: Breakaway = %v, want −p = %g", name, res.Breakaway, -p)
				}
			}
		}
	}
}

// TestRootLocus_DelayedDoubleZeroBreakaway (sweep case 671): z^−3·k(z−1)²/
// ((z+28)((z+22)²+22²)(z−10)) in companion form. The 7-state absorbed
// loop's conversion splits the double zero at 1 by ~1e-6; the undelayed
// N, D0 keep it exact, so 1 (K = ∞) and the delay poles' 0 are breakaways.
func TestRootLocus_DelayedDoubleZeroBreakaway(t *testing.T) {
	c := rlswGenerate()[671]
	oi, _ := c.oracles()
	sys, err := c.system()
	if err != nil {
		t.Fatal(err)
	}
	res, err := RootLocus(sys, []float64{0, 1})
	if err != nil {
		t.Fatal(err)
	}
	rlpBreak(t, "case 671", res.Breakaway, oi.required, 1e-6)
}

func fmtG(x float64) string {
	return strconv.FormatFloat(x, 'g', -1, 64)
}
