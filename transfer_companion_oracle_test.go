package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// ratCharPoly returns det(sI − A), descending (Faddeev–LeVerrier).
func ratCharPoly(A []*big.Rat, n int) []*big.Rat {
	coef := make([]*big.Rat, n+1)
	coef[0] = big.NewRat(1, 1)
	M := make([]*big.Rat, n*n)
	for i := range M {
		M[i] = new(big.Rat)
	}
	AM := make([]*big.Rat, n*n)
	t := new(big.Rat)
	for k := 1; k <= n; k++ {
		for i := range n {
			M[i*n+i].Add(M[i*n+i], coef[k-1])
		}
		for i := range n {
			for j := range n {
				s := new(big.Rat)
				for l := range n {
					s.Add(s, t.Mul(A[i*n+l], M[l*n+j]))
				}
				AM[i*n+j] = s
			}
		}
		tr := new(big.Rat)
		for i := range n {
			tr.Add(tr, AM[i*n+i])
		}
		coef[k] = tr.Neg(tr.Quo(tr, big.NewRat(int64(k), 1)))
		M, AM = AM, M
	}
	return coef
}

// ratSISOTF returns the exact den and num (descending, len n+1, num[0] = 0)
// of c(sI − A)⁻¹b: num = det(sI − A + bc) − det(sI − A).
func ratSISOTF(a, b, c []float64, n int) (den, num []float64) {
	A, ABC := make([]*big.Rat, n*n), make([]*big.Rat, n*n)
	for i := range n {
		for j := range n {
			A[i*n+j] = new(big.Rat).SetFloat64(a[i*n+j])
			bc := new(big.Rat).Mul(new(big.Rat).SetFloat64(b[i]), new(big.Rat).SetFloat64(c[j]))
			ABC[i*n+j] = bc.Sub(A[i*n+j], bc)
		}
	}
	pa, pbc := ratCharPoly(A, n), ratCharPoly(ABC, n)
	den, num = make([]float64, n+1), make([]float64, n+1)
	for k := range n + 1 {
		den[k], _ = pa[k].Float64()
		num[k], _ = new(big.Rat).Sub(pbc[k], pa[k]).Float64()
	}
	return den, num
}

// scaledCoeffErr is the error of got against exact (both descending,
// aligned at the constant term) for coefficients of s^j weighted by ρ^(j−deg),
// relative to the largest weighted exact coefficient.
func scaledCoeffErr(got, exact []float64, rho float64, deg int) float64 {
	w := func(p []float64, k int) float64 {
		j := len(p) - 1 - k
		return p[k] * math.Pow(rho, float64(j-deg))
	}
	norm := 0.0
	for k := range exact {
		norm = max(norm, math.Abs(w(exact, k)))
	}
	if norm == 0 {
		norm = 1
	}
	worst := 0.0
	L := max(len(got), len(exact))
	for i := range L {
		g, e := 0.0, 0.0
		if k := len(got) - L + i; k >= 0 {
			g = w(got, k)
		}
		if k := len(exact) - L + i; k >= 0 {
			e = w(exact, k)
		}
		worst = max(worst, math.Abs(g-e)/norm)
	}
	return worst
}

// oracleSISO converts (a, b, c) and returns its TF, the exact rational den
// and num, and the scaled error of the TF against them (+Inf on order loss).
func oracleSISO(t *testing.T, a, b, c []float64, n int) (sys *System, tf *TransferFunc, den, num []float64, err float64) {
	t.Helper()
	den, num = ratSISOTF(a, b, c, n)
	sys, e := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0)
	if e != nil {
		t.Fatal(e)
	}
	res, e := sys.TransferFunction(nil)
	if e != nil {
		t.Fatalf("TransferFunction: %v", e)
	}
	if len(res.TF.Den[0]) != n+1 {
		return sys, res.TF, den, num, math.Inf(1)
	}
	rho := 1e-6 * frobNorm(sys.A)
	for k := 1; k <= n; k++ {
		rho = max(rho, math.Pow(math.Abs(den[k]), 1/float64(k)))
	}
	err = max(scaledCoeffErr(res.TF.Den[0], den, rho, n), scaledCoeffErr(res.TF.Num[0][0], num, rho, n))
	return sys, res.TF, den, num, err
}

func evalFrRelErr(t *testing.T, sys *System, tf *TransferFunc, s complex128) float64 {
	t.Helper()
	g, err := sys.EvalFr(s)
	if err != nil {
		t.Fatal(err)
	}
	return cmplx.Abs(tf.eval(s)[0][0]-g[0][0]) / cmplx.Abs(g[0][0])
}

// P6VF2M: rotation noise on the diagonal of an integrator chain made Dgebal
// grade it like a zero diagonal and lose the finite zero, unchecked since
// the diagonal was not exactly zero.
func TestTransferFunctionRotatedChainKeepsZero(t *testing.T) {
	a := []float64{5.329070518200751e-17, -1.5543122344752193e-17, -1, -1, -5.551115123125783e-17, 0, 0, 0, -1.65625}
	sys, tf, _, _, e := oracleSISO(t, a, []float64{0, 0, 1}, []float64{2.5, -3.8281249999999996, 0}, 3)
	if e > 1e-12 {
		t.Errorf("oracle err %.3g; num %v den %v", e, tf.Num[0][0], tf.Den[0])
	}
	if num := tf.Num[0][0]; len(num) != 2 || math.Abs(num[0]+2.5) > 1e-12 {
		t.Errorf("num = %v, want ≈ [-2.5 -3.828125]", num)
	}
	if e := evalFrRelErr(t, sys, tf, 1i); e > 1e-12 {
		t.Errorf("TF(j) off EvalFr by %.3g", e)
	}
}

// The same chain over a seeded family: k(s+z)/(s²(s+p)) behind eps-level
// diagonal and coupling noise; the base kept Dgebal's lost zero in nearly
// all of them.
func TestTransferFunctionNoisyChainFamily(t *testing.T) {
	rng := rand.New(rand.NewPCG(3, 4))
	dy := func(lo, hi int) float64 {
		return math.Ldexp(float64(1+rng.IntN(255)), -(lo + rng.IntN(hi-lo+1)))
	}
	noise := func() float64 { return (rng.Float64() - 0.5) * 2 * eps() }
	for i := range 300 {
		a := []float64{noise(), noise(), -1, -1, noise(), 0, 0, 0, -dy(5, 9)}
		c := []float64{dy(5, 9), -dy(5, 9), 0}
		sys, tf, _, _, e := oracleSISO(t, a, []float64{0, 0, 1}, c, 3)
		if e2 := evalFrRelErr(t, sys, tf, 1i); e > 1e-10 || e2 > 1e-10 {
			t.Fatalf("case %d: oracle err %.3g, EvalFr err %.3g; A %v C %v num %v", i, e, e2, a, c, tf.Num[0][0])
		}
	}
}

// P6VF2M: the trimmed Dgebal result of an exact companion was within the
// 1e3·n·eps gate while its den was 1e4·eps off the last row of A, and the
// gate kept it over Osborne's exact one.
func TestTransferFunctionCompanionDenOracle(t *testing.T) {
	den := []float64{1, 123.0 / 4096, 1333.0 / 4194304, 26117 / math.Ldexp(1, 34), 1457427 / math.Ldexp(1, 48), 19918169 / math.Ldexp(1, 60), 0}
	n := 6
	a := make([]float64, n*n)
	for i := range n - 1 {
		a[i*n+i+1] = 1
	}
	for j := range n {
		a[(n-1)*n+j] = -den[n-j]
	}
	b := make([]float64, n)
	b[n-1] = 1
	c := []float64{2359 / math.Ldexp(1, 59), -7 / math.Ldexp(1, 41), 7 / math.Ldexp(1, 33), 0, 0, 0}
	_, tf, exDen, exNum, _ := oracleSISO(t, a, b, c, n)
	got := tf.Den[0]
	if len(got) != n+1 {
		t.Fatalf("den %v", got)
	}
	for k := 1; k < n; k++ {
		if exDen[k] != den[k] {
			t.Fatalf("oracle den[%d] = %g, want %g", k, exDen[k], den[k])
		}
		if r := math.Abs(got[k]-den[k]) / den[k]; r > 1e-13 {
			t.Errorf("den[%d] = %.17g, want %.17g (rel %.2g)", k, got[k], den[k], r)
		}
	}
	if math.Abs(got[n]) > 1e-13*den[n-1]*den[1] {
		t.Errorf("den[%d] = %g, want 0", n, got[n])
	}
	num := tf.Num[0][0]
	if len(num) != 3 {
		t.Fatalf("num %v", num)
	}
	for k, v := range num {
		want := exNum[n-2+k]
		if r := math.Abs(v-want) / math.Abs(want); r > 1e-12 {
			t.Errorf("num[%d] = %.17g, want %.17g (rel %.2g)", k, v, want, r)
		}
	}
}
