package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// ratPolyFromRoots returns the monic real polynomial of roots (each complex
// root standing for its conjugate pair), formed exactly and rounded once.
func ratPolyFromRoots(roots []complex128) []float64 {
	one := big.NewRat(1, 1)
	p := []*big.Rat{one}
	mul := func(q ...*big.Rat) {
		out := make([]*big.Rat, len(p)+len(q)-1)
		for k := range out {
			out[k] = new(big.Rat)
		}
		for i, v := range p {
			for j, w := range q {
				out[i+j].Add(out[i+j], new(big.Rat).Mul(v, w))
			}
		}
		p = out
	}
	rat := func(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }
	for _, r := range roots {
		if imag(r) == 0 {
			mul(one, rat(-real(r)))
			continue
		}
		re, im := rat(real(r)), rat(imag(r))
		m2 := new(big.Rat).Add(new(big.Rat).Mul(re, re), new(big.Rat).Mul(im, im))
		mul(one, new(big.Rat).Mul(big.NewRat(-2, 1), re), m2)
	}
	out := make([]float64, len(p))
	for i, v := range p {
		out[i], _ = v.Float64()
	}
	return out
}

// observableCompanion returns the upper Hessenberg realization with unit
// subdiagonal, last column −den reversed, C = eₙᵀ; its num is reversed(b).
func observableCompanion(den []float64) []float64 {
	n := len(den) - 1
	a := make([]float64, n*n)
	for i := 1; i < n; i++ {
		a[i*n+i-1] = 1
	}
	for i := range n {
		a[i*n+n-1] = -den[n-i]
	}
	return a
}

// DZG5LV: an observable companion of slow poles with graded B. Dgebal's den
// was exact but the rotations put B's 1e12 entries into the high num
// coefficients; Osborne's num was right but its den had a wrong DC term and
// a right-half-plane root, and the check points above the poles took it.
func TestTransferFunctionObservableCompanionGraded(t *testing.T) {
	n := 6
	lc := []float64{-1.1012337036710466e-23, -3.947356904973473e-19, -6.2876063073946964e-15, -5.4056859020206906e-11, -4.3199904331973903e-07, -0.001190106885168501}
	a := make([]float64, n*n)
	for i := 1; i < n; i++ {
		a[i*n+i-1] = 1
	}
	for i := range n {
		a[i*n+n-1] = lc[i]
	}
	b := []float64{2.651244509846853e12, 1.017315241503764e9, 2.0472325116298143e6, 262.46457040571573, 0, 0}
	c := []float64{0, 0, 0, 0, 0, 1}
	sys, tf, den, num, e := oracleSISO(t, a, b, c, n)
	if e > 1e-14 {
		t.Errorf("oracle err %.3g: den %v num %v", e, tf.Den[0], tf.Num[0][0])
	}
	for k, v := range tf.Den[0] {
		if r := math.Abs(v-den[k]) / den[k]; !(r <= 1e-14) {
			t.Errorf("den[%d] = %.17g, want %.17g", k, v, den[k])
		}
	}
	if got := tf.Num[0][0]; len(got) != 4 {
		t.Errorf("num %v, want %v", got, num[3:])
	} else {
		for k, v := range got {
			if r := math.Abs(v-num[3+k]) / num[3+k]; !(r <= 1e-14) {
				t.Errorf("num[%d] = %.17g, want %.17g", k, v, num[3+k])
			}
		}
	}
	for _, w := range []float64{2e-4, 2e-3, 2.2, 223} {
		if e := evalFrRelErr(t, sys, tf, complex(0, w)); e > 1e-12 {
			t.Errorf("TF(%gj) off EvalFr by %.3g", w, e)
		}
	}
}

// DZG5LV: slow companions, den[6] was 4e-6 relative off from the rotations
// over Dgebal's partial grading (superdiagonal 2^-4..2^-9, last row
// 1e-12..1e-3), and a clustered triple root (s+p)³ lost up to 1e-6.
func TestTransferFunctionSlowCompanionExactDen(t *testing.T) {
	p := 37 / math.Ldexp(1, 14)
	cases := []struct {
		name string
		den  []float64
		c    []float64
	}{
		{"partial_grading", []float64{1, 0.01708984375, 7.277727127075195e-05, -4.3958425521850586e-07, -2.9905926623996493e-09, 1.6186166990062034e-12, 5.667825675345585e-16, 0}, []float64{3.375, 0, 0, 0, 0, 0, 0}},
		{"triple_root", ratPolyFromRoots([]complex128{complex(-p, 0), complex(-p, 0), complex(-p, 0), complex(-5/math.Ldexp(1, 11), 3/math.Ldexp(1, 11)), 0}),
			[]float64{7 / math.Ldexp(1, 33), 3 / math.Ldexp(1, 20), 0, 0, 0, 0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.den) - 1
			res, err := slowCompanion(t, tc.den, tc.c, 0).TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			got := res.TF.Den[0]
			if len(got) != n+1 {
				t.Fatalf("den %v, want %v", got, tc.den)
			}
			for k, want := range tc.den {
				if got[k] != want {
					t.Errorf("den[%d] = %.17g, want %.17g", k, got[k], want)
				}
			}
		})
	}
}

// DZG5LV: an observable companion with two graded inputs converts each
// input in the given basis.
func TestTransferFunctionObservableCompanionMIMO(t *testing.T) {
	den := ratPolyFromRoots([]complex128{-1 / 1024.0, complex(-3/2048.0, 1/512.0), -5 / 4096.0, 0})
	n := len(den) - 1
	a := observableCompanion(den)
	b := [][]float64{{3e9, 7e5, 11, 0, 0}, {0, 1, 0, 2.5e-6, 0}}
	bm := make([]float64, n*2)
	for i := range n {
		bm[2*i], bm[2*i+1] = b[0][i], b[1][i]
	}
	c := make([]float64, n)
	c[n-1] = 1
	sys, err := NewFromSlices(n, 2, 1, a, bm, c, []float64{0, 0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	for j := range 2 {
		exDen, exNum := ratSISOTF(a, b[j], c, n)
		for k, v := range res.TF.Den[0] {
			if v != exDen[k] {
				t.Errorf("den[%d] = %.17g, want %.17g", k, v, exDen[k])
			}
		}
		got := res.TF.Num[0][j]
		off := n + 1 - len(got)
		for k := range exNum[:off] {
			if exNum[k] != 0 {
				t.Fatalf("num %d = %v, want %v", j, got, exNum)
			}
		}
		for k, v := range got {
			if want := exNum[off+k]; math.Abs(v-want) > 1e-15*math.Abs(want) {
				t.Errorf("num %d[%d] = %.17g, want %.17g", j, k, v, want)
			}
		}
	}
}

type hessenbergCase struct {
	a, b, c []float64
	n       int
}

func hessenbergSweepRoots(r *rand.Rand, n int, scale float64) []complex128 {
	dy := func() float64 { return float64(1+r.IntN(63)) / 16 * scale }
	sign := func(v float64) float64 {
		if r.IntN(5) == 0 {
			return v
		}
		return -v
	}
	var roots []complex128
	origins := 0
	for deg := 0; deg < n; {
		switch k := r.IntN(10); {
		case k == 0 && origins < 2:
			roots = append(roots, 0)
			origins++
			deg++
		case k <= 2 && deg+2 <= n:
			roots = append(roots, complex(sign(dy()), dy()))
			deg += 2
		case k == 3 && deg+3 <= n:
			p := -dy()
			roots = append(roots, complex(p, 0), complex(p, 0), complex(p, 0))
			deg += 3
		default:
			roots = append(roots, complex(sign(dy()), 0))
			deg++
		}
	}
	return roots
}

// newHessenbergCase draws family fam: 0 observable companion (graded B),
// 1 controllable companion, 2 tf2ss form (first row −den, B = e₁), 3 dense
// graded upper Hessenberg with B = e₁ or C = eₙᵀ, 4 dense graded lower
// Hessenberg with C = e₁ᵀ; poles at scale 2^-20..2^7.
func newHessenbergCase(r *rand.Rand, fam int) hessenbergCase {
	n := 2 + r.IntN(9)
	scale := math.Ldexp(1, -20+r.IntN(28))
	den := ratPolyFromRoots(hessenbergSweepRoots(r, n, scale))
	num := ratPolyFromRoots(hessenbergSweepRoots(r, r.IntN(n), scale*math.Ldexp(1, r.IntN(9)-4)))
	g := math.Ldexp(float64(1+r.IntN(63)), r.IntN(40)-20)
	vec := make([]float64, n)
	for k, v := range num {
		vec[len(num)-1-k] = v * g
	}
	a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
	switch fam {
	case 0:
		a = observableCompanion(den)
		copy(b, vec)
		c[n-1] = 1
	case 1:
		for i := range n - 1 {
			a[i*n+i+1] = 1
		}
		for j := range n {
			a[(n-1)*n+j] = -den[n-j]
		}
		b[n-1] = 1
		copy(c, vec)
	case 2:
		for j := range n {
			a[j] = -den[j+1]
		}
		for i := 1; i < n; i++ {
			a[i*n+i-1] = 1
		}
		b[0] = 1
		for k := range n {
			c[k] = vec[n-1-k]
		}
	default:
		d := make([]float64, n)
		for i := range d {
			d[i] = math.Ldexp(1, r.IntN(21)-10)
		}
		lower := fam == 4
		for i := range n {
			for j := range n {
				if (!lower && i <= j+1) || (lower && j <= i+1) {
					v := float64(r.IntN(129)-64) / 32 * scale
					if (i == j+1 || j == i+1) && v == 0 {
						v = scale
					}
					a[i*n+j] = v * d[j] / d[i]
				}
			}
			b[i] = float64(r.IntN(33)-16) / 8 / d[i]
			c[i] = float64(r.IntN(33)-16) / 8 * d[i]
		}
		switch {
		case lower:
			clear(c)
			c[0] = 1
		case r.IntN(2) == 0:
			clear(b)
			b[0] = 1
		default:
			clear(c)
			c[n-1] = 1
		}
	}
	return hessenbergCase{a, b, c, n}
}

// hessenbergErr is the oracle error of tf: the larger of the scaled
// coefficient error and the relative response error at jω, ω = ρ·10^(k/2),
// k = −6..6 (+Inf on order loss).
func hessenbergErr(tf *TransferFunc, den, num []float64, n int, rhoA float64) float64 {
	if len(tf.Den[0]) != n+1 {
		return math.Inf(1)
	}
	rho := 1e-6 * rhoA
	for k := 1; k <= n; k++ {
		rho = max(rho, math.Pow(math.Abs(den[k]), 1/float64(k)))
	}
	worst := max(scaledCoeffErr(tf.Den[0], den, rho, n), scaledCoeffErr(tf.Num[0][0], num, rho, n))
	for k := range 13 {
		w := rho * math.Pow(10, float64(k-6)/2) * 1.003
		want := ratRatioJ(num, den, w)
		if want == 0 || cmplx.IsInf(want) {
			continue
		}
		got := tf.eval(complex(0, w))[0][0]
		worst = max(worst, cmplx.Abs(got-want)/cmplx.Abs(want))
	}
	return worst
}

func rhpRoots(p []float64) int {
	for len(p) > 1 && p[len(p)-1] == 0 {
		p = p[:len(p)-1]
	}
	rts, err := Poly(p).Roots()
	if err != nil {
		return -1
	}
	k := 0
	for _, r := range rts {
		if real(r) > 0 {
			k++
		}
	}
	return k
}

// TestTransferFunctionHessenbergSweep converts seeded companion and graded
// Hessenberg realizations with and without the given-basis path (DZG5LV)
// and checks each against the exact big.Rat transfer function: the path
// must never be worse, nor flip the right-half-plane root count, and must
// fix the companion cases the rotations got wrong.
func TestTransferFunctionHessenbergSweep(t *testing.T) {
	skipSlowUnderRace(t)
	r := rand.New(rand.NewPCG(2026, 21))
	const trials = 1000
	better, worse := 0, 0
	for trial := range trials {
		cs := newHessenbergCase(r, trial%5)
		den, num := ratSISOTF(cs.a, cs.b, cs.c, cs.n)
		sys, err := NewFromSlices(cs.n, 1, 1, cs.a, cs.b, cs.c, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		rhoA := frobNorm(mat.NewDense(cs.n, cs.n, append([]float64(nil), cs.a...)))
		var tfs [2]*TransferFunc
		for k, noDirect := range []bool{false, true} {
			cv := newRowRealizationConverter(sys, &TransferFuncOpts{})
			cv.noDirect = noDirect
			res, err := cv.convert()
			if err != nil {
				t.Fatalf("trial %d: %v", trial, err)
			}
			tfs[k] = res.TF
		}
		eNew, eBase := hessenbergErr(tfs[0], den, num, cs.n, rhoA), hessenbergErr(tfs[1], den, num, cs.n, rhoA)
		if eNew > 2*eBase && eNew > 1e-12 {
			worse++
			t.Errorf("trial %d: err %.3g, without the path %.3g; A %v B %v C %v", trial, eNew, eBase, cs.a, cs.b, cs.c)
		}
		if eNew < 0.5*eBase && eBase > 1e-12 {
			better++
		}
		if len(tfs[0].Den[0]) == cs.n+1 && len(tfs[1].Den[0]) == cs.n+1 {
			if ex := rhpRoots(den); rhpRoots(tfs[0].Den[0]) != ex && rhpRoots(tfs[1].Den[0]) == ex {
				t.Errorf("trial %d: den %v flips the RHP root count of %v", trial, tfs[0].Den[0], den)
			}
		}
	}
	if better < trials/20 {
		t.Errorf("only %d of %d improved", better, trials)
	}
}
