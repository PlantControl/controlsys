package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"
)

// slowCompanion returns the controllable companion realization of the monic
// den (B = e_n), whose exact transfer function is reversed(c)/den + d.
func slowCompanion(t *testing.T, den, c []float64, d float64) *System {
	t.Helper()
	n := len(den) - 1
	a := make([]float64, n*n)
	for i := range n - 1 {
		a[i*n+i+1] = 1
	}
	for j := range n {
		a[(n-1)*n+j] = -den[n-j]
	}
	b := make([]float64, n)
	b[n-1] = 1
	sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{d}, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func polyFromRealRoots(roots ...float64) Poly {
	p := Poly{1}
	for _, r := range roots {
		p = p.Mul(Poly{1, -r})
	}
	return p
}

// ratRatioJ returns num(jw)/den(jw), evaluated exactly in rationals and
// rounded once.
func ratRatioJ(num, den []float64, w float64) complex128 {
	eval := func(p []float64) (re, im *big.Rat) {
		wr := new(big.Rat).SetFloat64(w)
		re, im = new(big.Rat), new(big.Rat)
		for _, c := range p {
			nr := new(big.Rat).Mul(im, wr)
			nr.Neg(nr).Add(nr, new(big.Rat).SetFloat64(c))
			im = new(big.Rat).Mul(re, wr)
			re = nr
		}
		return re, im
	}
	nr, ni := eval(num)
	dr, di := eval(den)
	q := new(big.Rat).Add(new(big.Rat).Mul(dr, dr), new(big.Rat).Mul(di, di))
	re := new(big.Rat).Add(new(big.Rat).Mul(nr, dr), new(big.Rat).Mul(ni, di))
	im := new(big.Rat).Sub(new(big.Rat).Mul(ni, dr), new(big.Rat).Mul(nr, di))
	fr, _ := re.Quo(re, q).Float64()
	fi, _ := im.Quo(im, q).Float64()
	return complex(fr, fi)
}

// slowCompanionCase is a seeded slow companion: force origin poles, then
// dyadic real poles (at most maxOrigin at the origin in all) and pairs and
// zeros at scale 2^-14..2^-8. lo and hi bound the nonzero pole and zero
// moduli; den and num are the exact coefficients.
type slowCompanionCase struct {
	den, c, num []float64
	lo, hi      float64
}

func newSlowCompanionCase(r *rand.Rand, n, maxOrigin, force int) slowCompanionCase {
	scale := math.Ldexp(1, -14+2*r.IntN(4))
	dy := func() float64 { return float64(r.IntN(64)-32) / 16 * scale }
	rat := func(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }
	mul := func(p []*big.Rat, q ...*big.Rat) []*big.Rat {
		out := make([]*big.Rat, len(p)+len(q)-1)
		for k := range out {
			out[k] = new(big.Rat)
		}
		for a, v := range p {
			for b, w := range q {
				out[a+b].Add(out[a+b], new(big.Rat).Mul(v, w))
			}
		}
		return out
	}
	one := big.NewRat(1, 1)
	den := []*big.Rat{one}
	poles := make([]float64, force)
	for range force {
		den = mul(den, one, new(big.Rat))
	}
	origins := force
	lo, hi := math.Inf(1), 0.0
	for deg := force; deg < n; {
		if deg+2 <= n && r.IntN(3) == 0 {
			re, im := dy(), math.Abs(dy())+scale/16
			m2 := new(big.Rat).Add(new(big.Rat).Mul(rat(re), rat(re)), new(big.Rat).Mul(rat(im), rat(im)))
			den = mul(den, one, rat(-2*re), m2)
			w := math.Hypot(re, im)
			lo, hi = min(lo, w), max(hi, w)
			deg += 2
			continue
		}
		p := dy()
		if p == 0 {
			if origins >= maxOrigin {
				continue
			}
			origins++
		} else {
			lo, hi = min(lo, math.Abs(p)), max(hi, math.Abs(p))
		}
		den = mul(den, one, rat(-p))
		poles = append(poles, p)
		deg++
	}
	num := []*big.Rat{rat(float64(1+r.IntN(64)) / 16)}
	for range r.IntN(n) {
		z := dy()
		for slices.Contains(poles, z) {
			z = dy()
		}
		if z != 0 {
			lo, hi = min(lo, math.Abs(z)), max(hi, math.Abs(z))
		}
		num = mul(num, one, rat(-z))
	}
	if hi == 0 {
		lo, hi = scale, scale
	}
	cs := slowCompanionCase{den: make([]float64, n+1), c: make([]float64, n), lo: lo, hi: hi}
	for k, v := range den {
		cs.den[k], _ = v.Float64()
	}
	for k, v := range num {
		cs.c[len(num)-1-k], _ = v.Float64()
	}
	cs.num = slices.Clone(cs.c[:len(num)])
	slices.Reverse(cs.num)
	return cs
}

// bandErr is the worst relative error of tf against the exact response
// over [lo/10, 100·hi] (the factor 1.003 keeps the points off jω poles).
func (cs slowCompanionCase) bandErr(tf *TransferFunc) float64 {
	worst := 0.0
	for k := range 9 {
		w := cs.lo / 10 * math.Pow(100.3*cs.hi/cs.lo, float64(k)/8)
		want := ratRatioJ(cs.num, cs.den, w)
		got := tf.eval(complex(0, w))[0][0]
		worst = max(worst, cmplx.Abs(got-want)/cmplx.Abs(want))
	}
	return worst
}

// TestTransferFunctionSlowCompanionSweep converts seeded companions of slow
// dyadic poles (at most one at the origin) and zeros, scale 2^-14..2^-8,
// and checks the response against the exact coefficients over the band of
// the poles and zeros (U3H63X).
func TestTransferFunctionSlowCompanionSweep(t *testing.T) {
	r := rand.New(rand.NewPCG(20, 63))
	fails := 0
	for trial := range 400 {
		n := 3 + r.IntN(4)
		cs := newSlowCompanionCase(r, n, 1, 0)
		res, err := slowCompanion(t, cs.den, cs.c, 0).TransferFunction(nil)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		tf := res.TF
		if e := cs.bandErr(tf); len(tf.Den[0]) != n+1 || !(e <= 1e-8) {
			fails++
			if fails <= 5 {
				t.Errorf("trial %d: den %v (want %v), num %v (want %v): band err %.3g", trial, tf.Den[0], cs.den, tf.Num[0][0], cs.num, e)
			}
		}
	}
	if fails > 0 {
		t.Errorf("%d of 400 failed", fails)
	}
}

// TestTransferFunctionSlowCompanionOrigins extends the sweep to n = 3..8 with
// 2..5 origin poles. The observability cut dropped states, and a kept
// k-fold origin pole split by the den tail's rounding noise^(1/k)
// (U3H63X). Every state must stay, the den tail be exactly zero, and the
// band response match; the residue of up to 1e-5 in some 8-state cases is
// Dgebal's partial grading of the companion (base and Osborne miss more).
func TestTransferFunctionSlowCompanionOrigins(t *testing.T) {
	r := rand.New(rand.NewPCG(77, 5))
	for trial := range 600 {
		n := 3 + r.IntN(6)
		k := 2 + r.IntN(4)
		cs := newSlowCompanionCase(r, n, k, min(k, n-1))
		res, err := slowCompanion(t, cs.den, cs.c, 0).TransferFunction(nil)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		den := res.TF.Den[0]
		zeros := 0
		for zeros < n && cs.den[n-zeros] == 0 {
			zeros++
		}
		if len(den) != n+1 || slices.ContainsFunc(den[n+1-zeros:], func(v float64) bool { return v != 0 }) {
			t.Errorf("trial %d: den %v, want %v", trial, den, cs.den)
			continue
		}
		if e := cs.bandErr(res.TF); !(e <= 2e-5) {
			t.Errorf("trial %d: den %v (want %v), num %v (want %v): band err %.3g", trial, den, cs.den, res.TF.Num[0][0], cs.num, e)
		}
	}
}

// TestTransferFunctionSlowCompanion covers companion realizations of slow
// poles (rho ≪ 1 against the unit superdiagonal), whose exact transfer
// function is the given coefficients. The Dgebal result is exact, but its
// untrimmed Markov noise far above the poles exceeded the error of the
// Osborne result, which loses a state; at those points the lost state barely
// shows, while in the pole band the den and the zero are wrong (U3H63X).
// MATLAB ss2tf keeps every state. The double-origin cases were exact before
// and must stay so.
func TestTransferFunctionSlowCompanion(t *testing.T) {
	a := 0.00079345703125
	z := 6.103515625e-05
	pair := Poly{1, 2 * 0.00030517578125, 0.00030517578125*0.00030517578125 + 0.00054931640625*0.00054931640625}
	cases := []struct {
		name string
		den  []float64
		c    []float64
		d    float64
		rho  float64
	}{
		{"double_pole_zero_s3", polyFromRealRoots(0, 0, 0, -a, -a), []float64{-2.494 * z, 2.494, 0, 0, 0}, 0, a},
		{"six_pole_mixed", polyFromRealRoots(0.0001220703125, -0.00048828125, -0.00079345703125, 0.00030517578125).Mul(pair),
			[]float64{-3 * -0.00067138671875, -3, 0, 0, 0, 0}, 0, 5e-4},
		{"stable_five_pole", []float64{1, 0.004265625, 6.073974609375e-06, 4.315250396728516e-09, 1.6911375522613525e-12, 2.739608753472567e-16},
			[]float64{-0.00102978515625, -0.59375, 0, 0, 0}, 0, 1e-3},
		{"unstable_pair_zero", []float64{1, 0.0025634765625, -6.07222318649292e-07, -7.907601684564725e-09, -8.747225166416683e-12, -3.142221183770333e-15},
			[]float64{-0.0025177001953125, -2.0625, 0, 0, 0}, 0, 1.5e-3},
		{"unstable_pair_zero_feedthrough", []float64{1, 0.0025634765625, -6.07222318649292e-07, -7.907601684564725e-09, -8.747225166416683e-12, -3.142221183770333e-15},
			[]float64{-0.0025177001953125, -2.0625, 0, 0, 0}, 0.5, 1.5e-3},
		{"double_origin_five", []float64{1, 1.049041748046875e-4, 1.051375875249505e-9, -5.483460907562687e-13, 0, 0},
			[]float64{1.0263912597528111e-07, -0.0005619489132617759, 3.0689903316269787, 0, 0}, 0, 1e-4},
		{"double_origin_six", []float64{1, 0.005462646484375, 2.0046718418598175e-07, -2.1223584667495743e-08, 8.022575008828636e-12, 0, 0},
			[]float64{4.075500212540671e-08, 0.0004899112466881561, 2.1053654729806555, 0, 0, 0}, 0, 1.5e-3},
		// Both conversions cut the last Hessenberg subdiagonal, an exact
		// product of tiny couplings, dropping the origin pole and the zero.
		{"origin_seven_cut", []float64{1, 0.0015716552734375, 1.0181684046983719e-06, 3.453202168657299e-10, 6.426315642821256e-14, 6.236857446227348e-18, 2.468235962998539e-22, 0},
			[]float64{-0.000118255615234375, 3.875, 0, 0, 0, 0, 0}, 0, 5e-4},
		{"double_integrator_zero", []float64{1, 0, 0}, []float64{-6.25e-9, 1}, 0, 6.25e-9},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys := slowCompanion(t, tc.den, tc.c, tc.d)
			n := len(tc.den) - 1
			res, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			den, num := res.TF.Den[0], res.TF.Num[0][0]
			if len(den) != n+1 || res.MinimalOrder != n {
				t.Fatalf("den %v order %d, want degree %d", den, res.MinimalOrder, n)
			}
			for k, want := range tc.den {
				if e := math.Abs(den[k]-want) / math.Pow(tc.rho, float64(k)); e > 1e-9 {
					t.Errorf("den[%d] = %v, want %v (scaled err %.3g)", k, den[k], want, e)
				}
			}
			wantNum := Poly{0}
			for _, v := range slices.Backward(tc.c) {
				wantNum = append(wantNum, v)
			}
			wantNum = wantNum.Add(Poly(tc.den).Scale(tc.d))
			for len(wantNum) > 1 && wantNum[0] == 0 {
				wantNum = wantNum[1:]
			}
			if len(num) != len(wantNum) {
				t.Fatalf("num %v, want %v", num, wantNum)
			}
			for k, want := range wantNum {
				if e := math.Abs(num[k]-want) / max(math.Abs(want), math.Abs(wantNum[0])*math.Pow(tc.rho, float64(k))); e > 1e-9 {
					t.Errorf("num[%d] = %v, want %v (scaled err %.3g)", k, num[k], want, e)
				}
			}
			for _, w := range []float64{1e-2, 1e-1, 3e-1, 1, 3, 1e1, 1e2} {
				s := complex(0, w*tc.rho)
				g := res.TF.eval(s)[0][0]
				want := ratRatioJ(wantNum, tc.den, w*tc.rho)
				if e := cmplx.Abs(g-want) / cmplx.Abs(want); e > 1e-8 {
					t.Errorf("G(%v) = %v, want %v (rel err %.3g)", s, g, want, e)
				}
			}
		})
	}
}
