package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"
)

type mimoCompanionCase struct {
	den, b, c []float64
	n, m, p   int
	a         []float64
}

func newObservableMIMO(den, b, c []float64, m, p int) mimoCompanionCase {
	n := len(den) - 1
	return mimoCompanionCase{den, b, c, n, m, p, observableCompanion(den)}
}

// ratObservableNum returns the exact num (descending, len n+1, num[0] = 0)
// of c(sI − A)⁻¹b for A = observableCompanion(den): with y = N/D,
// N = Σ bₖsᵏ, the state equations s·xₖ = xₖ₋₁ − den[n−k]·y + bₖ give
// xₖ = Pₖ/D with Pₖ = (Pₖ₋₁ − den[n−k]·N + bₖ·D)/s, an exact division.
func ratObservableNum(den, b, c []float64) []float64 {
	n := len(den) - 1
	rat := func(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }
	D, N := make([]*big.Rat, n+1), make([]*big.Rat, n+1)
	for k := range n + 1 {
		D[k], N[k] = rat(den[n-k]), new(big.Rat)
	}
	for k := range n {
		N[k] = rat(b[k])
	}
	P := make([]*big.Rat, n+1)
	acc := make([]*big.Rat, n+1)
	for k := range n + 1 {
		P[k], acc[k] = new(big.Rat), new(big.Rat)
	}
	t := new(big.Rat)
	for k := range n {
		dk, bk, ck := rat(den[n-k]), rat(b[k]), rat(c[k])
		T := make([]*big.Rat, n+1)
		for l := range n + 1 {
			T[l] = new(big.Rat).Set(P[l])
			T[l].Sub(T[l], t.Mul(dk, N[l]))
			T[l].Add(T[l], t.Mul(bk, D[l]))
		}
		if T[0].Sign() != 0 {
			panic("ratObservableNum: inexact division")
		}
		for l := range n {
			P[l] = T[l+1]
			acc[l].Add(acc[l], t.Mul(ck, P[l]))
		}
		P[n] = new(big.Rat)
	}
	num := make([]float64, n+1)
	for l := range n + 1 {
		num[n-l], _ = acc[l].Float64()
	}
	return num
}

// mimoExact holds the exact entries of a case at its check points.
type mimoExact struct {
	w    []float64
	want [][]complex128
}

// bigEvalJ returns p(jω) for real descending p, in big.Float wide enough
// that the sums of the float products are exact here.
func bigEvalJ(p []float64, w float64) (re, im *big.Float) {
	const prec = 4096
	nf := func() *big.Float { return new(big.Float).SetPrec(prec) }
	wb := nf().SetFloat64(w)
	re, im = nf(), nf()
	for _, c := range p {
		nr := nf().Mul(im, wb)
		nr.Sub(nf().SetFloat64(c), nr)
		im = nf().Mul(re, wb)
		re = nr
	}
	return re, im
}

// newMIMOExact samples each exact entry at jω, ω = ρ·10^(k/2), k = −6..6,
// ρ from the exact den.
func newMIMOExact(cs mimoCompanionCase) mimoExact {
	var ex mimoExact
	rho := 0.0
	for k := 1; k <= cs.n; k++ {
		rho = max(rho, math.Pow(math.Abs(cs.den[k]), 1/float64(k)))
	}
	if rho == 0 {
		rho = 1
	}
	var dre, dim []*big.Float
	var q []*big.Float
	for k := range 13 {
		w := rho * math.Pow(10, float64(k-6)/2) * 1.003
		r, i := bigEvalJ(cs.den, w)
		m2 := new(big.Float).SetPrec(r.Prec()).Mul(r, r)
		m2.Add(m2, new(big.Float).SetPrec(r.Prec()).Mul(i, i))
		ex.w, dre, dim, q = append(ex.w, w), append(dre, r), append(dim, i), append(q, m2)
	}
	bcol := make([]float64, cs.n)
	for i := range cs.p {
		crow := cs.c[i*cs.n : (i+1)*cs.n]
		for j := range cs.m {
			for k := range cs.n {
				bcol[k] = cs.b[k*cs.m+j]
			}
			num := ratObservableNum(cs.den, bcol, crow)
			wants := make([]complex128, len(ex.w))
			for k, w := range ex.w {
				if q[k].Sign() == 0 {
					wants[k] = cmplx.Inf()
					continue
				}
				nr, ni := bigEvalJ(num, w)
				pr := nr.Prec()
				re := new(big.Float).SetPrec(pr).Mul(nr, dre[k])
				re.Add(re, new(big.Float).SetPrec(pr).Mul(ni, dim[k]))
				im := new(big.Float).SetPrec(pr).Mul(ni, dre[k])
				im.Sub(im, new(big.Float).SetPrec(pr).Mul(nr, dim[k]))
				fr, _ := re.Quo(re, q[k]).Float64()
				fi, _ := im.Quo(im, q[k]).Float64()
				wants[k] = complex(fr, fi)
			}
			ex.want = append(ex.want, wants)
		}
	}
	return ex
}

// respErr is the worst relative response error of tf over the entries.
func (ex mimoExact) respErr(tf *TransferFunc, m int) float64 {
	worst := 0.0
	for e, wants := range ex.want {
		i, j := e/m, e%m
		for k, want := range wants {
			if want == 0 || cmplx.IsInf(want) || cmplx.IsNaN(want) {
				continue
			}
			r := cmplx.Abs(tf.eval(complex(0, ex.w[k]))[i][j]-want) / cmplx.Abs(want)
			if math.IsNaN(r) {
				r = math.Inf(1)
			}
			worst = max(worst, r)
		}
	}
	return worst
}

// newMIMOCompanionCase draws an observable companion with p outputs, one
// row eₙᵀ (at a random position) and the others dense, and graded B.
func newMIMOCompanionCase(r *rand.Rand) mimoCompanionCase {
	n := 2 + r.IntN(9)
	m := 1 + r.IntN(3)
	p := 2 + r.IntN(2)
	scale := math.Ldexp(1, -20+r.IntN(28))
	den := ratPolyFromRoots(hessenbergSweepRoots(r, n, scale))
	b := make([]float64, n*m)
	for k := range b {
		if r.IntN(3) > 0 {
			b[k] = math.Ldexp(float64(r.IntN(129)-64), r.IntN(41)-25)
		}
	}
	c := make([]float64, p*n)
	unit := r.IntN(p)
	for i := range p {
		if i == unit {
			c[i*n+n-1] = 1
			continue
		}
		for k := range n {
			c[i*n+k] = float64(r.IntN(17)-8) / 4
		}
	}
	return newObservableMIMO(den, b, c, m, p)
}

func convertMIMOCompanion(t *testing.T, cs mimoCompanionCase, noDirect bool) *TransferFunc {
	t.Helper()
	sys, err := NewFromSlices(cs.n, cs.m, cs.p, cs.a, cs.b, cs.c, make([]float64, cs.p*cs.m), 0)
	if err != nil {
		t.Fatal(err)
	}
	cv := newRowRealizationConverter(sys, &TransferFuncOpts{})
	cv.noDirect = noDirect
	res, err := cv.convert()
	if err != nil {
		t.Fatal(err)
	}
	return res.TF
}

// A multi-output observable companion with C = [eₙᵀ; dense]: the direct
// row 0 hid Osborne's miss from the check points above the poles, so every
// row switched to Osborne, whose staircase row 1 had wrong low den
// coefficients (598: a sign-flipped DC term; 473: a right-half-plane root).
func TestTransferFunctionObservableCompanionMixedRows(t *testing.T) {
	cases := []struct {
		name string
		cs   mimoCompanionCase
	}{
		{"598", newObservableMIMO([]float64{1, 8.791685104370117e-07, 7.416289804496046e-14, -5.161607022330893e-21},
			[]float64{966656, 0.0025634765625, 0, 0, 0, 0},
			[]float64{0, 0, 1, -1.75, 2, 0.25}, 2, 2)},
		{"473", newObservableMIMO([]float64{1, 0.01373291015625, 8.013099431991577e-05, 2.5520193958072923e-07, 4.621997923281995e-10, 4.52900172775311e-13, 2.392364643532841e-16, 8.69746111271906e-20},
			[]float64{4.492176230996847e-08, -6656, 0.0025234222412109375, 544, 47.25, 0, 294912, 13.5, 0, -0.5625, 0, -0.0078125, 0, -0.0107421875},
			[]float64{0, 0, 0, 0, 0, 0, 1, -1, -2, -2, -1.75, 2, 0.25, 0}, 2, 2)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			cs := tc.cs
			tf := convertMIMOCompanion(t, cs, false)
			den := cs.den
			bcol := make([]float64, cs.n)
			for i := range cs.p {
				crow := cs.c[i*cs.n : (i+1)*cs.n]
				for j := range cs.m {
					for k := range cs.n {
						bcol[k] = cs.b[k*cs.m+j]
					}
					exDen, exNum := ratSISOTF(cs.a, bcol, crow, cs.n)
					if !slices.Equal(exDen, den) || !slices.Equal(exNum, ratObservableNum(den, bcol, crow)) {
						t.Fatalf("ratObservableNum %v, Faddeev–LeVerrier %v", ratObservableNum(den, bcol, crow), exNum)
					}
				}
				got := tf.Den[i]
				if len(got) != cs.n+1 {
					t.Fatalf("row %d den %v, want %v", i, got, den)
				}
				for k, want := range den {
					if math.Abs(got[k]-want) > 1e-12*math.Abs(want) {
						t.Errorf("row %d den[%d] = %.17g, want %.17g", i, k, got[k], want)
					}
				}
				if rhpRoots(got) != rhpRoots(den) {
					t.Errorf("row %d den %v: RHP roots %d, want %d", i, got, rhpRoots(got), rhpRoots(den))
				}
			}
			ex := newMIMOExact(cs)
			for e, wants := range ex.want {
				for k := range cs.n {
					bcol[k] = cs.b[k*cs.m+e%cs.m]
				}
				num := ratObservableNum(den, bcol, cs.c[e/cs.m*cs.n:(e/cs.m+1)*cs.n])
				for k, w := range ex.w {
					if r := ratRatioJ(num, den, w); cmplx.Abs(wants[k]-r) > 1e-15*cmplx.Abs(r) {
						t.Fatalf("entry %d at %g: %v, ratRatioJ %v", e, w, wants[k], r)
					}
				}
			}
			eBase := ex.respErr(convertMIMOCompanion(t, cs, true), cs.m)
			if e := ex.respErr(tf, cs.m); e > max(2*eBase, 1e-12) {
				t.Errorf("response err %.3g, without the path %.3g", e, eBase)
			}
		})
	}
}

// TestTransferFunctionObservableCompanionMIMOSweep converts seeded
// multi-output observable companions, one row eₙᵀ and the others dense,
// with and without the given-basis path (DZG5LV) against the exact big.Rat
// entries: the path must never be worse, nor flip a row's right-half-plane
// root count.
func TestTransferFunctionObservableCompanionMIMOSweep(t *testing.T) {
	skipSlowUnderRace(t)
	r := rand.New(rand.NewPCG(99, 77))
	const trials = 1500
	for trial := range trials {
		cs := newMIMOCompanionCase(r)
		tfNew, tfBase := convertMIMOCompanion(t, cs, false), convertMIMOCompanion(t, cs, true)
		ex := newMIMOExact(cs)
		eNew, eBase := ex.respErr(tfNew, cs.m), ex.respErr(tfBase, cs.m)
		if eNew > 2*eBase && eNew > 1e-12 {
			t.Errorf("trial %d: err %.3g, without the path %.3g; A %v B %v C %v", trial, eNew, eBase, cs.a, cs.b, cs.c)
		}
		den := cs.den
		rhp := rhpRoots(den)
		for i := range cs.p {
			if len(tfNew.Den[i]) == cs.n+1 && len(tfBase.Den[i]) == cs.n+1 &&
				rhpRoots(tfNew.Den[i]) != rhp && rhpRoots(tfBase.Den[i]) == rhp {
				t.Errorf("trial %d: row %d den %v flips the RHP root count of %v", trial, i, tfNew.Den[i], den)
			}
		}
	}
}
