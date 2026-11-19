package controlsys

import (
	"math"
	"math/big"

	"plantcontrol.org/v1/gonum/mat"
)

// The Dare X certificate of a float64 Newton solution: a two-sided Loewner
// enclosure X_c − D_L ⪯ X* ⪯ X_c + D_U of the stabilizing solution X* of
// the stored data around a double-double centre X_c, from the affine KYP
// matrix N_K(X) = [[Ric_K(X), −g_K(X)ᵀ], [−g_K(X), W(X)]], with
// Ric_K(X) = A_KᵀXA_K − EᵀXE + Q_K, A_K = A − BK,
// Q_K = Q + KᵀRK − SK − KᵀSᵀ, g_K(X) = RK − Sᵀ − BᵀXA_K, W(X) = R + BᵀXB:
//
//	N_K(X_lo) ≻ 0 ⇒ X_lo ⪯ X* (K-free: Ric_K − g_KᵀW⁻¹g_K is the Riccati
//	residual) and W(X*) ≻ 0;
//	−Ric_K(X_up) ⪰ 0 with K stabilizing ⇒ X* ⪯ X_up; and both with
//	X_up − X_lo ≻ 0 make K stabilizing, no eigenvalue trusted.
//
// A float64 Newton step cannot certify X where the Stein operator of A − BK
// maps its backward error beyond newtonTol (‖E⁻¹(A − BK)‖ ≈ 1e6 left X
// 1.4e-8 to 5e-8 off with steps of 3e-11), and no bound on that step
// separates those X from correct ones: the residuals here are measured,
// at shifted points, in evaluation arithmetic with a proven entrywise
// bound, double-double first, else 256 bits.

// encMaxRefine is the number of centre refinements of the enclosure.
const encMaxRefine = 3

// encPrec is the precision of the enclosure's second evaluation tier.
const encPrec = 256

// encNum is a matrix in evaluation arithmetic whose value is known
// exactly: double-double (encDD) or encPrec bits (encBig).
type encNum[M any] interface {
	mul(b M, ta bool) M
	add(b M, s float64) M
	scale(s float64) M
	t() M
	round() *mat.Dense
	abs() *mat.Dense
}

func fastTwoSum(a, b float64) (s, e float64) {
	s = a + b
	return s, b - (s - a)
}

// ddPlus is AccurateDWPlusDW (Joldes, Muller, Popescu 2017), relative
// error at most 3u² + 13u³.
func ddPlus(xh, xl, yh, yl float64) (float64, float64) {
	sh, sl := twoSum(xh, yh)
	th, tl := twoSum(xl, yl)
	sl += th
	sh, sl = fastTwoSum(sh, sl)
	sl += tl
	return fastTwoSum(sh, sl)
}

// ddTimes is DWTimesDW3, relative error at most 4u².
func ddTimes(xh, xl, yh, yl float64) (float64, float64) {
	ch := xh * yh
	cl1 := math.FMA(xh, yh, -ch)
	tl0 := xl * yl
	tl1 := math.FMA(xh, yl, tl0)
	cl2 := math.FMA(xl, yh, tl1)
	return fastTwoSum(ch, cl1+cl2)
}

type encDD struct {
	r, c int
	h, l []float64
}

func newEncDD(r, c int) encDD {
	return encDD{r, c, make([]float64, r*c), make([]float64, r*c)}
}

func encDDOf(a *mat.Dense) encDD {
	r, c := a.Dims()
	o := newEncDD(r, c)
	raw := a.RawMatrix()
	copyStrided(o.h, c, raw.Data, raw.Stride, r, c)
	return o
}

func (a encDD) mul(b encDD, ta bool) encDD {
	ar, ac := a.r, a.c
	if ta {
		ar, ac = ac, ar
	}
	o := newEncDD(ar, b.c)
	for i := range ar {
		for j := range b.c {
			var sh, sl float64
			for k := range ac {
				p := i*a.c + k
				if ta {
					p = k*a.c + i
				}
				ph, pl := ddTimes(a.h[p], a.l[p], b.h[k*b.c+j], b.l[k*b.c+j])
				sh, sl = ddPlus(sh, sl, ph, pl)
			}
			o.h[i*b.c+j], o.l[i*b.c+j] = sh, sl
		}
	}
	return o
}

func (a encDD) add(b encDD, s float64) encDD {
	o := newEncDD(a.r, a.c)
	for i := range a.h {
		bh, bl := b.h[i], b.l[i]
		if s != 1 {
			bh, bl = ddTimes(bh, bl, s, 0)
		}
		o.h[i], o.l[i] = ddPlus(a.h[i], a.l[i], bh, bl)
	}
	return o
}

func (a encDD) scale(s float64) encDD {
	o := newEncDD(a.r, a.c)
	for i := range a.h {
		o.h[i], o.l[i] = ddTimes(a.h[i], a.l[i], s, 0)
	}
	return o
}

func (a encDD) t() encDD {
	o := newEncDD(a.c, a.r)
	for i := range a.r {
		for j := range a.c {
			o.h[j*a.r+i], o.l[j*a.r+i] = a.h[i*a.c+j], a.l[i*a.c+j]
		}
	}
	return o
}

func (a encDD) round() *mat.Dense {
	d := make([]float64, len(a.h))
	for i := range d {
		d[i] = a.h[i] + a.l[i]
	}
	return mat.NewDense(a.r, a.c, d)
}

func (a encDD) abs() *mat.Dense {
	d := make([]float64, len(a.h))
	for i := range d {
		d[i] = math.Abs(a.h[i]) + math.Abs(a.l[i])
	}
	return mat.NewDense(a.r, a.c, d)
}

type encBig struct {
	r, c int
	d    []*big.Float
}

func newEncBig(r, c int) encBig {
	d := make([]*big.Float, r*c)
	for i := range d {
		d[i] = new(big.Float).SetPrec(encPrec)
	}
	return encBig{r, c, d}
}

func encBigOf(a *mat.Dense) encBig {
	r, c := a.Dims()
	o := newEncBig(r, c)
	for i := range r {
		for j := range c {
			o.d[i*c+j].SetFloat64(a.At(i, j))
		}
	}
	return o
}

func encBigDD(a encDD) encBig {
	o := newEncBig(a.r, a.c)
	t := new(big.Float).SetPrec(encPrec)
	for i := range o.d {
		o.d[i].SetFloat64(a.h[i])
		o.d[i].Add(o.d[i], t.SetFloat64(a.l[i]))
	}
	return o
}

func (a encBig) dd() encDD {
	o := newEncDD(a.r, a.c)
	t := new(big.Float).SetPrec(encPrec)
	for i, v := range a.d {
		h, _ := v.Float64()
		t.Sub(v, new(big.Float).SetFloat64(h))
		l, _ := t.Float64()
		o.h[i], o.l[i] = fastTwoSum(h, l)
	}
	return o
}

func (a encBig) mul(b encBig, ta bool) encBig {
	ar, ac := a.r, a.c
	if ta {
		ar, ac = ac, ar
	}
	o := newEncBig(ar, b.c)
	t := new(big.Float).SetPrec(encPrec)
	for i := range ar {
		for j := range b.c {
			for k := range ac {
				p := i*a.c + k
				if ta {
					p = k*a.c + i
				}
				t.Mul(a.d[p], b.d[k*b.c+j])
				o.d[i*b.c+j].Add(o.d[i*b.c+j], t)
			}
		}
	}
	return o
}

func (a encBig) add(b encBig, s float64) encBig {
	o := newEncBig(a.r, a.c)
	f := new(big.Float).SetPrec(encPrec).SetFloat64(s)
	t := new(big.Float).SetPrec(encPrec)
	for i := range a.d {
		t.Mul(b.d[i], f)
		o.d[i].Add(a.d[i], t)
	}
	return o
}

func (a encBig) scale(s float64) encBig {
	o := newEncBig(a.r, a.c)
	f := new(big.Float).SetPrec(encPrec).SetFloat64(s)
	for i := range a.d {
		o.d[i].Mul(a.d[i], f)
	}
	return o
}

func (a encBig) t() encBig {
	o := newEncBig(a.c, a.r)
	for i := range a.r {
		for j := range a.c {
			o.d[j*a.r+i].Set(a.d[i*a.c+j])
		}
	}
	return o
}

func (a encBig) round() *mat.Dense {
	d := make([]float64, len(a.d))
	for i, v := range a.d {
		d[i], _ = v.Float64()
	}
	return mat.NewDense(a.r, a.c, d)
}

func (a encBig) abs() *mat.Dense {
	m := a.round()
	raw := m.RawMatrix().Data
	for i, v := range raw {
		raw[i] = math.Abs(v)
	}
	return m
}

// solve returns a⁻¹b by Gaussian elimination with partial pivoting in
// encPrec bits; ok is false on a zero pivot.
func (a encBig) solve(b encBig) (encBig, bool) {
	n, nc := a.r, b.c
	w := make([][]*big.Float, n)
	for i := range n {
		w[i] = make([]*big.Float, n+nc)
		for j := range n {
			w[i][j] = new(big.Float).SetPrec(encPrec).Set(a.d[i*n+j])
		}
		for j := range nc {
			w[i][n+j] = new(big.Float).SetPrec(encPrec).Set(b.d[i*nc+j])
		}
	}
	t, f := new(big.Float).SetPrec(encPrec), new(big.Float).SetPrec(encPrec)
	for k := range n {
		p := k
		for i := k + 1; i < n; i++ {
			if t.Abs(w[i][k]).Cmp(f.Abs(w[p][k])) > 0 {
				p = i
			}
		}
		if w[p][k].Sign() == 0 {
			return encBig{}, false
		}
		w[k], w[p] = w[p], w[k]
		for i := k + 1; i < n; i++ {
			if w[i][k].Sign() == 0 {
				continue
			}
			f.Quo(w[i][k], w[k][k])
			for j := k; j < n+nc; j++ {
				t.Mul(f, w[k][j])
				w[i][j].Sub(w[i][j], t)
			}
		}
	}
	x := newEncBig(n, nc)
	for c := range nc {
		for i := n - 1; i >= 0; i-- {
			f.Set(w[i][n+c])
			for j := i + 1; j < n; j++ {
				t.Mul(w[i][j], x.d[j*nc+c])
				f.Sub(f, t)
			}
			x.d[i*nc+c].Quo(f, w[i][i])
		}
	}
	return x, true
}

func encSym[M encNum[M]](a M) M { return a.add(a.t(), 1).scale(0.5) }

// encOps is the stored Riccati data in one evaluation arithmetic.
type encOps[M encNum[M]] struct {
	a, b, q, r, s, e M
	hasS             bool
}

func (o *encOps[M]) acl(k M) M { return o.a.add(o.b.mul(k, false), -1) }

func (o *encOps[M]) ric(x, k M) M {
	ak := o.acl(k)
	q := o.q.add(k.mul(o.r.mul(k, false), true), 1)
	if o.hasS {
		sk := o.s.mul(k, false)
		q = q.add(sk, -1).add(sk.t(), -1)
	}
	return encSym(ak.mul(x.mul(ak, false), true).add(o.e.mul(x.mul(o.e, false), true), -1).add(q, 1))
}

func (o *encOps[M]) g(x, k M) M {
	gk := o.r.mul(k, false)
	if o.hasS {
		gk = gk.add(o.s.t(), -1)
	}
	return gk.add(o.b.mul(x.mul(o.acl(k), false), true), -1)
}

func (o *encOps[M]) w(x M) M { return encSym(o.r.add(o.b.mul(x.mul(o.b, false), true), 1)) }

// nmat returns N_K(x).
func (o *encOps[M]) nmat(x, k M, cat func(a, b, c M) M) M {
	return cat(o.ric(x, k), o.g(x, k).scale(-1), o.w(x))
}

// encData is the stored data of a gated Dare with its absolute values,
// in both evaluation arithmetics (encPrec bits on first use).
type encData struct {
	n, m                   int
	A, B, Q, R, S, E       *mat.Dense
	aA, aB, aQ, aR, aS, aE *mat.Dense
	dd                     encOps[encDD]
	bg                     *encOps[encBig]
}

func newEncData(A, B, Q, R, S, E *mat.Dense) *encData {
	n, m := B.Dims()
	if E == nil {
		E = eyeDense(n)
	}
	d := &encData{n: n, m: m, A: A, B: B, Q: Q, R: R, S: S, E: E,
		aA: absDense(A), aB: absDense(B), aQ: absDense(Q), aR: absDense(R), aE: absDense(E)}
	d.dd = encOps[encDD]{a: encDDOf(A), b: encDDOf(B), q: encDDOf(Q), r: encDDOf(R), e: encDDOf(E), hasS: S != nil}
	if S != nil {
		d.aS = absDense(S)
		d.dd.s = encDDOf(S)
	}
	return d
}

func (d *encData) big() *encOps[encBig] {
	if d.bg == nil {
		o := &encOps[encBig]{a: encBigOf(d.A), b: encBigOf(d.B), q: encBigOf(d.Q), r: encBigOf(d.R), e: encBigOf(d.E), hasS: d.S != nil}
		if d.S != nil {
			o.s = encBigOf(d.S)
		}
		d.bg = o
	}
	return d.bg
}

func mul3T(a, b, c *mat.Dense) *mat.Dense {
	var t, o mat.Dense
	t.Mul(b, c)
	o.Mul(a.T(), &t)
	return &o
}

// psi returns the entrywise magnitudes Ψ of the terms of Ric_K(x),
// g_K(x) and W(x) at |x|, |k|, rounded up by 1e-10 against their own
// float64 rounding.
func (d *encData) psi(ax, ak *mat.Dense) (pr, pg, pw *mat.Dense) {
	ah := mat.DenseCopyOf(d.aA)
	var bk mat.Dense
	bk.Mul(d.aB, ak)
	ah.Add(ah, &bk)
	pr = mul3T(ah, ax, ah)
	pr.Add(pr, mul3T(d.aE, ax, d.aE))
	pr.Add(pr, d.aQ)
	pr.Add(pr, mul3T(ak, d.aR, ak))
	rk := mat.NewDense(d.m, d.n, nil)
	rk.Mul(d.aR, ak)
	if d.aS != nil {
		var sk mat.Dense
		sk.Mul(d.aS, ak)
		pr.Add(pr, &sk)
		pr.Add(pr, sk.T())
		rk.Add(rk, d.aS.T())
	}
	pg = mul3T(d.aB, ax, ah)
	pg.Add(pg, rk)
	pw = mul3T(d.aB, ax, d.aB)
	pw.Add(pw, d.aR)
	for _, p := range []*mat.Dense{pr, pg, pw} {
		p.Scale(1+1e-10, p)
	}
	return
}

// gam returns the bound factor γ, |computed − exact| ≤ γ·Ψ entrywise, of
// Ric_K, g_K and W evaluated as encOps forms them in an arithmetic of
// unit roundoff v per operation (DWPlusDW 3v, DWTimesDW 4v for double-
// double, v = u²; v = 2⁻ᵖ at p bits). Each matrix product of inner length
// k is within 4(k+1)v·|A||B| (k products, k−1 sums); A_K = A − BK within
// 4(m+1)v·(|A| + |B||K|), entering the triple product twice; the triple
// products, KᵀRK, SK and up to six sums and the symmetrization within
// 4(2n + 2m + 12)v of Ψ in all, rounded up to 8(n + m + 8)v.
func (d *encData) gam(v float64) float64 { return 8 * float64(d.n+d.m+8) * v }

// encPDDD proves the symmetric z, known within h entrywise, positive
// definite in float64: scaled to unit diagonal, shifted by the scaled
// bound's ∞-norm (with z's rounding to float64) and 4n²u for the
// factorization, its Cholesky factorization must complete.
func encPDDD(z encDD, h *mat.Dense) bool {
	n := z.r
	u := math.Ldexp(1, -53)
	zf := z.round()
	sc := make([]float64, n)
	for i := range n {
		v := zf.At(i, i)
		if !(v > 0) || math.IsInf(v, 0) {
			return false
		}
		sc[i] = 1 / math.Sqrt(v)
	}
	c := 0.0
	for i := range n {
		r := 0.0
		for j := range n {
			r += (h.At(i, j) + u*math.Abs(zf.At(i, j))) * sc[i] * sc[j]
		}
		c = math.Max(c, r)
	}
	c += 4 * float64(n*n) * u
	if !(c < 1) {
		return false
	}
	s := mat.NewSymDense(n, nil)
	for i := range n {
		for j := i; j < n; j++ {
			v := 0.5 * (zf.At(i, j) + zf.At(j, i)) * sc[i] * sc[j]
			if i == j {
				v -= c
			}
			s.SetSym(i, j, v)
		}
	}
	var ch mat.Cholesky
	return ch.Factorize(s)
}

// encPDBig is encPDDD with the factorization in encPrec bits: scaled to
// unit diagonal by the rounded square roots of its diagonal, shifted by
// the scaled bound's ∞-norm and n²·2⁻²⁴⁰, its LDLᵀ pivots must stay
// positive.
func encPDBig(z encBig, h *mat.Dense) bool {
	n := z.r
	sq := make([]*big.Float, n)
	sf := make([]float64, n)
	for i := range n {
		v := z.d[i*n+i]
		if v.Sign() <= 0 {
			return false
		}
		sq[i] = new(big.Float).SetPrec(encPrec).Sqrt(v)
		sf[i], _ = sq[i].Float64()
		if !(sf[i] > 0) || math.IsInf(sf[i], 0) {
			return false
		}
	}
	c := 0.0
	for i := range n {
		r := 0.0
		for j := range n {
			r += h.At(i, j) / (sf[i] * sf[j])
		}
		c = math.Max(c, r)
	}
	c = c*(1+1e-6) + float64(n*n)*math.Ldexp(1, -240)
	if !(c < 1) {
		return false
	}
	w := make([][]*big.Float, n)
	t, f := new(big.Float).SetPrec(encPrec), new(big.Float).SetPrec(encPrec)
	for i := range n {
		w[i] = make([]*big.Float, n)
		for j := range n {
			t.Mul(sq[i], sq[j])
			w[i][j] = new(big.Float).SetPrec(encPrec).Quo(z.d[i*n+j], t)
		}
		w[i][i].Sub(w[i][i], new(big.Float).SetPrec(encPrec).SetFloat64(c))
	}
	for k := range n {
		if w[k][k].Sign() <= 0 {
			return false
		}
		for i := k + 1; i < n; i++ {
			if w[i][k].Sign() == 0 {
				continue
			}
			f.Quo(w[i][k], w[k][k])
			for j := k + 1; j < n; j++ {
				t.Mul(f, w[k][j])
				w[i][j].Sub(w[i][j], t)
			}
		}
	}
	return true
}

func catDD(r, g, w encDD) encDD {
	n, m := r.r, w.r
	o := newEncDD(n+m, n+m)
	set := func(i, j int, h, l float64) { o.h[i*(n+m)+j], o.l[i*(n+m)+j] = h, l }
	for i := range n {
		for j := range n {
			set(i, j, r.h[i*n+j], r.l[i*n+j])
		}
	}
	for i := range m {
		for j := range n {
			set(n+i, j, g.h[i*n+j], g.l[i*n+j])
			set(j, n+i, g.h[i*n+j], g.l[i*n+j])
		}
		for j := range m {
			set(n+i, n+j, w.h[i*m+j], w.l[i*m+j])
		}
	}
	return o
}

func catBig(r, g, w encBig) encBig {
	n, m := r.r, w.r
	o := newEncBig(n+m, n+m)
	for i := range n {
		for j := range n {
			o.d[i*(n+m)+j].Set(r.d[i*n+j])
		}
	}
	for i := range m {
		for j := range n {
			o.d[(n+i)*(n+m)+j].Set(g.d[i*n+j])
			o.d[j*(n+m)+n+i].Set(g.d[i*n+j])
		}
		for j := range m {
			o.d[(n+i)*(n+m)+n+j].Set(w.d[i*m+j])
		}
	}
	return o
}

func catBound(hr, hg, hw *mat.Dense) *mat.Dense {
	n, _ := hr.Dims()
	m, _ := hw.Dims()
	o := mat.NewDense(n+m, n+m, nil)
	o.Slice(0, n, 0, n).(*mat.Dense).Copy(hr)
	o.Slice(n, n+m, 0, n).(*mat.Dense).Copy(hg)
	o.Slice(0, n, n, n+m).(*mat.Dense).Copy(hg.T())
	o.Slice(n, n+m, n, n+m).(*mat.Dense).Copy(hw)
	return o
}

// encTier records the evaluation tier that proved an enclosure.
type encTier int

const (
	encNone encTier = iota
	encTierDD
	encTierBig
)

// tests proves −Ric_K(xu) ≻ 0 and N_K(xl) ≻ 0, double-double first.
func (d *encData) tests(xu, xl, k encDD) encTier {
	u := math.Ldexp(1, -53)
	ak := k.abs()
	prU, _, _ := d.psi(xu.abs(), ak)
	prL, pgL, pwL := d.psi(xl.abs(), ak)
	bound := func(g float64) (*mat.Dense, *mat.Dense) {
		var hu mat.Dense
		hu.Scale(g, prU)
		hl := catBound(prL, pgL, pwL)
		hl.Scale(g, hl)
		return &hu, hl
	}
	hu, hl := bound(d.gam(u * u))
	if encPDDD(d.dd.ric(xu, k).scale(-1), hu) && encPDDD(d.dd.nmat(xl, k, catDD), hl) {
		return encTierDD
	}
	bg := d.big()
	bu, bl, bkk := encBigDD(xu), encBigDD(xl), encBigDD(k)
	hu, hl = bound(d.gam(math.Ldexp(1, -encPrec)))
	if encPDBig(bg.ric(bu, bkk).scale(-1), hu) && encPDBig(bg.nmat(bl, bkk, catBig), hl) {
		return encTierBig
	}
	return encNone
}

// wPD proves W(x) ≻ 0.
func (d *encData) wPD(x, k encDD) bool {
	u := math.Ldexp(1, -53)
	_, _, pw := d.psi(x.abs(), k.abs())
	var h mat.Dense
	h.Scale(d.gam(u*u), pw)
	if encPDDD(d.dd.w(x), &h) {
		return true
	}
	h.Scale(d.gam(math.Ldexp(1, -encPrec)), pw)
	return encPDBig(d.big().w(encBigDD(x)), &h)
}

// diffPD proves a − b ≻ 0 for double-double a, b: their difference,
// within 3u² of itself in double-double, else exact in encPrec bits.
func diffPD(a, b encDD) bool {
	u := math.Ldexp(1, -53)
	dd := a.add(b, -1)
	var h mat.Dense
	h.Scale(4*u*u, dd.abs())
	if encPDDD(dd, &h) {
		return true
	}
	return encPDBig(encBigDD(a).add(encBigDD(b), -1), mat.NewDense(a.r, a.c, nil))
}

// gain returns the gain (R + BᵀXB)⁻¹(BᵀXA + Sᵀ) of the centre x in
// double-double: a float64 solve refined twice on its double-double
// residual. Any K serves the tests; the gain of x makes them tight.
func (d *encData) gain(x encDD) (encDD, bool) {
	o := &d.dd
	w := o.w(x)
	gg := o.b.mul(x.mul(o.a, false), true)
	if o.hasS {
		gg = gg.add(o.s.t(), 1)
	}
	var lu mat.LU
	lu.Factorize(w.round())
	var k0 mat.Dense
	if lu.SolveTo(&k0, false, gg.round()) != nil {
		return encDD{}, false
	}
	k := encDDOf(&k0)
	for range 2 {
		res := gg.add(w.mul(k, false), -1)
		var dk mat.Dense
		if lu.SolveTo(&dk, false, res.round()) != nil {
			return encDD{}, false
		}
		k = k.add(encDDOf(&dk), 1)
	}
	if !finiteSlice(k.round().RawMatrix().Data) {
		return encDD{}, false
	}
	return k, true
}

// stein returns D with EᵀDE − A_KᵀDA_K = N: a float64 solve of the Stein
// equation in F = E⁻¹A_K, with E folded in and back out in encPrec bits
// (D double-double), or D = Y without E.
func (d *encData) stein(k encDD, nrm *mat.Dense) (encDD, bool) {
	n := d.n
	ident := isIdentityDescriptor(d.E)
	var F *mat.Dense
	var eb encBig
	if ident {
		F = d.dd.acl(k).round()
	} else {
		bg := d.big()
		eb = bg.e
		f, ok := eb.solve(bg.acl(encBigDD(k)))
		if !ok {
			return encDD{}, false
		}
		F = f.round()
	}
	if !finiteSlice(F.RawMatrix().Data) {
		return encDD{}, false
	}
	Y, err := dlyap(mat.DenseCopyOf(F.T()), nrm, nil)
	if err != nil || !finiteSlice(Y.RawMatrix().Data) {
		return encDD{}, false
	}
	if ident {
		symmetrize(Y.RawMatrix().Data, n, n)
		return encDDOf(Y), true
	}
	et := eb.t()
	t1, ok := et.solve(encBigOf(Y))
	if !ok {
		return encDD{}, false
	}
	D, ok := et.solve(t1.t())
	if !ok {
		return encDD{}, false
	}
	o := encSym(D).dd()
	if !finiteSlice(o.round().RawMatrix().Data) {
		return encDD{}, false
	}
	return o, true
}

// encMeasure is newtonStep's measure of d relative to ref.
func encMeasure(d, ref *mat.Dense) float64 {
	n, _ := d.Dims()
	fr, rr, c1, r1 := 0.0, 0.0, 0.0, 0.0
	for j := range n {
		dc, rc := 0.0, 0.0
		for i := range n {
			fr += d.At(i, j) * d.At(i, j)
			rr += ref.At(i, j) * ref.At(i, j)
			dc += math.Abs(d.At(i, j))
			rc += math.Abs(ref.At(i, j))
		}
		c1, r1 = math.Max(c1, dc), math.Max(r1, rc)
	}
	if rr == 0 || r1 == 0 {
		return math.Inf(1)
	}
	return math.Max(math.Sqrt(fr/rr), c1/r1)
}

// encRadius bounds the measure of M relative to c for −D ⪯ M ⪯ D, D ⪰ 0:
// ‖M‖_F ≤ ‖D‖_F and each column sum of |M| at most √D_jj Σ_i √D_ii.
func encRadius(D, c *mat.Dense) float64 {
	n, _ := D.Dims()
	sum, mx := 0.0, 0.0
	for i := range n {
		v := D.At(i, i)
		if !(v >= 0) {
			return math.Inf(1)
		}
		sum += math.Sqrt(v)
		mx = math.Max(mx, math.Sqrt(v))
	}
	fd, fc, c1 := 0.0, 0.0, 0.0
	for j := range n {
		cc := 0.0
		for i := range n {
			fd += D.At(i, j) * D.At(i, j)
			fc += c.At(i, j) * c.At(i, j)
			cc += math.Abs(c.At(i, j))
		}
		c1 = math.Max(c1, cc)
	}
	if fc == 0 || c1 == 0 {
		return math.Inf(1)
	}
	return (1 + 1e-10) * math.Max(math.Sqrt(fd/fc), mx*sum/c1)
}

// encMajor returns a positive definite majorant of the positive part of
// the symmetric r, times scale: the Jacobi-scaled spectral positive part
// plus a tenth of its largest eigenvalue magnitude.
func encMajor(r *mat.Dense, scale float64) (*mat.Dense, bool) {
	n, _ := r.Dims()
	mx := 0.0
	for i := range n {
		for j := range n {
			mx = math.Max(mx, math.Abs(r.At(i, j)))
		}
	}
	if !(mx > 0) || math.IsInf(mx, 0) || math.IsNaN(mx) {
		return nil, false
	}
	a := make([]float64, n)
	for i := range n {
		a[i] = math.Max(math.Abs(r.At(i, i)), 1e-30*mx)
	}
	for range 2 {
		for i := range n {
			for j := range n {
				if v := r.At(i, j) * r.At(i, j) / a[j]; v > a[i] {
					a[i] = v
				}
			}
		}
	}
	s := mat.NewSymDense(n, nil)
	for i := range n {
		for j := i; j < n; j++ {
			s.SetSym(i, j, 0.5*(r.At(i, j)+r.At(j, i))/math.Sqrt(a[i]*a[j]))
		}
	}
	var es mat.EigenSym
	if !es.Factorize(s, true) {
		return nil, false
	}
	vals := es.Values(nil)
	var v mat.Dense
	es.VectorsTo(&v)
	lm := 0.0
	for _, x := range vals {
		lm = math.Max(lm, math.Abs(x))
	}
	o := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			x := 0.0
			for k := range n {
				x += v.At(i, k) * (math.Max(vals[k], 0) + 0.1*lm) * v.At(j, k)
			}
			o.Set(i, j, scale*x*math.Sqrt(a[i]*a[j]))
		}
	}
	symmetrize(o.RawMatrix().Data, n, n)
	return o, true
}

// encVerdict is the enclosure's outcome at one centre.
type encVerdict struct {
	tier encTier
	// ret: X itself within newtonTol of X*; else fl(X_c) is.
	ret bool
}

// enclose tries the X-direction radius D = s·X_c, then the Stein
// direction, around the centre c with its gain k, against the newtonTol
// budget m(X − X_c) + b(D_U + D_L) for X, else for fl(X_c).
func (d *encData) enclose(c, k encDD, x *mat.Dense) (encVerdict, bool) {
	n := d.n
	cf := c.round()
	var diff mat.Dense
	diff.Sub(x, cf)
	m := encMeasure(&diff, cf)
	mr := encMeasure(mat.NewDense(n, n, append([]float64(nil), c.l...)), cf)
	try := func(xu, xl encDD) (encVerdict, bool) {
		b := encRadius(xu.add(xl, -1).round(), cf)
		in, inR := m+b <= newtonTol, mr+b <= newtonTol
		if !in && !inR || !diffPD(xu, c) || !diffPD(c, xl) {
			return encVerdict{}, false
		}
		t := d.tests(xu, xl, k)
		return encVerdict{tier: t, ret: in}, t != encNone
	}
	k1 := encRadius(cf, cf)
	for _, mm := range []float64{m, mr} {
		if !(mm < newtonTol) || !(k1 > 0) || math.IsInf(k1, 0) {
			continue
		}
		s := 0.999 * (newtonTol - mm) / (2 * k1)
		if v, ok := try(c.add(c, s), c.add(c, -s)); ok {
			return v, true
		}
	}
	o := &d.dd
	r := o.ric(c, k).round()
	gc := o.g(c, k).round()
	var lu mat.LU
	lu.Factorize(o.w(c).round())
	var wg mat.Dense
	if lu.SolveTo(&wg, false, gc) != nil {
		return encVerdict{}, false
	}
	var rho mat.Dense
	rho.Mul(gc.T(), &wg)
	rho.Sub(&rho, r)
	symmetrize(rho.RawMatrix().Data, n, n)
	for _, sc := range []float64{1, 8, 64, 512} {
		nu, ok1 := encMajor(r, sc)
		nl, ok2 := encMajor(&rho, 2*sc)
		if !ok1 || !ok2 {
			return encVerdict{}, false
		}
		du, ok1 := d.stein(k, nu)
		dl, ok2 := d.stein(k, nl)
		if !ok1 || !ok2 {
			continue
		}
		if v, ok := try(c.add(du, 1), c.add(dl, -1)); ok {
			return v, true
		}
	}
	return encVerdict{}, false
}

// encResult is the enclosure's outcome for a float64 Newton X.
type encResult int

const (
	encFailed encResult = iota
	encCertified
	encCorrected
)

// encloseDare certifies the float64 Newton solution X of the stored Dare
// (A, B, Q, R, S, E), its gain K: from the centre X + Δ, Δ the float64
// Stein step of Ric_K(X) for the double-double gain of X, the enclosure
// at up to encMaxRefine further centres, each one Stein step on with a
// fresh double-double gain. While W(X_c) is not proved positive definite
// the step takes the returned K, and W-not-PD fails only at the last
// centre. corrected is fl(X_c) and its gain where X_c, not X, is within
// the budget.
func encloseDare(A, B, Q, R, S, E, X, K *mat.Dense) (out encResult, xc, kc *mat.Dense, tier encTier) {
	d := newEncData(A, B, Q, R, S, E)
	c := encDDOf(X)
	symmetrize(c.h, d.n, d.n)
	kret := encDDOf(K)
	for it := 0; it <= encMaxRefine+1; it++ {
		k, ok := d.gain(c)
		if ok {
			ok = d.wPD(c, k)
		}
		if !ok {
			if it == encMaxRefine+1 {
				return encFailed, nil, nil, encNone
			}
			k = kret
		} else if it > 0 {
			if v, ok := d.enclose(c, k, X); ok {
				if v.ret {
					return encCertified, nil, nil, v.tier
				}
				return encCorrected, c.round(), k.round(), v.tier
			}
		}
		if it == encMaxRefine+1 {
			break
		}
		dx, ok := d.stein(k, d.dd.ric(c, k).round())
		if !ok {
			break
		}
		c = encSym(c.add(dx, 1))
	}
	return encFailed, nil, nil, encNone
}

// confirm records X as confirmed in extended precision (an extended
// solve's rounded iterate, or one hewerSolution confirmed or replaced),
// which dareGate exempts while X stays unchanged.
func (r *RiccatiResult) confirm() {
	if r != nil && r.X != nil {
		r.confirmedX = mat.DenseCopyOf(r.X)
	}
}

func (r *RiccatiResult) xConfirmed() bool {
	if r.confirmedX == nil {
		return false
	}
	a, b := r.X.RawMatrix(), r.confirmedX.RawMatrix()
	if a.Rows != b.Rows || a.Cols != b.Cols {
		return false
	}
	for i := range a.Rows {
		for j := range a.Cols {
			if a.Data[i*a.Stride+j] != b.Data[i*b.Stride+j] {
				return false
			}
		}
	}
	return true
}

// dareGate holds a Dare result at Accuracy 0 whose X is of float64
// provenance to the enclosure of encloseDare on the stored data. A
// certified X is kept; a corrected one is replaced by fl(X_c) and its
// gain, whose K the gain gates then certify, else Accuracy 1. Without an
// enclosure, tier 2: hewerSolution, then the extended solve as
// lostCandidateDare runs it, each accepted only at Accuracy 0 (the
// extended pair replacing X and K unless X is within newtonTol and K
// within gainFloor of it), else Accuracy 1.
func dareGate(A, B, Q, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult) *RiccatiResult {
	n, m := B.Dims()
	if n == 0 || m == 0 || res.xConfirmed() {
		return res
	}
	var S, E *mat.Dense
	if opts != nil {
		S, E = opts.S, opts.E
	}
	out, xc, kc, _ := encloseDare(A, B, Q, R, S, E, res.X, res.K)
	switch out {
	case encCertified:
		return res
	case encCorrected:
		c := *res
		symmetrize(xc.RawMatrix().Data, n, n)
		c.X, c.K = xc, kc
		c.step, c.x1, c.xe, c.xeStep, c.xeX, c.xeLoose = 0, nil, nil, 0, 0, false
		c.Eig = make([]complex128, n)
		if !closedLoopStable(A, B, c.K, E, c.Eig) {
			return dareTier2(A, B, Q, R, opts, res)
		}
		red, _ := reduceInputs(A, B, Q, R, opts)
		if r, _, ok := gainOfX(A, B, Q, R, opts, red, &c); ok && r.accuracy == 0 {
			return r
		}
		c.accuracy = 1
		return &c
	}
	return dareTier2(A, B, Q, R, opts, res)
}

func dareTier2(A, B, Q, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult) *RiccatiResult {
	n, _ := B.Dims()
	red, _ := reduceInputs(A, B, Q, R, opts)
	if h, ok := hewerSolution(A, B, Q, R, opts, red, res); ok && h.accuracy == 0 {
		return h
	}
	var o RiccatiOpts
	if opts != nil {
		o = *opts
	}
	plain := isIdentityDescriptor(o.E) && o.plantE == nil
	cond := 1.0
	if plain {
		o.E = eyeDense(n)
	} else if c, ok := extendedDareCond(B, &o); ok {
		cond = c
	} else {
		res.accuracy = 1
		return res
	}
	r, ok := extendedDareSolve(A, B, Q, R, &o, cond)
	if !ok || r.accuracy != 0 {
		res.accuracy = 1
		return res
	}
	if plain {
		r.xe = nil
	}
	var dx mat.Dense
	dx.Sub(res.X, r.X)
	if encMeasure(&dx, r.X) <= newtonTol && relDist(res.K, r.K) <= gainFloor {
		res.confirm()
		return res
	}
	r.confirm()
	return r
}
