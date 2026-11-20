package controlsys

import (
	"math"
	"math/big"
	"sync"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
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

// ddTimes is DWTimesDW3, relative error at most 4u². The conversions
// round the products: unconverted, the compiler may fuse xh·yh into the
// final sum (an FMA on arm64), and cl1 no longer is its rounding error.
func ddTimes(xh, xl, yh, yl float64) (float64, float64) {
	ch := float64(xh * yh)
	cl1 := math.FMA(xh, yh, -ch)
	tl0 := float64(xl * yl)
	tl1 := math.FMA(xh, yl, tl0)
	cl2 := math.FMA(xl, yh, tl1)
	return fastTwoSum(ch, cl1+cl2)
}

type encDD struct {
	r, c int
	h, l []float64
	ws   *encWS
}

// encWS is the workspace of one gate call: double-double entries, float64
// matrices and the Cholesky factorizations of the definiteness tests, all
// reused from the pool (encWSs) so that a gate passing at X allocates
// nothing once warm. Values from it die with the call (release).
type encWS struct {
	chunks [][]float64
	ci, co int
	dense  []*mat.Dense
	dn     int
	cents  []*encCentre
	cn     int
	sym    map[int]*mat.SymDense
	chol   map[int]*mat.Cholesky
	data   encData
}

var encWSs = sync.Pool{New: func() any {
	return &encWS{sym: map[int]*mat.SymDense{}, chol: map[int]*mat.Cholesky{}}
}}

// floats returns n zeroed float64s.
func (w *encWS) floats(n int) []float64 {
	const chunk = 4096
	for {
		if w.ci == len(w.chunks) {
			w.chunks = append(w.chunks, make([]float64, max(chunk, n)))
		}
		c := w.chunks[w.ci]
		if w.co+n <= len(c) {
			v := c[w.co : w.co+n : w.co+n]
			w.co += n
			clear(v)
			return v
		}
		w.ci, w.co = w.ci+1, 0
	}
}

// mat returns a zeroed r×c matrix.
func (w *encWS) mat(r, c int) *mat.Dense {
	if w.dn == len(w.dense) {
		w.dense = append(w.dense, new(mat.Dense))
	}
	m := w.dense[w.dn]
	w.dn++
	m.Reset()
	m.ReuseAs(r, c)
	return m
}

func (w *encWS) centre() *encCentre {
	if w.cn == len(w.cents) {
		w.cents = append(w.cents, new(encCentre))
	}
	e := w.cents[w.cn]
	w.cn++
	*e = encCentre{}
	return e
}

func (w *encWS) symDense(n int) *mat.SymDense {
	s, ok := w.sym[n]
	if !ok {
		s = mat.NewSymDense(n, nil)
		w.sym[n] = s
	}
	return s
}

func (w *encWS) cholesky(n int) *mat.Cholesky {
	c, ok := w.chol[n]
	if !ok {
		c = new(mat.Cholesky)
		w.chol[n] = c
	}
	return c
}

func (w *encWS) reset() { w.ci, w.co, w.dn, w.cn = 0, 0, 0, 0 }

func newEncDD(ws *encWS, r, c int) encDD {
	if ws != nil {
		return encDD{r, c, ws.floats(r * c), ws.floats(r * c), ws}
	}
	return encDD{r, c, make([]float64, r*c), make([]float64, r*c), nil}
}

func encDDOf(a *mat.Dense) encDD { return encDDIn(nil, a) }

func encDDIn(ws *encWS, a *mat.Dense) encDD {
	r, c := a.Dims()
	o := newEncDD(ws, r, c)
	raw := a.RawMatrix()
	copyStrided(o.h, c, raw.Data, raw.Stride, r, c)
	return o
}

// matOf is a zeroed r×c float64 matrix from ws, else allocated.
func matOf(ws *encWS, r, c int) *mat.Dense {
	if ws != nil {
		return ws.mat(r, c)
	}
	return mat.NewDense(r, c, nil)
}

func (a encDD) mul(b encDD, ta bool) encDD {
	ar, ac := a.r, a.c
	if ta {
		ar, ac = ac, ar
	}
	o := newEncDD(a.ws, ar, b.c)
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
	o := newEncDD(a.ws, a.r, a.c)
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
	o := newEncDD(a.ws, a.r, a.c)
	for i := range a.h {
		o.h[i], o.l[i] = ddTimes(a.h[i], a.l[i], s, 0)
	}
	return o
}

func (a encDD) t() encDD {
	o := newEncDD(a.ws, a.c, a.r)
	for i := range a.r {
		for j := range a.c {
			o.h[j*a.r+i], o.l[j*a.r+i] = a.h[i*a.c+j], a.l[i*a.c+j]
		}
	}
	return o
}

func (a encDD) round() *mat.Dense {
	m := matOf(a.ws, a.r, a.c)
	d := m.RawMatrix().Data
	for i := range d {
		d[i] = a.h[i] + a.l[i]
	}
	return m
}

func (a encDD) abs() *mat.Dense {
	m := matOf(a.ws, a.r, a.c)
	d := m.RawMatrix().Data
	for i := range d {
		d[i] = math.Abs(a.h[i]) + math.Abs(a.l[i])
	}
	return m
}

type encBig struct {
	r, c int
	d    []*big.Float
	ar   *encArena
}

// encArena hands out zeroed encPrec-bit values whose mantissas outlive
// one gate call (encArenas), so the encPrec tier does not allocate once
// warm; reset recycles every value at once.
type encArena struct {
	chunks [][]big.Float
	ptrs   []*big.Float
	next   int
}

var encArenas = sync.Pool{New: func() any { return new(encArena) }}

func (a *encArena) get() *big.Float {
	const chunk = 256
	if a.next == len(a.chunks)*chunk {
		a.chunks = append(a.chunks, make([]big.Float, chunk))
	}
	z := &a.chunks[a.next/chunk][a.next%chunk]
	a.next++
	return z.SetPrec(encPrec).SetInt64(0)
}

func (a *encArena) slice(n int) []*big.Float {
	if cap(a.ptrs)-len(a.ptrs) < n {
		a.ptrs = make([]*big.Float, 0, max(2*cap(a.ptrs), n, 1024))
	}
	d := a.ptrs[len(a.ptrs) : len(a.ptrs)+n]
	a.ptrs = a.ptrs[:len(a.ptrs)+n]
	for i := range d {
		d[i] = a.get()
	}
	return d
}

func (a *encArena) reset() { a.next, a.ptrs = 0, a.ptrs[:0] }

func newEncBig(ar *encArena, r, c int) encBig {
	return encBig{r, c, ar.slice(r * c), ar}
}

func encBigOf(ar *encArena, a *mat.Dense) encBig {
	r, c := a.Dims()
	o := newEncBig(ar, r, c)
	for i := range r {
		for j := range c {
			o.d[i*c+j].SetFloat64(a.At(i, j))
		}
	}
	return o
}

func encBigDD(ar *encArena, a encDD) encBig {
	o := newEncBig(ar, a.r, a.c)
	t := ar.get()
	for i := range o.d {
		o.d[i].SetFloat64(a.h[i])
		o.d[i].Add(o.d[i], t.SetFloat64(a.l[i]))
	}
	return o
}

func (a encBig) dd() encDD {
	o := newEncDD(nil, a.r, a.c)
	t, hf := a.ar.get(), a.ar.get()
	for i, v := range a.d {
		h, _ := v.Float64()
		t.Sub(v, hf.SetFloat64(h))
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
	o := newEncBig(a.ar, ar, b.c)
	t := a.ar.get()
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
	o := newEncBig(a.ar, a.r, a.c)
	f := a.ar.get().SetFloat64(s)
	t := a.ar.get()
	for i := range a.d {
		t.Mul(b.d[i], f)
		o.d[i].Add(a.d[i], t)
	}
	return o
}

func (a encBig) scale(s float64) encBig {
	o := newEncBig(a.ar, a.r, a.c)
	f := a.ar.get().SetFloat64(s)
	for i := range a.d {
		o.d[i].Mul(a.d[i], f)
	}
	return o
}

func (a encBig) t() encBig {
	o := newEncBig(a.ar, a.c, a.r)
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
			w[i][j] = a.ar.get().Set(a.d[i*n+j])
		}
		for j := range nc {
			w[i][n+j] = a.ar.get().Set(b.d[i*nc+j])
		}
	}
	t, f := a.ar.get(), a.ar.get()
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
	x := newEncBig(a.ar, n, nc)
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
	// eIdent: E is the identity, EᵀXE = X exactly (a product by 1 and
	// sums of exact zeros), so it is not formed.
	eIdent bool
}

// ete is EᵀxE.
func (o *encOps[M]) ete(x M) M {
	if o.eIdent {
		return x
	}
	return o.e.mul(x.mul(o.e, false), true)
}

func (o *encOps[M]) acl(k M) M { return o.a.add(o.b.mul(k, false), -1) }

func (o *encOps[M]) w(x M) M { return encSym(o.r.add(o.b.mul(x.mul(o.b, false), true), 1)) }

// encData is the stored data of a gated Dare with its absolute values,
// in both evaluation arithmetics (encPrec bits on first use).
type encData struct {
	n, m                   int
	A, B, Q, R, S, E       *mat.Dense
	aA, aB, aQ, aR, aS, aE *mat.Dense
	aSt                    *mat.Dense
	dd                     encOps[encDD]
	bg                     *encOps[encBig]
	ar                     *encArena
	ws                     *encWS
}

// release returns the encPrec arena and the workspace; no value of d is
// used after.
func (d *encData) release() {
	if d.ar != nil {
		d.ar.reset()
		encArenas.Put(d.ar)
		d.ar, d.bg = nil, nil
	}
	if ws := d.ws; ws != nil {
		ws.reset()
		*d = encData{}
		encWSs.Put(ws)
	}
}

func newEncData(A, B, Q, R, S, E *mat.Dense) *encData {
	n, m := B.Dims()
	ws := encWSs.Get().(*encWS)
	if E == nil {
		E = ws.mat(n, n)
		for i := range n {
			E.Set(i, i, 1)
		}
	}
	d := &ws.data
	*d = encData{n: n, m: m, A: A, B: B, Q: Q, R: R, S: S, E: E, ws: ws}
	d.aA, d.aB, d.aQ, d.aR, d.aE = d.absOf(A), d.absOf(B), d.absOf(Q), d.absOf(R), d.absOf(E)
	d.dd = encOps[encDD]{a: encDDIn(ws, A), b: encDDIn(ws, B), q: encDDIn(ws, Q), r: encDDIn(ws, R), e: encDDIn(ws, E), hasS: S != nil, eIdent: isIdentityDescriptor(E)}
	if S != nil {
		d.aS = d.absOf(S)
		d.aSt = d.transposed(d.aS)
		d.dd.s = encDDIn(ws, S)
	}
	return d
}

// absOf is |a| in the workspace.
func (d *encData) absOf(a *mat.Dense) *mat.Dense {
	r, c := a.Dims()
	o := d.ws.mat(r, c)
	o.Copy(a)
	raw := o.RawMatrix().Data
	for i, v := range raw {
		raw[i] = math.Abs(v)
	}
	return o
}

// transposed is aᵀ in the workspace.
func (d *encData) transposed(a *mat.Dense) *mat.Dense {
	r, c := a.Dims()
	o := d.ws.mat(c, r)
	for i := range r {
		for j := range c {
			o.Set(j, i, a.At(i, j))
		}
	}
	return o
}

func (d *encData) big() *encOps[encBig] {
	if d.bg == nil {
		if d.ar == nil {
			d.ar = encArenas.Get().(*encArena)
		}
		ar := d.ar
		o := &encOps[encBig]{a: encBigOf(ar, d.A), b: encBigOf(ar, d.B), q: encBigOf(ar, d.Q), r: encBigOf(ar, d.R), e: encBigOf(ar, d.E), hasS: d.S != nil, eIdent: d.dd.eIdent}
		if d.S != nil {
			o.s = encBigOf(ar, d.S)
		}
		d.bg = o
	}
	return d.bg
}

// mul3T is aᵀbc as Dense.Mul forms it (Gemm, aᵀ by its Trans flag).
func (d *encData) mul3T(a, b, c *mat.Dense) *mat.Dense {
	br, _ := b.Dims()
	_, cc := c.Dims()
	_, ac := a.Dims()
	t := d.ws.mat(br, cc)
	t.Mul(b, c)
	o := d.ws.mat(ac, cc)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, a.RawMatrix(), t.RawMatrix(), 0, o.RawMatrix())
	return o
}

// psi returns the entrywise magnitudes Ψ of the terms of Ric_K(x),
// g_K(x) and W(x) at |x|, |k|, so that γ·Ψ (gam) bounds the evaluation
// error, rounded up by 1e-10 against their own float64 rounding. ahat
// bounds the computed |A_K| of every tier (encCentre: the double-double
// A_K plus twice its formation error); the triple product then carries
// Â = |A| + |B||K| once per side only through A_K's own error:
//
// The computed A_K is A_K + δ with |δ| ≤ ε·Â, ε = 4(m + 1)v (inner length
// m, then one sum). The products in the computed A_K, here Ã, are within
// γ_P·|Ã|ᵀ|x||Ã| of ÃᵀxÃ, γ_P ≤ γ. And ÃᵀxÃ − A_KᵀxA_K = δᵀxÃ + Ãᵀxδ −
// δᵀxδ, so |·| ≤ ε(Âᵀ|x||Ã| + |Ã|ᵀ|x|Â) + ε²Âᵀ|x|Â. With |Ã| ≤ ahat,
// Ψ_R holds ahatᵀ|x|ahat, (ε/γ)(Âᵀ|x|ahat + ahatᵀ|x|Â) and (ε²/γ)Âᵀ|x|Â;
// ε/γ = (m + 1)/(2(n + m + 8)) and ε²/γ ≤ 2(m + 1)v are exact factors of
// gam's γ = 8(n + m + 8)v. Likewise Ψ_g holds |B|ᵀ|x|ahat and
// (ε/γ)|B|ᵀ|x|Â. Without ahat both carry Â on both sides, the bound
// gam was first derived for, which double counted A_K's error against
// the far larger Âᵀ|x|Â under cheap control.
func (d *encData) psi(ax, ak, ahat *mat.Dense, v float64) (pr, pg, pw *mat.Dense) {
	n, m := d.n, d.m
	ah := d.ws.mat(n, n)
	ah.Copy(d.aA)
	bk := d.ws.mat(n, n)
	bk.Mul(d.aB, ak)
	ah.Add(ah, bk)
	if ahat == nil {
		pr = d.mul3T(ah, ax, ah)
		pg = d.mul3T(d.aB, ax, ah)
	} else {
		c1 := float64(m+1) / float64(2*(n+m+8))
		c2 := 2 * float64(m+1) * v
		pr = d.mul3T(ahat, ax, ahat)
		x1 := d.mul3T(ah, ax, ahat)
		x1.Scale(c1, x1)
		pr.Add(pr, x1)
		pr.Add(pr, d.transposed(x1))
		x2 := d.mul3T(ah, ax, ah)
		x2.Scale(c2, x2)
		pr.Add(pr, x2)
		pg = d.mul3T(d.aB, ax, ahat)
		g1 := d.mul3T(d.aB, ax, ah)
		g1.Scale(c1, g1)
		pg.Add(pg, g1)
	}
	if d.dd.eIdent {
		pr.Add(pr, ax)
	} else {
		pr.Add(pr, d.mul3T(d.aE, ax, d.aE))
	}
	pr.Add(pr, d.aQ)
	pr.Add(pr, d.mul3T(ak, d.aR, ak))
	rk := d.ws.mat(m, n)
	rk.Mul(d.aR, ak)
	if d.aS != nil {
		sk := d.ws.mat(n, n)
		sk.Mul(d.aS, ak)
		pr.Add(pr, sk)
		pr.Add(pr, d.transposed(sk))
		rk.Add(rk, d.aSt)
	}
	pg.Add(pg, rk)
	pw = d.mul3T(d.aB, ax, d.aB)
	pw.Add(pw, d.aR)
	pr.Scale(1+1e-10, pr)
	pg.Scale(1+1e-10, pg)
	pw.Scale(1+1e-10, pw)
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
	var sc []float64
	if z.ws != nil {
		sc = z.ws.floats(n)
	} else {
		sc = make([]float64, n)
	}
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
	var s *mat.SymDense
	var ch *mat.Cholesky
	if z.ws != nil {
		s, ch = z.ws.symDense(n), z.ws.cholesky(n)
	} else {
		s, ch = mat.NewSymDense(n, nil), new(mat.Cholesky)
	}
	for i := range n {
		for j := i; j < n; j++ {
			v := 0.5 * (zf.At(i, j) + zf.At(j, i)) * sc[i] * sc[j]
			if i == j {
				v -= c
			}
			s.SetSym(i, j, v)
		}
	}
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
		sq[i] = z.ar.get().Sqrt(v)
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
	t, f := z.ar.get(), z.ar.get()
	for i := range n {
		w[i] = make([]*big.Float, n)
		for j := range n {
			t.Mul(sq[i], sq[j])
			w[i][j] = z.ar.get().Quo(z.d[i*n+j], t)
		}
		w[i][i].Sub(w[i][i], f.SetFloat64(c))
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
	o := newEncDD(r.ws, n+m, n+m)
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
	o := newEncBig(r.ar, n+m, n+m)
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

func catBound(ws *encWS, hr, hg, hw *mat.Dense) *mat.Dense {
	n, _ := hr.Dims()
	m, _ := hw.Dims()
	o := matOf(ws, n+m, n+m)
	for i := range n {
		for j := range n {
			o.Set(i, j, hr.At(i, j))
		}
	}
	for i := range m {
		for j := range n {
			o.Set(n+i, j, hg.At(i, j))
			o.Set(j, n+i, hg.At(i, j))
		}
		for j := range m {
			o.Set(n+i, n+j, hw.At(i, j))
		}
	}
	return o
}

// encTiers selects evaluation tiers.
type encTiers int

const (
	encDDOnly encTiers = 1 << iota
	encBigOnly
	encAllTiers = encDDOnly | encBigOnly
)

// encTier records the evaluation tier that proved an enclosure.
type encTier int

const (
	encNone encTier = iota
	encTierDD
	encTierBig
)

// encPieces are the forms at a centre c with gain k that the tests
// combine: Ric_K(c), Q_K, g_K(c), W(c), RK − Sᵀ and A_K. Every test
// matrix is affine in them and in the radius, so the test points c ± D are
// never rounded.
type encPieces[M encNum[M]] struct {
	r, qk, g, w, rks, ak M
}

func encPiecesOf[M encNum[M]](o *encOps[M], c, k M) encPieces[M] {
	var p encPieces[M]
	p.ak = o.acl(k)
	p.qk = o.q.add(k.mul(o.r.mul(k, false), true), 1)
	p.rks = o.r.mul(k, false)
	if o.hasS {
		sk := o.s.mul(k, false)
		p.qk = p.qk.add(sk, -1).add(sk.t(), -1)
		p.rks = p.rks.add(o.s.t(), -1)
	}
	cak := c.mul(p.ak, false)
	p.r = encSym(p.ak.mul(cak, true).add(o.ete(c), -1).add(p.qk, 1))
	p.g = p.rks.add(o.b.mul(cak, true), -1)
	p.w = encSym(o.r.add(o.b.mul(c.mul(o.b, false), true), 1))
	return p
}

// xdir returns −Ric_K(c(1 + s)) = sQ_K − (1 + s)Ric_K(c) and
// N_K(c(1 − s)) = [[(1 − s)r + sQ_K, −((1 − s)g + s(RK − Sᵀ))ᵀ], [·,
// (1 − s)W(c) + sR]].
func encXDir[M encNum[M]](o *encOps[M], p encPieces[M], s float64, cat func(a, b, c M) M) (M, M) {
	zu := encSym(p.qk.scale(s).add(p.r, -(1 + s)))
	z11 := encSym(p.r.scale(1-s).add(p.qk, s))
	g := p.g.scale(1-s).add(p.rks, s)
	w := encSym(p.w.scale(1-s).add(o.r, s))
	return zu, cat(z11, g.scale(-1), w)
}

// sdir returns −Ric_K(c + D_U) = −(r + A_KᵀD_UA_K − EᵀD_UE) and
// N_K(c − D_L) = [[r − A_KᵀD_LA_K + EᵀD_LE, −(g + BᵀD_LA_K)ᵀ], [·,
// W(c) − BᵀD_LB]].
func encSDir[M encNum[M]](o *encOps[M], p encPieces[M], du, dl M, cat func(a, b, c M) M) (M, M) {
	md := func(d M) M { return p.ak.mul(d.mul(p.ak, false), true).add(o.ete(d), -1) }
	zu := encSym(p.r.add(md(du), 1).scale(-1))
	z11 := encSym(p.r.add(md(dl), -1))
	g := p.g.add(o.b.mul(dl.mul(p.ak, false), true), 1)
	w := encSym(p.w.add(o.b.mul(dl.mul(o.b, false), true), -1))
	return zu, cat(z11, g.scale(-1), w)
}

// encBounds are the entrywise term magnitudes of the pieces at |c|, |k|.
type encBounds struct {
	pr, pq, pg, prks, pw, ah *mat.Dense
	r, qk, g, rks, w         *mat.Dense
}

func (d *encData) bounds(c, k encDD, p encPieces[encDD]) encBounds {
	var b encBounds
	n, m := d.n, d.m
	ak := k.abs()
	u2 := math.Ldexp(1, -106)
	ahat := p.ak.abs()
	bk0 := d.ws.mat(n, n)
	bk0.Mul(d.aB, ak)
	bk0.Add(bk0, d.aA)
	bk0.Scale(8*float64(m+1)*u2, bk0)
	ahat.Add(ahat, bk0)
	ahat.Scale(1+1e-10, ahat)
	b.pr, b.pg, b.pw = d.psi(c.abs(), ak, ahat, u2)
	b.pq = d.ws.mat(n, n)
	b.pq.Copy(d.aQ)
	b.pq.Add(b.pq, d.mul3T(ak, d.aR, ak))
	b.prks = d.ws.mat(m, n)
	b.prks.Mul(d.aR, ak)
	if d.aS != nil {
		sk := d.ws.mat(n, n)
		sk.Mul(d.aS, ak)
		b.pq.Add(b.pq, sk)
		b.pq.Add(b.pq, d.transposed(sk))
		b.prks.Add(b.prks, d.aSt)
	}
	b.pq.Scale(1+1e-10, b.pq)
	b.prks.Scale(1+1e-10, b.prks)
	b.ah = d.ws.mat(n, n)
	b.ah.Copy(d.aA)
	bk := d.ws.mat(n, n)
	bk.Mul(d.aB, ak)
	b.ah.Add(b.ah, bk)
	b.ah.Scale(1+1e-10, b.ah)
	b.r, b.qk, b.g, b.rks, b.w = d.absOf(p.r.round()), d.absOf(p.qk.round()), d.absOf(p.g.round()), d.absOf(p.rks.round()), d.absOf(p.w.round())
	return b
}

// lin is Σ|f_i|·m_i, each term scaled, then summed, in order.
func (d *encData) lin(f [4]float64, ms [4]*mat.Dense) *mat.Dense {
	r, c := ms[0].Dims()
	o := d.ws.mat(r, c)
	t := d.ws.mat(r, c)
	for i, m := range ms {
		if m == nil {
			break
		}
		t.Scale(math.Abs(f[i]), m)
		o.Add(o, t)
	}
	return o
}

// xdirBounds returns the bounds of encXDir in an arithmetic of unit v:
// γ times the combined term magnitudes plus 8v of the combined values for
// the combination's own two operations.
func (d *encData) xdirBounds(b encBounds, s, v float64) (*mat.Dense, *mat.Dense) {
	g := d.gam(v)
	f := 8 * v
	hu := d.lin([4]float64{g * (1 + s), g * s, f * (1 + s), f * s}, [4]*mat.Dense{b.pr, b.pq, b.r, b.qk})
	h11 := d.lin([4]float64{g * (1 - s), g * s, f * (1 - s), f * s}, [4]*mat.Dense{b.pr, b.pq, b.r, b.qk})
	hg := d.lin([4]float64{g * (1 - s), g * s, f * (1 - s), f * s}, [4]*mat.Dense{b.pg, b.prks, b.g, b.rks})
	hw := d.lin([4]float64{g, f, f * s}, [4]*mat.Dense{b.pw, b.w, d.aR})
	return hu, catBound(d.ws, h11, hg, hw)
}

// sdirBounds is xdirBounds for encSDir with radii |D_U|, |D_L|.
func (d *encData) sdirBounds(b encBounds, du, dl *mat.Dense, v float64) (*mat.Dense, *mat.Dense) {
	g := d.gam(v)
	f := 8 * v
	mpsi := func(x *mat.Dense) *mat.Dense {
		o := d.mul3T(b.ah, x, b.ah)
		if d.dd.eIdent {
			o.Add(o, x)
		} else {
			o.Add(o, d.mul3T(d.aE, x, d.aE))
		}
		return o
	}
	pu, pl := mpsi(du), mpsi(dl)
	var bda mat.Dense
	bda.Mul(d.aB.T(), dl)
	bdaA := mat.NewDense(d.m, d.n, nil)
	bdaA.Mul(&bda, b.ah)
	bdb := d.mul3T(d.aB, dl, d.aB)
	hu := d.lin([4]float64{g, g, f, f}, [4]*mat.Dense{b.pr, pu, b.r, pu})
	h11 := d.lin([4]float64{g, g, f, f}, [4]*mat.Dense{b.pr, pl, b.r, pl})
	hg := d.lin([4]float64{g, g, f, f}, [4]*mat.Dense{b.pg, bdaA, b.g, bdaA})
	hw := d.lin([4]float64{g, g, f, f}, [4]*mat.Dense{b.pw, bdb, b.w, bdb})
	return hu, catBound(d.ws, h11, hg, hw)
}

// encCentre is a centre with its gain and pieces in both arithmetics
// (encPrec bits on first use).
type encCentre struct {
	// zeroLo: the centre is a float64 X itself; k1, cPD its budget scale
	// and definiteness, on first use.
	zeroLo bool
	k1     float64
	cPD    bool
	d      *encData
	c, k   encDD
	p      encPieces[encDD]
	b      encBounds
	pb     *encPieces[encBig]
	cb, kb encBig
}

func (d *encData) centre(c, k encDD) *encCentre {
	e := d.ws.centre()
	e.d, e.c, e.k = d, c, k
	e.p = encPiecesOf(&d.dd, c, k)
	e.b = d.bounds(c, k, e.p)
	return e
}

func (e *encCentre) big() *encPieces[encBig] {
	if e.pb == nil {
		bg := e.d.big()
		e.cb, e.kb = encBigDD(e.d.ar, e.c), encBigDD(e.d.ar, e.k)
		p := encPiecesOf(bg, e.cb, e.kb)
		e.pb = &p
	}
	return e.pb
}

// testX proves −Ric_K(c(1 + s)) ⪰ 0 and N_K(c(1 − s)) ≻ 0 in the tiers
// allowed, double-double first; in encPrec bits only the one double-
// double did not prove.
func (e *encCentre) testX(s float64, tiers encTiers) encTier {
	uOK, lOK := false, false
	if tiers&encDDOnly != 0 {
		zu, zl := encXDir(&e.d.dd, e.p, s, catDD)
		hu, hl := e.d.xdirBounds(e.b, s, math.Ldexp(1, -106))
		uOK = encPDDD(zu, hu)
		lOK = uOK && encPDDD(zl, hl)
		if lOK {
			return encTierDD
		}
		if !uOK {
			lOK = encPDDD(zl, hl)
		}
	}
	if tiers&encBigOnly == 0 {
		return encNone
	}
	bu, bl := encXDir(e.d.big(), *e.big(), s, catBig)
	hu, hl := e.d.xdirBounds(e.b, s, math.Ldexp(1, -encPrec))
	if (uOK || encPDBig(bu, hu)) && (lOK || encPDBig(bl, hl)) {
		return encTierBig
	}
	return encNone
}

// testS proves −Ric_K(c + D_U) ⪰ 0 and N_K(c − D_L) ≻ 0.
func (e *encCentre) testS(du, dl encDD) encTier {
	u2 := math.Ldexp(1, -106)
	zu, zl := encSDir(&e.d.dd, e.p, du, dl, catDD)
	adu, adl := du.abs(), dl.abs()
	hu, hl := e.d.sdirBounds(e.b, adu, adl, u2)
	if encPDDD(zu, hu) && encPDDD(zl, hl) {
		return encTierDD
	}
	bu, bl := encSDir(e.d.big(), *e.big(), encBigDD(e.d.ar, du), encBigDD(e.d.ar, dl), catBig)
	hu, hl = e.d.sdirBounds(e.b, adu, adl, math.Ldexp(1, -encPrec))
	if encPDBig(bu, hu) && encPDBig(bl, hl) {
		return encTierBig
	}
	return encNone
}

// wPD proves W(c) ≻ 0.
func (e *encCentre) wPD() bool {
	h := e.d.ws.mat(e.d.m, e.d.m)
	h.Scale(e.d.gam(math.Ldexp(1, -106)), e.b.pw)
	if encPDDD(e.p.w, h) {
		return true
	}
	h.Scale(e.d.gam(math.Ldexp(1, -encPrec)), e.b.pw)
	return encPDBig(e.big().w, h)
}

// exactPD proves the double-double x positive definite (its value exact).
func (d *encData) exactPD(x encDD) bool {
	if encPDDD(x, matOf(x.ws, x.r, x.c)) {
		return true
	}
	d.big()
	return encPDBig(encBigDD(d.ar, x), mat.NewDense(x.r, x.c, nil))
}

// gain returns the gain (R + BᵀXB)⁻¹(BᵀXA + Sᵀ) of the centre x in
// double-double: a float64 solve refined on its double-double residual
// while the refinement converges, else solved in encPrec bits. Any K
// serves the tests, but a K off the gain of x enters Ric_K(x) as
// (K − K_x)ᵀW(K − K_x): under cheap control with rank-deficient B, where
// W is singular to float64, the refined K moved the next centre 6.5e-7
// from an X 1.4e-13 off.
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
	if solvedLU(&k0, lu.SolveTo(&k0, false, gg.round())) {
		k := encDDOf(&k0)
		for range 3 {
			res := gg.add(w.mul(k, false), -1)
			var dk mat.Dense
			if !solvedLU(&dk, lu.SolveTo(&dk, false, res.round())) {
				break
			}
			k = k.add(encDDOf(&dk), 1)
			kn := mat.Norm(k.round(), 1)
			if !(kn > 0) || math.IsInf(kn, 0) {
				break
			}
			if mat.Norm(&dk, 1) <= 1e-30*kn {
				return k, true
			}
		}
	}
	bg := d.big()
	bx := encBigDD(d.ar, x)
	bgg := bg.b.mul(bx.mul(bg.a, false), true)
	if bg.hasS {
		bgg = bgg.add(bg.s.t(), 1)
	}
	kb, ok := bg.w(bx).solve(bgg)
	if !ok {
		return encDD{}, false
	}
	k := kb.dd()
	if !finiteSlice(k.round().RawMatrix().Data) {
		return encDD{}, false
	}
	return k, true
}

// solvedLU reports an LU solve completed: an ill-conditioned W still
// gives a usable gain, which the tests judge, not its condition.
func solvedLU(dst *mat.Dense, err error) bool {
	_, cond := err.(mat.Condition)
	return (err == nil || cond) && !dst.IsEmpty() && finiteSlice(dst.RawMatrix().Data)
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
		f, ok := eb.solve(bg.acl(encBigDD(d.ar, k)))
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
	t1, ok := et.solve(encBigOf(d.ar, Y))
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

// enclose tries the X-direction radius D_U = D_L = s·X_c, s from the
// newtonTol budget m(X − X_c) + s·κ(X_c) (for X, else for fl(X_c)), then
// the Stein direction, D_U and D_L Stein solutions of majorants of
// Ric_K(X_c) and of g_KᵀW⁻¹g_K − Ric_K(X_c) (×1, 8, 64, 512), within
// m + b(D_U + D_L). −D ⪯ M ⪯ D bounds the measure of M by b(D) =
// max(‖D‖_F, max_j √D_jj Σ_i √D_ii) relative to X_c.
func (e *encCentre) enclose(x *mat.Dense) (encVerdict, bool) {
	if v, ok := e.encloseX(x, encAllTiers); ok {
		return v, true
	}
	d, n := e.d, e.d.n
	cf := e.c.round()
	m, mr := e.budget(x, cf)
	r := e.p.r.round()
	gc := e.p.g.round()
	var lu mat.LU
	lu.Factorize(e.p.w.round())
	var wg mat.Dense
	if !solvedLU(&wg, lu.SolveTo(&wg, false, gc)) {
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
		du, ok1 := d.stein(e.k, nu)
		dl, ok2 := d.stein(e.k, nl)
		if !ok1 || !ok2 {
			continue
		}
		b := encRadius(du.add(dl, 1).round(), cf)
		in, inR := m+b <= newtonTol, mr+b <= newtonTol
		if !in && !inR || !d.exactPD(du) || !d.exactPD(dl) {
			continue
		}
		if t := e.testS(du, dl); t != encNone {
			return encVerdict{tier: t, ret: in}, true
		}
	}
	return encVerdict{}, false
}

// budget returns the measures of X − X_c and fl(X_c) − X_c.
func (e *encCentre) budget(x, cf *mat.Dense) (m, mr float64) {
	n := e.d.n
	diff := e.d.ws.mat(n, n)
	diff.Sub(x, cf)
	m = encMeasure(diff, cf)
	if e.zeroLo {
		return m, m
	}
	return m, encMeasure(mat.NewDense(n, n, append([]float64(nil), e.c.l...)), cf)
}

// encloseX is enclose's X-direction radius alone, in the tiers allowed.
func (e *encCentre) encloseX(x *mat.Dense, tiers encTiers) (encVerdict, bool) {
	cf := e.c.round()
	m, mr := e.budget(x, cf)
	if e.k1 == 0 {
		e.k1 = encRadius(cf, cf)
		e.cPD = e.k1 > 0 && !math.IsInf(e.k1, 0) && e.d.exactPD(e.c)
	}
	if !e.cPD {
		return encVerdict{}, false
	}
	for i, mm := range []float64{m, mr} {
		if !(mm < newtonTol) || i == 1 && e.zeroLo {
			continue
		}
		s := 0.999 * (newtonTol - mm) / e.k1
		if t := e.testX(s, tiers); t != encNone {
			return encVerdict{tier: t, ret: m+s*e.k1 <= newtonTol}, true
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
// (A, B, Q, R, S, E), its gain K, cheapest first: the X-direction radius
// at X itself with K, in double-double; then at the centre X ⊕ Δ of the
// Newton step Δ verifyRiccati certified X with (dx, nil without), with its
// double-double gain; at both in encPrec bits; then from that centre (or
// X + Δ, Δ the float64 Stein step of Ric_K(X) for the gain of X) the
// enclosure at up to encMaxRefine further centres, each one Stein step on
// with a fresh gain. While W(X_c) is not proved positive definite the
// step takes the returned K, and W-not-PD fails only at the last centre.
// corrected is fl(X_c) and its gain where X_c, not X, is within budget.
func encloseDare(A, B, Q, R, S, E, X, K, dx *mat.Dense) (out encResult, xc, kc *mat.Dense, tier encTier) {
	d := newEncData(A, B, Q, R, S, E)
	defer d.release()
	c := encDDIn(d.ws, X)
	symmetrize(c.h, d.n, d.n)
	kret := encDDIn(d.ws, K)
	e0 := d.centre(c, kret)
	e0.zeroLo = true
	if v, ok := e0.encloseX(X, encDDOnly); ok && v.ret {
		return encCertified, nil, nil, v.tier
	}
	var e1 *encCentre
	if dx != nil {
		c1 := encSym(c.add(encDDIn(d.ws, dx), 1))
		if k1, ok := d.gain(c1); ok {
			e1 = d.centre(c1, k1)
			if v, ok := e1.encloseX(X, encDDOnly); ok {
				return e1.outcome(v)
			}
		}
	}
	if v, ok := e0.encloseX(X, encBigOnly); ok && v.ret {
		return encCertified, nil, nil, v.tier
	}
	if e1 != nil {
		if v, ok := e1.encloseX(X, encBigOnly); ok {
			return e1.outcome(v)
		}
		c = e1.c
	}
	start := 0
	if e1 != nil {
		start = 1
	}
	for it := start; it <= encMaxRefine+1; it++ {
		k, ok := d.gain(c)
		var e *encCentre
		if ok {
			e = d.centre(c, k)
			ok = e.wPD()
		}
		if !ok {
			if it == encMaxRefine+1 {
				return encFailed, nil, nil, encNone
			}
			k = kret
			e = d.centre(c, k)
		} else if it > 0 {
			if v, ok := e.enclose(X); ok {
				return e.outcome(v)
			}
		}
		if it == encMaxRefine+1 {
			break
		}
		step, ok := d.stein(k, e.p.r.round())
		if !ok {
			break
		}
		c = encSym(c.add(step, 1))
	}
	return encFailed, nil, nil, encNone
}

func (e *encCentre) outcome(v encVerdict) (encResult, *mat.Dense, *mat.Dense, encTier) {
	if v.ret {
		return encCertified, nil, nil, v.tier
	}
	return encCorrected, mat.DenseCopyOf(e.c.round()), mat.DenseCopyOf(e.k.round()), v.tier
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
// gain once the gain gates certify K with X unchanged (their refinement
// replaced a certified fl(X_c) by a float64 Newton iterate 1.6e-6 to
// 3.9e-6 off, zrwreg 20196, 12124), else the given X and K go to tier 2;
// so does a result carrying a certified EᵀXE (xe), which the X enclosure
// does not bound: Dlqr formed EᵀXE from the corrected X and E carried X's
// 4.3e-9 into a Y 1.78e-8 off (rv49 3641). Without an
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
	var dx *mat.Dense
	if res.newtonDelta != nil && mat.Equal(res.X, res.newtonX) {
		dx = res.newtonDelta
	}
	out, xc, kc, _ := encloseDare(A, B, Q, R, S, E, res.X, res.K, dx)
	switch out {
	case encCertified:
		return res
	case encCorrected:
		if res.xe != nil {
			return dareTier2(A, B, Q, R, opts, res)
		}
		c := *res
		symmetrize(xc.RawMatrix().Data, n, n)
		c.X, c.K = xc, kc
		c.step, c.x1, c.xe, c.xeStep, c.xeX, c.xeLoose = 0, nil, nil, 0, 0, false
		c.Eig = make([]complex128, n)
		if !closedLoopStable(A, B, c.K, E, c.Eig) {
			return dareTier2(A, B, Q, R, opts, res)
		}
		red, _ := reduceInputs(A, B, Q, R, opts)
		if r, _, ok := gainOfX(A, B, Q, R, opts, red, &c); ok && r.accuracy == 0 && mat.Equal(r.X, c.X) {
			return r
		}
		return dareTier2(A, B, Q, R, opts, res)
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
