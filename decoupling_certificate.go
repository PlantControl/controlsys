package controlsys

import (
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// decouplingSepTol is MATLAB's stabsep/freqsep SepTol default: G1 + G2 may
// differ from G by at most SepTol times the rounding accuracy of G(s).
const decouplingSepTol = 10

// decouplingCertificate decides whether the Sylvester decoupling of an
// ordered real Schur form, T11·X − X·T22 = −T12, gives parts with
// G1 + G2 within decouplingSepTol of the normwise rounding accuracy
// U·ε·(‖C‖‖M‖‖A‖‖M‖‖B‖ + ‖C‖‖M‖‖B‖), M = (sI − A)⁻¹, U = 4(n+5), of the
// model given to Dgees, at every s. The computed parts are exactly the
// decoupled parts of (T − [0 R; 0 0], B̃ + [δb; 0], C̃ + [0 δc]) with
// R = T11·X − X·T22 + T12 and δb, δc the rounding of B̃1 − X·B̃2 and
// C̃2 + C̃1·X, and the diagonal blocks of (sI − T)⁻¹ are bounded by the
// whole inverse, so (Frobenius norms, budget β·U·ε of the SepTol·U·ε bar
// left after the Schur form and the evaluation):
//
//	P1: ‖C̃1‖‖B̃2‖(‖R̂‖ + ½(n+1)ε((‖T11‖+‖T22‖)‖X‖ + ‖T12‖)) ≤ β·U·ε·‖A‖‖C‖‖B‖
//	P2: ½[(n2+1)‖C̃1‖(‖B̃1‖ + ‖X‖‖B̃2‖) + (n1+1)(‖C̃2‖ + ‖C̃1‖‖X‖)‖B̃2‖] ≤ β·U·‖C‖‖B‖
//
// R̂ is the computed residual; B̃1 and C̃2 are taken before the update.
// The coupling X matters only through outputs that see group 1 (C̃1) and
// inputs that drive group 2 (B̃2): a large X with negligible ‖C̃1‖‖B̃2‖
// is accepted.
type decouplingCertificate struct {
	n, n1, n2           int
	a, b, c             float64
	t11, t22, t12, x, r float64
	b1, b2, c1, c2      float64
}

// decouplingBudget is β: the share of the SepTol·U·ε bar the decoupling may
// use.
const decouplingBudget = decouplingSepTol - 1

func newDecouplingCertificate(sys *System, t, x, bt, ct []float64, n1 int) decouplingCertificate {
	n, m, p := sys.Dims()
	n2 := n - n1
	aRaw, bRaw, cRaw := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	d := decouplingCertificate{
		n: n, n1: n1, n2: n2,
		a:   froStrided(aRaw.Data, aRaw.Stride, n, n),
		b:   froStrided(bRaw.Data, bRaw.Stride, n, m),
		c:   froStrided(cRaw.Data, cRaw.Stride, p, n),
		t11: froStrided(t, n, n1, n1),
		t22: froStrided(t[n1*n+n1:], n, n2, n2),
		t12: froStrided(t[n1:], n, n1, n2),
		x:   froStrided(x, n2, n1, n2),
		b1:  froStrided(bt, m, n1, m),
		b2:  froStrided(bt[n1*m:], m, n2, m),
	}
	if p > 0 {
		d.c1, d.c2 = froStrided(ct, n, p, n1), froStrided(ct[n1:], n, p, n2)
	}
	res := make([]float64, n1*n2)
	copyStrided(res, n2, t[n1:], n, n1, n2)
	rg := blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: res}
	xg := blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: x}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n1, Cols: n1, Stride: n, Data: t}, xg, 1, rg)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, xg, blas64.General{Rows: n2, Cols: n2, Stride: n, Data: t[n1*n+n1:]}, 1, rg)
	d.r = froStrided(res, n2, n1, n2)
	return d
}

// accept reports P1 and P2 for the updated blocks b1 = B̃1 − X·B̃2 and
// c2 = C̃2 + C̃1·X, which must be finite like X and R̂. Ratios to ‖B‖ and
// ‖C‖ keep the products in range; a model with B = 0 or C = 0 has G ≡ D
// and is accepted.
func (d decouplingCertificate) accept(b1, c2 []float64) bool {
	if !isFinite(d.x) || !isFinite(d.r) || !allFinite(b1) || !allFinite(c2) {
		return false
	}
	if d.b == 0 || d.c == 0 {
		return true
	}
	e := eps()
	u := 4 * float64(d.n+5)
	c1, c2n, b1n, b2 := d.c1/d.c, d.c2/d.c, d.b1/d.b, d.b2/d.b
	p1 := c1 * b2 * (d.r + 0.5*float64(d.n+1)*e*((d.t11+d.t22)*d.x+d.t12))
	if !(p1 <= decouplingBudget*u*e*d.a) {
		return false
	}
	p2 := 0.5 * (float64(d.n2+1)*c1*(b1n+d.x*b2) + float64(d.n1+1)*(c2n+c1*d.x)*b2)
	return p2 <= decouplingBudget*u
}

// froStrided is the Frobenius norm of a strided block: a plain sum of
// squares, redone with the scaled l2 accumulator when it overflows or may
// have lost entries to underflow.
func froStrided(data []float64, stride, rows, cols int) float64 {
	if rows == 0 || cols == 0 {
		return 0
	}
	var sum float64
	for i := range rows {
		for _, v := range data[i*stride : i*stride+cols] {
			sum += v * v
		}
	}
	if sum > 0x1p-900 && sum < 0x1p900 {
		return math.Sqrt(sum)
	}
	return froStridedScaled(data, stride, rows, cols)
}

func froStridedScaled(data []float64, stride, rows, cols int) float64 {
	var s l2
	for i := range rows {
		for _, v := range data[i*stride : i*stride+cols] {
			s.add(v)
		}
	}
	return s.norm()
}

// descriptorBar holds the stored-data Frobenius norms of the normwise
// descriptor accuracy bar u·(‖C‖‖M‖(‖A‖ + |s|‖E‖)‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + ‖D‖),
// M = (sE − A)⁻¹, u = U·ε, U = 4(n+5).
type descriptorBar struct {
	n          int
	a, e, b, c float64
}

func newDescriptorBar(sys *System) descriptorBar {
	n, m, p := sys.Dims()
	aRaw, eRaw, bRaw, cRaw := sys.A.RawMatrix(), sys.E.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	return descriptorBar{
		n: n,
		a: froStrided(aRaw.Data, aRaw.Stride, n, n),
		e: froStrided(eRaw.Data, eRaw.Stride, n, n),
		b: froStrided(bRaw.Data, bRaw.Stride, n, m),
		c: froStrided(cRaw.Data, cRaw.Stride, p, n),
	}
}

func (d descriptorBar) u() float64 { return 4 * float64(d.n+5) }

// trivial reports G ≡ D, which every split reproduces exactly.
func (d descriptorBar) trivial() bool { return d.b == 0 || d.c == 0 }

// descriptorBudget accumulates the left sides of the descriptor decoupling
// tests as fractions of their budgets β·U·ε·‖A‖ (zS), β·U·ε·‖E‖ (zT) and
// β·U (z2, the ‖C‖‖M‖‖B‖ term), summed over every split that builds the
// parts; they are certified iff each is at most 1 and every computed block
// is finite. β = decouplingBudget; the rest of the SepTol bar covers the
// orthogonality of Q and Z, QᵀB, CZ and the evaluation.
type descriptorBudget struct {
	zS, zT, z2 float64
	failed     bool
}

func (d *descriptorBudget) accepted() bool {
	return !d.failed && d.zS <= 1 && d.zT <= 1 && d.z2 <= 1
}

func (d *descriptorBudget) require(ok bool) {
	if !ok {
		d.failed = true
	}
}

// gemmIf is blas64.Gemm that tolerates empty operands.
func gemmIf(tA, tB blas.Transpose, alpha float64, a, b blas64.General, beta float64, c blas64.General) {
	if c.Rows == 0 || c.Cols == 0 {
		return
	}
	k := a.Cols
	if tA == blas.Trans {
		k = a.Rows
	}
	if k == 0 {
		for i := range c.Rows {
			row := c.Data[i*c.Stride : i*c.Stride+c.Cols]
			for j := range row {
				row[j] *= beta
			}
		}
		return
	}
	blas64.Gemm(tA, tB, alpha, a, b, beta, c)
}

func gen(rows, cols, stride int, data []float64) blas64.General {
	return blas64.General{Rows: rows, Cols: cols, Stride: max(stride, 1), Data: data}
}

func tail(d []float64, off int) []float64 {
	if off >= len(d) {
		return nil
	}
	return d[off:]
}

// absProductFro is ‖|X|·|Y| + |W|‖_F for X r×k, Y k×c and W r×c (W may be
// nil): the componentwise bound on the rounding of X·Y − W.
func absProductFro(x []float64, ldx int, y []float64, ldy int, w []float64, ldw, r, k, c int) float64 {
	if r == 0 || c == 0 {
		return 0
	}
	out := make([]float64, r*c)
	for i := range r {
		row := out[i*c : (i+1)*c]
		if w != nil {
			for j := range row {
				row[j] = math.Abs(w[i*ldw+j])
			}
		}
		for l := range k {
			v := math.Abs(x[i*ldx+l])
			if v == 0 {
				continue
			}
			yr := y[l*ldy : l*ldy+c]
			for j := range row {
				row[j] += v * math.Abs(yr[j])
			}
		}
	}
	return froStrided(out, c, r, c)
}

func absMatrix(x []float64, ld, r, c int) []float64 {
	out := make([]float64, r*c)
	for i := range r {
		for j := range c {
			out[i*c+j] = math.Abs(x[i*ld+j])
		}
	}
	return out
}

// formResidual is ‖QᵀXZ − F̃‖_F/(U·ε·‖X‖_F), F̃ the n×n form f with its
// (n1:, :n1) block taken as zero (the form the split uses).
func formResidual(x blas64.General, f, q, z []float64, n, n1 int, nx, u float64) float64 {
	if nx == 0 {
		return 0
	}
	tmp := make([]float64, n*n)
	res := make([]float64, n*n)
	gemmIf(blas.Trans, blas.NoTrans, 1, gen(n, n, n, q), x, 0, gen(n, n, n, tmp))
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n, n, n, tmp), gen(n, n, n, z), 0, gen(n, n, n, res))
	for i := range n {
		for j := range n {
			if i >= n1 && j < n1 {
				continue
			}
			res[i*n+j] -= f[i*n+j]
		}
	}
	return froStrided(res, n, n, n) / (u * eps() * nx)
}

// addGeneralizedSplit adds the terms of one split at n1 of the generalized
// Schur form Qᵀ(A, E)Z = (S, T) of the stored model, with (R, L) the scaled
// solution of S11R − LS22 = −S12, T11R − LT22 = −T12 (unused when n1 = n),
// the norms b1, b2, c1, c2 of B̃ = QᵀB and C̃ = CZ before the update, bq and
// cz after it, and the folded G1 blocks Â1 = T11⁻¹S11, B̂1f = T11⁻¹B̂1:
//
//	P1S: ĉ1b̂2(‖R̂S‖ + ½(n+1)ε(‖S11‖‖R‖ + ‖L‖‖S22‖ + ‖S12‖)) + ĉ1(‖B̂1‖/‖B‖)(‖F̂A‖ + ρA) + fa·U·ε‖A‖ ≤ β·U·ε‖A‖
//	P1T: ĉ1b̂2(‖R̂T‖ + ½(n+1)ε(‖T11‖‖R‖ + ‖L‖‖T22‖ + ‖T12‖)) + fe·U·ε‖E‖ ≤ β·U·ε‖E‖
//	P2:  ½(n2+1)ĉ1(b̂1 + ‖L‖b̂2) + ½(n1+1)(ĉ2 + ĉ1‖R‖)b̂2 + ĉ1(‖F̂B‖ + ρB)/(ε‖B‖) ≤ β·U
//
// R̂S, R̂T are the computed Sylvester residuals; F̂A = T11Â1 − S11 and
// F̂B = T11B̂1f − B̂1 the fold residuals, with componentwise rounding
// ρA = ½(n1+1)ε‖|T11||Â1| + |S11|‖ and ρB likewise; fa, fe the measured
// form residuals ‖QᵀAZ − S̃‖/(U·ε‖A‖), ‖QᵀEZ − T̃‖/(U·ε‖E‖), S̃ and T̃ the
// form split, with zero (2,1) blocks and T after any β edit. The computed
// parts are exactly those of the pencil perturbed in its (1,2) block by
// R̂S − s·R̂T and in B̃1 and C̃2 by the update rounding, and to first order
// in its (1,1) block by the fold; the diagonal blocks of (sT − S)⁻¹ are
// bounded by ‖M‖.
func (d *descriptorBudget) addGeneralizedSplit(bar descriptorBar, sys *System, s, t, q, z []float64, n1 int,
	r, l []float64, b1, b2, c1, c2 float64, bq, cz []float64, a1, b1f *mat.Dense) {
	n, m, _ := sys.Dims()
	n2 := n - n1
	ep := eps()
	u := bar.u()
	beta := float64(decouplingBudget)
	aRaw, eRaw := sys.A.RawMatrix(), sys.E.RawMatrix()
	fa := formResidual(gen(n, n, aRaw.Stride, aRaw.Data), s, q, z, n, n1, bar.a, u)
	fe := formResidual(gen(n, n, eRaw.Stride, eRaw.Data), t, q, z, n, n1, bar.e, u)
	d.require(isFinite(fa) && isFinite(fe))
	if bar.trivial() {
		return
	}
	b1, b2, c1, c2 = b1/bar.b, b2/bar.b, c1/bar.c, c2/bar.c
	s11, t11 := froStrided(s, n, n1, n1), froStrided(t, n, n1, n1)
	if n2 > 0 {
		s22, t22 := froStrided(s[n1*n+n1:], n, n2, n2), froStrided(t[n1*n+n1:], n, n2, n2)
		s12, t12 := froStrided(s[n1:], n, n1, n2), froStrided(t[n1:], n, n1, n2)
		nr, nl := froStrided(r, n2, n1, n2), froStrided(l, n2, n1, n2)
		resid := func(f []float64) float64 {
			res := make([]float64, n1*n2)
			copyStrided(res, n2, f[n1:], n, n1, n2)
			gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n1, n1, n, f), gen(n1, n2, n2, r), 1, gen(n1, n2, n2, res))
			gemmIf(blas.NoTrans, blas.NoTrans, -1, gen(n1, n2, n2, l), gen(n2, n2, n, f[n1*n+n1:]), 1, gen(n1, n2, n2, res))
			return froStrided(res, n2, n1, n2)
		}
		rs, rt := resid(s), resid(t)
		half := 0.5 * float64(n+1) * ep
		d.zS += c1 * b2 * (rs + half*(s11*nr+nl*s22+s12)) / (beta * u * ep * bar.a)
		zt := c1 * b2 * (rt + half*(t11*nr+nl*t22+t12))
		if bar.e > 0 {
			d.zT += zt / (beta * u * ep * bar.e)
		} else {
			d.require(zt == 0)
		}
		d.z2 += (0.5*float64(n2+1)*c1*(b1+nl*b2) + 0.5*float64(n1+1)*(c2+c1*nr)*b2) / (beta * u)
		d.require(isFinite(nr) && isFinite(nl) && isFinite(rs) && isFinite(rt))
	}
	d.require(allFinite(bq) && allFinite(cz))
	half1 := 0.5 * float64(n1+1) * ep
	a1r := a1.RawMatrix()
	fA := make([]float64, n1*n1)
	copyStrided(fA, n1, s, n, n1, n1)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n1, n1, n, t), gen(n1, n1, a1r.Stride, a1r.Data), -1, gen(n1, n1, n1, fA))
	faFold := froStrided(fA, n1, n1, n1) + half1*absProductFro(t, n, a1r.Data, a1r.Stride, s, n, n1, n1, n1)
	b1hat := froStrided(bq, m, n1, m)
	d.zS += c1*(b1hat/bar.b)*faFold/(beta*u*ep*bar.a) + fa/beta
	d.zT += fe / beta
	d.require(allFinite(a1r.Data) && isFinite(faFold) && isFinite(b1hat))
	if m > 0 {
		b1r := b1f.RawMatrix()
		fB := make([]float64, n1*m)
		copyStrided(fB, m, bq, m, n1, m)
		gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n1, n1, n, t), gen(n1, m, b1r.Stride, b1r.Data), -1, gen(n1, m, m, fB))
		fbFold := froStrided(fB, m, n1, m) + half1*absProductFro(t, n, b1r.Data, b1r.Stride, bq, m, n1, n1, m)
		d.z2 += c1 * fbFold / (ep * bar.b) / (beta * u)
		d.require(allFinite(b1r.Data) && isFinite(fbFold))
	}
}

// addStiffInner adds the terms of splitStiffFullRank's inner explicit
// split of rest = (Â1, B̂1f, C̃1), the k-state G1 of its outer split with
// E block T11 = t[:k,:k] (stride ldt), against the outer bar. rest's
// resolvent is (sT11 − S11)⁻¹T11, so the inner terms of
// decouplingCertificate gain κ1 = √(‖T11‖₁‖T11‖∞) ≥ ‖T11‖₂ on the C̃1r side
// and w2 = ‖T11·Zr2·B̃2r‖/‖B‖ on the B side. The inner Schur form
// Â1 = Zr·Ts·Zrᵀ, B̃r = Zrᵀ·B̂1f and C̃r = C̃1·Zr add the measured terms
// F_r = T11(Â1 − Zr·Ts·Zrᵀ), weighted ĉ1·w1 with w1 = ‖T11·B̂1f‖/‖B‖, and
// the B and C transform residuals, each with componentwise (k+1)ε
// rounding. tr, zr are the ordered inner Schur form, x its decoupling, and
// bt, ct the inner B̃r, C̃r before the update.
func (d *descriptorBudget) addStiffInner(bar descriptorBar, t []float64, ldt, k int, rest *System, tr, zr []float64, n1 int, x, bt, ct []float64) {
	if bar.trivial() || n1 == 0 || n1 == k {
		return
	}
	_, m, p := rest.Dims()
	n2 := k - n1
	ep := eps()
	u := bar.u()
	beta := float64(decouplingBudget)
	var norm1, normInf float64
	for j := range k {
		var cs, rsum float64
		for i := range k {
			cs += math.Abs(t[i*ldt+j])
			rsum += math.Abs(t[j*ldt+i])
		}
		norm1, normInf = math.Max(norm1, cs), math.Max(normInf, rsum)
	}
	kap := math.Sqrt(norm1 * normInf)
	w := make([]float64, k*m)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, n2, k, zr[n1:]), gen(n2, m, m, tail(bt, n1*m)), 0, gen(k, m, m, w))
	tw := make([]float64, k*m)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, k, ldt, t), gen(k, m, m, w), 0, gen(k, m, m, tw))
	w2 := froStrided(tw, m, k, m) / bar.b
	t11, t22, t12 := froStrided(tr, k, n1, n1), froStrided(tr[n1*k+n1:], k, n2, n2), froStrided(tr[n1:], k, n1, n2)
	nx := froStrided(x, n2, n1, n2)
	res := make([]float64, n1*n2)
	copyStrided(res, n2, tr[n1:], k, n1, n2)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n1, n1, k, tr), gen(n1, n2, n2, x), 1, gen(n1, n2, n2, res))
	gemmIf(blas.NoTrans, blas.NoTrans, -1, gen(n1, n2, n2, x), gen(n2, n2, k, tr[n1*k+n1:]), 1, gen(n1, n2, n2, res))
	rr := froStrided(res, n2, n1, n2)
	c1r := froStrided(ct, k, p, n1) / bar.c
	c2r := froStrided(tail(ct, n1), k, p, n2) / bar.c
	b1r := froStrided(bt, m, n1, m) / bar.b
	b2r := froStrided(tail(bt, n1*m), m, n2, m) / bar.b
	d.zS += c1r * kap * w2 * (rr + 0.5*float64(k+1)*ep*((t11+t22)*nx+t12)) / (beta * u * ep * bar.a)
	d.z2 += (0.5*float64(n2+1)*c1r*kap*(b1r+nx*b2r) + 0.5*float64(n1+1)*(c2r+c1r*nx)*w2) / (beta * u)
	d.require(isFinite(nx) && isFinite(rr) && isFinite(w2))

	ra, rb, rc := rest.A.RawMatrix(), rest.B.RawMatrix(), rest.C.RawMatrix()
	zt := make([]float64, k*k)
	for i := range k {
		for j := range k {
			zt[i*k+j] = zr[j*k+i]
		}
	}
	zts := make([]float64, k*k)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, k, k, zr), gen(k, k, k, tr), 0, gen(k, k, k, zts))
	diff := make([]float64, k*k)
	copyStrided(diff, k, ra.Data, ra.Stride, k, k)
	gemmIf(blas.NoTrans, blas.NoTrans, -1, gen(k, k, k, zts), gen(k, k, k, zt), 1, gen(k, k, k, diff))
	fr := make([]float64, k*k)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, k, ldt, t), gen(k, k, k, diff), 0, gen(k, k, k, fr))
	absZ, absZt, absT := absMatrix(zr, k, k, k), absMatrix(zt, k, k, k), absMatrix(t, ldt, k, k)
	tmp := make([]float64, k*k)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, k, k, absZ), gen(k, k, k, absMatrix(tr, k, k, k)), 0, gen(k, k, k, tmp))
	absA := make([]float64, k*k)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, k, k, tmp), gen(k, k, k, absZt), 0, gen(k, k, k, absA))
	for i := range k {
		for j := range k {
			absA[i*k+j] += math.Abs(ra.Data[i*ra.Stride+j])
		}
	}
	rhoR := float64(k+1) * ep * absProductFro(absT, k, absA, k, nil, 0, k, k, k)
	c1 := froStrided(rc.Data, rc.Stride, p, k) / bar.c
	var w1, fB, fC float64
	if m > 0 {
		tb := make([]float64, k*m)
		gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, k, ldt, t), gen(k, m, rb.Stride, rb.Data), 0, gen(k, m, m, tb))
		w1 = froStrided(tb, m, k, m) / bar.b
		db := make([]float64, k*m)
		copyStrided(db, m, rb.Data, rb.Stride, k, m)
		gemmIf(blas.NoTrans, blas.NoTrans, -1, gen(k, k, k, zr), gen(k, m, m, bt), 1, gen(k, m, m, db))
		gb := make([]float64, k*m)
		gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, k, ldt, t), gen(k, m, m, db), 0, gen(k, m, m, gb))
		absB := make([]float64, k*m)
		gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(k, k, k, absZ), gen(k, m, m, absMatrix(bt, m, k, m)), 0, gen(k, m, m, absB))
		for i := range k {
			for j := range m {
				absB[i*m+j] += math.Abs(rb.Data[i*rb.Stride+j])
			}
		}
		fB = froStrided(gb, m, k, m) + float64(k+1)*ep*absProductFro(absT, k, absB, m, nil, 0, k, k, m)
	}
	if p > 0 {
		dc := make([]float64, p*k)
		copyStrided(dc, k, rc.Data, rc.Stride, p, k)
		gemmIf(blas.NoTrans, blas.NoTrans, -1, gen(p, k, k, ct), gen(k, k, k, zt), 1, gen(p, k, k, dc))
		fC = froStrided(dc, k, p, k) + float64(k+1)*ep*absProductFro(ct, k, absZt, k, rc.Data, rc.Stride, p, k, k)
	}
	d.zS += c1 * w1 * (froStrided(fr, k, k, k) + rhoR) / (beta * u * ep * bar.a)
	d.z2 += (c1*fB/(ep*bar.b) + w1*fC/(ep*bar.c)) / (beta * u)
	d.require(isFinite(w1) && isFinite(fB) && isFinite(fC))
}
