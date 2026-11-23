package controlsys

import (
	"math"
	"slices"

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
// parts. β = decouplingBudget; the rest of the SepTol bar covers the
// orthogonality of Q and Z, QᵀB, CZ and the evaluation. zS, zT, z2 hold the
// terms no tier sharpens (form residuals, the stiff inner split); splits
// holds each generalized split's decoupling and fold terms, plain and, on
// demand, ridge-sharpened.
//
// The parts are certified (accepted) iff every computed block is finite and
// ζ″ = max(z1, (z1 + θ·z2)/(1 + θ)) ≤ 1, z1 = max(zS, zT): first with the
// plain terms, then, if that fails, with each split term replaced by its
// ridge bound where smaller. θ ≥ ‖C‖‖M‖‖B‖ / (‖C‖‖M‖(‖A‖ + |s|‖E‖)‖M‖‖B‖)
// at every s (thetaBound), so the error is at most β·ζ″ times the sum of
// the two terms of the bar.
type descriptorBudget struct {
	zS, zT, z2 float64
	theta      float64
	splits     []splitTerms
	failed     bool
}

// splitTerms are one split's sharpenable terms: plain, and sharpen
// computing their ridge bounds (each at most the plain one).
type splitTerms struct {
	zS, zT, z2 float64
	sharpen    func() (zS, zT, z2 float64)
}

func (d *descriptorBudget) accepted() bool {
	return d.zeta(1) <= 1
}

// zeta is the ζ″ of accepted: the plain value when it is at most floor,
// else the least of it and the ridge-sharpened one, so the parts are
// accepted at any limit L ≥ floor iff zeta(floor) ≤ L; +Inf when a block
// failed, NaN when no tier is a number.
func (d *descriptorBudget) zeta(floor float64) float64 {
	if d.failed {
		return math.Inf(1)
	}
	theta := d.theta
	if !(theta > 0) {
		theta = 1
	}
	zS, zT, z2 := d.zS, d.zT, d.z2
	for _, sp := range d.splits {
		zS, zT, z2 = zS+sp.zS, zT+sp.zT, z2+sp.z2
	}
	plain := secondTierZeta(zS, zT, z2, theta)
	if plain <= floor {
		return plain
	}
	zS, zT, z2 = d.zS, d.zT, d.z2
	for _, sp := range d.splits {
		a, b, c := sp.zS, sp.zT, sp.z2
		if sp.sharpen != nil {
			sa, sb, sc := sp.sharpen()
			a, b, c = math.Min(a, sa), math.Min(b, sb), math.Min(c, sc)
		}
		zS, zT, z2 = zS+a, zT+b, z2+c
	}
	ridge := secondTierZeta(zS, zT, z2, theta)
	if math.IsNaN(plain) || ridge < plain {
		return ridge
	}
	return plain
}

// secondTierZeta is ζ″ = max(z1, (z1 + θ·z2)/(1 + θ)), z1 = max(zS, zT);
// NaN when any input is.
func secondTierZeta(zS, zT, z2, theta float64) float64 {
	if math.IsNaN(zS) || math.IsNaN(zT) || math.IsNaN(z2) {
		return math.NaN()
	}
	z1 := math.Max(zS, zT)
	return math.Max(z1, (z1+theta*z2)/(1+theta))
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

// thetaBound is θ = clamp(θ0, 10·U·ε, 1), θ0 the least over columns j and
// rows i of the split form (S̃, T̃) of max(‖S̃e_j‖/‖A‖ + (fa·U + n)ε,
// ‖T̃e_j‖/‖E‖ + (fe·U + n)ε), rows likewise; 1 when ‖A‖ or ‖E‖ is 0.
// σ_min(sE − A) ≤ ‖(sE − A)x‖ for x a column of Z (or a row of Qᵀ), so
// ‖C‖‖M‖‖B‖ ≤ θ·‖C‖‖M‖(‖A‖ + |s|‖E‖)‖M‖‖B‖ at every s; the floor keeps the
// bound to s with u(‖A‖ + |s|‖E‖)‖M(s)‖ ≤ 0.1.
func thetaBound(bar descriptorBar, s, t []float64, n, n1 int, fa, fe float64) float64 {
	if !(bar.a > 0) || !(bar.e > 0) {
		return 1
	}
	ep := eps()
	u := bar.u()
	slack := float64(n) * ep
	theta := 1.0
	for j := range n {
		var cs, ct, rs, rt l2
		for i := range n {
			if !(i >= n1 && j < n1) {
				cs.add(s[i*n+j])
				ct.add(t[i*n+j])
			}
			if !(j >= n1 && i < n1) {
				rs.add(s[j*n+i])
				rt.add(t[j*n+i])
			}
		}
		theta = math.Min(theta, math.Max(cs.norm()/bar.a+fa*u*ep+slack, ct.norm()/bar.e+fe*u*ep+slack))
		theta = math.Min(theta, math.Max(rs.norm()/bar.a+fa*u*ep+slack, rt.norm()/bar.e+fe*u*ep+slack))
	}
	return math.Min(math.Max(theta, 10*u*ep), 1)
}

// addGeneralizedSplit adds the terms of one split at n1 of the generalized
// Schur form Qᵀ(A, E)Z = (S, T) of the stored model, with (R, L) the scaled
// solution of S11R − LS22 = −S12, T11R − LT22 = −T12 (unused when n1 = n),
// B̃ = QᵀB and C̃ = CZ before the update (bq0, cz0) and after it (bq, cz),
// and the folded G1 blocks Â1 = T11⁻¹S11, B̂1f = T11⁻¹B̂1. Plain terms:
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
// in its (1,1) block by the fold; the plain terms bound the diagonal
// blocks of (sT − S)⁻¹ by ‖M‖, the sharpened ones (ridgeBounds) use
// (sT − S)⁻¹ = [M11, R·M22 − M11·L − M11·Res·M22; 0, M22].
func (d *descriptorBudget) addGeneralizedSplit(bar descriptorBar, sys *System, s, t, q, z []float64, n1 int,
	r, l []float64, bq0, cz0, bq, cz []float64, a1, b1f *mat.Dense) {
	n, m, p := sys.Dims()
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
	d.theta = math.Max(d.theta, thetaBound(bar, s, t, n, n1, fa, fe))
	d.zS += fa / beta
	d.zT += fe / beta
	b1 := froStrided(bq0, m, n1, m) / bar.b
	b2 := froStrided(tail(bq0, n1*m), m, n2, m) / bar.b
	c1 := froStrided(cz0, n, p, n1) / bar.c
	c2 := froStrided(tail(cz0, n1), n, p, n2) / bar.c
	var sp splitTerms
	var resS0, resT0, tb0, tc0, foldA0, foldB0 float64
	s11, t11 := froStrided(s, n, n1, n1), froStrided(t, n, n1, n1)
	half := 0.5 * float64(n+1) * ep
	var nr float64
	if n2 > 0 {
		s22, t22 := froStrided(s[n1*n+n1:], n, n2, n2), froStrided(t[n1*n+n1:], n, n2, n2)
		s12, t12 := froStrided(s[n1:], n, n1, n2), froStrided(t[n1:], n, n1, n2)
		var nl float64
		nr, nl = froStrided(r, n2, n1, n2), froStrided(l, n2, n1, n2)
		rs, rt := froStrided(sylvesterResidual(s, n, n1, r, l), n2, n1, n2), froStrided(sylvesterResidual(t, n, n1, r, l), n2, n1, n2)
		resS0 = c1 * b2 * (rs + half*(s11*nr+nl*s22+s12)) / (beta * u * ep * bar.a)
		zt := c1 * b2 * (rt + half*(t11*nr+nl*t22+t12))
		if bar.e > 0 {
			resT0 = zt / (beta * u * ep * bar.e)
		} else {
			d.require(zt == 0)
		}
		tb0 = 0.5 * float64(n2+1) * c1 * (b1 + nl*b2) / (beta * u)
		tc0 = 0.5 * float64(n1+1) * (c2 + c1*nr) * b2 / (beta * u)
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
	foldA0 = c1 * (b1hat / bar.b) * faFold / (beta * u * ep * bar.a)
	d.require(allFinite(a1r.Data) && isFinite(faFold) && isFinite(b1hat))
	var fB []float64
	var b1r blas64.General
	if m > 0 {
		b1r = b1f.RawMatrix()
		fB = make([]float64, n1*m)
		copyStrided(fB, m, bq, m, n1, m)
		gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n1, n1, n, t), gen(n1, m, b1r.Stride, b1r.Data), -1, gen(n1, m, m, fB))
		fbFold := froStrided(fB, m, n1, m) + half1*absProductFro(t, n, b1r.Data, b1r.Stride, bq, m, n1, n1, m)
		foldB0 = c1 * fbFold / (ep * bar.b) / (beta * u)
		d.require(allFinite(b1r.Data) && isFinite(fbFold))
	}
	sp.zS, sp.zT, sp.z2 = resS0+foldA0, resT0, tb0+tc0+foldB0
	if n2 > 0 {
		sp.sharpen = func() (float64, float64, float64) {
			rb := newRidgeBounds(n1, n2, m, p, r, l, bq0, cz0)
			if !rb.okR && !rb.okL {
				return resS0 + foldA0, resT0, tb0 + tc0 + foldB0
			}
			sharpRes := func(f []float64) float64 {
				res := sylvesterResidual(f, n, n1, r, l)
				x1 := absMulTight(stridedBlock(f, n, 0, 0, n1, n1), r, n1, n1, n2)
				x2 := absMulTight(l, stridedBlock(f, n, n1, n1, n2, n2), n1, n2, n2)
				for i := range n1 {
					for j := range n2 {
						k := i*n2 + j
						res[k] = math.Abs(res[k]) + half*(x1[k]+x2[k]+math.Abs(f[i*n+n1+j]))
					}
				}
				return math.Min(rb.c1*rb.right(res, n1), rb.left(res, n2)*rb.b2) / (bar.c * bar.b)
			}
			dc := absMulTight(rb.ct1, r, p, n1, n2)
			ct2 := stridedBlock(cz0, n, 0, n1, p, n2)
			for i := range dc {
				dc[i] = half1 * (dc[i] + math.Abs(ct2[i]))
			}
			db := absMulTight(l, rb.bt2, n1, n2, m)
			bt1 := stridedBlock(bq0, m, 0, 0, n1, m)
			half2 := 0.5 * float64(n2+1) * ep
			for i := range db {
				db[i] = half2 * (db[i] + math.Abs(bt1[i]))
			}
			wB := math.Min(b1hat, bar.b+nr*rb.b2) / bar.b
			fAabs := absMulTight(stridedBlock(t, n, 0, 0, n1, n1), stridedBlock(a1r.Data, a1r.Stride, 0, 0, n1, n1), n1, n1, n1)
			for i := range n1 {
				for j := range n1 {
					fAabs[i*n1+j] = math.Abs(fA[i*n1+j]) + half1*(fAabs[i*n1+j]+math.Abs(s[i*n+j]))
				}
			}
			zS := math.Min(resS0, sharpRes(s)/(beta*u*ep*bar.a)) + math.Min(foldA0, rb.left(fAabs, n1)/bar.c*wB/(beta*u*ep*bar.a))
			zT := resT0
			if bar.e > 0 {
				zT = math.Min(resT0, sharpRes(t)/(beta*u*ep*bar.e))
			}
			z2 := math.Min(tc0, rb.right(dc, p)/(ep*bar.c*bar.b)/(beta*u)) + math.Min(tb0, rb.left(db, m)/(ep*bar.c*bar.b)/(beta*u))
			fb := foldB0
			if m > 0 {
				fBabs := absMulTight(stridedBlock(t, n, 0, 0, n1, n1), stridedBlock(b1r.Data, b1r.Stride, 0, 0, n1, m), n1, n1, m)
				for i := range n1 {
					for j := range m {
						fBabs[i*m+j] = math.Abs(fB[i*m+j]) + half1*(fBabs[i*m+j]+math.Abs(bq[i*m+j]))
					}
				}
				fb = math.Min(foldB0, rb.left(fBabs, m)/(ep*bar.c*bar.b)/(beta*u))
			}
			return zS, zT, z2 + fb
		}
	}
	d.splits = append(d.splits, sp)
}

// sylvesterResidual is F11·R − L·F22 + F12 (n1×n2) of the n×n form f.
func sylvesterResidual(f []float64, n, n1 int, r, l []float64) []float64 {
	n2 := n - n1
	res := make([]float64, n1*n2)
	copyStrided(res, n2, f[n1:], n, n1, n2)
	gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n1, n1, n, f), gen(n1, n2, n2, r), 1, gen(n1, n2, n2, res))
	gemmIf(blas.NoTrans, blas.NoTrans, -1, gen(n1, n2, n2, l), gen(n2, n2, n, f[n1*n+n1:]), 1, gen(n1, n2, n2, res))
	return res
}

// ridgeBounds sharpens bounds through the decoupling (R, L) of one split:
// with N_R = Rᵀ(I + RRᵀ)⁻¹ and Y_R ≥ |I − N_R·R| (rounding included), a
// perturbation Δ (r×n2) splits as Δ = K1·R + K2, K1 = Δ·N_R, K2 = Δ·Y_R,
// so ‖Δ·M22·B̃2‖ ≤ ‖M‖(‖[K1 K2]‖‖B̃2‖ + ‖K1‖‖L·B̃2‖) (right); with
// N_L = Lᵀ(I + LLᵀ)⁻¹ and Y_L ≥ |I − L·N_L|, Δ (n1×c) = K1 − L·K2,
// K2 = −N_L·Δ, K1 = Y_L·Δ, so ‖C̃1·M11·Δ‖ ≤ ‖M‖(‖C̃1‖‖[K1; K2]‖ + ‖C̃1·R‖‖K2‖)
// (left). Each is taken only where smaller than the plain ‖Δ‖‖B̃2‖ or
// ‖C̃1‖‖Δ‖; a side whose R or L is non-finite or whose Cholesky fails stays
// plain. The second-order term K·M11·Res·M22 is dropped (first order).
type ridgeBounds struct {
	nmR, yR, nmL, yL []float64
	okR, okL         bool
	n1, n2           int
	ct1, bt2         []float64
	c1, b2, cr, lb   float64
}

func newRidgeBounds(n1, n2, m, p int, r, l, bq0, cz0 []float64) ridgeBounds {
	ep := eps()
	n := n1 + n2
	rb := ridgeBounds{n1: n1, n2: n2, ct1: stridedBlock(cz0, n, 0, 0, p, n1), bt2: stridedBlock(bq0, m, n1, 0, n2, m)}
	rb.c1, rb.b2 = froAll(rb.ct1), froAll(rb.bt2)
	rb.lb = absProductFro(l, n2, rb.bt2, m, nil, 0, n1, n2, m)
	if m > 0 {
		lbm := make([]float64, n1*m)
		gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(n1, n2, n2, l), gen(n2, m, m, rb.bt2), 0, gen(n1, m, m, lbm))
		rb.lb = math.Min(rb.lb, froAll(lbm)+0.5*float64(n2+1)*ep*rb.lb)
	}
	rb.cr = absProductFro(rb.ct1, n1, r, n2, nil, 0, p, n1, n2)
	if p > 0 {
		crm := make([]float64, p*n2)
		gemmIf(blas.NoTrans, blas.NoTrans, 1, gen(p, n1, n1, rb.ct1), gen(n1, n2, n2, r), 0, gen(p, n2, n2, crm))
		rb.cr = math.Min(rb.cr, froAll(crm)+0.5*float64(n1+1)*ep*rb.cr)
	}
	rb.nmR, rb.yR, rb.okR = ridgeFactor(r, n1, n2, true)
	rb.nmL, rb.yL, rb.okL = ridgeFactor(l, n1, n2, false)
	return rb
}

// right bounds ‖Δ·M22·B̃2‖/‖M‖ for |Δ| ≤ dabs (rows×n2).
func (rb ridgeBounds) right(dabs []float64, rows int) float64 {
	plain := froAll(dabs) * rb.b2
	if !rb.okR {
		return plain
	}
	k1 := froAll(absMulTight(dabs, rb.nmR, rows, rb.n2, rb.n1))
	k2 := froAll(absMulTight(dabs, rb.yR, rows, rb.n2, rb.n2))
	v := math.Hypot(k1, k2)*rb.b2 + k1*rb.lb
	if !(v < plain) {
		return plain
	}
	return v
}

// left bounds ‖C̃1·M11·Δ‖/‖M‖ for |Δ| ≤ dabs (n1×cols).
func (rb ridgeBounds) left(dabs []float64, cols int) float64 {
	plain := rb.c1 * froAll(dabs)
	if !rb.okL {
		return plain
	}
	h1 := froAll(absMulTight(rb.nmL, dabs, rb.n2, rb.n1, cols))
	h2 := froAll(absMulTight(rb.yL, dabs, rb.n1, rb.n1, cols))
	v := rb.c1*math.Hypot(h1, h2) + rb.cr*h1
	if !(v < plain) {
		return plain
	}
	return v
}

// ridgeFactor returns N = Xᵀ(I + XXᵀ)⁻¹ (n2×n1) for X n1×n2 and, for
// right, Y ≥ |I − N·X| (n2×n2), else Y ≥ |I − X·N| (n1×n1), each including
// a componentwise bound on its rounding.
func ridgeFactor(x []float64, n1, n2 int, right bool) (nm, y []float64, ok bool) {
	if n1 == 0 || n2 == 0 || !allFinite(x[:n1*n2]) {
		return nil, nil, false
	}
	X := mat.NewDense(n1, n2, slices.Clone(x[:n1*n2]))
	var g mat.SymDense
	g.SymOuterK(1, X)
	for i := range n1 {
		g.SetSym(i, i, g.At(i, i)+1)
	}
	var ch mat.Cholesky
	if !ch.Factorize(&g) {
		return nil, nil, false
	}
	var sol mat.Dense
	if err := ch.SolveTo(&sol, X); err != nil {
		return nil, nil, false
	}
	nm = make([]float64, n2*n1)
	for i := range n2 {
		for j := range n1 {
			nm[i*n1+j] = sol.At(j, i)
		}
	}
	if !allFinite(nm) {
		return nil, nil, false
	}
	if right {
		y = make([]float64, n2*n2)
		g := 0.5 * float64(n1+2) * eps()
		for i := range n2 {
			for j := range n2 {
				v, a := 0.0, 0.0
				if i == j {
					v, a = 1, 1
				}
				for k := range n1 {
					pr := nm[i*n1+k] * x[k*n2+j]
					v -= pr
					a += math.Abs(pr)
				}
				y[i*n2+j] = math.Abs(v) + g*a
			}
		}
	} else {
		y = make([]float64, n1*n1)
		g := 0.5 * float64(n2+2) * eps()
		for i := range n1 {
			for j := range n1 {
				v, a := 0.0, 0.0
				if i == j {
					v, a = 1, 1
				}
				for k := range n2 {
					pr := x[i*n2+k] * nm[k*n1+j]
					v -= pr
					a += math.Abs(pr)
				}
				y[i*n1+j] = math.Abs(v) + g*a
			}
		}
	}
	return nm, y, allFinite(y)
}

// absMulTight is |A|·|B| for A r×k, B k×c with tight strides.
func absMulTight(a, b []float64, r, k, c int) []float64 {
	out := make([]float64, r*c)
	for i := range r {
		for l := range k {
			v := math.Abs(a[i*k+l])
			if v == 0 {
				continue
			}
			for j := range c {
				out[i*c+j] += v * math.Abs(b[l*c+j])
			}
		}
	}
	return out
}

// subBlock copies the r×c block of d (stride ld) at (r0, c0).
func stridedBlock(d []float64, ld, r0, c0, r, c int) []float64 {
	out := make([]float64, r*c)
	for i := range r {
		copy(out[i*c:(i+1)*c], d[(r0+i)*ld+c0:(r0+i)*ld+c0+c])
	}
	return out
}

func froAll(x []float64) float64 {
	if len(x) == 0 {
		return 0
	}
	return froStrided(x, len(x), 1, len(x))
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
