package controlsys

import (
	"math"
	"math/big"
	"strings"

	"plantcontrol.org/v1/gonum/mat"
)

// Exact-E extended-precision reference for the Riccati equations with
// descriptor E, for tests and review sweeps (TDFWXM). Nothing E-derived is
// rounded to float64 before the end: not the explicit model Ã = E⁻¹A,
// B̃ = E⁻¹B (cond(E)·eps in K), and not Y before K is formed from it. The
// rv39 oracle did both; its gain of the rounded Y moved K by 2.7e-6 (seed
// 13857) and 2e-4 (37017) and produced false certified-wrong rows. Y solves
// the explicit equation (Y = EᵀXE, what Lqr and Dlqr return), X = E⁻ᵀYE⁻¹ the
// descriptor one (what Care and Dare return), K the gain (same for both).
//
// Two independent methods, each at a caller-chosen precision and sharing
// no code with riccati.go:
//   - riccatiExactReference: Dare by structure-preserving doubling, Care by
//     the scaled matrix sign function of the Hamiltonian; no initial gain;
//   - riccatiExactNewtonReference: Hewer (Dare) or Kleinman (Care) Newton
//     iteration with Kronecker Stein/Lyapunov solves from a stabilizing K0.
//
// riccatiExactSensitivity is the eps band of Y6BJEX rule 1: the first-order
// worst-case move of the reference under componentwise (1 ± eps)
// perturbations of A, B, Q, R, S and E (Q and R kept symmetric), plus the
// gain's move under Y(1 ± eps) and X(1 ± eps).

const riccatiExactPrec = 320

type xbMat struct {
	r, c int
	prec uint
	d    []*big.Float
}

func xbNew(r, c int, prec uint) xbMat {
	m := xbMat{r, c, prec, make([]*big.Float, r*c)}
	for i := range m.d {
		m.d[i] = new(big.Float).SetPrec(prec)
	}
	return m
}

func xbEye(n int, prec uint) xbMat {
	m := xbNew(n, n, prec)
	for i := range n {
		m.at(i, i).SetInt64(1)
	}
	return m
}

// xbFrom converts a (r×c zeros when nil) to prec bits exactly.
func xbFrom(a *mat.Dense, r, c int, prec uint) xbMat {
	m := xbNew(r, c, prec)
	if a == nil {
		return m
	}
	for i := range r {
		for j := range c {
			m.at(i, j).SetFloat64(a.At(i, j))
		}
	}
	return m
}

func (m xbMat) at(i, j int) *big.Float { return m.d[i*m.c+j] }

func (m xbMat) t() xbMat {
	o := xbNew(m.c, m.r, m.prec)
	for i := range m.r {
		for j := range m.c {
			o.at(j, i).Set(m.at(i, j))
		}
	}
	return o
}

func xbMul(a, b xbMat) xbMat {
	o := xbNew(a.r, b.c, a.prec)
	p := new(big.Float).SetPrec(a.prec)
	for i := range a.r {
		for k := range a.c {
			aik := a.at(i, k)
			if aik.Sign() == 0 {
				continue
			}
			for j := range b.c {
				o.at(i, j).Add(o.at(i, j), p.Mul(aik, b.at(k, j)))
			}
		}
	}
	return o
}

// xbAxpy returns a + s·b.
func xbAxpy(a xbMat, s float64, b xbMat) xbMat {
	o := xbNew(a.r, a.c, a.prec)
	p := new(big.Float).SetPrec(a.prec)
	sb := new(big.Float).SetPrec(a.prec).SetFloat64(s)
	for i := range o.d {
		o.d[i].Add(a.d[i], p.Mul(sb, b.d[i]))
	}
	return o
}

func xbScale(s *big.Float, a xbMat) xbMat {
	o := xbNew(a.r, a.c, a.prec)
	for i := range o.d {
		o.d[i].Mul(s, a.d[i])
	}
	return o
}

func xbSym(a xbMat) xbMat { return xbAxpy(xbScale(big.NewFloat(0.5), a), 0.5, a.t()) }

// xbSolve solves a·x = b by Gaussian elimination with partial pivoting and
// returns log₂|det a|; ok is false for an exactly singular pivot.
func xbSolve(a, b xbMat) (x xbMat, log2Det float64, ok bool) {
	n, prec := a.r, a.prec
	a, b = xbAxpy(a, 0, a), xbAxpy(b, 0, b)
	p, q := new(big.Float).SetPrec(prec), new(big.Float).SetPrec(prec)
	f := new(big.Float).SetPrec(prec)
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if p.Abs(a.at(i, k)).Cmp(q.Abs(a.at(piv, k))) > 0 {
				piv = i
			}
		}
		if a.at(piv, k).Sign() == 0 {
			return b, 0, false
		}
		for j := range n {
			a.d[k*n+j], a.d[piv*n+j] = a.d[piv*n+j], a.d[k*n+j]
		}
		for j := range b.c {
			b.d[k*b.c+j], b.d[piv*b.c+j] = b.d[piv*b.c+j], b.d[k*b.c+j]
		}
		mant := new(big.Float)
		e := a.at(k, k).MantExp(mant)
		mf, _ := mant.Float64()
		log2Det += float64(e) + math.Log2(math.Abs(mf))
		for i := k + 1; i < n; i++ {
			if a.at(i, k).Sign() == 0 {
				continue
			}
			f.Quo(a.at(i, k), a.at(k, k))
			for j := k; j < n; j++ {
				a.at(i, j).Sub(a.at(i, j), p.Mul(f, a.at(k, j)))
			}
			for j := range b.c {
				b.at(i, j).Sub(b.at(i, j), p.Mul(f, b.at(k, j)))
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		for j := range b.c {
			for k := i + 1; k < n; k++ {
				b.at(i, j).Sub(b.at(i, j), p.Mul(a.at(i, k), b.at(k, j)))
			}
			b.at(i, j).Quo(b.at(i, j), a.at(i, i))
		}
	}
	return b, log2Det, true
}

func xbDense(x xbMat) *mat.Dense {
	o := mat.NewDense(x.r, x.c, nil)
	for i := range x.r {
		for j := range x.c {
			v, _ := x.at(i, j).Float64()
			o.Set(i, j, v)
		}
	}
	return o
}

func xbMaxAbs(m xbMat) *big.Float {
	mx := new(big.Float).SetPrec(m.prec)
	v := new(big.Float).SetPrec(m.prec)
	for _, x := range m.d {
		if v.Abs(x).Cmp(mx) > 0 {
			mx.Set(v)
		}
	}
	return mx
}

// xbRelDiff returns max|a − b| / max|b| evaluated in a's precision.
func xbRelDiff(a, b xbMat) float64 {
	d := xbMaxAbs(xbAxpy(a, -1, b))
	nb := xbMaxAbs(b)
	if nb.Sign() == 0 {
		v, _ := d.Float64()
		return v
	}
	v, _ := d.Quo(d, nb).Float64()
	return v
}

// xbLyap solves the Lyapunov equation aᵀy + ya + q = 0 (continuous) or
// the Stein equation aᵀya − y + q = 0 by a Kronecker solve in a's
// precision; ok is false when the operator is exactly singular.
func xbLyap(continuous bool, a, q xbMat) (xbMat, bool) {
	n, prec := a.r, a.prec
	M, rhs := xbNew(n*n, n*n, prec), xbNew(n*n, 1, prec)
	p := new(big.Float).SetPrec(prec)
	for i := range n {
		for j := range n {
			row := i*n + j
			if continuous {
				rhs.d[row].Neg(q.at(i, j))
				for l := range n {
					c := M.at(row, l*n+j)
					c.Add(c, a.at(l, i))
					c = M.at(row, i*n+l)
					c.Add(c, a.at(l, j))
				}
				continue
			}
			rhs.d[row].Set(q.at(i, j))
			M.at(row, row).Add(M.at(row, row), big.NewFloat(1))
			for l := range n {
				for o := range n {
					c := M.at(row, l*n+o)
					c.Sub(c, p.Mul(a.at(l, i), a.at(o, j)))
				}
			}
		}
	}
	yv, _, ok := xbSolve(M, rhs)
	if !ok {
		return xbMat{}, false
	}
	y := xbNew(n, n, prec)
	for i := range y.d {
		y.d[i].Set(yv.d[i])
	}
	return xbSym(y), true
}

// xbStable reports whether acl is Hurwitz (continuous) or Schur stable,
// in acl's precision with nothing rounded to float64: the Lyapunov or
// Stein solution P for q = I is positive definite (symmetric elimination,
// every pivot > 0) exactly when it is. A float64 eigenvalue test of the
// rounded closed loop rejects stable non-normal loops: with eigenvalues
// −6.4e15 and −0.78, rounding the entries moves the small one to 0.
func xbStable(continuous bool, acl xbMat) bool {
	P, ok := xbLyap(continuous, acl, xbEye(acl.r, acl.prec))
	if !ok {
		return false
	}
	f, p := new(big.Float).SetPrec(P.prec), new(big.Float).SetPrec(P.prec)
	for k := range P.r {
		if P.at(k, k).Sign() <= 0 {
			return false
		}
		for i := k + 1; i < P.r; i++ {
			f.Quo(P.at(i, k), P.at(k, k))
			for j := k; j < P.r; j++ {
				P.at(i, j).Sub(P.at(i, j), p.Mul(f, P.at(k, j)))
			}
		}
	}
	return true
}

type xbModel struct {
	n, m                int
	prec                uint
	e, a, b, q, r, s, g xbMat
	as, h               xbMat
}

// xbExplicitModel forms Ã = E⁻¹A, B̃ = E⁻¹B in prec bits (E nil: I) and the
// cross-term-free data As = Ã − B̃R⁻¹Sᵀ, H = Q − SR⁻¹Sᵀ, G = B̃R⁻¹B̃ᵀ.
func xbExplicitModel(A, B, Q, R, S, E *mat.Dense, prec uint) (xbModel, bool) {
	n, m := B.Dims()
	md := xbModel{n: n, m: m, prec: prec}
	md.e = xbEye(n, prec)
	md.a, md.b = xbFrom(A, n, n, prec), xbFrom(B, n, m, prec)
	if E != nil {
		md.e = xbFrom(E, n, n, prec)
		var ok1, ok2 bool
		md.a, _, ok1 = xbSolve(md.e, md.a)
		md.b, _, ok2 = xbSolve(md.e, md.b)
		if !ok1 || !ok2 {
			return md, false
		}
	}
	md.q, md.r, md.s = xbFrom(Q, n, n, prec), xbFrom(R, m, m, prec), xbFrom(S, n, m, prec)
	rst, _, ok1 := xbSolve(md.r, md.s.t())
	rbt, _, ok2 := xbSolve(md.r, md.b.t())
	if !ok1 || !ok2 {
		return md, false
	}
	md.as = xbAxpy(md.a, -1, xbMul(md.b, rst))
	md.h = xbSym(xbAxpy(md.q, -1, xbMul(md.s, rst)))
	md.g = xbSym(xbMul(md.b, rbt))
	return md, true
}

func (md xbModel) gain(continuous bool, y xbMat) (xbMat, bool) {
	bty := xbMul(md.b.t(), y)
	if continuous {
		k, _, ok := xbSolve(md.r, xbAxpy(bty, 1, md.s.t()))
		return k, ok
	}
	k, _, ok := xbSolve(xbAxpy(md.r, 1, xbMul(bty, md.b)), xbAxpy(xbMul(bty, md.a), 1, md.s.t()))
	return k, ok
}

// residual returns the relative residual of y in the cross-term-free form.
func (md xbModel) residual(continuous bool, y xbMat) float64 {
	var parts []xbMat
	if continuous {
		parts = []xbMat{xbMul(md.as.t(), y), xbMul(y, md.as), xbScale(big.NewFloat(-1), xbMul(xbMul(y, md.g), y)), md.h}
	} else {
		w, _, ok := xbSolve(xbAxpy(xbEye(md.n, md.prec), 1, xbMul(md.g, y)), md.as)
		if !ok {
			return math.Inf(1)
		}
		parts = []xbMat{xbScale(big.NewFloat(-1), y), xbMul(xbMul(md.as.t(), y), w), md.h}
	}
	sum, scale := xbNew(md.n, md.n, md.prec), new(big.Float).SetPrec(md.prec)
	for _, p := range parts {
		sum = xbAxpy(sum, 1, p)
		if v := xbMaxAbs(p); v.Cmp(scale) > 0 {
			scale = v
		}
	}
	if scale.Sign() == 0 {
		return 0
	}
	v, _ := xbMaxAbs(sum).Quo(xbMaxAbs(sum), scale).Float64()
	return v
}

type riccatiExactRef struct {
	Y, X, K *mat.Dense
	y, x, k xbMat
	Iter    int
	Resid   float64
	OK      bool
}

func (md xbModel) finish(continuous bool, y xbMat, conv bool, iter int) riccatiExactRef {
	ref := riccatiExactRef{y: y, Iter: iter}
	k, ok := md.gain(continuous, y)
	if !ok {
		return ref
	}
	z, _, ok1 := xbSolve(md.e.t(), y)
	xt, _, ok2 := xbSolve(md.e.t(), z.t())
	if !ok1 || !ok2 {
		return ref
	}
	ref.k, ref.x = k, xbSym(xt.t())
	ref.Y, ref.X, ref.K = xbDense(y), xbDense(ref.x), xbDense(k)
	ref.Resid = md.residual(continuous, y)
	ref.OK = conv && ref.Resid <= 1e-40 && xbStable(continuous, xbAxpy(md.a, -1, xbMul(md.b, k)))
	return ref
}

// riccatiExactReference returns the stabilizing solution of the Dare
// (continuous false) or Care with descriptor E (nil for none) and cross
// term S (nil for none) by doubling (Dare) or the matrix sign function
// (Care) on the explicit model in prec bits.
func riccatiExactReference(continuous bool, A, B, Q, R, S, E *mat.Dense, prec uint) riccatiExactRef {
	md, ok := xbExplicitModel(A, B, Q, R, S, E, prec)
	if !ok {
		return riccatiExactRef{}
	}
	if continuous {
		return md.careSign()
	}
	return md.dareDoubling()
}

func (md xbModel) stopTol() float64 { return math.Ldexp(1, -int(md.prec)/2) }

// dareDoubling: A₊ = A(I+GH)⁻¹A, G₊ = G + A(I+GH)⁻¹GAᵀ, H₊ = H +
// AᵀH(I+GH)⁻¹A from (As, G, H); H → Y and A → 0 quadratically.
func (md xbModel) dareDoubling() riccatiExactRef {
	a, g, h := md.as, md.g, md.h
	eye := xbEye(md.n, md.prec)
	a0 := xbMaxAbs(a)
	a0.Add(a0, big.NewFloat(1))
	extra, conv := -1, false
	it := 0
	for ; it < 200 && extra != 0; it++ {
		w := xbAxpy(eye, 1, xbMul(g, h))
		wa, _, ok1 := xbSolve(w, a)
		wg, _, ok2 := xbSolve(w, g)
		if !ok1 || !ok2 {
			return riccatiExactRef{}
		}
		hn := xbSym(xbAxpy(h, 1, xbMul(xbMul(a.t(), h), wa)))
		g = xbSym(xbAxpy(g, 1, xbMul(xbMul(a, wg), a.t())))
		a = xbMul(a, wa)
		d := xbRelDiff(hn, h)
		h = hn
		if extra > 0 {
			extra--
		} else if d <= md.stopTol() {
			extra = 2
		}
	}
	if extra == 0 {
		ar := xbMaxAbs(a)
		v, _ := ar.Quo(ar, a0).Float64()
		conv = v <= 1e-30
	}
	return md.finish(false, h, conv, it)
}

// careSign: Z₊ = (cZ + (cZ)⁻¹)/2 from the Hamiltonian [As −G; −H −Asᵀ],
// c = |det Z|^(−1/2n) until the step is below 1e-3; then sign(H)·[I; Y] =
// −[I; Y] is solved for Y in the least-squares sense.
func (md xbModel) careSign() riccatiExactRef {
	n, prec := md.n, md.prec
	nn := 2 * n
	z := xbNew(nn, nn, prec)
	for i := range n {
		for j := range n {
			z.at(i, j).Set(md.as.at(i, j))
			z.at(i, n+j).Neg(md.g.at(i, j))
			z.at(n+i, j).Neg(md.h.at(i, j))
			z.at(n+i, n+j).Neg(md.as.at(j, i))
		}
	}
	eye := xbEye(nn, prec)
	scaled, extra := true, -1
	it := 0
	for ; it < 200 && extra != 0; it++ {
		inv, l2d, ok := xbSolve(z, eye)
		if !ok {
			return riccatiExactRef{}
		}
		c := new(big.Float).SetPrec(prec).SetInt64(1)
		if scaled {
			ex := -l2d / float64(nn)
			fl := math.Floor(ex)
			c.SetMantExp(big.NewFloat(math.Exp2(ex-fl)), int(fl))
		}
		ci := new(big.Float).SetPrec(prec).Quo(big.NewFloat(1), c)
		zn := xbScale(big.NewFloat(0.5), xbAxpy(xbScale(c, z), 1, xbScale(ci, inv)))
		d := xbRelDiff(zn, z)
		z = zn
		if d < 1e-3 {
			scaled = false
		}
		if extra > 0 {
			extra--
		} else if d <= md.stopTol() {
			extra = 2
		}
	}
	m, rhs := xbNew(nn, n, prec), xbNew(nn, n, prec)
	for i := range n {
		for j := range n {
			m.at(i, j).Set(z.at(i, n+j))
			m.at(n+i, j).Set(z.at(n+i, n+j))
			rhs.at(i, j).Neg(z.at(i, j))
			rhs.at(n+i, j).Neg(z.at(n+i, j))
		}
		m.at(n+i, i).Add(m.at(n+i, i), big.NewFloat(1))
		rhs.at(i, i).Sub(rhs.at(i, i), big.NewFloat(1))
	}
	y, _, ok := xbSolve(xbMul(m.t(), m), xbMul(m.t(), rhs))
	if !ok {
		return riccatiExactRef{}
	}
	return md.finish(true, xbSym(y), extra == 0, it)
}

// riccatiExactNewtonReference returns the stabilizing solution by Hewer's
// (Dare) or Kleinman's (Care) iteration in prec bits from the stabilizing
// gain K0, with Kronecker Stein or Lyapunov solves.
func riccatiExactNewtonReference(continuous bool, A, B, Q, R, S, E, K0 *mat.Dense, prec uint) riccatiExactRef {
	md, ok := xbExplicitModel(A, B, Q, R, S, E, prec)
	if !ok {
		return riccatiExactRef{}
	}
	n := md.n
	k := xbFrom(K0, md.m, n, prec)
	var y xbMat
	extra, it := -1, 0
	for ; it < 100 && extra != 0; it++ {
		acl := xbAxpy(md.a, -1, xbMul(md.b, k))
		sk := xbMul(md.s, k)
		qk := xbAxpy(xbAxpy(xbAxpy(md.q, 1, xbMul(xbMul(k.t(), md.r), k)), -1, sk), -1, sk.t())
		yn, ok := xbLyap(continuous, acl, qk)
		if !ok {
			return riccatiExactRef{}
		}
		d := math.Inf(1)
		if y.r > 0 {
			d = xbRelDiff(yn, y)
		}
		y = yn
		if k, ok = md.gain(continuous, y); !ok {
			return riccatiExactRef{}
		}
		if extra > 0 {
			extra--
		} else if d <= md.stopTol() {
			extra = 2
		}
	}
	return md.finish(continuous, y, extra == 0, it)
}

type riccatiExactSens struct {
	Y, X, K float64
	// KofY, KofX: the gain's move under Y(1 ± eps), X(1 ± eps).
	KofY, KofX float64
}

// KBand is rule 1's K sensitivity: data perturbations and the rounding of
// Y or X, from which a float64 solver forms its gain.
func (s riccatiExactSens) KBand() float64 { return max(s.K, s.KofY, s.KofX) }

// riccatiExactSensitivity returns the first-order worst case, over all
// componentwise perturbations |δm| ≤ eps·|m| of the entries of the
// matrices named in which (letters of "ABQRSE"; Q and R kept symmetric),
// of the relative 1-norm moves of Y, X and K: Σ |∂ref/∂m|·eps·|m| from
// one-ulp differences of the 320-bit reference per entry (zero entries
// stay zero). KofY and KofX bound the gain's move the same way under
// componentwise eps perturbations of Y or X. A perturbed problem the
// primary method does not certify is solved by Newton from ref.K; ok is
// false when neither certifies it.
func riccatiExactSensitivity(continuous bool, A, B, Q, R, S, E *mat.Dense, ref riccatiExactRef, which string) (riccatiExactSens, bool) {
	var s riccatiExactSens
	if !ref.OK {
		return s, false
	}
	n, m := B.Dims()
	bY, bX, bK := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil), mat.NewDense(m, n, nil)
	accum := func(b *mat.Dense, d xbMat, f float64) {
		dd := xbDense(d)
		raw, dr := b.RawMatrix(), dd.RawMatrix()
		for i := range raw.Rows {
			for j := range raw.Cols {
				raw.Data[i*raw.Stride+j] += f * math.Abs(dr.Data[i*dr.Stride+j])
			}
		}
	}
	data := []struct {
		name byte
		m    *mat.Dense
		sym  bool
	}{{'A', A, false}, {'B', B, false}, {'Q', Q, true}, {'R', R, true}, {'S', S, false}, {'E', E, false}}
	for di, d := range data {
		if d.m == nil || !strings.ContainsRune(which, rune(d.name)) {
			continue
		}
		r, c := d.m.Dims()
		for i := range r {
			for j := range c {
				v := d.m.At(i, j)
				if d.sym && j > i || v == 0 {
					continue
				}
				vp := math.Nextafter(v, math.Inf(1))
				f := eps() * math.Abs(v) / (vp - v)
				mp := mat.DenseCopyOf(d.m)
				mp.Set(i, j, vp)
				if d.sym {
					mp.Set(j, i, vp)
				}
				args := [6]*mat.Dense{A, B, Q, R, S, E}
				args[di] = mp
				p := riccatiExactReference(continuous, args[0], args[1], args[2], args[3], args[4], args[5], riccatiExactPrec)
				if !p.OK {
					p = riccatiExactNewtonReference(continuous, args[0], args[1], args[2], args[3], args[4], args[5], ref.K, riccatiExactPrec)
				}
				if !p.OK {
					return s, false
				}
				accum(bY, xbAxpy(p.y, -1, ref.y), f)
				accum(bX, xbAxpy(p.x, -1, ref.x), f)
				accum(bK, xbAxpy(p.k, -1, ref.k), f)
			}
		}
	}
	s.Y, s.X, s.K = mat.Norm(bY, 1)/mat.Norm(ref.Y, 1), mat.Norm(bX, 1)/mat.Norm(ref.X, 1), mat.Norm(bK, 1)/mat.Norm(ref.K, 1)
	md, ok := xbExplicitModel(A, B, Q, R, S, E, riccatiExactPrec)
	if !ok {
		return s, false
	}
	a, b := xbFrom(A, n, n, md.prec), xbFrom(B, n, m, md.prec)
	gainOfX := func(x xbMat) (xbMat, bool) {
		btx := xbMul(b.t(), x)
		if continuous {
			k, _, ok := xbSolve(md.r, xbAxpy(xbMul(btx, md.e), 1, md.s.t()))
			return k, ok
		}
		k, _, ok := xbSolve(xbAxpy(md.r, 1, xbMul(btx, b)), xbAxpy(xbMul(btx, a), 1, md.s.t()))
		return k, ok
	}
	bKY, bKX := mat.NewDense(m, n, nil), mat.NewDense(m, n, nil)
	for _, w := range []struct {
		z    xbMat
		b    *mat.Dense
		gain func(xbMat) (xbMat, bool)
	}{{ref.y, bKY, func(y xbMat) (xbMat, bool) { return md.gain(continuous, y) }}, {ref.x, bKX, gainOfX}} {
		for i := range n {
			for j := range i + 1 {
				if w.z.at(i, j).Sign() == 0 {
					continue
				}
				zp := xbAxpy(w.z, 0, w.z)
				dz := new(big.Float).SetPrec(md.prec).Abs(w.z.at(i, j))
				dz.Mul(dz, big.NewFloat(eps()))
				zp.at(i, j).Add(zp.at(i, j), dz)
				if i != j {
					zp.at(j, i).Add(zp.at(j, i), dz)
				}
				k, ok := w.gain(zp)
				if !ok {
					return s, false
				}
				accum(w.b, xbAxpy(k, -1, ref.k), 1)
			}
		}
	}
	s.KofY, s.KofX = mat.Norm(bKY, 1)/mat.Norm(ref.K, 1), mat.Norm(bKX, 1)/mat.Norm(ref.K, 1)
	return s, true
}

// riccatiCertifiedWrong applies Y6BJEX rule 1: Accuracy 0 with an X error
// above max(newtonTol, 4·xBand) or a K error above max(kFloor, 4·kBand),
// the bands from riccatiExactSensitivity. The bands cover data and X
// rounding but not the float64 evaluation of the gain from X; on the rv39
// sweep, explicit cheap-control rows with X exact to eps sit at 1–4× the
// K band, hence rule 1's "a few ×". kFloor is newtonTol for the strict
// letter, gainFloor for the detector band of amendment 2.
func riccatiCertifiedWrong(acc int, xErr, xBand, kErr, kBand, kFloor float64) bool {
	return acc == 0 && (xErr > max(newtonTol, 4*xBand) || kErr > max(kFloor, 4*kBand))
}
