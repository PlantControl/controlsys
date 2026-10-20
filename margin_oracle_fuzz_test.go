package controlsys

// Exact SISO loop oracle for the margin fuzz targets (FuzzAllMargin,
// FuzzMargin, FuzzBandwidth, FuzzDiskMargin).
//
// For the float64 realization (A, b, c, d) the loop L = N/D is formed
// exactly over big.Rat: D = det(λI − A), N = d·D + D − det(λI − A − bc)
// (matrix determinant lemma); a discrete delay of k samples multiplies D by
// z^k. On the boundary L is written in a real variable x: x = ω for
// continuous loops, x = tan(θ/2) with z = (1 + jx)/(1 − jx) for discrete
// ones (N and D share the factor (1 − jx)^deg, which cancels). Then
// |N|² − γ²|D|² and Im(N·conj D) are real polynomials in x with exact
// rational coefficients; their positive roots are isolated by Sturm
// sequences over big.Int and bisected on exact signs to adjacent floats.
// Phase crossovers are the roots of Im(N·conj D) with Re(N·conj D) < 0,
// plus the discrete Nyquist point z = −1 when L(−1) < 0.
//
// A crossing is required (the library must report it) only when it survives
// every model within rounding of the given entries: an odd, isolated root
// whose f (ln|L| − ln γ, or the phase offset from −180°) has opposite signs
// at ω0(1 ± r), the smallest r in 1e-6..1e-2 where both |f| exceed rsens
// (first-order change of f under eps-relative real perturbations of each
// entry of A, b, c, d) plus a floor; r is then its location tolerance. A
// phase crossing also needs L finite and Re L < 0 certain at the root and no
// other root of Im L in the bracket, so the sign change is not a ±180° wrap
// through a pole; a Nyquist crossing needs L(−1) above coefficient noise, as
// AllMargin does. A reported crossing is sound when an exact root lies
// within its tolerance or |f| there is within budget, a generous normwise
// bound on the evaluation error of the realization.

import (
	"math"
	"math/big"
	"math/cmplx"
	"slices"
)

type fzMgPoly []*big.Rat // ascending powers

func fzMgZeros(n int) fzMgPoly {
	p := make(fzMgPoly, n)
	for i := range p {
		p[i] = new(big.Rat)
	}
	return p
}

func (p fzMgPoly) trim() fzMgPoly {
	for len(p) > 0 && p[len(p)-1].Sign() == 0 {
		p = p[:len(p)-1]
	}
	return p
}

func fzMgAdd(p, q fzMgPoly, sq int) fzMgPoly {
	out := fzMgZeros(max(len(p), len(q)))
	for i, x := range p {
		out[i].Set(x)
	}
	var t big.Rat
	for i, x := range q {
		t.SetInt64(int64(sq))
		out[i].Add(out[i], t.Mul(&t, x))
	}
	return out
}

func fzMgMul(p, q fzMgPoly) fzMgPoly {
	if len(p) == 0 || len(q) == 0 {
		return nil
	}
	out := fzMgZeros(len(p) + len(q) - 1)
	var t big.Rat
	for i, x := range p {
		if x.Sign() == 0 {
			continue
		}
		for j, y := range q {
			out[i+j].Add(out[i+j], t.Mul(x, y))
		}
	}
	return out
}

func fzMgScale(p fzMgPoly, r *big.Rat) fzMgPoly {
	out := fzMgZeros(len(p))
	for i, x := range p {
		out[i].Mul(x, r)
	}
	return out
}

func fzMgEval(p fzMgPoly, x *big.Rat) *big.Rat {
	acc := new(big.Rat)
	for _, v := range slices.Backward(p) {
		acc.Mul(acc, x)
		acc.Add(acc, v)
	}
	return acc
}

type fzMgCPoly struct{ re, im fzMgPoly }

func fzMgCMul(a, b fzMgCPoly) fzMgCPoly {
	return fzMgCPoly{
		re: fzMgAdd(fzMgMul(a.re, b.re), fzMgMul(a.im, b.im), -1),
		im: fzMgAdd(fzMgMul(a.re, b.im), fzMgMul(a.im, b.re), 1),
	}
}

// fzMgLn returns ln|r| (−Inf for 0) without overflow.
func fzMgLn(r *big.Rat) float64 {
	if r.Sign() == 0 {
		return math.Inf(-1)
	}
	f := new(big.Float).SetPrec(128).SetRat(r)
	f.Abs(f)
	m := new(big.Float)
	e := f.MantExp(m)
	mf, _ := m.Float64()
	return math.Log(mf) + float64(e)*math.Ln2
}

// fzMgRatio returns a/b as float64 (±Inf for b = 0).
func fzMgRatio(a, b *big.Rat) float64 {
	if b.Sign() == 0 {
		if a.Sign() == 0 {
			return math.NaN()
		}
		return math.Inf(a.Sign() * b.Sign())
	}
	v, _ := new(big.Rat).Quo(a, b).Float64()
	return v
}

type fzMgLoop struct {
	discrete bool
	dt, tau  float64
	n, deg   int
	a, b, c  []float64
	d, normA float64
	num, den fzMgPoly  // ascending in s or z; den carries z^k
	N, D     fzMgCPoly // on the boundary, in x
	fast     []fzMgIEval
}

// fzMgNewLoop builds the exact loop of the SISO realization (a n×n, b, c, d)
// with delay samples of discrete delay (dt > 0) or tau seconds of
// continuous delay.
func fzMgNewLoop(n int, a, b, c []float64, d, dt float64, delay int, tau float64) *fzMgLoop {
	den := fzCharPolyExact(a, n)
	abc := make([]*big.Rat, n*n)
	for i := range n {
		for j := range n {
			abc[i*n+j] = new(big.Rat).Add(fzRat(a[i*n+j]), new(big.Rat).Mul(fzRat(b[i]), fzRat(c[j])))
		}
	}
	cp := fzCharPolyRat(abc, n)
	l := &fzMgLoop{discrete: dt > 0, dt: dt, tau: tau, n: n, a: a, b: b, c: c, d: d, normA: math.Sqrt(fzMgSumSq(a))}
	rd := fzRat(d)
	for k := n; k >= 0; k-- {
		strict := new(big.Rat).Sub(den[k], cp[k])
		l.num = append(l.num, new(big.Rat).Add(new(big.Rat).Mul(rd, den[k]), strict))
		l.den = append(l.den, new(big.Rat).Set(den[k]))
	}
	if delay > 0 {
		l.den = append(fzMgZeros(delay), l.den...)
	}
	l.deg = len(l.den) - 1
	l.N, l.D = l.boundary(l.num), l.boundary(l.den)
	return l
}

func (l *fzMgLoop) boundary(p fzMgPoly) fzMgCPoly {
	if !l.discrete {
		out := fzMgCPoly{fzMgZeros(len(p)), fzMgZeros(len(p))}
		for k, x := range p {
			switch k % 4 {
			case 0:
				out.re[k].Set(x)
			case 1:
				out.im[k].Set(x)
			case 2:
				out.re[k].Neg(x)
			default:
				out.im[k].Neg(x)
			}
		}
		return out
	}
	one := func() *big.Rat { return big.NewRat(1, 1) }
	plus := []fzMgCPoly{{fzMgPoly{one()}, fzMgPoly{new(big.Rat)}}}
	minus := []fzMgCPoly{{fzMgPoly{one()}, fzMgPoly{new(big.Rat)}}}
	for k := 1; k <= l.deg; k++ {
		plus = append(plus, fzMgCMul(plus[k-1], fzMgCPoly{fzMgPoly{one(), new(big.Rat)}, fzMgPoly{new(big.Rat), one()}}))
		minus = append(minus, fzMgCMul(minus[k-1], fzMgCPoly{fzMgPoly{one(), new(big.Rat)}, fzMgPoly{new(big.Rat), big.NewRat(-1, 1)}}))
	}
	out := fzMgCPoly{fzMgZeros(l.deg + 1), fzMgZeros(l.deg + 1)}
	for k, x := range p {
		if x.Sign() == 0 {
			continue
		}
		term := fzMgCMul(plus[k], minus[l.deg-k])
		out.re = fzMgAdd(out.re, fzMgScale(term.re, x), 1)
		out.im = fzMgAdd(out.im, fzMgScale(term.im, x), 1)
	}
	return out
}

func (l *fzMgLoop) nyq() float64 {
	if l.discrete {
		return math.Pi / l.dt
	}
	return math.Inf(1)
}

func (l *fzMgLoop) wOf(x float64) float64 {
	if l.discrete {
		return 2 * math.Atan(x) / l.dt
	}
	return x
}

func (l *fzMgLoop) xOf(w float64) float64 {
	if l.discrete {
		return math.Tan(w * l.dt / 2)
	}
	return w
}

func (l *fzMgLoop) atNyq(w float64) bool { return l.discrete && w >= l.nyq()*(1-1e-15) }

// values returns N and D at ω (common boundary factor included).
func (l *fzMgLoop) values(w float64) (nr, ni, dr, di *big.Rat) {
	if l.atNyq(w) {
		m1 := big.NewRat(-1, 1)
		return fzMgEval(l.num, m1), new(big.Rat), fzMgEval(l.den, m1), new(big.Rat)
	}
	if l.fast == nil {
		for _, p := range []fzMgPoly{l.N.re, l.N.im, l.D.re, l.D.im} {
			l.fast = append(l.fast, fzMgNewIEval(p))
		}
	}
	x := fzRat(l.xOf(w))
	return l.fast[0].at(x), l.fast[1].at(x), l.fast[2].at(x), l.fast[3].at(x)
}

// fzMgIEval evaluates a rational polynomial as an integer one over a common
// denominator, normalizing once per value.
type fzMgIEval struct {
	p   []*big.Int
	den *big.Int
}

func fzMgNewIEval(p fzMgPoly) fzMgIEval {
	l := big.NewInt(1)
	for _, x := range p {
		g := new(big.Int).GCD(nil, nil, l, x.Denom())
		l.Mul(l, new(big.Int).Quo(x.Denom(), g))
	}
	out := make([]*big.Int, len(p))
	for i, x := range p {
		out[i] = new(big.Int).Mul(x.Num(), new(big.Int).Quo(l, x.Denom()))
	}
	return fzMgIEval{out, l}
}

func (e fzMgIEval) at(x *big.Rat) *big.Rat {
	if len(e.p) == 0 {
		return new(big.Rat)
	}
	num, den := x.Num(), x.Denom()
	acc := new(big.Int).Set(e.p[len(e.p)-1])
	dp := big.NewInt(1)
	var t big.Int
	for i := len(e.p) - 2; i >= 0; i-- {
		dp.Mul(dp, den)
		acc.Mul(acc, num)
		acc.Add(acc, t.Mul(e.p[i], dp))
	}
	return new(big.Rat).SetFrac(acc, dp.Mul(dp, e.den))
}

func fzMgAbs2(re, im *big.Rat) *big.Rat {
	a := new(big.Rat).Mul(re, re)
	return a.Add(a, new(big.Rat).Mul(im, im))
}

// lnMag is ln|L(jω)| exactly rounded.
func (l *fzMgLoop) lnMag(w float64) float64 {
	nr, ni, dr, di := l.values(w)
	return (fzMgLn(fzMgAbs2(nr, ni)) - fzMgLn(fzMgAbs2(dr, di))) / 2
}

// reIm returns Re and Im of N·conj D at ω.
func (l *fzMgLoop) reIm(w float64) (re, im *big.Rat) {
	nr, ni, dr, di := l.values(w)
	re = new(big.Rat).Add(new(big.Rat).Mul(nr, dr), new(big.Rat).Mul(ni, di))
	im = new(big.Rat).Sub(new(big.Rat).Mul(ni, dr), new(big.Rat).Mul(nr, di))
	return re, im
}

func fzMgWrap(a float64) float64 {
	a = math.Mod(a, 2*math.Pi)
	if a > math.Pi {
		a -= 2 * math.Pi
	} else if a <= -math.Pi {
		a += 2 * math.Pi
	}
	return a
}

// phase is arg L(jω) in radians, delay included.
func (l *fzMgLoop) phase(w float64) float64 {
	re, im := l.reIm(w)
	s := new(big.Rat).Add(new(big.Rat).Abs(re), new(big.Rat).Abs(im))
	if s.Sign() == 0 {
		return math.NaN()
	}
	return fzMgWrap(math.Atan2(fzMgRatio(im, s), fzMgRatio(re, s)) - w*l.tau)
}

// phaseOff is arg(−L) in (−π, π]: zero at a −180° crossing.
func (l *fzMgLoop) phaseOff(w float64) float64 { return fzMgWrap(l.phase(w) - math.Pi) }

// loop returns L(jω) rounded to complex128.
func (l *fzMgLoop) loop(w float64) complex128 {
	mag := math.Exp(l.lnMag(w))
	return cmplx.Rect(mag, l.phase(w))
}

// resolvent returns x = R·b and y = c·R, R = (sI − A)⁻¹ at ω.
func (l *fzMgLoop) resolvent(w float64) (x, y []complex128, ok bool) {
	var s complex128
	switch {
	case l.atNyq(w):
		s = -1
	case l.discrete:
		s = cmplx.Exp(complex(0, w*l.dt))
	default:
		s = complex(0, w)
	}
	n := l.n
	m := make([]complex128, n*n)
	mt := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			m[i*n+j] = complex(-l.a[i*n+j], 0)
			mt[j*n+i] = m[i*n+j]
		}
		m[i*n+i] += s
		mt[i*n+i] += s
	}
	x, ok1 := fzMgSolve(m, fzLsC(l.b), n)
	y, ok2 := fzMgSolve(mt, fzLsC(l.c), n)
	return x, y, ok1 && ok2
}

// budget bounds, generously, the relative error of L(jω) evaluated by a
// backward-stable method on the realization (normwise eps perturbation):
// 64(n+1)·eps·(‖A‖‖cR‖‖Rb‖ + ‖b‖‖cR‖ + ‖c‖‖Rb‖ + |d|)/|L|, plus the
// rounding of e^{−jωτ}. It bounds reported values.
func (l *fzMgLoop) budget(w float64) float64 {
	const eps = 0x1p-52
	x, y, ok := l.resolvent(w)
	if !ok {
		return math.Inf(1)
	}
	nx, ny := fzMgCNorm(x), fzMgCNorm(y)
	nb, nc := math.Sqrt(fzMgSumSq(l.b)), math.Sqrt(fzMgSumSq(l.c))
	e := eps * (l.normA*nx*ny + nb*ny + nc*nx + math.Abs(l.d)) / math.Exp(l.lnMag(w))
	return 64*float64(l.n+1)*e + 4*eps*w*l.tau
}

// rsens returns 4(n+1)·eps times the first-order bounds on the change of
// ln|L| and of arg L at ω when every given entry of A, b, c, d moves by eps
// relative (real perturbations: δL = Σ y_i·δA_ij·x_j + y·δb + δc·x + δd).
// A sign of ln|L| − ln γ or of the phase offset larger than these is
// certain under rounding of the given data.
func (l *fzMgLoop) rsens(w float64) (gain, phase float64) {
	const eps = 0x1p-52
	x, y, ok := l.resolvent(w)
	if !ok {
		return math.Inf(1), math.Inf(1)
	}
	L := l.loop(w) * cmplx.Exp(complex(0, w*l.tau))
	if L == 0 || cmplx.IsInf(L) || cmplx.IsNaN(L) {
		return math.Inf(1), math.Inf(1)
	}
	il := 1 / L
	add := func(m float64, v complex128) {
		v *= il
		gain += m * math.Abs(real(v))
		phase += m * math.Abs(imag(v))
	}
	n := l.n
	add(math.Abs(l.d), 1)
	for i := range n {
		add(math.Abs(l.b[i]), y[i])
		add(math.Abs(l.c[i]), x[i])
		for j := range n {
			add(math.Abs(l.a[i*n+j]), y[i]*x[j])
		}
	}
	c := 4 * float64(n+1) * eps
	return c * gain, c*phase + 4*eps*w*l.tau
}

func fzMgCNorm(x []complex128) float64 {
	s := 0.0
	for _, v := range x {
		s += real(v)*real(v) + imag(v)*imag(v)
	}
	return math.Sqrt(s)
}

// fzMgSolve solves the n×n complex m·x = b by Gaussian elimination with
// partial pivoting; ok is false for an exactly singular pivot.
func fzMgSolve(m, b []complex128, n int) ([]complex128, bool) {
	m = append([]complex128(nil), m...)
	b = append([]complex128(nil), b...)
	for k := range n {
		piv := k
		for i := k + 1; i < n; i++ {
			if cmplx.Abs(m[i*n+k]) > cmplx.Abs(m[piv*n+k]) {
				piv = i
			}
		}
		if m[piv*n+k] == 0 {
			return nil, false
		}
		for j := range n {
			m[k*n+j], m[piv*n+j] = m[piv*n+j], m[k*n+j]
		}
		b[k], b[piv] = b[piv], b[k]
		for i := k + 1; i < n; i++ {
			f := m[i*n+k] / m[k*n+k]
			for j := k; j < n; j++ {
				m[i*n+j] -= f * m[k*n+j]
			}
			b[i] -= f * b[k]
		}
	}
	for k := n - 1; k >= 0; k-- {
		for j := k + 1; j < n; j++ {
			b[k] -= m[k*n+j] * b[j]
		}
		b[k] /= m[k*n+k]
	}
	return b, true
}

// ---- Sturm root isolation over big.Int ----

type fzMgIPoly []*big.Int // ascending

func fzMgTrimI(p fzMgIPoly) fzMgIPoly {
	for len(p) > 0 && p[len(p)-1].Sign() == 0 {
		p = p[:len(p)-1]
	}
	return p
}

func fzMgPrimitive(p fzMgIPoly) fzMgIPoly {
	g := new(big.Int)
	for _, x := range p {
		g.GCD(nil, nil, g, new(big.Int).Abs(x))
	}
	if g.Sign() == 0 || g.Cmp(big.NewInt(1)) == 0 {
		return p
	}
	out := make(fzMgIPoly, len(p))
	for i, x := range p {
		out[i] = new(big.Int).Quo(x, g)
	}
	return out
}

func fzMgToInt(p fzMgPoly) fzMgIPoly {
	l := big.NewInt(1)
	for _, x := range p {
		d := x.Denom()
		g := new(big.Int).GCD(nil, nil, l, d)
		l.Mul(l, new(big.Int).Quo(d, g))
	}
	out := make(fzMgIPoly, len(p))
	for i, x := range p {
		v := new(big.Int).Mul(x.Num(), new(big.Int).Quo(l, x.Denom()))
		out[i] = v
	}
	return fzMgPrimitive(fzMgTrimI(out))
}

func fzMgDerivI(p fzMgIPoly) fzMgIPoly {
	if len(p) <= 1 {
		return nil
	}
	out := make(fzMgIPoly, len(p)-1)
	for i := 1; i < len(p); i++ {
		out[i-1] = new(big.Int).Mul(p[i], big.NewInt(int64(i)))
	}
	return out
}

// fzMgPremNeg returns −c·(a mod b) for some c > 0, primitive.
func fzMgPremNeg(a, b fzMgIPoly) fzMgIPoly {
	r := make(fzMgIPoly, len(a))
	for i, x := range a {
		r[i] = new(big.Int).Set(x)
	}
	db := len(b) - 1
	lb := b[db]
	alb := new(big.Int).Abs(lb)
	for {
		r = fzMgTrimI(r)
		if len(r)-1 < db || len(r) == 0 {
			break
		}
		dr := len(r) - 1
		la := new(big.Int).Set(r[dr])
		if lb.Sign() < 0 {
			la.Neg(la)
		}
		for i := range r {
			r[i].Mul(r[i], alb)
		}
		var t big.Int
		for i := range b {
			r[i+dr-db].Sub(r[i+dr-db], t.Mul(la, b[i]))
		}
		r[dr].SetInt64(0)
		r = fzMgPrimitive(fzMgTrimI(r))
	}
	for i := range r {
		r[i] = new(big.Int).Neg(r[i])
	}
	return fzMgPrimitive(r)
}

func fzMgSignAt(p fzMgIPoly, x *big.Rat) int {
	if len(p) == 0 {
		return 0
	}
	num, den := x.Num(), x.Denom()
	acc := new(big.Int).Set(p[len(p)-1])
	dp := big.NewInt(1)
	var t big.Int
	for i := len(p) - 2; i >= 0; i-- {
		dp.Mul(dp, den)
		acc.Mul(acc, num)
		acc.Add(acc, t.Mul(p[i], dp))
	}
	return acc.Sign()
}

type fzMgSturm struct{ chain []fzMgIPoly }

func fzMgNewSturm(p fzMgIPoly) *fzMgSturm {
	s := &fzMgSturm{chain: []fzMgIPoly{p}}
	q := fzMgPrimitive(fzMgDerivI(p))
	for len(q) > 0 {
		s.chain = append(s.chain, q)
		if len(q) == 1 {
			break
		}
		r := fzMgPremNeg(s.chain[len(s.chain)-2], q)
		q = fzMgTrimI(r)
	}
	return s
}

func (s *fzMgSturm) variations(x float64) int {
	r := fzRat(x)
	v, last := 0, 0
	for _, p := range s.chain {
		sg := fzMgSignAt(p, r)
		if sg == 0 {
			continue
		}
		if last != 0 && sg != last {
			v++
		}
		last = sg
	}
	return v
}

type fzMgXRoot struct {
	lo, hi  float64 // adjacent floats bracketing the root (or an exact root lo = hi)
	odd     bool
	cluster bool
}

func fzMgMid(a, b float64) float64 {
	if a > 0 && b/a > 2 {
		return math.Sqrt(a) * math.Sqrt(b)
	}
	return a + (b-a)/2
}

// fzMgPositiveRoots isolates the roots of p in x > 0. zero reports p ≡ 0;
// ok is false when the roots lie outside the float64 range.
func fzMgPositiveRoots(p fzMgPoly) (roots []fzMgXRoot, zero, ok bool) {
	ip := fzMgToInt(p.trim())
	if len(ip) == 0 {
		return nil, true, true
	}
	for len(ip) > 0 && ip[0].Sign() == 0 {
		ip = ip[1:]
	}
	if len(ip) <= 1 {
		return nil, false, true
	}
	maxAbs := func(xs fzMgIPoly) *big.Float {
		m := new(big.Float)
		for _, x := range xs {
			if v := new(big.Float).SetInt(new(big.Int).Abs(x)); v.Cmp(m) > 0 {
				m = v
			}
		}
		return m
	}
	a0 := new(big.Float).SetInt(new(big.Int).Abs(ip[0]))
	an := new(big.Float).SetInt(new(big.Int).Abs(ip[len(ip)-1]))
	hiF := new(big.Float).Quo(maxAbs(ip[:len(ip)-1]), an)
	hiF.Add(hiF, big.NewFloat(1))
	loF := new(big.Float).Quo(a0, new(big.Float).Add(a0, maxAbs(ip[1:])))
	hi, _ := hiF.Float64()
	lo, _ := loF.Float64()
	hi, lo = 2*hi, lo/2
	if !(hi < 1e300) || !(lo > 1e-300) {
		return nil, false, false
	}
	st := fzMgNewSturm(ip)
	sign := func(x float64) int { return fzMgSignAt(ip, fzRat(x)) }
	var isolate func(a, b float64, va, vb int)
	isolate = func(a, b float64, va, vb int) {
		c := va - vb
		if c <= 0 {
			return
		}
		m := fzMgMid(a, b)
		if c > 1 && (m <= a || m >= b) {
			roots = append(roots, fzMgXRoot{lo: a, hi: b, cluster: true})
			return
		}
		if c > 1 {
			vm := st.variations(m)
			isolate(a, m, va, vm)
			isolate(m, b, vm, vb)
			return
		}
		sb := sign(b)
		if sb == 0 {
			l, r := math.Nextafter(b, 0), math.Nextafter(b, math.Inf(1))
			roots = append(roots, fzMgXRoot{lo: b, hi: b, odd: sign(l)*sign(r) < 0})
			return
		}
		sa := sign(a)
		if sa == 0 {
			sa = sign(math.Nextafter(a, math.Inf(1)))
		}
		if sa*sb < 0 {
			for {
				m := fzMgMid(a, b)
				if m <= a || m >= b {
					break
				}
				sm := sign(m)
				if sm == 0 {
					a, b = m, m
					break
				}
				if sm == sa {
					a = m
				} else {
					b = m
				}
			}
			roots = append(roots, fzMgXRoot{lo: a, hi: b, odd: true})
			return
		}
		for {
			m := fzMgMid(a, b)
			if m <= a || m >= b {
				break
			}
			vm := st.variations(m)
			if va-vm >= 1 {
				b, vb = m, vm
			} else {
				a, va = m, vm
			}
		}
		roots = append(roots, fzMgXRoot{lo: a, hi: b})
	}
	isolate(lo, hi, st.variations(lo), st.variations(hi))
	return roots, false, true
}

// fzMgCross is a crossing on the frequency axis.
type fzMgCross struct {
	w, lo, hi float64
	odd       bool
	cluster   bool
	nyquist   bool
	required  bool
	tol       float64 // relative location uncertainty of a required crossing
}

func (l *fzMgLoop) toCross(r fzMgXRoot) fzMgCross {
	lo, hi := l.wOf(r.lo), l.wOf(r.hi)
	return fzMgCross{w: fzMgMid(lo, hi), lo: lo, hi: hi, odd: r.odd, cluster: r.cluster}
}

const (
	fzMgRel      = 1e-6 // straddle distance for a required crossing
	fzMgMatch    = 1e-5 // matching distance for reported crossings
	fzMgGainTol  = 1e-8 // ln|L| floor (library accepts 1e-9)
	fzMgPhaseTol = 1e-8 // phase floor in rad (library accepts 1e-7°)
)

// straddles reports whether f changes sign robustly across c: at ω(1 ± r)
// for the smallest r in 1e-6..1e-2 where robust holds at both ends, so an
// odd number of crossings lies within r for every model within rounding. It
// records r as the crossing's location uncertainty.
func (l *fzMgLoop) straddles(c *fzMgCross, f func(float64) float64, robust func(float64) bool) bool {
	if !c.odd || c.cluster {
		return false
	}
	for r := fzMgRel; r <= 1.01e-2; r *= 3 {
		wa, wb := c.lo*(1-r), c.hi*(1+r)
		if l.discrete && wb >= l.nyq() {
			return false
		}
		if !(f(wa)*f(wb) < 0) {
			return false
		}
		if robust(wa) && robust(wb) {
			c.tol = 2 * r
			return true
		}
	}
	return false
}

// gainRobust reports whether the sign of ln|L| − ln γ at ω is certain and
// exceeds floor.
func (l *fzMgLoop) gainRobust(w, gamma, floor float64) bool {
	g, _ := l.rsens(w)
	return math.Abs(l.lnMag(w)-math.Log(gamma)) > floor+g
}

// phaseRobust reports whether Re L < 0 and the sign of Im L at ω are
// certain and the phase offset exceeds the floor.
func (l *fzMgLoop) phaseRobust(w float64) bool {
	_, e := l.rsens(w)
	f := math.Abs(l.phaseOff(w))
	return f > fzMgPhaseTol+e && f < math.Pi/2-e
}

// finiteRobust reports whether L(jω) is finite and nonzero for every model
// within rounding, so a sign change of Im(N·conj D) there is Im L passing
// through 0, not a pole or zero on the boundary.
func (l *fzMgLoop) finiteRobust(w float64) bool {
	g, _ := l.rsens(w)
	return g < 1
}

// gainCross returns the crossings of |L| = gamma; zero reports |L| ≡ gamma.
func (l *fzMgLoop) gainCross(gamma float64) (cs []fzMgCross, zero, ok bool) {
	g2 := new(big.Rat).Mul(fzRat(gamma), fzRat(gamma))
	n2 := fzMgAdd(fzMgMul(l.N.re, l.N.re), fzMgMul(l.N.im, l.N.im), 1)
	d2 := fzMgAdd(fzMgMul(l.D.re, l.D.re), fzMgMul(l.D.im, l.D.im), 1)
	roots, zero, ok := fzMgPositiveRoots(fzMgAdd(n2, fzMgScale(d2, g2), -1))
	if zero || !ok {
		return nil, zero, ok
	}
	lg := math.Log(gamma)
	f := func(w float64) float64 { return l.lnMag(w) - lg }
	for _, r := range roots {
		c := l.toCross(r)
		c.required = l.straddles(&c, f, func(w float64) bool { return l.gainRobust(w, gamma, fzMgGainTol) })
		cs = append(cs, c)
	}
	return cs, false, true
}

// phaseCross returns the −180° crossings (delay-free or discrete loops);
// zero reports Im L ≡ 0.
func (l *fzMgLoop) phaseCross() (cs []fzMgCross, zero, ok bool) {
	h := fzMgAdd(fzMgMul(l.N.im, l.D.re), fzMgMul(l.N.re, l.D.im), -1)
	roots, zero, ok := fzMgPositiveRoots(h)
	if zero || !ok {
		return nil, zero, ok
	}
	all := make([]fzMgCross, len(roots))
	for i, r := range roots {
		all[i] = l.toCross(r)
	}
	for i, c := range all {
		if l.discrete && c.w >= l.nyq()*(1-1e-15) {
			continue
		}
		re, _ := l.reIm(c.w)
		neg := re.Sign() < 0
		if !neg && l.budget(c.w) < 1e-3 {
			continue
		}
		// A sign change of the phase offset is a crossing, not a ±180° wrap,
		// only if Re L < 0 is certain at the lone root of Im L in the bracket.
		g, p := l.rsens(c.w)
		c.required = neg && g < 1 && p < 0.1 && l.straddles(&c, l.phaseOff, l.phaseRobust)
		for j, o := range all {
			if c.required && j != i && o.hi >= c.lo*(1-c.tol/2) && o.lo <= c.hi*(1+c.tol/2) {
				c.required = false
			}
		}
		cs = append(cs, c)
	}
	if l.discrete {
		w := l.nyq()
		if re, _ := l.reIm(w); re.Sign() < 0 || l.budget(w) >= 0.5 {
			cs = append(cs, fzMgCross{w: w, lo: w, hi: w, odd: true, nyquist: true,
				required: re.Sign() < 0 && l.finiteRobust(w) && l.nyqCoeffRobust() && l.phaseRobust(w*(1-1e-3))})
		}
	}
	return cs, false, true
}

// matches reports whether reported w lies on the exact crossing c.
func fzMgMatches(c fzMgCross, w float64) bool {
	tol := max(c.tol, fzMgMatch)
	return w >= c.lo*(1-tol) && w <= c.hi*(1+tol)
}

func fzMgSorted(ws []float64) bool {
	return slices.IsSortedFunc(ws, func(a, b float64) int {
		if a < b {
			return -1
		}
		return 1
	})
}

func fzMgSumSq(x []float64) float64 {
	s := 0.0
	for _, v := range x {
		s += v * v
	}
	return s
}

// nyqCoeffRobust reports whether N(−1) and D(−1) exceed the rounding of
// their coefficients, as AllMargin requires of a Nyquist crossover (L(−1)
// above the noise of an ulp of the transfer-function coefficients).
func (l *fzMgLoop) nyqCoeffRobust() bool {
	ok := func(p fzMgPoly) bool {
		v, _ := new(big.Rat).Abs(fzMgEval(p, big.NewRat(-1, 1))).Float64()
		s := 0.0
		for _, x := range p {
			a, _ := new(big.Rat).Abs(x).Float64()
			s += a
		}
		return v > 4*float64(len(p))*0x1p-52*s
	}
	return ok(l.num) && ok(l.den)
}
