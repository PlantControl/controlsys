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
// every model within rounding: an odd, isolated root whose f (ln|L| − ln γ,
// or the phase offset from −180°) has opposite signs at ω0(1 ± 1e-6) with
// both |f| above budget = 1e3·n·eps·cond(T)²·(κ_N + κ_D) plus a floor, κ the
// coefficient condition number of each polynomial on the boundary. A
// reported crossing is sound when an exact root lies within 1e-5 relative
// or |f| there is within budget.

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
	discrete       bool
	dt, tau        float64
	n, deg         int
	num, den       fzMgPoly // ascending in s or z; den carries z^k
	numMag, denMag []float64
	N, D           fzMgCPoly // on the boundary, in x
	scale          float64   // lenient relative error of L, for soundness
	sreq           float64   // structured coefficient rounding, for requirement
}

// fzMgNewLoop builds the exact loop of the SISO realization (a n×n, b, c, d)
// with delay samples of discrete delay (dt > 0) or tau seconds of
// continuous delay. Coefficient rounding is componentwise in the
// coefficients (transfer-function data), or, when normwise, bounded by the
// coefficient derivatives under a perturbation of eps·‖A‖_F, eps·‖b‖‖c‖ of
// the realization.
func fzMgNewLoop(n int, a, b, c []float64, d, dt float64, delay int, tau, condT float64, normwise bool) *fzMgLoop {
	den := fzCharPolyExact(a, n)
	abc := make([]*big.Rat, n*n)
	for i := range n {
		for j := range n {
			abc[i*n+j] = new(big.Rat).Add(fzRat(a[i*n+j]), new(big.Rat).Mul(fzRat(b[i]), fzRat(c[j])))
		}
	}
	cp := fzCharPolyRat(abc, n)
	l := &fzMgLoop{discrete: dt > 0, dt: dt, tau: tau, n: n}
	rd := fzRat(d)
	normA := math.Sqrt(fzMgSumSq(a))
	normBC := math.Sqrt(fzMgSumSq(b) * fzMgSumSq(c))
	for k := n; k >= 0; k-- {
		strict := new(big.Rat).Sub(den[k], cp[k])
		dd := new(big.Rat).Mul(rd, den[k])
		l.num = append(l.num, new(big.Rat).Add(dd, strict))
		l.den = append(l.den, new(big.Rat).Set(den[k]))
		ms, _ := new(big.Rat).Abs(strict).Float64()
		mden, _ := new(big.Rat).Abs(den[k]).Float64()
		if normwise && k > 0 {
			bk := 1.0
			for i := range k {
				bk = bk * float64(n-i) / float64(i+1)
			}
			mden = max(mden, bk*math.Pow(normA, float64(k)))
			ms = max(ms, float64(k)*bk*normBC*math.Pow(normA, float64(k-1)))
		}
		l.numMag = append(l.numMag, ms+math.Abs(d)*mden)
		l.denMag = append(l.denMag, mden)
	}
	if delay > 0 {
		l.den = append(fzMgZeros(delay), l.den...)
		l.denMag = append(make([]float64, delay), l.denMag...)
	}
	l.deg = len(l.den) - 1
	l.N, l.D = l.boundary(l.num), l.boundary(l.den)
	l.scale = 1e3 * float64(max(n, 1)) * 0x1p-52 * condT * condT
	l.sreq = float64(n+1) * 0x1p-52
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
	x := fzRat(l.xOf(w))
	return fzMgEval(l.N.re, x), fzMgEval(l.N.im, x), fzMgEval(l.D.re, x), fzMgEval(l.D.im, x)
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

// lnAbs is ln|P(boundary point)| from the x-domain values of P.
func (l *fzMgLoop) lnAbs(w float64, re, im *big.Rat) float64 {
	v := fzMgLn(fzMgAbs2(re, im)) / 2
	if l.discrete && !l.atNyq(w) {
		x := l.xOf(w)
		v -= float64(l.deg) / 2 * math.Log1p(x*x)
	}
	return v
}

func (l *fzMgLoop) lnMagSum(mag []float64, w float64) float64 {
	lw := 0.0
	if !l.discrete {
		lw = math.Log(w)
	}
	best := math.Inf(-1)
	terms := make([]float64, 0, len(mag))
	for k, m := range mag {
		if m > 0 {
			v := math.Log(m) + float64(k)*lw
			terms = append(terms, v)
			best = math.Max(best, v)
		}
	}
	if math.IsInf(best, -1) {
		return best
	}
	s := 0.0
	for _, v := range terms {
		s += math.Exp(v - best)
	}
	return best + math.Log(s)
}

// budget bounds the relative error of L(jω) for any realization within
// rounding of the float64 one.
func (l *fzMgLoop) budget(w float64) float64 {
	nr, ni, dr, di := l.values(w)
	kn := math.Exp(l.lnMagSum(l.numMag, w) - l.lnAbs(w, nr, ni))
	kd := math.Exp(l.lnMagSum(l.denMag, w) - l.lnAbs(w, dr, di))
	if math.IsNaN(kn) {
		kn = 0
	}
	return l.scale * (kn + kd)
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

// cvals returns N, D and the coefficient basis Z_k (N = Σ n_k·Z_k) at ω on
// the boundary, rounded to complex128.
func (l *fzMgLoop) cvals(w float64) (n, d complex128, z []complex128) {
	nr, ni, dr, di := l.values(w)
	f := func(x *big.Rat) float64 { v, _ := x.Float64(); return v }
	n, d = complex(f(nr), f(ni)), complex(f(dr), f(di))
	z = make([]complex128, l.deg+1)
	switch {
	case l.atNyq(w):
		for k := range z {
			z[k] = complex(math.Pow(-1, float64(k)), 0)
		}
	case l.discrete:
		x := l.xOf(w)
		for k := range z {
			z[k] = cmplx.Pow(complex(1, x), complex(float64(k), 0)) * cmplx.Pow(complex(1, -x), complex(float64(l.deg-k), 0))
		}
	default:
		for k := range z {
			z[k] = cmplx.Pow(complex(0, w), complex(float64(k), 0))
		}
	}
	return n, d, z
}

// sigma returns componentwise first-order bounds on the change of
// Re(N·conj D), Im(N·conj D) and |N|² − γ²|D|² per unit relative change of
// every coefficient of N and D.
func (l *fzMgLoop) sigma(w, gamma float64) (sr, si, sg float64) {
	n, d, z := l.cvals(w)
	for k, zk := range z {
		var mn, md float64
		if k < len(l.numMag) {
			mn = l.numMag[k]
		}
		if k < len(l.denMag) {
			md = l.denMag[k]
		}
		a, b := zk*cmplx.Conj(d), n*cmplx.Conj(zk)
		sr += mn*math.Abs(real(a)) + md*math.Abs(real(b))
		si += mn*math.Abs(imag(a)) + md*math.Abs(imag(b))
		sg += 2 * (mn*math.Abs(real(n*cmplx.Conj(zk))) + gamma*gamma*md*math.Abs(real(d*cmplx.Conj(zk))))
	}
	return sr, si, sg
}

// gainRobust reports whether |L(jω)| − γ keeps its sign under coefficient
// rounding and |ln|L| − ln γ| exceeds floor.
func (l *fzMgLoop) gainRobust(w, gamma, floor float64) bool {
	if !(math.Abs(l.lnMag(w)-math.Log(gamma)) > floor) {
		return false
	}
	nr, ni, dr, di := l.values(w)
	g := new(big.Rat).Sub(fzMgAbs2(nr, ni), new(big.Rat).Mul(fzMgAbs2(dr, di), new(big.Rat).Mul(fzRat(gamma), fzRat(gamma))))
	gv, _ := g.Float64()
	_, _, sg := l.sigma(w, gamma)
	return math.Abs(gv) > l.sreq*sg
}

// phaseRobust reports whether Re L < 0 and the sign of Im L survive
// coefficient rounding and the phase offset exceeds the floor.
func (l *fzMgLoop) phaseRobust(w float64) bool {
	if !(math.Abs(l.phaseOff(w)) > fzMgPhaseTol) {
		return false
	}
	re, im := l.reIm(w)
	rv, _ := re.Float64()
	iv, _ := im.Float64()
	sr, si, _ := l.sigma(w, 1)
	return rv < -l.sreq*sr && math.Abs(iv) > l.sreq*si
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
	for _, r := range roots {
		c := l.toCross(r)
		if l.discrete && c.w >= l.nyq()*(1-1e-15) {
			continue
		}
		re, _ := l.reIm(c.w)
		neg := re.Sign() < 0
		if !neg && l.budget(c.w) < 1e-3 {
			continue
		}
		c.required = neg && l.straddles(&c, l.phaseOff, l.phaseRobust)
		cs = append(cs, c)
	}
	if l.discrete {
		w := l.nyq()
		re, _ := l.reIm(w)
		rv, _ := re.Float64()
		sr, _, _ := l.sigma(w, 1)
		if re.Sign() < 0 || l.budget(w) >= 0.5 {
			cs = append(cs, fzMgCross{w: w, lo: w, hi: w, odd: true, nyquist: true,
				required: rv < -l.sreq*sr})
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
