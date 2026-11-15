package controlsys

import (
	"math"
	"math/big"
	"math/bits"
	"slices"
)

// marginCountMaxOrder caps the order whose gain crossings exactCrossings
// counts on the stored model's exact crossing polynomial (a cost cap).
const marginCountMaxOrder = 24

// ipoly is an integer polynomial, ascending coefficients.
type ipoly []*big.Int

func newIpoly(n int) ipoly {
	p := make(ipoly, n)
	slab := make([]big.Int, n)
	for i := range p {
		p[i] = &slab[i]
	}
	return p
}

func (p ipoly) trim() ipoly {
	for len(p) > 0 && p[len(p)-1].Sign() == 0 {
		p = p[:len(p)-1]
	}
	return p
}

func (p ipoly) clone() ipoly {
	q := make(ipoly, len(p))
	for i, v := range p {
		q[i] = new(big.Int).Set(v)
	}
	return q
}

func ipolyMul(a, b ipoly) ipoly {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	o := newIpoly(len(a) + len(b) - 1)
	t := new(big.Int)
	for i, x := range a {
		if x.Sign() == 0 {
			continue
		}
		for j, y := range b {
			if y.Sign() != 0 {
				o[i+j].Add(o[i+j], t.Mul(x, y))
			}
		}
	}
	return o.trim()
}

// ipolyAxpy is a·x + b·y.
func ipolyAxpy(a *big.Int, x ipoly, b *big.Int, y ipoly) ipoly {
	o := newIpoly(max(len(x), len(y)))
	t := new(big.Int)
	for i := range o {
		if i < len(x) {
			o[i].Add(o[i], t.Mul(a, x[i]))
		}
		if i < len(y) {
			o[i].Add(o[i], t.Mul(b, y[i]))
		}
	}
	return o.trim()
}

func ipolyDeriv(p ipoly) ipoly {
	if len(p) <= 1 {
		return nil
	}
	o := newIpoly(len(p) - 1)
	for i := 1; i < len(p); i++ {
		o[i-1].Mul(p[i], big.NewInt(int64(i)))
	}
	return o.trim()
}

func ipolyContent(p ipoly) *big.Int {
	g := new(big.Int)
	for _, v := range p {
		g.GCD(nil, nil, g, new(big.Int).Abs(v))
	}
	return g
}

func ipolyPrimitive(p ipoly) ipoly {
	g := ipolyContent(p)
	if g.Sign() == 0 || g.Cmp(big.NewInt(1)) == 0 {
		return p
	}
	o := make(ipoly, len(p))
	for i, v := range p {
		o[i] = new(big.Int).Quo(v, g)
	}
	return o
}

// ipolyPseudoRem is the pseudo-remainder of a by b.
func ipolyPseudoRem(a, b ipoly) ipoly {
	r := a.clone()
	lb := b[len(b)-1]
	t := new(big.Int)
	for len(r) >= len(b) && len(r) > 0 {
		lr := new(big.Int).Set(r[len(r)-1])
		sh := len(r) - len(b)
		for i := range r {
			r[i].Mul(r[i], lb)
		}
		for i := range b {
			r[sh+i].Sub(r[sh+i], t.Mul(lr, b[i]))
		}
		r = r[:len(r)-1].trim()
	}
	return r
}

// ipolyGCD is the primitive gcd of a and b (primitive PRS).
func ipolyGCD(a, b ipoly) ipoly {
	a, b = ipolyPrimitive(a.trim()), ipolyPrimitive(b.trim())
	if len(a) < len(b) {
		a, b = b, a
	}
	for len(b) > 0 {
		r := ipolyPrimitive(ipolyPseudoRem(a, b))
		a, b = b, r
	}
	return a
}

// ipolyExactQuo divides a by b, which divides it exactly over the rationals;
// the quotient is scaled to a primitive integer polynomial.
func ipolyExactQuo(a, b ipoly) ipoly {
	n := len(a) - len(b) + 1
	if n <= 0 {
		return nil
	}
	num := make([]*big.Rat, len(a))
	for i, v := range a {
		num[i] = new(big.Rat).SetInt(v)
	}
	lb := new(big.Rat).SetInt(b[len(b)-1])
	q := make([]*big.Rat, n)
	t := new(big.Rat)
	for k := n - 1; k >= 0; k-- {
		q[k] = new(big.Rat).Quo(num[k+len(b)-1], lb)
		for i := range b {
			num[k+i].Sub(num[k+i], t.Mul(q[k], new(big.Rat).SetInt(b[i])))
		}
	}
	den := big.NewInt(1)
	for _, v := range q {
		d := v.Denom()
		g := new(big.Int).GCD(nil, nil, den, d)
		den.Mul(den, new(big.Int).Quo(d, g))
	}
	o := newIpoly(n)
	for i, v := range q {
		o[i].Mul(v.Num(), new(big.Int).Quo(den, v.Denom()))
	}
	return ipolyPrimitive(o.trim())
}

// signAt is the sign of p at num/2^q.
func (p ipoly) signAt(num *big.Int, q int) int {
	if len(p) == 0 {
		return 0
	}
	d := len(p) - 1
	acc := new(big.Int)
	t := new(big.Int)
	for i := d; i >= 0; i-- {
		acc.Mul(acc, num)
		acc.Add(acc, t.Lsh(p[i], uint(q*(d-i))))
	}
	return acc.Sign()
}

func signVariations(p ipoly) int {
	v, prev := 0, 0
	for _, c := range p {
		s := c.Sign()
		if s == 0 {
			continue
		}
		if prev != 0 && s != prev {
			v++
		}
		prev = s
	}
	return v
}

// taylorShift1 is p(y+1).
func taylorShift1(p ipoly) ipoly {
	q := p.clone()
	n := len(q)
	for i := range n {
		for j := n - 2; j >= i; j-- {
			q[j].Add(q[j], q[j+1])
		}
	}
	return q
}

func ipolyReverse(p ipoly) ipoly {
	q := make(ipoly, len(p))
	for i, v := range p {
		q[len(p)-1-i] = v
	}
	return q
}

// rootInterval is a root of a squarefree polynomial in (lo, hi)/2^q (open),
// or exactly lo/2^q when exact.
type rootInterval struct {
	lo, hi *big.Int
	q      int
	exact  bool
}

// isolatePositive isolates the positive roots of the squarefree p by
// Descartes' rule of signs on Möbius-transformed bisections (VCA). ok is
// false where the recursion does not separate the roots within depth.
func isolatePositive(p ipoly) ([]rootInterval, bool) {
	p = p.trim()
	for len(p) > 0 && p[0].Sign() == 0 {
		p = p[1:]
	}
	d := len(p) - 1
	if d <= 0 {
		return nil, true
	}
	if signVariations(p) == 0 {
		return nil, true
	}
	lead := p[d].BitLen()
	bnd := 0
	for k := 1; k <= d; k++ {
		if p[d-k].Sign() == 0 {
			continue
		}
		bnd = max(bnd, (p[d-k].BitLen()-lead+k)/k+1)
	}
	bnd++
	q := newIpoly(d + 1)
	for i, v := range p {
		q[i].Lsh(v, uint(bnd*i))
	}
	var out []rootInterval
	type job struct {
		q     ipoly
		depth int
		k     *big.Int
	}
	stack := []job{{q, 0, new(big.Int)}}
	for len(stack) > 0 {
		j := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		v := signVariations(taylorShift1(ipolyReverse(j.q)))
		if v == 0 {
			continue
		}
		den := j.depth - bnd
		if v == 1 {
			lo := new(big.Int).Set(j.k)
			hi := new(big.Int).Add(j.k, big.NewInt(1))
			out = append(out, rootInterval{lo: lo, hi: hi, q: den})
			continue
		}
		if j.depth > 400 {
			return nil, false
		}
		left := newIpoly(d + 1)
		for i, c := range j.q {
			left[i].Lsh(c, uint(d-i))
		}
		k2 := new(big.Int).Lsh(j.k, 1)
		if left.signAt(big.NewInt(1), 0) == 0 {
			out = append(out, rootInterval{lo: new(big.Int).Add(k2, big.NewInt(1)), q: den + 1, exact: true})
		}
		right := taylorShift1(left)
		stack = append(stack, job{left, j.depth + 1, k2}, job{right, j.depth + 1, new(big.Int).Add(k2, big.NewInt(1))})
	}
	for i := range out {
		if out[i].q < 0 {
			s := uint(-out[i].q)
			out[i].lo.Lsh(out[i].lo, s)
			if out[i].hi != nil {
				out[i].hi.Lsh(out[i].hi, s)
			}
			out[i].q = 0
		}
	}
	return out, true
}

// floatScale is the least k ≥ 0 with v·2^k an integer for every v.
func floatScale(vs ...[]float64) int {
	k := 0
	for _, s := range vs {
		for _, v := range s {
			if v == 0 {
				continue
			}
			m, e := math.Frexp(math.Abs(v))
			mi := uint64(m * (1 << 53))
			tz := 0
			for mi&1 == 0 {
				mi >>= 1
				tz++
			}
			k = max(k, 53-e-tz)
		}
	}
	return k
}

// scaledInt is v·2^k, an integer by floatScale's choice of k.
func scaledInt(v float64, k int) *big.Int {
	if v == 0 {
		return new(big.Int)
	}
	m, e := math.Frexp(math.Abs(v))
	z := new(big.Int).SetUint64(uint64(m * (1 << 53)))
	if sh := e - 53 + k; sh >= 0 {
		z.Lsh(z, uint(sh))
	} else {
		z.Rsh(z, uint(-sh))
	}
	if v < 0 {
		z.Neg(z)
	}
	return z
}

// charpolyInt is det(tI − M) for the integer matrix M by Berkowitz's
// division-free algorithm, ascending coefficients.
func charpolyInt(n int, M []*big.Int) ipoly {
	_, full := charpolyLeading(n, M)
	return full
}

// charpolyLeading is det(tI − M) and det(tI − M₁₁) of its leading block of
// order n−1, from one Berkowitz pass.
func charpolyLeading(n int, M []*big.Int) (lead, full ipoly) {
	asc := func(c []*big.Int) ipoly {
		o := make(ipoly, len(c))
		for i := range o {
			o[i] = c[len(c)-1-i]
		}
		return o
	}
	c := []*big.Int{big.NewInt(1), new(big.Int).Neg(M[0])}
	if n == 1 {
		return ipoly{big.NewInt(1)}, asc(c)
	}
	t := new(big.Int)
	for r := 1; r < n; r++ {
		if r == n-1 {
			lead = asc(c)
		}
		tcol := berkowitzColumn(n, r, M)
		nc := make([]*big.Int, r+2)
		for i := range nc {
			nc[i] = new(big.Int)
			for j := 0; j <= min(i, len(c)-1); j++ {
				nc[i].Add(nc[i], t.Mul(tcol[i-j], c[j]))
			}
		}
		c = nc
	}
	return lead, asc(c)
}

// berkowitzColumn is the first column [1, −a, −R·S, −R·M·S, …] of order
// r+1 of Berkowitz's Toeplitz factor for the leading (r+1)×(r+1) block.
func berkowitzColumn(n, r int, M []*big.Int) []*big.Int {
	out := newIpoly(r + 2)
	out[0].SetInt64(1)
	out[1].Neg(M[r*n+r])
	t := new(big.Int)
	v, nv := newIpoly(r), newIpoly(r)
	for i := range r {
		v[i].Set(M[i*n+r])
	}
	for k := 2; k <= r+1; k++ {
		s := out[k]
		row := M[r*n : r*n+r]
		for i, m := range row {
			if m.Sign() != 0 && v[i].Sign() != 0 {
				s.Add(s, t.Mul(m, v[i]))
			}
		}
		s.Neg(s)
		if k == r+1 {
			break
		}
		for i := range r {
			acc := nv[i].SetInt64(0)
			mi := M[i*n : i*n+r]
			for l, m := range mi {
				if m.Sign() != 0 && v[l].Sign() != 0 {
					acc.Add(acc, t.Mul(m, v[l]))
				}
			}
		}
		v, nv = nv, v
	}
	return out
}

// absSqIm is |p(jw)|² as a polynomial in X = w².
func absSqIm(p ipoly) ipoly {
	var re, im ipoly
	for i, v := range p {
		s := new(big.Int).Set(v)
		if (i/2)%2 == 1 {
			s.Neg(s)
		}
		h := i / 2
		if i%2 == 0 {
			for len(re) <= h {
				re = append(re, new(big.Int))
			}
			re[h] = s
		} else {
			for len(im) <= h {
				im = append(im, new(big.Int))
			}
			im[h] = s
		}
	}
	o := ipolyMul(re, re)
	x := ipolyMul(im, im)
	if len(x) > 0 {
		x = append(ipoly{new(big.Int)}, x...)
	}
	one := big.NewInt(1)
	return ipolyAxpy(one, o, one, x)
}

// ipolyBilinear is Σ p_i (1+v)^i (1−v)^{n−i}: p(z)·(1−v)^n at z = (1+v)/(1−v).
func ipolyBilinear(p ipoly, n int) ipoly {
	plus := []ipoly{{big.NewInt(1)}}
	minus := []ipoly{{big.NewInt(1)}}
	for i := 1; i <= n; i++ {
		plus = append(plus, ipolyMul(plus[i-1], ipoly{big.NewInt(1), big.NewInt(1)}))
		minus = append(minus, ipolyMul(minus[i-1], ipoly{big.NewInt(1), big.NewInt(-1)}))
	}
	var o ipoly
	one := big.NewInt(1)
	for i, c := range p {
		if c.Sign() == 0 {
			continue
		}
		o = ipolyAxpy(one, o, c, ipolyMul(plus[i], minus[n-i]))
	}
	return o
}

// crossingCounter holds the exact crossing polynomial of an explicit SISO
// loop at level γ: P(X) = 2^{2kg}|N|² − g²|D|² with γ = g/2^kg, in X = τ²,
// τ = 2^K·ω (continuous) or X = w², w = tan(ωΔt/2) (discrete), whose
// positive roots of odd multiplicity are the gain crossings.
type crossingCounter struct {
	p  ipoly // empty where |L| ≡ γ: no isolated crossing
	k  int
	dt float64
}

func newCrossingCounter(n int, a, b, c []float64, d, dt, gamma float64) (*crossingCounter, bool) {
	if n == 0 || n > marginCountMaxOrder || !(gamma > 0) || math.IsInf(gamma, 0) {
		return nil, false
	}
	kb, kc, kd, kg := floatScale(b), floatScale(c), floatScale([]float64{d}), floatScale([]float64{gamma})
	K := floatScale(a)
	if K > 4000 || kb > 4000 || kc > 4000 || kd > 4000 || kg > 4000 {
		return nil, false
	}
	N1 := n + 1
	M := make([]*big.Int, N1*N1)
	for i := range n {
		for j := range n {
			M[i*N1+j] = scaledInt(a[i*n+j], K)
		}
		M[i*N1+n] = scaledInt(b[i], kb)
		M[n*N1+i] = scaledInt(c[i], kc)
	}
	M[n*N1+n] = new(big.Int)
	q1, qf := charpolyLeading(N1, M)
	R := ipolyAxpy(big.NewInt(1), append(ipoly{new(big.Int)}, q1...), big.NewInt(-1), qf)
	if dt > 0 {
		for i := range q1 {
			q1[i].Lsh(q1[i], uint(K*i))
		}
		for i := range R {
			R[i].Lsh(R[i], uint(K*i))
		}
	}
	p2kd := new(big.Int).Lsh(big.NewInt(1), uint(kd))
	dh := scaledInt(d, kd)
	if e := K - kb - kc; e >= 0 {
		R = ipolyAxpy(new(big.Int).Lsh(p2kd, uint(e)), R, new(big.Int), nil)
	} else {
		sh := new(big.Int).Lsh(big.NewInt(1), uint(-e))
		dh.Mul(dh, sh)
		p2kd.Mul(p2kd, sh)
		R = ipolyAxpy(new(big.Int).Lsh(big.NewInt(1), uint(kd)), R, new(big.Int), nil)
	}
	num := ipolyAxpy(big.NewInt(1), R, dh, q1)
	den := ipolyAxpy(p2kd, q1, new(big.Int), nil)
	if dt > 0 {
		num, den = ipolyBilinear(num, n), ipolyBilinear(den, n)
	}
	g := scaledInt(gamma, kg)
	P := ipolyAxpy(new(big.Int).Lsh(big.NewInt(1), uint(2*kg)), absSqIm(num), new(big.Int).Neg(new(big.Int).Mul(g, g)), absSqIm(den))
	return &crossingCounter{p: P, k: K, dt: dt}, true
}

// point is X at ω as num/2^q.
func (cc *crossingCounter) point(w float64) (*big.Int, int) {
	var x float64
	e := 0
	if cc.dt > 0 {
		x = math.Tan(w * cc.dt / 2)
	} else {
		x, e = w, cc.k
	}
	m, ex := math.Frexp(x)
	mi := new(big.Int).SetUint64(uint64(m * (1 << 53)))
	ex += e - 53
	mi.Mul(mi, mi)
	ex *= 2
	if ex >= 0 {
		return mi.Lsh(mi, uint(ex)), 0
	}
	return mi, -ex
}

func (cc *crossingCounter) signAtOmega(w float64) int {
	n, q := cc.point(w)
	return cc.p.signAt(n, q)
}

// omegaOf is the float ω whose X is nearest num/2^q.
func (cc *crossingCounter) omegaOf(num *big.Int, q int) float64 {
	f := new(big.Float).SetPrec(200).SetInt(num)
	f.SetMantExp(f, -q)
	f.Sqrt(f)
	x, _ := f.Float64()
	if cc.dt > 0 {
		return 2 * math.Atan(x) / cc.dt
	}
	return math.Ldexp(x, -cc.k)
}

// oddMultiplicityAt reports whether the root of P at num/2^q has odd
// multiplicity.
func (cc *crossingCounter) oddMultiplicityAt(num *big.Int, q int) bool {
	p := cc.p
	m := 0
	for len(p) > 0 && p.signAt(num, q) == 0 {
		p = ipolyDeriv(p)
		m++
	}
	return m%2 == 1
}

// crossings are the gain crossings in (0, wTop): the float ω at which the
// exact sign of P changes from its float predecessor; ok is false where two
// crossings fall between adjacent float64 ω, a crossing is subnormal, P
// vanishes identically, or the isolation does not converge.
func (cc *crossingCounter) crossings(wTop float64) (ws []float64, ok bool) {
	P := cc.p
	S := P
	if dp := ipolyDeriv(P); len(dp) > 0 && !squarefreeModPrime(P) {
		if g := ipolyGCD(P, dp); len(g) > 1 {
			S = ipolyExactQuo(P, g)
		}
	}
	roots, ok := isolatePositive(S)
	if !ok {
		return nil, false
	}
	for _, r := range roots {
		var lo, hi float64
		if r.exact {
			if !cc.oddMultiplicityAt(r.lo, r.q) {
				continue
			}
			w := cc.omegaOf(r.lo, r.q)
			lo, hi = w, w
		} else {
			sl, sh := P.signAt(r.lo, r.q), P.signAt(r.hi, r.q)
			if sl == sh {
				continue
			}
			lo, hi = cc.omegaOf(r.lo, r.q), cc.omegaOf(r.hi, r.q)
		}
		if !(lo < wTop) {
			continue
		}
		w, ok := cc.refine(lo, hi)
		if !ok {
			return nil, false
		}
		if w < wTop {
			ws = append(ws, w)
		}
	}
	slices.Sort(ws)
	return ws, true
}

// refine finds adjacent floats about [lo, hi] across which the exact sign
// changes, returning the upper one.
func (cc *crossingCounter) refine(lo, hi float64) (float64, bool) {
	hi = min(hi, math.MaxFloat64)
	if lo < 0x1p-1022 {
		sh := cc.signAtOmega(hi)
		for lo = hi / 2; lo >= 0x1p-1022 && cc.signAtOmega(lo) == sh; lo /= 2 {
		}
		if lo < 0x1p-1022 {
			return 0, false
		}
	}
	sl := cc.signAtOmega(lo)
	for range 64 {
		if sl != 0 && sl != cc.signAtOmega(hi) {
			break
		}
		lo = math.Nextafter(lo, 0)
		hi = math.Nextafter(hi, math.Inf(1))
		sl = cc.signAtOmega(lo)
	}
	sh := cc.signAtOmega(hi)
	if sl == 0 || sh == 0 || sl == sh {
		return 0, false
	}
	for {
		m := lo + (hi-lo)/2
		if hi > 2*lo {
			m = math.Sqrt(lo) * math.Sqrt(hi)
		}
		if m <= lo || m >= hi {
			break
		}
		if s := cc.signAtOmega(m); s == 0 {
			return m, true
		} else if s == sl {
			lo = m
		} else {
			hi = m
		}
	}
	return hi, true
}

// exactCrossings reconciles the gain crossings ws in (0, wTop) with the
// stored model's exact crossing polynomial: ws where Descartes' bound on
// its positive roots equals the crossings ws exhibits by exact sign
// changes, or than a root-free proof on every gap between them, else the
// exact crossings, keeping each ws value within 1e-9 of one. Crossings
// beyond those shown come in pairs where the shown ones alternate the sign
// of P from 0⁺ to ∞, so P of degree below shown+2 has no room for more. ok
// is false outside the polynomial's scope (explicit delay-free SISO loops
// whose structural core is of order up to marginCountMaxOrder) or where the
// exact crossings are not resolved in float64.
func (r *rationalLoop) exactCrossings(ws []float64, wTop float64) ([]float64, bool) {
	if r.exactFailed == r.gamma && r.gamma > 0 {
		return nil, false
	}
	out, ok := r.reconcileExact(ws, wTop)
	if !ok {
		r.exactFailed = r.gamma
	}
	return out, ok
}

func (r *rationalLoop) reconcileExact(ws []float64, wTop float64) ([]float64, bool) {
	cc, ok := r.crossingCounter()
	if !ok {
		return nil, false
	}
	if len(cc.p) == 0 {
		return ws, true
	}
	p := cc.p
	for len(p) > 0 && p[0].Sign() == 0 {
		p = p[1:]
	}
	shown := 0
	inside := 0
	prevHi := 0.0
	var ends []float64
	side := p[0].Sign()
	chained := true
	for _, w := range ws {
		if !(w < wTop) {
			continue
		}
		inside++
		lo, hi := w*(1-1e-9), w*(1+1e-9)
		if lo <= prevHi {
			break
		}
		prevHi = hi
		if sl, sh := cc.signAtOmega(lo), cc.signAtOmega(hi); sl != 0 && sh != 0 && sl != sh {
			shown++
			ends = append(ends, lo, hi)
			chained = chained && sl == side
			side = sh
		}
	}
	chained = chained && side == p[len(p)-1].Sign()
	if r.firstCrossing && len(ws) > 0 && ws[0] < wTop {
		if lo, hi := ws[0]*(1-1e-9), ws[0]*(1+1e-9); cc.firstIsolated(p, lo, hi) {
			return ws, true
		}
	}
	if shown == inside && chained && shown+2 > len(p)-1 {
		return ws, true
	}
	if shown == inside && (signVariations(p) == shown || cc.gapsEmpty(p, ends)) {
		return ws, true
	}
	exact, ok := cc.crossings(wTop)
	if !ok {
		return nil, false
	}
	out := make([]float64, 0, len(exact)+1)
	for _, c := range exact {
		v := c
		for _, w := range ws {
			if math.Abs(w-c) <= 1e-9*c {
				v = w
				break
			}
		}
		out = append(out, v)
	}
	for _, w := range ws {
		if !(w < wTop) {
			out = append(out, w)
		}
	}
	return out, true
}

// crossingCounter is the exact crossing polynomial of the stored loop at
// r.gamma, formed once per level.
func (r *rationalLoop) crossingCounter() (*crossingCounter, bool) {
	sys := r.exactCore()
	if sys == nil || !(r.gamma > 0) {
		return nil, false
	}
	n, _, _ := sys.Dims()
	if n > marginCountMaxOrder {
		return nil, false
	}
	if r.counter != nil && r.counterGamma == r.gamma {
		return r.counter, true
	}
	ar, br, cr := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	a, b, c := make([]float64, n*n), make([]float64, n), make([]float64, n)
	for i := range n {
		copy(a[i*n:(i+1)*n], ar.Data[i*ar.Stride:i*ar.Stride+n])
		b[i], c[i] = br.Data[i*br.Stride], cr.Data[i]
	}
	cc, ok := newCrossingCounter(n, a, b, c, sys.D.At(0, 0), sys.Dt, r.gamma)
	if !ok {
		return nil, false
	}
	r.counter, r.counterGamma = cc, r.gamma
	return cc, true
}

// structuralCore is the SISO explicit sys restricted to the states that
// the input reaches and that reach the output through the exact nonzero
// pattern of A: the other states enter the transfer function through
// exactly zero blocks only, so it is the same exactly. It is sys itself
// where every state is kept, and nil where none is.
func structuralCore(sys *System) *System {
	n, _, _ := sys.Dims()
	ar, br, cr := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	reach, obs := make([]bool, n), make([]bool, n)
	var stack []int
	for i := range n {
		if br.Data[i*br.Stride] != 0 {
			reach[i] = true
			stack = append(stack, i)
		}
	}
	for len(stack) > 0 {
		j := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for i := range n {
			if !reach[i] && ar.Data[i*ar.Stride+j] != 0 {
				reach[i] = true
				stack = append(stack, i)
			}
		}
	}
	for j := range n {
		if cr.Data[j] != 0 {
			obs[j] = true
			stack = append(stack, j)
		}
	}
	for len(stack) > 0 {
		i := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		for j := range n {
			if !obs[j] && ar.Data[i*ar.Stride+j] != 0 {
				obs[j] = true
				stack = append(stack, j)
			}
		}
	}
	var keep []int
	for i := range n {
		if reach[i] && obs[i] {
			keep = append(keep, i)
		}
	}
	if len(keep) == n {
		return sys
	}
	k := len(keep)
	if k == 0 {
		return nil
	}
	a, b, c := make([]float64, k*k), make([]float64, k), make([]float64, k)
	for ii, i := range keep {
		for jj, j := range keep {
			a[ii*k+jj] = ar.Data[i*ar.Stride+j]
		}
		b[ii], c[ii] = br.Data[i*br.Stride], cr.Data[i]
	}
	core, err := NewFromSlices(k, 1, 1, a, b, c, []float64{sys.D.At(0, 0)}, sys.Dt)
	if err != nil {
		return nil
	}
	return core
}

// exactCore is structuralCore of the stored loop for the exact paths
// (exact responses, the exact DC limit, the exact crossing count), formed
// once; nil outside their scope (MIMO, descriptor, internal delays) or
// where no state is kept.
func (r *rationalLoop) exactCore() *System {
	if !r.coreDone {
		r.coreDone = true
		sys := r.eval.sys
		if _, m, p := sys.Dims(); m == 1 && p == 1 && sys.internalDelayCount() == 0 && !newDescriptorPolicy(sys).isDescriptor() {
			r.core = structuralCore(sys)
		}
	}
	return r.core
}

// exactCoreOr is exactCore, or sys where it has none (MIMO, no kept state).
func (r *rationalLoop) exactCoreOr(sys *System) *System {
	if c := r.exactCore(); c != nil {
		return c
	}
	return sys
}

func (sys *System) n() int {
	n, _, _ := sys.Dims()
	return n
}

// taylorShiftBy is p(y + a).
func taylorShiftBy(p ipoly, a *big.Int) ipoly {
	q := p.clone()
	n := len(q)
	t := new(big.Int)
	for i := range n {
		for j := n - 2; j >= i; j-- {
			q[j].Add(q[j], t.Mul(a, q[j+1]))
		}
	}
	return q
}

// descartesIn bounds the roots of p in (a, b) (b nil: (a, ∞)), a and b as
// num/2^q, by Descartes' rule on the Möbius image.
func descartesIn(p ipoly, aNum *big.Int, aQ int, bNum *big.Int, bQ int) int {
	q := aQ
	if bNum != nil {
		q = max(q, bQ)
	}
	A := new(big.Int).Lsh(aNum, uint(q-aQ))
	d := len(p) - 1
	r := newIpoly(d + 1)
	for i, c := range p {
		r[i].Lsh(c, uint(q*(d-i)))
	}
	sh := taylorShiftBy(r, A)
	if bNum == nil {
		return signVariations(sh)
	}
	W := new(big.Int).Sub(new(big.Int).Lsh(bNum, uint(q-bQ)), A)
	pw := big.NewInt(1)
	for i := range sh {
		sh[i].Mul(sh[i], pw)
		pw = new(big.Int).Mul(pw, W)
	}
	return signVariations(taylorShift1(ipolyReverse(sh)))
}

// gapsEmpty reports whether Descartes' rule proves p root-free on each gap
// between the disjoint brackets ends (lo, hi pairs, ascending ω), from 0 to
// the first and from the last to ∞ (Nyquist when discrete), with exactly
// one root in each bracket.
func (cc *crossingCounter) gapsEmpty(p ipoly, ends []float64) bool {
	type pt struct {
		n *big.Int
		q int
	}
	pts := []pt{{new(big.Int), 0}}
	for _, w := range ends {
		n, q := cc.point(w)
		pts = append(pts, pt{n, q})
	}
	for i := 0; i+1 < len(pts); i++ {
		want := i % 2
		if descartesIn(p, pts[i].n, pts[i].q, pts[i+1].n, pts[i+1].q) != want {
			return false
		}
	}
	last := pts[len(pts)-1]
	return descartesIn(p, last.n, last.q, nil, 0) == 0
}

// squarefreePrime is the Mersenne prime 2^61 − 1.
const squarefreePrime = 1<<61 - 1

func mulMod(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	_, r := bits.Div64(hi, lo, squarefreePrime)
	return r
}

func powMod(a, e uint64) uint64 {
	r := uint64(1)
	for ; e > 0; e >>= 1 {
		if e&1 == 1 {
			r = mulMod(r, a)
		}
		a = mulMod(a, a)
	}
	return r
}

// squarefreeModPrime reports a proof that p is squarefree over the
// rationals: modulo the prime, which does not divide its leading
// coefficient, gcd(p, p′) is a constant. False proves nothing.
func squarefreeModPrime(p ipoly) bool {
	m := big.NewInt(squarefreePrime)
	red := func(q ipoly) []uint64 {
		o := make([]uint64, len(q))
		t := new(big.Int)
		for i, c := range q {
			o[i] = t.Mod(c, m).Uint64()
		}
		for len(o) > 0 && o[len(o)-1] == 0 {
			o = o[:len(o)-1]
		}
		return o
	}
	a := red(p)
	if len(a) != len(p) {
		return false
	}
	b := red(ipolyDeriv(p))
	if len(b) != len(p)-1 {
		return false
	}
	for len(b) > 0 {
		inv := powMod(b[len(b)-1], squarefreePrime-2)
		for len(a) >= len(b) {
			q := mulMod(a[len(a)-1], inv)
			sh := len(a) - len(b)
			for i, c := range b {
				a[sh+i] = (a[sh+i] + squarefreePrime - mulMod(q, c)) % squarefreePrime
			}
			for len(a) > 0 && a[len(a)-1] == 0 {
				a = a[:len(a)-1]
			}
			if len(a) == 0 {
				break
			}
		}
		a, b = b, a
	}
	return len(a) == 1
}

// firstIsolated reports whether Descartes' rule proves p root-free on
// (0, lo) and with one root in (lo, hi): the first crossing, the one
// Bandwidth's search (firstCrossing) takes, is then the bracketed one.
func (cc *crossingCounter) firstIsolated(p ipoly, lo, hi float64) bool {
	ln, lq := cc.point(lo)
	hn, hq := cc.point(hi)
	return descartesIn(p, new(big.Int), 0, ln, lq) == 0 && descartesIn(p, ln, lq, hn, hq) == 1
}
