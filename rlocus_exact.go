package controlsys

import (
	"cmp"
	"math"
	"math/big"
	"math/bits"
	"slices"
)

const (
	// rlExactMaxStates is the largest delay-free loop order whose break
	// points exactBreakaway computes; larger loops use the eig path.
	rlExactMaxStates = 12
	// rlExactBitBudget bounds n·(s+64), s the binary exponent spread of
	// the nonzero entries of A and B·C, which sets the integer size of the
	// exact path (about 50 ms at n = 12 on the budget's edge).
	rlExactBitBudget = 4800
	// rlExactMaxSteps caps the bisection steps per root.
	rlExactMaxSteps = 4000
)

// exactBreakaway returns the break points of the explicit delay-absorbed
// loop sys by exactBreakaway, from the delay-free undelayed loop with
// delay extra poles at z = 0 when there is one; the strictly proper
// numerator is trimmed to the relative degree lp.relDeg (lp.spRelDeg with
// a feedthrough) as in computeBreakaway.
func (lp *rlocusLoop) exactBreakaway(sys, undelayed *System, delay int) ([]complex128, bool) {
	ex, exDelay := sys, 0
	if delay > 0 && undelayed != nil {
		ex, exDelay = undelayed, delay
	}
	r := lp.relDeg
	if lp.d != 0 {
		r = lp.spRelDeg
	}
	keep := lp.n
	if r <= lp.n {
		keep = lp.n - r
	}
	n, _, _ := ex.Dims()
	if n < 1 || n > rlExactMaxStates {
		return nil, false
	}
	a := make([]float64, n*n)
	b := make([]float64, n)
	ar, br, cr := ex.A.RawMatrix(), ex.B.RawMatrix(), ex.C.RawMatrix()
	for i := range n {
		copy(a[i*n:i*n+n], ar.Data[i*ar.Stride:i*ar.Stride+n])
		b[i] = br.Data[i*br.Stride]
	}
	return exactBreakaway(a, b, cr.Data[:n], ex.D.At(0, 0), n, exDelay, keep)
}

// exactBreakaway returns the break points of the SISO loop
// G(x) = C(xI−A)⁻¹B + d with keep ≥ 0 the highest power of x kept in the
// numerator of C(xI−A)⁻¹B (the Markov relative-degree trim) and delay extra
// poles at x = 0, computed exactly from the stored floats: A, B, C, d are
// dyadic rationals, so N and D are exact integer polynomials (Berkowitz),
// common factors cancel exactly, and every distinct real root of
// q = N'D − ND' is kept when it is a multiple pole (K = 0), a multiple
// zero (K = ∞) or a stationary point with K = −D/N > 0, decided by exact
// signs. The points are rounded to float64 and sorted. ok is false when the
// loop is outside the exact path's size gates (rlExactMaxStates,
// rlExactBitBudget).
func exactBreakaway(a, b, c []float64, d float64, n, delay, keep int) (pts []complex128, ok bool) {
	if n < 1 || n > rlExactMaxStates {
		return nil, false
	}
	ad := make([]dyadic, n*n)
	bc := make([]dyadic, n*n)
	emin, emax := math.MaxInt, math.MinInt
	note := func(x dyadic) {
		if x.m.Sign() != 0 {
			emin = min(emin, x.e)
			emax = max(emax, x.e+x.m.BitLen())
		}
	}
	bd := make([]dyadic, n)
	cd := make([]dyadic, n)
	for i := range n {
		bd[i], cd[i] = newDyadic(b[i]), newDyadic(c[i])
	}
	for i := range n {
		for j := range n {
			ad[i*n+j] = newDyadic(a[i*n+j])
			bc[i*n+j] = bd[i].mul(cd[j])
			note(ad[i*n+j])
			note(bc[i*n+j])
		}
	}
	if emin == math.MaxInt {
		emin, emax = 0, 0
	}
	if n*(emax-emin+64) > rlExactBitBudget {
		return nil, false
	}
	toInt := func(x dyadic) *big.Int {
		if x.m.Sign() == 0 {
			return new(big.Int)
		}
		return new(big.Int).Lsh(x.m, uint(x.e-emin))
	}
	m := make([][]*big.Int, n)
	p := make([][]*big.Int, n)
	for i := range n {
		m[i], p[i] = make([]*big.Int, n), make([]*big.Int, n)
		for j := range n {
			m[i][j] = toInt(ad[i*n+j])
			p[i][j] = new(big.Int).Sub(m[i][j], toInt(bc[i*n+j]))
		}
	}
	den := intCharPoly(m)
	cp := intCharPoly(p)
	num := make([]*big.Int, n+1)
	for i := range num {
		num[i] = new(big.Int)
		if n-i <= keep {
			num[i].Sub(cp[i], den[i])
		}
	}
	if d != 0 {
		dd := newDyadic(d)
		t := new(big.Int)
		for i := range num {
			if dd.e >= 0 {
				num[i].Add(num[i], t.Mul(t.Lsh(dd.m, uint(dd.e)), den[i]))
			} else {
				num[i].Lsh(num[i], uint(-dd.e))
				num[i].Add(num[i], t.Mul(dd.m, den[i]))
			}
		}
	}
	for range delay {
		den = append(den, new(big.Int))
	}
	roots, ok := xpBreakPoints(num, den)
	if !ok {
		return nil, false
	}
	pts = []complex128{}
	for _, r := range roots {
		f := new(big.Float).SetInt(r.m)
		f.SetMantExp(f, r.e+emin)
		x, _ := f.Float64()
		if math.IsInf(x, 0) {
			continue
		}
		pts = append(pts, complex(x, 0))
	}
	slices.SortFunc(pts, func(x, y complex128) int { return cmp.Compare(real(x), real(y)) })
	return slices.Compact(pts), true
}

// xpRoot is the dyadic m·2^e.
type xpRoot struct {
	m *big.Int
	e int
}

// xpBreakPoints returns the break points of N/D (integer coefficients,
// leading first) as dyadic approximations within 2^−62 relative of the
// exact real roots of q = N'D − ND' that are multiple poles, multiple
// zeros or stationary points with N·D < 0; ok is false when a root needs
// more than rlExactMaxSteps bisections.
func xpBreakPoints(num, den []*big.Int) ([]xpRoot, bool) {
	num, den = xpTrim(num), xpTrim(den)
	if len(num) == 0 || len(den) == 0 {
		return nil, true
	}
	for len(num) > 1 && len(den) > 1 && num[len(num)-1].Sign() == 0 && den[len(den)-1].Sign() == 0 {
		num, den = num[:len(num)-1], den[:len(den)-1]
	}
	if g := xpGcd(num, den); len(g) > 1 {
		var ok1, ok2 bool
		num, ok1 = xpDivExact(num, g)
		den, ok2 = xpDivExact(den, g)
		if !ok1 || !ok2 {
			return nil, false
		}
	}
	q := xpPrimitive(xpSub(xpMul(xpDeriv(num), den), xpMul(num, xpDeriv(den))))
	if len(q) <= 1 {
		return nil, true
	}
	var out []xpRoot
	if q[len(q)-1].Sign() == 0 {
		for q[len(q)-1].Sign() == 0 {
			q = q[:len(q)-1]
		}
		n0, d0 := num[len(num)-1].Sign(), den[len(den)-1].Sign()
		if d0 == 0 || n0 == 0 || n0*d0 < 0 {
			out = append(out, xpRoot{new(big.Int), 0})
		}
	}
	if len(q) <= 1 {
		return out, true
	}
	sqf := q
	if g := xpGcd(q, xpDeriv(q)); len(g) > 1 {
		var ok bool
		if sqf, ok = xpDivExact(q, g); !ok {
			return nil, false
		}
	}
	gD, gN := xpGcd(sqf, den), xpGcd(sqf, num)
	for _, neg := range []bool{false, true} {
		f := sqf
		nf, df, gDf, gNf := num, den, gD, gN
		if neg {
			f, nf, df, gDf, gNf = xpNegArg(f), xpNegArg(num), xpNegArg(den), xpNegArg(gD), xpNegArg(gN)
		}
		for _, iv := range xpIsolatePositive(f) {
			r, keepIt, ok := xpClassify(f, nf, df, gDf, gNf, iv)
			if !ok {
				return nil, false
			}
			if keepIt {
				if neg {
					r.m.Neg(r.m)
				}
				out = append(out, r)
			}
		}
	}
	return out, true
}

// xpIval is the open interval (lo, hi)·2^e, or the exact root lo·2^e.
type xpIval struct {
	lo, hi *big.Int
	e      int
	exact  bool
}

// xpClassify refines the root of the square-free f in iv to 2^−62
// relative and reports whether it is a break point: a root of gD (a
// multiple pole) or gN (a multiple zero), else a point where N·D < 0.
func xpClassify(f, num, den, gD, gN []*big.Int, iv xpIval) (r xpRoot, keep, ok bool) {
	if iv.exact {
		lo, e := new(big.Int).Set(iv.lo), iv.e
		if xpRootAt(gD, lo, e) || xpRootAt(gN, lo, e) {
			return xpRoot{lo, e}, true, true
		}
		return xpRoot{lo, e}, xpSign(num, lo, e)*xpSign(den, lo, e) < 0, true
	}
	lo, hi, e := new(big.Int).Set(iv.lo), new(big.Int).Set(iv.hi), iv.e
	sLo := xpSign(f, lo, e)
	if sLo == 0 {
		// lo is the exact root split off by xpIsolatePositive: f just
		// right of it has the sign of f'(lo).
		sLo = xpSign(xpDeriv(f), lo, e)
	}
	w, t := new(big.Int), new(big.Int)
	mid := new(big.Int)
	resolved := func() bool {
		if w.Sub(hi, lo); t.Lsh(w, 62).Cmp(lo) > 0 {
			return false
		}
		return true
	}
	for step := 0; ; step++ {
		if step > rlExactMaxSteps {
			return r, false, false
		}
		if resolved() {
			break
		}
		lo.Lsh(lo, 1)
		hi.Lsh(hi, 1)
		e--
		mid.Add(lo, hi).Rsh(mid, 1)
		s := xpSign(f, mid, e)
		if s == 0 {
			return xpClassify(f, num, den, gD, gN, xpIval{lo: new(big.Int).Set(mid), e: e, exact: true})
		}
		if s == sLo {
			lo.Set(mid)
		} else {
			hi.Set(mid)
		}
	}
	at := func() xpRoot {
		m := new(big.Int).Add(lo, hi)
		return xpRoot{m, e - 1}
	}
	if xpSignChange(gD, lo, hi, e) || xpSignChange(gN, lo, hi, e) {
		return at(), true, true
	}
	for step := 0; ; step++ {
		if step > rlExactMaxSteps {
			return r, false, false
		}
		sn, sd := xpSign(num, lo, e), xpSign(den, lo, e)
		if sn != 0 && sd != 0 && xpDescartes(num, lo, hi, e) == 0 && xpDescartes(den, lo, hi, e) == 0 {
			return at(), sn*sd < 0, true
		}
		lo.Lsh(lo, 1)
		hi.Lsh(hi, 1)
		e--
		mid.Add(lo, hi).Rsh(mid, 1)
		s := xpSign(f, mid, e)
		if s == 0 {
			return xpClassify(f, num, den, gD, gN, xpIval{lo: new(big.Int).Set(mid), e: e, exact: true})
		}
		if s == sLo {
			lo.Set(mid)
		} else {
			hi.Set(mid)
		}
	}
}

// xpIsolatePositive returns isolating intervals of the positive roots of
// the square-free f (f(0) ≠ 0) by Descartes' rule on bisected intervals.
func xpIsolatePositive(f []*big.Int) []xpIval {
	if len(f) <= 1 {
		return nil
	}
	// Fujiwara: every root has |x| < 2·max_k |f_k/f_0|^(1/k).
	b := math.MinInt
	for k, c := range f[1:] {
		if c.Sign() != 0 {
			r, q := c.BitLen()-f[0].BitLen()+1, k+1
			if r >= 0 {
				r = (r + q - 1) / q
			} else {
				r = -(-r / q)
			}
			b = max(b, r+1)
		}
	}
	if b == math.MinInt {
		return nil
	}
	type node struct {
		lo, hi *big.Int
		e      int
	}
	stack := []node{{new(big.Int), big.NewInt(1), b}}
	var out []xpIval
	for len(stack) > 0 {
		nd := stack[len(stack)-1]
		stack = stack[:len(stack)-1]
		switch v := xpDescartes(f, nd.lo, nd.hi, nd.e); {
		case v == 0:
		case v == 1:
			out = append(out, xpIval{lo: nd.lo, hi: nd.hi, e: nd.e})
		default:
			lo := new(big.Int).Lsh(nd.lo, 1)
			hi := new(big.Int).Lsh(nd.hi, 1)
			mid := new(big.Int).Add(nd.lo, nd.hi)
			if xpSign(f, mid, nd.e-1) == 0 {
				out = append(out, xpIval{lo: mid, e: nd.e - 1, exact: true})
			}
			stack = append(stack, node{mid, hi, nd.e - 1}, node{lo, new(big.Int).Set(mid), nd.e - 1})
		}
	}
	return out
}

// xpDescartes returns the sign variations of (1+x)^d·p(lo + (hi−lo)/(1+x))
// for the interval (lo, hi)·2^e: 0 means no root of p in it, 1 exactly one.
func xpDescartes(p []*big.Int, lo, hi *big.Int, e int) int {
	d := len(p) - 1
	if d < 1 {
		return 0
	}
	u := make([]*big.Int, len(p))
	for i, c := range p {
		k := d - i
		if e >= 0 {
			u[i] = new(big.Int).Lsh(c, uint(e*k))
		} else {
			u[i] = new(big.Int).Lsh(c, uint(-e*(d-k)))
		}
	}
	u = intShift(u, lo)
	w := new(big.Int).Sub(hi, lo)
	pw := big.NewInt(1)
	for i := d; i >= 0; i-- {
		u[i].Mul(u[i], pw)
		pw.Mul(pw, w)
	}
	slices.Reverse(u)
	u = intShift(u, big.NewInt(1))
	v, prev := 0, 0
	for _, c := range u {
		if s := c.Sign(); s != 0 {
			if prev != 0 && s != prev {
				v++
			}
			prev = s
		}
	}
	return v
}

// xpSign returns the sign of p(m·2^e).
func xpSign(p []*big.Int, m *big.Int, e int) int {
	acc, t := new(big.Int), new(big.Int)
	if e >= 0 {
		x := new(big.Int).Lsh(m, uint(e))
		for _, c := range p {
			acc.Mul(acc, x)
			acc.Add(acc, c)
		}
		return acc.Sign()
	}
	// p(m·2^e)·2^(−e·d) = Σ cᵢ·m^(d−i)·2^(−e·i).
	for i, c := range p {
		acc.Mul(acc, m)
		acc.Add(acc, t.Lsh(c, uint(-e*i)))
	}
	return acc.Sign()
}

// xpRootAt reports whether m·2^e is a root of p (of degree ≥ 1).
func xpRootAt(p []*big.Int, m *big.Int, e int) bool {
	return len(p) > 1 && xpSign(p, m, e) == 0
}

// xpSignChange reports whether p (of degree ≥ 1) changes sign over
// (lo, hi)·2^e.
func xpSignChange(p []*big.Int, lo, hi *big.Int, e int) bool {
	return len(p) > 1 && xpSign(p, lo, e)*xpSign(p, hi, e) < 0
}

func xpTrim(p []*big.Int) []*big.Int {
	for len(p) > 0 && p[0].Sign() == 0 {
		p = p[1:]
	}
	return p
}

func xpDeriv(p []*big.Int) []*big.Int {
	d := len(p) - 1
	if d < 1 {
		return nil
	}
	out := make([]*big.Int, d)
	for i := range out {
		out[i] = new(big.Int).Mul(p[i], big.NewInt(int64(d-i)))
	}
	return out
}

func xpMul(a, b []*big.Int) []*big.Int {
	if len(a) == 0 || len(b) == 0 {
		return nil
	}
	out := make([]*big.Int, len(a)+len(b)-1)
	for i := range out {
		out[i] = new(big.Int)
	}
	t := new(big.Int)
	for i, x := range a {
		if x.Sign() == 0 {
			continue
		}
		for j, y := range b {
			out[i+j].Add(out[i+j], t.Mul(x, y))
		}
	}
	return out
}

func xpSub(a, b []*big.Int) []*big.Int {
	out := make([]*big.Int, max(len(a), len(b)))
	for i := range out {
		out[i] = new(big.Int)
		if j := i - (len(out) - len(a)); j >= 0 {
			out[i].Add(out[i], a[j])
		}
		if j := i - (len(out) - len(b)); j >= 0 {
			out[i].Sub(out[i], b[j])
		}
	}
	return xpTrim(out)
}

// xpNegArg returns p(−x).
func xpNegArg(p []*big.Int) []*big.Int {
	out := make([]*big.Int, len(p))
	d := len(p) - 1
	for i, c := range p {
		out[i] = new(big.Int).Set(c)
		if (d-i)%2 == 1 {
			out[i].Neg(out[i])
		}
	}
	return out
}

// xpPrimitive returns p divided by its content, leading coefficient
// positive.
func xpPrimitive(p []*big.Int) []*big.Int {
	p = xpTrim(p)
	if len(p) == 0 {
		return nil
	}
	g := new(big.Int)
	for _, c := range p {
		if c.Sign() != 0 {
			g.GCD(nil, nil, g, new(big.Int).Abs(c))
		}
	}
	if p[0].Sign() < 0 {
		g.Neg(g)
	}
	out := make([]*big.Int, len(p))
	for i, c := range p {
		out[i] = new(big.Int).Quo(c, g)
	}
	return out
}

// xpGcd returns the primitive gcd of a and b ([1] when coprime): a
// coprimality proof modulo 2^61−1 when both leading coefficients survive
// the reduction, else the primitive remainder sequence.
func xpGcd(a, b []*big.Int) []*big.Int {
	a, b = xpPrimitive(a), xpPrimitive(b)
	one := []*big.Int{big.NewInt(1)}
	switch {
	case len(a) == 0 && len(b) == 0:
		return one
	case len(a) == 0:
		return b
	case len(b) == 0:
		return a
	case len(a) == 1 || len(b) == 1:
		return one
	}
	if ma, mb := xpModP(a), xpModP(b); ma[0] != 0 && mb[0] != 0 && xpGcdDegModP(ma, mb) == 0 {
		return one
	}
	if len(a) < len(b) {
		a, b = b, a
	}
	for {
		r := xpPrimitive(xpPrem(a, b))
		if len(r) == 0 {
			return b
		}
		if len(r) == 1 {
			return one
		}
		a, b = b, r
	}
}

// xpPrem returns a pseudo-remainder of a by b (len(a) ≥ len(b) ≥ 2).
func xpPrem(a, b []*big.Int) []*big.Int {
	r := make([]*big.Int, len(a))
	for i, c := range a {
		r[i] = new(big.Int).Set(c)
	}
	t := new(big.Int)
	for len(r) >= len(b) {
		lr := new(big.Int).Set(r[0])
		for i := range r {
			r[i].Mul(r[i], b[0])
		}
		for j, c := range b {
			r[j].Sub(r[j], t.Mul(lr, c))
		}
		r = xpTrim(r[1:])
	}
	return r
}

// xpDivExact returns a/b when b divides a over the integers.
func xpDivExact(a, b []*big.Int) ([]*big.Int, bool) {
	a, b = xpTrim(a), xpTrim(b)
	if len(b) == 0 || len(a) < len(b) {
		return nil, len(a) == 0
	}
	r := make([]*big.Int, len(a))
	for i, c := range a {
		r[i] = new(big.Int).Set(c)
	}
	q := make([]*big.Int, len(a)-len(b)+1)
	t, rem := new(big.Int), new(big.Int)
	for i := range q {
		q[i], rem = new(big.Int).QuoRem(r[i], b[0], rem)
		if rem.Sign() != 0 {
			return nil, false
		}
		for j, c := range b {
			r[i+j].Sub(r[i+j], t.Mul(q[i], c))
		}
	}
	for _, c := range r[len(q):] {
		if c.Sign() != 0 {
			return nil, false
		}
	}
	return q, true
}

const xpPrime = 1<<61 - 1

func xpModP(p []*big.Int) []uint64 {
	m := new(big.Int).SetUint64(xpPrime)
	out := make([]uint64, len(p))
	t := new(big.Int)
	for i, c := range p {
		out[i] = t.Mod(c, m).Uint64()
	}
	return out
}

func xpMulMod(a, b uint64) uint64 {
	hi, lo := bits.Mul64(a, b)
	_, r := bits.Div64(hi, lo, xpPrime)
	return r
}

func xpInvMod(a uint64) uint64 {
	r, x := uint64(1), a
	for e := uint64(xpPrime - 2); e > 0; e >>= 1 {
		if e&1 == 1 {
			r = xpMulMod(r, x)
		}
		x = xpMulMod(x, x)
	}
	return r
}

// xpGcdDegModP returns the degree of gcd(a, b) over GF(2^61−1), leading
// coefficients nonzero.
func xpGcdDegModP(a, b []uint64) int {
	a, b = slices.Clone(a), slices.Clone(b)
	trim := func(p []uint64) []uint64 {
		for len(p) > 0 && p[0] == 0 {
			p = p[1:]
		}
		return p
	}
	if len(a) < len(b) {
		a, b = b, a
	}
	for len(b) > 0 {
		inv := xpInvMod(b[0])
		for len(a) >= len(b) {
			f := xpMulMod(a[0], inv)
			for j, c := range b {
				a[j] = (a[j] + xpPrime - xpMulMod(f, c)) % xpPrime
			}
			a = trim(a[1:])
		}
		a, b = b, a
	}
	return len(a) - 1
}
