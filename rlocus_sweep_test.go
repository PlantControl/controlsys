package controlsys

import (
	"encoding/json"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"os"
	"slices"
	"strings"
	"testing"
)

// Frozen RootLocus regression sweep (track rlocus, wave 19). The generator
// and the verdict rules below are frozen: testdata/rlocus_sweep_base.json
// holds the verdicts of base rlswBaseRev (AND of arm64 and GOARCH=amd64
// runs) and every later revision must keep each verdict base got right.
// Regenerate the file only with base code: RLOCUS_SWEEP_WRITE=<path>.
// rlocus_sweep_ext_test.go adds a second frozen set of harder families.
//
// Each case is a SISO loop k·Z(s)/P(s) (+D) with dyadic poles, zeros and
// gain, realized as a companion form, a Givens-rotated companion or a
// power-of-two-scaled companion, optionally discrete with an input delay.
// Two exact big.Rat oracles judge the breakaway set: the intended TF and
// the TF of the float realization; a result consistent with either one
// that is well posed passes. Asymptotes, departure angles and the arrival
// count are judged against the intended TF.

const (
	rlswCases = 2000
	rlswSeed1 = 0x5eed19
	rlswSeed2 = 0x0c0ffee
)

const (
	rlswBreak = 1 << iota
	rlswAsym
	rlswDepart
	rlswArrive
)

type rlswCase struct {
	n, L         int
	dt           float64
	a, b, c      []float64
	d            float64
	num, den     []*big.Rat // intended, descending; den monic without z^L
	poles, zeros []complex128
	desc         string
}

func (c *rlswCase) String() string {
	return fmt.Sprintf("%s n=%d L=%d dt=%g\nA=%v\nB=%v\nC=%v\nD=%v", c.desc, c.n, c.L, c.dt, c.a, c.b, c.c, c.d)
}

func rlswPolyFromRoots(roots []complex128) []*big.Rat {
	p := []*big.Rat{big.NewRat(1, 1)}
	for i := 0; i < len(roots); i++ {
		r := roots[i]
		if imag(r) == 0 {
			p = fzRatPolyMul(p, []*big.Rat{big.NewRat(1, 1), new(big.Rat).Neg(fzRat(real(r)))})
			continue
		}
		re, im := fzRat(real(r)), fzRat(imag(r))
		sum := new(big.Rat).Mul(re, big.NewRat(-2, 1))
		prod := new(big.Rat).Add(new(big.Rat).Mul(re, re), new(big.Rat).Mul(im, im))
		p = fzRatPolyMul(p, []*big.Rat{big.NewRat(1, 1), sum, prod})
		i++
	}
	return p
}

func rlswF(x *big.Rat) float64 {
	f, _ := x.Float64()
	return f
}

func rlswGenRoots(r *rand.Rand, cnt int, sc float64, allowSpecial bool) []complex128 {
	var out []complex128
	real1 := func() float64 { return float64(r.IntN(97)-64) / 32 * sc }
	pair := func(re, im float64) { out = append(out, complex(re, im), complex(re, -im)) }
	if allowSpecial && cnt >= 2 {
		switch r.IntN(10) {
		case 0:
			p := real1()
			m := 2
			if cnt >= 3 && r.IntN(2) == 0 {
				m = 3
			}
			for range m {
				out = append(out, complex(p, 0))
			}
		case 1:
			p := -float64(8+r.IntN(40)) / 32 * sc
			delta := math.Ldexp(1, -(8 + r.IntN(8)))
			out = append(out, complex(p, 0))
			if cnt >= 3 && r.IntN(2) == 0 {
				pair(p*(1+delta), math.Abs(p)*delta*float64(1+r.IntN(3))/4)
			} else {
				out = append(out, complex(p*(1+delta), 0))
			}
		case 2:
			re := -float64(8+r.IntN(40)) / 32 * sc
			pair(re, math.Abs(re)*math.Ldexp(1, -(6+r.IntN(14))))
		case 3:
			out = append(out, 0)
			if r.IntN(2) == 0 {
				out = append(out, 0)
			}
		}
	}
	for len(out) < cnt {
		if cnt-len(out) >= 2 && r.IntN(3) == 0 {
			pair(float64(r.IntN(65)-48)/32*sc, float64(1+r.IntN(48))/32*sc)
			continue
		}
		out = append(out, complex(real1(), 0))
	}
	return out
}

func rlswRotate(r *rand.Rand, n int, a, b, c []float64) {
	for range n + 1 {
		i, j := r.IntN(n), r.IntN(n)
		if i == j {
			continue
		}
		p, q := float64(1+r.IntN(7)), float64(1+r.IntN(7))
		h := p*p + q*q
		cs, sn := (p*p-q*q)/h, 2*p*q/h
		rot := func(x, y float64) (float64, float64) {
			return float64(cs*x) - float64(sn*y), float64(sn*x) + float64(cs*y)
		}
		for k := range n {
			a[i*n+k], a[j*n+k] = rot(a[i*n+k], a[j*n+k])
		}
		for k := range n {
			a[k*n+i], a[k*n+j] = rot(a[k*n+i], a[k*n+j])
		}
		b[i], b[j] = rot(b[i], b[j])
		c[i], c[j] = rot(c[i], c[j])
	}
}

func rlswGenerate() []rlswCase {
	r := rand.New(rand.NewPCG(rlswSeed1, rlswSeed2))
	var cases []rlswCase
	for len(cases) < rlswCases {
		n := 2 + r.IntN(4)
		e := -16 + r.IntN(27)
		sc := math.Ldexp(1, e)
		poles := rlswGenRoots(r, n, sc, true)
		nz := r.IntN(n)
		zeros := rlswGenRoots(r, nz, sc, r.IntN(4) == 0)
		if len(zeros) > nz {
			zeros = zeros[:nz]
			if nz > 0 && imag(zeros[nz-1]) > 0 {
				zeros[nz-1] = complex(real(zeros[nz-1]), 0)
			}
		}
		sign := 1.0
		if r.IntN(2) == 0 {
			sign = -1
		}
		k := sign * float64(1+r.IntN(15)) / 4 * math.Ldexp(1, e*(n-nz))
		d := 0.0
		if r.IntN(10) == 0 {
			d = float64(r.IntN(2)*2-1) * float64(1+r.IntN(8)) / 4
		}
		den := rlswPolyFromRoots(poles)
		zp := rlswPolyFromRoots(zeros)
		num := make([]*big.Rat, n+1)
		for i := range num {
			num[i] = new(big.Rat).Mul(fzRat(d), den[i])
			if j := i - (n - nz); j >= 0 {
				num[i].Add(num[i], new(big.Rat).Mul(fzRat(k), zp[j]))
			}
		}
		if len(fzTrimRat(num)) == 0 {
			continue
		}
		a := make([]float64, n*n)
		b := make([]float64, n)
		c := make([]float64, n)
		for i := 0; i < n-1; i++ {
			a[i*n+i+1] = 1
		}
		for j := range n {
			a[(n-1)*n+j] = -rlswF(den[n-j])
			c[j] = rlswF(new(big.Rat).Sub(num[n-j], new(big.Rat).Mul(fzRat(d), den[n-j])))
		}
		b[n-1] = 1
		desc := "companion"
		switch kind := r.IntN(20); {
		case kind >= 7 && kind < 16:
			rlswRotate(r, n, a, b, c)
			desc = "rotated"
		case kind >= 16:
			ks := make([]int, n)
			for i := range ks {
				ks[i] = r.IntN(9) - 4
			}
			for i := range n {
				for j := range n {
					a[i*n+j] = math.Ldexp(a[i*n+j], ks[i]-ks[j])
				}
				b[i] = math.Ldexp(b[i], ks[i])
				c[i] = math.Ldexp(c[i], -ks[i])
			}
			desc = "scaled"
		}
		cs := rlswCase{n: n, a: a, b: b, c: c, d: d, num: num, den: den, poles: poles, zeros: zeros, desc: desc}
		if r.IntN(5) == 0 {
			cs.dt, cs.L = 1, 1+r.IntN(3)
		}
		cases = append(cases, cs)
	}
	return cases
}

func (c *rlswCase) system() (*System, error) {
	sys, err := NewFromSlices(c.n, 1, 1, c.a, c.b, c.c, []float64{c.d}, c.dt)
	if err != nil {
		return nil, err
	}
	if c.L > 0 {
		if err := sys.SetInputDelay([]float64{float64(c.L)}); err != nil {
			return nil, err
		}
	}
	return sys, nil
}

// rlswOracle is the exact breakaway answer of one TF N/D (D including z^L).
type rlswOracle struct {
	posed    bool
	required []float64
	scale    float64
}

func rlswRatDeriv(p []*big.Rat) []*big.Rat {
	if len(p) <= 1 {
		return nil
	}
	out := make([]*big.Rat, len(p)-1)
	for i := range out {
		out[i] = new(big.Rat).Mul(p[i], big.NewRat(int64(len(p)-1-i), 1))
	}
	return out
}

func rlswRatSub(a, b []*big.Rat) []*big.Rat {
	out := make([]*big.Rat, max(len(a), len(b)))
	for i := range out {
		out[i] = new(big.Rat)
		if j := i - (len(out) - len(a)); j >= 0 {
			out[i].Add(out[i], a[j])
		}
		if j := i - (len(out) - len(b)); j >= 0 {
			out[i].Sub(out[i], b[j])
		}
	}
	return fzTrimRat(out)
}

func rlswRatQuo(a, b []*big.Rat) []*big.Rat {
	r := make([]*big.Rat, len(a))
	for i, x := range a {
		r[i] = new(big.Rat).Set(x)
	}
	q := make([]*big.Rat, len(a)-len(b)+1)
	var t big.Rat
	for i := range q {
		q[i] = new(big.Rat).Quo(r[i], b[0])
		for j := range b {
			r[i+j].Sub(r[i+j], t.Mul(q[i], b[j]))
		}
	}
	return q
}

// rlswRealRoots isolates the distinct real roots of the descending p.
func rlswRealRoots(p []*big.Rat) (roots []float64, ok bool) {
	p = fzTrimRat(p)
	asc := make(fzMgPoly, len(p))
	for i, x := range p {
		asc[len(p)-1-i] = x
	}
	if len(asc) > 0 && asc[0].Sign() == 0 {
		roots = append(roots, 0)
	}
	pos, _, ok1 := fzMgPositiveRoots(asc)
	neg := make(fzMgPoly, len(asc))
	for i, x := range asc {
		neg[i] = new(big.Rat).Set(x)
		if i%2 == 1 {
			neg[i].Neg(neg[i])
		}
	}
	negr, _, ok2 := fzMgPositiveRoots(neg)
	for _, x := range pos {
		if x.cluster {
			return nil, false
		}
		roots = append(roots, fzMgMid(x.lo, x.hi))
	}
	for _, x := range negr {
		if x.cluster {
			return nil, false
		}
		roots = append(roots, -fzMgMid(x.lo, x.hi))
	}
	return roots, ok1 && ok2
}

func rlswFloatRoots(p []*big.Rat) []complex128 {
	p = fzTrimRat(p)
	if len(p) <= 1 {
		return nil
	}
	f := make(Poly, len(p))
	for i, x := range p {
		f[i] = rlswF(x)
	}
	r, err := f.Roots()
	if err != nil {
		return nil
	}
	return r
}

func rlswSquareFree(p []*big.Rat) []*big.Rat {
	p = fzTrimRat(p)
	if len(p) <= 2 {
		return p
	}
	g := fzPolyGcdRat(p, rlswRatDeriv(p))
	if len(g) <= 1 {
		return p
	}
	return rlswRatQuo(p, g)
}

func newRlswOracle(num, den []*big.Rat) rlswOracle {
	num, den = fzTrimRat(num), fzTrimRat(den)
	o := rlswOracle{}
	if len(num) == 0 || len(den) == 0 {
		return o
	}
	if g := fzPolyGcdRat(den, num); len(g) > 1 {
		return o
	}
	pr := rlswFloatRoots(den)
	R := 0.0
	for _, x := range pr {
		R = math.Max(R, cmplx.Abs(x))
	}
	if R == 0 || math.IsNaN(R) {
		R = 1
	}
	o.scale = R
	S := func(x complex128) float64 { return math.Max(cmplx.Abs(x), 1e-3*R) }
	dsf, nsf := rlswFloatRoots(rlswSquareFree(den)), rlswFloatRoots(rlswSquareFree(num))
	all := slices.Concat(dsf, nsf)
	for i, x := range all {
		for _, y := range all[i+1:] {
			if cmplx.Abs(x-y) < 1e-5*S(x) {
				return o
			}
		}
	}
	q := rlswRatSub(fzRatPolyMul(rlswRatDeriv(num), den), fzRatPolyMul(num, rlswRatDeriv(den)))
	if len(q) == 0 {
		o.posed = true
		return o
	}
	gD := fzPolyGcdRat(q, den)
	qr := rlswRatQuo(q, gD)
	var gN []*big.Rat
	if len(num) > 1 {
		gN = fzPolyGcdRat(qr, num)
		qr = rlswRatQuo(qr, gN)
	}
	for _, x := range rlswFloatRoots(qr) {
		if imag(x) != 0 && math.Abs(imag(x)) < 1e-6*S(x) {
			return o
		}
	}
	cand, ok := rlswRealRoots(qr)
	if !ok {
		return o
	}
	slices.Sort(cand)
	for i := 1; i < len(cand); i++ {
		if cand[i]-cand[i-1] < 1e-6*S(complex(cand[i], 0)) {
			return o
		}
	}
	for _, g := range [][]*big.Rat{gD, gN} {
		if len(g) <= 1 {
			continue
		}
		rs, ok := rlswRealRoots(g)
		if !ok {
			return o
		}
		o.required = append(o.required, rs...)
	}
	for _, x := range cand {
		for _, y := range all {
			if cmplx.Abs(complex(x, 0)-y) < 1e-6*S(y) {
				return o
			}
		}
		xr := fzRat(x)
		dv, nv := fzPolyEvalRat(den, xr), fzPolyEvalRat(num, xr)
		if dv.Sign()*nv.Sign() < 0 {
			o.required = append(o.required, x)
		}
	}
	slices.Sort(o.required)
	o.required = slices.CompactFunc(o.required, func(x, y float64) bool { return rlswSamePoint(complex(x, 0), complex(y, 0)) })
	o.posed = true
	return o
}

// rlswSamePoint reports whether x and y are one break point listed twice.
func rlswSamePoint(x, y complex128) bool {
	return x == y || cmplx.Abs(x-y) <= 1e-6*math.Max(cmplx.Abs(x), 1e-300)
}

// rlswDuplicate returns a break point bp lists twice.
func rlswDuplicate(bp []complex128) (complex128, bool) {
	for i, x := range bp {
		if slices.ContainsFunc(bp[i+1:], func(y complex128) bool { return rlswSamePoint(x, y) }) {
			return x, true
		}
	}
	return 0, false
}

// accepts reports whether bp is the required set, each point listed once.
func (o rlswOracle) accepts(bp []complex128) bool {
	if _, dup := rlswDuplicate(bp); !o.posed || dup {
		return false
	}
	tol := func(x float64) float64 { return 1e-4 * math.Max(math.Abs(x), 1e-3*o.scale) }
	for _, z := range bp {
		if imag(z) != 0 || !slices.ContainsFunc(o.required, func(x float64) bool { return math.Abs(real(z)-x) <= tol(x) }) {
			return false
		}
	}
	for _, x := range o.required {
		if !slices.ContainsFunc(bp, func(z complex128) bool { return math.Abs(real(z)-x) <= tol(x) }) {
			return false
		}
	}
	return true
}

// rlswVerdict returns the components res gets right, the components the
// oracles can judge and whether Breakaway lists a point twice.
func rlswVerdict(c *rlswCase, oInt, oFlt rlswOracle) (ok, posed int, dup bool, err error) {
	sys, err := c.system()
	if err != nil {
		return 0, 0, false, err
	}
	res, rerr := RootLocus(sys, []float64{0, 1})
	if oInt.posed || oFlt.posed {
		posed |= rlswBreak
	}
	lead := c.num[len(c.num)-len(fzTrimRat(c.num))]
	neg := lead.Sign() < 0
	relDeg := c.n + c.L - (len(fzTrimRat(c.num)) - 1)
	posed |= rlswAsym | rlswArrive
	mag := 0.0
	for _, p := range c.poles {
		mag = math.Max(mag, cmplx.Abs(p))
	}
	if mag == 0 {
		mag = 1
	}
	poles := slices.Concat(c.poles, make([]complex128, c.L))
	simple := c.d == 0 && oInt.posed
	for i, p := range poles {
		for _, q := range poles[i+1:] {
			if cmplx.Abs(p-q) < 1e-3*mag {
				simple = false
			}
		}
		for _, z := range c.zeros {
			if cmplx.Abs(p-z) < 1e-3*mag {
				simple = false
			}
		}
	}
	if simple {
		posed |= rlswDepart
	}
	if rerr != nil {
		return 0, posed, false, nil
	}
	_, dup = rlswDuplicate(res.Breakaway)
	if oInt.accepts(res.Breakaway) || oFlt.accepts(res.Breakaway) {
		ok |= rlswBreak
	}
	asymOK := len(res.AsymptoteAngles) == max(relDeg, 0)
	phase0 := math.Pi
	if neg {
		phase0 = 0
	}
	for k, got := range res.AsymptoteAngles {
		if !asymOK {
			break
		}
		want := (2*float64(k)*math.Pi + phase0) / float64(relDeg)
		asymOK = math.Abs(got-want) <= 1e-9
	}
	if asymOK && relDeg > 0 {
		var sp, sz float64
		for _, p := range poles {
			sp += real(p)
		}
		for _, z := range c.zeros {
			sz += real(z)
		}
		want := (sp - sz) / float64(relDeg)
		got, cok := res.AsymptoteCentroid()
		asymOK = cok && math.Abs(got-want) <= 1e-6*math.Max(mag, math.Abs(want))
	}
	if asymOK {
		ok |= rlswAsym
	}
	if len(res.ArrivalAngles) == len(fzTrimRat(c.num))-1 {
		ok |= rlswArrive
	}
	if posed&rlswDepart != 0 {
		abs := sys
		if c.L > 0 {
			if abs, err = sys.AbsorbDelay(); err != nil {
				return ok, posed, dup, err
			}
		}
		got, err := abs.Poles()
		if err != nil {
			return ok, posed, dup, err
		}
		dok := len(got) == len(poles) && len(res.DepartureAngles) == len(poles)
		for i := 0; dok && i < len(got); i++ {
			j := 0
			for k := range poles {
				if cmplx.Abs(poles[k]-got[i]) < cmplx.Abs(poles[j]-got[i]) {
					j = k
				}
			}
			want := phase0
			for k, q := range poles {
				if k != j {
					want -= cmplx.Phase(poles[j] - q)
				}
			}
			for _, z := range c.zeros {
				want += cmplx.Phase(poles[j] - z)
			}
			diff := math.Remainder(res.DepartureAngles[i]-want, 2*math.Pi)
			dok = math.Abs(diff) <= 1e-6
		}
		if dok {
			ok |= rlswDepart
		}
	}
	return ok, posed, dup, nil
}

func (c *rlswCase) oracles() (oInt, oFlt rlswOracle) {
	lag := func(p []*big.Rat) []*big.Rat {
		out := slices.Clone(p)
		for range c.L {
			out = append(out, new(big.Rat))
		}
		return out
	}
	oInt = newRlswOracle(c.num, lag(c.den))
	den := fzCharPolyExact(c.a, c.n)
	abc := make([]*big.Rat, c.n*c.n)
	for i := range c.n {
		for j := range c.n {
			x := fzRat(c.a[i*c.n+j])
			abc[i*c.n+j] = x.Sub(x, new(big.Rat).Mul(fzRat(c.b[i]), fzRat(c.c[j])))
		}
	}
	cp := fzCharPolyRat(abc, c.n)
	num := make([]*big.Rat, c.n+1)
	for i := range num {
		num[i] = new(big.Rat).Sub(cp[i], den[i])
		num[i].Add(num[i], new(big.Rat).Mul(fzRat(c.d), den[i]))
	}
	oFlt = newRlswOracle(num, lag(den))
	return oInt, oFlt
}

const rlswBaseRev = "23be053+tfcap"

type rlswGolden struct {
	Base  string `json:"base"`
	OK    string `json:"ok"`
	Posed string `json:"posed"`
	Dup   []int  `json:"dup,omitempty"`
}

func rlswRun(t *testing.T, cases []rlswCase) (ok, posed, dups []int) {
	ok, posed = make([]int, len(cases)), make([]int, len(cases))
	for i := range cases {
		oi, of := cases[i].oracles()
		var err error
		var dup bool
		ok[i], posed[i], dup, err = rlswVerdict(&cases[i], oi, of)
		if err != nil {
			t.Fatalf("case %d: %v\n%s", i, err, &cases[i])
		}
		if dup {
			dups = append(dups, i)
		}
	}
	return ok, posed, dups
}

func rlswHex(v []int) string {
	var sb strings.Builder
	for _, x := range v {
		sb.WriteByte("0123456789abcdef"[x])
	}
	return sb.String()
}

func rlswUnhex(s string) ([]int, error) {
	out := make([]int, len(s))
	for i := range s {
		var x int
		if _, err := fmt.Sscanf(s[i:i+1], "%x", &x); err != nil {
			return nil, err
		}
		out[i] = x
	}
	return out, nil
}

// rlswUlpSensitive reports whether 1-ulp elementwise perturbations of the
// float A, B, C change the exact breakaway oracle of c: the answer is then
// below float64 resolution of the given coefficients.
func rlswUlpSensitive(c *rlswCase, seed int) bool {
	_, of := c.oracles()
	r := rand.New(rand.NewPCG(uint64(seed), 7))
	ulp := func(v []float64) []float64 {
		out := slices.Clone(v)
		for i, x := range out {
			if x == 0 {
				continue
			}
			switch r.IntN(3) {
			case 0:
				out[i] = math.Nextafter(x, math.Inf(1))
			case 1:
				out[i] = math.Nextafter(x, math.Inf(-1))
			}
		}
		return out
	}
	for range 6 {
		p := *c
		p.a, p.b, p.c = ulp(c.a), ulp(c.b), ulp(c.c)
		_, op := p.oracles()
		req := make([]complex128, len(op.required))
		for i, x := range op.required {
			req[i] = complex(x, 0)
		}
		if !op.posed || !of.posed || !of.accepts(req) {
			return true
		}
	}
	return false
}

// rlswCheck fails when a component that base got right on cases (golden
// file golden) is now wrong or errors; env names the regenerate variable.
// A Breakaway verdict listed in limits is reported, not enforced, when
// rlswUlpSensitive confirms it is a float64 precision limit. It also fails
// when a case's Breakaway lists a point twice where base (golden dup) did
// not.
func rlswCheck(t *testing.T, cases []rlswCase, golden, env string, limits map[int]string) {
	t.Helper()
	ok, posed, dups := rlswRun(t, cases)
	if path := os.Getenv(env); path != "" {
		data, err := json.Marshal(rlswGolden{Base: rlswBaseRev, OK: rlswHex(ok), Posed: rlswHex(posed), Dup: dups})
		if err != nil {
			t.Fatal(err)
		}
		if err := os.WriteFile(path, data, 0o644); err != nil {
			t.Fatal(err)
		}
		for _, m := range []int{rlswBreak, rlswAsym, rlswDepart, rlswArrive} {
			np, nok := 0, 0
			for i := range ok {
				if posed[i]&m != 0 {
					np++
					if ok[i]&m != 0 {
						nok++
					}
				}
			}
			t.Logf("component %d: posed %d, ok %d", m, np, nok)
		}
		return
	}
	data, err := os.ReadFile(golden)
	if err != nil {
		t.Fatal(err)
	}
	var g rlswGolden
	if err := json.Unmarshal(data, &g); err != nil {
		t.Fatal(err)
	}
	base, err := rlswUnhex(g.OK)
	if err != nil || len(base) != len(ok) {
		t.Fatalf("golden has %d cases, sweep %d (%v)", len(base), len(ok), err)
	}
	names := map[int]string{rlswBreak: "Breakaway", rlswAsym: "Asymptotes", rlswDepart: "DepartureAngles", rlswArrive: "ArrivalAngles count"}
	var fixed, regressed [4]int
	for i := range ok {
		for bit := range 4 {
			m := 1 << bit
			if posed[i]&m == 0 {
				continue
			}
			switch {
			case base[i]&m != 0 && ok[i]&m == 0 && m == rlswBreak && limits[i] != "" && rlswUlpSensitive(&cases[i], i):
				t.Logf("case %d: Breakaway differs from base at a precision limit: %s", i, limits[i])
			case base[i]&m != 0 && ok[i]&m == 0:
				regressed[bit]++
				if regressed[bit] <= 5 {
					sys, _ := cases[i].system()
					res, rerr := RootLocus(sys, []float64{0, 1})
					t.Errorf("case %d: %s regressed (err %v)\nres=%+v\n%s", i, names[m], rerr, res, &cases[i])
				}
			case base[i]&m == 0 && ok[i]&m != 0:
				fixed[bit]++
			}
		}
	}
	t.Logf("fixed vs base: break %d asym %d depart %d arrive %d; regressed: %v", fixed[0], fixed[1], fixed[2], fixed[3], regressed)
	k := 0
	for _, i := range dups {
		if slices.Contains(g.Dup, i) {
			continue
		}
		if k++; k <= 5 {
			sys, _ := cases[i].system()
			res, _ := RootLocus(sys, []float64{0, 1})
			t.Errorf("case %d: Breakaway %v lists a point twice\n%s", i, res.Breakaway, &cases[i])
		}
	}
}

// Verdicts right at 03edce8 but wrong on this branch (sw, dbl, ext sweeps,
// AND of arm64 and amd64) are not listed in the limit maps. None is lost
// vs 23be053 (same verdicts). The converted TF is backward-stable, not
// exact: its coefficient error vs the exact big.Rat TF of the floats is
// 0 to ~2x the spread of that TF under 1-ulp changes of A, B, C, up to
// 21-45x (sw 334 1648, ext 307) and 75-178x (ext 858). A rounding of the
// exact TF, realized by StateSpace, gets the same or a worse verdict in
// the first two groups below, so a more accurate TF cannot restore them. The defects are in rlocus.go,
// which fix/rlocus-wave22 fixes; with its rlocus.go on this branch all
// three sweeps lose none of them except limits it lists:
//   - exact multiple root of N'D−ND' (z^L or repeated pole/zero) listed
//     twice; 03edce8 split it by TF noise: sw 17 138 166 613 977 1389 1436
//     1518 1590 1648 1709 1745 1754, dbl 2 125 325 491 602 608 856, ext 27
//     1420 1751 1762.
//   - negative numerator lead, no K<0 sign handling (03edce8 got the
//     segments from a far noise zero, dropped by the tzero-parity numerator
//     cap): sw 38 (departure) 74 105 194 334 372 595 958 1187, dbl 107, ext
//     307 364 858 1081 1399 1683.
//   - one arch only, TF error ≤1.5x the 1-ulp spread (dbl 1413 bit-exact),
//     right from the rounded exact TF: precision limits of TF rounding plus
//     eig of the given realization: dbl 1413 (amd64), dbl 1452 (arm64),
//     ext 32 (arm64), ext 903 (amd64).
//
// rlswPrecisionLimits are base-correct breakaway verdicts below float64
// resolution of the given coefficients (1-ulp changes of A, B, C change the
// exact oracle; checked by rlswUlpSensitive), where base was right by luck.
var rlswPrecisionLimits = map[int]string{
	55:   "rotated double pole -1.526e-4 in a unit-norm A: exact charpoly of the floats is a complex pair (disc −5.8e-18, a 1-ulp entry change moves it ~1e-16), the converted D needs 2.1e5·u to merge it, past the u^(2/3) a resolved non-normal pair needs (wave 22 B1); oFlt only a far root 2.1e17",
	113:  "amd64: rotated triple pole -3.418e-3 in a unit-norm A: float charpoly -3.4184e-3, -3.4178e-3±3.8e-7i, oFlt unposed; 1-ulp perturbations leave it unposed or give far roots ~-1e16 (w19 #21)",
	539:  "rotated 4 real poles at 1e-5 in a unit-norm A: exact charpoly of the floats has a complex pair -5.7e-5±4.5e-5i, 1-ulp perturbations give breakaway sets [] or [-9.34e-5 -5.69e-5 -2.28e-5]",
	1012: "amd64: rotated triple pole -6.1035e-4 in a unit-norm A: float charpoly -6.1050e-4, -6.1028e-4±1.3e-7i; 1-ulp perturbations give [], a close pair -6.10355e-4/-6.10348e-4 or a far root -2.7e16 (w19 #21)",
	1720: "rotated triple pole at -3.357e-4: float charpoly splits it by 1.9e-6; 1-ulp perturbations give [] or a pair at -3.3569e-4 plus a far root",
}

// TestRootLocusSweepNoRegression fails when a component that base got
// right on the frozen 2000-case sweep is now wrong (or errors).
func TestRootLocusSweepNoRegression(t *testing.T) {
	if testing.Short() {
		t.Skip("frozen 2000-case sweep")
	}
	skipSlowUnderRace(t)
	rlswCheck(t, rlswGenerate(), "testdata/rlocus_sweep_base.json", "RLOCUS_SWEEP_WRITE", rlswPrecisionLimits)
}
