package controlsys

import (
	"math"
	"math/big"
	"math/rand/v2"
	"slices"
)

// pencilDetPoly returns the exact coefficients (ascending) of
// det(A − λE) for row-major rational A, E: fraction-free Bareiss
// determinants at λ = 0..n, interpolated by Newton divided differences.
func pencilDetPoly(n int, a, e []*big.Rat) []*big.Rat {
	den := big.NewInt(1)
	for _, x := range append(slices.Clone(a), e...) {
		if d := x.Denom(); d.Cmp(den) > 0 {
			den = new(big.Int).Set(d)
		}
	}
	scale := new(big.Rat).SetInt(den)
	toInt := func(x *big.Rat) *big.Int {
		v := new(big.Rat).Mul(x, scale)
		if !v.IsInt() {
			panic("pencilDetPoly: non-integer scaled entry")
		}
		return new(big.Int).Set(v.Num())
	}
	ai, ei := make([]*big.Int, n*n), make([]*big.Int, n*n)
	for i := range n * n {
		ai[i], ei[i] = toInt(a[i]), toInt(e[i])
	}
	vals := make([]*big.Rat, n+1)
	for j := range n + 1 {
		m := make([]*big.Int, n*n)
		lam := big.NewInt(int64(j))
		for i := range n * n {
			m[i] = new(big.Int).Sub(ai[i], new(big.Int).Mul(lam, ei[i]))
		}
		vals[j] = new(big.Rat).SetInt(pencilBareissDet(n, m))
	}
	dd := slices.Clone(vals)
	for lvl := 1; lvl <= n; lvl++ {
		for i := n; i >= lvl; i-- {
			d := new(big.Rat).Sub(dd[i], dd[i-1])
			dd[i] = d.Quo(d, new(big.Rat).SetInt64(int64(lvl)))
		}
	}
	coef := []*big.Rat{new(big.Rat).Set(dd[n])}
	for i := n - 1; i >= 0; i-- {
		x := new(big.Rat).SetInt64(int64(i))
		next := make([]*big.Rat, len(coef)+1)
		for k := range next {
			next[k] = new(big.Rat)
			if k > 0 {
				next[k].Add(next[k], coef[k-1])
			}
			if k < len(coef) {
				next[k].Sub(next[k], new(big.Rat).Mul(x, coef[k]))
			}
		}
		next[0].Add(next[0], dd[i])
		coef = next
	}
	return coef
}

// pencilBareissDet returns the determinant of the row-major n×n integer matrix m
// (modified in place).
func pencilBareissDet(n int, m []*big.Int) *big.Int {
	sign := 1
	prev := big.NewInt(1)
	for k := range n - 1 {
		if m[k*n+k].Sign() == 0 {
			p := -1
			for i := k + 1; i < n; i++ {
				if m[i*n+k].Sign() != 0 {
					p = i
					break
				}
			}
			if p < 0 {
				return new(big.Int)
			}
			for j := range n {
				m[k*n+j], m[p*n+j] = m[p*n+j], m[k*n+j]
			}
			sign = -sign
		}
		for i := k + 1; i < n; i++ {
			for j := k + 1; j < n; j++ {
				t := new(big.Int).Mul(m[i*n+j], m[k*n+k])
				t.Sub(t, new(big.Int).Mul(m[i*n+k], m[k*n+j]))
				m[i*n+j] = t.Quo(t, prev)
			}
		}
		prev = m[k*n+k]
	}
	d := new(big.Int).Set(m[n*n-1])
	if sign < 0 {
		d.Neg(d)
	}
	return d
}

const polyRootPrec = 1024

type bigCplx struct{ re, im *big.Float }

func bcNew() bigCplx {
	return bigCplx{new(big.Float).SetPrec(polyRootPrec), new(big.Float).SetPrec(polyRootPrec)}
}

func bcOf(re, im float64) bigCplx {
	z := bcNew()
	z.re.SetFloat64(re)
	z.im.SetFloat64(im)
	return z
}

func bcAdd(x, y bigCplx) bigCplx {
	z := bcNew()
	z.re.Add(x.re, y.re)
	z.im.Add(x.im, y.im)
	return z
}

func bcSub(x, y bigCplx) bigCplx {
	z := bcNew()
	z.re.Sub(x.re, y.re)
	z.im.Sub(x.im, y.im)
	return z
}

func bcMul(x, y bigCplx) bigCplx {
	z, t := bcNew(), new(big.Float).SetPrec(polyRootPrec)
	z.re.Mul(x.re, y.re)
	z.re.Sub(z.re, t.Mul(x.im, y.im))
	z.im.Mul(x.re, y.im)
	z.im.Add(z.im, t.Mul(x.im, y.re))
	return z
}

func bcAbs2(x bigCplx) *big.Float {
	s, t := new(big.Float).SetPrec(polyRootPrec), new(big.Float).SetPrec(polyRootPrec)
	s.Mul(x.re, x.re)
	return s.Add(s, t.Mul(x.im, x.im))
}

func bcQuo(x, y bigCplx) bigCplx {
	d := bcAbs2(y)
	z, t := bcNew(), new(big.Float).SetPrec(polyRootPrec)
	z.re.Mul(x.re, y.re)
	z.re.Add(z.re, t.Mul(x.im, y.im))
	z.re.Quo(z.re, d)
	z.im.Mul(x.im, y.re)
	z.im.Sub(z.im, t.Mul(x.re, y.im))
	z.im.Quo(z.im, d)
	return z
}

func (z bigCplx) complex128() complex128 {
	r, _ := z.re.Float64()
	i, _ := z.im.Float64()
	return complex(r, i)
}

// polyRoots returns all roots of the polynomial with exact coefficients
// coef (ascending, nonzero leading) by Aberth–Ehrlich iteration at
// polyRootPrec bits from Newton-polygon starting radii, and whether every
// correction fell below 2⁻⁹⁶⁰ relative.
func polyRoots(coef []*big.Rat) ([]bigCplx, bool) {
	m := len(coef) - 1
	c := make([]bigCplx, m+1)
	for i := range c {
		c[i] = bcNew()
		c[i].re.SetRat(coef[i])
	}
	logc := make([]float64, m+1)
	for i := range c {
		if c[i].re.Sign() == 0 {
			logc[i] = math.Inf(-1)
			continue
		}
		mant := new(big.Float)
		exp := c[i].re.MantExp(mant)
		f, _ := mant.Float64()
		logc[i] = math.Log2(math.Abs(f)) + float64(exp)
	}
	var hull []int
	for i := range logc {
		if math.IsInf(logc[i], -1) {
			continue
		}
		for len(hull) >= 2 {
			a, b := hull[len(hull)-2], hull[len(hull)-1]
			if (logc[b]-logc[a])*float64(i-a) <= (logc[i]-logc[a])*float64(b-a) {
				hull = hull[:len(hull)-1]
				continue
			}
			break
		}
		hull = append(hull, i)
	}
	z := make([]bigCplx, 0, m)
	for h := 1; h < len(hull); h++ {
		a, b := hull[h-1], hull[h]
		r := math.Exp2((logc[a] - logc[b]) / float64(b-a))
		for j := a; j < b; j++ {
			th := 2*math.Pi*float64(len(z))/float64(m) + 0.4
			z = append(z, bcOf(r*math.Cos(th), r*math.Sin(th)))
		}
	}
	for len(z) < m {
		z = append(z, bcOf(1, float64(len(z))))
	}
	eval := func(x bigCplx) (p, dp bigCplx) {
		p, dp = c[m], bcNew()
		for i := m - 1; i >= 0; i-- {
			dp = bcAdd(bcMul(dp, x), p)
			p = bcAdd(bcMul(p, x), c[i])
		}
		return p, dp
	}
	stop := new(big.Float).SetMantExp(big.NewFloat(1), -1920)
	one := bcOf(1, 0)
	for range 2000 {
		done := true
		for i := range m {
			p, dp := eval(z[i])
			if p.re.Sign() == 0 && p.im.Sign() == 0 {
				continue
			}
			ratio := bcQuo(p, dp)
			sum := bcNew()
			for j := range m {
				if j != i {
					sum = bcAdd(sum, bcQuo(one, bcSub(z[i], z[j])))
				}
			}
			w := bcQuo(ratio, bcSub(one, bcMul(ratio, sum)))
			z[i] = bcSub(z[i], w)
			if bcAbs2(w).Cmp(new(big.Float).Mul(stop, bcAbs2(z[i]))) > 0 {
				done = false
			}
		}
		if done {
			return z, true
		}
	}
	return z, false
}

// poleCountSensitivity is the 2⁻¹⁰⁰ sensitivity classifier of the stored
// pencil's det(A − λE) roots: the data are perturbed entrywise by the
// relative factor 1 ± 2⁻¹⁰⁰ (fixed PCG signs), and each root's sensitivity
// is S = |Δλ|/(|λ|·2⁻¹⁰⁰), matching every root to the nearest perturbed one.
type poleCountSensitivity struct {
	Roots []complex128
	S     []float64
	OK    bool
}

// trueCount returns the number of roots with S·2⁻⁵³ < 1.
func (s poleCountSensitivity) trueCount() int {
	n := 0
	for _, v := range s.S {
		if v*0x1p-53 < 1 {
			n++
		}
	}
	return n
}

// bandCounts splits the roots by the approach memo's band: S < 1e10 a true
// pole, S > 1e14 a rounding-created chain root, anything between gray.
func (s poleCountSensitivity) bandCounts() (finite, gray, smear int) {
	for _, v := range s.S {
		switch {
		case v < 1e10:
			finite++
		case v > 1e14:
			smear++
		default:
			gray++
		}
	}
	return finite, gray, smear
}

func classifyPencilRoots(n int, a, e []float64) poleCountSensitivity {
	rat := func(x []float64, pert *rand.Rand) []*big.Rat {
		out := make([]*big.Rat, len(x))
		for i, v := range x {
			out[i] = new(big.Rat).SetFloat64(v)
			if pert != nil {
				f := new(big.Rat).SetInt64(1)
				d := new(big.Rat).SetFrac(big.NewInt(1), new(big.Int).Lsh(big.NewInt(1), 100))
				if pert.IntN(2) == 0 {
					f.Add(f, d)
				} else {
					f.Sub(f, d)
				}
				out[i].Mul(out[i], f)
			}
		}
		return out
	}
	rng := rand.New(rand.NewPCG(100, 53))
	z0, ok0 := polyRoots(pencilDetPoly(n, rat(a, nil), rat(e, nil)))
	z1, ok1 := polyRoots(pencilDetPoly(n, rat(a, rng), rat(e, rng)))
	out := poleCountSensitivity{OK: ok0 && ok1}
	used := make([]bool, len(z1))
	for _, r := range z0 {
		best, bd := -1, (*big.Float)(nil)
		for j, q := range z1 {
			if used[j] {
				continue
			}
			if d := bcAbs2(bcSub(r, q)); best < 0 || d.Cmp(bd) < 0 {
				best, bd = j, d
			}
		}
		used[best] = true
		rel := new(big.Float).SetPrec(polyRootPrec).Quo(bd, bcAbs2(r))
		f, _ := rel.Float64()
		out.Roots = append(out.Roots, r.complex128())
		out.S = append(out.S, math.Sqrt(f)*0x1p100)
	}
	return out
}
