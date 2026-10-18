package controlsys

import "math"

// dd is an unevaluated double-double sum hi + lo, |lo| ≤ ulp(hi)/2.
type dd struct{ hi, lo float64 }

func twoSum(a, b float64) (s, e float64) {
	s = a + b
	bb := s - a
	return s, (a - (s - bb)) + (b - bb)
}

func quickTwoSum(a, b float64) dd {
	s := a + b
	return dd{s, b - (s - a)}
}

func ddProd(a, b float64) dd {
	p := float64(a * b)
	return dd{p, math.FMA(a, b, -p)}
}

func (x dd) add(y dd) dd {
	s, e := twoSum(x.hi, y.hi)
	t, f := twoSum(x.lo, y.lo)
	r := quickTwoSum(s, e+t)
	return quickTwoSum(r.hi, r.lo+f)
}

func (x dd) neg() dd { return dd{-x.hi, -x.lo} }

func (x dd) mul(y dd) dd {
	p := ddProd(x.hi, y.hi)
	return quickTwoSum(p.hi, p.lo+(x.hi*y.lo+x.lo*y.hi))
}

func (x dd) div(y dd) dd {
	q1 := x.hi / y.hi
	r := x.add(y.mul(dd{q1, 0}).neg())
	q2 := r.hi / y.hi
	r = r.add(y.mul(dd{q2, 0}).neg())
	q3 := r.hi / y.hi
	s := quickTwoSum(q1, q2)
	return s.add(dd{q3, 0})
}

type ddComplex struct{ re, im dd }

func (x ddComplex) add(y ddComplex) ddComplex { return ddComplex{x.re.add(y.re), x.im.add(y.im)} }

func (x ddComplex) sub(y ddComplex) ddComplex {
	return ddComplex{x.re.add(y.re.neg()), x.im.add(y.im.neg())}
}

func (x ddComplex) mul(y ddComplex) ddComplex {
	return ddComplex{x.re.mul(y.re).add(x.im.mul(y.im).neg()), x.re.mul(y.im).add(x.im.mul(y.re))}
}

func (x ddComplex) div(y ddComplex) ddComplex {
	n := y.re.mul(y.re).add(y.im.mul(y.im))
	t := x.mul(ddComplex{y.re, y.im.neg()})
	return ddComplex{t.re.div(n), t.im.div(n)}
}

func (x ddComplex) abs1() float64 { return math.Abs(x.re.hi) + math.Abs(x.im.hi) }

func (x ddComplex) complex128() complex128 { return complex(x.re.hi+x.re.lo, x.im.hi+x.im.lo) }

// ddSISOResponse evaluates the SISO response C(sE−A)⁻¹B + D of sys at pt in
// double-double arithmetic, as (sE−A)⁻¹ = −q·(pE+qA)⁻¹, whose entries are
// formed from the double p and q without rounding. It is the fallback where
// the double solve is singular to working precision but the model is not,
// as next to a pole cluster k/(z+1)ⁿ split by coefficient rounding, where
// |L| still crosses 1. ok is false at a pivot below 2⁻¹⁰⁰ of the scale.
func ddSISOResponse(sys *System, pt frequencyPoint) (h complex128, ok bool) {
	n, _ := sys.A.Dims()
	if n == 0 {
		return complex(sys.D.At(0, 0), 0), true
	}
	ar, br, cr := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	pr, pi, qr, qi := real(pt.p), imag(pt.p), real(pt.q), imag(pt.q)
	m := make([]ddComplex, n*n)
	x := make([]ddComplex, n)
	var scale float64
	for i := range n {
		for j := range n {
			a := ar.Data[i*ar.Stride+j]
			var e float64
			if sys.E != nil {
				e = sys.E.At(i, j)
			} else if i == j {
				e = 1
			}
			m[i*n+j] = ddComplex{ddProd(pr, e).add(ddProd(qr, a)), ddProd(pi, e).add(ddProd(qi, a))}
			scale = max(scale, m[i*n+j].abs1())
		}
		x[i] = ddComplex{re: dd{br.Data[i*br.Stride], 0}}
	}
	tol := float64(n) * scale * 0x1p-100
	for k := range n {
		piv, best := k, m[k*n+k].abs1()
		for i := k + 1; i < n; i++ {
			if v := m[i*n+k].abs1(); v > best {
				piv, best = i, v
			}
		}
		if !(best > tol) {
			return 0, false
		}
		if piv != k {
			for j := range n {
				m[k*n+j], m[piv*n+j] = m[piv*n+j], m[k*n+j]
			}
			x[k], x[piv] = x[piv], x[k]
		}
		for i := k + 1; i < n; i++ {
			if m[i*n+k].abs1() == 0 {
				continue
			}
			f := m[i*n+k].div(m[k*n+k])
			for j := k + 1; j < n; j++ {
				m[i*n+j] = m[i*n+j].sub(f.mul(m[k*n+j]))
			}
			x[i] = x[i].sub(f.mul(x[k]))
		}
	}
	var y ddComplex
	for i := n - 1; i >= 0; i-- {
		s := x[i]
		for j := i + 1; j < n; j++ {
			s = s.sub(m[i*n+j].mul(x[j]))
		}
		x[i] = s.div(m[i*n+i])
		y = y.add(ddComplex{re: dd{cr.Data[i], 0}}.mul(x[i]))
	}
	q := ddComplex{dd{-qr, 0}, dd{-qi, 0}}
	h = y.mul(q).complex128() + complex(sys.D.At(0, 0), 0)
	if math.IsNaN(real(h)) || math.IsInf(real(h), 0) || math.IsNaN(imag(h)) || math.IsInf(imag(h), 0) {
		return 0, false
	}
	return h, true
}
