package controlsys

import (
	"math"
	"math/cmplx"
)

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

func (x ddComplex) abs() float64 { return math.Hypot(x.re.hi, x.im.hi) }

func (x ddComplex) complex128() complex128 { return complex(x.re.hi+x.re.lo, x.im.hi+x.im.lo) }

const (
	// ddOpErr bounds the relative error of one double-double operation,
	// ~16× above its 2⁻¹⁰⁴.
	ddOpErr = 0x1p-100
	// ddPivotMargin is the pivot-to-error ratio a pivot below the
	// normwise floor needs: 2⁹⁰, the relative accuracy solve assumes.
	ddPivotMargin = 0x1p90
)

// ddSISOResponse evaluates the SISO response C(sE−A)⁻¹B + D of sys at pt in
// double-double arithmetic, as (sE−A)⁻¹ = −q·(pE+qA)⁻¹, whose entries are
// formed from the double p and q without rounding. It is the fallback where
// the double solve is singular to working precision but the model is not,
// as next to a pole cluster k/(z+1)ⁿ split by coefficient rounding, where
// |L| still crosses 1. ok is false at a pivot below 2⁻¹⁰⁰ of the scale
// unless it exceeds 2⁹⁰ times its running forward-error bound: beside an
// integrator triple with sub-1e-6 real poles, det(sI−A) ~ 1e-31 at s = 1e-6j
// is far below the floor yet exact to ~2⁻¹⁰⁰, while at a pole to dd
// precision the pivot is within its bound.
func ddSISOResponse(sys *System, pt frequencyPoint) (h complex128, ok bool) {
	h, _, _, ok = ddSISOSolve(sys, pt, false)
	return h, ok
}

// ddSISOResponseNoise is ddSISOResponse that also returns the |L| below
// which L is indistinguishable from zero: solve, for this solve's error,
// 2⁻⁹⁰ of |D| + |q|·‖C‖‖x‖ with x = (pE+qA)⁻¹B, which a cancellation of C·x
// beyond double-double leaves, as beside a zero cluster k·(z+1)⁴; and
// coeffs, for an ulp of each entry of the realization, the first-order bound
// eps·(|D| + |q|·(|C||x| + |y|ᵀ|B| + |y|ᵀ(|p||E| + |q||A|)|x|)), yᵀ = C(pE+qA)⁻¹,
// as of C = num − D·den rounded when D ≠ 0. coeffs is NaN when the double
// transposed solve for y fails.
func ddSISOResponseNoise(sys *System, pt frequencyPoint) (h complex128, solve, coeffs float64, ok bool) {
	return ddSISOSolve(sys, pt, true)
}

func ddSISOSolve(sys *System, pt frequencyPoint, withNoise bool) (h complex128, solve, coeffs float64, ok bool) {
	n, _ := sys.A.Dims()
	d := sys.D.At(0, 0)
	if n == 0 {
		return complex(d, 0), 0x1p-90 * math.Abs(d), eps() * math.Abs(d), true
	}
	ar, br, cr := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	pr, pi, qr, qi := real(pt.p), imag(pt.p), real(pt.q), imag(pt.q)
	m := make([]ddComplex, n*n)
	mErr := make([]float64, n*n)
	x := make([]ddComplex, n)
	var mScale float64
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
			mScale = max(mScale, m[i*n+j].abs1())
			ae, aa := math.Abs(e), math.Abs(a)
			mErr[i*n+j] = ddOpErr * ((math.Abs(pr)+math.Abs(pi))*ae + (math.Abs(qr)+math.Abs(qi))*aa)
		}
		x[i] = ddComplex{re: dd{br.Data[i*br.Stride], 0}}
	}
	var mt []complex128
	if withNoise {
		mt = make([]complex128, n*n)
		for i := range n {
			for j := range n {
				mt[j*n+i] = m[i*n+j].complex128()
			}
		}
	}
	tol := float64(n) * mScale * 0x1p-100
	for k := range n {
		piv, best := k, m[k*n+k].abs1()
		for i := k + 1; i < n; i++ {
			if v := m[i*n+k].abs1(); v > best {
				piv, best = i, v
			}
		}
		if !(best > tol || best > ddPivotMargin*mErr[piv*n+k]) {
			return 0, 0, 0, false
		}
		if piv != k {
			for j := range n {
				m[k*n+j], m[piv*n+j] = m[piv*n+j], m[k*n+j]
				mErr[k*n+j], mErr[piv*n+j] = mErr[piv*n+j], mErr[k*n+j]
			}
			x[k], x[piv] = x[piv], x[k]
		}
		pivMod := m[k*n+k].abs()
		for i := k + 1; i < n; i++ {
			mik, eik := m[i*n+k].abs(), mErr[i*n+k]
			if mik == 0 && eik == 0 {
				continue
			}
			fa := mik / pivMod
			fErr := (eik+fa*mErr[k*n+k])/pivMod + ddOpErr*fa
			for j := k + 1; j < n; j++ {
				mkj := m[k*n+j].abs1()
				mErr[i*n+j] += fa*mErr[k*n+j] + fErr*mkj + ddOpErr*(m[i*n+j].abs1()+fa*mkj)
			}
			if mik == 0 {
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
	h = y.mul(q).add(ddComplex{re: dd{d, 0}}).complex128()
	ap, aq := cmplx.Abs(pt.p), cmplx.Abs(pt.q)
	var cNorm, xNorm float64
	for i := range n {
		xi := x[i].complex128()
		cNorm += cr.Data[i] * cr.Data[i]
		xNorm += real(xi)*real(xi) + imag(xi)*imag(xi)
	}
	solve = 0x1p-90 * (aq*math.Sqrt(cNorm*xNorm) + math.Abs(d))
	coeffs = math.NaN()
	piv := make([]int, n)
	if withNoise && cLUFactor(mt, piv, n) {
		yt := make([]complex128, n)
		for i := range n {
			yt[i] = complex(cr.Data[i], 0)
		}
		cLUSolve(mt, piv, yt, n, 1)
		var sum float64
		for i := range n {
			yi := cmplx.Abs(yt[i])
			sum += math.Abs(cr.Data[i])*cmplx.Abs(x[i].complex128()) + yi*math.Abs(br.Data[i*br.Stride])
			for j := range n {
				var e float64
				if sys.E != nil {
					e = sys.E.At(i, j)
				} else if i == j {
					e = 1
				}
				sum += yi * (ap*math.Abs(e) + aq*math.Abs(ar.Data[i*ar.Stride+j])) * cmplx.Abs(x[j].complex128())
			}
		}
		coeffs = eps() * (aq*sum + math.Abs(d))
	}
	if math.IsNaN(real(h)) || math.IsInf(real(h), 0) || math.IsNaN(imag(h)) || math.IsInf(imag(h), 0) || math.IsNaN(solve) {
		return 0, 0, 0, false
	}
	return h, solve, coeffs, true
}
