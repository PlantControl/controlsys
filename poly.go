package controlsys

import (
	"errors"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// Poly represents a polynomial in descending power:
//
//	Poly{1, -3, 2} = s² - 3s + 2
type Poly []float64

func (p Poly) Degree() int { return len(p) - 1 }

func (p Poly) Eval(s complex128) complex128 {
	if len(p) == 0 {
		return 0
	}
	result := complex(p[0], 0)
	for i := 1; i < len(p); i++ {
		result = result*s + complex(p[i], 0)
	}
	return result
}

func (p Poly) Mul(q Poly) Poly {
	if len(p) == 0 || len(q) == 0 {
		return Poly{}
	}
	r := make(Poly, len(p)+len(q)-1)
	for i, pi := range p {
		for j, qj := range q {
			r[i+j] += pi * qj
		}
	}
	return r
}

func (p Poly) Add(q Poly) Poly {
	n := len(p)
	m := len(q)
	if n == 0 {
		r := make(Poly, m)
		copy(r, q)
		return r
	}
	if m == 0 {
		r := make(Poly, n)
		copy(r, p)
		return r
	}
	size := max(m, n)
	r := make(Poly, size)
	for i := range r {
		var pv, qv float64
		if i >= size-n {
			pv = p[i-(size-n)]
		}
		if i >= size-m {
			qv = q[i-(size-m)]
		}
		r[i] = pv + qv
	}
	return r
}

func (p Poly) IsMonic() bool {
	return len(p) > 0 && p[0] == 1
}

func (p Poly) Monic() (Poly, error) {
	if len(p) == 0 {
		return nil, errors.New("controlsys: monic of empty polynomial")
	}
	if p[0] == 0 {
		return nil, errors.New("controlsys: monic with zero leading coefficient")
	}
	r := make(Poly, len(p))
	inv := 1.0 / p[0]
	for i, v := range p {
		r[i] = v * inv
	}
	return r, nil
}

func (p Poly) Scale(s float64) Poly {
	r := make(Poly, len(p))
	for i, v := range p {
		r[i] = v * s
	}
	return r
}

func (p Poly) MulTo(dst Poly, q Poly) Poly {
	if len(p) == 0 || len(q) == 0 {
		return dst[:0]
	}
	need := len(p) + len(q) - 1
	if cap(dst) < need {
		dst = make(Poly, need)
	} else {
		dst = dst[:need]
	}
	for i := range dst {
		dst[i] = 0
	}
	for i, pi := range p {
		for j, qj := range q {
			dst[i+j] += pi * qj
		}
	}
	return dst
}

func (p Poly) AddTo(dst Poly, q Poly) Poly {
	n := len(p)
	m := len(q)
	if n == 0 && m == 0 {
		return dst[:0]
	}
	size := max(m, n)
	if cap(dst) < size {
		dst = make(Poly, size)
	} else {
		dst = dst[:size]
	}
	for i := range dst {
		var pv, qv float64
		if i >= size-n {
			pv = p[i-(size-n)]
		}
		if i >= size-m {
			qv = q[i-(size-m)]
		}
		dst[i] = pv + qv
	}
	return dst
}

func (p Poly) ScaleTo(dst Poly, s float64) Poly {
	if cap(dst) < len(p) {
		dst = make(Poly, len(p))
	} else {
		dst = dst[:len(p)]
	}
	for i, v := range p {
		dst[i] = v * s
	}
	return dst
}

func (p Poly) Roots() ([]complex128, error) {
	start := 0
	for start < len(p) && p[start] == 0 {
		start++
	}
	p = p[start:]

	deg := len(p) - 1
	if deg <= 0 {
		return nil, nil
	}

	if deg == 1 {
		return []complex128{complex(-p[1]/p[0], 0)}, nil
	}

	lead := p[0]
	data := make([]float64, deg*deg)
	for i := range deg {
		data[i*deg+deg-1] = -p[deg-i] / lead
		if i > 0 {
			data[i*deg+i-1] = 1
		}
	}
	comp := mat.NewDense(deg, deg, data)

	var eig mat.Eigen
	if !eig.Factorize(comp, mat.EigenNone) {
		return nil, ErrSingularTransform
	}
	roots := eig.Values(nil)

	snapTol := 100 * eps()
	for i, r := range roots {
		if imag(r) != 0 && math.Abs(imag(r)) < math.Abs(real(r))*snapTol {
			roots[i] = complex(real(r), 0)
		}
	}

	sortZeros(roots)
	return roots, nil
}

func (p Poly) Sub(q Poly) Poly {
	n := len(p)
	m := len(q)
	if n == 0 {
		r := make(Poly, m)
		for i, v := range q {
			r[i] = -v
		}
		return r
	}
	if m == 0 {
		r := make(Poly, n)
		copy(r, p)
		return r
	}
	size := max(m, n)
	r := make(Poly, size)
	for i := range r {
		var pv, qv float64
		if i >= size-n {
			pv = p[i-(size-n)]
		}
		if i >= size-m {
			qv = q[i-(size-m)]
		}
		r[i] = pv - qv
	}
	return r
}

func (p Poly) Derivative() Poly {
	if len(p) <= 1 {
		return Poly{0}
	}
	deg := len(p) - 1
	r := make(Poly, deg)
	for i := range deg {
		r[i] = p[i] * float64(deg-i)
	}
	return r
}

func (p Poly) Equal(q Poly, tol float64) bool {
	if len(p) != len(q) {
		return false
	}
	for i := range p {
		if math.Abs(p[i]-q[i]) > tol {
			return false
		}
	}
	return true
}
