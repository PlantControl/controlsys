package controlsys

import (
	"fmt"
	"math"
	"unsafe"

	"plantcontrol.org/v1/gonum/mat"
)

// Poly represents a polynomial in descending power:
//
//	Poly{1, -3, 2} = s² - 3s + 2
//
// Methods are coefficient-wise and never trim leading zeros: Poly{0, 1} has
// formal degree 1 and differs from Poly{1} under Equal. Only Roots trims
// leading zeros. Poly{} and any all-zero Poly represent the zero polynomial.
type Poly []float64

// Degree returns the formal degree len(p)-1, counting leading zeros; it is
// -1 for Poly{}.
func (p Poly) Degree() int { return len(p) - 1 }

// Eval returns p(s) by Horner's rule; the empty Poly evaluates to 0.
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

// Mul returns the product p·q in a new Poly; it is Poly{} if either factor is
// empty.
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

// Add returns p+q in a new Poly, aligning trailing (constant) coefficients.
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

// IsMonic reports whether the first coefficient is exactly 1.
func (p Poly) IsMonic() bool {
	return len(p) > 0 && p[0] == 1
}

// Monic returns p divided by its first coefficient. An empty Poly or a zero
// first coefficient returns ErrInvalidArgument.
func (p Poly) Monic() (Poly, error) {
	if len(p) == 0 {
		return nil, fmt.Errorf("Poly.Monic: empty polynomial: %w", ErrInvalidArgument)
	}
	if p[0] == 0 {
		return nil, fmt.Errorf("Poly.Monic: zero leading coefficient: %w", ErrInvalidArgument)
	}
	r := make(Poly, len(p))
	inv := 1.0 / p[0]
	for i, v := range p {
		r[i] = v * inv
	}
	return r, nil
}

// Scale returns s·p in a new Poly.
func (p Poly) Scale(s float64) Poly {
	r := make(Poly, len(p))
	for i, v := range p {
		r[i] = v * s
	}
	return r
}

// MulTo stores p·q in dst, growing it when its capacity is too small, and
// returns the result. dst may share storage with p or q; the product is then
// computed in a new slice.
func (p Poly) MulTo(dst Poly, q Poly) Poly {
	if len(p) == 0 || len(q) == 0 {
		return dst[:0]
	}
	need := len(p) + len(q) - 1
	if cap(dst) < need || polyOverlap(dst, p) || polyOverlap(dst, q) {
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

// AddTo stores p+q in dst, growing it when its capacity is too small, and
// returns the result. dst may share storage with p or q; the sum is then
// computed in a new slice.
func (p Poly) AddTo(dst Poly, q Poly) Poly {
	n := len(p)
	m := len(q)
	if n == 0 && m == 0 {
		return dst[:0]
	}
	size := max(m, n)
	if cap(dst) < size || polyOverlap(dst, p) || polyOverlap(dst, q) {
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

// polyOverlap reports whether the full capacity of dst shares memory with src.
func polyOverlap(dst, src Poly) bool {
	if cap(dst) == 0 || len(src) == 0 {
		return false
	}
	const size = unsafe.Sizeof(float64(0))
	d0 := uintptr(unsafe.Pointer(unsafe.SliceData(dst)))
	s0 := uintptr(unsafe.Pointer(unsafe.SliceData(src)))
	return d0 < s0+uintptr(len(src))*size && s0 < d0+uintptr(cap(dst))*size
}

// ScaleTo stores s·p in dst, growing it when its capacity is too small, and
// returns the result. dst may be p itself (in-place scaling).
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

// Roots returns the roots of p, as MATLAB roots, after trimming leading zeros.
// A nonzero constant has no roots. NaN or Inf coefficients and the zero
// polynomial (every s is a root) return ErrInvalidArgument.
func (p Poly) Roots() ([]complex128, error) {
	if err := requireFinite("Poly.Roots", "coefficient", p...); err != nil {
		return nil, err
	}
	start := 0
	for start < len(p) && p[start] == 0 {
		start++
	}
	p = p[start:]
	if len(p) == 0 {
		return nil, fmt.Errorf("Poly.Roots: zero polynomial: %w", ErrInvalidArgument)
	}

	deg := len(p) - 1
	if deg == 0 {
		return []complex128{}, nil
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
		return nil, fmt.Errorf("Poly.Roots: companion eigenvalues: %w", ErrSchurFailed)
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

// Sub returns p-q in a new Poly, aligning trailing (constant) coefficients.
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

// Derivative returns dp/ds; the derivative of a constant or empty Poly is
// Poly{0}.
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

// Equal reports whether p and q have the same length and every coefficient
// pair differs by at most tol. It does not trim leading zeros.
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
