package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// hessenbergSweep evaluates C(sI-A)^{-1}B + D over many frequencies after a
// single orthogonal reduction A = Q H Qᵀ (Laub 1981). Each point then costs
// one O(n²m) Hessenberg solve, and stays backward stable where a polynomial
// transfer-function form loses accuracy on high-order or clustered poles.
type hessenbergSweep struct {
	n, m, p int
	h       []float64
	bt      []float64
	ct      []float64
	d       []float64
	dStride int
	hMax    float64
	u       []complex128
	inv     []complex128
	rhs     []complex128 // column-major n×m
}

func newHessenbergSweep(sys *System, n, m, p int) *hessenbergSweep {
	lwork := max(1, n, m, p)
	f := make([]float64, n*n+n*m+p*n+2*n+lwork)
	c := make([]complex128, n*n+n+n*m)
	hs := &hessenbergSweep{
		n:   n,
		m:   m,
		p:   p,
		h:   f[: n*n : n*n],
		bt:  f[n*n : n*n+n*m : n*n+n*m],
		ct:  f[n*n+n*m : n*n+n*m+p*n : n*n+n*m+p*n],
		u:   c[: n*n : n*n],
		inv: c[n*n : n*n+n : n*n+n],
		rhs: c[n*n+n:],
	}
	if sys.D != nil {
		raw := sys.D.RawMatrix()
		hs.d, hs.dStride = raw.Data, raw.Stride
	}
	if n == 0 {
		return hs
	}
	a := sys.A.RawMatrix()
	copyStrided(hs.h, n, a.Data, a.Stride, n, n)
	if m > 0 {
		b := sys.B.RawMatrix()
		copyStrided(hs.bt, m, b.Data, b.Stride, n, m)
	}
	if p > 0 {
		c := sys.C.RawMatrix()
		copyStrided(hs.ct, n, c.Data, c.Stride, p, n)
	}
	scratch := f[n*n+n*m+p*n:]
	hs.balance(scratch[:n])
	if n >= 3 {
		hs.reduce(scratch[n:2*n-1], scratch[2*n:])
	}
	for _, v := range hs.h {
		hs.hMax = max(hs.hMax, math.Abs(v))
	}
	return hs
}

// reduce applies A = Q H Qᵀ to (A, B, C) with the minimal (unblocked)
// workspace; models here are small enough that blocking only costs memory.
func (hs *hessenbergSweep) reduce(tau, work []float64) {
	n, m, p := hs.n, hs.m, hs.p
	lwork := len(work)
	impl.Dgehrd(n, 0, n-1, hs.h, n, tau, work, lwork)
	if m > 0 {
		impl.Dormhr(blas.Left, blas.Trans, n, m, 0, n-1, hs.h, n, tau, hs.bt, m, work, lwork)
	}
	if p > 0 {
		impl.Dormhr(blas.Right, blas.NoTrans, p, n, 0, n-1, hs.h, n, tau, hs.ct, n, work, lwork)
	}
	for i := 2; i < n; i++ {
		clear(hs.h[i*n : i*n+i-1])
	}
}

// crecip skips runtime complex division's overflow scaling when |z|² is
// comfortably representable.
func crecip(z complex128) complex128 {
	re, im := real(z), imag(z)
	d := re*re + im*im
	if d < 0x1p-900 || d > 0x1p900 {
		return 1 / z
	}
	return complex(re/d, -im/d)
}

func cabs1(z complex128) float64 { return math.Abs(real(z)) + math.Abs(imag(z)) }

// evalInto factors sI-H by Gaussian elimination with adjacent-row pivoting.
// Row k of u holds the not-yet-pivoted row until step k settles it, so the
// pencil is never formed.
func (hs *hessenbergSweep) evalInto(s complex128, dst []complex128) error {
	n, m, p := hs.n, hs.m, hs.p
	if n == 0 {
		copyRealMatrixToComplex(dst, hs.d, hs.dStride, p, m)
		return nil
	}
	h, u, b, inv := hs.h, hs.u, hs.rhs, hs.inv
	for i := range n {
		for j := range m {
			b[j*n+i] = complex(hs.bt[i*m+j], 0)
		}
	}
	tol := float64(n) * (hs.hMax + cabs1(s)) * eps()
	if tol == 0 {
		tol = 1e-15
	}

	for j := range n {
		u[j] = complex(-h[j], 0)
	}
	u[0] += s
	for k := range n - 1 {
		next := h[(k+1)*n : (k+2)*n]
		uk := u[k*n : (k+1)*n]
		un := u[(k+1)*n : (k+2)*n]
		sub := complex(-next[k], 0)
		diag := s - complex(next[k+1], 0)
		if cabs1(sub) > cabs1(uk[k]) {
			if cabs1(sub) < tol {
				return errSingularPencil
			}
			ik := crecip(sub)
			f := uk[k] * ik
			inv[k] = ik
			un[k+1] = uk[k+1] - f*diag
			uk[k+1] = diag
			for j := k + 2; j < n; j++ {
				r := complex(-next[j], 0)
				un[j] = uk[j] - f*r
				uk[j] = r
			}
			for j := range m {
				x := b[j*n+k:]
				x[0], x[1] = x[1], x[0]-f*x[1]
			}
			continue
		}
		if cabs1(uk[k]) < tol {
			return errSingularPencil
		}
		ik := crecip(uk[k])
		f := sub * ik
		inv[k] = ik
		un[k+1] = diag - f*uk[k+1]
		for j := k + 2; j < n; j++ {
			un[j] = complex(-next[j], 0) - f*uk[j]
		}
		for j := range m {
			x := b[j*n+k:]
			x[1] -= f * x[0]
		}
	}
	last := u[n*n-1]
	if cabs1(last) < tol {
		return errSingularPencil
	}
	inv[n-1] = crecip(last)

	for j := range m {
		x := b[j*n : (j+1)*n]
		for i := n - 1; i >= 0; i-- {
			ui := u[i*n : (i+1)*n]
			acc := x[i]
			for k := i + 1; k < n; k++ {
				acc -= ui[k] * x[k]
			}
			x[i] = acc * inv[i]
		}
	}

	for r := range p {
		cr := hs.ct[r*n : (r+1)*n]
		out := dst[r*m : (r+1)*m]
		for j := range out {
			x := b[j*n : (j+1)*n]
			var re, im float64
			for k, c := range cr {
				re += c * real(x[k])
				im += c * imag(x[k])
			}
			if hs.d != nil {
				re += hs.d[r*hs.dStride+j]
			}
			out[j] = complex(re, im)
		}
	}
	return nil
}

var errSingularPencil = fmt.Errorf("controlsys: singular complex matrix: %w", ErrSingularTransform)

// balance applies the exact power-of-two similarity D⁻¹AD; without it the
// orthogonal reduction's normwise backward error swamps small entries of
// badly scaled realizations such as Padé cascades.
func (hs *hessenbergSweep) balance(scale []float64) {
	n, m, p := hs.n, hs.m, hs.p
	impl.Dgebal(lapack.Scale, n, hs.h, n, scale)
	for i, d := range scale {
		if d == 1 {
			continue
		}
		inv := 1 / d
		for j := range m {
			hs.bt[i*m+j] *= inv
		}
		for r := range p {
			hs.ct[r*n+i] *= d
		}
	}
}
