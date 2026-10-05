package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
)

// balancedRealization is (E, A, B, C, D) after an exact power-of-two state
// scaling, the shared starting point of the frequency solvers. e is nil for
// explicit models.
type balancedRealization struct {
	n, m, p int
	e       []float64 // Ê = D⁻¹ED
	a       []float64 // Â = D⁻¹AD
	b       []float64 // B̂ = D⁻¹B
	c       []float64 // Ĉ = CD
	d       []float64
	dStride int
}

func newBalancedRealization(sys *System, n, m, p int) balancedRealization {
	br := newRealizationCopy(sys, n, m, p)
	br.balance(nil)
	return br
}

// newRealizationCopy copies the realization into one fresh buffer, unscaled.
func newRealizationCopy(sys *System, n, m, p int) balancedRealization {
	f := make([]float64, n*n+n*m+p*n)
	br := balancedRealization{n: n, m: m, p: p, a: f[: n*n : n*n], b: f[n*n : n*n+n*m : n*n+n*m], c: f[n*n+n*m:]}
	if sys.D != nil {
		raw := sys.D.RawMatrix()
		br.d, br.dStride = raw.Data, raw.Stride
	}
	if n == 0 {
		return br
	}
	a := sys.A.RawMatrix()
	copyStrided(br.a, n, a.Data, a.Stride, n, n)
	if sys.E != nil {
		e := sys.E.RawMatrix()
		br.e = make([]float64, n*n)
		copyStrided(br.e, n, e.Data, e.Stride, n, n)
	}
	if m > 0 {
		b := sys.B.RawMatrix()
		copyStrided(br.b, m, b.Data, b.Stride, n, m)
	}
	if p > 0 {
		c := sys.C.RawMatrix()
		copyStrided(br.c, n, c.Data, c.Stride, p, n)
	}
	return br
}

// balancedDense evaluates each frequency by GEPP on sÊ-Â. GEPP is
// componentwise backward stable, so it needs no refinement; per point it
// costs O(n³) and beats the Hessenberg sweep for small n.
type balancedDense struct {
	balancedRealization
	lastA  []int // last structural nonzero column in each row of sÊ-Â
	last   []int
	pencil []complex128
	inv    []complex128
	x      []complex128 // row-major n×m
}

func newBalancedDense(sys *System, n, m, p int) *balancedDense {
	return newBalancedDenseOf(newBalancedRealization(sys, n, m, p))
}

// newBalancedDenseOf takes ownership of the balanced realization br.
func newBalancedDenseOf(br balancedRealization) *balancedDense {
	n, m := br.n, br.m
	c := make([]complex128, n*n+n+n*m)
	bd := &balancedDense{
		balancedRealization: br,
		pencil:              c[: n*n : n*n],
		inv:                 c[n*n : n*n+n : n*n+n],
		x:                   c[n*n+n:],
	}
	idx := make([]int, 2*n)
	bd.lastA, bd.last = idx[:n:n], idx[n:]
	for i := range n {
		last := n - 1
		for last > i && bd.a[i*n+last] == 0 && (bd.e == nil || bd.e[i*n+last] == 0) {
			last--
		}
		bd.lastA[i] = last
	}
	return bd
}

// fillPencil sets bd.pencil to sÊ-Â and returns the scale of its entries.
func (bd *balancedDense) fillPencil(s complex128) float64 {
	a, n := bd.pencil, bd.n
	maxAbs, sScale := 0.0, cabs1(s)
	if bd.e == nil {
		for i, v := range bd.a {
			a[i] = complex(-v, 0)
			maxAbs = max(maxAbs, math.Abs(v))
		}
		for i := range n {
			a[i*n+i] += s
		}
		return maxAbs + sScale
	}
	maxE := 0.0
	for i, v := range bd.a {
		e := bd.e[i]
		// Fused: near a pole zÊ-Â is far smaller than zÊ, and a
		// rounded product would cost ε|zÊ|/|zÊ-Â| relative accuracy.
		a[i] = complex(math.FMA(real(s), e, -v), imag(s)*e)
		maxAbs = max(maxAbs, math.Abs(v))
		maxE = max(maxE, math.Abs(e))
	}
	return maxAbs + sScale*maxE
}

// fillPencilAt sets bd.pencil to a matrix M and returns the scale of its
// entries and c with (sÊ-Â)⁻¹ = c·M⁻¹: sI-Â with exact-point diagonal
// shifts for explicit models, and pÊ+qÂ, c = −q, for descriptor models.
func (bd *balancedDense) fillPencilAt(pt frequencyPoint) (float64, complex128) {
	a, n := bd.pencil, bd.n
	maxAbs := 0.0
	if bd.e == nil {
		for i, v := range bd.a {
			a[i] = complex(-v, 0)
			maxAbs = max(maxAbs, math.Abs(v))
		}
		for i := range n {
			a[i*n+i] = pt.shift(bd.a[i*n+i])
		}
		return maxAbs + cabs1(pt.p), 1
	}
	pr, pi := real(pt.p), imag(pt.p)
	maxE := 0.0
	for i, v := range bd.a {
		e := bd.e[i]
		// With q = −conj(p), pe+qa = pr(e−a) + j·pi(e+a): near a pole e−a
		// is small, and a rounded product qr·a would cost ε|a|/|e−a|
		// relative accuracy.
		a[i] = complex(pr*(e-v), pi*(e+v))
		maxAbs = max(maxAbs, math.Abs(v))
		maxE = max(maxE, math.Abs(e))
	}
	return cabs1(pt.q)*maxAbs + cabs1(pt.p)*maxE, pt.scale()
}

func (bd *balancedDense) evalInto(pt frequencyPoint, dst []complex128) error {
	n, m, p := bd.n, bd.m, bd.p
	if n == 0 {
		copyRealMatrixToComplex(dst, bd.d, bd.dStride, p, m)
		return nil
	}
	a, x, inv := bd.pencil, bd.x, bd.inv
	scale, gs := 0.0, complex(1, 0)
	if pt.isPlain() {
		scale = bd.fillPencil(pt.value())
	} else {
		scale, gs = bd.fillPencilAt(pt)
	}
	for i, v := range bd.b {
		x[i] = complex(v, 0)
	}
	tol := float64(n) * scale * eps()
	if tol == 0 {
		tol = 1e-15
	}

	// last[i] bounds the nonzeros of row i, so banded and block-triangular
	// models skip their structural zeros in elimination and substitution.
	last := bd.last
	copy(last, bd.lastA)
	for k := range n {
		piv, best := k, cabs1(a[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := cabs1(a[i*n+k]); v > best {
				piv, best = i, v
			}
		}
		if best < tol {
			return errSingularPencil
		}
		if piv != k {
			rk, rp := a[k*n:(k+1)*n], a[piv*n:(piv+1)*n]
			for j := range rk {
				rk[j], rp[j] = rp[j], rk[j]
			}
			xk, xp := x[k*m:(k+1)*m], x[piv*m:(piv+1)*m]
			for j := range xk {
				xk[j], xp[j] = xp[j], xk[j]
			}
			last[k], last[piv] = last[piv], last[k]
		}
		ik := crecip(a[k*n+k])
		inv[k] = ik
		lk := last[k]
		rk := a[k*n+k+1 : k*n+lk+1]
		for i := k + 1; i < n; i++ {
			f := a[i*n+k] * ik
			if f == 0 {
				continue
			}
			ri := a[i*n+k+1 : i*n+lk+1]
			ri = ri[:len(rk)]
			for j, v := range rk {
				ri[j] -= f * v
			}
			for j := range m {
				x[i*m+j] -= f * x[k*m+j]
			}
			last[i] = max(last[i], lk)
		}
	}
	for i := n - 1; i >= 0; i-- {
		ui := a[i*n+i+1 : i*n+last[i]+1]
		for j := range m {
			xs := x[i*m+j:]
			var re, im float64
			for k, u := range ui {
				w := xs[(k+1)*m]
				re += real(u)*real(w) - imag(u)*imag(w)
				im += real(u)*imag(w) + imag(u)*real(w)
			}
			x[i*m+j] = (x[i*m+j] - complex(re, im)) * inv[i]
		}
	}

	for row := range p {
		cr := bd.c[row*n : (row+1)*n]
		out := dst[row*m : (row+1)*m]
		for j := range out {
			var re, im float64
			for k, c := range cr {
				v := x[k*m+j]
				re += c * real(v)
				im += c * imag(v)
			}
			if gs != 1 {
				g := complex(re, im) * gs
				re, im = real(g), imag(g)
			}
			if bd.d != nil {
				re += bd.d[row*bd.dStride+j]
			}
			out[j] = complex(re, im)
		}
	}
	return nil
}

// hessenbergSweep evaluates C(sI-A)^{-1}B + D over many frequencies after one
// orthogonal reduction Â = Q H Qᵀ of the balanced realization (Laub 1981).
// Each point costs an O(n²) Hessenberg factorization and O(n²m) solves. The
// reduction alone is only normwise backward stable: it mixes a slow decoupled
// mode's row with fast rows, so the slow pole is recovered only to ε‖Â‖.
// Iterative refinement against Â itself (Skeel) restores componentwise
// stability; one step normally suffices.
type hessenbergSweep struct {
	balancedRealization
	q       []float64
	h       []float64
	bt      []float64 // QᵀB̂
	ct      []float64 // ĈQ
	hMax    float64
	u       []complex128
	inv     []complex128
	shift   []complex128 // s-âᵢᵢ at the current point
	mult    []complex128
	swapped []bool
	y       []complex128 // column-major n×m, Hessenberg coordinates
	x       []complex128 // column-major n×m, balanced coordinates
	r       []complex128 // column-major n×m
}

func newHessenbergSweep(sys *System, n, m, p int) *hessenbergSweep {
	nn := n * n
	f := make([]float64, 2*nn+n*m+p*n)
	c := make([]complex128, nn+3*n+3*n*m)
	take := func(k int) []float64 {
		out := f[:k:k]
		f = f[k:]
		return out
	}
	ctake := func(k int) []complex128 {
		out := c[:k:k]
		c = c[k:]
		return out
	}
	hs := &hessenbergSweep{balancedRealization: newBalancedRealization(sys, n, m, p)}
	hs.q, hs.h, hs.bt, hs.ct = take(nn), take(nn), take(n*m), take(p*n)
	hs.u, hs.inv, hs.mult, hs.shift = ctake(nn), ctake(n), ctake(n), ctake(n)
	hs.y, hs.x, hs.r = ctake(n*m), ctake(n*m), ctake(n*m)
	hs.swapped = make([]bool, n)
	if n == 0 {
		return hs
	}
	copy(hs.h, hs.a)
	copy(hs.bt, hs.b)
	copy(hs.ct, hs.c)
	if n >= 3 {
		hs.reduce()
	} else {
		for i := range n {
			hs.q[i*n+i] = 1
		}
	}
	for _, v := range hs.h {
		hs.hMax = max(hs.hMax, math.Abs(v))
	}
	return hs
}

// reduce forms H = QᵀÂQ, QᵀB̂, ĈQ and the explicit Q used by refinement,
// with the minimal (unblocked) workspace; models here are small enough that
// blocking only costs memory.
func (hs *hessenbergSweep) reduce() {
	n, m, p := hs.n, hs.m, hs.p
	tau := make([]float64, n-1)
	work := make([]float64, max(1, n, m, p))
	lwork := len(work)
	impl.Dgehrd(n, 0, n-1, hs.h, n, tau, work, lwork)
	if m > 0 {
		impl.Dormhr(blas.Left, blas.Trans, n, m, 0, n-1, hs.h, n, tau, hs.bt, m, work, lwork)
	}
	if p > 0 {
		impl.Dormhr(blas.Right, blas.NoTrans, p, n, 0, n-1, hs.h, n, tau, hs.ct, n, work, lwork)
	}
	copy(hs.q, hs.h)
	impl.Dorghr(n, 0, n-1, hs.q, n, tau, work, lwork)
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

// factor computes the LU factorization of sI-H with adjacent-row pivoting,
// its diagonal shifted as pt.shift forms it,
// recording multipliers and swaps so it can be replayed on several
// right-hand sides. Row k of u holds the not-yet-pivoted row until step k
// settles it, so the pencil is never formed.
func (hs *hessenbergSweep) factor(pt frequencyPoint) error {
	s := pt.value()
	n := hs.n
	h, u, inv := hs.h, hs.u, hs.inv
	tol := float64(n) * (hs.hMax + cabs1(s)) * eps()
	if tol == 0 {
		tol = 1e-15
	}
	for j := range n {
		u[j] = complex(-h[j], 0)
	}
	if pt.isPlain() {
		u[0] += s
	} else {
		u[0] = pt.shift(h[0])
	}
	for k := range n - 1 {
		next := h[(k+1)*n : (k+2)*n]
		uk := u[k*n : (k+1)*n]
		un := u[(k+1)*n : (k+2)*n]
		sub := complex(-next[k], 0)
		diag := pt.shift(next[k+1])
		if cabs1(sub) > cabs1(uk[k]) {
			if cabs1(sub) < tol {
				return errSingularPencil
			}
			ik := crecip(sub)
			f := uk[k] * ik
			inv[k], hs.mult[k], hs.swapped[k] = ik, f, true
			un[k+1] = uk[k+1] - f*diag
			uk[k+1] = diag
			for j := k + 2; j < n; j++ {
				r := complex(-next[j], 0)
				un[j] = uk[j] - f*r
				uk[j] = r
			}
			continue
		}
		if cabs1(uk[k]) < tol {
			return errSingularPencil
		}
		ik := crecip(uk[k])
		f := sub * ik
		inv[k], hs.mult[k], hs.swapped[k] = ik, f, false
		un[k+1] = diag - f*uk[k+1]
		for j := k + 2; j < n; j++ {
			un[j] = complex(-next[j], 0) - f*uk[j]
		}
	}
	last := u[n*n-1]
	if cabs1(last) < tol {
		return errSingularPencil
	}
	inv[n-1] = crecip(last)
	return nil
}

// solve overwrites the column-major n×m b with (sI-H)⁻¹b.
func (hs *hessenbergSweep) solve(b []complex128) {
	n, u, inv := hs.n, hs.u, hs.inv
	for j := range hs.m {
		x := b[j*n : (j+1)*n]
		for k := range n - 1 {
			if hs.swapped[k] {
				x[k], x[k+1] = x[k+1], x[k]
			}
			x[k+1] -= hs.mult[k] * x[k]
		}
		backSubstitute(x, u, inv, n)
	}
}

// backSubstitute overwrites x with U⁻¹x for the row-major upper-triangular
// u whose reciprocal diagonal is inv.
func backSubstitute(x, u, inv []complex128, n int) {
	for i := n - 1; i >= 0; i-- {
		ut := u[i*n+i+1 : (i+1)*n]
		xt := x[i+1 : i+1+len(ut)]
		var re, im float64
		for k, v := range ut {
			w := xt[k]
			re += real(v)*real(w) - imag(v)*imag(w)
			im += real(v)*imag(w) + imag(v)*real(w)
		}
		x[i] = (x[i] - complex(re, im)) * inv[i]
	}
}

// qMulCols sets dst[:,j] = Q·src[:,j] for the real n×n row-major Q and
// column-major n×m complex dst, src.
func qMulCols(dst, src []complex128, q []float64, n, m int) {
	for j := range m {
		in, out := src[j*n:(j+1)*n], dst[j*n:(j+1)*n]
		for i := range out {
			row := q[i*n : (i+1)*n]
			var re, im float64
			for k, a := range row {
				re += a * real(in[k])
				im += a * imag(in[k])
			}
			out[i] = complex(re, im)
		}
	}
}

// qTMulCols sets dst[:,j] = Qᵀ·src[:,j].
func qTMulCols(dst, src []complex128, q []float64, n, m int) {
	for j := range m {
		in, out := src[j*n:(j+1)*n], dst[j*n:(j+1)*n]
		clear(out)
		for k, v := range in {
			re, im := real(v), imag(v)
			row := q[k*n : (k+1)*n]
			for i, a := range row {
				out[i] += complex(a*re, a*im)
			}
		}
	}
}

// maxRefineSteps bounds refinement; each step contracts the error by
// roughly ε‖Â‖‖(sI-Â)⁻¹‖, so well-conditioned points stop after one.
const maxRefineSteps = 3

// refineStop is the relative correction below which refinement stops. The
// correction δ estimates the error it removes and the contraction is about
// ‖δ‖/‖x‖, so the error left after applying δ is near ‖δ‖²/‖x‖ ≤ ε‖x‖.
var refineStop = math.Sqrt(eps())

// residual sets r = B̂ - (sI-Â)x̂. The diagonal enters as s-âᵢᵢ, as GEPP
// forms it (hs.shift), and stays out of the off-diagonal sum, so a slow pole near s
// keeps its relative accuracy instead of cancelling against sx̂ᵢ. A
// working-precision residual makes one refinement step componentwise
// backward stable (Skeel 1980), the accuracy class of GEPP on sI-Â;
// a compensated residual would go further at ~2.5x the per-point cost.
func (hs *hessenbergSweep) residual() {
	n, m, x, r := hs.n, hs.m, hs.x, hs.r
	for j := range m {
		xj, rj := x[j*n:(j+1)*n], r[j*n:(j+1)*n]
		for i := range n {
			row := hs.a[i*n : (i+1)*n]
			var re, im float64
			for k, a := range row[:i] {
				re += a * real(xj[k])
				im += a * imag(xj[k])
			}
			for k, a := range row[i+1:] {
				re += a * real(xj[i+1+k])
				im += a * imag(xj[i+1+k])
			}
			rj[i] = complex(hs.b[i*m+j]+re, im) - hs.shift[i]*xj[i]
		}
	}
}

func (hs *hessenbergSweep) evalInto(pt frequencyPoint, dst []complex128) error {
	n, m, p := hs.n, hs.m, hs.p
	if n == 0 {
		copyRealMatrixToComplex(dst, hs.d, hs.dStride, p, m)
		return nil
	}
	if err := hs.factor(pt); err != nil {
		return err
	}
	for i := range n {
		hs.shift[i] = pt.shift(hs.a[i*n+i])
	}
	y, x, r := hs.y, hs.x, hs.r
	for i := range n {
		for j := range m {
			y[j*n+i] = complex(hs.bt[i*m+j], 0)
		}
	}
	hs.solve(y)
	qMulCols(x, y, hs.q, n, m)

	// y ends as the last correction in Hessenberg coordinates; it is
	// applied through ĈQ, saving a final Q multiply.
	for step := range maxRefineSteps {
		if step > 0 {
			qMulCols(r, y, hs.q, n, m)
			for i, v := range r {
				x[i] += v
			}
		}
		hs.residual()
		qTMulCols(y, r, hs.q, n, m)
		hs.solve(y)
		var dMax, xMax float64
		for i, v := range y {
			dMax = max(dMax, cabs1(v))
			xMax = max(xMax, cabs1(x[i]))
		}
		if dMax <= refineStop*xMax {
			break
		}
	}

	for row := range p {
		cr := hs.c[row*n : (row+1)*n]
		ct := hs.ct[row*n : (row+1)*n]
		out := dst[row*m : (row+1)*m]
		for j := range out {
			xj, dj := x[j*n:(j+1)*n], y[j*n:(j+1)*n]
			var re, im, dre, dim float64
			for k, c := range cr {
				re += c * real(xj[k])
				im += c * imag(xj[k])
				dre += ct[k] * real(dj[k])
				dim += ct[k] * imag(dj[k])
			}
			if hs.d != nil {
				re += hs.d[row*hs.dStride+j]
			}
			out[j] = complex(re+dre, im+dim)
		}
	}
	return nil
}

var errSingularPencil = fmt.Errorf("controlsys: singular complex matrix: %w", ErrSingularTransform)

// balance applies an exact power-of-two state scaling x = D x̂ chosen to
// balance the rows and columns of [|A|+|E| B; C ·] (Osborne iteration as in
// SLICOT TB01ID; E enters as in MATLAB ssbal). When t is non-nil it is
// multiplied by D⁻¹. The orthogonal reduction is normwise backward stable in
// (A, B, C), so balancing A alone (Dgebal) is not enough: a slow
// input-driven mode with a near-empty A row is shrunk until B̂ and Ĉ span
// many decades and roundoff in QᵀB̂ swamps the response.
func (br *balancedRealization) balance(t []float64) {
	const (
		radix = 2.0
		maxIt = 100
	)
	n, m, p := br.n, br.m, br.p
	h, e, b, c := br.a, br.e, br.b, br.c
	for range maxIt {
		converged := true
		for i := range n {
			var col, row float64
			for j := range n {
				if j != i {
					col += math.Abs(h[j*n+i])
					row += math.Abs(h[i*n+j])
				}
			}
			if e != nil {
				for j := range n {
					if j != i {
						col += math.Abs(e[j*n+i])
						row += math.Abs(e[i*n+j])
					}
				}
			}
			for k := range p {
				col += math.Abs(c[k*n+i])
			}
			for k := range m {
				row += math.Abs(b[i*m+k])
			}
			if col == 0 || row == 0 {
				continue
			}
			f, sum := 1.0, col+row
			for col < row/radix && f < 0x1p500 {
				f *= radix
				col *= radix
				row /= radix
			}
			for col >= row*radix && f > 0x1p-500 {
				f /= radix
				col /= radix
				row *= radix
			}
			if col+row >= 0.95*sum {
				continue
			}
			converged = false
			inv := 1 / f
			for j := range n {
				h[i*n+j] *= inv
				h[j*n+i] *= f
			}
			if e != nil {
				for j := range n {
					e[i*n+j] *= inv
					e[j*n+i] *= f
				}
			}
			if t != nil {
				t[i] *= inv
			}
			for k := range m {
				b[i*m+k] *= inv
			}
			for k := range p {
				c[k*n+i] *= f
			}
		}
		if converged {
			return
		}
	}
}
