package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"

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

// finite reports whether every realization entry is finite.
func (br *balancedRealization) finite() bool {
	if !finiteSlice(br.a) || !finiteSlice(br.b) || !finiteSlice(br.c) || br.e != nil && !finiteSlice(br.e) {
		return false
	}
	if br.d == nil {
		return true
	}
	for i := range br.p {
		if !finiteSlice(br.d[i*br.dStride : i*br.dStride+br.m]) {
			return false
		}
	}
	return true
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

// balancedDense evaluates each frequency by GEPP on sÊ-Â; per point it
// costs O(n³) and beats the Hessenberg sweep for small n. GEPP is
// componentwise backward stable relative to the pivoted rows, and the
// pencil entries are rounded: next to a lightly damped pole at distance d
// the slow mode keeps only about ε/d relative accuracy. A pivot below
// 1/refineCondition of the pencil scale flags such a point, and so does a
// response entry that back substitution may have lost to cancellation
// (outputCancels); refine reuses the LU factors and, unless the first
// correction against the exact pencil (pencilResidual) shows the solution
// already accurate, refines with them.
type balancedDense struct {
	balancedRealization
	lastA  []int // last structural nonzero column in each row of sÊ-Â
	last   []int
	pencil []complex128
	inv    []complex128
	x      []complex128   // row-major n×m
	piv    []int          // evalInto's row interchanges, for refine
	r      []complex128   // row-major n×m, refine's residual
	dErr   []float64      // row-major n×m, substitutionBound's, made on first use
	pt     frequencyPoint // the point evalInto last factored, for refine
	tol    float64
	// stalled marks a point at which refine did not resolve every response
	// entry; solveErr then holds its error estimate per entry.
	stalled  bool
	solveErr []float64
	// nonFinite marks a NaN or Inf realization entry.
	nonFinite bool
	// sys, when set, is the delay-free model whose double-double response
	// (ddResponseInto) replaces a stalled point's inaccurate entries.
	sys *System
}

// refineCondition is the error amplification, relative to ε, above which
// the frequency solvers refine against the exact pencil: balancedDense
// checks points whose smallest GEPP pivot is below 1/refineCondition of the
// pencil scale, and both solvers refine when the first correction exceeds
// refineCondition·ε relative.
const refineCondition = 1e3

func newBalancedDense(sys *System, n, m, p int) *balancedDense {
	return newBalancedDenseOf(newBalancedRealization(sys, n, m, p))
}

// newBalancedDenseOf takes ownership of the balanced realization br.
func newBalancedDenseOf(br balancedRealization) *balancedDense {
	n, m := br.n, br.m
	c := make([]complex128, n*n+n+2*n*m)
	bd := &balancedDense{
		balancedRealization: br,
		pencil:              c[: n*n : n*n],
		inv:                 c[n*n : n*n+n : n*n+n],
		x:                   c[n*n+n : n*n+n+n*m : n*n+n+n*m],
		r:                   c[n*n+n+n*m:],
	}
	idx := make([]int, 3*n)
	bd.lastA, bd.last, bd.piv = idx[:n:n], idx[n:2*n:2*n], idx[2*n:]
	for i := range n {
		last := n - 1
		for last > i && bd.a[i*n+last] == 0 && (bd.e == nil || bd.e[i*n+last] == 0) {
			last--
		}
		bd.lastA[i] = last
	}
	bd.nonFinite = !br.finite()
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
	if bd.nonFinite || cmplx.IsNaN(pt.value()) {
		return errNonFinitePencil
	}
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
	bd.pt, bd.tol, bd.stalled = pt, tol, false
	refineBelow, refine := max(scale/refineCondition, tol), false

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
		if best < refineBelow {
			if best < bd.tol {
				return errSingularPencil
			}
			refine = true
		}
		bd.piv[k] = piv
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
			a[i*n+k] = f
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
	if refine || bd.outputCancels() {
		bd.refine()
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
	if bd.stalled && bd.sys != nil && bd.stallInaccurate(dst) {
		ddResponseInto(bd.sys, pt, dst)
	}
	return finiteResponse(dst)
}

// stallInaccurate reports whether a stalled point's error estimate exceeds
// refineCondition·ε of a response entry; ddResponseInto resolves it there
// (graded companion below its poles: 4e-2 relative off).
func (bd *balancedDense) stallInaccurate(dst []complex128) bool {
	for i, v := range dst {
		if bd.solveErr[i] > refineCondition*eps()*cabs1(v) {
			return true
		}
	}
	return false
}

// refine improves bd.x against the exact pencil at pt, M = (pÊ + qÂ)/κ with
// κ = −q for the explicit sI − Â of a non-plain point and 1 otherwise, with
// the factors evalInto left in bd.pencil and bd.piv. It stops once the
// correction is small normwise and in every response entry; where an entry
// passes only at the resolution floor of outputConverged, or the steps run
// out, the point is marked stalled.
func (bd *balancedDense) refine() {
	pt := bd.pt
	n, m := bd.n, bd.m
	x, r := bd.x, bd.r
	kappa := complex(1, 0)
	if !pt.isPlain() && bd.e == nil {
		kappa = pt.scale()
	}
	ik := crecip(kappa)
	for step := range maxRefineSteps {
		for j := range m {
			pencilResidual(r[j:], x[j:], m, bd.a, bd.e, n, bd.b[j:], m, n, pt.p, pt.q, kappa)
		}
		if kappa != 1 {
			for i := range r {
				r[i] *= ik
			}
		}
		cLUSolve(bd.pencil, bd.piv, r, n, m)
		var dMax, xMax float64
		for i, v := range r {
			dMax = max(dMax, cabs1(v))
			xMax = max(xMax, cabs1(x[i]))
		}
		// A working-precision residual cannot make this check: its
		// rounding, ε|M||x|, is the error being measured, and where it
		// rounds to zero an inaccurate x would pass.
		if step == 0 && dMax <= refineCondition*eps()*xMax {
			if exact, floored := outputConverged(bd.c, bd.c, r, x, n, m, bd.p, m, 1, refineCondition*eps()); floored {
				if !exact || bd.correctionNoise(r, refineCondition*eps()) {
					bd.stallErr(r)
				}
				return
			}
		}
		xMax = 0
		for i, v := range r {
			x[i] += v
			xMax = max(xMax, cabs1(x[i]))
		}
		if dMax <= refineStop*xMax {
			// An entry at the resolution floor reads states far below the
			// largest, where the normwise contraction does not bound its
			// error (CKTRTW: 6e-10 relative left); others take refineStop.
			if exact, floored := outputConverged(bd.c, bd.c, r, x, n, m, bd.p, m, 1, refineCondition*eps()); floored {
				if !exact || bd.correctionNoise(r, refineCondition*eps()) {
					bd.stallErr(r)
				}
				return
			}
			if exact, floored := outputConverged(bd.c, bd.c, r, x, n, m, bd.p, m, 1, refineStop); floored {
				if !exact || bd.correctionNoise(r, refineStop) {
					bd.stallErr(r)
				}
				return
			}
		}
	}
	bd.stallErr(r)
}

// stallErr records in bd.solveErr, per response entry, Σₖ|ĉᵢₖ||dₖⱼ| of
// refine's last correction d, or the back-substitution noise in it where
// larger, which estimates the error left where refinement stalls.
func (bd *balancedDense) stallErr(d []complex128) {
	n, m, p := bd.n, bd.m, bd.p
	if bd.solveErr == nil {
		bd.solveErr = make([]float64, p*m)
	}
	bd.substitutionBound(d)
	for i := range p {
		cr := bd.c[i*n : (i+1)*n]
		for j := range m {
			var w, we float64
			for k, c := range cr {
				w += math.Abs(c) * cabs1(d[k*m+j])
				we += math.Abs(c) * bd.dErr[k*m+j]
			}
			bd.solveErr[i*m+j] = max(w, eps()*we)
		}
	}
	bd.stalled = true
}

// correctionNoise reports whether the back-substitution rounding of the
// correction d, ε·Σₖ|ĉᵢₖ|eₖⱼ, exceeds tol of some response entry
// Σₖ|ĉᵢₖ||xₖⱼ|. An output reading states far below the largest takes its
// correction from substitution terms that cancel, and its rounding can
// exceed the correction it should resolve; a correction that then rounds
// to zero proves nothing (CKTRTW: amd64 1.3e-4 relative accepted).
func (bd *balancedDense) correctionNoise(d []complex128, tol float64) bool {
	n, m, x := bd.n, bd.m, bd.x
	bd.substitutionBound(d)
	for j := range m {
		for i := range bd.p {
			var w, we float64
			for k, c := range bd.c[i*n : (i+1)*n] {
				w += math.Abs(c) * cabs1(x[k*m+j])
				we += math.Abs(c) * bd.dErr[k*m+j]
			}
			if !(eps()*we <= tol*w) {
				return true
			}
		}
	}
	return false
}

// substitutionBound sets bd.dErr to the first-order rounding error, in
// units of ε, of back substitution for y: eᵢ = (|yᵢ||uᵢᵢ| + Σₖ|uᵢₖ|(|yₖ| +
// eₖ))/|uᵢᵢ|, as substitutionError bounds it for bd.x.
func (bd *balancedDense) substitutionBound(y []complex128) {
	n, m, a := bd.n, bd.m, bd.pencil
	if bd.dErr == nil {
		bd.dErr = make([]float64, n*m)
	}
	e := bd.dErr
	for i := n - 1; i >= 0; i-- {
		ui := a[i*n+i+1 : i*n+bd.last[i]+1]
		ii := cabs1(bd.inv[i])
		for j := range m {
			v := cabs1(y[i*m+j]) / ii
			for k, u := range ui {
				l := (i+1+k)*m + j
				v += cabs1(u) * (cabs1(y[l]) + e[l])
			}
			e[i*m+j] = v * ii
		}
	}
}

// outputCancels reports whether back substitution may have lost a response
// entry to cancellation (LEAZRP: below the poles of a graded reversed
// companion C = e₁ reads x₀ ≈ 1e-20, formed as a difference of terms near
// 976). Only an entry read from states at least refineCondition below the
// largest of its column can lose more than refineCondition, so the others
// skip substitutionError. An entry substitution left exactly zero is the
// extreme of that loss, not a resolved zero.
func (bd *balancedDense) outputCancels() bool {
	n, m, x := bd.n, bd.m, bd.x
	for j := range m {
		xMax := 0.0
		for k := range n {
			xMax = max(xMax, cabs1(x[k*m+j]))
		}
		for i := range bd.p {
			cr := bd.c[i*n : (i+1)*n]
			var w, cMax float64
			for k, c := range cr {
				w += math.Abs(c) * cabs1(x[k*m+j])
				cMax = max(cMax, math.Abs(c))
			}
			if w*refineCondition < cMax*xMax {
				return bd.substitutionError()
			}
		}
	}
	return false
}

// cancelCondition is the estimated error amplification, relative to ε, of
// a response entry above which substitutionError refines it: about 2e-10
// relative. The estimate is first order and usually pessimistic; refinement
// costs several solves, so entries GEPP leaves within ~1e-10 stay.
const cancelCondition = 1e6

// substitutionError reports whether the first-order rounding error of back
// substitution, eᵢ = (|xᵢ| + Σₖ|uᵢₖ|(|xₖ| + eₖ))/|uᵢᵢ| in units of ε,
// exceeds cancelCondition in some response entry relative to Σₖ|cₖ||xₖ|.
// e is kept in the real parts of bd.r, which refine overwrites.
func (bd *balancedDense) substitutionError() bool {
	n, m, a, x, e := bd.n, bd.m, bd.pencil, bd.x, bd.r
	for i := n - 1; i >= 0; i-- {
		ui := a[i*n+i+1 : i*n+bd.last[i]+1]
		ii := cabs1(bd.inv[i])
		for j := range m {
			v := cabs1(x[i*m+j]) / ii
			for k, u := range ui {
				l := (i+1+k)*m + j
				v += cabs1(u) * (cabs1(x[l]) + real(e[l]))
			}
			e[i*m+j] = complex(v*ii, 0)
		}
	}
	for j := range m {
		for i := range bd.p {
			var w, we float64
			for k, c := range bd.c[i*n : (i+1)*n] {
				w += math.Abs(c) * cabs1(x[k*m+j])
				we += math.Abs(c) * real(e[k*m+j])
			}
			if !(we <= cancelCondition*w) {
				return true
			}
		}
	}
	return false
}

// outputConverged reports whether the correction d is within tol of x in
// every response entry, weighted by the output rows: Σₖ|cdᵢₖ||dₖⱼ| ≤
// tol·Σₖ|cxᵢₖ||xₖⱼ|. A normwise test passes a correction that is tiny
// against ‖x‖ but not against the few states an output reads. cd and cx are
// row-major p×n; entry (k, j) of d and x is at k·ks + j·js. floored is the
// test with the weight floored at ε·max|cxᵢ|·max|xⱼ|, which also passes an
// entry below what the compensated residual resolves, about ε² of that
// scale.
func outputConverged(cd, cx []float64, d, x []complex128, n, m, p, ks, js int, tol float64) (exact, floored bool) {
	exact = true
	for j := range m {
		xMax := 0.0
		for k := range n {
			xMax = max(xMax, cabs1(x[k*ks+j*js]))
		}
		for i := range p {
			rd, rx := cd[i*n:(i+1)*n], cx[i*n:(i+1)*n]
			var wd, wx, cMax float64
			for k := range n {
				wd += math.Abs(rd[k]) * cabs1(d[k*ks+j*js])
				wx += math.Abs(rx[k]) * cabs1(x[k*ks+j*js])
				cMax = max(cMax, math.Abs(rx[k]))
			}
			if wd > tol*wx {
				exact = false
				if wd > tol*eps()*cMax*xMax {
					return false, false
				}
			}
		}
	}
	return exact, true
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
	q  []float64
	h  []float64
	bt []float64 // QᵀB̂
	ct []float64 // ĈQ
	// ctw bounds |ĈQ| with its reduction error, nε‖ĉᵢ‖₁ per row: a
	// normwise backward error that swamps the entries a graded Ĉ cancels in.
	ctw     []float64
	hMax    float64
	u       []complex128
	inv     []complex128
	shift   []complex128 // s-âᵢᵢ at the current point
	mult    []complex128
	swapped []bool
	y       []complex128 // column-major n×m, Hessenberg coordinates
	x       []complex128 // column-major n×m, balanced coordinates
	r       []complex128 // column-major n×m
	// nonFinite marks a NaN or Inf realization entry.
	nonFinite bool
	sys       *System
	dense     *balancedDense // denseRetry's solver, built on first use
	alt       []complex128
}

func newHessenbergSweep(sys *System, n, m, p int) *hessenbergSweep {
	nn := n * n
	f := make([]float64, 2*nn+n*m+2*p*n)
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
	hs := &hessenbergSweep{balancedRealization: newBalancedRealization(sys, n, m, p), sys: sys}
	hs.q, hs.h, hs.bt, hs.ct, hs.ctw = take(nn), take(nn), take(n*m), take(p*n), take(p*n)
	hs.u, hs.inv, hs.mult, hs.shift = ctake(nn), ctake(n), ctake(n), ctake(n)
	hs.y, hs.x, hs.r = ctake(n*m), ctake(n*m), ctake(n*m)
	hs.swapped = make([]bool, n)
	hs.nonFinite = !hs.finite()
	if n == 0 || hs.nonFinite {
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
	for i := range p {
		w := 0.0
		for _, v := range hs.c[i*n : (i+1)*n] {
			w += math.Abs(v)
		}
		w *= float64(n) * eps()
		for k, v := range hs.ct[i*n : (i+1)*n] {
			hs.ctw[i*n+k] = math.Abs(v) + w
		}
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

// sweepFirstStepTol is the relative output correction below which the
// Hessenberg sweep accepts its first, working-precision refinement step;
// later steps, against the exact pencil, accept only refineCondition·ε too,
// since a step's correction bounds the error it leaves only where the
// refinement contracts fast, which a graded model's slow states need not.
var sweepFirstStepTol = refineCondition * eps()

// refineStop is the relative correction below which refinement stops. The
// correction δ estimates the error it removes and the contraction is about
// ‖δ‖/‖x‖, so the error left after applying δ is near ‖δ‖²/‖x‖ ≤ ε‖x‖.
// That bound is normwise: an output at the resolution floor, reading states
// far below the largest, can keep 1e-9 relative error past a 1e-17 normwise
// step (CKTRTW), so balancedDense.refine holds it to refineCondition·ε.
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

// exactResidual sets r = B̂ - (sI-Â)x̂ against the exact point with
// pencilResidual. It replaces residual after the first step, when the
// correction shows a point near a pole: the rounded hs.shift and the
// working-precision sum would otherwise stop refinement at about ε/d
// relative accuracy at distance d from a lightly damped pole.
func (hs *hessenbergSweep) exactResidual(pt frequencyPoint) {
	n, m, r := hs.n, hs.m, hs.r
	kappa := pt.scale()
	for j := range m {
		pencilResidual(r[j*n:], hs.x[j*n:], 1, hs.a, nil, n, hs.b[j:], m, n, pt.p, pt.q, kappa)
	}
	if kappa != 1 {
		ik := crecip(kappa)
		for i := range r {
			r[i] *= ik
		}
	}
}

// outputConverged is outputConverged of the correction hs.y. Weighted by
// hs.ctw in Hessenberg coordinates it overstates the correction an output
// sees, so where that is not exact the test is redone on Qy in r, and viaQ
// reports that the correction must reach the output as Qy: ĈQ·y can then
// be all reduction error (CKTRTW #23: 1.7e-30 beside Ĉ·Qy = 2.5e-40 and
// |G| = 6.5e-27 on amd64).
func (hs *hessenbergSweep) outputConverged(tol float64) (exact, floored, viaQ bool) {
	n, m, p := hs.n, hs.m, hs.p
	if exact, _ := outputConverged(hs.ctw, hs.c, hs.y, hs.x, n, m, p, 1, n, tol); exact {
		return true, true, false
	}
	qMulCols(hs.r, hs.y, hs.q, n, m)
	exact, floored = outputConverged(hs.c, hs.c, hs.r, hs.x, n, m, p, 1, n, tol)
	return exact, floored, true
}

// denseRetry replaces dst with the balanced dense response at pt, built on
// first use, where refinement stalled: only an output correction within
// refineCondition·ε bounds the error left, and the reduction to H can leave
// a graded model's slow states, the ones a cancelling output reads, beyond
// what refinement through H resolves (CKTRTW: 2e10 relative). The dense
// solver takes the double-double response where its own refinement stalls.
// Where the dense pivots call the pencil singular, as a graded companion's
// can far from any pole (CKTRTW: 12.6 relative kept), a delay-free model
// takes the double-double response, and the dense error is returned where
// that is singular too.
func (hs *hessenbergSweep) denseRetry(pt frequencyPoint, dst []complex128) error {
	if hs.sys == nil {
		return nil
	}
	if hs.dense == nil {
		hs.dense = newBalancedDense(hs.sys, hs.n, hs.m, hs.p)
		if hs.sys.internalDelayCount() == 0 {
			hs.dense.sys = hs.sys
		}
		hs.alt = make([]complex128, hs.p*hs.m)
	}
	err := hs.dense.evalInto(pt, hs.alt)
	if errors.Is(err, errSingularPencil) && hs.dense.sys != nil && ddResponseInto(hs.sys, pt, hs.alt) {
		err = nil
	}
	if err == nil {
		copy(dst, hs.alt)
	}
	return err
}

func (hs *hessenbergSweep) evalInto(pt frequencyPoint, dst []complex128) error {
	n, m, p := hs.n, hs.m, hs.p
	if hs.nonFinite || cmplx.IsNaN(pt.value()) {
		return errNonFinitePencil
	}
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
	// applied through ĈQ, saving a final Q multiply, unless only the Qy
	// test passed (viaQ): then Qy, already in r, is added to x.
	converged, viaQ := false, false
	for step := range maxRefineSteps {
		if step > 0 {
			qMulCols(r, y, hs.q, n, m)
			for i, v := range r {
				x[i] += v
			}
		}
		if step == 0 {
			hs.residual()
		} else {
			hs.exactResidual(pt)
		}
		qTMulCols(y, r, hs.q, n, m)
		hs.solve(y)
		var dMax, xMax float64
		for i, v := range y {
			dMax = max(dMax, cabs1(v))
			xMax = max(xMax, cabs1(x[i]))
		}
		viaQ = false
		tol := refineCondition * eps()
		if step == 0 {
			tol = sweepFirstStepTol
		}
		if dMax <= tol*xMax {
			exact, floored, q := hs.outputConverged(tol)
			converged, viaQ = exact, q
			if floored {
				break
			}
		}
	}
	if viaQ {
		for i, v := range r {
			x[i] += v
		}
		clear(y)
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
	if !converged {
		if err := hs.denseRetry(pt, dst); err != nil {
			return err
		}
	}
	return finiteResponse(dst)
}

// pencilResidual sets r = κ·b − p·E·x − q·A·x for one right-hand side in
// compensated arithmetic, as if in twice the working precision, with E = I
// when e is nil. p, q and the real A, E and b enter exactly, so refining
// against r converges to the solution of the exact pencil, not of its
// rounded entries. x and r have element stride inc, b stride bInc, and a, e
// row stride ld.
func pencilResidual(r, x []complex128, inc int, a, e []float64, ld int, b []float64, bInc, n int, p, q, kappa complex128) {
	pr, pi, qr, qi := real(p), imag(p), real(q), imag(q)
	for i := range n {
		var axr, axi compensatedSum
		for k, av := range a[i*ld : i*ld+n] {
			v := x[k*inc]
			axr.addProd(av, real(v))
			axi.addProd(av, imag(v))
		}
		var re, im compensatedSum
		if kappa == 1 {
			re.add(b[i*bInc])
		} else {
			re.addProd(real(kappa), b[i*bInc])
			im.addProd(imag(kappa), b[i*bInc])
		}
		if e == nil {
			xr, xi := real(x[i*inc]), imag(x[i*inc])
			re.addProd(-pr, xr)
			re.addProd(pi, xi)
			im.addProd(-pr, xi)
			im.addProd(-pi, xr)
		} else {
			var exr, exi compensatedSum
			for k, ev := range e[i*ld : i*ld+n] {
				v := x[k*inc]
				exr.addProd(ev, real(v))
				exi.addProd(ev, imag(v))
			}
			re.addScaled(-pr, exr)
			re.addScaled(pi, exi)
			im.addScaled(-pr, exi)
			im.addScaled(-pi, exr)
		}
		re.addScaled(-qr, axr)
		re.addScaled(qi, axi)
		im.addScaled(-qr, axi)
		im.addScaled(-qi, axr)
		r[i*inc] = complex(re.value(), im.value())
	}
}

// cLUFactor overwrites the row-major n×n lu with its GEPP factors, unit L
// below the diagonal, recording the row interchanges in piv. It reports
// false at an exactly zero pivot.
func cLUFactor(lu []complex128, piv []int, n int) bool {
	for k := range n {
		p, best := k, cabs1(lu[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := cabs1(lu[i*n+k]); v > best {
				p, best = i, v
			}
		}
		if best == 0 {
			return false
		}
		piv[k] = p
		if p != k {
			rk, rp := lu[k*n:(k+1)*n], lu[p*n:(p+1)*n]
			for j := range rk {
				rk[j], rp[j] = rp[j], rk[j]
			}
		}
		ik := crecip(lu[k*n+k])
		for i := k + 1; i < n; i++ {
			f := lu[i*n+k] * ik
			lu[i*n+k] = f
			if f == 0 {
				continue
			}
			for j := k + 1; j < n; j++ {
				lu[i*n+j] -= f * lu[k*n+j]
			}
		}
	}
	return true
}

// cLUSolve overwrites the row-major n×m x with the solution for the factors
// from cLUFactor.
func cLUSolve(lu []complex128, piv []int, x []complex128, n, m int) {
	for k, pk := range piv {
		if pk != k {
			for j := range m {
				x[k*m+j], x[pk*m+j] = x[pk*m+j], x[k*m+j]
			}
		}
	}
	for k := range n {
		for i := k + 1; i < n; i++ {
			if f := lu[i*n+k]; f != 0 {
				for j := range m {
					x[i*m+j] -= f * x[k*m+j]
				}
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		inv := crecip(lu[i*n+i])
		for j := range m {
			v := x[i*m+j]
			for k := i + 1; k < n; k++ {
				v -= lu[i*n+k] * x[k*m+j]
			}
			x[i*m+j] = v * inv
		}
	}
}

var errSingularPencil = fmt.Errorf("singular complex matrix: %w", ErrSingularTransform)

// finiteResponse reports errSingularPencil for a non-finite response of a
// finite realization: the solve overflowed or underflowed, as next to a pole
// at extreme scale, so the pole limit's double-double fallback takes over.
func finiteResponse(dst []complex128) error {
	for _, v := range dst {
		if math.IsNaN(real(v)) || math.IsNaN(imag(v)) || math.IsInf(real(v), 0) || math.IsInf(imag(v), 0) {
			return errSingularPencil
		}
	}
	return nil
}

// errNonFinitePencil rejects a NaN point or non-finite realization: NaN
// comparisons defeat GEPP's pivot and structure bookkeeping.
var errNonFinitePencil = fmt.Errorf("non-finite complex matrix: %w", ErrInvalidArgument)

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
	var rowLo, rowHi, colLo, colHi float64
	addRow := func(v float64) float64 {
		if v = math.Abs(v); v != 0 {
			rowLo, rowHi = min(rowLo, v), max(rowHi, v)
		}
		return v
	}
	addCol := func(v float64) float64 {
		if v = math.Abs(v); v != 0 {
			colLo, colHi = min(colLo, v), max(colHi, v)
		}
		return v
	}
	for range maxIt {
		converged := true
		for i := range n {
			var col, row float64
			rowLo, rowHi, colLo, colHi = math.Inf(1), 0, math.Inf(1), 0
			for j := range n {
				if j != i {
					col += addCol(h[j*n+i])
					row += addRow(h[i*n+j])
				}
			}
			if e != nil {
				for j := range n {
					if j != i {
						col += addCol(e[j*n+i])
						row += addRow(e[i*n+j])
					}
				}
			}
			for k := range p {
				col += addCol(c[k*n+i])
			}
			for k := range m {
				row += addRow(b[i*m+k])
			}
			if col == 0 || row == 0 {
				continue
			}
			// Keep every scaled entry normal, so the scaling stays exact.
			fMax := min(0x1p500, rowLo/0x1p-1022, math.MaxFloat64/colHi)
			fMin := max(0x1p-500, 0x1p-1022/colLo, rowHi/math.MaxFloat64)
			f, sum := 1.0, col+row
			for col < row/radix && f*radix <= fMax {
				f *= radix
				col *= radix
				row /= radix
			}
			for col >= row*radix && f/radix >= fMin {
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
