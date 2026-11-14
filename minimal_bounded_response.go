package controlsys

import (
	"math"
	"math/cmplx"
	"sync"
)

// boundedResponse is the strictly proper response G(s) − D = C(sI−A)⁻¹B
// of an explicit model at points s, row-major p×m per point, with for each
// entry three first-order bounds: comp, on the error of the computed
// strictly proper response for the stored data (the exact identity
// G_sp − CÛ = C(sI−A)⁻¹(B − (sI−A)Û) with the residual formed in
// compensated arithmetic, one refinement step where the solve is poorly
// conditioned, and the rounding of the output product); ver, comp plus the
// a-priori GEPP term for the error of the computed C(sI−A)⁻¹, rigorous to
// second order and used only to verify; and data, ε·(|L||A||U| + |C||U| +
// |L||B|), U = (sI−A)⁻¹B and L = C(sI−A)⁻¹, the change of the response
// when every entry of A, B and C moves by ε relative: the resolution of
// the stored model; and terms, Σ_l |C_il||Û_lj|, the entry's own terms,
// the scale against which a cut is refuted (classifyProbe). GEPP's backward error is |L̂||Û|-shaped and fills
// structural zeros of A, so it is not a componentwise perturbation of the
// data and data does not bound it. D is left out: it is shared by a model
// and its cut, and (Ĝ_sp + D) − D loses Ĝ_sp below ε|D|.
//
// newBoundedResponse forms comp and ver from the residual in plain double
// (its rounding bounded by γ_{k+4}·(|B| + |A||Û| + √2|s||Û|), k the
// nonzeros of a row of A: the plain complex product s·û errs by at most
// 2√2u|s||û| ≤ γ_{k+4}·√2|s||û|, the k + 2-term sum by γ_{k+1} of its
// terms) and the output product in plain double; sharp
// marks the points where sharpen has replaced them by the compensated
// bounds. A looser bound can fail to verify but never verifies wrongly, so
// classifyProbe sharpens only the points the plain bound does not verify.
// Responses and bounds are invariant under the exact power-of-two
// balancing the solve uses. ok is false at points where GEPP meets an
// exactly zero pivot.
type boundedResponse struct {
	p, m  int
	g     []complex128
	comp  []float64
	ver   []float64
	data  []float64
	terms []float64
	ok    []bool
	sharp []bool
	ev    *responseEvaluator
}

// responseEvaluator keeps what sharpen needs per model: the balanced
// realization, the nonzeros of Â by row, Û and |L̂ᵀ| per point, and,
// formed at the first sharpen, the Veltkamp splits of Â's nonzeros and of
// Ĉ and the row constants of the compensated residual bound.
type responseEvaluator struct {
	br       balancedRealization
	n        int
	pts      []complex128
	rows     []int
	cols     []int
	aval     []float64
	gamP     []float64
	ahi, alo []float64
	gam2     []float64
	u        []complex128
	absV     []float64
	x        []complex128
	piv      []int
	lu       []complex128
	lupiv    []int
	chi, clo []float64
	scratch  []float64
	cscratch []complex128
	buf      *responseBuffers
}

// responseBuffers holds the storage of a boundedResponse for reuse across
// band calls (release), which otherwise dominate the band's cost on small
// models by allocation.
type responseBuffers struct {
	ints  []int
	f, fs []float64
	c, cs []complex128
	flags []bool
	lu    []complex128
	lupiv []int
	ev    responseEvaluator
}

var responseBufferPool = sync.Pool{New: func() any { return new(responseBuffers) }}

// zeroed is s resized to n zero elements, reallocated only when short.
func zeroed[T any](s []T, n int) []T {
	if cap(s) < n {
		return make([]T, n)
	}
	s = s[:n]
	clear(s)
	return s
}

// release returns b's storage for reuse; b must not be used afterwards.
func (b *boundedResponse) release() {
	if b.ev == nil || b.ev.buf == nil {
		return
	}
	buf := b.ev.buf
	buf.ev = responseEvaluator{}
	*b = boundedResponse{}
	responseBufferPool.Put(buf)
}

func gammaN(k int) float64 {
	ku := float64(k) * eps()
	return ku / (1 - ku)
}

func newBoundedResponse(sys *System, pts []complex128) boundedResponse {
	n, m, p := sys.Dims()
	return boundedResponseOf(newBalancedRealization(sys, n, m, p), pts)
}

// boundedResponseOf is newBoundedResponse for the explicit realization br,
// taken as is.
func boundedResponseOf(rz balancedRealization, pts []complex128) boundedResponse {
	return boundedResponseKeep(rz, pts, false)
}

// boundedResponseKeep is boundedResponseOf keeping, with keep, the GEPP
// factors of every point for factorsAt and sharpenAt.
func boundedResponseKeep(rz balancedRealization, pts []complex128, keep bool) boundedResponse {
	n, m, p := rz.n, rz.m, rz.p
	np, pm := len(pts), p*m
	buf := responseBufferPool.Get().(*responseBuffers)
	ev := &buf.ev
	*ev = responseEvaluator{br: rz, n: n, pts: pts, buf: buf}
	br := &ev.br
	nnz := 0
	for _, a := range br.a {
		if a != 0 {
			nnz++
		}
	}
	buf.ints = zeroed(buf.ints, 2*n+1+nnz)
	ints := buf.ints
	ev.rows, ev.cols, ev.piv = ints[:n+1], ints[n+1:n+1+nnz], ints[n+1+nnz:]
	buf.f = zeroed(buf.f, nnz+n+4*np*pm+np*n*p+3*n*m)
	f := buf.f
	ev.aval, ev.gamP, f = f[:nnz], f[nnz:nnz+n], f[nnz+n:]
	nnz = 0
	for i := range n {
		for l, a := range br.a[i*n : (i+1)*n] {
			if a != 0 {
				ev.cols[nnz], ev.aval[nnz] = l, a
				nnz++
			}
		}
		ev.rows[i+1] = nnz
		ev.gamP[i] = gammaN(nnz - ev.rows[i] + 4)
	}
	out := boundedResponse{
		p: p, m: m,
		comp:  f[:np*pm],
		ver:   f[np*pm : 2*np*pm],
		data:  f[2*np*pm : 3*np*pm],
		terms: f[3*np*pm : 4*np*pm],
		ev:    ev,
	}
	buf.flags = zeroed(buf.flags, 2*np)
	flags := buf.flags
	out.ok, out.sharp = flags[:np], flags[np:]
	f = f[4*np*pm:]
	ev.absV, f = f[:np*n*p], f[np*n*p:]
	au, absU, rb := f[:n*m], f[n*m:2*n*m], f[2*n*m:]
	buf.c = zeroed(buf.c, np*pm+np*n*m+n*n+n*p)
	out.g = buf.c[:np*pm]
	w := buf.c[np*pm:]
	ev.u, ev.x = w[:np*n*m], w[np*n*m:np*n*m+n*n]
	v := w[np*n*m+n*n:]
	if keep {
		buf.lu, buf.lupiv = zeroed(buf.lu, np*n*n), zeroed(buf.lupiv, np*n)
		ev.lu, ev.lupiv = buf.lu, buf.lupiv
	}
	safety, gn := 1+2*float64(n+4)*eps(), gammaN(n)
	for k, s := range pts {
		if !ev.factor(s) {
			continue
		}
		if keep {
			copy(ev.lu[k*n*n:(k+1)*n*n], ev.x)
			copy(ev.lupiv[k*n:(k+1)*n], ev.piv)
		}
		u, absV := ev.u[k*n*m:(k+1)*n*m], ev.absV[k*n*p:(k+1)*n*p]
		for i, b := range br.b {
			u[i] = complex(b, 0)
		}
		for i := range n {
			for j := range p {
				v[i*p+j] = complex(br.c[j*n+i], 0)
			}
		}
		cgeppSolve(ev.x, n, ev.piv, u, m)
		cgeppSolveT(ev.x, n, ev.piv, v, p)
		ev.absProducts(u, absU, au, m)
		sr, si, as := real(s), imag(s), math.Sqrt2*cmplx.Abs(s)
		for i := range n {
			for j := range p {
				absV[i*p+j] = cabs(v[i*p+j])
			}
			for j := range m {
				b := br.b[i*m+j]
				re, im := b, 0.0
				for q := ev.rows[i]; q < ev.rows[i+1]; q++ {
					a, z := ev.aval[q], u[ev.cols[q]*m+j]
					re += a * real(z)
					im += a * imag(z)
				}
				z := u[i*m+j]
				re += si*imag(z) - sr*real(z)
				im -= sr*imag(z) + si*real(z)
				rb[i*m+j] = cabs(complex(re, im)) + ev.gamP[i]*(math.Abs(b)+au[i*m+j]+as*absU[i*m+j])
			}
		}
		o := k * pm
		for i := range p {
			ci := br.c[i*n : (i+1)*n]
			for j := range m {
				var gr, gi, t, tc, cu float64
				for l, c := range ci {
					z, al := u[l*m+j], absV[l*p+i]
					gr += c * real(z)
					gi += c * imag(z)
					cl := math.Abs(c) * absU[l*m+j]
					t += al*(au[l*m+j]+math.Abs(br.b[l*m+j])) + cl
					tc += al * rb[l*m+j]
					cu += cl
				}
				out.g[o+i*m+j] = complex(gr, gi)
				out.data[o+i*m+j] = eps() * t
				out.terms[o+i*m+j] = cu
				bnd := safety * (tc + gn*cu)
				out.comp[o+i*m+j], out.ver[o+i*m+j] = bnd, bnd
			}
		}
		out.ok[k] = true
	}
	return out
}

// factor overwrites ev.x with the GEPP factors of sI − Â, pivots in
// ev.piv; false at an exactly zero pivot.
func (ev *responseEvaluator) factor(s complex128) bool {
	x := ev.x
	for i, a := range ev.br.a {
		x[i] = complex(-a, 0)
	}
	for i := range ev.n {
		x[i*ev.n+i] += s
	}
	return cgeppFactor(x, ev.n, ev.piv)
}

// factorsAt returns the GEPP factors and pivots of point k, which factored
// without a zero pivot, as factor forms them: the kept ones, else factor's
// own in ev.x and ev.piv.
func (ev *responseEvaluator) factorsAt(k int) ([]complex128, []int) {
	n := ev.n
	if ev.lu != nil {
		return ev.lu[k*n*n : (k+1)*n*n], ev.lupiv[k*n : (k+1)*n]
	}
	ev.factor(ev.pts[k])
	return ev.x, ev.piv
}

// prepareSharpen forms the splits, row constants and scratch of sharpen.
func (ev *responseEvaluator) prepareSharpen() {
	br := &ev.br
	n, m, p, nnz := ev.n, br.m, br.p, len(ev.aval)
	nm, pm := n*m, p*m
	ev.buf.fs = zeroed(ev.buf.fs, 2*nnz+n+9*nm+3*pm+2*p*n+n*n)
	f := ev.buf.fs
	ev.ahi, ev.alo, ev.gam2, f = f[:nnz], f[nnz:2*nnz], f[2*nnz:2*nnz+n], f[2*nnz+n:]
	for q, a := range ev.aval {
		ev.ahi[q], ev.alo[q] = veltkampSplit(a)
	}
	for i := range n {
		g := gammaN(ev.rows[i+1] - ev.rows[i] + 3)
		ev.gam2[i] = 2 * g * g
	}
	ev.scratch = f
	ev.buf.cs = zeroed(ev.buf.cs, 2*nm+pm)
	ev.cscratch = ev.buf.cs
	cs := f[9*nm+3*pm:]
	ev.chi, ev.clo = cs[:p*n], cs[p*n:2*p*n]
	for i, c := range br.c {
		ev.chi[i], ev.clo[i] = veltkampSplit(c)
	}
}

// absProducts sets absU = |u| and au = |Â||u| for the n×m u.
func (ev *responseEvaluator) absProducts(u []complex128, absU, au []float64, m int) {
	for i, z := range u {
		absU[i] = cabs(z)
	}
	clear(au)
	for i := range ev.n {
		ai := au[i*m : (i+1)*m]
		for q := ev.rows[i]; q < ev.rows[i+1]; q++ {
			a := math.Abs(ev.aval[q])
			for j, x := range absU[ev.cols[q]*m : (ev.cols[q]+1)*m] {
				ai[j] += a * x
			}
		}
	}
}

// sharpen replaces the plain bounds of point k by the compensated ones
// (comp and ver as documented on boundedResponse), applying û += δ once,
// δ = M̂⁻¹r̂ the computed correction, when ver > 2²⁰·ε·|Ĝ_sp| at some entry:
// the bounds hold for any û, so a refined one only sharpens them. The
// residual r̂ = B + ÂÛ − sÛ and Ĝ_sp = ĈÛ are formed by Dot2 (Ogita, Rump,
// Oishi 2005) with Dekker's TwoProduct, so their error is u|r| + γ²_K·Σ|terms|
// per component, K the term count, and R = (1+2u)|r̂| + 2γ²_{k+3}·ρ, ρ =
// |B| + |Â||Û| + |s||Û|, bounds the exact residual. With G_sp − ĈÛ = L r
// exactly, comp = (1+2(n+4)u)·(|L̂|R + u(1+2u)|Ĝ_sp| + 2γ²_n|Ĉ||Û|). The
// error of L̂ adds γ_{18n}·|L̂|Pᵀ(|L̂_f|(|Û_f||δ|)) to ver (Higham, Accuracy
// and Stability, Thm 9.4 for complex GEPP on PM = L̂_fÛ_f), to first order
// in the solve's error. A non-finite result leaves the point unverifiable:
// comp = ver = +Inf with the plain response.
func (b *boundedResponse) sharpen(k int) { b.sharpenAt(k, true) }

// sharpenAt is sharpen with the refinement step allowed or not.
func (b *boundedResponse) sharpenAt(k int, refinement bool) {
	if b.sharp[k] || !b.ok[k] {
		return
	}
	b.sharp[k] = true
	ev, m, p := b.ev, b.m, b.p
	n, br, s := ev.n, &ev.br, ev.pts[k]
	nm, pm := n*m, p*m
	if ev.scratch == nil {
		ev.prepareSharpen()
	}
	sc := ev.scratch
	absU, au, rb, w, vv := sc[:nm], sc[nm:2*nm], sc[2*nm:3*nm], sc[3*nm:4*nm], sc[4*nm:5*nm]
	uh, ul := sc[5*nm:7*nm], sc[7*nm:9*nm]
	comp, ver, data := sc[9*nm:9*nm+pm], sc[9*nm+pm:9*nm+2*pm], sc[9*nm+2*pm:9*nm+3*pm]
	absX := sc[9*nm+3*pm+2*p*n:]
	r, del, g := ev.cscratch[:nm], ev.cscratch[nm:2*nm], ev.cscratch[2*nm:]
	x, piv := ev.factorsAt(k)
	haveAbsX := false
	u, absV := ev.u[k*nm:(k+1)*nm], ev.absV[k*n*p:(k+1)*n*p]
	e := eps()
	safety := 1 + 2*float64(n+4)*e
	gt, gn2 := gammaN(18*n), 2*gammaN(n)*gammaN(n)
	thr := 0x1p20 * e
	for pass := 0; ; pass++ {
		ev.residual(s, u, uh, ul, absU, au, r, rb, m)
		copy(del, r)
		cgeppSolve(x, n, piv, del, m)
		need := false
		for i := range p {
			ci := br.c[i*n : (i+1)*n]
			for j := range m {
				var gr, gi dot2Acc
				var t, tc, cu float64
				for l, c := range ci {
					al := absV[l*p+i]
					y := l*m + j
					if c != 0 {
						ch, cl := ev.chi[i*n+l], ev.clo[i*n+l]
						gr.add(twoProdSplit(c, ch, cl, real(u[y]), uh[2*y], ul[2*y]))
						gi.add(twoProdSplit(c, ch, cl, imag(u[y]), uh[2*y+1], ul[2*y+1]))
					}
					cul := math.Abs(c) * absU[y]
					t += al*(au[y]+math.Abs(br.b[y])) + cul
					tc += al * rb[y]
					cu += cul
				}
				y := i*m + j
				g[y] = complex(gr.val(), gi.val())
				comp[y] = safety * (tc + e*(1+2*e)*cabs(g[y]) + gn2*cu)
				data[y] = e * t
				b.terms[k*pm+y] = cu
				need = need || comp[y] > thr*cabs(g[y])
			}
		}
		refine := refinement && pass == 0 && finiteVec(del)
		if !(need && refine) {
			if !haveAbsX {
				for i, z := range x {
					absX[i] = cabs(z)
				}
				haveAbsX = true
			}
			aprioriTerm(absX, piv, del, w, vv, n, m)
			for i := range p {
				for j := range m {
					ta := 0.0
					for l := range n {
						ta += absV[l*p+i] * vv[l*m+j]
					}
					y := i*m + j
					ver[y] = comp[y] + safety*gt*ta
					need = need || ver[y] > thr*cabs(g[y])
				}
			}
		}
		if !(need && refine) {
			break
		}
		for i := range u {
			u[i] += del[i]
		}
	}
	o := k * pm
	for y := range pm {
		if math.IsInf(ver[y], 0) || math.IsNaN(ver[y]) || cmplx.IsNaN(g[y]) || cmplx.IsInf(g[y]) {
			for q := range pm {
				b.comp[o+q], b.ver[o+q] = math.Inf(1), math.Inf(1)
			}
			return
		}
	}
	copy(b.g[o:o+pm], g)
	copy(b.comp[o:o+pm], comp)
	copy(b.ver[o:o+pm], ver)
	copy(b.data[o:o+pm], data)
}

// stage1 returns the strictly proper response at point k and its comp
// bound as sharpen forms them in its first pass, by Dot2 for the current û,
// without refactoring sI − Â, refining û or forming the a-priori term: the
// bound holds for any û, so it is as sound as sharpen's. The slices are
// scratch of b, valid until its next stage1 or sharpen; b is unchanged.
func (b *boundedResponse) stage1(k int) ([]complex128, []float64) {
	ev, m, p := b.ev, b.m, b.p
	n, br, s := ev.n, &ev.br, ev.pts[k]
	nm, pm := n*m, p*m
	if ev.scratch == nil {
		ev.prepareSharpen()
	}
	sc := ev.scratch
	absU, au, rb := sc[:nm], sc[nm:2*nm], sc[2*nm:3*nm]
	uh, ul := sc[5*nm:7*nm], sc[7*nm:9*nm]
	comp := sc[9*nm : 9*nm+pm]
	r, g := ev.cscratch[:nm], ev.cscratch[2*nm:]
	u, absV := ev.u[k*nm:(k+1)*nm], ev.absV[k*n*p:(k+1)*n*p]
	e := eps()
	safety, gn2 := 1+2*float64(n+4)*e, 2*gammaN(n)*gammaN(n)
	ev.residual(s, u, uh, ul, absU, au, r, rb, m)
	for i := range p {
		ci := br.c[i*n : (i+1)*n]
		for j := range m {
			var gr, gi dot2Acc
			var tc, cu float64
			for l, c := range ci {
				y := l*m + j
				if c != 0 {
					ch, cl := ev.chi[i*n+l], ev.clo[i*n+l]
					gr.add(twoProdSplit(c, ch, cl, real(u[y]), uh[2*y], ul[2*y]))
					gi.add(twoProdSplit(c, ch, cl, imag(u[y]), uh[2*y+1], ul[2*y+1]))
				}
				tc += absV[l*p+i] * rb[y]
				cu += math.Abs(c) * absU[y]
			}
			y := i*m + j
			g[y] = complex(gr.val(), gi.val())
			comp[y] = safety * (tc + e*(1+2*e)*cabs(g[y]) + gn2*cu)
		}
	}
	return g, comp
}

// residual sets r = B̂ + ÂU − sU by Dot2 and rb to the bound R of sharpen
// on the exact residual, |r − r̂| ≤ u|r| + 2γ²_{k+3}·ρ, for the n×m U,
// with absU = |U|, au = |Â||U| and the Veltkamp splits uh, ul of U's
// real and imaginary parts, interleaved.
func (ev *responseEvaluator) residual(s complex128, u []complex128, uh, ul, absU, au []float64, r []complex128, rb []float64, m int) {
	br, e := &ev.br, eps()
	ev.absProducts(u, absU, au, m)
	for i, z := range u {
		uh[2*i], ul[2*i] = veltkampSplit(real(z))
		uh[2*i+1], ul[2*i+1] = veltkampSplit(imag(z))
	}
	sr, si, as := real(s), imag(s), cmplx.Abs(s)
	srh, srl := veltkampSplit(sr)
	sih, sil := veltkampSplit(si)
	for i := range ev.n {
		for j := range m {
			bij := br.b[i*m+j]
			dr, di := dot2Acc{hi: bij}, dot2Acc{}
			for q := ev.rows[i]; q < ev.rows[i+1]; q++ {
				a, ah, al := ev.aval[q], ev.ahi[q], ev.alo[q]
				y := ev.cols[q]*m + j
				z := u[y]
				dr.add(twoProdSplit(a, ah, al, real(z), uh[2*y], ul[2*y]))
				di.add(twoProdSplit(a, ah, al, imag(z), uh[2*y+1], ul[2*y+1]))
			}
			y := i*m + j
			z := u[y]
			if sr != 0 {
				dr.add(twoProdSplit(-sr, -srh, -srl, real(z), uh[2*y], ul[2*y]))
				di.add(twoProdSplit(-sr, -srh, -srl, imag(z), uh[2*y+1], ul[2*y+1]))
			}
			if si != 0 {
				dr.add(twoProdSplit(si, sih, sil, imag(z), uh[2*y+1], ul[2*y+1]))
				di.add(twoProdSplit(-si, -sih, -sil, real(z), uh[2*y], ul[2*y]))
			}
			r[y] = complex(dr.val(), di.val())
			rb[y] = (1+2*e)*cabs(r[y]) + ev.gam2[i]*(math.Abs(bij)+au[y]+as*absU[y])
		}
	}
}

// aprioriTerm sets vv = Pᵀ(|L̂_f|(|Û_f||δ|)) for the moduli absX of the
// factors of cgeppFactor, the permutation applied as in cgeppSolveT; w is
// scratch.
func aprioriTerm(absX []float64, piv []int, del []complex128, w, vv []float64, n, m int) {
	for i, z := range del {
		vv[i] = cabs(z)
	}
	clear(w)
	for l := range n {
		wl := w[l*m : (l+1)*m]
		for q := l; q < n; q++ {
			if a := absX[l*n+q]; a != 0 {
				for j, d := range vv[q*m : (q+1)*m] {
					wl[j] += a * d
				}
			}
		}
	}
	for i := range n {
		vi := vv[i*m : (i+1)*m]
		copy(vi, w[i*m:(i+1)*m])
		for l := range i {
			if a := absX[i*n+l]; a != 0 {
				for j, d := range w[l*m : (l+1)*m] {
					vi[j] += a * d
				}
			}
		}
	}
	for q := n - 1; q >= 0; q-- {
		if pv := piv[q]; pv != q {
			for j := range m {
				vv[q*m+j], vv[pv*m+j] = vv[pv*m+j], vv[q*m+j]
			}
		}
	}
}

// cabs is |z|, by math.Hypot only where the squares leave the normal range.
func cabs(z complex128) float64 {
	re, im := real(z), imag(z)
	if s := re*re + im*im; s > 0x1p-1000 && s < 0x1p1000 {
		return math.Sqrt(s)
	}
	return math.Hypot(re, im)
}

func finiteVec(v []complex128) bool {
	for _, z := range v {
		if cmplx.IsNaN(z) || cmplx.IsInf(z) {
			return false
		}
	}
	return true
}

// veltkampSplit splits a into hi + lo with 26 significant bits each; the
// explicit conversions stop the compiler fusing the products.
func veltkampSplit(a float64) (hi, lo float64) {
	c := float64(134217729 * a)
	hi = float64(c - float64(c-a))
	return hi, float64(a - hi)
}

// twoProdSplit is Dekker's TwoProduct a·b = p + e, exact without overflow
// or underflow, for the splits of a and b.
func twoProdSplit(a, ah, al, b, bh, bl float64) (p, e float64) {
	p = float64(a * b)
	e = float64(float64(float64(float64(ah*bh)-p)+float64(ah*bl)) + float64(al*bh))
	return p, float64(e + float64(al*bl))
}

// dot2Acc accumulates TwoProduct terms by Ogita, Rump and Oishi's Dot2.
type dot2Acc struct{ hi, lo float64 }

func (d *dot2Acc) add(p, e float64) {
	s, f := twoSum(d.hi, p)
	d.hi, d.lo = s, d.lo+(f+e)
}

func (d dot2Acc) val() float64 { return d.hi + d.lo }
