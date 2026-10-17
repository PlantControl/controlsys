package controlsys

import (
	"container/heap"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
)

// repeatedGap is the relative gap λ0 − λ1 <= repeatedGap·λ0 below which
// the top eigenvalue of g0^H·g0 counts as repeated in gramPair.expand.
const repeatedGap = 1e-6

// gramPair expands the top eigenvalue of the Gram matrix of a p×m matrix
// gain G(t) = g0 + t·g1 + t²·g2 + …, σ_max(G(t))², about t = 0. With
// n = min(p, m) and G replaced by G^H when p < m, M(t) = G^H·G =
// M0 + t·M1 + t²·M2 + …, M0 = g0^H·g0, M1 = g0^H·g1 + g1^H·g0 and
// M2 = g1^H·g1 + g0^H·g2 + g2^H·g0, and for a simple top eigenvalue λ0 of
// M0 with unit eigenvector v, λ(t) = λ0 + t·λ1 + t²·λ2 + O(t³) with
//
//	λ1 = v^H·M1·v,  λ2 = v^H·M2·v + w^H·(λ0·I − M0)⁺·w,  w = M1·v − λ1·v,
//
// the pseudo-inverse applied as (λ0·I − M0 + λ0·v·v^H)⁻¹, since w ⊥ v.
type gramPair struct {
	n, p, m          int
	block, eig, work []float64
	m0, m1, m2       []complex128
	lu               []complex128
	piv              []int
	v, w, y          []complex128
	reach            float64 // see expand
}

func newGramPair(p, m int) *gramPair {
	n := min(p, m)
	block, eig, q := make([]float64, 4*n*n), make([]float64, 2*n), make([]float64, 1)
	impl.Dsyev(lapack.EVCompute, blas.Upper, 2*n, block, 2*n, eig, q, -1)
	return &gramPair{
		n: n, p: p, m: m,
		block: block, eig: eig, work: make([]float64, int(q[0])),
		m0: make([]complex128, n*n), m1: make([]complex128, n*n), m2: make([]complex128, n*n),
		lu: make([]complex128, n*n), piv: make([]int, n),
		v: make([]complex128, n), w: make([]complex128, n), y: make([]complex128, n),
	}
}

// addGram adds A^H·B to dst for the p×m flats a, b, or A·B^H when p < m.
func (gp *gramPair) addGram(dst, a, b []complex128) {
	n, p, m := gp.n, gp.p, gp.m
	for i := range n {
		for j := range n {
			var s complex128
			if m <= p {
				for r := range p {
					s += cmplx.Conj(a[r*m+i]) * b[r*m+j]
				}
			} else {
				for c := range m {
					s += a[i*m+c] * cmplx.Conj(b[j*m+c])
				}
			}
			dst[i*n+j] += s
		}
	}
}

// expand returns λ0 = σ_max(g0)², λ1, λ2 and the relative gap
// (λ0 − λ_next)/λ0 of M0 = g0^H·g0, and sets reach = (λ0 − λ_next)/(4·‖M1‖_F),
// a t below which the eigenvalues of M(t) stay apart, so that λ0 + t·λ1 +
// t²·λ2 describes the top one. ok is false where the top eigenvalue is
// repeated to repeatedGap, λ1 then being the largest of its splitting only
// in direction, so λ1 and λ2 are not returned, or the eigensolver fails.
func (gp *gramPair) expand(g0, g1, g2 []complex128) (l0, l1, l2, gap float64, ok bool) {
	n := gp.n
	clear(gp.m0)
	clear(gp.m1)
	clear(gp.m2)
	gp.addGram(gp.m0, g0, g0)
	gp.addGram(gp.m1, g0, g1)
	gp.addGram(gp.m1, g1, g0)
	gp.addGram(gp.m2, g1, g1)
	gp.addGram(gp.m2, g0, g2)
	gp.addGram(gp.m2, g2, g0)
	scale := 0.0
	for _, z := range gp.m0 {
		scale = max(scale, cmplx.Abs(z))
	}
	if scale == 0 || math.IsInf(scale, 0) || math.IsNaN(scale) {
		return 0, 0, 0, 0, false
	}
	s := 2 * n
	for i := range n {
		for j := range n {
			z := gp.m0[i*n+j] / complex(scale, 0)
			gp.block[i*s+j] = real(z)
			gp.block[i*s+n+j] = -imag(z)
			gp.block[(n+i)*s+j] = imag(z)
			gp.block[(n+i)*s+n+j] = real(z)
		}
	}
	if !impl.Dsyev(lapack.EVCompute, blas.Upper, s, gp.block, s, gp.eig, gp.work, len(gp.work)) {
		return 0, 0, 0, 0, false
	}
	l0 = gp.eig[s-1] * scale
	gap = 1.0
	if n > 1 {
		gap = (gp.eig[s-1] - gp.eig[s-3]) / gp.eig[s-1]
	}
	if !(gap > repeatedGap) {
		return l0, 0, 0, gap, false
	}
	var m1n float64
	for _, z := range gp.m1 {
		m1n += real(z)*real(z) + imag(z)*imag(z)
	}
	gp.reach = gap * l0 / (4 * math.Sqrt(m1n))
	var nv float64
	for i := range n {
		gp.v[i] = complex(gp.block[i*s+s-1], gp.block[(n+i)*s+s-1])
		nv += real(gp.v[i])*real(gp.v[i]) + imag(gp.v[i])*imag(gp.v[i])
	}
	nv = math.Sqrt(nv)
	for i := range gp.v {
		gp.v[i] /= complex(nv, 0)
	}
	quad := func(a []complex128) (float64, []complex128) {
		var q complex128
		for i := range n {
			var r complex128
			for j := range n {
				r += a[i*n+j] * gp.v[j]
			}
			gp.w[i] = r
			q += cmplx.Conj(gp.v[i]) * r
		}
		return real(q), gp.w
	}
	l2, _ = quad(gp.m2)
	l1, w := quad(gp.m1)
	for i := range n {
		w[i] -= complex(l1, 0) * gp.v[i]
		for j := range n {
			gp.lu[i*n+j] = -gp.m0[i*n+j] + complex(l0, 0)*gp.v[i]*cmplx.Conj(gp.v[j])
		}
		gp.lu[i*n+i] += complex(l0, 0)
	}
	y := gp.y
	copy(y, w)
	if !cLUFactor(gp.lu, gp.piv, n) {
		return l0, 0, 0, gap, false
	}
	cLUSolve(gp.lu, gp.piv, y, n, 1)
	for i := range n {
		l2 += real(cmplx.Conj(w[i]) * y[i])
	}
	return l0, l1, l2, gap, true
}

// hermTop returns the eigenvalues of the n×n Hermitian flat h in
// descending order and, as the columns of the n×k flat basis, an
// orthonormal basis of the eigenvectors of the k largest, those within
// rel·|λ_max| + abs of λ_max. ok is false where the eigensolver fails.
func hermTop(h []complex128, n int, rel, abs float64) (vals []float64, basis []complex128, k int, ok bool) {
	s := 2 * n
	b, e, q := make([]float64, s*s), make([]float64, s), make([]float64, 1)
	for i := range n {
		for j := range n {
			z := h[i*n+j]
			b[i*s+j], b[i*s+n+j] = real(z), -imag(z)
			b[(n+i)*s+j], b[(n+i)*s+n+j] = imag(z), real(z)
		}
	}
	impl.Dsyev(lapack.EVCompute, blas.Upper, s, b, s, e, q, -1)
	if !impl.Dsyev(lapack.EVCompute, blas.Upper, s, b, s, e, make([]float64, int(q[0])), int(q[0])) {
		return nil, nil, 0, false
	}
	vals = make([]float64, n)
	for i := range n {
		vals[i] = e[s-1-2*i]
	}
	tol := rel*math.Abs(vals[0]) + abs
	for k < n && vals[k] >= vals[0]-tol {
		k++
	}
	// The real embedding doubles each eigenvalue: its top 2k eigenvectors
	// x + j·y span the eigenspace twice over (u and j·u), so complex
	// Gram–Schmidt picking the largest residual keeps k of them.
	cand := make([][]complex128, 2*k)
	for c := range cand {
		v := make([]complex128, n)
		for i := range n {
			v[i] = complex(b[i*s+s-1-c], b[(n+i)*s+s-1-c])
		}
		cand[c] = v
	}
	basis = make([]complex128, n*k)
	for j := range k {
		best, bn := -1, 0.0
		for c, v := range cand {
			if v != nil {
				if nv := cFrobenius(v); nv > bn {
					best, bn = c, nv
				}
			}
		}
		if best < 0 {
			return nil, nil, 0, false
		}
		u := cand[best]
		cand[best] = nil
		for i := range n {
			basis[i*k+j] = u[i] / complex(bn, 0)
		}
		for range 2 {
			for _, v := range cand {
				if v == nil {
					continue
				}
				var d complex128
				for i := range n {
					d += cmplx.Conj(basis[i*k+j]) * v[i]
				}
				for i := range n {
					v[i] -= d * basis[i*k+j]
				}
			}
		}
	}
	return vals, basis, k, true
}

// clusterPencil extends expand to a top eigenvalue λ0 of M0 repeated to
// repeatedGap, with eigenspace V, along the paths φ = φ* + s·t through a
// maximiser φ* of λ0(φ): G = g0 + t·(g1 + s·d0) + t²·(g2 + s·d1 + s²/2·dd0)
// + O(t³), d0 = g0′, d1 = g1′, dd0 = g0″ in φ. The eigenvalues of M(t)
// near λ0 are λ0 + t·μ + t²·ν + O(t³), μ those of A1 = V^H·M1·V, s-free
// since V^H·M0′·V = 0 at a maximiser. The top branches keep the
// eigenspace C of μ1 = λ_max(A1), to tolMu, of dimension r, and there ν
// are the eigenvalues of K(s) = U^H·M2·U + W^H·(λ0·I − M0)⁺·W, U = V·C,
// W = M1·U − V·V^H·M1·U (degenerate perturbation with the reduced
// resolvent outside the cluster), applied as (λ0·I − M0 + λ0·V·V^H)⁻¹.
// K(s) = Q0 + s·Q1 + s²·Q2 is quadratic in s; q holds the r×r Q_i. reach
// is as for expand with λ_next the largest eigenvalue below the cluster,
// +Inf without one.
func (gp *gramPair) clusterPencil(g0, g1, g2, d0, d1, dd0 []complex128, tolMu float64) (q [3][]complex128, r int, mu1, reach float64, ok bool) {
	n := gp.n
	clear(gp.m0)
	gp.addGram(gp.m0, g0, g0)
	vals, v, k, ok := hermTop(gp.m0, n, repeatedGap, 0)
	if !ok || !(vals[0] > 0) {
		return q, 0, 0, 0, false
	}
	l0 := vals[0]
	pm := len(g0)
	g1s, g2s := make([]complex128, pm), make([]complex128, pm)
	path := func(s float64) {
		for i := range pm {
			g1s[i] = g1[i] + complex(s, 0)*d0[i]
			g2s[i] = g2[i] + complex(s, 0)*d1[i] + complex(s*s/2, 0)*dd0[i]
		}
		clear(gp.m1)
		clear(gp.m2)
		gp.addGram(gp.m1, g0, g1s)
		gp.addGram(gp.m1, g1s, g0)
		gp.addGram(gp.m2, g1s, g1s)
		gp.addGram(gp.m2, g0, g2s)
		gp.addGram(gp.m2, g2s, g0)
	}
	// mul returns X^H·A·Y for the n×n A, n×cx X and n×cy Y.
	mul := func(x, a, y []complex128, cx, cy int) []complex128 {
		out := make([]complex128, cx*cy)
		for i := range cx {
			for j := range cy {
				var acc complex128
				for row := range n {
					var ay complex128
					for col := range n {
						ay += a[row*n+col] * y[col*cy+j]
					}
					acc += cmplx.Conj(x[row*cx+i]) * ay
				}
				out[i*cy+j] = acc
			}
		}
		return out
	}
	path(0)
	muVals, c, r, ok := hermTop(mul(v, gp.m1, v, k, k), k, 0, tolMu)
	if !ok {
		return q, 0, 0, 0, false
	}
	mu1 = muVals[0]
	u := make([]complex128, n*r)
	for i := range n {
		for j := range r {
			for l := range k {
				u[i*r+j] += v[i*k+l] * c[l*r+j]
			}
		}
	}
	for i := range n {
		for j := range n {
			var vv complex128
			for l := range k {
				vv += v[i*k+l] * cmplx.Conj(v[j*k+l])
			}
			gp.lu[i*n+j] = -gp.m0[i*n+j] + complex(l0, 0)*vv
		}
		gp.lu[i*n+i] += complex(l0, 0)
	}
	if !cLUFactor(gp.lu, gp.piv, n) {
		return q, 0, 0, 0, false
	}
	reach = math.Inf(1)
	if k < n {
		reach = (l0 - vals[k]) / (4 * cFrobenius(gp.m1))
	}
	w, y, vw := make([]complex128, n*r), make([]complex128, n*r), make([]complex128, k*r)
	var kAt [3][]complex128
	for idx, s := range []float64{-1, 0, 1} {
		path(s)
		k2 := mul(u, gp.m2, u, r, r)
		clear(vw)
		for i := range n {
			for j := range r {
				var z complex128
				for l := range n {
					z += gp.m1[i*n+l] * u[l*r+j]
				}
				w[i*r+j] = z
				for l := range k {
					vw[l*r+j] += cmplx.Conj(v[i*k+l]) * z
				}
			}
		}
		for i := range n {
			for j := range r {
				for l := range k {
					w[i*r+j] -= v[i*k+l] * vw[l*r+j]
				}
			}
		}
		copy(y, w)
		cLUSolve(gp.lu, gp.piv, y, n, r)
		for i := range r {
			for j := range r {
				var z complex128
				for l := range n {
					z += cmplx.Conj(w[l*r+i]) * y[l*r+j]
				}
				k2[i*r+j] += z
			}
		}
		kAt[idx] = k2
	}
	for i := range q {
		q[i] = make([]complex128, r*r)
	}
	for i := range r * r {
		q[0][i] = kAt[1][i]
		q[1][i] = (kAt[2][i] - kAt[0][i]) / 2
		q[2][i] = (kAt[2][i]+kAt[0][i])/2 - kAt[1][i]
	}
	return q, r, mu1, reach, true
}

// pencilSpan is an interval [a, b] of s with ub, a bound on λ_max(K(s))
// over it.
type pencilSpan struct{ a, b, ub float64 }

type pencilSpans []pencilSpan

func (h pencilSpans) Len() int           { return len(h) }
func (h pencilSpans) Less(i, j int) bool { return h[i].ub > h[j].ub }
func (h pencilSpans) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *pencilSpans) Push(x any)        { *h = append(*h, x.(pencilSpan)) }
func (h *pencilSpans) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// pencilSup returns bounds lo <= sup_s λ_max(K(s)) <= hi, K(s) = Q0 +
// s·Q1 + s²·Q2, for the r×r Hermitian q, false unless Q2 is negative
// definite, −λ_max(Q2) = ν > 1e-9·‖Q2‖_F. The sup lies in |s| <= sB, past
// which ‖Q0‖ + |s|·‖Q1‖ − ν·s² < −‖Q0‖. For r = 1 it is Q0 + Q1²/(4ν).
// For r > 1, with s = m + δ, K(s) = K(m) + δ·D + δ²·Q2 ≼ K(m) + δ·D,
// D = Q1 + 2m·Q2, and λ_max of the right side is convex in δ, so over
// |δ| <= w/2 it is bounded by its value at δ = ±w/2. Spans are split,
// largest bound first, until hi − lo <= 1e-9·(‖Q0‖ + |lo|) or 4096
// splits; lo is the largest λ_max(K(m)) seen.
func pencilSup(q [3][]complex128, r int) (lo, hi float64, ok bool) {
	top := func(a []complex128) float64 {
		vals, _, _, ok := hermTop(a, r, 0, 0)
		if !ok {
			return math.NaN()
		}
		return vals[0]
	}
	nu := -top(q[2])
	if !(nu > 1e-9*cFrobenius(q[2])) {
		return 0, 0, false
	}
	if r == 1 {
		q1 := real(q[1][0])
		v := real(q[0][0]) + q1*q1/(4*nu)
		return v, v, true
	}
	n0, n1 := cFrobenius(q[0]), cFrobenius(q[1])
	sB := (n1 + math.Sqrt(n1*n1+8*nu*n0)) / (2 * nu)
	km, buf := make([]complex128, r*r), make([]complex128, r*r)
	lo = math.Inf(-1)
	bad := false
	span := func(a, b float64) pencilSpan {
		m, d := (a+b)/2, (b-a)/2
		for i := range km {
			km[i] = q[0][i] + complex(m, 0)*q[1][i] + complex(m*m, 0)*q[2][i]
		}
		fm := top(km)
		ub := math.Inf(-1)
		for _, sg := range []float64{-d, d} {
			for i := range buf {
				buf[i] = km[i] + complex(sg, 0)*(q[1][i]+complex(2*m, 0)*q[2][i])
			}
			ub = max(ub, top(buf))
		}
		if math.IsNaN(fm) || math.IsNaN(ub) {
			bad = true
		}
		lo = max(lo, fm)
		return pencilSpan{a, b, ub}
	}
	const initial = 64
	h := make(pencilSpans, 0, initial+4096)
	step := 2 * sB / initial
	for i := range initial {
		a := -sB + float64(i)*step
		h = append(h, span(a, a+step))
	}
	heap.Init(&h)
	for range 4096 {
		if bad {
			return 0, 0, false
		}
		if h[0].ub-lo <= 1e-9*(n0+math.Abs(lo)) {
			break
		}
		sp := heap.Pop(&h).(pencilSpan)
		m := (sp.a + sp.b) / 2
		heap.Push(&h, span(sp.a, m))
		heap.Push(&h, span(m, sp.b))
	}
	if bad {
		return 0, 0, false
	}
	return lo, max(lo, h[0].ub), true
}
