package controlsys

import (
	"container/heap"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
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

// clusterQuad is K(s) = Q0 + Σ_i s_i·Q1_i + Σ_ij s_i·s_j·Q2_ij, s ∈ R^dim,
// with r×r Hermitian Q and Q2_ij = Q2_ji.
type clusterQuad struct {
	r  int
	q0 []complex128
	q1 [][]complex128
	q2 [][][]complex128
}

// clusterPencil is clusterPencilN for one phase, q holding Q0, Q1, Q2.
func (gp *gramPair) clusterPencil(g0, g1, g2, d0, d1, dd0 []complex128, tolMu float64) (q [3][]complex128, r int, mu1, reach float64, ok bool) {
	k, mu1, reach, ok := gp.clusterPencilN(g0, g1, g2, [][]complex128{d0}, [][]complex128{d1}, [][][]complex128{{dd0}}, tolMu)
	if !ok {
		return q, 0, 0, 0, false
	}
	return [3][]complex128{k.q0, k.q1[0], k.q2[0][0]}, k.r, mu1, reach, true
}

// clusterPencilN extends expand to a top eigenvalue λ0 of M0 repeated to
// repeatedGap, with eigenspace V, along the paths θ = θ* + s·t, s ∈ R^dim,
// through a maximiser θ* of λ0(θ): G = g0 + t·(g1 + Σ s_i·d0_i) +
// t²·(g2 + Σ s_i·d1_i + ½·Σ_ij s_i·s_j·dd0_ij) + O(t³), d0_i = ∂_i g0,
// d1_i = ∂_i g1, dd0_ij = ∂_i∂_j g0, dd0[i][j] read for j <= i only. The
// eigenvalues of M(t) near λ0 are λ0 + t·μ + t²·ν + O(t³), μ those of
// A1 = V^H·M1·V, s-free since V^H·∂_i M0·V = 0 at a maximiser. The top
// branches keep the eigenspace C of μ1 = λ_max(A1), to tolMu, of
// dimension r, and there ν are the eigenvalues of K(s) = U^H·M2·U +
// W^H·(λ0·I − M0)⁺·W, U = V·C, W = M1·U − V·V^H·M1·U (degenerate
// perturbation with the reduced resolvent outside the cluster), applied
// as (λ0·I − M0 + λ0·V·V^H)⁻¹. K is quadratic in s, so its clusterQuad
// follows exactly from K at s = 0, ±e_i and e_i + e_j. reach is as for
// expand with λ_next the largest eigenvalue below the cluster, +Inf
// without one.
func (gp *gramPair) clusterPencilN(g0, g1, g2 []complex128, d0, d1 [][]complex128, dd0 [][][]complex128, tolMu float64) (kq clusterQuad, mu1, reach float64, ok bool) {
	n := gp.n
	nd := len(d0)
	clear(gp.m0)
	gp.addGram(gp.m0, g0, g0)
	vals, v, k, ok := hermTop(gp.m0, n, repeatedGap, 0)
	if !ok || !(vals[0] > 0) {
		return kq, 0, 0, false
	}
	l0 := vals[0]
	pm := len(g0)
	g1s, g2s := make([]complex128, pm), make([]complex128, pm)
	path := func(s []float64) {
		copy(g1s, g1)
		copy(g2s, g2)
		for a := range nd {
			if s[a] == 0 {
				continue
			}
			sa := complex(s[a], 0)
			for i := range pm {
				g1s[i] += sa * d0[a][i]
				g2s[i] += sa * d1[a][i]
			}
			for b := range a + 1 {
				if s[b] == 0 {
					continue
				}
				f := complex(s[a]*s[b], 0)
				if a == b {
					f = complex(s[a]*s[a]/2, 0)
				}
				for i := range pm {
					g2s[i] += f * dd0[a][b][i]
				}
			}
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
	s := make([]float64, nd)
	path(s)
	muVals, c, r, ok := hermTop(mul(v, gp.m1, v, k, k), k, 0, tolMu)
	if !ok {
		return kq, 0, 0, false
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
		return kq, 0, 0, false
	}
	reach = math.Inf(1)
	if k < n {
		reach = (l0 - vals[k]) / (4 * cFrobenius(gp.m1))
	}
	w, y, vw := make([]complex128, n*r), make([]complex128, n*r), make([]complex128, k*r)
	kAt := func() []complex128 {
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
		return k2
	}
	rr := r * r
	kq = clusterQuad{r: r, q1: make([][]complex128, nd), q2: make([][][]complex128, nd)}
	for a := range nd {
		kq.q2[a] = make([][]complex128, nd)
	}
	kq.q0 = kAt()
	for a := range nd {
		s[a] = -1
		km := kAt()
		s[a] = 1
		kp := kAt()
		s[a] = 0
		q1, q2 := make([]complex128, rr), make([]complex128, rr)
		for i := range rr {
			q1[i] = (kp[i] - km[i]) / 2
			q2[i] = (kp[i]+km[i])/2 - kq.q0[i]
		}
		kq.q1[a], kq.q2[a][a] = q1, q2
	}
	for a := range nd {
		for b := range a {
			s[a], s[b] = 1, 1
			kab := kAt()
			s[a], s[b] = 0, 0
			q2 := make([]complex128, rr)
			for i := range rr {
				q2[i] = (kab[i] - kq.q0[i] - kq.q1[a][i] - kq.q1[b][i] - kq.q2[a][a][i] - kq.q2[b][b][i]) / 2
			}
			kq.q2[a][b], kq.q2[b][a] = q2, q2
		}
	}
	return kq, mu1, reach, true
}

// pencilBox is a box [a, b] of s ∈ R^dim with ub, a bound on λ_max(K(s))
// over it.
type pencilBox struct {
	a, b []float64
	ub   float64
}

type pencilBoxes []pencilBox

func (h pencilBoxes) Len() int           { return len(h) }
func (h pencilBoxes) Less(i, j int) bool { return h[i].ub > h[j].ub }
func (h pencilBoxes) Swap(i, j int)      { h[i], h[j] = h[j], h[i] }
func (h *pencilBoxes) Push(x any)        { *h = append(*h, x.(pencilBox)) }
func (h *pencilBoxes) Pop() any {
	old := *h
	x := old[len(old)-1]
	*h = old[:len(old)-1]
	return x
}

// pencilSup is pencilSupN for K(s) = Q0 + s·Q1 + s²·Q2, s ∈ R.
func pencilSup(q [3][]complex128, r int) (lo, hi float64, ok bool) {
	return pencilSupN(clusterQuad{r: r, q0: q[0], q1: [][]complex128{q[1]}, q2: [][][]complex128{{q[2]}}})
}

// pencilSupN returns bounds lo <= sup_s λ_max(K(s)) <= hi over s ∈ R^dim,
// false unless K is uniformly concave: with B the dim·r × dim·r Hermitian
// block matrix [Q2_ij], (s⊗x)^H·B·(s⊗x) = Σ s_i·s_j·x^H·Q2_ij·x, so
// −λ_max(B) = ν > 1e-9·‖B‖_F gives Σ s_i·s_j·Q2_ij ≼ −ν·|s|²·I. The sup lies
// in |s| <= sB, past which ‖Q0‖ + |s|·‖Q1‖ − ν·|s|² < −‖Q0‖, ‖Q1‖² =
// Σ ‖Q1_i‖_F². For r = 1 it is Q0 + ¼·q1ᵀ·A⁻¹·q1, A = −Q2. For r > 1, with
// s = m + δ, K(s) ≼ K(m) + Σ δ_i·D_i, D_i = Q1_i + 2·Σ_j m_j·Q2_ij, whose
// λ_max is convex in δ, so over a box it is bounded by its largest value at
// the vertices. The cube [−sB, sB]^dim is cut into boxes, split along their
// longest side, largest bound first, until hi − lo <= 1e-9·(‖Q0‖ + |lo|)
// or 4096·dim splits; lo is the largest λ_max(K(m)) seen.
func pencilSupN(kq clusterQuad) (lo, hi float64, ok bool) {
	r, nd := kq.r, len(kq.q1)
	var tb, te, tw []float64
	top := func(a []complex128, n int) float64 {
		s := 2 * n
		if len(tb) < s*s {
			tb, te = make([]float64, s*s), make([]float64, s)
			q := make([]float64, 1)
			impl.Dsyev(lapack.EVNone, blas.Upper, s, tb, s, te, q, -1)
			tw = make([]float64, int(q[0]))
		}
		b := tb[:s*s]
		for i := range n {
			for j := range n {
				z := a[i*n+j]
				b[i*s+j], b[i*s+n+j] = real(z), -imag(z)
				b[(n+i)*s+j], b[(n+i)*s+n+j] = imag(z), real(z)
			}
		}
		if !impl.Dsyev(lapack.EVNone, blas.Upper, s, b, s, te[:s], tw, len(tw)) {
			return math.NaN()
		}
		return te[s-1]
	}
	bn := nd * r
	bm := make([]complex128, bn*bn)
	for a := range nd {
		for b := range nd {
			for i := range r {
				for j := range r {
					bm[(a*r+i)*bn+b*r+j] = kq.q2[a][b][i*r+j]
				}
			}
		}
	}
	nu := -top(bm, bn)
	if !(nu > 1e-9*cFrobenius(bm)) {
		return 0, 0, false
	}
	if r == 1 {
		if nd == 1 {
			q1 := real(kq.q1[0][0])
			v := real(kq.q0[0]) + q1*q1/(4*nu)
			return v, v, true
		}
		a := mat.NewSymDense(nd, nil)
		g := mat.NewVecDense(nd, nil)
		for i := range nd {
			g.SetVec(i, real(kq.q1[i][0]))
			for j := range i + 1 {
				a.SetSym(i, j, -real(kq.q2[i][j][0]))
			}
		}
		var ch mat.Cholesky
		y := mat.NewVecDense(nd, nil)
		if !ch.Factorize(a) || ch.SolveVecTo(y, g) != nil {
			return 0, 0, false
		}
		v := real(kq.q0[0]) + mat.Dot(g, y)/4
		return v, v, true
	}
	n0 := cFrobenius(kq.q0)
	var n1 float64
	for _, q := range kq.q1 {
		n1 = math.Hypot(n1, cFrobenius(q))
	}
	sB := (n1 + math.Sqrt(n1*n1+8*nu*n0)) / (2 * nu)
	rr := r * r
	km, buf, dm := make([]complex128, rr), make([]complex128, rr), make([][]complex128, nd)
	for i := range dm {
		dm[i] = make([]complex128, rr)
	}
	m, d := make([]float64, nd), make([]float64, nd)
	lo = math.Inf(-1)
	bad := false
	box := func(a, b []float64) pencilBox {
		for i := range nd {
			m[i], d[i] = (a[i]+b[i])/2, (b[i]-a[i])/2
		}
		copy(km, kq.q0)
		for i := range nd {
			mi := complex(m[i], 0)
			copy(dm[i], kq.q1[i])
			for l := range rr {
				km[l] += mi * kq.q1[i][l]
			}
			for j := range nd {
				f, g := complex(m[i]*m[j], 0), complex(2*m[j], 0)
				for l := range rr {
					km[l] += f * kq.q2[i][j][l]
					dm[i][l] += g * kq.q2[i][j][l]
				}
			}
		}
		fm := top(km, r)
		ub := math.Inf(-1)
		for vx := range 1 << nd {
			copy(buf, km)
			for i := range nd {
				sg := complex(-d[i], 0)
				if vx>>i&1 == 1 {
					sg = complex(d[i], 0)
				}
				for l := range rr {
					buf[l] += sg * dm[i][l]
				}
			}
			ub = max(ub, top(buf, r))
		}
		if math.IsNaN(fm) || math.IsNaN(ub) {
			bad = true
		}
		lo = max(lo, fm)
		return pencilBox{a, b, ub}
	}
	initial := 64
	if nd > 1 {
		initial = max(2, int(math.Pow(1024, 1/float64(nd))))
	}
	step := 2 * sB / float64(initial)
	cells := 1
	for range nd {
		cells *= initial
	}
	splits := 4096 * nd
	h := make(pencilBoxes, 0, cells+splits+4)
	for c := range cells {
		a, b := make([]float64, nd), make([]float64, nd)
		for i, ci := 0, c; i < nd; i, ci = i+1, ci/initial {
			a[i] = -sB + float64(ci%initial)*step
			b[i] = a[i] + step
		}
		h = append(h, box(a, b))
	}
	heap.Init(&h)
	for range splits {
		if bad {
			return 0, 0, false
		}
		if h[0].ub-lo <= 1e-9*(n0+math.Abs(lo)) {
			break
		}
		bx := heap.Pop(&h).(pencilBox)
		w := 0
		for i := range nd {
			if bx.b[i]-bx.a[i] > bx.b[w]-bx.a[w] {
				w = i
			}
		}
		mid := (bx.a[w] + bx.b[w]) / 2
		b1, a2 := slices.Clone(bx.b), slices.Clone(bx.a)
		b1[w], a2[w] = mid, mid
		heap.Push(&h, box(bx.a, b1))
		heap.Push(&h, box(a2, bx.b))
	}
	if bad {
		return 0, 0, false
	}
	return lo, max(lo, h[0].ub), true
}
