package controlsys

import (
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
