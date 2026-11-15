package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// exactPointResponse is the memoized exact response of a model at one
// frequency point (exactResponseAt).
type exactPointResponse struct {
	g        []complex128
	overflow bool
	ok       bool
}

// exactResponseAt is the stored model's response G(z̃) = D − q·C(pI + qA)⁻¹B
// at the exact point z̃ = −p/q of pt (p, q exact doubles; for a plain point
// q = −1, D + C(pI − A)⁻¹B), solved exactly: pI + qA is Gaussian-dyadic, so
// with A, B, C scaled to integers by exactScaledModel and p, q by a common
// power of two, its real 2n×2n embedding [[Re, −Im], [Im, Re]] with right
// side [B; 0] is solved fraction-free by bareissSolve, −q·Cx formed in
// integers and each component rounded to nearest by dyadicQuoFloat64, as
// exactDCGainAtNonsingular does at DC. An entry is ±Inf only where its exact
// value lies beyond float64's range, which overflow reports. ok is false
// where pI + qA is exactly singular. ps memoizes it per point (the model
// is ps's).
func exactResponseAt(ps *poleSchurData, pt frequencyPoint) (g []complex128, overflow, ok bool) {
	ps.mu.Lock()
	if r, hit := ps.exactAt[pt]; hit {
		ps.mu.Unlock()
		return r.g, r.overflow, r.ok
	}
	ps.mu.Unlock()
	g, overflow, ok = solveExactAt(ps, pt)
	ps.mu.Lock()
	if ps.exactAt == nil {
		ps.exactAt = make(map[frequencyPoint]exactPointResponse)
	}
	ps.exactAt[pt] = exactPointResponse{g: g, overflow: overflow, ok: ok}
	ps.mu.Unlock()
	return g, overflow, ok
}

func solveExactAt(ps *poleSchurData, pt frequencyPoint) (g []complex128, overflow, ok bool) {
	n, m, p := ps.n, ps.m, ps.p
	A := mat.NewDense(n, n, ps.a)
	B := mat.NewDense(n, m, ps.b)
	C := mat.NewDense(p, n, ps.c)
	pHat, bHat, cHat, shift := exactScaledModel(nil, A, B, C, 0)
	ea := math.MaxInt
	for _, v := range ps.a {
		if v != 0 {
			_, e := math.Frexp(v)
			ea = min(ea, e)
		}
	}
	if ea == math.MaxInt {
		ea = 0
	}
	k := ea - 53
	comps := [4]float64{real(pt.p), imag(pt.p), real(pt.q), imag(pt.q)}
	ep := math.MaxInt
	for _, v := range comps {
		if v != 0 {
			_, e := math.Frexp(v)
			ep = min(ep, e)
		}
	}
	var pq [4]*big.Int
	for i, v := range comps {
		x := new(big.Int)
		if v != 0 {
			f, e := math.Frexp(v)
			x.SetInt64(int64(math.Ldexp(f, 53)))
			x.Lsh(x, uint(e-ep))
		}
		pq[i] = x
	}
	pr, pi, qr, qi := pq[0], pq[1], pq[2], pq[3]
	if k < 0 {
		pr.Lsh(pr, uint(-k))
		pi.Lsh(pi, uint(-k))
	}
	n2, w := 2*n, 2*n+m
	a := make([]*big.Int, n2*w)
	for i := range a {
		a[i] = new(big.Int)
	}
	t := new(big.Int)
	for i := range n {
		for j := range n {
			ah := new(big.Int).Neg(pHat[i*n+j])
			if k > 0 {
				ah.Lsh(ah, uint(k))
			}
			re, im := a[i*w+j], a[(n+i)*w+j]
			re.Mul(qr, ah)
			im.Mul(qi, ah)
			if i == j {
				re.Add(re, pr)
				im.Add(im, pi)
			}
			a[(n+i)*w+n+j].Set(re)
			a[i*w+n+j].Neg(im)
		}
		for j := range m {
			a[i*w+n2+j].Set(bHat[i*m+j])
		}
	}
	y := make([]*big.Int, n2*m)
	for i := range y {
		y[i] = new(big.Int)
	}
	det, ok := bareissSolve(a, y, n2, m, t)
	if !ok {
		return nil, false, false
	}
	exp := shift + max(k, 0)
	g = make([]complex128, p*m)
	cyr, cyi, vr, vi := new(big.Int), new(big.Int), new(big.Int), new(big.Int)
	for i := range p {
		for j := range m {
			cyr.SetInt64(0)
			cyi.SetInt64(0)
			for l, c := range cHat[i*n : i*n+n] {
				if c.Sign() != 0 {
					cyr.Add(cyr, t.Mul(c, y[l*m+j]))
					cyi.Add(cyi, t.Mul(c, y[(n+l)*m+j]))
				}
			}
			vr.Mul(qr, cyr)
			vr.Sub(vr, t.Mul(qi, cyi))
			vr.Neg(vr)
			vi.Mul(qr, cyi)
			vi.Add(vi, t.Mul(qi, cyr))
			vi.Neg(vi)
			h := complex(dyadicQuoFloat64(vr, det, exp, ps.d[i*m+j]), dyadicQuoFloat64(vi, det, exp, 0))
			overflow = overflow || cmplx.IsInf(h)
			g[i*m+j] = h
		}
	}
	return g, overflow, true
}

// inExactDiskScope reports whether s, off DC, lies within the dc rounding
// disk of the cluster of the explicit delay-free sys of order up to
// exactDetMaxOrder: there the response is exactResponseAt's (Q1 ruling (c),
// A2OX65), superseding the double-double response and its ddAtPole veto.
func inExactDiskScope(sys *System, s complex128) (*poleSchurData, bool) {
	ps, ok := cachedPoleSchur(sys)
	if !ok || ps.n > exactDetMaxOrder {
		return nil, false
	}
	dc := dcPoint(sys)
	cl, ok := ps.dcCluster(dc)
	return ps, ok && s != dc && cmplx.Abs(s-dc) <= cl.disk
}
