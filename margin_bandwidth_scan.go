package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

const (
	// coarsePencilRel is eps times the largest entry of the gain pencil of
	// |L| = γ over that of its state block above which Bandwidth scans rather
	// than trusting the pencil. QZ's backward error scales with the largest
	// entry, (D/γ)² or ‖B‖‖C‖/γ, so a small γ next to a large D moves the
	// boundary roots off the crossings by more than polish reaches: beside a
	// DC zero cancelled by D the crossings were lost from about 1e-6 (CNGTCF).
	coarsePencilRel = 1e-8
	// dcCancelRel is the fraction of the size of the terms of the DC gain
	// below which dipCheckedBandwidth scans past DC.
	dcCancelRel = 1e-2
	// gainScanPerDecade is the density of gainDropScan, whose local minima
	// are refined by golden-section search.
	gainScanPerDecade = 25
)

// coarsePencilBandwidth is the bandwidth search where the gain pencil of
// |L| = γ is too coarse to place its roots; ok is false where it does not
// apply and bandwidth's pencil search stands. γ sits in the coefficient noise
// of the pencil when the DC gain cancels most of D, as of D + C(sI − A)⁻¹B
// with a DC zero: the pencil's roots fall off the crossings past polish's
// brackets, and a dip below γ past DC is two crossings with the same sign at
// either end, which endCrossings cannot see. The search then reads the DC
// gain in double-double and scans the response past DC (gainDropScan); a
// pencil crossing below the scan's, verified in double-double, takes its
// place. With certified set the threshold is the
// double-double DC response, which DCGain certified to its own rounding, and
// an exact zero there is a zero DC gain, +Inf as in bandwidth. A DC margin
// within the double-double noise returns ErrAccuracyUnverified.
func (r *rationalLoop) coarsePencilBandwidth(dc, dbDrop float64, certified bool) (bw float64, err error, ok bool) {
	drop := math.Pow(10, dbDrop/20)
	gamma := math.Abs(dc) * drop
	if r.eval.sys.internalDelayCount() > 0 || !r.gainPencilCoarse(gamma) {
		return 0, nil, false
	}
	h0, solve, ok := r.ddDC()
	if !ok {
		return 0, nil, false
	}
	if certified {
		if h0 == 0 {
			return math.Inf(1), nil, true
		}
		gamma = h0 * drop
	}
	if !(h0-gamma > 4*solve) {
		return 0, fmt.Errorf("DC response %g within %g of the bandwidth level %g: %w", h0, solve, gamma, ErrAccuracyUnverified), true
	}
	res, ok := r.dcResolvent()
	if !ok {
		return 0, nil, false
	}
	bw = r.gainDropScan(gamma, h0, r.dcPinned(res, (h0-gamma)/4), math.Inf(1))
	r.eval.exact = true
	ws, err := r.gainCrossings(gamma)
	if err != nil {
		return 0, err, true
	}
	if r.eval.err != nil {
		return 0, r.eval.err, true
	}
	if len(ws) > 0 {
		bw = min(bw, ws[0])
	}
	return bw, nil, true
}

// dipCheckedBandwidth is bw, the first crossing gainCrossings verified or
// +Inf, unless gainDropScan finds |L| below gamma before it. It scans where
// the pencil may have lost a dip below γ past DC: polish rejected a pencil
// root, or the DC gain cancels the realization's terms, D + C·M·B with M as
// in dcResolvent, to below dcCancelRel of their size. Beside a k-fold zero
// near DC cancelled by D the 2k roots of |L|² = γ² cluster and fall past
// polish's brackets or vanish, while the response there is resolved. Where
// the double-double DC margin to gamma or dcResolvent does not resolve, bw
// stands.
func (r *rationalLoop) dipCheckedBandwidth(gamma, bw float64) float64 {
	if r.eval.sys.internalDelayCount() > 0 {
		return bw
	}
	res, ok := r.dcResolvent()
	if !ok || !r.gainRejected && !(res.cancel*dcCancelRel > math.Abs(res.dc)) {
		return bw
	}
	h0, solve, ok := r.ddDC()
	if !ok || !(h0-gamma > 4*solve) {
		return bw
	}
	return min(bw, r.gainDropScan(gamma, h0, r.dcPinned(res, (h0-gamma)/4), bw))
}

// ddDC is |L| at DC in double-double with its solve noise; ok is false at a
// pole there.
func (r *rationalLoop) ddDC() (h0, solve float64, ok bool) {
	r.eval.ddSolves++
	h, solve, _, ok := ddSISOResponseNoise(r.eval.sys, r.eval.td.frequencyPoint(0))
	if !ok || cmplx.IsInf(h) || cmplx.IsNaN(h) {
		return 0, 0, false
	}
	return cmplx.Abs(h), solve, true
}

// gainPencilCoarse reports whether eps times the largest entry gainCandidates
// puts in the gain pencil of |L| = gamma exceeds coarsePencilRel of the
// largest entry of A and E.
func (r *rationalLoop) gainPencilCoarse(gamma float64) bool {
	var ref float64
	for i := range r.a {
		ref = max(ref, math.Abs(r.a[i]), math.Abs(r.e[i]))
	}
	bs, cs := ioScale(r.b, r.c, gamma)
	d := r.d / gamma
	top := math.Abs(d*d - 1)
	var cMax float64
	for i := range r.n {
		top = max(top, math.Abs(r.b[i]*bs), math.Abs(r.c[i]*cs*d))
		cMax = max(cMax, math.Abs(r.c[i]*cs))
	}
	top = max(top, cMax*cMax)
	return eps()*top > coarsePencilRel*ref
}

// dcResolvent holds the DC resolvent M = (E − A)⁻¹ at z = 1, or (−A)⁻¹
// at s = 0, of the loop: ‖MB‖, ‖ME‖, ‖CME‖, the DC gain D + CMB and the size
// |D| + Σ|cᵢ(MB)ᵢ| of the terms it cancels.
type dcResolvent struct {
	mb, me, cme, c, dc, cancel float64
}

// dcResolvent is false where E − A, or A, is singular to working precision.
func (r *rationalLoop) dcResolvent() (dcResolvent, bool) {
	sys := r.eval.sys
	n, _, _ := sys.Dims()
	d := sys.D.At(0, 0)
	if n == 0 {
		return dcResolvent{dc: d, cancel: math.Abs(d)}, true
	}
	g := mat.NewDense(n, n, nil)
	rhs := mat.NewDense(n, n+1, nil)
	gr, rr := g.RawMatrix(), rhs.RawMatrix()
	ar, br := sys.A.RawMatrix(), sys.B.RawMatrix()
	for i := range n {
		for j := range n {
			var e float64
			if sys.E != nil {
				e = sys.E.At(i, j)
			} else if i == j {
				e = 1
			}
			v := -ar.Data[i*ar.Stride+j]
			if r.discrete() {
				v += e
			}
			gr.Data[i*gr.Stride+j] = v
			rr.Data[i*rr.Stride+1+j] = e
		}
		rr.Data[i*rr.Stride] = br.Data[i*br.Stride]
	}
	var lu mat.LU
	lu.Factorize(g)
	if !(lu.Cond() < 1/(float64(n)*eps())) {
		return dcResolvent{}, false
	}
	x := mat.NewDense(n, n+1, nil)
	if err := lu.SolveTo(x, false, rhs); err != nil {
		return dcResolvent{}, false
	}
	xr, cr := x.RawMatrix(), sys.C.RawMatrix()
	res := dcResolvent{dc: d, cancel: math.Abs(d)}
	for i := range n {
		v := xr.Data[i*xr.Stride]
		res.mb += v * v
		res.dc += cr.Data[i] * v
		res.cancel += math.Abs(cr.Data[i] * v)
		for j := range n {
			v := xr.Data[i*xr.Stride+1+j]
			res.me += v * v
		}
	}
	for j := range n {
		var s float64
		for i := range n {
			s += cr.Data[i] * xr.Data[i*xr.Stride+1+j]
		}
		res.cme += s * s
		res.c += cr.Data[j] * cr.Data[j]
	}
	res.mb, res.me, res.cme, res.c = math.Sqrt(res.mb), math.Sqrt(res.me), math.Sqrt(res.cme), math.Sqrt(res.c)
	if math.IsNaN(res.mb+res.me+res.cme) || math.IsInf(res.mb+res.me+res.cme, 0) {
		return dcResolvent{}, false
	}
	return res, true
}

// dcPinned returns an ω > 0 below which |L(jω) − L(0)| ≤ tol. With μ = s,
// or z − 1, |μ| ≤ ωT, the resolvent is (I + μME)⁻¹M = Σ (−μME)ʲM, so
// L − L(0) = Σⱼ≥₁ (−μ)ʲmⱼ with the moments mⱼ = C(ME)ʲMB. A DC zero of
// order k cancelled by D leaves m₁ … m_{k−1} at its rounding, so the first
// n+1 moments, in double-double, with ‖C‖‖ME‖ʲ‖MB‖ for the tail past them,
// place ω far above the first-order bound |μ|‖CME‖‖MB‖/(1 − |μ|‖ME‖), which
// stands where the moments do not resolve. Half of tol covers the rounding
// of M and of the moments.
func (r *rationalLoop) dcPinned(res dcResolvent, tol float64) float64 {
	half := tol / 2
	mu := half / (res.cme*res.mb + half*res.me)
	if m, ok := r.dcMoments(); ok {
		k := len(m)
		bound := func(x float64) float64 {
			if !(x*res.me < 1) {
				return math.Inf(1)
			}
			var s float64
			p := 1.0
			for _, v := range m {
				p *= x
				s += v * p
			}
			return s + res.c*math.Pow(res.me, float64(k+1))*res.mb*p*x/(1-x*res.me)
		}
		lo, hi := mu, 1/(2*res.me)
		if math.IsInf(hi, 1) {
			hi = r.highFrequency()
		}
		if bound(hi) <= half {
			lo = hi
		} else if bound(lo) <= half {
			for range 64 {
				x := math.Sqrt(lo * hi)
				if x <= lo || x >= hi {
					break
				}
				if bound(x) <= half {
					lo = x
				} else {
					hi = x
				}
			}
		}
		mu = max(mu, lo)
	}
	if r.discrete() {
		mu /= r.dt
	}
	return mu
}

// dcMoments returns |mⱼ| + its rounding for the moments mⱼ = C(ME)ʲMB,
// j = 1 … n+1, of dcPinned, from a double-double LU factorization of E − A,
// or −A; ok is false at a pivot within its rounding.
func (r *rationalLoop) dcMoments() ([]float64, bool) {
	sys := r.eval.sys
	n, _, _ := sys.Dims()
	if n == 0 {
		return nil, false
	}
	at := func(m *mat.Dense, i, j int) float64 {
		if m == nil {
			if i == j {
				return 1
			}
			return 0
		}
		return m.At(i, j)
	}
	g := make([]dd, n*n)
	var scale float64
	for i := range n {
		for j := range n {
			v := dd{-sys.A.At(i, j), 0}
			if r.discrete() {
				v = v.add(dd{at(sys.E, i, j), 0})
			}
			g[i*n+j] = v
			scale = max(scale, math.Abs(v.hi))
		}
	}
	piv := make([]int, n)
	for k := range n {
		p := k
		for i := k + 1; i < n; i++ {
			if math.Abs(g[i*n+k].hi) > math.Abs(g[p*n+k].hi) {
				p = i
			}
		}
		piv[k] = p
		if !(math.Abs(g[p*n+k].hi) > float64(n)*scale*0x1p-90) {
			return nil, false
		}
		for j := range n {
			g[k*n+j], g[p*n+j] = g[p*n+j], g[k*n+j]
		}
		for i := k + 1; i < n; i++ {
			f := g[i*n+k].div(g[k*n+k])
			g[i*n+k] = f
			for j := k + 1; j < n; j++ {
				g[i*n+j] = g[i*n+j].add(f.mul(g[k*n+j]).neg())
			}
		}
	}
	solve := func(x []dd) {
		for k := range n {
			x[k], x[piv[k]] = x[piv[k]], x[k]
		}
		for k := range n {
			for i := k + 1; i < n; i++ {
				x[i] = x[i].add(g[i*n+k].mul(x[k]).neg())
			}
		}
		for i := n - 1; i >= 0; i-- {
			s := x[i]
			for j := i + 1; j < n; j++ {
				s = s.add(g[i*n+j].mul(x[j]).neg())
			}
			x[i] = s.div(g[i*n+i])
		}
	}
	x := make([]dd, n)
	for i := range n {
		x[i] = dd{sys.B.At(i, 0), 0}
	}
	solve(x)
	m := make([]float64, n+1)
	ex := make([]dd, n)
	for j := range m {
		for i := range n {
			var s dd
			for l := range n {
				s = s.add(dd{at(sys.E, i, l), 0}.mul(x[l]))
			}
			ex[i] = s
		}
		copy(x, ex)
		solve(x)
		var c dd
		var cx float64
		for i := range n {
			ci := sys.C.At(0, i)
			c = c.add(dd{ci, 0}.mul(x[i]))
			cx += math.Abs(ci * x[i].hi)
		}
		m[j] = math.Abs(c.hi) + 0x1p-90*cx
		if math.IsNaN(m[j]) || math.IsInf(m[j], 0) {
			return nil, false
		}
	}
	return m, true
}

// gainDropScan returns the first ω above wLo where |L| drops below gamma on a
// log scan of gainScanPerDecade steps per decade, in double-double, or +Inf
// once |L| rises past twice h0, its DC value, or reaches just below upTo,
// Nyquist or highFrequency. A step counts only once the response's distance
// from gamma exceeds the double-double noise. A local minimum of |L| on the
// grid at least halfway from h0 down to gamma is refined by golden-section
// search between its neighbours, for a shallow dip that crosses gamma within
// one step. The first ω certified
// below gamma is bisected from the last certified above it, to 1e-12
// relative or to a point within the noise of gamma. A dip that is not
// unimodal within two steps, or one past the rise, is outside the scan.
func (r *rationalLoop) gainDropScan(gamma, h0, wLo, upTo float64) float64 {
	sys := r.eval.sys
	top := r.highFrequency()
	if r.discrete() {
		top = r.nyquist()
	}
	top = min(top, upTo*(1-1e-6))
	eval := func(w float64) (g float64, st int) {
		r.eval.ddSolves++
		h, solve, _, ok := ddSISOSolve(sys, r.eval.td.frequencyPoint(w), false)
		if !ok {
			return math.Inf(1), 1
		}
		switch g = cmplx.Abs(h); {
		case g-gamma > solve:
			st = 1
		case gamma-g > solve:
			st = -1
		}
		return g, st
	}
	bisect := func(lo, hi float64) float64 {
		for hi > lo*(1+1e-12) {
			m := math.Sqrt(lo * hi)
			if m <= lo || m >= hi {
				break
			}
			switch _, st := eval(m); st {
			case 1:
				lo = m
			case -1:
				hi = m
			default:
				return m
			}
		}
		return math.Sqrt(lo * hi)
	}
	ratio := math.Pow(10, 1.0/gainScanPerDecade)
	w0, g0 := wLo, h0
	w1, g1 := wLo, h0
	above := wLo
	for w := wLo; w < top; {
		w = min(w*ratio, top)
		g, st := eval(w)
		if st < 0 {
			return bisect(above, w)
		}
		if st > 0 && g1 < g0 && g1 <= g && g1-gamma < (h0-gamma)/2 {
			a, b := math.Log(w0), math.Log(w)
			const phi = 0.6180339887498949
			x, y := b-phi*(b-a), a+phi*(b-a)
			gx, sx := eval(math.Exp(x))
			gy, sy := eval(math.Exp(y))
			for range 60 {
				switch {
				case sx < 0:
					return bisect(w0, math.Exp(x))
				case sy < 0:
					return bisect(w0, math.Exp(y))
				case b-a < 1e-9:
				case gx < gy:
					b, y, gy, sy = y, x, gx, sx
					x = b - phi*(b-a)
					gx, sx = eval(math.Exp(x))
					continue
				default:
					a, x, gx, sx = x, y, gy, sy
					y = a + phi*(b-a)
					gy, sy = eval(math.Exp(y))
					continue
				}
				break
			}
		}
		if st > 0 {
			above = w
			if g-2*h0 > 0 {
				break
			}
		}
		w0, g0, w1, g1 = w1, g1, w, g
	}
	return math.Inf(1)
}
