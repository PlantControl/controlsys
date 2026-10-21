package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
)

// poleResponse is the response entry at one of its poles. Like MATLAB's 1/0
// it is real +Inf; multiplying by a unit-modulus delay factor keeps
// cmplx.IsInf true.
var poleResponse = complex(math.Inf(1), 0)

const (
	poleStepFraction = 1e-5
	poleResidueTol   = 1e-6
	poleResidueFloor = 1e-10
	poleRelativeTol  = 1e-3
	poleClusterTol   = 1e-6
	// poleClusterSpread allows for the conditioning of a multiple pole's
	// eigenvectors in its eps^{1/k} spread.
	poleClusterSpread = 10
)

// evalWithPoleLimit sets dst to eval(pt), or where eval fails to the
// response there: at a numerical pole of the realization its pole limit
// (see poleLimitInto), and off every pole the double-double response of the
// delay-free realization (ddResponseInto). A computed pole p is near pt
// when it lies within poleRelativeTol of s relative to max(|s|, |p|) and
// within the pole limit's step poleStepFraction·rho of s, where its samples
// see p at s; farther out, as beside a lightly damped pair, they straddle p
// and misread the finite response as a pole. At DC (s = 0, or z = 1 for a
// discrete model) a near pole is a pole at pt, and for s = 0 so is one
// within poleClusterTol of the spectral scale, the radius poleSpacing takes
// as s itself: a computed integrator is of order eps·‖A‖ and a k-fold one
// spreads by eps^{1/k}. Elsewhere a computed pole locates the realization's
// own pole only to about eps over its separation from the next pole, which
// beside a lightly damped pair exceeds the step, so the double-double
// response decides: pt is a pole where that is singular too or its
// simple-pole fit (ddAtPole) puts the pole within working precision of s.
// Next to a pole cluster at the origin the double solve fails at |s| of the
// cluster's size, where no pole is near s and the response is large but
// finite. The pole limit's samples fail there too;
// each failed sample is solved in double-double, and where that is singular
// as well the step is taken past the cluster at s (poleSpacing's wide).
// An internal-delay model always takes the pole limit; a failed sample is
// the LFT of its delay-free plant solved in double-double (ddLFTSampler).
func evalWithPoleLimit(eval func(frequencyPoint, []complex128) error, sys *System, pt frequencyPoint, dst []complex128) error {
	return evalWithPoleLimitGated(eval, sys, pt, dst, ddGate{})
}

// ddGate restricts the double-double fallback of evalWithPoleLimitGated off
// a pole. With limit set the pole limit applies there too, for a caller
// tuned to it: the rational margin search takes near-roots of an
// asymptotically real L next to an origin cluster for crossings once that
// region evaluates. Otherwise, where allow(w) reports false, eval's error is
// returned; a nil allow allows the fallback. With allow or snap set a near
// computed pole is a pole at pt as at DC, without the double-double check;
// the margin search keeps the pole limit's +Inf there. allow and limit also
// keep the pole limit's samples in double, so its errors stand for the
// caller; for an internal-delay model so does snap, and only the zero gate
// solves failed samples in double-double.
type ddGate struct {
	limit bool
	snap  bool
	allow func(float64) bool
	w     float64
}

// evalWithPoleLimitGated is evalWithPoleLimit with the fallback off a pole
// restricted by gate.
func evalWithPoleLimitGated(eval func(frequencyPoint, []complex128) error, sys *System, pt frequencyPoint, dst []complex128, gate ddGate) error {
	err := eval(pt, dst)
	if err == nil {
		return nil
	}
	s := pt.value()
	sp, perr := poleSpacing(sys, s)
	if perr != nil {
		return perr
	}
	rho, wide := sp.rho, sp.wide
	delayFree := sys.internalDelayCount() == 0
	dc := s == 0 || !sys.IsContinuous() && s == 1
	if !gate.limit && delayFree && (!sp.near || gate.allow == nil && !gate.snap && !dc) {
		if gate.allow != nil && !gate.allow(gate.w) {
			return err
		}
		if ddResponseInto(sys, pt, dst) && (!sp.near || !ddAtPole(sys, s, sp, dst)) {
			return nil
		}
	}
	if !delayFree && !gate.limit && !gate.snap && gate.allow == nil {
		dd := newDDLFTSampler(sys)
		sample := func(pt frequencyPoint, dst []complex128) error {
			if err := eval(pt, dst); err == nil || !dd.responseInto(pt, dst) {
				return err
			}
			return nil
		}
		return poleLimitInto(sample, s, rho, dst, err)
	}
	if !delayFree || gate.limit || gate.allow != nil {
		return poleLimitInto(eval, s, rho, dst, err)
	}
	sample := func(pt frequencyPoint, dst []complex128) error {
		if err := eval(pt, dst); err == nil || !ddResponseInto(sys, pt, dst) {
			return err
		}
		return nil
	}
	if lerr := poleLimitInto(sample, s, rho, dst, err); lerr == nil || wide <= rho {
		return lerr
	}
	return poleLimitInto(sample, s, wide, dst, err)
}

// poleLimitInto sets dst to the response at s, a numerical pole of the
// realization at which eval fails. Each entry is the quadratic extrapolation
// to s of eval at s+kδ, k = 1, 2, 3, unless kδ·G(s+kδ) extrapolates to a
// nonzero residue, in which case the entry is poleResponse. A residue below
// poleResidueFloor times the largest one is rounding from the solve's large
// state and marks a channel the pole does not reach. δ is a fraction of rho,
// the distance from s to the nearest other singularity, so finite entries
// carry an extrapolation error of order (δ/rho)³ plus rounding of order
// eps·(rho/δ)^q from a pole of order q; for q > 1 they are instead quartic
// extrapolations from five points with δ = rho·eps^{1/(q+5)}, which balances
// the two. When eval also fails off s the model is singular everywhere and
// singularErr is returned.
func poleLimitInto(eval func(frequencyPoint, []complex128) error, s complex128, rho float64, dst []complex128, singularErr error) error {
	k := len(dst)
	g := make([]complex128, 5*k)
	sample := func(points int, delta float64) error {
		for i := range points {
			if err := eval(pointAt(s+complex(float64(i+1)*delta, 0)), g[i*k:(i+1)*k]); err != nil {
				return singularErr
			}
		}
		return nil
	}
	delta := poleStepFraction * rho
	if err := sample(3, delta); err != nil {
		return err
	}
	d := complex(delta, 0)
	residue := make([]float64, k)
	maxResidue := 0.0
	for j := range dst {
		residue[j] = cmplx.Abs(3 * d * (g[j] - 2*g[k+j] + g[2*k+j]))
		maxResidue = max(maxResidue, residue[j])
	}
	pole := make([]bool, k)
	order := 1
	for j := range dst {
		scale := max(cmplx.Abs(d*g[j]), cmplx.Abs(2*d*g[k+j]), cmplx.Abs(3*d*g[2*k+j]))
		if residue[j] > poleResidueTol*scale && residue[j] > poleResidueFloor*maxResidue {
			pole[j] = true
			order = max(order, int(math.Round(math.Log2(cmplx.Abs(g[j])/cmplx.Abs(g[k+j])))))
		}
	}
	weights := []complex128{3, -3, 1}
	if order > 1 {
		weights = []complex128{5, -10, 10, -5, 1}
		if err := sample(len(weights), rho*math.Pow(eps(), 1/float64(order+5))); err != nil {
			return err
		}
	}
	for j := range dst {
		if pole[j] {
			dst[j] = poleResponse
			continue
		}
		var v complex128
		for i, w := range weights {
			v += w * g[i*k+j]
		}
		dst[j] = v
	}
	return nil
}

// ddLFTSampler evaluates an internal-delay model where its double solve
// fails off a pole, as next to a pole cluster at the origin far below ‖A‖:
// the delay-free plant H in double-double (ddResponseInto), closed by Δ in
// double. It builds H on first use.
type ddLFTSampler struct {
	sys  *System
	h    *System
	hv   []complex128
	work []complex128
}

func newDDLFTSampler(sys *System) *ddLFTSampler { return &ddLFTSampler{sys: sys} }

// responseInto sets dst to the LFT of H at pt. It reports false where H is
// singular in double-double, I − H22·Δ is singular, or an entry cancels
// terms over lftCancelLimit times its size: there the loop moves a pole of
// H, and H's rounding of order 1/d leaves the LFT no correct digits.
func (d *ddLFTSampler) responseInto(pt frequencyPoint, dst []complex128) bool {
	sys := d.sys
	n, m, p := sys.Dims()
	lft := sys.LFT
	N := len(lft.Tau)
	mN, pN := m+N, p+N
	if d.h == nil {
		d.h = &System{
			A:  sys.A,
			B:  newDense(n, mN),
			C:  newDense(pN, n),
			D:  newDense(pN, mN),
			E:  sys.E,
			Dt: sys.Dt,
		}
		if n > 0 {
			setBlock(d.h.B, 0, 0, sys.B)
			setBlock(d.h.B, 0, m, lft.B2)
			setBlock(d.h.C, 0, 0, sys.C)
			setBlock(d.h.C, p, 0, lft.C2)
		}
		setBlock(d.h.D, 0, 0, sys.D)
		setBlock(d.h.D, 0, m, lft.D12)
		setBlock(d.h.D, p, 0, lft.D21)
		setBlock(d.h.D, p, m, lft.D22)
		d.hv = make([]complex128, pN*mN)
		d.work = make([]complex128, N+N*N+N*m)
	}
	h := d.hv
	if !ddResponseInto(d.h, pt, h) {
		return false
	}
	delta, lhs, x := d.work[:N], d.work[N:N+N*N], d.work[N+N*N:]
	s := pt.value()
	for j, tau := range lft.Tau {
		if sys.IsContinuous() {
			delta[j] = cmplx.Exp(-s * complex(tau, 0))
		} else {
			delta[j] = pt.delayFactor(tau)
		}
	}
	for i := range N {
		row := h[(p+i)*mN:]
		for j := range N {
			lhs[i*N+j] = -row[m+j] * delta[j]
		}
		lhs[i*N+i] += 1
		copy(x[i*m:(i+1)*m], row[:m])
	}
	if cSolveInPlace(lhs, x, N, m) != nil {
		return false
	}
	for i := range p {
		row := h[i*mN:]
		for j := range m {
			v := row[j]
			terms := cabs1(v)
			for k := range N {
				t := row[m+k] * delta[k] * x[k*m+j]
				v += t
				terms += cabs1(t)
			}
			if !finiteComplex(v) || terms > lftCancelLimit*cabs1(v) {
				return false
			}
			dst[i*m+j] = v
		}
	}
	return true
}

// ddAtPole reports whether g, the double-double response at s next to a
// computed pole, is that of a pole within working precision of s, at
// poleClusterSpread·eps of the spectral scale. A simple pole p with residue
// r has 1/G(z) ≈ (z−p)/r, so G at s and at s+h, h ≪ the distance to the
// next pole, give r = h/(1/G(s+h) − 1/G(s)) and s−p = r/G(s), in the
// largest entry. A multiple pole fits nearer than it is. A dd failure at
// s+h counts as a pole.
func ddAtPole(sys *System, s complex128, sp poleGeometry, g []complex128) bool {
	tol := poleClusterSpread * eps() * sp.scale
	h := max(2*sp.gap, tol)
	gh := make([]complex128, len(g))
	if !ddResponseInto(sys, pointAt(s+complex(h, 0)), gh) {
		return true
	}
	j := 0
	for i, v := range g {
		if cmplx.Abs(v) > cmplx.Abs(g[j]) {
			j = i
		}
	}
	if g[j] == 0 {
		return false
	}
	r := complex(h, 0) / (1/gh[j] - 1/g[j])
	return cmplx.Abs(r/g[j]) <= tol
}

// poleGeometry locates s among the poles of the delay-free plant (A, E).
// rho is the distance from s to the nearest pole that is distinct from s,
// bounded by the overall spectral scale and, for continuous internal
// delays, by 1/τ, over which e^{-sτ} varies. wide is rho with the k nearest
// poles also taken as s when all lie within poleClusterSpread·eps^{1/k} of
// the scale, the spread of a computed k-fold pole. gap is the distance to
// the nearest finite pole, scale the largest of |s| and the finite pole
// magnitudes, and near reports a computed pole near s as
// evalWithPoleLimit defines it.
type poleGeometry struct {
	rho, wide, gap, scale float64
	near                  bool
}

func poleSpacing(sys *System, s complex128) (poleGeometry, error) {
	n, _, _ := sys.Dims()
	var poles []complex128
	if n > 0 {
		var err error
		if poles, err = newDescriptorPolicy(sys).poles("pole limit", sys.A, n); err != nil {
			return poleGeometry{}, err
		}
	}
	scale := cmplx.Abs(s)
	for _, p := range poles {
		if !cmplx.IsInf(p) && !cmplx.IsNaN(p) {
			scale = max(scale, cmplx.Abs(p))
		}
	}
	if scale == 0 {
		scale = 1
	}
	rho := scale
	near := false
	dist := make([]float64, 0, len(poles))
	relative := false
	for _, p := range poles {
		d := cmplx.Abs(p - s)
		if d > poleClusterTol*scale && d < rho {
			rho = d
		}
		if !cmplx.IsInf(p) {
			relative = relative || d <= poleRelativeTol*max(cmplx.Abs(s), cmplx.Abs(p))
			near = near || s == 0 && d <= poleClusterTol*scale
			dist = append(dist, d)
		}
	}
	slices.Sort(dist)
	cluster := 0
	for k, d := range dist {
		if d <= poleClusterSpread*math.Pow(eps(), 1/float64(k+1))*scale {
			cluster = k + 1
		}
	}
	wide := scale
	if cluster < len(dist) {
		wide = min(scale, max(rho, dist[cluster]))
	}
	if sys.IsContinuous() && sys.LFT != nil {
		for _, tau := range sys.LFT.Tau {
			if tau > 0 {
				rho = min(rho, 1/tau)
				wide = min(wide, 1/tau)
			}
		}
	}
	near = near || relative && dist[0] <= poleStepFraction*rho
	gap := math.Inf(1)
	if len(dist) > 0 {
		gap = dist[0]
	}
	return poleGeometry{rho: rho, wide: wide, gap: gap, scale: scale, near: near}, nil
}

// ddResponseInto sets dst to the p×m response C(sE−A)⁻¹B + D of the
// delay-free realization at pt as ddSISOResponse forms it, with the same
// elimination for each input column. It reports false, leaving dst
// unchanged, at a pivot below 2⁻¹⁰⁰ of the magnitudes it was formed from,
// where it is cancellation noise. A pivot small against the whole pencil but
// not against its own terms is accurate, as is a companion form's last pivot
// next to an exact origin pole.
func ddResponseInto(sys *System, pt frequencyPoint, dst []complex128) bool {
	n, m, p := sys.Dims()
	if n == 0 {
		for i := range p {
			for j := range m {
				dst[i*m+j] = complex(sys.D.At(i, j), 0)
			}
		}
		return true
	}
	ar, br, cr := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	pr, pi, qr, qi := real(pt.p), imag(pt.p), real(pt.q), imag(pt.q)
	a := make([]ddComplex, n*n)
	x := make([]ddComplex, n*m)
	mag := make([]float64, n*n)
	for i := range n {
		for j := range n {
			av := ar.Data[i*ar.Stride+j]
			var e float64
			if sys.E != nil {
				e = sys.E.At(i, j)
			} else if i == j {
				e = 1
			}
			a[i*n+j] = ddComplex{ddProd(pr, e).add(ddProd(qr, av)), ddProd(pi, e).add(ddProd(qi, av))}
			mag[i*n+j] = math.Abs(pr*e) + math.Abs(qr*av) + math.Abs(pi*e) + math.Abs(qi*av)
		}
		for j := range m {
			x[i*m+j] = ddComplex{re: dd{br.Data[i*br.Stride+j], 0}}
		}
	}
	tol := float64(n) * 0x1p-100
	for k := range n {
		piv, best := k, a[k*n+k].abs1()
		for i := k + 1; i < n; i++ {
			if v := a[i*n+k].abs1(); v > best {
				piv, best = i, v
			}
		}
		if !(best > tol*mag[piv*n+k]) {
			return false
		}
		if piv != k {
			for j := range n {
				a[k*n+j], a[piv*n+j] = a[piv*n+j], a[k*n+j]
				mag[k*n+j], mag[piv*n+j] = mag[piv*n+j], mag[k*n+j]
			}
			for j := range m {
				x[k*m+j], x[piv*m+j] = x[piv*m+j], x[k*m+j]
			}
		}
		for i := k + 1; i < n; i++ {
			if a[i*n+k].abs1() == 0 {
				continue
			}
			f := a[i*n+k].div(a[k*n+k])
			fm := f.abs1()
			for j := k + 1; j < n; j++ {
				a[i*n+j] = a[i*n+j].sub(f.mul(a[k*n+j]))
				mag[i*n+j] += fm * mag[k*n+j]
			}
			for j := range m {
				x[i*m+j] = x[i*m+j].sub(f.mul(x[k*m+j]))
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		for c := range m {
			v := x[i*m+c]
			for j := i + 1; j < n; j++ {
				v = v.sub(a[i*n+j].mul(x[j*m+c]))
			}
			x[i*m+c] = v.div(a[i*n+i])
		}
	}
	q := ddComplex{dd{-qr, 0}, dd{-qi, 0}}
	for r := range p {
		for c := range m {
			var y ddComplex
			for i := n - 1; i >= 0; i-- {
				y = y.add(ddComplex{re: dd{cr.Data[r*cr.Stride+i], 0}}.mul(x[i*m+c]))
			}
			dst[r*m+c] = y.mul(q).complex128() + complex(sys.D.At(r, c), 0)
		}
	}
	return true
}

// evalFrLFTFrozenInto sets ws.g to G(s) by closing the delay channels with
// their values at s, K = Δ(I-D22Δ)⁻¹ = (I-ΔD22)⁻¹Δ, and solving the complex
// realization (sE - A - B2·K·C2, B1 + B2·K·D21, C1 + D12·K·C2, D11 + D12·K·D21)
// once; descriptor models off the plain point solve it as
// c·(pE + qA − c·B2·K·C2)⁻¹, c = −q, with the point's exact p and q. Unlike
// the H-then-LFT form it stays regular at poles of the delay-free plant that
// the delay loop moves, such as an integrator inside it. Where a pivot of
// the solve is below 1/refineCondition of the pencil, frozenRefine refines
// it against the factored loop.
func evalFrLFTFrozenInto(ws *lftWorkspace, pt frequencyPoint, N, p, m int) error {
	br := &ws.bd.balancedRealization
	n, mN := br.n, m+N
	nm := n + m
	if ws.frozen == nil {
		ws.frozen = make([]complex128, 2*N*N+N*nm+2*n*n+2*n*m)
	}
	buf := ws.frozen
	clear(buf)
	take := func(k int) []complex128 {
		out := buf[:k:k]
		buf = buf[k:]
		return out
	}
	K, lhs, T, pencil, x := take(N*N), take(N*N), take(N*nm), take(n*n), take(n*m)
	lu, r := take(n*n), take(n*m)
	for i := range N {
		K[i*N+i] = ws.delta[i]
		for j := range N {
			lhs[i*N+j] = -ws.delta[i] * complex(br.d[(p+i)*br.dStride+m+j], 0)
		}
		lhs[i*N+i] += 1
	}
	if err := cSolveInPlace(lhs, K, N, N); err != nil {
		return err
	}
	for i := range N {
		for k := range N {
			kik := K[i*N+k]
			if kik == 0 {
				continue
			}
			row := p + k
			for j := range n {
				T[i*nm+j] += kik * complex(br.c[row*n+j], 0)
			}
			for j := range m {
				T[i*nm+n+j] += kik * complex(br.d[row*br.dStride+j], 0)
			}
		}
	}
	scaled := !pt.isPlain() && br.e != nil
	for i := range n {
		for j := range n {
			a := br.a[i*n+j]
			var v complex128
			switch {
			case scaled:
				e := br.e[i*n+j]
				v = complex(real(pt.p)*(e-a), imag(pt.p)*(e+a))
			case br.e != nil:
				v = -complex(a, 0) + pt.value()*complex(br.e[i*n+j], 0)
			case i == j && !pt.isPlain():
				v = pt.shift(a)
			default:
				v = -complex(a, 0)
				if i == j {
					v += pt.value()
				}
			}
			pencil[i*n+j] = v
		}
		for j := range m {
			x[i*m+j] = complex(br.b[i*mN+j], 0)
		}
		for k := range N {
			b2 := complex(br.b[i*mN+m+k], 0)
			if b2 == 0 {
				continue
			}
			b2c := b2
			if scaled {
				b2c *= pt.scale()
			}
			for j := range n {
				pencil[i*n+j] -= b2c * T[k*nm+j]
			}
			for j := range m {
				x[i*m+j] += b2 * T[k*nm+n+j]
			}
		}
	}
	if n > 0 {
		copy(lu, pencil)
		growth, err := cSolveGrowthInPlace(pencil, x, n, m)
		if err != nil {
			return err
		}
		if growth > refineCondition {
			ws.frozenRefine(pt, K, lu, x, r, n, p, m, N, scaled)
		}
		if scaled {
			for i := range x {
				x[i] *= pt.scale()
			}
		}
	}
	for i := range p {
		for j := range m {
			v := complex(br.d[i*br.dStride+j], 0)
			for k := range N {
				v += complex(br.d[i*br.dStride+m+k], 0) * T[k*nm+n+j]
			}
			for l := range n {
				c := complex(br.c[i*n+l], 0)
				for k := range N {
					c += complex(br.d[i*br.dStride+m+k], 0) * T[k*nm+l]
				}
				v += c * x[l*m+j]
			}
			ws.g[i*m+j] = v
		}
	}
	return nil
}

// frozenRefine refines x, evalFrLFTFrozenInto's solve at pt, against its
// closed-loop pencil with the loop B2·K·C2 kept factored, given lu, the
// pencil as formed, to factor. Forming B2·K·C2 rounds the pencil by
// ε|B2||K||C2|, which next to a closed-loop pole costs ε/d relative accuracy
// at distance d, where the rank-N loop that the LFT keeps does not.
func (ws *lftWorkspace) frozenRefine(pt frequencyPoint, K, lu, x, r []complex128, n, p, m, N int, scaled bool) {
	if ws.frozenPiv == nil {
		ws.frozenPiv = make([]int, n)
		ws.frozenLoop = make([]compensatedComplex, 2*N)
	}
	piv := ws.frozenPiv
	if !cLUFactor(lu, piv, n) {
		return
	}
	kappa, mu := complex(1, 0), complex(1, 0)
	switch {
	case scaled:
		mu = pt.scale()
	case !pt.isPlain() && ws.bd.e == nil:
		kappa, mu = pt.scale(), pt.scale()
	}
	ik := crecip(kappa)
	for step := range maxRefineSteps {
		ws.frozenResidual(pt, K, x, r, n, p, m, N, kappa, mu)
		if kappa != 1 {
			for i := range r {
				r[i] *= ik
			}
		}
		cLUSolve(lu, piv, r, n, m)
		var dMax, xMax float64
		for i, v := range r {
			dMax = max(dMax, cabs1(v))
			xMax = max(xMax, cabs1(x[i]))
		}
		if step == 0 && dMax <= refineCondition*eps()*xMax {
			return
		}
		xMax = 0
		for i, v := range r {
			x[i] += v
			xMax = max(xMax, cabs1(x[i]))
		}
		if dMax <= refineStop*xMax {
			return
		}
	}
}

// frozenResidual sets r = κ·B1 − (pE + qA)·x + B2·K·(κ·D21 + μ·C2·x) in
// compensated arithmetic, κ·r being the residual of evalFrLFTFrozenInto's
// pencil times κ.
func (ws *lftWorkspace) frozenResidual(pt frequencyPoint, K, x, r []complex128, n, p, m, N int, kappa, mu complex128) {
	br := &ws.bd.balancedRealization
	mN := m + N
	v, w := ws.frozenLoop[:N], ws.frozenLoop[N:]
	for j := range m {
		for l := range N {
			var cx compensatedComplex
			for k, c := range br.c[(p+l)*n : (p+l+1)*n] {
				cx.addProd(complex(c, 0), x[k*m+j])
			}
			v[l] = compensatedComplex{}
			v[l].addScaled(mu, cx)
			v[l].addProd(kappa, complex(br.d[(p+l)*br.dStride+j], 0))
		}
		for k := range N {
			w[k] = compensatedComplex{}
			for l := range N {
				w[k].addScaled(K[k*N+l], v[l])
			}
		}
		for i := range n {
			var ax, ex compensatedComplex
			for k, av := range br.a[i*n : (i+1)*n] {
				ax.addProd(complex(av, 0), x[k*m+j])
			}
			if br.e == nil {
				ex.re.s, ex.im.s = real(x[i*m+j]), imag(x[i*m+j])
			} else {
				for k, ev := range br.e[i*n : (i+1)*n] {
					ex.addProd(complex(ev, 0), x[k*m+j])
				}
			}
			var res compensatedComplex
			res.addProd(kappa, complex(br.b[i*mN+j], 0))
			res.addScaled(-pt.p, ex)
			res.addScaled(-pt.q, ax)
			for k := range N {
				if b2 := br.b[i*mN+m+k]; b2 != 0 {
					res.re.addScaled(b2, w[k].re)
					res.im.addScaled(b2, w[k].im)
				}
			}
			r[i*m+j] = complex(res.re.value(), res.im.value())
		}
	}
}

// compensatedComplex is a complex compensatedSum.
type compensatedComplex struct{ re, im compensatedSum }

func (z *compensatedComplex) addProd(a, b complex128) {
	z.re.addProd(real(a), real(b))
	z.re.addProd(-imag(a), imag(b))
	z.im.addProd(real(a), imag(b))
	z.im.addProd(imag(a), real(b))
}

// addScaled adds a times the unrounded value of b.
func (z *compensatedComplex) addScaled(a complex128, b compensatedComplex) {
	z.re.addScaled(real(a), b.re)
	z.re.addScaled(-imag(a), b.im)
	z.im.addScaled(real(a), b.im)
	z.im.addScaled(imag(a), b.re)
}
