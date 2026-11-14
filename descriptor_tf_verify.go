package controlsys

import (
	"encoding/binary"
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"sync"

	"plantcontrol.org/v1/gonum/mat"
)

// pencilBoundedResp is the response of a descriptor model at points s by GEPP
// on sE − A, per entry (row-major p×m per point) with the first-order
// bounds of boundedResponse generalised to the pencil. comp bounds the
// error of the computed strictly proper part Ĉ(sE−A)⁻¹B̂: Û is refined
// pencilRefineSteps times against the residual B − (sE − A)Û formed in
// double-double (pencilResidualDD), and comp = (1 + 2(n+4)ε)·(Λ·ρ +
// γ_{n+1}|C||Û|), ρ the bound on the exact residual and Λ = |L̂| + (η/(1 −
// η) + γ_n)·|C||R̂| the multiplier that covers the error of the computed
// L̂ = Ĉ(sE−A)⁻¹ and R̂ = (sE−A)⁻¹. data, ε·(Λ(|s||E| + |A|)|U| + |C||U| +
// Λ|B| + |D|), is the componentwise change of G(s) when every stored entry
// moves by ε relative: the pencil form of sensWork.bound. terms is the
// entry's terms T̂ = Σ_l |C_il||Û_lj| + |D_ij| and dTer the bound
// ε·(|C|(|R̂|(|s||E| + |A|)|Û| + |R̂||B|) + T̂) + comp on |T̂ − T|, as
// zpkPointFactors bounds the terms of an explicit model. eta is the
// point's determinacy η (pencilDeterminacy); where η < 1 comp, data and
// dTer are scaled by 1/(1 − η), and where η ≥ 1 the point is undetermined
// and its bounds are not valid. ok is false at a point where GEPP meets an
// exactly zero pivot. g excludes D.
type pencilBoundedResp struct {
	p, m                    int
	g                       []complex128
	comp, data, terms, dTer []float64
	eta                     []float64
	ok                      []bool
}

func newPencilBoundedResponse(sys *System, pts []complex128) pencilBoundedResp {
	n, m, p := sys.Dims()
	np, pm := len(pts), p*m
	out := pencilBoundedResp{p: p, m: m, g: make([]complex128, np*pm), comp: make([]float64, np*pm),
		data: make([]float64, np*pm), terms: make([]float64, np*pm), dTer: make([]float64, np*pm),
		eta: make([]float64, np), ok: make([]bool, np)}
	rinv := make([]complex128, n*n)
	absR := make([]float64, n*n)
	rw, rbm := make([]float64, n*m), make([]float64, n*m)
	res, lo := make([]complex128, n*m), make([]complex128, n*m)
	a, e, b, c := denseData(sys.A), denseData(sys.E), denseData(sys.B), denseData(sys.C)
	x := make([]complex128, n*n)
	piv, perm := make([]int, n), make([]int, n)
	luRow := make([]float64, n)
	cr := make([]float64, p*n)
	u, v := make([]complex128, n*m), make([]complex128, n*p)
	absU, au, eu, rb := make([]float64, n*m), make([]float64, n*m), make([]float64, n*m), make([]float64, n*m)
	gam := make([]float64, n)
	for i := range n {
		k := 0
		for j := range n {
			if a[i*n+j] != 0 || e[i*n+j] != 0 {
				k++
			}
		}
		gam[i] = float64(24*k+8) * ddOpErr
	}
	safety, gn := 1+2*float64(n+4)*eps(), gammaN(n+1)
	for k, s := range pts {
		for i := range n * n {
			x[i] = s*complex(e[i], 0) - complex(a[i], 0)
		}
		if !cgeppFactor(x, n, piv) {
			continue
		}
		for i, bv := range b {
			u[i] = complex(bv, 0)
		}
		for i := range n {
			for j := range p {
				v[i*p+j] = complex(c[j*n+i], 0)
			}
		}
		cgeppSolve(x, n, piv, u, m)
		cgeppSolveT(x, n, piv, v, p)
		clear(rinv)
		for i := range n {
			rinv[i*n+i] = 1
		}
		cgeppSolve(x, n, piv, rinv, n)
		for i, z := range rinv {
			absR[i] = cabs(z)
		}
		as := cmplx.Abs(s)
		eta := pencilDeterminacy(x, piv, absR, n, perm, luRow)
		out.eta[k] = eta
		grow := 1.0
		if eta < 1 {
			grow = 1 / (1 - eta)
		}
		clear(lo)
		for range pencilRefineSteps {
			pencilResidualDD(a, e, b, s, u, lo, n, m, res)
			cgeppSolve(x, n, piv, res, m)
			for i := range lo {
				lo[i] += res[i]
			}
		}
		pencilResidualDD(a, e, b, s, u, lo, n, m, res)
		for i := range u {
			u[i] += lo[i]
		}
		for i := range n * m {
			absU[i] = cabs(u[i])
		}
		for i := range n {
			for j := range m {
				var sa, se float64
				for l := range n {
					sa += math.Abs(a[i*n+l]) * absU[l*m+j]
					se += math.Abs(e[i*n+l]) * absU[l*m+j]
				}
				au[i*m+j], eu[i*m+j] = sa, se
				rb[i*m+j] = (1+2*eps())*cabs(res[i*m+j]) + gam[i]*(math.Abs(b[i*m+j])+sa+as*se)
			}
		}
		for i := range n {
			for j := range m {
				var w, bb float64
				for l := range n {
					w += absR[i*n+l] * (au[l*m+j] + as*eu[l*m+j])
					bb += absR[i*n+l] * math.Abs(b[l*m+j])
				}
				rw[i*m+j], rbm[i*m+j] = w, bb
			}
		}
		mult := 1.0
		if eta < 1 {
			mult = eta/(1-eta) + gammaN(n)
		}
		for i := range p {
			for l := range n {
				w := 0.0
				for q := range n {
					w += math.Abs(c[i*n+q]) * absR[q*n+l]
				}
				cr[i*n+l] = w
			}
		}
		o := k * pm
		for i := range p {
			for j := range m {
				var g complex128
				var tc, cu, t, ct float64
				for l := range n {
					cl := c[i*n+l]
					g += complex(cl, 0) * u[l*m+j]
					al := cabs(v[l*p+i]) + mult*cr[i*n+l]
					tc += al * rb[l*m+j]
					cu += math.Abs(cl) * absU[l*m+j]
					t += al * (au[l*m+j] + as*eu[l*m+j] + math.Abs(b[l*m+j]))
					ct += math.Abs(cl) * (rw[l*m+j] + rbm[l*m+j])
				}
				d := math.Abs(sys.D.At(i, j))
				out.g[o+i*m+j] = g
				out.comp[o+i*m+j] = grow * safety * (tc + gn*cu)
				out.data[o+i*m+j] = grow * eps() * (t + cu + d)
				out.terms[o+i*m+j] = cu + d
				out.dTer[o+i*m+j] = grow*eps()*(ct+cu+d) + out.comp[o+i*m+j]
			}
		}
		out.ok[k] = true
	}
	return out
}

// exactCertPrec is the precision at which descriptorTFVerified evaluates
// the returned coefficients.
const exactCertPrec = 320

// ratPolyAt evaluates the ascending polynomial c at s by Horner's rule at
// exactCertPrec bits, rounded to complex128.
func ratPolyAt(c []*big.Rat, s complex128) complex128 {
	nw := func() *big.Float { return new(big.Float).SetPrec(exactCertPrec) }
	sr, si := nw().SetFloat64(real(s)), nw().SetFloat64(imag(s))
	re, im := nw(), nw()
	var t1, t2 big.Float
	t1.SetPrec(exactCertPrec)
	t2.SetPrec(exactCertPrec)
	for _, v := range slices.Backward(c) {
		nr := nw().Sub(t1.Mul(re, sr), t2.Mul(im, si))
		ni := nw().Add(t1.Mul(re, si), t2.Mul(im, sr))
		re, im = nr.Add(nr, nw().SetRat(v)), ni
	}
	a, _ := re.Float64()
	b, _ := im.Float64()
	return complex(a, b)
}

// floatPolyAt is ratPolyAt for descending float64 coefficients.
func floatPolyAt(c []float64, s complex128) complex128 {
	r := make([]*big.Rat, len(c))
	for k, v := range c {
		r[len(c)-1-k] = new(big.Rat).SetFloat64(v)
	}
	return ratPolyAt(r, s)
}

// pencilRefineSteps is the number of refinement steps of the solve in
// newPencilBoundedResponse: the correction is accumulated apart from Û, so
// the residual sees Û + δ unrounded.
const pencilRefineSteps = 2

// pencilResidualDD sets r to B − (sE − A)·(u + w) for the n×n row-major a,
// e and the n×m b, u, w, every product and sum in double-double
// (margin_ddeval.go), then rounded. Each of the k column positions of the
// row where A or E is nonzero adds, for u and w each, A's two exact
// products (ddProd) and one dd addition, and s·E·z as two exact products
// sr·e, si·e, four dd multiplications, two dd additions and one dd
// subtraction: at most 12 dd operations per position and vector, so a term
// passes through at most 24k + 8 operations of relative error ddOpErr
// each, and |r − r̂| ≤ ε|r̂| + (24k + 8)·ddOpErr·(|B| + (|A| + |s||E|)|u + w|)
// per entry, k the nonzeros of the row of A and E counted once per column.
func pencilResidualDD(a, e, b []float64, s complex128, u, w []complex128, n, m int, r []complex128) {
	sr, si := real(s), imag(s)
	for i := range n {
		for j := range m {
			acc := ddComplex{re: dd{b[i*m+j], 0}}
			for l := range n {
				av, ev := a[i*n+l], e[i*n+l]
				if av == 0 && ev == 0 {
					continue
				}
				er, ei := ddProd(sr, ev), ddProd(si, ev)
				for _, z := range [2]complex128{u[l*m+j], w[l*m+j]} {
					if z == 0 {
						continue
					}
					zr, zi := dd{real(z), 0}, dd{imag(z), 0}
					acc = acc.add(ddComplex{ddProd(av, real(z)), ddProd(av, imag(z))})
					acc = acc.sub(ddComplex{er.mul(zr).add(ei.mul(zi).neg()), er.mul(zi).add(ei.mul(zr))})
				}
			}
			r[i*m+j] = acc.complex128()
		}
	}
}

// pencilDeterminacy is η(s) = γ_n·max_i Σ_l |R_il|·w_l, w = Pᵀ(|L_f||U_f|)·1
// the row sums of the backward error bound of the GEPP factors PM = L_fU_f
// of M = sE − A (cgeppFactor's x and piv, M + ΔM = PᵀL_fU_f with |ΔM| ≤
// γ_n·Pᵀ|L_f||U_f|), and absR = |R̂|, R̂ the computed M⁻¹: a bound on
// ‖R̂·ΔM‖ in the row norm. Below 1 the factorization determines the
// response at s and the first-order bounds hold scaled by 1/(1 − η); at or
// above 1 the point is undetermined and is never counted. It is the one
// determinacy test of the descriptor TF certificate and its census. perm
// and luRow are scratch of length n.
func pencilDeterminacy(x []complex128, piv []int, absR []float64, n int, perm []int, luRow []float64) float64 {
	for i := range perm {
		perm[i] = i
	}
	for k := range n {
		perm[k], perm[piv[k]] = perm[piv[k]], perm[k]
	}
	for r := range n {
		w := 0.0
		for q := 0; q <= r; q++ {
			lq := 1.0
			if q < r {
				lq = cabs(x[r*n+q])
			}
			if lq == 0 {
				continue
			}
			uq := 0.0
			for j := q; j < n; j++ {
				uq += cabs(x[q*n+j])
			}
			w += lq * uq
		}
		luRow[perm[r]] = w
	}
	worst := 0.0
	for i := range n {
		v := 0.0
		for l := range n {
			v += absR[i*n+l] * luRow[l]
		}
		worst = max(worst, v)
	}
	return gammaN(n) * worst
}

func denseData(a *mat.Dense) []float64 {
	r, c := a.Dims()
	out := make([]float64, r*c)
	for i := range r {
		for j := range c {
			out[i*c+j] = a.At(i, j)
		}
	}
	return out
}

// tfShape returns the roots of tf, the poles of every row's denominator
// and the zeros of every entry's numerator, the poles alone, its largest
// row denominator degree and the most by which an entry's numerator degree
// exceeds its row's; false when a root search fails.
func tfShape(tf *TransferFunc) (roots, poles []complex128, nf, polyDeg int, ok bool) {
	add := func(dst []complex128, c []float64) ([]complex128, bool) {
		c = trimLeadingZeros(c)
		if len(c) < 2 {
			return dst, true
		}
		r, err := Poly(c).Roots()
		if err != nil {
			return nil, false
		}
		for _, z := range r {
			if !slices.Contains(dst, z) {
				dst = append(dst, z)
			}
		}
		return dst, true
	}
	for i, den := range tf.Den {
		d := len(trimLeadingZeros(den)) - 1
		nf = max(nf, d)
		if poles, ok = add(poles, den); !ok {
			return nil, nil, 0, 0, false
		}
		for _, num := range tf.Num[i] {
			polyDeg = max(polyDeg, len(trimLeadingZeros(num))-1-d)
			if roots, ok = add(roots, num); !ok {
				return nil, nil, 0, 0, false
			}
		}
	}
	for _, z := range poles {
		if !slices.Contains(roots, z) {
			roots = append(roots, z)
		}
	}
	return roots, poles, nf, polyDeg, true
}

// rootBand returns the band of roots on the frequency axis: for a
// continuous model the least and largest nonzero finite |λ|, falling back
// to the data scale ‖A‖/‖E‖ of sys when there are none; for a discrete one
// the least nonzero |λ − 1| (at least √ε) and π.
func rootBand(sys *System, roots []complex128) (lo, hi float64) {
	discrete := sys.Dt != 0
	lo, hi = math.Inf(1), 0.0
	for _, z := range roots {
		w := cmplx.Abs(z)
		if discrete {
			w = cmplx.Abs(z - 1)
		}
		if w > 0 && !math.IsInf(w, 0) && !math.IsNaN(w) {
			lo, hi = math.Min(lo, w), math.Max(hi, w)
		}
	}
	if discrete {
		lo = math.Max(math.Min(lo, math.Pi), math.Sqrt(eps()))
		return lo, math.Pi
	}
	if hi == 0 {
		w := mat.Norm(sys.A, 2) / mat.Norm(sys.E, 2)
		if !(w > 0) || math.IsInf(w, 0) {
			w = 1
		}
		lo, hi = w, w
	}
	return lo, hi
}

// axisPoint is j·w for a continuous model, e^{jw} for a discrete one.
func axisPoint(discrete bool, w float64) complex128 {
	if discrete {
		return cmplx.Rect(1, w)
	}
	return complex(0, w)
}

// descriptorTFLadder returns the half-decade frequencies 10^(k/2), k
// integer, over [10⁻³·lo, 10³·hi] (over [10⁻³·lo, π] for a discrete model):
// the census of descriptorTFVerified judges at these and the certificate
// probes them all, so every census point is a certificate point.
func descriptorTFLadder(lo, hi float64, discrete bool) []complex128 {
	var pts []complex128
	top := 1e3 * hi
	if discrete {
		top = math.Pi
	}
	for k := math.Floor(2 * math.Log10(1e-3*lo)); math.Pow(10, k/2) <= top; k++ {
		pts = append(pts, axisPoint(discrete, math.Pow(10, k/2)))
	}
	return pts
}

// descriptorTFProbes returns the structured points of descriptorTFVerified
// for tf of sys: the axis points of tf's roots (eigenAxisProbes,
// gapProbePoints), s = 0 (z = 1 and z = −1 for a discrete model), and the
// half-decade ladder (descriptorTFLadder) over the roots' band.
func descriptorTFProbes(sys *System, roots []complex128) []complex128 {
	discrete := sys.Dt != 0
	var pts []complex128
	push := func(s complex128) {
		if !cmplx.IsNaN(s) && !cmplx.IsInf(s) && !slices.Contains(pts, s) {
			pts = append(pts, s)
		}
	}
	for _, s := range eigenAxisProbes(roots, discrete) {
		push(s)
	}
	for _, s := range gapProbePoints(roots, discrete) {
		push(s)
	}
	if discrete {
		push(1)
		push(-1)
	} else {
		push(0)
	}
	lo, hi := rootBand(sys, roots)
	for _, s := range descriptorTFLadder(lo, hi, discrete) {
		push(s)
	}
	return pts
}

// tfPoleWindows returns the windows of the roots of tf, the poles of every
// row's denominator and the zeros of every entry's numerator: per root λ of
// a polynomial c of degree d, the disk of radius β/tol about the centroid
// μ of the cluster of m roots about λ that rounding cannot separate, β =
// (Σ_k bound_k·|μ|^k / |c^(m)(μ)/m!|)^(1/m) its bound from coefficient
// bounds bound_k = (d + 1)·ε·‖c‖_∞, the rounding the reduction leaves in c:
// from m = 1 (μ = λ), the next nearest root of c joins while the radius
// recomputed for the merged cluster, with m + 1 about its centroid, still
// covers every member. The disk is capped at the distance from μ to the
// nearest pole of poles outside the cluster, and exists when β ≤ tol·|μ|
// or the cluster is rounding-level (|μ| ≤ β): there a rounding-level move
// of the roots changes the response by more than tol, so a point inside
// refutes nothing. A polynomial whose leading coefficient is within its own
// bound has no determined degree, and its roots get no window: a root
// that bound places anywhere would otherwise take its neighbours into
// a cluster with an unbounded disk.
func tfPoleWindows(tf *TransferFunc, poles []complex128, tol float64) poleWindows {
	var w poleWindows
	add := func(c []float64) {
		c = trimLeadingZeros(c)
		d := len(c) - 1
		if d < 1 {
			return
		}
		norm := 0.0
		for _, v := range c {
			norm = math.Max(norm, math.Abs(v))
		}
		bk := float64(d+1) * eps() * norm
		if bk >= math.Abs(c[0]) {
			return
		}
		r, err := Poly(c).Roots()
		if err != nil {
			return
		}
		radius := func(mu complex128, m int) float64 {
			sum, am := 0.0, cmplx.Abs(mu)
			for k := range d + 1 {
				sum += bk * math.Pow(am, float64(k))
			}
			return math.Pow(sum/cmplx.Abs(taylorCoef(c, mu, m)), 1/float64(m))
		}
		for _, lam := range r {
			members, mu := []complex128{lam}, lam
			beta := radius(mu, 1)
			for len(members) < d {
				next, best := complex(0, 0), math.Inf(1)
				for _, o := range r {
					if dist := cmplx.Abs(o - mu); !slices.Contains(members, o) && dist < best {
						next, best = o, dist
					}
				}
				cand := append(slices.Clone(members), next)
				var sum complex128
				for _, z := range cand {
					sum += z
				}
				nmu := sum / complex(float64(len(cand)), 0)
				nb := radius(nmu, len(cand))
				spread := 0.0
				for _, z := range cand {
					spread = math.Max(spread, cmplx.Abs(z-nmu))
				}
				if !(spread <= nb) {
					break
				}
				members, mu, beta = cand, nmu, nb
			}
			am := cmplx.Abs(mu)
			if !(beta <= tol*am || am <= beta) {
				continue
			}
			lim := math.Inf(1)
			for _, o := range poles {
				if !slices.Contains(members, o) {
					lim = math.Min(lim, cmplx.Abs(mu-o))
				}
			}
			w.centre = append(w.centre, mu)
			w.radius = append(w.radius, math.Min(beta/tol, lim))
		}
	}
	for i, den := range tf.Den {
		add(den)
		for _, num := range tf.Num[i] {
			add(num)
		}
	}
	return w
}

// descriptorTFPoint is the judgement of tf at one point: verified, refuted,
// or neither: undetermined at η ≥ 1, inside a pole window, or open, an
// entry outside every window that the bounds of pencilBoundedResp neither
// verify nor refute. An open entry is evaluated exactly (exactPencilAt,
// within exactDetMaxOrder) and refutes tf when |tf − G| > tol·max(scale, T) +
// δ_data; otherwise, or past that cost cap, it is conceded: it neither
// counts as verified nor blocks the certificate. The exact evaluation is
// skipped where the bounds already exclude both outcomes: |G − tf| lies in
// |Ĝ − tf| ± δ and T in T̂ ± δ_T, so an entry whose upper |Ĝ − tf| + δ is
// within tol·max(scale, T̂ − δ_T) + δ_data and whose lower |Ĝ − tf| − δ
// exceeds tol·max(scale, T̂ + δ_T) can neither refute nor verify; worst
// is the largest (|Ĝ − tf| − δ)/(tol·max(scale, T̂ + δ_T) + δ_data) over its
// entries that can refute, above 1 exactly when the point refutes tf.
type descriptorTFPoint struct {
	determined, verified, refuted, open bool
	worst                               float64
}

// judgeDescriptorTF judges tf against full, the bounded pencil response of
// sys at pts (see descriptorTFVerified), with win the pole windows.
func judgeDescriptorTF(sys *System, tf *TransferFunc, pts []complex128, full pencilBoundedResp, win poleWindows) []descriptorTFPoint {
	_, m, p := sys.Dims()
	tol := math.Sqrt(eps())
	pm := p * m
	out := make([]descriptorTFPoint, len(pts))
	r := make([]complex128, m)
	for k, s := range pts {
		if !full.ok[k] || !(full.eta[k] < 1) {
			continue
		}
		jp := descriptorTFPoint{determined: true, verified: true}
		in := win.contain(s)
		var ur, ui [][]*big.Rat
		exact := false
		for i := range p {
			dv := floatPolyAt(tf.Den[i], s)
			if dv == 0 {
				jp.verified = false
				break
			}
			scale := 0.0
			for j := range m {
				r[j] = floatPolyAt(tf.Num[i][j], s) / dv
				g := full.g[k*pm+i*m+j] + complex(sys.D.At(i, j), 0)
				scale = math.Max(scale, math.Max(cmplx.Abs(r[j]), cmplx.Abs(g)))
			}
			for j := range m {
				e := k*pm + i*m + j
				d := math.Abs(sys.D.At(i, j))
				g := full.g[e] + complex(sys.D.At(i, j), 0)
				diff := cmplx.Abs(g - r[j])
				delta := full.comp[e] + eps()*(cmplx.Abs(g)+d) + 4*eps()*cmplx.Abs(r[j])
				T, dT := full.terms[e], full.dTer[e]
				ver := diff+delta <= tol*math.Max(scale, T-dT)
				thr := tol*math.Max(scale, T+dT) + full.data[e]
				ref := dT < T && diff-delta > thr
				lo, hi := math.Max(0, diff-delta), diff+delta
				canRef := hi > tol*math.Max(scale, math.Max(0, T-dT))+full.data[e]
				canVer := lo <= tol*math.Max(scale, T+dT)
				if !ver && !ref && !in && (canRef || canVer || !tfPruneExact) {
					if !exact {
						exact = true
						if n, _, _ := sys.Dims(); n <= exactDetMaxOrder {
							ur, ui = exactPencilAt(sys, s)
						}
					}
					if ur != nil {
						g, T = exactEntry(sys, ur, ui, i, j)
						diff = cmplx.Abs(g - r[j])
						delta = 4 * eps() * (cmplx.Abs(g) + cmplx.Abs(r[j]) + T)
						ver = diff+delta <= tol*math.Max(scale, T)
						thr = tol*math.Max(scale, T) + full.data[e]
						ref = diff-delta > thr
					}
					jp.open = jp.open || !ver && !ref
				}
				if !ver {
					jp.verified = false
				}
				if !in && (dT < T || ur != nil) {
					jp.refuted = jp.refuted || ref
					jp.worst = math.Max(jp.worst, (diff-delta)/thr)
				}
			}
		}
		out[k] = jp
	}
	return out
}

// descriptorTFVerified reports whether tf, the transfer function
// TransferFunction returns for the descriptor sys (no internal delays), is
// verified against the stored model. The contract, per entry at every axis
// point where GEPP on sE − A determines the response (pencilDeterminacy
// η < 1), is |G − tf| ≤ tol·max(scale, T), tol = √ε, G = C(sE − A)⁻¹B + D,
// scale the largest |G|, |tf| of the row and T = Σ_l |C_il||U_lj| + |D_ij|
// the entry's terms, U = (sE − A)⁻¹B, as zpkRowCheck.check states it for an
// explicit model. Ĝ and its bound δ_G come from pencilBoundedResp, and tf
// from its float64 coefficients as returned, evaluated at exactCertPrec
// bits and divided, δ_tf = 4ε·|tf| for the rounding of numerator,
// denominator and quotient. An entry verifies when |Ĝ − tf| + δ ≤
// tol·max(scale, T̂ − δ_T), δ = δ_G + ε(|Ĝ| + |D|) + δ_tf, and refutes tf
// when |Ĝ − tf| − δ > tol·max(scale, T̂ + δ_T) + δ_data outside a pole
// window (tfPoleWindows), δ_data the data bound, the resolution of the
// stored model; with δ_T ≥ T̂ it only verifies against scale, and a point
// with η ≥ 1 is never counted. A point verifies when all its entries do.
// tf is verified when no point refutes it and the verified points,
// conjugate-weighted (probeWeight), give at least n + nf + polyDeg + 1
// interpolation conditions, n the order of sys, nf the largest row
// denominator degree of tf and polyDeg its polynomial degree: G − tf has a
// numerator of degree at most n + nf + polyDeg, so agreement there fixes
// it in exact arithmetic but bounds it nowhere between the points. The
// points are descriptorTFProbes of tf's roots, its poles and the zeros of
// every entry; for a continuous model the ladder then continues upward
// to dominatedFrequency, a point at a time, skipping undetermined points
// (η ≥ 1 or a zero pivot), and
// the point at ∞ is judged (descriptorTFInfinityRefutes), refuting but
// never counted; while short of the count, up to three rounds of
// zpkFillProbes add ⌈k/2⌉·2^r new points over [lo/10, 10·hi] of the roots,
// k the shortfall, points already judged skipped. Fixed fill rounds of
// 4·2^r points (r = 0, 1, 2) are judged whatever the count, and the ladder
// continues below 10⁻³·lo, a point at a time, down to belowLadderFloor,
// skipping undetermined points: there the response tends to its static
// value, which the stored data determine. The census of these models
// judges the same points.
func descriptorTFVerified(sys *System, tf *TransferFunc) bool {
	if !tfVerdictMemo {
		return descriptorTFVerify(sys, tf)
	}
	key := tfVerdictKey(sys, tf)
	tfVerdicts.Lock()
	v, ok := tfVerdicts.m[key]
	tfVerdicts.Unlock()
	if ok {
		return v
	}
	v = descriptorTFVerify(sys, tf)
	tfVerdicts.Lock()
	if tfVerdicts.m == nil || len(tfVerdicts.m) >= tfVerdictMemoSize {
		tfVerdicts.m = make(map[string]bool)
	}
	tfVerdicts.m[key] = v
	tfVerdicts.Unlock()
	return v
}

// tfVerdicts memoizes descriptorTFVerified per model and transfer function,
// so TransferFunction, IsProper and Step, which judge the same reduction,
// share one verdict. The key holds every stored entry and coefficient
// bit for bit, and the verdict is a deterministic function of them, so a
// memoized verdict is the one a fresh judgement gives. tfVerdictMemoSize
// bounds the entries; the map is dropped when full. tfVerdictMemo is a
// test seam.
var (
	tfVerdicts struct {
		sync.Mutex
		m map[string]bool
	}
	tfVerdictMemo = true
	// tfPruneExact is a test seam: false forces the exact evaluation of
	// every open entry.
	tfPruneExact = true
)

const tfVerdictMemoSize = 1024

// tfVerdictKey encodes sys's A, B, C, D, E and Dt and tf's coefficients.
func tfVerdictKey(sys *System, tf *TransferFunc) string {
	var b []byte
	put := func(v float64) { b = binary.LittleEndian.AppendUint64(b, math.Float64bits(v)) }
	putDense := func(a *mat.Dense) {
		if a == nil {
			put(-1)
			return
		}
		r, c := a.Dims()
		put(float64(r))
		put(float64(c))
		for i := range r {
			for j := range c {
				put(a.At(i, j))
			}
		}
	}
	for _, a := range []*mat.Dense{sys.A, sys.B, sys.C, sys.D, sys.E} {
		putDense(a)
	}
	put(sys.Dt)
	for i, den := range tf.Den {
		put(float64(len(den)))
		for _, v := range den {
			put(v)
		}
		for _, num := range tf.Num[i] {
			put(float64(len(num)))
			for _, v := range num {
				put(v)
			}
		}
	}
	return string(b)
}

// descriptorTFVerify is descriptorTFVerified without the memo.
func descriptorTFVerify(sys *System, tf *TransferFunc) bool {
	n, m, p := sys.Dims()
	if n == 0 || m == 0 || p == 0 {
		return true
	}
	roots, poles, nf, polyDeg, ok := tfShape(tf)
	if !ok {
		return false
	}
	need := n + nf + polyDeg + 1
	win := tfPoleWindows(tf, poles, math.Sqrt(eps()))
	var seen []complex128
	verified := 0
	judge := func(cand []complex128) bool {
		var pts []complex128
		for _, s := range cand {
			if !slices.Contains(seen, s) {
				seen = append(seen, s)
				pts = append(pts, s)
			}
		}
		if len(pts) == 0 {
			return true
		}
		for k, jp := range judgeDescriptorTF(sys, tf, pts, newPencilBoundedResponse(sys, pts), win) {
			if jp.refuted {
				return false
			}
			if jp.verified {
				verified += probeWeight(pts[k])
			}
		}
		return true
	}
	if !judge(descriptorTFProbes(sys, roots)) {
		return false
	}
	if sys.Dt == 0 {
		_, hi := rootBand(sys, roots)
		sdom := dominatedFrequency(sys)
		for k := math.Ceil(2*math.Log10(1e3*hi)) + 1; ; k++ {
			w := math.Pow(10, k/2)
			if !(w <= sdom) {
				break
			}
			s := complex(0, w)
			if slices.Contains(seen, s) {
				continue
			}
			seen = append(seen, s)
			full := newPencilBoundedResponse(sys, []complex128{s})
			if !full.ok[0] || !(full.eta[0] < 1) {
				continue
			}
			jp := judgeDescriptorTF(sys, tf, []complex128{s}, full, win)[0]
			if jp.refuted {
				return false
			}
			if jp.verified {
				verified += probeWeight(s)
			}
		}
	}
	if sys.Dt == 0 && descriptorTFInfinityRefutes(sys, tf) {
		return false
	}
	for r := range 3 {
		if !judge(zpkFillProbes(roots, nil, sys.Dt != 0, 4<<r)) {
			return false
		}
	}
	for r := 0; r < 3 && verified < need; r++ {
		if !judge(zpkFillProbes(roots, nil, sys.Dt != 0, (need-verified+1)/2<<r)) {
			return false
		}
	}
	lo, _ := rootBand(sys, roots)
	discrete := sys.Dt != 0
	floor := belowLadderFloor(sys)
	for k := math.Floor(2*math.Log10(1e-3*lo)) - 1; ; k-- {
		w := math.Pow(10, k/2)
		if !(w > floor) {
			break
		}
		s := axisPoint(discrete, w)
		if slices.Contains(seen, s) {
			continue
		}
		seen = append(seen, s)
		full := newPencilBoundedResponse(sys, []complex128{s})
		if !full.ok[0] || !(full.eta[0] < 1) {
			continue
		}
		jp := judgeDescriptorTF(sys, tf, []complex128{s}, full, win)[0]
		if jp.refuted {
			return false
		}
		if jp.verified {
			verified += probeWeight(s)
		}
	}
	return verified >= need
}

// belowLadderFloor is the frequency under which the continuation of the
// ladder below the roots of descriptorTFVerified stops: the largest w at
// which some row l of the pencil still tells s = jw from s = 0 to the data,
// w·Σ_j|E_lj| > ε·Σ_j|A_lj| (for a discrete model z = e^{jw} from z = 1,
// w·Σ_j|E_lj| > ε·Σ_j|E_lj − A_lj|), and at least the least positive
// normal number: below it the static probe covers the band.
func belowLadderFloor(sys *System) float64 {
	n, _, _ := sys.Dims()
	floor := 0.0
	for l := range n {
		var se, sa float64
		for j := range n {
			e, a := sys.E.At(l, j), sys.A.At(l, j)
			se += math.Abs(e)
			if sys.Dt != 0 {
				sa += math.Abs(e - a)
			} else {
				sa += math.Abs(a)
			}
		}
		if se > 0 {
			floor = math.Max(floor, eps()*sa/se)
		}
	}
	return math.Max(floor, 0x1p-1022)
}

// exactPencilAt returns U = (sE − A)⁻¹B at s exactly, s's parts and the
// data taken as the rationals they are, by elimination on the real 2n×2n
// embedding; nil when sE − A is singular.
func exactPencilAt(sys *System, s complex128) (ur, ui [][]*big.Rat) {
	n, m, _ := sys.Dims()
	N := 2 * n
	M := make([][]*big.Rat, N)
	for i := range N {
		M[i] = make([]*big.Rat, N+2*m)
		for j := range M[i] {
			M[i][j] = new(big.Rat)
		}
	}
	sr, si := new(big.Rat).SetFloat64(real(s)), new(big.Rat).SetFloat64(imag(s))
	var t big.Rat
	for i := range n {
		for j := range n {
			e, a := new(big.Rat).SetFloat64(sys.E.At(i, j)), new(big.Rat).SetFloat64(sys.A.At(i, j))
			re := new(big.Rat).Sub(t.Mul(sr, e), a)
			im := new(big.Rat).Mul(si, e)
			M[i][j].Set(re)
			M[n+i][n+j].Set(re)
			M[i][n+j].Neg(im)
			M[n+i][j].Set(im)
		}
		for j := range m {
			M[i][N+j].SetFloat64(sys.B.At(i, j))
		}
	}
	for col := range N {
		piv := -1
		for r := col; r < N; r++ {
			if M[r][col].Sign() != 0 {
				piv = r
				break
			}
		}
		if piv < 0 {
			return nil, nil
		}
		M[col], M[piv] = M[piv], M[col]
		inv := new(big.Rat).Inv(M[col][col])
		for j := col; j < len(M[col]); j++ {
			M[col][j].Mul(M[col][j], inv)
		}
		for r := range N {
			if r == col || M[r][col].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Set(M[r][col])
			for j := col; j < len(M[r]); j++ {
				M[r][j].Sub(M[r][j], t.Mul(f, M[col][j]))
			}
		}
	}
	ur, ui = make([][]*big.Rat, n), make([][]*big.Rat, n)
	for i := range n {
		ur[i], ui[i] = make([]*big.Rat, m), make([]*big.Rat, m)
		for j := range m {
			ur[i][j], ui[i][j] = M[i][N+j], M[n+i][N+j]
		}
	}
	return ur, ui
}

// exactEntry returns G_ij = Σ_l C_il·U_lj + D_ij from the exact U of
// exactPencilAt, rounded, and its terms T = Σ_l |C_il||U_lj| + |D_ij|.
func exactEntry(sys *System, ur, ui [][]*big.Rat, i, j int) (complex128, float64) {
	n, _, _ := sys.Dims()
	gr, gi := new(big.Rat).SetFloat64(sys.D.At(i, j)), new(big.Rat)
	var t big.Rat
	T := math.Abs(sys.D.At(i, j))
	for l := range n {
		c := new(big.Rat).SetFloat64(sys.C.At(i, l))
		gr.Add(gr, t.Mul(c, ur[l][j]))
		gi.Add(gi, t.Mul(c, ui[l][j]))
		a, _ := ur[l][j].Float64()
		b, _ := ui[l][j].Float64()
		T += math.Abs(sys.C.At(i, l)) * cmplx.Abs(complex(a, b))
	}
	a, _ := gr.Float64()
	b, _ := gi.Float64()
	return complex(a, b), T
}

// taylorCoef returns c^(m)(λ)/m! for the descending coefficients c.
func taylorCoef(c []float64, lam complex128, m int) complex128 {
	d := len(c) - 1
	var v complex128
	for k := m; k <= d; k++ {
		binom := 1.0
		for i := range m {
			binom = binom * float64(k-i) / float64(i+1)
		}
		v += complex(binom*c[d-k], 0) * cmplx.Pow(lam, complex(float64(k-m), 0))
	}
	return v
}

// dominatedFrequency is s_dom = max_l Σ_j|A_lj| / (ε·Σ_j|E_lj|) over the
// rows of the pencil with E nonzero: above it sE − A rounds to sE row by
// row, and the point at ∞ (descriptorTFInfinityRefutes) judges what is
// left of the band.
func dominatedFrequency(sys *System) float64 {
	n, _, _ := sys.Dims()
	sdom := 0.0
	for l := range n {
		var se, sa float64
		for j := range n {
			se += math.Abs(sys.E.At(l, j))
			sa += math.Abs(sys.A.At(l, j))
		}
		if se > 0 {
			sdom = math.Max(sdom, sa/(eps()*se))
		}
	}
	if sdom == 0 || math.IsInf(sdom, 0) {
		return math.MaxFloat64
	}
	return sdom
}

// descriptorTFInfinityRefutes judges tf of the continuous descriptor sys at
// s = ∞ through the swapped pencil tA − E at t = 0. Its determinacy η_∞
// (pencilDeterminacy of GEPP on −E) at or above 1 concedes ∞ (false).
// Otherwise E is robustly invertible, every model within terms of the data
// is proper with G(∞) = D, and an entry refutes tf when tf has an s-term
// (polyDeg ≥ 1) or |t₀ − D| > √ε·max(|t₀|, |D|) + ε|D|/(1 − η_∞), t₀ = tf(∞)
// in exact rationals. When D = 0 = t₀ the Markov parameters decide: g_k =
// C(E⁻¹A)^(k−1)E⁻¹B exactly, t_k the s^(−k) coefficient of tf exactly, T_k
// = Σ_l |C_il||(E⁻¹A)^(k−1)E⁻¹B|_lj, and δ_1 the data bound of the swapped
// pencil's response at t = 0 (pencilBoundedResp, D = 0); order k refutes when |t_k − g_k| > √ε·max(|t_k|,
// |g_k|, T_k) + δ_k, and the next order is judged while t_k, g_k and T_k
// are all 0, up to n orders. δ_k above order 1 is markovBounds.bound over
// (1 − η_∞)^k.
func descriptorTFInfinityRefutes(sys *System, tf *TransferFunc) bool {
	n, m, p := sys.Dims()
	if n == 0 || n > exactDetMaxOrder {
		return false
	}
	sw := sys.Copy()
	sw.A, sw.E = mat.DenseCopyOf(sys.E), mat.DenseCopyOf(sys.A)
	sw.D = mat.NewDense(p, m, nil)
	full := newPencilBoundedResponse(sw, []complex128{0})
	if !full.ok[0] || !(full.eta[0] < 1) {
		return false
	}
	eta := full.eta[0]
	tol := math.Sqrt(eps())
	if tfPolyDegree(tf) >= 1 {
		return true
	}
	U := make([][][]*big.Rat, 0, n)
	var mb *markovBounds
	for i := range p {
		ser := tfInfinitySeries(tf.Num[i], tf.Den[i], n+1)
		for j := range m {
			t0, _ := ser[j][0].Float64()
			d := sys.D.At(i, j)
			if math.Abs(t0-d) > tol*math.Max(math.Abs(t0), math.Abs(d))+eps()*math.Abs(d)/(1-eta) {
				return true
			}
			if d != 0 || ser[j][0].Sign() != 0 {
				continue
			}
			for k := 1; k <= n; k++ {
				for len(U) < k {
					prev := (*[][]*big.Rat)(nil)
					if len(U) > 0 {
						prev = &U[len(U)-1]
					}
					next, ok := markovSolve(sys, prev)
					if !ok {
						return false
					}
					U = append(U, next)
				}
				g, T := new(big.Rat), 0.0
				var tmp big.Rat
				for l := range n {
					c := new(big.Rat).SetFloat64(sys.C.At(i, l))
					g.Add(g, tmp.Mul(c, U[k-1][l][j]))
					v, _ := U[k-1][l][j].Float64()
					T += math.Abs(sys.C.At(i, l)) * math.Abs(v)
				}
				tk, _ := ser[j][k].Float64()
				gk, _ := g.Float64()
				delta := full.data[i*m+j]
				if k > 1 {
					if mb == nil {
						var mok bool
						if mb, mok = newMarkovBounds(sys); !mok {
							return false
						}
					}
					delta = mb.bound(k, i, j) / math.Pow(1-eta, float64(k))
				}
				if math.Abs(tk-gk) > tol*math.Max(math.Max(math.Abs(tk), math.Abs(gk)), T)+delta {
					return true
				}
				if tk != 0 || gk != 0 || T != 0 {
					break
				}
			}
		}
	}
	return false
}

// tfPolyDegree is the most by which an entry's numerator degree exceeds
// its row denominator's.
func tfPolyDegree(tf *TransferFunc) int {
	pd := 0
	for i, den := range tf.Den {
		d := len(trimLeadingZeros(den)) - 1
		for _, num := range tf.Num[i] {
			pd = max(pd, len(trimLeadingZeros(num))-1-d)
		}
	}
	return pd
}

// tfInfinitySeries returns, per numerator of a row over den (descending
// float64 coefficients, deg num ≤ deg den), the first k coefficients of
// its expansion num/den = Σ_j c_j·s^(−j) at s = ∞, exactly.
func tfInfinitySeries(nums [][]float64, den []float64, k int) [][]*big.Rat {
	d := trimLeadingZeros(den)
	dr := make([]*big.Rat, len(d))
	for i, v := range d {
		dr[i] = new(big.Rat).SetFloat64(v)
	}
	out := make([][]*big.Rat, len(nums))
	for j, num := range nums {
		nm := trimLeadingZeros(num)
		r := make([]*big.Rat, len(d)+k)
		for i := range r {
			r[i] = new(big.Rat)
		}
		off := len(d) - len(nm)
		for i, v := range nm {
			if off+i >= 0 {
				r[off+i].SetFloat64(v)
			}
		}
		c := make([]*big.Rat, k)
		var t big.Rat
		for q := range k {
			c[q] = new(big.Rat).Quo(r[q], dr[0])
			for i := 1; i < len(dr) && q+i < len(r); i++ {
				r[q+i].Sub(r[q+i], t.Mul(c[q], dr[i]))
			}
		}
		out[j] = c
	}
	return out
}

// markovSolve returns E⁻¹B (prev nil) or E⁻¹A·prev exactly, by elimination
// on E in rationals; false when E is exactly singular.
func markovSolve(sys *System, prev *[][]*big.Rat) ([][]*big.Rat, bool) {
	n, m, _ := sys.Dims()
	M := make([][]*big.Rat, n)
	var t big.Rat
	for i := range n {
		M[i] = make([]*big.Rat, n+m)
		for j := range n {
			M[i][j] = new(big.Rat).SetFloat64(sys.E.At(i, j))
		}
		for j := range m {
			v := new(big.Rat)
			if prev == nil {
				v.SetFloat64(sys.B.At(i, j))
			} else {
				for l := range n {
					a := new(big.Rat).SetFloat64(sys.A.At(i, l))
					v.Add(v, t.Mul(a, (*prev)[l][j]))
				}
			}
			M[i][n+j] = v
		}
	}
	for col := range n {
		piv := -1
		for r := col; r < n; r++ {
			if M[r][col].Sign() != 0 {
				piv = r
				break
			}
		}
		if piv < 0 {
			return nil, false
		}
		M[col], M[piv] = M[piv], M[col]
		inv := new(big.Rat).Inv(M[col][col])
		for j := col; j < n+m; j++ {
			M[col][j].Mul(M[col][j], inv)
		}
		for r := range n {
			if r == col || M[r][col].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Set(M[r][col])
			for j := col; j < n+m; j++ {
				M[r][j].Sub(M[r][j], t.Mul(f, M[col][j]))
			}
		}
	}
	out := make([][]*big.Rat, n)
	for i := range n {
		out[i] = M[i][n:]
	}
	return out, true
}

// markovBounds holds what markovBounds.bound needs: the computed Ê⁻¹, R =
// |Ê⁻¹|, |A|, |E|, B and C of a model with E invertible.
type markovBounds struct {
	n, m, p          int
	einv, a, b, c    *mat.Dense
	absR, absA, absE *mat.Dense
}

func newMarkovBounds(sys *System) (*markovBounds, bool) {
	n, m, p := sys.Dims()
	var einv mat.Dense
	if err := einv.Inverse(sys.E); err != nil {
		return nil, false
	}
	abs := func(x *mat.Dense) *mat.Dense {
		out := mat.DenseCopyOf(x)
		out.Apply(func(_, _ int, v float64) float64 { return math.Abs(v) }, out)
		return out
	}
	return &markovBounds{n: n, m: m, p: p, einv: &einv, a: sys.A, b: sys.B, c: sys.C,
		absR: abs(&einv), absA: abs(sys.A), absE: abs(sys.E)}, true
}

// bound returns the first-order change of entry (i, j) of the Markov
// parameter g_k = C(E⁻¹A)^(k−1)E⁻¹B when every entry of A, E, B and C moves
// by ε relative, componentwise: ε times the sum, over the 2k − 1 factor
// positions of the product and over C and B, of the product with that one
// factor replaced by its perturbation bound and every other factor by the
// modulus of its partial product, |L_p|·F_p·|R_p|, F_p = |A| at an A
// position and R|E|R at an E⁻¹ position (δ(E⁻¹) = −E⁻¹δE·E⁻¹, R = |Ê⁻¹|
// bounding |E⁻¹| to first order), |C|·|R_0| for C and |L_{2k−1}|·|B| for B,
// times 1 + γ_{(2k+1)n} for the rounding of its own evaluation. At k = 1 its
// E⁻¹ term ε·|C|R|E|R|B| bounds ε·|CE⁻¹||E||E⁻¹B|, the corresponding term
// of the data bound δ₁ of pencilBoundedResp, which order 1 keeps.
func (mb *markovBounds) bound(k, i, j int) float64 {
	n := mb.n
	factors := make([]*mat.Dense, 0, 2*k-1)
	abs := make([]*mat.Dense, 0, 2*k-1)
	rer := new(mat.Dense)
	rer.Product(mb.absR, mb.absE, mb.absR)
	for q := range 2*k - 1 {
		if q%2 == 0 {
			factors, abs = append(factors, mb.einv), append(abs, rer)
		} else {
			factors, abs = append(factors, mb.a), append(abs, mb.absA)
		}
	}
	left := make([]*mat.VecDense, 2*k)
	left[0] = mat.NewVecDense(n, nil)
	for l := range n {
		left[0].SetVec(l, mb.c.At(i, l))
	}
	for q, f := range factors {
		left[q+1] = mat.NewVecDense(n, nil)
		left[q+1].MulVec(f.T(), left[q])
	}
	right := make([]*mat.VecDense, 2*k)
	right[2*k-1] = mat.NewVecDense(n, nil)
	for l := range n {
		right[2*k-1].SetVec(l, mb.b.At(l, j))
	}
	for q := 2*k - 2; q >= 0; q-- {
		right[q] = mat.NewVecDense(n, nil)
		right[q].MulVec(factors[q], right[q+1])
	}
	absVec := func(v *mat.VecDense) *mat.VecDense {
		out := mat.NewVecDense(v.Len(), nil)
		for l := range v.Len() {
			out.SetVec(l, math.Abs(v.AtVec(l)))
		}
		return out
	}
	sum := 0.0
	var tmp mat.VecDense
	for q := range factors {
		tmp.MulVec(abs[q], absVec(right[q+1]))
		sum += mat.Dot(absVec(left[q]), &tmp)
	}
	absC := mat.NewVecDense(n, nil)
	absB := mat.NewVecDense(n, nil)
	for l := range n {
		absC.SetVec(l, math.Abs(mb.c.At(i, l)))
		absB.SetVec(l, math.Abs(mb.b.At(l, j)))
	}
	sum += mat.Dot(absC, absVec(right[0])) + mat.Dot(absVec(left[2*k-1]), absB)
	return eps() * sum * (1 + gammaN((2*k+1)*n))
}
