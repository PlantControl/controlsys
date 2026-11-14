package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"sync"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/mat"
)

// poleResidueWeights are the 5-point binomial residue weights: with samples
// g_k = G(s + kδ), k = 1…5, r̂ = 5δ·Σ w_k g_k is the degree-4 extrapolation
// of x·G(s + x) to x = 0, the residue of the cluster at s.
var poleResidueWeights = [5]float64{1, -4, 6, -4, 1}

// poleExtrap3 and poleExtrap5 extrapolate G − its cluster term to s from
// the first three and all five samples.
var (
	poleExtrap3 = [3]float64{3, -3, 1}
	poleExtrap5 = [5]float64{5, -10, 10, -5, 1}
)

const (
	unitRoundoff   = 0x1p-53
	ddUnitRoundoff = 0x1p-104
)

// poleSchurData is the explicit delay-free realization (A, B, C, D) of sys
// with A in real Schur form A = Z·T·Zᵀ, row-major.
type poleSchurData struct {
	mu            sync.Mutex
	cluster       *poleCluster
	clusterOK     bool
	clusterSet    bool
	exactOnce     sync.Once
	dcGainOnce    sync.Once
	dcGainOK      bool
	exactGain     *mat.Dense
	exactOverflow bool
	exactSingular bool
	n, m, p       int
	a, t, z       []float64
	b, c, d       []float64
	normA         float64
	rowC, colB    []float64
	// dcCheck memoizes dcCheckValue.
	dcCheckOnce sync.Once
	dcCheck     []complex128
	dcCheckOK   bool
	// exactAt memoizes exactResponseAt per frequency point.
	exactAt map[frequencyPoint]exactPointResponse
}

var poleSchurCache struct {
	mu   sync.Mutex
	key  []float64
	dims [4]int
	data *poleSchurData
	ok   bool
}

// cachedPoleSchur returns poleSchurData of sys, factorized once per model:
// a model whose A, E, B, C, D and Dt match the last one reuses its Schur
// form. ok is false for an internal-delay or descriptor model, or where
// the factorization fails.
func cachedPoleSchur(sys *System) (*poleSchurData, bool) {
	if sys == nil || sys.internalDelayCount() > 0 {
		return nil, false
	}
	n, m, p := sys.Dims()
	if n == 0 {
		return nil, false
	}
	key := make([]float64, 0, n*n*2+n*m+p*n+p*m+1)
	add := func(d *mat.Dense) {
		if d == nil {
			key = append(key, math.NaN())
			return
		}
		raw := d.RawMatrix()
		for i := range raw.Rows {
			key = append(key, raw.Data[i*raw.Stride:i*raw.Stride+raw.Cols]...)
		}
	}
	if newDescriptorPolicy(sys).isDescriptor() {
		return nil, false
	}
	add(sys.A)
	add(sys.B)
	add(sys.C)
	add(sys.D)
	key = append(key, sys.Dt)
	dims := [4]int{n, m, p, 0}
	c := &poleSchurCache
	c.mu.Lock()
	defer c.mu.Unlock()
	if c.dims == dims && slices.EqualFunc(c.key, key, func(a, b float64) bool { return a == b || math.IsNaN(a) && math.IsNaN(b) }) {
		return c.data, c.ok
	}
	c.data, c.ok = newPoleSchurData(sys, n, m, p)
	c.key, c.dims = key, dims
	return c.data, c.ok
}

func newPoleSchurData(sys *System, n, m, p int) (*poleSchurData, bool) {
	ps := &poleSchurData{n: n, m: m, p: p}
	a, b := sys.A, sys.B
	ps.a = make([]float64, n*n)
	copyStrided(ps.a, n, a.RawMatrix().Data, a.RawMatrix().Stride, n, n)
	ps.b = make([]float64, n*m)
	copyStrided(ps.b, m, b.RawMatrix().Data, b.RawMatrix().Stride, n, m)
	ps.c = make([]float64, p*n)
	copyStrided(ps.c, n, sys.C.RawMatrix().Data, sys.C.RawMatrix().Stride, p, n)
	ps.d = make([]float64, p*m)
	copyStrided(ps.d, m, sys.D.RawMatrix().Data, sys.D.RawMatrix().Stride, p, m)
	var ok bool
	if ps.t, ps.z, ok = realSchur(mat.NewDense(n, n, slices.Clone(ps.a))); !ok {
		return nil, false
	}
	var na l2
	for _, v := range ps.a {
		na.add(v)
	}
	ps.normA = na.norm()
	ps.rowC = make([]float64, p)
	for i := range p {
		var nc l2
		for l := range n {
			nc.add(ps.c[i*n+l])
		}
		ps.rowC[i] = nc.norm()
	}
	ps.colB = make([]float64, m)
	for j := range m {
		var nb l2
		for l := range n {
			nb.add(ps.b[l*m+j])
		}
		ps.colB[j] = nb.norm()
	}
	return ps, true
}

// residueTerms is the structured residue test of certifiedPoleLimit
// (A2OX65) on five samples of every channel: |r̂ − R| ≤ Θ|R| + B_abs, R
// the residue at DC, that of the core of poleCluster. The rest, its grown
// members and regular part, is charged absolutely: with the Schur form
// reordered core first and separated by the Sylvester solution X = W of
// poleCluster, (zI − A)⁻¹ splits into the core part, bounded relatively,
// and the rest Q, bounded absolutely through
// ‖Q(zI − A)⁻¹Q‖ ≤ (1 + ‖X‖)²‖(z − T_rest)⁻¹‖; ρ is rho0. Each
// sample is modelled as a backward-stable solve, ‖E_k‖ ≤ γ₃ₙρ_g‖M_k‖ for
// M_k = z_kI − A and ρ_g the growth of its LU.
//
// For a single real pole λ, right and left vectors x, y:
// θ_k = c_k/|z_k − λ| + μ + ν_k with c_k = γρ_g|y|ᵀ(|Ã| + |z_k|I)|x| its
// rounding disk, μ = γ(|y|ᵀ|B|/|yᵀB| + |C||x|/|Cx|) and
// ν_k = ‖E_k‖‖QM_k⁻¹Q‖(‖y‖‖B‖/|yᵀB| + ‖C‖‖x‖/|Cx|), and
// Θ = Σ 5δ|w_k|/|z_k − λ|·θ_k/(1 − θ_k), valid where every θ_k < 1. A
// channel with Cx = 0 or yᵀB = 0 has no cluster term, and the rounding
// that could create one is bounded absolutely in B_abs. For a larger
// cluster block T_cl, θ_k = ‖(z_k − T_cl)⁻¹‖‖E_k‖, and its term is bounded
// absolutely by θ_k/(1 − θ_k) + 2γ of ‖Ĉ‖‖(z_k − T_cl)⁻¹‖‖B̂‖.
//
// B_abs = B_trunc + Σ 5δ|w_k|·E_reg,k with E_reg,k =
// γρ_g‖C‖‖B‖q_k²‖M_k‖ + 2γ‖C‖‖B‖q_k + γ|D| + 2u₅₃|g_k|, the last the
// rounding of every sample, double-double ones included, to complex128,
// q_k = ‖QM_k⁻¹Q‖, and
// B_trunc = 1920·M·(δ/ρ)⁴·δ, M = |D| + ‖Č‖‖B̌‖·sup over |z − s| = ρ/2 of
// ‖(z − T_reg)⁻¹‖ (Henrici), the regular part analytic within ρ of s.
//
// pole: r̂ > B_abs with Θ valid. finite: r̂ ≤ B_abs with Θ valid, the
// 3- and 5-point extrapolants agreeing within their bounds (rounding Σ|c|e_k,
// truncation 48M(δ/ρ)³ and 3840M(δ/ρ)⁵, and the leak of a residue up to
// (r̂ + B_abs)/(1 − Θ)), and the 3-point bound b₃ within tau of the
// extrapolant. Otherwise the channel is undecided.
type residueTerms struct {
	rhat, babs, theta []float64
	b3                []float64
	valid             []bool
	pole, finite      []bool
	q                 int
}

// structuredResidue computes residueTerms with unit roundoff u for the
// samples g at s + kδ of ps; ok is false where the cluster cannot be
// separated.
func structuredResidue(ps *poleSchurData, cl *poleCluster, s complex128, delta float64, g []complex128, u, tau float64) *residueTerms {
	n, m, p := ps.n, ps.m, ps.p
	k := p * m
	uc := 4 * u
	gamma := func(c int) float64 { return float64(c) * uc / (1 - float64(c)*uc) }
	g3n, gn := gamma(3*n), gamma(n+1)
	t, z, q, x, rho := cl.t, cl.z, cl.q0, cl.w, cl.rho0
	r := n - q
	kV := (1 + cl.normW) * (1 + cl.normW)
	bt := make([]float64, n*m)
	for i := range n {
		for j := range m {
			for l := range n {
				bt[i*m+j] += z[l*n+i] * ps.b[l*m+j]
			}
		}
	}
	cz := make([]float64, p*n)
	for i := range p {
		for j := range n {
			for l := range n {
				cz[i*n+j] += ps.c[i*n+l] * z[l*n+j]
			}
		}
	}
	bHat := make([]float64, q*m)
	for i := range q {
		for j := range m {
			v := bt[i*m+j]
			for l := range r {
				v -= x[i*r+l] * bt[(q+l)*m+j]
			}
			bHat[i*m+j] = v
		}
	}
	cCheck := make([]float64, p*r)
	for i := range p {
		for j := range r {
			v := cz[i*n+q+j]
			for l := range q {
				v += cz[i*n+l] * x[l*r+j]
			}
			cCheck[i*r+j] = v
		}
	}
	rowNorm := func(a []float64, cols, i int) float64 {
		var v l2
		for j := range cols {
			v.add(a[i*cols+j])
		}
		return v.norm()
	}
	colNorm := func(a []float64, rows, cols, j int) float64 {
		var v l2
		for i := range rows {
			v.add(a[i*cols+j])
		}
		return v.norm()
	}
	henrici := 0.0
	if r > 0 {
		regEig := schurEigenvaluesRaw(sliceBlock(t, n, q, r), r)
		rhoReg := math.Inf(1)
		for _, l := range regEig {
			rhoReg = min(rhoReg, cmplx.Abs(l-s))
		}
		var nacc l2
		for i := range r {
			for j := range r {
				if i != j {
					nacc.add(t[(q+i)*n+q+j])
				}
			}
		}
		nN := nacc.norm()
		d := max(rhoReg-rho/2, rho/2)
		term := 1 / d
		for range r {
			henrici += term
			term *= nN / d
			if term == 0 || math.IsInf(henrici, 0) {
				break
			}
		}
	}
	rt := &residueTerms{rhat: make([]float64, k), babs: make([]float64, k), theta: make([]float64, k),
		b3: make([]float64, k), valid: make([]bool, k), pole: make([]bool, k), finite: make([]bool, k), q: q}
	f3 := make([]complex128, k)
	f5 := make([]complex128, k)
	mBound := make([]float64, k)
	for j := range k {
		rt.valid[j] = true
		var rh complex128
		for kk, w := range poleResidueWeights {
			rh += complex(w, 0) * g[kk*k+j]
		}
		rt.rhat[j] = cmplx.Abs(complex(5*delta, 0) * rh)
		for kk, w := range poleExtrap3 {
			f3[j] += complex(w, 0) * g[kk*k+j]
		}
		for kk, w := range poleExtrap5 {
			f5[j] += complex(w, 0) * g[kk*k+j]
		}
		i, l := j/m, j%m
		mBound[j] = math.Abs(ps.d[j]) + rowNorm(cCheck, r, i)*colNorm(bt[q*m:], r, m, l)*henrici
		rt.babs[j] = 1920 * mBound[j] * math.Pow(delta/rho, 4) * delta
	}
	var lam complex128
	var xv, yv []float64
	cx := make([]float64, p)
	cxAbs := make([]float64, p)
	yb := make([]float64, m)
	ybAbs := make([]float64, m)
	var normX1, normY float64
	if q == 1 {
		var ax, ay l2
		lam = complex(t[0], 0)
		xv = make([]float64, n)
		yv = make([]float64, n)
		for i := range n {
			xv[i] = z[i*n]
			yv[i] = z[i*n]
			for l := range r {
				yv[i] -= z[i*n+q+l] * x[l]
			}
			ax.add(xv[i])
			ay.add(yv[i])
		}
		normX1, normY = ax.norm(), ay.norm()
		for i := range p {
			for l := range n {
				cx[i] += ps.c[i*n+l] * xv[l]
				cxAbs[i] += math.Abs(ps.c[i*n+l] * xv[l])
			}
		}
		for j := range m {
			for l := range n {
				yb[j] += yv[l] * ps.b[l*m+j]
				ybAbs[j] += math.Abs(yv[l] * ps.b[l*m+j])
			}
		}
	}
	errRel := make([]float64, 5*k)
	errAbs := make([]float64, 5*k)
	lu := make([]complex128, n*n)
	ipiv := make([]int, n)
	reg := make([]complex128, r*r)
	regInv := make([]complex128, r*r)
	clm := make([]complex128, q*q)
	clInv := make([]complex128, q*q)
	for kk := range 5 {
		zk := s + complex(float64(kk+1)*delta, 0)
		var maxM, maxU float64
		for i := range n {
			for l := range n {
				v := complex(-ps.a[i*n+l], 0)
				if i == l {
					v += zk
				}
				lu[i*n+l] = v
				maxM = max(maxM, cmplx.Abs(v))
			}
		}
		if !impl.Zgetrf(n, n, lu, n, ipiv) {
			return nil
		}
		for i := range n {
			for l := i; l < n; l++ {
				maxU = max(maxU, cmplx.Abs(lu[i*n+l]))
			}
		}
		rhoG := max(1, maxU/maxM)
		normM := ps.normA + cmplx.Abs(zk)
		ek := g3n * rhoG * normM
		qk := 0.0
		if r > 0 {
			for i := range r {
				for l := range r {
					v := complex(-t[(q+i)*n+q+l], 0)
					if i == l {
						v += zk
					}
					reg[i*r+l] = v
					regInv[i*r+l] = 0
				}
				regInv[i*r+i] = 1
			}
			if !impl.Zgetrf(r, r, reg, r, ipiv[:r]) {
				return nil
			}
			impl.Zgetrs(blas.NoTrans, r, r, reg, r, ipiv[:r], regInv, r)
			var nr l2
			for _, v := range regInv {
				nr.addC(v)
			}
			qk = kV * nr.norm()
		}
		var clNorm float64
		if q > 1 {
			for i := range q {
				for l := range q {
					v := complex(-t[i*n+l], 0)
					if i == l {
						v += zk
					}
					clm[i*q+l] = v
					clInv[i*q+l] = 0
				}
				clInv[i*q+i] = 1
			}
			if !impl.Zgetrf(q, q, clm, q, ipiv[:q]) {
				return nil
			}
			impl.Zgetrs(blas.NoTrans, q, q, clm, q, ipiv[:q], clInv, q)
			var nc l2
			for _, v := range clInv {
				nc.addC(v)
			}
			clNorm = nc.norm()
		}
		var ck, dist float64
		if q == 1 {
			dist = cmplx.Abs(zk - lam)
			for i := range n {
				var ax float64
				for l := range n {
					ax += math.Abs(ps.a[i*n+l]) * math.Abs(xv[l])
				}
				ax += cmplx.Abs(zk) * math.Abs(xv[i])
				ck += math.Abs(yv[i]) * ax
			}
			ck *= g3n * rhoG
		}
		wk := 5 * delta * math.Abs(poleResidueWeights[kk])
		for j := range k {
			i, l := j/m, j%m
			nc, nb := ps.rowC[i], ps.colB[l]
			abs := g3n*rhoG*nc*nb*qk*qk*normM + 2*gn*nc*nb*qk + gn*math.Abs(ps.d[j]) + 2*unitRoundoff*cmplx.Abs(g[kk*k+j])
			switch {
			case q == 1 && cx[i] != 0 && yb[l] != 0:
				mu := gn * (ybAbs[l]/math.Abs(yb[l]) + cxAbs[i]/math.Abs(cx[i]))
				nu := ek * qk * (normY*nb/math.Abs(yb[l]) + nc*normX1/math.Abs(cx[i]))
				th := ck/dist + mu + nu
				if !(th < 1) {
					rt.valid[j] = false
					continue
				}
				rt.theta[j] += wk / dist * th / (1 - th)
				errRel[kk*k+j] = th / (1 - th) / dist
			case q == 1:
				abs += (gn*(cxAbs[i]*math.Abs(yb[l])+math.Abs(cx[i])*ybAbs[l]) + ek*qk*(normY*nb*math.Abs(cx[i])+nc*normX1*math.Abs(yb[l]))) / dist
			case q > 1:
				th := clNorm * ek
				if !(th < 1) {
					rt.valid[j] = false
					continue
				}
				abs += (th/(1-th) + 2*gn) * rowNorm(cz, n, i) * clNorm * colNorm(bHat, q, m, l)
			}
			errAbs[kk*k+j] = abs
			rt.babs[j] += wk * abs
		}
	}
	for j := range k {
		if !rt.valid[j] || !(rt.theta[j] < 1) {
			rt.valid[j] = false
			continue
		}
		rMax := (rt.rhat[j] + rt.babs[j]) / (1 - rt.theta[j])
		if q == 0 {
			rMax = 0
		}
		var e3, e5 float64
		var leak3, leak5 complex128
		for kk := range 5 {
			e := errAbs[kk*k+j] + errRel[kk*k+j]*rMax
			zk := s + complex(float64(kk+1)*delta, 0)
			pole := zk - s
			if q == 1 {
				pole = zk - lam
			}
			if kk < 3 {
				e3 += math.Abs(poleExtrap3[kk]) * e
				leak3 += complex(poleExtrap3[kk], 0) / pole
			}
			e5 += math.Abs(poleExtrap5[kk]) * e
			leak5 += complex(poleExtrap5[kk], 0) / pole
		}
		b3 := e3 + 48*mBound[j]*math.Pow(delta/rho, 3) + rMax*cmplx.Abs(leak3)
		b5 := e5 + 3840*mBound[j]*math.Pow(delta/rho, 5) + rMax*cmplx.Abs(leak5)
		rt.b3[j] = b3
		if rt.rhat[j] > rt.babs[j] {
			rt.pole[j] = q > 0
		} else if cmplx.Abs(f5[j]-f3[j]) <= b3+b5 && b3 <= tau*cmplx.Abs(f3[j]) {
			rt.finite[j] = true
		}
	}
	return rt
}

// poleCluster is the cluster at DC of poleSchurData (A2OX65): the Schur
// form reordered as [core, grown, regular], the core the q0 eigenvalues
// at DC as growCluster selects them, the grown
// members added until the cluster's Sylvester separation from the
// regular part, T_cl·X − X·T_reg = −T₁₂, has ‖X‖_F = normZ ≤ 1, and W
// the Sylvester separation of the core from the rest, grown and regular,
// with ‖W‖_F = normW. rho0 is the distance from DC to the nearest
// eigenvalue outside the core, the scale of the residue at DC, and disk
// the largest finite rounding move of a core eigenvalue: the test
// samples at δ = poleStepFraction·rho0, so the grown members enter, with
// the regular part, as the absolute terms of the rest decoupled by W.
type poleCluster struct {
	t, z, w []float64
	q0, q   int
	normZ   float64
	normW   float64
	rho0    float64
	disk    float64
}

// exactPoleAtDC reports whether dc·I − A is exactly singular, the stored
// model having an eigenvalue exactly at dc; decided is false above
// exactDetMaxOrder.
func (ps *poleSchurData) exactPoleAtDC(dc complex128) (exact, decided bool) {
	if ps.n > exactDetMaxOrder {
		return false, false
	}
	_, _, singular := ps.exactDC(dc)
	return singular, true
}

// exactDC is the stored model's exact response D + C(dc·I − A)⁻¹B by one
// fraction-free solve (exactDCGainAtNonsingular), n ≤ exactDetMaxOrder,
// computed once per model; singular reports dc·I − A exactly singular.
func (ps *poleSchurData) exactDC(dc complex128) (gain *mat.Dense, overflow, singular bool) {
	ps.exactOnce.Do(func() {
		a := mat.NewDense(ps.n, ps.n, ps.a)
		b := mat.NewDense(ps.n, ps.m, ps.b)
		c := mat.NewDense(ps.p, ps.n, ps.c)
		d := mat.NewDense(ps.p, ps.m, ps.d)
		var ok bool
		ps.exactGain, ps.exactOverflow, ok = exactDCGainAtNonsingular(nil, a, b, c, d, real(dc))
		ps.exactSingular = !ok
	})
	return ps.exactGain, ps.exactOverflow, ps.exactSingular
}

// dcGainCertified reports whether DCGain of sys returns the exact
// response at DC with a nil error, computed once per model.
func (ps *poleSchurData) dcGainCertified(sys *System) bool {
	ps.dcGainOnce.Do(func() {
		g, err := sys.DCGain()
		ex, overflow, singular := ps.exactDC(dcPoint(sys))
		if err != nil || singular || overflow {
			return
		}
		gr, er := g.RawMatrix(), ex.RawMatrix()
		for i := range gr.Rows {
			for j := range gr.Cols {
				if gr.Data[i*gr.Stride+j] != er.Data[i*er.Stride+j] {
					return
				}
			}
		}
		ps.dcGainOK = true
	})
	return ps.dcGainOK
}

// resolvableFromDC reports whether an eigenvalue at distance d from dc can
// be told from dc by samples at dc + kδ, δ = poleStepFraction·d: δ is
// normal and moves dc by more than 64 ulps.
func resolvableFromDC(d float64, dc complex128) bool {
	step := poleStepFraction * d
	return step > 0x1p-1000 && step > 64*eps()*cmplx.Abs(dc)
}

// growCluster forms the cluster at dc, ok false where no eigenvalue lies
// within its own rounding move of dc. The core is the in-disk eigenvalues,
// except, where some can be told from dc by samples (resolvableFromDC)
// and dc·I − A is exactly nonsingular (exactPoleAtDC), only those that
// cannot: the residue AT dc is the stored model's, and its resolvable
// poles beside dc are analytic there, so its exact residue at dc is not
// claimed nonzero from their sum (the shear realizations, ±8.8e-9 inside
// a 1.1e-8 disk, exact G(0) = 447.6997). A computed multiple pole spread
// by rounding about an exact one at dc stays whole. Above exactDetMaxOrder
// such a split is undecided and ok is false. Then, in order of distance
// from dc the nearest regular eigenvalue while the actual Sylvester
// solution has ‖X‖_F > 1, up to the whole spectrum. A core that is the
// whole spectrum takes rho0 = max(|dc|, |λ|) (1 if that is 0), the
// spectral scale poleSpacing takes for rho there. ok is false where the
// reordering or a Sylvester separation fails.
func growCluster(ps *poleSchurData, dc complex128) (*poleCluster, bool) {
	n := ps.n
	pe, ok := newPoleEigen(mat.NewDense(n, n, slices.Clone(ps.a)))
	if !ok {
		return nil, false
	}
	schur := schurEigenvaluesRaw(ps.t, n)
	dist := make([]float64, n)
	for i, l := range schur {
		dist[i] = cmplx.Abs(l - dc)
	}
	radius, disk, core, resolvable := -1.0, 0.0, -1.0, false
	a := mat.NewDense(n, n, ps.a)
	for k, p := range pe.poles {
		yax, yx := pe.coupling(a, k, k)
		if move := eps() * yax / cmplx.Abs(yx); cmplx.Abs(p-dc) <= move {
			if !math.IsInf(move, 0) {
				disk = max(disk, move)
			}
			near := 0
			for i, l := range schur {
				if cmplx.Abs(l-p) < cmplx.Abs(schur[near]-p) {
					near = i
				}
			}
			radius = max(radius, dist[near])
			if resolvableFromDC(dist[near], dc) {
				resolvable = true
			} else {
				core = max(core, dist[near])
			}
		}
	}
	slices.Sort(dist)
	if radius < 0 {
		return nil, false
	}
	if resolvable {
		switch exact, decided := ps.exactPoleAtDC(dc); {
		case !decided:
			return nil, false
		case exact:
			core = radius
		}
	}
	nextAfter := func(cut float64) float64 {
		for _, d := range dist {
			if d > cut {
				return d
			}
		}
		return math.Inf(1)
	}
	within := func(cut float64) func(complex128) bool {
		return func(l complex128) bool { return cmplx.Abs(l-dc) <= cut || cmplx.Abs(cmplx.Conj(l)-dc) <= cut }
	}
	radius = core
	rho0 := nextAfter(core)
	if math.IsInf(rho0, 1) {
		rho0 = cmplx.Abs(dc)
		for _, l := range schur {
			rho0 = max(rho0, cmplx.Abs(l))
		}
		if rho0 == 0 {
			rho0 = 1
		}
	}
	for {
		t, z := slices.Clone(ps.t), slices.Clone(ps.z)
		q, err := orderSchurGroupFirst(t, z, n, within(radius))
		if err != nil {
			return nil, false
		}
		x, ok := clusterSylvester(t, n, q)
		var nz l2
		for _, v := range x {
			nz.add(v)
		}
		if q == n || ok && nz.norm() <= 1 {
			q0, err := orderSchurGroupFirst(t, z, n, within(core))
			if err != nil {
				return nil, false
			}
			w, ok := clusterSylvester(t, n, q0)
			if !ok {
				return nil, false
			}
			var nw l2
			for _, v := range w {
				nw.add(v)
			}
			return &poleCluster{t: t, z: z, w: w, q0: q0, q: q, normZ: nz.norm(), normW: nw.norm(), rho0: rho0, disk: disk}, true
		}
		next := nextAfter(radius)
		if math.IsInf(next, 1) {
			return nil, false
		}
		radius = next
	}
}

// dcCluster is growCluster at dc, formed once per model.
func (ps *poleSchurData) dcCluster(dc complex128) (*poleCluster, bool) {
	ps.mu.Lock()
	defer ps.mu.Unlock()
	if !ps.clusterSet {
		ps.cluster, ps.clusterOK = growCluster(ps, dc)
		ps.clusterSet = true
	}
	return ps.cluster, ps.clusterOK
}

// sliceBlock returns the r×r trailing block of the n×n t from row and column q.
func sliceBlock(t []float64, n, q, r int) []float64 {
	out := make([]float64, r*r)
	for i := range r {
		copy(out[i*r:(i+1)*r], t[(q+i)*n+q:(q+i)*n+q+r])
	}
	return out
}

// clusterSylvester solves T_cl·X − X·T_reg = −T12 for the q×(n−q) X of the
// quasi-triangular t partitioned at q, by back-substitution over the
// diagonal blocks of T_reg, each a small Kronecker solve; ok is false
// where a block is singular to working precision.
func clusterSylvester(t []float64, n, q int) ([]float64, bool) {
	r := n - q
	x := make([]float64, q*r)
	if q == 0 || r == 0 {
		return x, true
	}
	for j0 := 0; j0 < r; {
		bs := 1
		if j0+1 < r && t[(q+j0+1)*n+q+j0] != 0 {
			bs = 2
		}
		rhs := make([]float64, q*bs)
		for i := range q {
			for c := range bs {
				v := -t[i*n+q+j0+c]
				for l := range j0 {
					v += x[i*r+l] * t[(q+l)*n+q+j0+c]
				}
				rhs[i*bs+c] = v
			}
		}
		dim := q * bs
		kr := mat.NewDense(dim, dim, nil)
		for i := range q {
			for c := range bs {
				row := i*bs + c
				for l := range q {
					kr.Set(row, l*bs+c, kr.At(row, l*bs+c)+t[i*n+l])
				}
				for c2 := range bs {
					kr.Set(row, i*bs+c2, kr.At(row, i*bs+c2)-t[(q+j0+c2)*n+q+j0+c])
				}
			}
		}
		var lu mat.LU
		lu.Factorize(kr)
		if !(lu.Cond()*eps() < 1e-2) {
			return nil, false
		}
		sol := mat.NewVecDense(dim, nil)
		if lu.SolveVecTo(sol, false, mat.NewVecDense(dim, rhs)) != nil {
			return nil, false
		}
		for i := range q {
			for c := range bs {
				x[i*r+j0+c] = sol.AtVec(i*bs + c)
			}
		}
		j0 += bs
	}
	for _, v := range x {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, false
		}
	}
	return x, true
}

// l2 accumulates a Euclidean norm without overflow or underflow, as LAPACK
// dlassq.
type l2 struct{ scale, ssq float64 }

func (a *l2) add(v float64) {
	v = math.Abs(v)
	if v == 0 {
		return
	}
	if a.scale < v {
		a.ssq = 1 + a.ssq*(a.scale/v)*(a.scale/v)
		a.scale = v
	} else {
		a.ssq += (v / a.scale) * (v / a.scale)
	}
}

func (a *l2) addC(v complex128) {
	a.add(real(v))
	a.add(imag(v))
}

func (a l2) norm() float64 { return a.scale * math.Sqrt(a.ssq) }
