package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

type zpkVerdict int

const (
	zpkOpen zpkVerdict = iota
	zpkVerified
	zpkRefuted
)

type zpkPointVerdict int

const (
	zpkPointOpen zpkPointVerdict = iota
	zpkPointVerified
	// zpkPointUndetermined is a point where an entry fails against
	// tol·scale and the data do not determine its terms (δ_ap ≥ T̂).
	zpkPointUndetermined
	zpkPointRefuted
	// zpkPointConceded is a structured point where double-double GEPP
	// pivots exactly zero: an exact pole of the stored matrices.
	zpkPointConceded
)

// zpkRowCheck verifies candidate zero-pole-gain forms of one output row
// against ref, the row's structurally minimal part (Sminreal), an explicit
// single-output model of order n_r: the stored model's response restricted
// exactly. evals are the eigenvalues of ref's A (no eigenvectors); ρ₀ =
// n_r·ε·‖Â‖_F, Â the power-of-two balanced A of boundedResponse; the
// rounding bounds β (eigenRoundingBounds) and the per-pole resolutions
// ρ_k = min(ρ₀, 2n_r·β_k) of the poles resolved from zero (modelPoles) are
// formed only when a point lies within n_r·ε·|s| + ρ₀ of an eigenvalue, a
// shared-pole window is needed or a spectral gap is found.
type zpkRowCheck struct {
	ref      *System
	n, m     int
	discrete bool
	tol      float64
	evals    []complex128
	rho0     float64
	mp       modelPoles
	betaDone bool
	betaOK   bool
	gapDone  bool
	cutoff   float64
	gapOK    bool
	pf       zpkPointFactors
}

func newZPKRowCheck(ref *System) (*zpkRowCheck, bool) {
	n, m, _ := ref.Dims()
	c := &zpkRowCheck{ref: ref, n: n, m: m, discrete: ref.Dt != 0, tol: math.Sqrt(eps())}
	if n == 0 {
		return c, false
	}
	var eig mat.Eigen
	if !eig.Factorize(ref.A, mat.EigenNone) {
		return c, false
	}
	c.evals = eig.Values(nil)
	norm := 0.0
	for _, a := range newBalancedRealization(ref, n, m, 1).a {
		norm = math.Hypot(norm, a)
	}
	c.rho0 = float64(n) * eps() * norm
	return c, true
}

// ensureBeta forms the rounding bounds and resolutions of evals.
func (c *zpkRowCheck) ensureBeta() bool {
	if c.betaDone {
		return c.betaOK
	}
	c.betaDone = true
	ev, beta, ok := eigenRoundingBounds(c.ref.A)
	if !ok {
		return false
	}
	nk := len(c.evals)
	mp := modelPoles{evals: c.evals, beta: make([]float64, nk), resolved: make([]bool, nk), rho: make([]float64, nk), rho0: c.rho0}
	for k, e := range c.evals {
		if j := zpkNearest(ev, e); j >= 0 {
			mp.beta[k] = beta[j]
		}
		mp.resolved[k] = e != 0 && mp.beta[k] < cmplx.Abs(e)
		mp.rho[k] = c.rho0
		if mp.resolved[k] {
			mp.rho[k] = math.Min(c.rho0, 2*float64(c.n)*mp.beta[k])
		}
	}
	c.mp, c.betaOK = mp, true
	return true
}

// dropped reports whether s lies within n_r·ε·|s| + ρ_k of eigenvalue λ_k,
// where sI − A is singular to the precision the solve reaches (axisProbes);
// ok is false when the bounds it needs do not converge.
func (c *zpkRowCheck) dropped(s complex128) (drop, ok bool) {
	rs := float64(c.n) * eps() * cmplx.Abs(s)
	near := false
	for _, e := range c.evals {
		if cmplx.Abs(s-e) <= rs+c.rho0 {
			near = true
			break
		}
	}
	if !near {
		return false, true
	}
	if !c.ensureBeta() {
		return false, false
	}
	for k, e := range c.evals {
		if cmplx.Abs(s-e) <= rs+c.mp.rho[k] {
			return true, true
		}
	}
	return false, true
}

// inWindow reports whether s lies in a shared-pole window of the
// candidate's poles (zpkSharedPoleWindows), false when the bounds do not
// converge.
func (c *zpkRowCheck) inWindow(cand *zpkCandidate, s complex128) bool {
	if cand.win == nil {
		if !c.ensureBeta() {
			cand.win = &poleWindows{}
			return false
		}
		w := zpkSharedPoleWindows(c.mp, cand.poles, c.tol)
		cand.win = &w
	}
	return cand.win.contain(s)
}

// gap returns the cutoff of a resolved spectral gap of ref (resolvedGap),
// 0 when there is none or the model is discrete; ok is false when the
// bounds it needs do not converge.
func (c *zpkRowCheck) gap() (float64, bool) {
	if c.gapDone {
		return c.cutoff, c.gapOK
	}
	c.gapDone, c.gapOK = true, true
	if c.discrete || c.n < 2 {
		return 0, true
	}
	if _, ok := topGap(c.evals, nil); !ok {
		return 0, true
	}
	if !c.ensureBeta() {
		c.gapOK = false
		return 0, false
	}
	if cutoff, ok := topGap(c.evals, c.mp.resolved); ok {
		c.cutoff = cutoff
	}
	return c.cutoff, true
}

// check verifies cand against ref. The contract, per entry G_j − Z_j of the
// row at every axis point s where the stored realization determines its
// terms in floating point, outside the shared-pole windows of its poles,
// is |G_j(s) − Z_j(s)| ≤ tol·max(scale, T_j), tol = √ε, scale the largest
// |G_j|, |Z_j| of the row and T_j = Σ_l |C_l||X_lj(s)| + |D_j| the entry's
// terms, X = (sI − A)⁻¹B: the terms exceed |G_j| only where the response
// cancels, below which no roots computed in floating point place a zero.
//
// At a point, with computed Ĝ (boundedResponse plus D, its addition
// rounded by γ₁·|Ĝ|) and Ẑ (zpkEvalBounded), an entry verifies when
// |Ĝ − Ẑ| + δ_G + δ_Z ≤ tol·max(scale, T̂ − δ_T), δ the verification
// bounds and δ_T the bound on |T̂ − T| (zpkPointFactors), and refutes cand
// when |Ĝ − Ẑ| − δ_G,comp − δ_Z > tol·max(scale, T̂ + δ_T) outside a
// window: the exact deviation is then within, or beyond, the contract. An
// entry with δ_T ≥ T̂ has only scale as a lower bound of its contract:
// failing against it, the point is undetermined, neither verifying nor
// refuting. An entry left open is judged again with the compensated bounds
// (sharpenAt), refining Û only where η < 1 (zpkPointFactors), and the
// terms bound of the refined Û is used with that of the plain one.
//
// The structured points are the axis points beside every root of cand and
// every eigenvalue of ref (eigenAxisProbes), the static point (s = 0, z =
// 1) and, for a discrete model, z = −1. Each must verify or be conceded: a
// point within n_r·ε·|s| + ρ_k of eigenvalue λ_k (dropped), an undetermined
// point, or an open one inside a shared-pole window. A structured point
// where double GEPP meets an exactly zero pivot is judged in double-double
// (zpkDDPoint), and conceded only where that pivots exactly zero too, at
// an exact pole of the stored matrices. The verified points,
// conjugate-weighted (probeWeight), must give at least n_r + d + 1
// interpolation conditions, d the largest root count of the row's
// channels: G_j − Z_j has a numerator of degree at most n_r + d, so
// agreement at n_r + d + 1 points fixes it in exact arithmetic, but bounds
// it nowhere between them. While short of the count, up to three rounds of
// zpkFillProbes add a geometric fill of ⌈s/2⌉·2^r points in round r, s the
// shortfall after the structured points. Across a resolved spectral gap of
// a continuous ref (gap) each side needs its own count, by
// frequencyAxisCutAcrossGap's rule, with the eigenvalues of ref and the
// poles of cand in the roles of the model's and the cut's.
//
// Every structured point double leaves undetermined is judged in
// double-double before the row is accepted: a refutation there drops cand,
// and one still undetermined is open, so no structured point is conceded
// for undetermined terms. A row still short or open is judged again in
// double-double at its remaining open and undetermined points, structured
// and fill.
func (c *zpkRowCheck) check(cand *zpkCandidate) zpkVerdict {
	var roots, zeros, poles []complex128
	d := 0
	for j := range c.m {
		roots = append(roots, cand.zeros[j]...)
		roots = append(roots, cand.poles[j]...)
		zeros = append(zeros, cand.zeros[j]...)
		d = max(d, len(cand.poles[j]), len(cand.zeros[j]))
		if len(cand.poles[j]) > len(poles) {
			poles = cand.poles[j]
		}
	}
	structured := eigenAxisProbes(roots, c.discrete)
	structured = append(structured, eigenAxisProbes(c.evals, c.discrete)...)
	if c.discrete {
		structured = append(structured, 1, -1)
	} else {
		structured = append(structured, 0)
	}
	pts, ok := c.kept(nil, structured)
	if !ok {
		return c.done(cand, zpkOpen, nil, nil)
	}
	nStruct := len(pts)
	verdicts, refuted := c.judge(cand, pts, true)
	if refuted {
		return c.done(cand, zpkRefuted, pts, verdicts)
	}
	cutoff, ok := c.gap()
	if !ok {
		return c.done(cand, zpkOpen, pts, verdicts)
	}
	need, needSlow, needFast, slowScale := c.n+d+1, 1, 1, 0.0
	sided := cutoff > 0
	if sided {
		for _, set := range [][]complex128{c.evals, poles} {
			for _, ev := range set {
				if a := cmplx.Abs(ev); a >= cutoff {
					needFast++
				} else {
					needSlow++
					slowScale = math.Max(slowScale, a)
				}
			}
		}
	}
	var verified, slow, fast int
	tally := func() {
		verified, slow, fast = 0, 0, 0
		for k, v := range verdicts {
			if v != zpkPointVerified {
				continue
			}
			w := probeWeight(pts[k])
			verified += w
			switch a := cmplx.Abs(pts[k]); {
			case !sided:
			case a >= cutoff:
				fast += w
			case a <= slowScale:
				slow += w
			}
		}
	}
	short := func() int {
		s := need - verified
		if sided {
			s = max(s, needSlow-slow, needFast-fast)
		}
		return s
	}
	structuredOK := func(concedeUndetermined bool) bool {
		for k, v := range verdicts[:nStruct] {
			switch {
			case v == zpkPointVerified || v == zpkPointConceded:
			case v == zpkPointUndetermined && concedeUndetermined:
			case v == zpkPointOpen && c.inWindow(cand, pts[k]):
			default:
				return false
			}
		}
		return true
	}
	tally()
	base := (short() + 1) / 2
	for round := 0; round < 3 && short() > 0; round++ {
		fill, ok := c.kept(pts, zpkFillProbes(c.evals, zeros, c.discrete, base<<round))
		if !ok {
			return c.done(cand, zpkOpen, pts, verdicts)
		}
		v, refuted := c.judge(cand, fill[len(pts):], false)
		if refuted {
			return c.done(cand, zpkRefuted, fill, append(verdicts, v...))
		}
		pts = fill
		verdicts = append(verdicts, v...)
		tally()
	}
	if zpkSkipDD {
		if short() <= 0 && structuredOK(true) {
			return c.done(cand, zpkVerified, pts, verdicts)
		}
		return c.done(cand, zpkOpen, pts, verdicts)
	}
	dd := make([]bool, len(pts))
	rowLevel := false
	redo := func(k int) bool {
		dd[k] = true
		if zpkDDHook != nil {
			zpkDDHook(rowLevel)
		}
		v, ok := c.ddPoint(cand, pts[k], k < nStruct)
		if v == zpkPointRefuted {
			verdicts[k] = v
			return false
		}
		if ok {
			verdicts[k] = v
		}
		return true
	}
	for k, v := range verdicts[:nStruct] {
		if v == zpkPointUndetermined && !redo(k) {
			return c.done(cand, zpkRefuted, pts, verdicts)
		}
	}
	tally()
	if short() <= 0 && structuredOK(false) {
		return c.done(cand, zpkVerified, pts, verdicts)
	}
	rowLevel = true
	for k, v := range verdicts {
		if !dd[k] && (v == zpkPointOpen || v == zpkPointUndetermined) && !redo(k) {
			return c.done(cand, zpkRefuted, pts, verdicts)
		}
	}
	tally()
	if short() <= 0 && structuredOK(false) {
		return c.done(cand, zpkVerified, pts, verdicts)
	}
	return c.done(cand, zpkOpen, pts, verdicts)
}

// done reports the verdict of cand with its points to zpkCheckHook.
func (c *zpkRowCheck) done(cand *zpkCandidate, v zpkVerdict, pts []complex128, verdicts []zpkPointVerdict) zpkVerdict {
	if zpkCheckHook != nil {
		zpkCheckHook(c.ref, cand, v, pts, verdicts)
	}
	return v
}

// zpkCheckHook is a test-only seam, nil outside tests: it sees the verdict
// of every candidate check judges, with its points and their verdicts so
// far.
var zpkCheckHook func(ref *System, cand *zpkCandidate, v zpkVerdict, pts []complex128, verdicts []zpkPointVerdict)

// zpkSkipDD is a test-only seam, false outside tests: true leaves out the
// double-double stage of check.
var zpkSkipDD bool

// kept appends to pts the points of extra not already in it and not
// dropped; ok is false when the drop rule's bounds do not converge.
func (c *zpkRowCheck) kept(pts, extra []complex128) ([]complex128, bool) {
	out := pts
	for _, s := range extra {
		if slices.Contains(out, s) {
			continue
		}
		drop, ok := c.dropped(s)
		if !ok {
			return nil, false
		}
		if !drop {
			out = append(out, s)
		}
	}
	return out, true
}

// zpkEntryBounds are the values check judges an entry by: the computed
// response Ĝ (with D), its verification and computational bounds, and the
// lower and upper bounds lo ≤ T ≤ hi on its terms.
type zpkEntryBounds struct {
	g          complex128
	ver, comp  float64
	lo, hi     float64
	determined bool
}

// zpkJudgeEntries classifies a point from its entry bounds e and the
// candidate's values zv with their bounds zb (check).
func (c *zpkRowCheck) zpkJudgeEntries(cand *zpkCandidate, s complex128, e []zpkEntryBounds, zv []complex128, zb []float64) zpkPointVerdict {
	scale := 0.0
	for j := range e {
		scale = math.Max(scale, math.Max(cmplx.Abs(e[j].g), cmplx.Abs(zv[j])))
	}
	ts := c.tol * scale
	v := zpkPointVerified
	for j, x := range e {
		gr := cmplx.Abs(x.g) * gammaN(1)
		diff := cmplx.Abs(x.g - zv[j])
		lhs := diff + x.ver + gr + zb[j]
		if lhs <= ts {
			continue
		}
		if !x.determined || !(x.lo > 0) {
			if v == zpkPointVerified {
				v = zpkPointUndetermined
			}
			continue
		}
		if lhs <= c.tol*math.Max(scale, x.lo) {
			continue
		}
		if diff-x.comp-gr-zb[j] > c.tol*math.Max(scale, x.hi) && !c.inWindow(cand, s) {
			return zpkPointRefuted
		}
		v = zpkPointOpen
	}
	return v
}

// judge classifies every point of pts for cand in double (check);
// refuted reports a refuting point, at which judging stops.
func (c *zpkRowCheck) judge(cand *zpkCandidate, pts []complex128, structured bool) ([]zpkPointVerdict, bool) {
	out := make([]zpkPointVerdict, len(pts))
	if len(pts) == 0 {
		return out, false
	}
	full := newBoundedResponse(c.ref, pts)
	defer full.release()
	m, n := c.m, c.n
	zv := make([]complex128, m)
	zb := make([]float64, m)
	e := make([]zpkEntryBounds, m)
	dRaw := c.ref.D.RawMatrix()
	gc := zpkGEPPConstant(n, eps())
	scaleAP := math.Max(1, gc/gammaN(18*n))
	for k, s := range pts {
		finite := true
		for j := range m {
			var ok bool
			zv[j], zb[j], ok = zpkEvalBounded(cand.zeros[j], cand.poles[j], cand.gains[j], s)
			finite = finite && ok
		}
		if !finite {
			continue
		}
		pf := &c.pf
		pivotZero := func() ([]zpkPointVerdict, bool, bool) {
			out[k] = zpkPointUndetermined
			if structured && !zpkSkipDD {
				v, ok := c.ddPoint(cand, s, true)
				if v == zpkPointRefuted {
					out[k] = v
					return out, true, true
				}
				if ok {
					out[k] = v
				}
			}
			return nil, false, false
		}
		if !full.ok[k] {
			if r, refuted, stop := pivotZero(); stop {
				return r, refuted
			}
			continue
		}
		f := 1.0
		o := k * m
		build := func(lo, hi []float64) {
			for j := range m {
				dj := dRaw.Data[j]
				e[j] = zpkEntryBounds{
					g:    full.g[o+j] + complex(dj, 0),
					ver:  f * (full.comp[o+j] + scaleAP*(full.ver[o+j]-full.comp[o+j])),
					comp: f * full.comp[o+j],
				}
				if lo != nil {
					e[j].lo, e[j].hi, e[j].determined = lo[j]+math.Abs(dj), hi[j]+math.Abs(dj), true
				}
			}
		}
		build(nil, nil)
		v := c.zpkJudgeEntries(cand, s, e, zv, zb)
		if v != zpkPointVerified {
			if !pf.factor(&full.ev.br, s) {
				if r, refuted, stop := pivotZero(); stop {
					return r, refuted
				}
				continue
			}
			if !(pf.eta < 1) {
				out[k] = zpkPointUndetermined
				continue
			}
			f = 1 / (1 - pf.eta)
			lo, hi := pf.termsBounds(&full.ev.br, full.ev.u[k*n*m:(k+1)*n*m], full.terms[o:o+m], f)
			build(lo, hi)
			v = c.zpkJudgeEntries(cand, s, e, zv, zb)
			if v == zpkPointOpen && !full.sharp[k] {
				full.sharpenAt(k, true)
				lo1, hi1 := pf.refinedTermsBounds(full.ev, s, full.ev.u[k*n*m:(k+1)*n*m], full.terms[o:o+m], f)
				for j := range m {
					lo[j], hi[j] = math.Max(lo[j], lo1[j]), math.Min(hi[j], hi1[j])
				}
				build(lo, hi)
				v = c.zpkJudgeEntries(cand, s, e, zv, zb)
			}
		}
		out[k] = v
		if v == zpkPointRefuted {
			return out, true
		}
	}
	return out, false
}

// zpkGEPPConstant is the constant γ_{3n+20}, at unit roundoff u, of the
// backward error of complex GEPP and the two triangular solves as
// cgeppFactor and cgeppSolve form them: (M + ΔM)x̂ = b with |ΔM| ≤
// γ·Pᵀ|L_f||U_f|. Following Higham (Accuracy and Stability, Thms 9.3 and
// 9.4) with complex arithmetic: a complex sum errs by u componentwise, a
// product by √2·γ₂ ≤ γ₃ relative (Lemma 3.5), and crecip, the reciprocal
// the multipliers and the back substitution use, by γ₄. An entry of U is
// formed by at most n − 1 updates of a product and a difference, γ_{n+2};
// a multiplier adds its reciprocal and product, γ₇, so LU has γ_{n+9};
// the forward solve with unit L has γ_{n+2}, the back solve γ_{n+9}; and
// (L + ΔL)(U + ΔU) = M + ΔM_LU + ΔL·U + L·ΔU + ΔL·ΔU gives γ_{3n+20}.
func zpkGEPPConstant(n int, u float64) float64 {
	k := float64(3*n+20) * u
	return k / (1 - k)
}

// zpkPointFactors are the GEPP factors PM̂ = L_fU_f of M̂ = sI − Â at one
// point (Â the balanced A of boundedResponse), W = U_f⁻¹L_f⁻¹ = M̂⁻¹Pᵀ
// computed, and η = γ·‖|W|‖_∞·‖|L_f||U_f|‖_∞, γ zpkGEPPConstant: where
// η < 1, M = M̂ with the factorization's backward error ΔM, |ΔM| ≤
// γ·Pᵀ|L_f||U_f|, has |M⁻¹| ≤ |W|P/(1 − η) to first order in W's own
// rounding, so every bound formed with the computed inverse — comp, the
// a-priori term of ver and the terms bounds — carries 1/(1 − η), and only
// there is the refined Û bounded. They are formed only at the points the
// plain bounds of boundedResponse, first order as documented there, leave
// unverified.
type zpkPointFactors struct {
	n            int
	x, w         []complex128
	piv          []int
	absLU        []float64
	v1, v2, v3   []float64
	eta          float64
	uh, ul       []float64
	absU, au, rb []float64
	r            []complex128
}

// factor forms the factors, W and η at s; false at an exactly zero pivot.
func (pf *zpkPointFactors) factor(br *balancedRealization, s complex128) bool {
	n := br.n
	pf.n = n
	pf.x, pf.w = zeroed(pf.x, n*n), zeroed(pf.w, n*n)
	pf.piv, pf.absLU = zeroed(pf.piv, n), zeroed(pf.absLU, n*n)
	pf.v1, pf.v2, pf.v3 = zeroed(pf.v1, n), zeroed(pf.v2, n), zeroed(pf.v3, n)
	x, w := pf.x, pf.w
	for i, a := range br.a {
		x[i] = complex(-a, 0)
	}
	for i := range n {
		x[i*n+i] += s
	}
	if !cgeppFactor(x, n, pf.piv) {
		return false
	}
	for col := range n {
		for i := col; i < n; i++ {
			v := complex(0, 0)
			if i == col {
				v = 1
			}
			for l := col; l < i; l++ {
				v -= x[i*n+l] * w[l*n+col]
			}
			w[i*n+col] = v
		}
		for i := n - 1; i >= 0; i-- {
			v := w[i*n+col]
			for l := i + 1; l < n; l++ {
				v -= x[i*n+l] * w[l*n+col]
			}
			w[i*n+col] = v / x[i*n+i]
		}
	}
	nw, nlu := 0.0, 0.0
	for i := range n {
		rw, rlu := 0.0, 0.0
		for j := range n {
			rw += cabs(w[i*n+j])
			t := 0.0
			for l := 0; l <= min(i, j); l++ {
				li := 1.0
				if l != i {
					li = cabs(x[i*n+l])
				}
				t += li * cabs(x[l*n+j])
			}
			pf.absLU[i*n+j] = t
			rlu += t
		}
		nw, nlu = math.Max(nw, rw), math.Max(nlu, rlu)
	}
	pf.eta = zpkGEPPConstant(n, eps()) * nw * nlu
	if math.IsNaN(pf.eta) {
		pf.eta = math.Inf(1)
	}
	return true
}

// inverseTerms returns Σ_l |ĉ_l|·(|W|v)_l for the single-output br.
func (pf *zpkPointFactors) inverseTerms(br *balancedRealization, v []float64) float64 {
	n := pf.n
	t := 0.0
	for l := range n {
		wl := 0.0
		for q := range n {
			wl += cabs(pf.w[l*n+q]) * v[q]
		}
		t += math.Abs(br.c[l]) * wl
	}
	return t
}

// termsBounds returns, per input j, lower and upper bounds T̂_j ∓ δ_j on the
// exact terms T_j = Σ_l |C_l||U_lj| for the unrefined GEPP solution Û (u)
// with computed terms T̂ (terms): δ = f·γ·|Ĉ||W|(|L_f||U_f||Û|) + γ_{n+3}·T̂,
// γ zpkGEPPConstant, f = 1/(1 − η). The backward error gives |U − Û| ≤
// |M⁻¹||ΔM||Û| ≤ f·γ·|W||L_f||U_f||Û|, |M⁻¹|Pᵀ = |W| to first order; the
// second term bounds the rounding of forming T̂ from Û: n products and sums
// and the complex moduli.
func (pf *zpkPointFactors) termsBounds(br *balancedRealization, u []complex128, terms []float64, f float64) (lo, hi []float64) {
	n, m := pf.n, br.m
	lo, hi = make([]float64, m), make([]float64, m)
	g, gf := zpkGEPPConstant(n, eps()), gammaN(n+3)
	for j := range m {
		for i := range n {
			pf.v1[i] = 0
			for l := range n {
				pf.v1[i] += pf.absLU[i*n+l] * cabs(u[l*m+j])
			}
		}
		d := f*g*pf.inverseTerms(br, pf.v1) + gf*terms[j]
		lo[j], hi[j] = terms[j]-d, terms[j]+d
	}
	return lo, hi
}

// refinedTermsBounds is termsBounds for the refined û of sharpen: U − û =
// M⁻¹r exactly, r the residual of û, bounded componentwise by the
// compensated R of ev.residual, so δ = f·|Ĉ||W|(P·R) + γ_{n+3}·T̂.
func (pf *zpkPointFactors) refinedTermsBounds(ev *responseEvaluator, s complex128, u []complex128, terms []float64, f float64) (lo, hi []float64) {
	n, m := pf.n, ev.br.m
	nm := n * m
	pf.uh, pf.ul = zeroed(pf.uh, 2*nm), zeroed(pf.ul, 2*nm)
	pf.absU, pf.au, pf.rb = zeroed(pf.absU, nm), zeroed(pf.au, nm), zeroed(pf.rb, nm)
	pf.r = zeroed(pf.r, nm)
	if ev.scratch == nil {
		ev.prepareSharpen()
	}
	ev.residual(s, u, pf.uh, pf.ul, pf.absU, pf.au, pf.r, pf.rb, m)
	lo, hi = make([]float64, m), make([]float64, m)
	gf := gammaN(n + 3)
	for j := range m {
		for i := range n {
			pf.v1[i] = pf.rb[i*m+j]
		}
		for k := range n {
			if p := pf.piv[k]; p != k {
				pf.v1[k], pf.v1[p] = pf.v1[p], pf.v1[k]
			}
		}
		d := f*pf.inverseTerms(&ev.br, pf.v1) + gf*terms[j]
		lo[j], hi[j] = terms[j]-d, terms[j]+d
	}
	return lo, hi
}

// ddPoint judges cand at s in double-double: GEPP of sI − Â in complex
// double-double with the bounds of judge at unit roundoff ddOpErr, the
// documented error of one double-double operation (zpkDDResponse). ok is
// false where the double-double factorization fails to bound the point
// (η ≥ 1); a point where it pivots exactly zero, an exact pole of the
// stored matrices, is conceded. At a structured point double-double leaves
// open or undetermined with a verification bound above tol·max(scale,
// T̂ − δ_T), as beside a multi-fold zero, the response is evaluated exactly
// (zpkExactResponse) and judged by that; above its cap the point stays as
// double-double left it.
func (c *zpkRowCheck) ddPoint(cand *zpkCandidate, s complex128, structured bool) (zpkPointVerdict, bool) {
	m := c.m
	zv := make([]complex128, m)
	zb := make([]float64, m)
	for j := range m {
		var ok bool
		zv[j], zb[j], ok = zpkEvalBounded(cand.zeros[j], cand.poles[j], cand.gains[j], s)
		if !ok {
			return zpkPointOpen, false
		}
	}
	br := newBalancedRealization(c.ref, c.n, m, 1)
	e, zero, ok := zpkDDResponse(&br, s)
	if zero {
		return zpkPointConceded, true
	}
	if !ok {
		return zpkPointOpen, false
	}
	c.addD(e)
	v := c.zpkJudgeEntries(cand, s, e, zv, zb)
	if !structured || (v != zpkPointOpen && v != zpkPointUndetermined) || !c.unresolved(e, zv) {
		return v, true
	}
	if zpkExactHook != nil {
		zpkExactHook()
	}
	x, zero, ok := zpkExactResponse(c.ref, s)
	if zero {
		return zpkPointConceded, true
	}
	if !ok {
		if zpkExactCapHook != nil {
			zpkExactCapHook()
		}
		return v, true
	}
	c.addD(x)
	return c.zpkJudgeEntries(cand, s, x, zv, zb), true
}

// addD adds the feedthrough D of ref to the entry bounds e.
func (c *zpkRowCheck) addD(e []zpkEntryBounds) {
	dRaw := c.ref.D.RawMatrix()
	for j := range e {
		dj := dRaw.Data[j]
		e[j].g += complex(dj, 0)
		e[j].lo += math.Abs(dj)
		e[j].hi += math.Abs(dj)
	}
}

// unresolved reports whether some entry's verification bound exceeds
// tol·max(scale, lo), so that no candidate verifies against it.
func (c *zpkRowCheck) unresolved(e []zpkEntryBounds, zv []complex128) bool {
	scale := 0.0
	for j := range e {
		scale = math.Max(scale, math.Max(cmplx.Abs(e[j].g), cmplx.Abs(zv[j])))
	}
	for _, x := range e {
		if x.ver > c.tol*math.Max(scale, x.lo) {
			return true
		}
	}
	return false
}

// zpkDDHook is a test-only seam, nil outside tests: it is called at every
// double-double judgement check makes after the fill, rowLevel false for
// the undetermined structured points, true for the row-level stage.
var zpkDDHook func(rowLevel bool)

// zpkExactHook is a test-only seam, nil outside tests: it is called at
// every exact evaluation (zpkExactResponse).
var zpkExactHook func()

// zpkExactCapHook is a test-only seam, nil outside tests: it is called at
// every structured point zpkExactResponse declines for its cap.
var zpkExactCapHook func()

// zpkExactCapBits caps the size of the rationals zpkExactResponse forms.
const zpkExactCapBits = 1 << 15

// zpkExactResponse evaluates the strictly proper single-output response
// C(sI − A)⁻¹B of the stored data at the floating-point s exactly, by
// Gaussian elimination in complex rationals, and returns it rounded with
// its bounds: ver = comp = 2ε|Ĝ| plus two units of the smallest subnormal
// (the rounding of each part), and the terms T̂ = Σ|c_l||X_l| formed in
// double from X rounded, within γ_{n+5}·T̂ of the exact ones. zero reports
// an exactly singular sI − A, s an exact eigenvalue; ok is false when an
// entry's numerator or denominator exceeds zpkExactCapBits bits (the cap)
// or a value is not finite in double.
func zpkExactResponse(ref *System, s complex128) (e []zpkEntryBounds, zero, ok bool) {
	n, m, _ := ref.Dims()
	type crat struct{ re, im *big.Rat }
	newC := func(re, im float64) crat {
		return crat{new(big.Rat).SetFloat64(re), new(big.Rat).SetFloat64(im)}
	}
	big1 := func(x *big.Rat) bool {
		return x.Num().BitLen() > zpkExactCapBits || x.Denom().BitLen() > zpkExactCapBits
	}
	aRaw, bRaw, cRaw := ref.A.RawMatrix(), ref.B.RawMatrix(), ref.C.RawMatrix()
	w := n + m
	a := make([]crat, n*w)
	for i := range n {
		for j := range n {
			v := -aRaw.Data[i*aRaw.Stride+j]
			if i == j {
				a[i*w+j] = crat{new(big.Rat).SetFloat64(v), new(big.Rat).SetFloat64(imag(s))}
				a[i*w+j].re.Add(a[i*w+j].re, new(big.Rat).SetFloat64(real(s)))
			} else {
				a[i*w+j] = newC(v, 0)
			}
		}
		for j := range m {
			a[i*w+n+j] = newC(bRaw.Data[i*bRaw.Stride+j], 0)
		}
	}
	t1, t2, t3 := new(big.Rat), new(big.Rat), new(big.Rat)
	mul := func(x, y crat) crat {
		re := new(big.Rat).Mul(x.re, y.re)
		re.Sub(re, t1.Mul(x.im, y.im))
		im := new(big.Rat).Mul(x.re, y.im)
		im.Add(im, t1.Mul(x.im, y.re))
		return crat{re, im}
	}
	inv := func(x crat) crat {
		d := new(big.Rat).Mul(x.re, x.re)
		d.Add(d, t2.Mul(x.im, x.im))
		re := new(big.Rat).Quo(x.re, d)
		im := new(big.Rat).Quo(t3.Neg(x.im), d)
		return crat{re, im}
	}
	for k := range n {
		p := -1
		for i := k; i < n; i++ {
			if a[i*w+k].re.Sign() != 0 || a[i*w+k].im.Sign() != 0 {
				p = i
				break
			}
		}
		if p < 0 {
			return nil, true, false
		}
		if p != k {
			for j := range w {
				a[k*w+j], a[p*w+j] = a[p*w+j], a[k*w+j]
			}
		}
		ip := inv(a[k*w+k])
		for i := k + 1; i < n; i++ {
			if a[i*w+k].re.Sign() == 0 && a[i*w+k].im.Sign() == 0 {
				continue
			}
			f := mul(a[i*w+k], ip)
			for j := k + 1; j < w; j++ {
				d := mul(f, a[k*w+j])
				a[i*w+j].re.Sub(a[i*w+j].re, d.re)
				a[i*w+j].im.Sub(a[i*w+j].im, d.im)
				if big1(a[i*w+j].re) || big1(a[i*w+j].im) {
					return nil, false, false
				}
			}
		}
	}
	x := make([]crat, n*m)
	for j := range m {
		for i := n - 1; i >= 0; i-- {
			v := crat{new(big.Rat).Set(a[i*w+n+j].re), new(big.Rat).Set(a[i*w+n+j].im)}
			for l := i + 1; l < n; l++ {
				d := mul(a[i*w+l], x[l*m+j])
				v.re.Sub(v.re, d.re)
				v.im.Sub(v.im, d.im)
			}
			x[i*m+j] = mul(v, inv(a[i*w+i]))
			if big1(x[i*m+j].re) || big1(x[i*m+j].im) {
				return nil, false, false
			}
		}
	}
	e = make([]zpkEntryBounds, m)
	for j := range m {
		g := crat{new(big.Rat), new(big.Rat)}
		t := 0.0
		for l := range n {
			c := cRaw.Data[l]
			if c == 0 {
				continue
			}
			d := mul(newC(c, 0), x[l*m+j])
			g.re.Add(g.re, d.re)
			g.im.Add(g.im, d.im)
			xr, _ := x[l*m+j].re.Float64()
			xi, _ := x[l*m+j].im.Float64()
			t += math.Abs(c) * cabs(complex(xr, xi))
		}
		gr, _ := g.re.Float64()
		gi, _ := g.im.Float64()
		gv := complex(gr, gi)
		if cmplx.IsNaN(gv) || cmplx.IsInf(gv) || math.IsInf(t, 0) || math.IsNaN(t) {
			return nil, false, false
		}
		vb := 2*eps()*cmplx.Abs(gv) + 2*math.SmallestNonzeroFloat64
		dT := gammaN(n+5)*t + float64(n)*math.SmallestNonzeroFloat64
		e[j] = zpkEntryBounds{g: gv, ver: vb, comp: vb, lo: t - dT, hi: t + dT, determined: true}
	}
	return e, false, true
}

// zpkDDResponse evaluates the strictly proper single-output response
// ĉ(sI − Â)⁻¹B̂ of br at s by complex double-double GEPP, per input the
// rounded response with bounds: with u = ddOpErr, γ = zpkGEPPConstant(n,
// u), the inverse M̂⁻¹ formed in double-double, η = γ·‖|M̂⁻¹|‖_∞·
// ‖|L_f||U_f|‖_∞ and f = 1/(1 − η), δ_U = f·γ·|M̂⁻¹|Pᵀ(|L_f||U_f||Û|) bounds
// |U − Û|, so ver = comp = |ĉ|δ_U + γ_{n+3}(u)·|ĉ||Û| + 2ε|Ĝ| (the product
// in double-double, then the rounding to double) and the terms T̂ = Σ|ĉ||Û|,
// formed in double from Û rounded, lie within |ĉ|δ_U + γ_{n+5}(ε)·T̂ of
// the exact ones. zero reports an exactly zero pivot; ok is false when
// η ≥ 1 or a value is not finite.
func zpkDDResponse(br *balancedRealization, s complex128) (e []zpkEntryBounds, zero, ok bool) {
	n, m := br.n, br.m
	a := make([]ddComplex, n*n)
	for i := range n {
		for j := range n {
			if i == j {
				re, lo := twoSum(real(s), -br.a[i*n+j])
				a[i*n+j] = ddComplex{re: quickTwoSum(re, lo), im: dd{imag(s), 0}}
			} else {
				a[i*n+j] = ddComplex{re: dd{-br.a[i*n+j], 0}}
			}
		}
	}
	perm := make([]int, n)
	for k := range n {
		p, best := k, a[k*n+k].abs1()
		for i := k + 1; i < n; i++ {
			if v := a[i*n+k].abs1(); v > best {
				p, best = i, v
			}
		}
		if best == 0 {
			return nil, true, false
		}
		if math.IsNaN(best) || math.IsInf(best, 0) {
			return nil, false, false
		}
		perm[k] = p
		if p != k {
			for j := range n {
				a[k*n+j], a[p*n+j] = a[p*n+j], a[k*n+j]
			}
		}
		for i := k + 1; i < n; i++ {
			if a[i*n+k].abs1() == 0 {
				continue
			}
			f := a[i*n+k].div(a[k*n+k])
			a[i*n+k] = f
			for j := k + 1; j < n; j++ {
				a[i*n+j] = a[i*n+j].sub(f.mul(a[k*n+j]))
			}
		}
	}
	inv := make([]ddComplex, n*n)
	for i := range n {
		inv[i*n+i] = ddComplex{re: dd{1, 0}}
	}
	ddLUSolve(a, perm, inv, n, n)
	x := make([]ddComplex, n*m)
	for i := range n {
		for j := range m {
			x[i*m+j] = ddComplex{re: dd{br.b[i*m+j], 0}}
		}
	}
	ddLUSolve(a, perm, x, n, m)
	u := ddOpErr
	g := zpkGEPPConstant(n, u)
	nInv, nLU := 0.0, 0.0
	absLU := make([]float64, n*n)
	for i := range n {
		ri, rl := 0.0, 0.0
		for j := range n {
			ri += inv[i*n+j].abs()
			t := 0.0
			for l := 0; l <= min(i, j); l++ {
				li := 1.0
				if l != i {
					li = a[i*n+l].abs()
				}
				t += li * a[l*n+j].abs()
			}
			absLU[i*n+j] = t
			rl += t
		}
		nInv, nLU = math.Max(nInv, ri), math.Max(nLU, rl)
	}
	eta := g * nInv * nLU
	if !(eta < 1) {
		return nil, false, false
	}
	f := 1 / (1 - eta)
	v := make([]float64, n)
	w := make([]float64, n)
	e = make([]zpkEntryBounds, m)
	for j := range m {
		for i := range n {
			v[i] = 0
			for l := range n {
				v[i] += absLU[i*n+l] * x[l*m+j].abs()
			}
		}
		for k := n - 1; k >= 0; k-- {
			if p := perm[k]; p != k {
				v[k], v[p] = v[p], v[k]
			}
		}
		var gs ddComplex
		cu, du, t := 0.0, 0.0, 0.0
		for l := range n {
			wl := 0.0
			for q := range n {
				wl += inv[l*n+q].abs() * v[q]
			}
			w[l] = f * g * wl
			c := math.Abs(br.c[l])
			gs = gs.add(ddComplex{re: dd{br.c[l], 0}}.mul(x[l*m+j]))
			cu += c * x[l*m+j].abs()
			du += c * w[l]
			t += c * cabs(x[l*m+j].complex128())
		}
		gv := gs.complex128()
		if cmplx.IsNaN(gv) || cmplx.IsInf(gv) || math.IsInf(du, 0) || math.IsNaN(du) {
			return nil, false, false
		}
		vb := du + gammaDD(n+3)*cu + 2*eps()*cmplx.Abs(gv)
		dT := du + gammaN(n+5)*t
		e[j] = zpkEntryBounds{g: gv, ver: vb, comp: vb, lo: t - dT, hi: t + dT, determined: true}
	}
	return e, false, true
}

// gammaDD is γ_k at the double-double unit ddOpErr.
func gammaDD(k int) float64 {
	ku := float64(k) * ddOpErr
	return ku / (1 - ku)
}

// zpkFillProbes returns k axis points log-spaced at the midpoints of k
// equal steps across the band of the model's eigenvalues and the zeros:
// for a continuous model ω over [lo/10, 10·hi], lo and hi the smallest and
// largest nonzero root modulus; for a discrete one θ over [lo/10, π], lo
// the smallest nonzero |r − 1| (at least √ε).
func zpkFillProbes(evals, zeros []complex128, discrete bool, k int) []complex128 {
	lo, hi := math.Inf(1), 0.0
	for _, set := range [][]complex128{evals, zeros} {
		for _, r := range set {
			w := cmplx.Abs(r)
			if discrete {
				w = cmplx.Abs(r - 1)
			}
			if w > 0 && !math.IsInf(w, 0) && !math.IsNaN(w) {
				lo, hi = math.Min(lo, w), math.Max(hi, w)
			}
		}
	}
	if hi == 0 {
		lo, hi = 1, 1
	}
	a, b := lo/10, 10*hi
	if discrete {
		a, b = math.Max(lo/10, math.Sqrt(eps())), math.Pi
		a = math.Min(a, b/10)
	}
	pts := make([]complex128, k)
	for i := range k {
		w := a * math.Pow(b/a, (float64(i)+0.5)/float64(k))
		if discrete {
			pts[i] = cmplx.Rect(1, w)
		} else {
			pts[i] = complex(0, w)
		}
	}
	return pts
}

// zpkSharedPoleWindows is sharedPoleWindows for poles, the per-channel poles
// of a candidate row, exact data
// with no rounding bound or resolution of their own: a pole of z shares a
// model pole λ_G within β_G + ρ_G of it, and its window, the disk of
// radius β_G/tol capped at the distance to the nearest other model pole,
// exists when every pole it shares is determined to tol or is a
// rounding-level zero.
func zpkSharedPoleWindows(mp modelPoles, poles [][]complex128, tol float64) poleWindows {
	var w poleWindows
	var seen []complex128
	for _, ps := range poles {
		for _, lr := range ps {
			if slices.Contains(seen, lr) {
				continue
			}
			seen = append(seen, lr)
			shared, ok, rad, lim := false, true, 0.0, math.Inf(1)
			for jj, lg := range mp.evals {
				rj, bG := mp.rho[jj], mp.beta[jj]
				if cmplx.Abs(lr-lg) > bG+rj {
					continue
				}
				unresolved := cmplx.Abs(lg) <= bG && bG <= rj && cmplx.Abs(lr) <= bG
				if !(bG <= tol*math.Max(cmplx.Abs(lg), rj) || unresolved) {
					ok = false
					break
				}
				shared = true
				rad = math.Max(rad, bG)
				for k, lo := range mp.evals {
					if k != jj {
						lim = math.Min(lim, cmplx.Abs(lr-lo))
					}
				}
			}
			if ok && shared && rad > 0 {
				w.centre = append(w.centre, lr)
				w.radius = append(w.radius, math.Min(rad/tol, lim))
			}
		}
	}
	return w
}

// zpkEvalBounded returns k·Π(s − zᵢ)/Π(s − pᵢ) evaluated in floating
// point, a bound on its error for the given roots and gain, and false where
// s is a pole or the value is not finite. Each factor w = s − r is formed
// with relative error at most u; a pole's reciprocal is w̄/|w|², w first
// scaled by a power of two, with relative error at most γ₃; each complex
// product errs by at most √5·u ≤ γ₃ relative (Brent, Percival and
// Zimmermann; 2u with a fused multiply-add), and the running product is
// rescaled by powers of two, exactly. A zero factor costs γ₄ and a pole
// factor γ₇, so the computed v satisfies |v − Z| ≤ γ_K·|Z|, K = 4·n_z +
// 7·n_p, hence |v − Z| ≤ γ_K·|v|/(1 − γ_K); two units of the smallest
// subnormal cover the final unscaling.
func zpkEvalBounded(zs, ps []complex128, k float64, s complex128) (complex128, float64, bool) {
	if k == 0 {
		return 0, 0, true
	}
	acc, ex := complex(k, 0), 0
	norm := func() {
		mx := math.Max(math.Abs(real(acc)), math.Abs(imag(acc)))
		if mx == 0 || math.IsInf(mx, 0) || math.IsNaN(mx) {
			return
		}
		_, e := math.Frexp(mx)
		acc = complex(math.Ldexp(real(acc), -e), math.Ldexp(imag(acc), -e))
		ex += e
	}
	for i := 0; i < len(zs) || i < len(ps); i++ {
		if i < len(zs) {
			acc *= s - zs[i]
			norm()
		}
		if i < len(ps) {
			w := s - ps[i]
			a, b := real(w), imag(w)
			mx := math.Max(math.Abs(a), math.Abs(b))
			if mx == 0 || math.IsInf(mx, 0) || math.IsNaN(mx) {
				return 0, 0, false
			}
			_, e := math.Frexp(mx)
			a, b = math.Ldexp(a, -e), math.Ldexp(b, -e)
			q := a*a + b*b
			acc *= complex(a/q, -b/q)
			ex -= e
			norm()
		}
	}
	v := complex(math.Ldexp(real(acc), ex), math.Ldexp(imag(acc), ex))
	if cmplx.IsNaN(v) || cmplx.IsInf(v) {
		return 0, 0, false
	}
	g := gammaN(4*len(zs) + 7*len(ps))
	return v, g*cmplx.Abs(v)/(1-g) + 2*math.SmallestNonzeroFloat64, true
}
