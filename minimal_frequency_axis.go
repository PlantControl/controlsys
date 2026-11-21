package controlsys

import (
	"cmp"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// frequencyAxisCutInBand reports whether the staircase reduction red of
// sys is verified to match it to tol (relative) on the frequency axis, the
// unit circle for a discrete model and the imaginary axis for a continuous
// one; removed holds the stage blocks of the removed states. The staircase
// judges couplings against ε·‖A‖, but the response on the axis is that of
// (sI−A)⁻¹, whose norm can be far above 1/(ε·‖A‖): for modes clustered
// near z = 1 in a graded realization (matched C2D), or for a realization
// graded by a diagonal similarity of many decades. A coupling below the
// floor then carries an O(1) share of the response.
//
// The error of the cut need not peak beside the removed modes: when a
// removed pole is far from the axis the change C̃₂(sI−A₂₂)⁻¹R(sI−A₁₁)⁻¹B₁
// varies like the response itself, and the relative error is set by the
// kept poles and the zeros (matched C2D puts its zeros at z = −1). So
// every point of axisProbes is classified (classifyProbe), and the cut is
// accepted only when none rejects and the verified points give at least
// n + nr interpolation conditions, n and nr the orders of sys and red:
// G − R has a numerator of degree at most n + nr − 1, so agreement at
// n + nr points fixes it in exact arithmetic. A point off the real axis
// counts twice, since G − R is real-rational and agreement at s implies
// agreement at s̄; a point on it (s = 0, z = ±1) counts once. The points
// the plain bounds verify are settled first. A point is dead when some
// entry deviates by more than tol·scale beyond both responses' comp
// bounds: its exact deviation then exceeds tol, so it can never verify,
// only reject or be uninformative. The dead points under the plain bounds
// are refuted there where they can be and otherwise set aside; the others,
// in decreasing order of their refutation margin (probeMargin), are judged
// dead or live by the first, cheap pass of sharpen (stage1), skipping the
// points inside a shared-pole window that no sharper bound could verify or
// make reject (probeSurelyUninformative); only then are the live points
// sharpened and classified, and the dead ones last, for a reject only. The
// count is checked as each point is set aside, so a cut that cannot reach
// it stops before sharpening any point that could not change the verdict;
// the verdict does not depend on the order.
//
// Agreement within tol at those points gives no uniform bound between
// them: the probes are anchored on the data, not a node set with a known
// Lebesgue constant, and G − R can have poles near the axis between
// probes. Every eigenvalue of A has a probe at its nearest axis point, so
// pole-driven peaks are sampled to first order, while the zeros of G,
// where the relative error peaks, are sampled only by the fixed and decade
// grids. Nor is the check exact: the responses and their bounds are
// evaluated in floating point, so a point where the model's response is
// below its own first-order bound can neither verify nor refute a cut.
// At a verified point the exact responses agree to tol. The reference is
// the stored model's exact response: a point refutes a cut that deviates
// from it beyond tol relative to the larger of the response and the
// entry's own terms (classifyProbe), and the model's data resolution enters
// only in what a point inside a shared-pole window can refute.
// Nor is the count sufficient when the verified points lie far above a
// group of poles, a cluster near the axis or unresolved slow poles:
// poleScalesCovered counts per pole scale. Across a resolved spectral gap
// frequencyAxisCutAcrossGap adds one count per side of the gap. A cut
// that passes is then judged in double-double against the stored model at
// the points left open and beside the poles within rounding of the axis
// (judgeNearPoles). False also when the eigenvalues of A or of red do not
// converge.
func frequencyAxisCutInBand(sys, red *System, removed []*mat.Dense, tol float64) bool {
	return frequencyAxisCutAcrossGap(sys, red, removed, tol, 0)
}

// frequencyAxisCutAcrossGap is frequencyAxisCutInBand for a model whose
// eigenvalue magnitudes have a resolved gap at cutoff (resolvedGap); a
// cutoff ≤ 0 selects the band's rule alone. The global count n + nr is
// necessary but, across a gap of 1/tol or more, not sufficient: a point s
// constrains a pole λ beyond the leading Markov term C·B/s only through
// terms of relative size |λ|/|s|, so a point at the fast scale cannot see a
// slow pole to tol, and n + nr can be met on the fast side alone. On a
// continuous model the verified points, conjugate-weighted, must therefore
// also meet a count on each side: n_f + nr_f + 1 with |s| ≥ cutoff and
// n_s + nr_s + 1 with |s| ≤ ρ_s, n_x and nr_x the
// eigenvalues of A and of red on that side of the cutoff. Across the gap
// G − R = Δ_s + Δ_f exactly. At a slow point Δ_f is a constant to relative
// |s|/|λ_f|, so the slow side tests Δ_s plus a constant, of numerator
// degree n_s + nr_s; at a fast point Δ_s = d/s + O(λ_s/s²), so the fast
// side tests Δ_f + d/s, of numerator degree n_f + nr_f: each needs one
// zero more than its degree. The slow limit is the slow scale ρ_s, the
// largest |λ| below the cutoff among the eigenvalues of A and of red: the
// cut must match the stored model's slow dynamics to tol, a slow pole error
// Δ shows at a point s only as Δ/|s| once |s| > |λ_s|, so a point tests the
// slow side to tol only where the slow terms are not attenuated, |s| ≤ ρ_s
// (the probe beside the largest slow pole sits at |s| = ρ_s). The points
// between ρ_s and the cutoff count globally only. A discrete model takes the global
// count alone: a mode with |λ| ≤ √ε contributes C·B/z + O(λ/z²) at every
// point of the unit circle, invisible beyond C·B there.
func frequencyAxisCutAcrossGap(sys, red *System, removed []*mat.Dense, tol, cutoff float64) bool {
	n, _, _ := sys.Dims()
	nr, _, _ := red.Dims()
	pts, poles, ok := axisProbes(sys, removed)
	if !ok {
		return false
	}
	win, ok := sharedPoleWindows(poles, red, tol)
	if !ok {
		return false
	}
	sided := sys.Dt == 0 && cutoff > 0
	full, cut, inRange := bandResponses(sys, red, pts)
	defer full.release()
	defer cut.release()
	if !inRange {
		return false
	}
	var slow, fast int
	verified := 0
	isVerified := make([]bool, len(pts))
	accept := func() bool {
		if bandAcceptHook != nil {
			bandAcceptHook(sys, red, tol, cutoff, pts, isVerified, slow, fast, &full, &cut)
		}
		return true
	}
	type openProbe struct {
		k   int
		dev float64
	}
	var open []openProbe
	reach := 0
	for k, s := range pts {
		if !full.ok[k] {
			continue
		}
		if !cut.ok[k] {
			return false
		}
		if ok, dev := probeMargin(&full, &cut, k, tol); ok {
			isVerified[k] = true
		} else {
			open = append(open, openProbe{k, dev})
		}
		reach += probeWeight(s)
	}
	if reach < n+nr {
		return false
	}
	slices.SortStableFunc(open, func(a, b openProbe) int { return cmp.Compare(b.dev, a.dev) })
	var live, dead []int
	for _, o := range open {
		k := o.k
		if o.dev <= 1 {
			live = append(live, k)
			continue
		}
		if probeRefuted(&full, &cut, k, tol, win.contain(pts[k])) {
			return false
		}
		dead = append(dead, k)
		if reach -= probeWeight(pts[k]); reach < n+nr {
			return false
		}
	}
	undecided := live[:0]
	for _, k := range live {
		if probeSurelyUninformative(&full, &cut, k, tol, win.contain(pts[k])) {
			if reach -= probeWeight(pts[k]); reach < n+nr {
				return false
			}
			continue
		}
		if full.ok[k] && cut.ok[k] && !full.sharp[k] && !cut.sharp[k] {
			g, gc := full.stage1(k)
			r, rc := cut.stage1(k)
			if probeDead(g, r, gc, rc, tol) {
				dead = append(dead, k)
				if reach -= probeWeight(pts[k]); reach < n+nr {
					return false
				}
				continue
			}
		}
		undecided = append(undecided, k)
	}
	for _, k := range undecided {
		switch classifyProbe(&full, &cut, k, tol, win.contain(pts[k])) {
		case probeReject:
			return false
		case probeVerified:
			isVerified[k] = true
		default:
			if reach -= probeWeight(pts[k]); reach < n+nr {
				return false
			}
		}
	}
	for _, k := range dead {
		in := win.contain(pts[k])
		if probeSurelyUninformative(&full, &cut, k, tol, in) {
			continue
		}
		switch classifyProbe(&full, &cut, k, tol, in) {
		case probeReject:
			return false
		case probeVerified:
			isVerified[k] = true
		}
	}
	needSlow, needFast, slowScale := 1, 1, 0.0
	if sided {
		count := func(evals []complex128) {
			for _, ev := range evals {
				if a := cmplx.Abs(ev); a >= cutoff {
					needFast++
				} else {
					needSlow++
					slowScale = math.Max(slowScale, a)
				}
			}
		}
		count(poles.evals)
		count(win.cutPoles)
	}
	counted := func() bool {
		verified, slow, fast = 0, 0, 0
		for k, s := range pts {
			if !isVerified[k] {
				continue
			}
			w := probeWeight(s)
			verified += w
			switch a := cmplx.Abs(s); {
			case a >= cutoff:
				fast += w
			case a <= slowScale:
				slow += w
			}
		}
		return verified >= n+nr && (!sided || slow >= needSlow && fast >= needFast)
	}
	if !counted() {
		return false
	}
	discrete := sys.Dt != 0
	if !poleScalesCovered(discrete, poles.evals, win, pts, isVerified, tol) {
		if !resolveOpenProbes(&full, &cut, pts, isVerified, win, tol) || !counted() || !poleScalesCovered(discrete, poles.evals, win, pts, isVerified, tol) {
			return false
		}
	}
	if !judgeNearPoles(&full, &cut, pts, isVerified, poles, win, discrete, tol) {
		return false
	}
	return accept()
}

// resolveOpenProbes judges, in complex double-double, every point outside
// the shared-pole windows that the double bounds left open, for a cut the
// global count accepts but poleScalesCovered does not: GEPP in double
// errs by up to κ·ε, which beside a cluster of poles near the axis or an
// unresolved slow pole is the size of the cut's error, while double-double
// shrinks it by 2⁻⁵². The responses and their rigorous bounds are those of
// zpkDDResponse (comp = ver; the terms within their stated band
// [lo, hi]): a point is verified when |Ĝ−R̂| + δ_G + δ_R ≤ tol·max(scale,
// lo) at every entry, the exact deviation then within Reduce's bound
// tol·max(scale, terms_G) (lo > 0 of a determined band, as
// zpkJudgeEntries), and an entry refutes the cut when |Ĝ−R̂| − δ_G − δ_R >
// tol·max(scale, hi), which classifyProbe's refutation is. It
// reports false on a refutation; a point the double-double solve cannot
// bound stays open. Points inside a window stay open: there the
// refutation charges the data bounds, which are not formed here.
func resolveOpenProbes(full, cut *boundedResponse, pts []complex128, verified []bool, win poleWindows, tol float64) bool {
	dd := newDDJudge(full, tol)
	for k, s := range pts {
		if verified[k] || !full.ok[k] || !cut.ok[k] || win.contain(s) {
			continue
		}
		switch dd.judge(full, cut, s) {
		case probeReject:
			return false
		case probeVerified:
			verified[k] = true
		}
	}
	return true
}

// ddJudge judges a point in complex double-double (resolveOpenProbes),
// with its buffers.
type ddJudge struct {
	w    zpkDDWork
	g, r []zpkEntryBounds
	tol  float64
}

func newDDJudge(full *boundedResponse, tol float64) *ddJudge {
	pm := full.p * full.m
	return &ddJudge{g: make([]zpkEntryBounds, pm), r: make([]zpkEntryBounds, pm), tol: tol}
}

// judge is resolveOpenProbes' verdict at s: uninformative where either
// double-double solve cannot bound its response.
func (d *ddJudge) judge(full, cut *boundedResponse, s complex128) probeVerdict {
	if !ddEntryBounds(&full.ev.br, s, &d.w, d.g) || !ddEntryBounds(&cut.ev.br, s, &d.w, d.r) {
		return probeUninformative
	}
	return d.verdict()
}

// refutes reports whether judge rejects at pts[k] of cb, the cut's double
// response with its bounds. The cut's double-double solve is skipped where
// cb shows judge cannot reject: with Ĝ the model's double-double response,
// R̂ the cut's double one, R its exact one and δ_R ≥ |R − R̂| cb's ver
// bound (rigorous, as a skip must be; comp is first order),
// |Ĝ−R_dd| − δ_R,dd ≤ |Ĝ−R̂| + δ_R, so no entry is refuted when
// |Ĝ−R̂| + δ_R − δ_G ≤ tol·max(max|Ĝ|, T_G,hi), T_G,hi its terms' upper
// bound. The test is tried first on the model's solve stage alone
// (ddEntrySolve), with comp = 0 and T_G,hi = T̂, either of which only
// raises its left side or lowers its right: where that passes the full
// bounds pass too, and their inverse and |L||U| are not formed. A cut's
// response at points of its own (cb ≠ cut) is sharpened for that test
// before the model's bounds are formed: beside a pole its plain bound is
// too loose to dismiss the point.
func (d *ddJudge) refutes(full, cut, cb *boundedResponse, k int, s complex128) bool {
	if !ddEntrySolve(&full.ev.br, s, &d.w, d.g) {
		return false
	}
	if d.cutCannotRefute(cb, k, false) || cb != cut && d.cutCannotRefute(cb, k, true) {
		return false
	}
	if !ddEntryBoundStage(&full.ev.br, &d.w, d.g) {
		return false
	}
	if d.cutCannotRefute(cb, k, false) || cb != cut && d.cutCannotRefute(cb, k, true) {
		return false
	}
	if !ddEntryBounds(&cut.ev.br, s, &d.w, d.r) {
		return false
	}
	return d.verdict() == probeReject
}

// verifies is judge at pts[k] of cb (refutes) for a point that must verify:
// a verification by the cut's double response with its ver bounds, the
// model's in double-double, stands without the cut's double-double solve.
func (d *ddJudge) verifies(full, cut, cb *boundedResponse, k int, s complex128) bool {
	if !ddEntryBounds(&full.ev.br, s, &d.w, d.g) {
		return false
	}
	if d.cutVerifies(cb, k, false) || d.cutVerifies(cb, k, true) {
		return true
	}
	if !ddEntryBounds(&cut.ev.br, s, &d.w, d.r) {
		return false
	}
	return d.verdict() == probeVerified
}

// cutCannotRefute is refutes' test with cb's bounds at k, sharpened first
// with sharpen; a non-finite quantity fails the test.
func (d *ddJudge) cutCannotRefute(cb *boundedResponse, k int, sharpen bool) bool {
	if !cb.ok[k] || sharpen && cb.sharp[k] {
		return false
	}
	if sharpen {
		cb.sharpen(k)
	}
	g := d.g
	o := k * len(g)
	sg := 0.0
	for i := range g {
		sg = math.Max(sg, cmplx.Abs(g[i].g))
	}
	for i := range g {
		up := (cmplx.Abs(g[i].g-cb.g[o+i])+cb.ver[o+i])*(1+8*eps()) - g[i].comp
		thr := d.tol * math.Max(sg, g[i].hi) * (1 - 8*eps())
		if !finiteAll(up, thr) || !(up <= thr) {
			return false
		}
	}
	return true
}

// cutVerifies is verdict's verification with cb's response and ver bounds
// at k for the cut's, sharpened first with sharpen; a non-finite quantity
// does not verify.
func (d *ddJudge) cutVerifies(cb *boundedResponse, k int, sharpen bool) bool {
	if !cb.ok[k] || sharpen && cb.sharp[k] {
		return false
	}
	if sharpen {
		cb.sharpen(k)
	}
	g, tol := d.g, d.tol
	o := k * len(g)
	scale := 0.0
	for i := range g {
		scale = math.Max(scale, math.Max(cmplx.Abs(g[i].g), cmplx.Abs(cb.g[o+i])))
	}
	if !isFinite(tol * scale) {
		return false
	}
	for i := range g {
		lhs := cmplx.Abs(g[i].g-cb.g[o+i]) + g[i].ver + cb.ver[o+i]
		if !finiteAll(lhs, g[i].lo) || !(lhs <= tol*scale || g[i].determined && g[i].lo > 0 && lhs <= tol*math.Max(scale, g[i].lo)) {
			return false
		}
	}
	return true
}

// verdict judges the bounds judge formed, an entry or scale with a
// non-finite quantity as classifyProbe does.
func (d *ddJudge) verdict() probeVerdict {
	g, r, tol := d.g, d.r, d.tol
	scale := 0.0
	for i := range g {
		scale = math.Max(scale, math.Max(cmplx.Abs(g[i].g), cmplx.Abs(r[i].g)))
	}
	if !isFinite(tol * scale) {
		return probeUninformative
	}
	v := probeVerified
	for i := range g {
		diff := cmplx.Abs(g[i].g - r[i].g)
		lhs := diff + g[i].ver + r[i].ver
		if !finiteAll(lhs, g[i].comp, r[i].comp, g[i].lo, g[i].hi) {
			v = probeUninformative
			continue
		}
		if lhs <= tol*scale || g[i].determined && g[i].lo > 0 && lhs <= tol*math.Max(scale, g[i].lo) {
			continue
		}
		if diff-g[i].comp-r[i].comp > tol*math.Max(scale, g[i].hi) {
			return probeReject
		}
		v = probeUninformative
	}
	return v
}

// judgeNearPoles judges, in complex double-double (ddJudge) against the
// stored model, the points the band's double bounds left open and the
// axis points beside the poles its probes cannot see, for a cut the band
// would accept; false when one refutes the cut or a hidden pole's side
// point does not verify it. A point it verifies is not counted: the count
// is the band's.
//
// An open probe beside a pole that lies just beyond its reach of the axis
// is singular to the double solve's precision, as is one inside a window
// of a pole the data place off the axis, where a cut that moves the pole
// within its rounding bound β_G (≤ tol·|λ_G|) is wrong beyond tol near it:
// double-double resolves both, so they are judged, and may refute. Inside
// the window of a shared pole within rounding of the axis (axisShared:
// its foot within β_G + β_R of it) the probes keep the band's data slack,
// and Reduce certifies nothing there: neither the stored model nor the
// cut places that pole better than its rounding bound. The other shared
// windows, exempt before, are judged (a contract change). Just outside
// each such axis window the cut is judged and may refute
// (axisWindowEdges): no probe lies there, and a cut pole further from the
// model's than the window's bound is wrong there first.
//
// A hidden model pole (modelPoles) that is not axisShared is one the cut
// drops, carries only off its location, or shares where the data place it
// off the axis by more than β_G + β_R, while within its reach of the axis:
// the stored model's exact pole may lie on the axis, so a rounding-level
// residue r on it (a rotated uncontrollable or unobservable integrator,
// z = −1, e^{jθ} or ±jω mode) puts r/d into the response at the foot, d
// the pole's distance from it. axisProbes drops the foot, and a probe at
// distance D ≫ d bounds r only to tol·D·scale, so the probes at √ε and
// beyond verify a cut that is O(1) wrong at the foot. Double-double
// resolves the pole to ε² of the data, far inside the reach, so the foot is
// judged and, at axial distance reach on either side (one side for a real
// pole, its mirror the conjugate), the side points must verify: where the
// stored model's residue on the pole is exactly zero the response there is
// smooth and determined, and where it is not, it is r/reach there.
func judgeNearPoles(full, cut *boundedResponse, pts []complex128, verified []bool, mp modelPoles, win poleWindows, discrete bool, tol float64) bool {
	var dd *ddJudge
	judge := func() *ddJudge {
		if dd == nil {
			dd = newDDJudge(full, tol)
		}
		return dd
	}
	for k, s := range pts {
		if verified[k] || win.contain(s) && !win.strictlyJudged(s) {
			continue
		}
		if judge().refutes(full, cut, cut, k, s) {
			return false
		}
	}
	var near []complex128
	var sides []int
	for j, ev := range mp.evals {
		if !mp.hidden[j] || imag(ev) < 0 || win.axisShared[j] {
			continue
		}
		foot, d := mp.foot[j], mp.reach[j]
		near = append(near, foot)
		if discrete {
			th := cmplx.Phase(foot)
			near = append(near, cmplx.Rect(1, th+d))
			if imag(ev) != 0 {
				near = append(near, cmplx.Rect(1, th-d))
			}
		} else {
			near = append(near, complex(0, imag(foot)+d))
			if imag(ev) != 0 {
				near = append(near, complex(0, imag(foot)-d))
			}
		}
		sides = append(sides, len(near))
	}
	near = axisWindowEdges(near, win, discrete)
	if len(near) == 0 {
		return true
	}
	cb := boundedResponseOf(cut.ev.br, near)
	defer cb.release()
	k := 0
	for _, end := range sides {
		if judge().refutes(full, cut, &cb, k, near[k]) {
			return false
		}
		for k++; k < end; k++ {
			if !judge().verifies(full, cut, &cb, k, near[k]) {
				return false
			}
		}
	}
	for ; k < len(near); k++ {
		if judge().refutes(full, cut, &cb, k, near[k]) {
			return false
		}
	}
	return true
}

// axisWindowEdges appends to pts the axis points just outside each axis
// window (poleWindows.axisPole), at axial distance (1+√ε)·radius from its
// centre on either side (one side for a real centre, its mirror the
// conjugate), but none inside another window that is not judged. Beyond the
// window the cut is judged: a cut pole whose exact distance from the
// model's exceeds the window's bound β_G + β_R + min(|λ_R − λ_G|, β_G +
// β_R) (a first-order bound for one shared pole, which eigensolver error
// can exceed, or a cut pole moved off it) deviates there by that distance
// over the radius, more than tol, and the deviation is largest at
// the edge, falling as 1/d beyond it, where no probe sees it.
func axisWindowEdges(pts []complex128, win poleWindows, discrete bool) []complex128 {
	f := 1 + math.Sqrt(eps())
	for i, c := range win.centre {
		if !win.axisPole[i] || imag(c) < 0 {
			continue
		}
		r := f * win.radius[i]
		var s [2]complex128
		if discrete {
			th := cmplx.Phase(c)
			s = [2]complex128{cmplx.Rect(1, th+r), cmplx.Rect(1, th-r)}
		} else {
			s = [2]complex128{complex(0, imag(c)+r), complex(0, imag(c)-r)}
		}
		for j, p := range s {
			if j == 1 && imag(c) == 0 {
				break
			}
			if win.contain(p) && !win.strictlyJudged(p) {
				continue
			}
			pts = append(pts, p)
		}
	}
	return pts
}

// ddEntryBounds sets e, row-major p×m, to the strictly proper response of
// br at s with zpkDDResponse's bounds, each output row's as zpkDDResponse
// forms it alone; false where the double-double solve meets a zero pivot
// or cannot bound it.
func ddEntryBounds(br *balancedRealization, s complex128, w *zpkDDWork, e []zpkEntryBounds) bool {
	if br.p == 0 {
		return true
	}
	if _, ok := zpkDDSolve(br, s, w); !ok {
		return false
	}
	return ddEntryBoundStage(br, w, e)
}

// ddEntryBoundStage is ddEntryBounds after ddEntrySolve, continuing from
// the factorization and solution in w: one factorization of sI − Â serves
// every output row, as the rows share it.
func ddEntryBoundStage(br *balancedRealization, w *zpkDDWork, e []zpkEntryBounds) bool {
	n, m := br.n, br.m
	if !zpkDDBound(n, m, w) {
		return false
	}
	for i := range br.p {
		for j := range m {
			var ok bool
			if e[i*m+j], ok = zpkDDEntryBounds(br.c[i*n:(i+1)*n], w, n, m, j); !ok {
				return false
			}
		}
	}
	return true
}

// ddEntrySolve is ddEntryBounds' solve stage alone: e holds the responses
// Ĝ, bitwise those ddEntryBounds forms, with comp = 0 and hi = lo = T̂,
// the terms before their bound is added (T̂ ≤ hi of ddEntryBounds).
func ddEntrySolve(br *balancedRealization, s complex128, w *zpkDDWork, e []zpkEntryBounds) bool {
	if br.p == 0 {
		return true
	}
	if _, ok := zpkDDSolve(br, s, w); !ok {
		return false
	}
	n, m := br.n, br.m
	for i := range br.p {
		for j := range m {
			g, t := zpkDDEntry(br.c[i*n:(i+1)*n], w, n, m, j)
			if cmplx.IsNaN(g) || cmplx.IsInf(g) {
				return false
			}
			e[i*m+j] = zpkEntryBounds{g: g, lo: t, hi: t, determined: true}
		}
	}
	return true
}

// poleScalesCovered reports whether the verified points give every group
// of poles enough interpolation conditions at its own scale. The scale of
// a pole is |λ| on a continuous model and |ln λ| on a discrete one (z =
// e^{sT} maps the unit circle near z = 1 to s = jθ/T), that of a point |s|
// or |θ|. For each pole scale σ, the poles of the model and the cut at or
// below it, a pole the two share (one with a window) counted once, form a
// group whose part of G − R has, beyond σ, the expansion Σ_m M_m/s^{m+1}
// with |M_m| ≤ σ^m·Σ|r|: at |s| = d > σ its m-th term is (σ/d)^m of the
// leading one, so points at d or beyond, verified to tol, fix at most
// K(d) = 1 + ⌊ln(1/tol)/ln(d/σ)⌋ of its coefficients, whatever their
// number. Points at or below σ count fully, and those beyond count, in
// order of distance, up to the least K(d) of the points at d or beyond;
// the group needs as many as it has poles. Without this the global count
// n + nr can be met by points far above a cluster of poles near the axis
// (matched C2D near z = 1) or above unresolved slow poles that leave no
// resolved gap, while every point at their scale is uninformative. At
// tol ≥ 1 the contract holds at every point, |G−R| = |Σ_l C_il X_lj| ≤
// terms ≤ tol·terms, and K's premise ln(1/tol) > 0 is empty, so the count
// per scale does not apply.
func poleScalesCovered(discrete bool, evG []complex128, win poleWindows, pts []complex128, verified []bool, tol float64) bool {
	if !(tol < 1) {
		return true
	}
	poleScale := func(ev complex128) float64 {
		if !discrete {
			return cmplx.Abs(ev)
		}
		if ev == 0 {
			return math.Inf(1)
		}
		return cmplx.Abs(cmplx.Log(ev))
	}
	type scaled struct {
		a float64
		w int
	}
	var poles, at []scaled
	for _, ev := range evG {
		poles = append(poles, scaled{poleScale(ev), 1})
	}
	for _, ev := range win.cutPoles {
		poles = append(poles, scaled{poleScale(ev), 1})
	}
	for _, c := range win.centre {
		poles = append(poles, scaled{poleScale(c), -1})
	}
	for k, s := range pts {
		if !verified[k] {
			continue
		}
		a := cmplx.Abs(s)
		if discrete {
			a = math.Abs(cmplx.Phase(s))
		}
		at = append(at, scaled{a, probeWeight(s)})
	}
	byScale := func(x, y scaled) int { return cmp.Compare(x.a, y.a) }
	slices.SortFunc(poles, byScale)
	slices.SortFunc(at, byScale)
	lt := math.Log(1 / tol)
	need := 0
	for i, pl := range poles {
		need += pl.w
		sigma := pl.a
		if i+1 < len(poles) && poles[i+1].a == sigma || math.IsInf(sigma, 1) {
			continue
		}
		have, beyond, capped := 0, 0, math.MaxInt
		for _, q := range at {
			if q.a <= sigma {
				have += q.w
				continue
			}
			k := 1
			if sigma > 0 {
				k += int(math.Floor(lt / math.Log(q.a/sigma)))
			}
			capped = min(capped, beyond+k)
			beyond += q.w
		}
		if have+min(beyond, capped) < need {
			return false
		}
	}
	return true
}

// bandAcceptHook is a test-only seam, nil outside tests: it sees every cut
// the band accepts, with the gate's cutoff, the probes it verified and
// their conjugate-weighted slow and fast counts.
var bandAcceptHook func(sys, red *System, tol, cutoff float64, pts []complex128, verified []bool, slow, fast int, full, cut *boundedResponse)

type probeVerdict int

const (
	probeVerified probeVerdict = iota
	// probeUninformative is a point where neither response is determined
	// well enough to verify or refute the cut.
	probeUninformative
	probeReject
)

// classifyProbe judges the cut at point k of full and cut, the strictly
// proper responses Ĝ, R̂ of the model and the cut with their bounds
// (boundedResponse), per entry against tol·scale, scale the largest |Ĝ|,
// |R̂| of the point. An entry is verified when |Ĝ−R̂| + δ_G + δ_R ≤
// tol·scale, δ the verification bounds: the exact deviation is then within
// tol. The reference is the stored model's exact response, so outside a
// shared-pole window an entry is refuted, rejecting the point, when
// |Ĝ−R̂| − δ_G − δ_R > tol·max(scale, terms_G), δ the computational bounds
// and terms_G the entry's own terms Σ_l |C_il||Û_lj|: the exact deviation
// then exceeds tol relative to the larger of the response and the terms
// that form it, the scale a cut's zero can be placed to. Otherwise the
// entry is open, and the point rejects when the model is determined to tol
// (δ_G ≤ tol·|Ĝ|) while the cut is not, when the cut's response is below
// its own resolution, data or computational (cutNoise: a positive
// δ_R,data or δ_R at or above |R̂|), or when it deviates from the model's
// by more than the model's own bound (|Ĝ−R̂| − δ_G > tol·max(scale,
// terms_G)); it is uninformative otherwise. inShared marks the window of a
// pole the model shares with the cut (sharedPoleWindows), where the cut's
// pole is known only to its rounding and a rounding-level move of either
// pole changes the response by up to its data bound: there the refutation
// is |Ĝ−R̂| − δ_G − δ_R − δ_G,data − δ_R,data > tol·scale, none of the
// heuristics applies, and an open entry is uninformative. Nothing inside a
// window verifies, so this only turns a spurious reject into an
// uninformative point. A point the plain bounds verify is verified, and
// one they refute is refuted (the plain comp bounds the sharp one);
// otherwise both responses are sharpened there before the judgement.
//
// A point where GEPP meets an exactly singular model is uninformative, one
// with a singular cut rejects. An entry with a non-finite quantity
// (probeEntryFinite) is undetermined: it neither verifies nor refutes and
// its point does not verify, while the point's finite entries are judged
// as before; a point whose tol·scale is not finite is uninformative, as
// its true scale is unknown. The other judges of the band follow the
// same rule.
func classifyProbe(full, cut *boundedResponse, k int, tol float64, inShared bool) probeVerdict {
	if !full.ok[k] {
		return probeUninformative
	}
	if !cut.ok[k] {
		return probeReject
	}
	pm := full.p * full.m
	o := k * pm
	if !full.sharp[k] || !cut.sharp[k] {
		if probeVerifies(full, cut, k, tol) {
			return probeVerified
		}
		if probeRefuted(full, cut, k, tol, inShared) {
			return probeReject
		}
		full.sharpen(k)
		cut.sharpen(k)
	}
	g, r := full.g[o:o+pm], cut.g[o:o+pm]
	scale := probeScale(g, r)
	if !isFinite(tol * scale) {
		return probeUninformative
	}
	verdict := probeVerified
	for i, v := range g {
		gc, rc := full.comp[o+i], cut.comp[o+i]
		diff := cmplx.Abs(v - r[i])
		if !probeEntryFinite(full, cut, o+i, diff) {
			verdict = probeUninformative
			continue
		}
		if diff+full.ver[o+i]+cut.ver[o+i] <= tol*scale {
			continue
		}
		if inShared {
			if diff-gc-rc-full.data[o+i]-cut.data[o+i] > tol*scale {
				return probeReject
			}
		} else if gc <= tol*cmplx.Abs(v) || cutNoise(rc, cut.data[o+i], cmplx.Abs(r[i])) || diff-gc > tol*math.Max(scale, full.terms[o+i]) {
			return probeReject
		}
		verdict = probeUninformative
	}
	return verdict
}

// probeRefuted reports whether some entry of point k is refuted with the
// bounds at hand, by classifyProbe's refutation.
func probeRefuted(full, cut *boundedResponse, k int, tol float64, inShared bool) bool {
	pm := full.p * full.m
	o := k * pm
	g, r := full.g[o:o+pm], cut.g[o:o+pm]
	scale := probeScale(g, r)
	if !isFinite(tol * scale) {
		return false
	}
	for i, v := range g {
		diff := cmplx.Abs(v - r[i])
		if !probeEntryFinite(full, cut, o+i, diff) {
			continue
		}
		d := diff - full.comp[o+i] - cut.comp[o+i]
		if inShared {
			if d-full.data[o+i]-cut.data[o+i] > tol*scale {
				return true
			}
		} else if d > tol*math.Max(scale, full.terms[o+i]) {
			return true
		}
	}
	return false
}

// probeDead reports whether some entry of a point deviates by more than
// tol·scale beyond both responses' comp bounds, g and r the responses and
// gc, rc their comps: the exact deviation then exceeds tol·scale, so no
// sharper bound verifies the point, and it can only reject or be
// uninformative.
func probeDead(g, r []complex128, gc, rc []float64, tol float64) bool {
	ts := tol * probeScale(g, r)
	if !isFinite(ts) {
		return false
	}
	for i, v := range g {
		d := cmplx.Abs(v - r[i])
		if !finiteAll(d, gc[i], rc[i]) {
			continue
		}
		if d-gc[i]-rc[i] > ts {
			return true
		}
	}
	return false
}

// bandResponses returns the bounded responses of sys and red at pts, and
// whether the band may judge them. A response or bound that overflows, or
// with 2^(eB+eC) < 1 an entry not structurally zero whose terms fall below
// 2^−969, where rounding stops being relative (inRange), leaves the double
// range; 2^eB and 2^eC are the scales of sys's largest |B̂_ij| and |Ĉ_ij|.
// On overflow the band refuses the cut: the pole resolutions its windows
// and probes read come from a balancing that saw the huge B̂ and Ĉ, so a
// point rescaled into range may refute but must never verify, and a band
// that cannot verify cannot certify. On underflow with eB ≤ 0 and eC ≤ 0
// both responses are formed instead on the realizations with B̂ and Ĉ
// scaled up by 2^−eB and 2^−eC, exact for both or the cut is refused:
// every quantity a judge reads is bilinear in B̂ and Ĉ and compared only
// relative to the others, and tiny B̂ and Ĉ do not inflate those
// resolutions, so the rescaled points judge as the stored data's would.
// Underflow with either exponent positive is refused as overflow is: the
// larger of B̂ and Ĉ took part in the balancing. With 2^(eB+eC) < 2^−512 they are scaled up
// before the first evaluation, which is then the only one.
func bandResponses(sys, red *System, pts []complex128) (boundedResponse, boundedResponse, bool) {
	n, m, p := sys.Dims()
	nr, _, _ := red.Dims()
	rf, rc := newBalancedRealization(sys, n, m, p), newBalancedRealization(red, nr, m, p)
	eB, eC := maxExponent(rf.b), maxExponent(rf.c)
	under, scaled := eB+eC < 0, false
	up := under && eB <= 0 && eC <= 0
	if up && eB+eC < -512 {
		sf, okF := scaledIO(rf, eB, eC)
		sc, okC := scaledIO(rc, eB, eC)
		if okF && okC {
			rf, rc, scaled = sf, sc, true
		}
	}
	full, cut := boundedResponseOf(rf, pts), boundedResponseOf(rc, pts)
	finF, bigF := full.inRange(&rf, under)
	finC, bigC := cut.inRange(&rc, under)
	if !finF || !finC {
		return full, cut, false
	}
	if bigF && bigC || scaled {
		return full, cut, bigF && bigC
	}
	if !up {
		return full, cut, false
	}
	sf, okF := scaledIO(rf, eB, eC)
	sc, okC := scaledIO(rc, eB, eC)
	if !okF || !okC {
		return full, cut, false
	}
	full.release()
	cut.release()
	full, cut = boundedResponseOf(sf, pts), boundedResponseOf(sc, pts)
	finF, bigF = full.inRange(&sf, under)
	finC, bigC = cut.inRange(&sc, under)
	return full, cut, finF && finC && bigF && bigC
}

// inRange reports whether every response and bound at the ok points of b,
// the response of rz, is finite, and, with under, whether every entry that
// is not structurally zero (no path from its input through the nonzeros of
// Â to its output) has terms of at least 2^−969, above which their rounding
// is relative, or exactly zero and still zero on B̂ and Ĉ scaled up
// (zerosExact): below the floor a response and its bounds underflow, to an
// exact zero that verifies any cut.
func (b *boundedResponse) inRange(rz *balancedRealization, under bool) (finite, big bool) {
	const tiny = 0x1p-969
	n, m, p := rz.n, rz.m, rz.p
	var live []bool
	if under {
		live = make([]bool, p*m)
		reach := make([]bool, n)
		stack := make([]int, 0, n)
		for j := range m {
			clear(reach)
			stack = stack[:0]
			for l := range n {
				if rz.b[l*m+j] != 0 {
					reach[l] = true
					stack = append(stack, l)
				}
			}
			for len(stack) > 0 {
				l := stack[len(stack)-1]
				stack = stack[:len(stack)-1]
				for q := range n {
					if !reach[q] && rz.a[q*n+l] != 0 {
						reach[q] = true
						stack = append(stack, q)
					}
				}
			}
			for i := range p {
				for l := range n {
					if reach[l] && rz.c[i*n+l] != 0 {
						live[i*m+j] = true
						break
					}
				}
			}
		}
	}
	pm := p * m
	finite, big = true, true
	var zs []int
	for k, ok := range b.ok {
		if !ok {
			continue
		}
		zero := false
		for y := range pm {
			o := k*pm + y
			g := b.g[o]
			finite = finite && finiteAll(real(g), imag(g), b.comp[o], b.ver[o], b.data[o], b.terms[o])
			if under && live[y] {
				t := b.terms[o]
				big = big && !(t > 0 && t < tiny)
				zero = zero || t == 0
			}
		}
		if zero {
			zs = append(zs, k)
		}
	}
	if finite && big && len(zs) > 0 {
		big = b.zerosExact(rz, zs, live)
	}
	return finite, big
}

// zerosExact reports whether the live entries with zero terms at the
// points zs of b, the response of rz, stay exactly zero with B̂ and Ĉ
// scaled up by powers of two to largest magnitudes 2^400: a zero that
// does is the stored data's own, a zero that does not had underflowed.
func (b *boundedResponse) zerosExact(rz *balancedRealization, zs []int, live []bool) bool {
	sz, ok := scaledIO(*rz, maxExponent(rz.b)-400, maxExponent(rz.c)-400)
	if !ok {
		return false
	}
	pts := make([]complex128, len(zs))
	for i, k := range zs {
		pts[i] = b.ev.pts[k]
	}
	z := boundedResponseOf(sz, pts)
	defer z.release()
	pm := b.p * b.m
	for i, k := range zs {
		for y := range pm {
			if live[y] && b.terms[k*pm+y] == 0 && (!z.ok[i] || z.terms[i*pm+y] != 0) {
				return false
			}
		}
	}
	return true
}

// maxExponent is the binary exponent e of the largest |v_i|, 2^(e−1) ≤
// |v_i| < 2^e, 0 for a zero or empty v.
func maxExponent(v []float64) int {
	mx := 0.0
	for _, x := range v {
		mx = math.Max(mx, math.Abs(x))
	}
	if mx == 0 || !isFinite(mx) {
		return 0
	}
	_, e := math.Frexp(mx)
	return e
}

// scaledIO is rz with B̂ scaled by 2^−eB and Ĉ by 2^−eC, ok when every
// entry scales exactly.
func scaledIO(rz balancedRealization, eB, eC int) (balancedRealization, bool) {
	scale := func(v []float64, e int) ([]float64, bool) {
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = math.Ldexp(x, -e)
			if math.Ldexp(out[i], e) != x {
				return nil, false
			}
		}
		return out, true
	}
	out := rz
	var okB, okC bool
	out.b, okB = scale(rz.b, eB)
	out.c, okC = scale(rz.c, eC)
	return out, okB && okC
}

// probeEntryFinite reports whether entry o of full and cut, with deviation
// diff = |Ĝ−R̂|, has finite responses, bounds, terms and deviation.
func probeEntryFinite(full, cut *boundedResponse, o int, diff float64) bool {
	return finiteAll(diff, full.comp[o], full.ver[o], full.data[o], full.terms[o],
		cut.comp[o], cut.ver[o], cut.data[o], cut.terms[o])
}

// probeScale is the largest |Ĝ|, |R̂| of a point's entries.
func probeScale(g, r []complex128) float64 {
	scale := 0.0
	for i, v := range g {
		scale = math.Max(scale, math.Max(cmplx.Abs(v), cmplx.Abs(r[i])))
	}
	return scale
}

// cutNoise reports whether a cut's response rs is below its own
// resolution, data or computational: a positive bound at or above rs. An
// exactly zero response with zero bounds is an exact zero, not noise.
func cutNoise(comp, data, rs float64) bool {
	return data > 0 && data >= rs || comp > 0 && comp >= rs
}

// probeSurelyUninformative reports whether classifyProbe would find open
// point k inside a shared-pole window uninformative under any sharpened
// bounds, so it need not sharpen: some entry cannot verify (|Ĝ−R̂| >
// tol·scale), and for every such entry no comp ≥ 0 lets the refutation
// fire, since |Ĝ−R̂| − δ_G,data − δ_R,data ≤ tol·scale. The sharp responses
// differ from the plain ones by rounding, so these conditions keep the
// verdict. Outside a window no open point is surely uninformative: the
// sharp comps have no positive lower bound, so the heuristics can fire.
func probeSurelyUninformative(full, cut *boundedResponse, k int, tol float64, inShared bool) bool {
	if !inShared {
		return false
	}
	pm := full.p * full.m
	o := k * pm
	g, r := full.g[o:o+pm], cut.g[o:o+pm]
	ts := tol * probeScale(g, r)
	for i, v := range g {
		diff := cmplx.Abs(v - r[i])
		if diff <= ts || diff-full.data[o+i]-cut.data[o+i] > ts {
			return false
		}
	}
	return true
}

// probeMargin reports whether every entry of point k verifies with the
// bounds at hand, and the largest refutation margin (|Ĝ−R̂| − δ_G −
// δ_R)/(tol·scale) of its entries.
func probeMargin(full, cut *boundedResponse, k int, tol float64) (bool, float64) {
	pm := full.p * full.m
	o := k * pm
	g, r := full.g[o:o+pm], cut.g[o:o+pm]
	ts := tol * probeScale(g, r)
	if !isFinite(ts) {
		return false, math.Inf(-1)
	}
	ok, dev := true, math.Inf(-1)
	for i, v := range g {
		d, b := cmplx.Abs(v-r[i]), full.ver[o+i]+cut.ver[o+i]
		if !probeEntryFinite(full, cut, o+i, d) {
			ok = false
			continue
		}
		ok = ok && d+b <= ts
		dev = math.Max(dev, d-b)
	}
	return ok, dev / ts
}

// probeVerifies reports whether every entry of point k verifies with the
// bounds at hand.
func probeVerifies(full, cut *boundedResponse, k int, tol float64) bool {
	ok, _ := probeMargin(full, cut, k, tol)
	return ok
}

// probeWeight is the number of interpolation conditions a point gives:
// two off the real axis, one on it.
func probeWeight(s complex128) int {
	if imag(s) != 0 {
		return 2
	}
	return 1
}

// modelPoles holds the eigenvalues of the model's A with their
// componentwise rounding bounds β (eigenRoundingBounds), ‖A_bal‖_F, the
// eigensolver's resolution ρ₀ = n·ε·‖A_bal‖_F, A_bal the power-of-two
// balanced A of boundedResponse, and, per pole, the resolution
// ρ_j = min(ρ₀, 2n·β_j) of a pole the model resolves from zero
// (β_j < |λ_j|), ρ₀ of one it does not: a backward-stable eigensolver's
// error is componentwise in the entries a pole's eigenvectors see, ε‖A_bal‖
// on a dense matrix and the rounding of its own rows on a graded one. The
// factor 2 calibrates a Gaussian-dense pole with κ = 1 (n·β_j ≈ ρ₀/2) to
// ρ₀; on non-stiff census models n·β_j/ρ₀ has quartiles 0.38, 0.75, 2.3.
//
// foot is a pole's nearest axis point (z = 1 for λ = 0 on the unit
// circle), reach its drop radius n·ε·|foot| + ρ_j there, and hidden marks
// a pole within its reach of the axis, whose nearest axis point
// axisProbes drops: the data do not place it off the axis.
type modelPoles struct {
	evals    []complex128
	beta     []float64
	resolved []bool
	rho      []float64
	norm     float64
	rho0     float64
	foot     []complex128
	reach    []float64
	hidden   []bool
}

// axisProbes returns the deduplicated frequency-axis points the cut is
// judged at: the points beside the removed modes (removedModeProbes) and
// beside every eigenvalue of A (eigenAxisProbes), the fixed grid
// (gapProbePoints, with θ = π for a discrete model: matched and Tustin
// C2D map the zeros at infinity to z = −1, and ZOH sampling zeros approach
// it), and the decade grid (decadeAxisProbes). A point s within
// n·ε·|s| + ρ_j of eigenvalue λ_j is dropped: there sI − A is singular to
// the precision the solve reaches near that pole, both responses are
// rounding and the first-order bound does not hold. β_j alone is not the
// radius: on a cluster it exceeds ρ₀ and would span the points that refute
// a wrong cut, hence the min; below ρ₀ it is the radius, so the determined
// probes beside a slow pole of a graded stiff model are kept. False when
// the eigenvalues of A do not converge.
func axisProbes(sys *System, removed []*mat.Dense) ([]complex128, modelPoles, bool) {
	n, m, p := sys.Dims()
	discrete := sys.Dt != 0
	evals, beta, ok := eigenRoundingBounds(sys.A)
	if !ok {
		return nil, modelPoles{}, false
	}
	ne := len(evals)
	mp := modelPoles{evals: evals, beta: beta, resolved: make([]bool, ne), rho: make([]float64, ne),
		foot: make([]complex128, ne), reach: make([]float64, ne), hidden: make([]bool, ne)}
	norm := 0.0
	for _, a := range newBalancedRealization(sys, n, m, p).a {
		norm = math.Hypot(norm, a)
	}
	mp.norm, mp.rho0 = norm, float64(n)*eps()*norm
	resolved := true
	for k, ev := range evals {
		mp.resolved[k] = ev != 0 && beta[k] < cmplx.Abs(ev)
		mp.rho[k] = mp.rho0
		if mp.resolved[k] {
			mp.rho[k] = math.Min(mp.rho0, 2*float64(n)*beta[k])
		} else {
			resolved = false
		}
		foot := complex(0, imag(ev))
		if discrete {
			foot = 1
			if r := cmplx.Abs(ev); r > 0 {
				foot = ev / complex(r, 0)
			}
		}
		mp.foot[k] = foot
		mp.reach[k] = float64(n)*eps()*cmplx.Abs(foot) + mp.rho[k]
		mp.hidden[k] = cmplx.Abs(foot-ev) <= mp.reach[k]
	}
	var pts []complex128
	push := func(s complex128) {
		if !slices.Contains(pts, s) {
			pts = append(pts, s)
		}
	}
	if rp, ok := removedModeProbes(removed, discrete); ok {
		for _, s := range rp {
			push(s)
		}
	}
	for _, s := range eigenAxisProbes(evals, discrete) {
		push(s)
	}
	for _, s := range gapProbePoints(evals, discrete) {
		push(s)
	}
	if discrete {
		push(complex(-1, 0))
	}
	for _, s := range decadeAxisProbes(evals, resolved, discrete) {
		push(s)
	}
	kept := pts[:0]
	for _, s := range pts {
		rs := float64(n) * eps() * cmplx.Abs(s)
		drop := false
		for k, ev := range evals {
			if cmplx.Abs(s-ev) <= rs+mp.rho[k] {
				drop = true
				break
			}
		}
		if !drop {
			kept = append(kept, s)
		}
	}
	return kept, mp, true
}

// eigenRoundingBounds returns the eigenvalues of A with their
// componentwise rounding bounds β = ε·|y|ᵀ|A||x|/|yᴴx|, computed for a
// zero eigenvalue too (eigenvalueRoundingTerms skips it): 0 when the
// terms are, as for a zero column of A, so a computed zero that rounding
// does move keeps its bound. False when the eigenvalues do not converge.
func eigenRoundingBounds(A *mat.Dense) ([]complex128, []float64, bool) {
	pe, ok := newPoleEigen(A)
	if !ok {
		return nil, nil, false
	}
	beta := make([]float64, len(pe.poles))
	for k := range pe.poles {
		yax, yx := pe.coupling(A, k, k)
		if yax != 0 {
			beta[k] = eps() * yax / cmplx.Abs(yx)
		}
	}
	return pe.poles, beta, true
}

// poleWindows are disks of the frequency plane.
type poleWindows struct {
	centre []complex128
	radius []float64
	// cutPoles are the eigenvalues of the cut sharedPoleWindows judged.
	cutPoles []complex128
	// axisPole marks a window that shares a model pole within rounding of
	// the axis, axisShared such a model pole: its foot (modelPoles) within
	// β_G + β_R of it, β_R that of the window's cut pole.
	axisPole   []bool
	axisShared []bool
}

func (w poleWindows) contain(s complex128) bool {
	for i, c := range w.centre {
		if cmplx.Abs(s-c) <= w.radius[i] {
			return true
		}
	}
	return false
}

// strictlyJudged reports whether s lies inside some window but inside none
// that shares a pole within rounding of the axis (axisPole), so that
// judgeNearPoles judges it against the stored model.
func (w poleWindows) strictlyJudged(s complex128) bool {
	in := false
	for i, c := range w.centre {
		if cmplx.Abs(s-c) > w.radius[i] {
			continue
		}
		if w.axisPole[i] {
			return false
		}
		in = true
	}
	return in
}

// sharedPoleWindows returns, for every pole λ_R of red such that every
// model pole λ_G it shares (|λ_R − λ_G| ≤ β_R + β_G + ρ_G + ρ_R, each pole
// known to its rounding bound and its realization's resolution, ρ_R the
// cut's as ρ_G the model's) is determined to tol (β_G ≤ tol·max(|λ_G|,
// ρ_G)) or is, with λ_R, a rounding-level zero (|λ_G|, |λ_R| ≤ β_G ≤ ρ_G:
// a rotated integrator, which no realization places better than β_G), and
// is carried by the cut to that pole's resolution (β_R ≤ ρ_G), the disk of
// radius max(β_G, β_R)/tol about it (an axis window's below): there a
// rounding-level move of either realization's pole changes the response by
// more than tol, so no realization is determined to tol, however exactly
// the model carries the pole (a kept integrator that the rotated cut holds
// only to rounding).
// A model pole the cut drops has no window, so a cut that drops a pole
// within rounding of the axis is judged beside it. One shared pole that
// refuses forbids the window, so a cut pole smeared off a graded pole
// cannot borrow the floor of an unresolved neighbour, and a model pole
// known worse than its normwise resolution (β_G > ρ_G, an ill-conditioned
// slow pole of a stiff model) is no rounding-level zero: its β_G/tol spans
// the slow band where the stored model's response is the reference. A cut
// pole known worse than the model's own resolution of it (β_R > ρ_G) is
// the cut's own ill-conditioning, which
// the window must not excuse: a cut that smears a graded slow pole by its
// dense norm has a β_R/tol that spans the slow band where the model is
// determined (nse twins), and a clustered cut pole a β_R/tol that spans
// the cluster. Nor may a window reach another model pole λ_G′, clustered
// with the shared one or not: there the model's other dynamics are tested,
// not the shared pole, so the radius is capped at the distance |λ_R − λ_G′|
// to the nearest one. On a stiff model ε‖A‖ is O(1), so a slow pole can
// pass as a rounding-level zero, and an uncapped window would span the
// slow band where the cut is wrong. A window that shares a model pole
// within rounding of the axis, its foot within β_G + β_R of it, is an axis
// window (axisPole): the cut may carry that pole anywhere within rounding,
// and inside the window the stored model is not judged (judgeNearPoles).
// The exact values of one shared pole lie, to first order, within β_G +
// β_R of each computed one, so the computed λ_R and λ_G of a pole the cut
// keeps differ by at most β_G + β_R and the exact ones by at most 2(β_G +
// β_R); a distance δ the cut shows as a deviation δ/d at distance d
// relative to the pole's term. The radius of an axis window is therefore
// δ/tol, δ = β_G + β_R + min(|λ_R − λ_G|, β_G + β_R), the largest over its
// axis-shared poles: the deviation rounding explains is tol at its edge. A
// computed distance beyond β_G + β_R is a real move of the pole, which
// buys no wider window (HO6UP6 condition 4): beyond the edge the cut is
// judged, at the edge first (axisWindowEdges), where a cut whose exact
// pole lies further off deviates most.
// The windows of the other shared poles, exempt before, are now judged.
// The test reads the computed eigenvalues, whose error can exceed β, so it
// can err either way: a window wrongly exempted is only left uncertified,
// and one wrongly judged is checked against the stored model, which can
// only refuse a cut, so neither error certifies a point where the cut is
// wrong.
// False when the eigenvalues of red do not converge.
func sharedPoleWindows(mp modelPoles, red *System, tol float64) (poleWindows, bool) {
	w := poleWindows{axisShared: make([]bool, len(mp.evals))}
	nr, _, _ := red.Dims()
	if nr == 0 {
		return w, true
	}
	evR, betaR, ok := eigenRoundingBounds(red.A)
	if !ok {
		return w, false
	}
	w.cutPoles = evR
	rho0R := 0.0
	for _, a := range newBalancedRealization(red, nr, red.B.RawMatrix().Cols, red.C.RawMatrix().Rows).a {
		rho0R = math.Hypot(rho0R, a)
	}
	rho0R *= float64(nr) * eps()
	var with []int
	for i, lr := range evR {
		shared, ok, rad, lim, axis := false, true, betaR[i], math.Inf(1), false
		axisRad := 0.0
		rR := math.Min(rho0R, 2*float64(nr)*betaR[i])
		with = with[:0]
		for j, lg := range mp.evals {
			rj, bG := mp.rho[j], mp.beta[j]
			if cmplx.Abs(lr-lg) > betaR[i]+bG+rj+rR {
				continue
			}
			unresolved := cmplx.Abs(lg) <= bG && bG <= rj && cmplx.Abs(lr) <= bG
			if !(betaR[i] <= rj) || !(bG <= tol*math.Max(cmplx.Abs(lg), rj) || unresolved) {
				ok = false
				break
			}
			shared = true
			if mp.foot != nil && cmplx.Abs(mp.foot[j]-lg) <= bG+betaR[i] {
				with = append(with, j)
				axis = true
				axisRad = math.Max(axisRad, bG+betaR[i]+math.Min(cmplx.Abs(lr-lg), bG+betaR[i]))
			}
			rad = math.Max(rad, bG)
			for k, lo := range mp.evals {
				if k != j {
					lim = math.Min(lim, cmplx.Abs(lr-lo))
				}
			}
		}
		if ok && shared {
			w.centre = append(w.centre, lr)
			w.radius = append(w.radius, math.Min(math.Max(rad, axisRad)/tol, lim))
			w.axisPole = append(w.axisPole, axis)
			for _, j := range with {
				w.axisShared[j] = true
			}
		}
	}
	return w, true
}

// eigenAxisProbes returns the axis points beside every eigenvalue, one set
// per conjugate pair: on the imaginary axis j·Im λ and j·|λ|; on the unit
// circle λ/|λ| (z = 1 for λ = 0) and the corner e^{jθ}, θ = min(|ln λ|, π)
// (z = −1 for λ = 0): λ = e^{(−a+jb)T} has |ln λ| = |λ_c|·T, the discrete
// form of j·|λ|. A corner nearer than √ε to z = 1 is skipped: so near its
// pole the model is not determined to √ε.
func eigenAxisProbes(evals []complex128, discrete bool) []complex128 {
	var pts []complex128
	for _, ev := range evals {
		if cmplx.IsNaN(ev) || cmplx.IsInf(ev) || imag(ev) < 0 {
			continue
		}
		if !discrete {
			pts = append(pts, complex(0, imag(ev)), complex(0, cmplx.Abs(ev)))
			continue
		}
		r := cmplx.Abs(ev)
		if r == 0 {
			pts = append(pts, 1, -1)
			continue
		}
		pts = append(pts, ev/complex(r, 0))
		switch th := cmplx.Abs(cmplx.Log(ev)); {
		case th >= math.Pi:
			pts = append(pts, -1)
		case th >= math.Sqrt(eps()):
			pts = append(pts, cmplx.Rect(1, th))
		}
	}
	return pts
}

// decadeAxisProbes returns points a decade apart across the band the
// model's data can carry. Continuous: ω from lo to max|λ|, lo = ε·max|λ|
// (the rounding floor of gapProbePoints) unless every eigenvalue is
// resolved from zero, then max(min|λ|, ε·max|λ|): an unresolved eigenvalue
// can sit anywhere below the resolution. Discrete: θ from lo to π, lo =
// max(min|λ−1|, √ε), or √ε when an eigenvalue is unresolved.
func decadeAxisProbes(evals []complex128, resolved, discrete bool) []complex128 {
	lo, hi := math.Inf(1), 0.0
	for _, ev := range evals {
		w := cmplx.Abs(ev)
		if discrete {
			w = cmplx.Abs(ev - 1)
		}
		if w > 0 && !math.IsInf(w, 0) {
			lo = math.Min(lo, w)
			hi = math.Max(hi, w)
		}
	}
	var pts []complex128
	if discrete {
		lo = math.Max(lo, math.Sqrt(eps()))
		if !resolved || math.IsInf(lo, 0) {
			lo = math.Sqrt(eps())
		}
		for th := lo; th <= math.Pi; th *= 10 {
			pts = append(pts, cmplx.Rect(1, th))
		}
		return pts
	}
	if hi == 0 {
		return nil
	}
	if !resolved {
		lo = eps() * hi
	}
	lo = math.Max(lo, eps()*hi)
	for w := lo; w <= hi; w *= 10 {
		pts = append(pts, complex(0, w))
	}
	return pts
}

// removedModeProbes returns the frequency-axis points beside the
// eigenvalues of the blocks, one set per conjugate pair: z = λ/|λ| (z = 1
// for λ = 0) on the unit circle, and on the imaginary axis both the
// nearest point j·Im λ and j·|λ|. For a real λ the nearest point is s = 0,
// while (sI−A₂₂)⁻¹ keeps its size up to |s| ≈ |λ|, beyond which a
// realization inexact in its Markov parameters (CAᵏB) shows its error;
// j·|λ| sits at that corner. False when the eigenvalues do not converge or
// are not finite.
func removedModeProbes(blocks []*mat.Dense, discrete bool) ([]complex128, bool) {
	var pts []complex128
	push := func(s complex128) {
		if !slices.Contains(pts, s) {
			pts = append(pts, s)
		}
	}
	add := func(ev complex128) bool {
		if cmplx.IsNaN(ev) || cmplx.IsInf(ev) {
			return false
		}
		if imag(ev) < 0 {
			return true
		}
		if !discrete {
			push(complex(0, imag(ev)))
			push(complex(0, cmplx.Abs(ev)))
			return true
		}
		z := complex(1, 0)
		if r := cmplx.Abs(ev); r > 0 {
			z = ev / complex(r, 0)
		}
		push(z)
		return true
	}
	for _, blk := range blocks {
		raw := blk.RawMatrix()
		switch raw.Rows {
		case 1:
			if !add(complex(raw.Data[0], 0)) {
				return nil, false
			}
		case 2:
			a, b, c, d := raw.Data[0], raw.Data[1], raw.Data[raw.Stride], raw.Data[raw.Stride+1]
			h := (a + d) / 2
			disc := (a-d)*(a-d)/4 + b*c
			ev := []complex128{complex(h+math.Sqrt(disc), 0), complex(h-math.Sqrt(disc), 0)}
			if disc < 0 {
				ev = []complex128{complex(h, math.Sqrt(-disc))}
			}
			for _, e := range ev {
				if !add(e) {
					return nil, false
				}
			}
		default:
			var eig mat.Eigen
			if !eig.Factorize(blk, mat.EigenNone) {
				return nil, false
			}
			for _, e := range eig.Values(nil) {
				if !add(e) {
					return nil, false
				}
			}
		}
	}
	return pts, true
}

// cgeppFactor overwrites the n×n row-major x with its partial-pivoting LU
// factors, the unit lower factor's multipliers below the diagonal, and
// records the row swaps in piv; false at an exactly zero or NaN pivot.
func cgeppFactor(x []complex128, n int, piv []int) bool {
	for k := range n {
		p, best := k, cabs1(x[k*n+k])
		for i := k + 1; i < n; i++ {
			if v := cabs1(x[i*n+k]); v > best {
				p, best = i, v
			}
		}
		if !(best > 0) || math.IsInf(best, 0) {
			return false
		}
		piv[k] = p
		if p != k {
			rk, rp := x[k*n:(k+1)*n], x[p*n:(p+1)*n]
			for j := range rk {
				rk[j], rp[j] = rp[j], rk[j]
			}
		}
		ik := crecip(x[k*n+k])
		rk := x[k*n+k+1 : (k+1)*n]
		for i := k + 1; i < n; i++ {
			f := x[i*n+k] * ik
			if f == 0 {
				continue
			}
			x[i*n+k] = f
			ri := x[i*n+k+1 : (i+1)*n]
			for j, v := range rk {
				ri[j] -= f * v
			}
		}
	}
	return true
}

// cgeppSolve overwrites the n×r row-major y with X⁻¹y for the factors of
// cgeppFactor: PX = LU, the multipliers swapped along with their rows.
func cgeppSolve(x []complex128, n int, piv []int, y []complex128, r int) {
	for k := range n {
		if p := piv[k]; p != k {
			yk, yp := y[k*r:(k+1)*r], y[p*r:(p+1)*r]
			for j := range yk {
				yk[j], yp[j] = yp[j], yk[j]
			}
		}
	}
	for k := range n {
		yk := y[k*r : (k+1)*r]
		for i := k + 1; i < n; i++ {
			if f := x[i*n+k]; f != 0 {
				yi := y[i*r : (i+1)*r]
				for j, v := range yk {
					yi[j] -= f * v
				}
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		yi := y[i*r : (i+1)*r]
		for l := i + 1; l < n; l++ {
			if f := x[i*n+l]; f != 0 {
				yl := y[l*r : (l+1)*r]
				for j, v := range yl {
					yi[j] -= f * v
				}
			}
		}
		ii := crecip(x[i*n+i])
		for j := range yi {
			yi[j] *= ii
		}
	}
}

// cgeppSolveT overwrites the n×r row-major y with X⁻ᵀy for the factors of
// cgeppFactor: PX = LU gives Xᵀ = UᵀLᵀP.
func cgeppSolveT(x []complex128, n int, piv []int, y []complex128, r int) {
	for i := range n {
		yi := y[i*r : (i+1)*r]
		for l := range i {
			if f := x[l*n+i]; f != 0 {
				yl := y[l*r : (l+1)*r]
				for j, v := range yl {
					yi[j] -= f * v
				}
			}
		}
		ii := crecip(x[i*n+i])
		for j := range yi {
			yi[j] *= ii
		}
	}
	for i := n - 1; i >= 0; i-- {
		yi := y[i*r : (i+1)*r]
		for l := i + 1; l < n; l++ {
			if f := x[l*n+i]; f != 0 {
				yl := y[l*r : (l+1)*r]
				for j, v := range yl {
					yi[j] -= f * v
				}
			}
		}
	}
	for k := n - 1; k >= 0; k-- {
		if p := piv[k]; p != k {
			yk, yp := y[k*r:(k+1)*r], y[p*r:(p+1)*r]
			for j := range yk {
				yk[j], yp[j] = yp[j], yk[j]
			}
		}
	}
}
