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
// Across a resolved spectral gap the count is not sufficient, and
// frequencyAxisCutAcrossGap adds one per side of the gap.
// False also when the eigenvalues of A or of red do not converge.
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
	full, cut := newBoundedResponse(sys, pts), newBoundedResponse(red, pts)
	defer full.release()
	defer cut.release()
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
		if nr > 0 {
			evR, _, ok := eigenRoundingBounds(red.A)
			if !ok {
				return false
			}
			count(evR)
		}
	}
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
	if verified < n+nr {
		return false
	}
	if sided && (slow < needSlow || fast < needFast) {
		return false
	}
	return accept()
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
// with a singular cut rejects.
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
	verdict := probeVerified
	for i, v := range g {
		gc, rc := full.comp[o+i], cut.comp[o+i]
		diff := cmplx.Abs(v - r[i])
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
	for i, v := range g {
		d := cmplx.Abs(v-r[i]) - full.comp[o+i] - cut.comp[o+i]
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
	for i, v := range g {
		if cmplx.Abs(v-r[i])-gc[i]-rc[i] > ts {
			return true
		}
	}
	return false
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
	ok, dev := true, math.Inf(-1)
	for i, v := range g {
		d, b := cmplx.Abs(v-r[i]), full.ver[o+i]+cut.ver[o+i]
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
type modelPoles struct {
	evals    []complex128
	beta     []float64
	resolved []bool
	rho      []float64
	norm     float64
	rho0     float64
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
	mp := modelPoles{evals: evals, beta: beta, resolved: make([]bool, len(evals)), rho: make([]float64, len(evals))}
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
}

func (w poleWindows) contain(s complex128) bool {
	for i, c := range w.centre {
		if cmplx.Abs(s-c) <= w.radius[i] {
			return true
		}
	}
	return false
}

// sharedPoleWindows returns, for every pole λ_R of red such that every
// model pole λ_G it shares (|λ_R − λ_G| ≤ β_R + β_G + ρ_G + ρ_R, each pole
// known to its rounding bound and its realization's resolution, ρ_R the
// cut's as ρ_G the model's) is determined to tol (β_G ≤ tol·max(|λ_G|,
// ρ_G)) or is, with λ_R, a rounding-level zero (|λ_G|, |λ_R| ≤ β_G ≤ ρ_G:
// a rotated integrator, which no realization places better than β_G), and
// is carried by the cut to that pole's resolution (β_R ≤ ρ_G), the disk of
// radius max(β_G, β_R)/tol about it: there a rounding-level move of either
// realization's pole changes the response by more than tol, so no
// realization is determined to tol, however exactly the model carries the
// pole (a kept integrator that the rotated cut holds only to rounding).
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
// slow band where the cut is wrong. False when the eigenvalues of red do not
// converge.
func sharedPoleWindows(mp modelPoles, red *System, tol float64) (poleWindows, bool) {
	var w poleWindows
	nr, _, _ := red.Dims()
	if nr == 0 {
		return w, true
	}
	evR, betaR, ok := eigenRoundingBounds(red.A)
	if !ok {
		return w, false
	}
	rho0R := 0.0
	for _, a := range newBalancedRealization(red, nr, red.B.RawMatrix().Cols, red.C.RawMatrix().Rows).a {
		rho0R = math.Hypot(rho0R, a)
	}
	rho0R *= float64(nr) * eps()
	for i, lr := range evR {
		shared, ok, rad, lim := false, true, betaR[i], math.Inf(1)
		rR := math.Min(rho0R, 2*float64(nr)*betaR[i])
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
			rad = math.Max(rad, bG)
			for k, lo := range mp.evals {
				if k != j {
					lim = math.Min(lim, cmplx.Abs(lr-lo))
				}
			}
		}
		if ok && shared {
			w.centre = append(w.centre, lr)
			w.radius = append(w.radius, math.Min(rad/tol, lim))
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
