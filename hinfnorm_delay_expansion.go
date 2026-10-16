package controlsys

import (
	"cmp"
	"math"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// expansionOrder is the order N of the expansion of highFrequencyExpansion.
const expansionOrder = 3

// highFrequencyExpansion expands a gain with one class of commensurate delays τ_i = k_i·h about ω = ∞. With t = 1/ω,
// φ = ωh and u = 1/s = −j·t, G(jω) = G(t, φ) for
//
//	G(t, φ) = H11 + H12·Δ(I − H22·Δ)⁻¹·H21,  Hij = Σ_k u^k·Hij,k,  Hij,0 = Dij,  Hij,k = Ci·A^{k−1}·Bj,  Δ = diag(e^{−j·k_i·φ}),
//
// analytic in t for each φ, so G = Σ_{k<=N} t^k·g_k + R with g0 = G∞(φ) and
// g_k = (−j)^k·G_k, where, with M = (I − D22Δ)⁻¹ and Y = (I − H22Δ)⁻¹H21,
//
//	Y_k = M·(H21,k + Σ_{i=1..k} H22,i·Δ·Y_{k−i}),  G_k = H11,k + Σ_{i=0..k} H12,i·Δ·Y_{k−i}.
//
// For complex |t| <= 1/w, ‖G − g0‖ <= E = tailGain(0, w), so by Cauchy's
// estimate ‖R‖ <= E·q^{N+1}/(1 − q), q = t·w.
//
// Each passage of φ = ωh through a maximiser φ* of S = ‖g0‖ peaks at about
// sup_φ ‖g0 + t·g1 + t²·g2‖, so the finite peaks rise above lim = S(φ*) as
// t → 0 when a = Re⟨g0, g1⟩ > 0 at φ*, or, when a(φ*) = 0, when
// κ = ‖g1‖² + 2·Re⟨g0, g2⟩ + a′²/α > 0 with α = −½·(S²)″ and a′ = da/dφ at
// φ*: the maximum over δ = φ − φ* of S² + 2t·a = lim² − α·δ² + 2t·a′·δ + …
// is lim² + t²·a′²/α. For a real model G∞(−φ) = conj G∞(φ), so a is odd in
// φ: a maximiser off 0 and π comes with its mirror where a has the other
// sign, and a finite peak exceeds lim; at φ* = 0 or π, a = 0 and κ decides.
type highFrequencyExpansion struct {
	e      *delayLFT
	hf     *highFrequencyGain
	vector bool // p = 1 or m = 1, where ‖·‖ is the Euclidean norm
	buf    []complex128

	h11, h12, h21, h22 [expansionOrder + 1][]complex128 // Hij,k, k >= 1; nil when zero
	y, g               [expansionOrder + 1][]complex128
	yZero              [expansionOrder + 1]bool // Y_k = 0 for every φ

	// see sampleCircle
	poly     []float64
	byS      []int
	polyMax  [polyLen]float64
	singular bool
	vals     []float64
	stamp    []int
	epoch    int
}

// newHighFrequencyExpansion returns the expansion, or nil outside its scope:
// no states, or more than one class of delays.
func newHighFrequencyExpansion(e *delayLFT, hf *highFrequencyGain) *highFrequencyExpansion {
	if hf == nil || hf.dim != 1 || e.n == 0 {
		return nil
	}
	lft := e.sys.LFT
	nd, p, m := e.nd, e.p, e.m
	x := &highFrequencyExpansion{e: e, hf: hf, vector: p == 1 || m == 1, buf: make([]complex128, p*m)}
	terms := func(dst *[expansionOrder + 1][]complex128, c, b *mat.Dense, rows, cols int) {
		var ab, next mat.Dense
		empty := c == nil || b == nil || c.IsEmpty() || b.IsEmpty()
		if !empty {
			ab.CloneFrom(b)
		}
		for k := 1; k <= expansionOrder; k++ {
			out := make([]complex128, rows*cols)
			if !empty {
				var cb mat.Dense
				cb.Mul(c, &ab)
				for i := range rows {
					for j := range cols {
						out[i*cols+j] = complex(cb.At(i, j), 0)
					}
				}
				next.Mul(e.sys.A, &ab)
				ab.CloneFrom(&next)
			}
			if slices.ContainsFunc(out, func(v complex128) bool { return v != 0 }) {
				dst[k] = out
			}
		}
	}
	terms(&x.h11, e.sys.C, e.sys.B, p, m)
	terms(&x.h12, e.sys.C, lft.B2, p, nd)
	terms(&x.h21, lft.C2, e.sys.B, nd, m)
	terms(&x.h22, lft.C2, lft.B2, nd, nd)
	for k := range expansionOrder + 1 {
		x.y[k], x.g[k] = make([]complex128, nd*m), make([]complex128, p*m)
		x.yZero[k] = k > 0 && x.h21[k] == nil
		for i := 1; i <= k; i++ {
			x.yZero[k] = x.yZero[k] && (x.h22[i] == nil || x.yZero[k-i])
		}
	}
	return x
}

// addDelta adds c·Δ·v to dst for the r×nd c and nd×m v.
func addDelta(dst, c, phase, v []complex128, r, nd, m int) {
	for i := range r {
		for k := range nd {
			f := c[i*nd+k] * phase[k]
			if f == 0 {
				continue
			}
			for j := range m {
				dst[i*m+j] += f * v[k*m+j]
			}
		}
	}
}

// at fills g_k, k <= N, at phase φ; false where I − D22Δ is singular.
func (x *highFrequencyExpansion) at(phi float64) bool {
	h := x.hf
	if math.IsInf(h.at([]float64{phi}), 1) {
		return false
	}
	nd, p, m := h.e.nd, h.e.p, h.e.m
	copy(x.y[0], h.x)
	copy(x.g[0], h.g)
	rot := complex(1, 0)
	for k := 1; k <= expansionOrder; k++ {
		y, g := x.y[k], x.g[k]
		if !x.yZero[k] {
			clear(y)
			copy(y, x.h21[k])
			for i := 1; i <= k; i++ {
				if x.h22[i] != nil && !x.yZero[k-i] {
					addDelta(y, x.h22[i], h.phase, x.y[k-i], nd, nd, m)
				}
			}
			cLUSolve(h.lu, h.piv, y, nd, m)
		}
		clear(g)
		copy(g, x.h11[k])
		if !x.yZero[k] {
			addDelta(g, h.d12, h.phase, y, p, nd, m)
		}
		for i := 1; i <= k; i++ {
			if x.h12[i] != nil && !x.yZero[k-i] {
				addDelta(g, x.h12[i], h.phase, x.y[k-i], p, nd, m)
			}
		}
		rot *= complex(0, -1)
		for i := range g {
			g[i] *= rot
		}
	}
	return true
}

// model returns σ_max(Σ_{k<=N} t^k·g_k) at φ.
func (x *highFrequencyExpansion) model(phi, t float64) float64 {
	if !x.at(phi) {
		return math.Inf(1)
	}
	return x.sum(t)
}

// sum returns σ_max(Σ_{k<=N} t^k·g_k) at the phase of the last at.
func (x *highFrequencyExpansion) sum(t float64) float64 {
	tc, s := complex(t, 0), 0.0
	for i := range x.g[0] {
		v := x.g[expansionOrder][i]
		for k := expansionOrder - 1; k >= 0; k-- {
			v = v*tc + x.g[k][i]
		}
		x.buf[i] = v
		s += real(v)*real(v) + imag(v)*imag(v)
	}
	if x.vector {
		return math.Sqrt(s)
	}
	sv, err := x.e.svd.maximumFromFlat(x.buf, 0, x.e.p, x.e.m)
	if err != nil {
		return math.NaN()
	}
	return sv
}

// coefficients returns, at φ, S², a = Re⟨g0, g1⟩ and
// c = ‖g1‖² + 2·Re⟨g0, g2⟩, the coefficients of t and t² in ‖g0 + t·g1 +
// t²·g2‖², for a vector gain. For a matrix gain a = ½·dσ_max²/dt at t = 0,
// by a five-point difference, and c = ‖g1‖_F², a scale only.
func (x *highFrequencyExpansion) coefficients(phi float64) (s2, a, c float64) {
	if !x.at(phi) {
		return math.Inf(1), 0, 0
	}
	if !x.vector {
		s := x.sum(0)
		for _, g := range x.g[1] {
			c += real(g)*real(g) + imag(g)*imag(g)
		}
		if c == 0 || s == 0 {
			return s * s, 0, c
		}
		d := 1e-3 * s / math.Sqrt(c)
		l := func(t float64) float64 { v := x.sum(t); return v * v }
		a = (8*(l(d)-l(-d)) - (l(2*d) - l(-2*d))) / (24 * d)
		return s * s, a, c
	}
	for i, g := range x.g[0] {
		g1, g2 := x.g[1][i], x.g[2][i]
		s2 += real(g)*real(g) + imag(g)*imag(g)
		a += real(g)*real(g1) + imag(g)*imag(g1)
		c += real(g1)*real(g1) + imag(g1)*imag(g1) + 2*(real(g)*real(g2)+imag(g)*imag(g2))
	}
	return s2, a, c
}

// polyLen is the number of coefficients of ‖Σ_{k<=N} t^k·g_k‖² in t.
const polyLen = 2*expansionOrder + 1

// sampleCircle fills, at the 1024·k_max samples φ_i of the circle, the
// coefficients c_n = Σ_{i+j=n} Re⟨g_i, g_j⟩ of the squared model
// Σ_n c_n·t^n, so the model is evaluated on the samples for any t without
// solving at φ_i again.
func (x *highFrequencyExpansion) sampleCircle() {
	if x.poly != nil {
		return
	}
	k := 1024 * int(x.hf.kmax)
	x.poly = make([]float64, k*polyLen)
	step := 2 * math.Pi / float64(k)
	for i := range k {
		c := x.poly[i*polyLen : (i+1)*polyLen]
		if !x.at(float64(i) * step) {
			c[0] = math.Inf(1)
			x.singular = true
			continue
		}
		if !x.vector {
			v := x.sum(0)
			c[0] = v * v
			continue
		}
		x.squared(c)
		for n := 1; n < polyLen; n++ {
			x.polyMax[n] = max(x.polyMax[n], math.Abs(c[n]))
		}
	}
	x.byS = make([]int, k)
	for i := range k {
		x.byS[i] = i
	}
	slices.SortFunc(x.byS, func(a, b int) int { return cmp.Compare(x.poly[b*polyLen], x.poly[a*polyLen]) })
	x.vals, x.stamp = make([]float64, k), make([]int, k)
}

// squared adds to c the coefficients c_n = Σ_{i+j=n} Re⟨g_i, g_j⟩ of the
// squared model at the phase of the last at.
func (x *highFrequencyExpansion) squared(c []float64) {
	for r := range x.g[0] {
		for a := range expansionOrder + 1 {
			ga := x.g[a][r]
			for b := range expansionOrder + 1 {
				gb := x.g[b][r]
				c[a+b] += real(ga)*real(gb) + imag(ga)*imag(gb)
			}
		}
	}
}

// overInterval returns Σ_n c_n·t_n with t_n = t2ⁿ where c_n > 0, else t1ⁿ,
// which bounds Σ_n c_n·tⁿ for every t ∈ [t1, t2], 0 <= t1 <= t2.
func overInterval(c []float64, t1, t2 float64) float64 {
	v, p1, p2 := c[0], 1.0, 1.0
	for _, cn := range c[1:] {
		p1, p2 = p1*t1, p2*t2
		if cn > 0 {
			v += cn * p2
		} else {
			v += cn * p1
		}
	}
	return v
}

// sampleModel returns the bound overInterval of the squared model at sample
// i for t ∈ [t1, t2], square-rooted and cached for this epoch.
func (x *highFrequencyExpansion) sampleModel(i int, t1, t2 float64) float64 {
	if x.stamp[i] == x.epoch {
		return x.vals[i]
	}
	v := overInterval(x.poly[i*polyLen:(i+1)*polyLen], t1, t2)
	x.stamp[i], x.vals[i] = x.epoch, math.Sqrt(max(v, 0))
	return x.vals[i]
}

// modelMax returns sup_φ ‖Σ_{k<=N} t^k·g_k‖; see modelMaxOver.
func (x *highFrequencyExpansion) modelMax(t float64) float64 {
	return x.modelMaxOver(t, t)
}

// modelMaxOver returns sup over φ and t ∈ [t1, t2] of the bound
// overInterval of the squared model, square-rooted, which for t1 = t2 is
// the model ‖Σ_{k<=N} t^k·g_k‖ itself: the largest sample, local maxima
// within a factor 1 − 1e-3 of it refined by golden-section search as in
// maxima. The bound at sample i is at most c_0,i + Σ_{n>=1} max_j |c_n,j|·t2^n,
// so walking the samples by decreasing c_0 stops once that falls 1% below
// the largest sample. +Inf for a matrix gain, whose σ_max has no such
// polynomial.
func (x *highFrequencyExpansion) modelMaxOver(t1, t2 float64) float64 {
	x.sampleCircle()
	if x.singular || !x.vector {
		return math.Inf(1)
	}
	k := len(x.byS)
	x.epoch++
	slack := 0.0
	for n := polyLen - 1; n >= 1; n-- {
		slack = (slack + x.polyMax[n]) * t2
	}
	top, last := 0.0, k
	for j, i := range x.byS {
		if math.Sqrt(max(x.poly[i*polyLen]+slack, 0)) < top*(1-1e-2) {
			last = j
			break
		}
		top = max(top, x.sampleModel(i, t1, t2))
	}
	f := func(phi float64) float64 { return x.model(phi, t1) }
	if t1 != t2 {
		var c [polyLen]float64
		f = func(phi float64) float64 {
			if !x.at(phi) {
				return math.Inf(1)
			}
			c = [polyLen]float64{}
			x.squared(c[:])
			return math.Sqrt(max(overInterval(c[:], t1, t2), 0))
		}
	}
	step := 2 * math.Pi / float64(k)
	var idx []int
	var est []float64
	for _, i := range x.byS[:last] {
		v := x.vals[i]
		y0, y2 := x.sampleModel((i+k-1)%k, t1, t2), x.sampleModel((i+1)%k, t1, t2)
		if v < y0 || v < y2 {
			continue
		}
		e := v
		if d := 2*v - y0 - y2; d > 0 {
			e += (y0 - y2) * (y0 - y2) / (8 * d)
		}
		idx, est = append(idx, i), append(est, e)
	}
	cut := slices.Max(est) * (1 - 1e-3)
	best := top
	for j, i := range idx {
		if est[j] < cut {
			continue
		}
		phi := float64(i) * step
		_, g := goldenMax(phi-step, phi+step, phi, x.vals[i], f)
		best = max(best, g)
	}
	return best
}

// tailBelow reports whether ‖G(jω)‖ <= level for ω >= 1/T: by
// modelMaxOver plus the Cauchy bound on the remainder over the intervals
// t ∈ [T·2^−(i+1), T·2^−i], i < 40, and [0, T·2^−40], which cover (0, T].
// Like limitVerdict, it rests on sampling the circle of φ.
func (x *highFrequencyExpansion) tailBelow(level, T float64) bool {
	w := x.radius(T)
	if w == 0 {
		return false
	}
	E := x.e.tailGain(0, w)
	for i := 0; i <= 40; i++ {
		t2 := T * math.Ldexp(1, -i)
		t1 := t2 / 2
		if i == 40 {
			t1 = 0
		}
		if !(x.modelMaxOver(t1, t2)+x.remainder(E, t2*w) <= level) {
			return false
		}
	}
	return true
}

// maxima returns the local maxima of φ ↦ f(φ) on the circle within a factor
// 1 − rel of the largest, from samples f(2π·i/k), i < k, refined by
// golden-section search. Only the sampled maxima whose parabolic
// interpolation lies within a factor 1 − max(rel, 1e-3) of the largest are
// refined; at this sampling density the interpolation error is far smaller.
func (x *highFrequencyExpansion) maxima(samples []float64, f func(float64) float64, rel float64) (phis, vals []float64) {
	k := len(samples)
	step := 2 * math.Pi / float64(k)
	var idx []int
	var est []float64
	for i, v := range samples {
		y0, y2 := samples[(i+k-1)%k], samples[(i+1)%k]
		if v < y0 || v < y2 {
			continue
		}
		e := v
		if d := 2*v - y0 - y2; d > 0 {
			e += (y0 - y2) * (y0 - y2) / (8 * d)
		}
		idx, est = append(idx, i), append(est, e)
	}
	if len(idx) == 0 {
		return nil, nil
	}
	cut := slices.Max(est) * (1 - max(rel, 1e-3))
	for j, i := range idx {
		if est[j] < cut {
			continue
		}
		t := float64(i) * step
		w, g := goldenMax(t-step, t+step, t, samples[i], f)
		phis, vals = append(phis, w), append(vals, g)
	}
	top := slices.Max(vals)
	var outP, outV []float64
	for i, v := range vals {
		if v >= top*(1-rel) {
			outP, outV = append(outP, phis[i]), append(outV, v)
		}
	}
	return outP, outV
}

// limitVerdict decides whether the finite peaks of ‖G(jω)‖ exceed the limit
// lim = sup S. finite reports that they do, from a > 0 or κ > 0 at a
// maximiser of S, provided the predicted excess at wMin, a/(lim·wMin) or
// κ/(2·lim·wMin²), is above tol·lim, so that a sample can show it. Otherwise,
// with κ < 0 at every maximiser, W <= wMax is a frequency past which
// ‖G(jω)‖ <= lim·(1 + tol) is checked: sup_φ ‖Σ_{k<=N} t^k·g_k‖, maximised
// over φ as in onePhaseSup, plus the Cauchy bound on R stays below it at
// t = T·2^−i, i = 0…40, T = 1/W; past these samples the leading term κ·t²
// and the remainder O(t^{N+1}) only shrink. This +Inf verdict rests on the
// expansion, not on a certificate over every t. ok is false when neither is
// decided, and for a matrix gain whenever a ≈ 0 at every maximiser, since κ
// and the model maximum are only formed for a vector gain.
func (x *highFrequencyExpansion) limitVerdict(lim, tol, wMin, wMax float64) (finite bool, W float64, ok bool) {
	x.sampleCircle()
	s := make([]float64, len(x.poly)/polyLen)
	for i := range s {
		s[i] = math.Sqrt(x.poly[i*polyLen])
	}
	phis, _ := x.maxima(s, func(phi float64) float64 {
		s2, _, _ := x.coefficients(phi)
		return math.Sqrt(s2)
	}, 1e-9)
	if len(phis) == 0 {
		return false, 0, false
	}
	allNegative := true
	for _, phi := range phis {
		s2, a, c := x.coefficients(phi)
		const ha = 1e-5
		_, aPlus, _ := x.coefficients(phi + ha)
		_, aMinus, _ := x.coefficients(phi - ha)
		ap := (aPlus - aMinus) / (2 * ha)
		// φ* is located to about 1e-8 only, so a(φ*) carries an error of
		// about a′·1e-8 that a zero of a must absorb.
		if eps := 1e-6 * (math.Abs(ap) + math.Sqrt(s2*max(c, 1))); a > eps {
			return a/(lim*wMin) > tol*lim, 0, a/(lim*wMin) > tol*lim
		} else if a < -eps {
			continue
		}
		if !x.vector {
			return false, 0, false
		}
		const hs = 1e-4
		sp, _, _ := x.coefficients(phi + hs)
		sm, _, _ := x.coefficients(phi - hs)
		alpha := -(sp - 2*s2 + sm) / (2 * hs * hs)
		if !(alpha > 0) {
			return false, 0, false
		}
		kappa := c + ap*ap/alpha
		switch {
		case kappa > 1e-6*s2:
			seen := kappa/(2*lim*wMin*wMin) > tol*lim
			return seen, 0, seen
		case !(kappa < -1e-6*s2):
			allNegative = false
		}
	}
	if !allNegative {
		return false, 0, false
	}
	level := lim * (1 + tol)
	// T halves, so the t of successive T coincide exactly.
	sup := map[float64]float64{}
	holds := func(T float64) bool {
		w := x.radius(T)
		if w == 0 {
			return false
		}
		E := x.e.tailGain(0, w)
		for t := T; t > T*0x1p-40; t /= 2 {
			r := x.remainder(E, t*w)
			if !(r <= level) {
				return false
			}
			m, ok := sup[t]
			if !ok {
				m = x.modelMax(t)
				sup[t] = m
			}
			if !(m+r <= level) {
				return false
			}
		}
		return true
	}
	for T := 1 / max(x.e.normA*4+1, 1); T*wMax >= 1; T /= 2 {
		if holds(T) {
			return false, 1 / T, true
		}
	}
	return false, 0, false
}

// remainder returns the Cauchy bound E·q^{N+1}/(1 − q) on the remainder of
// the expansion of order N.
func (x *highFrequencyExpansion) remainder(E, q float64) float64 {
	return E * math.Pow(q, expansionOrder+1) / (1 - q)
}

// radius returns the w < 0.9/T that minimises the Cauchy bound remainder at
// t = T, or 0 when none is finite.
func (x *highFrequencyExpansion) radius(T float64) float64 {
	best, wBest := math.Inf(1), 0.0
	for w := x.e.normA*1.01 + 1e-9; w*T < 0.9; w *= 1.2 {
		if b := x.remainder(x.e.tailGain(0, w), w*T); b < best {
			best, wBest = b, w
		}
	}
	return wBest
}
