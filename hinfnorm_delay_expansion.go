package controlsys

import (
	"math"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// highFrequencyExpansion expands a vector-valued gain (p = 1 or m = 1) with
// one class of commensurate delays τ_i = k_i·h about ω = ∞. With t = 1/ω,
// φ = ωh and u = 1/s = −j·t, G(jω) = G(t, φ) for
//
//	G(t, φ) = H11 + H12·Δ(I − H22·Δ)⁻¹·H21,  Hij = Dij + u·CiBj + u²·CiABj + …,  Δ = diag(e^{−j·k_i·φ}),
//
// analytic in t for each φ, so G = g0 + t·g1 + t²·g2 + R with g0 = G∞(φ),
// g1 = −j·G1, g2 = −G2, M = (I − D22Δ)⁻¹, L0 = M·D21 and
//
//	L1 = M·(C2B1 + C2B2·ΔL0),            G1 = C1B1 + C1B2·ΔL0 + D12·ΔL1,
//	L2 = M·(C2AB1 + C2AB2·ΔL0 + C2B2·ΔL1), G2 = C1AB1 + C1AB2·ΔL0 + C1B2·ΔL1 + D12·ΔL2.
//
// For complex |t| <= 1/w, ‖G − g0‖ <= E = tailGain(0, w), so by Cauchy's
// estimate ‖R‖ <= E·q³/(1 − q), q = t·w.
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
	e  *delayLFT
	hf *highFrequencyGain

	p11, p12, p21, p22 []complex128 // CiBj
	q11, q12, q21, q22 []complex128 // CiABj
	x, l1, l2          []complex128
	g0, g1, g2         []complex128

	poly    [][5]float64 // see sampleCircle
	samples []float64
}

// newHighFrequencyExpansion returns the expansion, or nil outside its scope:
// a matrix gain, no states, or more than one class of delays.
func newHighFrequencyExpansion(e *delayLFT, hf *highFrequencyGain) *highFrequencyExpansion {
	if hf == nil || hf.dim != 1 || (e.p != 1 && e.m != 1) || e.n == 0 {
		return nil
	}
	lft := e.sys.LFT
	nd, p, m := e.nd, e.p, e.m
	a := e.sys.A
	prod := func(c, b *mat.Dense, rows, cols int, withA bool) []complex128 {
		out := make([]complex128, rows*cols)
		if c == nil || b == nil || c.IsEmpty() || b.IsEmpty() {
			return out
		}
		var cb mat.Dense
		if withA {
			cb.Product(c, a, b)
		} else {
			cb.Mul(c, b)
		}
		for i := range rows {
			for j := range cols {
				out[i*cols+j] = complex(cb.At(i, j), 0)
			}
		}
		return out
	}
	x := &highFrequencyExpansion{
		e: e, hf: hf,
		p11: prod(e.sys.C, e.sys.B, p, m, false), p12: prod(e.sys.C, lft.B2, p, nd, false),
		p21: prod(lft.C2, e.sys.B, nd, m, false), p22: prod(lft.C2, lft.B2, nd, nd, false),
		q11: prod(e.sys.C, e.sys.B, p, m, true), q12: prod(e.sys.C, lft.B2, p, nd, true),
		q21: prod(lft.C2, e.sys.B, nd, m, true), q22: prod(lft.C2, lft.B2, nd, nd, true),
		x: make([]complex128, nd*m), l1: make([]complex128, nd*m), l2: make([]complex128, nd*m),
		g0: make([]complex128, p*m), g1: make([]complex128, p*m), g2: make([]complex128, p*m),
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

// at fills g0, g1 and g2 at phase φ; false where I − D22Δ is singular.
func (x *highFrequencyExpansion) at(phi float64) bool {
	h := x.hf
	if math.IsInf(h.at([]float64{phi}), 1) {
		return false
	}
	nd, p, m := h.e.nd, h.e.p, h.e.m
	copy(x.x, h.x)
	copy(x.g0, h.g)
	copy(x.l1, x.p21)
	addDelta(x.l1, x.p22, h.phase, x.x, nd, nd, m)
	cLUSolve(h.lu, h.piv, x.l1, nd, m)
	copy(x.l2, x.q21)
	addDelta(x.l2, x.q22, h.phase, x.x, nd, nd, m)
	addDelta(x.l2, x.p22, h.phase, x.l1, nd, nd, m)
	cLUSolve(h.lu, h.piv, x.l2, nd, m)
	copy(x.g1, x.p11)
	addDelta(x.g1, x.p12, h.phase, x.x, p, nd, m)
	addDelta(x.g1, h.d12, h.phase, x.l1, p, nd, m)
	copy(x.g2, x.q11)
	addDelta(x.g2, x.q12, h.phase, x.x, p, nd, m)
	addDelta(x.g2, x.p12, h.phase, x.l1, p, nd, m)
	addDelta(x.g2, h.d12, h.phase, x.l2, p, nd, m)
	for i := range x.g1 {
		x.g1[i] *= complex(0, -1)
		x.g2[i] = -x.g2[i]
	}
	return true
}

// model returns ‖g0 + t·g1 + t²·g2‖ at φ.
func (x *highFrequencyExpansion) model(phi, t float64) float64 {
	if !x.at(phi) {
		return math.Inf(1)
	}
	s := 0.0
	for i, g := range x.g0 {
		v := g + complex(t, 0)*(x.g1[i]+complex(t, 0)*x.g2[i])
		s += real(v)*real(v) + imag(v)*imag(v)
	}
	return math.Sqrt(s)
}

// coefficients returns S², a = Re⟨g0, g1⟩ and c = ‖g1‖² + 2·Re⟨g0, g2⟩ at φ.
func (x *highFrequencyExpansion) coefficients(phi float64) (s2, a, c float64) {
	if !x.at(phi) {
		return math.Inf(1), 0, 0
	}
	for i, g := range x.g0 {
		g1, g2 := x.g1[i], x.g2[i]
		s2 += real(g)*real(g) + imag(g)*imag(g)
		a += real(g)*real(g1) + imag(g)*imag(g1)
		c += real(g1)*real(g1) + imag(g1)*imag(g1) + 2*(real(g)*real(g2)+imag(g)*imag(g2))
	}
	return s2, a, c
}

// sampleCircle fills, at the 1024·k_max samples φ_i of the circle, the
// coefficients of ‖g0 + t·g1 + t²·g2‖² = Σ_j poly[i][j]·t^j, so the model
// is evaluated on the samples for any t without solving at φ_i again.
func (x *highFrequencyExpansion) sampleCircle() {
	if x.poly != nil {
		return
	}
	k := 1024 * int(x.hf.kmax)
	x.poly = make([][5]float64, k)
	step := 2 * math.Pi / float64(k)
	for i := range k {
		c := &x.poly[i]
		if !x.at(float64(i) * step) {
			c[0] = math.Inf(1)
			continue
		}
		for j, g := range x.g0 {
			g1, g2 := x.g1[j], x.g2[j]
			c[0] += real(g)*real(g) + imag(g)*imag(g)
			c[1] += 2 * (real(g)*real(g1) + imag(g)*imag(g1))
			c[2] += real(g1)*real(g1) + imag(g1)*imag(g1) + 2*(real(g)*real(g2)+imag(g)*imag(g2))
			c[3] += 2 * (real(g1)*real(g2) + imag(g1)*imag(g2))
			c[4] += real(g2)*real(g2) + imag(g2)*imag(g2)
		}
	}
}

// modelMax returns sup_φ ‖g0 + t·g1 + t²·g2‖: the largest sample of the
// model, each local maximum refined by golden-section search.
func (x *highFrequencyExpansion) modelMax(t float64) float64 {
	x.sampleCircle()
	if x.samples == nil {
		x.samples = make([]float64, len(x.poly))
	}
	for i, c := range x.poly {
		x.samples[i] = math.Sqrt(max(c[0]+t*(c[1]+t*(c[2]+t*(c[3]+t*c[4]))), 0))
	}
	_, vals := x.maxima(x.samples, func(phi float64) float64 { return x.model(phi, t) }, 0)
	if len(vals) == 0 {
		return math.Inf(1)
	}
	return slices.Max(vals)
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
// ‖G(jω)‖ <= lim·(1 + tol) is checked: sup_φ ‖g0 + t·g1 + t²·g2‖, maximised
// over φ as in onePhaseSup, plus the Cauchy bound on R stays below it at
// t = T·2^−i, i = 0…40, T = 1/W; past these samples the leading term κ·t²
// and the remainder O(t³) only shrink. This +Inf verdict rests on the
// expansion, not on a certificate over every t. ok is false when neither is
// decided.
func (x *highFrequencyExpansion) limitVerdict(lim, tol, wMin, wMax float64) (finite bool, W float64, ok bool) {
	x.sampleCircle()
	s := make([]float64, len(x.poly))
	for i, c := range x.poly {
		s[i] = math.Sqrt(c[0])
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
			q := t * w
			if !(E*q*q*q/(1-q) <= level) {
				return false
			}
			m, ok := sup[t]
			if !ok {
				m = x.modelMax(t)
				sup[t] = m
			}
			if !(m+E*q*q*q/(1-q) <= level) {
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

// radius returns the w < 0.9/T that minimises the Cauchy bound
// tailGain(0, w)·(Tw)³/(1 − Tw), or 0 when none is finite.
func (x *highFrequencyExpansion) radius(T float64) float64 {
	best, wBest := math.Inf(1), 0.0
	for w := x.e.normA*1.01 + 1e-9; w*T < 0.9; w *= 1.2 {
		q := w * T
		if b := x.e.tailGain(0, w) * q * q * q / (1 - q); b < best {
			best, wBest = b, w
		}
	}
	return wBest
}
