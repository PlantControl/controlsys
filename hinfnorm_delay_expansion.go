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
	pair   *gramPair  // matrix gain, set on first use
	theta  [1]float64 // φ of at

	h11, h12, h21, h22 [expansionOrder + 1][]complex128 // Hij,k, k >= 1; nil when zero
	y, g               [expansionOrder + 1][]complex128
	yZero              [expansionOrder + 1]bool // Y_k = 0 for every φ

	// see sampleCircle
	poly     []float64
	byS      []int
	polyMax  [polyLen]float64
	gMax     [expansionOrder + 1]float64 // matrix gain: largest ‖g_k‖_F over the samples
	sg       [][]complex128              // matrix gain: g_0…g_N at sample i, set on first use
	sn       [][expansionOrder + 1]float64
	flat     []complex128
	flatN    [expansionOrder + 1]float64
	gap      [expansionOrder + 1]float64 // secantGap of the current modelMaxOver
	sup0     float64                     // modelMax(0), 0 until set
	singular bool
	vals     []float64
	stamp    []int
	epoch    int

	torusG        []complex128 // g_0…g_N at the cells of torusGrid, see torusModelMax
	torusSingular bool

	kappa clusterVerdict // last torusClusterKappa, shared by risesPast and fallsPast
}

// clusterVerdict caches torusClusterKappa at theta for eps.
type clusterVerdict struct {
	theta  []float64
	eps    float64
	lo, hi float64
	ok     bool
}

// newHighFrequencyExpansion returns the expansion, or nil outside its scope:
// no states, or more than one class of delays.
func newHighFrequencyExpansion(e *delayLFT, hf *highFrequencyGain) *highFrequencyExpansion {
	if hf == nil || hf.dim != 1 {
		return nil
	}
	return newTorusExpansion(e, hf)
}

// newTorusExpansion returns the expansion on the torus of phases of hf, nil
// without states. For dim > 1 only atTheta and coefficientsAt apply; the
// rest samples the circle of one phase.
func newTorusExpansion(e *delayLFT, hf *highFrequencyGain) *highFrequencyExpansion {
	if hf == nil || e.n == 0 {
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
	x.theta[0] = phi
	return x.atTheta(x.theta[:])
}

// atTheta is at for the phases θ of the torus.
func (x *highFrequencyExpansion) atTheta(theta []float64) bool {
	h := x.hf
	if math.IsInf(h.at(theta), 1) {
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

// coefficients returns, at φ, S², a and c, the coefficients of t and t² in
// σ_max(Σ_{k<=N} t^k·g_k)² = S² + 2t·a + t²·c + O(t³): for a vector gain
// a = Re⟨g0, g1⟩ and c = ‖g1‖² + 2·Re⟨g0, g2⟩, for a matrix gain λ1/2 and
// λ2 of gramPair. scale is c for a vector gain and ‖g1‖_F² for a matrix
// gain, a size for the tolerance on a. ok is false where c is not formed:
// for a matrix gain whose σ_max(g0) is repeated, a then is ½·dσ_max²/dt at
// t = 0+ by a one-sided four-point difference with step d = 10⁻³·min_k
// (S/‖g_k‖)^{1/k}, the largest slope of its splitting.
func (x *highFrequencyExpansion) coefficients(phi float64) (s2, a, c, scale float64, ok bool) {
	x.theta[0] = phi
	return x.coefficientsAt(x.theta[:])
}

// coefficientsAt is coefficients at the phases θ of the torus.
func (x *highFrequencyExpansion) coefficientsAt(theta []float64) (s2, a, c, scale float64, ok bool) {
	if !x.atTheta(theta) {
		return math.Inf(1), 0, 0, 0, false
	}
	if !x.vector {
		if x.pair == nil {
			x.pair = newGramPair(x.e.p, x.e.m)
		}
		for _, g := range x.g[1] {
			scale += real(g)*real(g) + imag(g)*imag(g)
		}
		l0, l1, l2, _, exact := x.pair.expand(x.g[0], x.g[1], x.g[2])
		if exact {
			return l0, l1 / 2, l2, scale, true
		}
		s := x.sum(0)
		if scale == 0 || s == 0 {
			return s * s, 0, 0, scale, false
		}
		d := s / math.Sqrt(scale)
		for k := 2; k <= expansionOrder; k++ {
			var n float64
			for _, g := range x.g[k] {
				n += real(g)*real(g) + imag(g)*imag(g)
			}
			if n > 0 {
				d = min(d, math.Pow(s/math.Sqrt(n), 1/float64(k)))
			}
		}
		d *= 1e-3
		l := func(t float64) float64 { v := x.sum(t); return v * v }
		a = (-11*s*s + 18*l(d) - 9*l(2*d) + 2*l(3*d)) / (12 * d)
		return s * s, a, 0, scale, false
	}
	for i, g := range x.g[0] {
		g1, g2 := x.g[1][i], x.g[2][i]
		s2 += real(g)*real(g) + imag(g)*imag(g)
		a += real(g)*real(g1) + imag(g)*imag(g1)
		c += real(g1)*real(g1) + imag(g1)*imag(g1) + 2*(real(g)*real(g2)+imag(g)*imag(g2))
	}
	return s2, a, c, c, true
}

// clusterKappa is clusterKappaAt at the phase φ of one class.
func (x *highFrequencyExpansion) clusterKappa(phi, tolMu float64) (lo, hi, mu1, reach float64, ok bool) {
	return x.clusterKappaAt([]float64{phi}, tolMu)
}

// clusterKappaAt returns bounds lo <= κ <= hi on κ = sup_s λ_max(K(s)),
// s ∈ R^dim, of gramPair.clusterPencilN at a maximiser θ of S of a matrix
// gain where σ_max(g0) is repeated: the passage maxima have σ_max² − lim²
// ≈ κ·t² at most (for a simple σ_max K(s) = c + 2s·∇a − sᵀ·α·s, whose sup
// is c + ∇aᵀ·α⁻¹·∇a). It also returns μ1 (2a of the top branches) and the
// reach of the pencil. ∂_i g0, ∂_i g1 and ∂_i∂_j g0 in θ come from
// five-point stencils at h = 10⁻¹/k_max·2^−i, i = 0…30, along e_i and,
// for the mixed ∂_i∂_j g0, along e_i + e_j; κ is taken once its change
// over a halving has shrunk 8-fold twice running (the O(h⁴) regime, past
// the width of a sharp peak), from the h with the least change, which
// widens [lo, hi] twice over. tolMu is the tolerance on μ for the top
// branches. ok is false where I − D22Δ is singular nearby, the pencil is
// not concave in s, or κ does not settle.
func (x *highFrequencyExpansion) clusterKappaAt(theta []float64, tolMu float64) (lo, hi, mu1, reach float64, ok bool) {
	if x.pair == nil {
		x.pair = newGramPair(x.e.p, x.e.m)
	}
	nd, pm := len(theta), len(x.g[0])
	var f [2][5][]complex128
	for k := range f {
		for j := range f[k] {
			f[k][j] = make([]complex128, pm)
		}
	}
	d0, d1, dd0 := make([][]complex128, nd), make([][]complex128, nd), make([][][]complex128, nd)
	for i := range nd {
		d0[i], d1[i], dd0[i] = make([]complex128, pm), make([]complex128, pm), make([][]complex128, i+1)
		for j := range i + 1 {
			dd0[i][j] = make([]complex128, pm)
		}
	}
	th := make([]float64, nd)
	// stencil fills f with g0 and g1 at θ + (j − 2)·h·(e_a + e_b), j < 5.
	stencil := func(a, b int, h float64) bool {
		for j := range 5 {
			copy(th, theta)
			th[a] += float64(j-2) * h
			if b != a {
				th[b] += float64(j-2) * h
			}
			if !x.atTheta(th) {
				return false
			}
			copy(f[0][j], x.g[0])
			copy(f[1][j], x.g[1])
		}
		return true
	}
	second := func(dst []complex128, h float64) {
		for l := range pm {
			dst[l] = (-f[0][0][l] + 16*f[0][1][l] - 30*f[0][2][l] + 16*f[0][3][l] - f[0][4][l]) / complex(12*h*h, 0)
		}
	}
	if !x.atTheta(theta) {
		return 0, 0, 0, 0, false
	}
	s0 := x.sum(0)
	s0 *= s0
	prev, prevDiff, best := math.NaN(), math.Inf(1), math.Inf(1)
	settled := 0
	for i := range 31 {
		h := math.Ldexp(0.1/x.hf.kmax, -i)
		for a := range nd {
			if !stencil(a, a, h) {
				return 0, 0, 0, 0, false
			}
			for l := range pm {
				d0[a][l] = (f[0][0][l] - 8*f[0][1][l] + 8*f[0][3][l] - f[0][4][l]) / complex(12*h, 0)
				d1[a][l] = (f[1][0][l] - 8*f[1][1][l] + 8*f[1][3][l] - f[1][4][l]) / complex(12*h, 0)
			}
			second(dd0[a][a], h)
		}
		for a := range nd {
			for b := range a {
				if !stencil(a, b, h) {
					return 0, 0, 0, 0, false
				}
				second(dd0[a][b], h)
				for l := range pm {
					dd0[a][b][l] = (dd0[a][b][l] - dd0[a][a][l] - dd0[b][b][l]) / 2
				}
			}
		}
		if !x.atTheta(theta) {
			return 0, 0, 0, 0, false
		}
		kq, mu, rch, pok := x.pair.clusterPencilN(x.g[0], x.g[1], x.g[2], d0, d1, dd0, tolMu)
		if !pok {
			return 0, 0, 0, 0, false
		}
		kLo, kHi, pok := pencilSupN(kq)
		if !pok {
			return 0, 0, 0, 0, false
		}
		k := (kLo + kHi) / 2
		if i > 0 {
			diff := math.Abs(k - prev)
			tiny := diff <= 1e-13*(math.Abs(k)+s0)
			if tiny {
				settled = 2
			} else if diff <= prevDiff/8 {
				settled++
			} else if settled < 2 {
				settled = 0
			}
			if settled >= 2 && diff < best {
				best, lo, hi, mu1, reach = diff, kLo-2*diff, kHi+2*diff, mu, rch
			}
			if tiny || (settled >= 2 && diff > 2*prevDiff) {
				break
			}
			prevDiff = diff
		}
		prev = k
	}
	return lo, hi, mu1, reach, settled >= 2
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
			for n := 1; n <= expansionOrder; n++ {
				x.gMax[n] = max(x.gMax[n], cFrobenius(x.g[n]))
			}
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
	if !x.vector {
		x.sg, x.sn = make([][]complex128, k), make([][expansionOrder + 1]float64, k)
	}
}

func cFrobenius(g []complex128) float64 {
	var s float64
	for _, v := range g {
		s += real(v)*real(v) + imag(v)*imag(v)
	}
	return math.Sqrt(s)
}

// flatten copies g_0…g_N at the phase of the last at into dst, (N+1)·p·m
// long, and sets n[k] = σ_max(g_k), k >= 2.
func (x *highFrequencyExpansion) flatten(dst []complex128, n *[expansionOrder + 1]float64) {
	pm := len(x.g[0])
	for k := range expansionOrder + 1 {
		copy(dst[k*pm:], x.g[k])
		if k >= 2 {
			n[k], _ = x.e.svd.maximumFromFlat(dst, k*pm, x.e.p, x.e.m)
		}
	}
}

// sampleTerms returns g_0…g_N and σ_max(g_k) at sample i of a matrix gain.
func (x *highFrequencyExpansion) sampleTerms(i int) ([]complex128, *[expansionOrder + 1]float64) {
	if x.sg[i] == nil {
		x.at(float64(i) * (2 * math.Pi / float64(len(x.byS))))
		x.sg[i] = make([]complex128, (expansionOrder+1)*len(x.g[0]))
		x.flatten(x.sg[i], &x.sn[i])
	}
	return x.sg[i], &x.sn[i]
}

// sigmaAt returns σ_max(Σ_k t^k·g_k) for the flattened g.
func (x *highFrequencyExpansion) sigmaAt(g []complex128, t float64) float64 {
	pm := len(x.buf)
	tc := complex(t, 0)
	for i := range x.buf {
		v := g[expansionOrder*pm+i]
		for k := expansionOrder - 1; k >= 0; k-- {
			v = v*tc + g[k*pm+i]
		}
		x.buf[i] = v
	}
	sv, err := x.e.svd.maximumFromFlat(x.buf, 0, x.e.p, x.e.m)
	if err != nil {
		return math.Inf(1)
	}
	return sv
}

// secantGap returns max over t ∈ [t1, t2] of the secant of t^k through t1
// and t2 minus t^k, 0 <= t1 <= t2.
func secantGap(k int, t1, t2 float64) float64 {
	if t1 == t2 {
		return 0
	}
	f := float64(k)
	p1, p2 := math.Pow(t1, f), math.Pow(t2, f)
	s := (p2 - p1) / (t2 - t1)
	t := math.Pow(s/f, 1/(f-1))
	return max(p1+s*(t-t1)-math.Pow(t, f), 0)
}

// intervalSigma bounds σ_max(Σ_k t^k·g_k) over t ∈ [t1, t2] for the
// flattened g with n[k] = σ_max(g_k): each t^k, k >= 2, is its secant L_k(t)
// less a gap in [0, gap[k]] (secantGap), and σ_max(g0 + t·g1 + Σ_k L_k(t)·g_k) is
// convex in t, the norm of an affine function, so it peaks at t1 or t2,
// where L_k = t^k. Exact for t1 = t2.
func (x *highFrequencyExpansion) intervalSigma(g []complex128, n *[expansionOrder + 1]float64, t1, t2 float64, gap *[expansionOrder + 1]float64) float64 {
	v := x.sigmaAt(g, t2)
	if t1 == t2 {
		return v
	}
	v = max(v, x.sigmaAt(g, t1))
	for k := 2; k <= expansionOrder; k++ {
		v += gap[k] * n[k]
	}
	return v
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
	if !x.vector {
		g, n := x.sampleTerms(i)
		x.stamp[i], x.vals[i] = x.epoch, x.intervalSigma(g, n, t1, t2, &x.gap)
		return x.vals[i]
	}
	v := overInterval(x.poly[i*polyLen:(i+1)*polyLen], t1, t2)
	x.stamp[i], x.vals[i] = x.epoch, math.Sqrt(max(v, 0))
	return x.vals[i]
}

// modelMax returns sup_φ ‖Σ_{k<=N} t^k·g_k‖; see modelMaxOver.
func (x *highFrequencyExpansion) modelMax(t float64) float64 {
	return x.modelMaxOver(t, t, math.Inf(1))
}

// modelMaxOver returns sup over φ and t ∈ [t1, t2] of the bound
// overInterval of the squared model, square-rooted, which for t1 = t2 is
// the model ‖Σ_{k<=N} t^k·g_k‖ itself: the largest sample, local maxima
// within a factor 1 − 1e-3 of it refined by golden-section search as in
// maxima. The bound at sample i is at most c_0,i + Σ_{n>=1} max_j |c_n,j|·t2^n,
// so walking the samples by decreasing c_0 stops once that falls 1% below
// the largest sample. For a matrix gain the bound is intervalSigma, at most
// σ_max(g0) + Σ_{k>=1} max_j ‖g_k,j‖_F·(t2^k + secantGap). It returns a
// value above stop as soon as one is found, and for finite stop the bound
// from modelMax(0) and these slacks when that is at most stop.
func (x *highFrequencyExpansion) modelMaxOver(t1, t2, stop float64) float64 {
	x.sampleCircle()
	if x.singular {
		return math.Inf(1)
	}
	k := len(x.byS)
	if !math.IsInf(stop, 1) && x.sup0 == 0 {
		x.sup0 = x.modelMaxOver(0, 0, math.Inf(1))
	}
	slack := x.prepare(t1, t2)
	if !math.IsInf(stop, 1) {
		b := math.Sqrt(x.sup0*x.sup0 + slack)
		if !x.vector {
			b = x.sup0 + slack
		}
		if b <= stop {
			return b
		}
	}
	top, last := 0.0, k
	for j, i := range x.byS {
		if x.upper(i, slack) < top*(1-1e-2) {
			last = j
			break
		}
		if top = max(top, x.sampleModel(i, t1, t2)); top > stop {
			return top
		}
	}
	f := func(phi float64) float64 { return x.model(phi, t1) }
	if !x.vector && t1 != t2 {
		if x.flat == nil {
			x.flat = make([]complex128, (expansionOrder+1)*len(x.g[0]))
		}
		f = func(phi float64) float64 {
			if !x.at(phi) {
				return math.Inf(1)
			}
			x.flatten(x.flat, &x.flatN)
			return x.intervalSigma(x.flat, &x.flatN, t1, t2, &x.gap)
		}
	} else if t1 != t2 {
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
		if best = max(best, g); best > stop {
			return best
		}
	}
	return best
}

// prepare starts a new epoch of sampleModel for t ∈ [t1, t2] and returns
// the slack of upper.
func (x *highFrequencyExpansion) prepare(t1, t2 float64) float64 {
	x.epoch++
	slack := 0.0
	if x.vector {
		for n := polyLen - 1; n >= 1; n-- {
			slack = (slack + x.polyMax[n]) * t2
		}
		return slack
	}
	for n := 1; n <= expansionOrder; n++ {
		x.gap[n] = secantGap(n, t1, t2)
		slack += x.gMax[n] * (math.Pow(t2, float64(n)) + x.gap[n])
	}
	return slack
}

// upper bounds sampleModel at sample i from its c_0 and the slack of prepare.
func (x *highFrequencyExpansion) upper(i int, slack float64) float64 {
	c0 := x.poly[i*polyLen]
	if x.vector {
		return math.Sqrt(max(c0+slack, 0))
	}
	return math.Sqrt(c0) + slack
}

// phaseRate returns h, with φ = ωh the phase of G∞ at ω.
func (x *highFrequencyExpansion) phaseRate() float64 {
	return 2 * math.Pi / x.hf.period()
}

// sampleAt returns the index of the circle sample nearest φ.
func (x *highFrequencyExpansion) sampleAt(phi float64) int {
	x.sampleCircle()
	k := len(x.byS)
	u := phi * (float64(k) / (2 * math.Pi))
	if math.Abs(u) >= 1<<52 {
		u = math.Mod(phi, 2*math.Pi) / (2 * math.Pi / float64(k))
	}
	i := int(math.Round(u)) % k
	return (i + k) % k
}

// phaseMask marks the circle samples near which sampleModel over t ∈ [t1,
// t2] may exceed level: those above it, the local maxima whose parabolic
// interpolation is, and their two neighbours on either side. Walking the
// samples by decreasing c_0 stops once upper falls below level·(1 − 1e-3).
// nil where I − D22Δ is singular.
func (x *highFrequencyExpansion) phaseMask(t1, t2, level float64) []bool {
	x.sampleCircle()
	if x.singular {
		return nil
	}
	k := len(x.byS)
	slack := x.prepare(t1, t2)
	mask := make([]bool, k)
	for _, i := range x.byS {
		if x.upper(i, slack) < level*(1-1e-3) {
			break
		}
		v := x.sampleModel(i, t1, t2)
		bad := v > level
		if y0, y2 := x.sampleModel((i+k-1)%k, t1, t2), x.sampleModel((i+1)%k, t1, t2); !bad && v >= y0 && v >= y2 {
			e := v
			if d := 2*v - y0 - y2; d > 0 {
				e += (y0 - y2) * (y0 - y2) / (8 * d)
			}
			bad = e > level
		}
		if bad {
			for d := -2; d <= 2; d++ {
				mask[(i+d+k)%k] = true
			}
		}
	}
	return mask
}

// circleCost returns the number of solves sampleCircle still takes, the
// grid points worth scanning before tailBelow is tried instead.
func (x *highFrequencyExpansion) circleCost() float64 {
	if x.poly != nil {
		return 0
	}
	return 1024 * x.hf.kmax
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
		r := x.remainder(E, t2*w)
		if !(x.modelMaxOver(t1, t2, level-r)+r <= level) {
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
// κ·t²/(2·lim), t = 1/wMin or for a matrix gain at most the reach of
// gramPair, is above tol·lim, so that a sample can show it. Otherwise,
// with a < 0 or κ < 0 at every maximiser, W <= wMax is a frequency past which
// ‖G(jω)‖ <= lim·(1 + tol) is checked: sup_φ ‖Σ_{k<=N} t^k·g_k‖, maximised
// over φ as in onePhaseSup, plus the Cauchy bound on R stays below it at
// t = T·2^−i, i = 0…40, T = 1/W; past these samples the leading term κ·t²
// and the remainder O(t^{N+1}) only shrink. This +Inf verdict rests on the
// expansion, not on a certificate over every t. ok is false when neither is
// decided. For a matrix gain with σ_max(g0) repeated at a maximiser and
// a ≈ 0 there, κ is clusterKappa, from the degenerate expansion of the
// cluster; its pencil not concave leaves it undecided.
func (x *highFrequencyExpansion) limitVerdict(lim, tol, wMin, wMax float64) (finite bool, W float64, ok bool) {
	x.sampleCircle()
	s := make([]float64, len(x.poly)/polyLen)
	for i := range s {
		s[i] = math.Sqrt(x.poly[i*polyLen])
	}
	phis, _ := x.maxima(s, func(phi float64) float64 {
		s2, _, _, _, _ := x.coefficients(phi)
		return math.Sqrt(s2)
	}, 1e-9)
	if len(phis) == 0 {
		return false, 0, false
	}
	allNegative := true
	for _, phi := range phis {
		s2, a, c, scale, exact := x.coefficients(phi)
		tSeen := 1 / wMin
		if exact && !x.vector {
			tSeen = min(tSeen, x.pair.reach)
		}
		ap, apErr := richardson(1e-4, func(h float64) float64 {
			_, aPlus, _, _, _ := x.coefficients(phi + h)
			_, aMinus, _, _, _ := x.coefficients(phi - h)
			return (aPlus - aMinus) / (2 * h)
		})
		// φ* is located to about 1e-8 only, so a(φ*) carries an error of
		// about a′·1e-8 that a zero of a must absorb.
		eps := 1e-6 * (math.Abs(ap) + math.Sqrt(s2*max(scale, 1)))
		if a > eps {
			return a/(lim*wMin) > tol*lim, 0, a/(lim*wMin) > tol*lim
		} else if a < -eps {
			continue
		}
		if !exact {
			if x.vector {
				return false, 0, false
			}
			kLo, kHi, mu1, reach, kok := x.clusterKappa(phi, 2*eps)
			if !kok || math.Abs(mu1) > 2*eps {
				return false, 0, false
			}
			margin := 1e-6 * s2
			switch {
			case kLo > margin:
				t := min(tSeen, reach)
				seen := kLo*t*t/(2*lim) > tol*lim
				return seen, 0, seen
			case !(kHi < -margin):
				allNegative = false
			}
			continue
		}
		alpha, alphaErr := richardson(1e-4, func(h float64) float64 {
			sp, _, _, _, _ := x.coefficients(phi + h)
			sm, _, _, _, _ := x.coefficients(phi - h)
			return -(sp - 2*s2 + sm) / (2 * h * h)
		})
		if !(alpha > alphaErr) {
			return false, 0, false
		}
		// At a sharp peak c and a′²/α nearly cancel, so the margin scales
		// with them and with the error of the difference estimates.
		q := ap * ap / alpha
		kappa := c + q
		margin := 1e-6*s2 + 1e-9*(math.Abs(c)+q) + 2*math.Abs(ap)*apErr/alpha + q*alphaErr/(alpha-alphaErr)
		switch {
		case kappa > margin:
			seen := kappa*tSeen*tSeen/(2*lim) > tol*lim
			return seen, 0, seen
		case !(kappa < -margin):
			allNegative = false
		}
	}
	if !allNegative {
		return false, 0, false
	}
	W, ok = x.horizon(lim*(1+tol), wMax, x.modelMax)
	return false, W, ok
}

// horizon returns a frequency W <= wMax past which ‖G(jω)‖ <= level by the
// expansion: modelMax(t), the sup over the phases of ‖Σ_{k<=N} t^k·g_k‖,
// plus the Cauchy bound on R stays below level at t = T·2^−i, i = 0…40,
// T = 1/W, for the first T, halving from 1/(4‖A‖ + 1), where it does.
func (x *highFrequencyExpansion) horizon(level, wMax float64, modelMax func(t float64) float64) (float64, bool) {
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
				m = modelMax(t)
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
			return 1 / T, true
		}
	}
	return 0, false
}

// richardson returns the limit as h → 0 of the central difference d(h),
// error O(h²), by Richardson extrapolation over steps h0·2^−i, with an
// estimate of its error. It stops once that estimate, already small, grows,
// where roundoff takes over; growth before that is h closing in on the scale
// on which d varies.
func richardson(h0 float64, d func(h float64) float64) (v, err float64) {
	prev := d(h0)
	v, err = prev, math.Inf(1)
	var last float64
	for i := 1; i <= 16; i++ {
		cur := d(h0 * math.Ldexp(1, -i))
		r := (4*cur - prev) / 3
		prev = cur
		if i > 1 {
			e := math.Abs(r - last)
			if e > 2*err && err <= 1e-3*math.Abs(v) {
				break
			}
			if e < err {
				v, err = r, e
			}
			if err <= 1e-12*math.Abs(v) {
				break
			}
		}
		last = r
	}
	return v, err
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
