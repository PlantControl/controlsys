package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
)

// zpkUnverifiedRow returns the first row of z, the ZPK of the explicit
// delay-free model sys, that is not verified against the stored model's
// exact response to tol = √ε, or −1 when every row is. Row i is verified
// when no probe refutes it and the probes where every entry of the row
// verifies give at least n + deg_i + 1 interpolation conditions
// (probeWeight), n the order of sys and deg_i the largest root count of the
// row's entries: an entry G − Z is a difference of proper real-rational
// functions of degrees n and deg_i, whose numerator has degree at most
// n + deg_i, so agreement at n + deg_i + 1 points fixes it in exact
// arithmetic. As for the cuts of Reduce (frequencyAxisCutInBand),
// agreement within tol at those points gives no uniform bound between them.
//
// The probes are those of axisProbes, the axis points beside the zeros and
// poles of z (eigenAxisProbes) and the static point (s = 0, z = 1), where the
// relative error of a ZPK peaks, or where a misplaced pole of z puts the
// peak of G − Z; while some row is short of its count,
// zpkFillProbes adds a geometric grid sized to the shortfall. A probe
// within rounding of an eigenvalue of A is dropped (axisProbes).
//
// At a probe, an entry with computed responses Ĝ (boundedResponse plus D)
// and Ẑ (zpkEvalBounded) verifies when |Ĝ − Ẑ| + δ_G + δ_Z ≤ tol·scale,
// δ the rigorous first-order bounds on their evaluation for the stored
// data and the returned roots and gain, and scale the largest |Ĝ|, |Ẑ| of
// the probe's entries: the exact deviation is then within tol·scale. It
// refutes the row when |Ĝ − Ẑ| − δ_G − δ_Z > tol·max(scale, terms_G), δ_G
// the computational bound and terms_G the entry's own terms Σ_l
// |C_il||Û_lj| + |D_ij|, as classifyProbe refutes a cut: the exact
// deviation then exceeds tol relative to the larger of the response and the
// terms that form it, the scale a zero of the model is placed to, unless
// the probe lies in a shared-pole window (zpkSharedPoleWindows), where the
// model's pole is known only to its rounding and the probe counts for
// neither. A probe the plain bounds leave open is sharpened once; one where
// GEPP meets an exactly singular model, or where Ẑ is not finite, counts
// for neither.
func (sys *System) zpkUnverifiedRow(z *ZPK) int {
	n, m, p := sys.Dims()
	if n == 0 {
		return -1
	}
	tol := math.Sqrt(eps())
	discrete := sys.Dt != 0
	pts, mp, ok := axisProbes(sys, nil)
	if !ok {
		return 0
	}
	win := zpkSharedPoleWindows(mp, z, tol)
	var zeros, roots []complex128
	for i := range p {
		for j := range m {
			zeros = append(zeros, z.Zeros[i][j]...)
			roots = append(roots, z.Zeros[i][j]...)
			roots = append(roots, z.Poles[i][j]...)
		}
	}
	extra := eigenAxisProbes(roots, discrete)
	if discrete {
		extra = append(extra, 1)
	} else {
		extra = append(extra, 0)
	}
	pts = zpkKeptProbes(pts, extra, mp)
	need := make([]int, p)
	for i := range p {
		deg := 0
		for j := range m {
			deg = max(deg, len(z.Poles[i][j]), len(z.Zeros[i][j]))
		}
		need[i] = n + deg + 1
	}
	pm := p * m
	zv := make([]complex128, pm)
	zb := make([]float64, pm)
	zok := make([]bool, pm)
	weight := make([]int, p)
	verified := make([]bool, p)
	d := sys.D.RawMatrix()
	judge := func(pts []complex128) int {
		full := newBoundedResponse(sys, pts)
		defer full.release()
		gfull := func(o, i, j int) (complex128, float64) {
			v := full.g[o+i*m+j] + complex(d.Data[i*d.Stride+j], 0)
			return v, cmplx.Abs(v) * gammaN(1)
		}
		for k, s := range pts {
			if !full.ok[k] {
				continue
			}
			for i := range p {
				for j := range m {
					zv[i*m+j], zb[i*m+j], zok[i*m+j] = zpkEvalBounded(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j], s)
				}
			}
			o := k * pm
			inWin := win.contain(s)
			for pass := 0; ; pass++ {
				scale := 0.0
				for i := range p {
					for j := range m {
						g, _ := gfull(o, i, j)
						scale = math.Max(scale, math.Max(cmplx.Abs(g), cmplx.Abs(zv[i*m+j])))
					}
				}
				ts := tol * scale
				open := false
				for i := range p {
					verified[i] = true
					for j := range m {
						e := i*m + j
						if !zok[e] {
							verified[i] = false
							continue
						}
						g, gr := gfull(o, i, j)
						diff := cmplx.Abs(g - zv[e])
						if diff+full.ver[o+e]+gr+zb[e] <= ts {
							continue
						}
						verified[i] = false
						if !inWin && diff-full.comp[o+e]-gr-zb[e] > tol*math.Max(scale, full.terms[o+e]+math.Abs(d.Data[i*d.Stride+j])) {
							return i
						}
						open = true
					}
				}
				if open && pass == 0 && !full.sharp[k] {
					full.sharpen(k)
					continue
				}
				for i := range p {
					if verified[i] {
						weight[i] += probeWeight(s)
					}
				}
				break
			}
		}
		return -1
	}
	if i := judge(pts); i >= 0 {
		return i
	}
	for round := range 3 {
		short := 0
		for i := range p {
			short = max(short, need[i]-weight[i])
		}
		if short <= 0 {
			return -1
		}
		fill := zpkKeptProbes(nil, zpkFillProbes(mp.evals, zeros, discrete, (short+1)/2<<round), mp)
		if i := judge(fill); i >= 0 {
			return i
		}
	}
	for i := range p {
		if weight[i] < need[i] {
			return i
		}
	}
	return -1
}

// zpkKeptProbes appends to pts the points of extra not already in it and
// not within rounding of an eigenvalue of the model (axisProbes' rule).
func zpkKeptProbes(pts, extra []complex128, mp modelPoles) []complex128 {
	n := len(mp.evals)
	for _, s := range extra {
		if slices.Contains(pts, s) {
			continue
		}
		rs := float64(n) * eps() * cmplx.Abs(s)
		drop := false
		for k, ev := range mp.evals {
			if cmplx.Abs(s-ev) <= rs+mp.rho[k] {
				drop = true
				break
			}
		}
		if !drop {
			pts = append(pts, s)
		}
	}
	return pts
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

// zpkSharedPoleWindows is sharedPoleWindows for the poles of z, exact data
// with no rounding bound or resolution of their own: a pole of z shares a
// model pole λ_G within β_G + ρ_G of it, and its window, the disk of
// radius β_G/tol capped at the distance to the nearest other model pole,
// exists when every pole it shares is determined to tol or is a
// rounding-level zero.
func zpkSharedPoleWindows(mp modelPoles, z *ZPK, tol float64) poleWindows {
	var w poleWindows
	var seen []complex128
	for i := range z.Poles {
		for j := range z.Poles[i] {
			for _, lr := range z.Poles[i][j] {
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

// zpkVerifiable reports whether zpkUnverifiedRow applies to sys: an
// explicit, finite model without delays or an LFT.
func (sys *System) zpkVerifiable() bool {
	return !sys.IsDescriptor() && !sys.HasDelay() && sys.LFT == nil && requireFiniteSystem("ZPKModel", sys) == nil
}
