package controlsys

import (
	"cmp"
	"fmt"
	"math"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// torusMaximum is a local maximiser θ of σ_max(G∞(θ)) on the torus of
// phases, with its value.
type torusMaximum struct {
	theta []float64
	value float64
}

// torusMaxima returns the local maxima of σ_max(G∞) on the torus of hf
// reached by Newton ascent from the largest few cells of torusGrid, the
// largest first. Each value is a sample of G∞, a lower bound on its sup.
func torusMaxima(hf *highFrequencyGain) []torusMaximum {
	const starts = 12
	cells, _ := hf.torusGrid()
	nd := hf.dim
	k := len(cells) / nd
	vals := make([]float64, k)
	idx := make([]int, k)
	for i := range k {
		vals[i], idx[i] = hf.at(cells[i*nd:(i+1)*nd]), i
	}
	slices.SortFunc(idx, func(a, b int) int { return cmp.Compare(vals[b], vals[a]) })
	var out []torusMaximum
	for _, i := range idx[:min(starts, k)] {
		if math.IsInf(vals[i], 0) || math.IsNaN(vals[i]) {
			continue
		}
		th, v := torusAscent(hf, cells[i*nd:(i+1)*nd], vals[i])
		dup := false
		for _, o := range out {
			if torusDistance(o.theta, th) < 1e-6 {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, torusMaximum{th, v})
		}
	}
	slices.SortFunc(out, func(a, b torusMaximum) int { return cmp.Compare(b.value, a.value) })
	return out
}

func torusDistance(a, b []float64) float64 {
	d := 0.0
	for i := range a {
		x := math.Mod(math.Abs(a[i]-b[i]), 2*math.Pi)
		d = max(d, min(x, 2*math.Pi-x))
	}
	return d
}

// torusAscent climbs f = σ_max(G∞)² from θ0 by Newton steps on central
// differences, falling back to gradient steps where the Hessian is not
// negative definite, each step halved until f increases.
func torusAscent(hf *highFrequencyGain, theta0 []float64, v0 float64) ([]float64, float64) {
	nd := len(theta0)
	th := slices.Clone(theta0)
	tmp := make([]float64, nd)
	f := func(x []float64) float64 { v := hf.at(x); return v * v }
	at := func(d ...float64) float64 {
		for i := range nd {
			tmp[i] = th[i] + d[i]
		}
		return f(tmp)
	}
	best := v0 * v0
	g := make([]float64, nd)
	hess := mat.NewSymDense(nd, nil)
	step := make([]float64, nd)
	off := make([]float64, nd)
	for range 60 {
		const h = 1e-4
		for i := range nd {
			off[i] = h
			fp := at(off...)
			off[i] = -h
			fm := at(off...)
			off[i] = 0
			g[i] = (fp - fm) / (2 * h)
			hess.SetSym(i, i, (fp-2*best+fm)/(h*h))
			for j := range i {
				off[i], off[j] = h, h
				pp := at(off...)
				off[j] = -h
				pm := at(off...)
				off[i] = -h
				mm := at(off...)
				off[j] = h
				mp := at(off...)
				off[i], off[j] = 0, 0
				hess.SetSym(i, j, (pp-pm-mp+mm)/(4*h*h))
			}
		}
		var neg mat.SymDense
		neg.ScaleSym(-1, hess)
		var ch mat.Cholesky
		newton := ch.Factorize(&neg)
		if newton {
			sv := mat.NewVecDense(nd, step)
			if ch.SolveVecTo(sv, mat.NewVecDense(nd, g)) != nil {
				newton = false
			}
		}
		if !newton {
			n := floats2Norm(g)
			if n == 0 {
				break
			}
			for i := range nd {
				step[i] = 0.05 * g[i] / n
			}
		}
		moved := false
		for range 40 {
			for i := range nd {
				tmp[i] = th[i] + step[i]
			}
			if v := f(tmp); v > best {
				copy(th, tmp)
				best, moved = v, true
				break
			}
			for i := range step {
				step[i] /= 2
			}
		}
		if !moved || floats2Norm(step) < 1e-13 {
			break
		}
	}
	return th, math.Sqrt(best)
}

func floats2Norm(v []float64) float64 {
	s := 0.0
	for _, x := range v {
		s = math.Hypot(s, x)
	}
	return s
}

// torusPastLimit handles several classes of delays whose limit sup σ_max(G∞)
// over the torus of phases is not below the grid peak best. The values of
// Δ(jω) as ω → ∞ approach every point of the torus only when the classes'
// periods are rationally independent, which floating-point delays cannot
// confirm, so the limit is never reported as the peak at ω = +Inf; a finite
// peak above it is. With t = 1/ω, G(jω) = G∞(θ) + t·g1(θ) + O(t²), and where
// a = Re⟨g0, g1⟩ ≠ 0 at a maximiser θ* of S = σ_max(G∞), one of θ* and its
// mirror −θ*, where a has the other sign, has a > 0, so passages of
// θ = ω·h close to it rise above S(θ*); at a fixed point θ* ∈ {0, π}^dim,
// where a = 0, κ > 0 suggests the same (see risesPast). When a maximiser
// within 1e-9 of the largest does, a level lim >= sup S is certified by
// below, and the grid is extended past rGrid, doubling up to span, until a
// sample exceeds lim·(1 + tol), unless best already does. It returns lim,
// that sample on the extension past rGrid (nil for best) and the end of the
// scan, else errDelayPeakUnsupported.
func (e *delayLFT) torusPastLimit(hf *highFrequencyGain, best, tol, rGrid, span float64,
	points func(lo, R float64) []float64) (float64, *pastLimit, float64, error) {
	unsupported := fmt.Errorf("gain may peak as ω → ∞ through a delayed feedthrough: %w", errDelayPeakUnsupported)
	x := newTorusExpansion(e, hf)
	if x == nil || math.IsInf(span, 1) {
		return 0, nil, 0, unsupported
	}
	ms := torusMaxima(hf)
	if len(ms) == 0 {
		return 0, nil, 0, unsupported
	}
	sup := ms[0].value
	rises := false
	for _, m := range ms {
		if m.value < sup*(1-1e-9) {
			break
		}
		if rises = x.risesPast(m.theta); rises {
			break
		}
	}
	if !rises {
		return 0, nil, 0, unsupported
	}
	const budget = 1 << 20
	lim, ok := 0.0, false
	for _, d := range []float64{1e-9, 1e-7, 1e-5} {
		if lim = sup * (1 + d); hf.below(lim, hf.samples+budget) {
			ok = true
			break
		}
	}
	if !ok {
		return 0, nil, 0, unsupported
	}
	level := lim * (1 + tol)
	if best > level {
		return lim, nil, rGrid, nil
	}
	hi := rGrid
	R0 := max(rGrid, 2*e.normA+1)
	for R := 2 * R0; ; R *= 2 {
		Rs := min(R, rGrid+span)
		ext := points(hi, Rs)
		b, w, err := e.scan(ext)
		if err != nil {
			return 0, nil, 0, err
		}
		if b > level {
			return lim, &pastLimit{points(rGrid, Rs), b, w}, Rs, nil
		}
		hi = Rs
		if Rs >= rGrid+span {
			return 0, nil, 0, unsupported
		}
	}
}

// risesPast reports whether the finite peaks near the maximiser θ of
// S = σ_max(G∞) on the torus rise above S(θ) as ω → ∞ along a passage
// through θ: when a ≠ 0, or when a = 0 and κ = c + ∇aᵀ·α⁻¹·∇a > 0 with
// α = −½·∇²S², the torus analogue of limitVerdict, by central differences.
// With several classes a passage need not come close enough to θ, so this
// only steers the search of torusPastLimit.
func (x *highFrequencyExpansion) risesPast(theta []float64) bool {
	s2, a, c, scale, exact := x.coefficientsAt(theta)
	if math.Abs(a) > 1e-6*math.Sqrt(s2*max(scale, 1)) {
		return true
	}
	if !exact {
		return false
	}
	nd := len(theta)
	th := make([]float64, nd)
	coef := func(i, j int, di, dj float64) (float64, float64) {
		copy(th, theta)
		th[i] += di
		th[j] += dj
		s, a, _, _, _ := x.coefficientsAt(th)
		return s, a
	}
	const h = 1e-4
	grad := make([]float64, nd)
	alpha := mat.NewSymDense(nd, nil)
	for i := range nd {
		sp, ap := coef(i, i, h/2, h/2)
		sm, am := coef(i, i, -h/2, -h/2)
		grad[i] = (ap - am) / (2 * h)
		alpha.SetSym(i, i, -(sp-2*s2+sm)/(2*h*h))
		for j := range i {
			pp, _ := coef(i, j, h, h)
			pm, _ := coef(i, j, h, -h)
			mp, _ := coef(i, j, -h, h)
			mm, _ := coef(i, j, -h, -h)
			alpha.SetSym(i, j, -(pp-pm-mp+mm)/(8*h*h))
		}
	}
	var ch mat.Cholesky
	if !ch.Factorize(alpha) {
		return false
	}
	y := mat.NewVecDense(nd, nil)
	g := mat.NewVecDense(nd, grad)
	if ch.SolveVecTo(y, g) != nil {
		return false
	}
	return c+mat.Dot(g, y) > 1e-6*s2
}
