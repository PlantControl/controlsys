package controlsys

import (
	"cmp"
	"errors"
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
	fixed bool // θ ∈ {0, π}^dim, see snapFixedPoints
}

// torusMaxima returns the local maxima of σ_max(G∞) on the torus of hf
// reached by Newton ascent from the largest few cells of torusGrid, the
// largest first. Each value is a sample of G∞, a lower bound on its sup.
func torusMaxima(hf *highFrequencyGain) []torusMaximum {
	const starts = 12
	cells, _ := hf.torusGrid()
	nd := hf.dim
	vals := hf.gridValues()
	var out []torusMaximum
	for _, i := range largestCells(vals, starts) {
		if math.IsInf(vals[i], 0) || math.IsNaN(vals[i]) {
			continue
		}
		th, v := torusAscent(hf.at, cells[i*nd:(i+1)*nd], vals[i])
		dup := false
		for _, o := range out {
			if torusDistance(o.theta, th) < 1e-6 {
				dup = true
				break
			}
		}
		if !dup {
			out = append(out, torusMaximum{theta: th, value: v})
		}
	}
	slices.SortFunc(out, func(a, b torusMaximum) int { return cmp.Compare(b.value, a.value) })
	return out
}

// largestCells returns the indices of the n largest vals, largest first, as
// a descending sort by cmp.Compare orders them. When the n+1 largest are
// distinct the order is forced and a selection finds it; ties fall back to
// the sort, whose order among them is its own.
func largestCells(vals []float64, n int) []int {
	m := min(n+1, len(vals))
	top := make([]int, 0, m)
	for i, v := range vals {
		if len(top) == m && cmp.Compare(v, vals[top[m-1]]) <= 0 {
			if v == vals[top[m-1]] || math.IsNaN(v) && math.IsNaN(vals[top[m-1]]) {
				return sortedCells(vals, n)
			}
			continue
		}
		j := len(top)
		if j < m {
			top = append(top, i)
		} else {
			j--
		}
		for ; j > 0 && cmp.Compare(v, vals[top[j-1]]) >= 0; j-- {
			top[j] = top[j-1]
		}
		top[j] = i
	}
	for j := 1; j < len(top); j++ {
		if cmp.Compare(vals[top[j-1]], vals[top[j]]) <= 0 {
			return sortedCells(vals, n)
		}
	}
	return top[:min(n, len(top))]
}

func sortedCells(vals []float64, n int) []int {
	idx := make([]int, len(vals))
	for i := range idx {
		idx[i] = i
	}
	slices.SortFunc(idx, func(a, b int) int { return cmp.Compare(vals[b], vals[a]) })
	return idx[:min(n, len(idx))]
}

func torusDistance(a, b []float64) float64 {
	d := 0.0
	for i := range a {
		x := math.Mod(math.Abs(a[i]-b[i]), 2*math.Pi)
		d = max(d, min(x, 2*math.Pi-x))
	}
	return d
}

// torusAscent climbs f = gain² from θ0, gain(θ0) = v0, by Newton steps on central
// differences, falling back to gradient steps where the Hessian is not
// negative definite, each step halved until f increases.
func torusAscent(gain func([]float64) float64, theta0 []float64, v0 float64) ([]float64, float64) {
	nd := len(theta0)
	th := slices.Clone(theta0)
	tmp := make([]float64, nd)
	f := func(x []float64) float64 { v := gain(x); return v * v }
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
// over the torus of phases is not below the grid peak best at wBest. Classes
// that stay apart after commensurateMultiples are taken as rationally
// independent, as MATLAB does, so that Δ(jω) as ω → ∞ comes arbitrarily
// close to every point of the torus and sup_ω σ_max(G(jω)) is the larger of
// the finite peaks and S* = sup S, S = σ_max(G∞). With t = 1/ω,
// G(jω) = G∞(θ) + t·g1(θ) + O(t²), and where a = Re⟨g0, g1⟩ ≠ 0 at a
// maximiser θ* of S, one of θ* and its mirror −θ*, where a has the other
// sign, has a > 0, so passages of θ = ω·h close to it rise above S(θ*); at a
// fixed point θ* ∈ {0, π}^dim, where a = 0, κ > 0 suggests the same (see
// risesPast) and κ < 0 that they stay below (see fallsPast). When a
// maximiser within 1e-9 of the largest rises, torusRisesPast searches the
// finite peak above S*, and when it finds none and every such maximiser is
// a fixed point, where the rise is O(1/ω²), torusAtInf decides as below;
// when its scan of the passages reaches the end of the span with no sample
// above S* and some such maximiser is not a fixed point, torusFarPassages
// searches the passages past it;
// when every one of them falls, torusAtInf decides between a finite peak
// and S* at ω = +Inf. final reports that the result
// needs no further tail closing: the peak is found's sample, or S* at
// ω = +Inf when found.w is +Inf. Otherwise it returns a level lim >= S*,
// the sample above it on the extension past rGrid (nil for best) and the end
// of the scan, with far, when some such maximiser is not a fixed point, for
// a tail peak cannot close within the span; else errDelayPeakUnsupported.
func (e *delayLFT) torusPastLimit(hf *highFrequencyGain, best, wBest, tol, rGrid, step, span float64,
	points func(lo, R float64) []float64) (lim float64, found *pastLimit, hi float64, final bool, far torusFar, err error) {
	x := newTorusExpansion(e, hf)
	if x == nil || math.IsInf(span, 1) {
		return 0, nil, 0, false, nil, errTorusUnsupported
	}
	ms := torusMaxima(hf)
	if len(ms) == 0 {
		return 0, nil, 0, false, nil, errTorusUnsupported
	}
	snapFixedPoints(hf, ms)
	sup := ms[0].value
	rises, falls, fixed := false, true, true
	for _, m := range ms {
		if m.value < sup*(1-1e-9) {
			break
		}
		fixed = fixed && m.fixed && !x.slopedAt(m.theta)
		if !rises {
			rises = x.risesPast(m.theta)
			falls = falls && m.fixed && x.fallsPast(m.theta)
		}
	}
	switch {
	case rises:
		lim, found, hi, err = e.torusRisesPast(x, hf, ms, best, tol, rGrid, step, span, points)
		if fixed && errors.Is(err, errTorusUnsupported) {
			lim, found, hi, final, err = e.torusAtInf(x, hf, ms, best, wBest, tol, rGrid, span, true, points)
			return lim, found, hi, final, nil, err
		}
		if errors.Is(err, errTorusScanned) {
			lim, found, hi, final, err = e.torusFarPassages(x, hf, ms, best, wBest, rGrid+span)
			return lim, found, hi, final, nil, err
		}
		if err == nil && !fixed {
			far = func(best, wBest, end float64) (*pastLimit, error) {
				_, f, _, _, err := e.torusFarPassages(x, hf, ms, best, wBest, end)
				return f, err
			}
		}
		return lim, found, hi, false, far, err
	case falls:
		lim, found, hi, final, err = e.torusAtInf(x, hf, ms, best, wBest, tol, rGrid, span, false, points)
		return lim, found, hi, final, nil, err
	}
	return 0, nil, 0, false, nil, errTorusUnsupported
}

// torusFar decides by torusFarPassages a tail of the rising case that peak
// cannot close within the span: the grid up to end is certified to the
// peak best at wBest, and the result is final.
type torusFar func(best, wBest, end float64) (*pastLimit, error)

var errTorusUnsupported = fmt.Errorf("gain may peak as ω → ∞ over the phase torus of incommensurate delays: %w", errDelayPeakUnsupported)

// errTorusScanned is errTorusUnsupported from torusRisesPast when its scan
// of the passages reached the end of the span without a sample above the
// lowest level it tries.
var errTorusScanned = fmt.Errorf("%w", errTorusUnsupported)

// torusRisesPast certifies a level lim >= sup S by below and extends the
// grid of the given step past rGrid, doubling up to span, sampled only on
// the passages of torusPassages, at most torusScanPoints samples, until one
// exceeds lim·(1 + tol), unless best already does. It returns lim, that
// sample on the extension past rGrid (nil for best) and the end of the scan,
// else errDelayPeakUnsupported. The scan is shared by every trial lim, and
// run first, so a give-up decided by it costs no certificate.
func (e *delayLFT) torusRisesPast(x *highFrequencyExpansion, hf *highFrequencyGain, ms []torusMaximum, best, tol, rGrid, step, span float64,
	points func(lo, R float64) []float64) (float64, *pastLimit, float64, error) {
	unsupported := errTorusUnsupported
	sup := ms[0].value
	const budget = 1 << 20
	R0 := max(rGrid, 2*e.minNormA()+1)
	sc := &torusScan{e: e, near: newTorusPassages(e, hf, ms, sup, R0), step: step, r0: R0,
		end: rGrid + span, hi: rGrid, R: 2 * R0}
	// The outcome at level L is unsupported, whatever below certifies, when
	// best and every scanned band stay at or below L; levels only rise with d.
	hopeless := func(level float64) bool {
		return !(best > level) && sc.walk(level).kind == bandGiveUp
	}
	ds := [...]float64{1e-9, 1e-7, 1e-5}
	if l0 := sup * (1 + ds[0]) * (1 + tol); hopeless(l0) {
		if sc.walk(l0).last {
			unsupported = errTorusScanned
		}
		return 0, nil, 0, unsupported
	}
	// below fails at every level when it fails at the highest: a lower level
	// splits a superset of its cells, generation by generation.
	top := sup * (1 + ds[len(ds)-1])
	if hf.over > top || !hf.below(top, hf.samples+budget) {
		return 0, nil, 0, unsupported
	}
	lim, ok := 0.0, false
	for i, d := range ds {
		if lim = sup * (1 + d); hf.over > lim {
			continue // a torus sample above lim: below(lim) cannot hold
		}
		if hopeless(lim * (1 + tol)) {
			return 0, nil, 0, unsupported
		}
		if i == len(ds)-1 || hf.below(lim, hf.samples+budget) {
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
	switch bd := sc.walk(level); bd.kind {
	case bandFailed:
		e.err = bd.eErr
		return 0, nil, 0, bd.err
	case bandGiveUp:
		return 0, nil, 0, unsupported
	default:
		return lim, &pastLimit{points(rGrid, bd.Rs), bd.b, bd.w}, bd.Rs, nil
	}
}

// torusScan samples G on the passages of torusPassages past the grid end in
// bands doubling in reach, at most torusScanPoints samples past 2·R0, each
// band computed once on demand, so that the walks at several levels share
// them.
type torusScan struct {
	e     *delayLFT
	near  *torusPassages
	step  float64
	r0    float64
	end   float64 // rGrid + span
	hi, R float64
	evals int
	ks    []int
	run   []float64
	bands []torusBand
}

type bandKind int

const (
	bandScanned bandKind = iota
	bandGiveUp           // sample cap reached, or the last band scanned
	bandFailed           // e.scan failed
)

// torusBand is the largest sample b at w of the band up to Rs, last when it
// reaches the end; a band that gives up or fails ends the scan.
type torusBand struct {
	kind bandKind
	b, w float64
	Rs   float64
	last bool
	err  error
	eErr error // e.err set by the failure, cleared until a walk reaches it
}

// walk returns the first band above level, else the band that ends the
// scan: one that gives up or fails, or the last scanned, as bandGiveUp.
func (s *torusScan) walk(level float64) torusBand {
	for i := 0; ; i++ {
		if i == len(s.bands) {
			s.next()
		}
		bd := s.bands[i]
		if bd.kind != bandScanned || bd.b > level {
			return bd
		}
		if bd.last {
			bd.kind = bandGiveUp
			return bd
		}
	}
}

func (s *torusScan) next() {
	e := s.e
	Rs := min(s.R, s.end)
	s.R *= 2
	s.ks = s.near.indices(s.hi, Rs, s.step, s.ks[:0])
	ks := s.ks
	bd := torusBand{b: math.Inf(-1), Rs: Rs, last: Rs >= s.end}
	for i := 0; i < len(ks); {
		j := i + 1
		for j < len(ks) && ks[j] == ks[j-1]+1 {
			j++
		}
		s.run = s.run[:0]
		for k := ks[i] - 1; k <= ks[j-1]+1; k++ {
			w := float64(k) * s.step
			s.run = append(s.run, w)
			if w > 2*s.r0 {
				s.evals++
			}
		}
		if s.evals > torusScanPoints {
			s.bands = append(s.bands, torusBand{kind: bandGiveUp})
			return
		}
		prev := e.err
		bi, wi, err := e.scan(s.run)
		if err != nil {
			s.bands = append(s.bands, torusBand{kind: bandFailed, err: err, eErr: e.err})
			e.err = prev
			return
		}
		if bi > bd.b {
			bd.b, bd.w = bi, wi
		}
		i = j
	}
	s.hi = Rs
	s.bands = append(s.bands, bd)
}

// torusScanPoints caps the frequencies torusPastLimit samples past the grid
// end before it gives up on a finite peak above the limit.
const torusScanPoints = 1 << 15

// torusTailPoints is the grid extension past which peak tries
// torusTailBelow to close the tail of several delay classes nearer.
const torusTailPoints = 1 << 14

// torusPassages selects the frequencies ω whose phases θ = ω·h pass close
// to a maximiser θ* of S = σ_max(G∞) or its mirror −θ*, the only places
// where G(jω) = G∞(θ) + O(1/ω) can rise above sup S once ω is large: the
// rise near θ* is O(1/ω) and S falls quadratically away from it, so the
// reach shrinks as 1/√ω. torusPassageReach is calibrated on random 2×2 and
// SISO models with two classes of delays, whose finite peaks above the limit
// lay within 2·√(R0/ω) of a maximiser in the max-norm of K_c·θ_c; every ω
// up to 2·R0 is kept. The selection steers the search only.
type torusPassages struct {
	h       []float64 // h_c, θ_c = ω·h_c
	scale   []float64
	targets [][]float64
	r0      float64
}

const torusPassageReach = 4

func newTorusPassages(e *delayLFT, hf *highFrequencyGain, ms []torusMaximum, sup, r0 float64) *torusPassages {
	p := &torusPassages{h: make([]float64, hf.dim), scale: hf.scale, r0: r0}
	for i, c := range hf.grp {
		k := 1.0
		if hf.k != nil {
			k = hf.k[i]
		}
		p.h[c] = e.tau[i] / k
	}
	for _, m := range ms {
		if m.value < sup*(1-1e-3) {
			break
		}
		neg := make([]float64, len(m.theta))
		for c, t := range m.theta {
			neg[c] = -t
		}
		p.targets = append(p.targets, m.theta, neg)
	}
	return p
}

func (p *torusPassages) reach(w float64) float64 {
	return torusPassageReach * math.Sqrt(p.r0/w)
}

// indices appends to dst, ascending, the k with lo < k·step <= hi whose
// ω = k·step is selected: all up to 2·R0, past it those within reach of a
// target, found from the windows where class 0 alone is, each bracketed by
// the grid points about it, as it may be narrower than step.
func (p *torusPassages) indices(lo, hi, step float64, dst []int) []int {
	k0 := int(math.Floor(lo/step)) + 1
	k1 := int(math.Floor(hi / step))
	kd := min(k1, int(math.Floor(2*p.r0/step)))
	for k := k0; k <= kd; k++ {
		dst = append(dst, k)
	}
	k0 = max(k0, kd+1)
	if k0 > k1 {
		return dst
	}
	n0 := len(dst)
	h0, s0 := p.h[0], p.scale[0]
	period := 2 * math.Pi / h0
	wlo, whi := float64(k0)*step, float64(k1)*step
	for _, t := range p.targets {
		n := math.Ceil((wlo*h0-t[0])/(2*math.Pi) - 1)
		for ; ; n++ {
			wc := (t[0] + 2*math.Pi*n) / h0
			half := p.reach(max(2*p.r0, wc-period)) / (s0 * h0)
			if wc-half > whi {
				break
			}
			lo, hi, ok := p.window(wc, t)
			if !ok {
				continue
			}
			a := max(k0, int(math.Floor(lo/step)))
			b := min(k1, int(math.Ceil(hi/step)))
			for k := a; k <= b; k++ {
				dst = append(dst, k)
			}
		}
	}
	tail := dst[n0:]
	slices.Sort(tail)
	return append(dst[:n0], slices.Compact(tail)...)
}

// window returns the interval of ω about wc, where class 0 is at t[0],
// on which every class is within reach of t, the phases linear in ω
// across it; ok is false when it is empty.
func (p *torusPassages) window(wc float64, t []float64) (lo, hi float64, ok bool) {
	r := p.reach(wc)
	lo, hi = math.Inf(-1), math.Inf(1)
	for c, h := range p.h {
		x := wc*h - t[c]
		x -= 2 * math.Pi * math.Round(x/(2*math.Pi))
		d := r / p.scale[c]
		lo = max(lo, wc+(-d-x)/h)
		hi = min(hi, wc+(d-x)/h)
	}
	return lo, hi, lo < hi
}

// risesPast reports whether the finite peaks near the maximiser θ of
// S = σ_max(G∞) on the torus rise above S(θ) as ω → ∞ along a passage
// through θ: when a ≠ 0, or when a = 0 and κ = c + ∇aᵀ·α⁻¹·∇a > 0 with
// α = −½·∇²S², the torus analogue of limitVerdict, by central differences.
// It routes torusPastLimit to the search for a finite peak, which alone
// decides the result: a passage need not come close enough to θ before the
// search ends.
func (x *highFrequencyExpansion) risesPast(theta []float64) bool {
	s2, a, c, scale, exact := x.coefficientsAt(theta)
	eps := 1e-6 * math.Sqrt(s2*max(scale, 1))
	if math.Abs(a) > eps {
		return true
	}
	if !exact {
		lo, _, ok := x.torusClusterKappa(theta, eps)
		return ok && lo > 1e-6*s2
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

// slopedAt reports whether a = Re⟨g0, g1⟩ ≠ 0 at θ, to the tolerance of
// risesPast: passages close to θ or its mirror then rise above S(θ) by
// O(1/ω), not O(1/ω²).
func (x *highFrequencyExpansion) slopedAt(theta []float64) bool {
	s2, a, _, scale, _ := x.coefficientsAt(theta)
	return math.IsInf(s2, 1) || !(math.Abs(a) <= 1e-6*math.Sqrt(s2*max(scale, 1)))
}

// torusClusterKappa returns the bounds of clusterKappaAt at θ, where
// σ_max(G∞(θ)) of a matrix gain is repeated and a ≈ 0 to eps, ok only when
// they are formed and μ1, the slope of the top branches, is 0 to 2·eps.
func (x *highFrequencyExpansion) torusClusterKappa(theta []float64, eps float64) (lo, hi float64, ok bool) {
	if x.vector {
		return 0, 0, false
	}
	if k := &x.kappa; k.theta != nil && k.eps == eps && slices.Equal(k.theta, theta) {
		return k.lo, k.hi, k.ok
	}
	lo, hi, mu1, _, ok := x.clusterKappaAt(theta, 2*eps)
	ok = ok && math.Abs(mu1) <= 2*eps
	x.kappa = clusterVerdict{slices.Clone(theta), eps, lo, hi, ok}
	return lo, hi, ok
}

// snapFixedPoints moves each maximiser within 1e-6 of a fixed point
// θ ∈ {0, π}^dim onto it, where G∞(−θ) = conj G∞(θ) makes S stationary, when
// S there is not lower, marks it fixed and sorts ms again, largest first.
func snapFixedPoints(hf *highFrequencyGain, ms []torusMaximum) {
	th := make([]float64, hf.dim)
	for i := range ms {
		m := &ms[i]
		near := true
		for c, t := range m.theta {
			r := math.Remainder(t, math.Pi)
			if math.Abs(r) > 1e-6 {
				near = false
				break
			}
			th[c] = math.Pi * math.Round((t-r)/math.Pi)
		}
		if !near {
			continue
		}
		if v := hf.at(th); v >= m.value*(1-1e-12) {
			m.theta, m.value, m.fixed = slices.Clone(th), max(v, m.value), true
		}
	}
	slices.SortStableFunc(ms, func(a, b torusMaximum) int { return cmp.Compare(b.value, a.value) })
}

// fallsPast reports whether the finite peaks near the fixed point θ ∈
// {0, π}^dim, a maximiser of S = σ_max(G∞) where a = 0, stay below S(θ) as
// ω → ∞: κ = c + ∇aᵀ·α⁻¹·∇a < 0, α = −½·∇²S² positive definite, the torus
// analogue of limitVerdict, with ∇a and α by Richardson-extrapolated
// central differences and a margin for their errors.
func (x *highFrequencyExpansion) fallsPast(theta []float64) bool {
	s2, a, c, scale, exact := x.coefficientsAt(theta)
	eps := 1e-6 * math.Sqrt(s2*max(scale, 1))
	if math.Abs(a) > eps {
		return false
	}
	if !exact {
		_, hi, ok := x.torusClusterKappa(theta, eps)
		return ok && hi < -1e-6*s2
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
	grad := make([]float64, nd)
	alpha := mat.NewSymDense(nd, nil)
	var gErr, aErr float64
	for i := range nd {
		v, err := richardson(1e-4, func(h float64) float64 {
			_, ap := coef(i, i, h, 0)
			_, am := coef(i, i, -h, 0)
			return (ap - am) / (2 * h)
		})
		grad[i], gErr = v, math.Hypot(gErr, err)
		for j := range i + 1 {
			v, err := richardson(1e-4, func(h float64) float64 {
				if i == j {
					sp, _ := coef(i, i, h, 0)
					sm, _ := coef(i, i, -h, 0)
					return -(sp - 2*s2 + sm) / (2 * h * h)
				}
				pp, _ := coef(i, j, h, h)
				pm, _ := coef(i, j, h, -h)
				mp, _ := coef(i, j, -h, h)
				mm, _ := coef(i, j, -h, -h)
				return -(pp - pm - mp + mm) / (8 * h * h)
			})
			alpha.SetSym(i, j, v)
			aErr = math.Hypot(aErr, err)
			if i != j {
				aErr = math.Hypot(aErr, err)
			}
		}
	}
	var es mat.EigenSym
	if !es.Factorize(alpha, false) {
		return false
	}
	lmin := slices.Min(es.Values(nil))
	if !(lmin > 2*aErr) {
		return false
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
	q := mat.Dot(g, y)
	ny := mat.Norm(y, 2)
	qErr := 2*ny*gErr + ny*ny*aErr/(1-aErr/lmin)
	margin := 1e-6*s2 + 1e-9*(math.Abs(c)+q) + qErr
	return c+q < -margin
}

// flatSigma returns ‖Σ_k t^k·g_k‖ for the flattened g_0…g_N.
func (x *highFrequencyExpansion) flatSigma(g []complex128, t float64) float64 {
	if !x.vector {
		return x.sigmaAt(g, t)
	}
	pm := len(x.buf)
	tc := complex(t, 0)
	s := 0.0
	for i := range pm {
		v := g[expansionOrder*pm+i]
		for k := expansionOrder - 1; k >= 0; k-- {
			v = v*tc + g[k*pm+i]
		}
		s += real(v)*real(v) + imag(v)*imag(v)
	}
	return math.Sqrt(s)
}

// torusModelMax estimates sup_θ ‖Σ_{k<=N} t^k·g_k(θ)‖ over the torus from
// below, as onePhaseSup does on the circle: the largest value on the cells
// of torusGrid, whose g_k are solved once, and the ascents of torusAscent
// from the given number of largest cells and from starts; +Inf where
// I − D22Δ is singular at a cell.
func (x *highFrequencyExpansion) torusModelMax(t float64, cells int, starts [][]float64) float64 {
	h := x.hf
	grid, _ := h.torusGrid()
	nd := h.dim
	n := len(grid) / nd
	size := (expansionOrder + 1) * len(x.g[0])
	if x.torusG == nil {
		x.torusG = make([]complex128, n*size)
		for i := range n {
			if !x.atTheta(grid[i*nd : (i+1)*nd]) {
				x.torusSingular = true
				break
			}
			for k := range expansionOrder + 1 {
				copy(x.torusG[i*size+k*len(x.g[0]):], x.g[k])
			}
		}
	}
	if x.torusSingular {
		return math.Inf(1)
	}
	vals := make([]float64, n)
	for i := range n {
		vals[i] = x.flatSigma(x.torusG[i*size:(i+1)*size], t)
	}
	f := func(th []float64) float64 {
		if !x.atTheta(th) {
			return math.Inf(1)
		}
		return x.sum(t)
	}
	best := slices.Max(vals)
	for _, i := range largestCells(vals, cells) {
		_, v := torusAscent(f, grid[i*nd:(i+1)*nd], vals[i])
		best = max(best, v)
	}
	for _, s := range starts {
		_, v := torusAscent(f, s, f(s))
		best = max(best, v)
	}
	return best
}

// torusAtInf decides, when every maximiser of S = σ_max(G∞) within 1e-9 of
// S* = sup S falls (see fallsPast), between a finite peak and S* at
// ω = +Inf. below certifies lim >= S* as torusRisesPast does; when best
// exceeds lim·(1 + tol) the tail is left to peak. Otherwise, with level =
// max(best, S*)·(1 + tol), horizon finds W past which ‖G(jω)‖ <= level by
// the expansion, its model maximised over the torus by torusModelMax — like
// limitVerdict for one phase, this rests on the expansion, not on a
// certificate over every t — and the grid is extended densely to W. The
// peak is the largest sample there when above level, else best when above
// S*·(1 + tol), else S* at ω = +Inf.
//
// rising is the case where those maximisers are fixed points θ* ∈ {0, π}^dim
// with a = 0 and some rise (κ > 0, see risesPast) but torusRisesPast finds
// no sample above lim: passages then rise above S* by O(κ/ω²) only, and
// only where they come within O(1/ω) of θ*, which for dim >= 2 is rare.
// lim must then be S*·(1 + 1e-9), level is max(best, S*)·(1 + peakCertTol),
// and the peak is the largest of best, S* and the samples up to W, with
// the extension returned for peak to certify.
func (e *delayLFT) torusAtInf(x *highFrequencyExpansion, hf *highFrequencyGain, ms []torusMaximum, best, wBest, tol, rGrid, span float64,
	rising bool, points func(lo, R float64) []float64) (float64, *pastLimit, float64, bool, error) {
	const budget = 1 << 20
	sup := ms[0].value
	var lim float64
	if rising {
		lim = sup * (1 + 1e-9)
		if hf.over > lim || !hf.below(lim, hf.samples+budget) {
			return 0, nil, 0, false, errTorusUnsupported
		}
	} else {
		top := sup * (1 + 1e-5)
		if hf.over > top || !hf.below(top, hf.samples+budget) {
			return 0, nil, 0, false, errTorusUnsupported
		}
		lim = top
		for _, d := range []float64{1e-9, 1e-7} {
			if l := sup * (1 + d); !(hf.over > l) && hf.below(l, hf.samples+budget) {
				lim = l
				break
			}
		}
	}
	if best > lim*(1+tol) {
		return lim, nil, rGrid, false, nil
	}
	level := max(best, sup) * (1 + tol)
	if rising {
		level = max(best, sup) * (1 + peakCertTol)
	}
	var starts [][]float64
	for _, m := range ms {
		starts = append(starts, m.theta)
	}
	W, ok := x.horizon(level, 0, min(rGrid+span, 1e12), func(t float64) float64 {
		return x.torusModelMax(t, 12, starts)
	})
	if !ok || W-rGrid > span {
		return 0, nil, 0, false, errTorusUnsupported
	}
	hi := max(rGrid, W)
	var ext []float64
	if W > rGrid {
		ext = points(rGrid, W)
		b, w, err := e.scan(ext)
		if err != nil {
			return 0, nil, 0, false, err
		}
		if b > level || rising && b > max(best, sup) {
			return lim, &pastLimit{ext, b, w}, hi, true, nil
		}
	}
	if !rising {
		ext = nil
	}
	if best > sup*(1+tol) || rising && best > sup {
		return lim, &pastLimit{ext, best, wBest}, hi, true, nil
	}
	return lim, &pastLimit{ext, sup, math.Inf(1)}, hi, true, nil
}

// atFrequency returns the gain on the torus of h with the blocks of
// G(jw), D + C(jwI − A)⁻¹B for each, in place of those at ω = ∞, and the
// norm bounds of delayLFT.h; nil without states, for w <= ‖A‖ or where
// jwI − A is singular.
func (h *highFrequencyGain) atFrequency(w float64) *highFrequencyGain {
	e := h.e
	if e.n == 0 || e.sys.E != nil || !(w > e.minNormA()) {
		return nil
	}
	n, nd, p, m := e.n, e.nd, e.p, e.m
	lft := e.sys.LFT
	a := e.sys.A.RawMatrix()
	lu := make([]complex128, n*n)
	for i := range n {
		for j := range n {
			lu[i*n+j] = complex(-a.Data[i*a.Stride+j], 0)
		}
		lu[i*n+i] += complex(0, w)
	}
	piv := make([]int, n)
	if !cLUFactor(lu, piv, n) {
		return nil
	}
	cols := m + nd
	x := make([]complex128, n*cols)
	for k, b := range []*mat.Dense{e.sys.B, lft.B2} {
		off, c := k*m, m
		if k == 1 {
			c = nd
		}
		if b == nil || b.IsEmpty() {
			continue
		}
		for i := range n {
			for j := range c {
				x[i*cols+off+j] = complex(b.At(i, j), 0)
			}
		}
	}
	cLUSolve(lu, piv, x, n, cols)
	block := func(d []complex128, c *mat.Dense, off, r, cc int) []complex128 {
		out := slices.Clone(d)
		if c == nil || c.IsEmpty() {
			return out
		}
		for i := range r {
			for j := range cc {
				var s complex128
				for k := range n {
					s += complex(c.At(i, k), 0) * x[k*cols+off+j]
				}
				out[i*cc+j] += s
			}
		}
		return out
	}
	f := &highFrequencyGain{
		e: e, dim: h.dim, grp: h.grp, k: h.k, kmax: h.kmax, scale: h.scale, over: math.Inf(-1),
		d11: block(h.d11, e.sys.C, 0, p, m), d12: block(h.d12, e.sys.C, m, p, nd),
		d21: block(h.d21, lft.C2, 0, nd, m), d22: block(h.d22, lft.C2, m, nd, nd),
		lu: make([]complex128, nd*nd), x: make([]complex128, nd*m),
		phase: make([]complex128, nd), piv: make([]int, nd), g: make([]complex128, p*m),
		grid: h.grid, gridR: h.gridR,
		n1221: math.Inf(1), n22: math.Inf(1),
	}
	for _, s := range e.realizations() {
		f.n1221 = minBound(f.n1221, s.h(e.d12, s.c1, s.b2, w)*s.h(e.d21, s.c2, s.b1, w))
		f.n22 = minBound(f.n22, s.h(e.d22, s.c2, s.b2, w))
	}
	f.eps = f.n22 - e.d22
	f.lip = f.lipschitz()
	return f
}

// torusTailBelow reports whether σ_max(G(jω)) <= level for every ω >= R,
// given S = σ_max(G∞) <= lim on the torus hf. With t = 1/ω and the phases θ
// held, G is analytic in t for |t| <= 1/w, w > ‖A‖, with ‖G − G∞‖ <= E =
// tailGain(0, w), so on t ∈ [0, 1/R] it lies within err = E/4·(w/(R − w))²
// (1/8 of the squared interval times the Cauchy bound on ∂²G/∂t²) of the
// chord from G∞(θ) to F(θ) = G at t = 1/R, whose gain is convex along it:
// σ_max(G) <= max(S, σ_max(F)) + err. below then certifies σ_max(F) <=
// level − err over the torus, within budget samples, once the grid of
// torusGrid does not exceed it; spent reports that below ran out of budget.
func (e *delayLFT) torusTailBelow(hf *highFrequencyGain, lim, level, R float64, budget int) (ok, spent bool) {
	errT := math.Inf(1)
	for w := e.minNormA()*1.01 + 1e-9; w < 0.9*R; w *= 1.2 {
		q := w / (R - w)
		errT = min(errT, e.tailGain(0, w)/4*q*q)
	}
	room := level - errT
	if !(room >= lim) {
		return false, false
	}
	f := hf.atFrequency(R)
	if f == nil {
		return false, false
	}
	for _, v := range f.gridValues() {
		if !(v <= room) {
			return false, false
		}
	}
	ok = f.below(room, f.samples+budget)
	return ok, !ok && !(f.over > room)
}
