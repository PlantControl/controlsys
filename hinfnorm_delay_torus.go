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
	vals := hf.gridValues()
	var out []torusMaximum
	for _, i := range largestCells(vals, starts) {
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
// confirm (classes commensurate up to rounding are merged, see
// commensurateMultiples), so the limit is never reported as the peak at ω = +Inf; a finite
// peak above it is. With t = 1/ω, G(jω) = G∞(θ) + t·g1(θ) + O(t²), and where
// a = Re⟨g0, g1⟩ ≠ 0 at a maximiser θ* of S = σ_max(G∞), one of θ* and its
// mirror −θ*, where a has the other sign, has a > 0, so passages of
// θ = ω·h close to it rise above S(θ*); at a fixed point θ* ∈ {0, π}^dim,
// where a = 0, κ > 0 suggests the same (see risesPast). When a maximiser
// within 1e-9 of the largest does, a level lim >= sup S is certified by
// below, and the grid of the given step is extended past rGrid, doubling up
// to span, sampled only on the passages of torusPassages, at most
// torusScanPoints samples, until one exceeds lim·(1 + tol), unless best
// already does. It returns lim,
// that sample on the extension past rGrid (nil for best) and the end of the
// scan, else errDelayPeakUnsupported. The scan is shared by every trial lim,
// and run first, so a give-up decided by it costs no certificate.
func (e *delayLFT) torusPastLimit(hf *highFrequencyGain, best, tol, rGrid, step, span float64,
	points func(lo, R float64) []float64) (float64, *pastLimit, float64, error) {
	unsupported := fmt.Errorf("gain may peak as ω → ∞ over the phase torus of incommensurate delays: %w", errDelayPeakUnsupported)
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
	R0 := max(rGrid, 2*e.normA+1)
	sc := &torusScan{e: e, near: newTorusPassages(e, hf, ms, sup, R0), step: step, r0: R0,
		end: rGrid + span, hi: rGrid, R: 2 * R0}
	// The outcome at level L is unsupported, whatever below certifies, when
	// best and every scanned band stay at or below L; levels only rise with d.
	hopeless := func(level float64) bool {
		return !(best > level) && sc.walk(level).kind == bandGiveUp
	}
	ds := [...]float64{1e-9, 1e-7, 1e-5}
	if hopeless(sup * (1 + ds[0]) * (1 + tol)) {
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
const torusScanPoints = 1 << 12

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
// target, found from the windows where class 0 alone is.
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
			a := max(k0, int(math.Ceil((wc-half)/step)))
			b := min(k1, int(math.Floor((wc+half)/step)))
			for k := a; k <= b; k++ {
				if p.near(float64(k)*step, t) {
					dst = append(dst, k)
				}
			}
		}
	}
	tail := dst[n0:]
	slices.Sort(tail)
	return append(dst[:n0], slices.Compact(tail)...)
}

func (p *torusPassages) near(w float64, t []float64) bool {
	r := p.reach(w)
	for c, h := range p.h {
		x := w*h - t[c]
		x -= 2 * math.Pi * math.Round(x/(2*math.Pi))
		if p.scale[c]*math.Abs(x) >= r {
			return false
		}
	}
	return true
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
