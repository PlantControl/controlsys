package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"
)

type NyquistResult struct {
	Omega         []float64
	Contour       []complex128
	ContourN      []complex128
	Encirclements int
	RHPPoles      int
	RHPZerosCL    int
}

// Nyquist computes the Nyquist response of a SISO model.
//
// Omega, Contour and ContourN follow MATLAB nyquist: Contour[k] is the
// response at Omega[k] (jω, or e^{jωdt} for discrete models up to the Nyquist
// frequency), ContourN is the negative-frequency branch obtained by symmetry,
// and the auto grid (omega nil) is strictly increasing and skips the
// frequencies of poles on the stability boundary.
//
// Encirclements, RHPPoles and RHPZerosCL extend MATLAB: Encirclements is the
// clockwise winding of the loop response around -1 along the closed Nyquist
// contour (imaginary axis closed at infinity, or the unit circle), indented
// around poles on the stability boundary so that those poles count as stable.
// RHPPoles counts open-loop poles strictly outside the stability boundary and
// RHPZerosCL = Encirclements + RHPPoles is the number of unstable closed-loop
// poles under unit negative feedback. Relative to the largest pole magnitude,
// a pole is on the boundary within sqrt(eps), or within 10*cbrt(eps) for a group
// of two or more poles (a computed repeated boundary pole); closed-loop poles
// inside the indentation around a boundary pole are not resolved. The counts
// never depend on omega.
func (sys *System) Nyquist(omega []float64, nPoints int) (*NyquistResult, error) {
	if _, err := newSISOLoopModel(sys, "Nyquist"); err != nil {
		return nil, err
	}

	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}

	bd := classifyNyquistBoundary(poles, sys.IsContinuous(), sys.Dt)
	rhpPoles := 0
	for i, pole := range poles {
		if bd.cluster[i] < 0 && poleOutsideStabilityBoundary(pole, sys.IsContinuous(), 0) {
			rhpPoles++
		}
	}

	if omega == nil {
		omega = autoNyquistFreqs(sys, poles, bd, nPoints)
	}

	contour := make([]complex128, len(omega))
	if len(omega) > 0 {
		resp, err := sys.FreqResponse(omega)
		if err != nil {
			return nil, err
		}
		for k := range omega {
			contour[k] = resp.At(k, 0, 0)
		}
	}

	contourN := make([]complex128, len(contour))
	for k := range contour {
		contourN[k] = cmplx.Conj(contour[len(contour)-1-k])
	}

	enc, err := nyquistEncirclements(sys, poles, bd)
	if err != nil {
		return nil, err
	}

	return &NyquistResult{
		Omega:         append([]float64(nil), omega...),
		Contour:       contour,
		ContourN:      contourN,
		Encirclements: enc,
		RHPPoles:      rhpPoles,
		RHPZerosCL:    enc + rhpPoles,
	}, nil
}

// nyquistBoundary groups the open-loop poles on the stability boundary into
// clusters sharing one indentation of the Nyquist contour.
type nyquistBoundary struct {
	continuous bool
	dt         float64
	clusters   []boundaryCluster
	cluster    []int // per pole: index into clusters, or -1
}

type boundaryCluster struct {
	w0     float64
	spread float64 // max distance from center of a member pole folded to Im >= 0
}

// classifyNyquistBoundary puts on the boundary every pole within
// sqrt(eps)*max(1, max|p|) of it, and every group of two or more poles within
// cbrt(eps)*max(1, max|p|) of the boundary and of each other: computed
// boundary poles carry eigenvalue error, repeated ones split by about
// eps^(1/m), and evaluating the response that close to a pole is singular.
// A cluster whose indentation would reach the real axis is centred on it.
func classifyNyquistBoundary(poles []complex128, continuous bool, dt float64) nyquistBoundary {
	bd := nyquistBoundary{continuous: continuous, dt: dt, cluster: make([]int, len(poles))}
	maxAbs := 1.0
	for _, pole := range poles {
		maxAbs = max(maxAbs, cmplx.Abs(pole))
	}
	single := math.Sqrt(eps()) * maxAbs
	loose := math.Cbrt(eps()) * maxAbs
	dist := func(p complex128) float64 {
		if continuous {
			return math.Abs(real(p))
		}
		return math.Abs(cmplx.Abs(p) - 1)
	}

	on := make([]bool, len(poles))
	for i, p := range poles {
		on[i] = dist(p) < single
		if on[i] || dist(p) >= loose {
			continue
		}
		for j, q := range poles {
			if j != i && dist(q) < loose && cmplx.Abs(foldUpper(p)-foldUpper(q)) < 2*loose {
				on[i] = true
				break
			}
		}
	}

	scale := 1.0
	if !continuous {
		scale = dt
	}
	snap := max(loose, nyquistIndentBase)
	angles := make([]float64, len(poles))
	for i, p := range poles {
		bd.cluster[i] = -1
		if !on[i] {
			continue
		}
		angle := imag(foldUpper(p))
		if !continuous {
			angle = cmplx.Phase(foldUpper(p))
		}
		switch {
		case angle < snap:
			angle = 0
		case !continuous && math.Pi-angle < snap:
			angle = math.Pi
		}
		angles[i] = angle
		bd.assign(i, angle/scale, 10*loose)
	}
	bd.updateSpreads(poles)

	snapped := false
	for i, k := range bd.cluster {
		if k < 0 {
			continue
		}
		c := bd.clusters[k]
		reach := 4 * c.spread
		switch {
		case angles[i] > 0 && angles[i] <= reach:
			angles[i], snapped = 0, true
		case !continuous && angles[i] < math.Pi && math.Pi-angles[i] <= reach:
			angles[i], snapped = math.Pi, true
		}
	}
	if snapped {
		bd.clusters = bd.clusters[:0]
		for i, k := range bd.cluster {
			if k >= 0 {
				bd.assign(i, angles[i]/scale, 10*loose)
			}
		}
		bd.updateSpreads(poles)
	}

	order := make([]int, len(bd.clusters))
	for k := range order {
		order[k] = k
	}
	sort.Slice(order, func(a, b int) bool { return bd.clusters[order[a]].w0 < bd.clusters[order[b]].w0 })
	rank := make([]int, len(order))
	sorted := make([]boundaryCluster, len(order))
	for r, k := range order {
		rank[k] = r
		sorted[r] = bd.clusters[k]
	}
	bd.clusters = sorted
	for i, k := range bd.cluster {
		if k >= 0 {
			bd.cluster[i] = rank[k]
		}
	}
	return bd
}

// assign puts pole i in the cluster at w0, merging within tol (in angle).
func (bd *nyquistBoundary) assign(i int, w0, tol float64) {
	scale := 1.0
	if !bd.continuous {
		scale = bd.dt
	}
	k := slices.IndexFunc(bd.clusters, func(c boundaryCluster) bool {
		return math.Abs(c.w0-w0)*scale < tol
	})
	if k < 0 {
		k = len(bd.clusters)
		bd.clusters = append(bd.clusters, boundaryCluster{w0: w0})
	}
	bd.cluster[i] = k
}

func (bd *nyquistBoundary) updateSpreads(poles []complex128) {
	for k := range bd.clusters {
		bd.clusters[k].spread = 0
	}
	for i, p := range poles {
		if k := bd.cluster[i]; k >= 0 {
			c := &bd.clusters[k]
			c.spread = max(c.spread, cmplx.Abs(foldUpper(p)-bd.center(c.w0)))
		}
	}
}

func foldUpper(p complex128) complex128 {
	if imag(p) < 0 {
		return cmplx.Conj(p)
	}
	return p
}

// center is the boundary point at frequency w0.
func (bd nyquistBoundary) center(w0 float64) complex128 {
	if bd.continuous {
		return complex(0, w0)
	}
	switch theta := w0 * bd.dt; {
	case theta == 0:
		return 1
	case theta >= math.Pi*(1-1e-12):
		return -1
	default:
		return cmplx.Exp(complex(0, theta))
	}
}

// indentRadius sizes the detour around cluster k so it covers the cluster's
// split poles and no other open-loop pole.
func (bd nyquistBoundary) indentRadius(k int, base float64, poles []complex128) float64 {
	c := bd.clusters[k]
	center := bd.center(c.w0)
	r := max(base, 4*c.spread)
	for i, pole := range poles {
		if bd.cluster[i] != k {
			r = min(r, cmplx.Abs(pole-center)/4)
		}
	}
	return r
}

func autoNyquistFreqs(sys *System, poles []complex128, bd nyquistBoundary, nPoints int) []float64 {
	if nPoints <= 0 {
		nPoints = 500
	}

	td := newTimeDomain(sys.Dt)
	natFreqs := make([]float64, 0, len(poles))
	for i, p := range poles {
		if k := bd.cluster[i]; k >= 0 && bd.clusters[k].w0 == 0 {
			continue
		}
		wn := td.naturalFrequency(p)
		if wn > 0 && !math.IsInf(wn, 0) {
			natFreqs = append(natFreqs, wn)
		}
	}

	wMin, wMax := 0.01, 100.0
	if len(natFreqs) > 0 {
		lo, hi := natFreqs[0], natFreqs[0]
		for _, w := range natFreqs[1:] {
			lo = min(lo, w)
			hi = max(hi, w)
		}
		wMin = max(lo/100, 1e-6)
		wMax = min(hi*100, 1e6)
	}

	if sys.IsDiscrete() {
		wMax = min(wMax, math.Pi/sys.Dt)
		wMin = min(wMin, wMax/100)
	}

	baseN := max(nPoints*7/10, 50)
	omega := logspace(math.Log10(wMin), math.Log10(wMax), baseN)

	for _, c := range bd.clusters {
		w0 := c.w0
		if w0 == 0 {
			continue
		}
		lo := max(w0*0.99, wMin)
		hi := min(w0*1.01, wMax)
		if lo < hi {
			omega = append(omega, logspace(math.Log10(lo), math.Log10(hi), 30)...)
		}
	}

	sort.Float64s(omega)
	deduped := omega[:1]
	for i := 1; i < len(omega); i++ {
		if omega[i]-deduped[len(deduped)-1] > 1e-12*omega[i] {
			deduped = append(deduped, omega[i])
		}
	}

	filtered := deduped[:0]
	for _, w := range deduped {
		tooClose := false
		for _, c := range bd.clusters {
			if c.w0 > 0 && math.Abs(w-c.w0) < 1e-4*c.w0 {
				tooClose = true
				break
			}
		}
		if !tooClose {
			filtered = append(filtered, w)
		}
	}
	return filtered
}

var errNyquistIndent = fmt.Errorf("Nyquist: boundary poles too close to indent separately: %w", ErrSingularTransform)

const (
	nyquistIndentBase   = 1e-4
	nyquistMaxPhaseStep = math.Pi / 4
	nyquistMaxDepth     = 30
	nyquistTailRatio    = 0.1
	nyquistMaxFreq      = 1e15
)

// nyquistPath samples the loop response along the positive-frequency half of
// the Nyquist contour, bisecting wherever arg(1+L) moves by more than
// nyquistMaxPhaseStep between neighbours so the winding count is not aliased.
type nyquistPath struct {
	sys   *System
	eval  func(complex128, []complex128) error
	buf   []complex128
	vals  []complex128
	seeds []float64
}

// seedResonances adds contour parameters (ω, or θ = ωdt for discrete models)
// around each pole, where the response varies on the scale of the pole's
// distance to the boundary rather than of the base grid spacing.
func (p *nyquistPath) seedResonances(poles []complex128, continuous bool) {
	for _, pole := range poles {
		at, d := imag(pole), math.Abs(real(pole))
		if !continuous {
			at, d = cmplx.Phase(pole), math.Abs(1-cmplx.Abs(pole))
		}
		if at < 0 || d == 0 {
			continue
		}
		for _, k := range []float64{-16, -8, -4, -2, -1, -0.5, -0.25, 0, 0.25, 0.5, 1, 2, 4, 8, 16} {
			if x := at + k*d; x > 0 {
				p.seeds = append(p.seeds, x)
			}
		}
	}
	sort.Float64s(p.seeds)
}

// withSeeds merges the seeds inside (a, b), mapped through toT, into ts.
func (p *nyquistPath) withSeeds(ts []float64, a, b float64, toT func(float64) float64) []float64 {
	lo := sort.SearchFloat64s(p.seeds, a)
	for _, x := range p.seeds[lo:] {
		if x >= b {
			break
		}
		if x > a {
			ts = append(ts, toT(x))
		}
	}
	sort.Float64s(ts)
	return ts
}

func (p *nyquistPath) at(s complex128) (complex128, error) {
	if err := p.eval(s, p.buf); err != nil {
		return 0, err
	}
	applyIODelayAtS(p.sys, s, p.buf, 1, 1, true)
	return p.buf[0], nil
}

// piece samples f on ts (increasing parameters), refining adaptively.
func (p *nyquistPath) piece(f func(float64) complex128, ts []float64) error {
	prevT := ts[0]
	prev, err := p.at(f(prevT))
	if err != nil {
		return err
	}
	p.vals = append(p.vals, prev)
	for _, t := range ts[1:] {
		v, err := p.at(f(t))
		if err != nil {
			return err
		}
		if err := p.refine(f, prevT, prev, t, v, 0); err != nil {
			return err
		}
		p.vals = append(p.vals, v)
		prevT, prev = t, v
	}
	return nil
}

func (p *nyquistPath) refine(f func(float64) complex128, ta float64, va complex128, tb float64, vb complex128, depth int) error {
	if depth >= nyquistMaxDepth || !(phaseStep(1+va, 1+vb) > nyquistMaxPhaseStep) {
		return nil
	}
	tm := (ta + tb) / 2
	vm, err := p.at(f(tm))
	if err != nil {
		return err
	}
	if err := p.refine(f, ta, va, tm, vm, depth+1); err != nil {
		return err
	}
	p.vals = append(p.vals, vm)
	return p.refine(f, tm, vm, tb, vb, depth+1)
}

func phaseStep(a, b complex128) float64 {
	if a == 0 || b == 0 {
		return 0
	}
	return math.Abs(cmplx.Phase(b / a))
}

func linspace(a, b float64, n int) []float64 {
	ts := make([]float64, n)
	for i := range ts {
		ts[i] = a + (b-a)*float64(i)/float64(n-1)
	}
	ts[n-1] = b
	return ts
}

// arc samples center*(1+eps*e^{jφ}) for discrete models or center+eps*e^{jφ}
// for continuous ones, φ from phi0 to phi1. The detour leaves the boundary pole
// on the stable side.
func (p *nyquistPath) arc(center complex128, eps, phi0, phi1 float64, discrete bool) error {
	return p.piece(func(phi float64) complex128 {
		d := complex(eps, 0) * cmplx.Exp(complex(0, phi))
		if discrete {
			return center * (1 + d)
		}
		return center + d
	}, linspace(phi0, phi1, 33))
}

func nyquistEncirclements(sys *System, poles []complex128, bd nyquistBoundary) (int, error) {
	e, err := validFrequencyEvaluator(sys, "Nyquist")
	if err != nil {
		return 0, err
	}
	path := &nyquistPath{sys: sys, eval: e.pointEval(), buf: make([]complex128, 1)}
	path.seedResonances(poles, sys.IsContinuous())

	var closure complex128
	if sys.IsContinuous() {
		closure = complex(sys.D.At(0, 0), 0)
		err = path.continuousHalf(poles, bd, closure)
	} else {
		err = path.discreteHalf(poles, bd)
	}
	if err != nil {
		return 0, err
	}

	half := path.vals
	full := make([]complex128, 0, 2*len(half)+1)
	for _, h := range slices.Backward(half) {
		full = append(full, cmplx.Conj(h))
	}
	full = append(full, half...)
	if sys.IsContinuous() {
		full = append(full, closure)
	}
	return -windingNumber(full, -1), nil
}

func (p *nyquistPath) continuousHalf(poles []complex128, bd nyquistBoundary, closure complex128) error {
	wLo, wHi := 1e-3, 1e3
	scales := make([]float64, 0, len(poles))
	for _, pole := range poles {
		if a := cmplx.Abs(pole); a > 0 {
			scales = append(scales, a)
		}
	}
	for _, tau := range nyquistDelays(p.sys) {
		if tau > 0 {
			scales = append(scales, 1/tau)
		}
	}
	if len(scales) > 0 {
		wLo, wHi = math.Inf(1), 0
		for _, a := range scales {
			wLo = min(wLo, a/1e3)
			wHi = max(wHi, a*1e3)
		}
	}

	axis := func(w float64) complex128 { return complex(0, w) }
	logAxis := func(u float64) complex128 { return complex(0, math.Exp(u)) }
	segment := func(a, b float64) error {
		if !(b > a) {
			return errNyquistIndent
		}
		if a == 0 {
			first := min(wLo, b/10)
			if err := p.piece(axis, []float64{0, first}); err != nil {
				return err
			}
			p.vals = p.vals[:len(p.vals)-1]
			a = first
		}
		n := max(int(math.Ceil(10*math.Log10(b/a))), 2)
		return p.piece(logAxis, p.withSeeds(linspace(math.Log(a), math.Log(b), n), a, b, math.Log))
	}

	cur := 0.0
	for k, c := range bd.clusters {
		w0 := c.w0
		r := bd.indentRadius(k, nyquistIndentBase*math.Max(1, w0), poles)
		if w0 == 0 {
			if err := p.arc(0, r, 0, math.Pi/2, false); err != nil {
				return err
			}
			cur = r
			continue
		}
		if err := segment(cur, w0-r); err != nil {
			return err
		}
		if err := p.arc(complex(0, w0), r, -math.Pi/2, math.Pi/2, false); err != nil {
			return err
		}
		cur = w0 + r
	}
	wHi = max(wHi, 10*cur)
	if err := segment(cur, wHi); err != nil {
		return err
	}

	if p.sys.HasDelay() && closure != 0 {
		return nil
	}
	target := nyquistTailRatio * math.Max(cmplx.Abs(1+closure), eps())
	for cmplx.Abs(p.vals[len(p.vals)-1]-closure) > target && wHi < nyquistMaxFreq {
		if err := segment(wHi, wHi*10); err != nil {
			return err
		}
		wHi *= 10
	}
	return nil
}

func (p *nyquistPath) discreteHalf(poles []complex128, bd nyquistBoundary) error {
	dt := p.sys.Dt
	circle := func(theta float64) complex128 { return cmplx.Exp(complex(0, theta)) }
	segment := func(a, b float64) error {
		if !(b > a) {
			return errNyquistIndent
		}
		n := max(int(math.Ceil((b-a)/(math.Pi/200))), 2)
		return p.piece(circle, p.withSeeds(linspace(a, b, n), a, b, func(x float64) float64 { return x }))
	}

	cur := 0.0
	closed := false
	for k, c := range bd.clusters {
		theta := min(c.w0*dt, math.Pi)
		center := bd.center(c.w0)
		r := bd.indentRadius(k, nyquistIndentBase, poles)
		switch center {
		case 1:
			if err := p.arc(1, r, 0, math.Pi/2, true); err != nil {
				return err
			}
			cur = r
		case -1:
			if err := segment(cur, math.Pi-r); err != nil {
				return err
			}
			if err := p.arc(-1, r, -math.Pi/2, 0, true); err != nil {
				return err
			}
			closed = true
		default:
			if err := segment(cur, theta-r); err != nil {
				return err
			}
			if err := p.arc(center, r, -math.Pi/2, math.Pi/2, true); err != nil {
				return err
			}
			cur = theta + r
		}
	}
	if closed {
		return nil
	}
	return segment(cur, math.Pi)
}

func nyquistDelays(sys *System) []float64 {
	var taus []float64
	if d := effectiveIODelayMatrix(sys, 1, 1, true); d != nil {
		taus = append(taus, d.At(0, 0))
	}
	if sys.LFT != nil {
		taus = append(taus, sys.LFT.Tau...)
	}
	return taus
}

func windingNumber(contour []complex128, point complex128) int {
	if len(contour) < 2 {
		return 0
	}

	totalAngle := 0.0
	n := len(contour)
	for k := range n {
		z1 := contour[k] - point
		z2 := contour[(k+1)%n] - point

		if cmplx.Abs(z1) < 1e-15 || cmplx.Abs(z2) < 1e-15 {
			continue
		}

		dtheta := cmplx.Phase(z2) - cmplx.Phase(z1)
		for dtheta > math.Pi {
			dtheta -= 2 * math.Pi
		}
		for dtheta < -math.Pi {
			dtheta += 2 * math.Pi
		}
		totalAngle += dtheta
	}

	return int(math.Round(totalAngle / (2 * math.Pi)))
}
