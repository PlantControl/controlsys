package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"

	"plantcontrol.org/v1/gonum/mat"
)

// NyquistResult is the Nyquist response and stability count of a SISO loop;
// see (*System).Nyquist.
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
// never depend on omega. When 1+L vanishes on the contour (closed-loop poles
// on the stability boundary) the count is undefined and Nyquist returns
// ErrSingularTransform. A delay loop whose winding needs more than 2^21
// samples (|L| > 1 out to ω with ωτ in the millions), or where 1+L vanishes
// within tolerance past ωτ = 1e5 (closed-loop roots closer to the axis than
// the response resolves), returns ErrDelayUnsupported. A neutral delay loop, whose high-frequency gain may
// reach 1, is counted by the closed-loop delay Nyquist test when its
// difference operator 1 + L∞ is certified free of zeros in Re s >= 0
// (d1·e^{−s} + d2·e^{−2s} with 1 + d1·z + d2·z² nonzero on |z| <= 1), and
// otherwise returns ErrDelayUnsupported: for |D| >= 1 with I/O delays the
// closed loop has infinitely many roots near Re s = ln|D|/τ.
//
// omega nil selects the auto grid with nPoints samples (0 means 500); a
// non-nil omega must be non-empty, finite, non-negative and, for discrete
// models, at most π/dt. Negative nPoints or an invalid omega returns
// ErrInvalidArgument.
func (sys *System) Nyquist(omega []float64, nPoints int) (*NyquistResult, error) {
	if err := requireSystem("Nyquist", sys); err != nil {
		return nil, err
	}
	if _, err := newSISOLoopModel(sys, "Nyquist"); err != nil {
		return nil, err
	}
	if err := validateNyquistGrid(sys, omega, nPoints); err != nil {
		return nil, fmt.Errorf("Nyquist: %w", err)
	}

	poles, err := sys.Poles()
	if err != nil {
		return nil, fmt.Errorf("Nyquist: %w", err)
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
			return nil, fmt.Errorf("Nyquist: %w", err)
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
	if errors.Is(err, errNeutralDelayTail) {
		if z, zerr := neutralLoopRoots(sys); zerr == nil {
			enc, err = z-rhpPoles, nil
		} else if !errors.Is(zerr, errNeutralDelayTail) {
			err = zerr
		}
	}
	if err != nil {
		return nil, fmt.Errorf("Nyquist: %w", err)
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

var errNeutralDelayTail = fmt.Errorf("delay loop tail |L(jω)| < 1 not certified (neutral, |L(j∞)| may reach 1): %w", errDelayLoopUnsupported)

var errNyquistIndent = fmt.Errorf("boundary poles too close to indent separately: %w", ErrSingularTransform)

func validateNyquistGrid(sys *System, omega []float64, nPoints int) error {
	if nPoints < 0 {
		return fmt.Errorf("nPoints %d must be non-negative (0 = default): %w", nPoints, ErrInvalidArgument)
	}
	if omega == nil {
		return nil
	}
	if len(omega) == 0 {
		return fmt.Errorf("omega is empty; pass nil for the automatic grid: %w", ErrInvalidArgument)
	}
	nyq := math.Inf(1)
	if sys.IsDiscrete() {
		nyq = math.Pi / sys.Dt
	}
	for k, w := range omega {
		if !isFinite(w) || w < 0 || w > nyq*(1+1e-12) {
			return fmt.Errorf("omega[%d] = %g must be finite and in [0, %g]: %w", k, w, nyq, ErrInvalidArgument)
		}
	}
	return nil
}

const (
	nyquistIndentBase   = 1e-4
	nyquistMaxPhaseStep = math.Pi / 4
	nyquistMaxDepth     = 30
	nyquistTailRatio    = 0.1
	nyquistMaxFreq      = 1e15
	nyquistOnContourTol = 1e-12
	// nyquistDelayResolution is the ωτ past which a delay loop's closed-loop
	// roots near the axis, at Re s ~ 2π/(ωτ²) for a high-gain loop, fall
	// within the on-contour tolerance of 1+L.
	nyquistDelayResolution = 1e5
)

// nyquistPath samples the loop response along the positive-frequency half of
// the Nyquist contour, bisecting wherever arg(1+L) moves by more than
// nyquistMaxPhaseStep between neighbours so the winding count is not aliased.
type nyquistPath struct {
	sys   *System
	eval  func(frequencyPoint, []complex128) error
	buf   []complex128
	vals  []complex128
	seeds []float64
	// tau is the total loop delay; delayAliases splits steps below wDelay.
	tau, wDelay float64
	tailLim     float64 // tail bound on limsup |L(jω)|, > 0 with LFT feedthrough D12·D21 even when D11 = 0
	// envelope bounds |L| at the last sample over all delay values, from
	// the delay-free channels; nil means |L|, exact for I/O delays.
	envelope func() float64
	env      float64 // envelope at the last sample
	samples  int     // evaluations, capped at delayNyquistMaxPoints for delay loops
	ioDelay  *mat.Dense
	// onContour records that 1+L vanished on the contour: a sample came
	// within nyquistOnContourTol of -1 or bisection could not resolve arg(1+L) at nyquistMaxDepth.
	onContour bool
	contourW  float64 // smallest |ω| where onContour was set
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

func (p *nyquistPath) markOnContour(s complex128) {
	w := math.Abs(imag(s))
	if p.onContour {
		w = min(w, p.contourW)
	}
	p.onContour, p.contourW = true, w
}

func (p *nyquistPath) at(s complex128) (complex128, error) {
	if p.samples++; p.tau > 0 && p.samples > delayNyquistMaxPoints {
		return 0, fmt.Errorf("delay loop winding needs more than %d samples: %w", delayNyquistMaxPoints, ErrDelayUnsupported)
	}
	if err := p.eval(pointAt(s), p.buf); err != nil {
		return 0, err
	}
	if p.ioDelay != nil {
		applyIODelayMatrixAtS(p.sys, pointAt(s), p.buf, 1, 1, p.ioDelay)
	}
	if p.envelope != nil {
		p.env = p.envelope()
	} else {
		p.env = cmplx.Abs(p.buf[0])
	}
	if v := p.buf[0]; cmplx.Abs(1+v) <= nyquistOnContourTol*math.Max(1, cmplx.Abs(v)) {
		p.markOnContour(s)
	}
	return p.buf[0], nil
}

// piece samples f on ts (increasing parameters), refining adaptively.
func (p *nyquistPath) piece(f func(float64) complex128, ts []float64) error {
	prevT := ts[0]
	prev, err := p.at(f(prevT))
	if err != nil {
		return err
	}
	prevE := p.env
	p.vals = append(p.vals, prev)
	for _, t := range ts[1:] {
		v, err := p.at(f(t))
		if err != nil {
			return err
		}
		e := p.env
		if err := p.refine(f, nyquistSample{prevT, prev, prevE}, nyquistSample{t, v, e}, 0); err != nil {
			return err
		}
		p.vals = append(p.vals, v)
		prevT, prev, prevE = t, v, e
	}
	return nil
}

// nyquistSample is a contour parameter t with L and its envelope there.
type nyquistSample struct {
	t   float64
	v   complex128
	env float64
}

func (p *nyquistPath) refine(f func(float64) complex128, a, b nyquistSample, depth int) error {
	var rot bool
	if p.tau > 0 {
		var subunit bool
		if rot, subunit = p.delayAliases(f(a.t), f(b.t), a.env, b.env); subunit {
			return nil
		}
	}
	if !rot && !(phaseStep(1+a.v, 1+b.v) > nyquistMaxPhaseStep) {
		return nil
	}
	if depth >= nyquistMaxDepth && !rot || depth >= 2*nyquistMaxDepth {
		p.markOnContour(f(a.t))
		return nil
	}
	tm := (a.t + b.t) / 2
	vm, err := p.at(f(tm))
	if err != nil {
		return err
	}
	m := nyquistSample{tm, vm, p.env}
	if err := p.refine(f, a, m, depth+1); err != nil {
		return err
	}
	p.vals = append(p.vals, vm)
	return p.refine(f, m, b, depth+1)
}

// delayAliases reports a step from sa to sb along which e^{-sτ} turns by
// more than nyquistMaxPhaseStep while |L| may reach 1: there arg(1+L) can
// turn by π or more between the samples, which phaseStep would alias. Where
// |L| < 1, 1+L stays in the right half-plane and cannot wind; subunit reports
// such a turning step, whose endpoints then give the exact arg(1+L) change.
// The envelope ea, eb of |L| at the endpoints depends only on delay-free
// magnitudes, smooth on the seeded base grid, so max(ea, eb) < 1/2, or
// < 1 by more than |eb-ea|, rules |L| >= 1 out; |L| itself ripples on the
// scale 1/τ with internal delays. Beyond wDelay the tail bound rules it out.
func (p *nyquistPath) delayAliases(sa, sb complex128, ea, eb float64) (rot, subunit bool) {
	if !(cmplx.Abs(sb-sa)*p.tau > nyquistMaxPhaseStep) {
		return false, false
	}
	if math.Min(math.Abs(imag(sa)), math.Abs(imag(sb))) >= p.wDelay {
		return false, false
	}
	hi := max(ea, eb)
	if hi < 0.5 {
		return false, false
	}
	if hi+math.Abs(eb-ea) < 1 {
		return false, true
	}
	return true, false
}

// lftEnvelope bounds |L(s)| over |Δ| <= 1 by |H11| + |H12|·Σ_k |H22|^k·|H21|
// from the delay-free channels H at the last evaluation, for an acyclic delay
// network with nd delays (H22 nilpotent), or +Inf when H was not formed.
func lftEnvelope(ws *lftWorkspace, nd int) func() float64 {
	k := 1 + nd
	sum := make([]float64, nd*nd)
	pow := make([]float64, nd*nd)
	next := make([]float64, nd*nd)
	h22 := make([]float64, nd*nd)
	return func() float64 {
		if !ws.hOK {
			return math.Inf(1)
		}
		h := ws.h
		for i := range nd {
			for j := range nd {
				h22[i*nd+j] = cmplx.Abs(h[(1+i)*k+1+j])
			}
		}
		clear(sum)
		clear(pow)
		for i := range nd {
			sum[i*nd+i], pow[i*nd+i] = 1, 1
		}
		for range nd - 1 {
			clear(next)
			for i := range nd {
				for l := range nd {
					if q := pow[i*nd+l]; q != 0 {
						for j := range nd {
							next[i*nd+j] += q * h22[l*nd+j]
						}
					}
				}
			}
			pow, next = next, pow
			for i := range sum {
				sum[i] += pow[i]
			}
		}
		bound := cmplx.Abs(h[0])
		for i := range nd {
			for j := range nd {
				bound += cmplx.Abs(h[1+i]) * sum[i*nd+j] * cmplx.Abs(h[(1+j)*k])
			}
		}
		return bound
	}
}

// delayTailFreq returns a frequency beyond which the rational tail bound
// certifies |L(s)| <= (1+|L(j∞)|)/2 < 1 on the closed right half-plane, or
// +Inf when it cannot, and the bound's limit lim at j∞ (+Inf when unknown).
func delayTailFreq(sys *System) (w, lim float64) {
	bound, err := newRationalTailBound(sys)
	if err != nil {
		return math.Inf(1), math.Inf(1)
	}
	lim = bound.at(math.Inf(1))
	if !(lim < 1) {
		return math.Inf(1), lim
	}
	target := (1 + lim) / 2
	hi := 1.0
	for hi < math.MaxFloat64 && !(bound.at(hi) <= target) {
		hi *= 2
	}
	if !(bound.at(hi) <= target) {
		return math.Inf(1), lim
	}
	lo := hi / 2
	for range 10 {
		mid := (lo + hi) / 2
		if bound.at(mid) <= target {
			hi = mid
		} else {
			lo = mid
		}
	}
	return hi, lim
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

// neutralLoopRoots counts the closed-loop RHP roots of a neutral delay loop
// by the delay Nyquist test of 1/(1+L), normalized by its difference
// operator 1 + L∞: with that operator certified free of zeros in Re s >= 0
// (for L = d1·e^{−s} + d2·e^{−2s}, the roots of 1 + d1·z + d2·z² outside
// the closed unit disk, as for d1 = d2 = 0.6 although |d1| + |d2| > 1), χ
// has finitely many RHP roots. Otherwise it returns errNeutralDelayTail.
func neutralLoopRoots(sys *System) (int, error) {
	eye, err := NewGain(mat.NewDense(1, 1, []float64{1}), sys.Dt)
	if err != nil {
		return 0, err
	}
	cl, err := Feedback(eye, sys, -1)
	if err != nil || cl.LFT == nil {
		return 0, errNeutralDelayTail
	}
	e, err := newDelayLFT(cl)
	if err != nil || e.unstableDifferenceOperator() || math.IsInf(e.difference().rhp, 1) {
		return 0, errNeutralDelayTail
	}
	_, res, err := e.stability()
	if err != nil {
		return 0, err
	}
	if res.axisRoot {
		return 0, fmt.Errorf("closed loop has poles on the stability boundary (1+L = 0 on the contour): %w", ErrSingularTransform)
	}
	return res.roots, nil
}

func nyquistEncirclements(sys *System, poles []complex128, bd nyquistBoundary) (int, error) {
	e, err := validFrequencyEvaluator(sys, "Nyquist")
	if err != nil {
		return 0, err
	}
	path := &nyquistPath{sys: sys, eval: e.pointEval(), buf: make([]complex128, 1), ioDelay: effectiveIODelayMatrix(sys, 1, 1, true)}
	path.seedResonances(poles, sys.IsContinuous())

	var closure complex128
	if sys.IsContinuous() {
		if path.tau = sisoLoopDelay(sys); path.tau > 0 {
			path.wDelay, path.tailLim = delayTailFreq(sys)
			if n, _, _ := sys.Dims(); sys.HasInternalDelay() && lftDelayLoopAcyclic(sys, n) {
				nd := sys.internalDelayCount()
				ws := newLFTWorkspace(sys, n, nd, 1, 1)
				path.eval = func(pt frequencyPoint, dst []complex128) error {
					if err := evalFrLFTInto(ws, sys, pt, nd, 1, 1); err != nil {
						return err
					}
					dst[0] = ws.g[0]
					return nil
				}
				path.envelope = lftEnvelope(ws, nd)
			}
		}
		closure = complex(sys.D.At(0, 0), 0)
		err = path.continuousHalf(poles, bd, closure)
	} else {
		err = path.discreteHalf(poles, bd)
	}
	if err != nil {
		return 0, err
	}
	if path.onContour && path.tau > 0 && path.contourW*path.tau > nyquistDelayResolution {
		return 0, fmt.Errorf("delay loop closed-loop roots within resolution of the axis at ωτ = %.3g: %w", path.contourW*path.tau, errDelayLoopUnsupported)
	}
	if path.onContour {
		return 0, fmt.Errorf("closed loop has poles on the stability boundary (1+L = 0 on the contour): %w", ErrSingularTransform)
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

	// With at most one internal delay the tail bound's limit is the exact
	// limsup |L(jω)|; with more it can exceed it, so only a nonzero closure
	// rules out the rational tail below.
	if p.sys.HasDelay() && math.IsInf(p.wDelay, 1) && (closure != 0 || p.sys.internalDelayCount() <= 1) {
		return errNeutralDelayTail
	}
	// LFT feedthrough D12·D21 keeps limsup |L(jω)| > 0 with D = 0, so the
	// rational tail below never settles; the certified wDelay ends it instead.
	if p.sys.HasDelay() && (closure != 0 || p.tailLim > 0 && !math.IsInf(p.wDelay, 1)) {
		if w1 := p.wDelay; w1 > wHi {
			return segment(wHi, w1)
		}
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
