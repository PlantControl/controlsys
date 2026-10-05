package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// errDelayPeakUnsupported reports a peak gain the delay path cannot certify.
var errDelayPeakUnsupported = fmt.Errorf("peak gain outside the certified delay scope: %w", ErrDelayUnsupported)

// delayLFT evaluates a continuous model with internal delays,
//
//	G(s) = H11 + H12·Δ(I − H22·Δ)⁻¹·H21,  Hij = Dij + Ci(sI−A)⁻¹Bj,  Δ = diag(e^{−sτ}),
//
// whose characteristic function χ(s) = det(sI−A)·det(I − H22(s)Δ(s)) is
// entire, so its closed-loop roots are the zeros of χ.
type delayLFT struct {
	sys     *System
	n, m, p int
	nd      int
	tau     []float64
	td      timeDomain

	h22sys *System
	h22    func(frequencyPoint, []complex128) error
	hbuf   []complex128
	lu     []complex128
	piv    []int

	g       func(frequencyPoint, []complex128) error
	gbuf    []complex128
	ioDelay *mat.Dense
	svd     *complexSVDWorkspace

	poles []complex128
	normA float64
	// spectral norms of C, C2, B, B2 and the D blocks
	c1, c2, b1, b2     float64
	d11, d12, d21, d22 float64
	directFeedthrough  bool
	desc               *descriptorResponse // for certified peaks
	err                error
}

func newDelayLFT(sys *System) (*delayLFT, error) {
	n, m, p := sys.Dims()
	lft := sys.LFT
	nd := len(lft.Tau)
	e := &delayLFT{
		sys: sys, n: n, m: m, p: p, nd: nd,
		tau:  lft.Tau,
		td:   newTimeDomain(0),
		hbuf: make([]complex128, nd*nd),
		lu:   make([]complex128, nd*nd),
		piv:  make([]int, nd),
		gbuf: make([]complex128, p*m),
		g:    newFrequencyEvaluator(sys).pointEval(),
	}
	e.ioDelay = effectiveIODelayMatrix(sys, p, m, true)
	if p > 2 || m > 2 {
		e.svd = newComplexSVDWorkspace(p, m)
	}
	if n > 0 {
		h22sys := &System{A: sys.A, B: lft.B2, C: lft.C2, D: lft.D22}
		if err := h22sys.Validate(); err != nil {
			return nil, err
		}
		e.h22sys = h22sys
		e.h22 = newFrequencyEvaluator(h22sys).pointSolver(math.MaxInt).evalInto
		poles, err := h22sys.Poles()
		if err != nil {
			return nil, err
		}
		e.poles = poles
		e.normA = mat.Norm(sys.A, 2)
		e.c1, e.c2 = spectralNorm(sys.C), spectralNorm(lft.C2)
		e.b1, e.b2 = spectralNorm(sys.B), spectralNorm(lft.B2)
	}
	e.d11, e.d12 = spectralNorm(sys.D), spectralNorm(lft.D12)
	e.d21, e.d22 = spectralNorm(lft.D21), spectralNorm(lft.D22)
	e.directFeedthrough = lftHasDirectFeedthrough(lft)
	return e, nil
}

func spectralNorm(m *mat.Dense) float64 {
	if m == nil || m.IsEmpty() {
		return 0
	}
	return mat.Norm(m, 2)
}

func (e *delayLFT) fail(w float64, err error) {
	if e.err == nil {
		e.err = fmt.Errorf("response at ω=%g: %w", w, err)
	}
}

// charMinusOne returns det(I − H22(jω)Δ(jω)) − 1, the loop L of the Nyquist
// test whose 1+L has the zeros of χ and the poles of eig(A) it does not
// cancel.
func (e *delayLFT) charMinusOne(w float64) complex128 {
	pt := e.td.frequencyPoint(w)
	s := pt.value()
	nd := e.nd
	if e.h22 != nil {
		if err := evalWithPoleLimit(e.h22, e.h22sys, pt, e.hbuf); err != nil {
			e.fail(w, err)
			return complex(math.NaN(), math.NaN())
		}
	} else {
		clear(e.hbuf)
		if d22 := e.sys.LFT.D22; d22 != nil {
			for i := range nd {
				for j := range nd {
					e.hbuf[i*nd+j] = complex(d22.At(i, j), 0)
				}
			}
		}
	}
	for j := range nd {
		dj := cmplx.Exp(-s * complex(e.tau[j], 0))
		for i := range nd {
			e.lu[i*nd+j] = -e.hbuf[i*nd+j] * dj
		}
	}
	for i := range nd {
		e.lu[i*nd+i]++
	}
	if !cLUFactor(e.lu, e.piv, nd) {
		return -1
	}
	det := complex(1, 0)
	for k := range nd {
		det *= e.lu[k*nd+k]
		if e.piv[k] != k {
			det = -det
		}
	}
	return det - 1
}

// tail bounds |det(I − H22Δ) − 1| for Re s ≥ 0, |s| ≥ w by
// (1 + ‖H22‖)^nd − 1, since |det(I+M) − 1| ≤ ∏(1+σᵢ(M)) − 1 and
// ‖H22(s)‖ ≤ ‖D22‖ + ‖C2‖‖B2‖/(|s| − ‖A‖).
func (e *delayLFT) tail(w float64) float64 {
	h := e.h(e.d22, e.c2, e.b2, w)
	return math.Pow(1+h, float64(e.nd)) - 1
}

// h bounds ‖D + C(sI−A)⁻¹B‖ for |s| ≥ w.
func (e *delayLFT) h(d, c, b, w float64) float64 {
	cb := c * b
	if cb == 0 || math.IsInf(w, 1) {
		return d
	}
	if w <= e.normA {
		return math.Inf(1)
	}
	return d + cb/(w-e.normA)
}

func (e *delayLFT) loop() *delayLoop {
	l := &delayLoop{at: e.charMinusOne, tail: e.tail, tailLimit: e.tail(math.Inf(1))}
	for _, t := range e.tau {
		l.tau += t
	}
	l.addPoles(e.poles)
	return l
}

// sigma returns σ_max(G(jω)) with the exact delay factors.
func (e *delayLFT) sigma(w float64) float64 {
	pt := e.td.frequencyPoint(w)
	if err := evalWithPoleLimit(e.g, e.sys, pt, e.gbuf); err != nil {
		e.fail(w, err)
		return math.NaN()
	}
	if e.ioDelay != nil {
		applyIODelayMatrixAtS(e.sys, pt.value(), e.gbuf, e.p, e.m, e.ioDelay)
	}
	sv, err := e.svd.maximumFromFlat(e.gbuf, 0, e.p, e.m)
	if err != nil {
		e.fail(w, err)
		return math.NaN()
	}
	return sv
}

// highFrequencyGain evaluates σ_max(G∞(θ)), G∞ = D11 + D12·Δ(I − D22·Δ)⁻¹·D21
// with Δ = diag(e^{−jθ}), the values G(jω) approaches as ω → ∞.
type highFrequencyGain struct {
	e       *delayLFT
	d11     []complex128
	d12     []complex128
	d21     []complex128
	d22     []complex128
	lu, x   []complex128
	piv     []int
	g       []complex128
	lip     float64
	samples int
}

func newHighFrequencyGain(e *delayLFT) *highFrequencyGain {
	lft := e.sys.LFT
	nd, p, m := e.nd, e.p, e.m
	toC := func(d *mat.Dense, r, c int) []complex128 {
		out := make([]complex128, r*c)
		if d == nil {
			return out
		}
		for i := range r {
			for j := range c {
				out[i*c+j] = complex(d.At(i, j), 0)
			}
		}
		return out
	}
	return &highFrequencyGain{
		e:   e,
		d11: toC(e.sys.D, p, m), d12: toC(lft.D12, p, nd),
		d21: toC(lft.D21, nd, m), d22: toC(lft.D22, nd, nd),
		lu: make([]complex128, nd*nd), x: make([]complex128, nd*m),
		piv: make([]int, nd), g: make([]complex128, p*m),
		lip: e.d12 * e.d21 / ((1 - e.d22) * (1 - e.d22)),
	}
}

func (h *highFrequencyGain) at(theta []float64) float64 {
	h.samples++
	nd, p, m := h.e.nd, h.e.p, h.e.m
	for i := range nd {
		for j := range nd {
			h.lu[i*nd+j] = -h.d22[i*nd+j] * cmplx.Exp(complex(0, -theta[j]))
		}
		h.lu[i*nd+i]++
	}
	if !cLUFactor(h.lu, h.piv, nd) {
		return math.Inf(1)
	}
	copy(h.x, h.d21)
	cLUSolve(h.lu, h.piv, h.x, nd, m)
	copy(h.g, h.d11)
	for i := range p {
		for k := range nd {
			c := h.d12[i*nd+k] * cmplx.Exp(complex(0, -theta[k]))
			if c == 0 {
				continue
			}
			for j := range m {
				h.g[i*m+j] += c * h.x[k*m+j]
			}
		}
	}
	sv, err := h.e.svd.maximumFromFlat(h.g, 0, p, m)
	if err != nil {
		return math.NaN()
	}
	return sv
}

// oneDelaySup returns sup_θ σ_max(G∞(θ)) for one internal delay, where θ =
// ωτ sweeps the whole circle as ω → ∞: the largest of 1024 samples, each
// local maximum refined by golden-section search.
func (h *highFrequencyGain) oneDelaySup() float64 {
	theta := make([]float64, 1)
	at := func(t float64) float64 {
		theta[0] = t
		return h.at(theta)
	}
	const k = 1024
	step := 2 * math.Pi / k
	vals := make([]float64, k)
	for i := range k {
		vals[i] = at(float64(i) * step)
	}
	best := math.Inf(-1)
	for i, v := range vals {
		if v < vals[(i+k-1)%k] || v < vals[(i+1)%k] {
			continue
		}
		t := float64(i) * step
		_, g := goldenMax(t-step, t+step, t, v, at)
		best = max(best, g)
	}
	return best
}

// torusGrid returns the centres of a uniform grid of about 4096 cells on
// the torus of phases and their half-width.
func (h *highFrequencyGain) torusGrid() ([][]float64, float64) {
	nd := h.e.nd
	n0 := max(2, int(math.Pow(4096, 1/float64(nd))))
	r := math.Pi / float64(n0)
	var cells [][]float64
	idx := make([]int, nd)
	for {
		c := make([]float64, nd)
		for i, k := range idx {
			c[i] = (2*float64(k) + 1) * r
		}
		cells = append(cells, c)
		i := 0
		for ; i < nd; i++ {
			idx[i]++
			if idx[i] < n0 {
				break
			}
			idx[i] = 0
		}
		if i == nd {
			return cells, r
		}
	}
}

// estimate returns a lower estimate of sup σ_max(G∞): the exact supremum
// for one delay, else the largest sample of torusGrid.
func (h *highFrequencyGain) estimate() float64 {
	if h.e.nd == 1 {
		return h.oneDelaySup()
	}
	cells, _ := h.torusGrid()
	best := math.Inf(-1)
	for _, c := range cells {
		best = max(best, h.at(c))
	}
	return best
}

// below reports whether σ_max(G∞) <= level on the whole torus of phases,
// by branch and bound: G∞ is Lipschitz in θ with constant
// ‖D12‖‖D21‖/(1 − ‖D22‖)² in the max-norm, since
// dΔ(I − D22Δ)⁻¹ = (I − ΔD22)⁻¹·dΔ·(I − D22Δ)⁻¹, so a cell of half-width r
// around a sample v cannot exceed v + lip·r. Cells that may exceed level are
// split until none remain, a sample exceeds level, or the sample budget is
// spent.
func (h *highFrequencyGain) below(level float64, budget int) bool {
	nd := h.e.nd
	cells, r := h.torusGrid()
	for len(cells) > 0 {
		var next [][]float64
		for _, c := range cells {
			v := h.at(c)
			if !(v <= level) {
				return false
			}
			if v+h.lip*r <= level {
				continue
			}
			for mask := range 1 << nd {
				child := make([]float64, nd)
				for i := range nd {
					child[i] = c[i] - r/2
					if mask&(1<<i) != 0 {
						child[i] = c[i] + r/2
					}
				}
				next = append(next, child)
			}
		}
		if h.samples+len(next) > budget {
			return false
		}
		cells, r = next, r/2
	}
	return true
}

// tailGain bounds σ_max(G(jω)) for ω ≥ w by lim + ‖G − G∞‖, with
// ‖G − G∞‖ ≤ (h11 − d11) + h12·h21/(1 − h22) − d12·d21/(1 − d22) from the
// resolvent bound on each block.
func (e *delayLFT) tailGain(lim, w float64) float64 {
	h11 := e.h(e.d11, e.c1, e.b1, w)
	h12 := e.h(e.d12, e.c1, e.b2, w)
	h21 := e.h(e.d21, e.c2, e.b1, w)
	h22 := e.h(e.d22, e.c2, e.b2, w)
	if !(h22 < 1) {
		return math.Inf(1)
	}
	return lim + (h11 - e.d11) + h12*h21/(1-h22) - e.d12*e.d21/(1-e.d22)
}

// stability counts the closed-loop right-half-plane roots of χ with the
// Nyquist test of the delay loop.
func (e *delayLFT) stability() (*delayLoop, delayNyquist, error) {
	l := e.loop()
	res, err := l.nyquist(0.5)
	if err == nil && e.err != nil {
		err = e.err
	}
	return l, res, err
}

// internalDelayStable reports whether every root of χ of a continuous model
// with internal delays lies in the open left half-plane.
func internalDelayStable(sys *System) (bool, error) {
	e, err := newDelayLFT(sys)
	if err != nil {
		return false, err
	}
	_, res, err := e.stability()
	if err != nil {
		return false, err
	}
	return res.stable, nil
}

// hinfNormDelayed returns the H∞ norm of a continuous model with internal
// delays, or +Inf for an unstable one, and whether the model is stable.
func hinfNormDelayed(sys *System) (norm, omega float64, stable bool, err error) {
	e, l, res, err := delayPeakSetup(sys)
	if err != nil {
		return 0, 0, false, err
	}
	if !res.stable {
		return math.Inf(1), math.Inf(1), false, nil
	}
	norm, omega, err = e.peakOrZero(l, res)
	if err != nil {
		return 0, 0, false, err
	}
	return norm, omega, true, nil
}

// linfNormDelayed returns the L∞ norm of a continuous model with internal
// delays, sup_ω σ_max(G(jω)) regardless of stability: +Inf at ω = axisW when χ
// has a root on the imaginary axis, else the peak over the Nyquist grid, which
// is complete whenever the RHP roots are counted.
func linfNormDelayed(sys *System) (norm, omega float64, err error) {
	e, l, res, err := delayPeakSetup(sys)
	if err != nil {
		return 0, 0, err
	}
	if res.axisRoot {
		return math.Inf(1), res.axisW, nil
	}
	return e.peakOrZero(l, res)
}

func delayPeakSetup(sys *System) (*delayLFT, *delayLoop, delayNyquist, error) {
	e, err := newDelayLFT(sys)
	if err != nil {
		return nil, nil, delayNyquist{}, err
	}
	l, res, err := e.stability()
	if err != nil {
		return nil, nil, delayNyquist{}, err
	}
	return e, l, res, nil
}

func (e *delayLFT) peakOrZero(l *delayLoop, res delayNyquist) (float64, float64, error) {
	if e.m == 0 || e.p == 0 {
		return 0, 0, nil
	}
	return e.peak(l, res.w)
}

// peak searches sup_ω σ_max(G(jω)) on the Nyquist grid, which bisects
// wherever arg χ turns by more than π/4 and so is densest at the lightly
// damped closed-loop roots where G resonates, together with the midpoints of
// its intervals. Each local maximum is refined by golden-section search on
// the exact response. Past the grid end the gain is bounded by tailGain; the
// delay grid of the Nyquist test is extended until that bound stays below
// the peak. When the high-frequency limit exceeds every finite sample it is
// reported at ω = +Inf, as σ_max(D) is for a rational model, once no finite
// sample on the grid exceeds it. Between samples, up to the grid end, the
// peak is certified to peakCertTol by descriptorResponse.certify.
func (e *delayLFT) peak(l *delayLoop, grid []float64) (float64, float64, error) {
	const tol = 1e-10
	best, wBest, err := e.scan(grid)
	if err != nil {
		return 0, 0, err
	}
	lim := e.d11
	if e.directFeedthrough {
		hf := newHighFrequencyGain(e)
		est := hf.estimate()
		level := (est + best) / 2
		switch {
		case est < best && hf.below(level, 1<<21):
			lim = level
		case e.nd == 1 && est > best*(1+tol):
			lim = est
		default:
			return 0, 0, fmt.Errorf("gain may peak as ω → ∞ through a delayed feedthrough: %w", errDelayPeakUnsupported)
		}
	}
	cert := append([]float64{0}, grid...)
	if lim > best*(1+tol) {
		top, wTop, err := e.certify(l, cert, lim, math.Inf(1))
		if err != nil {
			return 0, 0, err
		}
		if top == lim {
			return lim, math.Inf(1), nil
		}
		best, wBest = top, wTop
	} else if best, wBest, err = e.certify(l, cert, best, wBest); err != nil {
		return 0, 0, err
	}

	rGrid := grid[len(grid)-1]
	R := max(rGrid, 2*e.normA+1)
	for range 200 {
		if e.tailGain(lim, R) <= best*(1+tol) {
			break
		}
		R *= 2
	}
	if !(e.tailGain(lim, R) <= best*(1+tol)) {
		return 0, 0, fmt.Errorf("no finite frequency bound for the gain: %w", errDelayPeakUnsupported)
	}
	if R <= rGrid {
		return best, wBest, nil
	}
	if l.tau > 0 && (R-rGrid)*8*l.tau/math.Pi > float64(l.pointBudget()) {
		return 0, 0, fmt.Errorf("delay grid exceeds %d points: %w", l.pointBudget(), errDelayPeakUnsupported)
	}
	ext := []float64{rGrid}
	for _, w := range l.grid(R, R) {
		if w > rGrid {
			ext = append(ext, w)
		}
	}
	b, w, err := e.scan(ext)
	if err != nil {
		return 0, 0, err
	}
	if b > best {
		best, wBest = b, w
	}
	return e.certify(l, ext, best, wBest)
}

// certify raises (peak, w) to within peakCertTol of sup σ_max(G(jω)) over
// the span of the sorted grid ws; see descriptorResponse.certify.
func (e *delayLFT) certify(l *delayLoop, ws []float64, peak, w float64) (float64, float64, error) {
	if e.desc == nil {
		desc, err := newDescriptorResponse(e.sys)
		if err != nil {
			return 0, 0, err
		}
		e.desc = desc
	}
	ps, wps := []float64{peak}, []float64{w}
	if err := e.desc.certify(ws, []float64{0}, ps, wps, l.pointBudget(), errDelayPeakUnsupported); err != nil {
		return 0, 0, err
	}
	return ps[0], wps[0], nil
}

// scan samples σ_max on the sorted grid and its interval midpoints and
// refines every local maximum within a factor 2 of the largest sample.
func (e *delayLFT) scan(grid []float64) (float64, float64, error) {
	ws := make([]float64, 0, 2*len(grid))
	for i, w := range grid {
		if i > 0 {
			ws = append(ws, (grid[i-1]+w)/2)
		}
		ws = append(ws, w)
	}
	vals := make([]float64, len(ws))
	for i, w := range ws {
		vals[i] = e.sigma(w)
	}
	if e.err != nil {
		return 0, 0, e.err
	}
	if slices.ContainsFunc(vals, func(v float64) bool { return math.IsInf(v, 0) }) {
		return 0, 0, fmt.Errorf("gain unbounded on the imaginary axis: %w", errDelayPeakUnsupported)
	}
	top := slices.Max(vals)
	best, wBest := math.Inf(-1), 0.0
	for k, v := range vals {
		if v < 0.5*top || k > 0 && v < vals[k-1] || k+1 < len(vals) && v < vals[k+1] {
			continue
		}
		w, g := goldenMax(ws[max(k-1, 0)], ws[min(k+1, len(ws)-1)], ws[k], v, e.sigma)
		if g > best {
			best, wBest = g, w
		}
	}
	if e.err != nil {
		return 0, 0, e.err
	}
	return best, wBest, nil
}
