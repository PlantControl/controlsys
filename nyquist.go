package controlsys

import (
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
// poles under unit negative feedback. Poles within cbrt(eps) of the boundary,
// relative to the largest pole magnitude, count as on it, so closed-loop poles
// that close to a boundary pole are not resolved. The counts never depend on
// omega.
func (sys *System) Nyquist(omega []float64, nPoints int) (*NyquistResult, error) {
	if _, err := newSISOLoopModel(sys, "Nyquist"); err != nil {
		return nil, err
	}

	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}

	maxAbs := 0.0
	for _, pole := range poles {
		if a := cmplx.Abs(pole); a > maxAbs {
			maxAbs = a
		}
	}
	tol := math.Cbrt(eps()) * math.Max(maxAbs, 1)

	rhpPoles := countRHPPoles(poles, sys.IsContinuous(), tol)
	boundary := findBoundaryPoleFreqs(poles, sys.IsContinuous(), sys.Dt, tol)

	if omega == nil {
		omega = autoNyquistFreqs(sys, poles, boundary, nPoints, tol)
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

	enc, err := nyquistEncirclements(sys, poles, boundary, tol)
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

func countRHPPoles(poles []complex128, continuous bool, tol float64) int {
	count := 0
	for _, pole := range poles {
		if poleOutsideStabilityBoundary(pole, continuous, tol) {
			count++
		}
	}
	return count
}

// boundaryPoleFreq reports whether pole lies on the stability boundary and
// its non-negative frequency there. tol is cbrt(eps)-sized relative to the
// largest pole because computed repeated boundary poles (double and triple
// integrators) split by about eps^(1/m); evaluating the response between the
// split poles would be singular.
func boundaryPoleFreq(pole complex128, continuous bool, dt, tol float64) (float64, bool) {
	if !poleOnStabilityBoundary(pole, continuous, tol) {
		return 0, false
	}
	w0 := math.Abs(imag(pole))
	if !continuous {
		w0 = math.Abs(cmplx.Phase(pole))
	}
	if w0 < tol {
		return 0, true
	}
	if !continuous {
		w0 /= dt
	}
	return w0, true
}

// findBoundaryPoleFreqs returns the distinct frequencies of poles on the
// stability boundary, sorted ascending.
func findBoundaryPoleFreqs(poles []complex128, continuous bool, dt, tol float64) []float64 {
	scale := 1.0
	if !continuous {
		scale = dt
	}
	var freqs []float64
	for _, pole := range poles {
		w0, ok := boundaryPoleFreq(pole, continuous, dt, tol)
		if !ok {
			continue
		}
		merged := false
		for _, f := range freqs {
			if math.Abs(f-w0)*scale < tol*10*math.Max(1, w0*scale) {
				merged = true
				break
			}
		}
		if !merged {
			freqs = append(freqs, w0)
		}
	}
	sort.Float64s(freqs)
	return freqs
}

func autoNyquistFreqs(sys *System, poles []complex128, boundary []float64, nPoints int, tol float64) []float64 {
	if nPoints <= 0 {
		nPoints = 500
	}

	td := newTimeDomain(sys.Dt)
	natFreqs := make([]float64, 0, len(poles))
	for _, p := range poles {
		if w0, ok := boundaryPoleFreq(p, sys.IsContinuous(), sys.Dt, tol); ok && w0 == 0 {
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

	for _, w0 := range boundary {
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
		for _, w0 := range boundary {
			if w0 > 0 && math.Abs(w-w0) < 1e-4*w0 {
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

const (
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
// around lightly damped poles, where the response varies on the scale of the
// pole's distance to the boundary rather than of the base grid spacing.
func (p *nyquistPath) seedResonances(poles []complex128, continuous bool) {
	for _, pole := range poles {
		at, d := imag(pole), math.Abs(real(pole))
		if !continuous {
			at, d = cmplx.Phase(pole), math.Abs(1-cmplx.Abs(pole))
		}
		if at < 0 || d == 0 || d > 0.1*math.Max(at, 1e-300) {
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
	if depth >= nyquistMaxDepth || phaseStep(1+va, 1+vb) <= nyquistMaxPhaseStep {
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

// indentRadius sizes the detour around the boundary poles clustered within
// 10*tol of center so it encloses the cluster and no other open-loop pole.
func indentRadius(center complex128, base, tol float64, poles []complex128) float64 {
	r := max(base, 20*tol)
	cluster := 10 * tol
	for _, pole := range poles {
		if d := cmplx.Abs(pole - center); d > cluster {
			r = min(r, d/4)
		}
	}
	return r
}

func nyquistEncirclements(sys *System, poles []complex128, boundary []float64, tol float64) (int, error) {
	e, err := validFrequencyEvaluator(sys, "Nyquist")
	if err != nil {
		return 0, err
	}
	path := &nyquistPath{sys: sys, eval: e.pointEval(), buf: make([]complex128, 1)}
	path.seedResonances(poles, sys.IsContinuous())

	var closure complex128
	if sys.IsContinuous() {
		closure = complex(sys.D.At(0, 0), 0)
		err = path.continuousHalf(poles, boundary, closure, tol)
	} else {
		err = path.discreteHalf(poles, boundary, tol)
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

func (p *nyquistPath) continuousHalf(poles []complex128, boundary []float64, closure complex128, tol float64) error {
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
	for _, w0 := range boundary {
		r := indentRadius(complex(0, w0), 1e-4*math.Max(1, w0), tol, poles)
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

func (p *nyquistPath) discreteHalf(poles []complex128, boundary []float64, tol float64) error {
	dt := p.sys.Dt
	circle := func(theta float64) complex128 { return cmplx.Exp(complex(0, theta)) }
	segment := func(a, b float64) error {
		n := max(int(math.Ceil((b-a)/(math.Pi/200))), 2)
		return p.piece(circle, p.withSeeds(linspace(a, b, n), a, b, func(x float64) float64 { return x }))
	}

	cur := 0.0
	closed := false
	for _, w0 := range boundary {
		theta := min(w0*dt, math.Pi)
		center := cmplx.Exp(complex(0, theta))
		if theta == 0 {
			center = 1
		}
		r := indentRadius(center, 1e-4, tol, poles)
		switch {
		case theta == 0:
			if err := p.arc(1, r, 0, math.Pi/2, true); err != nil {
				return err
			}
			cur = r
		case theta >= math.Pi*(1-1e-12):
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
