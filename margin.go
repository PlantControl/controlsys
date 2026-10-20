package controlsys

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// MarginResult holds the classical margins MATLAB margin reports. The
// margins are always defined (+Inf when the loop never reaches the
// corresponding crossover); the crossover frequencies exist only when the
// loop crosses, so they are comma-ok methods.
type MarginResult struct {
	GainMargin  float64 // dB; +Inf if no phase crossover
	PhaseMargin float64 // degrees in (-180,180]; +Inf if no gain crossover

	gainCrossover, phaseCrossover float64
	hasGainCross, hasPhaseCross   bool
}

// GainCrossover returns the 0 dB crossover frequency where PhaseMargin is
// measured (MATLAB Wcp); ok is false when |L| never crosses 1.
func (r *MarginResult) GainCrossover() (w float64, ok bool) {
	return r.gainCrossover, r.hasGainCross
}

// PhaseCrossover returns the −180° crossover frequency where GainMargin is
// measured (MATLAB Wcg); ok is false when the phase never crosses −180°.
func (r *MarginResult) PhaseCrossover() (w float64, ok bool) {
	return r.phaseCrossover, r.hasPhaseCross
}

// AllMarginResult lists every crossover, as MATLAB allmargin; empty slices
// mean the loop has no crossover of that kind.
type AllMarginResult struct {
	GainMargins     []float64 // dB at each phase crossover
	PhaseMargins    []float64 // degrees in (-180,180] at each gain crossover
	GainCrossFreqs  []float64 // omega where |G|=0dB
	PhaseCrossFreqs []float64 // omega where angle(G)=-180deg mod 360
}

// DiskMarginResult holds a SISO disk margin, mirroring the fields of MATLAB
// diskmargin. Alpha, GainMargin, PhaseMargin and Frequency follow the skew
// Skew; PeakSensitivity and PeakFreq always describe ‖S‖∞. An unstable
// closed loop has no peak frequencies, so those are comma-ok methods.
type DiskMarginResult struct {
	Alpha           float64    // disk margin αmax = 1/‖S+(σ−1)/2‖∞
	Skew            float64    // σ of the gain-variation disk
	GainMargin      [2]float64 // [low, high] linear gain factors
	GainMarginDB    [2]float64 // [low, high] in dB
	PhaseMargin     float64    // +/- degrees
	PeakSensitivity float64    // Ms = ‖S‖∞; +Inf for an unstable closed loop

	frequency, peakFreq float64
	stable              bool
}

// Frequency returns the ω where the disk margin is attained; ok is false
// for an unstable closed loop.
func (r *DiskMarginResult) Frequency() (w float64, ok bool) { return r.frequency, r.stable }

// PeakFreq returns the ω where |S| peaks; ok is false for an unstable
// closed loop.
func (r *DiskMarginResult) PeakFreq() (w float64, ok bool) { return r.peakFreq, r.stable }

// sisoEval evaluates a SISO loop on the boundary of the stability region.
// It reuses one point evaluator, the state-space solver or the internal-delay
// LFT workspace, so refinement loops pay its setup once and high-order loops
// keep state-space accuracy. Unless extended or exact, each value the double
// solve resolves is bit-identical to FreqResponse at the same ω. Where it
// fails, at takes the pole limit in double (ddGate.limit) instead of
// FreqResponse's double-double response, and a failure there stands: the
// rational margin search and the PID screens read it as unresolved, as next
// to an origin or Nyquist cluster or multiple poles at z = 1. resolve
// returns the response as FreqResponse resolves it, for a caller that needs
// it rather than that unresolved state.
//
// at keeps the signature func(float64) complex128 so it can be passed as a
// response function; a failed evaluation returns NaN and is kept in err,
// which callers check once their search is done.
type sisoEval struct {
	sys   *System
	eval  func(frequencyPoint, []complex128) error
	delay *mat.Dense
	td    timeDomain
	dst   []complex128
	err   error
	// skipped is try's first failure and resolved whether at has succeeded
	// at some ω > 0; see checkSkipped.
	skipped  error
	resolved bool
	// extended falls back to ddSISOResponse where the double solve fails;
	// see rationalLoop.gainCrossings and the delayed grid in allMargin.
	// ddAllow, when set, limits the fallback to the frequencies it accepts.
	extended bool
	ddAllow  func(w float64) bool
	// exact evaluates in double-double first: beside a pair 1 ± jε the
	// double solve's phase is off by up to ~1e-5°, beyond the crossing
	// tolerance; see resonanceCrossings.
	exact bool
	// ddSolves counts double-double solves.
	ddSolves int
}

func newSISOEval(sys *System) (*sisoEval, error) {
	e := &sisoEval{
		sys: sys,
		td:  newTimeDomain(sys.Dt),
		dst: make([]complex128, 1),
	}
	fe := newFrequencyEvaluator(sys)
	if sys.internalDelayCount() > 0 {
		e.eval = fe.pointEval()
	} else {
		if err := sys.Validate(); err != nil {
			return nil, err
		}
		e.eval = fe.pointSolver(math.MaxInt).evalInto
	}
	e.delay = effectiveIODelayMatrix(sys, 1, 1, true)
	return e, nil
}

func (e *sisoEval) at(w float64) complex128 {
	pt := e.td.frequencyPoint(w)
	if h, ok := e.exactAt(pt); ok {
		e.dst[0] = h
	} else if err := evalWithPoleLimitGated(e.eval, e.sys, pt, e.dst, ddGate{limit: !e.extended, snap: true, allow: e.ddAllow, w: w}); err != nil {
		h, ok := complex(0, 0), false
		if e.extended && e.sys.internalDelayCount() == 0 && (e.ddAllow == nil || e.ddAllow(w)) {
			h, ok = ddSISOResponse(e.sys, pt)
			e.ddSolves++
		}
		if !ok {
			return e.fail(w, err)
		}
		e.dst[0] = h
	}
	if e.delay != nil {
		applyIODelayMatrixAtS(e.sys, pt, e.dst, 1, 1, e.delay)
	}
	e.resolved = e.resolved || w > 0
	return e.dst[0]
}

// resolve is L(jω) as FreqResponse resolves it, ok false where that
// fails too: try with the extended fallback, and where at snaps a computed
// pole near ω to +Inf, the double-double response unless that is singular or
// ddAtPole puts the pole at ω. Beside a cluster at z = 1 the snap reaches ω
// tens of times off the pole (S2RNHI). at keeps the snap and its failures
// for the searches tuned to them: the gain search rejects brackets closing
// on +Inf, so taking the response there adds crossings at |L| within
// coefficient rounding of 1 next to an integrator, which pidtune would take
// for its crossover. pidtune's back-off gate samples through resolve where
// at fails; it leaves skipped and resolved as they were, so checkSkipped
// of the screens sharing e does not change.
func (e *sisoEval) resolve(w float64) (h complex128, ok bool) {
	saved, skipped, resolved := e.extended, e.skipped, e.resolved
	e.extended = true
	h, ok = e.try(w)
	e.extended, e.skipped, e.resolved = saved, skipped, resolved
	pt := e.td.frequencyPoint(w)
	s := pt.value()
	if !ok || !cmplx.IsInf(h) || e.sys.internalDelayCount() > 0 || s == 0 || !e.sys.IsContinuous() && s == 1 {
		return h, ok
	}
	sp, err := poleSpacing(e.sys, s)
	if err != nil {
		return h, ok
	}
	g := []complex128{0}
	e.ddSolves++
	if !ddResponseInto(e.sys, pt, g) || ddAtPole(e.sys, s, sp, g) {
		return h, ok
	}
	if e.delay != nil {
		applyIODelayMatrixAtS(e.sys, pt, g, 1, 1, e.delay)
	}
	return g[0], true
}

func (e *sisoEval) exactAt(pt frequencyPoint) (complex128, bool) {
	if !e.exact || e.sys.internalDelayCount() > 0 {
		return 0, false
	}
	e.ddSolves++
	return ddSISOResponse(e.sys, pt)
}

// try evaluates L(jω) without recording a failure, for candidates that may
// sit on a pole the solve cannot resolve, such as a pencil root next to an
// integrator; ok is false where at would fail.
func (e *sisoEval) try(w float64) (h complex128, ok bool) {
	return tryEval(e, func() complex128 { return e.at(w) })
}

// tryDouble is try without the extended fallback, for a check tuned to
// where the double solve and its pole limit fail.
func (e *sisoEval) tryDouble(w float64) (h complex128, ok bool) {
	saved := e.extended
	e.extended = false
	h, ok = e.try(w)
	e.extended = saved
	return h, ok
}

// tryEval runs g, which evaluates through e, with try's failure handling.
func tryEval[T any](e *sisoEval, g func() T) (v T, ok bool) {
	saved := e.err
	e.err = nil
	v = g()
	ok = e.err == nil
	if !ok && e.skipped == nil {
		e.skipped = e.err
	}
	e.err = saved
	return v, ok
}

// checkSkipped returns the evaluation error when try skipped failures and
// at never succeeded at ω > 0 but fails at a generic ω too, as for a
// singular pencil, which no frequency resolves.
func (e *sisoEval) checkSkipped() error {
	if e.err == nil && e.skipped != nil && !e.resolved {
		w := 1.0
		if e.sys.IsDiscrete() && e.sys.Dt > 0 {
			w = 1 / e.sys.Dt
		}
		e.at(w)
	}
	return e.err
}

func (e *sisoEval) fail(w float64, err error) complex128 {
	if e.err == nil {
		e.err = fmt.Errorf("loop response at ω=%g: %w", w, err)
	}
	return complex(math.NaN(), math.NaN())
}

const (
	marginMaxRefineDepth   = 10
	marginMaxDelayPoints   = 20000
	marginMaxExtendDecades = 10
	// walkClearance is the relative distance either side of a crossing found
	// next to a numerically singular band at which f must still change sign.
	walkClearance = 1e-3
	// realPoleDDBelow is the |phase offset| in degrees below which the scan
	// beside real poles next to z = 1 re-evaluates in double-double; the
	// double solve's phase error there is at most ~1e-7° in the WSF2HP
	// tests and ~1e-15° for high-order loops with many slow real poles.
	realPoleDDBelow = 1e-2
	// clusterRadius is the distance from z = 1 within which complex poles
	// off the resonance band join the scan beside real poles there; the
	// pairs of rings of five to eight poles around z = 1 whose crossings the
	// pencil lost lie up to 1.6e-2 from it (S2RNHI).
	clusterRadius = 5e-2
	// strayRootDeg is the angle of L off the real axis, in degrees, past
	// which a pencil root of L − L~ is off the root of L it stands for.
	strayRootDeg = 1e-2
	// nyquistScanDDAbove is the distance of L from the negative real axis,
	// |L|·min(|f|, 1) with f the phase offset in radians, over the coefficient
	// noise above which the scan towards Nyquist keeps the double solve. In the
	// 6OPXWW tests and the high-order loops the double solve's error in L
	// stays below 0.22 times that noise.
	nyquistScanDDAbove = 1e4
)

func marginLoopDelay(sys *System) float64 {
	tau := ioDelayTotal(sys, 0, 0)
	if sys.LFT != nil {
		for _, t := range sys.LFT.Tau {
			tau += t
		}
	}
	if sys.IsDiscrete() && sys.Dt > 0 {
		tau *= sys.Dt
	}
	return tau
}

// extendMarginRange moves w by factor per decade while |L| approaches 1
// without crossing it, so crossovers set by gain rather than poles are kept.
func extendMarginRange(w, factor float64, logMag func(float64) float64) float64 {
	for range marginMaxExtendDecades {
		m0, m1 := logMag(w), logMag(w*factor)
		if math.IsNaN(m0) || math.IsNaN(m1) || math.IsInf(m0, 0) || math.IsInf(m1, 0) {
			break
		}
		if m0*m1 <= 0 {
			return w * factor
		}
		if math.Abs(m1) >= math.Abs(m0)-1e-3 {
			break
		}
		w *= factor
	}
	return w
}

func marginGrid(sys *System, eval *sisoEval) ([]float64, []complex128, error) {
	poles, err := sys.Poles()
	if err != nil {
		return nil, nil, err
	}
	wMin, wMax := autoFreqRange(sys, poles, nil)
	discrete := sys.IsDiscrete() && sys.Dt > 0
	tau := marginLoopDelay(sys)
	if tau > 0 {
		wMin = min(wMin, 0.1/tau)
		if !discrete {
			wMax = max(wMax, 10/tau)
		}
	}
	logMag := func(w float64) float64 { return math.Log(cmplx.Abs(eval.at(w))) }
	wMin = extendMarginRange(wMin, 0.1, logMag)
	if !discrete {
		wMax = extendMarginRange(wMax, 10, logMag)
	}

	n := max(1000, int(100*math.Log10(wMax/wMin))+1)
	base := logspace(math.Log10(wMin), math.Log10(wMax), n)
	if tau > 0 {
		step := math.Pi / (8 * tau)
		nl := min(int((wMax-wMin)/step), marginMaxDelayPoints)
		for k := 1; k <= nl; k++ {
			base = append(base, wMin+float64(k)*step)
		}
		slices.Sort(base)
		base = slices.Compact(base)
	}
	for len(base) > 0 && base[len(base)-1] >= wMax {
		base = base[:len(base)-1]
	}
	base = append(base, wMax)

	omega := make([]float64, 0, len(base))
	resp := make([]complex128, 0, len(base))
	var split func(w0, w1 float64, h0, h1 complex128, depth int)
	split = func(w0, w1 float64, h0, h1 complex128, depth int) {
		if depth >= marginMaxRefineDepth || !marginNeedsSplit(h0, h1) {
			return
		}
		wm := math.Sqrt(w0 * w1)
		hm := eval.at(wm)
		split(w0, wm, h0, hm, depth+1)
		omega = append(omega, wm)
		resp = append(resp, hm)
		split(wm, w1, hm, h1, depth+1)
	}
	prev := eval.at(base[0])
	omega = append(omega, base[0])
	resp = append(resp, prev)
	for _, w := range base[1:] {
		h := eval.at(w)
		split(omega[len(omega)-1], w, prev, h, 0)
		omega = append(omega, w)
		resp = append(resp, h)
		prev = h
	}
	return omega, resp, nil
}

func marginNeedsSplit(h0, h1 complex128) bool {
	a0, a1 := cmplx.Abs(h0), cmplx.Abs(h1)
	if a0 == 0 || a1 == 0 || math.IsNaN(a0+a1) || math.IsInf(a0+a1, 0) {
		return false
	}
	dPhase := wrapDegrees((cmplx.Phase(h1) - cmplx.Phase(h0)) * 180 / math.Pi)
	return math.Abs(dPhase) > 30 || math.Abs(20*math.Log10(a1/a0)) > 6
}

// wrapDegrees wraps x to (-180,180].
func wrapDegrees(x float64) float64 {
	return x - 360*math.Ceil((x-180)/360)
}

// phaseOffsetDeg is the phase of h from −180°, in (−180, 180]. It is taken
// as arg(−h): adding 180 to arg h rounds the offset to ulp(360) ≈ 6e-14°,
// and where the phase is that flat, beside a pair 1 ± jε, the crossing
// moves by 1e-6 relative.
func phaseOffsetDeg(h complex128) float64 {
	return wrapDegrees(cmplx.Phase(-h) * 180 / math.Pi)
}

// crossing is a level crossing on segment idx of a sampled curve, at
// fraction frac of the segment and interpolated frequency w.
type crossing struct {
	idx  int
	frac float64
	w    float64
}

// findCrossings finds where vals passes level between samples. A sample
// exactly at level counts once: on the segment it ends, or on the first.
func findCrossings(omega, vals []float64, level float64) []crossing {
	var result []crossing
	for k := 0; k < len(vals)-1; k++ {
		a := vals[k] - level
		b := vals[k+1] - level
		if math.IsInf(a, 0) || math.IsInf(b, 0) || !signCrosses(a, b, k) {
			continue
		}
		result = append(result, interpCrossing(omega, k, a, b))
	}
	return result
}

// phaseCrossings finds where phase passes target (mod 360). A sign change of
// the wrapped offset with a jump near 360 is the opposite wrap, not a crossing.
func phaseCrossings(omega, phase []float64, target float64) []crossing {
	var result []crossing
	for k := 0; k < len(phase)-1; k++ {
		a := wrapDegrees(phase[k] - target)
		b := wrapDegrees(phase[k+1] - target)
		if signCrosses(a, b, k) && math.Abs(a-b) < 180 {
			result = append(result, interpCrossing(omega, k, a, b))
		}
	}
	return result
}

func signCrosses(a, b float64, k int) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}
	return a*b < 0 || (b == 0 && a != 0) || (a == 0 && k == 0)
}

// interpCrossing interpolates linearly in log omega, or in omega when the
// segment starts at 0.
func interpCrossing(omega []float64, k int, a, b float64) crossing {
	frac := 0.0
	if a != b {
		frac = a / (a - b)
	}
	w0, w1 := omega[k], omega[k+1]
	w := w0 + frac*(w1-w0)
	if w0 > 0 {
		w = math.Exp(math.Log(w0) + frac*(math.Log(w1)-math.Log(w0)))
	}
	return crossing{idx: k, frac: frac, w: w}
}

func refineCrossing(wLo, wHi float64, evalFn func(float64) float64) float64 {
	fLo := evalFn(wLo)
	for range 64 {
		wMid := math.Sqrt(wLo * wHi)
		if wMid <= wLo || wMid >= wHi {
			break
		}
		fMid := evalFn(wMid)
		if fLo*fMid <= 0 {
			wHi = wMid
		} else {
			wLo = wMid
			fLo = fMid
		}
	}
	return math.Sqrt(wLo * wHi)
}

// ulpTol widens tol for a bisected sign change of f at c to what one
// rounding step of c moves f by, measured as four times the mean step over
// ±64 ulps, capped at limit. Beside a resonance with ζ ≲ 1e-7, f moves by more
// than tol per ulp, so no float c meets tol: about 8e-14/ζ degrees. A jump J
// of f, as across the ±180 wrap or an undamped pole, gives J/32 whatever the
// limit; a limit of 1e-3 rejects it, one above J/32 relies on |f| at the jump.
func ulpTol(c float64, f func(float64) float64, tol, limit float64) float64 {
	u := math.Nextafter(c, math.Inf(1)) - c
	step := math.Abs(f(c+64*u)-f(c-64*u)) / 128
	if !(step < math.Inf(1)) {
		return tol
	}
	return max(tol, min(4*step, limit))
}

func phaseMarginDeg(h complex128) float64 {
	pm := 180 + cmplx.Phase(h)*180/math.Pi
	if pm > 180 {
		pm -= 360
	}
	return pm
}

// AllMargin computes every gain and phase crossover of the SISO loop sys, as
// MATLAB allmargin (https://www.mathworks.com/help/control/ref/dynamicsystem.allmargin.html).
// For delay-free loops, and for discrete loops whose delays are powers of z,
// the crossovers are boundary eigenvalues of pencils built from the
// realization (the Hamiltonian pencil of |L| = 1, as in the H∞ norm, and the
// system pencil of L − L~), refined on the exact state-space response, so no
// frequency grid can miss them and high-order loops lose none to
// transfer-function round-off. Descriptor loops are supported.
// Continuous delays keep |L(jω)| rational, so gain crossovers stay exact;
// phase crossovers of continuous delayed loops, and both kinds for continuous
// internal delays, come from a delay-aware adaptive frequency search. A loop
// whose gain stays at 1 or whose phase stays at −180° has no isolated
// crossings and reports none; ω = 0 is never reported.
func AllMargin(sys *System) (*AllMarginResult, error) {
	if _, err := newSISOLoopModel(sys, "AllMargin"); err != nil {
		return nil, err
	}
	res, err := allMargin(sys)
	if err != nil {
		return nil, fmt.Errorf("AllMargin: %w", err)
	}
	return res, nil
}

func allMargin(sys *System) (*AllMarginResult, error) {
	loop, err := newRationalLoop(sys)
	if errors.Is(err, ErrContinuousInternalDelay) {
		loop = nil
	} else if err != nil {
		return nil, err
	}
	res := &AllMarginResult{}
	var eval *sisoEval
	if loop != nil {
		eval = loop.eval
		if res.GainCrossFreqs, err = loop.gainCrossings(1); err != nil {
			return nil, err
		}
		if !loop.delayed {
			if res.PhaseCrossFreqs, err = loop.phaseCrossings(); err != nil {
				return nil, err
			}
		}
	} else if eval, err = newSISOEval(sys); err != nil {
		return nil, err
	}
	if loop == nil || loop.delayed {
		// Next to an origin pole cluster the double solve fails off every
		// pole; the grid walk needs the double-double response there.
		eval.extended = true
		gain, phase, err := gridMarginCrossings(sys, eval, loop == nil)
		eval.extended = false
		if err != nil {
			return nil, err
		}
		if loop == nil {
			res.GainCrossFreqs = gain
		}
		res.PhaseCrossFreqs = phase
	}
	// Beside a pair 1 ± jε the double solve's |L| is off by up to ~1e-5.
	eval.extended, eval.exact = loop != nil, loop != nil
	for _, w := range res.GainCrossFreqs {
		res.PhaseMargins = append(res.PhaseMargins, phaseMarginDeg(eval.at(w)))
	}
	for _, w := range res.PhaseCrossFreqs {
		h := eval.at(w)
		if loop != nil && loop.discrete() && w == loop.nyquist() {
			if hn, _, _, ok := loop.nyquistResponse(); ok {
				h = hn
			}
		}
		res.GainMargins = append(res.GainMargins, -20*math.Log10(cmplx.Abs(h)))
	}
	eval.extended, eval.exact = false, false
	if err := eval.checkSkipped(); err != nil {
		return nil, err
	}
	return res, nil
}

// gridMarginCrossings searches an adaptive grid of a continuous delayed loop
// for phase crossovers, and for gain crossovers when withGain is set.
func gridMarginCrossings(sys *System, eval *sisoEval, withGain bool) (gain, phase []float64, err error) {
	omega, resp, err := marginGrid(sys, eval)
	if err != nil {
		return nil, nil, err
	}
	magDB := make([]float64, len(omega))
	phaseDeg := make([]float64, len(omega))
	for k, h := range resp {
		magDB[k] = 20 * math.Log10(cmplx.Abs(h))
		phaseDeg[k] = finitePhaseDeg(h)
	}
	if withGain {
		for _, c := range findCrossings(omega, magDB, 0.0) {
			w := refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
				return 20 * math.Log10(cmplx.Abs(eval.at(w)))
			})
			if math.Abs(20*math.Log10(cmplx.Abs(eval.at(w)))) <= 1e-2 {
				gain = append(gain, w)
			}
		}
	}
	for _, c := range phaseCrossings(omega, phaseDeg, -180.0) {
		w := refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
			return wrapDegrees(finitePhaseDeg(-eval.at(w)))
		})
		if math.Abs(phaseOffsetDeg(eval.at(w))) <= 1 {
			phase = append(phase, w)
		}
	}
	return gain, phase, nil
}

// finitePhaseDeg is arg h in degrees, or NaN where h is infinite: the pole
// limit's sample beside an origin cluster has no phase, and its arg ±45° or
// ±135° paired with a neighbour within 180° of it reads as a crossing (G7VBK5).
func finitePhaseDeg(h complex128) float64 {
	if cmplx.IsInf(h) {
		return math.NaN()
	}
	return cmplx.Phase(h) * 180 / math.Pi
}

// Margin returns the gain margin closest to 0 dB and the phase margin
// closest to 0° of the SISO loop sys with their crossover frequencies, like
// MATLAB [Gm,Pm,Wcg,Wcp] = margin(sys) with Gm in dB; see AllMargin for the
// crossover search. Where MATLAB reports NaN for a missing crossover
// frequency, GainCrossover and PhaseCrossover report ok = false.
func Margin(sys *System) (*MarginResult, error) {
	if _, err := newSISOLoopModel(sys, "Margin"); err != nil {
		return nil, err
	}
	all, err := allMargin(sys)
	if err != nil {
		return nil, fmt.Errorf("Margin: %w", err)
	}
	return pickMargins(all), nil
}

// pickMargins selects, as MATLAB margin
// (https://www.mathworks.com/help/control/ref/dynamicsystem.margin.html),
// the gain margin closest to 0 dB and the phase margin closest to 0°, each
// with its frequency; the lower frequency wins a tie.
func pickMargins(all *AllMarginResult) *MarginResult {
	result := &MarginResult{}
	result.GainMargin, result.phaseCrossover, result.hasPhaseCross = selectMargin(all.GainMargins, all.PhaseCrossFreqs)
	result.PhaseMargin, result.gainCrossover, result.hasGainCross = selectMargin(all.PhaseMargins, all.GainCrossFreqs)
	return result
}

func selectMargin(margins, freqs []float64) (m, w float64, ok bool) {
	m = math.Inf(1)
	for i, v := range margins {
		if !ok || math.Abs(v) < math.Abs(m) {
			m, w, ok = v, freqs[i], true
		}
	}
	return m, w, ok
}

// Bandwidth returns the first frequency where the gain of the SISO model sys
// drops dbDrop dB below its DC value, as MATLAB bandwidth(sys,dbdrop)
// (https://www.mathworks.com/help/control/ref/dynamicsystem.bandwidth.html);
// dbDrop = 0 selects MATLAB's default −3. Without continuous internal delays
// it is the smallest boundary eigenvalue of the Hamiltonian pencil of
// |L| = g, refined on the exact response; with them, the first crossing on
// an adaptive frequency grid. It returns +Inf when the gain never drops that
// far, as for a DC gain of 0 (MATLAB's high-pass result). An infinite DC
// gain (an integrator) puts the threshold at ∞, so the bandwidth is
// undefined rather than +Inf; python-control returns NaN, which the Go error
// rule maps to ErrInvalidArgument. A dbDrop that is not a finite negative
// scalar also returns ErrInvalidArgument; a model without inputs or outputs returns
// ErrDimensionMismatch and a MIMO model ErrNotSISO, as MATLAB rejects it.
func Bandwidth(sys *System, dbDrop float64) (float64, error) {
	if dbDrop == 0 {
		dbDrop = -3
	}
	if !(dbDrop < 0) || math.IsInf(dbDrop, -1) {
		return 0, fmt.Errorf("Bandwidth: dbDrop must be a finite negative scalar, got %g: %w", dbDrop, ErrInvalidArgument)
	}
	if err := requireSystem("Bandwidth", sys); err != nil {
		return 0, err
	}
	if _, m, p := sys.Dims(); m == 0 || p == 0 {
		return 0, fmt.Errorf("Bandwidth: model has no inputs or no outputs: %w", ErrDimensionMismatch)
	}
	if _, err := newSISOLoopModel(sys, "Bandwidth"); err != nil {
		return 0, err
	}
	bw, err := bandwidth(sys, dbDrop)
	if err != nil {
		return 0, fmt.Errorf("Bandwidth: %w", err)
	}
	return bw, nil
}

func bandwidth(sys *System, dbDrop float64) (float64, error) {
	dcGain, err := sys.DCGain()
	if err != nil {
		return 0, err
	}
	dcMag := math.Abs(dcGain.At(0, 0))
	if dcMag == 0 {
		return math.Inf(1), nil
	}
	if math.IsNaN(dcMag) || math.IsInf(dcMag, 0) {
		return 0, fmt.Errorf("DC gain is %g, bandwidth undefined: %w", dcGain.At(0, 0), ErrInvalidArgument)
	}
	threshold := 20*math.Log10(dcMag) + dbDrop

	loop, err := newRationalLoop(sys)
	if err != nil && !errors.Is(err, ErrContinuousInternalDelay) {
		return 0, err
	}
	if err == nil {
		ws, err := loop.gainCrossings(math.Pow(10, threshold/20))
		if err != nil {
			return 0, err
		}
		if loop.eval.err != nil {
			return 0, loop.eval.err
		}
		if len(ws) == 0 {
			return math.Inf(1), nil
		}
		return ws[0], nil
	}

	omega, err := sys.DefaultFrequencyGrid(1000)
	if err != nil {
		return 0, err
	}
	if len(omega) == 0 {
		return math.Inf(1), nil
	}
	eval, err := newSISOEval(sys)
	if err != nil {
		return 0, err
	}
	gainDB := func(w float64) float64 {
		if g := cmplx.Abs(eval.at(w)); g > 0 {
			return 20 * math.Log10(g)
		}
		return -1000
	}
	for range 12 {
		if gainDB(omega[0]) >= threshold {
			break
		}
		omega = append([]float64{omega[0] / 10}, omega...)
	}
	magDB := make([]float64, len(omega))
	for k, w := range omega {
		magDB[k] = gainDB(w)
	}
	if eval.err != nil {
		return 0, eval.err
	}
	if magDB[0] < threshold {
		return 0, fmt.Errorf("gain stays %g dB below DC down to ω=%g: %w", magDB[0]-threshold-dbDrop, omega[0], ErrInvalidArgument)
	}
	crossings := findCrossings(omega, magDB, threshold)
	if len(crossings) == 0 {
		return math.Inf(1), nil
	}
	c := crossings[0]
	w := refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
		return gainDB(w) - threshold
	})
	if eval.err != nil {
		return 0, eval.err
	}
	return w, nil
}

// DiskMargin computes the balanced (skew σ = 0) disk margin of the SISO loop
// sys, the default of MATLAB diskmargin
// (https://www.mathworks.com/help/robust/ref/dynamicsystem.diskmargin.html).
// It is DiskMarginSkew(sys, 0): αmax = 1/‖S − 1/2‖∞, GainMargin
// [(2−α)/(2+α), (2+α)/(2−α)] and PhaseMargin 2·atan(α/2).
func DiskMargin(sys *System) (*DiskMarginResult, error) {
	if _, err := newSISOLoopModel(sys, "DiskMargin"); err != nil {
		return nil, err
	}
	res, err := diskMargin(sys, 0)
	if err != nil {
		return nil, fmt.Errorf("DiskMargin: %w", err)
	}
	return res, nil
}

// DiskMarginSkew computes the disk margin of the SISO loop sys for the skew
// sigma, as MATLAB diskmargin(L, sigma). The loop tolerates every
// multiplicative gain and phase variation F = (1 + α(1−σ)/2·δ)/(1 − α(1+σ)/2·δ)
// with |δ| < 1 and α < αmax = 1/‖S + (σ−1)/2‖∞, S = 1/(1+L). σ = 0 balances
// gain increase and decrease, σ > 0 favours increase, and σ = 1 gives the
// sensitivity-based margin αmax = 1/‖S‖∞. GainMargin and PhaseMargin are the
// real-axis intercepts and the unit-circle extent of the F disk, as MATLAB
// dm2gm; a negative lower intercept is reported as 0. An unstable closed
// loop gives Alpha 0, GainMargin [1 1] and PhaseMargin 0, as MATLAB
// diskmargin does.
//
// Discrete loop delays are absorbed exactly. For continuous loops with delays
// closed-loop stability comes from a Nyquist encirclement count of 1+L on a
// delay-aware adaptive frequency grid, and the peaks from the exact frequency
// response, certified between grid samples to 1e-9 relative; see
// delayLoop.nyquist for the resolution limits. Loops whose internal delays
// form a feedback cycle return ErrContinuousInternalDelay; loops whose
// high-frequency gain may reach 1 (neutral type), whose delay grid or peak
// certification would exceed 2^21 points, or whose encirclement count does
// not resolve return ErrDelayUnsupported. Descriptor loops return
// ErrDescriptorUnsupported, as HinfNorm does.
func DiskMarginSkew(sys *System, sigma float64) (*DiskMarginResult, error) {
	if err := requireFinite("DiskMarginSkew", "skew", sigma); err != nil {
		return nil, err
	}
	if _, err := newSISOLoopModel(sys, "DiskMarginSkew"); err != nil {
		return nil, err
	}
	res, err := diskMargin(sys, sigma)
	if err != nil {
		return nil, fmt.Errorf("DiskMarginSkew: %w", err)
	}
	return res, nil
}

func diskMargin(sys *System, sigma float64) (*DiskMarginResult, error) {
	shift := (sigma - 1) / 2
	if sys.IsContinuous() && sys.HasDelay() {
		return diskMarginDelayed(sys, sigma, shift)
	}

	eye, err := NewGain(mat.NewDense(1, 1, []float64{1}), sys.Dt)
	if err != nil {
		return nil, err
	}
	S, err := Feedback(eye, sys, -1)
	if err != nil {
		return nil, fmt.Errorf("cannot form sensitivity: %w", err)
	}
	Ms, wMs, err := HinfNorm(S)
	if err != nil {
		return nil, err
	}
	if math.IsInf(Ms, 1) {
		return unstableDiskMargin(sigma), nil
	}
	peak, wPeak := Ms, wMs
	if shift != 0 {
		c, err := NewGain(mat.NewDense(1, 1, []float64{shift}), sys.Dt)
		if err != nil {
			return nil, err
		}
		Sc, err := Parallel(S, c)
		if err != nil {
			return nil, fmt.Errorf("cannot form shifted sensitivity: %w", err)
		}
		if peak, wPeak, err = HinfNorm(Sc); err != nil {
			return nil, err
		}
	}
	return diskMarginFromPeak(sigma, peak, wPeak, Ms, wMs), nil
}

func unstableDiskMargin(sigma float64) *DiskMarginResult {
	return &DiskMarginResult{
		Alpha:           0,
		Skew:            sigma,
		GainMargin:      [2]float64{1, 1},
		GainMarginDB:    [2]float64{0, 0},
		PhaseMargin:     0,
		PeakSensitivity: math.Inf(1),
	}
}

func diskMarginDelayed(sys *System, sigma, shift float64) (*DiskMarginResult, error) {
	loop, err := delayLoopFromSystem(sys, "delay loop")
	if err != nil {
		return nil, err
	}
	stable, peaks, err := loop.sensitivityPeaks(0, shift)
	if err != nil {
		return nil, err
	}
	if !stable {
		return unstableDiskMargin(sigma), nil
	}
	return diskMarginFromPeak(sigma, peaks[1].peak, peaks[1].w, peaks[0].peak, peaks[0].w), nil
}

func diskMarginFromPeak(sigma, peak, wPeak, Ms, wMs float64) *DiskMarginResult {
	alpha := 1 / peak
	gm, pm := diskGainPhaseMargin(alpha, sigma)
	return &DiskMarginResult{
		Alpha:           alpha,
		Skew:            sigma,
		GainMargin:      gm,
		GainMarginDB:    [2]float64{20 * math.Log10(gm[0]), 20 * math.Log10(gm[1])},
		PhaseMargin:     pm,
		PeakSensitivity: Ms,
		frequency:       wPeak,
		peakFreq:        wMs,
		stable:          true,
	}
}

// diskGainPhaseMargin returns the gain interval and phase arc around 1
// covered by the image of the unit disk under
// F(δ) = (1 + aδ)/(1 − bδ), a = α(1−σ)/2, b = α(1+σ)/2, as MATLAB dm2gm.
// F is increasing on real δ, so the interval runs from F(−1) to F(1) unless
// the pole δ = 1/b lies in [−1, 1]. e^{jθ} is in the image when
// |e^{jθ}−1| ≤ |a + b·e^{jθ}|, i.e. 2(1+ab)·cosθ ≥ 2 − a² − b².
func diskGainPhaseMargin(alpha, sigma float64) (gm [2]float64, pm float64) {
	if math.IsInf(alpha, 1) {
		return [2]float64{0, math.Inf(1)}, 180
	}
	a, b := alpha*(1-sigma)/2, alpha*(1+sigma)/2
	gm = [2]float64{0, math.Inf(1)}
	if b > -1 {
		gm[0] = max((1-a)/(1+b), 0)
	}
	if b < 1 {
		gm[1] = (1 + a) / (1 - b)
	}
	pm = 180
	if 1+a*b > 0 {
		pm = math.Acos(max(-1, min(1, (2-a*a-b*b)/(2*(1+a*b))))) * 180 / math.Pi
	}
	return gm, pm
}

// rationalLoop is the finite-dimensional part of a SISO loop, whose
// unit-gain and real-axis crossings are boundary eigenvalues of pencils built
// from its realization, as for MATLAB allmargin on LTI models. Discrete
// delays are absorbed as states. A continuous loop delay leaves |L(jω)|
// rational but not angle(L), so delayed reports that only gain crossings are
// exact.
type rationalLoop struct {
	n          int
	a, b, c, e []float64
	d          float64
	dt         float64
	delayed    bool
	eval       *sisoEval
	// absorbed is the discrete loop with its delays absorbed as states, for
	// L at z = −1, where e^{−jkπ} = (−1)^k sets the sign.
	absorbed *System
}

// newRationalLoop fails with ErrContinuousInternalDelay when sys has
// continuous internal delays, for which no finite-dimensional form exists.
func newRationalLoop(sys *System) (*rationalLoop, error) {
	fsys, err := finiteDimensionalModel(sys, "margin")
	if err != nil {
		return nil, err
	}
	r := &rationalLoop{dt: fsys.Dt}
	if r.eval, err = newSISOEval(fsys); err != nil {
		return nil, err
	}
	fd := fsys
	if fsys.IsDiscrete() {
		if fd, err = fsys.AbsorbDelay(); err != nil {
			return nil, err
		}
		r.absorbed = fd
	} else {
		r.delayed = ioDelayTotal(fsys, 0, 0) != 0
	}
	r.n, _, _ = fd.Dims()
	br := newBalancedRealization(fd, r.n, 1, 1)
	r.a, r.b, r.c, r.e = br.a, br.b, br.c, br.e
	if r.e == nil {
		r.e = make([]float64, r.n*r.n)
		for i := range r.n {
			r.e[i*r.n+i] = 1
		}
	}
	r.d = fd.D.At(0, 0)
	return r, nil
}

func (r *rationalLoop) discrete() bool { return r.dt > 0 }

func (r *rationalLoop) nyquist() float64 { return math.Pi / r.dt }

// nyquistResponse is L at z = −1 exactly, in double-double, with the noise
// bounds of ddSISOResponseNoise. frequencyPoint(π/Dt) lies at z = −1 +
// 1.2e-16j, where L′(−1)·1.2e-16j outweighs an L(−1) ≲ 1e-16·|L′(−1)|, as
// beside a (z+1) cluster, and sets both phase and |L|.
func (r *rationalLoop) nyquistResponse() (h complex128, solve, coeffs float64, ok bool) {
	r.eval.ddSolves++
	return ddSISOResponseNoise(r.absorbed, pointAt(-1))
}

// gainCandidates returns approximate ω > 0 where |L| = gamma: boundary
// eigenvalues of the system pencil of L~·L/γ² − 1, with L~(s) = L(−s)ᵀ, or
// L(1/z)ᵀ when discrete. Over [x; p; u] it is, continuous,
//
//	s·diag(E, Eᵀ, 0) − [A 0 B; −cᵀc −Aᵀ −cᵀd; d·c Bᵀ d²−1]
//
// and discrete
//
//	z·[E 0 0; 0 Aᵀ 0; 0 −Bᵀ 0] − [A 0 B; −cᵀc Eᵀ −cᵀd; d·c 0 d²−1]
//
// with c = C/γ and d = D/γ. For γ = 1 the continuous pencil is the
// Hamiltonian pencil of the H∞ norm test.
func (r *rationalLoop) gainCandidates(gamma float64) ([]float64, error) {
	n := r.n
	N := 2*n + 1
	m, k := make([]float64, N*N), make([]float64, N*N)
	u := 2 * n
	d := r.d / gamma
	bs, cs := ioScale(r.b, r.c, gamma)
	for i := range n {
		ci := r.c[i] * cs
		for j := range n {
			m[i*N+j] = r.a[i*n+j]
			m[(n+i)*N+j] = -ci * r.c[j] * cs
			k[i*N+j] = r.e[i*n+j]
			if r.discrete() {
				m[(n+i)*N+n+j] = r.e[j*n+i]
				k[(n+i)*N+n+j] = r.a[j*n+i]
			} else {
				m[(n+i)*N+n+j] = -r.a[j*n+i]
				k[(n+i)*N+n+j] = r.e[j*n+i]
			}
		}
		bi := r.b[i] * bs
		m[i*N+u] = bi
		m[(n+i)*N+u] = -ci * d
		m[u*N+i] = d * ci
		if r.discrete() {
			k[u*N+n+i] = -bi
		} else {
			m[u*N+n+i] = bi
		}
	}
	m[u*N+u] = d*d - 1
	return r.boundaryRoots(m, k, N)
}

// ioScale splits 1/gamma between B and C as bs·cs = 1/gamma, powers of two
// up to the split, that equalise ‖B·bs‖ and ‖C·cs‖. A realization with
// B ≪ C/γ, as for small DC gain at the bandwidth level, otherwise puts
// C·Cᵀ/γ² far above A and B, and QZ loses the boundary roots.
func ioScale(b, c []float64, gamma float64) (bs, cs float64) {
	nb, nc := frobenius(b), frobenius(c)
	if nb == 0 || nc == 0 {
		return 1, 1 / gamma
	}
	_, eb := math.Frexp(math.Sqrt(nc / (gamma * nb)))
	bs = math.Ldexp(1, eb)
	return bs, 1 / (gamma * bs)
}

// realCandidates returns approximate ω > 0 where L is real: boundary
// eigenvalues of the system pencil of L − L~. Over [x₁; x₂; u] it is,
// continuous,
//
//	s·diag(E, E, 0) − [A 0 B; 0 −A B; C C 0]
//
// and discrete
//
//	z·[E 0 0; 0 A 0; 0 C 0] − [A 0 B; 0 E −B; C 0 0].
func (r *rationalLoop) realCandidates() ([]float64, error) {
	n := r.n
	N := 2*n + 1
	m, k := make([]float64, N*N), make([]float64, N*N)
	u := 2 * n
	for i := range n {
		for j := range n {
			m[i*N+j] = r.a[i*n+j]
			k[i*N+j] = r.e[i*n+j]
			if r.discrete() {
				m[(n+i)*N+n+j] = r.e[i*n+j]
				k[(n+i)*N+n+j] = r.a[i*n+j]
			} else {
				m[(n+i)*N+n+j] = -r.a[i*n+j]
				k[(n+i)*N+n+j] = r.e[i*n+j]
			}
		}
		m[i*N+u] = r.b[i]
		m[u*N+i] = r.c[i]
		if r.discrete() {
			m[(n+i)*N+u] = -r.b[i]
			k[u*N+n+i] = r.c[i]
		} else {
			m[(n+i)*N+u] = r.b[i]
			m[u*N+n+i] = r.c[i]
		}
	}
	return r.boundaryRoots(m, k, N)
}

// marginBoundaryBand is the relative distance from the imaginary axis, or
// from the unit circle, within which a pencil eigenvalue is a candidate. QZ
// does not preserve the pencils' symmetry, so boundary eigenvalues leave the
// boundary by their rounding error; polish verifies every candidate.
const marginBoundaryBand = 1e-2

// boundaryRoots maps the finite eigenvalues of the N×N pencil (m, k) near
// the imaginary axis (continuous) or the unit circle (discrete) to
// frequencies in (0, ∞) or (0, π/Dt], merging those that agree to rounding.
func (r *rationalLoop) boundaryRoots(m, k []float64, N int) ([]float64, error) {
	var ref float64
	for i := range r.a {
		ref = max(ref, math.Abs(r.a[i]), math.Abs(r.e[i]))
	}
	scaleInputRowCol(m, k, N, ref)
	kNorm := frobenius(k)
	alphar, alphai, beta := make([]float64, N), make([]float64, N), make([]float64, N)
	work := make([]float64, 8*N)
	if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, N, m, N, k, N, alphar, alphai, beta, nil, 1, nil, 1, work, len(work)) {
		return nil, fmt.Errorf("QZ: %w", ErrSchurFailed)
	}
	tol := 100 * float64(N) * eps()
	var ws []float64
	for j := range N {
		if math.Abs(beta[j]) <= tol*kNorm {
			continue
		}
		z := complex(alphar[j]/beta[j], alphai[j]/beta[j])
		var w float64
		if r.discrete() {
			if math.Abs(cmplx.Abs(z)-1) > marginBoundaryBand {
				continue
			}
			w = math.Abs(cmplx.Phase(z)) / r.dt
			if math.Abs(imag(z)) <= tol {
				w = 0
				if real(z) < 0 {
					w = r.nyquist()
				}
			}
		} else {
			if math.Abs(real(z)) > marginBoundaryBand*cmplx.Abs(z) {
				continue
			}
			w = math.Abs(imag(z))
		}
		if w > 0 && !math.IsInf(w, 0) {
			ws = append(ws, w)
		}
	}
	slices.Sort(ws)
	return slices.CompactFunc(ws, func(a, b float64) bool { return math.Abs(b-a) <= 1e-9*max(a, b) }), nil
}

// scaleInputRowCol scales the last column, then the last row, of the pencil
// (m, k) by a power of two up to ref, the largest magnitude of A and E,
// when it lies below √eps of it; equivalence leaves the eigenvalues as they
// are. B or C of a loop with tiny gain, as of k/(z+1)⁴ with |k| ≈ 1e-14,
// sit below QZ's backward error relative to the state block, which then
// loses boundary roots far from the cluster. Moderate imbalance is left: it
// does not hide roots, and scaling it moves roots beside a pole pair 1 ± jε.
// ref excludes the C·Cᵀ/γ² block of the gain pencil: for small γ it dwarfs
// A, and scaling up to it swamps the pencil in its last column.
func scaleInputRowCol(m, k []float64, N int, ref float64) {
	u := N - 1
	scale := func(at func(i int) int) {
		var mx float64
		for i := range N {
			mx = max(mx, math.Abs(m[at(i)]), math.Abs(k[at(i)]))
		}
		if mx == 0 || mx > math.Sqrt(eps())*ref {
			return
		}
		s := math.Ldexp(1, int(math.Round(math.Log2(ref/mx))))
		for i := range N {
			m[at(i)] *= s
			k[at(i)] *= s
		}
	}
	scale(func(i int) int { return i*N + u })
	scale(func(j int) int { return u*N + j })
}

func frobenius(a []float64) float64 {
	var s float64
	for _, v := range a {
		s += v * v
	}
	return math.Sqrt(s)
}

// gainCrossings returns the verified ω > 0 where |L(jω)| = gamma, ascending.
// Where the double solve is singular it evaluates in double-double: beside
// a cluster k/(z+1)⁴ with tiny k, |L| reaches gamma where the pivot of
// zI − A is below n·eps·‖A‖, yet the model is resolved there. Phase
// crossings keep the double solve wherever it resolves, as beside such a
// cluster, where the phase of the model as rounded departs from that of its
// factored form; only resonanceCrossings evaluates phase in double-double
// first, beside pairs that pass the conditioning bound of resonancePoles.
// Margins at any verified crossover use it too.
func (r *rationalLoop) gainCrossings(gamma float64) ([]float64, error) {
	cands, err := r.gainCandidates(gamma)
	if err != nil {
		return nil, err
	}
	lg := math.Log(gamma)
	f := func(w float64) float64 { return math.Log(cmplx.Abs(r.eval.at(w))) - lg }
	r.eval.extended = true
	defer func() { r.eval.extended = false }()
	return r.endCrossings(r.polish(cands, f, 1e-9, false), f, 1e-9), nil
}

// endCrossings adds a crossing of f = log|L| − log γ that the pencil missed
// below the first crossing ws[0], or, discrete, above the last: |L| → ∞ at a
// pole, and else |L| at ω = 0 or Nyquist, fixes the sign of f there, and a
// differing sign next to the outer crossing brackets one more. Pencil roots
// next to a pole cluster at z = 1 or −1, as of k/(z−1)ⁿ with tiny k, lose
// accuracy beyond any polish bracket. log|L| is continuous on the boundary
// away from its poles and zeros, where it does not change sign, so every
// bracket is a crossing. Next to the cluster the response carries relative
// rounding of order eps/θⁿ, so a root is accepted on its bisected finite
// sign bracket rather than on |f| at the root; a bracket closing on an
// unevaluable ω is rejected.
func (r *rationalLoop) endCrossings(ws []float64, f func(float64) float64, tol float64) []float64 {
	ft := func(w float64) float64 {
		v, ok := tryEval(r.eval, func() float64 { return f(w) })
		if !ok {
			return math.NaN()
		}
		return v
	}
	sgn := func(v float64) int {
		if math.IsNaN(v) || math.Abs(v) <= tol {
			return 0
		}
		if v > 0 {
			return 1
		}
		return -1
	}
	endSign := func(w float64) int {
		h, ok := r.eval.try(w)
		if !ok || cmplx.IsInf(h) {
			return 1
		}
		return sgn(f(w))
	}
	find := func(a, b float64) (float64, bool) {
		fa, fb := ft(a), ft(b)
		for range 64 {
			m := math.Sqrt(a * b)
			if m <= a || m >= b {
				break
			}
			if fm := ft(m); fa*fm <= 0 {
				b, fb = m, fm
			} else {
				a, fa = m, fm
			}
		}
		finite := !math.IsNaN(fa) && !math.IsNaN(fb) && !math.IsInf(fa, 0) && !math.IsInf(fb, 0)
		return math.Sqrt(a * b), finite && fa*fb <= 0
	}
	scan := func(t float64, at func(float64) float64, want int) (float64, bool) {
		step := 10.0
		for range 2000 {
			tn := t / step
			w, wn := at(t), at(tn)
			if tn < 1e-300 || wn == w {
				break
			}
			switch v := ft(wn); {
			case sgn(v) == want:
				return find(min(w, wn), max(w, wn))
			case !math.IsNaN(v):
				t = tn
			default:
				if step = math.Sqrt(step); step < 1+1e-9 {
					return 0, false
				}
			}
		}
		return 0, false
	}
	wTop := math.Inf(1)
	if r.discrete() {
		wTop = r.nyquist()
	}
	pLo, pHi := wTop/2, wTop/2
	if len(ws) > 0 {
		pLo, pHi = ws[0]*(1-1e-6), ws[len(ws)-1]*(1+1e-6)
	}
	if !math.IsInf(pLo, 0) {
		if s0, s := endSign(0), sgn(ft(pLo)); s0 != 0 && s != 0 && s != s0 {
			if c, ok := scan(pLo, func(t float64) float64 { return t }, s0); ok && (len(ws) == 0 || c < ws[0]*(1-1e-9)) {
				ws = append([]float64{c}, ws...)
			}
		}
	}
	if r.discrete() && pHi < wTop {
		if sTop, s := endSign(wTop), sgn(ft(pHi)); sTop != 0 && s != 0 && s != sTop {
			if c, ok := scan(wTop-pHi, func(t float64) float64 { return wTop - t }, sTop); ok && (len(ws) == 0 || c > ws[len(ws)-1]*(1+1e-9)) {
				ws = append(ws, c)
			}
		}
	}
	return ws
}

// phaseCrossings returns the verified ω > 0 where L(jω) is real and negative,
// ascending. The pencil root L − L~ always has at Nyquist, where L is real,
// is left to nyquistPhaseCrossings: polish would judge it on the double
// solve's phase at frequencyPoint(π/Dt). The candidates are polished with
// the extended fallback: beside a pole cluster near z = 1 the double solve
// fails over decades off every pole, and a crossing whose walk clearance
// lies there was lost (S2RNHI); there the pencil roots also fall far off
// the crossings and the double solve's phase is noisy, which polish's
// wideBracket and exactCrossing take up. polish's origin-pole guard keeps
// the double solve's failures, so near-roots of an asymptotically real L
// next to an origin cluster stay rejected. Beside a ring of poles around
// z = 1 the roots fall up to 6× off, or are missing, beyond any bracket;
// when a root is unevaluable, off the real axis by strayRootDeg, or fails
// polish, the scan beside real poles next to z = 1 takes in the ring too.
func (r *rationalLoop) phaseCrossings() ([]float64, error) {
	reals, err := r.realCandidates()
	if err != nil {
		return nil, err
	}
	r.eval.extended = true
	var cands []float64
	stray := false
	for _, w := range reals {
		if r.discrete() && w >= r.nyquist() {
			continue
		}
		h, ok := r.eval.try(w)
		off := math.Abs(phaseOffsetDeg(h))
		stray = stray || !ok || min(off, 180-off) > strayRootDeg
		if !ok || off < 90 {
			cands = append(cands, w)
		}
	}
	f := func(w float64) float64 { return phaseOffsetDeg(r.eval.at(w)) }
	out := r.polish(cands, f, 1e-7, true)
	r.eval.extended = false
	pairs, realPoles, cluster := r.resonancePoles()
	if len(cluster) > 0 && (stray || len(out) < len(cands)) {
		realPoles = cluster
	}
	out = r.dropRefuted(out, realPoles, f)
	out = r.mergeResonanceCrossings(out, r.resonanceCrossings(pairs, realPoles, f, 1e-7), f)
	return r.nyquistPhaseCrossings(out, f, 1e-7), nil
}

// nyquistPhaseCrossings adds the discrete phase crossings below Nyquist and
// above the others in ws that the pencil missed, then Nyquist itself as
// nyquistCrossover decides; ws holds none at Nyquist. Beside a zero cluster
// at z = −1, as of k·(z+1)⁴, the pencil roots of L − L~ fall up to ~1e-4
// off a crossing that lies within that of Nyquist, beyond any polish
// bracket, and the double solve's phase there is off by more than tol. It
// scans in double-double towards Nyquist, halving the distance, and keeps
// each sign change that refines to ulpTol; a ±180° wrap refines to |f| ~ 180
// and fails it. A sign change is skipped unrefined only when f jumps by 180°
// or more with both ends past ±90°: beside the cluster L can run from ~0°
// to near +0, −180° off, between scan points. A scan crossing with |L|
// within the solve noise of ddSISOResponseNoise is rounding residue, as of a
// cluster k·(z+1)⁴ at θ ~ π − 1e-10, |L| ~ 1e-41, where L → +0; with D ≠ 0
// it must also exceed the coefficient noise, of C = num − D·den rounded,
// as nyquistCrossover's L(−1); a NaN bound counts as above it. With D = 0 a
// companion realization keeps num as given, and crossings beside a cluster
// lie at |L| ~ that bound. A pole at Nyquist, where rounding makes sign
// changes beside the cluster, skips the scan. f must be phaseOffsetDeg of
// r.eval.at: a point keeps the double solve while L lies more than
// nyquistScanDDAbove times the larger coefficient noise at the scan's ends
// off the negative real axis, so a sign of f there is no rounding; a
// high-order loop pays two noise solves rather than ~50 double-double
// solves. The noise at the ends stands for the band's: crossings beside
// poles on the arc there are the resonance scan's.
func (r *rationalLoop) nyquistPhaseCrossings(ws []float64, f func(float64) float64, tol float64) []float64 {
	if !r.discrete() {
		return ws
	}
	wTop := r.nyquist()
	if h, ok := r.eval.try(wTop); !ok || cmplx.IsInf(h) || cmplx.IsNaN(h) {
		return ws
	}
	t := wTop / 2
	if len(ws) > 0 {
		t = ws[len(ws)-1] * (1 + 1e-6)
	}
	floor := math.Inf(1)
	if t < wTop && r.eval.sys.internalDelayCount() == 0 {
		_, _, c0, ok0 := ddSISOResponseNoise(r.eval.sys, r.eval.td.frequencyPoint(t))
		_, _, c1, ok1 := ddSISOResponseNoise(r.eval.sys, r.eval.td.frequencyPoint(wTop))
		r.eval.ddSolves += 2
		if ok0 && ok1 {
			floor = nyquistScanDDAbove * max(c0, c1)
		}
	}
	r.eval.extended = true
	defer func() { r.eval.extended, r.eval.exact = false, false }()
	ft := func(w float64) float64 {
		r.eval.exact = false
		v, ok := tryEval(r.eval, func() float64 { return f(w) })
		if ok && cmplx.Abs(r.eval.dst[0])*min(math.Abs(v)*math.Pi/180, 1) > floor {
			return v
		}
		r.eval.exact = true
		v, ok = tryEval(r.eval, func() float64 { return f(w) })
		if !ok {
			return math.NaN()
		}
		return v
	}
	aboveNoise := func(w float64, coeffs bool) bool {
		if r.eval.sys.internalDelayCount() > 0 {
			return true
		}
		h, solve, round, ok := ddSISOResponseNoise(r.eval.sys, r.eval.td.frequencyPoint(w))
		if coeffs {
			solve = max(solve, round)
		}
		return !ok || !(cmplx.Abs(h) <= solve)
	}
	biproper := r.eval.sys.D.At(0, 0) != 0
	out := slices.Clone(ws)
	a, fa := t, math.NaN()
	if t < wTop {
		fa = ft(t)
	}
	for gap := (wTop - t) / 2; gap > 0; gap /= 2 {
		b := wTop - gap
		if b <= a {
			break
		}
		fb := ft(b)
		if math.IsNaN(fb) {
			break
		}
		if !math.IsNaN(fa) && fa*fb < 0 && (math.Abs(fa-fb) < 180 || min(math.Abs(fa), math.Abs(fb)) < 90) {
			c := refineCrossing(a, b, ft)
			if math.Abs(ft(c)) <= ulpTol(c, ft, tol, 1e-3) && (len(out) == 0 || c > out[len(out)-1]*(1+1e-9)) && aboveNoise(c, biproper) {
				out = append(out, c)
			}
		}
		a, fa = b, fb
	}
	if (len(out) == 0 || wTop > out[len(out)-1]*(1+1e-9)) && r.nyquistCrossover(ft, tol, aboveNoise) {
		out = append(out, wTop)
	}
	return out
}

// nyquistCrossover reports whether L(−1) of the loop as given is a phase
// crossover: real and negative beyond the solve noise, and, with D ≠ 0,
// beyond the coefficient noise, as L(−1) of an ulp of |D| from C = num −
// D·den rounded; a NaN bound counts as above it. L real below Nyquist too has
// no isolated crossing there. Where the solve at z = −1 fails, the phase at
// frequencyPoint(π/Dt) decides, gated on both bounds there.
func (r *rationalLoop) nyquistCrossover(ft func(float64) float64, tol float64, aboveNoise func(float64, bool) bool) bool {
	wTop := r.nyquist()
	h, solve, coeffs, ok := r.nyquistResponse()
	if !ok {
		return math.Abs(ft(wTop)) <= tol && math.Abs(ft(wTop*(1-1e-3))) > tol && aboveNoise(wTop, true)
	}
	if r.eval.sys.D.At(0, 0) != 0 {
		solve = max(solve, coeffs)
	}
	return real(h) < 0 && !(-real(h) <= solve) && math.Abs(ft(wTop*(1-1e-3))) > tol
}

// dropRefuted removes the polished ws inside the scan band of a real pole
// pair next to z = 1 whose phase, in double-double, has one sign within 1e-5
// either side: below the pair the double solve's phase error makes sign
// changes up to a few percent off the crossing, which the scan supplies.
func (r *rationalLoop) dropRefuted(ws, poles []float64, f func(float64) float64) []float64 {
	if len(poles) == 0 {
		return ws
	}
	r.eval.extended, r.eval.exact = true, true
	defer func() { r.eval.extended, r.eval.exact = false, false }()
	at := func(w float64) float64 {
		v, ok := tryEval(r.eval, func() float64 { return f(w) })
		if !ok {
			return math.NaN()
		}
		return v
	}
	return slices.DeleteFunc(ws, func(w float64) bool {
		if !slices.ContainsFunc(poles, func(wp float64) bool { return w >= 1e-4*wp && w <= 10*wp }) {
			return false
		}
		lo, hi := at(w*(1-1e-5)), at(w*(1+1e-5))
		return math.Abs(lo) < 90 && math.Abs(hi) < 90 && lo*hi > 0
	})
}

// mergeResonanceCrossings inserts the resonance-scan crossings cs into the
// polished ws. Where the phase stays within tol of −180° over a band, as
// below a pair 1 ± jε, the double solve passes tol up to ~1e-6 off the
// crossing; a polished ω within 1e-5 of c whose phase, in double-double,
// has the sign it has between it and c is that crossing and gives way to c.
func (r *rationalLoop) mergeResonanceCrossings(ws, cs []float64, f func(float64) float64) []float64 {
	if len(cs) == 0 {
		return ws
	}
	r.eval.extended, r.eval.exact = true, true
	defer func() { r.eval.extended, r.eval.exact = false, false }()
	sign := func(w float64) int {
		v, ok := tryEval(r.eval, func() float64 { return f(w) })
		if !ok || math.IsNaN(v) {
			return 0
		}
		return cmp.Compare(v, 0)
	}
	for _, c := range cs {
		i, _ := slices.BinarySearch(ws, c)
		j := -1
		for _, k := range []int{i - 1, i} {
			if k >= 0 && k < len(ws) && math.Abs(ws[k]-c) <= 1e-5*c && (j < 0 || math.Abs(ws[k]-c) < math.Abs(ws[j]-c)) {
				j = k
			}
		}
		switch {
		case j >= 0 && math.Abs(ws[j]-c) <= 1e-7*c:
		case j >= 0 && sign(ws[j]) != 0 && sign(ws[j]) == sign((ws[j]+c)/2):
			ws[j] = c
		default:
			ws = slices.Insert(ws, i, c)
		}
	}
	slices.Sort(ws)
	return ws
}

// resonancePoles returns in pairs the boundary frequencies of the complex
// poles of L within 1% of their frequency from the boundary: lightly damped
// pairs, and pairs 1 ± jε next to z = 1. Discrete, reals holds |ln p|/Δt for real poles
// p within 1e-2 of z = 1 when at least two poles lie there: beside a real
// pair 1 ± ε the phase can stay within ~1e-6 rad of −180° and cross it at θ
// down to ~1e-2·ε, where the double solve's phase error is of that order.
// A pole whose first-order perturbation bound
// cond(p)·eps·(‖A‖ + |p|·‖E‖) exceeds 1e-3 of its frequency is left out: a
// double or quadruple real pole, as of an integrator squared, splits by
// rounding into such poles, whose crossings the loop lacks. Discrete,
// cluster holds |ln p|/Δt for the other pairs within clusterRadius of z = 1,
// with the real poles of reals however many lie there, or nothing without
// such a pair.
func (r *rationalLoop) resonancePoles() (pairs, reals, cluster []float64) {
	n := r.n
	if n == 0 {
		return nil, nil, nil
	}
	a, e := slices.Clone(r.a), slices.Clone(r.e)
	aNorm, eNorm := frobenius(a), frobenius(e)
	alphar, alphai, beta := make([]float64, n), make([]float64, n), make([]float64, n)
	vl, vr := make([]float64, n*n), make([]float64, n*n)
	work := make([]float64, max(8*n, 1))
	if !impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n, a, n, e, n, alphar, alphai, beta, vl, n, vr, n, work, len(work)) {
		return nil, nil, nil
	}
	x, y := make([]complex128, n), make([]complex128, n)
	resolved := func(p complex128, scale float64) bool {
		var nx, ny float64
		for i := range n {
			nx += real(x[i])*real(x[i]) + imag(x[i])*imag(x[i])
			ny += real(y[i])*real(y[i]) + imag(y[i])*imag(y[i])
		}
		var yex complex128
		for i := range n {
			var ex complex128
			for k := range n {
				ex += complex(r.e[i*n+k], 0) * x[k]
			}
			yex += cmplx.Conj(y[i]) * ex
		}
		cond := math.Sqrt(nx*ny) / cmplx.Abs(yex)
		return cond*eps()*(aNorm+cmplx.Abs(p)*eNorm) <= 1e-3*scale
	}
	var ws, nearOne []float64
	nOne := 0
	for j := 0; j < n; j++ {
		if alphai[j] == 0 {
			if !r.discrete() || beta[j] == 0 {
				continue
			}
			p := alphar[j] / beta[j]
			if math.Abs(p-1) > 1e-2 {
				continue
			}
			nOne++
			for i := range n {
				x[i], y[i] = complex(vr[i*n+j], 0), complex(vl[i*n+j], 0)
			}
			if th := math.Abs(math.Log(p)); th > 0 && resolved(complex(p, 0), th) {
				nearOne = append(nearOne, th/r.dt)
			}
			continue
		}
		jj := j
		j++
		if beta[jj] == 0 || alphai[jj] < 0 {
			continue
		}
		p := complex(alphar[jj]/beta[jj], alphai[jj]/beta[jj])
		for i := range n {
			x[i] = complex(vr[i*n+jj], vr[i*n+jj+1])
			y[i] = complex(vl[i*n+jj], vl[i*n+jj+1])
		}
		if !resolved(p, imag(p)) {
			continue
		}
		var w, dist float64
		if r.discrete() {
			w, dist = cmplx.Phase(p)/r.dt, math.Abs(cmplx.Abs(p)-1)/cmplx.Phase(p)
		} else {
			w, dist = imag(p), math.Abs(real(p))/imag(p)
		}
		if dist <= 1e-2 {
			ws = append(ws, w)
		} else if r.discrete() && cmplx.Abs(p-1) <= clusterRadius {
			cluster = append(cluster, cmplx.Abs(cmplx.Log(p))/r.dt)
		}
	}
	if len(cluster) > 0 {
		cluster = append(cluster, nearOne...)
	}
	if nOne < 2 {
		nearOne = nil
	}
	return ws, nearOne, cluster
}

// resonanceCrossings scans f, in double-double, beside each resonance pole
// on a grid graded towards it, from 1e-4 to ten times its frequency: below
// a pair 1 ± jε the phase can stay within 1e-8° of −180° and cross it down
// to ~1e-2 of the pole frequency, where pencil roots land far off and the
// double solve's phase error exceeds tol next to the pole. Pencil roots near a
// pole pair 1 ± jε, ε ≲ 1e-4, are off by more than ε: the pencil of L − L~
// carries the pair, its reflection and the crossings in a cluster of radius
// ε, whose roots move by a root of the backward error. A sign change between
// evaluable grid points with |f| < 90, the far side of the pole and the
// ±180 wrap excluded, is kept when it refines to |f| ≤ tol and still holds
// walkClearance either side, or to the bracket when closer; tol widens to
// ulpTol, up to 30°, where f moves by more than tol per ulp: with |f| < 90 no
// jump of 180° or 360° makes a sign change, and pairs down to ζ ~ 1e-14 cross
// within ulps of the pole (BYP4ZM). A bracket end bisected to |f| = 90, as
// beside such a pair, is the clearance bound there.
func (r *rationalLoop) resonanceCrossings(pairs, reals []float64, f func(float64) float64, tol float64) []float64 {
	wTop := math.Inf(1)
	if r.discrete() {
		wTop = r.nyquist()
	}
	ddBelow := math.Inf(1)
	ft := func(w float64) float64 {
		var v float64
		ok := false
		if ddBelow < math.Inf(1) {
			r.eval.exact = false
			v, ok = tryEval(r.eval, func() float64 { return f(w) })
			ok = ok && math.Abs(v) >= ddBelow
		}
		if !ok {
			r.eval.exact = true
			v, ok = tryEval(r.eval, func() float64 { return f(w) })
		}
		if !ok || math.Abs(v) >= 90 {
			return math.NaN()
		}
		return v
	}
	r.eval.extended = true
	defer func() { r.eval.extended, r.eval.exact = false, false }()
	const perDecade = 20
	var out []float64
	var grids [][]float64
	for _, wp := range pairs {
		var grid []float64
		for i := 3 * perDecade; i > 0; i-- {
			grid = append(grid, wp*math.Pow(10, -1-float64(i)/perDecade))
		}
		for i := 4 * perDecade; i >= 0; i-- {
			grid = append(grid, wp*(1-0.9*math.Pow(10, -float64(i)/perDecade)))
		}
		for i := range 5*perDecade + 1 {
			if w := wp * (1 + 9*math.Pow(10, -float64(5*perDecade-i)/perDecade)); w < wTop {
				grid = append(grid, w)
			}
		}
		slices.Sort(grid)
		grids = append(grids, grid)
	}
	if len(reals) > 0 {
		lo, hi := slices.Min(reals)*1e-4, min(slices.Max(reals)*10, wTop)
		var grid []float64
		for i := 0; ; i++ {
			w := lo * math.Pow(10, float64(i)/(2*perDecade))
			if w >= hi {
				break
			}
			grid = append(grid, w)
		}
		grids = append(grids, grid)
	}
	for k, grid := range grids {
		if k == len(pairs) {
			ddBelow = realPoleDDBelow
		}
		next := math.NaN()
		if len(grid) > 0 {
			next = ft(grid[0])
		}
		for i := 1; i < len(grid); i++ {
			a, b := grid[i-1], grid[i]
			fa, fb := next, ft(b)
			next = fb
			if math.IsNaN(fa) == math.IsNaN(fb) {
				if math.IsNaN(fa) || fa*fb > 0 {
					continue
				}
			} else {
				ok, bad := a, b
				if math.IsNaN(fa) {
					ok, bad = b, a
				}
				for range 60 {
					m := (ok + bad) / 2
					if m == ok || m == bad {
						break
					}
					if math.IsNaN(ft(m)) {
						bad = m
					} else {
						ok = m
					}
				}
				if math.IsNaN(fa) {
					a, fa = ok, ft(ok)
				} else {
					b, fb = ok, ft(ok)
				}
				if a >= b || fa*fb > 0 {
					continue
				}
			}
			c := refineCrossing(a, b, ft)
			if math.Abs(ft(c)) <= ulpTol(c, ft, tol, 30) && ft(max(a, c*(1-walkClearance)))*ft(min(b, c*(1+walkClearance))) < 0 {
				out = append(out, c)
			}
		}
	}
	slices.Sort(out)
	return slices.CompactFunc(out, func(a, b float64) bool { return math.Abs(b-a) <= 1e-7*b })
}

// polish refines each candidate to a sign change of f within half the gap to
// its neighbours, so close crossing pairs stay apart, and keeps those where
// |f| ≤ tol. A candidate without a sign change, a tangency or the discrete
// Nyquist frequency, is kept only if |f| ≤ tol there. When ω = 0 itself
// satisfies |f| ≤ tol, as for the root L − L~ always has there, a candidate
// below every kept crossing whose |f| stays within tol down to half its
// frequency is that root moved off 0 by rounding, and is dropped; above a
// kept crossing the test would drop crossings at twice another, as for
// k·z^−d with k < 0. When L has a pole at ω = 0, a
// candidate is dropped only when L cannot be evaluated at half its
// frequency, or at itself: a pencil root that the pole's rounding moved off
// 0. |f| ≈ 0 at half its frequency is no test there, as for
// k/((z−1)²·z^d), whose crossovers include twice one another. f is only
// probed where L evaluates: a bracket next to a pole cluster, as at z = −1
// of k/(z+1)⁴, may reach a numerically singular ω, which rejects it, and a
// refinement that meets one there ends off the crossing and is dropped.
// Pencil roots next to a pole pair 1 ± jε land up to a few percent off the
// crossing, at times on or past the pole: when no bracket spans the crossing
// and the candidate or a bracket end is unevaluable, walk scans out to a
// tenth of the frequency, below first, past the singular band, and keeps the
// first sign change that refines to |f| ≤ tol and still holds walkClearance
// either side. Rounding grows without bound towards the band, as eps/δ⁴
// beside k/(z+1)⁴, and makes sign changes there that the loop lacks. When
// steep, a refined sign change is held to ulpTol rather than tol, and one
// that refines to a ±180° phase wrap, |f| > 90, as a wide bracket below the
// crossing at Nyquist may span, leaves the candidate unbracketed.
func (r *rationalLoop) polish(cands []float64, f func(float64) float64, tol float64, steep bool) []float64 {
	wTop := math.Inf(1)
	if r.discrete() {
		wTop = r.nyquist()
	}
	ft := func(w float64) float64 {
		v, ok := tryEval(r.eval, func() float64 { return f(w) })
		if !ok {
			return math.NaN()
		}
		return v
	}
	within := func(w float64) bool { return math.Abs(ft(w)) <= tol }
	crossTol := func(c float64) float64 {
		if steep {
			return ulpTol(c, ft, tol, 1e-3)
		}
		return tol
	}
	// The phase search evaluates extended, but its guard against roots next
	// to an origin pole reads the double solve's failures: there only they
	// mark near-roots of an asymptotically real L.
	guard := r.eval.try
	if steep {
		guard = r.eval.tryDouble
	}
	h0, ok := r.eval.try(0)
	pole0 := !ok || cmplx.IsInf(h0)
	zeroRoot := !pole0 && math.Abs(f(0)) <= tol
	walk := func(w, lim float64) (float64, bool) {
		pw, pf := w, ft(w)
		for d := 1e-12; ; d *= 2 {
			x := w + math.Copysign(min(d*w, math.Abs(lim)), lim)
			fx := ft(x)
			switch {
			case math.IsNaN(fx):
				pw, pf = x, fx
			case math.IsNaN(pf):
				e := x
				for range 60 {
					m := (pw + e) / 2
					if m == pw || m == e {
						break
					}
					if math.IsNaN(ft(m)) {
						pw = m
					} else {
						e = m
					}
				}
				pw, pf = e, ft(e)
				fallthrough
			default:
				if pf*fx <= 0 && pw != x {
					c := refineCrossing(min(pw, x), max(pw, x), ft)
					if hc := c * (1 + walkClearance); hc < wTop && math.Abs(ft(c)) <= crossTol(c) && ft(c*(1-walkClearance))*ft(hc) < 0 {
						return c, true
					}
				}
				pw, pf = x, fx
			}
			if d*w >= math.Abs(lim) {
				return 0, false
			}
		}
	}
	var out []float64
	for i, w := range cands {
		lo, hi := w/2, (wTop-w)/2
		if i > 0 {
			lo = (w - cands[i-1]) / 2
		}
		if i+1 < len(cands) {
			hi = min(hi, (cands[i+1]-w)/2)
		}
		best, bracketed, unevaluable := w, false, false
		if _, ok := r.eval.try(w); !ok {
			unevaluable = true
		} else {
			for d := 1e-12; d <= 1e-2; d *= 100 {
				a, b := w-min(d*w, lo), w+min(d*w, hi)
				if a >= b {
					continue
				}
				fa, fb := ft(a), ft(b)
				if fa*fb <= 0 {
					if c := refineCrossing(a, b, ft); !steep || math.Abs(ft(c)) <= 90 {
						best, bracketed = c, true
					}
					break
				}
				unevaluable = unevaluable || math.IsNaN(fa) || math.IsNaN(fb)
			}
			if steep && !bracketed {
				best, bracketed = r.wideBracket(w, lo, hi, wTop, tol, ft)
			}
		}
		if !bracketed && unevaluable {
			c, ok := walk(w, -min(lo, w/10))
			if !ok {
				c, ok = walk(w, min(hi, w/10))
			}
			if ok {
				best, bracketed = c, true
			} else if _, ok := r.eval.try(w); !ok {
				continue
			}
		}
		v, vTol := math.Abs(ft(best)), tol
		if bracketed {
			vTol = crossTol(best)
		}
		if steep && bracketed && v > vTol && !r.eval.exact {
			if c, ok := r.exactCrossing(best, wTop, ft, crossTol); ok {
				best, v = c, 0
			}
		}
		if _, ok := guard(best / 2); zeroRoot && len(out) == 0 && within(best/2) || pole0 && !ok {
			continue
		}
		if v <= vTol && (len(out) == 0 || best > out[len(out)-1]*(1+1e-12)) {
			out = append(out, best)
		}
	}
	if len(out) > 0 && r.flat(out, within, wTop) {
		return nil
	}
	return out
}

// wideBracket looks for a phase crossing of a pencil root w, off it by more
// than tol, that polish's symmetric brackets up to 1% missed, on one side of
// w at a time, out to 10% and then to lo or hi. Beside a pole cluster near
// z = 1 the roots of L − L~ fall up to ~25% off the crossing (S2RNHI), where
// a symmetric bracket meets the next ±180° wrap before the crossing. A
// bisection onto a wrap, |f| > 90, is no crossing, nor one without walk's
// clearance below wTop: beside a cluster at z = −1 the wide bracket reaches
// sign changes of L within its coefficient noise next to Nyquist.
func (r *rationalLoop) wideBracket(w, lo, hi, wTop, tol float64, ft func(float64) float64) (float64, bool) {
	fw := ft(w)
	if !(math.Abs(fw) > tol) {
		return w, false
	}
	for _, d := range [...]float64{0.1, 1} {
		for _, x := range [...]float64{w - min(d*w, lo), w + min(d*w, hi)} {
			if fx := ft(x); x == w || !(fx*fw <= 0) {
				continue
			}
			c := refineCrossing(min(x, w), max(x, w), ft)
			if hc := c * (1 + walkClearance); hc < wTop && math.Abs(ft(c)) <= 90 && ft(c*(1-walkClearance))*ft(hc) < 0 {
				return c, true
			}
		}
	}
	return w, false
}

// exactCrossing re-bisects a bracketed phase crossing c that misses its
// tolerance in double-double over c(1 ± 1e-6). Beside a pole cluster near
// z = 1 the double solve's phase is off by ~1e-7°, so the bisected sign
// change of a crossing several degrees per percent steep sits there at
// |f| ~ 2e-7° (S2RNHI). It needs walk's clearance below wTop, as
// wideBracket: next to Nyquist beside a cluster at z = −1 a sign change of
// L within its coefficient noise refines in double-double too.
func (r *rationalLoop) exactCrossing(c, wTop float64, ft, crossTol func(float64) float64) (float64, bool) {
	if c*(1+walkClearance) >= wTop {
		return c, false
	}
	r.eval.exact = true
	defer func() { r.eval.exact = false }()
	a, b := c*(1-1e-6), c*(1+1e-6)
	if !(ft(a)*ft(b) <= 0) || !(ft(c*(1-walkClearance))*ft(c*(1+walkClearance)) < 0) {
		return c, false
	}
	c = refineCrossing(a, b, ft)
	return c, math.Abs(ft(c)) <= crossTol(c)
}

// flat reports whether within holds also between and beyond the crossings ws,
// as when the pencil is singular because |L| ≡ γ or L is real on the whole
// boundary; such a loop has no isolated crossings.
func (r *rationalLoop) flat(ws []float64, within func(float64) bool, wTop float64) bool {
	probes := []float64{ws[0] / 2}
	for i := 1; i < len(ws); i++ {
		probes = append(probes, math.Sqrt(ws[i-1]*ws[i]))
	}
	if last := ws[len(ws)-1]; last < wTop {
		probes = append(probes, min(2*last, (last+wTop)/2))
	}
	for _, w := range probes {
		if !within(w) {
			return false
		}
	}
	return true
}
