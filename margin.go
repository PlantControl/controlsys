package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

type MarginResult struct {
	GainMargin  float64 // dB; +Inf if no phase crossover
	PhaseMargin float64 // degrees in (-180,180]; +Inf if no gain crossover
	WgFreq      float64 // gain crossover freq (0dB); NaN if none
	WpFreq      float64 // phase crossover freq (-180deg mod 360); NaN if none
}

type AllMarginResult struct {
	GainMargins     []float64 // dB at each phase crossover
	PhaseMargins    []float64 // degrees in (-180,180] at each gain crossover
	GainCrossFreqs  []float64 // omega where |G|=0dB
	PhaseCrossFreqs []float64 // omega where angle(G)=-180deg mod 360
}

// DiskMarginResult holds a SISO disk margin, mirroring the fields of MATLAB
// diskmargin. Alpha, GainMargin, PhaseMargin and Frequency follow the skew
// Skew; PeakSensitivity and PeakFreq always describe ‖S‖∞.
type DiskMarginResult struct {
	Alpha           float64    // disk margin αmax = 1/‖S+(σ−1)/2‖∞
	Skew            float64    // σ of the gain-variation disk
	Frequency       float64    // ω where the disk margin is attained
	GainMargin      [2]float64 // [low, high] linear gain factors
	GainMarginDB    [2]float64 // [low, high] in dB
	PhaseMargin     float64    // +/- degrees
	PeakSensitivity float64    // Ms = ‖S‖∞
	PeakFreq        float64    // ω where |S| peaks
}

// sisoEval evaluates a SISO loop on the boundary of the stability region.
// Delay-free realizations reuse one state-space solver, so refinement loops
// pay its setup once and high-order loops keep state-space accuracy.
type sisoEval struct {
	sys    *System
	solver frequencyPointSolver
	delay  *mat.Dense
	lft    bool
	td     timeDomain
	dst    []complex128
}

func newSISOEval(sys *System) (*sisoEval, error) {
	e := &sisoEval{
		sys: sys,
		td:  newTimeDomain(sys.Dt),
		dst: make([]complex128, 1),
	}
	if sys.internalDelayCount() > 0 {
		e.lft = true
		return e, nil
	}
	if err := sys.Validate(); err != nil {
		return nil, err
	}
	e.solver = newFrequencyEvaluator(sys).pointSolver(math.MaxInt)
	e.delay = effectiveIODelayMatrix(sys, 1, 1, true)
	return e, nil
}

func (e *sisoEval) at(w float64) complex128 {
	if e.lft {
		h, _ := evalSISOFreqResponse(e.sys, w)
		return h
	}
	s := e.td.frequencyVariable(w)
	if err := evalWithPoleLimit(e.solver.evalInto, e.sys, s, e.dst); err != nil {
		return complex(math.NaN(), math.NaN())
	}
	if e.delay != nil {
		applyIODelayMatrixAtS(e.sys, s, e.dst, 1, 1, e.delay)
	}
	return e.dst[0]
}

const (
	marginMaxRefineDepth   = 10
	marginMaxDelayPoints   = 20000
	marginMaxExtendDecades = 10
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

func phaseOffsetDeg(h complex128) float64 {
	return wrapDegrees(cmplx.Phase(h)*180/math.Pi + 180)
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

func evalSISOFreqResponse(sys *System, w float64) (complex128, error) {
	resp, err := sys.FreqResponse([]float64{w})
	if err != nil {
		return 0, err
	}
	return resp.At(0, 0, 0), nil
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
		gain, phase, err := gridMarginCrossings(sys, eval, loop == nil)
		if err != nil {
			return nil, err
		}
		if loop == nil {
			res.GainCrossFreqs = gain
		}
		res.PhaseCrossFreqs = phase
	}
	for _, w := range res.GainCrossFreqs {
		res.PhaseMargins = append(res.PhaseMargins, phaseMarginDeg(eval.at(w)))
	}
	for _, w := range res.PhaseCrossFreqs {
		res.GainMargins = append(res.GainMargins, -20*math.Log10(cmplx.Abs(eval.at(w))))
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
		phaseDeg[k] = cmplx.Phase(h) * 180 / math.Pi
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
			return phaseOffsetDeg(eval.at(w))
		})
		if math.Abs(phaseOffsetDeg(eval.at(w))) <= 1 {
			phase = append(phase, w)
		}
	}
	return gain, phase, nil
}

func Margin(sys *System) (*MarginResult, error) {
	all, err := AllMargin(sys)
	if err != nil {
		return nil, err
	}
	return pickMargins(all), nil
}

// pickMargins selects, as MATLAB margin
// (https://www.mathworks.com/help/control/ref/dynamicsystem.margin.html),
// the gain margin closest to 0 dB and the phase margin closest to 0°, each
// with its frequency; the lower frequency wins a tie.
func pickMargins(all *AllMarginResult) *MarginResult {
	result := &MarginResult{}
	result.GainMargin, result.WpFreq = selectMargin(all.GainMargins, all.PhaseCrossFreqs)
	result.PhaseMargin, result.WgFreq = selectMargin(all.PhaseMargins, all.GainCrossFreqs)
	return result
}

func selectMargin(margins, freqs []float64) (float64, float64) {
	m, w := math.Inf(1), math.NaN()
	for i, v := range margins {
		if math.Abs(v) < math.Abs(m) {
			m, w = v, freqs[i]
		}
	}
	return m, w
}

// Bandwidth returns the first frequency where the gain of sys drops dbDrop dB
// (default −3) below its DC value, as MATLAB bandwidth
// (https://www.mathworks.com/help/control/ref/dynamicsystem.bandwidth.html).
// For SISO models without continuous internal delays it is the smallest
// boundary eigenvalue of the Hamiltonian pencil of |L| = g, refined on the
// exact response; MIMO models, which MATLAB rejects, use σmax on a frequency
// grid. It returns +Inf when
// the gain never drops that far and 0 when the DC gain is 0 or not finite.
// dbDrop must be 0 (default) or a finite negative scalar, and a model with
// no inputs or no outputs is rejected.
func Bandwidth(sys *System, dbDrop float64) (float64, error) {
	if dbDrop == 0 {
		dbDrop = -3
	}
	if !(dbDrop < 0) || math.IsInf(dbDrop, -1) {
		return 0, fmt.Errorf("Bandwidth: dbDrop must be a finite negative scalar, got %g: %w", dbDrop, ErrInvalidArgument)
	}
	_, m, p := sys.Dims()
	if m == 0 || p == 0 {
		return 0, fmt.Errorf("Bandwidth: model has no inputs or no outputs: %w", ErrDimensionMismatch)
	}

	dcGain, err := sys.DCGain()
	if err != nil {
		return 0, fmt.Errorf("Bandwidth: %w", err)
	}

	var dcMag float64
	if p == 1 && m == 1 {
		dcMag = math.Abs(dcGain.At(0, 0))
	} else {
		dcMag = maxSVDense(dcGain, p, m)
	}
	if dcMag == 0 || math.IsNaN(dcMag) || math.IsInf(dcMag, 0) {
		return 0, nil
	}

	threshold := 20*math.Log10(dcMag) + dbDrop
	siso := p == 1 && m == 1
	if siso {
		loop, err := newRationalLoop(sys)
		if err != nil && !errors.Is(err, ErrContinuousInternalDelay) {
			return 0, err
		}
		if err == nil {
			ws, err := loop.gainCrossings(math.Pow(10, threshold/20))
			if err != nil {
				return 0, err
			}
			if len(ws) == 0 {
				return math.Inf(1), nil
			}
			return ws[0], nil
		}
	}

	omega, err := sys.DefaultFrequencyGrid(1000)
	if err != nil {
		return 0, err
	}
	if len(omega) == 0 {
		return math.Inf(1), nil
	}

	nw := len(omega)
	magDB := make([]float64, nw)
	gainDB := func(g float64) float64 {
		if g > 0 {
			return 20 * math.Log10(g)
		}
		return -1000
	}

	var eval *sisoEval
	if siso {
		eval, err = newSISOEval(sys)
		if err != nil {
			return 0, err
		}
		for k, w := range omega {
			magDB[k] = gainDB(cmplx.Abs(eval.at(w)))
		}
	} else {
		sigma, err := sys.Sigma(omega, 0)
		if err != nil {
			return 0, err
		}
		for k := range nw {
			magDB[k] = gainDB(sigma.At(k, 0))
		}
	}

	if magDB[0] < threshold {
		return 0, nil
	}

	crossings := findCrossings(omega, magDB, threshold)
	if len(crossings) == 0 {
		return math.Inf(1), nil
	}

	c := crossings[0]
	w := refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
		if siso {
			return gainDB(cmplx.Abs(eval.at(w))) - threshold
		}
		sig, err := sys.Sigma([]float64{w}, 0)
		if err != nil {
			return 0
		}
		return gainDB(sig.At(0, 0)) - threshold
	})

	return w, nil
}

// DiskMargin computes the balanced (skew σ = 0) disk margin of the SISO loop
// sys, the default of MATLAB diskmargin
// (https://www.mathworks.com/help/robust/ref/dynamicsystem.diskmargin.html).
// It is DiskMarginSkew(sys, 0): αmax = 1/‖S − 1/2‖∞, GainMargin
// [(2−α)/(2+α), (2+α)/(2−α)] and PhaseMargin 2·atan(α/2).
func DiskMargin(sys *System) (*DiskMarginResult, error) {
	return DiskMarginSkew(sys, 0)
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
// response; see delayLoop.nyquist for the resolution limits. Loops whose
// internal delays form a feedback cycle return ErrContinuousInternalDelay;
// loops whose high-frequency gain may reach 1 (neutral type), whose delay grid
// would exceed 2^21 points, or whose encirclement count does not resolve
// return ErrDelayUnsupported. Descriptor loops return
// ErrDescriptorUnsupported, as HinfNorm does.
func DiskMarginSkew(sys *System, sigma float64) (*DiskMarginResult, error) {
	if math.IsNaN(sigma) || math.IsInf(sigma, 0) {
		return nil, fmt.Errorf("DiskMargin: skew must be finite, got %g", sigma)
	}
	if _, err := newSISOLoopModel(sys, "DiskMargin"); err != nil {
		return nil, err
	}
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
		return nil, fmt.Errorf("DiskMargin: cannot form sensitivity: %w", err)
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
			return nil, fmt.Errorf("DiskMargin: cannot form shifted sensitivity: %w", err)
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
	loop, err := delayLoopFromSystem(sys, "DiskMargin")
	if err != nil {
		return nil, diskMarginDelayError(err)
	}
	stable, peaks, err := loop.sensitivityPeaks(0, shift)
	if err != nil {
		return nil, diskMarginDelayError(err)
	}
	if !stable {
		return unstableDiskMargin(sigma), nil
	}
	return diskMarginFromPeak(sigma, peaks[1].peak, peaks[1].w, peaks[0].peak, peaks[0].w), nil
}

func diskMarginDelayError(err error) error {
	if errors.Is(err, errDelayLoopUnsupported) || errors.Is(err, ErrContinuousInternalDelay) {
		return fmt.Errorf("DiskMargin: %w", err)
	}
	return err
}

func diskMarginFromPeak(sigma, peak, wPeak, Ms, wMs float64) *DiskMarginResult {
	alpha := 1 / peak
	gm, pm := diskGainPhaseMargin(alpha, sigma)
	return &DiskMarginResult{
		Alpha:           alpha,
		Skew:            sigma,
		Frequency:       wPeak,
		GainMargin:      gm,
		GainMarginDB:    [2]float64{20 * math.Log10(gm[0]), 20 * math.Log10(gm[1])},
		PhaseMargin:     pm,
		PeakSensitivity: Ms,
		PeakFreq:        wMs,
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
	for i := range n {
		ci := r.c[i] / gamma
		for j := range n {
			m[i*N+j] = r.a[i*n+j]
			m[(n+i)*N+j] = -ci * r.c[j] / gamma
			k[i*N+j] = r.e[i*n+j]
			if r.discrete() {
				m[(n+i)*N+n+j] = r.e[j*n+i]
				k[(n+i)*N+n+j] = r.a[j*n+i]
			} else {
				m[(n+i)*N+n+j] = -r.a[j*n+i]
				k[(n+i)*N+n+j] = r.e[j*n+i]
			}
		}
		m[i*N+u] = r.b[i]
		m[(n+i)*N+u] = -ci * d
		m[u*N+i] = d * ci
		if r.discrete() {
			k[u*N+n+i] = -r.b[i]
		} else {
			m[u*N+n+i] = r.b[i]
		}
	}
	m[u*N+u] = d*d - 1
	return r.boundaryRoots(m, k, N)
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
	kNorm := frobenius(k)
	alphar, alphai, beta := make([]float64, N), make([]float64, N), make([]float64, N)
	work := make([]float64, 8*N)
	if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, N, m, N, k, N, alphar, alphai, beta, nil, 1, nil, 1, work, len(work)) {
		return nil, fmt.Errorf("margin: QZ: %w", ErrSchurFailed)
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

func frobenius(a []float64) float64 {
	var s float64
	for _, v := range a {
		s += v * v
	}
	return math.Sqrt(s)
}

// gainCrossings returns the verified ω > 0 where |L(jω)| = gamma, ascending.
func (r *rationalLoop) gainCrossings(gamma float64) ([]float64, error) {
	cands, err := r.gainCandidates(gamma)
	if err != nil {
		return nil, err
	}
	lg := math.Log(gamma)
	f := func(w float64) float64 { return math.Log(cmplx.Abs(r.eval.at(w))) - lg }
	return r.polish(cands, f, 1e-9), nil
}

// phaseCrossings returns the verified ω > 0 where L(jω) is real and negative,
// ascending.
func (r *rationalLoop) phaseCrossings() ([]float64, error) {
	reals, err := r.realCandidates()
	if err != nil {
		return nil, err
	}
	var cands []float64
	for _, w := range reals {
		if math.Abs(phaseOffsetDeg(r.eval.at(w))) < 90 {
			cands = append(cands, w)
		}
	}
	return r.polish(cands, func(w float64) float64 { return phaseOffsetDeg(r.eval.at(w)) }, 1e-7), nil
}

// polish refines each candidate to a sign change of f within half the gap to
// its neighbours, so close crossing pairs stay apart, and keeps those where
// |f| ≤ tol. A candidate without a sign change, a tangency or the discrete
// Nyquist frequency, is kept only if |f| ≤ tol there. When ω = 0 itself
// satisfies |f| ≤ tol, as for the root L − L~ always has there, a candidate
// whose |f| stays within tol down to half its frequency is that root moved
// off 0 by rounding, and is dropped.
func (r *rationalLoop) polish(cands []float64, f func(float64) float64, tol float64) []float64 {
	wTop := math.Inf(1)
	if r.discrete() {
		wTop = r.nyquist()
	}
	zeroRoot := math.Abs(f(0)) <= tol
	var out []float64
	for i, w := range cands {
		lo, hi := w/2, (wTop-w)/2
		if i > 0 {
			lo = (w - cands[i-1]) / 2
		}
		if i+1 < len(cands) {
			hi = min(hi, (cands[i+1]-w)/2)
		}
		best := w
		for d := 1e-12; d <= 1e-2; d *= 100 {
			a, b := w-min(d*w, lo), w+min(d*w, hi)
			if a >= b || !(f(a)*f(b) <= 0) {
				continue
			}
			best = refineCrossing(a, b, f)
			break
		}
		v := math.Abs(f(best))
		if zeroRoot && math.Abs(f(best/2)) <= tol {
			continue
		}
		if v <= tol && (len(out) == 0 || best > out[len(out)-1]*(1+1e-12)) {
			out = append(out, best)
		}
	}
	if len(out) > 0 && r.flat(out, f, tol, wTop) {
		return nil
	}
	return out
}

// flat reports whether |f| ≤ tol also between and beyond the crossings ws,
// as when the pencil is singular because |L| ≡ γ or L is real on the whole
// boundary; such a loop has no isolated crossings.
func (r *rationalLoop) flat(ws []float64, f func(float64) float64, tol, wTop float64) bool {
	probes := []float64{ws[0] / 2}
	for i := 1; i < len(ws); i++ {
		probes = append(probes, math.Sqrt(ws[i-1]*ws[i]))
	}
	if last := ws[len(ws)-1]; last < wTop {
		probes = append(probes, min(2*last, (last+wTop)/2))
	}
	for _, w := range probes {
		if !(math.Abs(f(w)) <= tol) {
			return false
		}
	}
	return true
}
