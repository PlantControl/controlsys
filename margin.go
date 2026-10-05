package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

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

// sisoEval caches the TF and reuses a single-element buffer for evalInto,
// eliminating repeated TransferFunction calls in refinement loops.
type sisoEval struct {
	sys  *System
	tf   *TransferFunc
	lft  bool
	cont bool
	dt   float64
	dst  []complex128
}

func newSISOEval(sys *System) (*sisoEval, error) {
	e := &sisoEval{
		sys:  sys,
		cont: sys.IsContinuous(),
		dt:   sys.Dt,
		dst:  make([]complex128, 1),
	}
	if sys.internalDelayCount() > 0 {
		e.lft = true
	} else {
		res, err := sys.TransferFunction(nil)
		if err != nil {
			return nil, err
		}
		e.tf = res.TF
	}
	return e, nil
}

func (e *sisoEval) at(w float64) complex128 {
	if e.lft {
		h, _ := evalSISOFreqResponse(e.sys, w)
		return h
	}
	var s complex128
	if e.cont {
		s = complex(0, w)
	} else {
		s = cmplx.Exp(complex(0, w*e.dt))
	}
	e.tf.evalInto(s, e.dst)
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
// the crossovers are the boundary roots of |N|² − |D|² and Im(N·conj(D)),
// refined on the exact response, so no frequency grid can miss them.
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
	if err != nil {
		return nil, err
	}
	res := &AllMarginResult{}
	var eval *sisoEval
	if loop != nil {
		eval = loop.eval
		res.GainCrossFreqs = loop.gainCrossings(1)
		if !loop.delayed {
			res.PhaseCrossFreqs = loop.phaseCrossings()
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
// boundary root of |N|² − g²|D|², refined on the exact response; MIMO models,
// which MATLAB rejects, use σmax on a frequency grid. It returns +Inf when
// the gain never drops that far and 0 when the DC gain is 0 or not finite.
func Bandwidth(sys *System, dbDrop float64) (float64, error) {
	if dbDrop == 0 {
		dbDrop = -3
	}
	_, m, p := sys.Dims()

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
	if dbDrop > 0 {
		return 0, nil
	}

	threshold := 20*math.Log10(dcMag) + dbDrop
	siso := p == 1 && m == 1
	if siso {
		loop, err := newRationalLoop(sys)
		if err != nil {
			return 0, err
		}
		if loop != nil {
			ws := loop.gainCrossings(math.Pow(10, threshold/20))
			if len(ws) == 0 {
				return math.Inf(1), nil
			}
			return ws[0], nil
		}
	}

	omega, err := autoBodeFreqs(sys, 1000)
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
// internal delays form a feedback cycle, whose high-frequency gain may reach
// 1 (neutral type), or whose delay grid would exceed 2^21 points return
// ErrContinuousInternalDelay. Descriptor loops return
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
	if errors.Is(err, errDelayLoopUnsupported) {
		return fmt.Errorf("DiskMargin: %v: %w", err, ErrContinuousInternalDelay)
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

// rationalLoop is the rational part N/D of a SISO loop, whose unit-gain and
// real-axis crossings are roots of polynomials, as for MATLAB allmargin on
// LTI models. Discrete delays are folded into D as powers of z. A continuous
// loop delay leaves |L(jω)| rational but not angle(L), so delayed reports
// that only gain crossings are exact.
type rationalLoop struct {
	num, den Poly
	dt       float64
	delayed  bool
	eval     *sisoEval
}

// newRationalLoop returns nil when sys has continuous internal delays, for
// which no rational form exists.
func newRationalLoop(sys *System) (*rationalLoop, error) {
	if sys.IsContinuous() && sys.HasInternalDelay() {
		return nil, nil
	}
	fsys, err := finiteDimensionalModel(sys, "margin")
	if err != nil {
		return nil, err
	}
	res, err := fsys.TransferFunction(nil)
	if err != nil {
		return nil, err
	}
	tf := res.TF
	r := &rationalLoop{
		num: slices.Clone(Poly(tf.Num[0][0])),
		den: slices.Clone(Poly(tf.Den[0])),
		dt:  fsys.Dt,
	}
	if tf.Delay != nil && tf.Delay[0][0] != 0 {
		if r.dt == 0 {
			r.delayed = true
		} else {
			r.den = append(r.den, make(Poly, int(math.Round(tf.Delay[0][0])))...)
		}
	}
	if r.eval, err = newSISOEval(fsys); err != nil {
		return nil, err
	}
	return r, nil
}

func (r *rationalLoop) discrete() bool { return r.dt > 0 }

func (r *rationalLoop) nyquist() float64 { return math.Pi / r.dt }

// mirror returns P(-s) for a continuous loop and z^n·P(1/z) for a discrete
// one, with n the common degree of N and D, so that on the boundary
// mirror(P) is conj(P) (times z^n).
func (r *rationalLoop) mirror(p Poly) Poly {
	if r.discrete() {
		q := r.pad(p)
		slices.Reverse(q)
		return q
	}
	q := slices.Clone(p)
	for i := len(q) - 2; i >= 0; i -= 2 {
		q[i] = -q[i]
	}
	return q
}

func (r *rationalLoop) pad(p Poly) Poly {
	n := max(len(r.num), len(r.den))
	return append(make(Poly, n-len(p)), p...)
}

// gainCandidates returns approximate ω > 0 where |L| = gamma: boundary roots
// of N·N~ − γ²·D·D~.
func (r *rationalLoop) gainCandidates(gamma float64) []float64 {
	num, den := r.pad(r.num), r.pad(r.den)
	p := num.Mul(r.mirror(r.num)).Sub(den.Mul(r.mirror(r.den)).Scale(gamma * gamma))
	return r.boundaryRoots(p)
}

// realCandidates returns approximate ω > 0 where L is real: boundary roots of
// N·D~ − N~·D.
func (r *rationalLoop) realCandidates() []float64 {
	num, den := r.pad(r.num), r.pad(r.den)
	return r.boundaryRoots(num.Mul(r.mirror(r.den)).Sub(r.mirror(r.num).Mul(den)))
}

// boundaryRoots maps roots of p near the imaginary axis (continuous) or the
// unit circle (discrete) to frequencies in (0, ∞) or (0, π/Dt]. Tangential
// and repeated roots move off the boundary by about eps^(1/multiplicity),
// so the acceptance band is loose; callers verify every candidate.
func (r *rationalLoop) boundaryRoots(p Poly) []float64 {
	roots, err := p.Roots()
	if err != nil {
		return nil
	}
	tol := max(1e-6, math.Pow(eps(), 1/float64(max(1, len(p)-1))))
	var ws []float64
	for _, z := range roots {
		var w float64
		if r.discrete() {
			if math.Abs(cmplx.Abs(z)-1) > tol {
				continue
			}
			w = math.Abs(cmplx.Phase(z)) / r.dt
		} else {
			if math.Abs(real(z)) > tol*cmplx.Abs(z) {
				continue
			}
			w = math.Abs(imag(z))
		}
		if w > 0 && !math.IsInf(w, 0) {
			ws = append(ws, w)
		}
	}
	slices.Sort(ws)
	return slices.Compact(ws)
}

// gainCrossings returns the verified ω > 0 where |L(jω)| = gamma, ascending.
func (r *rationalLoop) gainCrossings(gamma float64) []float64 {
	lg := math.Log(gamma)
	f := func(w float64) float64 { return math.Log(cmplx.Abs(r.eval.at(w))) - lg }
	return r.polish(r.gainCandidates(gamma), f, 1e-9)
}

// phaseCrossings returns the verified ω > 0 where L(jω) is real and negative,
// ascending.
func (r *rationalLoop) phaseCrossings() []float64 {
	var cands []float64
	for _, w := range r.realCandidates() {
		if math.Abs(phaseOffsetDeg(r.eval.at(w))) < 90 {
			cands = append(cands, w)
		}
	}
	return r.polish(cands, func(w float64) float64 { return phaseOffsetDeg(r.eval.at(w)) }, 1e-7)
}

// polish refines each candidate to a sign change of f within half the gap to
// its neighbours, so close crossing pairs stay apart, and keeps those where
// |f| ≤ tol. A candidate without a sign change, a tangency or the discrete
// Nyquist frequency, is kept only if |f| ≤ tol there.
func (r *rationalLoop) polish(cands []float64, f func(float64) float64, tol float64) []float64 {
	wTop := math.Inf(1)
	if r.discrete() {
		wTop = r.nyquist()
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
		best := w
		for d := 1e-12; d <= 1e-2; d *= 100 {
			a, b := w-min(d*w, lo), w+min(d*w, hi)
			if a >= b || !(f(a)*f(b) <= 0) {
				continue
			}
			best = refineCrossing(a, b, f)
			break
		}
		v := f(best)
		if math.Abs(v) <= tol && (len(out) == 0 || best > out[len(out)-1]*(1+1e-12)) {
			out = append(out, best)
		}
	}
	return out
}
