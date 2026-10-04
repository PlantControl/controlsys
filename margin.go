package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

type MarginResult struct {
	GainMargin  float64 // dB; +Inf if no phase crossover
	PhaseMargin float64 // degrees; +Inf if no gain crossover
	WgFreq      float64 // gain crossover freq (0dB); NaN if none
	WpFreq      float64 // phase crossover freq (-180deg); NaN if none
}

type AllMarginResult struct {
	GainMargins     []float64 // dB at each phase crossover
	PhaseMargins    []float64 // degrees at each gain crossover
	GainCrossFreqs  []float64 // omega where |G|=0dB
	PhaseCrossFreqs []float64 // omega where angle(G)=-180deg
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

type crossing struct {
	idx int
	w   float64
}

// sisoEval caches the TF and reuses a single-element buffer for evalInto,
// eliminating repeated TransferFunction calls in refinement loops.
type sisoEval struct {
	sys  *System
	tf   *TransferFunc
	lft  bool
	cont bool
	dt   float64
	tau  float64 // combined InputDelay[0] + OutputDelay[0]
	dst  []complex128
}

func newSISOEval(sys *System) (*sisoEval, error) {
	e := &sisoEval{
		sys:  sys,
		cont: sys.IsContinuous(),
		dt:   sys.Dt,
		dst:  make([]complex128, 1),
	}
	if sys.HasInternalDelay() {
		e.lft = true
	} else {
		res, err := sys.TransferFunction(nil)
		if err != nil {
			return nil, err
		}
		e.tf = res.TF
	}
	if sys.InputDelay != nil {
		e.tau += sys.InputDelay[0]
	}
	if sys.OutputDelay != nil {
		e.tau += sys.OutputDelay[0]
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
	h := e.dst[0]
	if e.tau != 0 {
		if e.cont {
			h *= cmplx.Exp(-s * complex(e.tau, 0))
		} else {
			d := int(math.Round(e.tau))
			for range d {
				h /= s
			}
		}
	}
	return h
}

func marginFreqs(sys *System, nPoints int) ([]float64, error) {
	omega, err := autoBodeFreqs(sys, nPoints)
	if err != nil {
		return nil, err
	}
	if sys.IsDiscrete() && sys.Dt > 0 {
		nyq := math.Pi / sys.Dt
		n := 0
		for _, w := range omega {
			if w < nyq {
				omega[n] = w
				n++
			}
		}
		omega = omega[:n]
	}
	return omega, nil
}

func findCrossings(omega, vals []float64, level float64) []crossing {
	var result []crossing
	for k := 0; k < len(vals)-1; k++ {
		a := vals[k] - level
		b := vals[k+1] - level
		if math.IsNaN(a) || math.IsNaN(b) || math.IsInf(a, 0) || math.IsInf(b, 0) {
			continue
		}
		if a*b < 0 {
			frac := a / (a - b)
			if frac < 0 || frac > 1 {
				frac = 0.5
			}
			logW := math.Log(omega[k]) + frac*(math.Log(omega[k+1])-math.Log(omega[k]))
			result = append(result, crossing{idx: k, w: math.Exp(logW)})
		}
	}
	return result
}

func phaseCrossings(omega, phase []float64, target float64) []crossing {
	shifted := make([]float64, len(phase))
	for k := range phase {
		s := math.Mod(phase[k]-target+180, 360)
		if s < 0 {
			s += 360
		}
		shifted[k] = s - 180
	}
	return findCrossings(omega, shifted, 0)
}

func refineCrossing(wLo, wHi float64, evalFn func(float64) float64) float64 {
	fLo := evalFn(wLo)
	for range 60 {
		wMid := math.Sqrt(wLo * wHi)
		fMid := evalFn(wMid)
		if fLo*fMid <= 0 {
			wHi = wMid
		} else {
			wLo = wMid
			fLo = fMid
		}
		if (wHi-wLo)/wMid < 1e-10 {
			break
		}
	}
	return math.Sqrt(wLo * wHi)
}

func unwrapPhase(phase []float64) {
	for k := 1; k < len(phase); k++ {
		diff := phase[k] - phase[k-1]
		if diff > 180 {
			phase[k] -= 360
		}
		if diff < -180 {
			phase[k] += 360
		}
	}
}

func evalSISOFreqResponse(sys *System, w float64) (complex128, error) {
	resp, err := sys.FreqResponse([]float64{w})
	if err != nil {
		return 0, err
	}
	return resp.At(0, 0, 0), nil
}

func AllMargin(sys *System) (*AllMarginResult, error) {
	if _, err := newSISOLoopModel(sys, "AllMargin"); err != nil {
		return nil, err
	}

	omega, err := marginFreqs(sys, 1000)
	if err != nil {
		return nil, err
	}
	if len(omega) == 0 {
		return &AllMarginResult{}, nil
	}

	eval, err := newSISOEval(sys)
	if err != nil {
		return nil, err
	}

	nw := len(omega)
	magDB := make([]float64, nw)
	phase := make([]float64, nw)
	for k, w := range omega {
		h := eval.at(w)
		magDB[k] = 20 * math.Log10(cmplx.Abs(h))
		phase[k] = cmplx.Phase(h) * 180 / math.Pi
	}
	unwrapPhase(phase)

	gainCross := findCrossings(omega, magDB, 0.0)
	phaseCross := phaseCrossings(omega, phase, -180.0)

	gcFreqs := make([]float64, len(gainCross))
	phaseMargins := make([]float64, len(gainCross))
	for i, c := range gainCross {
		w := refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
			return 20 * math.Log10(cmplx.Abs(eval.at(w)))
		})
		gcFreqs[i] = w

		h := eval.at(w)
		ph := cmplx.Phase(h) * 180 / math.Pi
		bodeRef := phase[c.idx]
		k := math.Round((bodeRef - ph) / 360)
		phaseMargins[i] = 180 + ph + k*360
	}

	pcFreqs := make([]float64, len(phaseCross))
	gainMargins := make([]float64, len(phaseCross))
	for i, c := range phaseCross {
		w := refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
			ph := cmplx.Phase(eval.at(w)) * 180 / math.Pi
			s := math.Mod(ph+180+180, 360)
			if s < 0 {
				s += 360
			}
			return s - 180
		})
		pcFreqs[i] = w
		gainMargins[i] = -20 * math.Log10(cmplx.Abs(eval.at(w)))
	}

	return &AllMarginResult{
		GainMargins:     gainMargins,
		PhaseMargins:    phaseMargins,
		GainCrossFreqs:  gcFreqs,
		PhaseCrossFreqs: pcFreqs,
	}, nil
}

func Margin(sys *System) (*MarginResult, error) {
	all, err := AllMargin(sys)
	if err != nil {
		return nil, err
	}

	result := &MarginResult{
		GainMargin:  math.Inf(1),
		PhaseMargin: math.Inf(1),
		WgFreq:      math.NaN(),
		WpFreq:      math.NaN(),
	}

	for i, gm := range all.GainMargins {
		if gm > 0 && gm < result.GainMargin {
			result.GainMargin = gm
			result.WpFreq = all.PhaseCrossFreqs[i]
		}
	}
	if math.IsInf(result.GainMargin, 1) {
		for i, gm := range all.GainMargins {
			if gm > result.GainMargin || math.IsInf(result.GainMargin, 1) {
				result.GainMargin = gm
				result.WpFreq = all.PhaseCrossFreqs[i]
			}
		}
	}

	for i, pm := range all.PhaseMargins {
		if pm > 0 && pm < result.PhaseMargin {
			result.PhaseMargin = pm
			result.WgFreq = all.GainCrossFreqs[i]
		}
	}
	if math.IsInf(result.PhaseMargin, 1) {
		for i, pm := range all.PhaseMargins {
			if pm > result.PhaseMargin || math.IsInf(result.PhaseMargin, 1) {
				result.PhaseMargin = pm
				result.WgFreq = all.GainCrossFreqs[i]
			}
		}
	}

	return result, nil
}

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

	threshold := 20*math.Log10(dcMag) + dbDrop

	omega, err := marginFreqs(sys, 1000)
	if err != nil {
		return 0, err
	}
	if len(omega) == 0 {
		return math.Inf(1), nil
	}

	siso := p == 1 && m == 1
	nw := len(omega)
	magDB := make([]float64, nw)

	var eval *sisoEval
	if siso {
		eval, err = newSISOEval(sys)
		if err != nil {
			return 0, err
		}
		for k, w := range omega {
			mag := cmplx.Abs(eval.at(w))
			if mag > 0 {
				magDB[k] = 20 * math.Log10(mag)
			} else {
				magDB[k] = -1000
			}
		}
	} else {
		sigma, err := sys.Sigma(omega, 0)
		if err != nil {
			return 0, err
		}
		for k := range nw {
			sv := sigma.At(k, 0)
			if sv > 0 {
				magDB[k] = 20 * math.Log10(sv)
			} else {
				magDB[k] = -1000
			}
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
			mag := cmplx.Abs(eval.at(w))
			if mag > 0 {
				return 20*math.Log10(mag) - threshold
			}
			return -1000 - threshold
		}
		sig, err := sys.Sigma([]float64{w}, 0)
		if err != nil {
			return 0
		}
		sv := sig.At(0, 0)
		if sv > 0 {
			return 20*math.Log10(sv) - threshold
		}
		return -1000 - threshold
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
