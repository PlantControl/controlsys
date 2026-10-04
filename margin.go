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

type DiskMarginResult struct {
	Alpha           float64    // disk margin = 1/Ms
	GainMargin      [2]float64 // [low, high] linear gain factors
	GainMarginDB    [2]float64 // [low, high] in dB
	PhaseMargin     float64    // +/- degrees
	PeakSensitivity float64    // Ms = ||S||_inf
	PeakFreq        float64    // omega where sensitivity peaks
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

const (
	marginMaxRefineDepth   = 10
	marginMaxDelayPoints   = 20000
	marginMaxExtendDecades = 10
)

func marginRange(sys *System) (wMin, wMax float64, err error) {
	poles, err := sys.Poles()
	if err != nil {
		return 0, 0, err
	}
	td := newTimeDomain(sys.Dt)
	lo, hi := math.Inf(1), 0.0
	for _, p := range poles {
		wn := td.naturalFrequency(p)
		if wn > 0 && !math.IsInf(wn, 0) {
			lo = min(lo, wn)
			hi = max(hi, wn)
		}
	}
	wMin, wMax = 0.01, 100.0
	if hi > 0 {
		wMin, wMax = lo/10, hi*10
	}
	if sys.IsDiscrete() && sys.Dt > 0 {
		wMax = math.Pi / sys.Dt
		wMin = min(wMin, wMax/100)
	}
	return wMin, wMax, nil
}

func marginFreqs(sys *System, nPoints int) ([]float64, error) {
	wMin, wMax, err := marginRange(sys)
	if err != nil {
		return nil, err
	}
	omega := logspace(math.Log10(wMin), math.Log10(wMax), nPoints)
	omega[len(omega)-1] = wMax
	return omega, nil
}

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
	wMin, wMax, err := marginRange(sys)
	if err != nil {
		return nil, nil, err
	}
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

func wrapDegrees(x float64) float64 {
	x = math.Mod(x+180, 360)
	if x < 0 {
		x += 360
	}
	return x - 180
}

func phaseOffsetDeg(h complex128) float64 {
	return wrapDegrees(cmplx.Phase(h)*180/math.Pi + 180)
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
			result = append(result, interpCrossing(omega, k, a, b))
		}
	}
	return result
}

func interpCrossing(omega []float64, k int, a, b float64) crossing {
	frac := a / (a - b)
	if frac < 0 || frac > 1 {
		frac = 0.5
	}
	logW := math.Log(omega[k]) + frac*(math.Log(omega[k+1])-math.Log(omega[k]))
	return crossing{idx: k, w: math.Exp(logW)}
}

// phaseCrossings finds where phase passes target (mod 360). A sign change of
// the wrapped offset with a jump near 360 is the opposite wrap, not a crossing.
func phaseCrossings(omega, phase []float64, target float64) []crossing {
	var result []crossing
	for k := 0; k < len(phase)-1; k++ {
		a := wrapDegrees(phase[k] - target)
		b := wrapDegrees(phase[k+1] - target)
		if a*b < 0 && math.Abs(a-b) < 180 {
			result = append(result, interpCrossing(omega, k, a, b))
		}
	}
	return result
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

func AllMargin(sys *System) (*AllMarginResult, error) {
	if _, err := newSISOLoopModel(sys, "AllMargin"); err != nil {
		return nil, err
	}

	eval, err := newSISOEval(sys)
	if err != nil {
		return nil, err
	}
	omega, resp, err := marginGrid(sys, eval)
	if err != nil {
		return nil, err
	}

	nw := len(omega)
	magDB := make([]float64, nw)
	phase := make([]float64, nw)
	for k, h := range resp {
		magDB[k] = 20 * math.Log10(cmplx.Abs(h))
		phase[k] = cmplx.Phase(h) * 180 / math.Pi
	}

	res := &AllMarginResult{}
	for _, c := range findCrossings(omega, magDB, 0.0) {
		w := refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
			return 20 * math.Log10(cmplx.Abs(eval.at(w)))
		})
		h := eval.at(w)
		if math.Abs(20*math.Log10(cmplx.Abs(h))) > 1e-2 {
			continue
		}
		res.GainCrossFreqs = append(res.GainCrossFreqs, w)
		res.PhaseMargins = append(res.PhaseMargins, phaseMarginDeg(h))
	}
	for _, c := range phaseCrossings(omega, phase, -180.0) {
		w := refineCrossing(omega[c.idx], omega[c.idx+1], func(w float64) float64 {
			return phaseOffsetDeg(eval.at(w))
		})
		h := eval.at(w)
		if math.Abs(phaseOffsetDeg(h)) > 1 {
			continue
		}
		res.PhaseCrossFreqs = append(res.PhaseCrossFreqs, w)
		res.GainMargins = append(res.GainMargins, -20*math.Log10(cmplx.Abs(h)))
	}

	if sys.IsDiscrete() && sys.Dt > 0 {
		nyq, h := omega[nw-1], resp[nw-1]
		near := func(ws []float64) bool { return len(ws) > 0 && ws[len(ws)-1] > nyq*(1-1e-8) }
		if math.Abs(magDB[nw-1]) < 1e-9 && !near(res.GainCrossFreqs) {
			res.GainCrossFreqs = append(res.GainCrossFreqs, nyq)
			res.PhaseMargins = append(res.PhaseMargins, phaseMarginDeg(h))
		}
		if math.Abs(phaseOffsetDeg(h)) < 1e-6 && !near(res.PhaseCrossFreqs) {
			res.PhaseCrossFreqs = append(res.PhaseCrossFreqs, nyq)
			res.GainMargins = append(res.GainMargins, -20*math.Log10(cmplx.Abs(h)))
		}
	}
	return res, nil
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

func DiskMargin(sys *System) (*DiskMarginResult, error) {
	if _, err := newSISOLoopModel(sys, "DiskMargin"); err != nil {
		return nil, err
	}

	eye, err := NewGain(mat.NewDense(1, 1, []float64{1}), sys.Dt)
	if err != nil {
		return nil, err
	}

	S, err := Feedback(eye, sys, -1)
	if err != nil {
		return nil, fmt.Errorf("DiskMargin: cannot form sensitivity: %w", err)
	}

	Ms, wPeak, err := HinfNorm(S)
	if err != nil {
		if errors.Is(err, ErrUnstable) {
			return &DiskMarginResult{
				Alpha:           0,
				GainMargin:      [2]float64{1, 1},
				GainMarginDB:    [2]float64{0, 0},
				PhaseMargin:     0,
				PeakSensitivity: math.Inf(1),
				PeakFreq:        0,
			}, nil
		}
		return nil, err
	}

	if Ms <= 0 {
		return &DiskMarginResult{
			Alpha:           1,
			GainMargin:      [2]float64{0.5, math.Inf(1)},
			GainMarginDB:    [2]float64{-20 * math.Log10(2), math.Inf(1)},
			PhaseMargin:     180,
			PeakSensitivity: Ms,
			PeakFreq:        wPeak,
		}, nil
	}

	alpha := 1.0 / Ms

	gmLow := 1.0 / (1.0 + alpha)
	var gmHigh float64
	if alpha >= 1 {
		gmHigh = math.Inf(1)
	} else {
		gmHigh = 1.0 / (1.0 - alpha)
	}

	var pm float64
	if alpha >= 2 {
		pm = 180
	} else {
		pm = 2 * math.Asin(alpha/2) * 180 / math.Pi
	}

	return &DiskMarginResult{
		Alpha:           alpha,
		GainMargin:      [2]float64{gmLow, gmHigh},
		GainMarginDB:    [2]float64{20 * math.Log10(gmLow), 20 * math.Log10(gmHigh)},
		PhaseMargin:     pm,
		PeakSensitivity: Ms,
		PeakFreq:        wPeak,
	}, nil
}
