package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"strings"
)

// PidtuneOptions holds the optional settings of Pidtune, as MATLAB
// pidtuneOptions; nil or zero fields select the defaults.
type PidtuneOptions struct {
	// PhaseMargin is the target phase margin in degrees, in (0, 180). 0
	// means a best-effort 60°; an explicit margin the controller family
	// cannot reach at wc returns ErrPIDTuningTargetUnattainable.
	PhaseMargin float64
}

// PidtuneType names a PID controller family, as the MATLAB pidtune type
// argument; case is ignored.
type PidtuneType string

const (
	PidtuneP    PidtuneType = "P"
	PidtuneI    PidtuneType = "I"
	PidtunePI   PidtuneType = "PI"
	PidtunePD   PidtuneType = "PD"
	PidtunePDF  PidtuneType = "PDF"
	PidtunePID  PidtuneType = "PID"
	PidtunePIDF PidtuneType = "PIDF"
)

// Pidtune tunes a PID of family pidType for the SISO plant at crossover wc,
// as MATLAB pidtune(sys,type,wc,opts)
// (https://www.mathworks.com/help/control/ref/dynamicsystem.pidtune.html).
// wc = 0 picks the crossover from the plant, capped at (180°−PM)/(2τ) for a
// loop delay τ so the delay lag leaves room for the margin; a negative or
// non-finite wc, or a phase margin outside (0, 180), returns
// ErrInvalidArgument. The loop crossover and phase margin are placed using
// the realized controller response: discrete plants use the discrete PID
// terms, a discrete PID without filter uses a backward-Euler derivative to
// stay causal, and a discrete derivative whose forward-Euler filter pole
// 1−Dt/Tf would leave the unit circle (Tf ≤ Dt/2) uses backward Euler. An
// explicitly requested phase margin the family cannot reach at wc returns
// ErrPIDTuningTargetUnattainable instead of a controller with a different
// margin. Every gain takes one sign: that of the DC gain for a stable plant,
// and for an integrating or unstable plant the sign stabilization forces
// (see pidtuneGainSign), so Pidtune(−G) mirrors Pidtune(G). A design whose
// closed loop is unstable by its poles (delay absorbed for discrete plants)
// or by a Nyquist test (continuous delays) is rejected; loops outside both
// tests are not screened. With wc = 0 a rejected design is retried at
// crossovers lowered up to three decades, accepting the first that passes
// and keeps the phase margin at every gain crossover; otherwise, and for an
// explicit wc, the result is ErrPIDTuningTargetUnattainable. TunePID offers
// bounded search with achieved-target evidence.
func Pidtune(plant *System, pidType PidtuneType, wc float64, opts *PidtuneOptions) (*PID, error) {
	if err := requireFiniteSystem("Pidtune", plant); err != nil {
		return nil, err
	}
	if _, err := newSISOLoopModel(plant, "Pidtune"); err != nil {
		return nil, err
	}
	if !finitePID(wc) || wc < 0 {
		return nil, fmt.Errorf("Pidtune: wc %g must be finite and nonnegative (0 = automatic): %w", wc, ErrInvalidArgument)
	}
	pm, explicit := 60.0, false
	if opts != nil && opts.PhaseMargin != 0 {
		pm, explicit = opts.PhaseMargin, true
	}
	if !finitePID(pm) || pm <= 0 || pm >= 180 {
		return nil, fmt.Errorf("Pidtune: phase margin %g must be in (0, 180) degrees: %w", pm, ErrInvalidArgument)
	}

	pidType = PidtuneType(strings.ToUpper(string(pidType)))
	switch pidType {
	case PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF:
	default:
		return nil, fmt.Errorf("Pidtune: unsupported type %q: %w", pidType, ErrInvalidArgument)
	}

	auto := wc == 0
	wc, err := findCrossoverFreq(plant, wc, pm)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}

	eval, err := newSISOEval(plant)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}
	_, hasI, _, _, _ := pidTuningTerms(pidType)
	sign := pidtuneGainSign(plant, eval.at, wc, hasI)
	if eval.err != nil {
		return nil, fmt.Errorf("Pidtune: %w", eval.err)
	}
	screen, err := newPIDStabilityScreen(plant, eval, 1)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}

	steps := 0
	if auto {
		steps = pidtuneBackoffSteps
	}
	var first error
	var grid *pidtuneLoopGrid
	for k := 0; k <= steps; k++ {
		w := wc * math.Pow(10, -float64(k)/pidtuneBackoffPerDecade)
		pid, err := pidtuneDesign(plant, screen, eval, pidType, w, pm, sign, explicit)
		if err == nil && k > 0 {
			if grid == nil {
				if grid, err = newPidtuneLoopGrid(plant, wc); err != nil {
					return nil, fmt.Errorf("Pidtune: %w", err)
				}
			}
			if m := grid.phaseMargin(pid); m < pm-pidtuneMarginTol {
				err = fmt.Errorf("%s design at wc=%g has phase margin %.3g° at another crossover: %w", pidType, w, m, ErrPIDTuningTargetUnattainable)
			}
		}
		if err == nil {
			return pid, nil
		}
		if first == nil {
			first = err
		}
		if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			break
		}
	}
	return nil, fmt.Errorf("Pidtune: %w", first)
}

// An automatic crossover whose design fails is lowered geometrically, up to
// pidtuneBackoffSteps/pidtuneBackoffPerDecade decades, until the design is
// stable and keeps the phase margin at every gain crossover of the loop.
const (
	pidtuneBackoffPerDecade = 20
	pidtuneBackoffSteps     = 60
)

func pidtuneDesign(plant *System, screen *pidStabilityScreen, eval *sisoEval, pidType PidtuneType, wc, pm, sign float64, explicit bool) (*PID, error) {
	pid, err := computePIDGains(plant, pidType, wc, pm, sign)
	if err != nil {
		return nil, err
	}
	pid.Dt = plant.Dt
	if explicit {
		if err := checkPidtuneMargin(plant, pid, wc, pm); err != nil {
			return nil, err
		}
	}
	if err := checkPidtuneStability(screen, eval, pid, pidType, wc); err != nil {
		return nil, err
	}
	return pid, nil
}

// pidtuneGainSign returns the sign of every Pidtune gain: that of the DC
// gain of a stable plant, else of N(0)/a for the minimal G = N/D with D's
// leading coefficient a (N(1)/a in z) when it is forced, else of the
// high-frequency gain. A stable monic closed-loop polynomial is positive at
// s = 0 (z = 1), where it is D(0)/a + K·N(0)/a for P/PD and K·N(0)/a under
// integral action, so K·N(0)/a > 0 is forced when D(0)/a ≤ 0 (an integrator
// or an odd count of unstable real poles) or integral is set. That is the
// DC gain sign of G with integrators removed and unstable poles reflected.
// It returns 1 when no rule applies.
func pidtuneGainSign(plant *System, at func(float64) complex128, wc float64, integral bool) float64 {
	if s := pidTuningDCSign(plant, at, wc); s != 0 {
		return s
	}
	if stable, err := plant.IsStable(); err != nil || stable {
		return 1
	}
	res, err := plant.rationalTransferFunction(nil)
	if err != nil {
		return 1
	}
	num, den := res.TF.Num[0][0], res.TF.Den[0]
	if len(num) == 0 || len(den) == 0 || den[0] == 0 {
		return 1
	}
	x := 1.0
	if plant.Dt == 0 {
		x = wc
	}
	// atOrigin returns p(0) (p(1) in z) and whether it is nonzero relative
	// to the terms of p at |s| = wc (|z| = 1).
	atOrigin := func(p []float64) (float64, bool) {
		v, scale := 0.0, 0.0
		for i, c := range p {
			if plant.Dt > 0 {
				v += c
			} else if i == len(p)-1 {
				v = c
			}
			scale += math.Abs(c) * math.Pow(x, float64(len(p)-1-i))
		}
		return v, finitePID(v) && finitePID(scale) && math.Abs(v) > 1e3*0x1p-52*scale
	}
	n0, nOK := atOrigin(num)
	d0, dOK := atOrigin(den)
	a := math.Copysign(1, den[0])
	if integral || !dOK || d0*a < 0 {
		if nOK && n0*a < 0 {
			return -1
		}
		return 1
	}
	scale := 0.0
	for i, c := range num {
		scale = math.Max(scale, math.Abs(c)*math.Pow(x, float64(len(num)-1-i)))
	}
	for i, c := range num {
		if math.Abs(c)*math.Pow(x, float64(len(num)-1-i)) > 1e3*0x1p-52*scale {
			return math.Copysign(1, c) * a
		}
	}
	return 1
}

// pidtuneLoopGrid holds the plant response on a log grid spanning the
// back-off range, to measure the phase margin at every gain crossover.
type pidtuneLoopGrid struct {
	w  []float64
	g  []complex128
	dt float64
}

const pidtuneLoopGridPerDecade = 100

func newPidtuneLoopGrid(plant *System, wc float64) (*pidtuneLoopGrid, error) {
	eval, err := newSISOEval(plant)
	if err != nil {
		return nil, err
	}
	lo, hi := wc*1e-6, wc*1e3
	if plant.Dt > 0 {
		hi = math.Min(hi, math.Pi/plant.Dt)
	}
	n := int(math.Ceil(math.Log10(hi/lo)*pidtuneLoopGridPerDecade)) + 1
	grid := &pidtuneLoopGrid{w: make([]float64, 0, n), g: make([]complex128, 0, n), dt: plant.Dt}
	for i := range n {
		w := math.Min(lo*math.Pow(10, float64(i)/pidtuneLoopGridPerDecade), hi)
		eval.err = nil
		if g := eval.at(w); eval.err == nil && pidFiniteComplex(g) {
			grid.w = append(grid.w, w)
			grid.g = append(grid.g, g)
		}
	}
	return grid, nil
}

// phaseMargin returns the smallest phase margin over the gain crossovers of
// pid·G found on the grid, or +Inf when |L| does not cross 1 there.
func (grid *pidtuneLoopGrid) phaseMargin(pid *PID) float64 {
	pm := math.Inf(1)
	var prev complex128
	for i, w := range grid.w {
		l := pidtuneControllerAt(pid, w, grid.dt) * grid.g[i]
		if i > 0 {
			a, b := math.Log(cmplx.Abs(prev)), math.Log(cmplx.Abs(l))
			if a*b <= 0 && a != b {
				x := prev + complex(a/(a-b), 0)*(l-prev)
				pm = math.Min(pm, 180-math.Abs(cmplx.Phase(x))*180/math.Pi)
			}
		}
		prev = l
	}
	return pm
}

func pidtuneControllerAt(pid *PID, w, dt float64) complex128 {
	terms := pidtuneTerms{wc: w, dt: dt, dFormula: pid.DFormula}
	c := complex(pid.Kp, 0) + complex(pid.Ki, 0)*terms.integral()
	if pid.Kd != 0 {
		c += complex(pid.Kd, 0) * terms.derivative(pid.Tf)
	}
	return c
}

// checkPidtuneStability rejects a design whose closed loop is certifiably
// unstable; loops outside the screen's scope pass unchecked.
func checkPidtuneStability(screen *pidStabilityScreen, eval *sisoEval, pid *PID, pidType PidtuneType, wc float64) error {
	stable, _, err := screen.at(wc, pidType)
	if errors.Is(err, ErrContinuousInternalDelay) {
		return nil
	}
	if err != nil || stable == nil {
		return err
	}
	ok, err := stable(&PID2{Kp: pid.Kp, Ki: pid.Ki, Kd: pid.Kd, Tf: pid.Tf, Dt: pid.Dt, IFormula: pid.IFormula, DFormula: pid.DFormula})
	if err == nil {
		err = eval.err
	}
	if err != nil {
		return err
	}
	if !ok {
		return fmt.Errorf("%s design at wc=%g gives an unstable closed loop: %w", pidType, wc, ErrPIDTuningTargetUnattainable)
	}
	return nil
}

const pidtuneMarginTol = 0.5

// checkPidtuneMargin rejects a design whose phase margin at wc differs from
// the requested one, which happens when the family's phase range had to be
// clamped.
func checkPidtuneMargin(plant *System, pid *PID, wc, pm float64) error {
	h, err := evalPlantAt(plant, wc)
	if err != nil {
		return err
	}
	achieved := 180 + cmplx.Phase(h*pidtuneControllerAt(pid, wc, plant.Dt))*180/math.Pi
	if achieved > 180 {
		achieved -= 360
	}
	if math.Abs(achieved-pm) > pidtuneMarginTol {
		return fmt.Errorf("phase margin %g° unreachable by this family at wc=%g (achievable %.3g°): %w", pm, wc, achieved, ErrPIDTuningTargetUnattainable)
	}
	return nil
}

// findCrossoverFreq returns wcTarget, or when it is 0 a crossover from the
// plant magnitude capped at (180°−pm)/(2τ) for a total loop delay τ, so the
// delay lag spends at most half the phase left after the margin.
func findCrossoverFreq(plant *System, wcTarget, pm float64) (float64, error) {
	if wcTarget > 0 {
		return wcTarget, nil
	}
	wc, err := plantCrossoverFreq(plant)
	if err != nil {
		return 0, err
	}
	tau := sisoLoopDelay(plant)
	if plant.Dt > 0 {
		tau *= plant.Dt
	}
	if tau > 0 {
		wc = math.Min(wc, (180-pm)*math.Pi/180/(2*tau))
	}
	return wc, nil
}

func plantCrossoverFreq(plant *System) (float64, error) {

	all, err := AllMargin(plant)
	if err != nil {
		return 0, err
	}
	if len(all.GainCrossFreqs) > 0 {
		return all.GainCrossFreqs[0], nil
	}

	bw, err := Bandwidth(plant, -3)
	if err != nil {
		return 0, err
	}
	if bw > 0 && !math.IsInf(bw, 0) {
		return bw, nil
	}

	poles, err := plant.Poles()
	if err != nil {
		return 0, err
	}
	if len(poles) > 0 {
		minW := math.Inf(1)
		for _, pole := range poles {
			w := cmplx.Abs(pole)
			if w > 0 && w < minW {
				minW = w
			}
		}
		if !math.IsInf(minW, 1) {
			return minW * 0.5, nil
		}
	}

	return 0, fmt.Errorf("no gain crossover, bandwidth or nonzero pole to infer a crossover from; pass wc > 0: %w", ErrInvalidArgument)
}

func evalPlantAt(plant *System, w float64) (complex128, error) {
	resp, err := plant.FreqResponse([]float64{w})
	if err != nil {
		return 0, err
	}
	return resp.At(0, 0, 0), nil
}

// pidtuneTerms evaluates the realized controller terms at the crossover so
// discrete designs match pidDiscrete rather than the continuous prototype.
type pidtuneTerms struct {
	wc, dt   float64
	dFormula PIDFormula
}

func pidtuneIntegrator(f PIDFormula, w, dt float64) complex128 {
	if dt == 0 {
		return complex(0, -1/w)
	}
	z1 := newTimeDomain(dt).frequencyPoint(w).shift(1)
	return complex(dt, 0) * (1/z1 + complex(pidFormulaWeight(f), 0))
}

func (t pidtuneTerms) integral() complex128 { return pidtuneIntegrator(ForwardEuler, t.wc, t.dt) }

func (t pidtuneTerms) derivative(tf float64) complex128 {
	return 1 / (complex(tf, 0) + pidtuneIntegrator(t.dFormula, t.wc, t.dt))
}

// computePIDGains designs for sign·G and returns the gains scaled by sign,
// so a negative-DC plant gets the mirror of its positive twin's design.
func computePIDGains(plant *System, pidType PidtuneType, wc, pmDeg, sign float64) (*PID, error) {
	pid, err := computePIDGainsPositive(plant, pidType, wc, pmDeg, sign)
	if err != nil {
		return nil, err
	}
	pid.Kp *= sign
	pid.Ki *= sign
	pid.Kd *= sign
	return pid, nil
}

func computePIDGainsPositive(plant *System, pidType PidtuneType, wc, pmDeg, sign float64) (*PID, error) {
	if plant.Dt > 0 && wc >= math.Pi/plant.Dt {
		return nil, fmt.Errorf("crossover %g must be below the Nyquist frequency %g: %w", wc, math.Pi/plant.Dt, ErrInvalidArgument)
	}
	h, err := evalPlantAt(plant, wc)
	if err != nil {
		return nil, err
	}
	h *= complex(sign, 0)
	magP := cmplx.Abs(h)
	phaseP := cmplx.Phase(h) * 180 / math.Pi

	if magP == 0 || math.IsNaN(magP) || math.IsInf(magP, 0) {
		return nil, fmt.Errorf("plant has zero or infinite gain at wc=%g: %w", wc, ErrPIDTuningTargetUnattainable)
	}

	phiC := -180 + pmDeg - phaseP

	for phiC > 180 {
		phiC -= 360
	}
	for phiC < -180 {
		phiC += 360
	}

	dFormula := ForwardEuler
	if plant.Dt > 0 && pidType == PidtunePID {
		dFormula = BackwardEuler
	}
	pid, err := pidtuneFamily(pidType, pidtuneTerms{wc: wc, dt: plant.Dt, dFormula: dFormula}, magP, phiC)
	if err == nil && plant.Dt > 0 && pid.DFormula == ForwardEuler && pid.Kd != 0 && pid.Tf <= plant.Dt/2 {
		// The forward-Euler filter pole 1−Dt/Tf is outside the unit circle.
		pid, err = pidtuneFamily(pidType, pidtuneTerms{wc: wc, dt: plant.Dt, dFormula: BackwardEuler}, magP, phiC)
	}
	return pid, err
}

// pidtuneFamily designs pidType at terms.wc with the derivative formula of terms.
func pidtuneFamily(pidType PidtuneType, terms pidtuneTerms, magP, phiC float64) (*PID, error) {
	pid := &PID{DFormula: terms.dFormula}
	switch pidType {
	case PidtuneP:
		pid.Kp = 1.0 / magP

	case PidtuneI:
		pid.Ki = 1 / (magP * cmplx.Abs(terms.integral()))

	case PidtunePI:
		computePI(pid, terms, magP, phiC)

	case PidtunePD:
		computePD(pid, terms, magP, phiC)

	case PidtunePDF:
		pid.Tf = .1 / terms.wc
		target := cmplx.Rect(1/magP, phiC*math.Pi/180)
		d := terms.derivative(pid.Tf)
		pid.Kd = imag(target) / imag(d)
		pid.Kp = real(target) - pid.Kd*real(d)
		if pid.Kp < 0 || pid.Kd < 0 {
			return nil, fmt.Errorf("requested phase is unattainable by a same-sign PDF at wc=%g: %w", terms.wc, ErrPIDTuningTargetUnattainable)
		}
	case PidtunePID:
		computePID(pid, terms, magP, phiC)

	case PidtunePIDF:
		computePIDF(pid, terms, magP, phiC)
	}

	return pid, nil
}

func computePI(pid *PID, terms pidtuneTerms, magP, phiC float64) {
	i := terms.integral()
	phiCRad := phiC * math.Pi / 180

	// PI phase range: (arg I, 0)
	if phiCRad >= 0 {
		phiCRad = -0.05
	}
	if lower := cmplx.Phase(i); phiCRad <= lower {
		phiCRad = lower + 0.05
	}

	target := cmplx.Rect(1/magP, phiCRad)
	pid.Ki = imag(target) / imag(i)
	pid.Kp = real(target) - pid.Ki*real(i)
}

// computePD preserves the legacy PD tuning seed, including its Td/10 filter.
// Callers requesting an ideal PD must explicitly set Tf=0.
func computePD(pid *PID, terms pidtuneTerms, magP, phiC float64) {
	// C = Kp*(1 + Td*D), angle = arg(1 + Td*D)
	phiCRad := phiC * math.Pi / 180

	// PD phase range: (0, 90)
	if phiCRad <= 0 {
		phiCRad = 0.05
	}
	if phiCRad >= math.Pi/2 {
		phiCRad = math.Pi/2 - 0.05
	}

	d := terms.derivative(0)
	tan := math.Tan(phiCRad)
	Td := tan / (imag(d) - real(d)*tan)
	if Td <= 0 {
		Td = 0.1 / terms.wc
	}

	pid.Kp = 1.0 / (magP * cmplx.Abs(1+complex(Td, 0)*d))
	pid.Kd = pid.Kp * Td
	pid.Tf = Td / 10
	if pid.Tf <= 0 {
		pid.Tf = 1.0 / (10 * terms.wc)
	}
}

// pidFromPhase solves Kp + Ki*I + Kd*D = target with Ki = Kp*wc/b, relaxing
// the integral ratio and finally dropping Kd when the derivative turns negative.
func pidFromPhase(terms pidtuneTerms, magP, phiCRad, b float64) (Kp, Ki, Kd float64) {
	target := cmplx.Rect(1/magP, phiCRad)
	i, d := terms.integral(), terms.derivative(0)
	solve := func(ratio float64) (kp, kd float64) {
		a11, a12 := 1+ratio*real(i), real(d)
		a21, a22 := ratio*imag(i), imag(d)
		det := a11*a22 - a12*a21
		kp = (real(target)*a22 - a12*imag(target)) / det
		kd = (a11*imag(target) - a21*real(target)) / det
		return kp, kd
	}

	ratio := terms.wc / b
	Kp, Kd = solve(ratio)
	if Kd < 0 {
		ratio = terms.wc / (b * 2)
		Kp, Kd = solve(ratio)
	}
	if Kd >= 0 {
		return Kp, Kp * ratio, Kd
	}
	Kd = 0
	Ki = imag(target) / imag(i)
	Kp = real(target) - Ki*real(i)
	if Ki < 0 {
		ratio = terms.wc / (b * 4)
		Kp = real(target) / (1 + ratio*real(i))
		Ki = Kp * ratio
	}
	return Kp, Ki, Kd
}

func computePID(pid *PID, terms pidtuneTerms, magP, phiC float64) {
	phiCRad := phiC * math.Pi / 180
	if phiCRad >= math.Pi/2 {
		phiCRad = math.Pi/2 - 0.05
	}
	if phiCRad <= -math.Pi/2 {
		phiCRad = -math.Pi/2 + 0.05
	}

	pid.Kp, pid.Ki, pid.Kd = pidFromPhase(terms, magP, phiCRad, 5.0)
}

func computePIDF(pid *PID, terms pidtuneTerms, magP, phiC float64) {
	// Iteratively design PID then compensate for filter (Tf=Td/10) phase loss.
	wc := terms.wc
	phiCRad := phiC * math.Pi / 180
	if phiCRad >= math.Pi/2 {
		phiCRad = math.Pi/2 - 0.05
	}
	if phiCRad <= -math.Pi/2 {
		phiCRad = -math.Pi/2 + 0.05
	}

	i := terms.integral()
	phiAdj := phiCRad
	var Kp, Ki, Kd float64
	for range 100 {
		Kp, Ki, Kd = pidFromPhase(terms, magP, phiAdj, 5.0)
		Td := 0.0
		if Kp > 0 {
			Td = Kd / Kp
		}
		Tf := Td / 10
		if Tf <= 0 {
			Tf = 1.0 / (10 * wc)
		}

		C := complex(Kp, 0) + complex(Ki, 0)*i + complex(Kd, 0)*terms.derivative(Tf)
		cMag := cmplx.Abs(C)
		if cMag == 0 {
			break
		}
		scale := 1.0 / (cMag * magP)
		Kp *= scale
		Ki *= scale
		Kd *= scale

		pErr := phiCRad - cmplx.Phase(C)
		if math.Abs(pErr) < 1e-12 {
			break
		}
		phiAdj += pErr
		if phiAdj >= math.Pi/2 {
			phiAdj = math.Pi/2 - 0.05
		}
		if phiAdj <= -math.Pi/2 {
			phiAdj = -math.Pi/2 + 0.05
		}
	}

	pid.Kp = Kp
	pid.Ki = Ki
	pid.Kd = Kd
	Td := 0.0
	if Kp > 0 {
		Td = Kd / Kp
	}
	pid.Tf = Td / 10
	if pid.Tf <= 0 {
		pid.Tf = 1.0 / (10 * wc)
	}
}
