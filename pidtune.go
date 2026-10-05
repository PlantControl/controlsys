package controlsys

import (
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
// wc = 0 picks the crossover from the plant; a negative or non-finite wc, or
// a phase margin outside (0, 180), returns ErrInvalidArgument. The loop
// crossover and phase margin are placed using the realized controller
// response: discrete plants use the discrete PID terms, and a discrete PID
// without filter uses a backward-Euler derivative to stay causal. An
// explicitly requested phase margin the family cannot reach at wc returns
// ErrPIDTuningTargetUnattainable instead of a controller with a different
// margin. TunePID offers bounded search with achieved-target evidence.
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

	wc, err := findCrossoverFreq(plant, wc)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}

	pid, err := computePIDGains(plant, pidType, wc, pm)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}
	pid.Dt = plant.Dt
	if explicit {
		if err := checkPidtuneMargin(plant, pid, wc, pm); err != nil {
			return nil, fmt.Errorf("Pidtune: %w", err)
		}
	}
	return pid, nil
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
	terms := pidtuneTerms{wc: wc, dt: plant.Dt, dFormula: pid.DFormula}
	c := complex(pid.Kp, 0) + complex(pid.Ki, 0)*terms.integral()
	if pid.Kd != 0 {
		c += complex(pid.Kd, 0) * terms.derivative(pid.Tf)
	}
	achieved := 180 + cmplx.Phase(h*c)*180/math.Pi
	if achieved > 180 {
		achieved -= 360
	}
	if math.Abs(achieved-pm) > pidtuneMarginTol {
		return fmt.Errorf("phase margin %g° unreachable by this family at wc=%g (achievable %.3g°): %w", pm, wc, achieved, ErrPIDTuningTargetUnattainable)
	}
	return nil
}

func findCrossoverFreq(plant *System, wcTarget float64) (float64, error) {
	if wcTarget > 0 {
		return wcTarget, nil
	}

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
	z := cmplx.Exp(complex(0, w*dt))
	return complex(dt, 0) * (1/(z-1) + complex(pidFormulaWeight(f), 0))
}

func (t pidtuneTerms) integral() complex128 { return pidtuneIntegrator(ForwardEuler, t.wc, t.dt) }

func (t pidtuneTerms) derivative(tf float64) complex128 {
	return 1 / (complex(tf, 0) + pidtuneIntegrator(t.dFormula, t.wc, t.dt))
}

func computePIDGains(plant *System, pidType PidtuneType, wc, pmDeg float64) (*PID, error) {
	if plant.Dt > 0 && wc >= math.Pi/plant.Dt {
		return nil, fmt.Errorf("crossover %g must be below the Nyquist frequency %g: %w", wc, math.Pi/plant.Dt, ErrInvalidArgument)
	}
	h, err := evalPlantAt(plant, wc)
	if err != nil {
		return nil, err
	}
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

	pid := &PID{}
	if plant.Dt > 0 && pidType == PidtunePID {
		pid.DFormula = BackwardEuler
	}
	terms := pidtuneTerms{wc: wc, dt: plant.Dt, dFormula: pid.DFormula}

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
		pid.Tf = .1 / wc
		target := cmplx.Rect(1/magP, phiC*math.Pi/180)
		d := terms.derivative(pid.Tf)
		pid.Kd = imag(target) / imag(d)
		pid.Kp = real(target) - pid.Kd*real(d)
		if pid.Kp < 0 || pid.Kd < 0 {
			return nil, fmt.Errorf("requested phase is unattainable by a positive-gain PDF at wc=%g: %w", wc, ErrPIDTuningTargetUnattainable)
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
