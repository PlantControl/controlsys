package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"strings"
	"sync"

	"plantcontrol.org/v1/gonum/mat"
)

// PIDDesignFocus weights reference tracking against disturbance rejection,
// as MATLAB pidtuneOptions DesignFocus.
type PIDDesignFocus string

const (
	PIDFocusBalanced  PIDDesignFocus = "balanced"
	PIDFocusTracking  PIDDesignFocus = "tracking"
	PIDFocusRejection PIDDesignFocus = "rejection"
)

// PIDStabilityEvidence names how a TunePID design's closed-loop stability was
// checked.
type PIDStabilityEvidence string

const (
	// PIDStabilityClosedLoopPoles checks the closed-loop poles of a
	// delay-free (or delay-absorbed discrete) plant.
	PIDStabilityClosedLoopPoles PIDStabilityEvidence = "closed-loop-poles"
	// PIDStabilityNyquist counts Nyquist encirclements of a continuous delayed
	// plant on a delay-aware adaptive grid.
	PIDStabilityNyquist PIDStabilityEvidence = "nyquist-encirclement"
	// PIDStabilitySampledOnly checks only sampled frequencies; it is not a
	// global stability certificate.
	PIDStabilitySampledOnly PIDStabilityEvidence = "sampled-frequency-only"
)

// PIDTuningTermination names why the TunePID search stopped.
type PIDTuningTermination string

const (
	PIDTerminationConverged       PIDTuningTermination = "converged"
	PIDTerminationRoundLimit      PIDTuningTermination = "round-limit"
	PIDTerminationEvaluationLimit PIDTuningTermination = "evaluation-limit"
)

// PIDTuningError reports a TunePID or TunePIDFRD search that met no
// candidate satisfying the target and stability checks. It unwraps to
// ErrPIDTuningTargetUnattainable and carries the search evidence.
type PIDTuningError struct {
	Op       string
	Evidence PIDTuningEvidence
}

func (e *PIDTuningError) Error() string {
	return e.Op + ": " + ErrPIDTuningTargetUnattainable.Error()
}

func (e *PIDTuningError) Unwrap() error { return ErrPIDTuningTargetUnattainable }

// PIDTuningWeights enables two-degree-of-freedom tuning. A nil field is free in
// [0,1]; a nonnil field is fixed exactly. Nil options.Weights means b=c=1.
type PIDTuningWeights struct{ FixedB, FixedC *float64 }

// PIDTuningOptions holds the TunePID targets; zero values select defaults.
type PIDTuningOptions struct {
	// CrossoverFrequency is the target gain crossover in rad/s; 0 picks it
	// from the plant (TunePID) or the geometric band centre (TunePIDFRD).
	CrossoverFrequency float64
	// PhaseMargin in degrees, in (0, 180); 0 means 60, or the plant's own
	// margin for the one-gain P and I families.
	PhaseMargin        float64
	Focus              PIDDesignFocus
	IFormula, DFormula PIDFormula
	Weights            *PIDTuningWeights
	// MaxEvaluations bounds candidate evaluations, 1..4096; 0 means 4096.
	MaxEvaluations int
	// UnstablePoles is optional user knowledge for TunePIDFRD; nil means
	// unknown. TunePID rejects it with ErrOptionUnsupported.
	UnstablePoles *int
}

// PIDTuningObjective holds the frequency-averaged tracking, rejection and
// effort terms of the tuning objective.
type PIDTuningObjective struct{ Tracking, Rejection, Effort float64 }

// PIDTuningEvidence records the targets, achieved margins and search of a
// TunePID design. AchievedCrossover equals RequestedCrossover because every
// candidate is gain-normalized at it. SeedObjective is +Inf in a
// PIDTuningError when no seed was feasible.
type PIDTuningEvidence struct {
	RequestedCrossover, RequestedPhaseMargin float64
	AchievedCrossover, AchievedPhaseMargin   float64
	GainCrossovers, PhaseMargins             []float64
	FrequencyBand                            [2]float64
	Objective, SeedObjective                 float64
	Components, Normalizers                  PIDTuningObjective
	Evaluations                              int
	Termination                              PIDTuningTermination
	// Stability names the check applied to every accepted candidate.
	// Discrete delays are absorbed into the closed-loop poles. Continuous
	// delayed plants use a Nyquist encirclement count on a delay-aware
	// adaptive grid when the plant is standard with an acyclic delay network;
	// otherwise only sampled frequencies are checked, which is deliberately
	// not a global stability certificate.
	Stability PIDStabilityEvidence
	Warnings  []string
}

// PIDTuningResult is a feasible TunePID design and its evidence.
type PIDTuningResult struct {
	Controller *PID2
	Evidence   PIDTuningEvidence
}

type pidTuningPlant struct {
	dt, low, high float64
	at            func(float64) complex128
	stable        func(*PID2) (bool, error)
	stability     PIDStabilityEvidence
	warnings      []string
}

// TunePID designs a SISO negative-feedback controller. The original Pidtune API
// retains its historical behavior. This operation enforces finite targets,
// bounded computation and explicit achieved-target/stability evidence.
//
// An unattainable target returns a *PIDTuningError (wrapping
// ErrPIDTuningTargetUnattainable) carrying the search evidence; invalid
// options return ErrInvalidArgument; a numerical failure of a stability
// check aborts the search with its error.
func TunePID(ctx context.Context, plant *System, family PidtuneType, opts PIDTuningOptions) (*PIDTuningResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("TunePID: nil context: %w", ErrInvalidArgument)
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("TunePID: %w", err)
	}
	if err := requireFiniteSystem("TunePID", plant); err != nil {
		return nil, err
	}
	if _, err := newSISOLoopModel(plant, "TunePID"); err != nil {
		return nil, err
	}
	if opts.UnstablePoles != nil {
		return nil, fmt.Errorf("TunePID: UnstablePoles applies only to TunePIDFRD: %w", ErrOptionUnsupported)
	}
	wc := opts.CrossoverFrequency
	if wc == 0 {
		var err error
		wc, err = findCrossoverFreq(plant, 0)
		if err != nil {
			return nil, fmt.Errorf("TunePID: %w", err)
		}
		opts.CrossoverFrequency = wc
	}
	eval, err := newSISOEval(plant)
	if err != nil {
		return nil, fmt.Errorf("TunePID: %w", err)
	}
	p := pidTuningPlant{dt: plant.Dt, low: wc / 100, high: wc * 100, at: eval.at, stability: PIDStabilitySampledOnly}
	if plant.Dt > 0 {
		p.high = math.Min(p.high, .99*math.Pi/plant.Dt)
	}
	model := plant
	if plant.Dt > 0 && plant.HasDelay() {
		if model, err = plant.AbsorbDelay(AbsorbAll); err != nil {
			return nil, fmt.Errorf("TunePID: %w", err)
		}
	}
	if !model.HasDelay() {
		p.stability = PIDStabilityClosedLoopPoles
		hiddenStable := sync.OnceValues(func() (bool, error) { return pidTuningHiddenModesStable(model) })
		p.stable = func(c *PID2) (bool, error) {
			controller := &PID{Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf, Dt: c.Dt, IFormula: c.IFormula, DFormula: c.DFormula}
			cs, err := controller.System()
			if err != nil {
				if c.Kd != 0 && c.Tf == 0 {
					if ok, err := hiddenStable(); !ok || err != nil {
						return false, err
					}
					return pidTuningIdealStable(model, c)
				}
				return false, err
			}
			loop, err := Feedback(model, cs, -1)
			if errors.Is(err, ErrAlgebraicLoop) {
				return false, nil
			}
			if err != nil {
				return false, err
			}
			return loop.IsStable()
		}
	} else {
		stable, ok, err := pidTuningDelayStability(plant, eval, wc, pidTuningIdealDerivative(family))
		if err != nil {
			return nil, fmt.Errorf("TunePID: %w", err)
		}
		if ok {
			p.stability = PIDStabilityNyquist
			p.stable = stable
		} else {
			p.warnings = []string{"Exact delay retained; finite frequency samples do not certify global closed-loop stability."}
		}
	}
	res, err := tunePID(ctx, "TunePID", p, family, opts)
	if eval.err != nil {
		return nil, fmt.Errorf("TunePID: %w", eval.err)
	}
	return res, err
}

func tunePID(ctx context.Context, op string, p pidTuningPlant, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	invalid := func(format string, args ...any) error {
		return fmt.Errorf("%s: %s: %w", op, fmt.Sprintf(format, args...), ErrInvalidArgument)
	}
	if !finitePID(o.CrossoverFrequency) || o.CrossoverFrequency <= 0 {
		return nil, invalid("crossover %g must be positive and finite", o.CrossoverFrequency)
	}
	if !finitePID(p.dt) || p.dt < 0 {
		return nil, invalid("sample time %g must be finite and nonnegative", p.dt)
	}
	wc := o.CrossoverFrequency
	if p.dt > 0 && wc >= math.Pi/p.dt {
		return nil, invalid("crossover %g must be below Nyquist %g", wc, math.Pi/p.dt)
	}
	if wc <= p.low || wc >= p.high {
		return nil, invalid("crossover %g needs frequency coverage on both sides of [%g, %g]", wc, p.low, p.high)
	}
	free := o.PhaseMargin == 0
	if free {
		o.PhaseMargin = 60
	}
	if !finitePID(o.PhaseMargin) || o.PhaseMargin <= 0 || o.PhaseMargin >= 180 {
		return nil, invalid("phase margin %g must be between 0 and 180 degrees", o.PhaseMargin)
	}
	if o.Focus == "" {
		o.Focus = PIDFocusBalanced
	}
	tracking := .5
	switch o.Focus {
	case PIDFocusTracking:
		tracking = .95
	case PIDFocusRejection:
		tracking = .05
	case PIDFocusBalanced:
	default:
		return nil, invalid("unknown focus %q", o.Focus)
	}
	if o.IFormula < ForwardEuler || o.IFormula > Trapezoidal || o.DFormula < ForwardEuler || o.DFormula > Trapezoidal {
		return nil, invalid("unknown discrete formula")
	}
	if o.Weights != nil {
		for _, v := range []*float64{o.Weights.FixedB, o.Weights.FixedC} {
			if v != nil && !finitePID(*v) {
				return nil, invalid("fixed weights must be finite")
			}
		}
	}
	if o.MaxEvaluations == 0 {
		o.MaxEvaluations = 4096
	}
	if o.MaxEvaluations < 1 || o.MaxEvaluations > 4096 {
		return nil, invalid("evaluation bound %d must be 1..4096", o.MaxEvaluations)
	}
	family = PidtuneType(strings.ToUpper(string(family)))
	hasP, hasI, hasD, filtered := true, false, false, false
	switch family {
	case PidtuneP:
	case PidtuneI:
		hasP = false
		hasI = true
	case PidtunePI:
		hasI = true
	case PidtunePD:
		hasD = true
	case PidtunePDF:
		hasD = true
		filtered = true
	case PidtunePID:
		hasI = true
		hasD = true
	case PidtunePIDF:
		hasI = true
		hasD = true
		filtered = true
	default:
		return nil, invalid("unsupported family %q", family)
	}
	// Only explicitly filtered families introduce a derivative filter.
	tf := 0.
	if filtered {
		tf = .1 / wc
	}
	c := PID2{Tf: tf, Dt: p.dt, IFormula: o.IFormula, DFormula: o.DFormula, B: 1, C: 1}
	h := p.at(wc)
	if !pidFiniteComplex(h) || cmplx.Abs(h) < 1e-15 {
		return nil, invalid("plant gain %v at crossover is zero or not finite", h)
	}
	// P and I have one gain: its crossover fixes the phase margin, so an
	// unspecified margin takes the plant's instead of the 60° default.
	oneGain := free && !hasD && hasP != hasI
	if oneGain {
		unit := c
		if hasI {
			unit.Ki = 1
		} else {
			unit.Kp = 1
		}
		margin, ok := pidOneGainMargin(h * pidTuningFeedback(unit, wc))
		if !ok {
			return nil, &PIDTuningError{Op: op, Evidence: PIDTuningEvidence{RequestedCrossover: wc, Stability: p.stability}}
		}
		o.PhaseMargin = margin
	}
	target := cmplx.Rect(1, (-180+o.PhaseMargin)*math.Pi/180) / h
	ib, db := pidTuningBasis(c, wc)
	if hasP && hasI {
		if hasD {
			c.Kd = math.Copysign(math.Abs(imag(target)/imag(db))+.5*cmplx.Abs(target)/wc, real(target))
		}
		c.Ki = (imag(target) - c.Kd*imag(db)) / imag(ib)
		c.Kp = real(target) - c.Ki*real(ib) - c.Kd*real(db)
	} else if hasP && hasD {
		c.Kd = imag(target) / imag(db)
		c.Kp = real(target) - c.Kd*real(db)
	} else if hasI {
		c.Ki = cmplx.Abs(target) / cmplx.Abs(ib)
	} else {
		c.Kp = cmplx.Abs(target)
	}
	// Negative-gain plants require a signed P or I seed.
	if !hasP || (!hasI && !hasD) {
		a := c
		a.Kp = -a.Kp
		a.Ki = -a.Ki
		if pidPhaseDistance(p.at(wc)*pidTuningFeedback(a, wc), o.PhaseMargin) < pidPhaseDistance(p.at(wc)*pidTuningFeedback(c, wc), o.PhaseMargin) {
			c = a
		}
	}
	if !finitePID(c.Kp) || !finitePID(c.Ki) || !finitePID(c.Kd) {
		return nil, fmt.Errorf("%s: singular controller basis at crossover %g: %w", op, wc, ErrSingularTransform)
	}
	const points = 241
	omega, plant := make([]float64, points), make([]complex128, points)
	for k := range omega {
		omega[k] = math.Exp(math.Log(p.low) + float64(k)*math.Log(p.high/p.low)/float64(points-1))
		plant[k] = p.at(omega[k])
		if !pidFiniteComplex(plant[k]) {
			return nil, invalid("nonfinite plant response at %g", omega[k])
		}
	}
	// Include wc exactly so target evidence never depends on a nearby grid sample.
	mid := 0
	for i := range omega {
		if math.Abs(math.Log(omega[i]/wc)) < math.Abs(math.Log(omega[mid]/wc)) {
			mid = i
		}
	}
	omega[mid] = wc
	plant[mid] = h
	components := func(c PID2) PIDTuningObjective {
		v := PIDTuningObjective{}
		for k, w := range omega {
			fb := pidTuningFeedback(c, w)
			cr := pidTuningReference(c, w)
			den := 1 + plant[k]*fb
			m := complex(wc, 0) / (complex(wc, w))
			tr := plant[k] * cr / den
			dr := plant[k] / den
			ur := cr / den
			v.Tracking += pidAbsSquared(tr - m)
			v.Rejection += pidAbsSquared(dr)
			v.Effort += pidAbsSquared(ur) / (1 + w*w/(wc*wc))
		}
		v.Tracking /= points
		v.Rejection /= points
		v.Effort /= points
		return v
	}
	pidTuningSolveWeights(&c, o.Weights, omega, plant, wc)
	norms := components(c)
	if !finitePID(norms.Tracking) || !finitePID(norms.Rejection) || !finitePID(norms.Effort) {
		return nil, fmt.Errorf("%s: objective overflows; rescale plant units: %w", op, ErrOverflow)
	}
	norms.Tracking = math.Max(norms.Tracking, 1e-3)
	norms.Rejection = math.Max(norms.Rejection, 1e-3)
	norms.Effort = math.Max(norms.Effort, 1e-3)
	score := func(v PIDTuningObjective) float64 {
		return tracking*v.Tracking/norms.Tracking + (1-tracking)*v.Rejection/norms.Rejection + .001*v.Effort/norms.Effort
	}
	evidence := PIDTuningEvidence{RequestedCrossover: wc, RequestedPhaseMargin: o.PhaseMargin, Stability: p.stability, Warnings: append([]string(nil), p.warnings...), Normalizers: norms, Termination: PIDTerminationConverged}

	bestScore := math.Inf(1)
	var best PID2
	var bestParts PIDTuningObjective
	evaluate := func(candidate PID2) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if evidence.Evaluations >= o.MaxEvaluations {
			return nil
		}
		evidence.Evaluations++
		if filtered && (candidate.Tf < math.Max(.001/wc, p.dt*.51) || candidate.Tf > 10/wc) {
			return nil
		}
		mag := cmplx.Abs(h * pidTuningFeedback(candidate, wc))
		if !finitePID(mag) || mag < 1e-15 {
			return nil
		}
		candidate.Kp /= mag
		candidate.Ki /= mag
		candidate.Kd /= mag
		if pidPhaseDistance(h*pidTuningFeedback(candidate, wc), o.PhaseMargin) > 3 {
			return nil
		}
		pidTuningSolveWeights(&candidate, o.Weights, omega, plant, wc)
		v := components(candidate)
		j := score(v)
		if !finitePID(j) || j >= bestScore {
			return nil
		}
		if p.stable != nil {
			stable, err := p.stable(&candidate)
			if err != nil {
				return fmt.Errorf("stability check: %w", err)
			}
			if !stable {
				return nil
			}
		}
		// A free one-gain design has no margin to hold at other crossovers:
		// with a model-based stability certificate it needs only positive
		// margins and reports its lowest. Sampled responses keep the floor,
		// since their wrapped phase can hide a negative margin.
		relaxed := oneGain && p.stable != nil
		floor := o.PhaseMargin - 3
		if relaxed {
			floor = 0
		}
		_, margins := pidTuningCrossings(p, candidate, omega, wc)
		for _, margin := range margins {
			if margin < floor || relaxed && margin <= 0 {
				return nil
			}
		}
		best, bestScore, bestParts = candidate, j, v
		return nil
	}
	if err := evaluate(c); err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if hasP && hasI && hasD {
		seed := c
		seed.Kd = 0
		seed.Ki = imag(target) / imag(ib)
		seed.Kp = real(target) - seed.Ki*real(ib)
		if err := evaluate(seed); err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
	}
	evidence.SeedObjective = bestScore
	// Signed, scaled coordinate pattern search; each candidate is gain-normalized
	// at wc and must preserve target phase and model-based stability when available.
	current := c
	step := .5
	for round := 0; round < 64 && evidence.Evaluations < o.MaxEvaluations; round++ {
		previous := bestScore
		if finitePID(bestScore) {
			current = best
		}
		for axis := range 4 {
			if axis == 0 && !hasP || axis == 1 && !hasI || axis == 2 && !hasD || axis == 3 && !filtered {
				continue
			}
			for _, sign := range []float64{-1, 1} {
				candidate := current
				scale := 1 / cmplx.Abs(h)
				switch axis {
				case 0:
					candidate.Kp += sign * step * scale
				case 1:
					candidate.Ki += sign * step * scale * wc
				case 2:
					candidate.Kd += sign * step * scale / wc
				case 3:
					candidate.Tf *= math.Exp(sign * step)
				}
				if err := evaluate(candidate); err != nil {
					return nil, fmt.Errorf("%s: %w", op, err)
				}
			}
		}
		if bestScore >= previous-1e-6*math.Max(1, math.Abs(previous)) || !finitePID(bestScore) {
			step *= .5
		}
		if step < 1e-5 {
			break
		}
		if round == 63 {
			evidence.Termination = PIDTerminationRoundLimit
		}
	}
	if evidence.Evaluations >= o.MaxEvaluations {
		evidence.Termination = PIDTerminationEvaluationLimit
	}
	if !finitePID(bestScore) {
		return nil, &PIDTuningError{Op: op, Evidence: evidence}
	}
	if !finitePID(evidence.SeedObjective) {
		evidence.SeedObjective = bestScore
		evidence.Warnings = append(evidence.Warnings, "Initial seed was infeasible; objective baseline is the first feasible search result.")
	}
	evidence.Objective = bestScore
	evidence.Components = bestParts
	evidence.AchievedCrossover = wc
	evidence.AchievedPhaseMargin = 180 + cmplx.Phase(h*pidTuningFeedback(best, wc))*180/math.Pi
	if evidence.AchievedPhaseMargin > 180 {
		evidence.AchievedPhaseMargin -= 360
	}
	evidence.GainCrossovers, evidence.PhaseMargins = pidTuningCrossings(p, best, omega, wc)
	for _, margin := range evidence.PhaseMargins {
		evidence.AchievedPhaseMargin = math.Min(evidence.AchievedPhaseMargin, margin)
	}
	evidence.FrequencyBand = [2]float64{p.low, p.high}
	evidence.Warnings = append(evidence.Warnings, "Margins describe the recorded finite analysis band; unresolved crossings outside that band are not excluded.")
	return &PIDTuningResult{Controller: &best, Evidence: evidence}, nil
}

func pidTuningBasis(c PID2, w float64) (integral, derivative complex128) {
	if c.Dt == 0 {
		s := complex(0, w)
		return 1 / s, s / (1 + complex(c.Tf, 0)*s)
	}
	integral = pidtuneIntegrator(c.IFormula, w, c.Dt)
	d := 1 / pidtuneIntegrator(c.DFormula, w, c.Dt)
	return integral, d / (1 + complex(c.Tf, 0)*d)
}
func pidTuningFeedback(c PID2, w float64) complex128 {
	i, d := pidTuningBasis(c, w)
	return complex(c.Kp, 0) + complex(c.Ki, 0)*i + complex(c.Kd, 0)*d
}
func pidTuningReference(c PID2, w float64) complex128 {
	i, d := pidTuningBasis(c, w)
	return complex(c.B*c.Kp, 0) + complex(c.Ki, 0)*i + complex(c.C*c.Kd, 0)*d
}
func pidFiniteComplex(v complex128) bool { return finitePID(real(v)) && finitePID(imag(v)) }
func pidAbsSquared(v complex128) float64 { return real(v)*real(v) + imag(v)*imag(v) }

// pidOneGainMargin is the phase margin of the unit-gain loop at its
// crossover under the gain sign that makes it positive.
func pidOneGainMargin(loop complex128) (float64, bool) {
	if !pidFiniteComplex(loop) || loop == 0 {
		return 0, false
	}
	margin := 180 + cmplx.Phase(loop)*180/math.Pi
	if margin >= 180 {
		margin -= 180
	}
	if margin <= 0 || margin >= 180 {
		return 0, false
	}
	return margin, true
}

func pidPhaseDistance(loop complex128, pm float64) float64 {
	return math.Abs(math.Remainder(cmplx.Phase(loop)*180/math.Pi-(-180+pm), 360))
}

// Solve the two-dimensional box least squares exactly by checking the interior
// and faces. Fixed weights are removed from the equations, not rounded to a grid.
func pidTuningSolveWeights(c *PID2, o *PIDTuningWeights, omega []float64, plant []complex128, wc float64) {
	if o == nil {
		c.B = 1
		c.C = 1
		return
	}
	bFixed, cFixed := o.FixedB, o.FixedC
	if c.Kp == 0 && bFixed == nil {
		v := 1.
		bFixed = &v
	}
	if c.Kd == 0 && cFixed == nil {
		v := 1.
		cFixed = &v
	}
	aa, ab, bb, ay, by := 0., 0., 0., 0., 0.
	for k, w := range omega {
		ib, db := pidTuningBasis(*c, w)
		den := 1 + plant[k]*pidTuningFeedback(*c, w)
		a := plant[k] * complex(c.Kp, 0) / den
		b := plant[k] * complex(c.Kd, 0) * db / den
		y := complex(wc, 0)/complex(wc, w) - plant[k]*complex(c.Ki, 0)*ib/den
		aa += pidAbsSquared(a)
		bb += pidAbsSquared(b)
		ab += real(cmplx.Conj(a) * b)
		ay += real(cmplx.Conj(a) * y)
		by += real(cmplx.Conj(b) * y)
	}
	clamp := func(v float64) float64 { return math.Max(0, math.Min(1, v)) }
	if bFixed != nil && cFixed != nil {
		c.B = *bFixed
		c.C = *cFixed
		return
	}
	if bFixed != nil {
		c.B = *bFixed
		c.C = clamp((by - ab*c.B) / math.Max(bb, 1e-30))
		return
	}
	if cFixed != nil {
		c.C = *cFixed
		c.B = clamp((ay - ab*c.C) / math.Max(aa, 1e-30))
		return
	}
	best := math.Inf(1)
	accept := func(b, d float64) {
		if b < 0 || b > 1 || d < 0 || d > 1 {
			return
		}
		j := aa*b*b + 2*ab*b*d + bb*d*d - 2*ay*b - 2*by*d
		if j < best {
			best = j
			c.B = b
			c.C = d
		}
	}
	determinant := aa*bb - ab*ab
	if determinant > 1e-14*aa*bb {
		accept((ay*bb-by*ab)/determinant, (by*aa-ay*ab)/determinant)
	}
	for _, v := range []float64{0, 1} {
		accept(v, clamp((by-ab*v)/math.Max(bb, 1e-30)))
		accept(clamp((ay-ab*v)/math.Max(aa, 1e-30)), v)
	}
}

// Feedback cannot move uncontrollable or unobservable modes, and the transfer
// function used by pidTuningIdealStable omits them.
func pidTuningHiddenModesStable(plant *System) (bool, error) {
	if n, _, _ := plant.Dims(); n == 0 {
		return true, nil
	}
	stabilizable, err := IsStabilizable(plant.A, plant.B, plant.Dt == 0)
	if err != nil || !stabilizable {
		return false, err
	}
	return IsDetectable(plant.A, plant.C, plant.Dt == 0)
}

// Ideal derivative controllers may be improper on their own while their
// closed-loop characteristic is well-defined. Test the characteristic directly.
func pidTuningIdealStable(plant *System, c *PID2) (bool, error) {
	transfer, err := plant.rationalTransferFunction(nil)
	if err != nil {
		return false, err
	}
	var ip, id, dn, dd Poly
	if c.Ki == 0 {
		ip, id = Poly{0}, Poly{1}
	} else if c.Dt == 0 {
		ip, id = Poly{1}, Poly{1, 0}
	} else {
		switch c.IFormula {
		case BackwardEuler:
			ip, id = Poly{c.Dt, 0}, Poly{1, -1}
		case Trapezoidal:
			ip, id = Poly{c.Dt / 2, c.Dt / 2}, Poly{1, -1}
		default:
			ip, id = Poly{c.Dt}, Poly{1, -1}
		}
	}
	if c.Dt == 0 {
		dn, dd = Poly{1, 0}, Poly{1}
	} else {
		switch c.DFormula {
		case BackwardEuler:
			dn, dd = Poly{1, -1}, Poly{c.Dt, 0}
		case Trapezoidal:
			dn, dd = Poly{2, -2}, Poly{c.Dt, c.Dt}
		default:
			dn, dd = Poly{1, -1}, Poly{c.Dt}
		}
	}
	scale := func(p Poly, k float64) Poly {
		r := append(Poly(nil), p...)
		for i := range r {
			r[i] *= k
		}
		return r
	}
	denominator := id.Mul(dd)
	numerator := scale(denominator, c.Kp).Add(scale(ip.Mul(dd), c.Ki)).Add(scale(dn.Mul(id), c.Kd))
	characteristic := Poly(transfer.TF.Den[0]).Mul(denominator).Add(Poly(transfer.TF.Num[0][0]).Mul(numerator))
	for len(characteristic) > 1 && characteristic[0] == 0 {
		characteristic = characteristic[1:]
	}
	if len(characteristic) == 0 || characteristic[0] == 0 {
		return false, nil
	}
	poles, err := characteristic.Roots()
	if err != nil {
		return false, err
	}
	for _, pole := range poles {
		if !pidFiniteComplex(pole) || c.Dt == 0 && real(pole) >= 0 || c.Dt > 0 && cmplx.Abs(pole) >= 1 {
			return false, nil
		}
	}
	return true, nil
}

// Report all located gain crossings, retaining the exactly normalized target.
// Endpoint/grid evidence is explicitly finite-band even for a known rational plant.
func pidTuningCrossings(p pidTuningPlant, c PID2, omega []float64, wc float64) (crossings, margins []float64) {
	crossings = []float64{wc}
	previous := cmplx.Abs(p.at(omega[0])*pidTuningFeedback(c, omega[0])) - 1
	for k := 1; k < len(omega); k++ {
		next := cmplx.Abs(p.at(omega[k])*pidTuningFeedback(c, omega[k])) - 1
		if previous*next < 0 {
			lo, hi := omega[k-1], omega[k]
			lowSign := previous
			for range 24 {
				mid := math.Sqrt(lo * hi)
				v := cmplx.Abs(p.at(mid)*pidTuningFeedback(c, mid)) - 1
				if lowSign*v <= 0 {
					hi = mid
				} else {
					lo = mid
					lowSign = v
				}
			}
			w := math.Sqrt(lo * hi)
			if math.Abs(math.Log(w/wc)) > 1e-6 {
				crossings = append(crossings, w)
			}
		}
		previous = next
	}
	for _, w := range crossings {
		phase := 180 + cmplx.Phase(p.at(w)*pidTuningFeedback(c, w))*180/math.Pi
		if phase > 180 {
			phase -= 360
		}
		margins = append(margins, phase)
	}
	return crossings, margins
}

func pidTuningIdealDerivative(family PidtuneType) bool {
	f := PidtuneType(strings.ToUpper(string(family)))
	return f == PidtunePD || f == PidtunePID
}

// pidTuningDelayStability returns a Nyquist stability test for loops
// C(s)·P(s) with a continuous delayed plant, or false when the plant is
// outside the test's scope. Ideal derivative controllers need D = 0 and no
// internal delays so that |Kd·s·P(s)| has a finite high-frequency bound. A
// candidate loop outside the Nyquist test's scope is rejected; other
// failures are errors.
func pidTuningDelayStability(plant *System, eval *sisoEval, wc float64, idealDerivative bool) (func(*PID2) (bool, error), bool, error) {
	if !plant.IsContinuous() {
		return nil, false, nil
	}
	base, err := delayLoopFromSystem(plant, "delay loop")
	if errors.Is(err, errDelayLoopUnsupported) {
		return nil, false, nil
	}
	if err != nil {
		return nil, false, err
	}
	plantTail, plantLimit := base.tail, base.tailLimit
	var derivTail func(float64) float64
	if !plant.HasInternalDelay() && plant.D.At(0, 0) == 0 {
		n, _, _ := plant.Dims()
		cb, cab, normA := 0.0, 0.0, 0.0
		if n > 0 {
			var v, ca mat.Dense
			v.Mul(plant.C, plant.B)
			cb = math.Abs(v.At(0, 0))
			ca.Mul(plant.C, plant.A)
			cab = mat.Norm(&ca, 2) * mat.Norm(plant.B, 2)
			normA = mat.Norm(plant.A, 2)
		}
		derivTail = func(w float64) float64 {
			if cab == 0 || math.IsInf(w, 1) {
				return cb
			}
			if w <= normA {
				return math.Inf(1)
			}
			return cb + cab/(w-normA)
		}
	}
	if idealDerivative && derivTail == nil {
		return nil, false, nil
	}
	return func(c *PID2) (bool, error) {
		l := *base
		l.axis = slices.Clone(base.axis)
		l.scales = append(slices.Clone(base.scales), wc)
		if c.Ki != 0 {
			l.addAxisPole(0, 1)
		}
		kp, ki, kd := math.Abs(c.Kp), math.Abs(c.Ki), math.Abs(c.Kd)
		cand := *c
		l.eval = eval
		l.at = func(w float64) complex128 {
			s := complex(0, w)
			c := complex(cand.Kp, 0)
			if cand.Ki != 0 {
				c += complex(cand.Ki, 0) / s
			}
			if cand.Kd != 0 {
				c += complex(cand.Kd, 0) * s / (1 + complex(cand.Tf, 0)*s)
			}
			return eval.at(w) * c
		}
		switch {
		case kd == 0 || c.Tf > 0:
			gain := kp
			if kd != 0 {
				gain += kd / c.Tf
				l.scales = append(l.scales, 1/c.Tf)
			}
			l.tail = func(w float64) float64 { return (gain + ki/w) * plantTail(w) }
			l.tailLimit = gain * plantLimit
		case derivTail != nil:
			l.tail = func(w float64) float64 { return (kp+ki/w)*plantTail(w) + kd*derivTail(w) }
			l.tailLimit = kp*plantLimit + kd*derivTail(math.Inf(1))
		default:
			return false, nil
		}
		stable, err := l.stableClosedLoop()
		if errors.Is(err, errDelayLoopUnsupported) {
			return false, nil
		}
		return stable, err
	}, true, nil
}
