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
	// excess: least |S| excess among candidates meeting the margins; 0 if
	// none did or the bound was off.
	excess float64
	// unstable: candidates met the margins, and the stability check
	// rejected every one.
	unstable bool
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
	// from the plant, capped by its delay phase budget (TunePID), or the
	// geometric band centre (TunePIDFRD).
	CrossoverFrequency float64
	// PhaseMargin in degrees, in (0, 180); 0 means 60, or the plant's own
	// margin for the one-gain P and I families. TunePID treats it as a
	// floor, as pidtune does: it targets PM±3° and, failing that, the
	// nearest feasible margin above.
	PhaseMargin        float64
	Focus              PIDDesignFocus
	IFormula, DFormula PIDFormula
	Weights            *PIDTuningWeights
	// MaxEvaluations bounds candidate evaluations of each TunePID search,
	// 1..4096; 0 means 4096. The ±3° attempt and each moved crossover are
	// searches; the floor bands share one. Evidence counts the whole call.
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
	// gainSign is the sign every nonzero gain must take: that of the DC gain
	// of a stable plant, or 0 when unknown.
	gainSign float64
	// maxSensitivity bounds the sampled peak of |1/(1+L)| over the band, or
	// 0 for none; tau is the loop delay in seconds that sets its grid step.
	maxSensitivity, tau float64
	// modes are the plant's upper-half-plane poles and zeros in s, which
	// refine that grid near lightly damped resonances.
	modes []complex128
	// crossFloor, when hasCrossFloor, is the margin floor at the other gain
	// crossovers once the wc target is raised above the requested margin.
	crossFloor    float64
	hasCrossFloor bool
	// crossLow and crossHigh widen the band in which the floor finds the
	// other crossovers to the plant's dynamics, since a floor design's
	// high gain can cross 0 dB near a mode outside the band.
	crossLow, crossHigh float64
}

// pidTuningMaxSensitivity bounds |S| for a delay-capped or moved default
// crossover, so the design keeps at least 2.5 dB of gain margin and 14° of
// phase margin at every frequency. A margin met only at wc can leave
// |L| ≈ 1 where the delay phase passes −180° or the lead fades.
const pidTuningMaxSensitivity = 4

// pidTuningDCSign returns the sign of the DC gain of a stable plant, or 0
// when the plant is not known to be stable or its DC gain is zero relative
// to the peak of |G| sampled over six decades around wc.
func pidTuningDCSign(plant *System, at func(float64) complex128, wc float64) float64 {
	if stable, err := plant.IsStable(); err != nil || !stable {
		return 0
	}
	dc, err := evalPlantAt(plant, 0)
	if err != nil || !pidFiniteComplex(dc) || math.Abs(imag(dc)) > 1e-9*cmplx.Abs(dc) {
		return 0
	}
	peak := math.Abs(real(dc))
	for k := -3; k <= 3; k++ {
		w := wc * math.Pow(10, float64(k))
		if plant.Dt > 0 && w >= math.Pi/plant.Dt {
			break
		}
		if g := cmplx.Abs(at(w)); finitePID(g) {
			peak = math.Max(peak, g)
		}
	}
	if math.Abs(real(dc)) <= 1e3*0x1p-52*peak {
		return 0
	}
	return math.Copysign(1, real(dc))
}

// pidTuningSignConsistent reports whether every nonzero gain of c has sign
// want; want = 0 accepts any signs. Other signs on a stable plant come from
// the wrapped-phase branch of the crossover equations.
func pidTuningSignConsistent(c PID2, want float64) bool {
	if want == 0 {
		return true
	}
	for _, g := range []float64{c.Kp, c.Ki, c.Kd} {
		if g != 0 && math.Copysign(1, g) != want {
			return false
		}
	}
	return true
}

// TunePID designs a SISO negative-feedback controller. The original Pidtune API
// retains its historical behavior. This operation enforces finite targets,
// bounded computation and explicit achieved-target/stability evidence.
//
// A zero CrossoverFrequency is capped by the loop delay τ at (180°−PM)/(2τ),
// as Pidtune. For a stable plant every nonzero gain takes the sign of its DC
// gain; when that leaves a default-crossover PD or PDF design infeasible, the
// crossover moves to where the plant phase is 15°, 30°, 45° or 60° below
// −180°+PM, the lead same-sign gains supply. A default design on a delayed
// plant, or at a moved crossover, must keep |S| <= 4 (2.5 dB gain margin)
// at every sampled frequency, since a margin met at the crossover alone does
// not bound it there. Only when neither the default nor the moved crossovers
// meet PM±3° does the margin become a floor at the original crossover, as in
// pidtune: the design takes the nearest feasible 6° band above PM, with
// PM−3° at every crossover over the plant's dynamics. Bands whose phase is
// out of reach with |L| ≥ 1 below the crossover are skipped; the bands stop
// at one whose margin-meeting candidates are all unstable and, under the
// |S| bound, once the excess stops falling, so a feasible band can be
// missed. A continuous delayed loop with an ideal derivative must keep
// limsup |L(jω)| <= ½, so a near-neutral loop is rejected even when its
// sampled margins look healthy.
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
	eval, err := newSISOEval(plant)
	if err != nil {
		return nil, fmt.Errorf("TunePID: %w", err)
	}
	model := plant
	if plant.Dt > 0 && plant.HasDelay() {
		if model, err = plant.AbsorbDelay(AbsorbAll); err != nil {
			return nil, fmt.Errorf("TunePID: %w", err)
		}
	}
	hiddenStable := sync.OnceValues(func() (bool, error) { return pidTuningHiddenModesStable(model) })
	modes := sync.OnceValue(func() []complex128 { return pidTuningModes(model) })
	span := sync.OnceValues(func() (float64, float64) { return pidTuningDynamicsSpan(model) })
	tau := sisoLoopDelay(plant)
	if plant.Dt > 0 {
		tau *= plant.Dt
	}
	used := 0
	tuneAt := func(wc, gainSign float64, robust, floor bool) (*PIDTuningResult, error) {
		opts.CrossoverFrequency = wc
		p := pidTuningPlant{dt: plant.Dt, low: wc / 100, high: wc * 100, at: eval.at, stability: PIDStabilitySampledOnly, gainSign: gainSign, tau: tau}
		if robust {
			p.maxSensitivity = pidTuningMaxSensitivity
		}
		if robust || floor {
			p.modes = modes()
		}
		if floor {
			lo, hi := span()
			p.crossLow, p.crossHigh = math.Min(p.low, lo/10), math.Max(p.high, hi*10)
		}
		if plant.Dt > 0 {
			p.high = math.Min(p.high, .99*math.Pi/plant.Dt)
			p.crossHigh = math.Min(p.crossHigh, .99*math.Pi/plant.Dt)
		}
		if !model.HasDelay() {
			p.stability = PIDStabilityClosedLoopPoles
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
		tune := tunePID
		if floor {
			tune = tunePIDFloor
		}
		res, err := tune(ctx, "TunePID", p, family, opts)
		if eval.err != nil {
			return nil, fmt.Errorf("TunePID: %w", eval.err)
		}
		if te, ok := errors.AsType[*PIDTuningError](err); ok {
			used += te.Evidence.Evaluations
		} else if err == nil {
			used += res.Evidence.Evaluations
		}
		return res, err
	}
	// stamp reports the whole call's evaluations on the returned evidence.
	stamp := func(res *PIDTuningResult, err error) (*PIDTuningResult, error) {
		var ev *PIDTuningEvidence
		if err == nil {
			ev = &res.Evidence
		} else if te, ok := errors.AsType[*PIDTuningError](err); ok {
			ev = &te.Evidence
		} else {
			return res, err
		}
		ev.Evaluations = used
		return res, err
	}
	// The ±3° band is tried first; failing that, the margin is a floor.
	withFloor := func(wc, gainSign float64, robust bool, res *PIDTuningResult, err error) (*PIDTuningResult, error) {
		if !errors.As(err, new(*PIDTuningError)) {
			return stamp(res, err)
		}
		r, e := tuneAt(wc, gainSign, robust, true)
		if errors.As(e, new(*PIDTuningError)) {
			return stamp(res, err)
		}
		return stamp(r, e)
	}
	if wc := opts.CrossoverFrequency; wc != 0 {
		gainSign := pidTuningDCSign(plant, eval.at, wc)
		res, err := tuneAt(wc, gainSign, false, false)
		return withFloor(wc, gainSign, false, res, err)
	}
	pm := opts.PhaseMargin
	if pm <= 0 || pm >= 180 || !finitePID(pm) {
		pm = 60
	}
	wc0, err := findCrossoverFreq(plant, 0, pm)
	if err != nil {
		return nil, fmt.Errorf("TunePID: %w", err)
	}
	gainSign := pidTuningDCSign(plant, eval.at, wc0)
	res, err := tuneAt(wc0, gainSign, tau > 0, false)
	fam := PidtuneType(strings.ToUpper(string(family)))
	if gainSign == 0 || fam != PidtunePD && fam != PidtunePDF || !errors.As(err, new(*PIDTuningError)) {
		return withFloor(wc0, gainSign, tau > 0, res, err)
	}
	// Same-sign PD gains add lead in (0°, 90°) only, so retry where the
	// plant phase leaves a moderate lead to supply.
	wMax := wc0 * 1e4
	if plant.Dt > 0 {
		wMax = math.Min(wMax, .98*math.Pi/plant.Dt)
	}
	ladderWarning := ""
	for _, lead := range []float64{15, 30, 45, 60} {
		wc, ok := pidTuningPhaseCrossing(eval.at, gainSign, -180+pm-lead, wc0*1e-6, wMax, tau)
		if eval.err != nil {
			return nil, fmt.Errorf("TunePID: %w", eval.err)
		}
		if !ok || wc == wc0 {
			continue
		}
		r, e := tuneAt(wc, gainSign, true, false)
		if e == nil {
			r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("Default crossover %.4g moved to %.4g rad/s, where same-sign PD gains can meet the phase margin.", wc0, wc))
			return stamp(r, nil)
		}
		var te *PIDTuningError
		if !errors.As(e, &te) {
			return nil, e
		}
		for _, w := range te.Evidence.Warnings {
			if strings.Contains(w, "sensitivity peak") {
				ladderWarning = fmt.Sprintf("Moved crossover %.4g: %s", wc, w)
			}
		}
	}
	var te *PIDTuningError
	if ladderWarning != "" && errors.As(err, &te) && !slices.Contains(te.Evidence.Warnings, ladderWarning) {
		te.Evidence.Warnings = append(te.Evidence.Warnings, ladderWarning)
	}
	return withFloor(wc0, gainSign, tau > 0, res, err)
}

// pidTuningPhaseCrossing returns the first ω in [wLo, wHi] at which the
// phase of sign·G(jω), unwrapped from its wrapped value at wLo, falls to
// target degrees.
func pidTuningPhaseCrossing(at func(float64) complex128, sign, target, wLo, wHi, tau float64) (float64, bool) {
	w, _, ok := pidTuningPhaseWalk(at, sign, target, wLo, wHi, tau)
	return w, ok
}

// pidTuningUnwrappedPhase returns the phase in degrees of sign·G(jw),
// unwrapped from its wrapped value at w·1e-6, or NaN when the walk fails.
func pidTuningUnwrappedPhase(at func(float64) complex128, sign, w, tau float64) float64 {
	end, phase, ok := pidTuningPhaseWalk(at, sign, math.Inf(-1), w*1e-6, w, tau)
	if ok || end != w {
		return math.NaN()
	}
	return phase
}

// pidTuningPhaseWalk walks the unwrapped phase of sign·G(jω) up from wLo.
// It returns the first ω at which the phase falls to target degrees, or
// false with wHi and the phase there in degrees; a nonfinite or zero
// response ends the walk with ω = 0.
func pidTuningPhaseWalk(at func(float64) complex128, sign, target, wLo, wHi, tau float64) (float64, float64, bool) {
	target *= math.Pi / 180
	w := wLo
	g := complex(sign, 0) * at(w)
	if !pidFiniteComplex(g) || g == 0 {
		return 0, 0, false
	}
	phase := cmplx.Phase(g)
	for range 1 << 16 {
		if w >= wHi {
			return w, phase * 180 / math.Pi, false
		}
		next := math.Min(w*1.02, wHi)
		if tau > 0 {
			next = math.Min(next, w+math.Pi/8/tau)
		}
		var ng complex128
		var step float64
		for {
			ng = complex(sign, 0) * at(next)
			if !pidFiniteComplex(ng) || ng == 0 {
				return 0, 0, false
			}
			step = cmplx.Phase(ng / g)
			if math.Abs(step) <= math.Pi/2 || next-w < 1e-12*w {
				break
			}
			next = w + (next-w)/2
		}
		if phase+step <= target {
			return w + (next-w)*(phase-target)/(-step), target * 180 / math.Pi, true
		}
		w, g, phase = next, ng, phase+step
	}
	return 0, 0, false
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
	hasP, hasI, hasD, filtered, ok := pidTuningTerms(family)
	if !ok {
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
		if p.gainSign != 0 {
			if math.Copysign(1, c.Kp+c.Ki) != p.gainSign {
				c = a
			}
		} else if pidPhaseDistance(p.at(wc)*pidTuningFeedback(a, wc), o.PhaseMargin) < pidPhaseDistance(p.at(wc)*pidTuningFeedback(c, wc), o.PhaseMargin) {
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
	crossOmega := omega
	if p.hasCrossFloor {
		crossOmega = pidTuningCrossGrid(p, omega)
	}
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

	// The search follows the incumbent; best also meets the sensitivity
	// bound, which a path to a robust optimum may cross. The incumbent is
	// ranked by score, so a feasible optimum reached through infeasible
	// points is kept. When that finds no design within the bound, a second
	// pass ranks it by least excess over the bound, then score.
	bestScore := math.Inf(1)
	var best, incumbent PID2
	var bestParts PIDTuningObjective
	var incumbentScore, incumbentExcess float64
	signRejected, sensitivityRejected := false, false
	stabilityChecked, stableFound := false, false
	leastExcess := math.Inf(1)
	excessFirst := false
	evaluate := func(candidate PID2) error {
		if evidence.Evaluations >= o.MaxEvaluations {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		evidence.Evaluations++
		if filtered && (candidate.Tf < math.Max(.001/wc, p.dt*.51) || candidate.Tf > 10/wc) {
			return nil
		}
		if !pidTuningSignConsistent(candidate, p.gainSign) {
			signRejected = true
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
		// A free one-gain design has no margin to hold at other crossovers:
		// with a model-based stability certificate it needs only positive
		// margins and reports its lowest. Sampled responses keep the floor,
		// since their wrapped phase can hide a negative margin.
		relaxed := oneGain && p.stable != nil
		floor := o.PhaseMargin - 3
		if p.hasCrossFloor {
			floor = p.crossFloor
		}
		if relaxed {
			floor = 0
		}
		_, margins := pidTuningCrossings(p, candidate, crossOmega, wc)
		for _, margin := range margins {
			if margin < floor || relaxed && margin <= 0 {
				return nil
			}
		}
		if p.stable != nil {
			stable, err := p.stable(&candidate)
			if err != nil {
				return fmt.Errorf("stability check: %w", err)
			}
			stabilityChecked = true
			if !stable {
				return nil
			}
			stableFound = true
		}
		excess := 0.
		if p.maxSensitivity > 0 {
			excess = math.Max(0, pidTuningSensitivityPeak(p, candidate)-p.maxSensitivity)
		}
		leastExcess = math.Min(leastExcess, excess)
		if excessFirst && (excess < incumbentExcess || excess == incumbentExcess && j < incumbentScore) || !excessFirst && j < incumbentScore {
			incumbent, incumbentScore, incumbentExcess = candidate, j, excess
		}
		if excess > 0 {
			sensitivityRejected = true
			return nil
		}
		best, bestScore, bestParts = candidate, j, v
		return nil
	}
	// A floor band under the bound runs only the excess-ranked pass, which
	// halves its cost.
	first := 0
	if p.hasCrossFloor && p.maxSensitivity > 0 {
		first = 1
	}
	for pass := first; pass < 2; pass++ {
		excessFirst = pass == 1
		if pass > first && (finitePID(bestScore) || !sensitivityRejected || evidence.Evaluations >= o.MaxEvaluations) {
			break
		}
		incumbentScore, incumbentExcess = math.Inf(1), math.Inf(1)
		evidence.Termination = PIDTerminationConverged
		// A seed is feasible for the pass's ranking: any incumbent when
		// ranked by score, one within the bound when ranked by excess.
		seeded := func() bool {
			return excessFirst && finitePID(bestScore) || !excessFirst && finitePID(incumbentScore)
		}
		if err := evaluate(c); err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		if hasP && hasI && hasD {
			// The lead-heavy seed can break a derivative tail bound; smaller Kd
			// on the same target are tried only when no seed is feasible.
			for _, frac := range []float64{0, .75, .5, .25} {
				if frac > 0 && seeded() {
					break
				}
				seed := c
				seed.Kd = frac * c.Kd
				seed.Ki = (imag(target) - seed.Kd*imag(db)) / imag(ib)
				seed.Kp = real(target) - seed.Ki*real(ib) - seed.Kd*real(db)
				if err := evaluate(seed); err != nil {
					return nil, fmt.Errorf("%s: %w", op, err)
				}
			}
		}
		if pass == first {
			evidence.SeedObjective = bestScore
		}
		// Signed, scaled coordinate pattern search; each candidate is gain-normalized
		// at wc and must preserve target phase and model-based stability when available.
		current := c
		step := .5
		for round := 0; round < 64 && evidence.Evaluations < o.MaxEvaluations; round++ {
			previous, previousExcess := incumbentScore, incumbentExcess
			if finitePID(incumbentScore) {
				current = incumbent
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
			tol := previous - 1e-6*math.Max(1, math.Abs(previous))
			improved := finitePID(incumbentScore) && !(incumbentScore >= tol)
			if excessFirst {
				improved = finitePID(incumbentExcess) && (!finitePID(previousExcess) ||
					incumbentExcess < previousExcess-1e-6*math.Max(1, previousExcess) ||
					incumbentExcess == previousExcess && incumbentScore < tol)
			}
			if !improved {
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
	}
	if !finitePID(bestScore) {
		if signRejected {
			evidence.Warnings = append(evidence.Warnings, "Candidates meeting the phase target needed a gain of sign opposite to the stable plant's DC gain; such designs are rejected.")
		}
		if sensitivityRejected {
			evidence.Warnings = append(evidence.Warnings, fmt.Sprintf("Candidates meeting the phase target had a sensitivity peak above %g at the default crossover; such designs are rejected.", p.maxSensitivity))
		}
		e := &PIDTuningError{Op: op, Evidence: evidence, unstable: stabilityChecked && !stableFound}
		if finitePID(leastExcess) {
			e.excess = leastExcess
		}
		return nil, e
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
	evidence.GainCrossovers, evidence.PhaseMargins = pidTuningCrossings(p, best, crossOmega, wc)
	for _, margin := range evidence.PhaseMargins {
		evidence.AchievedPhaseMargin = math.Min(evidence.AchievedPhaseMargin, margin)
	}
	evidence.FrequencyBand = [2]float64{crossOmega[0], crossOmega[len(crossOmega)-1]}
	evidence.Warnings = append(evidence.Warnings, "Margins describe the recorded finite analysis band; unresolved crossings outside that band are not excluded.")
	return &PIDTuningResult{Controller: &best, Evidence: evidence}, nil
}

// pidTuningTerms reports the terms of an upper-case family.
func pidTuningTerms(family PidtuneType) (hasP, hasI, hasD, filtered, ok bool) {
	switch family {
	case PidtuneP:
		return true, false, false, false, true
	case PidtuneI:
		return false, true, false, false, true
	case PidtunePI:
		return true, true, false, false, true
	case PidtunePD:
		return true, false, true, false, true
	case PidtunePDF:
		return true, false, true, true, true
	case PidtunePID:
		return true, true, true, false, true
	case PidtunePIDF:
		return true, true, true, true, true
	}
	return false, false, false, false, false
}

// pidTuningReachable reports whether a candidate of the family can put the
// loop phase at wc within 3° of −180°+pm. Gains of one sign span the cone
// of the term phasors; a filter only pulls the derivative phasor toward the
// proportional one, so its smallest admissible Tf bounds the cone. Unknown
// gain sign reaches every phase.
//
// phase is the unwrapped phase of gainSign·G(jwc) in degrees, or NaN. The
// loop phase starts within ±90° at low frequency, so if |L| ≥ 1 below wc,
// a stable loop needs its unwrapped phase at wc above −180°; with phase
// known the band is judged unwrapped. This excludes designs whose |L|
// dips below 1 and recovers below wc. Otherwise a cone wider than a
// half-plane reaches every phase.
func pidTuningReachable(p pidTuningPlant, family PidtuneType, o PIDTuningOptions, pm, phase float64) bool {
	hasP, hasI, hasD, filtered, ok := pidTuningTerms(family)
	if !ok || p.gainSign == 0 {
		return true
	}
	wc := o.CrossoverFrequency
	c := PID2{Dt: p.dt, IFormula: o.IFormula, DFormula: o.DFormula}
	if filtered {
		c.Tf = math.Max(.001/wc, p.dt*.51)
	}
	ib, db := pidTuningBasis(c, wc)
	lo, hi := math.Inf(1), math.Inf(-1)
	for _, t := range []struct {
		on bool
		v  complex128
	}{{hasP, 1}, {hasI, ib}, {hasD, db}} {
		if t.on {
			a := cmplx.Phase(t.v) * 180 / math.Pi
			lo, hi = math.Min(lo, a), math.Max(hi, a)
		}
	}
	want := -180 + pm
	if !math.IsNaN(phase) {
		return want >= phase+lo-3-1e-9 && want <= phase+hi+3+1e-9
	}
	if hi-lo >= 180 {
		return true
	}
	want -= cmplx.Phase(complex(p.gainSign, 0)*p.at(wc)) * 180 / math.Pi
	return !(math.Abs(math.Remainder(want-(lo+hi)/2, 360)) > (hi-lo)/2+3+1e-9)
}

// tunePIDFloor treats the phase margin as a floor, as pidtune does, once the
// ±3° band around it is unattainable: it targets the nearest band above,
// stepping 6° so the bands tile [PM+3°, 180°], and keeps PM−3° at the other
// crossovers, found over the band widened to the plant's dynamics. A free
// one-gain design already targets its own margin. The bands share one
// evaluation budget. Bands pidTuningReachable rules out are skipped. Two
// heuristics for speed end the search, though a later band's controller
// shape may still be feasible: a band whose candidates met the margins but
// all failed the stability check, since each check can cost a Nyquist
// sweep and, on a 2,000-plant sample, every design found after such a band
// crossed 0 dB decades from wc; and, under an |S| bound, a band whose least
// excess is no smaller than the band below's.
func tunePIDFloor(ctx context.Context, op string, p pidTuningPlant, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
	fam := PidtuneType(strings.ToUpper(string(family)))
	if o.PhaseMargin == 0 && (fam == PidtuneP || fam == PidtuneI) {
		return nil, &PIDTuningError{Op: op, Evidence: PIDTuningEvidence{RequestedCrossover: o.CrossoverFrequency, Stability: p.stability}}
	}
	pm := o.PhaseMargin
	if pm == 0 {
		pm = 60
	}
	budget := o.MaxEvaluations
	if budget == 0 {
		budget = 4096
	}
	p.crossFloor, p.hasCrossFloor = pm-3, true
	used, excess := 0, 0.
	phase := math.NaN()
	if p.gainSign != 0 {
		phase = pidTuningUnwrappedPhase(p.at, p.gainSign, o.CrossoverFrequency, p.tau)
	}
	last := &PIDTuningError{Op: op, Evidence: PIDTuningEvidence{RequestedCrossover: o.CrossoverFrequency, RequestedPhaseMargin: pm, Stability: p.stability}}
	for target := pm + 6; target < 180 && used < budget; target += 6 {
		if !pidTuningReachable(p, fam, o, target, phase) {
			continue
		}
		o.PhaseMargin = target
		o.MaxEvaluations = budget - used
		r, err := tunePID(ctx, op, p, family, o)
		if errors.As(err, &last) {
			used += last.Evidence.Evaluations
			if last.unstable || last.excess > 0 && excess > 0 && last.excess >= excess {
				break
			}
			excess = last.excess
			continue
		}
		if err != nil {
			return nil, err
		}
		r.Evidence.Evaluations += used
		r.Evidence.RequestedPhaseMargin = pm
		r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("Phase margin %g° is a floor: no design met it within ±3°, so the nearest feasible band above, %g°±3°, was targeted.", pm, target))
		return r, nil
	}
	last.Evidence.Evaluations = used
	return nil, last
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

// pidTuningDynamicsSpan returns the least and greatest |s| over the plant
// poles and zeros, discrete ones mapped by s = ln(z)/dt, ignoring those
// below 1e-9 of the greatest as rounded zeros; +Inf and 0 when there are
// none.
func pidTuningDynamicsSpan(plant *System) (lo, hi float64) {
	lo = math.Inf(1)
	poles, err := plant.Poles()
	if err != nil {
		return lo, 0
	}
	zeros, err := plant.Zeros()
	if err != nil {
		zeros = nil
	}
	var mags []float64
	for _, m := range slices.Concat(poles, zeros) {
		if plant.Dt > 0 {
			m = cmplx.Log(m) / complex(plant.Dt, 0)
		}
		if a := cmplx.Abs(m); finitePID(a) {
			mags = append(mags, a)
			hi = math.Max(hi, a)
		}
	}
	for _, a := range mags {
		if a > 1e-9*hi {
			lo = math.Min(lo, a)
		}
	}
	return lo, hi
}

// pidTuningCrossGrid extends the log grid omega at its own step over
// [p.crossLow, p.crossHigh] and adds points within 4|σ| of each lightly
// damped mode σ+jω₀, so a crossover pair at a resonance is not stepped over.
func pidTuningCrossGrid(p pidTuningPlant, omega []float64) []float64 {
	n := len(omega)
	step := math.Log(omega[n-1]/omega[0]) / float64(n-1)
	var grid []float64
	for k := math.Ceil(math.Log(omega[0]/p.crossLow) / step); k >= 1; k-- {
		grid = append(grid, omega[0]*math.Exp(-k*step))
	}
	grid = append(grid, omega...)
	for k := 1.; k <= math.Ceil(math.Log(p.crossHigh/omega[n-1])/step); k++ {
		grid = append(grid, math.Min(omega[n-1]*math.Exp(k*step), p.crossHigh))
	}
	for _, m := range p.modes {
		d := math.Max(math.Abs(real(m)), 1e-6*imag(m))
		for k := -8.; k <= 8; k++ {
			if w := imag(m) + k*d/2; w > grid[0] && w < grid[len(grid)-1] {
				grid = append(grid, w)
			}
		}
	}
	slices.Sort(grid)
	return slices.Compact(grid)
}

// pidTuningModes returns the plant poles and zeros σ+jω₀ with |σ| < 0.05·ω₀,
// discrete ones mapped by s = ln(z)/dt, for the |S| grid; nil when unknown.
// Better damped modes never cut its 1% step.
func pidTuningModes(plant *System) []complex128 {
	var modes []complex128
	poles, err := plant.Poles()
	if err != nil {
		return nil
	}
	zeros, err := plant.Zeros()
	if err != nil {
		zeros = nil
	}
	for _, m := range slices.Concat(poles, zeros) {
		if plant.Dt > 0 {
			m = cmplx.Log(m) / complex(plant.Dt, 0)
		}
		if pidFiniteComplex(m) && math.Abs(real(m)) < .05*imag(m) {
			modes = append(modes, m)
		}
	}
	return modes
}

// pidTuningSensitivityPeak returns max |1/(1+L(jω))| over [p.low, p.high].
// The grid step turns the delay phase by at most π/16 and, near a plant mode
// σ+jω₀, is at most max(|σ|, |ω−ω₀|)/4, so a lightly damped mode's narrow
// peak is bracketed. Grid maxima within 0.8 of the running peak are then
// refined by golden section on log ω, since a delay ripple can sit between
// samples and read ~10% low.
func pidTuningSensitivityPeak(p pidTuningPlant, c PID2) float64 {
	s := func(w float64) float64 { return 1 / cmplx.Abs(1+p.at(w)*pidTuningFeedback(c, w)) }
	peak := 0.
	var w0, v0, w1, v1 float64
	for k, w := 0, p.low; ; k++ {
		v := s(w)
		if k >= 2 && v1 >= v0 && v1 >= v && v1 >= .8*math.Max(peak, v) {
			_, g := pidTuningGoldenMax(s, w0, w)
			peak = math.Max(peak, g)
		}
		peak = math.Max(peak, v)
		w0, v0, w1, v1 = w1, v1, w, v
		if w >= p.high {
			return peak
		}
		step := .01 * w
		if p.tau > 0 {
			step = math.Min(step, math.Pi/16/p.tau)
		}
		for _, m := range p.modes {
			step = math.Min(step, math.Max(math.Max(math.Abs(real(m)), math.Abs(w-imag(m))), 1e-6*w)/4)
		}
		w = math.Min(w+step, p.high)
	}
}

// pidTuningGoldenMax returns the largest f found by golden-section search
// over log ω in [lo, hi], where f is unimodal, and its ω.
func pidTuningGoldenMax(f func(float64) float64, lo, hi float64) (float64, float64) {
	const r = 0.6180339887498949
	a, b := math.Log(lo), math.Log(hi)
	x1, x2 := b-r*(b-a), a+r*(b-a)
	f1, f2 := f(math.Exp(x1)), f(math.Exp(x2))
	for range 24 {
		if f1 >= f2 {
			b, x2, f2 = x2, x1, f1
			x1 = b - r*(b-a)
			f1 = f(math.Exp(x1))
		} else {
			a, x1, f1 = x1, x2, f2
			x2 = a + r*(b-a)
			f2 = f(math.Exp(x2))
		}
	}
	if f1 >= f2 {
		return math.Exp(x1), f1
	}
	return math.Exp(x2), f2
}

// Report all located gain crossings, retaining the exactly normalized target.
// Endpoint/grid evidence is explicitly finite-band even for a known rational plant.
// Under the floor, a grid extremum of |L| within a factor 2 of 0 dB is
// refined by golden section, so a crossover pair between samples, as at a
// lightly damped resonance, is found.
func pidTuningCrossings(p pidTuningPlant, c PID2, omega []float64, wc float64) (crossings, margins []float64) {
	crossings = []float64{wc}
	// Outside the floor the scan is kept verbatim: the search amplifies
	// last-bit changes in |L| into different designs.
	if !p.hasCrossFloor {
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
		return crossings, pidTuningMargins(p, c, crossings)
	}
	excess := func(w float64) float64 { return cmplx.Abs(p.at(w)*pidTuningFeedback(c, w)) - 1 }
	add := func(lo, hi, lowSign float64) {
		for range 24 {
			mid := math.Sqrt(lo * hi)
			v := excess(mid)
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
	var before float64
	previous := excess(omega[0])
	for k := 1; k < len(omega); k++ {
		next := excess(omega[k])
		if previous*next < 0 {
			add(omega[k-1], omega[k], previous)
		} else if k >= 2 && before*previous > 0 && previous*next > 0 &&
			(previous < 0 && previous > -.5 && previous >= before && previous >= next ||
				previous > 0 && previous < 1 && previous <= before && previous <= next) {
			sign := math.Copysign(1, previous)
			w, v := pidTuningGoldenMax(func(w float64) float64 { return -sign * excess(w) }, omega[k-2], omega[k])
			if v > 0 {
				add(omega[k-2], w, previous)
				add(w, omega[k], -v*sign)
			}
		}
		before, previous = previous, next
	}
	return crossings, pidTuningMargins(p, c, crossings)
}

func pidTuningMargins(p pidTuningPlant, c PID2, crossings []float64) (margins []float64) {
	for _, w := range crossings {
		phase := 180 + cmplx.Phase(p.at(w)*pidTuningFeedback(c, w))*180/math.Pi
		if phase > 180 {
			phase -= 360
		}
		margins = append(margins, phase)
	}
	return margins
}

func pidTuningIdealDerivative(family PidtuneType) bool {
	f := PidtuneType(strings.ToUpper(string(family)))
	return f == PidtunePD || f == PidtunePID
}

// pidDerivTail bounds |s·G(s)| = |CB + CA(sI−A)⁻¹B| for Re s ≥ 0, |s| ≥ w of a
// strictly proper SISO plant G, delays excluded.
func pidDerivTail(plant *System) func(float64) float64 {
	n, _, _ := plant.Dims()
	cb, cab, normA := 0.0, 0.0, 0.0
	if n > 0 {
		var v, ca mat.Dense
		v.Mul(plant.C, plant.B)
		cb = math.Abs(v.At(0, 0))
		ca.Mul(plant.C, plant.A)
		cab = spectralNorm(&ca) * spectralNorm(plant.B)
		normA = spectralNorm(plant.A)
	}
	return func(w float64) float64 {
		if cab == 0 || math.IsInf(w, 1) {
			return cb
		}
		if w <= normA {
			return math.Inf(1)
		}
		return cb + cab/(w-normA)
	}
}

// pidTuningMaxTail bounds limsup |L(jω)| of a delayed loop with an ideal
// derivative. Above 1 the loop is unstable; near 1 the neutral root chain at
// Re s = ln|L(j∞)|/τ that Kd·CB sets is barely damped, which no finite-band
// phase margin reveals. 1 − 1/pidTuningMaxSensitivity keeps |S| along the
// chain within the bound used for default designs. Other candidates need
// only limsup |L| < 1.
const pidTuningMaxTail = 1 - 1.0/pidTuningMaxSensitivity

// pidTuningDelayStability returns a Nyquist stability test for loops
// C(s)·P(s) with a continuous delayed plant, or false when the plant is
// outside the test's scope. Ideal derivative controllers need no internal
// delays; with D ≠ 0, |Kd·s·P(s)| is unbounded, so every Kd ≠ 0 is rejected.
// A candidate loop outside the Nyquist test's scope, or an ideal derivative
// whose high-frequency gain exceeds pidTuningMaxTail, is rejected; other
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
		derivTail = pidDerivTail(plant)
	}
	if idealDerivative && plant.HasInternalDelay() {
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
			if l.tailLimit > pidTuningMaxTail {
				return false, nil
			}
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
