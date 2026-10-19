package controlsys

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"strconv"
	"strings"

	"plantcontrol.org/v1/gonum/mat"
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
// ErrInvalidArgument, as does a plant whose frequency response is
// numerically singular where Pidtune evaluates it (the error also wraps
// ErrSingularTransform), as for a high-order discrete realization with
// poles clustered near z = 1 that rounding scatters. The loop crossover and phase margin are placed using
// the realized controller response: discrete plants use the discrete PID
// terms, a discrete PID without filter uses a backward-Euler derivative to
// stay causal, and a discrete derivative whose forward-Euler filter pole
// 1−Dt/Tf would leave the unit circle (Tf ≤ Dt/2) uses backward Euler unless
// only the forward-Euler design gives a stable loop that keeps the margin at
// every gain crossover and |S| ≤ 4, for any wc, as its controller is
// unstable. An explicitly
// requested phase margin the family cannot reach at wc returns
// ErrPIDTuningTargetUnattainable instead of a controller with a different
// margin; without one, a PDF needing phase lag at wc takes its least lead,
// a small Kd, as MATLAB does. A PIDF takes Tf = Td/10 unless that filter's
// lag keeps the margin from being reached at wc; it then takes Tf = 0.1/wc,
// else the off-target Td/10 design, whichever first gives a stable loop
// that, with wc = 0, keeps the margin at every gain crossover and |S| ≤ 4.
// Every gain takes one sign: that of the DC
// gain for a stable plant, and for an integrating or unstable plant the
// sign stabilization forces, else that of the numerator term largest at wc
// (see pidtuneGainSign), so Pidtune(−G) mirrors Pidtune(G); where the force
// lapses or there is none, as for an ideal derivative large enough to flip
// the closed-loop polynomial's leading coefficient or a stable plant with
// zero DC gain, the other sign is tried when the first one fails. A
// design whose closed loop is unstable by its poles (delay absorbed for
// discrete plants) or by a Nyquist test (continuous delays) is rejected;
// loops outside both tests are not screened. With wc = 0 a design must
// also pass the back-off gate: keep the phase margin at every gain
// crossover, those above the band the first wc sets included up to where
// the loop's Markov bound keeps |L| ≤ 3/4 (continuous plants without
// internal delays), and keep |S| ≤ 4; a back-off design must also not cross
// over above that band, which lowering wc would not raise. A first design
// failing it is retried
// at crossovers lowered up to three decades, or to a hundredth of the
// slowest non-integrating pole's natural frequency when that is lower, six
// decades at most (six when wc came from a pole because G has no gain
// crossover or bandwidth, or because its -3 dB bandwidth lies beyond ten
// times every pole and zero natural frequency with |G| peaking over 1e3
// times its value there, set by zeros near the integrator of a stable plant
// without a continuous delay, when wc is the first such step below ten
// times that frequency; not for an integrating or unstable plant with a
// continuous delay), accepting the first that passes. When none passes, the
// first design is returned if it passed the stability test, except for a
// wc from a pole of a plant without bandwidth, which must pass the gate;
// on a continuous plant with zero DC gain and no delay, a k·sᵐ high-pass
// plant leaving no phase to back off into, a P, or a PD clamped short of
// the margin at wc (with the default margin; an explicit one rejects it),
// passes there when the pole test certifies it stable with |S| ≤ 4. On a
// loop the pole test certifies stable, the margin at each gain crossover,
// wc included (TunePID excludes wc), is the classical 180° − |∠L|. A
// continuous delay is gated before its Nyquist test on the signed
// 180° + ∠L; a design failing only at reflected crossings, signed margin at
// most −(pm − 0.5°), with |S| ≤ 4 passes on the classical margin when the
// Nyquist test then certifies it stable, and keeps the signed failure where
// that test does not apply (as for an ideal derivative on a relative-degree-1
// plant, or internal delays). Otherwise the result is ErrPIDTuningTargetUnattainable.
// An explicit wc is honoured as in MATLAB: the design at wc is returned
// whenever its closed loop passes the stability test (an unstable
// forward-Euler filter aside), with |L(wc)| = 1 (a PD places its phase on
// its Td/10 filtered response, clamped to [0.05 rad, its most lead], else,
// when the most lead gives an unstable loop, keeps the Td of the automatic
// design, which clamps phase on the ideal derivative), without back-off and
// without checking the margin at other gain crossovers or |S| (the
// achieved margins are the caller's to check, as with AllMargin); only an
// unstable design, or an explicit phase margin unreachable at wc, returns
// ErrPIDTuningTargetUnattainable.
// A family with an integral term on a plant with a zero at the integrator
// (s = 0, z = 1, up to rounding) leaves a closed-loop pole there at any
// gain and also returns ErrPIDTuningTargetUnattainable; a zero strictly
// inside the stability region does not, unless it leaves that pole within
// the stability screen's tolerance of the boundary. TunePID offers bounded search with
// achieved-target evidence.
func Pidtune(plant *System, pidType PidtuneType, wc float64, opts *PidtuneOptions) (*PID, error) {
	pid, err := pidtune(plant, pidType, wc, opts)
	if errors.Is(err, ErrSingularTransform) && !errors.Is(err, ErrInvalidArgument) {
		return nil, fmt.Errorf("%w: plant realization too ill-conditioned to evaluate: %w", err, ErrInvalidArgument)
	}
	return pid, err
}

func pidtune(plant *System, pidType PidtuneType, wc float64, opts *PidtuneOptions) (*PID, error) {
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

	_, hasI, _, _, _ := pidTuningTerms(pidType)
	if hasI && pidtuneZeroAtIntegrator(plant) {
		return nil, fmt.Errorf("Pidtune: %s: a plant zero at the integrator cancels the integral term, leaving a closed-loop pole on the stability boundary: %w", pidType, ErrPIDTuningTargetUnattainable)
	}
	auto := wc == 0
	wc, src, err := autoCrossoverSource(plant, wc, pm)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}

	eval, err := newSISOEval(plant)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}
	sign, rule := pidtuneGainSign(plant, eval.at, wc, hasI)
	if eval.err != nil {
		return nil, fmt.Errorf("Pidtune: %w", eval.err)
	}
	screen, err := newPIDStabilityScreen(plant, eval, 1)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}
	run := &pidtuneRun{plant: plant, screen: screen, loop: newPidtuneLoop(screen.model), eval: eval, pidType: pidType, pm: pm, explicit: explicit, auto: auto, wc: wc, src: src, fromPole: src != pidtuneCrossGain, mirror: rule != pidtuneSignFixed, rule: rule}

	pid, first := run.design(wc, sign, auto)
	if first == nil {
		return pid, nil
	}
	if !auto || !errors.Is(first, ErrPIDTuningTargetUnattainable) {
		return nil, fmt.Errorf("Pidtune: %w", first)
	}
	// An unstable plant needs bandwidth that a continuous delay already
	// caps, and each back-off step would run a Nyquist test.
	if !((rule == pidtuneSignForced || rule == pidtuneSignGuessed) && plant.Dt == 0 && plant.HasDelay()) {
		steps := int(math.Ceil(pidtuneBackoffPerDecade*math.Log10(wc/run.backoffFloor()) - 1e-9))
		for k := 1; k <= steps; k++ {
			pid, err := run.design(wc*math.Pow(10, -float64(k)/pidtuneBackoffPerDecade), sign, true)
			if err == nil {
				return pid, nil
			}
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				break
			}
		}
	}
	if src == pidtuneCrossGain || src == pidtuneCrossRolloff {
		if pid, err := run.design(wc, sign, false); err == nil {
			return pid, nil
		}
	}
	return nil, fmt.Errorf("Pidtune: %w", first)
}

// An automatic crossover whose design fails is lowered geometrically,
// pidtuneBackoffPerDecade steps a decade, down to pidtuneBackoffFloor, until
// the design is stable, keeps the phase margin at every gain crossover of
// the loop and keeps |S| ≤ pidTuningMaxSensitivity.
const (
	pidtuneBackoffPerDecade = 20
	pidtuneBackoffDecades   = 3
	pidtuneBackoffMaxDecade = 6
	pidtuneSlowPoleBackoff  = 100
)

// pidtuneBackoffFloor returns the lowest back-off crossover for an automatic
// crossover wc: pidtuneBackoffDecades below it, or lower, by at most
// pidtuneBackoffMaxDecade decades, to 1/pidtuneSlowPoleBackoff of the
// slowest non-integrating pole, whose phase lag can make every crossover
// above it unattainable for a family without lead.
func pidtuneBackoffFloor(plant *System, wc float64) float64 {
	floor := wc * math.Pow(10, -pidtuneBackoffDecades)
	slow := pidtuneSlowestPole(plant, wc)
	if slow > 0 {
		floor = math.Max(math.Min(floor, slow/pidtuneSlowPoleBackoff), wc*math.Pow(10, -pidtuneBackoffMaxDecade))
	}
	return floor
}

// pidtuneSlowestPole returns the smallest natural frequency of a plant pole
// (|ln z|/Dt for discrete plants) outside pidtuneNearIntegrator's tolerance
// of the integrator, or 0 when there is none.
func pidtuneSlowestPole(plant *System, wc float64) float64 {
	model, err := finiteDimensionalModel(plant, "Pidtune")
	if err != nil {
		return 0
	}
	poles, err := model.Poles()
	if err != nil {
		return 0
	}
	at, tol := pidtuneIntegratorTol(plant, poles, wc)
	slow := pidtuneSlowestRoot(plant, poles, at, tol)
	if math.IsInf(slow, 1) {
		return 0
	}
	return slow
}

// pidtuneSlowestRoot returns the smallest natural frequency (|ln z|/Dt for
// discrete plants) of roots farther than tol from at, or +Inf when there is
// none.
func pidtuneSlowestRoot(plant *System, roots []complex128, at complex128, tol float64) float64 {
	slow := math.Inf(1)
	for _, p := range roots {
		if cmplx.Abs(p-at) <= tol {
			continue
		}
		w := cmplx.Abs(p)
		if plant.Dt > 0 {
			w = cmplx.Abs(cmplx.Log(p)) / plant.Dt
		}
		if w > 0 && w < slow {
			slow = w
		}
	}
	return slow
}

// pidtuneIntegratorTol returns the integrator location, s = 0 or z = 1, and
// the distance within which pidtuneNearIntegrator counts a pole as one.
func pidtuneIntegratorTol(plant *System, poles []complex128, wc float64) (complex128, float64) {
	if plant.Dt > 0 {
		return 1, max(1e-6*wc*plant.Dt, 1e-8)
	}
	tol := 1e-6 * wc
	for _, p := range poles {
		tol = max(tol, 1e-8*cmplx.Abs(p))
	}
	return 0, tol
}

// pidtuneRun holds what Pidtune's designs at successive crossovers share.
type pidtuneRun struct {
	plant   *System
	screen  *pidStabilityScreen
	loop    *pidtuneLoop
	at      pidtunePlantPoint
	eval    *sisoEval
	pidType PidtuneType
	pm, wc  float64
	src     pidtuneCrossover
	// floor caches backoffFloor.
	floor float64
	// fromPole: wc is plantCrossoverFreq's pole fallback.
	fromPole bool
	explicit bool
	auto     bool
	// mirror tries the design against the chosen gain sign after the
	// signed one fails.
	mirror bool
	rule   pidtuneSignRule
	gate   *pidtuneGate
	// stable caches openStable: 0 unknown, 1 stable, −1 not.
	stable int
}

// design designs pidType at wc for sign·G, then for −sign·G when r.mirror
// is set and the screen has a stability test there, and screens each; gated
// designs must also pass the back-off gate. A discrete derivative moved to
// backward Euler for its filter pole falls back to the forward-Euler design
// when only that one passes, or the backward-Euler gains are unattainable;
// the fallback must always pass the back-off gate.
// Without an explicit margin, a PDF needing phase lag falls back to its
// least lead, gated under an automatic crossover since its margin exceeds
// the requested one.
func (r *pidtuneRun) design(wc, sign float64, gated bool) (*PID, error) {
	pid, lag, err := r.mirrored(wc, sign, gated, false)
	if err != nil && !r.explicit && lag {
		if alt, _, altErr := r.mirrored(wc, sign, gated || r.auto, true); altErr == nil {
			return alt, nil
		}
	}
	return pid, err
}

// mirrored designs for sign·G, then for −sign·G as design describes; with
// lagFloor a PDF needing phase lag gets the least lead instead. lag reports
// that the signed PDF needed phase lag, or under pidtuneSignStable either
// one: elsewhere the mirrored sign opposes the one stabilization favours,
// and its least lead can leave a closed-loop root the pole test misses.
func (r *pidtuneRun) mirrored(wc, sign float64, gated, lagFloor bool) (pid *PID, lag bool, err error) {
	pid, err = r.signed(wc, sign, gated, lagFloor)
	lag = errors.Is(err, errPidtunePDFLag)
	if r.mirror && errors.Is(err, ErrPIDTuningTargetUnattainable) && r.screened(wc) {
		alt, altErr := r.signed(wc, -sign, gated, lagFloor)
		if altErr == nil {
			return alt, false, nil
		}
		lag = lag || r.rule == pidtuneSignStable && errors.Is(altErr, errPidtunePDFLag)
	}
	return pid, lag, err
}

// screened returns whether checkPidtuneStability tests designs at wc: an
// unscreened design against the unforced sign may be certainly unstable.
func (r *pidtuneRun) screened(wc float64) bool {
	if r.loop != nil {
		return true
	}
	stable, _, err := r.screen.at(wc, r.pidType)
	return err == nil && stable != nil
}

func (r *pidtuneRun) signed(wc, sign float64, gated, lagFloor bool) (*PID, error) {
	pid, err := r.checked(wc, sign, gated, true, lagFloor)
	if err == nil || pid != nil && pid.DFormula != BackwardEuler {
		return pid, err
	}
	if r.plant.Dt == 0 || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		return nil, err
	}
	fe, feErr := r.checked(wc, sign, true, false, lagFloor)
	if feErr != nil || fe.DFormula != ForwardEuler {
		return nil, err
	}
	return fe, nil
}

// checked designs and checks one candidate, returning the design with its
// check error so signed can see its derivative formula. A PIDF whose
// Td/10 filter misses the target phase is replaced by its design at the
// default filter, else by the missed design itself, whichever first passes
// check, gated for wc = 0: a missed design can cross over elsewhere with
// almost no margin. An explicit-wc PD at its most lead failing check is
// replaced by its legacy-Td design (pidtunePDLegacyTd), whose larger Td sits
// on the ideal-derivative clamp, when that one passes.
func (r *pidtuneRun) checked(wc, sign float64, gated, stableFilter, lagFloor bool) (*PID, error) {
	pd := pidtunePDLegacy
	if !r.auto {
		pd = pidtunePDPlaced
	}
	pid, err := pidtuneGains(r.plant, &r.at, r.pidType, wc, r.pm, sign, stableFilter, lagFloor, pd, pidtunePIDFIterate)
	if err == nil {
		err = r.check(pid, wc, gated)
		if err != nil && pd == pidtunePDPlaced && r.pidType == PidtunePD && errors.Is(err, ErrPIDTuningTargetUnattainable) {
			alt, altErr := pidtuneGains(r.plant, &r.at, r.pidType, wc, r.pm, sign, stableFilter, lagFloor, pidtunePDLegacyTd, pidtunePIDFIterate)
			if altErr == nil && alt.Kd/alt.Kp > pid.Kd/pid.Kp && r.check(alt, wc, gated) == nil {
				return alt, nil
			}
		}
		return pid, err
	}
	if !errors.Is(err, errPidtunePIDFMiss) {
		return nil, err
	}
	for _, mode := range []pidtunePIDFMode{pidtunePIDFDefaultTf, pidtunePIDFKeep} {
		alt, altErr := pidtuneGains(r.plant, &r.at, r.pidType, wc, r.pm, sign, stableFilter, lagFloor, pd, mode)
		if altErr == nil && r.check(alt, wc, gated || r.auto) == nil {
			return alt, nil
		}
	}
	return nil, err
}

// check screens pid; the gate, cheaper than a Nyquist test of a continuous
// delay, runs before it there and after the pole test otherwise. There the
// gate counts signed margins 180° + ∠L, as the loop is not yet certified;
// when it fails only on reflected crossings, whose classical margin
// 180° − |∠L| holds, and |S| passes, the design passes if the Nyquist test
// certifies it stable, and keeps the signed failure when no test applies.
// On a stable plant, a design failing only at reflected crossings, or a
// back-off design the gate passes, is rejected, sparing the test, when the
// gate's samples show 1+L winding about 0 (pidtuneGate.winds).
func (r *pidtuneRun) check(pid *PID, wc float64, gated bool) error {
	if r.explicit {
		if err := checkPidtuneMargin(r.plant, pid, wc, r.pm); err != nil {
			return err
		}
	}
	gateFirst := r.plant.Dt == 0 && r.plant.HasDelay()
	if gated && gateFirst {
		err := r.gated(pid, wc, false)
		if err != nil && !errors.Is(err, errPidtuneReflected) {
			return err
		}
		if (err != nil || wc != r.wc) && r.openStable() && r.gateOf().winds(pidtuneGateController(pid)) {
			if err != nil {
				return err
			}
			if r.screened(wc) {
				return &pidtuneUnstableError{pidType: r.pidType, wc: wc}
			}
		}
		if err != nil && !r.screened(wc) {
			return err
		}
	}
	if err := checkPidtuneStability(r.screen, r.loop, r.eval, pid, r.pidType, wc); err != nil {
		return err
	}
	if gated && !gateFirst {
		if r.unshaped(pid, wc) {
			return r.gateOf().sensitive(pid, r.pidType, wc)
		}
		return r.gated(pid, wc, r.screened(wc))
	}
	return nil
}

// unshaped returns whether pid is a first design at a continuous wc from a
// pole of a zero-DC plant that the pole test certifies stable and that
// leaves the loop phase to the plant: a P, or a PD clamped short of the
// requested margin at wc. A k·sᵐ high-pass plant leads up to 180° below its
// poles, where no back-off helps, so only |S| gates it.
func (r *pidtuneRun) unshaped(pid *PID, wc float64) bool {
	if r.src != pidtuneCrossZeroDC || r.plant.Dt != 0 || wc != r.wc || r.loop == nil {
		return false
	}
	switch r.pidType {
	case PidtuneP:
		return true
	case PidtunePD:
		return checkPidtuneMargin(r.plant, pid, wc, r.pm) != nil
	}
	return false
}

// openStable returns whether the plant is stable, computed once.
func (r *pidtuneRun) openStable() bool {
	if r.stable == 0 {
		r.stable = -1
		if ok, err := r.plant.IsStable(); err == nil && ok {
			r.stable = 1
		}
	}
	return r.stable > 0
}

func (r *pidtuneRun) backoffFloor() float64 {
	if r.floor == 0 {
		r.floor = pidtuneBackoffFloor(r.plant, r.wc)
		if r.fromPole {
			r.floor = r.wc * math.Pow(10, -pidtuneBackoffMaxDecade)
		}
	}
	return r.floor
}

func (r *pidtuneRun) gated(pid *PID, wc float64, certified bool) error {
	return r.gateOf().check(pid, r.pidType, wc, r.pm, certified)
}

func (r *pidtuneRun) gateOf() *pidtuneGate {
	if r.gate == nil {
		r.gate = newPidtuneGate(r.plant, r.screen.model, r.eval, r.wc, r.backoffFloor())
	}
	return r.gate
}

// pidtuneSignRule says how pidtuneGainSign chose its sign, and so whether
// the mirrored design may be tried when the signed one fails.
type pidtuneSignRule int

const (
	// pidtuneSignFixed: the DC gain sign of a stable plant; never
	// mirrored.
	pidtuneSignFixed pidtuneSignRule = iota
	// pidtuneSignStable: a stable plant without a DC gain sign takes the
	// sign the other rules give; mirrored, and backed off as Fixed.
	pidtuneSignStable
	// pidtuneSignForced: K·N(0)/a > 0, which a stable loop needs whenever
	// its characteristic polynomial's leading coefficient has the sign of
	// a; mirrored when that coefficient may flip.
	pidtuneSignForced
	// pidtuneSignGuessed: the sign of N/a's term largest at wc, or 1; no
	// rule.
	pidtuneSignGuessed
)

// pidtuneGainSign returns the sign of every Pidtune gain: that of the DC
// gain of a stable plant, else of N(0)/a for the minimal G = N/D with D's
// leading coefficient a (N(1)/a in z) when it is forced, else of c/a for
// N's term c·sᵏ largest at |s| = wc (c·zᵏ largest at |z| = 1). The closed-loop polynomial P = D·Dc + N·Nc of a
// stable loop keeps the sign of its leading coefficient on [0, ∞) (z ≥ 1),
// where P(0) is D(0) + K·N(0) for P/PD and K·N(0) under integral action.
// When that leading coefficient has the sign of a, as for every loop whose
// L(∞) is 0, K·N(0)/a > 0 is forced if D(0)/a ≤ 0 (an integrator or an odd
// count of unstable real poles) or integral is set: the DC gain sign of G
// with integrators removed and unstable poles reflected. A stable plant
// with no DC sign and a pole at the integrator (pidtuneNearIntegrator) takes
// this rule too; one without such a pole, a zero-DC plant, takes the same
// signs under pidtuneSignStable, so −G gets −sign and either is mirrored.
func pidtuneGainSign(plant *System, at func(float64) complex128, wc float64, integral bool) (float64, pidtuneSignRule) {
	if s := pidTuningDCSign(plant, at, wc); s != 0 {
		return s, pidtuneSignFixed
	}
	stable, err := plant.IsStable()
	if err != nil {
		return 1, pidtuneSignFixed
	}
	if stable && !pidtuneNearIntegrator(plant, wc) {
		s, _ := pidtuneForcedSign(plant, wc, integral)
		return s, pidtuneSignStable
	}
	return pidtuneForcedSign(plant, wc, integral)
}

// pidtuneForcedSign returns pidtuneGainSign's sign and rule past the DC
// gain of a stable plant.
func pidtuneForcedSign(plant *System, wc float64, integral bool) (float64, pidtuneSignRule) {
	res, err := plant.rationalTransferFunction(nil)
	if err != nil {
		return 1, pidtuneSignGuessed
	}
	num, den := res.TF.Num[0][0], res.TF.Den[0]
	if len(num) == 0 || len(den) == 0 || den[0] == 0 {
		return 1, pidtuneSignGuessed
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
		if !nOK {
			return 1, pidtuneSignGuessed
		}
		return math.Copysign(1, n0*a), pidtuneSignForced
	}
	dom, scale := 0.0, 0.0
	for i, c := range num {
		if v := math.Abs(c) * math.Pow(x, float64(len(num)-1-i)); v > scale {
			dom, scale = c, v
		}
	}
	if !finitePID(scale) || scale == 0 {
		return 1, pidtuneSignGuessed
	}
	return math.Copysign(1, dom) * a, pidtuneSignGuessed
}

// pidtuneNearIntegrator returns whether plant has more poles than zeros
// within 1e-6·wc of s = 0 (1e-6·wc·Dt of z = 1), or within 1e-8 of it
// relative to the largest pole (to 1): rounding can leave an integrator just
// inside the stability boundary, where it still acts as one over the design
// band unless a zero there cancels it.
func pidtuneNearIntegrator(plant *System, wc float64) bool {
	model, err := finiteDimensionalModel(plant, "Pidtune")
	if err != nil {
		return false
	}
	poles, err := model.Poles()
	if err != nil {
		return false
	}
	zeros, _ := model.Zeros()
	at, tol := pidtuneIntegratorTol(plant, poles, wc)
	near := 0
	for _, p := range poles {
		if cmplx.Abs(p-at) <= tol {
			near++
		}
	}
	for _, z := range zeros {
		if cmplx.Abs(z-at) <= tol {
			near--
		}
	}
	return near > 0
}

// pidtuneZeroAtIntegrator returns whether plant has more zeros than poles at
// s = 0 (z = 1) up to rounding, counted as pidtuneIntegratorCluster does
// with the root scale of poles and zeros together: G vanishes there, and an
// integral term's pole cancels against the zero into a closed-loop pole at
// the integrator for every gain. A zero strictly inside the stability
// region, however close, leaves that pole strictly stable.
func pidtuneZeroAtIntegrator(plant *System) bool {
	model, err := finiteDimensionalModel(plant, "Pidtune")
	if err != nil {
		return false
	}
	poles, err := model.Poles()
	if err != nil {
		return false
	}
	zeros, err := model.Zeros()
	if err != nil {
		return false
	}
	at, scale := complex128(0), 0.0
	if plant.Dt > 0 {
		at, scale = 1, 1
	} else {
		for _, r := range slices.Concat(poles, zeros) {
			scale = max(scale, cmplx.Abs(r))
		}
	}
	return pidtuneClusterSize(zeros, at, scale) > pidtuneClusterSize(poles, at, scale)
}

// pidtuneGate checks a backed-off design over the whole back-off band with
// TunePID's sampling: |S| ≤ pidTuningMaxSensitivity, and the phase margin
// at every gain crossover, close pairs at a resonance or a near-miss of
// 0 dB included. The plant response and the discrete controller terms on
// both grids are computed once.
type pidtuneGate struct {
	p pidTuningPlant
	// wc is the first design's crossover; designs below it are back-off
	// steps.
	wc    float64
	sens  pidtuneGrid
	cross pidtuneGrid
	at    func(float64) complex128
	// tail bounds |L| above the band (pidTuningLoopTail), built on first use.
	plant    *System
	tail     func(*PID2, float64) float64
	markov   *pidMarkovTail
	tailDone bool
	// rootSum and rootMax are the sum and largest of the plant's pole and
	// zero magnitudes, built by rotates on first use (NaN when unknown).
	rootSum, rootMax float64
	rootsDone        bool
	// tailW, tailG and tailTop are tailPeak's grid, plant samples and
	// whether the scan's band ends at each; tailHigh is the band's end.
	tailW    []float64
	tailG    []complex128
	tailTop  []bool
	tailHigh float64
}

// pidtuneGrid caches, at each frequency w, the plant response g and, for
// discrete loops, each formula's integrator and its inverse, so a design is
// checked without trigonometry or plant evaluations at the grid points.
type pidtuneGrid struct {
	w        []float64
	g        []complex128
	dt       float64
	integral map[PIDFormula][]complex128
	inverse  map[PIDFormula][]complex128
}

// feedback returns i ↦ pidTuningFeedback(c, w[i]), bit for bit.
func (q *pidtuneGrid) feedback(c PID2) func(int) complex128 {
	if q.dt == 0 {
		return func(i int) complex128 { return pidTuningFeedback(c, q.w[i]) }
	}
	integral := q.integrator(c.IFormula)
	q.integrator(c.DFormula)
	inverse := q.inverse[c.DFormula]
	return func(i int) complex128 { return pidTuningDiscreteFeedback(c, integral[i], inverse[i]) }
}

func (q *pidtuneGrid) integrator(f PIDFormula) []complex128 {
	if v, ok := q.integral[f]; ok {
		return v
	}
	v, inv := make([]complex128, len(q.w)), make([]complex128, len(q.w))
	for i, w := range q.w {
		v[i] = pidtuneIntegrator(f, w, q.dt)
		inv[i] = 1 / v[i]
	}
	if q.integral == nil {
		q.integral, q.inverse = map[PIDFormula][]complex128{}, map[PIDFormula][]complex128{}
	}
	q.integral[f], q.inverse[f] = v, inv
	return v
}

const pidtuneGatePerDecade = 100

func newPidtuneGate(plant, model *System, eval *sisoEval, wc, floor float64) *pidtuneGate {
	lo, hi := floor*1e-3, wc*1e3
	if plant.Dt > 0 {
		hi = math.Pi / plant.Dt
	}
	tau := sisoLoopDelay(plant)
	if plant.Dt > 0 {
		tau *= plant.Dt
	}
	// at leaves eval.err as it was, so a failed sample cannot fail the
	// stability screens that share eval; it reads NaN instead.
	at := func(w float64) complex128 {
		saved := eval.err
		eval.err = nil
		g := eval.at(w)
		if eval.err != nil {
			g = cmplx.NaN()
		}
		eval.err = saved
		return g
	}
	memo := map[float64]complex128{}
	p := pidTuningPlant{
		dt: plant.Dt, low: lo, high: hi, tau: tau,
		maxSensitivity: pidTuningMaxSensitivity, modes: pidTuningModes(model),
		hasCrossFloor: true, crossLow: lo, crossHigh: hi,
		at: func(w float64) complex128 {
			g, ok := memo[w]
			if !ok {
				g = at(w)
				memo[w] = g
			}
			return g
		},
	}
	n := int(math.Ceil(math.Log10(hi/lo)*pidtuneGatePerDecade)) + 1
	omega := make([]float64, n)
	for i := range omega {
		omega[i] = math.Min(lo*math.Pow(10, float64(i)/pidtuneGatePerDecade), hi)
	}
	gate := &pidtuneGate{p: p, wc: wc, at: at, plant: plant}
	gate.cross = pidtuneGrid{w: pidTuningCrossGrid(p, omega), dt: plant.Dt}
	gate.sens.dt = plant.Dt
	for w := lo; ; {
		gate.sens.w = append(gate.sens.w, w)
		gate.sens.g = append(gate.sens.g, at(w))
		if w >= hi {
			break
		}
		step := .01 * w
		if tau > 0 {
			step = math.Min(step, math.Pi/16/tau)
		}
		for _, m := range p.modes {
			step = math.Min(step, math.Max(math.Max(math.Abs(real(m)), math.Abs(w-imag(m))), 1e-6*w)/4)
		}
		w = math.Min(w+step, hi)
	}
	return gate
}

// check rejects pid when |S| exceeds the bound or a gain crossover keeps
// less than pm; on a loop certified stable, every crossover, wc included,
// counts its classical margin 180° − |∠L|, the distance from −1 either way
// round the circle, which is |m| for the wrapped margin m. wc counts only
// when |L(wc)| = 1: a design clamped short of the gain there must cross
// elsewhere.
func (g *pidtuneGate) check(pid *PID, pidType PidtuneType, wc, pm float64, certified bool) error {
	if err := g.sensitive(pid, pidType, wc); err != nil {
		return err
	}
	c := pidtuneGateController(pid)
	_, margins := pidTuningCrossingsOn(g.p, c, g.cross.w, wc, g.crossExcess(c))
	// A back-off wc lies between grid points, where the scan cannot see a
	// crossover paired with it; pidTuningCrossingsOn seeks one when wc is on
	// the grid.
	if i, ok := slices.BinarySearch(g.cross.w, wc); !ok && i > 0 && i < len(g.cross.w) {
		_, m := pidTuningCrossingsOn(g.p, c, []float64{g.cross.w[i-1], wc, g.cross.w[i]}, wc, nil)
		margins = append(margins, m[1:]...)
	}
	if above, _ := g.beyond(&c); above != nil {
		_, m := pidTuningCrossingsOn(g.p, c, above, wc, nil)
		if wc < g.wc && len(m) > 1 {
			return fmt.Errorf("%s design at wc=%g, below wc=%g: %w", pidType, wc, g.wc, errPidtuneAboveBand)
		}
		margins = append(margins, m[1:]...)
	}
	floor := pm - pidtuneMarginTol
	if math.Abs(cmplx.Abs(g.p.at(wc)*pidTuningFeedback(c, wc))-1) > 1e-6 {
		margins = margins[1:]
		if len(margins) == 0 {
			return fmt.Errorf("%s design at wc=%g has no gain crossover: %w", pidType, wc, ErrPIDTuningTargetUnattainable)
		}
	}
	short := math.NaN()
	for _, m := range margins {
		if certified {
			m = math.Abs(m)
		}
		if !(m >= floor) {
			if math.IsNaN(short) {
				short = m
			}
			if !(m <= -floor) {
				return fmt.Errorf("%s design at wc=%g has phase margin %.3g° at a gain crossover: %w", pidType, wc, m, ErrPIDTuningTargetUnattainable)
			}
		}
	}
	if !math.IsNaN(short) {
		return fmt.Errorf("%s design at wc=%g has phase margin %.3g° at a gain crossover: %w", pidType, wc, short, errPidtuneReflected)
	}
	return nil
}

// winds reports whether the sampled Nyquist curve of a continuous loop
// with c on a stable plant shows its closed loop unstable, sparing the
// Nyquist test. Over [0⁺, ∞) a stable loop turns 1+L by 0, or by π/2 from
// −π/2 under integral action with Ki·G(0) > 0. The turn is unwrapped on the
// |S| grid, where |1+L| ≥ 1/maxSensitivity, taking 1+L not to wind below the
// band, and above it up to where the tail bound keeps |L| ≤
// 1 − 1/maxSensitivity, so 1+L returns to 1 inside Re > 0; a turn over π/2
// off the stable one is then unstable. Without that bound, a loop whose ∠L
// falls throughout [top, ∞), the delay outrunning every pole and zero (see
// rotates), crosses (−∞, −1) only clockwise there, so 1+L turns back by less
// than 2π; a turn short of the stable one by over 2.5π is then unstable.
func (g *pidtuneGate) winds(c PID2) bool {
	if g.plant.Dt != 0 || c.Tf < 0 || c.Kd != 0 && c.Tf == 0 {
		return false
	}
	g0 := g.at(0)
	if !pidFiniteComplex(g0) {
		return false
	}
	start, target := cmplx.Phase(1+g0*complex(c.Kp, 0)), 0.0
	if c.Ki != 0 {
		if !(c.Ki*real(g0) > 0) || math.Abs(imag(g0)) > 1e-9*math.Abs(real(g0)) {
			return false
		}
		start, target = -math.Pi/2, math.Pi/2
	}
	prev, total := cmplx.Rect(1, start), 0.0
	step := func(v complex128) bool {
		if !pidFiniteComplex(v) || v == 0 {
			return false
		}
		total += cmplx.Phase(v / prev)
		prev = v
		return true
	}
	k := g.sens.feedback(c)
	for i := range g.sens.w {
		if !step(1 + g.sens.g[i]*k(i)) {
			return false
		}
	}
	above, level := g.beyond(&c)
	if g.tail != nil && level == 0 && (above != nil || g.tail(&c, g.p.high) <= 1-1/g.p.maxSensitivity) {
		for _, w := range above {
			if !step(1 + g.p.at(w)*pidTuningFeedback(c, w)) {
				return false
			}
		}
		return math.Abs(total-cmplx.Phase(prev)-target) > math.Pi/2
	}
	return total < target-2.5*math.Pi && g.rotates(c)
}

// rotates returns whether ∠L falls throughout [top, ∞) for the band's top:
// a root r of the plant or c with |r| ≤ top/2 turns ∠L there at a rate
// below 4|r|/top², and the delay at τ.
func (g *pidtuneGate) rotates(c PID2) bool {
	if !g.rootsDone {
		g.rootsDone = true
		g.rootSum, g.rootMax = math.NaN(), math.NaN()
		model, err := finiteDimensionalModel(g.plant, "Pidtune")
		if err != nil {
			return false
		}
		poles, err := model.Poles()
		if err != nil {
			return false
		}
		zeros, err := model.Zeros()
		if err != nil {
			return false
		}
		g.rootSum, g.rootMax = 0, 0
		for _, r := range slices.Concat(poles, zeros) {
			g.rootSum += cmplx.Abs(r)
			g.rootMax = max(g.rootMax, cmplx.Abs(r))
		}
	}
	sum, big := g.rootSum, g.rootMax
	ctrl := []float64{c.Kd + c.Kp*c.Tf, c.Kp + c.Ki*c.Tf, c.Ki}
	if c.Tf > 0 {
		sum, big = sum+1/c.Tf, max(big, 1/c.Tf)
	}
	for len(ctrl) > 1 && ctrl[0] == 0 {
		ctrl = ctrl[1:]
	}
	zeros, err := Poly(ctrl).Roots()
	if err != nil {
		return false
	}
	for _, z := range zeros {
		sum, big = sum+cmplx.Abs(z), max(big, cmplx.Abs(z))
	}
	top := g.p.high
	return big <= top/2 && 4*sum < g.p.tau*top*top
}

// errPidtuneAboveBand marks a back-off design crossing over above the band
// the first crossover sets: lowering wc must not raise the bandwidth, as a
// gain placed where a zero near the integrator leaves |G| small does,
// decades up.
var errPidtuneAboveBand = fmt.Errorf("crosses over above the band of the first crossover: %w", ErrPIDTuningTargetUnattainable)

// errPidtuneReflected marks a gate failure at reflected crossings only,
// whose signed margin 180° + ∠L is at most −(pm − tol): their classical
// margin 180° − |∠L| holds.
var errPidtuneReflected = fmt.Errorf("classical margin met at every gain crossover: %w", ErrPIDTuningTargetUnattainable)

// sensitive rejects pid when |S| exceeds the bound, on the band, above it
// as far as beyond reaches, or as ω → ∞ on a loop whose gain levels off.
func (g *pidtuneGate) sensitive(pid *PID, pidType PidtuneType, wc float64) error {
	c := pidtuneGateController(pid)
	above, level := g.beyond(&c)
	ms := max(g.sensitivityOver(c, g.p.maxSensitivity), level)
	if above != nil && ms <= g.p.maxSensitivity {
		ms = max(ms, g.tailPeak(c, above[len(above)-1]))
	}
	if ms > g.p.maxSensitivity {
		return fmt.Errorf("%s design at wc=%g has sensitivity peak %.3g: %w", pidType, wc, ms, ErrPIDTuningTargetUnattainable)
	}
	return nil
}

// tailPeak is pidTuningSensitivityPeak from the band's top with the tail
// bound, stopping by reach instead of pidTuningTailReach times the top (bit
// for bit when both stop by the bound), sampling the plant on its grid once
// for all designs: the grid does not depend on c, only where the scan stops.
func (g *pidtuneGate) tailPeak(c PID2, reach float64) float64 {
	s := func(gw complex128, w float64) float64 { return 1 / cmplx.Abs(1+gw*pidTuningFeedback(c, w)) }
	refine := func(w float64) float64 { return s(g.at(w), w) }
	bound := 1 - 1/g.p.maxSensitivity
	peak := 0.
	var w0, v0, w1, v1 float64
	for k := 0; ; k++ {
		g.tailGrid(k)
		w := g.tailW[k]
		v := s(g.tailG[k], w)
		if k >= 2 && v1 >= v0 && v1 >= v && v1 >= .8*math.Max(peak, v) {
			_, r := pidTuningGoldenMax(refine, w0, w)
			peak = math.Max(peak, r)
		}
		peak = math.Max(peak, v)
		w0, v0, w1, v1 = w1, v1, w, v
		if k >= pidTuningTailPoints {
			return peak
		}
		if g.tailTop[k] && (w >= reach || g.tail(&c, w) <= bound) {
			return peak
		}
	}
}

// tailGrid extends tailPeak's grid through index k.
func (g *pidtuneGate) tailGrid(k int) {
	p := g.p
	for n := len(g.tailW); n <= k; n++ {
		w := p.high
		if n == 0 {
			g.tailHigh = p.high
		} else {
			w = g.tailW[n-1]
			if g.tailTop[n-1] {
				g.tailHigh = math.Min(2*w, p.high*pidtuneGateReach)
			}
			step := .01 * w
			if p.tau > 0 {
				step = math.Min(step, math.Pi/16/p.tau)
			}
			for _, m := range p.modes {
				step = math.Min(step, math.Max(math.Max(math.Abs(real(m)), math.Abs(w-imag(m))), 1e-6*w)/4)
			}
			w = math.Min(w+step, g.tailHigh)
		}
		g.tailW, g.tailG = append(g.tailW, w), append(g.tailG, g.at(w))
		g.tailTop = append(g.tailTop, w >= g.tailHigh)
	}
}

// pidtuneGateReach caps, as a multiple of the band's top, how far beyond
// extends the gate's scans.
const pidtuneGateReach = 1e12

// beyond returns a log grid from the top of a continuous gate's band up to
// where no crossover or |S| excess lies further up, and, for a loop whose
// gain levels off at L(j∞) ≠ 0, the limit of sup |S| as ω → ∞. A loop
// rolling off ends where pidTuningLoopTail bounds |L| by 1 − 1/maxSensitivity,
// however far a large Kp, or Kd/Tf, at a crossover backed off below the band
// pushes it; one leveling off ends where asymptote keeps L within half the
// distance its limit leaves to |L| = 1 and to |1+L| = 1/maxSensitivity. The
// grid is nil when the top already ends it, the bound is unknown (discrete
// bands reach π/Dt, ideal derivatives), or it does not end within
// pidtuneGateReach times the top, left to the band as before.
func (g *pidtuneGate) beyond(c *PID2) (grid []float64, level float64) {
	if !g.tailDone {
		g.tailDone = true
		if g.plant.Dt == 0 {
			g.tail = pidTuningLoopTail(g.plant)
		}
	}
	top, bound := g.p.high, 1-1/g.p.maxSensitivity
	if g.tail == nil || g.tail(c, top) <= bound {
		return nil, 0
	}
	end, reach := top, top*pidtuneGateReach
	if g.tail(c, math.Inf(1)) <= bound {
		for end < reach && g.tail(c, end) > bound {
			end *= 2
		}
		if g.tail(c, end) > bound {
			return nil, 0
		}
	} else {
		lim, dev, ok := g.asymptote(c)
		if !ok {
			return nil, 0
		}
		// Delay turns L(j∞) round the circle of radius |L(j∞)|.
		dist := 1 - math.Abs(lim)
		gap := dist - 1/g.p.maxSensitivity
		if g.p.tau == 0 {
			dist = math.Abs(1 + lim)
			gap = min(math.Abs(math.Abs(lim)-1), dist-1/g.p.maxSensitivity)
		}
		level = math.Inf(1)
		if dist > 0 {
			level = 1 / dist
		}
		if dev == nil || !(gap > 0) {
			return nil, level
		}
		for end < reach && !(dev(end) <= gap/2) {
			end *= 2
		}
		if !(dev(end) <= gap/2) {
			return nil, level
		}
		if end == top {
			return nil, level
		}
	}
	n := int(math.Ceil(math.Log10(end/top)*pidtuneGatePerDecade)) + 1
	grid = make([]float64, n)
	for i := range grid {
		grid[i] = math.Min(top*math.Pow(10, float64(i)/pidtuneGatePerDecade), end)
	}
	return grid, level
}

// asymptote returns L(j∞) of a continuous loop with c leveling off, and
// dev(ω) bounding |L(jω′) − L(j∞)| over ω′ ≥ ω: with C∞ = Kp + Kd/Tf and
// r = (|Ki| + |Kd|/Tf²)/ω bounding |C − C∞|, |L − L∞| ≤ (|C∞| + r)·|G − D|
// + r·|D|, |G − D| bounded by pidMarkovTail. An ideal derivative on a
// strictly proper plant levels off at Kd·CB, with dev nil; ok is false for
// one on a biproper plant, unbounded, or a descriptor plant.
func (g *pidtuneGate) asymptote(c *PID2) (lim float64, dev func(float64) float64, ok bool) {
	if g.plant.E != nil {
		return 0, nil, false
	}
	if c.Kd != 0 && !(c.Tf > 0) {
		if g.plant.D.At(0, 0) != 0 {
			return 0, nil, false
		}
		n, _ := g.plant.A.Dims()
		cb := 0.0
		for i := range n {
			cb += g.plant.C.At(0, i) * g.plant.B.At(i, 0)
		}
		return c.Kd * cb, nil, true
	}
	if g.markov == nil {
		g.markov = newPIDMarkovTail(g.plant)
	}
	d, k, kr := g.plant.D.At(0, 0), c.Kp, math.Abs(c.Ki)
	if c.Kd != 0 {
		k += c.Kd / c.Tf
		kr += math.Abs(c.Kd) / (c.Tf * c.Tf)
	}
	return k * d, func(w float64) float64 {
		r := kr / w
		return (math.Abs(k)+r)*g.markov.at(w, 0) + r*math.Abs(d)
	}, true
}

func pidtuneGateController(pid *PID) PID2 {
	return PID2{Kp: pid.Kp, Ki: pid.Ki, Kd: pid.Kd, Tf: pid.Tf, Dt: pid.Dt, IFormula: pid.IFormula, DFormula: pid.DFormula}
}

// crossExcess returns k ↦ |L(cross.w[k])|−1, sampling the plant on the
// first call.
func (g *pidtuneGate) crossExcess(c PID2) func(int) float64 {
	if g.cross.g == nil {
		for _, w := range g.cross.w {
			g.cross.g = append(g.cross.g, g.p.at(w))
		}
	}
	k := g.cross.feedback(c)
	return func(i int) float64 { return cmplx.Abs(g.cross.g[i]*k(i)) - 1 }
}

// sensitivityOver returns a sampled |S| above bound, or the largest sampled
// |S| when none is; as pidTuningSensitivityPeak, grid maxima above half the
// bound are refined by golden section. Unevaluable samples are skipped.
func (g *pidtuneGate) sensitivityOver(c PID2, bound float64) float64 {
	s := func(gw, k complex128) float64 { return 1 / cmplx.Abs(1+gw*k) }
	refine := func(w float64) float64 { return s(g.at(w), pidTuningFeedback(c, w)) }
	k := g.sens.feedback(c)
	peak := 0.0
	var v0, v1 float64
	for i, w := range g.sens.w {
		v := s(g.sens.g[i], k(i))
		if math.IsNaN(v) {
			v0, v1 = 0, 0
			continue
		}
		if i >= 2 && v1 >= v0 && v1 >= v && v1 > bound/2 {
			_, r := pidTuningGoldenMax(refine, g.sens.w[i-2], w)
			if r > bound {
				return r
			}
		}
		if v > bound {
			return v
		}
		peak = math.Max(peak, v)
		v0, v1 = v1, v
	}
	return peak
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
// A non-nil loop, whose model has no delay, stands for screen's
// closed-loop pole test.
func checkPidtuneStability(screen *pidStabilityScreen, loop *pidtuneLoop, eval *sisoEval, pid *PID, pidType PidtuneType, wc float64) error {
	var ok bool
	var err error
	if loop != nil {
		loop.c = PID2{Kp: pid.Kp, Ki: pid.Ki, Kd: pid.Kd, Tf: pid.Tf, Dt: pid.Dt, IFormula: pid.IFormula, DFormula: pid.DFormula}
		ok, err = loop.stable(screen, &loop.c)
	} else {
		var stable func(*PID2) (bool, error)
		stable, _, err = screen.at(wc, pidType)
		if errors.Is(err, ErrContinuousInternalDelay) {
			return nil
		}
		if err != nil || stable == nil {
			return err
		}
		ok, err = stable(&PID2{Kp: pid.Kp, Ki: pid.Ki, Kd: pid.Kd, Tf: pid.Tf, Dt: pid.Dt, IFormula: pid.IFormula, DFormula: pid.DFormula})
	}
	if err == nil {
		err = eval.err
	}
	if err != nil {
		return err
	}
	if !ok {
		return &pidtuneUnstableError{pidType: pidType, wc: wc}
	}
	return nil
}

// pidtuneUnstableError reports an unstable closed loop; most are dropped
// by the back-off, so the message is formatted only when read.
type pidtuneUnstableError struct {
	pidType PidtuneType
	wc      float64
}

func (e *pidtuneUnstableError) Error() string {
	return string(e.pidType) + " design at wc=" + strconv.FormatFloat(e.wc, 'g', -1, 64) + " gives an unstable closed loop: " + ErrPIDTuningTargetUnattainable.Error()
}

func (e *pidtuneUnstableError) Unwrap() error { return ErrPIDTuningTargetUnattainable }

// pidtuneLoop is pidStabilityScreen.closedLoopPoles for a standard SISO
// model without delays, with buffers reused across designs: the controller
// realization, I−D1·D2 solve and closed-loop A take PID.System's and
// Feedback's operations in their order, so the poles are bit for bit those
// of Feedback(model, C, −1) and the verdict is closedLoopPoles'; for SISO
// Feedback's two feedthrough solves agree, so one serves both. Anything
// else, errors included, goes to closedLoopPoles.
type pidtuneLoop struct {
	model *System
	n1    int
	c     PID2
	// p, i are the realizations controller returns.
	p, i System
	// ca, cb, cc, cd realize a continuous controller without derivative.
	ca, cb, cc, cd        *mat.Dense
	eye, ill, e21         *mat.Dense
	lu                    mat.LU
	b1e12, b1e12d2, block *mat.Dense
	byOrder               map[int]*pidtuneLoopBuffers
	eig                   mat.Eigen
	poles                 []complex128
}

// pidtuneLoopBuffers holds the blocks for an n2-state controller.
type pidtuneLoopBuffers struct {
	a, block2, b2e21, b2e21c1, b2e21d1, block3 *mat.Dense
}

func newPidtuneLoop(model *System) *pidtuneLoop {
	if model.HasDelay() || model.E != nil || model.validate() != nil {
		return nil
	}
	n1, m1, p1 := model.Dims()
	if m1 != 1 || p1 != 1 || !pidtuneLoopShaped(model, n1) {
		return nil
	}
	scalar := func() *mat.Dense { return mat.NewDense(1, 1, nil) }
	l := &pidtuneLoop{
		model: model, n1: n1, byOrder: map[int]*pidtuneLoopBuffers{},
		ca: scalar(), cb: scalar(), cc: scalar(), cd: scalar(),
		eye: eyeDense(1), ill: scalar(), e21: scalar(),
	}
	l.p = System{A: &mat.Dense{}, B: &mat.Dense{}, C: &mat.Dense{}, D: l.cd}
	l.i = System{A: l.ca, B: l.cb, C: l.cc, D: l.cd}
	if n1 > 0 {
		l.b1e12, l.b1e12d2, l.block = mat.NewDense(n1, 1, nil), mat.NewDense(n1, 1, nil), mat.NewDense(n1, n1, nil)
	}
	return l
}

// pidtuneLoopShaped reports whether sys's matrices are stored at their
// SISO n-state dimensions, so no product meets mulDims's empty operands.
func pidtuneLoopShaped(sys *System, n int) bool {
	shaped := func(m *mat.Dense, r, c int) bool {
		if r == 0 || c == 0 {
			return true
		}
		mr, mc := m.Dims()
		return mr == r && mc == c
	}
	return sys.A != nil && sys.B != nil && sys.C != nil && sys.D != nil &&
		shaped(sys.A, n, n) && shaped(sys.B, n, 1) && shaped(sys.C, 1, n) && shaped(sys.D, 1, 1)
}

func (l *pidtuneLoop) buffers(n2 int) *pidtuneLoopBuffers {
	if b, ok := l.byOrder[n2]; ok {
		return b
	}
	n1 := l.n1
	b := &pidtuneLoopBuffers{a: newDense(n1+n2, n1+n2)}
	if n2 > 0 {
		b.b2e21, b.b2e21d1, b.block3 = mat.NewDense(n2, 1, nil), mat.NewDense(n2, 1, nil), mat.NewDense(n2, n2, nil)
		if n1 > 0 {
			b.block2, b.b2e21c1 = mat.NewDense(n1, n2, nil), mat.NewDense(n2, n1, nil)
		}
	}
	l.byOrder[n2] = b
	return b
}

// controller returns c's realization as PID.System gives it, the
// derivative-free continuous ones (whose entries are copies of gains) in
// l's buffers.
func (l *pidtuneLoop) controller(c *PID2) (*System, error) {
	if c.Dt != 0 || c.Kd != 0 || validatePID(c.Kp, c.Ki, c.Kd, c.Tf, c.Dt, c.IFormula, c.DFormula) != nil {
		return (&PID{Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf, Dt: c.Dt, IFormula: c.IFormula, DFormula: c.DFormula}).System()
	}
	l.cd.Set(0, 0, c.Kp)
	if c.Ki == 0 {
		return &l.p, nil
	}
	l.ca.Set(0, 0, 0)
	l.cb.Set(0, 0, 1)
	l.cc.Set(0, 0, c.Ki)
	return &l.i, nil
}

func (l *pidtuneLoop) stable(screen *pidStabilityScreen, c *PID2) (bool, error) {
	cs, err := l.controller(c)
	if err != nil {
		return screen.closedLoopPoles(c)
	}
	n2, m2, p2 := cs.Dims()
	if m2 != 1 || p2 != 1 || cs.E != nil || cs.HasDelay() || cs.Dt != l.model.Dt || !pidtuneLoopShaped(cs, n2) {
		return screen.closedLoopPoles(c)
	}
	plant := l.model
	l.ill.Mul(plant.D, cs.D)
	l.ill.Scale(-1, l.ill)
	l.ill.Sub(l.eye, l.ill)
	l.lu.Factorize(l.ill)
	if nearSingularCondition(l.lu.Cond()) || l.lu.SolveTo(l.e21, false, l.eye) != nil {
		return screen.closedLoopPoles(c)
	}
	e21 := l.e21
	n1, n := l.n1, l.n1+n2
	if n == 0 {
		return true, nil
	}
	const sign = 1.0

	b := l.buffers(n2)
	if n1 > 0 {
		l.b1e12.Mul(plant.B, e21)
		l.b1e12d2.Mul(l.b1e12, cs.D)
		l.block.Mul(l.b1e12d2, plant.C)
		l.block.Scale(sign, l.block)
		setBlock(b.a, 0, 0, plant.A)
		subBlock(b.a, 0, 0, l.block)
		if n2 > 0 {
			b.block2.Mul(l.b1e12, cs.C)
			b.block2.Scale(-sign, b.block2)
			setBlock(b.a, 0, n1, b.block2)
		}
	}
	if n2 > 0 {
		b.b2e21.Mul(cs.B, e21)
		if n1 > 0 {
			b.b2e21c1.Mul(b.b2e21, plant.C)
			setBlock(b.a, n1, 0, b.b2e21c1)
		}
		setBlock(b.a, n1, n1, cs.A)
		b.b2e21d1.Mul(b.b2e21, plant.D)
		b.block3.Mul(b.b2e21d1, cs.C)
		b.block3.Scale(sign, b.block3)
		subBlock(b.a, n1, n1, b.block3)
	}

	if err := requireFiniteDense("Poles", "A", b.a); err != nil {
		return false, err
	}
	if !l.eig.Factorize(b.a, mat.EigenNone) {
		return false, fmt.Errorf("Poles: eigenvalue decomposition failed to converge: %w", ErrSchurFailed)
	}
	if cap(l.poles) < n {
		l.poles = make([]complex128, n)
	}
	l.poles = l.eig.Values(l.poles[:n])
	return pidTuningPolesStable(l.poles, b.a, plant, cs), nil
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
	wc, _, err := autoCrossoverFreq(plant, wcTarget, pm)
	return wc, err
}

// autoCrossoverFreq is findCrossoverFreq, also reporting whether the
// crossover came from plantCrossoverFreq's pole fallback.
func autoCrossoverFreq(plant *System, wcTarget, pm float64) (float64, bool, error) {
	wc, src, err := autoCrossoverSource(plant, wcTarget, pm)
	return wc, src != pidtuneCrossGain, err
}

// pidtuneCrossover names where plantCrossoverFreq took its crossover from.
type pidtuneCrossover int

const (
	// pidtuneCrossGain: a gain crossover, the bandwidth, an integrating
	// plant's pick, or an explicit wc.
	pidtuneCrossGain pidtuneCrossover = iota
	// pidtuneCrossRolloff: pidtuneRolloffCrossover below the bandwidth.
	pidtuneCrossRolloff
	// pidtuneCrossPole: half the slowest pole of a plant without bandwidth.
	pidtuneCrossPole
	// pidtuneCrossZeroDC: half the slowest pole of a plant with zero DC
	// gain, a k·sᵐ high-pass plant.
	pidtuneCrossZeroDC
)

func autoCrossoverSource(plant *System, wcTarget, pm float64) (float64, pidtuneCrossover, error) {
	if wcTarget > 0 {
		return wcTarget, pidtuneCrossGain, nil
	}
	wc, src, err := plantCrossoverSource(plant)
	if err != nil {
		return 0, pidtuneCrossGain, err
	}
	tau := sisoLoopDelay(plant)
	if plant.Dt > 0 {
		tau *= plant.Dt
	}
	if tau > 0 {
		wc = math.Min(wc, (180-pm)*math.Pi/180/(2*tau))
	}
	return wc, src, nil
}

func plantCrossoverFreq(plant *System) (float64, error) {
	wc, _, err := plantCrossoverFreqFrom(plant)
	return wc, err
}

// plantCrossoverFreqFrom is plantCrossoverFreq, also reporting whether the
// crossover came from a pole fallback: G has no gain crossover and a
// finite DC gain that is 0 (a zero at the integrator, whose rounded DC gain
// would give a -3 dB bandwidth at rounding level) or has no -3 dB
// bandwidth, as for a high-pass plant with |G| < 1, both taking half the
// slowest pole, or one pidtuneRolloffCrossover finds set by zeros near the
// integrator of a stable plant without a continuous delay; an unstable one
// needs the gain the bandwidth gives, which a crossover capped by its roots
// cannot reach, and a delay's lag at the capped crossover leaves a PDF
// needing phase lag where the delay-capped bandwidth has a robust design.
// A DC gain rounded to 0 on a pidtuneRoundedIntegrator plant is infinite.
func plantCrossoverFreqFrom(plant *System) (float64, bool, error) {
	wc, src, err := plantCrossoverSource(plant)
	return wc, src != pidtuneCrossGain, err
}

func plantCrossoverSource(plant *System) (float64, pidtuneCrossover, error) {
	all, err := AllMargin(plant)
	if err != nil {
		return 0, pidtuneCrossGain, err
	}
	if len(all.GainCrossFreqs) > 0 {
		return all.GainCrossFreqs[0], pidtuneCrossGain, nil
	}
	dc, err := plant.DCGain()
	if err == nil && (math.IsInf(dc.At(0, 0), 0) || dc.At(0, 0) == 0 && pidtuneRoundedIntegrator(plant)) {
		wc, err := pidtuneIntegratingCrossover(plant)
		return wc, pidtuneCrossGain, err
	}
	pole := pidtuneCrossZeroDC
	if err != nil || dc.At(0, 0) != 0 && !pidtuneZeroAtIntegrator(plant) {
		pole = pidtuneCrossPole
		bw, err := Bandwidth(plant, -3)
		if err != nil {
			return 0, pidtuneCrossGain, err
		}
		if bw > 0 && !math.IsInf(bw, 0) {
			if stable, err := plant.IsStable(); err == nil && stable && !(plant.Dt == 0 && plant.HasDelay()) {
				if wc, ok := pidtuneRolloffCrossover(plant, bw); ok {
					return wc, pidtuneCrossRolloff, nil
				}
			}
			return bw, pidtuneCrossGain, nil
		}
	}

	poles, err := plant.Poles()
	if err != nil {
		return 0, pidtuneCrossGain, err
	}
	at, tol := pidtuneIntegratorCluster(plant, poles)
	if w := pidtuneSlowestRoot(plant, poles, at, tol); !math.IsInf(w, 1) {
		w *= 0.5
		if plant.Dt > 0 {
			w = math.Min(w, 0.1*math.Pi/plant.Dt)
		}
		return w, pole, nil
	}

	return 0, pidtuneCrossGain, fmt.Errorf("no gain crossover, bandwidth or nonzero pole to infer a crossover from; pass wc > 0: %w", ErrInvalidArgument)
}

// A -3 dB bandwidth beyond pidtuneRolloffFactor times every finite pole and
// zero natural frequency lies on G's c/s^r asymptote, where G falls to
// 0.708·|G(0)| only because zeros attenuate G(0). When |G| also peaks more
// than pidtuneRolloffGain times above its value there, as for a zero near
// the integrator, gains placing the crossover at the bandwidth exceed the
// plant's scale by as much, up to 1e18, leaving a closed-loop pole at the
// zero; a milder rise, as from a mid-band or non-minimum-phase zero, keeps
// the bandwidth.
const (
	pidtuneRolloffFactor = 10
	pidtuneRolloffGain   = 1e3
)

// pidtuneRolloffCrossover returns the crossover for a plant whose -3 dB
// bandwidth bw is set by DC attenuation, as pidtuneRolloffFactor describes:
// the first back-off step bw·10^(−k/pidtuneBackoffPerDecade) at or below
// pidtuneRolloffFactor times the fastest root, so the back-off from there
// visits the crossovers a back-off from bw would below it. ok is false when
// bw is a passband edge.
func pidtuneRolloffCrossover(plant *System, bw float64) (wc float64, ok bool) {
	limit := pidtuneRolloffFactor * pidtuneFastestRoot(plant)
	if !(bw > limit) || limit == 0 {
		return 0, false
	}
	const decades, perDecade = 9, 20
	omega := make([]float64, decades*perDecade+2)
	for i := range len(omega) - 1 {
		omega[i] = limit * math.Pow(10, -float64(i)/perDecade)
	}
	omega[len(omega)-1] = bw
	resp, err := plant.FreqResponse(omega)
	if err != nil {
		return 0, false
	}
	edge, peak := cmplx.Abs(resp.At(len(omega)-1, 0, 0)), 0.0
	for i := range len(omega) - 1 {
		peak = max(peak, cmplx.Abs(resp.At(i, 0, 0)))
	}
	if !(peak > pidtuneRolloffGain*edge) {
		return 0, false
	}
	k := math.Ceil(pidtuneBackoffPerDecade*math.Log10(bw/limit) - 1e-9)
	return bw * math.Pow(10, -k/pidtuneBackoffPerDecade), true
}

// pidtuneFastestRoot returns the largest natural frequency (|ln z|/Dt for
// discrete plants) of a finite pole or zero of plant, +Inf for a root at
// z = 0 or when the roots are unavailable.
func pidtuneFastestRoot(plant *System) float64 {
	model, err := finiteDimensionalModel(plant, "Pidtune")
	if err != nil {
		return math.Inf(1)
	}
	poles, err := model.Poles()
	if err != nil {
		return math.Inf(1)
	}
	zeros, err := model.Zeros()
	if err != nil {
		return math.Inf(1)
	}
	fast := 0.0
	for _, r := range slices.Concat(poles, zeros) {
		w := cmplx.Abs(r)
		if plant.Dt > 0 {
			w = cmplx.Abs(cmplx.Log(r)) / plant.Dt
		}
		if math.IsNaN(w) {
			return math.Inf(1)
		}
		fast = max(fast, w)
	}
	return fast
}

// pidtuneIntegratingCrossover picks the crossover of an integrating plant
// with no gain crossover, whose bandwidth is undefined: half the slowest
// natural frequency of a pole, else of a zero, away from s = 0 (z = 1),
// capped for discrete plants a decade below Nyquist, which is also the pick
// when there is neither.
func pidtuneIntegratingCrossover(plant *System) (float64, error) {
	model, err := finiteDimensionalModel(plant, "Pidtune")
	if err != nil {
		return 0, err
	}
	poles, err := model.Poles()
	if err != nil {
		return 0, err
	}
	at, tol := pidtuneIntegratorCluster(plant, poles)
	if tol == 0 {
		tol = pidtuneDisplacedIntegrator(plant, poles, at)
	}
	w := pidtuneSlowestRoot(plant, poles, at, tol)
	if math.IsInf(w, 1) {
		zeros, err := model.Zeros()
		if err != nil {
			return 0, err
		}
		at, tol = pidtuneIntegratorCluster(plant, zeros)
		w = pidtuneSlowestRoot(plant, zeros, at, tol)
	}
	w *= 0.5
	if plant.Dt > 0 {
		w = math.Min(w, 0.1*math.Pi/plant.Dt)
	}
	if math.IsInf(w, 1) {
		return 0, fmt.Errorf("integrating plant without gain crossover or nonintegrating pole or zero to infer a crossover from; pass wc > 0: %w", ErrInvalidArgument)
	}
	return w, nil
}

// pidtuneIntegratorCluster returns the integrator location, s = 0 or z = 1,
// and a radius holding the pidtuneClusterSize roots it counts there, within
// 100·ε^(1/m)·scale of it, ε the machine epsilon: rounding splits an m-fold
// root by about ε^(1/m) relative to the root scale (1 for z = 1, the largest
// root magnitude for s = 0).
func pidtuneIntegratorCluster(plant *System, roots []complex128) (complex128, float64) {
	at, scale := complex128(0), 0.0
	if plant.Dt > 0 {
		at, scale = 1, 1
	} else {
		for _, r := range roots {
			scale = max(scale, cmplx.Abs(r))
		}
	}
	m := pidtuneClusterSize(roots, at, scale)
	if m == 0 {
		return at, 0
	}
	return at, pidtuneClusterRadius(m, scale)
}

// pidtuneDisplacedIntegrator returns, for the poles of a plant whose DC
// gain is infinite but which have no pidtuneIntegratorCluster at at, the
// distance of the pole nearest at when it lies within
// pidtuneClusterRadius(2, scale), else 0: an ill-conditioned eigenproblem
// moves a simple integrator pole beyond 100·ε·scale.
func pidtuneDisplacedIntegrator(plant *System, poles []complex128, at complex128) float64 {
	scale, near := 0.0, math.Inf(1)
	if plant.Dt > 0 {
		scale = 1
	}
	for _, p := range poles {
		if plant.Dt == 0 {
			scale = max(scale, cmplx.Abs(p))
		}
		near = min(near, cmplx.Abs(p-at))
	}
	if near <= pidtuneClusterRadius(2, scale) {
		return near
	}
	return 0
}

// pidtuneRoundedIntegrator reports whether a discrete plant whose DCGain is
// 0 is an integrating plant: no zero at the integrator and a pole displaced
// from z = 1 beyond pidtuneIntegratorCluster but within
// pidtuneDisplacedIntegrator. Its true DC gain is infinite; an
// ill-conditioned (z−1)·∏(z−pᵢ) realization rounds C(I−A)⁻¹B to 0, and the
// pole fallback would take the displaced pole as the slowest, a crossover
// ~1e-8 where (zI−A) is singular. A clustered integrator is already
// excluded there.
func pidtuneRoundedIntegrator(plant *System) bool {
	if plant.Dt == 0 || pidtuneZeroAtIntegrator(plant) {
		return false
	}
	model, err := finiteDimensionalModel(plant, "Pidtune")
	if err != nil {
		return false
	}
	poles, err := model.Poles()
	if err != nil {
		return false
	}
	if _, tol := pidtuneIntegratorCluster(plant, poles); tol > 0 {
		return false
	}
	return pidtuneDisplacedIntegrator(plant, poles, 1) > 0
}

// pidtuneClusterSize returns the largest m such that the m roots nearest at
// lie within pidtuneClusterRadius(m, scale) of it and their mean lies within
// pidtuneClusterCentroid·scale of it. Rounding splits an m-fold root into a
// ring whose mean, the trace, stays accurate; distinct roots swept in by the
// radius, which grows with m, move the mean off at.
func pidtuneClusterSize(roots []complex128, at complex128, scale float64) int {
	r := slices.Clone(roots)
	slices.SortFunc(r, func(a, b complex128) int {
		return cmp.Compare(cmplx.Abs(a-at), cmplx.Abs(b-at))
	})
	var sum complex128
	sums := make([]complex128, len(r))
	for i, x := range r {
		sum += x - at
		sums[i] = sum
	}
	for m := len(r); m > 0; m-- {
		if cmplx.Abs(r[m-1]-at) <= pidtuneClusterRadius(m, scale) &&
			cmplx.Abs(sums[m-1])/float64(m) <= pidtuneClusterCentroid*scale {
			return m
		}
	}
	return 0
}

const pidtuneClusterCentroid = 1e-6

// pidtuneClusterRadius is 100·ε^(1/m)·scale, capped at 1e-2·scale so that
// a cluster never reaches the roots that set scale.
func pidtuneClusterRadius(m int, scale float64) float64 {
	return min(100*math.Pow(0x1p-52, 1/float64(m)), 1e-2) * scale
}

// pidtunePlantPoint evaluates evalPlantAt(plant, w) bit for bit: it
// remembers the last frequency and, from the second frequency on, for a
// standard SISO model without internal delays, keeps FreqResponse's
// single-point solver, whose workspaces evalInto overwrites at each point.
// A nil pidtunePlantPoint calls evalPlantAt.
type pidtunePlantPoint struct {
	fe      frequencyEvaluator
	solve   func(frequencyPoint, []complex128) error
	delay   *mat.Dense
	dst     [1]complex128
	lastW   float64
	lastH   complex128
	lastErr error
	calls   int
}

func (p *pidtunePlantPoint) at(plant *System, w float64) (complex128, error) {
	if p == nil {
		return evalPlantAt(plant, w)
	}
	if p.calls > 0 && math.Float64bits(w) == math.Float64bits(p.lastW) {
		return p.lastH, p.lastErr
	}
	if p.calls == 1 && !plant.HasInternalDelay() && !plant.IsDescriptor() {
		p.fe = newFrequencyEvaluator(plant)
		if bd, ok := p.fe.pointSolver(1).(*balancedDense); ok && p.fe.m == 1 && p.fe.p == 1 {
			p.solve = bd.evalInto
			p.delay = effectiveIODelayMatrix(plant, 1, 1, true)
		}
	}
	p.calls++
	var h complex128
	var err error
	if p.solve == nil {
		h, err = evalPlantAt(plant, w)
	} else {
		pt := p.fe.pointAt(w)
		if err = evalWithPoleLimit(p.solve, plant, pt, p.dst[:]); err == nil {
			if p.delay != nil {
				applyIODelayMatrixAtS(plant, pt, p.dst[:], 1, 1, p.delay)
			}
			h = p.dst[0]
		}
	}
	p.lastW, p.lastH, p.lastErr = w, h, err
	return h, err
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
	// lagFloor raises a PDF phase below the family's range to its floor.
	lagFloor bool
	pd       pidtunePDMode
	pidf     pidtunePIDFMode
}

// pidtunePDMode selects computePD's design.
type pidtunePDMode int

const (
	// pidtunePDLegacy: phase and gain placed on the ideal derivative, the
	// Td/10 filter then added, so |L(wc)| ≠ 1, as the automatic crossover's
	// gate was tuned on.
	pidtunePDLegacy pidtunePDMode = iota
	// pidtunePDPlaced: phase and gain placed on the realized Td/10 response
	// (placedPD).
	pidtunePDPlaced
	// pidtunePDLegacyTd: the legacy Td, with Kp for |L(wc)| = 1 on the
	// realized response.
	pidtunePDLegacyTd
)

// pidtunePIDFMode selects computePIDF's design.
type pidtunePIDFMode int

const (
	// pidtunePIDFIterate: Tf = Td/10, errPidtunePIDFMiss when the
	// iteration misses the target phase.
	pidtunePIDFIterate pidtunePIDFMode = iota
	// pidtunePIDFKeep: Tf = Td/10, returning a missed design.
	pidtunePIDFKeep
	// pidtunePIDFDefaultTf: Tf = 0.1/wc as PDF and TunePID, solved exactly.
	pidtunePIDFDefaultTf
)

// errPidtunePIDFMiss reports that the Td/10 filter iteration missed the
// target phase.
var errPidtunePIDFMiss = fmt.Errorf("PIDF with Tf = Td/10 misses the target phase: %w", ErrPIDTuningTargetUnattainable)

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
// With stableFilter, a discrete forward-Euler derivative whose filter pole
// 1−Dt/Tf leaves the unit circle is redesigned with backward Euler.
func computePIDGains(plant *System, pidType PidtuneType, wc, pmDeg, sign float64, stableFilter bool) (*PID, error) {
	return pidtuneGains(plant, nil, pidType, wc, pmDeg, sign, stableFilter, false, pidtunePDLegacy, pidtunePIDFKeep)
}

// pidtuneGains is computePIDGains, lagFloor, pd and pidf aside,
// evaluating the plant with at.
func pidtuneGains(plant *System, at *pidtunePlantPoint, pidType PidtuneType, wc, pmDeg, sign float64, stableFilter, lagFloor bool, pd pidtunePDMode, pidf pidtunePIDFMode) (*PID, error) {
	pid, err := computePIDGainsPositive(plant, at, pidType, wc, pmDeg, sign, stableFilter, lagFloor, pd, pidf)
	if err != nil {
		return nil, err
	}
	pid.Dt = plant.Dt
	pid.Kp *= sign
	pid.Ki *= sign
	pid.Kd *= sign
	return pid, nil
}

func computePIDGainsPositive(plant *System, at *pidtunePlantPoint, pidType PidtuneType, wc, pmDeg, sign float64, stableFilter, lagFloor bool, pd pidtunePDMode, pidf pidtunePIDFMode) (*PID, error) {
	if plant.Dt > 0 && wc >= math.Pi/plant.Dt {
		return nil, fmt.Errorf("crossover %g must be below the Nyquist frequency %g: %w", wc, math.Pi/plant.Dt, ErrInvalidArgument)
	}
	h, err := at.at(plant, wc)
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
	terms := pidtuneTerms{wc: wc, dt: plant.Dt, dFormula: dFormula, lagFloor: lagFloor, pd: pd, pidf: pidf}
	pid, err := pidtuneFamily(pidType, terms, magP, phiC)
	if !stableFilter || plant.Dt == 0 || dFormula != ForwardEuler {
		return pid, err
	}
	filter := pid
	if errors.Is(err, errPidtunePIDFMiss) {
		keep := terms
		keep.pidf = pidtunePIDFKeep
		filter, _ = pidtuneFamily(pidType, keep, magP, phiC)
	}
	if filter != nil && filter.Kd != 0 && filter.Tf <= plant.Dt/2 {
		// The forward-Euler filter pole 1−Dt/Tf is outside the unit circle.
		terms.dFormula = BackwardEuler
		pid, err = pidtuneFamily(pidType, terms, magP, phiC)
	}
	return pid, err
}

// pidtunePDFLagFloor is the PDF phase, in radians, used in place of a
// requested phase lag: the least lead of a PDF with a small Kd, as computePD
// floors its phase.
const pidtunePDFLagFloor = 0.05

// errPidtunePDFLag reports that a PDF would need phase lag at the crossover.
var errPidtunePDFLag = fmt.Errorf("PDF needs phase lag: %w", ErrPIDTuningTargetUnattainable)

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
		d := terms.derivative(pid.Tf)
		phiCRad := phiC * math.Pi / 180
		// Same-sign PDF phase range: [0, arg d].
		if phiCRad < 0 && terms.lagFloor && cmplx.Phase(d) > 2*pidtunePDFLagFloor {
			phiCRad = pidtunePDFLagFloor
		}
		target := cmplx.Rect(1/magP, phiCRad)
		pid.Kd = imag(target) / imag(d)
		pid.Kp = real(target) - pid.Kd*real(d)
		if pid.Kp < 0 || pid.Kd < 0 {
			err := ErrPIDTuningTargetUnattainable
			if phiCRad < 0 {
				err = errPidtunePDFLag
			}
			return nil, fmt.Errorf("requested phase is unattainable by a same-sign PDF at wc=%g: %w", terms.wc, err)
		}
	case PidtunePID:
		computePID(pid, terms, magP, phiC)

	case PidtunePIDF:
		if err := computePIDF(pid, terms, magP, phiC); err != nil {
			return nil, err
		}
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

// computePD designs a Td/10-filtered PD as terms.pd selects; the legacy
// design places phase and gain on the ideal derivative, leaving |L(wc)| ≠ 1
// (≈ 0.49 at the upper clamp). Callers requesting an ideal PD must
// explicitly set Tf=0.
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

	if terms.pd == pidtunePDPlaced {
		placedPD(pid, terms, magP, phiC)
		return
	}

	d := terms.derivative(0)
	tan := math.Tan(phiCRad)
	Td := tan / (imag(d) - real(d)*tan)
	if Td <= 0 {
		Td = 0.1 / terms.wc
	}

	pid.Tf = Td / 10
	if terms.pd == pidtunePDLegacyTd {
		d = terms.derivative(pid.Tf)
	}
	pid.Kp = 1.0 / (magP * cmplx.Abs(1+complex(Td, 0)*d))
	pid.Kd = pid.Kp * Td
}

// placedPD sets a Td/10-filtered PD whose realized phase at wc is phiC
// degrees clamped to [0.05 rad, the family's most lead] (56.4° for a
// continuous plant, at Td·wc = 1/√0.11), with Kp for |C(jwc)| = 1/magP, so
// |L(wc)| = 1 as MATLAB places the crossover. computePIDF clamps its phase
// on the ideal derivative instead.
func placedPD(pid *PID, terms pidtuneTerms, magP, phiC float64) {
	integ := pidtuneIntegrator(terms.dFormula, terms.wc, terms.dt)
	c := func(logX float64) complex128 {
		td := math.Exp(logX) / terms.wc
		return 1 + complex(td, 0)/(complex(td/10, 0)+integ)
	}
	lead := func(logX float64) float64 { return cmplx.Phase(c(logX)) }

	const lo, hi = -18.0, 14.0
	const g = 0.6180339887498949
	a, b := lo, hi
	x1, x2 := b-g*(b-a), a+g*(b-a)
	f1, f2 := lead(x1), lead(x2)
	for b-a > 1e-10 {
		if f1 < f2 {
			a, x1, f1 = x1, x2, f2
			x2 = a + g*(b-a)
			f2 = lead(x2)
		} else {
			b, x2, f2 = x2, x1, f1
			x1 = b - g*(b-a)
			f1 = lead(x1)
		}
	}
	logX := (a + b) / 2
	most := lead(logX)
	phiCRad := math.Max(phiC*math.Pi/180, 0.05)
	if most > phiCRad {
		a, b = lo, logX
		for b-a > 1e-13 {
			m := (a + b) / 2
			if lead(m) < phiCRad {
				a = m
			} else {
				b = m
			}
		}
		logX = (a + b) / 2
	}
	if !(lead(logX) > 0) {
		logX = math.Log(0.1)
	}
	Td := math.Exp(logX) / terms.wc

	pid.Tf = Td / 10
	pid.Kp = 1.0 / (magP * cmplx.Abs(c(logX)))
	pid.Kd = pid.Kp * Td
}

// pidFromPhase solves Kp + Ki*I + Kd*D = target with Ki = Kp*wc/b and D the
// derivative filtered by tf, relaxing the integral ratio and finally dropping
// Kd when the derivative turns negative.
func pidFromPhase(terms pidtuneTerms, magP, phiCRad, b, tf float64) (Kp, Ki, Kd float64) {
	target := cmplx.Rect(1/magP, phiCRad)
	i, d := terms.integral(), terms.derivative(tf)
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

	pid.Kp, pid.Ki, pid.Kd = pidFromPhase(terms, magP, phiCRad, 5.0, 0)
}

// computePIDF designs a PIDF for the phase phiC, clamped to (−90°, 90°), as
// terms.pidf selects; the Td/10 iteration can miss it when the filter lag
// outgrows the lead the derivative adds.
func computePIDF(pid *PID, terms pidtuneTerms, magP, phiC float64) error {
	wc := terms.wc
	phiCRad := phiC * math.Pi / 180
	if phiCRad >= math.Pi/2 {
		phiCRad = math.Pi/2 - 0.05
	}
	if phiCRad <= -math.Pi/2 {
		phiCRad = -math.Pi/2 + 0.05
	}

	i := terms.integral()
	missed := func(kp, ki, kd, tf float64) bool {
		c := complex(kp, 0) + complex(ki, 0)*i + complex(kd, 0)*terms.derivative(tf)
		return !(math.Abs(cmplx.Phase(c)-phiCRad) <= pidtuneMarginTol*math.Pi/180)
	}
	if terms.pidf == pidtunePIDFDefaultTf {
		tf := .1 / wc
		pid.Kp, pid.Ki, pid.Kd = pidFromPhase(terms, magP, phiCRad, 5.0, tf)
		pid.Tf = tf
		if pid.Kp < 0 || pid.Ki < 0 || pid.Kd < 0 || missed(pid.Kp, pid.Ki, pid.Kd, tf) {
			return fmt.Errorf("PIDF with Tf = 0.1/wc misses the target phase at wc=%g: %w", wc, ErrPIDTuningTargetUnattainable)
		}
		return nil
	}
	// Iteratively design PID then compensate for filter (Tf=Td/10) phase loss.
	phiAdj := phiCRad
	var Kp, Ki, Kd float64
	for range 100 {
		Kp, Ki, Kd = pidFromPhase(terms, magP, phiAdj, 5.0, 0)
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
	if terms.pidf == pidtunePIDFIterate && missed(pid.Kp, pid.Ki, pid.Kd, pid.Tf) {
		return errPidtunePIDFMiss
	}
	return nil
}
