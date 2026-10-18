package controlsys

import (
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
// ErrInvalidArgument. The loop crossover and phase margin are placed using
// the realized controller response: discrete plants use the discrete PID
// terms, a discrete PID without filter uses a backward-Euler derivative to
// stay causal, and a discrete derivative whose forward-Euler filter pole
// 1−Dt/Tf would leave the unit circle (Tf ≤ Dt/2) uses backward Euler unless
// only the forward-Euler design gives a stable loop that keeps the margin at
// every gain crossover and |S| ≤ 4. An explicitly
// requested phase margin the family cannot reach at wc returns
// ErrPIDTuningTargetUnattainable instead of a controller with a different
// margin; without one, a PDF needing phase lag at wc takes its least lead,
// a small Kd, as MATLAB does. A PIDF takes Tf = Td/10 unless that filter's
// lag keeps the margin from being reached at wc; it then takes Tf = 0.1/wc,
// else the off-target Td/10 design, whichever first keeps the margin at
// every gain crossover and |S| ≤ 4. Every gain takes one sign: that of the DC
// gain for a stable plant, and for an integrating or unstable plant the
// sign stabilization forces
// (see pidtuneGainSign), so Pidtune(−G) mirrors Pidtune(G); where the force
// lapses, as for an ideal derivative large enough to flip the closed-loop
// polynomial's leading coefficient, the other sign is tried when the forced
// one fails. A design whose closed loop is unstable by its poles (delay
// absorbed for discrete plants) or by a Nyquist test (continuous delays) is
// rejected; loops outside both tests are not screened. With wc = 0 a
// rejected design is retried at crossovers lowered up to three decades, or
// to a hundredth of the slowest non-integrating pole's natural frequency
// when that is lower, six decades at most (not for an integrating or
// unstable plant with a continuous delay),
// accepting the first that passes, keeps the phase margin at every gain
// crossover and keeps |S| ≤ 4; otherwise, and for an explicit wc, the
// result is ErrPIDTuningTargetUnattainable. TunePID offers bounded search
// with achieved-target evidence.
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
	sign, rule := pidtuneGainSign(plant, eval.at, wc, hasI)
	if eval.err != nil {
		return nil, fmt.Errorf("Pidtune: %w", eval.err)
	}
	screen, err := newPIDStabilityScreen(plant, eval, 1)
	if err != nil {
		return nil, fmt.Errorf("Pidtune: %w", err)
	}
	run := &pidtuneRun{plant: plant, screen: screen, loop: newPidtuneLoop(screen.model), eval: eval, pidType: pidType, pm: pm, explicit: explicit, auto: auto, wc: wc, mirror: rule != pidtuneSignFixed}

	pid, first := run.design(wc, sign, false)
	if first == nil {
		return pid, nil
	}
	// An unstable plant needs bandwidth that a continuous delay already
	// caps, and each back-off step would run a Nyquist test.
	if !auto || !errors.Is(first, ErrPIDTuningTargetUnattainable) ||
		rule != pidtuneSignFixed && plant.Dt == 0 && plant.HasDelay() {
		return nil, fmt.Errorf("Pidtune: %w", first)
	}
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
	// floor caches backoffFloor.
	floor    float64
	explicit bool
	auto     bool
	// mirror tries the design against the chosen gain sign after the
	// signed one fails.
	mirror bool
	gate   *pidtuneGate
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
	pid, err := r.mirrored(wc, sign, gated, false)
	if err != nil && !r.explicit && errors.Is(err, errPidtunePDFLag) {
		if alt, altErr := r.mirrored(wc, sign, gated || r.auto, true); altErr == nil {
			return alt, nil
		}
	}
	return pid, err
}

// mirrored designs for sign·G, then for −sign·G as design describes; with
// lagFloor a PDF needing phase lag gets the least lead instead.
func (r *pidtuneRun) mirrored(wc, sign float64, gated, lagFloor bool) (*PID, error) {
	pid, err := r.signed(wc, sign, gated, lagFloor)
	if r.mirror && errors.Is(err, ErrPIDTuningTargetUnattainable) && r.screened(wc) {
		if alt, altErr := r.signed(wc, -sign, gated, lagFloor); altErr == nil {
			return alt, nil
		}
	}
	return pid, err
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
// the back-off gate: a missed design can cross over elsewhere with almost
// no margin.
func (r *pidtuneRun) checked(wc, sign float64, gated, stableFilter, lagFloor bool) (*PID, error) {
	pid, err := pidtuneGains(r.plant, &r.at, r.pidType, wc, r.pm, sign, stableFilter, lagFloor, pidtunePIDFIterate)
	if err == nil {
		return pid, r.check(pid, wc, gated)
	}
	if !errors.Is(err, errPidtunePIDFMiss) {
		return nil, err
	}
	for _, mode := range []pidtunePIDFMode{pidtunePIDFDefaultTf, pidtunePIDFKeep} {
		alt, altErr := pidtuneGains(r.plant, &r.at, r.pidType, wc, r.pm, sign, stableFilter, lagFloor, mode)
		if altErr == nil && r.check(alt, wc, true) == nil {
			return alt, nil
		}
	}
	return nil, err
}

// check screens pid; the gate, cheaper than a Nyquist test of a continuous
// delay, runs before it there and after the pole test otherwise.
func (r *pidtuneRun) check(pid *PID, wc float64, gated bool) error {
	if r.explicit {
		if err := checkPidtuneMargin(r.plant, pid, wc, r.pm); err != nil {
			return err
		}
	}
	gateFirst := r.plant.Dt == 0 && r.plant.HasDelay()
	if gated && gateFirst {
		if err := r.gated(pid, wc); err != nil {
			return err
		}
	}
	if err := checkPidtuneStability(r.screen, r.loop, r.eval, pid, r.pidType, wc); err != nil {
		return err
	}
	if gated && !gateFirst {
		return r.gated(pid, wc)
	}
	return nil
}

func (r *pidtuneRun) backoffFloor() float64 {
	if r.floor == 0 {
		r.floor = pidtuneBackoffFloor(r.plant, r.wc)
	}
	return r.floor
}

func (r *pidtuneRun) gated(pid *PID, wc float64) error {
	if r.gate == nil {
		r.gate = newPidtuneGate(r.plant, r.screen.model, r.eval, r.wc, r.backoffFloor())
	}
	return r.gate.check(pid, r.pidType, wc, r.pm)
}

// pidtuneSignRule says how pidtuneGainSign chose its sign, and so whether
// the mirrored design may be tried when the signed one fails.
type pidtuneSignRule int

const (
	// pidtuneSignFixed: the DC gain sign of a stable plant, or 1 for a
	// stable plant without one; never mirrored.
	pidtuneSignFixed pidtuneSignRule = iota
	// pidtuneSignForced: K·N(0)/a > 0, which a stable loop needs whenever
	// its characteristic polynomial's leading coefficient has the sign of
	// a; mirrored when that coefficient may flip.
	pidtuneSignForced
	// pidtuneSignGuessed: the high-frequency gain sign, or 1; no rule.
	pidtuneSignGuessed
)

// pidtuneGainSign returns the sign of every Pidtune gain: that of the DC
// gain of a stable plant, else of N(0)/a for the minimal G = N/D with D's
// leading coefficient a (N(1)/a in z) when it is forced, else of the
// high-frequency gain. The closed-loop polynomial P = D·Dc + N·Nc of a
// stable loop keeps the sign of its leading coefficient on [0, ∞) (z ≥ 1),
// where P(0) is D(0) + K·N(0) for P/PD and K·N(0) under integral action.
// When that leading coefficient has the sign of a, as for every loop whose
// L(∞) is 0, K·N(0)/a > 0 is forced if D(0)/a ≤ 0 (an integrator or an odd
// count of unstable real poles) or integral is set: the DC gain sign of G
// with integrators removed and unstable poles reflected. A stable plant
// with no DC sign and a pole at the integrator (pidtuneNearIntegrator) takes
// this rule too.
func pidtuneGainSign(plant *System, at func(float64) complex128, wc float64, integral bool) (float64, pidtuneSignRule) {
	if s := pidTuningDCSign(plant, at, wc); s != 0 {
		return s, pidtuneSignFixed
	}
	if stable, err := plant.IsStable(); err != nil || stable && !pidtuneNearIntegrator(plant, wc) {
		return 1, pidtuneSignFixed
	}
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
	scale := 0.0
	for i, c := range num {
		scale = math.Max(scale, math.Abs(c)*math.Pow(x, float64(len(num)-1-i)))
	}
	for i, c := range num {
		if math.Abs(c)*math.Pow(x, float64(len(num)-1-i)) > 1e3*0x1p-52*scale {
			return math.Copysign(1, c) * a, pidtuneSignGuessed
		}
	}
	return 1, pidtuneSignGuessed
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

// pidtuneGate checks a backed-off design over the whole back-off band with
// TunePID's sampling: |S| ≤ pidTuningMaxSensitivity, and the phase margin
// at every gain crossover, close pairs at a resonance or a near-miss of
// 0 dB included. The plant response and the discrete controller terms on
// both grids are computed once.
type pidtuneGate struct {
	p     pidTuningPlant
	sens  pidtuneGrid
	cross pidtuneGrid
	at    func(float64) complex128
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
		hi = math.Min(hi, math.Pi/plant.Dt)
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
	gate := &pidtuneGate{p: p, at: at}
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

func (g *pidtuneGate) check(pid *PID, pidType PidtuneType, wc, pm float64) error {
	c := PID2{Kp: pid.Kp, Ki: pid.Ki, Kd: pid.Kd, Tf: pid.Tf, Dt: pid.Dt, IFormula: pid.IFormula, DFormula: pid.DFormula}
	if ms := g.sensitivityOver(c, g.p.maxSensitivity); ms > g.p.maxSensitivity {
		return fmt.Errorf("%s design at wc=%g has sensitivity peak %.3g: %w", pidType, wc, ms, ErrPIDTuningTargetUnattainable)
	}
	_, margins := pidTuningCrossingsOn(g.p, c, g.cross.w, wc, g.crossExcess(c))
	for _, m := range margins {
		if !(m >= pm-pidtuneMarginTol) {
			return fmt.Errorf("%s design at wc=%g has phase margin %.3g° at a gain crossover: %w", pidType, wc, m, ErrPIDTuningTargetUnattainable)
		}
	}
	return nil
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
// of Feedback(model, C, −1) and the verdict is closedLoopPoles'. Anything else,
// errors included, goes to closedLoopPoles.
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
	e12, de, ded          *mat.Dense
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
		e12: scalar(), de: scalar(), ded: scalar(),
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
	l.de.Mul(cs.D, e21)
	l.ded.Mul(l.de, plant.D)
	l.ded.Scale(sign, l.ded)
	l.e12.Set(0, 0, 1)
	l.e12.Sub(l.e12, l.ded)

	b := l.buffers(n2)
	if n1 > 0 {
		l.b1e12.Mul(plant.B, l.e12)
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
	if dc, err := plant.DCGain(); err == nil && math.IsInf(dc.At(0, 0), 0) {
		return pidtuneIntegratingCrossover(plant)
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
	at, tol := pidtuneIntegratorCluster(plant, poles)
	if w := pidtuneSlowestRoot(plant, poles, at, tol); !math.IsInf(w, 1) {
		w *= 0.5
		if plant.Dt > 0 {
			w = math.Min(w, 0.1*math.Pi/plant.Dt)
		}
		return w, nil
	}

	return 0, fmt.Errorf("no gain crossover, bandwidth or nonzero pole to infer a crossover from; pass wc > 0: %w", ErrInvalidArgument)
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
// and a radius holding the largest m roots that lie within
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
	d := make([]float64, len(roots))
	for i, r := range roots {
		d[i] = cmplx.Abs(r - at)
	}
	slices.Sort(d)
	for m := len(d); m > 0; m-- {
		if tol := 100 * math.Pow(0x1p-52, 1/float64(m)) * scale; d[m-1] <= tol {
			return at, tol
		}
	}
	return at, 0
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
	pidf     pidtunePIDFMode
}

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
	return pidtuneGains(plant, nil, pidType, wc, pmDeg, sign, stableFilter, false, pidtunePIDFKeep)
}

// pidtuneGains is computePIDGains, lagFloor and pidf aside, evaluating the
// plant with at.
func pidtuneGains(plant *System, at *pidtunePlantPoint, pidType PidtuneType, wc, pmDeg, sign float64, stableFilter, lagFloor bool, pidf pidtunePIDFMode) (*PID, error) {
	pid, err := computePIDGainsPositive(plant, at, pidType, wc, pmDeg, sign, stableFilter, lagFloor, pidf)
	if err != nil {
		return nil, err
	}
	pid.Dt = plant.Dt
	pid.Kp *= sign
	pid.Ki *= sign
	pid.Kd *= sign
	return pid, nil
}

func computePIDGainsPositive(plant *System, at *pidtunePlantPoint, pidType PidtuneType, wc, pmDeg, sign float64, stableFilter, lagFloor bool, pidf pidtunePIDFMode) (*PID, error) {
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
	terms := pidtuneTerms{wc: wc, dt: plant.Dt, dFormula: dFormula, lagFloor: lagFloor, pidf: pidf}
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
