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
// 1−Dt/Tf would leave the unit circle (Tf ≤ Dt/2) uses backward Euler unless
// only the forward-Euler design gives a stable loop. An explicitly
// requested phase margin the family cannot reach at wc returns
// ErrPIDTuningTargetUnattainable instead of a controller with a different
// margin. Every gain takes one sign: that of the DC gain for a stable plant,
// and for an integrating or unstable plant the sign stabilization forces
// (see pidtuneGainSign), so Pidtune(−G) mirrors Pidtune(G); where the force
// lapses, as for an ideal derivative large enough to flip the closed-loop
// polynomial's leading coefficient, the other sign is tried when the forced
// one fails. A design whose closed loop is unstable by its poles (delay
// absorbed for discrete plants) or by a Nyquist test (continuous delays) is
// rejected; loops outside both tests are not screened. With wc = 0 a
// rejected design is retried at crossovers lowered up to three decades
// (not for an integrating or unstable plant with a continuous delay),
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
	run := &pidtuneRun{plant: plant, screen: screen, eval: eval, pidType: pidType, pm: pm, explicit: explicit, wc: wc}
	switch rule {
	case pidtuneSignGuessed:
		run.mirror = func(*PID) bool { return true }
	case pidtuneSignForced:
		run.mirror = pidtuneMirrorAdmissible(screen.model)
	}

	steps := 0
	// An unstable plant needs bandwidth that a continuous delay already
	// caps, and each back-off step would run a Nyquist test.
	if auto && (rule == pidtuneSignFixed || plant.Dt > 0 || !plant.HasDelay()) {
		steps = pidtuneBackoffSteps
	}
	var first error
	for k := 0; k <= steps; k++ {
		pid, err := run.design(wc*math.Pow(10, -float64(k)/pidtuneBackoffPerDecade), sign, k > 0)
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
// stable, keeps the phase margin at every gain crossover of the loop and
// keeps |S| ≤ pidTuningMaxSensitivity.
const (
	pidtuneBackoffPerDecade = 20
	pidtuneBackoffSteps     = 60
)

// pidtuneRun holds what Pidtune's designs at successive crossovers share.
type pidtuneRun struct {
	plant    *System
	screen   *pidStabilityScreen
	eval     *sisoEval
	pidType  PidtuneType
	pm, wc   float64
	explicit bool
	// mirror, when set, admits a design against the chosen gain sign to be
	// tried after the signed one fails.
	mirror func(*PID) bool
	gate   *pidtuneGate
}

// design designs pidType at wc for sign·G, then for −sign·G when r.mirror
// admits it, and screens each; gated designs must also pass the back-off
// gate. A discrete derivative moved to backward Euler for its filter pole
// falls back to the forward-Euler design when only that one passes.
func (r *pidtuneRun) design(wc, sign float64, gated bool) (*PID, error) {
	pid, err := r.signed(wc, sign, nil, gated)
	if r.mirror != nil && errors.Is(err, ErrPIDTuningTargetUnattainable) {
		if alt, altErr := r.signed(wc, -sign, r.mirror, gated); altErr == nil {
			return alt, nil
		}
	}
	return pid, err
}

func (r *pidtuneRun) signed(wc, sign float64, admissible func(*PID) bool, gated bool) (*PID, error) {
	pid, err := computePIDGains(r.plant, r.pidType, wc, r.pm, sign, true)
	if err != nil {
		return nil, err
	}
	err = r.check(pid, wc, sign, admissible, gated)
	if err == nil || pid.DFormula != BackwardEuler || r.plant.Dt == 0 || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		return pid, err
	}
	fe, feErr := computePIDGains(r.plant, r.pidType, wc, r.pm, sign, false)
	if feErr != nil || fe.DFormula != ForwardEuler || r.check(fe, wc, sign, admissible, gated) != nil {
		return nil, err
	}
	return fe, nil
}

// check screens pid; the gate, cheaper than a Nyquist test of a continuous
// delay, runs before it there and after the pole test otherwise.
func (r *pidtuneRun) check(pid *PID, wc, sign float64, admissible func(*PID) bool, gated bool) error {
	if admissible != nil && !admissible(pid) {
		return fmt.Errorf("%s design at wc=%g with gain sign %g cannot be stable: %w", r.pidType, wc, sign, ErrPIDTuningTargetUnattainable)
	}
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
	if err := checkPidtuneStability(r.screen, r.eval, pid, r.pidType, wc); err != nil {
		return err
	}
	if gated && !gateFirst {
		return r.gated(pid, wc)
	}
	return nil
}

func (r *pidtuneRun) gated(pid *PID, wc float64) error {
	if r.gate == nil {
		r.gate = newPidtuneGate(r.plant, r.screen.model, r.eval, r.wc)
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
	// a (see pidtuneMirrorAdmissible).
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
// with integrators removed and unstable poles reflected.
func pidtuneGainSign(plant *System, at func(float64) complex128, wc float64, integral bool) (float64, pidtuneSignRule) {
	if s := pidTuningDCSign(plant, at, wc); s != 0 {
		return s, pidtuneSignFixed
	}
	if stable, err := plant.IsStable(); err != nil || stable {
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

// pidtuneMirrorAdmissible returns whether a design against the forced sign
// may be stable on model (discrete delays absorbed; a continuous delay,
// positive on the real axis, is left out): only when the leading
// coefficient of its closed-loop polynomial D·Dc + N·Nc lacks the sign of
// D's, as when a relative-degree-one plant meets an ideal derivative with
// Kd·N(s)/D(s)·s → Kd·n/a < −1. Unknown polynomials admit every design.
func pidtuneMirrorAdmissible(model *System) func(*PID) bool {
	res, err := model.rationalTransferFunction(nil)
	if err != nil {
		return func(*PID) bool { return true }
	}
	num, den := Poly(res.TF.Num[0][0]), Poly(res.TF.Den[0])
	peak := 0.0
	for _, c := range num {
		peak = math.Max(peak, math.Abs(c))
	}
	for len(num) > 0 && math.Abs(num[0]) <= 1e3*0x1p-52*peak {
		num = num[1:]
	}
	return func(pid *PID) bool {
		if len(den) == 0 || den[0] == 0 || len(num) == 0 {
			return true
		}
		numC, denC := pidtuneControllerPoly(pid, model.Dt)
		char := den.Mul(denC).Add(num.Mul(numC))
		scale := 0.0
		for _, c := range den.Mul(denC) {
			scale = math.Max(scale, math.Abs(c))
		}
		for _, c := range num.Mul(numC) {
			scale = math.Max(scale, math.Abs(c))
		}
		return !finitePID(char[0]) || char[0]*math.Copysign(1, den[0]) <= 1e3*0x1p-52*scale
	}
}

// pidtuneControllerPoly returns the numerator and denominator of pid in s,
// or in z when dt > 0.
func pidtuneControllerPoly(pid *PID, dt float64) (num, den Poly) {
	var x, dd, ki Poly
	if dt == 0 {
		x, dd, ki = Poly{1, 0}, Poly{pid.Tf, 1}, Poly{pid.Ki}
	} else {
		w := pidFormulaWeight(pid.DFormula)
		x = Poly{1, -1}
		dd = Poly{pid.Tf + w*dt, dt - pid.Tf - w*dt}
		wi := pidFormulaWeight(pid.IFormula)
		ki = Poly{pid.Ki * dt * wi, pid.Ki * dt * (1 - wi)}
	}
	for len(dd) > 1 && dd[0] == 0 {
		dd = dd[1:]
	}
	switch {
	case pid.Ki != 0 && pid.Kd != 0:
		return Poly{pid.Kp}.Mul(x).Mul(dd).Add(ki.Mul(dd)).Add(Poly{pid.Kd}.Mul(x).Mul(x)), x.Mul(dd)
	case pid.Ki != 0:
		return Poly{pid.Kp}.Mul(x).Add(ki), x
	case pid.Kd != 0:
		return Poly{pid.Kp}.Mul(dd).Add(Poly{pid.Kd}.Mul(x)), dd
	}
	return Poly{pid.Kp}, Poly{1}
}

// pidtuneGate checks a backed-off design over the whole back-off band with
// TunePID's sampling: |S| ≤ pidTuningMaxSensitivity, and the phase margin
// at every gain crossover, close pairs at a resonance or a near-miss of
// 0 dB included. The plant response on both grids is computed once.
type pidtuneGate struct {
	p     pidTuningPlant
	omega []float64
	sw    []float64
	sg    []complex128
	at    func(float64) complex128
}

const pidtuneGatePerDecade = 100

func newPidtuneGate(plant, model *System, eval *sisoEval, wc float64) *pidtuneGate {
	lo, hi := wc*1e-6, wc*1e3
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
	gate := &pidtuneGate{p: p, omega: pidTuningCrossGrid(p, omega), at: at}
	for w := lo; ; {
		gate.sw = append(gate.sw, w)
		gate.sg = append(gate.sg, at(w))
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
	_, margins := pidTuningCrossings(g.p, c, g.omega, wc)
	for _, m := range margins {
		if !(m >= pm-pidtuneMarginTol) {
			return fmt.Errorf("%s design at wc=%g has phase margin %.3g° at a gain crossover: %w", pidType, wc, m, ErrPIDTuningTargetUnattainable)
		}
	}
	return nil
}

// sensitivityOver returns a sampled |S| above bound, or the largest sampled
// |S| when none is; as pidTuningSensitivityPeak, grid maxima above half the
// bound are refined by golden section. Unevaluable samples are skipped.
func (g *pidtuneGate) sensitivityOver(c PID2, bound float64) float64 {
	s := func(w float64, gw complex128) float64 { return 1 / cmplx.Abs(1+gw*pidTuningFeedback(c, w)) }
	refine := func(w float64) float64 { return s(w, g.at(w)) }
	peak := 0.0
	var v0, v1 float64
	for i, w := range g.sw {
		v := s(w, g.sg[i])
		if math.IsNaN(v) {
			v0, v1 = 0, 0
			continue
		}
		if i >= 2 && v1 >= v0 && v1 >= v && v1 > bound/2 {
			_, r := pidTuningGoldenMax(refine, g.sw[i-2], w)
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
// With stableFilter, a discrete forward-Euler derivative whose filter pole
// 1−Dt/Tf leaves the unit circle is redesigned with backward Euler.
func computePIDGains(plant *System, pidType PidtuneType, wc, pmDeg, sign float64, stableFilter bool) (*PID, error) {
	pid, err := computePIDGainsPositive(plant, pidType, wc, pmDeg, sign, stableFilter)
	if err != nil {
		return nil, err
	}
	pid.Dt = plant.Dt
	pid.Kp *= sign
	pid.Ki *= sign
	pid.Kd *= sign
	return pid, nil
}

func computePIDGainsPositive(plant *System, pidType PidtuneType, wc, pmDeg, sign float64, stableFilter bool) (*PID, error) {
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
	if err == nil && stableFilter && plant.Dt > 0 && pid.DFormula == ForwardEuler && pid.Kd != 0 && pid.Tf <= plant.Dt/2 {
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
