package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/big"
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
	// searches; the floor bands at one crossover share one, as do all
	// backed-off crossovers, and all raised ones. A search that verifies its design under the
	// refined crossing check and searches again spends one budget. Evidence
	// counts the whole call.
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
	// try, if set, evaluates at without recording a failure.
	try       func(float64) (complex128, bool)
	stable    func(*PID2) (bool, error)
	stability PIDStabilityEvidence
	warnings  []string
	// gainSign is the sign every nonzero gain must take: that of the DC gain
	// of a stable plant, or 0 when unknown.
	gainSign float64
	// residueSign, when gainSign is 0, is the sign every nonzero gain must
	// take on an integrating plant (pidTuningIntegratorSign), or 0. The
	// phase heuristics keyed on gainSign assume sign·G starts near 0°,
	// which −90° per integrator breaks, so it is kept apart.
	residueSign float64
	// dcGain, if nonzero, is the DC gain of a plant assumed stable whose
	// gain signs are otherwise free, as sampled data's under MATLAB's
	// default of no unstable poles; highGain is its response at high.
	// pidTuningDCAdmissible rejects candidates with a real unstable
	// closed-loop root they imply, pidTuningFRDNyquist those the sampled
	// Nyquist count finds or cannot certify stable.
	dcGain   float64
	highGain complex128
	// nyquistCount marks the strict count on sampled data under that
	// default (pidTuningFRDStrictPlant), where a candidate the count
	// cannot certify is rejected. When dcGain is 0, lowAnchor marks a low
	// end G ≈ K·(jω)^lowOrder that anchors the count instead
	// (pidTuningFRDLowOrder).
	nyquistCount bool
	lowAnchor    bool
	lowOrder     int
	// lowAnchored marks a dcGain sampled at ω = 0 rather than extrapolated
	// from the band; topOrder, when topAnchored, is the order n of a high
	// end G ≈ K·(jω)^n that continuous data establish
	// (pidTuningFRDTopOrder). The strict count certifies only loops whose
	// ends both anchor so.
	lowAnchored bool
	topOrder    int
	topAnchored bool
	// resonances holds, by the index k of its lower sample, the fitted
	// interpolants of each step samples[k] → samples[k+1] of the strict
	// plant across which a resonance the data do not resolve may loop
	// (pidTuningFRDResonances).
	resonances [][]pidTuningFRDMode
	// markov, if nonzero, is lim sG(s) that continuous sampled data show at
	// their top (pidTuningFRDMarkov), and markovSpread its change over the
	// top half decade.
	markov, markovSpread float64
	// maxSensitivity bounds the sampled peak of |1/(1+L)| over the band, or
	// 0 for none; tau is the loop delay in seconds that sets its grid step.
	maxSensitivity, tau float64
	// modes are the plant's upper-half-plane poles and zeros in s, which
	// refine that grid and the crossover grid near lightly damped
	// resonances.
	modes []complex128
	// samples are the measured frequencies of sampled data, whose
	// interpolant can cross 0 dB between the crossover grid's points.
	samples []float64
	// crossFloor, when hasCrossFloor, is the margin floor at the other gain
	// crossovers once the wc target is raised above the requested margin.
	crossFloor    float64
	hasCrossFloor bool
	// crossLow and crossHigh widen the band in which the floor finds the
	// other crossovers to the plant's dynamics, since a floor design's
	// high gain can cross 0 dB near a mode outside the band. The |S| scan
	// also reaches crossHigh, the reported band's top.
	crossLow, crossHigh float64
	// zeroScan checks the returned design's crossings down to its
	// controller zeros (pidTuningZeroGrid) and fails with pidZeroRejected
	// when one is below the floor; zeroSearch checks every candidate so.
	// Sampled data, without a model below its band, leaves both off.
	zeroScan, zeroSearch bool
	// zeroLow, if positive, is a tenth of the plant's slowest pole or zero,
	// down to which the zero grid also reaches: a slow plant zero or
	// resonance can lift |L| back through 1 below the band.
	zeroLow float64
	// crossScan selects the margin check of a search outside the floor.
	crossScan pidCrossScan
	// floorRungs, if positive, caps the floor bands tried; minTfSeed seeds
	// a filtered family at its least admissible Tf when no other seed is
	// feasible. A backed-off crossover's probes set both; TunePID retries
	// a failed filtered search with minTfSeed.
	floorRungs int
	minTfSeed  bool
	// returnSensitivity, if set, returns |S(∞)| of a candidate, +Inf when
	// the screen rejects it; above maxReturnSensitivity it counts as |S|
	// excess, so the search may cross it as it crosses the sampled bound.
	returnSensitivity    func(*PID2) float64
	maxReturnSensitivity float64
	// loopTail, if set, bounds |L(jω')| over ω' ≥ ω for a candidate, so the
	// |S| scan, given maxSensitivity, runs past high until |S| <=
	// maxSensitivity is certain, and the zero grid past its top until no
	// 0 dB crossing remains (pidTuningZeroGridTop).
	loopTail func(c *PID2, w float64) float64
	// dcEdge and nyquistEdge, if finite and nonzero, are G at ω = 0 and,
	// discrete, at z = −1, the ends the |S| scan's band stops short of: a
	// design whose closed-loop root sits near s = 0 or z = −1 hides its
	// |S| peak there.
	dcEdge, nyquistEdge complex128
	// sensLow and sensHigh, if set, widen the |S| scan below low and above
	// its top, a peak there being away from both band and edges; samples
	// there come from try, cached in sensCache, and count as 0 where it
	// fails.
	sensLow, sensHigh float64
	sensCache         map[float64]complex128
}

type pidCrossScan int

const (
	// pidCrossAuto searches with the coarse scan, then verifies the design
	// with the refined one and searches again under it if that fails.
	pidCrossAuto pidCrossScan = iota
	pidCrossCoarse
	pidCrossRefined
	// pidCrossGrid holds the grid's crossings to the floor in the search
	// and reports, without rejecting, every sign change of |L| − 1 on the
	// refined grid, which holds sampled data's samples. Free one-gain
	// designs on sampled data use it: without a stability certificate a
	// crossing the samples add can be neither counted the far way round
	// from −1 nor rejected without losing robust designs, so its wrapped
	// margin lowers the achieved one.
	pidCrossGrid
)

// pidCoarseRejected reports a coarse-scan design that the refined crossing
// check rejects.
type pidCoarseRejected struct{ evaluations int }

func (*pidCoarseRejected) Error() string { return "coarse design fails refined crossing check" }

// pidZeroRejected reports a design that fails the crossing check down to
// its controller zeros.
type pidZeroRejected struct{ evaluations int }

func (*pidZeroRejected) Error() string { return "design fails crossing check at its controller zeros" }

// pidNonfiniteResponse reports a plant sample in the scan band that is not
// finite, as for a band far under integrators' rounded poles.
type pidNonfiniteResponse struct{ w float64 }

func (e *pidNonfiniteResponse) Error() string {
	return fmt.Sprintf("nonfinite plant response at %g: %v", e.w, ErrInvalidArgument)
}

func (*pidNonfiniteResponse) Unwrap() error { return ErrInvalidArgument }

// pidTuningMaxSensitivity bounds |S| for a delay-capped or moved default
// crossover, so the design keeps at least 2.5 dB of gain margin and 14° of
// phase margin at every frequency. A margin met only at wc can leave
// |L| ≈ 1 where the delay phase passes −180° or the lead fades.
const pidTuningMaxSensitivity = 4

// pidSBound selects how TunePID's search bounds |S| by
// pidTuningMaxSensitivity.
type pidSBound int

const (
	pidSFree pidSBound = iota
	// pidSSearch rejects every candidate over the bound.
	pidSSearch
	// pidSCheck searches unbounded and, when the design exceeds the bound,
	// searches again as pidSSearch: the unbounded search is cheaper and
	// keeps every design within the bound.
	pidSCheck
)

// pidTuningDCSign returns the sign of the DC gain of a stable plant, or 0
// when the plant is not known to be stable, Pidtune recognises an
// integrator, or its DC gain is zero relative to the peak of |G| sampled
// over six decades around wc. Pidtune recognises an integrator itself, by
// integratingDCGain, whose transfer limit takes a DC gain at coefficient
// rounding for infinite: DCGain, like MATLAB dcgain, is finite for an
// integrator rounded inside the unit circle. A pole radius alone cannot
// tell: rounded integrators lie up to 3e-4 from z = 1, beside genuine slow
// poles. A frequency response at DC is no substitute either: next to such
// an integrator it is finite, and the samples below wc then fail. A sample
// try cannot resolve is skipped, so it never fails the design.
func pidTuningDCSign(plant *System, try func(float64) (complex128, bool), wc float64) float64 {
	if stable, err := plant.IsStable(); err != nil || !stable {
		return 0
	}
	g, err := plant.integratingDCGain()
	if err != nil {
		return 0
	}
	dc := g.At(0, 0)
	if !finitePID(dc) {
		return 0
	}
	peak := math.Abs(dc)
	for k := -3; k <= 3; k++ {
		w := wc * math.Pow(10, float64(k))
		if plant.Dt > 0 && w >= math.Pi/plant.Dt {
			break
		}
		if h, ok := try(w); ok && finitePID(cmplx.Abs(h)) {
			peak = math.Max(peak, cmplx.Abs(h))
		}
	}
	if math.Abs(dc) <= 1e3*0x1p-52*peak {
		return 0
	}
	return math.Copysign(1, dc)
}

// pidTuningIntegratorSign returns, for a plant whose DC gain
// integratingDCGain finds infinite and whose poles off the integrator
// (pidtuneIntegratorTol) are all stable, the sign of that gain: the sign of
// the integrator's residue, which MATLAB pidtune gives every gain. Other
// plants return 0: on 1/(s(s−1)) the residue is −1 but every stabilising PD
// has positive gains.
func pidTuningIntegratorSign(plant *System, wc float64) float64 {
	g, err := plant.integratingDCGain()
	if err != nil || !math.IsInf(g.At(0, 0), 0) {
		return 0
	}
	model, err := finiteDimensionalModel(plant, "TunePID")
	if err != nil {
		return 0
	}
	poles, err := model.Poles()
	if err != nil {
		return 0
	}
	at, tol := pidtuneIntegratorTol(plant, poles, wc)
	for _, p := range poles {
		if cmplx.Abs(p-at) <= tol {
			continue
		}
		if plant.Dt > 0 && !(cmplx.Abs(p) < 1) || plant.Dt == 0 && !(real(p) < 0) {
			return 0
		}
	}
	return math.Copysign(1, g.At(0, 0))
}

// pidStabilityScreen certifies closed-loop stability of PID candidates on a
// SISO plant: by closed-loop poles when the model is delay-free (discrete
// delays absorbed), by a Nyquist test for supported continuous delays.
// maxTail bounds limsup |L| of a delayed ideal-derivative loop: 1 accepts
// every stable loop, pidTuningMaxTail adds TunePID's robustness margin.
type pidStabilityScreen struct {
	plant, model *System
	eval         *sisoEval
	hiddenStable func() (bool, error)
	maxTail      float64
	// rational is model, or for a discrete plant with only external
	// delays the undelayed plant, whose loop is model's with the delay
	// z^-lag factored out: the shift chain AbsorbDelay adds defeats
	// TransferFunction and the exact test's order cap.
	rational *System
	lag      int
	// transferRes caches rational's transfer function for ideal-derivative
	// screens; transferErr its failure.
	transferRes *TransferFuncResult
	transferErr error
}

func (s *pidStabilityScreen) transfer() (*TransferFuncResult, error) {
	if s.transferRes == nil && s.transferErr == nil {
		s.transferRes, s.transferErr = s.rational.rationalTransferFunction(nil)
	}
	return s.transferRes, s.transferErr
}

// pidTuningExplicitPlant returns plant with E folded in when E is the
// identity or well conditioned (normExplicitForm), else plant: the search,
// screens and tails read A, B, C as an explicit realization.
func pidTuningExplicitPlant(plant *System) *System {
	if plant.E == nil {
		return plant
	}
	if !plant.IsDescriptor() {
		cp := plant.Copy()
		cp.E = nil
		return cp
	}
	if ex, err := normExplicitForm(plant, "TunePID"); err == nil {
		return ex
	}
	return plant
}

func newPIDStabilityScreen(plant *System, eval *sisoEval, maxTail float64) (*pidStabilityScreen, error) {
	model, rational, lag := plant, plant, 0
	if plant.Dt > 0 && plant.HasDelay() {
		var err error
		if model, err = plant.AbsorbDelay(AbsorbAll); err != nil {
			return nil, err
		}
		rational = model
		if !plant.HasInternalDelay() {
			rational = plant.Copy()
			rational.Delay, rational.InputDelay, rational.OutputDelay = nil, nil, nil
			n, _, _ := model.Dims()
			n0, _, _ := plant.Dims()
			lag = n - n0
		}
	}
	return &pidStabilityScreen{
		plant: plant, model: model, rational: rational, lag: lag, eval: eval, maxTail: maxTail,
		hiddenStable: sync.OnceValues(func() (bool, error) { return pidTuningHiddenModesStable(model) }),
	}, nil
}

// at returns the stability test for family tuned at wc and its evidence
// kind, or a nil test with PIDStabilitySampledOnly when the loop is outside
// the scope of both certificates.
func (s *pidStabilityScreen) at(wc float64, family PidtuneType) (func(*PID2) (bool, error), PIDStabilityEvidence, error) {
	if !s.model.HasDelay() {
		return s.closedLoopPoles, PIDStabilityClosedLoopPoles, nil
	}
	stable, ok, err := pidTuningDelayStability(s.plant, s.eval, wc, pidTuningIdealDerivative(family), s.maxTail)
	if err != nil || !ok {
		return nil, PIDStabilitySampledOnly, err
	}
	return stable, PIDStabilityNyquist, nil
}

// returnSensitivity returns |S(∞)| = 1/(1+L(∞)) of a continuous delay-free
// loop with an ideal derivative, +Inf when 1+L(∞) is below
// pidTuningIdealReturnTol, and 0 for other loops; see pidTuningIdealReturn.
func (s *pidStabilityScreen) returnSensitivity(c *PID2) float64 {
	if c.Kd == 0 || c.Tf != 0 || c.Dt != 0 || s.model.Dt != 0 || s.model.HasDelay() {
		return 0
	}
	r := pidTuningIdealReturn(s.model, c.Kd)
	if !(r >= pidTuningIdealReturnTol) {
		return math.Inf(1)
	}
	return 1 / r
}

func (s *pidStabilityScreen) closedLoopPoles(c *PID2) (bool, error) {
	controller := &PID{Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf, Dt: c.Dt, IFormula: c.IFormula, DFormula: c.DFormula}
	cs, err := controller.explicitSystem()
	if err != nil {
		if c.Kd != 0 && c.Tf == 0 {
			if math.IsInf(s.returnSensitivity(c), 1) {
				return false, nil
			}
			if ok, err := s.hiddenStable(); !ok || err != nil {
				return false, err
			}
			return pidTuningIdealStable(s.rational, s.transfer, c, s.lag)
		}
		return false, err
	}
	loop, err := Feedback(s.model, cs, -1)
	if errors.Is(err, ErrAlgebraicLoop) {
		return false, nil
	}
	if err != nil {
		return false, err
	}
	poles, err := loop.Poles()
	if err != nil {
		return false, err
	}
	return pidTuningPolesStable(poles, loop.A, s.model, cs), nil
}

const (
	// pidTuningExactBand is how far outside the stability boundary a float
	// pole may lie and still be retested exactly.
	pidTuningExactBand = 1e-3
	// pidTuningExactOrder caps the closed-loop order tested exactly.
	pidTuningExactOrder = 16
	// pidTuningExactLagOrder caps plant order plus absorbed delay samples
	// of an ideal-derivative loop given the Schur–Cohn recursion when the
	// float verdict on its exact characteristic is in doubt: the delay's
	// states need no characteristic-polynomial expansion, so that
	// characteristic is built at any lag.
	pidTuningExactLagOrder = 64
)

// pidTuningPolesStable decides stability of the negative-feedback loop of
// delay-free SISO standard models plant and c from the float poles of its
// state matrix a, as IsStable, except that the exact test decides when the
// float verdict is in doubt: every pole the float test rejects lies within
// pidTuningExactBand outside the boundary (with |p| <= 1 in continuous
// time) or within ten error scales of it, or a pole it accepts has a
// stability margin below its error estimate (pidTuningPoleUncertain). A
// cluster of poles near the boundary, as a slow plant pole and a tiny-gain
// integrator give near z = 1, carries eigenvalue errors far above the
// margin the gains leave; so does a slow pole beside a fast one of a
// high-gain loop, whose errors scale with ‖A‖. The loop's A is
// then built from the realizations in exact integer arithmetic, and its
// characteristic polynomial must have every root at |z| < 1−1e-10, or
// Re s < −1e-10, as poleStabilityTolerance asks of a pole with |p| <= 1.
// Loops outside the exact test's scope keep the float verdict.
func pidTuningPolesStable(poles []complex128, a *mat.Dense, plant, c *System) bool {
	if len(poles) == 0 {
		return true
	}
	stable, decided := pidTuningFloatVerdict(poles, plant.Dt == 0, mat.Norm(a, math.Inf(1)))
	if decided {
		return stable
	}
	exact, ok := pidTuningExactStable(plant, c)
	if !ok {
		return stable
	}
	return exact
}

// pidTuningFloatVerdict is the float half of pidTuningPolesStable for
// poles of a matrix of norm anorm: decided is false when the exact test
// must settle the verdict.
func pidTuningFloatVerdict(poles []complex128, continuous bool, anorm float64) (stable, decided bool) {
	stable, doubt := true, false
	eta := pidTuningPoleErrorScale * float64(len(poles)) * 0x1p-52 * anorm
	for i, p := range poles {
		if poleInsideStabilityBoundary(p, continuous, poleStabilityTolerance(p)) {
			margin := 1 - cmplx.Abs(p)
			if continuous {
				margin = -real(p)
			}
			doubt = doubt || pidTuningPoleUncertain(poles, i, margin/pidTuningPoleMarginSafety, eta)
			continue
		}
		stable = false
		excess := cmplx.Abs(p) - 1 + poleStabilityTolerance(p)
		if continuous {
			excess = real(p) + poleStabilityTolerance(p)
		}
		if excess > pidTuningPoleMarginSafety*eta && (continuous && cmplx.Abs(p) > 1 || excess > pidTuningExactBand) {
			return false, true
		}
	}
	return stable, stable && !doubt
}

const (
	// pidTuningPoleErrorScale and pidTuningPoleMarginSafety pad the
	// first-order eigenvalue error model of pidTuningPoleUncertain against
	// non-normal realizations.
	pidTuningPoleErrorScale   = 1e3
	pidTuningPoleMarginSafety = 10
)

// pidTuningPoleUncertain reports whether float eigenvalue errors of size
// eta, a backward error of the state matrix, may move poles[i] by e or more.
// With δj = |poles[i]−poles[j]|, a perturbation eta of the characteristic
// polynomial moves the root by about the e solving e·∏ max(e, δj) = eta:
// eta/∏δj for an isolated root, eta^(1/k) inside a k-fold cluster. The left
// side increases with e, so the root may move by e when it is below eta.
// A matrix perturbation eta also moves a simple eigenvalue by about eta
// itself, however far the others lie, so e below eta is uncertain too.
func pidTuningPoleUncertain(poles []complex128, i int, e, eta float64) bool {
	if !(e > 0) {
		return true
	}
	f := e
	for j, q := range poles {
		if j != i {
			f *= max(e, cmplx.Abs(poles[i]-q))
		}
	}
	return e < eta || f < eta
}

// pidTuningExactStable is pidTuningPolesStable's exact test; ok is false
// when the loop is outside its scope.
func pidTuningExactStable(plant, c *System) (stable, ok bool) {
	if plant.IsDescriptor() || c.IsDescriptor() || plant.HasDelay() || c.HasDelay() || c.Dt != plant.Dt {
		return false, false
	}
	n1, m1, p1 := plant.Dims()
	n2, m2, p2 := c.Dims()
	if m1 != 1 || p1 != 1 || m2 != 1 || p2 != 1 || n1+n2 == 0 || n1+n2 > pidTuningExactOrder {
		return false, false
	}
	for _, a := range []*mat.Dense{plant.A, plant.B, plant.C, plant.D, c.A, c.B, c.C, c.D} {
		if a != nil && requireFiniteDense("", "", a) != nil {
			return false, false
		}
	}
	m, ln, ld := pidTuningExactClosedLoop(plant, c, n1, n2)
	if m == nil {
		return false, false
	}
	return pidTuningExactRootsStable(intCharPoly(m), ln, ld, plant.Dt == 0), true
}

// pidTuningExactRootsStable reports whether every root of p·ln/ld, p
// leading coefficient first and scaled in place, has Re s < −1e-10 or
// |z| < 1−1e-10.
func pidTuningExactRootsStable(p []*big.Int, ln, ld *big.Int, continuous bool) bool {
	const tol = 1e-10
	if continuous {
		t := new(big.Rat).SetFloat64(tol)
		intScaleRoots(p, big.NewInt(1), new(big.Int).Mul(ld, t.Denom()))
		return intHurwitz(intShift(p, new(big.Int).Neg(new(big.Int).Mul(ln, t.Num()))))
	}
	rho := new(big.Rat).SetFloat64(1 - tol)
	intScaleRoots(p, new(big.Int).Mul(ln, rho.Num()), new(big.Int).Mul(ld, rho.Denom()))
	return intSchur(p)
}

// intScaleRoots scales p, leading coefficient first, in place so that its
// roots are those of p times den/num.
func intScaleRoots(p []*big.Int, num, den *big.Int) {
	scale := big.NewInt(1)
	for _, v := range slices.Backward(p) {
		v.Mul(v, scale)
		scale.Mul(scale, num)
	}
	scale.SetInt64(1)
	for _, v := range p {
		v.Mul(v, scale)
		scale.Mul(scale, den)
	}
}

// dyadic is the exact value m·2^e.
type dyadic struct {
	m *big.Int
	e int
}

func newDyadic(x float64) dyadic {
	f, e := math.Frexp(x)
	return dyadic{big.NewInt(int64(math.Ldexp(f, 53))), e - 53}
}

func (d dyadic) mul(o dyadic) dyadic { return dyadic{new(big.Int).Mul(d.m, o.m), d.e + o.e} }

// sub returns d − o.
func (d dyadic) add(o dyadic) dyadic { return d.sub(dyadic{new(big.Int).Neg(o.m), o.e}) }

func (d dyadic) sub(o dyadic) dyadic {
	if d.e <= o.e {
		t := new(big.Int).Lsh(o.m, uint(o.e-d.e))
		return dyadic{t.Sub(d.m, t), d.e}
	}
	t := new(big.Int).Lsh(d.m, uint(d.e-o.e))
	return dyadic{t.Sub(t, o.m), o.e}
}

// pidTuningExactClosedLoop returns an integer matrix m and positive integers
// ln, ld such that the A of Feedback(plant, c, −1) is m·ld/ln exactly, or a
// nil m for an algebraic loop. With δ = 1+D1D2, δ·A is
// [δA1−D2B1C1, −B1C2; B2C1, δA2−D1B2C2], whose entries are dyadic.
func pidTuningExactClosedLoop(plant, c *System, n1, n2 int) (m [][]*big.Int, ln, ld *big.Int) {
	at := func(a *mat.Dense, i, j int) dyadic { return newDyadic(a.At(i, j)) }
	d1, d2 := at(plant.D, 0, 0), at(c.D, 0, 0)
	zero := newDyadic(0)
	delta := newDyadic(1).sub(zero.sub(d1.mul(d2)))
	if delta.m.Sign() == 0 {
		return nil, nil, nil
	}
	n := n1 + n2
	ent := make([][]dyadic, n)
	for i := range ent {
		ent[i] = make([]dyadic, n)
	}
	for i := range n1 {
		b1 := at(plant.B, i, 0)
		for j := range n1 {
			ent[i][j] = delta.mul(at(plant.A, i, j)).sub(d2.mul(b1).mul(at(plant.C, 0, j)))
		}
		for j := range n2 {
			ent[i][n1+j] = zero.sub(b1.mul(at(c.C, 0, j)))
		}
	}
	for i := range n2 {
		b2 := at(c.B, i, 0)
		for j := range n1 {
			ent[n1+i][j] = b2.mul(at(plant.C, 0, j))
		}
		for j := range n2 {
			ent[n1+i][n1+j] = delta.mul(at(c.A, i, j)).sub(d1.mul(b2).mul(at(c.C, 0, j)))
		}
	}
	emin := delta.e
	for _, row := range ent {
		for _, v := range row {
			emin = min(emin, v.e)
		}
	}
	neg := delta.m.Sign() < 0
	m = make([][]*big.Int, n)
	for i, row := range ent {
		m[i] = make([]*big.Int, n)
		for j, v := range row {
			x := new(big.Int).Lsh(v.m, uint(v.e-emin))
			if neg {
				x.Neg(x)
			}
			m[i][j] = x
		}
	}
	ln, ld = new(big.Int).Abs(delta.m), big.NewInt(1)
	if s := delta.e - emin; s >= 0 {
		ln.Lsh(ln, uint(s))
	} else {
		ld.Lsh(ld, uint(-s))
	}
	return m, ln, ld
}

// intCharPoly returns det(λI−m), leading coefficient first, by Berkowitz's
// division-free recursion: the polynomial of the leading (r+1)×(r+1) block
// is the lower-triangular Toeplitz matrix with first column
// (1, −m_rr, −R·C, −R·M·C, …, −R·M^(r−1)·C) times that of the leading r×r
// block M, with R and C the rest of row and column r.
func intCharPoly(m [][]*big.Int) []*big.Int {
	n := len(m)
	c := []*big.Int{big.NewInt(1), new(big.Int).Neg(m[0][0])}
	t := new(big.Int)
	for r := 1; r < n; r++ {
		col := make([]*big.Int, r+2)
		col[0], col[1] = big.NewInt(1), new(big.Int).Neg(m[r][r])
		v, w := make([]*big.Int, r), make([]*big.Int, r)
		for i := range v {
			v[i], w[i] = new(big.Int).Set(m[i][r]), new(big.Int)
		}
		for k := range r {
			s := new(big.Int)
			for j := range r {
				s.Add(s, t.Mul(m[r][j], v[j]))
			}
			col[k+2] = s.Neg(s)
			if k == r-1 {
				break
			}
			for i := range r {
				w[i].SetInt64(0)
				for j := range r {
					w[i].Add(w[i], t.Mul(m[i][j], v[j]))
				}
			}
			v, w = w, v
		}
		next := make([]*big.Int, r+2)
		for i := range next {
			s := new(big.Int)
			for j := max(0, i-r-1); j <= min(i, r); j++ {
				s.Add(s, t.Mul(col[i-j], c[j]))
			}
			next[i] = s
		}
		c = next
	}
	return c
}

// intShift returns the coefficients of p(x+h), leading first.
func intShift(p []*big.Int, h *big.Int) []*big.Int {
	q := make([]*big.Int, len(p))
	for i, v := range p {
		q[i] = new(big.Int).Set(v)
	}
	t := new(big.Int)
	for i := 1; i < len(q); i++ {
		for j := 1; j <= len(q)-i; j++ {
			q[j].Add(q[j], t.Mul(q[j-1], h))
		}
	}
	return q
}

// intContent divides p by the gcd of its coefficients.
func intContent(p []*big.Int) {
	g := new(big.Int)
	for _, v := range p {
		g.GCD(nil, nil, g, new(big.Int).Abs(v))
	}
	if g.Sign() == 0 || g.IsInt64() && g.Int64() == 1 {
		return
	}
	for _, v := range p {
		v.Quo(v, g)
	}
}

// intHurwitz reports whether every root of p, leading coefficient first,
// has negative real part, by the fraction-free Routh array: strict
// stability needs every first-column entry nonzero and of one sign, and
// rows scaled by positive factors keep their signs.
func intHurwitz(p []*big.Int) bool {
	if len(p) == 0 || p[0].Sign() == 0 {
		return false
	}
	neg := p[0].Sign() < 0
	row := func(off int) []*big.Int {
		var out []*big.Int
		for i := off; i < len(p); i += 2 {
			v := new(big.Int).Set(p[i])
			if neg {
				v.Neg(v)
			}
			out = append(out, v)
		}
		return out
	}
	prev, cur := row(0), row(1)
	t := new(big.Int)
	for range len(p) - 1 {
		if len(cur) == 0 || cur[0].Sign() <= 0 {
			return false
		}
		next := make([]*big.Int, len(prev)-1)
		for j := range next {
			v := new(big.Int).Mul(cur[0], prev[j+1])
			if j+1 < len(cur) {
				v.Sub(v, t.Mul(prev[0], cur[j+1]))
			}
			next[j] = v
		}
		intContent(next)
		prev, cur = cur, next
	}
	return true
}

// intSchur reports whether every root of p, leading coefficient first,
// lies strictly inside the unit circle, by the fraction-free Schur–Cohn
// recursion.
func intSchur(p []*big.Int) bool {
	c := p
	t, u := new(big.Int), new(big.Int)
	for n := len(c) - 1; n > 0; n-- {
		if t.Abs(c[n]).Cmp(u.Abs(c[0])) >= 0 {
			return false
		}
		q := make([]*big.Int, n)
		for i := range q {
			q[i] = new(big.Int).Mul(c[0], c[i])
			q[i].Sub(q[i], t.Mul(c[n], c[n-i]))
		}
		intContent(q)
		c = q
	}
	return true
}

// wantSign returns the sign every nonzero gain must take, or 0 for any.
func (p pidTuningPlant) wantSign() float64 {
	if p.gainSign != 0 {
		return p.gainSign
	}
	return p.residueSign
}

// pidTuningDCAdmissible reports whether c leaves the characteristic
// polynomial of the loop with a stable plant of DC gain dc the same sign at
// s = 0 (z = 1) as at +∞, where it has the signs of 1 + L(0⁺) and of
// 1 + L(∞). A sign change puts a real closed-loop root in the right half
// plane (outside the unit circle). 1 + L(0⁺) has the sign of Ki·dc with
// integral action, else of 1 + Kp·dc. 1 + L(∞) is read off the top sample
// wHi of response gHi: positive where the loop has rolled off, |L| < 1;
// else, continuous only, that of 1 + Re L where L lies within 30° of the
// real axis, as an ideal derivative on a relative-degree-one plant leaves
// it. There |1 + L| < 0.1 rejects c: its sign is unresolved and |S| > 10.
// The discrete top sample is z = −1, not +∞. Otherwise 1 + L(∞) is
// unknown, which accepts c, as does dc = 0.
func pidTuningDCAdmissible(c PID2, dc, wHi float64, gHi complex128) bool {
	if dc == 0 {
		return true
	}
	l := gHi * pidTuningFeedback(c, wHi)
	inf := 1.
	if !(cmplx.Abs(l) < 1) {
		if c.Dt > 0 || !(math.Abs(imag(l)) <= .5*cmplx.Abs(l)) {
			return true
		}
		if cmplx.Abs(1+l) < .1 {
			return false
		}
		inf = 1 + real(l)
	}
	zero := 1 + c.Kp*dc
	if c.Ki != 0 {
		zero = c.Ki * dc
	}
	return zero*inf > 0
}

// pidTuningKeepsIntegral reports whether c keeps the integral action its
// family has. A search step can land Ki on exactly 0, a P/PD loop that
// bypasses the integral-only screens, such as the integrator cancelling a
// plant zero at s = 0 (JJJR63).
func pidTuningKeepsIntegral(c PID2, hasI bool) bool {
	return !hasI || c.Ki != 0
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

// TunePID designs a SISO negative-feedback controller. Unlike the
// closed-form Pidtune, this operation enforces finite targets,
// bounded computation and explicit achieved-target/stability evidence.
//
// A zero CrossoverFrequency is capped by the loop delay τ at (180°−PM)/(2τ),
// as Pidtune. On a plant stable but for integrators every nonzero gain takes
// the sign of their residue, as in MATLAB pidtune.
// For a stable plant every nonzero gain takes the sign of its DC
// gain; when that leaves a default-crossover PD or PDF design infeasible, the
// crossover moves to where the plant phase is 15°, 30°, 45° or 60° below
// −180°+PM, the lead same-sign gains supply; when those and the margin floor
// at the default crossover fail, to where it is 7.5° or 2.5° below if that is
// within an octave of a lightly damped mode, since on a resonance, where the
// phase swings fast, only a small lead can leave the crossings on the peak
// clear of −1. Likewise, when a default PI
// design fails at the default crossover, also under the margin floor, unless the
// delayed stable plant's back-off below handles it, the crossover moves to
// the frequency nearest the default, on either side, where the plant phase
// is 15°, 30°, 45° or 60° above −180°+PM, the lag PI gains supply (for an
// unknown gain sign, of either sign·G); this is how an unstable or
// integrating plant needing lead at its own crossover gets a PI design that
// still meets PM±3°, where Pidtune keeps the crossover and gives up margin.
// A default or moved-crossover design must keep |S| <= 4 (2.5 dB gain
// margin) at every sampled frequency, since a margin met at the crossover
// alone does not bound it there; an explicit-crossover design is not bound
// but carries a warning when its sampled |S| exceeds 4; an undelayed plant's default design is
// searched unbounded first and under the bound only when it exceeds it.
// For a continuous plant without internal delays the samples run past 100·wc until a Markov-parameter bound gives |L| <= ¾.
// The samples also reach down to a tenth of the slowest plant pole or zero
// and, discrete, up to π/Dt, where a peak off the band hides. Only when neither the default nor the moved crossovers
// meet PM±3° does the margin become a floor at the original crossover, as in
// pidtune: the design takes the nearest feasible 6° band above PM, with
// PM−3° at every crossover over the plant's dynamics. Bands whose phase is
// out of reach with |L| ≥ 1 below the crossover are skipped; the bands stop
// at one whose margin-meeting candidates are all unstable and, under the
// |S| bound, once the excess stops falling, so a feasible band can be
// missed. A continuous delayed loop with an ideal derivative must keep
// limsup |L(jω)| <= ¾ (|S| <= 4 there), so a near-neutral loop is rejected
// even when its sampled margins look healthy; a delay-free one must keep
// 1+L(∞) ≥ ¼ (|S(∞)| ≤ 4), an |S| bound the search path may cross, and
// 1+L(∞) > 0 on that path too, since below 0 every derivative filter
// destabilizes it.
//
// When both fail for a default-crossover PI, PID or PIDF design on a stable
// delayed plant, and either a candidate exceeded |S| <= 4 or a lightly damped
// mode lies within a factor 100 of the default, the crossover backs off in
// half octaves, down to 1/1024 of the default, under |S| <= 4 sampled up to
// 100× the default. Each crossover tried targets PM±3° where the plant phase
// can reach it, then only the floor's lowest reachable band. Depths are tried
// in a fixed order within one evaluation budget: the two shallowest, then the
// two deepest, then depths 9, 12 and 16 (crossovers 2^-4.5, 2^-6 and 2^-8 of
// the default); if all fail, the back-off ends. Otherwise the depths between
// the shallow pair and the first that succeeded are scanned shallowest first
// and the first design is returned, or that one if the budget runs out: a
// lightly damped resonance near the delay cap can leave only a low-bandwidth
// design, as pidtune returns. A default-crossover PD, PDF, PID or PIDF
// design on a delay-free plant backs off the same way when it fails; on a
// plant that is not stable or has zero DC gain, a failed back-off then
// raises the crossover in half octaves up to 1024× the default, below the
// Nyquist frequency, within one more budget, and returns the first design,
// as Pidtune does: an unstable pole above the default, as on 1/(s(s−1)),
// needs crossovers past it.
//
// Another gain crossover whose wrapped margin is −floor or below has ∠L in
// (0°, 180° − floor], at least the floor from −1 the other way round: it
// passes only when a pole or Nyquist check certifies the loop stable; with
// sampled data only, its wrapped margin stands. A free P or I design on
// sampled data checks, against its own margin less 3°, only the other
// crossovers the grid finds, since the refined scan's additions near a
// resonance would reject robust designs with nothing to certify them. When
// neither wc0 nor the unit loop's 60° point gives a free P design with
// sampled |S| <= 4, as on an unstable plant whose phase caps the margin
// below 60° or a resonance at the 60° point, crossovers 2^(j/2)·wc0 for j
// from 6 down to −20 are tried and the design whose margin is nearest 60° is
// returned with a warning; free I instead backs off below wc0. Evidence
// reports wrapped margins, as allmargin does, and the achieved margin is the
// least so evaluated. An unattainable target names the crossover nearest to
// passing that failed.
//
// A stable delay-free continuous plant with a real zero within the stability
// tolerance left of s = 0 leaves every I, PI, PID or PIDF loop a closed-loop
// pole there (pidTuningIntegralZero), so those families are searched only at
// the requested or default crossover and fail with a warning naming the zero.
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
	plant = pidTuningExplicitPlant(plant)
	if opts.UnstablePoles != nil {
		return nil, fmt.Errorf("TunePID: UnstablePoles applies only to TunePIDFRD: %w", ErrOptionUnsupported)
	}
	eval, err := newSISOEval(plant)
	if err != nil {
		return nil, fmt.Errorf("TunePID: %w", err)
	}
	screen, err := newPIDStabilityScreen(plant, eval, pidTuningMaxTail)
	if err != nil {
		return nil, fmt.Errorf("TunePID: %w", err)
	}
	// resp falls back to FreqResponse's solve where eval's fails or is not
	// finite, as beside rounded multiple poles at z = 1 or s = 0 where it
	// resolves the response over decades eval cannot; only a sample neither
	// resolves fails eval. fellBack records a fallback sample: a search
	// failing with one tells nothing of its depth, which stays blind.
	pp := &pidtunePlantPoint{}
	fellBack := false
	resp := func(w float64) complex128 {
		saved := eval.err
		eval.err = nil
		g := eval.at(w)
		if eval.err != nil || !pidFiniteComplex(g) {
			if h, err := pp.at(plant, w); err == nil && pidFiniteComplex(h) {
				g, eval.err, fellBack = h, nil, true
			}
		}
		if saved != nil {
			eval.err = saved
		}
		return g
	}
	tryResp := func(w float64) (complex128, bool) { return tryEval(eval, func() complex128 { return resp(w) }) }
	var dcEdge, nyquistEdge complex128
	if g, ok := eval.resolve(0); ok {
		dcEdge = g
	}
	if plant.Dt > 0 {
		if g, ok := eval.resolve(math.Pi / plant.Dt); ok {
			nyquistEdge = g
		}
	}
	modes := sync.OnceValue(func() []complex128 { return pidTuningModes(screen.model) })
	loopTail := sync.OnceValue(func() func(*PID2, float64) float64 { return pidTuningLoopTail(plant) })
	span := sync.OnceValues(func() (float64, float64) { return pidTuningDynamicsSpan(screen.model) })
	sensCache := map[float64]complex128{}
	tau := sisoLoopDelay(plant)
	if plant.Dt > 0 {
		tau *= plant.Dt
	}
	// minHigh keeps a backed-off crossover's band over the default one,
	// where the resonance that forced the back-off sits.
	// sRejected records a search that failed with candidates over the |S|
	// bound.
	// probing marks the back-off's searches.
	// resolveFloor raises an explicit crossover's band floor off samples
	// the solve cannot resolve.
	used, minHigh, sRejected, probing, resolveFloor := 0, 0., false, false, false
	residueSign := 0.
	// lastPlant is the last search's plant, for the explicit-crossover
	// |S| warning.
	var lastPlant pidTuningPlant
	var tuneAt func(wc, gainSign float64, bound pidSBound, floor bool) (*PIDTuningResult, error)
	tuneAt = func(wc, gainSign float64, bound pidSBound, floor bool) (*PIDTuningResult, error) {
		opts.CrossoverFrequency, fellBack = wc, false
		low := wc / 100
		if resolveFloor {
			low = pidTuningResolvedFloor(tryResp, low, wc)
		}
		p := pidTuningPlant{dt: plant.Dt, low: low, high: math.Max(wc*100, minHigh), at: resp, try: tryResp, stability: PIDStabilitySampledOnly, gainSign: gainSign, residueSign: residueSign, tau: tau, zeroScan: true, dcEdge: dcEdge, nyquistEdge: nyquistEdge}
		p.loopTail = loopTail()
		if bound == pidSSearch {
			p.maxSensitivity = pidTuningMaxSensitivity
		}
		p.modes = modes()
		if lo, _ := span(); finitePID(lo) && lo > 0 {
			p.zeroLow = lo / 10
		}
		if floor {
			lo, hi := span()
			p.crossLow, p.crossHigh = math.Min(p.low, lo/10), math.Max(p.high, hi*10)
		}
		if plant.Dt > 0 {
			p.high = math.Min(p.high, .99*math.Pi/plant.Dt)
			p.crossHigh = math.Min(p.crossHigh, .99*math.Pi/plant.Dt)
		}
		stable, evidence, err := screen.at(wc, family)
		if err != nil {
			return nil, fmt.Errorf("TunePID: %w", err)
		}
		p.stable, p.stability = stable, evidence
		p.returnSensitivity, p.maxReturnSensitivity = screen.returnSensitivity, 1/(1-pidTuningMaxTail)
		if probing {
			p.floorRungs, p.minTfSeed = 1, true
		}
		if stable == nil {
			p.warnings = []string{"Exact delay retained; finite frequency samples do not certify global closed-loop stability."}
		}
		// wide scans |S| off the band too, down to a tenth of the slowest
		// plant root and, discrete, up to π/Dt, where a peak shows neither
		// in the band nor at its edges. A design passing the band's bound
		// is checked there and, failing, searched for again under it.
		wide := p
		if plant.Dt > 0 {
			wide.sensHigh = pidTuningSensNyquist * math.Pi / plant.Dt
		}
		if lo, _ := span(); lo/10 < p.low {
			wide.sensLow = lo / 10
		}
		wide.sensCache = sensCache
		lastPlant = wide
		res, err := pidTuningSearch(ctx, p, family, opts, floor, wc, eval)
		if bound == pidSSearch && err == nil && pidTuningSensitivityPeak(wide, *res.Controller) > pidTuningMaxSensitivity {
			sRejected = true
			budget, first := opts.MaxEvaluations, res.Evidence.Evaluations
			if budget == 0 {
				budget = 4096
			}
			if first >= budget {
				res, err = nil, &PIDTuningError{Op: "TunePID", Evidence: PIDTuningEvidence{RequestedCrossover: wc, RequestedPhaseMargin: opts.PhaseMargin, Evaluations: first, Termination: PIDTerminationEvaluationLimit, Stability: p.stability}}
			} else {
				o := opts
				o.MaxEvaluations = budget - first
				res, err = pidTuningSearch(ctx, wide, family, o, floor, wc, eval)
				if te, ok := errors.AsType[*PIDTuningError](err); ok {
					te.Evidence.Evaluations += first
				} else if err == nil {
					res.Evidence.Evaluations += first
				}
			}
		}
		if te, ok := errors.AsType[*PIDTuningError](err); ok {
			used += te.Evidence.Evaluations
			sRejected = sRejected || te.excess > 0
		} else if err == nil {
			used += res.Evidence.Evaluations
		}
		if bound == pidSCheck && err == nil {
			wide.maxSensitivity, wide.loopTail = pidTuningMaxSensitivity, loopTail()
			if pidTuningSensitivityPeak(wide, *res.Controller) > pidTuningMaxSensitivity {
				return tuneAt(wc, gainSign, pidSSearch, floor)
			}
		}
		return res, err
	}
	// evalFailed clears an evaluation failure off wc0, as near rounded
	// multiple poles at z = 1 or s = 0, so a moved crossover is skipped;
	// skipped also counts a moved crossover whose band is not finite or
	// whose failed search read a fallback sample.
	evalFailed := func() bool {
		if eval.err == nil {
			return false
		}
		eval.err = nil
		return true
	}
	skipped := func(err error) bool {
		blind := err != nil && fellBack
		fellBack = false
		return evalFailed() || blind || errors.As(err, new(*pidNonfiniteResponse))
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
	withFloor := func(wc, gainSign float64, bound pidSBound, res *PIDTuningResult, err error) (*PIDTuningResult, error) {
		if !errors.As(err, new(*PIDTuningError)) {
			return stamp(res, err)
		}
		r, e := tuneAt(wc, gainSign, bound, true)
		if fe, ok := errors.AsType[*PIDTuningError](e); ok {
			// A floor design rejected over the |S| bound says why the
			// target failed, so its warning is kept.
			if te, ok := errors.AsType[*PIDTuningError](err); ok && fe.excess > 0 {
				for _, w := range fe.Evidence.Warnings {
					if strings.Contains(w, "sensitivity peak") && !slices.Contains(te.Evidence.Warnings, w) {
						te.Evidence.Warnings = append(te.Evidence.Warnings, w)
					}
				}
			}
			return stamp(res, err)
		}
		return stamp(r, e)
	}
	// backOff lowers a failed default crossover by half octaves down to
	// wc0/1024 and keeps the first, shallowest, design: success need not be
	// monotone in depth, as a design can sit just below wc0 with failures
	// deeper. Each crossover tries PM±3° when pidTuningReachable admits it,
	// then the floor's lowest admitted band only, which keeps a failing
	// crossover near one search's minimum; a deep floor design sits there.
	// All its searches share one budget. After the two shallowest depths,
	// the two deepest, furthest from the delay and the resonance, then
	// depths 9, 12 and 16, gate the rest: the first of them to succeed opens
	// a scan of the depths between it and the shallow pair, and when all
	// fail the back-off ends, so an untunable plant costs seven depths, not
	// twenty. Designs between can come in short runs (depths 13-18, 9-10,
	// 12 alone on sweep plants), which the probes catch. When the scan
	// between runs out of budget, the gate's design is kept. A depth whose
	// response cannot be evaluated, as near multiple poles at z = 1, tells
	// nothing, so when one did and all gates fail, the depths left are
	// scanned shallowest first within the budget.
	// stepDesign tries the moved crossover wc of a back-off or raise
	// sharing the budget spent since start: PM±3° when pidTuningReachable
	// admits it, then the floor's lowest admitted band. blind reports a
	// search that could not be evaluated.
	stepDesign := func(wc, wc0, gainSign, pm float64, budget, start int, verb string) (r *PIDTuningResult, e error, blind bool) {
		o := opts
		o.CrossoverFrequency = wc
		p := pidTuningPlant{dt: plant.Dt, at: resp, gainSign: gainSign}
		// A walk failing on rounded poles at z = 1 or s = 0 leaves the
		// phase unknown; the depth is still tried.
		phase := math.NaN()
		if gainSign != 0 {
			phase = pidTuningUnwrappedPhase(resp, gainSign, wc, tau)
			evalFailed()
		}
		reachable := pidTuningReachable(p, family, o, pm, phase)
		for _, floor := range []bool{false, true} {
			if !floor && !reachable || used-start >= budget {
				continue
			}
			opts.MaxEvaluations = budget - (used - start)
			r, e := tuneAt(wc, gainSign, pidSSearch, floor)
			if skipped(e) {
				blind = true
				continue
			}
			if e == nil {
				hi := math.Max(minHigh, wc*100)
				if plant.Dt > 0 {
					hi = math.Pi / plant.Dt
				}
				r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("Default crossover %.4g %s to %.4g rad/s, where the design keeps sampled |S| <= %d up to %.4g rad/s.", wc0, verb, wc, pidTuningMaxSensitivity, hi))
				return r, nil, blind
			}
			if !errors.As(e, new(*PIDTuningError)) {
				return nil, e, blind
			}
		}
		return nil, nil, blind
	}
	backOff := func(wc0, gainSign, pm float64, res *PIDTuningResult, err error) (*PIDTuningResult, error) {
		minHigh, probing = wc0*100, true
		budget, start := opts.MaxEvaluations, used
		if budget == 0 {
			budget = 4096
		}
		const deepest = 20
		blind := false
		at := func(j int) float64 { return wc0 * math.Pow(2, -float64(j)/2) }
		design := func(j int) (*PIDTuningResult, error) {
			r, e, b := stepDesign(at(j), wc0, gainSign, pm, budget, start, "lowered")
			blind = blind || b
			return r, e
		}
		// first returns the first design over depths lo..hi, stepping toward
		// hi, and the depth it sits at.
		lowest := 0
		first := func(lo, hi int) (*PIDTuningResult, int, error) {
			step := 1
			if hi < lo {
				step = -1
			}
			for j := lo; j != hi+step && used-start < budget; j += step {
				lowest = max(lowest, j)
				if r, e := design(j); r != nil || e != nil {
					return r, j, e
				}
			}
			return nil, 0, nil
		}
		if r, _, e := first(1, 2); r != nil || e != nil {
			return stamp(r, e)
		}
		deep, gate, e := first(deepest, deepest-1)
		for _, j := range []int{9, 12, 16} {
			if deep != nil || e != nil || used-start >= budget {
				break
			}
			deep, gate, e = first(j, j)
		}
		if e != nil {
			return nil, e
		}
		for j := 3; deep == nil && blind && j < deepest-1 && used-start < budget; j++ {
			if j == 9 || j == 12 || j == 16 {
				continue
			}
			if r, e := design(j); r != nil || e != nil {
				return stamp(r, e)
			}
			lowest = max(lowest, j)
		}
		if deep == nil {
			if te, ok := errors.AsType[*PIDTuningError](err); ok && lowest > 0 {
				te.Evidence.Warnings = append(te.Evidence.Warnings, fmt.Sprintf("Crossovers down to %.4g rad/s were also tried.", at(lowest)))
			}
			return stamp(res, err)
		}
		if r, _, e := first(3, gate-1); r != nil || e != nil {
			return stamp(r, e)
		}
		return stamp(deep, nil)
	}
	// tuneOneGain designs a free P or I loop. Its one gain takes the plant's
	// margin at wc0, which a lightly damped mode can leave near zero at
	// another crossing, so the design must keep sampled |S| within the bound
	// over a band reaching past every mode; failing that, wc moves to the
	// nearest frequency where the unit loop has the pm margin, as MATLAB
	// targets 60° by moving wc; failing that, free P scans half octaves and
	// free I backs off below wc0.
	tuneOneGain := func(wc0, gainSign, pm float64) (*PIDTuningResult, error) {
		for _, m := range modes() {
			minHigh = math.Max(minHigh, 10*imag(m))
		}
		res, err := tuneAt(wc0, gainSign, pidSSearch, false)
		res, err = withFloor(wc0, gainSign, pidSSearch, res, err)
		if !errors.As(err, new(*PIDTuningError)) {
			return res, err
		}
		unit := PID2{Dt: plant.Dt, IFormula: opts.IFormula, B: 1, C: 1}
		if strings.EqualFold(string(family), string(PidtuneI)) {
			unit.Ki = 1
		} else {
			unit.Kp = 1
		}
		atUnit := func(w float64) complex128 { return resp(w) * pidTuningFeedback(unit, w) }
		signs := []float64{gainSign}
		if gainSign == 0 {
			signs = []float64{1, -1}
		}
		wHi := wc0 * 1e4
		if plant.Dt > 0 {
			wHi = math.Min(wHi, .98*math.Pi/plant.Dt)
		}
		for _, sign := range signs {
			lo := wc0 * 1e-6
			wc, _, ok := pidTuningPhaseWalk(atUnit, sign, -180+pm, lo, wHi, tau, true)
			// A walk failing at its floor or starting past the target, as
			// under a double integrator, cannot see the phase rise to it.
			if evalFailed() || !ok || wc < lo*1.03 {
				wc, ok = pidTuningNearestPhase(atUnit, sign, -180+pm, wc0, wc0*1e-4, wHi, tau)
			}
			if evalFailed() || !ok || math.Abs(math.Log(wc/wc0)) < 1e-9 {
				continue
			}
			r, e := tuneAt(wc, gainSign, pidSSearch, false)
			if skipped(e) {
				continue
			}
			if e == nil {
				r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("Default crossover %.4g moved to %.4g rad/s, where the one-gain loop has a %g° phase margin and sampled |S| <= %d.", wc0, wc, pm, pidTuningMaxSensitivity))
				return stamp(r, nil)
			}
			if !errors.As(e, new(*PIDTuningError)) {
				return nil, e
			}
		}
		budget, start := opts.MaxEvaluations, used
		if budget == 0 {
			budget = 4096
		}
		if !strings.EqualFold(string(family), string(PidtuneI)) {
			// Kp = ±1/|G(wc)| leaves the margin to the plant phase, which
			// an unstable plant can cap below pm and a resonance can leave
			// with |S| over the bound at the pm point, so wc scans half
			// octaves from 8·wc0 down to wc0/1024 and keeps the design
			// whose margin is nearest pm, the higher wc on a tie. The scan
			// starts above wc0 since the margin there can exceed pm.
			var best *PIDTuningResult
			hiTried, loTried := 0., 0.
			for j := 6; j >= -20 && used-start < budget; j-- {
				if j == 0 {
					continue
				}
				wc := wc0 * math.Pow(2, float64(j)/2)
				if plant.Dt > 0 && wc >= .98*math.Pi/plant.Dt {
					continue
				}
				hiTried, loTried = math.Max(hiTried, wc), wc
				opts.MaxEvaluations = budget - (used - start)
				r, e := tuneAt(wc, gainSign, pidSSearch, false)
				if skipped(e) {
					continue
				}
				if e != nil {
					if !errors.As(e, new(*PIDTuningError)) {
						return nil, e
					}
					continue
				}
				if best == nil || math.Abs(r.Evidence.AchievedPhaseMargin-pm) < math.Abs(best.Evidence.AchievedPhaseMargin-pm) {
					best = r
				}
			}
			if best == nil {
				if te, ok := errors.AsType[*PIDTuningError](err); ok && hiTried > 0 {
					te.Evidence.Warnings = append(te.Evidence.Warnings, fmt.Sprintf("Crossovers %.4g to %.4g rad/s were also tried.", loTried, hiTried))
				}
				return stamp(res, err)
			}
			best.Evidence.Warnings = append(best.Evidence.Warnings, fmt.Sprintf("Default crossover %.4g moved to %.4g rad/s, the scanned crossover whose %.3g° phase margin is nearest %g° with sampled |S| <= %d.", wc0, best.Evidence.AchievedCrossover, best.Evidence.AchievedPhaseMargin, pm, pidTuningMaxSensitivity))
			return stamp(best, nil)
		}
		// A free I rotates a lightly damped peak to −180°, where only a slow
		// integrator keeps |L| < 1, so wc steps down by half octaves to
		// wc0/1024. Ki ≈ wc/|G(0)| makes the peak |L| grow with wc, so the
		// shallowest design is the fastest robust one.
		lowest := 0.
		for j := 1; j <= 20 && used-start < budget; j++ {
			wc := wc0 * math.Pow(2, -float64(j)/2)
			lowest = wc
			opts.MaxEvaluations = budget - (used - start)
			r, e := tuneAt(wc, gainSign, pidSSearch, false)
			if skipped(e) {
				continue
			}
			if e == nil {
				hi := math.Max(wc*100, minHigh)
				if plant.Dt > 0 {
					hi = math.Pi / plant.Dt
				}
				r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("Default crossover %.4g lowered to %.4g rad/s, where the one-gain integrator keeps sampled |S| <= %d up to %.4g rad/s.", wc0, wc, pidTuningMaxSensitivity, hi))
				return stamp(r, nil)
			}
			if !errors.As(e, new(*PIDTuningError)) {
				return nil, e
			}
		}
		if te, ok := errors.AsType[*PIDTuningError](err); ok && lowest > 0 {
			te.Evidence.Warnings = append(te.Evidence.Warnings, fmt.Sprintf("Crossovers down to %.4g rad/s were also tried.", lowest))
		}
		return stamp(res, err)
	}
	fam := PidtuneType(strings.ToUpper(string(family)))
	zero, blocked := pidTuningIntegralZero(screen.model, fam)
	// integralBlocked tunes once at wc, as the full search finds nothing.
	integralBlocked := func(wc, gainSign float64) (*PIDTuningResult, error) {
		res, err := tuneAt(wc, gainSign, pidSFree, false)
		if te, ok := errors.AsType[*PIDTuningError](err); ok {
			te.Evidence.Warnings = append(te.Evidence.Warnings, fmt.Sprintf("Plant zero %.4g is within the stability tolerance of s = 0: every %s loop keeps a closed-loop pole between it and 0, so none is certified stable.", zero, fam))
		}
		return stamp(res, err)
	}
	if wc := opts.CrossoverFrequency; wc != 0 {
		gainSign := pidTuningDCSign(plant, tryResp, wc)
		if gainSign == 0 {
			residueSign = pidTuningIntegratorSign(plant, wc)
		}
		if blocked {
			return integralBlocked(wc, gainSign)
		}
		resolveFloor = true
		res, err := tuneAt(wc, gainSign, pidSFree, false)
		res, err = withFloor(wc, gainSign, pidSFree, res, err)
		if err == nil {
			q := lastPlant
			q.maxSensitivity, q.loopTail = pidTuningMaxSensitivity, loopTail()
			if peak := pidTuningSensitivityPeak(q, *res.Controller); peak > pidTuningMaxSensitivity {
				res.Evidence.Warnings = append(res.Evidence.Warnings, fmt.Sprintf("Sampled |S| peaks at %.3g, above %d (2.5 dB gain margin); an explicit crossover does not bound it.", peak, pidTuningMaxSensitivity))
			}
			eval.err = nil
		}
		return res, err
	}
	pm := opts.PhaseMargin
	if pm <= 0 || pm >= 180 || !finitePID(pm) {
		pm = 60
	}
	wc0, err := findCrossoverFreq(plant, 0, pm)
	if err != nil {
		return nil, fmt.Errorf("TunePID: %w", err)
	}
	gainSign := pidTuningDCSign(plant, tryResp, wc0)
	if gainSign == 0 {
		residueSign = pidTuningIntegratorSign(plant, wc0)
	}
	if blocked {
		return integralBlocked(wc0, gainSign)
	}
	if (fam == PidtuneP || fam == PidtuneI) && opts.PhaseMargin == 0 {
		return tuneOneGain(wc0, gainSign, pm)
	}
	defaultBound := pidSSearch
	if tau == 0 {
		defaultBound = pidSCheck
	}
	res, err := tuneAt(wc0, gainSign, defaultBound, false)
	if _, hasI, _, _, _ := pidTuningTerms(fam); hasI && fam != PidtuneI && gainSign != 0 && tau > 0 {
		res, err = withFloor(wc0, gainSign, pidSSearch, res, err)
		if !errors.As(err, new(*PIDTuningError)) {
			return res, err
		}
		// Only an |S| rejection or a lightly damped mode near wc0 suggests
		// that a lower crossover can succeed.
		resonant := slices.ContainsFunc(modes(), func(m complex128) bool { return imag(m) > wc0/100 && imag(m) < wc0*100 })
		if !sRejected && !resonant {
			return res, err
		}
		return backOff(wc0, gainSign, pm, res, err)
	}
	if fam == PidtunePI {
		res, err = withFloor(wc0, gainSign, defaultBound, res, err)
		if !errors.As(err, new(*PIDTuningError)) {
			return res, err
		}
		// PI gains of one sign add lag in (0°, 90°) only, so retry where
		// the plant phase leaves a moderate lag to supply, on either side
		// of wc0.
		signs := []float64{gainSign}
		if gainSign == 0 {
			signs = []float64{1, -1}
		}
		wLo, wHi := wc0*1e-4, wc0*1e4
		if plant.Dt > 0 {
			wHi = math.Min(wHi, .98*math.Pi/plant.Dt)
		}
		for _, lag := range []float64{15, 30, 45, 60} {
			for _, sign := range signs {
				wc, ok := pidTuningNearestPhase(resp, sign, -180+pm+lag, wc0, wLo, wHi, tau)
				if evalFailed() || !ok || math.Abs(math.Log(wc/wc0)) < 1e-9 {
					continue
				}
				r, e := tuneAt(wc, gainSign, pidSSearch, false)
				if skipped(e) {
					continue
				}
				if e == nil {
					r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("Default crossover %.4g moved to %.4g rad/s, where PI gains can meet the phase margin.", wc0, wc))
					return stamp(r, nil)
				}
				if !errors.As(e, new(*PIDTuningError)) {
					return nil, e
				}
			}
		}
		return stamp(res, err)
	}
	// raise, as Pidtune's raised, steps a failed default crossover of a
	// plant that is not stable or has zero DC gain up by half octaves to
	// wc0·2^10, below the Nyquist frequency, and keeps the first design:
	// an unstable pole above wc0, as on 1/(s(s−1)), needs crossovers past
	// it, where the lead the D term supplies meets the margin. All its
	// searches share one budget.
	raise := func(wc0, gainSign, pm float64, res *PIDTuningResult, err error) (*PIDTuningResult, error) {
		if stable, e := screen.model.IsStable(); e != nil || stable && !pidtuneZeroDC(screen.model) {
			return res, err
		}
		budget, start := opts.MaxEvaluations, used
		if budget == 0 {
			budget = 4096
		}
		highest := 0.
		for j := 1; j <= 20 && used-start < budget; j++ {
			wc := wc0 * math.Pow(2, float64(j)/2)
			if plant.Dt > 0 && wc >= .98*math.Pi/plant.Dt {
				break
			}
			highest = wc
			r, e, _ := stepDesign(wc, wc0, gainSign, pm, budget, start, "raised")
			if r != nil || e != nil {
				return stamp(r, e)
			}
		}
		if te, ok := errors.AsType[*PIDTuningError](err); ok && highest > 0 {
			te.Evidence.Warnings = append(te.Evidence.Warnings, fmt.Sprintf("Crossovers up to %.4g rad/s were also tried.", highest))
		}
		return stamp(res, err)
	}
	// undelayed backs off a failed undelayed PID or PD default crossover:
	// a plant crossover above a lag can leave the loop phase there out of
	// reach, as under a discrete double integrator, while a lower one has
	// a design.
	undelayed := func(res *PIDTuningResult, err error) (*PIDTuningResult, error) {
		if tau > 0 || !errors.As(err, new(*PIDTuningError)) {
			return res, err
		}
		if _, _, hasD, _, _ := pidTuningTerms(fam); !hasD {
			return res, err
		}
		maxEvaluations := opts.MaxEvaluations
		res, err = backOff(wc0, gainSign, pm, res, err)
		if !errors.As(err, new(*PIDTuningError)) {
			return res, err
		}
		opts.MaxEvaluations = maxEvaluations
		return raise(wc0, gainSign, pm, res, err)
	}
	if gainSign == 0 || fam != PidtunePD && fam != PidtunePDF || !errors.As(err, new(*PIDTuningError)) {
		return undelayed(withFloor(wc0, gainSign, defaultBound, res, err))
	}
	// Same-sign PD gains add lead in (0°, 90°) only, so retry where the
	// plant phase leaves a moderate lead to supply.
	wMax := wc0 * 1e4
	if plant.Dt > 0 {
		wMax = math.Min(wMax, .98*math.Pi/plant.Dt)
	}
	// ladder keeps, per kind, the last moved crossover's rejection warning.
	kinds := []string{"sensitivity peak", "crossed 0 dB again"}
	ladder := make([]string, len(kinds))
	// move returns the first design at the leads' crossovers, or nil;
	// resonant restricts it to crossovers within an octave of a lightly
	// damped mode.
	move := func(resonant bool, leads ...float64) (*PIDTuningResult, error) {
		for _, lead := range leads {
			wc, ok := pidTuningPhaseCrossing(resp, gainSign, -180+pm-lead, wc0*1e-6, wMax, tau)
			if evalFailed() || !ok || wc == wc0 {
				continue
			}
			if resonant && !slices.ContainsFunc(modes(), func(m complex128) bool { return imag(m) > wc/2 && imag(m) < wc*2 }) {
				continue
			}
			r, e := tuneAt(wc, gainSign, pidSSearch, false)
			if skipped(e) {
				continue
			}
			if e == nil {
				r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("Default crossover %.4g moved to %.4g rad/s, where same-sign PD gains can meet the phase margin.", wc0, wc))
				return stamp(r, nil)
			}
			var te *PIDTuningError
			if !errors.As(e, &te) {
				return nil, e
			}
			for _, w := range te.Evidence.Warnings {
				for i, kind := range kinds {
					if strings.Contains(w, kind) {
						ladder[i] = fmt.Sprintf("Moved crossover %.4g: %s", wc, w)
					}
				}
			}
		}
		return nil, nil
	}
	if r, e := move(false, 15, 30, 45, 60); r != nil || e != nil {
		return r, e
	}
	res, err = withFloor(wc0, gainSign, defaultBound, res, err)
	if !errors.As(err, new(*PIDTuningError)) {
		return res, err
	}
	// These leads come after the floor so that they change no design the
	// floor finds. Past a lightly damped peak the plant phase falls through
	// −180°…−210° within about 1%, where only near-pure D (lead 75°, 85°)
	// keeps every crossing's margin.
	if r, e := move(true, 7.5, 2.5, 75, 85); r != nil || e != nil {
		return r, e
	}
	if te, ok := errors.AsType[*PIDTuningError](err); ok {
		for _, w := range ladder {
			if w != "" && !slices.Contains(te.Evidence.Warnings, w) {
				te.Evidence.Warnings = append(te.Evidence.Warnings, w)
			}
		}
	}
	return undelayed(stamp(res, err))
}

// pidTuningSearch runs TunePID's search for p, the floor's when floor. A
// design whose controller zeros notch |L| through 1 below the band, as tiny
// Kp and Ki against Kd do, is searched for again with every candidate so
// checked, then, for a filtered family, from the least admissible Tf: the
// default Tf's lag can leave only such near-pure-D designs. Designs passing
// the check are unchanged. An evaluation failure of eval aborts it.
func pidTuningSearch(ctx context.Context, p pidTuningPlant, family PidtuneType, opts PIDTuningOptions, floor bool, wc float64, eval *sisoEval) (*PIDTuningResult, error) {
	tune := tunePID
	if floor {
		tune = tunePIDFloor
	}
	res, err := tune(ctx, "TunePID", p, family, opts)
	if eval.err != nil {
		return nil, fmt.Errorf("TunePID: %w", eval.err)
	}
	zr, ok := errors.AsType[*pidZeroRejected](err)
	if !ok {
		return res, err
	}
	budget := opts.MaxEvaluations
	if budget == 0 {
		budget = 4096
	}
	spent := zr.evaluations
	p.zeroSearch = true
	seeds := []bool{p.minTfSeed}
	if _, _, _, filtered, _ := pidTuningTerms(PidtuneType(strings.ToUpper(string(family)))); filtered && !p.minTfSeed {
		seeds = append(seeds, true)
	}
	for _, minTf := range seeds {
		if spent >= budget {
			break
		}
		o := opts
		o.MaxEvaluations = budget - spent
		p.minTfSeed = minTf
		res, err = tune(ctx, "TunePID", p, family, o)
		if eval.err != nil {
			return nil, fmt.Errorf("TunePID: %w", eval.err)
		}
		te, ok := errors.AsType[*PIDTuningError](err)
		if !ok {
			break
		}
		spent += te.Evidence.Evaluations
	}
	if te, ok := errors.AsType[*PIDTuningError](err); ok {
		te.Evidence.Evaluations = spent
	} else if err == nil {
		res.Evidence.Evaluations += spent
	} else if errors.As(err, new(*pidZeroRejected)) {
		res, err = nil, &PIDTuningError{Op: "TunePID", Evidence: PIDTuningEvidence{RequestedCrossover: wc, RequestedPhaseMargin: opts.PhaseMargin, Evaluations: spent, Termination: PIDTerminationEvaluationLimit, Stability: p.stability}}
	}
	return res, err
}

// pidTuningResolvedFloor returns lo, or, when try fails or is not finite
// there, as next to rounded multiple poles at z = 1 or s = 0 where |G| ≫ 1
// outruns the solve, one band step over the least frequency above lo where
// it resolves. A floor that would pass wc/2 stays lo, so the band reports
// the failure.
func pidTuningResolvedFloor(try func(float64) (complex128, bool), lo, wc float64) float64 {
	ok := func(w float64) bool {
		h, ok := try(w)
		return ok && pidFiniteComplex(h)
	}
	if ok(lo) {
		return lo
	}
	step := math.Pow(1e4, 1./240)
	for w := lo * step; w*step < wc/2; w *= step {
		if ok(w) && ok(w*step) {
			return w * step
		}
	}
	return lo
}

// pidTuningPhaseCrossing returns the first ω in [wLo, wHi] at which the
// phase of sign·G(jω), unwrapped from its wrapped value at wLo, falls to
// target degrees.
func pidTuningPhaseCrossing(at func(float64) complex128, sign, target, wLo, wHi, tau float64) (float64, bool) {
	w, _, ok := pidTuningPhaseWalk(at, sign, target, wLo, wHi, tau, false)
	return w, ok
}

// pidTuningNearestPhase returns the ω in [wLo, wHi] nearest wc0 in log
// distance at which the wrapped phase of sign·G(jω) crosses target degrees,
// walking out from wc0 on both sides. A crossing is accepted only between
// samples within 90° of the target, so a wrap at ±180° away from it is not
// one.
func pidTuningNearestPhase(at func(float64) complex128, sign, target, wc0, wLo, wHi, tau float64) (float64, bool) {
	dev := func(w float64) (float64, bool) {
		g := complex(sign, 0) * at(w)
		if !pidFiniteComplex(g) || g == 0 {
			return 0, false
		}
		return math.Remainder(cmplx.Phase(g)*180/math.Pi-target, 360), true
	}
	walk := func(up bool) (float64, bool) {
		w := wc0
		d, ok := dev(w)
		if !ok {
			return 0, false
		}
		if d == 0 {
			return w, true
		}
		for range 1 << 16 {
			if up && w >= wHi || !up && w <= wLo {
				return 0, false
			}
			var next float64
			if up {
				next = math.Min(w*1.02, wHi)
				if tau > 0 {
					next = math.Min(next, w+math.Pi/8/tau)
				}
			} else {
				next = math.Max(w/1.02, wLo)
			}
			nd, ok := dev(next)
			if !ok {
				return 0, false
			}
			if math.Signbit(d) != math.Signbit(nd) && math.Abs(d) < 90 && math.Abs(nd) < 90 {
				a, b, da := w, next, d
				for range 60 {
					m := math.Sqrt(a * b)
					dm, ok := dev(m)
					if !ok {
						return 0, false
					}
					if math.Signbit(dm) == math.Signbit(da) {
						a, da = m, dm
					} else {
						b = m
					}
				}
				return math.Sqrt(a * b), true
			}
			w, d = next, nd
		}
		return 0, false
	}
	lo, okLo := walk(false)
	hi, okHi := walk(true)
	switch {
	case okLo && okHi:
		if math.Log(wc0/lo) <= math.Log(hi/wc0) {
			return lo, true
		}
		return hi, true
	case okLo:
		return lo, true
	}
	return hi, okHi
}

// pidTuningUnwrappedPhase returns the phase in degrees of sign·G(jw),
// unwrapped from its wrapped value at w·1e-6, or NaN when the walk fails.
func pidTuningUnwrappedPhase(at func(float64) complex128, sign, w, tau float64) float64 {
	end, phase, ok := pidTuningPhaseWalk(at, sign, math.Inf(-1), w*1e-6, w, tau, false)
	if ok || end != w {
		return math.NaN()
	}
	return phase
}

// pidTuningPhaseWalk walks the unwrapped phase of sign·G(jω) up from wLo.
// It returns the first ω at which the phase falls to target degrees, or
// false with wHi and the phase there in degrees; a nonfinite or zero
// response ends the walk with ω = 0. The crossing is interpolated linearly
// within a step, which can turn 90° and so miss target by tens of degrees
// past a lightly damped mode, or with bisect found by bisection.
func pidTuningPhaseWalk(at func(float64) complex128, sign, target, wLo, wHi, tau float64, bisect bool) (float64, float64, bool) {
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
		if phase+step <= target && bisect {
			a, b := w, next
			for range 60 {
				m := math.Sqrt(a * b)
				gm := complex(sign, 0) * at(m)
				if !pidFiniteComplex(gm) || gm == 0 {
					return 0, 0, false
				}
				if phase+cmplx.Phase(gm/g) > target {
					a = m
				} else {
					b = m
				}
			}
			return math.Sqrt(a * b), target * 180 / math.Pi, true
		}
		if phase+step <= target {
			return w + (next-w)*(phase-target)/(-step), target * 180 / math.Pi, true
		}
		w, g, phase = next, ng, phase+step
	}
	return 0, 0, false
}

// tunePID outside the floor searches with the grid-only crossing scan, whose
// path the refined scan's extra rejections would change, and accepts the
// design only if the refined scan finds every crossing above the floor;
// otherwise it searches again under the refined scan with the rest of the
// budget.
func tunePID(ctx context.Context, op string, p pidTuningPlant, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
	if p.hasCrossFloor || p.crossScan != pidCrossAuto {
		return tunePIDScan(ctx, op, p, family, o)
	}
	if hasP, hasI, hasD, _, ok := pidTuningTerms(family); ok && p.stable == nil && o.PhaseMargin == 0 && !hasD && hasP != hasI {
		p.crossScan = pidCrossGrid
		return tunePIDScan(ctx, op, p, family, o)
	}
	p.crossScan = pidCrossCoarse
	r, err := tunePIDScan(ctx, op, p, family, o)
	cr, ok := errors.AsType[*pidCoarseRejected](err)
	if !ok {
		return r, err
	}
	budget := o.MaxEvaluations
	if budget == 0 {
		budget = 4096
	}
	p.crossScan = pidCrossRefined
	if cr.evaluations >= budget {
		return nil, &PIDTuningError{Op: op, Evidence: PIDTuningEvidence{RequestedCrossover: o.CrossoverFrequency, RequestedPhaseMargin: o.PhaseMargin, Evaluations: cr.evaluations, Termination: PIDTerminationEvaluationLimit, Stability: p.stability}}
	}
	o.MaxEvaluations = budget - cr.evaluations
	r, err = tunePIDScan(ctx, op, p, family, o)
	if te, ok := errors.AsType[*PIDTuningError](err); ok {
		te.Evidence.Evaluations += cr.evaluations
	} else if err == nil {
		r.Evidence.Evaluations += cr.evaluations
	}
	return r, err
}

func tunePIDScan(ctx context.Context, op string, p pidTuningPlant, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
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
	s := p.wantSign()
	// seedAt puts a controller with filter tf on the target.
	seedAt := func(tf float64) (c PID2, ib, db complex128) {
		c = PID2{Tf: tf, Dt: p.dt, IFormula: o.IFormula, DFormula: o.DFormula, B: 1, C: 1}
		ib, db = pidTuningBasis(c, wc)
		if hasP && hasI {
			if hasD {
				c.Kd = math.Copysign(math.Abs(imag(target)/imag(db))+.5*cmplx.Abs(target)/wc, real(target))
			}
			c.Ki = (imag(target) - c.Kd*imag(db)) / imag(ib)
			c.Kp = real(target) - c.Ki*real(ib) - c.Kd*real(db)
			// pidTuningReachable admits a band whose center lies up to 3°
			// past the integral edge of the same-sign PI cone; the exact
			// seed then has Kp of the wrong sign, and the feasible sliver
			// along that edge can fall between pattern steps, so seed on it.
			// The proportional edge would drop integral action.
			if !hasD && s != 0 && c.Kp != 0 && math.Copysign(1, c.Kp) != s && math.Copysign(1, c.Ki) == s {
				c.Kp, c.Ki = 0, s*cmplx.Abs(target)/cmplx.Abs(ib)
			}
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
			if s != 0 {
				if math.Copysign(1, c.Kp+c.Ki) != s {
					c = a
				}
			} else if pidPhaseDistance(p.at(wc)*pidTuningFeedback(a, wc), o.PhaseMargin) < pidPhaseDistance(p.at(wc)*pidTuningFeedback(c, wc), o.PhaseMargin) {
				c = a
			}
		}
		return c, ib, db
	}
	c, ib, db := seedAt(tf)
	if !finitePID(c.Kp) || !finitePID(c.Ki) || !finitePID(c.Kd) {
		return nil, fmt.Errorf("%s: singular controller basis at crossover %g: %w", op, wc, ErrSingularTransform)
	}
	const points = 241
	omega, plant := make([]float64, points), make([]complex128, points)
	for k := range omega {
		omega[k] = math.Exp(math.Log(p.low) + float64(k)*math.Log(p.high/p.low)/float64(points-1))
		plant[k] = p.at(omega[k])
		if !pidFiniteComplex(plant[k]) {
			return nil, fmt.Errorf("%s: %w", op, &pidNonfiniteResponse{omega[k]})
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
	refinedOmega := pidTuningCrossGrid(p, omega)
	crossOmega, crossings := refinedOmega, pidTuningCrossings
	if p.crossScan == pidCrossCoarse || p.crossScan == pidCrossGrid {
		crossOmega, crossings = omega, pidTuningCoarseCrossings
	}
	// gridFor gives a candidate's crossing grid and q the plant that scans
	// it, reusing samples off the band across candidates. Below the band
	// q reads a sample the solve cannot resolve, as next to rounded
	// multiple poles at z = 1 or s = 0, as infinite, since the integrators
	// keep |L| ≫ 1 there; above every band it reads one as NaN, which
	// brackets no crossing.
	q, gridFor := p, func(_ PID2, grid []float64) []float64 { return grid }
	if p.zeroScan {
		cache := map[float64]complex128{}
		top := math.Max(p.high, p.crossHigh)
		q.at = func(w float64) complex128 {
			if v, ok := cache[w]; ok {
				return v
			}
			if w >= p.low && w <= top || p.try == nil {
				return p.at(w)
			}
			if v, ok := p.try(w); ok {
				return v
			}
			if w > top {
				return cmplx.NaN()
			}
			return cmplx.Inf()
		}
		step := math.Log(p.high/p.low) / (points - 1)
		gridFor = func(c PID2, grid []float64) []float64 { return pidTuningZeroGrid(p, c, grid, step, cache) }
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
		// A P or I gain only scales the loop, so a seed off the phase band
		// means no candidate meets it; such a seed can sit on −1 at wc.
		if !hasD && hasP != hasI && pidPhaseDistance(h*pidTuningFeedback(c, wc), o.PhaseMargin) > 3 {
			return nil, &PIDTuningError{Op: op, Evidence: PIDTuningEvidence{RequestedCrossover: wc, RequestedPhaseMargin: o.PhaseMargin, Stability: p.stability, Warnings: append([]string(nil), p.warnings...)}}
		}
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
	// nearest is the candidate least below the margin floor at other
	// crossovers, summed over them, for a restart.
	var nearest PID2
	nearestViolation := math.Inf(1)
	// A free one-gain design has no margin to hold at other crossovers:
	// with a model-based stability certificate it needs only positive
	// margins, as pidTuningCrossingMargin evaluates them, and reports its
	// lowest. Sampled responses keep the floor, since their wrapped phase
	// can hide a negative margin, at the grid's crossings (pidCrossGrid).
	relaxed := oneGain && p.stable != nil
	floor := o.PhaseMargin - 3
	if p.hasCrossFloor {
		floor = p.crossFloor
	}
	if relaxed {
		floor = 0
	}
	// violationOf reports whether a crossing's margin, as
	// pidTuningCrossingMargin evaluates it under reflect and as wrapped
	// otherwise, falls below the floor; worstAt and worst record the
	// crossing furthest below. reflected reports a distance from −1 that
	// holds only if the stability check passes. violation, which ranks
	// candidates for a restart, sums the wrapped margins' shortfall.
	violationOf := func(c PID2, crossings, margins []float64, reflect bool) (violation float64, violated bool, worstAt, worst float64, reflected bool) {
		worst = math.Inf(1)
		for i, m := range margins {
			if m < floor || relaxed && m <= 0 {
				violation += floor - m
			}
			margin := m
			if reflect {
				margin = pidTuningCrossingMargin(wc, crossings[i], m, floor, p.stable != nil)
			}
			reflected = reflected || p.stable != nil && margin != m
			if margin < floor || relaxed && margin <= 0 {
				violated = true
				if margin < worst {
					worstAt, worst = crossings[i], margin
				}
			}
		}
		return violation, violated, worstAt, worst, reflected
	}
	// crossingAt and crossingMargin describe nearest's worst crossing.
	var crossingAt, crossingMargin float64
	returnRejected, dcRejected, nyquistRejected := false, false, false
	sampledExcess := func(c PID2, returnExcess float64) float64 {
		if p.maxSensitivity > 0 {
			return math.Max(returnExcess, pidTuningSensitivityPeak(p, c)-p.maxSensitivity)
		}
		return returnExcess
	}
	// phaseRejected reports that evaluate's last candidate left the phase
	// band at wc.
	phaseRejected := false
	evaluate := func(candidate PID2) error {
		phaseRejected = false
		if evidence.Evaluations >= o.MaxEvaluations {
			return nil
		}
		if err := ctx.Err(); err != nil {
			return err
		}
		evidence.Evaluations++
		if !pidTuningKeepsIntegral(candidate, hasI) {
			return nil
		}
		if filtered && (candidate.Tf < math.Max(.001/wc, p.dt*.51) || candidate.Tf > 10/wc) {
			return nil
		}
		if !pidTuningSignConsistent(candidate, p.wantSign()) {
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
			phaseRejected = true
			return nil
		}
		if !pidTuningDCAdmissible(candidate, p.dcGain, p.high, p.highGain) {
			dcRejected = true
			return nil
		}
		returnExcess := 0.
		if p.returnSensitivity != nil {
			s := p.returnSensitivity(&candidate)
			if math.IsInf(s, 1) {
				return nil
			}
			returnExcess = math.Max(0, s-p.maxReturnSensitivity)
		}
		pidTuningSolveWeights(&candidate, o.Weights, omega, plant, wc)
		v := components(candidate)
		j := score(v)
		if !finitePID(j) || j >= bestScore {
			return nil
		}
		grid := crossOmega
		if p.zeroSearch {
			grid = gridFor(candidate, grid)
		}
		at, margins := crossings(q, candidate, grid, wc)
		violation, violated, worstAt, worst, reflected := violationOf(candidate, at, margins, true)
		// A crossing counted the far way round from −1 holds only for a
		// stable loop; otherwise its wrapped margin stands.
		known := false
		if !violated && reflected {
			stable, err := p.stable(&candidate)
			if err != nil {
				return fmt.Errorf("stability check: %w", err)
			}
			if known = stable; !stable {
				_, violated, worstAt, worst, _ = violationOf(candidate, at, margins, false)
			}
		}
		if violated {
			if violation < nearestViolation {
				nearest, nearestViolation = candidate, violation
				crossingAt, crossingMargin = worstAt, worst
			}
			return nil
		}
		excess, peaked := 0., false
		beats := func() bool {
			return excessFirst && (excess < incumbentExcess || excess == incumbentExcess && j < incumbentScore) || !excessFirst && j < incumbentScore
		}
		if p.stable != nil {
			// Past a stable incumbent, a stable candidate over the bound
			// changes nothing unless it would become incumbent or lower
			// leastExcess, so the costly stability check is skipped.
			if (p.maxSensitivity > 0 || returnExcess > 0) && finitePID(incumbentScore) && sensitivityRejected {
				excess, peaked = sampledExcess(candidate, returnExcess), true
				if excess > 0 && excess >= leastExcess && !beats() {
					return nil
				}
			}
			stabilityChecked = true
			if !known {
				stable, err := p.stable(&candidate)
				if err != nil {
					return fmt.Errorf("stability check: %w", err)
				}
				if !stable {
					return nil
				}
			}
			stableFound = true
		}
		if !peaked {
			excess = sampledExcess(candidate, returnExcess)
		}
		// A loop the sampled Nyquist count finds unstable still moves the
		// search, as one over the sensitivity bound does, so the path to a
		// stable optimum through it is kept.
		nyquist := pidTuningFRDNyquist(candidate, p)
		if nyquist {
			leastExcess = math.Min(leastExcess, excess)
		}
		if beats() {
			incumbent, incumbentScore, incumbentExcess = candidate, j, excess
		}
		if !nyquist {
			nyquistRejected = true
			return nil
		}
		if excess > 0 {
			sensitivityRejected = true
			returnRejected = returnRejected || returnExcess > 0
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
		incumbentScore, incumbentExcess, nearestViolation = math.Inf(1), math.Inf(1), math.Inf(1)
		evidence.Termination = PIDTerminationConverged
		// A seed is feasible for the pass's ranking: any incumbent when
		// ranked by score, one within the bound when ranked by excess.
		seeded := func() bool {
			return excessFirst && finitePID(bestScore) || !excessFirst && finitePID(incumbentScore)
		}
		seedFrom := func(c PID2, ib, db complex128) error {
			if err := evaluate(c); err != nil || !hasP || !hasI || !hasD {
				return err
			}
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
					return err
				}
			}
			return nil
		}
		err := seedFrom(c, ib, db)
		if err == nil && filtered && p.minTfSeed && !seeded() {
			err = seedFrom(seedAt(math.Max(.001/wc, p.dt*.51)))
		}
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		if pass == first {
			evidence.SeedObjective = bestScore
		}
		seedViolated := finitePID(nearestViolation)
		// Signed, scaled coordinate pattern search; each candidate is gain-normalized
		// at wc and must preserve target phase and model-based stability when available.
		// When a seed failed only the margins at other crossovers and the
		// search from c finds no incumbent, it restarts from the candidate
		// least below the floor, as such a seed can sit next to the
		// feasible set; the restart counts a smaller violation as progress.
		// Floor bands, which share one budget, skip it. Under a model-based
		// stability check a PIDF incumbent is then polished by Kd and Tf
		// steps with Kp and Ki re-solved to hold C(jwc), as the coordinate
		// steps leave the phase band there; the polish reaches loops whose
		// resonances sparse frequency data do not resolve.
		for restart := range 3 {
			current := c
			if restart == 1 {
				if p.hasCrossFloor || !seedViolated || finitePID(incumbentScore) {
					continue
				}
				current = nearest
			}
			if restart == 2 && !(hasP && hasI && hasD && filtered && p.stable != nil && finitePID(incumbentScore)) {
				break
			}
			step := .5
			for round := 0; round < 64 && evidence.Evaluations < o.MaxEvaluations; round++ {
				previous, previousExcess, previousViolation := incumbentScore, incumbentExcess, nearestViolation
				if finitePID(incumbentScore) {
					current = incumbent
				} else if restart == 1 {
					current = nearest
				}
				// A Tf step without Kd is a null move whose rounding-level
				// score change would steer the search. Tf still decides
				// where a Kd step lands, so under an exact stability check
				// the Tf axis then steps Kd too, at a Tf stride that does
				// not shrink, as no score ranks Tf there, and with Kp and
				// Ki put back on the incumbent's phase at wc if off the
				// band. The probe can reach loops near the stability
				// boundary, which sampled checks cannot certify.
				for axis := range 6 {
					diagonal := axis == 3 && current.Kd == 0
					if (restart == 2) != (axis >= 4) || axis == 0 && !hasP || axis == 1 && !hasI || axis == 2 && !hasD ||
						axis == 3 && (!filtered || diagonal && p.stable == nil) || axis == 5 && current.Kd == 0 {
						continue
					}
					kdSigns := []float64{0}
					if diagonal {
						kdSigns = []float64{-1, 1}
					}
					for _, sign := range []float64{-1, 1} {
						for _, kdSign := range kdSigns {
							candidate := current
							scale := 1 / cmplx.Abs(h)
							switch axis {
							case 0:
								candidate.Kp += sign * step * scale
							case 1:
								candidate.Ki += sign * step * scale * wc
							case 2, 4:
								candidate.Kd += sign * step * scale / wc
							case 3, 5:
								tfStep := step
								if diagonal {
									tfStep = math.Max(step, .5)
								}
								candidate.Tf *= math.Exp(sign * tfStep)
								candidate.Kd += kdSign * step * scale / wc
							}
							if axis >= 4 {
								held := pidTuningFeedback(current, wc)
								ib, db := pidTuningBasis(candidate, wc)
								candidate.Ki = (imag(held) - candidate.Kd*imag(db)) / imag(ib)
								candidate.Kp = real(held) - candidate.Ki*real(ib) - candidate.Kd*real(db)
							}
							if err := evaluate(candidate); err != nil {
								return nil, fmt.Errorf("%s: %w", op, err)
							}
							if diagonal && hasP && hasI && phaseRejected {
								on := pidTuningFeedback(current, wc)
								ib, db := pidTuningBasis(candidate, wc)
								candidate.Ki = (imag(on) - candidate.Kd*imag(db)) / imag(ib)
								candidate.Kp = real(on) - candidate.Ki*real(ib) - candidate.Kd*real(db)
								if err := evaluate(candidate); err != nil {
									return nil, fmt.Errorf("%s: %w", op, err)
								}
							}
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
				if restart == 1 && !finitePID(incumbentScore) {
					improved = !(nearestViolation >= previousViolation-1e-6*math.Max(1, previousViolation))
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
		}
		if evidence.Evaluations >= o.MaxEvaluations {
			evidence.Termination = PIDTerminationEvaluationLimit
		}
	}
	if !finitePID(bestScore) {
		if signRejected && p.gainSign == 0 {
			evidence.Warnings = append(evidence.Warnings, "Candidates meeting the phase target needed a gain of sign opposite to the integrator's residue; such designs are rejected.")
		} else if signRejected {
			evidence.Warnings = append(evidence.Warnings, "Candidates meeting the phase target needed a gain of sign opposite to the stable plant's DC gain; such designs are rejected.")
		}
		if dcRejected {
			evidence.Warnings = append(evidence.Warnings, "Candidates meeting the phase target had a real unstable closed-loop root on the stable plant the data are taken to come from (no unstable poles given); such designs are rejected.")
		}
		if nyquistRejected {
			evidence.Warnings = append(evidence.Warnings, "Candidates meeting the phase target encircled −1 on the sampled Nyquist plot, or had not rolled off where the data stop, so the sampled Nyquist count could not certify a stable loop on the stable plant the data are taken to come from (no unstable poles given); such designs are rejected.")
		}
		if returnRejected {
			evidence.Warnings = append(evidence.Warnings, fmt.Sprintf("Candidates meeting the phase target had a high-frequency sensitivity 1/(1+Kd·CB) above %g with the ideal derivative; such designs are rejected.", p.maxReturnSensitivity))
		}
		if sensitivityRejected && p.maxSensitivity > 0 {
			evidence.Warnings = append(evidence.Warnings, fmt.Sprintf("Candidates meeting the phase target had a sensitivity peak above %g at the default crossover; such designs are rejected.", p.maxSensitivity))
		}
		if finitePID(nearestViolation) {
			evidence.Warnings = append(evidence.Warnings, fmt.Sprintf("Candidates meeting the phase target crossed 0 dB again with a margin below the %g° floor (closest candidate: %.4g° at %.4g rad/s); such designs are rejected.", floor, crossingMargin, crossingAt))
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
	evidenceCrossings := pidTuningCrossings
	if p.crossScan == pidCrossGrid {
		evidenceCrossings = pidTuningCoarseCrossings
	}
	evidenceOmega := gridFor(best, refinedOmega)
	evidence.GainCrossovers, evidence.PhaseMargins = evidenceCrossings(q, best, evidenceOmega, wc)
	if _, violated, _, _, _ := violationOf(best, evidence.GainCrossovers, evidence.PhaseMargins, true); violated && p.crossScan == pidCrossCoarse {
		return nil, &pidCoarseRejected{evaluations: evidence.Evaluations}
	} else if violated && !p.zeroSearch && p.crossScan != pidCrossGrid {
		return nil, &pidZeroRejected{evaluations: evidence.Evaluations}
	}
	for i, margin := range evidence.PhaseMargins {
		margin = pidTuningCrossingMargin(wc, evidence.GainCrossovers[i], margin, floor, p.stable != nil)
		evidence.AchievedPhaseMargin = math.Min(evidence.AchievedPhaseMargin, margin)
	}
	evidence.FrequencyBand = [2]float64{evidenceOmega[0], evidenceOmega[len(evidenceOmega)-1]}
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
// phase is the unwrapped phase of gainSign·G(jwc) in degrees, or NaN, as
// always under residueSign, whose integrators leave the walk unanchored. The
// loop phase starts within ±90° at low frequency, so if |L| ≥ 1 below wc,
// a stable loop needs its unwrapped phase at wc above −180°; with phase
// known and an integral term, which puts |L| above 1 near DC, the band is
// judged unwrapped. This excludes designs whose |L| dips below 1 and
// recovers below wc. Without one, |L(0)| = |Kp·G(0)| can be below 1, and
// a loop crossing −180° there can be stable at any wrapped phase, so the
// band is judged wrapped, as when phase is NaN; wrapped, a cone wider than
// a half-plane reaches every phase.
func pidTuningReachable(p pidTuningPlant, family PidtuneType, o PIDTuningOptions, pm, phase float64) bool {
	hasP, hasI, hasD, filtered, ok := pidTuningTerms(family)
	if !ok || p.wantSign() == 0 {
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
	if hasI && !math.IsNaN(phase) {
		return want >= phase+lo-3-1e-9 && want <= phase+hi+3+1e-9
	}
	if hi-lo >= 180 {
		return true
	}
	want -= cmplx.Phase(complex(p.wantSign(), 0)*p.at(wc)) * 180 / math.Pi
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
	// least is the least positive excess over all bands, which the
	// returned error carries.
	used, excess, least := 0, 0., 0.
	phase := math.NaN()
	if p.gainSign != 0 {
		phase = pidTuningUnwrappedPhase(p.at, p.gainSign, o.CrossoverFrequency, p.tau)
	}
	last := &PIDTuningError{Op: op, Evidence: PIDTuningEvidence{RequestedCrossover: o.CrossoverFrequency, RequestedPhaseMargin: pm, Stability: p.stability}}
	rungs := 0
	for target := pm + 6; target < 180 && used < budget; target += 6 {
		if !pidTuningReachable(p, fam, o, target, phase) {
			continue
		}
		if p.floorRungs > 0 && rungs == p.floorRungs {
			break
		}
		rungs++
		o.PhaseMargin = target
		o.MaxEvaluations = budget - used
		r, err := tunePID(ctx, op, p, family, o)
		if zr, ok := errors.AsType[*pidZeroRejected](err); ok {
			zr.evaluations += used
		}
		if errors.As(err, &last) {
			used += last.Evidence.Evaluations
			if last.excess > 0 && (least == 0 || last.excess < least) {
				least = last.excess
			}
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
	last.excess = least
	return nil, last
}

func pidTuningBasis(c PID2, w float64) (integral, derivative complex128) {
	if c.Dt == 0 {
		s := complex(0, w)
		return 1 / s, s / (1 + complex(c.Tf, 0)*s)
	}
	return pidtuneIntegrator(c.IFormula, w, c.Dt), pidTuningDiscreteDerivative(c, 1/pidtuneIntegrator(c.DFormula, w, c.Dt))
}

// pidTuningDiscreteDerivative is the filtered derivative term for d, the
// inverse of the derivative formula's integrator.
func pidTuningDiscreteDerivative(c PID2, d complex128) complex128 {
	return d / (1 + complex(c.Tf, 0)*d)
}

func pidTuningFeedback(c PID2, w float64) complex128 {
	i, d := pidTuningBasis(c, w)
	return complex(c.Kp, 0) + complex(c.Ki, 0)*i + complex(c.Kd, 0)*d
}

// pidTuningDiscreteFeedback is pidTuningFeedback of a discrete c from its
// integral term and the inverse d of its derivative formula's integrator.
func pidTuningDiscreteFeedback(c PID2, integral, d complex128) complex128 {
	return complex(c.Kp, 0) + complex(c.Ki, 0)*integral + complex(c.Kd, 0)*pidTuningDiscreteDerivative(c, d)
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

// pidTuningIdealReturn returns 1+L(∞) = 1+Kd·CB, the high-frequency return
// difference of a continuous delay-free strictly proper SISO plant under an
// ideal derivative, or +Inf when D ≠ 0 makes |L(∞)| unbounded. A loop whose
// characteristic is Hurwitz while 1+L(∞) ≤ 0 is positive feedback at high
// frequency: any derivative filter or unmodelled roll-off Tf adds a root near
// −(1+L(∞))/Tf in the right half-plane, and |S(∞)| = 1/|1+L(∞)| is unbounded
// as it nears 0. The screen, for Pidtune and TunePID, rejects 1+L(∞) below
// pidTuningIdealReturnTol; TunePID's search also bounds |S(∞)| by
// 1/(1−pidTuningMaxTail), as that bounds the delayed chain.
func pidTuningIdealReturn(model *System, kd float64) float64 {
	if model.D.At(0, 0) != 0 {
		return math.Inf(1)
	}
	if n, _, _ := model.Dims(); n == 0 {
		return 1
	}
	return 1 + kd*mat.Dot(model.C.RowView(0), model.B.ColView(0))
}

// pidTuningIdealReturnTol is the least 1+L(∞) treated as positive: below it
// cancellation in 1+Kd·CB leaves its sign unresolved.
const pidTuningIdealReturnTol = 0x1p-26

// Ideal derivative controllers may be improper on their own while their
// closed-loop characteristic is well-defined. Test the characteristic
// directly, of the loop with model, transfer its transfer function, delayed
// by lag samples. Every loop takes one chain: a float verdict, exact
// Schur–Cohn (pidTuningIdealExact) when it is in doubt and n+lag is within
// pidTuningExactLagOrder, else pidTuningIdealRealizationFloat (whose
// Bauer–Fike discs certify loops the polynomial error model leaves in
// doubt), else the loop fails closed as unstable. TransferFunction drops or
// corrupts near-zero poles and zeros, and the float verdict on its
// characteristic then decides confidently wrong, so above
// pidTuningExactOrder the transfer is never used and the realization's
// float verdict comes first. Up to pidTuningExactOrder, a delayed loop, or
// one whose transfer is of lower degree than model or fails, has its
// characteristic built exactly from model's realization, and the float
// verdict on the rounded exact characteristic comes first. transfer is
// called only for loops out of both scopes, undelayed loops, or a rounded
// characteristic that overflows beyond pidTuningExactLagOrder.
func pidTuningIdealStable(model *System, transfer func() (*TransferFuncResult, error), c *PID2, lag int) (bool, error) {
	n, _, _ := model.Dims()
	realizable := pidTuningIdealRealizable(model, c)
	if n > pidTuningExactOrder && realizable {
		if stable, decided := pidTuningIdealRealizationFloat(model, c, lag); decided {
			return stable, nil
		}
		exact, ok := pidTuningIdealExact(model, c, lag)
		return exact && ok, nil
	}
	unresolved := func() bool {
		stable, decided := false, false
		if realizable {
			stable, decided = pidTuningIdealRealizationFloat(model, c, lag)
		}
		return stable && decided
	}
	inScope := pidTuningIdealInScope(model, c)
	var tr *TransferFuncResult
	if !inScope || lag == 0 {
		var err error
		if tr, err = transfer(); err != nil && !inScope {
			return false, err
		}
	}
	if inScope && (tr == nil || len(tr.TF.Den[0]) <= n) {
		ch := pidTuningIdealCharacteristicIn(model, c, lag)
		f, ok := dyadicPolyFloat(ch)
		if ok {
			stable, decided, err := pidTuningIdealFloat(f, c.Dt == 0)
			if err != nil || decided {
				return stable, err
			}
		}
		if n+lag <= pidTuningExactLagOrder {
			return pidTuningDyadicStable(ch, c.Dt == 0), nil
		}
		if ok {
			return unresolved(), nil
		}
		var err error
		if tr, err = transfer(); err != nil {
			return false, err
		}
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
	characteristic := slices.Concat(Poly(tr.TF.Den[0]).Mul(denominator), make(Poly, lag)).Add(Poly(tr.TF.Num[0][0]).Mul(numerator))
	stable, decided, err := pidTuningIdealFloat(characteristic, c.Dt == 0)
	if err != nil || decided {
		return stable, err
	}
	if exact, ok := pidTuningIdealExact(model, c, lag); ok {
		return exact, nil
	}
	return unresolved(), nil
}

// pidTuningIdealRealizationFloat is the float verdict of pidTuningIdealStable
// above pidTuningExactOrder: the eigenvalues of the closed-loop state matrix
// of model, delayed by a lag-sample output shift chain, under the ideal
// derivative kd·λ + k0 plus a one-state integrator, with u solved out of
// u = −(kd·λy + …) when D' = 0 (dividing by r = 1+kd·C'B') or kept as a
// state when D' ≠ 0 (dividing by kd·D). Its order is the degree of
// pidTuningIdealExact's characteristic, so its eigenvalues are that
// characteristic's roots. The error scale bounds the formed entries by
// their absolute terms, as the elimination cancels. pidTuningFloatVerdict
// decides first, then pidTuningEigenDiscVerdict; decided is false when
// both are in doubt or the realization degenerates.
func pidTuningIdealRealizationFloat(model *System, c *PID2, lag int) (stable, decided bool) {
	am, absm, ok := pidTuningIdealRealization(model, c, lag)
	if !ok {
		return false, false
	}
	order, _ := am.Dims()
	poles, err := descriptorPolicy{}.poles("", am, order)
	if err != nil {
		return false, false
	}
	for _, p := range poles {
		if !pidFiniteComplex(p) {
			return false, false
		}
	}
	if stable, decided = pidTuningFloatVerdict(poles, c.Dt == 0, mat.Norm(absm, math.Inf(1))); decided {
		return stable, true
	}
	return pidTuningEigenDiscVerdict(am, c.Dt == 0, pidTuningPoleErrorScale*float64(order)*0x1p-52*mat.Norm(absm, 2))
}

// pidTuningIdealRealization returns pidTuningIdealRealizationFloat's
// closed-loop state matrix and the bound on its entries' absolute terms;
// ok is false when it degenerates or overflows.
func pidTuningIdealRealization(model *System, c *PID2, lag int) (a, abs *mat.Dense, ok bool) {
	n, _, _ := model.Dims()
	kd, k0, k0Abs := c.Kd, c.Kp, math.Abs(c.Kp)
	nx, ax, cx, dx := 0, 0.0, 0.0, 0.0
	if c.Ki != 0 {
		nx, cx = 1, c.Ki
	}
	if c.Dt != 0 {
		if c.DFormula != ForwardEuler {
			return nil, nil, false
		}
		kd, k0, k0Abs = c.Kd/c.Dt, c.Kp-c.Kd/c.Dt, math.Abs(c.Kp)+math.Abs(c.Kd/c.Dt)
		ax, cx = 1, c.Ki*c.Dt
		switch c.IFormula {
		case BackwardEuler:
			dx = c.Ki * c.Dt
		case Trapezoidal:
			dx = c.Ki * c.Dt / 2
		}
	}
	dc, dcAbs := k0+dx, k0Abs+math.Abs(dx)
	np := n + lag
	ap, bp, cp := make([]float64, np*np), make([]float64, np), make([]float64, np)
	for i := range n {
		copy(ap[i*np:i*np+n], model.A.RawRowView(i))
		bp[i] = model.B.At(i, 0)
	}
	d := model.D.At(0, 0)
	if lag == 0 {
		copy(cp, model.C.RawRowView(0))
	} else {
		copy(ap[n*np:n*np+n], model.C.RawRowView(0))
		bp[n] = d
		for i := n + 1; i < np; i++ {
			ap[i*np+i-1] = 1
		}
		cp[np-1], d = 1, 0
	}
	ca, caAbs := make([]float64, np), make([]float64, np)
	var cb, cbAbs float64
	for i, ci := range cp {
		if ci == 0 {
			continue
		}
		for j := range np {
			ca[j] += ci * ap[i*np+j]
			caAbs[j] += math.Abs(ci * ap[i*np+j])
		}
		cb += ci * bp[i]
		cbAbs += math.Abs(ci * bp[i])
	}
	nu := 0
	if d != 0 {
		if kd == 0 {
			return nil, nil, false
		}
		nu = 1
	}
	order := np + nx + nu
	am, absm := make([]float64, order*order), make([]float64, order*order)
	a, abs = mat.NewDense(order, order, am), mat.NewDense(order, order, absm)
	for i := range np {
		for j := range np {
			am[i*order+j] = ap[i*np+j]
			absm[i*order+j] = math.Abs(ap[i*np+j])
		}
	}
	if nx == 1 {
		for j := range np {
			am[np*order+j], absm[np*order+j] = cp[j], math.Abs(cp[j])
		}
		am[np*order+np], absm[np*order+np] = ax, ax
	}
	// g·x + h·ξ + q·u is u (D' = 0) or λu (D' ≠ 0).
	g, gAbs := make([]float64, np+nx+nu), make([]float64, np+nx+nu)
	var den, denAbs float64
	if nu == 0 {
		den, denAbs = 1+kd*cb, 1+math.Abs(kd)*cbAbs
	} else {
		den, denAbs = kd*d, math.Abs(kd*d)
		g[np+nx], gAbs[np+nx] = 1+kd*cb+dc*d, 1+math.Abs(kd)*cbAbs+dcAbs*math.Abs(d)
	}
	if den == 0 || !finitePID(den) {
		return nil, nil, false
	}
	for j := range np {
		g[j], gAbs[j] = kd*ca[j]+dc*cp[j], math.Abs(kd)*caAbs[j]+dcAbs*math.Abs(cp[j])
	}
	if nx == 1 {
		g[np], gAbs[np] = cx, math.Abs(cx)
	}
	for j := range g {
		v := -g[j] / den
		g[j], gAbs[j] = v, gAbs[j]/math.Abs(den)+math.Abs(v)*denAbs/math.Abs(den)
	}
	if nu == 0 {
		for i := range np {
			for j := range np + nx {
				am[i*order+j] += bp[i] * g[j]
				absm[i*order+j] += math.Abs(bp[i]) * gAbs[j]
			}
		}
	} else {
		for i := range np {
			am[i*order+np+nx], absm[i*order+np+nx] = bp[i], math.Abs(bp[i])
		}
		if nx == 1 {
			am[np*order+np+nx], absm[np*order+np+nx] = d, math.Abs(d)
		}
		copy(am[(np+nx)*order:], g)
		copy(absm[(np+nx)*order:], gAbs)
	}
	for _, v := range am {
		if !finitePID(v) {
			return nil, nil, false
		}
	}
	return a, abs, true
}

// pidTuningEigenDiscVerdict decides stability of a's eigenvalues by the
// Bauer–Fike theorem when pidTuningFloatVerdict's polynomial error model,
// which treats every nearby eigenvalue as a root cluster, leaves a loop of
// many well-separated modes in doubt: with V the unit right eigenvectors
// and s_i = |y_iᴴx_i| for unit left ones, κ(V) ≤ √(n·Σ1/s_i²), and every
// eigenvalue of a matrix within eta of a in the 2-norm lies in a disc of
// radius κ(V)·eta, padded by pidTuningPoleMarginSafety, about a computed
// one. The computed pairs are exact for a + R·V⁻¹, R = aV − VΛ, so eta
// grows by ‖R‖_F·‖V⁻¹‖_F ≤ ‖R‖_F·√(Σ1/s_i²). The discs must be pairwise
// disjoint, so each holds exactly one, and each must lie wholly inside or
// outside the exact test's boundary
// (Re s < −1e-10, |z| < 1−1e-10; inside also within poleStabilityTolerance).
func pidTuningEigenDiscVerdict(a *mat.Dense, continuous bool, eta float64) (stable, decided bool) {
	var eig mat.Eigen
	if !eig.Factorize(a, mat.EigenBoth) {
		return false, false
	}
	vals := eig.Values(nil)
	var vr, vl mat.CDense
	eig.VectorsTo(&vr)
	eig.LeftVectorsTo(&vl)
	n := len(vals)
	var sum, res float64
	col := make([]complex128, n)
	for i := range n {
		var nr, ri float64
		for k := range n {
			col[k] = vr.At(k, i)
			nr += real(col[k])*real(col[k]) + imag(col[k])*imag(col[k])
		}
		for k := range n {
			d := -vals[i] * col[k]
			for j, v := range a.RawRowView(k) {
				d += complex(v, 0) * col[j]
			}
			ri += real(d)*real(d) + imag(d)*imag(d)
		}
		res += ri / nr
	}
	for i := range n {
		var dot complex128
		var nr, nl float64
		for k := range n {
			r, l := vr.At(k, i), vl.At(k, i)
			dot += cmplx.Conj(l) * r
			nr += real(r)*real(r) + imag(r)*imag(r)
			nl += real(l)*real(l) + imag(l)*imag(l)
		}
		s := cmplx.Abs(dot) / math.Sqrt(nr*nl)
		if !(s > 0) {
			return false, false
		}
		sum += 1 / (s * s)
	}
	r := pidTuningPoleMarginSafety * math.Sqrt(float64(n)*sum) * (eta + math.Sqrt(sum*res))
	if !finitePID(r) {
		return false, false
	}
	for i, p := range vals {
		if !pidFiniteComplex(p) {
			return false, false
		}
		for _, q := range vals[i+1:] {
			if cmplx.Abs(p-q) <= 2*r {
				return false, false
			}
		}
	}
	const tol = 1e-10
	stable = true
	for _, p := range vals {
		lo, hi, inside := cmplx.Abs(p)-r, cmplx.Abs(p)+r, 1-tol
		outside := inside
		if continuous {
			lo, hi, outside = real(p)-r, real(p)+r, -tol
			inside = -tol * max(1, cmplx.Abs(p)+r)
		}
		if lo >= outside {
			return false, true
		}
		stable = stable && hi < inside
	}
	return stable, stable
}

// pidTuningIdealFloat is the float verdict of pidTuningIdealStable on
// characteristic, leading coefficient first; decided is false when the
// exact test must settle it.
func pidTuningIdealFloat(characteristic Poly, continuous bool) (stable, decided bool, err error) {
	for len(characteristic) > 1 && characteristic[0] == 0 {
		characteristic = characteristic[1:]
	}
	if len(characteristic) == 0 || characteristic[0] == 0 {
		return false, true, nil
	}
	poles, err := characteristic.Roots()
	if err != nil {
		return false, true, err
	}
	for _, pole := range poles {
		if !pidFiniteComplex(pole) {
			return false, true, nil
		}
	}
	norm := 1.0
	for _, v := range characteristic[1:] {
		norm += math.Abs(v / characteristic[0])
	}
	stable, decided = pidTuningFloatVerdict(poles, continuous, norm)
	return stable, decided, nil
}

// pidTuningIdealExact is pidTuningIdealStable's exact test on model's
// realization, in the scope of pidTuningExactStable: with Δ = det(λI−A),
// det(λI−A+BC) = Δ·(1+C(λI−A)⁻¹B), so N = det(λI−A+BC) − Δ + D·Δ is the
// plant numerator over Δ and the characteristic is Δ·λ^lag·den(c) + N·num(c).
func pidTuningIdealExact(model *System, c *PID2, lag int) (stable, ok bool) {
	n, _, _ := model.Dims()
	if n+lag > pidTuningExactLagOrder {
		return false, false
	}
	ch, ok := pidTuningIdealCharacteristic(model, c, lag)
	if !ok {
		return false, false
	}
	return pidTuningDyadicStable(ch, c.Dt == 0), true
}

// pidTuningIdealCharacteristic returns pidTuningIdealExact's characteristic,
// leading coefficient first and nonzero, or empty when it vanishes; ok is
// false out of scope.
func pidTuningIdealCharacteristic(model *System, c *PID2, lag int) (ch []dyadic, ok bool) {
	if !pidTuningIdealInScope(model, c) {
		return nil, false
	}
	return pidTuningIdealCharacteristicIn(model, c, lag), true
}

// pidTuningIdealInScope reports whether pidTuningIdealCharacteristic can
// build the characteristic of model and c.
func pidTuningIdealInScope(model *System, c *PID2) bool {
	n, _, _ := model.Dims()
	return n <= pidTuningExactLagOrder && pidTuningIdealRealizable(model, c)
}

// pidTuningIdealRealizable reports whether model and c are a finite SISO
// standard delay-free loop, as pidTuningIdealRealizationFloat asks; E = I
// is standard.
func pidTuningIdealRealizable(model *System, c *PID2) bool {
	_, m, p := model.Dims()
	if newDescriptorPolicy(model).isDescriptor() || model.HasDelay() || m != 1 || p != 1 {
		return false
	}
	for _, a := range []*mat.Dense{model.A, model.B, model.C, model.D} {
		if a != nil && requireFiniteDense("", "", a) != nil {
			return false
		}
	}
	for _, v := range []float64{c.Kp, c.Ki, c.Kd, c.Dt} {
		if !finitePID(v) {
			return false
		}
	}
	return true
}

// pidTuningIdealCharacteristicIn is pidTuningIdealCharacteristic in scope.
func pidTuningIdealCharacteristicIn(model *System, c *PID2, lag int) []dyadic {
	n, _, _ := model.Dims()
	at := func(a *mat.Dense, i, j int) dyadic { return newDyadic(a.At(i, j)) }
	a, abc := make([][]dyadic, n), make([][]dyadic, n)
	for i := range n {
		a[i], abc[i] = make([]dyadic, n), make([]dyadic, n)
		for j := range n {
			a[i][j] = at(model.A, i, j)
			abc[i][j] = a[i][j].sub(at(model.B, i, 0).mul(at(model.C, 0, j)))
		}
	}
	delta := dyadicCharPoly(a)
	num := dyadicPolyAdd(dyadicCharPoly(abc), dyadicPolyScale(delta, at(model.D, 0, 0).sub(newDyadic(1))))

	one, zero, dt := newDyadic(1), newDyadic(0), newDyadic(c.Dt)
	minus := newDyadic(-1)
	half := dyadic{big.NewInt(1), -1}
	var ip, id, dn, dd []dyadic
	if c.Ki == 0 {
		ip, id = []dyadic{zero}, []dyadic{one}
	} else if c.Dt == 0 {
		ip, id = []dyadic{one}, []dyadic{one, zero}
	} else {
		id = []dyadic{one, minus}
		switch c.IFormula {
		case BackwardEuler:
			ip = []dyadic{dt, zero}
		case Trapezoidal:
			ip = []dyadic{dt.mul(half), dt.mul(half)}
		default:
			ip = []dyadic{dt}
		}
	}
	if c.Dt == 0 {
		dn, dd = []dyadic{one, zero}, []dyadic{one}
	} else {
		switch c.DFormula {
		case BackwardEuler:
			dn, dd = []dyadic{one, minus}, []dyadic{dt, zero}
		case Trapezoidal:
			two := newDyadic(2)
			dn, dd = []dyadic{two, two.mul(minus)}, []dyadic{dt, dt}
		default:
			dn, dd = []dyadic{one, minus}, []dyadic{dt}
		}
	}
	den := dyadicPolyMul(id, dd)
	ctrl := dyadicPolyAdd(dyadicPolyAdd(dyadicPolyScale(den, newDyadic(c.Kp)), dyadicPolyScale(dyadicPolyMul(ip, dd), newDyadic(c.Ki))), dyadicPolyScale(dyadicPolyMul(dn, id), newDyadic(c.Kd)))
	shifted := dyadicPolyMul(delta, den)
	for range lag {
		shifted = append(shifted, zero)
	}
	ch := dyadicPolyAdd(shifted, dyadicPolyMul(num, ctrl))
	for len(ch) > 0 && ch[0].m.Sign() == 0 {
		ch = ch[1:]
	}
	return ch
}

// pidTuningDyadicStable reports whether every root of ch, leading
// coefficient first, passes pidTuningExactRootsStable; an empty ch fails.
func pidTuningDyadicStable(ch []dyadic, continuous bool) bool {
	if len(ch) == 0 {
		return false
	}
	emin := ch[0].e
	for _, v := range ch {
		emin = min(emin, v.e)
	}
	ints := make([]*big.Int, len(ch))
	for i, v := range ch {
		ints[i] = new(big.Int).Lsh(v.m, uint(v.e-emin))
	}
	return pidTuningExactRootsStable(ints, big.NewInt(1), big.NewInt(1), continuous)
}

// dyadicPolyFloat returns p divided by its leading coefficient, rounded to
// float64, with ok false when a coefficient overflows. Each coefficient
// keeps its own relative accuracy, as tiny roots of p ask.
func dyadicPolyFloat(p []dyadic) (Poly, bool) {
	if len(p) == 0 {
		return Poly{0}, true
	}
	shift := p[0].m.BitLen() + p[0].e
	out := make(Poly, len(p))
	for i, v := range p {
		f := new(big.Float).SetInt(v.m)
		out[i], _ = f.SetMantExp(f, v.e-shift).Float64()
		if math.IsInf(out[i], 0) {
			return nil, false
		}
	}
	return out, true
}

// dyadicCharPoly returns det(λI−a), leading coefficient first: with
// a = m·2^e for an integer m, its λ^(n−k) coefficient is 2^(k·e) times m's.
func dyadicCharPoly(a [][]dyadic) []dyadic {
	if len(a) == 0 {
		return []dyadic{newDyadic(1)}
	}
	e := a[0][0].e
	for _, row := range a {
		for _, v := range row {
			e = min(e, v.e)
		}
	}
	m := make([][]*big.Int, len(a))
	for i, row := range a {
		m[i] = make([]*big.Int, len(row))
		for j, v := range row {
			m[i][j] = new(big.Int).Lsh(v.m, uint(v.e-e))
		}
	}
	c := intCharPoly(m)
	out := make([]dyadic, len(c))
	for k, v := range c {
		out[k] = dyadic{v, k * e}
	}
	return out
}

// dyadicPolyMul returns a·b, leading coefficients first.
func dyadicPolyMul(a, b []dyadic) []dyadic {
	out := make([]dyadic, len(a)+len(b)-1)
	for i := range out {
		out[i] = newDyadic(0)
	}
	for i, x := range a {
		for j, y := range b {
			out[i+j] = out[i+j].add(x.mul(y))
		}
	}
	return out
}

// dyadicPolyAdd returns a+b, leading coefficients first.
func dyadicPolyAdd(a, b []dyadic) []dyadic {
	if len(a) < len(b) {
		a, b = b, a
	}
	out := slices.Clone(a)
	for i, v := range b {
		k := i + len(a) - len(b)
		out[k] = out[k].add(v)
	}
	return out
}

func dyadicPolyScale(a []dyadic, k dyadic) []dyadic {
	out := make([]dyadic, len(a))
	for i, v := range a {
		out[i] = v.mul(k)
	}
	return out
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
// [p.crossLow, p.crossHigh] under the floor, a bound unset (≤ 0 or +Inf)
// widening nothing, and adds points within 4|σ| of
// each lightly damped mode σ+jω₀, and p.samples, so a crossover pair at a
// resonance is not stepped over. Added points where p.try fails or is not
// finite are dropped, as near rounded multiple poles at z = 1 or s = 0 under
// the band, where the solve cannot resolve the response and the integrators
// keep |L| ≫ 1.
func pidTuningCrossGrid(p pidTuningPlant, omega []float64) []float64 {
	n := len(omega)
	step := math.Log(omega[n-1]/omega[0]) / float64(n-1)
	below, above := 0., 0.
	if p.hasCrossFloor && step > 0 && !math.IsInf(step, 0) {
		if p.crossLow > 0 {
			below = math.Ceil(math.Log(omega[0]/p.crossLow) / step)
		}
		if p.crossHigh < math.Inf(1) {
			above = math.Ceil(math.Log(p.crossHigh/omega[n-1]) / step)
		}
	}
	var grid []float64
	for k := below; k >= 1; k-- {
		grid = append(grid, omega[0]*math.Exp(-k*step))
	}
	grid = append(grid, omega...)
	for k := 1.; k <= above; k++ {
		grid = append(grid, math.Min(omega[n-1]*math.Exp(k*step), p.crossHigh))
	}
	lo, hi := grid[0], grid[len(grid)-1]
	for _, m := range p.modes {
		d := math.Max(math.Abs(real(m)), 1e-6*imag(m))
		for k := -8.; k <= 8; k++ {
			if w := imag(m) + k*d/2; w > lo && w < hi {
				grid = append(grid, w)
			}
		}
	}
	for _, w := range p.samples {
		if w > lo && w < hi {
			grid = append(grid, w)
		}
	}
	slices.Sort(grid)
	grid = slices.Compact(grid)
	if p.try == nil {
		return grid
	}
	return slices.DeleteFunc(grid, func(w float64) bool {
		if _, ok := slices.BinarySearch(omega, w); ok {
			return false
		}
		h, ok := p.try(w)
		return !ok || !pidFiniteComplex(h)
	})
}

// pidTuningZeroGrid adds to grid, sorted, the points where c's zeros can
// put a 0 dB crossing that grid misses: a lightly damped pair σ+jω₀ of
// Kd·s² + Kp·s + Ki, as when tiny Kp and Ki set it below the band, notches
// |L| through 1 within a few |σ| of ω₀. It extends grid at step (in log ω)
// down to a tenth of the slowest zero, of p.zeroLow's plant dynamics and,
// with integral action, of |Ki·G(0)|, where |L| ≈ |Ki·G(0)|/ω crosses 1,
// at most four decades, and adds points within 4|σ| of ω₀ and of each
// p.modes mode in that extension. Discrete controllers use the continuous zeros,
// which match theirs well below Nyquist. An integral-free c whose |L| − 1
// changes sign between ω = 0 and lo walks further down
// (pidTuningDCCrossingFloor). It extends grid up at step too, as far as
// pidTuningZeroGridTop, since a large Kd can cross 0 dB above it, and adds
// points within 4|σ| of each p.modes mode above hi up to that top (without
// p.loopTail, up to pidTuningTailReach·hi): a resonance there lifts |L|
// through 1 in a pair the coarse step skips. Points where p.try fails are
// dropped; cache keeps the samples of the
// extensions, whose points recur.
func pidTuningZeroGrid(p pidTuningPlant, c PID2, grid []float64, step float64, cache map[float64]complex128) []float64 {
	a, b, k := c.Kp*c.Tf+c.Kd, c.Kp+c.Ki*c.Tf, c.Ki
	var zeros []complex128
	if a != 0 {
		q := -.5 * (complex(b, 0) + complex(math.Copysign(1, b), 0)*cmplx.Sqrt(complex(b*b-4*a*k, 0)))
		if q != 0 {
			zeros = append(zeros, q/complex(a, 0), complex(k, 0)/q)
		}
	} else if b != 0 {
		zeros = append(zeros, complex(-k/b, 0))
	}
	lo, hi := grid[0], grid[len(grid)-1]
	slowest := math.Inf(1)
	var extra []float64
	for _, z := range zeros {
		m := cmplx.Abs(z)
		if !finitePID(m) || m == 0 {
			continue
		}
		slowest = math.Min(slowest, m)
		if w0 := math.Abs(imag(z)); w0 > 0 && imag(z) > 0 {
			d := math.Max(math.Abs(real(z)), 1e-6*w0)
			for j := -8.; j <= 8; j++ {
				extra = append(extra, w0+j*d/2)
			}
		}
	}
	floor := math.Max(slowest/10, lo*1e-4)
	if p.zeroLow > 0 {
		floor = math.Min(floor, math.Max(p.zeroLow, lo*1e-4))
	}
	if g := p.dcEdge; c.Ki != 0 && g != 0 && pidFiniteComplex(g) {
		if wa := cmplx.Abs(complex(c.Ki, 0) * g); wa > 0 && finitePID(wa) {
			floor = math.Min(floor, math.Max(wa/10, lo*1e-4))
		}
	}
	if w, ok := pidTuningDCCrossingFloor(p, c, lo, step, cache); ok {
		floor = math.Min(floor, w)
	}
	for j := 1.; j <= math.Ceil(math.Log(lo/floor)/step); j++ {
		extra = append(extra, lo*math.Exp(-j*step))
	}
	bottom := math.Min(lo, floor*math.Exp(-step))
	sample := func(w float64) (complex128, bool) {
		if h, ok := cache[w]; ok {
			return h, pidFiniteComplex(h)
		}
		h, ok := p.at(w), true
		if p.try != nil {
			h, ok = p.try(w)
		}
		if !ok {
			h = cmplx.NaN()
		}
		cache[w] = h
		return h, pidFiniteComplex(h)
	}
	top := pidTuningZeroGridTop(p, c, hi, step, sample)
	var modal []float64
	for _, m := range p.modes {
		d := math.Max(math.Abs(real(m)), 1e-6*imag(m))
		for j := -8.; j <= 8; j++ {
			if w := imag(m) + j*d/2; w < lo && w >= bottom {
				modal = append(modal, w)
			} else if w > hi {
				modal = append(modal, w)
				if p.loopTail == nil && p.dt == 0 && w <= hi*pidTuningTailReach {
					top = math.Max(top, w)
				}
			}
		}
	}
	for j := 1.; j <= math.Ceil(math.Log(top/hi)/step); j++ {
		extra = append(extra, math.Min(hi*math.Exp(j*step), top))
	}
	extra = append(extra, modal...)
	extra = slices.DeleteFunc(extra, func(w float64) bool {
		if w < bottom || w > top || w <= 0 {
			return true
		}
		if _, ok := slices.BinarySearch(grid, w); ok {
			return true
		}
		h, ok := cache[w]
		if !ok {
			if p.try != nil {
				h, ok = p.try(w)
			} else {
				h, ok = p.at(w), true
			}
			if !ok {
				h = cmplx.NaN()
			}
			if w < lo || w > hi {
				cache[w] = h
			}
		}
		return !pidFiniteComplex(h)
	})
	if len(extra) == 0 {
		return grid
	}
	out := slices.Concat(grid, extra)
	slices.Sort(out)
	return slices.Compact(out)
}

// pidTuningZeroGridTop returns the top of c's zero grid, hi or above it:
// discrete, pidTuningSensNyquist·π/Dt; continuous, the first point of
// hi·e^{j·step} where p.loopTail bounds |L| below 1 beyond, or, without
// it, where |L| − 1 turns negative, at most pidTuningTailReach·hi. Where
// p.loopTail has no bound there yet, as its Markov bound holds only above
// ‖A‖, far above the band of a slow crossover, the grid walks on to the
// first point it bounds, at most pidTuningTailReach²·hi. Past the band a
// large Kd keeps |L| over 1 or lifts it back to 1, so 0 dB crossings can
// lie above the band's top. sample gives G.
func pidTuningZeroGridTop(p pidTuningPlant, c PID2, hi, step float64, sample func(float64) (complex128, bool)) float64 {
	if !(step > 0) || !finitePID(hi) || hi <= 0 {
		return hi
	}
	if p.dt > 0 {
		return math.Max(hi, pidTuningSensNyquist*math.Pi/p.dt)
	}
	reach := hi * pidTuningTailReach
	for j := 0.; ; j++ {
		w := math.Min(hi*math.Exp(j*step), reach)
		if p.loopTail != nil {
			if p.loopTail(&c, w) < 1 {
				return w
			}
		} else if h, ok := sample(w); !ok || cmplx.Abs(h*pidTuningFeedback(c, w)) < 1 {
			return w
		}
		if w >= reach {
			break
		}
	}
	if p.loopTail == nil {
		return reach
	}
	last := reach * pidTuningTailReach
	for w := reach; ; w = math.Min(w*math.Exp(step), last) {
		if !math.IsInf(p.loopTail(&c, w), 1) || w >= last {
			return w
		}
	}
}

// pidTuningDCCrossingFloor returns, for an integral-free c on a plant whose
// DC gain p.dcEdge resolves, the first point of the grid lo·e^{−j·step} at
// which |L| − 1 takes the sign of |Kp·G(0)| − 1, when |L(lo)| − 1 has the
// other: an odd count of 0 dB crossings then lies below lo. It walks at most
// six decades and stops at a sample p.try cannot resolve.
func pidTuningDCCrossingFloor(p pidTuningPlant, c PID2, lo, step float64, cache map[float64]complex128) (float64, bool) {
	g := p.dcEdge
	if c.Ki != 0 || g == 0 || !pidFiniteComplex(g) || !(step > 0) {
		return 0, false
	}
	dc := cmplx.Abs(complex(c.Kp, 0)*g) - 1
	if dc == 0 || !finitePID(dc) {
		return 0, false
	}
	excess := func(w float64, h complex128) float64 { return cmplx.Abs(h*pidTuningFeedback(c, w)) - 1 }
	if v := excess(lo, p.at(lo)); dc*v >= 0 || !finitePID(v) {
		return 0, false
	}
	for j := 1.; j <= math.Ceil(math.Log(1e6)/step); j++ {
		w := lo * math.Exp(-j*step)
		h, ok := cache[w]
		if !ok {
			if p.try != nil {
				h, ok = p.try(w)
			} else {
				h, ok = p.at(w), true
			}
			if !ok {
				h = cmplx.NaN()
			}
			cache[w] = h
		}
		v := excess(w, h)
		if !finitePID(v) {
			return 0, false
		}
		if dc*v > 0 {
			return w, true
		}
	}
	return 0, false
}

// pidTuningIntegralZero returns a real zero z of a stable, strictly proper,
// delay-free continuous plant with −poleStabilityTolerance(z) <= z <= 0, when
// family has an integral term. Under C = Nc/(s·Dc), Dc(s) = 1 or 1+Tf·s, the
// characteristic s·Dc·Dp + Nc·Np, Dp monic, has a positive leading
// coefficient (for ideal PID 1+Kd·CB, which the screen requires positive)
// and equals Ki·Np(0) at 0 and z·Dc(z)·Dp(z) ≤ 0 at z, so it has a real root
// in [z, 0] when Ki takes the DC gain's sign and in (0, ∞) otherwise: no such
// loop passes IsStable. A zero just right of 0 gives no such sign change: an
// I loop on (1e-10−s)/(s+1)² with Ki = 0.9 has its slow pole near −9e-10.
func pidTuningIntegralZero(plant *System, family PidtuneType) (float64, bool) {
	if _, hasI, _, _, ok := pidTuningTerms(family); !ok || !hasI {
		return 0, false
	}
	if plant.Dt != 0 || plant.HasDelay() || plant.IsDescriptor() || plant.D.At(0, 0) != 0 {
		return 0, false
	}
	if stable, err := plant.IsStable(); err != nil || !stable {
		return 0, false
	}
	zeros, err := plant.Zeros()
	if err != nil {
		return 0, false
	}
	for _, z := range zeros {
		if imag(z) == 0 && real(z) <= 0 && -real(z) <= poleStabilityTolerance(z) {
			return real(z), true
		}
	}
	return 0, false
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

// pidTuningSensitivityPeak returns max |1/(1+L(jω))| over [min(p.low,
// p.sensLow), top], top = max(p.high, p.crossHigh, p.sensHigh): the reported
// band widened, by TunePID, to a tenth of the slowest plant root and,
// discrete, π/Dt, since a peak there shows neither in the band nor at its
// edges; extended, by at
// most pidTuningTailPoints samples up to pidTuningTailReach·top, while
// p.loopTail admits |L| > 1 − 1/p.maxSensitivity beyond, since a large derivative gain can cross 0 dB again far above the
// crossover.
// The grid step turns the delay phase by at most π/16 and, near a plant mode
// σ+jω₀, is at most max(|σ|, |ω−ω₀|)/4, so a lightly damped mode's narrow
// peak is bracketed. Grid maxima within 0.8 of the running peak are then
// refined by golden section on log ω, since a delay ripple can sit between
// samples and read ~10% low. |S| at p.dcEdge and p.nyquistEdge counts too.
func pidTuningSensitivityPeak(p pidTuningPlant, c PID2) float64 {
	band := math.Max(p.high, p.crossHigh)
	top, start := math.Max(band, p.sensHigh), p.low
	if p.sensLow > 0 {
		start = math.Min(start, p.sensLow)
	}
	outside := func(w float64) bool { return p.try != nil && (w < p.low || w > band && w <= p.sensHigh) }
	sample := func(w float64, cached bool) float64 {
		g, ok, in := complex128(0), true, !outside(w)
		switch {
		case in:
			g = p.at(w)
		case cached && p.sensCache != nil:
			if g, ok = p.sensCache[w]; !ok {
				g, ok = p.try(w)
				if !ok || !pidFiniteComplex(g) {
					g = cmplx.NaN()
				}
				p.sensCache[w] = g
			}
			ok = pidFiniteComplex(g)
		default:
			g, ok = p.try(w)
		}
		v := 1 / cmplx.Abs(1+g*pidTuningFeedback(c, w))
		if !in && (!ok || math.IsNaN(v)) {
			return 0
		}
		return v
	}
	s := func(w float64) float64 { return sample(w, false) }
	peak := pidTuningEdgeSensitivity(p, c)
	high, extend, inBand := top, p.loopTail != nil && p.maxSensitivity > 1, 0
	var w0, v0, w1, v1 float64
	for k, w := 0, start; ; k++ {
		if w <= top {
			inBand = k
		}
		v := sample(w, true)
		if k >= 2 && v1 >= v0 && v1 >= v && v1 >= .8*math.Max(peak, v) {
			_, g := pidTuningGoldenMax(s, w0, w)
			peak = math.Max(peak, g)
		}
		peak = math.Max(peak, v)
		w0, v0, w1, v1 = w1, v1, w, v
		if k-inBand >= pidTuningTailPoints {
			return peak
		}
		if w >= high {
			if !extend || w >= top*pidTuningTailReach || p.loopTail(&c, w) <= 1-1/p.maxSensitivity {
				return peak
			}
			high = math.Min(2*w, top*pidTuningTailReach)
		}
		step := .01 * w
		if p.tau > 0 {
			step = math.Min(step, math.Pi/16/p.tau)
		}
		for _, m := range p.modes {
			step = math.Min(step, math.Max(math.Max(math.Abs(real(m)), math.Abs(w-imag(m))), 1e-6*w)/4)
		}
		w = math.Min(w+step, high)
	}
}

// pidTuningEdgeSensitivity returns the larger |S| at p.dcEdge and
// p.nyquistEdge, 0 where unset. Without integral action L(0) = Kp·G(0),
// since every derivative term vanishes there; with it |L(0)| is infinite.
// At z = −1 each integrator dt·(1/(z−1) + w), w its formula's weight, is
// dt·(w − ½), so the derivative term is Kd/(Tf + dt·(w − ½)), infinite
// when that is 0. An ideal derivative on continuous data showing
// lim sG = p.markov ends at L(∞) = Kd·p.markov.
func pidTuningEdgeSensitivity(p pidTuningPlant, c PID2) float64 {
	peak := 0.
	s := func(l complex128) {
		if v := 1 / cmplx.Abs(1+l); !math.IsNaN(v) {
			peak = math.Max(peak, v)
		}
	}
	if g := p.dcEdge; g != 0 && pidFiniteComplex(g) && c.Ki == 0 {
		s(g * complex(c.Kp, 0))
	}
	if p.markov != 0 && c.Dt == 0 && c.Tf == 0 && c.Kd != 0 {
		s(complex(c.Kd*p.markov, 0))
	}
	if g := p.nyquistEdge; g != 0 && pidFiniteComplex(g) && c.Dt > 0 {
		k := c.Kp + c.Ki*c.Dt*(pidFormulaWeight(c.IFormula)-.5)
		if c.Kd != 0 {
			d := c.Tf + c.Dt*(pidFormulaWeight(c.DFormula)-.5)
			if d == 0 {
				return peak
			}
			k += c.Kd / d
		}
		s(g * complex(k, 0))
	}
	return peak
}

// pidTuningTailReach and pidTuningTailPoints cap the |S| scan's extension
// past its band.
const (
	pidTuningTailReach  = 1e4
	pidTuningTailPoints = 4096
)

// pidTuningSensNyquist is the discrete |S| scan's top as a fraction of π/Dt,
// short of a pole at z = −1 that p.nyquistEdge resolves.
const pidTuningSensNyquist = .9999

// pidTuningLoopTail returns a bound on |C(jω')·P(jω')| over ω' ≥ ω for a
// continuous plant without internal delays, or nil; see pidMarkovTail.
func pidTuningLoopTail(plant *System) func(*PID2, float64) float64 {
	if !plant.IsContinuous() || plant.HasInternalDelay() {
		return nil
	}
	markov, d := newPIDMarkovTail(plant), math.Abs(plant.D.At(0, 0))
	return func(c *PID2, w float64) float64 {
		g := d + markov.at(w, 0)
		k := math.Abs(c.Kp) + math.Abs(c.Ki)/w
		switch {
		case c.Kd == 0:
			return k * g
		case c.Tf > 0:
			return (k + math.Abs(c.Kd)/c.Tf) * g
		case d != 0:
			return math.Inf(1)
		}
		return k*g + math.Abs(c.Kd)*markov.at(w, 1)
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
// A grid extremum of |L| within a factor 2 of 0 dB is refined by golden
// section, so a crossover pair between samples, as at a lightly damped
// resonance, is found.
func pidTuningCrossings(p pidTuningPlant, c PID2, omega []float64, wc float64) (crossings, margins []float64) {
	return pidTuningCrossingsOn(p, c, omega, wc, nil)
}

// pidTuningCrossingsOn is pidTuningCrossings with grid(k), when set, giving
// the floor scan's |L(omega[k])|−1.
func pidTuningCrossingsOn(p pidTuningPlant, c PID2, omega []float64, wc float64, grid func(int) float64) (crossings, margins []float64) {
	crossings = []float64{wc}
	excess := func(w float64) float64 { return cmplx.Abs(p.at(w)*pidTuningFeedback(c, w)) - 1 }
	if grid == nil {
		grid = func(k int) float64 { return excess(omega[k]) }
	}
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
		if !slices.ContainsFunc(crossings, func(x float64) bool { return math.Abs(math.Log(w/x)) <= 1e-6 }) {
			crossings = append(crossings, w)
		}
	}
	// |L(wc)| = 1 hides the sign of the grid sample at wc, so a crossover
	// paired with wc inside an adjacent interval is sought by golden section
	// for a sign opposite to the far endpoint's.
	if i, ok := slices.BinarySearch(omega, wc); ok {
		for _, f := range []int{i - 1, i + 1} {
			if f < 0 || f >= len(omega) {
				continue
			}
			ef := excess(omega[f])
			sign := math.Copysign(1, ef)
			lo, hi := min(wc, omega[f]), max(wc, omega[f])
			w, v := pidTuningGoldenMax(func(w float64) float64 { return -sign * excess(w) }, lo, hi)
			if v > 0 {
				add(omega[f], w, ef)
			}
		}
	}
	var before float64
	previous := grid(0)
	for k := 1; k < len(omega); k++ {
		next := grid(k)
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

// pidTuningCoarseCrossings scans omega for sign changes of |L| − 1 only,
// bisecting each.
func pidTuningCoarseCrossings(p pidTuningPlant, c PID2, omega []float64, wc float64) (crossings, margins []float64) {
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

// pidTuningCrossingMargin returns the margin a crossing at w, whose wrapped
// margin m lies in (−180°, 180°] as allmargin reports it, keeps against the
// floor. A margin at or below −floor puts ∠L in (0°, 180° − floor], at
// least floor from −1 the other way round the circle, as where a resonance
// lifts |L| over 1 below wc while lead holds the phase. For a loop a
// Nyquist or pole check certifies stable, that distance, −m, is the margin
// such a crossing erodes; certified asks for it, and the caller must
// confirm stability. Without a certificate m stands: the phase at one
// crossing, unwrapped or not, cannot count encirclements of −1.
func pidTuningCrossingMargin(wc, w, m, floor float64, certified bool) float64 {
	if certified && m <= -floor && w != wc {
		return -m
	}
	return m
}

func pidTuningIdealDerivative(family PidtuneType) bool {
	f := PidtuneType(strings.ToUpper(string(family)))
	return f == PidtunePD || f == PidtunePID
}

// pidMarkovTail bounds |s^p·C(sI−A)⁻¹B|, p ∈ {0, 1}, for Re s ≥ 0, |s| ≥ w of
// a SISO plant, delays and D excluded. Expanding (sI−A)⁻¹ to order k gives
// Σ_{j<k} |CAʲB|·w^{p−j−1} + ‖CAᵏ‖‖B‖·w^{p−k}/(w−‖A‖); the minimum over
// k = 0..n decays like the plant's relative degree, so the delay grid ends
// near crossover instead of where a first-order bound falls below 1. Both
// the given and the balanced realization give a bound and the smaller is
// used: a companion form's ‖A‖ can exceed its spectral radius by orders of
// magnitude, which would stretch the linear delay grid to ‖A‖.
type pidMarkovTail struct {
	reals []pidMarkovRealization
}

type pidMarkovRealization struct {
	normA  float64
	markov []float64 // |CAʲB|
	rem    []float64 // ‖CAᵏ‖‖B‖
}

func newPIDMarkovTail(plant *System) *pidMarkovTail {
	n, _, _ := plant.Dims()
	if n == 0 {
		return &pidMarkovTail{}
	}
	br := newBalancedRealization(plant, n, 1, 1)
	return &pidMarkovTail{reals: []pidMarkovRealization{
		newPIDMarkovRealization(plant.A, plant.B, plant.C),
		newPIDMarkovRealization(mat.NewDense(n, n, br.a), mat.NewDense(n, 1, br.b), mat.NewDense(1, n, br.c)),
	}}
}

func newPIDMarkovRealization(a, b, c *mat.Dense) pidMarkovRealization {
	n, _ := a.Dims()
	r := pidMarkovRealization{markov: make([]float64, n), rem: make([]float64, n+1), normA: spectralNorm(a)}
	normB := spectralNorm(b)
	ca := mat.DenseCopyOf(c)
	var next, v mat.Dense
	for k := range n + 1 {
		r.rem[k] = spectralNorm(ca) * normB
		if k == n {
			break
		}
		v.Mul(ca, b)
		r.markov[k] = math.Abs(v.At(0, 0))
		next.Mul(ca, a)
		ca.Copy(&next)
	}
	return r
}

func (t *pidMarkovTail) at(w float64, p int) float64 {
	if len(t.reals) == 0 {
		return 0
	}
	best := math.Inf(1)
	for i := range t.reals {
		best = min(best, t.reals[i].at(w, p))
	}
	return best
}

func (t *pidMarkovRealization) at(w float64, p int) float64 {
	if math.IsInf(w, 1) {
		if p == 1 {
			return t.markov[0]
		}
		return 0
	}
	best, sum, f := math.Inf(1), 0.0, math.Pow(w, float64(p-1))
	for k := range t.rem {
		switch {
		case t.rem[k] == 0:
			best = min(best, sum)
		case w > t.normA:
			best = min(best, sum+t.rem[k]*f*w/(w-t.normA))
		}
		if k < len(t.markov) {
			sum += t.markov[k] * f
			f /= w
		}
	}
	return best
}

// pidDerivTail bounds |s·G(s)| for Re s ≥ 0, |s| ≥ w of a strictly proper
// SISO plant G, delays excluded.
func pidDerivTail(plant *System) func(float64) float64 {
	t := newPIDMarkovTail(plant)
	return func(w float64) float64 { return t.at(w, 1) }
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
// whose high-frequency gain exceeds maxTail (≤ 1), is rejected; other
// failures are errors.
func pidTuningDelayStability(plant *System, eval *sisoEval, wc float64, idealDerivative bool, maxTail float64) (func(*PID2) (bool, error), bool, error) {
	return pidTuningDelayStabilityBudget(plant, eval, wc, idealDerivative, maxTail, 0)
}

// pidTuningLeastZero returns the least invariant zero magnitude of a
// continuous plant, delays set to zero; +Inf when there is none or on
// failure. An integrator cancels a zero at s = 0: the characteristic
// s·den(s) + (Kp·s+Ki)·num(s)·e^{−τs} keeps the root s = 0 for any Ki, which
// L, and so the Nyquist test, hides: r→u is not asymptotically stable. A
// zero within 1e-9·wc of the origin is taken as there; one further out
// leaves a visible closed-loop root.
func pidTuningLeastZero(plant *System) float64 {
	zeros, err := plant.Zeros()
	least := math.Inf(1)
	if err != nil {
		return least
	}
	for _, m := range zeros {
		least = math.Min(least, cmplx.Abs(m))
	}
	return least
}

// pidTuningDelayStabilityBudget is pidTuningDelayStability with the Nyquist
// grid capped at maxPoints; 0 selects delayNyquistMaxPoints.
func pidTuningDelayStabilityBudget(plant *System, eval *sisoEval, wc float64, idealDerivative bool, maxTail float64, maxPoints int) (func(*PID2) (bool, error), bool, error) {
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
	base.maxPoints = maxPoints
	plantTail, plantLimit := base.tail, base.tailLimit
	var derivTail func(float64) float64
	if !plant.HasInternalDelay() {
		markov, d := newPIDMarkovTail(plant), math.Abs(plant.D.At(0, 0))
		plantTail = func(w float64) float64 { return min(base.tail(w), d+markov.at(w, 0)) }
		if d == 0 {
			derivTail = func(w float64) float64 { return markov.at(w, 1) }
		}
	}
	if idealDerivative && plant.HasInternalDelay() {
		return nil, false, nil
	}
	leastZero := sync.OnceValue(func() float64 { return pidTuningLeastZero(plant) })
	origin := sync.OnceValue(func() pidTuningOrigin { return newPIDTuningOrigin(plant, wc) })
	return func(c *PID2) (bool, error) {
		if c.Ki != 0 && leastZero() <= 1e-9*wc {
			return false, nil
		}
		l := *base
		l.atS, l.clusters = nil, nil
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
			if l.tailLimit > maxTail {
				return false, nil
			}
		default:
			return false, nil
		}
		if l.tailLimit < 1 && origin().unstable(c) {
			return false, nil
		}
		cert, err := pidLoopCertifier(plant, &cand)
		if errors.Is(err, errDelayLoopUnsupported) {
			return false, nil
		}
		if err != nil {
			return false, err
		}
		l.cert = cert
		stable, err := l.stableClosedLoop()
		if errors.Is(err, errDelayLoopUnsupported) {
			return false, nil
		}
		return stable, err
	}, true, nil
}

// pidTuningOrigin holds N(0) and D(0) of a continuous plant G = N/D without
// internal delays, signed by D's leading coefficient, with their rounding
// scales Σ|cₖ|·wcᵏ.
type pidTuningOrigin struct {
	n0, d0, nScale, dScale float64
	ok                     bool
}

func newPIDTuningOrigin(plant *System, wc float64) pidTuningOrigin {
	if plant.E != nil || plant.HasInternalDelay() || !plant.IsContinuous() {
		return pidTuningOrigin{}
	}
	res, err := plant.rationalTransferFunction(nil)
	if err != nil {
		return pidTuningOrigin{}
	}
	num, den := res.TF.Num[0][0], res.TF.Den[0]
	if len(num) == 0 || len(den) == 0 || den[0] == 0 {
		return pidTuningOrigin{}
	}
	scale := func(q []float64) float64 {
		v := 0.0
		for i, c := range q {
			v += math.Abs(c) * math.Pow(wc, float64(len(q)-1-i))
		}
		return v
	}
	a := math.Copysign(1, den[0])
	o := pidTuningOrigin{n0: a * num[len(num)-1], d0: a * den[len(den)-1], nScale: scale(num), dScale: scale(den)}
	o.ok = finitePID(o.n0) && finitePID(o.d0) && finitePID(o.nScale) && finitePID(o.dScale)
	return o
}

// unstable reports a real closed-loop root in [0, ∞) by the sign of the
// characteristic P(s) = s·D·(1+Tf·s) + ((Kp·s+Ki)(1+Tf·s) + Kd·s)·N·e^{−sτ}
// (without the s and Kp·s factors when Ki = 0), real on the real axis.
// When limsup |L| < 1 there, P/a > 0 as s → +∞, so P(0)/a = Ki·N(0)/a, or
// (D(0) + Kp·N(0))/a, below its rounding is certainly unstable. The Nyquist
// grid can miss such a root far below wc: an indentation merging the
// integrator with a pair at 2.2e-7 enclosed the root 5e-8 that a zero at
// +5e-8 leaves under a PI crossing at 0.83.
func (o pidTuningOrigin) unstable(c *PID2) bool {
	if !o.ok {
		return false
	}
	p0, scale := c.Ki*o.n0, math.Abs(c.Ki)*o.nScale
	if c.Ki == 0 {
		p0, scale = o.d0+c.Kp*o.n0, o.dScale+math.Abs(c.Kp)*o.nScale
	}
	return finitePID(p0) && -p0 > 1e3*0x1p-52*scale
}
