package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

type c2dPlan struct {
	sys             *System
	dt              float64
	opts            C2DOptions
	method          C2DMethod
	delayModeling   C2DDelayModeling
	contInputDelay  []float64
	contOutputDelay []float64
	pathDelay       *mat.Dense
	workSys         *System
	lsSlow          *System
	matchedRef      *System
	// gate judges the result of a public ZOH, FOH or Tustin C2D of a
	// descriptor (holdGate); nil otherwise.
	gate *holdGate
	// rec, for C2DMap, records the model discretizeMethod discretized.
	rec *c2dCoords
	// foldScale and reduced are conversionStandardFormScaled's of sys.
	foldScale int
	reduced   bool
}

func newC2DPlan(sys *System, dt float64, opts C2DOptions) (c2dPlan, error) {
	if sys.IsDiscrete() {
		return c2dPlan{}, fmt.Errorf("system already discrete: %w", ErrWrongDomain)
	}
	var err error
	var lsSlow *System
	ref := sys
	if opts.Method == C2DMethodLeastSquares {
		if lsSlow, err = leastSquaresPencilSplit(sys); err != nil {
			return c2dPlan{}, err
		}
	}
	matchedStiff, foldScale, reduced := false, 0, false
	if opts.Method == C2DMethodMatched {
		if matchedStiff, err = stiffDelayedDescriptor(sys); err != nil {
			return c2dPlan{}, err
		}
	}
	if !matchedStiff && lsSlow == nil {
		if sys, reduced, foldScale, err = conversionStandardFormScaled(sys, "descriptor reduction"); err != nil {
			return c2dPlan{}, err
		}
	}
	opts, err = normalizeC2DOptions(dt, opts)
	if err != nil {
		return c2dPlan{}, err
	}
	method := opts.Method
	delayModeling := opts.DelayModeling

	cp := sys.Copy()
	plan := c2dPlan{
		sys:             sys,
		dt:              dt,
		opts:            opts,
		method:          method,
		delayModeling:   delayModeling,
		contInputDelay:  sys.InputDelay,
		contOutputDelay: sys.OutputDelay,
		workSys:         cp,
		lsSlow:          lsSlow,
		matchedRef:      ref,
		foldScale:       foldScale,
		reduced:         reduced,
	}
	plan.workSys.InputDelay = nil
	plan.workSys.OutputDelay = nil
	if sys.Delay != nil && (opts.ThiranOrder > 0 || ((method == C2DMethodZOH || method == C2DMethodFOH) && conversionHasFractionalPathDelay(sys.Delay, dt))) {
		decomp := decomposedDelayMatrix(sys.Delay)
		if decomp.hasResidual() {
			plan.workSys.Delay = decomp.residual
		} else {
			plan.workSys.Delay = nil
		}
		plan.contInputDelay = mergeDelays(sys.InputDelay, decomp.inputDelay)
		plan.contOutputDelay = mergeDelays(sys.OutputDelay, decomp.outputDelay)
	}

	if method == C2DMethodTustin || method == C2DMethodMatched {
		if opts.ThiranOrder == 0 {
			plan.workSys.Delay, err = roundConversionPathDelays(plan.workSys.Delay, dt)
			if err != nil {
				return c2dPlan{}, err
			}
		} else if conversionHasFractionalPathDelay(plan.workSys.Delay, dt) {
			plan.pathDelay, plan.workSys.Delay = plan.workSys.Delay, nil
		}
	}

	return plan, nil
}

func (p c2dPlan) run() (*System, error) {
	if !p.sys.HasInternalDelay() && (p.method == C2DMethodZOH || p.method == C2DMethodFOH || p.method == C2DMethodImpulse) {
		if conversionHasFractionalPathDelay(p.workSys.Delay, p.dt) || ((p.method == C2DMethodFOH || p.method == C2DMethodImpulse) && conversionHasFractionalExternalDelay(p.sys, p.dt)) {
			out, err := discretizeDelayedChannels(p.sys, p.dt, p.opts)
			if err != nil || p.gate == nil {
				return out, err
			}
			return out, p.gate.check(out)
		}
	}
	if p.workSys.HasInternalDelay() {
		p.rec.setCont(p.sys, nil, p.foldScale, p.reduced)
		return p.discretizeInternalDelay()
	}
	disc, src, z, err := p.discretizeMethod()
	if err != nil && !carriesModel(err) {
		return nil, err
	}
	p.rec.setCont(src, z, p.foldScale, p.reduced)
	out, derr := p.applyExternalDelaysFrom(src, disc)
	if derr != nil {
		return nil, derr
	}
	if p.gate != nil && p.sys.HasDelay() && (err == nil || carriesModel(err)) {
		err = p.gate.check(out)
	}
	return out, err
}

func (p c2dPlan) discretizeInternalDelay() (*System, error) {
	if (p.method == C2DMethodZOH || p.method == C2DMethodFOH) && conversionHoldFeedbackNeedsAbsorption(p.workSys, p.contInputDelay, p.contOutputDelay, p.dt) {
		return discretizeHoldFeedback(p.workSys, p.contInputDelay, p.contOutputDelay, p.dt, p.opts)
	}
	disc, err := discretizeWithInternalDelay(p.workSys, p.dt, p.opts)
	if err != nil {
		return nil, err
	}
	return p.applyExternalDelays(disc)
}

// discretizeMethod discretizes workSys by the method, returning the model
// it discretized (the delays are applied in its coordinates); with a gate, a
// ZOH or FOH result of a dense workSys that fails it is discretized again
// from the real Schur form of workSys, in those coordinates and without
// state names, and the first result that passes is returned, else the
// first with the gate's error. A fold of a dense, non-normal E⁻¹A with
// ‖A·dt‖ ≫ 1 holds DC poorly in its own coordinates, where even its
// rounded exact hold is far off; orthogonal ones keep it. z is the Schur
// vectors of that retry (src's state is z times the result's), else nil.
func (p c2dPlan) discretizeMethod() (*System, *System, *mat.Dense, error) {
	disc, err := p.discretizeWith(p.workSys)
	if p.gate == nil || err != nil {
		return disc, p.workSys, nil, err
	}
	gerr := p.gate.check(disc)
	if gerr == nil || p.gate.refused != nil || (p.method != C2DMethodZOH && p.method != C2DMethodFOH) || upperQuasiTriangular(p.workSys.A) {
		return disc, p.workSys, nil, gerr
	}
	schur, z, ok := realSchurPartZ(p.workSys)
	if !ok {
		return disc, p.workSys, nil, gerr
	}
	if sd, err := p.discretizeWith(schur); err == nil && p.gate.check(sd) == nil {
		return sd, schur, z, nil
	}
	return disc, p.workSys, nil, gerr
}

func (p c2dPlan) discretizeWith(sys *System) (*System, error) {
	switch p.method {
	case C2DMethodZOH:
		return sys.discretizeZOH(p.dt)
	case C2DMethodTustin:
		return sys.discretizeTustin(p.dt, p.opts.PrewarpFrequency)
	case C2DMethodFOH:
		return sys.discretizeModifiedFOH(p.dt)
	case C2DMethodLeastSquares:
		return sys.discretizeLeastSquares(p.dt, p.opts.FitOrder, p.lsSlow)
	case C2DMethodImpulse:
		return sys.discretizeImpulseParity(p.dt)
	case C2DMethodMatched:
		if sys.IsDescriptor() {
			return discretizeMatchedDescriptor(sys, p.dt, p.matchedRef)
		}
		return sys.discretizeMatched(p.dt, p.matchedRef)
	default:
		panic("unvalidated C2D method")
	}
}

// stiffDelayedDescriptor reports that the descriptor sys with I/O delays
// and no internal ones has stiffDescriptorParts without the delays, so
// matched C2D maps the transfer function of the pencil as without delays
// instead of that of the LU fold E⁻¹A.
// A split that is not certified returns its error.
func stiffDelayedDescriptor(sys *System) (bool, error) {
	if !sys.IsDescriptor() || sys.LFT != nil || !sys.HasDelay() {
		return false, nil
	}
	core := *sys
	core.Delay, core.InputDelay, core.OutputDelay = nil, nil, nil
	sp, ok := stiffDescriptorParts(&core)
	if ok {
		if err := sp.within(1); err != nil {
			return false, err
		}
	}
	return ok, nil
}

// discretizeMatchedDescriptor is discretizeMatched of the delay-free part
// of the stiffDelayedDescriptor sys, with the I/O delay of sys in samples;
// ref is the caller's model.
func discretizeMatchedDescriptor(sys *System, dt float64, ref *System) (*System, error) {
	core := *sys
	core.Delay = nil
	out, err := core.discretizeMatched(dt, ref)
	if err != nil && !carriesModel(err) {
		return nil, err
	}
	delayed, derr := newDelayConversionPolicy(dt).applyDiscreteDelayFields(out, sys)
	if derr != nil {
		return nil, derr
	}
	return delayed, err
}

func (p c2dPlan) applyExternalDelays(disc *System) (*System, error) {
	return p.applyExternalDelaysFrom(p.workSys, disc)
}

// applyExternalDelaysFrom is applyExternalDelays for disc discretized from
// cont, whose coordinates a fractional ZOH input delay is held in.
func (p c2dPlan) applyExternalDelaysFrom(cont, disc *System) (*System, error) {
	if p.pathDelay != nil {
		var err error
		if disc, err = conversionPathThiran(disc, p.pathDelay, p.dt, p.opts); err != nil {
			return nil, err
		}
	}
	return applyConversionExternalDelays(cont, disc, p.contInputDelay, p.contOutputDelay, p.dt, p.opts)
}

type d2cPlan struct {
	sys    *System
	ref    *System
	method C2DMethod
	opts   D2COptions
}

func newD2CPlan(sys *System, opts D2COptions) (d2cPlan, error) {
	if sys.IsContinuous() {
		return d2cPlan{}, fmt.Errorf("system already continuous: %w", ErrWrongDomain)
	}
	if err := validateConversionSampleTime(sys.Dt); err != nil {
		return d2cPlan{}, err
	}
	if opts.Method == "" {
		opts.Method = C2DMethodZOH
	}
	switch opts.Method {
	case C2DMethodZOH, C2DMethodTustin, C2DMethodFOH, C2DMethodMatched:
	default:
		return d2cPlan{}, fmt.Errorf("unsupported method %q: %w", opts.Method, ErrInvalidConversionOptions)
	}
	if err := validatePrewarp(sys.Dt, opts.Method, opts.PrewarpFrequency); err != nil {
		return d2cPlan{}, err
	}
	ref := sys
	sys, _, err := conversionStandardForm(sys, "descriptor reduction")
	if err != nil {
		return d2cPlan{}, err
	}
	return d2cPlan{sys: sys, ref: ref, method: opts.Method, opts: opts}, nil
}

func (p d2cPlan) run() (*System, error) {
	switch p.method {
	case C2DMethodZOH:
		return p.sys.d2cZOH()
	case C2DMethodTustin:
		return p.sys.undiscretizeTustin(p.opts.PrewarpFrequency)
	case C2DMethodFOH:
		return p.sys.d2cFOH()
	case C2DMethodMatched:
		return p.sys.d2cMatched(p.ref)
	default:
		panic("unvalidated D2C method")
	}
}

type d2dPlan struct {
	sys   *System
	newDt float64
	opts  C2DOptions
}

func newD2DPlan(sys *System, newDt float64, opts C2DOptions) (d2dPlan, error) {
	if sys.IsContinuous() {
		return d2dPlan{}, fmt.Errorf("system is continuous: %w", ErrWrongDomain)
	}
	if err := validateConversionSampleTime(sys.Dt); err != nil {
		return d2dPlan{}, err
	}
	opts, err := normalizeC2DOptions(newDt, opts)
	if err != nil {
		return d2dPlan{}, err
	}
	switch opts.Method {
	case C2DMethodZOH, C2DMethodTustin:
	default:
		return d2dPlan{}, fmt.Errorf("unsupported method %q: %w", opts.Method, ErrInvalidConversionOptions)
	}
	if err := validatePrewarp(sys.Dt, opts.Method, opts.PrewarpFrequency); err != nil {
		return d2dPlan{}, err
	}
	return d2dPlan{sys: sys, newDt: newDt, opts: opts}, nil
}

func (p d2dPlan) run() (*System, error) {
	if math.Abs(p.newDt-p.sys.Dt) < 1e-14*math.Max(p.newDt, p.sys.Dt) {
		return p.sys.Copy(), nil
	}

	contSys, err := p.sys.d2c(D2COptions{Method: p.opts.Method, PrewarpFrequency: p.opts.PrewarpFrequency})
	if err != nil {
		return nil, err
	}
	result, err := contSys.c2d(p.newDt, p.opts)
	if err != nil {
		return nil, err
	}
	propagateNames(result, p.sys)
	if n, _, _ := result.Dims(); len(result.StateName) != n {
		result.StateName = nil
	}
	return result, nil
}
