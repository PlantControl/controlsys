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
}

func newC2DPlan(sys *System, dt float64, opts C2DOptions) (c2dPlan, error) {
	if sys.IsDiscrete() {
		return c2dPlan{}, fmt.Errorf("DiscretizeWithOpts: system already discrete: %w", ErrWrongDomain)
	}
	sys, _, err := conversionStandardForm(sys, "DiscretizeWithOpts")
	if err != nil {
		return c2dPlan{}, err
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
			return discretizeDelayedChannels(p.sys, p.dt, p.opts)
		}
	}
	if p.workSys.HasInternalDelay() {
		return p.discretizeInternalDelay()
	}
	disc, err := p.discretizeMethod()
	if err != nil {
		return nil, err
	}
	return p.applyExternalDelays(disc)
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

func (p c2dPlan) discretizeMethod() (*System, error) {
	switch p.method {
	case "zoh":
		return p.workSys.discretizeZOH(p.dt)
	case "tustin":
		return p.workSys.discretizeTustin(p.dt, p.opts.PrewarpFrequency)
	case "foh":
		return p.workSys.discretizeModifiedFOH(p.dt)
	case C2DMethodLeastSquares:
		return p.workSys.discretizeLeastSquares(p.dt, p.opts.FitOrder)
	case "impulse":
		return p.workSys.discretizeImpulseParity(p.dt)
	case "matched":
		return p.workSys.discretizeMatched(p.dt)
	default:
		panic("unvalidated C2D method")
	}
}

func (p c2dPlan) applyExternalDelays(disc *System) (*System, error) {
	if p.pathDelay != nil {
		var err error
		if disc, err = conversionPathThiran(disc, p.pathDelay, p.dt, p.opts); err != nil {
			return nil, err
		}
	}
	return applyConversionExternalDelays(p.workSys, disc, p.contInputDelay, p.contOutputDelay, p.dt, p.opts)
}

type d2cPlan struct {
	sys    *System
	method C2DMethod
	opts   D2COptions
}

func newD2CPlan(sys *System, opts D2COptions) (d2cPlan, error) {
	if sys.IsContinuous() {
		return d2cPlan{}, fmt.Errorf("D2C: system already continuous: %w", ErrWrongDomain)
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
		return d2cPlan{}, fmt.Errorf("D2C: unsupported method %q: %w", opts.Method, ErrInvalidConversionOptions)
	}
	if err := validatePrewarp(sys.Dt, opts.Method, opts.PrewarpFrequency); err != nil {
		return d2cPlan{}, err
	}
	sys, _, err := conversionStandardForm(sys, "D2C")
	if err != nil {
		return d2cPlan{}, err
	}
	return d2cPlan{sys: sys, method: opts.Method, opts: opts}, nil
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
		return p.sys.d2cMatched()
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
		return d2dPlan{}, fmt.Errorf("D2D: system is continuous: %w", ErrWrongDomain)
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
		return d2dPlan{}, fmt.Errorf("D2D: unsupported method %q: %w", opts.Method, ErrInvalidConversionOptions)
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

	contSys, err := p.sys.D2CWithOpts(D2COptions{Method: p.opts.Method, PrewarpFrequency: p.opts.PrewarpFrequency})
	if err != nil {
		return nil, fmt.Errorf("D2D: %w", err)
	}
	result, err := contSys.DiscretizeWithOpts(p.newDt, p.opts)
	if err != nil {
		return nil, fmt.Errorf("D2D: %w", err)
	}
	propagateNames(result, p.sys)
	if n, _, _ := result.Dims(); len(result.StateName) != n {
		result.StateName = nil
	}
	return result, nil
}
