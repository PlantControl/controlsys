package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

type controllerObserverPolicy struct {
	sys     *System
	context string
	n       int
	m       int
	p       int
}

func newControllerObserverPolicy(sys *System, context string) (controllerObserverPolicy, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		return controllerObserverPolicy{}, fmt.Errorf("%s: system has no states: %w", context, ErrDimensionMismatch)
	}
	descriptor := newDescriptorPolicy(sys)
	if err := descriptor.validate(n); err != nil {
		return controllerObserverPolicy{}, fmt.Errorf("%s: %w", context, err)
	}
	if err := descriptor.requireNonsingular(context); err != nil {
		return controllerObserverPolicy{}, err
	}
	if sys.HasDelay() {
		return controllerObserverPolicy{}, fmt.Errorf("%s: %w", context, ErrDelayUnsupported)
	}
	return controllerObserverPolicy{sys: sys, context: context, n: n, m: m, p: p}, nil
}

func (p controllerObserverPolicy) validateNoise(Qn, Rn *mat.Dense) error {
	if err := validateCovarianceRole(p.context, covarianceProcessNoise, Qn, p.m); err != nil {
		return err
	}
	return validateCovarianceRole(p.context, covarianceMeasurementNoise, Rn, p.p)
}

// rejectOptsE rejects opts.E for model-based designs, whose E is sys.E.
func (p controllerObserverPolicy) rejectOptsE(opts *RiccatiOpts) error {
	if opts != nil && opts.E != nil {
		return fmt.Errorf("%s: opts.E (use the model's E): %w", p.context, ErrOptionUnsupported)
	}
	return nil
}

func validateRegulatorGains(context string, sys *System, K, L *mat.Dense) (n, m, p int, err error) {
	n, m, p = sys.Dims()
	if n == 0 {
		return 0, 0, 0, fmt.Errorf("%s: system has no states: %w", context, ErrDimensionMismatch)
	}
	kr, kc := K.Dims()
	if kr != m || kc != n {
		return 0, 0, 0, ErrDimensionMismatch
	}
	lr, lc := L.Dims()
	if lr != n || lc != p {
		return 0, 0, 0, ErrDimensionMismatch
	}
	return n, m, p, nil
}
