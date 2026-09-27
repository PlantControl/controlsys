package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

type realizationTransformPolicy struct {
	sys     *System
	n, m, p int
}

func newRealizationTransformPolicy(sys *System) realizationTransformPolicy {
	n, m, p := sys.Dims()
	return realizationTransformPolicy{sys: sys, n: n, m: m, p: p}
}

func (p realizationTransformPolicy) requireStandard(context string) error {
	return newDescriptorPolicy(p.sys).requireStandard(context)
}

func (p realizationTransformPolicy) requireDelayFree(context string) error {
	if p.sys.HasDelay() {
		return fmt.Errorf("controlsys: %s does not support delayed systems; use Pade/AbsorbDelay first", context)
	}
	return nil
}

func (p realizationTransformPolicy) zeroOrderCopy() *System {
	return p.sys.Copy()
}

func (p realizationTransformPolicy) result(A, B, C, D *mat.Dense) (*System, error) {
	result, err := newNoCopy(A, B, C, D, p.sys.Dt)
	if err != nil {
		return nil, err
	}
	if p.sys.Delay != nil {
		result.Delay = copyDelayOrNil(p.sys.Delay)
	}
	propagateIONames(result, p.sys)
	return result, nil
}

func (p realizationTransformPolicy) resultWithOriginalFeedthrough(A, B, C *mat.Dense) (*System, error) {
	return p.result(A, B, C, denseCopy(p.sys.D))
}

func (p realizationTransformPolicy) resultWithZeroFeedthrough(A, B, C *mat.Dense) (*System, error) {
	return p.result(A, B, C, denseCopySafe(nil, p.p, p.m))
}

func (p realizationTransformPolicy) zeroOrderOriginalFeedthrough() *System {
	sys := &System{
		A:     &mat.Dense{},
		B:     &mat.Dense{},
		C:     &mat.Dense{},
		D:     denseCopySafe(p.sys.D, p.p, p.m),
		Delay: copyDelayOrNil(p.sys.Delay),
		Dt:    p.sys.Dt,
	}
	propagateIONames(sys, p.sys)
	return sys
}

func (p realizationTransformPolicy) zeroOrderZeroFeedthrough() *System {
	sys := &System{
		A:  &mat.Dense{},
		B:  &mat.Dense{},
		C:  &mat.Dense{},
		D:  denseCopySafe(nil, p.p, p.m),
		Dt: p.sys.Dt,
	}
	propagateIONames(sys, p.sys)
	return sys
}

func (p realizationTransformPolicy) copyWithZeroFeedthrough() *System {
	cp := p.sys.Copy()
	cp.D = denseCopySafe(nil, p.p, p.m)
	return cp
}
