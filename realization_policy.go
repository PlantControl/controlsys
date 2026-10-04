package controlsys

import (
	"fmt"
	"slices"

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

// result builds a realization of the source I/O map. External delays act on
// the I/O channels only, so they carry over; internal delays couple to the
// states and must be rebuilt by the caller via resultWithInternalDelay.
func (p realizationTransformPolicy) result(A, B, C, D *mat.Dense) (*System, error) {
	if p.sys.internalDelayCount() > 0 {
		return nil, fmt.Errorf("realization transform cannot carry internal delays: %w", ErrDelayNotRepresentable)
	}
	result, err := newNoCopy(A, B, C, D, p.sys.Dt)
	if err != nil {
		return nil, err
	}
	p.carryExternal(result)
	return result, nil
}

func (p realizationTransformPolicy) resultWithInternalDelay(A, B, C, D, B2, C2 *mat.Dense) (*System, error) {
	result, err := newNoCopy(A, B, C, D, p.sys.Dt)
	if err != nil {
		return nil, err
	}
	p.carryExternal(result)
	src := p.sys.LFT
	result.LFT = &LFTDelay{
		Tau: slices.Clone(src.Tau),
		B2:  B2,
		C2:  C2,
		D12: denseCopy(src.D12),
		D21: denseCopy(src.D21),
		D22: denseCopy(src.D22),
	}
	return result, nil
}

func (p realizationTransformPolicy) carryExternal(dst *System) {
	dst.Delay = copyDelayOrNil(p.sys.Delay)
	dst.InputDelay = copyFloatSlice(p.sys.InputDelay)
	dst.OutputDelay = copyFloatSlice(p.sys.OutputDelay)
	propagateIONames(dst, p.sys)
}

func (p realizationTransformPolicy) resultWithOriginalFeedthrough(A, B, C *mat.Dense) (*System, error) {
	return p.result(A, B, C, denseCopy(p.sys.D))
}

func (p realizationTransformPolicy) resultWithZeroFeedthrough(A, B, C *mat.Dense) (*System, error) {
	return p.result(A, B, C, denseCopySafe(nil, p.p, p.m))
}

func (p realizationTransformPolicy) zeroOrderOriginalFeedthrough() *System {
	sys := &System{
		A:  &mat.Dense{},
		B:  &mat.Dense{},
		C:  &mat.Dense{},
		D:  denseCopySafe(p.sys.D, p.p, p.m),
		Dt: p.sys.Dt,
	}
	p.carryExternal(sys)
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
	p.carryExternal(sys)
	return sys
}

func (p realizationTransformPolicy) copyWithZeroFeedthrough() *System {
	cp := p.sys.Copy()
	cp.D = denseCopySafe(nil, p.p, p.m)
	return cp
}
