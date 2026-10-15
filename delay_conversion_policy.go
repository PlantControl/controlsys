package controlsys

import "plantcontrol.org/v1/gonum/mat"

type delayConversionPolicy struct {
	dt float64
}

func newDelayConversionPolicy(dt float64) delayConversionPolicy {
	return delayConversionPolicy{dt: dt}
}

func (p delayConversionPolicy) applyDiscreteDelayFields(disc, cont *System) (*System, error) {
	var err error
	if cont.Delay != nil {
		disc.Delay, err = convertDelayToDiscrete(cont.Delay, p.dt)
		if err != nil {
			return nil, err
		}
	}
	disc.InputDelay, err = convertSliceDelayToDiscrete(cont.InputDelay, p.dt, 0)
	if err != nil {
		return nil, err
	}
	disc.OutputDelay, err = convertSliceDelayToDiscrete(cont.OutputDelay, p.dt, 0)
	if err != nil {
		return nil, err
	}
	return disc, nil
}

func (p delayConversionPolicy) applyContinuousDelayFields(cont, disc *System) {
	if disc.Delay != nil {
		cont.Delay = convertDelayToContinuous(disc.Delay, p.dt)
	}
	cont.InputDelay = convertSliceDelayToContinuous(disc.InputDelay, p.dt)
	cont.OutputDelay = convertSliceDelayToContinuous(disc.OutputDelay, p.dt)
	if disc.LFT != nil {
		cont.LFT = &LFTDelay{
			Tau: convertSliceDelayToContinuous(disc.LFT.Tau, p.dt),
			B2:  mat.DenseCopyOf(disc.LFT.B2),
			C2:  mat.DenseCopyOf(disc.LFT.C2),
			D12: mat.DenseCopyOf(disc.LFT.D12),
			D21: mat.DenseCopyOf(disc.LFT.D21),
			D22: mat.DenseCopyOf(disc.LFT.D22),
		}
	}
}

func (p delayConversionPolicy) applyDiscreteExternal(disc *System, contInputDelay, contOutputDelay []float64) (*System, error) {
	var err error
	disc.InputDelay, err = convertSliceDelayToDiscrete(contInputDelay, p.dt, 0)
	if err != nil {
		return nil, err
	}
	disc.OutputDelay, err = convertSliceDelayToDiscrete(contOutputDelay, p.dt, 0)
	if err != nil {
		return nil, err
	}
	return disc, nil
}
