package controlsys

import "plantcontrol.org/v1/gonum/mat"

func Augstate(sys *System) (*System, error) {
	n, m, p := sys.Dims()
	if n == 0 {
		return sys.Copy(), nil
	}

	pNew := p + n

	cNew := mat.NewDense(pNew, n, nil)
	cRaw := cNew.RawMatrix()
	origC := sys.C.RawMatrix()
	for i := range p {
		copy(cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n], origC.Data[i*origC.Stride:i*origC.Stride+n])
	}
	for i := range n {
		cRaw.Data[(p+i)*cRaw.Stride+i] = 1
	}

	dNew := mat.NewDense(pNew, m, nil)
	dRaw := dNew.RawMatrix()
	origD := sys.D.RawMatrix()
	for i := range p {
		copy(dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m], origD.Data[i*origD.Stride:i*origD.Stride+m])
	}

	result := &System{
		A:  denseCopy(sys.A),
		B:  denseCopy(sys.B),
		C:  cNew,
		D:  dNew,
		E:  copyDescriptorE(sys.E),
		Dt: sys.Dt,
	}

	if sys.Delay != nil {
		result.Delay = copyDelayOrNil(sys.Delay)
	}
	if sys.InputDelay != nil {
		result.InputDelay = make([]float64, len(sys.InputDelay))
		copy(result.InputDelay, sys.InputDelay)
	}
	if sys.OutputDelay != nil {
		result.OutputDelay = make([]float64, len(sys.OutputDelay))
		copy(result.OutputDelay, sys.OutputDelay)
	}
	if sys.LFT != nil {
		result.LFT = &LFTDelay{
			Tau: append([]float64(nil), sys.LFT.Tau...),
			B2:  copyDelayOrNil(sys.LFT.B2),
			C2:  copyDelayOrNil(sys.LFT.C2),
			D12: copyDelayOrNil(sys.LFT.D12),
			D21: copyDelayOrNil(sys.LFT.D21),
			D22: copyDelayOrNil(sys.LFT.D22),
		}
	}

	result.InputName = copyStringSlice(sys.InputName)
	var outNames []string
	if sys.OutputName != nil {
		outNames = append(outNames, sys.OutputName...)
	} else {
		outNames = make([]string, p)
	}
	stNames := sys.stateLabels()
	outNames = append(outNames, stNames...)
	result.OutputName = outNames
	result.StateName = copyStringSlice(sys.StateName)

	return result, nil
}
