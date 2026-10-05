package controlsys

import "plantcontrol.org/v1/gonum/mat"

// Augstate appends the states to the outputs: y_aug = [y; x].
// State outputs are the undelayed x(t) of the state equations: they inherit
// InputDelay and internal delays through x, but carry no OutputDelay, IODelay
// matrix, or direct internal-delay feedthrough.
func Augstate(sys *System) (*System, error) {
	if err := requireSystem("Augstate", sys); err != nil {
		return nil, err
	}
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

	dNew := newDense(pNew, m)
	if m > 0 {
		dRaw := dNew.RawMatrix()
		origD := sys.D.RawMatrix()
		for i := range p {
			copy(dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m], origD.Data[i*origD.Stride:i*origD.Stride+m])
		}
	}

	result := &System{
		A:  denseCopy(sys.A),
		B:  denseCopy(sys.B),
		C:  cNew,
		D:  dNew,
		E:  copyDescriptorE(sys.E),
		Dt: sys.Dt,
	}

	result.Delay = padZeroRows(sys.Delay, pNew)
	if sys.InputDelay != nil {
		result.InputDelay = append([]float64(nil), sys.InputDelay...)
	}
	if sys.OutputDelay != nil {
		result.OutputDelay = make([]float64, pNew)
		copy(result.OutputDelay, sys.OutputDelay)
	}
	if sys.LFT != nil {
		d12 := newDense(pNew, len(sys.LFT.Tau))
		setBlock(d12, 0, 0, sys.LFT.D12)
		result.LFT = &LFTDelay{
			Tau: append([]float64(nil), sys.LFT.Tau...),
			B2:  copyDelayOrNil(sys.LFT.B2),
			C2:  copyDelayOrNil(sys.LFT.C2),
			D12: d12,
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

func padZeroRows(src *mat.Dense, rows int) *mat.Dense {
	if src == nil {
		return nil
	}
	_, c := src.Dims()
	dst := newDense(rows, c)
	if c == 0 {
		return dst
	}
	s, d := src.RawMatrix(), dst.RawMatrix()
	for i := range s.Rows {
		copy(d.Data[i*d.Stride:i*d.Stride+c], s.Data[i*s.Stride:i*s.Stride+c])
	}
	return dst
}
