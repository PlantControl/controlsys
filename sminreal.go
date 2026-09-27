package controlsys

import "plantcontrol.org/v1/gonum/mat"

func Sminreal(sys *System) (*System, error) {
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Sminreal"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Sminreal"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 {
		return policy.zeroOrderCopy(), nil
	}

	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()

	reachable := make([]bool, n)
	observable := make([]bool, n)

	for i := range n {
		for j := range m {
			if bRaw.Data[i*bRaw.Stride+j] != 0 {
				reachable[i] = true
				break
			}
		}
	}

	for i := range n {
		for j := range p {
			if cRaw.Data[j*cRaw.Stride+i] != 0 {
				observable[i] = true
				break
			}
		}
	}

	changed := true
	for changed {
		changed = false
		for i := range n {
			if reachable[i] {
				continue
			}
			for j := range n {
				if reachable[j] && aRaw.Data[i*aRaw.Stride+j] != 0 {
					reachable[i] = true
					changed = true
					break
				}
			}
		}
	}

	changed = true
	for changed {
		changed = false
		for i := range n {
			if observable[i] {
				continue
			}
			for j := range n {
				if observable[j] && aRaw.Data[j*aRaw.Stride+i] != 0 {
					observable[i] = true
					changed = true
					break
				}
			}
		}
	}

	var keep []int
	for i := range n {
		if reachable[i] && observable[i] {
			keep = append(keep, i)
		}
	}

	if len(keep) == n {
		return sys.Copy(), nil
	}

	r := len(keep)
	if r == 0 {
		g, err := NewGain(denseCopy(sys.D), sys.Dt)
		if err != nil {
			return nil, err
		}
		propagateIONames(g, sys)
		return g, nil
	}

	ar := mat.NewDense(r, r, nil)
	br := mat.NewDense(r, m, nil)
	cr := mat.NewDense(p, r, nil)

	for ii, i := range keep {
		for jj, j := range keep {
			ar.Set(ii, jj, aRaw.Data[i*aRaw.Stride+j])
		}
		for j := range m {
			br.Set(ii, j, bRaw.Data[i*bRaw.Stride+j])
		}
		for j := range p {
			cr.Set(j, ii, cRaw.Data[j*cRaw.Stride+i])
		}
	}

	return policy.result(ar, br, cr, denseCopy(sys.D))
}
