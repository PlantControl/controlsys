package controlsys

import "plantcontrol.org/v1/gonum/mat"

// Sminreal removes states that are structurally disconnected from the inputs
// or outputs, judged from the sparsity pattern of A, B and C, as MATLAB
// sminreal. It does not change the transfer function and, unlike
// MinimalRealization, involves no numerical rank decisions. Descriptor and
// delayed models are not supported. See
// https://www.mathworks.com/help/control/ref/dynamicsystem.sminreal.html.
func Sminreal(sys *System) (*System, error) {
	if err := requireFiniteSystem("Sminreal", sys); err != nil {
		return nil, err
	}
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

	keep := structuralStates(sys, true, true)
	if len(keep) == n {
		return sys.Copy(), nil
	}

	if len(keep) == 0 {
		if err := storableStaticGain("Sminreal", p, m); err != nil {
			return nil, err
		}
		g, err := newGain(denseCopy(sys.D), sys.Dt)
		if err != nil {
			return nil, err
		}
		propagateIONames(g, sys)
		return g, nil
	}

	ar, br, cr := structuralSubsystem(sys, keep)
	return policy.result(ar, br, cr, denseCopy(sys.D))
}

// structuralStates returns, in order, the states of sys reachable from an
// input (when reach) and observable from an output (when obs) through the
// exact sparsity pattern of A, B and C; a disabled side keeps every state.
func structuralStates(sys *System, reach, obs bool) []int {
	n, m, p := sys.Dims()
	aRaw, bRaw, cRaw := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	reachable := make([]bool, n)
	observable := make([]bool, n)
	for i := range n {
		reachable[i] = !reach
		observable[i] = !obs
		for j := range m {
			reachable[i] = reachable[i] || bRaw.Data[i*bRaw.Stride+j] != 0
		}
		for j := range p {
			observable[i] = observable[i] || cRaw.Data[j*cRaw.Stride+i] != 0
		}
	}
	structuralClosure(aRaw, reachable, false)
	structuralClosure(aRaw, observable, true)
	keep := make([]int, 0, n)
	for i := range n {
		if reachable[i] && observable[i] {
			keep = append(keep, i)
		}
	}
	return keep
}

// structuralSubsystem copies the rows and columns of A, B and C that keep
// selects, exactly and in order.
func structuralSubsystem(sys *System, keep []int) (a, b, c *mat.Dense) {
	_, m, p := sys.Dims()
	r := len(keep)
	aRaw, bRaw, cRaw := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	a, b, c = newDense(r, r), newDense(r, m), newDense(p, r)
	ad, bd, cd := a.RawMatrix().Data, b.RawMatrix().Data, c.RawMatrix().Data
	for ii, i := range keep {
		for jj, j := range keep {
			ad[ii*r+jj] = aRaw.Data[i*aRaw.Stride+j]
		}
		copy(bd[ii*m:(ii+1)*m], bRaw.Data[i*bRaw.Stride:i*bRaw.Stride+m])
		for j := range p {
			cd[j*r+ii] = cRaw.Data[j*cRaw.Stride+i]
		}
	}
	return a, b, c
}
