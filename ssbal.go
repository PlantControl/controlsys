package controlsys

import "plantcontrol.org/v1/gonum/mat"

type SsbalResult struct {
	Sys *System
	T   *mat.Dense
}

// Ssbal balances a delay-free explicit model by a diagonal similarity
// transformation T, returning (T·A/T, T·B, C/T, D) so that the rows and
// columns of [T·A/T, T·B; C/T, 0] have approximately equal 1-norms, as
// MATLAB ssbal.
// The entries of T are powers of two, so the scaling is exact. Diagonal
// entries of A are excluded from the norms and the Osborne iteration follows
// SLICOT TB01ID; unlike MATLAB, T is never permuted and there is no condT
// bound.
func Ssbal(sys *System) (*SsbalResult, error) {
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Ssbal"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Ssbal"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 {
		eye := &mat.Dense{}
		return &SsbalResult{Sys: policy.zeroOrderCopy(), T: eye}, nil
	}

	br := newRealizationCopy(sys, n, m, p)
	t := make([]float64, n)
	for i := range t {
		t[i] = 1
	}
	br.balance(t)

	Anew := mat.NewDense(n, n, br.a)
	Bnew := mat.NewDense(n, m, br.b)
	Cnew := mat.NewDense(p, n, br.c)
	Dnew := denseCopy(sys.D)

	T := mat.NewDense(n, n, nil)
	for i, v := range t {
		T.Set(i, i, v)
	}

	newSys, err := policy.result(Anew, Bnew, Cnew, Dnew)
	if err != nil {
		return nil, err
	}
	return &SsbalResult{Sys: newSys, T: T}, nil
}
