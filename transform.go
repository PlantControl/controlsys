package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

func SS2SS(sys *System, T *mat.Dense) (*System, error) {
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("SS2SS"); err != nil {
		return nil, err
	}
	n := policy.n
	if n == 0 {
		return policy.zeroOrderCopy(), nil
	}

	tr, tc := T.Dims()
	if tr != n || tc != n {
		return nil, fmt.Errorf("SS2SS: T must be %d×%d, got %d×%d: %w", n, n, tr, tc, ErrDimensionMismatch)
	}

	var lu mat.LU
	lu.Factorize(T)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("SS2SS: T is singular: %w", ErrSingularTransform)
	}

	var Tinv mat.Dense
	ones := make([]float64, n)
	for i := range ones {
		ones[i] = 1
	}
	eye := mat.NewDiagDense(n, ones)
	if err := lu.SolveTo(&Tinv, false, eye); err != nil {
		return nil, fmt.Errorf("SS2SS: %w", ErrSingularTransform)
	}

	var tmp, A2, B2, C2 mat.Dense
	tmp.Mul(sys.A, &Tinv)
	A2.Mul(T, &tmp)
	B2.Mul(T, sys.B)
	C2.Mul(sys.C, &Tinv)

	result := sys.Copy()
	result.A = mat.DenseCopyOf(&A2)
	result.B = mat.DenseCopyOf(&B2)
	result.C = mat.DenseCopyOf(&C2)
	return result, nil
}

func Xperm(sys *System, perm []int) (*System, error) {
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Xperm"); err != nil {
		return nil, err
	}
	n := policy.n
	if len(perm) != n {
		return nil, fmt.Errorf("Xperm: perm length %d != state dim %d: %w", len(perm), n, ErrDimensionMismatch)
	}
	if n == 0 {
		return policy.zeroOrderCopy(), nil
	}

	seen := make([]bool, n)
	for _, v := range perm {
		if v < 0 || v >= n {
			return nil, fmt.Errorf("Xperm: index %d out of range [0,%d): %w", v, n, ErrDimensionMismatch)
		}
		if seen[v] {
			return nil, fmt.Errorf("Xperm: duplicate index %d: %w", v, ErrDimensionMismatch)
		}
		seen[v] = true
	}

	P := mat.NewDense(n, n, nil)
	for i, j := range perm {
		P.Set(i, j, 1)
	}

	return SS2SS(sys, P)
}
