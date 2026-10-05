package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// SS2SS applies the state coordinate transformation x̄ = T·x, as MATLAB
// ss2ss (R2021b and later). An explicit model becomes
// (T·A·T⁻¹, T·B, C·T⁻¹, D); a descriptor model keeps its equations and
// becomes (E·T⁻¹, A·T⁻¹, B, C·T⁻¹, D). Internal-delay channels transform
// like B and C, and I/O delays carry over unchanged. A nil T returns
// ErrInvalidArgument; a model with no states accepts only an empty T.
func SS2SS(sys *System, T *mat.Dense) (*System, error) {
	if err := requireSystem("SS2SS", sys); err != nil {
		return nil, err
	}
	if T == nil {
		return nil, fmt.Errorf("SS2SS: T is nil: %w", ErrInvalidArgument)
	}
	policy := newRealizationTransformPolicy(sys)
	n := policy.n
	if n == 0 {
		if !T.IsEmpty() {
			tr, tc := T.Dims()
			return nil, fmt.Errorf("SS2SS: T must be empty for a model with no states, got %d×%d: %w", tr, tc, ErrDimensionMismatch)
		}
		return policy.zeroOrderCopy(), nil
	}
	if T.IsEmpty() {
		return nil, fmt.Errorf("SS2SS: T must be %d×%d, got empty: %w", n, n, ErrDimensionMismatch)
	}

	tr, tc := T.Dims()
	if tr != n || tc != n {
		return nil, fmt.Errorf("SS2SS: T must be %d×%d, got %d×%d: %w", n, n, tr, tc, ErrDimensionMismatch)
	}
	if err := requireFiniteDense("SS2SS", "T", T); err != nil {
		return nil, err
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

	result := sys.Copy()
	if policy.p > 0 {
		result.C.Mul(sys.C, &Tinv)
	}
	if sys.internalDelayCount() > 0 {
		result.LFT.C2.Mul(sys.LFT.C2, &Tinv)
	}
	if sys.IsDescriptor() {
		result.A.Mul(sys.A, &Tinv)
		result.E.Mul(sys.E, &Tinv)
		return result, nil
	}
	var tmp mat.Dense
	tmp.Mul(sys.A, &Tinv)
	result.A.Mul(T, &tmp)
	if policy.m > 0 {
		result.B.Mul(T, sys.B)
	}
	if sys.internalDelayCount() > 0 {
		result.LFT.B2.Mul(T, sys.LFT.B2)
	}
	return result, nil
}

// Xperm reorders the states so that new state i is old state perm[i], as
// MATLAB xperm: A and E become A[perm,perm] and E[perm,perm], B (and the
// internal-delay B2) take rows perm, C (and C2) take columns perm, and
// StateName is permuted. Delays carry over unchanged.
func Xperm(sys *System, perm []int) (*System, error) {
	if err := requireSystem("Xperm", sys); err != nil {
		return nil, err
	}
	policy := newRealizationTransformPolicy(sys)
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
			return nil, fmt.Errorf("Xperm: index %d out of range [0,%d): %w", v, n, ErrInvalidArgument)
		}
		if seen[v] {
			return nil, fmt.Errorf("Xperm: duplicate index %d: %w", v, ErrInvalidArgument)
		}
		seen[v] = true
	}

	result := sys.Copy()
	permuteSquare(result.A, sys.A, perm)
	if sys.E != nil {
		permuteSquare(result.E, sys.E, perm)
	}
	if policy.m > 0 {
		permuteRows(result.B, sys.B, perm)
	}
	if policy.p > 0 {
		permuteCols(result.C, sys.C, perm)
	}
	if sys.internalDelayCount() > 0 {
		permuteRows(result.LFT.B2, sys.LFT.B2, perm)
		permuteCols(result.LFT.C2, sys.LFT.C2, perm)
	}
	if sys.StateName != nil {
		for i, j := range perm {
			result.StateName[i] = sys.StateName[j]
		}
	}
	return result, nil
}

func permuteSquare(dst, src *mat.Dense, perm []int) {
	for i, pi := range perm {
		for j, pj := range perm {
			dst.Set(i, j, src.At(pi, pj))
		}
	}
}

func permuteRows(dst, src *mat.Dense, perm []int) {
	for i, pi := range perm {
		dst.SetRow(i, src.RawRowView(pi))
	}
}

func permuteCols(dst, src *mat.Dense, perm []int) {
	r, _ := src.Dims()
	for i := range r {
		for j, pj := range perm {
			dst.Set(i, j, src.At(i, pj))
		}
	}
}
