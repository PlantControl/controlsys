package controlsys

import "fmt"

// PolesResult holds the poles of a model with a bracket on how many of them
// are finite and a flag for whether the two agree.
type PolesResult struct {
	// Poles are the poles exactly as Poles returns them.
	Poles []complex128
	// FiniteLo and FiniteHi bracket the number of finite poles of a
	// descriptor pencil (A, E), counted by the Van Dooren staircase of the
	// stored data computed exactly. FiniteHi counts a level singular value
	// as zero only at or below MATLAB's rank tolerance n·eps(σ₁(E)), at
	// every level. FiniteLo also counts as zero every value within a
	// Davis–Kahan/Wedin bound of how far a normwise perturbation of the
	// data of that size (n·eps(σ₁(E)) on E, n·eps(σ₁(A)) on A) moves it;
	// the bound can fall short after a level deflates a value far above
	// the tolerance, so FiniteLo is not a proven lower bound. "Finite" is
	// meant in this normwise sense, that of backward-stable QZ (LAPACK)
	// and MATLAB rank: a pole that only an entrywise perturbation smaller
	// than the rounding of the data resolves need not lie in the bracket.
	// Both are n for an explicit model.
	FiniteLo, FiniteHi int
	// Accuracy is 0 when FiniteLo == FiniteHi == len(Poles): both
	// staircase counts and Poles agree. It is not a certificate: FiniteHi
	// counts a value at or below n·eps(σ₁(E)) as zero, and a pencil within
	// that rounding of the data on which the value is not zero can have
	// one finite pole more, so even at Accuracy 0 the count (and the poles
	// of large magnitude) may be wrong. Accuracy is 1 otherwise: the count
	// is not settled by the data in float64, or Poles disagrees with it.
	// It follows the 0/1 convention of RiccatiResult.Accuracy.
	Accuracy int
}

// PolesDetail returns the poles of Poles together with a bracket on the
// number of finite poles and an Accuracy flag (see PolesResult). Errors
// are those of Poles. Explicit models (E nil or the identity) have
// FiniteLo = FiniteHi = n and Accuracy 0 at the cost of Poles. For a
// descriptor model the bracket comes from a staircase run twice in
// 320-bit arithmetic, identical on every platform. Its one-sided Jacobi
// SVDs cost O(n³) multiple-precision operations per sweep and level: tens
// of milliseconds for n ≈ 10 and seconds for n ≈ 40, typically several
// hundred and up to a few thousand times Poles, so it is a diagnostic,
// not for inner loops.
func (sys *System) PolesDetail() (*PolesResult, error) {
	n, _, _ := sys.Dims()
	if n == 0 {
		return &PolesResult{}, nil
	}
	if sys.LFT != nil {
		zd, err := sys.ZeroDelayApprox()
		if err != nil {
			return nil, fmt.Errorf("PolesDetail: %w", err)
		}
		sys = zd
	}
	policy := newDescriptorPolicy(sys)
	poles, err := policy.poles("PolesDetail", sys.A, n)
	if err != nil {
		return nil, err
	}
	res := &PolesResult{Poles: poles, FiniteLo: n, FiniteHi: n}
	if policy.isDescriptor() {
		c := finiteCountBracket(sys.A, sys.E, nil)
		res.FiniteLo, res.FiniteHi = c.lo, c.hi
	}
	if res.FiniteLo != res.FiniteHi || res.FiniteHi != len(poles) {
		res.Accuracy = 1
	}
	return res, nil
}
