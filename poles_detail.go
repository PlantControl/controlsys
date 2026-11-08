package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// PolesDetail returns the poles exactly as Poles returns them, with a nil
// error only when their number is normwise-certified: the model is
// explicit (E nil or the identity), or E is nonsingular beyond MATLAB's
// rank tolerance, σ_min(E) > n·eps(σ₁(E)), and Poles returns all n
// eigenvalues of (A, E). Every pencil within that tolerance of the data,
// normwise, then has n finite poles, up to the backward error of the
// float64 SVD that computes σ_min (a few eps·‖E‖, as for MATLAB rank).
// "Finite" is meant in this normwise sense, that of backward-stable QZ
// (LAPACK) and MATLAB rank: a pole that only an entrywise perturbation
// smaller than the rounding of the data resolves is not certified.
//
// A singular E, or one within the tolerance of singular, has pencils
// within it with another number of finite poles (a stiff pole can become
// infinite, or the reverse), so the data does not decide its count.
// PolesDetail then returns the poles of Poles, non-nil, with an error
// wrapping ErrAccuracyUnverified, whether or not Poles returns the count of
// the exact pencil. A nil error on a descriptor thus means a robustly
// nonsingular E. Other errors are those of Poles, with nil poles. An
// explicit model costs the same as Poles; a descriptor one adds the
// singular values of E.
func (sys *System) PolesDetail() ([]complex128, error) {
	n, _, _ := sys.Dims()
	if n == 0 {
		return nil, nil
	}
	if err := finiteSystemMatrices("PolesDetail", sys, ErrInvalidArgument); err != nil {
		return nil, err
	}
	if sys.LFT != nil {
		zd, err := sys.zeroDelayApprox()
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
	if policy.isDescriptor() && !(len(poles) == n && eFullRank(sys.E, n)) {
		if poles == nil {
			poles = []complex128{}
		}
		return poles, fmt.Errorf("PolesDetail: pole count not certified: %w", ErrAccuracyUnverified)
	}
	return poles, nil
}

// eFullRank reports whether E has full MATLAB rank by the singular values
// and tolerance with which generalizedPoles counts infinite eigenvalues
// (descriptorRank, full: σ_n > n·eps(σ₁)).
func eFullRank(E *mat.Dense, n int) bool {
	var svd mat.SVD
	if !svd.Factorize(E, mat.SVDNone) {
		return false
	}
	sv := svd.Values(nil)
	r, _, _ := descriptorRank(sv, sv[0], n, true)
	return r == n
}
