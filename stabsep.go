package controlsys

import "fmt"

// StabsepResult is the result of Stabsep: sys = Stable + Unstable.
type StabsepResult struct {
	Stable   *System
	Unstable *System
}

// Stabsep splits sys = Stable + Unstable (MATLAB stabsep). Stable holds the
// modes inside the stability boundary and clear of the boundary band of
// IsStable (Re λ < −tol or |λ| < 1 − tol, tol = 1e-10·max(1, |λ|)); a mode
// within the band goes to Unstable. The band is a deliberate departure from
// MATLAB's strict Re λ < 0 / |λ| < 1, since float64 cannot certify the side
// of an eigenvalue within it. Stable is always proper and carries
// the feedthrough D; Unstable holds the remaining finite modes and, for
// singular-E descriptors, the infinite (nondynamic/improper) modes.
//
// For explicit models, and descriptors with a well-conditioned nonsingular E
// (split through the fold E⁻¹A, the model the bar refers to), a nil error
// means Stable + Unstable equals sys within MATLAB's default SepTol = 10
// times the normwise rounding accuracy of the response,
// ‖G − Stable − Unstable‖_F ≤ 10·u·(‖C‖‖M‖‖A‖‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + ‖D‖),
// M = (sI − A)⁻¹, u = 4(n+5)ε, at every s; this is certified from the
// Sylvester decoupling, not sampled. The split is the stable/unstable
// split of a model within that rounding of sys: a pole's side is promised
// only beyond rounding of the band, so a pole within it, in particular
// within about (Kε)^(1/k)·‖A‖_F (K = 4(n+5)) of a defective cluster of k poles there,
// may land in either part, and the parts' poles may differ from
// Poles(sys) by as much. When the certificate fails, roughly when
// ‖X‖_F·‖C̃1‖‖B̃2‖/(‖C‖‖B‖) exceeds 72(n+5)/(n+2) for the decoupling X
// and the Schur-coordinate blocks C̃1 (outputs of the stable modes) and B̃2
// (inputs to the unstable ones), as for clusters straddling the boundary,
// Stabsep returns ErrSchurFailed ("cannot be separated reliably"), even
// when the split would have been accurate. MATLAB instead warns and keeps
// such modes together, and offers a SepTol option this API does not.
// For descriptors split by the generalized Schur form (singular E, or
// cond₁(E) ≥ 1e4), a nil error means
// ‖G − Stable − Unstable‖_F(s) ≤ 10·u·(‖C‖‖M‖(‖A‖ + |s|‖E‖)‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + ‖D‖),
// M = (sE − A)⁻¹, u = 4(n+5)ε, Frobenius norms of the stored data, to
// first order, at every s with u·(‖A‖ + |s|‖E‖)·‖M(s)‖ ≤ 0.1. Stabsep returns
// ErrSchurFailed instead when (i) the coupling of the retained and
// discarded modes, weighted by ‖C̃1‖‖B̃2‖/(‖C‖‖B‖), exceeds that budget,
// (ii) the cut falls inside a 2×2 block of the generalized Schur form (a
// near-defective real pair on the stability boundary), or (iii) the reduction to that
// form, including the staircase rank decisions and the stiff-mode β edit,
// is not within the budget of an ε-rounding of (A, E).
// A part without states on a model with no inputs or no outputs would be a
// p×0 or 0×m static gain, which cannot be stored: ErrDimensionMismatch.
// See https://www.mathworks.com/help/control/ref/dynamicsystem.stabsep.html.
func Stabsep(sys *System) (*StabsepResult, error) {
	if err := requireFiniteSystem("Stabsep", sys); err != nil {
		return nil, err
	}
	isStable := func(ev complex128) bool {
		return poleInsideStabilityBoundary(ev, sys.IsContinuous(), poleStabilityTolerance(ev))
	}

	stable, unstable, err := decomposeByEigenvalues("Stabsep", sys, isStable, true)
	if err != nil {
		return nil, fmt.Errorf("Stabsep: %w", err)
	}

	return &StabsepResult{Stable: stable, Unstable: unstable}, nil
}
