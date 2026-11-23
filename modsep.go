package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
)

// ModsepResult is the slow/fast split sys = Slow + Fast returned by Modsep.
type ModsepResult struct {
	Slow *System
	Fast *System
}

// Modsep splits sys = Slow + Fast around cutoff (MATLAB freqsep). Slow holds
// the modes with natural frequency below cutoff; Fast holds the rest, the
// feedthrough D and, for singular-E descriptors, the infinite modes. The
// natural frequency is |s| for continuous and |ln z|/Ts for discrete models.
// cutoff must be positive and finite (ErrInvalidArgument).
// A part without states on a model with no inputs or no outputs would be a
// p×0 or 0×m static gain, which cannot be stored: ErrDimensionMismatch.
//
// For explicit models, and descriptors with a well-conditioned nonsingular E
// (split through the fold E⁻¹A, the model the bar refers to), a nil error
// means Slow + Fast equals sys within MATLAB's default SepTol = 10 times the
// normwise rounding accuracy of the response, as for Stabsep, certified from
// the Sylvester decoupling. A mode's side of cutoff is promised only beyond
// rounding of the stored data: a mode within that rounding of cutoff, in
// particular within about (Kε)^(1/k)·‖A‖_F (K = 4(n+5)) of a defective cluster of k
// modes, may land in either part. A decoupling the certificate cannot accept
// (roughly ‖X‖_F·‖C̃1‖‖B̃2‖/(‖C‖‖B‖) above 72(n+5)/(n+2), as for clusters
// straddling cutoff) returns ErrSchurFailed ("cannot be separated
// reliably"), even when the split would have been accurate; MATLAB warns
// and keeps such modes together instead, and has a SepTol option this API
// does not.
// For descriptors split by the generalized Schur form (singular E, or
// cond₁(E) ≥ 1e4), a nil error means
// ‖G − Slow − Fast‖_F(s) ≤ 10·u·(‖C‖‖M‖(‖A‖ + |s|‖E‖)‖M‖‖B‖ + ‖C‖‖M‖‖B‖ + ‖D‖),
// M = (sE − A)⁻¹, u = 4(n+5)ε, Frobenius norms of the stored data, to
// first order, at every s with u·(‖A‖ + |s|‖E‖)·‖M(s)‖ ≤ 0.1. Modsep returns
// ErrSchurFailed instead when (i) the coupling of the retained and
// discarded modes, weighted by ‖C̃1‖‖B̃2‖/(‖C‖‖B‖), exceeds that budget,
// (ii) the cut falls inside a 2×2 block of the generalized Schur form (a
// near-defective real pair on the cutoff), or (iii) the reduction to that
// form, including the staircase rank decisions and the stiff-mode β edit,
// is not within the budget of an ε-rounding of (A, E).
// See https://www.mathworks.com/help/control/ref/dynamicsystem.freqsep.html.
func Modsep(sys *System, cutoff float64) (*ModsepResult, error) {
	if err := requireFiniteSystem("Modsep", sys); err != nil {
		return nil, err
	}
	if !(cutoff > 0) || math.IsInf(cutoff, 1) {
		return nil, fmt.Errorf("Modsep: cutoff must be positive and finite, got %g: %w", cutoff, ErrInvalidArgument)
	}

	isSlow := func(ev complex128) bool {
		return cmplx.Abs(ev) < cutoff
	}
	if sys.IsDiscrete() {
		dt := complex(sys.Dt, 0)
		isSlow = func(ev complex128) bool {
			return cmplx.Abs(cmplx.Log(ev)/dt) < cutoff
		}
	}

	slow, fast, err := decomposeByEigenvalues("Modsep", sys, isSlow, false)
	if err != nil {
		return nil, err
	}

	return &ModsepResult{Slow: slow, Fast: fast}, nil
}
