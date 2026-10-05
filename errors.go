package controlsys

import "errors"

var (
	ErrDimensionMismatch = errors.New("controlsys: dimension mismatch")
	ErrSingularTransform = errors.New("controlsys: singular matrix in transformation")
	ErrWrongDomain       = errors.New("controlsys: wrong time domain for operation")
	ErrInvalidSampleTime = errors.New("controlsys: sample time must be positive")
	ErrSingularDenom     = errors.New("controlsys: zero or near-zero leading denominator")
	ErrOverflow          = errors.New("controlsys: coefficient overflow")
	ErrImproperTF        = errors.New("controlsys: transfer function is improper (numerator degree exceeds denominator degree)")

	ErrNegativeDelay        = errors.New("controlsys: delay must be non-negative")
	ErrFractionalDelay      = errors.New("controlsys: discrete delay must be non-negative integer")
	ErrNonUniformInputDelay = errors.New("controlsys: AbsorbDelay requires uniform delay per input column")

	ErrFixedInputDelayMismatch = errors.New("controlsys: nonzero fixed inputs must share input and I/O delays")

	ErrZeroInternalDelay = errors.New("controlsys: internal delay must be positive (tau=0 creates algebraic loop)")

	ErrInternalDelayUnsupported = errors.New("controlsys: operation does not support internal delays; use AbsorbDelay or Pade first")
	ErrContinuousInternalDelay  = errors.New("controlsys: continuous model with internal delays has infinitely many poles; use Pade/AbsorbDelay first")

	ErrAlgebraicLoop = errors.New("controlsys: algebraic loop: (I-D22) singular")

	ErrDomainMismatch  = errors.New("controlsys: systems must share the same time domain")
	ErrFeedbackDelay   = errors.New("controlsys: feedback with delays not supported")
	ErrMixedDelayTypes = errors.New("controlsys: InternalDelay and IODelay cannot coexist")

	ErrDelayNotRepresentable = errors.New("controlsys: target model cannot represent the time delays")
	ErrInternalDelayImpulse  = errors.New("controlsys: continuous impulse response undefined: input feeds an internal delay directly (D21 != 0)")

	ErrNotSymmetric     = errors.New("controlsys: matrix must be symmetric")
	ErrNotPSD           = errors.New("controlsys: Q matrix must be positive semi-definite")
	ErrSchurFailed      = errors.New("controlsys: Schur decomposition failed to converge")
	ErrSingularEquation = errors.New("controlsys: matrix equation is singular or nearly singular")
	ErrNoStabilizing    = errors.New("controlsys: no stabilizing solution exists")
	ErrSingularR        = errors.New("controlsys: R matrix is singular or not positive definite")
	ErrUnstableGramian  = errors.New("controlsys: gramian undefined for unstable system")
	ErrUnstable         = errors.New("controlsys: system is unstable")
	ErrNotMinimal       = errors.New("controlsys: gramian not positive definite; system may not be minimal")
	ErrInvalidOrder     = errors.New("controlsys: reduction order out of range")
	ErrSingularA22      = errors.New("controlsys: A22 block singular; singular perturbation not applicable")

	ErrNotSISO           = errors.New("controlsys: system must be single-input")
	ErrConjugatePairs    = errors.New("controlsys: complex poles must appear in conjugate pairs")
	ErrPoleCount         = errors.New("controlsys: number of poles must equal state dimension")
	ErrUncontrollable    = errors.New("controlsys: uncontrollable mode cannot be assigned")
	ErrInsufficientData  = errors.New("controlsys: insufficient data for estimation")
	ErrInvalidExpression = errors.New("controlsys: invalid sumblk expression")
	ErrSignalNotFound    = errors.New("controlsys: signal name not found")

	ErrNotStabilizable  = errors.New("controlsys: system is not stabilizable")
	ErrNotDetectable    = errors.New("controlsys: system is not detectable")
	ErrInvalidPartition = errors.New("controlsys: invalid generalized plant partition dimensions")
	ErrNoFiniteH2Norm   = errors.New("controlsys: H2 synthesis requires D11 = 0")
	// Deprecated: H2Syn handles D22 != 0 by loop shifting and no longer returns this error.
	ErrH2DirectFeedthrough    = errors.New("controlsys: H2 synthesis requires D22 = 0")
	ErrGammaNotAchievable     = errors.New("controlsys: no stabilizing controller exists for given gamma")
	ErrDescriptorSingular     = errors.New("controlsys: descriptor matrix E is singular")
	ErrDescriptorRiccati      = errors.New("controlsys: standard Riccati solvers do not support descriptor systems (E != I)")
	ErrDescriptorUnsupported  = errors.New("controlsys: operation does not support descriptor systems (E != I)")
	ErrImproperModel          = errors.New("controlsys: cannot simulate the time response of improper models")
	ErrDescriptorInitialState = errors.New("controlsys: cannot simulate state trajectory for models with singular E matrix")
	ErrDelayUnsupported       = errors.New("controlsys: operation does not support this delay structure")
	ErrOptionUnsupported      = errors.New("controlsys: option not supported by this operation")
	ErrNoiseFeedthrough       = errors.New("controlsys: noise inputs must not feed through to outputs (D != 0)")
	ErrInvalidArgument        = errors.New("controlsys: invalid argument")
)
