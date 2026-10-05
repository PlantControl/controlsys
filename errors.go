package controlsys

import "errors"

// Sentinel errors. Every error returned by an exported operation wraps exactly
// one of these with %w, so callers can branch with errors.Is; see CONTEXT.md
// "Error conventions" for the message format and the choice between them.
var (
	// ErrInvalidArgument reports an argument that is wrong regardless of the
	// model's shape: a nil model, function or required matrix, a NaN/Inf value,
	// a value outside its documented range, an unknown enum or method, or
	// out-of-range or non-increasing indices. Use ErrDimensionMismatch when
	// the problem is a size disagreement.
	ErrInvalidArgument = errors.New("controlsys: invalid argument")

	// ErrDimensionMismatch reports incompatible matrix, signal or model sizes,
	// including results that cannot be stored (a model or matrix with no
	// states, or a static gain with exactly one of inputs and outputs zero).
	ErrDimensionMismatch = errors.New("controlsys: dimension mismatch")

	// ErrOptionUnsupported reports a valid option that this operation does not
	// implement for the given model (for example a SISO-only method on MIMO).
	ErrOptionUnsupported = errors.New("controlsys: option not supported by this operation")

	// ErrInvalidConversionOptions reports invalid continuous/discrete conversion
	// options, such as an unknown method or a prewarp frequency above Nyquist.
	ErrInvalidConversionOptions = errors.New("controlsys: invalid conversion options")

	// ErrSingularTransform reports a singular or numerically singular matrix
	// that an operation must invert.
	ErrSingularTransform = errors.New("controlsys: singular matrix in transformation")

	// ErrWrongDomain reports a model in the wrong time domain for the
	// operation, such as discretizing an already discrete model.
	ErrWrongDomain = errors.New("controlsys: wrong time domain for operation")

	// ErrDomainMismatch reports models combined across different time domains
	// or sample times.
	ErrDomainMismatch = errors.New("controlsys: systems must share the same time domain")

	// ErrInvalidSampleTime reports a sample time that is not positive and
	// finite where a discrete model is required.
	ErrInvalidSampleTime = errors.New("controlsys: sample time must be positive")

	// ErrSingularDenom reports a transfer-function denominator whose leading
	// coefficient is zero or negligible.
	ErrSingularDenom = errors.New("controlsys: zero or near-zero leading denominator")

	// ErrOverflow reports polynomial or matrix coefficients that overflow
	// float64 during a conversion.
	ErrOverflow = errors.New("controlsys: coefficient overflow")

	// ErrImproperTF reports a transfer function whose numerator degree exceeds
	// its denominator degree where a proper model is required.
	ErrImproperTF = errors.New("controlsys: transfer function is improper (numerator degree exceeds denominator degree)")

	// ErrNegativeDelay reports a negative time delay.
	ErrNegativeDelay = errors.New("controlsys: delay must be non-negative")

	// ErrFractionalDelay reports a discrete-time delay that is not a whole
	// number of samples.
	ErrFractionalDelay = errors.New("controlsys: discrete delay must be non-negative integer")

	// ErrNonUniformInputDelay reports an AbsorbDelay request that needs the
	// same delay on every channel of an input.
	ErrNonUniformInputDelay = errors.New("controlsys: AbsorbDelay requires uniform delay per input column")

	// ErrFixedInputDelayMismatch reports nonzero fixed inputs whose input and
	// I/O delays differ.
	ErrFixedInputDelayMismatch = errors.New("controlsys: nonzero fixed inputs must share input and I/O delays")

	// ErrZeroInternalDelay reports an internal delay of zero, which would
	// close an algebraic loop.
	ErrZeroInternalDelay = errors.New("controlsys: internal delay must be positive (tau=0 creates algebraic loop)")

	// ErrInternalDelayUnsupported reports an operation that cannot handle
	// internal delays.
	ErrInternalDelayUnsupported = errors.New("controlsys: operation does not support internal delays; use AbsorbDelay or Pade first")

	// ErrContinuousInternalDelay reports a continuous model with internal
	// delays where a finite pole set is required.
	ErrContinuousInternalDelay = errors.New("controlsys: continuous model with internal delays has infinitely many poles; use Pade/AbsorbDelay first")

	// ErrAlgebraicLoop reports an ill-posed interconnection whose I-D22 is
	// singular.
	ErrAlgebraicLoop = errors.New("controlsys: algebraic loop: (I-D22) singular")

	// ErrFeedbackDelay reports a feedback interconnection whose delays are not
	// supported.
	ErrFeedbackDelay = errors.New("controlsys: feedback with delays not supported")

	// ErrMixedDelayTypes reports a model carrying both internal and I/O
	// delays where only one kind is allowed.
	ErrMixedDelayTypes = errors.New("controlsys: InternalDelay and IODelay cannot coexist")

	// ErrDelayNotRepresentable reports a target representation that cannot
	// carry the model's delays.
	ErrDelayNotRepresentable = errors.New("controlsys: target model cannot represent the time delays")

	// ErrInternalDelayImpulse reports a continuous impulse response that is
	// undefined because an input feeds an internal delay directly.
	ErrInternalDelayImpulse = errors.New("controlsys: continuous impulse response undefined: input feeds an internal delay directly (D21 != 0)")

	// ErrDelayUnsupported reports a delay structure the operation does not
	// support.
	ErrDelayUnsupported = errors.New("controlsys: operation does not support this delay structure")

	// ErrNotSymmetric reports a matrix argument that must be symmetric.
	ErrNotSymmetric = errors.New("controlsys: matrix must be symmetric")

	// ErrNotPSD reports a weighting matrix that must be positive semidefinite.
	ErrNotPSD = errors.New("controlsys: Q matrix must be positive semi-definite")

	// ErrSchurFailed reports that an eigenvalue, Schur or QZ iteration did not
	// converge.
	ErrSchurFailed = errors.New("controlsys: Schur decomposition failed to converge")

	// ErrSingularEquation reports a Lyapunov, Sylvester or Riccati equation
	// without a unique solution.
	ErrSingularEquation = errors.New("controlsys: matrix equation is singular or nearly singular")

	// ErrNoStabilizing reports a Riccati equation with no stabilizing solution.
	ErrNoStabilizing = errors.New("controlsys: no stabilizing solution exists")

	// ErrSingularR reports a control or noise weight R that is singular or not
	// positive definite.
	ErrSingularR = errors.New("controlsys: R matrix is singular or not positive definite")

	// ErrUnstableGramian reports a gramian requested for an unstable model.
	ErrUnstableGramian = errors.New("controlsys: gramian undefined for unstable system")

	// ErrUnstable reports an operation that requires a stable model.
	ErrUnstable = errors.New("controlsys: system is unstable")

	// ErrNotMinimal reports a gramian that is not positive definite, so the
	// model is not minimal.
	ErrNotMinimal = errors.New("controlsys: gramian not positive definite; system may not be minimal")

	// ErrInvalidOrder reports a requested model or approximation order out of
	// range.
	ErrInvalidOrder = errors.New("controlsys: reduction order out of range")

	// ErrSingularA22 reports a singular fast-state block in singular
	// perturbation reduction.
	ErrSingularA22 = errors.New("controlsys: A22 block singular; singular perturbation not applicable")

	// ErrNotSISO reports an operation restricted to single-input (or SISO)
	// models.
	ErrNotSISO = errors.New("controlsys: system must be single-input")

	// ErrConjugatePairs reports complex poles that do not come in conjugate
	// pairs.
	ErrConjugatePairs = errors.New("controlsys: complex poles must appear in conjugate pairs")

	// ErrPoleCount reports a pole list whose length differs from the state
	// dimension.
	ErrPoleCount = errors.New("controlsys: number of poles must equal state dimension")

	// ErrUncontrollable reports a requested pole on an uncontrollable mode.
	ErrUncontrollable = errors.New("controlsys: uncontrollable mode cannot be assigned")

	// ErrPoleMultiplicity reports a pole repeated more often than rank(B).
	ErrPoleMultiplicity = errors.New("controlsys: pole multiplicity exceeds rank(B)")

	// ErrInsufficientData reports too few samples for an estimation.
	ErrInsufficientData = errors.New("controlsys: insufficient data for estimation")

	// ErrInvalidExpression reports a malformed SumBlk expression.
	ErrInvalidExpression = errors.New("controlsys: invalid sumblk expression")

	// ErrSignalNotFound reports a signal name absent from the model.
	ErrSignalNotFound = errors.New("controlsys: signal name not found")

	// ErrNotStabilizable reports a model with an unstable uncontrollable mode.
	ErrNotStabilizable = errors.New("controlsys: system is not stabilizable")

	// ErrNotDetectable reports a model with an unstable unobservable mode.
	ErrNotDetectable = errors.New("controlsys: system is not detectable")

	// ErrInvalidPartition reports generalized-plant partition sizes that do
	// not fit the plant.
	ErrInvalidPartition = errors.New("controlsys: invalid generalized plant partition dimensions")

	// ErrNoFiniteH2Norm reports an H2 synthesis plant with D11 != 0.
	ErrNoFiniteH2Norm = errors.New("controlsys: H2 synthesis requires D11 = 0")

	// ErrH2DirectFeedthrough reports an H2 synthesis plant with D22 != 0.
	//
	// Deprecated: H2Syn handles D22 != 0 by loop shifting and no longer returns this error.
	ErrH2DirectFeedthrough = errors.New("controlsys: H2 synthesis requires D22 = 0")

	// ErrGammaNotAchievable reports an H-infinity level with no stabilizing
	// controller.
	ErrGammaNotAchievable = errors.New("controlsys: no stabilizing controller exists for given gamma")

	// ErrDescriptorSingular reports a singular descriptor matrix E where a
	// nonsingular one is required.
	ErrDescriptorSingular = errors.New("controlsys: descriptor matrix E is singular")

	// ErrDescriptorRiccati reports a descriptor plant passed to H2/H-infinity
	// synthesis.
	ErrDescriptorRiccati = errors.New("controlsys: H2/H-infinity synthesis does not support descriptor systems (E != I)")

	// ErrDescriptorUnsupported reports a descriptor model passed to an
	// operation that requires E = I.
	ErrDescriptorUnsupported = errors.New("controlsys: operation does not support descriptor systems (E != I)")

	// ErrImproperModel reports an improper model where a time response or
	// proper realization is required.
	ErrImproperModel = errors.New("controlsys: cannot simulate the time response of improper models")

	// ErrDescriptorInitialState reports an initial-state simulation of a
	// descriptor model with singular E.
	ErrDescriptorInitialState = errors.New("controlsys: cannot simulate state trajectory for models with singular E matrix")

	// ErrNoiseFeedthrough reports noise inputs that feed through to the
	// outputs where Kalman design requires they do not.
	ErrNoiseFeedthrough = errors.New("controlsys: noise inputs must not feed through to outputs (D != 0)")

	// ErrPIDTuningTargetUnattainable reports that no PID controller met the
	// requested target and stability checks within the bounded search. It is
	// not proof of impossibility.
	ErrPIDTuningTargetUnattainable = errors.New("controlsys: PID tuning target unattainable within search bounds")

	// ErrProcessData reports malformed process-identification data.
	ErrProcessData = errors.New("controlsys: invalid process identification data")

	// ErrProcessExcitation reports process-identification data without enough
	// excitation to identify the model.
	ErrProcessExcitation = errors.New("controlsys: insufficient process identification excitation")

	// ErrProcessFit reports that no candidate process model fit the data.
	ErrProcessFit = errors.New("controlsys: no valid process fit")
)
