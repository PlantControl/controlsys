package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// EKFModel defines the nonlinear model for an Extended Kalman Filter.
type EKFModel struct {
	F    func(x, u *mat.VecDense) *mat.VecDense // state transition x_{k+1} = f(x_k, u_k)
	H    func(x *mat.VecDense) *mat.VecDense    // measurement y_k = h(x_k)
	FJac func(x, u *mat.VecDense) *mat.Dense    // Jacobian df/dx at (x, u)
	HJac func(x *mat.VecDense) *mat.Dense       // Jacobian dh/dx at x
	Q    *mat.Dense                             // process noise covariance (n×n)
	R    *mat.Dense                             // measurement noise covariance (p×p)
}

// EKF implements an Extended Kalman Filter for nonlinear systems, like
// MATLAB extendedKalmanFilter with Predict and Update for predict and correct
// (https://www.mathworks.com/help/control/ref/extendedkalmanfilter.html). The
// state estimate and its covariance are read with State and StateCovariance
// and replaced with SetState and SetStateCovariance, which validate them.
type EKF struct {
	model *EKFModel
	x     *mat.VecDense // state estimate
	cov   *mat.Dense    // error covariance
	n     int           // state dimension
	p     int           // measurement dimension

	ap, cp, s     *mat.Dense
	kt, k         *mat.Dense
	ikc, ikcP, kR *mat.Dense
	kRKt          *mat.Dense
	innov, kInnov *mat.VecDense
}

// NewEKF creates an Extended Kalman Filter from a nonlinear model,
// initial state x0, and initial covariance P0.
func NewEKF(model *EKFModel, x0 *mat.VecDense, P0 *mat.Dense) (*EKF, error) {
	if model == nil {
		return nil, fmt.Errorf("NewEKF: nil model: %w", ErrInvalidArgument)
	}
	if model.F == nil || model.H == nil || model.FJac == nil || model.HJac == nil {
		return nil, fmt.Errorf("NewEKF: nil function in model: %w", ErrInvalidArgument)
	}
	if x0 == nil || P0 == nil || model.Q == nil || model.R == nil {
		return nil, fmt.Errorf("NewEKF: nil matrix argument: %w", ErrInvalidArgument)
	}

	n := x0.Len()
	if n == 0 {
		return nil, fmt.Errorf("NewEKF: zero-length state: %w", ErrDimensionMismatch)
	}

	if err := validateCovarianceRole("NewEKF", covarianceStateEstimate, P0, n); err != nil {
		return nil, err
	}
	if err := validateCovarianceRole("NewEKF", covarianceProcessNoise, model.Q, n); err != nil {
		return nil, err
	}
	rr, rc := model.R.Dims()
	if rr != rc {
		return nil, fmt.Errorf("NewEKF: R is %dx%d, must be square: %w", rr, rc, ErrDimensionMismatch)
	}
	p := rr
	contract := newLocalApproximationContract("NewEKF", n, 0, p)

	hProbe := model.H(x0)
	if err := contract.validateMeasurementResult("H(x0)", hProbe); err != nil {
		return nil, err
	}

	xCopy := mat.VecDenseCopyOf(x0)
	pCopy := mat.DenseCopyOf(P0)
	modelCopy := *model
	modelCopy.Q = mat.DenseCopyOf(model.Q)
	modelCopy.R = mat.DenseCopyOf(model.R)

	return &EKF{
		model:  &modelCopy,
		x:      xCopy,
		cov:    pCopy,
		n:      n,
		p:      p,
		ap:     mat.NewDense(n, n, nil),
		cp:     mat.NewDense(p, n, nil),
		s:      mat.NewDense(p, p, nil),
		kt:     mat.NewDense(p, n, nil),
		k:      mat.NewDense(n, p, nil),
		ikc:    mat.NewDense(n, n, nil),
		ikcP:   mat.NewDense(n, n, nil),
		kR:     mat.NewDense(n, p, nil),
		kRKt:   mat.NewDense(n, n, nil),
		innov:  mat.NewVecDense(p, nil),
		kInnov: mat.NewVecDense(n, nil),
	}, nil
}

// Copy returns a deep copy of the filter state, covariance, and work buffers.
// The copy of a nil *EKF is nil.
func (e *EKF) Copy() *EKF {
	if e == nil {
		return nil
	}
	modelCopy := *e.model
	modelCopy.Q = mat.DenseCopyOf(e.model.Q)
	modelCopy.R = mat.DenseCopyOf(e.model.R)
	return &EKF{
		model:  &modelCopy,
		x:      mat.VecDenseCopyOf(e.x),
		cov:    mat.DenseCopyOf(e.cov),
		n:      e.n,
		p:      e.p,
		ap:     mat.DenseCopyOf(e.ap),
		cp:     mat.DenseCopyOf(e.cp),
		s:      mat.DenseCopyOf(e.s),
		kt:     mat.DenseCopyOf(e.kt),
		k:      mat.DenseCopyOf(e.k),
		ikc:    mat.DenseCopyOf(e.ikc),
		ikcP:   mat.DenseCopyOf(e.ikcP),
		kR:     mat.DenseCopyOf(e.kR),
		kRKt:   mat.DenseCopyOf(e.kRKt),
		innov:  mat.VecDenseCopyOf(e.innov),
		kInnov: mat.VecDenseCopyOf(e.kInnov),
	}
}

// State returns a copy of the state estimate.
func (e *EKF) State() *mat.VecDense {
	return mat.VecDenseCopyOf(e.x)
}

// StateCovariance returns a copy of the state estimation error covariance.
func (e *EKF) StateCovariance() *mat.Dense {
	return mat.DenseCopyOf(e.cov)
}

// SetState replaces the state estimate with a copy of x, which must have the
// filter's state dimension and finite entries.
func (e *EKF) SetState(x *mat.VecDense) error {
	if x == nil {
		return fmt.Errorf("EKF.SetState: x is nil: %w", ErrInvalidArgument)
	}
	if x.Len() != e.n {
		return fmt.Errorf("EKF.SetState: x length %d, want %d: %w", x.Len(), e.n, ErrDimensionMismatch)
	}
	xCopy := mat.VecDenseCopyOf(x)
	if err := requireFinite("EKF.SetState", "x", xCopy.RawVector().Data...); err != nil {
		return err
	}
	e.x = xCopy
	return nil
}

// SetStateCovariance replaces the state estimation error covariance with a
// copy of P, which must be n×n, symmetric and positive semidefinite.
func (e *EKF) SetStateCovariance(P *mat.Dense) error {
	if P == nil {
		return fmt.Errorf("EKF.SetStateCovariance: P is nil: %w", ErrInvalidArgument)
	}
	if err := validateCovarianceRole("EKF.SetStateCovariance", covarianceStateEstimate, P, e.n); err != nil {
		return err
	}
	e.cov.Copy(P)
	return nil
}

// Predict performs the EKF prediction step using control input u, which is
// passed unchanged to F and FJac (nil when the model has no inputs). The
// filter keeps its own copy of the vector F returns.
func (e *EKF) Predict(u *mat.VecDense) error {
	contract := newLocalApproximationContract("EKF.Predict", e.n, 0, e.p)
	xPred := e.model.F(e.x, u)
	if err := contract.validateStateResult("F", xPred); err != nil {
		return err
	}

	A := e.model.FJac(e.x, u)
	if err := contract.validateStateJacobian("FJac", A); err != nil {
		return err
	}

	// P = A * P * A' + Q
	e.ap.Mul(A, e.cov)
	e.cov.Mul(e.ap, A.T())
	e.cov.Add(e.cov, e.model.Q)

	e.x.CopyVec(xPred)
	return nil
}

// Update performs the EKF measurement update step using observation z.
func (e *EKF) Update(z *mat.VecDense) error {
	if z == nil {
		return fmt.Errorf("EKF.Update: z is nil: %w", ErrInvalidArgument)
	}
	if z.Len() != e.p {
		return fmt.Errorf("EKF.Update: z length %d, want %d: %w", z.Len(), e.p, ErrDimensionMismatch)
	}
	contract := newLocalApproximationContract("EKF.Update", e.n, 0, e.p)

	yPred := e.model.H(e.x)
	if err := contract.validateMeasurementResult("H", yPred); err != nil {
		return err
	}

	C := e.model.HJac(e.x)
	if err := contract.validateMeasurementJacobian("HJac", C); err != nil {
		return err
	}

	// S = C * P * C' + R  (p×p)
	e.cp.Mul(C, e.cov)
	e.s.Mul(e.cp, C.T())
	e.s.Add(e.s, e.model.R)

	// K = P * C' * S^{-1}  (n×p)
	// From K * S = P * C', solve S' * K' = (P*C')' = C * P.
	// S is symmetric so: S * K' = C * P. Solve for K' (p×n), then transpose.
	if err := e.kt.Solve(e.s, e.cp); err != nil {
		return fmt.Errorf("EKF.Update: innovation covariance solve failed: %w", err)
	}
	e.k.Copy(e.kt.T())

	e.innov.SubVec(z, yPred)

	e.kInnov.MulVec(e.k, e.innov)
	e.x.AddVec(e.x, e.kInnov)

	// Joseph form: P = (I - K*C) * P * (I - K*C)' + K * R * K'
	e.ikc.Mul(e.k, C)
	for i := range e.n {
		for j := range e.n {
			if i == j {
				e.ikc.Set(i, j, 1-e.ikc.At(i, j))
			} else {
				e.ikc.Set(i, j, -e.ikc.At(i, j))
			}
		}
	}

	e.ikcP.Mul(e.ikc, e.cov)
	e.cov.Mul(e.ikcP, e.ikc.T())

	e.kR.Mul(e.k, e.model.R)
	e.kRKt.Mul(e.kR, e.k.T())
	e.cov.Add(e.cov, e.kRKt)

	return nil
}

// Step performs a predict-then-update cycle.
func (e *EKF) Step(u, z *mat.VecDense) error {
	if err := e.Predict(u); err != nil {
		return err
	}
	return e.Update(z)
}
