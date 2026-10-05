package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

type LqgResult struct {
	Controller *System
	K          *mat.Dense
	L          *mat.Dense
	Xc         *mat.Dense
	Xf         *mat.Dense
}

// Lqg computes the linear-quadratic-Gaussian regulator for a state-space system.
// Q, R are state/input weights for LQR; Qn, Rn are process/measurement noise covariances.
// opts applies only to the regulator Riccati equation; the Kalman filter is
// solved with default options. When opts.Workspace is set, K and Xc may share
// its storage. Plants with delays are rejected with ErrDelayUnsupported.
// Returns the observer-based controller and all intermediate gains and Riccati solutions.
func Lqg(sys *System, Q, R, Qn, Rn *mat.Dense, opts *RiccatiOpts) (*LqgResult, error) {
	policy, err := newControllerObserverPolicy(sys, "Lqg")
	if err != nil {
		return nil, err
	}

	kRes, err := policy.regulator(Q, R, opts)
	if err != nil {
		return nil, fmt.Errorf("Lqg: %w", err)
	}

	lRes, err := policy.estimator(Qn, Rn, nil)
	if err != nil {
		return nil, fmt.Errorf("Lqg: %w", err)
	}

	ctrl, err := Reg(sys, kRes.K, lRes.K)
	if err != nil {
		return nil, fmt.Errorf("Lqg: %w", err)
	}

	return &LqgResult{
		Controller: ctrl,
		K:          kRes.K,
		L:          lRes.K,
		Xc:         kRes.X,
		Xf:         lRes.X,
	}, nil
}
