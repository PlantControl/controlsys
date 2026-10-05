package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/mat"
)

type GramType int

const (
	GramControllability GramType = iota
	GramObservability
)

type GramResult struct {
	X *mat.Dense
	L *mat.Dense
}

// Gram computes the controllability or observability gramian of a stable LTI system.
//
// Controllability gramian Wc satisfies:
//
//	Continuous: A·Wc + Wc·A' + B·B' = 0
//	Discrete:   A·Wc·A' - Wc + B·B' = 0
//
// Observability gramian Wo satisfies:
//
//	Continuous: A'·Wo + Wo·A + C'·C = 0
//	Discrete:   A'·Wo·A - Wo + C'·C = 0
//
// Like MATLAB gram, models with internal delays (either domain) return
// ErrInternalDelayUnsupported; absorb or approximate the delays first.
func Gram(sys *System, typ GramType) (*GramResult, error) {
	policy := newEnergyAnalysisPolicy(sys)
	if err := policy.requireStandard("Gram"); err != nil {
		return nil, err
	}
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("Gram: %w", ErrInternalDelayUnsupported)
	}
	n := policy.n
	if n == 0 {
		return &GramResult{X: &mat.Dense{}}, nil
	}

	if err := policy.requireStable(ErrUnstableGramian); err != nil {
		return nil, err
	}
	Aarg, Q, err := policy.gramianInputs(typ)
	if err != nil {
		return nil, err
	}
	X, err := policy.solveLyapunov(Aarg, Q)
	if err != nil {
		return nil, err
	}

	res := &GramResult{X: X}

	xRaw := X.RawMatrix()
	lData := make([]float64, n*n)
	copyStrided(lData, n, xRaw.Data, xRaw.Stride, n, n)
	ok := impl.Dpotrf(blas.Upper, n, lData, n)
	if ok {
		res.L = mat.NewDense(n, n, lData)
		for i := range n {
			for j := range i {
				lData[i*n+j] = 0
			}
		}
	}

	return res, nil
}
