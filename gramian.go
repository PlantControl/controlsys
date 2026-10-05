package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/mat"
)

// GramType selects the gramian or gramian factor Gram computes, like the
// type argument of MATLAB gram: 'c', 'o', 'cf' and 'of'.
type GramType int

const (
	// GramControllability is the controllability gramian Wc ('c').
	GramControllability GramType = iota
	// GramObservability is the observability gramian Wo ('o').
	GramObservability
	// GramControllabilityFactor is the upper triangular Rc with Wc = Rcᵀ·Rc ('cf').
	GramControllabilityFactor
	// GramObservabilityFactor is the upper triangular Ro with Wo = Roᵀ·Ro ('of').
	GramObservabilityFactor
)

// Gram computes the controllability or observability gramian of a stable LTI
// system, or its Cholesky factor, like MATLAB gram(sys,type).
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
// The factor types return an upper triangular R with W = Rᵀ·R; a semidefinite
// gramian (uncontrollable or unobservable model) still has such a factor.
//
// An unknown type returns ErrInvalidArgument, an unstable model
// ErrUnstableGramian and a model without states ErrDimensionMismatch. Like
// MATLAB gram, models with internal delays (either domain) return
// ErrInternalDelayUnsupported; absorb or approximate the delays first.
// See https://www.mathworks.com/help/control/ref/statespacemodel.gram.html.
func Gram(sys *System, typ GramType) (*mat.Dense, error) {
	if err := requireSystem("Gram", sys); err != nil {
		return nil, err
	}
	base := typ
	switch typ {
	case GramControllability, GramObservability:
	case GramControllabilityFactor:
		base = GramControllability
	case GramObservabilityFactor:
		base = GramObservability
	default:
		return nil, fmt.Errorf("Gram: unknown type %d: %w", typ, ErrInvalidArgument)
	}
	policy := newEnergyAnalysisPolicy(sys)
	if err := policy.requireStandard("Gram"); err != nil {
		return nil, err
	}
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("Gram: %w", ErrInternalDelayUnsupported)
	}
	n := policy.n
	if n == 0 {
		return nil, fmt.Errorf("Gram: system has no states: %w", ErrDimensionMismatch)
	}

	if err := policy.requireStable(ErrUnstableGramian); err != nil {
		return nil, fmt.Errorf("Gram: %w", err)
	}
	Aarg, Q, err := policy.gramianInputs(base)
	if err != nil {
		return nil, fmt.Errorf("Gram: %w", err)
	}
	X, err := policy.solveLyapunov(Aarg, Q)
	if err != nil {
		return nil, fmt.Errorf("Gram: %w", err)
	}
	if typ == base {
		return X, nil
	}
	R, err := gramFactor(X, n)
	if err != nil {
		return nil, fmt.Errorf("Gram: %w", err)
	}
	return R, nil
}

// gramFactor returns an upper triangular R with X = Rᵀ·R for a symmetric
// positive semidefinite X: the Cholesky factor when X is definite, otherwise
// the R of a QR factorization of Λ^½·Vᵀ from X = V·Λ·Vᵀ.
func gramFactor(X *mat.Dense, n int) (*mat.Dense, error) {
	xRaw := X.RawMatrix()
	r := make([]float64, n*n)
	copyStrided(r, n, xRaw.Data, xRaw.Stride, n, n)
	if impl.Dpotrf(blas.Upper, n, r, n) {
		for i := range n {
			clear(r[i*n : i*n+i])
		}
		return mat.NewDense(n, n, r), nil
	}
	var eig mat.EigenSym
	if !eig.Factorize(mat.NewSymDense(n, symmetricData(X, n)), true) {
		return nil, fmt.Errorf("gramian factor: eigenvalue iteration did not converge: %w", ErrSchurFailed)
	}
	vals := eig.Values(nil)
	var V mat.Dense
	eig.VectorsTo(&V)
	M := mat.NewDense(n, n, nil)
	for i, lambda := range vals {
		s := math.Sqrt(max(lambda, 0))
		for j := range n {
			M.Set(i, j, s*V.At(j, i))
		}
	}
	var qr mat.QR
	qr.Factorize(M)
	R := mat.NewDense(n, n, nil)
	qr.RTo(R)
	return R, nil
}

func symmetricData(X *mat.Dense, n int) []float64 {
	data := make([]float64, n*n)
	for i := range n {
		for j := range n {
			data[i*n+j] = (X.At(i, j) + X.At(j, i)) / 2
		}
	}
	return data
}
