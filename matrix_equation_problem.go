package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

type riccatiProblem struct {
	A  *mat.Dense
	B  *mat.Dense
	Q  *mat.Dense
	R  *mat.Dense
	S  *mat.Dense
	E  *mat.Dense
	ws *RiccatiWorkspace
	n  int
	m  int
}

func newRiccatiProblem(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (riccatiProblem, error) {
	na, nac := A.Dims()
	if na != nac {
		return riccatiProblem{}, ErrDimensionMismatch
	}
	nb, m := B.Dims()
	if nb != na {
		return riccatiProblem{}, ErrDimensionMismatch
	}
	qr, qc := Q.Dims()
	if qr != na || qc != na {
		return riccatiProblem{}, ErrDimensionMismatch
	}
	rr, rc := R.Dims()
	if rr != m || rc != m {
		return riccatiProblem{}, ErrDimensionMismatch
	}
	if na == 0 {
		return riccatiProblem{}, fmt.Errorf("controlsys: Riccati problem has no states: %w", ErrDimensionMismatch)
	}
	if !isSymmetric(Q, eps()*denseNorm(Q)) {
		return riccatiProblem{}, ErrNotSymmetric
	}
	if !isSymmetric(R, eps()*denseNorm(R)) {
		return riccatiProblem{}, ErrNotSymmetric
	}
	if !isPSD(Q) {
		return riccatiProblem{}, ErrNotPSD
	}

	var S *mat.Dense
	if opts != nil && opts.S != nil {
		sr, sc := opts.S.Dims()
		if sr != na || sc != m {
			return riccatiProblem{}, ErrDimensionMismatch
		}
		S = opts.S
	}

	E, err := riccatiDescriptor(opts, na)
	if err != nil {
		return riccatiProblem{}, err
	}

	var ws *RiccatiWorkspace
	if opts != nil && opts.Workspace != nil {
		ws = opts.Workspace
	} else {
		ws = NewRiccatiWorkspace(na, m)
	}

	return riccatiProblem{A: A, B: B, Q: Q, R: R, S: S, E: E, ws: ws, n: na, m: m}, nil
}

// riccatiDescriptor returns opts.E, or nil when it is absent or the identity.
func riccatiDescriptor(opts *RiccatiOpts, n int) (*mat.Dense, error) {
	if opts == nil || opts.E == nil {
		return nil, nil
	}
	if er, ec := opts.E.Dims(); er != n || ec != n {
		return nil, ErrDimensionMismatch
	}
	if isIdentityDescriptor(opts.E) {
		return nil, nil
	}
	var lu mat.LU
	lu.Factorize(opts.E)
	if luNearSingular(&lu) {
		return nil, ErrDescriptorSingular
	}
	return opts.E, nil
}

type lyapunovProblem struct {
	A  *mat.Dense
	Q  *mat.Dense
	ws *LyapunovWorkspace
	n  int
}

func newLyapunovProblem(A, Q *mat.Dense, opts *LyapunovOpts) (lyapunovProblem, error) {
	n, nc := A.Dims()
	if n != nc {
		return lyapunovProblem{}, ErrDimensionMismatch
	}
	qr, qc := Q.Dims()
	if qr != n || qc != n {
		return lyapunovProblem{}, ErrDimensionMismatch
	}
	if n == 0 {
		return lyapunovProblem{A: A, Q: Q, n: n}, nil
	}
	if !isSymmetric(Q, eps()*denseNorm(Q)) {
		return lyapunovProblem{}, ErrNotSymmetric
	}

	var ws *LyapunovWorkspace
	if opts != nil {
		ws = opts.Workspace
	}
	return lyapunovProblem{A: A, Q: Q, ws: ws, n: n}, nil
}
