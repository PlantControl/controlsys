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
		return riccatiProblem{}, fmt.Errorf("A is %d×%d, want square: %w", na, nac, ErrDimensionMismatch)
	}
	nb, m := B.Dims()
	if nb != na {
		return riccatiProblem{}, fmt.Errorf("B has %d rows, want %d: %w", nb, na, ErrDimensionMismatch)
	}
	qr, qc := Q.Dims()
	if qr != na || qc != na {
		return riccatiProblem{}, fmt.Errorf("Q is %d×%d, want %d×%d: %w", qr, qc, na, na, ErrDimensionMismatch)
	}
	rr, rc := R.Dims()
	if rr != m || rc != m {
		return riccatiProblem{}, fmt.Errorf("R is %d×%d, want %d×%d: %w", rr, rc, m, m, ErrDimensionMismatch)
	}
	if na == 0 {
		return riccatiProblem{}, fmt.Errorf("Riccati problem has no states: %w", ErrDimensionMismatch)
	}
	if !isSymmetric(Q, eps()*denseNorm(Q)) {
		return riccatiProblem{}, fmt.Errorf("Q: %w", ErrNotSymmetric)
	}
	if !isSymmetric(R, eps()*denseNorm(R)) {
		return riccatiProblem{}, fmt.Errorf("R: %w", ErrNotSymmetric)
	}
	if !isPSD(Q) {
		return riccatiProblem{}, fmt.Errorf("Q: %w", ErrNotPSD)
	}

	var S *mat.Dense
	if opts != nil && opts.S != nil {
		sr, sc := opts.S.Dims()
		if sr != na || sc != m {
			return riccatiProblem{}, fmt.Errorf("S is %d×%d, want %d×%d: %w", sr, sc, na, m, ErrDimensionMismatch)
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
		return nil, fmt.Errorf("E is %d×%d, want %d×%d: %w", er, ec, n, n, ErrDimensionMismatch)
	}
	if isIdentityDescriptor(opts.E) {
		return nil, nil
	}
	var lu mat.LU
	lu.Factorize(opts.E)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("E: %w", ErrDescriptorSingular)
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
		return lyapunovProblem{}, fmt.Errorf("A is %d×%d, want square: %w", n, nc, ErrDimensionMismatch)
	}
	qr, qc := Q.Dims()
	if qr != n || qc != n {
		return lyapunovProblem{}, fmt.Errorf("Q is %d×%d, want %d×%d: %w", qr, qc, n, n, ErrDimensionMismatch)
	}
	if n == 0 {
		return lyapunovProblem{A: A, Q: Q, n: n}, nil
	}
	if !isSymmetric(Q, eps()*denseNorm(Q)) {
		return lyapunovProblem{}, fmt.Errorf("Q: %w", ErrNotSymmetric)
	}

	var ws *LyapunovWorkspace
	if opts != nil {
		ws = opts.Workspace
	}
	return lyapunovProblem{A: A, Q: Q, ws: ws, n: n}, nil
}
