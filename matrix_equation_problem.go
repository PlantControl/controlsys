package controlsys

import (
	"fmt"
	"math"

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
	// t is the state scaling x̂ = diag(t)·x of A, B, Q, S, E, or nil.
	t []float64
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

// scale replaces A, B, Q, S and E by their images under the exact
// power-of-two state scaling x̂ = T·x that balances [|A|+|E| B; C ·], with
// Q = C'C represented by the row √diag(Q) and S by S' (both transform like
// C): Â = TAT⁻¹, Ê = TET⁻¹, B̂ = TB, Q̂ = T⁻¹QT⁻¹, Ŝ = T⁻¹S. R is unchanged.
func (problem *riccatiProblem) scale(opts *RiccatiOpts) {
	if opts != nil && opts.NoScaling {
		return
	}
	n, m, ws := problem.n, problem.m, problem.ws
	p := 1
	if problem.S != nil {
		p += m
	}
	br := balancedRealization{n: n, m: m, p: p,
		a: ws.scaledA[:n*n], b: ws.scaledB[:n*m], c: ws.scaleProxy[:p*n]}
	aRaw := problem.A.RawMatrix()
	copyStrided(br.a, n, aRaw.Data, aRaw.Stride, n, n)
	bRaw := problem.B.RawMatrix()
	copyStrided(br.b, m, bRaw.Data, bRaw.Stride, n, m)
	if problem.E != nil {
		br.e = ws.scaledE[:n*n]
		eRaw := problem.E.RawMatrix()
		copyStrided(br.e, n, eRaw.Data, eRaw.Stride, n, n)
	}
	qRaw := problem.Q.RawMatrix()
	for i := range n {
		br.c[i] = math.Sqrt(max(0, qRaw.Data[i*qRaw.Stride+i]))
	}
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		for j := range m {
			for i := range n {
				br.c[(1+j)*n+i] = sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	t := ws.scaleT[:n]
	for i := range t {
		t[i] = 1
	}
	br.balance(t)
	identity := true
	for _, v := range t {
		identity = identity && v == 1
	}
	if identity {
		return
	}

	q := ws.scaledQ[:n*n]
	for i := range n {
		for j := range n {
			q[i*n+j] = qRaw.Data[i*qRaw.Stride+j] / (t[i] * t[j])
		}
	}
	problem.A = mat.NewDense(n, n, br.a)
	problem.B = mat.NewDense(n, m, br.b)
	problem.Q = mat.NewDense(n, n, q)
	if problem.E != nil {
		problem.E = mat.NewDense(n, n, br.e)
	}
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		s := ws.scaledS[:n*m]
		for i := range n {
			for j := range m {
				s[i*m+j] = sRaw.Data[i*sRaw.Stride+j] / t[i]
			}
		}
		problem.S = mat.NewDense(n, m, s)
	}
	problem.t = t
}

// result maps the scaled solution back: X = T·X̂·T, K = K̂·T.
func (problem *riccatiProblem) result(X, K *mat.Dense, eig []complex128, rcnd float64) *RiccatiResult {
	if t := problem.t; t != nil {
		n, m := problem.n, problem.m
		x, k := X.RawMatrix(), K.RawMatrix()
		for i := range n {
			for j := range n {
				x.Data[i*x.Stride+j] *= t[i] * t[j]
			}
		}
		for i := range m {
			for j := range n {
				k.Data[i*k.Stride+j] *= t[j]
			}
		}
	}
	return &RiccatiResult{X: X, K: K, Eig: eig, Rcnd: rcnd}
}
