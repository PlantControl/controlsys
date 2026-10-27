package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"

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
	// costScaling enables the cost scaling of rescaleCost; sigma is its
	// current σ (1 when unset) and balanced selects the S-free, input
	// balanced extended pencil of riccatiProblem.pencil.
	costScaling bool
	balanced    bool
	sigma       float64
	// rescueErr is the error of a failed first pencil solve that
	// pencilSolution rescued.
	rescueErr error
	// resolved reports that pencilSolution solved more than one pencil;
	// with collecting, candidates holds the solutions it did not return.
	resolved, collecting bool
	candidates           []riccatiPass
	// eFactored reports that ws.eLU holds the LU factors of E.
	eFactored bool
	// poor is the stabilizing solution of smallest first Newton step that
	// verifyRiccati did not certify, or nil.
	poor *riccatiPoor
	// certStep is the first Newton step of the X verifyRiccati last
	// certified, and certX1 (Dare, step above eps) the iterate it reached.
	certStep float64
	certX1   []float64
	// xe is EᵀXE of the X verifyRiccati last certified when
	// refineDescriptorGain formed it, else nil.
	xe []float64
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
	// isPSD's Cholesky shift is 0 for Q = 0, which icare/idare accept.
	if denseNorm(Q) != 0 && !(opts != nil && opts.reducedQ) && !isPSD(Q) {
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

	return riccatiProblem{A: A, B: B, Q: Q, R: R, S: S, E: E, ws: ws, n: na, m: m, sigma: 1}, nil
}

// riccatiDescriptor returns opts.E, or nil when it is absent or the identity.
// An E with LU condition estimate at or above 1/ε that MATLAB rank finds
// deficient is singular.
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
	if luNearSingular(&lu) && eRankDeficient(opts.E, &lu) {
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
// power-of-two state scaling x̂ = T·x that balances [|A|+|E| B̃; C ·], with
// Q = C'C represented by the row √diag(Q) and S by S' (both transform like
// C): Â = TAT⁻¹, Ê = TET⁻¹, B̂ = TB, Q̂ = T⁻¹QT⁻¹, Ŝ = T⁻¹S. R is unchanged.
// B̃ = B·U⁻¹ (R = UᵀU, so B̃B̃ᵀ = BR⁻¹Bᵀ) scaled to ‖C‖_F balances the
// cost-scaled Hamiltonian [A, −σG; −Q/σ, −Aᵀ] at ‖σG‖ ≈ ‖Q/σ‖: B in place
// of B̃ ignores R, and under cheap control (R ≪ BᵀXB) such a T leaves no
// cost scale that deflates the pencil (3CVH7I). Without costProxy, or
// with an R not positive definite or a zero C, it balances with B.
func (problem *riccatiProblem) scale(opts *RiccatiOpts, costProxy bool) {
	if opts != nil && opts.NoScaling {
		return
	}
	problem.costScaling = true
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
	if costProxy {
		cn := impl.Dlange(lapack.Frobenius, p, n, br.c, n, nil)
		rChol := ws.rChol[:m*m]
		rRaw := problem.R.RawMatrix()
		copyStrided(rChol, m, rRaw.Data, rRaw.Stride, m, m)
		costProxy = cn > 0 && impl.Dpotrf(blas.Upper, m, rChol, m)
		if costProxy {
			bt := blas64.General{Rows: n, Cols: m, Data: br.b, Stride: m}
			blas64.Trsm(blas.Right, blas.NoTrans, 1,
				blas64.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit, N: m, Data: rChol, Stride: m}, bt)
			bn := impl.Dlange(lapack.Frobenius, n, m, br.b, m, nil)
			costProxy = bn > 0 && !math.IsInf(bn, 0)
			if costProxy {
				blas64.Scal(cn/bn, blas64.Vector{N: n * m, Data: br.b, Inc: 1})
			}
		}
		if !costProxy {
			copyStrided(br.b, m, bRaw.Data, bRaw.Stride, n, m)
		}
	}
	t := ws.scaleT[:n]
	for i := range t {
		t[i] = 1
	}
	br.balance(t)
	if costProxy {
		for i := range n {
			for k := range m {
				br.b[i*m+k] = bRaw.Data[i*bRaw.Stride+k] * t[i]
			}
		}
	}
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
func (problem *riccatiProblem) result(X, K *mat.Dense, eig []complex128, rcnd float64, accuracy int) *RiccatiResult {
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
	res := &RiccatiResult{X: X, K: K, Eig: eig, Rcnd: rcnd, Accuracy: accuracy}
	if accuracy == 0 {
		res.step = problem.certStep
		if x1 := problem.certX1; x1 != nil {
			n, t := problem.n, problem.t
			if t != nil {
				for i := range n {
					for j := range n {
						x1[i*n+j] *= t[i] * t[j]
					}
				}
			}
			res.x1 = mat.NewDense(n, n, x1)
			problem.certX1 = nil
		}
		if xe := problem.xe; xe != nil {
			n, t := problem.n, problem.t
			if t != nil {
				for i := range n {
					for j := range n {
						xe[i*n+j] *= t[i] * t[j]
					}
				}
			}
			res.xe = mat.NewDense(n, n, xe)
			problem.xe = nil
		}
	} else if problem.poor != nil {
		res.step = problem.poor.step
	}
	return res
}
