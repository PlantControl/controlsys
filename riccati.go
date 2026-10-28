package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// RiccatiWorkspace is reusable scratch storage for Care and Dare, sized by
// NewRiccatiWorkspace for a state dimension n and m columns of B. It serves
// any problem with at most n states and m inputs; a smaller one returns
// ErrDimensionMismatch. Kalman, Kalmd and Lqe solve the dual problem, so
// their workspace needs m = number of measured outputs.
//
// A workspace is not safe for concurrent use, and results computed with it
// share its storage until the next call that reuses it.
type RiccatiWorkspace struct {
	n, m        int
	rChol       []float64
	aWork       []float64
	qWork       []float64
	rinvBt      []float64
	rinvSt      []float64
	g           []float64
	h           []float64
	wr          []float64
	wi          []float64
	vs          []float64
	bwork       []bool
	work        []float64
	u11         []float64
	u21         []float64
	ipiv        []int
	xData       []float64
	eig         []complex128
	kData       []float64
	z           []float64
	beta        []float64
	pencilH     []float64
	pencilJ     []float64
	pencilInput []float64
	tau         []float64
	btx         []float64
	rbar        []float64
	costV       []float64
	gainRes     []float64
	eLU         []float64
	eIpiv       []int
	eRes        []float64
	iwork       []int
	scaleT      []float64
	scaledA     []float64
	scaledE     []float64
	scaledQ     []float64
	scaledB     []float64
	scaledS     []float64
	scaleProxy  []float64
	certX1      []float64
	xe          []float64
	cancel      []float64
	gainStep    []float64
	refine      []float64
	lyap        *LyapunovWorkspace
}

// NewRiccatiWorkspace returns a workspace for problems with up to n states
// and m inputs.
func NewRiccatiWorkspace(n, m int) *RiccatiWorkspace {
	nn := 2 * n
	ws := &RiccatiWorkspace{
		n:           n,
		m:           m,
		rChol:       make([]float64, m*m),
		aWork:       make([]float64, n*n),
		qWork:       make([]float64, n*n),
		rinvBt:      make([]float64, m*n),
		rinvSt:      make([]float64, m*n),
		g:           make([]float64, n*n),
		h:           make([]float64, nn*nn),
		wr:          make([]float64, nn),
		wi:          make([]float64, nn),
		vs:          make([]float64, nn*nn),
		bwork:       make([]bool, nn),
		work:        make([]float64, nn*50),
		u11:         make([]float64, n*n),
		u21:         make([]float64, n*n),
		ipiv:        make([]int, n),
		xData:       make([]float64, n*n),
		eig:         make([]complex128, n),
		kData:       make([]float64, m*n),
		z:           make([]float64, nn*nn),
		beta:        make([]float64, nn),
		pencilH:     make([]float64, (nn+m)*nn),
		pencilJ:     make([]float64, (nn+m)*nn),
		pencilInput: make([]float64, (nn+m)*m),
		tau:         make([]float64, m),
		btx:         make([]float64, m*n),
		rbar:        make([]float64, m*m),
		costV:       make([]float64, m*n),
		gainRes:     make([]float64, 2*m*n),
		iwork:       make([]int, n),
	}
	scale := make([]float64, n+3*n*n+2*n*m+(1+m)*n)
	ws.scaleT, scale = scale[:n:n], scale[n:]
	ws.scaledA, scale = scale[:n*n:n*n], scale[n*n:]
	ws.scaledE, scale = scale[:n*n:n*n], scale[n*n:]
	ws.scaledQ, scale = scale[:n*n:n*n], scale[n*n:]
	ws.scaledB, scale = scale[:n*m:n*m], scale[n*m:]
	ws.scaledS, ws.scaleProxy = scale[:n*m:n*m], scale[n*m:]
	return ws
}

// RiccatiOpts holds the optional arguments of Care and Dare; nil means none.
type RiccatiOpts struct {
	// S is the optional cross-term matrix. It is read during the call.
	S *mat.Dense
	// E is the optional nonsingular descriptor matrix of E x' = Ax + Bu
	// (E x[k+1] = Ax[k] + Bu[k]), as in MATLAB icare/idare. nil means I.
	// It is read during the call.
	E *mat.Dense
	// Workspace supplies reusable scratch storage. Results may share its storage
	// until the next call that reuses the same workspace, and it must not be
	// shared across goroutines.
	Workspace *RiccatiWorkspace
	// NoScaling disables the default power-of-two state and cost scaling,
	// as the MATLAB icare/idare 'noscaling' option. A pencil solve (Dare,
	// or Care with E) that fails, or leaves a Riccati residual above 1e-12
	// of its terms, is still retried with cost scaling; so is a plain Care
	// solve that fails or leaves a residual above 1e-6 of its terms. A
	// solve in either mode that finds no stabilizing solution, or one no
	// Newton step certifies (Accuracy 1), is retried once in the other,
	// and Care's then state-scaled ignoring R.
	NoScaling bool
	// plainScaling balances Care's state scaling with B in place of its
	// cost proxy (scale): otherScaling's last resort.
	plainScaling bool
	// reducedQ marks Q as Dare's Q − S₂R₂₂⁻¹S₂ᵀ of reduceInputs, which
	// may be indefinite as Q − SR⁻¹Sᵀ of the given Dare may.
	reducedQ bool
	// plantE is the model E when E is its transpose (Kalman's dual), so
	// nearSingularEGate keeps the window of both condition estimates.
	plantE *mat.Dense
}

// RiccatiResult is the stabilizing solution of Care or Dare.
type RiccatiResult struct {
	// X is the stabilizing solution, K the state-feedback gain and Eig the
	// closed-loop eigenvalues. They are caller-owned unless a workspace was
	// supplied.
	X   *mat.Dense
	K   *mat.Dense
	Eig []complex128
	// Rcnd is the reciprocal 1-norm condition estimate of U11, where
	// [U11; U21] spans the stable invariant subspace and X = U21·U11⁻¹. A
	// tiny Rcnd means X may be inaccurate. U11 is that of the state- and
	// cost-scaled problem actually solved or, for Dare's pole-placement
	// fallback, of the orthonormal basis of span[I; X̂/σ], σ ≈ ‖X̂‖.
	Rcnd float64
	// Accuracy is the Report of MATLAB icare/idare for X: 0 when a Newton
	// step certifies X (to about 1e-8 relative), 1 when X is a stabilizing
	// Hamiltonian or pencil solution (A − BK stable in float64 arithmetic)
	// that no Newton step certifies, so X may be inaccurate and, near the
	// stability boundary, A − BK only rounded to stable; check it before
	// use. A non-finite X or no stabilizing solution (Report 2, 3) is an
	// error, not a result, as is an X with an unstabilizable mode of (A, B)
	// on the boundary to within rounding (PBH).
	Accuracy int

	// step is the first Newton step of an Accuracy 1 X or, at Accuracy 0,
	// that of the X verifyRiccati certified, its relative error, and x1
	// (Dare, step above eps) the Newton iterate it reached.
	step float64
	x1   *mat.Dense
	// xe is EᵀXE of a certified descriptor solution refined by
	// refineDescriptorGain or refineDescriptorDare, else nil, and xeStep
	// the Newton step of EᵀXE of the certified X.
	xe     *mat.Dense
	xeStep float64
	// xeX is ‖X − X_dd‖/‖X‖ for the double-double iterate X_dd whose K
	// and EᵀXE refineDescriptorDare returned in xe.
	xeX float64
	// xeLoose reports an xe of refineDescriptorDare whose next Newton
	// step exceeds newtonRefineTol (explicitRiccatiSolution: Accuracy 1).
	xeLoose bool
	// gainUncertified reports a certified X whose K Care cannot certify
	// with it (descriptorGainCertified); Care returns it with Accuracy 1.
	gainUncertified bool
	// xeCertified reports an xe and K of nearSingularDare certified while
	// X, at Accuracy 1, is not: explicitRiccatiSolution returns them at
	// Accuracy 0.
	xeCertified bool
	// kw is the noise gain (R + BᵀXB)⁻¹BᵀX of nearSingularDare, formed in
	// extended precision, for Lqg's current estimator; kwCertified when
	// eps perturbations of the data move it by at most newtonTol.
	kw          *mat.Dense
	kwCertified bool
}

// riccatiArgs rejects nil or non-finite Riccati data before it reaches
// LAPACK.
func riccatiArgs(op string, A, B, Q, R *mat.Dense, opts *RiccatiOpts) error {
	for _, a := range []struct {
		name string
		m    *mat.Dense
	}{{"A", A}, {"B", B}, {"Q", Q}, {"R", R}} {
		if err := requireFiniteDense(op, a.name, a.m); err != nil {
			return err
		}
	}
	if opts == nil {
		return nil
	}
	if opts.S != nil {
		if err := requireFiniteDense(op, "S", opts.S); err != nil {
			return err
		}
	}
	if opts.E != nil {
		if err := requireFiniteDense(op, "E", opts.E); err != nil {
			return err
		}
	}
	return nil
}

func (ws *RiccatiWorkspace) fits(n, m int) error {
	if ws.n < n || ws.m < m {
		return fmt.Errorf("workspace sized for n=%d m=%d, need n=%d m=%d: %w", ws.n, ws.m, n, m, ErrDimensionMismatch)
	}
	return nil
}

// Care solves the continuous algebraic Riccati equation:
//
//	A'X + XA - (XB+S)*R⁻¹*(B'X+S') + Q = 0
//
// When opts is nil or opts.S is nil, the cross-term is zero:
//
//	A'X + XA - XB*R⁻¹*B'X + Q = 0
//
// A is n×n, B is n×m, Q is n×n symmetric, R is m×m symmetric positive definite.
//
// With opts.E, Care solves the generalized equation of MATLAB icare
// (https://www.mathworks.com/help/control/ref/icare.html)
//
//	A'XE + E'XA - (E'XB+S)*R⁻¹*(B'XE+S') + Q = 0,  K = R⁻¹*(B'XE+S')
//
// from the stable deflating subspace of the extended pencil, and Eig holds
// the generalized eigenvalues of (A-BK, E). X is the generalized solution;
// E'XE solves the Care of the explicit model (E⁻¹A, E⁻¹B). A singular E
// (LU condition estimate at or above 1/ε and rank deficient by MATLAB rank)
// returns ErrDescriptorSingular, as does a near-singular E (either test
// alone) whose result is not certified (Accuracy 1) or whose K or E'XE
// cancels in X ~ 1/σ_min(E) (nearSingularEGate). Nil or NaN/Inf arguments
// return ErrInvalidArgument.
//
// As MATLAB icare, Care first balances the states by an exact power-of-two
// diagonal scaling, so X, K and Eig do not depend on the units of x; the
// balance is that of the Hamiltonian, with BR⁻¹Bᵀ weighed as Q. When
// ‖X‖ is far from the scale of the Hamiltonian (for example a cost Q, R
// tiny against A), Care solves again with Q, R, S divided by a power of
// two near ‖X‖, which keeps the relative accuracy of X. opts.NoScaling
// disables both. A failed first solve (Schur failure, no stabilizing
// subspace) is retried, also under NoScaling, at the cost scale
// √(‖Q̄‖/‖G‖); the retry's X is kept only with a Riccati residual within
// 1e-6 of its terms and a stable A − BK, else the first error is returned.
// A solution with a residual above 1e-6 of its terms is likewise re-solved
// at that cost scale, and the more accurate of the two is kept.
//
// Care checks X by a Newton–Kleinman step from its gain (one Lyapunov
// solve of the double-double Riccati residual; with E, for E⁻¹(A − BK)
// and EᵀΔE), as Dare does
// with Hewer's step: the Hamiltonian forms BR⁻¹Bᵀ, which under cheap
// control (R ≪ BᵀXB) swamps A and loses digits the Riccati residual does
// not show. An X the step moves by more than 1e-8 relative is refined by
// Newton's method, and an iterate is kept only once Newton has converged
// (a step within 1e-10); when that fails (or A − BK is unstable, or the
// Hamiltonian solve fails, or E is given) Care solves the extended pencil
// of MATLAB icare, verified the same way at each of its cost scales. A
// certified X has Accuracy 0. With E, where K and EᵀXE formed from X ~
// 1/σ_min(E) cancel (beyond nearSingularCancelLimit), or the Newton step
// of X moves K or EᵀXE beyond 1e-10 (a step of 1.7e-9 left K off by
// 9.2e-7), K and the EᵀXE of Lqr are those of P = XE refined by Newton's
// method in double-double, and X is certified only once that converges
// or the cancellation times eps and the step's move of K are within
// 1e-8; without cancellation X stays certified, at Accuracy 1 when the
// step moves K beyond 1e-8. With none certified, Care returns, as
// MATLAB icare with Report 1, the stabilizing solution (A − BK stable in
// float64) of smallest first Newton step (or, without one, a stabilizing
// X whose Lyapunov equation is singular) with Accuracy 1, unrefined:
// under cheap control with weak actuation Newton has no quadratic region
// (its first step from an X accurate to 1e-11 can be 1e-6), and such an
// X may be accurate or off by far more than its step; when (A, B) has a
// mode on the boundary (to its eigenvalue's rounding error) that fails
// the PBH rank test, no stabilizing solution exists and Care returns
// ErrNoStabilizing instead, in either mode. An X whose U11 has
// rcnd below eps is not finite (icare Report 2) and is never returned.
// A solve that finds no stabilizing solution, or only an Accuracy 1 one,
// is retried with NoScaling flipped and then state-scaled balancing B in
// place of BR⁻¹Bᵀ: the first certified X is returned, else the Accuracy
// 1 X of smallest first Newton step; with none Care returns
// ErrNoStabilizing (or the Hamiltonian's error) of the first.
func Care(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	if err := riccatiArgs("Care", A, B, Q, R, opts); err != nil {
		return nil, err
	}
	res, err := otherScaling(care, true, A, B, Q, R, opts)
	if err != nil {
		err = fmt.Errorf("Care: %w", err)
	}
	res, err = nearSingularEGate("Care", true, A, B, R, opts, res, err)
	if err == nil && res.Accuracy == 0 && res.gainUncertified {
		res.Accuracy = 1
	}
	return res, err
}

// nearSingularCancelLimit is the largest ratio ‖|M₁||X||M₂|‖/‖M₁XM₂ + C‖
// of the products that K and EᵀXE are formed from at which a certified X
// of a near-singular E keeps them to within that factor of its own
// accuracy.
const nearSingularCancelLimit = 1e2

// nearSingularEGate screens the result of a Care or Dare whose E (or, for
// Kalman's dual, opts.plantE) is near singular: its LU condition estimate
// reaches 1/ε or MATLAB rank (eRankDeficient) finds it deficient. MATLAB
// lqr and icare accept an E of full MATLAB rank as nonsingular, but there
// X ~ 1/σ_min(E), and K = R⁻¹(BᵀXE + Sᵀ) (continuous), the discrete K
// from BᵀXA and BᵀXB, and EᵀXE cancel: a certified X correct to every
// digit left L of Kalman off by 0.3. Such a result is returned only with
// Accuracy 0 and none of these products cancelling beyond
// nearSingularCancelLimit, else ErrDescriptorSingular. A solver error
// stays, except that one with a condition estimate at or above 1/ε, which
// the E check rejected before MATLAB rank admitted it, is
// ErrDescriptorSingular.
func nearSingularEGate(op string, continuous bool, A, B, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult, err error) (*RiccatiResult, error) {
	if opts == nil || isIdentityDescriptor(opts.E) {
		return res, err
	}
	if err != nil && (errors.Is(err, ErrInvalidArgument) || errors.Is(err, ErrDimensionMismatch) ||
		errors.Is(err, ErrNotSymmetric) || errors.Is(err, ErrNotPSD) || errors.Is(err, ErrDescriptorSingular)) {
		return res, err
	}
	condNear, rankDeficient := nearSingularE(opts.E)
	if opts.plantE != nil {
		c, r := nearSingularE(opts.plantE)
		condNear, rankDeficient = condNear || c, rankDeficient || r
	}
	if !condNear && !rankDeficient || err != nil && !condNear {
		return res, err
	}
	if err == nil && res.Accuracy == 0 && !nearSingularCancels(continuous, A, B, R, opts, res.X) {
		return res, nil
	}
	return nil, fmt.Errorf("%s: E: %w", op, ErrDescriptorSingular)
}

// nearSingularE reports whether the LU condition estimate of E reaches 1/ε
// and whether E is rank deficient by MATLAB rank.
func nearSingularE(E *mat.Dense) (condNear, rankDeficient bool) {
	var lu mat.LU
	lu.Factorize(E)
	return luNearSingular(&lu), eRankDeficient(E, &lu)
}

// nearSingularCancels reports whether a product K or EᵀXE is formed from
// cancels by more than nearSingularCancelLimit.
func nearSingularCancels(continuous bool, A, B, R *mat.Dense, opts *RiccatiOpts, X *mat.Dense) bool {
	absX := absDense(X)
	cancels := func(M1t, M2, add mat.Matrix) bool {
		var t, p, at, ap mat.Dense
		t.Mul(M1t, X)
		p.Mul(&t, M2)
		at.Mul(absDense(mat.DenseCopyOf(M1t)), absX)
		ap.Mul(&at, absDense(mat.DenseCopyOf(M2)))
		if add != nil {
			p.Add(&p, add)
			ap.Add(&ap, absDense(mat.DenseCopyOf(add)))
		}
		return !(denseNorm(&ap) <= nearSingularCancelLimit*denseNorm(&p))
	}
	E := opts.E
	if cancels(E.T(), E, nil) {
		return true
	}
	var St mat.Matrix
	if opts.S != nil {
		St = opts.S.T()
	}
	if continuous {
		return cancels(B.T(), E, St)
	}
	return cancels(B.T(), A, St) || cancels(B.T(), B, R)
}

// otherScaling solves by solve and, when that finds no stabilizing
// solution (ErrNoStabilizing or ErrSchurFailed) or one no Newton step
// certifies (Accuracy 1), once more with opts.NoScaling flipped, and then,
// when continuous (Care), state-scaled with B in place of its cost proxy
// (opts.plainScaling).
// State scaling can push a cheap-control pencil (R ≪ BᵀXB) to where no
// cost scale deflates it while the unscaled one solves (w17 #8), and the
// reverse; each solution passes the same Newton verification. The first
// certified result is kept; else of Accuracy 1 results the one of
// smallest first Newton step (the first on a tie), and with all failing
// the first error. An unstabilizable boundary mode
// (errBoundaryUnstabilizable) in any mode overrides every Accuracy 1
// result but not a certified one: its rounding band is a heuristic that
// a non-normal block can inflate past a mode inside the boundary (seen
// unscaled only), while a Newton-certified X is stable. A certified X
// with a closed-loop mode within rounding of the boundary
// (boundaryCertified) is a boundary verdict when both the unscaled and
// the state-scaled problem have an unstabilizable boundary mode: an
// exactly uncontrollable Jordan block at 1 left the closed loop at
// 1 − 3e-16 ± 1e-8i with X ≈ 1e49 that Newton could not move (MPWOJM).
// A kept Accuracy 1 result lives in opts.Workspace, so later solves run
// without it.
func otherScaling(solve func(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error), continuous bool, A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	res, err := solve(A, B, Q, R, opts)
	if err == nil && res.Accuracy == 0 && boundaryCertified(continuous, A, B, Q, R, opts, res) {
		res, err = nil, errBoundaryUnstabilizable
	}
	if err == nil && res.Accuracy == 0 ||
		err != nil && !errors.Is(err, ErrNoStabilizing) && !errors.Is(err, ErrSchurFailed) {
		return res, err
	}
	var boundary error
	if errors.Is(err, errBoundaryUnstabilizable) {
		boundary = err
	}
	var base RiccatiOpts
	if opts != nil {
		base = *opts
	}
	flipped, plain := base, base
	flipped.NoScaling = !base.NoScaling
	plain.NoScaling, plain.plainScaling = false, true
	retries := []RiccatiOpts{flipped, plain}
	if !continuous {
		retries = retries[:1]
	}
	for _, o := range retries {
		if res != nil {
			o.Workspace = nil
		}
		r, e := solve(A, B, Q, R, &o)
		if e == nil && r.Accuracy == 0 && boundaryCertified(continuous, A, B, Q, R, opts, r) {
			r, e = nil, errBoundaryUnstabilizable
		}
		switch {
		case errors.Is(e, errBoundaryUnstabilizable):
			if boundary == nil {
				boundary = e
			}
		case e != nil:
		case r.Accuracy == 0:
			return r, nil
		case res == nil || r.step < res.step:
			res = r
		}
	}
	if boundary != nil {
		return nil, boundary
	}
	if res != nil {
		return res, nil
	}
	return nil, err
}

// boundaryCertified reports a closed-loop eigenvalue of the certified
// res within the rounding of a well-conditioned eigenvalue, n·eps·‖A‖_F,
// of the stability boundary, where the certificate shows no stability,
// with boundaryUnstabilizable true both unscaled and state-scaled as
// solve scales (Care with its cost proxy). Wider bands (√eps, 10·n·eps·‖A‖_F)
// rejected oracle-exact X of stable uncontrollable modes at 1 − 1e-9 and
// 1 − 1e-11 that boundaryUnstabilizable's κ-weighted band covers.
func boundaryCertified(continuous bool, A, B, Q, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult) bool {
	if len(res.Eig) == 0 {
		return false
	}
	n, _ := A.Dims()
	band := float64(n) * eps() * denseNorm(A)
	near := false
	for _, l := range res.Eig {
		if continuous && -real(l) <= band || !continuous && 1-cmplx.Abs(l) <= band {
			near = true
			break
		}
	}
	if !near {
		return false
	}
	var base RiccatiOpts
	if opts != nil {
		base = *opts
	}
	base.Workspace, base.NoScaling, base.plainScaling = nil, true, false
	problem, err := newRiccatiProblem(A, B, Q, R, &base)
	if err != nil || !problem.boundaryUnstabilizable(continuous) {
		return false
	}
	base.NoScaling = false
	problem.scale(&base, continuous)
	return problem.boundaryUnstabilizable(continuous)
}

func care(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	problem, err := newRiccatiProblem(A, B, Q, R, opts)
	if err != nil {
		return nil, err
	}
	n, m := problem.n, problem.m
	if err := problem.ws.fits(n, m); err != nil {
		return nil, err
	}
	problem.scale(opts, opts == nil || !opts.plainScaling)
	A, B, Q, S, ws := problem.A, problem.B, problem.Q, problem.S, problem.ws

	// Cholesky factor R
	rChol := ws.rChol[:m*m]
	rRaw := R.RawMatrix()
	copyStrided(rChol, m, rRaw.Data, rRaw.Stride, m, m)
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
	}
	if problem.E != nil {
		X, K, rcnd, accuracy, err := problem.pencilRiccati(true)
		if err != nil {
			return nil, err
		}
		return problem.result(X, K, ws.eig[:n], rcnd, accuracy), nil
	}

	// Working copies of A and Q for cross-term transformation
	aWork := ws.aWork[:n*n]
	aRaw := A.RawMatrix()
	copyStrided(aWork, n, aRaw.Data, aRaw.Stride, n, n)
	qWork := ws.qWork[:n*n]
	qRaw := Q.RawMatrix()
	copyStrided(qWork, n, qRaw.Data, qRaw.Stride, n, n)

	// Compute R⁻¹*B' (m×n): solve R*W = B' via Dpotrs
	rinvBt := ws.rinvBt[:m*n]
	bRaw := B.RawMatrix()
	for i := range n {
		for j := range m {
			rinvBt[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rChol, m, rinvBt, n)

	if S != nil {
		// Abar = A - B*R^-1*S'
		// Z = R^-1*S' (m x n): solve R*Z = S'
		rinvSt := ws.rinvSt[:m*n]
		sRaw := S.RawMatrix()
		for i := range n {
			for j := range m {
				rinvSt[j*n+i] = sRaw.Data[i*sRaw.Stride+j]
			}
		}
		impl.Dpotrs(blas.Upper, m, n, rChol, m, rinvSt, n)

		// aWork -= B * Z
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: aWork, Stride: n})

		// qWork -= S * Z
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			-1, blas64.General{Rows: n, Cols: m, Data: sRaw.Data, Stride: sRaw.Stride},
			blas64.General{Rows: m, Cols: n, Data: rinvSt, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: qWork, Stride: n})
		symmetrize(qWork, n, n)
	}

	// G = B * R⁻¹ * B' (n×n symmetric)
	g := ws.g[:n*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: m, Cols: n, Data: rinvBt, Stride: n},
		0, blas64.General{Rows: n, Cols: n, Data: g, Stride: n})
	symmetrize(g, n, n)

	eig := ws.eig[:n]
	costScaling := problem.costScaling
	X, rcnd, err := problem.hamiltonianSolution(aWork, qWork, g, eig)
	var K *mat.Dense
	if err == nil {
		K, err = problem.acceptRiccati(true, X, eig, rcnd)
	}
	accuracy := 0
	if err != nil {
		first := err
		problem.costScaling, problem.balanced, problem.sigma = costScaling, false, 1
		problem.rescueErr, problem.resolved, problem.candidates = nil, false, problem.candidates[:0]
		if X, K, rcnd, accuracy, err = problem.pencilRiccati(true); err != nil {
			if errors.Is(err, errBoundaryUnstabilizable) {
				return nil, err
			}
			return nil, first
		}
	}
	return problem.result(X, K, eig, rcnd, accuracy), nil
}

// hamiltonianSolution returns the stabilizing solution X of care's
// Hamiltonian [[Ā, −G], [−Q̄, −Āᵀ]] (n×n blocks a, q, g) and rcnd, with the
// closed-loop eigenvalues in eig, cost-scaled and re-solved as Care
// describes.
func (problem *riccatiProblem) hamiltonianSolution(aWork, qWork, g []float64, eig []complex128) (*mat.Dense, float64, error) {
	n, ws := problem.n, problem.ws
	nn := 2 * n
	wr := ws.wr[:nn]
	wi := ws.wi[:nn]
	vs := ws.vs[:nn*nn]
	bwork := ws.bwork[:nn]
	selctg := func(wr, wi float64) bool { return wr < 0 }
	var norms riccatiNorms
	var prev, orig *riccatiPass
	var X *mat.Dense
	var xData []float64
	var rcnd float64
solve:
	for {
		// Hamiltonian of the cost-scaled problem: H = [[A, -σG], [-Q/σ, -A']]
		sigma, isigma := problem.sigma, 1/problem.sigma
		h := ws.h[:nn*nn]
		for i := range n {
			for j := range n {
				h[i*nn+j] = aWork[i*n+j]
				h[i*nn+n+j] = -sigma * g[i*n+j]
				h[(n+i)*nn+j] = -isigma * qWork[i*n+j]
				h[(n+i)*nn+n+j] = -aWork[j*n+i]
			}
		}

		var workQuery [1]float64
		impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
			nn, h, nn, wr, wi, vs, nn, workQuery[:], -1, bwork)
		lwork := int(workQuery[0])
		if len(ws.work) < lwork {
			ws.work = make([]float64, lwork)
		}
		sdim, ok := impl.Dgees(lapack.SchurHess, lapack.SortSelected, selctg,
			nn, h, nn, wr, wi, vs, nn, ws.work, lwork, bwork)
		err := ErrSchurFailed
		if ok {
			err = ErrNoStabilizing
			if sdim == n {
				X, xData, rcnd, err = problem.stabilizingSolution(vs)
			}
		}
		if err == nil && !finiteSlice(xData) {
			err = ErrNoStabilizing
		}
		if err != nil {
			switch {
			case prev != nil:
				X, xData, rcnd = problem.restorePass(prev)
			case orig != nil:
				X, xData, rcnd = problem.restorePass(orig)
				break solve
			case problem.rescueErr != nil:
				return nil, 0, problem.rescueErr
			default:
				problem.rescueErr, problem.costScaling, problem.resolved = err, true, true
				norms = hamiltonianNorms(n, aWork, qWork, g)
				problem.sigma, problem.balanced = norms.firstSigma(), true
				continue
			}
		} else {
			if prev == nil && problem.costScaling && problem.rescueErr == nil && orig == nil {
				norms = hamiltonianNorms(n, aWork, qWork, g)
			}
			if problem.rescaleCost(true, norms, xData, &prev, rcnd) {
				problem.resolved = true
				continue
			}
			if problem.keepBest(true, prev, xData) {
				X, xData, rcnd = problem.restorePass(prev)
			}
		}
		res, scale := hamiltonianResidual(n, aWork, qWork, g, xData, ws.h[:nn*nn])
		if problem.rescueErr != nil {
			if !(res <= rescueTol*scale) {
				return nil, 0, problem.rescueErr
			}
			break
		}
		if orig != nil {
			if !(res <= rescueTol*scale && res < orig.res) {
				X, xData, rcnd = problem.restorePass(orig)
			}
			break
		}
		if res <= rescueTol*scale {
			break
		}
		problem.savePass(&orig, xData, rcnd, res)
		prev, problem.costScaling, problem.resolved = nil, true, true
		norms = hamiltonianNorms(n, aWork, qWork, g)
		problem.sigma, problem.balanced = norms.firstSigma(), true
	}

	for i := range n {
		eig[i] = complex(wr[i], wi[i])
	}
	return X, rcnd, nil
}

// hamiltonianResidual returns ‖ĀᵀX + XĀ − XGX + Q̄‖_F of care's
// Hamiltonian blocks and the sum of the Frobenius norms of its terms, using
// work (≥ 3n²) as scratch.
func hamiltonianResidual(n int, a, q, g, x, work []float64) (res, scale float64) {
	gen := func(d []float64) blas64.General { return blas64.General{Rows: n, Cols: n, Data: d, Stride: n} }
	xa, gx, xgx := work[:n*n], work[n*n:2*n*n], work[2*n*n:3*n*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(x), gen(a), 0, gen(xa))
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(g), gen(x), 0, gen(gx))
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(x), gen(gx), 0, gen(xgx))
	scale = 2*impl.Dlange(lapack.Frobenius, n, n, xa, n, nil) +
		impl.Dlange(lapack.Frobenius, n, n, xgx, n, nil) +
		impl.Dlange(lapack.Frobenius, n, n, q, n, nil)
	for i := range n {
		for j := range n {
			gx[i*n+j] = xa[i*n+j] + xa[j*n+i] - xgx[i*n+j] + q[i*n+j]
		}
	}
	return impl.Dlange(lapack.Frobenius, n, n, gx, n, nil), scale
}

// hamiltonianNorms returns the riccatiNorms of care's Hamiltonian blocks
// Ā, Q̄ and G (n×n, row-major).
func hamiltonianNorms(n int, a, q, g []float64) riccatiNorms {
	return riccatiNorms{
		a:     impl.Dlange(lapack.Frobenius, n, n, a, n, nil),
		q:     impl.Dlange(lapack.Frobenius, n, n, q, n, nil),
		g:     impl.Dlange(lapack.Frobenius, n, n, g, n, nil),
		kappa: 1,
	}
}

// Dare solves the discrete algebraic Riccati equation:
//
//	A'XA - X - (A'XB+S)*(R+B'XB)⁻¹*(B'XA+S') + Q = 0
//
// When opts is nil or opts.S is nil, the cross-term is zero:
//
//	A'XA - X - A'XB*(R+B'XB)⁻¹*B'XA + Q = 0
//
// A is n×n, B is n×m, Q is n×n symmetric, R is m×m symmetric positive definite.
//
// With opts.E, Dare solves the generalized equation of MATLAB idare
// (https://www.mathworks.com/help/control/ref/idare.html)
//
//	A'XA - E'XE - (A'XB+S)*(R+B'XB)⁻¹*(B'XA+S') + Q = 0
//
// with the same K formula, and Eig holds the generalized eigenvalues of
// (A-BK, E). E'XE solves the Dare of the explicit model (E⁻¹A, E⁻¹B). A
// singular or near-singular E is handled as in Care. Nil or NaN/Inf
// arguments return ErrInvalidArgument. Where E'XE of the certified X
// cancels beyond nearSingularCancelLimit, X ~ 1/σ_min(E)² and its float64
// rounding moves K (a Kalman L off by 1.4e-5 at E's last pivot 1e-5), or
// the Newton step moves E'XE beyond 1e-10, K and the E'XE of Dlqr are, as
// those idare takes from the deflating subspace, the double-double gain
// and E'XE of X refined by Hewer's method in double-double; X keeps its
// certificate. Should that not converge, X is certified only while the
// rounding of X moves K, and the Newton step K, within 1e-8.
//
// Dare always deflates the extended pencil, as MATLAB idare does, so A may
// be singular or have near-zero eigenvalues. A solution with R+B'XB not
// positive definite returns ErrNoStabilizing.
//
// Dare checks the pencil's X by a Newton (Hewer) step from its gain, a
// Stein solve (with E, for E⁻¹(A − BK) and EᵀΔE) of the double-double
// Riccati residual that is blind
// to the rounding of (R+B'XB)⁻¹ that hides lost digits from the residual
// itself. An X the step moves by more than 1e-8 relative is refined by
// Newton's method until a step is within 1e-10; if that does not
// converge, the other cost scales' solutions are tried. A certified X has
// Accuracy 0; with none certified Dare returns the stabilizing solution of
// smallest first step with Accuracy 1 (MATLAB idare Report 1, as Care,
// including its PBH check of boundary modes, and its retry in the other
// NoScaling mode). With no stabilizing pencil solution in either mode and
// no unstabilizable boundary mode, Dare runs Hewer's
// iteration from a pole-placement gain and returns its X only when the
// same Newton check certifies it (Accuracy 0; Rcnd then that of span[I; X/σ] at σ ≈ ‖X‖),
// else ErrNoStabilizing: under cheap control of a strongly unstable plant
// no cost scale of the pencil may resolve its split. It runs them too
// over an Accuracy 1 pencil solution, which a certified placed X
// replaces: under cheap control with m = n the pencil's X can be off by
// 4e-3 where Hewer from a placed gain certifies it to 1e-10. Not with B
// of condition above rotationCond in the given inputs, where Newton's
// R + BᵀXB has lost its smallest eigenvalue and certified placed X off by
// 2e-6 over pencil X accurate to 1e-7.
//
// Dare scales the states and the cost as Care does (MATLAB idare), its
// state scaling balanced with B (the cost proxy of Care brought no
// certification on cheap-control sweeps and cost 10-20% time);
// opts.NoScaling disables it. An unscaled pencil that fails QZ or the
// stable split is solved again cost-scaled before an error is returned.
//
// With m ≥ 2 and B of condition above rotationCond, Dare solves in the
// inputs w = Vᵀu of B's right singular vectors V (reduceInputs): X, Eig
// and Rcnd are unchanged, and R + (BV)ᵀX(BV) keeps on its diagonal the
// smallest eigenvalue that R + BᵀXB loses to rounding under cheap
// control, where Newton certified X off by 5e-7 (or the pencil's X by
// 1e-5) on well-conditioned problems with nearly collinear inputs. With
// a cross term S and rank(B) = r < m (always when m > n), Dare instead
// eliminates the inputs B does not see, as their optimal value
// w₂ = −R₂₂⁻¹(R₂₁w₁ + S₂ᵀx) is explicit, and solves the problem of r
// inputs: R + BᵀXB loses their R₂₂, and with it a cost xᵀS₂R₂₂⁻¹S₂ᵀx
// comparable to Q, so Newton certified a wrong X ≈ Q. Its X is returned
// once the same Newton check certifies it in the m inputs w, with K
// and Eig from there, BV₂ = 0 there when B's null space is exact
// (duplicated inputs, or m > n): the rounding of V in BV₂ ≈ 1e-17 moved
// their X by 1e-5. Either K is returned only with A − BK in the given
// inputs stable; should that, or the problem in w, fail, the given one
// is solved, and its X keeps Accuracy 0 only once the Newton check in w
// certifies it, from its X or else the Hewer step X₁ of its K (then with
// X₁, its K and Eig): there Newton certified X off by 1e-3 to 0.16, and
// the cost of K equal to X shows only that, not that K is the gain of X.
// Without S, rank(B) < m loses only K's part on B's null space, fixed by
// R alone: once ‖B‖²‖X‖ exceeds ‖R‖ (cheap control), where the rounding
// of BᵀXB swamps R there, the X of the given problem is certified again
// in the inputs w and K taken from there (VᵀRV costs K only eps·cond(R)),
// and a given problem that fails or is poor is solved in r inputs; either
// K keeps Accuracy 0 only as the double-double gain of its X (gainMiss),
// else the one nearest it is returned with Accuracy 1: with a
// rank-deficient Q, K was off by O(1) on B's range with X exact. Under
// cheap control the same holds for the K of the problem with a cross term
// solved in r inputs, and for that of a B without reduced inputs where
// BᵀXB may round R away (rLost), there with Accuracy 1 otherwise: with a
// full-rank square B and a rank-deficient Q, BᵀXB is singular and the
// float64 gain lost R against it, K off by O(1) with X exact to 1e-14.
// That K must also be the gain of X one certified Newton step on, whose
// error the gain may amplify by 1e9 (K off by 9e-3 with X off by 7e-14):
// a K that is not is replaced, before Accuracy 1, by the double-double
// gain of X taken one Newton step on, once that X is certified and its
// A − BK stable (refined). Of an X whose Newton step is within eps, a K
// off its double-double gain by more than 1e-6 is replaced by that gain
// (A − BK stable), X being its own Newton iterate: the pencil K was off
// by 7.3e-6 with X exact and its gain within 3.4e-9. K and EᵀXE that a
// descriptor refinement in double-double formed (refineDescriptorDare)
// are kept: the gain of the float64 X lost cond(K)·eps ≈ 1e-5 against
// them. Their X keeps Accuracy 0 only within newtonTol of the
// double-double iterate they came from (3e-8 off it with E's smallest
// singular value 1.2e-6), in both its Frobenius and 1-norm (X off by
// 1.25e-8 with 9.97e-9 in the first). With reduced inputs, under cheap
// control, K is neither the gain of the rounded X nor kept from a
// refinement; a K off the gain of X by more than 1e-6 is replaced by the
// gain of X taken on by Hewer steps in extended precision (gainOfX): with
// X exact to 1e-16 the pencil K was 6.1e-6 off the 320-bit gain and the
// gain of the rounded X 1.4e-6, that one 1e-10. With an ill-conditioned
// E, a K that extended-precision Hewer gain contradicts beyond its band
// is returned with Accuracy 1: the float64 X wandered 1e-10 off, its gain
// 3.6e-4 off the 320-bit one (band 1.3e-9).
// A descriptor Dare left with no stabilizing solution whose E is
// ill-conditioned but nonsingular is solved once more on its explicit
// model in extended precision (nearSingularDare).
func Dare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	if err := riccatiArgs("Dare", A, B, Q, R, opts); err != nil {
		return nil, err
	}
	res, err := dareInputs(A, B, Q, R, opts)
	if r, ok := nearSingularDare(A, B, Q, R, opts, err); ok {
		res, err = r, nil
	}
	res, err = nearSingularEGate("Dare", false, A, B, R, opts, res, err)
	if err == nil && res.Accuracy == 0 && res.xe != nil && res.xeX > newtonTol {
		res.Accuracy = 1
	}
	return res, err
}

// dareInputs is Dare after its argument check.
func dareInputs(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	red, ill := reduceInputs(A, B, Q, R, opts)
	if red == nil {
		res, err := dareSolve(A, B, Q, R, opts, !ill)
		n, m := B.Dims()
		if err != nil || res.Accuracy != 0 || n == 0 || m == 0 || !rLost(B, R, res.X) {
			return res, err
		}
		if r, _, ok := gainOfX(A, B, Q, R, opts, nil, res); ok {
			return r, nil
		}
		res.Accuracy = 1
		return res, nil
	}
	var E *mat.Dense
	if opts != nil {
		E = opts.E
	}
	if red.s == nil && !red.rotated {
		res, err := dareSolve(A, B, Q, R, opts, !ill)
		var poor *RiccatiResult
		poorMiss := math.Inf(1)
		gainOf := func(c *RiccatiResult) (*RiccatiResult, bool) {
			r, miss, ok := gainOfX(A, B, Q, R, opts, red, c)
			if !ok && (poor == nil || miss < poorMiss) {
				poor, poorMiss = c, miss
				if r != nil {
					poor = r
				}
			}
			return r, ok
		}
		if err == nil && res.Accuracy == 0 {
			if !(red.sigma*red.sigma*mat.Norm(res.X, 1) > mat.Norm(R, 1)) {
				return res, nil
			}
			if c, ok := red.certify(A, Q, opts, res, nil); ok && closedLoopStable(A, B, c.K, E, c.Eig) {
				if c, ok = gainOf(c); ok {
					return c, nil
				}
			}
		}
		o := opts
		if res != nil && opts != nil && opts.Workspace != nil {
			c := *opts
			c.Workspace = nil
			o = &c
		}
		if r := red.solve(A, B, Q, o); r != nil {
			if r, ok := gainOf(r); ok {
				return r, nil
			}
		}
		if poor != nil {
			poor.Accuracy = 1
			return poor, nil
		}
		if err == nil {
			res.Accuracy = 1
		}
		return res, err
	}
	// Rotated inputs are the given Dare under an orthogonal V: BV, VᵀRV
	// and SV round within the K band, so a K refined there is terminal.
	// The gain of the rounded X was up to 2.5 off the 320-bit gain where
	// that K was within 4x its band; held to it, K became 3.2e-6 off
	// (band 4.4e-13). An unrefined rotated K is held to it: the pencil K
	// was 0.69 off the 320-bit gain (band 8e-11) with X within 1.5e-11.
	gainOf := func(c *RiccatiResult) *RiccatiResult {
		if red.rotated && c.xe == nil || !red.rotated && red.sigma*red.sigma*mat.Norm(c.X, 1) > mat.Norm(R, 1) {
			r, _, ok := gainOfX(A, B, Q, R, opts, red, c)
			if ok {
				return r
			}
			if r != nil {
				c = r
			}
			c.Accuracy = 1
		}
		return c
	}
	if res := red.solve(A, B, Q, opts); res != nil {
		return gainOf(res), nil
	}
	res, err := dareSolve(A, B, Q, R, opts, !ill)
	if err != nil || res.Accuracy != 0 {
		return res, err
	}
	if c, ok := red.certify(A, Q, opts, res, res.K); ok && closedLoopStable(A, B, c.K, E, c.Eig) {
		return gainOf(c), nil
	}
	res.Accuracy = 1
	return res, nil
}

// solve returns the first solution of the Dare of red whose gain in the
// inputs u stabilizes A − BK (closedLoopStable), or nil: that of dareSolve
// and, should its K fail there, those of dare in the other scaling mode
// and of placedDare in both (as dareSolve, none boundaryCertified).
// Under cheap control A − BK can be so
// sensitive to K that the K of X accurate to 1e-14 and that of 3e-13 lie
// on either side of the unit circle, while each mode rounds it
// differently.
func (red *reducedInputs) solve(A, B, Q *mat.Dense, opts *RiccatiOpts) *RiccatiResult {
	var o RiccatiOpts
	if opts != nil {
		o = *opts
	}
	o.S, o.reducedQ = red.s, !red.rotated
	accept := func(res *RiccatiResult, ws *RiccatiWorkspace) *RiccatiResult {
		ok := true
		if red.rotated {
			res.K = red.gain(res.K, ws)
		} else {
			res, ok = red.certify(A, Q, opts, res, nil)
		}
		if ok && closedLoopStable(A, B, res.K, o.E, res.Eig) {
			return res
		}
		return nil
	}
	res, err := dareSolve(A, red.b, red.q, red.r, &o, true)
	if err != nil {
		return nil
	}
	if r := accept(res, o.Workspace); r != nil {
		return r
	}
	o.Workspace = nil
	flipped := o
	flipped.NoScaling = !o.NoScaling
	if r, err := dare(A, red.b, red.q, red.r, &flipped); err == nil && r.Accuracy == 0 && !boundaryCertified(false, A, red.b, red.q, red.r, &o, r) {
		if r = accept(r, nil); r != nil {
			return r
		}
	}
	for _, p := range []*RiccatiOpts{&o, &flipped} {
		if r, err := placedDare(A, red.b, red.q, red.r, p); err == nil && !boundaryCertified(false, A, red.b, red.q, red.r, &o, r) {
			if r = accept(r, nil); r != nil {
				return r
			}
		}
	}
	return nil
}

// closedLoopStable reports whether the eigenvalues of A − BK (with E, of
// E⁻¹(A − BK)), A − BK formed in double-double, lie inside the unit
// circle, and writes them to eig when they do. Dare checks the K of the
// inputs u it returns: under cheap control ‖K‖ ≈ 1e13 rounds by 1e-3 in
// every direction of u, which BV₁ = O(1) can carry to an unstable A − BK
// although that of the inputs w is stable.
func closedLoopStable(A, B, K, E *mat.Dense, eig []complex128) bool {
	n, m := B.Dims()
	aRaw, bRaw, kRaw := A.RawMatrix(), B.RawMatrix(), K.RawMatrix()
	hi, lo := make([]float64, n*n), make([]float64, n*n)
	copyStrided(hi, n, aRaw.Data, aRaw.Stride, n, n)
	ddMulAcc(n, n, m, -1, bRaw.Data, bRaw.Stride, 1, kRaw.Data, kRaw.Stride, 1, hi, lo, n)
	for i := range hi {
		hi[i] += lo[i]
	}
	acl := mat.NewDense(n, n, hi)
	if E != nil {
		var lu mat.LU
		lu.Factorize(E)
		var f mat.Dense
		if lu.SolveTo(&f, false, acl) != nil {
			return false
		}
		acl = &f
	}
	var ed mat.Eigen
	if !ed.Factorize(acl, mat.EigenNone) {
		return false
	}
	vals := ed.Values(nil)
	for _, v := range vals {
		if !(cmplx.Abs(v) < 1) {
			return false
		}
	}
	copy(eig, vals)
	return true
}

// certify returns the X of res, the solution of the reduced Dare, once
// verifyRiccati certifies it as that of the given one in the inputs w,
// whose R + (BV)ᵀX(BV) keeps R₂₂ on its diagonal, with the gain and
// Eig of A − BK there: BV₂ is zero only to rounding, and under cheap
// control the K₂ ≈ R₂₂⁻¹S₂ᵀ it multiplies can move X, or destabilize
// A − BK, well beyond the rounding of B. With an exact null space of B
// (nullExact) BV₂ is 0 there: its rounding, that of V, moved the exact
// X of duplicated inputs by 1e-5.
func (red *reducedInputs) certify(A, Q *mat.Dense, opts *RiccatiOpts, res *RiccatiResult, k0 *mat.Dense) (*RiccatiResult, bool) {
	var kw func() *mat.Dense
	if k0 != nil {
		kw = func() *mat.Dense {
			var k mat.Dense
			k.Mul(red.v.T(), k0)
			return &k
		}
	}
	out, ok := certifyIn(A, red.bw, Q, red.rw, red.sw, opts, res, kw, false)
	if ok {
		out.K = red.gain(out.K, nil)
	}
	return out, ok
}

// refined returns res with its X one Newton step on, certified as by
// certify (in the inputs w with reduced inputs, else as given), and K the
// ddGain of that X (as gainMiss forms it), once A − BK is stable and
// gainMiss accepts it. Dare tries it on a K gainMiss rejects: the gain of
// a cheap-control Dare can amplify the error of a certified X by 1e9, and
// an X off by 5e-14 within newtonTol gave a K off by 2e-4 to 9e-3, while
// X one step on was the rounded 320-bit X and its ddGain within K's own
// eps-sensitivity. A K and EᵀXE the certification of the given inputs
// refined in double-double (c.xe) are kept and returned without gainMiss:
// the ddGain of the rounded X was 1.9e-6 off the 320-bit gain where the
// refined K matched it.
func refined(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) (*RiccatiResult, gainDist, bool) {
	var c *RiccatiResult
	ok := false
	switch {
	case red != nil && red.exact:
		if c, ok = certifyIn(A, red.bw, Q, red.rw, red.sw, opts, res, nil, true); ok {
			var kw *mat.Dense
			if kw, _, ok = ddGain(A, red.bw, red.rw, red.sw, c.X); ok {
				c.K = red.gain(kw, nil)
			}
		}
	case red != nil:
		if c, ok = certifyIn(A, red.bw, Q, red.rw, red.sw, opts, res, nil, true); ok {
			c.K, _, ok = ddGain(A, B, R, crossTerm(opts), c.X)
		}
	default:
		if c, ok = certifyIn(A, B, Q, R, crossTerm(opts), opts, res, nil, true); ok && c.xe == nil {
			c.K, _, ok = ddGain(A, B, R, crossTerm(opts), c.X)
		}
	}
	var E *mat.Dense
	if opts != nil {
		E = opts.E
	}
	if !ok || !closedLoopStable(A, B, c.K, E, c.Eig) {
		return nil, gainDist{}, false
	}
	if red == nil && c.xe != nil {
		return c, gainDist{}, true
	}
	return c, gainMissIn(A, B, R, opts, red, c), true
}

// gainOfX returns res once its K is the gain of its X to the tolerance of
// gainMissIn with the floor gainTol, else res refined (refined) once its
// K is; failing both, the same with the floor gainFloor, as Dare kept
// before. ok is false when none is, miss then the distance of res in
// units of the tolerance with gainFloor. A K off the gain of an X whose
// Newton step is within eps by more than gainTol is first replaced by
// that gain (gainSubstituted): X is then its own Newton iterate, and the
// pencil K was off by 7.3e-6, 43 times its eps-sensitivity, with X exact
// and its gain within 3.4e-9, inside a tolerance of 1e-5 set by the move
// of that gain under X∘(1 + eps). A K refined in double-double (res.xe)
// is returned as it is: measured against the gain of the rounded X it
// was off by 7.7e-6 while within 1e-17 of the 320-bit gain. So is one
// refined returns, its X taken one Newton step on (steppedX). Both hold
// only for the given inputs (red nil): with reduced inputs K is measured
// by gainMissIn and taken as Dare kept before, a substituted K and an X
// its certification did not step being 4.7e-6 and 1.25e-8 off the
// 320-bit solution at Accuracy 0. There a K off the gain of X by more
// than gainFloor is instead replaced by the gain of the
// extended-precision Hewer iterate of X (hewerGainSubstituted, Y6BJEX
// am.3), which gainMissIn must accept; X is kept. With rotated inputs a
// refined X whose K gainMissIn declines is refined once more: one Newton
// step took X from 1e-10 to 5e-14 off, its gain 2.4e-7 off the 320-bit
// one (band 1.9e-9), and the second to 1.8e-17 and 8e-11. With reduced
// inputs and a descriptor E, a K the Hewer gain of its X contradicts
// (hewerGainRef) is not returned with ok; then r is, of res and the X
// refined, that whose K is nearest the Hewer gain, for Accuracy
// 1: there the gain of the float64 X was 3.6e-4 off the 320-bit gain at
// Accuracy 0 (band 1.3e-9).
func gainOfX(A, B, Q, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) (r *RiccatiResult, miss float64, ok bool) {
	if red == nil && res.xe != nil {
		return steppedX(A, B, Q, R, opts, res), 0, true
	}
	d := gainMissIn(A, B, R, opts, red, res)
	ref := newHewerGainRef(A, B, Q, R, opts, red, res)
	contra := false
	contradicted := func(r *RiccatiResult, move, tol float64) func() bool {
		known, v := false, false
		return func() bool {
			if !known {
				known, v = true, ref.contradicts(r, move, tol)
				contra = contra || v
			}
			return v
		}
	}
	resOff := contradicted(res, d.move, d.tol)
	if d.units(gainTol) <= 1 {
		if red == nil && d.d > gainFloor && res.x1 == nil && res.step <= eps() {
			if r, ok := gainSubstituted(A, B, R, opts, res); ok {
				return r, 0, true
			}
		}
		if red != nil && d.d > gainFloor {
			if r, ok := hewerGainSubstituted(A, B, Q, R, opts, red, res); ok {
				return r, 0, true
			}
		}
		if !resOff() {
			return res, 0, true
		}
	}
	c, cd, stable := refined(A, B, Q, R, opts, red, res)
	if stable && red != nil && red.rotated && !(cd.units(gainTol) <= 1) {
		if c2, cd2, ok := refined(A, B, Q, R, opts, red, c); ok && cd2.units(gainTol) <= 1 {
			c, cd = c2, cd2
		}
	}
	var cOff func() bool
	if stable {
		cOff = contradicted(c, gainMove(d.k, c.K), cd.tol)
	}
	if stable && (red == nil && c.xe != nil || cd.units(gainTol) <= 1 && !cOff()) {
		return c, 0, true
	}
	if d.units(gainFloor) <= 1 && !resOff() {
		return res, 0, true
	}
	if stable && cd.units(gainFloor) <= 1 && !cOff() {
		return c, 0, true
	}
	if contra {
		if h := ref.substituted(); h != nil {
			return h, 0, false
		}
	}
	return nil, d.units(gainFloor), false
}

// steppedX returns res with X one Newton step on (certifyIn) once that
// X is certified: with its own K and EᵀXE when the certification refines
// them in double-double and A − BK is stable, else with K, Eig and EᵀXE
// of res (terminal, Y6BJEX am.3) when its Newton step is smaller; else
// res. The float64 X such a refinement came with, its Newton step
// 9.7e-9, was 1e-8 off the 320-bit X and the X one step on within 1e-17.
// Of the given inputs only (gainOfX).
func steppedX(A, B, Q, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult) *RiccatiResult {
	if !(res.step > eps()) {
		return res
	}
	c, ok := certifyIn(A, B, Q, R, crossTerm(opts), opts, res, nil, true)
	if !ok {
		return res
	}
	if c.xe != nil {
		var E *mat.Dense
		if opts != nil {
			E = opts.E
		}
		if closedLoopStable(A, B, c.K, E, c.Eig) {
			return c
		}
		return res
	}
	if !(c.step < res.step) {
		return res
	}
	r := *res
	r.X, r.step = c.X, c.step
	return &r
}

// gainSubstituted returns res with K the ddGain of its X and Eig that of
// A − BK, once A − BK is stable.
func gainSubstituted(A, B, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult) (*RiccatiResult, bool) {
	K, _, ok := ddGain(A, B, R, crossTerm(opts), res.X)
	if !ok {
		return nil, false
	}
	var E *mat.Dense
	if opts != nil {
		E = opts.E
	}
	eig := make([]complex128, len(res.Eig))
	if !closedLoopStable(A, B, K, E, eig) {
		return nil, false
	}
	c := *res
	c.K, c.Eig = K, eig
	return &c, true
}

// gainIn returns the gain gainMissIn measures K against: the ddGain of
// the inputs of red when its null space of B is exact, else of the given
// inputs.
func gainIn(A, B, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs) func(X *mat.Dense) (*mat.Dense, float64, bool) {
	if red != nil && red.exact {
		return func(X *mat.Dense) (*mat.Dense, float64, bool) {
			kw, cond, ok := ddGain(A, red.bw, red.rw, red.sw, X)
			if !ok {
				return nil, 0, false
			}
			return red.gain(kw, nil), cond, true
		}
	}
	return func(X *mat.Dense) (*mat.Dense, float64, bool) {
		return ddGain(A, B, R, crossTerm(opts), X)
	}
}

// gainMissIn returns the gainMiss of res with reduced inputs red, else
// its gainMissOf from the ddGain of the given inputs.
func gainMissIn(A, B, R *mat.Dense, opts *RiccatiOpts, red *reducedInputs, res *RiccatiResult) gainDist {
	if red != nil {
		return red.gainMiss(A, B, R, opts, res)
	}
	return gainMissOf(A.RawMatrix().Rows, res, func(X *mat.Dense) (*mat.Dense, float64, bool) {
		return ddGain(A, B, R, crossTerm(opts), X)
	})
}

// certifyIn returns the X of res once verifyRiccati certifies it as the
// solution of the Dare (A, B, Q, R, S), with its gain and Eig; with kw,
// should X fail, the Hewer step X₁ of the gain kw() is certified instead,
// and with refine X is first taken one Newton step on.
func certifyIn(A, B, Q, R, S *mat.Dense, opts *RiccatiOpts, res *RiccatiResult, kw func() *mat.Dense, refine bool) (*RiccatiResult, bool) {
	var o RiccatiOpts
	if opts != nil {
		o = *opts
	}
	o.S, o.Workspace = S, nil
	problem, err := newRiccatiProblem(A, B, Q, R, &o)
	if err != nil {
		return nil, false
	}
	problem.scale(&o, false)
	n, m, t := problem.n, problem.m, problem.t
	X := mat.NewDense(n, n, problem.ws.xData[:n*n])
	setX := func() {
		xr := res.X.RawMatrix()
		for i := range n {
			for j := range n {
				v := xr.Data[i*xr.Stride+j]
				if t != nil {
					v /= t[i] * t[j]
				}
				X.Set(i, j, v)
			}
		}
	}
	setX()
	if refine {
		K, err := problem.gain(false, X.RawMatrix().Data)
		if err != nil {
			return nil, false
		}
		x1, err := problem.newtonCost(false, K, X.RawMatrix().Data)
		if err != nil {
			return nil, false
		}
		copy(X.RawMatrix().Data, x1)
	}
	eig := problem.ws.eig[:n]
	K, err := problem.acceptRiccati(false, X, eig, res.Rcnd)
	if err != nil && kw != nil {
		setX()
		k := kw()
		if t != nil {
			kr := k.RawMatrix()
			for i := range m {
				for j := range n {
					kr.Data[i*kr.Stride+j] /= t[j]
				}
			}
		}
		var x1 []float64
		if x1, err = problem.newtonCost(false, k, X.RawMatrix().Data); err == nil {
			copy(X.RawMatrix().Data, x1)
			K, err = problem.acceptRiccati(false, X, eig, res.Rcnd)
		}
	}
	if err != nil {
		return nil, false
	}
	return problem.result(X, K, eig, res.Rcnd, 0), true
}

// dareSolve is Dare without its input reduction; swapPoor lets placedDare
// replace an Accuracy 1 pencil X, which Dare declines on a B of condition
// above rotationCond in the inputs u, where Newton's R + BᵀXB has lost
// its smallest eigenvalue and certifies a wrong X.
func dareSolve(A, B, Q, R *mat.Dense, opts *RiccatiOpts, swapPoor bool) (*RiccatiResult, error) {
	res, err := otherScaling(dare, false, A, B, Q, R, opts)
	poor := swapPoor && err == nil && res.Accuracy != 0
	if poor || (errors.Is(err, ErrNoStabilizing) || errors.Is(err, ErrSchurFailed)) && !errors.Is(err, errBoundaryUnstabilizable) {
		var base RiccatiOpts
		if opts != nil {
			base = *opts
		}
		if poor {
			base.Workspace = nil
		}
		flipped := base
		flipped.NoScaling = !base.NoScaling
		for _, o := range []RiccatiOpts{base, flipped} {
			if r, e := placedDare(A, B, Q, R, &o); e == nil && !boundaryCertified(false, A, B, Q, R, opts, r) {
				return r, nil
			}
		}
	}
	if err != nil {
		return nil, fmt.Errorf("Dare: %w", err)
	}
	return res, nil
}

func dare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	problem, err := newRiccatiProblem(A, B, Q, R, opts)
	if err != nil {
		return nil, err
	}
	n, m := problem.n, problem.m
	if err := problem.ws.fits(n, m); err != nil {
		return nil, err
	}
	problem.scale(opts, false)
	ws := problem.ws

	// Cholesky factor R
	rChol := ws.rChol[:m*m]
	rRaw := R.RawMatrix()
	copyStrided(rChol, m, rRaw.Data, rRaw.Stride, m, m)
	if !impl.Dpotrf(blas.Upper, m, rChol, m) {
		return nil, ErrSingularR
	}

	X, K, rcnd, accuracy, err := problem.pencilRiccati(false)
	if err != nil {
		return nil, err
	}
	return problem.result(X, K, ws.eig[:n], rcnd, accuracy), nil
}

// reducedInputs is the Dare of reduceInputs in the inputs w = Vᵀu, V = [V₁ V₂]
// the right singular vectors of B with BV₂ = 0 to rounding: B₁ = BV₁ (b),
// Q − S₂R₂₂⁻¹S₂ᵀ (q), R₁₁ − R₁₂R₂₂⁻¹R₂₁ (r) and S₁ − S₂R₂₂⁻¹R₂₁ (s) for
// VᵀRV = [R₁₁ R₁₂; R₂₁ R₂₂] and SV = [S₁ S₂]. With V₂ empty (rotated) it
// is the given Dare in the inputs w: BV, Q, VᵀRV and SV.
type reducedInputs struct {
	b, q, r, s *mat.Dense
	// bw, rw and sw are BV, VᵀRV and SV (nil without S).
	bw, rw, sw *mat.Dense
	v          *mat.Dense
	rotated    bool
	// exact reports a null space of B spanned exactly by V₂ (nullExact),
	// BV₂ then set to 0.
	exact bool
	// sigma is B's largest singular value.
	sigma float64
}

// rotationCond is the condition of B above which Dare solves in the
// inputs of B's right singular vectors: R + BᵀXB then loses its smallest
// eigenvalue, ≈ R₂₂ + σ²_min·xᵀXx, to the rounding eps·σ²_max·‖X‖ of its
// entries, a relative error of up to eps·cond(B)², while (BV)ᵀX(BV) keeps
// it on its diagonal.
const rotationCond = 1e3

// reduceInputs returns the reducedInputs of a Dare with rank(B) < m (s nil
// and q = Q without S), rank as placedHewer counts it, or, with
// rank(B) = m ≥ 2 and cond(B) above rotationCond, the rotated one; nil when there are none
// or R₂₂ is not positive definite. ill reports m ≥ 2 and cond(B) above
// rotationCond.
func reduceInputs(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (red *reducedInputs, ill bool) {
	n, m := B.Dims()
	if m < 2 || n == 0 {
		return nil, ill
	}
	var S *mat.Dense
	if opts != nil {
		S = opts.S
		if opts.Workspace != nil && opts.Workspace.fits(n, m) != nil {
			return nil, ill
		}
	}
	if an, _ := A.Dims(); an != n {
		return nil, ill
	}
	if S != nil {
		if sr, sc := S.Dims(); sr != n || sc != m {
			return nil, ill
		}
	}
	if rr, rc := R.Dims(); rr != m || rc != m {
		return nil, ill
	}
	if qr, qc := Q.Dims(); qr != n || qc != n {
		return nil, ill
	}
	var svd mat.SVD
	if !svd.Factorize(B, mat.SVDFullV) {
		return nil, ill
	}
	sv := svd.Values(nil)
	r := 0
	for _, s := range sv {
		if s > float64(max(n, m))*eps()*sv[0] {
			r++
		}
	}
	ill = len(sv) < m || !(sv[m-1] >= sv[0]/rotationCond)
	if r == 0 || !ill {
		return nil, ill
	}
	red = &reducedInputs{v: &mat.Dense{}, sigma: sv[0]}
	svd.VTo(red.v)
	var rv, rt mat.Dense
	rv.Mul(R, red.v)
	rt.Mul(red.v.T(), &rv)
	symmetrize(rt.RawMatrix().Data, m, rt.RawMatrix().Stride)
	red.bw, red.rw = &mat.Dense{}, &rt
	red.bw.Mul(B, red.v)
	var sv2 mat.Dense
	if S != nil {
		sv2.Mul(S, red.v)
		red.sw = &sv2
	}
	if r == m {
		red.b, red.q, red.r, red.s, red.rotated = red.bw, Q, red.rw, red.sw, true
		return red, ill
	}
	red.b = mat.DenseCopyOf(red.bw.Slice(0, n, 0, r))
	if red.exact = nullExact(B, red.bw, red.v, sv, r); red.exact {
		bwr := red.bw.RawMatrix()
		for i := range n {
			clear(bwr.Data[i*bwr.Stride+r : i*bwr.Stride+m])
		}
	}
	r22 := mat.NewSymDense(m-r, nil)
	for i := range m - r {
		for j := i; j < m-r; j++ {
			r22.SetSym(i, j, rt.At(r+i, r+j))
		}
	}
	var r22c mat.Cholesky
	if !r22c.Factorize(r22) {
		return nil, ill
	}
	var r21 mat.Dense
	if r22c.SolveTo(&r21, rt.Slice(r, m, 0, r)) != nil {
		return nil, ill
	}
	var t mat.Dense
	t.Mul(rt.Slice(0, r, r, m), &r21)
	red.r = &mat.Dense{}
	red.r.Sub(rt.Slice(0, r, 0, r), &t)
	symmetrize(red.r.RawMatrix().Data, r, r)
	if S == nil {
		red.q = Q
		return red, ill
	}
	s2 := sv2.Slice(0, n, r, m)
	var s2t mat.Dense
	if r22c.SolveTo(&s2t, s2.T()) != nil {
		return nil, ill
	}
	t.Reset()
	t.Mul(s2, &r21)
	red.s = &mat.Dense{}
	red.s.Sub(sv2.Slice(0, n, 0, r), &t)
	t.Reset()
	t.Mul(s2, &s2t)
	red.q = &mat.Dense{}
	red.q.Sub(Q, &t)
	symmetrize(red.q.RawMatrix().Data, n, n)
	return red, ill
}

// rLost reports whether the rounding eps·‖B‖²‖X‖ of BᵀXB, bounded by
// eps·‖B‖₁‖B‖∞‖X‖₁, exceeds 1e-9 of the smallest eigenvalue of R, bounded
// by 1/‖R⁻¹‖₁ (or, without forming R⁻¹, by its Gershgorin bound g as
// √m/g), so that the float64 gain may lose R in R + BᵀXB: the K that
// Dare certified off by 1e-5 to 5.6 with X exact had it above 9e-6.
func rLost(B, R, X *mat.Dense) bool {
	b := eps() * mat.Norm(B, 1) * mat.Norm(B, math.Inf(1)) * mat.Norm(X, 1)
	m, _ := R.Dims()
	rr := R.RawMatrix()
	g := math.Inf(1)
	for i := range m {
		row := rr.Data[i*rr.Stride : i*rr.Stride+m]
		d := row[i]
		for j, v := range row {
			if j != i {
				d -= math.Abs(v)
			}
		}
		g = min(g, d)
	}
	if g > 0 && b*math.Sqrt(float64(m)) <= 1e-9*g {
		return false
	}
	var ri mat.Dense
	if ri.Inverse(R) != nil {
		return true
	}
	return b*mat.Norm(&ri, 1) > 1e-9
}

// gainTol is the relative distance from the gain of its X that a K Dare
// returns at Accuracy 0 may have, unless the gain itself moves more under
// eps perturbations of X or is resolved only more coarsely in
// double-double; gainCap bounds that resolution: a gain unresolved to it
// is not certified. It is newtonTol, the floor of the X certificate: a
// floor of 1e-6 kept a cheap-control K off by 2.8e-7, 500 times its move
// per ulp of X, at Accuracy 0, while the gain of X one Newton step on
// was exact. That floor, gainFloor, remains the fallback once neither K
// nor that of X one step on meets gainTol (gainOfX), and bounds what the
// Newton iterate of X adds to the tolerance (gainMissOf): with a
// closed-loop pole at 1 − 3e-7, K off by 2.3e-7 was within a tenth of its
// eps-sensitivity, which no perturbation of X shows (its move per ulp of
// X 3e-13), and Newton steps of X stayed in that noise.
const (
	gainTol   = newtonTol
	gainFloor = 1e-6
	gainCap   = 1e-4
)

// gainMiss returns the distance ‖K − K*‖₁/‖K*‖₁ of res.K from the gain
// K* = (R + BᵀXB)⁻¹(BᵀXA + Sᵀ) of res.X (ddGain, S that of opts), with
// an exact null space of B (nullExact) formed in the inputs w as V times
// the ddGain of BV, VᵀRV, SV and X, in units of its tolerance
// (gainMissOf). In the inputs u, R + BᵀXB loses R₂₂ even in
// double-double under cheap control (K* off by 1e-6 at ‖BᵀXB‖/‖R‖ ≈
// 1e26), where (BV)ᵀX(BV) keeps it on the diagonal; without one, BV₂
// rounded in float64 moved K* by O(1) there. Under cheap control with
// rank(B) < m and a rank-deficient Q, the float64 gain loses also the
// part of K on B's range fixed by R against X's small eigenvalues: K
// came out off by O(1) with X exact to 1e-16, and as the cost of K
// equals X there, no Newton or Hewer step sees it.
func (red *reducedInputs) gainMiss(A, B, R *mat.Dense, opts *RiccatiOpts, res *RiccatiResult) gainDist {
	n, _ := A.Dims()
	return gainMissOf(n, res, gainIn(A, B, R, opts, red))
}

// crossTerm returns the S of opts, nil for none.
func crossTerm(opts *RiccatiOpts) *mat.Dense {
	if opts == nil {
		return nil
	}
	return opts.S
}

// gainMissOf returns the distance ‖K − K*‖₁/‖K*‖₁ of res.K from the
// gain K* of res.X that gain forms and its tolerance: the largest of
// 8× the move of K* under the relative eps
// perturbations X∘(1 + eps) and X∘(1 ± eps) of X, and 32·eps²·cond for
// the componentwise condition cond of gain, the resolution of K* in
// double-double. The distance is +Inf when that tolerance exceeds gainCap
// or gain fails. With the Newton iterate res.x1 of a certified step above eps,
// the distance is the larger of those from K* and from the gain of x1:
// the gain of a cheap-control Dare amplified an X off by 5e-14 (its step
// 7e-14, inside newtonTol) to a K off by 2e-4 to 9e-3 from that of the
// solution, K the gain of X, while that of x1 was within K's own
// eps-sensitivity. The ±eps perturbations of X do not bound this move: in
// their direction it was 1e4 times that along the step. The tolerance
// then also takes, up to gainFloor, 8× the move of the gain of x1 under
// those perturbations of x1: the gain of an X off by 1.3e-9 moved under
// gainTol per ulp, that of x1 by 1.7e-8, and K, the pencil gain, was
// exact to 1.8e-9.
func gainMissOf(n int, res *RiccatiResult, gain func(X *mat.Dense) (*mat.Dense, float64, bool)) gainDist {
	fail := gainDist{d: math.Inf(1), tol: 1}
	K, mcond, ok := gain(res.X)
	if !ok {
		return fail
	}
	nk := mat.Norm(K, 1)
	if nk == 0 {
		if mat.Norm(res.K, 1) == 0 {
			return gainDist{d: 0, tol: 1}
		}
		return fail
	}
	tol := 0.0
	xp := mat.NewDense(n, n, nil)
	pr := xp.RawMatrix()
	ulpMove := func(X, K *mat.Dense) (float64, bool) {
		move := 0.0
		xr := X.RawMatrix()
		for p := range 2 {
			for i := range n {
				for j := range n {
					f := 1.0
					if p == 1 && (i+j+i*j)&1 == 1 {
						f = -1
					}
					pr.Data[i*pr.Stride+j] = xr.Data[i*xr.Stride+j] * (1 + f*eps())
				}
			}
			kp, _, ok := gain(xp)
			if !ok {
				return 0, false
			}
			kp.Sub(kp, K)
			move = max(move, 8*mat.Norm(kp, 1)/nk)
		}
		return move, true
	}
	move, ok := ulpMove(res.X, K)
	if !ok {
		return fail
	}
	tol = max(tol, move)
	var k1 *mat.Dense
	if res.x1 != nil {
		if k1, _, ok = gain(res.x1); !ok {
			return fail
		}
		if move, ok = ulpMove(res.x1, k1); !ok {
			return fail
		}
		tol = max(tol, min(move, gainFloor))
	}
	tol = max(tol, 32*eps()*eps()*mcond)
	if !(tol <= gainCap) {
		return fail
	}
	var d mat.Dense
	d.Sub(res.K, K)
	miss := mat.Norm(&d, 1) / nk
	stepMove := 0.0
	if k1 != nil {
		d.Sub(res.K, k1)
		miss = max(miss, mat.Norm(&d, 1)/nk)
		d.Sub(k1, K)
		stepMove = mat.Norm(&d, 1) / nk
	}
	return gainDist{d: miss, tol: tol, move: stepMove, k: K}
}

// gainDist is the distance d of a K from the gain k of its X, the
// resolution tol of k and its move across the Newton step x1
// (gainMissOf).
type gainDist struct {
	d, tol, move float64
	k            *mat.Dense
}

// gainMove returns ‖K − k‖₁/‖k‖₁, +Inf without k.
func gainMove(k, K *mat.Dense) float64 {
	if k == nil {
		return math.Inf(1)
	}
	var d mat.Dense
	d.Sub(K, k)
	return mat.Norm(&d, 1) / mat.Norm(k, 1)
}

// units returns d in units of tol, at least floor.
func (g gainDist) units(floor float64) float64 {
	return g.d / max(g.tol, floor)
}

// ddGain returns the gain (R + BᵀXB)⁻¹(BᵀXA + Sᵀ) of X (S nil for
// none), its products formed and the system solved (Gaussian
// elimination, partial pivoting) in double-double, with the componentwise
// condition ‖|M⁻¹|·(|R| + |B|ᵀ|X||B|)·|K| + |M⁻¹|·(|B|ᵀ|X||A| +
// |S|ᵀ)‖₁/‖K‖₁ of K, M = R + BᵀXB, the
// relative error of K per unit rounding of its double-double operations;
// ok is false when a pivot vanishes or K is not finite.
func ddGain(A, B, R, S, X *mat.Dense) (K *mat.Dense, cond float64, ok bool) {
	xRaw := X.RawMatrix()
	return ddGainOf(A, B, R, S, xRaw.Data, nil, xRaw.Stride)
}

// ddGainOf is ddGain of the double-double X = xh + xl (xl nil for none),
// row-major with stride ldx.
func ddGainOf(A, B, R, S *mat.Dense, xh, xl []float64, ldx int) (K *mat.Dense, cond float64, ok bool) {
	n, m := B.Dims()
	aRaw, bRaw, rRaw := A.RawMatrix(), B.RawMatrix(), R.RawMatrix()
	bxH, bxL := make([]float64, m*n), make([]float64, m*n)
	ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, xh, ldx, 1, bxH, bxL, n)
	if xl != nil {
		ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, xl, ldx, 1, bxH, bxL, n)
	}
	c := 2*m + n
	hi, lo := make([]float64, m*c), make([]float64, m*c)
	for i := range m {
		copy(hi[i*c:i*c+m], rRaw.Data[i*rRaw.Stride:i*rRaw.Stride+m])
		hi[i*c+m+n+i] = 1
	}
	if S != nil {
		sRaw := S.RawMatrix()
		for i := range n {
			for j := range m {
				hi[j*c+m+i] = sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	for _, bx := range [][]float64{bxH, bxL} {
		ddMulAcc(m, m, n, 1, bx, n, 1, bRaw.Data, bRaw.Stride, 1, hi, lo, c)
		ddMulAcc(m, n, n, 1, bx, n, 1, aRaw.Data, aRaw.Stride, 1, hi[m:], lo[m:], c)
	}
	w := make([]dd, m*c)
	for i := range w {
		w[i] = quickTwoSum(twoSum(hi[i], lo[i]))
	}
	for k := range m {
		p := k
		for i := k + 1; i < m; i++ {
			if math.Abs(w[i*c+k].hi) > math.Abs(w[p*c+k].hi) {
				p = i
			}
		}
		if !(w[p*c+k].hi != 0) || math.IsInf(w[p*c+k].hi, 0) {
			return nil, 0, false
		}
		if p != k {
			for j := k; j < c; j++ {
				w[k*c+j], w[p*c+j] = w[p*c+j], w[k*c+j]
			}
		}
		for i := k + 1; i < m; i++ {
			f := w[i*c+k].div(w[k*c+k])
			for j := k + 1; j < c; j++ {
				w[i*c+j] = w[i*c+j].add(f.mul(w[k*c+j]).neg())
			}
		}
	}
	for k := m - 1; k >= 0; k-- {
		for j := m; j < c; j++ {
			s := w[k*c+j]
			for i := k + 1; i < m; i++ {
				s = s.add(w[k*c+i].mul(w[i*c+j]).neg())
			}
			w[k*c+j] = s.div(w[k*c+k])
		}
	}
	K = mat.NewDense(m, n, nil)
	kRaw := K.RawMatrix()
	for i := range m {
		for j := range n {
			kRaw.Data[i*kRaw.Stride+j] = w[i*c+m+j].hi
		}
	}
	if !finiteSlice(kRaw.Data) {
		return nil, 0, false
	}
	abs := func(d []float64, r, cols, stride int) *mat.Dense {
		out := mat.NewDense(r, cols, nil)
		o := out.RawMatrix().Data
		for i := range r {
			for j := range cols {
				o[i*cols+j] = math.Abs(d[i*stride+j])
			}
		}
		return out
	}
	var bx, bxb, mag, t mat.Dense
	bx.Mul(abs(bRaw.Data, n, m, bRaw.Stride).T(), abs(xh, n, n, ldx))
	bxb.Mul(&bx, abs(bRaw.Data, n, m, bRaw.Stride))
	bxb.Add(&bxb, abs(rRaw.Data, m, m, rRaw.Stride))
	mag.Mul(&bxb, abs(kRaw.Data, m, n, kRaw.Stride))
	t.Mul(&bx, abs(aRaw.Data, n, n, aRaw.Stride))
	mag.Add(&mag, &t)
	if S != nil {
		sRaw := S.RawMatrix()
		mag.Add(&mag, abs(sRaw.Data, n, m, sRaw.Stride).T())
	}
	minv := mat.NewDense(m, m, nil)
	mi := minv.RawMatrix().Data
	for i := range m {
		for j := range m {
			mi[i*m+j] = math.Abs(w[i*c+m+n+j].hi)
		}
	}
	t.Reset()
	t.Mul(minv, &mag)
	cond = mat.Norm(&t, 1) / mat.Norm(K, 1)
	return K, cond, !math.IsInf(cond, 0) && !math.IsNaN(cond)
}

// nullExact reports whether columns r.. of V span a null space of B
// exactly, as with duplicated or m > n inputs: one step refining each
// against B's leading r singular triplets (BV the columns σᵢuᵢ) brings
// B·v, formed in double-double, to the rounding eps²·σ₁ of the product,
// where a B with σ_min ≈ eps·σ₁ keeps σ_min there.
func nullExact(B, BV, V *mat.Dense, sv []float64, r int) bool {
	n, m := B.Dims()
	bRaw, vRaw, uRaw := B.RawMatrix(), V.RawMatrix(), BV.RawMatrix()
	hi, lo := make([]float64, n), make([]float64, n)
	c, d := make([]float64, r), make([]float64, m)
	tol := 64 * float64(n*m) * eps() * eps() * sv[0]
	for j := r; j < m; j++ {
		clear(hi)
		clear(lo)
		ddMulAcc(n, 1, m, 1, bRaw.Data, bRaw.Stride, 1, vRaw.Data[j:], vRaw.Stride, 1, hi, lo, 1)
		for i := range n {
			hi[i] += lo[i]
		}
		for k := range r {
			c[k] = 0
			for i := range n {
				c[k] += uRaw.Data[i*uRaw.Stride+k] * hi[i]
			}
			c[k] /= sv[k] * sv[k]
		}
		for i := range m {
			d[i] = 0
			for k := range r {
				d[i] += vRaw.Data[i*vRaw.Stride+k] * c[k]
			}
		}
		res := 0.0
		for i := range n {
			s := hi[i]
			for k := range m {
				s -= bRaw.Data[i*bRaw.Stride+k] * d[k]
			}
			res = math.Hypot(res, s)
		}
		if !(res <= tol) {
			return false
		}
	}
	return true
}

// gain returns the gain VK of the inputs u from the gain K of the inputs
// w, in ws.kData when ws is given.
func (red *reducedInputs) gain(K *mat.Dense, ws *RiccatiWorkspace) *mat.Dense {
	m, _ := red.v.Dims()
	_, n := K.Dims()
	var out *mat.Dense
	if ws != nil {
		out = mat.NewDense(m, n, ws.kData[:m*n])
	} else {
		out = mat.NewDense(m, n, nil)
	}
	out.Mul(red.v, mat.DenseCopyOf(K))
	return out
}

// placedDare solves the Dare by placedHewer, state-scaled as dare.
func placedDare(A, B, Q, R *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
	problem, err := newRiccatiProblem(A, B, Q, R, opts)
	if err != nil {
		return nil, err
	}
	if err := problem.ws.fits(problem.n, problem.m); err != nil {
		return nil, err
	}
	problem.scale(opts, false)
	eig := problem.ws.eig[:problem.n]
	X, K, rcnd, err := problem.placedHewer(eig)
	if err != nil {
		return nil, err
	}
	return problem.result(X, K, eig, rcnd, 0), nil
}

// pencilRiccati solves the Riccati equation from the extended pencil
// (pencilSolution) and returns X, K, rcnd and the Accuracy of X with Eig
// in ws.eig. A solution acceptRiccati rejects is replaced by the first
// accepted one of the other passes' solutions and then of lastResort;
// with none accepted, by problem.poor (Accuracy 1).
func (problem *riccatiProblem) pencilRiccati(continuous bool) (*mat.Dense, *mat.Dense, float64, int, error) {
	n := problem.n
	problem.collecting = true
	subspace, X, _, rcnd, err := problem.pencilSolution(continuous)
	eig := problem.ws.eig[:n]
	var K *mat.Dense
	if err == nil {
		for i := range n {
			eig[i] = complex(subspace.alphaR[i]/subspace.beta[i], subspace.alphaI[i]/subspace.beta[i])
		}
		K, err = problem.acceptRiccati(continuous, X, eig, rcnd)
	}
	if err != nil {
		for i := range problem.candidates {
			c := &problem.candidates[i]
			X, _, rcnd = problem.restorePass(c)
			for j := range n {
				eig[j] = complex(c.wr[j]/c.beta[j], c.wi[j]/c.beta[j])
			}
			if k, cerr := problem.acceptRiccati(continuous, X, eig, rcnd); cerr == nil {
				K, err = k, nil
				break
			}
		}
	}
	if err != nil {
		if x, k, rc, cerr := problem.lastResort(continuous, eig); cerr == nil {
			X, K, rcnd, err = x, k, rc, nil
		}
	}
	if err != nil {
		// Before poor: a pass with no Accuracy 1 X must not let the
		// other scaling mode return one.
		if problem.boundaryUnstabilizable(continuous) {
			return nil, nil, 0, 0, errBoundaryUnstabilizable
		}
		if problem.poor == nil {
			return nil, nil, 0, 0, err
		}
		X, K, rcnd = problem.poorSolution(eig)
		return X, K, rcnd, 1, nil
	}
	return X, K, rcnd, 0, nil
}

// lastResort solves the cost-balanced pencil at the cost scales of
// riccatiNorms.candidates not yet tried and returns the first solution
// verifyRiccati accepts.
func (problem *riccatiProblem) lastResort(continuous bool, eig []complex128) (*mat.Dense, *mat.Dense, float64, error) {
	problem.resolved = true
	tried := func(sigma float64) bool {
		for _, c := range problem.candidates {
			if c.sigma == sigma {
				return true
			}
		}
		return false
	}
	norms := problem.pencilNorms()
	rRaw := problem.R.RawMatrix()
	rn := impl.Dlange(lapack.Frobenius, problem.m, problem.m, rRaw.Data, rRaw.Stride, nil)
	for _, sigma := range []float64{norms.firstSigma(), pow2Near(rn), pow2Near(norms.q), pow2Near(math.Sqrt(rn) * math.Sqrt(norms.q))} {
		if !(sigma > 0) || math.IsInf(sigma, 0) || tried(sigma) {
			continue
		}
		problem.pencilNorms()
		problem.sigma, problem.balanced = sigma, true
		subspace, err := problem.generalizedStableSubspace(continuous, problem.pencil())
		if err != nil {
			continue
		}
		X, xData, rcnd, err := problem.stabilizingSolution(subspace.vectors)
		if err != nil || !finiteSlice(xData) {
			continue
		}
		for i := range eig {
			eig[i] = complex(subspace.alphaR[i]/subspace.beta[i], subspace.alphaI[i]/subspace.beta[i])
		}
		if K, err := problem.acceptRiccati(continuous, X, eig, rcnd); err == nil {
			return X, K, rcnd, nil
		}
		problem.candidates = append(problem.candidates, riccatiPass{sigma: sigma})
	}
	return nil, nil, 0, ErrNoStabilizing
}

// placedHewer is Dare's last source of a stabilizing solution, tried by
// Dare (placedDare, as given and then with NoScaling flipped) only
// when neither scaling mode's pencil gave a certified X nor a boundary
// verdict: Hewer's
// iteration, which from any stabilizing gain converges to the stabilizing
// solution, started from a pole-placement gain. Under cheap control of a
// strongly unstable plant (ρ(A) ≈ 500, Q/R ≈ 1e26, ‖X‖ ≈ 1e30) every cost
// scale of the pencil misplaces its eigenvalues or yields an X whose gain
// destabilizes, while Hewer from a placed gain reaches the oracle X to
// 1e-10. Placement (KNV, then Varga's Schur method) only supplies
// a K0 with E⁻¹(A − BK0) stable in float64, so its pole accuracy does not
// matter; the first iterate X0 solves the Stein equation of the cost of
// K0, A_K0ᵀX0A_K0 − EᵀX0E + Q_K0 = 0, and the result is the one
// verifyRiccati certifies from it (Accuracy 0), else an error. Rcnd is
// that of U11 for the orthonormal basis of span[I; X̂/σ], σ the power of
// two nearest ‖X̂‖ (the subspace a pencil at that cost scale deflates).
// It declines a cross term S with rank(B) < m: inputs in the null space
// of B lower the cost by about xᵀSR⁻¹Sᵀx, comparable to Q under cheap
// control, which R + BᵀXB (singular in float64) loses, so Newton then
// certifies a wrong X (≈ Q).
func (problem *riccatiProblem) placedHewer(eig []complex128) (*mat.Dense, *mat.Dense, float64, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	A, B := problem.A, problem.B
	var eLU mat.LU
	if problem.E != nil {
		eLU.Factorize(problem.E)
		var ea, eb mat.Dense
		if err := eLU.SolveTo(&ea, false, A); err != nil {
			return nil, nil, 0, ErrNoStabilizing
		}
		if err := eLU.SolveTo(&eb, false, B); err != nil {
			return nil, nil, 0, ErrNoStabilizing
		}
		A, B = &ea, &eb
	}
	var svd mat.SVD
	if !svd.Factorize(B, mat.SVDFull) {
		return nil, nil, 0, ErrNoStabilizing
	}
	sv := svd.Values(nil)
	r := 0
	for _, s := range sv {
		if s > float64(max(n, m))*eps()*sv[0] {
			r++
		}
	}
	if r == 0 || (problem.S != nil && r < m) {
		return nil, nil, 0, ErrNoStabilizing
	}
	nf := float64(n)
	poleSets := [][]complex128{make([]complex128, n), make([]complex128, n)}
	for i := range n {
		poleSets[0][i] = complex(0.1+0.8*float64(i)/nf, 0)
		poleSets[1][i] = complex(-0.6+1.2*(float64(i)+0.5)/nf, 0)
	}
	xData := ws.xData[:n*n]
	try := func(K0 *mat.Dense) (*mat.Dense, *mat.Dense, float64, error) {
		if K0 == nil || !finiteSlice(K0.RawMatrix().Data) {
			return nil, nil, 0, ErrNoStabilizing
		}
		if _, ok := problem.stableGain(false, K0); !ok {
			return nil, nil, 0, ErrNoStabilizing
		}
		var bk, acl, rk, qk mat.Dense
		bk.Mul(problem.B, K0)
		acl.Sub(problem.A, &bk)
		rk.Mul(problem.R, K0)
		qk.Mul(K0.T(), &rk)
		qk.Add(&qk, problem.Q)
		if problem.S != nil {
			var sk mat.Dense
			sk.Mul(problem.S, K0)
			qk.Sub(&qk, &sk)
			qk.Sub(&qk, sk.T())
		}
		f := mat.DenseCopyOf(acl.T())
		if problem.E != nil {
			var ft mat.Dense
			if err := eLU.SolveTo(&ft, false, &acl); err != nil {
				return nil, nil, 0, ErrNoStabilizing
			}
			f = mat.DenseCopyOf(ft.T())
		}
		symmetrize(qk.RawMatrix().Data, n, n)
		Y, err := dlyap(f, &qk, nil)
		if err != nil {
			return nil, nil, 0, ErrNoStabilizing
		}
		X0 := Y
		if problem.E != nil {
			var w, x mat.Dense
			if err := eLU.SolveTo(&w, true, Y); err != nil {
				return nil, nil, 0, ErrNoStabilizing
			}
			if err := eLU.SolveTo(&x, true, w.T()); err != nil {
				return nil, nil, 0, ErrNoStabilizing
			}
			X0 = &x
		}
		copyStrided(xData, n, X0.RawMatrix().Data, X0.RawMatrix().Stride, n, n)
		symmetrize(xData, n, n)
		if !finiteSlice(xData) {
			return nil, nil, 0, ErrNoStabilizing
		}
		X := mat.NewDense(n, n, xData)
		K, err := problem.acceptRiccati(false, X, eig, 0)
		if err != nil {
			return nil, nil, 0, err
		}
		return X, K, placedRcnd(xData, n), nil
	}
	for _, poles := range poleSets {
		if r >= 2 {
			if X, K, rcnd, err := try(placeKNV(A, &svd, sv[:r], poles)); err == nil {
				return X, K, rcnd, nil
			}
		}
		if K0, err := placeSchur(A, B, poles); err == nil {
			if X, K, rcnd, err := try(K0); err == nil {
				return X, K, rcnd, nil
			}
		}
	}
	return nil, nil, 0, ErrNoStabilizing
}

// placedRcnd returns the reciprocal 2-norm condition of U11 for the
// orthonormal basis of span[I; X/σ], σ the power of two nearest ‖X‖_F:
// √((1 + λmin²)/(1 + λmax²)) over the eigenvalues λ of X/σ ordered by
// magnitude.
func placedRcnd(xData []float64, n int) float64 {
	sigma := pow2Near(impl.Dlange(lapack.Frobenius, n, n, xData, n, nil))
	var es mat.EigenSym
	if !es.Factorize(mat.NewSymDense(n, slices.Clone(xData)), false) {
		return 0
	}
	lo, hi := math.Inf(1), 0.0
	for _, v := range es.Values(nil) {
		a := math.Abs(v / sigma)
		lo, hi = min(lo, a), max(hi, a)
	}
	return math.Sqrt((1 + lo*lo) / (1 + hi*hi))
}

// acceptRiccati returns the gain of the solution X (with rcnd), which it
// verifies (verifyRiccati); X and eig may be refined in place.
func (problem *riccatiProblem) acceptRiccati(continuous bool, X *mat.Dense, eig []complex128, rcnd float64) (*mat.Dense, error) {
	K, err := problem.gain(continuous, X.RawMatrix().Data)
	if err != nil {
		return nil, err
	}
	return problem.verifyRiccati(continuous, X, K, eig, rcnd)
}

// riccatiPoor is a stabilizing solution no Newton step certified: X, its
// gain K, the eigenvalues of A − BK, rcnd and its first Newton step.
type riccatiPoor struct {
	x, k       []float64
	eig        []complex128
	rcnd, step float64
}

// keepPoor saves X (xData, gain K) in problem.poor with the eigenvalues
// eig of A − BK (nil: those of the last newtonCost), unless the saved one
// has a smaller first Newton step or rcnd is below poorRcnd.
func (problem *riccatiProblem) keepPoor(xData []float64, K *mat.Dense, eig []complex128, rcnd, step float64) bool {
	if !(rcnd >= poorRcnd) {
		return false
	}
	p := problem.poor
	if p == nil {
		p = &riccatiPoor{}
		problem.poor = p
	} else if !(step < p.step) {
		return false
	}
	n, m := problem.n, problem.m
	p.x = append(p.x[:0], xData...)
	p.k = append(p.k[:0], K.RawMatrix().Data[:m*n]...)
	if eig != nil {
		p.eig = append(p.eig[:0], eig...)
	} else {
		p.eig = append(p.eig[:0], make([]complex128, n)...)
		problem.closedLoopEig(p.eig)
	}
	p.rcnd, p.step = rcnd, step
	return true
}

// errBoundaryUnstabilizable is the ErrNoStabilizing of
// boundaryUnstabilizable: structural, so it holds in either scaling mode
// over any Accuracy 1 X, though not over a certified one (otherScaling).
var errBoundaryUnstabilizable = fmt.Errorf("%w: a mode on or beyond the stability boundary is unstabilizable", ErrNoStabilizing)

// boundaryUnstabilizable reports a finite eigenvalue λ of (A, E) on or
// beyond the stability boundary, or within rounding of it, at which
// [A − λE, B] has rank below n by the PBH test: no stabilizing solution
// exists (icare/idare Report 3), and an Accuracy 1 X would be stable only
// to rounding, its exact closed loop unstable (K72KOR). Within rounding
// is 10× the first-order bound τ = (n·eps·(|y|ᵀ|A||x| + |λ|·|y|ᵀ|E||x|)
// + |y|ᵀ|(A − λE)x|)/|yᴴEx| of λ's error (x, y its right and left
// eigenvectors): componentwise rounding plus the residual, which carries
// QZ's backward error, so an exact mode at 0 or 1 of a dense A, computed
// eps·‖A‖·κ off, is tested whatever the rounding (zero1 under -race), while a mode at −3e-16 that is an exact diagonal entry
// of a triangular A is known to relative eps and stays stable (BWUCB5),
// as does a lightly damped mode at −3e-9 ± i. PBH rank is σmin ≤
// n·eps·‖[‖A‖ + |λ|‖E‖, ‖B‖]‖ + τ‖E‖. A mode in a Jordan block with a
// controllable one splits by √(eps·κ)·‖A‖ beyond first order: x ⊥ y
// there, and τ, past the split, would swallow a stable mode at −1e-5. λ is
// tested alone only when τ is below its distance to every other
// eigenvalue; the mean of each nearest-first cluster of λ within
// max(√eps·(‖A‖ + |λ|‖E‖), τ) is tested too, to 10·n·eps·(‖A‖ +
// |mean|‖E‖) times the cluster condition 1/min(pl, pr) (Dtgsen; a Jordan
// block at 0 with an ill-conditioned P is off by more than eps·‖A‖,
// MPWOJM) when it is within √eps of the boundary, and only when that
// band is below the mean's distance to the eigenvalues left out.
func (problem *riccatiProblem) boundaryUnstabilizable(continuous bool) bool {
	n, m := problem.n, problem.m
	aRaw, bRaw := problem.A.RawMatrix(), problem.B.RawMatrix()
	a := make([]float64, n*n)
	copyStrided(a, n, aRaw.Data, aRaw.Stride, n, n)
	e := make([]float64, n*n)
	for i := range n {
		e[i*n+i] = 1
	}
	if problem.E != nil {
		eRaw := problem.E.RawMatrix()
		copyStrided(e, n, eRaw.Data, eRaw.Stride, n, n)
	}
	an, en := frobenius(a), frobenius(e)
	bn := impl.Dlange(lapack.Frobenius, n, m, bRaw.Data, bRaw.Stride, nil)
	ac, ec := slices.Clone(a), slices.Clone(e)
	wr, wi, beta := make([]float64, n), make([]float64, n), make([]float64, n)
	vl, vr := make([]float64, n*n), make([]float64, n*n)
	work := make([]float64, max(1, 8*n))
	if !impl.Dggev(lapack.LeftEVCompute, lapack.RightEVCompute, n, ac, n, ec, n, wr, wi, beta, vl, n, vr, n, work, len(work)) {
		return false
	}
	nf, ep := float64(n), eps()
	inBand := func(lam complex128, tol float64) bool {
		if continuous {
			return real(lam) >= -tol
		}
		return cmplx.Abs(lam) >= 1-tol
	}
	g := mat.NewDense(2*n, 2*(n+m), nil)
	var svd mat.SVD
	pbhDeficient := func(lam complex128, tol float64) bool {
		lr, li := real(lam), imag(lam)
		for i := range n {
			for j := range n {
				re, im := a[i*n+j]-lr*e[i*n+j], -li*e[i*n+j]
				g.Set(i, j, re)
				g.Set(i, n+m+j, -im)
				g.Set(n+i, j, im)
				g.Set(n+i, n+m+j, re)
			}
			for j := range m {
				b := bRaw.Data[i*bRaw.Stride+j]
				g.Set(i, n+j, b)
				g.Set(n+i, n+m+n+j, b)
			}
		}
		if !svd.Factorize(g, mat.SVDNone) {
			return false
		}
		sv := svd.Values(nil)
		return sv[2*n-1] <= nf*ep*math.Hypot(an+cmplx.Abs(lam)*en, bn)+tol*en
	}
	var sa, se, sar, sai, sbeta []float64
	clusterCond := func(lam complex128, rho float64) float64 {
		if sa == nil {
			sa, se = slices.Clone(a), slices.Clone(e)
			sar, sai, sbeta = make([]float64, n), make([]float64, n), make([]float64, n)
			var query [1]float64
			impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
				n, sa, n, se, n, sar, sai, sbeta, nil, 1, nil, 1, query[:], -1, nil)
			w := make([]float64, int(query[0]))
			if _, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
				n, sa, n, se, n, sar, sai, sbeta, nil, 1, nil, 1, w, len(w), nil); !ok {
				sbeta = nil
			}
		}
		if sbeta == nil {
			return 1
		}
		sel := make([]bool, n)
		for j := 0; j < n; j++ {
			in := func(k int) bool {
				return sbeta[k] != 0 && cmplx.Abs(complex(sar[k]/sbeta[k], sai[k]/sbeta[k])-lam) <= rho
			}
			if sai[j] != 0 && j+1 < n {
				sel[j] = in(j) || in(j+1)
				sel[j+1] = sel[j]
				j++
				continue
			}
			sel[j] = in(j)
		}
		ta, te := slices.Clone(sa), slices.Clone(se)
		ar, ai, b := slices.Clone(sar), slices.Clone(sai), slices.Clone(sbeta)
		lw := max(4*n+16, n*n)
		_, pl, pr, _, ok := impl.Dtgsen(1, false, false, sel, n, ta, n, te, n, ar, ai, b,
			nil, 1, nil, 1, make([]float64, lw), lw, make([]int, n+6), n+6)
		if !ok || !(pl > 0) || !(pr > 0) {
			return 1
		}
		return 1 / min(pl, pr)
	}
	lams := make([]complex128, n)
	finite := make([]bool, n)
	for k := range n {
		if finite[k] = math.Abs(beta[k]) > 0; finite[k] {
			lams[k] = complex(wr[k]/beta[k], wi[k]/beta[k])
		}
	}
	x, y := make([]complex128, n), make([]complex128, n)
	ds := make([]float64, 0, n)
	for k := 0; k < n; k++ {
		kk := k
		if wi[k] != 0 {
			k++
		}
		if !finite[kk] {
			continue
		}
		lam := lams[kk]
		for i := range n {
			if imag(lam) == 0 {
				x[i], y[i] = complex(vr[i*n+kk], 0), complex(vl[i*n+kk], 0)
			} else {
				x[i], y[i] = complex(vr[i*n+kk], vr[i*n+kk+1]), complex(vl[i*n+kk], vl[i*n+kk+1])
			}
		}
		var ya, ye, yr float64
		var yex complex128
		for i := range n {
			var ra, re float64
			var ax, ex complex128
			for j := range n {
				ra += math.Abs(a[i*n+j]) * cmplx.Abs(x[j])
				re += math.Abs(e[i*n+j]) * cmplx.Abs(x[j])
				ax += complex(a[i*n+j], 0) * x[j]
				ex += complex(e[i*n+j], 0) * x[j]
			}
			ay := cmplx.Abs(y[i])
			ya += ay * ra
			ye += ay * re
			yr += ay * cmplx.Abs(ax-lam*ex)
			yex += cmplx.Conj(y[i]) * ex
		}
		mod := cmplx.Abs(lam)
		tol := 10 * (nf*ep*(ya+mod*ye) + yr) / cmplx.Abs(yex)
		rho := math.Sqrt(ep) * (an + mod*en)
		ds = ds[:0]
		for j := range n {
			if finite[j] {
				ds = append(ds, cmplx.Abs(lams[j]-lam))
			}
		}
		slices.Sort(ds)
		gap := math.Inf(1)
		if len(ds) > 1 {
			gap = ds[1]
		}
		if tol < gap && inBand(lam, tol) && pbhDeficient(lam, tol) {
			return true
		}
		for c := 1; c < len(ds) && ds[c] <= max(rho, tol); c++ {
			r := ds[c]
			if c+1 < len(ds) {
				if ds[c+1] == r {
					continue
				}
				r = (r + ds[c+1]) / 2
			}
			var mean complex128
			for j := range n {
				if finite[j] && cmplx.Abs(lams[j]-lam) <= r {
					mean += lams[j]
				}
			}
			mean /= complex(float64(c+1), 0)
			sep := math.Inf(1)
			for j := range n {
				if finite[j] && cmplx.Abs(lams[j]-lam) > r {
					sep = min(sep, cmplx.Abs(lams[j]-mean))
				}
			}
			scale := an + cmplx.Abs(mean)*en
			mtol := 10 * nf * ep * scale
			if !inBand(mean, mtol) && inBand(mean, math.Sqrt(ep)*scale) {
				mtol *= clusterCond(lam, r)
			}
			if mtol < sep && inBand(mean, mtol) && pbhDeficient(mean, mtol) {
				return true
			}
		}
	}
	return false
}

// poorSolution restores problem.poor to ws.xData, ws.kData and eig.
func (problem *riccatiProblem) poorSolution(eig []complex128) (X, K *mat.Dense, rcnd float64) {
	n, m, ws, p := problem.n, problem.m, problem.ws, problem.poor
	xData, kData := ws.xData[:n*n], ws.kData[:m*n]
	copy(xData, p.x)
	copy(kData, p.k)
	copy(eig, p.eig)
	return mat.NewDense(n, n, xData), mat.NewDense(m, n, kData), p.rcnd
}

// gain returns the gain of X: careGain or dareGain.
func (problem *riccatiProblem) gain(continuous bool, xData []float64) (*mat.Dense, error) {
	if continuous {
		return problem.careGain(xData), nil
	}
	return problem.dareGain(xData)
}

// careGain returns K = R⁻¹(BᵀXE + Sᵀ) (E = I when absent) in ws.kData;
// R is Cholesky-factored in ws.rChol.
func (problem *riccatiProblem) careGain(xData []float64) *mat.Dense {
	n, m, ws := problem.n, problem.m, problem.ws
	bRaw := problem.B.RawMatrix()
	kData := ws.kData[:m*n]
	btx := kData
	if problem.E != nil {
		btx = ws.btx[:m*n]
	}
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n})
	if problem.E != nil {
		eRaw := problem.E.RawMatrix()
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
			blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
			0, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n})
	}
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		for j := range m {
			row := kData[j*n:]
			for i := range n {
				row[i] += sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	impl.Dpotrs(blas.Upper, m, n, ws.rChol[:m*m], m, kData, n)
	return mat.NewDense(m, n, kData)
}

// dareGain returns K = (R + BᵀXB)⁻¹(BᵀXA + Sᵀ) in ws.kData, or
// ErrNoStabilizing when R + BᵀXB is not positive definite.
func (problem *riccatiProblem) dareGain(xData []float64) (*mat.Dense, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	aRaw, bRaw, rRaw := problem.A.RawMatrix(), problem.B.RawMatrix(), problem.R.RawMatrix()
	btx := ws.btx[:m*n]
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n})

	rbar := ws.rbar[:m*m]
	copyStrided(rbar, m, rRaw.Data, rRaw.Stride, m, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
		blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		1, blas64.General{Rows: m, Cols: m, Data: rbar, Stride: m})
	if !impl.Dpotrf(blas.Upper, m, rbar, m) {
		return nil, ErrNoStabilizing
	}

	kData := ws.kData[:m*n]
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: btx, Stride: n},
		blas64.General{Rows: n, Cols: n, Data: aRaw.Data, Stride: aRaw.Stride},
		0, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n})
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		for j := range m {
			row := kData[j*n:]
			for i := range n {
				row[i] += sRaw.Data[i*sRaw.Stride+j]
			}
		}
	}
	impl.Dpotrs(blas.Upper, m, n, rbar, m, kData, n)
	return mat.NewDense(m, n, kData), nil
}

// verifyRiccati returns the gain of the solution X (gain K) once a Newton
// step certifies it. A step is ‖X₁ − X‖/‖X₁‖ (stepNorms) for the Newton
// iterate X₁ = X + Δ of newtonCost (Newton–Kleinman for Care, Hewer for
// Dare), whose Δ solves the Lyapunov (Stein) equation of A − BK for the
// double-double Riccati residual of X: to first order the forward error
// of X. X is kept when its step is within newtonTol with A − BK stable.
//
// The relative Riccati residual is no certificate: under cheap control
// the rounded stabilizing X has a residual of 1e-6 of its terms while an
// X wrong by 5e-7 has 3e-11 (Dare w15 #16), and against the rounding of
// the gain form both sit at 1e-17. The step errs the safe way for the
// pencil's X (it overstates the error where Newton has no quadratic
// region, cheap control with ‖X‖‖BR⁻¹Bᵀ‖‖L_K⁻¹‖ ≫ 1), but not for Newton
// iterates, whose error can lie where the step does not see it: a step
// of 3e-9 from an iterate wrong by 2.4e-7, or Newton–Kleinman creeping
// from an X accurate to 1e-11 to 1e-6 and stalling there. So a larger
// first step is iterated, up to newtonSteps solves, and an iterate is kept
// only once its step is within newtonRefineTol, where Newton has
// converged to its rounding level; the iterate and its K then replace X
// and K. Eig is that of A − BK for the returned K. A first solve whose
// Lyapunov equation is singular is kept with a residual within rescueTol
// of its terms and a stable A − BK as a candidate of problem.poor, as is
// a stabilizing X whose first step exceeds newtonTol (keepPoor, before
// any iterate overwrites it). From a stabilizing K, Newton's iterates
// stay stabilizing and converge in exact arithmetic, so when they break
// down (an iterate unstable or unsolvable) or newtonSteps solves pass
// without convergence or a stall within newtonGlobalTol, rounding rules
// the iteration and the first step is no error estimate (75DTNG: an
// unscaled non-normal pencil X off by 5e8, first step 0.5): the
// candidate it displaced is restored, else, as a singular first solve,
// it is kept with step +Inf only with a residual within rescueTol of its
// terms (3XCWZW: a Hamiltonian pair on the axis to rounding left an X of
// residual 0.9 of its terms, first step 1, that no stabilizing solution
// underlies). Otherwise verifyRiccati returns
// problem.rescueErr, or ErrNoStabilizing, and the caller falls back to
// problem.poor with Accuracy 1. X and eig are updated in place.
func (problem *riccatiProblem) verifyRiccati(continuous bool, X, K *mat.Dense, eig []complex128, rcnd float64) (*mat.Dense, error) {
	fail := problem.rescueErr
	if fail == nil {
		fail = ErrNoStabilizing
	}
	xData := X.RawMatrix().Data
	bestStep, stalls := math.Inf(1), 0
	kept, prevPoor := false, (*riccatiPoor)(nil)
	defer func() {
		if !kept {
			return
		}
		if prevPoor != nil {
			problem.poor = prevPoor
			return
		}
		if res, scale := problem.residualTerms(continuous, problem.poor.x); res <= rescueTol*scale {
			problem.poor.step = math.Inf(1)
		} else {
			problem.poor = nil
		}
	}()
	for it := range newtonSteps {
		x1, err := problem.newtonCost(continuous, K, xData)
		if err != nil {
			if it == 0 && !errors.Is(err, ErrNoStabilizing) && !problem.resolved {
				if res, scale := problem.residualTerms(continuous, xData); res <= rescueTol*scale {
					if vals, ok := problem.stableGain(continuous, K); ok {
						problem.keepPoor(xData, K, vals, rcnd, math.Inf(1))
					}
				}
			}
			return nil, fail
		}
		d, x1n := problem.stepNorms(x1, xData)
		step := 0.0
		if d != 0 {
			step = math.Sqrt(d / x1n)
		}
		if it == 0 {
			if step <= newtonTol {
				problem.closedLoopEig(eig)
				var c1 []float64
				if !continuous && step > eps() {
					c1 = append(problem.ws.certX1[:0], x1...)
					problem.ws.certX1 = c1
				}
				if !problem.descriptorGainCertified(continuous, K, xData, x1, eig) {
					problem.keepPoor(xData, K, eig, rcnd, step)
					return nil, fail
				}
				problem.certStep, problem.certX1 = step, nil
				if problem.xe == nil {
					problem.certX1 = c1
				}
				return K, nil
			}
			if p := problem.poor; p != nil && step < p.step {
				prevPoor = &riccatiPoor{slices.Clone(p.x), slices.Clone(p.k), slices.Clone(p.eig), p.rcnd, p.step}
			}
			kept = problem.keepPoor(xData, K, nil, rcnd, step)
		}
		if it > 0 && step <= newtonRefineTol {
			kept = false
			problem.certStep, problem.certX1 = 0, nil
			prev := append([]float64(nil), xData...)
			copy(xData, x1)
			if K1, x2, err := problem.closedLoopGain(continuous, xData, eig); err == nil {
				if !problem.descriptorGainCertified(continuous, K1, xData, x2, eig) {
					return nil, fail
				}
				return K1, nil
			}
			copy(xData, prev)
			if K, x1, err = problem.closedLoopGain(continuous, xData, eig); err != nil ||
				!problem.descriptorGainCertified(continuous, K, xData, x1, eig) {
				return nil, fail
			}
			return K, nil
		}
		if step < bestStep {
			bestStep, stalls = step, 0
		} else if stalls++; stalls == 3 && step <= newtonGlobalTol {
			kept = false
			break
		}
		copy(xData, x1)
		if K, err = problem.gain(continuous, xData); err != nil {
			break
		}
	}
	return nil, fail
}

// stepNorms returns ‖X₁ − X‖² and ‖X₁‖² (Frobenius) of x1 and xData in
// the caller's coordinates, X = T·X̂·T, so that a step measures the
// relative error of the returned X rather than that of the state-scaled
// iterate, where an error in the small entries of X̂ can hide.
func (problem *riccatiProblem) stepNorms(x1, xData []float64) (d, x1n float64) {
	t, n := problem.t, problem.n
	for i, v := range x1 {
		e := v - xData[i]
		if t != nil {
			w := t[i/n] * t[i%n]
			v, e = v*w, e*w
		}
		d += e * e
		x1n += v * v
	}
	return d, x1n
}

// closedLoopGain returns the gain K of xData and its Newton iterate x1
// (newtonCost) with the eigenvalues of A − BK in eig, or an error when
// A − BK is not stable or its cost does not solve.
func (problem *riccatiProblem) closedLoopGain(continuous bool, xData []float64, eig []complex128) (*mat.Dense, []float64, error) {
	K, err := problem.gain(continuous, xData)
	if err != nil {
		return nil, nil, err
	}
	x1, err := problem.newtonCost(continuous, K, xData)
	if err != nil {
		return nil, nil, err
	}
	problem.closedLoopEig(eig)
	return K, x1, nil
}

// closedLoopEig copies the eigenvalues of A − BK of the last newtonCost
// to eig.
func (problem *riccatiProblem) closedLoopEig(eig []complex128) {
	lw := problem.ws.lyap
	for i := range eig {
		eig[i] = complex(lw.wr[i], lw.wi[i])
	}
}

// newtonCost returns the Newton iterate X₁ = X + Δ of the iterate X
// (xData) of gain K of verifyRiccati, with the eigenvalues of A − BK
// (with E, of (A − BK, E)) in ws.lyap.wr, ws.lyap.wi; ErrNoStabilizing
// when they are not stable, or the error of the Lyapunov solve. With
// A_K = A − BK (E = I when absent),
//
//	continuous: A_KᵀΔE + EᵀΔA_K + Ric(X) = 0
//	discrete:   A_KᵀΔA_K − EᵀΔE + Ric(X) = 0
//
// solved with E as FᵀY + YF + Ric(X) = 0 or FᵀYF − Y + Ric(X) = 0 for
// F = E⁻¹A_K and Y = EᵀΔE,
// for the Riccati residual Ric(X) of riccatiResidual, formed in
// double-double arithmetic: in float64 its terms (‖A_K‖²‖X‖, KᵀRK under
// an ill-conditioned R, A − BK under cheap control) round it to a noise
// that the solve maps to a step of the size of the error it should
// measure, while Δ's own rounding is relative to the small ‖Δ‖.
func (problem *riccatiProblem) newtonCost(continuous bool, K *mat.Dense, xData []float64) ([]float64, error) {
	x1, err := problem.newtonSolve(continuous, K, xData, nil)
	if err != nil {
		return nil, err
	}
	n := problem.n
	if problem.E != nil {
		eLU, eIpiv := problem.descriptorLU()
		impl.Dgetrs(blas.Trans, n, n, eLU, n, eIpiv, x1, n)
		for i := range n {
			for j := i + 1; j < n; j++ {
				x1[i*n+j], x1[j*n+i] = x1[j*n+i], x1[i*n+j]
			}
		}
		impl.Dgetrs(blas.Trans, n, n, eLU, n, eIpiv, x1, n)
		symmetrize(x1, n, n)
	}
	for i, v := range xData {
		x1[i] += v
	}
	if !finiteSlice(x1) {
		return nil, ErrSingularEquation
	}
	return x1, nil
}

// newtonSolve returns the solution Δ (with E, Y = EᵀΔE) of newtonCost's
// Lyapunov (Stein) equation for the residual of riccatiResidual (of P = pe
// when not nil), with the eigenvalues of A − BK in ws.lyap.
func (problem *riccatiProblem) newtonSolve(continuous bool, K *mat.Dense, xData, pe []float64) ([]float64, error) {
	n, ws := problem.n, problem.ws
	if ws.lyap == nil {
		ws.lyap = &LyapunovWorkspace{}
	}
	aclT := ws.aWork[:n*n]
	res := ws.z[:n*n]
	problem.riccatiResidual(continuous, K.RawMatrix().Data, xData, pe, aclT, res)
	symmetrize(res, n, n)

	if problem.E != nil {
		eLU, eIpiv := problem.descriptorLU()
		f := ws.h[:n*n]
		for i := range n {
			for j := range n {
				f[j*n+i] = aclT[i*n+j]
			}
		}
		impl.Dgetrs(blas.NoTrans, n, n, eLU, n, eIpiv, f, n)
		for i := range n {
			for j := range n {
				aclT[j*n+i] = f[i*n+j]
			}
		}
	}

	solve, stable := dlyap, func(wr, wi float64) bool { return math.Hypot(wr, wi) < 1 }
	if continuous {
		solve, stable = lyap, func(wr, _ float64) bool { return wr < 0 }
	}
	D, err := solve(mat.NewDense(n, n, aclT), mat.NewDense(n, n, res), &LyapunovOpts{Workspace: ws.lyap})
	if err != nil && !errors.Is(err, ErrSingularEquation) {
		return nil, err
	}
	lw := ws.lyap
	for i := range n {
		if !stable(lw.wr[i], lw.wi[i]) {
			return nil, ErrNoStabilizing
		}
	}
	if err != nil {
		return nil, err
	}
	return D.RawMatrix().Data[:n*n], nil
}

// descriptorCancellation returns how far K = R⁻¹(BᵀXE + Sᵀ) and EᵀXE of
// the continuous descriptor X (xData) cancel: the larger of
// ‖|Bᵀ||X||E| + |Sᵀ|‖/‖BᵀXE + Sᵀ‖ and ‖|Eᵀ||X||E|‖/‖EᵀXE‖. There X ~
// 1/σ_min(E) in some direction and its rounding, which a Newton step
// relative to ‖X‖ does not see, moves both by up to that ratio times eps.
func (problem *riccatiProblem) descriptorCancellation(xData []float64) float64 {
	n, m := problem.n, problem.m
	nn := n * n
	ws := problem.ws
	if len(ws.cancel) < 5*nn+3*m*n {
		ws.cancel = make([]float64, 5*nn+3*m*n)
	}
	buf := ws.cancel
	absX, absE, xe, axe := buf[:nn], buf[nn:2*nn], buf[2*nn:3*nn], buf[3*nn:4*nn]
	ete, g, ag, absB := buf[4*nn:5*nn], buf[5*nn:5*nn+m*n], buf[5*nn+m*n:5*nn+2*m*n], buf[5*nn+2*m*n:5*nn+3*m*n]
	eRaw, bRaw := problem.E.RawMatrix(), problem.B.RawMatrix()
	gen := func(r, c int, d []float64, s int) blas64.General {
		return blas64.General{Rows: r, Cols: c, Data: d, Stride: s}
	}
	for i := range n {
		for j := range n {
			absX[i*n+j] = math.Abs(xData[i*n+j])
			absE[i*n+j] = math.Abs(eRaw.Data[i*eRaw.Stride+j])
		}
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, xData, n), gen(n, n, eRaw.Data, eRaw.Stride), 0, gen(n, n, xe, n))
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, absX, n), gen(n, n, absE, n), 0, gen(n, n, axe, n))
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, n, eRaw.Data, eRaw.Stride), gen(n, n, xe, n), 0, gen(n, n, ete, n))
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, n, absE, n), gen(n, n, axe, n), 0, gen(n, n, absX, n))
	ratio := cancelRatio(impl.Dlange(lapack.Frobenius, n, n, absX, n, nil), impl.Dlange(lapack.Frobenius, n, n, ete, n, nil))
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, m, bRaw.Data, bRaw.Stride), gen(n, n, xe, n), 0, gen(m, n, g, n))
	for i := range n {
		for j := range m {
			absB[i*m+j] = math.Abs(bRaw.Data[i*bRaw.Stride+j])
		}
	}
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, m, absB, m), gen(n, n, axe, n), 0, gen(m, n, ag, n))
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		for i := range m {
			for j := range n {
				s := sRaw.Data[j*sRaw.Stride+i]
				g[i*n+j] += s
				ag[i*n+j] += math.Abs(s)
			}
		}
	}
	return max(ratio, cancelRatio(impl.Dlange(lapack.Frobenius, m, n, ag, n, nil), impl.Dlange(lapack.Frobenius, m, n, g, n, nil)))
}

// cancelRatio returns abs/v, 0 when abs is 0, the cancellation of a
// product v whose terms sum to abs in magnitude.
func cancelRatio(abs, v float64) float64 {
	if abs == 0 {
		return 0
	}
	return abs / v
}

// explicitUncertified reports whether EᵀXE of a certified X cancels so
// far that the rounding of X alone can move it beyond newtonTol:
// ‖|Eᵀ||X||E|‖·eps > newtonTol·‖EᵀXE‖.
func explicitUncertified(X, E *mat.Dense) bool {
	var xe, ete, ax, aete mat.Dense
	xe.Mul(X, E)
	ete.Mul(E.T(), &xe)
	absE := absDense(E)
	ax.Mul(absDense(X), absE)
	aete.Mul(absE.T(), &ax)
	return !(cancelRatio(denseNorm(&aete), denseNorm(&ete))*eps() <= newtonTol)
}

// descriptorGainCertified reports whether the gain K of an X (xData)
// verifyRiccati certifies is certified with it, x1 the Newton iterate of
// X (also of an iterated X: the iterate kept after its successor's gain
// was unstable left Lqr's EᵀXE off by 3.5e-4 unseen), and records the Newton step of EᵀXE
// (gainStep) in problem.xeStep. With E, where EᵀXE (continuous, also K)
// cancels beyond nearSingularCancelLimit (descriptorCancellation,
// dareCancellation), or the Newton step moves EᵀXE (continuous, also K)
// beyond newtonRefineTol, K and EᵀXE are refined, continuous by
// refineDescriptorGain, discrete by refineDescriptorDare. Should that
// fail, X is certified only when the gain's ratio times eps and the
// step's move of K are within newtonTol, except for an EᵀXE that does not
// cancel: there X is certified as without refinement, Lqr's and Dlqr's
// EᵀXE lose Accuracy 0 (xeStep), and a continuous K the step moves beyond
// newtonTol is gainUncertified (Accuracy 1). Failing on them turned
// correct values into ErrNoStabilizing: the discrete gain ratio, with
// |W⁻¹| of a cheap-control W = R + BᵀXB, overstates the move of K by up
// to 1e16, and a continuous K moved 1.35e-8 was 5.7e-9 off the 320-bit
// gain (E's last pivot 4.4e-14), while one moved 3.8e-8 was 4.8e-8 off. A step of X within newtonTol may leave K off by far
// more: it is relative to ‖X‖ ~ 1/σ_min(E), and K sees X in E's small
// directions (Lqr K off by 9.2e-7, X by 1.7e-9).
func (problem *riccatiProblem) descriptorGainCertified(continuous bool, K *mat.Dense, xData, x1 []float64, eig []complex128) bool {
	problem.xe, problem.xeStep, problem.xeX, problem.xeLoose, problem.gainUncertified = nil, 0, 0, false, false
	if problem.E == nil {
		return true
	}
	var cE, cK float64
	if continuous {
		cE = problem.descriptorCancellation(xData)
		cK = cE
	} else {
		cE, cK = problem.dareCancellation(K, xData)
	}
	move, xeStep := 0.0, 0.0
	if x1 != nil {
		move, xeStep = problem.gainStep(continuous, K, xData, x1)
	}
	problem.xeStep = xeStep
	if cE <= nearSingularCancelLimit && xeStep <= newtonRefineTol && (!continuous || move <= newtonRefineTol) {
		return true
	}
	if continuous && problem.refineDescriptorGain(K, xData, eig) ||
		!continuous && problem.refineDescriptorDare(K, xData, eig) {
		problem.xeStep = 0
		return true
	}
	if cE <= nearSingularCancelLimit {
		problem.gainUncertified = continuous && move > newtonTol
		return true
	}
	return cK*eps() <= newtonTol && move <= newtonTol
}

// dareCancellation returns how far EᵀXE and the discrete gain K of the
// descriptor X (xData) cancel: ‖|Eᵀ||X||E|‖/‖EᵀXE‖ and
// ‖|W⁻¹||Bᵀ||X||A_K|‖/‖K‖, W = R + BᵀXB and A_K = A − BK, which bounds
// the move of K = W⁻¹(BᵀXA + Sᵀ) under the rounding of X in eps units
// (δK = W⁻¹BᵀδXA_K). There X ~ 1/σ_min(E)², and the double-double gain
// of the 320-bit X rounded to float64 was off by 4.9e-6 (Kalman, E's
// last pivot 1e-5).
func (problem *riccatiProblem) dareCancellation(K *mat.Dense, xData []float64) (cE, cK float64) {
	n, m := problem.n, problem.m
	nn := n * n
	ws := problem.ws
	if len(ws.cancel) < 6*nn+3*m*n+m*m {
		ws.cancel = make([]float64, 6*nn+3*m*n+m*m)
	}
	buf := ws.cancel
	absX, absE, xe, axe := buf[:nn], buf[nn:2*nn], buf[2*nn:3*nn], buf[3*nn:4*nn]
	ete, acl := buf[4*nn:5*nn], buf[5*nn:6*nn]
	bx, absB, g := buf[6*nn:6*nn+m*n], buf[6*nn+m*n:6*nn+2*m*n], buf[6*nn+2*m*n:6*nn+3*m*n]
	w := buf[6*nn+3*m*n : 6*nn+3*m*n+m*m]
	eRaw, bRaw, aRaw := problem.E.RawMatrix(), problem.B.RawMatrix(), problem.A.RawMatrix()
	kRaw := K.RawMatrix()
	gen := func(r, c int, d []float64, s int) blas64.General {
		return blas64.General{Rows: r, Cols: c, Data: d, Stride: s}
	}
	for i := range n {
		for j := range n {
			absX[i*n+j] = math.Abs(xData[i*n+j])
			absE[i*n+j] = math.Abs(eRaw.Data[i*eRaw.Stride+j])
		}
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, xData, n), gen(n, n, eRaw.Data, eRaw.Stride), 0, gen(n, n, xe, n))
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, absX, n), gen(n, n, absE, n), 0, gen(n, n, axe, n))
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, n, eRaw.Data, eRaw.Stride), gen(n, n, xe, n), 0, gen(n, n, ete, n))
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, n, absE, n), gen(n, n, axe, n), 0, gen(n, n, xe, n))
	cE = cancelRatio(impl.Dlange(lapack.Frobenius, n, n, xe, n, nil), impl.Dlange(lapack.Frobenius, n, n, ete, n, nil))

	copyStrided(acl, n, aRaw.Data, aRaw.Stride, n, n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, gen(n, m, bRaw.Data, bRaw.Stride), gen(m, n, kRaw.Data, kRaw.Stride), 1, gen(n, n, acl, n))
	for i := range acl {
		acl[i] = math.Abs(acl[i])
	}
	if !problem.dareWeight(xData, bx, w) || !impl.Dpotri(blas.Upper, m, w, m) {
		return cE, math.Inf(1)
	}
	for i := range m {
		for j := range i {
			w[i*m+j] = w[j*m+i]
		}
	}
	for i := range w {
		w[i] = math.Abs(w[i])
	}
	for i := range n {
		for j := range m {
			absB[j*n+i] = math.Abs(bRaw.Data[i*bRaw.Stride+j])
		}
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, absX, n), gen(n, n, acl, n), 0, gen(n, n, axe, n))
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(m, n, absB, n), gen(n, n, axe, n), 0, gen(m, n, g, n))
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(m, m, w, m), gen(m, n, g, n), 0, gen(m, n, bx, n))
	return cE, cancelRatio(impl.Dlange(lapack.Frobenius, m, n, bx, n, nil), impl.Dlange(lapack.Frobenius, m, n, kRaw.Data, kRaw.Stride, nil))
}

// dareWeight forms BᵀX in bx and the Cholesky factor (upper) of
// W = R + BᵀXB in w, false when W is not positive definite.
func (problem *riccatiProblem) dareWeight(xData, bx, w []float64) bool {
	n, m := problem.n, problem.m
	bRaw, rRaw := problem.B.RawMatrix(), problem.R.RawMatrix()
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
		0, blas64.General{Rows: m, Cols: n, Data: bx, Stride: n})
	copyStrided(w, m, rRaw.Data, rRaw.Stride, m, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: m, Cols: n, Data: bx, Stride: n},
		blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		1, blas64.General{Rows: m, Cols: m, Data: w, Stride: m})
	return impl.Dpotrf(blas.Upper, m, w, m)
}

// gainStep returns, in the caller's coordinates, ‖δK‖/‖K‖ for the move
// δK of the gain K of X (xData) to first order along the Newton step
// D = X₁ − X, R⁻¹BᵀDE continuous, W⁻¹BᵀDA_K discrete (dareWeight), and
// the step ‖EᵀDE‖/‖EᵀXE‖ of EᵀXE: relative to ‖X‖ ~ 1/σ_min(E) a step
// of 1.9e-9 left EᵀXE off by 2.6e-7.
func (problem *riccatiProblem) gainStep(continuous bool, K *mat.Dense, xData, x1 []float64) (move, xeStep float64) {
	n, m, t := problem.n, problem.m, problem.t
	nn := n * n
	ws := problem.ws
	if len(ws.gainStep) < 5*nn+2*m*n+m*m {
		ws.gainStep = make([]float64, 5*nn+2*m*n+m*m)
	}
	buf := ws.gainStep
	d, md, acl, de, xe := buf[:nn], buf[nn:2*nn], buf[2*nn:3*nn], buf[3*nn:4*nn], buf[4*nn:5*nn]
	g, bx, w := buf[5*nn:5*nn+m*n], buf[5*nn+m*n:5*nn+2*m*n], buf[5*nn+2*m*n:]
	for i := range d {
		d[i] = x1[i] - xData[i]
	}
	bRaw, kRaw, eRaw := problem.B.RawMatrix(), K.RawMatrix(), problem.E.RawMatrix()
	gen := func(r, c int, v []float64, s int) blas64.General {
		return blas64.General{Rows: r, Cols: c, Data: v, Stride: s}
	}
	wnorm := func(v []float64) float64 {
		sum := 0.0
		for i, x := range v {
			if t != nil {
				x *= t[i/n] * t[i%n]
			}
			sum += x * x
		}
		return math.Sqrt(sum)
	}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, d, n), gen(n, n, eRaw.Data, eRaw.Stride), 0, gen(n, n, de, n))
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, n, eRaw.Data, eRaw.Stride), gen(n, n, de, n), 0, gen(n, n, md, n))
	dn := wnorm(md)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, xData, n), gen(n, n, eRaw.Data, eRaw.Stride), 0, gen(n, n, de, n))
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, n, eRaw.Data, eRaw.Stride), gen(n, n, de, n), 0, gen(n, n, xe, n))
	xeStep = cancelRatio(dn, wnorm(xe))
	chol := ws.rChol[:m*m]
	if continuous {
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, d, n), gen(n, n, eRaw.Data, eRaw.Stride), 0, gen(n, n, md, n))
	} else {
		aRaw := problem.A.RawMatrix()
		copyStrided(acl, n, aRaw.Data, aRaw.Stride, n, n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, gen(n, m, bRaw.Data, bRaw.Stride), gen(m, n, kRaw.Data, kRaw.Stride), 1, gen(n, n, acl, n))
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(n, n, d, n), gen(n, n, acl, n), 0, gen(n, n, md, n))
		if !problem.dareWeight(xData, bx, w) {
			return math.Inf(1), xeStep
		}
		chol = w
	}
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, gen(n, m, bRaw.Data, bRaw.Stride), gen(n, n, md, n), 0, gen(m, n, g, n))
	impl.Dpotrs(blas.Upper, m, n, chol, m, g, n)
	norm := func(v []float64, s int) float64 {
		sum := 0.0
		for i := range m {
			for j := range n {
				x := v[i*s+j]
				if t != nil {
					x *= t[j]
				}
				sum += x * x
			}
		}
		return math.Sqrt(sum)
	}
	return cancelRatio(norm(g, n), norm(kRaw.Data, kRaw.Stride)), xeStep
}

// refineDescriptorDare replaces the gain K of the certified discrete
// descriptor X (xData) by the double-double gain (ddGainOf) of X refined
// by Hewer's method in double-double, from the residual (riccatiResidual)
// of X itself, and saves EᵀXE of that X in problem.xe, when the
// iteration converges: steps ‖Δ‖/‖X‖, ‖EᵀΔE‖/‖EᵀXE‖ and ‖K₊ − K‖/‖K₊‖
// within newtonRefineTol, with ‖X − X_dd‖/‖X‖ in problem.xeX, the larger
// in the Frobenius and 1-norm. As in idare, K and EᵀXE are then those of the
// solution, which the float64 X cannot carry (dareCancellation). X keeps
// its certificate: replaced by the rounded refined X, a cheap-control
// Dare with reduced inputs no longer certified it. It reports false, K
// and eig unchanged, when the iteration fails or stalls. eig receives
// the eigenvalues of A − BK of the last step.
func (problem *riccatiProblem) refineDescriptorDare(K *mat.Dense, xData []float64, eig []complex128) bool {
	n, m, ws, t := problem.n, problem.m, problem.ws, problem.t
	nn := n * n
	if len(ws.refine) < 6*nn {
		ws.refine = make([]float64, 6*nn)
	}
	buf := ws.refine
	pe, xe, ete := buf[:2*nn], buf[2*nn:4*nn], buf[4*nn:6*nn]
	clear(pe)
	xH, xL := pe[:nn], pe[nn:]
	copy(xH, xData)
	gain := func() *mat.Dense {
		k, _, ok := ddGainOf(problem.A, problem.B, problem.R, problem.S, xH, xL, n)
		if !ok {
			return nil
		}
		return k
	}
	norm := func(d []float64, both bool) float64 {
		s := 0.0
		for i, v := range d {
			if t != nil {
				v *= t[i%n]
				if both {
					v *= t[i/n]
				}
			}
			s += v * v
		}
		return math.Sqrt(s)
	}
	colNorm := func(d []float64) float64 {
		s := 0.0
		for j := range n {
			c := 0.0
			for i := range n {
				v := d[i*n+j]
				if t != nil {
					v *= t[i] * t[j]
				}
				c += math.Abs(v)
			}
			s = max(s, c)
		}
		return s
	}
	kd := gain()
	if kd == nil {
		return false
	}
	eLU, eIpiv := problem.descriptorLU()
	eRaw := problem.E.RawMatrix()
	xeH, xeL, eH, eL := xe[:nn], xe[nn:], ete[:nn], ete[nn:]
	best, stalls := math.Inf(1), 0
	for range newtonSteps {
		y, err := problem.newtonSolve(false, kd, xH, pe)
		if err != nil {
			return false
		}
		yn := norm(y, true)
		impl.Dgetrs(blas.Trans, n, n, eLU, n, eIpiv, y, n)
		for i := range n {
			for j := i + 1; j < n; j++ {
				y[i*n+j], y[j*n+i] = y[j*n+i], y[i*n+j]
			}
		}
		impl.Dgetrs(blas.Trans, n, n, eLU, n, eIpiv, y, n)
		symmetrize(y, n, n)
		if !finiteSlice(y) {
			return false
		}
		dn := norm(y, true)
		for i, v := range y {
			ddAdd(&xH[i], &xL[i], v, 0)
		}
		ddNormalize(xH, xL)
		kn := gain()
		if kn == nil {
			return false
		}
		clear(xe)
		clear(ete)
		for _, x := range [][]float64{xH, xL} {
			ddMulAcc(n, n, n, 1, x, n, 1, eRaw.Data, eRaw.Stride, 1, xeH, xeL, n)
		}
		ddNormalize(xeH, xeL)
		ddMulAcc(n, n, n, 1, eRaw.Data, 1, eRaw.Stride, xeH, n, 1, eH, eL, n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: xeL, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: eL, Stride: n})
		for i := range eH {
			eH[i] += eL[i]
		}
		kr, nr := kd.RawMatrix().Data, kn.RawMatrix().Data
		for i, v := range nr {
			kr[i] -= v
		}
		step := max(cancelRatio(dn, norm(xH, true)), cancelRatio(yn, norm(eH, true)),
			cancelRatio(norm(kr, false), norm(nr, false)))
		kd = kn
		if step <= newtonRefineTol {
			for i, v := range xData {
				y[i] = xH[i] - v + xL[i]
			}
			problem.xeX = max(cancelRatio(norm(y, true), norm(xH, true)), cancelRatio(colNorm(y), colNorm(xH)))
			problem.xeLoose = true
			if y, err := problem.newtonSolve(false, kn, xH, pe); err == nil {
				problem.xeLoose = !(cancelRatio(norm(y, true), norm(eH, true)) <= newtonRefineTol)
			}
			symmetrize(eH, n, n)
			copy(K.RawMatrix().Data[:m*n], nr)
			problem.closedLoopEig(eig)
			problem.xe = append(ws.xe[:0], eH...)
			ws.xe = problem.xe
			return true
		}
		if step < best {
			best, stalls = step, 0
		} else if stalls++; stalls == 3 {
			return false
		}
	}
	return false
}

// refineDescriptorGain replaces the gain K of the certified continuous
// descriptor X (xData) by that of P = XE refined by Newton's method in
// double-double, and saves EᵀXE = EᵀP in problem.xe, when the iteration
// converges: steps ‖Y‖/‖EᵀP‖ and ‖K₊ − K‖/‖K₊‖ within newtonRefineTol,
// for Y = EᵀΔE of newtonSolve from the residual of P itself and
// P₊ = P + E⁻ᵀY. K and EᵀXE are icare's; formed from the rounded X they
// lose up to descriptorCancellation times eps (Kalman L off by
// 4.6e-4 at cond(E) 1e13 with X exact), which the Newton step of X,
// relative to ‖X‖ ~ 1/σ_min(E), does not see. The residual of P has no
// such rounding and each step contracts by about cond(E)·eps. It reports
// false, K and eig unchanged, when the iteration fails or stalls. eig
// receives the eigenvalues of A − BK of the last step.
func (problem *riccatiProblem) refineDescriptorGain(K *mat.Dense, xData []float64, eig []complex128) bool {
	n, m, ws, t := problem.n, problem.m, problem.ws, problem.t
	nn := n * n
	if len(ws.refine) < 3*nn+3*m*n {
		ws.refine = make([]float64, 3*nn+3*m*n)
	}
	buf := ws.refine
	pe, ep := buf[:2*nn], buf[2*nn:3*nn]
	cur, next, gL := buf[3*nn:3*nn+m*n], buf[3*nn+m*n:3*nn+2*m*n], buf[3*nn+2*m*n:3*nn+3*m*n]
	pH, pL := pe[:nn], pe[nn:]
	clear(pe)
	eRaw, bRaw := problem.E.RawMatrix(), problem.B.RawMatrix()
	ddMulAcc(n, n, n, 1, xData, n, 1, eRaw.Data, eRaw.Stride, 1, pH, pL, n)
	ddNormalize(pH, pL)
	gain := func(k []float64) {
		clear(k)
		clear(gL)
		ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, pH, n, 1, k, gL, n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: pL, Stride: n},
			1, blas64.General{Rows: m, Cols: n, Data: gL, Stride: n})
		if problem.S != nil {
			sRaw := problem.S.RawMatrix()
			for i := range m {
				for j := range n {
					ddAdd(&k[i*n+j], &gL[i*n+j], sRaw.Data[j*sRaw.Stride+i], 0)
				}
			}
		}
		for i := range k {
			k[i] += gL[i]
		}
		impl.Dpotrs(blas.Upper, m, n, ws.rChol[:m*m], m, k, n)
	}
	norm := func(d []float64, both bool) float64 {
		s := 0.0
		for i, v := range d {
			if t != nil {
				v *= t[i%n]
				if both {
					v *= t[i/n]
				}
			}
			s += v * v
		}
		return math.Sqrt(s)
	}
	gain(cur)
	kd := mat.NewDense(m, n, cur)
	eLU, eIpiv := problem.descriptorLU()
	best, stalls := math.Inf(1), 0
	for range newtonSteps {
		y, err := problem.newtonSolve(true, kd, xData, pe)
		if err != nil {
			return false
		}
		yn := norm(y, true)
		impl.Dgetrs(blas.Trans, n, n, eLU, n, eIpiv, y, n)
		for i, v := range y {
			ddAdd(&pH[i], &pL[i], v, 0)
		}
		ddNormalize(pH, pL)
		gain(next)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: pH, Stride: n},
			0, blas64.General{Rows: n, Cols: n, Data: ep, Stride: n})
		step := cancelRatio(yn, norm(ep, true))
		for i, v := range next {
			cur[i] -= v
		}
		step = max(step, cancelRatio(norm(cur, false), norm(next, false)))
		copy(cur, next)
		if !finiteSlice(cur) {
			return false
		}
		if step <= newtonRefineTol {
			blas64.Gemm(blas.Trans, blas.NoTrans, 1,
				blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
				blas64.General{Rows: n, Cols: n, Data: pL, Stride: n},
				1, blas64.General{Rows: n, Cols: n, Data: ep, Stride: n})
			symmetrize(ep, n, n)
			copy(K.RawMatrix().Data[:m*n], cur)
			problem.closedLoopEig(eig)
			problem.xe = append(ws.xe[:0], ep...)
			ws.xe = problem.xe
			return true
		}
		if step < best {
			best, stalls = step, 0
		} else if stalls++; stalls == 3 {
			return false
		}
	}
	return false
}

// descriptorLU returns the LU factors of E (row-major, with pivots) in
// ws.eLU and ws.eIpiv, factored on first use.
func (problem *riccatiProblem) descriptorLU() ([]float64, []int) {
	n, ws := problem.n, problem.ws
	if !problem.eFactored {
		if len(ws.eLU) < n*n {
			ws.eLU, ws.eIpiv = make([]float64, n*n), make([]int, n)
		}
		eRaw := problem.E.RawMatrix()
		copyStrided(ws.eLU[:n*n], n, eRaw.Data, eRaw.Stride, n, n)
		impl.Dgetrf(n, n, ws.eLU[:n*n], n, ws.eIpiv[:n])
		problem.eFactored = true
	}
	return ws.eLU[:n*n], ws.eIpiv[:n]
}

// riccatiResidual writes (A − BK)ᵀ to aclT and the Riccati residual of
// xData to res, rounded from double-double sums of exact products
// (ddMulAcc). It forms the cost residual of xData under the gain kData,
//
//	continuous: A_KᵀXE + EᵀXA_K + Q_K
//	discrete:   A_KᵀXA_K − EᵀXE + Q_K
//
// (E = I when absent) with Q_K = Q + KᵀRK − SK − KᵀSᵀ, and subtracts its
// excess over the K-free residual, GᵀW⁻¹G with G = BᵀXE + Sᵀ − RK, W = R
// (continuous) or G = BᵀXA_K + Sᵀ − RK, W = R + BᵀXB (discrete), so that
// the rounding of the gain drops out. It uses ws.h, ws.btx, ws.costV,
// ws.gainRes, ws.rbar and, with E, ws.eRes as scratch. A non-nil pe
// (with E) is, continuous, P = XE in double-double (hi then lo, 2n²),
// which replaces XE and, as PᵀA_K, EᵀXA_K: the residual of P itself,
// free of the rounding of X ~ 1/σ_min(E); discrete, X itself in
// double-double, which replaces xData.
func (problem *riccatiProblem) riccatiResidual(continuous bool, kData, xData, pe, aclT, res []float64) {
	n, m, ws := problem.n, problem.m, problem.ws
	aRaw, bRaw, qRaw, rRaw := problem.A.RawMatrix(), problem.B.RawMatrix(), problem.Q.RawMatrix(), problem.R.RawMatrix()
	nn := n * n
	acH, acL := ws.h[:nn], ws.h[nn:2*nn]
	yH, yL := ws.h[2*nn:3*nn], ws.h[3*nn:4*nn]
	copyStrided(acH, n, aRaw.Data, aRaw.Stride, n, n)
	clear(acL)
	ddMulAcc(n, n, m, -1, bRaw.Data, bRaw.Stride, 1, kData, n, 1, acH, acL, n)
	ddNormalize(acH, acL)
	for i := range n {
		for j := range n {
			aclT[j*n+i] = acH[i*n+j]
		}
	}

	var xLo []float64
	if pe != nil && !continuous {
		xData, xLo, pe = pe[:nn], pe[nn:2*nn], nil
	}
	clear(yH)
	clear(yL)
	if pe == nil {
		ddMulAcc(n, n, n, 1, xData, n, 1, acH, n, 1, yH, yL, n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			1, blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
			blas64.General{Rows: n, Cols: n, Data: acL, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: yL, Stride: n})
		if xLo != nil {
			blas64.Gemm(blas.NoTrans, blas.NoTrans,
				1, blas64.General{Rows: n, Cols: n, Data: xLo, Stride: n},
				blas64.General{Rows: n, Cols: n, Data: acH, Stride: n},
				1, blas64.General{Rows: n, Cols: n, Data: yL, Stride: n})
		}
	}

	var eRaw blas64.General
	var xeH, xeL []float64
	if pe != nil {
		xeH, xeL = pe[:nn], pe[nn:2*nn]
		ddMulAcc(n, n, n, 1, xeH, 1, n, acH, n, 1, yH, yL, n)
		for _, f := range [][2][]float64{{xeH, acL}, {xeL, acH}} {
			blas64.Gemm(blas.Trans, blas.NoTrans,
				1, blas64.General{Rows: n, Cols: n, Data: f[0], Stride: n},
				blas64.General{Rows: n, Cols: n, Data: f[1], Stride: n},
				1, blas64.General{Rows: n, Cols: n, Data: yL, Stride: n})
		}
	} else if problem.E != nil {
		eRaw = problem.E.RawMatrix()
		if len(ws.eRes) < 4*nn {
			ws.eRes = make([]float64, 4*nn)
		}
		xeH, xeL = ws.eRes[2*nn:3*nn], ws.eRes[3*nn:4*nn]
		clear(xeH)
		clear(xeL)
		ddMulAcc(n, n, n, 1, xData, n, 1, eRaw.Data, eRaw.Stride, 1, xeH, xeL, n)
		if xLo != nil {
			blas64.Gemm(blas.NoTrans, blas.NoTrans,
				1, blas64.General{Rows: n, Cols: n, Data: xLo, Stride: n},
				blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
				1, blas64.General{Rows: n, Cols: n, Data: xeL, Stride: n})
		}
	}

	resL := ws.z[nn : 2*nn]
	copyStrided(res, n, qRaw.Data, qRaw.Stride, n, n)
	clear(resL)
	if continuous {
		zH, zL := yH, yL
		if problem.E != nil && pe == nil {
			zH, zL = ws.eRes[:nn], ws.eRes[nn:2*nn]
			clear(zH)
			clear(zL)
			ddMulAcc(n, n, n, 1, eRaw.Data, 1, eRaw.Stride, yH, n, 1, zH, zL, n)
			blas64.Gemm(blas.Trans, blas.NoTrans, 1,
				blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
				blas64.General{Rows: n, Cols: n, Data: yL, Stride: n},
				1, blas64.General{Rows: n, Cols: n, Data: zL, Stride: n})
		}
		for i := range n {
			for j := range n {
				ddAdd(&res[i*n+j], &resL[i*n+j], zH[i*n+j], zL[i*n+j])
				ddAdd(&res[i*n+j], &resL[i*n+j], zH[j*n+i], zL[j*n+i])
			}
		}
	} else {
		if problem.E == nil {
			for i, v := range xData {
				ddAdd(&res[i], &resL[i], -v, 0)
			}
		} else {
			ddMulAcc(n, n, n, -1, eRaw.Data, 1, eRaw.Stride, xeH, n, 1, res, resL, n)
			blas64.Gemm(blas.Trans, blas.NoTrans, -1,
				blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
				blas64.General{Rows: n, Cols: n, Data: xeL, Stride: n},
				1, blas64.General{Rows: n, Cols: n, Data: resL, Stride: n})
		}
		ddMulAcc(n, n, n, 1, acH, 1, n, yH, n, 1, res, resL, n)
		ac := blas64.General{Rows: n, Cols: n, Data: acH, Stride: n}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, ac,
			blas64.General{Rows: n, Cols: n, Data: yL, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: resL, Stride: n})
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: n, Data: acL, Stride: n},
			blas64.General{Rows: n, Cols: n, Data: yH, Stride: n},
			1, blas64.General{Rows: n, Cols: n, Data: resL, Stride: n})
	}

	rkH, rkL := ws.btx[:m*n], ws.costV[:m*n]
	clear(rkH)
	clear(rkL)
	ddMulAcc(m, n, m, 1, rRaw.Data, rRaw.Stride, 1, kData, n, 1, rkH, rkL, n)
	ddMulAcc(n, n, m, 1, kData, 1, n, rkH, n, 1, res, resL, n)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		1, blas64.General{Rows: m, Cols: n, Data: kData, Stride: n},
		blas64.General{Rows: m, Cols: n, Data: rkL, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: resL, Stride: n})
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		ddMulAcc(n, n, m, -1, sRaw.Data, sRaw.Stride, 1, kData, n, 1, res, resL, n)
		ddMulAcc(n, n, m, -1, kData, 1, n, sRaw.Data, 1, sRaw.Stride, res, resL, n)
	}
	for i := range res {
		res[i] += resL[i]
	}

	gH, gL := ws.gainRes[:m*n], ws.gainRes[m*n:2*m*n]
	for i := range gH {
		gH[i], gL[i] = -rkH[i], -rkL[i]
	}
	if problem.S != nil {
		sRaw := problem.S.RawMatrix()
		for i := range m {
			for j := range n {
				ddAdd(&gH[i*n+j], &gL[i*n+j], sRaw.Data[j*sRaw.Stride+i], 0)
			}
		}
	}
	w := ws.rbar[:m*m]
	if continuous {
		if problem.E == nil {
			ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, xData, n, 1, gH, gL, n)
		} else {
			ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, xeH, n, 1, gH, gL, n)
			blas64.Gemm(blas.Trans, blas.NoTrans, 1,
				blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
				blas64.General{Rows: n, Cols: n, Data: xeL, Stride: n},
				1, blas64.General{Rows: m, Cols: n, Data: gL, Stride: n})
		}
		copy(w, ws.rChol[:m*m])
	} else {
		ddMulAcc(m, n, n, 1, bRaw.Data, 1, bRaw.Stride, yH, n, 1, gH, gL, n)
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: yL, Stride: n},
			1, blas64.General{Rows: m, Cols: n, Data: gL, Stride: n})
		bx := ws.btx[:m*n]
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: xData, Stride: n},
			0, blas64.General{Rows: m, Cols: n, Data: bx, Stride: n})
		copyStrided(w, m, rRaw.Data, rRaw.Stride, m, m)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: m, Cols: n, Data: bx, Stride: n},
			blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
			1, blas64.General{Rows: m, Cols: m, Data: w, Stride: m})
		if !impl.Dpotrf(blas.Upper, m, w, m) {
			return
		}
	}
	for i := range gH {
		gH[i] += gL[i]
	}
	copy(gL, gH)
	impl.Dpotrs(blas.Upper, m, n, w, m, gL, n)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		-1, blas64.General{Rows: m, Cols: n, Data: gH, Stride: n},
		blas64.General{Rows: m, Cols: n, Data: gL, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: res, Stride: n})
}

// ddMulAcc adds alpha·A·B to the double-double matrix (hi, lo) (r×c,
// stride ldc), with A r×k at a[i*ai + p*ap] and B k×c at b[p*bp + j*bj],
// alpha ±1. Products are exact (FMA) and sums compensated (TwoSum).
func ddMulAcc(r, c, k int, alpha float64, a []float64, ai, ap int, b []float64, bp, bj int, hi, lo []float64, ldc int) {
	for i := range r {
		h, l := hi[i*ldc:i*ldc+c], lo[i*ldc:i*ldc+c]
		for p := range k {
			x := alpha * a[i*ai+p*ap]
			if x == 0 {
				continue
			}
			bRow := b[p*bp:]
			for j := range h {
				y := bRow[j*bj]
				// The conversion stops Go fusing x*y into the sum
				// below (arm64 FMA), which would lose pe.
				pr := float64(x * y)
				pe := math.FMA(x, y, -pr)
				sh := h[j]
				s := sh + pr
				bb := s - sh
				l[j] += (sh - (s - bb)) + (pr - bb) + pe
				h[j] = s
			}
		}
	}
}

// ddAdd adds the double-double (h, l) to (*hi, *lo).
func ddAdd(hi, lo *float64, h, l float64) {
	s := *hi + h
	bb := s - *hi
	*lo += (*hi - (s - bb)) + (h - bb) + l
	*hi = s
}

// ddNormalize rounds each hi[i] + lo[i] to hi[i] and leaves the rest in
// lo[i].
func ddNormalize(hi, lo []float64) {
	for i, h := range hi {
		s := h + lo[i]
		lo[i] -= s - h
		hi[i] = s
	}
}

// newtonTol bounds the first Newton step that keeps the given X in
// verifyRiccati, newtonRefineTol the step that ends an iteration, and
// newtonSteps its solves. Three steps without a new smallest one end it
// once within newtonGlobalTol; larger steps are Newton–Kleinman's global
// phase, which need not decrease.
//
// poorRcnd = eps bounds the rcnd of U11 of an Accuracy 1 X from below:
// under it U11 is numerically singular and X not finite (MATLAB
// icare/idare Report 2), its K so large that a float64-stable A − BK
// says nothing (Kalman fuzz e5d054af185f91b9 under FMA rounding: rcnd
// 1.5e-17, three exact closed-loop roots outside the unit circle).
// weakBStall's X has rcnd 8e-15.
const (
	poorRcnd        = 0x1p-52
	newtonTol       = 1e-8
	newtonRefineTol = 1e-10
	newtonGlobalTol = 1e-5
	newtonSteps     = 30
)

// riccatiNorms are the Frobenius norms of the pencil blocks that cost
// scaling balances: a of A, Ā = A − BR⁻¹Sᵀ (and E), q of Q̄ = Q − SR⁻¹Sᵀ,
// g of G = BR⁻¹Bᵀ. kappa is the error amplification of the given pencil's
// input columns [B; −S; R] in its QR, max(‖B‖, ‖S‖, ‖R‖)/min(‖B‖, ‖R‖)
// (1 for the Hamiltonian, which forms G).
type riccatiNorms struct{ a, q, g, kappa float64 }

// cost is the predicted error amplification eps·cost/sep of X, of norm x,
// for the cost scale sigma (see rescaleCost).
func (norms riccatiNorms) cost(x, sigma float64) float64 {
	return max(norms.a, sigma*norms.g, norms.q/sigma) * (x/sigma + sigma/x)
}

// candidates returns the cost scales near x and √(q/g), 0 when undefined.
func (norms riccatiNorms) candidates(x float64) [2]float64 {
	var c [2]float64
	if x > 0 && !math.IsInf(x, 0) {
		c[0] = pow2Near(x)
	}
	if norms.q > 0 && norms.g > 0 {
		c[1] = pow2Near(math.Sqrt(norms.q) / math.Sqrt(norms.g))
	}
	return c
}

// firstSigma returns the cost scale for a retry without a solution:
// √(q/g) (SLICOT SB02MD SCAL='G'), which minimizes max(σg, q/σ) and, when
// q·g ≤ a², is the geometric middle of the σ range where the pencil is
// balanced; 1 when undefined.
func (norms riccatiNorms) firstSigma() float64 {
	if c := norms.candidates(0)[1]; c > 0 {
		return c
	}
	return 1
}

// riccatiPass is a saved solution of an earlier cost scale.
type riccatiPass struct {
	x, wr, wi, beta  []float64
	rcnd, sigma, res float64
	solves           int
}

// rescaleCost picks the cost scale σ for another solve from the solution
// xData of this one, saving it in *prev. The problem (Q/σ, R/σ, S/σ) has
// the solution X/σ and the same K. The Schur method's relative error in X
// is about eps·max(a, σg, q/σ)·(‖X‖/σ + σ/‖X‖)·kappa/sep, smallest near
// σ = ‖X‖ (SLICOT SB02MD SCAL='G' uses σ = √(q/g), right only when
// X ≈ √(q/g)), so X keeps its relative accuracy when ‖X‖ is far below (or
// above) the Hamiltonian scale. A zero first X is below accuracy and
// estimated by q/a. Later solves use the balanced pencil (kappa 1). It
// reports false, leaving σ, when the predicted gain is under 1024 or after
// three solves. The estimate can overrate the gain (an ill-conditioned R),
// so *prev keeps the saved solution of smallest Riccati residual, which
// keepBest weighs against the last.
func (problem *riccatiProblem) rescaleCost(continuous bool, norms riccatiNorms, xData []float64, prev **riccatiPass, rcnd float64) bool {
	if !problem.costScaling || (*prev != nil && (*prev).solves >= 2) {
		return false
	}
	n := problem.n
	x := impl.Dlange(lapack.Frobenius, n, n, xData, n, nil)
	if x == 0 && *prev == nil {
		x = norms.q / norms.a
	}
	if !(x > 0) || math.IsInf(x, 0) {
		return false
	}
	cost := func(sigma float64) float64 { return norms.cost(x, sigma) }
	cur := problem.sigma
	best, bestCost := cur, cost(cur)
	curCost := bestCost
	if !problem.balanced {
		curCost *= norms.kappa
	}
	for _, c := range norms.candidates(x) {
		if c > 0 {
			if v := cost(c); v < bestCost {
				best, bestCost = c, v
			}
		}
	}
	if !(curCost >= 1024*bestCost) {
		return false
	}
	problem.savePass(prev, xData, rcnd, problem.residual(continuous, xData))
	problem.sigma, problem.balanced = best, true
	return true
}

// savePass counts a solve in *prev and saves its solution xData there
// unless the saved one has a smaller Riccati residual than res.
func (problem *riccatiProblem) savePass(prev **riccatiPass, xData []float64, rcnd, res float64) {
	if *prev == nil {
		*prev = &riccatiPass{}
	}
	p, ws, n := *prev, problem.ws, problem.n
	p.solves++
	if p.solves > 1 && !(res < p.res) {
		return
	}
	p.res = res
	p.x = append(p.x[:0], xData...)
	p.wr = append(p.wr[:0], ws.wr[:n]...)
	p.wi = append(p.wi[:0], ws.wi[:n]...)
	p.beta = append(p.beta[:0], ws.beta[:n]...)
	p.rcnd, p.sigma = rcnd, problem.sigma
}

// keepBest reports whether the saved pass prev has a smaller Riccati
// residual than xData, the last solution.
func (problem *riccatiProblem) keepBest(continuous bool, prev *riccatiPass, xData []float64) bool {
	return prev != nil && !(problem.residual(continuous, xData) < prev.res)
}

// residual returns the Riccati residual of residualTerms.
func (problem *riccatiProblem) residual(continuous bool, xData []float64) float64 {
	res, _ := problem.residualTerms(continuous, xData)
	return res
}

// residualTerms returns ‖Ric(X)‖_F of the state-scaled problem
//
//	continuous: AᵀXE + EᵀXA − (EᵀXB + S)R⁻¹(BᵀXE + Sᵀ) + Q
//	discrete:   AᵀXA − EᵀXE − (AᵀXB + S)(R + BᵀXB)⁻¹(BᵀXA + Sᵀ) + Q
//
// and the sum of the Frobenius norms of its four terms.
func (problem *riccatiProblem) residualTerms(continuous bool, xData []float64) (res, scale float64) {
	n := problem.n
	X := mat.NewDense(n, n, xData)
	var E mat.Matrix = eyeDense(n)
	if problem.E != nil {
		E = problem.E
	}
	A, B := problem.A, problem.B
	var xe, xa, ric, t, m, w mat.Dense
	xe.Mul(X, E)
	xa.Mul(X, A)
	r := mat.DenseCopyOf(problem.R)
	if continuous {
		ric.Mul(A.T(), &xe)
		t.Mul(E.T(), &xa)
		scale = mat.Norm(&ric, 2) + mat.Norm(&t, 2)
		ric.Add(&ric, &t)
		m.Mul(xe.T(), B)
	} else {
		ric.Mul(A.T(), &xa)
		t.Mul(E.T(), &xe)
		scale = mat.Norm(&ric, 2) + mat.Norm(&t, 2)
		ric.Sub(&ric, &t)
		m.Mul(xa.T(), B)
		var bxb, xb mat.Dense
		xb.Mul(X, B)
		bxb.Mul(B.T(), &xb)
		r.Add(r, &bxb)
	}
	if problem.S != nil {
		m.Add(&m, problem.S)
	}
	if err := w.Solve(r, m.T()); err != nil {
		if _, ok := err.(mat.Condition); !ok || !finiteSlice(w.RawMatrix().Data) {
			return math.Inf(1), math.Inf(1)
		}
	}
	var mw mat.Dense
	mw.Mul(&m, &w)
	scale += mat.Norm(&mw, 2) + mat.Norm(problem.Q, 2)
	ric.Sub(&ric, &mw)
	ric.Add(&ric, problem.Q)
	return mat.Norm(&ric, 2), scale
}

// restorePass restores the solution saved by rescaleCost.
func (problem *riccatiProblem) restorePass(p *riccatiPass) (*mat.Dense, []float64, float64) {
	n, ws := problem.n, problem.ws
	xData := ws.xData[:n*n]
	copy(xData, p.x)
	copy(ws.wr[:n], p.wr)
	copy(ws.wi[:n], p.wi)
	copy(ws.beta[:n], p.beta)
	problem.sigma, problem.balanced = p.sigma, false
	return mat.NewDense(n, n, xData), xData, p.rcnd
}

// pow2Near returns the power of two nearest v within [2⁻⁵⁰⁰, 2⁵⁰⁰].
func pow2Near(v float64) float64 {
	frac, e := math.Frexp(v)
	if frac < math.Sqrt2/2 {
		e--
	}
	return math.Ldexp(1, min(max(e, -500), 500))
}

func finiteSlice(v []float64) bool {
	for _, x := range v {
		if !isFinite(x) {
			return false
		}
	}
	return true
}

// pencilNorms returns the riccatiNorms of the extended pencil, with
// ‖U⁻ᵀBᵀ‖_F² ≥ ‖G‖₂ for g (R = UᵀU in ws.rChol, U⁻ᵀBᵀ in ws.rinvBt), and
// with S forms Ā in ws.g and Q̄ in ws.qWork.
func (problem *riccatiProblem) pencilNorms() riccatiNorms {
	n, m, ws := problem.n, problem.m, problem.ws
	aRaw := problem.A.RawMatrix()
	norms := riccatiNorms{a: impl.Dlange(lapack.Frobenius, n, n, aRaw.Data, aRaw.Stride, nil)}
	if problem.E != nil {
		eRaw := problem.E.RawMatrix()
		norms.a = max(norms.a, impl.Dlange(lapack.Frobenius, n, n, eRaw.Data, eRaw.Stride, nil))
	} else {
		norms.a = max(norms.a, math.Sqrt(float64(n)))
	}
	rChol := ws.rChol[:m*m]
	upper := blas64.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit, N: m, Data: rChol, Stride: m}
	w := ws.rinvBt[:m*n]
	bRaw := problem.B.RawMatrix()
	for i := range n {
		for j := range m {
			w[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
		}
	}
	blas64.Trsm(blas.Left, blas.Trans, 1, upper, blas64.General{Rows: m, Cols: n, Data: w, Stride: n})
	gn := impl.Dlange(lapack.Frobenius, m, n, w, n, nil)
	norms.g = gn * gn
	rRaw := problem.R.RawMatrix()
	bn := impl.Dlange(lapack.Frobenius, n, m, bRaw.Data, bRaw.Stride, nil)
	rn := impl.Dlange(lapack.Frobenius, m, m, rRaw.Data, rRaw.Stride, nil)
	norms.kappa = max(bn, rn) / min(bn, rn)
	qRaw := problem.Q.RawMatrix()
	if !(norms.kappa >= 1) || math.IsInf(norms.kappa, 0) {
		norms.kappa = 1
	}
	if problem.S == nil {
		norms.q = impl.Dlange(lapack.Frobenius, n, n, qRaw.Data, qRaw.Stride, nil)
		return norms
	}
	v := ws.rinvSt[:m*n]
	sRaw := problem.S.RawMatrix()
	for i := range n {
		for j := range m {
			v[j*n+i] = sRaw.Data[i*sRaw.Stride+j]
		}
	}
	blas64.Trsm(blas.Left, blas.Trans, 1, upper, blas64.General{Rows: m, Cols: n, Data: v, Stride: n})
	qbar := ws.qWork[:n*n]
	copyStrided(qbar, n, qRaw.Data, qRaw.Stride, n, n)
	blas64.Gemm(blas.Trans, blas.NoTrans,
		-1, blas64.General{Rows: m, Cols: n, Data: v, Stride: n},
		blas64.General{Rows: m, Cols: n, Data: v, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: qbar, Stride: n})
	norms.q = impl.Dlange(lapack.Frobenius, n, n, qbar, n, nil)
	blas64.Trsm(blas.Left, blas.NoTrans, 1, upper, blas64.General{Rows: m, Cols: n, Data: v, Stride: n})
	abar := ws.g[:n*n]
	copyStrided(abar, n, aRaw.Data, aRaw.Stride, n, n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans,
		-1, blas64.General{Rows: n, Cols: m, Data: bRaw.Data, Stride: bRaw.Stride},
		blas64.General{Rows: m, Cols: n, Data: v, Stride: n},
		1, blas64.General{Rows: n, Cols: n, Data: abar, Stride: n})
	norms.a = max(norms.a, impl.Dlange(lapack.Frobenius, n, n, abar, n, nil))
	norms.kappa = max(norms.kappa, impl.Dlange(lapack.Frobenius, n, m, sRaw.Data, sRaw.Stride, nil)/min(bn, rn))
	return norms
}

// riccatiPencil is the data of the extended pencil of
// generalizedStableSubspace: A, Q and S (S.Data nil for none), with Q and S
// divided by σ = 1/isigma and the input row [S' (B') R] multiplied by mu.
// A non-nil bt (m×n) replaces B and R by B̃ = btᵀ and I, the inputs ũ = Uu
// of R = UᵀU, which leaves the deflation of R well-conditioned.
type riccatiPencil struct {
	a, q, s    blas64.General
	bt         []float64
	isigma, mu float64
}

// pencil returns the pencil data: the given A, Q, S at first, else for
// the cost scale σ Ā, Q̄ of pencilNorms and no S, so that S/σ does not
// swamp B in the input columns, the normalized inputs B̃ = BU⁻¹ of
// pencilNorms, and the power of two mu ≈ ‖B̃‖/‖I‖ balancing μI against B̃.
func (problem *riccatiProblem) pencil() riccatiPencil {
	d := riccatiPencil{a: problem.A.RawMatrix(), q: problem.Q.RawMatrix(), isigma: 1, mu: 1}
	if !problem.balanced {
		if problem.S != nil {
			d.s = problem.S.RawMatrix()
		}
		return d
	}
	n, ws := problem.n, problem.ws
	d.isigma = 1 / problem.sigma
	if problem.S != nil {
		d.a = blas64.General{Rows: n, Cols: n, Data: ws.g[:n*n], Stride: n}
		d.q = blas64.General{Rows: n, Cols: n, Data: ws.qWork[:n*n], Stride: n}
	}
	m := problem.m
	d.bt = ws.rinvBt[:m*n]
	if bn := impl.Dlange(lapack.Frobenius, m, n, d.bt, n, nil); bn > 0 {
		d.mu = pow2Near(bn / math.Sqrt(float64(m)))
	}
	return d
}

// pencilSolution deflates the extended pencil and returns its stabilizing
// solution, solving again with a cost scale when rescaleCost predicts a
// more accurate X; a failed later solve keeps the earlier one.
//
// The given pencil can fail QZ or misplace eigenvalues at extreme cost
// scales, so a failed first solve is rescued, also under NoScaling, by
// solves from the cost scale of firstSigma. The rescued X is kept only
// with a Riccati residual within rescueTol of its terms (and, see
// verifyRiccati, a stable closed loop), else the first error is returned. Under
// NoScaling a first X with a residual above noScalingTol of its terms is
// rescued too, keeping the solution of smallest residual.
func (problem *riccatiProblem) pencilSolution(continuous bool) (subspace riccatiSubspace, X *mat.Dense, xData []float64, rcnd float64, err error) {
	var norms riccatiNorms
	var prev *riccatiPass
	rescue := func() {
		problem.resolved, problem.costScaling = true, true
		norms = problem.pencilNorms()
		problem.sigma, problem.balanced = norms.firstSigma(), true
	}
	for {
		subspace, err = problem.generalizedStableSubspace(continuous, problem.pencil())
		if err == nil {
			X, xData, rcnd, err = problem.stabilizingSolution(subspace.vectors)
		}
		if err == nil && !finiteSlice(xData) {
			err = ErrNoStabilizing
		}
		if err != nil && prev == nil {
			if problem.rescueErr != nil {
				return riccatiSubspace{}, nil, nil, 0, problem.rescueErr
			}
			problem.rescueErr = err
			rescue()
			continue
		}
		if err == nil && prev == nil && problem.rescueErr == nil {
			if !problem.costScaling {
				res, scale := problem.residualTerms(continuous, xData)
				if res <= noScalingTol*scale {
					return subspace, X, xData, rcnd, nil
				}
				problem.savePass(&prev, xData, rcnd, res)
				problem.keepCandidate(xData, rcnd)
				rescue()
				continue
			}
			norms = problem.pencilNorms()
		}
		if err == nil && problem.rescaleCost(continuous, norms, xData, &prev, rcnd) {
			problem.resolved = true
			problem.keepCandidate(xData, rcnd)
			continue
		}
		if err != nil || problem.keepBest(continuous, prev, xData) {
			if err == nil {
				problem.keepCandidate(xData, rcnd)
			}
			subspace, X, xData, rcnd = problem.restoredSolution(prev)
		}
		if problem.rescueErr != nil {
			if res, scale := problem.residualTerms(continuous, xData); !(res <= rescueTol*scale) {
				problem.keepCandidate(xData, rcnd)
				return riccatiSubspace{}, nil, nil, 0, problem.rescueErr
			}
		}
		return subspace, X, xData, rcnd, nil
	}
}

// keepCandidate saves the solution xData of a pencil solve in
// problem.candidates when collecting is set and another solve follows or
// replaces it.
func (problem *riccatiProblem) keepCandidate(xData []float64, rcnd float64) {
	if !problem.collecting {
		return
	}
	ws, n := problem.ws, problem.n
	problem.candidates = append(problem.candidates, riccatiPass{
		x:    append([]float64(nil), xData...),
		wr:   append([]float64(nil), ws.wr[:n]...),
		wi:   append([]float64(nil), ws.wi[:n]...),
		beta: append([]float64(nil), ws.beta[:n]...),
		rcnd: rcnd, sigma: problem.sigma,
	})
}

// rescueTol and noScalingTol bound the Riccati residual of pencilSolution
// relative to the sum of its terms: a rescued solution above rescueTol is
// rejected, and an unscaled one above noScalingTol is rescued.
const (
	rescueTol    = 1e-6
	noScalingTol = 1e-12
)

// stableGain returns the eigenvalues of A − BK (with E: E⁻¹(A − BK)) and
// reports whether they are stable.
func (problem *riccatiProblem) stableGain(continuous bool, K *mat.Dense) ([]complex128, bool) {
	var bk, ea mat.Dense
	bk.Mul(problem.B, K)
	acl := mat.NewDense(problem.n, problem.n, nil)
	acl.Sub(problem.A, &bk)
	if problem.E != nil {
		var lu mat.LU
		lu.Factorize(problem.E)
		if err := lu.SolveTo(&ea, false, acl); err != nil {
			if _, ok := err.(mat.Condition); !ok {
				return nil, false
			}
		}
		acl = &ea
	}
	var eig mat.Eigen
	if !eig.Factorize(acl, mat.EigenNone) {
		return nil, false
	}
	vals := eig.Values(nil)
	for _, v := range vals {
		if continuous && !(real(v) < 0) || !continuous && !(cmplx.Abs(v) < 1) {
			return nil, false
		}
	}
	return vals, true
}

// restoredSolution restores the saved pass prev as pencilSolution returns
// it.
func (problem *riccatiProblem) restoredSolution(prev *riccatiPass) (riccatiSubspace, *mat.Dense, []float64, float64) {
	X, xData, rcnd := problem.restorePass(prev)
	ws, nn := problem.ws, 2*problem.n
	return riccatiSubspace{alphaR: ws.wr[:nn], alphaI: ws.wi[:nn], beta: ws.beta[:nn]}, X, xData, rcnd
}

// stabilizingSolution returns X = σ·U21*(E*U11)⁻¹ from the stable subspace
// basis [U11; U21] of a pencil with 2n columns and cost scale σ.
func (problem *riccatiProblem) stabilizingSolution(vs []float64) (X *mat.Dense, xData []float64, rcnd float64, err error) {
	n, ws := problem.n, problem.ws
	u11 := ws.u11[:n*n]
	u21 := ws.u21[:n*n]
	copyStrided(u11, n, vs, 2*n, n, n)
	copyBlock(u21, n, 0, 0, vs, 2*n, n, 0, n, n)
	if problem.E != nil {
		eRaw := problem.E.RawMatrix()
		eu11 := ws.aWork[:n*n]
		blas64.Gemm(blas.NoTrans, blas.NoTrans,
			1, blas64.General{Rows: n, Cols: n, Data: eRaw.Data, Stride: eRaw.Stride},
			blas64.General{Rows: n, Cols: n, Data: u11, Stride: n},
			0, blas64.General{Rows: n, Cols: n, Data: eu11, Stride: n})
		u11 = eu11
	}

	work := ws.work
	anorm := impl.Dlange(lapack.MaxColumnSum, n, n, u11, n, work[:n])
	ipiv := ws.ipiv[:n]
	if !impl.Dgetrf(n, n, u11, n, ipiv) {
		return nil, nil, 0, ErrNoStabilizing
	}
	rcnd = impl.Dgecon(lapack.MaxColumnSum, n, u11, n, anorm, work[:4*n], ws.iwork[:n])

	xData = ws.xData[:n*n]
	for i := range n {
		for j := range n {
			xData[i*n+j] = u21[j*n+i]
		}
	}
	impl.Dgetrs(blas.Trans, n, n, u11, n, ipiv, xData, n)
	if problem.sigma != 1 {
		for i := range xData {
			xData[i] *= problem.sigma
		}
	}
	symmetrize(xData, n, n)
	return mat.NewDense(n, n, xData), xData, rcnd, nil
}

type riccatiSubspace struct {
	vectors []float64
	alphaR  []float64
	alphaI  []float64
	beta    []float64
}

// generalizedStableSubspace deflates the extended pencil
//
//	continuous: [A 0 B; -Q -A' -S; S' B' R] - λ[E 0 0; 0 E' 0; 0 0 0]
//	discrete:   [A 0 B; -Q E' -S; S' 0 R]   - λ[E 0 0; 0 A' 0; 0 -B' 0]
//
// to 2n×2n by a QR of its input columns and orders the stable eigenvalues
// first. E nil means I. With the cost scale of d it deflates the
// equivalent pencil diag(I, I/σ, μ)·(H − λJ)·diag(I, σI, I), and with d.bt
// that of the inputs ũ = Uu, diag(I, I, U⁻ᵀ)·(H − λJ)·diag(I, I, U⁻¹).
func (problem *riccatiProblem) generalizedStableSubspace(continuous bool, d riccatiPencil) (riccatiSubspace, error) {
	n, m, ws := problem.n, problem.m, problem.ws
	nn := 2 * n
	rows := nn + m
	isigma, mu := d.isigma, d.mu
	muSigma := mu / isigma

	hLeft := ws.pencilH[:rows*nn]
	jLeft := ws.pencilJ[:rows*nn]
	input := ws.pencilInput[:rows*m]
	clear(hLeft)
	clear(jLeft)
	clear(input)

	aRaw, qRaw, sRaw := d.a, d.q, d.s
	bRaw := problem.B.RawMatrix()
	rRaw := problem.R.RawMatrix()
	hasS := sRaw.Data != nil
	b := func(i, j int) float64 { return bRaw.Data[i*bRaw.Stride+j] }
	if d.bt != nil {
		b = func(i, j int) float64 { return d.bt[j*n+i] }
	}

	var eRaw blas64.General
	if problem.E != nil {
		eRaw = problem.E.RawMatrix()
	}
	e := func(i, j int) float64 {
		if problem.E != nil {
			return eRaw.Data[i*eRaw.Stride+j]
		}
		if i == j {
			return 1
		}
		return 0
	}
	for i := range n {
		for j := range n {
			hLeft[i*nn+j] = aRaw.Data[i*aRaw.Stride+j]
			hLeft[(n+i)*nn+j] = -isigma * qRaw.Data[i*qRaw.Stride+j]
			jLeft[i*nn+j] = e(i, j)
			if continuous {
				hLeft[(n+i)*nn+n+j] = -aRaw.Data[j*aRaw.Stride+i]
				jLeft[(n+i)*nn+n+j] = e(j, i)
			} else {
				hLeft[(n+i)*nn+n+j] = e(j, i)
				jLeft[(n+i)*nn+n+j] = aRaw.Data[j*aRaw.Stride+i]
			}
		}

		for j := range m {
			bij := b(i, j)
			input[i*m+j] = bij
			if hasS {
				input[(n+i)*m+j] = -isigma * sRaw.Data[i*sRaw.Stride+j]
				hLeft[(nn+j)*nn+i] = mu * sRaw.Data[i*sRaw.Stride+j]
			}
			if continuous {
				hLeft[(nn+j)*nn+n+i] = muSigma * bij
			} else {
				jLeft[(nn+j)*nn+n+i] = -muSigma * bij
			}
		}
	}
	for i := range m {
		if d.bt != nil {
			input[(nn+i)*m+i] = mu
			continue
		}
		for j := range m {
			input[(nn+i)*m+j] = mu * rRaw.Data[i*rRaw.Stride+j]
		}
	}

	if m != 0 {
		tau := ws.tau[:m]
		var qrQuery, applyQuery [1]float64
		impl.Dgeqrf(rows, m, nil, m, nil, qrQuery[:], -1)
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, m, nil, m, nil, nil, nn, applyQuery[:], -1)
		lwork := max(int(qrQuery[0]), int(applyQuery[0]))
		if len(ws.work) < lwork {
			ws.work = make([]float64, lwork)
		}
		impl.Dgeqrf(rows, m, input, m, tau, ws.work, lwork)
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, m, input, m, tau, hLeft, nn, ws.work, lwork)
		impl.Dormqr(blas.Left, blas.Trans, rows, nn, m, input, m, tau, jLeft, nn, ws.work, lwork)
	}

	h := ws.h[:nn*nn]
	j := ws.z[:nn*nn]
	copy(h, hLeft[m*nn:])
	copy(j, jLeft[m*nn:])

	alphaR := ws.wr[:nn]
	alphaI := ws.wi[:nn]
	beta := ws.beta[:nn]
	vectors := ws.vs[:nn*nn]
	bwork := ws.bwork[:nn]
	stable := func(alphaR, alphaI, beta float64) bool {
		return math.Hypot(alphaR, alphaI) < math.Abs(beta)
	}
	if continuous {
		stable = func(alphaR, _, beta float64) bool {
			return alphaR*beta < 0
		}
	}

	sdim, ok := stableDeflatingQZ(nn, h, j, hLeft[m*nn:], jLeft[m*nn:], stable, alphaR, alphaI, beta, vectors, &ws.work, bwork)
	if !ok {
		return riccatiSubspace{}, ErrSchurFailed
	}
	if sdim != n {
		return riccatiSubspace{}, ErrNoStabilizing
	}
	for i := range n {
		if beta[i] == 0 {
			return riccatiSubspace{}, ErrNoStabilizing
		}
	}

	return riccatiSubspace{
		vectors: vectors,
		alphaR:  alphaR,
		alphaI:  alphaI,
		beta:    beta,
	}, nil
}

// stableDeflatingQZ writes to vectors the right Schur vectors Z of the
// generalized Schur form of the nn×nn pencil (h, j), a copy of (h0, j0),
// ordered so that the sdim eigenvalues selected by stable lead, with the
// eigenvalues in alphaR, alphaI, beta; h and j are overwritten. A reorder
// that Dgges rejects (a borderline swap, Netlib DTGEX2 parity) is retried
// on the reciprocal pencil (j, h), whose deflating subspaces are the same,
// and kept only when its eigenvalues match the first run's one to one
// (qzRetryConsistent); ok is false otherwise.
func stableDeflatingQZ(nn int, h, j, h0, j0 []float64, stable func(ar, ai, b float64) bool,
	alphaR, alphaI, beta, vectors []float64, work *[]float64, bwork []bool) (sdim int, ok bool) {
	var workQuery [1]float64
	impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, stable,
		nn, h, nn, j, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, workQuery[:], -1, bwork)
	lwork := int(workQuery[0])
	if len(*work) < lwork {
		*work = make([]float64, lwork)
	}
	sdim, ok = impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, stable,
		nn, h, nn, j, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, *work, lwork, bwork)
	if ok {
		return sdim, true
	}
	first := make([]float64, 3*nn)
	copy(first, alphaR)
	copy(first[nn:], alphaI)
	copy(first[2*nn:], beta)
	copy(h, h0)
	copy(j, j0)
	reciprocal := func(ar, ai, b float64) bool { return stable(qzReciprocal(ar, ai, b)) }
	sdim, ok = impl.Dgges(lapack.SchurNone, lapack.SchurHess, lapack.SortSelected, reciprocal,
		nn, j, nn, h, nn, alphaR, alphaI, beta, nil, 1, vectors, nn, *work, lwork, bwork)
	for i := range nn {
		alphaR[i], alphaI[i], beta[i] = qzReciprocal(alphaR[i], alphaI[i], beta[i])
	}
	return sdim, ok && qzRetryConsistent(first, alphaR, alphaI, beta, stable)
}
