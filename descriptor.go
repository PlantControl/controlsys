package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

func copyDescriptorE(E *mat.Dense) *mat.Dense {
	if E == nil {
		return nil
	}
	return mat.DenseCopyOf(E)
}

type descriptorPolicy struct {
	E *mat.Dense
}

func newDescriptorPolicy(sys *System) descriptorPolicy {
	if sys == nil {
		return descriptorPolicy{}
	}
	return descriptorPolicy{E: sys.E}
}

func (p descriptorPolicy) isDescriptor() bool {
	return p.E != nil && !isIdentityDescriptor(p.E)
}

func (p descriptorPolicy) validate(n int) error {
	if p.E == nil {
		return nil
	}
	er, ec := p.E.Dims()
	if er != n || ec != n {
		return fmt.Errorf("descriptor matrix E is %d×%d, want %d×%d: %w", er, ec, n, n, ErrDimensionMismatch)
	}
	return nil
}

// poles returns the eigenvalues of A or of the pencil (A, E). Non-finite
// entries are rejected first: LAPACK balancing (Dgebal) never terminates on
// NaN.
func (p descriptorPolicy) poles(op string, A *mat.Dense, n int) ([]complex128, error) {
	if err := requireFiniteDense(op, "A", A); err != nil {
		return nil, err
	}
	if p.E != nil && !isIdentityDescriptor(p.E) {
		if err := requireFiniteDense(op, "E", p.E); err != nil {
			return nil, err
		}
		poles, err := generalizedPoles(A, p.E, n)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		return poles, nil
	}
	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenNone) {
		return nil, fmt.Errorf("%s: eigenvalue decomposition failed to converge: %w", op, ErrSchurFailed)
	}
	return eig.Values(nil), nil
}

func (p descriptorPolicy) requireStandard(context string) error {
	if p.E == nil || isIdentityDescriptor(p.E) {
		return nil
	}
	return fmt.Errorf("%s: %w", context, ErrDescriptorUnsupported)
}

// normDescriptorCondLimit is the largest 1-norm condition estimate of E for
// which the norm routines fold E into the state equation; beyond 1/√eps the
// explicit model E⁻¹A, E⁻¹B can lose half the significant digits.
var normDescriptorCondLimit = 1 / math.Sqrt(eps())

// normExplicitForm returns sys with a nonsingular E folded in, (E⁻¹A, E⁻¹B,
// C, D), which keeps the transfer function and the poles. Singular E and E
// with condition above normDescriptorCondLimit return ErrDescriptorUnsupported
// wrapping ErrDescriptorSingular.
func normExplicitForm(sys *System, context string) (*System, error) {
	if !sys.IsDescriptor() {
		return sys, nil
	}
	if err := requireFiniteDense(context, "E", sys.E); err != nil {
		return nil, err
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if cond := lu.Cond(); luNearSingular(&lu) || !(cond <= normDescriptorCondLimit) {
		return nil, fmt.Errorf("%s: cond(E) ≈ %.3g exceeds %.3g: %w: %w",
			context, cond, normDescriptorCondLimit, ErrDescriptorUnsupported, ErrDescriptorSingular)
	}
	out, err := sys.ToExplicit()
	if err != nil {
		return nil, fmt.Errorf("%s: %w: %w", context, ErrDescriptorUnsupported, err)
	}
	return out, nil
}

// requireNonsingular rejects a singular E, as MATLAB lqr, lqi, icare and
// idare require nonsingular E.
func (p descriptorPolicy) requireNonsingular(context string) error {
	if !p.isDescriptor() {
		return nil
	}
	var lu mat.LU
	lu.Factorize(p.E)
	if luNearSingular(&lu) {
		return fmt.Errorf("%s: %w", context, ErrDescriptorSingular)
	}
	return nil
}

// augmentedDescriptor returns blkdiag(E, I_extra), or nil when E is nil or
// the identity.
func augmentedDescriptor(E *mat.Dense, extra int) *mat.Dense {
	if isIdentityDescriptor(E) {
		return nil
	}
	n, _ := E.Dims()
	out := mat.NewDense(n+extra, n+extra, nil)
	setBlockOrIdentity(out, 0, E, n)
	for i := n; i < n+extra; i++ {
		out.Set(i, i, 1)
	}
	return out
}

func (p descriptorPolicy) requireRiccatiStandard(context string) error {
	if p.E == nil || isIdentityDescriptor(p.E) {
		return nil
	}
	return fmt.Errorf("%s: %w", context, ErrDescriptorRiccati)
}

func isIdentityDescriptor(E *mat.Dense) bool {
	if E == nil {
		return true
	}
	r, c := E.Dims()
	if r != c {
		return false
	}
	raw := E.RawMatrix()
	for i := range r {
		for j := range c {
			want := 0.0
			if i == j {
				want = 1
			}
			if raw.Data[i*raw.Stride+j] != want {
				return false
			}
		}
	}
	return true
}

// generalizedPoles computes the finite eigenvalues of the pencil (A, E),
// leaving out the infinite ones as MATLAB pole does. A nonsingular E (MATLAB
// rank) keeps every eigenvalue with β ≠ 0. For a singular E of rank r the
// n − r eigenvalues of least |β|/|α| of the whole-pencil QZ are infinite
// when a gap separates them (leastRatioInfTol), no large one remains and
// the pencil has index 1 (pencilIndexOne): a chain of length k has k
// infinite eigenvalues but lowers rank(E) by one. When E has a stiff
// singular value just above the rounding this is the classification,
// with β = 0 alone infinite if no gap shows or the pencil has index ≥ 2:
// its chain members QZ leaves at |β|/|α| ~ ε^(1/k), above the stiff finite pole's.
// Otherwise the infinite structure is deflated first (deflateInfinite),
// since QZ resolves an infinite Jordan chain of length k only to
// |β|/|α| ~ ε^(1/k), and the poles are the QZ eigenvalues of the finite
// block. When the deflation is ambiguous, |β| ≤ 100·n·ε·|α|·‖E‖/‖A‖ is
// infinite, and a large remaining pole is dropped as well when
// certifyFiniteRetained proves it one of the n − rank(E) infinite
// eigenvalues of an index-1 pencil; one it cannot classify is kept.
func generalizedPoles(A, E *mat.Dense, n int) ([]complex128, error) {
	aRaw, eRaw := A.RawMatrix(), E.RawMatrix()
	alphar, alphai, beta, ok := qzEigenvalues(n, aRaw.Data, aRaw.Stride, eRaw.Data, eRaw.Stride)
	if !ok {
		return nil, fmt.Errorf("generalized eigenvalue decomposition failed: %w", ErrSchurFailed)
	}
	normA, normE := mat.Norm(A, 2), mat.Norm(E, 2)
	if n == 0 || clearlyNonsingular(eRaw.Data, eRaw.Stride, n, make([]float64, n*n), qzInfiniteGap*1e3*float64(n)*eps()*normE) {
		return finitePoles(alphar, alphai, beta, 0), nil
	}
	var svd mat.SVD
	if svd.Factorize(E, mat.SVDNone) {
		sv := svd.Values(nil)
		r, stiff, _ := descriptorRank(sv, sv[0], n, true)
		if r == n {
			return finitePoles(alphar, alphai, beta, 0), nil
		}
		margin := 0.0
		if stiff {
			margin = sv[r-1] / (float64(n) * eps() * sv[0])
		}
		tol, sep := leastRatioInfTol(alphar, alphai, beta, n-r, margin)
		if stiff {
			if _, index1 := pencilIndexOne(A, E, n, normA); !index1 {
				tol = 0
			}
			return finitePoles(alphar, alphai, beta, tol), nil
		}
		if sep && !largeRetained(alphar, alphai, beta, tol, qzInfiniteTol(qzLargeRatio, normA, normE)) {
			if _, index1 := pencilIndexOne(A, E, n, normA); index1 {
				return finitePoles(alphar, alphai, beta, tol), nil
			}
		}
	}
	if a, e, _, _, nf, stiff, _, _, _, ok := deflateInfinite(A, E, false); ok && !stiff {
		ad, ed := a.RawMatrix(), e.RawMatrix()
		return qzFinitePoles(nf, ad.Data, ad.Stride, ed.Data, ed.Stride, 0)
	}
	infTol := qzInfiniteTol(100*float64(n)*eps(), normA, normE)
	if tol := finitePoleInfTol(A, E, n, alphar, alphai, beta, infTol, normA, normE); tol > 0 {
		infTol = tol
	}
	return finitePoles(alphar, alphai, beta, infTol), nil
}

// largeRetained reports whether an eigenvalue that infTol keeps is large,
// |β| ≤ largeTol·|α|, a possible member of an infinite Jordan chain.
func largeRetained(alphar, alphai, beta []float64, infTol, largeTol float64) bool {
	for i := range beta {
		if !qzInfinite(alphar[i], alphai[i], beta[i], infTol) && qzInfinite(alphar[i], alphai[i], beta[i], largeTol) {
			return true
		}
	}
	return false
}

// leastRatioInfTol returns the |β|/|α| bound that selects the d
// eigenvalues of least |β|/|α| as infinite, and whether a factor
// stiffRatioGap separates them from the rest (marginalRatioGap when the
// stiff singular value of E lies within stiffMarginal of the rounding
// n·ε·σ₁(E), margin their ratio); without one the bound is 0, so only
// β = 0 is infinite. An index-1 pencil has exactly d = n − rank(E)
// infinite eigenvalues, and QZ on the whole pencil resolves a stiff finite
// one better than the staircase, whose deflated blocks can mix it with the
// null space of E.
func leastRatioInfTol(alphar, alphai, beta []float64, d int, margin float64) (float64, bool) {
	n := len(beta)
	if d == 0 {
		return 0, true
	}
	gap := stiffRatioGap
	if margin > 0 && margin < stiffMarginal {
		gap = marginalRatioGap
	}
	ratio := make([]float64, n)
	for i := range n {
		ratio[i] = math.Abs(beta[i]) / math.Hypot(alphar[i], alphai[i])
	}
	slices.Sort(ratio)
	if d < n && !(ratio[d] > gap*ratio[d-1]) {
		return 0, false
	}
	return ratio[d-1] * math.Sqrt(gap), true
}

// stiffRatioGap is the least factor between the |β|/|α| of a stiff finite
// eigenvalue and of the infinite ones for leastRatioInfTol to tell them
// apart; marginalRatioGap applies when the stiff singular value of E is
// within stiffMarginal of the rounding, where platform rounding can swap
// the two.
const (
	stiffRatioGap    = 10.0
	marginalRatioGap = 100.0
	stiffMarginal    = 100.0
)

func qzFinitePoles(n int, a []float64, lda int, e []float64, lde int, infTol float64) ([]complex128, error) {
	alphar, alphai, beta, ok := qzEigenvalues(n, a, lda, e, lde)
	if !ok {
		return nil, fmt.Errorf("generalized eigenvalue decomposition failed: %w", ErrSchurFailed)
	}
	return finitePoles(alphar, alphai, beta, infTol), nil
}

// finitePoles returns (α_r + iα_i)/β for the eigenvalues that are not
// infinite at infTol.
func finitePoles(alphar, alphai, beta []float64, infTol float64) []complex128 {
	poles := make([]complex128, 0, len(beta))
	for i := range beta {
		if qzInfinite(alphar[i], alphai[i], beta[i], infTol) {
			continue
		}
		poles = append(poles, complex(alphar[i]/beta[i], alphai[i]/beta[i]))
	}
	return poles
}

// qzEigenvalues returns the generalized eigenvalues (α_r + iα_i)/β of the
// leading n×n pencil of (a, e), row-major with strides lda, lde.
func qzEigenvalues(n int, a []float64, lda int, e []float64, lde int) (alphar, alphai, beta []float64, ok bool) {
	aData := make([]float64, n*n)
	eData := make([]float64, n*n)
	copyStrided(aData, n, a, lda, n, n)
	copyStrided(eData, n, e, lde, n, n)
	buf := make([]float64, 3*n)
	alphar, alphai, beta = buf[:n], buf[n:2*n], buf[2*n:]
	var vsl, vsr [1]float64
	var workQuery [1]float64
	impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, aData, max(1, n), eData, max(1, n), alphar, alphai, beta, vsl[:], 1, vsr[:], 1, workQuery[:], -1, nil)
	work := make([]float64, int(workQuery[0]))
	_, ok = impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, aData, max(1, n), eData, max(1, n), alphar, alphai, beta, vsl[:], 1, vsr[:], 1, work, len(work), nil)
	return alphar, alphai, beta, ok
}

// finitePoleInfTol returns the |β|/|α| bound of certifyFiniteRetained when
// an eigenvalue that infTol keeps is large and certified infinite, else 0.
func finitePoleInfTol(A, E *mat.Dense, n int, alphar, alphai, beta []float64, infTol, normA, normE float64) float64 {
	largeTol := qzInfiniteTol(qzLargeRatio, normA, normE)
	large := false
	ninf := 0
	for i := range n {
		switch {
		case qzInfinite(alphar[i], alphai[i], beta[i], infTol):
			ninf++
		case qzInfinite(alphar[i], alphai[i], beta[i], largeTol):
			large = true
		}
	}
	if !large {
		return 0
	}
	buf := make([]float64, 3*n)
	ar, ai, b := buf[:n], buf[n:2*n], buf[2*n:]
	front, back := 0, n-ninf
	for i := range n {
		j := front
		if qzInfinite(alphar[i], alphai[i], beta[i], infTol) {
			j = back
			back++
		} else {
			front++
		}
		ar[j], ai[j], b[j] = alphar[i], alphai[i], beta[i]
	}
	tol, err := certifyFiniteRetained(A, E, n, n-ninf, ninf, ar, ai, b, largeTol, normA)
	if err != nil {
		return 0
	}
	return tol
}

// blkDiagDescriptorE returns the descriptor matrix of parts whose states are
// stacked in argument order: blkdiag(E_i), with identity for standard parts.
// It returns nil when every part is standard.
func blkDiagDescriptorE(parts ...*System) *mat.Dense {
	descriptor := false
	for _, s := range parts {
		if s.E != nil {
			descriptor = true
			break
		}
	}
	if !descriptor {
		return nil
	}
	n := 0
	for _, s := range parts {
		ni, _, _ := s.Dims()
		n += ni
	}
	if n == 0 {
		return nil
	}
	out := mat.NewDense(n, n, nil)
	off := 0
	for _, s := range parts {
		ni, _, _ := s.Dims()
		if ni > 0 {
			setBlockOrIdentity(out, off, s.E, ni)
		}
		off += ni
	}
	return out
}

// conversionStandardForm returns sys without E, as MATLAB c2d and d2c do for
// dss models. Invertible E is folded into the state equation and keeps the
// state coordinates. Singular E is reduced by properExplicitForm (MATLAB
// dss2ss): the algebraic states are eliminated, the state coordinates change
// and state names are dropped, so reduced reports true. Improper models and
// singular pencils are rejected with ErrDescriptorUnsupported wrapping
// ErrImproperModel or ErrDescriptorSingular.
func conversionStandardForm(sys *System, context string) (out *System, reduced bool, err error) {
	if !sys.IsDescriptor() {
		return sys, false, nil
	}
	if explicit, err := sys.ToExplicit(); err == nil {
		return explicit, false, nil
	}
	out, err = sys.properExplicitForm()
	if err != nil {
		return nil, false, fmt.Errorf("%s: %w: %w", context, ErrDescriptorUnsupported, err)
	}
	return out, true, nil
}

// timeResponseForm returns sys without E for time-domain simulation, as
// MATLAB's step, impulse, initial and lsim do for dss models. Invertible E
// keeps the state coordinates, so x0 carries over unchanged. Singular E is
// reduced by properExplicitForm, whose state is the slow subsystem only;
// as in MATLAB, a nonzero x0 is then rejected with ErrDescriptorInitialState
// and the returned x0 is nil. x0 must have the original state dimension.
func (sys *System) timeResponseForm(x0 *mat.VecDense) (*System, *mat.VecDense, bool, error) {
	n, _, _ := sys.Dims()
	if x0 != nil && x0.Len() != n {
		return nil, nil, false, fmt.Errorf("x0 length %d != state dimension %d: %w", x0.Len(), n, ErrDimensionMismatch)
	}
	sys, err := sys.splitIODelayForInitialState(x0)
	if err != nil {
		return nil, nil, false, err
	}
	if !sys.IsDescriptor() {
		return sys, x0, false, nil
	}
	if explicit, err := sys.ToExplicit(); err == nil {
		return explicit, x0, false, nil
	}
	if x0 != nil && !allZeroVec(x0) {
		return nil, nil, false, fmt.Errorf("%w: %w", ErrDescriptorInitialState, ErrDescriptorSingular)
	}
	reduced, err := sys.properExplicitForm()
	if err != nil {
		return nil, nil, false, err
	}
	return reduced, nil, true, nil
}

func allZeroVec(v *mat.VecDense) bool {
	for i := range v.Len() {
		if v.AtVec(i) != 0 {
			return false
		}
	}
	return true
}

// properExplicitForm reduces a descriptor model with singular E to an
// explicit model with the same transfer function (MATLAB dss2ss). The
// Weierstrass split Q·(sE−A)·Z = diag(sE_f−A_f, sE_∞−A_∞) is obtained from
// the ordered QZ form and a generalized Sylvester decoupling; the algebraic
// part x_∞ = −Σ N^k A_∞⁻¹ B_∞ s^k (N = A_∞⁻¹E_∞ nilpotent) must reduce to the
// static term −C_∞ A_∞⁻¹ B_∞, which joins D. Otherwise the model is improper
// (non-causal in discrete time) and ErrImproperModel is returned; IsProper
// decides by the same test. Entries of E_∞ below the finite/infinite
// threshold are rounding noise and are zeroed so index-1 parts give N = 0
// exactly. QZ resolves an infinite Jordan chain
// only to |β|/|α| ~ ε^(1/k); when it fails, keeps a large eigenvalue,
// keeps more than rank(E) (the most a regular pencil has finite) or the
// pencil has index ≥ 2 (fewer than rank(E) finite, pencilIndexOne), and the
// staircase (deflateInfinite) counts the finite ones differently, the
// split is taken from the deflated pencil instead; an index ≥ 2 pencil
// whose QZ split keeps rank(E) or more finite and that the staircase
// cannot resolve returns ErrDescriptorSingular. Internal
// delay channels are reduced together with the I/O channels.
func (sys *System) properExplicitForm() (*System, error) {
	out, err := sys.weierstrassReduce()
	if err != nil {
		return nil, err
	}
	if n, _, _ := out.Dims(); n == 0 {
		_, m, p := sys.Dims()
		if err := storableStaticGain("descriptor has no dynamic states", p, m); err != nil {
			return nil, err
		}
	}
	return out, nil
}

func (sys *System) weierstrassReduce() (*System, error) {
	out, poly, err := sys.weierstrassReduceParts()
	if err == nil && poly != nil {
		return nil, fmt.Errorf("descriptor has impulsive modes: %w", ErrImproperModel)
	}
	return out, err
}

// weierstrassReduceParts is weierstrassReduce keeping the polynomial part:
// poly[k-1] is the p×m coefficient of s^k, −C∞·N^k·A∞⁻¹B∞, with entries at
// the rounding level of algebraicPartStatic zeroed; poly is nil for a proper
// model. Impulsive internal-delay channels return ErrImproperModel. Terms
// c·N^k·b above the rounding level of both algebraicTerms and
// algebraicNormTols but within the widened level of the staircase (blur,
// leak and egap of deflateInfinite) are real only when sTermsStable: a
// finite pole near ∞ beside the chain, or ill-conditioned A row blocks,
// leave rounding there that ε-perturbed coefficients change at first order.
// Such terms that are not real, and terms above the widened level that
// absorbStiffLeak explains with more equations than unknowns and
// sTermsStable finds not real, are a residue the split moved from finite
// poles near ∞: when every one of them moves alike under sTermsStable's
// perturbation (leakDriftShared), they are absorbed back before
// reductionMatches confirms the reduction.
func (sys *System) weierstrassReduceParts() (*System, []*mat.Dense, error) {
	out, poly, _, err := sys.weierstrassReduceCore(false, false)
	return out, poly, err
}

// weierstrassReduceCore is weierstrassReduceParts; with probe it returns
// only the terms c·N^k·b of algebraicTerms (nil without a nilpotent part).
// nested skips chainSplitReproducible, for its own copies.
func (sys *System) weierstrassReduceCore(probe, nested bool) (*System, []*mat.Dense, []*mat.Dense, error) {
	n, m, p := sys.Dims()
	q := sys.internalDelayCount()
	mi, pi := max(m+q, 1), max(p+q, 1)
	var poly []*mat.Dense

	bAug := mat.NewDense(n, mi, nil)
	cAug := mat.NewDense(pi, n, nil)
	dAug := mat.NewDense(pi, mi, nil)
	if m > 0 {
		setBlock(bAug, 0, 0, sys.B)
	}
	if p > 0 {
		setBlock(cAug, 0, 0, sys.C)
	}
	if p > 0 && m > 0 {
		setBlock(dAug, 0, 0, sys.D)
	}
	if q > 0 {
		setBlock(bAug, 0, m, sys.LFT.B2)
		setBlock(cAug, p, 0, sys.LFT.C2)
		setBlock(dAug, p, m, sys.LFT.D22)
		if p > 0 {
			setBlock(dAug, 0, m, sys.LFT.D12)
		}
		if m > 0 {
			setBlock(dAug, p, 0, sys.LFT.D21)
		}
	}

	s := make([]float64, n*n)
	t := make([]float64, n*n)
	tol := 100 * float64(n) * eps() * denseNorm(sys.E)
	finite := func(_, _, beta float64) bool { return math.Abs(beta) > tol }
	verifyChain := false
	d, index1, ratioTol, stiffSplit := weierstrassSplit(sys.A, sys.E, n)
	blurTol, stiffTol := 1.0, 1.0
	if stiffSplit {
		finite = func(ar, ai, beta float64) bool { return !qzInfinite(ar, ai, beta, ratioTol) }
	}

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := make([]float64, n*n)
	vsr := make([]float64, n*n)
	nf, recip, ok := orderedQZ(n, sys.A.RawMatrix(), sys.E.RawMatrix(), finite, s, t, vsl, vsr, alphar, alphai, beta)
	if !ok || nf > n-d || !index1 || !stiffSplit && largeRetained(alphar, alphai, beta, 0, qzInfiniteTol(qzLargeRatio, mat.Norm(sys.A, 2), mat.Norm(sys.E, 2))) {
		resolved := false
		a, e, qd, zd, nfd, stiff, leak, blur, egap, dok := deflateInfinite(sys.A, sys.E, true)
		all := func(_, _, _ float64) bool { return true }
		if dok && !stiff {
			resolved = ok && nfd == nf
			if !resolved {
				if sd, td, qs, zs, n1, rc, err := deflatedSchur(a, e, qd, zd, nfd, all, true); err == nil && n1 == nfd {
					s, t, vsl, vsr, nf, recip, ok, resolved = sd, td, qs, zs, nfd, rc, true, true
				}
			}
			if resolved {
				// leak multiplies worst-case turns of each level; the
				// rounding Qᵀ·B carries tracks its square root (empirical,
				// index-2..6 chain sweeps), unless the staircase rank was
				// itself decided within the leak.
				blurTol = max(blur, math.Sqrt(leak))
				stiffTol = egap
			}
		}
		if ok && !resolved && !index1 && nf >= n-d {
			// A stiff E stopped the staircase at its null space; past it
			// the split is checked against the pencil (verifyChain).
			if !dok || !stiff {
				return nil, nil, nil, errChainLeft
			}
			a, e, qd, zd, nfd, leak, blur, egap, dok = deflateStiffChain(sys.A, sys.E)
			if !dok {
				return nil, nil, nil, errChainLeft
			}
			sd, td, qs, zs, n1, rc, err := deflatedSchur(a, e, qd, zd, nfd, all, true)
			if err != nil || n1 != nfd {
				return nil, nil, nil, errChainLeft
			}
			s, t, vsl, vsr, nf, recip = sd, td, qs, zs, nfd, rc
			blurTol, stiffTol, verifyChain = max(blur, math.Sqrt(leak)), egap, true
		}
	}
	if !ok {
		return nil, nil, nil, fmt.Errorf("generalized Schur decomposition failed: %w", ErrDescriptorSingular)
	}
	ni := n - nf

	Q := mat.NewDense(n, n, vsl)
	Z := mat.NewDense(n, n, vsr)
	S := mat.NewDense(n, n, s)
	T := mat.NewDense(n, n, t)
	var bq, cz mat.Dense
	bq.Mul(Q.T(), bAug)
	cz.Mul(cAug, Z)

	b1 := extractBlock(&bq, 0, 0, nf, mi)
	c2 := extractBlock(&cz, 0, nf, pi, ni)
	cScale := denseNorm(cAug)
	if nf > 0 && ni > 0 {
		r := denseCopy(extractBlock(S, 0, nf, nf, ni))
		l := denseCopy(extractBlock(T, 0, nf, nf, ni))
		r.Scale(-1, r)
		l.Scale(-1, l)
		scale, ok := qzSylvester(n, nf, recip, s, t, r.RawMatrix().Data, l.RawMatrix().Data)
		if !ok || scale == 0 {
			return nil, nil, nil, fmt.Errorf("Weierstrass decoupling failed: %w", ErrDescriptorSingular)
		}
		r.Scale(1/scale, r)
		l.Scale(1/scale, l)
		var lb, cr mat.Dense
		lb.Mul(l, extractBlock(&bq, nf, 0, ni, mi))
		b1.Sub(b1, &lb)
		cr.Mul(extractBlock(&cz, 0, 0, pi, nf), r)
		c2.Add(c2, &cr)
	}

	dRed := mat.DenseCopyOf(dAug)
	confirm := false
	var leakTerms []*mat.Dense
	var leakTols []float64
	leakOnly := false
	var leakDrift []float64
	chain := false
	if ni > 0 {
		var lu mat.LU
		s22 := extractBlock(S, nf, nf, ni, ni)
		lu.Factorize(s22)
		if luNearSingular(&lu) {
			return nil, nil, nil, fmt.Errorf("singular pencil sE-A: %w", ErrDescriptorSingular)
		}
		bScale := denseNorm(bAug) / denseNorm(s22)
		var b, nilpotent mat.Dense
		if err := lu.SolveTo(&b, false, extractBlock(&bq, nf, 0, ni, mi)); err != nil {
			return nil, nil, nil, fmt.Errorf("singular pencil sE-A: %w", ErrDescriptorSingular)
		}
		if !index1 || ni != d {
			chain = true
			e22 := extractBlock(T, nf, nf, ni, ni)
			e22.Apply(func(_, _ int, v float64) float64 {
				if math.Abs(v) <= tol {
					return 0
				}
				return v
			}, e22)
			if err := lu.SolveTo(&nilpotent, false, e22); err != nil {
				return nil, nil, nil, fmt.Errorf("singular pencil sE-A: %w", ErrDescriptorSingular)
			}
			rel := 1e3 * float64(n) * eps()
			terms, vecTols := algebraicTerms(c2, &nilpotent, &b, cScale, bScale, rel, 1)
			if probe {
				return nil, nil, terms, nil
			}
			normTols := algebraicNormTols(len(terms), c2, &nilpotent, &b, cScale, bScale, rel)
			normStatic := algebraicPartStatic(terms, normTols)
			if !normStatic || !algebraicPartStatic(terms, vecTols) {
				lowTols := make([]float64, len(terms))
				for k := range lowTols {
					lowTols[k] = min(vecTols[k], normTols[k])
				}
				_, wide := algebraicTerms(c2, &nilpotent, &b, cScale, bScale, rel, blurTol*stiffTol)
				_, blurred := algebraicTerms(c2, &nilpotent, &b, cScale, bScale, rel, blurTol)
				impulsive := !normStatic
				switch {
				case normStatic || algebraicPartStatic(terms, wide):
					if q == 0 {
						if stable, ok, drift := sys.sTermsStable(terms); ok {
							impulsive = stable
							leakDrift = drift
						}
					}
					if poly = algebraicPolyTerms(terms, lowTols, p, m); !impulsive {
						confirm = !normStatic && !algebraicPartStatic(terms, blurred)
						leakTerms, leakTols = terms, blurred
					}
				default:
					poly = algebraicPolyTerms(terms, blurred, p, m)
					if q == 0 && len(terms) >= 2 {
						leakTerms, leakTols, leakOnly = terms, blurred, true
					}
				}
				if !impulsive && !confirm {
					poly = nil
				}
				if impulsive && (poly == nil || q > 0) {
					if verifyChain {
						return nil, nil, nil, errChainLeft
					}
					return nil, nil, nil, fmt.Errorf("descriptor has impulsive modes: %w", ErrImproperModel)
				}
			}
		}
		var cb mat.Dense
		cb.Mul(c2, &b)
		dRed.Sub(dRed, &cb)
	}

	out := sys.Copy()
	out.E = nil
	out.StateName = nil
	out.D = subDense(dRed, 0, 0, p, m)
	out.A, out.B, out.C = newDense(0, 0), newDense(0, m), newDense(p, 0)
	var bRed mat.Dense
	if nf > 0 {
		var lu mat.LU
		aRed := mat.NewDense(nf, nf, nil)
		lu.Factorize(extractBlock(T, 0, 0, nf, nf))
		if err := lu.SolveTo(aRed, false, extractBlock(S, 0, 0, nf, nf)); err != nil {
			return nil, nil, nil, fmt.Errorf("slow subsystem: %w", ErrDescriptorSingular)
		}
		if err := lu.SolveTo(&bRed, false, b1); err != nil {
			return nil, nil, nil, fmt.Errorf("slow subsystem: %w", ErrDescriptorSingular)
		}
		out.A = aRed
		out.B = subDense(&bRed, 0, 0, nf, m)
		out.C = subDense(&cz, 0, 0, p, nf)
	}
	if q > 0 {
		out.LFT.B2, out.LFT.C2 = newDense(0, q), newDense(q, 0)
		if nf > 0 {
			out.LFT.B2 = subDense(&bRed, 0, m, nf, q)
			out.LFT.C2 = subDense(&cz, p, 0, q, nf)
		}
		out.LFT.D12 = subDense(dRed, 0, m, p, q)
		out.LFT.D21 = subDense(dRed, p, 0, q, m)
		out.LFT.D22 = subDense(dRed, p, m, q, q)
	}
	if leakOnly && poly != nil {
		if fixed, ok := absorbStiffLeak(out, leakTerms, leakTols, true); ok {
			if stable, sok, drift := sys.sTermsStable(leakTerms); sok && !stable && leakDriftShared(leakTerms, leakTols, drift) && reductionMatches(sys, fixed) {
				out, poly = fixed, nil
			}
		}
		if poly != nil {
			if fixed, ok := refitFromPencil(sys, out, leakTerms); ok {
				out, poly = fixed, nil
			}
		}
	}
	if confirm {
		if fixed, ok := absorbStiffLeak(out, leakTerms, leakTols, false); ok && leakDriftShared(leakTerms, leakTols, leakDrift) && reductionMatches(sys, fixed) {
			out, poly = fixed, nil
		} else if reductionMatches(sys, out) {
			poly = nil
		} else if fixed, ok := refitFromPencil(sys, out, leakTerms); ok {
			out, poly = fixed, nil
		} else if poly == nil {
			if verifyChain {
				return nil, nil, nil, errChainLeft
			}
			return nil, nil, nil, fmt.Errorf("descriptor has impulsive modes: %w", ErrImproperModel)
		}
	}
	if chain && poly == nil && q == 0 && nf > 0 {
		if fixed, ok := refitFromPencil(sys, out, nil); ok {
			out = fixed
		}
	}
	if verifyChain && (q > 0 || !reductionPolyMatches(sys, out, poly) || !nested && !sys.chainSplitReproducible(out, poly)) {
		return nil, nil, nil, errChainLeft
	}
	return out, poly, nil, nil
}

// errChainLeft is the error of weierstrassReduceCore when an infinite
// Jordan chain stays among the finite eigenvalues, or a split of
// deflateStiffChain does not reproduce the pencil's response
// (reductionPolyMatches) or its own under rounding of the data
// (chainSplitReproducible).
var errChainLeft = fmt.Errorf("infinite Jordan chain left among the finite eigenvalues: %w", ErrDescriptorSingular)

// reductionMatchTol bounds the relative mismatch reductionMatches accepts.
const reductionMatchTol = 1e-10

// reductionMatches reports whether the proper reduction red of the
// descriptor sys, its s-terms dropped, matches sys's response, solved
// directly from (sE − A), within reductionMatchTol at two off-axis points
// of modulus about ‖A‖/‖E‖. It confirms dropping s-terms that only the
// egap widening of weierstrassReduceParts admits: a finite pole near ∞
// turns the computed infinite subspace by ε·egap, and the reduction then
// carries an error of that order times egap^k, which also shows at low
// frequency. The points sit at the data scale because, for index ≥ 2,
// (sE − A) is too ill-conditioned above it for the direct solve to
// certify anything.
func reductionMatches(sys, red *System) bool {
	return reductionPolyMatches(sys, red, nil)
}

// reductionPolyMatches is reductionMatches for red plus the polynomial
// part poly of weierstrassReduceParts (q = 0).
func reductionPolyMatches(sys, red *System, poly []*mat.Dense) bool {
	_, m, p := sys.Dims()
	w := mat.Norm(sys.A, 2) / mat.Norm(sys.E, 2)
	direct, reduced := newFrequencyEvaluator(sys).pointEval(), polyResponse(red, poly)
	g, h := make([]complex128, p*m), make([]complex128, p*m)
	for _, s := range [...]complex128{complex(0.6*w, 0.8*w), complex(0.24*w, 0.18*w)} {
		if direct(pointAt(s), g) != nil || reduced(s, h) != nil {
			return false
		}
		var diff, size float64
		for i := range g {
			diff = max(diff, cmplx.Abs(g[i]-h[i]))
			size = max(size, cmplx.Abs(g[i]), cmplx.Abs(h[i]))
		}
		if !(diff <= reductionMatchTol*size) {
			return false
		}
	}
	return true
}

// polyResponse returns the evaluator of red plus Σ poly[k−1]·s^k, which
// writes the p×m response at s into dst, row-major.
func polyResponse(red *System, poly []*mat.Dense) func(s complex128, dst []complex128) error {
	eval := newFrequencyEvaluator(red).pointEval()
	_, m, p := red.Dims()
	return func(s complex128, dst []complex128) error {
		if err := eval(pointAt(s), dst); err != nil {
			return err
		}
		sk := complex(1, 0)
		for _, c := range poly {
			sk *= s
			raw := c.RawMatrix()
			for i := range p {
				for j := range m {
					dst[i*m+j] += complex(raw.Data[i*raw.Stride+j], 0) * sk
				}
			}
		}
		return nil
	}
}

// chainSplitDraws and chainSplitTol set chainSplitReproducible: the
// splits of deflateStiffChain that matched the oracle in chain sweeps
// (index 2..5 beside poles of E singular value 1e-9..1e-2 relative,
// cond(U), cond(V) up to 1e4) reproduce under 4ε to about 5e-6 near the
// largest pole; the ones that did not move by O(1) or fail.
const (
	chainSplitDraws = 3
	chainSplitTol   = 1e-5
)

// chainSplitReproducible reports whether copies of sys (no internal
// delays) with A, E, B and C entries scaled by 1 + δ, |δ| ≤ 4ε, reduce
// alike: of chainSplitDraws copies, those that do not fail closed
// (ErrDescriptorSingular), at least one, give reductions (red, poly) that
// agree with this one within chainSplitTol at the data scale ‖A‖/‖E‖, at
// 1e2..1e6 times it on the imaginary axis and about the largest finite
// pole, relative to the response there or at the data scale. The direct
// (sE − A) solve of reductionPolyMatches cannot check above the data scale;
// a split that depends on the rounding of the data moves there.
func (sys *System) chainSplitReproducible(red *System, poly []*mat.Dense) bool {
	_, m, p := sys.Dims()
	w := mat.Norm(sys.A, 2) / mat.Norm(sys.E, 2)
	pts := []complex128{complex(0.6*w, 0.8*w), complex(0, 1e2*w), complex(0, 1e4*w), complex(0, 1e6*w)}
	if nr, _ := red.A.Dims(); nr > 0 {
		var eig mat.Eigen
		if !eig.Factorize(red.A, mat.EigenNone) {
			return false
		}
		rho := 0.0
		for _, v := range eig.Values(nil) {
			rho = max(rho, cmplx.Abs(v))
		}
		pts = append(pts, complex(0.3*rho, 0.9*rho), complex(0, 3*rho))
	}
	rng := rand.New(rand.NewPCG(7, 11))
	jitter := func(x *mat.Dense) {
		raw := x.RawMatrix()
		for i := range raw.Rows {
			row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
			for j := range row {
				row[j] *= 1 + 4*eps()*(2*rng.Float64()-1)
			}
		}
	}
	g, h := make([]complex128, p*m), make([]complex128, p*m)
	own := polyResponse(red, poly)
	agreed := 0
	for range chainSplitDraws {
		c := sys.Copy()
		jitter(c.A)
		jitter(c.E)
		jitter(c.B)
		jitter(c.C)
		red2, poly2, _, err := c.weierstrassReduceCore(false, true)
		if errors.Is(err, ErrDescriptorSingular) {
			continue
		}
		if err != nil || len(poly2) != len(poly) {
			return false
		}
		agreed++
		floor := 0.0
		copyEval := polyResponse(red2, poly2)
		for k, s := range pts {
			if own(s, g) != nil || copyEval(s, h) != nil {
				return false
			}
			var diff, size float64
			for i := range g {
				diff = max(diff, cmplx.Abs(g[i]-h[i]))
				size = max(size, cmplx.Abs(g[i]), cmplx.Abs(h[i]))
			}
			if k == 0 {
				floor = size
			}
			if !(diff <= chainSplitTol*max(size, floor)) {
				return false
			}
		}
	}
	return agreed > 0
}

// absorbStiffLeak returns red with the s-terms c·N^k·b (terms, k = 1..K)
// absorbed into its finite poles of largest modulus, or false when they do
// not explain them to tols. A finite pole λ near ∞ chordally (small in E)
// has a deflating subspace that ε-perturbations turn by ε/sep, sep ~
// |λ|^(−k) beside an index-k chain: the computed split moves part of its
// residue ΔR into the nilpotent block, which then carries the Taylor terms
// of ΔR·s/(λ(s − λ)), c·N^k·b = ΔR/λ^(k+1), and the reduction misses
// ΔR·s/(λ(s − λ)) (a D error ΔR/λ). Only the poles of largest modulus (one
// real pole or a complex pair) form the group: one beside them of modulus
// O(1) would let any s-term fit. With A·X = X·Λ and Yᵀ·A = Λ_L·Yᵀ on the
// group (real Schur, X, Y orthonormal), ΔR = C·X·β + γ·Yᵀ·B: ΔB = X·β and
// ΔC = γ·Yᵀ solve C·X·Λ^(−(k+1))·β + γ·Λ_L^(−(k+1))·Yᵀ·B = c·N^k·b in least
// squares, and ΔD = C·X·Λ⁻¹·β + γ·Λ_L⁻¹·Yᵀ·B. The B side alone is tried
// first, then (p > 1) the C side, then (p, m > 1) both. Trailing terms
// within tols (past the nilpotency index) are not fitted. With strict, the
// fit must leave more equations than unknowns, so a fit that is always
// exact cannot pass for a real s-term.
func absorbStiffLeak(red *System, terms []*mat.Dense, tols []float64, strict bool) (*System, bool) {
	n, m, p := red.Dims()
	if n == 0 || m == 0 || p == 0 || len(terms) == 0 {
		return nil, false
	}
	K := 0
	for k, term := range terms {
		if denseNorm(term) > tols[k] {
			K = k + 1
		}
	}
	if K == 0 {
		return nil, false
	}
	terms, tols = terms[:K], tols[:K]
	t := make([]float64, n*n)
	copyStrided(t, n, red.A.RawMatrix().Data, red.A.RawMatrix().Stride, n, n)
	z := make([]float64, n*n)
	wr, wi := make([]float64, n), make([]float64, n)
	bwork := make([]bool, n)
	var query [1]float64
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, n, t, n, wr, wi, z, n, query[:], -1, bwork)
	work := make([]float64, int(query[0]))
	if _, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, n, t, n, wr, wi, z, n, work, len(work), bwork); !ok {
		return nil, false
	}
	var top float64
	for i := 0; i < n; i += schurBlockSize(t, n, i) {
		top = max(top, cmplx.Abs(schurBlockEigenvalue(t, n, i)))
	}
	cut := top * (1 - 1e-8)
	inGroup := func(v complex128) bool { return cmplx.Abs(v) >= cut }
	tr, zr := slices.Clone(t), slices.Clone(z)
	ns, err := orderSchurGroupFirst(tr, zr, n, inGroup)
	if err != nil || ns == 0 {
		return nil, false
	}
	nl, err := orderSchurGroupFirst(t, z, n, func(v complex128) bool { return !inGroup(v) })
	if err != nil || nl != n-ns {
		return nil, false
	}
	lam, lamL := mat.NewDense(ns, ns, nil), mat.NewDense(ns, ns, nil)
	x, y := mat.NewDense(n, ns, nil), mat.NewDense(n, ns, nil)
	copyStrided(lam.RawMatrix().Data, ns, tr, n, ns, ns)
	copyStrided(lamL.RawMatrix().Data, ns, t[nl*n+nl:], n, ns, ns)
	copyStrided(x.RawMatrix().Data, ns, zr, n, n, ns)
	copyStrided(y.RawMatrix().Data, ns, z[nl:], n, n, ns)
	var lu, luL mat.LU
	lu.Factorize(lam)
	luL.Factorize(lamL)
	if luNearSingular(&lu) || luNearSingular(&luL) {
		return nil, false
	}
	w, v := make([]*mat.Dense, K+1), make([]*mat.Dense, K+1)
	var cx, yb mat.Dense
	cx.Mul(red.C, x)
	yb.Mul(y.T(), red.B)
	wPrev, vPrev := &cx, &yb
	for k := range K + 1 {
		var wt mat.Dense
		if err := lu.SolveTo(&wt, true, wPrev.T()); err != nil {
			return nil, false
		}
		w[k] = mat.DenseCopyOf(wt.T())
		v[k] = new(mat.Dense)
		if err := luL.SolveTo(v[k], false, vPrev); err != nil {
			return nil, false
		}
		wPrev, vPrev = w[k], v[k]
	}
	rows := K * p * m
	rhs := mat.NewVecDense(rows, nil)
	for k := range K {
		for r := range p {
			for c := range m {
				rhs.SetVec((k*p+r)*m+c, terms[k].At(r, c))
			}
		}
	}
	for _, side := range [...]struct{ b, c bool }{{true, false}, {false, true}, {true, true}} {
		if side.c && p == 1 || side.b && side.c && m == 1 {
			continue
		}
		nb := 0
		if side.b {
			nb = ns * m
		}
		cols := nb
		if side.c {
			cols += p * ns
		}
		M := mat.NewDense(rows, cols, nil)
		for k := range K {
			for r := range p {
				for c := range m {
					row := (k*p+r)*m + c
					for i := range ns {
						if side.b {
							M.Set(row, i*m+c, w[k+1].At(r, i))
						}
						if side.c {
							M.Set(row, nb+r*ns+i, v[k+1].At(i, c))
						}
					}
				}
			}
		}
		var svd mat.SVD
		if !svd.Factorize(M, mat.SVDThin) {
			continue
		}
		rank := svd.Rank(float64(max(rows, cols)) * eps())
		if rank == 0 || strict && rows <= rank {
			continue
		}
		var sol mat.Dense
		svd.SolveTo(&sol, rhs, rank)
		beta, gamma := mat.NewDense(ns, m, nil), mat.NewDense(p, ns, nil)
		if side.b {
			copy(beta.RawMatrix().Data, sol.RawMatrix().Data[:nb])
		}
		if side.c {
			copy(gamma.RawMatrix().Data, sol.RawMatrix().Data[nb:cols])
		}
		rest := make([]*mat.Dense, K)
		for k := range K {
			var fb, fc mat.Dense
			fb.Mul(w[k+1], beta)
			fc.Mul(gamma, v[k+1])
			rest[k] = mat.DenseCopyOf(extractBlock(terms[k], 0, 0, p, m))
			rest[k].Sub(rest[k], &fb)
			rest[k].Sub(rest[k], &fc)
		}
		if !algebraicPartStatic(rest, tols) {
			continue
		}
		out := red.Copy()
		var db, dc, dd mat.Dense
		db.Mul(x, beta)
		out.B.Add(out.B, &db)
		dc.Mul(gamma, y.T())
		out.C.Add(out.C, &dc)
		dd.Mul(w[0], beta)
		out.D.Add(out.D, &dd)
		dd.Reset()
		dd.Mul(gamma, v[0])
		out.D.Add(out.D, &dd)
		return out, true
	}
	return nil, false
}

// sTermsStable reports whether the terms c·N^k·b of sys (no internal
// delays) are its own: recomputed from sTermsDraws copies of sys whose A, E,
// B and C entries are scaled by 1 + δ, |δ| ≤ 4ε, every one stays within
// sTermsStableTol of them, relative to the largest. Rounding left by the
// staircase or by a finite pole near ∞ is as large in each copy but
// independent, so it moves by its own size. ok is false when a copy
// reduces to another nilpotent order or fails. drift[k] is the largest
// change of terms[k] relative to itself over the copies drawn, which stop
// at the first that moves (nil when a copy failed).
func (sys *System) sTermsStable(terms []*mat.Dense) (stable, ok bool, drift []float64) {
	var size float64
	for _, t := range terms {
		size = max(size, denseNorm(t))
	}
	rng := rand.New(rand.NewPCG(1, 2))
	jitter := func(m *mat.Dense) {
		raw := m.RawMatrix()
		for i := range raw.Rows {
			row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
			for j := range row {
				row[j] *= 1 + 4*eps()*(2*rng.Float64()-1)
			}
		}
	}
	var diff mat.Dense
	drift = make([]float64, len(terms))
	for range sTermsDraws {
		c := sys.Copy()
		jitter(c.A)
		jitter(c.E)
		jitter(c.B)
		jitter(c.C)
		_, _, got, err := c.weierstrassReduceCore(true, false)
		if err != nil || len(got) != len(terms) {
			return false, false, nil
		}
		stable = true
		for k, t := range got {
			r, cc := t.Dims()
			if r0, c0 := terms[k].Dims(); r != r0 || cc != c0 {
				return false, false, nil
			}
			diff.Sub(t, terms[k])
			dn := denseNorm(&diff)
			stable = stable && dn <= sTermsStableTol*size
			if dn > 0 {
				drift[k] = max(drift[k], dn/denseNorm(terms[k]))
			}
			diff.Reset()
		}
		if !stable {
			return false, true, drift
		}
	}
	return true, true, drift
}

// leakDriftShared reports whether the drift of sTermsStable is that of a
// residue ΔR leaked from one finite pole λ near ∞: every term c·N^k·b =
// ΔR/λ^(k+1) then moves with ΔR, by the same relative amount. A term above
// leakOwnFloor of its tol that moves leakOwnRatio times less than the
// fitted ones (above tols) carries an s-term of its own, which absorbing
// would drop (chain sweeps: shared drifts agree within 1.7×; own terms
// of b tail 1e-10..1e-8 move 5..1e3 times less).
func leakDriftShared(terms []*mat.Dense, tols, drift []float64) bool {
	if len(drift) != len(terms) {
		return false
	}
	var lead float64
	for k, t := range terms {
		if denseNorm(t) > tols[k] {
			lead = max(lead, drift[k])
		}
	}
	for k, t := range terms {
		if denseNorm(t) > leakOwnFloor*tols[k] && leakOwnRatio*drift[k] < lead {
			return false
		}
	}
	return true
}

const (
	leakOwnFloor = 0.03
	leakOwnRatio = 4
)

// sTermsDraws and sTermsStableTol set sTermsStable. Real s-terms 1e5·ε
// above the cut keep 3 digits under 4ε perturbation; rounding left by the
// staircase changes by a factor 2 to 10 (chain sweeps, impulsive chains of
// b tail 1e-9..1e-8).
const (
	sTermsDraws     = 3
	sTermsStableTol = 0.1
)

// weierstrassSplit classifies the pencil (A, E) for weierstrassReduce as
// generalizedPoles does: d = n − rank(E) and index1 from pencilIndexOne;
// an index-1 pencil has N = 0, so its infinite block, once it has d
// eigenvalues, carries no s-terms however much rounding T∞ keeps. When E
// has a stiff singular value and a gap separates the d eigenvalues of
// least |β|/|α| (leastRatioInfTol), stiffSplit is set and ratioTol, inside
// that gap below the stiff pole's |β|/|α|, bounds the infinite ones: the
// flat |β| cut would send the stiff finite pole to the infinite block. The
// bound sits under the finite side because the reordering QZ recomputes β
// of the infinite ones, which may be 0 in one run and not the other.
func weierstrassSplit(A, E *mat.Dense, n int) (d int, index1 bool, ratioTol float64, stiffSplit bool) {
	d, index1 = pencilIndexOne(A, E, n, mat.Norm(A, 2))
	if d == 0 || !index1 {
		return d, index1, 0, false
	}
	var svd mat.SVD
	if !svd.Factorize(E, mat.SVDNone) {
		return d, index1, 0, false
	}
	sv := svd.Values(nil)
	r, stiff, _ := descriptorRank(sv, sv[0], n, true)
	if !stiff || r != n-d {
		return d, index1, 0, false
	}
	aRaw, eRaw := A.RawMatrix(), E.RawMatrix()
	alphar, alphai, beta, ok := qzEigenvalues(n, aRaw.Data, aRaw.Stride, eRaw.Data, eRaw.Stride)
	if !ok {
		return d, index1, 0, false
	}
	margin := sv[r-1] / (float64(n) * eps() * sv[0])
	if _, sep := leastRatioInfTol(alphar, alphai, beta, d, margin); !sep {
		return d, index1, 0, false
	}
	gap := stiffRatioGap
	if margin < stiffMarginal {
		gap = marginalRatioGap
	}
	ratio := make([]float64, n)
	for i := range n {
		ratio[i] = math.Abs(beta[i]) / math.Hypot(alphar[i], alphai[i])
	}
	slices.Sort(ratio)
	return d, index1, ratio[d] / math.Sqrt(gap), true
}

// algebraicTerms returns the terms c·N^k·b, k ≥ 1, of the algebraic part
// and, for each, the level tol up to which it is rounding: rel (1e3·n·ε)
// times a first-order bound of the change of c·N^k·b under relative
// perturbations of c and b, blur·(‖c·N^k‖·‖b‖ + ‖c‖·‖N^k·b‖), plus of N,
// ‖N‖·Σ_j ‖c·N^j‖·‖N^(k−1−j)·b‖. blur is the staircase leak (see
// weierstrassReduceParts), which leaves rounding in the components of b
// (Qᵀ·B) a proper chain does not drive. The bound uses the actual
// vectors, not ‖c‖·‖N‖^k·‖b‖, which for a non-normal N exceeds a real
// s^k term by orders of magnitude. cScale and bScale are the data scales
// of c and b, ‖C‖ and ‖B‖/‖A∞‖, floors of ‖c‖ and ‖b‖: an impulsive mode
// that is unobservable or uncontrollable leaves c or b at rounding level,
// which must not set the bound itself.
func algebraicTerms(c, nilpotent, b *mat.Dense, cScale, bScale, rel, blur float64) (terms []*mat.Dense, tols []float64) {
	ni, _ := nilpotent.Dims()
	if ni < 2 {
		return nil, nil
	}
	left, right := make([]float64, ni), make([]float64, ni)
	l, r := mat.DenseCopyOf(c), mat.DenseCopyOf(b)
	left[0], right[0] = denseNorm(l), denseNorm(r)
	terms = make([]*mat.Dense, ni-1)
	var next mat.Dense
	for k := 1; k < ni; k++ {
		next.Mul(l, nilpotent)
		l.Copy(&next)
		next.Reset()
		next.Mul(nilpotent, r)
		r.Copy(&next)
		next.Reset()
		left[k], right[k] = denseNorm(l), denseNorm(r)
		terms[k-1] = new(mat.Dense)
		terms[k-1].Mul(c, r)
	}
	nNorm := denseNorm(nilpotent)
	cn, bn := max(left[0], cScale), max(right[0], bScale)
	tols = make([]float64, ni-1)
	for k := 1; k < ni; k++ {
		cross := 0.0
		for j := range k {
			cross += left[j] * right[k-1-j]
		}
		tols[k-1] = rel * (blur*(left[k]*bn+cn*right[k]) + nNorm*cross)
	}
	return terms, tols
}

// algebraicNormTols returns, for the count terms c·N^k·b, k ≥ 1, the
// normwise rounding level rel·‖c‖·‖N‖^k·‖b‖ (‖c‖ and ‖b‖ floored at
// cScale and bScale). For a non-normal N it exceeds the algebraicTerms
// tols, which can drop a real s-term; for a normal N it is the tighter.
func algebraicNormTols(count int, c, nilpotent, b *mat.Dense, cScale, bScale, rel float64) []float64 {
	bound := rel * max(denseNorm(c), cScale) * max(denseNorm(b), bScale)
	nNorm := denseNorm(nilpotent)
	tols := make([]float64, count)
	for k := range tols {
		bound *= nNorm
		tols[k] = bound
	}
	return tols
}

// algebraicPartStatic reports whether every term of algebraicTerms is at
// rounding level, i.e. the algebraic part contributes no polynomial
// (impulsive or anticipative) term. A fixed relative cut such as 1e-8
// drops real s-terms that are small beside the largest C or B entry.
func algebraicPartStatic(terms []*mat.Dense, tols []float64) bool {
	for k, term := range terms {
		if denseNorm(term) > tols[k] {
			return false
		}
	}
	return true
}

// algebraicPolyTerms returns −c·N^k·b, k ≥ 1, restricted to its leading
// p×m block, up to the last term above its algebraicTerms tol; entries
// at most tol are zeroed, so the polynomial carries no leaked rounding
// the static decision discards. It returns nil when every term vanishes.
func algebraicPolyTerms(terms []*mat.Dense, tols []float64, p, m int) []*mat.Dense {
	if p == 0 || m == 0 {
		return nil
	}
	var poly []*mat.Dense
	last := 0
	for k, term := range terms {
		coef := mat.NewDense(p, m, nil)
		nonzero := false
		for i := range p {
			for j := range m {
				if x := term.At(i, j); math.Abs(x) > tols[k] {
					coef.Set(i, j, -x)
					nonzero = true
				}
			}
		}
		poly = append(poly, coef)
		if nonzero {
			last = k + 1
		}
	}
	if last == 0 {
		return nil
	}
	return poly[:last]
}

func subDense(m *mat.Dense, r0, c0, rows, cols int) *mat.Dense {
	if rows == 0 || cols == 0 {
		return newDense(rows, cols)
	}
	return extractBlock(m, r0, c0, rows, cols)
}

// pencilSample is the response G(s) of a descriptor solved directly from
// (sE − A) at s; scale is its largest entry modulus and unc the mismatch a
// reduction may show there, reductionMatchTol·scale plus the response's
// sensitivity to rounding of the data.
type pencilSample struct {
	s          complex128
	g          []complex128
	scale, unc float64
}

// pencilSamples evaluates sys from (sE − A) at two points a decade from a
// tenth of the least to ten times the largest modulus among poles and the
// data scale ‖A‖/‖E‖ (points within 1e-3·|s| of a pole turned off it). It
// keeps a point only where the direct solve succeeds and the response is
// determined by the data: its first-order change under normwise ε
// perturbations of A, E, B and C, ε·(‖C·R‖·‖R·B‖·(|s|·‖E‖ + ‖A‖) +
// ‖C‖·‖R·B‖ + ‖C·R‖·‖B‖), R = (sE − A)⁻¹, is at most pencilSampleCap of
// it. Above that the rounding of the data leaves s-terms and chain
// eigenvalues of its own (|λ| ~ ε^(−1/k) for an index-k chain), which the
// proper part MATLAB keeps does not have.
func pencilSamples(sys *System, poles []complex128) []pencilSample {
	n, m, p := sys.Dims()
	normA, normE := mat.Norm(sys.A, 2), mat.Norm(sys.E, 2)
	normB, normC := denseNorm(sys.B), denseNorm(sys.C)
	w := normA / normE
	lo, hi := w, w
	for _, v := range poles {
		if r := cmplx.Abs(v); r > 0 && !math.IsInf(r, 0) {
			lo, hi = min(lo, r), max(hi, r)
		}
	}
	if !(lo > 0) || math.IsInf(hi, 0) {
		return nil
	}
	k0, k1 := math.Floor(math.Log10(lo))-1, math.Ceil(math.Log10(hi))+1
	step := math.Max(1, math.Ceil((k1-k0)/pencilSampleDecades))
	direct := newFrequencyEvaluator(sys).pointEval()
	a, e, b, c := sys.A.RawMatrix(), sys.E.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	lu := make([]complex128, n*n)
	piv := make([]int, n)
	x, y := make([]complex128, n*m), make([]complex128, n*p)
	var out []pencilSample
	for k := k0; k <= k1; k += step {
		for _, dir := range [...]complex128{complex(0.6, 0.8), complex(-0.28, 0.96)} {
			s := dir * complex(math.Pow(10, k), 0)
			for range 4 {
				near := false
				for _, v := range poles {
					near = near || cmplx.Abs(s-v) < 1e-3*cmplx.Abs(s)
				}
				if !near {
					break
				}
				s *= cmplx.Rect(1, 0.05)
			}
			g := make([]complex128, p*m)
			if direct(pointAt(s), g) != nil {
				continue
			}
			var scale float64
			for _, v := range g {
				scale = max(scale, cmplx.Abs(v))
			}
			if !(scale > 0) || math.IsInf(scale, 0) {
				continue
			}
			for i := range n {
				for j := range n {
					lu[i*n+j] = s*complex(e.Data[i*e.Stride+j], 0) - complex(a.Data[i*a.Stride+j], 0)
				}
			}
			if !cLUFactor(lu, piv, n) {
				continue
			}
			for i := range n {
				for j := range m {
					x[i*m+j] = complex(b.Data[i*b.Stride+j], 0)
				}
			}
			cLUSolve(lu, piv, x, n, m)
			for i := range n {
				for j := range n {
					lu[i*n+j] = s*complex(e.Data[j*e.Stride+i], 0) - complex(a.Data[j*a.Stride+i], 0)
				}
			}
			if !cLUFactor(lu, piv, n) {
				continue
			}
			for i := range n {
				for j := range p {
					y[i*p+j] = complex(c.Data[j*c.Stride+i], 0)
				}
			}
			cLUSolve(lu, piv, y, n, p)
			rb, cr := cFrobenius(x), cFrobenius(y)
			sens := eps() * (cr*rb*(cmplx.Abs(s)*normE+normA) + normC*rb + cr*normB)
			if sens <= pencilSampleCap*scale {
				out = append(out, pencilSample{s, g, scale, reductionMatchTol*scale + sens})
			}
		}
	}
	return out
}

// pencilSampleDecades caps the decades pencilSamples spans one by one.
const pencilSampleDecades = 24

// pencilSampleCap bounds the relative sensitivity of a kept sample.
const pencilSampleCap = 1e-4

// resolventBases returns, for each sample, Φ = C·(sI − A)⁻¹ (p×nf) and
// Ψ = (sI − A)⁻¹·B (nf×m) of the explicit model red, row-major; ok is
// false at an exactly singular sI − A.
func resolventBases(red *System, samples []pencilSample) (phi, psi [][]complex128, ok bool) {
	nf, m, p := red.Dims()
	lu := make([]complex128, nf*nf)
	piv := make([]int, nf)
	a, b, c := red.A.RawMatrix(), red.B.RawMatrix(), red.C.RawMatrix()
	for _, smp := range samples {
		for i := range nf {
			for j := range nf {
				lu[i*nf+j] = complex(-a.Data[i*a.Stride+j], 0)
			}
			lu[i*nf+i] += smp.s
		}
		if !cLUFactor(lu, piv, nf) {
			return nil, nil, false
		}
		x := make([]complex128, nf*m)
		for i := range nf {
			for j := range m {
				x[i*m+j] = complex(b.Data[i*b.Stride+j], 0)
			}
		}
		cLUSolve(lu, piv, x, nf, m)
		psi = append(psi, x)
		for i := range nf {
			for j := range nf {
				lu[i*nf+j] = complex(-a.Data[j*a.Stride+i], 0)
			}
			lu[i*nf+i] += smp.s
		}
		if !cLUFactor(lu, piv, nf) {
			return nil, nil, false
		}
		y := make([]complex128, nf*p)
		for i := range nf {
			for j := range p {
				y[i*p+j] = complex(c.Data[j*c.Stride+i], 0)
			}
		}
		cLUSolve(lu, piv, y, nf, p)
		f := make([]complex128, p*nf)
		for i := range p {
			for j := range nf {
				f[i*nf+j] = y[j*p+i]
			}
		}
		phi = append(phi, f)
	}
	return phi, psi, true
}

// samplesMismatch is the largest mismatch of red against samples, each
// entry relative to its sample's uncertainty unc.
func samplesMismatch(red *System, samples []pencilSample, psi [][]complex128) float64 {
	nf, m, p := red.Dims()
	c, d := red.C.RawMatrix(), red.D.RawMatrix()
	var worst float64
	for k, smp := range samples {
		for i := range p {
			for j := range m {
				h := complex(d.Data[i*d.Stride+j], 0)
				for l := range nf {
					h += complex(c.Data[i*c.Stride+l], 0) * psi[k][l*m+j]
				}
				worst = max(worst, cmplx.Abs(h-smp.g[i*m+j])/smp.unc)
			}
		}
	}
	return worst
}

// refitFromPencil returns red, the proper reduction of sys, with B and D,
// and for p > 1 alternately C and D, refitted in weighted least squares to
// pencilSamples of sys, A and so the poles kept; ok reports that the
// result (red itself when it already does) matches every sample to its
// uncertainty, from at least refitSurplus samples more than red has poles.
// With dropped, the s-terms the reduction drops, ok also needs them
// visible at some sample (droppedVisible). The response is well
// conditioned where the Weierstrass split is not: beside an index ≥ 2
// chain that QZ smears to |λ| ~ ε^(−1/k), a large finite pole λ (large in A
// or small in E) has a deflating subspace turned by ε/sep, which moves a
// residue ΔR, up to O(1)·|λ|, between λ, the nilpotent block and D. The
// error ΔR·s/(λ(s − λ)) shows below |λ|, where the direct solve is
// accurate, even when the data-scale points of reductionMatches cannot be
// solved; the first-order fit of absorbStiffLeak leaves a second-order
// remainder of it. Internal-delay models are not refitted.
func refitFromPencil(sys, red *System, dropped []*mat.Dense) (*System, bool) {
	nf, m, p := red.Dims()
	if m == 0 || p == 0 || sys.internalDelayCount() > 0 {
		return nil, false
	}
	var poles []complex128
	if nf > 0 {
		var eig mat.Eigen
		if !eig.Factorize(red.A, mat.EigenNone) {
			return nil, false
		}
		poles = eig.Values(nil)
	}
	samples := pencilSamples(sys, poles)
	if len(samples) < nf+refitSurplus || dropped != nil && !droppedVisible(samples, dropped) {
		return nil, false
	}
	phi, psi, ok := resolventBases(red, samples)
	if !ok {
		return nil, false
	}
	if samplesMismatch(red, samples, psi) <= 1 {
		return red, true
	}
	out := red.Copy()
	for round := range 2*refitRounds - 1 {
		bSide := round%2 == 0
		if !bSide && p == 1 {
			break
		}
		basis := psi
		if bSide {
			basis = phi
		}
		if !refitSide(out, samples, basis, bSide) {
			return nil, false
		}
		if phi, psi, ok = resolventBases(out, samples); !ok {
			return nil, false
		}
		if samplesMismatch(out, samples, psi) <= 1 {
			return out, true
		}
	}
	return nil, false
}

// droppedVisible reports whether the s-terms dropped, Σ dropped[k−1]·s^k,
// exceed refitVisible times the uncertainty of some sample: a proper model
// that then matches every sample shows they are not in the response.
func droppedVisible(samples []pencilSample, dropped []*mat.Dense) bool {
	for _, smp := range samples {
		r, c := dropped[0].Dims()
		for i := range r {
			for j := range c {
				var v, pow complex128 = 0, 1
				for _, t := range dropped {
					pow *= smp.s
					v += complex(t.At(i, j), 0) * pow
				}
				if cmplx.Abs(v) > refitVisible*smp.unc {
					return true
				}
			}
		}
	}
	return false
}

// refitSurplus, refitRounds (B-side fits, C-side ones between for p > 1)
// and refitVisible set refitFromPencil.
const (
	refitSurplus = 4
	refitRounds  = 3
	refitVisible = 100
	refitRankTol = 1e-13
)

// refitSide solves, in place, for B and D of red with bSide (basis Φ per
// sample), else C and D (basis Ψ), one input column (output row) at a
// time: Re and Im of Φ·B(:,j) + D(:,j) = G(:,j) at each sample, weighted
// by 1/unc, with equilibrated columns, by SVD least squares truncated at
// refitRankTol: a pole far above every sample is a column nearly equal to
// D's, and a minimal-norm fit keeps it from a cancelling residue.
func refitSide(red *System, samples []pencilSample, basis [][]complex128, bSide bool) bool {
	nf, m, p := red.Dims()
	outer, inner := m, p
	if !bSide {
		outer, inner = p, m
	}
	rows, cols := 2*len(samples)*inner, nf+inner
	M := mat.NewDense(rows, cols, nil)
	rhs := mat.NewVecDense(rows, nil)
	colScale := make([]float64, cols)
	var svd mat.SVD
	var sol mat.Dense
	for o := range outer {
		for k, smp := range samples {
			wk := 1 / smp.unc
			for i := range inner {
				r := 2 * (k*inner + i)
				var g complex128
				if bSide {
					g = smp.g[i*m+o]
				} else {
					g = smp.g[o*m+i]
				}
				rhs.SetVec(r, wk*real(g))
				rhs.SetVec(r+1, wk*imag(g))
				for l := range nf {
					v := basis[k][i*nf+l]
					if !bSide {
						v = basis[k][l*m+i]
					}
					M.Set(r, l, wk*real(v))
					M.Set(r+1, l, wk*imag(v))
				}
				for l := range inner {
					M.Set(r, nf+l, 0)
					M.Set(r+1, nf+l, 0)
				}
				M.Set(r, nf+i, wk)
			}
		}
		for l := range cols {
			colScale[l] = mat.Norm(M.ColView(l), 2)
			if colScale[l] == 0 {
				return false
			}
			for r := range rows {
				M.Set(r, l, M.At(r, l)/colScale[l])
			}
		}
		if !svd.Factorize(M, mat.SVDThin) {
			return false
		}
		rank := svd.Rank(refitRankTol)
		if rank == 0 {
			return false
		}
		svd.SolveTo(&sol, rhs, rank)
		for l := range nf {
			v := sol.At(l, 0) / colScale[l]
			if bSide {
				red.B.Set(l, o, v)
			} else {
				red.C.Set(o, l, v)
			}
		}
		for l := range inner {
			v := sol.At(nf+l, 0) / colScale[nf+l]
			if bSide {
				red.D.Set(l, o, v)
			} else {
				red.D.Set(o, l, v)
			}
		}
	}
	return true
}
