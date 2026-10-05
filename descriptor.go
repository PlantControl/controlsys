package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
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
		return fmt.Errorf("E %dx%d != %dx%d: %w", er, ec, n, n, ErrDimensionMismatch)
	}
	return nil
}

func (p descriptorPolicy) poles(A *mat.Dense, n int) ([]complex128, error) {
	if p.E != nil && !isIdentityDescriptor(p.E) {
		return generalizedPoles(A, p.E, n)
	}
	var eig mat.Eigen
	ok := eig.Factorize(A, mat.EigenNone)
	if !ok {
		return nil, fmt.Errorf("controlsys: eigenvalue decomposition failed to converge")
	}
	return eig.Values(nil), nil
}

func (p descriptorPolicy) requireStandard(context string) error {
	if p.E == nil || isIdentityDescriptor(p.E) {
		return nil
	}
	return fmt.Errorf("%s: %w", context, ErrDescriptorUnsupported)
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

// generalizedPoles computes the finite eigenvalues of the pencil (A, E)
// via the QZ algorithm (DGGES). Infinite eigenvalues (beta=0) are excluded.
func generalizedPoles(A, E *mat.Dense, n int) ([]complex128, error) {
	aData := make([]float64, n*n)
	eData := make([]float64, n*n)
	aRaw := A.RawMatrix()
	eRaw := E.RawMatrix()
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)
	copyStrided(eData, n, eRaw.Data, eRaw.Stride, n, n)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := make([]float64, n*n)
	vsr := make([]float64, n*n)

	var workQuery [1]float64
	impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, aData, n, eData, n, alphar, alphai, beta, vsl, n, vsr, n, workQuery[:], -1, nil)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)
	_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, aData, n, eData, n, alphar, alphai, beta, vsl, n, vsr, n, work, lwork, nil)
	if !ok {
		return nil, fmt.Errorf("controlsys: generalized eigenvalue decomposition failed")
	}

	poles := make([]complex128, 0, n)
	for i := range n {
		if beta[i] == 0 {
			continue
		}
		poles = append(poles, complex(alphar[i]/beta[i], alphai[i]/beta[i]))
	}
	return poles, nil
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
// (non-causal in discrete time) and ErrImproperModel is returned. Entries of
// E_∞ below the finite/infinite threshold are rounding noise and are zeroed
// so index-1 parts give N = 0 exactly. Internal
// delay channels are reduced together with the I/O channels.
func (sys *System) properExplicitForm() (*System, error) {
	n, m, p := sys.Dims()
	q := sys.internalDelayCount()
	mi, pi := max(m+q, 1), max(p+q, 1)

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
	copyStrided(s, n, sys.A.RawMatrix().Data, sys.A.RawMatrix().Stride, n, n)
	copyStrided(t, n, sys.E.RawMatrix().Data, sys.E.RawMatrix().Stride, n, n)
	tol := 100 * float64(n) * eps() * denseNorm(sys.E)
	finite := func(_, _, beta float64) bool { return math.Abs(beta) > tol }

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := make([]float64, n*n)
	vsr := make([]float64, n*n)
	bwork := make([]bool, n)
	var query [1]float64
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, finite,
		n, s, n, t, n, alphar, alphai, beta, vsl, n, vsr, n, query[:], -1, bwork)
	work := make([]float64, int(query[0]))
	nf, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, finite,
		n, s, n, t, n, alphar, alphai, beta, vsl, n, vsr, n, work, len(work), bwork)
	if !ok {
		return nil, fmt.Errorf("controlsys: generalized Schur decomposition failed: %w", ErrDescriptorSingular)
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
	if nf > 0 && ni > 0 {
		r := denseCopy(extractBlock(S, 0, nf, nf, ni))
		l := denseCopy(extractBlock(T, 0, nf, nf, ni))
		r.Scale(-1, r)
		l.Scale(-1, l)
		rRaw, lRaw := r.RawMatrix(), l.RawMatrix()
		iwork := make([]int, n+6)
		impl.Dtgsyl(blas.NoTrans, 0, nf, ni, s, n, s[nf*n+nf:], n, rRaw.Data, rRaw.Stride,
			t, n, t[nf*n+nf:], n, lRaw.Data, lRaw.Stride, query[:], -1, iwork)
		work = make([]float64, max(int(query[0]), 1))
		scale, _, ok := impl.Dtgsyl(blas.NoTrans, 0, nf, ni, s, n, s[nf*n+nf:], n, rRaw.Data, rRaw.Stride,
			t, n, t[nf*n+nf:], n, lRaw.Data, lRaw.Stride, work, len(work), iwork)
		if !ok || scale == 0 {
			return nil, fmt.Errorf("controlsys: Weierstrass decoupling failed: %w", ErrDescriptorSingular)
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
	if ni > 0 {
		var lu mat.LU
		lu.Factorize(extractBlock(S, nf, nf, ni, ni))
		if luNearSingular(&lu) {
			return nil, fmt.Errorf("controlsys: singular pencil sE-A: %w", ErrDescriptorSingular)
		}
		var b, nilpotent mat.Dense
		if err := lu.SolveTo(&b, false, extractBlock(&bq, nf, 0, ni, mi)); err != nil {
			return nil, fmt.Errorf("controlsys: singular pencil sE-A: %w", ErrDescriptorSingular)
		}
		e22 := extractBlock(T, nf, nf, ni, ni)
		e22.Apply(func(_, _ int, v float64) float64 {
			if math.Abs(v) <= tol {
				return 0
			}
			return v
		}, e22)
		if err := lu.SolveTo(&nilpotent, false, e22); err != nil {
			return nil, fmt.Errorf("controlsys: singular pencil sE-A: %w", ErrDescriptorSingular)
		}
		if !algebraicPartStatic(c2, &nilpotent, &b) {
			return nil, ErrImproperModel
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
			return nil, fmt.Errorf("controlsys: slow subsystem: %w", ErrDescriptorSingular)
		}
		if err := lu.SolveTo(&bRed, false, b1); err != nil {
			return nil, fmt.Errorf("controlsys: slow subsystem: %w", ErrDescriptorSingular)
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
	return out, nil
}

// algebraicPartStatic reports whether c·N^k·b vanishes for k ≥ 1, i.e. the
// algebraic part contributes no polynomial (impulsive or anticipative) term.
func algebraicPartStatic(c, nilpotent, b *mat.Dense) bool {
	ni, _ := nilpotent.Dims()
	cNorm, nNorm, bNorm := denseNorm(c), denseNorm(nilpotent), denseNorm(b)
	v := mat.DenseCopyOf(b)
	var next, term mat.Dense
	bound := cNorm * bNorm
	for range ni - 1 {
		next.Mul(nilpotent, v)
		v.Copy(&next)
		bound *= nNorm
		term.Mul(c, v)
		if denseNorm(&term) > 1e-8*bound {
			return false
		}
	}
	return true
}

func subDense(m *mat.Dense, r0, c0, rows, cols int) *mat.Dense {
	if rows == 0 || cols == 0 {
		return newDense(rows, cols)
	}
	return extractBlock(m, r0, c0, rows, cols)
}
