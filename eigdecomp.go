package controlsys

import (
	"cmp"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// decomposeByEigenvalues splits sys additively, G = G1 + G2, where G1 holds the
// modes selected by isGroup1 and the feedthrough D goes to G1 when
// feedthroughInGroup1 is set, else to G2. Nonsingular-E descriptors are made
// explicit first so the split follows the generalized eigenvalues of (A, E);
// ill-conditioned E (descriptorFoldInaccurate) is not.
// The ordered real Schur form is block-diagonalized with a Sylvester solve, so
// G1 + G2 is exact. Singular-E descriptors are split with the ordered
// generalized Schur form instead; infinite eigenvalues are never selected, so
// the improper/nondynamic part always stays in G2.
// op names the exported caller for errors; non-finite models are rejected
// before the Schur work.
func decomposeByEigenvalues(op string, sys *System, isGroup1 func(complex128) bool, feedthroughInGroup1 bool) (group1, group2 *System, err error) {
	if err := requireFiniteSystem(op, sys); err != nil {
		return nil, nil, err
	}
	group1, group2, err = decomposeModes(sys, isGroup1)
	if err != nil {
		return nil, nil, fmt.Errorf("%s: %w", op, err)
	}
	if feedthroughInGroup1 {
		group1.D, group2.D = group2.D, group1.D
	}
	return group1, group2, nil
}

func modesAllInGroup2(policy realizationTransformPolicy, sys *System) (group1, group2 *System, err error) {
	group1, err = policy.zeroOrderZeroFeedthrough()
	if err != nil {
		return nil, nil, err
	}
	return group1, sys.Copy(), nil
}

func modesAllInGroup1(policy realizationTransformPolicy, group1 *System) (*System, *System, error) {
	group2, err := policy.zeroOrderOriginalFeedthrough()
	if err != nil {
		return nil, nil, err
	}
	return group1, group2, nil
}

func decomposeModes(sys *System, isGroup1 func(complex128) bool) (group1, group2 *System, err error) {
	if sys.HasDelay() {
		return nil, nil, fmt.Errorf("delayed model; use Pade or AbsorbDelay first: %w", ErrDelayUnsupported)
	}
	if sys.IsDescriptor() && descriptorFoldInaccurate(sys.E) {
		return decomposeGeneralized(sys, isGroup1)
	}
	explicit, err := sys.ToExplicit()
	if errors.Is(err, ErrDescriptorSingular) {
		return decomposeGeneralized(sys, isGroup1)
	}
	if err != nil {
		return nil, nil, err
	}
	sys = explicit
	policy := newRealizationTransformPolicy(sys)
	n, m, p := sys.Dims()
	if n == 0 {
		return modesAllInGroup2(policy, sys)
	}

	t, z, err := modalSchur(sys)
	if err != nil {
		return nil, nil, err
	}
	n1, err := orderSchurGroupFirst(t, z, n, isGroup1)
	if err != nil {
		return nil, nil, err
	}
	if n1 == 0 {
		return modesAllInGroup2(policy, sys)
	}
	if n1 == n {
		return modesAllInGroup1(policy, policy.copyWithZeroFeedthrough())
	}

	x, err := separateSchurBlocks(t, n, n1, "decomposition")
	if err != nil {
		return nil, nil, err
	}
	n2 := n - n1
	bt, ct := modalInputOutput(sys, z, n, m, p)
	xGen := blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: x}

	B1 := extractModalBlock(bt, m, 0, n1, 0, m)
	B2 := extractModalBlock(bt, m, n1, n, 0, m)
	if m > 0 {
		blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, xGen, B2.RawMatrix(), 1, B1.RawMatrix())
	}

	C1 := extractModalBlock(ct, n, 0, p, 0, n1)
	C2 := extractModalBlock(ct, n, 0, p, n1, n)
	if p > 0 {
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, C1.RawMatrix(), xGen, 1, C2.RawMatrix())
	}

	sys1, err := policy.resultWithZeroFeedthrough(extractModalBlock(t, n, 0, n1, 0, n1), B1, C1)
	if err != nil {
		return nil, nil, err
	}
	sys2, err := policy.resultWithOriginalFeedthrough(extractModalBlock(t, n, n1, n, n1, n), B2, C2)
	if err != nil {
		return nil, nil, err
	}
	return sys1, sys2, nil
}

// descriptorFoldInaccurate reports whether E has a 1-norm condition
// estimate of illConditionedECond or more, full rank or not: the ordered
// Schur form of the LU fold E⁻¹A then smears the stiff direction over the
// slow modes (at cond 1e15 a 1e-1 error of G1 + G2), and the generalized
// Schur form of (A, E) splits them instead, as TransferFunction does.
func descriptorFoldInaccurate(E *mat.Dense) bool {
	var lu mat.LU
	lu.Factorize(E)
	return !(lu.Cond() < illConditionedECond)
}

// decomposeGeneralized splits a singular-E descriptor with the ordered
// generalized Schur form Qᵀ(A, E)Z = (S, T), selected finite eigenvalues
// leading, then removes the coupling blocks with the generalized Sylvester
// equation S11 R - L S22 = -S12, T11 R - L T22 = -T12. G1 is returned explicit
// (T11 is nonsingular); G2 keeps E = T22. The infinite structure is deflated
// to the trailing block first (deflateInfinite), so QZ never has to resolve
// an infinite Jordan chain, which it does only to |β|/|α| ~ ε^(1/k).
// When the rank decisions are ambiguous or blurred, or E has a stiff singular value
// and the pencil index ≥ 2 (the staircase then deflates the null space of E
// only), the whole pencil goes to QZ instead (decomposeGeneralizedQZ).
// A mode with |β| ≤ 10⁻³·ε·|α|·‖E‖_F/‖A‖_F is never selected: no finite
// pole of a nonsingular block is that large (|λ| ≤ ‖A‖/σ_min(E)), but QZ
// zeroes β of a stiff mode of a full-rank E within ε·‖E‖ and a reorder
// swap leaves it at ~ε²; folded into the explicit G1 that pole (~1e31)
// wrecks G1 + G2, so it stays in G2, whose E keeps it. When E has full
// MATLAB rank that mode is finite: Poles takes its value λ from E⁻¹A
// (fullRankPoles), and for a single real one splitStiffFullRank sets
// its β to α/λ, a change of order ε·‖E‖, and puts it as an explicit
// state in the part λ selects, so G1 and G2 (then both explicit)
// together have exactly the poles of Poles. A stiff mode whose β is tiny but
// nonzero is selected by its value; when the descriptor G2 that keeps it
// fails MATLAB rank although E does not, Poles(G2) would drop it, so it too
// goes through splitStiffFullRank at its value α/β (stiffTinyBetaPole).
func decomposeGeneralized(sys *System, isGroup1 func(complex128) bool) (group1, group2 *System, err error) {
	a, e, qd, zd, nf, stiff, _, blur, _, ok := deflateInfinite(sys.A, sys.E, true)
	ok = ok && blur == 1
	if ok && stiff {
		_, ok = pencilIndexOne(sys.A, sys.E, sys.A.RawMatrix().Rows, mat.Norm(sys.A, 2))
	}
	if !ok {
		return decomposeGeneralizedQZ(sys, isGroup1)
	}
	policy := newRealizationTransformPolicy(sys)
	if nf == 0 {
		return modesAllInGroup2(policy, sys)
	}
	normA, normE := mat.Norm(sys.A, 2), mat.Norm(sys.E, 2)
	infTol := qzInfiniteTol(1e-3*eps(), normA, normE)
	sawInfinite := false
	selct := func(ar, ai, b float64) bool {
		if b == 0 || qzInfinite(ar, ai, b, infTol) {
			sawInfinite = true
			return false
		}
		return isGroup1(complex(ar/b, ai/b))
	}
	s, t, q, z, n1, recip, err := deflatedSchur(a, e, qd, zd, nf, selct, false)
	if err != nil {
		return nil, nil, err
	}
	if sawInfinite && nf == a.RawMatrix().Rows {
		if lambda, ok := stiffFullRankPole(sys.A, sys.E); ok && (n1 > 0 || isGroup1(lambda)) {
			s, t, q, z, recip := s, t, q, z, recip
			if n1 == 0 {
				s, t, q, z, _, recip, err = deflatedSchur(a, e, qd, zd, nf, selct, true)
			}
			if err == nil {
				stiffTol := qzInfiniteTol(16*float64(nf)*eps(), normA, normE)
				if g1, g2, ok := splitStiffFullRank(policy, sys, s, t, q, z, recip, stiffTol, real(lambda), isGroup1); ok {
					return g1, g2, nil
				}
			}
		}
	}
	if n1 == 0 {
		return modesAllInGroup2(policy, sys)
	}
	if n := nf; !sawInfinite && n == a.RawMatrix().Rows && n1 < n &&
		descriptorERankDeficient(extractModalBlock(t, n, n1, n, n1, n)) && !descriptorERankDeficient(sys.E) {
		if lambda, tol, ok := stiffTinyBetaPole(sys.A, sys.E); ok {
			if g1, g2, ok := splitStiffFullRank(policy, sys, s, t, q, z, recip, tol, real(lambda), isGroup1); ok && (g2.E == nil || !descriptorERankDeficient(g2.E)) {
				return g1, g2, nil
			}
		}
	}
	return splitGeneralizedSchur(policy, sys, s, t, q, z, n1, recip)
}

// descriptorERankDeficient reports whether E fails MATLAB rank
// (eRankDeficient).
func descriptorERankDeficient(E *mat.Dense) bool {
	var lu mat.LU
	lu.Factorize(E)
	return eRankDeficient(E, &lu)
}

// stiffTinyBetaPole returns the pole α/β that Poles reports for the QZ
// eigenvalue of (A, E) of least |β|/|α| when no β is zero, it is real, and
// stiffRatioGap separates its ratio from the next; tol is the geometric mean
// of the two ratios, which finds its block in another Schur form of (A, E).
func stiffTinyBetaPole(A, E *mat.Dense) (lambda complex128, tol float64, ok bool) {
	n := A.RawMatrix().Rows
	aRaw, eRaw := A.RawMatrix(), E.RawMatrix()
	alphar, alphai, beta, ok := qzEigenvalues(n, aRaw.Data, aRaw.Stride, eRaw.Data, eRaw.Stride)
	if !ok || n < 2 {
		return 0, 0, false
	}
	at, least, next := -1, math.Inf(1), math.Inf(1)
	for i, b := range beta {
		if b == 0 {
			return 0, 0, false
		}
		r := math.Abs(b) / math.Hypot(alphar[i], alphai[i])
		switch {
		case r < least:
			at, least, next = i, r, least
		case r < next:
			next = r
		}
	}
	if at < 0 || alphai[at] != 0 || !(next > stiffRatioGap*least) {
		return 0, 0, false
	}
	l := alphar[at] / beta[at]
	if l == 0 || math.IsInf(l, 0) || math.IsNaN(l) {
		return 0, 0, false
	}
	return complex(l, 0), math.Sqrt(least * next), true
}

// stiffFullRankPole returns the stiff pole that Poles reports for an E of
// full MATLAB rank (fullRankPoles) when QZ gives exactly one β = 0 and its
// value from E⁻¹A is real and finite.
func stiffFullRankPole(A, E *mat.Dense) (complex128, bool) {
	n := A.RawMatrix().Rows
	aRaw, eRaw := A.RawMatrix(), E.RawMatrix()
	alphar, alphai, beta, ok := qzEigenvalues(n, aRaw.Data, aRaw.Stride, eRaw.Data, eRaw.Stride)
	if !ok {
		return 0, false
	}
	poles := fullRankPoles(A, E, alphar, alphai, beta)
	if len(poles) != n || len(finitePoles(alphar, alphai, beta, 0)) != n-1 {
		return 0, false
	}
	lambda := poles[n-1]
	if imag(lambda) != 0 || real(lambda) == 0 || math.IsInf(real(lambda), 0) || math.IsNaN(real(lambda)) {
		return 0, false
	}
	return lambda, true
}

// splitStiffFullRank splits the generalized Schur form (s, t) of a
// full-rank E with a single stiff mode (value λ from stiffFullRankPole),
// the one 1×1 block with |β| ≤ stiffTol·|α|. That block is moved to the
// trailing position (a swap perturbs its β by ~ε·‖E‖, so it is found by
// stiffTol, not by the β = 0 test of the ordering), its β set to α/λ,
// decoupled from the rest, and returned as the explicit state (λ, b/β,
// c): kept as a descriptor row it leaves E an entry ~ε·‖E‖ that Poles
// ranks deficient once its part has another mode, and folded to explicit
// while still coupled it costs G1 + G2 up to 1e-2. The rest, with a
// well-conditioned T11, is split by decomposeModes, and the stiff state
// joins the part isGroup1(λ) picks. It reports false, leaving the
// caller's form untouched, when there is not exactly one such block or
// the blocks cannot be separated.
func splitStiffFullRank(policy realizationTransformPolicy, sys *System, s, t, q, z []float64, recip bool, stiffTol, lambda float64, isGroup1 func(complex128) bool) (group1, group2 *System, ok bool) {
	n, m, p := sys.Dims()
	if n < 2 {
		return nil, nil, false
	}
	at := -1
	for i := range n {
		sii, tii := s[i*n+i], t[i*n+i]
		if sii == 0 || !(math.Abs(tii) <= stiffTol*math.Abs(sii)) {
			continue
		}
		if at >= 0 {
			return nil, nil, false
		}
		at = i
	}
	if at < 0 {
		return nil, nil, false
	}
	s, t, q, z = slices.Clone(s), slices.Clone(t), slices.Clone(q), slices.Clone(z)
	k := n - 1
	if at != k {
		qzA, qzB := s, t
		if recip {
			qzA, qzB = t, s
		}
		work := make([]float64, 4*n+16)
		if _, last, ok := impl.Dtgexc(true, true, n, qzA, n, qzB, n, q, n, z, n, at, k, work, len(work)); !ok || last != k {
			return nil, nil, false
		}
	}
	if s[k*n+k-1] != 0 || t[k*n+k-1] != 0 || s[k*n+k] == 0 {
		return nil, nil, false
	}
	t[k*n+k] = s[k*n+k] / lambda
	rest, stiff, err := splitGeneralizedSchur(policy, sys, s, t, q, z, k, recip)
	if err != nil {
		return nil, nil, false
	}
	beta := stiff.E.At(0, 0)
	r1, r2, err := decomposeModes(rest, isGroup1)
	if err != nil {
		return nil, nil, false
	}
	r2.D = denseCopy(sys.D)
	b := make([]float64, m)
	for j := range b {
		b[j] = stiff.B.At(0, j) / beta
	}
	c := make([]float64, p)
	for i := range c {
		c[i] = stiff.C.At(i, 0)
	}
	if isGroup1(complex(lambda, 0)) {
		r1, err = appendExplicitState(policy, r1, lambda, b, c)
	} else {
		r2, err = appendExplicitState(policy, r2, lambda, b, c)
	}
	if err != nil {
		return nil, nil, false
	}
	return r1, r2, true
}

// appendExplicitState returns the explicit g with one more state x' = a·x
// + b·u, y += c·x, keeping g's D.
func appendExplicitState(policy realizationTransformPolicy, g *System, a float64, b, c []float64) (*System, error) {
	n, m, p := g.Dims()
	A := mat.NewDense(n+1, n+1, nil)
	B := mat.NewDense(n+1, m, nil)
	C := mat.NewDense(p, n+1, nil)
	if n > 0 {
		A.Slice(0, n, 0, n).(*mat.Dense).Copy(g.A)
		if m > 0 {
			B.Slice(0, n, 0, m).(*mat.Dense).Copy(g.B)
		}
		if p > 0 {
			C.Slice(0, p, 0, n).(*mat.Dense).Copy(g.C)
		}
	}
	A.Set(n, n, a)
	for j, v := range b {
		B.Set(n, j, v)
	}
	for i, v := range c {
		C.Set(i, n, v)
	}
	return policy.result(A, B, C, denseCopy(g.D))
}

// deflatedSchur completes the deflation Qdᵀ(A, E)Zd = (a, e) of
// deflateInfinite (finite block nf leading) to a generalized Schur form
// Qᵀ(A, E)Z = (S, T): the finite block by orderedQZ, its n1 eigenvalues
// selected by selct leading, and the infinite block by QZ on its own, so the
// exact block structure is kept. When n1 is 0 the form is built only if
// full is set.
func deflatedSchur(a, e, qd, zd *mat.Dense, nf int, selct func(ar, ai, b float64) bool, full bool) (s, t, q, z []float64, n1 int, recip bool, err error) {
	n := a.RawMatrix().Rows
	ninf := n - nf
	sf := make([]float64, nf*nf)
	tf := make([]float64, nf*nf)
	qf := make([]float64, nf*nf)
	zf := make([]float64, nf*nf)
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	aRaw, eRaw := a.RawMatrix(), e.RawMatrix()
	if nf > 0 {
		var ok bool
		n1, recip, ok = orderedQZ(nf, blas64.General{Rows: nf, Cols: nf, Stride: aRaw.Stride, Data: aRaw.Data},
			blas64.General{Rows: nf, Cols: nf, Stride: eRaw.Stride, Data: eRaw.Data}, selct, sf, tf, qf, zf, alphar[:nf], alphai[:nf], beta[:nf])
		if !ok {
			return nil, nil, nil, nil, 0, false, fmt.Errorf("ordered generalized Schur failed: %w", ErrSchurFailed)
		}
	}
	if n1 == 0 && !full {
		return nil, nil, nil, nil, 0, false, nil
	}

	s = make([]float64, n*n)
	t = make([]float64, n*n)
	q = make([]float64, n*n)
	z = make([]float64, n*n)
	copyStrided(s, n, sf, nf, nf, nf)
	copyStrided(t, n, tf, nf, nf, nf)
	qfGen := blas64.General{Rows: nf, Cols: nf, Stride: nf, Data: qf}
	zfGen := blas64.General{Rows: nf, Cols: nf, Stride: nf, Data: zf}
	qdRaw, zdRaw := qd.RawMatrix(), zd.RawMatrix()
	if nf > 0 {
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: nf, Stride: qdRaw.Stride, Data: qdRaw.Data}, qfGen,
			0, blas64.General{Rows: n, Cols: nf, Stride: n, Data: q})
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: nf, Stride: zdRaw.Stride, Data: zdRaw.Data}, zfGen,
			0, blas64.General{Rows: n, Cols: nf, Stride: n, Data: z})
	}
	if ninf > 0 {
		si := make([]float64, ninf*ninf)
		ti := make([]float64, ninf*ninf)
		qi := make([]float64, ninf*ninf)
		zi := make([]float64, ninf*ninf)
		copyStrided(si, ninf, aRaw.Data[nf*aRaw.Stride+nf:], aRaw.Stride, ninf, ninf)
		copyStrided(ti, ninf, eRaw.Data[nf*eRaw.Stride+nf:], eRaw.Stride, ninf, ninf)
		first, second := si, ti
		if recip {
			first, second = ti, si
		}
		bwork := make([]bool, ninf)
		var query [1]float64
		ar, ai, b := alphar[nf:], alphai[nf:], beta[nf:]
		impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, ninf, first, ninf, second, ninf,
			ar, ai, b, qi, ninf, zi, ninf, query[:], -1, bwork)
		work := make([]float64, int(query[0]))
		if _, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortNone, nil, ninf, first, ninf, second, ninf,
			ar, ai, b, qi, ninf, zi, ninf, work, len(work), bwork); !ok {
			return nil, nil, nil, nil, 0, false, fmt.Errorf("generalized Schur of the infinite part failed: %w", ErrSchurFailed)
		}
		copyStrided(s[nf*n+nf:], n, si, ninf, ninf, ninf)
		copyStrided(t[nf*n+nf:], n, ti, ninf, ninf, ninf)
		qiGen := blas64.General{Rows: ninf, Cols: ninf, Stride: ninf, Data: qi}
		ziGen := blas64.General{Rows: ninf, Cols: ninf, Stride: ninf, Data: zi}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: ninf, Stride: qdRaw.Stride, Data: qdRaw.Data[nf:]}, qiGen,
			0, blas64.General{Rows: n, Cols: ninf, Stride: n, Data: q[nf:]})
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: ninf, Stride: zdRaw.Stride, Data: zdRaw.Data[nf:]}, ziGen,
			0, blas64.General{Rows: n, Cols: ninf, Stride: n, Data: z[nf:]})
		if nf == 0 {
			return s, t, q, z, n1, recip, nil
		}
		tmp := make([]float64, nf*ninf)
		tmpGen := blas64.General{Rows: nf, Cols: ninf, Stride: ninf, Data: tmp}
		for _, pair := range [2]struct {
			src blas64.General
			dst []float64
		}{{blas64.General{Rows: nf, Cols: ninf, Stride: aRaw.Stride, Data: aRaw.Data[nf:]}, s[nf:]},
			{blas64.General{Rows: nf, Cols: ninf, Stride: eRaw.Stride, Data: eRaw.Data[nf:]}, t[nf:]}} {
			blas64.Gemm(blas.Trans, blas.NoTrans, 1, qfGen, pair.src, 0, tmpGen)
			blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tmpGen, ziGen, 0, blas64.General{Rows: nf, Cols: ninf, Stride: n, Data: pair.dst})
		}
	}
	return s, t, q, z, n1, recip, nil
}

// deflateInfinite reduces the regular pencil (A, E) by orthogonal Q, Z to
// Qᵀ(A, E)Z = ([Af Ax; 0 A∞], [Ef Ex; 0 N]), Ef nonsingular and the
// trailing n − nf eigenvalues all infinite (A∞ nonsingular, N nilpotent),
// with the staircase of Van Dooren: compress the rows of E to expose its
// null space, compress the matching rows of A onto the trailing columns,
// and repeat on the leading block until its E is nonsingular. The rank of
// E is that of descriptorRank; a nonsingular E deflates nothing (nf = n).
// When it reports stiff, only the null space of E is deflated (index 1 is
// assumed) and stiff is set: the deflated block can mix the stiff finite
// pole with that null space, so callers that can should classify on the
// whole pencil. leak, at most qzInfiniteGap, is the factor by which
// rounding of A, turning Z of each level by ε·‖A‖/σ_min(A row block), may
// leave more than ε-level rounding in the deflated blocks and in Qᵀ·B:
// callers widen their zero tests by it. A rank of a deflated E block that
// descriptorRank finds ambiguous is decided by deflatedRank, which may
// deflate no more than infiniteChainCount directions per level (a level
// past the first asserts that many chains of length ≥ 2) and only values
// leakedZeros finds to be rounding: a finite pole small in E is otherwise
// taken as infinite; blur is then leak, else 1.
// egap, the E-side counterpart of leak, is the largest σ₁(E)/σ_r of a
// deflated E block, σ_r its least nonzero singular value: a
// finite pole that is small in E (near ∞ chordally) turns the null space
// of that block, so Q and Z, by ε·egap, and that rounding of Qᵀ·B and the
// deflated blocks is not bounded by leak.
// An ambiguous rank, or a near rank-deficient row block of A
// (a nearly singular pencil), sets ok false.
func deflateInfinite(A, E *mat.Dense, wantQZ bool) (a, e, q, z *mat.Dense, nf int, stiff bool, leak, blur, egap float64, ok bool) {
	a, e, q, z, nf, stiff, leak, blur, egap, _, ok = deflateStaircase(A, E, wantQZ, false, false, -1, 0)
	return a, e, q, z, nf, stiff, leak, blur, egap, ok
}

// deflateStiffChain is deflateInfinite for a pencil of index ≥ 2
// (pencilIndexOne) whose E has stiff singular values: those are finite, as
// in MATLAB's rank (D62TAA), and the staircase goes on past the null space
// of E, deciding the ranks of the deflated E blocks by stiffChainRank. It
// fails (ok false) when E is not stiff or the index is 1.
func deflateStiffChain(A, E *mat.Dense) (a, e, q, z *mat.Dense, nf int, leak, blur, egap float64, ok bool) {
	a, e, q, z, nf, _, leak, blur, egap, _, ok = deflateStaircase(A, E, true, true, false, -1, 0)
	return a, e, q, z, nf, leak, blur, egap, ok
}

// deflateStaircase is deflateInfinite, or deflateStiffChain with
// stiffChain; indexKnown skips its pencilIndexOne test for a caller that
// found index ≥ 2 already. With probe ≥ 0 it stops at that level (0 the
// full E) and returns only the cnt eigenvalues of its leading k×k pencil
// nearest ∞ (nearInfiniteEigs) with k, nil when the staircase ends first;
// zeros deflatedRank takes on the way are not checked by leakedZeros.
func deflateStaircase(A, E *mat.Dense, wantQZ, stiffChain, indexKnown bool, probe, cnt int) (a, e, q, z *mat.Dense, nf int, stiff bool, leak, blur, egap float64, probeEig []complex128, ok bool) {
	n, _ := A.Dims()
	if n == 0 {
		return nil, nil, nil, nil, 0, false, 0, 0, 0, nil, false
	}
	buf := make([]float64, 6*n*n+n)
	ad, ed, ud, vtd, sd, td := buf[:n*n], buf[n*n:2*n*n], buf[2*n*n:3*n*n], buf[3*n*n:4*n*n], buf[4*n*n:5*n*n], buf[5*n*n:6*n*n]
	sv := buf[6*n*n:]
	aRaw, eRaw := A.RawMatrix(), E.RawMatrix()
	copyStrided(ad, n, aRaw.Data, aRaw.Stride, n, n)
	copyStrided(ed, n, eRaw.Data, eRaw.Stride, n, n)
	var qd, zd []float64
	if wantQZ {
		qd, zd = make([]float64, n*n), make([]float64, n*n)
		for i := range n {
			qd[i*n+i], zd[i*n+i] = 1, 1
		}
	}
	gen := func(data []float64, r, c, stride int) blas64.General {
		return blas64.General{Rows: r, Cols: c, Stride: stride, Data: data}
	}
	var work []float64
	grow := func(lw float64) {
		if len(work) < int(lw) {
			work = make([]float64, int(lw))
		}
	}
	svd := func(jobU, jobVT lapack.SVDJob, m, k int) bool {
		var query [1]float64
		impl.Dgesvd(jobU, jobVT, m, k, sd, k, sv, ud, max(1, m), vtd, k, query[:], -1)
		grow(query[0])
		return impl.Dgesvd(jobU, jobVT, m, k, sd, k, sv, ud, max(1, m), vtd, k, work, len(work))
	}
	mulInto := func(dst []float64, stride, rows, cols int, x, y blas64.General, tx blas.Transpose) {
		blas64.Gemm(tx, blas.NoTrans, 1, x, y, 0, gen(td, rows, cols, cols))
		copyStrided(dst, stride, td, cols, rows, cols)
	}
	ratio := 1e3 * float64(n) * eps()
	normA := mat.Norm(A, 2)
	tolA := ratio * normA
	scaleE := mat.Norm(E, 2)
	leak, blur, egap = 1.0, 1.0, 1.0
	chains := -1
	stiffPoles := 0
	k := n
	for level := 0; k > 0 && !stiff; level++ {
		if clearlyNonsingular(ed, n, k, sd, qzInfiniteGap*ratio*scaleE) {
			break
		}
		copyStrided(sd, k, ed, n, k, k)
		if !svd(lapack.SVDAll, lapack.SVDNone, k, k) {
			return nil, nil, nil, nil, 0, false, 0, 0, 0, nil, false
		}
		if k == n {
			scaleE = sv[0]
		}
		if level == probe {
			eig, eok := nearInfiniteEigs(ad, ed, n, k, cnt)
			return nil, nil, nil, nil, k, false, 0, 0, 0, eig, eok
		}
		r, st, ok := descriptorRank(sv[:k], scaleE, n, k == n)
		switch {
		case stiffChain && k == n:
			if !st {
				return nil, nil, nil, nil, 0, false, 0, 0, 0, nil, false
			}
			if !indexKnown {
				if _, index1 := pencilIndexOne(A, E, n, normA); index1 {
					return nil, nil, nil, nil, 0, false, 0, 0, 0, nil, false
				}
			}
			st = false
			for _, s := range sv[:r] {
				if s <= qzInfiniteGap*1e3*float64(n)*eps()*scaleE {
					stiffPoles++
				}
			}
		case stiffChain:
			r, ok = stiffChainRank(sv[:k], 1e3*float64(n)*eps()*scaleE, stiffPoles)
		}
		if !ok && k < n && !stiffChain {
			tol := 1e3 * float64(n) * eps() * scaleE
			if r, ok = deflatedRank(sv[:k], tol, leak); ok && r < k {
				if chains < 0 {
					chains = infiniteChainCount(A, E, n, normA)
				}
				ok = k-r <= chains && (probe >= 0 || leakedZeros(A, E, ad, ed, level, k, k-r))
			}
			blur = max(blur, leak)
		}
		if !ok {
			return nil, nil, nil, nil, 0, false, 0, 0, 0, nil, false
		}
		stiff = st
		d := k - r
		if d == 0 {
			break
		}
		if r > 0 {
			egap = max(egap, scaleE/sv[r-1])
		}
		u := gen(ud, k, k, k)
		mulInto(ad, n, k, n, u, gen(ad, k, n, n), blas.Trans)
		mulInto(ed, n, k, n, u, gen(ed, k, n, n), blas.Trans)
		if wantQZ {
			blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(qd, n, k, n), u, 0, gen(td, n, k, k))
			copyStrided(qd, n, td, k, n, k)
		}
		for i := r; i < k; i++ {
			clear(ed[i*n : i*n+k])
		}

		for i := range k {
			for j := range d {
				ud[i*k+j] = ad[(r+j)*n+i]
			}
		}
		tau := sv[:d]
		var query [2]float64
		impl.Dgeqrf(k, d, ud, k, tau, query[:], -1)
		impl.Dorgqr(k, k, d, ud, k, tau, query[1:], -1)
		grow(max(query[0], query[1]))
		impl.Dgeqrf(k, d, ud, k, tau, work, len(work))
		clear(sd[:d*d])
		for i := range d {
			copy(sd[i*d+i:i*d+d], ud[i*k+i:i*k+d])
		}
		impl.Dorgqr(k, k, d, ud, k, tau, work, len(work))
		if !svd(lapack.SVDNone, lapack.SVDNone, d, d) || sv[d-1] <= qzInfiniteGap*tolA {
			return nil, nil, nil, nil, 0, false, 0, 0, 0, nil, false
		}
		leak = min(qzInfiniteGap, leak*max(1, normA/sv[d-1]))
		zk := vtd[:k*k]
		for i := range k {
			for j := range k {
				zk[i*k+(j+r)%k] = ud[i*k+j]
			}
		}
		zg := gen(zk, k, k, k)
		mulInto(ad, n, k, k, gen(ad, k, k, n), zg, blas.NoTrans)
		mulInto(ed, n, k, k, gen(ed, k, k, n), zg, blas.NoTrans)
		if wantQZ {
			blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, gen(zd, n, k, n), zg, 0, gen(td, n, k, k))
			copyStrided(zd, n, td, k, n, k)
		}
		for i := r; i < k; i++ {
			clear(ad[i*n : i*n+r])
		}
		k = r
	}
	return mat.NewDense(n, n, ad), mat.NewDense(n, n, ed), denseOrNil(qd, n), denseOrNil(zd, n), k, stiff, leak, blur, egap, nil, true
}

// leakedZeros reports whether the cnt singular values that deflatedRank
// takes as zero in the E block of staircase level (size k, deflated pencil
// ad, ed) of (A, E) are rounding the leak left rather than a finite pole
// small in E, which deflating would drop. Kept, such a value makes the
// leading k×k pencil carry cnt eigenvalues near ∞ (nearInfiniteEigs). From
// a finite pole they are a function of the data: the same eigenvalues come
// back to leakStableTol from each of leakDraws copies of A and E with
// entries scaled by 1 + δ, |δ| ≤ 4ε. From rounding of an infinite chain
// they are perturbed chain eigenvalues, which move at first order (ε^(1/j)
// for a chain of length j) and so do not; a copy that reaches that level
// with another block size also reports leaked.
func leakedZeros(A, E *mat.Dense, ad, ed []float64, level, k, cnt int) bool {
	n, _ := A.Dims()
	ref, ok := nearInfiniteEigs(ad, ed, n, k, cnt)
	if !ok {
		return true
	}
	rng := rand.New(rand.NewPCG(3, 5))
	for range leakDraws {
		_, _, _, _, kc, _, _, _, _, got, ok := deflateStaircase(jitterCopy(A, rng), jitterCopy(E, rng), false, false, false, level, cnt)
		if !ok || kc != k || got == nil {
			return true
		}
		for _, w := range ref {
			found := false
			for _, g := range got {
				found = found || cmplx.Abs(g-w) <= leakStableTol*cmplx.Abs(w)
			}
			if !found {
				return true
			}
		}
	}
	return false
}

// jitterCopy returns m with each entry scaled by 1 + δ, δ uniform in
// [−4ε, 4ε] from rng.
func jitterCopy(m *mat.Dense, rng *rand.Rand) *mat.Dense {
	c := mat.DenseCopyOf(m)
	raw := c.RawMatrix()
	for i := range raw.Rows {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for j := range row {
			row[j] *= 1 + 4*eps()*(2*rng.Float64()-1)
		}
	}
	return c
}

// nearInfiniteEigs returns the cnt generalized eigenvalues of the leading
// k×k block of the pencil (ad, ed) (stride n) of least |β|/|α|; ok is false
// when QZ fails or one of them is infinite or not finite.
func nearInfiniteEigs(ad, ed []float64, n, k, cnt int) (eig []complex128, ok bool) {
	alphar, alphai, beta, ok := qzEigenvalues(k, ad, n, ed, n)
	if !ok {
		return nil, false
	}
	idx := make([]int, k)
	for i := range idx {
		idx[i] = i
	}
	ratio := func(i int) float64 { return math.Abs(beta[i]) / math.Hypot(alphar[i], alphai[i]) }
	slices.SortFunc(idx, func(i, j int) int { return cmp.Compare(ratio(i), ratio(j)) })
	for _, i := range idx[:cnt] {
		if beta[i] == 0 {
			return nil, false
		}
		w := complex(alphar[i]/beta[i], alphai[i]/beta[i])
		if cmplx.IsInf(w) || cmplx.IsNaN(w) {
			return nil, false
		}
		eig = append(eig, w)
	}
	return eig, true
}

// leakDraws and leakStableTol set leakedZeros (leakDraws also
// jitteredEigenvalues). Over the package tests and 3000 chains beside a pole small in E (cond(U),
// cond(V) up to 1e4), the eigenvalues of finite poles moved by at most
// 5.6e-6 relative, those of leaked rounding by at least 2.9e-2.
const (
	leakDraws     = 3
	leakStableTol = 1e-3
)

// deflatedRank decides the rank of a deflated E block of the staircase that
// descriptorRank finds ambiguous: Z of each level is the row space of a block
// of A with σ_min σa, so rounding of A, ε·‖A‖, turns it by ε·‖A‖/σa and
// leaks into the next E block as that times its rounding tol. A singular
// value up to leak·tol is zero, leak ≤ qzInfiniteGap keeping the nonzero
// side of descriptorRank as is.
func deflatedRank(sv []float64, tol, leak float64) (rank int, ok bool) {
	for j, s := range sv {
		switch {
		case s > qzInfiniteGap*tol:
		case s <= leak*tol:
			return j, true
		default:
			return 0, false
		}
	}
	return len(sv), true
}

// stiffChainRank is descriptorRank of a deflated E block of
// deflateStiffChain, tol = 1e3·n·ε·σ₁(E): the stiff singular values of the
// full E, stiff of them, are finite as in MATLAB's rank (D62TAA) and stay
// in each deflated E block, so up to stiff values in (tol,
// qzInfiniteGap·tol] are nonzero; more there are ambiguous (ok false). A
// value at most tol is zero.
func stiffChainRank(sv []float64, tol float64, stiff int) (rank int, ok bool) {
	for j, s := range sv {
		switch {
		case s > qzInfiniteGap*tol:
		case s <= tol:
			return j, true
		case stiff > 0:
			stiff--
		default:
			return 0, false
		}
	}
	return len(sv), true
}

// clearlyNonsingular reports whether the leading k×k block of e (stride
// lde) has σ_min > 10·bound, from its LU and estimated 1-norm reciprocal
// condition number: σ_min ≥ rcond·‖B‖₁/√k, with a factor 10 for an
// estimate too large. scratch (k×k) is overwritten.
func clearlyNonsingular(e []float64, lde, k int, scratch []float64, bound float64) bool {
	copyStrided(scratch, k, e, lde, k, k)
	work := make([]float64, 4*k)
	iwork := make([]int, k)
	anorm := impl.Dlange(lapack.MaxColumnSum, k, k, scratch, k, work)
	if anorm == 0 || !impl.Dgetrf(k, k, scratch, k, iwork) {
		return false
	}
	rcond := impl.Dgecon(lapack.MaxColumnSum, k, scratch, k, anorm, work, iwork)
	return rcond*anorm/math.Sqrt(float64(k)) > 10*bound
}

func denseOrNil(d []float64, n int) *mat.Dense {
	if d == nil {
		return nil
	}
	return mat.NewDense(n, n, d)
}

// decomposeGeneralizedQZ is decomposeGeneralized on the whole pencil:
// eigenvalues with |β| ≤ 100·n·ε·|α|·‖E‖/‖A‖ are infinite; a large retained
// one that certifyFiniteRetained cannot certify finite fails closed.
func decomposeGeneralizedQZ(sys *System, isGroup1 func(complex128) bool) (group1, group2 *System, err error) {
	policy := newRealizationTransformPolicy(sys)
	n, _, _ := sys.Dims()
	s := make([]float64, n*n)
	t := make([]float64, n*n)
	q := make([]float64, n*n)
	z := make([]float64, n*n)
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	normA, normE := mat.Norm(sys.A, 2), mat.Norm(sys.E, 2)
	infTol := qzInfiniteTol(100*float64(n)*eps(), normA, normE)
	selct := func(ar, ai, b float64) bool {
		return !qzInfinite(ar, ai, b, infTol) && isGroup1(complex(ar/b, ai/b))
	}
	n1, recip, ok := orderedQZ(n, sys.A.RawMatrix(), sys.E.RawMatrix(), selct, s, t, q, z, alphar, alphai, beta)
	if !ok {
		return nil, nil, fmt.Errorf("ordered generalized Schur failed: %w", ErrSchurFailed)
	}
	ninf := 0
	for i := range n {
		if beta[i] == 0 && alphar[i] == 0 && alphai[i] == 0 {
			return nil, nil, fmt.Errorf("singular pencil (A, E): %w", ErrDescriptorSingular)
		}
		if qzInfinite(alphar[i], alphai[i], beta[i], infTol) {
			ninf++
		}
	}
	if n1 > 0 {
		retol, err := certifyFiniteRetained(sys.A, sys.E, n, n1, ninf, alphar, alphai, beta, qzInfiniteTol(qzLargeRatio, normA, normE), normA)
		if err != nil {
			return nil, nil, err
		}
		if retol > 0 {
			infTol = retol
			n1, recip, ok = orderedQZ(n, sys.A.RawMatrix(), sys.E.RawMatrix(), selct, s, t, q, z, alphar, alphai, beta)
			if !ok {
				return nil, nil, fmt.Errorf("ordered generalized Schur failed: %w", ErrSchurFailed)
			}
		}
	}
	if n1 == 0 {
		return modesAllInGroup2(policy, sys)
	}
	return splitGeneralizedSchur(policy, sys, s, t, q, z, n1, recip)
}

// splitGeneralizedSchur decouples the n1 leading modes of the generalized
// Schur form Qᵀ(A, E)Z = (S, T) of orderedQZ (recip as it returns) from the
// rest; see decomposeGeneralized.
func splitGeneralizedSchur(policy realizationTransformPolicy, sys *System, s, t, q, z []float64, n1 int, recip bool) (group1, group2 *System, err error) {
	n, m, p := sys.Dims()
	n2 := n - n1

	bq := make([]float64, n*m)
	cz := make([]float64, p*n)
	bRaw, cRaw := sys.B.RawMatrix(), sys.C.RawMatrix()
	qGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: q}
	zGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: z}
	if m > 0 {
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, qGen, blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
			0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bq})
	}
	if p > 0 {
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data}, zGen,
			0, blas64.General{Rows: p, Cols: n, Stride: n, Data: cz})
	}

	if n2 > 0 {
		r := make([]float64, n1*n2)
		l := make([]float64, n1*n2)
		for i := range n1 {
			for j := range n2 {
				r[i*n2+j] = -s[i*n+n1+j]
				l[i*n2+j] = -t[i*n+n1+j]
			}
		}
		scale, ok := qzSylvester(n, n1, recip, s, t, r, l)
		if !ok || scale < 1e-12 {
			return nil, nil, fmt.Errorf("retained and discarded modes cannot be separated reliably: %w", ErrSchurFailed)
		}
		for i := range r {
			r[i] /= scale
			l[i] /= scale
		}
		lGen := blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: l}
		rGen := blas64.General{Rows: n1, Cols: n2, Stride: n2, Data: r}
		if m > 0 {
			blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, lGen, blas64.General{Rows: n2, Cols: m, Stride: m, Data: bq[n1*m:]},
				1, blas64.General{Rows: n1, Cols: m, Stride: m, Data: bq})
		}
		if p > 0 {
			blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: p, Cols: n1, Stride: n, Data: cz}, rGen,
				1, blas64.General{Rows: p, Cols: n2, Stride: n, Data: cz[n1:]})
		}
	}

	A1 := extractModalBlock(s, n, 0, n1, 0, n1)
	B1 := extractModalBlock(bq, m, 0, n1, 0, m)
	if recip {
		var lu mat.LU
		lu.Factorize(extractModalBlock(t, n, 0, n1, 0, n1))
		if err := lu.SolveTo(A1, false, A1); err != nil {
			return nil, nil, fmt.Errorf("retained modes: %w", ErrSchurFailed)
		}
		if m > 0 {
			if err := lu.SolveTo(B1, false, B1); err != nil {
				return nil, nil, fmt.Errorf("retained modes: %w", ErrSchurFailed)
			}
		}
	} else {
		t11 := blas64.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit, N: n1, Stride: n, Data: t}
		blas64.Trsm(blas.Left, blas.NoTrans, 1, t11, A1.RawMatrix())
		if m > 0 {
			blas64.Trsm(blas.Left, blas.NoTrans, 1, t11, B1.RawMatrix())
		}
	}
	group1, err = policy.resultWithZeroFeedthrough(A1, B1, extractModalBlock(cz, n, 0, p, 0, n1))
	if err != nil {
		return nil, nil, err
	}
	if n2 == 0 {
		return modesAllInGroup1(policy, group1)
	}
	group2, err = policy.resultWithOriginalFeedthrough(
		extractModalBlock(s, n, n1, n, n1, n), extractModalBlock(bq, m, n1, n, 0, m), extractModalBlock(cz, n, 0, p, n1, n))
	if err != nil {
		return nil, nil, err
	}
	group2.E = extractModalBlock(t, n, n1, n, n1, n)
	return group1, group2, nil
}

// orderSchurGroupFirst reorders the real Schur form (t, z) so the blocks whose
// eigenvalues satisfy inGroup lead, and returns their total size.
func orderSchurGroupFirst(t, z []float64, n int, inGroup func(complex128) bool) (int, error) {
	work := make([]float64, n)
	placed := 0
	for i := 0; i < n; {
		size := schurBlockSize(t, n, i)
		if !inGroup(schurBlockEigenvalue(t, n, i)) {
			i += size
			continue
		}
		if i != placed {
			if _, _, ok := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, i, placed, work); !ok {
				return 0, fmt.Errorf("reordering the Schur form: %w", ErrSchurFailed)
			}
		}
		// Dtrexc may split a moved 2x2 block (e.g. a double real pole stored as
		// a ±1e-16i pair) into two 1x1 blocks; classify each part.
		end := placed + size
		for j := placed; j < end; {
			bs := schurBlockSize(t, n, j)
			if inGroup(schurBlockEigenvalue(t, n, j)) {
				if j != placed {
					if _, _, ok := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, j, placed, work); !ok {
						return 0, fmt.Errorf("reordering the Schur form: %w", ErrSchurFailed)
					}
				}
				placed += bs
			}
			j += bs
		}
		i += size
	}
	return placed, nil
}

func schurBlockEigenvalue(t []float64, n, i int) complex128 {
	if schurBlockSize(t, n, i) == 2 {
		return schurBlock2x2Eig(t, n, i)
	}
	return complex(t[i*n+i], 0)
}

func schurEigenvaluesRaw(t []float64, n int) []complex128 {
	evals := make([]complex128, n)
	i := 0
	for i < n {
		if i+1 < n && t[(i+1)*n+i] != 0 {
			evals[i], evals[i+1] = schur2x2Eigenvalues(t[i*n+i], t[i*n+i+1], t[(i+1)*n+i], t[(i+1)*n+i+1])
			i += 2
		} else {
			evals[i] = complex(t[i*n+i], 0)
			i++
		}
	}
	return evals
}

// schur2x2Eigenvalues returns the eigenvalues of [a b; c d] via Dlanv2, which
// avoids the trace/determinant cancellation of the quadratic formula. Complex
// pairs come positive imaginary part first, real pairs in descending order.
func schur2x2Eigenvalues(a, b, c, d float64) (complex128, complex128) {
	_, _, _, _, rt1r, rt1i, rt2r, rt2i, _, _ := impl.Dlanv2(a, b, c, d)
	if rt1i == 0 && rt1r < rt2r {
		rt1r, rt2r = rt2r, rt1r
	}
	return complex(rt1r, rt1i), complex(rt2r, rt2i)
}

func isConjugate(a, b complex128) bool {
	return math.Abs(real(a)-real(b)) < 1e-10*math.Max(1, math.Abs(real(a))) &&
		math.Abs(imag(a)+imag(b)) < 1e-10*math.Max(1, math.Abs(imag(a)))
}

// orderedQZ writes the generalized Schur form Qᵀ(A, E)Z = (S, T) of the
// n×n pencil (a, e) to s and t, Q to q and Z to z, with eigenvalues
// (α_r + iα_i)/β, ordered so that the n1 selected by selct lead. Dgges,
// like LAPACK DTGEX2, rejects a swap of blocks decided by rounding, also
// for well-separated eigenvalues; the reorder is then retried on the
// reciprocal pencil (E, A), which has the same deflating subspaces, and
// recip is set: S is then upper triangular and T quasi-triangular. The
// retry is kept only when its eigenvalues match the first run's one to
// one within qzRetryTol (chordal), each selected alike; otherwise ok is
// false.
func orderedQZ(n int, a, e blas64.General, selct func(ar, ai, b float64) bool,
	s, t, q, z, alphar, alphai, beta []float64) (n1 int, recip, ok bool) {
	copyStrided(s, n, a.Data, a.Stride, n, n)
	copyStrided(t, n, e.Data, e.Stride, n, n)
	bwork := make([]bool, n)
	var query [1]float64
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selct, n, s, n, t, n,
		alphar, alphai, beta, q, n, z, n, query[:], -1, bwork)
	work := make([]float64, int(query[0]))
	n1, ok = impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selct, n, s, n, t, n,
		alphar, alphai, beta, q, n, z, n, work, len(work), bwork)
	if ok || n1 == 0 {
		return n1, false, ok
	}

	first := make([]float64, 3*n)
	copy(first, alphar)
	copy(first[n:], alphai)
	copy(first[2*n:], beta)
	reciprocal := func(ar, ai, b float64) bool {
		return selct(qzReciprocal(ar, ai, b))
	}
	copyStrided(s, n, a.Data, a.Stride, n, n)
	copyStrided(t, n, e.Data, e.Stride, n, n)
	n1r, okr := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, reciprocal, n, t, n, s, n,
		alphar, alphai, beta, q, n, z, n, work, len(work), bwork)
	for i := range n {
		alphar[i], alphai[i], beta[i] = qzReciprocal(alphar[i], alphai[i], beta[i])
	}
	if !okr || n1r != n1 || !qzRetryConsistent(first, alphar, alphai, beta, selct) {
		return n1, false, false
	}
	return n1, true, true
}

// qzReciprocal maps an eigenvalue μ = (α_r + iα_i)/β of the reciprocal
// pencil (E, A) to λ = 1/μ of (A, E), as (α_r, α_i, β) with β ≥ 0.
func qzReciprocal(ar, ai, b float64) (float64, float64, float64) {
	mod := math.Hypot(ar, ai)
	if mod == 0 {
		return b, 0, 0
	}
	return b * (ar / mod), -b * (ai / mod), mod
}

// qzInfiniteTol returns the bound on |β|/|α| below which an eigenvalue of
// the pencil (A, E), with Frobenius norms normA and normE, counts as
// infinite: |λ| ≥ ‖A‖/(ratio·‖E‖), MATLAB's |β| ≤ tol·|α| on the
// norm-scaled pencil. Decomposition uses ratio 100·n·ε, since a
// rank-deficient E leaves rounding-level β, not 0.
func qzInfiniteTol(ratio, normA, normE float64) float64 {
	if normA == 0 {
		return 0
	}
	return ratio * normE / normA
}

func qzInfinite(ar, ai, b, tol float64) bool {
	return math.Abs(b) <= tol*math.Hypot(ar, ai)
}

// qzLargeRatio is the norm-scaled |β|/|α| below which a retained
// eigenvalue may be an infinite one that QZ resolved only to
// |β|/|α| ~ ε/σ_min(A22) (index 1, A22 nearly singular) or ε^(1/k)
// (an infinite Jordan chain of length k ≥ 2).
const qzLargeRatio = 1e-3

// certifyFiniteRetained checks the n1 retained eigenvalues of the
// generalized Schur form of (a, e) once one is large (|β| ≤ largeTol·|α|).
// They are certified finite when the pencil has index 1, so Ulᵀ·A·Vr is
// well conditioned for the null spaces Ul, Vr of E, and QZ marked all
// d = n − rank(E) of its infinite eigenvalues (ninf). When it marked
// fewer, the d eigenvalues of least |β|/|α| are the infinite ones and
// infTol, if nonzero, is the |β|/|α| bound that selects them; anything
// else fails closed.
func certifyFiniteRetained(a, e *mat.Dense, n, n1, ninf int, alphar, alphai, beta []float64, largeTol, normA float64) (infTol float64, err error) {
	large := false
	for i := range n1 {
		large = large || qzInfinite(alphar[i], alphai[i], beta[i], largeTol)
	}
	if !large {
		return 0, nil
	}
	leak := fmt.Errorf("a retained mode may be infinite: %w", ErrSchurFailed)
	d, index1 := pencilIndexOne(a, e, n, normA)
	if d == 0 {
		return 0, nil
	}
	if !index1 {
		return 0, leak
	}
	if ninf >= d {
		return 0, nil
	}
	ratio := make([]float64, n)
	for i := range n {
		ratio[i] = math.Abs(beta[i]) / math.Hypot(alphar[i], alphai[i])
	}
	slices.Sort(ratio)
	next := math.Inf(1)
	if d < n {
		next = ratio[d]
	}
	if ratio[d-1] > largeTol || !(next > qzInfiniteGap*ratio[d-1]) {
		return 0, leak
	}
	return ratio[d-1] * math.Sqrt(qzInfiniteGap), nil
}

// pencilIndexOne returns d = n − rank(E) (descriptorRank, MATLAB rank) and
// whether the pencil (a, e) has index at most 1: Ulᵀ·A·Vr is well
// conditioned, σ_min > tol·normA, for the null spaces Ul, Vr of E, with
// tol = √ε, or when E has a stiff singular value σs the larger of
// stiffIndexTol and stiffIndexCoef·n·ε·σ₁(E)/σs. Only
// then does E have exactly d infinite eigenvalues; an infinite Jordan chain
// leaves Ulᵀ·A·Vr singular. A failed SVD reports index1 false.
func pencilIndexOne(a, e *mat.Dense, n int, normA float64) (d int, index1 bool) {
	sv, sv22, rank, stiff, ok := nullCoupling(a, e, n)
	if !ok {
		return n, false
	}
	d = n - rank
	if d == 0 {
		return 0, true
	}
	tol := math.Sqrt(eps())
	if stiff {
		tol = max(stiffIndexTol, stiffIndexCoef*float64(n)*eps()*sv[0]/sv[rank-1])
	}
	return d, sv22 != nil && sv22[d-1] > tol*normA
}

// infiniteChainCount returns the number of infinite Jordan chains of length
// at least 2 of (a, e) when E's rank is not stiff: the singular values of
// Ulᵀ·A·Vr (nullCoupling) at most 100·n·ε·normA, which a chain leaves at
// rounding level (Ulᵀ·A·Vr couples the tail and head of each chain) while
// an index-1 pencil with ill-conditioned coordinates keeps them orders of
// magnitude above it. It is -1 when undecided (failed SVD, stiff rank).
func infiniteChainCount(a, e *mat.Dense, n int, normA float64) int {
	_, sv22, rank, stiff, ok := nullCoupling(a, e, n)
	if !ok || stiff || rank < n && sv22 == nil {
		return -1
	}
	c := 0
	for _, s := range sv22 {
		if s <= 100*float64(n)*eps()*normA {
			c++
		}
	}
	return c
}

// nullCoupling returns the singular values sv of E, its rank
// (descriptorRank, MATLAB rank) with the stiff flag, and the singular
// values sv22 of Ulᵀ·A·Vr for the null spaces Ul, Vr of E, nil when E is
// nonsingular or their SVD fails. ok is false when the SVD of E fails.
func nullCoupling(a, e *mat.Dense, n int) (sv, sv22 []float64, rank int, stiff, ok bool) {
	var svd mat.SVD
	if !svd.Factorize(e, mat.SVDFull) {
		return nil, nil, 0, false, false
	}
	sv = svd.Values(nil)
	rank, stiff, _ = descriptorRank(sv, sv[0], n, true)
	if rank == n {
		return sv, nil, rank, stiff, true
	}
	var u, v mat.Dense
	svd.UTo(&u)
	svd.VTo(&v)
	var av, a22 mat.Dense
	av.Mul(a, v.Slice(0, n, rank, n))
	a22.Mul(u.Slice(0, n, rank, n).T(), &av)
	var svd22 mat.SVD
	if !svd22.Factorize(&a22, mat.SVDNone) {
		return sv, nil, rank, stiff, true
	}
	return sv, svd22.Values(nil), rank, stiff, true
}

// stiffIndexTol and stiffIndexCoef bound pencilIndexOne when E has a stiff
// singular value σs: rounding tilts the null spaces of E toward the stiff
// direction by ~n·ε·σ₁/σs, which leaves Ulᵀ·A·Vr of an index-2 chain up to
// ~3e-4·n·ε·σ₁/σs·‖A‖ instead of rounding level (7e-5·‖A‖ with σs 3× the
// rounding); an index-1 pencil keeps it near ‖A‖.
const (
	stiffIndexTol  = 1e-5
	stiffIndexCoef = 3e-3
)

// qzInfiniteGap is the least factor between the |β|/|α| of the infinite
// and of the finite eigenvalues for certifyFiniteRetained to tell them apart,
// and the least factor above a rank tolerance for deflateInfinite to call a
// singular value nonzero.
const qzInfiniteGap = 1e4

// descriptorRank returns the rank of E from its descending singular values
// sv, n the order of the pencil and scale ≈ σ_max(E), with tol =
// 1e3·n·ε·scale and floor = n·ε·scale. For the whole E (full) the rank is
// MATLAB's: the singular values above floor = n·eps(scale), with eps(scale)
// the float spacing at scale (ε·scale/2 just below a power of two). A
// nonsingular E thus has no infinite modes, however stiff its poles, and
// one in (floor, qzInfiniteGap·tol] of a singular E sets stiff: a stiff
// finite pole beside the rounding of an infinite one, which the staircase
// cannot separate reliably. The deflated
// blocks of the staircase carry more rounding: a singular value at most tol
// is zero, one above qzInfiniteGap·tol is nonzero and one in between is
// ambiguous (ok false).
func descriptorRank(sv []float64, scale float64, n int, full bool) (rank int, stiff, ok bool) {
	floor := float64(n) * eps() * scale
	tol := 1e3 * floor
	if full {
		floor = float64(n) * (math.Nextafter(scale, math.Inf(1)) - scale)
	}
	for j, s := range sv {
		switch {
		case s <= floor:
			return j, stiff, true
		case s > qzInfiniteGap*tol:
		case full:
			stiff = true
		case s <= tol:
			return j, false, true
		default:
			return 0, false, false
		}
	}
	return len(sv), false, true
}

// qzSylvester solves S11·R − L·S22 = scale·C, T11·R − L·T22 = scale·F for
// the n1-leading split of the generalized Schur form (s, t) of orderedQZ,
// with C, F given in r, l and overwritten by R, L.
func qzSylvester(n, n1 int, recip bool, s, t, r, l []float64) (scale float64, ok bool) {
	n2 := n - n1
	if recip {
		s, t = t, s
		for i := range r {
			r[i], l[i] = l[i], r[i]
		}
	}
	iwork := make([]int, n+6)
	var query [1]float64
	impl.Dtgsyl(blas.NoTrans, 0, n1, n2, s, n, s[n1*n+n1:], n, r, n2, t, n, t[n1*n+n1:], n, l, n2, query[:], -1, iwork)
	work := make([]float64, max(1, int(query[0])))
	scale, _, ok = impl.Dtgsyl(blas.NoTrans, 0, n1, n2, s, n, s[n1*n+n1:], n, r, n2, t, n, t[n1*n+n1:], n, l, n2, work, len(work), iwork)
	return scale, ok
}

// qzRetryTol bounds the chordal distance between the eigenvalues of the
// first and the reciprocal QZ of orderedQZ.
const qzRetryTol = 1e-8

// qzRetryConsistent reports whether the eigenvalues (alphar, alphai, beta)
// match those in first (α_r, α_i, β blocks) one to one within qzRetryTol,
// each selected alike by selct.
func qzRetryConsistent(first, alphar, alphai, beta []float64, selct func(ar, ai, b float64) bool) bool {
	n := len(alphar)
	used := make([]bool, n)
	for i := range n {
		ai := complex(alphar[i], alphai[i])
		ni := math.Hypot(math.Hypot(alphar[i], alphai[i]), beta[i])
		best, bestD := -1, math.Inf(1)
		for j := range n {
			if used[j] {
				continue
			}
			aj := complex(first[j], first[n+j])
			nj := math.Hypot(math.Hypot(first[j], first[n+j]), first[2*n+j])
			d := cmplx.Abs(ai*complex(first[2*n+j], 0)-aj*complex(beta[i], 0)) / (ni * nj)
			if d < bestD {
				best, bestD = j, d
			}
		}
		if !(bestD <= qzRetryTol) ||
			selct(alphar[i], alphai[i], beta[i]) != selct(first[best], first[n+best], first[2*n+best]) {
			return false
		}
		used[best] = true
	}
	return true
}
