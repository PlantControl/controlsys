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

// decomposeByEigenvalues splits sys additively, G = G1 + G2, where G1 holds the
// modes selected by isGroup1 and the feedthrough D goes to G1 when
// feedthroughInGroup1 is set, else to G2. Nonsingular-E descriptors are made
// explicit first so the split follows the generalized eigenvalues of (A, E).
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

// decomposeGeneralized splits a singular-E descriptor with the ordered
// generalized Schur form Qᵀ(A, E)Z = (S, T), selected finite eigenvalues
// leading, then removes the coupling blocks with the generalized Sylvester
// equation S11 R - L S22 = -S12, T11 R - L T22 = -T12. G1 is returned explicit
// (T11 is nonsingular); G2 keeps E = T22. The infinite structure is deflated
// to the trailing block first (deflateInfinite), so QZ never has to resolve
// an infinite Jordan chain, which it does only to |β|/|α| ~ ε^(1/k).
// When the rank decisions are ambiguous the whole pencil goes to QZ
// instead (decomposeGeneralizedQZ).
func decomposeGeneralized(sys *System, isGroup1 func(complex128) bool) (group1, group2 *System, err error) {
	a, e, qd, zd, nf, ok := deflateInfinite(sys.A, sys.E, true)
	if !ok {
		return decomposeGeneralizedQZ(sys, isGroup1)
	}
	policy := newRealizationTransformPolicy(sys)
	n := sys.A.RawMatrix().Rows
	if nf == 0 {
		return modesAllInGroup2(policy, sys)
	}
	ninf := n - nf
	sf := make([]float64, nf*nf)
	tf := make([]float64, nf*nf)
	qf := make([]float64, nf*nf)
	zf := make([]float64, nf*nf)
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	aRaw, eRaw := a.RawMatrix(), e.RawMatrix()
	selct := func(ar, ai, b float64) bool {
		return b != 0 && isGroup1(complex(ar/b, ai/b))
	}
	n1, recip, ok := orderedQZ(nf, blas64.General{Rows: nf, Cols: nf, Stride: aRaw.Stride, Data: aRaw.Data},
		blas64.General{Rows: nf, Cols: nf, Stride: eRaw.Stride, Data: eRaw.Data}, selct, sf, tf, qf, zf, alphar[:nf], alphai[:nf], beta[:nf])
	if !ok {
		return nil, nil, fmt.Errorf("ordered generalized Schur failed: %w", ErrSchurFailed)
	}
	if n1 == 0 {
		return modesAllInGroup2(policy, sys)
	}

	s := make([]float64, n*n)
	t := make([]float64, n*n)
	q := make([]float64, n*n)
	z := make([]float64, n*n)
	copyStrided(s, n, sf, nf, nf, nf)
	copyStrided(t, n, tf, nf, nf, nf)
	qfGen := blas64.General{Rows: nf, Cols: nf, Stride: nf, Data: qf}
	zfGen := blas64.General{Rows: nf, Cols: nf, Stride: nf, Data: zf}
	qdRaw, zdRaw := qd.RawMatrix(), zd.RawMatrix()
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: nf, Stride: qdRaw.Stride, Data: qdRaw.Data}, qfGen,
		0, blas64.General{Rows: n, Cols: nf, Stride: n, Data: q})
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: nf, Stride: zdRaw.Stride, Data: zdRaw.Data}, zfGen,
		0, blas64.General{Rows: n, Cols: nf, Stride: n, Data: z})
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
			return nil, nil, fmt.Errorf("generalized Schur of the infinite part failed: %w", ErrSchurFailed)
		}
		copyStrided(s[nf*n+nf:], n, si, ninf, ninf, ninf)
		copyStrided(t[nf*n+nf:], n, ti, ninf, ninf, ninf)
		qiGen := blas64.General{Rows: ninf, Cols: ninf, Stride: ninf, Data: qi}
		ziGen := blas64.General{Rows: ninf, Cols: ninf, Stride: ninf, Data: zi}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: ninf, Stride: qdRaw.Stride, Data: qdRaw.Data[nf:]}, qiGen,
			0, blas64.General{Rows: n, Cols: ninf, Stride: n, Data: q[nf:]})
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: ninf, Stride: zdRaw.Stride, Data: zdRaw.Data[nf:]}, ziGen,
			0, blas64.General{Rows: n, Cols: ninf, Stride: n, Data: z[nf:]})
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
	return splitGeneralizedSchur(policy, sys, s, t, q, z, n1, recip)
}

// deflateInfinite reduces the regular pencil (A, E) by orthogonal Q, Z to
// Qᵀ(A, E)Z = ([Af Ax; 0 A∞], [Ef Ex; 0 N]), Ef nonsingular and the
// trailing n − nf eigenvalues all infinite (A∞ nonsingular, N nilpotent),
// with the staircase of Van Dooren: compress the rows of E to expose its
// null space, compress the matching rows of A onto the trailing columns,
// and repeat on the leading block until its E is nonsingular. A singular
// value of E within a factor qzInfiniteGap above the rank tolerance, or a
// near rank-deficient row block of A (a nearly singular pencil), is
// ambiguous and ok is false.
func deflateInfinite(A, E *mat.Dense, wantQZ bool) (a, e, q, z *mat.Dense, nf int, ok bool) {
	n, _ := A.Dims()
	if n == 0 {
		return nil, nil, nil, nil, 0, false
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
	tolA := ratio * mat.Norm(A, 2)
	tolE := ratio * mat.Norm(E, 2)
	k := n
	for k > 0 {
		if clearlyNonsingular(ed, n, k, sd, qzInfiniteGap*tolE) {
			break
		}
		copyStrided(sd, k, ed, n, k, k)
		if !svd(lapack.SVDAll, lapack.SVDNone, k, k) {
			return nil, nil, nil, nil, 0, false
		}
		if k == n {
			tolE = ratio * sv[0]
		}
		r := 0
		for _, s := range sv[:k] {
			if s > tolE {
				if s <= qzInfiniteGap*tolE {
					return nil, nil, nil, nil, 0, false
				}
				r++
			}
		}
		d := k - r
		if d == 0 {
			break
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
			return nil, nil, nil, nil, 0, false
		}
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
	return mat.NewDense(n, n, ad), mat.NewDense(n, n, ed), denseOrNil(qd, n), denseOrNil(zd, n), k, true
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
	var svd mat.SVD
	if !svd.Factorize(e, mat.SVDFull) {
		return 0, leak
	}
	sv := svd.Values(nil)
	rank := 0
	for _, v := range sv {
		if v > 1e3*float64(n)*eps()*sv[0] {
			rank++
		}
	}
	d := n - rank
	if d == 0 {
		return 0, nil
	}
	var u, v mat.Dense
	u.Reset()
	svd.UTo(&u)
	v.Reset()
	svd.VTo(&v)
	var av, a22 mat.Dense
	av.Mul(a, v.Slice(0, n, rank, n))
	a22.Mul(u.Slice(0, n, rank, n).T(), &av)
	var svd22 mat.SVD
	if !svd22.Factorize(&a22, mat.SVDNone) || svd22.Values(nil)[d-1] <= math.Sqrt(eps())*normA {
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

// qzInfiniteGap is the least factor between the |β|/|α| of the infinite
// and of the finite eigenvalues for certifyFiniteRetained to tell them apart.
const qzInfiniteGap = 1e4

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
