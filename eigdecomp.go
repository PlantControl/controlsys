package controlsys

import (
	"errors"
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// decomposeByEigenvalues splits sys additively, G = G1 + G2, where G1 holds the
// modes selected by isGroup1 and the feedthrough D goes to G1 when
// feedthroughInGroup1 is set, else to G2. Nonsingular-E descriptors are made
// explicit first so the split follows the generalized eigenvalues of (A, E).
// The ordered real Schur form is block-diagonalized with a Sylvester solve, so
// G1 + G2 is exact. Singular-E descriptors are split with the ordered
// generalized Schur form instead; infinite eigenvalues are never selected, so
// the improper/nondynamic part always stays in G2.
func decomposeByEigenvalues(sys *System, isGroup1 func(complex128) bool, feedthroughInGroup1 bool) (group1, group2 *System, err error) {
	group1, group2, err = decomposeModes(sys, isGroup1)
	if err != nil {
		return nil, nil, err
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
		return nil, nil, fmt.Errorf("controlsys: decomposition does not support delayed systems; use Pade/AbsorbDelay first")
	}
	explicit, err := sys.ToExplicit()
	if errors.Is(err, ErrDescriptorSingular) {
		return decomposeGeneralized(sys, isGroup1)
	}
	if err != nil {
		return nil, nil, fmt.Errorf("controlsys: decomposition: %w", err)
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
// (T11 is nonsingular); G2 keeps E = T22.
func decomposeGeneralized(sys *System, isGroup1 func(complex128) bool) (group1, group2 *System, err error) {
	policy := newRealizationTransformPolicy(sys)
	n, m, p := sys.Dims()
	s := make([]float64, n*n)
	t := make([]float64, n*n)
	aRaw, eRaw := sys.A.RawMatrix(), sys.E.RawMatrix()
	copyStrided(s, n, aRaw.Data, aRaw.Stride, n, n)
	copyStrided(t, n, eRaw.Data, eRaw.Stride, n, n)
	q := make([]float64, n*n)
	z := make([]float64, n*n)
	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	bwork := make([]bool, n)
	selct := func(ar, ai, b float64) bool {
		return b != 0 && isGroup1(complex(ar/b, ai/b))
	}
	var query [1]float64
	impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selct, n, s, n, t, n,
		alphar, alphai, beta, q, n, z, n, query[:], -1, bwork)
	work := make([]float64, int(query[0]))
	n1, ok := impl.Dgges(lapack.SchurHess, lapack.SchurHess, lapack.SortSelected, selct, n, s, n, t, n,
		alphar, alphai, beta, q, n, z, n, work, len(work), bwork)
	if !ok {
		return nil, nil, fmt.Errorf("controlsys: decomposition: ordered generalized Schur failed: %w", ErrSchurFailed)
	}
	for i := range n {
		if beta[i] == 0 && alphar[i] == 0 && alphai[i] == 0 {
			return nil, nil, fmt.Errorf("controlsys: decomposition: singular pencil (A, E): %w", ErrDescriptorSingular)
		}
	}
	if n1 == 0 {
		return modesAllInGroup2(policy, sys)
	}
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
		iwork := make([]int, n+6)
		impl.Dtgsyl(blas.NoTrans, 0, n1, n2, s, n, s[n1*n+n1:], n, r, n2, t, n, t[n1*n+n1:], n, l, n2, query[:], -1, iwork)
		syl := make([]float64, max(1, int(query[0])))
		scale, _, ok := impl.Dtgsyl(blas.NoTrans, 0, n1, n2, s, n, s[n1*n+n1:], n, r, n2, t, n, t[n1*n+n1:], n, l, n2, syl, len(syl), iwork)
		if !ok || scale < 1e-12 {
			return nil, nil, fmt.Errorf("controlsys: decomposition: retained and discarded modes cannot be separated reliably: %w", ErrSchurFailed)
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
	t11 := blas64.Triangular{Uplo: blas.Upper, Diag: blas.NonUnit, N: n1, Stride: n, Data: t}
	blas64.Trsm(blas.Left, blas.NoTrans, 1, t11, A1.RawMatrix())
	if m > 0 {
		blas64.Trsm(blas.Left, blas.NoTrans, 1, t11, B1.RawMatrix())
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
		if !inGroup(schurEigenvaluesRaw(t, n)[i]) {
			i += size
			continue
		}
		if i != placed {
			if _, _, ok := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, i, placed, work); !ok {
				return 0, ErrSchurFailed
			}
		}
		size = schurBlockSize(t, n, placed)
		placed += size
		i += size
	}
	return placed, nil
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
