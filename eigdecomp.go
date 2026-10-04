package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
)

// decomposeByEigenvalues splits sys additively, G = G1 + G2, where G1 holds the
// modes selected by isGroup1. Descriptor models are made explicit first so the
// split follows the generalized eigenvalues of (A, E). The ordered real Schur
// form is block-diagonalized with a Sylvester solve, so G1 + G2 is exact.
func decomposeByEigenvalues(sys *System, isGroup1 func(complex128) bool) (group1, group2 *System, err error) {
	if sys.HasDelay() {
		return nil, nil, fmt.Errorf("controlsys: decomposition does not support delayed systems; use Pade/AbsorbDelay first")
	}
	sys, err = sys.ToExplicit()
	if err != nil {
		return nil, nil, fmt.Errorf("controlsys: decomposition: %w: %w", ErrDescriptorUnsupported, err)
	}
	policy := newRealizationTransformPolicy(sys)
	n, m, p := sys.Dims()
	if n == 0 {
		return sys.Copy(), policy.zeroOrderZeroFeedthrough(), nil
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
		return policy.zeroOrderZeroFeedthrough(), sys.Copy(), nil
	}
	if n1 == n {
		return policy.copyWithZeroFeedthrough(), policy.zeroOrderOriginalFeedthrough(), nil
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
	b1Raw := B1.RawMatrix()
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, xGen, B2.RawMatrix(), 1, b1Raw)

	C1 := extractModalBlock(ct, n, 0, p, 0, n1)
	C2 := extractModalBlock(ct, n, 0, p, n1, n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, C1.RawMatrix(), xGen, 1, C2.RawMatrix())

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
			a := t[i*n+i]
			b := t[i*n+i+1]
			c := t[(i+1)*n+i]
			d := t[(i+1)*n+i+1]
			tr := a + d
			det := a*d - b*c
			disc := tr*tr - 4*det
			if disc < 0 {
				re := tr / 2
				im := math.Sqrt(-disc) / 2
				evals[i] = complex(re, im)
				evals[i+1] = complex(re, -im)
			} else {
				sq := math.Sqrt(disc)
				evals[i] = complex((tr+sq)/2, 0)
				evals[i+1] = complex((tr-sq)/2, 0)
			}
			i += 2
		} else {
			evals[i] = complex(t[i*n+i], 0)
			i++
		}
	}
	return evals
}

func isConjugate(a, b complex128) bool {
	return math.Abs(real(a)-real(b)) < 1e-10*math.Max(1, math.Abs(real(a))) &&
		math.Abs(imag(a)+imag(b)) < 1e-10*math.Max(1, math.Abs(imag(a)))
}
