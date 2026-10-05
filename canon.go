package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// CanonForm selects the canonical realization computed by Canon, as the
// MATLAB canon type argument. The zero value selects CanonModal, the MATLAB
// default.
type CanonForm string

const (
	// CanonModal is the real modal form: A is block diagonal with 1×1 blocks
	// for real eigenvalues and 2×2 blocks [σ ω; -ω σ] for complex pairs,
	// ordered by ascending eigenvalue magnitude.
	CanonModal CanonForm = "modal"
	// CanonCompanion is the MATLAB companion form: A has ones on the
	// subdiagonal and the negated characteristic polynomial coefficients in
	// its last column. It needs a model controllable from its first input.
	CanonCompanion CanonForm = "companion"
)

// CanonResult holds the outputs of MATLAB [csys,T] = canon(sys,type): the
// canonical state is xc = T·x, so Sys.A = T·A·T⁻¹, Sys.B = T·B and
// Sys.C = C·T⁻¹.
type CanonResult struct {
	Sys *System
	T   *mat.Dense
}

// Canon computes a canonical state-space realization of sys, as MATLAB
// canon; see https://www.mathworks.com/help/control/ref/dynamicsystem.canon.html.
// The modal form uses an eigenvector basis and falls back to an ordered real
// Schur form (quasi-triangular, not block diagonal) when the eigenvectors
// are ill-conditioned. The companion form returns ErrSingularTransform when
// sys is not controllable from its first input. Descriptor and delayed
// models are rejected; a model with no states returns ErrDimensionMismatch.
func Canon(sys *System, form CanonForm) (*CanonResult, error) {
	if err := requireFiniteSystem("Canon", sys); err != nil {
		return nil, err
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Canon"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Canon"); err != nil {
		return nil, err
	}
	if n, _, _ := sys.Dims(); n == 0 {
		return nil, fmt.Errorf("Canon: system has no states: %w", ErrDimensionMismatch)
	}
	switch form {
	case CanonModal, "":
		return canonModal(sys)
	case CanonCompanion:
		return canonCompanion(sys)
	default:
		return nil, fmt.Errorf("Canon: unknown form %q: %w", form, ErrInvalidArgument)
	}
}

type eigBlock struct {
	mag   float64
	real1 float64
	imag1 float64
	idx   int
}

func canonModal(sys *System) (*CanonResult, error) {
	policy := newRealizationTransformPolicy(sys)
	res, err := canonModalEig(sys, policy)
	if err == nil {
		return res, nil
	}
	return canonModalSchur(sys, policy)
}

// canonModalEig attempts the eigendecomposition-based modal form.
// Returns error if eigenvector matrix is too ill-conditioned.
func canonModalEig(sys *System, policy realizationTransformPolicy) (*CanonResult, error) {
	n, m, p := policy.n, policy.m, policy.p
	var eig mat.Eigen
	ok := eig.Factorize(sys.A, mat.EigenRight)
	if !ok {
		return nil, fmt.Errorf("eigendecomposition did not converge: %w", ErrSchurFailed)
	}

	vals := eig.Values(nil)
	var vecs mat.CDense
	eig.VectorsTo(&vecs)

	blocks := make([]eigBlock, 0, n)
	used := make([]bool, n)
	for i := range n {
		if used[i] {
			continue
		}
		eigMag := cmplx.Abs(vals[i])
		realTol := 1e-10 * math.Max(1, eigMag)
		conj := -1
		if imag(vals[i]) != 0 {
			for j := i + 1; j < n; j++ {
				if !used[j] && isConjugate(vals[i], vals[j]) {
					conj = j
					break
				}
			}
		}
		if conj >= 0 {
			blocks = append(blocks, eigBlock{
				mag:   eigMag,
				real1: real(vals[i]),
				imag1: math.Abs(imag(vals[i])),
				idx:   i,
			})
			used[i] = true
			used[conj] = true
		} else if math.Abs(imag(vals[i])) < realTol {
			blocks = append(blocks, eigBlock{
				mag:   math.Abs(real(vals[i])),
				real1: real(vals[i]),
				imag1: 0,
				idx:   i,
			})
			used[i] = true
		} else {
			return nil, fmt.Errorf("complex eigenvalue without conjugate pair: %w", ErrConjugatePairs)
		}
	}

	sort.Slice(blocks, func(i, j int) bool {
		return blocks[i].mag < blocks[j].mag
	})

	tData := make([]float64, n*n)
	col := 0
	for _, blk := range blocks {
		if blk.imag1 == 0 {
			for row := range n {
				tData[row*n+col] = real(vecs.At(row, blk.idx))
			}
			col++
		} else {
			for row := range n {
				v := vecs.At(row, blk.idx)
				tData[row*n+col] = real(v)
				tData[row*n+col+1] = imag(v)
			}
			col += 2
		}
	}

	T := mat.NewDense(n, n, tData)

	var lu mat.LU
	lu.Factorize(T)

	cond := lu.Cond()
	if math.IsNaN(cond) || math.IsInf(cond, 1) || cond > 1e12 {
		return nil, fmt.Errorf("eigenvector matrix ill-conditioned (κ=%.1e): %w", cond, ErrSingularTransform)
	}

	AT := mat.NewDense(n, n, nil)
	AT.Mul(sys.A, T)
	Anew := mat.NewDense(n, n, nil)
	if err := lu.SolveTo(Anew, false, AT); err != nil {
		return nil, fmt.Errorf("Canon: eigenvector basis solve failed: %w", ErrSingularTransform)
	}

	Bnew := newDense(n, m)
	if m > 0 {
		if err := lu.SolveTo(Bnew, false, sys.B); err != nil {
			return nil, fmt.Errorf("Canon: eigenvector basis solve failed: %w", ErrSingularTransform)
		}
	}

	Cnew := mulDims(p, n, sys.C, T)

	newSys, err := policy.resultWithOriginalFeedthrough(Anew, Bnew, Cnew)
	if err != nil {
		return nil, err
	}
	Tinv := mat.NewDense(n, n, nil)
	if err := lu.SolveTo(Tinv, false, eyeDense(n)); err != nil {
		return nil, fmt.Errorf("Canon: eigenvector basis solve failed: %w", ErrSingularTransform)
	}
	return &CanonResult{Sys: newSys, T: Tinv}, nil
}

// canonModalSchur computes a quasi-modal form via ordered real Schur decomposition.
// Numerically stable even for near-repeated eigenvalues.
// A_modal is quasi-upper-triangular: 1×1 blocks for real eigenvalues,
// 2×2 blocks for complex pairs, ordered by eigenvalue magnitude.
func canonModalSchur(sys *System, policy realizationTransformPolicy) (*CanonResult, error) {
	n, m, p := policy.n, policy.m, policy.p
	aRaw := sys.A.RawMatrix()

	t := make([]float64, n*n)
	copyStrided(t, n, aRaw.Data, aRaw.Stride, n, n)

	z := make([]float64, n*n)
	wr := make([]float64, n)
	wi := make([]float64, n)
	bwork := make([]bool, n)

	workQuery := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		n, t, n, wr, wi, z, n, workQuery, -1, bwork)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)

	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		n, t, n, wr, wi, z, n, work, lwork, bwork)
	if !ok {
		return nil, fmt.Errorf("Canon: real Schur decomposition did not converge: %w", ErrSchurFailed)
	}

	if err := schurSortByMagnitude(t, z, n); err != nil {
		return nil, err
	}

	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()

	zGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: z}

	bNew := make([]float64, n*m)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, zGen,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bNew})

	cNew := make([]float64, p*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		zGen,
		0, blas64.General{Rows: p, Cols: n, Stride: n, Data: cNew})

	Anew := mat.NewDense(n, n, t)
	Bnew := mat.NewDense(n, m, bNew)
	Cnew := mat.NewDense(p, n, cNew)
	newSys, err := policy.resultWithOriginalFeedthrough(Anew, Bnew, Cnew)
	if err != nil {
		return nil, err
	}

	T := mat.NewDense(n, n, nil)
	T.CloneFrom(mat.NewDense(n, n, z).T())
	return &CanonResult{Sys: newSys, T: T}, nil
}

// schurSortByMagnitude reorders Schur blocks by eigenvalue magnitude (ascending).
func schurSortByMagnitude(t, z []float64, n int) error {
	trexcWork := make([]float64, n)
	nPlaced := 0
	for nPlaced < n {
		evals := schurEigenvaluesRaw(t, n)

		bestIdx := nPlaced
		bestMag := cmplx.Abs(evals[nPlaced])
		i := nPlaced
		for i < n {
			blockSize := 1
			if i+1 < n && t[(i+1)*n+i] != 0 {
				blockSize = 2
			}
			mag := cmplx.Abs(evals[i])
			if mag < bestMag {
				bestMag = mag
				bestIdx = i
			}
			i += blockSize
		}

		if bestIdx != nPlaced {
			_, _, ok := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, bestIdx, nPlaced, trexcWork)
			if !ok {
				return fmt.Errorf("Canon: Schur block reordering failed: %w", ErrSchurFailed)
			}
		}

		if nPlaced+1 < n && t[(nPlaced+1)*n+nPlaced] != 0 {
			nPlaced += 2
		} else {
			nPlaced++
		}
	}
	return nil
}

func canonCompanion(sys *System) (*CanonResult, error) {
	policy := newRealizationTransformPolicy(sys)
	n, m, p := policy.n, policy.m, policy.p
	if m == 0 {
		return nil, fmt.Errorf("Canon: companion form needs an input: %w", ErrDimensionMismatch)
	}

	K := mat.NewDense(n, n, nil)
	col := mat.NewVecDense(n, nil)
	for i := range n {
		col.SetVec(i, sys.B.At(i, 0))
	}
	for j := range n {
		K.SetCol(j, col.RawVector().Data)
		if j+1 < n {
			col.MulVec(sys.A, mat.VecDenseCopyOf(col))
		}
	}

	var lu mat.LU
	lu.Factorize(K)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("Canon: system not controllable from its first input: %w", ErrSingularTransform)
	}

	AK := mat.NewDense(n, n, nil)
	AK.Mul(sys.A, K)
	Anew := mat.NewDense(n, n, nil)
	if err := lu.SolveTo(Anew, false, AK); err != nil {
		return nil, fmt.Errorf("Canon: %w", ErrSingularTransform)
	}
	Bnew := mat.NewDense(n, m, nil)
	if err := lu.SolveTo(Bnew, false, sys.B); err != nil {
		return nil, fmt.Errorf("Canon: %w", ErrSingularTransform)
	}
	Cnew := mulDims(p, n, sys.C, K)
	T := mat.NewDense(n, n, nil)
	if err := lu.SolveTo(T, false, eyeDense(n)); err != nil {
		return nil, fmt.Errorf("Canon: %w", ErrSingularTransform)
	}

	newSys, err := policy.resultWithOriginalFeedthrough(Anew, Bnew, Cnew)
	if err != nil {
		return nil, err
	}
	return &CanonResult{Sys: newSys, T: T}, nil
}
