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

type CanonForm string

const (
	CanonModal     CanonForm = "modal"
	CanonCompanion CanonForm = "companion"
)

type CanonResult struct {
	Sys *System
	T   *mat.Dense
}

func Canon(sys *System, form CanonForm) (*CanonResult, error) {
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Canon"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Canon"); err != nil {
		return nil, err
	}
	switch form {
	case CanonModal:
		return canonModal(sys)
	case CanonCompanion:
		return canonCompanion(sys)
	default:
		return nil, fmt.Errorf("controlsys: unknown canonical form %q", form)
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
	n := policy.n
	if n == 0 {
		return &CanonResult{Sys: policy.zeroOrderCopy(), T: &mat.Dense{}}, nil
	}

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
		return nil, fmt.Errorf("controlsys: eigendecomposition failed")
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
			return nil, fmt.Errorf("controlsys: complex eigenvalue without conjugate pair")
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
		return nil, fmt.Errorf("controlsys: eigenvector matrix ill-conditioned (κ=%.1e)", cond)
	}

	// Anew = T⁻¹*A*T: compute AT = A*T, then solve T*Anew = AT
	AT := mat.NewDense(n, n, nil)
	AT.Mul(sys.A, T)
	Anew := mat.NewDense(n, n, nil)
	if err := lu.SolveTo(Anew, false, AT); err != nil {
		return nil, fmt.Errorf("controlsys: solve failed: %w", ErrSingularTransform)
	}

	// Bnew = T⁻¹*B: solve T*Bnew = B
	Bnew := mat.NewDense(n, m, nil)
	if err := lu.SolveTo(Bnew, false, sys.B); err != nil {
		return nil, fmt.Errorf("controlsys: solve failed: %w", ErrSingularTransform)
	}

	Cnew := mat.NewDense(p, n, nil)
	Cnew.Mul(sys.C, T)

	newSys, err := policy.resultWithOriginalFeedthrough(Anew, Bnew, Cnew)
	if err != nil {
		return nil, err
	}

	return &CanonResult{Sys: newSys, T: T}, nil
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
		return nil, fmt.Errorf("controlsys: Schur decomposition failed: %w", ErrSchurFailed)
	}

	// Sort Schur blocks by eigenvalue magnitude (ascending)
	schurSortByMagnitude(t, z, n)

	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()

	zGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: z}

	// B_modal = Z' * B
	bNew := make([]float64, n*m)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1, zGen,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bNew})

	// C_modal = C * Z
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

	T := mat.NewDense(n, n, z)
	return &CanonResult{Sys: newSys, T: T}, nil
}

// schurSortByMagnitude reorders Schur blocks by eigenvalue magnitude (ascending).
func schurSortByMagnitude(t, z []float64, n int) {
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
				break
			}
		}

		if nPlaced+1 < n && t[(nPlaced+1)*n+nPlaced] != 0 {
			nPlaced += 2
		} else {
			nPlaced++
		}
	}
}

func canonCompanion(sys *System) (*CanonResult, error) {
	policy := newRealizationTransformPolicy(sys)
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 {
		return &CanonResult{Sys: policy.zeroOrderCopy(), T: &mat.Dense{}}, nil
	}
	if m != 1 || p != 1 {
		return nil, fmt.Errorf("controlsys: companion form requires SISO system: %w", ErrNotSISO)
	}

	// Observable companion form via similarity: T = Oc^{-1} * Ocomp
	// where Oc = obsv(A,C) and Ocomp = obsv(Ac, Cc) with Cc = e1'

	charPoly := characteristicPoly(sys.A, n)
	lead := charPoly[n]
	for i := range charPoly {
		charPoly[i] /= lead
	}

	// Build companion A: superdiagonal 1s, last row = -coeffs
	Ac := mat.NewDense(n, n, nil)
	for i := 0; i < n-1; i++ {
		Ac.Set(i, i+1, 1)
	}
	for j := range n {
		Ac.Set(n-1, j, -charPoly[j])
	}

	// Cc = [1 0 ... 0]
	Cc := mat.NewDense(1, n, nil)
	Cc.Set(0, 0, 1)

	Oc := buildObsvMatrix(sys.A, sys.C, n, p)
	Ocomp := buildObsvMatrix(Ac, Cc, n, 1)

	var luOc mat.LU
	luOc.Factorize(Oc)
	if luNearSingular(&luOc) {
		return nil, fmt.Errorf("controlsys: system not observable, companion form undefined: %w", ErrSingularTransform)
	}

	// T = Oc \ Ocomp  (i.e. Oc * T = Ocomp => T = Oc^{-1} * Ocomp)
	T := mat.NewDense(n, n, nil)
	if err := luOc.SolveTo(T, false, Ocomp); err != nil {
		return nil, fmt.Errorf("controlsys: transformation solve failed: %w", ErrSingularTransform)
	}

	var luT mat.LU
	luT.Factorize(T)
	if luNearSingular(&luT) {
		return nil, fmt.Errorf("controlsys: transformation matrix singular: %w", ErrSingularTransform)
	}

	// Anew = T⁻¹*A*T: compute AT = A*T, then solve T*Anew = AT
	AT := mat.NewDense(n, n, nil)
	AT.Mul(sys.A, T)
	Anew := mat.NewDense(n, n, nil)
	if err := luT.SolveTo(Anew, false, AT); err != nil {
		return nil, fmt.Errorf("controlsys: solve failed: %w", ErrSingularTransform)
	}

	// Bnew = T⁻¹*B: solve T*Bnew = B
	Bnew := mat.NewDense(n, 1, nil)
	if err := luT.SolveTo(Bnew, false, sys.B); err != nil {
		return nil, fmt.Errorf("controlsys: solve failed: %w", ErrSingularTransform)
	}

	Cnew := mat.NewDense(1, n, nil)
	Cnew.Mul(sys.C, T)

	newSys, err := policy.resultWithOriginalFeedthrough(Anew, Bnew, Cnew)
	if err != nil {
		return nil, err
	}

	return &CanonResult{Sys: newSys, T: T}, nil
}

func characteristicPoly(A *mat.Dense, n int) []float64 {
	if n == 0 {
		return []float64{1}
	}

	var eig mat.Eigen
	ok := eig.Factorize(A, mat.EigenNone)
	if !ok {
		coeffs := make([]float64, n+1)
		coeffs[n] = 1
		return coeffs
	}
	vals := eig.Values(nil)

	// Build polynomial from roots incrementally: prod(s - lambda_i)
	// coeffs in descending power: coeffs[0]*s^n + ... + coeffs[n]
	coeffs := make([]float64, n+1)
	coeffs[0] = 1.0
	cur := 1
	for _, lam := range vals {
		if math.Abs(imag(lam)) < 1e-10*math.Max(1, cmplx.Abs(lam)) {
			r := real(lam)
			for j := cur; j >= 1; j-- {
				coeffs[j] = coeffs[j-1] - r*coeffs[j]
			}
			coeffs[0] = -r * coeffs[0]
			cur++
		} else if imag(lam) > 0 {
			a := -2 * real(lam)
			b := real(lam)*real(lam) + imag(lam)*imag(lam)
			newCoeffs := make([]float64, n+1)
			for j := 0; j <= cur; j++ {
				newCoeffs[j+2] += coeffs[j]
				newCoeffs[j+1] += a * coeffs[j]
				newCoeffs[j] += b * coeffs[j]
			}
			copy(coeffs, newCoeffs)
			cur += 2
		}
	}

	// Result in ascending power order: result[i] = coefficient of s^i
	result := make([]float64, n+1)
	for i := range result {
		result[i] = coeffs[n-i]
	}
	return result
}

func buildObsvMatrix(A, C *mat.Dense, n, p int) *mat.Dense {
	O := mat.NewDense(n*p, n, nil)
	setBlock(O, 0, 0, C)
	CA := mat.NewDense(p, n, nil)
	CA.Mul(C, A)
	prev := CA
	setBlock(O, p, 0, CA)
	for i := 2; i < n; i++ {
		next := mat.NewDense(p, n, nil)
		next.Mul(prev, A)
		setBlock(O, i*p, 0, next)
		prev = next
	}

	if n*p == n {
		return O
	}
	return extractSubmatrix(O, 0, n, 0, n)
}
