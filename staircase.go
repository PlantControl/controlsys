package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// staircaseResult is the orthogonal staircase form behind CtrbF and ObsvF:
// A = T·A₀·Tᵀ, B = T·B₀, C = C₀·Tᵀ with the NCont controllable states first,
// in blocks of sizes BlockSizes.
type staircaseResult struct {
	A *mat.Dense
	B *mat.Dense
	C *mat.Dense
	// T is the orthogonal similarity transformation.
	T          *mat.Dense
	NCont      int
	BlockSizes []int
}

// controllabilityStaircase reduces (A, B, C) to controllability staircase
// form by orthogonal transformations. C may be nil when the transformed C is
// not needed. A singular value of a step counts as rank above
// max(tol·σ₁(step), tol·‖A‖_F), as SLICOT MB03OY with SVLMAX = ‖A‖: the
// absolute term keeps rounding noise in a block small against ‖A‖ from
// counting as rank. tol = 0 selects n²·eps. wantT accumulates the
// transformation into the result's T.
func controllabilityStaircase(A, B, C *mat.Dense, tol float64, wantT bool) (*staircaseResult, error) {
	return controllabilityStaircaseFloor(A, B, C, tol, -1, wantT)
}

// staircaseFloor is the absolute rank floor tol·‖A‖_F, tol = 0 selecting
// n²·eps.
func staircaseFloor(A *mat.Dense, tol float64) float64 {
	n, _ := A.Dims()
	if tol == 0 {
		tol = float64(n*n) * eps()
	}
	return tol * frobNorm(A)
}

// frobNorm is the Frobenius norm, finite when the plain sum of squares
// overflows (‖A‖ ≈ 1e155).
func frobNorm(m *mat.Dense) float64 {
	f := denseNorm(m)
	if !math.IsInf(f, 1) {
		return f
	}
	raw := m.RawMatrix()
	mx := 0.0
	for i := range raw.Rows {
		for _, v := range raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols] {
			mx = math.Max(mx, math.Abs(v))
		}
	}
	if math.IsInf(mx, 0) || math.IsNaN(mx) {
		return f
	}
	sum := 0.0
	for i := range raw.Rows {
		for _, v := range raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols] {
			sum += (v / mx) * (v / mx)
		}
	}
	return mx * math.Sqrt(sum)
}

// ctrbfFloor is staircaseFloor raised to at least MATLAB ctrbf/obsvf's
// default absolute tolerance 10·n·‖A‖₁·eps when tol = 0.
func ctrbfFloor(A *mat.Dense, tol float64) float64 {
	f := staircaseFloor(A, tol)
	if tol != 0 {
		return f
	}
	raw := A.RawMatrix()
	norm1 := 0.0
	for j := range raw.Cols {
		s := 0.0
		for i := range raw.Rows {
			s += math.Abs(raw.Data[i*raw.Stride+j])
		}
		norm1 = math.Max(norm1, s)
	}
	return math.Max(f, 10*float64(raw.Rows)*eps()*norm1)
}

// controllabilityStaircaseFloor is controllabilityStaircase with the
// absolute floor given; floor < 0 selects staircaseFloor(A, tol). A reduced
// subsystem passes the floor of the model it came from, whose rotations
// left the rounding noise.
func controllabilityStaircaseFloor(A, B, C *mat.Dense, tol, floor float64, wantT bool) (*staircaseResult, error) {
	if A == nil || B == nil {
		return nil, fmt.Errorf("A or B is nil: %w", ErrInvalidArgument)
	}
	n, nc := A.Dims()
	br, m := B.Dims()
	if n != nc {
		return nil, fmt.Errorf("A is %d×%d, want square: %w", n, nc, ErrDimensionMismatch)
	}
	if !B.IsEmpty() && br != n {
		return nil, fmt.Errorf("B has %d rows, want %d: %w", br, n, ErrDimensionMismatch)
	}
	if C != nil && !C.IsEmpty() {
		if _, cc := C.Dims(); cc != n {
			return nil, fmt.Errorf("C has %d columns, want %d: %w", cc, n, ErrDimensionMismatch)
		}
	}
	if tol < 0 || math.IsNaN(tol) || math.IsInf(tol, 0) {
		return nil, fmt.Errorf("tol is %g: %w", tol, ErrInvalidArgument)
	}

	var tWork *mat.Dense
	if wantT && n > 0 {
		tWork = eyeDense(n)
	}

	if n == 0 || m == 0 {
		return &staircaseResult{
			A:          denseCopy(A),
			B:          denseCopy(B),
			C:          denseCopy(C),
			T:          tWork,
			NCont:      0,
			BlockSizes: nil,
		}, nil
	}

	absFloor := floor
	if absFloor < 0 {
		absFloor = staircaseFloor(A, tol)
	}
	if tol == 0 {
		tol = float64(n*n) * eps()
	}

	aWork := mat.DenseCopyOf(A)
	bWork := mat.DenseCopyOf(B)
	var cWork *mat.Dense
	if C != nil {
		cWork = mat.DenseCopyOf(C)
	}

	var p int
	if cWork != nil {
		p, _ = cWork.Dims()
	}

	ncont := 0
	var blockSizes []int

	bufSize := n * n
	if nm := n * m; nm > bufSize {
		bufSize = nm
	}
	if pn := p * n; pn > bufSize {
		bufSize = pn
	}
	tempBuf := make([]float64, bufSize)
	blockBuf := make([]float64, bufSize)

	bRawInit := bWork.RawMatrix()
	copyBlock(blockBuf, m, 0, 0, bRawInit.Data, bRawInit.Stride, 0, 0, n, m)
	block := mat.NewDense(n, m, blockBuf[:n*m])

	var svd mat.SVD
	var uFull mat.Dense

	for {
		bRows, bCols := block.Dims()
		if bRows == 0 || bCols == 0 {
			break
		}

		if frobNorm(block) <= absFloor {
			break
		}
		if !svd.Factorize(block, mat.SVDFull) {
			return nil, fmt.Errorf("SVD of staircase block %d did not converge: %w", len(blockSizes)+1, ErrSchurFailed)
		}
		vals := svd.Values(nil)
		threshold := math.Max(tol*vals[0], absFloor)

		rank := 0
		for _, v := range vals {
			if v > threshold {
				rank++
			}
		}
		if rank == 0 {
			break
		}

		blockSizes = append(blockSizes, rank)

		uFull.Reset()
		svd.UTo(&uFull)
		uRaw := uFull.RawMatrix()
		aRaw := aWork.RawMatrix()

		uGen := blas64.General{Rows: bRows, Cols: bRows, Stride: uRaw.Stride, Data: uRaw.Data}

		// A = Q^T * A * Q in-place (Q = I with U at [ncont:ncont+bRows])
		// Step 1: A[:, ncont:ncont+bRows] = A[:, ncont:ncont+bRows] * U
		aSub := blas64.General{Rows: n, Cols: bRows, Stride: aRaw.Stride, Data: aRaw.Data[ncont:]}
		tGen := blas64.General{Rows: n, Cols: bRows, Stride: bRows, Data: tempBuf[:n*bRows]}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, aSub, uGen, 0, tGen)
		copyStrided(aRaw.Data[ncont:], aRaw.Stride, tempBuf, bRows, n, bRows)

		// Step 2: A[ncont:ncont+bRows, :] = U^T * A[ncont:ncont+bRows, :]
		aSubRows := blas64.General{Rows: bRows, Cols: n, Stride: aRaw.Stride, Data: aRaw.Data[ncont*aRaw.Stride:]}
		tGenRows := blas64.General{Rows: bRows, Cols: n, Stride: n, Data: tempBuf[:bRows*n]}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, uGen, aSubRows, 0, tGenRows)
		copyStrided(aRaw.Data[ncont*aRaw.Stride:], aRaw.Stride, tempBuf, n, bRows, n)

		// B = Q^T * B; only rows [ncont:ncont+bRows] change
		bRaw := bWork.RawMatrix()
		bSubRows := blas64.General{Rows: bRows, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data[ncont*bRaw.Stride:]}
		tGenB := blas64.General{Rows: bRows, Cols: m, Stride: m, Data: tempBuf[:bRows*m]}
		blas64.Gemm(blas.Trans, blas.NoTrans, 1, uGen, bSubRows, 0, tGenB)
		copyStrided(bRaw.Data[ncont*bRaw.Stride:], bRaw.Stride, tempBuf, m, bRows, m)

		// C = C * Q; only cols [ncont:ncont+bRows] change
		if cWork != nil {
			cRaw := cWork.RawMatrix()
			cSub := blas64.General{Rows: p, Cols: bRows, Stride: cRaw.Stride, Data: cRaw.Data[ncont:]}
			tGenC := blas64.General{Rows: p, Cols: bRows, Stride: bRows, Data: tempBuf[:p*bRows]}
			blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, cSub, uGen, 0, tGenC)
			copyStrided(cRaw.Data[ncont:], cRaw.Stride, tempBuf, bRows, p, bRows)
		}

		if tWork != nil {
			tRaw := tWork.RawMatrix()
			tSubRows := blas64.General{Rows: bRows, Cols: n, Stride: tRaw.Stride, Data: tRaw.Data[ncont*tRaw.Stride:]}
			tGenT := blas64.General{Rows: bRows, Cols: n, Stride: n, Data: tempBuf[:bRows*n]}
			blas64.Gemm(blas.Trans, blas.NoTrans, 1, uGen, tSubRows, 0, tGenT)
			copyStrided(tRaw.Data[ncont*tRaw.Stride:], tRaw.Stride, tempBuf, n, bRows, n)
		}

		ncont += rank

		remaining := n - ncont
		if remaining <= 0 {
			break
		}

		block = mat.NewDense(remaining, rank, blockBuf[:remaining*rank])
		blkRaw := block.RawMatrix()
		copyBlock(blkRaw.Data, blkRaw.Stride, 0, 0, aRaw.Data, aRaw.Stride, ncont, ncont-rank, remaining, rank)
	}

	return &staircaseResult{
		A:          aWork,
		B:          bWork,
		C:          cWork,
		T:          tWork,
		NCont:      ncont,
		BlockSizes: blockSizes,
	}, nil
}
