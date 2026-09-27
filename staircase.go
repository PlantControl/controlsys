package controlsys

import (
	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

type StaircaseResult struct {
	A          *mat.Dense
	B          *mat.Dense
	C          *mat.Dense
	NCont      int
	BlockSizes []int
}

func ControllabilityStaircase(A, B, C *mat.Dense, tol float64) *StaircaseResult {
	n, _ := A.Dims()
	_, m := B.Dims()

	if n == 0 || m == 0 {
		return &StaircaseResult{
			A:          denseCopy(A),
			B:          denseCopy(B),
			C:          denseCopy(C),
			NCont:      0,
			BlockSizes: nil,
		}
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

	absFloor := tol * denseNorm(aWork)

	var svd mat.SVD
	var uFull mat.Dense

	for {
		bRows, bCols := block.Dims()
		if bRows == 0 || bCols == 0 {
			break
		}

		fnorm := denseNorm(block)
		if fnorm <= absFloor {
			break
		}
		threshold := tol * fnorm

		ok := svd.Factorize(block, mat.SVDFull)
		if !ok {
			break
		}
		vals := svd.Values(nil)

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

		ncont += rank

		remaining := n - ncont
		if remaining <= 0 {
			break
		}

		block = mat.NewDense(remaining, rank, blockBuf[:remaining*rank])
		blkRaw := block.RawMatrix()
		copyBlock(blkRaw.Data, blkRaw.Stride, 0, 0, aRaw.Data, aRaw.Stride, ncont, ncont-rank, remaining, rank)
	}

	return &StaircaseResult{
		A:          aWork,
		B:          bWork,
		C:          cWork,
		NCont:      ncont,
		BlockSizes: blockSizes,
	}
}
