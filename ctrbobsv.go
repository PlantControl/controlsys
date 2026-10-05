package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// Ctrb returns the n×(n·m) controllability matrix [B, AB, A²B, …, Aⁿ⁻¹B].
func Ctrb(A, B *mat.Dense) (*mat.Dense, error) {
	n, nc := A.Dims()
	if n != nc {
		return nil, ErrDimensionMismatch
	}
	br, m := B.Dims()
	if br != n {
		return nil, ErrDimensionMismatch
	}
	if n == 0 {
		return &mat.Dense{}, nil
	}

	cols := n * m
	data := make([]float64, n*cols)
	bRaw := B.RawMatrix()
	copyStrided(data, cols, bRaw.Data, bRaw.Stride, n, m)

	aGen := rawToGen(A)

	for k := 1; k < n; k++ {
		prev := blas64.General{Rows: n, Cols: m, Stride: cols, Data: data[(k-1)*m:]}
		cur := blas64.General{Rows: n, Cols: m, Stride: cols, Data: data[k*m:]}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, aGen, prev, 0, cur)
	}

	return mat.NewDense(n, cols, data), nil
}

// Obsv returns the (n·p)×n observability matrix [C; CA; CA²; …; CAⁿ⁻¹].
func Obsv(A, C *mat.Dense) (*mat.Dense, error) {
	n, nc := A.Dims()
	if n != nc {
		return nil, ErrDimensionMismatch
	}
	p, cc := C.Dims()
	if cc != n {
		return nil, ErrDimensionMismatch
	}
	if n == 0 {
		return &mat.Dense{}, nil
	}

	rows := n * p
	data := make([]float64, rows*n)
	cRaw := C.RawMatrix()
	copyStrided(data, n, cRaw.Data, cRaw.Stride, p, n)

	aGen := rawToGen(A)

	for k := 1; k < n; k++ {
		prev := blas64.General{Rows: p, Cols: n, Stride: n, Data: data[(k-1)*p*n:]}
		cur := blas64.General{Rows: p, Cols: n, Stride: n, Data: data[k*p*n:]}
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, prev, aGen, 0, cur)
	}

	return mat.NewDense(rows, n, data), nil
}

// rawToGen returns a blas64.General for m, copying only if stride != cols.
func rawToGen(m *mat.Dense) blas64.General {
	raw := m.RawMatrix()
	if raw.Stride == raw.Cols {
		return blas64.General{Rows: raw.Rows, Cols: raw.Cols, Stride: raw.Stride, Data: raw.Data}
	}
	n := raw.Rows
	data := make([]float64, n*raw.Cols)
	copyStrided(data, raw.Cols, raw.Data, raw.Stride, n, raw.Cols)
	return blas64.General{Rows: n, Cols: raw.Cols, Stride: raw.Cols, Data: data}
}

// CtrbF computes the controllability staircase form.
// C may be nil if not needed in the transformed system.
func CtrbF(A, B, C *mat.Dense) (*StaircaseResult, error) {
	n, nc := A.Dims()
	if n != nc {
		return nil, ErrDimensionMismatch
	}
	br, _ := B.Dims()
	if br != n {
		return nil, ErrDimensionMismatch
	}
	if C != nil {
		_, cc := C.Dims()
		if cc != n {
			return nil, ErrDimensionMismatch
		}
	}
	res, err := controllabilityStaircase(A, B, C, 0, true)
	if err != nil {
		return nil, fmt.Errorf("CtrbF: %w", err)
	}
	return res, nil
}

// ObsvF computes the observability staircase form via duality.
// B may be nil if not needed in the transformed system.
func ObsvF(A, B, C *mat.Dense) (*StaircaseResult, error) {
	n, nc := A.Dims()
	if n != nc {
		return nil, ErrDimensionMismatch
	}
	_, cc := C.Dims()
	if cc != n {
		return nil, ErrDimensionMismatch
	}
	if B != nil {
		br, _ := B.Dims()
		if br != n {
			return nil, ErrDimensionMismatch
		}
	}

	acT := mat.DenseCopyOf(A.T())
	ccT := mat.DenseCopyOf(C.T())
	var bcT *mat.Dense
	if B != nil {
		bcT = mat.DenseCopyOf(B.T())
	}

	dual, err := controllabilityStaircase(acT, ccT, bcT, 0, true)
	if err != nil {
		return nil, fmt.Errorf("ObsvF: %w", err)
	}

	nobs := dual.NCont
	res := &StaircaseResult{
		T:          dual.T,
		NCont:      nobs,
		BlockSizes: dual.BlockSizes,
	}

	res.A = mat.DenseCopyOf(dual.A.T())

	if dual.C != nil {
		_, mc := dual.C.Dims()
		if mc > 0 {
			res.B = mat.DenseCopyOf(dual.C.T())
		} else {
			res.B = &mat.Dense{}
		}
	} else {
		res.B = &mat.Dense{}
	}

	dr, _ := dual.B.Dims()
	if dr > 0 {
		res.C = mat.DenseCopyOf(dual.B.T())
	} else {
		res.C = &mat.Dense{}
	}

	return res, nil
}

func IsStabilizable(A, B *mat.Dense, continuous bool) (bool, error) {
	n, nc := A.Dims()
	if n != nc {
		return false, ErrDimensionMismatch
	}
	br, _ := B.Dims()
	if br != n {
		return false, ErrDimensionMismatch
	}
	if n == 0 {
		return true, nil
	}

	res, err := CtrbF(A, B, nil)
	if err != nil {
		return false, err
	}
	if res.NCont == n {
		return true, nil
	}

	nc2 := n - res.NCont
	auc := mat.NewDense(nc2, nc2, nil)
	aRaw := res.A.RawMatrix()
	for i := range nc2 {
		for j := range nc2 {
			auc.Set(i, j, aRaw.Data[(res.NCont+i)*aRaw.Stride+(res.NCont+j)])
		}
	}

	var eig mat.Eigen
	if !eig.Factorize(auc, mat.EigenNone) {
		return false, ErrSchurFailed
	}
	vals := eig.Values(nil)

	for _, v := range vals {
		if poleOnOrOutsideStabilityBoundary(v, continuous, poleStabilityTolerance(v)) {
			return false, nil
		}
	}
	return true, nil
}

func IsDetectable(A, C *mat.Dense, continuous bool) (bool, error) {
	n, nc := A.Dims()
	if n != nc {
		return false, ErrDimensionMismatch
	}
	_, cc := C.Dims()
	if cc != n {
		return false, ErrDimensionMismatch
	}
	return IsStabilizable(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C.T()), continuous)
}
