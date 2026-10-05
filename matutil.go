package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

func denseCopyTo(dst, src *mat.Dense) {
	if src == nil {
		dRaw := dst.RawMatrix()
		for i := 0; i < dRaw.Rows; i++ {
			row := dRaw.Data[i*dRaw.Stride : i*dRaw.Stride+dRaw.Cols]
			for j := range row {
				row[j] = 0
			}
		}
		return
	}
	dst.Copy(src)
}

func denseCopy(m *mat.Dense) *mat.Dense {
	if m == nil {
		return &mat.Dense{}
	}
	r, c := m.Dims()
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	return mat.DenseCopyOf(m)
}

func copyFloatSlice(s []float64) []float64 {
	if s == nil {
		return nil
	}
	out := make([]float64, len(s))
	copy(out, s)
	return out
}

func denseCopySafe(m *mat.Dense, r, c int) *mat.Dense {
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	if m == nil {
		return mat.NewDense(r, c, nil)
	}
	mr, mc := m.Dims()
	if mr == 0 || mc == 0 {
		return mat.NewDense(r, c, nil)
	}
	return mat.DenseCopyOf(m)
}

func newDense(r, c int) *mat.Dense {
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	return mat.NewDense(r, c, nil)
}

func denseFromData(r, c int, data []float64) *mat.Dense {
	if r == 0 || c == 0 {
		return &mat.Dense{}
	}
	return mat.NewDense(r, c, data)
}

// mulDims returns the r×c product a·b. Empty operands stand for zero-width
// blocks (gonum cannot hold n×0 matrices), so the product is then zero.
func mulDims(r, c int, a, b mat.Matrix) *mat.Dense {
	out := newDense(r, c)
	if out.IsEmpty() || isEmptyMatrix(a) || isEmptyMatrix(b) {
		return out
	}
	out.Mul(a, b)
	return out
}

// addMulDims returns the r×c matrix x + a·b, treating empty operands as
// zero blocks.
func addMulDims(r, c int, x, a, b *mat.Dense) *mat.Dense {
	out := mulDims(r, c, a, b)
	addBlock(out, 0, 0, x)
	return out
}

func eyeOrEmptyDense(n int) *mat.Dense {
	if n == 0 {
		return &mat.Dense{}
	}
	return eyeDense(n)
}

// rawOrEmpty returns m's raw storage, or an empty General for nil m (Copy
// stores empty delay blocks as nil).
func rawOrEmpty(m *mat.Dense) blas64.General {
	if m == nil {
		return blas64.General{}
	}
	return m.RawMatrix()
}

func isEmptyMatrix(a mat.Matrix) bool {
	r, c := a.Dims()
	return r == 0 || c == 0
}

// addDims returns the r×c sum a+b, or an empty matrix when r or c is zero.
func addDims(r, c int, a, b *mat.Dense) *mat.Dense {
	out := newDense(r, c)
	if out.IsEmpty() {
		return out
	}
	out.Add(a, b)
	return out
}

func denseNorm(m *mat.Dense) float64 {
	raw := m.RawMatrix()
	sum := 0.0
	for i := 0; i < raw.Rows; i++ {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for _, v := range row {
			sum += v * v
		}
	}
	return math.Sqrt(sum)
}

func eps() float64 {
	return math.Nextafter(1.0, 2.0) - 1.0
}

func luNearSingular(lu *mat.LU) bool {
	return nearSingularCondition(lu.Cond())
}

func nearSingularCondition(cond float64) bool {
	return math.IsNaN(cond) || math.IsInf(cond, 1) || cond*eps() >= 1
}

func isSymmetric(m *mat.Dense, tol float64) bool {
	r, c := m.Dims()
	if r != c {
		return false
	}
	raw := m.RawMatrix()
	for i := range r {
		for j := i + 1; j < c; j++ {
			if math.Abs(raw.Data[i*raw.Stride+j]-raw.Data[j*raw.Stride+i]) > tol {
				return false
			}
		}
	}
	return true
}

// isPSD checks if a symmetric matrix is positive semi-definite
// by attempting Cholesky on Q + tol*I. A small shift avoids
// rejecting borderline-zero eigenvalues from roundoff.
func isPSD(m *mat.Dense) bool {
	n, _ := m.Dims()
	if n == 0 {
		return true
	}
	raw := m.RawMatrix()
	tmp := make([]float64, n*n)
	copyStrided(tmp, n, raw.Data, raw.Stride, n, n)
	symmetrize(tmp, n, n)
	tol := eps() * denseNorm(m) * float64(n)
	for i := range n {
		tmp[i*n+i] += tol
	}
	return impl.Dpotrf(blas.Lower, n, tmp, n)
}

func symmetrize(data []float64, n, stride int) {
	for i := range n {
		for j := i + 1; j < n; j++ {
			avg := 0.5 * (data[i*stride+j] + data[j*stride+i])
			data[i*stride+j] = avg
			data[j*stride+i] = avg
		}
	}
}

func copyStrided(dst []float64, dstStride int, src []float64, srcStride int, rows, cols int) {
	for i := range rows {
		copy(dst[i*dstStride:i*dstStride+cols], src[i*srcStride:i*srcStride+cols])
	}
}

func copyBlock(dst []float64, dstStride, dstR0, dstC0 int, src []float64, srcStride, srcR0, srcC0 int, rows, cols int) {
	if rows == 0 || cols == 0 {
		return
	}
	copyStrided(dst[dstR0*dstStride+dstC0:], dstStride, src[srcR0*srcStride+srcC0:], srcStride, rows, cols)
}

func transposeDenseInto(dst []float64, src *mat.Dense) *mat.Dense {
	r, c := src.Dims()
	raw := src.RawMatrix()
	for i := range r {
		for j := range c {
			dst[j*r+i] = raw.Data[i*raw.Stride+j]
		}
	}
	return mat.NewDense(c, r, dst)
}

func invertSmall(m *mat.Dense, n int) (*mat.Dense, error) {
	var lu mat.LU
	lu.Factorize(m)
	inv := mat.NewDense(n, n, nil)
	eye := mat.NewDense(n, n, nil)
	for i := range n {
		eye.Set(i, i, 1)
	}
	if err := lu.SolveTo(inv, false, eye); err != nil {
		return nil, fmt.Errorf("%v: %w", err, ErrSingularTransform)
	}
	return inv, nil
}

// extractBlock copies the rows×cols block of m at (r0, c0). A block with a
// zero dimension is returned empty (gonum cannot hold n×0), so callers combine
// it through mulDims, addMulDims, setBlock or addBlock, never Dense.Add/Mul.
func extractBlock(m *mat.Dense, r0, c0, rows, cols int) *mat.Dense {
	if rows == 0 || cols == 0 {
		return &mat.Dense{}
	}
	raw := m.RawMatrix()
	data := make([]float64, rows*cols)
	copyBlock(data, cols, 0, 0, raw.Data, raw.Stride, r0, c0, rows, cols)
	return mat.NewDense(rows, cols, data)
}
