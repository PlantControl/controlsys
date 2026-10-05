package controlsys

import (
	"fmt"
	"math"
	"plantcontrol.org/v1/gonum/mat"
)

// matLog computes the real principal logarithm by inverse scaling and squaring.
// Eigenvalues on the non-positive real axis are outside its real branch.
func matLog(a *mat.Dense) (*mat.Dense, error) {
	n, c := a.Dims()
	if n != c {
		return nil, fmt.Errorf("matLog: non-square %dx%d: %w", n, c, ErrDimensionMismatch)
	}
	if n == 0 {
		return newDense(0, 0), nil
	}
	if logarithm, err := matLogSpectral(a); err == nil {
		return logarithm, nil
	}
	if !matrixRightHalfPlane(a) {
		var eig mat.Eigen
		if !eig.Factorize(a, mat.EigenNone) {
			return nil, fmt.Errorf("matLog: eigendecomposition failed: %w", ErrSchurFailed)
		}
		for _, v := range eig.Values(nil) {
			if imag(v) == 0 && real(v) <= 0 {
				return nil, fmt.Errorf("matLog: eigenvalue %v on non-positive real axis: %w", v, ErrSingularTransform)
			}
		}
	}

	return matrixLogISS(a)
}

func matrixLogISS(a *mat.Dense) (*mat.Dense, error) {
	n, _ := a.Dims()
	eye := mat.NewDense(n, n, nil)
	for i := range n {
		eye.Set(i, i, 1)
	}
	x := denseCopy(a)
	var diff mat.Dense
	roots := 0
	for {
		diff.Sub(x, eye)
		if mat.Norm(&diff, 1) <= 0.5 {
			break
		}
		if roots == 32 {
			return nil, fmt.Errorf("matLog: square-root scaling did not converge: %w", ErrSingularTransform)
		}
		var err error
		x, err = matrixPrincipalSqrt(x, eye)
		if err != nil {
			return nil, err
		}
		roots++
	}
	var plus, minus mat.Dense
	plus.Add(x, eye)
	minus.Sub(x, eye)
	var lu mat.LU
	lu.Factorize(&plus)
	var z mat.Dense
	if err := lu.SolveTo(&z, false, &minus); err != nil {
		return nil, fmt.Errorf("matLog: logarithm solve failed: %w", ErrSingularTransform)
	}
	var z2 mat.Dense
	z2.Mul(&z, &z)
	term := denseCopy(&z)
	sum := denseCopy(&z)
	next, contribution := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for k := 1; k <= 100; k++ {
		next.Mul(term, &z2)
		term, next = next, term
		contribution.Scale(1/float64(2*k+1), term)
		sum.Add(sum, contribution)
		if mat.Norm(contribution, 1) <= 2e-16*math.Max(1, mat.Norm(sum, 1)) {
			sum.Scale(math.Ldexp(2, roots), sum)
			if norm := mat.Norm(sum, 1); math.IsInf(norm, 0) || math.IsNaN(norm) {
				return nil, fmt.Errorf("matLog: result overflow: %w", ErrOverflow)
			}
			return sum, nil
		}
	}

	return nil, fmt.Errorf("matLog: logarithm series did not converge: %w", ErrSingularTransform)
}

func matrixPrincipalSqrt(a, eye *mat.Dense) (*mat.Dense, error) {
	y, z := denseCopy(a), denseCopy(eye)
	for range 100 {
		var ly, lz mat.LU
		ly.Factorize(y)
		lz.Factorize(z)
		var iy, iz mat.Dense
		if err := ly.SolveTo(&iy, false, eye); err != nil {
			return nil, fmt.Errorf("matLog: square-root inverse failed: %w", ErrSingularTransform)
		}
		if err := lz.SolveTo(&iz, false, eye); err != nil {
			return nil, fmt.Errorf("matLog: square-root inverse failed: %w", ErrSingularTransform)
		}
		var yn, zn, change mat.Dense
		yn.Add(y, &iz)
		yn.Scale(0.5, &yn)
		zn.Add(z, &iy)
		zn.Scale(0.5, &zn)
		change.Sub(&yn, y)
		y, z = denseCopy(&yn), denseCopy(&zn)
		if mat.Norm(&change, 1) <= 4e-15*math.Max(1, mat.Norm(y, 1)) {
			return y, nil
		}
	}
	return nil, fmt.Errorf("matLog: principal square root did not converge: %w", ErrSingularTransform)
}

func matrixRightHalfPlane(a *mat.Dense) bool {
	n, _ := a.Dims()
	raw := a.RawMatrix()
	for i := range n {
		row := raw.Data[i*raw.Stride : i*raw.Stride+n]
		radius := 0.0
		for j, v := range row {
			if j != i {
				radius += math.Abs(v)
			}
		}
		if !(row[i] > radius) {
			return false
		}
	}
	return true
}
