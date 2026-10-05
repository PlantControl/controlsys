package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// matLog computes the principal matrix logarithm of a real square matrix via
// eigendecomposition: A = V·Λ·V⁻¹ ⇒ log(A) = V·log(Λ)·V⁻¹.
//
// Returns ErrSingularTransform if A has an eigenvalue on the branch cut
// (real, ≤ 0) or if the eigenvector matrix is singular/ill-conditioned
// (defective / near-defective A).
//
// Implementation: skips explicit V⁻¹ by solving V^T·L^T = (V·D)^T in the
// 2n×2n real-augmented representation of the complex system, then transposes.
// One LU factorization + one back-solve with n complex RHS.
func matLogSpectral(A *mat.Dense) (*mat.Dense, error) {
	n, c := A.Dims()
	if n != c {
		return nil, fmt.Errorf("matLog: non-square %dx%d: %w", n, c, ErrDimensionMismatch)
	}
	if n == 0 {
		return newDense(0, 0), nil
	}
	if err := requireFiniteDense("matLog", "A", A); err != nil {
		return nil, err
	}

	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenRight) {
		return nil, fmt.Errorf("matLog: eigendecomposition failed: %w", ErrSchurFailed)
	}
	vals := eig.Values(nil)

	logVals := make([]complex128, n)
	for i, v := range vals {
		if imag(v) == 0 && real(v) <= 0 {
			return nil, fmt.Errorf("matLog: eigenvalue[%d]=%v on non-positive real axis (branch cut): %w",
				i, v, ErrSingularTransform)
		}
		if cmplx.Abs(v) == 0 {
			return nil, fmt.Errorf("matLog: eigenvalue[%d]=0: %w", i, ErrSingularTransform)
		}
		logVals[i] = cmplx.Log(v)
	}

	var vecC mat.CDense
	eig.VectorsTo(&vecC)
	vRaw := vecC.RawCMatrix()

	// Build V^T_aug = [[Re(V)^T, -Im(V)^T], [Im(V)^T, Re(V)^T]]  (2n × 2n).
	// Build Y^T_aug where Y = V·D (D=diag(logVals)); (Y^T)_{i,j} = d_i · V_{j,i}.
	vTaug := mat.NewDense(2*n, 2*n, nil)
	yTaug := mat.NewDense(2*n, n, nil)
	vaRaw := vTaug.RawMatrix()
	yaRaw := yTaug.RawMatrix()
	for i := range n {
		vRow := vRaw.Data[i*vRaw.Stride : i*vRaw.Stride+n]
		for j := range n {
			v := vRow[j]
			a := real(v)
			b := imag(v)
			// V^T_aug = [[Re(V)^T, -Im(V)^T], [Im(V)^T, Re(V)^T]]; V_{i,j} lands at (j,i).
			vaRaw.Data[j*vaRaw.Stride+i] = a
			vaRaw.Data[j*vaRaw.Stride+(i+n)] = -b
			vaRaw.Data[(j+n)*vaRaw.Stride+i] = b
			vaRaw.Data[(j+n)*vaRaw.Stride+(i+n)] = a
		}
	}
	// Y^T_aug: for row i of Y^T (complex), value at col j is d_i · V_{j,i}.
	for i := range n {
		d := logVals[i]
		dr, di := real(d), imag(d)
		reRow := yaRaw.Data[i*yaRaw.Stride : i*yaRaw.Stride+n]
		imRow := yaRaw.Data[(i+n)*yaRaw.Stride : (i+n)*yaRaw.Stride+n]
		for j := range n {
			v := vRaw.Data[j*vRaw.Stride+i]
			a := real(v)
			b := imag(v)
			reRow[j] = dr*a - di*b
			imRow[j] = di*a + dr*b
		}
	}

	var lu mat.LU
	lu.Factorize(vTaug)
	if cond := lu.Cond(); math.IsNaN(cond) || math.IsInf(cond, 0) || cond > 1e6 {
		return nil, fmt.Errorf("matLog: eigenvector matrix singular (defective A): %w", ErrSingularTransform)
	}

	xAug := mat.NewDense(2*n, n, nil)
	if err := lu.SolveTo(xAug, false, yTaug); err != nil {
		return nil, fmt.Errorf("matLog: LU solve failed: %w", ErrSingularTransform)
	}

	// xAug represents L^T in complex form: top half = Re(L^T), bottom half = Im(L^T).
	// Take the real part of L = (Re(L^T))^T and bound the residual imaginary part.
	xRaw := xAug.RawMatrix()
	result := mat.NewDense(n, n, nil)
	resRaw := result.RawMatrix()
	maxIm, maxRe := 0.0, 0.0
	for i := range n {
		reRow := xRaw.Data[i*xRaw.Stride : i*xRaw.Stride+n]
		imRow := xRaw.Data[(i+n)*xRaw.Stride : (i+n)*xRaw.Stride+n]
		for j := range n {
			re := reRow[j]
			im := imRow[j]
			// (L^T)_{i,j} = X_{i,j}  ⇒  L_{j,i} = X_{i,j}; set result[j,i].
			resRaw.Data[j*resRaw.Stride+i] = re
			if im < 0 {
				im = -im
			}
			if re > maxRe {
				maxRe = re
			} else if -re > maxRe {
				maxRe = -re
			}
			if im > maxIm {
				maxIm = im
			}
		}
	}

	tol := 1e-8 * math.Max(1, maxRe)
	if maxIm > tol {
		return nil, fmt.Errorf("matLog: residual imaginary part %.3e exceeds tolerance %.3e (defective or near-defective A): %w",
			maxIm, tol, ErrSingularTransform)
	}
	return result, nil
}
