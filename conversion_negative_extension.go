package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

func (sys *System) compressZOHRealExtension(re, im, rb, ib *mat.Dense, aliases int) (*System, error) {
	n, m, p := sys.Dims()
	if aliases <= 0 || aliases > n {
		return nil, ErrSingularTransform
	}
	for _, matrix := range []*mat.Dense{re, im, rb, ib} {
		raw := matrix.RawMatrix()
		for i := range raw.Rows {
			for j := range raw.Cols {
				if value := raw.Data[i*raw.Stride+j]; math.IsInf(value, 0) || math.IsNaN(value) {
					return nil, ErrOverflow
				}
			}
		}
	}
	var svd mat.SVD
	if !svd.Factorize(im, mat.SVDThinU) {
		return nil, fmt.Errorf("zoh: negative-subspace factorization failed: %w", ErrSingularTransform)
	}
	values := svd.Values(nil)
	tolerance := 1e-9 * values[0]
	if values[aliases-1] <= tolerance || (aliases < n && values[aliases] > tolerance) {
		return nil, fmt.Errorf("zoh: unresolved negative spectral subspace: %w", ErrSingularTransform)
	}
	var u mat.Dense
	svd.UTo(&u)
	basis := mat.NewDense(n, aliases, nil)
	ur, qr := u.RawMatrix(), basis.RawMatrix()
	for i := range n {
		copy(qr.Data[i*qr.Stride:i*qr.Stride+aliases], ur.Data[i*ur.Stride:i*ur.Stride+aliases])
	}
	var imCoordinates, imReconstructed, reBasis, reCoordinates, reReconstructed mat.Dense
	imCoordinates.Mul(basis.T(), im)
	imReconstructed.Mul(basis, &imCoordinates)
	reBasis.Mul(re, basis)
	reCoordinates.Mul(basis.T(), &reBasis)
	reReconstructed.Mul(basis, &reCoordinates)
	if !zohExtensionResidual(im, &imReconstructed) || !zohExtensionResidual(&reBasis, &reReconstructed) {
		return nil, fmt.Errorf("zoh: negative subspace is not invariant within numerical precision: %w", ErrSingularTransform)
	}
	var ibCoordinates mat.Dense
	if m > 0 {
		var reconstructed mat.Dense
		ibCoordinates.Mul(basis.T(), ib)
		reconstructed.Mul(basis, &ibCoordinates)
		if !zohExtensionResidual(ib, &reconstructed) {
			return nil, fmt.Errorf("zoh: input lies outside the negative spectral subspace: %w", ErrSingularTransform)
		}
	}
	var imBasis mat.Dense
	imBasis.Mul(im, basis)
	total := n + aliases
	a, b, c := mat.NewDense(total, total, nil), newDense(total, m), newDense(p, total)
	ar, br, cr := a.RawMatrix(), b.RawMatrix(), c.RawMatrix()
	rer, imbr, imcr, recr, rbr := re.RawMatrix(), imBasis.RawMatrix(), imCoordinates.RawMatrix(), reCoordinates.RawMatrix(), rb.RawMatrix()
	for i := range n {
		copy(ar.Data[i*ar.Stride:i*ar.Stride+n], rer.Data[i*rer.Stride:i*rer.Stride+n])
		for j := range aliases {
			ar.Data[i*ar.Stride+n+j] = -imbr.Data[i*imbr.Stride+j]
		}
		copy(br.Data[i*br.Stride:i*br.Stride+m], rbr.Data[i*rbr.Stride:i*rbr.Stride+m])
	}
	for i := range aliases {
		copy(ar.Data[(n+i)*ar.Stride:(n+i)*ar.Stride+n], imcr.Data[i*imcr.Stride:i*imcr.Stride+n])
		copy(ar.Data[(n+i)*ar.Stride+n:(n+i)*ar.Stride+total], recr.Data[i*recr.Stride:i*recr.Stride+aliases])
	}
	if m > 0 {
		ibr := ibCoordinates.RawMatrix()
		for i := range aliases {
			copy(br.Data[(n+i)*br.Stride:(n+i)*br.Stride+m], ibr.Data[i*ibr.Stride:i*ibr.Stride+m])
		}
	}
	sr := sys.C.RawMatrix()
	for i := range p {
		copy(cr.Data[i*cr.Stride:i*cr.Stride+n], sr.Data[i*sr.Stride:i*sr.Stride+n])
	}
	out := &System{A: a, B: b, C: c, D: denseCopy(sys.D)}
	propagateNames(out, sys)
	if len(sys.StateName) > 0 {
		out.StateName = make([]string, total)
		copy(out.StateName, sys.StateName)
		for i := range aliases {
			out.StateName[n+i] = fmt.Sprintf("alias_%d", i+1)
		}
	}
	return out, nil
}

func zohExtensionResidual(expected, actual *mat.Dense) bool {
	er, ar := expected.RawMatrix(), actual.RawMatrix()
	norm, difference := 0., 0.
	for i := range er.Rows {
		for j := range er.Cols {
			value := er.Data[i*er.Stride+j]
			norm = math.Hypot(norm, value)
			difference = math.Hypot(difference, value-ar.Data[i*ar.Stride+j])
		}
	}
	return !math.IsNaN(difference) && !math.IsInf(difference, 0) && difference <= 1e-9*math.Max(1, norm)
}
