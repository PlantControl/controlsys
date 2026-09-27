package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

func copyDescriptorE(E *mat.Dense) *mat.Dense {
	if E == nil {
		return nil
	}
	return mat.DenseCopyOf(E)
}

type descriptorPolicy struct {
	E *mat.Dense
}

func newDescriptorPolicy(sys *System) descriptorPolicy {
	if sys == nil {
		return descriptorPolicy{}
	}
	return descriptorPolicy{E: sys.E}
}

func (p descriptorPolicy) isDescriptor() bool {
	return p.E != nil && !isIdentityDescriptor(p.E)
}

func (p descriptorPolicy) validate(n int) error {
	if p.E == nil {
		return nil
	}
	er, ec := p.E.Dims()
	if er != n || ec != n {
		return fmt.Errorf("E %dx%d != %dx%d: %w", er, ec, n, n, ErrDimensionMismatch)
	}
	return nil
}

func (p descriptorPolicy) poles(A *mat.Dense, n int) ([]complex128, error) {
	if p.E != nil && !isIdentityDescriptor(p.E) {
		return generalizedPoles(A, p.E, n)
	}
	var eig mat.Eigen
	ok := eig.Factorize(A, mat.EigenNone)
	if !ok {
		return nil, fmt.Errorf("controlsys: eigenvalue decomposition failed to converge")
	}
	return eig.Values(nil), nil
}

func (p descriptorPolicy) requireStandard(context string) error {
	if p.E == nil || isIdentityDescriptor(p.E) {
		return nil
	}
	return fmt.Errorf("%s: %w", context, ErrDescriptorUnsupported)
}

func (p descriptorPolicy) requireRiccatiStandard(context string) error {
	if p.E == nil || isIdentityDescriptor(p.E) {
		return nil
	}
	return fmt.Errorf("%s: %w", context, ErrDescriptorRiccati)
}

func isIdentityDescriptor(E *mat.Dense) bool {
	if E == nil {
		return true
	}
	r, c := E.Dims()
	if r != c {
		return false
	}
	raw := E.RawMatrix()
	for i := range r {
		for j := range c {
			want := 0.0
			if i == j {
				want = 1
			}
			if raw.Data[i*raw.Stride+j] != want {
				return false
			}
		}
	}
	return true
}

// generalizedPoles computes the finite eigenvalues of the pencil (A, E)
// via the QZ algorithm (DGGES). Infinite eigenvalues (beta=0) are excluded.
func generalizedPoles(A, E *mat.Dense, n int) ([]complex128, error) {
	aData := make([]float64, n*n)
	eData := make([]float64, n*n)
	aRaw := A.RawMatrix()
	eRaw := E.RawMatrix()
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)
	copyStrided(eData, n, eRaw.Data, eRaw.Stride, n, n)

	alphar := make([]float64, n)
	alphai := make([]float64, n)
	beta := make([]float64, n)
	vsl := make([]float64, n*n)
	vsr := make([]float64, n*n)

	var workQuery [1]float64
	impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, aData, n, eData, n, alphar, alphai, beta, vsl, n, vsr, n, workQuery[:], -1, nil)
	lwork := int(workQuery[0])
	work := make([]float64, lwork)
	_, ok := impl.Dgges(lapack.SchurNone, lapack.SchurNone, lapack.SortNone, nil,
		n, aData, n, eData, n, alphar, alphai, beta, vsl, n, vsr, n, work, lwork, nil)
	if !ok {
		return nil, fmt.Errorf("controlsys: generalized eigenvalue decomposition failed")
	}

	poles := make([]complex128, 0, n)
	for i := range n {
		if beta[i] == 0 {
			continue
		}
		poles = append(poles, complex(alphar[i]/beta[i], alphai[i]/beta[i]))
	}
	return poles, nil
}
