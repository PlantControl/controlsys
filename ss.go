package controlsys

import (
	"fmt"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// LFTDelay holds the internal delay representation using a linear
// fractional transformation (LFT) structure.
type LFTDelay struct {
	Tau                   []float64
	B2, C2, D12, D21, D22 *mat.Dense
}

// System represents a linear time-invariant (LTI) state-space model:
//
//	Continuous: dx/dt = Ax + Bu,  y = Cx + Du
//	Discrete:   x[k+1] = Ax[k] + Bu[k],  y[k] = Cx[k] + Du[k]
type System struct {
	A           *mat.Dense
	B           *mat.Dense
	C           *mat.Dense
	D           *mat.Dense
	E           *mat.Dense // descriptor matrix; nil = identity (standard state-space)
	Delay       *mat.Dense // p×m IODelay; nil = no delay
	InputDelay  []float64  // length m; nil = zeros
	OutputDelay []float64  // length p; nil = zeros
	LFT         *LFTDelay  // internal delays; nil = none

	Dt float64

	InputName  []string
	OutputName []string
	StateName  []string
	Notes      string
}

func (sys *System) IsDescriptor() bool { return newDescriptorPolicy(sys).isDescriptor() }

func (sys *System) internalDelayCount() int {
	if sys.LFT == nil {
		return 0
	}
	return len(sys.LFT.Tau)
}

func (sys *System) Dims() (n, m, p int) {
	if sys.A != nil {
		n, _ = sys.A.Dims()
	}
	if n > 0 {
		if sys.B != nil {
			_, m = sys.B.Dims()
		}
		if sys.C != nil {
			p, _ = sys.C.Dims()
		}
	} else if sys.D != nil {
		p, m = sys.D.Dims()
	}
	return
}

func (sys *System) IsContinuous() bool { return newTimeDomain(sys.Dt).isContinuous() }
func (sys *System) IsDiscrete() bool   { return newTimeDomain(sys.Dt).isDiscrete() }

func (sys *System) Validate() error {
	if sys == nil {
		return fmt.Errorf("system is nil: %w", ErrDimensionMismatch)
	}
	A, B, C, D := nonEmptyDense(sys.A), nonEmptyDense(sys.B), nonEmptyDense(sys.C), nonEmptyDense(sys.D)
	n, m, p, err := validateDims(A, B, C, D, sys.Dt)
	if err != nil {
		return err
	}
	if err := newDescriptorPolicy(sys).validate(n); err != nil {
		return err
	}
	if err := validateDelay(sys.Delay, p, m, sys.Dt); err != nil {
		return err
	}
	if err := validateSliceDelay(sys.InputDelay, m, sys.Dt); err != nil {
		return err
	}
	if err := validateSliceDelay(sys.OutputDelay, p, sys.Dt); err != nil {
		return err
	}
	if sys.LFT != nil {
		N := len(sys.LFT.Tau)
		if err := validateSliceDelay(sys.LFT.Tau, N, sys.Dt); err != nil {
			return err
		}
		if slices.Contains(sys.LFT.Tau, 0) {
			return ErrZeroInternalDelay
		}
		if err := validateLFTDims(n, m, p, N, sys.LFT.B2, sys.LFT.C2, sys.LFT.D12, sys.LFT.D21, sys.LFT.D22); err != nil {
			return err
		}
	}
	return nil
}

func nonEmptyDense(m *mat.Dense) *mat.Dense {
	if m == nil {
		return nil
	}
	r, c := m.Dims()
	if r == 0 || c == 0 {
		return nil
	}
	return m
}

// Poles returns the eigenvalues of A, or the generalized eigenvalues of (A, E)
// for descriptor models. As in MATLAB pole, internal delays, continuous or
// discrete, are set to zero (zero-order Padé) so the model has finitely many
// poles; an ill-posed zero-delay loop returns ErrAlgebraicLoop. Discrete delay
// poles at z = 0 appear only after AbsorbDelay. Input/output delays add no
// poles. See https://www.mathworks.com/help/control/ref/dynamicsystem.pole.html.
func (sys *System) Poles() ([]complex128, error) {
	n, _, _ := sys.Dims()
	if n == 0 {
		return nil, nil
	}
	if sys.LFT != nil {
		zd, err := sys.ZeroDelayApprox()
		if err != nil {
			return nil, fmt.Errorf("Poles: %w", err)
		}
		sys = zd
	}
	return newDescriptorPolicy(sys).poles(sys.A, n)
}

func (sys *System) IsStable() (bool, error) {
	poles, err := sys.Poles()
	if err != nil {
		return false, err
	}
	for _, p := range poles {
		if poleOnOrOutsideStabilityBoundary(p, sys.IsContinuous(), poleStabilityTolerance(p)) {
			return false, nil
		}
	}
	return true, nil
}

func validateDims(A, B, C, D *mat.Dense, dt float64) (n, m, p int, err error) {
	if err := newTimeDomain(dt).validateSampleTime(); err != nil {
		return 0, 0, 0, err
	}
	if A != nil {
		r, c := A.Dims()
		if r != c {
			return 0, 0, 0, fmt.Errorf("A must be square (%d×%d): %w", r, c, ErrDimensionMismatch)
		}
		n = r
	}
	if B != nil {
		br, bc := B.Dims()
		if A != nil && br != n {
			return 0, 0, 0, fmt.Errorf("B rows %d != A rows %d: %w", br, n, ErrDimensionMismatch)
		}
		if A == nil {
			n = br
		}
		m = bc
	}
	if C != nil {
		cr, cc := C.Dims()
		p = cr
		if A != nil && cc != n {
			return 0, 0, 0, fmt.Errorf("C cols %d != A cols %d: %w", cc, n, ErrDimensionMismatch)
		}
		if A == nil && B != nil && cc != n {
			return 0, 0, 0, fmt.Errorf("C cols %d != state dim %d: %w", cc, n, ErrDimensionMismatch)
		}
	}
	if D != nil {
		dr, dc := D.Dims()
		if C != nil && dr != p {
			return 0, 0, 0, fmt.Errorf("D rows %d != C rows %d: %w", dr, p, ErrDimensionMismatch)
		}
		if B != nil && dc != m {
			return 0, 0, 0, fmt.Errorf("D cols %d != B cols %d: %w", dc, m, ErrDimensionMismatch)
		}
		if C == nil {
			p = dr
		}
		if B == nil {
			m = dc
		}
	}
	return n, m, p, nil
}

// New validates dimension compatibility and returns a System.
// Matrices are copied to prevent aliasing bugs.
func New(A, B, C, D *mat.Dense, dt float64) (*System, error) {
	n, m, p, err := validateDims(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	return &System{
		A:  denseCopySafe(A, n, n),
		B:  denseCopySafe(B, n, m),
		C:  denseCopySafe(C, p, n),
		D:  denseCopySafe(D, p, m),
		Dt: dt,
	}, nil
}

func newNoCopy(A, B, C, D *mat.Dense, dt float64) (*System, error) {
	n, m, p, err := validateDims(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	if A == nil {
		A = newDense(n, n)
	}
	if B == nil {
		B = newDense(n, m)
	}
	if C == nil {
		C = newDense(p, n)
	}
	if D == nil {
		D = newDense(p, m)
	}
	return &System{A: A, B: B, C: C, D: D, Dt: dt}, nil
}

func NewGain(D *mat.Dense, dt float64) (*System, error) {
	if dt < 0 {
		return nil, ErrInvalidSampleTime
	}
	if D == nil {
		return nil, fmt.Errorf("D matrix required for gain system: %w", ErrDimensionMismatch)
	}
	p, m := D.Dims()
	return &System{
		A:  &mat.Dense{},
		B:  &mat.Dense{},
		C:  &mat.Dense{},
		D:  denseCopySafe(D, p, m),
		Dt: dt,
	}, nil
}

func NewFromSlices(n, m, p int, a, b, c, d []float64, dt float64) (*System, error) {
	var A, B, C, D *mat.Dense
	if n > 0 {
		if len(a) != n*n {
			return nil, fmt.Errorf("a length %d != n²=%d: %w", len(a), n*n, ErrDimensionMismatch)
		}
		A = mat.NewDense(n, n, append([]float64(nil), a...))
	}
	if n > 0 && m > 0 {
		if len(b) != n*m {
			return nil, fmt.Errorf("b length %d != n*m=%d: %w", len(b), n*m, ErrDimensionMismatch)
		}
		B = mat.NewDense(n, m, append([]float64(nil), b...))
	}
	if p > 0 && n > 0 {
		if len(c) != p*n {
			return nil, fmt.Errorf("c length %d != p*n=%d: %w", len(c), p*n, ErrDimensionMismatch)
		}
		C = mat.NewDense(p, n, append([]float64(nil), c...))
	}
	if p > 0 && m > 0 {
		if d != nil {
			if len(d) != p*m {
				return nil, fmt.Errorf("d length %d != p*m=%d: %w", len(d), p*m, ErrDimensionMismatch)
			}
			D = mat.NewDense(p, m, append([]float64(nil), d...))
		}
	}

	if n == 0 {
		if dt < 0 {
			return nil, ErrInvalidSampleTime
		}
		var Dm *mat.Dense
		if p > 0 && m > 0 {
			if d != nil {
				Dm = mat.NewDense(p, m, d)
			} else {
				Dm = mat.NewDense(p, m, nil)
			}
		} else {
			Dm = &mat.Dense{}
		}
		return NewGain(Dm, dt)
	}

	return newNoCopy(A, B, C, D, dt)
}

func (sys *System) Copy() *System {
	cp := &System{
		A:     denseCopy(sys.A),
		B:     denseCopy(sys.B),
		C:     denseCopy(sys.C),
		D:     denseCopy(sys.D),
		E:     copyDescriptorE(sys.E),
		Delay: copyDelayOrNil(sys.Delay),
		Dt:    sys.Dt,
	}
	if sys.InputDelay != nil {
		cp.InputDelay = make([]float64, len(sys.InputDelay))
		copy(cp.InputDelay, sys.InputDelay)
	}
	if sys.OutputDelay != nil {
		cp.OutputDelay = make([]float64, len(sys.OutputDelay))
		copy(cp.OutputDelay, sys.OutputDelay)
	}
	if sys.LFT != nil {
		cp.LFT = &LFTDelay{
			Tau: append([]float64(nil), sys.LFT.Tau...),
			B2:  copyDelayOrNil(sys.LFT.B2),
			C2:  copyDelayOrNil(sys.LFT.C2),
			D12: copyDelayOrNil(sys.LFT.D12),
			D21: copyDelayOrNil(sys.LFT.D21),
			D22: copyDelayOrNil(sys.LFT.D22),
		}
	}
	cp.InputName = copyStringSlice(sys.InputName)
	cp.OutputName = copyStringSlice(sys.OutputName)
	cp.StateName = copyStringSlice(sys.StateName)
	cp.Notes = sys.Notes
	return cp
}
