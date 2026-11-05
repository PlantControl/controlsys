package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// requireSystem rejects a nil or structurally invalid model at the boundary
// of the exported operation op. It does not check finiteness, so internal
// paths can pass derived models whose responses may diverge; exported
// operations on caller models use requireFiniteSystem.
func requireSystem(op string, sys *System) error {
	if sys == nil {
		return fmt.Errorf("%s: system is nil: %w", op, ErrInvalidArgument)
	}
	if err := sys.validate(); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}

// requireSystems is requireSystem for the several models of an
// interconnection, naming the offending one by its 0-based position.
func requireSystems(op string, systems ...*System) error {
	for i, sys := range systems {
		if sys == nil {
			return fmt.Errorf("%s: system %d is nil: %w", op, i, ErrInvalidArgument)
		}
		if err := sys.validate(); err != nil {
			return fmt.Errorf("%s: system %d: %w", op, i, err)
		}
	}
	return nil
}

// requireFiniteSystem is requireSystem plus a finiteness check of every
// state-space and internal-delay matrix, for operations that take a caller
// model: a NaN/Inf entry returns ErrInvalidArgument.
func requireFiniteSystem(op string, sys *System) error {
	if err := requireSystem(op, sys); err != nil {
		return err
	}
	return finiteSystemMatrices(op, sys, ErrInvalidArgument)
}

// requireFiniteSystems is requireFiniteSystem for the several models of an
// interconnection, naming the offending one by its 0-based position.
func requireFiniteSystems(op string, systems ...*System) error {
	if err := requireSystems(op, systems...); err != nil {
		return err
	}
	for i, sys := range systems {
		if err := systemMatricesFinite(sys, ErrInvalidArgument); err != nil {
			return fmt.Errorf("%s: system %d: %w", op, i, err)
		}
	}
	return nil
}

// derivedOp names a model-building operation for derivedOp.model.
type derivedOp string

// model passes through the model that operation op derived from finite
// inputs, unless it has a NaN/Inf entry: that returns ErrOverflow, never a
// non-finite model with a nil error.
func (op derivedOp) model(sys *System, err error) (*System, error) {
	if err != nil {
		return nil, err
	}
	if err := finiteSystemMatrices(string(op), sys, ErrOverflow); err != nil {
		return nil, err
	}
	return sys, nil
}

// finiteSystemMatrices reports the first NaN/Inf entry of the state-space and
// internal-delay matrices of sys, wrapping sentinel.
func finiteSystemMatrices(op string, sys *System, sentinel error) error {
	if err := systemMatricesFinite(sys, sentinel); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}

func systemMatricesFinite(sys *System, sentinel error) error {
	type named struct {
		name string
		m    *mat.Dense
	}
	ms := []named{{"A", sys.A}, {"B", sys.B}, {"C", sys.C}, {"D", sys.D}, {"E", sys.E}}
	if l := sys.LFT; l != nil {
		ms = append(ms, named{"B2", l.B2}, named{"C2", l.C2}, named{"D12", l.D12}, named{"D21", l.D21}, named{"D22", l.D22})
	}
	for _, nm := range ms {
		if nm.m == nil {
			continue
		}
		if err := denseFinite(nm.name, nm.m, sentinel); err != nil {
			return err
		}
	}
	return nil
}

// newDerived is New for data an operation derived from finite inputs: a
// NaN/Inf entry is an overflow of that derivation and returns ErrOverflow,
// not the ErrInvalidArgument of caller-supplied data.
func newDerived(A, B, C, D *mat.Dense, dt float64) (*System, error) {
	return finiteDerived(newCopy(A, B, C, D, dt))
}

// newDerivedGain is newDerived for NewGain.
func newDerivedGain(D *mat.Dense, dt float64) (*System, error) {
	return finiteDerived(newGain(D, dt))
}

// newDerivedFromSlices is newDerived for NewFromSlices.
func newDerivedFromSlices(n, m, p int, a, b, c, d []float64, dt float64) (*System, error) {
	return finiteDerived(newFromSlices(n, m, p, a, b, c, d, dt))
}

// newDerivedDescriptor is newDerived for NewDescriptor.
func newDerivedDescriptor(A, B, C, D, E *mat.Dense, dt float64) (*System, error) {
	return finiteDerived(newDescriptor(A, B, C, D, E, dt))
}

func finiteDerived(sys *System, err error) (*System, error) {
	if err != nil {
		return nil, err
	}
	if err := systemMatricesFinite(sys, ErrOverflow); err != nil {
		return nil, err
	}
	return sys, nil
}

// requireFiniteDense rejects a nil matrix or one with a NaN/Inf entry. Callers
// with an optional matrix check nil themselves first.
func requireFiniteDense(op, name string, m *mat.Dense) error {
	if m == nil {
		return fmt.Errorf("%s: %s is nil: %w", op, name, ErrInvalidArgument)
	}
	return finiteDense(op, name, m, ErrInvalidArgument)
}

func finiteDense(op, name string, m *mat.Dense, sentinel error) error {
	if err := denseFinite(name, m, sentinel); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}

func denseFinite(name string, m *mat.Dense, sentinel error) error {
	raw := m.RawMatrix()
	if raw.Stride == raw.Cols && allFinite(raw.Data[:raw.Rows*raw.Cols]) {
		return nil
	}
	for i := range raw.Rows {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		if allFinite(row) {
			continue
		}
		for j, v := range row {
			if !isFinite(v) {
				return fmt.Errorf("%s[%d,%d] is %g: %w", name, i, j, v, sentinel)
			}
		}
	}
	return nil
}

// requireFinite rejects NaN/Inf values. A single value is reported by name,
// several by name and index.
func requireFinite(op, name string, v ...float64) error {
	for i, x := range v {
		if isFinite(x) {
			continue
		}
		if len(v) == 1 {
			return fmt.Errorf("%s: %s is %g: %w", op, name, x, ErrInvalidArgument)
		}
		return fmt.Errorf("%s: %s[%d] is %g: %w", op, name, i, x, ErrInvalidArgument)
	}
	return nil
}

// allFinite reports whether every entry of x is finite without a branch per
// entry: v*0 is ±0 for finite v and NaN for NaN or ±Inf, so the sums stay 0
// exactly when x is finite.
func allFinite(x []float64) bool {
	var s0, s1, s2, s3, s4, s5, s6, s7 float64
	for len(x) >= 8 {
		y := x[:8:8]
		s0 += y[0] * 0
		s1 += y[1] * 0
		s2 += y[2] * 0
		s3 += y[3] * 0
		s4 += y[4] * 0
		s5 += y[5] * 0
		s6 += y[6] * 0
		s7 += y[7] * 0
		x = x[8:]
	}
	for _, v := range x {
		s0 += v * 0
	}
	return (s0+s1)+(s2+s3)+(s4+s5)+(s6+s7) == 0
}

func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
