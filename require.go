package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// requireSystem rejects a nil or structurally invalid model at the boundary
// of the exported operation op. It does not check finiteness: plain arithmetic
// on a model propagates NaN/Inf as MATLAB does; use requireFiniteSystem before
// eigen/LAPACK work.
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
// state-space and internal-delay matrix, for operations that hand the model
// to eigen/LAPACK routines.
func requireFiniteSystem(op string, sys *System) error {
	if err := requireSystem(op, sys); err != nil {
		return err
	}
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
		if err := requireFiniteDense(op, nm.name, nm.m); err != nil {
			return err
		}
	}
	return nil
}

// requireFiniteDense rejects a nil matrix or one with a NaN/Inf entry. Callers
// with an optional matrix check nil themselves first.
func requireFiniteDense(op, name string, m *mat.Dense) error {
	if m == nil {
		return fmt.Errorf("%s: %s is nil: %w", op, name, ErrInvalidArgument)
	}
	raw := m.RawMatrix()
	for i := range raw.Rows {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for j, v := range row {
			if !isFinite(v) {
				return fmt.Errorf("%s: %s[%d,%d] is %g: %w", op, name, i, j, v, ErrInvalidArgument)
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

func isFinite(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
