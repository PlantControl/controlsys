package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// NewDescriptor returns the descriptor model E·dx/dt = A·x + B·u,
// y = C·x + D·u, as MATLAB dss(A, B, C, D, E, Ts). A nil E means E = I.
// Matrices are copied.
func NewDescriptor(A, B, C, D, E *mat.Dense, dt float64) (*System, error) {
	sys, err := newCopy(A, B, C, D, dt)
	if err != nil {
		return nil, fmt.Errorf("NewDescriptor: %w", err)
	}
	n, _, _ := sys.Dims()
	policy := descriptorPolicy{E: E}
	if err := policy.validate(n); err != nil {
		return nil, fmt.Errorf("NewDescriptor: %w", err)
	}
	sys.E = copyDescriptorE(E)
	return sys, nil
}

func newDescriptorOwned(A, B, C, D, E *mat.Dense, dt float64) (*System, error) {
	sys, err := newNoCopy(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	n, _, _ := sys.Dims()
	if err := (descriptorPolicy{E: E}).validate(n); err != nil {
		return nil, err
	}
	sys.E = E
	return sys, nil
}

// DescriptorE returns a copy of the descriptor matrix E, or the n×n identity
// for an explicit model, as MATLAB dssdata. A model with no states returns an
// empty matrix.
func (sys *System) DescriptorE() *mat.Dense {
	if sys.E != nil {
		return mat.DenseCopyOf(sys.E)
	}
	n, _, _ := sys.Dims()
	if n == 0 {
		return &mat.Dense{}
	}
	return eyeDense(n)
}

// ToExplicit returns the equivalent explicit model (E⁻¹A, E⁻¹B, C, D) of a
// descriptor model, or a copy of an explicit one. A numerically singular E
// returns ErrDescriptorSingular.
func (sys *System) ToExplicit() (*System, error) {
	if err := requireSystem("ToExplicit", sys); err != nil {
		return nil, err
	}
	if !sys.IsDescriptor() {
		cp := sys.Copy()
		cp.E = nil
		return cp, nil
	}
	var lu mat.LU
	lu.Factorize(sys.E)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("ToExplicit: %w", ErrDescriptorSingular)
	}

	result := sys.Copy()
	result.E = nil
	if err := lu.SolveTo(result.A, false, sys.A); err != nil {
		return nil, fmt.Errorf("ToExplicit: %w", ErrDescriptorSingular)
	}
	if _, m, _ := sys.Dims(); m > 0 {
		if err := lu.SolveTo(result.B, false, sys.B); err != nil {
			return nil, fmt.Errorf("ToExplicit: %w", ErrDescriptorSingular)
		}
	}
	if result.LFT != nil && result.LFT.B2 != nil {
		if err := lu.SolveTo(result.LFT.B2, false, sys.LFT.B2); err != nil {
			return nil, fmt.Errorf("ToExplicit: %w", ErrDescriptorSingular)
		}
	}
	return result, nil
}

// EliminateStates removes the states elim with Modred using method.
func (sys *System) EliminateStates(elim []int, method StateProjection) (*System, error) {
	return Modred(sys, elim, method)
}

// StateTransform applies the state transformation x̄ = T·x with SS2SS.
func (sys *System) StateTransform(T *mat.Dense) (*System, error) {
	return SS2SS(sys, T)
}

// FixedInputReduction holds the inputs in fixed at constant values and returns
// a model whose inputs are the remaining inputs, in order, followed by one
// offset input. Driving the offset input with 1 reproduces the original
// response; its channel is Σ fixed[j]·H(:,j). Descriptor E, internal delays,
// output delays and names carry over. Nonzero fixed inputs must share the same
// InputDelay and IODelay column, which the offset input inherits; otherwise
// ErrFixedInputDelayMismatch is returned. An empty fixed map returns a copy
// without an offset input.
func (sys *System) FixedInputReduction(fixed map[int]float64, offsetName string) (*System, error) {
	if err := requireSystem("FixedInputReduction", sys); err != nil {
		return nil, err
	}
	if len(fixed) == 0 {
		return sys.Copy(), nil
	}
	n, m, p := sys.Dims()
	fixedSeen := make([]bool, m)
	for idx := range fixed {
		if idx < 0 || idx >= m {
			return nil, fmt.Errorf("FixedInputReduction: input index %d out of range [0,%d): %w", idx, m, ErrInvalidArgument)
		}
		fixedSeen[idx] = true
	}
	keep := make([]int, 0, m-len(fixed))
	ref := -1
	for j := range m {
		if !fixedSeen[j] {
			keep = append(keep, j)
			continue
		}
		if fixed[j] == 0 {
			continue
		}
		if ref < 0 {
			ref = j
		} else if !sameInputDelays(sys, ref, j) {
			return nil, fmt.Errorf("FixedInputReduction: inputs %d and %d: %w", ref, j, ErrFixedInputDelayMismatch)
		}
	}

	mNew := len(keep) + 1
	result, err := newNoCopy(denseCopy(sys.A), selectColumnsWithOffset(sys.B, n, keep, fixed), denseCopy(sys.C), selectColumnsWithOffset(sys.D, p, keep, fixed), sys.Dt)
	if err != nil {
		return nil, fmt.Errorf("FixedInputReduction: %w", err)
	}
	result.E = copyDescriptorE(sys.E)
	if sys.Delay != nil {
		result.Delay = mat.NewDense(p, mNew, nil)
		for i := range p {
			for c, j := range keep {
				result.Delay.Set(i, c, sys.Delay.At(i, j))
			}
			if ref >= 0 {
				result.Delay.Set(i, len(keep), sys.Delay.At(i, ref))
			}
		}
	}
	if sys.InputDelay != nil {
		result.InputDelay = selectDelaySlice(sys.InputDelay, keep)
		offsetDelay := 0.0
		if ref >= 0 {
			offsetDelay = sys.InputDelay[ref]
		}
		result.InputDelay = append(result.InputDelay, offsetDelay)
	}
	result.OutputDelay = copySliceOrNil(sys.OutputDelay)
	if sys.LFT != nil {
		N := len(sys.LFT.Tau)
		result.LFT = &LFTDelay{
			Tau: append([]float64(nil), sys.LFT.Tau...),
			B2:  copyDelayOrNil(sys.LFT.B2),
			C2:  copyDelayOrNil(sys.LFT.C2),
			D12: copyDelayOrNil(sys.LFT.D12),
			D21: selectColumnsWithOffset(sys.LFT.D21, N, keep, fixed),
			D22: copyDelayOrNil(sys.LFT.D22),
		}
	}
	if sys.InputName != nil || offsetName != "" {
		names := selectStringSlice(sys.InputName, keep)
		if names == nil {
			names = make([]string, len(keep), mNew)
		}
		result.InputName = append(names, offsetName)
	}
	result.OutputName = copyStringSlice(sys.OutputName)
	result.StateName = copyStringSlice(sys.StateName)
	return result, nil
}

func sameInputDelays(sys *System, a, b int) bool {
	if sys.InputDelay != nil && sys.InputDelay[a] != sys.InputDelay[b] {
		return false
	}
	if sys.Delay != nil {
		p, _ := sys.Delay.Dims()
		for i := range p {
			if sys.Delay.At(i, a) != sys.Delay.At(i, b) {
				return false
			}
		}
	}
	return true
}

// selectColumnsWithOffset returns [src(:,keep) Σ fixed[j]·src(:,j)].
func selectColumnsWithOffset(src *mat.Dense, rows int, keep []int, fixed map[int]float64) *mat.Dense {
	if src == nil {
		return nil
	}
	cols := len(keep) + 1
	if rows == 0 {
		return &mat.Dense{}
	}
	out := mat.NewDense(rows, cols, nil)
	outRaw := out.RawMatrix()
	srcRaw := src.RawMatrix()
	for c, j := range keep {
		for i := range rows {
			outRaw.Data[i*outRaw.Stride+c] = srcRaw.Data[i*srcRaw.Stride+j]
		}
	}
	_, m := src.Dims()
	for j := range m {
		v, ok := fixed[j]
		if !ok || v == 0 {
			continue
		}
		for i := range rows {
			outRaw.Data[i*outRaw.Stride+len(keep)] += srcRaw.Data[i*srcRaw.Stride+j] * v
		}
	}
	return out
}

// AugmentInternalDelayOutputs appends the internal-delay input signals
// z = C2·x + D21·u + D22·w to the outputs. Appended rows inherit InputDelay
// and carry no OutputDelay or IODelay.
func (sys *System) AugmentInternalDelayOutputs(prefix string) (*System, error) {
	if err := requireSystem("AugmentInternalDelayOutputs", sys); err != nil {
		return nil, err
	}
	if !sys.HasInternalDelay() {
		return sys.Copy(), nil
	}
	n, m, p := sys.Dims()
	N := len(sys.LFT.Tau)
	C := mat.NewDense(p+N, n, nil)
	D := mat.NewDense(p+N, m, nil)
	setBlock(C, 0, 0, sys.C)
	setBlock(D, 0, 0, sys.D)
	setBlock(C, p, 0, sys.LFT.C2)
	setBlock(D, p, 0, sys.LFT.D21)

	D12 := mat.NewDense(p+N, N, nil)
	setBlock(D12, 0, 0, sys.LFT.D12)
	setBlock(D12, p, 0, sys.LFT.D22)

	result := sys.Copy()
	result.C = C
	result.D = D
	result.LFT.D12 = D12
	result.Delay = padZeroRows(sys.Delay, p+N)
	if sys.OutputDelay != nil {
		result.OutputDelay = make([]float64, p+N)
		copy(result.OutputDelay, sys.OutputDelay)
	}
	names := make([]string, p, p+N)
	copy(names, sys.OutputName)
	result.OutputName = append(names, autoLabel(prefix, N)...)
	return result, nil
}

func selectDelaySlice(values []float64, indices []int) []float64 {
	if values == nil {
		return nil
	}
	out := make([]float64, len(indices))
	for i, idx := range indices {
		out[i] = values[idx]
	}
	return out
}
