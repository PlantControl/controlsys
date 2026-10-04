package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

func NewDescriptor(A, B, C, D, E *mat.Dense, dt float64) (*System, error) {
	sys, err := New(A, B, C, D, dt)
	if err != nil {
		return nil, err
	}
	n, _, _ := sys.Dims()
	policy := descriptorPolicy{E: E}
	if err := policy.validate(n); err != nil {
		return nil, err
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

func (sys *System) DescriptorE() *mat.Dense {
	if sys == nil {
		return nil
	}
	return copyDescriptorE(sys.E)
}

func (sys *System) ToExplicit() (*System, error) {
	if sys == nil {
		return nil, fmt.Errorf("ToExplicit: nil system: %w", ErrDimensionMismatch)
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

func (sys *System) EliminateStates(elim []int, method BalredMethod) (*System, error) {
	return Modred(sys, elim, method)
}

func (sys *System) StateTransform(T *mat.Dense) (*System, error) {
	return SS2SS(sys, T)
}

func (sys *System) FixedInputReduction(fixed map[int]float64, offsetName string) (*System, error) {
	if sys == nil {
		return nil, fmt.Errorf("FixedInputReduction: nil system: %w", ErrDimensionMismatch)
	}
	if len(fixed) == 0 {
		return sys.Copy(), nil
	}
	if err := newDescriptorPolicy(sys).requireStandard("FixedInputReduction"); err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()
	keep := make([]int, 0, m-len(fixed))
	fixedSeen := make([]bool, m)
	for idx := range fixed {
		if idx < 0 || idx >= m {
			return nil, fmt.Errorf("FixedInputReduction: input index %d out of range: %w", idx, ErrDimensionMismatch)
		}
		fixedSeen[idx] = true
	}
	for j := range m {
		if !fixedSeen[j] {
			keep = append(keep, j)
		}
	}

	mNew := len(keep) + 1
	B := mat.NewDense(n, mNew, nil)
	D := mat.NewDense(p, mNew, nil)
	bRaw := B.RawMatrix()
	dRaw := D.RawMatrix()
	srcBRaw := sys.B.RawMatrix()
	srcDRaw := sys.D.RawMatrix()
	for outCol, inCol := range keep {
		for i := range n {
			bRaw.Data[i*bRaw.Stride+outCol] = srcBRaw.Data[i*srcBRaw.Stride+inCol]
		}
		for i := range p {
			dRaw.Data[i*dRaw.Stride+outCol] = srcDRaw.Data[i*srcDRaw.Stride+inCol]
		}
	}
	offsetCol := len(keep)
	for inCol, value := range fixed {
		for i := range n {
			bRaw.Data[i*bRaw.Stride+offsetCol] += srcBRaw.Data[i*srcBRaw.Stride+inCol] * value
		}
		for i := range p {
			dRaw.Data[i*dRaw.Stride+offsetCol] += srcDRaw.Data[i*srcDRaw.Stride+inCol] * value
		}
	}

	result, err := newNoCopy(denseCopy(sys.A), B, denseCopy(sys.C), D, sys.Dt)
	if err != nil {
		return nil, err
	}
	result.E = copyDescriptorE(sys.E)
	result.Delay = selectInputDelayWithOffset(sys.Delay, keep, p, mNew)
	result.InputDelay = selectDelaySliceWithOffset(sys.InputDelay, keep)
	result.OutputDelay = copySliceOrNil(sys.OutputDelay)
	if sys.LFT != nil {
		result.LFT = &LFTDelay{
			Tau: append([]float64(nil), sys.LFT.Tau...),
			B2:  copyDelayOrNil(sys.LFT.B2),
			C2:  copyDelayOrNil(sys.LFT.C2),
			D12: selectLFTD12Columns(sys.LFT.D12, keep),
			D21: copyDelayOrNil(sys.LFT.D21),
			D22: copyDelayOrNil(sys.LFT.D22),
		}
	}
	result.InputName = append(selectStringSlice(sys.InputName, keep), offsetName)
	result.OutputName = copyStringSlice(sys.OutputName)
	result.StateName = copyStringSlice(sys.StateName)
	return result, nil
}

// AugmentInternalDelayOutputs appends the internal-delay input signals
// z = C2·x + D21·u + D22·w to the outputs. Appended rows inherit InputDelay
// and carry no OutputDelay or IODelay.
func (sys *System) AugmentInternalDelayOutputs(prefix string) (*System, error) {
	if sys == nil {
		return nil, fmt.Errorf("AugmentInternalDelayOutputs: nil system: %w", ErrDimensionMismatch)
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
	result.OutputName = append(copyStringSlice(sys.OutputName), autoLabel(prefix, N)...)
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

func selectDelaySliceWithOffset(values []float64, indices []int) []float64 {
	if values == nil {
		return nil
	}
	return append(selectDelaySlice(values, indices), 0)
}

func selectInputDelayWithOffset(delay *mat.Dense, inputs []int, p, mNew int) *mat.Dense {
	if delay == nil {
		return nil
	}
	out := mat.NewDense(p, mNew, nil)
	for i := range p {
		for j, idx := range inputs {
			out.Set(i, j, delay.At(i, idx))
		}
	}
	return out
}

func selectLFTD12Columns(D12 *mat.Dense, inputs []int) *mat.Dense {
	if D12 == nil {
		return nil
	}
	r, _ := D12.Dims()
	out := mat.NewDense(r, len(inputs)+1, nil)
	for i := range r {
		for j, idx := range inputs {
			out.Set(i, j, D12.At(i, idx))
		}
	}
	return out
}
