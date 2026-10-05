package controlsys

import (
	"fmt"
	"slices"
)

// ModelArray is an N-dimensional array of LTI models sharing inputs, outputs,
// sample time and signal names, like a MATLAB model array. A slot may be
// void (no model); state counts may differ between models.
type ModelArray struct {
	models     []*System
	shape      []int
	m          int
	p          int
	dt         float64
	inputName  []string
	outputName []string
}

// ModelArrayFreqResponse holds the frequency response of each model of a
// ModelArray, indexed like the array.
type ModelArrayFreqResponse struct {
	Shape     []int
	Omega     []float64
	P         int
	M         int
	responses []*FreqResponseMatrix
}

// ResponseFlat returns the response of the model at flat index i, or
// ErrVoidModel for a void slot.
func (r *ModelArrayFreqResponse) ResponseFlat(i int) (*FreqResponseMatrix, error) {
	if r == nil {
		return nil, fmt.Errorf("ModelArrayFreqResponse.ResponseFlat: nil response: %w", ErrInvalidArgument)
	}
	return voidableAt("ModelArrayFreqResponse.ResponseFlat", r.responses, i)
}

// ModelArrayTimeResponse holds the time response of each model of a
// ModelArray, indexed like the array.
type ModelArrayTimeResponse struct {
	Shape     []int
	responses []*TimeResponse
}

// ResponseFlat returns the response of the model at flat index i, or
// ErrVoidModel for a void slot.
func (r *ModelArrayTimeResponse) ResponseFlat(i int) (*TimeResponse, error) {
	if r == nil {
		return nil, fmt.Errorf("ModelArrayTimeResponse.ResponseFlat: nil response: %w", ErrInvalidArgument)
	}
	return voidableAt("ModelArrayTimeResponse.ResponseFlat", r.responses, i)
}

func voidableAt[T any](op string, items []*T, i int) (*T, error) {
	if i < 0 || i >= len(items) {
		return nil, fmt.Errorf("%s: index %d out of range [0,%d): %w", op, i, len(items), ErrInvalidArgument)
	}
	if items[i] == nil {
		return nil, fmt.Errorf("%s: index %d: %w", op, i, ErrVoidModel)
	}
	return items[i], nil
}

// NewModelArray builds a model array of the given shape from models listed
// in row-major order (last index fastest); a nil entry is a void slot. The
// models are copied. At least one model must be present and all must share
// inputs, outputs and signal names (ErrDimensionMismatch, ErrInvalidArgument)
// and sample time (ErrDomainMismatch).
func NewModelArray(shape []int, models []*System) (*ModelArray, error) {
	if len(shape) == 0 {
		return nil, fmt.Errorf("NewModelArray: shape is empty: %w", ErrInvalidArgument)
	}
	total := 1
	for _, dim := range shape {
		if dim <= 0 {
			return nil, fmt.Errorf("NewModelArray: invalid shape dimension %d: %w", dim, ErrInvalidArgument)
		}
		total *= dim
	}
	if total != len(models) {
		return nil, fmt.Errorf("NewModelArray: shape product %d != %d models: %w", total, len(models), ErrDimensionMismatch)
	}

	arr := &ModelArray{
		models: make([]*System, len(models)),
		shape:  slices.Clone(shape),
	}
	var ref *System
	for _, sys := range models {
		if sys == nil {
			continue
		}
		if err := sys.validate(); err != nil {
			return nil, fmt.Errorf("NewModelArray: %w", err)
		}
		if ref == nil {
			ref = sys
			_, arr.m, arr.p = sys.Dims()
			arr.dt = sys.Dt
			arr.inputName = copyStringSlice(sys.InputName)
			arr.outputName = copyStringSlice(sys.OutputName)
			continue
		}
		if err := validateModelArrayCompatible(ref, sys); err != nil {
			return nil, fmt.Errorf("NewModelArray: %w", err)
		}
	}
	if ref == nil {
		return nil, fmt.Errorf("NewModelArray: every slot is void: %w", ErrInvalidArgument)
	}
	for i, sys := range models {
		if sys != nil {
			arr.models[i] = sys.Copy()
		}
	}
	return arr, nil
}

// StackModelArrays stacks arrays of equal shape along a new leading
// dimension, like MATLAB stack(1,...).
func StackModelArrays(arrays ...*ModelArray) (*ModelArray, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("StackModelArrays: no arrays: %w", ErrInvalidArgument)
	}
	var ref *ModelArray
	for _, arr := range arrays {
		if arr == nil {
			return nil, fmt.Errorf("StackModelArrays: nil array: %w", ErrInvalidArgument)
		}
		if ref == nil {
			ref = arr
			continue
		}
		if !slices.Equal(ref.shape, arr.shape) {
			return nil, fmt.Errorf("StackModelArrays: shape %v != %v: %w", arr.shape, ref.shape, ErrDimensionMismatch)
		}
		if err := validateModelArrayHeadersCompatible(ref, arr); err != nil {
			return nil, fmt.Errorf("StackModelArrays: %w", err)
		}
	}
	shape := make([]int, len(ref.shape)+1)
	shape[0] = len(arrays)
	copy(shape[1:], ref.shape)
	models := copyModelArrays(arrays)
	return NewModelArray(shape, models)
}

// ConcatModelArrays concatenates arrays into a one-dimensional array of
// their models in order.
func ConcatModelArrays(arrays ...*ModelArray) (*ModelArray, error) {
	if len(arrays) == 0 {
		return nil, fmt.Errorf("ConcatModelArrays: no arrays: %w", ErrInvalidArgument)
	}
	var ref *ModelArray
	total := 0
	for _, arr := range arrays {
		if arr == nil {
			return nil, fmt.Errorf("ConcatModelArrays: nil array: %w", ErrInvalidArgument)
		}
		if ref == nil {
			ref = arr
		} else if err := validateModelArrayHeadersCompatible(ref, arr); err != nil {
			return nil, fmt.Errorf("ConcatModelArrays: %w", err)
		}
		total += arr.Len()
	}
	models := copyModelArrays(arrays)
	return NewModelArray([]int{total}, models)
}

func copyModelArrays(arrays []*ModelArray) []*System {
	total := 0
	for _, arr := range arrays {
		total += arr.Len()
	}
	models := make([]*System, 0, total)
	for _, arr := range arrays {
		for _, sys := range arr.models {
			if sys == nil {
				models = append(models, nil)
			} else {
				models = append(models, sys.Copy())
			}
		}
	}
	return models
}

// Shape returns the array dimensions.
func (a *ModelArray) Shape() []int {
	if a == nil {
		return nil
	}
	return slices.Clone(a.shape)
}

// Len returns the number of slots, void or not.
func (a *ModelArray) Len() int {
	if a == nil {
		return 0
	}
	return len(a.models)
}

// IOSize returns the number of outputs and inputs shared by every model,
// like MATLAB size(sys,1:2). Models may differ in state count.
func (a *ModelArray) IOSize() (p, m int) {
	if a == nil {
		return 0, 0
	}
	return a.p, a.m
}

// InputName returns the input names shared by the models.
func (a *ModelArray) InputName() []string {
	if a == nil {
		return nil
	}
	return copyStringSlice(a.inputName)
}

// OutputName returns the output names shared by the models.
func (a *ModelArray) OutputName() []string {
	if a == nil {
		return nil
	}
	return copyStringSlice(a.outputName)
}

// Model returns a copy of the model at the given array index, ErrVoidModel
// for a void slot and ErrInvalidArgument for an index out of range.
func (a *ModelArray) Model(index ...int) (*System, error) {
	if a == nil {
		return nil, fmt.Errorf("ModelArray.Model: nil array: %w", ErrInvalidArgument)
	}
	flat, err := a.flatIndex("ModelArray.Model", index)
	if err != nil {
		return nil, err
	}
	return a.modelFlat("ModelArray.Model", flat)
}

// ModelFlat returns a copy of the model at flat (row-major) index i,
// ErrVoidModel for a void slot and ErrInvalidArgument for i out of range.
func (a *ModelArray) ModelFlat(i int) (*System, error) {
	if a == nil {
		return nil, fmt.Errorf("ModelArray.ModelFlat: nil array: %w", ErrInvalidArgument)
	}
	return a.modelFlat("ModelArray.ModelFlat", i)
}

func (a *ModelArray) modelFlat(op string, i int) (*System, error) {
	sys, err := voidableAt(op, a.models, i)
	if err != nil {
		return nil, err
	}
	return sys.Copy(), nil
}

// IsVoid reports whether the slot at the given array index holds no model.
func (a *ModelArray) IsVoid(index ...int) (bool, error) {
	if a == nil {
		return false, fmt.Errorf("ModelArray.IsVoid: nil array: %w", ErrInvalidArgument)
	}
	flat, err := a.flatIndex("ModelArray.IsVoid", index)
	if err != nil {
		return false, err
	}
	return a.models[flat] == nil, nil
}

// SelectFlat returns a one-dimensional array of the slots at the given flat
// indices; selecting only void slots returns ErrInvalidArgument.
func (a *ModelArray) SelectFlat(indices ...int) (*ModelArray, error) {
	if a == nil {
		return nil, fmt.Errorf("ModelArray.SelectFlat: nil array: %w", ErrInvalidArgument)
	}
	models := make([]*System, len(indices))
	for i, idx := range indices {
		if idx < 0 || idx >= len(a.models) {
			return nil, fmt.Errorf("ModelArray.SelectFlat: index %d out of range: %w", idx, ErrInvalidArgument)
		}
		if a.models[idx] != nil {
			models[i] = a.models[idx].Copy()
		}
	}
	arr, err := NewModelArray([]int{len(indices)}, models)
	if err != nil {
		return nil, fmt.Errorf("ModelArray.SelectFlat: %w", err)
	}
	return arr, nil
}

// FreqResponse evaluates every model at omega; void slots stay void.
func (a *ModelArray) FreqResponse(omega []float64) (*ModelArrayFreqResponse, error) {
	if a == nil {
		return nil, fmt.Errorf("ModelArray.FreqResponse: nil array: %w", ErrInvalidArgument)
	}
	out := &ModelArrayFreqResponse{
		Shape:     slices.Clone(a.shape),
		Omega:     copyFloatSlice(omega),
		P:         a.p,
		M:         a.m,
		responses: make([]*FreqResponseMatrix, len(a.models)),
	}
	for i, sys := range a.models {
		if sys == nil {
			continue
		}
		resp, err := sys.FreqResponse(omega)
		if err != nil {
			return nil, fmt.Errorf("ModelArray.FreqResponse: model %d: %w", i, err)
		}
		out.responses[i] = resp
	}
	return out, nil
}

// Step simulates the step response of every model to tFinal; void slots
// stay void.
func (a *ModelArray) Step(tFinal float64) (*ModelArrayTimeResponse, error) {
	if a == nil {
		return nil, fmt.Errorf("ModelArray.Step: nil array: %w", ErrInvalidArgument)
	}
	out := &ModelArrayTimeResponse{
		Shape:     slices.Clone(a.shape),
		responses: make([]*TimeResponse, len(a.models)),
	}
	for i, sys := range a.models {
		if sys == nil {
			continue
		}
		resp, err := Step(sys, tFinal)
		if err != nil {
			return nil, fmt.Errorf("ModelArray.Step: model %d: %w", i, err)
		}
		out.responses[i] = resp
	}
	return out, nil
}

func (a *ModelArray) flatIndex(op string, index []int) (int, error) {
	if len(index) != len(a.shape) {
		return 0, fmt.Errorf("%s: got %d indices for rank %d: %w", op, len(index), len(a.shape), ErrInvalidArgument)
	}
	flat := 0
	stride := 1
	for dim, v := range slices.Backward(a.shape) {
		idx := index[dim]
		if idx < 0 || idx >= v {
			return 0, fmt.Errorf("%s: index %d out of range for dimension %d: %w", op, idx, dim, ErrInvalidArgument)
		}
		flat += idx * stride
		stride *= v
	}
	return flat, nil
}

func validateModelArrayCompatible(ref, sys *System) error {
	_, rm, rp := ref.Dims()
	_, sm, sp := sys.Dims()
	if rm != sm || rp != sp {
		return fmt.Errorf("model dimensions (%d,%d) != (%d,%d): %w", sp, sm, rp, rm, ErrDimensionMismatch)
	}
	if ref.Dt != sys.Dt {
		return fmt.Errorf("sample time %g != %g: %w", sys.Dt, ref.Dt, ErrDomainMismatch)
	}
	if !stringSlicesCompatible(ref.InputName, sys.InputName) || !stringSlicesCompatible(ref.OutputName, sys.OutputName) {
		return fmt.Errorf("signal names differ: %w", ErrInvalidArgument)
	}
	return nil
}

func validateModelArrayHeadersCompatible(ref, arr *ModelArray) error {
	if ref.m != arr.m || ref.p != arr.p {
		return fmt.Errorf("model dimensions (%d,%d) != (%d,%d): %w", arr.p, arr.m, ref.p, ref.m, ErrDimensionMismatch)
	}
	if ref.dt != arr.dt {
		return fmt.Errorf("sample time %g != %g: %w", arr.dt, ref.dt, ErrDomainMismatch)
	}
	if !stringSlicesCompatible(ref.inputName, arr.inputName) || !stringSlicesCompatible(ref.outputName, arr.outputName) {
		return fmt.Errorf("signal names differ: %w", ErrInvalidArgument)
	}
	return nil
}

func stringSlicesCompatible(a, b []string) bool {
	return len(a) == 0 || len(b) == 0 || slices.Equal(a, b)
}
