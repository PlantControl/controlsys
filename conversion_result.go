package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

type ConversionResult struct {
	System          *System
	Method          C2DMethod
	InitialStateMap *mat.Dense
	SourceStates    int
	Inputs          int
	DelaySignals    int
	Approximate     bool
	Warnings        []string

	statesEliminated bool
}

func (r *ConversionResult) MapInitialState(state, input, delayOutput []float64) ([]float64, error) {
	values := make([]float64, r.SourceStates+r.Inputs+r.DelaySignals)
	offset := 0
	for i, part := range [][]float64{state, input, delayOutput} {
		count := []int{r.SourceStates, r.Inputs, r.DelaySignals}[i]
		if len(part) != 0 && len(part) != count {
			return nil, fmt.Errorf("initial-condition part %d has %d values, need %d: %w", i, len(part), count, ErrDimensionMismatch)
		}
		for _, value := range part {
			if math.IsNaN(value) || math.IsInf(value, 0) {
				return nil, fmt.Errorf("initial conditions must be finite: %w", ErrInvalidConversionOptions)
			}
		}
		copy(values[offset:offset+count], part)
		offset += count
	}
	n, _, _ := r.System.Dims()
	if r.InitialStateMap == nil {
		if n > 0 && (r.Method == C2DMethodFOH || r.Method == C2DMethodTustin) {
			for _, value := range input {
				if value != 0 {
					return nil, fmt.Errorf("this conversion has no exact initial-input mapping: %w", ErrSingularTransform)
				}
			}
		}
		for _, value := range state {
			if value != 0 && r.statesEliminated {
				return nil, fmt.Errorf("%w: %w", ErrDescriptorInitialState, ErrDescriptorSingular)
			}
			if value != 0 {
				return nil, fmt.Errorf("this conversion has no exact nonzero initial-state mapping: %w", ErrSingularTransform)
			}
		}
		for _, value := range delayOutput {
			if value != 0 {
				return nil, fmt.Errorf("this conversion has no nonzero delay-history mapping: %w", ErrSingularTransform)
			}
		}
		return make([]float64, n), nil
	}
	rows, columns := r.InitialStateMap.Dims()
	if rows != n || columns != len(values) {
		return nil, fmt.Errorf("invalid initial-condition map: %w", ErrDimensionMismatch)
	}
	result := make([]float64, rows)
	raw := r.InitialStateMap.RawMatrix()
	for i := range rows {
		for j, value := range values {
			result[i] += raw.Data[i*raw.Stride+j] * value
		}
		if math.IsNaN(result[i]) || math.IsInf(result[i], 0) {
			return nil, ErrOverflow
		}
	}
	return result, nil
}

// DiscretizeWithResult discretizes sys like DiscretizeWithOpts and reports
// the initial-condition map (MATLAB [sysd,G] = c2d(sysc,Ts)) and limitations.
// A singular-E descriptor model is reduced as in MATLAB dss2ss before
// conversion; its algebraic states are eliminated, so InitialStateMap is nil
// and MapInitialState rejects a nonzero source state with
// ErrDescriptorInitialState, as the time responses do.
func (sys *System) DiscretizeWithResult(dt float64, opts C2DOptions) (*ConversionResult, error) {
	opts, err := normalizeC2DOptions(dt, opts)
	if err != nil {
		return nil, err
	}
	source := sys
	sys, reduced, err := conversionStandardForm(sys, "DiscretizeWithResult")
	if err != nil {
		return nil, err
	}
	out, err := sys.DiscretizeWithOpts(dt, opts)
	if err != nil {
		return nil, err
	}
	result := newConversionResult(source, out, opts.Method)
	result.statesEliminated = reduced
	result.Approximate = opts.Method == C2DMethodTustin || opts.Method == C2DMethodMatched || opts.Method == C2DMethodLeastSquares
	if !reduced && conversionCanMapState(sys, out, opts.Method, dt) && !(opts.ThiranOrder > 0 && conversionInitialMapHasFraction(sys, dt)) {
		result.InitialStateMap, err = conversionStateMap(sys, out, opts.Method, opts.PrewarpFrequency, false)
		if err != nil {
			return nil, err
		}
	}
	if (opts.Method == C2DMethodTustin || opts.Method == C2DMethodMatched) && conversionInitialMapHasFraction(sys, dt) {
		if opts.ThiranOrder > 0 {
			result.Warnings = append(result.Warnings, "Thiran filters approximate fractional delays.")
		} else {
			result.Warnings = append(result.Warnings, "Fractional delays are rounded to the nearest sample.")
		}
	}
	if sys.HasInternalDelay() && (opts.Method == C2DMethodZOH || opts.Method == C2DMethodFOH) {
		result.Approximate = true
		result.Warnings = append(result.Warnings, "Hold conversion approximates intersample internal feedback.")
	}
	result.describeLimitations(sys)
	if opts.Method == C2DMethodImpulse && denseNorm(sys.D) != 0 {
		result.Warnings = append(result.Warnings, "Impulse conversion omits the continuous direct impulse term.")
	}
	return result, nil
}

// D2CWithResult converts sys like D2CWithOpts and reports the
// initial-condition map and limitations. Singular-E descriptor models are
// handled as in DiscretizeWithResult.
func (sys *System) D2CWithResult(opts D2COptions) (*ConversionResult, error) {
	if opts.Method == "" {
		opts.Method = C2DMethodZOH
	}
	source := sys
	sys, reduced, err := conversionStandardForm(sys, "D2CWithResult")
	if err != nil {
		return nil, err
	}
	out, err := sys.D2CWithOpts(opts)
	if err != nil {
		return nil, err
	}
	result := newConversionResult(source, out, opts.Method)
	result.statesEliminated = reduced
	result.Approximate = opts.Method == C2DMethodTustin || opts.Method == C2DMethodMatched
	if sys.HasInternalDelay() && (opts.Method == C2DMethodZOH || opts.Method == C2DMethodFOH) {
		result.Approximate = true
		result.Warnings = append(result.Warnings, "Hold conversion approximates intersample internal feedback.")
	}
	if !reduced && conversionCanMapState(sys, out, opts.Method, sys.Dt) {
		result.InitialStateMap, err = conversionStateMap(sys, out, opts.Method, opts.PrewarpFrequency, true)
		if err != nil {
			return nil, err
		}
	}
	result.Warnings = append(result.Warnings, "The inverse selects an intersample model; frequencies lost through sampling cannot be recovered.")
	result.describeLimitations(sys)
	return result, nil
}

func (sys *System) D2DWithResult(dt float64, opts C2DOptions) (*ConversionResult, error) {
	plan, err := newD2DPlan(sys, dt, opts)
	if err != nil {
		return nil, err
	}
	opts = plan.opts
	if math.Abs(dt-sys.Dt) < 1e-14*math.Max(dt, sys.Dt) {
		result := newConversionResult(sys, sys.Copy(), opts.Method)
		n, _, _ := sys.Dims()
		result.InitialStateMap = initialStateIdentity(n, result.SourceStates, result.SourceStates+result.Inputs+result.DelaySignals)
		return result, nil
	}
	inverse, err := sys.D2CWithResult(D2COptions{Method: opts.Method, PrewarpFrequency: opts.PrewarpFrequency})
	if err != nil {
		return nil, err
	}
	forward, err := inverse.System.DiscretizeWithResult(dt, opts)
	if err != nil {
		return nil, err
	}
	result := newConversionResult(sys, forward.System, opts.Method)
	result.statesEliminated = inverse.statesEliminated
	if inverse.InitialStateMap != nil && forward.InitialStateMap != nil && result.DelaySignals == 0 {
		nc, _, _ := inverse.System.Dims()
		nf, columns := forward.InitialStateMap.Dims()
		_, sourceColumns := inverse.InitialStateMap.Dims()
		if columns == nc+result.Inputs && sourceColumns == result.SourceStates+result.Inputs {
			result.InitialStateMap = newDense(nf, sourceColumns)
			target := result.InitialStateMap.RawMatrix()
			f := forward.InitialStateMap.RawMatrix()
			b := inverse.InitialStateMap.RawMatrix()
			for i := range nf {
				for j := range sourceColumns {
					for k := range nc {
						target.Data[i*target.Stride+j] += f.Data[i*f.Stride+k] * b.Data[k*b.Stride+j]
					}
					if j >= result.SourceStates {
						target.Data[i*target.Stride+j] += f.Data[i*f.Stride+nc+j-result.SourceStates]
					}
				}
			}
		}
	}
	result.Approximate = forward.Approximate
	result.Warnings = append(result.Warnings, inverse.Warnings...)
	result.Warnings = append(result.Warnings, forward.Warnings...)
	return result, nil
}

func newConversionResult(source, out *System, method C2DMethod) *ConversionResult {
	n, m, _ := source.Dims()
	return &ConversionResult{System: out, Method: method, SourceStates: n, Inputs: m, DelaySignals: source.internalDelayCount()}
}

func (r *ConversionResult) describeLimitations(source *System) {
	targetStates, _, _ := r.System.Dims()
	if targetStates > r.SourceStates {
		r.Warnings = append(r.Warnings, fmt.Sprintf("Conversion adds %d states.", targetStates-r.SourceStates))
	}
	if source.HasDelay() || r.System.HasDelay() {
		r.Warnings = append(r.Warnings, "Delay history must be initialized separately; the returned mapping covers zero-history conversion only.")
	}
	if r.statesEliminated {
		r.Warnings = append(r.Warnings, "Algebraic descriptor states are eliminated; the state coordinates change.")
	}
	if r.InitialStateMap == nil && r.SourceStates > 0 {
		r.Warnings = append(r.Warnings, "An exact nonzero initial-state mapping is unavailable for this realization.")
	}
}

func initialStateIdentity(rows, sourceStates, columns int) *mat.Dense {
	mapping := newDense(rows, columns)
	raw := mapping.RawMatrix()
	for i := range min(rows, sourceStates) {
		raw.Data[i*raw.Stride+i] = 1
	}
	return mapping
}

func conversionStateMap(source, out *System, method C2DMethod, prewarp float64, inverse bool) (*mat.Dense, error) {
	n, m, _ := source.Dims()
	nt, _, _ := out.Dims()
	if nt == 0 || method == C2DMethodMatched || method == C2DMethodLeastSquares {
		return nil, nil
	}
	mapping := initialStateIdentity(nt, n, n+m+source.internalDelayCount())
	if method == C2DMethodZOH || method == C2DMethodImpulse || n == 0 {
		return mapping, nil
	}
	raw := mapping.RawMatrix()
	switch method {
	case C2DMethodTustin:
		dt := out.Dt
		if inverse {
			dt = source.Dt
		}
		beta, err := tustinBeta(dt, prewarp)
		if err != nil {
			return nil, err
		}
		scale := beta
		if inverse {
			scale = 2
		}
		a, b := source.A.RawMatrix(), source.B.RawMatrix()
		for i := range n {
			for j := range n {
				value := -a.Data[i*a.Stride+j]
				if inverse {
					value = a.Data[i*a.Stride+j]
				}
				if i == j {
					if inverse {
						value += 1
					} else {
						value += beta
					}
				}
				raw.Data[i*raw.Stride+j] = value / scale
			}
			for j := range m {
				value := b.Data[i*b.Stride+j] / scale
				if !inverse {
					value = -value
				}
				raw.Data[i*raw.Stride+n+j] = value
			}
		}
		if source.HasInternalDelay() {
			b2 := source.LFT.B2.RawMatrix()
			for i := range n {
				for j := range source.internalDelayCount() {
					value := b2.Data[i*b2.Stride+j] / scale
					if !inverse {
						value = -value
					}
					raw.Data[i*raw.Stride+n+m+j] = value
				}
			}
		}

	case C2DMethodFOH:
		continuous := source
		dt := out.Dt
		if inverse {
			continuous = out
			dt = source.Dt
		}
		if nt != n {
			return nil, nil
		}
		_, _, k1 := fohKernel(continuous.A, dt)
		var gamma mat.Dense
		if m > 0 {
			gamma.Mul(k1, continuous.B)
		}
		gr := gamma.RawMatrix()
		for i := range n {
			for j := range m {
				value := gr.Data[i*gr.Stride+j]
				if !inverse {
					value = -value
				}
				raw.Data[i*raw.Stride+n+j] = value
			}
		}
		if continuous.HasInternalDelay() {
			var gamma2 mat.Dense
			if n > 0 {
				gamma2.Mul(k1, continuous.LFT.B2)
			}
			g2 := gamma2.RawMatrix()
			for i := range n {
				for j := range source.internalDelayCount() {
					value := g2.Data[i*g2.Stride+j]
					if !inverse {
						value = -value
					}
					raw.Data[i*raw.Stride+n+m+j] = value
				}
			}
		}
	default:
		return nil, nil
	}
	return mapping, nil
}

func conversionInitialMapHasFraction(sys *System, dt float64) bool {
	if conversionHasFractionalExternalDelay(sys, dt) || conversionHasFractionalPathDelay(sys.Delay, dt) {
		return true
	}
	if sys.HasInternalDelay() {
		for _, tau := range sys.LFT.Tau {
			if !isIntegerSampleDelay(tau / dt) {
				return true
			}
		}
	}
	return false
}

func conversionCanMapState(source, out *System, method C2DMethod, dt float64) bool {
	if !source.HasDelay() && !out.HasDelay() {
		return true
	}
	n, _, _ := source.Dims()
	nt, _, _ := out.Dims()
	if method == C2DMethodTustin {
		return nt == n && source.internalDelayCount() == out.internalDelayCount()
	}
	if method == C2DMethodFOH && nt == n && source.HasInternalDelay() && !hasExternalDelay(source, true) {
		for _, tau := range source.LFT.Tau {
			if source.IsContinuous() && !isIntegerSampleDelay(tau/dt) {
				return false
			}
		}
		return true
	}
	if method != C2DMethodZOH || nt < n {
		return false
	}
	for _, part := range [][]float64{source.InputDelay, source.OutputDelay} {
		for _, tau := range part {
			if !isIntegerSampleDelay(tau/dt) && source.IsContinuous() {
				return false
			}
		}
	}
	if source.Delay != nil {
		r, c := source.Delay.Dims()
		for i := range r {
			for j := range c {
				if source.IsContinuous() && !isIntegerSampleDelay(source.Delay.At(i, j)/dt) {
					return false
				}
			}
		}
	}
	if source.HasInternalDelay() {
		for _, tau := range source.LFT.Tau {
			if source.IsContinuous() && !isIntegerSampleDelay(tau/dt) {
				return false
			}
		}
	}
	return true
}
