package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

// C2DMap converts sys like C2D and also returns the initial-condition map G
// of MATLAB [sysd,G] = c2d(sysc,Ts,opts): x[0] = G·[x0; u0; w0], where x0
// is the continuous state, u0 the input at t = 0 and w0 the internal-delay
// outputs (zero history). G is nd×(n+m+N). Zero-order hold and impulse maps
// are exact; Tustin and FOH maps account for the initial input.
//
// When no exact map exists for the conversion (matched and least-squares
// methods, Thiran-approximated fractional delays, delays that change the
// state dimension, a result with no states), C2DMap returns
// ErrOptionUnsupported; C2D still converts. A singular-E descriptor model,
// whose algebraic states are eliminated, returns ErrDescriptorInitialState.
func (sys *System) C2DMap(dt float64, opts C2DOptions) (*System, *mat.Dense, error) {
	if sys == nil {
		return nil, nil, fmt.Errorf("C2DMap: system is nil: %w", ErrInvalidArgument)
	}
	opts, err := normalizeC2DOptions(dt, opts)
	if err != nil {
		return nil, nil, fmt.Errorf("C2DMap: %w", err)
	}
	std, reduced, err := conversionStandardForm(sys, "C2DMap")
	if err != nil {
		return nil, nil, err
	}
	if reduced {
		return nil, nil, fmt.Errorf("C2DMap: algebraic descriptor states are eliminated: %w", ErrDescriptorInitialState)
	}
	out, err := std.C2D(dt, opts)
	if err != nil {
		return nil, nil, err
	}
	if !conversionCanMapState(std, out, opts.Method, dt) || (opts.ThiranOrder > 0 && conversionInitialMapHasFraction(std, dt)) {
		return nil, nil, fmt.Errorf("C2DMap: no exact initial-condition map for this delay structure: %w", ErrOptionUnsupported)
	}
	G, err := conversionStateMap(std, out, opts.Method, opts.PrewarpFrequency, false)
	if err != nil {
		return nil, nil, fmt.Errorf("C2DMap: %w", err)
	}
	return out, G, nil
}

// D2CMap converts sys like D2C and also returns the map G of MATLAB
// [sysc,G] = d2c(sysd,opts): xc(k·Ts) = G·[xd[k]; u[k]; w[k]]. Its
// availability follows C2DMap.
func (sys *System) D2CMap(opts D2COptions) (*System, *mat.Dense, error) {
	if sys == nil {
		return nil, nil, fmt.Errorf("D2CMap: system is nil: %w", ErrInvalidArgument)
	}
	if opts.Method == "" {
		opts.Method = C2DMethodZOH
	}
	std, reduced, err := conversionStandardForm(sys, "D2CMap")
	if err != nil {
		return nil, nil, err
	}
	if reduced {
		return nil, nil, fmt.Errorf("D2CMap: algebraic descriptor states are eliminated: %w", ErrDescriptorInitialState)
	}
	out, err := std.D2C(opts)
	if err != nil {
		return nil, nil, err
	}
	if !conversionCanMapState(std, out, opts.Method, std.Dt) {
		return nil, nil, fmt.Errorf("D2CMap: no exact initial-condition map for this delay structure: %w", ErrOptionUnsupported)
	}
	G, err := conversionStateMap(std, out, opts.Method, opts.PrewarpFrequency, true)
	if err != nil {
		return nil, nil, fmt.Errorf("D2CMap: %w", err)
	}
	return out, G, nil
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
	if nt == 0 {
		return nil, fmt.Errorf("converted model has no states: %w", ErrDimensionMismatch)
	}
	if method == C2DMethodMatched || method == C2DMethodLeastSquares {
		return nil, fmt.Errorf("method %q has no exact initial-condition map: %w", method, ErrOptionUnsupported)
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
			return nil, fmt.Errorf("foh changes the state dimension: %w", ErrOptionUnsupported)
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
		return nil, fmt.Errorf("method %q has no initial-condition map: %w", method, ErrOptionUnsupported)
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
