package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

func conversionHoldFeedbackNeedsAbsorption(sys *System, input, output []float64, dt float64) bool {
	for _, delays := range [][]float64{sys.LFT.Tau, input, output} {
		for _, tau := range delays {
			if !isIntegerSampleDelay(tau / dt) {
				return true
			}
		}
	}
	return conversionHasFractionalPathDelay(sys.Delay, dt)
}

// Fractional feedback is absorbed into the open-loop input ports before hold
// discretization; closing sampled feedback then approximates intersample signals.
func discretizeHoldFeedback(sys *System, input, output []float64, dt float64, opts C2DOptions) (*System, error) {
	_, m, p := sys.Dims()
	if conversionHasFractionalPathDelay(sys.Delay, dt) {
		return discretizeHoldFeedbackPaths(sys, input, output, dt, opts)
	}
	count := len(sys.LFT.Tau)
	for _, value := range sys.LFT.Tau {
		if value <= 0 || math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, ErrZeroInternalDelay
		}
	}
	whole, residual, err := conversionDelayParts(sys.LFT.Tau, count, dt)
	if err != nil {
		return nil, err
	}
	inputWhole, inputResidual, err := conversionDelayParts(input, m, dt)
	if err != nil {
		return nil, err
	}
	outputWhole, outputResidual, err := conversionDelayParts(output, p, dt)
	if err != nil {
		return nil, err
	}
	rational := conversionAugmentedRational(sys)
	rational.InputDelay = make([]float64, m+count)
	rational.OutputDelay = make([]float64, p+count)
	copy(rational.InputDelay, inputResidual)
	copy(rational.InputDelay[m:], residual)
	copy(rational.OutputDelay, outputResidual)
	openOpts := opts
	var disc *System
	if opts.Method == C2DMethodFOH && !conversionHasNonzeroDelay(rational.OutputDelay) && rational.Delay == nil {
		disc, err = discretizeFOHInputFractions(rational, dt)
	} else {
		disc, err = rational.C2D(dt, openOpts)
	}
	if err != nil {
		return nil, err
	}
	if opts.DelayModeling == C2DDelayModelingState && disc.HasInternalDelay() {
		disc, err = absorbConversionInternal(disc)
		if err != nil {
			return nil, err
		}
	}
	if disc.HasDelay() {
		disc, err = disc.PullDelaysToLFT()
		if err != nil {
			return nil, err
		}
	}
	tau := append([]float64(nil), whole...)
	if disc.HasInternalDelay() {
		tau = append(tau, disc.LFT.Tau...)
		disc = conversionAugmentedRational(disc)
	}
	out := splitConversionRational(disc, m, p, tau)
	out, err = eliminateConversionZeroDelays(out)
	if err != nil {
		return nil, fmt.Errorf("fractional hold feedback: %w", err)
	}
	out.InputDelay, out.OutputDelay = inputWhole, outputWhole
	if sys.Delay != nil {
		out.Delay, err = convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
	}
	propagateNames(out, sys)
	return out, err
}

func conversionHasNonzeroDelay(values []float64) bool {
	for _, value := range values {
		if value != 0 {
			return true
		}
	}
	return false
}

func discretizeFOHInputFractions(cont *System, dt float64) (*System, error) {
	n, m, p := cont.Dims()
	whole, residual, err := conversionDelayParts(cont.InputDelay, m, dt)
	if err != nil {
		return nil, err
	}
	ordinary := *cont
	ordinary.InputDelay, ordinary.OutputDelay, ordinary.Delay = nil, nil, nil
	out, err := ordinary.discretizeModifiedFOH(dt)
	if err != nil {
		return nil, err
	}
	out.InputDelay = whole
	oldB, oldD := newDense(n, m), newDense(p, m)
	for j, rho := range residual {
		if rho == 0 {
			continue
		}
		column := newDense(n, 1)
		for i := range n {
			column.Set(i, 0, cont.B.At(i, j))
		}
		var plus, minus mat.Dense
		if n > 0 {
			_, total := conversionZOHKernel(cont.A, column, dt)
			_, k0r, k1r := fohKernel(cont.A, rho)
			ef, _, k1f := fohKernel(cont.A, dt-rho)
			var weighted, kernel, adplus mat.Dense
			weighted.Sub(k0r, k1r)
			weighted.Scale(rho/dt, &weighted)
			kernel.Mul(ef, &weighted)
			minus.Mul(&kernel, column)
			plus.Mul(k1f, column)
			plus.Scale((dt-rho)/dt, &plus)
			adplus.Mul(out.A, &plus)
			for i := range n {
				out.B.Set(i, j, total.At(i, 0)+adplus.At(i, 0)-plus.At(i, 0)-minus.At(i, 0))
				oldB.Set(i, j, minus.At(i, 0))
			}
		}
		for i := range p {
			value := cont.D.At(i, j) * (1 - rho/dt)
			for k := range n {
				value += cont.C.At(i, k) * plus.At(k, 0)
			}
			out.D.Set(i, j, value)
			oldD.Set(i, j, cont.D.At(i, j)*rho/dt)
		}
	}
	return attachConversionInputHistory(out, oldB, oldD, newDense(p, m)), nil
}

func discretizeHoldFeedbackPaths(sys *System, input, output []float64, dt float64, opts C2DOptions) (*System, error) {
	n, m, p := sys.Dims()
	var combined *System
	for i := range p {
		for j := range m {
			channel := &System{A: denseCopy(sys.A), B: newDense(n, 1), C: newDense(1, n), D: newDense(1, 1)}
			for k := range n {
				channel.B.Set(k, 0, sys.B.At(k, j))
				channel.C.Set(0, k, sys.C.At(i, k))
			}
			channel.D.Set(0, 0, sys.D.At(i, j))
			count := len(sys.LFT.Tau)
			channel.LFT = &LFTDelay{Tau: append([]float64(nil), sys.LFT.Tau...), B2: denseCopy(sys.LFT.B2), C2: denseCopy(sys.LFT.C2), D12: newDense(1, count), D21: newDense(count, 1), D22: denseCopy(sys.LFT.D22)}
			for k := range count {
				channel.LFT.D12.Set(0, k, sys.LFT.D12.At(i, k))
				channel.LFT.D21.Set(k, 0, sys.LFT.D21.At(k, j))
			}
			in, out := 0., sys.Delay.At(i, j)
			if len(input) > 0 {
				in = input[j]
			}
			if len(output) > 0 {
				out += output[i]
			}
			disc, err := discretizeHoldFeedback(channel, []float64{in}, []float64{out}, dt, opts)
			if err != nil {
				return nil, err
			}
			if disc.HasDelay() {
				disc, err = disc.PullDelaysToLFT()
				if err != nil {
					return nil, err
				}
			}
			expanded := embedConversionChannel(disc, m, p, i, j)
			if combined == nil {
				combined = expanded
			} else {
				combined, err = Parallel(combined, expanded)
				if err != nil {
					return nil, err
				}
			}
		}
	}
	propagateIONames(combined, sys)
	return combined, nil
}
