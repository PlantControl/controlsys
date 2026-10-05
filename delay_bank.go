package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

type delayBankKind int

const (
	delayBankSample delayBankKind = iota
	delayBankPade
)

type delayBankSpec struct {
	sampleDelay     []float64
	continuousDelay []float64
	channels        int
	dt              float64
	thiranOrder     int
	padeOrder       int
	kind            delayBankKind
}

// continuousToSampleDelay converts continuous delays to sample counts at dt.
func continuousToSampleDelay(contDelay []float64, dt float64) []float64 {
	samples := make([]float64, len(contDelay))
	for i, d := range contDelay {
		samples[i] = d / dt
	}
	return samples
}

func buildDiscreteSampleDelayBank(sampleDelay []float64, channels int, dt float64, thiranOrder int) (*System, error) {
	return buildDelayBank(delayBankSpec{
		sampleDelay: sampleDelay,
		channels:    channels,
		dt:          dt,
		thiranOrder: thiranOrder,
		kind:        delayBankSample,
	})
}

func buildContinuousPadeDelayBank(contDelay []float64, order int) (*System, error) {
	return buildDelayBank(delayBankSpec{
		continuousDelay: contDelay,
		channels:        len(contDelay),
		padeOrder:       order,
		kind:            delayBankPade,
	})
}

// buildDelayBank returns the block-diagonal bank of per-channel delay models.
func buildDelayBank(spec delayBankSpec) (*System, error) {
	if spec.channels == 0 {
		return nil, fmt.Errorf("delay bank has no channels: %w", ErrDimensionMismatch)
	}
	var bank *System
	for i := range spec.channels {
		ch, err := buildDelayChannel(spec, i)
		if err != nil {
			return nil, err
		}
		if bank == nil {
			bank = ch
			continue
		}
		bank, err = Append(bank, ch)
		if err != nil {
			return nil, err
		}
	}
	return bank, nil
}

func hasFractionalSampleDelay(sampleDelay []float64) bool {
	for _, samples := range sampleDelay {
		if samples == 0 {
			continue
		}
		if !isIntegerSampleDelay(samples) {
			return true
		}
	}
	return false
}

func buildDelayChannel(spec delayBankSpec, channel int) (*System, error) {
	if spec.kind == delayBankPade {
		return buildPadeDelayChannel(spec.continuousDelay[channel], spec.padeOrder)
	}
	return buildSampleDelayChannel(spec.sampleDelay[channel], spec.dt, spec.thiranOrder)
}

func buildSampleDelayChannel(samples, dt float64, thiranOrder int) (*System, error) {
	if samples == 0 {
		return NewGain(mat.NewDense(1, 1, []float64{1}), dt)
	}
	if isIntegerSampleDelay(samples) {
		return integerDelaySS(int(math.Round(samples)), dt)
	}
	return ThiranDelay(samples*dt, thiranOrder, dt)
}

func buildPadeDelayChannel(tau float64, order int) (*System, error) {
	if tau == 0 {
		return NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	}
	pd, err := PadeDelay(tau, order)
	if err != nil {
		return nil, fmt.Errorf("buildPadeDelayBank: %w", err)
	}
	return pd, nil
}

func isIntegerSampleDelay(samples float64) bool {
	return math.Abs(samples-math.Round(samples)) < 1e-9
}
