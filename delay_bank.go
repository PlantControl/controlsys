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
	sampleDelay             []float64
	continuousDelay         []float64
	channels                int
	dt                      float64
	thiranOrder             int
	padeOrder               int
	requiresFractionalModel bool
	kind                    delayBankKind
}

func buildContinuousDelayBank(contDelay []float64, channels int, dt float64, thiranOrder int) (*System, error) {
	sampleDelay := make([]float64, channels)
	for i := range channels {
		sampleDelay[i] = contDelay[i] / dt
	}
	return buildDelayBank(delayBankSpec{
		sampleDelay:             sampleDelay,
		channels:                channels,
		dt:                      dt,
		thiranOrder:             thiranOrder,
		requiresFractionalModel: true,
	})
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

func buildDelayBank(spec delayBankSpec) (*System, error) {
	if spec.requiresFractionalModel && !hasFractionalSampleDelay(spec.sampleDelay) {
		return nil, nil
	}

	var bank *System
	for i := 0; i < spec.channels; i++ {
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
