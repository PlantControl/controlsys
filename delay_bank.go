package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// buildContinuousPadeDelayBank returns the block-diagonal bank of per-channel
// Padé delay models.
func buildContinuousPadeDelayBank(contDelay []float64, order int) (*System, error) {
	if len(contDelay) == 0 {
		return nil, fmt.Errorf("delay bank has no channels: %w", ErrDimensionMismatch)
	}
	var bank *System
	for _, tau := range contDelay {
		ch, err := buildPadeDelayChannel(tau, order)
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
