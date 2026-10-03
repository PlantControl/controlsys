package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

func conversionThiranChannel(samples, dt float64, maxOrder int) (*System, error) {
	if samples < 0 || math.IsNaN(samples) || math.IsInf(samples, 0) {
		return nil, ErrNegativeDelay
	}
	if maxOrder <= 0 {
		return nil, ErrInvalidConversionOptions
	}
	if isIntegerSampleDelay(samples) {
		gain, err := NewGain(mat.NewDense(1, 1, []float64{1}), dt)
		if samples > 0 {
			gain.InputDelay = []float64{math.Round(samples)}
		}
		return gain, err
	}
	ceiling := math.Ceil(samples)
	if ceiling >= float64(int(^uint(0)>>1)) {
		return nil, ErrOverflow
	}
	order := min(int(ceiling), maxOrder)
	whole := int(ceiling) - order
	coefficients := thiranCoeffs(samples-float64(whole), order)
	numerator := make([]float64, order+1)
	for i, value := range coefficients {
		if math.IsNaN(value) || math.IsInf(value, 0) {
			return nil, fmt.Errorf("Thiran coefficients overflow: %w", ErrOverflow)
		}
		numerator[order-i] = value
	}
	transfer := &TransferFunc{Num: [][][]float64{{numerator}}, Den: [][]float64{coefficients}, Dt: dt}
	result, err := transfer.StateSpace(nil)
	if err != nil {
		return nil, err
	}
	channel := result.Sys
	if whole > 0 {
		channel.InputDelay = []float64{float64(whole)}
	}
	return channel, nil
}

func conversionThiranBank(delays []float64, dt float64, maxOrder int, modeling C2DDelayModeling) (*System, error) {
	channels := make([]*System, len(delays))
	for i, tau := range delays {
		channel, err := conversionThiranChannel(tau/dt, dt, maxOrder)
		if err != nil {
			return nil, err
		}
		channels[i] = channel
	}
	bank, err := BlkDiag(channels...)
	if err != nil {
		return nil, err
	}
	if modeling == C2DDelayModelingInternal {
		integer := bank.InputDelay
		bank = conversionStateAsDelay(bank)
		bank.InputDelay = integer
	}
	return bank, nil
}

func discretizeInternalThiran(sys *System, dt float64, opts C2DOptions) (*System, error) {
	_, m, p := sys.Dims()
	for _, tau := range sys.LFT.Tau {
		if tau <= 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
			return nil, ErrZeroInternalDelay
		}
	}
	rational, err := conversionAugmentedRational(sys).discretizeTustin(dt, opts.PrewarpFrequency)
	if err != nil {
		return nil, err
	}
	bank, err := conversionThiranBank(sys.LFT.Tau, dt, opts.ThiranOrder, opts.DelayModeling)
	if err != nil {
		return nil, err
	}
	out, err := LFT(rational, bank, m, p)
	if err != nil {
		return nil, err
	}
	if sys.Delay != nil {
		out.Delay, err = convertDelayToDiscrete(sys.Delay, dt)
		if err != nil {
			return nil, err
		}
	}
	if err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	if n, _, _ := out.Dims(); len(out.StateName) != n {
		out.StateName = nil
	}
	return out, nil
}
