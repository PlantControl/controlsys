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
		if err != nil {
			return nil, err
		}
		if samples > 0 {
			gain.InputDelay = []float64{math.Round(samples)}
		}
		return gain, nil
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
	result, err := transfer.stateSpace()
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
	states := 0
	var whole []float64
	for i, tau := range delays {
		channel, err := conversionThiranChannel(tau/dt, dt, maxOrder)
		if err != nil {
			return nil, err
		}
		channels[i] = channel
		n, _, _ := channel.Dims()
		states += n
		if len(channel.InputDelay) > 0 && channel.InputDelay[0] != 0 {
			if whole == nil {
				whole = make([]float64, len(delays))
			}
			whole[i] = channel.InputDelay[0]
		}
	}
	r := len(delays)
	bank := &System{A: newDense(states, states), B: newDense(states, r), C: newDense(r, states), D: newDense(r, r), Dt: dt, InputDelay: whole}
	offset := 0
	for i, channel := range channels {
		n, _, _ := channel.Dims()
		if n > 0 {
			setBlock(bank.A, offset, offset, channel.A)
			setBlock(bank.B, offset, i, channel.B)
			setBlock(bank.C, i, offset, channel.C)
		}
		bank.D.Set(i, i, channel.D.At(0, 0))
		offset += n
	}
	if modeling == C2DDelayModelingInternal && states > 0 {
		bank.LFT = &LFTDelay{Tau: make([]float64, states), B2: newDense(0, states), C2: newDense(states, 0), D12: bank.C, D21: bank.B, D22: bank.A}
		for i := range bank.LFT.Tau {
			bank.LFT.Tau[i] = 1
		}
		bank.A, bank.B, bank.C = newDense(0, 0), newDense(0, r), newDense(r, 0)
	}
	return bank, nil
}

// conversionSeries returns s2*s1 for systems whose only external delays are
// s1.InputDelay and s2.OutputDelay. The bank side is diagonal, so a path
// delay matrix on either operand passes through unchanged.
func conversionSeries(s1, s2 *System) *System {
	n1, m, _ := s1.Dims()
	n2, _, p := s2.Dims()
	q1, q2 := s1.internalDelayCount(), s2.internalDelayCount()
	n, q := n1+n2, q1+q2
	out := &System{A: newDense(n, n), B: newDense(n, m), C: newDense(p, n), D: newDense(p, m), Dt: s1.Dt}
	out.InputDelay, out.OutputDelay = s1.InputDelay, s2.OutputDelay
	if s1.Delay != nil {
		out.Delay = s1.Delay
	} else {
		out.Delay = s2.Delay
	}
	setBlock(out.A, 0, 0, s1.A)
	setBlock(out.A, n1, n1, s2.A)
	mulBlock(out.A, n1, 0, s2.B, s1.C)
	setBlock(out.B, 0, 0, s1.B)
	mulBlock(out.B, n1, 0, s2.B, s1.D)
	mulBlock(out.C, 0, 0, s2.D, s1.C)
	setBlock(out.C, 0, n1, s2.C)
	out.D.Mul(s2.D, s1.D)
	if q == 0 {
		return out
	}
	lft := &LFTDelay{Tau: make([]float64, 0, q), B2: newDense(n, q), C2: newDense(q, n), D12: newDense(p, q), D21: newDense(q, m), D22: newDense(q, q)}
	if q1 > 0 {
		lft.Tau = append(lft.Tau, s1.LFT.Tau...)
		setBlock(lft.B2, 0, 0, s1.LFT.B2)
		mulBlock(lft.B2, n1, 0, s2.B, s1.LFT.D12)
		setBlock(lft.C2, 0, 0, s1.LFT.C2)
		mulBlock(lft.D12, 0, 0, s2.D, s1.LFT.D12)
		setBlock(lft.D21, 0, 0, s1.LFT.D21)
		setBlock(lft.D22, 0, 0, s1.LFT.D22)
	}
	if q2 > 0 {
		lft.Tau = append(lft.Tau, s2.LFT.Tau...)
		setBlock(lft.B2, n1, q1, s2.LFT.B2)
		setBlock(lft.C2, q1, n1, s2.LFT.C2)
		mulBlock(lft.C2, q1, 0, s2.LFT.D21, s1.C)
		setBlock(lft.D12, 0, q1, s2.LFT.D12)
		mulBlock(lft.D21, q1, 0, s2.LFT.D21, s1.D)
		setBlock(lft.D22, q1, q1, s2.LFT.D22)
		if q1 > 0 {
			mulBlock(lft.D22, q1, 0, s2.LFT.D21, s1.LFT.D12)
		}
	}
	out.LFT = lft
	return out
}

func mulBlock(dst *mat.Dense, r0, c0 int, a, b *mat.Dense) {
	ar, ac := a.Dims()
	_, bc := b.Dims()
	if ar == 0 || bc == 0 || ac == 0 {
		return
	}
	dst.Slice(r0, r0+ar, c0, c0+bc).(*mat.Dense).Mul(a, b)
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
