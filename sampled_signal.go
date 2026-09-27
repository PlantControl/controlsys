package controlsys

import (
	"fmt"

	"plantcontrol.org/v1/gonum/mat"
)

type sampledSignalOrientation int

const (
	sampledChannelsBySamples sampledSignalOrientation = iota
	sampledSamplesByChannels
)

type sampledSignal struct {
	context     string
	data        *mat.Dense
	channels    int
	samples     int
	orientation sampledSignalOrientation
}

func newSampledSignal(context string, data *mat.Dense, channels, samples int, orientation sampledSignalOrientation) sampledSignal {
	return sampledSignal{context: context, data: data, channels: channels, samples: samples, orientation: orientation}
}

func validateSampledSignal(context string, data *mat.Dense, channels, samples int, orientation sampledSignalOrientation) (sampledSignal, error) {
	if data == nil {
		return sampledSignal{}, ErrInsufficientData
	}
	rows, cols := data.Dims()
	wantRows, wantCols := channels, samples
	if orientation == sampledSamplesByChannels {
		wantRows, wantCols = samples, channels
	}
	if rows != wantRows || cols != wantCols {
		return sampledSignal{}, fmt.Errorf("%s: sampled signal is %dx%d, want %dx%d: %w",
			context, rows, cols, wantRows, wantCols, ErrDimensionMismatch)
	}
	if samples == 0 || (channels == 0 && orientation == sampledChannelsBySamples) {
		return sampledSignal{}, ErrInsufficientData
	}
	return newSampledSignal(context, data, channels, samples, orientation), nil
}

func validateSampledSignalPair(context string, input, output *mat.Dense, dt float64) (sampledSignal, sampledSignal, error) {
	if dt <= 0 {
		return sampledSignal{}, sampledSignal{}, ErrInvalidSampleTime
	}
	if input == nil || output == nil {
		return sampledSignal{}, sampledSignal{}, ErrInsufficientData
	}
	m, nIn := input.Dims()
	p, nOut := output.Dims()
	if nIn != nOut {
		return sampledSignal{}, sampledSignal{}, fmt.Errorf("%s: input has %d samples, output has %d: %w", context, nIn, nOut, ErrDimensionMismatch)
	}
	in, err := validateSampledSignal(context, input, m, nIn, sampledChannelsBySamples)
	if err != nil {
		return sampledSignal{}, sampledSignal{}, err
	}
	out, err := validateSampledSignal(context, output, p, nOut, sampledChannelsBySamples)
	if err != nil {
		return sampledSignal{}, sampledSignal{}, err
	}
	return in, out, nil
}

func validateLsimInputSignal(context string, u *mat.Dense, steps, inputs int) (sampledSignal, error) {
	return validateSampledSignal(context, u, inputs, steps, sampledSamplesByChannels)
}

func (s sampledSignal) channelsBySamplesDense() *mat.Dense {
	if s.orientation == sampledChannelsBySamples {
		return s.data
	}
	if s.channels == 0 {
		return mat.NewDense(1, s.samples, nil).Slice(0, 0, 0, s.samples).(*mat.Dense)
	}
	uSim := mat.NewDense(s.channels, s.samples, nil)
	src := s.data.RawMatrix()
	dst := uSim.RawMatrix()
	for k := 0; k < s.samples; k++ {
		for ch := 0; ch < s.channels; ch++ {
			dst.Data[ch*dst.Stride+k] = src.Data[k*src.Stride+ch]
		}
	}
	return uSim
}

type markovSequence struct {
	terms []*mat.Dense
	p     int
	m     int
	order int
	dt    float64
}

func validateMarkovSignalSequence(markov []*mat.Dense, order int, dt float64) (markovSequence, error) {
	if len(markov) == 0 {
		return markovSequence{}, fmt.Errorf("ERA: empty markov sequence: %w", ErrInsufficientData)
	}
	if order <= 0 {
		return markovSequence{}, fmt.Errorf("ERA: order must be positive: %w", ErrInvalidOrder)
	}
	if dt <= 0 {
		return markovSequence{}, fmt.Errorf("ERA: %w", ErrInvalidSampleTime)
	}
	if markov[0] == nil {
		return markovSequence{}, fmt.Errorf("ERA: markov[0] is nil: %w", ErrInsufficientData)
	}

	p, m := markov[0].Dims()
	if p == 0 || m == 0 {
		return markovSequence{}, fmt.Errorf("ERA: empty markov[0]: %w", ErrInsufficientData)
	}
	for i := 1; i < len(markov); i++ {
		if markov[i] == nil {
			return markovSequence{}, fmt.Errorf("ERA: markov[%d] is nil: %w", i, ErrInsufficientData)
		}
		ri, ci := markov[i].Dims()
		if ri != p || ci != m {
			return markovSequence{}, fmt.Errorf("ERA: markov[%d] is %dx%d, expected %dx%d: %w",
				i, ri, ci, p, m, ErrDimensionMismatch)
		}
	}

	minLen := 2*order + 1
	if len(markov) < minLen {
		return markovSequence{}, fmt.Errorf("ERA: need >= %d markov params for order %d, got %d: %w",
			minLen, order, len(markov), ErrInsufficientData)
	}
	return markovSequence{terms: markov, p: p, m: m, order: order, dt: dt}, nil
}
