package controlsys

import (
	"fmt"
	"math"

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
		return sampledSignal{}, fmt.Errorf("%s: sampled signal is nil: %w", context, ErrInvalidArgument)
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
		return sampledSignal{}, fmt.Errorf("%s: sampled signal has %d channels and %d samples: %w", context, channels, samples, ErrInsufficientData)
	}
	if !data.IsEmpty() {
		if err := requireFiniteDense(context, "sampled signal", data); err != nil {
			return sampledSignal{}, err
		}
	}
	return newSampledSignal(context, data, channels, samples, orientation), nil
}

func validateSampledSignalPair(context string, input, output *mat.Dense, dt float64) (sampledSignal, sampledSignal, error) {
	if err := requireDiscreteSampleTime(context, dt); err != nil {
		return sampledSignal{}, sampledSignal{}, err
	}
	if input == nil || output == nil {
		return sampledSignal{}, sampledSignal{}, fmt.Errorf("%s: input or output is nil: %w", context, ErrInvalidArgument)
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

// requireDiscreteSampleTime rejects a sample time that is not positive and
// finite.
func requireDiscreteSampleTime(context string, dt float64) error {
	if !(dt > 0) || math.IsInf(dt, 0) {
		return fmt.Errorf("%s: dt is %g: %w", context, dt, ErrInvalidSampleTime)
	}
	return nil
}

func validateMarkovSignalSequence(context string, markov []*mat.Dense, order int, dt float64) (markovSequence, error) {
	if len(markov) == 0 {
		return markovSequence{}, fmt.Errorf("%s: empty markov sequence: %w", context, ErrInsufficientData)
	}
	if order <= 0 {
		return markovSequence{}, fmt.Errorf("%s: order %d must be positive: %w", context, order, ErrInvalidOrder)
	}
	if err := requireDiscreteSampleTime(context, dt); err != nil {
		return markovSequence{}, err
	}
	if markov[0] == nil {
		return markovSequence{}, fmt.Errorf("%s: markov[0] is nil: %w", context, ErrInvalidArgument)
	}

	p, m := markov[0].Dims()
	if p == 0 || m == 0 {
		return markovSequence{}, fmt.Errorf("%s: empty markov[0]: %w", context, ErrInsufficientData)
	}
	for i := range markov {
		if markov[i] == nil {
			return markovSequence{}, fmt.Errorf("%s: markov[%d] is nil: %w", context, i, ErrInvalidArgument)
		}
		ri, ci := markov[i].Dims()
		if ri != p || ci != m {
			return markovSequence{}, fmt.Errorf("%s: markov[%d] is %dx%d, expected %dx%d: %w",
				context, i, ri, ci, p, m, ErrDimensionMismatch)
		}
		if err := requireFiniteDense(context, fmt.Sprintf("markov[%d]", i), markov[i]); err != nil {
			return markovSequence{}, err
		}
	}

	minLen := 2*order + 1
	if len(markov) < minLen {
		return markovSequence{}, fmt.Errorf("%s: need >= %d markov params for order %d, got %d: %w",
			context, minLen, order, len(markov), ErrInsufficientData)
	}
	return markovSequence{terms: markov, p: p, m: m, order: order, dt: dt}, nil
}
