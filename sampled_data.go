package controlsys

import "plantcontrol.org/v1/gonum/mat"

func validateSampledIO(context string, input, output *mat.Dense, dt float64) (m, p, n int, err error) {
	in, out, err := validateSampledSignalPair(context, input, output, dt)
	if err != nil {
		return 0, 0, 0, err
	}
	return in.channels, out.channels, in.samples, nil
}

func validateMarkovSequence(markov []*mat.Dense, order int, dt float64) (p, m int, err error) {
	seq, err := validateMarkovSignalSequence(markov, order, dt)
	if err != nil {
		return 0, 0, err
	}
	return seq.p, seq.m, nil
}
