package controlsys

import (
	"errors"
	"fmt"
	"math"
	"strings"

	"plantcontrol.org/v1/gonum/mat"
)

// AlgebraicLoopError describes a singular or numerically singular direct-feedthrough loop.
type AlgebraicLoopError struct {
	// Signals contains the named input and output channels active in the
	// singular feedthrough mode. It is empty when the operation has no names.
	Signals []string
	// Condition is the estimated condition number of the feedthrough equation.
	// An exactly singular equation has an infinite condition number.
	Condition float64

	loop        *mat.Dense
	feedthrough *mat.Dense
	cause       error
}

// Error reports the singular loop, its condition number and any implicated
// signals.
func (e *AlgebraicLoopError) Error() string {
	message := ErrAlgebraicLoop.Error()
	if e.cause != nil {
		message = e.cause.Error()
	} else {
		message = fmt.Sprintf("%s (condition number %g)", message, e.Condition)
	}
	if len(e.Signals) != 0 {
		message += "; implicated signals: " + strings.Join(e.Signals, ", ")
	}
	return message
}

// Unwrap returns the underlying cause, ErrAlgebraicLoop unless the loop was
// reported under another sentinel.
func (e *AlgebraicLoopError) Unwrap() error {
	if e.cause != nil {
		return e.cause
	}
	return ErrAlgebraicLoop
}

func newAlgebraicLoopError(loop, feedthrough *mat.Dense, condition float64) *AlgebraicLoopError {
	return &AlgebraicLoopError{
		Condition:   condition,
		loop:        loop,
		feedthrough: mat.DenseCopyOf(feedthrough),
	}
}

func withAlgebraicLoopSignals(err error, inputNames, outputNames []string) error {
	var diagnostic *AlgebraicLoopError
	if !errors.As(err, &diagnostic) {
		return err
	}
	signals := diagnostic.signalNames(inputNames, outputNames)
	if len(signals) == 0 {
		return err
	}
	enriched := *diagnostic
	enriched.Signals = signals
	enriched.loop = nil
	enriched.feedthrough = nil
	enriched.cause = err
	return &enriched
}

func (e *AlgebraicLoopError) signalNames(inputNames, outputNames []string) []string {
	if e.loop == nil || e.feedthrough == nil {
		return nil
	}
	n, c := e.loop.Dims()
	p, m := e.feedthrough.Dims()
	if n == 0 || c != n || m != n || len(inputNames) != n || len(outputNames) != p {
		return nil
	}

	var svd mat.SVD
	if !svd.Factorize(e.loop, mat.SVDFull) {
		return nil
	}
	values := svd.Values(nil)
	if len(values) == 0 {
		return nil
	}
	var rightVectors mat.Dense
	svd.VTo(&rightVectors)

	cutoff := values[0] * eps() * float64(n)
	firstMode := len(values)
	for firstMode > 0 && values[firstMode-1] <= cutoff {
		firstMode--
	}
	if firstMode == len(values) {
		firstMode--
	}

	activeInputs := make([]bool, n)
	activeOutputs := make([]bool, p)
	mode := make([]float64, n)
	response := make([]float64, p)
	for k := firstMode; k < len(values); k++ {
		for i := range n {
			mode[i] = rightVectors.At(i, k)
		}
		markActive(activeInputs, mode)

		raw := e.feedthrough.RawMatrix()
		for i := range p {
			sum := 0.0
			for j := range n {
				sum += raw.Data[i*raw.Stride+j] * mode[j]
			}
			response[i] = sum
		}
		markActive(activeOutputs, response)
	}

	signals := make([]string, 0, n+p)
	seen := make(map[string]struct{}, n+p)
	appendActive := func(names []string, active []bool) {
		for i, name := range names {
			if !active[i] || name == "" {
				continue
			}
			if _, ok := seen[name]; ok {
				continue
			}
			seen[name] = struct{}{}
			signals = append(signals, name)
		}
	}
	appendActive(inputNames, activeInputs)
	appendActive(outputNames, activeOutputs)
	return signals
}

func markActive(active []bool, values []float64) {
	scale := 0.0
	for _, value := range values {
		scale = math.Max(scale, math.Abs(value))
	}
	tolerance := scale * eps() * float64(len(values))
	for i, value := range values {
		if math.Abs(value) > tolerance {
			active[i] = true
		}
	}
}
