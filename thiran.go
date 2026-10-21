package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// ThiranDelay returns the discrete-time allpass Thiran filter approximating
// a delay of tau seconds at sample time dt, as MATLAB thiran(tau, Ts). With
// D = tau/dt samples, an integer D gives the exact delay z^-D (a unit gain
// for D = 0); otherwise the filter order is N = ceil(D), which keeps the
// filter stable. The system is SISO. A negative or non-finite tau returns
// ErrNegativeDelay, a non-positive or non-finite dt ErrInvalidSampleTime, and
// a filter needing more than MaxAbsorbedDelayStates states ErrInvalidArgument,
// and non-finite coefficients ErrOverflow.
// See https://www.mathworks.com/help/control/ref/thiran.html.
func ThiranDelay(tau, dt float64) (*System, error) {
	if err := validateThiranArgs(tau, dt); err != nil {
		return nil, fmt.Errorf("ThiranDelay: %w", err)
	}
	D := tau / dt
	if isIntegerSampleDelay(D) {
		sys, err := integerDelaySS(D, dt)
		if err != nil {
			return nil, fmt.Errorf("ThiranDelay: %w", err)
		}
		return sys, nil
	}
	if err := checkAbsorbedStates(math.Ceil(D)); err != nil {
		return nil, fmt.Errorf("ThiranDelay: %w", err)
	}
	sys, err := thiranFilter(D, int(math.Ceil(D)), dt)
	if err != nil {
		return nil, fmt.Errorf("ThiranDelay: %w", err)
	}
	return sys, nil
}

func validateThiranArgs(tau, dt float64) error {
	if tau < 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
		return fmt.Errorf("tau is %g: %w", tau, ErrNegativeDelay)
	}
	if !(dt > 0) || math.IsInf(dt, 0) {
		return fmt.Errorf("dt is %g: %w", dt, ErrInvalidSampleTime)
	}
	return nil
}

// thiranFilter realizes the order-N Thiran allpass filter for D samples.
func thiranFilter(D float64, N int, dt float64) (*System, error) {
	a := thiranCoeffs(D, N)
	for _, v := range a {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return nil, fmt.Errorf("thiran coefficients overflow: %w", ErrOverflow)
		}
	}

	// Transfer function in descending powers of z:
	// den = [1, a_1, a_2, ..., a_N]
	// num = [a_N, a_{N-1}, ..., a_1, 1]  (reversed, allpass)
	den := make([]float64, N+1)
	num := make([]float64, N+1)
	for k := 0; k <= N; k++ {
		den[k] = a[k]
		num[k] = a[N-k]
	}

	tf := &TransferFunc{
		Num: [][][]float64{{num}},
		Den: [][]float64{den},
		Dt:  dt,
	}

	result, err := tf.stateSpace()
	if err != nil {
		return nil, err
	}
	return result.Sys, nil
}

// thiranCoeffs computes Thiran allpass filter coefficients
// a_k = (-1)^k * C(N,k) * prod_{n=0}^{N} (D-N+n)/(D-N+k+n) for k=0..N
// by the ratio a_k/a_{k-1} = -(N-k+1)/k * (D-N+k-1)/(D+k), which avoids
// overflowing C(N,k) for large N.
func thiranCoeffs(D float64, N int) []float64 {
	a := make([]float64, N+1)
	a[0] = 1
	fN := float64(N)
	for k := 1; k <= N; k++ {
		fk := float64(k)
		a[k] = -a[k-1] * ((fN - fk + 1) / fk) * ((D - fN + fk - 1) / (D + fk))
	}
	return a
}

// integerDelaySS returns a discrete state-space system implementing z^{-d}
// for d rounded to whole samples, rejecting d beyond MaxAbsorbedDelayStates
// or non-finite with ErrInvalidArgument.
func integerDelaySS(samples, dt float64) (*System, error) {
	if err := checkAbsorbedStates(math.Round(samples)); err != nil {
		return nil, err
	}
	d := int(math.Round(samples))
	if d <= 0 {
		return NewGain(mat.NewDense(1, 1, []float64{1}), dt)
	}
	A := mat.NewDense(d, d, nil)
	for i := 1; i < d; i++ {
		A.Set(i, i-1, 1)
	}
	B := mat.NewDense(d, 1, nil)
	B.Set(0, 0, 1)
	C := mat.NewDense(1, d, nil)
	C.Set(0, d-1, 1)
	D := mat.NewDense(1, 1, nil)
	return newNoCopy(A, B, C, D, dt)
}
