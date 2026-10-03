package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// ThiranDelay returns a discrete-time allpass state-space system approximating
// a fractional delay of tau seconds at sample time dt with a fixed filter order
// N (1-10). The system is SISO.
//
// A delay of D = tau/dt samples is stable only for D > N-1; smaller delays
// return ErrFractionalDelay. Near that bound the poles approach -1 and the
// filter is poorly conditioned. Integer D returns the exact delay z^-D with D
// states for any N. For D > N+0.5 the excess whole samples are realized as a
// leading shift register, so the system has round(D) states.
func ThiranDelay(tau float64, order int, dt float64) (*System, error) {
	if tau < 0 || math.IsNaN(tau) || math.IsInf(tau, 0) {
		return nil, ErrNegativeDelay
	}
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, ErrInvalidSampleTime
	}
	if order < 1 || order > 10 {
		return nil, fmt.Errorf("ThiranDelay: order must be 1-10: %w", ErrDimensionMismatch)
	}

	D := tau / dt
	N := order

	if isIntegerSampleDelay(D) {
		return integerDelaySS(int(math.Round(D)), dt)
	}
	if D <= float64(N-1) {
		return nil, fmt.Errorf("ThiranDelay: delay %.4f samples <= order-1 = %d (unstable): %w",
			D, N-1, ErrFractionalDelay)
	}

	// Split into integer shift + fractional Thiran when D >> N
	intPart := 0
	Deff := D
	if D > float64(N)+0.5 {
		intPart = int(math.Round(D)) - N
		Deff = D - float64(intPart)
	}

	a := thiranCoeffs(Deff, N)

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

	result, err := tf.StateSpace(nil)
	if err != nil {
		return nil, fmt.Errorf("ThiranDelay: %w", err)
	}

	if intPart > 0 {
		intSys, err := integerDelaySS(intPart, dt)
		if err != nil {
			return nil, err
		}
		return Series(intSys, result.Sys)
	}

	return result.Sys, nil
}

// thiranCoeffs computes Thiran allpass filter coefficients.
// a_k = (-1)^k * C(N,k) * prod_{n=0}^{N} (D-N+n)/(D-N+k+n) for k=0..N
func thiranCoeffs(D float64, N int) []float64 {
	a := make([]float64, N+1)
	for k := 0; k <= N; k++ {
		sign := 1.0
		if k%2 != 0 {
			sign = -1.0
		}
		binom := binomial(N, k)
		prod := 1.0
		for n := 0; n <= N; n++ {
			prod *= (D - float64(N) + float64(n)) / (D - float64(N) + float64(k) + float64(n))
		}
		a[k] = sign * binom * prod
	}
	return a
}

func binomial(n, k int) float64 {
	if k < 0 || k > n {
		return 0
	}
	if k == 0 || k == n {
		return 1
	}
	if k > n-k {
		k = n - k
	}
	result := 1.0
	for i := 0; i < k; i++ {
		result *= float64(n-i) / float64(i+1)
	}
	return result
}

// integerDelaySS returns a discrete state-space system implementing z^{-d}.
func integerDelaySS(d int, dt float64) (*System, error) {
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
