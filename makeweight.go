package controlsys

import (
	"fmt"
	"math"
)

// Makeweight returns a weighting function with a monotonic gain profile, as
// MATLAB makeweight(dcgain,[freq,mag],hfgain,Ts,N); see
// https://www.mathworks.com/help/robust/ref/makeweight.html.
//
// freqMag is [freq, mag], the gain passing through mag at freq rad/s, or
// [wc], MATLAB's makeweight(dcgain,wc,hfgain) form, meaning [wc, 1]. Gains
// are absolute, not dB, and must satisfy |dcgain| < mag < |hfgain| or
// |dcgain| > mag > |hfgain|; hfgain = 0 (or dcgain = 0) gives roll-off
// without leveling. Ts = 0 gives a continuous weight with W(0) = dcgain,
// W(∞) = hfgain and |W(j·freq)| = mag; Ts > 0 a discrete one with W(1) =
// dcgain, W(−1) = hfgain and |W(e^(j·freq·Ts))| = mag, requiring
// freq·Ts < π. MATLAB's unspecified sample time Ts = −1 has no counterpart
// here and returns ErrInvalidSampleTime.
//
// N is the order (MATLAB's default is 1). N = 1 is MATLAB's
// (hfgain·s + dcgain·p)/(s + p) with p = freq·√((mag²−hfgain²)/(dcgain²−mag²)),
// keeping the gains' signs, so gains of opposite sign give a
// right-half-plane zero as in MATLAB. For N > 1 the poles and zeros lie in
// the Butterworth pattern the MathWorks page describes: poles of radius w0
// and zeros of radius w0·(|dcgain|/|hfgain|)^(1/N), so that
// |W(jω)|² = (dcgain² + hfgain²·(ω/w0)^(2N)) / (1 + (ω/w0)^(2N)), with w0
// chosen to pass through mag at freq; the page gives no closed form, so this
// realization is inferred from that description. N > 1 with gains of
// opposite sign returns ErrOptionUnsupported. A discrete weight is the
// continuous design mapped by the Tustin transform prewarped at freq, which
// for N = 1 reproduces MATLAB's discrete formula.
//
// Non-finite arguments, freq or mag not positive, gains out of order,
// freq·Ts ≥ π and N < 1 return ErrInvalidArgument.
func Makeweight(dcgain float64, freqMag []float64, hfgain, Ts float64, N int) (*System, error) {
	const op = "Makeweight"
	if err := requireFinite(op, "dcgain", dcgain); err != nil {
		return nil, err
	}
	if err := requireFinite(op, "hfgain", hfgain); err != nil {
		return nil, err
	}
	if err := requireFinite(op, "[freq,mag]", freqMag...); err != nil {
		return nil, err
	}
	if err := requireFinite(op, "Ts", Ts); err != nil {
		return nil, err
	}
	var freq, mag float64
	switch len(freqMag) {
	case 1:
		freq, mag = freqMag[0], 1
	case 2:
		freq, mag = freqMag[0], freqMag[1]
	default:
		return nil, fmt.Errorf("%s: [freq,mag] has %d elements, want 1 or 2: %w", op, len(freqMag), ErrInvalidArgument)
	}
	if freq <= 0 {
		return nil, fmt.Errorf("%s: freq = %g, want > 0: %w", op, freq, ErrInvalidArgument)
	}
	if mag <= 0 {
		return nil, fmt.Errorf("%s: mag = %g, want > 0: %w", op, mag, ErrInvalidArgument)
	}
	dc, hf := math.Abs(dcgain), math.Abs(hfgain)
	if !(dc < mag && mag < hf) && !(dc > mag && mag > hf) {
		return nil, fmt.Errorf("%s: mag = %g must lie strictly between |dcgain| = %g and |hfgain| = %g: %w", op, mag, dc, hf, ErrInvalidArgument)
	}
	if Ts < 0 {
		return nil, fmt.Errorf("%s: Ts = %g: %w", op, Ts, ErrInvalidSampleTime)
	}
	if Ts > 0 && freq*Ts >= math.Pi {
		return nil, fmt.Errorf("%s: freq·Ts = %g, want < π: %w", op, freq*Ts, ErrInvalidArgument)
	}
	if N < 1 {
		return nil, fmt.Errorf("%s: N = %d, want ≥ 1: %w", op, N, ErrInvalidArgument)
	}
	if N > 1 && dcgain*hfgain < 0 {
		return nil, fmt.Errorf("%s: N = %d with dcgain and hfgain of opposite sign: %w", op, N, ErrOptionUnsupported)
	}

	w0 := freq * math.Pow((mag*mag-hf*hf)/(dc*dc-mag*mag), 1/(2*float64(N)))
	num, den := makeweightContinuous(dcgain, hfgain, w0, N)
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}
	ssr, err := tf.stateSpace()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	W := ssr.Sys
	if Ts > 0 {
		W, err = W.c2d(Ts, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: freq})
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
	}
	return W, nil
}

// makeweightContinuous returns the descending numerator and monic
// denominator of the order-N Butterworth-pattern weight with pole radius w0.
// The numerator coefficient of s^k is a_k·|dc|^((N−k)/N)·|hf|^(k/N)·w0^(N−k)
// and the denominator's a_k·w0^(N−k), a_k being the normalized Butterworth
// coefficients; N = 1 gives (hf·s + dc·w0)/(s + w0) with signs kept.
func makeweightContinuous(dcgain, hfgain, w0 float64, N int) (num, den []float64) {
	if N == 1 {
		return []float64{hfgain, dcgain * w0}, []float64{1, w0}
	}
	a := butterworthCoefficients(N)
	dc, hf := math.Abs(dcgain), math.Abs(hfgain)
	sign := 1.0
	if dcgain < 0 || hfgain < 0 {
		sign = -1
	}
	num = make([]float64, N+1)
	den = make([]float64, N+1)
	for k := 0; k <= N; k++ {
		scale := math.Pow(w0, float64(N-k))
		num[N-k] = sign * a[k] * math.Pow(dc, float64(N-k)/float64(N)) * math.Pow(hf, float64(k)/float64(N)) * scale
		den[N-k] = a[k] * scale
	}
	return num, den
}

// butterworthCoefficients returns a_0..a_N of the normalized Butterworth
// polynomial Σ a_k s^k (a_0 = a_N = 1), by the product recurrence
// a_(k+1) = a_k·cos(kγ)/sin((k+1)γ), γ = π/(2N).
func butterworthCoefficients(N int) []float64 {
	a := make([]float64, N+1)
	a[0] = 1
	g := math.Pi / (2 * float64(N))
	for k := range N {
		a[k+1] = a[k] * math.Cos(float64(k)*g) / math.Sin(float64(k+1)*g)
	}
	a[N] = 1
	return a
}
