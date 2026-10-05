package controlsys

import (
	"fmt"
	"math"
)

// GenSig generates a periodic test signal u on the time grid t = 0:ts:tf,
// like MATLAB [u,t] = gensig(type,tau,Tf,Ts), for use with Lsim. sigType is
// "sine" (sin(2πt/tau)), "square" (0 on the first half of each period, 1 on
// the second) or "pulse" (a unit sample at the first grid point at or after
// each positive multiple of tau). All signals have unit amplitude and start
// at 0. tf = 0 selects the default 5·tau and ts = 0 the default tau/64. A
// non-positive or non-finite tau, a negative or non-finite tf or ts, and an
// unknown sigType return ErrInvalidArgument.
// See https://www.mathworks.com/help/control/ref/gensig.html.
func GenSig(sigType string, tau, tf, ts float64) (u, t []float64, err error) {
	if !(tau > 0) || math.IsInf(tau, 1) {
		return nil, nil, fmt.Errorf("GenSig: tau is %g, want positive and finite: %w", tau, ErrInvalidArgument)
	}
	if !(tf >= 0) || math.IsInf(tf, 1) {
		return nil, nil, fmt.Errorf("GenSig: tf is %g, want non-negative and finite: %w", tf, ErrInvalidArgument)
	}
	if !(ts >= 0) || math.IsInf(ts, 1) {
		return nil, nil, fmt.Errorf("GenSig: ts is %g, want non-negative and finite: %w", ts, ErrInvalidArgument)
	}
	if tf == 0 {
		tf = 5 * tau
	}
	if ts == 0 {
		ts = tau / 64
	}
	var at func(k int, tk float64) float64
	switch sigType {
	case "sine":
		at = func(_ int, tk float64) float64 { return math.Sin(2 * math.Pi * tk / tau) }
	case "square":
		at = func(_ int, tk float64) float64 {
			if gensigPhase(tk, tau) >= 0.5-gridTol {
				return 1
			}
			return 0
		}
	case "pulse":
		at = func(k int, tk float64) float64 {
			if k > 0 && math.Floor(tk/tau+gridTol) > math.Floor((tk-ts)/tau+gridTol) {
				return 1
			}
			return 0
		}
	default:
		return nil, nil, fmt.Errorf("GenSig: unknown signal type %q (use sine, square or pulse): %w", sigType, ErrInvalidArgument)
	}

	steps := gridSampleCount(tf, ts)
	t = make([]float64, steps)
	u = make([]float64, steps)
	for k := range t {
		t[k] = float64(k) * ts
		u[k] = at(k, t[k])
	}
	return u, t, nil
}

func gensigPhase(t, tau float64) float64 {
	phase := t / tau
	return phase - math.Floor(phase+gridTol)
}
