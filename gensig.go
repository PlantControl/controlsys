package controlsys

import (
	"fmt"
	"math"
)

func GenSig(sigType string, period, dt float64) (t, u []float64, err error) {
	if !(period > 0) || math.IsInf(period, 1) {
		return nil, nil, fmt.Errorf("GenSig: period must be positive and finite")
	}
	if !(dt > 0) || math.IsInf(dt, 1) {
		return nil, nil, fmt.Errorf("GenSig: dt must be positive and finite")
	}

	steps := gridSampleCount(period, dt)
	t = make([]float64, steps)
	u = make([]float64, steps)

	for k := range t {
		t[k] = float64(k) * dt
	}

	switch sigType {
	case "step":
		for k := range u {
			u[k] = 1
		}
	case "sine":
		for k := range u {
			u[k] = math.Sin(2 * math.Pi * t[k] / period)
		}
	case "square":
		for k := range u {
			phase := t[k] / period
			phase -= math.Floor(phase + gridTol)
			if phase < 0.5-gridTol {
				u[k] = 1
			} else {
				u[k] = -1
			}
		}
	case "pulse":
		u[0] = 1.0 / dt
	default:
		return nil, nil, fmt.Errorf("GenSig: unknown signal type %q (use step, sine, square, pulse)", sigType)
	}

	return t, u, nil
}
