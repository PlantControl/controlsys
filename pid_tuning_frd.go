package controlsys

import (
	"context"
	"fmt"
	"math"
	"sort"
)

// TunePIDFRD tunes directly against finite measured SISO frequency data. It
// interpolates complex responses in log frequency and never extrapolates or
// invents a time-domain realization. Pole knowledge is recorded, not treated as
// sufficient evidence for global stability outside the measured frequency band.
func TunePIDFRD(ctx context.Context, frd *FRD, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("TunePIDFRD: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if frd == nil || len(frd.Omega) < 3 || len(frd.Omega) > 10000 || len(frd.Response) != len(frd.Omega) {
		return nil, fmt.Errorf("TunePIDFRD: require 3..10000 matched frequency samples")
	}
	if !finitePID(frd.Dt) || frd.Dt < 0 {
		return nil, fmt.Errorf("TunePIDFRD: invalid sample time")
	}
	for k, w := range frd.Omega {
		if !finitePID(w) || w <= 0 || k > 0 && w <= frd.Omega[k-1] {
			return nil, fmt.Errorf("TunePIDFRD: frequencies must be finite, positive, strictly increasing")
		}
		if frd.Dt > 0 && w >= math.Pi/frd.Dt {
			return nil, fmt.Errorf("TunePIDFRD: samples must be below Nyquist")
		}
		if len(frd.Response[k]) != 1 || len(frd.Response[k][0]) != 1 || !pidFiniteComplex(frd.Response[k][0][0]) {
			return nil, fmt.Errorf("TunePIDFRD: responses must be finite SISO values")
		}
	}
	if o.UnstablePoles != nil && (*o.UnstablePoles < 0 || *o.UnstablePoles > 256) {
		return nil, fmt.Errorf("TunePIDFRD: unstable pole count must be 0..256 or unknown")
	}
	// Own the sampled snapshot while evaluating; callers may reuse their source.
	data := frd.Copy()
	low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
	if o.CrossoverFrequency == 0 {
		o.CrossoverFrequency = math.Exp((math.Log(low) + math.Log(high)) / 2)
	}
	warnings := []string{"Finite measured frequency coverage cannot certify global closed-loop stability or provide time responses."}
	if o.UnstablePoles == nil {
		warnings = append(warnings, "Unstable open-loop pole count is unknown.")
	} else {
		warnings = append(warnings, fmt.Sprintf("User-supplied unstable open-loop pole count: %d; unmeasured endpoint behavior remains unknown.", *o.UnstablePoles))
	}
	for k := 1; k < len(data.Omega); k++ {
		if data.Omega[k]/data.Omega[k-1] > 1.2 {
			warnings = append(warnings, "Sparse frequency coverage may hide resonances and phase rotations; inspect measured resolution.")
			break
		}
	}
	p := pidTuningPlant{dt: data.Dt, low: low, high: high, stability: "sampled-frequency-only", warnings: warnings, at: func(w float64) complex128 {
		// Core-generated endpoints may differ by a few ulps after exp(log(endpoint)).
		if w <= low {
			return data.Response[0][0][0]
		}
		if w >= high {
			return data.Response[len(data.Omega)-1][0][0]
		}
		k := sort.SearchFloat64s(data.Omega, w)
		if data.Omega[k] == w {
			return data.Response[k][0][0]
		}
		alpha := math.Log(w/data.Omega[k-1]) / math.Log(data.Omega[k]/data.Omega[k-1])
		return complex(1-alpha, 0)*data.Response[k-1][0][0] + complex(alpha, 0)*data.Response[k][0][0]
	}}
	return tunePID(ctx, p, family, o)
}
