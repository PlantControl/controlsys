package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

type timeDomain struct {
	dt float64
}

func newTimeDomain(dt float64) timeDomain {
	return timeDomain{dt: dt}
}

func (td timeDomain) isContinuous() bool {
	return td.dt == 0
}

func (td timeDomain) isDiscrete() bool {
	return td.dt > 0
}

// validateSampleTime accepts dt = 0 (continuous) or a finite positive dt
// (discrete). MATLAB's unspecified Ts = -1 is not supported.
func (td timeDomain) validateSampleTime() error {
	if !(td.dt >= 0) || math.IsInf(td.dt, 1) {
		return fmt.Errorf("sample time %g: %w", td.dt, ErrInvalidSampleTime)
	}
	return nil
}

func (td timeDomain) ensureCompatible(other timeDomain) error {
	if td.dt != other.dt {
		return ErrDomainMismatch
	}
	return nil
}

func (td timeDomain) frequencyVariable(w float64) complex128 {
	if td.isContinuous() {
		return complex(0, w)
	}
	return cmplx.Exp(complex(0, w*td.dt))
}

func (td timeDomain) naturalFrequency(p complex128) float64 {
	if td.isContinuous() {
		return cmplx.Abs(p)
	}
	return cmplx.Abs(cmplx.Log(p)) / td.dt
}

func (td timeDomain) continuousDelay(samples float64) float64 {
	return samples * td.dt
}

func (td timeDomain) discreteDelaySamples(tau float64) (float64, error) {
	samples := tau / td.dt
	rounded := math.Round(samples)
	if math.Abs(samples-rounded) > 1e-9 {
		return 0, fmt.Errorf("delay=%g not integer multiple of dt=%g: %w", tau, td.dt, ErrFractionalDelay)
	}
	return rounded, nil
}

func (td timeDomain) convertDelayMatrixToDiscrete(delay *mat.Dense) (*mat.Dense, error) {
	r, c := delay.Dims()
	out := mat.NewDense(r, c, nil)
	inRaw := delay.RawMatrix()
	outRaw := out.RawMatrix()
	for i := range r {
		for j := range c {
			tau := inRaw.Data[i*inRaw.Stride+j]
			samples, err := td.discreteDelaySamples(tau)
			if err != nil {
				return nil, fmt.Errorf("delay[%d][%d]=%g not integer multiple of dt=%g: %w",
					i, j, tau, td.dt, ErrFractionalDelay)
			}
			outRaw.Data[i*outRaw.Stride+j] = samples
		}
	}
	return out, nil
}

func (td timeDomain) convertDelayMatrixToContinuous(delay *mat.Dense) *mat.Dense {
	r, c := delay.Dims()
	out := mat.NewDense(r, c, nil)
	inRaw := delay.RawMatrix()
	outRaw := out.RawMatrix()
	for i := range r {
		for j := range c {
			outRaw.Data[i*outRaw.Stride+j] = td.continuousDelay(inRaw.Data[i*inRaw.Stride+j])
		}
	}
	return out
}
