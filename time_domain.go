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

// frequencyPoint is a frequency variable s (z for discrete models) in the
// resolvent form (sE − A)⁻¹ = −q·(pE + qA)⁻¹, with p and q exact. Unless
// the point is plain, q = −conj(p) with |p| = 1 to rounding.
type frequencyPoint struct {
	p, q complex128
}

// pointAt is the arbitrary complex point s: p = s, q = −1.
func pointAt(s complex128) frequencyPoint {
	return frequencyPoint{p: s, q: -1}
}

// value is s rounded, for delay factors and pole distances: −p/q = p/conj(p)
// = p²/|p|² ≈ p² off the plain point.
func (pt frequencyPoint) value() complex128 {
	if pt.isPlain() {
		return pt.p
	}
	pr, pi := real(pt.p), imag(pt.p)
	return complex(math.FMA(pr, pr, -pi*pi), 2*pr*pi)
}

// isPlain reports whether pt is solved as sE − A. A unit-circle point
// with q = −1 is z = 1 = s.
func (pt frequencyPoint) isPlain() bool {
	return pt.q == -1
}

// scale is the factor c = −q in (sE − A)⁻¹ = c·(pE + qA)⁻¹.
func (pt frequencyPoint) scale() complex128 { return -pt.q }

// shift is the diagonal entry s − v of sI − A. For a plain point it is
// rounded from the rounded s; otherwise it is (p + q·v)·(−1/q), with
// −1/q = p/|p|² ≈ p, whose rounding is a few ε relative to the exact s − v,
// so it stays accurate next to a pole, where s − v is small and s itself is
// not representable.
func (pt frequencyPoint) shift(v float64) complex128 {
	if pt.isPlain() {
		return pt.p - complex(v, 0)
	}
	return complex(math.FMA(real(pt.q), v, real(pt.p)), math.FMA(imag(pt.q), v, imag(pt.p))) * pt.p
}

// frequencyPoint is the point at frequency w. A discrete z = e^{jωT} rounded
// to complex128 lies up to ε off the unit circle, which moves the response by
// about ε/d relative at distance d from a lightly damped pole. Instead, with
// p = cos(ωT/2) + j·sin(ωT/2) as rounded, q = −conj(p) and z = −p/q =
// p/conj(p) lies exactly on the circle whatever the rounding, at a frequency
// off ω by about ε relative; p and q enter the pencil exactly. This is
// z = (1+jν)/(1−jν), ν = tan(ωT/2), scaled by cos(ωT/2).
func (td timeDomain) frequencyPoint(w float64) frequencyPoint {
	if td.isContinuous() {
		return pointAt(complex(0, w))
	}
	sn, cs := math.Sincos(w * td.dt / 2)
	return frequencyPoint{p: complex(cs, sn), q: complex(-cs, sn)}
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
