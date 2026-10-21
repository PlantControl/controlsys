package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"sort"
)

// TunePIDFRD tunes directly against finite measured SISO frequency data. It
// interpolates complex responses in log frequency and never extrapolates or
// invents a time-domain realization. Pole knowledge is recorded, not treated as
// sufficient evidence for global stability outside the measured frequency band.
// Errors follow TunePID.
func TunePIDFRD(ctx context.Context, frd *FRD, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("TunePIDFRD: nil context: %w", ErrInvalidArgument)
	}
	if err := ctx.Err(); err != nil {
		return nil, fmt.Errorf("TunePIDFRD: %w", err)
	}
	invalid := func(detail string) error {
		return fmt.Errorf("TunePIDFRD: %s: %w", detail, ErrInvalidArgument)
	}
	if frd == nil || len(frd.Omega) < 3 || len(frd.Omega) > 10000 || len(frd.Response) != len(frd.Omega) {
		return nil, invalid("require 3..10000 matched frequency samples")
	}
	if !finitePID(frd.Dt) || frd.Dt < 0 {
		return nil, invalid("invalid sample time")
	}
	for k, w := range frd.Omega {
		if !finitePID(w) || w <= 0 || k > 0 && w <= frd.Omega[k-1] {
			return nil, invalid("frequencies must be finite, positive, strictly increasing")
		}
		if frd.Dt > 0 && w >= math.Pi/frd.Dt {
			return nil, invalid("samples must be below Nyquist")
		}
		if len(frd.Response[k]) != 1 || len(frd.Response[k][0]) != 1 || !pidFiniteComplex(frd.Response[k][0][0]) {
			return nil, invalid("responses must be finite SISO values")
		}
	}
	if o.UnstablePoles != nil && (*o.UnstablePoles < 0 || *o.UnstablePoles > 256) {
		return nil, invalid("unstable pole count must be 0..256 or unknown")
	}
	// Own the sampled snapshot while evaluating; callers may reuse their source.
	data := frd.Copy()
	if o.CrossoverFrequency == 0 {
		low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
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
	p := pidTuningFRDPlant(data, o.UnstablePoles)
	p.warnings = warnings
	res, err := tunePID(ctx, "TunePIDFRD", p, family, o)
	if err != nil {
		return res, err
	}
	return pidTuningFRDBounded(ctx, p, family, o, res), nil
}

// pidTuningFRDBounded returns res, or, when its sampled |S| peak exceeds
// pidTuningMaxSensitivity on data the Nyquist count certifies (p.dcGain
// nonzero), the design of a search under that bound, first at the target
// band and then with the margin as a floor, as TunePID's. When neither
// finds one within the evaluations left, res is kept with a warning: a
// hard bound turns stable designs into errors. The floor's crossing band
// is the measured one, as the data have no model beyond it.
func pidTuningFRDBounded(ctx context.Context, p pidTuningPlant, family PidtuneType, o PIDTuningOptions, res *PIDTuningResult) *PIDTuningResult {
	p.maxSensitivity = pidTuningMaxSensitivity
	p.crossLow, p.crossHigh = p.low, p.high
	peak := pidTuningSensitivityPeak(p, *res.Controller)
	if !(peak > pidTuningMaxSensitivity) {
		return res
	}
	budget := o.MaxEvaluations
	if budget == 0 {
		budget = 4096
	}
	used := res.Evidence.Evaluations
	if p.dcGain != 0 {
		for _, tune := range []func(context.Context, string, pidTuningPlant, PidtuneType, PIDTuningOptions) (*PIDTuningResult, error){tunePID, tunePIDFloor} {
			if used >= budget {
				break
			}
			q := o
			q.MaxEvaluations = budget - used
			r, err := tune(ctx, "TunePIDFRD", p, family, q)
			if err == nil {
				r.Evidence.Evaluations += used
				return r
			}
			te, ok := errors.AsType[*PIDTuningError](err)
			if !ok {
				break
			}
			used += te.Evidence.Evaluations
		}
	}
	res.Evidence.Evaluations = used
	res.Evidence.Warnings = append(res.Evidence.Warnings, fmt.Sprintf("Sampled sensitivity peak %.4g exceeds %g; no design within that bound was found at the target or with the margin as a floor.", peak, float64(pidTuningMaxSensitivity)))
	return res
}

// pidTuningFRDPlant is the tuning plant of sampled data it owns.
func pidTuningFRDPlant(data *FRD, unstablePoles *int) pidTuningPlant {
	low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
	p := pidTuningPlant{dt: data.Dt, low: low, high: high, samples: data.Omega, stability: PIDStabilitySampledOnly, at: func(w float64) complex128 {
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
	if unstablePoles == nil || *unstablePoles == 0 {
		p.dcGain, p.highGain = pidTuningFRDDCGain(data, p.at), p.at(high)
	}
	return p
}

// pidTuningFRDDCGain returns the DC gain that sampled data shows, its lowest
// sample's real part, or 0 when its low end is not a flat, nonzero real
// gain: from the lowest sample up half a decade sign·G must stay within 45°
// of the positive real axis and |G| within a factor 1.4 of the lowest
// sample's, so a rotated, integrating or resonant low end is not taken for
// DC. A gain at rounding of the sampled peak, as for TunePID, is zero.
func pidTuningFRDDCGain(data *FRD, at func(float64) complex128) float64 {
	low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
	g0 := data.Response[0][0][0]
	peak := 0.
	for _, r := range data.Response {
		peak = math.Max(peak, cmplx.Abs(r[0][0]))
	}
	if cmplx.Abs(g0) <= 1e3*0x1p-52*peak {
		return 0
	}
	sign := complex(math.Copysign(1, real(g0)), 0)
	for _, w := range []float64{low, low * math.Pow(10, .25), low * math.Sqrt(10)} {
		g := sign * at(math.Min(w, high))
		if r := cmplx.Abs(g) / cmplx.Abs(g0); !(r >= .7 && r <= 1.4) || math.Abs(cmplx.Phase(g)) > math.Pi/4 {
			return 0
		}
	}
	return real(g0)
}

// pidTuningFRDNyquist reports whether c closes a stable loop on sampled data
// of a stable plant (p.dcGain nonzero) by the Nyquist count MATLAB pidtune
// makes with NumUnstablePoles 0, and accepts every c on other data. With no
// unstable open-loop poles the number of unstable closed-loop roots is
// −Δ/π, plus ½ with integral action for its indentation at s = 0 (z = 1),
// where Δ is the phase change of 1 + L along ω from 0⁺ to +∞ (z = −1). The
// continuous derivative filter pole and the discrete ones (Tf > Dt/2 for
// forward Euler) are stable, so they add none. Below the band G runs
// linearly from the DC gain to the lowest sample, between samples along
// their interpolant, each step bisected until 1 + L turns by under 90°; one
// that cannot be rejects c, as its |S| is unbounded. Above the band 1 + L
// ends at 1 once |L| < 1 at the top sample. A discrete top sample within
// 10% of Nyquist with L within 30° of the real axis instead ends it at
// 1 + L(z = −1), which is real; Re L is extrapolated there as even in the
// distance to Nyquist from the top two samples, and an estimate whose sign
// that step leaves unresolved rejects c. Otherwise |L| >= 1 at the top: a
// continuous L within 30° of the real axis, as an ideal derivative on a
// relative-degree-one plant leaves it, ends at 1 + Re L, rejecting c when
// |1 + L| < 0.1 there; any other L below 2 and falling from the sample
// before is taken to roll off and ends at 1. Any other end is unknown and
// rejects c, since the loop has not rolled off where the data stop.
func pidTuningFRDNyquist(c PID2, p pidTuningPlant) bool {
	stable, known := pidTuningFRDNyquistCount(c, p)
	return stable && known || p.dcGain == 0
}

// pidTuningFRDNyquistCount is pidTuningFRDNyquist with known false where
// the data leave the end of 1 + L above the band unknown or p.dcGain is 0.
func pidTuningFRDNyquistCount(c PID2, p pidTuningPlant) (stable, known bool) {
	if p.dcGain == 0 {
		return true, false
	}
	dc := complex(p.dcGain, 0)
	gLow := p.at(p.low)
	at := func(w float64) complex128 {
		g := gLow
		if w < p.low {
			g = dc + (gLow-dc)*complex(w/p.low, 0)
		} else if w > p.low {
			g = p.at(w)
		}
		return 1 + g*pidTuningFeedback(c, w)
	}
	l := p.highGain * pidTuningFeedback(c, p.high)
	end := complex128(1)
	nearReal := math.Abs(imag(l)) <= .5*cmplx.Abs(l)
	nyquist := c.Dt > 0 && p.high*c.Dt >= .9*math.Pi
	switch {
	case nyquist && nearReal:
		w2 := p.samples[len(p.samples)-2]
		e1, e2 := math.Pi/c.Dt-p.high, math.Pi/c.Dt-w2
		r1, r2 := real(l), real(p.at(w2)*pidTuningFeedback(c, w2))
		r0 := (r1*e2*e2 - r2*e1*e1) / (e2*e2 - e1*e1)
		if !(math.Abs(1+r0) > .5*math.Abs(r0-r1)) {
			return false, true
		}
		end = complex(1+r0, 0)
	case cmplx.Abs(l) < 1:
	case !nearReal || c.Dt > 0:
		w2 := p.samples[len(p.samples)-2]
		dec := cmplx.Abs(p.at(w2)*pidTuningFeedback(c, w2)) > cmplx.Abs(l)
		if !dec || !(cmplx.Abs(l) < 2) {
			return true, false
		}
	case cmplx.Abs(1+l) < .1:
		return false, true
	default:
		end = complex(1+real(l), 0)
	}
	start, expected := complex(1+c.Kp*p.dcGain, 0), 0.
	w0 := 1e-6 * p.low
	if c.Ki != 0 {
		start, expected = complex(0, -c.Ki*p.dcGain), math.Pi/2
		w0 = 1e-3 * math.Min(p.low, math.Abs(c.Ki*p.dcGain)/(1+math.Abs(c.Kp*p.dcGain)))
	}
	if start == 0 || !(w0 > 0) {
		return false, true
	}
	turn := func(a, b complex128) float64 { return cmplx.Phase(b / a) }
	var walk func(wa, wb float64, fa, fb complex128, depth int) (float64, bool)
	walk = func(wa, wb float64, fa, fb complex128, depth int) (float64, bool) {
		if fa == 0 || fb == 0 || !pidFiniteComplex(fa) || !pidFiniteComplex(fb) {
			return 0, false
		}
		d := turn(fa, fb)
		if math.Abs(d) <= math.Pi/2 {
			return d, true
		}
		if depth == 30 {
			return 0, false
		}
		wm := math.Sqrt(wa * wb)
		fm := at(wm)
		d1, ok := walk(wa, wm, fa, fm, depth+1)
		if !ok {
			return 0, false
		}
		d2, ok := walk(wm, wb, fm, fb, depth+1)
		return d1 + d2, ok
	}
	wPrev, fPrev := w0, at(w0)
	total := turn(start, fPrev)
	if !(math.Abs(total) < math.Pi/2) {
		return false, true
	}
	step := func(w float64) bool {
		f := at(w)
		d, ok := walk(wPrev, w, fPrev, f, 0)
		total += d
		wPrev, fPrev = w, f
		return ok
	}
	for w := w0 * math.Sqrt(10); w < p.low; w *= math.Sqrt(10) {
		if !step(w) {
			return false, true
		}
	}
	for _, w := range p.samples {
		if w > wPrev && !step(w) {
			return false, true
		}
	}
	if fPrev == 0 {
		return false, true
	}
	total += turn(fPrev, end)
	return math.Abs(expected-total) < math.Pi/2, true
}
