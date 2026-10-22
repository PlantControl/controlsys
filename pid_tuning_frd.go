package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"
)

// TunePIDFRD tunes directly against finite measured SISO frequency data. It
// interpolates complex responses in log frequency and never extrapolates or
// invents a time-domain realization. Pole knowledge is recorded, not treated as
// sufficient evidence for global stability outside the measured frequency band.
// Unless UnstablePoles is positive, the design is first searched for among
// loops a strict sampled Nyquist count certifies stable, as MATLAB's default
// of no unstable poles has it; that count needs the data's low end to show
// a DC gain or an integrating or differentiating asymptote and the loop's
// end above the band to be known. When it certifies none, the design is
// searched for again, with an evaluation budget of its own, under the
// looser count (pidTuningFRDNyquist without the strict plant), which
// accepts every loop on data without a flat DC end, and carries a warning
// that its stability is not certified. A design whose sampled |S| peak
// exceeds 4 is searched for again under that bound at its crossover; at
// the default crossover, failing that, at crossovers lowered and then
// raised by half octaves within the band, and failing those the call
// returns ErrPIDTuningTargetUnattainable, as the true peak is at least the
// sampled one. An explicit crossover keeps such a design with a warning.
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
	explicit := o.CrossoverFrequency != 0
	if !explicit {
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
	base := pidTuningFRDPlant(data, o.UnstablePoles)
	base.warnings = warnings
	strict, certify := pidTuningFRDStrictPlant(data, o.UnstablePoles)
	strict.warnings = warnings
	return pidTuningFRDDesign(ctx, base, strict, certify, family, o, explicit)
}

// pidTuningFRDDesign is TunePIDFRD's search on the looser plant base and,
// when certify, first on the strict plant: each design passes
// pidTuningFRDGate, the strict one first; at the default crossover, when
// neither does, pidTuningFRDMove lowers and raises it on the strict plant
// and then, if base returned a design, on base. A design of base carries
// the uncertified warning. Over the |S| bound, an explicit crossover keeps
// the strict design or else base's with a warning, and the default one
// fails. Evidence counts every search.
func pidTuningFRDDesign(ctx context.Context, base, strict pidTuningPlant, certify bool, family PidtuneType, o PIDTuningOptions, explicit bool) (*PIDTuningResult, error) {
	used := 0
	pass := func(p pidTuningPlant) (*PIDTuningResult, bool, error) {
		r, err := tunePID(ctx, "TunePIDFRD", p, family, o)
		if err != nil {
			if te, ok := errors.AsType[*PIDTuningError](err); ok {
				used += te.Evidence.Evaluations
			}
			return nil, false, err
		}
		r, ok := pidTuningFRDGate(ctx, p, family, o, r)
		used += r.Evidence.Evaluations
		return r, ok, nil
	}
	counted := o.UnstablePoles == nil || *o.UnstablePoles == 0
	finish := func(r *PIDTuningResult, p pidTuningPlant) *PIDTuningResult {
		r.Evidence.Evaluations = used
		switch {
		case p.nyquistCount || !counted:
		case certify:
			r.Evidence.Warnings = append(r.Evidence.Warnings, "The strict sampled Nyquist count certified no design; this one passed only the looser count, which can accept unstable loops, so its closed-loop stability is not certified.")
		default:
			r.Evidence.Warnings = append(r.Evidence.Warnings, "The data's low end shows neither a DC gain nor an integrating or differentiating asymptote, so the sampled Nyquist count cannot run and closed-loop stability is not certified.")
		}
		return r
	}
	fail := func(err error) error {
		if te, ok := errors.AsType[*PIDTuningError](err); ok {
			te.Evidence.Evaluations = used
		}
		return err
	}
	var over *PIDTuningResult
	var overPlant pidTuningPlant
	if certify {
		r, ok, err := pass(strict)
		if err != nil && !errors.As(err, new(*PIDTuningError)) {
			return nil, err
		}
		if ok {
			return finish(r, strict), nil
		}
		over, overPlant = r, strict
	}
	r, ok, err := pass(base)
	if err != nil && !errors.As(err, new(*PIDTuningError)) {
		return nil, err
	}
	if ok {
		return finish(r, base), nil
	}
	if over == nil && r != nil {
		over, overPlant = r, base
	}
	if !explicit && certify {
		m, e := pidTuningFRDMove(ctx, strict, family, o, &used)
		if e != nil {
			return nil, e
		}
		if m != nil {
			return finish(m, strict), nil
		}
	}
	if over == nil {
		return nil, fail(err)
	}
	q := overPlant
	q.maxSensitivity, q.crossLow, q.crossHigh = pidTuningMaxSensitivity, q.low, q.high
	peak := pidTuningSensitivityPeak(q, *over.Controller)
	if explicit {
		over.Evidence.Warnings = append(over.Evidence.Warnings, fmt.Sprintf("Sampled sensitivity peak %.4g exceeds %d; no design within that bound was found at the target or with the margin as a floor, and an explicit crossover does not bound it.", peak, pidTuningMaxSensitivity))
		return finish(over, overPlant), nil
	}
	ev := PIDTuningEvidence{RequestedCrossover: o.CrossoverFrequency, RequestedPhaseMargin: o.PhaseMargin, Evaluations: used, Termination: over.Evidence.Termination, Stability: over.Evidence.Stability}
	ev.Warnings = append(slices.Clone(over.Evidence.Warnings), fmt.Sprintf("Designs at the default crossover %.4g rad/s have sampled sensitivity peak %.4g, above %d, and none within that bound was found there or at crossovers lowered and raised by half octaves within the band; the true peak is at least the sampled one.", o.CrossoverFrequency, peak, pidTuningMaxSensitivity))
	return nil, &PIDTuningError{Op: "TunePIDFRD", Evidence: ev, excess: peak - pidTuningMaxSensitivity}
}

// pidTuningFRDGate returns res and true when its sampled |S| peak is within
// pidTuningMaxSensitivity, or else the design of a search under that bound,
// first at the target band and then with the margin as a floor, as
// TunePID's, sharing res's budget, and true; failing both, res and false,
// its evaluations counting those searches. The floor's crossing band is the
// measured one, as the data have no model beyond it. On the looser plant
// without a DC gain, whose count checks nothing, no search is made, as one
// would trade a stable design over the bound for any loop within it.
func pidTuningFRDGate(ctx context.Context, p pidTuningPlant, family PidtuneType, o PIDTuningOptions, res *PIDTuningResult) (*PIDTuningResult, bool) {
	p.maxSensitivity = pidTuningMaxSensitivity
	p.crossLow, p.crossHigh = p.low, p.high
	if !(pidTuningSensitivityPeak(p, *res.Controller) > pidTuningMaxSensitivity) {
		return res, true
	}
	budget := o.MaxEvaluations
	if budget == 0 {
		budget = 4096
	}
	used := res.Evidence.Evaluations
	for _, tune := range []func(context.Context, string, pidTuningPlant, PidtuneType, PIDTuningOptions) (*PIDTuningResult, error){tunePID, tunePIDFloor} {
		if used >= budget || !p.nyquistCount && p.dcGain == 0 {
			break
		}
		q := o
		q.MaxEvaluations = budget - used
		r, err := tune(ctx, "TunePIDFRD", p, family, q)
		if err == nil {
			r.Evidence.Evaluations += used
			return r, true
		}
		te, ok := errors.AsType[*PIDTuningError](err)
		if !ok {
			break
		}
		used += te.Evidence.Evaluations
	}
	res.Evidence.Evaluations = used
	return res, false
}

// pidTuningFRDMove lowers the default crossover o.CrossoverFrequency by half
// octaves, at most 20, down to the band's lowest sample, then raises it up
// to its top, as Pidtune backs off and raises a failed one, and returns the
// first, shallowest, design within the |S| bound, or nil. Each direction
// tries every depth, shallowest first, for PM±3° and then again with the
// margin as a floor, whose search costs several times more, sharing one
// budget; used counts every search. A margin left 0 is 60° there, as a
// moved one-gain crossover targets MATLAB's 60° rather than the plant's
// own margin.
func pidTuningFRDMove(ctx context.Context, p pidTuningPlant, family PidtuneType, o PIDTuningOptions, used *int) (*PIDTuningResult, error) {
	wc0 := o.CrossoverFrequency
	budget := o.MaxEvaluations
	if budget == 0 {
		budget = 4096
	}
	p.maxSensitivity = pidTuningMaxSensitivity
	p.crossLow, p.crossHigh = p.low, p.high
	for _, dir := range []float64{-1, 1} {
		spent := 0
		for _, tune := range []func(context.Context, string, pidTuningPlant, PidtuneType, PIDTuningOptions) (*PIDTuningResult, error){tunePID, tunePIDFloor} {
			for j := 1; j <= 20 && spent < budget; j++ {
				wc := wc0 * math.Pow(2, dir*float64(j)/2)
				if wc < p.low || wc > p.high {
					break
				}
				q := o
				q.CrossoverFrequency, q.MaxEvaluations = wc, budget-spent
				if q.PhaseMargin == 0 {
					q.PhaseMargin = 60
				}
				r, err := tune(ctx, "TunePIDFRD", p, family, q)
				if err == nil {
					*used += r.Evidence.Evaluations
					verb := "lowered"
					if dir > 0 {
						verb = "raised"
					}
					r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("Default crossover %.4g %s to %.4g rad/s, where the design keeps sampled |S| <= %d.", wc0, verb, wc, pidTuningMaxSensitivity))
					return r, nil
				}
				te, ok := errors.AsType[*PIDTuningError](err)
				if !ok {
					return nil, err
				}
				spent += te.Evidence.Evaluations
				*used += te.Evidence.Evaluations
			}
		}
	}
	return nil, nil
}

// pidTuningFRDPlant is the tuning plant of sampled data it owns, under the
// looser count: its DC gain, when unstable poles are not given, is
// pidTuningFRDDCGain's.
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
		p.dcEdge = complex(p.dcGain, 0)
	}
	return p
}

// pidTuningFRDStrictPlant is pidTuningFRDPlant under the strict count, with
// the DC gain pidTuningFRDStrictDCGain takes, the low-end order
// pidTuningFRDLowOrder takes when that is 0, and the Markov end of
// continuous data. certify reports that the count can run: no unstable
// poles are given and a DC gain or low-end order anchors it.
func pidTuningFRDStrictPlant(data *FRD, unstablePoles *int) (p pidTuningPlant, certify bool) {
	p = pidTuningFRDPlant(data, unstablePoles)
	if unstablePoles != nil && *unstablePoles != 0 {
		return p, false
	}
	p.nyquistCount = true
	p.dcGain = pidTuningFRDStrictDCGain(data, p.at)
	p.dcEdge = complex(p.dcGain, 0)
	if p.dcGain == 0 {
		p.lowOrder, p.lowAnchor = pidTuningFRDLowOrder(data, p.at)
	}
	if data.Dt == 0 {
		p.markov, p.markovSpread = pidTuningFRDMarkov(data, p.at)
	}
	return p, p.dcGain != 0 || p.lowAnchor
}

// pidTuningFRDMarkov returns m = lim sG(s) that continuous sampled data
// show at their top, as a relative-degree-one plant leaves it, with spread
// the change of jω·G over the top half decade, or 0, 0 when they show none:
// there the log slope of |G| must stay within 0.1 of −1 and jω·G within 30°
// of one side of the real axis at both ends.
func pidTuningFRDMarkov(data *FRD, at func(float64) complex128) (m, spread float64) {
	low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
	w1 := math.Max(high/math.Sqrt(10), low)
	if !(w1 < high) {
		return 0, 0
	}
	g1, g2 := complex(0, w1)*at(w1), complex(0, high)*at(high)
	if g1 == 0 || g2 == 0 {
		return 0, 0
	}
	slope := math.Log(cmplx.Abs(g2)/cmplx.Abs(g1))/math.Log(high/w1) - 1
	if !(math.Abs(slope+1) <= .1) || !(math.Abs(imag(g1)) <= .5*cmplx.Abs(g1)) || !(math.Abs(imag(g2)) <= .5*cmplx.Abs(g2)) || real(g1)*real(g2) <= 0 {
		return 0, 0
	}
	return real(g2), math.Abs(real(g2 - g1))
}

// pidTuningFRDLowOrder returns n of a low end G ≈ K·(jω)^n, K real, that
// sampled data show, with ok false when they show none: over the lowest
// quarter decade the log slope of |G| must stay within 0.25 of n, n in ±1
// or ±2, and G/jⁿ within 30° of one side of the real axis, as an
// integrating or differentiating low end leaves it, turning toward that
// axis as ω falls (pidTuningFRDHeadsToAxis).
func pidTuningFRDLowOrder(data *FRD, at func(float64) complex128) (n int, ok bool) {
	low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
	w2 := math.Min(low*math.Pow(10, .25), high)
	g1, g2 := data.Response[0][0][0], at(w2)
	if !(w2 > low) || g1 == 0 || g2 == 0 {
		return 0, false
	}
	slope := math.Log(cmplx.Abs(g2)/cmplx.Abs(g1)) / math.Log(w2/low)
	r := math.Round(slope)
	if !(math.Abs(slope) > .75 && math.Abs(r) <= 2 && math.Abs(slope-r) <= .25) {
		return 0, false
	}
	j := cmplx.Pow(1i, complex(r, 0))
	h1, h2 := g1/j, g2/j
	if math.Abs(imag(h1)) <= .5*cmplx.Abs(h1) && math.Abs(imag(h2)) <= .5*cmplx.Abs(h2) && real(h1)*real(h2) > 0 && pidTuningFRDHeadsToAxis(h1, h2) && pidTuningFRDHiddenTurn(data, at, int(r)) <= math.Pi/180 {
		return int(r), true
	}
	return 0, false
}

// pidTuningFRDHeadsToAxis reports whether h, at the lowest sample h1 and a
// quarter decade up h2 on one side of the real axis, turns no further from
// it toward the lower one, within half a degree, as an end analytic in ω
// does, whose phase off its asymptote vanishes with ω. An integrator, or a
// pole or zero, hidden below the band by a zero or pole near it turns h
// away from the axis as ω falls, as the turn of (jω + z)/jω grows there.
func pidTuningFRDHeadsToAxis(h1, h2 complex128) bool {
	sign := complex(math.Copysign(1, real(h1)), 0)
	return math.Abs(cmplx.Phase(sign*h1)) <= math.Abs(cmplx.Phase(sign*h2))+math.Pi/360
}

// pidTuningFRDHiddenTurn returns |c| for the fit φ = a·x + b·x³ + c/x, x =
// ω/low, of the phase of sign·G/jⁿ, n the low end's order, at the lowest
// sample and a quarter and half decade up: an end analytic at ω = 0 has
// an odd phase, c = 0, while a pole and zero pair z below the band, as an
// integrator behind a zero, adds ±z/ω, c = ±z/low. With less than that
// half decade sampled it is +Inf.
func pidTuningFRDHiddenTurn(data *FRD, at func(float64) complex128, n int) float64 {
	low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
	if !(low*math.Sqrt(10) <= high) {
		return math.Inf(1)
	}
	j := cmplx.Pow(1i, complex(float64(n), 0))
	sign := complex(math.Copysign(1, real(data.Response[0][0][0]/j)), 0)
	var a [3][4]float64
	for i, x := range []float64{1, math.Pow(10, .25), math.Sqrt(10)} {
		a[i] = [4]float64{x, x * x * x, 1 / x, cmplx.Phase(sign * at(low*x) / j)}
	}
	for c := range 3 {
		p := c
		for r := c + 1; r < 3; r++ {
			if math.Abs(a[r][c]) > math.Abs(a[p][c]) {
				p = r
			}
		}
		a[c], a[p] = a[p], a[c]
		for r := range 3 {
			if r != c {
				f := a[r][c] / a[c][c]
				for k := c; k < 4; k++ {
					a[r][k] -= f * a[c][k]
				}
			}
		}
	}
	return math.Abs(a[2][3] / a[2][2])
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

// pidTuningFRDStrictDCGain is pidTuningFRDDCGain for the strict count: the
// lowest quarter decade must be flat as there, and its phase must head to
// the real axis as ω falls (pidTuningFRDHeadsToAxis), so a DC look an
// integrator hidden behind a zero just below the band gives is not taken;
// past the quarter decade the flat test may fail when a phase linear in ω
// there extrapolates to within 15° of the real axis at ω = 0, as a DC end
// leaves it.
func pidTuningFRDStrictDCGain(data *FRD, at func(float64) complex128) float64 {
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
	flat := func(w float64) bool {
		g := sign * at(math.Min(w, high))
		r := cmplx.Abs(g) / cmplx.Abs(g0)
		return r >= .7 && r <= 1.4 && math.Abs(cmplx.Phase(g)) <= math.Pi/4
	}
	w1, w2 := low, math.Min(low*math.Pow(10, .25), high)
	if !(w2 > w1) || !flat(w1) || !flat(w2) || !pidTuningFRDHeadsToAxis(g0, at(w2)) || pidTuningFRDHiddenTurn(data, at, 0) > math.Pi/180 {
		return 0
	}
	if flat(low * math.Sqrt(10)) {
		return real(g0)
	}
	p1, p2 := cmplx.Phase(sign*g0), cmplx.Phase(sign*at(w2))
	if p0 := p1 - (p2-p1)*w1/(w2-w1); !(math.Abs(p0) <= math.Pi/12) {
		return 0
	}
	return real(g0)
}

// pidTuningFRDNyquist reports whether c closes a stable loop on sampled data
// by the Nyquist count MATLAB pidtune makes with NumUnstablePoles 0. On the
// strict plant (p.nyquistCount) a c the count cannot certify is rejected;
// on the looser one every c is accepted on data without a DC gain, as on
// data with other pole counts. With no unstable open-loop poles the number
// of unstable closed-loop roots is −Δ/π, plus ½ with integral action for
// its indentation at s = 0 (z = 1), where Δ is the phase change of 1 + L
// along ω from 0⁺ to +∞ (z = −1). The continuous derivative filter pole is
// stable, so it adds none; a discrete one, 1 − Dt/(Tf + w·Dt) for the
// derivative formula's weight w, outside the unit circle rejects c. On it,
// at z = −1 (an unfiltered trapezoidal derivative), L is infinite: c is
// rejected unless the data reach within 10% of Nyquist with G within 30° of
// the real axis, |L| >= 2, and the derivative term at least twice the rest
// of c, all at the top sample; then 1 + L heads to j·s·∞, s the sign of
// Kd·Re G, and the indentation turns it by −π/2 to s·∞ on the real axis.
// Without integral action on a DC end c is rejected unless |1 + Kp·dcGain|
// exceeds |Kp| times the change in Re G over the lowest quarter decade,
// which bounds the gap between the lowest sample's real part and the DC
// gain; nearer, the loop's real root at s ≈ 0 takes the sign that gap
// leaves unknown. Below the band G runs linearly from the DC gain to the
// lowest sample, or on a low end of order n (strict plant) as
// G(low)·(ω/low)^n; there L ≈ k·K·(jω)^m, k the lowest-order gain of c and
// m the order of the loop, which starts on the real axis at sign(k·K) and
// turns by m·π/2 in the indentation when m < 0, needing |L(low)| >= 2 so
// its gap below the band is the indentation's alone, and c to have
// proportional or integral action, as a lone derivative's zero cancels an
// integrator, or starts at 1 when m > 0 and c has no integral action,
// needing |L(low)| <= 1/2; any other c is unknown. Between samples G runs
// along their interpolant, each step bisected until 1 + L turns by under
// 90°; one that cannot be rejects c, as its |S| is unbounded.
//
// A discrete top sample within 10% of Nyquist with L within 30° of the real
// axis ends 1 + L at 1 + L(z = −1), which is real; Re L is extrapolated
// there as even in the distance to Nyquist from the top two samples, and an
// estimate whose sign that step leaves unresolved rejects c. Otherwise the
// two plants part. On the looser one, 1 + L ends at 1 once |L| < 1 at the
// top sample; any other discrete L within 10% of Nyquist rejects c; a
// continuous L with |L| >= 1 within 30° of the real axis ends at 1 + Re L,
// rejecting c when |1 + L| < 0.1 there; any other L below 2 and falling
// from the sample before ends at 1; any other end is unknown. On the strict
// one an end is unknown unless the data fix it: a forward-Euler ideal
// derivative (Dt > 0, Tf = 0) is improper, and its pole at z = ∞ the data
// cannot show; an ideal derivative (Tf = 0) on continuous data showing
// lim sG = m at their top (pidTuningFRDMarkov) ends at 1 + Kd·m when
// Re(1 + L) at the top sample, less twice the change of Re(Kd·jω·G) over
// the top half decade, keeps its sign; and any other end is 1 when
// pidTuningFRDTopBounded keeps |L| below 1 past the band (below ½ when
// |L| < 1 rises at the top sample), needing |L| < 2 there, and short of
// 10% of Nyquist or with |L| < 1 at the top sample.
func pidTuningFRDNyquist(c PID2, p pidTuningPlant) bool {
	stable, known := pidTuningFRDNyquistCount(c, p)
	return stable && known || !p.nyquistCount && p.dcGain == 0
}

// pidTuningFRDTopBounded reports whether |L| above the top sample high
// stays below limit over a decade (continuous) or up to Nyquist
// (discrete), so 1 + L is taken to end at 1: c is evaluated exactly, at
// eight log-spaced points past high, and |G| extrapolated at its log slope
// from w2 to high, or, discrete, held at least |G(high)|, as a discrete
// |G| flattens toward Nyquist and a pole near z = −1 lifts it. A continuous
// ideal derivative, unbounded, is not unless |G| falls at least as
// ω^−1.5 there, as on a plant of relative degree two or more, past which
// L keeps falling.
func pidTuningFRDTopBounded(c PID2, high, w2 float64, gHigh, g2 complex128, limit float64) bool {
	if !(w2 < high) || gHigh == 0 || g2 == 0 {
		return false
	}
	end := 10 * high
	slope := math.Log(cmplx.Abs(gHigh)/cmplx.Abs(g2)) / math.Log(high/w2)
	if c.Dt > 0 {
		end, slope = math.Pi/c.Dt, math.Max(slope, 0)
	} else if c.Tf == 0 && c.Kd != 0 && !(slope <= -1.5) {
		return false
	}
	for k := 1; k <= 8; k++ {
		r := math.Pow(end/high, float64(k)/8)
		if !(cmplx.Abs(gHigh)*math.Pow(r, slope)*cmplx.Abs(pidTuningFeedback(c, high*r)) < limit) {
			return false
		}
	}
	return true
}

// pidTuningFRDNyquistCount is pidTuningFRDNyquist with known false where
// the data leave an end of 1 + L unknown or nothing anchors the count.
func pidTuningFRDNyquistCount(c PID2, p pidTuningPlant) (stable, known bool) {
	if p.dcGain == 0 && !p.lowAnchor {
		return true, false
	}
	if p.nyquistCount && c.Dt > 0 && c.Tf == 0 && c.Kd != 0 && c.DFormula == ForwardEuler {
		return true, false
	}
	k := c.Tf + c.Dt*pidFormulaWeight(c.DFormula)
	if c.Dt > 0 && c.Kd != 0 && k != 0 && k < c.Dt/2 {
		return false, true
	}
	circle := c.Dt > 0 && c.Kd != 0 && k == c.Dt/2
	dc := complex(p.dcGain, 0)
	gLow := p.at(p.low)
	at := func(w float64) complex128 {
		g := gLow
		if w < p.low && p.lowAnchor {
			g = gLow * complex(math.Pow(w/p.low, float64(p.lowOrder)), 0)
		} else if w < p.low {
			g = dc + (gLow-dc)*complex(w/p.low, 0)
		} else if w > p.low {
			g = p.at(w)
		}
		return 1 + g*pidTuningFeedback(c, w)
	}
	l := p.highGain * pidTuningFeedback(c, p.high)
	end, bump := complex128(1), 0.
	nearReal := math.Abs(imag(l)) <= .5*cmplx.Abs(l)
	nyquist := c.Dt > 0 && p.high*c.Dt >= .9*math.Pi
	w2 := p.samples[len(p.samples)-2]
	l2 := p.at(w2) * pidTuningFeedback(c, w2)
	falling := cmplx.Abs(l2) > cmplx.Abs(l)
	switch {
	case circle:
		i, d := pidTuningBasis(c, p.high)
		g, s := p.highGain, math.Copysign(1, c.Kd*real(p.highGain))
		if !nyquist || !(math.Abs(imag(g)) <= .5*cmplx.Abs(g)) || !(cmplx.Abs(complex(c.Kd, 0)*d) >= 2*cmplx.Abs(complex(c.Kp, 0)+complex(c.Ki, 0)*i)) || !(cmplx.Abs(l) >= 2) || s*imag(l) <= 0 {
			return false, true
		}
		end, bump = complex(0, s), -math.Pi/2
	case nyquist && nearReal:
		e1, e2 := math.Pi/c.Dt-p.high, math.Pi/c.Dt-w2
		r1, r2 := real(l), real(l2)
		r0 := (r1*e2*e2 - r2*e1*e1) / (e2*e2 - e1*e1)
		if !(math.Abs(1+r0) > .5*math.Abs(r0-r1)) {
			return false, true
		}
		end = complex(1+r0, 0)
	case !p.nyquistCount:
		switch {
		case cmplx.Abs(l) < 1:
		case nyquist:
			return true, false
		case !nearReal || c.Dt > 0:
			if !falling || !(cmplx.Abs(l) < 2) {
				return true, false
			}
		case cmplx.Abs(1+l) < .1:
			return false, true
		default:
			end = complex(1+real(l), 0)
		}
	case c.Dt == 0 && c.Tf == 0 && c.Kd != 0 && p.markov != 0:
		end = complex(1+c.Kd*p.markov, 0)
		if !(math.Abs(real(end)) > math.Abs(real(1+l-end))+2*math.Abs(c.Kd)*p.markovSpread) {
			return true, false
		}
	default:
		limit := 1.
		if cmplx.Abs(l) < 1 && !falling {
			limit = .5
		}
		if nyquist && !(cmplx.Abs(l) < 1) || !(cmplx.Abs(l) < 2) || !pidTuningFRDTopBounded(c, p.high, w2, p.highGain, p.at(w2), limit) {
			return true, false
		}
	}
	start, expected := complex(1+c.Kp*p.dcGain, 0), 0.
	w0 := 1e-6 * p.low
	if p.lowAnchor {
		coef, m, ratio := c.Kd, p.lowOrder+1, 1.
		switch {
		case c.Ki != 0:
			coef, m = c.Ki, p.lowOrder-1
			ratio = math.Abs(c.Ki) / (math.Abs(c.Kp) + math.Abs(c.Kd)*p.low + math.Abs(c.Ki)/p.low)
		case c.Kp != 0:
			coef, m = c.Kp, p.lowOrder
			ratio = math.Abs(c.Kp) / (math.Abs(c.Kp) + math.Abs(c.Kd)*p.low)
		}
		j := cmplx.Pow(1i, complex(float64(p.lowOrder), 0))
		ll := cmplx.Abs(gLow * pidTuningFeedback(c, p.low))
		switch {
		case m < 0 && ll >= 2 && (c.Kp != 0 || c.Ki != 0):
			start = complex(coef*real(gLow/j), 0) * cmplx.Pow(1i, complex(float64(m), 0))
			expected = -float64(m) * math.Pi / 2
		case m > 0 && ll <= .5 && c.Ki == 0:
		default:
			return true, false
		}
		w0 = 1e-3 * p.low * ratio
	} else if c.Ki != 0 {
		start, expected = complex(0, -c.Ki*p.dcGain), math.Pi/2
		w0 = 1e-3 * math.Min(p.low, math.Abs(c.Ki*p.dcGain)/(1+math.Abs(c.Kp*p.dcGain)))
	}
	if start == 0 || !(w0 > 0) {
		return false, true
	}
	if c.Ki == 0 && !p.lowAnchor && !(math.Abs(real(start)) > math.Abs(c.Kp)*math.Abs(real(p.at(math.Min(p.low*math.Pow(10, .25), p.high)))-p.dcGain)) {
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
	total += turn(fPrev, end) + bump
	return math.Abs(expected-total) < math.Pi/2, true
}
