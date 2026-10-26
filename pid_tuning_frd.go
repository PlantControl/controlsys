package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"
	"strings"
)

// TunePIDFRD tunes directly against finite measured SISO frequency data. It
// interpolates complex responses in log frequency and never extrapolates or
// invents a time-domain realization. Pole knowledge is recorded, not treated as
// sufficient evidence for global stability outside the measured frequency band.
// A sample at ω = 0, whose response must be real, gives the DC gain.
// Unless UnstablePoles is positive, the design is first searched for among
// loops a strict sampled Nyquist count finds stable, as MATLAB's default
// of no unstable poles has it; that count needs the data's low end to show
// a DC gain or an integrating or differentiating asymptote and the loop's
// end above the band to be known. It certifies a design only when both
// ends are anchored: the DC gain is sampled, and the band's low end shows
// it, and the data reach Nyquist (discrete) or an asymptote above the
// band (continuous); a design it finds stable on extrapolated ends carries
// a warning that its stability is not certified. When it finds none, the
// design is searched for again, with an evaluation budget of its own,
// under the looser count (pidTuningFRDNyquist without the strict plant),
// which accepts every loop on data without a flat DC end, and carries
// that warning; one the strict count finds unstable fails with
// ErrPIDTuningTargetUnattainable instead, whether or not the data reach
// the end above the band it took.
// That search follows the strict count along the chords between samples;
// a design it passes that a resonance the samples may not resolve leaves
// uncertain (pidTuningFRDUnresolved) carries the uncertified warning, as
// no search on the same samples can rule that resonance out; so does one
// whose loop, along a fit across such a step, reaches |L| = 1 or |S| > 4,
// or stays below |L| = 1 where a lighter pair in the step could lift it
// through. The returned design's margins count its crossings along the
// data's resonance fits and the cubic across smooth steps too
// (pidTuningFRDFitMargins); a design with a crossing those leave
// unresolved, in a step the chord and the polynomials through its
// neighbours read apart, carries the uncertified warning, nor are its
// margins certified. A certified design whose
// |S| the data leave above 4 where the sampled scan reads none (along a
// fit between samples, between DC and the band, or past its top;
// pidTuningFRDUnsampledPeak) is returned unchanged with a warning that
// its sensitivity peak is not certified.
// A design whose sampled |S| peak
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
	if frd == nil || len(frd.Omega) < 3 || len(frd.Omega) > 10001 || len(frd.Response) != len(frd.Omega) || frd.Omega[0] != 0 && len(frd.Omega) > 10000 {
		return nil, invalid("require 3..10000 matched frequency samples")
	}
	if !finitePID(frd.Dt) || frd.Dt < 0 {
		return nil, invalid("invalid sample time")
	}
	for k, w := range frd.Omega {
		if !finitePID(w) || w < 0 || k > 0 && w <= frd.Omega[k-1] {
			return nil, invalid("frequencies must be finite, nonnegative, strictly increasing")
		}
		if frd.Dt > 0 && w >= math.Pi/frd.Dt {
			return nil, invalid("samples must be below Nyquist")
		}
		if len(frd.Response[k]) != 1 || len(frd.Response[k][0]) != 1 || !pidFiniteComplex(frd.Response[k][0][0]) {
			return nil, invalid("responses must be finite SISO values")
		}
	}
	if frd.Omega[0] == 0 {
		if len(frd.Omega) < 4 {
			return nil, invalid("require 3..10000 matched frequency samples above DC")
		}
		if g := frd.Response[0][0][0]; !(math.Abs(imag(g)) <= 1e-8*cmplx.Abs(g)) {
			return nil, invalid("DC response must be real")
		}
	}
	if o.UnstablePoles != nil && (*o.UnstablePoles < 0 || *o.UnstablePoles > 256) {
		return nil, invalid("unstable pole count must be 0..256 or unknown")
	}
	// Own the sampled snapshot while evaluating; callers may reuse their source.
	data := frd.Copy()
	explicit := o.CrossoverFrequency != 0
	pos, _, _ := pidTuningFRDSplitDC(data)
	if !explicit {
		low, high := pos.Omega[0], pos.Omega[len(pos.Omega)-1]
		o.CrossoverFrequency = math.Exp((math.Log(low) + math.Log(high)) / 2)
	}
	warnings := []string{"Finite measured frequency coverage cannot certify global closed-loop stability or provide time responses."}
	if o.UnstablePoles == nil {
		warnings = append(warnings, "Unstable open-loop pole count is unknown.")
	} else {
		warnings = append(warnings, fmt.Sprintf("User-supplied unstable open-loop pole count: %d; unmeasured endpoint behavior remains unknown.", *o.UnstablePoles))
	}
	for k := 1; k < len(pos.Omega); k++ {
		if pos.Omega[k]/pos.Omega[k-1] > 1.2 {
			warnings = append(warnings, "Sparse frequency coverage may hide resonances and phase rotations; inspect measured resolution.")
			break
		}
	}
	base := pidTuningFRDPlant(data, o.UnstablePoles)
	base.warnings = warnings
	strict, certify := pidTuningFRDStrictPlant(data, o.UnstablePoles)
	strict.warnings = warnings
	chord := strict
	chord.unresolved = nil
	r, err := pidTuningFRDDesign(ctx, base, chord, certify, family, o, explicit)
	if err != nil || !certify {
		return r, err
	}
	resolved := pidTuningFRDFitMargins(r, strict)
	c := *r.Controller
	switch {
	case !pidTuningFRDCertified(c, chord):
	case !pidTuningFRDCertified(c, strict):
		r.Evidence.Warnings = append(r.Evidence.Warnings, "The strict sampled Nyquist count passed this design along the chords between samples, but a step of the data may hide a lightly damped resonance whose loop winds 1 + L round 0 or takes |L| through 1 or |S| past 4 between them, so its closed-loop stability is not certified, nor its margins and sensitivity peak there.")
	case !resolved:
		r.Evidence.Warnings = append(r.Evidence.Warnings, "The strict sampled Nyquist count passed this design, but a gain crossing lies in a step of the data no fit follows, where the chord and the cubic and quadratics through the neighbouring samples read its phase margin apart, so the design is not certified, nor its margins there.")
	default:
		if v := pidTuningFRDUnsampledPeak(strict, c); v > pidTuningMaxSensitivity {
			r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("The strict sampled Nyquist count passed this design, but the data leave its sensitivity peak |S| up to %.4g, above %d, along a fitted resonance between samples or below or above the band, so its sensitivity peak is not certified.", v, pidTuningMaxSensitivity))
		}
	}
	return r, nil
}

// pidTuningFRDUnsampledPeak returns the largest |S| of c the strict plant
// p's data leave open where the sampled scan (pidTuningSensitivityPeak)
// reads none: along every fit of each step crossArcs, unresolved or arcs
// holds, both fits taken as the samples may not fix the pole's σ; along
// the line the count runs G on from a sampled DC gain to the lowest
// sample; discrete, at z = −1 extrapolated from the top two samples as the
// count does where it ends 1 + L there (within 10% of Nyquist, L within
// 30° of the real axis, no derivative pole at z = −1); and continuous, with an
// ideal derivative, at L(∞) = Kd·m for any m between the Markov end the
// data show at their top and its extrapolation from there and half a
// decade below, Re jωG being m + O(ω⁻²). It only reads: the design is
// unchanged.
func pidTuningFRDUnsampledPeak(p pidTuningPlant, c PID2) float64 {
	peak := 0.
	at := func(l complex128) {
		if v := 1 / cmplx.Abs(1+l); !math.IsNaN(v) {
			peak = math.Max(peak, v)
		}
	}
	for _, set := range [][][]pidTuningFRDMode{p.crossArcs, p.unresolved, p.arcs} {
		for k, fits := range set {
			for _, m := range fits {
				_, v := m.max(func(w float64) float64 {
					v := 1 / cmplx.Abs(1+m.at(w)*pidTuningFeedback(c, w))
					if math.IsNaN(v) {
						return math.Inf(1)
					}
					return v
				}, p.samples[k], p.samples[k+1])
				peak = math.Max(peak, v)
			}
		}
	}
	if p.lowAnchored && !p.lowAnchor && p.dcGain != 0 {
		dc, gLow := complex(p.dcGain, 0), p.at(p.low)
		s := func(w float64) float64 {
			v := 1 / cmplx.Abs(1+(dc+(gLow-dc)*complex(w/p.low, 0))*pidTuningFeedback(c, w))
			if math.IsNaN(v) {
				return 0
			}
			return v
		}
		const pieces = 64
		grid := func(i int) float64 { return p.low * math.Pow(1e-6, 1-float64(i)/pieces) }
		best, bv := 0, -1.
		for i := 0; i <= pieces; i++ {
			if v := s(grid(i)); v > bv {
				best, bv = i, v
			}
		}
		_, g := pidTuningGoldenMax(s, grid(max(best-1, 0)), grid(min(best+1, pieces)))
		peak = math.Max(peak, math.Max(bv, g))
	}
	n := len(p.samples)
	if n < 2 {
		return peak
	}
	high, w2 := p.samples[n-1], p.samples[n-2]
	l, l2 := p.at(high)*pidTuningFeedback(c, high), p.at(w2)*pidTuningFeedback(c, w2)
	circle := c.Dt > 0 && c.Kd != 0 && c.Tf+c.Dt*pidFormulaWeight(c.DFormula) == c.Dt/2
	switch {
	case c.Dt > 0 && high*c.Dt >= .9*math.Pi && !circle && math.Abs(imag(l)) <= .5*cmplx.Abs(l):
		e1, e2 := math.Pi/c.Dt-high, math.Pi/c.Dt-w2
		at(complex((real(l)*e2*e2-real(l2)*e1*e1)/(e2*e2-e1*e1), 0))
	case c.Dt == 0 && c.Tf == 0 && c.Kd != 0 && p.markov != 0:
		w1 := math.Max(high/math.Sqrt(10), p.low)
		r1 := real(complex(0, w1) * p.at(w1))
		m := (p.markov*high*high - r1*w1*w1) / (high*high - w1*w1)
		lo, hi := c.Kd*p.markov, c.Kd*m
		if lo > hi {
			lo, hi = hi, lo
		}
		at(complex(math.Min(math.Max(-1, lo), hi), 0))
	}
	return peak
}

// pidTuningFRDFitMargins adds to r's evidence its design's gain crossings
// along the fits of the strict plant p's data (pidTuningFRDFitCrossings)
// and along the cubic across smooth steps (pidTuningFRDSmoothCrossing)
// that it lacks, its achieved margin taken at their least: the search
// reads crossings off the chord and, across a lightly damped mode, the
// fits of steps whose circle turns past a quarter turn, while a chord
// cutting inside a circle turning less can still put a crossing's phase
// 8° off, or miss the crossing pair a resonance lifts between samples.
// It reports whether every crossing is read where the data resolve it: on
// a sample, outside the samples, in a marked step, along a fit of its
// step or the cubic's gate, in a step with a trusted fit
// (pidTuningFRDTrusted), or where no reading of pidTuningFRDStepReading
// falls more than pidTuningFRDMarginSlack below the achieved margin.
// Reading is all it does; the search and so the design are unchanged.
func pidTuningFRDFitMargins(r *PIDTuningResult, p pidTuningPlant) (resolved bool) {
	c := *r.Controller
	q := p
	q.unresolved = nil
	stable, known, _ := pidTuningFRDNyquistVerdict(c, q)
	ev := &r.Evidence
	ws, ms := pidTuningFRDFitCrossings(p, c)
	nfit := len(ws)
	lw, lm := pidTuningFRDLowCrossings(p, c)
	ws, ms = append(ws, lw...), append(ms, lm...)
	var raw *FRD
	var open []float64
	for _, w := range ev.GainCrossovers {
		v, m, smooth, lo, hi := pidTuningFRDStepReading(p, c, w)
		if smooth {
			ws, ms = append(ws, v), append(ms, m)
			continue
		}
		k := sort.SearchFloat64s(p.samples, w) - 1
		if k < 0 || k+1 >= len(p.samples) || !(w > p.samples[k] && w < p.samples[k+1]) || slices.ContainsFunc([][][]pidTuningFRDMode{p.crossArcs, p.unresolved, p.arcs}, func(set [][]pidTuningFRDMode) bool { return set != nil && set[k] != nil }) {
			continue
		}
		if a, b := p.samples[k], p.samples[k+1]; slices.ContainsFunc(ws[:nfit], func(x float64) bool { return x > a && x < b }) {
			continue
		}
		if raw == nil {
			raw = pidTuningFRDSampled(p)
		}
		if f, nf, _ := pidTuningFRDStepFits(raw, k); len(pidTuningFRDTrusted(f[:nf])) > 0 {
			continue
		}
		if stable && known {
			switch {
			case hi <= 0:
				lo = -hi
			case lo < 0:
				lo = 0
			}
		}
		open = append(open, lo)
	}
	for i, w := range ws {
		if slices.ContainsFunc(ev.GainCrossovers, func(x float64) bool { return math.Abs(math.Log(w/x)) <= 1e-6 }) {
			continue
		}
		m := ms[i]
		if stable && known {
			m = math.Abs(m)
		}
		ev.GainCrossovers, ev.PhaseMargins = append(ev.GainCrossovers, w), append(ev.PhaseMargins, m)
		ev.AchievedPhaseMargin = math.Min(ev.AchievedPhaseMargin, m)
	}
	for _, lo := range open {
		if !(lo >= ev.AchievedPhaseMargin-pidTuningFRDMarginSlack) {
			return false
		}
	}
	return true
}

// pidTuningFRDSampled is the strict plant p's samples as FRD data.
func pidTuningFRDSampled(p pidTuningPlant) *FRD {
	n := len(p.samples)
	vals, rows := make([]complex128, n), make([][]complex128, n)
	raw := &FRD{Omega: p.samples, Response: make([][][]complex128, n), Dt: p.dt}
	for i, w := range p.samples {
		vals[i] = p.at(w)
		rows[i] = vals[i : i+1]
		raw.Response[i] = rows[i : i+1]
	}
	return raw
}

// pidTuningFRDSmoothCrossing re-reads c's gain crossing w, read off the
// chord inside a step of the strict plant p's samples that no fit follows,
// along the cubic in ln ω through ln G at the step's samples and their
// outer neighbours, and returns the crossing of that interpolant nearest
// w, its margin wrapped as pidTuningMargins, and true; false when w sits
// on a sample, a step of the four samples is marked or turns G past 30°,
// the quadratics through the first and last three disagree at w by more
// than a quarter of the cubic's departure from the chord, as near a mode
// the samples barely resolve, or the cubic has no crossing there. Across
// a step where G turns smoothly but c's loop does not, as about a lightly
// damped zero pair of c that dips |L| through 1 between two samples far
// above it, the chord's error in G alone can put the margin 5° off; the
// cubic's is of fourth order in the step.
func pidTuningFRDSmoothCrossing(p pidTuningPlant, c PID2, w float64) (float64, float64, bool) {
	v, m, smooth, _, _ := pidTuningFRDStepReading(p, c, w)
	return v, m, smooth
}

// pidTuningFRDStepReading is pidTuningFRDSmoothCrossing's re-reading of
// c's gain crossing w, smooth its gate, with lo and hi the least and
// greatest margin of the chord at w and, nearest w, of the cubic and
// both quadratics across w's step, which no mark covers and G turns at most 30° over,
// its neighbours marked or turning or not, among those crossing there,
// as one that does not leaves no lower margin; ±Inf where the step has
// no outer neighbours, is marked or turns further. Beside a lightly
// damped mode the chord can read 3° high where the gate fails, as its
// neighbour turns, while the cubic reads 2° low, as at ζ 0.01.
func pidTuningFRDStepReading(p pidTuningPlant, c PID2, w float64) (v, pm float64, smooth bool, lo, hi float64) {
	s := p.samples
	k := sort.SearchFloat64s(s, w) - 1
	if k < 1 || k+2 >= len(s) || !(w > s[k] && w < s[k+1]) {
		return 0, 0, false, math.Inf(-1), math.Inf(1)
	}
	marked := func(j int) bool {
		for _, set := range [][][]pidTuningFRDMode{p.crossArcs, p.unresolved, p.arcs, p.resonances} {
			if set != nil && set[j] != nil {
				return true
			}
		}
		return false
	}
	turns := func(j int) bool { return !(math.Abs(cmplx.Phase(p.at(s[j+1])/p.at(s[j]))) <= math.Pi/6) }
	if marked(k) || turns(k) {
		return 0, 0, false, math.Inf(-1), math.Inf(1)
	}
	var x [4]float64
	var z [4]complex128
	for i := range 4 {
		g := p.at(s[k-1+i])
		if g == 0 || !pidFiniteComplex(g) {
			return 0, 0, false, math.Inf(-1), math.Inf(1)
		}
		x[i], z[i] = math.Log(s[k-1+i]), cmplx.Log(g)
		if i > 0 {
			z[i] -= complex(0, 2*math.Pi*math.Round((imag(z[i])-imag(z[i-1]))/(2*math.Pi)))
		}
	}
	interp := func(w float64, from, to int) complex128 {
		t := math.Log(w)
		var v complex128
		for i := from; i <= to; i++ {
			l := 1.
			for j := from; j <= to; j++ {
				if j != i {
					l *= (t - x[j]) / (x[i] - x[j])
				}
			}
			v += complex(l, 0) * z[i]
		}
		return v
	}
	a, b := s[k], s[k+1]
	cross := func(g func(float64) complex128) (float64, float64, bool) {
		l := func(w float64) complex128 { return g(w) * pidTuningFeedback(c, w) }
		e := func(w float64) float64 { return cmplx.Abs(l(w)) - 1 }
		const pieces = 64
		best, found := 0., false
		wa, ea := a, e(a)
		for i := 1; i <= pieces; i++ {
			wb := a * math.Pow(b/a, float64(i)/pieces)
			eb := e(wb)
			if ea*eb < 0 {
				x, y, sx := wa, wb, ea
				for range 60 {
					mid := math.Sqrt(x * y)
					if v := e(mid); sx*v <= 0 {
						y = mid
					} else {
						x = mid
					}
				}
				if v := math.Sqrt(x * y); !found || math.Abs(math.Log(v/w)) < math.Abs(math.Log(best/w)) {
					best, found = v, true
				}
			}
			wa, ea = wb, eb
		}
		if !found {
			return 0, 0, false
		}
		pm := 180 + cmplx.Phase(l(best))*180/math.Pi
		if pm > 180 {
			pm -= 360
		}
		return best, pm, true
	}
	gate := !marked(k-1) && !marked(k+1) && !turns(k-1) && !turns(k+1) && cmplx.Abs(interp(w, 0, 2)-interp(w, 1, 3)) <= .25*cmplx.Abs(interp(w, 0, 3)-cmplx.Log(p.at(w)))
	v, pm, found := cross(func(w float64) complex128 { return cmplx.Exp(interp(w, 0, 3)) })
	if gate && found {
		return v, pm, true, pm, pm
	}
	ref := 180 + cmplx.Phase(p.at(w)*pidTuningFeedback(c, w))*180/math.Pi
	if ref > 180 {
		ref -= 360
	}
	lo, hi = ref, ref
	if found {
		m := ref + pidTuningWrapDeg(pm-ref)
		lo, hi = math.Min(lo, m), math.Max(hi, m)
	}
	for _, g := range []func(float64) complex128{func(w float64) complex128 { return cmplx.Exp(interp(w, 0, 2)) }, func(w float64) complex128 { return cmplx.Exp(interp(w, 1, 3)) }} {
		if _, m, ok := cross(g); ok {
			m = ref + pidTuningWrapDeg(m-ref)
			lo, hi = math.Min(lo, m), math.Max(hi, m)
		}
	}
	return 0, 0, false, lo, hi
}

// pidTuningFRDMarginSlack is how far, in degrees, a certified design's
// reported margin may read above its plant's.
const pidTuningFRDMarginSlack = 1.5

// pidTuningWrapDeg wraps d degrees into [−180, 180).
func pidTuningWrapDeg(d float64) float64 { return d - 360*math.Floor((d+180)/360) }

// pidTuningFRDFitCrossings returns c's gain crossings, and their margins
// wrapped as pidTuningMargins, along every fit of p's data that follows a
// lightly damped mode across a step: each of crossArcs, of unresolved
// (a lone or disagreeing fit, which the margin reading takes as it is,
// the step then uncertified) and of arcs. Each fit is read at 64 points
// even in the angle its pole sees and at its |L| peak, and each sign
// change of |L| − 1 bisected. Any other step whose |L| crosses 1 or comes
// within pidTuningFRDNearUnity of it at a sample is read along its trusted
// fits too (pidTuningFRDStepFits, pidTuningFRDTrusted), as the chord there
// can miss a crossing pair a mild curvature lifts between samples, or put
// a crossing's phase off.
func pidTuningFRDFitCrossings(p pidTuningPlant, c PID2) (crossings, margins []float64) {
	var raw *FRD
	for k := 0; k+1 < len(p.samples); k++ {
		var fits []pidTuningFRDMode
		marked := false
		for _, set := range [][][]pidTuningFRDMode{p.crossArcs, p.unresolved, p.arcs} {
			if set != nil && set[k] != nil {
				fits, marked = set[k], true
				break
			}
		}
		a, b := p.samples[k], p.samples[k+1]
		if !marked {
			ea, eb := cmplx.Abs(p.at(a)*pidTuningFeedback(c, a))-1, cmplx.Abs(p.at(b)*pidTuningFeedback(c, b))-1
			if !(math.Min(math.Abs(ea), math.Abs(eb)) < pidTuningFRDNearUnity) && !(ea*eb <= 0) {
				continue
			}
			if raw == nil {
				raw = pidTuningFRDSampled(p)
			}
			f, nf, _ := pidTuningFRDStepFits(raw, k)
			fits = pidTuningFRDTrusted(f[:nf])
			if len(fits) == 0 && ea*eb < 0 {
				fits = f[:nf]
			}
		}
		for _, m := range fits {
			l := func(w float64) complex128 { return m.at(w) * pidTuningFeedback(c, w) }
			e := func(w float64) float64 { return cmplx.Abs(l(w)) - 1 }
			w0, sigma := real(m.pole), imag(m.pole)
			ta, tb := math.Atan((a-w0)/sigma), math.Atan((b-w0)/sigma)
			const pieces = 64
			grid := make([]float64, 0, pieces+2)
			for i := 0; i <= pieces; i++ {
				grid = append(grid, math.Min(math.Max(w0+sigma*math.Tan(ta+(tb-ta)*float64(i)/pieces), a), b))
			}
			wp, _ := m.max(func(w float64) float64 { return cmplx.Abs(l(w)) }, a, b)
			grid = append(grid, wp)
			slices.Sort(grid)
			lo, elo := a, e(a)
			for _, hi := range grid[1:] {
				ehi := e(hi)
				if elo*ehi < 0 {
					x, y, sx := lo, hi, elo
					for range 60 {
						mid := math.Sqrt(x * y)
						if v := e(mid); sx*v <= 0 {
							y = mid
						} else {
							x = mid
						}
					}
					w := math.Sqrt(x * y)
					pm := 180 + cmplx.Phase(l(w))*180/math.Pi
					if pm > 180 {
						pm -= 360
					}
					crossings, margins = append(crossings, w), append(margins, pm)
				}
				lo, elo = hi, ehi
			}
		}
	}
	return crossings, margins
}

// pidTuningFRDLowCrossings returns c's gain crossings below the band of
// the strict plant p, and their margins wrapped as pidTuningMargins,
// along the line the count runs G on from the sampled DC gain to the
// lowest sample, when the band's low end shows that gain (lowAnchored):
// a loop with |L| at the lowest sample just past 1 and |Kp·G(0)| below it
// crosses there, outside the band the search reads.
func pidTuningFRDLowCrossings(p pidTuningPlant, c PID2) (crossings, margins []float64) {
	if !p.lowAnchored || p.lowAnchor || p.dcGain == 0 {
		return nil, nil
	}
	dc, gLow := complex(p.dcGain, 0), p.at(p.low)
	l := func(w float64) complex128 { return (dc + (gLow-dc)*complex(w/p.low, 0)) * pidTuningFeedback(c, w) }
	e := func(w float64) float64 { return cmplx.Abs(l(w)) - 1 }
	const pieces = 64
	lo, elo := 1e-6*p.low, e(1e-6*p.low)
	for i := 1; i <= pieces; i++ {
		hi := p.low * math.Pow(1e-6, 1-float64(i)/pieces)
		if i == pieces {
			hi = p.low
		}
		ehi := e(hi)
		if elo*ehi < 0 || ehi == 0 {
			x, y, sx := lo, hi, elo
			for range 60 {
				mid := math.Sqrt(x * y)
				if v := e(mid); sx*v <= 0 {
					y = mid
				} else {
					x = mid
				}
			}
			w := math.Sqrt(x * y)
			pm := 180 + cmplx.Phase(l(w))*180/math.Pi
			if pm > 180 {
				pm -= 360
			}
			crossings, margins = append(crossings, w), append(margins, pm)
		}
		lo, elo = hi, ehi
	}
	return crossings, margins
}

// pidTuningFRDNearUnity is how near 1 sampled |L| − 1 at a step's end must
// come for pidTuningFRDFitCrossings to read the step along its fits.
const pidTuningFRDNearUnity = .1

// pidTuningFRDCertified reports whether the strict count p certifies c:
// stable, with both ends anchored (pidTuningFRDNyquistCount).
func pidTuningFRDCertified(c PID2, p pidTuningPlant) bool {
	stable, known := pidTuningFRDNyquistCount(c, p)
	return stable && known
}

// pidTuningFRDDesign is TunePIDFRD's search on the looser plant base and,
// when certify, first on the strict plant: each design passes
// pidTuningFRDGate, the strict one first; at the default crossover, when
// neither does, pidTuningFRDMove lowers and raises it on the strict plant
// and then on base, as where the samples leave the strict count unknown
// for every loop near some crossover. A design of base carries the
// uncertified warning and the margins of its crossings along the strict
// plant's fitted arcs (pidTuningFRDArcCrossings), or fails when the
// strict count finds it unstable, at either end; a strict one whose ends
// are not anchored carries a warning too. Over the |S| bound, an
// explicit crossover keeps the strict design or else base's with a
// warning, and the default one fails. Evidence counts every search.
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
	finish := func(r *PIDTuningResult, p pidTuningPlant) (*PIDTuningResult, error) {
		r.Evidence.Evaluations = used
		switch {
		case !counted:
		case p.nyquistCount:
			if _, known := pidTuningFRDNyquistCount(*r.Controller, p); !known {
				var why []string
				if !p.lowAnchored {
					why = append(why, "the data have no sample at ω = 0 to anchor the DC end")
				}
				if _, _, top := pidTuningFRDNyquistVerdict(*r.Controller, p); !top {
					why = append(why, "they stop before the loop's end above the band is established")
				}
				r.Evidence.Warnings = append(r.Evidence.Warnings, fmt.Sprintf("The strict sampled Nyquist count passed this design, but %s, so its closed-loop stability is not certified.", strings.Join(why, " and ")))
			}
		default:
			stable, known, _ := pidTuningFRDNyquistVerdict(*r.Controller, strict)
			if !stable && known {
				ev := r.Evidence
				ev.Warnings = append(slices.Clone(ev.Warnings), "The looser count's design is one the strict sampled Nyquist count finds unstable, and the strict count certified none, so no design is returned.")
				return nil, &PIDTuningError{Op: "TunePIDFRD", Evidence: ev, unstable: true}
			}
			if certify {
				ws, ms := pidTuningFRDArcCrossings(strict.crossArcs, strict.samples, strict.at, *r.Controller)
				for i, m := range ms {
					if stable && known {
						ms[i] = math.Abs(m)
					}
					r.Evidence.AchievedPhaseMargin = math.Min(r.Evidence.AchievedPhaseMargin, ms[i])
				}
				r.Evidence.GainCrossovers, r.Evidence.PhaseMargins = append(r.Evidence.GainCrossovers, ws...), append(r.Evidence.PhaseMargins, ms...)
				r.Evidence.Warnings = append(r.Evidence.Warnings, "The strict sampled Nyquist count certified no design; this one passed only the looser count, which can accept unstable loops, so its closed-loop stability is not certified.")
			} else {
				r.Evidence.Warnings = append(r.Evidence.Warnings, "The data's low end fixes neither a DC gain nor an integrating or differentiating asymptote the strict sampled Nyquist count can anchor on, so this design passed only the looser count and its closed-loop stability is not certified.")
			}
		}
		return r, nil
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
			return finish(r, strict)
		}
		over, overPlant = r, strict
	}
	r, ok, err := pass(base)
	if err != nil && !errors.As(err, new(*PIDTuningError)) {
		return nil, err
	}
	if ok {
		return finish(r, base)
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
			return finish(m, strict)
		}
		if m, e = pidTuningFRDMove(ctx, base, family, o, &used); e != nil {
			return nil, e
		}
		if m != nil {
			return finish(m, base)
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
		return finish(over, overPlant)
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
// budget; used counts every search. A crossover where the data's gain is
// below 1e-15 or not finite, which tunePID rejects as an invalid argument
// the caller never gave, is skipped. A margin left 0 is 60° there, as a
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
				if g := p.at(wc); !pidFiniteComplex(g) || cmplx.Abs(g) < 1e-15 {
					continue
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

// pidTuningFRDSplitDC returns data without a sample at ω = 0, and that
// sample's response and true when it has one.
func pidTuningFRDSplitDC(data *FRD) (pos *FRD, dc complex128, ok bool) {
	if data.Omega[0] != 0 {
		return data, 0, false
	}
	return &FRD{Omega: data.Omega[1:], Response: data.Response[1:], Dt: data.Dt}, data.Response[0][0][0], true
}

// pidTuningFRDSampledDC returns the real DC gain of a sample at ω = 0, 0
// when it is at rounding of the sampled peak, and true when data have one.
func pidTuningFRDSampledDC(data *FRD) (float64, bool) {
	_, dc, ok := pidTuningFRDSplitDC(data)
	if !ok {
		return 0, false
	}
	peak := 0.
	for _, r := range data.Response {
		peak = math.Max(peak, cmplx.Abs(r[0][0]))
	}
	if cmplx.Abs(dc) <= 1e3*0x1p-52*peak {
		return 0, true
	}
	return real(dc), true
}

// pidTuningFRDPlant is the tuning plant of sampled data it owns, under the
// looser count: its DC gain, when unstable poles are not given, is that of
// a sample at ω = 0, or else pidTuningFRDDCGain's. Its band starts at the
// lowest sample above DC.
func pidTuningFRDPlant(data *FRD, unstablePoles *int) pidTuningPlant {
	full := data
	data, _, _ = pidTuningFRDSplitDC(data)
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
		dc, sampled := pidTuningFRDSampledDC(full)
		if !sampled {
			dc = pidTuningFRDDCGain(data, p.at)
		}
		p.dcGain, p.highGain = dc, p.at(high)
		p.dcEdge = complex(p.dcGain, 0)
	}
	p.arcs = pidTuningFRDArcs(data)
	return p
}

// pidTuningFRDStrictPlant is pidTuningFRDPlant under the strict count, with
// the DC gain of a sample at ω = 0, which anchors the count's low end
// (lowAnchored) when the band's low end shows it, as the DC gain
// pidTuningFRDStrictDCGain takes from there within a factor 1.4 of it, so
// no mode between DC and the band turns the loop unseen; or else with the
// DC gain pidTuningFRDStrictDCGain extrapolates,
// the low-end order pidTuningFRDLowOrder extrapolates when that is 0, the
// Markov end of continuous data and the high-frequency asymptote
// pidTuningFRDTopOrder takes. certify reports that the count can run: no
// unstable poles are given and a DC gain or low-end order anchors it,
// sampled or extrapolated.
func pidTuningFRDStrictPlant(data *FRD, unstablePoles *int) (p pidTuningPlant, certify bool) {
	p = pidTuningFRDPlant(data, unstablePoles)
	if unstablePoles != nil && *unstablePoles != 0 {
		return p, false
	}
	full := data
	data, _, _ = pidTuningFRDSplitDC(data)
	p.nyquistCount = true
	dc, sampled := pidTuningFRDSampledDC(full)
	if sampled && dc != 0 {
		x := pidTuningFRDStrictDCGain(data, p.at)
		p.dcGain, p.lowAnchored = dc, x/dc >= .7 && x/dc <= 1.4
	} else {
		p.dcGain = pidTuningFRDStrictDCGain(data, p.at)
	}
	p.dcEdge = complex(p.dcGain, 0)
	if p.dcGain == 0 {
		p.lowOrder, p.lowAnchor = pidTuningFRDLowOrder(data, p.at)
	}
	p.resonances = pidTuningFRDResonances(data)
	for k, modes := range p.resonances {
		trusted := pidTuningFRDTrusted(modes)
		if len(trusted) == 0 || p.arcs != nil && p.arcs[k] != nil {
			continue
		}
		if p.arcs == nil {
			p.arcs = make([][]pidTuningFRDMode, len(p.resonances))
		}
		p.arcs[k] = trusted
	}
	for k := 0; k+1 < len(data.Omega); k++ {
		fits, nf, span := pidTuningFRDStepFits(data, k)
		if nf < 2 || !(span > math.Pi/8) && (p.resonances == nil || len(p.resonances[k]) == 0) {
			continue
		}
		if p.crossArcs == nil {
			p.crossArcs = make([][]pidTuningFRDMode, len(data.Omega)-1)
		}
		p.crossArcs[k] = append([]pidTuningFRDMode(nil), fits[:nf]...)
	}
	p.unresolved = pidTuningFRDUnresolved(data)
	if data.Dt == 0 {
		p.markov, p.markovSpread = pidTuningFRDMarkov(data, p.at)
		p.markovBend = pidTuningFRDTopBend(data, -1)
		p.markovAnchored = p.markov != 0 && p.markovBend <= pidTuningFRDTopCurvature && pidTuningFRDMarkovSettles(data, p.at)
		p.topOrder, p.topAnchored = pidTuningFRDTopOrder(data, p.at)
	}
	return p, p.dcGain != 0 || p.lowAnchor
}

// pidTuningFRDTopOrder returns n of a high end G ≈ K·(jω)^n, K real and n
// negative, that continuous sampled data establish, with ok false when
// they do not: over each quarter decade of the top half decade the log
// slope of |G| must stay within 0.25 of n, and G/jⁿ within 30° of one side
// of the real axis, turning no further from it as ω rises, as a strictly
// proper end whose relative degree the data show leaves it. A zero or pole
// above the band bends that slope, so data stopping short of it show none:
// G/jⁿ must head to the axis over each quarter decade, not only from end
// to end, as the pull of a pole or zero below the band fades with ω while
// one above grows; a lightly damped pair just above the band turns the
// phase back long before the 0.25 bound on the slope trips.
func pidTuningFRDTopOrder(data *FRD, at func(float64) complex128) (n int, ok bool) {
	low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
	w1, wm := high/math.Sqrt(10), high/math.Pow(10, .25)
	if !(w1 >= low) {
		return 0, false
	}
	g1, gm, g2 := at(w1), at(wm), at(high)
	if g1 == 0 || gm == 0 || g2 == 0 {
		return 0, false
	}
	s1 := math.Log(cmplx.Abs(gm)/cmplx.Abs(g1)) / math.Log(wm/w1)
	s2 := math.Log(cmplx.Abs(g2)/cmplx.Abs(gm)) / math.Log(high/wm)
	r := math.Round(s2)
	if !(r <= -1 && math.Abs(s1-r) <= .25 && math.Abs(s2-r) <= .25) {
		return 0, false
	}
	j := cmplx.Pow(1i, complex(r, 0))
	h1, hm, h2 := g1/j, gm/j, g2/j
	if math.Abs(imag(h1)) <= .5*cmplx.Abs(h1) && math.Abs(imag(h2)) <= .5*cmplx.Abs(h2) && real(h1)*real(h2) > 0 && real(h1)*real(hm) > 0 && pidTuningFRDHeadsToAxis(hm, h1) && pidTuningFRDHeadsToAxis(h2, hm) && pidTuningFRDTopFlat(data, r) {
		return int(r), true
	}
	return 0, false
}

// pidTuningFRDTopFlat reports whether ln|G| − n·ln ω over the top half
// decade of continuous data shows no dynamics above the band: fitted
// exactly through the samples nearest four points even in ln ω over it as
// C + α/u + β·u + γ/u², u = (ω/high)², or, with fewer samples there,
// through three without γ, the terms α and γ of poles and zeros below the
// band fading with ω, |β| must stay within pidTuningFRDTopCurvature. A pole
// pair at ω_r above the band adds β ≈ (high/ω_r)², a zero or real pole at
// z ±(high/z)²/2: a lightly damped pair a few times above the band hides
// its phase, under half a degree at ζ 0.01, but not this bend in |G|,
// and lifts |L| past the band by 1/(2ζ).
func pidTuningFRDTopFlat(data *FRD, n float64) bool {
	return pidTuningFRDTopBend(data, n) <= pidTuningFRDTopCurvature
}

// pidTuningFRDTopBend is |β| of pidTuningFRDTopFlat, +Inf when the top
// half decade has under three distinct samples to fit.
func pidTuningFRDTopBend(data *FRD, n float64) float64 {
	w := data.Omega
	high := w[len(w)-1]
	near := func(x float64) int {
		k := sort.SearchFloat64s(w, x)
		if k > 0 && (k == len(w) || math.Log(x/w[k-1]) < math.Log(w[k]/x)) {
			k--
		}
		return k
	}
	pick := func(m int) []int {
		idx := make([]int, m)
		for i := range m {
			idx[i] = near(high * math.Pow(10, -.5*float64(m-1-i)/float64(m-1)))
			if i > 0 && idx[i] <= idx[i-1] {
				return nil
			}
		}
		return idx
	}
	idx := pick(4)
	if idx == nil {
		if idx = pick(3); idx == nil {
			return math.Inf(1)
		}
	}
	m := len(idx)
	a := make([][]float64, m)
	for i, k := range idx {
		u := (w[k] / high) * (w[k] / high)
		a[i] = append([]float64{1, 1 / u, u, 1 / (u * u)}[:m:m], math.Log(cmplx.Abs(data.Response[k][0][0]))-n*math.Log(w[k]/high))
	}
	for c := range m {
		p := c
		for r := c + 1; r < m; r++ {
			if math.Abs(a[r][c]) > math.Abs(a[p][c]) {
				p = r
			}
		}
		a[c], a[p] = a[p], a[c]
		for r := range m {
			if r != c {
				f := a[r][c] / a[c][c]
				for k := c; k <= m; k++ {
					a[r][k] -= f * a[c][k]
				}
			}
		}
	}
	return math.Abs(a[2][m] / a[2][2])
}

// pidTuningFRDTopCurvature bounds β of pidTuningFRDTopFlat: a pole pair
// within about ten times the band's top, or a zero or pole within seven,
// leaves the top unestablished.
const pidTuningFRDTopCurvature = .01

// pidTuningFRDIdealEndSettled reports whether an ideal derivative's loop
// end L(∞) = Kd·m on an anchored Markov end, of top bend |β|
// (pidTuningFRDTopBend), is established: |L(∞)| ≤ ½, or β within
// pidTuningFRDTopCurvature/3, no pole pair within about 17 times the top.
// L does not roll off, so a pair beyond the ten-times zone of
// pidTuningFRDTopFlat still lifts |L(∞)| by 1/(2ζ) through 1 there, where
// the data read no crossing: a ζ 0.07 pair 14 times above the top
// (β 0.0052) put |L(∞)| 0.83 at a 13.6° margin.
func pidTuningFRDIdealEndSettled(end, bend float64) bool {
	return math.Abs(end) <= .5 || bend <= pidTuningFRDTopCurvature/3
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

// pidTuningFRDMarkovSettles reports whether jω·G of continuous sampled
// data heads to the real axis over each quarter decade of the top half
// decade and across it (pidTuningFRDHeadsToAxis), as an end whose
// dynamics all lie below the band does. A zero and pole above the band,
// whose bends in |G| nearly cancel in pidTuningFRDTopFlat, still turn
// jω·G away from the axis as ω rises, and leave lim sG beyond the top
// sample's: at 2.8 and 3.1 times the top, 8% further.
func pidTuningFRDMarkovSettles(data *FRD, at func(float64) complex128) bool {
	low, high := data.Omega[0], data.Omega[len(data.Omega)-1]
	w1, wm := high/math.Sqrt(10), high/math.Pow(10, .25)
	if !(w1 >= low) {
		return false
	}
	h1, hm, h2 := complex(0, w1)*at(w1), complex(0, wm)*at(wm), complex(0, high)*at(high)
	return pidTuningFRDHeadsToAxis(hm, h1) && pidTuningFRDHeadsToAxis(h2, hm) && pidTuningFRDHeadsToAxis(h2, h1)
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
// 90°; one that cannot be rejects c, as its |S| is unbounded. Between two
// consecutive samples the count is unknown when 0 lies between their chord
// of 1 + L and its arc on the circle through them and the sample before,
// or that arc, the circle's major one, passes within ¼ of 0, and likewise
// on the one through them and the sample after (where each exists), as a
// resonance the samples do not resolve winds 1 + L round 0 along that
// arc, opposite to the chord. On the strict plant, across a step that
// pidTuningFRDResonances marks, 1 + L follows each fitted resonance
// instead of the chord, the count unknown unless they agree on its turn,
// as two that differ by a whole turn leave the loop's encirclements to
// samples that do not fix them; a marked step with no fit leaves it
// unknown unless |L| < ¼ at both its samples, a loop small enough there
// for the chord. On a step pidTuningFRDUnresolved marks, a pole pair the
// samples miss can carry 1 + L clockwise round any point left of its chord,
// however small |L| is at the samples, adding a whole clockwise turn, so a
// count finding c stable is unknown when 0 lies there; one finding it
// unstable stays so, as such turns only add unstable roots. It is unknown
// too when a fit across such a step (pidTuningFRDStepFits) meets |L| = 1
// or carries |S| past pidTuningMaxSensitivity between the samples: the
// samples do not fix the pole's σ, a smaller one lifts that peak further,
// and the margins and |S| there are the data's to leave open.
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
	stable, known, _ := pidTuningFRDNyquistVerdict(c, p)
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
	slope := math.Log(cmplx.Abs(gHigh)/cmplx.Abs(g2)) / math.Log(high/w2)
	if c.Dt > 0 {
		slope = math.Max(slope, 0)
	} else if c.Tf == 0 && c.Kd != 0 && !(slope <= -1.5) {
		return false
	}
	return pidTuningFRDTopBoundedSlope(c, high, gHigh, slope, limit)
}

// pidTuningFRDTopBoundedSlope is pidTuningFRDTopBounded with |G| taken at
// the given log slope past high, whose end is ten times high (continuous)
// or Nyquist (discrete).
func pidTuningFRDTopBoundedSlope(c PID2, high float64, gHigh complex128, slope, limit float64) bool {
	end := 10 * high
	if c.Dt > 0 {
		end = math.Pi / c.Dt
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
// the data leave an end of 1 + L unknown or nothing anchors the count. On
// the strict plant a loop the count finds stable is known only when both
// ends anchor: the DC gain is sampled at ω = 0 (lowAnchored), and the end
// above the band is one the data reach, at Nyquist (discrete) or on an
// established asymptote (lim sG, or topAnchored with |L| bounded at its
// slope), as data stopping short of either leave the loop's ends
// extrapolated. A loop it finds unstable stays known.
func pidTuningFRDNyquistCount(c PID2, p pidTuningPlant) (stable, known bool) {
	stable, known, top := pidTuningFRDNyquistVerdict(c, p)
	if stable && known && p.nyquistCount && !(p.lowAnchored && top) {
		return true, false
	}
	return stable, known
}

// pidTuningFRDNyquistVerdict is the count without the strict plant's
// anchoring, with top reporting whether the end above the band it took is
// one the data reach.
func pidTuningFRDNyquistVerdict(c PID2, p pidTuningPlant) (stable, known, top bool) {
	if p.dcGain == 0 && !p.lowAnchor {
		return true, false, false
	}
	if p.nyquistCount && c.Dt > 0 && c.Tf == 0 && c.Kd != 0 && c.DFormula == ForwardEuler {
		return true, false, false
	}
	k := c.Tf + c.Dt*pidFormulaWeight(c.DFormula)
	if c.Dt > 0 && c.Kd != 0 && k != 0 && k < c.Dt/2 {
		return false, true, false
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
			return false, true, false
		}
		end, bump, top = complex(0, s), -math.Pi/2, true
	case nyquist && nearReal:
		e1, e2 := math.Pi/c.Dt-p.high, math.Pi/c.Dt-w2
		r1, r2 := real(l), real(l2)
		r0 := (r1*e2*e2 - r2*e1*e1) / (e2*e2 - e1*e1)
		if !(math.Abs(1+r0) > .5*math.Abs(r0-r1)) {
			return false, true, false
		}
		end, top = complex(1+r0, 0), true
	case !p.nyquistCount:
		switch {
		case cmplx.Abs(l) < 1:
		case nyquist:
			return true, false, false
		case !nearReal || c.Dt > 0:
			if !falling || !(cmplx.Abs(l) < 2) {
				return true, false, false
			}
		case cmplx.Abs(1+l) < .1:
			return false, true, false
		default:
			end = complex(1+real(l), 0)
		}
	case c.Dt == 0 && c.Tf == 0 && c.Kd != 0 && p.markov != 0:
		end, top = complex(1+c.Kd*p.markov, 0), p.markovAnchored && pidTuningFRDIdealEndSettled(c.Kd*p.markov, p.markovBend)
		if !(math.Abs(real(end)) > math.Abs(real(1+l-end))+2*math.Abs(c.Kd)*p.markovSpread) {
			return true, false, false
		}
	default:
		limit := 1.
		if cmplx.Abs(l) < 1 && !falling {
			limit = .5
		}
		if nyquist && !(cmplx.Abs(l) < 1) || !(cmplx.Abs(l) < 2) || !pidTuningFRDTopBounded(c, p.high, w2, p.highGain, p.at(w2), limit) {
			return true, false, false
		}
		top = nyquist || c.Dt == 0 && p.topAnchored && pidTuningFRDTopBoundedSlope(c, p.high, p.highGain, float64(p.topOrder), limit)
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
			return true, false, false
		}
		w0 = 1e-3 * p.low * ratio
	} else if c.Ki != 0 {
		start, expected = complex(0, -c.Ki*p.dcGain), math.Pi/2
		w0 = 1e-3 * math.Min(p.low, math.Abs(c.Ki*p.dcGain)/(1+math.Abs(c.Kp*p.dcGain)))
	}
	if start == 0 || !(w0 > 0) {
		return false, true, false
	}
	if c.Ki == 0 && !p.lowAnchor && !(math.Abs(real(start)) > math.Abs(c.Kp)*math.Abs(real(p.at(math.Min(p.low*math.Pow(10, .25), p.high)))-p.dcGain)) {
		return false, true, false
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
		return false, true, false
	}
	var last float64
	step := func(w float64, f complex128) bool {
		d, ok := walk(wPrev, w, fPrev, f, 0)
		last = d
		total += d
		wPrev, fPrev = w, f
		return ok
	}
	for w := w0 * math.Sqrt(10); w < p.low; w *= math.Sqrt(10) {
		if !step(w, at(w)) {
			return false, true, false
		}
	}
	s := p.samples
	hidden := false
	var f0, f1, f2, f3 complex128
	for i, w := range s {
		f0, f1, f2 = f1, f2, f3
		if i == 0 {
			f2 = at(w)
		}
		if i+1 < len(s) {
			f3 = at(s[i+1])
		}
		if w > wPrev && !step(w, f2) {
			return false, true, false
		}
		if i > 0 && p.unresolved != nil && p.unresolved[i-1] != nil {
			if len(p.unresolved[i-1]) == 0 && pidTuningFRDModeLifts(p, i-1) && !(math.Max(cmplx.Abs(f1-1), cmplx.Abs(f2-1)) < .25) {
				return true, false, false
			}
			u := f2 - f1
			hidden = hidden || real(u)*imag(-f1)-imag(u)*real(-f1) > 0 || pidTuningFRDFitsReach(p.unresolved[i-1], c, s[i-1], w, pidTuningFRDModeLifts(p, i-1))
		}
		if i > 0 && p.resonances != nil && p.resonances[i-1] != nil {
			if len(p.resonances[i-1]) == 0 && !(math.Max(cmplx.Abs(f1-1), cmplx.Abs(f2-1)) < .25) {
				return true, false, false
			}
			ref := last
			for j, m := range p.resonances[i-1] {
				d, ok := pidTuningFRDModeTurn(c, m, s[i-1], w)
				if j == 0 {
					ref = d
				}
				if !ok || !(math.Abs(d-ref) < math.Pi/2) {
					return true, false, false
				}
			}
			total += ref - last
		}
		if i > 0 && (i > 1 || i+1 < len(s)) && (i < 2 || pidTuningFRDArcHidesZero(f1, f2, f0)) && (i+1 == len(s) || pidTuningFRDArcHidesZero(f1, f2, f3)) {
			return true, false, false
		}
	}
	if fPrev == 0 {
		return false, true, false
	}
	total += turn(fPrev, end) + bump
	if stable := math.Abs(expected-total) < math.Pi/2; !stable || !hidden {
		return stable, true, top
	}
	return true, false, false
}

// pidTuningFRDMode is a lightly damped pole pair's interpolant
// G(ω) = c + r/(ω − pole) across one step of the samples, through both,
// pole = ω₀ + jσ; peaked reports that the middle of the three samples
// its pole was fitted through is above both others, which straddle the
// peak, so the far dynamics skew that fit least.
type pidTuningFRDMode struct {
	pole, c, r complex128
	peaked     bool
}

// pidTuningFRDResonances returns, for each step samples[k] → samples[k+1]
// of data (above DC) that a resonance the samples do not resolve may span,
// the interpolants of that resonance across it, and nil for any other
// step, or nil when no step is marked. Near a lightly damped pole pair G
// is a Möbius map of ω, c + r/(ω − ω₀ − jσ), tracing a circle clockwise,
// so the step's two samples and either neighbour fix its pole; each fit
// with σ > 0 gives an interpolant through the step's samples. A step is
// marked when a fit's circle turns across it by more than a quarter turn,
// beyond what the count's chord follows, or when arg G does while |G|
// peaks there, the larger of its two samples above both outer neighbours,
// as a pole pair between them lifts both, while a delay's phase turns
// with |G| smooth and an antiresonance dips. Both samples more than 4
// times above both outer neighbours, with arg G turning by at most a
// quarter turn, mark a step with no fit: one pole pair there turns G by
// about a half turn, so the samples straddle two or more whose turn wraps.
// A marked step with no fit, as also when a sample sits on the peak and
// the far dynamics skew both fits, is the count's to bound by |L| alone.
func pidTuningFRDResonances(data *FRD) [][]pidTuningFRDMode {
	n := len(data.Omega)
	g := func(k int) complex128 { return data.Response[k][0][0] }
	var out [][]pidTuningFRDMode
	for k := 0; k+1 < n; k++ {
		a, b := cmplx.Abs(g(k)), cmplx.Abs(g(k+1))
		peak, turn := math.Max(a, b), math.Abs(cmplx.Phase(g(k+1)/g(k)))
		marked := turn > math.Pi/2 && !(k > 0 && !(peak > cmplx.Abs(g(k-1))) || k+2 < n && !(peak > cmplx.Abs(g(k+2))))
		wrapped := k > 0 && k+2 < n && !(turn > math.Pi/2) && math.Min(a, b) > 4*math.Max(cmplx.Abs(g(k-1)), cmplx.Abs(g(k+2)))
		fits, nf, span := pidTuningFRDStepFits(data, k)
		if !marked && !wrapped && !(span > math.Pi/2) {
			continue
		}
		if out == nil {
			out = make([][]pidTuningFRDMode, n-1)
		}
		if wrapped {
			out[k] = []pidTuningFRDMode{}
			continue
		}
		out[k] = append([]pidTuningFRDMode{}, fits[:nf]...)
	}
	return out
}

// pidTuningFRDUnresolved marks, by step samples[k] → samples[k+1] of data
// (above DC), the steps across which a lightly damped pole pair the samples
// do not resolve may lie, each with its fits (pidTuningFRDStepFits), none
// for an unmarked step, or returns nil when there is none: arg G turns by
// more than a quarter turn across the step, as across a pole pair, or both
// its samples' |G| exceed both outer neighbours', as about a peak that two
// pole pairs in one step, turning G by a whole turn, leave; a step whose
// two Möbius fits agree on the pole within the smaller σ
// (pidTuningFRDTrusted) resolves it and is not marked.
func pidTuningFRDUnresolved(data *FRD) [][]pidTuningFRDMode {
	n := len(data.Omega)
	g := func(k int) float64 { return cmplx.Abs(data.Response[k][0][0]) }
	var out [][]pidTuningFRDMode
	for k := 0; k+1 < n; k++ {
		turn := math.Abs(cmplx.Phase(data.Response[k+1][0][0] / data.Response[k][0][0]))
		peak := k > 0 && k+2 < n && math.Min(g(k), g(k+1)) > math.Max(g(k-1), g(k+2))
		if !(turn > math.Pi/2) && !peak {
			continue
		}
		fits, nf, _ := pidTuningFRDStepFits(data, k)
		if len(pidTuningFRDTrusted(fits[:nf])) == 2 {
			continue
		}
		if out == nil {
			out = make([][]pidTuningFRDMode, n-1)
		}
		out[k] = append([]pidTuningFRDMode{}, fits[:nf]...)
	}
	return out
}

// pidTuningFRDModeLifts reports whether step k of p's samples, marked
// unresolved, may hold a pole pair: arg G turns past a quarter turn across
// it, or its samples' |G| exceed both outer neighbours' by more than
// √min(ρ, 1 + 1/ρ), ρ the step's ratio. A pole pair at ω₀ inside the step
// lifts them by at least min(ρ, 1 + 1/ρ) (|G| ∝ 1/|ω − ω₀|, ω₀ at an end),
// while the two samples about a smooth maximum differ from their
// neighbours only to second order in log ρ.
func pidTuningFRDModeLifts(p pidTuningPlant, k int) bool {
	s := p.samples
	a, b := p.at(s[k]), p.at(s[k+1])
	if math.Abs(cmplx.Phase(b/a)) > math.Pi/2 || k == 0 || k+2 >= len(s) {
		return true
	}
	rho := s[k+1] / s[k]
	lift := math.Min(cmplx.Abs(a), cmplx.Abs(b)) / math.Max(cmplx.Abs(p.at(s[k-1])), cmplx.Abs(p.at(s[k+2])))
	return !(lift <= math.Sqrt(math.Min(rho, 1+1/rho)))
}

// pidTuningFRDMinDamping is the least ζ of a pole pair an unresolved step
// is taken to hide (pidTuningFRDFitsReach).
const pidTuningFRDMinDamping = 1e-3

// pidTuningFRDFitsReach reports whether c's loop along any of fits over
// [a, b] meets |L| = 1 or reaches |S| > pidTuningMaxSensitivity; one
// staying above |L| = 1 throughout keeps 1 + L clear of 0 there. When the
// step may hold a pole pair (lifts, pidTuningFRDModeLifts), a fit whose
// pole lies in the step and whose |L| stays below 1 reaches too when
// a σ shrunk by that peak |L|, lifting the peak to about 1, still leaves
// the pair damped by pidTuningFRDMinDamping or more: the samples do not
// fix σ, so such a fit bounds neither the crossings nor their margins
// there.
func pidTuningFRDFitsReach(fits []pidTuningFRDMode, c PID2, a, b float64, lifts bool) bool {
	for _, m := range fits {
		l := func(w float64) complex128 { return m.at(w) * pidTuningFeedback(c, w) }
		_, hi := m.max(func(w float64) float64 { return cmplx.Abs(l(w)) }, a, b)
		_, lo := m.max(func(w float64) float64 { return -cmplx.Abs(l(w)) }, a, b)
		if !(hi < 1) && !(-lo > 1) {
			return true
		}
		if w0 := real(m.pole); lifts && hi < 1 && w0 >= a && w0 <= b && imag(m.pole)*hi >= pidTuningFRDMinDamping*w0 {
			return true
		}
		if _, v := m.max(func(w float64) float64 { return 1 / cmplx.Abs(1+l(w)) }, a, b); !(v <= pidTuningMaxSensitivity) {
			return true
		}
	}
	return false
}

// pidTuningFRDModeThrough is the interpolant c + r/(ω − pole) through
// (wa, ga) and (wb, gb), and whether it is finite.
func pidTuningFRDModeThrough(pole complex128, wa, wb float64, ga, gb complex128) (pidTuningFRDMode, bool) {
	da, db := 1/(complex(wa, 0)-pole), 1/(complex(wb, 0)-pole)
	r := (ga - gb) / (da - db)
	m := pidTuningFRDMode{pole: pole, c: ga - r*da, r: r}
	return m, pidFiniteComplex(m.c) && pidFiniteComplex(m.r)
}

// pidTuningFRDStepFits returns the interpolants across step samples[k] →
// samples[k+1] of data (above DC) that the Möbius fits through its two
// samples and either neighbour give, those with σ > 0 whose ω₀ lies within
// σ of the three samples' span, as a pole fitted beyond them is the far
// dynamics' extrapolation, not a mode the samples see, and the largest
// turn of a fit's circle across the step, 0 without one.
func pidTuningFRDStepFits(data *FRD, k int) (fits [2]pidTuningFRDMode, n int, span float64) {
	w := data.Omega
	g := func(k int) complex128 { return data.Response[k][0][0] }
	for _, j := range []int{k - 1, k + 2} {
		if j < 0 || j >= len(w) {
			continue
		}
		pole := pidTuningFRDMobiusPole(w[k], w[k+1], w[j], g(k), g(k+1), g(j))
		if !pidFiniteComplex(pole) || !(imag(pole) > 0) || real(pole) < w[min(j, k)]-imag(pole) || real(pole) > w[max(j, k+1)]+imag(pole) {
			continue
		}
		span = math.Max(span, 2*(math.Atan((w[k+1]-real(pole))/imag(pole))-math.Atan((w[k]-real(pole))/imag(pole))))
		if m, ok := pidTuningFRDModeThrough(pole, w[k], w[k+1], g(k), g(k+1)); ok {
			mid := k
			if j > k {
				mid = k + 1
			}
			m.peaked = cmplx.Abs(g(mid)) > cmplx.Abs(g(mid-1)) && cmplx.Abs(g(mid)) > cmplx.Abs(g(mid+1))
			fits[n] = m
			n++
		}
	}
	return fits, n, span
}

// pidTuningFRDArcs returns, by step as pidTuningFRDResonances, the fitted
// interpolants (pidTuningFRDStepFits) of each step of data (above DC)
// whose fitted circle turns past π/8 across it, those pidTuningFRDTrusted
// keeps, or nil when there is none: there the samples resolve a lightly
// damped mode only coarsely, and the chord between them cuts inside its
// circle, missing an |S| peak between them that the fit follows.
func pidTuningFRDArcs(data *FRD) [][]pidTuningFRDMode {
	var out [][]pidTuningFRDMode
	for k := 0; k+1 < len(data.Omega); k++ {
		fits, nf, span := pidTuningFRDStepFits(data, k)
		trusted := pidTuningFRDTrusted(fits[:nf])
		if !(span > math.Pi/8) || len(trusted) == 0 {
			continue
		}
		if out == nil {
			out = make([][]pidTuningFRDMode, len(data.Omega)-1)
		}
		out[k] = append([]pidTuningFRDMode(nil), trusted...)
	}
	return out
}

// pidTuningFRDTrusted returns the fits of one step a reading of |S| may
// follow: both when they agree on the pole within the
// smaller σ, or else the one, if any, fitted through samples straddling
// the peak (peaked), as two modes about one step, or the far dynamics,
// skew the others into a resonance the plant does not have.
func pidTuningFRDTrusted(fits []pidTuningFRDMode) []pidTuningFRDMode {
	if len(fits) == 2 && cmplx.Abs(fits[0].pole-fits[1].pole) <= math.Min(imag(fits[0].pole), imag(fits[1].pole)) {
		return fits
	}
	for i, m := range fits {
		if m.peaked {
			return fits[i : i+1]
		}
	}
	return nil
}

// pidTuningFRDArcPeak returns the largest |S| of c along the fitted
// interpolants arcs of the steps of samples (pidTuningFRDArcs), each read
// at 16 points even in the angle its pole sees and refined about its
// largest by golden section, or 0 without arcs.
func pidTuningFRDArcPeak(arcs [][]pidTuningFRDMode, samples []float64, c PID2) float64 {
	peak := 0.
	for k, modes := range arcs {
		for _, m := range modes {
			s := func(w float64) float64 {
				v := 1 / cmplx.Abs(1+m.at(w)*pidTuningFeedback(c, w))
				if math.IsNaN(v) {
					return math.Inf(1)
				}
				return v
			}
			_, v := m.max(s, samples[k], samples[k+1])
			peak = math.Max(peak, v)
		}
	}
	return peak
}

// at is G at ω along m's interpolant.
func (m pidTuningFRDMode) at(w float64) complex128 { return m.c + m.r/(complex(w, 0)-m.pole) }

// turn is the angle m's circle turns through from a to b.
func (m pidTuningFRDMode) turn(a, b float64) float64 {
	w0, sigma := real(m.pole), imag(m.pole)
	return 2 * (math.Atan((b-w0)/sigma) - math.Atan((a-w0)/sigma))
}

// max returns the largest f along m's interpolant over [a, b] and its ω:
// f is read at 16 points even in the angle m's pole sees and refined about
// its largest by golden section.
func (m pidTuningFRDMode) max(f func(float64) float64, a, b float64) (float64, float64) {
	w0, sigma := real(m.pole), imag(m.pole)
	ta, tb := math.Atan((a-w0)/sigma), math.Atan((b-w0)/sigma)
	const pieces = 16
	at := func(i int) float64 {
		return math.Min(math.Max(w0+sigma*math.Tan(ta+(tb-ta)*float64(i)/pieces), a), b)
	}
	best, bw, bv := 0, a, math.Inf(-1)
	for i := 0; i <= pieces; i++ {
		w := at(i)
		if v := f(w); v > bv {
			best, bw, bv = i, w, v
		}
	}
	if w, g := pidTuningGoldenMax(f, at(max(best-1, 0)), at(min(best+1, pieces))); g > bv {
		return w, g
	}
	return bw, bv
}

// pidTuningFRDArcCrossings returns c's gain crossings along the fitted
// interpolants arcs (crossArcs) of the steps of samples, as the chord
// between two samples cuts inside a lightly damped mode's circle: where
// sampled |L| (at) is below 1 at both ends a resonance can lift it over 1
// between them, a crossover pair the chord misses, sought either side of
// the fit's peak; where |L| crosses 1 across a step the fit's circle turns
// past a quarter turn over, the chord's phase there can be tens of degrees
// off the fit's. A step counts only with both its fits, through either
// neighbour, and each fit's crossings count, the margins, wrapped as
// pidTuningMargins, taken at their least: a lone fit across two modes can
// put a crossing's phase 20° off, and near a mode of σ under 1% of ω₀ the
// far dynamics skew the two fits apart by more than σ, so that one reads a
// pair the other misses or margins 30° apart that bracket the plant's.
func pidTuningFRDArcCrossings(arcs [][]pidTuningFRDMode, samples []float64, at func(float64) complex128, c PID2) (crossings, margins []float64) {
	for k, modes := range arcs {
		a, b := samples[k], samples[k+1]
		if len(modes) < 2 {
			continue
		}
		ea, eb := cmplx.Abs(at(a)*pidTuningFeedback(c, a))-1, cmplx.Abs(at(b)*pidTuningFeedback(c, b))-1
		if !(ea < 0) && !(eb < 0) {
			continue
		}
		for _, m := range modes {
			l := func(w float64) complex128 { return m.at(w) * pidTuningFeedback(c, w) }
			cross := func(lo, hi, lowSign float64) {
				for range 40 {
					mid := math.Sqrt(lo * hi)
					if v := cmplx.Abs(l(mid)) - 1; lowSign*v <= 0 {
						hi = mid
					} else {
						lo = mid
					}
				}
				w := math.Sqrt(lo * hi)
				pm := 180 + cmplx.Phase(l(w))*180/math.Pi
				if pm > 180 {
					pm -= 360
				}
				crossings, margins = append(crossings, w), append(margins, pm)
			}
			switch {
			case ea < 0 && eb < 0:
				if wp, v := m.max(func(w float64) float64 { return cmplx.Abs(l(w)) }, a, b); v > 1 {
					cross(a, wp, -1)
					cross(wp, b, 1)
				}
			case ea*eb < 0 && m.turn(a, b) > math.Pi/2:
				cross(a, b, ea)
			}
		}
	}
	return crossings, margins
}

// pidTuningFRDMobiusPole is the pole of the Möbius map
// G(ω) = (αω + β)/(ω − pole) through (ωᵢ, gᵢ), i = 1..3.
func pidTuningFRDMobiusPole(w1, w2, w3 float64, g1, g2, g3 complex128) complex128 {
	det := func(a, b, c [3]complex128) complex128 {
		return a[0]*(b[1]*c[2]-b[2]*c[1]) - a[1]*(b[0]*c[2]-b[2]*c[0]) + a[2]*(b[0]*c[1]-b[1]*c[0])
	}
	x1, x2, x3 := complex(w1, 0), complex(w2, 0), complex(w3, 0)
	return det([3]complex128{x1, 1, g1 * x1}, [3]complex128{x2, 1, g2 * x2}, [3]complex128{x3, 1, g3 * x3}) / det([3]complex128{x1, 1, g1}, [3]complex128{x2, 1, g2}, [3]complex128{x3, 1, g3})
}

// pidTuningFRDModeTurn is the turn of 1 + L along mode m's interpolant from
// wa to wb, and whether it is followed: the step is walked evenly in the
// angle the pole sees, each piece bisected until 1 + L turns by at most π/8
// over it. A loop passing near 0 along it is the |S| gate's, which reads
// the same interpolant (p.arcs), not the count's: an unknown count there
// would stop the search ranking such loops by their |S| excess.
func pidTuningFRDModeTurn(c PID2, m pidTuningFRDMode, wa, wb float64) (float64, bool) {
	w0, sigma := real(m.pole), imag(m.pole)
	f := func(th float64) complex128 {
		w := w0 + sigma*math.Tan(th)
		return 1 + (m.c+m.r/(complex(w, 0)-m.pole))*pidTuningFeedback(c, w)
	}
	near := func(v complex128) bool { return !pidFiniteComplex(v) || v == 0 }
	var walk func(ta, tb float64, fa, fb complex128, depth int) (float64, bool)
	walk = func(ta, tb float64, fa, fb complex128, depth int) (float64, bool) {
		d := cmplx.Phase(fb / fa)
		if math.Abs(d) <= math.Pi/8 {
			return d, true
		}
		if depth == 30 {
			return 0, false
		}
		tm := (ta + tb) / 2
		fm := f(tm)
		if near(fm) {
			return 0, false
		}
		d1, ok := walk(ta, tm, fa, fm, depth+1)
		if !ok {
			return 0, false
		}
		d2, ok := walk(tm, tb, fm, fb, depth+1)
		return d1 + d2, ok
	}
	ta, tb := math.Atan((wa-w0)/sigma), math.Atan((wb-w0)/sigma)
	const pieces = 32
	t0, f0 := ta, f(ta)
	if near(f0) {
		return 0, false
	}
	total := 0.
	for i := 1; i <= pieces; i++ {
		t1 := ta + (tb-ta)*float64(i)/pieces
		f1 := f(t1)
		if near(f1) {
			return 0, false
		}
		d, ok := walk(t0, t1, f0, f1, 0)
		if !ok {
			return 0, false
		}
		total += d
		t0, f0 = t1, f1
	}
	return total, true
}

// pidTuningFRDArcHidesZero reports whether 0 lies between the chord from a
// to b and the arc from a to b, not through x, of the circle through a, b
// and x, or that arc, when it is the circle's major one (1 + L sweeping
// most of a resonance loop between two samples), passes within ¼ of 0
// (|S| > 4), as a lightly damped mode traces such a circle and the count's
// interpolant follows the chord; the margin keeps the search from walking
// to a design whose 0 sits on the fitted circle, which the fit cannot
// place on either side.
func pidTuningFRDArcHidesZero(a, b, x complex128) bool {
	cross := func(u, v complex128) float64 { return real(u)*imag(v) - imag(u)*real(v) }
	u, v := b-a, x-a
	den := 2 * cross(u, v)
	if den == 0 || !pidFiniteComplex(u) || !pidFiniteComplex(v) {
		return false
	}
	nu, nv := real(u)*real(u)+imag(u)*imag(u), real(v)*real(v)+imag(v)*imag(v)
	o := complex(imag(v)*nu-imag(u)*nv, real(u)*nv-real(v)*nu) / complex(den, 0)
	r, d := cmplx.Abs(o), cmplx.Abs(-a-o)
	if d < r && cross(u, -a)*cross(u, v) < 0 {
		return true
	}
	if !(math.Abs(d-r) < .25) || d == 0 || cross(u, o)*cross(u, v) >= 0 {
		return false
	}
	return cross(u, o+(-a-o)*complex(r/d, 0))*cross(u, v) < 0
}
