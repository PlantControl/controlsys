package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"strings"
)

// ErrPIDTuningTargetUnattainable means no controller passed the requested target
// and stability checks within the bounded search. It is not proof of impossibility.
var ErrPIDTuningTargetUnattainable = errors.New("PID tuning target unattainable within search bounds")

type PIDDesignFocus string

const (
	PIDFocusBalanced  PIDDesignFocus = "balanced"
	PIDFocusTracking  PIDDesignFocus = "tracking"
	PIDFocusRejection PIDDesignFocus = "rejection"
)

// PIDTuningWeights enables two-degree-of-freedom tuning. A nil field is free in
// [0,1]; a nonnil field is fixed exactly. Nil options.Weights means b=c=1.
type PIDTuningWeights struct{ FixedB, FixedC *float64 }

type PIDTuningOptions struct {
	CrossoverFrequency float64
	PhaseMargin        float64
	Focus              PIDDesignFocus
	IFormula, DFormula PIDFormula
	Weights            *PIDTuningWeights
	MaxEvaluations     int
	// UnstablePoles is optional user knowledge for FRD; nil means unknown.
	UnstablePoles *int
}

type PIDTuningObjective struct{ Tracking, Rejection, Effort float64 }
type PIDTuningEvidence struct {
	RequestedCrossover, RequestedPhaseMargin float64
	AchievedCrossover, AchievedPhaseMargin   float64
	GainCrossovers, PhaseMargins             []float64
	FrequencyBand                            [2]float64
	Objective, SeedObjective                 float64
	Components, Normalizers                  PIDTuningObjective
	Evaluations                              int
	Termination                              string
	// Stability is "closed-loop-poles" or "sampled-frequency-only". The latter
	// is deliberately not a global stability certificate.
	Stability string
	Warnings  []string
	Feasible  bool
}
type PIDTuningResult struct {
	Controller *PID2
	Evidence   PIDTuningEvidence
}

type pidTuningPlant struct {
	dt, low, high float64
	at            func(float64) complex128
	stable        func(*PID2) bool
	stability     string
	warnings      []string
}

// TunePID designs a SISO negative-feedback controller. The original Pidtune API
// retains its historical behavior. This operation enforces finite targets,
// bounded computation and explicit achieved-target/stability evidence.
func TunePID(ctx context.Context, plant *System, family PidtuneType, opts PIDTuningOptions) (*PIDTuningResult, error) {
	if ctx == nil {
		return nil, fmt.Errorf("TunePID: nil context")
	}
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if _, err := newSISOLoopModel(plant, "TunePID"); err != nil {
		return nil, err
	}
	wc := opts.CrossoverFrequency
	if wc == 0 {
		var err error
		wc, err = findCrossoverFreq(plant, 0)
		if err != nil {
			return nil, err
		}
		opts.CrossoverFrequency = wc
	}
	eval, err := newSISOEval(plant)
	if err != nil {
		return nil, err
	}
	p := pidTuningPlant{dt: plant.Dt, low: wc / 100, high: wc * 100, at: eval.at, stability: "sampled-frequency-only"}
	if plant.Dt > 0 {
		p.high = math.Min(p.high, .99*math.Pi/plant.Dt)
	}
	if !plant.HasDelay() {
		p.stability = "closed-loop-poles"
		p.stable = func(c *PID2) bool {
			controller := NewPID(c.Kp, c.Ki, c.Kd, WithFilter(c.Tf), WithPIDFormulas(c.IFormula, c.DFormula))
			controller.Dt = c.Dt
			cs, err := controller.System()
			if err != nil {
				if c.Kd != 0 && c.Tf == 0 {
					return pidTuningIdealStable(plant, c)
				}
				return false
			}
			loop, err := Feedback(plant, cs, -1)
			if err != nil {
				return false
			}
			stable, err := loop.IsStable()
			return err == nil && stable
		}
	} else {
		p.warnings = []string{"Exact delay retained; finite frequency samples do not certify global closed-loop stability."}
	}
	return tunePID(ctx, p, family, opts)
}

func tunePID(ctx context.Context, p pidTuningPlant, family PidtuneType, o PIDTuningOptions) (*PIDTuningResult, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	if !finitePID(o.CrossoverFrequency) || o.CrossoverFrequency <= 0 {
		return nil, fmt.Errorf("TunePID: crossover must be positive and finite")
	}
	if !finitePID(p.dt) || p.dt < 0 {
		return nil, fmt.Errorf("TunePID: sample time must be finite and nonnegative")
	}
	wc := o.CrossoverFrequency
	if p.dt > 0 && wc >= math.Pi/p.dt {
		return nil, fmt.Errorf("TunePID: crossover must be below Nyquist")
	}
	if wc <= p.low || wc >= p.high {
		return nil, fmt.Errorf("TunePID: crossover needs frequency coverage on both sides")
	}
	if o.PhaseMargin == 0 {
		o.PhaseMargin = 60
	}
	if !finitePID(o.PhaseMargin) || o.PhaseMargin <= 0 || o.PhaseMargin >= 180 {
		return nil, fmt.Errorf("TunePID: phase margin must be between 0 and 180 degrees")
	}
	if o.Focus == "" {
		o.Focus = PIDFocusBalanced
	}
	tracking := .5
	switch o.Focus {
	case PIDFocusTracking:
		tracking = .95
	case PIDFocusRejection:
		tracking = .05
	case PIDFocusBalanced:
	default:
		return nil, fmt.Errorf("TunePID: unknown focus %q", o.Focus)
	}
	if o.IFormula < ForwardEuler || o.IFormula > Trapezoidal || o.DFormula < ForwardEuler || o.DFormula > Trapezoidal {
		return nil, fmt.Errorf("TunePID: unknown discrete formula")
	}
	if o.Weights != nil {
		for _, v := range []*float64{o.Weights.FixedB, o.Weights.FixedC} {
			if v != nil && !finitePID(*v) {
				return nil, fmt.Errorf("TunePID: fixed weights must be finite")
			}
		}
	}
	if o.MaxEvaluations == 0 {
		o.MaxEvaluations = 4096
	}
	if o.MaxEvaluations < 1 || o.MaxEvaluations > 4096 {
		return nil, fmt.Errorf("TunePID: evaluation bound must be 1..4096")
	}
	family = PidtuneType(strings.ToUpper(string(family)))
	hasP, hasI, hasD, filtered := true, false, false, false
	switch family {
	case PidtuneP:
	case PidtuneI:
		hasP = false
		hasI = true
	case PidtunePI:
		hasI = true
	case PidtunePD:
		hasD = true
	case PidtunePDF:
		hasD = true
		filtered = true
	case PidtunePID:
		hasI = true
		hasD = true
	case PidtunePIDF:
		hasI = true
		hasD = true
		filtered = true
	default:
		return nil, fmt.Errorf("TunePID: unsupported family %q", family)
	}
	// Only explicitly filtered families introduce a derivative filter.
	tf := 0.
	if filtered {
		tf = .1 / wc
	}
	c := PID2{Tf: tf, Dt: p.dt, IFormula: o.IFormula, DFormula: o.DFormula, B: 1, C: 1}
	h := p.at(wc)
	if !pidFiniteComplex(h) || cmplx.Abs(h) < 1e-15 {
		return nil, fmt.Errorf("TunePID: invalid plant gain at crossover")
	}
	target := cmplx.Rect(1, (-180+o.PhaseMargin)*math.Pi/180) / h
	ib, db := pidTuningBasis(c, wc)
	if hasP && hasI {
		if hasD {
			c.Kd = math.Copysign(math.Abs(imag(target)/imag(db))+.5*cmplx.Abs(target)/wc, real(target))
		}
		c.Ki = (imag(target) - c.Kd*imag(db)) / imag(ib)
		c.Kp = real(target) - c.Ki*real(ib) - c.Kd*real(db)
	} else if hasP && hasD {
		c.Kd = imag(target) / imag(db)
		c.Kp = real(target) - c.Kd*real(db)
	} else if hasI {
		c.Ki = cmplx.Abs(target) / cmplx.Abs(ib)
	} else {
		c.Kp = cmplx.Abs(target)
	}
	// Negative-gain plants require a signed P or I seed.
	if !hasP || (!hasI && !hasD) {
		a := c
		a.Kp = -a.Kp
		a.Ki = -a.Ki
		if pidPhaseDistance(p.at(wc)*pidTuningFeedback(a, wc), o.PhaseMargin) < pidPhaseDistance(p.at(wc)*pidTuningFeedback(c, wc), o.PhaseMargin) {
			c = a
		}
	}
	if !finitePID(c.Kp) || !finitePID(c.Ki) || !finitePID(c.Kd) {
		return nil, fmt.Errorf("TunePID: singular controller basis at requested crossover")
	}
	const points = 241
	omega, plant := make([]float64, points), make([]complex128, points)
	for k := range omega {
		omega[k] = math.Exp(math.Log(p.low) + float64(k)*math.Log(p.high/p.low)/float64(points-1))
		plant[k] = p.at(omega[k])
		if !pidFiniteComplex(plant[k]) {
			return nil, fmt.Errorf("TunePID: nonfinite plant response at %g", omega[k])
		}
	}
	// Include wc exactly so target evidence never depends on a nearby grid sample.
	mid := 0
	for i := range omega {
		if math.Abs(math.Log(omega[i]/wc)) < math.Abs(math.Log(omega[mid]/wc)) {
			mid = i
		}
	}
	omega[mid] = wc
	plant[mid] = h
	components := func(c PID2) PIDTuningObjective {
		v := PIDTuningObjective{}
		for k, w := range omega {
			fb := pidTuningFeedback(c, w)
			cr := pidTuningReference(c, w)
			den := 1 + plant[k]*fb
			m := complex(wc, 0) / (complex(wc, w))
			tr := plant[k] * cr / den
			dr := plant[k] / den
			ur := cr / den
			v.Tracking += pidAbsSquared(tr - m)
			v.Rejection += pidAbsSquared(dr)
			v.Effort += pidAbsSquared(ur) / (1 + w*w/(wc*wc))
		}
		v.Tracking /= points
		v.Rejection /= points
		v.Effort /= points
		return v
	}
	pidTuningSolveWeights(&c, o.Weights, omega, plant, wc)
	norms := components(c)
	if !finitePID(norms.Tracking) || !finitePID(norms.Rejection) || !finitePID(norms.Effort) {
		return nil, fmt.Errorf("TunePID: objective overflows; rescale plant units")
	}
	norms.Tracking = math.Max(norms.Tracking, 1e-3)
	norms.Rejection = math.Max(norms.Rejection, 1e-3)
	norms.Effort = math.Max(norms.Effort, 1e-3)
	score := func(v PIDTuningObjective) float64 {
		return tracking*v.Tracking/norms.Tracking + (1-tracking)*v.Rejection/norms.Rejection + .001*v.Effort/norms.Effort
	}
	evidence := PIDTuningEvidence{RequestedCrossover: wc, RequestedPhaseMargin: o.PhaseMargin, Stability: p.stability, Warnings: append([]string(nil), p.warnings...), Normalizers: norms, Termination: "converged"}

	bestScore := math.Inf(1)
	var best PID2
	var bestParts PIDTuningObjective
	evaluate := func(candidate PID2) error {
		if err := ctx.Err(); err != nil {
			return err
		}
		if evidence.Evaluations >= o.MaxEvaluations {
			return nil
		}
		evidence.Evaluations++
		if filtered && (candidate.Tf < math.Max(.001/wc, p.dt*.51) || candidate.Tf > 10/wc) {
			return nil
		}
		mag := cmplx.Abs(h * pidTuningFeedback(candidate, wc))
		if !finitePID(mag) || mag < 1e-15 {
			return nil
		}
		candidate.Kp /= mag
		candidate.Ki /= mag
		candidate.Kd /= mag
		if pidPhaseDistance(h*pidTuningFeedback(candidate, wc), o.PhaseMargin) > 3 {
			return nil
		}
		pidTuningSolveWeights(&candidate, o.Weights, omega, plant, wc)
		v := components(candidate)
		j := score(v)
		if !finitePID(j) || j >= bestScore {
			return nil
		}
		if p.stable != nil && !p.stable(&candidate) {
			return nil
		}
		_, margins := pidTuningCrossings(p, candidate, omega, wc)
		for _, margin := range margins {
			if margin < o.PhaseMargin-3 {
				return nil
			}
		}
		best, bestScore, bestParts = candidate, j, v
		return nil
	}
	if err := evaluate(c); err != nil {
		return nil, err
	}
	if hasP && hasI && hasD {
		seed := c
		seed.Kd = 0
		seed.Ki = imag(target) / imag(ib)
		seed.Kp = real(target) - seed.Ki*real(ib)
		if err := evaluate(seed); err != nil {
			return nil, err
		}
	}
	evidence.SeedObjective = bestScore
	// Signed, scaled coordinate pattern search; each candidate is gain-normalized
	// at wc and must preserve target phase and model-based stability when available.
	current := c
	step := .5
	for round := 0; round < 64 && evidence.Evaluations < o.MaxEvaluations; round++ {
		previous := bestScore
		if finitePID(bestScore) {
			current = best
		}
		for axis := 0; axis < 4; axis++ {
			if axis == 0 && !hasP || axis == 1 && !hasI || axis == 2 && !hasD || axis == 3 && !filtered {
				continue
			}
			for _, sign := range []float64{-1, 1} {
				candidate := current
				scale := 1 / cmplx.Abs(h)
				switch axis {
				case 0:
					candidate.Kp += sign * step * scale
				case 1:
					candidate.Ki += sign * step * scale * wc
				case 2:
					candidate.Kd += sign * step * scale / wc
				case 3:
					candidate.Tf *= math.Exp(sign * step)
				}
				if err := evaluate(candidate); err != nil {
					return nil, err
				}
			}
		}
		if bestScore >= previous-1e-6*math.Max(1, math.Abs(previous)) || !finitePID(bestScore) {
			step *= .5
		}
		if step < 1e-5 {
			break
		}
		if round == 63 {
			evidence.Termination = "round-limit"
		}
	}
	if evidence.Evaluations >= o.MaxEvaluations {
		evidence.Termination = "evaluation-limit"
	}
	if !finitePID(bestScore) {
		evidence.SeedObjective = 0
		return &PIDTuningResult{Evidence: evidence}, ErrPIDTuningTargetUnattainable
	}
	if !finitePID(evidence.SeedObjective) {
		evidence.SeedObjective = bestScore
		evidence.Warnings = append(evidence.Warnings, "Initial seed was infeasible; objective baseline is the first feasible search result.")
	}
	evidence.Objective = bestScore
	evidence.Components = bestParts
	evidence.AchievedCrossover = wc
	evidence.AchievedPhaseMargin = 180 + cmplx.Phase(h*pidTuningFeedback(best, wc))*180/math.Pi
	if evidence.AchievedPhaseMargin > 180 {
		evidence.AchievedPhaseMargin -= 360
	}
	evidence.GainCrossovers, evidence.PhaseMargins = pidTuningCrossings(p, best, omega, wc)
	for _, margin := range evidence.PhaseMargins {
		evidence.AchievedPhaseMargin = math.Min(evidence.AchievedPhaseMargin, margin)
	}
	evidence.FrequencyBand = [2]float64{p.low, p.high}
	evidence.Warnings = append(evidence.Warnings, "Margins describe the recorded finite analysis band; unresolved crossings outside that band are not excluded.")
	evidence.Feasible = true
	return &PIDTuningResult{Controller: &best, Evidence: evidence}, nil
}

func pidTuningBasis(c PID2, w float64) (integral, derivative complex128) {
	if c.Dt == 0 {
		s := complex(0, w)
		return 1 / s, s / (1 + complex(c.Tf, 0)*s)
	}
	z := cmplx.Exp(complex(0, w*c.Dt))
	basis := func(f PIDFormula) complex128 {
		switch f {
		case BackwardEuler:
			return complex(c.Dt, 0) * z / (z - 1)
		case Trapezoidal:
			return complex(c.Dt/2, 0) * (z + 1) / (z - 1)
		default:
			return complex(c.Dt, 0) / (z - 1)
		}
	}
	integral = basis(c.IFormula)
	d := 1 / basis(c.DFormula)
	return integral, d / (1 + complex(c.Tf, 0)*d)
}
func pidTuningFeedback(c PID2, w float64) complex128 {
	i, d := pidTuningBasis(c, w)
	return complex(c.Kp, 0) + complex(c.Ki, 0)*i + complex(c.Kd, 0)*d
}
func pidTuningReference(c PID2, w float64) complex128 {
	i, d := pidTuningBasis(c, w)
	return complex(c.B*c.Kp, 0) + complex(c.Ki, 0)*i + complex(c.C*c.Kd, 0)*d
}
func pidFiniteComplex(v complex128) bool { return finitePID(real(v)) && finitePID(imag(v)) }
func pidAbsSquared(v complex128) float64 { return real(v)*real(v) + imag(v)*imag(v) }
func pidPhaseDistance(loop complex128, pm float64) float64 {
	return math.Abs(math.Remainder(cmplx.Phase(loop)*180/math.Pi-(-180+pm), 360))
}

// Solve the two-dimensional box least squares exactly by checking the interior
// and faces. Fixed weights are removed from the equations, not rounded to a grid.
func pidTuningSolveWeights(c *PID2, o *PIDTuningWeights, omega []float64, plant []complex128, wc float64) {
	if o == nil {
		c.B = 1
		c.C = 1
		return
	}
	bFixed, cFixed := o.FixedB, o.FixedC
	if c.Kp == 0 && bFixed == nil {
		v := 1.
		bFixed = &v
	}
	if c.Kd == 0 && cFixed == nil {
		v := 1.
		cFixed = &v
	}
	aa, ab, bb, ay, by := 0., 0., 0., 0., 0.
	for k, w := range omega {
		ib, db := pidTuningBasis(*c, w)
		den := 1 + plant[k]*pidTuningFeedback(*c, w)
		a := plant[k] * complex(c.Kp, 0) / den
		b := plant[k] * complex(c.Kd, 0) * db / den
		y := complex(wc, 0)/complex(wc, w) - plant[k]*complex(c.Ki, 0)*ib/den
		aa += pidAbsSquared(a)
		bb += pidAbsSquared(b)
		ab += real(cmplx.Conj(a) * b)
		ay += real(cmplx.Conj(a) * y)
		by += real(cmplx.Conj(b) * y)
	}
	clamp := func(v float64) float64 { return math.Max(0, math.Min(1, v)) }
	if bFixed != nil && cFixed != nil {
		c.B = *bFixed
		c.C = *cFixed
		return
	}
	if bFixed != nil {
		c.B = *bFixed
		c.C = clamp((by - ab*c.B) / math.Max(bb, 1e-30))
		return
	}
	if cFixed != nil {
		c.C = *cFixed
		c.B = clamp((ay - ab*c.C) / math.Max(aa, 1e-30))
		return
	}
	best := math.Inf(1)
	accept := func(b, d float64) {
		if b < 0 || b > 1 || d < 0 || d > 1 {
			return
		}
		j := aa*b*b + 2*ab*b*d + bb*d*d - 2*ay*b - 2*by*d
		if j < best {
			best = j
			c.B = b
			c.C = d
		}
	}
	determinant := aa*bb - ab*ab
	if determinant > 1e-14*aa*bb {
		accept((ay*bb-by*ab)/determinant, (by*aa-ay*ab)/determinant)
	}
	for _, v := range []float64{0, 1} {
		accept(v, clamp((by-ab*v)/math.Max(bb, 1e-30)))
		accept(clamp((ay-ab*v)/math.Max(aa, 1e-30)), v)
	}
}

// Ideal derivative controllers may be improper on their own while their
// closed-loop characteristic is well-defined. Test the characteristic directly.
func pidTuningIdealStable(plant *System, c *PID2) bool {
	transfer, err := plant.TransferFunction(nil)
	if err != nil {
		return false
	}
	var ip, id, dn, dd Poly
	if c.Ki == 0 {
		ip, id = Poly{0}, Poly{1}
	} else if c.Dt == 0 {
		ip, id = Poly{1}, Poly{1, 0}
	} else {
		switch c.IFormula {
		case BackwardEuler:
			ip, id = Poly{c.Dt, 0}, Poly{1, -1}
		case Trapezoidal:
			ip, id = Poly{c.Dt / 2, c.Dt / 2}, Poly{1, -1}
		default:
			ip, id = Poly{c.Dt}, Poly{1, -1}
		}
	}
	if c.Dt == 0 {
		dn, dd = Poly{1, 0}, Poly{1}
	} else {
		switch c.DFormula {
		case BackwardEuler:
			dn, dd = Poly{1, -1}, Poly{c.Dt, 0}
		case Trapezoidal:
			dn, dd = Poly{2, -2}, Poly{c.Dt, c.Dt}
		default:
			dn, dd = Poly{1, -1}, Poly{c.Dt}
		}
	}
	scale := func(p Poly, k float64) Poly {
		r := append(Poly(nil), p...)
		for i := range r {
			r[i] *= k
		}
		return r
	}
	denominator := id.Mul(dd)
	numerator := scale(denominator, c.Kp).Add(scale(ip.Mul(dd), c.Ki)).Add(scale(dn.Mul(id), c.Kd))
	characteristic := Poly(transfer.TF.Den[0]).Mul(denominator).Add(Poly(transfer.TF.Num[0][0]).Mul(numerator))
	for len(characteristic) > 1 && characteristic[0] == 0 {
		characteristic = characteristic[1:]
	}
	if len(characteristic) == 0 || characteristic[0] == 0 {
		return false
	}
	poles, err := characteristic.Roots()
	if err != nil {
		return false
	}
	for _, pole := range poles {
		if !pidFiniteComplex(pole) || c.Dt == 0 && real(pole) >= 0 || c.Dt > 0 && cmplx.Abs(pole) >= 1 {
			return false
		}
	}
	return true
}

// Report all located gain crossings, retaining the exactly normalized target.
// Endpoint/grid evidence is explicitly finite-band even for a known rational plant.
func pidTuningCrossings(p pidTuningPlant, c PID2, omega []float64, wc float64) (crossings, margins []float64) {
	crossings = []float64{wc}
	previous := cmplx.Abs(p.at(omega[0])*pidTuningFeedback(c, omega[0])) - 1
	for k := 1; k < len(omega); k++ {
		next := cmplx.Abs(p.at(omega[k])*pidTuningFeedback(c, omega[k])) - 1
		if previous*next < 0 {
			lo, hi := omega[k-1], omega[k]
			lowSign := previous
			for round := 0; round < 24; round++ {
				mid := math.Sqrt(lo * hi)
				v := cmplx.Abs(p.at(mid)*pidTuningFeedback(c, mid)) - 1
				if lowSign*v <= 0 {
					hi = mid
				} else {
					lo = mid
					lowSign = v
				}
			}
			w := math.Sqrt(lo * hi)
			if math.Abs(math.Log(w/wc)) > 1e-6 {
				crossings = append(crossings, w)
			}
		}
		previous = next
	}
	for _, w := range crossings {
		phase := 180 + cmplx.Phase(p.at(w)*pidTuningFeedback(c, w))*180/math.Pi
		if phase > 180 {
			phase -= 360
		}
		margins = append(margins, phase)
	}
	return crossings, margins
}
