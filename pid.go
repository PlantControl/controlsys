package controlsys

import (
	"fmt"
	"math"

	"gonum.org/v1/gonum/mat"
)

type PIDForm int

const (
	PIDParallel PIDForm = iota
	PIDStandard
)

// PIDFormula selects the discrete approximation of an integral or derivative.
// The zero value preserves the historical forward Euler realization.
type PIDFormula int

const (
	ForwardEuler PIDFormula = iota
	BackwardEuler
	Trapezoidal
)

func WithPIDFormulas(integral, derivative PIDFormula) PIDOption {
	return func(p *PID) { p.IFormula, p.DFormula = integral, derivative }
}

type PID struct {
	IFormula PIDFormula
	DFormula PIDFormula
	Kp       float64
	Ki       float64
	Kd       float64
	Tf       float64
	Dt       float64
	Form     PIDForm
}

// Copy returns a copy of the PID controller.
func (p *PID) Copy() *PID {
	if p == nil {
		return nil
	}
	cp := *p
	return &cp
}

type PIDOption func(*PID)

func WithFilter(Tf float64) PIDOption {
	return func(p *PID) { p.Tf = Tf }
}

func WithTs(dt float64) PIDOption {
	return func(p *PID) { p.Dt = dt }
}

func NewPID(Kp, Ki, Kd float64, opts ...PIDOption) *PID {
	p := &PID{Kp: Kp, Ki: Ki, Kd: Kd, Form: PIDParallel}
	for _, o := range opts {
		o(p)
	}
	return p
}

// NewPIDStd creates a PID in standard (ISA) form:
//
//	C(s) = Kp * (1 + 1/(Ti*s) + Td*s/(Tf*s + 1))
//
// Relation to parallel: Ki = Kp/Ti, Kd = Kp*Td.
func NewPIDStd(Kp, Ti, Td float64, opts ...PIDOption) (*PID, error) {
	if math.IsNaN(Ti) || math.IsInf(Ti, -1) || math.IsNaN(Td) || math.IsInf(Td, 0) || math.IsNaN(Kp) || math.IsInf(Kp, 0) {
		return nil, fmt.Errorf("controlsys: invalid standard PID parameters")
	}
	if Ti == 0 && Kp != 0 {
		return nil, fmt.Errorf("controlsys: Ti must be nonzero in standard form")
	}
	var Ki, Kd float64
	if Ti != 0 {
		Ki = Kp / Ti
	}
	Kd = Kp * Td
	p := &PID{Kp: Kp, Ki: Ki, Kd: Kd, Form: PIDStandard}
	for _, o := range opts {
		o(p)
	}
	return p, nil
}

// Parallel returns a copy in parallel form (Kp, Ki, Kd).
func (p *PID) Parallel() *PID {
	cp := *p
	cp.Form = PIDParallel
	return &cp
}

// Standard returns an equivalent standard parameterization. A nonzero integral
// or derivative with zero proportional gain cannot be represented in this form.
func (p *PID) Standard() (*PID, error) {
	if p.Kp == 0 && (p.Ki != 0 || p.Kd != 0) {
		return nil, fmt.Errorf("controlsys: nonzero I or D with zero Kp has no standard form")
	}
	cp := *p
	cp.Form = PIDStandard
	return &cp, nil
}

// Ti returns the integral time constant; +Inf denotes disabled integral action.
func (p *PID) Ti() float64 {
	if p.Ki == 0 {
		return math.Inf(1)
	}
	return p.Kp / p.Ki
}

// Td returns the derivative time constant (standard form). Returns 0 if Kp=0.
func (p *PID) Td() float64 {
	if p.Kp == 0 {
		return 0
	}
	return p.Kd / p.Kp
}

// PID2 represents a 2-DOF PID controller.
//
//	u = Kp*(b*r - y) + Ki/s*(r - y) + Kd*s/(Tf*s+1)*(c*r - y)
//
// The System() method produces a 2-input (r, y) to 1-output (u) system.
type PID2 struct {
	IFormula PIDFormula
	DFormula PIDFormula
	Kp       float64
	Ki       float64
	Kd       float64
	Tf       float64
	B        float64 // setpoint weight on proportional
	C        float64 // setpoint weight on derivative
	Dt       float64
}

// Copy returns a copy of the 2-DOF PID controller.
func (p *PID2) Copy() *PID2 {
	if p == nil {
		return nil
	}
	cp := *p
	return &cp
}

func NewPID2(Kp, Ki, Kd, Tf, b, c float64, opts ...PIDOption) *PID2 {
	p2 := &PID2{Kp: Kp, Ki: Ki, Kd: Kd, Tf: Tf, B: b, C: c}
	tmp := &PID{Dt: p2.Dt}
	for _, o := range opts {
		o(tmp)
	}
	p2.Dt = tmp.Dt
	p2.IFormula, p2.DFormula = tmp.IFormula, tmp.DFormula
	return p2
}

type pidRealizationSpec struct {
	hasI bool
	hasD bool
}

func newPIDRealizationSpec(Ki, Kd, Tf float64, context string) (pidRealizationSpec, error) {
	if Kd != 0 && Tf == 0 {
		return pidRealizationSpec{}, fmt.Errorf("controlsys: %s without filter (Tf=0) is improper; set Tf > 0", context)
	}
	return pidRealizationSpec{
		hasI: Ki != 0,
		hasD: Kd != 0,
	}, nil
}

// System converts the 2-DOF PID to a 2-input (r,y) 1-output (u) state-space.
func (p *PID2) System() (*System, error) {
	if err := validatePID(p.Kp, p.Ki, p.Kd, p.Tf, p.Dt, p.IFormula, p.DFormula); err != nil {
		return nil, err
	}
	if !finitePID(p.B) || !finitePID(p.C) {
		return nil, fmt.Errorf("controlsys: PID weights must be finite")
	}
	if p.Dt > 0 {
		return pidDiscrete(p.Kp, p.Ki, p.Kd, p.Tf, p.Dt, p.IFormula, p.DFormula, []float64{p.B, -1}, []float64{1, -1}, []float64{p.C, -1})
	}
	spec, err := newPIDRealizationSpec(p.Ki, p.Kd, p.Tf, "2-DOF PID derivative term")
	if err != nil {
		return nil, err
	}
	hasI := spec.hasI
	hasD := spec.hasD

	n := 0
	if hasI {
		n++
	}
	if hasD {
		n++
	}

	if n == 0 {
		D := mat.NewDense(1, 2, []float64{p.Kp * p.B, -p.Kp})
		return NewGain(D, p.Dt)
	}

	A := mat.NewDense(n, n, nil)
	Bmat := mat.NewDense(n, 2, nil)
	Cmat := mat.NewDense(1, n, nil)
	Dmat := mat.NewDense(1, 2, nil)

	idx := 0
	dFeedR := p.Kp * p.B
	dFeedY := -p.Kp

	if hasI {
		A.Set(idx, idx, 0)
		Bmat.Set(idx, 0, 1)
		Bmat.Set(idx, 1, -1)
		Cmat.Set(0, idx, p.Ki)
		idx++
	}

	if hasD {
		invTf := 1.0 / p.Tf
		A.Set(idx, idx, -invTf)
		Bmat.Set(idx, 0, p.C*invTf)
		Bmat.Set(idx, 1, -invTf)
		Cmat.Set(0, idx, -p.Kd*invTf)
		dFeedR += p.Kd * invTf * p.C
		dFeedY += -p.Kd * invTf
	}

	Dmat.Set(0, 0, dFeedR)
	Dmat.Set(0, 1, dFeedY)

	return New(A, Bmat, Cmat, Dmat, 0)
}

func (p *PID) System() (*System, error) {
	if err := validatePID(p.Kp, p.Ki, p.Kd, p.Tf, p.Dt, p.IFormula, p.DFormula); err != nil {
		return nil, err
	}
	if p.Dt > 0 {
		return pidDiscrete(p.Kp, p.Ki, p.Kd, p.Tf, p.Dt, p.IFormula, p.DFormula, []float64{1}, []float64{1}, []float64{1})
	}
	return p.continuousSystem()
}

func (p *PID) continuousSystem() (*System, error) {
	spec, err := newPIDRealizationSpec(p.Ki, p.Kd, p.Tf, "PID derivative term")
	if err != nil {
		return nil, err
	}
	hasI := spec.hasI
	hasD := spec.hasD

	switch {
	case !hasI && !hasD:
		return NewGain(mat.NewDense(1, 1, []float64{p.Kp}), 0)

	case hasI && !hasD:
		return New(
			mat.NewDense(1, 1, []float64{0}),
			mat.NewDense(1, 1, []float64{1}),
			mat.NewDense(1, 1, []float64{p.Ki}),
			mat.NewDense(1, 1, []float64{p.Kp}),
			0,
		)

	case !hasI && hasD:
		invTf := 1.0 / p.Tf
		return New(
			mat.NewDense(1, 1, []float64{-invTf}),
			mat.NewDense(1, 1, []float64{invTf}),
			mat.NewDense(1, 1, []float64{-p.Kd * invTf}),
			mat.NewDense(1, 1, []float64{p.Kp + p.Kd*invTf}),
			0,
		)

	default:
		invTf := 1.0 / p.Tf
		return New(
			mat.NewDense(2, 2, []float64{0, 0, 0, -invTf}),
			mat.NewDense(2, 1, []float64{1, invTf}),
			mat.NewDense(1, 2, []float64{p.Ki, -p.Kd * invTf}),
			mat.NewDense(1, 1, []float64{p.Kp + p.Kd*invTf}),
			0,
		)
	}
}

func finitePID(v float64) bool { return !math.IsNaN(v) && !math.IsInf(v, 0) }
func validatePID(kp, ki, kd, tf, dt float64, i, d PIDFormula) error {
	for _, v := range []float64{kp, ki, kd, tf, dt} {
		if !finitePID(v) {
			return fmt.Errorf("controlsys: PID parameters must be finite")
		}
	}
	if tf < 0 || dt < 0 {
		return fmt.Errorf("controlsys: PID filter and sample time must be nonnegative")
	}
	if i < ForwardEuler || i > Trapezoidal || d < ForwardEuler || d > Trapezoidal {
		return fmt.Errorf("controlsys: invalid PID discrete formula")
	}
	return nil
}

func pidFormulaWeight(f PIDFormula) float64 {
	switch f {
	case BackwardEuler:
		return 1
	case Trapezoidal:
		return .5
	}
	return 0
}

func pidDiscrete(kp, ki, kd, tf, dt float64, integral, derivative PIDFormula, pw, iw, dw []float64) (*System, error) {
	n := 0
	if ki != 0 {
		n++
	}
	if kd != 0 {
		n++
	}
	m := len(pw)
	feed := mat.NewDense(1, m, nil)
	for j, w := range pw {
		feed.Set(0, j, kp*w)
	}
	if n == 0 {
		return NewGain(feed, dt)
	}
	a := mat.NewDense(n, n, nil)
	b := mat.NewDense(n, m, nil)
	c := mat.NewDense(1, n, nil)
	row := 0
	if ki != 0 {
		a.Set(row, row, 1)
		c.Set(0, row, ki)
		for j, w := range iw {
			b.Set(row, j, dt*w)
			feed.Set(0, j, feed.At(0, j)+ki*pidFormulaWeight(integral)*dt*w)
		}
		row++
	}
	if kd != 0 {
		denominator := tf + pidFormulaWeight(derivative)*dt
		if denominator == 0 {
			return nil, fmt.Errorf("controlsys: forward Euler ideal derivative is noncausal; choose a derivative filter or another formula")
		}
		a.Set(row, row, 1-dt/denominator)
		c.Set(0, row, -kd/denominator)
		for j, w := range dw {
			b.Set(row, j, dt/denominator*w)
			feed.Set(0, j, feed.At(0, j)+kd/denominator*w)
		}
	}
	return New(a, b, c, feed, dt)
}
