package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// PIDForm records the parameterization a PID was built in; the gains are
// always stored in parallel form.
type PIDForm int

const (
	PIDParallel PIDForm = iota
	PIDStandard
)

// PIDFormula selects the discrete approximation of an integral or derivative,
// as MATLAB IFormula/DFormula. The zero value is forward Euler.
type PIDFormula int

const (
	ForwardEuler PIDFormula = iota
	BackwardEuler
	Trapezoidal
)

// PIDOption sets an optional MATLAB name-value property of a PID or PID2.
type PIDOption func(*pidFormulas)

type pidFormulas struct{ i, d PIDFormula }

// WithPIDFormulas sets the discrete integral and derivative formulas, as the
// MATLAB 'IFormula' and 'DFormula' properties.
func WithPIDFormulas(integral, derivative PIDFormula) PIDOption {
	return func(f *pidFormulas) { f.i, f.d = integral, derivative }
}

// PID is a 1-DOF controller in parallel form,
//
//	C = Kp + Ki·I(s) + Kd·s/(Tf·s+1),
//
// continuous when Dt = 0 and discrete with the IFormula/DFormula
// approximations otherwise.
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

func newPIDFormulas(op string, opts []PIDOption) (pidFormulas, error) {
	var f pidFormulas
	for _, o := range opts {
		o(&f)
	}
	if !validPIDFormula(f.i) || !validPIDFormula(f.d) {
		return f, fmt.Errorf("%s: unknown discrete formula (%d, %d): %w", op, f.i, f.d, ErrInvalidArgument)
	}
	return f, nil
}

func validPIDFormula(f PIDFormula) bool { return f >= ForwardEuler && f <= Trapezoidal }

func validatePIDGains(op string, kp, ki, kd, tf, ts float64) error {
	if err := requireFinite(op, "gains", kp, ki, kd); err != nil {
		return err
	}
	if !finitePID(tf) || tf < 0 {
		return fmt.Errorf("%s: Tf %g must be finite and nonnegative: %w", op, tf, ErrInvalidArgument)
	}
	if !finitePID(ts) || ts < 0 {
		return fmt.Errorf("%s: Ts %g must be 0 (continuous) or positive: %w", op, ts, ErrInvalidArgument)
	}
	return nil
}

// NewPID creates a parallel-form PID, as MATLAB pid(Kp,Ki,Kd,Tf,Ts)
// (https://www.mathworks.com/help/control/ref/pid.html). Tf = 0 means no
// derivative filter and Ts = 0 a continuous controller. Non-finite gains, a
// negative Tf or Ts, or an unknown formula return ErrInvalidArgument.
func NewPID(Kp, Ki, Kd, Tf, Ts float64, opts ...PIDOption) (*PID, error) {
	if err := validatePIDGains("NewPID", Kp, Ki, Kd, Tf, Ts); err != nil {
		return nil, err
	}
	f, err := newPIDFormulas("NewPID", opts)
	if err != nil {
		return nil, err
	}
	return &PID{Kp: Kp, Ki: Ki, Kd: Kd, Tf: Tf, Dt: Ts, IFormula: f.i, DFormula: f.d, Form: PIDParallel}, nil
}

// NewPIDStd creates a PID in standard (ISA) form, as MATLAB
// pidstd(Kp,Ti,Td,N,Ts) (https://www.mathworks.com/help/control/ref/pidstd.html):
//
//	C(s) = Kp * (1 + 1/(Ti*s) + Td*s/((Td/N)*s + 1))
//
// Ti is positive (+Inf disables integral action), Td finite and nonnegative,
// N positive (+Inf means no derivative filter) and Ts 0 or positive; other
// values return ErrInvalidArgument. The gains are stored in parallel form:
// Ki = Kp/Ti, Kd = Kp*Td, Tf = Td/N.
func NewPIDStd(Kp, Ti, Td, N, Ts float64, opts ...PIDOption) (*PID, error) {
	const op = "NewPIDStd"
	if err := requireFinite(op, "Kp", Kp); err != nil {
		return nil, err
	}
	if math.IsNaN(Ti) || Ti <= 0 {
		return nil, fmt.Errorf("%s: Ti %g must be positive (+Inf disables integral action): %w", op, Ti, ErrInvalidArgument)
	}
	if !finitePID(Td) || Td < 0 {
		return nil, fmt.Errorf("%s: Td %g must be finite and nonnegative: %w", op, Td, ErrInvalidArgument)
	}
	if math.IsNaN(N) || N <= 0 {
		return nil, fmt.Errorf("%s: N %g must be positive (+Inf means no filter): %w", op, N, ErrInvalidArgument)
	}
	p, err := NewPID(Kp, Kp/Ti, Kp*Td, Td/N, Ts, opts...)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	p.Form = PIDStandard
	return p, nil
}

// Parallel returns a copy in parallel form (Kp, Ki, Kd).
func (p *PID) Parallel() *PID {
	if p == nil {
		return nil
	}
	cp := *p
	cp.Form = PIDParallel
	return &cp
}

// Standard returns an equivalent standard parameterization. A nonzero integral
// or derivative with zero proportional gain cannot be represented in this form
// and returns ErrInvalidArgument.
func (p *PID) Standard() (*PID, error) {
	if p.Kp == 0 && (p.Ki != 0 || p.Kd != 0) {
		return nil, fmt.Errorf("PID.Standard: nonzero I or D with zero Kp has no standard form: %w", ErrInvalidArgument)
	}
	cp := *p
	cp.Form = PIDStandard
	return &cp, nil
}

// Ti returns the standard-form integral time Kp/Ki; +Inf when integral
// action is disabled (Ki = 0). ok is false when the controller has no
// standard form (Kp = 0 with Ki or Kd nonzero).
func (p *PID) Ti() (ti float64, ok bool) {
	if p.Ki == 0 {
		return math.Inf(1), p.Kp != 0 || p.Kd == 0
	}
	if p.Kp == 0 {
		return 0, false
	}
	return p.Kp / p.Ki, true
}

// Td returns the standard-form derivative time Kd/Kp; 0 when derivative
// action is disabled. ok is false when the controller has no standard form.
func (p *PID) Td() (td float64, ok bool) {
	if p.Kp == 0 {
		return 0, p.Ki == 0 && p.Kd == 0
	}
	return p.Kd / p.Kp, true
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

// NewPID2 creates a 2-DOF PID, as MATLAB pid2(Kp,Ki,Kd,Tf,b,c,Ts)
// (https://www.mathworks.com/help/control/ref/pid2.html). Arguments are
// validated as in NewPID, and b and c must be finite.
func NewPID2(Kp, Ki, Kd, Tf, b, c, Ts float64, opts ...PIDOption) (*PID2, error) {
	if err := validatePIDGains("NewPID2", Kp, Ki, Kd, Tf, Ts); err != nil {
		return nil, err
	}
	if err := requireFinite("NewPID2", "setpoint weights", b, c); err != nil {
		return nil, err
	}
	f, err := newPIDFormulas("NewPID2", opts)
	if err != nil {
		return nil, err
	}
	return &PID2{Kp: Kp, Ki: Ki, Kd: Kd, Tf: Tf, B: b, C: c, Dt: Ts, IFormula: f.i, DFormula: f.d}, nil
}

type pidRealizationSpec struct {
	hasI bool
	hasD bool
}

func newPIDRealizationSpec(Ki, Kd, Tf float64, context string) (pidRealizationSpec, error) {
	if Kd != 0 && Tf == 0 {
		return pidRealizationSpec{}, fmt.Errorf("%s without filter (Tf=0) is improper; set Tf > 0: %w", context, ErrImproperModel)
	}
	return pidRealizationSpec{
		hasI: Ki != 0,
		hasD: Kd != 0,
	}, nil
}

// System converts the 2-DOF PID to a 2-input (r,y) 1-output (u) state-space
// model. A continuous ideal derivative (Kd ≠ 0, Tf = 0) or a noncausal
// discrete one returns ErrImproperModel.
func (p *PID2) System() (*System, error) {
	sys, err := p.system()
	if err != nil {
		return nil, fmt.Errorf("PID2.System: %w", err)
	}
	return sys, nil
}

func (p *PID2) system() (*System, error) {
	if err := validatePID(p.Kp, p.Ki, p.Kd, p.Tf, p.Dt, p.IFormula, p.DFormula); err != nil {
		return nil, err
	}
	if !finitePID(p.B) || !finitePID(p.C) {
		return nil, fmt.Errorf("setpoint weights must be finite: %w", ErrInvalidArgument)
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

// System returns the controller as a SISO state-space model. A continuous
// ideal derivative (Kd ≠ 0, Tf = 0) or a noncausal discrete one returns
// ErrImproperModel; invalid parameters return ErrInvalidArgument.
func (p *PID) System() (*System, error) {
	sys, err := p.system()
	if err != nil {
		return nil, fmt.Errorf("PID.System: %w", err)
	}
	return sys, nil
}

func (p *PID) system() (*System, error) {
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
			return fmt.Errorf("PID parameters must be finite: %w", ErrInvalidArgument)
		}
	}
	if tf < 0 || dt < 0 {
		return fmt.Errorf("PID filter and sample time must be nonnegative: %w", ErrInvalidArgument)
	}
	if !validPIDFormula(i) || !validPIDFormula(d) {
		return fmt.Errorf("invalid PID discrete formula: %w", ErrInvalidArgument)
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
			return nil, fmt.Errorf("forward Euler ideal derivative is noncausal; choose a derivative filter or another formula: %w", ErrImproperModel)
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
