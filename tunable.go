package controlsys

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"

	"plantcontrol.org/v1/gonum/mat"
)

// TunableBounds is the admissible range [Lower, Upper] of a TunableReal.
// An unbounded side is ±Inf, as MATLAB realp's Minimum = -Inf and
// Maximum = Inf defaults.
type TunableBounds struct {
	Lower float64
	Upper float64
}

func (b TunableBounds) finite() bool { return isFinite(b.Lower) && isFinite(b.Upper) }

// TunableReal is a named scalar tunable parameter, as MATLAB realp. A
// parameter is free unless SetFixed(true) holds it at its current value.
type TunableReal struct {
	name   string
	value  float64
	bounds TunableBounds
	fixed  bool
}

// NewTunableReal returns the free parameter name with initial value value and
// bounds (-Inf, Inf), as MATLAB realp(name, value). Set bounds with SetBounds.
// An empty name or a non-finite value returns ErrInvalidArgument. See
// https://www.mathworks.com/help/control/ref/realp.html.
func NewTunableReal(name string, value float64) (*TunableReal, error) {
	if name == "" {
		return nil, fmt.Errorf("NewTunableReal: name is empty: %w", ErrInvalidArgument)
	}
	if err := requireFinite("NewTunableReal", "value", value); err != nil {
		return nil, err
	}
	return &TunableReal{name: name, value: value, bounds: TunableBounds{Lower: math.Inf(-1), Upper: math.Inf(1)}}, nil
}

// Name returns the parameter name.
func (p *TunableReal) Name() string { return p.name }

// Value returns the current value.
func (p *TunableReal) Value() float64 { return p.value }

// Bounds returns the admissible range; an unbounded side is ±Inf.
func (p *TunableReal) Bounds() TunableBounds { return p.bounds }

// Fixed reports whether the parameter is held at its current value.
func (p *TunableReal) Fixed() bool { return p.fixed }

// SetFixed holds the parameter at its current value (true) or frees it, as
// MATLAB's realp Free property negated.
func (p *TunableReal) SetFixed(fixed bool) { p.fixed = fixed }

// SetBounds sets the admissible range, as MATLAB's realp Minimum and Maximum
// properties; use ±Inf for an unbounded side. NaN bounds, lower > upper or a
// current value outside the range return ErrInvalidArgument.
func (p *TunableReal) SetBounds(lower, upper float64) error {
	if math.IsNaN(lower) || math.IsNaN(upper) || lower > upper {
		return fmt.Errorf("TunableReal.SetBounds: bounds [%g, %g] for %q: %w", lower, upper, p.name, ErrInvalidArgument)
	}
	if p.value < lower || p.value > upper {
		return fmt.Errorf("TunableReal.SetBounds: value %g of %q outside [%g, %g]: %w", p.value, p.name, lower, upper, ErrInvalidArgument)
	}
	p.bounds = TunableBounds{Lower: lower, Upper: upper}
	return nil
}

// SetValue sets the current value, which must be finite and within Bounds.
func (p *TunableReal) SetValue(value float64) error {
	if err := p.validateValue(value); err != nil {
		return fmt.Errorf("TunableReal.SetValue: %w", err)
	}
	p.value = value
	return nil
}

// Sample returns a copy whose value is values[Name()] when the parameter is
// free and present in values, and the current value otherwise.
func (p *TunableReal) Sample(values map[string]float64) (*TunableReal, error) {
	if p == nil {
		return nil, fmt.Errorf("TunableReal.Sample: nil parameter: %w", ErrInvalidArgument)
	}
	cp := p.copy()
	if cp.fixed {
		return cp, nil
	}
	value, ok := values[cp.name]
	if !ok {
		return cp, nil
	}
	if err := cp.validateValue(value); err != nil {
		return nil, fmt.Errorf("TunableReal.Sample: %w", err)
	}
	cp.value = value
	return cp, nil
}

// RandomSample returns a copy whose value, if free, is drawn uniformly from
// its bounds with rng. A nil rng, or a free parameter with an infinite bound,
// returns ErrInvalidArgument.
func (p *TunableReal) RandomSample(rng *rand.Rand) (*TunableReal, error) {
	if p == nil {
		return nil, fmt.Errorf("TunableReal.RandomSample: nil parameter: %w", ErrInvalidArgument)
	}
	if rng == nil {
		return nil, fmt.Errorf("TunableReal.RandomSample: rng is nil: %w", ErrInvalidArgument)
	}
	cp := p.copy()
	if cp.fixed {
		return cp, nil
	}
	if !cp.bounds.finite() {
		return nil, fmt.Errorf("TunableReal.RandomSample: %q has bounds [%g, %g]; finite bounds required: %w", cp.name, cp.bounds.Lower, cp.bounds.Upper, ErrInvalidArgument)
	}
	cp.value = cp.bounds.Lower + rng.Float64()*(cp.bounds.Upper-cp.bounds.Lower)
	return cp, nil
}

func (p *TunableReal) validateValue(value float64) error {
	if !isFinite(value) {
		return fmt.Errorf("value %g of %q is not finite: %w", value, p.name, ErrInvalidArgument)
	}
	if value < p.bounds.Lower || value > p.bounds.Upper {
		return fmt.Errorf("value %g of %q outside [%g,%g]: %w", value, p.name, p.bounds.Lower, p.bounds.Upper, ErrInvalidArgument)
	}
	return nil
}

func (p *TunableReal) copy() *TunableReal {
	cp := *p
	return &cp
}

// TunableGain is a static gain whose entries are tunable parameters, as
// MATLAB tunableGain; see
// https://www.mathworks.com/help/control/ref/tunablegain.html.
type TunableGain struct {
	name string
	// Gain holds the entries, as MATLAB blk.Gain; entry (i, j) is named
	// "<name>.Gain(i+1,j+1)". Set values, bounds and Free through the
	// entries; replacing an entry with a shared TunableReal ties blocks.
	Gain [][]*TunableReal
	// Dt is the sample time, MATLAB blk.Ts; set it to the plant's.
	Dt                    float64
	InputName, OutputName []string
}

// NewTunableGain returns the ny×nu gain block name, as MATLAB
// tunableGain(name, ny, nu): every entry free, zero and unbounded, Dt = 0.
// An empty name or negative size returns ErrInvalidArgument and a zero size
// ErrDimensionMismatch, as a static gain without inputs or outputs cannot be
// stored.
func NewTunableGain(name string, ny, nu int) (*TunableGain, error) {
	const op = "NewTunableGain"
	if err := requireTunableName(op, name); err != nil {
		return nil, err
	}
	if err := requireTunableIO(op, ny, nu); err != nil {
		return nil, err
	}
	return newTunableGain(op, name, mat.NewDense(ny, nu, nil))
}

// NewTunableGainFrom returns the gain block name initialized to G, as MATLAB
// tunableGain(name, G). A nil or non-finite G or an empty name returns
// ErrInvalidArgument.
func NewTunableGainFrom(name string, G *mat.Dense) (*TunableGain, error) {
	const op = "NewTunableGainFrom"
	if err := requireTunableName(op, name); err != nil {
		return nil, err
	}
	if err := requireFiniteDense(op, "G", G); err != nil {
		return nil, err
	}
	return newTunableGain(op, name, G)
}

func newTunableGain(op, name string, G *mat.Dense) (*TunableGain, error) {
	b := &TunableGain{name: name, Gain: newTunableMatrix(name+".Gain", G, nil)}
	if _, err := b.CurrentSystem(); err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return b, nil
}

// Name returns the block name.
func (b *TunableGain) Name() string { return b.name }

// CurrentSystem returns the gain at the current parameter values.
func (b *TunableGain) CurrentSystem() (*System, error) {
	const op = "TunableGain.CurrentSystem"
	D, err := tunableMatrixValues(b.Gain, "Gain")
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if D.IsEmpty() {
		return nil, fmt.Errorf("%s: Gain is empty: %w", op, ErrDimensionMismatch)
	}
	sys, err := NewGain(D, b.Dt)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return applyBlockNames(op, sys, b.InputName, b.OutputName)
}

// Sample returns a copy with free parameters set from values.
func (b *TunableGain) Sample(values map[string]float64) (*TunableGain, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.Sample(values) })
}

// RandomSample returns a copy with free parameters drawn uniformly from their
// bounds; see TunableReal.RandomSample.
func (b *TunableGain) RandomSample(rng *rand.Rand) (*TunableGain, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.RandomSample(rng) })
}

func (b *TunableGain) mapParams(f func(*TunableReal) (*TunableReal, error)) (*TunableGain, error) {
	G, err := mapTunableMatrix(b.Gain, f)
	if err != nil {
		return nil, err
	}
	return &TunableGain{name: b.name, Gain: G, Dt: b.Dt, InputName: copyStringSlice(b.InputName), OutputName: copyStringSlice(b.OutputName)}, nil
}

// FreeParameters returns the distinct free parameters by name. They are the
// block's own parameters, so SetValue or SetFixed on them changes the block.
func (b *TunableGain) FreeParameters() []*TunableReal {
	return uniqueFreeTunableMatrices(b.Gain)
}

// SampleBlock is Sample as a TunableBlock.
func (b *TunableGain) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

// TunablePID is a parallel-form PID controller
//
//	C = Kp + Ki·I(s) + Kd·s/(Tf·s+1)
//
// with tunable Kp, Ki, Kd and Tf, as MATLAB tunablePID; see
// https://www.mathworks.com/help/control/ref/tunablepid.html. The
// parameters are named "<name>.Kp", "<name>.Ki", "<name>.Kd" and
// "<name>.Tf".
type TunablePID struct {
	name                  string
	Kp, Ki, Kd, Tf        *TunableReal
	IFormula, DFormula    PIDFormula
	Dt                    float64
	InputName, OutputName []string
}

// NewTunablePID returns the PID block name of family typ with sample time ts
// (0 for continuous), as MATLAB tunablePID(name, type, Ts). typ is P, PI, PD
// or PID (case ignored), the families MATLAB accepts: P fixes Ki = Kd = 0 and
// Tf = 1, PI fixes Kd = 0 and Tf = 1, PD fixes Ki = 0, and every other
// parameter is free and unbounded except Tf ≥ 0. MATLAB leaves the initial
// values undocumented; here the free gains start at 0 and Tf at 1. An empty
// name or another family returns ErrInvalidArgument and an invalid ts
// ErrInvalidSampleTime.
func NewTunablePID(name string, typ PidtuneType, ts float64) (*TunablePID, error) {
	const op = "NewTunablePID"
	hasI, hasD, err := tunablePIDFamily(op, name, typ, ts)
	if err != nil {
		return nil, err
	}
	b := &TunablePID{name: name, Dt: ts}
	b.Kp, b.Ki, b.Kd, b.Tf = newTunablePIDGains(name, 0, 0, 0, 1, hasI, hasD)
	return b, b.check(op)
}

// NewTunablePIDFrom returns the PID block name initialized from sys, as
// MATLAB tunablePID(name, sys): gains, Tf, sample time and discrete formulas
// are copied, a zero Ki or Kd is fixed at zero (Kd = 0 also fixes Tf at 1),
// and the rest are free and unbounded except Tf ≥ 0. A nil or invalid sys or
// an empty name returns ErrInvalidArgument.
func NewTunablePIDFrom(name string, sys *PID) (*TunablePID, error) {
	const op = "NewTunablePIDFrom"
	if err := requireTunableName(op, name); err != nil {
		return nil, err
	}
	if sys == nil {
		return nil, fmt.Errorf("%s: sys is nil: %w", op, ErrInvalidArgument)
	}
	if err := validatePID(sys.Kp, sys.Ki, sys.Kd, sys.Tf, sys.Dt, sys.IFormula, sys.DFormula); err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	hasD := sys.Kd != 0
	tf := 1.0
	if hasD {
		tf = sys.Tf
	}
	b := &TunablePID{name: name, Dt: sys.Dt, IFormula: sys.IFormula, DFormula: sys.DFormula}
	b.Kp, b.Ki, b.Kd, b.Tf = newTunablePIDGains(name, sys.Kp, sys.Ki, sys.Kd, tf, sys.Ki != 0, hasD)
	return b, b.check(op)
}

func (b *TunablePID) check(op string) error {
	if _, err := b.CurrentSystem(); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}

// Name returns the block name.
func (b *TunablePID) Name() string { return b.name }

// CurrentSystem returns the PID controller at the current parameter values.
func (b *TunablePID) CurrentSystem() (*System, error) {
	const op = "TunablePID.CurrentSystem"
	v, err := tunableScalarValues(b.Kp, b.Ki, b.Kd, b.Tf)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	pid := &PID{Kp: v[0], Ki: v[1], Kd: v[2], Tf: v[3], Dt: b.Dt, IFormula: b.IFormula, DFormula: b.DFormula}
	sys, err := pid.System()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return applyBlockNames(op, sys, b.InputName, b.OutputName)
}

// Sample returns a copy with free parameters set from values.
func (b *TunablePID) Sample(values map[string]float64) (*TunablePID, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.Sample(values) })
}

// RandomSample returns a copy with free parameters drawn uniformly from their
// bounds; see TunableReal.RandomSample.
func (b *TunablePID) RandomSample(rng *rand.Rand) (*TunablePID, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.RandomSample(rng) })
}

func (b *TunablePID) mapParams(f func(*TunableReal) (*TunableReal, error)) (*TunablePID, error) {
	v, err := mapTunableVector([]*TunableReal{b.Kp, b.Ki, b.Kd, b.Tf}, f)
	if err != nil {
		return nil, err
	}
	cp := *b
	cp.Kp, cp.Ki, cp.Kd, cp.Tf = v[0], v[1], v[2], v[3]
	cp.InputName, cp.OutputName = copyStringSlice(b.InputName), copyStringSlice(b.OutputName)
	return &cp, nil
}

// FreeParameters returns the distinct free parameters by name; they are the
// block's own parameters.
func (b *TunablePID) FreeParameters() []*TunableReal {
	return uniqueFreeTunableMatrices([][]*TunableReal{{b.Kp, b.Ki, b.Kd, b.Tf}})
}

// SampleBlock is Sample as a TunableBlock.
func (b *TunablePID) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

// TunablePID2 is a 2-DOF PID controller with inputs (r, y) and output
//
//	u = Kp·(b·r − y) + Ki·I(s)·(r − y) + Kd·s/(Tf·s+1)·(c·r − y)
//
// whose Kp, Ki, Kd, Tf, b and c are tunable, as MATLAB tunablePID2; see
// https://www.mathworks.com/help/control/ref/tunablepid2.html. The
// parameters are named "<name>.Kp", ..., "<name>.b" and "<name>.c".
type TunablePID2 struct {
	name                  string
	Kp, Ki, Kd, Tf, B, C  *TunableReal
	IFormula, DFormula    PIDFormula
	Dt                    float64
	InputName, OutputName []string
}

// NewTunablePID2 returns the 2-DOF PID block name of family typ with sample
// time ts, as MATLAB tunablePID2(name, type, Ts). Kp, Ki, Kd and Tf follow
// NewTunablePID; b is free, and c is free when the family has a derivative
// term and fixed otherwise. MATLAB leaves the initial values undocumented;
// here the free gains start at 0, Tf, b and c at 1. Errors follow
// NewTunablePID.
func NewTunablePID2(name string, typ PidtuneType, ts float64) (*TunablePID2, error) {
	const op = "NewTunablePID2"
	hasI, hasD, err := tunablePIDFamily(op, name, typ, ts)
	if err != nil {
		return nil, err
	}
	b := &TunablePID2{name: name, Dt: ts}
	b.Kp, b.Ki, b.Kd, b.Tf = newTunablePIDGains(name, 0, 0, 0, 1, hasI, hasD)
	b.B, b.C = newTunablePID2Weights(name, 1, 1, hasD)
	return b, b.check(op)
}

// NewTunablePID2From returns the 2-DOF PID block name initialized from sys,
// as MATLAB tunablePID2(name, sys); the free parameters are chosen as in
// NewTunablePIDFrom, with b free and c free when Kd ≠ 0. A nil or invalid sys
// or an empty name returns ErrInvalidArgument.
func NewTunablePID2From(name string, sys *PID2) (*TunablePID2, error) {
	const op = "NewTunablePID2From"
	if err := requireTunableName(op, name); err != nil {
		return nil, err
	}
	if sys == nil {
		return nil, fmt.Errorf("%s: sys is nil: %w", op, ErrInvalidArgument)
	}
	if err := validatePID(sys.Kp, sys.Ki, sys.Kd, sys.Tf, sys.Dt, sys.IFormula, sys.DFormula); err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if err := requireFinite(op, "setpoint weights", sys.B, sys.C); err != nil {
		return nil, err
	}
	hasD := sys.Kd != 0
	tf := 1.0
	if hasD {
		tf = sys.Tf
	}
	b := &TunablePID2{name: name, Dt: sys.Dt, IFormula: sys.IFormula, DFormula: sys.DFormula}
	b.Kp, b.Ki, b.Kd, b.Tf = newTunablePIDGains(name, sys.Kp, sys.Ki, sys.Kd, tf, sys.Ki != 0, hasD)
	b.B, b.C = newTunablePID2Weights(name, sys.B, sys.C, hasD)
	return b, b.check(op)
}

func (b *TunablePID2) check(op string) error {
	if _, err := b.CurrentSystem(); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}

// Name returns the block name.
func (b *TunablePID2) Name() string { return b.name }

// CurrentSystem returns the 2-input (r, y), 1-output controller at the
// current parameter values.
func (b *TunablePID2) CurrentSystem() (*System, error) {
	const op = "TunablePID2.CurrentSystem"
	v, err := tunableScalarValues(b.Kp, b.Ki, b.Kd, b.Tf, b.B, b.C)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	pid := &PID2{Kp: v[0], Ki: v[1], Kd: v[2], Tf: v[3], B: v[4], C: v[5], Dt: b.Dt, IFormula: b.IFormula, DFormula: b.DFormula}
	sys, err := pid.system()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return applyBlockNames(op, sys, b.InputName, b.OutputName)
}

// Sample returns a copy with free parameters set from values.
func (b *TunablePID2) Sample(values map[string]float64) (*TunablePID2, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.Sample(values) })
}

// RandomSample returns a copy with free parameters drawn uniformly from their
// bounds; see TunableReal.RandomSample.
func (b *TunablePID2) RandomSample(rng *rand.Rand) (*TunablePID2, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.RandomSample(rng) })
}

func (b *TunablePID2) mapParams(f func(*TunableReal) (*TunableReal, error)) (*TunablePID2, error) {
	v, err := mapTunableVector([]*TunableReal{b.Kp, b.Ki, b.Kd, b.Tf, b.B, b.C}, f)
	if err != nil {
		return nil, err
	}
	cp := *b
	cp.Kp, cp.Ki, cp.Kd, cp.Tf, cp.B, cp.C = v[0], v[1], v[2], v[3], v[4], v[5]
	cp.InputName, cp.OutputName = copyStringSlice(b.InputName), copyStringSlice(b.OutputName)
	return &cp, nil
}

// FreeParameters returns the distinct free parameters by name; they are the
// block's own parameters.
func (b *TunablePID2) FreeParameters() []*TunableReal {
	return uniqueFreeTunableMatrices([][]*TunableReal{{b.Kp, b.Ki, b.Kd, b.Tf, b.B, b.C}})
}

// SampleBlock is Sample as a TunableBlock.
func (b *TunablePID2) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

func tunablePIDFamily(op, name string, typ PidtuneType, ts float64) (hasI, hasD bool, err error) {
	if err := requireTunableName(op, name); err != nil {
		return false, false, err
	}
	switch PidtuneType(strings.ToUpper(string(typ))) {
	case PidtuneP:
	case PidtunePI:
		hasI = true
	case PidtunePD:
		hasD = true
	case PidtunePID:
		hasI, hasD = true, true
	default:
		return false, false, fmt.Errorf("%s: type %q, want P, PI, PD or PID: %w", op, typ, ErrInvalidArgument)
	}
	if err := requireTunableSampleTime(op, ts); err != nil {
		return false, false, err
	}
	return hasI, hasD, nil
}

func newTunablePIDGains(name string, kp, ki, kd, tf float64, hasI, hasD bool) (Kp, Ki, Kd, Tf *TunableReal) {
	if !hasI {
		ki = 0
	}
	if !hasD {
		kd = 0
	}
	Tf = newTunableParam(name+".Tf", tf, hasD)
	Tf.bounds.Lower = 0
	return newTunableParam(name+".Kp", kp, true), newTunableParam(name+".Ki", ki, hasI), newTunableParam(name+".Kd", kd, hasD), Tf
}

func newTunablePID2Weights(name string, b, c float64, hasD bool) (*TunableReal, *TunableReal) {
	return newTunableParam(name+".b", b, true), newTunableParam(name+".c", c, hasD)
}

// TunableTF is a SISO transfer function
//
//	(a_m·s^m + ... + a_0) / (s^n + b_(n-1)·s^(n-1) + ... + b_0)
//
// with tunable coefficients, as MATLAB tunableTF; see
// https://www.mathworks.com/help/control/ref/tunabletf.html.
type TunableTF struct {
	name string
	// Numerator holds a_m, ..., a_0 named "<name>.Numerator(k)", k from 1.
	Numerator []*TunableReal
	// Denominator holds 1, b_(n-1), ..., b_0 named "<name>.Denominator(k)";
	// the leading coefficient is fixed at 1.
	Denominator           []*TunableReal
	Dt                    float64
	InputName, OutputName []string
}

// NewTunableTF returns the SISO block name with nz zeros, np poles and
// sample time ts (0 for continuous), as MATLAB tunableTF(name, Nz, Np, Ts).
// Every coefficient but the leading denominator one is free and unbounded.
// MATLAB initializes to an undocumented stable, strictly proper model; here
// it is 1/(s+1)^np, or 1/(z−0.5)^np in discrete time. A negative count or an
// empty name returns ErrInvalidArgument, nz > np ErrImproperTF and an invalid
// ts ErrInvalidSampleTime.
func NewTunableTF(name string, nz, np int, ts float64) (*TunableTF, error) {
	const op = "NewTunableTF"
	if err := requireTunableName(op, name); err != nil {
		return nil, err
	}
	if nz < 0 || np < 0 {
		return nil, fmt.Errorf("%s: Nz %d and Np %d must be nonnegative: %w", op, nz, np, ErrInvalidArgument)
	}
	if nz > np {
		return nil, fmt.Errorf("%s: Nz %d exceeds Np %d: %w", op, nz, np, ErrImproperTF)
	}
	if err := requireTunableSampleTime(op, ts); err != nil {
		return nil, err
	}
	root := -1.0
	if ts > 0 {
		root = 0.5
	}
	den := []float64{1}
	for range np {
		den = append(den, 0)
		for k := len(den) - 1; k > 0; k-- {
			den[k] -= root * den[k-1]
		}
	}
	num := make([]float64, nz+1)
	num[nz] = 1
	return newTunableTF(op, name, num, den, ts)
}

// NewTunableTFFrom returns the block name initialized from the SISO model
// sys, as MATLAB tunableTF(name, sys): Nz and Np are the numerator and
// denominator degrees, the coefficients are normalized to a monic
// denominator, and Dt and signal names are copied. A nil sys or empty name
// returns ErrInvalidArgument, a MIMO sys ErrNotSISO (tunableTF is SISO only),
// a delay ErrDelayUnsupported, a zero leading denominator coefficient
// ErrSingularDenom and an improper sys ErrImproperTF.
func NewTunableTFFrom(name string, sys *TransferFunc) (*TunableTF, error) {
	const op = "NewTunableTFFrom"
	if err := requireTunableName(op, name); err != nil {
		return nil, err
	}
	if sys == nil {
		return nil, fmt.Errorf("%s: sys is nil: %w", op, ErrInvalidArgument)
	}
	p, m, err := sys.validateShape()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if p != 1 || m != 1 {
		return nil, fmt.Errorf("%s: sys is %d×%d: %w", op, p, m, ErrNotSISO)
	}
	if sys.Delay != nil && sys.Delay[0][0] != 0 {
		return nil, fmt.Errorf("%s: sys has a delay: %w", op, ErrDelayUnsupported)
	}
	num, den := trimLeadingZeros(sys.Num[0][0]), sys.Den[0]
	if err := requireFinite(op, "coefficients", append(copyFloatSlice(num), den...)...); err != nil {
		return nil, err
	}
	if len(den) == 0 || den[0] == 0 {
		return nil, fmt.Errorf("%s: leading denominator coefficient is zero: %w", op, ErrSingularDenom)
	}
	if len(num) > len(den) {
		return nil, fmt.Errorf("%s: numerator degree %d exceeds denominator degree %d: %w", op, len(num)-1, len(den)-1, ErrImproperTF)
	}
	lead := den[0]
	num, den = copyFloatSlice(num), copyFloatSlice(den)
	for i := range num {
		num[i] /= lead
	}
	for i := range den {
		den[i] /= lead
	}
	b, err := newTunableTF(op, name, num, den, sys.Dt)
	if err != nil {
		return nil, err
	}
	b.InputName, b.OutputName = copyStringSlice(sys.InputName), copyStringSlice(sys.OutputName)
	return b, b.check(op)
}

func newTunableTF(op, name string, num, den []float64, ts float64) (*TunableTF, error) {
	b := &TunableTF{
		name:        name,
		Numerator:   newTunableVector(name+".Numerator", num),
		Denominator: newTunableVector(name+".Denominator", den),
		Dt:          ts,
	}
	b.Denominator[0].fixed = true
	return b, b.check(op)
}

func trimLeadingZeros(c []float64) []float64 {
	for len(c) > 1 && c[0] == 0 {
		c = c[1:]
	}
	if len(c) == 0 {
		return []float64{0}
	}
	return c
}

func (b *TunableTF) check(op string) error {
	if _, err := b.CurrentSystem(); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}

// Name returns the block name.
func (b *TunableTF) Name() string { return b.name }

// CurrentSystem realizes the transfer function at the current coefficients.
func (b *TunableTF) CurrentSystem() (*System, error) {
	const op = "TunableTF.CurrentSystem"
	num, err := tunableScalarValues(b.Numerator...)
	if err != nil {
		return nil, fmt.Errorf("%s: Numerator: %w", op, err)
	}
	den, err := tunableScalarValues(b.Denominator...)
	if err != nil {
		return nil, fmt.Errorf("%s: Denominator: %w", op, err)
	}
	tf := &TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: b.Dt}
	result, err := tf.stateSpace()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return applyBlockNames(op, result.Sys, b.InputName, b.OutputName)
}

// Sample returns a copy with free coefficients set from values.
func (b *TunableTF) Sample(values map[string]float64) (*TunableTF, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.Sample(values) })
}

// RandomSample returns a copy with free coefficients drawn uniformly from
// their bounds; see TunableReal.RandomSample.
func (b *TunableTF) RandomSample(rng *rand.Rand) (*TunableTF, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.RandomSample(rng) })
}

func (b *TunableTF) mapParams(f func(*TunableReal) (*TunableReal, error)) (*TunableTF, error) {
	num, err := mapTunableVector(b.Numerator, f)
	if err != nil {
		return nil, err
	}
	den, err := mapTunableVector(b.Denominator, f)
	if err != nil {
		return nil, err
	}
	return &TunableTF{name: b.name, Numerator: num, Denominator: den, Dt: b.Dt, InputName: copyStringSlice(b.InputName), OutputName: copyStringSlice(b.OutputName)}, nil
}

// FreeParameters returns the distinct free coefficients by name; they are
// the block's own parameters.
func (b *TunableTF) FreeParameters() []*TunableReal {
	return uniqueFreeTunableMatrices([][]*TunableReal{b.Numerator, b.Denominator})
}

// SampleBlock is Sample as a TunableBlock.
func (b *TunableTF) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

// TunableSSStructure constrains the A matrix of a TunableSS, as the MATLAB
// tunableSS Astruct argument. The zero value selects TunableSSTridiag, the
// MATLAB default.
type TunableSSStructure string

const (
	// TunableSSTridiag frees the main, first super- and first subdiagonal of
	// A and fixes the other entries at zero.
	TunableSSTridiag TunableSSStructure = "tridiag"
	// TunableSSFull frees every entry of A, as the MATLAB Astruct table
	// states (its Free-defaults paragraph says the opposite).
	TunableSSFull TunableSSStructure = "full"
	// TunableSSCompanion is the Canon companion form: the subdiagonal and
	// the last column, which holds the characteristic polynomial, are free
	// and the other entries fixed at zero. MATLAB's Free-defaults paragraph
	// lists the first row instead, which contradicts its own description of
	// the form.
	TunableSSCompanion TunableSSStructure = "companion"
)

// TunableSS is a state-space model with tunable A, B, C and D entries, as
// MATLAB tunableSS; see https://www.mathworks.com/help/control/ref/tunabless.html.
// Entry (i, j) of A is named "<name>.A(i+1,j+1)", and likewise for B, C, D.
type TunableSS struct {
	name                  string
	A, B, C, D            [][]*TunableReal
	Dt                    float64
	InputName, OutputName []string
}

// NewTunableSS returns the block name with nx states, ny outputs, nu inputs
// and sample time ts, A constrained by astruct, as MATLAB
// tunableSS(name, Nx, Ny, Nu, Ts, Astruct). B, C and D are free. MATLAB
// leaves the initial values undocumented; here B, C and D are zero and A has
// characteristic polynomial (s+1)^nx (A = −I unless companion), or z^nx in
// discrete time. An empty name, a negative size or an unknown astruct
// returns ErrInvalidArgument, ny or nu zero ErrDimensionMismatch and an
// invalid ts ErrInvalidSampleTime.
func NewTunableSS(name string, nx, ny, nu int, ts float64, astruct TunableSSStructure) (*TunableSS, error) {
	const op = "NewTunableSS"
	if err := requireTunableName(op, name); err != nil {
		return nil, err
	}
	if nx < 0 {
		return nil, fmt.Errorf("%s: Nx %d is negative: %w", op, nx, ErrInvalidArgument)
	}
	if err := requireTunableIO(op, ny, nu); err != nil {
		return nil, err
	}
	free, err := tunableSSFreeA(op, astruct, nx)
	if err != nil {
		return nil, err
	}
	if err := requireTunableSampleTime(op, ts); err != nil {
		return nil, err
	}
	var A *mat.Dense
	if nx > 0 {
		A = mat.NewDense(nx, nx, nil)
		pole := -1.0
		if ts > 0 {
			pole = 0
		}
		if astruct == TunableSSCompanion {
			coeffs := []float64{1}
			for range nx {
				coeffs = append(coeffs, 0)
				for k := len(coeffs) - 1; k > 0; k-- {
					coeffs[k] -= pole * coeffs[k-1]
				}
			}
			for i := range nx {
				if i > 0 {
					A.Set(i, i-1, 1)
				}
				A.Set(i, nx-1, -coeffs[nx-i])
			}
		} else {
			for i := range nx {
				A.Set(i, i, pole)
			}
		}
	}
	var B, C *mat.Dense
	if nx > 0 {
		B, C = mat.NewDense(nx, nu, nil), mat.NewDense(ny, nx, nil)
	}
	return newTunableSS(op, name, A, B, C, mat.NewDense(ny, nu, nil), ts, free)
}

// NewTunableSSFrom returns the block name initialized from sys realized in
// the astruct form, as MATLAB tunableSS(name, sys, Astruct): tridiag uses the
// Canon modal form, companion the Canon companion form and full sys itself.
// Dt and signal names are copied. A nil or non-finite sys, an empty name or
// an unknown astruct returns ErrInvalidArgument, a descriptor or delayed sys
// ErrDescriptorUnsupported or ErrDelayUnsupported, sys without inputs or
// outputs ErrDimensionMismatch, and a modal form that is not tridiagonal
// (ill-conditioned eigenvectors) or a companion form of a model not
// controllable from its first input ErrSingularTransform.
func NewTunableSSFrom(name string, sys *System, astruct TunableSSStructure) (*TunableSS, error) {
	const op = "NewTunableSSFrom"
	if err := requireTunableName(op, name); err != nil {
		return nil, err
	}
	if err := requireFiniteSystem(op, sys); err != nil {
		return nil, err
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard(op); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree(op); err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()
	if err := requireTunableIO(op, p, m); err != nil {
		return nil, err
	}
	free, err := tunableSSFreeA(op, astruct, n)
	if err != nil {
		return nil, err
	}
	realized := sys
	if n > 0 && astruct != TunableSSFull {
		form := CanonModal
		if astruct == TunableSSCompanion {
			form = CanonCompanion
		}
		c, err := Canon(sys, form)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		realized = c.Sys
		tol := 1e-10 * mat.Norm(realized.A, math.Inf(1))
		for i := range n {
			for j := range n {
				if free(i, j) {
					continue
				}
				if math.Abs(realized.A.At(i, j)) > tol {
					return nil, fmt.Errorf("%s: %s form has A[%d,%d] = %g outside its structure: %w", op, form, i, j, realized.A.At(i, j), ErrSingularTransform)
				}
				realized.A.Set(i, j, 0)
			}
		}
	}
	var A, B, C *mat.Dense
	if n > 0 {
		A, B, C = realized.A, realized.B, realized.C
	}
	b, err := newTunableSS(op, name, A, B, C, realized.D, sys.Dt, free)
	if err != nil {
		return nil, err
	}
	b.InputName, b.OutputName = copyStringSlice(sys.InputName), copyStringSlice(sys.OutputName)
	return b, b.check(op)
}

func tunableSSFreeA(op string, astruct TunableSSStructure, n int) (func(i, j int) bool, error) {
	switch astruct {
	case TunableSSTridiag, "":
		return func(i, j int) bool { return i-j <= 1 && j-i <= 1 }, nil
	case TunableSSFull:
		return func(int, int) bool { return true }, nil
	case TunableSSCompanion:
		return func(i, j int) bool { return j == n-1 || i == j+1 }, nil
	default:
		return nil, fmt.Errorf("%s: Astruct %q, want tridiag, full or companion: %w", op, astruct, ErrInvalidArgument)
	}
}

func newTunableSS(op, name string, A, B, C, D *mat.Dense, ts float64, freeA func(i, j int) bool) (*TunableSS, error) {
	b := &TunableSS{
		name: name,
		A:    newTunableMatrix(name+".A", A, freeA),
		B:    newTunableMatrix(name+".B", B, nil),
		C:    newTunableMatrix(name+".C", C, nil),
		D:    newTunableMatrix(name+".D", D, nil),
		Dt:   ts,
	}
	return b, b.check(op)
}

func (b *TunableSS) check(op string) error {
	if _, err := b.CurrentSystem(); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}

// Name returns the block name.
func (b *TunableSS) Name() string { return b.name }

// CurrentSystem returns the model at the current parameter values.
func (b *TunableSS) CurrentSystem() (*System, error) {
	const op = "TunableSS.CurrentSystem"
	var ms [4]*mat.Dense
	for i, nm := range []struct {
		name string
		p    [][]*TunableReal
	}{{"A", b.A}, {"B", b.B}, {"C", b.C}, {"D", b.D}} {
		m, err := tunableMatrixValues(nm.p, nm.name)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		ms[i] = m
	}
	sys, err := New(ms[0], ms[1], ms[2], ms[3], b.Dt)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	return applyBlockNames(op, sys, b.InputName, b.OutputName)
}

// Sample returns a copy with free parameters set from values.
func (b *TunableSS) Sample(values map[string]float64) (*TunableSS, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.Sample(values) })
}

// RandomSample returns a copy with free parameters drawn uniformly from their
// bounds; see TunableReal.RandomSample.
func (b *TunableSS) RandomSample(rng *rand.Rand) (*TunableSS, error) {
	return b.mapParams(func(p *TunableReal) (*TunableReal, error) { return p.RandomSample(rng) })
}

func (b *TunableSS) mapParams(f func(*TunableReal) (*TunableReal, error)) (*TunableSS, error) {
	var out [4][][]*TunableReal
	for i, m := range [][][]*TunableReal{b.A, b.B, b.C, b.D} {
		mapped, err := mapTunableMatrix(m, f)
		if err != nil {
			return nil, err
		}
		out[i] = mapped
	}
	return &TunableSS{name: b.name, A: out[0], B: out[1], C: out[2], D: out[3], Dt: b.Dt, InputName: copyStringSlice(b.InputName), OutputName: copyStringSlice(b.OutputName)}, nil
}

// FreeParameters returns the distinct free parameters by name; they are the
// block's own parameters.
func (b *TunableSS) FreeParameters() []*TunableReal {
	return uniqueFreeTunableMatrices(b.A, b.B, b.C, b.D)
}

// SampleBlock is Sample as a TunableBlock.
func (b *TunableSS) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

func requireTunableName(op, name string) error {
	if name == "" {
		return fmt.Errorf("%s: name is empty: %w", op, ErrInvalidArgument)
	}
	return nil
}

func requireTunableIO(op string, ny, nu int) error {
	if ny < 0 || nu < 0 {
		return fmt.Errorf("%s: Ny %d and Nu %d must be nonnegative: %w", op, ny, nu, ErrInvalidArgument)
	}
	if ny == 0 || nu == 0 {
		return fmt.Errorf("%s: block is %d×%d; inputs and outputs required: %w", op, ny, nu, ErrDimensionMismatch)
	}
	return nil
}

func requireTunableSampleTime(op string, ts float64) error {
	if err := newTimeDomain(ts).validateSampleTime(); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return nil
}

func applyBlockNames(op string, sys *System, inputName, outputName []string) (*System, error) {
	if inputName != nil {
		if err := sys.SetInputName(inputName...); err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
	}
	if outputName != nil {
		if err := sys.SetOutputName(outputName...); err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
	}
	return sys, nil
}

func newTunableParam(name string, value float64, free bool) *TunableReal {
	return &TunableReal{name: name, value: value, bounds: TunableBounds{Lower: math.Inf(-1), Upper: math.Inf(1)}, fixed: !free}
}

func newTunableVector(prefix string, values []float64) []*TunableReal {
	out := make([]*TunableReal, len(values))
	for k, v := range values {
		out[k] = newTunableParam(fmt.Sprintf("%s(%d)", prefix, k+1), v, true)
	}
	return out
}

// newTunableMatrix returns parameters "<prefix>(i,j)" holding m, free where
// free reports true (all when free is nil); a nil m yields no rows.
func newTunableMatrix(prefix string, m *mat.Dense, free func(i, j int) bool) [][]*TunableReal {
	if m == nil || m.IsEmpty() {
		return nil
	}
	r, c := m.Dims()
	out := make([][]*TunableReal, r)
	for i := range r {
		out[i] = make([]*TunableReal, c)
		for j := range c {
			out[i][j] = newTunableParam(fmt.Sprintf("%s(%d,%d)", prefix, i+1, j+1), m.At(i, j), free == nil || free(i, j))
		}
	}
	return out
}

func tunableScalarValues(params ...*TunableReal) ([]float64, error) {
	out := make([]float64, len(params))
	for k, p := range params {
		if p == nil {
			return nil, fmt.Errorf("parameter %d is nil: %w", k, ErrInvalidArgument)
		}
		out[k] = p.Value()
	}
	return out, nil
}

func tunableMatrixValues(params [][]*TunableReal, context string) (*mat.Dense, error) {
	if len(params) == 0 || len(params[0]) == 0 {
		return &mat.Dense{}, nil
	}
	cols := len(params[0])
	data := make([]float64, 0, len(params)*cols)
	for i, row := range params {
		if len(row) != cols {
			return nil, fmt.Errorf("%s: ragged row %d: %w", context, i, ErrDimensionMismatch)
		}
		for _, param := range row {
			if param == nil {
				return nil, fmt.Errorf("%s: nil parameter: %w", context, ErrInvalidArgument)
			}
			data = append(data, param.Value())
		}
	}
	return mat.NewDense(len(params), cols, data), nil
}

func mapTunableVector(params []*TunableReal, f func(*TunableReal) (*TunableReal, error)) ([]*TunableReal, error) {
	out := make([]*TunableReal, len(params))
	for k, p := range params {
		mapped, err := f(p)
		if err != nil {
			return nil, err
		}
		out[k] = mapped
	}
	return out, nil
}

func mapTunableMatrix(params [][]*TunableReal, f func(*TunableReal) (*TunableReal, error)) ([][]*TunableReal, error) {
	if params == nil {
		return nil, nil
	}
	out := make([][]*TunableReal, len(params))
	for i := range params {
		row, err := mapTunableVector(params[i], f)
		if err != nil {
			return nil, err
		}
		out[i] = row
	}
	return out, nil
}

func uniqueFreeTunableMatrices(matrices ...[][]*TunableReal) []*TunableReal {
	seen := make(map[string]bool)
	var out []*TunableReal
	for _, matrix := range matrices {
		for _, row := range matrix {
			for _, param := range row {
				if param == nil || param.Fixed() || seen[param.Name()] {
					continue
				}
				seen[param.Name()] = true
				out = append(out, param)
			}
		}
	}
	return out
}

func copyFloatRows(rows [][]float64) [][]float64 {
	if rows == nil {
		return nil
	}
	out := make([][]float64, len(rows))
	for i := range rows {
		out[i] = copyFloatSlice(rows[i])
	}
	return out
}
