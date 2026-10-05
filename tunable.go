package controlsys

import (
	"fmt"
	"math"
	"math/rand/v2"

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
// MATLAB tunableGain. Unlike tunableGain(name, ny, nu), which creates
// zero-initialized parameters, the constructor takes the parameter matrix.
type TunableGain struct {
	name string
	dt   float64
	D    [][]*TunableReal
}

// NewTunableGain returns the gain block name with entries D (copied), see
// https://www.mathworks.com/help/control/ref/tunablegain.html. An empty name,
// a nil or empty D or nil entry returns ErrInvalidArgument, a ragged D
// ErrDimensionMismatch and an invalid dt ErrInvalidSampleTime.
func NewTunableGain(name string, D [][]*TunableReal, dt float64) (*TunableGain, error) {
	if err := requireTunableBlock("NewTunableGain", name, dt, D); err != nil {
		return nil, err
	}
	b := &TunableGain{name: name, D: copyTunableMatrix(D), dt: dt}
	if _, err := b.CurrentSystem(); err != nil {
		return nil, fmt.Errorf("NewTunableGain: %w", err)
	}
	return b, nil
}

// CurrentSystem returns the gain at the current parameter values.
func (b *TunableGain) CurrentSystem() (*System, error) {
	D, err := tunableMatrixValues(b.D, "D")
	if err != nil {
		return nil, fmt.Errorf("TunableGain.CurrentSystem: %w", err)
	}
	if D.IsEmpty() {
		return nil, fmt.Errorf("TunableGain.CurrentSystem: D is empty: %w", ErrInvalidArgument)
	}
	return NewGain(D, b.dt)
}

// Sample returns a copy with free parameters set from values.
func (b *TunableGain) Sample(values map[string]float64) (*TunableGain, error) {
	D, err := sampleTunableMatrix(b.D, values)
	if err != nil {
		return nil, err
	}
	return &TunableGain{name: b.name, D: D, dt: b.dt}, nil
}

// RandomSample returns a copy with free parameters drawn uniformly from their
// bounds; see TunableReal.RandomSample.
func (b *TunableGain) RandomSample(rng *rand.Rand) (*TunableGain, error) {
	D, err := randomSampleTunableMatrix(b.D, rng)
	if err != nil {
		return nil, err
	}
	return &TunableGain{name: b.name, D: D, dt: b.dt}, nil
}

// FreeParameters returns the distinct free parameters by name. They are the
// block's own parameters, so SetValue or SetFixed on them changes the block.
func (b *TunableGain) FreeParameters() []*TunableReal {
	return uniqueFreeTunableMatrices(b.D)
}

// SampleBlock is Sample as a TunableBlock.
func (b *TunableGain) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

// TunablePID is a PID controller with tunable gains Kp, Ki, Kd and a fixed
// derivative filter time constant Tf, as MATLAB tunablePID. Unlike
// tunablePID(name, type), the constructor takes the gain parameters; hold a
// term at zero with a fixed TunableReal of value 0.
type TunablePID struct {
	IFormula, DFormula PIDFormula
	name               string
	Kp, Ki, Kd         *TunableReal
	Tf                 float64
	Dt                 float64
}

// NewTunablePID returns the PID block name with gains kp, ki, kd (copied),
// filter time constant tf and sample time dt; see
// https://www.mathworks.com/help/control/ref/tunablepid.html. A nil gain, an
// empty name or a negative or non-finite tf returns ErrInvalidArgument.
func NewTunablePID(name string, kp, ki, kd *TunableReal, tf, dt float64) (*TunablePID, error) {
	if err := requireTunableBlock("NewTunablePID", name, dt, [][]*TunableReal{{kp, ki, kd}}); err != nil {
		return nil, err
	}
	if !(tf >= 0) || math.IsInf(tf, 0) {
		return nil, fmt.Errorf("NewTunablePID: tf is %g: %w", tf, ErrInvalidArgument)
	}
	b := &TunablePID{name: name, Kp: kp.copy(), Ki: ki.copy(), Kd: kd.copy(), Tf: tf, Dt: dt}
	if _, err := b.CurrentSystem(); err != nil {
		return nil, fmt.Errorf("NewTunablePID: %w", err)
	}
	return b, nil
}

// CurrentSystem returns the PID controller at the current gains.
func (b *TunablePID) CurrentSystem() (*System, error) {
	if b.Kp == nil || b.Ki == nil || b.Kd == nil {
		return nil, fmt.Errorf("TunablePID.CurrentSystem: nil gain parameter: %w", ErrInvalidArgument)
	}
	pid := &PID{Kp: b.Kp.Value(), Ki: b.Ki.Value(), Kd: b.Kd.Value(), Tf: b.Tf, Dt: b.Dt, IFormula: b.IFormula, DFormula: b.DFormula}
	return pid.System()
}

// Sample returns a copy with free gains set from values.
func (b *TunablePID) Sample(values map[string]float64) (*TunablePID, error) {
	kp, err := b.Kp.Sample(values)
	if err != nil {
		return nil, err
	}
	ki, err := b.Ki.Sample(values)
	if err != nil {
		return nil, err
	}
	kd, err := b.Kd.Sample(values)
	if err != nil {
		return nil, err
	}
	return &TunablePID{name: b.name, Kp: kp, Ki: ki, Kd: kd, Tf: b.Tf, Dt: b.Dt, IFormula: b.IFormula, DFormula: b.DFormula}, nil
}

// RandomSample returns a copy with free gains drawn uniformly from their
// bounds; see TunableReal.RandomSample.
func (b *TunablePID) RandomSample(rng *rand.Rand) (*TunablePID, error) {
	kp, err := b.Kp.RandomSample(rng)
	if err != nil {
		return nil, err
	}
	ki, err := b.Ki.RandomSample(rng)
	if err != nil {
		return nil, err
	}
	kd, err := b.Kd.RandomSample(rng)
	if err != nil {
		return nil, err
	}
	return &TunablePID{name: b.name, Kp: kp, Ki: ki, Kd: kd, Tf: b.Tf, Dt: b.Dt, IFormula: b.IFormula, DFormula: b.DFormula}, nil
}

// FreeParameters returns the distinct free gains by name; they are the
// block's own parameters.
func (b *TunablePID) FreeParameters() []*TunableReal {
	return uniqueFreeTunableMatrices([][]*TunableReal{{b.Kp, b.Ki, b.Kd}})
}

// SampleBlock is Sample as a TunableBlock.
func (b *TunablePID) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

// TunableTF is a transfer function with tunable numerator coefficients and a
// fixed denominator per output row, as MATLAB tunableTF (which also tunes the
// denominator; here Den is fixed).
type TunableTF struct {
	name string
	Num  [][][]*TunableReal
	Den  [][]float64
	Dt   float64
}

// NewTunableTF returns the block name with numerator parameters num[i][j]
// (descending powers, copied) and row denominators den; see
// https://www.mathworks.com/help/control/ref/tunabletf.html. The model must
// be proper and well formed.
func NewTunableTF(name string, num [][][]*TunableReal, den [][]float64, dt float64) (*TunableTF, error) {
	if err := requireTunableBlock("NewTunableTF", name, dt, num...); err != nil {
		return nil, err
	}
	if len(num) == 0 || len(den) == 0 {
		return nil, fmt.Errorf("NewTunableTF: num or den is empty: %w", ErrInvalidArgument)
	}
	b := &TunableTF{name: name, Num: copyTunableTensor(num), Den: copyFloatRows(den), Dt: dt}
	if _, err := b.CurrentSystem(); err != nil {
		return nil, fmt.Errorf("NewTunableTF: %w", err)
	}
	return b, nil
}

// CurrentSystem realizes the transfer function at the current coefficients.
func (b *TunableTF) CurrentSystem() (*System, error) {
	p := len(b.Num)
	num := make([][][]float64, p)
	for i := range b.Num {
		num[i] = make([][]float64, len(b.Num[i]))
		for j := range b.Num[i] {
			num[i][j] = make([]float64, len(b.Num[i][j]))
			for k, param := range b.Num[i][j] {
				if param == nil {
					return nil, fmt.Errorf("TunableTF.CurrentSystem: nil numerator parameter: %w", ErrInvalidArgument)
				}
				num[i][j][k] = param.Value()
			}
		}
	}
	tf := &TransferFunc{Num: num, Den: copyFloatRows(b.Den), Dt: b.Dt}
	result, err := tf.StateSpace(nil)
	if err != nil {
		return nil, fmt.Errorf("TunableTF.CurrentSystem: %w", err)
	}
	return result.Sys, nil
}

// Sample returns a copy with free coefficients set from values.
func (b *TunableTF) Sample(values map[string]float64) (*TunableTF, error) {
	num, err := sampleTunableTensor(b.Num, values)
	if err != nil {
		return nil, err
	}
	return &TunableTF{name: b.name, Num: num, Den: copyFloatRows(b.Den), Dt: b.Dt}, nil
}

// RandomSample returns a copy with free coefficients drawn uniformly from
// their bounds; see TunableReal.RandomSample.
func (b *TunableTF) RandomSample(rng *rand.Rand) (*TunableTF, error) {
	num, err := randomSampleTunableTensor(b.Num, rng)
	if err != nil {
		return nil, err
	}
	return &TunableTF{name: b.name, Num: num, Den: copyFloatRows(b.Den), Dt: b.Dt}, nil
}

// FreeParameters returns the distinct free coefficients by name; they are
// the block's own parameters.
func (b *TunableTF) FreeParameters() []*TunableReal {
	return uniqueFreeTunableMatrices(b.Num...)
}

// SampleBlock is Sample as a TunableBlock.
func (b *TunableTF) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

// TunableSS is a state-space model with tunable A, B, C, D entries, as
// MATLAB tunableSS. Unlike tunableSS(name, nx, ny, nu), the constructor takes
// the parameter matrices.
type TunableSS struct {
	name       string
	A, B, C, D [][]*TunableReal
	Dt         float64
}

// NewTunableSS returns the block name with parameter matrices A, B, C, D
// (copied); see https://www.mathworks.com/help/control/ref/tunabless.html.
// Nil entries return ErrInvalidArgument and inconsistent sizes
// ErrDimensionMismatch.
func NewTunableSS(name string, A, B, C, D [][]*TunableReal, dt float64) (*TunableSS, error) {
	if err := requireTunableBlock("NewTunableSS", name, dt, A, B, C, D); err != nil {
		return nil, err
	}
	b := &TunableSS{
		name: name,
		A:    copyTunableMatrix(A),
		B:    copyTunableMatrix(B),
		C:    copyTunableMatrix(C),
		D:    copyTunableMatrix(D),
		Dt:   dt,
	}
	if _, err := b.CurrentSystem(); err != nil {
		return nil, fmt.Errorf("NewTunableSS: %w", err)
	}
	return b, nil
}

// CurrentSystem returns the model at the current parameter values.
func (b *TunableSS) CurrentSystem() (*System, error) {
	var ms [4]*mat.Dense
	for i, nm := range []struct {
		name string
		p    [][]*TunableReal
	}{{"A", b.A}, {"B", b.B}, {"C", b.C}, {"D", b.D}} {
		m, err := tunableMatrixValues(nm.p, nm.name)
		if err != nil {
			return nil, fmt.Errorf("TunableSS.CurrentSystem: %w", err)
		}
		ms[i] = m
	}
	sys, err := New(ms[0], ms[1], ms[2], ms[3], b.Dt)
	if err != nil {
		return nil, fmt.Errorf("TunableSS.CurrentSystem: %w", err)
	}
	return sys, nil
}

// Sample returns a copy with free parameters set from values.
func (b *TunableSS) Sample(values map[string]float64) (*TunableSS, error) {
	A, err := sampleTunableMatrix(b.A, values)
	if err != nil {
		return nil, err
	}
	B, err := sampleTunableMatrix(b.B, values)
	if err != nil {
		return nil, err
	}
	C, err := sampleTunableMatrix(b.C, values)
	if err != nil {
		return nil, err
	}
	D, err := sampleTunableMatrix(b.D, values)
	if err != nil {
		return nil, err
	}
	return &TunableSS{name: b.name, A: A, B: B, C: C, D: D, Dt: b.Dt}, nil
}

// RandomSample returns a copy with free parameters drawn uniformly from their
// bounds; see TunableReal.RandomSample.
func (b *TunableSS) RandomSample(rng *rand.Rand) (*TunableSS, error) {
	A, err := randomSampleTunableMatrix(b.A, rng)
	if err != nil {
		return nil, err
	}
	B, err := randomSampleTunableMatrix(b.B, rng)
	if err != nil {
		return nil, err
	}
	C, err := randomSampleTunableMatrix(b.C, rng)
	if err != nil {
		return nil, err
	}
	D, err := randomSampleTunableMatrix(b.D, rng)
	if err != nil {
		return nil, err
	}
	return &TunableSS{name: b.name, A: A, B: B, C: C, D: D, Dt: b.Dt}, nil
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

// requireTunableBlock validates a block constructor's name, sample time and
// parameter matrices (nil entries rejected).
func requireTunableBlock(op, name string, dt float64, matrices ...[][]*TunableReal) error {
	if name == "" {
		return fmt.Errorf("%s: name is empty: %w", op, ErrInvalidArgument)
	}
	if err := newTimeDomain(dt).validateSampleTime(); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	for _, m := range matrices {
		for i, row := range m {
			for j, param := range row {
				if param == nil {
					return fmt.Errorf("%s: parameter (%d,%d) is nil: %w", op, i, j, ErrInvalidArgument)
				}
			}
		}
	}
	return nil
}

func tunableMatrixValues(params [][]*TunableReal, context string) (*mat.Dense, error) {
	if len(params) == 0 {
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

func copyTunableMatrix(params [][]*TunableReal) [][]*TunableReal {
	if params == nil {
		return nil
	}
	out := make([][]*TunableReal, len(params))
	for i := range params {
		out[i] = make([]*TunableReal, len(params[i]))
		for j := range params[i] {
			out[i][j] = params[i][j].copy()
		}
	}
	return out
}

func sampleTunableMatrix(params [][]*TunableReal, values map[string]float64) ([][]*TunableReal, error) {
	out := make([][]*TunableReal, len(params))
	for i := range params {
		out[i] = make([]*TunableReal, len(params[i]))
		for j, param := range params[i] {
			sampled, err := param.Sample(values)
			if err != nil {
				return nil, err
			}
			out[i][j] = sampled
		}
	}
	return out, nil
}

func randomSampleTunableMatrix(params [][]*TunableReal, rng *rand.Rand) ([][]*TunableReal, error) {
	out := make([][]*TunableReal, len(params))
	for i := range params {
		out[i] = make([]*TunableReal, len(params[i]))
		for j, param := range params[i] {
			sampled, err := param.RandomSample(rng)
			if err != nil {
				return nil, err
			}
			out[i][j] = sampled
		}
	}
	return out, nil
}

func copyTunableTensor(params [][][]*TunableReal) [][][]*TunableReal {
	if params == nil {
		return nil
	}
	out := make([][][]*TunableReal, len(params))
	for i := range params {
		out[i] = copyTunableMatrix(params[i])
	}
	return out
}

func sampleTunableTensor(params [][][]*TunableReal, values map[string]float64) ([][][]*TunableReal, error) {
	out := make([][][]*TunableReal, len(params))
	for i := range params {
		sampled, err := sampleTunableMatrix(params[i], values)
		if err != nil {
			return nil, err
		}
		out[i] = sampled
	}
	return out, nil
}

func randomSampleTunableTensor(params [][][]*TunableReal, rng *rand.Rand) ([][][]*TunableReal, error) {
	out := make([][][]*TunableReal, len(params))
	for i := range params {
		sampled, err := randomSampleTunableMatrix(params[i], rng)
		if err != nil {
			return nil, err
		}
		out[i] = sampled
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
