package controlsys

import (
	"fmt"
	"math/rand"

	"gonum.org/v1/gonum/mat"
)

type TunableBounds struct {
	Lower float64
	Upper float64
}

type TunableReal struct {
	name   string
	value  float64
	bounds TunableBounds
	fixed  bool
}

func NewTunableReal(name string, value float64, bounds TunableBounds) (*TunableReal, error) {
	if name == "" {
		return nil, fmt.Errorf("NewTunableReal: name is empty: %w", ErrDimensionMismatch)
	}
	if bounds.Lower != 0 || bounds.Upper != 0 {
		if bounds.Lower > bounds.Upper {
			return nil, fmt.Errorf("NewTunableReal: lower bound %g > upper bound %g: %w", bounds.Lower, bounds.Upper, ErrDimensionMismatch)
		}
		if value < bounds.Lower || value > bounds.Upper {
			return nil, fmt.Errorf("NewTunableReal: value %g outside [%g,%g]: %w", value, bounds.Lower, bounds.Upper, ErrDimensionMismatch)
		}
	}
	return &TunableReal{name: name, value: value, bounds: bounds}, nil
}

func (p *TunableReal) Name() string {
	if p == nil {
		return ""
	}
	return p.name
}

func (p *TunableReal) Value() float64 {
	if p == nil {
		return 0
	}
	return p.value
}

func (p *TunableReal) Bounds() TunableBounds {
	if p == nil {
		return TunableBounds{}
	}
	return p.bounds
}

func (p *TunableReal) Fixed() bool {
	return p != nil && p.fixed
}

func (p *TunableReal) SetFixed(fixed bool) {
	if p != nil {
		p.fixed = fixed
	}
}

func (p *TunableReal) SetValue(value float64) error {
	if p == nil {
		return fmt.Errorf("TunableReal.SetValue: nil parameter: %w", ErrDimensionMismatch)
	}
	if err := p.validateValue(value); err != nil {
		return err
	}
	p.value = value
	return nil
}

func (p *TunableReal) Sample(values map[string]float64) (*TunableReal, error) {
	cp := p.copy()
	if cp == nil {
		return nil, fmt.Errorf("TunableReal.Sample: nil parameter: %w", ErrDimensionMismatch)
	}
	if cp.fixed {
		return cp, nil
	}
	value, ok := values[cp.name]
	if !ok {
		return cp, nil
	}
	if err := cp.SetValue(value); err != nil {
		return nil, err
	}
	return cp, nil
}

func (p *TunableReal) RandomSample(rng *rand.Rand) (*TunableReal, error) {
	cp := p.copy()
	if cp == nil {
		return nil, fmt.Errorf("TunableReal.RandomSample: nil parameter: %w", ErrDimensionMismatch)
	}
	if cp.fixed || (cp.bounds.Lower == 0 && cp.bounds.Upper == 0) {
		return cp, nil
	}
	if rng == nil {
		rng = rand.New(rand.NewSource(1))
	}
	cp.value = cp.bounds.Lower + rng.Float64()*(cp.bounds.Upper-cp.bounds.Lower)
	return cp, nil
}

func (p *TunableReal) validateValue(value float64) error {
	if p.bounds.Lower == 0 && p.bounds.Upper == 0 {
		return nil
	}
	if value < p.bounds.Lower || value > p.bounds.Upper {
		return fmt.Errorf("TunableReal.SetValue: value %g outside [%g,%g]: %w", value, p.bounds.Lower, p.bounds.Upper, ErrDimensionMismatch)
	}
	return nil
}

func (p *TunableReal) copy() *TunableReal {
	if p == nil {
		return nil
	}
	cp := *p
	return &cp
}

type TunableGain struct {
	name string
	dt   float64
	D    [][]*TunableReal
}

func NewTunableGain(name string, D [][]*TunableReal, dt float64) *TunableGain {
	return &TunableGain{name: name, D: copyTunableMatrix(D), dt: dt}
}

func (b *TunableGain) CurrentSystem() (*System, error) {
	D, err := tunableMatrixValues(b.D, "TunableGain")
	if err != nil {
		return nil, err
	}
	return NewGain(D, b.dt)
}

func (b *TunableGain) Sample(values map[string]float64) (*TunableGain, error) {
	D, err := sampleTunableMatrix(b.D, values)
	if err != nil {
		return nil, err
	}
	return &TunableGain{name: b.name, D: D, dt: b.dt}, nil
}

func (b *TunableGain) RandomSample(rng *rand.Rand) (*TunableGain, error) {
	D, err := randomSampleTunableMatrix(b.D, rng)
	if err != nil {
		return nil, err
	}
	return &TunableGain{name: b.name, D: D, dt: b.dt}, nil
}

func (b *TunableGain) FreeParameters() []*TunableReal {
	if b == nil {
		return nil
	}
	return uniqueFreeTunableReals(b.D)
}

func (b *TunableGain) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

type TunablePID struct {
	IFormula, DFormula PIDFormula
	name               string
	Kp, Ki, Kd         *TunableReal
	Tf                 float64
	Dt                 float64
}

func NewTunablePID(name string, kp, ki, kd *TunableReal, tf, dt float64) *TunablePID {
	return &TunablePID{name: name, Kp: kp.copy(), Ki: ki.copy(), Kd: kd.copy(), Tf: tf, Dt: dt}
}

func (b *TunablePID) CurrentSystem() (*System, error) {
	pid := NewPID(b.Kp.Value(), b.Ki.Value(), b.Kd.Value(), WithFilter(b.Tf), WithTs(b.Dt), WithPIDFormulas(b.IFormula, b.DFormula))
	return pid.System()
}

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

func (b *TunablePID) FreeParameters() []*TunableReal {
	if b == nil {
		return nil
	}
	return uniqueFreeTunableReals([][]*TunableReal{{b.Kp, b.Ki, b.Kd}})
}

func (b *TunablePID) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

type TunableTF struct {
	name string
	Num  [][][]*TunableReal
	Den  [][]float64
	Dt   float64
}

func NewTunableTF(name string, num [][][]*TunableReal, den [][]float64, dt float64) *TunableTF {
	return &TunableTF{name: name, Num: copyTunableTensor(num), Den: copyFloatRows(den), Dt: dt}
}

func (b *TunableTF) CurrentSystem() (*System, error) {
	p := len(b.Num)
	num := make([][][]float64, p)
	for i := range b.Num {
		num[i] = make([][]float64, len(b.Num[i]))
		for j := range b.Num[i] {
			num[i][j] = make([]float64, len(b.Num[i][j]))
			for k, param := range b.Num[i][j] {
				if param == nil {
					return nil, fmt.Errorf("TunableTF: nil numerator parameter: %w", ErrDimensionMismatch)
				}
				num[i][j][k] = param.Value()
			}
		}
	}
	tf := &TransferFunc{Num: num, Den: copyFloatRows(b.Den), Dt: b.Dt}
	result, err := tf.StateSpace(nil)
	if err != nil {
		return nil, err
	}
	return result.Sys, nil
}

func (b *TunableTF) Sample(values map[string]float64) (*TunableTF, error) {
	num, err := sampleTunableTensor(b.Num, values)
	if err != nil {
		return nil, err
	}
	return &TunableTF{name: b.name, Num: num, Den: copyFloatRows(b.Den), Dt: b.Dt}, nil
}

func (b *TunableTF) RandomSample(rng *rand.Rand) (*TunableTF, error) {
	num, err := randomSampleTunableTensor(b.Num, rng)
	if err != nil {
		return nil, err
	}
	return &TunableTF{name: b.name, Num: num, Den: copyFloatRows(b.Den), Dt: b.Dt}, nil
}

func (b *TunableTF) FreeParameters() []*TunableReal {
	if b == nil {
		return nil
	}
	return uniqueFreeTunableTensors(b.Num)
}

func (b *TunableTF) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
}

type TunableSS struct {
	name       string
	A, B, C, D [][]*TunableReal
	Dt         float64
}

func NewTunableSS(name string, A, B, C, D [][]*TunableReal, dt float64) *TunableSS {
	return &TunableSS{
		name: name,
		A:    copyTunableMatrix(A),
		B:    copyTunableMatrix(B),
		C:    copyTunableMatrix(C),
		D:    copyTunableMatrix(D),
		Dt:   dt,
	}
}

func (b *TunableSS) CurrentSystem() (*System, error) {
	A, err := tunableMatrixValues(b.A, "TunableSS.A")
	if err != nil {
		return nil, err
	}
	B, err := tunableMatrixValues(b.B, "TunableSS.B")
	if err != nil {
		return nil, err
	}
	C, err := tunableMatrixValues(b.C, "TunableSS.C")
	if err != nil {
		return nil, err
	}
	D, err := tunableMatrixValues(b.D, "TunableSS.D")
	if err != nil {
		return nil, err
	}
	return New(A, B, C, D, b.Dt)
}

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

func (b *TunableSS) FreeParameters() []*TunableReal {
	if b == nil {
		return nil
	}
	return uniqueFreeTunableMatrices(b.A, b.B, b.C, b.D)
}

func (b *TunableSS) SampleBlock(values map[string]float64) (TunableBlock, error) {
	return b.Sample(values)
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
				return nil, fmt.Errorf("%s: nil parameter: %w", context, ErrDimensionMismatch)
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

func uniqueFreeTunableTensors(tensors ...[][][]*TunableReal) []*TunableReal {
	seen := make(map[string]bool)
	var out []*TunableReal
	for _, tensor := range tensors {
		for _, matrix := range tensor {
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
