package controlsys

import (
	"fmt"
	"math/cmplx"
	"sort"
)

type ZPK struct {
	Zeros      [][][]complex128
	Poles      [][][]complex128
	Gain       [][]float64
	Dt         float64
	InputName  []string
	OutputName []string
}

// Copy returns a deep copy of the zero-pole-gain model.
func (z *ZPK) Copy() *ZPK {
	if z == nil {
		return nil
	}
	return &ZPK{
		Zeros:      copyComplexTensor(z.Zeros),
		Poles:      copyComplexTensor(z.Poles),
		Gain:       copyFloatRows(z.Gain),
		Dt:         z.Dt,
		InputName:  copyStringSlice(z.InputName),
		OutputName: copyStringSlice(z.OutputName),
	}
}

func NewZPK(zeros, poles []complex128, gain, dt float64) (*ZPK, error) {
	if err := newTimeDomain(dt).validateSampleTime(); err != nil {
		return nil, err
	}
	if err := validatePoles(zeros); err != nil {
		return nil, err
	}
	if err := validatePoles(poles); err != nil {
		return nil, err
	}
	return &ZPK{
		Zeros: [][][]complex128{{copyComplex(zeros)}},
		Poles: [][][]complex128{{copyComplex(poles)}},
		Gain:  [][]float64{{gain}},
		Dt:    dt,
	}, nil
}

func NewZPKMIMO(zeros, poles [][][]complex128, gain [][]float64, dt float64) (*ZPK, error) {
	if err := newTimeDomain(dt).validateSampleTime(); err != nil {
		return nil, err
	}
	p := len(gain)
	if p == 0 {
		return nil, ErrDimensionMismatch
	}
	m := len(gain[0])
	if m == 0 {
		return nil, ErrDimensionMismatch
	}
	if len(zeros) != p || len(poles) != p {
		return nil, ErrDimensionMismatch
	}
	for i := range p {
		if len(gain[i]) != m || len(zeros[i]) != m || len(poles[i]) != m {
			return nil, ErrDimensionMismatch
		}
	}

	zCopy := make([][][]complex128, p)
	pCopy := make([][][]complex128, p)
	gCopy := make([][]float64, p)
	for i := range p {
		zCopy[i] = make([][]complex128, m)
		pCopy[i] = make([][]complex128, m)
		gCopy[i] = make([]float64, m)
		copy(gCopy[i], gain[i])
		for j := range m {
			if err := validatePoles(zeros[i][j]); err != nil {
				return nil, err
			}
			if err := validatePoles(poles[i][j]); err != nil {
				return nil, err
			}
			zCopy[i][j] = copyComplex(zeros[i][j])
			pCopy[i][j] = copyComplex(poles[i][j])
		}
	}
	return &ZPK{Zeros: zCopy, Poles: pCopy, Gain: gCopy, Dt: dt}, nil
}

func (z *ZPK) Dims() (p, m int) {
	p = len(z.Gain)
	if p > 0 {
		m = len(z.Gain[0])
	}
	return
}

func (z *ZPK) validateShape() (p, m int, err error) {
	return validateZPKChannelShape(z)
}

func (z *ZPK) IsContinuous() bool { return z.Dt == 0 }
func (z *ZPK) IsDiscrete() bool   { return z.Dt > 0 }

func (z *ZPK) Eval(s complex128) [][]complex128 {
	p, m := z.Dims()
	result := make([][]complex128, p)
	for i := range p {
		result[i] = make([]complex128, m)
		for j := range m {
			result[i][j] = zpkEvalChannel(s, z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j])
		}
	}
	return result
}

func zpkEvalChannel(s complex128, zeros, poles []complex128, gain float64) complex128 {
	return newRationalChannel(zeros, poles, gain).eval(s)
}

func (z *ZPK) FreqResponse(omega []float64) (*FreqResponseMatrix, error) {
	if len(omega) == 0 {
		return nil, ErrDimensionMismatch
	}
	p, m := z.Dims()
	data := make([]complex128, len(omega)*p*m)
	continuous := z.IsContinuous()
	dt := z.Dt
	for k, w := range omega {
		var s complex128
		if continuous {
			s = complex(0, w)
		} else {
			s = cmplx.Exp(complex(0, w*dt))
		}
		off := k * p * m
		for i := range p {
			for j := range m {
				data[off+i*m+j] = zpkEvalChannel(s, z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j])
			}
		}
	}
	return newFreqResponseMatrix(data, omega, p, m, z.InputName, z.OutputName), nil
}

func (z *ZPK) TransferFunction() (*TransferFunc, error) {
	p, m, err := z.validateShape()
	if err != nil {
		return nil, err
	}
	tf := &TransferFunc{
		Num: make([][][]float64, p),
		Den: make([][]float64, p),
		Dt:  z.Dt,
	}

	for i := range p {
		commonPoles := commonDenomPoles(z.Poles[i])
		tf.Den[i] = []float64(polyFromComplexRoots(sortConjugatePairs(commonPoles)))
		tf.Num[i] = make([][]float64, m)
		for j := range m {
			ch := newRationalChannel(z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j])
			tf.Num[i][j] = ch.numeratorForCommonPoles(commonPoles)
		}
	}
	tf.InputName = copyStringSlice(z.InputName)
	tf.OutputName = copyStringSlice(z.OutputName)
	return tf, nil
}

func (tf *TransferFunc) ZPK() (*ZPK, error) {
	p, m, err := tf.validateShape()
	if err != nil {
		return nil, err
	}
	if tf.HasDelay() {
		return nil, fmt.Errorf("ZPK: %w", ErrDelayNotRepresentable)
	}
	z := &ZPK{
		Zeros: make([][][]complex128, p),
		Poles: make([][][]complex128, p),
		Gain:  make([][]float64, p),
		Dt:    tf.Dt,
	}

	for i := range p {
		z.Zeros[i] = make([][]complex128, m)
		z.Poles[i] = make([][]complex128, m)
		z.Gain[i] = make([]float64, m)
		for j := range m {
			ch, err := rationalChannelFromPolynomials(tf.Num[i][j], tf.Den[i])
			if err != nil {
				return nil, err
			}
			z.Zeros[i][j] = ch.zeros
			z.Poles[i][j] = copyComplex(ch.poles)
			z.Gain[i][j] = ch.gain
		}
	}
	z.InputName = copyStringSlice(tf.InputName)
	z.OutputName = copyStringSlice(tf.OutputName)
	return z, nil
}

func (sys *System) ZPKModel(opts *TransferFuncOpts) (*ZPKResult, error) {
	tfResult, err := sys.TransferFunction(opts)
	if err != nil {
		return nil, err
	}
	zpk, err := tfResult.TF.ZPK()
	if err != nil {
		return nil, err
	}
	zpk.InputName = copyStringSlice(sys.InputName)
	zpk.OutputName = copyStringSlice(sys.OutputName)
	return &ZPKResult{
		ZPK:          zpk,
		MinimalOrder: tfResult.MinimalOrder,
		RowDegrees:   tfResult.RowDegrees,
	}, nil
}

type ZPKResult struct {
	ZPK          *ZPK
	MinimalOrder int
	RowDegrees   []int
}

func (z *ZPK) StateSpace(opts *StateSpaceOpts) (*StateSpaceResult, error) {
	tf, err := z.TransferFunction()
	if err != nil {
		return nil, err
	}
	return tf.StateSpace(opts)
}

func copyComplex(src []complex128) []complex128 {
	if src == nil {
		return nil
	}
	dst := make([]complex128, len(src))
	copy(dst, src)
	return dst
}

func copyComplexTensor(src [][][]complex128) [][][]complex128 {
	if src == nil {
		return nil
	}
	dst := make([][][]complex128, len(src))
	for i := range src {
		dst[i] = make([][]complex128, len(src[i]))
		for j := range src[i] {
			dst[i][j] = copyComplex(src[i][j])
		}
	}
	return dst
}

func sortConjugatePairs(roots []complex128) []complex128 {
	n := len(roots)
	if n == 0 {
		return nil
	}

	result := make([]complex128, 0, n)
	cmplx_ := make([]complex128, 0, n)
	for _, r := range roots {
		if imag(r) == 0 {
			result = append(result, r)
		} else {
			cmplx_ = append(cmplx_, r)
		}
	}

	sort.Slice(result, func(i, j int) bool { return real(result[i]) < real(result[j]) })

	machTol := 100 * eps()
	used := make([]bool, len(cmplx_))
	for i, r := range cmplx_ {
		if used[i] {
			continue
		}
		if imag(r) < 0 {
			r = cmplx.Conj(r)
		}
		conjR := cmplx.Conj(r)
		tol := machTol * (1 + cmplx.Abs(r))
		matched := false
		for k := i + 1; k < len(cmplx_); k++ {
			if !used[k] && cmplx.Abs(cmplx_[k]-conjR) < tol {
				used[i] = true
				used[k] = true
				matched = true
				result = append(result, r, conjR)
				break
			}
		}
		if !matched {
			used[i] = true
			result = append(result, r, conjR)
		}
	}
	return result
}

func commonDenomPoles(channelPoles [][]complex128) []complex128 {
	if len(channelPoles) == 0 {
		return nil
	}
	if len(channelPoles) == 1 {
		return copyComplex(channelPoles[0])
	}

	tol := 100 * eps()
	result := copyComplex(channelPoles[0])
	for j := 1; j < len(channelPoles); j++ {
		avail := copyComplex(result)
		for _, p := range channelPoles[j] {
			found := false
			matchTol := tol * (1 + cmplx.Abs(p))
			for k := 0; k < len(avail); k++ {
				if cmplx.Abs(avail[k]-p) < matchTol {
					avail = append(avail[:k], avail[k+1:]...)
					found = true
					break
				}
			}
			if !found {
				result = append(result, p)
			}
		}
	}
	return result
}

func poleSetDifference(all, subset []complex128) []complex128 {
	tol := 100 * eps()
	remaining := copyComplex(all)
	for _, p := range subset {
		matchTol := tol * (1 + cmplx.Abs(p))
		for k := 0; k < len(remaining); k++ {
			if cmplx.Abs(remaining[k]-p) < matchTol {
				remaining = append(remaining[:k], remaining[k+1:]...)
				break
			}
		}
	}
	return remaining
}
