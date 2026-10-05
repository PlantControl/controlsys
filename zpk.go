package controlsys

import (
	"fmt"
	"math/cmplx"
	"sort"
)

// ZPK is a p×m zero-pole-gain model, as MATLAB zpk: channel (i, j) is
// Gain[i][j]·Π(s-Zeros[i][j])/Π(s-Poles[i][j]).
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

// NewZPK returns the SISO model gain·Π(s-zeros)/Π(s-poles), as MATLAB
// zpk(z, p, k, Ts). Complex zeros and poles must come in conjugate pairs.
func NewZPK(zeros, poles []complex128, gain, dt float64) (*ZPK, error) {
	if err := newTimeDomain(dt).validateSampleTime(); err != nil {
		return nil, fmt.Errorf("NewZPK: %w", err)
	}
	if err := validatePoles(zeros); err != nil {
		return nil, fmt.Errorf("NewZPK: zeros: %w", err)
	}
	if err := validatePoles(poles); err != nil {
		return nil, fmt.Errorf("NewZPK: poles: %w", err)
	}
	return &ZPK{
		Zeros: [][][]complex128{{copyComplex(zeros)}},
		Poles: [][][]complex128{{copyComplex(poles)}},
		Gain:  [][]float64{{gain}},
		Dt:    dt,
	}, nil
}

// NewZPKMIMO returns the p×m model with channel (i, j) given by zeros[i][j],
// poles[i][j] and gain[i][j], as MATLAB zpk(Z, P, K, Ts) with cell arrays.
func NewZPKMIMO(zeros, poles [][][]complex128, gain [][]float64, dt float64) (*ZPK, error) {
	if err := newTimeDomain(dt).validateSampleTime(); err != nil {
		return nil, fmt.Errorf("NewZPKMIMO: %w", err)
	}
	p := len(gain)
	if p == 0 {
		return nil, fmt.Errorf("NewZPKMIMO: gain has no rows: %w", ErrDimensionMismatch)
	}
	m := len(gain[0])
	if m == 0 {
		return nil, fmt.Errorf("NewZPKMIMO: gain has no columns: %w", ErrDimensionMismatch)
	}
	if len(zeros) != p || len(poles) != p {
		return nil, fmt.Errorf("NewZPKMIMO: zeros has %d rows and poles %d, want %d: %w", len(zeros), len(poles), p, ErrDimensionMismatch)
	}
	for i := range p {
		if len(gain[i]) != m || len(zeros[i]) != m || len(poles[i]) != m {
			return nil, fmt.Errorf("NewZPKMIMO: row %d has %d gains, %d zero sets and %d pole sets, want %d: %w", i, len(gain[i]), len(zeros[i]), len(poles[i]), m, ErrDimensionMismatch)
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
				return nil, fmt.Errorf("NewZPKMIMO: zeros (%d,%d): %w", i, j, err)
			}
			if err := validatePoles(poles[i][j]); err != nil {
				return nil, fmt.Errorf("NewZPKMIMO: poles (%d,%d): %w", i, j, err)
			}
			zCopy[i][j] = copyComplex(zeros[i][j])
			pCopy[i][j] = copyComplex(poles[i][j])
		}
	}
	return &ZPK{Zeros: zCopy, Poles: pCopy, Gain: gCopy, Dt: dt}, nil
}

// Dims returns the number of outputs p and inputs m.
func (z *ZPK) Dims() (p, m int) {
	p = len(z.Gain)
	if p > 0 {
		m = len(z.Gain[0])
	}
	return
}

func (z *ZPK) validateShape() (p, m int, err error) {
	p, m, err = validateZPKChannelShape(z)
	if err != nil {
		return 0, 0, err
	}
	if err := newTimeDomain(z.Dt).validateSampleTime(); err != nil {
		return 0, 0, err
	}
	return p, m, nil
}

// IsContinuous reports whether z is a continuous-time model (Dt = 0).
func (z *ZPK) IsContinuous() bool { return z.Dt == 0 }

// IsDiscrete reports whether z is a discrete-time model (Dt > 0).
func (z *ZPK) IsDiscrete() bool { return z.Dt > 0 }

// Eval evaluates the frequency response H(s) at the complex point s, as
// MATLAB evalfr(sys, s). A malformed model returns an error.
func (z *ZPK) Eval(s complex128) ([][]complex128, error) {
	p, m, err := z.validateShape()
	if err != nil {
		return nil, fmt.Errorf("ZPK.Eval: %w", err)
	}
	result := make([][]complex128, p)
	for i := range p {
		result[i] = make([]complex128, m)
		for j := range m {
			result[i][j] = zpkEvalChannel(s, z.Zeros[i][j], z.Poles[i][j], z.Gain[i][j])
		}
	}
	return result, nil
}

func zpkEvalChannel(s complex128, zeros, poles []complex128, gain float64) complex128 {
	return newRationalChannel(zeros, poles, gain).eval(s)
}

// FreqResponse evaluates the frequency response at the real frequencies
// omega (rad/s), as MATLAB freqresp. An empty or non-finite omega returns
// ErrInvalidArgument.
func (z *ZPK) FreqResponse(omega []float64) (*FreqResponseMatrix, error) {
	p, m, err := z.validateShape()
	if err != nil {
		return nil, fmt.Errorf("ZPK.FreqResponse: %w", err)
	}
	if len(omega) == 0 {
		return nil, fmt.Errorf("ZPK.FreqResponse: omega is empty: %w", ErrInvalidArgument)
	}
	if err := requireFinite("ZPK.FreqResponse", "omega", omega...); err != nil {
		return nil, err
	}
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

// TransferFunction converts z to transfer-function form, as MATLAB tf(zpk).
func (z *ZPK) TransferFunction() (*TransferFunc, error) {
	tf, err := z.transferFunction()
	if err != nil {
		return nil, fmt.Errorf("ZPK.TransferFunction: %w", err)
	}
	return tf, nil
}

func (z *ZPK) transferFunction() (*TransferFunc, error) {
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

// ZPK converts tf to zero-pole-gain form, as MATLAB zpk(tf). Delays cannot
// be represented and return ErrDelayNotRepresentable.
func (tf *TransferFunc) ZPK() (*ZPK, error) {
	p, m, err := tf.validateShape()
	if err != nil {
		return nil, fmt.Errorf("TransferFunc.ZPK: %w", err)
	}
	if tf.HasDelay() {
		return nil, fmt.Errorf("TransferFunc.ZPK: %w", ErrDelayNotRepresentable)
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
				return nil, fmt.Errorf("TransferFunc.ZPK: channel (%d,%d): %w", i, j, err)
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

// ZPKModel returns the zero-pole-gain form of sys via TransferFunction, as
// MATLAB zpk(ss).
func (sys *System) ZPKModel(opts *TransferFuncOpts) (*ZPKResult, error) {
	tfResult, err := sys.TransferFunction(opts)
	if err != nil {
		return nil, fmt.Errorf("ZPKModel: %w", err)
	}
	zpk, err := tfResult.TF.ZPK()
	if err != nil {
		return nil, fmt.Errorf("ZPKModel: %w", err)
	}
	zpk.InputName = copyStringSlice(sys.InputName)
	zpk.OutputName = copyStringSlice(sys.OutputName)
	return &ZPKResult{
		ZPK:          zpk,
		MinimalOrder: tfResult.MinimalOrder,
		RowDegrees:   tfResult.RowDegrees,
	}, nil
}

// ZPKResult is the result of (*System).ZPKModel.
type ZPKResult struct {
	ZPK          *ZPK
	MinimalOrder int
	RowDegrees   []int
}

// StateSpace realizes z via its transfer function, as MATLAB ss(zpk); see
// (*TransferFunc).StateSpace.
func (z *ZPK) StateSpace() (*StateSpaceResult, error) {
	tf, err := z.transferFunction()
	if err != nil {
		return nil, fmt.Errorf("ZPK.StateSpace: %w", err)
	}
	res, err := tf.stateSpace()
	if err != nil {
		return nil, fmt.Errorf("ZPK.StateSpace: %w", err)
	}
	return res, nil
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
