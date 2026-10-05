package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"
)

// FRD represents a Frequency Response Data model — measured or computed
// complex response at discrete frequencies.
type FRD struct {
	Response [][][]complex128
	Omega    []float64
	Dt       float64

	InputName  []string
	OutputName []string
}

// Copy returns a deep copy of the frequency-response data model.
func (f *FRD) Copy() *FRD {
	if f == nil {
		return nil
	}
	return &FRD{
		Response:   copyComplexTensor(f.Response),
		Omega:      copyFloatSlice(f.Omega),
		Dt:         f.Dt,
		InputName:  copyStringSlice(f.InputName),
		OutputName: copyStringSlice(f.OutputName),
	}
}

// NewFRD creates an FRD model from response data and frequency vector.
// response[k] is the p*m complex response matrix at frequency omega[k].
func NewFRD(response [][][]complex128, omega []float64, dt float64) (*FRD, error) {
	if dt < 0 {
		return nil, fmt.Errorf("FRD: negative sample time: %w", ErrInvalidSampleTime)
	}
	if len(response) != len(omega) {
		return nil, fmt.Errorf("FRD: len(response)=%d != len(omega)=%d: %w",
			len(response), len(omega), ErrDimensionMismatch)
	}
	if len(omega) == 0 {
		return &FRD{Dt: dt}, nil
	}

	for i := range omega {
		if omega[i] < 0 {
			return nil, fmt.Errorf("FRD: omega[%d]=%v is negative: %w", i, omega[i], ErrDimensionMismatch)
		}
	}
	if !sort.Float64sAreSorted(omega) {
		return nil, fmt.Errorf("FRD: omega must be sorted ascending: %w", ErrDimensionMismatch)
	}

	if dt > 0 {
		nyquist := math.Pi / dt
		for i, w := range omega {
			if w > nyquist*(1+1e-10) {
				return nil, fmt.Errorf("FRD: omega[%d]=%v exceeds Nyquist frequency %v: %w",
					i, w, nyquist, ErrDimensionMismatch)
			}
		}
	}

	p := len(response[0])
	if p == 0 {
		return nil, fmt.Errorf("FRD: response matrix has zero rows: %w", ErrDimensionMismatch)
	}
	m := len(response[0][0])
	if m == 0 {
		return nil, fmt.Errorf("FRD: response matrix has zero columns: %w", ErrDimensionMismatch)
	}

	for k := range response {
		if len(response[k]) != p {
			return nil, fmt.Errorf("FRD: response[%d] has %d rows, want %d: %w",
				k, len(response[k]), p, ErrDimensionMismatch)
		}
		for i := range response[k] {
			if len(response[k][i]) != m {
				return nil, fmt.Errorf("FRD: response[%d][%d] has %d cols, want %d: %w",
					k, i, len(response[k][i]), m, ErrDimensionMismatch)
			}
		}
	}

	resp, data := newFRDResponseStorage(len(response), p, m)
	copyComplexGridInto(data, response, p, m)
	om := make([]float64, len(omega))
	copy(om, omega)

	return &FRD{
		Response: resp,
		Omega:    om,
		Dt:       dt,
	}, nil
}

// FRD computes the frequency response data model at the given frequencies.
func (sys *System) FRD(omega []float64) (*FRD, error) {
	if len(omega) == 0 {
		return &FRD{Dt: sys.Dt}, nil
	}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return nil, err
	}

	_, m, p := sys.Dims()
	nw := len(omega)

	response, data := newFRDResponseStorage(nw, p, m)
	copy(data, resp.Data)

	om := make([]float64, nw)
	copy(om, omega)

	return &FRD{
		Response:   response,
		Omega:      om,
		Dt:         sys.Dt,
		InputName:  copyStringSlice(sys.InputName),
		OutputName: copyStringSlice(sys.OutputName),
	}, nil
}

func newFRDFromFreqResponse(resp *FreqResponseMatrix, dt float64) (*FRD, error) {
	if resp == nil {
		return nil, ErrInsufficientData
	}
	response, data := newFRDResponseStorage(resp.NFreq, resp.P, resp.M)
	copy(data, resp.Data)
	frd, err := NewFRD(response, resp.Omega, dt)
	if err != nil {
		return nil, err
	}
	frd.InputName = copyStringSlice(resp.InputName)
	frd.OutputName = copyStringSlice(resp.OutputName)
	return frd, nil
}

func newFRDResponseStorage(nw, p, m int) ([][][]complex128, []complex128) {
	response := make([][][]complex128, nw)
	if nw == 0 || p == 0 || m == 0 {
		return response, nil
	}
	rows := make([][]complex128, nw*p)
	data := make([]complex128, nw*p*m)
	for k := range nw {
		response[k] = rows[k*p : (k+1)*p]
		block := data[k*p*m : (k+1)*p*m]
		for i := range p {
			start := i * m
			response[k][i] = block[start : start+m : start+m]
		}
	}
	return response, data
}

func copyComplexGridInto(dst []complex128, src [][][]complex128, p, m int) {
	pm := p * m
	for k := range src {
		base := k * pm
		for i := range p {
			copy(dst[base+i*m:base+(i+1)*m], src[k][i])
		}
	}
}

func copyComplexMatrixInto(dst []complex128, src [][]complex128, rows, cols int) {
	for i := range rows {
		copy(dst[i*cols:(i+1)*cols], src[i])
	}
}

func cMulNestedInto(dst []complex128, a, b [][]complex128, ra, ca, cb int) {
	for i := range ra {
		row := dst[i*cb : (i+1)*cb]
		for j := range cb {
			var sum complex128
			for k := range ca {
				sum += a[i][k] * b[k][j]
			}
			row[j] = sum
		}
	}
}

func cAddNestedInto(dst []complex128, a, b [][]complex128, rows, cols int) {
	for i := range rows {
		row := dst[i*cols : (i+1)*cols]
		for j := range cols {
			row[j] = a[i][j] + b[i][j]
		}
	}
}

func (f *FRD) Dims() (p, m int) {
	if len(f.Response) == 0 {
		return 0, 0
	}
	return len(f.Response[0]), len(f.Response[0][0])
}

func (f *FRD) NumFrequencies() int {
	return len(f.Omega)
}

func (f *FRD) IsContinuous() bool {
	return f.Dt == 0
}

func (f *FRD) IsDiscrete() bool {
	return f.Dt > 0
}

func (f *FRD) At(freqIdx, i, j int) complex128 {
	return f.Response[freqIdx][i][j]
}

type FRDResponseMapper func(freq int, omega float64, h [][]complex128) ([][]complex128, error)

type FRDPeakGainResult struct {
	Gain      float64
	Frequency float64
	Index     int
}

func (f *FRD) Abs() *FRD {
	p, m := f.Dims()
	nw := len(f.Omega)
	response, data := newFRDResponseStorage(nw, p, m)
	pm := p * m
	for k := range nw {
		base := k * pm
		for i := range p {
			for j := range m {
				data[base+i*m+j] = complex(cmplx.Abs(f.Response[k][i][j]), 0)
			}
		}
	}
	return f.withResponse(response)
}

func (f *FRD) SelectFrequencies(indices []int) (*FRD, error) {
	p, m := f.Dims()
	response, data := newFRDResponseStorage(len(indices), p, m)
	omega := make([]float64, len(indices))
	prev := -1
	pm := p * m
	for out, idx := range indices {
		if idx < 0 || idx >= len(f.Omega) {
			return nil, fmt.Errorf("FRD SelectFrequencies: index %d out of range: %w", idx, ErrDimensionMismatch)
		}
		if idx <= prev {
			return nil, fmt.Errorf("FRD SelectFrequencies: indices must be strictly increasing: %w", ErrDimensionMismatch)
		}
		prev = idx
		omega[out] = f.Omega[idx]
		copyComplexMatrixInto(data[out*pm:(out+1)*pm], f.Response[idx], p, m)
	}
	return f.withResponseAndOmega(response, omega), nil
}

func (f *FRD) SelectFrequencyRange(minOmega, maxOmega float64) (*FRD, error) {
	if minOmega > maxOmega {
		return nil, fmt.Errorf("FRD SelectFrequencyRange: min frequency exceeds max frequency: %w", ErrDimensionMismatch)
	}
	var indices []int
	for k, w := range f.Omega {
		if w >= minOmega && w <= maxOmega {
			indices = append(indices, k)
		}
	}
	return f.SelectFrequencies(indices)
}

func (f *FRD) MapResponse(mapper FRDResponseMapper) (*FRD, error) {
	if mapper == nil {
		return nil, fmt.Errorf("FRD MapResponse: mapper must not be nil: %w", ErrDimensionMismatch)
	}
	p, m := f.Dims()
	response, data := newFRDResponseStorage(len(f.Omega), p, m)
	pm := p * m
	for k, w := range f.Omega {
		mapped, err := mapper(k, w, f.EvalFr(k))
		if err != nil {
			return nil, fmt.Errorf("FRD MapResponse: frequency index %d: %w", k, err)
		}
		if len(mapped) != p {
			return nil, fmt.Errorf("FRD MapResponse: frequency index %d has %d rows, want %d: %w", k, len(mapped), p, ErrDimensionMismatch)
		}
		for i := range mapped {
			if len(mapped[i]) != m {
				return nil, fmt.Errorf("FRD MapResponse: frequency index %d row %d has %d cols, want %d: %w", k, i, len(mapped[i]), m, ErrDimensionMismatch)
			}
		}
		copyComplexMatrixInto(data[k*pm:(k+1)*pm], mapped, p, m)
	}
	return f.withResponse(response), nil
}

func (f *FRD) PeakGain() (*FRDPeakGainResult, error) {
	p, m := f.Dims()
	nw := len(f.Omega)
	if nw == 0 || p == 0 || m == 0 {
		return nil, fmt.Errorf("FRD PeakGain: no frequency response data: %w", ErrInsufficientData)
	}
	result := &FRDPeakGainResult{Gain: math.Inf(-1), Index: -1}
	if p == 1 && m == 1 {
		for k := range nw {
			gain := cmplx.Abs(f.Response[k][0][0])
			if gain > result.Gain {
				result.Gain = gain
				result.Frequency = f.Omega[k]
				result.Index = k
			}
		}
		return result, nil
	}
	ws := newComplexSVDWorkspace(p, m)
	sv := make([]float64, min(p, m))
	for k := range nw {
		ws.singularValuesFromNested(sv, f.Response[k], p, m)
		if sv[0] > result.Gain {
			result.Gain = sv[0]
			result.Frequency = f.Omega[k]
			result.Index = k
		}
	}
	return result, nil
}

func FRDConcat(first *FRD, rest ...*FRD) (*FRD, error) {
	if first == nil {
		return nil, fmt.Errorf("FRDConcat: first model must not be nil: %w", ErrDimensionMismatch)
	}
	p, m := first.Dims()
	total := len(first.Omega)
	for idx, f := range rest {
		if f == nil {
			return nil, fmt.Errorf("FRDConcat: model %d must not be nil: %w", idx+1, ErrDimensionMismatch)
		}
		fp, fm := f.Dims()
		if fp != p || fm != m {
			return nil, fmt.Errorf("FRDConcat: model %d has dimensions %dx%d, want %dx%d: %w", idx+1, fp, fm, p, m, ErrDimensionMismatch)
		}
		if f.Dt != first.Dt {
			return nil, fmt.Errorf("FRDConcat: model %d sample time %g does not match %g: %w", idx+1, f.Dt, first.Dt, ErrDomainMismatch)
		}
		total += len(f.Omega)
	}
	response, data := newFRDResponseStorage(total, p, m)
	omega := make([]float64, 0, total)
	pm := p * m
	pos := 0
	appendModel := func(f *FRD) error {
		for k, w := range f.Omega {
			if len(omega) > 0 && w <= omega[len(omega)-1] {
				return fmt.Errorf("FRDConcat: frequencies must be strictly increasing across models: %w", ErrDimensionMismatch)
			}
			omega = append(omega, w)
			copyComplexMatrixInto(data[pos*pm:(pos+1)*pm], f.Response[k], p, m)
			pos++
		}
		return nil
	}
	if err := appendModel(first); err != nil {
		return nil, err
	}
	for _, f := range rest {
		if err := appendModel(f); err != nil {
			return nil, err
		}
	}
	return first.withResponseAndOmega(response, omega), nil
}

func (f *FRD) withResponse(response [][][]complex128) *FRD {
	omega := make([]float64, len(f.Omega))
	copy(omega, f.Omega)
	return f.withResponseAndOmega(response, omega)
}

func (f *FRD) withResponseAndOmega(response [][][]complex128, omega []float64) *FRD {
	return &FRD{
		Response:   response,
		Omega:      omega,
		Dt:         f.Dt,
		InputName:  copyStringSlice(f.InputName),
		OutputName: copyStringSlice(f.OutputName),
	}
}

// EvalFr returns the p*m complex response at frequency omega[freqIdx].
func (f *FRD) EvalFr(freqIdx int) [][]complex128 {
	p, m := f.Dims()
	result := make([][]complex128, p)
	for i := range p {
		result[i] = make([]complex128, m)
		copy(result[i], f.Response[freqIdx][i])
	}
	return result
}

// FreqResponse returns the FRD data as a FreqResponseMatrix for compatibility.
func (f *FRD) FreqResponse() *FreqResponseMatrix {
	p, m := f.Dims()
	data := make([]complex128, len(f.Omega)*p*m)
	copyComplexGridInto(data, f.Response, p, m)
	return newFreqResponseMatrixOwned(data, f.Omega, p, m, f.InputName, f.OutputName)
}

// Bode computes magnitude (dB) and phase (degrees) from the FRD data.
func (f *FRD) Bode() *BodeResult {
	p, m := f.Dims()
	nw := len(f.Omega)
	omega := make([]float64, nw)
	copy(omega, f.Omega)
	pm := p * m
	magDB := make([]float64, nw*pm)
	phase := make([]float64, nw*pm)
	mag := newSampledScalarResponse(magDB, omega, p, m)
	phaseResp := newSampledScalarResponse(phase, omega, p, m)
	for k := range omega {
		for i := range p {
			for j := range m {
				h := f.Response[k][i][j]
				mag.set(k, i, j, 20*math.Log10(cmplx.Abs(h)))
				phaseResp.set(k, i, j, cmplx.Phase(h)*180/math.Pi)
			}
		}
	}
	unwrapBodePhase(phase, p, m, nw)
	return &BodeResult{
		Omega:      omega,
		magDB:      magDB,
		phase:      phase,
		p:          p,
		m:          m,
		InputName:  copyStringSlice(f.InputName),
		OutputName: copyStringSlice(f.OutputName),
	}
}

func (f *FRD) Nyquist() (*NyquistResult, error) {
	p, m := f.Dims()
	if p != 1 || m != 1 {
		return nil, fmt.Errorf("FRD Nyquist: SISO only, got %dx%d", p, m)
	}
	nw := len(f.Omega)
	contour := make([]complex128, nw)
	contourN := make([]complex128, nw)
	for k := range nw {
		contour[k] = f.Response[k][0][0]
		contourN[k] = cmplx.Conj(contour[k])
	}
	enc := 0
	for k := 1; k < nw; k++ {
		re0, im0 := real(contour[k-1])+1, imag(contour[k-1])
		re1, im1 := real(contour[k])+1, imag(contour[k])
		if (im0 >= 0) != (im1 >= 0) {
			xCross := re0 - im0*(re1-re0)/(im1-im0)
			if xCross < 0 {
				if im1 > im0 {
					enc++
				} else {
					enc--
				}
			}
		}
	}
	omega := make([]float64, nw)
	copy(omega, f.Omega)
	return &NyquistResult{Omega: omega, Contour: contour, ContourN: contourN, Encirclements: enc}, nil
}

func (f *FRD) Sigma() (*SigmaResult, error) {
	p, m := f.Dims()
	nw := len(f.Omega)
	nsv := min(p, m)
	if nsv == 0 {
		omega := make([]float64, nw)
		copy(omega, f.Omega)
		return &SigmaResult{Omega: omega, nSV: 0}, nil
	}
	sv := make([]float64, nw*nsv)
	if p == 1 && m == 1 {
		for k := range nw {
			sv[k] = cmplx.Abs(f.Response[k][0][0])
		}
		omega := make([]float64, nw)
		copy(omega, f.Omega)
		return &SigmaResult{Omega: omega, sv: sv, nSV: nsv}, nil
	}
	response := newSampledComplexGridResponse(f.Response, f.Omega, p, m)
	var ws *complexSVDWorkspace
	ws = newComplexSVDWorkspace(p, m)
	for k := range nw {
		response.singularValues(sv[k*nsv:(k+1)*nsv], ws, k)
	}
	omega := make([]float64, nw)
	copy(omega, f.Omega)
	return &SigmaResult{Omega: omega, sv: sv, nSV: nsv}, nil
}

// FRDMargin computes gain and phase margins of a SISO FRD loop by
// interpolating between frequency points (linear in log omega), like MATLAB
// margin/allmargin on frd. Phase crossovers are where the phase passes
// -180 deg mod 360 (so -540, -900, ... count); phase margins are wrapped to
// (-180,180]. Selection among multiple crossings matches Margin.
func FRDMargin(f *FRD) (*MarginResult, error) {
	p, m := f.Dims()
	if p != 1 || m != 1 {
		return nil, fmt.Errorf("FRDMargin: SISO only, got %dx%d", p, m)
	}
	nw := len(f.Omega)
	if nw < 2 {
		return nil, fmt.Errorf("FRDMargin: need at least 2 frequency points")
	}

	bode := f.Bode()
	magDB := bode.magDB
	phase := bode.phase
	omega := bode.Omega

	var allGM []float64
	var allGMFreq []float64
	var allPM []float64
	var allPMFreq []float64

	b := frdWrapDeg(phase[0] + 180)
	for k := 1; k < nw; k++ {
		a := b
		b = frdWrapDeg(phase[k] + 180)
		if !frdCrosses(a, b, k) || math.Abs(a-b) >= 180 {
			continue
		}
		frac := frdCrossFrac(a, b)
		allGM = append(allGM, -(magDB[k-1] + frac*(magDB[k]-magDB[k-1])))
		allGMFreq = append(allGMFreq, frdInterpOmega(omega[k-1], omega[k], frac))
	}

	for k := 1; k < nw; k++ {
		m0, m1 := magDB[k-1], magDB[k]
		if math.IsInf(m0, 0) || math.IsInf(m1, 0) || !frdCrosses(m0, m1, k) {
			continue
		}
		frac := frdCrossFrac(m0, m1)
		allPM = append(allPM, frdWrapDeg(180+phase[k-1]+frac*(phase[k]-phase[k-1])))
		allPMFreq = append(allPMFreq, frdInterpOmega(omega[k-1], omega[k], frac))
	}

	result := &MarginResult{
		GainMargin:  math.Inf(1),
		PhaseMargin: math.Inf(1),
		WgFreq:      math.NaN(),
		WpFreq:      math.NaN(),
	}

	for i, gm := range allGM {
		if gm > 0 && gm < result.GainMargin {
			result.GainMargin = gm
			result.WpFreq = allGMFreq[i]
		}
	}
	if math.IsInf(result.GainMargin, 1) {
		for i, gm := range allGM {
			if gm > result.GainMargin || math.IsInf(result.GainMargin, 1) {
				result.GainMargin = gm
				result.WpFreq = allGMFreq[i]
			}
		}
	}

	for i, pm := range allPM {
		if pm > 0 && pm < result.PhaseMargin {
			result.PhaseMargin = pm
			result.WgFreq = allPMFreq[i]
		}
	}
	if math.IsInf(result.PhaseMargin, 1) {
		for i, pm := range allPM {
			if pm > result.PhaseMargin || math.IsInf(result.PhaseMargin, 1) {
				result.PhaseMargin = pm
				result.WgFreq = allPMFreq[i]
			}
		}
	}

	return result, nil
}

// frdWrapDeg wraps degrees to (-180,180].
func frdWrapDeg(x float64) float64 {
	return x - 360*math.Ceil((x-180)/360)
}

// frdCrosses reports a sign change of a->b on segment k; an exact zero
// counts once, at the segment where it is the right endpoint (or k==1).
func frdCrosses(a, b float64, k int) bool {
	if math.IsNaN(a) || math.IsNaN(b) {
		return false
	}
	return a*b < 0 || (b == 0 && a != 0) || (a == 0 && k == 1)
}

func frdCrossFrac(a, b float64) float64 {
	if a == b {
		return 0
	}
	return a / (a - b)
}

func frdInterpOmega(w0, w1, frac float64) float64 {
	if w0 <= 0 {
		return w0 + frac*(w1-w0)
	}
	return math.Exp(math.Log(w0) + frac*(math.Log(w1)-math.Log(w0)))
}

func frdGridsMatch(f1, f2 *FRD) error {
	if f1.Dt != f2.Dt {
		return fmt.Errorf("frd: sample times %g != %g: %w", f1.Dt, f2.Dt, ErrDomainMismatch)
	}
	if len(f1.Omega) != len(f2.Omega) {
		return fmt.Errorf("frd: frequency grid lengths %d != %d", len(f1.Omega), len(f2.Omega))
	}
	for i := range f1.Omega {
		if math.Abs(f1.Omega[i]-f2.Omega[i]) > 1e-12*math.Max(1, math.Abs(f1.Omega[i])) {
			return fmt.Errorf("frd: frequency mismatch at index %d: %g != %g", i, f1.Omega[i], f2.Omega[i])
		}
	}
	return nil
}

type frdInterconnection struct {
	left, right *FRD
	omega       []float64
	dt          float64
	nw          int
}

func newFRDInterconnection(left, right *FRD) (frdInterconnection, error) {
	if err := frdGridsMatch(left, right); err != nil {
		return frdInterconnection{}, err
	}
	return frdInterconnection{
		left:  left,
		right: right,
		omega: left.Omega,
		dt:    left.Dt,
		nw:    len(left.Omega),
	}, nil
}

func (ic frdInterconnection) newResult(p, m int, inputName, outputName []string) (*FRD, []complex128) {
	resp, data := newFRDResponseStorage(ic.nw, p, m)
	return &FRD{
		Response:   resp,
		Omega:      append([]float64(nil), ic.omega...),
		Dt:         ic.dt,
		InputName:  inputName,
		OutputName: outputName,
	}, data
}

func FRDSeries(f1, f2 *FRD) (*FRD, error) {
	ic, err := newFRDInterconnection(f1, f2)
	if err != nil {
		return nil, err
	}
	p1, m1 := f1.Dims()
	p2, m2 := f2.Dims()
	if p1 != m2 {
		return nil, fmt.Errorf("frd series: f1 outputs %d != f2 inputs %d", p1, m2)
	}

	md := frdSeriesMetadata(f1, f2)
	result, data := ic.newResult(p2, m1, md.input, md.output)
	for w := 0; w < ic.nw; w++ {
		cMulNestedInto(data[w*p2*m1:(w+1)*p2*m1], f2.Response[w], f1.Response[w], p2, p1, m1)
	}
	return result, nil
}

func FRDParallel(f1, f2 *FRD) (*FRD, error) {
	ic, err := newFRDInterconnection(f1, f2)
	if err != nil {
		return nil, err
	}
	p1, m1 := f1.Dims()
	p2, m2 := f2.Dims()
	if p1 != p2 || m1 != m2 {
		return nil, fmt.Errorf("frd parallel: dims (%d,%d) != (%d,%d)", p1, m1, p2, m2)
	}

	md := frdParallelMetadata(f1)
	result, data := ic.newResult(p1, m1, md.input, md.output)
	for w := 0; w < ic.nw; w++ {
		cAddNestedInto(data[w*p1*m1:(w+1)*p1*m1], f1.Response[w], f2.Response[w], p1, m1)
	}
	return result, nil
}

func FRDFeedback(plant, controller *FRD, sign float64) (*FRD, error) {
	ic, err := newFRDInterconnection(plant, controller)
	if err != nil {
		return nil, err
	}
	pp, pm := plant.Dims()
	cp, cm := controller.Dims()
	if pp != cm {
		return nil, fmt.Errorf("frd feedback: plant outputs %d != controller inputs %d", pp, cm)
	}
	if pm != cp {
		return nil, fmt.Errorf("frd feedback: plant inputs %d != controller outputs %d", pm, cp)
	}

	md := frdFeedbackMetadata(plant)
	result, data := ic.newResult(pp, pm, md.input, md.output)
	ws := newFRDFeedbackWorkspace(pp, pm)

	for w := 0; w < ic.nw; w++ {
		copyComplexMatrixInto(ws.g, plant.Response[w], pp, pm)
		copyComplexMatrixInto(ws.k, controller.Response[w], cp, pp)
		cMulInto(ws.kg, ws.k, ws.g, cp, pp, pm)

		for i := range ws.n {
			for j := range ws.n {
				ws.ipkg[j*ws.n+i] = complex(-sign, 0) * ws.kg[i*ws.n+j]
			}
		}
		for i := 0; i < ws.n; i++ {
			ws.ipkg[i*ws.n+i] += 1
		}
		for i := range pp {
			for j := range ws.n {
				ws.rhs[j*pp+i] = ws.g[i*ws.n+j]
			}
		}
		if err := cSolveInPlace(ws.ipkg, ws.rhs, ws.n, pp); err != nil {
			return nil, fmt.Errorf("frd feedback: singular at freq index %d", w)
		}
		dst := data[w*pp*pm : (w+1)*pp*pm]
		for i := range pp {
			for j := range ws.n {
				dst[i*ws.n+j] = ws.rhs[j*pp+i]
			}
		}
	}

	return result, nil
}

type frdFeedbackWorkspace struct {
	g    []complex128
	k    []complex128
	kg   []complex128
	ipkg []complex128
	rhs  []complex128
	n    int
}

func newFRDFeedbackWorkspace(pp, pm int) *frdFeedbackWorkspace {
	n := pm
	return &frdFeedbackWorkspace{
		g:    make([]complex128, pp*pm),
		k:    make([]complex128, pm*pp),
		kg:   make([]complex128, n*n),
		ipkg: make([]complex128, n*n),
		rhs:  make([]complex128, n*pp),
		n:    n,
	}
}
