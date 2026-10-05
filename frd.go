package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
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

// NewFRD creates an FRD model from response data and frequency vector, as
// MATLAB frd(response,omega,dt). response[k] is the p×m complex response
// matrix at frequency omega[k]. The data must satisfy Validate.
func NewFRD(response [][][]complex128, omega []float64, dt float64) (*FRD, error) {
	if len(response) != len(omega) {
		return nil, fmt.Errorf("NewFRD: len(response)=%d != len(omega)=%d: %w",
			len(response), len(omega), ErrDimensionMismatch)
	}
	f := &FRD{Response: response, Omega: omega, Dt: dt}
	if err := f.validate(); err != nil {
		return nil, fmt.Errorf("NewFRD: %w", err)
	}
	p, m := f.Dims()
	resp, data := newFRDResponseStorage(len(response), p, m)
	copyComplexGridInto(data, response, p, m)
	return &FRD{
		Response: resp,
		Omega:    copyFloatSlice(omega),
		Dt:       dt,
	}, nil
}

// Validate reports whether f is a well-formed FRD: a valid sample time, at
// least one frequency, finite non-negative strictly increasing frequencies
// (at most π/Dt for discrete models), one p×m response matrix per frequency
// with p, m > 0, and no NaN response entries. Infinite entries are allowed;
// they mark a pole on the frequency grid. Call Validate after editing the
// fields directly; FRD methods that return an error validate first.
func (f *FRD) Validate() error {
	if err := f.validate(); err != nil {
		return fmt.Errorf("FRD.Validate: %w", err)
	}
	return nil
}

func (f *FRD) validate() error {
	if f == nil {
		return fmt.Errorf("FRD is nil: %w", ErrInvalidArgument)
	}
	if err := newTimeDomain(f.Dt).validateSampleTime(); err != nil {
		return err
	}
	if len(f.Omega) == 0 {
		return fmt.Errorf("no frequencies: %w", ErrInvalidArgument)
	}
	if len(f.Response) != len(f.Omega) {
		return fmt.Errorf("%d responses for %d frequencies: %w", len(f.Response), len(f.Omega), ErrDimensionMismatch)
	}
	for i, w := range f.Omega {
		if !(w >= 0) || math.IsInf(w, 1) {
			return fmt.Errorf("omega[%d]=%v must be finite and non-negative: %w", i, w, ErrInvalidArgument)
		}
		if i > 0 && w <= f.Omega[i-1] {
			return fmt.Errorf("omega must be strictly increasing, omega[%d]=%v <= omega[%d]=%v: %w", i, w, i-1, f.Omega[i-1], ErrInvalidArgument)
		}
	}
	if f.Dt > 0 {
		nyquist := math.Pi / f.Dt
		if w := f.Omega[len(f.Omega)-1]; w > nyquist*(1+1e-10) {
			return fmt.Errorf("omega %v exceeds Nyquist frequency %v: %w", w, nyquist, ErrInvalidArgument)
		}
	}
	p := len(f.Response[0])
	if p == 0 {
		return fmt.Errorf("response matrix has zero rows: %w", ErrDimensionMismatch)
	}
	m := len(f.Response[0][0])
	if m == 0 {
		return fmt.Errorf("response matrix has zero columns: %w", ErrDimensionMismatch)
	}
	for k := range f.Response {
		if len(f.Response[k]) != p {
			return fmt.Errorf("response[%d] has %d rows, want %d: %w", k, len(f.Response[k]), p, ErrDimensionMismatch)
		}
		for i, row := range f.Response[k] {
			if len(row) != m {
				return fmt.Errorf("response[%d][%d] has %d cols, want %d: %w", k, i, len(row), m, ErrDimensionMismatch)
			}
			for j, h := range row {
				if cmplx.IsNaN(h) {
					return fmt.Errorf("response[%d][%d][%d] is NaN: %w", k, i, j, ErrInvalidArgument)
				}
			}
		}
	}
	return nil
}

// FRD computes the frequency response data model at the given frequencies.
// An empty omega returns ErrInvalidArgument.
func (sys *System) FRD(omega []float64) (*FRD, error) {
	if len(omega) == 0 {
		return nil, fmt.Errorf("FRD: empty frequency vector: %w", ErrInvalidArgument)
	}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return nil, fmt.Errorf("FRD: %w", err)
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
		return nil, fmt.Errorf("frequency response is nil: %w", ErrInvalidArgument)
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

// Dims returns the number of outputs p and inputs m; 0, 0 for an FRD without
// response data.
func (f *FRD) Dims() (p, m int) {
	if len(f.Response) == 0 || len(f.Response[0]) == 0 {
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

// SelectFrequencies returns the FRD restricted to the strictly increasing
// frequency indices. Out-of-range or non-increasing indices, or no indices,
// return ErrInvalidArgument.
func (f *FRD) SelectFrequencies(indices []int) (*FRD, error) {
	if err := f.validate(); err != nil {
		return nil, fmt.Errorf("FRD.SelectFrequencies: %w", err)
	}
	if len(indices) == 0 {
		return nil, fmt.Errorf("FRD.SelectFrequencies: no frequencies selected: %w", ErrInvalidArgument)
	}
	p, m := f.Dims()
	response, data := newFRDResponseStorage(len(indices), p, m)
	omega := make([]float64, len(indices))
	prev := -1
	pm := p * m
	for out, idx := range indices {
		if idx < 0 || idx >= len(f.Omega) {
			return nil, fmt.Errorf("FRD.SelectFrequencies: index %d out of range [0,%d): %w", idx, len(f.Omega), ErrInvalidArgument)
		}
		if idx <= prev {
			return nil, fmt.Errorf("FRD.SelectFrequencies: indices must be strictly increasing: %w", ErrInvalidArgument)
		}
		prev = idx
		omega[out] = f.Omega[idx]
		copyComplexMatrixInto(data[out*pm:(out+1)*pm], f.Response[idx], p, m)
	}
	return f.withResponseAndOmega(response, omega), nil
}

// SelectFrequencyRange returns the FRD restricted to frequencies in
// [minOmega, maxOmega]. minOmega > maxOmega, or a range containing no
// frequency, returns ErrInvalidArgument.
func (f *FRD) SelectFrequencyRange(minOmega, maxOmega float64) (*FRD, error) {
	if err := f.validate(); err != nil {
		return nil, fmt.Errorf("FRD.SelectFrequencyRange: %w", err)
	}
	if !(minOmega <= maxOmega) {
		return nil, fmt.Errorf("FRD.SelectFrequencyRange: min %g exceeds max %g: %w", minOmega, maxOmega, ErrInvalidArgument)
	}
	var indices []int
	for k, w := range f.Omega {
		if w >= minOmega && w <= maxOmega {
			indices = append(indices, k)
		}
	}
	if len(indices) == 0 {
		return nil, fmt.Errorf("FRD.SelectFrequencyRange: no frequency in [%g, %g]: %w", minOmega, maxOmega, ErrInvalidArgument)
	}
	return f.SelectFrequencies(indices)
}

// MapResponse returns the FRD whose response at each frequency is mapper's
// result, which must keep the p×m shape.
func (f *FRD) MapResponse(mapper FRDResponseMapper) (*FRD, error) {
	if err := f.validate(); err != nil {
		return nil, fmt.Errorf("FRD.MapResponse: %w", err)
	}
	if mapper == nil {
		return nil, fmt.Errorf("FRD.MapResponse: mapper is nil: %w", ErrInvalidArgument)
	}
	p, m := f.Dims()
	response, data := newFRDResponseStorage(len(f.Omega), p, m)
	pm := p * m
	for k, w := range f.Omega {
		mapped, err := mapper(k, w, f.EvalFr(k))
		if err != nil {
			return nil, fmt.Errorf("FRD.MapResponse: frequency index %d: %w", k, err)
		}
		if len(mapped) != p {
			return nil, fmt.Errorf("FRD.MapResponse: frequency index %d has %d rows, want %d: %w", k, len(mapped), p, ErrDimensionMismatch)
		}
		for i := range mapped {
			if len(mapped[i]) != m {
				return nil, fmt.Errorf("FRD.MapResponse: frequency index %d row %d has %d cols, want %d: %w", k, i, len(mapped[i]), m, ErrDimensionMismatch)
			}
		}
		copyComplexMatrixInto(data[k*pm:(k+1)*pm], mapped, p, m)
	}
	return f.withResponse(response), nil
}

// PeakGain returns the largest gain (largest singular value for MIMO data)
// over the frequency grid and where it occurs, as MATLAB getPeakGain on frd
// data.
func (f *FRD) PeakGain() (*FRDPeakGainResult, error) {
	if err := f.validate(); err != nil {
		return nil, fmt.Errorf("FRD.PeakGain: %w", err)
	}
	p, m := f.Dims()
	nw := len(f.Omega)
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
	sv := make([]float64, 1)
	for k := range nw {
		if err := ws.singularValuesFromNested(sv, f.Response[k], p, m); err != nil {
			return nil, fmt.Errorf("FRD.PeakGain: omega=%g: %w", f.Omega[k], err)
		}
		if sv[0] > result.Gain {
			result.Gain = sv[0]
			result.Frequency = f.Omega[k]
			result.Index = k
		}
	}
	if result.Index < 0 {
		return nil, fmt.Errorf("FRD.PeakGain: no finite singular value: %w", ErrSingularTransform)
	}
	return result, nil
}

// FRDConcat joins FRD models of equal size and sample time along the
// frequency axis; the combined frequencies must be strictly increasing.
func FRDConcat(first *FRD, rest ...*FRD) (*FRD, error) {
	if err := first.validate(); err != nil {
		return nil, fmt.Errorf("FRDConcat: model 0: %w", err)
	}
	p, m := first.Dims()
	total := len(first.Omega)
	for idx, f := range rest {
		if err := f.validate(); err != nil {
			return nil, fmt.Errorf("FRDConcat: model %d: %w", idx+1, err)
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
				return fmt.Errorf("FRDConcat: frequencies must be strictly increasing across models: %w", ErrInvalidArgument)
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

// Nyquist computes the Nyquist response of SISO frequency response data.
//
// Contour is the data, ContourN the negative-frequency branch
// conj(Contour[nw-1-k]) by conjugate symmetry, as MATLAB nyquist plots it;
// the data must come from a real-coefficient model.
//
// Encirclements extends MATLAB: it is the clockwise winding of the loop
// response around -1 along ContourN, Contour and their closures at both ends
// of the grid. The count depends on the grid, which must resolve the phase
// around -1 and reach the asymptotic regions at both ends. At the high end the
// closure is a straight chord, assuming the data has rolled off. At the low
// end, n = round(-d ln|H| / d ln ω) from the first two distinct frequencies
// estimates the number of poles at s=0 (z=1); for n >= 1 the closure is the
// clockwise arc of about n*pi at radius |H(ω0)| that indenting the contour
// around those poles maps to, so they count as stable as in
// (*System).Nyquist. Other poles on the stability boundary cannot be
// indented from data and must lie outside the grid.
//
// Open-loop poles outside the stability boundary cannot be known from
// frequency data, so the result carries no pole counts: Encirclements is the
// closed-loop unstable pole count under unit negative feedback only for a
// stable open loop, and a caller who knows the open-loop count P adds it.
func (f *FRD) Nyquist() (*FRDNyquistResult, error) {
	if err := f.validate(); err != nil {
		return nil, fmt.Errorf("FRD.Nyquist: %w", err)
	}
	p, m := f.Dims()
	if p != 1 || m != 1 {
		return nil, fmt.Errorf("FRD.Nyquist: model is %d×%d: %w", p, m, ErrNotSISO)
	}
	nw := len(f.Omega)
	contour := make([]complex128, nw)
	contourN := make([]complex128, nw)
	for k := range nw {
		h := f.Response[k][0][0]
		if cmplx.IsNaN(h) || cmplx.IsInf(h) {
			return nil, fmt.Errorf("FRD.Nyquist: non-finite response at omega[%d]=%v: %w", k, f.Omega[k], ErrInvalidArgument)
		}
		contour[k] = h
		contourN[nw-1-k] = cmplx.Conj(h)
	}
	full := make([]complex128, 0, 2*nw)
	full = append(full, contourN...)
	full = append(full, frdLowFreqClosure(f.Omega, contour)...)
	full = append(full, contour...)
	enc := -windingNumber(full, -1)
	return &FRDNyquistResult{Omega: copyFloatSlice(f.Omega), Contour: contour, ContourN: contourN, Encirclements: enc}, nil
}

// FRDNyquistResult holds the Nyquist response of SISO frequency response
// data; see (*FRD).Nyquist. Unlike NyquistResult it has no open-loop or
// closed-loop pole counts, which frequency data cannot determine.
type FRDNyquistResult struct {
	Omega         []float64
	Contour       []complex128
	ContourN      []complex128
	Encirclements int
}

// frdLowFreqClosure returns the interior points of the arc from conj(h0) to
// h0 that the indentation around n poles at s=0 maps to: radius |h0| and
// clockwise rotation nearest to n*pi congruent to 2*arg(h0).
func frdLowFreqClosure(omega []float64, h []complex128) []complex128 {
	j := 1
	for j < len(omega) && omega[j] == omega[0] {
		j++
	}
	if omega[0] <= 0 || j == len(omega) || h[0] == 0 || h[j] == 0 {
		return nil
	}
	slope := math.Log(cmplx.Abs(h[j])/cmplx.Abs(h[0])) / math.Log(omega[j]/omega[0])
	n := math.Round(-slope)
	if n < 1 {
		return nil
	}
	a0 := cmplx.Phase(h[0])
	rot := 2*a0 + 2*math.Pi*math.Round((-n*math.Pi-2*a0)/(2*math.Pi))
	r := cmplx.Abs(h[0])
	k := max(int(math.Ceil(math.Abs(rot)/(math.Pi/50))), 2)
	arc := make([]complex128, 0, k-1)
	for i := 1; i < k; i++ {
		arc = append(arc, cmplx.Rect(r, -a0+rot*float64(i)/float64(k)))
	}
	return arc
}

// Sigma returns the singular values of the response at each frequency, as
// MATLAB sigma on frd data.
func (f *FRD) Sigma() (*SigmaResult, error) {
	if err := f.validate(); err != nil {
		return nil, fmt.Errorf("FRD.Sigma: %w", err)
	}
	p, m := f.Dims()
	nw := len(f.Omega)
	nsv := min(p, m)
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
	ws := newComplexSVDWorkspace(p, m)
	for k := range nw {
		if err := response.singularValues(sv[k*nsv:(k+1)*nsv], ws, k); err != nil {
			return nil, fmt.Errorf("FRD.Sigma: %w", err)
		}
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
	if err := f.validate(); err != nil {
		return nil, fmt.Errorf("FRDMargin: %w", err)
	}
	p, m := f.Dims()
	if p != 1 || m != 1 {
		return nil, fmt.Errorf("FRDMargin: model is %d×%d: %w", p, m, ErrNotSISO)
	}
	nw := len(f.Omega)
	if nw < 2 {
		return nil, fmt.Errorf("FRDMargin: %d frequency points, need at least 2: %w", nw, ErrInsufficientData)
	}

	bode := f.Bode()
	magDB, phase, omega := bode.magDB, bode.phase, bode.Omega
	lerp := func(v []float64, c crossing) float64 { return v[c.idx] + c.frac*(v[c.idx+1]-v[c.idx]) }

	all := &AllMarginResult{}
	for _, c := range phaseCrossings(omega, phase, -180) {
		all.GainMargins = append(all.GainMargins, -lerp(magDB, c))
		all.PhaseCrossFreqs = append(all.PhaseCrossFreqs, c.w)
	}
	for _, c := range findCrossings(omega, magDB, 0) {
		all.PhaseMargins = append(all.PhaseMargins, wrapDegrees(180+lerp(phase, c)))
		all.GainCrossFreqs = append(all.GainCrossFreqs, c.w)
	}
	return pickMargins(all), nil
}

func frdGridsMatch(f1, f2 *FRD) error {
	if err := f1.validate(); err != nil {
		return fmt.Errorf("first model: %w", err)
	}
	if err := f2.validate(); err != nil {
		return fmt.Errorf("second model: %w", err)
	}
	if f1.Dt != f2.Dt {
		return fmt.Errorf("sample times %g != %g: %w", f1.Dt, f2.Dt, ErrDomainMismatch)
	}
	if len(f1.Omega) != len(f2.Omega) {
		return fmt.Errorf("frequency grid lengths %d != %d: %w", len(f1.Omega), len(f2.Omega), ErrDimensionMismatch)
	}
	for i := range f1.Omega {
		if math.Abs(f1.Omega[i]-f2.Omega[i]) > 1e-12*math.Max(1, math.Abs(f1.Omega[i])) {
			return fmt.Errorf("frequency mismatch at index %d: %g != %g: %w", i, f1.Omega[i], f2.Omega[i], ErrInvalidArgument)
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

// FRDSeries returns the series connection f2·f1 (f1 feeds f2) on a common
// frequency grid.
func FRDSeries(f1, f2 *FRD) (*FRD, error) {
	ic, err := newFRDInterconnection(f1, f2)
	if err != nil {
		return nil, fmt.Errorf("FRDSeries: %w", err)
	}
	p1, m1 := f1.Dims()
	p2, m2 := f2.Dims()
	if p1 != m2 {
		return nil, fmt.Errorf("FRDSeries: f1 outputs %d != f2 inputs %d: %w", p1, m2, ErrDimensionMismatch)
	}

	md := frdSeriesMetadata(f1, f2)
	result, data := ic.newResult(p2, m1, md.input, md.output)
	for w := 0; w < ic.nw; w++ {
		cMulNestedInto(data[w*p2*m1:(w+1)*p2*m1], f2.Response[w], f1.Response[w], p2, p1, m1)
	}
	return result, nil
}

// FRDParallel returns the parallel connection f1+f2 on a common frequency
// grid.
func FRDParallel(f1, f2 *FRD) (*FRD, error) {
	ic, err := newFRDInterconnection(f1, f2)
	if err != nil {
		return nil, fmt.Errorf("FRDParallel: %w", err)
	}
	p1, m1 := f1.Dims()
	p2, m2 := f2.Dims()
	if p1 != p2 || m1 != m2 {
		return nil, fmt.Errorf("FRDParallel: dims %d×%d != %d×%d: %w", p1, m1, p2, m2, ErrDimensionMismatch)
	}

	md := frdParallelMetadata(f1)
	result, data := ic.newResult(p1, m1, md.input, md.output)
	for w := 0; w < ic.nw; w++ {
		cAddNestedInto(data[w*p1*m1:(w+1)*p1*m1], f1.Response[w], f2.Response[w], p1, m1)
	}
	return result, nil
}

// FRDFeedback returns the closed loop (I - sign·G·K)⁻¹·G of plant G with
// controller K in the feedback path, on a common frequency grid. sign = -1
// is negative feedback, as MATLAB feedback(G,K); sign must be ±1. A
// frequency where I - sign·K·G is singular returns ErrAlgebraicLoop.
func FRDFeedback(plant, controller *FRD, sign float64) (*FRD, error) {
	if sign != 1 && sign != -1 {
		return nil, fmt.Errorf("FRDFeedback: sign %g must be ±1: %w", sign, ErrInvalidArgument)
	}
	ic, err := newFRDInterconnection(plant, controller)
	if err != nil {
		return nil, fmt.Errorf("FRDFeedback: %w", err)
	}
	pp, pm := plant.Dims()
	cp, cm := controller.Dims()
	if pp != cm {
		return nil, fmt.Errorf("FRDFeedback: plant outputs %d != controller inputs %d: %w", pp, cm, ErrDimensionMismatch)
	}
	if pm != cp {
		return nil, fmt.Errorf("FRDFeedback: plant inputs %d != controller outputs %d: %w", pm, cp, ErrDimensionMismatch)
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
			return nil, fmt.Errorf("FRDFeedback: I-K·G singular at omega[%d]=%g: %v: %w", w, ic.omega[w], err, ErrAlgebraicLoop)
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
