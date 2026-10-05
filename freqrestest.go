package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/dsp/fourier"
	"plantcontrol.org/v1/gonum/dsp/window"
	"plantcontrol.org/v1/gonum/mat"
)

// FreqRespEstOpts configures FreqRespEst; the zero value selects MATLAB
// tfestimate's defaults where this API has them.
type FreqRespEstOpts struct {
	// NFFT is the segment and FFT length, at most the number of samples.
	// 0 selects min(256, N).
	NFFT int
	// Window fills the window: it receives a slice of NFFT ones and returns
	// the window, which may be the same slice modified in place. nil selects
	// Hann.
	Window func([]float64) []float64
	// NOverlap is the number of samples consecutive segments share,
	// 0 <= NOverlap < NFFT. nil selects NFFT/2.
	NOverlap *int
	// Method selects the estimator; "" selects FreqRespEstH1.
	Method FreqRespEstMethod
}

// FreqRespEstMethod selects the FreqRespEst estimator.
type FreqRespEstMethod string

// FreqRespEst estimators: H1 = Syu/Suu (MATLAB tfestimate "Estimator","H1",
// any number of channels), H2 = Syy/Suy (SISO only) and FFT, the ratio of
// one windowed FFT of the first NFFT samples, which has no coherence.
const (
	FreqRespEstH1  FreqRespEstMethod = "h1"
	FreqRespEstH2  FreqRespEstMethod = "h2"
	FreqRespEstFFT FreqRespEstMethod = "fft"
)

// FreqRespEstResult is a frequency-response estimate. H and Omega hold only
// the frequencies 2πk/(NFFT·Dt), k = 0..NFFT/2, at which the estimator is
// defined: a bin where the input carries no power (the input cross-spectrum
// is singular for H1, or the input FFT vanishes for FFT) is omitted.
type FreqRespEstResult struct {
	H     *FreqResponseMatrix
	Omega []float64
	Dt    float64

	coherence []float64
}

type freqRespEstPlan struct {
	nfft   int
	hop    int
	nSeg   int
	nFreq  int
	method FreqRespEstMethod
	win    []float64
}

// Coherence returns a copy of the magnitude-squared coherence
// |Syu|²/(Suu·Syy) of every output/input pair at each frequency of Omega,
// row-major per frequency, as MATLAB mscohere; it is 0 where the output has
// no power. ok is false for the FFT estimator, which has none.
func (r *FreqRespEstResult) Coherence() (coherence []float64, ok bool) {
	if r.coherence == nil {
		return nil, false
	}
	return copyFloatSlice(r.coherence), true
}

// CoherenceAt returns the coherence of output/input at Omega[freq]; ok is
// false for the FFT estimator.
func (r *FreqRespEstResult) CoherenceAt(freq, output, input int) (coherence float64, ok bool) {
	if r.coherence == nil {
		return 0, false
	}
	return newSampledScalarResponse(r.coherence, r.Omega, r.H.P, r.H.M).at(freq, output, input), true
}

// FRD returns the estimate as a frequency response data model.
func (r *FreqRespEstResult) FRD() (*FRD, error) {
	if r == nil || r.H == nil {
		return nil, fmt.Errorf("FreqRespEstResult.FRD: no estimate: %w", ErrInvalidArgument)
	}
	f, err := newFRDFromFreqResponse(r.H, r.Dt)
	if err != nil {
		return nil, fmt.Errorf("FreqRespEstResult.FRD: %w", err)
	}
	return f, nil
}

// FreqRespEst estimates the frequency response from input (m×N) to output
// (p×N) sampled at dt by Welch-averaged cross spectra, as MATLAB
// tfestimate(x,y,window,noverlap,nfft)
// (https://www.mathworks.com/help/signal/ref/tfestimate.html). Invalid
// options return ErrInvalidArgument; H2 on MIMO data returns
// ErrOptionUnsupported. H1 with m > 1 inputs needs at least m segments, and
// an estimate needs at least one identifiable frequency; otherwise it returns
// ErrInsufficientData.
func FreqRespEst(input, output *mat.Dense, dt float64, opts *FreqRespEstOpts) (*FreqRespEstResult, error) {
	m, p, N, err := validateSampledIO("FreqRespEst", input, output, dt)
	if err != nil {
		return nil, err
	}

	plan, err := newFreqRespEstPlan(opts, N, m, p)
	if err != nil {
		return nil, fmt.Errorf("FreqRespEst: %w", err)
	}

	var est freqRespEstimate
	switch {
	case plan.method == FreqRespEstFFT:
		est = freqRespEstFFT(input, output, m, p, plan)
	case m == 1 && p == 1:
		inRaw, outRaw := input.RawMatrix(), output.RawMatrix()
		est = welchSISO(inRaw.Data[:N], outRaw.Data[:N], plan)
	default:
		est = welchMIMO(input, output, m, p, plan)
	}
	res, err := est.result(plan, dt, p, m)
	if err != nil {
		return nil, fmt.Errorf("FreqRespEst: %w", err)
	}
	return res, nil
}

// freqRespEstimate holds the full-grid estimate, its coherence (nil for the
// FFT estimator) and which bins are defined.
type freqRespEstimate struct {
	h     []complex128
	cohRe []float64
	ok    []bool
}

func (e freqRespEstimate) result(plan freqRespEstPlan, dt float64, p, m int) (*FreqRespEstResult, error) {
	pm := p * m
	nw := 0
	for _, ok := range e.ok {
		if ok {
			nw++
		}
	}
	if nw == 0 {
		return nil, fmt.Errorf("input has no power at any frequency: %w", ErrInsufficientData)
	}
	omega := make([]float64, 0, nw)
	h := e.h
	coh := e.cohRe
	for f := range plan.nFreq {
		if !e.ok[f] {
			continue
		}
		k := len(omega)
		omega = append(omega, 2*math.Pi*float64(f)/(float64(plan.nfft)*dt))
		copy(h[k*pm:(k+1)*pm], e.h[f*pm:(f+1)*pm])
		if coh != nil {
			copy(coh[k*pm:(k+1)*pm], e.cohRe[f*pm:(f+1)*pm])
		}
	}
	h = h[:nw*pm : nw*pm]
	if coh != nil {
		coh = coh[:nw*pm : nw*pm]
	}
	return &FreqRespEstResult{
		H:         newFreqResponseMatrixOwned(h, omega, p, m, nil, nil),
		Omega:     omega,
		Dt:        dt,
		coherence: coh,
	}, nil
}

func newFreqRespEstPlan(opts *FreqRespEstOpts, N, m, p int) (freqRespEstPlan, error) {
	var o FreqRespEstOpts
	if opts != nil {
		o = *opts
	}
	plan := freqRespEstPlan{nfft: min(256, N), method: FreqRespEstH1}
	if o.Method != "" {
		plan.method = o.Method
	}
	switch plan.method {
	case FreqRespEstH1, FreqRespEstFFT:
	case FreqRespEstH2:
		if m != 1 || p != 1 {
			return freqRespEstPlan{}, fmt.Errorf("h2 estimator needs SISO data, got %d inputs and %d outputs: %w", m, p, ErrOptionUnsupported)
		}
	default:
		return freqRespEstPlan{}, fmt.Errorf("unknown method %q: %w", o.Method, ErrInvalidArgument)
	}
	if o.NFFT != 0 {
		if o.NFFT < 1 || o.NFFT > N {
			return freqRespEstPlan{}, fmt.Errorf("NFFT %d must be in [1, %d]: %w", o.NFFT, N, ErrInvalidArgument)
		}
		plan.nfft = o.NFFT
	}
	plan.nFreq = plan.nfft/2 + 1

	winFunc := o.Window
	if winFunc == nil {
		winFunc = window.Hann
	}
	ones := make([]float64, plan.nfft)
	for i := range ones {
		ones[i] = 1
	}
	plan.win = winFunc(ones)
	if len(plan.win) != plan.nfft {
		return freqRespEstPlan{}, fmt.Errorf("window has length %d, want NFFT=%d: %w", len(plan.win), plan.nfft, ErrDimensionMismatch)
	}
	if err := requireFinite("FreqRespEst", "window", plan.win...); err != nil {
		return freqRespEstPlan{}, err
	}

	noverlap := plan.nfft / 2
	if o.NOverlap != nil {
		noverlap = *o.NOverlap
		if noverlap < 0 || noverlap >= plan.nfft {
			return freqRespEstPlan{}, fmt.Errorf("NOverlap %d must be in [0, NFFT=%d): %w", noverlap, plan.nfft, ErrInvalidArgument)
		}
	}
	plan.hop = plan.nfft - noverlap
	plan.nSeg = (N - noverlap) / plan.hop
	if plan.method == FreqRespEstH1 && m > 1 && plan.nSeg < m {
		return freqRespEstPlan{}, fmt.Errorf("%d segments cannot identify %d inputs; use more data, a smaller NFFT or more overlap: %w", plan.nSeg, m, ErrInsufficientData)
	}
	return plan, nil
}

func welchSISO(uData, yData []float64, plan freqRespEstPlan) freqRespEstimate {
	nfft, nFreq := plan.nfft, plan.nFreq
	fft := fourier.NewFFT(nfft)
	seg := make([]float64, nfft)
	Suu := make([]float64, nFreq)
	Syy := make([]float64, nFreq)
	Syu := make([]complex128, nFreq)

	uCoeff := make([]complex128, nFreq)
	yCoeff := make([]complex128, nFreq)

	for s := range plan.nSeg {
		start := s * plan.hop

		for k := range nfft {
			seg[k] = uData[start+k] * plan.win[k]
		}
		fft.Coefficients(uCoeff, seg)

		for k := range nfft {
			seg[k] = yData[start+k] * plan.win[k]
		}
		fft.Coefficients(yCoeff, seg)

		for f := range nFreq {
			Suu[f] += real(uCoeff[f])*real(uCoeff[f]) + imag(uCoeff[f])*imag(uCoeff[f])
			Syy[f] += real(yCoeff[f])*real(yCoeff[f]) + imag(yCoeff[f])*imag(yCoeff[f])
			Syu[f] += cmplx.Conj(uCoeff[f]) * yCoeff[f]
		}
	}

	maxSuu := 0.0
	for _, v := range Suu {
		maxSuu = max(maxSuu, v)
	}
	sthresh := eps() * maxSuu

	est := freqRespEstimate{h: make([]complex128, nFreq), cohRe: make([]float64, nFreq), ok: make([]bool, nFreq)}
	for f := range nFreq {
		if !(Suu[f] > sthresh) {
			continue
		}
		if plan.method == FreqRespEstH2 {
			if cmplx.Abs(Syu[f]) < eps()*math.Sqrt(Syy[f]*Suu[f]) || Syu[f] == 0 {
				continue
			}
			est.h[f] = complex(Syy[f], 0) / cmplx.Conj(Syu[f])
		} else {
			est.h[f] = Syu[f] / complex(Suu[f], 0)
		}
		est.ok[f] = true
		if denom := Suu[f] * Syy[f]; denom > 0 {
			est.cohRe[f] = cmplx.Abs(Syu[f]) * cmplx.Abs(Syu[f]) / denom
		}
	}
	return est
}
func welchMIMO(input, output *mat.Dense, m, p int, plan freqRespEstPlan) freqRespEstimate {
	nfft, nFreq, win := plan.nfft, plan.nFreq, plan.win
	fft := fourier.NewFFT(nfft)
	seg := make([]float64, nfft)
	inR := input.RawMatrix()
	outR := output.RawMatrix()

	// Flat FFT storage: uFlat[ch*nFreq + f], yFlat[ch*nFreq + f]
	uFlat := make([]complex128, m*nFreq)
	yFlat := make([]complex128, p*nFreq)

	Suu := make([]complex128, nFreq*m*m)
	Syu := make([]complex128, nFreq*p*m)
	Syy := make([]complex128, nFreq*p*p)

	mm := m * m
	pm := p * m
	pp := p * p

	for s := range plan.nSeg {
		start := s * plan.hop

		for ch := range m {
			for k := range nfft {
				seg[k] = inR.Data[ch*inR.Stride+start+k] * win[k]
			}
			fft.Coefficients(uFlat[ch*nFreq:(ch+1)*nFreq], seg)
		}
		for ch := range p {
			for k := range nfft {
				seg[k] = outR.Data[ch*outR.Stride+start+k] * win[k]
			}
			fft.Coefficients(yFlat[ch*nFreq:(ch+1)*nFreq], seg)
		}

		// Fused accumulation: iterate channel pairs, then sweep frequencies
		for j1 := range m {
			u1 := uFlat[j1*nFreq : (j1+1)*nFreq]
			for j2 := range m {
				u2 := uFlat[j2*nFreq : (j2+1)*nFreq]
				off := j1*m + j2
				for f := range nFreq {
					Suu[f*mm+off] += cmplx.Conj(u1[f]) * u2[f]
				}
			}
		}
		for i := range p {
			yi := yFlat[i*nFreq : (i+1)*nFreq]
			for j := range m {
				uj := uFlat[j*nFreq : (j+1)*nFreq]
				off := i*m + j
				for f := range nFreq {
					Syu[f*pm+off] += cmplx.Conj(uj[f]) * yi[f]
				}
			}
		}
		for i1 := range p {
			y1 := yFlat[i1*nFreq : (i1+1)*nFreq]
			for i2 := range p {
				y2 := yFlat[i2*nFreq : (i2+1)*nFreq]
				off := i1*p + i2
				for f := range nFreq {
					Syy[f*pp+off] += cmplx.Conj(y1[f]) * y2[f]
				}
			}
		}
	}

	H := make([]complex128, nFreq*p*m)
	ok := make([]bool, nFreq)
	invBuf := make([]complex128, m*2*m)
	invRes := make([]complex128, m*m)

	for f := range nFreq {
		suu := Suu[f*m*m : (f+1)*m*m]
		syu := Syu[f*p*m : (f+1)*p*m]

		if err := cInvertInto(invRes, invBuf, suu, m); err != nil {
			continue
		}
		ok[f] = true
		cMulInto(H[f*p*m:(f+1)*p*m], syu, invRes, p, m, m)
	}

	coh := make([]float64, nFreq*p*m)
	for f := range nFreq {
		for i := range p {
			for j := range m {
				syy_ii := Syy[f*p*p+i*p+i]
				suu_jj := Suu[f*m*m+j*m+j]
				denom := real(syy_ii) * real(suu_jj)
				if denom > 0 {
					s := Syu[f*p*m+i*m+j]
					coh[f*p*m+i*m+j] = (real(s)*real(s) + imag(s)*imag(s)) / denom
				}
			}
		}
	}

	return freqRespEstimate{h: H, cohRe: coh, ok: ok}
}

func freqRespEstFFT(input, output *mat.Dense, m, p int, plan freqRespEstPlan) freqRespEstimate {
	nfft, nFreq, win := plan.nfft, plan.nFreq, plan.win
	fft := fourier.NewFFT(nfft)
	seg := make([]float64, nfft)

	inRaw := input.RawMatrix()
	outRaw := output.RawMatrix()

	yCoeffs := make([]complex128, p*nFreq)
	for i := range p {
		for k := range nfft {
			seg[k] = outRaw.Data[i*outRaw.Stride+k] * win[k]
		}
		fft.Coefficients(yCoeffs[i*nFreq:(i+1)*nFreq], seg)
	}

	H := make([]complex128, nFreq*p*m)
	ok := make([]bool, nFreq)
	for f := range ok {
		ok[f] = true
	}
	uCoeff := make([]complex128, nFreq)
	for j := range m {
		for k := range nfft {
			seg[k] = inRaw.Data[j*inRaw.Stride+k] * win[k]
		}
		fft.Coefficients(uCoeff, seg)
		for f := range nFreq {
			if cmplx.Abs(uCoeff[f]) < 1e-30 {
				ok[f] = false
				continue
			}
			for i := range p {
				H[f*p*m+i*m+j] = yCoeffs[i*nFreq+f] / uCoeff[f]
			}
		}
	}
	return freqRespEstimate{h: H, ok: ok}
}
