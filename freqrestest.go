package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/dsp/fourier"
	"plantcontrol.org/v1/gonum/dsp/window"
	"plantcontrol.org/v1/gonum/mat"
)

type FreqRespEstOpts struct {
	NFFT     int
	Window   func([]float64) []float64
	NOverlap int
	Method   FreqRespEstMethod
}

type FreqRespEstMethod string

const (
	FreqRespEstH1  FreqRespEstMethod = "h1"
	FreqRespEstH2  FreqRespEstMethod = "h2"
	FreqRespEstFFT FreqRespEstMethod = "fft"
)

type FreqRespEstResult struct {
	H         *FreqResponseMatrix
	Omega     []float64
	Dt        float64
	Coherence []float64
}

type freqRespEstPlan struct {
	nfft     int
	noverlap int
	hop      int
	nSeg     int
	nFreq    int
	method   FreqRespEstMethod
	win      []float64
	winPower float64
}

func (r *FreqRespEstResult) CoherenceAt(freq, output, input int) float64 {
	if r.Coherence == nil {
		return 0
	}
	return newSampledScalarResponse(r.Coherence, r.Omega, r.H.P, r.H.M).at(freq, output, input)
}

func (r *FreqRespEstResult) FRD() (*FRD, error) {
	if r == nil || r.H == nil {
		return nil, ErrInsufficientData
	}
	return newFRDFromFreqResponse(r.H, r.Dt)
}

func FreqRespEst(input, output *mat.Dense, dt float64, opts *FreqRespEstOpts) (*FreqRespEstResult, error) {
	m, p, N, err := validateSampledIO("FreqRespEst", input, output, dt)
	if err != nil {
		return nil, err
	}

	plan, err := newFreqRespEstPlan(opts, N, m, p)
	if err != nil {
		return nil, err
	}

	if plan.method == FreqRespEstFFT {
		return freqRespEstFFT(input, output, dt, m, p, N, plan.nfft, plan.win, plan.winPower)
	}

	fft := fourier.NewFFT(plan.nfft)
	seg := make([]float64, plan.nfft)
	coeff := make([]complex128, plan.nFreq)

	inRaw := input.RawMatrix()
	outRaw := output.RawMatrix()

	if m == 1 && p == 1 {
		return welchSISO(fft, seg, coeff, plan.win, inRaw.Data, inRaw.Stride, outRaw.Data, outRaw.Stride,
			dt, plan.nfft, plan.nFreq, plan.hop, plan.nSeg, plan.winPower, plan.method)
	}

	return welchMIMO(fft, seg, coeff, plan.win, input, output,
		dt, m, p, plan.nfft, plan.nFreq, plan.hop, plan.nSeg, plan.winPower, plan.method)
}

func validateFreqRespEstMethod(method FreqRespEstMethod, m, p int) error {
	switch method {
	case FreqRespEstH1, FreqRespEstH2, FreqRespEstFFT:
	default:
		return fmt.Errorf("FreqRespEst: unknown method %q: %w", method, ErrDimensionMismatch)
	}
	if method == FreqRespEstH2 && (m != 1 || p != 1) {
		return fmt.Errorf("FreqRespEst: h2 estimator is only supported for SISO data: %w", ErrDimensionMismatch)
	}
	return nil
}

func resolveOpts(opts *FreqRespEstOpts, N int) (nfft int, winFunc func([]float64) []float64, noverlap int, method FreqRespEstMethod) {
	nfft = 256
	winFunc = window.Hann
	noverlap = -1
	method = FreqRespEstH1

	if opts != nil {
		if opts.NFFT > 0 {
			nfft = opts.NFFT
		}
		if opts.Window != nil {
			winFunc = opts.Window
		}
		if opts.NOverlap > 0 {
			noverlap = opts.NOverlap
		}
		if opts.Method != "" {
			method = opts.Method
		}
	}

	if nfft > N {
		nfft = N
	}
	if noverlap < 0 {
		noverlap = nfft / 2
	}
	if noverlap >= nfft {
		noverlap = nfft - 1
	}

	return
}

func newFreqRespEstPlan(opts *FreqRespEstOpts, N, m, p int) (freqRespEstPlan, error) {
	nfft, winFunc, noverlap, method := resolveOpts(opts, N)
	if err := validateFreqRespEstMethod(method, m, p); err != nil {
		return freqRespEstPlan{}, err
	}

	plan := freqRespEstPlan{nfft: nfft, noverlap: noverlap, method: method}
	plan.rebuildWindow(winFunc)
	if method == FreqRespEstFFT {
		return plan, nil
	}

	plan.hop = plan.nfft - plan.noverlap
	plan.nSeg = (N - plan.noverlap) / plan.hop
	if plan.nSeg < 1 {
		plan.nfft = N
		plan.noverlap = 0
		plan.hop = N
		plan.nSeg = 1
		plan.rebuildWindow(winFunc)
	}
	return plan, nil
}

func (p *freqRespEstPlan) rebuildWindow(winFunc func([]float64) []float64) {
	p.win = make([]float64, p.nfft)
	for i := range p.win {
		p.win[i] = 1
	}
	winFunc(p.win)
	p.winPower = 0
	for _, w := range p.win {
		p.winPower += w * w
	}
	p.winPower /= float64(p.nfft)
	p.nFreq = p.nfft/2 + 1
}

func welchSISO(fft *fourier.FFT, seg []float64, _ []complex128, win []float64,
	uData []float64, _ int, yData []float64, _ int,
	dt float64, nfft, nFreq, hop, nSeg int, _ float64, method FreqRespEstMethod,
) (*FreqRespEstResult, error) {

	Suu := make([]float64, nFreq)
	Syy := make([]float64, nFreq)
	Syu := make([]complex128, nFreq)

	uCoeff := make([]complex128, nFreq)
	yCoeff := make([]complex128, nFreq)

	for s := range nSeg {
		start := s * hop

		for k := range nfft {
			seg[k] = uData[start+k] * win[k]
		}
		fft.Coefficients(uCoeff, seg)

		for k := range nfft {
			seg[k] = yData[start+k] * win[k]
		}
		fft.Coefficients(yCoeff, seg)

		for f := range nFreq {
			Suu[f] += real(uCoeff[f])*real(uCoeff[f]) + imag(uCoeff[f])*imag(uCoeff[f])
			Syy[f] += real(yCoeff[f])*real(yCoeff[f]) + imag(yCoeff[f])*imag(yCoeff[f])
			Syu[f] += cmplx.Conj(uCoeff[f]) * yCoeff[f]
		}
	}

	invN := 1.0 / float64(nSeg)
	for f := range nFreq {
		Suu[f] *= invN
		Syy[f] *= invN
		Syu[f] *= complex(invN, 0)
	}

	maxSuu := 0.0
	for _, v := range Suu {
		if v > maxSuu {
			maxSuu = v
		}
	}
	sthresh := eps() * maxSuu

	H := make([]complex128, nFreq)
	coh := make([]float64, nFreq)

	for f := range nFreq {
		if Suu[f] < sthresh {
			continue
		}
		if method == FreqRespEstH2 {
			if cmplx.Abs(Syu[f]) < eps()*math.Sqrt(Syy[f]*Suu[f]) {
				continue
			}
			H[f] = complex(Syy[f], 0) / cmplx.Conj(Syu[f])
		} else {
			H[f] = Syu[f] / complex(Suu[f], 0)
		}
		denom := Suu[f] * Syy[f]
		if denom > 0 {
			coh[f] = cmplx.Abs(Syu[f]) * cmplx.Abs(Syu[f]) / denom
		}
	}

	omega := make([]float64, nFreq)
	for f := range nFreq {
		omega[f] = 2 * math.Pi * float64(f) / (float64(nfft) * dt)
	}

	return &FreqRespEstResult{
		H:         newFreqResponseMatrixOwned(H, omega, 1, 1, nil, nil),
		Omega:     omega,
		Dt:        dt,
		Coherence: coh,
	}, nil
}

func welchMIMO(fft *fourier.FFT, seg []float64, _ []complex128, win []float64,
	input, output *mat.Dense,
	dt float64, m, p, nfft, nFreq, hop, nSeg int, _ float64, _ FreqRespEstMethod,
) (*FreqRespEstResult, error) {
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

	for s := range nSeg {
		start := s * hop

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

	invN := complex(1.0/float64(nSeg), 0)
	for i := range Suu {
		Suu[i] *= invN
	}
	for i := range Syu {
		Syu[i] *= invN
	}
	for i := range Syy {
		Syy[i] *= invN
	}

	H := make([]complex128, nFreq*p*m)
	invBuf := make([]complex128, m*2*m)
	invRes := make([]complex128, m*m)

	for f := range nFreq {
		suu := Suu[f*m*m : (f+1)*m*m]
		syu := Syu[f*p*m : (f+1)*p*m]

		err := cInvertInto(invRes, invBuf, suu, m)
		if err != nil {
			continue
		}

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

	omega := make([]float64, nFreq)
	for f := range nFreq {
		omega[f] = 2 * math.Pi * float64(f) / (float64(nfft) * dt)
	}

	return &FreqRespEstResult{
		H:         newFreqResponseMatrixOwned(H, omega, p, m, nil, nil),
		Omega:     omega,
		Dt:        dt,
		Coherence: coh,
	}, nil
}

func freqRespEstFFT(input, output *mat.Dense, dt float64, m, p, _, nfft int,
	win []float64, _ float64,
) (*FreqRespEstResult, error) {
	nFreq := nfft/2 + 1
	fft := fourier.NewFFT(nfft)
	seg := make([]float64, nfft)

	inRaw := input.RawMatrix()
	outRaw := output.RawMatrix()

	// Pre-compute all output FFTs once: O(p) instead of O(p*m)
	yCoeffs := make([]complex128, p*nFreq)
	for i := range p {
		for k := range nfft {
			seg[k] = outRaw.Data[i*outRaw.Stride+k] * win[k]
		}
		fft.Coefficients(yCoeffs[i*nFreq:(i+1)*nFreq], seg)
	}

	H := make([]complex128, nFreq*p*m)
	uCoeff := make([]complex128, nFreq)

	for j := range m {
		for k := range nfft {
			seg[k] = inRaw.Data[j*inRaw.Stride+k] * win[k]
		}
		fft.Coefficients(uCoeff, seg)

		for i := range p {
			yC := yCoeffs[i*nFreq : (i+1)*nFreq]
			for f := range nFreq {
				if cmplx.Abs(uCoeff[f]) < 1e-30 {
					continue
				}
				H[f*p*m+i*m+j] = yC[f] / uCoeff[f]
			}
		}
	}

	omega := make([]float64, nFreq)
	for f := range nFreq {
		omega[f] = 2 * math.Pi * float64(f) / (float64(nfft) * dt)
	}

	return &FreqRespEstResult{
		H:     newFreqResponseMatrixOwned(H, omega, p, m, nil, nil),
		Omega: omega,
		Dt:    dt,
	}, nil
}
