package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// PassivityOptions holds the optional arguments of SampledPassive and
// FRDPassive; nil or zero fields select the defaults.
type PassivityOptions struct {
	// Omega is the frequency grid in rad/s: finite, non-negative, strictly
	// increasing and at most π/Dt for discrete models. nil selects 0 plus 120
	// log-spaced points up to 100 rad/s (π/Dt for discrete models).
	// FRDPassive uses the FRD's own grid and ignores it.
	Omega []float64
	// Tol is the tolerance on the minimum Hermitian-part eigenvalue: the
	// model passes while it is >= -Tol. 0 means 1e-9; it must be finite and
	// non-negative.
	Tol float64
}

// PassivityStatus distinguishes a sampled pass from a conclusive violation.
type PassivityStatus string

const (
	// PassivityViolated means a grid frequency has a negative Hermitian
	// part: a conclusive counterexample.
	PassivityViolated PassivityStatus = "violated"
	// PassivitySampled means every grid frequency passed: evidence, not an
	// analytic certificate.
	PassivitySampled PassivityStatus = "sampled-pass"
)

// PassivityResult is the sampled positive-real test of SampledPassive or
// FRDPassive.
type PassivityResult struct {
	Status PassivityStatus
	// MinHermitianPart is the smallest eigenvalue of (H(jω)+H(jω)ᴴ)/2 over
	// the grid, reached at Frequency.
	MinHermitianPart float64
	Frequency        float64
	Omega            []float64
	Tolerance        float64
}

// Passive reports whether no grid frequency violated passivity, that is
// Status != PassivityViolated.
func (r *PassivityResult) Passive() bool { return r.Status != PassivityViolated }

// Passive reports passivity evidence over a finite frequency grid.
//
// Deprecated: use SampledPassive to make the sampled guarantee explicit.
func Passive(sys *System, opts *PassivityOptions) (*PassivityResult, error) {
	return SampledPassive(sys, opts)
}

// SampledPassive checks the positive-real frequency-domain condition of a
// stable, square, delay-free model over a finite grid, the sampled analogue
// of MATLAB isPassive
// (https://www.mathworks.com/help/control/ref/dynamicsystem.ispassive.html).
// PassivitySampled is evidence, not an analytic certificate;
// PassivityViolated identifies a conclusive sampled counterexample.
func SampledPassive(sys *System, opts *PassivityOptions) (*PassivityResult, error) {
	if err := requireFiniteSystem("SampledPassive", sys); err != nil {
		return nil, err
	}
	if err := newDescriptorPolicy(sys).requireStandard("SampledPassive"); err != nil {
		return nil, err
	}
	if sys.HasDelay() {
		return nil, fmt.Errorf("SampledPassive: delayed systems are not supported: %w", ErrDelayUnsupported)
	}
	stable, err := sys.IsStable()
	if err != nil {
		return nil, fmt.Errorf("SampledPassive: %w", err)
	}
	if !stable {
		return nil, fmt.Errorf("SampledPassive: %w", ErrUnstable)
	}
	omega, tol, err := passivityGrid(opts, sys.Dt)
	if err != nil {
		return nil, fmt.Errorf("SampledPassive: %w", err)
	}
	frd, err := sys.FRD(omega)
	if err != nil {
		return nil, fmt.Errorf("SampledPassive: %w", err)
	}
	res, err := FRDPassive(frd, &PassivityOptions{Tol: tol})
	if err != nil {
		return nil, fmt.Errorf("SampledPassive: %w", err)
	}
	return res, nil
}

// FRDPassive applies the sampled positive-real test of SampledPassive to the
// square frequency response data frd on its own grid.
func FRDPassive(frd *FRD, opts *PassivityOptions) (*PassivityResult, error) {
	if frd == nil || len(frd.Response) == 0 {
		return nil, fmt.Errorf("FRDPassive: insufficient data: %w", ErrInsufficientData)
	}
	if len(frd.Omega) != len(frd.Response) {
		return nil, fmt.Errorf("FRDPassive: %d responses for %d frequencies: %w", len(frd.Response), len(frd.Omega), ErrDimensionMismatch)
	}
	if err := validatePassivityGrid(frd.Omega, frd.Dt); err != nil {
		return nil, fmt.Errorf("FRDPassive: %w", err)
	}
	if len(frd.Response[0]) == 0 || len(frd.Response[0][0]) == 0 {
		return nil, fmt.Errorf("FRDPassive: empty response matrix: %w", ErrDimensionMismatch)
	}
	p, m := len(frd.Response[0]), len(frd.Response[0][0])
	if p != m {
		return nil, fmt.Errorf("FRDPassive: model must be square, got %dx%d: %w", p, m, ErrDimensionMismatch)
	}
	tol, err := passivityTolerance(opts)
	if err != nil {
		return nil, fmt.Errorf("FRDPassive: %w", err)
	}
	result := &PassivityResult{
		Status:           PassivitySampled,
		MinHermitianPart: math.Inf(1),
		Omega:            copyFloatSlice(frd.Omega),
		Tolerance:        tol,
	}
	for k, h := range frd.Response {
		if len(h) != p {
			return nil, fmt.Errorf("FRDPassive: response[%d] has %d rows, want %d: %w", k, len(h), p, ErrDimensionMismatch)
		}
		for i, row := range h {
			if len(row) != m {
				return nil, fmt.Errorf("FRDPassive: response[%d][%d] has %d columns, want %d: %w", k, i, len(row), m, ErrDimensionMismatch)
			}
			for j, value := range row {
				if !finiteComplex(value) {
					return nil, fmt.Errorf("FRDPassive: response[%d][%d][%d] is not finite: %w", k, i, j, ErrInvalidArgument)
				}
			}
		}
		minPart, err := minHermitianPart(h)
		if err != nil {
			return nil, fmt.Errorf("FRDPassive: response[%d]: %w", k, err)
		}
		if minPart < result.MinHermitianPart {
			result.MinHermitianPart = minPart
			if k < len(frd.Omega) {
				result.Frequency = frd.Omega[k]
			}
		}
	}
	if result.MinHermitianPart < -tol {
		result.Status = PassivityViolated
	}
	return result, nil
}

// SpectralFactor returns W with W~·W = sys for a square static gain with a
// non-negative diagonal. It is a restricted form of MATLAB spectralfact
// (https://www.mathworks.com/help/control/ref/dynamicsystem.spectralfact.html):
// dynamic models and non-diagonal gains return ErrOptionUnsupported, and a
// negative diagonal ErrNotPSD.
func SpectralFactor(sys *System) (*System, error) {
	if err := requireSystem("SpectralFactor", sys); err != nil {
		return nil, err
	}
	if err := newDescriptorPolicy(sys).requireStandard("SpectralFactor"); err != nil {
		return nil, err
	}
	if sys.HasDelay() {
		return nil, fmt.Errorf("SpectralFactor: delayed systems are not supported: %w", ErrDelayUnsupported)
	}
	n, m, p := sys.Dims()
	if n != 0 {
		return nil, fmt.Errorf("SpectralFactor: dynamic models are not implemented, only static gains: %w", ErrOptionUnsupported)
	}
	if m != p {
		return nil, fmt.Errorf("SpectralFactor: model must be square, got %dx%d: %w", p, m, ErrDimensionMismatch)
	}
	D := newDense(p, m)
	for i := range p {
		for j := range m {
			if i != j && sys.D.At(i, j) != 0 {
				return nil, fmt.Errorf("SpectralFactor: only diagonal static gains are implemented: %w", ErrOptionUnsupported)
			}
			if i == j {
				v := sys.D.At(i, j)
				if v < 0 {
					return nil, fmt.Errorf("SpectralFactor: negative diagonal value: %w", ErrNotPSD)
				}
				D.Set(i, j, math.Sqrt(v))
			}
		}
	}
	return NewGain(D, sys.Dt)
}

func passivityGrid(opts *PassivityOptions, dt float64) ([]float64, float64, error) {
	tol, err := passivityTolerance(opts)
	if err != nil {
		return nil, 0, err
	}
	if opts != nil && len(opts.Omega) > 0 {
		omega := copyFloatSlice(opts.Omega)
		if err := validatePassivityGrid(omega, dt); err != nil {
			return nil, 0, err
		}
		return omega, tol, nil
	}
	upper := 1e2
	if dt > 0 {
		upper = math.Pi / dt
	}
	lower := math.Min(1e-2, upper*1e-4)
	omega := make([]float64, 121)
	copy(omega[1:], logspace(math.Log10(lower), math.Log10(upper), 120))
	omega[1] = lower
	omega[len(omega)-1] = upper
	return omega, tol, nil
}

func passivityTolerance(opts *PassivityOptions) (float64, error) {
	tol := 1e-9
	if opts == nil || opts.Tol == 0 {
		return tol, nil
	}
	if math.IsNaN(opts.Tol) || math.IsInf(opts.Tol, 0) || opts.Tol < 0 {
		return 0, fmt.Errorf("invalid tolerance %g: %w", opts.Tol, ErrInvalidArgument)
	}
	return opts.Tol, nil
}

func validatePassivityGrid(omega []float64, dt float64) error {
	if len(omega) == 0 {
		return fmt.Errorf("frequency grid is empty: %w", ErrInsufficientData)
	}
	upper := math.Inf(1)
	if dt > 0 {
		upper = math.Pi / dt
	}
	for i, frequency := range omega {
		if math.IsNaN(frequency) || math.IsInf(frequency, 0) || frequency < 0 {
			return fmt.Errorf("omega[%d]=%g is invalid: %w", i, frequency, ErrInvalidArgument)
		}
		if frequency > upper*(1+1e-12) {
			return fmt.Errorf("omega[%d]=%g exceeds Nyquist frequency %g: %w", i, frequency, upper, ErrInvalidArgument)
		}
		if i > 0 && frequency <= omega[i-1] {
			return fmt.Errorf("frequency grid is not strictly increasing at index %d: %w", i, ErrInvalidArgument)
		}
	}
	return nil
}

func finiteComplex(value complex128) bool {
	return !math.IsNaN(real(value)) && !math.IsNaN(imag(value)) && !math.IsInf(real(value), 0) && !math.IsInf(imag(value), 0)
}

func minHermitianPart(h [][]complex128) (float64, error) {
	if len(h) == 1 && len(h[0]) == 1 {
		return real(h[0][0]), nil
	}
	n := len(h)
	if n == 2 {
		a := real(h[0][0])
		d := real(h[1][1])
		off := (h[0][1] + cmplx.Conj(h[1][0])) / 2
		halfDiff := (a - d) / 2
		radius := math.Sqrt(halfDiff*halfDiff + real(off)*real(off) + imag(off)*imag(off))
		return (a+d)/2 - radius, nil
	}
	dim := 2 * n
	sym := mat.NewSymDense(dim, nil)
	for i := range n {
		for j := range n {
			herm := (h[i][j] + cmplx.Conj(h[j][i])) / 2
			re := real(herm)
			im := imag(herm)
			sym.SetSym(i, j, re)
			sym.SetSym(i, n+j, -im)
			sym.SetSym(n+i, j, im)
			sym.SetSym(n+i, n+j, re)
		}
	}
	var eig mat.EigenSym
	if ok := eig.Factorize(sym, false); !ok {
		return 0, fmt.Errorf("Hermitian part eigenvalues: %w", ErrSchurFailed)
	}
	vals := eig.Values(nil)
	minPart := vals[0]
	for _, v := range vals[1:] {
		if v < minPart {
			minPart = v
		}
	}
	return minPart, nil
}
