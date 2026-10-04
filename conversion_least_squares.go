package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

// LeastSquaresResult describes an approximate frequency-response fit. Errors are
// measured on a validation grid distinct from the fitting grid. RMSRelativeError
// divides RMS absolute error by RMS source magnitude; MaxRelativeError divides
// maximum absolute error by maximum source magnitude. Zero-source errors are zero.
// Stable reports the fitted poles; unstable poles are never silently reflected.
type LeastSquaresResult struct {
	Sys              *System
	FitOrder         int
	RMSRelativeError float64
	MaxRelativeError float64
	Stable           bool
}

// DiscretizeLeastSquares fits an ordinary proper SISO model up to Nyquist.
// fitOrder=0 selects the source state order (after eliminating the algebraic
// states of a singular-E descriptor model); positive orders are accepted.
// The fit uses an equally weighted uniform grid with max(513,32*order+1) points
// and up to 12 denominator-reweighted real least-squares iterations followed
// by up to 8 response-error Gauss-Newton refinements. It selects
// the iterate with smallest frequency-response error. Stability is not imposed.
// Integrators retain their multiplicity and leading low-frequency residue;
// they require sufficient fit order and use an open midpoint frequency grid.
// Integer external delays are preserved; fractional and internal delays and
// improper descriptor models are rejected. A finite grid cannot certify intersample error
// or capture arbitrarily narrow resonances. No exact inverse is promised.
func (sys *System) DiscretizeLeastSquares(dt float64, fitOrder int) (*LeastSquaresResult, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeLeastSquares: %w", ErrWrongDomain)
	}
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, ErrInvalidSampleTime
	}
	if _, m, p := sys.Dims(); m != 1 || p != 1 {
		return nil, ErrNotSISO
	}
	sys, _, err := conversionStandardForm(sys, "DiscretizeLeastSquares")
	if err != nil {
		return nil, err
	}
	n, _, _ := sys.Dims()
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("DiscretizeLeastSquares: internal delays: %w", ErrFeedbackDelay)
	}
	if fitOrder < 0 {
		return nil, ErrInvalidOrder
	}
	if fitOrder == 0 {
		fitOrder = n
	}

	work := sys.Copy()
	work.InputDelay, work.OutputDelay, work.Delay = nil, nil, nil
	source, err := work.TransferFunction(nil)
	if err != nil {
		return nil, err
	}
	maxInt := int(^uint(0) >> 1)
	if fitOrder > (maxInt-1)/32 {
		return nil, ErrInvalidOrder
	}
	integrators, residue := leastSquaresIntegratorLimit(source.TF)
	if integrators > fitOrder {
		return nil, fmt.Errorf("fit order must retain %d integrators: %w", integrators, ErrInvalidOrder)
	}
	samples := max(513, 32*fitOrder+1)
	if samples > maxInt/2 || 2*fitOrder+1 > maxInt/(2*samples) {
		return nil, ErrInvalidOrder
	}
	target := make([]complex128, samples)
	for k := range samples {
		theta := math.Pi * float64(k) / float64(samples-1)
		if integrators > 0 {
			theta = math.Pi * (float64(k) + .5) / float64(samples)
		}
		target[k] = Poly(source.TF.Num[0][0]).Eval(complex(0, theta/dt)) / Poly(source.TF.Den[0]).Eval(complex(0, theta/dt))
		if cmplx.IsInf(target[k]) || cmplx.IsNaN(target[k]) {
			return nil, fmt.Errorf("DiscretizeLeastSquares: singular or nonfinite source frequency response: %w", ErrSingularTransform)
		}
	}
	var numerator, denominator []float64
	if integrators > 0 {
		numerator, denominator, err = fitDiscreteIntegrators(target, fitOrder, integrators, residue*math.Pow(dt, float64(integrators)))
	} else {
		numerator, denominator, err = fitDiscreteResponse(target, fitOrder)
	}
	if err != nil {
		return nil, err
	}
	fitted := &TransferFunc{Num: [][][]float64{{numerator}}, Den: [][]float64{denominator}, Dt: dt}
	realized, err := fitted.StateSpace(nil)
	if err != nil {
		return nil, err
	}
	out, err := newDelayConversionPolicy(dt, 0, 0).applyDiscreteDelayFields(realized.Sys, sys)
	if err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	out.StateName = nil
	rms, worst, err := leastSquaresFitQuality(source.TF, fitted, dt, samples)
	if err != nil {
		return nil, err
	}
	stable, err := out.IsStable()
	if err != nil {
		return nil, err
	}
	return &LeastSquaresResult{Sys: out, FitOrder: fitOrder, RMSRelativeError: rms, MaxRelativeError: worst, Stable: stable}, nil
}

func (sys *System) discretizeLeastSquares(dt float64, fitOrder int) (*System, error) {
	result, err := sys.DiscretizeLeastSquares(dt, fitOrder)
	if err != nil {
		return nil, err
	}
	return result.Sys, nil
}

func fitDiscreteResponse(target []complex128, order int) ([]float64, []float64, error) {
	if order == 0 {
		return []float64{real(target[0])}, []float64{1}, nil
	}
	rows, cols := 2*len(target), 2*order+1
	design := mat.NewDense(rows, cols, nil)
	rhs := mat.NewDense(rows, 1, nil)
	powers := make([]complex128, order+1)
	denominator := make([]float64, order+1)
	denominator[0] = 1
	bestError := math.Inf(1)
	var bestNum, bestDen []float64
	var solver frequencyFitSolver
	for range 12 {
		raw, b := design.RawMatrix(), rhs.RawMatrix()
		for k, value := range target {
			theta := math.Pi * float64(k) / float64(len(target)-1)
			q := cmplx.Exp(complex(0, -theta))
			powers[0] = 1
			for j := 1; j <= order; j++ {
				powers[j] = powers[j-1] * q
			}
			weight := 1 / math.Max(1e-8, cmplx.Abs(Poly(denominator).Eval(1/q)))
			for j := range cols {
				var v complex128
				if j < order {
					v = -value * powers[j+1]
				} else {
					v = powers[j-order]
				}
				v *= complex(weight, 0)
				raw.Data[2*k*raw.Stride+j], raw.Data[(2*k+1)*raw.Stride+j] = real(v), imag(v)
			}
			b.Data[2*k*b.Stride], b.Data[(2*k+1)*b.Stride] = weight*real(value), weight*imag(value)
		}
		x, err := solver.solve(design, rhs)
		if err != nil {
			return nil, nil, err
		}
		numerator := make([]float64, order+1)
		denominator = make([]float64, order+1)
		denominator[0] = 1
		for j := range order {
			denominator[j+1] = x[j]
		}
		for j := 0; j <= order; j++ {
			numerator[j] = x[order+j]
		}
		residual := 0.0
		for k, value := range target {
			z := cmplx.Exp(complex(0, math.Pi*float64(k)/float64(len(target)-1)))
			prediction := Poly(numerator).Eval(z) / Poly(denominator).Eval(z)
			residual = math.Hypot(residual, cmplx.Abs(prediction-value))
		}
		if residual < bestError {
			bestError, bestNum, bestDen = residual, numerator, denominator
		}
	}
	if bestNum == nil {
		return nil, nil, ErrSingularEquation
	}
	return refineDiscreteResponse(target, bestNum, bestDen)
}

func leastSquaresFitQuality(source, fitted *TransferFunc, dt float64, samples int) (float64, float64, error) {
	residual, magnitude, maxError, maxMagnitude := 0.0, 0.0, 0.0, 0.0
	for k := 0; k <= samples; k++ {
		theta := math.Pi * (float64(k) + 0.5) / float64(samples+1)
		actual := Poly(source.Num[0][0]).Eval(complex(0, theta/dt)) / Poly(source.Den[0]).Eval(complex(0, theta/dt))
		z := cmplx.Exp(complex(0, theta))
		prediction := Poly(fitted.Num[0][0]).Eval(z) / Poly(fitted.Den[0]).Eval(z)
		delta := cmplx.Abs(prediction - actual)
		if math.IsNaN(delta) || math.IsInf(delta, 0) {
			return 0, 0, ErrSingularTransform
		}
		amp := cmplx.Abs(actual)
		residual, magnitude = math.Hypot(residual, delta), math.Hypot(magnitude, amp)
		maxError, maxMagnitude = math.Max(maxError, delta), math.Max(maxMagnitude, amp)
	}
	if magnitude == 0 {
		return residual, maxError, nil
	}
	return residual / magnitude, maxError / maxMagnitude, nil
}

func leastSquaresIntegratorLimit(source *TransferFunc) (int, float64) {
	den := source.Den[0]
	norm := 0.0
	for _, x := range den {
		norm = math.Max(norm, math.Abs(x))
	}
	count := 0
	for len(den) > 1 && math.Abs(den[len(den)-1]) <= 1e-12*norm {
		count++
		den = den[:len(den)-1]
	}
	if count == 0 {
		return 0, 0
	}
	return count, source.Num[0][0][len(source.Num[0][0])-1] / den[len(den)-1]
}

func fitDiscreteIntegrators(target []complex128, order, integrators int, limit float64) ([]float64, []float64, error) {
	free := order - integrators
	rows, cols := 2*len(target), free+order
	design, rhs := mat.NewDense(rows, cols, nil), mat.NewDense(rows, 1, nil)
	base := Poly{1}
	for range integrators {
		base = base.Mul(Poly{1, -1})
	}
	denominator := make(Poly, free+1)
	denominator[0] = 1
	bestError := math.Inf(1)
	var bestNum, bestDen []float64
	powers := make([]complex128, order+1)
	var solver frequencyFitSolver
	for range 12 {
		raw, b := design.RawMatrix(), rhs.RawMatrix()
		for k, h := range target {
			theta := math.Pi * (float64(k) + .5) / float64(len(target))
			q := cmplx.Exp(complex(0, -theta))
			powers[0] = 1
			for j := 1; j <= order; j++ {
				powers[j] = powers[j-1] * q
			}
			baseValue := cmplx.Pow(1-q, complex(float64(integrators), 0))
			aValue := complex(0, 0)
			for j, a := range denominator {
				aValue += complex(a, 0) * powers[j]
			}
			weight := 1 / math.Max(1e-14, cmplx.Abs(baseValue*aValue))
			for j := range cols {
				var v complex128
				if j < free {
					v = complex(limit, 0) - h*baseValue*powers[j+1]
				} else {
					v = powers[j-free+1] - 1
				}
				raw.Data[2*k*raw.Stride+j], raw.Data[(2*k+1)*raw.Stride+j] = weight*real(v), weight*imag(v)
			}
			value := h*baseValue - complex(limit, 0)
			b.Data[2*k*b.Stride], b.Data[(2*k+1)*b.Stride] = weight*real(value), weight*imag(value)
		}
		x, err := solver.solve(design, rhs)
		if err != nil {
			return nil, nil, err
		}
		denominator = make(Poly, free+1)
		denominator[0] = 1
		numerator := make(Poly, order+1)
		numerator[0] = limit
		for j := range free {
			denominator[j+1] = x[j]
			numerator[0] += limit * x[j]
		}
		for j := 1; j <= order; j++ {
			numerator[j] = x[free+j-1]
			numerator[0] -= numerator[j]
		}
		fullDen := base.Mul(denominator)
		residual := 0.0
		for k, h := range target {
			z := cmplx.Exp(complex(0, math.Pi*(float64(k)+.5)/float64(len(target))))
			prediction := numerator.Eval(z) / fullDen.Eval(z)
			residual = math.Hypot(residual, cmplx.Abs(prediction-h))
		}
		if residual < bestError {
			bestError, bestNum, bestDen = residual, numerator, fullDen
		}
	}
	if bestNum == nil {
		return nil, nil, ErrSingularEquation
	}
	return bestNum, bestDen, nil
}

type frequencyFitSolver struct {
	svd          mat.SVD
	coefficients mat.Dense
	scales       []float64
}

func (solver *frequencyFitSolver) solve(design, rhs *mat.Dense) ([]float64, error) {
	raw := design.RawMatrix()
	rows, cols := design.Dims()
	if cap(solver.scales) < cols {
		solver.scales = make([]float64, cols)
	}
	scales := solver.scales[:cols]
	clear(scales)
	for j := range cols {
		for i := range rows {
			scales[j] = math.Hypot(scales[j], raw.Data[i*raw.Stride+j])
		}
		if scales[j] == 0 {
			scales[j] = 1
		}
		for i := range rows {
			raw.Data[i*raw.Stride+j] /= scales[j]
		}
	}
	if !solver.svd.Factorize(design, mat.SVDThin) {
		return nil, ErrSingularEquation
	}
	rank := solver.svd.Rank(1e-12)
	if rank == 0 {
		return nil, ErrSingularEquation
	}
	solver.svd.SolveTo(&solver.coefficients, rhs, rank)
	x := solver.coefficients.RawMatrix()
	result := make([]float64, cols)
	for j := range result {
		result[j] = x.Data[j*x.Stride] / scales[j]
	}
	return result, nil
}

func refineDiscreteResponse(target []complex128, numerator, denominator []float64) ([]float64, []float64, error) {
	order := len(denominator) - 1
	design, rhs := mat.NewDense(2*len(target), 2*order+1, nil), mat.NewDense(2*len(target), 1, nil)
	best := discreteResponseResidual(target, numerator, denominator)
	powers := make([]complex128, order+1)
	var solver frequencyFitSolver
	for range 8 {
		raw, b := design.RawMatrix(), rhs.RawMatrix()
		for k, h := range target {
			theta := math.Pi * float64(k) / float64(len(target)-1)
			q := cmplx.Exp(complex(0, -theta))
			powers[0] = 1
			for j := 1; j <= order; j++ {
				powers[j] = powers[j-1] * q
			}
			num, den := complex(0, 0), complex(0, 0)
			for j := 0; j <= order; j++ {
				num += complex(numerator[j], 0) * powers[j]
				den += complex(denominator[j], 0) * powers[j]
			}
			prediction := num / den
			for j := 0; j < 2*order+1; j++ {
				var value complex128
				if j < order {
					value = -prediction * powers[j+1] / den
				} else {
					value = powers[j-order] / den
				}
				raw.Data[2*k*raw.Stride+j], raw.Data[(2*k+1)*raw.Stride+j] = real(value), imag(value)
			}
			residual := h - prediction
			b.Data[2*k*b.Stride], b.Data[(2*k+1)*b.Stride] = real(residual), imag(residual)
		}
		delta, err := solver.solve(design, rhs)
		if err != nil {
			return numerator, denominator, nil
		}
		improved := false
		for step := 1.0; step >= 1.0/64; step /= 2 {
			trialNum, trialDen := make([]float64, order+1), make([]float64, order+1)
			trialDen[0] = 1
			for j := 1; j <= order; j++ {
				trialDen[j] = denominator[j] + step*delta[j-1]
			}
			for j := 0; j <= order; j++ {
				trialNum[j] = numerator[j] + step*delta[order+j]
			}
			residual := discreteResponseResidual(target, trialNum, trialDen)
			if residual < best {
				relativeImprovement := (best - residual) / math.Max(best, 1e-30)
				numerator, denominator, best = trialNum, trialDen, residual
				improved = true
				if relativeImprovement < 1e-9 {
					return numerator, denominator, nil
				}
				break
			}
		}
		if !improved {
			break
		}
	}
	return numerator, denominator, nil
}

func discreteResponseResidual(target []complex128, num, den []float64) float64 {
	residual := 0.0
	for k, h := range target {
		z := cmplx.Exp(complex(0, math.Pi*float64(k)/float64(len(target)-1)))
		prediction := Poly(num).Eval(z) / Poly(den).Eval(z)
		residual = math.Hypot(residual, cmplx.Abs(prediction-h))
	}
	return residual
}
