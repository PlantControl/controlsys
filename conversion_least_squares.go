package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// LeastSquaresFit describes an approximate frequency-response fit. Errors are
// measured on a validation grid distinct from the fitting grid. RMSRelativeError
// divides RMS absolute error by RMS source magnitude; MaxRelativeError divides
// maximum absolute error by maximum source magnitude. Both compare the
// fitted model with the source, each evaluated on its state-space realization.
// Zero-source errors are zero. FitOrder is the order used. Stable reports the
// fitted poles; unstable poles are never silently reflected.
type LeastSquaresFit struct {
	FitOrder         int
	RMSRelativeError float64
	MaxRelativeError float64
	Stable           bool
}

// C2DFit converts sys like C2D with opts.Method C2DMethodLeastSquares and
// also reports the fit quality, which MATLAB c2d does not return. Any other
// method returns ErrOptionUnsupported.
//
// The least-squares method fits an ordinary proper SISO model up to Nyquist
// (MATLAB c2d 'least-squares'; see
// https://www.mathworks.com/help/control/ug/continuous-discrete-conversion-methods.html).
// The source is sampled through its state-space realization.
// opts.FitOrder 0 ("auto", the MATLAB default) selects the source state order
// (after eliminating the algebraic states of a singular-E descriptor model);
// positive orders are accepted.
// The fit uses an equally weighted uniform grid with max(513,32*order+1) points
// and up to 12 denominator-reweighted real least-squares iterations followed
// by up to 8 response-error Gauss-Newton refinements. It selects
// the iterate with smallest frequency-response error. Stability is not imposed.
// Integrators, the poles at s = 0 of the minimal realization (null-space
// staircase with singular values at most 1e-10·‖A‖₁), retain their
// multiplicity and leading low-frequency residue as exact poles at z = 1;
// they require sufficient fit order and use an open midpoint frequency grid.
// Integer external delays are preserved; fractional and internal delays and
// improper descriptor models are rejected. A finite grid cannot certify intersample error
// or capture arbitrarily narrow resonances. No exact inverse is promised.
func (sys *System) C2DFit(dt float64, opts C2DOptions) (*System, LeastSquaresFit, error) {
	if err := requireFiniteSystem("C2DFit", sys); err != nil {
		return nil, LeastSquaresFit{}, err
	}
	if opts.Method != C2DMethodLeastSquares {
		return nil, LeastSquaresFit{}, fmt.Errorf("C2DFit: method %q is not least-squares: %w", opts.Method, ErrOptionUnsupported)
	}
	opts, err := normalizeC2DOptions(dt, opts)
	if err != nil {
		return nil, LeastSquaresFit{}, fmt.Errorf("C2DFit: %w", err)
	}
	out, fit, err := sys.leastSquaresFit(dt, opts.FitOrder)
	if err != nil {
		return nil, LeastSquaresFit{}, fmt.Errorf("C2DFit: %w", err)
	}
	return out, fit, nil
}

func (sys *System) leastSquaresFit(dt float64, fitOrder int) (*System, LeastSquaresFit, error) {
	var none LeastSquaresFit
	if sys.IsDiscrete() {
		return nil, none, ErrWrongDomain
	}
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, none, ErrInvalidSampleTime
	}
	if _, m, p := sys.Dims(); m != 1 || p != 1 {
		return nil, none, ErrNotSISO
	}
	sys, _, err := conversionStandardForm(sys, "descriptor reduction")
	if err != nil {
		return nil, none, err
	}
	n, _, _ := sys.Dims()
	if sys.HasInternalDelay() {
		return nil, none, fmt.Errorf("internal delays: %w", ErrFeedbackDelay)
	}
	if fitOrder < 0 {
		return nil, none, ErrInvalidOrder
	}
	if fitOrder == 0 {
		fitOrder = n
	}

	work := sys.Copy()
	work.InputDelay, work.OutputDelay, work.Delay = nil, nil, nil
	maxInt := int(^uint(0) >> 1)
	if fitOrder > (maxInt-1)/32 {
		return nil, none, ErrInvalidOrder
	}
	integrators, residue, err := leastSquaresIntegratorLimit(work)
	if err != nil {
		return nil, none, err
	}
	if integrators > fitOrder {
		return nil, none, fmt.Errorf("fit order must retain %d integrators: %w", integrators, ErrInvalidOrder)
	}
	samples := max(513, 32*fitOrder+1)
	if samples > maxInt/2 || 2*fitOrder+1 > maxInt/(2*samples) {
		return nil, none, ErrInvalidOrder
	}
	omega := make([]float64, samples)
	for k := range samples {
		theta := math.Pi * float64(k) / float64(samples-1)
		if integrators > 0 {
			theta = math.Pi * (float64(k) + .5) / float64(samples)
		}
		omega[k] = theta / dt
	}
	target, err := leastSquaresSourceResponse(work, omega)
	if err != nil {
		return nil, none, err
	}
	var realized *System
	if integrators > 0 {
		numerator, denominator, err := fitDiscreteIntegrators(target, fitOrder, integrators, residue*math.Pow(dt, float64(integrators)))
		if err != nil {
			return nil, none, err
		}
		realized, err = realizeDiscreteIntegrators(numerator, denominator, integrators, dt)
		if err != nil {
			return nil, none, err
		}
	} else {
		numerator, denominator, err := fitDiscreteResponse(target, fitOrder)
		if err != nil {
			return nil, none, err
		}
		fitted := &TransferFunc{Num: [][][]float64{{numerator}}, Den: [][]float64{denominator}, Dt: dt}
		result, err := fitted.StateSpace(nil)
		if err != nil {
			return nil, none, err
		}
		realized = result.Sys
	}
	rms, worst, err := leastSquaresFitQuality(work, realized, dt, samples)
	if err != nil {
		return nil, none, err
	}
	out, err := newDelayConversionPolicy(dt, 0, 0).applyDiscreteDelayFields(realized, sys)
	if err != nil {
		return nil, none, err
	}
	propagateNames(out, sys)
	out.StateName = nil
	stable, err := out.IsStable()
	if err != nil {
		return nil, none, err
	}
	return out, LeastSquaresFit{FitOrder: fitOrder, RMSRelativeError: rms, MaxRelativeError: worst, Stable: stable}, nil
}

func (sys *System) discretizeLeastSquares(dt float64, fitOrder int) (*System, error) {
	out, _, err := sys.leastSquaresFit(dt, fitOrder)
	return out, err
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

func leastSquaresSourceResponse(source *System, omega []float64) ([]complex128, error) {
	response, err := source.FreqResponse(omega)
	if err != nil {
		return nil, err
	}
	for _, h := range response.Data {
		if cmplx.IsInf(h) || cmplx.IsNaN(h) {
			return nil, fmt.Errorf("least-squares: singular or nonfinite source frequency response: %w", ErrSingularTransform)
		}
	}
	return response.Data, nil
}

func leastSquaresFitQuality(source, fitted *System, dt float64, samples int) (float64, float64, error) {
	omega := make([]float64, samples+1)
	for k := range omega {
		omega[k] = math.Pi * (float64(k) + 0.5) / float64(samples+1) / dt
	}
	actual, err := leastSquaresSourceResponse(source, omega)
	if err != nil {
		return 0, 0, err
	}
	predicted, err := fitted.FreqResponse(omega)
	if err != nil {
		return 0, 0, err
	}
	residual, magnitude, maxError, maxMagnitude := 0.0, 0.0, 0.0, 0.0
	for k, prediction := range predicted.Data {
		delta := cmplx.Abs(prediction - actual[k])
		if math.IsNaN(delta) || math.IsInf(delta, 0) {
			return 0, 0, ErrSingularTransform
		}
		amp := cmplx.Abs(actual[k])
		residual, magnitude = math.Hypot(residual, delta), math.Hypot(magnitude, amp)
		maxError, maxMagnitude = math.Max(maxError, delta), math.Max(maxMagnitude, amp)
	}
	if magnitude == 0 {
		return residual, maxError, nil
	}
	return residual / magnitude, maxError / maxMagnitude, nil
}

func dotVec(x, y []float64) float64 {
	return blas64.Dot(blas64.Vector{N: len(x), Inc: 1, Data: x}, blas64.Vector{N: len(y), Inc: 1, Data: y})
}

const leastSquaresIntegratorTol = 1e-10

// leastSquaresIntegratorLimit returns the multiplicity k of the pole at s = 0
// and lim s^k G(s). Eigenvalues of a k-fold Jordan chain spread by roundoff to
// eps^(1/k)·‖A‖, so the chain is found as a null-space staircase on the
// minimal realization instead: each step deflates a singular value at most
// leastSquaresIntegratorTol·‖A‖₁ with a Householder similarity, leaving
// A = [N A12; 0 A22] with N strictly upper triangular and A22 nonsingular, so
// lim s^k G(s) = C1 N^(k-1) (B1 - A12 A22⁻¹ B2).
func leastSquaresIntegratorLimit(source *System) (int, float64, error) {
	minimal, err := source.MinimalRealization()
	if err != nil {
		return 0, 0, err
	}
	n := minimal.Order
	if n == 0 {
		return 0, 0, nil
	}
	a := mat.DenseCopyOf(minimal.Sys.A)
	b := mat.VecDenseCopyOf(minimal.Sys.B.ColView(0))
	c := mat.VecDenseCopyOf(minimal.Sys.C.RowView(0))
	tol := leastSquaresIntegratorTol * mat.Norm(a, 1)
	raw := a.RawMatrix()
	var svd mat.SVD
	var v mat.Dense
	k := 0
	for ; k < n; k++ {
		m := n - k
		if !svd.Factorize(a.Slice(k, n, k, n), mat.SVDFullV) {
			return 0, 0, ErrSingularEquation
		}
		if svd.Values(nil)[m-1] > tol {
			break
		}
		v.Reset()
		svd.VTo(&v)
		u := make([]float64, m)
		for i := range m {
			u[i] = v.At(i, m-1)
		}
		u[0] += math.Copysign(1, u[0])
		beta := 2 / dotVec(u, u)
		for col := range n {
			dot := 0.0
			for i := range m {
				dot += u[i] * raw.Data[(k+i)*raw.Stride+col]
			}
			for i := range m {
				raw.Data[(k+i)*raw.Stride+col] -= beta * dot * u[i]
			}
		}
		for row := range n {
			r := raw.Data[row*raw.Stride+k : row*raw.Stride+n]
			dot := dotVec(r, u)
			for i := range m {
				r[i] -= beta * dot * u[i]
			}
		}
		bk, ck := b.RawVector().Data[k:n], c.RawVector().Data[k:n]
		db, dc := beta*dotVec(bk, u), beta*dotVec(ck, u)
		for i := range m {
			bk[i] -= db * u[i]
			ck[i] -= dc * u[i]
			raw.Data[(k+i)*raw.Stride+k] = 0
		}
	}
	if k == 0 {
		return 0, 0, nil
	}
	x := mat.VecDenseCopyOf(b.SliceVec(0, k))
	if k < n {
		var y mat.VecDense
		if err := y.SolveVec(a.Slice(k, n, k, n), b.SliceVec(k, n)); err != nil {
			return 0, 0, err
		}
		var shift mat.VecDense
		shift.MulVec(a.Slice(0, k, k, n), &y)
		x.SubVec(x, &shift)
	}
	next := mat.NewVecDense(k, nil)
	for range k - 1 {
		next.MulVec(a.Slice(0, k, 0, k), x)
		x, next = next, x
	}
	return k, mat.Dot(c.SliceVec(0, k), x), nil
}

func fitDiscreteIntegrators(target []complex128, order, integrators int, limit float64) ([]float64, []float64, error) {
	free := order - integrators
	rows, cols := 2*len(target), free+order
	design, rhs := mat.NewDense(rows, cols, nil), mat.NewDense(rows, 1, nil)
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
		residual := 0.0
		for k, h := range target {
			z := cmplx.Exp(complex(0, math.Pi*(float64(k)+.5)/float64(len(target))))
			prediction := numerator.Eval(z) / (cmplx.Pow(z-1, complex(float64(integrators), 0)) * denominator.Eval(z))
			residual = math.Hypot(residual, cmplx.Abs(prediction-h))
		}
		if residual < bestError {
			bestError, bestNum, bestDen = residual, numerator, denominator
		}
	}
	if bestNum == nil {
		return nil, nil, ErrSingularEquation
	}
	return bestNum, bestDen, nil
}

// realizeDiscreteIntegrators realizes N(q)/((1-q)^k a(q)), q = 1/z, keeping
// the k poles at z = 1 exact: multiplying out (1-q)^k a(q) moves them by
// roundoff amplified to eps^(1/k). The input drives a chain of k
// accumulators w(t+1) = w(t) + input, the last feeding the companion form of
// 1/a with states v_j = z^j φ. The output N_d(z)φ follows by shifting the
// state representation of φ through x ↦ Ax + Bu; u first appears at z^order φ.
func realizeDiscreteIntegrators(numerator, denominator []float64, integrators int, dt float64) (*System, error) {
	order, free := len(numerator)-1, len(denominator)-1
	a := mat.NewDense(order, order, nil)
	b := mat.NewDense(order, 1, nil)
	b.Set(0, 0, 1)
	for i := range integrators {
		a.Set(i, i, 1)
		if i > 0 {
			a.Set(i, i-1, 1)
		}
	}
	phi := integrators - 1
	if free > 0 {
		phi = integrators
		for j := range free - 1 {
			a.Set(integrators+j, integrators+j+1, 1)
		}
		last := order - 1
		a.Set(last, integrators-1, 1)
		for i := 1; i <= free; i++ {
			a.Set(last, order-i, -denominator[i])
		}
	}
	shift := make([]float64, order)
	shift[phi] = 1
	next := make([]float64, order)
	c := mat.NewDense(1, order, nil)
	raw := a.RawMatrix()
	feedthrough, input := 0.0, 0.0
	for power := 0; power <= order; power++ {
		coef := numerator[order-power]
		feedthrough += coef * input
		for j, v := range shift {
			c.Set(0, j, c.At(0, j)+coef*v)
		}
		input = dotVec(shift, b.RawMatrix().Data)
		clear(next)
		for i, v := range shift {
			if v == 0 {
				continue
			}
			row := raw.Data[i*raw.Stride : i*raw.Stride+order]
			for j, x := range row {
				next[j] += v * x
			}
		}
		shift, next = next, shift
	}
	return New(a, b, c, mat.NewDense(1, 1, []float64{feedthrough}), dt)
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
