package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// LeastSquaresFit describes an approximate frequency-response fit. Errors are
// measured on a validation grid distinct from the fitting grid. RMSRelativeError
// divides RMS absolute error by RMS source magnitude; MaxRelativeError divides
// maximum absolute error by maximum source magnitude. Both compare the
// fitted model with the source, each evaluated on its state-space realization
// (the pencil (sE−A)⁻¹ for a descriptor source sampled through it; see C2DFit).
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
// The source is sampled through its state-space realization; a descriptor
// model whose LU fold E⁻¹A is inaccurate (ill-conditioned full-rank E) is
// sampled through its pencil, (sE−A)⁻¹, for the fit and its quality.
// opts.FitOrder 0 ("auto", the MATLAB default) selects the source state order
// (after eliminating the algebraic states of a singular-E descriptor model);
// positive orders are accepted.
// The fit uses an equally weighted uniform grid with max(513,32*order+1) points
// and up to 12 denominator-reweighted real least-squares iterations followed
// by up to 8 response-error Gauss-Newton refinements. It selects
// the iterate with smallest frequency-response error. Stability is not imposed.
// Integrators, the poles at s = 0 of the minimal realization (null-space
// staircase with singular values at most 1e-10·‖A‖₁, capped by the number
// of poles componentwise rounding of A does not locate away from s = 0),
// retain their multiplicity and leading low-frequency residue as exact
// poles at z = 1;
// they require sufficient fit order and use an open midpoint frequency grid.
// Integer external delays are preserved; fractional and internal delays and
// improper descriptor models are rejected. A finite grid cannot certify intersample error
// or capture arbitrarily narrow resonances. No exact inverse is promised.
//
// A fit of a descriptor sampled through its pencil is not certified: equal
// counts of poles not stable (IsStable) of the fit and of the slow part of
// its pencil split do not make them the same poles. C2DFit returns it, with
// its quality, alongside an error wrapping ErrAccuracyUnverified, as C2D
// does. As guidance: more poles not stable than the slow part means a
// redundant pole of an order above the minimal one landed on or beyond
// z = −1, and reducing FitOrder is the remedy; fewer means an order below
// the minimal one dropped an unstable slow pole, and raising FitOrder is the
// remedy unless that pole lies above Nyquist, where a smaller dt is. Any
// other fit returns a nil error: its quality is that reported. Every other
// error comes with a nil model.
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
	out, fit, err := sys.leastSquaresFit(dt, opts.FitOrder, leastSquaresPencilSlow(sys))
	if errors.Is(err, errPencilFitUncertified) {
		return out, fit, fmt.Errorf("C2DFit: %v: %w", err, ErrAccuracyUnverified)
	}
	if err != nil {
		return nil, LeastSquaresFit{}, fmt.Errorf("C2DFit: %w", err)
	}
	return out, fit, nil
}

// errPencilFitUncertified is returned by leastSquaresFit, with the fit,
// for a descriptor sampled through its pencil; C2DFit and C2D map it to
// ErrAccuracyUnverified.
var errPencilFitUncertified = errors.New("least-squares fit through the descriptor pencil not certified")

// leastSquaresFit fits sys; slow is its leastSquaresPencilSlow, computed once
// by the caller (newC2DPlan gates the descriptor reduction on it). With slow
// non-nil a successful fit returns errPencilFitUncertified with the model.
func (sys *System) leastSquaresFit(dt float64, fitOrder int, slow *System) (*System, LeastSquaresFit, error) {
	var none LeastSquaresFit
	if sys.IsDiscrete() {
		return nil, none, ErrWrongDomain
	}
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, none, ErrInvalidSampleTime
	}
	if _, m, p := sys.Dims(); m != 1 || p != 1 {
		return nil, none, fmt.Errorf("least squares: sys is %d×%d, want SISO: %w", p, m, ErrNotSISO)
	}
	var pencil *System
	if slow != nil {
		core := *sys
		core.Delay, core.InputDelay, core.OutputDelay = nil, nil, nil
		pencil = &core
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
	source, detect := work, work
	if pencil != nil {
		source, detect = pencil, slow
	}
	maxInt := int(^uint(0) >> 1)
	if fitOrder > (maxInt-1)/32 {
		return nil, none, ErrInvalidOrder
	}
	integrators, residue, err := leastSquaresIntegratorLimit(detect, pencil == nil, dt)
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
	open := integrators > 0
	grid := func() []float64 {
		omega := make([]float64, samples)
		for k := range samples {
			omega[k] = discreteFitTheta(k, samples, open) / dt
		}
		return omega
	}
	target, err := leastSquaresSourceResponse(source, grid())
	if errors.Is(err, errLeastSquaresDCPole) && !open {
		open = true
		target, err = leastSquaresSourceResponse(source, grid())
	}
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
		numerator, denominator, err := fitDiscreteResponse(target, fitOrder, open)
		if err != nil {
			return nil, none, err
		}
		fitted := &TransferFunc{Num: [][][]float64{{numerator}}, Den: [][]float64{denominator}, Dt: dt}
		result, err := fitted.stateSpace()
		if err != nil {
			return nil, none, err
		}
		realized = result.Sys
	}
	rms, worst, err := leastSquaresFitQuality(source, realized, dt, samples)
	if err != nil {
		return nil, none, err
	}
	out, err := newDelayConversionPolicy(dt).applyDiscreteDelayFields(realized, sys)
	if err != nil {
		return nil, none, err
	}
	propagateNames(out, sys)
	out.StateName = nil
	unstable, err := out.nonStablePoleCount()
	if err != nil {
		return nil, none, err
	}
	fit := LeastSquaresFit{FitOrder: fitOrder, RMSRelativeError: rms, MaxRelativeError: worst, Stable: unstable == 0}
	if slow != nil {
		return out, fit, errPencilFitUncertified
	}
	return out, fit, nil
}

// continuousNonStablePoleCount is the number of poles of the explicit
// continuous sys that IsStable classifies as not stable. The slow part of
// the pencil split is T11⁻¹·S11 of the generalized Schur form, upper
// quasi-triangular, whose poles are those of its diagonal blocks. integrators
// is the multiplicity of the pole at s = 0 (leastSquaresIntegratorLimit): a
// k-fold chain spreads by roundoff to eps^(1/k)·‖A‖, beyond the IsStable
// tolerance, so the k poles nearest s = 0 count as not stable by rank.
func continuousNonStablePoleCount(sys *System, integrators int) (int, error) {
	var poles []complex128
	if !sys.IsDescriptor() && upperQuasiTriangular(sys.A) {
		n, _ := sys.A.Dims()
		raw := sys.A.RawMatrix()
		data := raw.Data
		if raw.Stride != n {
			data = make([]float64, n*n)
			for i := range n {
				copy(data[i*n:(i+1)*n], raw.Data[i*raw.Stride:])
			}
		}
		poles = schurEigenvaluesRaw(data, n)
	} else {
		var err error
		if poles, err = sys.Poles(); err != nil {
			return 0, err
		}
	}
	for k := range min(integrators, len(poles)) {
		nearest := k
		for i := k + 1; i < len(poles); i++ {
			if cmplx.Abs(poles[i]) < cmplx.Abs(poles[nearest]) {
				nearest = i
			}
		}
		poles[k], poles[nearest] = poles[nearest], poles[k]
	}
	count := min(integrators, len(poles))
	for _, p := range poles[count:] {
		if poleOnOrOutsideStabilityBoundary(p, true, poleStabilityTolerance(p)) {
			count++
		}
	}
	return count, nil
}

// leastSquaresPencilSlow returns, for a descriptor sys whose LU fold E⁻¹A
// of the delay-free core is inaccurate (foldInaccurateParts), the explicit
// slow part of that split, which holds any pole at s = 0; it is nil
// otherwise. When it is non-nil the least-squares fit samples the response
// of the pencil (sE−A)⁻¹ of that core, as MATLAB c2d fits the continuous
// response, instead of that of the fold, and counts the integrators on the
// slow part: the tolerance of leastSquaresIntegratorLimit scales with ‖A‖₁,
// which the fold inflates to the stiff pole (cond(E)), so a slow pole there
// passes for one. Coupling within rounding of the slow poles leaves them
// undetermined by the stored data: F=[-1,c,1;0,-2,1;0,0,-1], W(1,1,1e-13) at
// c ≥ 3e8 has them unstable under ±1 ulp of the pencil, so the fit is not
// right there on either split, and is returned with ErrAccuracyUnverified
// (the routed path).
func leastSquaresPencilSlow(sys *System) *System {
	if !sys.IsDescriptor() || sys.HasInternalDelay() {
		return nil
	}
	core := *sys
	core.Delay, core.InputDelay, core.OutputDelay = nil, nil, nil
	parts, ok := foldInaccurateParts(&core)
	if !ok {
		return nil
	}
	return parts[0]
}

func (sys *System) discretizeLeastSquares(dt float64, fitOrder int, slow *System) (*System, error) {
	out, _, err := sys.leastSquaresFit(dt, fitOrder, slow)
	return out, err
}

// fitDiscreteResponse fits target, sampled on the closed grid
// (discreteFitTheta) or, with open set, the open one.
func fitDiscreteResponse(target []complex128, order int, open bool) ([]float64, []float64, error) {
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
	grid := newDiscreteFitGrid(len(target), open)
	for range 12 {
		raw, b := design.RawMatrix(), rhs.RawMatrix()
		for k, value := range target {
			q := grid.q[k]
			powers[0] = 1
			for j := 1; j <= order; j++ {
				powers[j] = powers[j-1] * q
			}
			weight := 1 / math.Max(1e-8, cmplx.Abs(Poly(denominator).Eval(grid.invQ[k])))
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
		residual := discreteResponseResidual(target, grid, numerator, denominator)
		if residual < bestError {
			bestError, bestNum, bestDen = residual, numerator, denominator
		}
	}
	if bestNum == nil {
		return nil, nil, ErrSingularEquation
	}
	return refineDiscreteResponse(target, grid, design, rhs, &solver, bestNum, bestDen)
}

// discreteFitGrid is the fitting grid θ_k (discreteFitTheta) on the unit
// circle: z = e^{jθ}, q = e^{−jθ} and 1/q, which every reweighted solve and
// residual of fitDiscreteResponse reads.
type discreteFitGrid struct {
	z, q, invQ []complex128
}

// discreteFitTheta is θ_k of the closed grid πk/(samples−1), which samples
// DC, or of the open grid π(k+½)/samples, which does not. The fit holding
// integrators samples the open grid, as does a free fit whose source
// response is nonfinite at DC: FreqResponse takes the pole limit at s = 0
// beside a near pole the count resolves from zero, ±Inf although G(0) of
// the stored model is finite (NDFJC6).
func discreteFitTheta(k, samples int, open bool) float64 {
	if open {
		return math.Pi * (float64(k) + .5) / float64(samples)
	}
	return math.Pi * float64(k) / float64(samples-1)
}

func newDiscreteFitGrid(samples int, open bool) discreteFitGrid {
	points := make([]complex128, 3*samples)
	grid := discreteFitGrid{z: points[:samples], q: points[samples : 2*samples], invQ: points[2*samples:]}
	for k := range samples {
		theta := discreteFitTheta(k, samples, open)
		grid.z[k] = cmplx.Exp(complex(0, theta))
		grid.q[k] = cmplx.Exp(complex(0, -theta))
		grid.invQ[k] = 1 / grid.q[k]
	}
	return grid
}

func leastSquaresSourceResponse(source *System, omega []float64) ([]complex128, error) {
	response, err := source.FreqResponse(omega)
	if err != nil {
		return nil, err
	}
	for k, h := range response.Data {
		if cmplx.IsInf(h) || cmplx.IsNaN(h) {
			if omega[k] == 0 {
				return nil, errLeastSquaresDCPole
			}
			return nil, fmt.Errorf("least-squares: singular or nonfinite source frequency response: %w", ErrSingularTransform)
		}
	}
	return response.Data, nil
}

// errLeastSquaresDCPole is a source response nonfinite at DC.
var errLeastSquaresDCPole = fmt.Errorf("least-squares: singular or nonfinite source frequency response at DC: %w", ErrSingularTransform)

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
// of the minimal realization of source and lim s^k G(s)
// (integratorStaircaseLimit).
func leastSquaresIntegratorLimit(source *System, stored bool, dt float64) (int, float64, error) {
	minimal, err := source.MinimalRealization()
	if err != nil {
		return 0, 0, err
	}
	return integratorStaircaseLimit(minimal.Sys, stored, dt)
}

// integratorStaircaseLimit returns the multiplicity k of the pole at s = 0
// of the minimal SISO sys and lim s^k G(s). The staircase
// (integratorStaircase) takes the poles within rounding of s = 0; the
// coefficient of 1/s^q of G is then read for q from its step count down,
// and k is the first q whose coefficient exceeds its own rounding bound
// (integratorForm.coefficient). One within its bound is no integrator
// coefficient: to the precision it is computed the pole is not driven to
// the output, and its exact coefficient may be 0, as for a pole within
// rounding of s = 0 that the data leave uncontrollable or unobservable, or
// a slow pole of a stiff realization whose residue is cancellation of
// terms of the stiff pole's size (NDFJC6). k only shapes the least-squares
// fit, which holds the k poles at z = 1 and is judged by its own error, so a
// dropped integrator leaves a free pole to the fit.
func integratorStaircaseLimit(sys *System, stored bool, dt float64) (int, float64, error) {
	form, err := integratorStaircase(sys, stored, dt)
	if err != nil || form == nil {
		return 0, 0, err
	}
	for q := form.k; q >= 1; q-- {
		if limit, bound := form.coefficient(q); math.Abs(limit) > bound {
			return q, limit, nil
		}
	}
	return 0, 0, nil
}

// integratorForm is the staircase A = [N A12; 0 A22], b = [b1; b2],
// c = [c1; c2] of a minimal SISO realization, N k×k strictly upper
// triangular and A22 nonsingular, with y_j = A22^{-(j+1)}·b2 and the
// Laurent coefficients x_0 = b1 − A12·y_0, x_j = −A12·y_j of
// x(s) = b1 + A12(sI − A22)⁻¹b2 at s = 0, j < k.
type integratorForm struct {
	k, n                int
	a                   *mat.Dense
	b, c                *mat.VecDense
	normA, normB, normC float64
	lu                  mat.LU
	invA22              float64
	x, y                []*mat.VecDense
}

// integratorStaircase returns the staircase form of the minimal SISO sys
// with k the multiplicity of its pole at s = 0, or nil when k = 0.
// Eigenvalues of a k-fold Jordan chain spread by roundoff to eps^(1/k)·‖A‖,
// so the chain is found as a null-space staircase instead: each step
// deflates a singular value at most leastSquaresIntegratorTol·‖A‖₁ with a
// Householder similarity. With stored set, A is the model's own realization
// or its fold, and the staircase takes at most as many steps as A has poles
// within rounding of s = 0 or of z = 1 at dt (originUnresolvedPoles): the
// normwise tolerance is of the order of a stiff pole, so alone it takes a
// slow pole of a stiff graded realization, located to ε of itself and
// mapped well inside z = 1, for an integrator (KEEWB4). k so depends on
// dt: a resolved pole counts once e^{λ·dt} is within the existing IsStable
// band poleStabilityTolerance of z = 1. The slow part of a pencil split is
// not stored: it is quasi-triangular, its diagonal fixed exactly at the
// split's rounding, as a repeated integrator spread to ±1e-8 that
// componentwise rounding of the triangle would not move. The deflated
// singular values set to zero are not rounding: they are the move of the
// poles to s = 0 that defines the integrators.
func integratorStaircase(sys *System, stored bool, dt float64) (*integratorForm, error) {
	n, _ := sys.A.Dims()
	if n == 0 {
		return nil, nil
	}
	a := mat.DenseCopyOf(sys.A)
	b := mat.VecDenseCopyOf(sys.B.ColView(0))
	c := mat.VecDenseCopyOf(sys.C.RowView(0))
	f := &integratorForm{n: n, a: a, b: b, c: c, normA: mat.Norm(a, 2), normB: mat.Norm(b, 2), normC: mat.Norm(c, 2)}
	tol := leastSquaresIntegratorTol * mat.Norm(a, 1)
	raw := a.RawMatrix()
	var svd mat.SVD
	var v mat.Dense
	k, unresolved := 0, -1
	for ; k < n; k++ {
		m := n - k
		if !svd.Factorize(a.Slice(k, n, k, n), mat.SVDFullV) {
			return nil, ErrSingularEquation
		}
		if svd.Values(nil)[m-1] > tol {
			break
		}
		if stored && unresolved < 0 {
			var ok bool
			if unresolved, ok = originUnresolvedPoles(sys.A, n, dt); !ok {
				return nil, ErrSingularEquation
			}
		}
		if stored && k >= unresolved {
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
		return nil, nil
	}
	f.k = k
	f.x, f.y = make([]*mat.VecDense, k), make([]*mat.VecDense, k)
	f.x[0] = mat.VecDenseCopyOf(b.SliceVec(0, k))
	for j := 1; j < k; j++ {
		f.x[j] = mat.NewVecDense(k, nil)
	}
	if k == n {
		return f, nil
	}
	a22 := a.Slice(k, n, k, n)
	f.lu.Factorize(a22)
	f.invA22 = math.Sqrt(float64(n-k)) * f.lu.Cond() / mat.Norm(a22, 1)
	a12 := a.Slice(0, k, k, n)
	rhs := b.SliceVec(k, n)
	for j := range k {
		f.y[j] = mat.NewVecDense(n-k, nil)
		if err := f.lu.SolveVecTo(f.y[j], false, rhs); err != nil {
			return nil, err
		}
		rhs = f.y[j]
		var shift mat.VecDense
		shift.MulVec(a12, f.y[j])
		f.x[j].SubVec(f.x[j], &shift)
	}
	return f, nil
}

// coefficient returns the coefficient of 1/s^q, 1 ≤ q ≤ k, of G at s = 0,
// r_q = Σ_j c1ᵀN^(q−1+j)·x_j over j ≤ k−q, and its rounding bound: for
// q = k, lim s^k G(s) = c1ᵀN^(k−1)·x_0, and for q < k the limit
// lim s^q G(s) where the coefficients above q are zero.
//
// The bound is first order in the unit roundoff u. The k reflections
// return Q(A+ΔA)Qᵀ, Q(b+Δb), Q(c+Δc) with ‖ΔA‖ ≤ kγ‖A‖, ‖Δb‖ ≤ kγ‖b‖,
// ‖Δc‖ ≤ kγ‖c‖ (Higham, Lemma 19.3), and the LU solves and the products
// forming y_j, x_j and N^i·x_j are backward stable with γ_n. r_q is linear
// in b̃ as ℓ_qᵀb̃, ℓ_q = [u_0; −Σ_j A22^{-(j+1)T}A12ᵀu_j] with
// u_j = N^{(q−1+j)T}c1, and in c̃ as c̃ᵀ[v_q; 0], v_q = Σ_j N^(q−1+j)x_j,
// so Δb moves it by ‖ℓ_q‖‖Δb‖ and Δc by ‖Δc‖‖v_q‖. ΔA moves y_j by at most
// ‖ΔA‖·Σ_{i≤j} ‖A22⁻¹‖^(i+1)‖y_{j−i}‖ and x_j by ‖ΔA‖‖y_j‖ plus ‖A12‖ times
// that, and N^i by i‖ΔA‖‖N‖^(i−1). Summed with γ = γ_{(k+2)n} covering
// the reflections and the solves:
//
//	|Δr_q| ≤ γ·(‖ℓ_q‖‖b‖ + ‖c‖‖v_q‖ + ‖A‖·Σ_j (‖u_j‖·(‖y_j‖ + ‖A12‖·Σ_{i≤j} ‖A22⁻¹‖^(i+1)‖y_{j−i}‖)
//	          + (q−1+j)‖c1‖‖N‖^(q−2+j)‖x_j‖)),
//
// with ‖A22⁻¹‖₂ ≤ √(n−k)·‖A22⁻¹‖₁ from the LU condition estimate.
func (f *integratorForm) coefficient(q int) (float64, float64) {
	k, n := f.k, f.n
	nBlock := f.a.Slice(0, k, 0, k)
	c1 := f.c.SliceVec(0, k)
	normN, normC1 := mat.Norm(nBlock, 2), mat.Norm(c1, 2)
	last := k - q
	power := func(dst *mat.VecDense, src mat.Vector, p int, trans bool) {
		dst.CloneFromVec(src)
		var tmp mat.VecDense
		for range p {
			if trans {
				tmp.MulVec(nBlock.T(), dst)
			} else {
				tmp.MulVec(nBlock, dst)
			}
			dst.CloneFromVec(&tmp)
		}
	}
	var acc, v mat.VecDense
	acc.CloneFromVec(f.x[last])
	for j := last - 1; j >= 0; j-- {
		var tmp mat.VecDense
		tmp.MulVec(nBlock, &acc)
		acc.AddVec(f.x[j], &tmp)
	}
	power(&v, &acc, q-1, false)
	limit := mat.Dot(c1, &v)

	u := make([]*mat.VecDense, last+1)
	u[0] = new(mat.VecDense)
	power(u[0], c1, q-1, true)
	for j := 1; j <= last; j++ {
		u[j] = new(mat.VecDense)
		u[j].MulVec(nBlock.T(), u[j-1])
	}
	normL := mat.Norm(u[0], 2)
	terms := 0.0
	for j := 0; j <= last; j++ {
		if p := q - 1 + j; p > 0 {
			terms += float64(p) * normC1 * math.Pow(normN, float64(p-1)) * mat.Norm(f.x[j], 2)
		}
	}
	if k < n {
		a12 := f.a.Slice(0, k, k, n)
		normA12 := mat.Norm(a12, 2)
		var l2 mat.VecDense
		for j := last; j >= 0; j-- {
			var w mat.VecDense
			w.MulVec(a12.T(), u[j])
			if j < last {
				w.AddVec(&w, &l2)
			}
			if err := f.lu.SolveVecTo(&l2, true, &w); err != nil {
				return limit, math.Inf(1)
			}
		}
		normL = math.Hypot(normL, mat.Norm(&l2, 2))
		for j := 0; j <= last; j++ {
			moved := 0.0
			for i := 0; i <= j; i++ {
				moved += math.Pow(f.invA22, float64(i+1)) * mat.Norm(f.y[j-i], 2)
			}
			terms += mat.Norm(u[j], 2) * (mat.Norm(f.y[j], 2) + normA12*moved)
		}
	}
	g := float64((k+2)*n) * eps() / 2
	bound := g / (1 - g) * (normL*f.normB + f.normC*mat.Norm(&v, 2) + f.normA*terms)
	return limit, bound
}

// originUnresolvedPoles counts the eigenvalues λ of the n×n A that are
// integrators to a fit at dt: those of originPoleCluster, and any zero or
// with e^{λ·dt} within the IsStable boundary band poleStabilityTolerance
// of z = 1, where the discrete fit cannot place a free pole apart from an
// exact one. It reports false where the Schur factorization fails.
func originUnresolvedPoles(A *mat.Dense, n int, dt float64) (int, bool) {
	cluster, evals, ok := originPoleCluster(A, n)
	if !ok {
		return 0, false
	}
	count := cluster
	for _, p := range evals[cluster:] {
		re, im := real(p)*dt, imag(p)*dt
		half := math.Sin(im / 2)
		toOne := complex(math.Expm1(re)*math.Cos(im)-2*half*half, math.Exp(re)*math.Sin(im))
		nearOne := !cmplx.IsInf(toOne) && cmplx.Abs(toOne) <= poleStabilityTolerance(1+toOne)
		if p == 0 || nearOne {
			count++
		}
	}
	return count, true
}

// originPoleCluster returns the size of the largest cluster of the m
// eigenvalues of the n×n A smallest in modulus that componentwise
// rounding of A can carry to s = 0 together, with the eigenvalues of A
// ordered by modulus where the reordering succeeded. Closeness to s = 0 on
// the scale of ‖A‖ is not enough: a slow pole of a stiff graded
// realization is located to ε of itself, and rounding of A does not move
// it to s = 0 (KEEWB4, NDFJC6).
//
// With A = Z·T·Zᵀ in real Schur form ordered by modulus, the m leading
// eigenvalues are those of T11, with invariant bases X = Z1 and
// Y = Z1 + Z2·X12ᵀ, YᵀX = I, where T11·X12 − X12·T22 = T12. Rounding
// |E| ≤ γ_n|A| of A, whose entries are n-term sums (a similarity, a
// staircase or Reduce), moves them, to first order, to those of T11 + F,
// F = YᵀEX, ‖F‖₂ ≤ δ = γ_n‖|Y|ᵀ|A||X|‖_F. If rounding carries them all to zero,
// T11 + F is nilpotent, so every coefficient e_i of the characteristic
// polynomial of T11, the i-th elementary symmetric function of its
// eigenvalues, is within the first-order move of a sum of C(m,i) principal
// minors of order i, |e_i| ≤ C(m,i)·i·δ·‖T11‖₂^(i−1); a cluster that fails
// this is resolved from zero. For m = 1 it is the pole's own bound
// γ_n·|y|ᵀ|A||x|/|yᵀx|; a k-fold chain, whose computed poles spread to
// δ^{1/k}, passes as a cluster although each member exceeds its
// first-order bound, while poles each resolved from zero do not pass
// together. A cluster the Sylvester equation cannot separate from the
// rest does not pass, nor does one past a block the reordering cannot
// move. It reports false where the Schur factorization fails.
func originPoleCluster(A *mat.Dense, n int) (int, []complex128, bool) {
	return originPoleClusterScan(A, n, false)
}

// originPoleReached reports whether originPoleCluster is positive, or an
// eigenvalue is exactly 0, stopping at the first cluster that passes. ok
// is false where the Schur factorization fails.
func originPoleReached(A *mat.Dense, n int) (reached, ok bool) {
	cluster, evals, ok := originPoleClusterScan(A, n, true)
	return cluster > 0 || slices.Contains(evals, 0), ok
}

func originPoleClusterScan(A *mat.Dense, n int, first bool) (int, []complex128, bool) {
	t, z, ok := realSchur(A)
	if !ok {
		return 0, nil, false
	}
	work := make([]float64, n)
	ordered := 0
	for ordered < n {
		evals := schurEigenvaluesRaw(t, n)
		best := ordered
		for i := ordered; i < n; i += schurBlockSize(t, n, i) {
			if cmplx.Abs(evals[i]) < cmplx.Abs(evals[best]) {
				best = i
			}
		}
		if best != ordered {
			if _, _, ok := impl.Dtrexc(lapack.UpdateSchur, n, t, n, z, n, best, ordered, work); !ok {
				break
			}
		}
		ordered += schurBlockSize(t, n, ordered)
	}
	evals := schurEigenvaluesRaw(t, n)
	ar := A.RawMatrix()
	absAX := make([]float64, n*n)
	haveAX := 0
	absY := make([]float64, n*n)
	g := float64(n) * eps() / 2
	coef := make([]complex128, n+1)
	cluster := 0
	for m := 1; m <= ordered; m++ {
		if splitsSchurBlock(t, n, m) {
			continue
		}
		var x []float64
		if m < n {
			var err error
			if x, err = separateSchurBlocks(t, n, m, "integrator cluster"); err != nil {
				continue
			}
		}
		for ; haveAX < m; haveAX++ {
			for i := range n {
				s := 0.0
				for l, v := range ar.Data[i*ar.Stride : i*ar.Stride+n] {
					s += math.Abs(v) * math.Abs(z[l*n+haveAX])
				}
				absAX[haveAX*n+i] = s
			}
		}
		for i := range m {
			for l := range n {
				y := z[l*n+i]
				for q := range n - m {
					y -= z[l*n+m+q] * x[i*(n-m)+q]
				}
				absY[i*n+l] = math.Abs(y)
			}
		}
		yax := 0.0
		for j := range m {
			for i := range m {
				s := 0.0
				for l := range n {
					s += absY[i*n+l] * absAX[j*n+l]
				}
				yax = math.Hypot(yax, s)
			}
		}
		delta := g / (1 - g) * yax
		normT := -1.0
		clear(coef)
		coef[0] = 1
		for i, p := range evals[:m] {
			for d := i + 1; d >= 1; d-- {
				coef[d] -= p * coef[d-1]
			}
		}
		pass, binom := true, 1.0
		for i := 1; i <= m && pass; i++ {
			binom = binom * float64(m-i+1) / float64(i)
			if i == 2 {
				var t11 mat.Dense
				t11.CloneFrom(mat.NewDense(n, n, t).Slice(0, m, 0, m))
				normT = mat.Norm(&t11, 2)
			}
			pass = cmplx.Abs(coef[i]) <= binom*float64(i)*delta*math.Pow(normT, float64(i-1))
		}
		if pass {
			cluster = m
			if first {
				break
			}
		}
	}
	return cluster, evals, true
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
	return newDerived(a, b, c, mat.NewDense(1, 1, []float64{feedthrough}), dt)
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

// refineDiscreteResponse reuses the design, rhs and solver of the
// fitDiscreteResponse solves, whose shapes it shares.
func refineDiscreteResponse(target []complex128, grid discreteFitGrid, design, rhs *mat.Dense, solver *frequencyFitSolver, numerator, denominator []float64) ([]float64, []float64, error) {
	order := len(denominator) - 1
	best := discreteResponseResidual(target, grid, numerator, denominator)
	powers := make([]complex128, order+1)
	for range 8 {
		raw, b := design.RawMatrix(), rhs.RawMatrix()
		for k, h := range target {
			q := grid.q[k]
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
			residual := discreteResponseResidual(target, grid, trialNum, trialDen)
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

func discreteResponseResidual(target []complex128, grid discreteFitGrid, num, den []float64) float64 {
	residual := 0.0
	for k, h := range target {
		z := grid.z[k]
		prediction := Poly(num).Eval(z) / Poly(den).Eval(z)
		residual = math.Hypot(residual, cmplx.Abs(prediction-h))
	}
	return residual
}
