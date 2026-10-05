package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

// C2D converts a continuous-time model to discrete time with sample time
// dt, as MATLAB sysd = c2d(sysc,Ts,opts); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.c2d.html. The zero
// C2DOptions selects zero-order hold, the MATLAB default; MATLAB
// c2d(sysc,Ts,method) is C2D(dt, C2DOptions{Method: method}). Input, output
// and I/O delays become sample delays (fractional ones per opts), internal
// delays stay internal, and a singular-E descriptor model is first reduced as
// MATLAB dss2ss does. An already discrete model returns ErrWrongDomain;
// invalid options return ErrInvalidConversionOptions.
func (sys *System) C2D(dt float64, opts C2DOptions) (*System, error) {
	if sys == nil {
		return nil, fmt.Errorf("C2D: system is nil: %w", ErrInvalidArgument)
	}
	if !sys.HasDelay() && !sys.IsDescriptor() && opts.Method != C2DMethodLeastSquares {
		normalized, err := normalizeC2DOptions(dt, opts)
		if err != nil {
			return nil, fmt.Errorf("C2D: %w", err)
		}
		switch normalized.Method {
		case C2DMethodZOH:
			return sys.discretizeZOH(dt)
		case C2DMethodTustin:
			return sys.discretizeTustin(dt, normalized.PrewarpFrequency)
		case C2DMethodFOH:
			return sys.discretizeModifiedFOH(dt)
		case C2DMethodImpulse:
			if sys.IsDiscrete() {
				return nil, fmt.Errorf("C2D: system already discrete: %w", ErrWrongDomain)
			}
			return sys.discretizeImpulseParity(dt)
		case C2DMethodMatched:
			return sys.discretizeMatched(dt)
		}
	}
	plan, err := newC2DPlan(sys, dt, opts)
	if err != nil {
		return nil, err
	}
	return plan.run()
}

func (sys *System) discretizeTustin(dt, prewarp float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("C2D: system already discrete: %w", ErrWrongDomain)
	}
	beta, err := tustinBeta(dt, prewarp)
	if err != nil {
		return nil, err
	}
	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: C2DMethodTustin, PrewarpFrequency: prewarp})
	}
	out, err := bilinear(sys, -beta, -1, 1, beta)
	if err != nil {
		return nil, err
	}
	out.Dt = dt
	policy := newDelayConversionPolicy(dt, 0, 0)
	if out, err = policy.applyDiscreteDelayFields(out, sys); err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	return out, nil
}

func (sys *System) undiscretizeTustin(prewarp float64) (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("D2C: system already continuous: %w", ErrWrongDomain)
	}
	beta, err := tustinBeta(sys.Dt, prewarp)
	if err != nil {
		return nil, err
	}
	if sys.HasInternalDelay() {
		return undiscretizeInternalTustin(sys, beta)
	}
	out, err := bilinear(sys, 1, beta, 1, beta)
	if err != nil {
		return nil, err
	}
	out.Dt = 0
	policy := newDelayConversionPolicy(sys.Dt, 0, 0)
	policy.applyContinuousDelayFields(out, sys)
	propagateNames(out, sys)
	return out, nil
}

func bilinear(sys *System, palpha, pbeta, alpha, beta float64) (*System, error) {
	n, m, p := sys.Dims()

	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{
			A: newDense(0, 0),
			B: denseCopy(sys.B),
			C: denseCopy(sys.C),
			D: D,
		}
		propagateNames(out, sys)
		return out, nil
	}

	A := mat.NewDense(n, n, nil)
	A.Copy(sys.A)

	B := denseCopy(sys.B)
	C := denseCopy(sys.C)

	aRaw := A.RawMatrix()
	for i := range n {
		aRaw.Data[i*aRaw.Stride+i] += palpha
	}

	var lu mat.LU
	lu.Factorize(A)
	if luNearSingular(&lu) {
		return nil, fmt.Errorf("bilinear: (palpha·I + A) is singular: %w", ErrSingularTransform)
	}

	if m > 0 {
		err := lu.SolveTo(B, false, B)
		if err != nil {
			return nil, fmt.Errorf("bilinear: LU solve for B failed: %w", ErrSingularTransform)
		}
	}

	if p > 0 && m > 0 {
		D.Mul(C, B)
		D.Scale(-1, D)
		D.Add(D, sys.D)
	}

	twoAB := 2.0 * alpha * beta
	if math.IsInf(twoAB, 0) {
		return nil, fmt.Errorf("bilinear: 2*alpha*beta overflows: %w", ErrOverflow)
	}

	scaleB, scaleC := beta, 2.0
	if palpha < 0 {
		scaleB, scaleC = -2, -beta
	}
	if m > 0 {
		B.Scale(scaleB, B)
	}

	Ainv := mat.NewDense(n, n, nil)
	eye := mat.NewDense(n, n, nil)
	eyeRaw := eye.RawMatrix()
	for i := range n {
		eyeRaw.Data[i*eyeRaw.Stride+i] = 1
	}
	err := lu.SolveTo(Ainv, false, eye)
	if err != nil {
		return nil, fmt.Errorf("bilinear: LU inverse failed: %w", ErrSingularTransform)
	}

	if p > 0 {
		C.Mul(C, Ainv)
		C.Scale(scaleC, C)
	}
	Ainv.Scale(-twoAB, Ainv)
	ainvRaw := Ainv.RawMatrix()
	for i := range n {
		ainvRaw.Data[i*ainvRaw.Stride+i] += pbeta
	}

	if math.IsInf(denseNorm(Ainv), 0) || math.IsNaN(denseNorm(Ainv)) {
		return nil, fmt.Errorf("bilinear: result contains Inf/NaN: %w", ErrOverflow)
	}

	out := &System{A: Ainv, B: B, C: C, D: D}
	propagateNames(out, sys)
	return out, nil
}

// C2DOptions mirrors MATLAB c2dOptions; see
// https://www.mathworks.com/help/control/ref/c2doptions.html. Zero values
// select the MATLAB defaults: Method "" is zero-order hold, ThiranOrder 0
// rounds fractional delays to the nearest sample (tustin and matched only),
// DelayModeling "" models extra delays as internal delays, PrewarpFrequency 0
// (rad/s, tustin only, below Nyquist) disables prewarping, and FitOrder 0
// ("auto") fits least-squares models of the source order.
type C2DOptions struct {
	Method           C2DMethod
	ThiranOrder      int
	DelayModeling    C2DDelayModeling
	PrewarpFrequency float64
	FitOrder         int
}

// C2DMethod names a continuous/discrete conversion method, as the MATLAB
// c2d, d2c and d2d method strings. The empty method selects C2DMethodZOH.
type C2DMethod string

const (
	// C2DMethodZOH assumes piecewise-constant inputs (zero-order hold).
	C2DMethodZOH C2DMethod = "zoh"
	// C2DMethodTustin is the bilinear approximation, optionally prewarped.
	C2DMethodTustin C2DMethod = "tustin"
	// C2DMethodFOH assumes piecewise-linear inputs (triangle approximation).
	C2DMethodFOH C2DMethod = "foh"
	// C2DMethodImpulse is impulse-invariant discretization (c2d only).
	C2DMethodImpulse C2DMethod = "impulse"
	// C2DMethodMatched is zero-pole matching (SISO only).
	C2DMethodMatched C2DMethod = "matched"
	// C2DMethodLeastSquares fits the frequency response up to Nyquist (c2d,
	// SISO only).
	C2DMethodLeastSquares C2DMethod = "least-squares"
)

// C2DDelayModeling selects how C2D models delays that become extra
// discrete delays, as the MATLAB c2dOptions DelayModeling option. The empty
// value selects C2DDelayModelingInternal, the MATLAB default.
type C2DDelayModeling string

const (
	// C2DDelayModelingState models extra delays as additional states.
	C2DDelayModelingState C2DDelayModeling = "state"
	// C2DDelayModelingInternal models extra delays as internal delays.
	C2DDelayModelingInternal C2DDelayModeling = "delay"
)

func mergeDelays(existing, decomposed []float64) []float64 {
	if decomposed == nil {
		return existing
	}
	allZero := true
	for _, v := range decomposed {
		if v != 0 {
			allZero = false
			break
		}
	}
	if allZero {
		return existing
	}
	n := len(decomposed)
	out := make([]float64, n)
	copy(out, decomposed)
	for i := range existing {
		if i < n {
			out[i] += existing[i]
		}
	}
	return out
}

func convertSliceDelayToDiscrete(delay []float64, dt float64, thiranOrder int) ([]float64, error) {
	if delay == nil {
		return nil, nil
	}
	out := make([]float64, len(delay))
	for i, tau := range delay {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		rounded := math.Round(samples)
		if math.Abs(samples-rounded) < 1e-9 {
			out[i] = rounded
		} else if thiranOrder > 0 {
			out[i] = 0
		} else {
			return nil, fmt.Errorf("delay[%d]=%g not integer multiple of dt=%g: %w",
				i, tau, dt, ErrFractionalDelay)
		}
	}
	return out, nil
}

func convertSliceDelayToContinuous(delay []float64, dt float64) []float64 {
	if delay == nil {
		return nil
	}
	out := make([]float64, len(delay))
	for i, d := range delay {
		out[i] = d * dt
	}
	return out
}

func absorbFractionalDelays(disc *System, contInputDelay, contOutputDelay []float64, dt float64, thiranOrder int) (*System, error) {
	_, m, p := disc.Dims()

	if samples := continuousToSampleDelay(contInputDelay, dt); hasFractionalSampleDelay(samples) {
		bank, err := buildDiscreteSampleDelayBank(samples, m, dt, thiranOrder)
		if err != nil {
			return nil, err
		}
		disc, err = Series(bank, disc)
		if err != nil {
			return nil, err
		}
		disc.InputDelay = nil
	}

	if samples := continuousToSampleDelay(contOutputDelay, dt); hasFractionalSampleDelay(samples) {
		bank, err := buildDiscreteSampleDelayBank(samples, p, dt, thiranOrder)
		if err != nil {
			return nil, err
		}
		disc, err = Series(disc, bank)
		if err != nil {
			return nil, err
		}
		disc.OutputDelay = nil
	}

	return disc, nil
}

func (sys *System) discretizeZOH(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("C2D: system already discrete: %w", ErrWrongDomain)
	}
	if err := validateConversionSampleTime(dt); err != nil {
		return nil, err
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: C2DMethodZOH})
	}

	n, m, _ := sys.Dims()

	D := denseCopy(sys.D)
	C := denseCopy(sys.C)

	if n == 0 {
		out := &System{
			A:  newDense(0, 0),
			B:  denseCopy(sys.B),
			C:  C,
			D:  D,
			Dt: dt,
		}
		propagateNames(out, sys)
		return newDelayConversionPolicy(dt, 0, 0).applyDiscreteDelayFields(out, sys)
	}

	nm := n + m
	if m == 0 {
		var eA mat.Dense
		Adt := mat.NewDense(n, n, nil)
		Adt.Scale(dt, sys.A)
		eA.Exp(Adt)
		Ad := mat.NewDense(n, n, nil)
		Ad.Copy(&eA)
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		propagateNames(out, sys)
		return newDelayConversionPolicy(dt, 0, 0).applyDiscreteDelayFields(out, sys)
	}

	M := mat.NewDense(nm, nm, nil)
	mRaw := M.RawMatrix()
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	for i := range n {
		mRow := mRaw.Data[i*mRaw.Stride:]
		aRow := aRaw.Data[i*aRaw.Stride : i*aRaw.Stride+n]
		for j, v := range aRow {
			mRow[j] = v * dt
		}
		bRow := bRaw.Data[i*bRaw.Stride : i*bRaw.Stride+m]
		for j, v := range bRow {
			mRow[n+j] = v * dt
		}
	}

	var eM mat.Dense
	eM.Exp(M)

	emRaw := eM.RawMatrix()
	adData := make([]float64, n*n)
	bdData := make([]float64, n*m)
	for i := range n {
		copy(adData[i*n:], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		copy(bdData[i*m:], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
	}
	Ad := mat.NewDense(n, n, adData)
	Bd := mat.NewDense(n, m, bdData)

	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	policy := newDelayConversionPolicy(dt, 0, 0)
	out, err := policy.applyDiscreteDelayFields(out, sys)
	if err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	return out, nil
}

func discretizeWithInternalDelay(sys *System, dt float64, opts C2DOptions) (*System, error) {
	return discretizeInternalModel(sys, dt, opts)
}

func isStrictlyUpperTriangular(m *mat.Dense) bool {
	if m == nil {
		return true
	}
	raw := m.RawMatrix()
	for i := 0; i < raw.Rows; i++ {
		bound := min(raw.Cols-1, i)
		row := raw.Data[i*raw.Stride:]
		for j := 0; j <= bound; j++ {
			if row[j] != 0 {
				return false
			}
		}
	}
	return true
}

// D2COptions mirrors MATLAB d2cOptions. The zero Method selects zero-order
// hold, the MATLAB default; PrewarpFrequency is in rad/s, applies only to
// Tustin, and must be below Nyquist.
type D2COptions struct {
	Method           C2DMethod
	PrewarpFrequency float64
}

// D2C converts a discrete-time model to continuous time, as MATLAB
// sysc = d2c(sysd,opts); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.d2c.html.
// Methods are zoh (the zero-value default), tustin, foh and matched.
// Delays of k samples become delays of k·Ts seconds; internal delays stay
// internal, inverting C2D, except that matched rejects them as MATLAB does.
// An already continuous model returns ErrWrongDomain.
func (sys *System) D2C(opts D2COptions) (*System, error) {
	if sys == nil {
		return nil, fmt.Errorf("D2C: system is nil: %w", ErrInvalidArgument)
	}
	plan, err := newD2CPlan(sys, opts)
	if err != nil {
		return nil, err
	}
	return plan.run()
}

func (sys *System) d2cZOH() (*System, error) {
	return sys.d2cZOHRobust()
}

func d2cPropagateDelays(out, sys *System, dt float64) {
	policy := newDelayConversionPolicy(dt, 0, 0)
	policy.applyContinuousDelayFields(out, sys)
}

// D2DOptions mirrors MATLAB d2dOptions. The zero Method selects zero-order
// hold, the MATLAB default; PrewarpFrequency is in rad/s and applies only to
// Tustin.
type D2DOptions struct {
	Method           C2DMethod
	PrewarpFrequency float64
}

// D2D resamples a discrete-time model to sample time dt, as MATLAB
// sys1 = d2d(sys,Ts,opts); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.d2d.html. It
// converts with D2C and back with C2D using opts.Method, which must be zoh
// (the zero-value default) or tustin; a same-rate request returns a copy.
func (sys *System) D2D(dt float64, opts D2DOptions) (*System, error) {
	if sys == nil {
		return nil, fmt.Errorf("D2D: system is nil: %w", ErrInvalidArgument)
	}
	plan, err := newD2DPlan(sys, dt, C2DOptions{Method: opts.Method, PrewarpFrequency: opts.PrewarpFrequency})
	if err != nil {
		return nil, err
	}
	return plan.run()
}
