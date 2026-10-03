package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/mat"
)

func (sys *System) Discretize(dt float64) (*System, error) {
	if !sys.HasDelay() {
		return sys.discretizeTustin(dt, 0)
	}
	return sys.DiscretizeWithOpts(dt, C2DOptions{Method: C2DMethodTustin})
}

func (sys *System) discretizeTustin(dt, prewarp float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("Discretize: system already discrete: %w", ErrWrongDomain)
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

func (sys *System) Undiscretize() (*System, error) {
	return sys.undiscretizeTustin(0)
}

func (sys *System) undiscretizeTustin(prewarp float64) (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("Undiscretize: system already continuous: %w", ErrWrongDomain)
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

type C2DOptions struct {
	Method           C2DMethod
	ThiranOrder      int
	DelayModeling    C2DDelayModeling
	PrewarpFrequency float64
	FitOrder         int
}

type C2DMethod string

const (
	C2DMethodZOH          C2DMethod = "zoh"
	C2DMethodTustin       C2DMethod = "tustin"
	C2DMethodFOH          C2DMethod = "foh"
	C2DMethodImpulse      C2DMethod = "impulse"
	C2DMethodMatched      C2DMethod = "matched"
	C2DMethodLeastSquares C2DMethod = "least-squares"
)

type C2DDelayModeling string

const (
	C2DDelayModelingState    C2DDelayModeling = "state"
	C2DDelayModelingInternal C2DDelayModeling = "delay"
	C2DDelayModelingDelay    C2DDelayModeling = C2DDelayModelingInternal
)

func (sys *System) DiscretizeWithOpts(dt float64, opts C2DOptions) (*System, error) {
	plan, err := newC2DPlan(sys, dt, opts)
	if err != nil {
		return nil, err
	}
	return plan.run()
}

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

	if contInputDelay != nil {
		bank, err := buildContinuousDelayBank(contInputDelay, m, dt, thiranOrder)
		if err != nil {
			return nil, err
		}
		if bank != nil {
			disc, err = Series(bank, disc)
			if err != nil {
				return nil, err
			}
			disc.InputDelay = nil
		}
	}

	if contOutputDelay != nil {
		bank, err := buildContinuousDelayBank(contOutputDelay, p, dt, thiranOrder)
		if err != nil {
			return nil, err
		}
		if bank != nil {
			disc, err = Series(disc, bank)
			if err != nil {
				return nil, err
			}
			disc.OutputDelay = nil
		}
	}

	return disc, nil
}

func (sys *System) DiscretizeZOH(dt float64) (*System, error) {
	if !sys.HasDelay() {
		return sys.discretizeZOH(dt)
	}
	return sys.DiscretizeWithOpts(dt, C2DOptions{Method: C2DMethodZOH})
}

func (sys *System) discretizeZOH(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeZOH: system already discrete: %w", ErrWrongDomain)
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

func (sys *System) DiscretizeImpulse(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, ErrWrongDomain
	}
	if err := validateConversionSampleTime(dt); err != nil {
		return nil, err
	}
	if !sys.HasDelay() {
		return sys.discretizeImpulseParity(dt)
	}
	return sys.DiscretizeWithOpts(dt, C2DOptions{Method: C2DMethodImpulse})
}

func (sys *System) DiscretizeFOH(dt float64) (*System, error) {
	if !sys.HasDelay() {
		return sys.discretizeModifiedFOH(dt)
	}
	return sys.DiscretizeWithOpts(dt, C2DOptions{Method: C2DMethodFOH})
}

func (sys *System) DiscretizeMatched(dt float64) (*System, error) {
	if !sys.HasDelay() {
		return sys.discretizeMatched(dt)
	}
	return sys.DiscretizeWithOpts(dt, C2DOptions{Method: C2DMethodMatched})
}

// D2C converts a discrete-time model using ZOH, Tustin, modified FOH, or
// matched pole-zero assumptions. An empty method selects ZOH.
// Delay fields are converted to seconds using the original sample time.
func (sys *System) D2C(method C2DMethod) (*System, error) {
	return sys.D2CWithOpts(D2COptions{Method: method})
}

// D2CWithOpts converts a discrete-time model with optional Tustin prewarping.
func (sys *System) D2CWithOpts(opts D2COptions) (*System, error) {
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

// D2D resamples a discrete-time model using ZOH (the default) or Tustin.
// Other methods are rejected, including on same-rate requests.
func (sys *System) D2D(newDt float64, opts C2DOptions) (*System, error) {
	plan, err := newD2DPlan(sys, newDt, opts)
	if err != nil {
		return nil, err
	}
	return plan.run()
}
