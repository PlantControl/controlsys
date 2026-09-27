package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

func (sys *System) Discretize(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("Discretize: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: C2DMethodTustin})
	}

	beta := 2.0 / dt
	if math.IsInf(beta, 0) {
		return nil, fmt.Errorf("Discretize: dt too small, 2/dt overflows: %w", ErrOverflow)
	}
	out, err := bilinear(sys, -beta, -1.0, 1.0, beta)
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
	if sys.IsContinuous() {
		return nil, fmt.Errorf("Undiscretize: system already continuous: %w", ErrWrongDomain)
	}
	beta := 2.0 / sys.Dt
	if math.IsInf(beta, 0) {
		return nil, fmt.Errorf("Undiscretize: dt too small, 2/dt overflows: %w", ErrOverflow)
	}
	out, err := bilinear(sys, 1.0, beta, 1.0, beta)
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
	scale := math.Sqrt(math.Abs(twoAB))
	if palpha < 0 {
		scale = -scale
	}

	if m > 0 {
		B.Scale(scale, B)
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
		C.Scale(scale, C)
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
	Method        C2DMethod
	ThiranOrder   int
	DelayModeling C2DDelayModeling
}

type C2DMethod string

const (
	C2DMethodZOH     C2DMethod = "zoh"
	C2DMethodTustin  C2DMethod = "tustin"
	C2DMethodFOH     C2DMethod = "foh"
	C2DMethodImpulse C2DMethod = "impulse"
	C2DMethodMatched C2DMethod = "matched"
)

type C2DDelayModeling string

const (
	C2DDelayModelingState    C2DDelayModeling = "state"
	C2DDelayModelingInternal C2DDelayModeling = "internal"
)

func (sys *System) DiscretizeWithOpts(dt float64, opts C2DOptions) (*System, error) {
	plan, err := newC2DPlan(sys, dt, opts)
	if err != nil {
		return nil, err
	}
	return plan.run()
}

func discretizeDelaysAsInternal(disc *System, contInputDelay, contOutputDelay []float64, dt float64) (*System, error) {
	n, m, p := disc.Dims()

	type fracEntry struct {
		isInput bool
		idx     int
		frac    float64
	}
	var fracs []fracEntry

	inputDelayDisc := make([]float64, len(contInputDelay))
	for j, tau := range contInputDelay {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		fracPart := samples - math.Floor(samples)
		if fracPart < 1e-9 || fracPart > 1-1e-9 {
			inputDelayDisc[j] = math.Round(samples)
		} else {
			inputDelayDisc[j] = math.Floor(samples)
			fracs = append(fracs, fracEntry{isInput: true, idx: j, frac: fracPart})
		}
	}

	outputDelayDisc := make([]float64, len(contOutputDelay))
	for i, tau := range contOutputDelay {
		if tau == 0 {
			continue
		}
		samples := tau / dt
		fracPart := samples - math.Floor(samples)
		if fracPart < 1e-9 || fracPart > 1-1e-9 {
			outputDelayDisc[i] = math.Round(samples)
		} else {
			outputDelayDisc[i] = math.Floor(samples)
			fracs = append(fracs, fracEntry{isInput: false, idx: i, frac: fracPart})
		}
	}

	if len(contInputDelay) > 0 {
		disc.InputDelay = inputDelayDisc
	}
	if len(contOutputDelay) > 0 {
		disc.OutputDelay = outputDelayDisc
	}

	if len(fracs) == 0 {
		return disc, nil
	}

	N := len(fracs)
	internalDelay := make([]float64, N)
	b2Data := make([]float64, n*N)
	c2Data := make([]float64, N*n)
	d12Data := make([]float64, p*N)
	d21Data := make([]float64, N*m)
	d22Data := make([]float64, N*N)

	bRaw := disc.B.RawMatrix()
	cRaw := disc.C.RawMatrix()
	dRaw := disc.D.RawMatrix()

	newBData := make([]float64, n*m)
	for i := range n {
		copy(newBData[i*m:i*m+m], bRaw.Data[i*bRaw.Stride:i*bRaw.Stride+m])
	}
	newCData := make([]float64, p*n)
	for i := range p {
		copy(newCData[i*n:i*n+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
	}
	newDData := make([]float64, p*m)
	for i := range p {
		copy(newDData[i*m:i*m+m], dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m])
	}

	for k, f := range fracs {
		internalDelay[k] = f.frac
		if f.isInput {
			j := f.idx
			d21Data[k*m+j] = 1
			for i := range n {
				b2Data[i*N+k] = newBData[i*m+j]
				newBData[i*m+j] = 0
			}
			for i := range p {
				d12Data[i*N+k] = newDData[i*m+j]
				newDData[i*m+j] = 0
			}
		} else {
			i := f.idx
			for col := range n {
				c2Data[k*n+col] = newCData[i*n+col]
				newCData[i*n+col] = 0
			}
			for col := range m {
				d21Data[k*m+col] = newDData[i*m+col]
				newDData[i*m+col] = 0
			}
			d12Data[i*N+k] = 1
		}
	}

	if n > 0 && m > 0 {
		disc.B = mat.NewDense(n, m, newBData)
	}
	if p > 0 && n > 0 {
		disc.C = mat.NewDense(p, n, newCData)
	}
	if p > 0 && m > 0 {
		disc.D = mat.NewDense(p, m, newDData)
	}

	disc.LFT = &LFTDelay{
		Tau: internalDelay,
		B2:  mat.NewDense(n, N, b2Data),
		C2:  mat.NewDense(N, n, c2Data),
		D12: mat.NewDense(p, N, d12Data),
		D21: mat.NewDense(N, m, d21Data),
		D22: mat.NewDense(N, N, d22Data),
	}

	return disc, nil
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
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeZOH: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
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
		return out, nil
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
		return out, nil
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
	if sys.LFT != nil && !isStrictlyUpperTriangular(sys.LFT.D22) {
		return nil, fmt.Errorf("DiscretizeWithOpts: ZOH not supported for GLTI with non-upper-triangular D22: %w", ErrAlgebraicLoop)
	}

	method := opts.Method
	if method == "" {
		method = C2DMethodZOH
	}

	n, _, _ := sys.Dims()
	N := sys.internalDelayCount()

	policy := newDelayConversionPolicy(dt, 0, 0)
	discTau, err := policy.convertInternalTauToDiscrete(sys.LFT.Tau)
	if err != nil {
		return nil, err
	}

	var disc *System
	var Bd2 *mat.Dense

	switch method {
	case C2DMethodZOH:
		disc, Bd2, err = discretizeZOHAugmented(sys, dt)
	case C2DMethodTustin:
		disc, Bd2, err = discretizeTustinAugmented(sys, dt)
	case C2DMethodFOH:
		disc, Bd2, err = discretizeFOHAugmented(sys, dt)
	case C2DMethodImpulse:
		disc, Bd2, err = discretizeImpulseAugmented(sys, dt)
	default:
		return nil, fmt.Errorf("DiscretizeWithOpts: unknown method %q", method)
	}
	if err != nil {
		return nil, err
	}

	var b2Out *mat.Dense
	if n > 0 && N > 0 {
		b2Out = Bd2
	} else {
		b2Out = mat.DenseCopyOf(sys.LFT.B2)
	}
	disc.LFT = &LFTDelay{
		Tau: discTau,
		B2:  b2Out,
		C2:  mat.DenseCopyOf(sys.LFT.C2),
		D12: mat.DenseCopyOf(sys.LFT.D12),
		D21: mat.DenseCopyOf(sys.LFT.D21),
		D22: mat.DenseCopyOf(sys.LFT.D22),
	}

	if disc, err = policy.applyDiscreteDelayFields(disc, sys); err != nil {
		return nil, err
	}

	propagateNames(disc, sys)
	return disc, nil
}

func discretizeZOHAugmented(sys *System, dt float64) (*System, *mat.Dense, error) {
	n, m, _ := sys.Dims()
	N := sys.internalDelayCount()

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{A: newDense(0, 0), B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		return out, mat.DenseCopyOf(sys.LFT.B2), nil
	}

	mTotal := m + N
	nm := n + mTotal
	if mTotal == 0 {
		Adt := mat.NewDense(n, n, nil)
		Adt.Scale(dt, sys.A)
		Ad := mat.NewDense(n, n, nil)
		Ad.Exp(Adt)
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		return out, mat.DenseCopyOf(sys.LFT.B2), nil
	}

	M := mat.NewDense(nm, nm, nil)
	mRaw := M.RawMatrix()
	aRaw := sys.A.RawMatrix()
	var bRawZ, b2RawZ blas64.General
	if m > 0 {
		bRawZ = sys.B.RawMatrix()
	}
	if N > 0 {
		b2RawZ = sys.LFT.B2.RawMatrix()
	}

	for i := range n {
		row := mRaw.Data[i*mRaw.Stride:]
		for j := range n {
			row[j] = aRaw.Data[i*aRaw.Stride+j] * dt
		}
		if m > 0 {
			for j := range m {
				row[n+j] = bRawZ.Data[i*bRawZ.Stride+j] * dt
			}
		}
		if N > 0 {
			for j := range N {
				row[n+m+j] = b2RawZ.Data[i*b2RawZ.Stride+j] * dt
			}
		}
	}

	var eM mat.Dense
	eM.Exp(M)
	emRaw := eM.RawMatrix()

	adData := make([]float64, n*n)
	bdData := make([]float64, n*m)
	bd2Data := make([]float64, n*N)
	for i := range n {
		copy(adData[i*n:i*n+n], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		if m > 0 {
			copy(bdData[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
		}
		if N > 0 {
			copy(bd2Data[i*N:i*N+N], emRaw.Data[i*emRaw.Stride+n+m:i*emRaw.Stride+n+m+N])
		}
	}

	Ad := mat.NewDense(n, n, adData)
	var Bd *mat.Dense
	if m > 0 {
		Bd = mat.NewDense(n, m, bdData)
	} else {
		Bd = newDense(n, 0)
	}
	var Bd2 *mat.Dense
	if N > 0 {
		Bd2 = mat.NewDense(n, N, bd2Data)
	} else {
		Bd2 = newDense(n, 0)
	}

	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	return out, Bd2, nil
}

func discretizeTustinAugmented(sys *System, dt float64) (*System, *mat.Dense, error) {
	n, _, _ := sys.Dims()
	N := sys.internalDelayCount()

	beta := 2.0 / dt
	if math.IsInf(beta, 0) {
		return nil, nil, fmt.Errorf("discretizeTustinAugmented: dt too small: %w", ErrOverflow)
	}

	rational := &System{A: sys.A, B: sys.B, C: sys.C, D: sys.D}
	disc, err := bilinear(rational, -beta, -1.0, 1.0, beta)
	if err != nil {
		return nil, nil, err
	}
	disc.Dt = dt

	if n == 0 || N == 0 {
		return disc, mat.DenseCopyOf(sys.LFT.B2), nil
	}

	palpha := -beta
	A := mat.NewDense(n, n, nil)
	A.Copy(sys.A)
	aRaw := A.RawMatrix()
	for i := range n {
		aRaw.Data[i*aRaw.Stride+i] += palpha
	}

	var lu mat.LU
	lu.Factorize(A)

	B2 := mat.DenseCopyOf(sys.LFT.B2)
	if err := lu.SolveTo(B2, false, B2); err != nil {
		return nil, nil, fmt.Errorf("discretizeTustinAugmented: LU solve for B2 failed: %w", ErrSingularTransform)
	}

	twoAB := 2.0 * 1.0 * beta
	scale := math.Sqrt(math.Abs(twoAB))
	if palpha < 0 {
		scale = -scale
	}
	B2.Scale(scale, B2)

	return disc, B2, nil
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
		return nil, fmt.Errorf("DiscretizeImpulse: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: C2DMethodImpulse})
	}

	n, m, _ := sys.Dims()

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{
			A:  newDense(0, 0),
			B:  denseCopy(sys.B),
			C:  C,
			D:  D,
			Dt: dt,
		}
		propagateNames(out, sys)
		return out, nil
	}

	Adt := mat.NewDense(n, n, nil)
	Adt.Scale(dt, sys.A)
	Ad := mat.NewDense(n, n, nil)
	Ad.Exp(Adt)

	if m == 0 {
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		propagateNames(out, sys)
		return out, nil
	}

	Bd := mat.NewDense(n, m, nil)
	Bd.Mul(Ad, sys.B)
	Bd.Scale(dt, Bd)

	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	policy := newDelayConversionPolicy(dt, 0, 0)
	out, err := policy.applyDiscreteDelayFields(out, sys)
	if err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	return out, nil
}

func discretizeImpulseAugmented(sys *System, dt float64) (*System, *mat.Dense, error) {
	n, m, _ := sys.Dims()
	N := sys.internalDelayCount()

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{A: newDense(0, 0), B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		return out, mat.DenseCopyOf(sys.LFT.B2), nil
	}

	Adt := mat.NewDense(n, n, nil)
	Adt.Scale(dt, sys.A)
	Ad := mat.NewDense(n, n, nil)
	Ad.Exp(Adt)

	var Bd *mat.Dense
	if m > 0 {
		Bd = mat.NewDense(n, m, nil)
		Bd.Mul(Ad, sys.B)
		Bd.Scale(dt, Bd)
	} else {
		Bd = newDense(n, 0)
	}

	var Bd2 *mat.Dense
	if N > 0 {
		Bd2 = mat.NewDense(n, N, nil)
		Bd2.Mul(Ad, sys.LFT.B2)
		Bd2.Scale(dt, Bd2)
	} else {
		Bd2 = newDense(n, 0)
	}

	out := &System{A: Ad, B: Bd, C: C, D: D, Dt: dt}
	return out, Bd2, nil
}

func (sys *System) DiscretizeFOH(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeFOH: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	if sys.HasInternalDelay() {
		return discretizeWithInternalDelay(sys, dt, C2DOptions{Method: C2DMethodFOH})
	}

	n, m, p := sys.Dims()

	if n == 0 {
		return fohPureGain(sys, m, p, dt)
	}

	if m == 0 {
		Adt := mat.NewDense(n, n, nil)
		Adt.Scale(dt, sys.A)
		Ad := mat.NewDense(n, n, nil)
		Ad.Exp(Adt)
		out := &System{A: Ad, B: denseCopy(sys.B), C: denseCopy(sys.C), D: denseCopy(sys.D), Dt: dt}
		propagateNames(out, sys)
		return out, nil
	}

	nm := n + 2*m
	M := mat.NewDense(nm, nm, nil)
	mRaw := M.RawMatrix()
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()

	for i := range n {
		row := mRaw.Data[i*mRaw.Stride:]
		for j := range n {
			row[j] = aRaw.Data[i*aRaw.Stride+j] * dt
		}
		for j := range m {
			row[n+j] = bRaw.Data[i*bRaw.Stride+j] * dt
		}
	}
	for i := range m {
		mRaw.Data[(n+i)*mRaw.Stride+n+m+i] = 1
	}

	var eM mat.Dense
	eM.Exp(M)
	emRaw := eM.RawMatrix()

	adData := make([]float64, n*n)
	g0Data := make([]float64, n*m)
	g1Data := make([]float64, n*m)
	for i := range n {
		copy(adData[i*n:i*n+n], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		copy(g0Data[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
		copy(g1Data[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n+m:i*emRaw.Stride+n+2*m])
	}

	return buildFOHSystem(sys, n, m, p, dt, adData, g0Data, g1Data)
}

func fohPureGain(sys *System, m, p int, dt float64) (*System, error) {
	nNew := m
	aData := make([]float64, nNew*nNew)
	bData := make([]float64, nNew*m)
	for j := range m {
		bData[j*m+j] = 1
	}
	cData := make([]float64, p*nNew)
	if sys.D != nil {
		dRaw := sys.D.RawMatrix()
		for i := range p {
			copy(cData[i*nNew:i*nNew+m], dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m])
		}
	}
	dData := make([]float64, p*m)

	out := &System{
		A:  mat.NewDense(nNew, nNew, aData),
		B:  mat.NewDense(nNew, m, bData),
		C:  mat.NewDense(p, nNew, cData),
		D:  mat.NewDense(p, m, dData),
		Dt: dt,
	}
	policy := newDelayConversionPolicy(dt, 0, 0)
	var err error
	if out, err = policy.applyDiscreteDelayFields(out, sys); err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	fohAugmentStateNames(out, sys, 0, m)
	return out, nil
}

func buildFOHSystem(sys *System, n, m, p int, dt float64, adData, g0Data, g1Data []float64) (*System, error) {
	b0Data := make([]float64, n*m)
	for i := range g0Data {
		b0Data[i] = g0Data[i] - g1Data[i]
	}

	nNew := n + m
	aFoh := make([]float64, nNew*nNew)
	for i := range n {
		copy(aFoh[i*nNew:i*nNew+n], adData[i*n:i*n+n])
		copy(aFoh[i*nNew+n:i*nNew+n+m], b0Data[i*m:i*m+m])
	}

	bFoh := make([]float64, nNew*m)
	for i := range n {
		copy(bFoh[i*m:i*m+m], g1Data[i*m:i*m+m])
	}
	for j := range m {
		bFoh[(n+j)*m+j] = 1
	}

	cFoh := make([]float64, p*nNew)
	if sys.C != nil {
		cRaw := sys.C.RawMatrix()
		for i := range p {
			copy(cFoh[i*nNew:i*nNew+n], cRaw.Data[i*cRaw.Stride:i*cRaw.Stride+n])
		}
	}
	if sys.D != nil {
		dRaw := sys.D.RawMatrix()
		for i := range p {
			copy(cFoh[i*nNew+n:i*nNew+n+m], dRaw.Data[i*dRaw.Stride:i*dRaw.Stride+m])
		}
	}

	dFoh := make([]float64, p*m)

	out := &System{
		A:  mat.NewDense(nNew, nNew, aFoh),
		B:  mat.NewDense(nNew, m, bFoh),
		C:  mat.NewDense(p, nNew, cFoh),
		D:  mat.NewDense(p, m, dFoh),
		Dt: dt,
	}

	policy := newDelayConversionPolicy(dt, 0, 0)
	var err error
	if out, err = policy.applyDiscreteDelayFields(out, sys); err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	fohAugmentStateNames(out, sys, n, m)
	return out, nil
}

func fohAugmentStateNames(out, src *System, n, m int) {
	out.StateName = fohStateMetadata(src, n, m)
}

func discretizeFOHAugmented(sys *System, dt float64) (*System, *mat.Dense, error) {
	n, m, p := sys.Dims()
	N := sys.internalDelayCount()
	mTotal := m + N

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out, err := fohPureGain(sys, m, p, dt)
		if err != nil {
			return nil, nil, err
		}
		return out, mat.DenseCopyOf(sys.LFT.B2), nil
	}

	if mTotal == 0 {
		Adt := mat.NewDense(n, n, nil)
		Adt.Scale(dt, sys.A)
		Ad := mat.NewDense(n, n, nil)
		Ad.Exp(Adt)
		out := &System{A: Ad, B: denseCopy(sys.B), C: C, D: D, Dt: dt}
		return out, mat.DenseCopyOf(sys.LFT.B2), nil
	}

	sz := n + 2*mTotal
	M := mat.NewDense(sz, sz, nil)
	mRaw := M.RawMatrix()
	aRaw := sys.A.RawMatrix()
	var bRaw, b2Raw blas64.General
	if m > 0 {
		bRaw = sys.B.RawMatrix()
	}
	if N > 0 {
		b2Raw = sys.LFT.B2.RawMatrix()
	}

	for i := range n {
		row := mRaw.Data[i*mRaw.Stride:]
		for j := range n {
			row[j] = aRaw.Data[i*aRaw.Stride+j] * dt
		}
		if m > 0 {
			for j := range m {
				row[n+j] = bRaw.Data[i*bRaw.Stride+j] * dt
			}
		}
		if N > 0 {
			for j := range N {
				row[n+m+j] = b2Raw.Data[i*b2Raw.Stride+j] * dt
			}
		}
	}
	for i := range mTotal {
		mRaw.Data[(n+i)*mRaw.Stride+n+mTotal+i] = 1
	}

	var eM mat.Dense
	eM.Exp(M)
	emRaw := eM.RawMatrix()

	adData := make([]float64, n*n)
	g0BData := make([]float64, n*m)
	g1BData := make([]float64, n*m)
	g0B2Data := make([]float64, n*N)
	for i := range n {
		copy(adData[i*n:i*n+n], emRaw.Data[i*emRaw.Stride:i*emRaw.Stride+n])
		if m > 0 {
			copy(g0BData[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n:i*emRaw.Stride+n+m])
			copy(g1BData[i*m:i*m+m], emRaw.Data[i*emRaw.Stride+n+mTotal:i*emRaw.Stride+n+mTotal+m])
		}
		if N > 0 {
			copy(g0B2Data[i*N:i*N+N], emRaw.Data[i*emRaw.Stride+n+m:i*emRaw.Stride+n+m+N])
		}
	}

	out, err := buildFOHSystem(sys, n, m, p, dt, adData, g0BData, g1BData)
	if err != nil {
		return nil, nil, err
	}

	var Bd2 *mat.Dense
	if N > 0 {
		Bd2 = mat.NewDense(n, N, g0B2Data)
	} else {
		Bd2 = newDense(n, 0)
	}

	return out, Bd2, nil
}

func (sys *System) DiscretizeMatched(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("DiscretizeMatched: system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 {
		return nil, ErrInvalidSampleTime
	}

	n, m, p := sys.Dims()
	if m != 1 || p != 1 {
		return nil, fmt.Errorf("DiscretizeMatched: %w", ErrNotSISO)
	}

	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("DiscretizeMatched: internal delays not supported: %w", ErrFeedbackDelay)
	}

	if n == 0 {
		out := &System{
			A:  newDense(0, 0),
			B:  denseCopy(sys.B),
			C:  denseCopy(sys.C),
			D:  denseCopy(sys.D),
			Dt: dt,
		}
		propagateNames(out, sys)
		return out, nil
	}

	contPoles, err := sys.Poles()
	if err != nil {
		return nil, fmt.Errorf("DiscretizeMatched: %w", err)
	}
	contZeros, err := sys.Zeros()
	if err != nil {
		return nil, fmt.Errorf("DiscretizeMatched: %w", err)
	}

	discPoles := make([]complex128, len(contPoles))
	for i, p := range contPoles {
		discPoles[i] = cmplx.Exp(p * complex(dt, 0))
	}
	discZeros := make([]complex128, len(contZeros), len(contZeros)+len(contPoles))
	for i, z := range contZeros {
		discZeros[i] = cmplx.Exp(z * complex(dt, 0))
	}
	excess := len(contPoles) - len(contZeros)
	for range excess {
		discZeros = append(discZeros, complex(-1, 0))
	}

	gain, err := matchedGain(sys, discZeros, discPoles, dt)
	if err != nil {
		return nil, err
	}

	zpk, err := NewZPK(discZeros, discPoles, gain, dt)
	if err != nil {
		return nil, fmt.Errorf("DiscretizeMatched: %w", err)
	}
	ssResult, err := zpk.StateSpace(nil)
	if err != nil {
		return nil, fmt.Errorf("DiscretizeMatched: %w", err)
	}
	result := ssResult.Sys

	policy := newDelayConversionPolicy(dt, 0, 0)
	if result, err = policy.applyDiscreteDelayFields(result, sys); err != nil {
		return nil, err
	}
	propagateNames(result, sys)
	return result, nil
}

func matchedGain(sys *System, discZeros, discPoles []complex128, dt float64) (float64, error) {
	tfr, err := sys.TransferFunction(nil)
	if err != nil {
		return 0, fmt.Errorf("DiscretizeMatched: %w", err)
	}
	num := Poly(tfr.TF.Num[0][0])
	den := Poly(tfr.TF.Den[0])

	denAt0 := den.Eval(0)
	if cmplx.Abs(denAt0) > 1e-10 {
		contDC := num.Eval(0) / denAt0
		discDC := newRationalChannel(discZeros, discPoles, 1.0).eval(complex(1, 0))
		if cmplx.Abs(discDC) < 1e-14 {
			return matchedGainFallback(num, den, discZeros, discPoles, dt)
		}
		return real(contDC / discDC), nil
	}

	return matchedGainFallback(num, den, discZeros, discPoles, dt)
}

func matchedGainFallback(num, den Poly, discZeros, discPoles []complex128, dt float64) (float64, error) {
	sMatch := complex(0, math.Pi/(2*dt))
	zMatch := complex(0, 1)

	contVal := num.Eval(sMatch) / den.Eval(sMatch)
	discVal := newRationalChannel(discZeros, discPoles, 1.0).eval(zMatch)

	if cmplx.Abs(discVal) < 1e-14 {
		sMatch = complex(0, math.Pi/(4*dt))
		zMatch = cmplx.Exp(complex(0, math.Pi/4))
		contVal = num.Eval(sMatch) / den.Eval(sMatch)
		discVal = newRationalChannel(discZeros, discPoles, 1.0).eval(zMatch)
	}
	if cmplx.Abs(discVal) < 1e-14 {
		return 0, fmt.Errorf("DiscretizeMatched: cannot determine gain: %w", ErrSingularTransform)
	}
	return real(contVal / discVal), nil
}

// D2C converts a discrete-time system to continuous time.
//
// Supported methods:
//   - "tustin": bilinear (trapezoidal) inverse — equivalent to Undiscretize.
//     Always succeeds for non-pathological systems; exact inverse of Discretize.
//   - "zoh": zero-order-hold inverse via matrix logarithm.
//     A_c = log(A_d)/dt,  B_c = (A_d - I)⁻¹·A_c·B_d.
//     Requires A_d diagonalizable with no eigenvalue on the non-positive real
//     axis and A_d − I non-singular (no eigenvalue exactly at 1).
//
// Delay fields are converted back to continuous (τ_c = τ_d · Dt).
func (sys *System) D2C(method C2DMethod) (*System, error) {
	plan, err := newD2CPlan(sys, method)
	if err != nil {
		return nil, err
	}
	return plan.run()
}

func (sys *System) d2cZOH() (*System, error) {
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("D2C: zoh with internal delays not supported: %w", ErrFeedbackDelay)
	}

	n, m, _ := sys.Dims()
	dt := sys.Dt

	C := denseCopy(sys.C)
	D := denseCopy(sys.D)

	if n == 0 {
		out := &System{
			A:  newDense(0, 0),
			B:  denseCopy(sys.B),
			C:  C,
			D:  D,
			Dt: 0,
		}
		d2cPropagateDelays(out, sys, dt)
		propagateNames(out, sys)
		return out, nil
	}

	Alog, err := matLog(sys.A)
	if err != nil {
		return nil, fmt.Errorf("D2C zoh: %w", err)
	}
	Ac := mat.NewDense(n, n, nil)
	Ac.Scale(1.0/dt, Alog)

	var Bc *mat.Dense
	if m > 0 {
		AminusI := mat.NewDense(n, n, nil)
		AminusI.Copy(sys.A)
		raw := AminusI.RawMatrix()
		for i := range n {
			raw.Data[i*raw.Stride+i] -= 1
		}
		var lu mat.LU
		lu.Factorize(AminusI)
		if luNearSingular(&lu) {
			return nil, fmt.Errorf("D2C zoh: (A_d - I) singular (A_d has eigenvalue 1): %w", ErrSingularTransform)
		}
		AcB := mat.NewDense(n, m, nil)
		AcB.Mul(Ac, sys.B)
		Bc = mat.NewDense(n, m, nil)
		if err := lu.SolveTo(Bc, false, AcB); err != nil {
			return nil, fmt.Errorf("D2C zoh: LU solve for B_c failed: %w", ErrSingularTransform)
		}
	} else {
		Bc = denseCopy(sys.B)
	}

	out := &System{A: Ac, B: Bc, C: C, D: D, Dt: 0}
	d2cPropagateDelays(out, sys, dt)
	propagateNames(out, sys)
	return out, nil
}

func d2cPropagateDelays(out, sys *System, dt float64) {
	policy := newDelayConversionPolicy(dt, 0, 0)
	policy.applyContinuousDelayFields(out, sys)
}

func (sys *System) D2D(newDt float64, opts C2DOptions) (*System, error) {
	plan, err := newD2DPlan(sys, newDt, opts)
	if err != nil {
		return nil, err
	}
	return plan.run()
}
