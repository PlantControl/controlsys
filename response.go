package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

type TimeResponse struct {
	T          []float64
	Y          *mat.Dense
	OutputName []string
}

type timeResponsePlan struct {
	original      *System
	sim           *System
	t             []float64
	dt            float64
	steps         int
	wasContinuous bool
}

type timeResponsePlanner struct {
	sys *System
}

type standardInputResponse uint8

const (
	stepResponse standardInputResponse = iota
	impulseResponse
)

func newTimeResponsePlanner(sys *System) timeResponsePlanner {
	return timeResponsePlanner{sys: sys}
}

type DampInfo struct {
	Pole complex128
	Wn   float64
	Zeta float64
	Tau  float64
}

func autoTimeParams(sys *System) (dt, tFinal float64, err error) {
	n, _, _ := sys.Dims()
	if n == 0 {
		if sys.IsDiscrete() {
			return sys.Dt, 100 * sys.Dt, nil
		}
		return 0.01, 1.0, nil
	}

	poles, err := sys.Poles()
	if err != nil {
		return 0, 0, err
	}

	var wns []float64
	for _, p := range poles {
		var wn float64
		if sys.IsContinuous() {
			wn = cmplx.Abs(p)
		} else {
			lp := cmplx.Log(p)
			wn = cmplx.Abs(lp) / sys.Dt
		}
		if wn > 1e-10 {
			wns = append(wns, wn)
		}
	}

	if len(wns) == 0 {
		if sys.IsDiscrete() {
			return sys.Dt, 100 * sys.Dt, nil
		}
		return 0.01, 1.0, nil
	}

	minWn, maxWn := wns[0], wns[0]
	for _, w := range wns[1:] {
		if w < minWn {
			minWn = w
		}
		if w > maxWn {
			maxWn = w
		}
	}

	tFinal = 7.0 / minWn
	if tFinal > 1e4 {
		tFinal = 1e4
	}

	if sys.IsDiscrete() {
		dt = sys.Dt
	} else {
		dt = 1.0 / (20 * maxWn)
		if maxDt := tFinal / 100; dt > maxDt {
			dt = maxDt
		}
	}

	return dt, tFinal, nil
}

func prepareDiscrete(sys *System, tFinal, dt float64) (dsys *System, actualDt float64, steps int, wasContinuous bool, err error) {
	plan, err := prepareAutoTimeResponse(sys, tFinal, dt)
	if err != nil {
		return nil, 0, 0, false, err
	}
	return plan.sim, plan.dt, plan.steps, plan.wasContinuous, nil
}

func prepareAutoTimeResponse(sys *System, tFinal, dt float64) (timeResponsePlan, error) {
	return newTimeResponsePlanner(sys).auto(tFinal, dt)
}

func (p timeResponsePlanner) auto(tFinal, dt float64) (timeResponsePlan, error) {
	if p.sys.IsDiscrete() {
		actualDt := p.sys.Dt
		if tFinal <= 0 {
			var err error
			_, tFinal, err = autoTimeParams(p.sys)
			if err != nil {
				return timeResponsePlan{}, err
			}
		}
		steps := int(tFinal/actualDt) + 1
		return timeResponsePlan{
			original: p.sys,
			sim:      p.sys,
			t:        makeTimeVector(steps, actualDt),
			dt:       actualDt,
			steps:    steps,
		}, nil
	}

	if tFinal <= 0 || dt <= 0 {
		autoDt, autoTf, err2 := autoTimeParams(p.sys)
		if err2 != nil {
			return timeResponsePlan{}, err2
		}
		if tFinal <= 0 {
			tFinal = autoTf
		}
		if dt <= 0 {
			dt = autoDt
		}
	}

	dsys, err := p.sys.DiscretizeZOH(dt)
	if err != nil {
		return timeResponsePlan{}, fmt.Errorf("auto-discretize: %w", err)
	}
	actualDt := dt
	steps := int(tFinal/dt) + 1
	return timeResponsePlan{
		original:      p.sys,
		sim:           dsys,
		t:             makeTimeVector(steps, actualDt),
		dt:            actualDt,
		steps:         steps,
		wasContinuous: true,
	}, nil
}

func makeTimeVector(steps int, dt float64) []float64 {
	t := make([]float64, steps)
	for k := range t {
		t[k] = float64(k) * dt
	}
	return t
}

func (p timeResponsePlan) response(Y *mat.Dense) *TimeResponse {
	return &TimeResponse{T: p.t, Y: Y, OutputName: copyStringSlice(p.original.OutputName)}
}

func (p timeResponsePlan) allInputResponse(kind standardInputResponse) (*TimeResponse, error) {
	_, m, outputs := p.sim.Dims()
	if m <= 1 || outputs == 0 || p.sim.HasDelay() || p.sim.IsDescriptor() {
		return p.simulatedInputResponse(kind)
	}
	if err := p.sim.Validate(); err != nil {
		return nil, fmt.Errorf("input 0: %w", err)
	}
	return p.batchedInputResponse(kind), nil
}

func (p timeResponsePlan) simulatedInputResponse(kind standardInputResponse) (*TimeResponse, error) {
	_, m, outputs := p.sim.Dims()
	Y := mat.NewDense(outputs*m, p.steps, nil)
	amplitude := kind.amplitude(p)

	for input := range m {
		u := mat.NewDense(m, p.steps, nil)
		if kind == stepResponse {
			uRaw := u.RawMatrix()
			for sample := range p.steps {
				uRaw.Data[input*uRaw.Stride+sample] = amplitude
			}
		} else {
			u.Set(input, 0, amplitude)
		}
		resp, err := p.sim.Simulate(u, nil, nil)
		if err != nil {
			return nil, fmt.Errorf("input %d: %w", input, err)
		}
		if resp.Y == nil {
			continue
		}
		yRaw := Y.RawMatrix()
		responseRaw := resp.Y.RawMatrix()
		for output := range outputs {
			dstBase := (input*outputs + output) * yRaw.Stride
			srcBase := output * responseRaw.Stride
			copy(yRaw.Data[dstBase:dstBase+p.steps], responseRaw.Data[srcBase:srcBase+p.steps])
		}
	}

	return p.response(Y), nil
}

func (p timeResponsePlan) batchedInputResponse(kind standardInputResponse) *TimeResponse {
	states, inputs, outputs := p.sim.Dims()
	Y := mat.NewDense(outputs*inputs, p.steps, nil)
	yRaw := Y.RawMatrix()
	dRaw := p.sim.D.RawMatrix()
	amplitude := kind.amplitude(p)

	if states == 0 {
		for sample := range p.steps {
			if kind == impulseResponse && sample > 0 {
				break
			}
			for input := range inputs {
				for output := range outputs {
					yRaw.Data[(input*outputs+output)*yRaw.Stride+sample] = amplitude * dRaw.Data[output*dRaw.Stride+input]
				}
			}
		}
		return p.response(Y)
	}

	x := make([]float64, states*inputs)
	nextX := make([]float64, states*inputs)
	aRaw := p.sim.A.RawMatrix()
	bRaw := p.sim.B.RawMatrix()
	cRaw := p.sim.C.RawMatrix()

	for sample := range p.steps {
		inputActive := kind == stepResponse || sample == 0
		for input := range inputs {
			xInput := x[input*states : (input+1)*states]
			nextInput := nextX[input*states : (input+1)*states]
			for output := range outputs {
				value := 0.0
				if inputActive {
					value = amplitude * dRaw.Data[output*dRaw.Stride+input]
				}
				cRow := cRaw.Data[output*cRaw.Stride : output*cRaw.Stride+states]
				for state, coefficient := range cRow {
					value += coefficient * xInput[state]
				}
				yRaw.Data[(input*outputs+output)*yRaw.Stride+sample] = value
			}
			for state := range states {
				value := 0.0
				if inputActive {
					value = amplitude * bRaw.Data[state*bRaw.Stride+input]
				}
				aRow := aRaw.Data[state*aRaw.Stride : state*aRaw.Stride+states]
				for column, coefficient := range aRow {
					value += coefficient * xInput[column]
				}
				nextInput[state] = value
			}
		}
		x, nextX = nextX, x
	}

	return p.response(Y)
}

func (kind standardInputResponse) amplitude(p timeResponsePlan) float64 {
	if kind == impulseResponse && p.wasContinuous {
		return 1 / p.dt
	}
	return 1
}

func prepareLsimResponse(sys *System, u *mat.Dense, t []float64) (timeResponsePlan, *mat.Dense, error) {
	return newTimeResponsePlanner(sys).lsim(u, t)
}

func (p timeResponsePlanner) lsim(u *mat.Dense, t []float64) (timeResponsePlan, *mat.Dense, error) {
	if len(t) < 2 {
		return timeResponsePlan{}, nil, fmt.Errorf("Lsim: need at least 2 time points: %w", ErrDimensionMismatch)
	}

	_, m, _ := p.sys.Dims()
	sig, err := validateLsimInputSignal("Lsim", u, len(t), m)
	if err != nil {
		ur, uc := 0, 0
		if u != nil {
			ur, uc = u.Dims()
		}
		return timeResponsePlan{}, nil, fmt.Errorf("Lsim: u must be %d×%d, got %d×%d: %w", len(t), m, ur, uc, ErrDimensionMismatch)
	}

	dt, err := validateUniformTimeGrid("Lsim", t)
	if err != nil {
		return timeResponsePlan{}, nil, err
	}

	var dsys *System
	if p.sys.IsContinuous() {
		dsys, err = p.sys.DiscretizeZOH(dt)
		if err != nil {
			return timeResponsePlan{}, nil, fmt.Errorf("Lsim: %w", err)
		}
	} else {
		if math.Abs(p.sys.Dt-dt)/p.sys.Dt > 1e-6 {
			return timeResponsePlan{}, nil, fmt.Errorf("Lsim: time grid spacing %g does not match system Dt %g: %w", dt, p.sys.Dt, ErrDimensionMismatch)
		}
		dsys = p.sys
	}

	plan := timeResponsePlan{
		original:      p.sys,
		sim:           dsys,
		t:             t,
		dt:            dt,
		steps:         len(t),
		wasContinuous: p.sys.IsContinuous(),
	}
	return plan, sig.channelsBySamplesDense(), nil
}

func validateUniformTimeGrid(context string, t []float64) (float64, error) {
	dt := t[1] - t[0]
	if dt <= 0 {
		return 0, fmt.Errorf("%s: time step must be positive, got %g: %w", context, dt, ErrDimensionMismatch)
	}
	for k := 2; k < len(t); k++ {
		dk := t[k] - t[k-1]
		if math.Abs(dk-dt)/dt > 1e-6 {
			return 0, fmt.Errorf("%s: non-uniform time grid at index %d (dt=%g, expected %g); uniform grid required: %w", context, k, dk, dt, ErrDimensionMismatch)
		}
	}
	return dt, nil
}

func transposeSamplesToChannels(u *mat.Dense, steps, inputs int) *mat.Dense {
	return newSampledSignal("Lsim", u, inputs, steps, sampledSamplesByChannels).channelsBySamplesDense()
}

func (sys *System) DCGain() (*mat.Dense, error) {
	n, m, p := sys.Dims()

	if n == 0 && !sys.HasInternalDelay() {
		return denseCopy(sys.D), nil
	}
	if sys.HasInternalDelay() || sys.IsDescriptor() {
		return sys.dcGainByEvaluation()
	}

	if sys.IsContinuous() {
		var X mat.Dense
		err := X.Solve(sys.A, sys.B)
		if err != nil {
			return sys.dcGainByTransferFunctionLimit()
		}
		gain := mat.NewDense(p, m, nil)
		gain.Mul(sys.C, &X)
		gain.Scale(-1, gain)
		if sys.D != nil {
			gain.Add(gain, sys.D)
		}
		return gain, nil
	}

	ImA := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			v := -sys.A.At(i, j)
			if i == j {
				v += 1
			}
			ImA.Set(i, j, v)
		}
	}

	var X mat.Dense
	err := X.Solve(ImA, sys.B)
	if err != nil {
		return sys.dcGainByTransferFunctionLimit()
	}
	gain := mat.NewDense(p, m, nil)
	gain.Mul(sys.C, &X)
	if sys.D != nil {
		gain.Add(gain, sys.D)
	}
	return gain, nil
}

// dcGainByEvaluation closes internal delays (unity at DC) and honours E,
// which the explicit fast paths below ignore.
func (sys *System) dcGainByEvaluation() (*mat.Dense, error) {
	s0 := complex(0, 0)
	if !sys.IsContinuous() {
		s0 = 1
	}
	e, err := validFrequencyEvaluator(sys, "DCGain")
	if err != nil {
		return nil, err
	}
	g, err := e.eval(s0)
	if err != nil {
		return nil, fmt.Errorf("DCGain: %w", err)
	}
	_, m, p := sys.Dims()
	gain := mat.NewDense(p, m, nil)
	for i := range p {
		for j := range m {
			gain.Set(i, j, real(g[i][j]))
		}
	}
	return gain, nil
}

func (sys *System) dcGainByTransferFunctionLimit() (*mat.Dense, error) {
	if gain, ok := sys.dcGainByDecoupledSingularModes(); ok {
		return gain, nil
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		return nil, fmt.Errorf("DCGain: %w", err)
	}
	return dcGainFromTransferFunction(res.TF), nil
}

func (sys *System) dcGainByDecoupledSingularModes() (*mat.Dense, bool) {
	n, m, p := sys.Dims()
	target := 0.0
	if sys.IsDiscrete() {
		target = 1.0
	}
	tol := dcGainMatrixTol(sys.A)
	singular := make([]bool, n)
	regularCount := 0
	for k := range n {
		if math.Abs(sys.A.At(k, k)-target) <= tol && rowColDecoupledAt(sys.A, k, tol) {
			singular[k] = true
			continue
		}
		regularCount++
	}
	if regularCount == n {
		return nil, false
	}

	regular := make([]int, 0, regularCount)
	for k := range n {
		if !singular[k] {
			regular = append(regular, k)
		}
	}

	gain := denseCopySafe(sys.D, p, m)
	if regularCount > 0 {
		Areg := mat.NewDense(regularCount, regularCount, nil)
		Breg := mat.NewDense(regularCount, m, nil)
		Creg := mat.NewDense(p, regularCount, nil)
		for ri, srcRow := range regular {
			for ci, srcCol := range regular {
				Areg.Set(ri, ci, sys.A.At(srcRow, srcCol))
			}
			for j := range m {
				Breg.Set(ri, j, sys.B.At(srcRow, j))
			}
			for i := range p {
				Creg.Set(i, ri, sys.C.At(i, srcRow))
			}
		}

		var X mat.Dense
		var err error
		if sys.IsContinuous() {
			err = X.Solve(Areg, Breg)
		} else {
			ImA := mat.NewDense(regularCount, regularCount, nil)
			for i := 0; i < regularCount; i++ {
				for j := 0; j < regularCount; j++ {
					v := -Areg.At(i, j)
					if i == j {
						v += 1
					}
					ImA.Set(i, j, v)
				}
			}
			err = X.Solve(ImA, Breg)
		}
		if err != nil {
			return nil, false
		}
		var finite mat.Dense
		finite.Mul(Creg, &X)
		if sys.IsContinuous() {
			finite.Scale(-1, &finite)
		}
		gain.Add(gain, &finite)
	}

	residue := mat.NewDense(p, m, nil)
	for k := range n {
		if !singular[k] {
			continue
		}
		for i := range p {
			c := sys.C.At(i, k)
			if math.Abs(c) <= tol {
				continue
			}
			for j := range m {
				coeff := c * sys.B.At(k, j)
				if math.Abs(coeff) > tol {
					residue.Set(i, j, residue.At(i, j)+coeff)
				}
			}
		}
	}
	for i := range p {
		for j := range m {
			coeff := residue.At(i, j)
			if math.Abs(coeff) > tol {
				gain.Set(i, j, math.Inf(signInt(coeff)))
			}
		}
	}
	return gain, true
}

func rowColDecoupledAt(a *mat.Dense, k int, tol float64) bool {
	n, _ := a.Dims()
	for i := range n {
		if i == k {
			continue
		}
		if math.Abs(a.At(k, i)) > tol || math.Abs(a.At(i, k)) > tol {
			return false
		}
	}
	return true
}

func dcGainMatrixTol(m *mat.Dense) float64 {
	return 100 * eps() * math.Max(denseNorm(m), 1)
}

func dcGainFromTransferFunction(tf *TransferFunc) *mat.Dense {
	p, m := tf.Dims()
	point := 0.0
	if tf.Dt > 0 {
		point = 1.0
	}
	gain := mat.NewDense(p, m, nil)
	for i := range p {
		for j := range m {
			gain.Set(i, j, rationalLimitAtReal(tf.Num[i][j], tf.Den[i], point))
		}
	}
	return gain
}

func rationalLimitAtReal(num, den []float64, point float64) float64 {
	numMult, numValue, numZero := polynomialRootMultiplicityValue(num, point)
	if numZero {
		return 0
	}
	denMult, denValue, denZero := polynomialRootMultiplicityValue(den, point)
	if denZero {
		return math.NaN()
	}
	if denMult > numMult {
		return math.Inf(signInt(numValue / denValue))
	}
	if denMult < numMult {
		return 0
	}
	return numValue / denValue
}

func polynomialRootMultiplicityValue(poly []float64, root float64) (multiplicity int, value float64, zero bool) {
	p := trimLeadingNearZero(poly, dcGainPolyTol(poly))
	if len(p) == 0 {
		return 0, 0, true
	}
	tol := dcGainPolyTol(p)
	for len(p) > 1 {
		q, rem := deflateRealRoot(p, root)
		if math.Abs(rem) > tol {
			break
		}
		multiplicity++
		p = trimLeadingNearZero(q, tol)
		if len(p) == 0 {
			return multiplicity, 0, true
		}
		tol = dcGainPolyTol(p)
	}
	return multiplicity, real(Poly(p).Eval(complex(root, 0))), false
}

func deflateRealRoot(poly []float64, root float64) ([]float64, float64) {
	q := make([]float64, len(poly)-1)
	q[0] = poly[0]
	for i := 1; i < len(q); i++ {
		q[i] = poly[i] + root*q[i-1]
	}
	rem := poly[len(poly)-1] + root*q[len(q)-1]
	return q, rem
}

func trimLeadingNearZero(poly []float64, tol float64) []float64 {
	start := 0
	for start < len(poly) && math.Abs(poly[start]) <= tol {
		start++
	}
	return poly[start:]
}

func dcGainPolyTol(poly []float64) float64 {
	scale := 1.0
	for _, v := range poly {
		if a := math.Abs(v); a > scale {
			scale = a
		}
	}
	return 100 * eps() * scale
}

func signInt(v float64) int {
	if math.Signbit(v) {
		return -1
	}
	return 1
}

func Damp(sys *System) ([]DampInfo, error) {
	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	if len(poles) == 0 {
		return nil, nil
	}

	result := make([]DampInfo, len(poles))
	for i, p := range poles {
		var wn, zeta float64

		if sys.IsDiscrete() {
			sc := cmplx.Log(p) / complex(sys.Dt, 0)
			wn = cmplx.Abs(sc)
			if wn > 0 {
				zeta = -real(sc) / wn
			}
		} else {
			wn = cmplx.Abs(p)
			if wn > 0 {
				zeta = -real(p) / wn
			}
		}

		tau := math.Inf(1)
		if sigma := zeta * wn; sigma > 0 {
			tau = 1.0 / sigma
		}

		result[i] = DampInfo{Pole: p, Wn: wn, Zeta: zeta, Tau: tau}
	}
	return result, nil
}

func Step(sys *System, tFinal float64) (*TimeResponse, error) {
	plan, err := prepareAutoTimeResponse(sys, tFinal, 0)
	if err != nil {
		return nil, err
	}
	resp, err := plan.allInputResponse(stepResponse)
	if err != nil {
		return nil, fmt.Errorf("Step: %w", err)
	}
	return resp, nil
}

func Impulse(sys *System, tFinal float64) (*TimeResponse, error) {
	plan, err := prepareAutoTimeResponse(sys, tFinal, 0)
	if err != nil {
		return nil, err
	}

	resp, err := plan.allInputResponse(impulseResponse)
	if err != nil {
		return nil, fmt.Errorf("Impulse: %w", err)
	}
	return resp, nil
}

func Initial(sys *System, x0 *mat.VecDense, tFinal float64) (*TimeResponse, error) {
	if x0 == nil {
		return nil, fmt.Errorf("Initial: x0 must not be nil: %w", ErrDimensionMismatch)
	}

	plan, err := prepareAutoTimeResponse(sys, tFinal, 0)
	if err != nil {
		return nil, err
	}

	_, m, _ := plan.sim.Dims()
	u := mat.NewDense(m, plan.steps, nil)
	resp, err := plan.sim.Simulate(u, x0, nil)
	if err != nil {
		return nil, fmt.Errorf("Initial: %w", err)
	}

	return plan.response(resp.Y), nil
}

func Lsim(sys *System, u *mat.Dense, t []float64, x0 *mat.VecDense) (*TimeResponse, error) {
	plan, uSim, err := prepareLsimResponse(sys, u, t)
	if err != nil {
		return nil, err
	}
	resp, err := plan.sim.Simulate(uSim, x0, nil)
	if err != nil {
		return nil, fmt.Errorf("Lsim: %w", err)
	}

	return plan.response(resp.Y), nil
}
