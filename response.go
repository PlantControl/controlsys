package controlsys

import (
	"errors"
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

const (
	autoHorizonTimeConstants = 7.0
	autoMaxHorizon           = 1e4
	autoMaxSamples           = 100001
	autoMinDiscreteSamples   = 10
	gridTol                  = 1e-9
)

func autoTimeParams(sys *System) (dt, tFinal float64, err error) {
	tFinal, maxWn, err := autoTimeScales(sys)
	if err != nil {
		return 0, 0, err
	}
	if sys.IsDiscrete() {
		return sys.Dt, tFinal, nil
	}
	return autoContinuousDt(tFinal, maxWn), tFinal, nil
}

func autoTimeScales(sys *System) (tFinal, maxWn float64, err error) {
	n, _, _ := sys.Dims()
	var poles []complex128
	if n > 0 {
		if poles, err = sys.Poles(); err != nil {
			return 0, 0, err
		}
	}

	for _, p := range poles {
		s := p
		if sys.IsDiscrete() {
			s = cmplx.Log(p) / complex(sys.Dt, 0)
		}
		wn := cmplx.Abs(s)
		if math.IsInf(wn, 0) || math.IsNaN(wn) || wn <= 1e-10 {
			continue
		}
		maxWn = math.Max(maxWn, wn)
		horizon := autoHorizonTimeConstants / wn
		if sigma := -real(s); sigma > 1e-10*wn {
			horizon = autoHorizonTimeConstants / sigma
		}
		tFinal = math.Max(tFinal, horizon)
	}

	if tFinal == 0 {
		tFinal = 1
		if sys.IsDiscrete() {
			tFinal = 100 * sys.Dt
		}
	}
	tFinal = math.Min(tFinal+responseDelaySpan(sys), autoMaxHorizon)
	if sys.IsDiscrete() {
		minSamples := float64(max(autoMinDiscreteSamples, n+1))
		tFinal = math.Min(math.Max(tFinal, minSamples*sys.Dt), float64(autoMaxSamples-1)*sys.Dt)
	}
	return tFinal, maxWn, nil
}

func responseDelaySpan(sys *System) float64 {
	span := 0.0
	if total := sys.TotalDelay(); total != nil {
		span = mat.Max(total)
	}
	if sys.LFT != nil {
		for _, tau := range sys.LFT.Tau {
			span += tau
		}
	}
	if sys.IsDiscrete() {
		span *= sys.Dt
	}
	return span
}

func autoContinuousDt(tFinal, maxWn float64) float64 {
	dt := tFinal / 100
	if maxWn > 0 {
		dt = math.Min(dt, 1/(20*maxWn))
	}
	return math.Max(dt, tFinal/float64(autoMaxSamples-1))
}

func gridSampleCount(span, dt float64) int {
	return int(math.Floor(span/dt+gridTol)) + 1
}

func validateTimeHorizon(context string, tFinal float64) error {
	if math.IsNaN(tFinal) || math.IsInf(tFinal, 0) {
		return fmt.Errorf("%s: final time must be finite, got %g", context, tFinal)
	}
	return nil
}

func prepareAutoTimeResponse(sys *System, tFinal, dt float64) (timeResponsePlan, error) {
	return newTimeResponsePlanner(sys).auto(tFinal, dt)
}

func (p timeResponsePlanner) grid(tFinal, dt float64) (t []float64, actualDt float64, err error) {
	if err := validateTimeHorizon("time response", tFinal); err != nil {
		return nil, 0, err
	}
	continuous := p.sys.IsContinuous()
	if tFinal <= 0 || (continuous && dt <= 0) {
		autoTf, maxWn, err := autoTimeScales(p.sys)
		if err != nil {
			return nil, 0, err
		}
		if tFinal <= 0 {
			tFinal = autoTf
		}
		if continuous && dt <= 0 {
			dt = autoContinuousDt(tFinal, maxWn)
		}
	}
	if !continuous {
		return makeTimeVector(gridSampleCount(tFinal, p.sys.Dt), p.sys.Dt), p.sys.Dt, nil
	}
	intervals := max(int(math.Ceil(tFinal/dt-gridTol)), 1)
	actualDt = tFinal / float64(intervals)
	t = makeTimeVector(intervals+1, actualDt)
	t[intervals] = tFinal
	return t, actualDt, nil
}

func (p timeResponsePlanner) auto(tFinal, dt float64) (timeResponsePlan, error) {
	t, dt, err := p.grid(tFinal, dt)
	if err != nil {
		return timeResponsePlan{}, err
	}
	plan := timeResponsePlan{original: p.sys, sim: p.sys, t: t, dt: dt, steps: len(t)}
	if p.sys.IsDiscrete() {
		return plan, nil
	}
	if plan.sim, err = p.sys.DiscretizeZOH(dt); err != nil {
		return timeResponsePlan{}, fmt.Errorf("auto-discretize: %w", err)
	}
	plan.wasContinuous = true
	return plan, nil
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
	if outputs*m == 0 {
		return p.response(&mat.Dense{}), nil
	}
	Y := mat.NewDense(outputs*m, p.steps, nil)
	for input := range m {
		u := mat.NewDense(m, p.steps, nil)
		if kind == stepResponse {
			uRaw := u.RawMatrix()
			for sample := range p.steps {
				uRaw.Data[input*uRaw.Stride+sample] = 1
			}
		} else {
			u.Set(input, 0, 1)
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

	if states == 0 {
		for sample := range p.steps {
			if kind == impulseResponse && sample > 0 {
				break
			}
			for input := range inputs {
				for output := range outputs {
					yRaw.Data[(input*outputs+output)*yRaw.Stride+sample] = dRaw.Data[output*dRaw.Stride+input]
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
					value = dRaw.Data[output*dRaw.Stride+input]
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
					value = bRaw.Data[state*bRaw.Stride+input]
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

// DCGain returns the steady-state gain G(0) (G(1) for discrete models). Like
// MATLAB dcgain, entries reached by an integrator (a pole at s = 0 or z = 1)
// are infinite rather than an error, for every realization; internal delays are
// unity at DC.
// A model with no inputs or no outputs has an empty gain, returned as an
// empty matrix.
func (sys *System) DCGain() (*mat.Dense, error) {
	n, m, p := sys.Dims()
	if m == 0 || p == 0 {
		return &mat.Dense{}, nil
	}
	if n == 0 && !sys.HasInternalDelay() {
		return denseCopy(sys.D), nil
	}
	if sys.HasInternalDelay() || sys.IsDescriptor() {
		gain, err := sys.dcGainByEvaluation()
		if err == nil || !errors.Is(err, ErrSingularTransform) {
			return gain, err
		}
		return sys.dcGainOfDelayFreeExplicit(err)
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
	g, err := e.evalPoint(pointAt(s0), false)
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

// dcGainOfDelayFreeExplicit handles a pole at DC, where the pencil solve fails:
// internal delays are unity at DC, so closing them (Δ=I) and removing E leaves
// an explicit model whose fast path resolves singular A.
func (sys *System) dcGainOfDelayFreeExplicit(evalErr error) (*mat.Dense, error) {
	reduced, err := sys.ToExplicit()
	if err != nil {
		return nil, evalErr
	}
	if reduced.HasInternalDelay() {
		if reduced, err = reduced.ZeroDelayApprox(); err != nil {
			return nil, evalErr
		}
	}
	return reduced.DCGain()
}

func (sys *System) dcGainByTransferFunctionLimit() (*mat.Dense, error) {
	if gain, ok := sys.dcGainByDecoupledSingularModes(); ok {
		return gain, nil
	}
	res, err := sys.rationalTransferFunction(nil)
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
			switch {
			case math.IsInf(wn, 1):
				zeta = 1
			case wn > 0:
				zeta = -real(sc) / wn
			}
		} else {
			wn = cmplx.Abs(p)
			if wn > 0 {
				zeta = -real(p) / wn
			}
		}

		tau := math.Inf(1)
		if math.IsInf(wn, 1) && zeta > 0 {
			tau = 0
		} else if sigma := zeta * wn; sigma > 0 {
			tau = 1.0 / sigma
		}

		result[i] = DampInfo{Pole: p, Wn: wn, Zeta: zeta, Tau: tau}
	}
	return result, nil
}

// Step returns the step response of each input channel. Descriptor models
// follow MATLAB: proper singular-E models are reduced to an explicit model
// plus feedthrough, improper ones return ErrImproperModel. Continuous models
// with internal delays of one common length are sampled exactly (method of
// steps); other internal-delay models are simulated through the approximate
// ZOH discretization of the delay channels, as MATLAB does, with O(dt) error.
func Step(sys *System, tFinal float64) (*TimeResponse, error) {
	sys, _, _, err := sys.timeResponseForm(nil)
	if err != nil {
		return nil, fmt.Errorf("Step: %w", err)
	}
	if resp, ok, err := delayChainAuto(sys, tFinal, stepResponse); ok || err != nil {
		return resp, err
	}
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

// Impulse returns the impulse response of each input channel. For
// continuous models the Dirac feedthrough D·δ(t) is dropped and y(0) = C·B,
// as in MATLAB; for a proper singular-E descriptor model D includes the
// algebraic part's static gain. A continuous model whose input feeds an internal delay
// directly (LFT.D21 ≠ 0) carries delayed Diracs and returns
// ErrInternalDelayImpulse; MATLAB's impulse rejects every continuous
// internal-delay model, while this library samples the D21 = 0 case (exactly
// for one common delay length, else by the approximate ZOH discretization).
func Impulse(sys *System, tFinal float64) (*TimeResponse, error) {
	sys, _, _, err := sys.timeResponseForm(nil)
	if err != nil {
		return nil, fmt.Errorf("Impulse: %w", err)
	}
	kind := impulseResponse
	if sys.IsContinuous() {
		if sys.LFT != nil && sys.HasInternalDelay() && !allZeroDense(sys.LFT.D21) {
			return nil, fmt.Errorf("Impulse: %w", ErrInternalDelayImpulse)
		}
		if resp, ok, err := delayChainAuto(sys, tFinal, impulseResponse); ok || err != nil {
			return resp, err
		}
		derived, err := impulseAsStepModel(sys)
		if err != nil {
			return nil, fmt.Errorf("Impulse: %w", err)
		}
		if derived != nil {
			sys, kind = derived, stepResponse
		}
	}
	plan, err := prepareAutoTimeResponse(sys, tFinal, 0)
	if err != nil {
		return nil, err
	}

	resp, err := plan.allInputResponse(kind)
	if err != nil {
		return nil, fmt.Errorf("Impulse: %w", err)
	}
	return resp, nil
}

// impulseAsStepModel returns a model whose step response equals the
// continuous impulse response C·e^{At}·B sampled exactly: (A, A·B, C, C·B)
// with the same delays. The Dirac part D·δ(t) is dropped, as in MATLAB. It
// returns nil for models without inputs or outputs and requires D21 = 0.
func impulseAsStepModel(sys *System) (*System, error) {
	n, m, p := sys.Dims()
	if m == 0 || p == 0 {
		return nil, nil
	}
	src, err := sys.ToExplicit()
	if err != nil {
		return nil, err
	}
	derived := src.Copy()
	derived.D = mat.NewDense(p, m, nil)
	if n > 0 {
		derived.B.Mul(src.A, src.B)
		derived.D.Mul(src.C, src.B)
		if derived.LFT != nil && derived.LFT.D21 != nil {
			derived.LFT.D21.Mul(src.LFT.C2, src.B)
		}
	}
	return derived, nil
}

// delayChainAuto samples a standard response on the automatic grid with the
// exact internal-delay chain; ok is false when the model is outside its class.
func delayChainAuto(sys *System, tFinal float64, kind standardInputResponse) (*TimeResponse, bool, error) {
	if _, _, ok := delayChainModel(sys); !ok {
		return nil, false, nil
	}
	t, dt, err := newTimeResponsePlanner(sys).grid(tFinal, 0)
	if err != nil {
		return nil, false, err
	}
	Y, ok := delayChainStandardResponse(sys, t, dt, kind)
	if !ok {
		return nil, false, nil
	}
	return &TimeResponse{T: t, Y: Y, OutputName: copyStringSlice(sys.OutputName)}, true, nil
}

// Initial returns the free response from x0 with zero delay-line history.
// Output delays, including the output share of Delay, delay it; see Simulate.
// As in MATLAB, a descriptor model with singular E rejects a nonzero x0 with
// ErrDescriptorInitialState.
// Continuous models without internal delays, or whose internal delays share
// one length, are sampled exactly; other internal-delay models use the
// approximate ZOH discretization of the delay channels, as MATLAB does.
func Initial(sys *System, x0 *mat.VecDense, tFinal float64) (*TimeResponse, error) {
	if x0 == nil {
		return nil, fmt.Errorf("Initial: x0 must not be nil: %w", ErrDimensionMismatch)
	}
	sim, simX0, reduced, err := sys.timeResponseForm(x0)
	if err != nil {
		return nil, fmt.Errorf("Initial: %w", err)
	}
	if reduced {
		simX0 = &mat.VecDense{}
		if n, _, _ := sim.Dims(); n > 0 {
			simX0 = mat.NewVecDense(n, nil)
		}
	}
	x0 = simX0
	free := sim.Copy()
	free.InputDelay = nil
	free.Delay = nil

	if free.IsContinuous() && !free.HasInternalDelay() && !free.IsDescriptor() {
		return free.continuousFreeResponse(x0, tFinal)
	}
	if _, _, ok := delayChainModel(free); ok {
		t, dt, err := newTimeResponsePlanner(free).grid(tFinal, 0)
		if err != nil {
			return nil, err
		}
		Y, ok, err := delayChainForcedResponse(free, x0, nil, dt, len(t))
		if err != nil {
			return nil, fmt.Errorf("Initial: %w", err)
		}
		if ok {
			return &TimeResponse{T: t, Y: Y, OutputName: copyStringSlice(free.OutputName)}, nil
		}
	}

	plan, err := prepareAutoTimeResponse(free, tFinal, 0)
	if err != nil {
		return nil, err
	}

	ds := plan.sim
	_, m, p := ds.Dims()
	if p == 0 {
		return plan.response(&mat.Dense{}), nil
	}
	if m == 0 {
		ds = zeroInputModel(ds)
	}
	resp, err := ds.Simulate(mat.NewDense(max(m, 1), plan.steps, nil), x0, nil)
	if err != nil {
		return nil, fmt.Errorf("Initial: %w", err)
	}

	return plan.response(resp.Y), nil
}

// zeroInputModel returns sys with one zero input standing in for m = 0, since
// gonum cannot hold the n×0 and p×0 blocks of a model without inputs.
func zeroInputModel(sys *System) *System {
	n, _, p := sys.Dims()
	pad := sys.Copy()
	pad.B = newDense(n, 1)
	pad.D = newDense(p, 1)
	pad.InputDelay = nil
	pad.Delay = nil
	if pad.LFT != nil {
		pad.LFT.D21 = newDense(len(pad.LFT.Tau), 1)
	}
	return pad
}

// zeroOutputModel returns sys with one zero output standing in for p = 0, so
// the simulation kernels still propagate the state.
func zeroOutputModel(sys *System) *System {
	n, m, _ := sys.Dims()
	pad := sys.Copy()
	pad.C = newDense(1, n)
	pad.D = newDense(1, m)
	pad.OutputDelay = nil
	pad.Delay = nil
	if pad.LFT != nil {
		pad.LFT.D12 = newDense(1, len(pad.LFT.Tau))
	}
	return pad
}

func (sys *System) continuousFreeResponse(x0 *mat.VecDense, tFinal float64) (*TimeResponse, error) {
	t, dt, err := newTimeResponsePlanner(sys).grid(tFinal, 0)
	if err != nil {
		return nil, err
	}
	Y, err := sys.continuousFreeSamples(x0, len(t), dt)
	if err != nil {
		return nil, fmt.Errorf("Initial: %w", err)
	}
	return &TimeResponse{T: t, Y: Y, OutputName: copyStringSlice(sys.OutputName)}, nil
}

// continuousFreeSamples samples y_i(t) = C_i·e^{A(t−τ_i)}·x0 for t ≥ τ_i
// (zero before), τ_i the output delay, exactly at t = k·dt. Input delays do
// not act on the free response; timeResponseForm has already moved the
// output share of Delay into OutputDelay for a nonzero x0.
func (sys *System) continuousFreeSamples(x0 *mat.VecDense, steps int, dt float64) (*mat.Dense, error) {
	n, _, p := sys.Dims()
	if x0.Len() != n {
		return nil, fmt.Errorf("x0 length %d != state dimension %d: %w", x0.Len(), n, ErrDimensionMismatch)
	}
	if p == 0 || steps == 0 {
		return nil, nil
	}
	Y := mat.NewDense(p, steps, nil)
	if n == 0 {
		return Y, nil
	}

	var scaled, ad mat.Dense
	scaled.Scale(dt, sys.A)
	ad.Exp(&scaled)

	byDelay := make(map[float64][]int)
	var delays []float64
	for i := range p {
		tau := 0.0
		if sys.OutputDelay != nil {
			tau = sys.OutputDelay[i]
		}
		if _, ok := byDelay[tau]; !ok {
			delays = append(delays, tau)
		}
		byDelay[tau] = append(byDelay[tau], i)
	}

	x := mat.NewVecDense(n, nil)
	next := mat.NewVecDense(n, nil)
	yRaw := Y.RawMatrix()
	for _, tau := range delays {
		k0 := max(int(math.Ceil(tau/dt-gridTol)), 0)
		if k0 >= steps {
			continue
		}
		x.CopyVec(x0)
		if lead := float64(k0)*dt - tau; lead > 0 {
			var lag mat.Dense
			scaled.Scale(lead, sys.A)
			lag.Exp(&scaled)
			next.MulVec(&lag, x0)
			x, next = next, x
		}
		outputs := byDelay[tau]
		for k := k0; k < steps; k++ {
			for _, i := range outputs {
				yRaw.Data[i*yRaw.Stride+k] = mat.Dot(sys.C.RowView(i), x)
			}
			next.MulVec(&ad, x)
			x, next = next, x
		}
	}
	return Y, nil
}

// Lsim simulates the response to u held constant between samples (ZOH).
// Descriptor models are handled as in Step; with singular E a nonzero x0
// returns ErrDescriptorInitialState, as in MATLAB. x0 with Delay follows
// Simulate.
// Continuous models with internal delays of one common length are sampled
// exactly; other internal-delay models use the approximate ZOH
// discretization of the delay channels, as MATLAB does.
func Lsim(sys *System, u *mat.Dense, t []float64, x0 *mat.VecDense) (*TimeResponse, error) {
	sys, x0, _, err := sys.timeResponseForm(x0)
	if err != nil {
		return nil, fmt.Errorf("Lsim: %w", err)
	}
	if _, m, p := sys.Dims(); m == 0 && (u == nil || u.IsEmpty()) {
		if p == 0 {
			return nil, fmt.Errorf("Lsim: model has no inputs and no outputs: %w", ErrDimensionMismatch)
		}
		sys, u = zeroInputModel(sys), mat.NewDense(max(len(t), 1), 1, nil)
	}
	plan, uSim, err := prepareLsimResponse(sys, u, t)
	if err != nil {
		return nil, err
	}
	if plan.wasContinuous {
		Y, ok, err := delayChainForcedResponse(sys, x0, uSim, plan.dt, plan.steps)
		if err != nil {
			return nil, fmt.Errorf("Lsim: %w", err)
		}
		if ok {
			return plan.response(Y), nil
		}
	}
	exactFree := x0 != nil && plan.wasContinuous && sys.HasDelay() && !sys.HasInternalDelay() && !sys.IsDescriptor()
	simX0 := x0
	if exactFree {
		simX0 = nil
	}
	resp, err := plan.sim.Simulate(uSim, simX0, nil)
	if err != nil {
		return nil, fmt.Errorf("Lsim: %w", err)
	}
	if exactFree && resp.Y != nil {
		free, err := sys.continuousFreeSamples(x0, plan.steps, plan.dt)
		if err != nil {
			return nil, fmt.Errorf("Lsim: %w", err)
		}
		resp.Y.Add(resp.Y, free)
	}

	return plan.response(resp.Y), nil
}
