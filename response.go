package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// TimeResponse is a sampled time response. T holds the sample times and Y
// one row per output channel and one column per sample. For Step and
// Impulse the rows are input-major: row input*p+output is the response of
// output to input, p the number of outputs.
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

// DampInfo describes one pole as MATLAB damp
// (https://www.mathworks.com/help/control/ref/dynamicsystem.damp.html). For a
// discrete pole z the values refer to s = ln(z)/Ts. Wn = |s|; Zeta =
// -cos(∠s), so a pole at the origin has Zeta = -1 and an unstable pole a
// negative Zeta; Tau = 1/(Wn·Zeta) = -1/Re(s) is negative for an unstable
// pole, +Inf for a pole on the stability boundary (no decay) and 0 for z = 0.
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
	if total := newDelayTopology(sys).totalExternal(true); total != nil {
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

// validateTimeHorizon accepts a finite tFinal >= 0; 0 selects the automatic
// horizon.
func validateTimeHorizon(tFinal float64) error {
	if !isFinite(tFinal) || tFinal < 0 {
		return fmt.Errorf("final time %g must be finite and non-negative (0 = automatic): %w", tFinal, ErrInvalidArgument)
	}
	return nil
}

// requireTimeResponse validates the model and final time of the standard
// response op, which needs at least one input and one output.
func requireTimeResponse(op string, sys *System, tFinal float64) error {
	if err := requireSystem(op, sys); err != nil {
		return err
	}
	if err := validateTimeHorizon(tFinal); err != nil {
		return fmt.Errorf("%s: %w", op, err)
	}
	return requireInputsOutputs(op, sys)
}

func requireInputsOutputs(op string, sys *System) error {
	if _, m, p := sys.Dims(); m == 0 || p == 0 {
		return fmt.Errorf("%s: model has %d inputs and %d outputs, need at least one of each: %w", op, m, p, ErrDimensionMismatch)
	}
	return nil
}

func prepareAutoTimeResponse(sys *System, tFinal, dt float64) (timeResponsePlan, error) {
	return newTimeResponsePlanner(sys).auto(tFinal, dt)
}

func (p timeResponsePlanner) grid(tFinal, dt float64) (t []float64, actualDt float64, err error) {
	if err := validateTimeHorizon(tFinal); err != nil {
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
	if plan.sim, err = p.sys.C2D(dt, C2DOptions{}); err != nil {
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
	if err := p.sim.validate(); err != nil {
		return nil, fmt.Errorf("input 0: %w", err)
	}
	return p.batchedInputResponse(kind), nil
}

func (p timeResponsePlan) simulatedInputResponse(kind standardInputResponse) (*TimeResponse, error) {
	_, m, outputs := p.sim.Dims()
	if outputs*m == 0 {
		return nil, fmt.Errorf("model has %d inputs and %d outputs: %w", m, outputs, ErrDimensionMismatch)
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

	if err := requireFinite("Lsim", "t", t...); err != nil {
		return timeResponsePlan{}, nil, err
	}
	_, m, _ := p.sys.Dims()
	if u == nil {
		return timeResponsePlan{}, nil, fmt.Errorf("Lsim: u is nil: %w", ErrInvalidArgument)
	}
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
		dsys, err = p.sys.C2D(dt, C2DOptions{})
		if err != nil {
			return timeResponsePlan{}, nil, fmt.Errorf("Lsim: %w", err)
		}
	} else {
		if math.Abs(p.sys.Dt-dt)/p.sys.Dt > 1e-6 {
			return timeResponsePlan{}, nil, fmt.Errorf("Lsim: time grid spacing %g does not match system Dt %g: %w", dt, p.sys.Dt, ErrInvalidArgument)
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
		return 0, fmt.Errorf("%s: time step must be positive, got %g: %w", context, dt, ErrInvalidArgument)
	}
	for k := 2; k < len(t); k++ {
		dk := t[k] - t[k-1]
		if math.Abs(dk-dt)/dt > 1e-6 {
			return 0, fmt.Errorf("%s: non-uniform time grid at index %d (dt=%g, expected %g); uniform grid required: %w", context, k, dk, dt, ErrInvalidArgument)
		}
	}
	return dt, nil
}

// DCGain returns the steady-state gain G(0) (G(1) for discrete models). Like
// MATLAB dcgain, entries reached by an integrator (a pole at s = 0 or z = 1)
// are infinite rather than an error, for every realization; internal delays are
// unity at DC. Where I−A (A) is singular to working precision but no computed
// pole is near DC, the gain is solved in double-double as FreqResponse
// evaluates it, so a pencil ill-conditioned by poles merely close to DC keeps
// its finite gain.
// A model with no inputs or no outputs has an empty gain, which a
// *mat.Dense cannot hold, so it returns ErrDimensionMismatch; a NaN or Inf
// model entry returns ErrInvalidArgument.
func (sys *System) DCGain() (*mat.Dense, error) {
	return sys.dcGain(true)
}

// integratingDCGain is DCGain with an ill-conditioned DC solve always
// resolved by the transfer limit, whose coefficient tolerance takes a pole
// rounded off DC for the integrator it stands for; Pidtune's integrator
// classification relies on it.
func (sys *System) integratingDCGain() (*mat.Dense, error) {
	return sys.dcGain(false)
}

func (sys *System) dcGain(offPole bool) (*mat.Dense, error) {
	if err := requireFiniteSystem("DCGain", sys); err != nil {
		return nil, err
	}
	if err := requireInputsOutputs("DCGain", sys); err != nil {
		return nil, err
	}
	n, m, p := sys.Dims()
	if n == 0 && !sys.HasInternalDelay() {
		return denseCopy(sys.D), nil
	}
	if sys.HasInternalDelay() || sys.IsDescriptor() {
		gain, err := sys.dcGainByEvaluation()
		if err == nil || !errors.Is(err, ErrSingularTransform) {
			return gain, err
		}
		return sys.dcGainOfDelayFreeExplicit(err, offPole)
	}

	if sys.IsContinuous() {
		var X mat.Dense
		err := X.Solve(sys.A, sys.B)
		if err != nil {
			return sys.dcGainOffPole(offPole, err)
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
		return sys.dcGainOffPole(offPole, err)
	}
	gain := mat.NewDense(p, m, nil)
	gain.Mul(sys.C, &X)
	if sys.D != nil {
		gain.Add(gain, sys.D)
	}
	return gain, nil
}

// dcGainOffPole resolves a delay-free explicit model whose DC solve failed.
// With offPole set it does so as FreqResponse does: with no computed pole
// near DC (poleSpacing) the double-double solve decides, so an
// ill-conditioned but nonsingular pencil keeps its finite gain, even at an
// exact zero pivot of the double LU. A zero row or column of A − s0·I is an
// exact pole at DC and skips the pole search. Otherwise, or where the
// double-double solve is singular too, the transfer limit finds the
// integrators.
func (sys *System) dcGainOffPole(offPole bool, solveErr error) (*mat.Dense, error) {
	s0 := complex(0, 0)
	if sys.IsDiscrete() {
		s0 = 1
	}
	if _, ok := solveErr.(mat.Condition); offPole && ok && !hasZeroLine(sys.A, real(s0)) {
		_, m, p := sys.Dims()
		dst := make([]complex128, p*m)
		if sp, err := poleSpacing(sys, s0); err == nil && !sp.near && ddResponseInto(sys, pointAt(s0), dst) {
			data := make([]float64, p*m)
			for k, v := range dst {
				data[k] = real(v)
			}
			return mat.NewDense(p, m, data), nil
		}
	}
	return sys.dcGainByTransferFunctionLimit(offPole)
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
func (sys *System) dcGainOfDelayFreeExplicit(evalErr error, offPole bool) (*mat.Dense, error) {
	reduced, err := sys.ToExplicit()
	if err != nil {
		return nil, evalErr
	}
	if reduced.HasInternalDelay() {
		if reduced, err = reduced.ZeroDelayApprox(); err != nil {
			return nil, evalErr
		}
	}
	return reduced.dcGain(offPole)
}

// dcGainByTransferFunctionLimit rescales each input and output to the norm
// of A: the singular-mode and controllability tolerances are relative to
// ‖A‖, so a tiny but nonzero B or C would otherwise be dropped as rounding
// and an integrator's infinite gain reported as 0. Before the transfer
// limit it drops, per entry, states structurally disconnected from that
// input or output: the limit's root tests use an absolute coefficient
// floor, so an exact integrator–zero cancellation would otherwise hinge on
// the scale of B and C.
func (sys *System) dcGainByTransferFunctionLimit(offPole bool) (*mat.Dense, error) {
	if gain, ok := sys.dcGainTimeScaled(offPole); ok {
		return gain, nil
	}
	scaled, inScale, outScale := sys.dcGainIOScaled()
	gain, ok := scaled.dcGainByDecoupledSingularModes()
	if !ok {
		if gain, ok, err := sys.dcGainByStructuralEntries(offPole); ok {
			return gain, err
		}
		if gain, ok := sys.dcGainByBlockLaurent(); ok {
			return gain, nil
		}
		res, err := scaled.rationalTransferFunction(nil)
		if err != nil {
			return nil, fmt.Errorf("DCGain: %w", err)
		}
		gain = dcGainFromTransferFunction(res.TF)
	}
	if scaled == sys {
		return gain, nil
	}
	raw := gain.RawMatrix()
	for i, so := range outScale {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for j, si := range inScale {
			row[j] /= so
			row[j] /= si
		}
	}
	if sys.D != nil {
		gain.Add(gain, sys.D)
	}
	return gain, nil
}

// dcGainTimeScaled returns, for a continuous sys with ‖A‖ < 1/2, the
// gain of (2ᵏA, B, C, 0), ‖2ᵏA‖ in [1/2, 1), times 2ᵏ plus D: an exact
// rescaling of s that keeps pole orders at 0 and Inf signs, where the
// transfer limit's absolute coefficient floor of 1 would otherwise swamp
// numerators of order ‖A‖ⁿ. B stays as given so roundoff in the transfer
// numerator keeps its scale. A discrete pole at z = 1 forces ‖A‖ ≥ 1.
func (sys *System) dcGainTimeScaled(offPole bool) (*mat.Dense, bool) {
	if !sys.IsContinuous() {
		return nil, false
	}
	na := denseNorm(sys.A)
	if !(na > 0 && na < 0.5) {
		return nil, false
	}
	_, m, p := sys.Dims()
	_, e := math.Frexp(na)
	f := math.Ldexp(1, -e)
	a := mat.DenseCopyOf(sys.A)
	a.Scale(f, a)
	ts := &System{A: a, B: sys.B, C: sys.C, D: mat.NewDense(p, m, nil), Dt: sys.Dt}
	gain, err := ts.dcGainByTransferFunctionLimit(offPole)
	if err != nil {
		return nil, false
	}
	gain.Scale(f, gain)
	if sys.D != nil {
		gain.Add(gain, sys.D)
	}
	return gain, true
}

// dcGainByStructuralEntries evaluates each entry on the states reachable
// from its input and observable from its output by the exact sparsity of A,
// B and C (MATLAB sminreal per channel). ok is false when no entry loses a
// state, so the reduced channels never recurse back into it.
func (sys *System) dcGainByStructuralEntries(offPole bool) (*mat.Dense, bool, error) {
	n, m, p := sys.Dims()
	aRaw, bRaw, cRaw := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	reach := make([][]bool, m)
	for j := range m {
		r := make([]bool, n)
		for k := range n {
			r[k] = bRaw.Data[k*bRaw.Stride+j] != 0
		}
		structuralClosure(aRaw, r, false)
		reach[j] = r
	}
	obs := make([][]bool, p)
	for i := range p {
		o := make([]bool, n)
		for k := range n {
			o[k] = cRaw.Data[i*cRaw.Stride+k] != 0
		}
		structuralClosure(aRaw, o, true)
		obs[i] = o
	}
	reduced := false
	for i := range p {
		for j := range m {
			for k := range n {
				reduced = reduced || !reach[j][k] || !obs[i][k]
			}
		}
	}
	if !reduced {
		return nil, false, nil
	}
	gain := denseCopySafe(sys.D, p, m)
	keep := make([]int, 0, n)
	for i := range p {
		for j := range m {
			keep = keep[:0]
			for k := range n {
				if reach[j][k] && obs[i][k] {
					keep = append(keep, k)
				}
			}
			r := len(keep)
			if r == 0 {
				continue
			}
			ar := mat.NewDense(r, r, nil)
			br := mat.NewDense(r, 1, nil)
			cr := mat.NewDense(1, r, nil)
			arRaw, brData, crData := ar.RawMatrix(), br.RawMatrix().Data, cr.RawMatrix().Data
			for ii, k := range keep {
				row := aRaw.Data[k*aRaw.Stride:]
				for jj, l := range keep {
					arRaw.Data[ii*arRaw.Stride+jj] = row[l]
				}
				brData[ii] = bRaw.Data[k*bRaw.Stride+j]
				crData[ii] = cRaw.Data[i*cRaw.Stride+k]
			}
			sub := &System{A: ar, B: br, C: cr, D: mat.NewDense(1, 1, nil), Dt: sys.Dt}
			g, err := sub.dcGain(offPole)
			if err != nil {
				return nil, true, err
			}
			gain.Set(i, j, gain.At(i, j)+g.At(0, 0))
		}
	}
	return gain, true, nil
}

// dcGainByBlockLaurent expands G(x0+σ) = D + C(σI + P)⁻¹B, P = x0·I − A,
// in Laurent series by back-substitution over the strongly connected
// components of A's exact sparsity, which put A in block triangular form.
// A 1×1 component with A_kk = x0 exactly is a pole at x0 of the float
// model, counted exactly; exactly zero couplings stay zero, so the sign of
// an entry's leading coefficient does not hinge on roundoff in
// transfer-function coefficients. Given such a pole, a larger block that a
// regular solve would reject has its eigenvalues nearest x0 deflated to x0
// (deflateBlockAt), but only once the expansion on the model as given has
// declined: deflation moves poles by up to √ε·‖A_KK‖, which can raise the
// pole order at x0 and flip an Inf sign the float model fixes. A larger
// block with x0·I − A_KK exactly singular is a pole at x0 whose residue may
// cancel exactly, which no deflation resolves: with n ≤ exactDetMaxOrder
// the gain is then the exact limit (exactDCGainLimit); otherwise such a
// block always goes to deflation. An entry is ±∞ by its most
// negative order coefficient above 100·n·ε of its magnitude scale, else the
// σ⁰ coefficient. The deflated model's magnitudes are first those of the
// sums that formed it (laurentAbs); only if that cannot decide, and no
// block was exactly singular (its exact cancellations would be misjudged),
// are they taken as |entries|: an Inf sign may then rest on transform
// roundoff, but the exact pole still makes the entry infinite. ok is false
// with no exact pole at x0, a block deflation cannot settle, or a finite
// entry without three correct digits, leaving those to the transfer limit.
func (sys *System) dcGainByBlockLaurent() (*mat.Dense, bool) {
	n, _, _ := sys.Dims()
	x0 := 0.0
	if sys.IsDiscrete() {
		x0 = 1
	}
	if n <= exactDetMaxOrder {
		aRaw := sys.A.RawMatrix()
		if hasExactSingularBlock(aRaw, tarjanBlocks(aRaw, n), x0) {
			return exactDCGainLimit(sys.A, sys.B, sys.C, sys.D, x0), true
		}
	}
	for _, mode := range []laurentMode{laurentAsGiven, laurentDeflate, laurentDeflateLoose} {
		if g, ok := sys.blockLaurentGain(sys.A, sys.B, sys.C, nil, mode); ok {
			return g, true
		}
	}
	return nil, false
}

type laurentMode int

const (
	laurentAsGiven laurentMode = iota
	laurentDeflate
	laurentDeflateLoose
)

// laurentAbs holds, per entry of a deflated model, the magnitude of the
// terms summed to form it (|A|·|Z| and so on), so roundoff left by an exact
// cancellation is weighed against what cancelled rather than against itself.
type laurentAbs struct{ a, b, c *mat.Dense }

func absDense(m *mat.Dense) *mat.Dense {
	r := mat.DenseCopyOf(m)
	raw := r.RawMatrix()
	for i := range raw.Rows {
		row := raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols]
		for j, v := range row {
			row[j] = math.Abs(v)
		}
	}
	return r
}

// blockLaurentGain takes abs nil for |A|, |B|, |C|.
func (sys *System) blockLaurentGain(A, B, C *mat.Dense, abs *laurentAbs, mode laurentMode) (*mat.Dense, bool) {
	deflate := mode != laurentAsGiven
	n, m, p := sys.Dims()
	x0 := 0.0
	if sys.IsDiscrete() {
		x0 = 1
	}
	aRaw := A.RawMatrix()
	topo := aRaw
	if abs != nil {
		topo = abs.a.RawMatrix()
	}
	blocks := tarjanBlocks(topo, n)
	blockOf := make([]int, n)
	ns := 0
	for bi, blk := range blocks {
		for _, k := range blk {
			blockOf[k] = bi
		}
		if len(blk) == 1 && aRaw.Data[blk[0]*aRaw.Stride+blk[0]] == x0 {
			ns++
		}
	}
	if ns == 0 {
		return nil, false
	}
	type regular struct {
		lu     mat.LU
		invNrm float64
	}
	regs := make([]*regular, len(blocks))
	var illCond [][]int
	for bi, blk := range blocks {
		r := len(blk)
		if r == 1 && aRaw.Data[blk[0]*aRaw.Stride+blk[0]] == x0 {
			continue
		}
		pk := mat.NewDense(r, r, nil)
		pkRaw := pk.RawMatrix()
		for ii, k := range blk {
			for jj, l := range blk {
				v := -aRaw.Data[k*aRaw.Stride+l]
				if ii == jj {
					v += x0
				}
				pkRaw.Data[ii*pkRaw.Stride+jj] = v
			}
		}
		reg := &regular{}
		reg.lu.Factorize(pk)
		c := reg.lu.Cond()
		singular := math.IsInf(c, 0) || math.IsNaN(c) ||
			(!(c <= 1e12) && r <= exactDetMaxOrder && exactBlockSingularAt(aRaw, blk, x0))
		if singular && mode != laurentDeflate {
			return nil, false
		}
		if singular || deflate && r > 1 && !(c <= mat.ConditionTolerance) {
			illCond = append(illCond, blk)
			continue
		}
		reg.invNrm = c / mat.Norm(pk, 1)
		regs[bi] = reg
	}
	if len(illCond) > 0 {
		A, B, C = mat.DenseCopyOf(A), mat.DenseCopyOf(B), mat.DenseCopyOf(C)
		abs = &laurentAbs{absDense(A), absDense(B), absDense(C)}
		for _, blk := range illCond {
			if !deflateBlockAt(A, B, C, abs, blk, x0) {
				return nil, false
			}
		}
		if mode == laurentDeflateLoose {
			abs = nil
		}
		return sys.blockLaurentGain(A, B, C, abs, laurentAsGiven)
	}
	if deflate {
		return nil, false
	}

	nord := 2*ns + 1
	w := make([]float64, 2*nord*n*m)
	w, mag := w[:nord*n*m], w[nord*n*m:]
	at := func(s []float64, ord, k int) []float64 {
		o := ((ord+ns)*n + k) * m
		return s[o : o+m]
	}
	bRaw := B.RawMatrix()
	absA, absB := aRaw, bRaw
	if abs != nil {
		absA, absB = abs.a.RawMatrix(), abs.b.RawMatrix()
	}
	var sol, tmp *mat.Dense
	for bi, blk := range blocks {
		r := len(blk)
		rd, rm := make([]float64, r*nord*m), make([]float64, r*nord*m)
		for t := range nord {
			ord := t - ns
			for ii, k := range blk {
				row, mrow := rd[(t*r+ii)*m:(t*r+ii+1)*m], rm[(t*r+ii)*m:(t*r+ii+1)*m]
				if ord == 0 {
					for j := range m {
						row[j] = bRaw.Data[k*bRaw.Stride+j]
						mrow[j] = math.Abs(absB.Data[k*absB.Stride+j])
					}
				}
				for l := range n {
					a, am := aRaw.Data[k*aRaw.Stride+l], math.Abs(absA.Data[k*absA.Stride+l])
					if am == 0 || blockOf[l] == bi {
						continue
					}
					wl, ml := at(w, ord, l), at(mag, ord, l)
					for j := range m {
						row[j] += a * wl[j]
						mrow[j] += am * ml[j]
					}
				}
			}
		}
		reg := regs[bi]
		if reg == nil {
			k := blk[0]
			for t := range nord - 1 {
				copy(at(w, t-ns, k), rd[(t+1)*m:(t+2)*m])
				copy(at(mag, t-ns, k), rm[(t+1)*m:(t+2)*m])
			}
			continue
		}
		if sol == nil || sol.RawMatrix().Rows != r {
			sol, tmp = mat.NewDense(r, m, nil), mat.NewDense(r, m, nil)
		}
		td := tmp.RawMatrix().Data
		for t := range nord {
			ord := t - ns
			for ii, k := range blk {
				copy(td[ii*m:(ii+1)*m], rd[(t*r+ii)*m:(t*r+ii+1)*m])
				if t > 0 {
					prev := at(w, ord-1, k)
					for j := range m {
						td[ii*m+j] -= prev[j]
					}
				}
			}
			_ = reg.lu.SolveTo(sol, false, tmp)
			sd := sol.RawMatrix().Data
			for j := range m {
				bound := 0.0
				for ii, k := range blk {
					v := rm[(t*r+ii)*m+j]
					if t > 0 {
						v += at(mag, ord-1, k)[j]
					}
					bound = math.Max(bound, v)
				}
				bound *= reg.invNrm
				for ii, k := range blk {
					at(w, ord, k)[j] = sd[ii*m+j]
					at(mag, ord, k)[j] = math.Abs(sd[ii*m+j]) + bound
				}
			}
		}
	}

	gain := denseCopySafe(sys.D, p, m)
	cRaw := C.RawMatrix()
	absC := cRaw
	if abs != nil {
		absC = abs.c.RawMatrix()
	}
	tol := 100 * float64(n) * eps()
	for i := range p {
		crow := cRaw.Data[i*cRaw.Stride : i*cRaw.Stride+n]
		cmag := absC.Data[i*absC.Stride : i*absC.Stride+n]
		for j := range m {
			for ord := -ns; ord <= 0; ord++ {
				g, bound := 0.0, 0.0
				for k, c := range crow {
					if cm := math.Abs(cmag[k]); cm != 0 {
						g += c * at(w, ord, k)[j]
						bound += cm * at(mag, ord, k)[j]
					}
				}
				if ord == 0 {
					v := gain.At(i, j) + g
					if tol*bound > 1e-3*math.Abs(v) {
						return nil, false
					}
					gain.Set(i, j, v)
					break
				}
				if math.Abs(g) > tol*bound {
					gain.Set(i, j, math.Inf(signInt(g)))
					break
				}
			}
		}
	}
	return gain, true
}

// hasExactSingularBlock reports whether a block of a, at most
// exactDetMaxOrder states, has x0·I − A_KK exactly singular: a pole at x0
// of the float model that no 1×1 block shows.
func hasExactSingularBlock(a blas64.General, blocks [][]int, x0 float64) bool {
	for _, blk := range blocks {
		r := len(blk)
		if r < 2 || r > exactDetMaxOrder {
			continue
		}
		pk := mat.NewDense(r, r, nil)
		for ii, k := range blk {
			for jj, l := range blk {
				v := -a.Data[k*a.Stride+l]
				if ii == jj {
					v += x0
				}
				pk.Set(ii, jj, v)
			}
		}
		var lu mat.LU
		lu.Factorize(pk)
		if !(lu.Cond() <= 1e12) && exactBlockSingularAt(a, blk, x0) {
			return true
		}
	}
	return false
}

// exactBlockSingularAt reports whether A_KK − x0·I, K = blk, is exactly
// singular.
func exactBlockSingularAt(a blas64.General, blk []int, x0 float64) bool {
	r := len(blk)
	ak := blas64.General{Rows: r, Cols: r, Stride: r, Data: make([]float64, r*r)}
	for ii, k := range blk {
		for jj, l := range blk {
			ak.Data[ii*r+jj] = a.Data[k*a.Stride+l]
		}
	}
	return exactShiftedDetIsZero(ak, x0)
}

// exactDetMaxOrder caps the blocks whose determinant exactDetIsZero
// computes; larger ones keep the LU condition estimate alone.
const exactDetMaxOrder = 16

// exactDetIsZero reports whether the square matrix a is exactly singular,
// by fraction-free (Bareiss) elimination over the integers its entries
// become once scaled by a common power of two.
func exactDetIsZero(a blas64.General) bool {
	return exactShiftedDetIsZero(a, 0)
}

// exactShiftedDetIsZero reports whether a − x0·I is exactly singular, with
// the shift applied in exact arithmetic.
func exactShiftedDetIsZero(a blas64.General, x0 float64) bool {
	r := a.Rows
	emin := math.MaxInt
	if x0 != 0 {
		_, emin = math.Frexp(x0)
	}
	for i := range r {
		for _, v := range a.Data[i*a.Stride : i*a.Stride+r] {
			if v != 0 {
				_, e := math.Frexp(v)
				emin = min(emin, e)
			}
		}
	}
	if emin == math.MaxInt {
		return true
	}
	scaled := func(v float64) *big.Int {
		f, e := math.Frexp(v)
		x := big.NewInt(int64(math.Ldexp(f, 53)))
		return x.Lsh(x, uint(e-emin))
	}
	m := make([]*big.Int, r*r)
	for i := range r {
		for j, v := range a.Data[i*a.Stride : i*a.Stride+r] {
			m[i*r+j] = scaled(v)
			if i == j && x0 != 0 {
				m[i*r+j].Sub(m[i*r+j], scaled(x0))
			}
		}
	}
	return bareissDet(m, r).Sign() == 0
}

// bareissDet returns the determinant of the r×r integer matrix m by
// fraction-free (Bareiss) elimination, overwriting m.
func bareissDet(m []*big.Int, r int) *big.Int {
	prev := big.NewInt(1)
	t := new(big.Int)
	neg := false
	for k := range r - 1 {
		if m[k*r+k].Sign() == 0 {
			piv := -1
			for i := k + 1; i < r; i++ {
				if m[i*r+k].Sign() != 0 {
					piv = i
					break
				}
			}
			if piv < 0 {
				return new(big.Int)
			}
			for j := range r {
				m[k*r+j], m[piv*r+j] = m[piv*r+j], m[k*r+j]
			}
			neg = !neg
		}
		for i := k + 1; i < r; i++ {
			for j := k + 1; j < r; j++ {
				x := m[i*r+j]
				x.Mul(x, m[k*r+k])
				x.Sub(x, t.Mul(m[i*r+k], m[k*r+j]))
				x.Quo(x, prev)
			}
		}
		prev = m[k*r+k]
	}
	d := new(big.Int).Set(m[r*r-1])
	if neg {
		d.Neg(d)
	}
	return d
}

// exactDCGainLimit returns, entry by entry, the exact limit σ → 0+ of
// D + C(σI + P)⁻¹B, P = x0·I − A, on the float model: ±Inf by the sign of
// the leading Laurent coefficient at a pole, else the rounded finite value.
// With A, x0 scaled to integers by a common power of two s (σ̂ = s·σ), the
// numerator Ĉ adj(σ̂I + P̂) B̂ and det(σ̂I + P̂) are polynomials of degree
// at most n, interpolated from n+1 integer nodes σ̂ = k where σ̂I + P̂ is
// nonsingular: one fraction-free solve there gives det and adj·B̂ for every
// entry at once.
func exactDCGainLimit(A, B, C, D *mat.Dense, x0 float64) *mat.Dense {
	n, m := B.Dims()
	p, _ := C.Dims()
	aRaw, bRaw, cRaw := A.RawMatrix(), B.RawMatrix(), C.RawMatrix()
	minExp := func(r blas64.General, e0 int) int {
		for i := range r.Rows {
			for _, v := range r.Data[i*r.Stride : i*r.Stride+r.Cols] {
				if v != 0 {
					_, e := math.Frexp(v)
					e0 = min(e0, e)
				}
			}
		}
		if e0 == math.MaxInt {
			return 0
		}
		return e0
	}
	seedA := math.MaxInt
	if x0 != 0 {
		_, seedA = math.Frexp(x0)
	}
	ea, eb, ec := minExp(aRaw, seedA), minExp(bRaw, math.MaxInt), minExp(cRaw, math.MaxInt)
	scaled := func(v float64, e0 int) *big.Int {
		f, e := math.Frexp(v)
		x := big.NewInt(int64(math.Ldexp(f, 53)))
		return x.Lsh(x, uint(e-e0))
	}
	pHat := make([]*big.Int, n*n)
	for i := range n {
		for j := range n {
			v := scaled(aRaw.Data[i*aRaw.Stride+j], ea)
			v.Neg(v)
			if i == j && x0 != 0 {
				v.Add(v, scaled(x0, ea))
			}
			pHat[i*n+j] = v
		}
	}
	bHat := make([]*big.Int, n*m)
	for i := range n {
		for j := range m {
			bHat[i*m+j] = scaled(bRaw.Data[i*bRaw.Stride+j], eb)
		}
	}
	cHat := make([]*big.Int, p*n)
	for i := range p {
		for j := range n {
			cHat[i*n+j] = scaled(cRaw.Data[i*cRaw.Stride+j], ec)
		}
	}

	w := n + m
	work := make([]*big.Int, n*w)
	for i := range work {
		work[i] = new(big.Int)
	}
	y := make([]*big.Int, n*m)
	for i := range y {
		y[i] = new(big.Int)
	}
	t := new(big.Int)
	nodes := make([]int64, 0, n+1)
	dVals := make([]*big.Int, 0, n+1)
	nVals := make([][]*big.Int, p*m)
	for k := int64(0); len(nodes) < n+1; k++ {
		for ii := range n {
			for jj := range n {
				work[ii*w+jj].Set(pHat[ii*n+jj])
			}
			work[ii*w+ii].Add(work[ii*w+ii], big.NewInt(k))
			for j := range m {
				work[ii*w+n+j].Set(bHat[ii*m+j])
			}
		}
		det, ok := bareissSolve(work, y, n, m, t)
		if !ok {
			continue
		}
		nodes = append(nodes, k)
		dVals = append(dVals, det)
		for i := range p {
			for j := range m {
				v := new(big.Int)
				for l, c := range cHat[i*n : i*n+n] {
					if c.Sign() != 0 {
						v.Add(v, t.Mul(c, y[l*m+j]))
					}
				}
				nVals[i*m+j] = append(nVals[i*m+j], v)
			}
		}
	}
	od, dLow := lowestNonzero(newtonAscending(nodes, dVals))

	// G = D + 2^(53−ea) / 2^(53−eb) / 2^(53−ec) · N/d.
	shift := 53 - ea - (53 - eb) - (53 - ec)
	gain := mat.NewDense(p, m, nil)
	for i := range p {
		for j := range m {
			on, nLow := lowestNonzero(newtonAscending(nodes, nVals[i*m+j]))
			dij := 0.0
			if D != nil {
				dij = D.At(i, j)
			}
			switch {
			case on < 0 || on > od:
				gain.Set(i, j, dij)
			case on < od:
				gain.Set(i, j, math.Inf(nLow.Sign()*dLow.Sign()))
			default:
				v := new(big.Rat).Quo(nLow, dLow)
				sc := new(big.Rat).SetInt(new(big.Int).Lsh(big.NewInt(1), uint(max(shift, -shift))))
				if shift >= 0 {
					v.Mul(v, sc)
				} else {
					v.Quo(v, sc)
				}
				v.Add(v, new(big.Rat).SetFloat64(dij))
				f, _ := v.Float64()
				gain.Set(i, j, f)
			}
		}
	}
	return gain
}

// bareissSolve takes the n×(n+m) integer matrix a = [M | R], overwriting
// it, and when M is nonsingular returns det M and y = adj(M)·R = det M·M⁻¹R
// (n×m) by fraction-free elimination and back-substitution, every division
// exact; ok is false when M is singular. t is scratch.
func bareissSolve(a, y []*big.Int, n, m int, t *big.Int) (det *big.Int, ok bool) {
	w := n + m
	prev := big.NewInt(1)
	neg := false
	for k := range n {
		if a[k*w+k].Sign() == 0 {
			piv := -1
			for i := k + 1; i < n; i++ {
				if a[i*w+k].Sign() != 0 {
					piv = i
					break
				}
			}
			if piv < 0 {
				return nil, false
			}
			for j := range w {
				a[k*w+j], a[piv*w+j] = a[piv*w+j], a[k*w+j]
			}
			neg = !neg
		}
		for i := k + 1; i < n; i++ {
			for j := k + 1; j < w; j++ {
				x := a[i*w+j]
				x.Mul(x, a[k*w+k])
				x.Sub(x, t.Mul(a[i*w+k], a[k*w+j]))
				x.Quo(x, prev)
			}
		}
		prev = a[k*w+k]
	}
	dn := prev
	for i := n - 1; i >= 0; i-- {
		for j := range m {
			v := y[i*m+j]
			v.Mul(dn, a[i*w+n+j])
			for l := i + 1; l < n; l++ {
				if c := a[i*w+l]; c.Sign() != 0 {
					v.Sub(v, t.Mul(c, y[l*m+j]))
				}
			}
			v.Quo(v, a[i*w+i])
		}
	}
	det = new(big.Int).Set(dn)
	if neg {
		det.Neg(det)
		for _, v := range y {
			v.Neg(v)
		}
	}
	return det, true
}

// newtonAscending returns the ascending coefficients of the polynomial of
// degree < len(y) taking the values y at the distinct nodes xs.
func newtonAscending(xs []int64, y []*big.Int) []*big.Rat {
	q := len(y)
	c := make([]*big.Rat, q)
	for k, v := range y {
		c[k] = new(big.Rat).SetInt(v)
	}
	for j := 1; j < q; j++ {
		for k := q - 1; k >= j; k-- {
			c[k].Sub(c[k], c[k-1])
			c[k].Quo(c[k], big.NewRat(xs[k]-xs[k-j], 1))
		}
	}
	poly := make([]*big.Rat, q)
	for k := range poly {
		poly[k] = new(big.Rat)
	}
	poly[0].Set(c[q-1])
	t := new(big.Rat)
	for k := q - 2; k >= 0; k-- {
		xk := big.NewRat(xs[k], 1)
		for d := q - 1; d >= 1; d-- {
			poly[d].Sub(poly[d-1], t.Mul(poly[d], xk))
		}
		poly[0].Sub(c[k], t.Mul(poly[0], xk))
	}
	return poly
}

// lowestNonzero returns the order and value of the lowest nonzero
// coefficient of an ascending polynomial, order −1 when it is identically 0.
func lowestNonzero(c []*big.Rat) (int, *big.Rat) {
	for k, v := range c {
		if v.Sign() != 0 {
			return k, v
		}
	}
	return -1, nil
}

// deflateBlockAt applies x_K = Z·y to the states blk of (A, B, C), with
// Z the real Schur vectors of A_KK ordered so the eigenvalues nearest x0
// lead, and updates abs with the magnitudes of the sums it forms. Every
// eigenvalue within guard = √ε·max(‖A_KK‖, 1) of x0 is set to x0 exactly,
// its strictly lower part zeroed; false when a zeroed or shifted entry
// exceeds guard (a 2×2 bump of a defective pair) or x0·I − T on the rest
// is worse conditioned than a regular solve allows.
func deflateBlockAt(A, B, C *mat.Dense, abs *laurentAbs, blk []int, x0 float64) bool {
	r := len(blk)
	_, m := B.Dims()
	p, _ := C.Dims()
	n, _ := A.Dims()
	aRaw := A.RawMatrix()
	t := make([]float64, r*r)
	for ii, k := range blk {
		for jj, l := range blk {
			t[ii*r+jj] = aRaw.Data[k*aRaw.Stride+l]
		}
	}
	guard := math.Sqrt(eps()) * math.Max(mat.Norm(mat.NewDense(r, r, t), math.Inf(1)), 1)
	z := make([]float64, r*r)
	wr, wi := make([]float64, r), make([]float64, r)
	bwork := make([]bool, r)
	query := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, r, t, r, wr, wi, z, r, query, -1, bwork)
	work := make([]float64, int(query[0]))
	if _, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil, r, t, r, wr, wi, z, r, work, len(work), bwork); !ok {
		return false
	}
	dists := make([]float64, r)
	for i := range r {
		dists[i] = cmplx.Abs(complex(wr[i]-x0, wi[i]))
	}
	lim := -1.0
	for _, v := range dists {
		if v <= guard {
			lim = max(lim, v)
		}
	}
	if lim < 0 {
		return false
	}
	tc, zc := slices.Clone(t), slices.Clone(z)
	d, err := orderSchurGroupFirst(tc, zc, r, func(e complex128) bool { return cmplx.Abs(e-complex(x0, 0)) <= lim })
	if err != nil {
		return false
	}
	if d < r {
		rest := mat.NewDense(r-d, r-d, nil)
		rr := rest.RawMatrix()
		for i := d; i < r; i++ {
			for j := d; j < r; j++ {
				v := -tc[i*r+j]
				if i == j {
					v += x0
				}
				rr.Data[(i-d)*rr.Stride+j-d] = v
			}
		}
		var lu mat.LU
		lu.Factorize(rest)
		if !(lu.Cond() <= mat.ConditionTolerance) {
			return false
		}
	}
	for i := range d {
		for j := range i {
			if math.Abs(tc[i*r+j]) > guard {
				return false
			}
			tc[i*r+j] = 0
		}
		if math.Abs(tc[i*r+i]-x0) > guard {
			return false
		}
		tc[i*r+i] = x0
	}
	absA, absB, absC := abs.a.RawMatrix(), abs.b.RawMatrix(), abs.c.RawMatrix()
	for ii, k := range blk {
		for jj, l := range blk {
			aRaw.Data[k*aRaw.Stride+l] = tc[ii*r+jj]
			absA.Data[k*absA.Stride+l] = math.Abs(tc[ii*r+jj])
		}
	}
	inK := make([]bool, n)
	for _, k := range blk {
		inK[k] = true
	}
	buf, mbuf := make([]float64, r), make([]float64, r)
	rowTimesZ := func(v, mv blas64.General, i int) {
		for jj := range r {
			s, ms := 0.0, 0.0
			for ii, k := range blk {
				s += v.Data[i*v.Stride+k] * zc[ii*r+jj]
				ms += mv.Data[i*mv.Stride+k] * math.Abs(zc[ii*r+jj])
			}
			buf[jj], mbuf[jj] = s, ms
		}
		for jj, k := range blk {
			v.Data[i*v.Stride+k], mv.Data[i*mv.Stride+k] = buf[jj], mbuf[jj]
		}
	}
	ztTimesCol := func(v, mv blas64.General, j int) {
		for ii := range r {
			s, ms := 0.0, 0.0
			for jj, k := range blk {
				s += zc[jj*r+ii] * v.Data[k*v.Stride+j]
				ms += math.Abs(zc[jj*r+ii]) * mv.Data[k*mv.Stride+j]
			}
			buf[ii], mbuf[ii] = s, ms
		}
		for ii, k := range blk {
			v.Data[k*v.Stride+j], mv.Data[k*mv.Stride+j] = buf[ii], mbuf[ii]
		}
	}
	for l := range n {
		if !inK[l] {
			rowTimesZ(aRaw, absA, l)
			ztTimesCol(aRaw, absA, l)
		}
	}
	for j := range m {
		ztTimesCol(B.RawMatrix(), absB, j)
	}
	cRaw := C.RawMatrix()
	for i := range p {
		rowTimesZ(cRaw, absC, i)
	}
	return true
}

// tarjanBlocks returns the strongly connected components of the graph with
// an edge k → l when A[k,l] ≠ 0 (x_l drives x_k), each emitted after every
// component it depends on.
func tarjanBlocks(a blas64.General, n int) [][]int {
	index := make([]int, n)
	low := make([]int, n)
	onStack := make([]bool, n)
	for i := range index {
		index[i] = -1
	}
	var stack []int
	var blocks [][]int
	next := 0
	type frame struct{ v, l int }
	for root := range n {
		if index[root] >= 0 {
			continue
		}
		call := []frame{{root, 0}}
		index[root], low[root] = next, next
		next++
		stack = append(stack, root)
		onStack[root] = true
		for len(call) > 0 {
			f := &call[len(call)-1]
			v := f.v
			if f.l < n {
				l := f.l
				f.l++
				if l == v || a.Data[v*a.Stride+l] == 0 {
					continue
				}
				if index[l] < 0 {
					index[l], low[l] = next, next
					next++
					stack = append(stack, l)
					onStack[l] = true
					call = append(call, frame{l, 0})
				} else if onStack[l] {
					low[v] = min(low[v], index[l])
				}
				continue
			}
			call = call[:len(call)-1]
			if len(call) > 0 {
				u := call[len(call)-1].v
				low[u] = min(low[u], low[v])
			}
			if low[v] == index[v] {
				var blk []int
				for {
					x := stack[len(stack)-1]
					stack = stack[:len(stack)-1]
					onStack[x] = false
					blk = append(blk, x)
					if x == v {
						break
					}
				}
				blocks = append(blocks, blk)
			}
		}
	}
	return blocks
}

// structuralClosure grows set to every state driven by a member through a
// nonzero A entry (A[i,k] ≠ 0: x_k drives x_i), or with reverse to every
// state driving a member.
func structuralClosure(a blas64.General, set []bool, reverse bool) {
	n := len(set)
	for changed := true; changed; {
		changed = false
		for i := range n {
			if set[i] {
				continue
			}
			for k := range n {
				v := a.Data[i*a.Stride+k]
				if reverse {
					v = a.Data[k*a.Stride+i]
				}
				if set[k] && v != 0 {
					set[i] = true
					changed = true
					break
				}
			}
		}
	}
}

// dcGainIOScaled returns sys with B, C scaled by ioPow2Scales and D
// zeroed; any smaller scale lets a small internal coupling push a transfer
// numerator under the limit's absolute coefficient floor. Models already at
// that scale are returned as they are, D included.
func (sys *System) dcGainIOScaled() (*System, []float64, []float64) {
	_, m, p := sys.Dims()
	inScale, outScale, unit := ioPow2Scales(sys.A, sys.B, sys.C)
	if unit {
		return sys, inScale, outScale
	}
	b, c := mat.DenseCopyOf(sys.B), mat.DenseCopyOf(sys.C)
	applyIOScales(b, c, inScale, outScale)
	return &System{A: sys.A, B: b, C: c, D: mat.NewDense(p, m, nil), Dt: sys.Dt}, inScale, outScale
}

// ioPow2Scales returns exact powers of two (capped at 2¹⁰²³) for each
// column j of B and row i of C, bringing each nonzero one whose max-abs
// entry has a smaller binary exponent than max(‖A‖, 1) up to about
// max(‖A‖, 1). unit reports that every scale is 1.
func ioPow2Scales(a, b, c *mat.Dense) (inScale, outScale []float64, unit bool) {
	n, _ := a.Dims()
	_, m := b.Dims()
	p, _ := c.Dims()
	scales := make([]float64, m+p)
	inScale, outScale = scales[:m], scales[m:]
	sbRaw, scRaw := b.RawMatrix(), c.RawMatrix()
	for k := range n {
		for j, v := range sbRaw.Data[k*sbRaw.Stride : k*sbRaw.Stride+m] {
			inScale[j] = math.Max(inScale[j], math.Abs(v))
		}
	}
	for i := range p {
		for _, v := range scRaw.Data[i*scRaw.Stride : i*scRaw.Stride+n] {
			outScale[i] = math.Max(outScale[i], math.Abs(v))
		}
	}
	target := math.Max(denseNorm(a), 1)
	_, targetExp := math.Frexp(target)
	finite := !math.IsInf(target, 0) && !math.IsNaN(target)
	unit = true
	for k, v := range scales {
		scales[k] = 1
		if !finite || v == 0 || !(v < target) {
			continue
		}
		if _, e := math.Frexp(v); e < targetExp {
			scales[k] = math.Ldexp(1, min(targetExp-e, 1023))
			unit = false
		}
	}
	return inScale, outScale, unit
}

// applyIOScales scales column j of b by inScale[j] and row i of c by
// outScale[i] in place.
func applyIOScales(b, c *mat.Dense, inScale, outScale []float64) {
	bRaw, cRaw := b.RawMatrix(), c.RawMatrix()
	for k := range bRaw.Rows {
		row := bRaw.Data[k*bRaw.Stride : k*bRaw.Stride+bRaw.Cols]
		for j, s := range inScale {
			row[j] *= s
		}
	}
	for i, s := range outScale {
		row := cRaw.Data[i*cRaw.Stride : i*cRaw.Stride+cRaw.Cols]
		for k := range row {
			row[k] *= s
		}
	}
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

// hasZeroLine reports whether a − shift·I has an exactly zero row or column.
func hasZeroLine(a *mat.Dense, shift float64) bool {
	raw := a.RawMatrix()
	n := raw.Rows
	for k := range n {
		row, col := true, true
		for j := 0; j < n && (row || col); j++ {
			r, c := raw.Data[k*raw.Stride+j], raw.Data[j*raw.Stride+k]
			if j == k {
				r -= shift
				c = r
			}
			row = row && r == 0
			col = col && c == 0
		}
		if row || col {
			return true
		}
	}
	return false
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

// Damp returns the natural frequency, damping ratio and time constant of
// every pole of sys, in the order of Poles; see DampInfo. A static gain has no
// poles and returns an empty slice.
func Damp(sys *System) ([]DampInfo, error) {
	if err := requireSystem("Damp", sys); err != nil {
		return nil, err
	}
	poles, err := sys.Poles()
	if err != nil {
		return nil, fmt.Errorf("Damp: %w", err)
	}

	result := make([]DampInfo, len(poles))
	for i, p := range poles {
		s := p
		if sys.IsDiscrete() {
			s = cmplx.Log(p) / complex(sys.Dt, 0)
		}
		wn := cmplx.Abs(s)
		var zeta, tau float64
		switch {
		case math.IsInf(wn, 1):
			zeta, tau = 1, 0
		case wn == 0:
			zeta, tau = -1, math.Inf(1)
		case real(s) == 0:
			zeta, tau = 0, math.Inf(1)
		default:
			zeta, tau = -real(s)/wn, -1/real(s)
		}
		result[i] = DampInfo{Pole: p, Wn: wn, Zeta: zeta, Tau: tau}
	}
	return result, nil
}

// Step returns the step response of each input channel. Descriptor models
// follow MATLAB: proper singular-E models are reduced to an explicit model
// plus feedthrough, improper ones return ErrImproperModel. A continuous
// model without internal delays whose full-rank E is too ill-conditioned
// for E⁻¹A (as in TransferFunction, with LU condition estimate 1e6 or more;
// for Impulse from 1e4) is simulated from its generalized Schur parts, so
// the response follows the poles of the pencil (A, E); as in MATLAB, its
// I/O delays shift the delay-free response. Continuous models
// with internal delays of one common length are sampled exactly (method of
// steps) while the chain fits its size and work budget; past it, and for
// other internal-delay models, they are simulated through the approximate
// ZOH discretization of the delay channels, as MATLAB does, with O(dt) error.
// A continuous descriptor with internal delays whose E is too
// ill-conditioned for E⁻¹A (LU condition estimate 1e6 or more) always takes
// that discretization, of its generalized Schur parts with the delay
// feedback kept (as C2D), not the exact chain: one exponential of E⁻¹A would
// lose its slow modes to cond(E)·ε.
//
// tFinal = 0 selects the horizon automatically from the poles and delays, as
// MATLAB step(sys); a negative or non-finite tFinal returns
// ErrInvalidArgument. A model without inputs or outputs returns
// ErrDimensionMismatch.
func Step(sys *System, tFinal float64) (*TimeResponse, error) {
	if err := requireTimeResponse("Step", sys, tFinal); err != nil {
		return nil, err
	}
	if ex, ok := stiffLFTExplicit(sys); ok {
		sys = ex
	} else {
		var err error
		if sys, err = sys.zeroStateResponseForm(); err != nil {
			return nil, fmt.Errorf("Step: %w", err)
		}
		if resp, ok, err := delayChainAuto(sys, tFinal, stepResponse); err != nil {
			return nil, fmt.Errorf("Step: %w", err)
		} else if ok {
			return resp, nil
		}
	}
	plan, err := prepareAutoTimeResponse(sys, tFinal, 0)
	if err != nil {
		return nil, fmt.Errorf("Step: %w", err)
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
// for one common delay length within the chain budget, else by the
// approximate ZOH discretization). Continuous models without internal
// delays are sampled exactly as C·e^{A·t}·B, shifted by their I/O delays.
// tFinal and models without inputs or outputs are handled as in Step.
func Impulse(sys *System, tFinal float64) (*TimeResponse, error) {
	if err := requireTimeResponse("Impulse", sys, tFinal); err != nil {
		return nil, err
	}
	if parts, ok := stiffContinuousDescriptorParts(sys); ok {
		resp, err := stiffDescriptorImpulse(sys, parts, tFinal)
		if err != nil {
			return nil, fmt.Errorf("Impulse: %w", err)
		}
		return resp, nil
	}
	sys, _, _, err := sys.timeResponseForm(nil)
	if err != nil {
		return nil, fmt.Errorf("Impulse: %w", err)
	}
	kind := impulseResponse
	if sys.IsContinuous() {
		if sys.LFT != nil && sys.HasInternalDelay() && !allZeroDense(sys.LFT.D21) {
			return nil, fmt.Errorf("Impulse: %w", ErrInternalDelayImpulse)
		}
		if resp, ok, err := delayChainAuto(sys, tFinal, impulseResponse); err != nil {
			return nil, fmt.Errorf("Impulse: %w", err)
		} else if ok {
			return resp, nil
		}
		if n, _, _ := sys.Dims(); n > 0 && sys.LFT == nil {
			resp, err := explicitImpulse(sys, tFinal)
			if err != nil {
				return nil, fmt.Errorf("Impulse: %w", err)
			}
			return resp, nil
		}
		derived, err := impulseAsStepModel(sys)
		if err != nil {
			return nil, fmt.Errorf("Impulse: %w", err)
		}
		sys, kind = derived, stepResponse
	}
	plan, err := prepareAutoTimeResponse(sys, tFinal, 0)
	if err != nil {
		return nil, fmt.Errorf("Impulse: %w", err)
	}

	resp, err := plan.allInputResponse(kind)
	if err != nil {
		return nil, fmt.Errorf("Impulse: %w", err)
	}
	return resp, nil
}

// zeroStateResponseForm is timeResponseForm without an initial state,
// except that a continuous descriptor with foldInaccurateParts stays a
// descriptor: its poles are then those of the pencil and C2D discretizes
// its parts instead of the LU fold E⁻¹A.
func (sys *System) zeroStateResponseForm() (*System, error) {
	if foldInaccurateContinuous(sys) {
		return sys, nil
	}
	out, _, _, err := sys.timeResponseForm(nil)
	return out, err
}

// foldInaccurateContinuous reports whether the continuous sys has
// foldInaccurateParts of its I/O-delay-free core (ioDelayFreeCore): MATLAB
// simulates a dss with I/O delays as the shifted delay-free response.
func foldInaccurateContinuous(sys *System) bool {
	if !sys.IsContinuous() {
		return false
	}
	_, ok := foldInaccurateParts(ioDelayFreeCore(sys))
	return ok
}

// explicitImpulse is the impulse response of the continuous explicit sys
// without internal delays, sampled exactly as C·e^{A·k·dt}·B, shifted by
// the I/O delays. The step form of impulseAsStepModel would instead cancel
// C·B against the state: with ‖A‖ far above the slow poles (a stiff mode,
// or the fold E⁻¹A of a mildly ill-conditioned E) the tail then carries
// rounding of C·B amplified by 1/(|λ_slow|·dt).
func explicitImpulse(sys *System, tFinal float64) (*TimeResponse, error) {
	t, dt, err := newTimeResponsePlanner(sys).grid(tFinal, 0)
	if err != nil {
		return nil, err
	}
	if sys.HasDelay() {
		return shiftedPartsImpulse(sys, []*System{sys}, t, dt, holdExp), nil
	}
	return sampledPartsImpulse(sys, []*System{sys}, t, dt)
}

// stiffContinuousDescriptorParts is stiffDescriptorParts of the
// I/O-delay-free core of a continuous model.
func stiffContinuousDescriptorParts(sys *System) ([]*System, bool) {
	if !sys.IsContinuous() {
		return nil, false
	}
	return stiffDescriptorParts(ioDelayFreeCore(sys))
}

// stiffDescriptorImpulse is the impulse response of sys on the automatic
// grid from the parts of stiffDescriptorParts (sampledPartsImpulse). Channel (i, j) of a model with
// I/O delays is the delay-free one shifted by its total delay τ_ij.
func stiffDescriptorImpulse(sys *System, parts []*System, tFinal float64) (*TimeResponse, error) {
	t, dt, err := newTimeResponsePlanner(sys).grid(tFinal, 0)
	if err != nil {
		return nil, err
	}
	if sys.HasDelay() {
		return shiftedPartsImpulse(sys, parts, t, dt, partExp), nil
	}
	return sampledPartsImpulse(sys, parts, t, dt)
}

// sampledPartsImpulse is the impulse response on the grid t of the sum of
// the delay-free explicit parts, each sampled exactly as the discrete
// impulse response of (A_d, A_d·B, C, C·B), A_d = e^{A·dt}.
func sampledPartsImpulse(sys *System, parts []*System, t []float64, dt float64) (*TimeResponse, error) {
	disc := make([]*System, len(parts))
	for i, part := range parts {
		d, err := part.discretizeZOH(dt)
		if err != nil {
			return nil, fmt.Errorf("auto-discretize: %w", err)
		}
		d.B.Mul(d.A, part.B)
		d.D.Mul(part.C, part.B)
		disc[i] = d
	}
	sim, err := parallelExplicit(disc)
	if err != nil {
		return nil, err
	}
	propagateIONames(sim, sys)
	plan := timeResponsePlan{original: sys, sim: sim, t: t, dt: dt, steps: len(t), wasContinuous: true}
	return plan.allInputResponse(impulseResponse)
}

// shiftedPartsImpulse is the impulse response of a model with I/O delays
// from its delay-free explicit parts: y_ij(k·dt) = C·e^{A(k·dt − τ_ij)}·B_j
// of each part, zero before τ_ij, with the exponentials of exp.
func shiftedPartsImpulse(sys *System, parts []*System, t []float64, dt float64, exp expFunc) *TimeResponse {
	_, m, p := sys.Dims()
	tau := effectiveIODelayMatrix(sys, p, m, true)
	Y := mat.NewDense(p*m, len(t), nil)
	rows, taus := make([]int, p), make([]float64, p)
	var ad mat.Dense
	for _, part := range parts {
		pn, _, _ := part.Dims()
		if pn == 0 {
			continue
		}
		exp(&ad, part.A, dt)
		for j := range m {
			x := mat.Col(nil, j, part.B)
			for i := range p {
				rows[i], taus[i] = j*p+i, tau.At(i, j)
			}
			addShiftedPartSamples(Y, rows, taus, part, &ad, x, dt, exp)
		}
	}
	return &TimeResponse{T: t, Y: Y, OutputName: copyStringSlice(sys.OutputName)}
}

// expFunc sets ad to e^{A·dt}.
type expFunc func(ad *mat.Dense, a *mat.Dense, dt float64)

// partExp sets ad to e^{A·dt}.
func partExp(ad *mat.Dense, a *mat.Dense, dt float64) {
	var scaled mat.Dense
	scaled.Scale(dt, a)
	ad.Reset()
	ad.Exp(&scaled)
}

// holdExp is partExp by stiffExp when A·dt is stiff, as the ZOH
// discretization takes it.
func holdExp(ad *mat.Dense, a *mat.Dense, dt float64) {
	if e, ok := stiffExp(a, dt); ok {
		ad.Reset()
		ad.CloneFrom(e)
		return
	}
	partExp(ad, a, dt)
}

// addShiftedPartSamples adds C_i·e^{A(k·dt − τ_i)}·x of the explicit part,
// ad = e^{A·dt}, to Y[rows[i], k] for every sample k·dt ≥ τ_i, i < len(rows),
// exactly; a τ_i within gridTol samples of the grid is on it.
func addShiftedPartSamples(Y *mat.Dense, rows []int, taus []float64, part *System, ad *mat.Dense, x []float64, dt float64, exp expFunc) {
	_, steps := Y.Dims()
	pn, _, _ := part.Dims()
	yRaw, aRaw, cRaw := Y.RawMatrix(), ad.RawMatrix(), part.C.RawMatrix()
	done := make([]bool, len(rows))
	cur, next := make([]float64, pn), make([]float64, pn)
	var group []int
	for g := range rows {
		if done[g] {
			continue
		}
		tau := taus[g]
		group = group[:0]
		for i := g; i < len(rows); i++ {
			if !done[i] && taus[i] == tau {
				group, done[i] = append(group, i), true
			}
		}
		k0 := max(int(math.Ceil(tau/dt-gridTol)), 0)
		if k0 >= steps {
			continue
		}
		copy(cur, x)
		if lead := float64(k0)*dt - tau; lead > gridTol*dt {
			var lag mat.Dense
			exp(&lag, part.A, lead)
			v := mat.NewVecDense(pn, next)
			v.MulVec(&lag, mat.NewVecDense(pn, x))
			cur, next = next, cur
		}
		for k := k0; k < steps; k++ {
			live := false
			for _, i := range group {
				row := cRaw.Data[i*cRaw.Stride : i*cRaw.Stride+pn]
				s := 0.0
				for j, v := range row {
					s += v * cur[j]
				}
				yRaw.Data[rows[i]*yRaw.Stride+k] += s
			}
			for i := range pn {
				row := aRaw.Data[i*aRaw.Stride : i*aRaw.Stride+pn]
				s := 0.0
				for j, v := range row {
					s += v * cur[j]
				}
				next[i] = s
				live = live || s != 0
			}
			if !live {
				break
			}
			cur, next = next, cur
		}
	}
}

// impulseAsStepModel returns a model whose step response equals the
// continuous impulse response C·e^{At}·B sampled exactly: (A, A·B, C, C·B)
// with the same delays. The Dirac part D·δ(t) is dropped, as in MATLAB. It
// requires inputs, outputs and D21 = 0.
func impulseAsStepModel(sys *System) (*System, error) {
	n, m, p := sys.Dims()
	if m == 0 || p == 0 {
		return nil, fmt.Errorf("model has %d inputs and %d outputs: %w", m, p, ErrDimensionMismatch)
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
// exact internal-delay chain. ok false with a nil error means the model is
// outside the chain's class or budget; a non-nil error is a failure.
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
// ErrDescriptorInitialState. A continuous descriptor without internal
// delays whose full-rank E is too ill-conditioned for E⁻¹A (as in Step)
// follows the pencil (A, E): the impulse response of the input E·x0,
// delayed by the output delays.
// Continuous models without internal delays, or whose internal delays share
// one length (within the chain budget, see Step), are sampled exactly; other
// internal-delay models use the approximate ZOH discretization of the delay
// channels, as MATLAB does. A stiff-E descriptor with internal delays takes
// that discretization of its parts as in Step, from x0 mapped to the
// parts' coordinates.
//
// tFinal is handled as in Step. A model without outputs returns
// ErrDimensionMismatch; one without inputs is valid.
func Initial(sys *System, x0 *mat.VecDense, tFinal float64) (*TimeResponse, error) {
	if err := requireSystem("Initial", sys); err != nil {
		return nil, err
	}
	if err := validateTimeHorizon(tFinal); err != nil {
		return nil, fmt.Errorf("Initial: %w", err)
	}
	if _, _, p := sys.Dims(); p == 0 {
		return nil, fmt.Errorf("Initial: model has no outputs: %w", ErrDimensionMismatch)
	}
	if x0 == nil {
		return nil, fmt.Errorf("Initial: x0 must not be nil: %w", ErrInvalidArgument)
	}
	if n, _, _ := sys.Dims(); x0.Len() == n {
		if split, err := sys.splitIODelayForInitialState(x0); err == nil {
			if parts, ok := stiffFreeParts(split, x0); ok {
				resp, err := stiffDescriptorInitial(split, parts, x0, tFinal)
				if err != nil {
					return nil, fmt.Errorf("Initial: %w", err)
				}
				return resp, nil
			}
		}
	}
	stiffLFT := false
	if n, _, _ := sys.Dims(); x0.Len() == n {
		if ex, xt, ok := stiffLFTExplicitState(sys, x0); ok {
			sys, x0, stiffLFT = ex, xt, true
		}
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
	if _, _, ok := delayChainModel(free); ok && !stiffLFT {
		t, dt, err := newTimeResponsePlanner(free).grid(tFinal, 0)
		if err != nil {
			return nil, fmt.Errorf("Initial: %w", err)
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
		return nil, fmt.Errorf("Initial: %w", err)
	}

	ds := plan.sim
	if _, m, _ := ds.Dims(); m == 0 {
		ds = zeroInputModel(ds)
	}
	_, m, _ := ds.Dims()
	resp, err := ds.Simulate(mat.NewDense(m, plan.steps, nil), x0, nil)
	if err != nil {
		return nil, fmt.Errorf("Initial: %w", err)
	}

	return plan.response(resp.Y), nil
}

// stiffFreeParts returns the parts of foldInaccurateParts of
// (A, E, [E·x0, A·x0], C, 0) for a continuous descriptor. Column 0 of a
// part's B is its share s of x0 and column 1 is A_part·s. Sampled from the
// shares, the free response is an impulse response with no constant C·x0
// to cancel in a decaying tail, unlike the step response of the input A·x0.
// Input and output delays are left out of the parts: the free response
// ignores input delays and sampleFreeParts shifts it by the output delays.
// A model with a nonzero Delay matrix (split it first with
// splitIODelayForInitialState) or internal delays does not split.
func stiffFreeParts(sys *System, x0 *mat.VecDense) ([]*System, bool) {
	if !sys.IsContinuous() || !sys.IsDescriptor() || delayMatrixHasNonzero(sys.Delay) {
		return nil, false
	}
	n, _, p := sys.Dims()
	fm := ioDelayFreeCore(sys).Copy()
	fm.B, fm.D = newDense(n, 2), newDense(p, 2)
	fm.InputDelay, fm.Delay, fm.InputName = nil, nil, nil
	var ex, ax mat.VecDense
	ex.MulVec(sys.E, x0)
	ax.MulVec(sys.A, x0)
	fm.B.SetCol(0, ex.RawVector().Data)
	fm.B.SetCol(1, ax.RawVector().Data)
	return foldInaccurateParts(fm)
}

// freePartShare returns the share of x0 of part k of stiffFreeParts. A fast
// part (k ≥ 1 of a split; stiffExplicitParts puts the slow part first)
// takes A_part⁻¹·(A·x0 column): E·x0 projects x0 onto the nearly null
// directions of E, so its column carries cond(E)·ε relative error there,
// while A·x0 has no such cancellation and A_part, all of whose poles are
// large, is well conditioned to solve. The slow part, or a fast part whose
// A_part is singular, keeps the E·x0 column.
func freePartShare(parts []*System, k int) []float64 {
	part := parts[k]
	pn, _, _ := part.Dims()
	share := make([]float64, pn)
	if k > 0 {
		var lu mat.LU
		lu.Factorize(part.A)
		if lu.Cond() < 0x1p52 {
			var s mat.VecDense
			if err := lu.SolveVecTo(&s, false, part.B.ColView(1)); err == nil {
				copy(share, s.RawVector().Data)
				return share
			}
		}
	}
	mat.Col(share, 0, part.B)
	return share
}

// sampleFreeParts sums y(k·dt) of the parts of stiffFreeParts of sys, each
// from its share of x0 (freePartShare), sampled exactly as in
// continuousFreeSamples: output i is delayed by its OutputDelay τ_i, zero
// before τ_i, and y_i(τ_i) is C_i·x0 itself.
func sampleFreeParts(sys *System, parts []*System, x0 *mat.VecDense, steps int, dt float64) (*mat.Dense, error) {
	_, _, p := sys.Dims()
	if p == 0 || steps == 0 {
		return nil, fmt.Errorf("free response needs outputs and samples, got p=%d steps=%d: %w", p, steps, ErrDimensionMismatch)
	}
	Y := mat.NewDense(p, steps, nil)
	rows, taus := make([]int, p), make([]float64, p)
	for i := range p {
		rows[i] = i
		if sys.OutputDelay != nil {
			taus[i] = sys.OutputDelay[i]
		}
	}
	var ad mat.Dense
	for pk, part := range parts {
		if pn, _, _ := part.Dims(); pn == 0 {
			continue
		}
		partExp(&ad, part.A, dt)
		addShiftedPartSamples(Y, rows, taus, part, &ad, freePartShare(parts, pk), dt, partExp)
	}
	var y0 mat.VecDense
	y0.MulVec(sys.C, x0)
	for i, tau := range taus {
		k0 := max(int(math.Ceil(tau/dt-gridTol)), 0)
		if k0 < steps && float64(k0)*dt-tau <= gridTol*dt {
			Y.Set(i, k0, y0.AtVec(i))
		}
	}
	return Y, nil
}

// stiffDescriptorInitial is Initial from the parts of stiffFreeParts on the
// automatic grid of sys without its input delays, which do not act on the
// free response.
func stiffDescriptorInitial(sys *System, parts []*System, x0 *mat.VecDense, tFinal float64) (*TimeResponse, error) {
	free := *sys
	free.InputDelay, free.Delay = nil, nil
	t, dt, err := newTimeResponsePlanner(&free).grid(tFinal, 0)
	if err != nil {
		return nil, err
	}
	Y, err := sampleFreeParts(sys, parts, x0, len(t), dt)
	if err != nil {
		return nil, err
	}
	return &TimeResponse{T: t, Y: Y, OutputName: copyStringSlice(sys.OutputName)}, nil
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
		return nil, fmt.Errorf("Initial: %w", err)
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
		return nil, fmt.Errorf("free response needs outputs and samples, got p=%d steps=%d: %w", p, steps, ErrDimensionMismatch)
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
		first := math.Ceil(tau/dt - gridTol)
		if first >= float64(steps) {
			continue
		}
		k0 := max(int(first), 0)
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
// returns ErrDescriptorInitialState, as in MATLAB, and a nonzero x0 of a
// model Step simulates by the pencil adds its free response as Initial
// does. x0 with Delay follows Simulate.
// Continuous models with internal delays of one common length are sampled
// exactly within the chain budget (see Step); other internal-delay models use
// the approximate ZOH discretization of the delay channels, as MATLAB does.
//
// t must be a finite, uniform, increasing grid of at least 2 samples (with
// spacing Dt for discrete models) and u is len(t)×m; a model without outputs
// returns ErrDimensionMismatch.
func Lsim(sys *System, u *mat.Dense, t []float64, x0 *mat.VecDense) (*TimeResponse, error) {
	if err := requireSystem("Lsim", sys); err != nil {
		return nil, err
	}
	if _, _, p := sys.Dims(); p == 0 {
		return nil, fmt.Errorf("Lsim: model has no outputs: %w", ErrDimensionMismatch)
	}
	var err error
	var freeX0 *mat.VecDense
	stiff := foldInaccurateContinuous(sys)
	if stiff {
		if n, _, _ := sys.Dims(); x0 != nil && x0.Len() != n {
			return nil, fmt.Errorf("Lsim: x0 length %d != state dimension %d: %w", x0.Len(), n, ErrDimensionMismatch)
		}
		if x0 != nil && !allZeroVec(x0) {
			freeX0 = x0
			sys, err = sys.splitIODelayForInitialState(x0)
		}
		x0 = nil
	} else {
		sys, x0, _, err = sys.timeResponseForm(x0)
	}
	if err != nil {
		return nil, fmt.Errorf("Lsim: %w", err)
	}
	if _, m, _ := sys.Dims(); m == 0 && (u == nil || u.IsEmpty()) {
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
			if err := addStiffFreeSamples(sys, freeX0, plan, Y); err != nil {
				return nil, fmt.Errorf("Lsim: %w", err)
			}
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
	if err := addStiffFreeSamples(sys, freeX0, plan, resp.Y); err != nil {
		return nil, fmt.Errorf("Lsim: %w", err)
	}

	return plan.response(resp.Y), nil
}

// addStiffFreeSamples adds to Y the free response from x0 of a continuous
// descriptor with foldInaccurateParts on the grid of plan; a nil x0 adds
// nothing. A pencil that does not split (not expected: the split depends on
// (A, E) only) takes the fold E⁻¹A.
func addStiffFreeSamples(sys *System, x0 *mat.VecDense, plan timeResponsePlan, Y *mat.Dense) (err error) {
	if x0 == nil {
		return nil
	}
	var free *mat.Dense
	if parts, ok := stiffFreeParts(sys, x0); ok {
		free, err = sampleFreeParts(sys, parts, x0, plan.steps, plan.dt)
	} else if fold, ferr := sys.ToExplicit(); ferr != nil {
		err = ferr
	} else {
		free, err = fold.continuousFreeSamples(x0, plan.steps, plan.dt)
	}
	if err != nil {
		return err
	}
	Y.Add(Y, free)
	return nil
}
