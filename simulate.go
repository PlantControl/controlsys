package controlsys

import (
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

// Response is the result of (*System).Simulate.
type Response struct {
	// Y is the p×steps output, one column per sample. It is nil only for a
	// model without outputs, which Simulate accepts only with
	// SimulateOpts.FinalState.
	Y *mat.Dense
	// XFinal is the state after the last sample, set only when
	// SimulateOpts.FinalState requests it.
	XFinal *mat.VecDense
}

// SimulateOpts holds optional Simulate settings; nil selects the defaults.
type SimulateOpts struct {
	// Workspace is an optional length-n scratch vector reused across calls to
	// avoid an allocation. It is overwritten and never returned as XFinal.
	Workspace *mat.VecDense
	// FinalState requests Response.XFinal, as MATLAB lsim's optional x output
	// requests the state. Simulate returns an error when the final state is
	// not representable rather than leaving XFinal nil.
	FinalState bool
	// Steps is the number of samples, MATLAB's length(t). It is how a model
	// without inputs (whose n×0 input matrix gonum cannot hold) sets the
	// horizon; with u given it must be 0 or equal u's column count.
	Steps int

	yBuf  *mat.Dense
	duBuf *mat.Dense
}

// Simulate propagates a discrete model from x0 under u (one column per
// sample). A nil u means zero samples unless opts.Steps is set; a model
// without inputs takes nil u and opts.Steps, as MATLAB's lsim takes a
// length(t)×0 u. With a nonzero x0, Delay is split into input and output
// delays as PullDelaysToLFT does, so its output share delays the free
// response like OutputDelay; a Delay with no such split returns
// ErrDelayUnsupported. Descriptor models with invertible E are simulated in
// their own state coordinates. Singular E is reduced to its slow subsystem plus
// feedthrough, as MATLAB does: non-causal (improper) models return
// ErrImproperModel, and a nonzero x0 or a FinalState request returns
// ErrDescriptorInitialState because the algebraic states depend on future
// inputs.
//
// XFinal is computed only when opts.FinalState is set. With delays it is the
// state of the delay-free model at the final time, as MATLAB lsim's x: inputs
// reach it delayed by InputDelay plus the input share of Delay (the split
// PullDelaysToLFT uses); output delays do not affect it. Delay-line contents
// are not part of it, so continuing past the horizon needs the input history.
// When Delay has no input+output split no single state sees the delayed
// inputs, and a FinalState request returns ErrDelayUnsupported. A model
// without states has no final state to return, and a FinalState request
// returns ErrDimensionMismatch.
//
// As MATLAB lsim requires a time vector, zero samples return
// ErrInsufficientData. A model without outputs has no representable Y (gonum
// cannot hold a 0×steps matrix), so it is accepted only with FinalState, and
// Y is then nil; otherwise it returns ErrDimensionMismatch.
func (sys *System) Simulate(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) (*Response, error) {
	if err := requireSystem("Simulate", sys); err != nil {
		return nil, err
	}
	if sys.IsContinuous() {
		return nil, fmt.Errorf("Simulate: model must be discrete: %w", ErrWrongDomain)
	}
	if err := sys.validateSimulateInputs(u, x0, opts); err != nil {
		return nil, err
	}
	steps := 0
	if u != nil {
		_, steps = u.Dims()
	}
	if opts != nil {
		steps = max(steps, opts.Steps)
	}
	if steps == 0 {
		return nil, fmt.Errorf("Simulate: no samples: %w", ErrInsufficientData)
	}
	if _, _, p := sys.Dims(); p == 0 && (opts == nil || !opts.FinalState) {
		return nil, fmt.Errorf("Simulate: model has no outputs and FinalState is not requested: %w", ErrDimensionMismatch)
	}
	sim, simX0, reduced, err := sys.timeResponseForm(x0)
	if err != nil {
		return nil, fmt.Errorf("Simulate: %w", err)
	}
	finalState := opts != nil && opts.FinalState
	if finalState && reduced {
		return nil, fmt.Errorf("Simulate: final state: %w", ErrDescriptorInitialState)
	}
	n, m, p := sim.Dims()
	if finalState && n == 0 {
		return nil, fmt.Errorf("Simulate: final state of a model without states: %w", ErrDimensionMismatch)
	}
	if m == 0 && opts != nil && opts.Steps > 0 {
		sim, u = zeroInputModel(sim), mat.NewDense(1, opts.Steps, nil)
	}
	if !reduced {
		if p > 0 || n == 0 || u == nil {
			return simulationDispatcher{sys: sim, finalState: finalState}.run(u, simX0, opts)
		}
		if !finalState {
			return &Response{}, nil
		}
		opts = &SimulateOpts{Workspace: opts.Workspace}
		resp, err := simulationDispatcher{sys: zeroOutputModel(sim), finalState: true}.run(u, simX0, opts)
		if err != nil {
			return nil, err
		}
		resp.Y = nil
		return resp, nil
	}
	if opts != nil {
		opts = &SimulateOpts{yBuf: opts.yBuf, duBuf: opts.duBuf}
	}
	return simulationDispatcher{sys: sim}.run(u, nil, opts)
}

type simulationDispatcher struct {
	sys        *System
	finalState bool
}

func (d simulationDispatcher) run(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) (*Response, error) {
	resp, err := d.simulate(u, x0, opts)
	if err != nil {
		return nil, err
	}
	if !d.finalState {
		resp.XFinal = nil
	}
	return resp, nil
}

func (d simulationDispatcher) simulate(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) (*Response, error) {
	if d.sys.HasInternalDelay() {
		hasIODelay := d.sys.Delay != nil || d.sys.InputDelay != nil || d.sys.OutputDelay != nil
		if hasIODelay {
			merged, err := d.sys.PullDelaysToLFT()
			if err != nil {
				return nil, err
			}
			n, _, _ := d.sys.Dims()
			if mn, _, _ := merged.Dims(); d.finalState && mn != n {
				return nil, fmt.Errorf("Simulate: final state with Delay lacking an input+output split: %w", ErrDelayUnsupported)
			}
			return merged.simulateWithInternalDelay(u, x0, opts)
		}
		return d.sys.simulateWithInternalDelay(u, x0, opts)
	}

	if d.sys.HasDelay() {
		return d.sys.simulateWithDelay(u, x0, opts, d.finalState)
	}

	return d.sys.simulateNoDelay(u, x0, opts)
}

func (sys *System) validateSimulateInputs(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) error {
	n, m, p := sys.Dims()
	steps := 0
	if u != nil {
		ur, uc := u.Dims()
		if ur != m {
			return fmt.Errorf("Simulate: input rows %d != system inputs %d: %w", ur, m, ErrDimensionMismatch)
		}
		steps = uc
	}
	if opts != nil {
		if err := validateSimulateSteps(opts.Steps, steps, m, u != nil); err != nil {
			return err
		}
		steps = max(steps, opts.Steps)
	}
	if x0 != nil && x0.Len() != n {
		return fmt.Errorf("Simulate: x0 length %d != state dimension %d: %w", x0.Len(), n, ErrDimensionMismatch)
	}
	if opts == nil {
		return nil
	}
	if opts.Workspace != nil && opts.Workspace.Len() != n {
		return fmt.Errorf("Simulate: workspace length %d != state dimension %d: %w", opts.Workspace.Len(), n, ErrDimensionMismatch)
	}
	if opts.yBuf != nil {
		yr, yc := opts.yBuf.Dims()
		if yr != p || yc != steps {
			return fmt.Errorf("Simulate: yBuf %dx%d != %dx%d: %w", yr, yc, p, steps, ErrDimensionMismatch)
		}
	}
	if opts.duBuf != nil {
		dr, dc := opts.duBuf.Dims()
		if dr != p || dc != steps {
			return fmt.Errorf("Simulate: duBuf %dx%d != %dx%d: %w", dr, dc, p, steps, ErrDimensionMismatch)
		}
	}
	return nil
}

func validateSimulateSteps(want, cols, m int, hasU bool) error {
	switch {
	case want < 0:
		return fmt.Errorf("Simulate: Steps %d < 0: %w", want, ErrInvalidArgument)
	case want == 0:
		return nil
	case hasU && want != cols:
		return fmt.Errorf("Simulate: Steps %d != input columns %d: %w", want, cols, ErrDimensionMismatch)
	case !hasU && m > 0:
		return fmt.Errorf("Simulate: Steps without u needs a model with no inputs, have %d: %w", m, ErrDimensionMismatch)
	}
	return nil
}

type simulationProblem struct {
	sys   *System
	u     *mat.Dense
	x0    *mat.VecDense
	opts  *SimulateOpts
	n     int
	m     int
	p     int
	steps int
}

func newSimulationProblem(sys *System, u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) simulationProblem {
	n, m, p := sys.Dims()
	problem := simulationProblem{sys: sys, u: u, x0: x0, opts: opts, n: n, m: m, p: p}
	if u != nil {
		_, problem.steps = u.Dims()
	}
	return problem
}

func (p simulationProblem) newXFinal() *mat.VecDense {
	if p.n == 0 {
		return nil
	}
	x := mat.NewVecDense(p.n, nil)
	if p.x0 != nil {
		x.CopyVec(p.x0)
	}
	return x
}

func (p simulationProblem) newY() *mat.Dense {
	if p.p == 0 || p.steps == 0 {
		return nil
	}
	if p.opts != nil && p.opts.yBuf != nil {
		p.opts.yBuf.Zero()
		return p.opts.yBuf
	}
	return mat.NewDense(p.p, p.steps, nil)
}

func (sys *System) simulateNoDelay(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) (*Response, error) {
	problem := newSimulationProblem(sys, u, x0, opts)
	n, m, p, steps := problem.n, problem.m, problem.p, problem.steps

	if p == 0 || steps == 0 {
		return &Response{Y: nil, XFinal: problem.newXFinal()}, nil
	}

	Y := problem.newY()
	x := problem.newXFinal()

	if n > 0 {
		var tmp *mat.VecDense
		if opts != nil && opts.Workspace != nil {
			tmp = opts.Workspace
		} else {
			tmp = mat.NewVecDense(n, nil)
		}

		yRaw := Y.RawMatrix()
		aGen := sys.A.RawMatrix()
		cGen := sys.C.RawMatrix()
		xVec := x.RawVector()
		tmpVec := tmp.RawVector()
		ykData := make([]float64, p)
		ykVec := blas64.Vector{N: p, Inc: 1, Data: ykData}

		if m > 0 {
			uColData := make([]float64, m)
			uRaw := u.RawMatrix()
			bGen := sys.B.RawMatrix()
			uColVec := blas64.Vector{N: m, Inc: 1, Data: uColData}

			for k := range steps {
				blas64.Gemv(blas.NoTrans, 1, cGen, xVec, 0, ykVec)
				for i := range p {
					yRaw.Data[i*yRaw.Stride+k] = ykData[i]
				}

				for j := range m {
					uColData[j] = uRaw.Data[j*uRaw.Stride+k]
				}
				blas64.Gemv(blas.NoTrans, 1, aGen, xVec, 0, tmpVec)
				blas64.Gemv(blas.NoTrans, 1, bGen, uColVec, 1, tmpVec)
				x, tmp = tmp, x
				xVec, tmpVec = tmpVec, xVec
			}
		} else {
			for k := range steps {
				blas64.Gemv(blas.NoTrans, 1, cGen, xVec, 0, ykVec)
				for i := range p {
					yRaw.Data[i*yRaw.Stride+k] = ykData[i]
				}

				blas64.Gemv(blas.NoTrans, 1, aGen, xVec, 0, tmpVec)
				x, tmp = tmp, x
				xVec, tmpVec = tmpVec, xVec
			}
		}
	}

	if m > 0 {
		var du *mat.Dense
		if opts != nil && opts.duBuf != nil {
			du = opts.duBuf
		} else {
			du = mat.NewDense(p, steps, nil)
		}
		du.Mul(sys.D, u)
		Y.Add(Y, du)
	}

	if opts != nil && x == opts.Workspace {
		x = mat.VecDenseCopyOf(x)
	}
	return &Response{
		Y:      Y,
		XFinal: x,
	}, nil
}

func (sys *System) simulateWithDelay(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts, finalState bool) (*Response, error) {
	problem := newSimulationProblem(sys, u, x0, opts)
	n, m, p, steps := problem.n, problem.m, problem.p, problem.steps

	totalDelay := sys.TotalDelay()
	if totalDelay == nil {
		totalDelay = mat.NewDense(p, m, nil)
	}

	var xFinal *mat.VecDense
	var stateLag []int
	if finalState {
		var ok bool
		if stateLag, ok = sys.stateInputLags(); !ok {
			return nil, fmt.Errorf("Simulate: final state with Delay lacking an input+output split: %w", ErrDelayUnsupported)
		}
		xFinal = problem.newXFinal()
	}

	if p == 0 || steps == 0 {
		return &Response{Y: nil, XFinal: xFinal}, nil
	}

	Y := problem.newY()

	if n > 0 && x0 != nil {
		x := mat.NewVecDense(n, nil)
		x.CopyVec(x0)
		tmp := mat.NewVecDense(n, nil)
		yk := mat.NewVecDense(p, nil)
		yAutoRaw := Y.RawMatrix()
		ykRaw := yk.RawVector()
		outputLag := make([]int, p)
		if sys.OutputDelay != nil {
			for i := range p {
				outputLag[i] = int(math.Round(sys.OutputDelay[i]))
			}
		}
		for k := range steps {
			yk.MulVec(sys.C, x)
			for i := range p {
				if k+outputLag[i] < steps {
					yAutoRaw.Data[i*yAutoRaw.Stride+k+outputLag[i]] = ykRaw.Data[i*ykRaw.Inc]
				}
			}
			tmp.MulVec(sys.A, x)
			x, tmp = tmp, x
		}
		if xFinal != nil {
			xFinal.CopyVec(x)
		}
	}

	if m == 0 {
		return &Response{Y: Y, XFinal: xFinal}, nil
	}

	uRaw := u.RawMatrix()
	dRaw := sys.D.RawMatrix()
	delayRaw := totalDelay.RawMatrix()
	yRaw := Y.RawMatrix()

	delayIdx := make([]int, p*m)
	for i := range p {
		for j := range m {
			d := int(math.Round(delayRaw.Data[i*delayRaw.Stride+j]))
			if d < 0 {
				return nil, fmt.Errorf("%w: delay(%d,%d) = %d", ErrNegativeDelay, i, j, d)
			}
			delayIdx[i*m+j] = d
		}
	}

	yForced := mat.NewDense(p, m, nil)
	yForcedRaw := yForced.RawMatrix()

	var xCols, nextX *mat.Dense
	var bRaw blas64.General
	if n > 0 {
		xCols = mat.NewDense(n, m, nil)
		nextX = mat.NewDense(n, m, nil)
		bRaw = sys.B.RawMatrix()
	}

	for k := range steps {
		if xFinal != nil {
			addLaggedStates(xFinal, xCols, stateLag, steps-k)
		}
		var nextRaw blas64.General
		if n > 0 {
			yForced.Mul(sys.C, xCols)
			nextX.Mul(sys.A, xCols)
			nextRaw = nextX.RawMatrix()
		} else {
			yForced.Zero()
		}

		for j := range m {
			uk := uRaw.Data[j*uRaw.Stride+k]
			if n > 0 && uk != 0 {
				blas64.Axpy(uk,
					blas64.Vector{N: n, Inc: bRaw.Stride, Data: bRaw.Data[j:]},
					blas64.Vector{N: n, Inc: nextRaw.Stride, Data: nextRaw.Data[j:]},
				)
			}
			for i := range p {
				outIdx := k + delayIdx[i*m+j]
				if outIdx >= steps {
					continue
				}
				val := yForcedRaw.Data[i*yForcedRaw.Stride+j] + dRaw.Data[i*dRaw.Stride+j]*uk
				yRaw.Data[i*yRaw.Stride+outIdx] += val
			}
		}

		if n > 0 {
			xCols, nextX = nextX, xCols
		}
	}

	if xFinal != nil {
		addLaggedStates(xFinal, xCols, stateLag, 0)
	}

	return &Response{Y: Y, XFinal: xFinal}, nil
}

// stateInputLags returns the delay each input reaches the states with:
// InputDelay plus the input share of Delay, split as PullDelaysToLFT does.
// ok is false when Delay has no input+output split, so no single state of
// (A, B) sees the delayed inputs.
func (sys *System) stateInputLags() (lags []int, ok bool) {
	_, m, _ := sys.Dims()
	lags = make([]int, m)
	if sys.Delay != nil {
		in, _, residual := DecomposeIODelay(sys.Delay)
		if delayMatrixHasNonzero(residual) {
			return nil, false
		}
		for j, d := range in {
			lags[j] = int(math.Round(d))
		}
	}
	for j, d := range sys.InputDelay {
		lags[j] += int(math.Round(d))
	}
	return lags, true
}

// addLaggedStates adds column j of xCols, the state driven by undelayed u_j,
// to x for each input whose lag equals remaining: x_N driven by u_j delayed
// by lag is the undelayed state at N-lag.
func addLaggedStates(x *mat.VecDense, xCols *mat.Dense, lags []int, remaining int) {
	xRaw := xCols.RawMatrix()
	n := x.Len()
	for j, lag := range lags {
		if lag != remaining {
			continue
		}
		for i := range n {
			x.SetVec(i, x.AtVec(i)+xRaw.Data[i*xRaw.Stride+j])
		}
	}
}

func (sys *System) simulateWithInternalDelay(u *mat.Dense, x0 *mat.VecDense, opts *SimulateOpts) (*Response, error) {
	problem := newSimulationProblem(sys, u, x0, opts)
	n, m, p, steps := problem.n, problem.m, problem.p, problem.steps
	N := sys.internalDelayCount()

	if p == 0 || steps == 0 {
		return &Response{Y: nil, XFinal: problem.newXFinal()}, nil
	}

	delays := make([]int, N)
	maxDelay := 0
	for j := range N {
		delays[j] = int(math.Round(sys.LFT.Tau[j]))
		if delays[j] > maxDelay {
			maxDelay = delays[j]
		}
	}

	bufSize := maxDelay
	if N > 0 && bufSize == 0 {
		bufSize = 1
	}

	Y := problem.newY()
	x := problem.newXFinal()

	var zBuf []float64
	if N > 0 {
		zBuf = make([]float64, N*(bufSize+1))
	}

	wData := make([]float64, N)
	zData := make([]float64, N)

	var uColData []float64
	var uData []float64
	var uStride int
	if m > 0 {
		uColData = make([]float64, m)
		if u != nil {
			uRaw := u.RawMatrix()
			uData = uRaw.Data
			uStride = uRaw.Stride
		}
	}

	var tmp *mat.VecDense
	if n > 0 {
		tmp = mat.NewVecDense(n, nil)
	}

	yRaw := Y.RawMatrix()
	ykData := make([]float64, p)

	var aGen, bGen, b2Gen, cGen, dGen, d12Gen, c2Gen, d21Gen, d22Gen blas64.General
	var xVec, tmpVec blas64.Vector
	if n > 0 {
		aGen = sys.A.RawMatrix()
		cGen = sys.C.RawMatrix()
		xVec = x.RawVector()
		tmpVec = tmp.RawVector()
	}
	if m > 0 {
		if n > 0 {
			bGen = sys.B.RawMatrix()
		}
		dGen = sys.D.RawMatrix()
	}
	if N > 0 {
		if n > 0 {
			b2Gen = sys.LFT.B2.RawMatrix()
			c2Gen = sys.LFT.C2.RawMatrix()
		}
		d12Gen = sys.LFT.D12.RawMatrix()
		if m > 0 {
			d21Gen = sys.LFT.D21.RawMatrix()
		}
		d22Gen = sys.LFT.D22.RawMatrix()
	}
	ykVec := blas64.Vector{N: p, Inc: 1, Data: ykData}
	wVec := blas64.Vector{N: N, Inc: 1, Data: wData}
	zVec := blas64.Vector{N: N, Inc: 1, Data: zData}
	uColVec := blas64.Vector{N: m, Inc: 1, Data: uColData}

	for k := range steps {
		for j := range N {
			dj := delays[j]
			if dj == 0 {
				if k == 0 {
					wData[j] = 0
				} else {
					wData[j] = zBuf[j*(bufSize+1)+(k-1)%(bufSize+1)]
				}
			} else if k < dj {
				wData[j] = 0
			} else {
				wData[j] = zBuf[j*(bufSize+1)+(k-dj)%(bufSize+1)]
			}
		}

		if m > 0 {
			for i := range m {
				uColData[i] = uData[i*uStride+k]
			}
		}

		if n > 0 {
			blas64.Gemv(blas.NoTrans, 1, cGen, xVec, 0, ykVec)
		} else {
			for i := range ykData {
				ykData[i] = 0
			}
		}
		if m > 0 {
			blas64.Gemv(blas.NoTrans, 1, dGen, uColVec, 1, ykVec)
		}
		if N > 0 {
			blas64.Gemv(blas.NoTrans, 1, d12Gen, wVec, 1, ykVec)
		}
		for i := range p {
			yRaw.Data[i*yRaw.Stride+k] = ykData[i]
		}

		if N > 0 {
			if n > 0 {
				blas64.Gemv(blas.NoTrans, 1, c2Gen, xVec, 0, zVec)
			} else {
				for i := range zData {
					zData[i] = 0
				}
			}
			if m > 0 {
				blas64.Gemv(blas.NoTrans, 1, d21Gen, uColVec, 1, zVec)
			}
			blas64.Gemv(blas.NoTrans, 1, d22Gen, wVec, 1, zVec)

			idx := k % (bufSize + 1)
			for j := range N {
				zBuf[j*(bufSize+1)+idx] = zData[j]
			}
		}

		if n > 0 {
			blas64.Gemv(blas.NoTrans, 1, aGen, xVec, 0, tmpVec)
			if m > 0 {
				blas64.Gemv(blas.NoTrans, 1, bGen, uColVec, 1, tmpVec)
			}
			if N > 0 {
				blas64.Gemv(blas.NoTrans, 1, b2Gen, wVec, 1, tmpVec)
			}
			x, tmp = tmp, x
			xVec, tmpVec = tmpVec, xVec
		}
	}

	return &Response{Y: Y, XFinal: x}, nil
}
