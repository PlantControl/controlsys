package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"

	"plantcontrol.org/v1/gonum/mat"
)

type FreqResponseMatrix struct {
	Data       []complex128
	Omega      []float64
	NFreq      int
	P, M       int
	InputName  []string
	OutputName []string
}

func (f *FreqResponseMatrix) At(freq, output, input int) complex128 {
	return newSampledComplexResponse(f.Data, f.Omega, f.P, f.M).at(freq, output, input)
}

func newFreqResponseMatrix(data []complex128, omega []float64, p, m int, inputName, outputName []string) *FreqResponseMatrix {
	return &FreqResponseMatrix{
		Data:       data,
		Omega:      omega,
		NFreq:      len(omega),
		P:          p,
		M:          m,
		InputName:  copyStringSlice(inputName),
		OutputName: copyStringSlice(outputName),
	}
}

func newFreqResponseMatrixOwned(data []complex128, omega []float64, p, m int, inputName, outputName []string) *FreqResponseMatrix {
	return newFreqResponseMatrix(data, copyFloatSlice(omega), p, m, inputName, outputName)
}

type BodeResult struct {
	Omega      []float64
	magDB      []float64
	phase      []float64
	p, m       int
	InputName  []string
	OutputName []string
}

func (b *BodeResult) MagDBAt(freq, output, input int) float64 {
	return newSampledScalarResponse(b.magDB, b.Omega, b.p, b.m).at(freq, output, input)
}

func (b *BodeResult) PhaseAt(freq, output, input int) float64 {
	return newSampledScalarResponse(b.phase, b.Omega, b.p, b.m).at(freq, output, input)
}

func bodeResultFromResponse(omega []float64, data []complex128, p, m int, inputName, outputName []string) *BodeResult {
	response := newSampledComplexResponse(data, omega, p, m)
	return bodeResultFromAccessor(omega, p, m, inputName, outputName, response.at)
}

func bodeResultFromAccessor(omega []float64, p, m int, inputName, outputName []string, at func(k, i, j int) complex128) *BodeResult {
	nw := len(omega)
	pm := p * m
	magDB := make([]float64, nw*pm)
	phase := make([]float64, nw*pm)
	mag := newSampledScalarResponse(magDB, omega, p, m)
	phaseResp := newSampledScalarResponse(phase, omega, p, m)

	for k := range omega {
		for i := range p {
			for j := range m {
				h := at(k, i, j)
				mag.set(k, i, j, 20*math.Log10(cmplx.Abs(h)))
				phaseResp.set(k, i, j, cmplx.Phase(h)*180/math.Pi)
			}
		}
	}

	unwrapBodePhase(phase, p, m, nw)

	return &BodeResult{
		Omega:      omega,
		magDB:      magDB,
		phase:      phase,
		p:          p,
		m:          m,
		InputName:  inputName,
		OutputName: outputName,
	}
}

func unwrapBodePhase(phase []float64, p, m, nw int) {
	response := newSampledScalarResponse(phase, nil, p, m)
	for i := range p {
		for j := range m {
			for k := 1; k < nw; k++ {
				cur := response.layout.offset(k, i, j)
				prev := response.layout.offset(k-1, i, j)
				diff := phase[cur] - phase[prev]
				if diff > 180 {
					phase[cur] -= 360
				}
				if diff < -180 {
					phase[cur] += 360
				}
			}
		}
	}
}

// FreqResponse evaluates the response at each omega (rad/s; z = e^{jωDt} for
// discrete models).
//
// As MATLAB freqresp and evalfr return Inf rather than failing at a pole, a
// frequency at a pole of the model is not an error, for explicit, descriptor
// and internal-delay realizations alike: entries the pole reaches are
// infinite (cmplx.IsInf reports true; their phase is meaningless), and the
// other entries keep their finite values, extrapolated from nearby points to
// about 1e-12 relative accuracy. A pole of the delay-free plant that an
// internal delay loop moves, such as an integrator inside the loop, is not a
// pole of the model and evaluates to its finite value. A pole is recognised
// when the pencil is singular to working precision; a frequency merely near a
// pole gives large finite values. ErrSingularTransform remains for models
// singular at every frequency near omega, such as det(sE-A) ≡ 0.
func (sys *System) FreqResponse(omega []float64) (*FreqResponseMatrix, error) {
	e, err := validFrequencyEvaluator(sys, "FreqResponse")
	if err != nil {
		return nil, err
	}
	return e.response(omega)
}

// FreqResponsePointwise evaluates the frequency response with guaranteed
// per-frequency single-point arithmetic: the value at each omega[k] is
// bit-identical to FreqResponse([]float64{omega[k]}), regardless of
// len(omega). FreqResponse may evaluate long sweeps of large delay-free
// state-space models (n > 8m+8, A not upper Hessenberg) through one
// Hessenberg reduction of A, whose values agree with the single-point path
// to componentwise rounding but are not bit-identical;
// FreqResponsePointwise never does, at the cost of one dense solve per
// frequency. Use it when downstream comparisons require sweep results to
// reproduce single-point evaluations exactly.
func (sys *System) FreqResponsePointwise(omega []float64) (*FreqResponseMatrix, error) {
	e, err := validFrequencyEvaluator(sys, "FreqResponsePointwise")
	if err != nil {
		return nil, err
	}
	return e.responsePointwise(omega)
}

func (sys *System) Bode(omega []float64, nPoints int) (*BodeResult, error) {
	if omega == nil {
		var err2 error
		omega, err2 = autoBodeFreqs(sys, nPoints)
		if err2 != nil {
			return nil, err2
		}
	}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return nil, err
	}

	return bodeResultFromResponse(omega, resp.Data, resp.P, resp.M,
		copyStringSlice(sys.InputName),
		copyStringSlice(sys.OutputName),
	), nil
}

// EvalFr evaluates the response at the complex point s (z for discrete
// models). At a pole it follows the FreqResponse contract.
func (sys *System) EvalFr(s complex128) ([][]complex128, error) {
	e, err := validFrequencyEvaluator(sys, "EvalFr")
	if err != nil {
		return nil, err
	}
	return e.eval(s)
}

type frequencyEvaluator struct {
	sys *System
	n   int
	m   int
	p   int
}

// validFrequencyEvaluator rejects hand-built systems whose exported fields
// disagree in shape; the kernels index them unchecked.
func validFrequencyEvaluator(sys *System, op string) (frequencyEvaluator, error) {
	if err := sys.Validate(); err != nil {
		return frequencyEvaluator{}, fmt.Errorf("%s: %w", op, err)
	}
	return newFrequencyEvaluator(sys), nil
}

func newFrequencyEvaluator(sys *System) frequencyEvaluator {
	n, m, p := sys.Dims()
	return frequencyEvaluator{sys: sys, n: n, m: m, p: p}
}

func (e frequencyEvaluator) response(omega []float64) (*FreqResponseMatrix, error) {
	if len(omega) == 0 {
		return nil, nil
	}
	if e.sys.HasInternalDelay() {
		resp, err := freqResponseLFT(e.sys, omega, e.p, e.m)
		if err != nil {
			return nil, err
		}
		applyIODelayPhase(e.sys, omega, resp.Data, e.p, e.m, true)
		resp.InputName = copyStringSlice(e.sys.InputName)
		resp.OutputName = copyStringSlice(e.sys.OutputName)
		return resp, nil
	}

	data := make([]complex128, len(omega)*e.p*e.m)
	if e.sys.IsDescriptor() {
		if err := e.descriptorSweepInto(omega, data); err != nil {
			return nil, err
		}
		applyIODelayPhase(e.sys, omega, data, e.p, e.m, true)
		return e.matrix(data, omega), nil
	}
	if err := e.sweepInto(omega, data, e.pointSolver(len(omega))); err != nil {
		return nil, err
	}
	return e.matrix(data, omega), nil
}

// responsePointwise evaluates each frequency exactly as response would for
// a one-element sweep: balanced dense solve first, per-point
// transfer-function fallback on solve failure, with the delay phase applied
// per point using the flag of whichever path produced the value.
func (e frequencyEvaluator) responsePointwise(omega []float64) (*FreqResponseMatrix, error) {
	if len(omega) == 0 {
		return nil, nil
	}
	if e.sys.HasInternalDelay() || e.sys.IsDescriptor() {
		// These paths already evaluate one frequency at a time with
		// batch-size-independent arithmetic.
		return e.response(omega)
	}

	data := make([]complex128, len(omega)*e.p*e.m)
	if err := e.sweepInto(omega, data, e.pointSolver(1)); err != nil {
		return nil, err
	}
	return e.matrix(data, omega), nil
}

// frequencyPointSolver evaluates the delay-free explicit response at one
// complex frequency, failing when the frequency is numerically a pole.
type frequencyPointSolver interface {
	evalInto(s complex128, dst []complex128) error
}

// sweepInto evaluates each frequency with solver, taking the pole limit
// where the pencil is singular, and applies the I/O delay phase.
func (e frequencyEvaluator) sweepInto(omega []float64, data []complex128, solver frequencyPointSolver) error {
	pm := e.p * e.m
	delaySS := effectiveIODelayMatrix(e.sys, e.p, e.m, true)
	eval := solver.evalInto
	for k, w := range omega {
		s := e.sAt(w)
		dst := data[k*pm : (k+1)*pm]
		if err := evalWithPoleLimit(eval, e.sys, s, dst); err != nil {
			return err
		}
		if delaySS != nil {
			applyIODelayMatrixAtS(e.sys, s, dst, e.p, e.m, delaySS)
		}
	}
	return nil
}

func (e frequencyEvaluator) eval(s complex128) ([][]complex128, error) {
	return e.evalPoint(s, true)
}

// evalPoint evaluates one complex frequency. At a pole of the model it
// returns the pole limit (see FreqResponse), or with poleLimit false the
// solver's ErrSingularTransform.
func (e frequencyEvaluator) evalPoint(s complex128, poleLimit bool) ([][]complex128, error) {
	data := make([]complex128, e.p*e.m)
	eval := e.pointEval()
	if !poleLimit {
		if err := eval(s, data); err != nil {
			return nil, err
		}
	} else if err := evalWithPoleLimit(eval, e.sys, s, data); err != nil {
		return nil, err
	}
	applyIODelayAtS(e.sys, s, data, e.p, e.m, true)
	return complexFlatToGrid(data, e.p, e.m), nil
}

// pointEval returns a single-point evaluator of the response without I/O
// delays; it fails at a pole.
func (e frequencyEvaluator) pointEval() func(complex128, []complex128) error {
	if !e.sys.HasInternalDelay() {
		return e.pointSolver(1).evalInto
	}
	N := e.sys.internalDelayCount()
	ws := newLFTWorkspace(e.sys, e.n, N, e.p, e.m)
	return func(s complex128, dst []complex128) error {
		if err := evalFrLFTInto(ws, e.sys, s, N, e.p, e.m); err != nil {
			return err
		}
		copy(dst, ws.g[:e.p*e.m])
		return nil
	}
}

func (e frequencyEvaluator) sAt(w float64) complex128 {
	return newTimeDomain(e.sys.Dt).frequencyVariable(w)
}

// pointSolver picks the delay-free solver for an nw-point sweep. Both solve
// the balanced realization with componentwise-stable arithmetic; the
// Hessenberg sweep pays an O(n³) reduction once to make each point O(n²m)
// and handles only explicit models.
// FreqResponsePointwise and EvalFr use the nw=1 choice so single points stay
// bit-identical across entry points.
func (e frequencyEvaluator) pointSolver(nw int) frequencyPointSolver {
	if e.sys.IsDescriptor() || e.useDenseSweep(nw) {
		return newBalancedDense(e.sys, e.n, e.m, e.p)
	}
	return newHessenbergSweep(e.sys, e.n, e.m, e.p)
}

// useDenseSweep reports whether per-point GEPP (n³/3 per point) beats the
// refined Hessenberg sweep (O(n²m) solves, products with Q and one
// refinement step per point). On fully coupled models
// (BenchmarkFrequencySweepKernels, M1 Pro) the crossover is near
// n = 8m+8: 16 states for SISO, 24 for m=2, 40 for m=4. When A is already
// upper Hessenberg, GEPP skips the zero multipliers, costs O(n²) per point
// like the sweep, and needs no orthogonal reduction or refinement.
func (e frequencyEvaluator) useDenseSweep(nw int) bool {
	return nw <= 2 || e.n <= 8*max(e.m, 1)+8 || isUpperHessenberg(e.sys.A)
}

func isUpperHessenberg(a *mat.Dense) bool {
	raw := a.RawMatrix()
	for i := 2; i < raw.Rows; i++ {
		for _, v := range raw.Data[i*raw.Stride : i*raw.Stride+i-1] {
			if v != 0 {
				return false
			}
		}
	}
	return true
}

func (e frequencyEvaluator) descriptorSweepInto(omega []float64, dst []complex128) error {
	pm := e.p * e.m
	eval := e.pointSolver(len(omega)).evalInto
	for k, w := range omega {
		if err := evalWithPoleLimit(eval, e.sys, e.sAt(w), dst[k*pm:(k+1)*pm]); err != nil {
			return err
		}
	}
	return nil
}

func (e frequencyEvaluator) matrix(data []complex128, omega []float64) *FreqResponseMatrix {
	return newFreqResponseMatrix(data, omega, e.p, e.m, e.sys.InputName, e.sys.OutputName)
}

func autoBodeFreqs(sys *System, nPoints int) ([]float64, error) {
	if nPoints <= 0 {
		nPoints = 200
	}

	poles, err := sys.Poles()
	if err != nil {
		return nil, err
	}
	var natFreqs []float64
	for _, p := range poles {
		var wn float64
		wn = newTimeDomain(sys.Dt).naturalFrequency(p)
		if wn > 0 {
			natFreqs = append(natFreqs, wn)
		}
	}

	wMin, wMax := 0.01, 100.0
	if len(natFreqs) > 0 {
		lo, hi := natFreqs[0], natFreqs[0]
		for _, w := range natFreqs[1:] {
			if w < lo {
				lo = w
			}
			if w > hi {
				hi = w
			}
		}
		wMin = lo / 10
		wMax = hi * 10
		if wMin < 1e-4 {
			wMin = 1e-4
		}
		if wMax > 1e4 {
			wMax = 1e4
		}
	}

	return logspace(math.Log10(wMin), math.Log10(wMax), nPoints), nil
}

func logspace(start, stop float64, n int) []float64 {
	if n == 1 {
		return []float64{math.Pow(10, start)}
	}
	out := make([]float64, n)
	step := (stop - start) / float64(n-1)
	for i := range out {
		out[i] = math.Pow(10, start+float64(i)*step)
	}
	return out
}

func ioDelayTotal(sys *System, i, j int) float64 {
	var tau float64
	if sys.InputDelay != nil {
		tau += sys.InputDelay[j]
	}
	if sys.OutputDelay != nil {
		tau += sys.OutputDelay[i]
	}
	if sys.Delay != nil {
		dRaw := sys.Delay.RawMatrix()
		tau += dRaw.Data[i*dRaw.Stride+j]
	}
	return tau
}

func applyIODelayPhase(sys *System, omega []float64, data []complex128, p, m int, includeDelayMatrix bool) {
	delay := effectiveIODelayMatrix(sys, p, m, includeDelayMatrix)
	if delay == nil {
		return
	}

	pm := p * m
	for k, w := range omega {
		var s complex128
		s = newTimeDomain(sys.Dt).frequencyVariable(w)
		applyIODelayMatrixAtS(sys, s, data[k*pm:(k+1)*pm], p, m, delay)
	}
}

func applyIODelayAtS(sys *System, s complex128, data []complex128, p, m int, includeDelayMatrix bool) {
	delay := effectiveIODelayMatrix(sys, p, m, includeDelayMatrix)
	if delay == nil {
		return
	}
	applyIODelayMatrixAtS(sys, s, data, p, m, delay)
}

func applyIODelayMatrixAtS(sys *System, s complex128, data []complex128, p, m int, delay *mat.Dense) {
	dRaw := delay.RawMatrix()
	for i := range p {
		for j := range m {
			tau := dRaw.Data[i*dRaw.Stride+j]
			if tau == 0 {
				continue
			}
			off := i*m + j
			if sys.IsContinuous() {
				data[off] *= cmplx.Exp(-s * complex(tau, 0))
				continue
			}
			d := int(math.Round(tau))
			for range d {
				data[off] /= s
			}
		}
	}
}

func fillComplexPencil(dst []complex128, a []float64, aStride int, e []float64, eStride int, s complex128, n int) {
	for i := range n {
		row := i * n
		aRow := i * aStride
		if e == nil {
			for j := range n {
				dst[row+j] = -complex(a[aRow+j], 0)
			}
			dst[row+i] += s
			continue
		}
		eRow := i * eStride
		for j := range n {
			dst[row+j] = s*complex(e[eRow+j], 0) - complex(a[aRow+j], 0)
		}
	}
}

func copyRealMatrixToComplex(dst []complex128, src []float64, stride, rows, cols int) {
	if src == nil {
		clear(dst[:rows*cols])
		return
	}
	for i := range rows {
		for j := range cols {
			dst[i*cols+j] = complex(src[i*stride+j], 0)
		}
	}
}

func cSolveInPlace(a, b []complex128, n, nrhs int) error {
	if n == 1 {
		if a[0] == 0 {
			return fmt.Errorf("controlsys: singular complex matrix: %w", ErrSingularTransform)
		}
		for j := range nrhs {
			b[j] /= a[0]
		}
		return nil
	}

	maxAbs := 0.0
	for _, v := range a[:n*n] {
		if av := cmplx.Abs(v); av > maxAbs {
			maxAbs = av
		}
	}
	tol := float64(n) * maxAbs * eps()
	if tol == 0 {
		tol = 1e-15
	}

	for k := range n {
		pivot := k
		best := cmplx.Abs(a[k*n+k])
		for i := k + 1; i < n; i++ {
			if candidate := cmplx.Abs(a[i*n+k]); candidate > best {
				best = candidate
				pivot = i
			}
		}
		if best < tol {
			return fmt.Errorf("controlsys: singular complex matrix: %w", ErrSingularTransform)
		}
		if pivot != k {
			for j := range n {
				a[k*n+j], a[pivot*n+j] = a[pivot*n+j], a[k*n+j]
			}
			for j := range nrhs {
				b[k*nrhs+j], b[pivot*nrhs+j] = b[pivot*nrhs+j], b[k*nrhs+j]
			}
		}

		akk := a[k*n+k]
		for i := k + 1; i < n; i++ {
			factor := a[i*n+k] / akk
			a[i*n+k] = factor
			for j := k + 1; j < n; j++ {
				a[i*n+j] -= factor * a[k*n+j]
			}
		}
	}

	for i := 1; i < n; i++ {
		for k := 0; k < i; k++ {
			factor := a[i*n+k]
			for j := range nrhs {
				b[i*nrhs+j] -= factor * b[k*nrhs+j]
			}
		}
	}
	for i := n - 1; i >= 0; i-- {
		for k := i + 1; k < n; k++ {
			factor := a[i*n+k]
			for j := range nrhs {
				b[i*nrhs+j] -= factor * b[k*nrhs+j]
			}
		}
		diag := a[i*n+i]
		for j := range nrhs {
			b[i*nrhs+j] /= diag
		}
	}
	return nil
}

// complexFlatToGrid returns the rows of the row-major p×m data, aliasing it.
func complexFlatToGrid(data []complex128, p, m int) [][]complex128 {
	result := make([][]complex128, p)
	for i := range p {
		result[i] = data[i*m : (i+1)*m : (i+1)*m]
	}
	return result
}

func freqResponseLFT(sys *System, omega []float64, p, m int) (*FreqResponseMatrix, error) {
	data := make([]complex128, len(omega)*p*m)
	eval := newFrequencyEvaluator(sys).pointEval()
	td := newTimeDomain(sys.Dt)
	for k, w := range omega {
		if err := evalWithPoleLimit(eval, sys, td.frequencyVariable(w), data[k*p*m:(k+1)*p*m]); err != nil {
			return nil, err
		}
	}
	return newFreqResponseMatrix(data, omega, p, m, nil, nil), nil
}

type lftWorkspace struct {
	bd    *balancedDense
	h     []complex128 // (p+N)×(m+N) [H11 H12; H21 H22]
	delta []complex128
	lhs   []complex128 // I - H22·Δ
	x     []complex128 // N×m
	g     []complex128
}

// newLFTWorkspace balances the augmented plant [|A|+|E| B B2; C C2 ·] so
// the delay channels are solved on the same well-scaled realization as the
// I/O channels.
func newLFTWorkspace(sys *System, n, N, p, m int) *lftWorkspace {
	br := newRealizationCopy(sys, n, m, p)
	lft := sys.LFT
	mN, pN := m+N, p+N
	b := make([]float64, n*mN)
	c := make([]float64, pN*n)
	for i := range n {
		copy(b[i*mN:i*mN+m], br.b[i*m:(i+1)*m])
	}
	copy(c, br.c)
	if n > 0 {
		b2 := lft.B2.RawMatrix()
		copyStrided(b[m:], mN, b2.Data, b2.Stride, n, N)
		c2 := lft.C2.RawMatrix()
		copyStrided(c[p*n:], n, c2.Data, c2.Stride, N, n)
	}
	d := make([]float64, pN*mN)
	for _, blk := range []struct {
		src  *mat.Dense
		r, c int
	}{{sys.D, 0, 0}, {lft.D12, 0, m}, {lft.D21, p, 0}, {lft.D22, p, m}} {
		if blk.src == nil || blk.src.IsEmpty() {
			continue
		}
		raw := blk.src.RawMatrix()
		copyStrided(d[blk.r*mN+blk.c:], mN, raw.Data, raw.Stride, raw.Rows, raw.Cols)
	}
	br.m, br.p, br.b, br.c, br.d, br.dStride = mN, pN, b, c, d, mN
	br.balance(nil)
	cs := make([]complex128, pN*mN+N+N*N+N*m+p*m)
	return &lftWorkspace{
		bd:    newBalancedDenseOf(br),
		h:     cs[: pN*mN : pN*mN],
		delta: cs[pN*mN : pN*mN+N : pN*mN+N],
		lhs:   cs[pN*mN+N : pN*mN+N+N*N : pN*mN+N+N*N],
		x:     cs[pN*mN+N+N*N : pN*mN+N+N*N+N*m : pN*mN+N+N*N+N*m],
		g:     cs[pN*mN+N+N*N+N*m:],
	}
}

// evalFrLFTInto sets ws.g to G = H11 + H12·Δ·(I - H22·Δ)⁻¹·H21, or, where H
// or I - H22·Δ is singular, to the frozen-delay closed loop at s.
func evalFrLFTInto(ws *lftWorkspace, sys *System, s complex128, N, p, m int) error {
	cont := sys.IsContinuous()
	for j, tau := range sys.LFT.Tau {
		if cont {
			ws.delta[j] = cmplx.Exp(-s * complex(tau, 0))
		} else {
			d := int(math.Round(tau))
			ws.delta[j] = 1
			for range d {
				ws.delta[j] /= s
			}
		}
	}
	h := ws.h
	if err := ws.bd.evalInto(s, h); err != nil {
		return evalFrLFTFrozenInto(ws, s, N, p, m)
	}
	mN := m + N
	for i := range N {
		hRow := h[(p+i)*mN:]
		for j := range N {
			ws.lhs[i*N+j] = -hRow[m+j] * ws.delta[j]
		}
		ws.lhs[i*N+i] += 1
		copy(ws.x[i*m:(i+1)*m], hRow[:m])
	}
	if err := cSolveInPlace(ws.lhs, ws.x, N, m); err != nil {
		return evalFrLFTFrozenInto(ws, s, N, p, m)
	}
	for i := range p {
		hRow := h[i*mN:]
		for j := range m {
			v := hRow[j]
			for k := range N {
				v += hRow[m+k] * ws.delta[k] * ws.x[k*m+j]
			}
			ws.g[i*m+j] = v
		}
	}
	return nil
}

func cMulInto(dst, a, b []complex128, ar, ac, bc int) {
	for i := range ar {
		for j := range bc {
			var sum complex128
			for k := range ac {
				sum += a[i*ac+k] * b[k*bc+j]
			}
			dst[i*bc+j] = sum
		}
	}
}

// NicholsResult holds Nichols chart data: open-loop phase (degrees) vs magnitude (dB).
type NicholsResult struct {
	Omega      []float64
	magDB      []float64
	phase      []float64
	p, m       int
	InputName  []string
	OutputName []string
}

func (r *NicholsResult) MagDBAt(freq, output, input int) float64 {
	return newSampledScalarResponse(r.magDB, r.Omega, r.p, r.m).at(freq, output, input)
}

func (r *NicholsResult) PhaseAt(freq, output, input int) float64 {
	return newSampledScalarResponse(r.phase, r.Omega, r.p, r.m).at(freq, output, input)
}

func (sys *System) Nichols(omega []float64, nPoints int) (*NicholsResult, error) {
	bode, err := sys.Bode(omega, nPoints)
	if err != nil {
		return nil, err
	}
	if bode == nil {
		return nil, nil
	}

	phase := bode.phase
	p, m := bode.p, bode.m
	nw := len(bode.Omega)
	phaseResp := newSampledScalarResponse(phase, bode.Omega, p, m)

	for i := range p {
		for j := range m {
			base := phaseResp.at(0, i, j)
			shift := math.Ceil(base/360) * 360
			for k := range nw {
				phaseResp.set(k, i, j, phaseResp.at(k, i, j)-shift)
			}
		}
	}

	return &NicholsResult{
		Omega:      bode.Omega,
		magDB:      bode.magDB,
		phase:      phase,
		p:          p,
		m:          m,
		InputName:  bode.InputName,
		OutputName: bode.OutputName,
	}, nil
}

// SigmaResult holds singular value frequency response data.
type SigmaResult struct {
	Omega      []float64
	sv         []float64
	nSV        int
	InputName  []string
	OutputName []string
}

func (r *SigmaResult) At(freq, svIndex int) float64 {
	return r.sv[freq*r.nSV+svIndex]
}

func (r *SigmaResult) NSV() int {
	return r.nSV
}

func (sys *System) Sigma(omega []float64, nPoints int) (*SigmaResult, error) {
	if omega == nil {
		var err error
		omega, err = autoBodeFreqs(sys, nPoints)
		if err != nil {
			return nil, err
		}
	}

	_, m, p := sys.Dims()
	nSV := min(p, m)
	if nSV == 0 {
		return &SigmaResult{Omega: omega, nSV: 0, InputName: copyStringSlice(sys.InputName), OutputName: copyStringSlice(sys.OutputName)}, nil
	}

	resp, err := sys.FreqResponse(omega)
	if err != nil {
		return nil, err
	}
	if resp == nil {
		return &SigmaResult{Omega: omega, nSV: nSV, InputName: copyStringSlice(sys.InputName), OutputName: copyStringSlice(sys.OutputName)}, nil
	}

	data := resp.Data

	nw := len(omega)
	response := newSampledComplexResponse(data, omega, p, m)
	allSV := make([]float64, nw*nSV)
	var ws *complexSVDWorkspace
	if p != 1 || m != 1 {
		ws = newComplexSVDWorkspace(p, m)
	}

	for k := range nw {
		response.singularValues(allSV[k*nSV:(k+1)*nSV], ws, k)
	}

	return &SigmaResult{Omega: omega, sv: allSV, nSV: nSV, InputName: copyStringSlice(sys.InputName), OutputName: copyStringSlice(sys.OutputName)}, nil
}

func cInvertInto(dst, aug, src []complex128, n int) error {
	if n == 0 {
		return nil
	}
	if n == 1 {
		if src[0] == 0 {
			return fmt.Errorf("controlsys: singular complex matrix: %w", ErrSingularTransform)
		}
		dst[0] = 1 / src[0]
		return nil
	}

	w := 2 * n
	for i := range n {
		row := i * w
		copy(aug[row:row+n], src[i*n:(i+1)*n])
		for j := n; j < w; j++ {
			aug[row+j] = 0
		}
		aug[row+n+i] = 1
	}

	maxAbs := 0.0
	for _, v := range src[:n*n] {
		if a := cmplx.Abs(v); a > maxAbs {
			maxAbs = a
		}
	}
	tol := float64(n) * maxAbs * eps()
	if tol == 0 {
		tol = 1e-15
	}

	for col := range n {
		pivot := -1
		best := 0.0
		for row := col; row < n; row++ {
			v := cmplx.Abs(aug[row*w+col])
			if v > best {
				best = v
				pivot = row
			}
		}
		if best < tol {
			return fmt.Errorf("controlsys: singular complex matrix: %w", ErrSingularTransform)
		}
		if pivot != col {
			for j := range w {
				aug[col*w+j], aug[pivot*w+j] = aug[pivot*w+j], aug[col*w+j]
			}
		}
		inv := 1 / aug[col*w+col]
		for j := col; j < w; j++ {
			aug[col*w+j] *= inv
		}
		for row := range n {
			if row == col {
				continue
			}
			factor := aug[row*w+col]
			if factor == 0 {
				continue
			}
			for j := col; j < w; j++ {
				aug[row*w+j] -= factor * aug[col*w+j]
			}
		}
	}

	for i := range n {
		copy(dst[i*n:(i+1)*n], aug[i*w+n:i*w+w])
	}
	return nil
}
