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

func (sys *System) FreqResponse(omega []float64) (*FreqResponseMatrix, error) {
	return newFrequencyEvaluator(sys).response(omega)
}

// FreqResponsePointwise evaluates the frequency response with guaranteed
// per-frequency single-point arithmetic: the value at each omega[k] is
// bit-identical to FreqResponse([]float64{omega[k]}), regardless of
// len(omega). FreqResponse may switch long sweeps of delay-free
// state-space models to a transfer-function conversion whose values are
// close but not bit-identical to the single-point path;
// FreqResponsePointwise never does, at the cost of one dense solve per
// frequency. Use it when downstream comparisons require sweep results to
// reproduce single-point evaluations exactly.
func (sys *System) FreqResponsePointwise(omega []float64) (*FreqResponseMatrix, error) {
	return newFrequencyEvaluator(sys).responsePointwise(omega)
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

func (sys *System) EvalFr(s complex128) ([][]complex128, error) {
	return newFrequencyEvaluator(sys).eval(s)
}

type frequencyEvaluator struct {
	sys *System
	n   int
	m   int
	p   int
}

const directFrequencySweepWorkLimit = 80

func newFrequencyEvaluator(sys *System) frequencyEvaluator {
	n, m, p := sys.Dims()
	return frequencyEvaluator{sys: sys, n: n, m: m, p: p}
}

func (e frequencyEvaluator) response(omega []float64) (*FreqResponseMatrix, error) {
	if len(omega) == 0 {
		return nil, nil
	}
	if e.sys.HasInternalDelay() {
		if e.sys.IsDescriptor() {
			return nil, fmt.Errorf("FreqResponse: descriptor models with internal delays are not supported: %w", ErrDescriptorUnsupported)
		}
		resp, err := freqResponseLFT(e.sys, omega, e.p, e.m)
		if err != nil {
			return nil, err
		}
		applyIODelayPhase(e.sys, omega, resp.Data, e.p, e.m, true)
		resp.InputName = copyStringSlice(e.sys.InputName)
		resp.OutputName = copyStringSlice(e.sys.OutputName)
		return resp, nil
	}

	nw := len(omega)
	pm := e.p * e.m
	data := make([]complex128, nw*pm)
	if e.sys.IsDescriptor() {
		if err := e.evalStateSpaceSweepInto(omega, data); err != nil {
			return nil, err
		}
		applyIODelayPhase(e.sys, omega, data, e.p, e.m, true)
		return e.matrix(data, omega), nil
	}
	if e.useStateSpaceSweep(nw) {
		if err := e.evalStateSpaceSweepInto(omega, data); err == nil {
			applyIODelayPhase(e.sys, omega, data, e.p, e.m, true)
			return e.matrix(data, omega), nil
		}
	}
	if err := e.evalTransferFunctionSweepInto(omega, data); err != nil {
		return nil, err
	}
	applyIODelayPhase(e.sys, omega, data, e.p, e.m, false)
	return e.matrix(data, omega), nil
}

// responsePointwise evaluates each frequency exactly as response would for
// a one-element sweep: direct state-space solve first, per-point
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

	pm := e.p * e.m
	data := make([]complex128, len(omega)*pm)
	ws := newSSEvalWorkspace(e.n, e.p, e.m)
	// The effective delay matrices are pure functions of sys, which is
	// not mutated across the sweep, so hoisting them out of the loop
	// keeps every per-point value bit-identical while dropping the
	// per-omega allocations applyIODelayAtS would repeat.
	delaySS := effectiveIODelayMatrix(e.sys, e.p, e.m, true)
	var delayTF *mat.Dense
	var tf *TransferFunc
	for k, w := range omega {
		s := e.sAt(w)
		dst := data[k*pm : (k+1)*pm]
		if err := evalFrSSInto(ws, e.sys, s, e.n, e.p, e.m); err == nil {
			copy(dst, ws.g[:pm])
			if delaySS != nil {
				applyIODelayMatrixAtS(e.sys, s, dst, e.p, e.m, delaySS)
			}
			continue
		}
		if tf == nil {
			res, err := e.sys.TransferFunction(nil)
			if err != nil {
				return nil, err
			}
			tf = res.TF
			delayTF = effectiveIODelayMatrix(e.sys, e.p, e.m, false)
		}
		tf.evalInto(s, dst)
		if delayTF != nil {
			applyIODelayMatrixAtS(e.sys, s, dst, e.p, e.m, delayTF)
		}
	}
	return e.matrix(data, omega), nil
}

func (e frequencyEvaluator) eval(s complex128) ([][]complex128, error) {
	pm := e.p * e.m

	if e.sys.HasInternalDelay() {
		if e.sys.IsDescriptor() {
			return nil, fmt.Errorf("EvalFr: descriptor models with internal delays are not supported: %w", ErrDescriptorUnsupported)
		}
		g, err := evalFrLFT(e.sys, s, e.p, e.m)
		if err != nil {
			return nil, err
		}
		applyIODelayAtS(e.sys, s, g, e.p, e.m, true)
		return complexFlatToGrid(g, e.p, e.m), nil
	}

	data := make([]complex128, pm)
	if err := e.evalStateSpaceInto(s, data); err != nil {
		return nil, err
	}
	applyIODelayAtS(e.sys, s, data, e.p, e.m, true)
	return complexFlatToGrid(data, e.p, e.m), nil
}

func (e frequencyEvaluator) sAt(w float64) complex128 {
	return newTimeDomain(e.sys.Dt).frequencyVariable(w)
}

func (e frequencyEvaluator) evalStateSpaceInto(s complex128, dst []complex128) error {
	ws := newSSEvalWorkspace(e.n, e.p, e.m)
	if err := evalFrSSInto(ws, e.sys, s, e.n, e.p, e.m); err != nil {
		return err
	}
	copy(dst, ws.g[:e.p*e.m])
	return nil
}

func (e frequencyEvaluator) useStateSpaceSweep(nw int) bool {
	if nw <= 1 || e.n == 0 {
		return true
	}
	return nw <= directFrequencySweepWorkLimit/e.n
}

func (e frequencyEvaluator) evalStateSpaceSweepInto(omega []float64, dst []complex128) error {
	pm := e.p * e.m
	ws := newSSEvalWorkspace(e.n, e.p, e.m)
	for k, w := range omega {
		if err := evalFrSSInto(ws, e.sys, e.sAt(w), e.n, e.p, e.m); err != nil {
			return err
		}
		copy(dst[k*pm:(k+1)*pm], ws.g[:pm])
	}
	return nil
}

func (e frequencyEvaluator) evalTransferFunctionSweepInto(omega []float64, dst []complex128) error {
	res, err := e.sys.TransferFunction(nil)
	if err != nil {
		return err
	}
	pm := e.p * e.m
	for k, w := range omega {
		res.TF.evalInto(e.sAt(w), dst[k*pm:(k+1)*pm])
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

type ssEvalWorkspace struct {
	pencil []complex128
	rhs    []complex128
	g      []complex128
}

func newSSEvalWorkspace(n, p, m int) *ssEvalWorkspace {
	return &ssEvalWorkspace{
		pencil: make([]complex128, n*n),
		rhs:    make([]complex128, n*m),
		g:      make([]complex128, p*m),
	}
}

func evalFrSSInto(ws *ssEvalWorkspace, sys *System, s complex128, n, p, m int) error {
	var dData []float64
	var dStride int
	if sys.D != nil {
		dRaw := sys.D.RawMatrix()
		dData, dStride = dRaw.Data, dRaw.Stride
	}
	if n == 0 {
		copyRealMatrixToComplex(ws.g, dData, dStride, p, m)
		return nil
	}

	aRaw := sys.A.RawMatrix()
	if sys.E == nil {
		fillComplexPencil(ws.pencil, aRaw.Data, aRaw.Stride, nil, 0, s, n)
	} else {
		eRaw := sys.E.RawMatrix()
		fillComplexPencil(ws.pencil, aRaw.Data, aRaw.Stride, eRaw.Data, eRaw.Stride, s, n)
	}
	bRaw := sys.B.RawMatrix()
	copyRealMatrixToComplex(ws.rhs, bRaw.Data, bRaw.Stride, n, m)
	if err := cSolveInPlace(ws.pencil, ws.rhs, n, m); err != nil {
		return err
	}
	cRaw := sys.C.RawMatrix()
	cRealMulComplexInto(ws.g, cRaw.Data, cRaw.Stride, ws.rhs, dData, dStride, n, p, m)
	return nil
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

func cRealMulComplexInto(dst []complex128, a []float64, aStride int, b []complex128, d []float64, dStride, inner, rows, cols int) {
	for i := range rows {
		for j := range cols {
			var sum complex128
			for k := range inner {
				sum += complex(a[i*aStride+k], 0) * b[k*cols+j]
			}
			if d != nil {
				sum += complex(d[i*dStride+j], 0)
			}
			dst[i*cols+j] = sum
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

func complexFlatToGrid(data []complex128, p, m int) [][]complex128 {
	result := make([][]complex128, p)
	for i := range p {
		result[i] = make([]complex128, m)
		copy(result[i], data[i*m:(i+1)*m])
	}
	return result
}

func freqResponseLFT(sys *System, omega []float64, p, m int) (*FreqResponseMatrix, error) {
	nw := len(omega)
	data := make([]complex128, nw*p*m)
	n, _, _ := sys.Dims()
	N := sys.internalDelayCount()
	ws := newLFTWorkspace(n, N, p, m)

	for k, w := range omega {
		var s complex128
		s = newTimeDomain(sys.Dt).frequencyVariable(w)
		if err := evalFrLFTInto(ws, sys, s, n, N, p, m); err != nil {
			return nil, err
		}
		copy(data[k*p*m:], ws.g[:p*m])
	}

	return newFreqResponseMatrix(data, omega, p, m, nil, nil), nil
}

type lftWorkspace struct {
	sIA       []complex128
	resolvent []complex128
	hTemp     []complex128
	H11       []complex128
	H12       []complex128
	H21       []complex128
	H22       []complex128
	delta     []complex128
	H22D      []complex128
	ImH22D    []complex128
	invBuf    []complex128
	invResult []complex128
	temp      []complex128
	g         []complex128
}

func newLFTWorkspace(n, N, p, m int) *lftWorkspace {
	maxInv := max(n, N)
	return &lftWorkspace{
		sIA:       make([]complex128, n*n),
		resolvent: make([]complex128, n*n),
		hTemp:     make([]complex128, n*max(m, N, p)),
		H11:       make([]complex128, p*m),
		H12:       make([]complex128, p*N),
		H21:       make([]complex128, N*m),
		H22:       make([]complex128, N*N),
		delta:     make([]complex128, N),
		H22D:      make([]complex128, N*N),
		ImH22D:    make([]complex128, N*N),
		invBuf:    make([]complex128, maxInv*2*maxInv),
		invResult: make([]complex128, N*N),
		temp:      make([]complex128, N*m),
		g:         make([]complex128, p*m),
	}
}

func evalFrLFTInto(ws *lftWorkspace, sys *System, s complex128, n, N, p, m int) error {
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	var dData []float64
	var dStride int
	if sys.D != nil {
		dR := sys.D.RawMatrix()
		dData, dStride = dR.Data, dR.Stride
	}

	b2Raw := sys.LFT.B2.RawMatrix()
	c2Raw := sys.LFT.C2.RawMatrix()
	var d12Data, d21Data, d22Data []float64
	var d12Stride, d21Stride, d22Stride int
	if sys.LFT.D12 != nil {
		r := sys.LFT.D12.RawMatrix()
		d12Data, d12Stride = r.Data, r.Stride
	}
	if sys.LFT.D21 != nil {
		r := sys.LFT.D21.RawMatrix()
		d21Data, d21Stride = r.Data, r.Stride
	}
	if sys.LFT.D22 != nil {
		r := sys.LFT.D22.RawMatrix()
		d22Data, d22Stride = r.Data, r.Stride
	}

	if err := cResolventInto(ws.resolvent, ws.sIA, ws.invBuf, aRaw.Data, aRaw.Stride, s, n); err != nil {
		return err
	}

	cComputeHInto(ws.H11, ws.hTemp, ws.resolvent, cRaw.Data, cRaw.Stride, bRaw.Data, bRaw.Stride, dData, dStride, n, p, m)
	cComputeHInto(ws.H12, ws.hTemp, ws.resolvent, cRaw.Data, cRaw.Stride, b2Raw.Data, b2Raw.Stride, d12Data, d12Stride, n, p, N)
	cComputeHInto(ws.H21, ws.hTemp, ws.resolvent, c2Raw.Data, c2Raw.Stride, bRaw.Data, bRaw.Stride, d21Data, d21Stride, n, N, m)
	cComputeHInto(ws.H22, ws.hTemp, ws.resolvent, c2Raw.Data, c2Raw.Stride, b2Raw.Data, b2Raw.Stride, d22Data, d22Stride, n, N, N)

	cont := sys.IsContinuous()
	for j := range N {
		tau := sys.LFT.Tau[j]
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

	cMulDiagRightInto(ws.H22D, ws.H22, ws.delta, N, N)

	for i := 0; i < N*N; i++ {
		ws.ImH22D[i] = -ws.H22D[i]
	}
	for i := range N {
		ws.ImH22D[i*N+i] += 1
	}

	if err := cInvertInto(ws.invResult, ws.invBuf, ws.ImH22D, N); err != nil {
		return err
	}

	cMulInto(ws.temp, ws.invResult, ws.H21, N, N, m)
	cMulDiagLeftInto(ws.temp, ws.delta, ws.temp, N, m)

	copy(ws.g[:p*m], ws.H11[:p*m])
	cAddMul(ws.g, ws.H12, ws.temp, p, N, m)

	return nil
}

// evalFrLFT computes G(s) via LFT: G = H11 + H12 * Delta * (I - H22*Delta)^{-1} * H21
// Returns flat p*m complex slice (row-major).
func evalFrLFT(sys *System, s complex128, p, m int) ([]complex128, error) {
	n, _, _ := sys.Dims()
	N := sys.internalDelayCount()
	ws := newLFTWorkspace(n, N, p, m)
	if err := evalFrLFTInto(ws, sys, s, n, N, p, m); err != nil {
		return nil, err
	}
	result := make([]complex128, p*m)
	copy(result, ws.g[:p*m])
	return result, nil
}

func cResolventInto(dst, sIA, invBuf []complex128, aData []float64, aStride int, s complex128, n int) error {
	if n == 0 {
		return nil
	}
	for i := range n {
		row := i * n
		aRow := i * aStride
		for j := range n {
			sIA[row+j] = -complex(aData[aRow+j], 0)
		}
		sIA[row+i] += s
	}
	return cInvertInto(dst, invBuf, sIA, n)
}

func cDescriptorResolventInto(dst, sEA, invBuf []complex128, aData []float64, aStride int, eData []float64, eStride int, s complex128, n int) error {
	if n == 0 {
		return nil
	}
	for i := range n {
		row := i * n
		aRow := i * aStride
		eRow := i * eStride
		for j := range n {
			sEA[row+j] = s*complex(eData[eRow+j], 0) - complex(aData[aRow+j], 0)
		}
	}
	return cInvertInto(dst, invBuf, sEA, n)
}

func cComputeHInto(dst, temp, resolvent []complex128, cData []float64, cStride int, bData []float64, bStride int, dData []float64, dStride, n, rows, cols int) {
	for i := range dst[:rows*cols] {
		dst[i] = 0
	}
	if n > 0 {
		for i := range n {
			for j := range cols {
				var sum complex128
				for k := range n {
					sum += resolvent[i*n+k] * complex(bData[k*bStride+j], 0)
				}
				temp[i*cols+j] = sum
			}
		}
		for i := range rows {
			for j := range cols {
				var sum complex128
				for k := range n {
					sum += complex(cData[i*cStride+k], 0) * temp[k*cols+j]
				}
				dst[i*cols+j] = sum
			}
		}
	}
	if dData != nil {
		for i := range rows {
			for j := range cols {
				dst[i*cols+j] += complex(dData[i*dStride+j], 0)
			}
		}
	}
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

func cAddMul(dst, a, b []complex128, ar, ac, bc int) {
	for i := range ar {
		for j := range bc {
			var sum complex128
			for k := range ac {
				sum += a[i*ac+k] * b[k*bc+j]
			}
			dst[i*bc+j] += sum
		}
	}
}

func cMulDiagRightInto(dst, a, diag []complex128, rows, cols int) {
	for i := range rows {
		for j := range cols {
			dst[i*cols+j] = a[i*cols+j] * diag[j]
		}
	}
}

func cMulDiagLeftInto(dst, diag, a []complex128, rows, cols int) {
	for i := range rows {
		for j := range cols {
			dst[i*cols+j] = diag[i] * a[i*cols+j]
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
