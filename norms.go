package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"
	"sync/atomic"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

const NormH2 = 2

// Norm returns the H2 norm (normType 2) or the L∞ norm (normType +Inf) of
// sys, following MATLAB norm
// (https://www.mathworks.com/help/control/ref/dynamicsystem.norm.html).
//
// The H2 norm of an unstable model is +Inf. The L∞ norm is the peak gain
// over frequency without regard to stability; it equals the H∞ norm for
// stable models and is +Inf when a pole lies on the stability boundary
// (imaginary axis or unit circle).
func Norm(sys *System, normType float64) (float64, error) {
	if normType == 2 {
		norm, err := H2Norm(sys)
		if errors.Is(err, ErrUnstable) {
			return math.Inf(1), nil
		}
		return norm, err
	}
	if math.IsInf(normType, 1) {
		norm, _, err := linfNorm(sys)
		return norm, err
	}
	return 0, fmt.Errorf("controlsys: normType must be 2 or Inf, got %g", normType)
}

// H2Norm computes the H2 norm of a stable LTI system.
//
// For continuous systems with D ≠ 0, or with a delayed direct feedthrough
// through internal delays, the H2 norm is infinite. Input, output and I/O
// delays do not change the H2 norm. Discrete internal delays are absorbed
// exactly; continuous strictly proper internal-delay models return
// ErrContinuousInternalDelay.
func H2Norm(sys *System) (float64, error) {
	if err := newDescriptorPolicy(sys).requireStandard("H2Norm"); err != nil {
		return 0, err
	}
	if sys.HasInternalDelay() && sys.IsContinuous() &&
		(!allZeroDense(sys.D) || lftHasDirectFeedthrough(sys.LFT)) {
		return math.Inf(1), nil
	}
	sys, err := finiteDimensionalModel(sys, "H2Norm")
	if err != nil {
		return 0, err
	}
	policy := newEnergyAnalysisPolicy(sys)
	n, m, p := policy.n, policy.m, policy.p

	if n == 0 {
		if sys.IsContinuous() && !allZeroDense(sys.D) {
			return math.Inf(1), nil
		}
		return frobNormD(sys.D, p, m), nil
	}

	if err := policy.requireStable(ErrUnstable); err != nil {
		return 0, err
	}

	if sys.IsContinuous() && !allZeroDense(sys.D) {
		return math.Inf(1), nil
	}

	At, Q, err := policy.gramianInputs(GramObservability)
	if err != nil {
		return 0, err
	}
	X, err := policy.solveLyapunov(At, Q)
	if err != nil {
		return 0, err
	}

	xb := make([]float64, n*m)
	xRaw := X.RawMatrix()
	bRaw := sys.B.RawMatrix()
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: xRaw.Stride, Data: xRaw.Data},
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: m, Stride: m, Data: xb})

	tr := 0.0
	for i := range n {
		for j := range m {
			tr += bRaw.Data[i*bRaw.Stride+j] * xb[i*m+j]
		}
	}

	if sys.IsDiscrete() {
		dRaw := sys.D.RawMatrix()
		for i := range p {
			for j := range m {
				v := dRaw.Data[i*dRaw.Stride+j]
				tr += v * v
			}
		}
	}

	if tr < 0 {
		tr = 0
	}
	return math.Sqrt(tr), nil
}

// HSV computes the Hankel singular values of a stable LTI system in descending order.
//
// Discrete internal delays are absorbed exactly, so the result has one value
// per original state plus one per delay sample. Continuous internal-delay
// models return ErrContinuousInternalDelay.
func HSV(sys *System) ([]float64, error) {
	if err := newDescriptorPolicy(sys).requireStandard("HSV"); err != nil {
		return nil, err
	}
	sys, err := finiteDimensionalModel(sys, "HSV")
	if err != nil {
		return nil, err
	}
	policy := newEnergyAnalysisPolicy(sys)
	n := policy.n
	if n == 0 {
		return nil, nil
	}

	if err := policy.requireStable(ErrUnstable); err != nil {
		return nil, err
	}

	A, Qc, err := policy.gramianInputs(GramControllability)
	if err != nil {
		return nil, err
	}
	Wc, err := policy.solveLyapunov(A, Qc)
	if err != nil {
		return nil, err
	}
	At, Qo, err := policy.gramianInputs(GramObservability)
	if err != nil {
		return nil, err
	}
	Wo, err := policy.solveLyapunov(At, Qo)
	if err != nil {
		return nil, err
	}

	return hsvFromGramians(Wc, Wo, n)
}

func hsvFromGramians(Wc, Wo *mat.Dense, n int) ([]float64, error) {
	wcRaw := Wc.RawMatrix()
	lc := make([]float64, n*n)
	copyStrided(lc, n, wcRaw.Data, wcRaw.Stride, n, n)

	if !impl.Dpotrf(blas.Lower, n, lc, n) {
		return eigenvalueHSV(Wc, Wo, n), nil
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lc[i*n+j] = 0
		}
	}

	woRaw := Wo.RawMatrix()
	lo := make([]float64, n*n)
	copyStrided(lo, n, woRaw.Data, woRaw.Stride, n, n)

	if !impl.Dpotrf(blas.Lower, n, lo, n) {
		return eigenvalueHSV(Wc, Wo, n), nil
	}
	for i := range n {
		for j := i + 1; j < n; j++ {
			lo[i*n+j] = 0
		}
	}

	mData := make([]float64, n*n)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: lo},
		blas64.General{Rows: n, Cols: n, Stride: n, Data: lc},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: mData})

	s := make([]float64, n)
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, n, n, mData, n, s, nil, 1, nil, 1, wq, -1)
	work := make([]float64, int(wq[0]))
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, n, n, mData, n, s, nil, 1, nil, 1, work, len(work))

	return s, nil
}

func eigenvalueHSV(Wc, Wo *mat.Dense, n int) []float64 {
	wcRaw := Wc.RawMatrix()
	woRaw := Wo.RawMatrix()
	prod := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: wcRaw.Stride, Data: wcRaw.Data},
		blas64.General{Rows: n, Cols: n, Stride: woRaw.Stride, Data: woRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: prod})

	var eig mat.Eigen
	ok := eig.Factorize(mat.NewDense(n, n, prod), mat.EigenNone)
	if !ok {
		return make([]float64, n)
	}
	vals := eig.Values(nil)
	hsv := make([]float64, n)
	for i, v := range vals {
		r := real(v)
		if r < 0 {
			r = 0
		}
		hsv[i] = math.Sqrt(r)
	}
	sort.Sort(sort.Reverse(sort.Float64Slice(hsv)))
	return hsv
}

// HinfNorm computes the H∞ norm (peak gain) of an LTI system and the
// frequency at which it occurs.
//
// Unstable models, including poles on the stability boundary, return
// norm = omega = +Inf and a nil error, as MATLAB hinfnorm
// (https://www.mathworks.com/help/robust/ref/dynamicsystem.hinfnorm.html)
// does. Use Norm(sys, math.Inf(1)) for the L∞ peak gain of an unstable
// model.
//
// Input, output and I/O delays do not change the norm. Discrete internal
// delays are absorbed exactly; continuous internal-delay models return
// ErrContinuousInternalDelay because their stability cannot be decided from
// a finite pole set.
func HinfNorm(sys *System) (norm float64, omega float64, err error) {
	if err := newDescriptorPolicy(sys).requireStandard("HinfNorm"); err != nil {
		return 0, 0, err
	}
	sys, err = finiteDimensionalModel(sys, "HinfNorm")
	if err != nil {
		return 0, 0, err
	}
	policy := newEnergyAnalysisPolicy(sys)
	n, m, p := policy.n, policy.m, policy.p

	if n == 0 {
		sv := maxSVDense(sys.D, p, m)
		return sv, 0, nil
	}

	if err := policy.requireStable(ErrUnstable); err != nil {
		if errors.Is(err, ErrUnstable) {
			return math.Inf(1), math.Inf(1), nil
		}
		return 0, 0, err
	}
	return peakGain(sys)
}

// linfNorm returns the L∞ norm (peak gain regardless of stability) and the
// frequency at which it occurs. Poles on the stability boundary give +Inf at
// the frequency of that pole.
func linfNorm(sys *System) (norm float64, omega float64, err error) {
	if err := newDescriptorPolicy(sys).requireStandard("Norm"); err != nil {
		return 0, 0, err
	}
	sys, err = finiteDimensionalModel(sys, "Norm")
	if err != nil {
		return 0, 0, err
	}
	n, m, p := sys.Dims()
	if n == 0 {
		return maxSVDense(sys.D, p, m), 0, nil
	}
	poles, err := sys.Poles()
	if err != nil {
		return 0, 0, err
	}
	for _, pole := range poles {
		if !poleOnStabilityBoundary(pole, sys.IsContinuous(), poleStabilityTolerance(pole)) {
			continue
		}
		if sys.IsContinuous() {
			return math.Inf(1), math.Abs(imag(pole)), nil
		}
		return math.Inf(1), math.Abs(cmplx.Phase(pole)) / sys.Dt, nil
	}
	return peakGain(sys)
}

// peakGain computes sup_ω σ_max(G(jω)) of a model with no poles on the
// stability boundary from the Hamiltonian eigenvalue test; stability is not
// required. A sampled peak is first certified by one probe just above it;
// when the probe is inconclusive, the upper-bound search may still certify a
// raised peak, and bisection runs only if it does not.
// Discrete models are mapped by Tustin, and the peak frequency is unwarped
// back to the discrete axis.
func peakGain(sys *System) (norm float64, omega float64, err error) {
	n, m, p := sys.Dims()
	if sys.IsDiscrete() {
		csys, err := sys.Undiscretize()
		if err != nil {
			return 0, 0, err
		}
		norm, omega, err = peakGain(csys)
		return norm, 2 * math.Atan(omega*sys.Dt/2) / sys.Dt, err
	}

	gammaLow, omegaPeak := hinfLowerBound(sys, m, p)

	ws := newHamiltonianWS(sys, n, m, p)

	const tol = 1e-10
	gammaLow, omegaPeak, certified := ws.certifyPeak(sys, gammaLow, omegaPeak, tol)
	if certified {
		return gammaLow * (1 + tol/2), omegaPeak, nil
	}

	gammaLow, gammaHigh, omegaPeak, certified := ws.upperBound(sys, gammaLow, omegaPeak, tol)
	if certified {
		return gammaLow * (1 + tol/2), omegaPeak, nil
	}

	for range 100 {
		if gammaHigh-gammaLow < tol*gammaHigh {
			break
		}
		mid := (gammaLow + gammaHigh) / 2
		if !ws.hasImagEigs(mid) {
			gammaHigh = mid
			continue
		}
		peak, w, err := ws.candidatePeak(sys)
		if err != nil {
			gammaLow = mid
			continue
		}
		if peak < mid {
			gammaHigh = mid
			continue
		}
		gammaLow, omegaPeak, certified = ws.certifyPeak(sys, peak, w, tol)
		if certified {
			return gammaLow * (1 + tol/2), omegaPeak, nil
		}
	}

	return gammaHigh, omegaPeak, nil
}

// certifyPeak tries to prove that the attained gain gammaLow is within tol
// of the peak: no imaginary-axis Hamiltonian eigenvalue at
// gammaLow·(1+tol/2) makes that level an upper bound. Crossings found there
// raise gammaLow through their interval midpoints (Boyd–Balakrishnan,
// Bruinsma–Steinbuch), which converges quadratically. Near-axis eigenvalues
// whose candidate frequencies stay below the probe count as no crossing, the
// rule the bisection applies at every level. It reports false when the
// candidates cannot be evaluated, leaving the bisection to settle the peak.
func (ws *hamiltonianWS) certifyPeak(sys *System, gammaLow, omegaPeak, tol float64) (float64, float64, bool) {
	for range 20 {
		if !(gammaLow > 0) || math.IsInf(gammaLow, 1) {
			return gammaLow, omegaPeak, false
		}
		probe := gammaLow * (1 + tol/2)
		if !ws.hasImagEigs(probe) {
			return gammaLow, omegaPeak, true
		}
		peak, w, err := ws.candidatePeak(sys)
		if err != nil {
			return gammaLow, omegaPeak, false
		}
		if peak < probe {
			return gammaLow, omegaPeak, true
		}
		gammaLow, omegaPeak = peak, w
	}
	return gammaLow, omegaPeak, false
}

// upperBound doubles gammaHigh from 2·gammaLow until the Hamiltonian test
// shows no crossing there, under the candidate rule of certifyPeak and the
// bisection: near-axis eigenvalues whose candidate gains stay below gammaHigh
// do not count. The near-axis threshold grows with gamma, so without that
// rule a lightly damped mode is flagged at every level. A candidate gain at
// or above gammaHigh raises gammaLow through certifyPeak, which may settle
// the peak outright.
func (ws *hamiltonianWS) upperBound(sys *System, gammaLow, omegaPeak, tol float64) (float64, float64, float64, bool) {
	gammaHigh := math.Max(gammaLow*2, 1e-10)
	for range 50 {
		if !ws.hasImagEigs(gammaHigh) {
			break
		}
		peak, w, err := ws.candidatePeak(sys)
		if err == nil && peak < gammaHigh {
			break
		}
		if err == nil {
			var certified bool
			gammaLow, omegaPeak, certified = ws.certifyPeak(sys, peak, w, tol)
			if certified {
				return gammaLow, gammaLow * (1 + tol/2), omegaPeak, true
			}
		}
		gammaHigh = 2 * math.Max(gammaHigh, gammaLow)
	}
	return gammaLow, gammaHigh, omegaPeak, false
}

// hamiltonianWS holds pre-allocated buffers for the Hamiltonian eigenvalue test
// used across bisection iterations in HinfNorm.
type hamiltonianWS struct {
	n, m, p int
	nn      int
	dIsZero bool

	aData []float64
	aStr  int
	bData []float64
	bStr  int
	cData []float64
	cStr  int
	dData []float64
	dStr  int

	bbt []float64
	ctc []float64

	h     []float64
	cands []float64
	wr    []float64
	wi    []float64
	vs    []float64
	work  []float64

	r       []float64
	dtc     []float64
	bt      []float64
	rinvBt  []float64
	h11     []float64
	h12     []float64
	h21     []float64
	dRinvDt []float64
}

func newHamiltonianWS(sys *System, n, m, p int) *hamiltonianWS {
	nn := 2 * n
	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	dRaw := sys.D.RawMatrix()

	ws := &hamiltonianWS{
		n: n, m: m, p: p, nn: nn,
		dIsZero: allZeroDense(sys.D),
		aData:   aRaw.Data, aStr: aRaw.Stride,
		bData: bRaw.Data, bStr: bRaw.Stride,
		cData: cRaw.Data, cStr: cRaw.Stride,
		dData: dRaw.Data, dStr: dRaw.Stride,
		h:  make([]float64, nn*nn),
		wr: make([]float64, nn),
		wi: make([]float64, nn),
		vs: make([]float64, nn*nn),
	}

	ws.bbt = make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ws.bbt})

	ws.ctc = make([]float64, n*n)
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ws.ctc})

	if !ws.dIsZero {
		ws.r = make([]float64, m*m)
		ws.dtc = make([]float64, m*n)
		ws.bt = make([]float64, m*n)
		ws.rinvBt = make([]float64, m*n)
		ws.h11 = make([]float64, n*n)
		ws.h12 = make([]float64, n*n)
		ws.h21 = make([]float64, n*n)
		ws.dRinvDt = make([]float64, p*n)

		for i := range n {
			for j := range m {
				ws.bt[j*n+i] = bRaw.Data[i*bRaw.Stride+j]
			}
		}
	}

	wq := make([]float64, 1)
	impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		nn, ws.h, nn, ws.wr, ws.wi, ws.vs, nn, wq, -1, nil)
	ws.work = make([]float64, int(wq[0]))

	return ws
}

// hamiltonianEvals counts hasImagEigs calls, one Schur decomposition each.
var hamiltonianEvals atomic.Int64

func (ws *hamiltonianWS) hasImagEigs(gamma float64) bool {
	hamiltonianEvals.Add(1)
	n, m, p := ws.n, ws.m, ws.p
	nn := ws.nn
	g2 := gamma * gamma
	ws.cands = ws.cands[:0]

	h := ws.h
	for i := range len(h) {
		h[i] = 0
	}

	if ws.dIsZero {
		for i := range n {
			for j := range n {
				h[i*nn+j] = ws.aData[i*ws.aStr+j]
			}
		}

		scale := 1.0 / g2
		for i := range n {
			for j := range n {
				h[i*nn+(n+j)] = scale * ws.bbt[i*n+j]
			}
		}

		for i := range n {
			for j := range n {
				h[(n+i)*nn+j] = -ws.ctc[i*n+j]
			}
		}

		for i := range n {
			for j := range n {
				h[(n+i)*nn+(n+j)] = -ws.aData[j*ws.aStr+i]
			}
		}
	} else {
		r := ws.r
		for i := range m * m {
			r[i] = 0
		}
		for i := range m {
			r[i*m+i] = g2
		}
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: m, Stride: ws.dStr, Data: ws.dData},
			blas64.General{Rows: p, Cols: m, Stride: ws.dStr, Data: ws.dData},
			1, blas64.General{Rows: m, Cols: m, Stride: m, Data: r})

		if !impl.Dpotrf(blas.Upper, m, r, m) {
			return true
		}

		dtc := ws.dtc
		blas64.Gemm(blas.Trans, blas.NoTrans, 1,
			blas64.General{Rows: p, Cols: m, Stride: ws.dStr, Data: ws.dData},
			blas64.General{Rows: p, Cols: n, Stride: ws.cStr, Data: ws.cData},
			0, blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc})
		impl.Dpotrs(blas.Upper, m, n, r, m, dtc, n)

		rinvBt := ws.rinvBt
		copy(rinvBt, ws.bt)
		impl.Dpotrs(blas.Upper, m, n, r, m, rinvBt, n)

		h11 := ws.h11
		copyStrided(h11, n, ws.aData, ws.aStr, n, n)
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Stride: ws.bStr, Data: ws.bData},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc},
			1, blas64.General{Rows: n, Cols: n, Stride: n, Data: h11})

		h12 := ws.h12
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: n, Cols: m, Stride: ws.bStr, Data: ws.bData},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: rinvBt},
			0, blas64.General{Rows: n, Cols: n, Stride: n, Data: h12})

		h21 := ws.h21
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: n, Stride: ws.cStr, Data: ws.cData},
			blas64.General{Rows: p, Cols: n, Stride: ws.cStr, Data: ws.cData},
			0, blas64.General{Rows: n, Cols: n, Stride: n, Data: h21})

		dRinvDt := ws.dRinvDt
		blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
			blas64.General{Rows: p, Cols: m, Stride: ws.dStr, Data: ws.dData},
			blas64.General{Rows: m, Cols: n, Stride: n, Data: dtc},
			0, blas64.General{Rows: p, Cols: n, Stride: n, Data: dRinvDt})
		blas64.Gemm(blas.Trans, blas.NoTrans, -1,
			blas64.General{Rows: p, Cols: n, Stride: ws.cStr, Data: ws.cData},
			blas64.General{Rows: p, Cols: n, Stride: n, Data: dRinvDt},
			1, blas64.General{Rows: n, Cols: n, Stride: n, Data: h21})

		for i := range n {
			for j := range n {
				h[i*nn+j] = h11[i*n+j]
				h[i*nn+(n+j)] = h12[i*n+j]
				h[(n+i)*nn+j] = h21[i*n+j]
				h[(n+i)*nn+(n+j)] = -h11[j*n+i]
			}
		}
	}

	_, ok := impl.Dgees(lapack.SchurHess, lapack.SortNone, nil,
		nn, h, nn, ws.wr, ws.wi, ws.vs, nn, ws.work, len(ws.work), nil)
	if !ok {
		return true
	}

	threshold := math.Sqrt(eps()) * gamma
	for i := range nn {
		absLam := math.Sqrt(ws.wr[i]*ws.wr[i] + ws.wi[i]*ws.wi[i])
		if absLam > 0 && math.Abs(ws.wr[i]) < threshold*math.Max(1, absLam/gamma) && ws.wi[i] >= 0 {
			ws.cands = append(ws.cands, ws.wi[i])
		}
	}
	return len(ws.cands) > 0
}

// candidatePeak evaluates sigma_max at the near-axis eigenvalue frequencies
// found by hasImagEigs and at their midpoints (Bruinsma and Steinbuch). The
// Hamiltonian of a badly scaled system can show near-axis eigenvalues where
// no singular value reaches gamma, so only these evaluations certify a
// crossing.
func (ws *hamiltonianWS) candidatePeak(sys *System) (peak, omega float64, err error) {
	if len(ws.cands) == 0 {
		return 0, 0, ErrSchurFailed
	}
	slices.Sort(ws.cands)
	freqs := slices.Compact(ws.cands)
	for i := range len(freqs) - 1 {
		freqs = append(freqs, (freqs[i]+freqs[i+1])/2)
	}
	return sigmaMaxPointwise(sys, freqs)
}

// sigmaMaxPointwise returns the largest sigma_max over freqs and where it
// occurs. It solves the state space at each frequency: the batched sweep
// converts long, high-order sweeps to polynomials, which loses the peak.
func sigmaMaxPointwise(sys *System, freqs []float64) (peak, omega float64, err error) {
	_, m, p := sys.Dims()
	resp, err := sys.FreqResponsePointwise(freqs)
	if err != nil {
		return 0, 0, err
	}
	nSV := min(p, m)
	response := newSampledComplexResponse(resp.Data, freqs, p, m)
	var ws *complexSVDWorkspace
	if p != 1 || m != 1 {
		ws = newComplexSVDWorkspace(p, m)
	}
	sv := make([]float64, nSV)
	peak = math.Inf(-1)
	for k, w := range freqs {
		response.singularValues(sv, ws, k)
		if sv[0] > peak || math.IsNaN(sv[0]) {
			peak, omega = sv[0], w
		}
	}
	if math.IsNaN(peak) {
		return 0, 0, ErrSingularEquation
	}
	return peak, omega, nil
}

func hinfLowerBound(sys *System, m, p int) (gammaLow, omegaPeak float64) {
	poles, err := sys.Poles()
	if err != nil {
		return 0, 0
	}

	freqs := make([]float64, 0, 60)
	freqs = append(freqs, 0)

	wmin, wmax := math.Inf(1), 0.0
	for _, pole := range poles {
		w := cmplx.Abs(pole)
		if w > 0 {
			if w < wmin {
				wmin = w
			}
			if w > wmax {
				wmax = w
			}
		}
	}
	if wmax == 0 {
		wmin, wmax = 0.01, 100
	}
	wmin /= 10
	wmax *= 10

	for i := range 50 {
		w := wmin * math.Pow(wmax/wmin, float64(i)/49)
		freqs = append(freqs, w)
	}

	if peak, w, err := sigmaMaxPointwise(sys, freqs); err == nil {
		gammaLow, omegaPeak = peak, w
	}

	return gammaLow, omegaPeak
}

func allZeroDense(m *mat.Dense) bool {
	if m == nil {
		return true
	}
	raw := m.RawMatrix()
	for i := range raw.Rows {
		for j := range raw.Cols {
			if raw.Data[i*raw.Stride+j] != 0 {
				return false
			}
		}
	}
	return true
}

func frobNormD(D *mat.Dense, p, m int) float64 {
	if D == nil || p == 0 || m == 0 {
		return 0
	}
	return denseNorm(D)
}

func maxSVDense(D *mat.Dense, p, m int) float64 {
	if D == nil || p == 0 || m == 0 {
		return 0
	}
	raw := D.RawMatrix()
	data := make([]float64, p*m)
	copyStrided(data, m, raw.Data, raw.Stride, p, m)
	sv := make([]float64, min(p, m))
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, p, m, data, m, sv, nil, 1, nil, 1, wq, -1)
	work := make([]float64, int(wq[0]))
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, p, m, data, m, sv, nil, 1, nil, 1, work, len(work))
	if len(sv) == 0 {
		return 0
	}
	return sv[0]
}
