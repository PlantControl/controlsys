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

// NormH2 selects the H2 norm in Norm; pass math.Inf(1) for the L∞ norm.
const NormH2 = 2

// Norm returns the H2 norm (normType 2) or the L∞ norm (normType +Inf) of
// sys, following MATLAB norm
// (https://www.mathworks.com/help/control/ref/dynamicsystem.norm.html).
//
// The H2 norm of an unstable model is +Inf. The L∞ norm is the peak gain
// over frequency without regard to stability; it equals the H∞ norm for
// stable models and is +Inf when a pole lies on the stability boundary
// (imaginary axis or unit circle). For continuous internal-delay models the
// peak is computed as HinfNorm describes, stable or not; it is +Inf when the
// characteristic function det(sI−A)·det(I − H22(s)Δ(s)) has a root on the
// imaginary axis, which like the rational case counts hidden modes.
func Norm(sys *System, normType float64) (float64, error) {
	if err := requireSystem("Norm", sys); err != nil {
		return 0, err
	}
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
	return 0, fmt.Errorf("Norm: normType is %g, want 2 or Inf: %w", normType, ErrInvalidArgument)
}

// H2Norm computes the H2 norm of a stable LTI system.
//
// For continuous systems with D ≠ 0, or with a delayed direct feedthrough
// through internal delays, the H2 norm is infinite. Input, output and I/O
// delays do not change the H2 norm. Discrete internal delays are absorbed
// exactly; continuous strictly proper internal-delay models return
// ErrContinuousInternalDelay.
func H2Norm(sys *System) (float64, error) {
	if err := requireSystem("H2Norm", sys); err != nil {
		return 0, err
	}
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

	if n == 0 || m == 0 || p == 0 {
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
	if err := requireSystem("HSV", sys); err != nil {
		return nil, err
	}
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
		return eigenvalueHSV(Wc, Wo, n)
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
		return eigenvalueHSV(Wc, Wo, n)
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
	if !impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, n, n, mData, n, s, nil, 1, nil, 1, work, len(work)) {
		return nil, fmt.Errorf("HSV: singular value iteration did not converge: %w", ErrSchurFailed)
	}
	return s, nil
}

func eigenvalueHSV(Wc, Wo *mat.Dense, n int) ([]float64, error) {
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
		return nil, fmt.Errorf("HSV: eigenvalue iteration did not converge: %w", ErrSchurFailed)
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
	return hsv, nil
}

// HinfNorm computes the H∞ norm (peak gain) of an LTI system and the
// frequency at which it occurs, to 1e-10 relative accuracy. A continuous
// model whose gain approaches its peak σ_max(D) only as ω → ∞ returns
// omega = +Inf; the Nyquist frequency π/T plays that role for discrete
// models. The algorithm is that of MATLAB getPeakGain
// (https://www.mathworks.com/help/control/ref/dynamicsystem.getpeakgain.html):
// Bruinsma–Steinbuch iteration on the γ-crossings of the extended pencil.
//
// Unstable models, including poles on the stability boundary, return
// norm = omega = +Inf and a nil error, as MATLAB hinfnorm
// (https://www.mathworks.com/help/robust/ref/dynamicsystem.hinfnorm.html)
// does. Use Norm(sys, math.Inf(1)) for the L∞ peak gain of an unstable
// model.
//
// Input, output and I/O delays do not change the norm. Discrete internal
// delays are absorbed exactly.
//
// Continuous internal delays leave infinitely many poles, the roots of the
// entire function χ(s) = det(sI−A)·det(I − H22(s)Δ(s)), with
// H22 = D22 + C2(sI−A)⁻¹B2 and Δ = diag(e^{−sτ}). Stability is decided
// exactly by counting the right-half-plane roots of χ with the argument
// principle on a delay-aware adaptive grid (the Nyquist test of DiskMargin),
// and an unstable model, including a root on the imaginary axis, returns
// +Inf as above. The peak of a stable model is searched on that grid with
// the exact delay factors e^{−jωτ}, each local maximum refined by
// golden-section search; past the grid a resolvent bound on the delay LFT
// certifies that the gain stays below the peak. Between grid points the peak
// is certified to a relative 1e-9: second-order Taylor bounds from the
// descriptor form of the delay LFT, with Neumann-series bounds on its
// resolvent that hold for non-normal A, are bisected until no interval can
// exceed it, so a resonance narrower than the grid spacing is not missed.
// When the high-frequency limit exceeds every
// finite sample, the limit is returned with omega = +Inf, as σ_max(D) is for
// rational models; the gain may exceed it by the resolvent bound at the grid
// end, which the rational path's crossing probe rules out. Neutral-type models whose difference operator
// is not provably stable (‖D22‖ too large), and cases the grid cannot
// resolve or certify within its point budget, return ErrDelayUnsupported
// rather than an approximate answer. The MathWorks pages define the norm of
// an unstable model as Inf but say nothing specific about delays; this
// follows the exact-delay frequency response that MATLAB's frequency
// analysis of delay models uses.
func HinfNorm(sys *System) (norm float64, omega float64, err error) {
	if err := requireSystem("HinfNorm", sys); err != nil {
		return 0, 0, err
	}
	if err := newDescriptorPolicy(sys).requireStandard("HinfNorm"); err != nil {
		return 0, 0, err
	}
	if sys.IsContinuous() && sys.HasInternalDelay() {
		norm, omega, _, err = hinfNormDelayed(sys)
		if err != nil {
			return 0, 0, fmt.Errorf("HinfNorm: %w", err)
		}
		return norm, omega, nil
	}
	sys, err = finiteDimensionalModel(sys, "HinfNorm")
	if err != nil {
		return 0, 0, err
	}
	policy := newEnergyAnalysisPolicy(sys)
	n, m, p := policy.n, policy.m, policy.p

	if n == 0 || m == 0 || p == 0 {
		sv, err := maxSingularValue(sys.D)
		if err != nil {
			return 0, 0, fmt.Errorf("HinfNorm: %w", err)
		}
		return sv, 0, nil
	}

	poles, err := sys.Poles()
	if err != nil {
		return 0, 0, fmt.Errorf("HinfNorm: %w", err)
	}
	for _, pole := range poles {
		if poleOnOrOutsideStabilityBoundary(pole, sys.IsContinuous(), poleStabilityTolerance(pole)) {
			return math.Inf(1), math.Inf(1), nil
		}
	}
	norm, omega, err = peakGain(sys, poles)
	if err != nil {
		return 0, 0, fmt.Errorf("HinfNorm: %w", err)
	}
	return norm, omega, nil
}

// linfNorm returns the L∞ norm (peak gain regardless of stability) and the
// frequency at which it occurs. Poles on the stability boundary give +Inf at
// the frequency of that pole.
func linfNorm(sys *System) (norm float64, omega float64, err error) {
	if err := newDescriptorPolicy(sys).requireStandard("Norm"); err != nil {
		return 0, 0, err
	}
	if sys.IsContinuous() && sys.HasInternalDelay() {
		norm, omega, err := linfNormDelayed(sys)
		if err != nil {
			return 0, 0, fmt.Errorf("Norm: %w", err)
		}
		return norm, omega, nil
	}
	sys, err = finiteDimensionalModel(sys, "Norm")
	if err != nil {
		return 0, 0, err
	}
	n, m, p := sys.Dims()
	if n == 0 || m == 0 || p == 0 {
		sv, err := maxSingularValue(sys.D)
		if err != nil {
			return 0, 0, fmt.Errorf("Norm: %w", err)
		}
		return sv, 0, nil
	}
	poles, err := sys.Poles()
	if err != nil {
		return 0, 0, fmt.Errorf("Norm: %w", err)
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
	norm, omega, err = peakGain(sys, poles)
	if err != nil {
		return 0, 0, fmt.Errorf("Norm: %w", err)
	}
	return norm, omega, nil
}

// peakGain computes sup_ω σ_max(G(jω)) (G(e^{jωT}) for discrete models) of a
// model with no poles on the stability boundary from the γ-crossings of its
// extended pencil; stability is not required. A sampled peak, or σ_max(D)
// at ω = ∞ for continuous models, is first certified by one probe just
// above it; when the probe is inconclusive, the upper-bound search may still
// certify a raised peak, and bisection runs only if it does not. A
// continuous peak approached only as ω → ∞ returns omega = +Inf.
func peakGain(sys *System, poles []complex128) (norm float64, omega float64, err error) {
	n, m, p := sys.Dims()
	ws := newHamiltonianWS(sys, n, m, p)

	gammaLow, omegaPeak := ws.lowerBound(poles)
	if sys.IsContinuous() {
		sd, err := maxSingularValue(sys.D)
		if err != nil {
			return 0, 0, err
		}
		if sd > gammaLow {
			gammaLow, omegaPeak = sd, math.Inf(1)
		}
	}

	const tol = 1e-10
	gammaLow, omegaPeak, certified := ws.certifyPeak(gammaLow, omegaPeak, tol)
	if certified {
		return gammaLow * (1 + tol/2), omegaPeak, nil
	}

	gammaLow, gammaHigh, omegaPeak, certified := ws.upperBound(gammaLow, omegaPeak, tol)
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
		peak, w, err := ws.candidatePeak()
		if err != nil {
			gammaLow = mid
			continue
		}
		if peak < mid {
			gammaHigh = mid
			continue
		}
		gammaLow, omegaPeak, certified = ws.certifyPeak(peak, w, tol)
		if certified {
			return gammaLow * (1 + tol/2), omegaPeak, nil
		}
	}

	return gammaHigh, omegaPeak, nil
}

// certifyPeak tries to prove that the attained gain gammaLow is within tol
// of the peak: no γ-crossing at gammaLow·(1+tol/2) makes that level an upper
// bound. Crossings found there raise gammaLow through their interval
// midpoints (Boyd–Balakrishnan, Bruinsma–Steinbuch), and climb carries each
// raise to the local maximum, so the next probe usually certifies it.
// Candidates whose evaluated gains stay below the probe count as no
// crossing, the rule the bisection applies at every level. It reports false
// when the candidates cannot be evaluated, leaving the bisection to settle
// the peak.
func (ws *hamiltonianWS) certifyPeak(gammaLow, omegaPeak, tol float64) (float64, float64, bool) {
	for range 20 {
		if !(gammaLow > 0) || math.IsInf(gammaLow, 1) {
			return gammaLow, omegaPeak, false
		}
		probe := gammaLow * (1 + tol/2)
		if !ws.hasImagEigs(probe) {
			return gammaLow, omegaPeak, true
		}
		peak, w, err := ws.candidatePeak()
		if err != nil {
			return gammaLow, omegaPeak, false
		}
		if peak < probe {
			return gammaLow, omegaPeak, true
		}
		gammaLow, omegaPeak = ws.climb(peak, w)
	}
	return gammaLow, omegaPeak, false
}

// upperBound doubles gammaHigh from 2·gammaLow until the crossing test
// shows no crossing there, under the candidate rule of certifyPeak and the
// bisection: candidates whose evaluated gains stay below gammaHigh do not
// count. The pencil eigenvalues of a lightly damped mode stay inside the
// near-axis band at every level, so without that rule they would be flagged
// at every level. A candidate gain at or above gammaHigh raises gammaLow
// through certifyPeak, which may settle the peak outright.
func (ws *hamiltonianWS) upperBound(gammaLow, omegaPeak, tol float64) (float64, float64, float64, bool) {
	gammaHigh := math.Max(gammaLow*2, 1e-10)
	for range 50 {
		if !ws.hasImagEigs(gammaHigh) {
			break
		}
		peak, w, err := ws.candidatePeak()
		if err == nil && peak < gammaHigh {
			break
		}
		if err == nil {
			var certified bool
			gammaLow, omegaPeak, certified = ws.certifyPeak(peak, w, tol)
			if certified {
				return gammaLow, gammaLow * (1 + tol/2), omegaPeak, true
			}
		}
		gammaHigh = 2 * math.Max(gammaHigh, gammaLow)
	}
	return gammaLow, gammaHigh, omegaPeak, false
}

// hamiltonianWS holds the extended pencil of the γ-crossing test, reused
// across levels, and the evaluator that confirms its candidates.
//
// σ_i(G(jω)) = γ exactly when jω is a finite eigenvalue of the pencil
//
//	λ·[I 0 0 0; 0 I 0 0; 0 0 0 0; 0 0 0 0] − [A 0 B 0; 0 −Aᵀ 0 −Cᵀ; C 0 D −γI; 0 Bᵀ −γI Dᵀ]
//
// in (x, z, u, y); for discrete models e^{jωT} is an eigenvalue of
//
//	z·[I 0 0 0; 0 Aᵀ 0 Cᵀ; 0 0 0 0; 0 0 0 0] − [A 0 B 0; 0 I 0 0; C 0 D −γI; 0 Bᵀ −γI Dᵀ].
//
// An orthogonal transformation from the right compresses the algebraic rows,
// leaving a 2n×2n pencil for QZ (as SLICOT AB13DD). Unlike the Hamiltonian
// matrix, the pencil never inverts R = γ²I − DᵀD, so it keeps its accuracy
// as γ approaches σ_max(D); it also never forms BBᵀ/γ² and CᵀC, whose
// imbalance on near-optimal closed loops pushes crossings off the axis.
// Discrete models are tested on the unit circle directly, without a Tustin
// map whose rounding would shift lightly damped peaks.
type hamiltonianWS struct {
	sys     *System
	n, m, p int
	nn, nc  int
	k       int

	top, topE            []float64 // nn×nc
	botInit, bot         []float64 // nc×k: the algebraic rows, transposed
	tau                  []float64
	pae                  []float64 // 2nn×nc: top over topE
	a2, e2               []float64 // nn×nn
	alphar, alphai, beta []float64
	work                 []float64

	cands []float64
	freqs []float64
	eval  *sigmaEvaluator
}

func newHamiltonianWS(sys *System, n, m, p int) *hamiltonianWS {
	nn, k := 2*n, m+p
	nc := nn + k
	ws := &hamiltonianWS{
		sys: sys, n: n, m: m, p: p, nn: nn, nc: nc, k: k,
		top: make([]float64, nn*nc), topE: make([]float64, nn*nc),
		botInit: make([]float64, nc*k), bot: make([]float64, nc*k),
		tau: make([]float64, k),
		pae: make([]float64, 2*nn*nc),
		a2:  make([]float64, nn*nn), e2: make([]float64, nn*nn),
		alphar: make([]float64, nn), alphai: make([]float64, nn), beta: make([]float64, nn),
		eval: newSigmaEvaluator(sys, n, m, p),
	}
	a, b, c := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	top, topE, bot := ws.top, ws.topE, ws.botInit
	continuous := sys.IsContinuous()
	for i := range n {
		copy(top[i*nc:i*nc+n], a.Data[i*a.Stride:i*a.Stride+n])
		copy(top[i*nc+nn:i*nc+nn+m], b.Data[i*b.Stride:i*b.Stride+m])
		topE[i*nc+i] = 1
		row := (n + i) * nc
		if continuous {
			topE[row+n+i] = 1
		} else {
			top[row+n+i] = 1
		}
		sign, dual := -1.0, top
		if !continuous {
			sign, dual = 1, topE
		}
		for j := range n {
			dual[row+n+j] = sign * a.Data[j*a.Stride+i]
		}
		for j := range p {
			dual[row+nn+m+j] = sign * c.Data[j*c.Stride+i]
		}
		for j := range m {
			bot[(n+i)*k+p+j] = b.Data[i*b.Stride+j]
		}
	}
	for q := range p {
		for j := range n {
			bot[j*k+q] = c.Data[q*c.Stride+j]
		}
	}
	if sys.D != nil {
		d := sys.D.RawMatrix()
		for q := range p {
			for j := range m {
				bot[(nn+j)*k+q] = d.Data[q*d.Stride+j]
				bot[(nn+m+q)*k+p+j] = d.Data[q*d.Stride+j]
			}
		}
	}

	ws.work = make([]float64, 8*nn)
	return ws
}

// hamiltonianEvals counts hasImagEigs calls, one QZ decomposition each.
var hamiltonianEvals atomic.Int64

// nearAxisTol is the relative distance |Re λ|/|λ| (|ln|z||/|ln z| for
// discrete models) within which a pencil eigenvalue is taken as a crossing
// candidate. A near-optimal H∞ closed loop has σ_max flat to 1e-8 over
// decades, and unstructured QZ moves its crossings off the axis: by 1e-4
// relative on the TI6YYN loop and by 1e-3 to 1e-2 on a 50th-order
// mixed-sensitivity loop. The band is wide because it costs only σ_max
// evaluations: candidates steer the sampling, and every decision rests on an
// evaluated σ_max.
const nearAxisTol = 0.1

// hasImagEigs reports whether the pencil at level gamma has eigenvalues on or
// near the imaginary axis (unit circle) and records their frequencies as
// candidates for candidatePeak. A failed QZ counts as a crossing with no
// candidates.
func (ws *hamiltonianWS) hasImagEigs(gamma float64) bool {
	hamiltonianEvals.Add(1)
	nn, nc, k, m, p := ws.nn, ws.nc, ws.k, ws.m, ws.p
	ws.cands = ws.cands[:0]

	copy(ws.bot, ws.botInit)
	for q := range p {
		ws.bot[(nn+m+q)*k+q] = -gamma
	}
	for j := range m {
		ws.bot[(nn+j)*k+p+j] = -gamma
	}
	// Minimal workspaces: gonum's noasm GemvT clears all of y, so Dlarf
	// clears the whole work slice on every reflector (gonum ergo 2S5Y7S).
	impl.Dgeqrf(nc, k, ws.bot, k, ws.tau, ws.work[:k], k)
	copy(ws.pae, ws.top)
	copy(ws.pae[nn*nc:], ws.topE)
	impl.Dormqr(blas.Right, blas.NoTrans, 2*nn, nc, k, ws.bot, k, ws.tau, ws.pae, nc, ws.work[:2*nn], 2*nn)
	copyStrided(ws.a2, nn, ws.pae[k:], nc, nn, nn)
	copyStrided(ws.e2, nn, ws.pae[nn*nc+k:], nc, nn, nn)

	if !impl.Dggev(lapack.LeftEVNone, lapack.RightEVNone, nn, ws.a2, nn, ws.e2, nn,
		ws.alphar, ws.alphai, ws.beta, nil, 1, nil, 1, ws.work, len(ws.work)) {
		return true
	}

	dt := ws.sys.Dt
	for i := range nn {
		lam := complex(ws.alphar[i], ws.alphai[i]) / complex(ws.beta[i], 0)
		if ws.sys.IsDiscrete() {
			lam = cmplx.Log(lam)
		}
		if cmplx.IsInf(lam) || cmplx.IsNaN(lam) || imag(lam) < 0 {
			continue
		}
		if math.Abs(real(lam)) <= nearAxisTol*cmplx.Abs(lam) {
			w := imag(lam)
			if ws.sys.IsDiscrete() {
				w /= dt
			}
			ws.cands = append(ws.cands, w)
		}
	}
	return len(ws.cands) > 0
}

// candidatePeak evaluates σ_max at the candidate crossings found by
// hasImagEigs, at ω = 0 (and π/T for discrete models), and at the midpoints
// of the intervals they bound (Bruinsma and Steinbuch). Candidates only
// propose frequencies: a crossing counts when an evaluated σ_max reaches the
// level.
func (ws *hamiltonianWS) candidatePeak() (peak, omega float64, err error) {
	if len(ws.cands) == 0 {
		return 0, 0, ErrSchurFailed
	}
	slices.Sort(ws.cands)
	freqs := append(ws.freqs[:0], 0)
	freqs = append(freqs, ws.cands...)
	if ws.sys.IsDiscrete() {
		freqs = append(freqs, math.Pi/ws.sys.Dt)
	}
	freqs = slices.Compact(freqs)
	for i := range len(freqs) - 1 {
		freqs = append(freqs, (freqs[i]+freqs[i+1])/2)
	}
	ws.freqs = freqs
	return ws.eval.peak(freqs, 1e-6)
}

// climb maximises σ_max by Brent's method between the candidate points of
// candidatePeak that bracket w, where σ_max(w) = peak. One crossing test
// costs as much as tens of evaluations, and landing on the local maximum
// usually lets the next probe certify it. It returns the larger of peak and
// the refined gain at the maximiser.
func (ws *hamiltonianWS) climb(peak, w float64) (float64, float64) {
	a, b := math.Inf(-1), math.Inf(1)
	for _, f := range ws.freqs {
		if f < w {
			a = math.Max(a, f)
		} else if f > w {
			b = math.Min(b, f)
		}
	}
	if math.IsInf(a, 0) || math.IsInf(b, 0) {
		return peak, w
	}
	f := func(x float64) float64 {
		sv, err := ws.eval.sigma(x, false)
		if err != nil || math.IsNaN(sv) {
			return math.Inf(1)
		}
		return -sv
	}
	const cgold = 0.3819660112501051
	xtol := 1e-7 * (b - a)
	x, v, u := w, w, w
	fx := f(x)
	fv, fu := fx, fx
	wv, fw := w, fx
	var d, e float64
	for range 40 {
		m := (a + b) / 2
		tol := xtol + 4*eps()*math.Abs(x)
		if math.Abs(x-m) <= 2*tol-(b-a)/2 {
			break
		}
		parabolic := false
		if math.Abs(e) > tol {
			r := (x - wv) * (fx - fv)
			q := (x - v) * (fx - fw)
			pp := (x-v)*q - (x-wv)*r
			q = 2 * (q - r)
			if q > 0 {
				pp = -pp
			}
			q = math.Abs(q)
			if math.Abs(pp) < math.Abs(q*e/2) && pp > q*(a-x) && pp < q*(b-x) {
				e, d = d, pp/q
				parabolic = true
				if u = x + d; u-a < 2*tol || b-u < 2*tol {
					d = math.Copysign(tol, m-x)
				}
			}
		}
		if !parabolic {
			if x >= m {
				e = a - x
			} else {
				e = b - x
			}
			d = cgold * e
		}
		if math.Abs(d) >= tol {
			u = x + d
		} else {
			u = x + math.Copysign(tol, d)
		}
		fu = f(u)
		if fu <= fx {
			if u >= x {
				a = x
			} else {
				b = x
			}
			v, fv, wv, fw = wv, fw, x, fx
			x, fx = u, fu
			continue
		}
		if u < x {
			a = u
		} else {
			b = u
		}
		if fu <= fw || wv == x {
			v, fv, wv, fw = wv, fw, u, fu
		} else if fu <= fv || v == x || v == wv {
			v, fv = u, fu
		}
	}
	if sv, err := ws.eval.sigma(x, true); err == nil && sv > peak {
		return sv, x
	}
	return peak, w
}

// lowerBound samples σ_max at ω = 0 and 50 log-spaced frequencies spanning
// the magnitudes of poles; discrete models use the magnitudes of their Tustin
// equivalents, warped back to the unit circle, and add the Nyquist frequency.
// It is only a seed: when a sample hits a singular resolvent it returns the
// trivial bound 0, and the Hamiltonian search still finds the peak.
func (ws *hamiltonianWS) lowerBound(poles []complex128) (gammaLow, omegaPeak float64) {
	sys := ws.sys
	dt := sys.Dt
	wmin, wmax := math.Inf(1), 0.0
	for _, pole := range poles {
		if sys.IsDiscrete() {
			pole = complex(2/dt, 0) * (pole - 1) / (pole + 1)
		}
		w := cmplx.Abs(pole)
		if w > 0 && !math.IsInf(w, 0) && !math.IsNaN(w) {
			wmin = math.Min(wmin, w)
			wmax = math.Max(wmax, w)
		}
	}
	if wmax == 0 {
		wmin, wmax = 0.01, 100
	}
	wmin /= 10
	wmax *= 10

	freqs := append(ws.freqs[:0], 0)
	for i := range 50 {
		w := wmin * math.Pow(wmax/wmin, float64(i)/49)
		if sys.IsDiscrete() {
			w = 2 * math.Atan(w*dt/2) / dt
		}
		freqs = append(freqs, w)
	}
	if sys.IsDiscrete() {
		freqs = append(freqs, math.Pi/dt)
	}
	ws.freqs = freqs
	if peak, w, err := ws.eval.peak(freqs, 0); err == nil {
		gammaLow, omegaPeak = peak, w
	}
	return gammaLow, omegaPeak
}

// sigmaEvaluator computes σ_max(G) at single frequencies to near working
// precision. GEPP alone loses accuracy in proportion to the condition number
// of the resolvent, about 1/(ζω₀) next to a lightly damped pole; two
// refinement steps with the residual, and C·X + D, accumulated in
// compensated (FMA) arithmetic recover it. Each frequency is solved at
// timeDomain.frequencyPoint as (sI − A)⁻¹ = −q·(pI + qA)⁻¹, exactly on the
// unit circle for discrete models.
type sigmaEvaluator struct {
	sys     *System
	n, m, p int
	lu      []complex128
	piv     []int
	x, dx   []complex128 // n×m
	g       []complex128 // p×m
	vals    []float64
	svd     *complexSVDWorkspace
}

func newSigmaEvaluator(sys *System, n, m, p int) *sigmaEvaluator {
	e := &sigmaEvaluator{
		sys: sys, n: n, m: m, p: p,
		lu: make([]complex128, n*n), piv: make([]int, n),
		x: make([]complex128, n*m), dx: make([]complex128, n*m),
		g: make([]complex128, p*m),
	}
	if p > 2 || m > 2 {
		e.svd = newComplexSVDWorkspace(p, m)
	}
	return e
}

// peak returns the largest σ_max over freqs and where it occurs. Every
// frequency is solved by GEPP alone; those within the relative band of the
// largest are refined, so the result is an accurately evaluated gain.
func (e *sigmaEvaluator) peak(freqs []float64, band float64) (peak, omega float64, err error) {
	vals := e.vals[:0]
	top := math.Inf(-1)
	for _, w := range freqs {
		sv, err := e.sigma(w, false)
		if err != nil {
			return 0, 0, err
		}
		vals = append(vals, sv)
		if sv > top || math.IsNaN(sv) {
			top = sv
		}
	}
	e.vals = vals
	if math.IsNaN(top) {
		return 0, 0, ErrSingularEquation
	}
	peak = math.Inf(-1)
	for i, w := range freqs {
		if vals[i] < top*(1-band) {
			continue
		}
		sv, err := e.sigma(w, true)
		if err != nil {
			return 0, 0, err
		}
		if sv > peak || math.IsNaN(sv) {
			peak, omega = sv, w
		}
	}
	if math.IsNaN(peak) {
		return 0, 0, ErrSingularEquation
	}
	return peak, omega, nil
}

// sigma evaluates σ_max(G) at w, refined as the type comment describes or by
// GEPP alone.
func (e *sigmaEvaluator) sigma(w float64, refine bool) (float64, error) {
	n, m, p := e.n, e.m, e.p
	pt := newTimeDomain(e.sys.Dt).frequencyPoint(w)
	sp, sq, sc := pt.p, pt.q, pt.scale()
	if err := e.factor(sp, sq); err != nil {
		return 0, err
	}
	b := e.sys.B.RawMatrix()
	for i := range n {
		for j := range m {
			e.x[i*m+j] = complex(b.Data[i*b.Stride+j], 0)
		}
	}
	e.solve(e.x)
	if refine {
		for step := range 2 {
			e.residual(sp, sq)
			e.solve(e.dx)
			if step == 0 {
				for i, v := range e.dx {
					e.x[i] += v
				}
			}
		}
		e.output(sc)
	} else {
		e.outputPlain(sc)
	}
	applyIODelayAtS(e.sys, pt.value(), e.g, p, m, true)
	return e.svd.maximumFromFlat(e.g, 0, p, m)
}

// factor computes the GEPP factorization of pI + qA.
func (e *sigmaEvaluator) factor(sp, sq complex128) error {
	n, lu := e.n, e.lu
	a := e.sys.A.RawMatrix()
	for i := range n {
		for j := range n {
			lu[i*n+j] = sq * complex(a.Data[i*a.Stride+j], 0)
		}
		lu[i*n+i] += sp
	}
	if !cLUFactor(lu, e.piv, n) {
		return ErrSingularEquation
	}
	return nil
}

// solve overwrites the n×m right-hand side x with (pI + qA)⁻¹x.
func (e *sigmaEvaluator) solve(x []complex128) {
	cLUSolve(e.lu, e.piv, x, e.n, e.m)
}

// residual sets dx = B − (p·x + q·A·x) in compensated arithmetic.
func (e *sigmaEvaluator) residual(sp, sq complex128) {
	a, b := e.sys.A.RawMatrix(), e.sys.B.RawMatrix()
	for j := range e.m {
		pencilResidual(e.dx[j:], e.x[j:], e.m, a.Data, nil, a.Stride, b.Data[j:], b.Stride, e.n, sp, sq, 1)
	}
}

// output sets g = c·C·(x + dx) + D in compensated arithmetic.
func (e *sigmaEvaluator) output(sc complex128) {
	n, m, p := e.n, e.m, e.p
	c := e.sys.C.RawMatrix()
	cr, ci := real(sc), imag(sc)
	for i := range p {
		crow := c.Data[i*c.Stride : i*c.Stride+n]
		for j := range m {
			var ur, ui compensatedSum
			for k, cv := range crow {
				x, dx := e.x[k*m+j], e.dx[k*m+j]
				ur.addProd(cv, real(x))
				ur.addProd(cv, real(dx))
				ui.addProd(cv, imag(x))
				ui.addProd(cv, imag(dx))
			}
			var re, im compensatedSum
			if e.sys.D != nil {
				d := e.sys.D.RawMatrix()
				re.add(d.Data[i*d.Stride+j])
			}
			re.addScaled(cr, ur)
			re.addScaled(-ci, ui)
			im.addScaled(cr, ui)
			im.addScaled(ci, ur)
			e.g[i*m+j] = complex(re.value(), im.value())
		}
	}
}

// outputPlain sets g = c·C·x + D in working precision.
func (e *sigmaEvaluator) outputPlain(sc complex128) {
	n, m, p := e.n, e.m, e.p
	c := e.sys.C.RawMatrix()
	for i := range p {
		crow := c.Data[i*c.Stride : i*c.Stride+n]
		for j := range m {
			var re, im float64
			for k, cv := range crow {
				x := e.x[k*m+j]
				re += cv * real(x)
				im += cv * imag(x)
			}
			g := sc * complex(re, im)
			if e.sys.D != nil {
				d := e.sys.D.RawMatrix()
				g += complex(d.Data[i*d.Stride+j], 0)
			}
			e.g[i*m+j] = g
		}
	}
}

// compensatedSum accumulates sums and products with error-free
// transformations (Ogita, Rump and Oishi, Dot2): the result is as accurate
// as if computed in twice the working precision.
type compensatedSum struct{ s, c float64 }

func (a *compensatedSum) add(v float64) {
	t := a.s + v
	bp := t - a.s
	a.c += (a.s - (t - bp)) + (v - bp)
	a.s = t
}

func (a *compensatedSum) addProd(x, y float64) {
	// The conversion forbids fusing x*y into the sum, which would make the
	// FMA residual below meaningless.
	h := float64(x * y)
	a.add(h)
	a.c += math.FMA(x, y, -h)
}

// addScaled adds x times the unrounded value of b.
func (a *compensatedSum) addScaled(x float64, b compensatedSum) {
	a.addProd(x, b.s)
	a.addProd(x, b.c)
}

func (a *compensatedSum) value() float64 { return a.s + a.c }

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

// maxSingularValue returns σ_max(M), 0 for a nil or empty matrix.
func maxSingularValue(M *mat.Dense) (float64, error) {
	if M == nil || M.IsEmpty() {
		return 0, nil
	}
	r, c := M.Dims()
	raw := M.RawMatrix()
	data := make([]float64, r*c)
	copyStrided(data, c, raw.Data, raw.Stride, r, c)
	sv := make([]float64, min(r, c))
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, r, c, data, c, sv, nil, 1, nil, 1, wq, -1)
	work := make([]float64, int(wq[0]))
	if !impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, r, c, data, c, sv, nil, 1, nil, 1, work, len(work)) {
		return 0, fmt.Errorf("singular value iteration did not converge: %w", ErrSchurFailed)
	}
	return sv[0], nil
}
