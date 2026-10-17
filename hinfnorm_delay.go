package controlsys

import (
	"fmt"
	"math"
	"math/bits"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/mat"
)

// errDelayPeakUnsupported reports a peak gain the delay path cannot certify.
var errDelayPeakUnsupported = fmt.Errorf("peak gain outside the certified delay scope: %w", ErrDelayUnsupported)

// delayLFT evaluates a continuous model with internal delays,
//
//	G(s) = H11 + H12·Δ(I − H22·Δ)⁻¹·H21,  Hij = Dij + Ci(sI−A)⁻¹Bj,  Δ = diag(e^{−sτ}),
//
// whose characteristic function χ(s) = det(sI−A)·det(I − H22(s)Δ(s)) is
// entire, so its closed-loop roots are the zeros of χ.
type delayLFT struct {
	sys     *System
	n, m, p int
	nd      int
	tau     []float64
	mult    []int // see multiples, computed on demand
	multOK  bool
	td      timeDomain

	h22sys *System
	h22    func(frequencyPoint, []complex128) error
	hbuf   []complex128
	lu     []complex128
	piv    []int

	g       func(frequencyPoint, []complex128) error
	gbuf    []complex128
	ioDelay *mat.Dense
	svd     *complexSVDWorkspace

	poles []complex128
	normA float64
	// spectral norms of C, C2, B, B2 and the D blocks
	c1, c2, b1, b2     float64
	d11, d12, d21, d22 float64
	// entrywise |D12|, |D21|, |D22|, the row norms of C, C2, the column
	// norms of B, B2, and Q0 = (I − |D22|)⁻¹ with r0 = ‖Q0‖ (nil, +Inf when
	// ρ(|D22|) >= 1)
	abs12, abs21, abs22 []float64
	rowC1, rowC2        []float64
	colB1, colB2        []float64
	q0                  []float64
	r0                  float64
	entrywise           bool // norm bounds cannot certify the w = ∞ tail
	directFeedthrough   bool
	desc                *descriptorResponse // for certified peaks
	diff                *differenceBounds   // see difference, computed on demand
	d22c                []complex128
	err                 error
	sigmaCalls          int
	gInfCalls           int       // samples of σ_max(G∞), see highFrequencyGain.at
	scanW, scanV        []float64 // scan buffers
}

func newDelayLFT(sys *System) (*delayLFT, error) {
	n, m, p := sys.Dims()
	lft := sys.LFT
	nd := len(lft.Tau)
	e := &delayLFT{
		sys: sys, n: n, m: m, p: p, nd: nd,
		tau:  lft.Tau,
		td:   newTimeDomain(0),
		hbuf: make([]complex128, nd*nd),
		lu:   make([]complex128, nd*nd),
		piv:  make([]int, nd),
		gbuf: make([]complex128, p*m),
		g:    newFrequencyEvaluator(sys).pointEval(),
	}
	e.ioDelay = effectiveIODelayMatrix(sys, p, m, true)
	var sn spectralNormer
	if p > 2 || m > 2 {
		e.svd = newComplexSVDWorkspace(p, m)
	}
	if n > 0 {
		h22sys := &System{A: sys.A, B: lft.B2, C: lft.C2, D: lft.D22}
		if err := h22sys.Validate(); err != nil {
			return nil, err
		}
		e.h22sys = h22sys
		e.h22 = newFrequencyEvaluator(h22sys).pointSolver(math.MaxInt).evalInto
		poles, err := h22sys.Poles()
		if err != nil {
			return nil, err
		}
		e.poles = poles
		e.normA = sn.norm(sys.A)
		e.c1, e.c2 = sn.norm(sys.C), sn.norm(lft.C2)
		e.b1, e.b2 = sn.norm(sys.B), sn.norm(lft.B2)
	}
	e.d11, e.d12 = sn.norm(sys.D), sn.norm(lft.D12)
	e.d21, e.d22 = sn.norm(lft.D21), sn.norm(lft.D22)
	e.r0 = math.Inf(1)
	e.entrywise = nd > 0 && (e.d22 >= 1 || math.Pow(1+e.d22, float64(nd))-1 >= 1)
	if e.entrywise {
		e.abs12, e.abs21, e.abs22 = absEntries(lft.D12, p, nd), absEntries(lft.D21, nd, m), absEntries(lft.D22, nd, nd)
		e.rowC1, e.rowC2 = rowNorms(sys.C, p), rowNorms(lft.C2, nd)
		e.colB1, e.colB2 = colNorms(sys.B, m), colNorms(lft.B2, nd)
		if q0, ok := nonnegResolvent(e.abs22, nd); ok {
			e.q0, e.r0 = q0, sn.norm(mat.NewDense(nd, nd, q0))
		}
	}
	e.directFeedthrough = lftHasDirectFeedthrough(lft)
	return e, nil
}

// spectralNorm returns an upper bound on σ_max(m): the computed largest
// singular value raised by its rounding error, capped by the Frobenius norm.
func spectralNorm(m *mat.Dense) float64 {
	var s spectralNormer
	return s.norm(m)
}

// spectralNormer evaluates spectralNorm reusing one SVD buffer across calls.
type spectralNormer struct{ buf []float64 }

func (s *spectralNormer) norm(m *mat.Dense) float64 {
	if m == nil || m.IsEmpty() {
		return 0
	}
	frob := mat.Norm(m, 2)
	r, c := m.Dims()
	if r == 1 || c == 1 || frob == 0 || math.IsInf(frob, 0) || math.IsNaN(frob) {
		return frob
	}
	sv, ok, buf := svdMaxInto(m, s.buf)
	s.buf = buf
	if !ok {
		return frob
	}
	return min(frob, sv*(1+8*float64(max(r, c))*0x1p-52))
}

func (e *delayLFT) fail(w float64, err error) {
	if e.err == nil {
		e.err = fmt.Errorf("response at ω=%g: %w", w, err)
	}
}

// charMinusOne returns det(I − H22(jω)Δ(jω)) − 1, the loop L of the Nyquist
// test whose 1+L has the zeros of χ and the poles of eig(A) it does not
// cancel.
func (e *delayLFT) charMinusOne(w float64) complex128 {
	det, ok := e.h22Det(w)
	if !ok {
		return complex(math.NaN(), math.NaN())
	}
	return det - 1
}

// h22Det returns det(I − H22(jω)Δ(jω)).
func (e *delayLFT) h22Det(w float64) (complex128, bool) {
	pt := e.td.frequencyPoint(w)
	if e.h22 != nil {
		if err := evalWithPoleLimit(e.h22, e.h22sys, pt, e.hbuf); err != nil {
			e.fail(w, err)
			return 0, false
		}
	} else {
		copy(e.hbuf, e.d22Complex())
	}
	return e.loopDet(pt.value(), e.hbuf), true
}

// multiples returns commensurateMultiples of the delays, computed once.
func (e *delayLFT) multiples() []int {
	if !e.multOK {
		e.mult, e.multOK = commensurateMultiples(e.tau), true
	}
	return e.mult
}

func (e *delayLFT) d22Complex() []complex128 {
	if e.d22c == nil {
		nd := e.nd
		e.d22c = make([]complex128, nd*nd)
		if d22 := e.sys.LFT.D22; d22 != nil {
			for i := range nd {
				for j := range nd {
					e.d22c[i*nd+j] = complex(d22.At(i, j), 0)
				}
			}
		}
	}
	return e.d22c
}

// loopDet returns det(I − h·Δ(s)), 0 when it is singular.
func (e *delayLFT) loopDet(s complex128, h []complex128) complex128 {
	nd := e.nd
	if nd == 1 {
		return 1 + -h[0]*cmplx.Exp(-s*complex(e.tau[0], 0))
	}
	for j := range nd {
		dj := cmplx.Exp(-s * complex(e.tau[j], 0))
		for i := range nd {
			e.lu[i*nd+j] = -h[i*nd+j] * dj
		}
	}
	for i := range nd {
		e.lu[i*nd+i]++
	}
	if !cLUFactor(e.lu, e.piv, nd) {
		return 0
	}
	det := complex(1, 0)
	for k := range nd {
		det *= e.lu[k*nd+k]
		if e.piv[k] != k {
			det = -det
		}
	}
	return det
}

// normalizedMinusOne returns det(I − H22Δ)/det(I − D22Δ) − 1 at jω, the loop
// of the Nyquist test once the difference operator I − D22Δ is certified
// nonsingular on the closed right half-plane: χ/det(I − D22Δ) then has the
// zeros of χ there and the same poles.
func (e *delayLFT) normalizedMinusOne(w float64) complex128 {
	num, ok := e.h22Det(w)
	if !ok {
		return complex(math.NaN(), math.NaN())
	}
	den := e.loopDet(complex(0, w), e.d22Complex())
	if den == 0 {
		return complex(math.NaN(), math.NaN())
	}
	return num/den - 1
}

// normalizedTail returns the tail bound of normalizedMinusOne for
// ‖(I − D22Δ)⁻¹‖ <= r: det(I − H22Δ)/det(I − D22Δ) = det(I − (I − D22Δ)⁻¹EΔ)
// with E = C2(sI−A)⁻¹B2 of rank at most min(n, nd), so its distance from 1 is
// at most (1 + r‖E‖)^min(n,nd) − 1.
func (e *delayLFT) normalizedTail(r float64) func(float64) float64 {
	k := float64(min(e.n, e.nd))
	return func(w float64) float64 {
		return math.Pow(1+r*e.h(0, e.c2, e.b2, w), k) - 1
	}
}

// tail bounds |det(I − H22Δ) − 1| for Re s ≥ 0, |s| ≥ w by the smaller of
// (1 + ‖H22‖)^nd − 1, since |det(I+M) − 1| ≤ ∏(1+σᵢ(M)) − 1 and
// ‖H22(s)‖ ≤ ‖D22‖ + ‖C2‖‖B2‖/(|s| − ‖A‖), and per(I + P) − 1 for the
// entrywise bound |H22(s)Δ(s)| ≤ P, since every monomial of det(I − M) − 1
// appears in per(I + P) − 1 with coefficient +1. The permanent bound vanishes
// at w = ∞ when D22 is nilpotent in pattern, a cascade of delays with no
// algebraic delay cycle.
func (e *delayLFT) tail(w float64) float64 {
	h := e.h(e.d22, e.c2, e.b2, w)
	bound := math.Pow(1+h, float64(e.nd)) - 1
	if g := e.resolvent(w); e.entrywise && e.nd <= maxPermanentDelays && !math.IsInf(g, 1) {
		p22 := entryBound(e.abs22, e.rowC2, e.colB2, e.nd, e.nd, g)
		bound = min(bound, permanentOfIPlus(p22, e.nd)-1)
	}
	return bound
}

// maxPermanentDelays caps the O(2^nd·nd) permanent in tail.
const maxPermanentDelays = 12

// resolvent bounds ‖(sI−A)⁻¹‖ for |s| ≥ w; 0 without states or at w = ∞.
func (e *delayLFT) resolvent(w float64) float64 {
	switch {
	case e.n == 0 || math.IsInf(w, 1):
		return 0
	case w <= e.normA:
		return math.Inf(1)
	}
	return 1 / (w - e.normA)
}

// entryBound returns |D| + rows·colsᵀ·g, an entrywise bound on
// D + C(sI−A)⁻¹B when ‖(sI−A)⁻¹‖ ≤ g, from the row norms of C and the column
// norms of B.
func entryBound(abs, rows, cols []float64, r, c int, g float64) []float64 {
	out := slices.Clone(abs)
	for i := range r {
		for j := range c {
			if cb := rows[i] * cols[j]; cb > 0 {
				out[i*c+j] += cb * g
			}
		}
	}
	return out
}

// permanentOfIPlus returns per(I + P) for a nonnegative n×n P by dynamic
// programming over column subsets, free of cancellation.
func permanentOfIPlus(p []float64, n int) float64 {
	f := make([]float64, 1<<n)
	f[0] = 1
	for mask := 1; mask < len(f); mask++ {
		i := bits.OnesCount(uint(mask)) - 1
		sum := 0.0
		for j := range n {
			if mask&(1<<j) == 0 {
				continue
			}
			v := p[i*n+j]
			if i == j {
				v++
			}
			if v != 0 {
				sum += f[mask&^(1<<j)] * v
			}
		}
		f[mask] = sum
	}
	return f[len(f)-1]
}

// nonnegResolvent returns (I − P)⁻¹ for a nonnegative n×n P with
// ρ(P) < 1, which is then entrywise nonnegative and bounds |(I − M)⁻¹|
// entrywise for every |M| ≤ P by the Neumann series; ok is false otherwise.
func nonnegResolvent(p []float64, n int) ([]float64, bool) {
	a := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			v := -p[i*n+j]
			if i == j {
				v++
			}
			if math.IsInf(v, 0) || math.IsNaN(v) {
				return nil, false
			}
			a.Set(i, j, v)
		}
	}
	var eig mat.Eigen
	if !eig.Factorize(mat.NewDense(n, n, slices.Clone(p)), mat.EigenNone) {
		return nil, false
	}
	for _, v := range eig.Values(nil) {
		if !(cmplx.Abs(v) < 1) {
			return nil, false
		}
	}
	var inv mat.Dense
	if err := inv.Inverse(a); err != nil {
		return nil, false
	}
	q := inv.RawMatrix().Data
	scale := 0.0
	for _, v := range q {
		if math.IsInf(v, 0) || math.IsNaN(v) {
			return nil, false
		}
		scale = max(scale, math.Abs(v))
	}
	for i, v := range q {
		if v < -1e-12*scale {
			return nil, false
		}
		q[i] = max(v, 0)
	}
	return q, true
}

func absEntries(m *mat.Dense, r, c int) []float64 {
	out := make([]float64, r*c)
	if m == nil || m.IsEmpty() {
		return out
	}
	for i := range r {
		for j := range c {
			out[i*c+j] = math.Abs(m.At(i, j))
		}
	}
	return out
}

func rowNorms(m *mat.Dense, r int) []float64 {
	out := make([]float64, r)
	if m == nil || m.IsEmpty() {
		return out
	}
	for i := range r {
		out[i] = rowNorm(m, i)
	}
	return out
}

func colNorms(m *mat.Dense, c int) []float64 {
	out := make([]float64, c)
	if m == nil || m.IsEmpty() {
		return out
	}
	for j := range c {
		out[j] = colNorm(m, j)
	}
	return out
}

// unstableDifferenceOperator reports whether f(s) = det(I − D22·Δ(s)) has a
// zero s0 with Re s0 > 0: for commensurate delays τ_i = k_i·h > 0, an
// eigenvalue μ of liftD22 with |μ| > 1, so f vanishes on Re s = ln|μ|/h;
// otherwise, when the loop is neutral and its difference operator is not
// certified stable, a zero certified by expPoly.rhpRoot.
// f(s + 2πjq) is uniformly close to f(s) near s0 for infinitely many integers
// q, by Dirichlet's simultaneous approximation of the qτ_k by integers, and
// H22 → D22 uniformly there, so χ has infinitely many roots in Re s > 0 by
// Rouché.
func (e *delayLFT) unstableDifferenceOperator() bool {
	d22 := e.sys.LFT.D22
	if e.nd == 0 || !(e.d22 > 1) || d22 == nil || d22.IsEmpty() {
		return false
	}
	mult := e.multiples()
	if mult == nil || !(e.tau[0] > 0) {
		if e.tail(math.Inf(1)) < 1 || !math.IsInf(e.difference().rhp, 1) || e.nd > maxPermanentDelays {
			return false
		}
		return newExpPoly(d22, e.tau).rhpRoot(d22, e.tau)
	}
	var eig mat.Eigen
	if !eig.Factorize(liftD22(d22, mult), mat.EigenNone) {
		return false
	}
	for _, v := range eig.Values(nil) {
		if cmplx.Abs(v) > 1+1e-9 {
			return true
		}
	}
	return false
}

// h bounds ‖D + C(sI−A)⁻¹B‖ for |s| ≥ w.
func (e *delayLFT) h(d, c, b, w float64) float64 {
	cb := c * b
	if cb == 0 || math.IsInf(w, 1) {
		return d
	}
	if w <= e.normA {
		return math.Inf(1)
	}
	return d + cb/(w-e.normA)
}

// loop returns the delay loop of the Nyquist test. A neutral loop whose tail
// bound does not fall below 1 is normalized by its difference operator when
// that is certified stable; with axisOnly the torus bound also serves, which
// certifies the tail on the imaginary axis only, for the grid of the L∞
// peak, and the root count is then void.
func (e *delayLFT) loop(axisOnly bool) *delayLoop {
	l := &delayLoop{at: e.charMinusOne, tail: e.tail, tailLimit: e.tail(math.Inf(1)), multiChannel: e.nd > 1}
	if !(l.tailLimit < 1) {
		d := e.difference()
		r := d.rhp
		if axisOnly {
			r = min(r, d.torus)
		}
		if !math.IsInf(r, 1) {
			l.at, l.tail, l.tailLimit = e.normalizedMinusOne, e.normalizedTail(r), 0
		}
	}
	for _, t := range e.tau {
		l.tau += t
	}
	l.addPoles(e.poles)
	return l
}

// sigma returns σ_max(G(jω)) with the exact delay factors.
func (e *delayLFT) sigma(w float64) float64 {
	e.sigmaCalls++
	pt := e.td.frequencyPoint(w)
	if err := evalWithPoleLimit(e.g, e.sys, pt, e.gbuf); err != nil {
		e.fail(w, err)
		return math.NaN()
	}
	if e.ioDelay != nil {
		applyIODelayMatrixAtS(e.sys, pt, e.gbuf, e.p, e.m, e.ioDelay)
	}
	sv, err := e.svd.maximumFromFlat(e.gbuf, 0, e.p, e.m)
	if err != nil {
		e.fail(w, err)
		return math.NaN()
	}
	return sv
}

// highFrequencyGain evaluates σ_max(G∞(φ)), G∞ = D11 + D12·Δ(I − D22·Δ)⁻¹·D21
// with Δ_i = e^{−j·k_i·φ_g(i)}, the values G(jω) approaches as ω → ∞. The
// phases φ range over a torus of dim coordinates, one φ_c = ωh_c per class
// of commensurate delays τ_i = k_i·h_c (see commensurateClasses). The values
// of Δ(jω) as ω → ∞ lie on this torus, and fill it when the h_c are
// rationally independent, which the exact grouping cannot always confirm:
// below then still certifies an upper bound, and estimate is taken as the
// limit only for dim = 1, a single class, where φ = ωh sweeps the circle.
// Coordinate c is scaled by K_c = max k_i over its class: G∞ moves at most
// as fast along each, so cells of half-width r/K_c share one Lipschitz bound.
type highFrequencyGain struct {
	e       *delayLFT
	dim     int
	grp     []int
	k       []float64 // k_i, nil when all ones
	kmax    float64
	scale   []float64 // K_c
	d11     []complex128
	d12     []complex128
	d21     []complex128
	d22     []complex128
	lu, x   []complex128
	phase   []complex128
	piv     []int
	g       []complex128
	lip     float64
	samples int
	// factors of localBound and quadBound at the last sample, see local
	xm               []complex128
	z                []complex128
	locA, locX, locM float64
	linG, linJ       float64
	jac, vert        []complex128 // matrix G∞: J_c of localSlopes, quadBound buffer
	over             float64      // largest sample that failed below

	grid  []float64 // torusGrid cell centres, nd per cell
	gridR float64
	gridV []float64 // σ_max(G∞) at the cell centres, see gridValues
}

func newHighFrequencyGain(e *delayLFT) *highFrequencyGain {
	grp, mult, dim := commensurateClasses(e.tau)
	return newHighFrequencyGainOn(e, grp, mult, dim)
}

// distinctDelays returns the gain on the torus of one phase per distinct
// delay with k_i = 1, which also holds every limit of Δ(jω) as ω → ∞ and may
// certify a bound with fewer samples when some K_c is large; nil when it is
// the same torus.
func (h *highFrequencyGain) distinctDelays() *highFrequencyGain {
	tau := h.e.tau
	grp, mult, dim := make([]int, len(tau)), make([]int, len(tau)), 0
	for i, t := range tau {
		mult[i] = 1
		if f := slices.Index(tau, t); f < i {
			grp[i] = grp[f]
		} else {
			grp[i] = dim
			dim++
		}
	}
	if dim == h.dim && h.k == nil {
		return nil
	}
	return newHighFrequencyGainOn(h.e, grp, mult, dim)
}

func newHighFrequencyGainOn(e *delayLFT, grp, mult []int, dim int) *highFrequencyGain {
	lft := e.sys.LFT
	nd, p, m := e.nd, e.p, e.m
	toC := func(d *mat.Dense, r, c int) []complex128 {
		out := make([]complex128, r*c)
		if d == nil {
			return out
		}
		for i := range r {
			for j := range c {
				out[i*c+j] = complex(d.At(i, j), 0)
			}
		}
		return out
	}
	var k []float64
	kmax := 1.0
	scale := make([]float64, dim)
	for c := range scale {
		scale[c] = 1
	}
	if nd > 0 && slices.Max(mult) > 1 {
		k = make([]float64, nd)
		for i, ki := range mult {
			k[i] = float64(ki)
			kmax = max(kmax, k[i])
			scale[grp[i]] = max(scale[grp[i]], k[i])
		}
	}
	h := &highFrequencyGain{
		e: e, dim: dim, grp: grp, k: k, kmax: kmax, scale: scale, over: math.Inf(-1),
		d11: toC(e.sys.D, p, m), d12: toC(lft.D12, p, nd),
		d21: toC(lft.D21, nd, m), d22: toC(lft.D22, nd, nd),
		lu: make([]complex128, nd*nd), x: make([]complex128, nd*m),
		phase: make([]complex128, nd), piv: make([]int, nd), g: make([]complex128, p*m),
	}
	h.lip = h.lipschitz()
	return h
}

// lipschitz bounds the Lipschitz constant of G∞ in φ, in the max-norm of
// K_c·φ_c, by ‖D12‖‖D21‖·r², with r bounding ‖(I − D22Δ)⁻¹‖ on the phase torus,
// which also bounds ‖(I − ΔD22)⁻¹‖ = ‖Δ(I − D22Δ)⁻¹Δ⁻¹‖: 1/(1 − ‖D22‖)
// when ‖D22‖ < 1, ‖(I − |D22|)⁻¹‖ when ρ(|D22|) < 1, else the polydisk or
// torus bound of difference, or for one phase its right half-plane bound,
// since Δ(φ) then is Δ(jφ/h).
func (h *highFrequencyGain) lipschitz() float64 {
	e := h.e
	c := e.d12 * e.d21
	if c == 0 {
		return 0
	}
	r := e.r0
	if e.d22 < 1 {
		r = min(r, 1/(1-e.d22))
	}
	if math.IsInf(r, 1) {
		d := e.difference()
		r = min(d.polydisk, d.torus)
		if h.dim == 1 {
			r = min(r, d.rhp)
		}
	}
	return c * r * r
}

func (h *highFrequencyGain) at(theta []float64) float64 {
	h.samples++
	h.e.gInfCalls++
	nd, p, m := h.e.nd, h.e.p, h.e.m
	for i := range nd {
		t := theta[h.grp[i]]
		if h.k != nil {
			t *= h.k[i]
		}
		h.phase[i] = cmplx.Exp(complex(0, -t))
	}
	for i := range nd {
		for j := range nd {
			h.lu[i*nd+j] = -h.d22[i*nd+j] * h.phase[j]
		}
		h.lu[i*nd+i]++
	}
	if !cLUFactor(h.lu, h.piv, nd) {
		return math.Inf(1)
	}
	copy(h.x, h.d21)
	cLUSolve(h.lu, h.piv, h.x, nd, m)
	copy(h.g, h.d11)
	for i := range p {
		for k := range nd {
			c := h.d12[i*nd+k] * h.phase[k]
			if c == 0 {
				continue
			}
			for j := range m {
				h.g[i*m+j] += c * h.x[k*m+j]
			}
		}
	}
	sv, err := h.e.svd.maximumFromFlat(h.g, 0, p, m)
	if err != nil {
		return math.NaN()
	}
	return sv
}

// local sets the factors of localBound and quadBound at the last sample of
// at, from its LU factors of I − D22Δ: it solves (I − D22Δ)·[X M] =
// [D21 I] for M = (I − D22Δ)⁻¹ and sets the Frobenius norms locA of
// D12·Δ·M, locX of X and locM of M, then the slopes of localSlopes.
func (h *highFrequencyGain) local() {
	nd, p, m := h.e.nd, h.e.p, h.e.m
	w := m + nd
	if h.xm == nil {
		h.xm = make([]complex128, nd*w)
		h.z = make([]complex128, p*nd)
	}
	xm := h.xm
	clear(xm)
	for i := range nd {
		copy(xm[i*w:i*w+m], h.d21[i*m:(i+1)*m])
		xm[i*w+m+i] = 1
	}
	cLUSolve(h.lu, h.piv, xm, nd, w)
	var sx, sm, sa float64
	for i := range nd {
		for _, v := range xm[i*w : i*w+m] {
			sx += real(v)*real(v) + imag(v)*imag(v)
		}
		for _, v := range xm[i*w+m : (i+1)*w] {
			sm += real(v)*real(v) + imag(v)*imag(v)
		}
	}
	for i := range p {
		for j := range nd {
			var z complex128
			for k := range nd {
				z += h.d12[i*nd+k] * h.phase[k] * xm[k*w+m+j]
			}
			h.z[i*nd+j] = z
			sa += real(z)*real(z) + imag(z)*imag(z)
		}
	}
	h.locA, h.locX, h.locM = math.Sqrt(sa), math.Sqrt(sx), math.Sqrt(sm)
	h.localSlopes()
}

// localSlopes sets the slopes of quadBound: with Z = D12·Δ·M,
// ∂G∞/∂φ_c = J_c = −i·Σ_{g(i)=c} k_i·Z_{:i}·X_{i:}; for a vector G∞ (p = 1
// or m = 1) linG = Σ_c |Re⟨G∞, J_c⟩|/K_c and linJ = Σ_c ‖J_c‖/K_c, else
// jac holds the J_c.
func (h *highFrequencyGain) localSlopes() {
	nd, p, m := h.e.nd, h.e.p, h.e.m
	h.linG, h.linJ = math.NaN(), math.NaN()
	if p != 1 && m != 1 {
		if h.jac == nil {
			h.jac = make([]complex128, h.dim*p*m)
			h.vert = make([]complex128, p*m)
		}
		clear(h.jac)
		for i := range nd {
			k := 1.0
			if h.k != nil {
				k = h.k[i]
			}
			jc := h.jac[h.grp[i]*p*m : (h.grp[i]+1)*p*m]
			for r := range p {
				f := complex(0, -k) * h.z[r*nd+i]
				if f == 0 {
					continue
				}
				for col := range m {
					jc[r*m+col] += f * h.x[i*m+col]
				}
			}
		}
		return
	}
	var lg, lj float64
	for c := range h.dim {
		var re, nj float64
		for r := range p {
			for col := range m {
				var jv complex128
				for i := range nd {
					if h.grp[i] != c {
						continue
					}
					k := 1.0
					if h.k != nil {
						k = h.k[i]
					}
					jv += complex(k, 0) * h.z[r*nd+i] * h.x[i*m+col]
				}
				jv *= -1i
				g := h.g[r*m+col]
				re += real(g)*real(jv) + imag(g)*imag(jv)
				nj += real(jv)*real(jv) + imag(jv)*imag(jv)
			}
		}
		lg += math.Abs(re) / h.scale[c]
		lj += math.Sqrt(nj) / h.scale[c]
	}
	h.linG, h.linJ = lg, lj
}

// quadBound bounds σ_max(G∞) within scaled distance r of the last sample φ,
// with value v: E = Δ(φ') − Δ(φ) differs from its linear part
// E₁ = Σ_c (φ'_c − φ_c)·∂Δ/∂φ_c by at most r²/2 and
// G∞(φ') − G∞(φ) = D12·N·E·X + D12·N·E·M·D22·E·X' with N = (I − ΔD22)⁻¹, so
// G∞(φ') = G∞(φ) + Σ_c (φ'_c − φ_c)·J_c + R, ‖R‖ <= ‖Z‖‖X‖r²·(1/2 +
// ‖M‖‖D22‖/(1 − ‖M‖‖D22‖r)). For |d_c| <= r/K_c, ‖G∞ + Σ d_c·J_c‖² <= v² +
// 2r·linG + (r·linJ)² for a vector G∞; for a matrix σ_max(G∞ + Σ d_c·J_c),
// convex in d, peaks at a vertex d_c = ±r/K_c of the box. +Inf when not
// applicable.
func (h *highFrequencyGain) quadBound(v, r float64) float64 {
	q := h.locM * h.e.d22
	den := 1 - q*r
	if !(den > 0) {
		return math.Inf(1)
	}
	rem := h.locA * h.locX * r * r * (0.5 + q/den)
	if h.e.p != 1 && h.e.m != 1 {
		if h.jac == nil {
			return math.Inf(1)
		}
		pm := h.e.p * h.e.m
		lin := 0.0
		for mask := range 1 << h.dim {
			copy(h.vert, h.g)
			for c := range h.dim {
				d := complex(r/h.scale[c], 0)
				if mask&(1<<c) != 0 {
					d = -d
				}
				for i, j := range h.jac[c*pm : (c+1)*pm] {
					h.vert[i] += d * j
				}
			}
			sv, err := h.e.svd.maximumFromFlat(h.vert, 0, h.e.p, h.e.m)
			if err != nil {
				return math.Inf(1)
			}
			lin = max(lin, sv)
		}
		return lin + rem
	}
	if math.IsNaN(h.linG) {
		return math.Inf(1)
	}
	return math.Sqrt(v*v+2*r*h.linG+r*r*h.linJ*h.linJ) + rem
}

// localBound bounds σ_max(G∞(φ')) − σ_max(G∞(φ)) for φ' within scaled
// distance r of the last sample φ, with E = Δ(φ') − Δ(φ), ‖E‖ <= r: by
// G∞(φ') − G∞(φ) = D12·(I − ΔD22)⁻¹·E·X' with X' = (I − D22Δ')⁻¹D21 =
// X + M·D22·E·X', ‖D12(I − ΔD22)⁻¹‖ = ‖D12·Δ·M‖ and
// ‖X'‖ <= ‖X‖/(1 − ‖M‖‖D22‖r); else, or when larger, by lip·r.
func (h *highFrequencyGain) localBound(r float64) float64 {
	b := h.lip * r
	if den := 1 - h.locM*h.e.d22*r; den > 0 {
		b = min(b, h.locA*r*h.locX/den)
	}
	return b
}

// onePhaseSup returns sup_φ σ_max(G∞(φ)) for one phase, where φ = ωh
// sweeps the whole circle as ω → ∞: the largest of 1024·k_max samples, each
// local maximum refined by golden-section search.
func (h *highFrequencyGain) onePhaseSup() float64 {
	theta := make([]float64, 1)
	at := func(t float64) float64 {
		theta[0] = t
		return h.at(theta)
	}
	k := 1024 * int(h.kmax)
	step := 2 * math.Pi / float64(k)
	vals := make([]float64, k)
	for i := range k {
		vals[i] = at(float64(i) * step)
	}
	best := math.Inf(-1)
	for i, v := range vals {
		if v < vals[(i+k-1)%k] || v < vals[(i+1)%k] {
			continue
		}
		t := float64(i) * step
		_, g := goldenMax(t-step, t+step, t, v, at)
		best = max(best, g)
	}
	return best
}

// torusGrid returns the centres of a grid of about 4096 cells on the torus
// of phases, dim consecutive values per cell, with n_c ∝ K_c cells along
// coordinate c, and the scaled half-width r = max_c π·K_c/n_c: φ_c spans at
// most r/K_c either side of a centre.
func (h *highFrequencyGain) torusGrid() ([]float64, float64) {
	if h.grid != nil {
		return h.grid, h.gridR
	}
	nd := h.dim
	prod := 1.0
	for _, s := range h.scale {
		prod *= s
	}
	n0 := math.Pow(4096/prod, 1/float64(nd))
	n, w := make([]int, nd), make([]float64, nd)
	r, total := 0.0, nd
	for c, s := range h.scale {
		n[c] = max(2, int(s*n0))
		w[c] = math.Pi / float64(n[c])
		r = max(r, s*w[c])
		total *= n[c]
	}
	cells := make([]float64, 0, total)
	idx := make([]int, nd)
	for {
		for c, k := range idx {
			cells = append(cells, (2*float64(k)+1)*w[c])
		}
		i := 0
		for ; i < nd; i++ {
			idx[i]++
			if idx[i] < n[i] {
				break
			}
			idx[i] = 0
		}
		if i == nd {
			h.grid, h.gridR = cells, r
			return cells, r
		}
	}
}

// estimate returns a lower estimate of sup σ_max(G∞): the exact supremum
// for one phase, else the largest sample of torusGrid.
func (h *highFrequencyGain) estimate() float64 {
	if h.dim == 1 {
		return h.onePhaseSup()
	}
	best := math.Inf(-1)
	for _, v := range h.gridValues() {
		best = max(best, v)
	}
	return best
}

// gridValues returns σ_max(G∞) at the cells of torusGrid, sampled once.
func (h *highFrequencyGain) gridValues() []float64 {
	if h.gridV != nil {
		return h.gridV
	}
	cells, _ := h.torusGrid()
	vals := make([]float64, 0, len(cells)/h.dim)
	for c := range slices.Chunk(cells, h.dim) {
		vals = append(vals, h.at(c))
	}
	h.gridV = vals
	return vals
}

// below reports whether σ_max(G∞) <= level on the whole torus of phases,
// by branch and bound: G∞ is Lipschitz in θ with constant lip (see
// lipschitz) in the max-norm of K_c·θ_c, since
// dΔ(I − D22Δ)⁻¹ = (I − ΔD22)⁻¹·dΔ·(I − D22Δ)⁻¹ and |dΔ_i| <= k_i·|dθ_g(i)|,
// so a cell of scaled half-width r around a sample v cannot exceed v + lip·r,
// nor the tighter localBound and quadBound from the sample's own resolvent.
// Cells that may exceed level are split until none remain, a sample exceeds
// level, or the sample budget is spent.
func (h *highFrequencyGain) below(level float64, budget int) bool {
	nd := h.dim
	cells, r := h.torusGrid()
	var bufs [2][]float64
	for gen := 0; len(cells) > 0; gen++ {
		next := bufs[gen%2][:0]
		for c := range slices.Chunk(cells, nd) {
			v := h.at(c)
			if !(v <= level) {
				h.over = max(h.over, v)
				return false
			}
			if v+h.lip*r <= level {
				continue
			}
			if h.local(); min(v+h.localBound(r), h.quadBound(v, r)) <= level {
				continue
			}
			for mask := range 1 << nd {
				for i := range nd {
					d := r / 2 / h.scale[i]
					x := c[i] - d
					if mask&(1<<i) != 0 {
						x = c[i] + d
					}
					next = append(next, x)
				}
			}
		}
		if h.samples+len(next)/nd > budget {
			return false
		}
		bufs[gen%2] = next
		cells, r = next, r/2
	}
	return true
}

// levelBelow searches a level in (sup σ_max(G∞), best) that below certifies,
// starting midway from the lower estimate est. A sample above a trial level
// raises est to it; when below runs out of budget instead, the
// distinct-delay torus is tried with what remains of it, so all attempts
// together take at most one budget of samples.
func (h *highFrequencyGain) levelBelow(est, best float64) (float64, bool) {
	const budget = 1 << 21
	for range 16 {
		if !(est < best) {
			break
		}
		level := (est + best) / 2
		if h.below(level, budget) {
			return level, true
		}
		if h.over > level {
			est = max(est, h.over)
			continue
		}
		d := h.distinctDelays()
		return level, d != nil && d.below(level, budget-h.samples)
	}
	return 0, false
}

// period returns 2π/h, the period in ω of G∞ for one phase φ = ωh, else 0.
func (h *highFrequencyGain) period() float64 {
	if h.dim != 1 || len(h.e.tau) == 0 {
		return 0
	}
	k := 1.0
	if h.k != nil {
		k = h.k[0]
	}
	return 2 * math.Pi * k / h.e.tau[0]
}

// tailGain bounds σ_max(G(jω)) for ω ≥ w by lim + ‖G − G∞‖, with
// ‖G − G∞‖ ≤ (h11 − d11) + min(h12·h21/(1 − h22) − d12·d21/(1 − d22),
// ‖P12·Q·P21 − |D12|·Q0·|D21|‖) from the Neumann series of the delay loop,
// bounded termwise in norm or entrywise: Pij bounds |Hij| entrywise and
// Q = (I − P22)⁻¹, Q0 = (I − |D22|)⁻¹. Failing both, with ‖M0‖ <= r for
// M0 = (I − D22Δ)⁻¹ on the axis and ε = ‖H22 − D22‖ < 1/r, M = (I − H22Δ)⁻¹ =
// (I − M0EΔ)⁻¹M0 has ‖M‖ <= m = r/(1 − rε) and M − M0 = M·EΔ·M0, so the loop
// gap is at most (h12 − d12)·m·h21 + d12·r·m·ε·h21 + d12·r·(h21 − d21).
func (e *delayLFT) tailGain(lim, w float64) float64 {
	h11 := e.h(e.d11, e.c1, e.b1, w)
	h12 := e.h(e.d12, e.c1, e.b2, w)
	h21 := e.h(e.d21, e.c2, e.b1, w)
	h22 := e.h(e.d22, e.c2, e.b2, w)
	loop := math.Inf(1)
	if h22 < 1 {
		loop = h12*h21/(1-h22) - e.d12*e.d21/(1-e.d22)
	}
	loop = min(loop, e.entrywiseLoopGap(w))
	if math.IsInf(loop, 1) {
		d := e.difference()
		r := min(d.rhp, d.torus)
		if eps := h22 - e.d22; r*eps < 1 {
			m := r / (1 - r*eps)
			loop = (h12-e.d12)*m*h21 + e.d12*r*m*eps*h21 + e.d12*r*(h21-e.d21)
		}
	}
	return lim + (h11 - e.d11) + loop
}

// entrywiseLoopGap returns ‖P12·Q·P21 − |D12|·Q0·|D21|‖ for |s| ≥ w, or +Inf
// when ρ(P22) >= 1.
func (e *delayLFT) entrywiseLoopGap(w float64) float64 {
	nd, p, m := e.nd, e.p, e.m
	if e.q0 == nil {
		return math.Inf(1)
	}
	g := e.resolvent(w)
	q, ok := nonnegResolvent(entryBound(e.abs22, e.rowC2, e.colB2, nd, nd, g), nd)
	if !ok {
		return math.Inf(1)
	}
	p12 := mat.NewDense(p, nd, entryBound(e.abs12, e.rowC1, e.colB2, p, nd, g))
	p21 := mat.NewDense(nd, m, entryBound(e.abs21, e.rowC2, e.colB1, nd, m, g))
	var gap, ref mat.Dense
	gap.Product(p12, mat.NewDense(nd, nd, q), p21)
	ref.Product(mat.NewDense(p, nd, slices.Clone(e.abs12)), mat.NewDense(nd, nd, e.q0), mat.NewDense(nd, m, slices.Clone(e.abs21)))
	gap.Sub(&gap, &ref)
	raw := gap.RawMatrix()
	for i := range p {
		for j := range m {
			k := i*raw.Stride + j
			raw.Data[k] = max(raw.Data[k], 0)
		}
	}
	return spectralNorm(&gap)
}

// stability counts the closed-loop right-half-plane roots of χ with the
// Nyquist test of the delay loop.
func (e *delayLFT) stability() (*delayLoop, delayNyquist, error) {
	return e.nyquist(false)
}

func (e *delayLFT) nyquist(axisOnly bool) (*delayLoop, delayNyquist, error) {
	l := e.loop(axisOnly)
	res, err := l.nyquist(0.5)
	if err == nil && e.err != nil {
		err = e.err
	}
	return l, res, err
}

// internalDelayStable reports whether every root of χ of a continuous model
// with internal delays lies in the open left half-plane.
func internalDelayStable(sys *System) (bool, error) {
	e, err := newDelayLFT(sys)
	if err != nil {
		return false, err
	}
	if e.unstableDifferenceOperator() {
		return false, nil
	}
	_, res, err := e.stability()
	if err != nil {
		return false, err
	}
	return res.stable, nil
}

// hinfNormDelayed returns the H∞ norm of a continuous model with internal
// delays, or +Inf for an unstable one, and whether the model is stable.
func hinfNormDelayed(sys *System) (norm, omega float64, stable bool, err error) {
	e, err := newDelayLFT(sys)
	if err != nil {
		return 0, 0, false, err
	}
	if e.unstableDifferenceOperator() {
		return math.Inf(1), math.Inf(1), false, nil
	}
	l, res, err := e.stability()
	if err != nil {
		return 0, 0, false, err
	}
	if !res.stable {
		return math.Inf(1), math.Inf(1), false, nil
	}
	norm, omega, err = e.peakOrZero(l, res)
	if err != nil {
		return 0, 0, false, err
	}
	return norm, omega, true, nil
}

// linfNormDelayed returns the L∞ norm of a continuous model with internal
// delays, sup_ω σ_max(G(jω)) regardless of stability: +Inf at ω = axisW when χ
// has a root on the imaginary axis, else the peak over the Nyquist grid, which
// is complete whenever the grid tail is certified on the axis, also for an
// unstable difference operator bounded on the torus.
func linfNormDelayed(sys *System) (norm, omega float64, err error) {
	e, l, res, err := delayPeakSetup(sys)
	if err != nil {
		return 0, 0, err
	}
	if res.axisRoot {
		return math.Inf(1), res.axisW, nil
	}
	return e.peakOrZero(l, res)
}

// delayPeakSetup returns the Nyquist grid of the L∞ peak, whose root count is
// void when only the torus bound certifies the tail.
func delayPeakSetup(sys *System) (*delayLFT, *delayLoop, delayNyquist, error) {
	e, err := newDelayLFT(sys)
	if err != nil {
		return nil, nil, delayNyquist{}, err
	}
	l, res, err := e.nyquist(true)
	if err != nil {
		return nil, nil, delayNyquist{}, err
	}
	return e, l, res, nil
}

func (e *delayLFT) peakOrZero(l *delayLoop, res delayNyquist) (float64, float64, error) {
	if e.m == 0 || e.p == 0 {
		return 0, 0, nil
	}
	return e.peak(l, res.w)
}

// peak searches sup_ω σ_max(G(jω)) on the Nyquist grid, which bisects
// wherever arg χ turns by more than π/4 and so is densest at the lightly
// damped closed-loop roots where G resonates, together with the midpoints of
// its intervals. Each local maximum is refined by golden-section search on
// the exact response. Past the grid end the gain is bounded by tailGain; the
// delay grid of the Nyquist test is extended until that bound stays below
// the peak. When the high-frequency limit exceeds every finite sample it is
// reported at ω = +Inf, as σ_max(D) is for a rational model, once no finite
// sample exceeds it on the grid nor past it (see peakPastLimit); with
// several classes of delays only a finite peak above it is (see
// torusPastLimit). Between
// samples, up to the grid end, the peak is certified to peakCertTol by
// descriptorResponse.certify.
func (e *delayLFT) peak(l *delayLoop, grid []float64) (float64, float64, error) {
	const tol = 1e-10
	best, wBest, err := e.scan(grid)
	if err != nil {
		return 0, 0, err
	}
	lim, period := e.d11, 0.0
	var hfx *highFrequencyExpansion
	var torus *highFrequencyGain
	if e.directFeedthrough {
		hf := newHighFrequencyGain(e)
		period = hf.period()
		est := hf.estimate()
		level, ok := hf.levelBelow(est, best)
		switch {
		case ok:
			lim = level
		case hf.dim == 1 && est > best*(1+tol):
			lim = est
		case hf.dim > 1:
			torus = hf
		default:
			return 0, 0, fmt.Errorf("gain may peak as ω → ∞ through a delayed feedthrough: %w", errDelayPeakUnsupported)
		}
		hfx = newHighFrequencyExpansion(e, hf)
	}
	cert := append([]float64{0}, grid...)
	atInf := false
	if torus == nil && lim > best*(1+tol) {
		top, wTop, err := e.certify(l, cert, lim, math.Inf(1))
		if err != nil {
			return 0, 0, err
		}
		best, wBest, atInf = top, wTop, top == lim
	} else if best, wBest, err = e.certify(l, cert, best, wBest); err != nil {
		return 0, 0, err
	}

	rGrid := grid[len(grid)-1]
	span := math.Inf(1)
	if l.tau > 0 {
		span = float64(l.pointBudget()) * math.Pi / (8 * l.tau)
	}
	points := func(lo, R float64) []float64 {
		g := l.grid(R, R)
		i, _ := slices.BinarySearch(g, math.Nextafter(lo, math.Inf(1)))
		if i == 0 {
			return append([]float64{lo}, g...)
		}
		g[i-1] = lo
		return g[i-1:]
	}
	extend := func(lo, R float64) ([]float64, float64, float64, error) {
		ext := points(lo, R)
		b, w, err := e.scan(ext)
		return ext, b, w, err
	}
	hi := rGrid
	if torus != nil {
		var found *pastLimit
		if lim, found, hi, err = e.torusPastLimit(torus, best, tol, rGrid, math.Pi/(8*l.tau), span, points); err != nil {
			return 0, 0, err
		}
		if found != nil {
			if best, wBest, err = e.certify(l, found.ext, found.b, found.w); err != nil {
				return 0, 0, err
			}
		}
	}
	if atInf {
		found, err := e.peakPastLimit(lim, tol, period, hfx, rGrid+span, &hi, points)
		if err != nil {
			return 0, 0, err
		}
		if found == nil {
			return best, wBest, nil
		}
		best, wBest = found.b, found.w
		if best, wBest, err = e.certify(l, found.ext, best, wBest); err != nil {
			return 0, 0, err
		}
	}
	for {
		level := best * (1 + tol)
		R0 := max(hi, 2*e.normA+1)
		R, closed := R0, false
		for range 200 {
			if closed = e.tailGain(lim, R) <= level; closed {
				break
			}
			R *= 2
		}
		if hfx != nil && (!closed || (R-hi)*8*l.tau/math.Pi > hfx.circleCost()) {
			certified := false
			for Rt := R0; Rt < R && Rt-rGrid <= span; Rt *= 2 {
				if certified = hfx.tailBelow(level, 1/Rt); certified {
					R, closed = Rt, true
					break
				}
			}
			if !certified && (!closed || R-rGrid > span) && 2*R0-rGrid <= span {
				// The finite peak may lie further out, above best.
				R, closed = 2*R0, true
			}
		}
		if !closed {
			return 0, 0, fmt.Errorf("no finite frequency bound for the gain: %w", errDelayPeakUnsupported)
		}
		if R-rGrid > span {
			return 0, 0, fmt.Errorf("delay grid exceeds %d points: %w", l.pointBudget(), errDelayPeakUnsupported)
		}
		if R <= hi {
			return best, wBest, nil
		}
		ext, b, w, err := extend(hi, R)
		if err != nil {
			return 0, 0, err
		}
		if b > best {
			best, wBest = b, w
		}
		if best, wBest, err = e.certify(l, ext, best, wBest); err != nil {
			return 0, 0, err
		}
		hi = R
	}
}

// pastLimit is a finite sample above the high-frequency limit, found on the
// grid extension ext, which spans every band scanned past the grid end.
type pastLimit struct {
	ext  []float64
	b, w float64
}

// peakPastLimit scans past the grid end *hi for a finite gain above the
// limit lim, which every grid sample stays below. With the expansion of
// highFrequencyExpansion, limitVerdict decides: when the finite peaks exceed
// lim, the extension doubles until a sample does, and fails past rMax; when
// they stay below, it scans to the horizon W <= rMax of limitVerdict, densely
// up to the span below and past it only where phaseScan finds the expansion
// may reach lim. Undecided, or without the expansion (several delay
// classes), it scans an extension of span max(R, 2P), with R the grid end
// and P = 2π/h the period of G∞ for one phase φ = ωh, which tailGain cannot
// close since G − G∞ decays only as 1/ω. It returns nil when no sample
// exceeds lim, which is then the peak at ω = +Inf; *hi is advanced to the
// end of the scan.
func (e *delayLFT) peakPastLimit(lim, tol, period float64, hfx *highFrequencyExpansion, rMax float64, hi *float64,
	points func(lo, R float64) []float64) (*pastLimit, error) {
	level := lim * (1 + tol)
	extend := func(lo, R float64) (*pastLimit, error) {
		ext := points(lo, R)
		b, w, err := e.scan(ext)
		if err != nil || !(b > level) {
			return nil, err
		}
		return &pastLimit{ext, b, w}, nil
	}
	R0 := max(*hi, 2*e.normA+1)
	R := R0 + max(R0, 2*period)
	dense := R
	finite, W, decided := false, 0.0, false
	if hfx != nil {
		finite, W, decided = hfx.limitVerdict(lim, tol, *hi, min(rMax, 1e12))
	}
	if decided && !finite {
		R = max(R, W)
	}
	for Rt := R0; Rt < R; Rt *= 2 {
		if e.tailGain(lim, Rt) <= level {
			R = Rt
			break
		}
	}
	if Rs := min(R, rMax); decided && !finite && Rs > dense {
		h0 := *hi
		*hi = Rs
		found, err := e.phaseScan(hfx, level, h0, dense, Rs, points)
		if found != nil && err == nil {
			found, err = extend(h0, Rs)
		}
		return found, err
	}
	h0 := *hi
	for {
		if Rs := min(R, rMax); Rs > *hi {
			found, err := extend(*hi, Rs)
			*hi = Rs
			if found != nil {
				found.ext = points(h0, Rs)
			}
			if found != nil || err != nil {
				return found, err
			}
		}
		if !finite {
			return nil, nil
		}
		if *hi >= rMax {
			return nil, fmt.Errorf("finite peak above the ω → ∞ limit past %d grid points: %w", delayNyquistMaxPoints, errDelayPeakUnsupported)
		}
		R *= 2
	}
}

// phaseScan reports whether a grid point of (lo, R] may exceed level,
// scanning those up to dense, and past it, in octaves [ω1, ω2] of ω, only
// those near a phase φ = ωh where phaseMask cannot bound the expansion over
// t = 1/ω ∈ [1/ω2, 1/ω1] plus its Cauchy remainder by level. A sample above
// level is returned, for the dense scan to redo; nil otherwise.
func (e *delayLFT) phaseScan(hfx *highFrequencyExpansion, level, lo, dense, R float64, points func(lo, R float64) []float64) (*pastLimit, error) {
	ws := points(lo, R)
	keep := make([]bool, len(ws))
	h := hfx.phaseRate()
	var mask []bool
	bot, top := 0.0, 0.0
	for i, w := range ws {
		if w <= dense {
			keep[i] = true
			continue
		}
		if top == 0 || w > top {
			if top == 0 {
				bot, top = dense, 2*dense
			}
			for w > top {
				bot, top = top, 2*top
			}
			mask = nil
			t2 := 1 / bot
			if r := hfx.radius(t2); r > 0 {
				rem := hfx.remainder(e.tailGain(0, r), t2*r)
				mask = hfx.phaseMask(1/min(top, R), t2, level-rem)
			}
		}
		if mask == nil || mask[hfx.sampleAt(w*h)] {
			keep[max(i-1, 0)], keep[i] = true, true
			if i+1 < len(ws) {
				keep[i+1] = true
			}
		}
	}
	var sub []float64
	for i, w := range ws {
		if keep[i] {
			sub = append(sub, w)
		}
	}
	if len(sub) == 0 {
		return nil, nil
	}
	b, w, err := e.scan(sub)
	if err != nil || !(b > level) {
		return nil, err
	}
	return &pastLimit{sub, b, w}, nil
}

// certify raises (peak, w) to within peakCertTol of sup σ_max(G(jω)) over
// the span of the sorted grid ws; see descriptorResponse.certify.
func (e *delayLFT) certify(l *delayLoop, ws []float64, peak, w float64) (float64, float64, error) {
	if e.desc == nil {
		desc, err := newDescriptorResponse(e.sys)
		if err != nil {
			return 0, 0, err
		}
		e.desc = desc
	}
	ps, wps := []float64{peak}, []float64{w}
	if err := e.desc.certify(ws, []float64{0}, ps, wps, l.pointBudget(), errDelayPeakUnsupported); err != nil {
		return 0, 0, err
	}
	return ps[0], wps[0], nil
}

// scan samples σ_max on the sorted grid and its interval midpoints and
// refines every local maximum within a factor 2 of the largest sample.
func (e *delayLFT) scan(grid []float64) (float64, float64, error) {
	ws := slices.Grow(e.scanW[:0], 2*len(grid))
	for i, w := range grid {
		if i > 0 {
			ws = append(ws, (grid[i-1]+w)/2)
		}
		ws = append(ws, w)
	}
	vals := slices.Grow(e.scanV[:0], len(ws))[:len(ws)]
	e.scanW, e.scanV = ws, vals
	for i, w := range ws {
		vals[i] = e.sigma(w)
	}
	if e.err != nil {
		return 0, 0, e.err
	}
	if slices.ContainsFunc(vals, func(v float64) bool { return math.IsInf(v, 0) }) {
		return 0, 0, fmt.Errorf("gain unbounded on the imaginary axis: %w", errDelayPeakUnsupported)
	}
	top := slices.Max(vals)
	best, wBest := math.Inf(-1), 0.0
	for k, v := range vals {
		if v < 0.5*top || k > 0 && v < vals[k-1] || k+1 < len(vals) && v < vals[k+1] {
			continue
		}
		w, g := goldenMax(ws[max(k-1, 0)], ws[min(k+1, len(ws)-1)], ws[k], v, e.sigma)
		if g > best {
			best, wBest = g, w
		}
	}
	if e.err != nil {
		return 0, 0, e.err
	}
	return best, wBest, nil
}
