package controlsys

import (
	"errors"
	"fmt"
	"math"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// StateProjection selects how Balred and Modred eliminate states, as the
// MATLAB balredOptions StateProjection option and the modred method argument.
// The zero value is MatchDC, the MATLAB default for both.
type StateProjection int

const (
	// MatchDC residualizes the eliminated states (singular perturbation) so
	// the reduced model keeps the DC gain: their derivatives (continuous) or
	// increments (discrete) are set to zero.
	MatchDC StateProjection = iota
	// Truncate discards the eliminated states without altering the others.
	Truncate
)

func (m StateProjection) validate(op string) error {
	if m != MatchDC && m != Truncate {
		return fmt.Errorf("%s: unknown state projection %d: %w", op, m, ErrInvalidArgument)
	}
	return nil
}

// BalredOptions mirrors MATLAB balredOptions. The zero value selects MatchDC.
type BalredOptions struct {
	StateProjection StateProjection
}

// BalrealResult holds the outputs of MATLAB [sysb,g,TL,TR] = balreal(sys):
// Sys.A = TL·A·TR, Sys.B = TL·B, Sys.C = C·TR, and TL·TR = I to the
// rounding of its terms, so the balanced state is x̄ = TL·x and x = TR·x̄.
// HSV holds the Hankel singular values in decreasing order.
type BalrealResult struct {
	Sys *System
	HSV []float64
	TL  *mat.Dense
	TR  *mat.Dense
}

// Balreal computes the balanced realization of a stable, minimal LTI system,
// as MATLAB balreal; see
// https://www.mathworks.com/help/control/ref/dynamicsystem.balreal.html.
// The returned system has controllability and observability gramians equal
// to diag(σ₁, σ₂, …, σₙ), σᵢ the Hankel singular values, to within the
// certified pairing tolerance below relative to σ₁ and the rounding of the
// returned matrices, which a pole within δ of the stability boundary
// amplifies by about 1/δ.
//
// Unlike MATLAB, which balances the stable part and reports Inf for unstable
// modes, an unstable model returns ErrUnstable, decided by IsStable within
// its rounding of the band. Minimality is decided
// exactly on the stored data: a model whose controllability or
// observability matrix is rank deficient over the rationals returns
// ErrNotMinimal, and ErrMinimalityUndetermined when the exact test exceeds
// its work bound. The Hankel singular values are accurate relative to σ₁;
// a minimal model can still have some below the float64 rounding of σ₁,
// whose singular directions are known only to that rounding. Balreal
// certifies the pairing error E of the computed balancing vectors, with
// E = TL·TR − I evaluated exactly: ‖E‖_F ≤ min(2⁻¹⁰, max(10⁻⁸,
// 4·n·ε·σ₁/σₙ)). A model whose transformation fails this, being numerically
// non-minimal, returns ErrSingularTransform; Balred can still reduce it to a
// certified order. Balreal then returns TL corrected to (I+E)⁻¹·TL, so that
// TL·TR = I to the rounding of its terms and Sys is the similarity of the
// stored model by TR: ‖Sys.A − TR⁻¹·A·TR‖_F is within the rounding of the
// products forming it. A realization that IsStable does not confirm is
// returned with ErrAccuracyUnverified. Gramians are
// refined against the stored data; when the Lyapunov equations are too
// ill-conditioned for that, Balreal returns ErrSingularEquation. A model
// with no states returns ErrDimensionMismatch, since TL and TR cannot be
// formed. A result entry beyond float64 range returns ErrOverflow.
func Balreal(sys *System) (*BalrealResult, error) {
	br, err := balanceUncertified(sys)
	if err != nil {
		return nil, err
	}
	cert := newBalanceCert(br)
	if !cert.full() {
		return nil, fmt.Errorf("Balreal: balancing transformation numerically singular, TL·TR − I above tolerance (model numerically non-minimal): %w", ErrSingularTransform)
	}
	return balancedSimilarity("Balreal", sys, br, cert)
}

// balancedSimilarity returns the certified balancing br of sys as a
// similarity of sys (similarityCorrected), with ErrAccuracyUnverified when
// IsStable does not confirm it (confirmBalancedStable).
func balancedSimilarity(op string, sys *System, br *BalrealResult, cert *balanceCert) (*BalrealResult, error) {
	br, err := similarityCorrected(op, sys, br, cert)
	if err != nil {
		return nil, err
	}
	return confirmBalancedStable(op, br)
}

// similarityCorrected returns br unchanged when the float64 pairing error of
// TL·TR is within its rounding term. Otherwise it returns TL′ = (I+E)⁻¹·TL,
// E = TL·TR − I from compensated products, with Sys.A = TL′·A·TR and
// Sys.B = TL′·B; TR, Sys.C, Sys.D and HSV are unchanged.
func similarityCorrected(op string, sys *System, br *BalrealResult, cert *balanceCert) (*BalrealResult, error) {
	if !cert.pairingInexact() {
		return br, nil
	}
	if cert.e == nil {
		cert.compute()
	}
	n, m, _ := br.Sys.Dims()
	buf := make([]float64, 3*n*n+n*m)
	lu, tl, tmp, bb := buf[:n*n], buf[n*n:2*n*n], buf[2*n*n:3*n*n], buf[3*n*n:]
	copy(lu, cert.e)
	for i := range n {
		lu[i*n+i]++
	}
	copyStrided(tl, n, br.TL.RawMatrix().Data, br.TL.RawMatrix().Stride, n, n)
	ipiv := make([]int, n)
	if !impl.Dgetrf(n, n, lu, n, ipiv) {
		return nil, fmt.Errorf("%s: balancing pairing I + TL·TR − I singular: %w", op, ErrSingularTransform)
	}
	impl.Dgetrs(blas.NoTrans, n, n, lu, n, ipiv, tl, n)
	tlGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: tl}
	aRaw, bRaw := sys.A.RawMatrix(), sys.B.RawMatrix()
	ab := make([]float64, n*n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tlGen, aRaw, 0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tmp})
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, blas64.General{Rows: n, Cols: n, Stride: n, Data: tmp}, br.TR.RawMatrix(),
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: ab})
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tlGen, bRaw, 0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bb})
	TL, Ab, Bb := mat.NewDense(n, n, tl), mat.NewDense(n, n, ab), mat.NewDense(n, m, bb)
	for _, out := range []struct {
		name string
		m    *mat.Dense
	}{{"A", Ab}, {"B", Bb}, {"TL", TL}} {
		if err := finiteDense(op, out.name, out.m, ErrOverflow); err != nil {
			return nil, err
		}
	}
	res, err := newRealizationTransformPolicy(sys).result(Ab, Bb, br.Sys.C, br.Sys.D)
	if err != nil {
		return nil, err
	}
	return &BalrealResult{Sys: res, HSV: br.HSV, TL: TL, TR: br.TR}, nil
}

// confirmBalancedStable returns br, with ErrAccuracyUnverified when IsStable
// does not confirm br.Sys.
func confirmBalancedStable(op string, br *BalrealResult) (*BalrealResult, error) {
	st, err := br.Sys.IsStable()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	if !st {
		return br, fmt.Errorf("%s: balanced realization has a pole IsStable does not confirm, within its boundary band: %w", op, ErrAccuracyUnverified)
	}
	return br, nil
}

// balanceUncertified is Balreal without the certificate of TL·TR = I.
func balanceUncertified(sys *System) (*BalrealResult, error) {
	if err := requireFiniteSystem("Balreal", sys); err != nil {
		return nil, err
	}
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Balreal"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("Balreal"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 {
		return nil, fmt.Errorf("Balreal: system has no states: %w", ErrDimensionMismatch)
	}

	stable, err := sys.IsStable()
	if err != nil {
		return nil, fmt.Errorf("Balreal: %w", err)
	}
	if !stable {
		return nil, fmt.Errorf("Balreal: %w", ErrUnstable)
	}
	if m == 0 || p == 0 {
		return nil, fmt.Errorf("Balreal: model has no inputs or no outputs, so all Hankel singular values are zero: %w", ErrNotMinimal)
	}

	switch exactMinimality(sys.A, sys.B, sys.C) {
	case notMinimalExact:
		return nil, fmt.Errorf("Balreal: %w", ErrNotMinimal)
	case minimalityUndetermined:
		return nil, fmt.Errorf("Balreal: %w", ErrMinimalityUndetermined)
	}

	aRaw := sys.A.RawMatrix()
	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	bScaled, eB := pow2Scaled(sys.B)
	cScaled, eC := pow2Scaled(sys.C)
	bsRaw := bScaled.RawMatrix()
	csRaw := cScaled.RawMatrix()

	bufSize := 10*n*n + 2*n*m + 2*p*n + n
	ws := make([]float64, bufSize)
	wi := 0

	alloc := func(size int) []float64 {
		res := ws[wi : wi+size : wi+size]
		wi += size
		return res
	}

	aData := alloc(n * n)
	at := alloc(n * n)
	copyStrided(aData, n, aRaw.Data, aRaw.Stride, n, n)
	for i := range n {
		for j := range n {
			at[i*n+j] = aRaw.Data[j*aRaw.Stride+i]
		}
	}
	fb := alloc(n * m)
	copyStrided(fb, m, bsRaw.Data, bsRaw.Stride, n, m)
	fc := alloc(n * p)
	for i := range n {
		for j := range p {
			fc[i*p+j] = csRaw.Data[j*csRaw.Stride+i]
		}
	}

	A := mat.NewDense(n, n, aData)
	At := mat.NewDense(n, n, at)
	discrete := !sys.IsContinuous()
	gc, err := balrealGramian(A, fb, m, discrete, "controllability", nil)
	if err != nil {
		return nil, err
	}
	gob, err := balrealGramian(At, fc, p, discrete, "observability", &gc.schur)
	if err != nil {
		return nil, err
	}
	lc, err := semidefiniteFactor(gc.x, n)
	if err != nil {
		return nil, fmt.Errorf("Balreal: controllability gramian factor: %w", err)
	}
	lo, err := semidefiniteFactor(gob.x, n)
	if err != nil {
		return nil, fmt.Errorf("Balreal: observability gramian factor: %w", err)
	}

	mData := alloc(n * n)
	s := alloc(n)
	uData := alloc(n * n)
	vtData := alloc(n * n)
	gemmTN(lo, lc, mData, n)
	svd := func() error {
		wq := make([]float64, 1)
		impl.Dgesvd(lapack.SVDAll, lapack.SVDAll, n, n, mData, n, s,
			uData, n, vtData, n, wq, -1)
		svdWork := make([]float64, int(wq[0]))
		if !impl.Dgesvd(lapack.SVDAll, lapack.SVDAll, n, n, mData, n, s,
			uData, n, vtData, n, svdWork, len(svdWork)) {
			return fmt.Errorf("Balreal: SVD of gramian factors did not converge: %w", ErrSchurFailed)
		}
		return nil
	}
	if err := svd(); err != nil {
		return nil, err
	}
	if s[0] > 0 && eps()*scaledNorm2(gc.x)*scaledNorm2(gob.x)/(s[0]*s[0]) > hsvDDGate {
		ddc, ok, err := gc.ddCholesky()
		if err != nil {
			return nil, fmt.Errorf("Balreal: controllability gramian: %w", err)
		}
		ddo, ok2, err := gob.ddCholesky()
		if err != nil {
			return nil, fmt.Errorf("Balreal: observability gramian: %w", err)
		}
		if !ok || !ok2 {
			return nil, fmt.Errorf("Balreal: gramians too unbalanced for accurate Hankel singular values: %w", ErrSingularEquation)
		}
		ddProductTN(ddo, ddc, mData, n)
		for i := range lc {
			lc[i] = ddc[i].hi + ddc[i].lo
			lo[i] = ddo[i].hi + ddo[i].lo
		}
		if err := svd(); err != nil {
			return nil, err
		}
	}
	lcGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: lc}
	loGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: lo}
	if !ldexpFinite(lc, eB) || !ldexpFinite(lo, eC) || !ldexpFinite(s, eB+eC) {
		return nil, fmt.Errorf("Balreal: gramian factors exceed float64 range: %w", ErrOverflow)
	}
	if !(s[0] > 0) {
		return nil, fmt.Errorf("Balreal: Hankel singular values of the minimal model underflow to zero: %w", ErrSingularTransform)
	}
	// Guards the division for Hankel singular values that round to 0.
	sFloor := math.Ldexp(s[0], -106)

	// V = Vt' — scale columns of Vt' by 1/sqrt(σ) → vScaled
	// T = Lc * vScaled
	vScaled := alloc(n * n)
	for i := range n {
		invSqrt := 1.0 / math.Sqrt(max(s[i], sFloor))
		for j := range n {
			vScaled[j*n+i] = vtData[i*n+j] * invSqrt
		}
	}

	tData := alloc(n * n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, lcGen,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: vScaled},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tData})
	T := mat.NewDense(n, n, tData)

	// Tinv = Σ^{-1/2} * U' * Lo'
	// Scale columns of U by 1/sqrt(σ) → uScaled, then Tinv = uScaled' * Lo'
	for j := range n {
		invSqrt := 1.0 / math.Sqrt(max(s[j], sFloor))
		for i := range n {
			uData[i*n+j] *= invSqrt
		}
	}

	tinvData := alloc(n * n)
	blas64.Gemm(blas.Trans, blas.Trans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: uData},
		loGen,
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tinvData})
	Tinv := mat.NewDense(n, n, tinvData)

	tGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: tData}
	tiGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: tinvData}

	tmp := alloc(n * n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tiGen,
		blas64.General{Rows: n, Cols: n, Stride: aRaw.Stride, Data: aRaw.Data},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: tmp})
	abData := alloc(n * n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: tmp},
		tGen,
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: abData})

	bbData := alloc(n * m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tiGen,
		blas64.General{Rows: n, Cols: m, Stride: bRaw.Stride, Data: bRaw.Data},
		0, blas64.General{Rows: n, Cols: m, Stride: m, Data: bbData})

	cbData := alloc(p * n)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1,
		blas64.General{Rows: p, Cols: n, Stride: cRaw.Stride, Data: cRaw.Data},
		tGen,
		0, blas64.General{Rows: p, Cols: n, Stride: n, Data: cbData})

	Ab := mat.NewDense(n, n, abData)
	Bb := mat.NewDense(n, m, bbData)
	Cb := mat.NewDense(p, n, cbData)
	Db := denseCopy(sys.D)
	for _, out := range []struct {
		name string
		m    *mat.Dense
	}{{"A", Ab}, {"B", Bb}, {"C", Cb}, {"TL", Tinv}, {"TR", T}} {
		if err := finiteDense("Balreal", out.name, out.m, ErrOverflow); err != nil {
			return nil, err
		}
	}

	balSys, err := policy.result(Ab, Bb, Cb, Db)
	if err != nil {
		return nil, err
	}

	return &BalrealResult{Sys: balSys, HSV: s, TL: Tinv, TR: T}, nil
}

// hsvDDGate is the bound on ε·‖Wc‖·‖Wo‖/σ₁², the size of the float64
// rounding of the gramians relative to σ₁², above which the Hankel singular
// values are recomputed from double-double gramian factors.
const hsvDDGate = 1e-12

// balrealGramian returns the refined gramian of A and the n×k right-hand
// factor f, or ErrSingularEquation when refinement cannot make it accurate;
// transposeOf is as for newRefinedGramian.
func balrealGramian(A *mat.Dense, f []float64, k int, discrete bool, name string, transposeOf *schurLyapunov) (*refinedGramian, error) {
	n, _ := A.Dims()
	g, ok, err := newRefinedGramian(A, f, k, discrete, transposeOf)
	if err != nil {
		return nil, fmt.Errorf("Balreal: %s gramian: %w", name, err)
	}
	if !ok {
		return nil, fmt.Errorf("Balreal: %s gramian not accurate in float64, the Lyapunov equation is too ill-conditioned: %w", name, ErrSingularEquation)
	}
	if err := finiteDense("Balreal", name+" gramian", mat.NewDense(n, n, g.x), ErrOverflow); err != nil {
		return nil, err
	}
	return g, nil
}

// gemmTN writes Loᵀ·Lc into dst, all n×n of stride n.
func gemmTN(lo, lc, dst []float64, n int) {
	blas64.Gemm(blas.Trans, blas.NoTrans, 1,
		blas64.General{Rows: n, Cols: n, Stride: n, Data: lo},
		blas64.General{Rows: n, Cols: n, Stride: n, Data: lc},
		0, blas64.General{Rows: n, Cols: n, Stride: n, Data: dst})
}

// ddProductTN writes Loᵀ·Lc, accumulated in double-double and rounded once,
// into dst.
func ddProductTN(lo, lc []dd, dst []float64, n int) {
	for i := range n {
		for j := range n {
			var s dd
			for k := range n {
				s = s.add(lo[k*n+i].mul(lc[k*n+j]))
			}
			dst[i*n+j] = s.hi + s.lo
		}
	}
}

// Balred reduces sys to order states by balanced truncation, as MATLAB
// [rsys,info] = balred(sys,order,opts); see
// https://www.mathworks.com/help/control/ref/dynamicsystem.balred.html.
// hsv holds the Hankel singular values (MATLAB info.HSV). order must be in
// [0, n]; order n returns the balanced realization of Balreal, with its
// error, and order 0 a static gain. opts.StateProjection selects MatchDC (the zero value, MATLAB
// default) or Truncate. sys must satisfy the requirements of Balreal, except
// its certificate of TL·TR = I.
//
// When the transformation is certified, Balred at order < n reduces the
// balanced realization before Balreal's similarity correction of TL; the two
// differ by at most the certified tolerance on TL·TR − I. An order that would keep part of a tail of Hankel singular
// values at or below n·ε·σ₁ returns ErrSingularTransform naming the
// numerically minimal order (SLICOT AB09AD and AB09BD likewise cap the order
// there). When it is not, Balred reduces the leading balanced states it does
// certify, after truncating the directions whose Hankel singular values are
// at most n·ε·σ₁ or, failing their certificate, at most 2¹²·n·ε·σ₁; MatchDC
// then keeps the DC gain to within twice the sum of the truncated Hankel
// singular values. An order above the certified one returns
// ErrSingularTransform.
//
// Balanced truncation and singular perturbation are stable for σ_order >
// σ_order+1 but give no stability margin: an order splitting a near-equal
// pair can keep a pole within IsStable's boundary band, or within IsStable's
// rounding of it, as when a slow pole lies below ε·‖A‖_F beside a fast one
// that MatchDC creates. For order < n, a
// reduction that IsStable does not confirm is returned with
// ErrAccuracyUnverified (with ErrSingularTransform and no model when the
// transformation is not certified), as is Balreal's realization at order n. MatchDC returns ErrSingularA22 when the residualized block
// (A22, or A22 − I for discrete models) of the balanced realization is
// singular within its rounding, σmin ≤ 4·n·ε·‖A‖_F, as for Hankel singular
// values equal to rounding.
func Balred(sys *System, order int, opts BalredOptions) (*System, []float64, error) {
	if err := requireFiniteSystem("Balred", sys); err != nil {
		return nil, nil, err
	}
	if err := opts.StateProjection.validate("Balred"); err != nil {
		return nil, nil, err
	}
	br, err := balanceUncertified(sys)
	if err != nil {
		return nil, nil, err
	}
	n, _, _ := br.Sys.Dims()
	if order < 0 || order > n {
		return nil, nil, fmt.Errorf("Balred: order %d outside [0,%d]: %w", order, n, ErrInvalidOrder)
	}
	cert := newBalanceCert(br)
	full := cert.full()
	if order == n {
		if !full {
			return nil, nil, fmt.Errorf("Balred: balancing transformation numerically singular, TL·TR − I above tolerance (model numerically non-minimal): %w", ErrSingularTransform)
		}
		res, err := balancedSimilarity("Balred", sys, br, cert)
		if res == nil {
			return nil, nil, err
		}
		return res.Sys, res.HSV, err
	}
	src := br.Sys
	h := n
	head := !full
	if full {
		if nmin := cert.nmin(); order > nmin {
			return nil, nil, fmt.Errorf("Balred: order %d would keep part of the tail at the rounding of σ₁ above numerically minimal order %d: %w", order, nmin, ErrSingularTransform)
		}
	} else {
		var ok bool
		if h, ok = cert.certifiedOrder(); !ok {
			return nil, nil, fmt.Errorf("Balred: balancing transformation numerically singular above the rounding of σ₁: %w", ErrSingularTransform)
		}
		if order > h {
			return nil, nil, fmt.Errorf("Balred: order %d exceeds certified order %d of the balancing transformation (model numerically non-minimal): %w", order, h, ErrSingularTransform)
		}
		if src, err = balancedHead(sys, br, h); err != nil {
			return nil, nil, err
		}
	}
	if opts.StateProjection == MatchDC && order < h && residualizedBlockSingular(src, order, n) {
		return nil, nil, fmt.Errorf("Balred: residualized states %d..%d singular within the rounding of the balanced realization: %w", order, h-1, ErrSingularA22)
	}
	elim := make([]int, 0, h-order)
	for i := order; i < h; i++ {
		elim = append(elim, i)
	}
	red, err := modred(src, elim, opts.StateProjection)
	if err != nil {
		return nil, nil, fmt.Errorf("Balred: %w", err)
	}
	propagateIONames(red, sys)
	if red, err = derivedOp("Balred").model(red, nil); err != nil {
		return nil, nil, err
	}
	if err := reductionStability(red, order, br.HSV, head); err != nil {
		if errors.Is(err, ErrAccuracyUnverified) {
			return red, br.HSV, err
		}
		return nil, nil, err
	}
	return red, br.HSV, nil
}

// balanceA22K scales the rounding floor K·n·ε·‖A‖_F below which Balred treats
// the residualized block as singular.
const balanceA22K = 4

// residualizedBlockSingular reports whether the block of M = src.A (src.A − I
// when discrete) on states order.. has σmin ≤ balanceA22K·n·ε·‖M‖_F.
func residualizedBlockSingular(src *System, order, n int) bool {
	h, _, _ := src.Dims()
	r := h - order
	raw := src.A.RawMatrix()
	buf := make([]float64, h*h+r*r+r)
	m, m22, sv := buf[:h*h], buf[h*h:h*h+r*r], buf[h*h+r*r:]
	copyStrided(m, h, raw.Data, raw.Stride, h, h)
	if src.IsDiscrete() {
		for i := range h {
			m[i*h+i]--
		}
	}
	copyStrided(m22, r, m[order*h+order:], h, r, r)
	wq := make([]float64, 1)
	impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, r, r, m22, r, sv, nil, 1, nil, 1, wq, -1)
	work := make([]float64, int(wq[0]))
	if !impl.Dgesvd(lapack.SVDNone, lapack.SVDNone, r, r, m22, r, sv, nil, 1, nil, 1, work, len(work)) {
		return false
	}
	return sv[r-1] <= balanceA22K*float64(n)*eps()*scaledNorm2(m)
}

// reductionStability returns nil when red passes IsStable. Otherwise a
// reduction of a certified transformation returns ErrAccuracyUnverified,
// the caller keeping the model, and one of the certified head of a
// numerically non-minimal model ErrSingularTransform.
func reductionStability(red *System, order int, hsv []float64, head bool) error {
	st, err := red.IsStable()
	if err != nil {
		return fmt.Errorf("Balred: %w", err)
	}
	if st {
		return nil
	}
	if head {
		return fmt.Errorf("Balred: order %d reduction of the certified head is not stable: %w", order, ErrSingularTransform)
	}
	return fmt.Errorf("Balred: order %d reduction (σ[%d] = %.17g, σ[%d] = %.17g) has a pole IsStable does not confirm, within its boundary band or its rounding of it: %w", order, order-1, hsv[max(order-1, 0)], order, hsv[order], ErrAccuracyUnverified)
}

// balancedHead returns the leading h states of the balanced realization,
// (TL₁·A·TR₁, TL₁·B, C·TR₁, D) with TL₁ the first h rows of TL and TR₁ the
// first h columns of TR.
func balancedHead(sys *System, br *BalrealResult, h int) (*System, error) {
	policy := newRealizationTransformPolicy(sys)
	n, m, p := policy.n, policy.m, policy.p
	tl, tr := br.TL.RawMatrix(), br.TR.RawMatrix()
	tl1 := blas64.General{Rows: h, Cols: n, Stride: tl.Stride, Data: tl.Data}
	tr1 := blas64.General{Rows: n, Cols: h, Stride: tr.Stride, Data: tr.Data}
	aRaw, bRaw, cRaw := sys.A.RawMatrix(), sys.B.RawMatrix(), sys.C.RawMatrix()
	buf := make([]float64, n*h+h*h+h*m+p*h)
	at, ah, bh, ch := buf[:n*h], buf[n*h:n*h+h*h], buf[n*h+h*h:n*h+h*h+h*m], buf[n*h+h*h+h*m:]
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, aRaw, tr1, 0, blas64.General{Rows: n, Cols: h, Stride: h, Data: at})
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tl1, blas64.General{Rows: n, Cols: h, Stride: h, Data: at},
		0, blas64.General{Rows: h, Cols: h, Stride: h, Data: ah})
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, tl1, bRaw, 0, blas64.General{Rows: h, Cols: m, Stride: m, Data: bh})
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, cRaw, tr1, 0, blas64.General{Rows: p, Cols: h, Stride: h, Data: ch})
	Ah, Bh, Ch := mat.NewDense(h, h, ah), mat.NewDense(h, m, bh), mat.NewDense(p, h, ch)
	for _, out := range []struct {
		name string
		m    *mat.Dense
	}{{"A", Ah}, {"B", Bh}, {"C", Ch}} {
		if err := finiteDense("Balred", out.name, out.m, ErrOverflow); err != nil {
			return nil, err
		}
	}
	return policy.result(Ah, Bh, Ch, denseCopy(sys.D))
}

// Modred eliminates the states with 0-based indices elim, as MATLAB
// modred(sys,elim,method); see
// https://www.mathworks.com/help/control/ref/ss.modred.html. method MatchDC
// (the zero value, MATLAB default) residualizes them and keeps the DC gain;
// Truncate discards them. Indices out of range or repeated return
// ErrInvalidArgument; an empty elim returns a copy of sys, and eliminating
// every state returns a static gain. MatchDC returns ErrSingularA22 when the
// eliminated block (A22, or A22 - I for discrete models) is singular.
func Modred(sys *System, elim []int, method StateProjection) (*System, error) {
	if err := requireFiniteSystem("Modred", sys); err != nil {
		return nil, err
	}
	if err := method.validate("Modred"); err != nil {
		return nil, err
	}
	red, err := modred(sys, elim, method)
	if err != nil {
		return nil, fmt.Errorf("Modred: %w", err)
	}
	return derivedOp("Modred").model(red, nil)
}

func modred(sys *System, elim []int, method StateProjection) (*System, error) {
	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("descriptor"); err != nil {
		return nil, err
	}
	if err := policy.requireDelayFree("delay"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p
	if n == 0 || len(elim) == 0 {
		return policy.zeroOrderCopy(), nil
	}
	if m == 0 || p == 0 {
		return withZeroIOPadding("reduced model", sys, func(padded *System) (*System, error) {
			return modred(padded, elim, method)
		})
	}

	elimSet := make(map[int]bool, len(elim))
	for _, idx := range elim {
		if idx < 0 || idx >= n {
			return nil, fmt.Errorf("state index %d outside [0,%d): %w", idx, n, ErrInvalidArgument)
		}
		if elimSet[idx] {
			return nil, fmt.Errorf("duplicate state index %d: %w", idx, ErrInvalidArgument)
		}
		elimSet[idx] = true
	}

	keep := make([]int, 0, n-len(elim))
	for i := range n {
		if !elimSet[i] {
			keep = append(keep, i)
		}
	}
	r := len(keep)

	if r == 0 {
		d := denseCopy(sys.D)
		if method == MatchDC {
			var err error
			if d, err = residualizedGain(sys); err != nil {
				return nil, err
			}
		}
		g, err := newDerivedGain(d, sys.Dt)
		if err != nil {
			return nil, err
		}
		propagateIONames(g, sys)
		return g, nil
	}
	if r == n {
		return sys.Copy(), nil
	}

	perm := make([]int, n)
	copy(perm, keep)
	ei := 0
	for i := range n {
		if elimSet[i] {
			perm[r+ei] = i
			ei++
		}
	}

	aRaw := sys.A.RawMatrix()
	ap := make([]float64, n*n)
	for i := range n {
		srcRow := aRaw.Data[perm[i]*aRaw.Stride:]
		dstOff := i * n
		for j := range n {
			ap[dstOff+j] = srcRow[perm[j]]
		}
	}

	bRaw := sys.B.RawMatrix()
	bp := make([]float64, n*m)
	for i := range n {
		copy(bp[i*m:i*m+m], bRaw.Data[perm[i]*bRaw.Stride:perm[i]*bRaw.Stride+m])
	}

	cRaw := sys.C.RawMatrix()
	cp := make([]float64, p*n)
	for i := range p {
		srcRow := cRaw.Data[i*cRaw.Stride:]
		dstOff := i * n
		for j := range n {
			cp[dstOff+j] = srcRow[perm[j]]
		}
	}

	Ap := mat.NewDense(n, n, ap)
	Bp := mat.NewDense(n, m, bp)
	Cp := mat.NewDense(p, n, cp)

	A11 := extractSubmatrix(Ap, 0, r, 0, r)
	B1 := extractSubmatrix(Bp, 0, r, 0, m)
	C1 := extractSubmatrix(Cp, 0, p, 0, r)

	if method == Truncate {
		red, err := policy.result(A11, B1, C1, denseCopy(sys.D))
		if err != nil {
			return nil, err
		}
		return red, nil
	}

	red, err := singularPerturbation(Ap, Bp, Cp, sys.D, A11, B1, C1, n, m, p, r, sys.Dt)
	if err != nil {
		return nil, err
	}
	propagateIONames(red, sys)
	return red, nil
}

// singularPerturbation computes the reduced system via residualization.
// Ar = A11 - A12*inv(M)*A21, etc., with M = A22 (continuous) or A22-I
// (discrete, x2[k+1] = x2[k]). Uses Gemm beta=-1 to fuse multiply-subtract.
func singularPerturbation(
	Ab, Bb, Cb, Db *mat.Dense,
	A11, B1, C1 *mat.Dense,
	n, m, p, r int,
	dt float64,
) (*System, error) {
	A12 := extractSubmatrix(Ab, 0, r, r, n)
	A21 := extractSubmatrix(Ab, r, n, 0, r)
	A22 := extractSubmatrix(Ab, r, n, r, n)
	B2 := extractSubmatrix(Bb, r, n, 0, m)
	C2 := extractSubmatrix(Cb, 0, p, r, n)

	n2 := n - r
	slab := make([]float64, n2*n2+n2*r+n2*m+r*r+r*m+p*r+p*m)
	off := 0
	take := func(sz int) []float64 { s := slab[off : off+sz : off+sz]; off += sz; return s }

	a22Data := take(n2 * n2)
	a22Raw := A22.RawMatrix()
	copyStrided(a22Data, n2, a22Raw.Data, a22Raw.Stride, n2, n2)

	if dt > 0 {
		for i := range n2 {
			a22Data[i*n2+i]--
		}
	}

	ipiv := make([]int, n2)
	if !impl.Dgetrf(n2, n2, a22Data, n2, ipiv) {
		return nil, ErrSingularA22
	}

	rhs1 := take(n2 * r)
	a21Raw := A21.RawMatrix()
	copyStrided(rhs1, r, a21Raw.Data, a21Raw.Stride, n2, r)
	impl.Dgetrs(blas.NoTrans, n2, r, a22Data, n2, ipiv, rhs1, r)

	rhs2 := take(n2 * m)
	b2Raw := B2.RawMatrix()
	copyStrided(rhs2, m, b2Raw.Data, b2Raw.Stride, n2, m)
	impl.Dgetrs(blas.NoTrans, n2, m, a22Data, n2, ipiv, rhs2, m)

	invA22_A21 := blas64.General{Rows: n2, Cols: r, Stride: r, Data: rhs1}
	invA22_B2 := blas64.General{Rows: n2, Cols: m, Stride: m, Data: rhs2}
	a12Raw := A12.RawMatrix()
	a12Gen := blas64.General{Rows: r, Cols: n2, Stride: a12Raw.Stride, Data: a12Raw.Data}
	c2Raw := C2.RawMatrix()
	c2Gen := blas64.General{Rows: p, Cols: n2, Stride: c2Raw.Stride, Data: c2Raw.Data}

	arData := take(r * r)
	a11Raw := A11.RawMatrix()
	copyStrided(arData, r, a11Raw.Data, a11Raw.Stride, r, r)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, a12Gen, invA22_A21,
		1, blas64.General{Rows: r, Cols: r, Stride: r, Data: arData})

	brData := take(r * m)
	b1Raw := B1.RawMatrix()
	copyStrided(brData, m, b1Raw.Data, b1Raw.Stride, r, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, a12Gen, invA22_B2,
		1, blas64.General{Rows: r, Cols: m, Stride: m, Data: brData})

	crData := take(p * r)
	c1Raw := C1.RawMatrix()
	copyStrided(crData, r, c1Raw.Data, c1Raw.Stride, p, r)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, c2Gen, invA22_A21,
		1, blas64.General{Rows: p, Cols: r, Stride: r, Data: crData})

	drData := take(p * m)
	dRaw := Db.RawMatrix()
	copyStrided(drData, m, dRaw.Data, dRaw.Stride, p, m)
	blas64.Gemm(blas.NoTrans, blas.NoTrans, -1, c2Gen, invA22_B2,
		1, blas64.General{Rows: p, Cols: m, Stride: m, Data: drData})

	red, err := newNoCopy(
		mat.NewDense(r, r, arData),
		mat.NewDense(r, m, brData),
		mat.NewDense(p, r, crData),
		mat.NewDense(p, m, drData),
		dt,
	)
	return red, err
}

// residualizedGain returns D - C·M⁻¹·B with M = A (continuous) or A - I
// (discrete): the static gain left when every state is residualized.
func residualizedGain(sys *System) (*mat.Dense, error) {
	n, m, p := sys.Dims()
	M := mat.DenseCopyOf(sys.A)
	if sys.IsDiscrete() {
		for i := range n {
			M.Set(i, i, M.At(i, i)-1)
		}
	}
	var lu mat.LU
	lu.Factorize(M)
	if lu.Det() == 0 {
		return nil, ErrSingularA22
	}
	var x mat.Dense
	if err := lu.SolveTo(&x, false, sys.B); err != nil {
		return nil, ErrSingularA22
	}
	g := mat.NewDense(p, m, nil)
	g.Mul(sys.C, &x)
	g.Sub(sys.D, g)
	return g, nil
}
