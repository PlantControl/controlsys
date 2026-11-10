package controlsys

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
	"slices"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// ReduceMode selects which states Reduce removes.
type ReduceMode int

const (
	// ReduceAll removes uncontrollable and unobservable states (MATLAB minreal).
	ReduceAll ReduceMode = iota
	// ReduceUncontrollable removes only uncontrollable states.
	ReduceUncontrollable
	// ReduceUnobservable removes only unobservable states.
	ReduceUnobservable
)

// ReduceOpts configures Reduce; the zero value is MATLAB minreal(sys).
type ReduceOpts struct {
	Mode ReduceMode
	// Tol is the staircase rank tolerance (MATLAB minreal tol): a singular
	// value of a step counts as rank above Tol·max(σ₁(step), ‖A‖_F), ‖A‖
	// of the full model (equalized when Equalize is set), before Reduce
	// removes any state, in both staircase stages; 0 selects n²·eps with the floor
	// raised to MATLAB ctrbf/obsvf's 10·n·‖A‖₁·eps, the rule CtrbF and
	// ObsvF use. MATLAB minreal's documented default sqrt(eps) is not
	// copied: it is a pole-zero matching tolerance of its zpk algorithm,
	// not a staircase rank tolerance, and against ‖A‖ it drops weakly
	// coupled states that carry the response (a slow plant in a PI loop
	// drops to order 1; the 10^±3 diagonally scaled Kalman sweep gives 174
	// under-counts and 277 response errors). Reduce also accepts a cut only
	// when its response matches the model's to max(√ε, Tol) (relative), so
	// a larger Tol admits larger response changes; it can also return a
	// larger model, when the looser rank floor cuts more states and that
	// cut fails the check while the default cut would pass.
	Tol float64
	// Equalize balances A, B, C before the staircase reductions. Set it
	// for badly scaled models (state scales spanning many decades): an
	// orthogonal staircase judges couplings against ‖A‖. It is off by
	// default since it is not uniformly better: on the 4000-case Kalman
	// sweep (KALMAN_SWEEP_EQUALIZE=1) with 10^±4 state scales it fixes
	// all 73 response errors and 265 over-counts but adds 202 other
	// over-counts (kept hidden unstable states) and 1 response error,
	// and without scaling it adds 54 over-counts while fixing 66.
	Equalize bool
}

// ReduceResult is a reduced realization and its state count.
type ReduceResult struct {
	Sys   *System
	Order int
}

// Reduce removes uncontrollable and/or unobservable states of sys, like
// MATLAB minreal(sys,tol); the result has the same transfer function and
// keeps D and the delays. It works in two stages. First, the states that
// the sparsity pattern of A, B and C disconnects from every input (for the
// modes that remove uncontrollable states) or from every output (for those
// that remove unobservable states) are deleted and the others kept in their
// original coordinates, as MATLAB sminreal does: the deletion is exact,
// with no rank decision and no rounding. Running it before the staircase is
// a controlsys composition, not a documented minreal step: the minreal
// documentation does not say whether minreal does it, and the sminreal
// documentation calls its result "not necessarily minimal". Second, the
// remaining model is reduced by orthogonal staircase reductions. Each input and
// output is first scaled by a power of two toward ‖A‖ (undone on the
// result), so the order does not depend on I/O scaling and a tiny but
// nonzero B or C is kept. When the staircase removes states and the
// eigenvalue magnitudes have a gap of 1/√ε or more (a stiff model, whose
// floor ε·‖A‖ can exceed the couplings of its slow states), the modes on
// either side of the gap are also reduced separately. Then the lowest
// order of the two whose frequency response matches the model's to √ε
// (relative) at the eigenvalue scales is returned, else the model
// unreduced. Without such a gap, the staircase reduction is kept only when
// its response is verified to match the model's to max(√ε, Tol) on the
// frequency axis (the unit circle or the imaginary axis): at points beside
// every eigenvalue, on a fixed grid and on a decade grid, none may refute
// it, fail to verify it where the model's own rounding bound determines its
// response, or deviate from the model by more than that bound, and the
// points that verify it within the rounding bounds of both responses must
// fix a difference of degree n + nr (the orders of the model and the cut).
// The check is sampled and evaluated in floating point: agreement at those
// points does not bound the error between them, and where the model's
// response is below its own rounding bound it cannot decide. Otherwise the
// model is returned as the first stage left it (modes clustered near z = 1,
// as from matched C2D, or a realization graded over many decades can carry
// the response through couplings below the floor). That model is a correct
// answer, so this is a nil error, unlike a failed accuracy check elsewhere
// (ErrAccuracyUnverified), and MATLAB minreal, which applies its tolerance
// unverified, can remove states Reduce keeps. Reduce does not promise a
// minimal result: a nil error with only the first stage's removals means
// no further removal could be verified, not that the model is minimal. A
// nil opts is ReduceAll with the default tolerance. A descriptor model returns
// ErrDescriptorUnsupported; an unknown Mode, a negative or non-finite Tol,
// and a nil or non-finite model return ErrInvalidArgument.
// See https://www.mathworks.com/help/control/ref/dynamicsystem.minreal.html.
func (sys *System) Reduce(opts *ReduceOpts) (*ReduceResult, error) {
	if err := requireFiniteSystem("Reduce", sys); err != nil {
		return nil, err
	}
	if opts == nil {
		opts = &ReduceOpts{}
	}
	switch opts.Mode {
	case ReduceAll, ReduceUncontrollable, ReduceUnobservable:
	default:
		return nil, fmt.Errorf("Reduce: unknown mode %d: %w", opts.Mode, ErrInvalidArgument)
	}
	if !(opts.Tol >= 0) || math.IsInf(opts.Tol, 1) {
		return nil, fmt.Errorf("Reduce: Tol is %g, want finite and non-negative: %w", opts.Tol, ErrInvalidArgument)
	}

	policy := newRealizationTransformPolicy(sys)
	if err := policy.requireStandard("Reduce"); err != nil {
		return nil, err
	}
	n, m, p := policy.n, policy.m, policy.p

	if n == 0 {
		return &ReduceResult{Sys: policy.zeroOrderCopy(), Order: 0}, nil
	}
	if sys.internalDelayCount() > 0 {
		return reduceInternalDelay(sys, policy, opts)
	}

	if m == 0 && (opts.Mode == ReduceAll || opts.Mode == ReduceUncontrollable) {
		return zeroOrderResult(sys)
	}
	if p == 0 && (opts.Mode == ReduceAll || opts.Mode == ReduceUnobservable) {
		return zeroOrderResult(sys)
	}

	keep := structuralStates(sys, opts.Mode != ReduceUnobservable, opts.Mode != ReduceUncontrollable)
	switch len(keep) {
	case n:
		return reduceStates(sys, policy, opts, true, -1)
	case 0:
		return zeroOrderResult(sys)
	}
	return reduceStructuralRemainder(sys, policy, opts, keep)
}

// reduceStructuralRemainder runs the staircase stage on the states keep
// selects, with the floor of the full model. When it removes nothing the
// remainder is returned in the original coordinates: under Equalize the
// staircase's model is the remainder rescaled, not the exact sub-block.
func reduceStructuralRemainder(sys *System, policy realizationTransformPolicy, opts *ReduceOpts, keep []int) (*ReduceResult, error) {
	a, b, c := structuralSubsystem(sys, keep)
	sub, err := policy.result(a, b, c, denseCopySafe(sys.D, policy.p, policy.m))
	if err != nil {
		return nil, err
	}
	res, err := reduceStates(sub, newRealizationTransformPolicy(sub), opts, true, reduceFloor(sys, opts))
	if err != nil || res.Order < len(keep) {
		return res, err
	}
	return &ReduceResult{Sys: sub, Order: len(keep)}, nil
}

// reduceFloor is the staircase floor reduceByStaircase takes for sys.
func reduceFloor(sys *System, opts *ReduceOpts) float64 {
	if !opts.Equalize {
		return ctrbfFloor(sys.A, opts.Tol)
	}
	n, m, p := sys.Dims()
	A, B, C := mat.DenseCopyOf(sys.A), mat.DenseCopyOf(sys.B), mat.DenseCopyOf(sys.C)
	equalize(A, B, C, n, m, p)
	return ctrbfFloor(A, opts.Tol)
}

// reduceStates is Reduce of a model without internal delays and with
// m, p > 0 for the modes it needs. scaleIO selects the I/O scaling toward
// ‖A‖, which only the raw model gets. When the staircase removes states
// and A has a spectral gap, the reduction across the gap competes with it
// (reduceAcrossSpectralGap), and a candidate counts only within √ε of the
// model's response at gapProbePoints: across a gap |λ_s| ≤ √ε·|λ_f| the
// decoupled blocks are exact to O(ε·κ), sep ≥ (1−√ε)·|λ_f| bounding the
// Sylvester solution, while the staircase's slow response errs by
// ε·‖A‖/|λ_s| ≥ √ε, so the band separates the two. Without a candidate in
// it the model is returned unreduced. Otherwise the staircase cut must
// pass frequencyAxisCutInBand at max(√ε, Tol), or the model is returned
// unreduced; a
// continuous block of reduceAcrossSpectralGap (scaleIO false) is exempt,
// since its caller judges the assembled reduction against the whole model,
// while the band would judge the block against its own response, which can
// be a rounding-level part of the model's. floor < 0 selects the staircase
// floor of sys itself.
func reduceStates(sys *System, policy realizationTransformPolicy, opts *ReduceOpts, scaleIO bool, floor float64) (*ReduceResult, error) {
	base, removed, err := reduceByStaircase(sys, policy, opts, scaleIO, floor)
	if err != nil || base.Order == policy.n {
		return base, err
	}
	if res, ok, err := reduceAcrossResolvedGap(sys, policy, opts, base); ok || err != nil {
		return res, err
	}
	if (sys.Dt == 0 && !scaleIO) || frequencyAxisCutInBand(sys, base.Sys, removed, math.Max(math.Sqrt(eps()), opts.Tol)) {
		return base, nil
	}
	return unreducedResult(sys, policy)
}

// reduceAcrossResolvedGap is the spectral-gap stage of reduceStates; ok is
// false when A has no resolved gap or the split fails, which leaves base to
// the staircase's own rules.
func reduceAcrossResolvedGap(sys *System, policy realizationTransformPolicy, opts *ReduceOpts, base *ReduceResult) (*ReduceResult, bool, error) {
	if !gapPossible(sys.A) {
		return nil, false, nil
	}
	if _, _, ok := magnitudeGap(sys.A); !ok {
		return nil, false, nil
	}
	evals, cutoff, ok := resolvedGap(sys.A)
	if !ok {
		return nil, false, nil
	}
	tol := math.Sqrt(eps())
	probe := newProbedResponse(sys, gapProbePoints(evals, cutoff, sys.Dt != 0))
	if probe.errOf(base.Sys) <= tol {
		return base, true, nil
	}
	split, ok := reduceAcrossSpectralGap(sys, policy, opts, cutoff)
	if !ok {
		return nil, false, nil
	}
	for _, c := range slices.SortedFunc(slices.Values([]*ReduceResult{base, split}), func(a, b *ReduceResult) int { return a.Order - b.Order }) {
		if c.Order < policy.n && probe.errOf(c.Sys) <= tol {
			return c, true, nil
		}
	}
	full, err := unreducedResult(sys, policy)
	return full, true, err
}

func unreducedResult(sys *System, policy realizationTransformPolicy) (*ReduceResult, error) {
	full, err := policy.result(denseCopy(sys.A), denseCopy(sys.B), denseCopy(sys.C), denseCopySafe(sys.D, policy.p, policy.m))
	if err != nil {
		return nil, err
	}
	return &ReduceResult{Sys: full, Order: policy.n}, nil
}

// reduceByStaircase is reduceStates by the orthogonal staircases alone. It
// also returns the diagonal blocks of the stage forms that hold the removed
// states, whose spectra are the removed modes.
func reduceByStaircase(sys *System, policy realizationTransformPolicy, opts *ReduceOpts, scaleIO bool, floor float64) (*ReduceResult, []*mat.Dense, error) {
	n, m, p := policy.n, policy.m, policy.p
	A := mat.DenseCopyOf(sys.A)
	B := mat.DenseCopyOf(sys.B)
	C := mat.DenseCopyOf(sys.C)

	if opts.Equalize {
		equalize(A, B, C, n, m, p)
	}

	var inExp, outExp []int
	if scaleIO {
		inExp, outExp = reduceIOExponents(A, B, C)
		ldexpIO(B, C, inExp, outExp, 1)
	}

	ncont := n
	if floor < 0 {
		floor = ctrbfFloor(A, opts.Tol)
	}
	var removed []*mat.Dense

	if opts.Mode == ReduceAll || opts.Mode == ReduceUncontrollable {
		res, err := controllabilityStaircaseFloor(A, B, C, opts.Tol, floor, false)
		if err != nil {
			return nil, nil, fmt.Errorf("Reduce: %w", err)
		}
		ncont = res.NCont
		if ncont < n {
			removed = append(removed, extractSubmatrix(res.A, ncont, n, ncont, n))
		}
		// Nothing removed: the rotation would only turn exact zeros the
		// observability stage needs into rounding noise, and can move
		// ill-conditioned eigenvalues of a nonnormal A.
		if ncont < n {
			A = res.A
			B = res.B
			C = res.C
		}
	}

	nr := ncont

	if opts.Mode == ReduceAll || opts.Mode == ReduceUnobservable {
		if ncont == 0 {
			nr = 0
		} else {
			ac := extractSubmatrix(A, 0, ncont, 0, ncont)
			bc := extractSubmatrix(B, 0, ncont, 0, m)
			cc := extractSubmatrix(C, 0, p, 0, ncont)

			slab1 := make([]float64, ncont*ncont+ncont*p+m*ncont)
			acT := transposeDenseInto(slab1[:ncont*ncont], ac)
			ccT := transposeDenseInto(slab1[ncont*ncont:ncont*ncont+ncont*p], cc)
			bcT := transposeDenseInto(slab1[ncont*ncont+ncont*p:], bc)

			dualRes, err := controllabilityStaircaseFloor(acT, ccT, bcT, opts.Tol, floor, false)
			if err != nil {
				return nil, nil, fmt.Errorf("Reduce: %w", err)
			}
			nobs := dualRes.NCont
			if nobs < ncont {
				removed = append(removed, extractSubmatrix(dualRes.A, nobs, ncont, nobs, ncont))
			}

			// nobs == ncont keeps the stage input: the rotation alone can
			// move ill-conditioned eigenvalues of a nonnormal A.
			if nobs == 0 {
				nr = 0
			} else if nobs < ncont {
				nr = nobs

				aObsDual := extractSubmatrix(dualRes.A, 0, nobs, 0, nobs)
				slab2 := make([]float64, nobs*nobs+nobs*m+p*nobs)
				aObs := transposeDenseInto(slab2[:nobs*nobs], aObsDual)
				bObs := transposeDenseInto(slab2[nobs*nobs:nobs*nobs+nobs*m], extractSubmatrix(dualRes.C, 0, m, 0, nobs))
				cObs := transposeDenseInto(slab2[nobs*nobs+nobs*m:], extractSubmatrix(dualRes.B, 0, nobs, 0, p))

				// Pertranspose for upper block Hessenberg: P*A*P, P*B, C*P
				pertransposeSquare(aObs, nobs)
				reverseRows(bObs, nobs)
				reverseCols(cObs, nobs)

				A = aObs
				B = bObs
				C = cObs
			}
		}
	}

	if nr == 0 {
		res, err := zeroOrderResult(sys)
		return res, removed, err
	}

	ar := extractSubmatrix(A, 0, nr, 0, nr)
	br := extractSubmatrix(B, 0, nr, 0, m)
	cr := extractSubmatrix(C, 0, p, 0, nr)
	ldexpIO(br, cr, inExp, outExp, -1)
	dr := denseCopySafe(sys.D, p, m)

	reduced, err := policy.result(ar, br, cr, dr)
	if err != nil {
		return nil, nil, err
	}

	return &ReduceResult{Sys: reduced, Order: nr}, removed, nil
}

// reduceInternalDelay reduces the delay-free augmented model
// [A, [B B2]; [C; C2], 0]: the delays close around its I/O channels, so any
// state it does not need is not needed by the delayed system either.
func reduceInternalDelay(sys *System, policy realizationTransformPolicy, opts *ReduceOpts) (*ReduceResult, error) {
	n, m, p := policy.n, policy.m, policy.p
	N := sys.internalDelayCount()
	B := newDense(n, m+N)
	C := newDense(p+N, n)
	setBlock(B, 0, 0, sys.B)
	setBlock(B, 0, m, sys.LFT.B2)
	setBlock(C, 0, 0, sys.C)
	setBlock(C, p, 0, sys.LFT.C2)
	aug, err := newNoCopy(denseCopy(sys.A), B, C, newDense(p+N, m+N), sys.Dt)
	if err != nil {
		return nil, err
	}
	red, err := aug.Reduce(opts)
	if err != nil {
		return nil, err
	}
	nr := red.Order
	block := func(src *mat.Dense, r0, r1, c0, c1 int) *mat.Dense {
		if r1 == r0 || c1 == c0 {
			return &mat.Dense{}
		}
		return extractSubmatrix(src, r0, r1, c0, c1)
	}
	reduced, err := policy.resultWithInternalDelay(
		denseCopy(red.Sys.A),
		nonEmptyDense(block(red.Sys.B, 0, nr, 0, m)),
		nonEmptyDense(block(red.Sys.C, 0, p, 0, nr)),
		denseCopySafe(sys.D, p, m),
		block(red.Sys.B, 0, nr, m, m+N),
		block(red.Sys.C, p, p+N, 0, nr),
	)
	if err != nil {
		return nil, err
	}
	return &ReduceResult{Sys: reduced, Order: nr}, nil
}

// reduceAcrossSpectralGap reduces the modes of A above and below cutoff, a
// gap of 1/√ε or more in |λ|, separately. The staircase judges couplings
// against ‖A‖, so stiff fast modes raise the floor ε·‖A‖ above the
// couplings of the slow modes, which are of the order of their eigenvalues,
// and slow states that carry the response are dropped. A block-diagonal
// realization with disjoint spectra is minimal iff its blocks are, and each
// block gets the floor of its own norm. The gap is 1/√ε: there the
// staircase floor ε·‖A‖ is at least √ε of the slow couplings, so its slow
// rank decisions carry relative error √ε or more, while the Sylvester
// decoupling is conditioned by sep ≈ |λ_f| and costs O(ε). With no state
// removed the original realization is kept: the Schur rotations would
// smear the stiff direction over the slow modes. ok is false when the
// modes cannot be separated.
func reduceAcrossSpectralGap(sys *System, policy realizationTransformPolicy, opts *ReduceOpts, cutoff float64) (*ReduceResult, bool) {
	n, m, p := policy.n, policy.m, policy.p
	B, C := denseCopy(sys.B), denseCopy(sys.C)
	inExp, outExp := reduceIOExponents(sys.A, B, C)
	ldexpIO(B, C, inExp, outExp, 1)
	plain, err := newNoCopy(denseCopy(sys.A), B, C, newDense(p, m), sys.Dt)
	if err != nil {
		return nil, false
	}
	fast, slow, err := decomposeModes(plain, func(ev complex128) bool { return cmplx.Abs(ev) >= cutoff })
	if err != nil {
		return nil, false
	}
	if nf, _, _ := fast.Dims(); nf == 0 || nf == n {
		return nil, false
	}
	var parts []*System
	nr := 0
	for _, part := range []*System{fast, slow} {
		red, err := reduceStates(part, newRealizationTransformPolicy(part), opts, false, -1)
		if err != nil {
			return nil, false
		}
		if red.Order > 0 {
			parts = append(parts, red.Sys)
			nr += red.Order
		}
	}
	if nr == 0 {
		res, err := zeroOrderResult(sys)
		return res, err == nil
	}
	var ar, br, cr *mat.Dense
	if nr == n {
		ar, br, cr = denseCopy(sys.A), denseCopy(sys.B), denseCopy(sys.C)
	} else {
		ar, br, cr = newDense(nr, nr), newDense(nr, m), newDense(p, nr)
		at := 0
		for _, part := range parts {
			setBlock(ar, at, at, part.A)
			setBlock(br, at, 0, part.B)
			setBlock(cr, 0, at, part.C)
			at += part.A.RawMatrix().Rows
		}
		ldexpIO(br, cr, inExp, outExp, -1)
	}
	reduced, err := policy.result(ar, br, cr, denseCopySafe(sys.D, p, m))
	if err != nil {
		return nil, false
	}
	return &ReduceResult{Sys: reduced, Order: nr}, true
}

// gapPossible is a necessary condition for a magnitudeGap, cheaper than
// the eigenvalues: for any diagonal similarity D, |λ_max|/|λ_min| ≤
// κ₂(DAD⁻¹) ≤ ‖DAD⁻¹‖_F·‖(DAD⁻¹)⁻¹‖_F, so a gap |λ_k+1| ≤ √ε·|λ_k| needs
// that bound ≥ 1/√ε. D is LAPACK's power-of-two balancing (exact), which
// keeps the bound near |λ_max|/|λ_min| for graded A. The inverse from
// partial-pivoting LU and Dgetri errs forward by about κε·‖A⁻¹‖, so the
// computed bound is at least κ/(1 + O(nε)·κ): increasing in κ, and at
// κ = 1/√ε within 1 + O(n√ε) < 2 of it, so every gap passes the test
// against 1/(2√ε). A singular A passes.
func gapPossible(A *mat.Dense) bool {
	n, _ := A.Dims()
	if n < 2 {
		return false
	}
	a := make([]float64, n*n+n+4*n)
	sc, work := a[n*n:n*n+n], a[n*n+n:]
	a = a[:n*n]
	raw := A.RawMatrix()
	for i := range n {
		copy(a[i*n:(i+1)*n], raw.Data[i*raw.Stride:i*raw.Stride+n])
	}
	impl.Dgebal(lapack.Scale, n, a, n, sc)
	norm := impl.Dlange(lapack.Frobenius, n, n, a, n, nil)
	ipiv := make([]int, n)
	if !impl.Dgetrf(n, n, a, n, ipiv) || !impl.Dgetri(n, a, n, ipiv, work, len(work)) {
		return true
	}
	k := norm * impl.Dlange(lapack.Frobenius, n, n, a, n, nil)
	return !(k < 1/(2*math.Sqrt(eps())))
}

// magnitudeGap returns the eigenvalues of A and a magnitude inside the
// topmost gap |λ_k+1| ≤ √ε·|λ_k| of their sorted magnitudes, and false
// when there is none or the eigenvalues do not converge.
func magnitudeGap(A *mat.Dense) ([]complex128, float64, bool) {
	n, _ := A.Dims()
	if n < 2 {
		return nil, 0, false
	}
	var eig mat.Eigen
	if !eig.Factorize(A, mat.EigenNone) {
		return nil, 0, false
	}
	evals := eig.Values(nil)
	cutoff, ok := topGap(evals, nil)
	return evals, cutoff, ok
}

// resolvedGap is magnitudeGap where a gap needs an eigenvalue below it
// resolved from zero (resolvedFromZero): without one, the eigenvalues
// below it are rounding, and so are the couplings that a reduction of
// their own block would keep.
func resolvedGap(A *mat.Dense) ([]complex128, float64, bool) {
	n, _ := A.Dims()
	evals, resolved, ok := resolvedFromZero(A, n)
	if !ok {
		return nil, 0, false
	}
	cutoff, ok := topGap(evals, resolved)
	return evals, cutoff, ok
}

// topGap returns a magnitude inside the topmost gap |λ_k+1| ≤ √ε·|λ_k| of
// the sorted magnitudes of evals with, for non-nil resolved, a resolved
// eigenvalue below it.
func topGap(evals []complex128, resolved []bool) (float64, bool) {
	n := len(evals)
	order := make([]int, n)
	for i := range order {
		order[i] = i
		if w := cmplx.Abs(evals[i]); math.IsInf(w, 0) || math.IsNaN(w) {
			return 0, false
		}
	}
	slices.SortFunc(order, func(i, j int) int { return cmp.Compare(cmplx.Abs(evals[i]), cmplx.Abs(evals[j])) })
	below := make([]bool, n)
	for k, i := range order {
		below[k] = resolved == nil || resolved[i] || (k > 0 && below[k-1])
	}
	gap := math.Sqrt(eps())
	for k := n - 1; k > 0 && below[k-1]; k-- {
		lo, hi := cmplx.Abs(evals[order[k-1]]), cmplx.Abs(evals[order[k]])
		if lo <= gap*hi {
			return hi * math.Sqrt(gap), true
		}
	}
	return 0, false
}

// resolvedFromZero returns the eigenvalues of the n×n A with, for each,
// whether componentwise rounding of A moves it by less than its modulus:
// the first-order bound ε·|y|ᵀ|A||x| / |yᴴx| for right and left
// eigenvectors x, y, invariant under diagonal scaling of A. A rounded zero
// eigenvalue of a dense realization has a bound of the order of its own
// modulus and is not resolved; a slow pole of a stiff triangular one is.
// ok is false where the factorization fails.
func resolvedFromZero(A *mat.Dense, n int) ([]complex128, []bool, bool) {
	evals, yax, yx, ok := eigenvalueRoundingTerms(A, n)
	if !ok {
		return nil, nil, false
	}
	resolved := make([]bool, len(evals))
	for k, ev := range evals {
		resolved[k] = ev != 0 && eps()*yax[k] < cmplx.Abs(ev)*yx[k]
	}
	return evals, resolved, true
}

// gapProbePoints returns points of the frequency axis: s = jω at the
// smallest and largest nonzero |λ| on each side of cutoff for continuous
// models, z = e^{jθ} over [0, 3] for discrete ones. Off the axis, or at
// ω ~ ε·‖A‖, the states at the rounding level ε·‖A‖ that the staircase
// rightly drops dominate the response near their poles.
func gapProbePoints(evals []complex128, cutoff float64, discrete bool) []complex128 {
	if discrete {
		var pts []complex128
		for _, th := range []float64{0, 0.01, 0.1, 0.5, 1, 2, 3} {
			pts = append(pts, cmplx.Rect(1, th))
		}
		return pts
	}
	lo := [2]float64{math.Inf(1), math.Inf(1)}
	hi := [2]float64{}
	for _, ev := range evals {
		w := cmplx.Abs(ev)
		if w == 0 {
			continue
		}
		side := 0
		if w >= cutoff {
			side = 1
		}
		lo[side] = math.Min(lo[side], w)
		hi[side] = math.Max(hi[side], w)
	}
	var pts []complex128
	for side := range 2 {
		for _, w := range []float64{lo[side], hi[side]} {
			if w > 0 && !math.IsInf(w, 0) && (len(pts) == 0 || imag(pts[len(pts)-1]) != w) {
				pts = append(pts, complex(0, w))
			}
		}
	}
	return pts
}

// probedResponse is the response of a model at probe points, evaluated
// once for every candidate reduction compared with it; nil where the
// evaluation fails.
type probedResponse struct {
	sys *System
	pts []complex128
	g   [][][]complex128
}

func newProbedResponse(sys *System, pts []complex128) probedResponse {
	g := make([][][]complex128, len(pts))
	for k, s := range pts {
		if r, err := sys.EvalFr(s); err == nil {
			g[k] = r
		}
	}
	return probedResponse{sys: sys, pts: pts, g: g}
}

// errOf is the largest relative difference of the strictly proper parts
// of the model and red over the points; points where either evaluation
// fails are skipped.
func (pr probedResponse) errOf(red *System) float64 {
	worst := 0.0
	for k, s := range pr.pts {
		g := pr.g[k]
		if g == nil {
			continue
		}
		r, err := red.EvalFr(s)
		if err != nil {
			continue
		}
		scale, diff := 0.0, 0.0
		for i := range g {
			for j := range g[i] {
				d := complex(pr.sys.D.At(i, j), 0)
				scale = math.Max(scale, math.Max(cmplx.Abs(g[i][j]-d), cmplx.Abs(r[i][j]-d)))
				diff = math.Max(diff, cmplx.Abs(r[i][j]-g[i][j]))
			}
		}
		if math.IsNaN(diff) || math.IsNaN(scale) {
			return math.Inf(1)
		}
		if scale > 0 {
			worst = math.Max(worst, diff/scale)
		}
	}
	return worst
}

// MinimalRealization is Reduce(nil), MATLAB minreal(sys) with the default
// tolerance; use Reduce to set Tol.
func (sys *System) MinimalRealization() (*ReduceResult, error) {
	return sys.Reduce(nil)
}

// reduceIOExponents returns power-of-two exponents for each column of B
// and row of C that bring its max-abs entry to the binary exponent of
// ‖A‖_F (of 1 when A = 0). Input and output scaling leave the reachable
// and unobservable subspaces unchanged, so the rank decisions become
// invariant to it and a tiny but nonzero B or C is not dropped against ‖A‖.
// Only the raw model is scaled: a rotated remnant's small column is noise.
func reduceIOExponents(A, B, C *mat.Dense) (inExp, outExp []int) {
	_, m := B.Dims()
	p, _ := C.Dims()
	target := frobNorm(A)
	if target == 0 {
		target = 1
	}
	if math.IsInf(target, 0) || math.IsNaN(target) {
		return nil, nil
	}
	_, te := math.Frexp(target)
	exps := make([]int, m+p)
	inExp, outExp = exps[:m], exps[m:]
	shift := func(mx float64) int {
		if mx == 0 {
			return 0
		}
		_, e := math.Frexp(mx)
		return min(te, 1000) - e
	}
	bRaw, cRaw := B.RawMatrix(), C.RawMatrix()
	for j := range m {
		mx := 0.0
		for i := range bRaw.Rows {
			mx = math.Max(mx, math.Abs(bRaw.Data[i*bRaw.Stride+j]))
		}
		inExp[j] = shift(mx)
	}
	for i := range p {
		mx := 0.0
		for _, v := range cRaw.Data[i*cRaw.Stride : i*cRaw.Stride+cRaw.Cols] {
			mx = math.Max(mx, math.Abs(v))
		}
		outExp[i] = shift(mx)
	}
	return inExp, outExp
}

// ldexpIO scales column j of B by 2^(sign·inExp[j]) and row i of C by
// 2^(sign·outExp[i]) in place.
func ldexpIO(B, C *mat.Dense, inExp, outExp []int, sign int) {
	if inExp == nil {
		return
	}
	bRaw, cRaw := B.RawMatrix(), C.RawMatrix()
	for i := range bRaw.Rows {
		row := bRaw.Data[i*bRaw.Stride : i*bRaw.Stride+bRaw.Cols]
		for j, e := range inExp {
			if e != 0 {
				row[j] = math.Ldexp(row[j], sign*e)
			}
		}
	}
	for i, e := range outExp {
		if e == 0 {
			continue
		}
		row := cRaw.Data[i*cRaw.Stride : i*cRaw.Stride+cRaw.Cols]
		for k := range row {
			row[k] = math.Ldexp(row[k], sign*e)
		}
	}
}

func zeroOrderResult(sys *System) (*ReduceResult, error) {
	g, err := newRealizationTransformPolicy(sys).zeroOrderOriginalFeedthrough()
	if err != nil {
		return nil, fmt.Errorf("Reduce: %w", err)
	}
	return &ReduceResult{Sys: g, Order: 0}, nil
}

func extractSubmatrix(m *mat.Dense, r0, r1, c0, c1 int) *mat.Dense {
	return extractBlock(m, r0, c0, r1-r0, c1-c0)
}

func pertransposeSquare(a *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := a.RawMatrix()
	for i := 0; i < n/2; i++ {
		j := n - 1 - i
		ri := raw.Data[i*raw.Stride:]
		rj := raw.Data[j*raw.Stride:]
		for k := range n {
			ri[k], rj[k] = rj[k], ri[k]
		}
	}
	for k := range n {
		rk := raw.Data[k*raw.Stride:]
		for i := 0; i < n/2; i++ {
			j := n - 1 - i
			rk[i], rk[j] = rk[j], rk[i]
		}
	}
}

func reverseRows(m *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := m.RawMatrix()
	cols := raw.Cols
	for i := 0; i < n/2; i++ {
		j := n - 1 - i
		ri := raw.Data[i*raw.Stride : i*raw.Stride+cols]
		rj := raw.Data[j*raw.Stride : j*raw.Stride+cols]
		for k := range ri {
			ri[k], rj[k] = rj[k], ri[k]
		}
	}
}

func reverseCols(m *mat.Dense, n int) {
	if n <= 1 {
		return
	}
	raw := m.RawMatrix()
	for k := 0; k < raw.Rows; k++ {
		rk := raw.Data[k*raw.Stride:]
		for i := 0; i < n/2; i++ {
			j := n - 1 - i
			rk[i], rk[j] = rk[j], rk[i]
		}
	}
}

// equalize balances A by a diagonal similarity D such that A → D^{-1} A D,
// B → D^{-1} B, C → C D, following LAPACK DGEBAL with the SLICOT TB01ID
// extension that includes B rows and C columns in the row/column 1-norms.
// The scaling radix is DGEBAL's 2, not TB01ID's 10: the loop rescales the
// entries in place on every pass, so radix 10 rounds each entry once per
// pass, and a model whose response is ill-conditioned in its entries
// (poles clustered near z = 1, as from matched C2D) then changes its
// response far beyond √ε with no state removed. Powers of two are exact,
// like the integer powers of two of MATLAB balance.
func equalize(A, B, C *mat.Dense, n, m, p int) {
	const sclfac = 2.0
	const factor = 0.95
	sfmin1 := math.SmallestNonzeroFloat64 / eps()
	sfmax1 := 1.0 / sfmin1
	sfmin2 := sfmin1 * sclfac
	sfmax2 := 1.0 / sfmin2

	scale := make([]float64, n)
	for i := range scale {
		scale[i] = 1.0
	}

	aRaw := A.RawMatrix()
	bRaw := B.RawMatrix()
	cRaw := C.RawMatrix()

	for {
		noconv := false
		for i := range n {
			c := 0.0
			r := 0.0
			ca := 0.0
			ra := 0.0

			aRow := aRaw.Data[i*aRaw.Stride:]
			for j := range n {
				if j == i {
					continue
				}
				v := math.Abs(aRaw.Data[j*aRaw.Stride+i])
				c += v
				if v > ca {
					ca = v
				}
				v = math.Abs(aRow[j])
				r += v
				if v > ra {
					ra = v
				}
			}
			bRow := bRaw.Data[i*bRaw.Stride:]
			for j := range m {
				v := math.Abs(bRow[j])
				r += v
				if v > ra {
					ra = v
				}
			}
			for j := range p {
				v := math.Abs(cRaw.Data[j*cRaw.Stride+i])
				c += v
				if v > ca {
					ca = v
				}
			}

			if c == 0 || r == 0 {
				continue
			}

			g := r / sclfac
			f := 1.0
			s := c + r

			for c < g && max(f, c, ca) < sfmax2 && min(r, g, ra) > sfmin2 {
				f *= sclfac
				c *= sclfac
				ca *= sclfac
				g /= sclfac
				r /= sclfac
				ra /= sclfac
			}

			g = c / sclfac
			for g >= r && max(r, ra) < sfmax2 && min(f, c, g, ca) > sfmin2 {
				f /= sclfac
				c /= sclfac
				g /= sclfac
				ca /= sclfac
				r *= sclfac
				ra *= sclfac
			}

			if c+r >= factor*s {
				continue
			}
			if f < 1 && scale[i] < 1 && f*scale[i] <= sfmin1 {
				continue
			}
			if f > 1 && scale[i] > 1 && scale[i] >= sfmax1/f {
				continue
			}

			scale[i] *= f
			noconv = true

			fi := 1.0 / f
			for j := range n {
				aRow[j] *= fi
			}
			for j := range m {
				bRow[j] *= fi
			}
			for j := range n {
				aRaw.Data[j*aRaw.Stride+i] *= f
			}
			for j := range p {
				cRaw.Data[j*cRaw.Stride+i] *= f
			}
		}
		if !noconv {
			break
		}
	}
}
