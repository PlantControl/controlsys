package controlsys

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// RootLocusResult holds the closed-loop pole trajectories of a SISO loop
// under negative feedback u = -K·y, together with the open-loop locus
// construction data.
type RootLocusResult struct {
	// Gains are the feedback gains K in ascending order.
	Gains []float64
	// Branches[b][k] is the b-th closed-loop pole at Gains[k]; branches are
	// ordered for continuity between consecutive gains.
	Branches [][]complex128
	// Breakaway lists the real break-away/break-in points of the K ≥ 0
	// locus in ascending order, each once; empty when the locus has none
	// (see RootLocus for how they are computed).
	Breakaway []complex128
	// AsymptoteAngles are the angles in radians of the asymptotes along
	// which branches leave to infinity; empty when the number of poles
	// equals the number of zeros.
	AsymptoteAngles []float64
	// DepartureAngles[i] is the departure angle in radians from the i-th
	// open-loop pole.
	DepartureAngles []float64
	// ArrivalAngles[k] is the arrival angle in radians at the k-th open-loop
	// zero.
	ArrivalAngles []float64

	centroid float64
}

// AsymptoteCentroid returns the real-axis point where the asymptotes meet.
// ok is false when the locus has no asymptotes.
func (r *RootLocusResult) AsymptoteCentroid() (centroid float64, ok bool) {
	if len(r.AsymptoteAngles) == 0 {
		return 0, false
	}
	return r.centroid, true
}

// RootLocus computes the closed-loop poles of the SISO model sys under
// negative feedback u = -K·y for each gain K, as MATLAB rlocus(sys, k). The
// closed-loop state matrix is A - B·K/(1+K·D)·C. A nil gains slice selects a
// default logarithmic grid from 1e-6 to 1e6 plus K = 0; an explicitly empty
// slice or a non-finite gain is ErrInvalidArgument. Descriptor models are
// converted with ToExplicit (singular E returns its error). Discrete delays
// are absorbed as extra states; continuous delays return ErrDelayUnsupported,
// as MATLAB requires a Pade approximation first. A model with no states
// returns ErrDimensionMismatch, and a gain with 1+K·D = 0 returns
// ErrAlgebraicLoop.
//
// A loop whose high-frequency gain (D, or the first Markov parameter
// C·A^(r−1)·B resolved by the coefficients) is negative follows the 0°
// angle condition: its asymptote, departure and arrival angles and its
// real-axis segments are those of the K > 0 locus of that sign. Zeros
// beyond the n − r finite ones of relative degree r are rounding noise
// (MATLAB reports them as infinite) and are left out of the asymptotes and
// angles.
//
// Breakaway follows python-control's root_locus break points: the distinct
// real roots of N'D − ND' with K = −D/N ≥ 0, where K = 0 is a multiple pole
// and K = ∞ a multiple zero (break-in). They are computed exactly for the
// state-space data as given: A, B, C, D are exact binary rationals, so N
// and D (common factors cancelled) are exact integer polynomials, the
// numerator of C(sI−A)⁻¹B trimmed to the relative degree of the first
// Markov parameter the rule above resolves, decided on the exact hₖ so
// that no platform rounding enters, and kept whole when none is resolved
// (the data cannot then tell a rounding-level coefficient from an exact
// one, so a noise zero adds its far break point); and the roots of
// N'D − ND', their multiplicity and the sign of K are decided in exact
// arithmetic, then rounded to float64. No multiplicity is inferred: a
// realization whose rounding split a multiple pole or zero (a rotated or
// otherwise non-triangular similarity of a repeated root) reports the
// stationary points of the split system, which may differ from the
// textbook ones; build the model from its transfer function to keep the
// multiple root exact. MATLAB rlocus reports no break points. Loops with
// more than 12 states (delay states excluded), or whose entries of A and
// B·C span more than 4800/n − 64 binary orders of magnitude, use the
// eigenvalue-based approximation of computeBreakaway instead.
//
// See https://www.mathworks.com/help/control/ref/dynamicsystem.rlocus.html.
func RootLocus(sys *System, gains []float64) (*RootLocusResult, error) {
	const op = "RootLocus"
	if err := requireFiniteSystem(op, sys); err != nil {
		return nil, err
	}
	if _, err := newSISOLoopModel(sys, op); err != nil {
		return nil, err
	}
	if gains != nil && len(gains) == 0 {
		return nil, fmt.Errorf("%s: gains is empty: %w", op, ErrInvalidArgument)
	}
	if err := requireFinite(op, "gains", gains...); err != nil {
		return nil, err
	}
	var undelayed *System
	delay := 0
	if sys.HasDelay() {
		if sys.IsContinuous() {
			return nil, fmt.Errorf("%s: continuous model has time delays; use Pade first: %w", op, ErrDelayUnsupported)
		}
		if n0, _, _ := sys.Dims(); n0 > 0 && sys.internalDelayCount() == 0 {
			delay = int(effectiveIODelayData(sys, 1, 1, true)[0])
			undelayed = sys.Copy()
			undelayed.Delay, undelayed.InputDelay, undelayed.OutputDelay = nil, nil, nil
		}
		abs, err := sys.AbsorbDelay()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		sys = abs
	}
	if sys.IsDescriptor() {
		exp, err := sys.ToExplicit()
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		sys = exp
	}
	n, _, _ := sys.Dims()
	if n == 0 {
		return nil, fmt.Errorf("%s: system has no states: %w", op, ErrDimensionMismatch)
	}
	if undelayed != nil && undelayed.IsDescriptor() {
		exp, err := undelayed.ToExplicit()
		if err != nil {
			exp = nil
		}
		undelayed = exp
	}
	d := sys.D.At(0, 0)

	poles, err := sys.Poles()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}
	zeros, err := sys.Zeros()
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
	}

	if gains == nil {
		gains = defaultGains()
	} else {
		g := make([]float64, len(gains))
		copy(g, gains)
		gains = g
		sort.Float64s(gains)
	}

	bRaw := sys.B.RawMatrix()
	cRaw := sys.C.RawMatrix()
	bc := make([]float64, n*n)
	for i := range n {
		for j := range n {
			bc[i*n+j] = bRaw.Data[i*bRaw.Stride] * cRaw.Data[j]
		}
	}

	aRaw := sys.A.RawMatrix()
	aFlat := make([]float64, n*n)
	for i := range n {
		copy(aFlat[i*n:i*n+n], aRaw.Data[i*aRaw.Stride:i*aRaw.Stride+n])
	}

	allEigs := make([][]complex128, len(gains))
	work := make([]float64, n*n)

	for gi, K := range gains {
		den := 1 + K*d
		if den == 0 {
			return nil, fmt.Errorf("%s: 1+K·D = 0 at K=%g: %w", op, K, ErrAlgebraicLoop)
		}
		kEff := K / den
		for i := 0; i < n*n; i++ {
			work[i] = aFlat[i] - kEff*bc[i]
		}
		M := mat.NewDense(n, n, work)
		var eig mat.Eigen
		if !eig.Factorize(M, mat.EigenNone) {
			return nil, fmt.Errorf("%s: eigenvalues of A-BKC at K=%g: %w", op, K, ErrSchurFailed)
		}
		vals := eig.Values(nil)
		allEigs[gi] = vals
	}

	branches := makeBranches(allEigs, n, len(gains))

	bCol := make([]float64, n)
	for i := range n {
		bCol[i] = bRaw.Data[i*bRaw.Stride]
	}
	lp := rlocusLoop{n: n, a: aFlat, b: bCol, c: cRaw.Data[:n], d: d}
	lp.relDeg, lp.negGain = lp.highFrequencyGain()
	lp.spRelDeg = lp.relDeg
	if d != 0 {
		lp.spRelDeg, _ = lp.markovRelDegree()
	}
	if d == 0 {
		if lp.relDeg <= n {
			zeros = lp.finiteZeros(zeros, poles, lp.relDeg)
		}
		if r := n - len(zeros); r >= 1 && r != lp.relDeg {
			// Zeros() fixes the asymptote count; the angles, the 0° rule and
			// the eig path's numerator trim follow the Markov parameter of
			// that degree.
			lp.relDeg, lp.negGain = r, lp.markovSign(r)
			lp.spRelDeg = r
		}
	}
	breakaway, ok := systemExactBreakaway(sys, undelayed, delay)
	if !ok {
		breakaway, err = computeBreakaway(sys, undelayed, delay, &lp, poles, zeros)
		if err != nil {
			return nil, fmt.Errorf("%s: %w", op, err)
		}
		slices.SortStableFunc(breakaway, func(x, y complex128) int { return cmp.Compare(real(x), real(y)) })
	}
	phase0 := math.Pi
	if lp.negGain {
		phase0 = 0
	}

	nPoles := len(poles)
	nZeros := len(zeros)

	var asymAngles []float64
	var asymCentroid float64
	diff := nPoles - nZeros
	if diff > 0 {
		asymAngles = make([]float64, diff)
		for k := range diff {
			asymAngles[k] = (float64(2*k)*math.Pi + phase0) / float64(diff)
		}
		sumP := 0.0
		for _, pole := range poles {
			sumP += real(pole)
		}
		sumZ := 0.0
		for _, z := range zeros {
			sumZ += real(z)
		}
		asymCentroid = (sumP - sumZ) / float64(diff)
	}

	departAngles := make([]float64, nPoles)
	for i, pi := range poles {
		sumPoles := 0.0
		for j, pj := range poles {
			if j != i {
				sumPoles += cmplx.Phase(pi - pj)
			}
		}
		sumZeros := 0.0
		for _, zk := range zeros {
			sumZeros += cmplx.Phase(pi - zk)
		}
		departAngles[i] = phase0 - sumPoles + sumZeros
	}

	arrivalAngles := make([]float64, nZeros)
	for k, zk := range zeros {
		sumZeros := 0.0
		for j, zj := range zeros {
			if j != k {
				sumZeros += cmplx.Phase(zk - zj)
			}
		}
		sumPoles := 0.0
		for _, pi := range poles {
			sumPoles += cmplx.Phase(zk - pi)
		}
		arrivalAngles[k] = phase0 - sumZeros + sumPoles
	}

	return &RootLocusResult{
		Gains:           gains,
		Branches:        branches,
		Breakaway:       breakaway,
		AsymptoteAngles: asymAngles,
		DepartureAngles: departAngles,
		ArrivalAngles:   arrivalAngles,
		centroid:        asymCentroid,
	}, nil
}

func defaultGains() []float64 {
	n := 200
	gains := make([]float64, n+1)
	gains[0] = 0
	for i := range n {
		exp := -6.0 + 12.0*float64(i)/float64(n-1)
		gains[i+1] = math.Pow(10, exp)
	}
	sort.Float64s(gains)
	return gains
}

func makeBranches(allEigs [][]complex128, nStates, nGains int) [][]complex128 {
	if nStates == 0 || nGains == 0 {
		return nil
	}

	ordered := make([][]complex128, nGains)
	ordered[0] = make([]complex128, nStates)
	copy(ordered[0], allEigs[0])

	used := make([]bool, nStates)
	for gi := 1; gi < nGains; gi++ {
		cur := allEigs[gi]
		prev := ordered[gi-1]
		perm := make([]int, nStates)
		for i := range used {
			used[i] = false
		}

		for i := range nStates {
			bestJ := -1
			bestDist := math.Inf(1)
			for j := range nStates {
				if used[j] {
					continue
				}
				d := cmplx.Abs(prev[i] - cur[j])
				if d < bestDist {
					bestDist = d
					bestJ = j
				}
			}
			perm[i] = bestJ
			used[bestJ] = true
		}
		row := make([]complex128, nStates)
		for i := range nStates {
			row[i] = cur[perm[i]]
		}
		ordered[gi] = row
	}

	branches := make([][]complex128, nStates)
	for b := range nStates {
		branches[b] = make([]complex128, nGains)
		for gi := range nGains {
			branches[b][gi] = ordered[gi][b]
		}
	}
	return branches
}

// computeBreakaway returns the real roots of N'D − ND' on the K > 0 locus
// from the eigenvalues and the converted transfer function in float64: the
// fallback of exactBreakaway beyond its size gates.
// The numerator is trimmed to the loop's relative degree, so rounding-noise
// leading coefficients add no far stationary points, and N, D are scaled
// by powers of two so N'D − ND' cannot overflow. An absorbed delay of L
// samples is kept exact as D = z^L·D0 (N and D0 from the undelayed loop,
// whose conversion carries no noise from the delay states, else the
// full-order D with its L noise trailing coefficients dropped), so z = 0 is
// reported once, not rooted from an L-fold cluster; if rooting that form
// fails, the full D is rooted as is.
// A real pole or zero of net multiplicity m ≥ 2 (K = 0 or ∞, multipleRoots
// less cancelling roots of the other kind) is a break point under either
// angle condition, reported once in place of the m − 1 nearest roots of
// N'D − ND' it splits into; no absolute cut-off decides it, so slow or
// close resolved poles stay distinct. A resolved complex pair that only the
// converted polynomial merges (rootCluster.resolved) is no multiple root:
// the real root of N'D − ND' it claims is a stationary point. A root of
// N'D − ND' near a multiple pole or zero (multipleRootNear) is that root
// only within the cluster's own split radius. A root of N'D − ND' is real when its
// imaginary part is within 1e-6 of its modulus. A multiple root of N'D − ND'
// itself (multipleRootNear on it; a multiple zero of G − D, or of G split
// past its cluster test) is one stationary point, judged and reported once
// at its centre.
// Any other root is on the locus when K = −1/G(s) > 0 and G'(s) ≈ 0, both
// from the state space, so a sign rule over rounding-noise zeros or a
// conversion error in N/D cannot misplace it; where the sign of G is not
// resolved the real-axis rule (0° for a negative high-frequency gain)
// decides. Every reported stationary point passes that K > 0 test last; a
// K = 0 or ∞ point has no sign test. G ≡ 0 (an all-zero converted N) has
// no locus and no break points.
func computeBreakaway(sys, undelayed *System, delay int, lp *rlocusLoop, poles, zeros []complex128) ([]complex128, error) {
	num, den0, numFull, denFull, delay, err := breakawayTF(sys, undelayed, delay, lp.n)
	if err != nil {
		return nil, err
	}
	sNum, sDen0, sNumFull, sDenFull, sDelay := num, den0, numFull, denFull, delay
	if lp.d != 0 {
		// G' = (G − D)', so the stationary points come from the strictly
		// proper part, whose noise leading coefficients the Markov relative
		// degree trims; D·den in N would hide them.
		sp, spU := sys.Copy(), undelayed
		sp.D.Set(0, 0, 0)
		if spU != nil {
			spU = spU.Copy()
			spU.D.Set(0, 0, 0)
		}
		if a, b, c, e, l, serr := breakawayTF(sp, spU, delay, lp.n); serr == nil {
			sNum, sDen0, sNumFull, sDenFull, sDelay = a, b, c, e, l
		}
	}
	roots, stat, statBound, err := stationaryCandidates(sNum, sDen0, sDelay, lp.spRelDeg, lp.n)
	if err != nil && sDelay > 0 {
		roots, stat, statBound, err = stationaryCandidates(sNumFull, sDenFull, 0, lp.spRelDeg, lp.n)
		num, den0, delay = numFull, denFull, 0
	}
	if err != nil {
		return nil, err
	}
	num, den := trimmedTF(num, den0, delay, lp.relDeg, lp.n)
	if !slices.ContainsFunc(num, func(c float64) bool { return c != 0 }) {
		return []complex128{}, nil
	}

	sigmaBal, sigma := balancedNorm(lp.a, lp.n), 0.0
	for _, r := range slices.Concat(poles, zeros) {
		sigma = math.Max(sigma, cmplx.Abs(r))
	}
	sigma = math.Min(sigma, sigmaBal)
	numRoots, denRoots := polyRootsOrNil(num), polyRootsOrNil(den)
	clusters := netClusters(lp.zeroClusters(sys, zeros, num, sigma, sigmaBal), poles, denRoots)
	if ec, ok := eigCondition(lp.a, nil, lp.n); ok {
		clusters = append(clusters, netClusters(multipleRoots(ec.vals, ec.kappa, ec.pairCondition, den, lp.n, sigma, sigmaBal, sigmaBal), zeros, numRoots)...)
	}
	bl := lp.balanced()
	type point struct {
		s float64
		// multiple marks a K = 0 or ∞ point (a multiple pole or zero),
		// where the sign of G is no locus test.
		multiple bool
	}
	var pts []point
	add := func(p point) {
		if !slices.ContainsFunc(pts, func(q point) bool { return q.s == p.s }) {
			pts = append(pts, p)
		}
	}
	// A multiple root of multiplicity m is an (m−1)-fold root of N'D − ND',
	// split by rounding: the m−1 nearest roots are that break point, so it
	// is reported once: at the centre of exactly equal roots, else at the
	// nearest root when that is real. A point reached twice (a pole and a
	// zero cluster snapping to the same root, a near-real conjugate pair) is
	// listed once.
	used := make([]bool, len(roots))
	claim := func(c float64, k int) (first int) {
		first = -1
		for range k {
			best := -1
			for i, r := range roots {
				if !used[i] && (best < 0 || cmplx.Abs(r-complex(c, 0)) < cmplx.Abs(roots[best]-complex(c, 0))) {
					best = i
				}
			}
			if best < 0 {
				break
			}
			used[best] = true
			if first < 0 {
				first = best
			}
		}
		return first
	}
	isReal := func(r complex128) bool { return math.Abs(imag(r)) <= 1e-6*cmplx.Abs(r) }
	onLocusAt := func(s float64) bool {
		if g, _, ok := bl.eval(s); ok {
			return g < 0
		}
		return isOnRealAxisSegment(s, poles, zeros) != lp.negGain
	}
	for _, c := range clusters {
		if c.m >= 2 {
			at := complex(c.centre, 0)
			i := claim(c.centre, c.m-1)
			if i >= 0 && !c.exact && isReal(roots[i]) {
				at = complex(real(roots[i]), 0)
			}
			if c.resolved && (i < 0 || !isReal(roots[i])) {
				continue
			}
			add(point{real(at), !c.resolved})
		}
	}
	for i, r := range roots {
		if used[i] || !isReal(r) || slices.Contains(roots[:i], r) || slices.Contains(roots[:i], cmplx.Conj(r)) {
			continue
		}
		s := real(r)
		c, m := multipleRootNear(den, nil, s, lp.n, poles)
		if m == 0 {
			c, m = multipleRootNear(num, nil, s, lp.n, zeros)
		}
		multiple := m > 0
		onLocus := multiple
		if !onLocus && s != 0 {
			if cs, ms := multipleRootNear(stat, statBound, s, lp.n, roots); ms > 0 {
				claim(cs, ms)
				s = cs
			}
		}
		if onLocus {
			used[i] = true
			claim(c, m-2)
		} else if g, dg, ok := bl.eval(s); ok {
			onLocus = g < 0 && isStationary(s, g, dg, poles, zeros)
		} else {
			onLocus = isOnRealAxisSegment(s, poles, zeros) != lp.negGain
		}
		if onLocus {
			add(point{s, multiple})
		}
	}
	var result []complex128
	for _, p := range pts {
		if p.multiple || onLocusAt(p.s) {
			result = append(result, complex(p.s, 0))
		}
	}
	return result, nil
}

// breakawayTF converts sys to N/(z^delay·D0): D0 drops the L noise
// trailing coefficients of a full-order D, or comes from the undelayed loop
// when the conversion reduced the order; delay is returned 0 when neither
// applies. numFull, denFull are the converted N, D as is.
func breakawayTF(sys, undelayed *System, delay, n int) (num, den0, numFull, denFull Poly, outDelay int, err error) {
	tfr, err := sys.rationalTransferFunction(nil)
	if err != nil {
		return nil, nil, nil, nil, 0, err
	}
	numFull = Poly(tfr.TF.Num[0][0])
	denFull = Poly(tfr.TF.Den[0])
	num, den0 = numFull, denFull
	var u *TransferFuncResult
	if delay > 0 && undelayed != nil && slices.ContainsFunc(num, func(c float64) bool { return c != 0 }) {
		if ur, uerr := undelayed.rationalTransferFunction(nil); uerr == nil {
			u = ur
		}
	}
	switch {
	case delay <= 0:
		delay = 0
	case u != nil:
		num, den0 = u.TF.Num[0][0], u.TF.Den[0]
	case len(den0) == n+1:
		den0 = den0[:len(den0)-delay]
	default:
		delay = 0
	}
	return num, den0, numFull, denFull, delay, nil
}

// trimmedTF trims num to relative degree relDeg ≤ n of D = z^delay·D0 and
// unit-scales num and D (returned).
func trimmedTF(num, den0 Poly, delay, relDeg, n int) (Poly, Poly) {
	if keep := len(den0) + delay - relDeg; relDeg <= n && keep > 0 && keep < len(num) {
		num = num[len(num)-keep:]
	}
	if len(num) == 0 || len(den0) == 0 {
		return nil, nil
	}
	return unitScalePoly(num), slices.Concat(unitScalePoly(den0), make(Poly, delay))
}

// stationaryCandidates returns the roots of N'D − ND' for D = z^delay·D0,
// with exact common roots at 0 of N and D cancelled, N trimmed to relative
// degree relDeg and N, D unit-scaled; exact roots at 0 (z^(L−1) and zero
// trailing coefficients) are listed with their multiplicity; poly is
// N'D − ND' without them and bound its coefficientwise change per unit
// relative change of N's and D's coefficients, 2·(|N'|·|D| + |N|·|D'|)
// (+ L·|N|·|D0| terms). Roots is nil when the converted N, D or
// N'D − ND' are empty, all zero or not finite.
func stationaryCandidates(num, den0 Poly, delay, relDeg, n int) (roots []complex128, poly, bound Poly, err error) {
	for len(num) > 1 && num[len(num)-1] == 0 && (delay > 0 || len(den0) > 1 && den0[len(den0)-1] == 0) {
		if delay > 0 {
			delay--
		} else {
			den0 = den0[:len(den0)-1]
		}
		num = num[:len(num)-1]
	}
	num, den := trimmedTF(num, den0, delay, relDeg, n)
	if len(num) == 0 {
		return nil, nil, nil, nil
	}
	den0 = den[:len(den)-delay]
	notFinite := func(c float64) bool { return !isFinite(c) }
	if slices.ContainsFunc(num, notFinite) || slices.ContainsFunc(den0, notFinite) {
		return []complex128{}, nil, nil, nil
	}

	// D = z^L·D0 is exact, so N'D − ND' = z^(L−1)·(z·(N'D0 − ND0') − L·N·D0).
	poly = num.Derivative().Mul(den0).Sub(num.Mul(den0.Derivative()))
	an, ad := absPoly(num), absPoly(den0)
	bound = an.Derivative().Mul(ad).Add(an.Mul(ad.Derivative()))
	if delay > 0 {
		poly = slices.Concat(poly, Poly{0}).Sub(num.Mul(den0).Scale(float64(delay)))
		bound = slices.Concat(bound, Poly{0}).Add(an.Mul(ad).Scale(float64(delay)))
	}
	bound = bound.Scale(2)
	if !slices.ContainsFunc(poly, func(c float64) bool { return c != 0 }) || slices.ContainsFunc(poly, notFinite) {
		return []complex128{}, nil, nil, nil
	}
	atZero := max(delay-1, 0)
	for len(poly) > 1 && poly[len(poly)-1] == 0 {
		poly, bound = poly[:len(poly)-1], bound[:len(bound)-1]
		atZero++
	}
	if roots, err = poly.Roots(); err != nil {
		return nil, nil, nil, err
	}
	return append(roots, make([]complex128, atZero)...), poly, bound, nil
}

// rlocusLoop is the explicit SISO loop C(sI−A)⁻¹B + D, A row-major n×n.
type rlocusLoop struct {
	n       int
	a, b, c []float64
	d       float64
	relDeg  int
	negGain bool
	// spRelDeg is the relative degree of the strictly proper part G − D.
	spRelDeg int
}

// highFrequencyGain returns the relative degree r (0 with a feedthrough,
// else markovRelDegree) and whether lim s^r·G(s) < 0. The K > 0 locus of
// such a loop obeys the 0° rather than the 180° angle condition.
func (lp *rlocusLoop) highFrequencyGain() (int, bool) {
	if lp.d != 0 {
		return 0, lp.d < 0
	}
	return lp.markovRelDegree()
}

// markovRelDegree returns the relative degree of C(sI−A)⁻¹B, the index of
// the first Markov parameter hₖ = C·A^(k−1)·B resolved by the given
// coefficients (markovResolved; n+1 when none is), and whether hₖ < 0.
func (lp *rlocusLoop) markovRelDegree() (int, bool) {
	h, ok := lp.markovResolved()
	for k := range h {
		if ok[k] {
			return k + 1, h[k] < 0
		}
	}
	return lp.n + 1, false
}

// markovSign returns whether the high-frequency gain of a loop with
// relative degree r ≤ n is negative: the sign of the first resolved hₖ,
// k ≥ r, or of h_r when none is.
func (lp *rlocusLoop) markovSign(r int) bool {
	h, ok := lp.markovResolved()
	if r < 1 || r > len(h) {
		return false
	}
	for k := r - 1; k < len(h); k++ {
		if ok[k] {
			return h[k] < 0
		}
	}
	return h[r-1] < 0
}

// markovResolved returns h₁…hₙ and which of them the coefficients resolve,
// the one threshold of the relative degree, the high-frequency sign and the
// noise-zero trim. With ρₖ = |hₖ|/(eps·sensₖ), hₖ is resolved when
// ρₖ > 10, or when ρₖ > 1 stands 4× above every earlier ρⱼ and starts a
// plateau: every later ρⱼ is within [ρₖ/4, 10·ρₖ]. sensₖ is a first-order
// bound, loose by about the same factor for the exact hₖ of a realization
// (an exact h₂ at ρ 6.7 beside h₃…h₆ at 12…17; h₄…h₆ at 1.4…1.9 above
// rounding-level h₁…h₃ at ≤ 0.2), while a rounding-level hₖ sits at the
// level of the earlier ones or far below the later resolved ones (ρ 1.7
// beside 9e10). Where no ρ reaches that (exact h₂…h₆ at 0.1…1.5 of a
// realization with sens 1e17…1e25), none is resolved and RootLocus takes
// the relative degree of the asymptotes and of the eig path from the zero
// count (the exact break-point path, xpMarkovKeep, keeps the numerator).
func (lp *rlocusLoop) markovResolved() (h []float64, ok []bool) {
	var rho []float64
	lp.markov(func(k int, hk, sens float64) bool {
		r := 0.0
		if sens > 0 {
			r = math.Abs(hk) / (eps() * sens)
		}
		h, rho = append(h, hk), append(rho, r)
		return false
	})
	ok = make([]bool, len(h))
	prev := 0.0
	for k, r := range rho {
		ok[k] = r > 10 || r > 1 && r >= 4*prev && !slices.ContainsFunc(rho[k+1:], func(x float64) bool { return x < r/4 || x > 10*r })
		prev = math.Max(prev, r)
	}
	return h, ok
}

// markov calls f with hₖ and its first-order sensitivity sensₖ to
// elementwise relative changes of A, B and C,
// Σⱼ |C·Aʲ|·|A|·|A^(k−2−j)·B| + |C·A^(k−1)|·|B| + |C|·|A^(k−1)·B|, which
// unlike |C|·|A|^(k−1)·|B| stays tight for non-normal realizations, for
// k = 1…n until f returns true.
func (lp *rlocusLoop) markov(f func(k int, h, sens float64) bool) {
	n := lp.n
	abs := func(v []float64) []float64 {
		out := make([]float64, len(v))
		for i, x := range v {
			out[i] = math.Abs(x)
		}
		return out
	}
	dot := func(x, y []float64) float64 {
		var s float64
		for i := range x {
			s += x[i] * y[i]
		}
		return s
	}
	aAbs := abs(lp.a)
	bAbs, cAbs := abs(lp.b), abs(lp.c)
	var lAbs, arAbs [][]float64
	r, l := slices.Clone(lp.b), slices.Clone(lp.c)
	for k := 1; k <= n; k++ {
		if k > 1 {
			nr, nl := make([]float64, n), make([]float64, n)
			for i := range n {
				row := lp.a[i*n : i*n+n]
				nr[i] = dot(row, r)
				for j, aij := range row {
					nl[j] += l[i] * aij
				}
			}
			r, l = nr, nl
		}
		rAbs := abs(r)
		ar := make([]float64, n)
		for i := range n {
			ar[i] = dot(aAbs[i*n:i*n+n], rAbs)
		}
		lAbs = append(lAbs, abs(l))
		arAbs = append(arAbs, ar)
		h := dot(lp.c, r)
		sens := dot(lAbs[k-1], bAbs) + dot(cAbs, rAbs)
		for j := 0; j <= k-2; j++ {
			sens += dot(lAbs[j], arAbs[k-2-j])
		}
		if f(k, h, sens) {
			return
		}
	}
}

// finiteZeros drops the excess largest zeros beyond the n − r finite ones a
// loop of relative degree r has: rounding noise that MATLAB's tzero reports
// as infinite. r is markovRelDegree, whose resolution threshold
// (markovResolved) the sign rule and the numerator trim share: an hₖ that
// rounding alone could produce keeps no noise zero. Only zeros larger than
// every pole are dropped.
func (lp *rlocusLoop) finiteZeros(zeros, poles []complex128, r int) []complex128 {
	excess := len(zeros) - (lp.n - r)
	if r == 0 || r > lp.n || excess <= 0 {
		return zeros
	}
	maxPole := 0.0
	for _, p := range poles {
		maxPole = math.Max(maxPole, cmplx.Abs(p))
	}
	idx := make([]int, len(zeros))
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(i, j int) int { return cmp.Compare(cmplx.Abs(zeros[j]), cmplx.Abs(zeros[i])) })
	drop := make([]bool, len(zeros))
	for _, i := range idx[:excess] {
		if !(cmplx.Abs(zeros[i]) > maxPole) {
			break
		}
		drop[i] = true
	}
	kept := make([]complex128, 0, len(zeros))
	for i, z := range zeros {
		if !drop[i] {
			kept = append(kept, z)
		}
	}
	return kept
}

// eval returns G(s) and G'(s) = −C(sI−A)⁻²B at the real point s; ok is
// false when the sign of G is within the first-order rounding bound
// 1e2·n·eps·|y|·|sI−A|·|x| of x = (sI−A)⁻¹B, y = C(sI−A)⁻¹ (s at a pole
// or zero, up to rounding).
func (lp *rlocusLoop) eval(s float64) (g, dg float64, ok bool) {
	n := lp.n
	m := make([]float64, n*n)
	mt := make([]float64, n*n)
	for i := range n {
		for j := range n {
			v := -lp.a[i*n+j]
			if i == j {
				v += s
			}
			m[i*n+j] = v
			mt[j*n+i] = v
		}
	}
	x, ok := solveDense(slices.Clone(m), slices.Clone(lp.b), n)
	if !ok {
		return 0, 0, false
	}
	y, ok := solveDense(mt, slices.Clone(lp.c), n)
	if !ok {
		return 0, 0, false
	}
	g = lp.d
	bound := 0.0
	for i := range n {
		g += lp.c[i] * x[i]
		dg -= y[i] * x[i]
		var row float64
		for j := range n {
			row += math.Abs(m[i*n+j]) * math.Abs(x[j])
		}
		bound += math.Abs(y[i]) * row
	}
	if !isFinite(g) || !isFinite(dg) || math.Abs(g) <= 1e2*float64(n)*eps()*(bound+math.Abs(lp.d)) {
		return g, dg, false
	}
	return g, dg, true
}

// multipleRootNear returns the multiplicity m ≥ 2 (0 when none) and the
// centre c of a multiple root of p (the denominator, numerator or
// N'D − ND', with computed roots) up to a backward error of u = 2·n·eps in
// its coefficients (relative to their magnitudes pb, |p| when nil), at the
// centre of the root cluster near s, c being the root of p^(m−1) found by
// Newton from s. For the denominator (numerator) K = 0 (∞) there, so the
// cluster is a multiple pole (zero) split by rounding and a break point
// under either angle condition; for N'D − ND' it is one stationary point.
// m is the size of the cluster itself: the roots within 10·ρ₂ of c,
// ρ₂ = √(u·pb(|c|)/|p″(c)/2|) the rounding split of a double root there.
// A double-root test alone would merge m ≥ 3 distinct roots a relative δ
// apart, since p(c) ~ δ^m. s itself must lie within 10·ρₘ of c,
// ρₘ = (u·pb(|c|)/|p^(m)(c)/m!|)^(1/m) the split of that m-fold root: a
// resolved root of N'D − ND' beside a multiple pole is no part of it.
func multipleRootNear(p, pb Poly, s float64, n int, roots []complex128) (centre float64, m int) {
	u := 2 * float64(n) * eps()
	deg := len(p) - 1
	if deg < 2 {
		return 0, 0
	}
	if pb == nil {
		pb = absPoly(p)
	}
	bounds := []Poly{pb}
	boundAt := func(k int, c float64) float64 {
		for len(bounds) <= k {
			bounds = append(bounds, bounds[len(bounds)-1].Derivative())
		}
		return evalReal(bounds[k], math.Abs(c))
	}
	scale := 0.0
	for _, r := range roots {
		scale = math.Max(scale, cmplx.Abs(r))
	}
	if scale == 0 {
		scale = 1
	}
	ders := []Poly{p}
	der := func(k int) Poly {
		for len(ders) <= k {
			ders = append(ders, ders[len(ders)-1].Derivative())
		}
		return ders[k]
	}
	newton := func(m int) (float64, bool) {
		c := s
		for range 4 {
			f, df := evalReal(der(m-1), c), evalReal(der(m), c)
			if df == 0 {
				break
			}
			c -= f / df
		}
		return c, isFinite(c) && math.Abs(c-s) <= 1e-3*math.Max(math.Abs(s), scale)
	}
	c, ok := newton(2)
	if !ok {
		return 0, 0
	}
	rho := math.Sqrt(u * boundAt(0, c) / math.Abs(evalReal(der(2), c)/2))
	for _, r := range roots {
		if cmplx.Abs(r-complex(c, 0)) <= 10*rho {
			m++
		}
	}
	m = max(m, 2)
	if m > deg {
		return 0, 0
	}
	if m > 2 {
		if c, ok = newton(m); !ok {
			return 0, 0
		}
	}
	for j := range m {
		if math.Abs(evalReal(der(j), c)) > u*boundAt(j, c) {
			return 0, 0
		}
	}
	f := 1.0
	for k := 2; k <= m; k++ {
		f *= float64(k)
	}
	if math.Abs(s-c) > 10*math.Pow(u*boundAt(0, c)/math.Abs(evalReal(der(m), c)/f), 1/float64(m)) {
		return 0, 0
	}
	return c, m
}

// absPoly returns the polynomial of the coefficient magnitudes of p.
func absPoly(p Poly) Poly {
	out := make(Poly, len(p))
	for i, c := range p {
		out[i] = math.Abs(c)
	}
	return out
}

func evalReal(p Poly, x float64) float64 {
	var v float64
	for _, c := range p {
		v = v*x + c
	}
	return v
}

func evalRealBound(p Poly, x float64) (v, bound float64) {
	ax := math.Abs(x)
	for _, c := range p {
		v = v*x + c
		bound = bound*ax + math.Abs(c)
	}
	return v, bound
}

// isStationary reports whether G'/G at s is small against
// Σ 1/|s−p| + Σ 1/|s−z|, the scale of its pole and zero terms, so a root of
// the converted N'D − ND' that the state space does not confirm (a
// transfer-function conversion error) is dropped.
func isStationary(s, g, dg float64, poles, zeros []complex128) bool {
	var t float64
	for _, r := range slices.Concat(poles, zeros) {
		t += 1 / cmplx.Abs(complex(s, 0)-r)
	}
	return math.Abs(dg/g) <= 1e-2*t
}

// solveDense solves the row-major n×n system m·x = b in place by Gaussian
// elimination with partial pivoting; ok is false at an exactly zero pivot.
func solveDense(m, b []float64, n int) ([]float64, bool) {
	for k := range n {
		p := k
		for i := k + 1; i < n; i++ {
			if math.Abs(m[i*n+k]) > math.Abs(m[p*n+k]) {
				p = i
			}
		}
		if m[p*n+k] == 0 {
			return nil, false
		}
		if p != k {
			for j := range n {
				m[k*n+j], m[p*n+j] = m[p*n+j], m[k*n+j]
			}
			b[k], b[p] = b[p], b[k]
		}
		for i := k + 1; i < n; i++ {
			f := m[i*n+k] / m[k*n+k]
			for j := k + 1; j < n; j++ {
				m[i*n+j] -= f * m[k*n+j]
			}
			b[i] -= f * b[k]
		}
	}
	for k := n - 1; k >= 0; k-- {
		v := b[k]
		for j := k + 1; j < n; j++ {
			v -= m[k*n+j] * b[j]
		}
		b[k] = v / m[k*n+k]
	}
	return b, true
}

// unitScalePoly scales p by a power of two so its largest coefficient has
// magnitude in [0.5, 1). N'D − ND' then keeps its roots exactly and cannot
// overflow when the loop's coefficients are near the float64 range.
func unitScalePoly(p Poly) Poly {
	var m float64
	for _, c := range p {
		m = math.Max(m, math.Abs(c))
	}
	if m == 0 || !isFinite(m) {
		return p
	}
	_, e := math.Frexp(m)
	out := make(Poly, len(p))
	for i, c := range p {
		out[i] = math.Ldexp(c, -e)
	}
	return out
}

// netClusters lowers each cluster's multiplicity by the roots of the other
// kind within its radius: a zero at a multiple pole cancels in the minimal
// transfer function, so a net simple root there is no break point.
func netClusters(clusters []rootCluster, others, otherTF []complex128) []rootCluster {
	for i, c := range clusters {
		o := others
		if c.tf {
			o = otherTF
		}
		for _, r := range o {
			if cmplx.Abs(r-complex(c.centre, 0)) <= c.radius {
				clusters[i].m--
			}
		}
	}
	return clusters
}

// rootCluster is a real pole (K = 0) or zero (K = ∞) of net multiplicity m
// at centre, a break point under either angle condition when m ≥ 2; roots
// of the other kind within radius of the centre cancel against it.
type rootCluster struct {
	centre, radius float64
	m              int
	// exact marks a cluster of exactly equal roots.
	exact bool
	// tf marks a multiplicity confirmed by the converted polynomial, which
	// is netted against the other converted polynomial's roots: the
	// conversion may already have cancelled a common root.
	tf bool
	// resolved marks a tf multiplicity whose roots eig resolves (no u·κ
	// link), or a complex eig pair that only the √u·σ_b split merges and
	// the coefficients do not confirm: no proof of a multiple root, so it
	// is a break point only where K > 0.
	resolved bool
}

// polyRootsOrNil returns the roots of p, nil when it has none or rooting
// fails.
func polyRootsOrNil(p Poly) []complex128 {
	if len(p) < 2 {
		return nil
	}
	r, err := p.Roots()
	if err != nil {
		return nil
	}
	return r
}

// multipleRoots returns the real multiple roots among the computed roots
// of p (the unit-scaled denominator or numerator); kappa, when not nil,
// holds the roots' condition numbers (eigCondition, zeroCondition). With
// u = 2·n·eps, σ the smaller of the balanced norm σ_b of A and the largest
// pole or zero modulus, and ν = √u·σ, exactly equal roots form a group, and
// the others join when within 1e-2 of each other relative to their size,
// within ν, or merged to first order by a relative change u of A's
// entries, |λᵢ − λⱼ| ≤ u·(κᵢ + κⱼ); the largest m roots within u^(1/m)·σ of
// the origin join too, as rounding splits an m-fold root at 0 past ν for
// m ≥ 3. A group of m is a multiple root at its
// centre when
//   - its roots are equal;
//   - it lies within u^(1/m)·σ of the origin, where normwise rounding of a
//     realization of scale σ splits an m-fold root at 0, and for eig roots
//     its u·κ links connect all of it (two resolved slow doubles of a
//     fast/slow loop are no 4-fold root at 0); or it is split by
//     at most u^(1/m)·σₜ about a centre within 10·u·σₜ of the origin
//     (σₜ = σ_b for poles and for zeros with a feedthrough, else 0: the
//     centre of an eig cluster is its
//     trace, which rounding of A moves by only ~u·σ_b, so a resolved slow
//     cluster stays off 0 even when a realization of scale σ_b cannot split
//     its roots);
//   - each of its roots is merged by u with another and it is an eig pair
//     (pairCond set) whose mean u·κ_mean is below its split and that is
//     real or straddles the origin (a zero pair of zeroCondition's capped
//     normwise κ is left to the next rule: that κ links a resolved
//     near-axis complex zero pair); a complex pair off the origin that is
//     only split by at most √u·σ_b is no proof of a double (a non-normal A
//     of scale σ_b resolves 116±0.25i, σ_b 4.4e7): the coefficients must
//     merge it within a backward error u^(2/3), a relative root split
//     ~u^(1/3) (doubles of the frozen sweeps whose floats still hold a
//     double need ≤ 3e4·u; exactly realized resolved pairs 6e5·u and more,
//     −42±0.0026i and −3584±0.4375i of non-normal A among them, which √u
//     admitted; doubles whose float TF is already a resolved pair need
//     2e5…3.5e6·u and are precision limits), or
//     else it is a double only where the converted p splits it into two
//     real roots (eig and p disagree on the split, so it is rounding), and
//     otherwise it is a break point only where K > 0 (rootCluster.resolved);
//   - or p has, up to a backward error of 10·u in its coefficients, a root
//     of multiplicity m at the centre c found by Newton on p^(m−1) within
//     max(2·spread, ρ) of the group mean (ρ the split of that backward
//     error, splitRadius); 100·u for m ≥ 3 poles merged by u, which eig
//     cannot resolve and the TF conversion misses by more than 10·u (an
//     exact companion triple pole by ~30·u), while a resolved near triple
//     needs ~400·u; 100·u also confirms a real pole pair merged by u (a
//     semisimple double that eig splits by rounding only), whose isolation
//     from the rest of its group stays at 10·u.
//
// A group that is none is split at its widest link (splitWidest) and its
// parts tested by the last rule alone, a part only when no other root of
// the group lies within the radius splitRadius by which that backward
// error splits its m-fold root (else p(c) ~ δ^m merges a near (m+1)-fold
// cluster as an m-fold one): a double pole beside a resolved simple pole
// keeps its break point. A real pair part is also a double root by the pair
// rule, and a pair part whose mean u·κ_mean (pairCond) is not below its
// distance to the rest of the group is not tested at all: eig cannot order
// such roots, so which two of them pair is rounding.
//
// The coefficient test is no conditioning test of the cluster itself: a
// root nearby shrinks p and p′ at c, so a pair resolved by eig (a complex
// pair with no u·κ link) can pass it. Such a pair is marked resolved and
// computeBreakaway keeps it only where K > 0. Only a complex pair is: the
// real-axis rule is exact beside it, while a double root split into two
// real roots puts one on each side of c. Only eig roots (pairCond set) are:
// zeroCondition's normwise κ does not resolve a split double zero.
//
// Resolved roots, however slow or close, therefore stay apart; a near
// multiple root that only eig's rounding merges but the coefficients do not
// confirm is left to the stationary-point rules.
func multipleRoots(roots []complex128, kappa []float64, pairCond func(i, j int) float64, p Poly, n int, sigma, sigmaBal, sigmaTrace float64) []rootCluster {
	k := len(roots)
	if k < 2 {
		return nil
	}
	u := 2 * float64(n) * eps()
	nu, nuBal := math.Sqrt(u)*sigma, math.Sqrt(u)*sigmaBal
	group := make([]int, k)
	for i := range group {
		group[i] = i
	}
	find := func(i int) int {
		for group[i] != i {
			group[i] = group[group[i]]
			i = group[i]
		}
		return i
	}
	exact := make([]bool, k)
	for i := range k {
		for j := i + 1; j < k; j++ {
			if roots[i] == roots[j] {
				exact[i], exact[j] = true, true
				group[find(i)] = find(j)
			}
		}
	}
	for m := k; m >= 2; m-- {
		var in []int
		for i, r := range roots {
			if !exact[i] && cmplx.Abs(r) <= math.Pow(u, 1/float64(m))*sigma {
				in = append(in, i)
			}
		}
		if len(in) >= m {
			for _, i := range in[1:] {
				group[find(i)] = find(in[0])
			}
			break
		}
	}
	for i := range k {
		for j := i + 1; j < k; j++ {
			if exact[i] || exact[j] {
				continue
			}
			d := cmplx.Abs(roots[i] - roots[j])
			if d <= math.Max(1e-2*math.Max(cmplx.Abs(roots[i]), cmplx.Abs(roots[j])), nu) || kappa != nil && d <= u*(kappa[i]+kappa[j]) {
				group[find(i)] = find(j)
			}
		}
	}
	var out []rootCluster
	var test func(members, rest []int)
	test = func(members, rest []int) {
		sub := rest != nil
		m := len(members)
		if m < 2 {
			return
		}
		var sum complex128
		for _, j := range members {
			sum += roots[j]
		}
		c0 := sum / complex(float64(m), 0)
		spread := 0.0
		for _, j := range members {
			spread = math.Max(spread, cmplx.Abs(roots[j]-c0))
		}
		if math.Abs(imag(c0)) <= spread {
			radius := math.Max(10*spread, nu)
			linked := kappa != nil
			for _, a := range members {
				linked = linked && slices.ContainsFunc(members, func(b int) bool {
					return b != a && cmplx.Abs(roots[a]-roots[b]) <= u*(kappa[a]+kappa[b])
				})
			}
			connected := kappa != nil
			if connected {
				reach := []int{members[0]}
				for k := 0; k < len(reach); k++ {
					for _, b := range members {
						if !slices.Contains(reach, b) && cmplx.Abs(roots[reach[k]]-roots[b]) <= u*(kappa[reach[k]]+kappa[b]) {
							reach = append(reach, b)
						}
					}
				}
				connected = len(reach) == m
			}
			origin := math.Max(spread, cmplx.Abs(c0)) <= math.Pow(u, 1/float64(m))*sigma && (connected || pairCond == nil) ||
				cmplx.Abs(c0) <= 10*u*sigmaTrace && spread <= math.Pow(u, 1/float64(m))*sigmaTrace
			cplx := imag(roots[members[0]]) != 0
			pair := m == 2 && pairCond != nil && u*pairCond(members[0], members[1]) < spread
			// normwise marks a complex pair off the origin that only the √u·σ_b
			// rounding split admits: no proof of a double, so the converted
			// polynomial must confirm it, else K > 0 decides.
			normwise := pair && cplx && cmplx.Abs(c0) > spread
			pair = pair && (!normwise || spread <= nuBal)
			if sub && m == 2 && pairCond != nil {
				gap := math.Inf(1)
				for _, j := range rest {
					gap = math.Min(gap, cmplx.Abs(roots[j]-c0))
				}
				if u*pairCond(members[0], members[1]) >= gap {
					return
				}
			}
			if spread == 0 || !sub && origin || linked && pair && !normwise && (!sub || !cplx) {
				out = append(out, rootCluster{centre: real(c0), radius: radius, m: m, exact: spread == 0})
				return
			}
			tol := 10 * u
			if linked && m >= 3 && sigmaTrace > 0 {
				tol = 100 * u
			}
			conf := tol
			if linked && m == 2 && imag(roots[members[0]]) == 0 && sigmaTrace > 0 {
				conf = 100 * u
			}
			if linked && pair && normwise && !sub {
				conf = math.Pow(u, 2.0/3)
			}
			if m < len(p) {
				c, ok := multipleRootAt(p, real(c0), m, conf, math.Max(2*spread, splitRadius(p, real(c0), m, conf)))
				if ok && sub {
					rho := splitRadius(p, c, m, tol)
					ok = !slices.ContainsFunc(rest, func(j int) bool { return cmplx.Abs(roots[j]-complex(c, 0)) <= rho })
				}
				if ok {
					out = append(out, rootCluster{centre: c, radius: math.Max(radius, math.Sqrt(u)*math.Abs(c)), m: m, tf: true, resolved: pairCond != nil && m == 2 && !linked && cplx})
					return
				}
				if linked && pair && normwise && !sub {
					out = append(out, rootCluster{centre: real(c0), radius: radius, m: m, resolved: !realPairAt(p, real(c0), math.Max(10*spread, splitRadius(p, real(c0), m, conf)))})
					return
				}
			}
		}
		parts := splitWidest(roots, members)
		for i, part := range parts {
			test(part, slices.Concat(rest, parts[1-i]))
		}
	}
	for i := range k {
		if find(i) != i {
			continue
		}
		var members []int
		for j := range k {
			if find(j) == i {
				members = append(members, j)
			}
		}
		test(members, nil)
	}
	return mergeClusters(out, u, sigma, sigmaTrace)
}

// mergeClusters joins the clusters of multipleRoots in its origin band
// (|c| ≤ u^(1/m)·σ or ≤ 10·u·σₜ) into one multiple root at the origin.
// Exactly equal roots are grouped apart from the others, so exact delay
// poles at 0 and an eig cluster at 0 would otherwise each be reported.
func mergeClusters(cs []rootCluster, u, sigma, sigmaTrace float64) []rootCluster {
	atOrigin := func(c rootCluster) bool {
		return math.Abs(c.centre) <= math.Max(math.Pow(u, 1/float64(c.m))*sigma, 10*u*sigmaTrace)
	}
	for i := 0; i < len(cs); i++ {
		for j := i + 1; j < len(cs); j++ {
			a, b := cs[i], cs[j]
			d := math.Abs(a.centre - b.centre)
			if !atOrigin(a) || !atOrigin(b) {
				continue
			}
			m := a.m + b.m
			cs[i] = rootCluster{
				centre:   (float64(a.m)*a.centre + float64(b.m)*b.centre) / float64(m),
				radius:   max(a.radius, b.radius, d),
				m:        m,
				exact:    a.exact && b.exact && d == 0,
				tf:       a.tf && b.tf,
				resolved: a.resolved && b.resolved,
			}
			cs = slices.Delete(cs, j, j+1)
			j = i
		}
	}
	return cs
}

// splitWidest splits the group members of roots in two at the widest link
// of its single-linkage tree (minimum spanning tree), so a group that is no
// multiple root as a whole is tested as its tighter sub-groups. A complex
// root and its nearest conjugate stay together, as the roots of a real
// polynomial split into conjugate-closed sub-groups; nil when no split is
// left.
func splitWidest(roots []complex128, members []int) [][]int {
	var units [][]int
	used := make([]bool, len(members))
	for i, a := range members {
		if used[i] {
			continue
		}
		used[i] = true
		unit := []int{a}
		if imag(roots[a]) != 0 {
			best := -1
			for j, b := range members {
				if !used[j] && imag(roots[b])*imag(roots[a]) < 0 &&
					(best < 0 || cmplx.Abs(roots[b]-cmplx.Conj(roots[a])) < cmplx.Abs(roots[members[best]]-cmplx.Conj(roots[a]))) {
					best = j
				}
			}
			if best >= 0 {
				used[best] = true
				unit = append(unit, members[best])
			}
		}
		units = append(units, unit)
	}
	m := len(units)
	if m < 2 {
		return nil
	}
	gap := func(x, y []int) float64 {
		d := math.Inf(1)
		for _, a := range x {
			for _, b := range y {
				d = math.Min(d, cmplx.Abs(roots[a]-roots[b]))
			}
		}
		return d
	}
	in := make([]bool, m)
	dist := make([]float64, m)
	parent := make([]int, m)
	in[0] = true
	for j := 1; j < m; j++ {
		dist[j] = gap(units[j], units[0])
	}
	widest, cut := -1.0, -1
	for range m - 1 {
		next := -1
		for j := range m {
			if !in[j] && (next < 0 || dist[j] < dist[next]) {
				next = j
			}
		}
		in[next] = true
		if dist[next] > widest {
			widest, cut = dist[next], next
		}
		for j := range m {
			if d := gap(units[j], units[next]); !in[j] && d < dist[j] {
				dist[j], parent[j] = d, next
			}
		}
	}
	if widest <= 0 {
		return nil
	}
	side := make([]int, m)
	for j := range m {
		side[j] = -1
	}
	side[0], side[cut] = 0, 1
	var sideOf func(j int) int
	sideOf = func(j int) int {
		if side[j] < 0 {
			side[j] = sideOf(parent[j])
		}
		return side[j]
	}
	parts := make([][]int, 2)
	for j := range m {
		parts[sideOf(j)] = append(parts[sideOf(j)], units[j]...)
	}
	return parts
}

// zeroClusters returns the real multiple zeros (multipleRoots). With a
// feedthrough the zeros are the eigenvalues of A − B·C/D, whose condition
// numbers (against the entrywise rounding |A| + |B|·|C|/|D| of forming it)
// and its characteristic polynomial (converted as for the poles, with the
// trace rule) decide them as for poles, so an exact multiple zero is not
// lost to the rounding of the converted numerator; without one, the
// zeros' condition numbers in the system pencil (zeroCondition) do.
func (lp *rlocusLoop) zeroClusters(sys *System, zeros []complex128, num Poly, sigma, sigmaBal float64) []rootCluster {
	n := lp.n
	if lp.d != 0 {
		az, w := make([]float64, n*n), make([]float64, n*n)
		finite := true
		for i := range n {
			for j := range n {
				bc := lp.b[i] * lp.c[j] / lp.d
				az[i*n+j] = lp.a[i*n+j] - bc
				w[i*n+j] = math.Abs(lp.a[i*n+j]) + math.Abs(bc)
				finite = finite && isFinite(az[i*n+j]) && isFinite(w[i*n+j])
			}
		}
		if finite {
			if ec, ok := eigCondition(az, w, n); ok {
				p, sb := num, balancedNorm(az, n)
				if zs, err := NewFromSlices(n, 1, 1, az, lp.b, lp.c, []float64{0}, 0); err == nil {
					if tfr, err := zs.rationalTransferFunction(nil); err == nil {
						p = unitScalePoly(Poly(tfr.TF.Den[0]))
					}
				}
				return multipleRoots(ec.vals, ec.kappa, ec.pairCondition, p, n, sigma, sb, sb)
			}
		}
	}
	if cs, ok := lp.zeroDynamicsClusters(zeros, num, sigma); ok {
		return cs
	}
	return multipleRoots(zeros, zeroCondition(sys, zeros), nil, num, n, sigma, sigmaBal, 0)
}

// zeroDynamicsClusters decides the multiple zeros of a strictly proper loop
// of resolved relative degree r ≤ n − 2 as for poles: the zeros are the
// n − r eigenvalues of Az = A − B·C·A^r/h_r away from its r structural
// eigenvalues at 0, so the condition numbers of those eigenvalues (against
// |A| + |B|·|C·A^r|/|h_r|), each taken for the nearest of the given zeros
// (computed from the pencil, which forming Az does not round), and their
// pair conditioning merge a double zero that the similarity's
// rounding split, and the converted numerator confirms it. Forming Az
// rounds past those κ, so no cluster is marked resolved (a double zero −5
// that eig of Az splits ±6.2e-7i, just past its u·κ link, stays one).
// ok is false (zeroCondition decides) when r is not resolved, a zero lies
// in the origin band where the structural eigenvalues are, or eig fails.
func (lp *rlocusLoop) zeroDynamicsClusters(zeros []complex128, num Poly, sigma float64) ([]rootCluster, bool) {
	n, r := lp.n, lp.relDeg
	if r < 1 || r > n-2 || len(zeros) != n-r {
		return nil, false
	}
	u := 2 * float64(n) * eps()
	for _, z := range zeros {
		if cmplx.Abs(z) <= math.Sqrt(u)*sigma {
			return nil, false
		}
	}
	q := slices.Clone(lp.c)
	for range r - 1 {
		q = rowTimes(q, lp.a, n)
	}
	var h float64
	for i := range n {
		h += q[i] * lp.b[i]
	}
	if h == 0 {
		return nil, false
	}
	qa := rowTimes(q, lp.a, n)
	az, w := make([]float64, n*n), make([]float64, n*n)
	for i := range n {
		for j := range n {
			bq := lp.b[i] * qa[j] / h
			az[i*n+j] = lp.a[i*n+j] - bq
			w[i*n+j] = math.Abs(lp.a[i*n+j]) + math.Abs(bq)
			if !isFinite(az[i*n+j]) || !isFinite(w[i*n+j]) {
				return nil, false
			}
		}
	}
	ec, ok := eigCondition(az, w, n)
	if !ok {
		return nil, false
	}
	idx := make([]int, n)
	for i := range idx {
		idx[i] = i
	}
	slices.SortStableFunc(idx, func(i, j int) int { return cmp.Compare(cmplx.Abs(ec.vals[j]), cmplx.Abs(ec.vals[i])) })
	idx = idx[:n-r]
	vals, kappa := slices.Clone(zeros), make([]float64, n-r)
	used := make([]bool, n-r)
	match := make([]int, n-r)
	for k, z := range vals {
		best := -1
		for m, i := range idx {
			if !used[m] && (best < 0 || cmplx.Abs(ec.vals[i]-z) < cmplx.Abs(ec.vals[idx[best]]-z)) {
				best = m
			}
		}
		used[best] = true
		match[k] = idx[best]
		kappa[k] = ec.kappa[idx[best]]
	}
	idx = match
	pc := func(i, j int) float64 { return ec.pairCondition(idx[i], idx[j]) }
	cs := multipleRoots(vals, kappa, pc, num, n, sigma, balancedNorm(az, n), 0)
	for i := range cs {
		cs[i].resolved = false
	}
	return cs, true
}

// rowTimes returns the row vector x·A for the row-major n×n A.
func rowTimes(x, a []float64, n int) []float64 {
	out := make([]float64, n)
	for i, xi := range x {
		for j := range n {
			out[j] += xi * a[i*n+j]
		}
	}
	return out
}

// zeroCondition returns the normwise condition numbers of the zeros of
// the SISO model sys as generalized eigenvalues of its system pencil
// S(λ) = [A−λI B; C D], in the prescaled realization (zerosPrescale) whose
// pencil the zeros are computed from, accurate to ε·‖S‖:
// κ = ‖S‖·(‖x‖₁ + 1)·(‖y‖₁ + 1) / |C·(λI−A)⁻²·B| with x = (λI−A)⁻¹B,
// yᵀ = C(λI−A)⁻¹, so a change ε·‖S‖ of the pencil's entries moves λ by
// ≈ ε·κ to first order. κ is capped at ‖S‖/√u, u = 2·n·eps, so a root
// links (multipleRoots) only within the √u·‖S‖ rounding split of a double
// root, not with every zero where the first-order κ is infinite; κ is 0
// where λI−A is singular; nil for fewer than two zeros.
func zeroCondition(sys *System, zeros []complex128) []float64 {
	if len(zeros) < 2 {
		return nil
	}
	zs := zerosPrescale(sys)
	n, _, _ := zs.Dims()
	a, b, c := zs.A.RawMatrix(), zs.B.RawMatrix(), zs.C.RawMatrix()
	nrm := math.Abs(zs.D.At(0, 0))
	for i := range n {
		row := math.Abs(b.Data[i*b.Stride])
		for j := range n {
			row += math.Abs(a.Data[i*a.Stride+j])
		}
		nrm = math.Max(nrm, row)
	}
	row := math.Abs(zs.D.At(0, 0))
	for j := range n {
		row += math.Abs(c.Data[j])
	}
	nrm = math.Max(nrm, row)
	kmax := nrm / math.Sqrt(2*float64(n)*eps())
	kappa := make([]float64, len(zeros))
	m, mt := make([]complex128, n*n), make([]complex128, n*n)
	x, y := make([]complex128, n), make([]complex128, n)
	for k, z := range zeros {
		for i := range n {
			for j := range n {
				v := complex(-a.Data[i*a.Stride+j], 0)
				if i == j {
					v += z
				}
				m[i*n+j], mt[j*n+i] = v, v
			}
			x[i], y[i] = complex(b.Data[i*b.Stride], 0), complex(c.Data[i], 0)
		}
		if !solveDenseC(m, x, n) || !solveDenseC(mt, y, n) {
			continue
		}
		var den complex128
		nx, ny := 1.0, 1.0
		for i := range n {
			den += y[i] * x[i]
			nx += cmplx.Abs(x[i])
			ny += cmplx.Abs(y[i])
		}
		kappa[k] = kmax
		if kp := nrm * nx * ny / cmplx.Abs(den); kp < kmax {
			kappa[k] = kp
		}
	}
	return kappa
}

// solveDenseC solves the row-major complex n×n system m·x = b in place
// (m overwritten) by Gaussian elimination with partial pivoting; false at
// an exactly zero pivot.
func solveDenseC(m, b []complex128, n int) bool {
	for k := range n {
		p := k
		for i := k + 1; i < n; i++ {
			if cmplx.Abs(m[i*n+k]) > cmplx.Abs(m[p*n+k]) {
				p = i
			}
		}
		if m[p*n+k] == 0 {
			return false
		}
		if p != k {
			for j := range n {
				m[k*n+j], m[p*n+j] = m[p*n+j], m[k*n+j]
			}
			b[k], b[p] = b[p], b[k]
		}
		for i := k + 1; i < n; i++ {
			f := m[i*n+k] / m[k*n+k]
			for j := k + 1; j < n; j++ {
				m[i*n+j] -= f * m[k*n+j]
			}
			b[i] -= f * b[k]
		}
	}
	for k := n - 1; k >= 0; k-- {
		v := b[k]
		for j := k + 1; j < n; j++ {
			v -= m[k*n+j] * b[j]
		}
		b[k] = v / m[k*n+k]
	}
	return true
}

// eigCond holds eigenvalues of a matrix with their right and left
// eigenvectors and the entrywise rounding scale w of the matrix.
type eigCond struct {
	n      int
	vals   []complex128
	kappa  []float64
	vr, vl mat.CDense
	a, w   []float64
}

func (e *eigCond) weight(r, c int) float64 {
	if e.w != nil {
		return e.w[r*e.n+c]
	}
	return math.Abs(e.a[r*e.n+c])
}

// eigCondition returns the eigenvalues of the row-major n×n matrix a and
// their componentwise condition numbers κᵢ = |yᵢ|ᵀ·W·|xᵢ| / |yᵢᴴxᵢ|, W the
// entrywise rounding scale of a (|a| when w is nil): a relative change ε of
// W moves λᵢ by ≈ ε·κᵢ to first order.
func eigCondition(a, w []float64, n int) (*eigCond, bool) {
	var eig mat.Eigen
	if !eig.Factorize(mat.NewDense(n, n, slices.Clone(a)), mat.EigenBoth) {
		return nil, false
	}
	e := &eigCond{n: n, vals: eig.Values(nil), a: a, w: w, kappa: make([]float64, n)}
	eig.VectorsTo(&e.vr)
	eig.LeftVectorsTo(&e.vl)
	ax := make([]float64, n)
	for i := range n {
		var dot complex128
		for r := range n {
			dot += cmplx.Conj(e.vl.At(r, i)) * e.vr.At(r, i)
			var s float64
			for c := range n {
				s += e.weight(r, c) * cmplx.Abs(e.vr.At(c, i))
			}
			ax[r] = s
		}
		var num float64
		for r := range n {
			num += cmplx.Abs(e.vl.At(r, i)) * ax[r]
		}
		e.kappa[i] = num / cmplx.Abs(dot)
	}
	return e, true
}

// pairCondition returns the componentwise condition number of the mean
// (λᵢ + λⱼ)/2, ½·Σ W∘|Pᵀ| for the spectral projector P = X·(YᴴX)⁻¹·Yᴴ,
// X = [xᵢ xⱼ], Y = [yᵢ yⱼ], of the pair's invariant subspace. A double root split by rounding
// has a mean far better conditioned than its roots (u·κ_mean below the
// split), while two resolved roots of a non-normal matrix do not.
func (e *eigCond) pairCondition(i, j int) float64 {
	n := e.n
	idx := [2]int{i, j}
	var g [2][2]complex128
	for p, a := range idx {
		for q, b := range idx {
			for r := range n {
				g[p][q] += cmplx.Conj(e.vl.At(r, a)) * e.vr.At(r, b)
			}
		}
	}
	det := g[0][0]*g[1][1] - g[0][1]*g[1][0]
	if det == 0 {
		return math.Inf(1)
	}
	inv := [2][2]complex128{{g[1][1] / det, -g[0][1] / det}, {-g[1][0] / det, g[0][0] / det}}
	z := make([][2]complex128, n)
	for r := range n {
		for k := range 2 {
			z[r][k] = inv[k][0]*cmplx.Conj(e.vl.At(r, i)) + inv[k][1]*cmplx.Conj(e.vl.At(r, j))
		}
	}
	var kappa float64
	for r := range n {
		for c := range n {
			pcr := e.vr.At(c, i)*z[r][0] + e.vr.At(c, j)*z[r][1]
			kappa += e.weight(r, c) * cmplx.Abs(pcr)
		}
	}
	return kappa / 2
}

// splitRadius returns ρ = (tol·Σ|pᵢ||c|ⁱ / |p^(m)(c)/m!|)^(1/m), the radius
// by which a backward error tol in p's coefficients splits an m-fold root
// at c.
func splitRadius(p Poly, c float64, m int, tol float64) float64 {
	d, f := p, 1.0
	for k := 1; k <= m; k++ {
		d, f = d.Derivative(), f*float64(k)
	}
	_, b := evalRealBound(p, c)
	return math.Pow(tol*b/math.Abs(evalReal(d, c)/f), 1/float64(m))
}

// multipleRootAt reports whether p has, up to a backward error of tol in
// its coefficients, a root of multiplicity m at c, the root of p^(m−1)
// found by Newton from s within reach of s (p^(m−1)(c) is tested too, so
// a Newton run that did not reach a real root confirms nothing).
func multipleRootAt(p Poly, s float64, m int, tol, reach float64) (float64, bool) {
	ders := []Poly{p}
	for range m {
		ders = append(ders, ders[len(ders)-1].Derivative())
	}
	c := s
	for range 8 {
		f, df := evalReal(ders[m-1], c), evalReal(ders[m], c)
		if df == 0 {
			break
		}
		c -= f / df
	}
	if !isFinite(c) || math.Abs(c-s) > reach {
		return 0, false
	}
	for j := range m {
		if v, b := evalRealBound(ders[j], c); math.Abs(v) > tol*b {
			return 0, false
		}
	}
	return c, true
}

// realPairAt reports whether p has two real roots within reach of s about
// the root c of p′ found by Newton from s: p(c) and p(c ± reach) differ in
// sign on both sides.
func realPairAt(p Poly, s, reach float64) bool {
	d1 := p.Derivative()
	d2 := d1.Derivative()
	c := s
	for range 8 {
		f, df := evalReal(d1, c), evalReal(d2, c)
		if df == 0 {
			break
		}
		c -= f / df
	}
	if !isFinite(c) || math.Abs(c-s) > reach {
		return false
	}
	v := evalReal(p, c)
	return v*evalReal(p, c-reach) < 0 && v*evalReal(p, c+reach) < 0
}

// balance returns D⁻¹·A·D for LAPACK's diagonal balancing D (Dgebal,
// scaling only) and the diagonal of D.
func balance(a []float64, n int) (ab, scale []float64) {
	ab, scale = slices.Clone(a), make([]float64, n)
	impl.Dgebal(lapack.Scale, n, ab, n, scale)
	return ab, scale
}

// balanced returns the loop in the balanced coordinates D⁻¹·A·D, D⁻¹·B,
// C·D (balance), where solves with sI − A are accurate to the scale of
// its eigenvalues rather than of A's largest entries.
func (lp *rlocusLoop) balanced() *rlocusLoop {
	ab, scale := balance(lp.a, lp.n)
	bl := *lp
	bl.a, bl.b, bl.c = ab, make([]float64, lp.n), make([]float64, lp.n)
	for i, d := range scale {
		bl.b[i], bl.c[i] = lp.b[i]/d, lp.c[i]*d
	}
	return &bl
}

// balancedNorm returns the 1-norm of A after LAPACK diagonal balancing
// (Dgebal), the scale eig's rounding is relative to.
func balancedNorm(a []float64, n int) float64 {
	b, _ := balance(a, n)
	nrm := 0.0
	for j := range n {
		var s float64
		for i := range n {
			s += math.Abs(b[i*n+j])
		}
		nrm = math.Max(nrm, s)
	}
	return nrm
}

// isOnRealAxisSegment reports whether an odd number of poles and zeros lie
// right of s. Complex roots come in conjugate pairs with equal real parts,
// so counting them leaves the parity of the real ones unchanged.
func isOnRealAxisSegment(s float64, poles, zeros []complex128) bool {
	count := 0
	for _, r := range slices.Concat(poles, zeros) {
		if real(r) > s {
			count++
		}
	}
	return count%2 == 1
}
