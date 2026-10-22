package controlsys

import (
	"cmp"
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"

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
	// Breakaway lists real break-away/break-in points on the locus; empty
	// when the locus has none.
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
			zeros = lp.finiteZeros(zeros, poles, lp.nonVanishingDegree())
		}
		if r := n - len(zeros); r >= 1 && r != lp.relDeg {
			// Zeros() fixes the asymptote count; the angles, the 0° rule and
			// the numerator trim follow the Markov parameter of that degree.
			lp.relDeg, lp.negGain = r, lp.markovSign(r)
			lp.spRelDeg = r
		}
	}
	breakaway, err := computeBreakaway(sys, undelayed, delay, &lp, poles, zeros)
	if err != nil {
		return nil, fmt.Errorf("%s: %w", op, err)
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

// computeBreakaway returns the real roots of N'D − ND' on the K > 0 locus.
// The numerator is trimmed to the loop's relative degree, so rounding-noise
// leading coefficients add no far stationary points, and N, D are scaled
// by powers of two so N'D − ND' cannot overflow. An absorbed delay of L
// samples is kept exact as D = z^L·D0 (the L noise trailing coefficients of
// the full-order D dropped, or D0 from the undelayed loop when the
// conversion reduced the order), so z = 0 is reported once, not rooted from
// an L-fold cluster; if rooting that form fails, the full D is rooted as is.
// A root at a repeated pole or zero (K = 0 or ∞) is kept under either angle
// condition.
// Any other root is on the locus when K = −1/G(s) > 0 and G'(s) ≈ 0, both
// from the state space, so a sign rule over rounding-noise zeros or a
// conversion error in N/D cannot misplace it; where the sign of G is not
// resolved the real-axis rule (0° for a negative high-frequency gain)
// decides.
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
	roots, err := stationaryCandidates(sNum, sDen0, sDelay, lp.spRelDeg, lp.n)
	if err != nil && sDelay > 0 {
		roots, err = stationaryCandidates(sNumFull, sDenFull, 0, lp.spRelDeg, lp.n)
		num, den0, delay = numFull, denFull, 0
	}
	if err != nil {
		return nil, err
	}
	num, den := trimmedTF(num, den0, delay, lp.relDeg, lp.n)

	var result []complex128
	for _, r := range roots {
		if math.Abs(imag(r)) > 1e-6*math.Max(1, math.Abs(real(r))) {
			continue
		}
		s := real(r)
		if isNearRepeatedReal(s, poles) || isNearRepeatedReal(s, zeros) {
			result = append(result, complex(s, 0))
			continue
		}
		onLocus := multipleRootNear(den, s, lp.n, poles) || multipleRootNear(num, s, lp.n, zeros)
		if !onLocus {
			if g, dg, ok := lp.eval(s); ok {
				onLocus = g < 0 && isStationary(s, g, dg, poles, zeros)
			} else {
				onLocus = isOnRealAxisSegment(s, poles, zeros) != lp.negGain
			}
		}
		if onLocus {
			result = append(result, complex(s, 0))
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
	switch {
	case delay <= 0:
		delay = 0
	case len(den0) == n+1:
		den0 = den0[:len(den0)-delay]
	case undelayed != nil && slices.ContainsFunc(num, func(c float64) bool { return c != 0 }):
		if u, uerr := undelayed.rationalTransferFunction(nil); uerr == nil {
			num, den0 = u.TF.Num[0][0], u.TF.Den[0]
		} else {
			delay = 0
		}
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
// with N trimmed to relative degree relDeg and N, D unit-scaled. Roots is
// nil when the converted N, D or N'D − ND' are empty, all zero or not
// finite.
func stationaryCandidates(num, den0 Poly, delay, relDeg, n int) (roots []complex128, err error) {
	num, den := trimmedTF(num, den0, delay, relDeg, n)
	if len(num) == 0 {
		return nil, nil
	}
	den0 = den[:len(den)-delay]
	notFinite := func(c float64) bool { return !isFinite(c) }
	if slices.ContainsFunc(num, notFinite) || slices.ContainsFunc(den0, notFinite) {
		return []complex128{}, nil
	}

	// D = z^L·D0 is exact, so N'D − ND' = z^(L−1)·(z·(N'D0 − ND0') − L·N·D0).
	poly := num.Derivative().Mul(den0).Sub(num.Mul(den0.Derivative()))
	if delay > 0 {
		poly = slices.Concat(poly, Poly{0}).Sub(num.Mul(den0).Scale(float64(delay)))
	}
	if !slices.ContainsFunc(poly, func(c float64) bool { return c != 0 }) || slices.ContainsFunc(poly, notFinite) {
		return []complex128{}, nil
	}
	atZero := delay > 1
	for len(poly) > 1 && poly[len(poly)-1] == 0 {
		poly, atZero = poly[:len(poly)-1], true
	}
	if roots, err = poly.Roots(); err != nil {
		return nil, err
	}
	if atZero {
		roots = append(roots, 0)
	}
	return roots, nil
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
// coefficients (n+1 when none is), and whether hₖ < 0.
func (lp *rlocusLoop) markovRelDegree() (int, bool) {
	r, neg := lp.n+1, false
	lp.markov(func(k int, h, sens float64) bool {
		if resolved := math.Abs(h) > 10*eps()*sens; resolved {
			r, neg = k, h < 0
			return true
		}
		return false
	})
	return r, neg
}

// nonVanishingDegree returns the index of the first hₖ that no elementwise
// rounding of A, B and C can make zero to first order, |hₖ| > eps·sensₖ
// (n+1 when none): the least relative degree the coefficients allow.
func (lp *rlocusLoop) nonVanishingDegree() int {
	r := lp.n + 1
	lp.markov(func(k int, h, sens float64) bool {
		if math.Abs(h) > eps()*sens {
			r = k
			return true
		}
		return false
	})
	return r
}

// markovSign returns whether the high-frequency gain of a loop with
// relative degree r ≤ n is negative: the sign of the first resolved hₖ,
// k ≥ r, or of h_r when none is.
func (lp *rlocusLoop) markovSign(r int) bool {
	var neg, set bool
	lp.markov(func(k int, h, sens float64) bool {
		resolved := math.Abs(h) > 10*eps()*sens
		if k == r || (k > r && resolved) {
			neg, set = h < 0, true
		}
		return k >= r && resolved
	})
	return set && neg
}

// markov calls f with hₖ and its first-order sensitivity sensₖ to
// elementwise relative changes of A, B and C,
// Σⱼ |C·Aʲ|·|A|·|A^(k−2−j)·B| + |C·A^(k−1)|·|B| + |C|·|A^(k−1)·B|, which
// unlike |C|·|A|^(k−1)·|B| stays tight for non-normal realizations, for
// k = 1…n until f returns true. hₖ is resolved when it exceeds
// 10·eps·sensₖ.
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
// as infinite. r is nonVanishingDegree, used only when some hₖ is resolved
// (computed hₖ within the margin are not themselves reliable). Only zeros
// larger than every pole are dropped.
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

// multipleRootNear reports whether p (the denominator or numerator, with
// computed roots) has, up to a backward error of u = 2·n·eps in its
// coefficients, a root of multiplicity m ≥ 2 at the centre c of the root
// cluster near s, c being the root of p^(m−1) found by Newton from s. K = 0
// (∞) there, so the cluster is a multiple pole (zero) split by rounding and
// a break point under either angle condition. m is the size of the cluster
// itself: the roots within 10·ρ₂ of c, ρ₂ = √(u·Σ|pᵢ||c|ⁱ/|p″(c)/2|) the
// rounding split of a double root there. A double-root test alone would
// merge m ≥ 3 distinct roots a relative δ apart, since p(c) ~ δ^m.
func multipleRootNear(p Poly, s float64, n int, roots []complex128) bool {
	u := 2 * float64(n) * eps()
	deg := len(p) - 1
	if deg < 2 {
		return false
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
	centre := func(m int) (float64, bool) {
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
	c, ok := centre(2)
	if !ok {
		return false
	}
	_, b0 := evalRealBound(p, c)
	rho := math.Sqrt(u * b0 / math.Abs(evalReal(der(2), c)/2))
	m := 0
	for _, r := range roots {
		if cmplx.Abs(r-complex(c, 0)) <= 10*rho {
			m++
		}
	}
	m = max(m, 2)
	if m > deg {
		return false
	}
	if m > 2 {
		if c, ok = centre(m); !ok {
			return false
		}
	}
	for j := range m {
		v, b := evalRealBound(der(j), c)
		if math.Abs(v) > u*b {
			return false
		}
	}
	return true
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

func isOnRealAxisSegment(s float64, poles, zeros []complex128) bool {
	// Count real poles and zeros to the right of s
	count := 0
	for _, p := range poles {
		if math.Abs(imag(p)) < 1e-6 && real(p) > s {
			count++
		}
	}
	for _, z := range zeros {
		if math.Abs(imag(z)) < 1e-6 && real(z) > s {
			count++
		}
	}
	return count%2 == 1
}

func isNearRepeatedReal(s float64, roots []complex128) bool {
	count := 0
	for _, r := range roots {
		if math.Abs(imag(r)) < 1e-6 && math.Abs(real(r)-s) < 1e-4 {
			count++
		}
	}
	return count >= 2
}
