package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"sort"

	"plantcontrol.org/v1/gonum/mat"
)

// matchedDCRoot is the root at DC of a delay-free SISO model for matched
// conversion, the stored data's exact limit as DCGain takes it (U-3): order
// is that of the lowest nonzero term of its transfer function in s − x0
// (positive for zeros at DC, negative for poles, 0 for none) and lead that
// term's coefficient, rounded.
type matchedDCRoot struct {
	order int
	lead  float64
	// zeros and poles are the exact multiplicities at DC of the numerator
	// and denominator before cancellation: order = zeros − poles.
	zeros, poles int
}

// matchedReferenceOf is newMatchedReference with an exactly singular DC
// pencil x0·I − A (explicit, or E dropping out at s = 0; order up to
// exactDetMaxOrder) decided exactly, whether or not its LU meets a zero
// pivot: a regular DC solve that certifies proves it nonsingular, else
// exactShiftedDetIsZero decides. Where it is singular and DCGain is finite
// and nonzero (the stored exact limit cancels the pole at DC), the bar is
// u·(2|g0 − D| + |D|), u = 4(n+5)ε, the floor at the stored order of the
// descriptor bar with the group inverse A# in place of A⁻¹ (U-A), and the
// terms |D| + |g0 − D|.
func matchedReferenceOf(ref *System) matchedReference {
	r := newMatchedReference(ref)
	if r.zero {
		return r
	}
	free := ioDelayFree(ref)
	n, _, _ := free.Dims()
	x0 := 0.0
	if free.IsDiscrete() {
		x0 = 1
	}
	if n == 0 || n > exactDetMaxOrder || (x0 != 0 && free.E != nil) {
		return r
	}
	if r.err == nil && solveDCRegular(nil, free.A, free.B, free.C, free.D, x0, false).certified {
		return r
	}
	if !exactShiftedDetIsZero(free.A.RawMatrix(), x0) {
		return r
	}
	g, err := free.DCGain()
	if err != nil {
		return matchedReference{err: matchedUnverified("DCGain of the model: %v", err)}
	}
	g0 := g.At(0, 0)
	if g0 == 0 || math.IsInf(g0, 0) || math.IsNaN(g0) {
		return matchedReference{err: matchedUnverified("DCGain of the model is %v", g0)}
	}
	d := math.Abs(free.D.At(0, 0))
	e := math.Abs(g0 - free.D.At(0, 0))
	return matchedReference{g0: g0, terms: d + e, bar: 4 * float64(n+5) * 0x1p-52 * (2*e + d)}
}

// matchedDCRootOf is the root at DC of ref, read from DCGain: a finite
// nonzero gain has none unless noise (a gain within its own rounding terms
// of 0, which may stand for an exact zero); a gain of 0 or ±Inf, or noise,
// takes the order and leading coefficient of the exact Laurent expansion
// (matchedExactLaurent), whose sign at a pole must be that of DCGain. ok is false where DCGain fails or
// the expansion is unavailable or disagrees with it.
func matchedDCRootOf(ref *System, noise bool) (matchedDCRoot, bool) {
	free := ioDelayFree(ref)
	g, err := free.DCGain()
	if err != nil {
		return matchedDCRoot{}, false
	}
	v := g.At(0, 0)
	switch {
	case math.IsNaN(v):
		return matchedDCRoot{}, false
	case v != 0 && !math.IsInf(v, 0) && !noise:
		return matchedDCRoot{}, true
	}
	x0 := 0.0
	if free.IsDiscrete() {
		x0 = 1
	}
	zeros, poles, lead, ok := matchedExactExpansion(free.E, free.A, free.B, free.C, free.D, x0)
	order := zeros - poles
	switch {
	case !ok, v == 0 && order <= 0, math.IsInf(v, 0) && (order >= 0 || math.Signbit(v) != math.Signbit(lead)), !math.IsInf(v, 0) && order < 0:
		return matchedDCRoot{}, false
	}
	return matchedDCRoot{order: order, lead: lead, zeros: zeros, poles: poles}, true
}

// matchedExactLaurent returns the order of the lowest nonzero term of
// G(x0+σ) = D + C((x0+σ)E − A)⁻¹B in σ, E nil for the identity, exactly for
// the stored entries, and its coefficient rounded. Every entry is scaled to
// an integer by one power of two; det(σÊ + P̂) and
// det([σÊ + P̂, B̂; −Ĉ, D̂]) = det(σÊ + P̂)·Ĝ have degree at most n and are
// interpolated from the integer nodes σ = 0…n. ok is false for n = 0 or
// above exactDetMaxOrder, a singular pencil or a zero transfer function.
func matchedExactLaurent(E, A, B, C, D *mat.Dense, x0 float64) (order int, lead float64, ok bool) {
	zeros, poles, lead, ok := matchedExactExpansion(E, A, B, C, D, x0)
	return zeros - poles, lead, ok
}

// matchedExactExpansion is matchedExactLaurent with the orders of the
// lowest terms of the numerator and of det(σÊ + P̂) apart.
func matchedExactExpansion(E, A, B, C, D *mat.Dense, x0 float64) (zeros, poles int, lead float64, ok bool) {
	n, _ := A.Dims()
	if n == 0 || n > exactDetMaxOrder {
		return 0, 0, 0, false
	}
	emin := 1
	note := func(m *mat.Dense) {
		raw := m.RawMatrix()
		for i := range raw.Rows {
			for _, v := range raw.Data[i*raw.Stride : i*raw.Stride+raw.Cols] {
				if v != 0 {
					_, e := math.Frexp(v)
					emin = min(emin, e)
				}
			}
		}
	}
	for _, m := range []*mat.Dense{A, B, C, D, E} {
		if m != nil {
			note(m)
		}
	}
	scaled := func(v float64) *big.Int {
		f, e := math.Frexp(v)
		x := big.NewInt(int64(math.Ldexp(f, 53)))
		return x.Lsh(x, uint(e-emin))
	}
	w := n + 1
	eHat := make([]*big.Int, n*n)
	pHat := make([]*big.Int, n*n)
	for i := range n {
		for j := range n {
			switch {
			case E != nil:
				eHat[i*n+j] = scaled(E.At(i, j))
			case i == j:
				eHat[i*n+j] = scaled(1)
			default:
				eHat[i*n+j] = new(big.Int)
			}
			p := scaled(A.At(i, j))
			p.Neg(p)
			if x0 != 0 {
				p.Add(p, eHat[i*n+j])
			}
			pHat[i*n+j] = p
		}
	}
	nodes := make([]int64, w)
	dVals := make([]*big.Int, w)
	nVals := make([]*big.Int, w)
	sub := make([]*big.Int, n*n)
	full := make([]*big.Int, w*w)
	t := new(big.Int)
	for k := range w {
		nodes[k] = int64(k)
		for i := range n {
			for j := range n {
				v := new(big.Int).Mul(eHat[i*n+j], t.SetInt64(int64(k)))
				v.Add(v, pHat[i*n+j])
				sub[i*n+j] = v
				full[i*w+j] = new(big.Int).Set(v)
			}
			full[i*w+n] = scaled(B.At(i, 0))
			c := scaled(C.At(0, i))
			full[n*w+i] = c.Neg(c)
		}
		full[n*w+n] = scaled(D.At(0, 0))
		dVals[k] = bareissDet(sub, n)
		nVals[k] = bareissDet(full, w)
	}
	od, dLow := lowestNonzero(newtonAscending(nodes, dVals))
	on, nLow := lowestNonzero(newtonAscending(nodes, nVals))
	if od < 0 || on < 0 {
		return 0, 0, 0, false
	}
	f := new(big.Float).SetRat(new(big.Rat).Quo(nLow, dLow))
	shift := 53 - emin
	lead, _ = f.SetMantExp(f, -shift).Float64()
	return on, od, lead, true
}

// matchedMarkDCPair marks the roots at DC of a root set whose transfer
// function has root at DC: zeros and poles nearest DC in equal extra
// numbers beyond |order|, as many as the exact multiplicities and the sets
// allow, for copies the transfer function kept of a cancelled root at DC.
// Each count is tried from the fewest up; the first whose leading
// coefficient lead (with gain of the marked sets) is within terms
// (matchedLeadAgrees) is taken, else the nearest one. ok is false where no
// count can be marked.
func matchedMarkDCPair(cz, cp []complex128, root matchedDCRoot, lead func(z, p []complex128) complex128) (mz, mp []bool, ok bool) {
	best := math.Inf(1)
	for np := max(0, -root.order); np <= min(root.poles, len(cp)); np++ {
		nz := np + root.order
		if nz > min(root.zeros, len(cz)) {
			break
		}
		z, okz := matchedMarkDC(cz, nz)
		p, okp := matchedMarkDC(cp, np)
		if !okz || !okp {
			continue
		}
		v := lead(matchedAtDC(cz, z, 0), matchedAtDC(cp, p, 0))
		if matchedLeadAgrees(v, root.lead) {
			return z, p, true
		}
		if e := cmplx.Abs(v-complex(root.lead, 0)) / math.Max(math.Abs(root.lead), cmplx.Abs(v)); !(e >= best) {
			best, mz, mp, ok = e, z, p, true
		}
	}
	return mz, mp, ok
}

// matchedMarkDC marks the k roots of smallest modulus as the roots at DC,
// a conjugate pair only whole; ok is false where the roots cannot make up k.
func matchedMarkDC(roots []complex128, k int) ([]bool, bool) {
	idx := make([]int, len(roots))
	for i := range idx {
		idx[i] = i
	}
	sort.SliceStable(idx, func(a, b int) bool { return cmplx.Abs(roots[idx[a]]) < cmplx.Abs(roots[idx[b]]) })
	marks := make([]bool, len(roots))
	for _, i := range idx {
		if k == 0 {
			break
		}
		r := roots[i]
		switch {
		case marks[i]:
		case imag(r) == 0:
			marks[i] = true
			k--
		case k >= 2:
			j := -1
			for l, q := range roots {
				if l != i && !marks[l] && imag(q)*imag(r) < 0 && (j < 0 || cmplx.Abs(q-cmplx.Conj(r)) < cmplx.Abs(roots[j]-cmplx.Conj(r))) {
					j = l
				}
			}
			if j >= 0 {
				marks[i], marks[j] = true, true
				k -= 2
			}
		}
	}
	return marks, k == 0
}

// matchedAtDC returns roots with the marked ones set to at exactly.
func matchedAtDC(roots []complex128, marks []bool, at complex128) []complex128 {
	out := make([]complex128, len(roots))
	for i, r := range roots {
		out[i] = r
		if marks[i] {
			out[i] = at
		}
	}
	return out
}

// matchedLeadAgrees is the root gate at the leading term: the leading
// coefficient v of a root set with its own gain is within
// √ε·max(|lead|, |v|) of the model's exact one.
func matchedLeadAgrees(v complex128, lead float64) bool {
	tol := math.Sqrt(0x1p-52) * math.Max(math.Abs(lead), cmplx.Abs(v))
	return cmplx.Abs(v-complex(lead, 0)) <= tol
}

// matchedLaurentAgrees reports that the DC gain of the stored delay-free
// explicit out at x0, exact for its entries, is that of ref with its root
// at DC: at a pole, the exact expansion of out has the order of root and a
// coefficient of its sign (±Inf of that sign); at a zero, the DC gain of
// out is 0 or within the descriptor bar of ref (matchedZeroBar).
func matchedLaurentAgrees(out, ref *System, root matchedDCRoot, x0 float64) bool {
	order, lead, ok := matchedExactLaurent(nil, out.A, out.B, out.C, out.D, x0)
	switch {
	case !ok:
		return false
	case root.order < 0:
		return order == root.order && lead != 0 && math.Signbit(lead) == math.Signbit(root.lead)
	case order > 0:
		return true
	case order < 0:
		return false
	}
	bar, ok := matchedZeroBar(ref)
	return ok && math.Abs(lead) <= bar
}

// matchedZeroBar is the descriptor bar of the delay-free ref at s = 0 or
// z = 1 (newDCBar) for a model whose DC gain is exactly 0, where
// newMatchedReference has none; ok is false where no bar exists.
func matchedZeroBar(ref *System) (float64, bool) {
	b := newDCBar(ioDelayFree(ref))
	if b.reason != "" {
		return 0, false
	}
	bar := 0.0
	if b.lu == nil {
		bar = b.bar[0]
	} else {
		bar = b.fullBar(0)
	}
	return bar, !math.IsNaN(bar) && !math.IsInf(bar, 0)
}

// matchedGainAtFloatDC is r.matchedGain where a mapped root lies at DC
// exactly in float64 though ref has none there (ratio 0): the gain stays
// the leading-term one, base, under the root gate, and the DC gate on the
// result decides. Other cases are r.matchedGain's.
func (r matchedReference) matchedGainAtFloatDC(v, ratio complex128, base float64) (float64, error) {
	if r.err != nil || r.zero || ratio != 0 {
		return r.matchedGain(v, ratio, base)
	}
	if !r.rootsAgree(v) {
		return base, matchedUnverified("DC product %v of the roots is not within terms of DCGain %v", real(v), r.g0)
	}
	return base, nil
}

// matchedDCProduct is the leading product of a discrete root set at z = 1:
// Π(1 − z)/Π(1 − p), a root at exactly z = 1 taking the factor dt.
func matchedDCProduct(zeros, poles []complex128, dt float64) complex128 {
	v := complex(1, 0)
	for _, z := range zeros {
		if z == 1 {
			v *= complex(dt, 0)
		} else {
			v *= 1 - z
		}
	}
	for _, p := range poles {
		if p == 1 {
			v /= complex(dt, 0)
		} else {
			v /= 1 - p
		}
	}
	return v
}

// matchedLeadProduct is the leading product of a continuous root set at
// s = 0 with gain k: k·Π(−z)/Π(−p) over the roots not at exactly 0.
func matchedLeadProduct(k float64, zeros, poles []complex128) complex128 {
	v := complex(k, 0)
	for _, z := range zeros {
		if z != 0 {
			v *= -z
		}
	}
	for _, p := range poles {
		if p != 0 {
			v /= -p
		}
	}
	return v
}

// discretizeMatchedAtDC is discretizeMatched where ref has a root at DC
// (matchedDCRootOf): the roots nearest DC of the set mapped are set to s = 0
// exactly (z = 1) by matchedMarkDCPair, the others polished as elsewhere
// when they are the roots of the transfer function (else the pencil gain
// is refitted to the response of sys with them in place), and the gain
// matches the exact leading coefficient of ref, a root at DC taking the
// factor dt. The result is unverified unless the leading coefficient of the
// roots with their own gain is within terms of it (matchedLeadAgrees) and
// the DC gain of the result is that of ref (matchedLaurentAgrees); where
// the roots fail that gate, or cannot hold the root, they are mapped as
// given with their own leading-term gain, formed like the matched one.
func (sys *System) discretizeMatchedAtDC(dt float64, ref *System, rational *TransferFuncResult, cz, cp []complex128, k float64, transfer bool, root matchedDCRoot) (*System, error) {
	free := *sys
	free.Delay, free.InputDelay, free.OutputDelay = nil, nil, nil
	fitted := func(z, p []complex128) float64 {
		if transfer {
			return k
		}
		s0, _ := matchedProbePoints(p, z)
		if g0, ok := matchedEvalModel(&free, s0); ok {
			return real(g0 / matchedRootRatio(z, p, s0))
		}
		return k
	}
	mz, mp, ok := matchedMarkDCPair(cz, cp, root, func(z, p []complex128) complex128 {
		return matchedLeadProduct(fitted(z, p), z, p)
	})
	unmarked := func(cz, cp []complex128, k float64, unverified error) (*System, error) {
		dz, dp, gain, err := matchedDiscreteRoots(cz, cp, k, dt)
		if err != nil {
			return nil, err
		}
		if own := real(matchedLeadProduct(k, cz, cp)) / real(matchedMappedProduct(cz, cp, dz, dt, complex(dt, 0))); !math.IsNaN(own) && !math.IsInf(own, 0) && own != 0 {
			gain = own
		}
		realized, err := matchedCascade(dz, dp, gain, dt)
		if err != nil {
			return nil, err
		}
		return matchedDiscreteResult(realized, sys, dt, unverified)
	}
	if !ok {
		return unmarked(cz, cp, k, matchedUnverified("the roots do not hold the root of order %d at DC", root.order))
	}
	cz0, cp0, k0 := cz, cp, k
	cz, cp = matchedAtDC(cz, mz, 0), matchedAtDC(cp, mp, 0)
	k = fitted(cz, cp)
	dz, dp, gain, err := matchedDiscreteRoots(cz, cp, k, dt)
	if err != nil {
		return nil, err
	}
	if transfer {
		pz := matchedAtDC(matchedPolish(rational.TF.Num[0][0], cz), mz, 0)
		pp := matchedAtDC(matchedPolish(rational.TF.Den[0], cp), mp, 0)
		if z, p, g, perr := matchedDiscreteRoots(pz, pp, k, dt); perr == nil {
			cz, cp, dz, dp, gain = pz, pp, z, p, g
		}
	}
	var unverified error
	q := matchedMappedProduct(cz, cp, dz, dt, complex(dt, 0))
	switch g := root.lead / real(q); {
	case real(q) == 0 || math.IsNaN(g) || math.IsInf(g, 0) || g == 0:
		unverified = matchedUnverified("leading product %v of the mapped roots", q)
	default:
		gain = g
		if v := matchedLeadProduct(k, cz, cp); !matchedLeadAgrees(v, root.lead) {
			return unmarked(cz0, cp0, k0, matchedUnverified("leading coefficient %v of the roots is not within terms of %v", real(v), root.lead))
		}
	}
	realized, err := matchedCascade(dz, dp, gain, dt)
	if err != nil {
		return nil, err
	}
	if unverified == nil && !matchedLaurentAgrees(realized, ref, root, 1) {
		unverified = matchedUnverified("the result does not hold the root of order %d at DC", root.order)
	}
	return matchedDiscreteResult(realized, sys, dt, unverified)
}

// matchedD2CModelAtDC is matchedD2CModel where ref has a root at DC: the
// discrete roots nearest z = 1 (in s = log(z)/dt) are set to z = 1 and
// s = 0 exactly by matchedMarkDCPair, the others polished when num and den
// are given, and the gain matches the exact leading coefficient of ref at
// z = 1, times dt^order; roots from the pencil of fit (non-nil) have their
// gain refitted to its response with the roots at DC in place. The second
// result is unverified as for discretizeMatchedAtDC, the DC gain of the
// result taken at s = 0.
func matchedD2CModelAtDC(dz, dp []complex128, k float64, artificial int, dt float64, ref, fit *System, root matchedDCRoot, num, den []float64) (*System, error, error) {
	lz, err := matchedLogRoots(dz, dt)
	if err != nil {
		return nil, nil, err
	}
	lp, err := matchedLogRoots(dp, dt)
	if err != nil {
		return nil, nil, err
	}
	var free System
	if fit != nil {
		free = *fit
		free.Delay, free.InputDelay, free.OutputDelay = nil, nil, nil
	}
	fitted := func(z, p []complex128) float64 {
		if fit == nil {
			return k
		}
		all := matchedWithArtificial(z, artificial)
		s0, _ := matchedD2CProbePoints(dt, p, z)
		if g0, ok := matchedEvalModel(&free, s0); ok {
			return real(g0 / matchedRootRatio(all, p, s0))
		}
		return k
	}
	mz, mp, ok := matchedMarkDCPair(lz, lp, root, func(z, p []complex128) complex128 {
		dzm, dpm := make([]complex128, len(z)), make([]complex128, len(p))
		for i := range z {
			dzm[i] = dz[i]
			if z[i] == 0 {
				dzm[i] = 1
			}
		}
		for i := range p {
			dpm[i] = dp[i]
			if p[i] == 0 {
				dpm[i] = 1
			}
		}
		return complex(fitted(dzm, dpm), 0) * matchedDCProduct(matchedWithArtificial(dzm, artificial), dpm, 1)
	})
	if !ok {
		gain, err := matchedContinuousGain(matchedWithArtificial(dz, artificial), dp, lz, lp, k, dt)
		if err != nil {
			return nil, nil, err
		}
		out, _, _, err := matchedCascadeDC(lz, lp, gain, 0)
		return out, matchedUnverified("the roots do not hold the root of order %d at DC", root.order), err
	}
	dz, dp = matchedAtDC(dz, mz, 1), matchedAtDC(dp, mp, 1)
	k = fitted(dz, dp)
	if num != nil {
		pz, pp := matchedAtDC(matchedPolish(num, dz), mz, 1), matchedAtDC(matchedPolish(den, dp), mp, 1)
		if _, err := matchedLogRoots(pz, dt); err == nil {
			if _, err := matchedLogRoots(pp, dt); err == nil {
				dz, dp = pz, pp
			}
		}
	}
	cz, err := matchedLogRoots(dz, dt)
	if err != nil {
		return nil, nil, err
	}
	cp, err := matchedLogRoots(dp, dt)
	if err != nil {
		return nil, nil, err
	}
	all := matchedWithArtificial(dz, artificial)
	gain, err := matchedContinuousGain(all, dp, cz, cp, k, dt)
	if err != nil {
		return nil, nil, err
	}
	base := gain
	var unverified error
	q := matchedLeadProduct(1, cz, cp)
	switch g := root.lead * math.Pow(dt, float64(root.order)) / real(q); {
	case real(q) == 0 || math.IsNaN(g) || math.IsInf(g, 0) || g == 0:
		unverified = matchedUnverified("leading product %v of the mapped roots", q)
	default:
		gain = g
		if v := complex(k, 0) * matchedDCProduct(all, dp, 1); !matchedLeadAgrees(v, root.lead) {
			unverified = matchedUnverified("leading coefficient %v of the roots is not within terms of %v", real(v), root.lead)
			gain = base
		}
	}
	out, _, _, err := matchedCascadeDC(cz, cp, gain, 0)
	if err != nil {
		return nil, nil, err
	}
	if unverified == nil && !matchedLaurentAgrees(out, ref, root, 0) {
		unverified = matchedUnverified("the result does not hold the root of order %d at DC", root.order)
	}
	return out, unverified, nil
}

func matchedWithArtificial(dz []complex128, artificial int) []complex128 {
	all := make([]complex128, len(dz), len(dz)+artificial)
	copy(all, dz)
	for range artificial {
		all = append(all, -1)
	}
	return all
}

// matchedMappedProduct is the DC product Π(1 − z)/Π(1 − p) of the discrete
// roots mapped from the continuous cz, cp, each 1 − e^{root·dt} formed
// without cancellation from its continuous root (a stored e^{root·dt} near
// 1 cannot hold it, and a gain matched through it is off by as much
// everywhere else); zeros past cz (artificial) take 1 − z, and a root at
// exactly s = 0 the factor atDC.
func matchedMappedProduct(cz, cp, dz []complex128, dt float64, atDC complex128) complex128 {
	v := complex(1, 0)
	for i, z := range dz {
		switch {
		case i >= len(cz):
			v *= 1 - z
		case cz[i] == 0:
			v *= atDC
		default:
			v *= -matchedExpm1(cz[i] * complex(dt, 0))
		}
	}
	for _, p := range cp {
		if p == 0 {
			v /= atDC
		} else {
			v /= -matchedExpm1(p * complex(dt, 0))
		}
	}
	return v
}

// matchedExpm1 is e^w − 1 without cancellation for small |w|.
func matchedExpm1(w complex128) complex128 {
	x, y := real(w), imag(w)
	s := math.Sin(y / 2)
	return complex(math.Expm1(x)*math.Cos(y)-2*s*s, math.Exp(x)*math.Sin(y))
}
