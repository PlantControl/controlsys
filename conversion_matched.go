package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"slices"
	"sort"

	"plantcontrol.org/v1/gonum/mat"
)

// d2cMatched uses principal logarithms of finite roots. All zeros at z=-1
// are interpreted as the artificial zeros of matched discretization and removed.
// Other nonpositive real roots are rejected rather than inventing a complex model.
// The gain is set from the DCGain of ref, the caller's model, or where ref
// has a pole or zero at DC (matchedDCRootOf) from its exact leading
// coefficient; the result is returned with errMatchedUnverified unless the
// discrete roots' own DC product and the result's DC gain agree with it
// (matchedReference, matchedD2CModelAtDC).
func (sys *System) d2cMatched(ref *System) (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("matched: %w", ErrWrongDomain)
	}
	if sys.Dt <= 0 || math.IsNaN(sys.Dt) || math.IsInf(sys.Dt, 0) {
		return nil, ErrInvalidSampleTime
	}
	_, m, p := sys.Dims()
	if m != 1 || p != 1 {
		return nil, fmt.Errorf("matched: %w", ErrNotSISO)
	}
	if sys.IsDescriptor() {
		return nil, ErrDescriptorUnsupported
	}
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("matched: internal delays: %w", ErrFeedbackDelay)
	}
	result, err := sys.rationalTransferFunction(nil)
	if err != nil {
		return nil, fmt.Errorf("matched: %w", err)
	}
	numerator, artificial := matchedDeflateArtificialZeros(result.TF.Num[0][0])
	result.TF.Num[0][0] = numerator
	disc, err := result.TF.ZPK()
	if err != nil {
		return nil, err
	}
	dz, dp, k := disc.Zeros[0][0], disc.Poles[0][0], disc.Gain[0][0]
	var out *System
	var unverified error
	r, dcg := matchedReferenceOf(ref)
	var root matchedDCRoot
	if noise := r.err == nil && !r.zero && math.Abs(r.g0) <= 0x1p-52*r.terms; r.err != nil || noise {
		root, _ = matchedDCRootOf(ref, noise, dcg)
	}
	if z, p, g, ok := matchedD2CPencilRoots(sys, numerator, dz, dp, k, artificial); ok {
		out, unverified, err = matchedD2CModel(z, p, g, artificial, sys.Dt, ref, sys, r, root, nil, nil)
		if err != nil {
			out = nil
		}
	}
	if out == nil {
		if out, unverified, err = matchedD2CModel(dz, dp, k, artificial, sys.Dt, ref, nil, r, root, numerator, result.TF.Den[0]); err != nil {
			return nil, err
		}
	}
	newDelayConversionPolicy(sys.Dt).applyContinuousDelayFields(out, sys)
	propagateNames(out, sys)
	out.StateName = nil
	return out, unverified
}

// matchedD2CModel realizes the continuous matched model of the discrete
// roots dz (less the artificial zeros at −1), dp and gain k. The roots of
// the transfer function (num and den non-nil) are first polished on their
// polynomials, and the gain matches the DCGain of the reference r; the
// second result is the verdict of the root and DC gates against it. Where
// the model ref has a pole or zero at DC (root) matchedD2CModelAtDC takes
// over, fit the model whose pencil gave the roots, if any.
func matchedD2CModel(dz, dp []complex128, k float64, artificial int, dt float64, ref, fit *System, r matchedReference, root matchedDCRoot, num, den []float64) (*System, error, error) {
	if root.order != 0 {
		return matchedD2CModelAtDC(dz, dp, k, artificial, dt, ref, fit, root, num, den)
	}
	cz, err := matchedLogRoots(dz, dt)
	if err != nil {
		return nil, nil, err
	}
	cp, err := matchedLogRoots(dp, dt)
	if err != nil {
		return nil, nil, err
	}
	if num != nil {
		pz, pp := matchedPolish(num, dz), matchedPolish(den, dp)
		if lz, err := matchedLogRoots(pz, dt); err == nil {
			if lp, err := matchedLogRoots(pp, dt); err == nil {
				dz, dp, cz, cp = pz, pp, lz, lp
			}
		}
	}
	gain, err := matchedContinuousGain(matchedWithArtificial(dz, artificial), dp, cz, cp, k, dt)
	if err != nil {
		return nil, nil, err
	}
	gain, unverified := r.matchedGainAtFloatDC(matchedDiscreteProduct(k, dz, dp, artificial), matchedContinuousProduct(1, cz, cp), gain)
	out, dc, dcOK, err := matchedCascadeDC(cz, cp, gain, 0)
	if err != nil {
		return nil, nil, err
	}
	if unverified == nil && !r.dcAgrees(dc, dcOK) {
		unverified = matchedUnverified("DC gain of the result is not within the descriptor bar of DCGain")
	}
	return out, unverified, nil
}

// matchedD2CPencilRoots returns the poles eig(A), as MATLAB
// d2c(...,'matched') takes them, with zeros other than the artificial
// zeros at −1 and a gain fitted to the response of sys; polynomial roots of
// the transfer function lose tight pole clusters near z = 1. The zeros are
// the invariant zeros less the count nearest −1, as MATLAB takes them,
// unless the transfer-function zeros cz (roots of numerator) are well
// conditioned or tzero cannot resolve the multiple zero at −1. ok is false unless the fit at a probe point is no worse than
// that of cz, cp, k, at low frequency; eig(A) may hold more poles than cp
// only when the transfer function cancelled roots it should not have,
// fitting worse than 1e-6 where the pencil roots fit to 1e-9.
func matchedD2CPencilRoots(sys *System, numerator []float64, cz, cp []complex128, k float64, artificial int) (zeros, poles []complex128, gain float64, ok bool) {
	poles, err := sys.Poles()
	if err != nil || len(poles) == 0 || len(poles) < len(cp) {
		return nil, nil, 0, false
	}
	if poles, ok = matchedConjugateClosed(poles); !ok {
		return nil, nil, 0, false
	}
	withArtificial := func(z []complex128) []complex128 {
		out := slices.Clone(z)
		for range artificial {
			out = append(out, -1)
		}
		return out
	}
	extra := len(poles) - len(cp)
	free := *sys
	free.Delay, free.InputDelay, free.OutputDelay = nil, nil, nil
	tfZeros := withArtificial(cz)
	s0, s1 := matchedD2CProbePoints(sys.Dt, poles, cz, cp)
	g0, ok0 := matchedEvalModel(&free, s0)
	g1, ok1 := matchedEvalModel(&free, s1)
	if !ok0 || !ok1 {
		return nil, nil, 0, false
	}
	misfit := func(z, p []complex128, k float64) float64 {
		return cmplx.Abs(complex(k, 0)*matchedRootRatio(z, p, s1)-g1) / cmplx.Abs(g1)
	}
	limit := math.Max(2*misfit(tfZeros, cp, k), 1e-12)
	if extra > 0 {
		if limit <= 1e-6 {
			return nil, nil, 0, false
		}
		limit = 1e-9
	}
	var candidates [][]complex128
	if extra == 0 {
		candidates = append(candidates, cz)
	}
	if extra > 0 || !matchedRootsWellConditioned(numerator, cz) {
		if pz, ok := matchedD2CInvariantZeros(sys, len(cz)+artificial+extra, artificial); ok {
			candidates = append(candidates, pz)
		}
	}
	for _, z := range slices.Backward(candidates) {
		all := withArtificial(z)
		g := real(g0 / matchedRootRatio(all, poles, s0))
		if e := misfit(all, poles, g); e <= limit {
			return z, poles, g, true
		}
	}
	return nil, nil, 0, false
}

// matchedD2CInvariantZeros returns the count invariant zeros of sys less
// the artificial ones nearest −1.
func matchedD2CInvariantZeros(sys *System, count, artificial int) ([]complex128, bool) {
	pz, err := sys.zerosValue()
	if err != nil || len(pz) != count {
		return nil, false
	}
	pz, ok := matchedConjugateClosed(pz)
	if !ok {
		return nil, false
	}
	return matchedStripArtificialZeros(pz, artificial)
}

// matchedRootsWellConditioned reports whether every root r of the
// polynomial is resolved to 1e-10 of its continuous image log(r)/T: the
// first-order error ε·Σ|aₖ||r|ᵏ/|p'(r)| over |r|·|log r|. The
// invariant zeros are then not needed.
func matchedRootsWellConditioned(poly []float64, roots []complex128) bool {
	for _, r := range roots {
		var v, dv complex128
		var scale float64
		ar := cmplx.Abs(r)
		for _, c := range poly {
			dv = dv*r + v
			v = v*r + complex(c, 0)
			scale = scale*ar + math.Abs(c)
		}
		l := cmplx.Abs(cmplx.Log(r))
		if dv == 0 || l == 0 || !(0x1p-52*scale/cmplx.Abs(dv) <= 1e-10*ar*l) {
			return false
		}
	}
	return true
}

// matchedD2CProbePoints maps the matchedProbePoints of the continuous roots
// log(r)/T back to z = e^{sT}, so the fit is checked at low frequency on the
// scale of the continuous model rather than far from a cluster near z = 1.
func matchedD2CProbePoints(dt float64, sets ...[]complex128) (complex128, complex128) {
	var roots []complex128
	for _, set := range sets {
		for _, r := range set {
			if cmplx.Abs(r) > 0 && (real(r) > 0 || math.Abs(imag(r)) > 1e-10*cmplx.Abs(r)) {
				roots = append(roots, cmplx.Log(r)/complex(dt, 0))
			}
		}
	}
	s0, s1 := matchedProbePoints(roots)
	return cmplx.Exp(s0 * complex(dt, 0)), cmplx.Exp(s1 * complex(dt, 0))
}

// matchedStripArtificialZeros removes the count zeros nearest −1 from the
// zeros; ok is false unless each lies within 0.25 of −1, the spread tzero
// leaves on a many-fold zero there.
func matchedStripArtificialZeros(zeros []complex128, count int) ([]complex128, bool) {
	out := slices.Clone(zeros)
	for range count {
		best := -1
		for i, z := range out {
			if best < 0 || cmplx.Abs(z+1) < cmplx.Abs(out[best]+1) {
				best = i
			}
		}
		if best < 0 || cmplx.Abs(out[best]+1) > 0.25 {
			return nil, false
		}
		out = slices.Delete(out, best, best+1)
	}
	return out, true
}

func matchedLogRoots(roots []complex128, dt float64) ([]complex128, error) {
	out := make([]complex128, 0, len(roots))
	for _, root := range roots {
		if cmplx.IsNaN(root) || cmplx.IsInf(root) {
			return nil, ErrOverflow
		}
		if cmplx.Abs(root) == 0 || (real(root) <= 0 && math.Abs(imag(root)) <= 1e-10*(1+cmplx.Abs(root))) {
			return nil, fmt.Errorf("matched: nonpositive real root %v has no supported real principal logarithm: %w", root, ErrSingularTransform)
		}
		out = append(out, cmplx.Log(root)/complex(dt, 0))
	}
	used := make([]bool, len(out))
	for i, root := range out {
		if used[i] {
			continue
		}
		if math.Abs(imag(root)) <= 1e-10*(1+cmplx.Abs(root)) {
			out[i] = complex(real(root), 0)
			continue
		}
		found := false
		for j := i + 1; j < len(out); j++ {
			if !used[j] && cmplx.Abs(out[j]-cmplx.Conj(root)) <= 1e-7*(1+cmplx.Abs(root)) {
				averaged := (root + cmplx.Conj(out[j])) / 2
				out[i], out[j] = averaged, cmplx.Conj(averaged)
				used[j], found = true, true
				break
			}
		}
		if !found {
			return nil, fmt.Errorf("matched: non-real reconstruction: %w", ErrConjugatePairs)
		}
	}
	return out, nil
}

// matchedContinuousGain is gain times the ratio of the leading discrete
// product at z = 1 to the continuous one at s = 0 of the roots dz, dp
// (artificial zeros past the end of cz) and their images cz, cp. A root at
// DC, its continuous image exactly 0 or its discrete one exactly 1, takes
// the factor dt on the discrete side and none on the continuous one.
func matchedContinuousGain(dz, dp, cz, cp []complex128, gain, dt float64) (float64, error) {
	atDC := func(d, c []complex128, i int) bool {
		return d[i] == 1 || (i < len(c) && c[i] == 0)
	}
	discrete := complex(gain, 0)
	continuous := complex(1, 0)
	for i, z := range dz {
		if atDC(dz, cz, i) {
			discrete *= complex(dt, 0)
		} else {
			discrete *= 1 - z
		}
	}
	for i, p := range dp {
		if atDC(dp, cp, i) {
			discrete /= complex(dt, 0)
		} else {
			discrete /= 1 - p
		}
	}
	for i, z := range cz {
		if !atDC(dz, cz, i) {
			continuous *= -z
		}
	}
	for i, p := range cp {
		if !atDC(dp, cp, i) {
			continuous /= -p
		}
	}
	ratio := discrete / continuous
	if cmplx.IsNaN(ratio) || cmplx.IsInf(ratio) {
		return 0, ErrOverflow
	}
	if math.Abs(imag(ratio)) > 1e-7*math.Max(1, cmplx.Abs(ratio)) {
		return 0, ErrConjugatePairs
	}
	return real(ratio), nil
}

// discretizeMatched is matched C2D of sys, the model the route hands over,
// with the DC gain matched to the DCGain of ref, the caller's model, or
// where ref has a pole or zero at DC (matchedDCRootOf) its leading term
// (discretizeMatchedAtDC). The roots of the transfer function, when
// they are the ones mapped, are polished on their polynomials. The result
// comes with errMatchedUnverified when DCGain of ref is unavailable, the
// roots' own DC product is not within terms of it, or the result's DC gain
// is not within the descriptor bar of it (matchedReference). The gain is
// matched through 1 − e^{root·dt} formed from the continuous roots
// (matchedMappedProduct), else, where that fails the DC gate, through the
// stored discrete roots, unverified unless their DC product, rescaled to
// DCGain, is within its terms (R2's inequality: the rescale must not hide a
// refuted root set).
func (sys *System) discretizeMatched(dt float64, ref *System) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, ErrInvalidSampleTime
	}
	_, m, p := sys.Dims()
	if m != 1 || p != 1 {
		return nil, fmt.Errorf("matched: sys is %d×%d, want SISO: %w", p, m, ErrNotSISO)
	}
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("matched: internal delays: %w", ErrFeedbackDelay)
	}
	if sys.IsDescriptor() {
		sp, stiff := stiffDescriptorParts(sys)
		if !stiff {
			return nil, ErrDescriptorUnsupported
		}
		if err := sp.within(1); err != nil {
			return nil, err
		}
	}
	rational, err := sys.transferRational(nil)
	if err != nil {
		return nil, err
	}
	source, err := rational.TF.ZPK()
	if err != nil {
		return nil, err
	}
	cz, cp, k := source.Zeros[0][0], source.Poles[0][0], source.Gain[0][0]
	dz, dp, gain, err := matchedDiscreteRoots(cz, cp, k, dt)
	transfer := true
	if pz, pp, pk, ok := matchedPencilRoots(sys, cz, cp, k); ok {
		if z, p, g, perr := matchedDiscreteRoots(pz, pp, pk, dt); perr == nil {
			dz, dp, gain, err = z, p, g, nil
			cz, cp, k = pz, pp, pk
			transfer = false
		}
	}
	if err != nil {
		return nil, err
	}
	r, dcg := matchedReferenceOf(ref)
	if noise := r.err == nil && !r.zero && math.Abs(r.g0) <= 0x1p-52*r.terms; r.err != nil || noise {
		if root, ok := matchedDCRootOf(ref, noise, dcg); ok && root.order != 0 {
			return sys.discretizeMatchedAtDC(dt, ref, rational, cz, cp, k, transfer, root)
		}
	}
	if transfer {
		pz, pp := matchedPolish(rational.TF.Num[0][0], cz), matchedPolish(rational.TF.Den[0], cp)
		if z, p, g, perr := matchedDiscreteRoots(pz, pp, k, dt); perr == nil {
			cz, cp, dz, dp, gain = pz, pp, z, p, g
		}
	}
	v, base := matchedContinuousProduct(k, cz, cp), gain
	mapped := matchedMappedProduct(cz, cp, dz, dt, 0)
	gain, unverified := r.matchedGainAtFloatDC(v, mapped, base)
	realized, dc, dcOK, err := matchedCascadeDC(dz, dp, gain, dt)
	if err != nil {
		return nil, err
	}
	if r.err == nil && !r.zero && !r.dcAgrees(dc, dcOK) {
		refuted := matchedUnverified("DC gain of the result is not within the descriptor bar of DCGain")
		if stored := matchedDiscreteProduct(1, dz, dp, 0); stored != mapped {
			g, unv := r.matchedGainAtFloatDC(v, stored, base)
			if out, sdc, sok, serr := matchedCascadeDC(dz, dp, g, dt); serr == nil && r.dcAgrees(sdc, sok) {
				realized, refuted = out, unv
				if rescaled := complex(r.g0, 0) * stored / mapped; refuted == nil && !r.rootsAgree(rescaled) {
					refuted = matchedUnverified("DC product %v of the stored discrete roots is not within terms of DCGain %v", real(rescaled), r.g0)
				}
			}
		}
		if unverified == nil {
			unverified = refuted
		}
	}
	return matchedDiscreteResult(realized, sys, dt, unverified)
}

func matchedDiscreteResult(realized, sys *System, dt float64, unverified error) (*System, error) {
	out, err := newDelayConversionPolicy(dt).applyDiscreteDelayFields(realized, sys)
	if err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	out.StateName = nil
	return out, unverified
}

// matchedDiscreteRoots maps continuous roots to exp(root·dt), adds zeros at
// z = −1 up to relative degree one, and matches the gain at DC.
func matchedDiscreteRoots(cz, cp []complex128, k, dt float64) (dz, dp []complex128, gain float64, err error) {
	dz = make([]complex128, len(cz), max(len(cz), len(cp)-1))
	dp = make([]complex128, len(cp))
	for i, z := range cz {
		dz[i] = cmplx.Exp(z * complex(dt, 0))
	}
	for i, p := range cp {
		dp[i] = cmplx.Exp(p * complex(dt, 0))
	}
	for range max(len(cp)-len(cz)-1, 0) {
		dz = append(dz, -1)
	}
	scale, err := matchedContinuousGain(dz, dp, cz, cp, 1, dt)
	if err != nil {
		return nil, nil, 0, err
	}
	if scale == 0 {
		return nil, nil, 0, ErrSingularTransform
	}
	discrete, err := NewZPK(dz, dp, k/scale, dt)
	if err != nil {
		return nil, nil, 0, err
	}
	return dz, dp, discrete.Gain[0][0], nil
}

// matchedPencilRoots returns the poles eig(A, E), the invariant zeros and
// a gain fitted to the response of sys, as MATLAB zpk(ss) takes them;
// polynomial roots of the transfer function lose tight pole clusters. ok is
// false unless the roots are finite, as many as the transfer-function
// roots cz, cp with gain k (so the order and relative degree stay), and fit
// the response at a probe point no worse than those do.
func matchedPencilRoots(sys *System, cz, cp []complex128, k float64) (zeros, poles []complex128, gain float64, ok bool) {
	poles, err := sys.Poles()
	if err != nil || len(poles) == 0 || len(poles) != len(cp) {
		return nil, nil, 0, false
	}
	zeros, err = sys.zerosValue()
	if err != nil || len(zeros) != len(cz) {
		return nil, nil, 0, false
	}
	if poles, ok = matchedConjugateClosed(poles); !ok {
		return nil, nil, 0, false
	}
	if zeros, ok = matchedConjugateClosed(zeros); !ok {
		return nil, nil, 0, false
	}
	free := *sys
	free.Delay, free.InputDelay, free.OutputDelay = nil, nil, nil
	s0, s1 := matchedProbePoints(poles, zeros, cp, cz)
	g0, ok0 := matchedEvalModel(&free, s0)
	g1, ok1 := matchedEvalModel(&free, s1)
	if !ok0 || !ok1 {
		return nil, nil, 0, false
	}
	gain = real(g0 / matchedRootRatio(zeros, poles, s0))
	misfit := func(z, p []complex128, k float64) float64 {
		return cmplx.Abs(complex(k, 0)*matchedRootRatio(z, p, s1)-g1) / cmplx.Abs(g1)
	}
	if e := misfit(zeros, poles, gain); math.IsNaN(e) || e > math.Max(2*misfit(cz, cp, k), 1e-12) {
		return nil, nil, 0, false
	}
	return zeros, poles, gain, true
}

// matchedConjugateClosed pairs each root with positive imaginary part with
// the nearest unpaired root of negative imaginary part, as eigensolvers
// return them, and makes the pair exact conjugates.
func matchedConjugateClosed(roots []complex128) ([]complex128, bool) {
	out := slices.Clone(roots)
	used := make([]bool, len(out))
	for i, r := range out {
		if cmplx.IsNaN(r) || cmplx.IsInf(r) {
			return nil, false
		}
		if imag(r) <= 0 {
			continue
		}
		best := -1
		for j, q := range out {
			if !used[j] && imag(q) < 0 && (best < 0 || cmplx.Abs(q-cmplx.Conj(r)) < cmplx.Abs(out[best]-cmplx.Conj(r))) {
				best = j
			}
		}
		if best < 0 || cmplx.Abs(out[best]-cmplx.Conj(r)) > 1e-8*cmplx.Abs(r) {
			return nil, false
		}
		used[best] = true
		mid := (r + cmplx.Conj(out[best])) / 2
		out[i], out[best] = mid, cmplx.Conj(mid)
	}
	for j, q := range out {
		if imag(q) < 0 && !used[j] {
			return nil, false
		}
	}
	return out, true
}

func matchedEvalModel(sys *System, s complex128) (complex128, bool) {
	g, err := sys.EvalFr(s)
	if err != nil {
		return 0, false
	}
	v := g[0][0]
	if v == 0 || cmplx.IsNaN(v) || cmplx.IsInf(v) {
		return 0, false
	}
	return v, true
}

// matchedRootRatio is Π(s−zeros)/Π(s−poles), interleaved against overflow.
func matchedRootRatio(zeros, poles []complex128, s complex128) complex128 {
	r := complex(1, 0)
	for i, p := range poles {
		if i < len(zeros) {
			r *= (s - zeros[i]) / (s - p)
		} else {
			r /= s - p
		}
	}
	return r
}

// matchedProbePoints returns the two points, at the smallest nonzero root
// modulus times 1/2, 1 or 2 and angles from 0.25π to 0.75π, farthest from
// every root relative to their modulus. The matched gain is set at low
// frequency, where errors in stiff roots cancel as they do at DC.
func matchedProbePoints(sets ...[]complex128) (complex128, complex128) {
	var mags []float64
	for _, set := range sets {
		for _, r := range set {
			if a := cmplx.Abs(r); a > 0 {
				mags = append(mags, a)
			}
		}
	}
	rho := 1.0
	if len(mags) > 0 {
		rho = slices.Min(mags)
	}
	var best [2]complex128
	score := [2]float64{-1, -1}
	for _, f := range []float64{0.5, 1, 2} {
		for _, theta := range []float64{0.25, 0.4, 0.5, 0.6, 0.75} {
			s := cmplx.Rect(f*rho, theta*math.Pi)
			d := math.Inf(1)
			for _, set := range sets {
				for _, r := range set {
					d = math.Min(d, cmplx.Abs(s-r)/(f*rho))
				}
			}
			switch {
			case d > score[0]:
				best[1], score[1] = best[0], score[0]
				best[0], score[0] = s, d
			case d > score[1]:
				best[1], score[1] = s, d
			}
		}
	}
	return best[0], best[1]
}

func matchedDeflateArtificialZeros(numerator []float64) ([]float64, int) {
	count := 0
	for len(numerator) > 1 {
		quotient := make([]float64, len(numerator)-1)
		quotient[0] = numerator[0]
		norm := math.Abs(numerator[0])
		for i := 1; i < len(quotient); i++ {
			quotient[i] = numerator[i] - quotient[i-1]
			norm += math.Abs(numerator[i])
		}
		norm += math.Abs(numerator[len(numerator)-1])
		remainder := numerator[len(numerator)-1] - quotient[len(quotient)-1]
		if norm == 0 || math.Abs(remainder) > 1e-10*norm {
			break
		}
		numerator = quotient
		count++
	}
	return numerator, count
}

type matchedSection struct {
	p1, p2 complex128
	order  int
	zeros  []complex128
}

// matchedCascade realizes the conjugate-closed SISO model
// gain·Π(z−zeros)/Π(z−poles) as a cascade of first- and second-order
// sections with block upper-triangular A, so its eigenvalues are the given
// poles to rounding; a companion realization loses clustered poles. Models
// the cascade cannot hold in finite float64 keep the companion realization.
func matchedCascade(zeros, poles []complex128, gain, dt float64) (*System, error) {
	sys, _, _, err := matchedCascadeDC(zeros, poles, gain, dt)
	return sys, err
}

// matchedCascadeDC is matchedCascade with the DC gain of the cascade (at
// z = 1, or s = 0 for dt = 0) and its bound against the exact gain of the
// stored entries; ok is false for the companion realization or a section
// whose DC is not resolved.
func matchedCascadeDC(zeros, poles []complex128, gain, dt float64) (*System, matchedDC, bool, error) {
	if sys, dc, ok := matchedCascadeSections(zeros, poles, gain, dt); sys != nil {
		return sys, dc, ok, nil
	}
	discrete, err := NewZPK(zeros, poles, gain, dt)
	if err != nil {
		return nil, matchedDC{}, false, err
	}
	res, err := discrete.StateSpace()
	if err != nil {
		return nil, matchedDC{}, false, err
	}
	if len(zeros) == 0 && len(poles) == 0 {
		return res.Sys, mddFloat(gain), true, nil
	}
	return res.Sys, matchedDC{}, false, nil
}

func matchedCascadeFinite(zeros, poles []complex128, gain, dt float64) *System {
	sys, _, _ := matchedCascadeSections(zeros, poles, gain, dt)
	return sys
}

// matchedCascadeSections realizes the cascade with the gain on B and D, so
// every other entry is a copy of a section entry (B and D factors 0 or 1)
// and the stored model is gain·ΠSᵢ exactly; discrete two-zero sections
// take realizeAtOne. It also returns gain·ΠSᵢ(x0) by matchedSectionDC.
func matchedCascadeSections(zeros, poles []complex128, gain, dt float64) (*System, matchedDC, bool) {
	if len(zeros) > len(poles) || len(poles) == 0 || !finiteSlice([]float64{gain}) {
		return nil, matchedDC{}, false
	}
	for _, r := range zeros {
		if cmplx.IsNaN(r) || cmplx.IsInf(r) {
			return nil, matchedDC{}, false
		}
	}
	for _, r := range poles {
		if cmplx.IsNaN(r) || cmplx.IsInf(r) {
			return nil, matchedDC{}, false
		}
	}
	x0 := 0.0
	if dt > 0 {
		x0 = 1
	}
	dc, dcOK := mddFloat(gain), true
	n := len(poles)
	a := make([]float64, n*n)
	b := make([]float64, n)
	c := make([]float64, n)
	d := 1.0
	off := n
	for _, s := range matchedSections(zeros, poles) {
		as, bs, cs, ds := s.realize()
		if dt > 0 {
			as, bs, cs, ds = s.realizeAtOne()
		}
		if dcOK {
			var sd matchedDC
			if sd, dcOK = matchedSectionDC(as, bs, cs, ds, x0); dcOK {
				dc = matchedDCTimes(dc, sd)
			}
		}
		q := s.order
		off -= q
		for i := range q {
			row := (off + i) * n
			for j := range q {
				a[row+off+j] = as[i*q+j]
			}
			for j := off + q; j < n; j++ {
				a[row+j] = bs[i] * c[j]
			}
			b[off+i] = bs[i] * d
		}
		for j := off + q; j < n; j++ {
			c[j] *= ds
		}
		copy(c[off:off+q], cs)
		d *= ds
	}
	for j := range b {
		b[j] *= gain
	}
	d *= gain
	if !finiteSlice(a) || !finiteSlice(b) || !finiteSlice(c) || !finiteSlice([]float64{d}) {
		return nil, matchedDC{}, false
	}
	sys, err := newDerived(mat.NewDense(n, n, a), mat.NewDense(n, 1, b), mat.NewDense(1, n, c), mat.NewDense(1, 1, []float64{d}), dt)
	if err != nil {
		return nil, matchedDC{}, false
	}
	return sys, dc, dcOK
}

// matchedSections groups conjugate pairs and real roots into sections of
// order at most two, each with no more zeros than poles. Each zero joins
// the nearest pole in log z (the continuous root times T), so a section's
// gain stays moderate; a complex zero pair without a free complex pole pair
// takes the two nearest real poles.
func matchedSections(zeros, poles []complex128) []matchedSection {
	var realPoles []float64
	var realZeros, pairZeros []complex128
	sections := make([]matchedSection, 0, len(poles))
	for _, p := range poles {
		switch {
		case imag(p) == 0:
			realPoles = append(realPoles, real(p))
		case imag(p) > 0:
			sections = append(sections, matchedSection{p1: p, p2: cmplx.Conj(p), order: 2})
		}
	}
	for _, z := range zeros {
		switch {
		case imag(z) == 0:
			realZeros = append(realZeros, z)
		case imag(z) > 0:
			pairZeros = append(pairZeros, z)
		}
	}
	sort.Float64s(realPoles)
	used := make([]bool, len(realPoles))
	nearestReal := func(z complex128) int {
		best := -1
		for i, p := range realPoles {
			if !used[i] && (best < 0 || matchedLogDistance(z, complex(p, 0)) < matchedLogDistance(z, complex(realPoles[best], 0))) {
				best = i
			}
		}
		return best
	}
	for _, z := range pairZeros {
		best, dist := -1, math.Inf(1)
		for i, s := range sections {
			if len(s.zeros) == 0 && imag(s.p1) != 0 {
				if e := matchedLogDistance(z, s.p1); best < 0 || e < dist {
					best, dist = i, e
				}
			}
		}
		if free := len(realPoles) - countTrue(used); free >= 2 {
			i := nearestReal(z)
			used[i] = true
			k := nearestReal(z)
			used[i] = false
			if e := math.Max(matchedLogDistance(z, complex(realPoles[i], 0)), matchedLogDistance(z, complex(realPoles[k], 0))); best < 0 || e < dist {
				used[i], used[k] = true, true
				sections = append(sections, matchedSection{p1: complex(realPoles[min(i, k)], 0), p2: complex(realPoles[max(i, k)], 0), order: 2})
				best = len(sections) - 1
			}
		}
		sections[best].zeros = []complex128{z, cmplx.Conj(z)}
	}
	for i, p := range realPoles {
		if !used[i] {
			sections = append(sections, matchedSection{p1: complex(p, 0), order: 1})
		}
	}
	for _, z := range realZeros {
		best, dist := -1, math.Inf(1)
		for i, s := range sections {
			if len(s.zeros) >= s.order {
				continue
			}
			e := matchedLogDistance(z, s.p1)
			if s.order == 2 {
				e = math.Min(e, matchedLogDistance(z, s.p2))
			}
			if best < 0 || e < dist {
				best, dist = i, e
			}
		}
		sections[best].zeros = append(sections[best].zeros, z)
	}
	return sections
}

func matchedLogDistance(z, p complex128) float64 {
	if z == p {
		return 0
	}
	if z == 0 || p == 0 {
		return math.MaxFloat64
	}
	return cmplx.Abs(cmplx.Log(z) - cmplx.Log(p))
}

func countTrue(b []bool) int {
	n := 0
	for _, v := range b {
		if v {
			n++
		}
	}
	return n
}

// realize returns (A, B, C, D) of the section with monic numerator and
// denominator: A = [a b; −b a] for a pole pair a±ib, [p1 1; 0 p2] for two
// real poles, B = e₂, and C from the remainder of the numerator.
func (s matchedSection) realize() (a, b, c []float64, d float64) {
	if s.order == 1 {
		p := real(s.p1)
		if len(s.zeros) == 1 {
			return []float64{p}, []float64{1}, []float64{p - real(s.zeros[0])}, 1
		}
		return []float64{p}, []float64{1}, []float64{1}, 0
	}
	var r1, r0 float64
	switch len(s.zeros) {
	case 2:
		z1, z2 := s.zeros[0], s.zeros[1]
		r1 = real((s.p1 - z1) + (s.p2 - z2))
		r0 = real(z1*(z2-s.p2) + s.p2*(z1-s.p1))
		d = 1
	case 1:
		r1, r0 = 1, -real(s.zeros[0])
	default:
		r0 = 1
	}
	b = []float64{0, 1}
	if imag(s.p1) != 0 {
		re, im := real(s.p1), math.Abs(imag(s.p1))
		return []float64{re, im, -im, re}, b, []float64{(r0 + re*r1) / im, r1}, d
	}
	p1, p2 := real(s.p1), real(s.p2)
	return []float64{p1, 1, 0, p2}, b, []float64{r0 + p1*r1, r1}, d
}

// realizeAtOne is realize for a discrete section, with the residual
// r(z) = num − den of a two-zero section set from
// r(1) = (1−z₁)(1−z₂) − (1−p₁)(1−p₂), formed from the differences 1 − root,
// so the section's gain at z = 1 does not cancel r₁ + r₀.
func (s matchedSection) realizeAtOne() (a, b, c []float64, d float64) {
	if s.order != 2 || len(s.zeros) != 2 {
		return s.realize()
	}
	z1, z2 := s.zeros[0], s.zeros[1]
	r1 := real((s.p1 - z1) + (s.p2 - z2))
	r1at := real((1-z1)*(1-z2) - (1-s.p1)*(1-s.p2))
	b = []float64{0, 1}
	if imag(s.p1) != 0 {
		re, im := real(s.p1), math.Abs(imag(s.p1))
		return []float64{re, im, -im, re}, b, []float64{(r1at - (1-re)*r1) / im, r1}, 1
	}
	p1, p2 := real(s.p1), real(s.p2)
	return []float64{p1, 1, 0, p2}, b, []float64{r1at - (1-p1)*r1, r1}, 1
}
