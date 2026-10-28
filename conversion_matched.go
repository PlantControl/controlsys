package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"sort"

	"plantcontrol.org/v1/gonum/mat"
)

// d2cMatched uses principal logarithms of finite roots. All zeros at z=-1
// are interpreted as the artificial zeros of matched discretization and removed.
// Other nonpositive real roots are rejected rather than inventing a complex model.
// Gain matches the leading low-frequency term, including poles or zeros at DC.
func (sys *System) d2cMatched() (*System, error) {
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
	dz, dp := disc.Zeros[0][0], disc.Poles[0][0]
	cz, err := matchedLogRoots(dz, sys.Dt)
	if err != nil {
		return nil, err
	}
	cp, err := matchedLogRoots(dp, sys.Dt)
	if err != nil {
		return nil, err
	}
	for range artificial {
		dz = append(dz, -1)
	}
	gain, err := matchedContinuousGain(dz, dp, cz, cp, disc.Gain[0][0], sys.Dt)
	if err != nil {
		return nil, err
	}
	model, err := NewZPK(cz, cp, gain, 0)
	if err != nil {
		return nil, err
	}
	ss, err := model.StateSpace()
	if err != nil {
		return nil, err
	}
	out := ss.Sys
	newDelayConversionPolicy(sys.Dt).applyContinuousDelayFields(out, sys)
	propagateNames(out, sys)
	out.StateName = nil
	return out, nil
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

func matchedContinuousGain(dz, dp, cz, cp []complex128, gain, dt float64) (float64, error) {
	discrete := complex(gain, 0)
	continuous := complex(1, 0)
	for _, z := range dz {
		if cmplx.Abs(z-1) <= 1e-9 {
			discrete *= complex(dt, 0)
		} else {
			discrete *= 1 - z
		}
	}
	for _, p := range dp {
		if cmplx.Abs(p-1) <= 1e-9 {
			discrete /= complex(dt, 0)
		} else {
			discrete /= 1 - p
		}
	}
	for _, z := range cz {
		if cmplx.Abs(z) > 1e-9/dt {
			continuous *= -z
		}
	}
	for _, p := range cp {
		if cmplx.Abs(p) > 1e-9/dt {
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

func (sys *System) discretizeMatched(dt float64) (*System, error) {
	if sys.IsDiscrete() {
		return nil, fmt.Errorf("system already discrete: %w", ErrWrongDomain)
	}
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, ErrInvalidSampleTime
	}
	_, m, p := sys.Dims()
	if m != 1 || p != 1 {
		return nil, ErrNotSISO
	}
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("matched: internal delays: %w", ErrFeedbackDelay)
	}
	if sys.IsDescriptor() {
		if _, stiff := stiffDescriptorParts(sys); !stiff {
			return nil, ErrDescriptorUnsupported
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
	cz, cp := source.Zeros[0][0], source.Poles[0][0]
	dz := make([]complex128, len(cz), max(len(cz), len(cp)-1))
	dp := make([]complex128, len(cp))
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
		return nil, err
	}
	if scale == 0 {
		return nil, ErrSingularTransform
	}
	discrete, err := NewZPK(dz, dp, source.Gain[0][0]/scale, dt)
	if err != nil {
		return nil, err
	}
	realized, err := matchedCascade(dz, dp, discrete.Gain[0][0], dt)
	if err != nil {
		return nil, err
	}
	out, err := newDelayConversionPolicy(dt).applyDiscreteDelayFields(realized, sys)
	if err != nil {
		return nil, err
	}
	propagateNames(out, sys)
	out.StateName = nil
	return out, nil
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
	if sys := matchedCascadeFinite(zeros, poles, gain, dt); sys != nil {
		return sys, nil
	}
	discrete, err := NewZPK(zeros, poles, gain, dt)
	if err != nil {
		return nil, err
	}
	res, err := discrete.StateSpace()
	if err != nil {
		return nil, err
	}
	return res.Sys, nil
}

func matchedCascadeFinite(zeros, poles []complex128, gain, dt float64) *System {
	if len(zeros) > len(poles) || len(poles) == 0 || !finiteSlice([]float64{gain}) {
		return nil
	}
	for _, r := range zeros {
		if cmplx.IsNaN(r) || cmplx.IsInf(r) {
			return nil
		}
	}
	for _, r := range poles {
		if cmplx.IsNaN(r) || cmplx.IsInf(r) {
			return nil
		}
	}
	n := len(poles)
	a := make([]float64, n*n)
	b := make([]float64, n)
	c := make([]float64, n)
	d := 1.0
	off := n
	for _, s := range matchedSections(zeros, poles) {
		as, bs, cs, ds := s.realize()
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
	for j := range c {
		c[j] *= gain
	}
	d *= gain
	if !finiteSlice(a) || !finiteSlice(b) || !finiteSlice(c) || !finiteSlice([]float64{d}) {
		return nil
	}
	sys, err := New(mat.NewDense(n, n, a), mat.NewDense(n, 1, b), mat.NewDense(1, n, c), mat.NewDense(1, 1, []float64{d}), dt)
	if err != nil {
		return nil
	}
	return sys
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
