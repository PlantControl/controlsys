package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
)

// d2cMatched uses principal logarithms of finite roots. All zeros at z=-1
// are interpreted as the artificial zeros of matched discretization and removed.
// Other nonpositive real roots are rejected rather than inventing a complex model.
// Gain matches the leading low-frequency term, including poles or zeros at DC.
func (sys *System) d2cMatched() (*System, error) {
	if sys.IsContinuous() {
		return nil, fmt.Errorf("D2C matched: %w", ErrWrongDomain)
	}
	if sys.Dt <= 0 || math.IsNaN(sys.Dt) || math.IsInf(sys.Dt, 0) {
		return nil, ErrInvalidSampleTime
	}
	_, m, p := sys.Dims()
	if m != 1 || p != 1 {
		return nil, fmt.Errorf("D2C matched: %w", ErrNotSISO)
	}
	if sys.IsDescriptor() {
		return nil, ErrDescriptorUnsupported
	}
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("D2C matched: internal delays: %w", ErrFeedbackDelay)
	}
	result, err := sys.rationalTransferFunction(nil)
	if err != nil {
		return nil, fmt.Errorf("D2C matched: %w", err)
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
	newDelayConversionPolicy(sys.Dt, 0, 0).applyContinuousDelayFields(out, sys)
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
			return nil, fmt.Errorf("D2C matched: nonpositive real root %v has no supported real principal logarithm: %w", root, ErrSingularTransform)
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
			return nil, fmt.Errorf("D2C matched: non-real reconstruction: %w", ErrConjugatePairs)
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
		return nil, fmt.Errorf("DiscretizeMatched: %w", ErrWrongDomain)
	}
	if dt <= 0 || math.IsNaN(dt) || math.IsInf(dt, 0) {
		return nil, ErrInvalidSampleTime
	}
	_, m, p := sys.Dims()
	if m != 1 || p != 1 {
		return nil, ErrNotSISO
	}
	if sys.IsDescriptor() {
		return nil, ErrDescriptorUnsupported
	}
	if sys.HasInternalDelay() {
		return nil, fmt.Errorf("DiscretizeMatched: internal delays: %w", ErrFeedbackDelay)
	}
	rational, err := sys.rationalTransferFunction(nil)
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
	realized, err := discrete.StateSpace()
	if err != nil {
		return nil, err
	}
	out, err := newDelayConversionPolicy(dt, 0, 0).applyDiscreteDelayFields(realized.Sys, sys)
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
