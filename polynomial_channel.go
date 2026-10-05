package controlsys

import "fmt"

type rationalChannel struct {
	zeros []complex128
	poles []complex128
	gain  float64
}

func newRationalChannel(zeros, poles []complex128, gain float64) rationalChannel {
	return rationalChannel{zeros: zeros, poles: poles, gain: gain}
}

func rationalChannelFromPolynomials(num, den []float64) (rationalChannel, error) {
	num = trimLeadingFloatZeros(num)
	if len(num) == 1 && num[0] == 0 {
		return rationalChannel{gain: 0}, nil
	}
	zeros, err := Poly(num).Roots()
	if err != nil {
		return rationalChannel{}, err
	}
	poles, err := Poly(den).Roots()
	if err != nil {
		return rationalChannel{}, err
	}
	return rationalChannel{
		zeros: zeros,
		poles: poles,
		gain:  channelPolynomialGain(num, den),
	}, nil
}

func (ch rationalChannel) eval(s complex128) complex128 {
	if ch.gain == 0 {
		return 0
	}
	nz := len(ch.zeros)
	np := len(ch.poles)
	h := complex(ch.gain, 0)

	shared := min(nz, np)
	for k := range shared {
		h *= (s - ch.zeros[k]) / (s - ch.poles[k])
	}
	for k := shared; k < nz; k++ {
		h *= s - ch.zeros[k]
	}
	for k := shared; k < np; k++ {
		h /= s - ch.poles[k]
	}
	return h
}

// evalAt is eval at pt with each factor s − v formed as pt.pencilAt(v)/c,
// c = −q, so a discrete frequency point stays on the unit circle next to a
// lightly damped pole.
func (ch rationalChannel) evalAt(pt frequencyPoint) complex128 {
	if pt.isPlain() {
		return ch.eval(pt.p)
	}
	if ch.gain == 0 {
		return 0
	}
	nz := len(ch.zeros)
	np := len(ch.poles)
	h := complex(ch.gain, 0)

	shared := min(nz, np)
	for k := range shared {
		h *= pt.pencilAt(ch.zeros[k]) / pt.pencilAt(ch.poles[k])
	}
	for k := shared; k < nz; k++ {
		h *= pt.pencilAt(ch.zeros[k])
	}
	for k := shared; k < np; k++ {
		h /= pt.pencilAt(ch.poles[k])
	}
	c := pt.scale()
	for range nz - np {
		h /= c
	}
	for range np - nz {
		h *= c
	}
	return h
}

func (ch rationalChannel) numeratorForCommonPoles(commonPoles []complex128) []float64 {
	if ch.gain == 0 {
		return []float64{0}
	}
	missing := poleSetDifference(commonPoles, ch.poles)
	poly := polyFromComplexRoots(sortConjugatePairs(ch.zeros))
	if len(missing) > 0 {
		missingPoly := polyFromComplexRoots(sortConjugatePairs(missing))
		poly = poly.Mul(missingPoly)
	}
	return []float64(poly.Scale(ch.gain))
}

func trimLeadingFloatZeros(p []float64) []float64 {
	i := 0
	for i < len(p) && p[i] == 0 {
		i++
	}
	if i == len(p) {
		return []float64{0}
	}
	return p[i:]
}

func channelPolynomialGain(num, den []float64) float64 {
	num = trimLeadingFloatZeros(num)
	den = trimLeadingFloatZeros(den)
	if len(num) == 1 && num[0] == 0 {
		return 0
	}
	return num[0] / den[0]
}

func validateTransferChannelShape(tf *TransferFunc) (p, m int, err error) {
	if tf == nil {
		return 0, 0, fmt.Errorf("TransferFunc: nil model: %w", ErrDimensionMismatch)
	}
	p = len(tf.Den)
	if p == 0 {
		return 0, 0, nil
	}
	if len(tf.Num) != p {
		return 0, 0, fmt.Errorf("TransferFunc: numerator has %d rows, want %d: %w", len(tf.Num), p, ErrDimensionMismatch)
	}
	m = len(tf.Num[0])
	if m == 0 {
		return 0, 0, fmt.Errorf("TransferFunc: zero input channels: %w", ErrDimensionMismatch)
	}
	if tf.Delay != nil && len(tf.Delay) != p {
		return 0, 0, fmt.Errorf("TransferFunc: delay has %d rows, want %d: %w", len(tf.Delay), p, ErrDimensionMismatch)
	}
	for i := range p {
		if len(tf.Den[i]) == 0 {
			return 0, 0, fmt.Errorf("TransferFunc: row %d has empty denominator: %w", i, ErrDimensionMismatch)
		}
		if len(tf.Num[i]) != m {
			return 0, 0, fmt.Errorf("TransferFunc: row %d has %d numerator channels, want %d: %w", i, len(tf.Num[i]), m, ErrDimensionMismatch)
		}
		if tf.Delay != nil && len(tf.Delay[i]) != m {
			return 0, 0, fmt.Errorf("TransferFunc: row %d has %d delays, want %d: %w", i, len(tf.Delay[i]), m, ErrDimensionMismatch)
		}
		for j := range m {
			if len(tf.Num[i][j]) == 0 {
				return 0, 0, fmt.Errorf("TransferFunc: channel (%d,%d) has empty numerator: %w", i, j, ErrDimensionMismatch)
			}
		}
	}
	return p, m, nil
}

func validateZPKChannelShape(z *ZPK) (p, m int, err error) {
	if z == nil {
		return 0, 0, fmt.Errorf("ZPK: nil model: %w", ErrDimensionMismatch)
	}
	p = len(z.Gain)
	if p == 0 {
		return 0, 0, fmt.Errorf("ZPK: zero output channels: %w", ErrDimensionMismatch)
	}
	m = len(z.Gain[0])
	if m == 0 {
		return 0, 0, fmt.Errorf("ZPK: zero input channels: %w", ErrDimensionMismatch)
	}
	if len(z.Zeros) != p || len(z.Poles) != p {
		return 0, 0, fmt.Errorf("ZPK: channel rows do not match gain rows: %w", ErrDimensionMismatch)
	}
	for i := range p {
		if len(z.Gain[i]) != m || len(z.Zeros[i]) != m || len(z.Poles[i]) != m {
			return 0, 0, fmt.Errorf("ZPK: row %d channel count mismatch: %w", i, ErrDimensionMismatch)
		}
	}
	return p, m, nil
}
