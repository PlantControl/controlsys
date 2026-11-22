package controlsys

import "math"

// twoProductLo returns a·b − h exactly for h = float64(a·b), the product
// rounded explicitly so that it does not fuse (TwoProduct): a fused
// multiply-add in hardware, else Dekker's split, which is exact unless a
// split overflows (|a| or |b| ≥ 2⁹⁹⁵) or the product's error underflows,
// where the software math.FMA is used.
func twoProductLo(a, b, h float64) float64 {
	if hasFMA {
		return math.FMA(a, b, -h)
	}
	if aa, ab, ah := math.Abs(a), math.Abs(b), math.Abs(h); aa >= 0x1p995 || ab >= 0x1p995 || ah < 0x1p-969 {
		return math.FMA(a, b, -h)
	}
	return dekkerProductLo(a, b, h)
}

// dekkerProductLo is twoProductLo by Dekker's split, for operands and
// products in range.
func dekkerProductLo(a, b, h float64) float64 {
	const split = 134217729
	c := float64(split * a)
	a1 := float64(c - float64(c-a))
	a2 := float64(a - a1)
	c = float64(split * b)
	b1 := float64(c - float64(c-b))
	b2 := float64(b - b1)
	return float64(a2*b2) - float64(float64(float64(h-float64(a1*b1))-float64(a2*b1))-float64(a1*b2))
}
