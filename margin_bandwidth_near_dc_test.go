package controlsys

import (
	"math"
	"math/big"
	"testing"
)

const bandwidthNearDCPrec = 512

// bandwidthNearDCGainSq is |H(z)|² of the upper-triangular realization at
// z = (1+jν)/(1−jν) on the unit circle, by back-substitution in big.Float
// from the float64 data, without rounding the pole distance z − a_ii.
func bandwidthNearDCGainSq(a, b, c []float64, nu *big.Float) *big.Float {
	n := len(b)
	f := func(v float64) *big.Float { return new(big.Float).SetPrec(bandwidthNearDCPrec).SetFloat64(v) }
	nf := func() *big.Float { return new(big.Float).SetPrec(bandwidthNearDCPrec) }
	nu2 := nf().Mul(nu, nu)
	den := nf().Add(f(1), nu2)
	zr := nf().Quo(nf().Sub(f(1), nu2), den)
	zi := nf().Quo(nf().Mul(f(2), nu), den)
	xr, xi := make([]*big.Float, n), make([]*big.Float, n)
	yr, yi := nf(), nf()
	for i := n - 1; i >= 0; i-- {
		rr, ri := f(b[i]), nf()
		for j := i + 1; j < n; j++ {
			aij := f(a[i*n+j])
			rr.Add(rr, nf().Mul(aij, xr[j]))
			ri.Add(ri, nf().Mul(aij, xi[j]))
		}
		dr := nf().Sub(zr, f(a[i*n+i]))
		d2 := nf().Add(nf().Mul(dr, dr), nf().Mul(zi, zi))
		xr[i] = nf().Quo(nf().Add(nf().Mul(rr, dr), nf().Mul(ri, zi)), d2)
		xi[i] = nf().Quo(nf().Sub(nf().Mul(ri, dr), nf().Mul(rr, zi)), d2)
		yr.Add(yr, nf().Mul(f(c[i]), xr[i]))
		yi.Add(yi, nf().Mul(f(c[i]), xi[i]))
	}
	return nf().Add(nf().Mul(yr, yr), nf().Mul(yi, yi))
}

// bandwidthNearDCOracle is the first frequency where |H| falls dbDrop dB below
// |H(1)|, by bisection on ν between 0 and a νHi known to lie past the crossing.
func bandwidthNearDCOracle(a, b, c []float64, dt, dbDrop, nuHi float64) (bw, dc float64) {
	nf := func() *big.Float { return new(big.Float).SetPrec(bandwidthNearDCPrec) }
	dc2 := bandwidthNearDCGainSq(a, b, c, nf())
	dcBig := nf().Sqrt(dc2)
	dc, _ = dcBig.Float64()
	target := nf().Mul(dc2, nf().SetFloat64(math.Pow(10, dbDrop/10)))
	lo, hi := nf(), nf().SetFloat64(nuHi)
	half := nf().SetFloat64(0.5)
	for range 400 {
		mid := nf().Mul(nf().Add(lo, hi), half)
		if bandwidthNearDCGainSq(a, b, c, mid).Cmp(target) > 0 {
			lo = mid
		} else {
			hi = mid
		}
	}
	nu, _ := hi.Float64()
	return 2 * math.Atan(nu) / dt, dc
}

// A discrete pole within rounding of z = 1 has a finite DCGain, but the double
// solve snaps the response near it to the DC pole's +Inf: FreqResponse(0) is
// +Inf, the pencil root folds to ω = 0, and the crossing was lost, giving
// Bandwidth +Inf with a nil error (ER7QTR). Bandwidth must agree with DCGain,
// as the continuous 1e14/(s+1) does.
func TestBandwidthDiscretePoleWithinRoundingOfDC(t *testing.T) {
	const nearOne = 0.9999999999999998
	sqrt2Drop := -10 * math.Log10(2)
	tests := []struct {
		name    string
		a, b, c []float64
		dt      float64
		nuHi    float64
	}{
		{"first order", []float64{nearOne}, []float64{0.04}, []float64{1}, 0.04, 1e-3},
		{"first order 3 ulp", []float64{1 - 3*0x1p-53}, []float64{-2.5}, []float64{0.3}, 1, 1e-3},
		{
			"second order non-normal",
			[]float64{nearOne, 0.3, 0, 0.5},
			[]float64{0.04, 0.7},
			[]float64{1, -0.2},
			0.04, 1e-3,
		},
		{
			"third order non-normal",
			[]float64{0.5, 0.4, -0.1, 0, nearOne, 0.3, 0, 0, -0.25},
			[]float64{0.2, 0.04, 0.7},
			[]float64{1, -0.2, 0.6},
			0.1, 1e-3,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.b)
			sys, err := NewFromSlices(n, 1, 1, tc.a, tc.b, tc.c, []float64{0}, tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			for _, drop := range []float64{-3, sqrt2Drop, -20} {
				want, wantDC := bandwidthNearDCOracle(tc.a, tc.b, tc.c, tc.dt, drop, tc.nuHi)
				dc, err := sys.DCGain()
				if err != nil {
					t.Fatal(err)
				}
				if got := math.Abs(dc.At(0, 0)); math.Abs(got-wantDC) > 1e-12*wantDC {
					t.Fatalf("|DCGain| = %.17g, exact %.17g", got, wantDC)
				}
				bw, err := Bandwidth(sys, drop)
				if err != nil {
					t.Fatalf("Bandwidth(%g) error: %v", drop, err)
				}
				if math.Abs(bw-want) > 1e-9*want {
					t.Errorf("Bandwidth(%g) = %.17g, exact %.17g", drop, bw, want)
				}
			}
		})
	}
}

// The first-order case against the closed form of |H(e^{jθ})| = |H(1)|·g:
// (1−a)²·(1/g²−1) = 4a·sin²(θ/2); g = 1/√2 drops the factor 1/g²−1.
func TestBandwidthDiscretePoleWithinRoundingOfDCClosedForm(t *testing.T) {
	a, dt := 0.9999999999999998, 0.04
	sys, err := NewFromSlices(1, 1, 1, []float64{a}, []float64{0.04}, []float64{1}, []float64{0}, dt)
	if err != nil {
		t.Fatal(err)
	}
	for _, drop := range []float64{-3, -10 * math.Log10(2)} {
		want := 2 * math.Asin((1-a)*math.Sqrt((math.Pow(10, -drop/10)-1)/(4*a))) / dt
		bw, err := Bandwidth(sys, drop)
		if err != nil {
			t.Fatal(err)
		}
		if math.Abs(bw-want) > 1e-9*want {
			t.Errorf("Bandwidth(%g) = %.17g, closed form %.17g", drop, bw, want)
		}
	}
}

// Reference for the discrete test: the same DC gain with a resolvable pole.
func TestBandwidthLargeDCGainContinuousReference(t *testing.T) {
	sys, err := NewFromSlices(1, 1, 1, []float64{-1}, []float64{1e14}, []float64{1}, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	bw, err := Bandwidth(sys, -3)
	if err != nil {
		t.Fatal(err)
	}
	if want := math.Sqrt(math.Pow(10, 0.3) - 1); math.Abs(bw-want) > 1e-12*want {
		t.Errorf("Bandwidth = %.17g, want %.17g", bw, want)
	}
}
