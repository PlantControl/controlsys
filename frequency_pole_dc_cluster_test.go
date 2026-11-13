package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// dcPairModel is the discrete pair 1 ± p of A = [1, 2p; p/2, 1], inside its
// own rounding disk 2.2e-16 of z = 1 for p ≤ 1e-290 (A2OX65, found with
// CWMCXC): G(1) of the stored data is about −6e289.
func dcPairModel(t *testing.T, p float64) (*System, func(theta float64) complex128) {
	t.Helper()
	a := []float64{1, 2 * p, p / 2, 1}
	b := []float64{1, 0.5}
	c := []float64{0.7e-10 * p * 1e300, -0.2e-10 * p * 1e300}
	sys, err := NewFromSlices(2, 1, 1, a, b, c, []float64{0.01}, 1)
	if err != nil {
		t.Fatal(err)
	}
	return sys, func(theta float64) complex128 {
		s := math.Sin(theta / 2)
		w := complex(-2*s*s, math.Sin(theta))
		num := complex(c[0], 0)*(w*complex(b[0], 0)+complex(a[1]*b[1], 0)) + complex(c[1], 0)*(complex(a[2]*b[0], 0)+w*complex(b[1], 0))
		return 0.01 + num/(w*w)
	}
}

// TestFreqResponseDCPoleClusterCertified: the pole limit's tuned residue
// test read the residue 6e-16 as rounding and returned ≈ D with a nil
// error at z = 1 and within the pair's rounding disk. The pair cannot be
// told from z = 1 by samples and I − A is exactly nonsingular, so the
// response at z = 1 is the stored model's exact one, as DCGain certifies
// it (−6e289), and a point inside the disk off z = 1 is its resolved
// double-double response or ErrAccuracyUnverified (A2OX65 CORE-CONFLICT
// ruling); outside the disk the solved response stands.
func TestFreqResponseDCPoleClusterCertified(t *testing.T) {
	for _, p := range []float64{1e-305, 1e-307} {
		sys, oracle := dcPairModel(t, p)
		dc, err := sys.DCGain()
		if err != nil {
			t.Fatalf("p=%g DCGain: %v", p, err)
		}
		fr, err := sys.FreqResponse([]float64{0})
		if err != nil {
			t.Fatalf("p=%g FreqResponse(0): %v", p, err)
		}
		if g, want := fr.At(0, 0, 0), complex(dc.At(0, 0), 0); g != want {
			t.Errorf("p=%g FreqResponse(0) = %v, want DCGain %v", p, g, want)
		}
		fr, err = sys.FreqResponse([]float64{1e-300})
		if err != nil {
			if !errors.Is(err, ErrAccuracyUnverified) {
				t.Errorf("p=%g FreqResponse(1e-300): %v, want ErrAccuracyUnverified", p, err)
			}
		} else if g, want := fr.At(0, 0, 0), oracle(1e-300); !(cmplx.Abs(g-want) <= 1e-9*cmplx.Abs(want)) {
			t.Errorf("p=%g FreqResponse(1e-300) = %v, want %v or ErrAccuracyUnverified", p, g, want)
		}
		fr, err = sys.FreqResponse([]float64{1e-14})
		if err != nil {
			t.Fatalf("p=%g FreqResponse(1e-14): %v", p, err)
		}
		if g, want := fr.At(0, 0, 0), oracle(1e-14); !(cmplx.Abs(g-want) <= 1e-9*cmplx.Abs(want)) {
			t.Errorf("p=%g FreqResponse(1e-14) = %v, want %v", p, g, want)
		}
	}
}

// TestFreqResponseDCPoleLimitZeroResidue: the integrator at s = 0 is
// unobserved (C·x = 0), so its residue at DC is exactly 0, while the
// cluster grown beside it, the pole at −0.1 coupled by 0.3, has a nonzero
// residue; the certified pole limit samples at the integrator's own scale
// and returns the finite DC value D + C₂₃(−A₂₃)⁻¹B₂₃ = 11.632 (A2OX65).
func TestFreqResponseDCPoleLimitZeroResidue(t *testing.T) {
	a := []float64{0, 0.3, 0.5, 0, -0.1, 0.25, 0, 0, -5}
	sys, err := NewFromSlices(3, 1, 1, a, []float64{0.7, 1.3, -0.4}, []float64{0, 0.9, 1.1}, []float64{0.2}, 0)
	if err != nil {
		t.Fatal(err)
	}
	fr, err := sys.FreqResponse([]float64{0})
	if err != nil {
		t.Fatal(err)
	}
	if g := fr.At(0, 0, 0); !(cmplx.Abs(g-11.632) <= 1e-12*11.632) {
		t.Errorf("FreqResponse(0) = %v, want 11.632", g)
	}
}

// TestAllMarginDCPoleClusterUnverified: the gain crossover of dcPairModel
// lies at 6.0003e-16 (p = 1e-305) and 6.0003e-18 (p = 1e-307), where
// AllMargin returned no crossing with a nil error. The bracket from the
// certified DC gain does not close on solved responses, the double-double
// response next to the pair being no bracket end, so both are
// ErrAccuracyUnverified (A2OX65).
func TestAllMarginDCPoleClusterUnverified(t *testing.T) {
	for _, p := range []float64{1e-305, 1e-307} {
		sys, _ := dcPairModel(t, p)
		if res, err := AllMargin(sys); !errors.Is(err, ErrAccuracyUnverified) {
			t.Errorf("p=%g AllMargin = %v, %v; want ErrAccuracyUnverified", p, res, err)
		}
	}
}

// TestBandwidthNyquistGap: A = [−1, 2q; q/2, −1] puts the dip of |G| to 0 at
// the exact z = −1 within about q of it. For q ≤ 1e-17 it lies between
// fl(π) and π, which no float64 ω reaches; nyquistResponse certifies |G(−1)|
// below the threshold, so the crossing is fl(π)/Dt (A2OX65). The oracle for
// q = 1e-12 is a 320-bit scan with exact π.
func TestBandwidthNyquistGap(t *testing.T) {
	for _, tc := range []struct{ q, want float64 }{
		{1e-12, 3.1415926535882361},
		{1e-17, math.Pi},
		{1e-20, math.Pi},
	} {
		sys, err := NewFromSlices(2, 1, 1, []float64{-1, 2 * tc.q, tc.q / 2, -1}, []float64{1, 0}, []float64{0, 2 * tc.q}, []float64{1}, 1)
		if err != nil {
			t.Fatal(err)
		}
		bw, err := Bandwidth(sys, -3)
		if err != nil {
			t.Fatalf("q=%g Bandwidth: %v", tc.q, err)
		}
		if !(math.Abs(bw-tc.want) <= 4e-16*tc.want) {
			t.Errorf("q=%g Bandwidth = %.17g, want %.17g", tc.q, bw, tc.want)
		}
	}
}
