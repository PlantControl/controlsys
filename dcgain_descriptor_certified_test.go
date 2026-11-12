package controlsys

import (
	"math"
	"math/big"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func descriptorFromSlices(t *testing.T, n, m, p int, a, b, c, d, e []float64, dt float64) *System {
	t.Helper()
	sys, err := NewDescriptor(mat.NewDense(n, n, a), mat.NewDense(n, m, b), mat.NewDense(p, n, c), mat.NewDense(p, m, d), mat.NewDense(n, n, e), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func scaledSlice(v []float64, k float64) []float64 {
	o := make([]float64, len(v))
	for i, x := range v {
		o[i] = k * x
	}
	return o
}

// OECG2I: DCGain evaluated a descriptor model without a certificate and
// returned the double evaluation with a nil error. The explicit
// realizations of the UP7H7O literals, with E = 2I, A and B doubled (the
// same float model), cancel D against C(x0·E − A)⁻¹B below an ulp of D.
func TestDCGainDescriptorCancelledDCResolvesFloatModelGain(t *testing.T) {
	two := func(n int) []float64 {
		e := make([]float64, n*n)
		for i := range n {
			e[i*n+i] = 2
		}
		return e
	}
	tests := []struct {
		name       string
		n          int
		a, b, c    []float64
		d, dt, val float64
	}{
		{"dipulp o2", 2, []float64{0, -0.48999999999999994, 1, -1.4}, []float64{-0.024499999999999997, -0.06999999999999999}, []float64{0, 1}, 0.05, 0, 1.3594567648471302e-18},
		{"0.05s²/(s²+1.4s+0.49)", 2, []float64{0, -0.49, 1, -1.4}, []float64{-0.0245, -0.06999999999999999}, []float64{0, 1}, 0.05, 0, -5.664403186863049e-20},
		{"(z-1)/(z-0.175)", 1, []float64{0.175}, []float64{-0.825}, []float64{1}, 1, 0.1, 6.728624391667616e-17},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sys := descriptorFromSlices(t, tc.n, 1, 1, scaledSlice(tc.a, 2), scaledSlice(tc.b, 2), tc.c, []float64{tc.d}, two(tc.n), tc.dt)
			if exact := dcGainExactFloat(t, sys); exact != tc.val {
				t.Fatalf("exact gain %g, literal %g", exact, tc.val)
			}
			g, err := sys.DCGain()
			if err != nil || !(math.Abs(g.At(0, 0)-tc.val) <= 1e-12*math.Abs(tc.val)) {
				t.Fatalf("DCGain = %v, %v; want %v", g.At(0, 0), err, tc.val)
			}
		})
	}
}

// A discrete descriptor model with non-symmetric A and E, E not diagonal,
// so a transposed E − A or a dropped low word of E_ij − A_ij moves the gain.
// With D the rounded −C(E − A)⁻¹B the gain cancels to its residual, and a
// nil error promises it to within the rounding of its own terms (AZFT6M).
func TestDCGainDescriptorDiscreteNonSymmetricPencil(t *testing.T) {
	a := []float64{0.3, 0.1, -0.2, 0.05, -0.4, 0.15, 0.2, 0.07, 0.1}
	e := []float64{1.1, 0.2, 0, -0.3, 0.9, 0.1, 0.05, 0.4, 1.3}
	b := []float64{1, 0.5, -0.3}
	c := []float64{0.7, -0.2, 0.4}
	sys := descriptorFromSlices(t, 3, 1, 1, a, b, c, []float64{0}, e, 0.1)
	_, gs, ok := dcGainExactRat(sys)
	if !ok {
		t.Fatal("E − A exactly singular")
	}
	cx, _ := gs[0].Float64()
	var plain float64
	for _, d := range []float64{0.25, -cx} {
		sys.D.Set(0, 0, d)
		want := dcGainExactFloat(t, sys)
		if d == 0.25 {
			plain = want
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("D = %v: DCGain error %v", d, err)
		}
		if d == 0.25 && !(math.Abs(g.At(0, 0)-want) <= 1e-15*math.Abs(want)) {
			t.Errorf("D = %v: DCGain = %v; want %v", d, g.At(0, 0), want)
		}
		if ex, _ := dcGainOwnRoundingExcess(sys, g); ex > 0 {
			t.Errorf("D = %v: DCGain = %v exceeds its own rounding of %v by %g", d, g.At(0, 0), want, ex)
		}
	}
	tr := descriptorFromSlices(t, 3, 1, 1, transposed(a, 3), b, c, []float64{0.25}, transposed(e, 3), 0.1)
	if gt, _ := tr.DCGain(); math.Abs(gt.At(0, 0)-plain) <= 1e-6*math.Abs(plain) {
		t.Fatal("transposed pencil has the same gain; the case cannot detect transposition")
	}
}

func transposed(v []float64, n int) []float64 {
	o := make([]float64, n*n)
	for i := range n {
		for j := range n {
			o[j*n+i] = v[i*n+j]
		}
	}
	return o
}

// Descriptor models whose exact DC gain is 0 keep a certified 0: dyadic
// pencils with non-symmetric A and non-diagonal E whose solve is exact.
func TestDCGainDescriptorExactZeroStaysCertifiedZero(t *testing.T) {
	cont := descriptorFromSlices(t, 2, 1, 1, []float64{-1, 0.5, 0, -2}, []float64{1, 1}, []float64{1, 1}, []float64{-1.75}, []float64{2, 0.5, 0.25, 1}, 0)
	disc := descriptorFromSlices(t, 2, 1, 1, []float64{0.5, 0.25, 0.125, 0.75}, []float64{1, 1}, []float64{1, 1}, []float64{-2.25}, []float64{2.5, 0.5, 0.125, 1.25}, 0.1)
	for _, sys := range []*System{cont, disc} {
		_, gs, ok := dcGainExactRat(sys)
		if !ok || gs[0].Cmp(new(big.Rat)) != 0 {
			t.Fatalf("exact gain %v, ok %v; want 0", gs, ok)
		}
		g, err := sys.DCGain()
		if err != nil || g.At(0, 0) != 0 {
			t.Errorf("dt=%v: DCGain = %v, %v; want 0, nil", sys.Dt, g.At(0, 0), err)
		}
	}
}

// An order-24 dyadic descriptor model whose exact DC gain is 0, past the
// exact check: the regular solve leaves it unproven, and DCGain keeps the
// evaluation route's 0 with a nil error rather than ErrAccuracyUnverified
// (OECG2I census dz_n24_r0_dt0).
func TestDCGainDescriptorExactZeroPastExactCheck(t *testing.T) {
	anz := [][3]float64{{0, 0, -1}, {1, 1, -0.25}, {1, 2, -0.25}, {2, 2, -2}, {2, 3, -0.5}, {2, 4, 0.5}, {3, 3, -4}, {3, 4, -0.25}, {4, 4, -1}, {4, 5, 0.5}, {4, 6, 0.5}, {5, 5, -8}, {5, 6, 0.25}, {5, 7, -0.25}, {6, 6, -2}, {6, 7, 0.25}, {6, 8, -0.5}, {7, 7, -4}, {7, 8, -0.5}, {7, 9, 0.25}, {8, 8, -8}, {8, 9, -0.25}, {9, 9, -8}, {9, 10, -0.25}, {9, 11, -0.5}, {10, 10, -1}, {10, 11, 0.25}, {10, 12, 0.5}, {11, 11, -2}, {11, 12, 0.25}, {11, 13, -0.25}, {12, 12, -1}, {12, 13, -0.5}, {12, 14, 0.5}, {13, 13, -8}, {13, 15, 0.5}, {14, 14, -1}, {14, 16, -0.5}, {15, 15, -2}, {15, 16, 0.5}, {16, 16, -2}, {16, 17, -0.25}, {16, 18, 0.5}, {17, 17, -1}, {17, 18, 0.25}, {18, 18, -4}, {18, 19, 0.25}, {19, 19, -8}, {20, 20, -0.5}, {20, 21, 0.5}, {20, 22, 0.25}, {21, 21, -0.25}, {21, 22, -0.5}, {21, 23, -0.5}, {22, 22, -2}, {22, 23, 0.5}, {23, 23, -0.5}}
	enz := [][3]float64{{0, 0, 1}, {0, 1, -0.25}, {0, 14, 0.125}, {0, 16, 0.125}, {0, 20, -0.25}, {1, 1, 1}, {1, 4, -0.25}, {1, 9, -0.25}, {1, 10, -0.25}, {1, 11, 0.125}, {1, 19, 0.25}, {2, 1, 0.125}, {2, 2, 1}, {2, 5, 0.125}, {2, 7, -0.25}, {2, 8, -0.125}, {2, 9, -0.25}, {2, 10, 0.125}, {2, 11, 0.25}, {2, 15, 0.125}, {2, 22, 0.125}, {2, 23, -0.25}, {3, 3, 1}, {3, 4, -0.125}, {3, 11, 0.125}, {3, 16, 0.25}, {3, 21, 0.125}, {4, 0, -0.25}, {4, 4, 1}, {4, 13, 0.125}, {4, 18, -0.25}, {4, 19, 0.125}, {4, 22, -0.125}, {4, 23, 0.125}, {5, 0, 0.125}, {5, 1, 0.125}, {5, 4, 0.25}, {5, 5, 1}, {5, 6, -0.125}, {5, 8, -0.25}, {5, 9, -0.125}, {5, 19, -0.25}, {5, 20, 0.25}, {5, 22, -0.125}, {6, 0, 0.125}, {6, 1, 0.25}, {6, 5, 0.25}, {6, 6, 1}, {6, 9, 0.125}, {6, 10, -0.25}, {6, 17, -0.25}, {6, 18, -0.125}, {7, 2, -0.125}, {7, 5, -0.125}, {7, 7, 1}, {7, 9, -0.125}, {7, 13, -0.125}, {7, 21, 0.125}, {8, 2, -0.25}, {8, 4, -0.125}, {8, 5, 0.25}, {8, 8, 1}, {8, 12, 0.25}, {8, 14, -0.25}, {8, 20, -0.125}, {8, 22, 0.25}, {8, 23, 0.25}, {9, 4, -0.125}, {9, 6, -0.25}, {9, 9, 1}, {9, 15, 0.125}, {9, 16, -0.125}, {9, 17, 0.25}, {9, 18, 0.125}, {9, 23, -0.125}, {10, 2, -0.125}, {10, 3, -0.25}, {10, 9, 0.125}, {10, 10, 1}, {10, 11, -0.125}, {10, 12, -0.125}, {10, 16, -0.125}, {10, 21, 0.125}, {11, 0, -0.125}, {11, 3, 0.25}, {11, 4, 0.125}, {11, 11, 1}, {11, 12, -0.25}, {11, 14, -0.25}, {11, 15, -0.25}, {11, 16, -0.125}, {12, 11, -0.25}, {12, 12, 1}, {12, 18, 0.125}, {12, 23, 0.25}, {13, 6, -0.125}, {13, 12, -0.125}, {13, 13, 1}, {13, 14, -0.125}, {13, 20, 0.25}, {13, 21, -0.25}, {14, 0, 0.125}, {14, 2, 0.25}, {14, 7, 0.125}, {14, 14, 1}, {14, 18, 0.25}, {14, 23, -0.25}, {15, 2, 0.125}, {15, 3, -0.125}, {15, 4, -0.25}, {15, 5, 0.125}, {15, 7, -0.25}, {15, 11, -0.25}, {15, 12, -0.25}, {15, 13, -0.125}, {15, 14, -0.25}, {15, 15, 1}, {15, 23, 0.125}, {16, 0, 0.25}, {16, 1, -0.25}, {16, 3, 0.125}, {16, 4, 0.125}, {16, 10, 0.125}, {16, 11, -0.25}, {16, 13, 0.25}, {16, 16, 1}, {16, 17, -0.125}, {16, 19, 0.125}, {16, 20, 0.25}, {16, 23, -0.125}, {17, 3, 0.25}, {17, 4, -0.25}, {17, 5, 0.125}, {17, 12, 0.25}, {17, 13, -0.125}, {17, 15, -0.25}, {17, 17, 1}, {17, 18, 0.25}, {17, 19, 0.25}, {17, 21, -0.25}, {18, 6, -0.25}, {18, 9, 0.125}, {18, 14, 0.25}, {18, 18, 1}, {18, 22, 0.125}, {19, 9, -0.25}, {19, 10, 0.125}, {19, 15, -0.125}, {19, 16, -0.25}, {19, 18, 0.25}, {19, 19, 1}, {20, 0, -0.25}, {20, 10, 0.25}, {20, 12, 0.125}, {20, 13, -0.125}, {20, 14, 0.25}, {20, 15, 0.125}, {20, 20, 1}, {21, 4, 0.125}, {21, 14, 0.25}, {21, 17, 0.125}, {21, 20, -0.125}, {21, 21, 1}, {21, 22, -0.25}, {22, 0, 0.25}, {22, 3, 0.25}, {22, 9, 0.125}, {22, 22, 1}, {23, 1, 0.25}, {23, 2, 0.25}, {23, 5, 0.125}, {23, 11, -0.25}, {23, 13, -0.25}, {23, 14, 0.25}, {23, 15, 0.25}, {23, 16, 0.125}, {23, 19, -0.125}, {23, 23, 1}}
	b := []float64{-0.5, -1, -1, 0.5, 1.5, -0.5, -0.5, 1, -1, 1, 0.5, 0, 1.5, -1, -1, -1, -1.5, 0.5, -0.5, -1, -1, 0.5, -1.5, 1}
	c := []float64{-1.5, 1, -1, -0.5, -1, 1, -0.5, 0, 0, -0.5, 1, 0.5, 1.5, -1.5, 1.5, 0, 0.5, -1.5, 1.5, -1, 0, -0.5, 0.5, -0.5}
	d := []float64{3.284371895422659}
	const n = 24
	a, e := make([]float64, n*n), make([]float64, n*n)
	for _, v := range anz {
		a[int(v[0])*n+int(v[1])] = v[2]
	}
	for _, v := range enz {
		e[int(v[0])*n+int(v[1])] = v[2]
	}
	sys := descriptorFromSlices(t, n, 1, 1, a, b, c, d, e, 0)
	if _, gs, ok := dcGainExactRat(sys); !ok || gs[0].Sign() != 0 {
		t.Fatalf("exact gain %v, ok %v; want 0", gs, ok)
	}
	g, err := sys.DCGain()
	if err != nil || g.At(0, 0) != 0 {
		t.Errorf("DCGain = %v, %v; want 0, nil", g.At(0, 0), err)
	}
}
