package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas/blas64"
)

// dcGainExactLimit is the exact rational limit of C(x0·I − A)⁻¹B + D on
// the float model, entry by entry.
func dcGainExactLimit(t *testing.T, sys *System) [][]float64 {
	t.Helper()
	n, m, p := sys.Dims()
	a := make([]float64, n*n)
	for i := range n {
		for j := range n {
			a[i*n+j] = sys.A.At(i, j)
		}
	}
	cp := fzCharPolyExact(a, n)
	x0 := fzRat(0)
	if sys.IsDiscrete() {
		x0 = fzRat(1)
	}
	out := make([][]float64, p)
	for i := range p {
		c := make([]float64, n)
		for k := range n {
			c[k] = sys.C.At(i, k)
		}
		for j := range m {
			b := make([]float64, n)
			for k := range n {
				b[k] = sys.B.At(k, j)
			}
			out[i] = append(out[i], fzLimitAt(fzPencilDetRat(n, 1, a, b, c, []float64{sys.D.At(i, j)}), cp, x0))
		}
	}
	return out
}

// 2TUAD6: x0·I − A exactly singular in a dense strongly connected block,
// with no 1×1 block A_kk = x0, left DCGain finite (≈1e13–1e14) where the
// float model has an exact pole at DC (MATLAB dcgain: Inf). All fail on
// base f86231f; the integer models are from a seeded sweep of rank-deficient
// integer K with A = I + K.
func TestDCGain_ExactlySingularDenseBlockInfinite(t *testing.T) {
	cases := []struct {
		n, m, p    int
		a, b, c, d []float64
		dt         float64
	}{
		{3, 2, 2,
			[]float64{0.38629023905802873, 3.851534645926291, -3.23782488498432, 0, 4, -3, 4, 1.6251019182011088, -4.625101918201109},
			[]float64{-0.5, 0, -1.75, 0, 0, 0},
			[]float64{-1, -2.176206339653932, 1, 0, -0.06848213510526557, 0},
			[]float64{5.737696031494993, 0, 0.75, 0}, 0.1},
		{5, 1, 1,
			[]float64{9, 5, -6, -1, 8, 3, 1, 10, -2, -10, -9, 0, 5, 0, -10, -19, 2, -4, -9, -8, 10, -2, 8, 10, -1},
			[]float64{2.500095871898054, -0.33706379991360613, 0.0362057162469831, -0.15844195418544205, -1.3532882081085718},
			[]float64{0.2164853863984775, 0.2871569247337147, -0.09211104860467367, -3.150311594839768, -0.32918595706412646},
			[]float64{0.5}, 0.1},
		{5, 1, 1,
			[]float64{8, 1, 0, 2, -7, -10, 1, 9, -3, 2, 5, -2, 2, -10, 2, -2, -5, 3, 9, -2, 3, -7, 4, -2, 1},
			[]float64{0.05698786387621729, 0.24870705011830574, 0.5172612151143173, -0.3108227475469052, -1.018681772950203},
			[]float64{-0.09340862009512899, 0.4443345587740295, -1.3463268611146315, 0.32944827259410614, -1.5109645398842768},
			[]float64{0.5}, 0.1},
		{5, 1, 1,
			[]float64{7, -5, 10, -10, -6, -10, -1, 14, -7, 2, 0, 7, 11, 3, -9, 4, 2, -8, 1, -10, -6, 0, 6, -7, -7},
			[]float64{0.06968518395714107, -0.8444961361963549, -0.8572185579281555, -0.6856176512729019, -0.6167227165892905},
			[]float64{0.82237127410032, 0.444390070070919, -1.382978862991128, -0.19857300570662595, -0.453742642090064},
			[]float64{0.5}, 0.1},
	}
	for ci, tc := range cases {
		sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("case %d: %v", ci, err)
		}
		want := dcGainExactLimit(t, sys)
		for i := range tc.p {
			for j := range tc.m {
				if got := g.At(i, j); got != want[i][j] {
					t.Errorf("case %d: DCGain[%d,%d] = %v, exact limit %v", ci, i, j, got, want[i][j])
				}
			}
		}
	}
}

func TestExactShiftedDetIsZeroShiftsExactly(t *testing.T) {
	tiny := math.Ldexp(1, -80)
	cases := []struct {
		a    []float64
		x0   float64
		want bool
	}{
		{[]float64{1}, 1, true},
		{[]float64{3}, 3, true},
		{[]float64{math.Nextafter(1, 2)}, 1, false},
		{[]float64{-9, 10, -10, 11}, 1, true},
		{[]float64{0.5, 0.5, 0.5, 0.5}, 1, true},
		// float 1 − 2⁻⁸⁰ rounds to 1, making the rounded I − A singular.
		{[]float64{tiny, 1, 1, 0}, 1, false},
		{[]float64{tiny, 1, 1, 0}, 0, false},
		{[]float64{4, -2, -2, 0, 3, -3, 4, 2, -6}, 0, true},
	}
	for _, tc := range cases {
		r := int(math.Sqrt(float64(len(tc.a))))
		if got := exactShiftedDetIsZero(blas64.General{Rows: r, Cols: r, Stride: r, Data: tc.a}, tc.x0); got != tc.want {
			t.Errorf("%v − %v·I: exactShiftedDetIsZero = %v, want %v", tc.a, tc.x0, got, tc.want)
		}
	}
}
