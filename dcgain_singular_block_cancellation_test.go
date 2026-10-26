package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// 2TUAD6 review: x0·I − A exactly singular in a dense block whose pole the
// input cannot reach or the output cannot see. Deflating the block left the
// cancelled 1/σ coefficient as roundoff, giving ±Inf (sign by arch) for a
// finite gain; base f86231f matched the exact limit, also pinned here as
// literal values.
func TestDCGain_ExactlySingularDenseBlockCancelledPoleFinite(t *testing.T) {
	cases := []struct {
		n, m, p    int
		a, b, c, d []float64
		dt         float64
		row        int
		want       []float64
	}{
		{3, 1, 2,
			[]float64{0, 27, 0, 12, 33, -18, 6, 30, -9},
			[]float64{1.6848098638733329, -1.4015085686052942, 0.7925322214671411},
			[]float64{0, -6, 0, 2, 7, -3},
			[]float64{0, 0}, 0, 0, []float64{-0.49343990647231056}},
		{4, 2, 2,
			[]float64{0.25, -0.625, 0.375, 0, -0.25, 0, 0.75, 0.125, -0.5, -1.25, 1.75, 0, 0.75, -1.375, 1.125, 1.25},
			[]float64{-0.8201486044489479, -0.2994097035955181, 0.849668345042276, 0.7454224663849938,
				-0.40627556712309687, -0.4576832773612235, 1.3112718795358278, 1.6150160503054698},
			[]float64{0, 2, 0, 1, 2, 0, 0, 0},
			[]float64{0.5, -0.75, 0.5, 0.75}, 0.1, 1, []float64{-0.709267948456445, 0.6372495685989554}},
	}
	for ci, tc := range cases {
		sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		exact := dcGainExactLimit(t, sys)
		for j, w := range tc.want {
			if exact[tc.row][j] != w {
				t.Fatalf("case %d: oracle [%d,%d] = %v, want %v", ci, tc.row, j, exact[tc.row][j], w)
			}
		}
		checkDCGainExact(t, ci, sys, exact)
	}
}

// 2TUAD6 review: deflating an exactly singular dense block flipped the sign
// of an infinite DC gain against the exact σ → 0+ limit, which base
// f86231f matched.
func TestDCGain_ExactlySingularDenseBlockInfiniteSign(t *testing.T) {
	inf := math.Inf(1)
	cases := []struct {
		n, m, p    int
		a, b, c, d []float64
		dt         float64
		want       [][]float64
	}{
		{3, 2, 1,
			[]float64{1.5, 1.5, 4.5, -0.49999999997089617, -0.5, -1.5, -0.5, -0.5, -1.5},
			[]float64{-1.9565960523876103, -0.7395325727734102, 1.5050715883173882, 0.8938557339291633, 1.2175042573836528, -0.4160250147440223},
			[]float64{1.483813693813647, 1.0444815524682294, -0.714299270598148},
			[]float64{-0.5, -3}, 0,
			[][]float64{{-inf, inf}}},
		{3, 2, 2,
			[]float64{7, 0, 6, -7, 7, -7, -6, 0, -5},
			[]float64{0, 2, -4, 1, 0, -2},
			[]float64{1.4376730372347413, 0.541009906978126, -0.18850692238421018, -1.4527016618901731, 0.42235497779344666, -1.0651336050962672},
			[]float64{0, 0.25, 0, 0.25}, 0.1,
			[][]float64{{0.36067327131875065, inf}, {0.2815699851956311, -inf}}},
	}
	for ci, tc := range cases {
		sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
		if err != nil {
			t.Fatal(err)
		}
		exact := dcGainExactLimit(t, sys)
		for i, row := range tc.want {
			for j, w := range row {
				if exact[i][j] != w {
					t.Fatalf("case %d: oracle [%d,%d] = %v, want %v", ci, i, j, exact[i][j], w)
				}
			}
		}
		checkDCGainExact(t, ci, sys, exact)
	}
}

// checkDCGainExact requires DCGain and integratingDCGain to equal the exact
// limit: Inf entries exactly, finite ones to 1e-12 relative.
func checkDCGainExact(t *testing.T, ci int, sys *System, exact [][]float64) {
	t.Helper()
	for _, gain := range []func() (*mat.Dense, error){sys.DCGain, sys.integratingDCGain} {
		g, err := gain()
		if err != nil {
			t.Fatalf("case %d: %v", ci, err)
		}
		for i, row := range exact {
			for j, w := range row {
				got := g.At(i, j)
				if math.IsInf(w, 0) && got != w || !math.IsInf(w, 0) && !(math.Abs(got-w) <= 1e-12*math.Abs(w)) {
					t.Errorf("case %d: gain[%d,%d] = %v, exact limit %v", ci, i, j, got, w)
				}
			}
		}
	}
}
