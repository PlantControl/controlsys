package controlsys

import (
	"math/cmplx"
	"testing"
)

// ZWGUUN: with more plant inputs than outputs the m x m I+K·G loses the
// identity to rounding once ||K·G|| >> 1/eps; the p x p I+G·K stays exact.
func TestFRDFeedbackWideLargeGainMatchesScalarLoop(t *testing.T) {
	k := [][]complex128{{0.3}, {0.2}}
	for _, c := range []float64{1, 1e8, 1e14, 1e15, 1e16, 1e20} {
		g1 := complex(3*c, c)
		g2 := complex(-2*c, 0.5*c)
		plant, err := NewFRD([][][]complex128{{{g1, g2}}}, []float64{1}, 0)
		if err != nil {
			t.Fatal(err)
		}
		ctrl, err := NewFRD([][][]complex128{k}, []float64{1}, 0)
		if err != nil {
			t.Fatal(err)
		}
		cl, err := FRDFeedback(plant, ctrl, -1)
		if err != nil {
			t.Fatalf("c=%g: %v", c, err)
		}
		loop := 1 + g1*k[0][0] + g2*k[1][0]
		for j, gj := range []complex128{g1, g2} {
			want := gj / loop
			got := cl.Response[0][0][j]
			if rel := cmplx.Abs(got-want) / cmplx.Abs(want); rel > 1e-12 {
				t.Errorf("c=%g col %d: got %v want %v rel %g", c, j, got, want, rel)
			}
		}
	}
}

func TestFRDFeedbackWideMatchesTransposedTall(t *testing.T) {
	g := [][]complex128{
		{complex(1, 2), complex(-0.5, 0.3), complex(0.7, -1.1)},
		{complex(0.2, -0.4), complex(2, 1), complex(-1.3, 0.6)},
	}
	k := [][]complex128{
		{complex(0.4, 0.1), complex(-0.2, 0.3)},
		{complex(0.1, -0.5), complex(0.6, 0.2)},
		{complex(-0.3, 0.2), complex(0.25, -0.1)},
	}
	gt := make([][]complex128, 3)
	for i := range gt {
		gt[i] = []complex128{g[0][i], g[1][i]}
	}
	kt := [][]complex128{{k[0][0], k[1][0], k[2][0]}, {k[0][1], k[1][1], k[2][1]}}
	mk := func(r [][]complex128) *FRD {
		f, err := NewFRD([][][]complex128{r}, []float64{1}, 0)
		if err != nil {
			t.Fatal(err)
		}
		return f
	}
	for _, sign := range []float64{-1, 1} {
		cl, err := FRDFeedback(mk(g), mk(k), sign)
		if err != nil {
			t.Fatal(err)
		}
		clt, err := FRDFeedback(mk(gt), mk(kt), sign)
		if err != nil {
			t.Fatal(err)
		}
		for i := range 2 {
			for j := range 3 {
				got, want := cl.Response[0][i][j], clt.Response[0][j][i]
				if cmplx.Abs(got-want) > 1e-13*(1+cmplx.Abs(want)) {
					t.Errorf("sign=%g (%d,%d): wide %v tall^T %v", sign, i, j, got, want)
				}
			}
		}
	}
}
