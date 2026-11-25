package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// zeroCancelSystem has C·Aᵏ·B = 0 by cancellation: B is an eigenvector of
// the non-symmetric A and C is orthogonal to it. perturb adds one ulp to
// A[0][1], which makes C·A·B nonzero.
func zeroCancelSystem(perturb bool) (A, B, C *mat.Dense) {
	a01 := 1.0
	if perturb {
		a01 = math.Nextafter(1, 2)
	}
	A = mat.NewDense(2, 2, []float64{-3, a01, -1, -1})
	B = mat.NewDense(2, 1, []float64{1, 1})
	C = mat.NewDense(1, 2, []float64{1, -1})
	return A, B, C
}

func TestExactZeroTransfer_DecidesOnStoredData(t *testing.T) {
	structA := mat.NewDense(3, 3, []float64{-1, 0.5, 0.25, 0, -2, 0, 0, 0.75, -3})
	structB := mat.NewDense(3, 2, []float64{1, -2, 0, 0, 0, 0})
	structC := mat.NewDense(1, 3, []float64{0, 1, 4})
	cancelA, cancelB, cancelC := zeroCancelSystem(false)
	ulpA, ulpB, ulpC := zeroCancelSystem(true)
	ulpCdirect := mat.NewDense(1, 2, []float64{1, math.Nextafter(-1, 0)})

	big := 33
	bigA := mat.NewDense(big, big, nil)
	bigB := mat.NewDense(big, 1, nil)
	bigC := mat.NewDense(1, big, nil)
	for i := range big {
		bigA.Set(i, i, -1-float64(i))
		if i > 0 {
			bigA.Set(i, i-1, 0.5)
		}
	}
	bigB.Set(0, 0, 1)
	bigC.Set(0, 0, 1)

	cases := []struct {
		name    string
		A, B, C *mat.Dense
		want    exactZeroVerdict
	}{
		{"structural", structA, structB, structC, transferZero},
		{"cancellation", cancelA, cancelB, cancelC, transferZero},
		{"one ulp in A", ulpA, ulpB, ulpC, transferNonzero},
		{"one ulp in C", cancelA, cancelB, ulpCdirect, transferNonzero},
		{"zero B", structA, mat.NewDense(3, 1, nil), structC, transferZero},
		{"n above cap", bigA, bigB, bigC, transferUndetermined},
	}
	for _, c := range cases {
		if got := exactZeroTransfer(c.A, c.B, c.C); got != c.want {
			t.Errorf("%s: exactZeroTransfer = %d, want %d", c.name, got, c.want)
		}
	}

	for _, c := range cases[:4] {
		got := markovExact(newDyadicImage(c.A, false), newDyadicImage(c.B, false), newDyadicImage(c.C, false))
		if got != c.want {
			t.Errorf("%s: markovExact = %d, want %d", c.name, got, c.want)
		}
	}

	wide := mat.NewDense(2, 2, []float64{-1, 0x1p-900, 0, -0x1p300})
	wideB := mat.NewDense(2, 1, []float64{0, 0x1p-500})
	wideC := mat.NewDense(1, 2, []float64{0x1p400, 0})
	if got := markovExact(newDyadicImage(wide, false), newDyadicImage(wideB, false), newDyadicImage(wideC, false)); got != transferNonzero {
		t.Errorf("wide exponent span: markovExact = %d, want nonzero (C·A·B = 2⁻¹⁰⁰⁰)", got)
	}
	if got := exactZeroTransfer(wide, wideB, wideC); got != transferNonzero {
		t.Errorf("wide exponent span: exactZeroTransfer = %d, want nonzero", got)
	}
}
