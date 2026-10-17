package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPIDDerivTailUsesInducedTwoNorm(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-3, 0, -4, -5})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	plant, err := New(A, B, C, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	tail := pidDerivTail(plant)
	sigma, frob := math.Sqrt(45), math.Sqrt(50)
	if got := tail(0.999 * sigma); !math.IsInf(got, 1) {
		t.Fatalf("tail(%g) = %g below σ_max(A) = √45, want +Inf", 0.999*sigma, got)
	}
	cab := 3 * math.Sqrt2
	for _, w := range []float64{6.8, 7, frob} {
		want := 1 + cab/(w-sigma)
		if got := tail(w); got > want*(1+1e-12) {
			t.Errorf("tail(%g) = %.17g above first-order bound 1+‖CA‖‖B‖/(w−√45) = %.17g", w, got, want)
		}
	}
	for w := 6.8; w < 1e4; w *= 1.3 {
		s := complex(0, w)
		if got := cmplx.Abs(s / (s + 3)); got > tail(w) {
			t.Errorf("|jωG(jω)| = %g exceeds tail(%g) = %g", got, w, tail(w))
		}
	}
	if got := tail(math.Inf(1)); got != 1 {
		t.Errorf("tail(∞) = %g, want |CB| = 1", got)
	}
}
