package controlsys

import (
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestPrescale_Simple(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkPrescalePolesPreserved(t, sys, pr.Sys, 1e-10)
	checkPrescaleBodePreserved(t, sys, pr.Sys, 1e-6)
	checkPrescaleDCGainPreserved(t, sys, pr.Sys, 1e-10)

	for i, s := range pr.Info.StateScale {
		if math.Abs(s-1.0) > 1e-10 {
			t.Errorf("StateScale[%d] = %g, want ~1", i, s)
		}
	}
}

func TestPrescale_LargeGainSpread(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -1000}),
		mat.NewDense(2, 1, []float64{1, 1000}),
		mat.NewDense(1, 2, []float64{1000, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkPrescalePolesPreserved(t, sys, pr.Sys, 1e-10)
	checkPrescaleDCGainPreserved(t, sys, pr.Sys, 1e-6)
	checkPrescaleBodePreserved(t, sys, pr.Sys, 1e-6)

	if len(pr.Info.InputScale) != 1 {
		t.Fatalf("InputScale length = %d, want 1", len(pr.Info.InputScale))
	}
	if len(pr.Info.OutputScale) != 1 {
		t.Fatalf("OutputScale length = %d, want 1", len(pr.Info.OutputScale))
	}
}

func TestPrescale_MIMO(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 100, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 100}),
		mat.NewDense(2, 2, []float64{100, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{0, 0, 0, 0}), 0)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkPrescalePolesPreserved(t, sys, pr.Sys, 1e-10)
	checkPrescaleDCGainPreserved(t, sys, pr.Sys, 1e-6)
}

func TestPrescale_PureGain(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, _, _ := pr.Sys.Dims()
	if n != 0 {
		t.Errorf("expected pure gain, got n=%d", n)
	}
}

func TestPrescale_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	pr, err := Prescale(sys)
	if err != nil {
		t.Fatal(err)
	}

	checkPrescalePolesPreserved(t, sys, pr.Sys, 1e-10)
	checkPrescaleDCGainPreserved(t, sys, pr.Sys, 1e-10)
}

func checkPrescalePolesPreserved(t *testing.T, orig, scaled *System, tol float64) {
	t.Helper()
	p1, err := orig.Poles()
	if err != nil {
		t.Fatalf("orig poles: %v", err)
	}
	p2, err := scaled.Poles()
	if err != nil {
		t.Fatalf("scaled poles: %v", err)
	}
	if len(p1) != len(p2) {
		t.Fatalf("pole count %d != %d", len(p1), len(p2))
	}
	sort.Slice(p1, func(i, j int) bool {
		if real(p1[i]) != real(p1[j]) {
			return real(p1[i]) < real(p1[j])
		}
		return imag(p1[i]) < imag(p1[j])
	})
	sort.Slice(p2, func(i, j int) bool {
		if real(p2[i]) != real(p2[j]) {
			return real(p2[i]) < real(p2[j])
		}
		return imag(p2[i]) < imag(p2[j])
	})
	for i := range p1 {
		if cmplx.Abs(p1[i]-p2[i]) > tol {
			t.Errorf("pole[%d]: %v != %v", i, p1[i], p2[i])
		}
	}
}

func checkPrescaleBodePreserved(t *testing.T, orig, scaled *System, tol float64) {
	t.Helper()
	freqs := []float64{0.01, 0.1, 1, 10, 100}
	for _, w := range freqs {
		var s complex128
		if orig.IsContinuous() {
			s = complex(0, w)
		} else {
			s = cmplx.Exp(complex(0, w*orig.Dt))
		}
		g1, err1 := orig.EvalFr(s)
		g2, err2 := scaled.EvalFr(s)
		if err1 != nil || err2 != nil {
			t.Errorf("EvalFr error at w=%g", w)
			continue
		}
		_, m1, p1 := orig.Dims()
		for i := range p1 {
			for j := range m1 {
				diff := cmplx.Abs(g1[i][j] - g2[i][j])
				mag := cmplx.Abs(g1[i][j])
				relTol := tol
				if mag > 1e-10 {
					relTol = tol * mag
				}
				if diff > relTol {
					t.Errorf("w=%g: G[%d,%d] diff=%g > tol=%g", w, i, j, diff, relTol)
				}
			}
		}
	}
}

func checkPrescaleDCGainPreserved(t *testing.T, orig, scaled *System, tol float64) {
	t.Helper()
	dc1, err := orig.DCGain()
	if err != nil {
		t.Fatalf("orig DCGain: %v", err)
	}
	dc2, err := scaled.DCGain()
	if err != nil {
		t.Fatalf("scaled DCGain: %v", err)
	}
	r, c := dc1.Dims()
	for i := range r {
		for j := range c {
			diff := math.Abs(dc1.At(i, j) - dc2.At(i, j))
			mag := math.Abs(dc1.At(i, j))
			relTol := tol
			if mag > 1e-10 {
				relTol = tol * mag
			}
			if diff > relTol {
				t.Errorf("DCGain[%d,%d]: %g != %g (diff=%g)", i, j, dc2.At(i, j), dc1.At(i, j), diff)
			}
		}
	}
}
