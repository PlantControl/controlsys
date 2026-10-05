package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestCanonModal_RealEigenvalues(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -6, -5}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	Amod := res.Sys.A
	poles, _ := res.Sys.Poles()
	sortPolesByReal(poles)

	if math.Abs(real(poles[0])+3) > 1e-10 || math.Abs(real(poles[1])+2) > 1e-10 {
		t.Errorf("poles = %v, want [-3, -2]", poles)
	}

	for i := range 2 {
		for j := range 2 {
			if i != j && math.Abs(Amod.At(i, j)) > 1e-8 {
				t.Errorf("A_modal[%d,%d] = %g, want 0 (should be diagonal)", i, j, Amod.At(i, j))
			}
		}
	}

	dc, err := res.Sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dc.At(0, 0)-1.0/6.0) > 1e-8 {
		t.Errorf("dcgain = %g, want 1/6", dc.At(0, 0))
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
}

func TestCanonModal_ComplexEigenvalues(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -1, 0}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	Amod := res.Sys.A
	if math.Abs(Amod.At(0, 0)) > 1e-8 || math.Abs(Amod.At(1, 1)) > 1e-8 {
		t.Errorf("diagonal should be ~0, got [%g, %g]", Amod.At(0, 0), Amod.At(1, 1))
	}
	if math.Abs(math.Abs(Amod.At(0, 1))-1) > 1e-8 || math.Abs(math.Abs(Amod.At(1, 0))-1) > 1e-8 {
		t.Errorf("off-diagonal should be ±1, got [%g, %g]", Amod.At(0, 1), Amod.At(1, 0))
	}
	if Amod.At(0, 1)*Amod.At(1, 0) > 0 {
		t.Errorf("off-diagonal should have opposite signs")
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
}

func TestCanonModal_LargeRealSmallImagPair(t *testing.T) {
	const (
		a = 1e4
		b = 1e-7
	)
	sys, err := New(
		mat.NewDense(2, 2, []float64{a, b, -b, a}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	Amod := res.Sys.A
	if math.Abs(Amod.At(0, 0)-a) > 1e-6 || math.Abs(Amod.At(1, 1)-a) > 1e-6 {
		t.Errorf("diagonal = [%g, %g], want [%g, %g]", Amod.At(0, 0), Amod.At(1, 1), a, a)
	}
	if math.Abs(Amod.At(0, 1)) < 1e-10 || math.Abs(Amod.At(1, 0)) < 1e-10 {
		t.Fatalf("expected 2x2 modal block, got %v", mat.Formatted(Amod))
	}
	if Amod.At(0, 1)*Amod.At(1, 0) > 0 {
		t.Errorf("off-diagonal should have opposite signs")
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
	checkFreqPreserved(t, sys, res.Sys, 1e-8)
}

func TestCanonModal_Mixed(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, 0, 1, 0, -4, 0}),
		mat.NewDense(3, 1, []float64{1, 0, 1}),
		mat.NewDense(1, 3, []float64{1, 1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	poles, _ := res.Sys.Poles()
	realCount := 0
	complexCount := 0
	for _, p := range poles {
		if math.Abs(imag(p)) < 1e-8 {
			realCount++
		} else {
			complexCount++
		}
	}
	if realCount != 1 || complexCount != 2 {
		t.Errorf("expected 1 real + 2 complex poles, got %d real + %d complex", realCount, complexCount)
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)
	checkFreqPreserved(t, sys, res.Sys, 1e-6)
}

func TestCanonModal_Empty(t *testing.T) {
	sys, err := New(nil, nil, nil, mat.NewDense(1, 1, []float64{5}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	dc, _ := res.Sys.DCGain()
	if dc.At(0, 0) != 5 {
		t.Errorf("dcgain = %g, want 5", dc.At(0, 0))
	}
}

func TestCanonCompanion_SISO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -6, -5}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonCompanion)
	if err != nil {
		t.Fatal(err)
	}

	checkEigsPreserved(t, sys, res.Sys, 1e-10)

	dc, err := res.Sys.DCGain()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(dc.At(0, 0)-1.0/6.0) > 1e-8 {
		t.Errorf("dcgain = %g, want 1/6", dc.At(0, 0))
	}
}

func TestCanonCompanion_MIMOError(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -6, -5}),
		mat.NewDense(2, 2, []float64{0, 1, 1, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 2, []float64{0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Canon(sys, CanonCompanion)
	if err == nil {
		t.Error("expected error for MIMO system")
	}
}

func TestCanonInvalidForm(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Canon(sys, "bogus")
	if err == nil {
		t.Error("expected error for unknown form")
	}
}

func sortPolesByReal(poles []complex128) {
	sort.Slice(poles, func(i, j int) bool {
		if real(poles[i]) != real(poles[j]) {
			return real(poles[i]) < real(poles[j])
		}
		return imag(poles[i]) < imag(poles[j])
	})
}

func TestCanonModal_FrequencyPreserved(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 2, 0, -3}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}

	for _, w := range []float64{0.01, 0.1, 1, 10, 100} {
		g1, _ := sys.EvalFr(complex(0, w))
		g2, _ := res.Sys.EvalFr(complex(0, w))
		diff := cmplx.Abs(g1[0][0] - g2[0][0])
		if diff > 1e-6 {
			t.Errorf("w=%g: freq response diff=%g", w, diff)
		}
	}
}

func TestCanonModal_BadlyScaledPairOrderedByMagnitude(t *testing.T) {
	sys := badlyScaledPairSystem(t, -1e8, 1, -2e8, 0)
	res, err := Canon(sys, CanonModal)
	if err != nil {
		t.Fatal(err)
	}
	A := res.Sys.A
	if math.Abs(A.At(0, 0)+1e8) > 1e-6 || math.Abs(A.At(1, 1)+1e8) > 1e-6 || math.Abs(A.At(2, 2)+2e8) > 1e-6 {
		t.Fatalf("modal A not ordered by magnitude:\n%v", mat.Formatted(A))
	}
	if prod := A.At(0, 1) * A.At(1, 0); math.Abs(prod+1) > 1e-6 {
		t.Fatalf("pair block off-diagonal product = %g, want -1 (imag ±1)", prod)
	}
	assertPolesMatch(t, "modal", res.Sys, []complex128{complex(-1e8, 1), complex(-1e8, -1), -2e8}, 1e-6)
}

func TestCanonRejectsInvalidSystem(t *testing.T) {
	if _, err := Canon(nil, CanonModal); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil: err = %v, want ErrInvalidArgument", err)
	}
	sys, err := New(mat.NewDense(2, 2, []float64{-1, math.Inf(1), 0, -3}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, form := range []CanonForm{CanonModal, CanonCompanion} {
		if _, err := Canon(sys, form); !errors.Is(err, ErrInvalidArgument) {
			t.Errorf("%s Inf: err = %v, want ErrInvalidArgument", form, err)
		}
	}
}
