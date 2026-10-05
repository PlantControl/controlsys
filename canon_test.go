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
	for _, form := range []CanonForm{CanonModal, CanonCompanion} {
		if _, err := Canon(sys, form); !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("%s: err = %v, want ErrDimensionMismatch", form, err)
		}
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

func TestCanonCompanion_MIMO(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-1, 2, 0, 0.5, -3, 1, 0, 1, -2})
	B := mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, 1})
	C := mat.NewDense(2, 3, []float64{1, 0, 1, 0, 1, 0})
	D := mat.NewDense(2, 2, []float64{0.5, 0, 0, 0})
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Canon(sys, CanonCompanion)
	if err != nil {
		t.Fatal(err)
	}
	checkCanonT(t, sys, res, 1e-9)
	// det(sI-A) = s^3 + 6s^2 + 9s + 3
	want := mat.NewDense(3, 3, []float64{0, 0, -3, 1, 0, -9, 0, 1, -6})
	if !mat.EqualApprox(res.Sys.A, want, 1e-9) {
		t.Errorf("A = %v, want %v", mat.Formatted(res.Sys.A), mat.Formatted(want))
	}
	if got := mat.Col(nil, 0, res.Sys.B); math.Abs(got[0]-1) > 1e-12 || math.Abs(got[1]) > 1e-12 || math.Abs(got[2]) > 1e-12 {
		t.Errorf("B(:,1) = %v, want e1", got)
	}

	unctrl, err := New(mat.NewDense(2, 2, []float64{-1, 0, 0, -2}), mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{1, 1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Canon(unctrl, CanonCompanion); !errors.Is(err, ErrSingularTransform) {
		t.Errorf("uncontrollable: err = %v, want ErrSingularTransform", err)
	}
}

func TestCanonDefaultIsModal(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{-1, 2, 0, -3}), mat.NewDense(2, 1, []float64{1, 1}), mat.NewDense(1, 2, []float64{1, 0}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := Canon(sys, "")
	if err != nil {
		t.Fatal(err)
	}
	checkCanonT(t, sys, res, 1e-9)
	if _, err := Canon(sys, "jordan"); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("unknown form: err = %v, want ErrInvalidArgument", err)
	}
}

// checkCanonT verifies the MATLAB convention xc = T·x: Ac = T·A·T⁻¹,
// Bc = T·B, Cc = C·T⁻¹.
func checkCanonT(t *testing.T, sys *System, res *CanonResult, tol float64) {
	t.Helper()
	n, m, p := sys.Dims()
	var lu mat.LU
	lu.Factorize(res.T)
	TA := mat.NewDense(n, n, nil)
	TA.Mul(res.T, sys.A)
	var ac mat.Dense
	ac.Solve(res.T.T(), TA.T())
	if !mat.EqualApprox(ac.T(), res.Sys.A, tol) {
		t.Errorf("T·A·T⁻¹ = %v, want %v", mat.Formatted(ac.T()), mat.Formatted(res.Sys.A))
	}
	bc := mat.NewDense(n, m, nil)
	bc.Mul(res.T, sys.B)
	if !mat.EqualApprox(bc, res.Sys.B, tol) {
		t.Errorf("T·B = %v, want %v", mat.Formatted(bc), mat.Formatted(res.Sys.B))
	}
	var cc mat.Dense
	cc.Solve(res.T.T(), sys.C.T())
	if r, c := cc.Dims(); r != n || c != p || !mat.EqualApprox(cc.T(), res.Sys.C, tol) {
		t.Errorf("C·T⁻¹ = %v, want %v", mat.Formatted(cc.T()), mat.Formatted(res.Sys.C))
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

func TestCanonModalTransformConvention(t *testing.T) {
	for name, A := range map[string]*mat.Dense{
		"eig":   mat.NewDense(3, 3, []float64{-1, 4, 0, -2, -1, 1, 0, 0.5, -3}),
		"schur": mat.NewDense(3, 3, []float64{-1, 1, 0, 0, -1, 1, 0, 0, -1}),
	} {
		for _, dt := range []float64{0, 0.1} {
			if dt > 0 {
				A = mat.DenseCopyOf(A)
				A.Scale(0.2, A)
			}
			sys, err := New(A, mat.NewDense(3, 2, []float64{1, 0, 0, 1, 1, 1}), mat.NewDense(2, 3, []float64{1, 0, 1, 0, 1, 0}), mat.NewDense(2, 2, []float64{0.5, 0, 0, 0}), dt)
			if err != nil {
				t.Fatal(err)
			}
			res, err := Canon(sys, CanonModal)
			if err != nil {
				t.Fatalf("%s dt=%g: %v", name, dt, err)
			}
			checkCanonT(t, sys, res, 1e-8)
		}
	}
}
