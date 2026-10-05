package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"sort"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func matClose(a, b *mat.Dense, tol float64) bool {
	ar, ac := a.Dims()
	br, bc := b.Dims()
	if ar != br || ac != bc {
		return false
	}
	aRaw := a.RawMatrix()
	bRaw := b.RawMatrix()
	for i := range ar {
		for j := range ac {
			if math.Abs(aRaw.Data[i*aRaw.Stride+j]-bRaw.Data[i*bRaw.Stride+j]) > tol {
				return false
			}
		}
	}
	return true
}

func sortPoles(p []complex128) {
	sort.Slice(p, func(i, j int) bool {
		ri, rj := real(p[i]), real(p[j])
		if math.Abs(ri-rj) > 1e-10 {
			return ri < rj
		}
		return imag(p[i]) < imag(p[j])
	})
}

func polesClose(a, b []complex128, tol float64) bool {
	if len(a) != len(b) {
		return false
	}
	sa := make([]complex128, len(a))
	sb := make([]complex128, len(b))
	copy(sa, a)
	copy(sb, b)
	sortPoles(sa)
	sortPoles(sb)
	for i := range sa {
		if cmplx.Abs(sa[i]-sb[i]) > tol {
			return false
		}
	}
	return true
}

// --- Augstate tests ---

func TestAugstate_Basic(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := mat.NewDense(2, 1, []float64{5, 6})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	aug, err := Augstate(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := aug.Dims()
	if n != 2 || m != 1 || p != 3 {
		t.Fatalf("dims = (%d,%d,%d), want (2,1,3)", n, m, p)
	}

	wantC := mat.NewDense(3, 2, []float64{
		1, 0,
		1, 0,
		0, 1,
	})
	wantD := mat.NewDense(3, 1, []float64{0, 0, 0})

	if !matClose(aug.C, wantC, 1e-12) {
		t.Errorf("C mismatch:\ngot  %v\nwant %v", mat.Formatted(aug.C), mat.Formatted(wantC))
	}
	if !matClose(aug.D, wantD, 1e-12) {
		t.Errorf("D mismatch:\ngot  %v\nwant %v", mat.Formatted(aug.D), mat.Formatted(wantD))
	}
}

func TestAugstate_StaticGain(t *testing.T) {
	D := mat.NewDense(1, 1, []float64{5})
	sys, err := NewGain(D, 0)
	if err != nil {
		t.Fatal(err)
	}

	aug, err := Augstate(sys)
	if err != nil {
		t.Fatal(err)
	}

	n, m, p := aug.Dims()
	if n != 0 || m != 1 || p != 1 {
		t.Errorf("static gain augstate dims = (%d,%d,%d), want (0,1,1)", n, m, p)
	}
	if aug.D.At(0, 0) != 5 {
		t.Errorf("D(0,0) = %v, want 5", aug.D.At(0, 0))
	}
}

// --- SS2SS tests ---

func TestSS2SS_Basic(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	T := mat.NewDense(2, 2, []float64{1, 1, 0, 1})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	result, err := SS2SS(sys, T)
	if err != nil {
		t.Fatal(err)
	}

	wantA := mat.NewDense(2, 2, []float64{-2, 0, -2, -1})
	wantB := mat.NewDense(2, 1, []float64{1, 1})
	wantC := mat.NewDense(1, 2, []float64{1, -1})
	wantD := mat.NewDense(1, 1, []float64{0})

	if !matClose(result.A, wantA, 1e-12) {
		t.Errorf("A2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.A), mat.Formatted(wantA))
	}
	if !matClose(result.B, wantB, 1e-12) {
		t.Errorf("B2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.B), mat.Formatted(wantB))
	}
	if !matClose(result.C, wantC, 1e-12) {
		t.Errorf("C2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.C), mat.Formatted(wantC))
	}
	if !matClose(result.D, wantD, 1e-12) {
		t.Errorf("D2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.D), mat.Formatted(wantD))
	}

	poles, err := result.Poles()
	if err != nil {
		t.Fatal(err)
	}
	wantPoles := []complex128{-1, -2}
	if !polesClose(poles, wantPoles, 1e-10) {
		t.Errorf("poles = %v, want %v", poles, wantPoles)
	}
}

func TestSS2SS_Identity(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	I := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	result, err := SS2SS(sys, I)
	if err != nil {
		t.Fatal(err)
	}

	if !matClose(result.A, sys.A, 1e-12) {
		t.Error("A changed with identity T")
	}
	if !matClose(result.B, sys.B, 1e-12) {
		t.Error("B changed with identity T")
	}
	if !matClose(result.C, sys.C, 1e-12) {
		t.Error("C changed with identity T")
	}
}

func TestSS2SS_Singular(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	Tsingular := mat.NewDense(2, 2, []float64{1, 2, 2, 4})
	_, err = SS2SS(sys, Tsingular)
	if !errors.Is(err, ErrSingularTransform) {
		t.Errorf("expected ErrSingularTransform, got %v", err)
	}
}

// --- Xperm tests ---

func TestXperm_Diagonal(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 0, 0,
		0, -2, 0,
		0, 0, -3,
	})
	B := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
	C := mat.NewDense(3, 3, []float64{
		1, 0, 0,
		0, 1, 0,
		0, 0, 1,
	})
	D := mat.NewDense(3, 3, nil)

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	result, err := Xperm(sys, []int{2, 0, 1})
	if err != nil {
		t.Fatal(err)
	}

	wantA := mat.NewDense(3, 3, []float64{
		-3, 0, 0,
		0, -1, 0,
		0, 0, -2,
	})
	wantB := mat.NewDense(3, 3, []float64{
		0, 0, 1,
		1, 0, 0,
		0, 1, 0,
	})
	wantC := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		1, 0, 0,
	})

	if !matClose(result.A, wantA, 1e-12) {
		t.Errorf("A2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.A), mat.Formatted(wantA))
	}
	if !matClose(result.B, wantB, 1e-12) {
		t.Errorf("B2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.B), mat.Formatted(wantB))
	}
	if !matClose(result.C, wantC, 1e-12) {
		t.Errorf("C2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.C), mat.Formatted(wantC))
	}
}

func TestXperm_NonSymmetric(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := mat.NewDense(2, 1, []float64{5, 6})
	C := mat.NewDense(1, 2, []float64{7, 8})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	result, err := Xperm(sys, []int{1, 0})
	if err != nil {
		t.Fatal(err)
	}

	wantA := mat.NewDense(2, 2, []float64{4, 3, 2, 1})
	wantB := mat.NewDense(2, 1, []float64{6, 5})
	wantC := mat.NewDense(1, 2, []float64{8, 7})

	if !matClose(result.A, wantA, 1e-12) {
		t.Errorf("A2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.A), mat.Formatted(wantA))
	}
	if !matClose(result.B, wantB, 1e-12) {
		t.Errorf("B2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.B), mat.Formatted(wantB))
	}
	if !matClose(result.C, wantC, 1e-12) {
		t.Errorf("C2 mismatch:\ngot  %v\nwant %v", mat.Formatted(result.C), mat.Formatted(wantC))
	}
}

func TestXperm_Identity(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := mat.NewDense(2, 1, []float64{5, 6})
	C := mat.NewDense(1, 2, []float64{7, 8})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	result, err := Xperm(sys, []int{0, 1})
	if err != nil {
		t.Fatal(err)
	}

	if !matClose(result.A, sys.A, 1e-12) {
		t.Error("A changed with identity perm")
	}
	if !matClose(result.B, sys.B, 1e-12) {
		t.Error("B changed with identity perm")
	}
	if !matClose(result.C, sys.C, 1e-12) {
		t.Error("C changed with identity perm")
	}
}

func TestXperm_Invalid(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := mat.NewDense(2, 1, []float64{5, 6})
	C := mat.NewDense(1, 2, []float64{7, 8})
	D := mat.NewDense(1, 1, []float64{0})

	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Xperm(sys, []int{0, 0})
	if err == nil {
		t.Error("expected error for duplicate indices")
	}

	_, err = Xperm(sys, []int{0, 2})
	if err == nil {
		t.Error("expected error for out-of-range index")
	}

	_, err = Xperm(sys, []int{0})
	if err == nil {
		t.Error("expected error for wrong perm length")
	}
}

// --- Inv tests ---

func TestInv_SISO(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	inv, err := Inv(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantA := mat.NewDense(1, 1, []float64{-3})
	wantB := mat.NewDense(1, 1, []float64{1})
	wantC := mat.NewDense(1, 1, []float64{-1})
	wantD := mat.NewDense(1, 1, []float64{1})

	if !matClose(inv.A, wantA, 1e-12) {
		t.Errorf("A_inv = %v, want %v", mat.Formatted(inv.A), mat.Formatted(wantA))
	}
	if !matClose(inv.B, wantB, 1e-12) {
		t.Errorf("B_inv = %v, want %v", mat.Formatted(inv.B), mat.Formatted(wantB))
	}
	if !matClose(inv.C, wantC, 1e-12) {
		t.Errorf("C_inv = %v, want %v", mat.Formatted(inv.C), mat.Formatted(wantC))
	}
	if !matClose(inv.D, wantD, 1e-12) {
		t.Errorf("D_inv = %v, want %v", mat.Formatted(inv.D), mat.Formatted(wantD))
	}
}

func TestInv_MIMO(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 1}),
		mat.NewDense(2, 2, []float64{1, 0, 0, 2}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	inv, err := Inv(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantDinv := mat.NewDense(2, 2, []float64{1, 0, 0, 0.5})
	if !matClose(inv.D, wantDinv, 1e-12) {
		t.Errorf("D_inv = %v, want %v", mat.Formatted(inv.D), mat.Formatted(wantDinv))
	}
}

func TestInv_NonSquare(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Inv(sys)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("expected ErrDimensionMismatch, got %v", err)
	}
}

func TestInv_SingularD(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	_, err = Inv(sys)
	if !errors.Is(err, ErrSingularTransform) {
		t.Errorf("expected ErrSingularTransform, got %v", err)
	}
}

func TestInv_DiscretePreservesDt(t *testing.T) {
	sys, err := New(
		mat.NewDense(1, 1, []float64{0.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		0.01,
	)
	if err != nil {
		t.Fatal(err)
	}

	inv, err := Inv(sys)
	if err != nil {
		t.Fatal(err)
	}
	if inv.Dt != 0.01 {
		t.Errorf("Dt = %v, want 0.01", inv.Dt)
	}
}

// --- Pzmap tests ---

func TestPzmap_PolesAndZeros(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -6, -5}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 1}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := Pzmap(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantPoles := []complex128{-2, -3}
	wantZeros := []complex128{-1}

	if !polesClose(result.Poles, wantPoles, 1e-10) {
		t.Errorf("poles = %v, want %v", result.Poles, wantPoles)
	}
	if !polesClose(result.Zeros, wantZeros, 1e-10) {
		t.Errorf("zeros = %v, want %v", result.Zeros, wantZeros)
	}
}

func TestPzmap_NoZeros(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}

	result, err := Pzmap(sys)
	if err != nil {
		t.Fatal(err)
	}

	wantPoles := []complex128{-1, -2}
	if !polesClose(result.Poles, wantPoles, 1e-10) {
		t.Errorf("poles = %v, want %v", result.Poles, wantPoles)
	}
	if len(result.Zeros) != 0 {
		t.Errorf("zeros = %v, want empty", result.Zeros)
	}
}

// --- IsProper tests ---

func TestIsProper_StateSpace(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{0, 1, -2, -3}),
		mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}),
		0,
	)
	if err != nil {
		t.Fatal(err)
	}
	if !mustIsProper(t, sys) {
		t.Error("state-space should always be proper")
	}
}

func TestIsProper_TFProper(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1}}},
		Den: [][]float64{{1, 1}},
	}
	if !mustIsProper(t, tf) {
		t.Error("tf(1, [1 1]) should be proper")
	}
}

func TestIsProper_TFImproper(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 0}}},
		Den: [][]float64{{1}},
	}
	if mustIsProper(t, tf) {
		t.Error("tf([1 0], [1]) should be improper")
	}
}

func TestIsProper_TFEqualDegree(t *testing.T) {
	tf := &TransferFunc{
		Num: [][][]float64{{{1, 1}}},
		Den: [][]float64{{1, 2}},
	}
	if !mustIsProper(t, tf) {
		t.Error("tf([1 1], [1 2]) should be proper (equal degree)")
	}
}
