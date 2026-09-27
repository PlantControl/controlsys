package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestCtrb_SISO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := mat.NewDense(2, 1, []float64{5, 7})

	got, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 2, []float64{5, 19, 7, 43})
	assertMatNearT(t, "Ctrb_SISO", got, want, 1e-10)
}

func TestCtrb_MIMO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	B := mat.NewDense(2, 2, []float64{5, 6, 7, 8})

	got, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 4, []float64{5, 6, 19, 22, 7, 8, 43, 50})
	assertMatNearT(t, "Ctrb_MIMO", got, want, 1e-10)
}

func TestCtrb_3x3(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		0, 1, 0,
		0, 0, 1,
		-6, -11, -6,
	})
	B := mat.NewDense(3, 1, []float64{0, 0, 1})

	got, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}
	r, c := got.Dims()
	if r != 3 || c != 3 {
		t.Fatalf("dims = (%d,%d), want (3,3)", r, c)
	}
	want := mat.NewDense(3, 3, []float64{
		0, 0, 1,
		0, 1, -6,
		1, -6, 25,
	})
	assertMatNearT(t, "Ctrb_3x3", got, want, 1e-10)
}

func TestCtrb_DimMismatch(t *testing.T) {
	_, err := Ctrb(mat.NewDense(2, 3, nil), mat.NewDense(2, 1, nil))
	if err != ErrDimensionMismatch {
		t.Errorf("non-square A: got %v, want ErrDimensionMismatch", err)
	}
	_, err = Ctrb(mat.NewDense(2, 2, nil), mat.NewDense(3, 1, nil))
	if err != ErrDimensionMismatch {
		t.Errorf("B rows mismatch: got %v, want ErrDimensionMismatch", err)
	}
}

func TestCtrb_Empty(t *testing.T) {
	A := &mat.Dense{}
	B := &mat.Dense{}
	got, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}
	r, c := got.Dims()
	if r != 0 || c != 0 {
		t.Errorf("dims = (%d,%d), want (0,0)", r, c)
	}
}

func TestObsv_SISO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	C := mat.NewDense(1, 2, []float64{5, 7})

	got, err := Obsv(A, C)
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 2, []float64{5, 7, 26, 38})
	assertMatNearT(t, "Obsv_SISO", got, want, 1e-10)
}

func TestObsv_MIMO(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 2, 3, 4})
	C := mat.NewDense(2, 2, []float64{5, 6, 7, 8})

	got, err := Obsv(A, C)
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(4, 2, []float64{
		5, 6,
		7, 8,
		23, 34,
		31, 46,
	})
	assertMatNearT(t, "Obsv_MIMO", got, want, 1e-10)
}

func TestObsv_DimMismatch(t *testing.T) {
	_, err := Obsv(mat.NewDense(2, 3, nil), mat.NewDense(1, 2, nil))
	if err != ErrDimensionMismatch {
		t.Errorf("non-square A: got %v, want ErrDimensionMismatch", err)
	}
	_, err = Obsv(mat.NewDense(2, 2, nil), mat.NewDense(1, 3, nil))
	if err != ErrDimensionMismatch {
		t.Errorf("C cols mismatch: got %v, want ErrDimensionMismatch", err)
	}
}

func TestObsv_Empty(t *testing.T) {
	got, err := Obsv(&mat.Dense{}, &mat.Dense{})
	if err != nil {
		t.Fatal(err)
	}
	r, c := got.Dims()
	if r != 0 || c != 0 {
		t.Errorf("dims = (%d,%d), want (0,0)", r, c)
	}
}

func TestCtrbF_FullRank(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})

	res, err := CtrbF(A, B, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.NCont != 2 {
		t.Errorf("NCont = %d, want 2", res.NCont)
	}
}

func TestCtrbF_PartialRank(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	B := mat.NewDense(2, 1, []float64{1, 0})

	res, err := CtrbF(A, B, nil)
	if err != nil {
		t.Fatal(err)
	}
	if res.NCont != 1 {
		t.Errorf("NCont = %d, want 1", res.NCont)
	}
}

func TestCtrbF_DimMismatch(t *testing.T) {
	_, err := CtrbF(mat.NewDense(2, 3, nil), mat.NewDense(2, 1, nil), nil)
	if err != ErrDimensionMismatch {
		t.Errorf("got %v, want ErrDimensionMismatch", err)
	}
}

func TestObsvF_FullRank(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	C := mat.NewDense(1, 2, []float64{1, 0})

	res, err := ObsvF(A, nil, C)
	if err != nil {
		t.Fatal(err)
	}
	if res.NCont != 2 {
		t.Errorf("NObs = %d, want 2", res.NCont)
	}
}

func TestObsvF_PartialRank(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	C := mat.NewDense(1, 2, []float64{1, 0})

	res, err := ObsvF(A, nil, C)
	if err != nil {
		t.Fatal(err)
	}
	if res.NCont != 1 {
		t.Errorf("NObs = %d, want 1", res.NCont)
	}
}

func TestObsvF_Duality(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 2, 0,
		0, -3, 1,
		0, 0, -5,
	})
	B := mat.NewDense(3, 1, []float64{1, 0, 0})
	C := mat.NewDense(1, 3, []float64{0, 0, 1})

	ctrbRes, _ := CtrbF(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C.T()), nil)
	obsvRes, _ := ObsvF(A, B, C)

	if ctrbRes.NCont != obsvRes.NCont {
		t.Errorf("duality: CtrbF(A',C').NCont=%d != ObsvF(A,B,C).NCont=%d",
			ctrbRes.NCont, obsvRes.NCont)
	}
}

func TestObsvF_DimMismatch(t *testing.T) {
	_, err := ObsvF(mat.NewDense(2, 3, nil), nil, mat.NewDense(1, 2, nil))
	if err != ErrDimensionMismatch {
		t.Errorf("got %v, want ErrDimensionMismatch", err)
	}
}

// python-control: duality ctrb(A,B) == obsv(A',B')'
func TestCtrb_Obsv_Duality(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1.2, -2.3, 3.4, -4.5})
	B := mat.NewDense(2, 2, []float64{5.8, 6.9, 8.0, 9.1})

	wc, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}

	wo, err := Obsv(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T()))
	if err != nil {
		t.Fatal(err)
	}

	woT := mat.DenseCopyOf(wo.T())
	assertMatNearT(t, "duality", wc, woT, 1e-10)
}

func TestCtrbF_WithC(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})

	res, err := CtrbF(A, B, C)
	if err != nil {
		t.Fatal(err)
	}
	if res.NCont != 2 {
		t.Errorf("NCont = %d, want 2", res.NCont)
	}
	if res.C == nil {
		t.Fatal("expected transformed C")
	}
	cr, cc := res.C.Dims()
	if cr != 1 || cc != 2 {
		t.Errorf("C dims = (%d,%d), want (1,2)", cr, cc)
	}
}

func TestCtrbF_PartialRank_WithC(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 0, 0,
		0, -2, 0,
		0, 0, -3,
	})
	B := mat.NewDense(3, 1, []float64{1, 0, 0})
	C := mat.NewDense(1, 3, []float64{1, 1, 1})

	res, err := CtrbF(A, B, C)
	if err != nil {
		t.Fatal(err)
	}
	if res.NCont != 1 {
		t.Errorf("NCont = %d, want 1", res.NCont)
	}
}

func TestObsvF_WithB(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})

	res, err := ObsvF(A, B, C)
	if err != nil {
		t.Fatal(err)
	}
	if res.NCont != 2 {
		t.Errorf("NObs = %d, want 2", res.NCont)
	}
	if res.B == nil {
		t.Fatal("expected transformed B")
	}
}

func TestCtrb_1x1(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-5})
	B := mat.NewDense(1, 1, []float64{3})

	got, err := Ctrb(A, B)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.At(0, 0)-3) > 1e-10 {
		t.Errorf("Ctrb = %v, want [[3]]", mat.Formatted(got))
	}
}

func TestObsv_1x1(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{-5})
	C := mat.NewDense(1, 1, []float64{2})

	got, err := Obsv(A, C)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(got.At(0, 0)-2) > 1e-10 {
		t.Errorf("Obsv = %v, want [[2]]", mat.Formatted(got))
	}
}

func TestIsStabilizable_FullyControllable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	ok, err := IsStabilizable(A, B, true)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Error("expected stabilizable (fully controllable)")
	}
}

func TestIsStabilizable_StableUncontrollable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
	B := mat.NewDense(2, 1, []float64{1, 0})
	ok, err := IsStabilizable(A, B, true)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Error("expected stabilizable (uncontrollable mode is stable)")
	}
}

func TestIsStabilizable_UnstableUncontrollable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, -2})
	B := mat.NewDense(2, 1, []float64{0, 1})
	ok, err := IsStabilizable(A, B, true)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Error("expected NOT stabilizable (uncontrollable mode is unstable)")
	}
}

func TestIsStabilizable_Discrete(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0, 0, 1.5})
	B := mat.NewDense(2, 1, []float64{0, 1})
	ok, err := IsStabilizable(A, B, false)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Error("expected stabilizable (uncontrollable mode |0.5| < 1)")
	}

	A2 := mat.NewDense(2, 2, []float64{1.5, 0, 0, 0.5})
	B2 := mat.NewDense(2, 1, []float64{0, 1})
	ok2, err := IsStabilizable(A2, B2, false)
	if err != nil {
		t.Fatal(err)
	}
	if ok2 {
		t.Error("expected NOT stabilizable (uncontrollable mode |1.5| >= 1)")
	}
}

func TestIsDetectable_Dual(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	C := mat.NewDense(1, 2, []float64{1, 0})
	ok, err := IsDetectable(A, C, true)
	if err != nil {
		t.Fatal(err)
	}
	if !ok {
		t.Error("expected detectable (fully observable)")
	}
}

func TestIsDetectable_Undetectable(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, -1})
	C := mat.NewDense(1, 2, []float64{0, 1})
	ok, err := IsDetectable(A, C, true)
	if err != nil {
		t.Fatal(err)
	}
	if ok {
		t.Error("expected NOT detectable (unobservable mode is unstable)")
	}
}

func TestIsStabilizable_DimError(t *testing.T) {
	_, err := IsStabilizable(mat.NewDense(2, 3, nil), mat.NewDense(2, 1, nil), true)
	if err != ErrDimensionMismatch {
		t.Errorf("non-square A: got %v, want ErrDimensionMismatch", err)
	}
	_, err = IsStabilizable(mat.NewDense(2, 2, nil), mat.NewDense(3, 1, nil), true)
	if err != ErrDimensionMismatch {
		t.Errorf("B rows mismatch: got %v, want ErrDimensionMismatch", err)
	}
}

func assertMatNearT(t *testing.T, label string, got, want *mat.Dense, tol float64) {
	t.Helper()
	gr, gc := got.Dims()
	wr, wc := want.Dims()
	if gr != wr || gc != wc {
		t.Fatalf("%s: dims (%d,%d), want (%d,%d)", label, gr, gc, wr, wc)
	}
	gRaw := got.RawMatrix()
	wRaw := want.RawMatrix()
	for i := range gr {
		for j := range gc {
			g := gRaw.Data[i*gRaw.Stride+j]
			w := wRaw.Data[i*wRaw.Stride+j]
			if math.Abs(g-w) > tol {
				t.Errorf("%s[%d,%d] = %.15g, want %.15g", label, i, j, g, w)
			}
		}
	}
}
