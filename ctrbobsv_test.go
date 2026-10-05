package controlsys

import (
	"errors"
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
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("non-square A: got %v, want ErrDimensionMismatch", err)
	}
	_, err = Ctrb(mat.NewDense(2, 2, nil), mat.NewDense(3, 1, nil))
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("B rows mismatch: got %v, want ErrDimensionMismatch", err)
	}
}

func TestCtrb_Empty(t *testing.T) {
	if _, err := Ctrb(&mat.Dense{}, &mat.Dense{}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("no states: err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := Ctrb(nil, mat.NewDense(1, 1, nil)); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil A: err = %v, want ErrInvalidArgument", err)
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
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("non-square A: got %v, want ErrDimensionMismatch", err)
	}
	_, err = Obsv(mat.NewDense(2, 2, nil), mat.NewDense(1, 3, nil))
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("C cols mismatch: got %v, want ErrDimensionMismatch", err)
	}
}

func TestObsv_Empty(t *testing.T) {
	if _, err := Obsv(&mat.Dense{}, &mat.Dense{}); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("no states: err = %v, want ErrDimensionMismatch", err)
	}
	if _, err := Obsv(nil, mat.NewDense(1, 1, nil)); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("nil A: err = %v, want ErrInvalidArgument", err)
	}
}

func TestCtrbF_FullRank(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})

	res, err := CtrbF(A, B, testOnesRow(A), 0)
	if err != nil {
		t.Fatal(err)
	}
	if sumInts(res.K) != 2 {
		t.Errorf("NCont = %d, want 2", sumInts(res.K))
	}
}

func TestCtrbF_PartialRank(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	B := mat.NewDense(2, 1, []float64{1, 0})

	res, err := CtrbF(A, B, testOnesRow(A), 0)
	if err != nil {
		t.Fatal(err)
	}
	if sumInts(res.K) != 1 {
		t.Errorf("NCont = %d, want 1", sumInts(res.K))
	}
}

func TestCtrbF_DimMismatch(t *testing.T) {
	_, err := CtrbF(mat.NewDense(2, 3, nil), mat.NewDense(2, 1, nil), testOnesRow(mat.NewDense(2, 3, nil)), 0)
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("got %v, want ErrDimensionMismatch", err)
	}
}

func TestObsvF_FullRank(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	C := mat.NewDense(1, 2, []float64{1, 0})

	res, err := ObsvF(A, testOnesCol(A), C, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sumInts(res.K) != 2 {
		t.Errorf("NObs = %d, want 2", sumInts(res.K))
	}
}

func TestObsvF_PartialRank(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1, 0, 0, 2})
	C := mat.NewDense(1, 2, []float64{1, 0})

	res, err := ObsvF(A, testOnesCol(A), C, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sumInts(res.K) != 1 {
		t.Errorf("NObs = %d, want 1", sumInts(res.K))
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

	ctrbRes, _ := CtrbF(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(C.T()), mat.DenseCopyOf(B.T()), 0)
	obsvRes, _ := ObsvF(A, B, C, 0)

	if sumInts(ctrbRes.K) != sumInts(obsvRes.K) {
		t.Errorf("duality: CtrbF(A',C').NCont=%d != ObsvF(A,B,C).NCont=%d",
			sumInts(ctrbRes.K), sumInts(obsvRes.K))
	}
}

func TestObsvF_DimMismatch(t *testing.T) {
	_, err := ObsvF(mat.NewDense(2, 3, nil), testOnesCol(mat.NewDense(2, 3, nil)), mat.NewDense(1, 2, nil), 0)
	if !errors.Is(err, ErrDimensionMismatch) {
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

	res, err := CtrbF(A, B, C, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sumInts(res.K) != 2 {
		t.Errorf("NCont = %d, want 2", sumInts(res.K))
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

	res, err := CtrbF(A, B, C, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sumInts(res.K) != 1 {
		t.Errorf("NCont = %d, want 1", sumInts(res.K))
	}
}

func TestObsvF_WithB(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})

	res, err := ObsvF(A, B, C, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sumInts(res.K) != 2 {
		t.Errorf("NObs = %d, want 2", sumInts(res.K))
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
	if !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("non-square A: got %v, want ErrDimensionMismatch", err)
	}
	_, err = IsStabilizable(mat.NewDense(2, 2, nil), mat.NewDense(3, 1, nil), true)
	if !errors.Is(err, ErrDimensionMismatch) {
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

func testOnesRow(A *mat.Dense) *mat.Dense {
	_, n := A.Dims()
	r := mat.NewDense(1, n, nil)
	for j := range n {
		r.Set(0, j, 1)
	}
	return r
}

func testOnesCol(A *mat.Dense) *mat.Dense {
	n, _ := A.Dims()
	c := mat.NewDense(n, 1, nil)
	for i := range n {
		c.Set(i, 0, 1)
	}
	return c
}

func sumInts(v []int) int {
	s := 0
	for _, x := range v {
		s += x
	}
	return s
}

func TestCtrbFObsvFMatlabForm(t *testing.T) {
	c, s := math.Cos(0.7), math.Sin(0.7)
	R := mat.NewDense(3, 3, []float64{c, 0, -s, 0, 1, 0, s, 0, c})
	A0 := mat.NewDense(3, 3, []float64{1, 2, 0.4, 3, 4, -0.2, 0, 0, 5})
	B0 := mat.NewDense(3, 2, []float64{1, 0.5, 0, 0, 0, 0})
	var A, B mat.Dense
	A.Product(R, A0, R.T())
	B.Mul(R, B0)
	C := mat.NewDense(2, 3, []float64{1, -0.3, 2, 0.5, 1, 0})

	check := func(label string, f *StaircaseForm, A, B, C *mat.Dense) {
		t.Helper()
		var tt, want mat.Dense
		tt.Mul(f.T, f.T.T())
		assertMatNearT(t, label+" T·Tᵀ", &tt, eye(3), 1e-12)
		want.Product(f.T, A, f.T.T())
		assertMatNearT(t, label+" Abar", f.A, &want, 1e-12)
		want.Reset()
		want.Mul(f.T, B)
		assertMatNearT(t, label+" Bbar", f.B, &want, 1e-12)
		want.Reset()
		want.Mul(C, f.T.T())
		assertMatNearT(t, label+" Cbar", f.C, &want, 1e-12)
	}

	cf, err := CtrbF(&A, &B, C, 0)
	if err != nil {
		t.Fatal(err)
	}
	if sumInts(cf.K) != 2 {
		t.Fatalf("CtrbF K = %v, want sum 2", cf.K)
	}
	check("CtrbF", cf, &A, &B, C)
	for j := 1; j < 3; j++ {
		if v := cf.A.At(0, j); math.Abs(v) > 1e-12 {
			t.Errorf("CtrbF Abar[0,%d] = %g, want 0 (Anc block decoupled)", j, v)
		}
	}
	for j := range 2 {
		if v := cf.B.At(0, j); math.Abs(v) > 1e-12 {
			t.Errorf("CtrbF Bbar[0,%d] = %g, want 0", j, v)
		}
	}
	if math.Abs(cf.A.At(0, 0)-5) > 1e-12 {
		t.Errorf("uncontrollable eigenvalue = %g, want 5", cf.A.At(0, 0))
	}

	var At, Ct, Bt mat.Dense
	At.CloneFrom(A.T())
	Ct.CloneFrom(C.T())
	Bt.CloneFrom(B.T())
	of, err := ObsvF(&At, &Ct, &Bt, 0)
	if err != nil {
		t.Fatal(err)
	}
	check("ObsvF", of, &At, &Ct, &Bt)
	for i := 1; i < 3; i++ {
		if v := of.A.At(i, 0); math.Abs(v) > 1e-12 {
			t.Errorf("ObsvF Abar[%d,0] = %g, want 0", i, v)
		}
	}
	for i := range 2 {
		if v := of.C.At(i, 0); math.Abs(v) > 1e-12 {
			t.Errorf("ObsvF Cbar[%d,0] = %g, want 0", i, v)
		}
	}

	if _, err := CtrbF(&A, &B, nil, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("CtrbF nil C: err = %v", err)
	}
	if _, err := ObsvF(&A, nil, C, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("ObsvF nil B: err = %v", err)
	}
	if _, err := CtrbF(&A, &B, C, -1); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("CtrbF negative tol: err = %v", err)
	}
	bad := mat.DenseCopyOf(&A)
	bad.Set(1, 1, math.NaN())
	if _, err := CtrbF(bad, &B, C, 0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("CtrbF NaN A: err = %v", err)
	}
	if _, err := IsStabilizable(bad, &B, true); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("IsStabilizable NaN A: err = %v", err)
	}
	if _, err := IsDetectable(bad, C, true); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("IsDetectable NaN A: err = %v", err)
	}
	if _, err := IsStabilizable(nil, &B, true); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("IsStabilizable nil A: err = %v", err)
	}
}
