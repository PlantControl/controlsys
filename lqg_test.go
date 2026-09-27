package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestLqg_ContinuousDoubleIntegrator(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	Q := eye(2)
	R := eye(1)
	Qn := eye(1)
	Rn := eye(1)

	res, err := Lqg(sys, Q, R, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	cn, cm, cp := res.Controller.Dims()
	if cn != 2 || cm != 1 || cp != 1 {
		t.Fatalf("controller dims = (%d,%d,%d), want (2,1,1)", cn, cm, cp)
	}

	cl, err := Feedback(sys, res.Controller, 1)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		poles, _ := cl.Poles()
		t.Errorf("closed-loop should be stable, poles=%v", poles)
	}
}

func TestLqg_MatchesManualComposition(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, -2, -3})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	Q := eye(2)
	R := eye(1)
	Qn := eye(1)
	Rn := eye(1)

	lqrRes, err := Lqr(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	kalRes, err := Kalman(sys, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}
	wantCtrl, err := Reg(sys, lqrRes.K, kalRes.K)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Lqg(sys, Q, R, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	assertMatEqual(t, "K", res.K, lqrRes.K, 1e-12)
	assertMatEqual(t, "L", res.L, kalRes.K, 1e-12)
	assertMatEqual(t, "Xc", res.Xc, lqrRes.X, 1e-12)
	assertMatEqual(t, "Xf", res.Xf, kalRes.X, 1e-12)
	assertMatEqual(t, "Controller.A", res.Controller.A, wantCtrl.A, 1e-12)
	assertMatEqual(t, "Controller.B", res.Controller.B, wantCtrl.B, 1e-12)
	assertMatEqual(t, "Controller.C", res.Controller.C, wantCtrl.C, 1e-12)
	assertMatEqual(t, "Controller.D", res.Controller.D, wantCtrl.D, 1e-12)
}

func TestLqg_Discrete(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	csys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}
	dsys, err := csys.Discretize(0.1)
	if err != nil {
		t.Fatal(err)
	}

	Q := eye(2)
	R := eye(1)
	Qn := eye(1)
	Rn := eye(1)

	res, err := Lqg(dsys, Q, R, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	if !res.Controller.IsDiscrete() {
		t.Error("controller should be discrete")
	}
	if res.Controller.Dt != 0.1 {
		t.Errorf("controller Dt = %v, want 0.1", res.Controller.Dt)
	}

	cl, err := Feedback(dsys, res.Controller, 1)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		poles, _ := cl.Poles()
		t.Errorf("closed-loop should be stable, poles=%v", poles)
	}
}

func TestLqg_DimensionErrors(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0, 1, 0, 0})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, nil)
	sys, _ := New(A, B, C, D, 0)

	t.Run("bad Q", func(t *testing.T) {
		Q := eye(3)
		R := eye(1)
		Qn := eye(1)
		Rn := eye(1)
		_, err := Lqg(sys, Q, R, Qn, Rn, nil)
		if err == nil {
			t.Error("expected error for wrong Q dims")
		}
	})

	t.Run("bad Rn", func(t *testing.T) {
		Q := eye(2)
		R := eye(1)
		Qn := eye(1)
		Rn := eye(3)
		_, err := Lqg(sys, Q, R, Qn, Rn, nil)
		if err == nil {
			t.Error("expected error for wrong Rn dims")
		}
	})

	t.Run("no states", func(t *testing.T) {
		gain, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
		_, err := Lqg(gain, eye(1), eye(1), eye(1), eye(1), nil)
		if !errors.Is(err, ErrDimensionMismatch) {
			t.Errorf("expected ErrDimensionMismatch, got %v", err)
		}
	})
}

func TestLqg_MIMO(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 0.5, 0,
		0, -2, 1,
		0, 0, -3,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		1, 1,
	})
	C := mat.NewDense(2, 3, []float64{
		1, 0, 0,
		0, 1, 0,
	})
	D := mat.NewDense(2, 2, nil)
	sys, err := New(A, B, C, D, 0)
	if err != nil {
		t.Fatal(err)
	}

	Q := eye(3)
	R := eye(2)
	Qn := eye(2)
	Rn := eye(2)

	res, err := Lqg(sys, Q, R, Qn, Rn, nil)
	if err != nil {
		t.Fatal(err)
	}

	cn, cm, cp := res.Controller.Dims()
	if cn != 3 || cm != 2 || cp != 2 {
		t.Fatalf("controller dims = (%d,%d,%d), want (3,2,2)", cn, cm, cp)
	}

	kr, kc := res.K.Dims()
	if kr != 2 || kc != 3 {
		t.Errorf("K dims = (%d,%d), want (2,3)", kr, kc)
	}
	lr, lc := res.L.Dims()
	if lr != 3 || lc != 2 {
		t.Errorf("L dims = (%d,%d), want (3,2)", lr, lc)
	}

	cl, err := Feedback(sys, res.Controller, 1)
	if err != nil {
		t.Fatal(err)
	}
	stable, err := cl.IsStable()
	if err != nil {
		t.Fatal(err)
	}
	if !stable {
		poles, _ := cl.Poles()
		t.Errorf("closed-loop should be stable, poles=%v", poles)
	}
}

func assertMatEqual(t *testing.T, name string, got, want *mat.Dense, tol float64) {
	t.Helper()
	gr, gc := got.Dims()
	wr, wc := want.Dims()
	if gr != wr || gc != wc {
		t.Errorf("%s dims = (%d,%d), want (%d,%d)", name, gr, gc, wr, wc)
		return
	}
	for i := range gr {
		for j := range gc {
			if math.Abs(got.At(i, j)-want.At(i, j)) > tol {
				t.Errorf("%s(%d,%d) = %v, want %v", name, i, j, got.At(i, j), want.At(i, j))
			}
		}
	}
}

func eye(n int) *mat.Dense {
	d := make([]float64, n*n)
	for i := range n {
		d[i*n+i] = 1
	}
	return mat.NewDense(n, n, d)
}
