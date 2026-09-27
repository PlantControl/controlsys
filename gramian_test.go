package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/blas"
	"plantcontrol.org/v1/gonum/blas/blas64"
	"plantcontrol.org/v1/gonum/mat"
)

func gramResidualCont(A, X, Q *mat.Dense) float64 {
	n, _ := A.Dims()
	aRaw := A.RawMatrix()
	xRaw := X.RawMatrix()
	qRaw := Q.RawMatrix()

	ax := make([]float64, n*n)
	aGen := blas64.General{Rows: n, Cols: n, Stride: aRaw.Stride, Data: aRaw.Data}
	xGen := blas64.General{Rows: n, Cols: n, Stride: xRaw.Stride, Data: xRaw.Data}
	rGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: ax}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, aGen, xGen, 0, rGen)

	sum := 0.0
	for i := range n {
		for j := range n {
			v := ax[i*n+j] + ax[j*n+i] + qRaw.Data[i*qRaw.Stride+j]
			sum += v * v
		}
	}
	return math.Sqrt(sum) / float64(n)
}

func gramResidualDisc(A, X, Q *mat.Dense) float64 {
	n, _ := A.Dims()
	aRaw := A.RawMatrix()
	xRaw := X.RawMatrix()
	qRaw := Q.RawMatrix()

	ax := make([]float64, n*n)
	aGen := blas64.General{Rows: n, Cols: n, Stride: aRaw.Stride, Data: aRaw.Data}
	xGen := blas64.General{Rows: n, Cols: n, Stride: xRaw.Stride, Data: xRaw.Data}
	axGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: ax}
	blas64.Gemm(blas.NoTrans, blas.NoTrans, 1, aGen, xGen, 0, axGen)

	axa := make([]float64, n*n)
	axaGen := blas64.General{Rows: n, Cols: n, Stride: n, Data: axa}
	blas64.Gemm(blas.NoTrans, blas.Trans, 1, axGen, aGen, 0, axaGen)

	sum := 0.0
	for i := range n {
		for j := range n {
			v := axa[i*n+j] - xRaw.Data[i*xRaw.Stride+j] + qRaw.Data[i*qRaw.Stride+j]
			sum += v * v
		}
	}
	return math.Sqrt(sum) / float64(n)
}

func TestGram_Controllability_1x1_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	res, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	got := res.X.RawMatrix().Data[0]
	want := 0.5
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Wc = %g, want %g", got, want)
	}
}

func TestGram_Observability_1x1_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{-2}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{3}),
		mat.NewDense(1, 1, []float64{0}), 0)

	res, err := Gram(sys, GramObservability)
	if err != nil {
		t.Fatal(err)
	}
	got := res.X.RawMatrix().Data[0]
	want := 9.0 / 4.0
	if math.Abs(got-want) > 1e-12 {
		t.Errorf("Wo = %g, want %g", got, want)
	}
}

func TestGram_Controllability_2x2_NonSymA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	res, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}

	BBt := mat.NewDense(2, 2, nil)
	BBt.Mul(B, B.T())
	r := gramResidualCont(A, res.X, BBt)
	if r > 1e-10 {
		t.Errorf("residual = %e", r)
	}
	if !isSymmetric(res.X, 1e-10) {
		t.Error("gramian not symmetric")
	}
}

func TestGram_Observability_2x2_NonSymA(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0)

	res, err := Gram(sys, GramObservability)
	if err != nil {
		t.Fatal(err)
	}

	CtC := mat.NewDense(2, 2, nil)
	CtC.Mul(C.T(), C)

	At := mat.DenseCopyOf(A.T())
	r := gramResidualCont(At, res.X, CtC)
	if r > 1e-10 {
		t.Errorf("residual = %e", r)
	}
}

func TestGram_Controllability_Discrete(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0.1)

	res, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}

	BBt := mat.NewDense(2, 2, nil)
	BBt.Mul(B, B.T())
	r := gramResidualDisc(A, res.X, BBt)
	if r > 1e-10 {
		t.Errorf("residual = %e", r)
	}
}

func TestGram_Observability_Discrete(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.5, 0.1, 0, 0.8})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, 0})
	D := mat.NewDense(1, 1, []float64{0})
	sys, _ := New(A, B, C, D, 0.1)

	res, err := Gram(sys, GramObservability)
	if err != nil {
		t.Fatal(err)
	}

	CtC := mat.NewDense(2, 2, nil)
	CtC.Mul(C.T(), C)
	At := mat.DenseCopyOf(A.T())
	r := gramResidualDisc(At, res.X, CtC)
	if r > 1e-10 {
		t.Errorf("residual = %e", r)
	}
}

func TestGram_MIMO(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{
		-1, 0.5, 0,
		0, -2, 0.3,
		0, 0, -3,
	})
	B := mat.NewDense(3, 2, []float64{
		1, 0,
		0, 1,
		1, 1,
	})
	C := mat.NewDense(2, 3, []float64{
		1, 0, 1,
		0, 1, 0,
	})
	D := mat.NewDense(2, 2, nil)
	sys, _ := New(A, B, C, D, 0)

	for _, typ := range []GramType{GramControllability, GramObservability} {
		res, err := Gram(sys, typ)
		if err != nil {
			t.Fatal(err)
		}
		if !isSymmetric(res.X, 1e-10) {
			t.Errorf("type=%d: gramian not symmetric", typ)
		}
	}
}

func TestGram_Unstable_Continuous(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Gram(sys, GramControllability)
	if !errors.Is(err, ErrUnstableGramian) {
		t.Errorf("got %v, want ErrUnstableGramian", err)
	}
}

func TestGram_Unstable_Discrete(t *testing.T) {
	sys, _ := New(
		mat.NewDense(1, 1, []float64{1.5}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{1}),
		mat.NewDense(1, 1, []float64{0}), 0.1)

	_, err := Gram(sys, GramControllability)
	if !errors.Is(err, ErrUnstableGramian) {
		t.Errorf("got %v, want ErrUnstableGramian", err)
	}
}

func TestGram_ZeroB(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{0, 0}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	res, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}

	raw := res.X.RawMatrix()
	for i := range raw.Rows {
		for j := range raw.Cols {
			if math.Abs(raw.Data[i*raw.Stride+j]) > 1e-14 {
				t.Errorf("X[%d,%d] = %g, want 0", i, j, raw.Data[i*raw.Stride+j])
			}
		}
	}
	if res.L != nil {
		t.Error("Cholesky factor should be nil for zero gramian")
	}
}

func TestGram_CholeskyFactor(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0.5, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	res, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	if res.L == nil {
		t.Fatal("Cholesky factor is nil")
	}

	LtL := mat.NewDense(2, 2, nil)
	LtL.Mul(res.L.T(), res.L)
	assertMatNearT(t, "L'L=X", LtL, res.X, 1e-10)
}

func TestGram_PythonControl_Wc(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
		mat.NewDense(2, 2, []float64{4, 5, 6, 7}),
		mat.NewDense(2, 2, []float64{13, 14, 15, 16}), 0)

	res, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 2, []float64{18.5, 24.5, 24.5, 32.5})
	assertMatNearT(t, "Wc", res.X, want, 1e-3)
}

func TestGram_PythonControl_Wo(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 2, []float64{5, 6, 7, 8}),
		mat.NewDense(2, 2, []float64{4, 5, 6, 7}),
		mat.NewDense(2, 2, []float64{13, 14, 15, 16}), 0)

	res, err := Gram(sys, GramObservability)
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 2, []float64{257.5, -94.5, -94.5, 56.5})
	assertMatNearT(t, "Wo", res.X, want, 1e-2)
}

func TestGram_PythonControl_Wc2(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{9}), 0)

	res, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 2, []float64{7.166667, 9.833333, 9.833333, 13.5})
	assertMatNearT(t, "Wc2", res.X, want, 1e-3)
}

func TestGram_PythonControl_Wo2(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{1, -2, 3, -4}),
		mat.NewDense(2, 1, []float64{5, 7}),
		mat.NewDense(1, 2, []float64{6, 8}),
		mat.NewDense(1, 1, []float64{9}), 0)

	res, err := Gram(sys, GramObservability)
	if err != nil {
		t.Fatal(err)
	}
	want := mat.NewDense(2, 2, []float64{198., -72., -72., 44.})
	assertMatNearT(t, "Wo2", res.X, want, 1e-2)
}

func TestGram_Empty(t *testing.T) {
	sys, _ := New(nil, nil, nil, mat.NewDense(1, 1, []float64{1}), 0)
	res, err := Gram(sys, GramControllability)
	if err != nil {
		t.Fatal(err)
	}
	r, c := res.X.Dims()
	if r != 0 || c != 0 {
		t.Errorf("dims = (%d,%d), want (0,0)", r, c)
	}
}
