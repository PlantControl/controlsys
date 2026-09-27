package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestLinearize_LinearSystem(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-1, 2, 0, -3})
	B := mat.NewDense(2, 1, []float64{1, 0.5})
	C := mat.NewDense(1, 2, []float64{1, -1})
	D := mat.NewDense(1, 1, []float64{0.25})

	model := &NonlinearModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			res := mat.NewVecDense(2, nil)
			res.MulVec(A, x)
			var bu mat.VecDense
			bu.MulVec(B, u)
			res.AddVec(res, &bu)
			return res
		},
		H: func(x, u *mat.VecDense) *mat.VecDense {
			res := mat.NewVecDense(1, nil)
			res.MulVec(C, x)
			var du mat.VecDense
			du.MulVec(D, u)
			res.AddVec(res, &du)
			return res
		},
		N: 2, M: 1, P: 1,
	}

	x0 := mat.NewVecDense(2, []float64{3, -1})
	u0 := mat.NewVecDense(1, []float64{2})

	sys, err := Linearize(model, x0, u0)
	if err != nil {
		t.Fatal(err)
	}

	const tol = 1e-7
	checkDense(t, "A", sys.A, A, tol)
	checkDense(t, "B", sys.B, B, tol)
	checkDense(t, "C", sys.C, C, tol)
	checkDense(t, "D", sys.D, D, tol)
}

func TestLinearize_Pendulum(t *testing.T) {
	const g, L = 9.81, 1.0

	model := &NonlinearModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			return mat.NewVecDense(2, []float64{
				x.AtVec(1),
				-g / L * math.Sin(x.AtVec(0)),
			})
		},
		H: func(x, u *mat.VecDense) *mat.VecDense {
			return mat.NewVecDense(1, []float64{x.AtVec(0)})
		},
		N: 2, M: 1, P: 1,
	}

	x0 := mat.NewVecDense(2, []float64{0, 0})
	u0 := mat.NewVecDense(1, []float64{0})

	sys, err := Linearize(model, x0, u0)
	if err != nil {
		t.Fatal(err)
	}

	wantA := mat.NewDense(2, 2, []float64{0, 1, -g / L, 0})
	wantB := mat.NewDense(2, 1, []float64{0, 0})
	wantC := mat.NewDense(1, 2, []float64{1, 0})
	wantD := mat.NewDense(1, 1, []float64{0})

	const tol = 1e-7
	checkDense(t, "A", sys.A, wantA, tol)
	checkDense(t, "B", sys.B, wantB, tol)
	checkDense(t, "C", sys.C, wantC, tol)
	checkDense(t, "D", sys.D, wantD, tol)
}

func TestLinearize_VanDerPol(t *testing.T) {
	const mu = 1.0

	model := &NonlinearModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			x1, x2 := x.AtVec(0), x.AtVec(1)
			return mat.NewVecDense(2, []float64{
				x2,
				mu*(1-x1*x1)*x2 - x1,
			})
		},
		H: func(x, u *mat.VecDense) *mat.VecDense {
			return mat.NewVecDense(1, []float64{x.AtVec(0)})
		},
		N: 2, M: 1, P: 1,
	}

	x0 := mat.NewVecDense(2, []float64{0, 0})
	u0 := mat.NewVecDense(1, []float64{0})

	sys, err := Linearize(model, x0, u0)
	if err != nil {
		t.Fatal(err)
	}

	wantA := mat.NewDense(2, 2, []float64{0, 1, -1, mu})

	const tol = 1e-7
	checkDense(t, "A", sys.A, wantA, tol)
}

func TestLinearize_WithInput(t *testing.T) {
	model := &NonlinearModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			x1, x2 := x.AtVec(0), x.AtVec(1)
			u1 := u.AtVec(0)
			return mat.NewVecDense(2, []float64{
				-x1 + u1,
				x1 * x2,
			})
		},
		H: func(x, u *mat.VecDense) *mat.VecDense {
			return mat.NewVecDense(1, []float64{x.AtVec(0) + u.AtVec(0)})
		},
		N: 2, M: 1, P: 1,
	}

	x0 := mat.NewVecDense(2, []float64{1, 0})
	u0 := mat.NewVecDense(1, []float64{0})

	sys, err := Linearize(model, x0, u0)
	if err != nil {
		t.Fatal(err)
	}

	wantA := mat.NewDense(2, 2, []float64{-1, 0, 0, 1})
	wantB := mat.NewDense(2, 1, []float64{1, 0})
	wantC := mat.NewDense(1, 2, []float64{1, 0})
	wantD := mat.NewDense(1, 1, []float64{1})

	const tol = 1e-7
	checkDense(t, "A", sys.A, wantA, tol)
	checkDense(t, "B", sys.B, wantB, tol)
	checkDense(t, "C", sys.C, wantC, tol)
	checkDense(t, "D", sys.D, wantD, tol)
}

func TestLinearize_NilChecks(t *testing.T) {
	validModel := &NonlinearModel{
		F: func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) },
		H: func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) },
		N: 1, M: 1, P: 1,
	}
	x0 := mat.NewVecDense(1, []float64{0})
	u0 := mat.NewVecDense(1, []float64{0})

	if _, err := Linearize(nil, x0, u0); err == nil {
		t.Error("expected error for nil model")
	}
	if _, err := Linearize(validModel, nil, u0); err == nil {
		t.Error("expected error for nil x0")
	}
	if _, err := Linearize(validModel, x0, nil); err == nil {
		t.Error("expected error for nil u0")
	}

	wrongX := mat.NewVecDense(3, []float64{0, 0, 0})
	if _, err := Linearize(validModel, wrongX, u0); err == nil {
		t.Error("expected error for wrong x0 dimension")
	}

	wrongU := mat.NewVecDense(2, []float64{0, 0})
	if _, err := Linearize(validModel, x0, wrongU); err == nil {
		t.Error("expected error for wrong u0 dimension")
	}
}

func TestLinearize_RejectsWrongFunctionOutputDimensions(t *testing.T) {
	x0 := mat.NewVecDense(2, []float64{0, 0})
	u0 := mat.NewVecDense(1, []float64{0})

	tests := []struct {
		name string
		F    func(x, u *mat.VecDense) *mat.VecDense
		H    func(x, u *mat.VecDense) *mat.VecDense
	}{
		{
			name: "state function",
			F:    func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) },
			H:    func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) },
		},
		{
			name: "measurement function",
			F:    func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(2, nil) },
			H:    func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(2, nil) },
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			model := &NonlinearModel{F: tc.F, H: tc.H, N: 2, M: 1, P: 1}
			if _, err := Linearize(model, x0, u0); !errors.Is(err, ErrDimensionMismatch) {
				t.Fatalf("got %v, want ErrDimensionMismatch", err)
			}
		})
	}
}

func checkDense(t *testing.T, name string, got, want *mat.Dense, tol float64) {
	t.Helper()
	r, c := want.Dims()
	gr, gc := got.Dims()
	if gr != r || gc != c {
		t.Errorf("%s dims = (%d,%d), want (%d,%d)", name, gr, gc, r, c)
		return
	}
	for i := range r {
		for j := range c {
			if math.Abs(got.At(i, j)-want.At(i, j)) > tol {
				t.Errorf("%s(%d,%d) = %v, want %v", name, i, j, got.At(i, j), want.At(i, j))
			}
		}
	}
}
