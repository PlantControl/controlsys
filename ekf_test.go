package controlsys

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestEKF_LinearSystem(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.95})
	B := mat.NewDense(2, 1, []float64{0, 1})
	C := mat.NewDense(1, 2, []float64{1, 0})

	qScale := 0.01
	rScale := 0.1
	Q := mat.NewDense(2, 2, []float64{qScale, 0, 0, qScale})
	R := mat.NewDense(1, 1, []float64{rScale})

	model := &EKFModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			out := mat.NewVecDense(2, nil)
			out.MulVec(A, x)
			var bu mat.VecDense
			bu.MulVec(B, u)
			out.AddVec(out, &bu)
			return out
		},
		H: func(x *mat.VecDense) *mat.VecDense {
			out := mat.NewVecDense(1, nil)
			out.MulVec(C, x)
			return out
		},
		FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(A) },
		HJac: func(x *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(C) },
		Q:    Q,
		R:    R,
	}

	x0 := mat.NewVecDense(2, []float64{0, 0})
	P0 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})

	ekf, err := NewEKF(model, x0, P0)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(42, 0))
	xTrue := mat.NewVecDense(2, []float64{1, 0.5})
	u := mat.NewVecDense(1, []float64{0.1})

	nSteps := 50
	for k := range nSteps {
		wNoise := mat.NewVecDense(2, []float64{
			rng.NormFloat64() * math.Sqrt(qScale),
			rng.NormFloat64() * math.Sqrt(qScale),
		})
		xNext := mat.NewVecDense(2, nil)
		xNext.MulVec(A, xTrue)
		var bu mat.VecDense
		bu.MulVec(B, u)
		xNext.AddVec(xNext, &bu)
		xNext.AddVec(xNext, wNoise)

		vNoise := mat.NewVecDense(1, []float64{rng.NormFloat64() * math.Sqrt(rScale)})
		z := mat.NewVecDense(1, nil)
		z.MulVec(C, xNext)
		z.AddVec(z, vNoise)

		xTrue = xNext

		if err := ekf.Step(u, z); err != nil {
			t.Fatalf("step %d: %v", k, err)
		}
	}

	for i := range 2 {
		diff := math.Abs(ekf.x.AtVec(i) - xTrue.AtVec(i))
		if diff > 1.0 {
			t.Errorf("state %d: |ekf-true| = %v, want < 1.0", i, diff)
		}
	}
}

func TestEKFCopyAndNoiseMatricesAreIndependent(t *testing.T) {
	A := mat.NewDense(1, 1, []float64{1})
	C := mat.NewDense(1, 1, []float64{1})
	Q := mat.NewDense(1, 1, []float64{0.1})
	R := mat.NewDense(1, 1, []float64{0.2})
	model := &EKFModel{
		F:    func(x, u *mat.VecDense) *mat.VecDense { return mat.VecDenseCopyOf(x) },
		H:    func(x *mat.VecDense) *mat.VecDense { return mat.VecDenseCopyOf(x) },
		FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(A) },
		HJac: func(x *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(C) },
		Q:    Q,
		R:    R,
	}

	ekf, err := NewEKF(model, mat.NewVecDense(1, []float64{1}), mat.NewDense(1, 1, []float64{1}))
	if err != nil {
		t.Fatal(err)
	}
	cp := ekf.Copy()

	Q.Set(0, 0, 99)
	R.Set(0, 0, 99)
	if got := ekf.model.Q.At(0, 0); got != 0.1 {
		t.Fatalf("NewEKF Q aliases caller matrix: got %v, want 0.1", got)
	}
	if got := ekf.model.R.At(0, 0); got != 0.2 {
		t.Fatalf("NewEKF R aliases caller matrix: got %v, want 0.2", got)
	}

	ekf.x.SetVec(0, 99)
	ekf.cov.Set(0, 0, 99)
	ekf.model.Q.Set(0, 0, 98)
	ekf.model.R.Set(0, 0, 98)

	if got := cp.model.Q.At(0, 0); got != 0.1 {
		t.Fatalf("Copy Q = %v, want 0.1", got)
	}
	if got := cp.model.R.At(0, 0); got != 0.2 {
		t.Fatalf("Copy R = %v, want 0.2", got)
	}
	if got := cp.x.AtVec(0); got != 1 {
		t.Fatalf("Copy X = %v, want 1", got)
	}
	if got := cp.cov.At(0, 0); got != 1 {
		t.Fatalf("Copy P = %v, want 1", got)
	}
}

func TestEKF_NonlinearPendulum(t *testing.T) {
	const (
		g  = 9.81
		L  = 1.0
		dt = 0.01
	)

	model := &EKFModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			theta, omega := x.AtVec(0), x.AtVec(1)
			return mat.NewVecDense(2, []float64{
				theta + dt*omega,
				omega + dt*(-g/L*math.Sin(theta)),
			})
		},
		H: func(x *mat.VecDense) *mat.VecDense {
			return mat.NewVecDense(1, []float64{x.AtVec(0)})
		},
		FJac: func(x, u *mat.VecDense) *mat.Dense {
			theta := x.AtVec(0)
			return mat.NewDense(2, 2, []float64{
				1, dt,
				-dt * g / L * math.Cos(theta), 1,
			})
		},
		HJac: func(x *mat.VecDense) *mat.Dense {
			return mat.NewDense(1, 2, []float64{1, 0})
		},
		Q: mat.NewDense(2, 2, []float64{1e-4, 0, 0, 1e-4}),
		R: mat.NewDense(1, 1, []float64{0.01}),
	}

	x0 := mat.NewVecDense(2, []float64{0, 0})
	P0 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	ekf, err := NewEKF(model, x0, P0)
	if err != nil {
		t.Fatal(err)
	}

	rng := rand.New(rand.NewPCG(99, 0))
	xTrue := mat.NewVecDense(2, []float64{0.3, 0})
	noU := mat.NewVecDense(1, nil)

	nSteps := 100
	for k := range nSteps {
		theta, omega := xTrue.AtVec(0), xTrue.AtVec(1)
		xTrue = mat.NewVecDense(2, []float64{
			theta + dt*omega + rng.NormFloat64()*1e-2,
			omega + dt*(-g/L*math.Sin(theta)) + rng.NormFloat64()*1e-2,
		})

		z := mat.NewVecDense(1, []float64{xTrue.AtVec(0) + rng.NormFloat64()*0.1})

		if err := ekf.Step(noU, z); err != nil {
			t.Fatalf("step %d: %v", k, err)
		}
	}

	diff := math.Abs(ekf.x.AtVec(0) - xTrue.AtVec(0))
	if diff > 0.5 {
		t.Errorf("|theta_ekf - theta_true| = %v, want < 0.5", diff)
	}
}

func TestEKF_StepEquivalence(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.95})
	C := mat.NewDense(1, 2, []float64{1, 0})
	Q := mat.NewDense(2, 2, []float64{0.01, 0, 0, 0.01})
	R := mat.NewDense(1, 1, []float64{0.1})

	makeModel := func() *EKFModel {
		return &EKFModel{
			F: func(x, u *mat.VecDense) *mat.VecDense {
				out := mat.NewVecDense(2, nil)
				out.MulVec(A, x)
				return out
			},
			H: func(x *mat.VecDense) *mat.VecDense {
				out := mat.NewVecDense(1, nil)
				out.MulVec(C, x)
				return out
			},
			FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(A) },
			HJac: func(x *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(C) },
			Q:    Q,
			R:    R,
		}
	}

	x0 := mat.NewVecDense(2, []float64{1, 2})
	P0 := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
	u := mat.NewVecDense(1, nil)
	z := mat.NewVecDense(1, []float64{0.8})

	ekf1, _ := NewEKF(makeModel(), x0, P0)
	ekf2, _ := NewEKF(makeModel(), x0, P0)

	if err := ekf1.Predict(u); err != nil {
		t.Fatal(err)
	}
	if err := ekf1.Update(z); err != nil {
		t.Fatal(err)
	}
	if err := ekf2.Step(u, z); err != nil {
		t.Fatal(err)
	}

	for i := range 2 {
		if math.Abs(ekf1.x.AtVec(i)-ekf2.x.AtVec(i)) > 1e-14 {
			t.Errorf("X[%d]: predict+update=%v step=%v", i, ekf1.x.AtVec(i), ekf2.x.AtVec(i))
		}
	}
	r1, c1 := ekf1.cov.Dims()
	for i := range r1 {
		for j := range c1 {
			if math.Abs(ekf1.cov.At(i, j)-ekf2.cov.At(i, j)) > 1e-14 {
				t.Errorf("P[%d,%d]: predict+update=%v step=%v", i, j, ekf1.cov.At(i, j), ekf2.cov.At(i, j))
			}
		}
	}
}

func TestEKF_CovariancePSD(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.1, 0, 0.95})
	C := mat.NewDense(1, 2, []float64{1, 0})

	model := &EKFModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			out := mat.NewVecDense(2, nil)
			out.MulVec(A, x)
			return out
		},
		H: func(x *mat.VecDense) *mat.VecDense {
			out := mat.NewVecDense(1, nil)
			out.MulVec(C, x)
			return out
		},
		FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(A) },
		HJac: func(x *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(C) },
		Q:    mat.NewDense(2, 2, []float64{0.01, 0, 0, 0.01}),
		R:    mat.NewDense(1, 1, []float64{0.1}),
	}

	ekf, _ := NewEKF(model, mat.NewVecDense(2, []float64{1, 0}), mat.NewDense(2, 2, []float64{1, 0, 0, 1}))
	rng := rand.New(rand.NewPCG(7, 0))
	u := mat.NewVecDense(1, nil)

	for k := range 30 {
		z := mat.NewVecDense(1, []float64{rng.NormFloat64()})
		if err := ekf.Step(u, z); err != nil {
			t.Fatalf("step %d: %v", k, err)
		}

		n, _ := ekf.cov.Dims()
		for i := range n {
			for j := range n {
				if math.Abs(ekf.cov.At(i, j)-ekf.cov.At(j, i)) > 1e-12 {
					t.Fatalf("step %d: P not symmetric: P[%d,%d]=%v P[%d,%d]=%v",
						k, i, j, ekf.cov.At(i, j), j, i, ekf.cov.At(j, i))
				}
			}
		}

		var eig mat.Eigen
		ok := eig.Factorize(ekf.cov, mat.EigenNone)
		if !ok {
			t.Fatalf("step %d: eigen decomposition failed", k)
		}
		for _, ev := range eig.Values(nil) {
			if real(ev) < -1e-10 {
				t.Fatalf("step %d: negative eigenvalue %v", k, ev)
			}
		}
	}
}

func TestEKF_Validation(t *testing.T) {
	dummyF := func(x, u *mat.VecDense) *mat.VecDense { return x }
	dummyH := func(x *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) }
	dummyFJac := func(x, u *mat.VecDense) *mat.Dense { return mat.NewDense(2, 2, nil) }
	dummyHJac := func(x *mat.VecDense) *mat.Dense { return mat.NewDense(1, 2, nil) }

	x0 := mat.NewVecDense(2, nil)
	P0 := mat.NewDense(2, 2, nil)
	Q := mat.NewDense(2, 2, nil)
	R := mat.NewDense(1, 1, nil)

	t.Run("nil model", func(t *testing.T) {
		_, err := NewEKF(nil, x0, P0)
		if err == nil {
			t.Fatal("expected error for nil model")
		}
	})

	t.Run("nil function", func(t *testing.T) {
		_, err := NewEKF(&EKFModel{F: dummyF, H: dummyH, FJac: nil, HJac: dummyHJac, Q: Q, R: R}, x0, P0)
		if err == nil {
			t.Fatal("expected error for nil FJac")
		}
	})

	t.Run("P0 dimension mismatch", func(t *testing.T) {
		badP := mat.NewDense(3, 3, nil)
		m := &EKFModel{F: dummyF, H: dummyH, FJac: dummyFJac, HJac: dummyHJac, Q: Q, R: R}
		_, err := NewEKF(m, x0, badP)
		if err == nil {
			t.Fatal("expected error for P0 size mismatch")
		}
	})

	t.Run("Q dimension mismatch", func(t *testing.T) {
		badQ := mat.NewDense(3, 3, nil)
		m := &EKFModel{F: dummyF, H: dummyH, FJac: dummyFJac, HJac: dummyHJac, Q: badQ, R: R}
		_, err := NewEKF(m, x0, P0)
		if err == nil {
			t.Fatal("expected error for Q size mismatch")
		}
	})

	t.Run("nil x0", func(t *testing.T) {
		m := &EKFModel{F: dummyF, H: dummyH, FJac: dummyFJac, HJac: dummyHJac, Q: Q, R: R}
		_, err := NewEKF(m, nil, P0)
		if err == nil {
			t.Fatal("expected error for nil x0")
		}
	})
}

func TestEKF_RejectsWrongFunctionOutputDimensions(t *testing.T) {
	x0 := mat.NewVecDense(2, nil)
	P0 := mat.NewDense(2, 2, nil)
	Q := mat.NewDense(2, 2, nil)
	R := mat.NewDense(1, 1, nil)
	u := mat.NewVecDense(1, nil)
	z := mat.NewVecDense(1, nil)

	t.Run("predict state", func(t *testing.T) {
		model := &EKFModel{
			F:    func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) },
			H:    func(x *mat.VecDense) *mat.VecDense { return mat.NewVecDense(1, nil) },
			FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.NewDense(2, 2, nil) },
			HJac: func(x *mat.VecDense) *mat.Dense { return mat.NewDense(1, 2, nil) },
			Q:    Q,
			R:    R,
		}
		ekf, err := NewEKF(model, x0, P0)
		if err != nil {
			t.Fatal(err)
		}
		if err := ekf.Predict(u); !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("got %v, want ErrDimensionMismatch", err)
		}
	})

	t.Run("update measurement", func(t *testing.T) {
		badMeasurement := false
		model := &EKFModel{
			F: func(x, u *mat.VecDense) *mat.VecDense { return mat.NewVecDense(2, nil) },
			H: func(x *mat.VecDense) *mat.VecDense {
				if badMeasurement {
					return mat.NewVecDense(2, nil)
				}
				return mat.NewVecDense(1, nil)
			},
			FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.NewDense(2, 2, nil) },
			HJac: func(x *mat.VecDense) *mat.Dense { return mat.NewDense(1, 2, nil) },
			Q:    Q,
			R:    R,
		}
		ekf, err := NewEKF(model, x0, P0)
		if err != nil {
			t.Fatal(err)
		}
		badMeasurement = true
		if err := ekf.Update(z); !errors.Is(err, ErrDimensionMismatch) {
			t.Fatalf("got %v, want ErrDimensionMismatch", err)
		}
	})
}

func TestEKFDoesNotAliasFOutputAndValidatesState(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.9, 0.2, -0.1, 0.8})
	C := mat.NewDense(1, 2, []float64{1, 0.5})
	Q := mat.NewDense(2, 2, []float64{0.1, 0.01, 0.01, 0.2})
	R := mat.NewDense(1, 1, []float64{0.3})
	buf := mat.NewVecDense(2, nil)
	model := &EKFModel{
		F: func(x, u *mat.VecDense) *mat.VecDense {
			buf.MulVec(A, x)
			buf.AddScaledVec(buf, u.AtVec(0), mat.NewVecDense(2, []float64{1, 0.5}))
			return buf
		},
		H:    func(x *mat.VecDense) *mat.VecDense { v := mat.NewVecDense(1, nil); v.MulVec(C, x); return v },
		FJac: func(x, u *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(A) },
		HJac: func(x *mat.VecDense) *mat.Dense { return mat.DenseCopyOf(C) },
		Q:    Q,
		R:    R,
	}
	x0 := mat.NewVecDense(2, []float64{1, -1})
	P0 := mat.NewDense(2, 2, []float64{1, 0.2, 0.2, 2})
	e, err := NewEKF(model, x0, P0)
	if err != nil {
		t.Fatal(err)
	}
	u := mat.NewVecDense(1, []float64{0.4})
	z := mat.NewVecDense(1, []float64{0.7})
	if err := e.Step(u, z); err != nil {
		t.Fatal(err)
	}

	var xp mat.VecDense
	xp.MulVec(A, x0)
	xp.AddScaledVec(&xp, 0.4, mat.NewVecDense(2, []float64{1, 0.5}))
	var ap, pp mat.Dense
	ap.Mul(A, P0)
	pp.Mul(&ap, A.T())
	pp.Add(&pp, Q)
	var cp, s mat.Dense
	cp.Mul(C, &pp)
	s.Mul(&cp, C.T())
	s.Add(&s, R)
	var pct, k mat.Dense
	pct.Mul(&pp, C.T())
	k.Scale(1/s.At(0, 0), &pct)
	innov := 0.7 - (C.At(0, 0)*xp.AtVec(0) + C.At(0, 1)*xp.AtVec(1))
	got := e.State()
	for i := range 2 {
		want := xp.AtVec(i) + k.At(i, 0)*innov
		if math.Abs(got.AtVec(i)-want) > 1e-12 {
			t.Errorf("x[%d] = %v, want %v", i, got.AtVec(i), want)
		}
		if buf.AtVec(i) != xp.AtVec(i) {
			t.Errorf("Update mutated F's returned vector: buf[%d] = %v, want %v", i, buf.AtVec(i), xp.AtVec(i))
		}
	}
	got.SetVec(0, 42)
	if e.State().AtVec(0) == 42 {
		t.Error("State aliases filter state")
	}

	if err := e.Update(nil); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("Update(nil): err = %v", err)
	}
	if err := e.SetState(mat.NewVecDense(3, nil)); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("SetState wrong length: err = %v", err)
	}
	if err := e.SetState(mat.NewVecDense(2, []float64{math.NaN(), 0})); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("SetState NaN: err = %v", err)
	}
	if err := e.SetStateCovariance(mat.NewDense(1, 1, []float64{1})); err == nil {
		t.Error("SetStateCovariance wrong size: nil error")
	}
	if err := e.SetStateCovariance(mat.NewDense(2, 2, []float64{2, 0, 0, 3})); err != nil {
		t.Fatal(err)
	}
	if v := e.StateCovariance().At(1, 1); v != 3 {
		t.Errorf("StateCovariance after Set = %v, want 3", v)
	}
	if _, err := NewEKF(nil, x0, P0); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("NewEKF(nil model): err = %v", err)
	}
}
