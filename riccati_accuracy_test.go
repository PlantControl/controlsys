package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func careVerifyCaseNamed(t *testing.T, name string) careVerifyCase {
	t.Helper()
	for _, tc := range careVerifyCases {
		if tc.name == name {
			return tc
		}
	}
	t.Fatalf("no case %q", name)
	return careVerifyCase{}
}

// TestRiccatiAccuracy_Propagated: Lqr and Lqe (dual) of weakBStall, whose
// X no Newton step certifies, return it with Accuracy 1 as Care does,
// not ErrNoStabilizing (376552c) or Accuracy 0 (PHWRNR, MATLAB lqr and
// lqe return icare's X).
func TestRiccatiAccuracy_Propagated(t *testing.T) {
	A, B, Q, R, _ := careVerifyCaseNamed(t, "weakBStall").dense()
	n, _ := A.Dims()
	check := func(name string, res *RiccatiResult, err error, A, B *mat.Dense, gainLeft bool) {
		t.Helper()
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if res.Accuracy != 1 {
			t.Fatalf("%s: Accuracy %d, want 1", name, res.Accuracy)
		}
		var bk, acl mat.Dense
		if gainLeft {
			bk.Mul(res.K, B)
		} else {
			bk.Mul(B, res.K)
		}
		acl.Sub(A, &bk)
		var eig mat.Eigen
		if !eig.Factorize(&acl, mat.EigenNone) {
			t.Fatalf("%s: Eigen failed", name)
		}
		for _, v := range eig.Values(nil) {
			if !(real(v) < 0) {
				t.Fatalf("%s: closed-loop eigenvalue %v", name, v)
			}
		}
	}
	res, err := Lqr(A, B, Q, R, nil)
	check("Lqr", res, err, A, B, false)
	At, Bt := mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T())
	res, err = Lqe(At, eyeDense(n), Bt, Q, R, nil, nil)
	check("Lqe", res, err, At, Bt, true)
}

// TestRiccatiAccuracy_FailClosed: Lqg and H2Syn return controllers, which
// cannot report the Accuracy of their Riccati solutions, so an X no Newton
// step certifies (weakBStall) stays ErrNoStabilizing there.
func TestRiccatiAccuracy_FailClosed(t *testing.T) {
	A, B, Q, R, _ := careVerifyCaseNamed(t, "weakBStall").dense()
	n, _ := A.Dims()
	sys, err := New(A, B, eyeDense(n), mat.NewDense(n, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := Lqg(sys, blkdiagTest(Q, R), eyeDense(2*n), nil); !errors.Is(err, ErrNoStabilizing) {
		t.Errorf("Lqg: err %v, want ErrNoStabilizing", err)
	}

	var chol mat.Cholesky
	if !chol.Factorize(mat.NewSymDense(n, mat.DenseCopyOf(Q).RawMatrix().Data)) {
		t.Fatal("Q not positive definite")
	}
	var U mat.TriDense
	chol.UTo(&U)
	Bp := mat.NewDense(n, 2*n+1, nil)
	Cp := mat.NewDense(2*n+1, n, nil)
	Dp := mat.NewDense(2*n+1, 2*n+1, nil)
	for i := range n {
		Bp.Set(i, i, 1)
		Bp.Set(i, 2*n, B.At(i, 0))
		for j := range n {
			Cp.Set(i, j, U.At(i, j))
		}
		Cp.Set(n+1+i, i, 1)
		Dp.Set(n+1+i, n+i, 1)
	}
	Dp.Set(n, 2*n, math.Sqrt(R.At(0, 0)))
	P, err := New(A, Bp, Cp, Dp, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := H2Syn(P, n, 1); !errors.Is(err, ErrNoStabilizing) {
		t.Errorf("H2Syn: err %v, want ErrNoStabilizing", err)
	}
}

// TestRiccatiAccuracy_SingularU11Rejected: an X whose U11 has rcnd below
// eps is not finite (MATLAB icare Report 2) and is no Accuracy 1
// candidate, however small its Newton step; weakBStall's rcnd 8e-15 is.
// Kalman fuzz e5d054af185f91b9 (FMA rounding) returned such an X, rcnd
// 1.5e-17, whose exact closed loop has three roots outside the unit
// circle.
func TestRiccatiAccuracy_SingularU11Rejected(t *testing.T) {
	A, B, Q, R, _ := careVerifyCaseNamed(t, "weakBStall").dense()
	problem, err := newRiccatiProblem(A, B, Q, R, nil)
	if err != nil {
		t.Fatal(err)
	}
	n, m := problem.n, problem.m
	x := make([]float64, n*n)
	K := mat.NewDense(m, n, nil)
	eig := make([]complex128, n)
	for i := range eig {
		eig[i] = -1
	}
	problem.keepPoor(x, K, eig, 1.5e-17, 0.03)
	if problem.poor != nil {
		t.Fatal("rcnd 1.5e-17 kept as an Accuracy 1 candidate")
	}
	problem.keepPoor(x, K, eig, 8e-15, 0.03)
	if problem.poor == nil {
		t.Fatal("rcnd 8e-15 rejected")
	}
}
