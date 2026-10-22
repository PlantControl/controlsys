package controlsys

import (
	"errors"
	"fmt"
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

// TestRiccati_OtherScaling: a solve that finds no stabilizing solution
// in one scaling mode is retried once in the other, keeping the other
// options (INY4MG: Care "modeOnly" failed by default, solved with
// NoScaling); other errors are not retried, and with both modes failing
// the first error is returned.
func TestRiccati_OtherScaling(t *testing.T) {
	S := mat.NewDense(1, 1, []float64{2})
	first := fmt.Errorf("first: %w", ErrSchurFailed)
	for _, tc := range []struct {
		name   string
		opts   *RiccatiOpts
		errs   map[bool]error
		want   []bool
		wantOK bool
		err    error
	}{
		{"default fails", nil, map[bool]error{false: ErrNoStabilizing}, []bool{false, true}, true, nil},
		{"NoScaling fails", &RiccatiOpts{S: S, NoScaling: true}, map[bool]error{true: first}, []bool{true, false}, true, nil},
		{"both fail", &RiccatiOpts{S: S}, map[bool]error{false: first, true: ErrNoStabilizing}, []bool{false, true}, false, first},
		{"singular R", nil, map[bool]error{false: ErrSingularR}, []bool{false}, false, ErrSingularR},
		{"solves", nil, nil, []bool{false}, true, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls []bool
			solve := func(_, _, _, _ *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
				ns := opts != nil && opts.NoScaling
				calls = append(calls, ns)
				if opts != nil && tc.opts != nil && opts.S != tc.opts.S {
					t.Error("options not kept")
				}
				if err := tc.errs[ns]; err != nil {
					return nil, err
				}
				return &RiccatiResult{}, nil
			}
			res, err := otherScaling(solve, nil, nil, nil, nil, tc.opts)
			if fmt.Sprint(calls) != fmt.Sprint(tc.want) {
				t.Errorf("NoScaling calls %v, want %v", calls, tc.want)
			}
			if (res != nil) != tc.wantOK || err != tc.err {
				t.Errorf("got %v, %v; want ok %v, err %v", res, err, tc.wantOK, tc.err)
			}
		})
	}
}
