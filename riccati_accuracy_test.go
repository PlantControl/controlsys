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

// decoupledCareOracle returns the Care solution of A = [[a, 0], [c, F]],
// B = [0; g], diagonal Q and scalar R by blocks: Z of the (n−1)-state
// Care of (F, g), w of (F − gKᵀ + aI)ᵀw = −Zc and the scalar x₁₁, none of
// which sees the near-singular closed-loop mode a.
func decoupledCareOracle(t *testing.T, A, B, Q *mat.Dense, r float64) *mat.Dense {
	t.Helper()
	n, _ := A.Dims()
	m := n - 1
	F := mat.DenseCopyOf(A.Slice(1, n, 1, n))
	g := mat.DenseCopyOf(B.Slice(1, n, 0, 1))
	c := mat.DenseCopyOf(A.Slice(1, n, 0, 1))
	zr, err := Care(F, g, mat.DenseCopyOf(Q.Slice(1, n, 1, n)), mat.NewDense(1, 1, []float64{r}), nil)
	if err != nil || zr.Accuracy != 0 {
		t.Fatalf("oracle Care: %v", err)
	}
	Z := zr.X
	var gz, fk mat.Dense
	gz.Mul(g.T(), Z)
	fk.Mul(g, &gz)
	fk.Scale(-1/r, &fk)
	fk.Add(F, &fk)
	M := mat.DenseCopyOf(fk.T())
	for i := range m {
		M.Set(i, i, M.At(i, i)+A.At(0, 0))
	}
	var rhs, w mat.Dense
	rhs.Mul(Z, c)
	rhs.Scale(-1, &rhs)
	if err := w.Solve(M, &rhs); err != nil {
		t.Fatal(err)
	}
	cw, gw := 0.0, 0.0
	for i := range m {
		cw += c.At(i, 0) * w.At(i, 0)
		gw += g.At(i, 0) * w.At(i, 0)
	}
	X := mat.NewDense(n, n, nil)
	X.Set(0, 0, (Q.At(0, 0)+2*cw-gw*gw/r)/(-2*A.At(0, 0)))
	for i := range m {
		X.Set(0, i+1, w.At(i, 0))
		X.Set(i+1, 0, w.At(i, 0))
		for j := range m {
			X.Set(i+1, j+1, Z.At(i, j))
		}
	}
	return X
}

// TestCare_LyapunovSingularRescue (BWUCB5): an uncontrollable mode at
// −1e-14..−1e-16 makes the first Newton solve's Lyapunov equation
// singular, so verifyRiccati keeps X on its Riccati residual alone (within
// rescueTol). That is no certificate: "wrong" is off by 0.11 at a residual
// below 1e-6. Such an X is returned with Accuracy 1 (MATLAB icare Report
// 1), never as certified; an accurate one ("right") is still returned, not
// an error.
func TestCare_LyapunovSingularRescue(t *testing.T) {
	cases := []struct {
		name      string
		a, b, q   []float64
		r         float64
		noScaling bool
		wantTol   float64
	}{
		{
			name: "wrong",
			a: []float64{-7.295994576490059e-14, 0, 0, 0,
				0.5373416551501615, 1.2194221167525805, -0.8329644073802775, 0.30297255018420743,
				-1.1406539062241194, 0.6682865141940897, 0.16960245948631908, -0.31201750782666604,
				-1.247658747000422, -0.07038377553086222, 0.7562133980924511, 1.8135932643301729},
			b: []float64{0, 1.5255271270406408, -0.07293550823079031, -0.3862511290607623},
			q: []float64{0.013473239362458653, 0, 0, 0, 0, 0.09943227100362198, 0, 0,
				0, 0, 0.6345392749439289, 0, 0, 0, 0, 0.07249003024896149},
			r: 32.07665464756477,
		},
		{
			name:      "right",
			a:         []float64{-2.9889744621332566e-16, 0, 0.6967271130867092, -1.9887231533259282},
			b:         []float64{0, -1.971948704093606},
			q:         []float64{0.20656946223998846, 0, 0, 0.027864401791373995},
			r:         0.013447720841695682,
			noScaling: true,
			wantTol:   1e-10,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.b)
			A := mat.NewDense(n, n, tc.a)
			B := mat.NewDense(n, 1, tc.b)
			Q := mat.NewDense(n, n, tc.q)
			res, err := Care(A, B, Q, mat.NewDense(1, 1, []float64{tc.r}), &RiccatiOpts{NoScaling: tc.noScaling})
			if err != nil {
				t.Fatal(err)
			}
			Xo := decoupledCareOracle(t, A, B, Q, tc.r)
			var d mat.Dense
			d.Sub(res.X, Xo)
			rel := mat.Norm(&d, 2) / mat.Norm(Xo, 2)
			if res.Accuracy == 0 && !(rel <= 1e-8) {
				t.Errorf("Accuracy 0 with relative error %.3g", rel)
			}
			if tc.wantTol > 0 && !(rel <= tc.wantTol) {
				t.Errorf("relative error %.3g, want ≤ %g", rel, tc.wantTol)
			}
			var bk, acl mat.Dense
			bk.Mul(B, res.K)
			acl.Sub(A, &bk)
			for j := 1; j < n; j++ {
				if acl.At(0, j) != 0 {
					t.Fatalf("A − BK couples the uncontrollable mode: %v", acl.At(0, j))
				}
			}
			var eig mat.Eigen
			if !eig.Factorize(acl.Slice(1, n, 1, n), mat.EigenNone) {
				t.Fatal("Eigen failed")
			}
			for _, v := range eig.Values(nil) {
				if !(real(v) < 0) || math.IsNaN(imag(v)) {
					t.Errorf("closed-loop eigenvalue %v", v)
				}
			}
		})
	}
}
