package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"strings"
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

// TestRiccatiAccuracy_LqgH2Syn: Lqg and H2Syn of ch360 and weakBStall,
// whose regulator X no Newton step certifies, return the controller with
// Accuracy 1, as MATLAB lqg and h2syn consume icare's X with a warning,
// not ErrNoStabilizing (MQM2E7). The controller is checked by separation:
// A + B·Cc and A − Bc·C stable, Ac = A + B·Cc − Bc·C. Their cheap-control
// gains (weakBStall ‖K‖ = 8e8) leave the assembled closed loop unstable
// in float64 (weakBStall: an eigenvalue +0.02 where those of A − BK lie
// at −0.56 and below), so H2Norm, and with it H2Syn, fails; H2Syn keeps
// base's ErrNoStabilizing alongside H2Norm's ErrUnstable.
func TestRiccatiAccuracy_LqgH2Syn(t *testing.T) {
	for _, name := range []string{"ch360", "weakBStall"} {
		t.Run(name, func(t *testing.T) { lqgH2SynAccuracy(t, name) })
	}
}

func lqgH2SynAccuracy(t *testing.T, name string) {
	A, B, Q, R, _ := careVerifyCaseNamed(t, name).dense()
	n, _ := A.Dims()
	stable := func(name string, a mat.Matrix) {
		t.Helper()
		var eig mat.Eigen
		if !eig.Factorize(a, mat.EigenNone) {
			t.Fatalf("%s: Eigen failed", name)
		}
		for _, v := range eig.Values(nil) {
			if !(real(v) < 0) {
				t.Fatalf("%s: closed-loop eigenvalue %v", name, v)
			}
		}
	}
	separated := func(name string, ctrl *System, B, C *mat.Dense) {
		t.Helper()
		if !allZeroDense(ctrl.D) {
			t.Fatalf("%s: controller D = %v", name, ctrl.D.RawMatrix().Data)
		}
		var bf, acl, lc, est, want mat.Dense
		bf.Mul(B, ctrl.C)
		acl.Add(A, &bf)
		stable(name+" A + B·Cc", &acl)
		lc.Mul(ctrl.B, C)
		est.Sub(A, &lc)
		stable(name+" A − Bc·C", &est)
		want.Add(&est, &bf)
		if e := relErr(ctrl.A, &want); !(e <= 1e-12) {
			t.Fatalf("%s: Ac ≠ A + B·Cc − Bc·C (rel %g)", name, e)
		}
	}
	sys, err := New(A, B, eyeDense(n), mat.NewDense(n, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	lqg, err := Lqg(sys, blkdiagTest(Q, R), eyeDense(2*n), nil)
	if err != nil {
		t.Fatalf("Lqg: %v", err)
	}
	if lqg.Accuracy != 1 {
		t.Fatalf("Lqg: Accuracy %d, want 1", lqg.Accuracy)
	}
	separated("Lqg", lqg.Controller, B, sys.C)

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
	h2, err := h2Syn(P, n, 1)
	if err != nil {
		t.Fatalf("h2Syn: %v", err)
	}
	if h2.Accuracy != 1 {
		t.Fatalf("h2Syn: Accuracy %d, want 1", h2.Accuracy)
	}
	separated("h2Syn", h2.K, B, mat.DenseCopyOf(Cp.Slice(n+1, 2*n+1, 0, n)))
	if _, err := H2Syn(P, n, 1); !errors.Is(err, ErrUnstable) || !errors.Is(err, ErrNoStabilizing) {
		t.Fatalf("H2Syn: err %v, want ErrUnstable and ErrNoStabilizing", err)
	}
}

// TestRiccati_OtherScalingAccuracy: an Accuracy 1 result is retried in
// the other scaling mode and then state-scaled with B (plainScaling,
// 3CVH7I), without the workspace that holds it (MQM2E7); a certified
// retry replaces it, an Accuracy 1 retry only with a smaller first Newton
// step, and a failed retry never turns it into an error. An
// unstabilizable boundary mode in any mode is ErrNoStabilizing (K72KOR)
// unless another mode certifies X: a first-mode boundary verdict still
// retries, and a certified retry is returned (Dare NoScaling regression
// of the MPWOJM cluster band on a non-normal block).
func TestRiccati_OtherScalingAccuracy(t *testing.T) {
	ws := NewRiccatiWorkspace(1, 1)
	poor := &RiccatiResult{Accuracy: 1, step: 1e-6}
	acc1 := func(step float64) *RiccatiResult { return &RiccatiResult{Accuracy: 1, step: step} }
	certified := &RiccatiResult{}
	type out struct {
		res *RiccatiResult
		err error
	}
	for _, tc := range []struct {
		name          string
		second, third out
		want          int
		calls         int
		err           error
	}{
		{"retry certified", out{certified, nil}, out{}, 1, 2, nil},
		{"retry smaller step", out{acc1(1e-7), nil}, out{acc1(1e-7), nil}, 1, 3, nil},
		{"retry larger step", out{acc1(1e-5), nil}, out{acc1(1e-5), nil}, 0, 3, nil},
		{"retry equal step", out{acc1(1e-6), nil}, out{acc1(1e-6), nil}, 0, 3, nil},
		{"retry Inf step", out{acc1(math.Inf(1)), nil}, out{nil, ErrNoStabilizing}, 0, 3, nil},
		{"retry fails", out{nil, ErrNoStabilizing}, out{nil, ErrSchurFailed}, 0, 3, nil},
		{"plain certified", out{acc1(1e-7), nil}, out{certified, nil}, 2, 3, nil},
		{"plain smaller step", out{nil, ErrNoStabilizing}, out{acc1(1e-7), nil}, 2, 3, nil},
		{"boundary", out{nil, errBoundaryUnstabilizable}, out{acc1(1e-7), nil}, -1, 3, errBoundaryUnstabilizable},
		{"plain boundary", out{acc1(1e-7), nil}, out{nil, errBoundaryUnstabilizable}, -1, 3, errBoundaryUnstabilizable},
		{"boundary then certified", out{nil, errBoundaryUnstabilizable}, out{certified, nil}, 2, 3, nil},
	} {
		t.Run(tc.name, func(t *testing.T) {
			var calls []RiccatiOpts
			solve := func(_, _, _, _ *mat.Dense, opts *RiccatiOpts) (*RiccatiResult, error) {
				calls = append(calls, *opts)
				switch len(calls) {
				case 1:
					return poor, nil
				case 2:
					return tc.second.res, tc.second.err
				}
				return tc.third.res, tc.third.err
			}
			res, err := otherScaling(solve, true, nil, nil, nil, nil, &RiccatiOpts{Workspace: ws})
			if err != tc.err {
				t.Fatalf("err %v, want %v", err, tc.err)
			}
			if len(calls) != tc.calls || calls[0].Workspace != ws ||
				!calls[1].NoScaling || calls[1].plainScaling || calls[1].Workspace != nil ||
				len(calls) > 2 && (calls[2].NoScaling || !calls[2].plainScaling || calls[2].Workspace != nil) {
				t.Fatalf("calls %+v", calls)
			}
			want := []*RiccatiResult{poor, tc.second.res, tc.third.res}
			if tc.want < 0 && res != nil || tc.want >= 0 && res != want[tc.want] {
				t.Errorf("got %+v, want %d", res, tc.want)
			}
		})
	}
	calls := 0
	res, err := otherScaling(func(_, _, _, _ *mat.Dense, _ *RiccatiOpts) (*RiccatiResult, error) {
		calls++
		return certified, nil
	}, true, nil, nil, nil, nil, nil)
	if err != nil || res != certified || calls != 1 {
		t.Errorf("certified first: %v %v, %d calls", res, err, calls)
	}
	for _, tc := range []struct {
		name   string
		second out
		want   *RiccatiResult
		err    error
	}{
		{"boundary first, certified retry", out{certified, nil}, certified, nil},
		{"boundary first, Accuracy 1 retry", out{acc1(1e-9), nil}, nil, errBoundaryUnstabilizable},
		{"boundary first, failed retry", out{nil, ErrNoStabilizing}, nil, errBoundaryUnstabilizable},
	} {
		calls = 0
		res, err := otherScaling(func(_, _, _, _ *mat.Dense, _ *RiccatiOpts) (*RiccatiResult, error) {
			if calls++; calls == 1 {
				return nil, errBoundaryUnstabilizable
			}
			return tc.second.res, tc.second.err
		}, false, nil, nil, nil, nil, nil)
		if res != tc.want || err != tc.err || calls != 2 {
			t.Errorf("%s: %v %v, %d calls", tc.name, res, err, calls)
		}
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
// in one scaling mode is retried in the other and then state-scaled with
// B, keeping the other options (INY4MG: Care "modeOnly" failed by
// default, solved with NoScaling), the last for Care only; other first
// errors are not retried, and with all failing the first error is
// returned.
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
		{"both fail", &RiccatiOpts{S: S}, map[bool]error{false: first, true: ErrNoStabilizing}, []bool{false, true, false}, false, first},
		{"both fail, Dare", &RiccatiOpts{S: S}, map[bool]error{false: first, true: ErrNoStabilizing}, []bool{false, true}, false, first},
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
			res, err := otherScaling(solve, !strings.HasSuffix(tc.name, "Dare"), nil, nil, nil, nil, tc.opts)
			if fmt.Sprint(calls) != fmt.Sprint(tc.want) {
				t.Errorf("NoScaling calls %v, want %v", calls, tc.want)
			}
			if (res != nil) != tc.wantOK || err != tc.err {
				t.Errorf("got %v, %v; want ok %v, err %v", res, err, tc.wantOK, tc.err)
			}
		})
	}
}

// bigRiccatiNewton refines x0 to the Riccati solution of single-input
// (A, B, Q, r) by Newton–Kleinman (continuous) or Hewer (discrete)
// steps in 256-bit arithmetic, each Lyapunov (Stein) equation solved
// exactly by Gaussian elimination on its n(n+1)/2 symmetric unknowns.
// The inputs are exact float64 values, so a closed-loop mode near the
// boundary only scales its equations; started near the stabilizing
// solution it converges quadratically there.
func bigRiccatiNewton(t *testing.T, continuous bool, A, B, Q *mat.Dense, r float64, x0 *mat.Dense) *mat.Dense {
	t.Helper()
	const prec = 256
	n, _ := A.Dims()
	nb := func(v float64) *big.Float { return new(big.Float).SetPrec(prec).SetFloat64(v) }
	zero := func() *big.Float { return new(big.Float).SetPrec(prec) }
	mul := func(a, b *big.Float) *big.Float { return zero().Mul(a, b) }
	add := func(a, b *big.Float) *big.Float { return zero().Add(a, b) }
	grid := func(m *mat.Dense) [][]*big.Float {
		g := make([][]*big.Float, n)
		for i := range n {
			g[i] = make([]*big.Float, n)
			for j := range n {
				g[i][j] = nb(m.At(i, j))
			}
		}
		return g
	}
	a, q, x := grid(A), grid(Q), grid(x0)
	b := make([]*big.Float, n)
	for i := range n {
		b[i] = nb(B.At(i, 0))
	}
	rb := nb(r)
	type idx struct{ i, j int }
	var vars []idx
	pos := map[idx]int{}
	for i := range n {
		for j := i; j < n; j++ {
			pos[idx{i, j}] = len(vars)
			vars = append(vars, idx{i, j})
		}
	}
	v := func(i, j int) int { return pos[idx{min(i, j), max(i, j)}] }
	nv := len(vars)
	tiny := new(big.Float).SetMantExp(big.NewFloat(1), -200)
	for it := range 40 {
		bx := make([]*big.Float, n)
		for j := range n {
			bx[j] = zero()
			for i := range n {
				bx[j] = add(bx[j], mul(b[i], x[i][j]))
			}
		}
		den := rb
		k := make([]*big.Float, n)
		if continuous {
			for j := range n {
				k[j] = zero().Quo(bx[j], den)
			}
		} else {
			den = zero().Set(rb)
			for i := range n {
				den = add(den, mul(bx[i], b[i]))
			}
			for j := range n {
				s := zero()
				for l := range n {
					s = add(s, mul(bx[l], a[l][j]))
				}
				k[j] = zero().Quo(s, den)
			}
		}
		acl := make([][]*big.Float, n)
		for i := range n {
			acl[i] = make([]*big.Float, n)
			for j := range n {
				acl[i][j] = zero().Sub(a[i][j], mul(b[i], k[j]))
			}
		}
		m := make([][]*big.Float, nv)
		for e, p := range vars {
			m[e] = make([]*big.Float, nv+1)
			for c := range nv + 1 {
				m[e][c] = zero()
			}
			i, j := p.i, p.j
			if continuous {
				for l := range n {
					m[e][v(l, j)] = add(m[e][v(l, j)], acl[l][i])
					m[e][v(i, l)] = add(m[e][v(i, l)], acl[l][j])
				}
			} else {
				for l := range n {
					for h := range n {
						m[e][v(l, h)] = add(m[e][v(l, h)], mul(acl[l][i], acl[h][j]))
					}
				}
				m[e][v(i, j)] = zero().Sub(m[e][v(i, j)], nb(1))
			}
			rhs := add(q[i][j], mul(mul(k[i], rb), k[j]))
			m[e][nv] = zero().Neg(rhs)
		}
		for c := range nv {
			p := c
			for r := c + 1; r < nv; r++ {
				if zero().Abs(m[r][c]).Cmp(zero().Abs(m[p][c])) > 0 {
					p = r
				}
			}
			m[c], m[p] = m[p], m[c]
			if m[c][c].Sign() == 0 {
				t.Fatal("bigRiccatiNewton: singular Lyapunov equation")
			}
			for r := c + 1; r < nv; r++ {
				f := zero().Quo(m[r][c], m[c][c])
				for h := c; h <= nv; h++ {
					m[r][h] = zero().Sub(m[r][h], mul(f, m[c][h]))
				}
			}
		}
		sol := make([]*big.Float, nv)
		for r := nv - 1; r >= 0; r-- {
			s := zero().Set(m[r][nv])
			for h := r + 1; h < nv; h++ {
				s = zero().Sub(s, mul(m[r][h], sol[h]))
			}
			sol[r] = zero().Quo(s, m[r][r])
		}
		diff, xn := zero(), zero()
		for i := range n {
			for j := range n {
				nx := sol[v(i, j)]
				d := zero().Sub(nx, x[i][j])
				diff = add(diff, mul(d, d))
				xn = add(xn, mul(nx, nx))
				x[i][j] = nx
			}
		}
		if it > 0 && diff.Cmp(mul(mul(tiny, tiny), xn)) <= 0 {
			X := mat.NewDense(n, n, nil)
			for i := range n {
				for j := range n {
					f, _ := x[i][j].Float64()
					X.Set(i, j, f)
				}
			}
			return X
		}
	}
	t.Fatal("bigRiccatiNewton: no convergence")
	return nil
}

// TestCare_NearZeroModeCertified (FD55OF, sweep trial 15411): an
// uncontrollable mode at −4.9e-14 puts x₁₁ at 1.1e14. Care's Accuracy 0
// X is accurate to rounding by the 256-bit Newton oracle; the 1.06e-5
// "error" seen in wave 18 was its block oracle's, whose x₁₁ = (q₁₁ +
// 2cᵀw − (gᵀw)²/r)/(−2a) cancels terms of 1e6 down to 10, amplifying the
// rounding of w by 1e5.
func TestCare_NearZeroModeCertified(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{-4.8886081920856386e-14, 0, 0, 0,
		-0.2861586449992375, 0.06606670596086822, 0.574760111137941, -0.7235675955669786,
		-2.6179030009302813, 0.6720012532378372, 1.365927429041818, 0.3365883674792016,
		1.880986207556907, -2.3818877413028474, 1.2587313452390165, -0.012608645067475499})
	B := mat.NewDense(4, 1, []float64{0, 0.30354268565747744, -2.012261693667556, -0.31391235859071565})
	Q := mat.NewDiagDense(4, []float64{0.28040849362468895, 0.7244457884686082, 0.01152886251304561, 94.43110253034001})
	const r = 54.32991215802281
	Qd := mat.DenseCopyOf(Q)
	for _, ns := range []bool{false, true} {
		res, err := Care(A, B, Qd, mat.NewDense(1, 1, []float64{r}), &RiccatiOpts{NoScaling: ns})
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		if res.Accuracy != 0 {
			t.Errorf("NoScaling %v: Accuracy %d, want 0", ns, res.Accuracy)
		}
		Xo := bigRiccatiNewton(t, true, A, B, Qd, r, res.X)
		var d mat.Dense
		d.Sub(res.X, Xo)
		if rel := mat.Norm(&d, 2) / mat.Norm(Xo, 2); !(rel <= 1e-12) {
			t.Errorf("NoScaling %v: relative error %.3g", ns, rel)
		}
	}
}

// TestCare_LyapunovSingularRescue (BWUCB5, FD55OF): a closed-loop mode a
// at −1e-14..−1e-17 makes the first Newton solve's Lyapunov equation
// singular (2a within the eps·‖T‖ perturbation of Dtrsyl), so no step
// measures X: verifyRiccati keeps it, on its Riccati residual alone, as
// an Accuracy 1 candidate (step +Inf), never as certified (MATLAB icare
// Report 1). The first two have an uncontrollable mode, the decoupled
// ones a controllable mode of zero cost (exact X = diag(0, Z)). The
// 256-bit Newton oracle shows each X accurate; the 0.11 error once
// reported for "uncontrollable" was that of a block oracle whose x₁₁
// cancels (TestCare_NearZeroModeCertified).
func TestCare_LyapunovSingularRescue(t *testing.T) {
	decoupled := func(a float64) []float64 {
		return []float64{a, 0, 0, 0,
			0, 0.06606670596086822, 0.574760111137941, -0.7235675955669786,
			0, 0.6720012532378372, 1.365927429041818, 0.3365883674792016,
			0, -2.3818877413028474, 1.2587313452390165, -0.012608645067475499}
	}
	dB := []float64{0.7, 0.30354268565747744, -2.012261693667556, -0.31391235859071565}
	dQ := []float64{0, 0, 0, 0, 0, 0.72, 0, 0, 0, 0, 0.0115, 0, 0, 0, 0, 94.4}
	cases := []struct {
		name    string
		a, b, q []float64
		r       float64
	}{
		{
			name: "uncontrollable",
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
			name: "uncontrollable2",
			a:    []float64{-2.9889744621332566e-16, 0, 0.6967271130867092, -1.9887231533259282},
			b:    []float64{0, -1.971948704093606},
			q:    []float64{0.20656946223998846, 0, 0, 0.027864401791373995},
			r:    0.013447720841695682,
		},
		{name: "decoupled1e-14", a: decoupled(-1e-14), b: dB, q: dQ, r: 54.3},
		{name: "decoupled1e-17", a: decoupled(-1e-17), b: dB, q: dQ, r: 54.3},
	}
	for _, tc := range cases {
		for _, ns := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/NoScaling=%v", tc.name, ns), func(t *testing.T) {
				n := len(tc.b)
				A := mat.NewDense(n, n, tc.a)
				B := mat.NewDense(n, 1, tc.b)
				Q := mat.NewDense(n, n, tc.q)
				res, err := Care(A, B, Q, mat.NewDense(1, 1, []float64{tc.r}), &RiccatiOpts{NoScaling: ns})
				if err != nil {
					t.Fatal(err)
				}
				if res.Accuracy != 1 || !math.IsInf(res.step, 1) {
					t.Errorf("Accuracy %d, step %g; want the Lyapunov-singular candidate (1, +Inf)", res.Accuracy, res.step)
				}
				Xo := bigRiccatiNewton(t, true, A, B, Q, tc.r, res.X)
				var d mat.Dense
				d.Sub(res.X, Xo)
				if rel := mat.Norm(&d, 2) / mat.Norm(Xo, 2); !(rel <= 1e-9) {
					t.Errorf("relative error %.3g", rel)
				}
				var bk, acl mat.Dense
				bk.Mul(B, res.K)
				acl.Sub(A, &bk)
				a := tc.a[0]
				for j := 1; j < n; j++ {
					if math.Abs(acl.At(0, j))*math.Abs(acl.At(j, 0)) > 1e-3*math.Abs(a) {
						t.Fatalf("A − BK couples the mode at %g: %v, %v", a, acl.At(0, j), acl.At(j, 0))
					}
				}
				if !(acl.At(0, 0) < 0) {
					t.Errorf("closed-loop mode %v", acl.At(0, 0))
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
}

// TestDare_NearCircleZeroCostMode: a controllable mode of zero cost at
// |λ| = 1 − 5e-13 under a non-normal block (t₁₂ = −1024) stays in the
// closed loop, so the Stein equation of the Hewer step pairs λ with λ̄
// at 1 − |λ|² = 1e-12. The step still resolves it: X = diag(0, Z) is
// certified and accurate by the 256-bit Hewer oracle in both modes.
func TestDare_NearCircleZeroCostMode(t *testing.T) {
	const bb = 1024
	A := mat.NewDense(5, 5, []float64{
		0.5, -bb, 0, 0, 0,
		(0.75 - 1e-12) / bb, 0.5, 0, 0, 0,
		0, 0, 0.5, 0.2, -0.1,
		0, 0, 0.3, 0.9, 0.4,
		0, 0, -0.6, 0.1, 1.1})
	B := mat.NewDense(5, 1, []float64{0.7, -0.3, 0.5, 1.2, -0.8})
	Q := mat.NewDense(5, 5, nil)
	Q.Set(2, 2, 1)
	Q.Set(3, 3, 0.5)
	Q.Set(4, 4, 2)
	const r = 0.8
	for _, ns := range []bool{false, true} {
		res, err := Dare(A, B, Q, mat.NewDense(1, 1, []float64{r}), &RiccatiOpts{NoScaling: ns})
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		if res.Accuracy != 0 {
			t.Errorf("NoScaling %v: Accuracy %d, want 0", ns, res.Accuracy)
		}
		Xo := bigRiccatiNewton(t, false, A, B, Q, r, res.X)
		var d mat.Dense
		d.Sub(res.X, Xo)
		if rel := mat.Norm(&d, 2) / mat.Norm(Xo, 2); !(rel <= 1e-10) {
			t.Errorf("NoScaling %v: relative error %.3g", ns, rel)
		}
		for i := range 2 {
			for j := range 5 {
				if v := Xo.At(i, j); !(math.Abs(v) <= 1e-30) {
					t.Fatalf("oracle X[%d,%d] = %g, want 0", i, j, v)
				}
			}
		}
	}
}

// TestDare_SteinSingularRescue (FD55OF #7, 75DTNG): zero-cost
// uncontrollable modes at 1 − δ and 1 − δ ± iε, the pair a non-normal
// block (t₁₂ ≫ t₂₁), stay in the closed loop. Unscaled, the Stein
// equation of the first Hewer step couples the real mode with the pair
// through a Kronecker block whose unbalanced entries (t₁₂ against
// ε²/t₁₂) tripped Dgetc2's pivot test, so dare kept the pencil X as the
// Lyapunov-singular candidate (Accuracy 1, step +Inf), off by 1e-5 to
// 3e-3 relative in the first three cases and by 99 and 190 in the next
// two (its X₁₁ held only by the residual) against the 256-bit Hewer
// oracle. DLyap now balances that block as it does a diagonal 2×2 block
// and the Stein solve succeeds; Newton then certifies the unscaled X or
// breaks down from it (an unstable iterate, no convergence), and a
// broken-down candidate's first step is no error estimate: 0.5 for an X
// off by 5e8 (256/1e-7), 0.99 off by 2e9 (512/1e-9), 0.003 off by 8e5
// (1e4/1e-6), so it is +Inf, as for the singular candidate. In the last
// four cases the unscaled boundary check flags the non-normal pair, a
// mode 1e-6 inside the circle, as unstabilizable (its cluster band
// inflated by the pair's condition), or found no X: Dare(NoScaling)
// returned ErrNoStabilizing although the scaled retry certifies the
// oracle's X. Both public modes return an X accurate to the oracle.
func TestDare_SteinSingularRescue(t *testing.T) {
	cases := []struct{ t12, delta, eps2 float64 }{
		{64, 1e-10, 1.92e-13},
		{256, 1e-10, 7.68e-12},
		{256, 1e-9, 7.68e-12},
		{64, 1e-8, 6.4e-13},
		{1024, 1e-8, 1.024e-11},
		{256, 1e-7, 1e-12},
		{512, 1e-9, 5e-12},
		{1024, 1e-10, 5e-12},
		{1e4, 1e-6, 1e-12},
		{1e6, 1e-6, 1e-10},
		{3e5, 1e-6, 1e-10},
		{1e4, 1e-10, 1e-12},
		{1e8, 1e-8, 1e-12},
	}
	const r = 0.8
	for _, tc := range cases {
		t.Run(fmt.Sprintf("t12=%g/delta=%g/eps2=%g", tc.t12, tc.delta, tc.eps2), func(t *testing.T) {
			A := mat.NewDense(6, 6, []float64{
				1 - tc.delta, 0, 0, 0, 0, 0,
				0, 1 - tc.delta, tc.t12, 0, 0, 0,
				0, -tc.eps2 / tc.t12, 1 - tc.delta, 0, 0, 0,
				0, 0, 0, 0.5, 0.2, -0.1,
				0, 0, 0, 0.3, 0.9, 0.4,
				0, 0, 0, -0.6, 0.1, 1.1})
			B := mat.NewDense(6, 1, []float64{0, 0, 0, 1.2, -0.8, 0.5})
			Q := mat.NewDense(6, 6, nil)
			Q.Set(3, 3, 1)
			Q.Set(4, 4, 0.5)
			Q.Set(5, 5, 2)
			R := mat.NewDense(1, 1, []float64{r})
			var Xo *mat.Dense
			relErr := func(X *mat.Dense) float64 {
				var d mat.Dense
				d.Sub(X, Xo)
				return mat.Norm(&d, 2) / mat.Norm(Xo, 2)
			}
			for _, ns := range []bool{false, true} {
				res, err := Dare(A, B, Q, R, &RiccatiOpts{NoScaling: ns})
				if err != nil {
					t.Fatalf("NoScaling %v: %v", ns, err)
				}
				if res.Accuracy != 0 {
					t.Errorf("NoScaling %v: Accuracy %d, want 0", ns, res.Accuracy)
				}
				if Xo == nil {
					Xo = bigRiccatiNewton(t, false, A, B, Q, r, res.X)
					for i := range 3 {
						for j := range 6 {
							if v := Xo.At(i, j); !(math.Abs(v) <= 1e-30) {
								t.Fatalf("oracle X[%d,%d] = %g, want 0", i, j, v)
							}
						}
					}
				}
				if rel := relErr(res.X); !(rel <= 1e-9) {
					t.Errorf("NoScaling %v: relative error %.3g", ns, rel)
				}
			}
			cand, err := dare(A, B, Q, R, &RiccatiOpts{NoScaling: true})
			if err != nil {
				if !errors.Is(err, ErrNoStabilizing) {
					t.Errorf("unscaled dare: %v", err)
				}
				return
			}
			rel := relErr(cand.X)
			switch {
			case cand.Accuracy == 0 && !(rel <= 1e-9):
				t.Errorf("unscaled dare: certified X off by %.3g", rel)
			case cand.Accuracy != 0 && !(cand.step >= rel/10):
				t.Errorf("unscaled dare: Accuracy %d, step %g for an X off by %.3g; want a certified X or a step that shows its error", cand.Accuracy, cand.step, rel)
			}
		})
	}
}

// TestKalman_BoundaryUndetectableNoStabilizing (K72KOR): FuzzKalman inputs
// whose (C, A) has an undetectable mode at z = 1 (two poles there with one
// output; four). Base returned Accuracy 1 with A − LC stable in float64
// but, by the exact rational char poly of the returned gain, a root
// outside the unit circle: the first in the default build, the second
// under FMA rounding (-gcflags=all=-d=libfuzzer). No stabilizing solution
// exists (MATLAB idare Report 3), so Kalman returns ErrNoStabilizing.
func TestKalman_BoundaryUndetectableNoStabilizing(t *testing.T) {
	cases := []struct {
		name string
		kind uint8
		v    [fzNVals]float64
	}{
		{"double", 0xa7, [fzNVals]float64{-0.3010299956639812, 45.6, -68, 1, 0, -6, -30, 156, 1.2, 23.7824, 135.4, 48, -10, -2, 47, -1.1111}},
		{"quad", 0xa7, [fzNVals]float64{-69.301, 45.6, -68, 1, 0, -0.6, -30, 192, 1.2, 23.7824, 135.4, 48, -10, -2, 47, -1.1111}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			s := syKalmanSpec(fzDecode(t, tc.kind, tc.v))
			m, p := s.M, s.P
			qn, rn, nn := syNoise(s, m, p)
			var nnD *mat.Dense
			if nn != nil {
				nnD = mqDense(m, p, nn)
			}
			res, err := Kalman(s.System(t, fzSysOpts{}), mqDense(m, m, qn), mqDense(p, p, rn), nnD, nil)
			if !errors.Is(err, ErrNoStabilizing) {
				t.Fatalf("got %v (result %v), want ErrNoStabilizing", err, res != nil)
			}
			// F77544: the unobservable mode of this Jordan pair landed inside
			// the unit circle under FMA contraction (-race on arm64).
			if ok, derr := IsDetectable(mqDense(s.N, s.N, s.A), mqDense(p, s.N, s.C), false); derr != nil || ok {
				t.Errorf("IsDetectable = %v, %v; exact answer false", ok, derr)
			}
		})
	}
}

// TestKalman_WeaklyDetectableUnitPolesNotCertified (K72KOR, FuzzKalman
// 72eefa39e2d17cab, vals[6] = 0: dare_unit_poles_weakly_detectable; and
// its variant vals[6] = 89: dare_unit_poles_weakly_detectable_fma): two
// poles at z = 1 and a pair on the unit circle, detectable (PBH σmin ≈
// 5e-3 against ‖C‖ = √2, 6.5e5 times n·eps·‖[A C]‖, ‖A‖ ≈ 5.7e6 from a
// pole at 48041), so the PBH check rightly passes it: that the task
// repro is Report 3 was wrong. ‖X‖ ≈ 1e10; the task repro's exact closed
// loop was unstable only under FMA rounding (-gcflags=all=-d=libfuzzer),
// the variant's on amd64 too (a root at 1 + 1.9e-5 by Schur–Cohn
// bisection), within the rounding of forming A − LC. Whether MATLAB idare
// reports 1 or 3 here is open; X must never be certified.
func TestKalman_WeaklyDetectableUnitPolesNotCertified(t *testing.T) {
	for _, v6 := range []float64{0, 89} {
		v := [fzNVals]float64{0.26667, -1.5, 0.16667, 14, -35.4, 4, v6, -90, -0.0238, 91, 5.2, -73, 100, -58, 4.5, -170}
		s := syKalmanSpec(fzDecode(t, 0xe9, v))
		m, p := s.M, s.P
		qn, rn, nn := syNoise(s, m, p)
		var nnD *mat.Dense
		if nn != nil {
			nnD = mqDense(m, p, nn)
		}
		for _, ns := range []bool{false, true} {
			res, err := Kalman(s.System(t, fzSysOpts{}), mqDense(m, m, qn), mqDense(p, p, rn), nnD, &RiccatiOpts{NoScaling: ns})
			if err != nil {
				if !errors.Is(err, ErrNoStabilizing) {
					t.Fatalf("v6=%g NoScaling %v: %v", v6, ns, err)
				}
				continue
			}
			if res.Accuracy == 0 {
				t.Fatalf("v6=%g NoScaling %v: Accuracy 0", v6, ns)
			}
		}
	}
}

// TestCare_LightlyDampedUncontrollableModeStable (K72KOR review): under
// cheap control Care ends at Accuracy 1, and (A, B) has an exactly
// uncontrollable mode at −δ ± iω, stable by δ ≫ n·eps·‖A‖. The PBH
// boundary check must not reject it as Report 3 (icare returns X): a band
// of √eps·|λ| (1.5e-6 at ω = 100) once did. X = diag(0, Z) is within 1e-5
// of the 256-bit Newton oracle on arm64 and amd64 for these cases (an
// Accuracy 1 X need not be: at δ = 3e-9, ω = 100 amd64 is off by 0.26);
// A − BK is block lower triangular, so its modes are −δ ± iω and those of
// A₂₂ − B₂K₂.
func TestCare_LightlyDampedUncontrollableModeStable(t *testing.T) {
	cases := []struct{ omega, delta float64 }{{100, 3e-7}, {100, 1e-8}, {1, 3e-9}}
	z := []float64{856429.0639350016, -1065801.0616290835, 746125.9028657239,
		-1065801.0616290835, 5877372.764982905, 1077549.257325759,
		746125.9028657239, 1077549.257325759, 4766586.529900663}
	const r = 3.78006200097873e-15
	for _, tc := range cases {
		for _, ns := range []bool{false, true} {
			t.Run(fmt.Sprintf("omega=%g/delta=%g/NoScaling=%v", tc.omega, tc.delta, ns), func(t *testing.T) {
				A := mat.NewDense(5, 5, []float64{
					-tc.delta, tc.omega, 0, 0, 0,
					-tc.omega, -tc.delta, 0, 0, 0,
					0, 0, 0.27657708515800555, -0.3487414614536205, -0.05759508390159393,
					0, 0, -0.185475044479898, 0.7254440051529344, -0.06988678228186773,
					0, 0, 0.2823003271660112, 0.04502583137978022, 0.09058571535133608})
				B := mat.NewDense(5, 1, []float64{0, 0, -0.4580242863432319, 0.8817080503903288, -0.15544484785480472})
				Q := mat.NewDense(5, 5, nil)
				for i := range 3 {
					for j := range 3 {
						Q.Set(2+i, 2+j, z[3*i+j])
					}
				}
				res, err := Care(A, B, Q, mat.NewDense(1, 1, []float64{r}), &RiccatiOpts{NoScaling: ns})
				if err != nil {
					t.Fatal(err)
				}
				Xo := bigRiccatiNewton(t, true, A, B, Q, r, res.X)
				var d mat.Dense
				d.Sub(res.X, Xo)
				if rel := mat.Norm(&d, 2) / mat.Norm(Xo, 2); !(rel <= 1e-5) {
					t.Errorf("Accuracy %d: relative error %.3g", res.Accuracy, rel)
				}
				var bk, acl mat.Dense
				bk.Mul(B, res.K)
				acl.Sub(A, &bk)
				var eig mat.Eigen
				if !eig.Factorize(acl.Slice(2, 5, 2, 5), mat.EigenNone) {
					t.Fatal("Eigen failed")
				}
				for _, v := range eig.Values(nil) {
					if !(real(v) < -1e-3) {
						t.Errorf("closed-loop eigenvalue %v", v)
					}
				}
			})
		}
	}
}
