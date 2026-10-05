package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func blkdiagTest(parts ...*mat.Dense) *mat.Dense {
	n := 0
	for _, p := range parts {
		r, _ := p.Dims()
		n += r
	}
	out := mat.NewDense(n, n, nil)
	at := 0
	for _, p := range parts {
		r, _ := p.Dims()
		setBlock(out, at, at, p)
		at += r
	}
	return out
}

func lqgWeights(Q, N, R *mat.Dense) *mat.Dense {
	n, _ := Q.Dims()
	m, _ := R.Dims()
	W := mat.NewDense(n+m, n+m, nil)
	setBlock(W, 0, 0, Q)
	setBlock(W, 0, n, N)
	setBlock(W, n, 0, mat.DenseCopyOf(N.T()))
	setBlock(W, n, n, R)
	return W
}

func assertLqgNear(t *testing.T, name string, got, want *mat.Dense, rel float64) {
	t.Helper()
	gr, gc := got.Dims()
	wr, wc := want.Dims()
	if gr != wr || gc != wc {
		t.Fatalf("%s dims = (%d,%d), want (%d,%d)", name, gr, gc, wr, wc)
	}
	for i := range gr {
		for j := range gc {
			w := want.At(i, j)
			if math.Abs(got.At(i, j)-w) > rel*math.Max(1, math.Abs(w)) {
				t.Errorf("%s(%d,%d) = %.6g, want %.6g", name, i, j, got.At(i, j), w)
			}
		}
	}
}

func assertSmall(t *testing.T, name string, M *mat.Dense, tol float64) {
	t.Helper()
	if nrm := mat.Norm(M, math.Inf(1)); nrm > tol {
		t.Errorf("%s residual = %g, want <= %g", name, nrm, tol)
	}
}

func eigValues(t *testing.T, A mat.Matrix) []complex128 {
	t.Helper()
	var e mat.Eigen
	if !e.Factorize(A, mat.EigenNone) {
		t.Fatal("eigen factorization failed")
	}
	return e.Values(nil)
}

// lqrResidual checks X and K against the CARE/DARE with cross term N.
func lqrResidual(t *testing.T, name string, discrete bool, A, B, Q, N, R, X, K *mat.Dense) {
	t.Helper()
	var BtX, G, H, res mat.Dense
	BtX.Mul(B.T(), X)
	if discrete {
		G.Mul(&BtX, B)
		G.Add(&G, R)
		H.Mul(&BtX, A)
	} else {
		G.CloneFrom(R)
		H.CloneFrom(&BtX)
	}
	H.Add(&H, N.T())
	var Kw mat.Dense
	if err := Kw.Solve(&G, &H); err != nil {
		t.Fatal(err)
	}
	var d mat.Dense
	d.Sub(K, &Kw)
	assertSmall(t, name+" K", &d, 1e-9)
	var AtX mat.Dense
	AtX.Mul(A.T(), X)
	if discrete {
		res.Mul(&AtX, A)
		res.Sub(&res, X)
	} else {
		res.Add(&AtX, mat.DenseCopyOf(AtX.T()))
	}
	var HtK mat.Dense
	HtK.Mul(H.T(), &Kw)
	res.Sub(&res, &HtK)
	res.Add(&res, Q)
	assertSmall(t, name+" Riccati", &res, 1e-9*math.Max(1, mat.Norm(X, math.Inf(1))))
}

// kalmanResidual checks P and L for x' = Ax + w, y = Cx + v with
// E{ww'} = Qn, E{vv'} = Rn, E{wv'} = Nn.
func kalmanResidual(t *testing.T, name string, discrete bool, A, C, Qn, Nn, Rn, P, L *mat.Dense) {
	t.Helper()
	var At, Ct mat.Dense
	At.CloneFrom(A.T())
	Ct.CloneFrom(C.T())
	lqrResidual(t, name, discrete, &At, &Ct, Qn, Nn, Rn, P, mat.DenseCopyOf(L.T()))
}

// lqgPositiveFeedback closes u = Cc xc + Dy y around the plant and returns
// the closed-loop A and the plant-output map y = Ccl [x; xc].
func lqgPositiveFeedback(t *testing.T, sys *System, Ac, By, Cc, Dy *mat.Dense) (Acl, Ccl *mat.Dense) {
	t.Helper()
	n, m, p := sys.Dims()
	nc, _ := Ac.Dims()
	W := eye(m)
	var DyD mat.Dense
	DyD.Mul(Dy, sys.D)
	W.Sub(W, &DyD)
	var Wi mat.Dense
	if err := Wi.Inverse(W); err != nil {
		t.Fatal(err)
	}
	var uX, uC mat.Dense
	uX.Mul(&Wi, mulDims(m, n, Dy, sys.C))
	uC.Mul(&Wi, Cc)
	var yX, yC mat.Dense
	yX.Mul(sys.D, &uX)
	yX.Add(&yX, sys.C)
	yC.Mul(sys.D, &uC)
	Acl = mat.NewDense(n+nc, n+nc, nil)
	var a11, a12, a21, a22 mat.Dense
	a11.Mul(sys.B, &uX)
	a11.Add(&a11, sys.A)
	a12.Mul(sys.B, &uC)
	a21.Mul(By, &yX)
	a22.Mul(By, &yC)
	a22.Add(&a22, Ac)
	setBlock(Acl, 0, 0, &a11)
	setBlock(Acl, 0, n, &a12)
	setBlock(Acl, n, 0, &a21)
	setBlock(Acl, n, n, &a22)
	Ccl = mat.NewDense(p, n+nc, nil)
	setBlock(Ccl, 0, 0, &yX)
	setBlock(Ccl, 0, n, &yC)
	return Acl, Ccl
}

func separationEigs(t *testing.T, A, B, K, L, C *mat.Dense) []complex128 {
	t.Helper()
	var ABK, ALC mat.Dense
	ABK.Mul(B, K)
	ABK.Sub(A, &ABK)
	ALC.Mul(L, C)
	ALC.Sub(A, &ALC)
	want := eigValues(t, &ABK)
	return append(want, eigValues(t, &ALC)...)
}

func lqgCrossWeights(n, m, p int) (QXU, QWV *mat.Dense) {
	Q := mat.NewDense(n, n, nil)
	for i := range n {
		Q.Set(i, i, 1+0.3*float64(i))
	}
	Q.Set(0, 1, 0.2)
	Q.Set(1, 0, 0.2)
	R := mat.NewDense(m, m, nil)
	for i := range m {
		R.Set(i, i, 1.5+0.5*float64(i))
	}
	N := mat.NewDense(n, m, nil)
	N.Set(0, 0, 0.1)
	N.Set(n-1, m-1, -0.15)
	Qn := mat.NewDense(n, n, nil)
	for i := range n {
		Qn.Set(i, i, 0.8+0.2*float64(i))
	}
	Qn.Set(0, n-1, 0.1)
	Qn.Set(n-1, 0, 0.1)
	Rn := mat.NewDense(p, p, nil)
	for i := range p {
		Rn.Set(i, i, 0.6+0.1*float64(i))
	}
	Nn := mat.NewDense(n, p, nil)
	Nn.Set(1, 0, 0.05)
	Nn.Set(0, p-1, -0.08)
	return lqgWeights(Q, N, R), lqgWeights(Qn, Nn, Rn)
}

func TestLqg_MatlabDocExample(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0, 1, 0, 0, 0, 1, 1, 0, 0}),
		mat.NewDense(3, 2, []float64{0.3, 1, 0, 1, -0.3, 0.9}),
		mat.NewDense(1, 3, []float64{1.9, 1.3, 1}),
		mat.NewDense(1, 2, []float64{0.53, -0.61}), 0)
	if err != nil {
		t.Fatal(err)
	}
	var Qx mat.Dense
	Qx.Scale(0.1, eye(3))
	QXU := blkdiagTest(&Qx, mat.NewDense(2, 2, []float64{1, 0, 0, 2}))
	QWV := blkdiagTest(mat.NewDense(3, 3, []float64{4, 2, 0, 2, 1, 0, 0, 0, 1}), mat.NewDense(1, 1, []float64{0.7}))
	const tol = 1e-3

	reg, err := Lqg(sys, QXU, QWV, nil)
	if err != nil {
		t.Fatal(err)
	}
	assertLqgNear(t, "reg.A", reg.Controller.A, mat.NewDense(3, 3, []float64{
		-6.212, -3.814, -4.136,
		-4.038, -3.196, -1.791,
		-1.418, -1.973, -1.766}), tol)
	assertLqgNear(t, "reg.B", reg.Controller.B, mat.NewDense(3, 1, []float64{2.365, 1.432, 0.7684}), tol)
	assertLqgNear(t, "reg.C", reg.Controller.C, mat.NewDense(2, 3, []float64{
		-0.02904, 0.0008272, 0.0303,
		-0.7147, -0.7115, -0.7132}), tol)
	assertLqgNear(t, "reg.D", reg.Controller.D, mat.NewDense(2, 1, nil), 0)

	servoA := mat.NewDense(4, 4, []float64{
		-7.626, -5.068, -4.891, 0.9018,
		-5.108, -4.146, -2.362, 0.6762,
		-2.121, -2.604, -2.141, 0.4088,
		0, 0, 0, 0})
	servoC := mat.NewDense(2, 4, []float64{
		-0.5388, -0.4173, -0.2481, 0.5578,
		-1.492, -1.388, -1.131, 0.5869})

	one, err := Lqg(sys, QXU, QWV, &LqgOpts{QI: eye(1), OneDOF: true})
	if err != nil {
		t.Fatal(err)
	}
	assertLqgNear(t, "1dof.A", one.Controller.A, servoA, tol)
	assertLqgNear(t, "1dof.B", one.Controller.B, mat.NewDense(4, 1, []float64{-2.365, -1.432, -0.7684, 1}), tol)
	assertLqgNear(t, "1dof.C", one.Controller.C, servoC, tol)
	assertLqgNear(t, "1dof.D", one.Controller.D, mat.NewDense(2, 1, nil), 0)

	two, err := Lqg(sys, QXU, QWV, &LqgOpts{QI: eye(1)})
	if err != nil {
		t.Fatal(err)
	}
	assertLqgNear(t, "2dof.A", two.Controller.A, servoA, tol)
	assertLqgNear(t, "2dof.B", two.Controller.B, mat.NewDense(4, 2, []float64{
		0, 2.365,
		0, 1.432,
		0, 0.7684,
		1, -1}), tol)
	assertLqgNear(t, "2dof.C", two.Controller.C, servoC, tol)
	assertLqgNear(t, "2dof.D", two.Controller.D, mat.NewDense(2, 2, nil), 0)
}

func TestLqg_RegulatorRiccatiOracle(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := obsTestPlant(t, dt, false)
		n, m, p := sys.Dims()
		QXU, QWV := lqgCrossWeights(n, m, p)
		res, err := Lqg(sys, QXU, QWV, nil)
		if err != nil {
			t.Fatal(err)
		}
		disc := dt > 0
		lqrResidual(t, "regulator", disc, sys.A, sys.B,
			subDense(QXU, 0, 0, n, n), subDense(QXU, 0, n, n, m), subDense(QXU, n, n, m, m), res.Xc, res.K)
		kalmanResidual(t, "estimator", disc, sys.A, sys.C,
			subDense(QWV, 0, 0, n, n), subDense(QWV, 0, n, n, p), subDense(QWV, n, n, p, p), res.Xf, res.L)
		for name, get := range map[string]func() (*mat.Dense, bool){"Ki": res.Ki, "Kw": res.Kw, "Mx": res.Mx, "Mw": res.Mw} {
			if g, ok := get(); ok || g != nil {
				t.Errorf("dt=%v: regulator %s() = %v, %v; want nil, false", dt, name, g, ok)
			}
		}

		ctrl := res.Controller
		if ctrl.Dt != dt {
			t.Errorf("controller Dt = %v, want %v", ctrl.Dt, dt)
		}
		Acl, _ := lqgPositiveFeedback(t, sys, ctrl.A, ctrl.B, ctrl.C, ctrl.D)
		want := separationEigs(t, sys.A, sys.B, res.K, res.L, sys.C)
		if got := eigValues(t, Acl); !complexSetsApprox(got, want, 1e-8) {
			t.Errorf("dt=%v: closed-loop eig %v, want %v", dt, got, want)
		}
		for _, e := range want {
			if (disc && cmplxAbs(e) >= 1) || (!disc && real(e) >= 0) {
				t.Errorf("dt=%v: unstable separated eigenvalue %v", dt, e)
			}
		}
	}
}

func cmplxAbs(z complex128) float64 { return math.Hypot(real(z), imag(z)) }

// The pre-MATLAB Lqg modelled process noise at the plant inputs (G = B,
// H = D, Qn m×m). That model is QWV = [B; D] Qw [B; D]' + blkdiag(0, Rn).
func TestLqg_InputNoiseModelMapping(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := obsTestPlant(t, dt, false)
		n, m, p := sys.Dims()
		Qw := mat.NewDense(m, m, []float64{1, 0.2, 0.2, 0.7})
		Rn := mat.NewDense(p, p, []float64{0.5, 0, 0, 0.3})
		BD := mat.NewDense(n+p, m, nil)
		setBlock(BD, 0, 0, sys.B)
		setBlock(BD, n, 0, sys.D)
		var QWV mat.Dense
		QWV.Mul(BD, mulDense(Qw, mat.DenseCopyOf(BD.T())))
		rr := subDense(&QWV, n, n, p, p)
		rr.Add(rr, Rn)
		setBlock(&QWV, n, n, rr)

		kal, err := Kalman(sys, Qw, Rn, nil, nil)
		if err != nil {
			t.Fatal(err)
		}
		res, err := Lqg(sys, eye(n+m), &QWV, nil)
		if err != nil {
			t.Fatal(err)
		}
		assertMatEqual(t, "L", res.L, kal.K, 1e-9)
		assertMatEqual(t, "Xf", res.Xf, kal.X, 1e-9)
	}
}

func lqgServoAugmented(sys *System, QXU, QI *mat.Dense) (Aa, Ba, Qa, Na, R *mat.Dense) {
	n, m, p := sys.Dims()
	h, ii := 1.0, 0.0
	if sys.IsDiscrete() {
		h, ii = sys.Dt, 1
	}
	Aa = mat.NewDense(n+p, n+p, nil)
	Ba = mat.NewDense(n+p, m, nil)
	for i := range n {
		for j := range n {
			Aa.Set(i, j, sys.A.At(i, j))
		}
		for j := range m {
			Ba.Set(i, j, sys.B.At(i, j))
		}
	}
	for i := range p {
		for j := range n {
			Aa.Set(n+i, j, -h*sys.C.At(i, j))
		}
		for j := range m {
			Ba.Set(n+i, j, -h*sys.D.At(i, j))
		}
		Aa.Set(n+i, n+i, ii)
	}
	Qa = blkdiagTest(subDense(QXU, 0, 0, n, n), QI)
	Na = mat.NewDense(n+p, m, nil)
	setBlock(Na, 0, 0, subDense(QXU, 0, n, n, m))
	return Aa, Ba, Qa, Na, subDense(QXU, n, n, m, m)
}

func TestLqg_ServoRiccatiOracle(t *testing.T) {
	QI := mat.NewDense(2, 2, []float64{2, 0.3, 0.3, 1})
	for _, dt := range []float64{0, 0.1} {
		sys := obsTestPlant(t, dt, false)
		n, m, p := sys.Dims()
		QXU, QWV := lqgCrossWeights(n, m, p)
		two, err := Lqg(sys, QXU, QWV, &LqgOpts{QI: QI})
		if err != nil {
			t.Fatal(err)
		}
		Aa, Ba, Qa, Na, R := lqgServoAugmented(sys, QXU, QI)
		Kfull := mat.NewDense(m, n+p, nil)
		setBlock(Kfull, 0, 0, two.K)
		setBlock(Kfull, 0, n, two.ki)
		lqrResidual(t, "servo", dt > 0, Aa, Ba, Qa, Na, R, two.Xc, Kfull)
		kalmanResidual(t, "servo estimator", dt > 0, sys.A, sys.C,
			subDense(QWV, 0, 0, n, n), subDense(QWV, 0, n, n, p), subDense(QWV, n, n, p, p), two.Xf, two.L)

		ctrl := two.Controller
		if _, nin, _ := ctrl.Dims(); nin != 2*p {
			t.Fatalf("2dof inputs = %d, want %d", nin, 2*p)
		}
		By := subDense(ctrl.B, 0, p, n+p, p)
		Dy := subDense(ctrl.D, 0, p, m, p)
		Acl, Ccl := lqgPositiveFeedback(t, sys, ctrl.A, By, ctrl.C, Dy)
		var ABK, ALC mat.Dense
		ABK.Mul(Ba, Kfull)
		ABK.Sub(Aa, &ABK)
		want := eigValues(t, &ABK)
		ALC.Mul(two.L, sys.C)
		ALC.Sub(sys.A, &ALC)
		want = append(want, eigValues(t, &ALC)...)
		if got := eigValues(t, Acl); !complexSetsApprox(got, want, 1e-8) {
			t.Errorf("dt=%v: servo closed-loop eig %v, want %v", dt, got, want)
		}

		Br := subDense(ctrl.B, 0, 0, n+p, p)
		h := 1.0
		if dt > 0 {
			h = dt
		}
		wantBr := mat.NewDense(n+p, p, nil)
		for i := range p {
			wantBr.Set(n+i, i, h)
		}
		assertMatEqual(t, "2dof B_r", Br, wantBr, 0)

		one, err := Lqg(sys, QXU, QWV, &LqgOpts{QI: QI, OneDOF: true})
		if err != nil {
			t.Fatal(err)
		}
		var negBy, negDy mat.Dense
		negBy.Scale(-1, By)
		negDy.Scale(-1, Dy)
		assertMatEqual(t, "1dof A", one.Controller.A, ctrl.A, 1e-12)
		assertMatEqual(t, "1dof B", one.Controller.B, &negBy, 1e-12)
		assertMatEqual(t, "1dof C", one.Controller.C, ctrl.C, 1e-12)
		assertMatEqual(t, "1dof D", one.Controller.D, &negDy, 1e-12)

		Bcl := mat.NewDense(2*n+p, p, nil)
		setBlock(Bcl, n, 0, Br)
		var M mat.Dense
		if dt > 0 {
			M.Sub(eye(2*n+p), Acl)
		} else {
			M.Scale(-1, Acl)
		}
		var X, dc mat.Dense
		if err := X.Solve(&M, Bcl); err != nil {
			t.Fatal(err)
		}
		dc.Mul(Ccl, &X)
		assertMatEqual(t, "servo DC gain r->y", &dc, eye(p), 1e-9)
	}
}

// lqgStepCheck verifies one controller step against the MATLAB lqg
// equations: u = -Kx x̂ - Ki xi - F inn with inn = y - C xp - D u, F = 0
// for the delayed estimator, x̂ = xp + Mx inn, and
// xp+ = A xp + B u + L inn, xi+ = xi + Ts(r - y).
func lqgStepCheck(t *testing.T, sys *System, res *LqgResult, current bool) {
	t.Helper()
	n, m, p := sys.Dims()
	q := 0
	if res.ki != nil {
		q = p
	}
	xp := mat.NewVecDense(n, []float64{0.3, -0.7, 1.1})
	xi := mat.NewVecDense(p, []float64{0.4, -0.2})
	r := mat.NewVecDense(p, []float64{-0.5, 0.9})
	y := mat.NewVecDense(p, []float64{1.2, 0.25})
	xc := mat.NewVecDense(n+q, nil)
	in := mat.NewVecDense(p, nil)
	for i := range n {
		xc.SetVec(i, xp.AtVec(i))
	}
	if q > 0 {
		for i := range p {
			xc.SetVec(n+i, xi.AtVec(i))
		}
		in = mat.NewVecDense(2*p, nil)
		for i := range p {
			in.SetVec(i, r.AtVec(i))
			in.SetVec(p+i, y.AtVec(i))
		}
	} else {
		in.CopyVec(y)
	}
	ctrl := res.Controller
	mv := func(A mat.Matrix, x mat.Vector) *mat.VecDense {
		r, _ := A.Dims()
		v := mat.NewVecDense(r, nil)
		v.MulVec(A, x)
		return v
	}
	u := mv(ctrl.C, xc)
	u.AddVec(u, mv(ctrl.D, in))

	inn := mv(sys.C, xp)
	inn.AddVec(inn, mv(sys.D, u))
	inn.SubVec(y, inn)

	xhat := mat.VecDenseCopyOf(xp)
	if current {
		xhat.AddVec(xhat, mv(res.mx, inn))
	}
	wantU := mv(res.K, xhat)
	if current {
		wantU.AddVec(wantU, mv(res.kw, mv(res.mw, inn)))
	}
	if q > 0 {
		wantU.AddVec(wantU, mv(res.ki, xi))
	}
	wantU.ScaleVec(-1, wantU)
	for i := range m {
		if math.Abs(u.AtVec(i)-wantU.AtVec(i)) > 1e-10 {
			t.Errorf("u[%d] = %.12g, want %.12g", i, u.AtVec(i), wantU.AtVec(i))
		}
	}

	next := mv(ctrl.A, xc)
	next.AddVec(next, mv(ctrl.B, in))
	wantXp := mv(sys.A, xp)
	wantXp.AddVec(wantXp, mv(sys.B, u))
	wantXp.AddVec(wantXp, mv(res.L, inn))
	for i := range n {
		if math.Abs(next.AtVec(i)-wantXp.AtVec(i)) > 1e-10 {
			t.Errorf("xp+[%d] = %.12g, want %.12g", i, next.AtVec(i), wantXp.AtVec(i))
		}
	}
	for i := range q {
		want := xi.AtVec(i) + sys.Dt*(r.AtVec(i)-y.AtVec(i))
		if math.Abs(next.AtVec(n+i)-want) > 1e-12 {
			t.Errorf("xi+[%d] = %.12g, want %.12g", i, next.AtVec(n+i), want)
		}
	}
}

func TestLqg_DiscreteEstimatorForms(t *testing.T) {
	sys := obsTestPlant(t, 0.1, false)
	n, m, p := sys.Dims()
	QXU, QWV := lqgCrossWeights(n, m, p)
	Qn := subDense(QWV, 0, 0, n, n)
	Nn := subDense(QWV, 0, n, n, p)
	Rn := subDense(QWV, n, n, p, p)
	R := subDense(QXU, n, n, m, m)
	QI := mat.NewDense(2, 2, []float64{2, 0.3, 0.3, 1})

	for _, tc := range []struct {
		name    string
		opts    *LqgOpts
		current bool
	}{
		{"delayed regulator", nil, false},
		{"delayed servo", &LqgOpts{QI: QI}, false},
		{"current regulator", &LqgOpts{Current: true}, true},
		{"current servo", &LqgOpts{QI: QI, Current: true}, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			res, err := Lqg(sys, QXU, QWV, tc.opts)
			if err != nil {
				t.Fatal(err)
			}
			kalmanResidual(t, "estimator", true, sys.A, sys.C, Qn, Nn, Rn, res.Xf, res.L)
			if _, ok := res.Ki(); ok != (tc.opts != nil && tc.opts.QI != nil) {
				t.Errorf("Ki() ok = %v, want %v", ok, !ok)
			}
			for name, get := range map[string]func() (*mat.Dense, bool){"Kw": res.Kw, "Mx": res.Mx, "Mw": res.Mw} {
				if g, ok := get(); ok != tc.current || ok != (g != nil) {
					t.Errorf("%s() ok = %v with value %v, want ok = %v", name, ok, g != nil, tc.current)
				}
			}
			if !tc.current {
				assertMatEqual(t, "D", res.Controller.D, mat.NewDense(m, p*(1+btoi(res.ki != nil)), nil), 0)
				lqgStepCheck(t, sys, res, false)
				return
			}

			var PCt, S, Sinv, Mx, Mw mat.Dense
			PCt.Mul(res.Xf, sys.C.T())
			S.Mul(sys.C, &PCt)
			S.Add(&S, Rn)
			if err := Sinv.Inverse(&S); err != nil {
				t.Fatal(err)
			}
			Mx.Mul(&PCt, &Sinv)
			Mw.Mul(Nn, &Sinv)
			assertMatEqual(t, "Mx", res.mx, &Mx, 1e-10)
			assertMatEqual(t, "Mw", res.mw, &Mw, 1e-10)
			var AMx mat.Dense
			AMx.Mul(sys.A, res.mx)
			AMx.Add(&AMx, res.mw)
			assertMatEqual(t, "L = A*Mx + Mw", res.L, &AMx, 1e-10)

			Ba := sys.B
			if res.ki != nil {
				_, Ba, _, _, _ = lqgServoAugmented(sys, QXU, QI)
			}
			var BtX, G, Kw mat.Dense
			BtX.Mul(Ba.T(), res.Xc)
			G.Mul(&BtX, Ba)
			G.Add(&G, R)
			if err := Kw.Solve(&G, subDense(&BtX, 0, 0, m, n)); err != nil {
				t.Fatal(err)
			}
			assertMatEqual(t, "Kw", res.kw, &Kw, 1e-10)
			if mat.Norm(res.Controller.D, 1) == 0 {
				t.Error("current controller must have feedthrough")
			}
			lqgStepCheck(t, sys, res, true)

			ctrl := res.Controller
			nc, _ := ctrl.A.Dims()
			By := subDense(ctrl.B, 0, 0, nc, p)
			Dy := subDense(ctrl.D, 0, 0, m, p)
			if res.ki != nil {
				By = subDense(ctrl.B, 0, p, nc, p)
				Dy = subDense(ctrl.D, 0, p, m, p)
			}
			Acl, _ := lqgPositiveFeedback(t, sys, ctrl.A, By, ctrl.C, Dy)
			Aa, Ba2, K := sys.A, sys.B, res.K
			if res.ki != nil {
				Aa, Ba2, _, _, _ = lqgServoAugmented(sys, QXU, QI)
				K = mat.NewDense(m, n+p, nil)
				setBlock(K, 0, 0, res.K)
				setBlock(K, 0, n, res.ki)
			}
			var ABK, ALC mat.Dense
			ABK.Mul(Ba2, K)
			ABK.Sub(Aa, &ABK)
			ALC.Mul(res.L, sys.C)
			ALC.Sub(sys.A, &ALC)
			want := append(eigValues(t, &ABK), eigValues(t, &ALC)...)
			if got := eigValues(t, Acl); !complexSetsApprox(got, want, 1e-8) {
				t.Errorf("closed-loop eig %v, want %v", got, want)
			}
		})
	}
}

func btoi(b bool) int {
	if b {
		return 1
	}
	return 0
}

func TestLqg_Errors(t *testing.T) {
	csys := obsTestPlant(t, 0, false)
	dsys := obsTestPlant(t, 0.1, false)
	asym := eye(5)
	asym.Set(0, 4, 0.1)
	cases := []struct {
		name     string
		sys      *System
		qxu, qwv *mat.Dense
		opts     *LqgOpts
		want     error
	}{
		{"QXU dims", csys, eye(3), eye(5), nil, ErrDimensionMismatch},
		{"QWV dims", csys, eye(5), eye(3), nil, ErrDimensionMismatch},
		{"nil QWV", csys, eye(5), nil, nil, ErrInvalidArgument},
		{"QI dims", csys, eye(5), eye(5), &LqgOpts{QI: eye(3)}, ErrDimensionMismatch},
		{"QXU asymmetric", csys, asym, eye(5), nil, ErrNotSymmetric},
		{"QWV asymmetric", csys, eye(5), asym, nil, ErrNotSymmetric},
		{"OneDOF without QI", csys, eye(5), eye(5), &LqgOpts{OneDOF: true}, ErrInvalidArgument},
		{"Current continuous", csys, eye(5), eye(5), &LqgOpts{Current: true}, ErrWrongDomain},
		{"Current ok", dsys, eye(5), eye(5), &LqgOpts{Current: true}, nil},
	}
	for _, tc := range cases {
		_, err := Lqg(tc.sys, tc.qxu, tc.qwv, tc.opts)
		if !errors.Is(err, tc.want) {
			t.Errorf("%s: err = %v, want %v", tc.name, err, tc.want)
		}
	}

	gain, _ := NewGain(mat.NewDense(1, 1, []float64{1}), 0)
	if _, err := Lqg(gain, eye(1), eye(1), nil); !errors.Is(err, ErrDimensionMismatch) {
		t.Errorf("no states: err = %v, want ErrDimensionMismatch", err)
	}

	delayed := obsTestPlant(t, 0, false)
	if err := delayed.SetInputDelay([]float64{0.2, 0}); err != nil {
		t.Fatal(err)
	}
	if _, err := Lqg(delayed, eye(5), eye(5), nil); !errors.Is(err, ErrDelayUnsupported) {
		t.Errorf("delay: err = %v, want ErrDelayUnsupported", err)
	}
}

func TestLqg_SignalNames(t *testing.T) {
	sys := obsTestPlant(t, 0, false)
	sys.InputName = []string{"u1", "u2"}
	sys.OutputName = []string{"y1", "y2"}
	reg, err := Lqg(sys, eye(5), eye(5), nil)
	if err != nil {
		t.Fatal(err)
	}
	if got := reg.Controller.InputName; len(got) != 2 || got[0] != "y1" || got[1] != "y2" {
		t.Errorf("regulator inputs = %v", got)
	}
	if got := reg.Controller.OutputName; len(got) != 2 || got[0] != "u1" || got[1] != "u2" {
		t.Errorf("regulator outputs = %v", got)
	}
	servo, err := Lqg(sys, eye(5), eye(5), &LqgOpts{QI: eye(2)})
	if err != nil {
		t.Fatal(err)
	}
	if got := servo.Controller.InputName; len(got) != 4 || got[2] != "y1" || got[3] != "y2" {
		t.Errorf("servo inputs = %v", got)
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

func TestLqg_DescriptorMatchesExplicitTwin(t *testing.T) {
	QXU := mat.NewDense(5, 5, []float64{
		2, 0.3, -0.1, 0.05, 0,
		0.3, 1.5, 0.2, 0, 0.04,
		-0.1, 0.2, 1.2, 0.02, -0.03,
		0.05, 0, 0.02, 1.1, 0.1,
		0, 0.04, -0.03, 0.1, 0.9,
	})
	QWV := mat.NewDense(5, 5, []float64{
		1.2, 0.2, -0.1, 0.05, 0.02,
		0.2, 0.9, 0.1, -0.03, 0.04,
		-0.1, 0.1, 1.1, 0.01, -0.02,
		0.05, -0.03, 0.01, 0.8, -0.1,
		0.02, 0.04, -0.02, -0.1, 1.0,
	})
	QI := mat.NewDense(2, 2, []float64{0.5, 0.1, 0.1, 0.7})
	cases := []struct {
		name string
		dt   float64
		opts *LqgOpts
	}{
		{"continuous regulator", 0, nil},
		{"continuous servo", 0, &LqgOpts{QI: QI}},
		{"continuous 1dof", 0, &LqgOpts{QI: QI, OneDOF: true}},
		{"discrete regulator", 0.1, nil},
		{"discrete servo", 0.1, &LqgOpts{QI: QI}},
		{"discrete current", 0.1, &LqgOpts{Current: true}},
		{"discrete current servo", 0.1, &LqgOpts{QI: QI, Current: true}},
	}
	for _, tc := range cases {
		sys := obsTestPlant(t, tc.dt, true)
		E := sys.E
		var Einv mat.Dense
		if err := Einv.Inverse(E); err != nil {
			t.Fatal(err)
		}
		Qn := subDense(QWV, 0, 0, 3, 3)
		var qnBar, nnBar mat.Dense
		qnBar.Mul(&Einv, mulDense(Qn, mat.DenseCopyOf(Einv.T())))
		nnBar.Mul(&Einv, subDense(QWV, 0, 3, 3, 2))
		QWVBar := mat.DenseCopyOf(QWV)
		setBlock(QWVBar, 0, 0, &qnBar)
		setBlock(QWVBar, 0, 3, &nnBar)
		setBlock(QWVBar, 3, 0, mat.DenseCopyOf(nnBar.T()))
		symmetrize(QWVBar.RawMatrix().Data, 5, 5)

		got, err := Lqg(sys, QXU, QWV, tc.opts)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		want, err := Lqg(descriptorTwin(t, sys), QXU, QWVBar, tc.opts)
		if err != nil {
			t.Fatal(err)
		}
		assertLqgNear(t, tc.name+" K", got.K, want.K, 1e-9)
		assertLqgNear(t, tc.name+" Xc", got.Xc, want.Xc, 1e-9)
		assertLqgNear(t, tc.name+" Xf", got.Xf, want.Xf, 1e-9)
		assertLqgNear(t, tc.name+" L = E*Lbar", got.L, mulDense(E, want.L), 1e-9)
		if want.ki != nil {
			assertLqgNear(t, tc.name+" Ki", got.ki, want.ki, 1e-9)
		}
		if want.kw != nil {
			assertLqgNear(t, tc.name+" Mx", got.mx, want.mx, 1e-9)
			assertLqgNear(t, tc.name+" Mw = E*Mwbar", got.mw, mulDense(E, want.mw), 1e-9)
			assertLqgNear(t, tc.name+" Kw*E = Kwbar", mulDense(got.kw, E), want.kw, 1e-9)
		}
		assertSameFrequencyResponse(t, tc.name+" controller", got.Controller, want.Controller, 1e-9)
	}
}
