package controlsys

import (
	"errors"
	"math"
	"math/big"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// nearSingularTriangularE returns a descriptor model with upper triangular E
// whose last diagonal entry is k·ε, k in [0.2, 2): σ_min(E) ≈ ε·σ_max, at
// MATLAB's rank tolerance n·eps(σ_max), and exact in float64.
func nearSingularTriangularE(seed uint64, dt float64) *System {
	rng := rand.New(rand.NewPCG(seed, 3536))
	n := 2 + rng.IntN(4)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	k := 0.2 + 1.8*rng.Float64()
	E, A := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		E.Set(i, i, 0.5+rng.Float64())
		for j := range n {
			A.Set(i, j, rng.NormFloat64())
			if j > i {
				E.Set(i, j, rng.NormFloat64())
			}
		}
	}
	E.Set(n-1, n-1, k*eps())
	B, C := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil)
	for i := range n {
		for j := range m {
			B.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			C.Set(j, i, rng.NormFloat64())
		}
	}
	sys, err := NewDescriptor(A, B, C, mat.NewDense(p, m, nil), E, dt)
	if err != nil {
		panic(err)
	}
	return sys
}

// nearSingularRotatedE returns U·W·V with orthogonal U, V and W diagonal
// with last entry 1.5·k·ε, k in [0.5, 2] (the near-singular E class of the
// descriptor rank sweeps); rounding of the products perturbs σ_min(E) by
// about itself.
func nearSingularRotatedE(seed uint64, dt float64) *System {
	rng := rand.New(rand.NewPCG(seed, 3535))
	n := 2 + rng.IntN(4)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	k := 0.5 + 1.5*rng.Float64()
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		F.Set(i, i, rng.NormFloat64()-0.3)
		W.Set(i, i, 0.5+rng.Float64())
		for j := i + 1; j < n; j++ {
			F.Set(i, j, rng.NormFloat64())
		}
	}
	W.Set(n-1, n-1, 1.5*k*eps())
	b, c := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil)
	for i := range n {
		for j := range m {
			b.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			c.Set(j, i, rng.NormFloat64())
		}
	}
	orth := func() *mat.Dense {
		g := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				g.Set(i, j, rng.NormFloat64())
			}
		}
		var qr mat.QR
		qr.Factorize(g)
		var q mat.Dense
		qr.QTo(&q)
		return &q
	}
	U, V := orth(), orth()
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	sys, err := NewDescriptor(&A, &B, &C, mat.NewDense(p, m, nil), &E, dt)
	if err != nil {
		panic(err)
	}
	return sys
}

// nearSingularOracle returns the explicit solution Y, the descriptor
// solution X = E⁻ᵀYE⁻¹ and the gain K of the Riccati equation with E, by
// Newton (Kleinman, Hewer) on the explicit model formed in 320 bits from the
// stabilizing gain K0, and whether it converged to a positive definite Y.
func nearSingularOracle(continuous bool, A, B, Q, R, S, E, K0 *mat.Dense) (Y, X, K *mat.Dense, ok bool) {
	e := hbFrom(E)
	a, b := hbSolve(e, hbFrom(A)), hbSolve(e, hbFrom(B))
	q, r, s := hbFrom(Q), hbFrom(R), hbFrom(S)
	k := hbFrom(K0)
	var y, prev hbMat
	conv := false
	for range 40 {
		var rb, rh hbMat
		if continuous {
			y = kleinmanCareHB(a, b, q, r, s, k, 1)
			rb, rh = r, hbMul(b.t(), y)
		} else {
			y = hewerDareHB(a, b, q, r, s, k, 1)
			btx := hbMul(b.t(), y)
			rb, rh = hbAdd(r, hbMul(btx, b), 1), hbMul(btx, a)
		}
		if s.r > 0 {
			rh = hbAdd(rh, s.t(), 1)
		}
		k = hbSolve(rb, rh)
		if prev.r > 0 {
			if d, yn := hbDense(hbAdd(y, prev, -1)), hbDense(y); mat.Norm(d, math.Inf(1)) <= 1e-40*mat.Norm(yn, math.Inf(1)) {
				conv = true
				break
			}
		}
		prev = y
	}
	z := hbSolve(e.t(), y)
	return hbDense(y), hbDense(hbSolve(e.t(), z.t()).t()), hbDense(k), conv && hbPositiveDefinite(y)
}

func hbPositiveDefinite(y hbMat) bool {
	n := y.r
	a := hbAdd(y, newHbMat(n, n), 1)
	p := new(big.Float).SetPrec(hbPrec)
	for k := range n {
		if a.at(k, k).Sign() <= 0 {
			return false
		}
		for i := k + 1; i < n; i++ {
			f := new(big.Float).SetPrec(hbPrec).Quo(a.at(i, k), a.at(k, k))
			for j := k; j < n; j++ {
				a.at(i, j).Sub(a.at(i, j), p.Mul(f, a.at(k, j)))
			}
		}
	}
	return true
}

type nearSingularRiccati struct {
	continuous       bool
	A, B, Q, R, S, E *mat.Dense
	// generalized compares X with the descriptor solution, else with the
	// explicit one E'XE.
	generalized bool
}

// honest reports whether X and K are within max(1e-7, 100·s) of the
// oracle, s the spread of the oracle over three 1-ulp perturbations of A, B
// and E.
func (pr nearSingularRiccati) honest(t *testing.T, X, K *mat.Dense) (eX, eK float64, ok bool) {
	t.Helper()
	pick := func(Y, Xg *mat.Dense) *mat.Dense {
		if pr.generalized {
			return Xg
		}
		return Y
	}
	Y, Xg, Ko, conv := nearSingularOracle(pr.continuous, pr.A, pr.B, pr.Q, pr.R, pr.S, pr.E, K)
	if !conv {
		return math.Inf(1), math.Inf(1), false
	}
	want := pick(Y, Xg)
	eX, eK = relErr(X, want), relErr(K, Ko)
	rng := rand.New(rand.NewPCG(7, 7))
	perturb := func(a *mat.Dense) *mat.Dense {
		o := mat.DenseCopyOf(a)
		raw := o.RawMatrix()
		for i, v := range raw.Data {
			raw.Data[i] = v * (1 + (2*rng.Float64()-1)*eps())
		}
		return o
	}
	sX, sK := 0.0, 0.0
	for range 3 {
		Yp, Xp, Kp, okp := nearSingularOracle(pr.continuous, perturb(pr.A), perturb(pr.B), pr.Q, pr.R, pr.S, perturb(pr.E), Ko)
		if !okp {
			// A 1-ulp change of E moves the problem off any stabilizing
			// solution near K: X is undetermined to working precision.
			return eX, eK, true
		}
		sX, sK = math.Max(sX, relErr(pick(Yp, Xp), want)), math.Max(sK, relErr(Kp, Ko))
	}
	return eX, eK, eX <= math.Max(1e-7, 100*sX) && eK <= math.Max(1e-7, 100*sK)
}

// kalmanDual is the Care/Dare Kalman solves for sys with Qn = I, Rn = I and
// noise input G (G = B with feedthrough D, or G = I without, as in Lqg).
func kalmanDual(sys *System, G, H *mat.Dense) nearSingularRiccati {
	p, _ := sys.C.Dims()
	var qbar mat.Dense
	qbar.Mul(G, G.T())
	R := eyeDense(p)
	var S *mat.Dense
	if H != nil && !allZeroDense(H) {
		S = mulDense(G, mat.DenseCopyOf(H.T()))
		var hh mat.Dense
		hh.Mul(H, H.T())
		R.Add(R, &hh)
	}
	return nearSingularRiccati{sys.IsContinuous(), mat.DenseCopyOf(sys.A.T()), mat.DenseCopyOf(sys.C.T()), &qbar, R, S, mat.DenseCopyOf(sys.E.T()), true}
}

type nearSingularTally struct{ values, errs int }

// checkNearSingularDesigns runs Lqr (Dlqr), Kalman and Lqg with identity
// weights on sys and fails on a result off the oracle or of Accuracy 1, or
// on an error other than ErrDescriptorSingular, ErrNoStabilizing or
// ErrSchurFailed.
func checkNearSingularDesigns(t *testing.T, label string, sys *System, tally map[string]*nearSingularTally) {
	t.Helper()
	n, m, p := sys.Dims()
	cont := sys.IsContinuous()
	Q, R := eyeDense(n), eyeDense(m)
	lqrProblem := nearSingularRiccati{cont, sys.A, sys.B, Q, R, nil, sys.E, false}
	record := func(api string, err error) bool {
		if tally[api] == nil {
			tally[api] = &nearSingularTally{}
		}
		if err != nil {
			tally[api].errs++
			if !errors.Is(err, ErrDescriptorSingular) && !errors.Is(err, ErrNoStabilizing) && !errors.Is(err, ErrSchurFailed) {
				t.Errorf("%s %s: unexpected error %v", label, api, err)
			}
			return false
		}
		tally[api].values++
		return true
	}
	check := func(api string, acc int, pr nearSingularRiccati, X, K *mat.Dense) {
		if acc != 0 {
			t.Errorf("%s %s: Accuracy %d returned for a near-singular E", label, api, acc)
		}
		if eX, eK, ok := pr.honest(t, X, K); !ok {
			t.Errorf("%s %s: Accuracy %d, X off the oracle by %.3g, K by %.3g", label, api, acc, eX, eK)
		}
	}
	var res *RiccatiResult
	var err error
	if cont {
		res, err = Lqr(sys.A, sys.B, Q, R, &RiccatiOpts{E: sys.E})
	} else {
		res, err = Dlqr(sys.A, sys.B, Q, R, &RiccatiOpts{E: sys.E})
	}
	if record("Lqr", err) {
		check("Lqr", res.Accuracy, lqrProblem, res.X, res.K)
	}
	kres, err := Kalman(sys, eyeDense(m), eyeDense(p), nil, nil)
	if record("Kalman", err) {
		check("Kalman", kres.Accuracy, kalmanDual(sys, sys.B, sys.D), kres.X, mat.DenseCopyOf(kres.K.T()))
	}
	lres, err := Lqg(sys, eyeDense(n+m), eyeDense(n+p), nil)
	if record("Lqg", err) {
		check("Lqg Xc", lres.Accuracy, lqrProblem, lres.Xc, lres.K)
		check("Lqg Xf", lres.Accuracy, kalmanDual(sys, eyeDense(n), nil), lres.Xf, mat.DenseCopyOf(lres.L.T()))
	}
}

// nearSingularWindow reports whether E is in the near-singular window the
// Riccati entry points screen: LU condition estimate at or above 1/ε, or
// rank deficient by MATLAB rank.
func nearSingularWindow(t *testing.T, E *mat.Dense) (condNear, fullRank bool) {
	t.Helper()
	var lu mat.LU
	lu.Factorize(E)
	return luNearSingular(&lu), matlabRankFull(t, E)
}

// TestLqrKalmanLqgNearSingularEMatchOracle sweeps Lqr/Dlqr, Kalman and Lqg
// over descriptor models whose E is at MATLAB's rank tolerance, against the
// 320-bit Newton solution of the explicit model. E of full MATLAB rank
// with LU condition estimate at or above 1/ε (rejected before, as MATLAB lqr
// accepts it) must give a value only when it is right; one whose X is
// certified but whose K or E'XE cancels (‖X‖ ~ 1/σ_min(E)) returned L of
// Kalman off by 0.3 with Accuracy 0 when admitted unscreened. Rank-deficient
// E the LU estimate passed returned such Kalman and Lqg gains (off by 1e-3
// to 0.25, Accuracy 0) before. Lqr of the full-rank window must return some
// values (M23EGI, RL6IT7 amendment 1).
func TestLqrKalmanLqgNearSingularEMatchOracle(t *testing.T) {
	tally := map[string]*nearSingularTally{}
	admitted := 0
	for seed := range uint64(1500) {
		for _, dt := range []float64{0, 1} {
			sys := nearSingularTriangularE(seed, dt)
			condNear, full := nearSingularWindow(t, sys.E)
			if condNear != full {
				continue
			}
			before := 0
			if tally["Lqr"] != nil {
				before = tally["Lqr"].values
			}
			checkNearSingularDesigns(t, "triangular", sys, tally)
			if condNear && full && tally["Lqr"].values > before {
				admitted++
			}
		}
	}
	for seed := range uint64(600) {
		for _, dt := range []float64{0, 1} {
			sys := nearSingularRotatedE(seed, dt)
			if condNear, full := nearSingularWindow(t, sys.E); condNear || !full {
				checkNearSingularDesigns(t, "rotated", sys, tally)
			}
		}
	}
	for api, c := range tally {
		t.Logf("%s: %d values, %d errors", api, c.values, c.errs)
	}
	t.Logf("Lqr values on full-rank E with cond·ε ≥ 1: %d", admitted)
	if admitted == 0 {
		t.Error("no Lqr value on a full-rank E with LU condition estimate at or above 1/ε")
	}
}

// TestLqrKalmanLqgFullRankNearSingularE runs the full-rank E with condition
// estimate above 1/ε of the descriptor rank tests through Lqr, Kalman and
// Lqg: each returns a value matching the oracle or ErrDescriptorSingular.
func TestLqrKalmanLqgFullRankNearSingularE(t *testing.T) {
	sys := fullRankNearSingularE(t, 2.46e-15)
	if condNear, full := nearSingularWindow(t, sys.E); !condNear || !full {
		t.Fatalf("model is not in the window: cond·ε ≥ 1 %v, full rank %v", condNear, full)
	}
	tally := map[string]*nearSingularTally{}
	checkNearSingularDesigns(t, "fullRankNearSingularE", sys, tally)
	for api, c := range tally {
		t.Logf("%s: %d values, %d errors", api, c.values, c.errs)
	}
}
