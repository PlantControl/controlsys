package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// nearSingularEModel returns sys = (U·F·V, U·b, c·V, D, U·W·V) and its
// diagonal-E twin, W diagonal with W[n-1,n-1] = 1.5·k·ε: the stiff pole
// F[n-1,n-1]/W[n-1,n-1] sits beside MATLAB's rank boundary of E. n = 0 or
// k = 0 draws them as the rv30rNSE generator does (4IBC75, VDATJL).
func nearSingularEModel(t *testing.T, seed uint64, nIn int, kIn float64) (sys, twin *System) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 3030))
	n := 2 + rng.IntN(4)
	if nIn > 0 {
		n = nIn
	}
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	ks := []float64{0.3, 0.8, 1.2, 1.5, 2, 3, 4, 6, 10, 30, 1e3, 1e5}
	k := ks[rng.IntN(len(ks))]
	if kIn > 0 {
		k = kIn
	}
	return nearSingularEDraw(t, rng, n, m, p, k)
}

// nearSingularEContinuousK is nearSingularEModel with n in 2..6 and k
// log-uniform on [0.2, 2.5], so σ_n of E spans MATLAB's rank boundary
// continuously.
func nearSingularEContinuousK(t *testing.T, seed uint64) *System {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 7704))
	n := 2 + rng.IntN(5)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
	sys, _ := nearSingularEDraw(t, rng, n, m, p, k)
	return sys
}

func nearSingularEDraw(t *testing.T, rng *rand.Rand, n, m, p int, k float64) (sys, twin *System) {
	t.Helper()
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	for i := range n {
		F.Set(i, i, rng.NormFloat64()-0.3)
		W.Set(i, i, 0.5+rng.Float64())
		for j := i + 1; j < n; j++ {
			F.Set(i, j, rng.NormFloat64())
		}
	}
	W.Set(n-1, n-1, 1.5*k*eps())
	b, c, D := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil)
	for i := range n {
		for j := range m {
			b.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			c.Set(j, i, rng.NormFloat64())
		}
	}
	for i := range p {
		for j := range m {
			D.Set(i, j, rng.NormFloat64())
		}
	}
	U, V := randomOrthogonal(rng, n), randomOrthogonal(rng, n)
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	var err error
	if sys, err = NewDescriptor(&A, &B, &C, D, &E, 0); err != nil {
		t.Fatal(err)
	}
	if twin, err = NewDescriptor(F, b, c, D, W, 0); err != nil {
		t.Fatal(err)
	}
	return sys, twin
}

func matlabRankFull(t *testing.T, E *mat.Dense) bool {
	t.Helper()
	n, _ := E.Dims()
	var svd mat.SVD
	if !svd.Factorize(E, mat.SVDNone) {
		t.Fatal("SVD of E failed")
	}
	sv := svd.Values(nil)
	tol := float64(n) * (math.Nextafter(sv[0], math.Inf(1)) - sv[0])
	return sv[n-1] > tol
}

func maxPencilRel(t *testing.T, oracle *System, pts []complex128, got func(s complex128) [][]complex128) float64 {
	t.Helper()
	worst := 0.0
	for _, s := range pts {
		want := pencilResponse(t, oracle, s)
		g := got(s)
		for i := range want {
			for j := range want[i] {
				worst = math.Max(worst, cmplx.Abs(g[i][j]-want[i][j])/math.Max(1, cmplx.Abs(want[i][j])))
			}
		}
	}
	if math.IsNaN(worst) {
		return math.Inf(1)
	}
	return worst
}

// checkERankAgreement asserts that ToExplicit, IsProper, Step and
// TransferFunction follow one E-rank rule, MATLAB rank(E), and that the
// transfer function and the Stabsep sum match the pencil of oracle.
func checkERankAgreement(t *testing.T, label string, sys, oracle *System) {
	t.Helper()
	full := matlabRankFull(t, sys.E)
	_, xerr := sys.ToExplicit()
	if full && xerr != nil {
		t.Errorf("%s: ToExplicit of full-rank E: %v", label, xerr)
	}
	if !full && !errors.Is(xerr, ErrDescriptorSingular) {
		t.Errorf("%s: ToExplicit of rank-deficient E: err %v, want ErrDescriptorSingular", label, xerr)
	}
	if pr, err := sys.IsProper(); err != nil || !pr {
		t.Errorf("%s: IsProper = %v, %v; want true", label, pr, err)
	}
	if _, err := Step(sys, 5); err != nil {
		t.Errorf("%s: Step: %v", label, err)
	}
	pts := []complex128{0.05i, 0.37i, complex(0.5, 3), 10i}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Errorf("%s: TransferFunction: %v", label, err)
	} else if rel := maxPencilRel(t, oracle, pts, func(s complex128) [][]complex128 {
		g, err := res.TF.Eval(s)
		if err != nil {
			t.Fatalf("%s: Eval: %v", label, err)
		}
		return g
	}); rel > 1e-6 {
		t.Errorf("%s: TransferFunction off the pencil by %.3g", label, rel)
	}
	sep, err := Stabsep(sys)
	if err != nil {
		t.Errorf("%s: Stabsep: %v", label, err)
		return
	}
	if rel := maxPencilRel(t, oracle, pts[1:3], func(s complex128) [][]complex128 {
		g, h := pencilResponse(t, sep.Stable, s), pencilResponse(t, sep.Unstable, s)
		for i := range g {
			for j := range g[i] {
				g[i][j] += h[i][j]
			}
		}
		return g
	}); rel > 1e-6 {
		t.Errorf("%s: Stable + Unstable off the pencil by %.3g", label, rel)
	}
}

// E = [[1.4, 3.4], [0, h]] has σ₂ above MATLAB's rank tolerance 2·eps(σ₁)
// for h > 2.33e-15 and a 1-norm condition of 1/ε or more for h < 2.59e-15:
// full rank although the LU condition estimate reads singular. Upper
// triangular E keeps both computed exactly enough on every platform.
func fullRankNearSingularE(t *testing.T, h float64) *System {
	t.Helper()
	A := mat.NewDense(2, 2, []float64{-1, 0.3, 0.2, -2})
	B := mat.NewDense(2, 1, []float64{1, -0.7})
	C := mat.NewDense(1, 2, []float64{0.8, 1.3})
	D := mat.NewDense(1, 1, []float64{0.25})
	E := mat.NewDense(2, 2, []float64{1.4, 3.4, 0, h})
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestDescriptorFullRankEWithSingularLUConditionIsExplicit(t *testing.T) {
	sys := fullRankNearSingularE(t, 2.46e-15)
	var lu mat.LU
	lu.Factorize(sys.E)
	if !matlabRankFull(t, sys.E) || !luNearSingular(&lu) {
		t.Fatalf("model is not in the window: full %v, cond·ε %.3g", matlabRankFull(t, sys.E), lu.Cond()*eps())
	}
	if _, err := sys.ToExplicit(); err != nil {
		t.Errorf("ToExplicit: %v", err)
	}
	if pr, err := sys.IsProper(); err != nil || !pr {
		t.Errorf("IsProper = %v, %v; want true", pr, err)
	}
	if pol, err := sys.Poles(); err != nil || len(pol) != 2 {
		t.Errorf("Poles = %v, %v; want 2 finite", pol, err)
	}
	st, err := Step(sys, 5)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	_, cols := st.Y.Dims()
	for k := range cols {
		if v := st.Y.At(0, k); math.IsNaN(v) || math.IsInf(v, 0) {
			t.Fatalf("Step sample %d = %v", k, v)
		}
	}
}

// Stabsep of a full-rank E with cond(E) in [1e4, 1/ε) folded E⁻¹A and lost
// the slow modes: G1 + G2 was off the pencil by about 1 on these.
func TestStabsepIllConditionedEMatchesPencil(t *testing.T) {
	for _, c := range []struct {
		n    int
		k    float64
		seed uint64
	}{{2, 2, 28}, {2, 1.5, 38}, {2, 1, 43}, {3, 1.2, 42}} {
		sys, twin := nearSingularEModel(t, c.seed, c.n, c.k)
		sep, err := Stabsep(sys)
		if err != nil {
			t.Fatalf("n=%d k=%v seed=%d: %v", c.n, c.k, c.seed, err)
		}
		if rel := maxPencilRel(t, twin, []complex128{0.37i, complex(0.5, 3)}, func(s complex128) [][]complex128 {
			g, q := pencilResponse(t, sep.Stable, s), pencilResponse(t, sep.Unstable, s)
			for i := range g {
				for j := range g[i] {
					g[i][j] += q[i][j]
				}
			}
			return g
		}); rel > 1e-6 {
			t.Errorf("n=%d k=%v seed=%d: Stable + Unstable off the pencil by %.3g", c.n, c.k, c.seed, rel)
		}
	}
}

// Seed 1624 (k = 0.8, n = 3) has σ₃ just above MATLAB's rank tolerance and
// cond(E)·ε ≈ 1.8; TransferFunction failed with "slow subsystem: descriptor
// matrix E is singular" (VDATJL).
func TestTransferFunctionNearSingularEModelMatchesPencil(t *testing.T) {
	sys, twin := nearSingularEModel(t, 1624, 0, 0)
	checkERankAgreementTF(t, sys, twin)
}

func checkERankAgreementTF(t *testing.T, sys, twin *System) {
	t.Helper()
	if pr, err := sys.IsProper(); err != nil || !pr {
		t.Errorf("IsProper = %v, %v; want true", pr, err)
	}
	if _, err := Step(sys, 5); err != nil {
		t.Errorf("Step: %v", err)
	}
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatalf("TransferFunction: %v", err)
	}
	if rel := maxPencilRel(t, twin, []complex128{0.05i, 0.37i, complex(0.5, 3), 10i}, func(s complex128) [][]complex128 {
		g, err := res.TF.Eval(s)
		if err != nil {
			t.Fatal(err)
		}
		return g
	}); rel > 1e-8 {
		t.Errorf("TransferFunction off the pencil by %.3g", rel)
	}
}

func TestDescriptorNearSingularEModelsAgreeAcrossAPIs(t *testing.T) {
	seeds := uint64(6)
	if testing.Short() {
		seeds = 2
	}
	for _, k := range []float64{0.5, 0.7, 0.8, 0.9, 1, 1.1, 1.2, 1.5, 2} {
		for n := 2; n <= 5; n++ {
			for seed := range seeds {
				sys, twin := nearSingularEModel(t, seed+40, n, k)
				checkERankAgreement(t, fmt.Sprintf("k=%v n=%d seed=%d", k, n, seed+40), sys, twin)
			}
		}
	}
}

// The generalized Schur form zeroes the E entry of a stiff mode whose σ_n
// lies within ε‖E‖ although MATLAB rank(E) is full; TransferFunction then
// folded E in whole and the stiff pole near 1/ε swamped the slow ones
// (literal 1: off the pencil by 1.38 on both arches, literal 2: 0.957 on
// amd64, denominator of order 1).
func TestTransferFunctionStiffModeDeflatedByQZKeepsSlowPoles(t *testing.T) {
	model := func(a, b, c []float64, d float64, e []float64) *System {
		sys, err := NewDescriptor(mat.NewDense(3, 3, a), mat.NewDense(3, 1, b), mat.NewDense(1, 3, c),
			mat.NewDense(1, 1, []float64{d}), mat.NewDense(3, 3, e), 0)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	cases := map[string]*System{
		"literal 1": model(
			[]float64{0.48183367983483827, 0.02194172359683036, 0.3125942093617087, 0.031175700420809086, -0.08312389461428946, -0.11921583987828199, -0.13878703891954458, 1.2752340032961218, -0.04620546167813694},
			[]float64{-0.8266135256850403, 0.4962938310619698, 0.1997252733191939},
			[]float64{-0.24668887742578766, -0.4891475656619139, 0.7993999478702887}, -0.8501554151268462,
			[]float64{0.075029881031847, -0.1008963391206154, -0.6646984857762108, 0.4791288452228757, 0.33805115635506466, 0.38789676231208653, 0.547940315656329, 0.21367437046148713, -0.3718739090884265}),
		"literal 2": model(
			[]float64{0.5625274460361358, -0.6407286013614327, 0.5026553490392252, 0.3653322516032209, 1.4044694523891432, -0.3174031473250085, 0.15459687343604056, 0.33533431698174915, -0.7915061398096794},
			[]float64{0.09415594089797513, 0.6342443518062528, -0.8776086054621898},
			[]float64{1.4386639092184832, -0.6837301171758893, -0.830597432505679}, 0.14982469859342595,
			[]float64{0.36111507420312494, -0.464622654922211, -0.16025987959200083, 0.11602906673378945, 0.7889788330468326, -0.25621413919277564, 0.47811368921432795, 0.20641790152016704, -0.3914433749837351}),
	}
	// Seeds whose TF was off the pencil by 1e-7..3.9 on arm64 or amd64.
	for _, seed := range []uint64{17063, 18268, 48903, 54949, 57207, 74809, 80013, 117306, 170665, 180034} {
		cases[fmt.Sprintf("seed %d", seed)] = nearSingularEContinuousK(t, seed)
	}
	for name, sys := range cases {
		t.Run(name, func(t *testing.T) {
			checkERankAgreementTF(t, sys, sys)
			pol, err := sys.Poles()
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatal(err)
			}
			if _, m, p := sys.Dims(); m == 1 && p == 1 {
				if got := len(res.TF.Den[0]) - 1; got != len(pol) {
					t.Errorf("denominator order %d, want %d finite poles", got, len(pol))
				}
			}
		})
	}
}
