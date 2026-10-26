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
