package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// nearSingularEFold rebuilds the nearSingularEContinuousK draw of seed and
// returns the explicit folds of the rotated model and of its triangular
// twin: one state with |λ| ~ 1e15 and the rest O(1).
func nearSingularEFold(t *testing.T, seed uint64) (rotated, twin *System) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 7704))
	n := 2 + rng.IntN(5)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
	sys, tw := nearSingularEDraw(t, rng, n, m, p, k)
	var err error
	if rotated, err = sys.ToExplicit(); err != nil {
		t.Fatal(err)
	}
	if twin, err = tw.ToExplicit(); err != nil {
		t.Fatal(err)
	}
	return rotated, twin
}

// reducedResponseErr is the worst entrywise |H_red − H| / max|H| of the
// strictly proper parts over pts.
func reducedResponseErr(t *testing.T, sys, red *System, pts []complex128) float64 {
	t.Helper()
	worst := 0.0
	for _, s := range pts {
		g, err := sys.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		r, err := red.EvalFr(s)
		if err != nil {
			t.Fatal(err)
		}
		scale, diff := 0.0, 0.0
		for i := range g {
			for j := range g[i] {
				d := complex(sys.D.At(i, j), 0)
				scale = math.Max(scale, cmplx.Abs(g[i][j]-d))
				diff = math.Max(diff, cmplx.Abs(r[i][j]-g[i][j]))
			}
		}
		if scale > 0 {
			worst = math.Max(worst, diff/scale)
		}
	}
	return worst
}

var slowProbePoints = []complex128{0.3i, 1i, 3i, 10i, 1e3i}

// TestMinimalRealizationStiffFoldKeepsSlowStates covers NZIRSS: the
// ‖A‖-relative staircase floor (~30 at ‖A‖ ~ 5e15) dropped the O(1) slow
// states of these folds (orders 1 instead of 3 and 2) with an O(1)
// response error. The rotated fold of seed 50 is left out: its slow pole
// sits at the rounding level of the fold (0.25 ≈ 0.86·ε·‖A‖_F on arm64,
// below ε·‖A‖_F/2 on amd64), so it keeps or drops it by architecture.
func TestMinimalRealizationStiffFoldKeepsSlowStates(t *testing.T) {
	for _, tc := range []struct {
		seed    uint64
		rotated bool
		want    int
	}{{8, true, 3}, {8, false, 3}, {50, false, 2}} {
		rotated, twin := nearSingularEFold(t, tc.seed)
		ex, name := twin, "twin"
		if tc.rotated {
			ex, name = rotated, "rotated"
		}
		res, err := ex.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if res.Order != tc.want {
			t.Errorf("seed %d %s: order %d, want %d", tc.seed, name, res.Order, tc.want)
		}
		if e := reducedResponseErr(t, ex, res.Sys, slowProbePoints); e > 1e-9 {
			t.Errorf("seed %d %s: response error %g", tc.seed, name, e)
		}
	}
}

// stiffPlantedCase is an upper triangular A with one or two stiff diagonal
// entries -2^k (k in 33..50) and O(1) slow ones, generic B, C, an optional
// uncontrollable last state (zero B row) and an optional unobservable first
// state (zero C column), then a permutation and a power-of-two diagonal
// similarity, all exact.
func stiffPlantedCase(rng *rand.Rand) *System {
	n := 3 + rng.IntN(4)
	m, p := 1+rng.IntN(2), 1+rng.IntN(2)
	A := mat.NewDense(n, n, nil)
	B := mat.NewDense(n, m, nil)
	C := mat.NewDense(p, n, nil)
	nFast := 1 + rng.IntN(2)
	fast := rng.Perm(n)[:nFast]
	for i := range n {
		A.Set(i, i, -0.2-2*rng.Float64())
		for j := i + 1; j < n; j++ {
			A.Set(i, j, rng.NormFloat64())
		}
		for j := range m {
			B.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			C.Set(j, i, rng.NormFloat64())
		}
	}
	for _, i := range fast {
		A.Set(i, i, -math.Ldexp(1+rng.Float64(), 33+rng.IntN(18)))
		for j := range m {
			B.Set(i, j, B.At(i, j)*math.Abs(A.At(i, i)))
		}
	}
	if rng.IntN(2) == 0 {
		for j := range m {
			B.Set(n-1, j, 0)
		}
	}
	if rng.IntN(2) == 0 {
		for j := range p {
			C.Set(j, 0, 0)
		}
	}
	perm := rng.Perm(n)
	exp := make([]int, n)
	for i := range exp {
		exp[i] = rng.IntN(9) - 4
	}
	Ap := mat.NewDense(n, n, nil)
	Bp := mat.NewDense(n, m, nil)
	Cp := mat.NewDense(p, n, nil)
	for i := range n {
		for j := range n {
			Ap.Set(perm[i], perm[j], math.Ldexp(A.At(i, j), exp[j]-exp[i]))
		}
		for j := range m {
			Bp.Set(perm[i], j, math.Ldexp(B.At(i, j), -exp[i]))
		}
		for j := range p {
			Cp.Set(j, perm[i], math.Ldexp(C.At(j, i), exp[i]))
		}
	}
	sys, err := New(Ap, Bp, Cp, mat.NewDense(p, m, nil), 0)
	if err != nil {
		panic(err)
	}
	return sys
}

// exactMinimalOrder is the rank of the Hankel matrix O·K of the stored
// float64 data in exact rational arithmetic.
func exactMinimalOrder(sys *System) int {
	n, m, _ := sys.Dims()
	rat := func(M *mat.Dense) [][]*big.Rat {
		r, c := M.Dims()
		out := make([][]*big.Rat, r)
		for i := range r {
			out[i] = make([]*big.Rat, c)
			for j := range c {
				out[i][j] = new(big.Rat).SetFloat64(M.At(i, j))
			}
		}
		return out
	}
	mul := func(X, Y [][]*big.Rat) [][]*big.Rat {
		out := make([][]*big.Rat, len(X))
		for i := range X {
			out[i] = make([]*big.Rat, len(Y[0]))
			for j := range Y[0] {
				s := new(big.Rat)
				for k := range Y {
					s.Add(s, new(big.Rat).Mul(X[i][k], Y[k][j]))
				}
				out[i][j] = s
			}
		}
		return out
	}
	A, B, C := rat(sys.A), rat(sys.B), rat(sys.C)
	K := make([][]*big.Rat, n)
	for i := range K {
		K[i] = make([]*big.Rat, 0, n*m)
	}
	cur := B
	for range n {
		for i := range n {
			K[i] = append(K[i], cur[i]...)
		}
		cur = mul(A, cur)
	}
	var O [][]*big.Rat
	cur = C
	for range n {
		O = append(O, cur...)
		cur = mul(cur, A)
	}
	H := mul(O, K)
	rank := 0
	rows, cols := len(H), n*m
	for c := 0; c < cols && rank < rows; c++ {
		piv := -1
		for r := rank; r < rows; r++ {
			if H[r][c].Sign() != 0 {
				piv = r
				break
			}
		}
		if piv < 0 {
			continue
		}
		H[rank], H[piv] = H[piv], H[rank]
		for r := rank + 1; r < rows; r++ {
			if H[r][c].Sign() == 0 {
				continue
			}
			f := new(big.Rat).Quo(H[r][c], H[rank][c])
			for k := c; k < cols; k++ {
				H[r][k].Sub(H[r][k], new(big.Rat).Mul(f, H[rank][k]))
			}
		}
		rank++
	}
	return rank
}

// TestMinimalRealizationStiffKnownOrder checks MinimalRealization on exact
// stiff upper triangular models against the exact Hankel rank (NZIRSS): no
// slow state that carries the response may go, and the planted
// uncontrollable and unobservable states must still go. Over-counts keep
// the response: in 5 the staircase keeps every state, in 5 more it keeps
// one noise state within √ε of the response and stands, and in 1 (seed
// 187) the split keeps one as rotation noise just above its block floor.
// Base 19acbb9 under-counted 88 of the 200.
func TestMinimalRealizationStiffKnownOrder(t *testing.T) {
	planted, over := 0, 0
	for seed := range uint64(200) {
		sys := stiffPlantedCase(rand.New(rand.NewPCG(seed, 0x5713ff)))
		n, _, _ := sys.Dims()
		want := exactMinimalOrder(sys)
		res, err := sys.MinimalRealization()
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if want < n {
			planted++
		}
		switch {
		case res.Order < want:
			t.Errorf("seed %d: order %d, want %d (n %d)", seed, res.Order, want, n)
			continue
		case res.Order > want:
			over++
			t.Logf("seed %d: order %d, want %d (n %d)", seed, res.Order, want, n)
		}
		if e := reducedResponseErr(t, sys, res.Sys, slowProbePoints); e > 1e-8 {
			t.Errorf("seed %d: response error %g", seed, e)
		}
	}
	t.Logf("non-minimal %d of 200, over-counted %d", planted, over)
	if planted < 100 || over > 11 {
		t.Errorf("non-minimal %d (want ≥ 100), over-counted %d (want ≤ 11)", planted, over)
	}
}

// TestMinimalRealizationStiffRotatedUncontrollableFastMode checks a dense
// stiff model, a random rotation of an upper triangular one whose fast state
// is undriven: the O(1) slow states, which the ‖A‖-relative floor dropped
// (base 19acbb9 under-counted 12 of 20, most to order 1, with response
// errors up to 2), must stay (NZIRSS). Removing the fast mode perturbs the
// slow response by about ε·‖A‖/|λ_slow| (1e-5..1e-2 here), outside the √ε
// band, so the unreduced model is the answer and its response is exact.
// The reverse case, undriven slow states, is not tested: the rotation
// leaves ε·‖A‖ in the slow block, a real slow response.
func TestMinimalRealizationStiffRotatedUncontrollableFastMode(t *testing.T) {
	for seed := range uint64(20) {
		rng := rand.New(rand.NewPCG(seed, 0xca9c))
		n := 3 + rng.IntN(3)
		T := mat.NewDense(n, n, nil)
		b, c := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
		fast := math.Pow(10, 12+3*rng.Float64())
		for i := range n {
			T.Set(i, i, -0.2-2*rng.Float64())
			for j := i + 1; j < n; j++ {
				T.Set(i, j, rng.NormFloat64())
			}
			if i < n-1 {
				b.Set(i, 0, rng.NormFloat64())
			}
			c.Set(0, i, rng.NormFloat64())
		}
		T.Set(n-1, n-1, -fast)
		Q := randomOrthogonal(rng, n)
		var tmp, A, B, C mat.Dense
		tmp.Mul(Q, T)
		A.Mul(&tmp, Q.T())
		B.Mul(Q, b)
		C.Mul(c, Q.T())
		sys, err := New(&A, &B, &C, mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if res.Order < n-1 {
			t.Errorf("seed %d: order %d, want ≥ %d", seed, res.Order, n-1)
			continue
		}
		if e := reducedResponseErr(t, sys, res.Sys, slowProbePoints); e > 1e-8 {
			t.Errorf("seed %d: response error %g", seed, e)
		}
	}
}

// TestMinimalRealizationRotatedUncontrollableIntegratorReduced checks that a
// rotated uncontrollable integrator beside O(1) modes (the n41 source of
// TestLeastSquaresHighOrderSources) still goes: its computed eigenvalue,
// ~1e-15 against O(1) modes, is a spectral gap, but its componentwise
// eigenvalue bound exceeds its modulus, so it is not resolved from zero
// and the staircase result stands (NZIRSS). A split there kept it.
func TestMinimalRealizationRotatedUncontrollableIntegratorReduced(t *testing.T) {
	src := newLSQModalSource(t, rand.New(rand.NewPCG(7, 7)), 0, 20, 0, 1, 1, 0, true)
	evals, resolved, ok := resolvedFromZero(src.sys.A, 41)
	if !ok {
		t.Fatal("eigenvalues did not converge")
	}
	for k, ev := range evals {
		if cmplx.Abs(ev) < 1e-8 && resolved[k] {
			t.Errorf("rounded integrator %v resolved from zero", ev)
		}
	}
	res, err := src.sys.MinimalRealization()
	if err != nil {
		t.Fatal(err)
	}
	if res.Order != 40 {
		t.Errorf("order %d, want 40", res.Order)
	}
}

// TestMinimalRealizationStiffTwinSlowPoleResolved checks the other side of
// that rule: the slow pole 0.217 of the seed-50 twin fold, below
// ε·‖A‖_F ≈ 0.29, is resolved from zero componentwise (the fold is upper
// triangular), so the split keeps it at order 2 (NZIRSS).
func TestMinimalRealizationStiffTwinSlowPoleResolved(t *testing.T) {
	_, twin := nearSingularEFold(t, 50)
	evals, resolved, ok := resolvedFromZero(twin.A, 2)
	if !ok {
		t.Fatal("eigenvalues did not converge")
	}
	for k, ev := range evals {
		if cmplx.Abs(ev) < 10 && !resolved[k] {
			t.Errorf("slow pole %v not resolved from zero", ev)
		}
	}
	res, err := twin.MinimalRealization()
	if err != nil {
		t.Fatal(err)
	}
	if res.Order != 2 {
		t.Errorf("order %d, want 2", res.Order)
	}
}

// TestGapPossibleIsNecessary checks that gapPossible, the κ_F bound that
// lets Reduce skip the eigenvalues, holds on every model with a magnitude
// gap among the stiff folds, the planted stiff family and random fuzz specs
// (NZIRSS).
func TestGapPossibleIsNecessary(t *testing.T) {
	var models []*System
	for seed := range uint64(400) {
		rng := rand.New(rand.NewPCG(seed, 7704))
		n := 2 + rng.IntN(5)
		m, p := 1, 1
		if rng.IntN(3) == 0 {
			m, p = 1+rng.IntN(2), 1+rng.IntN(2)
		}
		k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
		sys, tw := nearSingularEDraw(t, rng, n, m, p, k)
		for _, s := range []*System{sys, tw} {
			if ex, err := s.ToExplicit(); err == nil {
				models = append(models, ex)
			}
		}
	}
	for seed := range uint64(200) {
		models = append(models, stiffPlantedCase(rand.New(rand.NewPCG(seed, 0x5713ff))))
	}
	rng := rand.New(rand.NewPCG(1, 2))
	for range 3000 {
		kind := uint8(rng.IntN(256))
		var v [fzNVals]float64
		for i := range v {
			v[i] = 3 * rng.NormFloat64()
		}
		var sys *System
		t.Run("spec", func(t *testing.T) {
			defer func() { recover() }()
			sys = fzDecode(t, kind, v).System(t, fzSysOpts{})
		})
		if sys != nil {
			models = append(models, sys)
		}
	}
	gaps := 0
	for i, sys := range models {
		if _, _, ok := magnitudeGap(sys.A); ok {
			gaps++
			if !gapPossible(sys.A) {
				t.Errorf("model %d: magnitude gap but gapPossible false", i)
			}
		}
	}
	if gaps < 300 {
		t.Errorf("only %d models with a gap", gaps)
	}
}
