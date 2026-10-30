package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// polesBitsEqual reports whether a and b hold the same values bit for bit.
func polesBitsEqual(a, b []complex128) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if math.Float64bits(real(a[i])) != math.Float64bits(real(b[i])) || math.Float64bits(imag(a[i])) != math.Float64bits(imag(b[i])) {
			return false
		}
	}
	return true
}

// checkPolesDetail returns PolesDetail of sys after checking that its Poles
// are those of Poles bit for bit and that Accuracy follows the bracket.
func checkPolesDetail(t *testing.T, name string, sys *System) *PolesResult {
	t.Helper()
	want, err := sys.Poles()
	if err != nil {
		t.Fatalf("%s: Poles: %v", name, err)
	}
	res, err := sys.PolesDetail()
	if err != nil {
		t.Fatalf("%s: PolesDetail: %v", name, err)
	}
	if !polesBitsEqual(res.Poles, want) {
		t.Errorf("%s: PolesDetail poles %v, Poles %v", name, res.Poles, want)
	}
	acc := 1
	if res.FiniteLo == res.FiniteHi && res.FiniteHi == len(res.Poles) {
		acc = 0
	}
	if res.Accuracy != acc || res.FiniteLo > res.FiniteHi {
		t.Errorf("%s: Accuracy %d with bracket [%d, %d] and %d poles", name, res.Accuracy, res.FiniteLo, res.FiniteHi, len(res.Poles))
	}
	return res
}

func randDense(rng *rand.Rand, r, c int) *mat.Dense {
	m := mat.NewDense(r, c, nil)
	for i := range r {
		for j := range c {
			m.Set(i, j, rng.NormFloat64())
		}
	}
	return m
}

// TestPolesDetail_ExplicitModelsAreDecided: with E nil or the identity,
// continuous or discrete, with or without internal delays, every
// eigenvalue of A is a pole: FiniteLo = FiniteHi = n, Accuracy 0.
func TestPolesDetail_ExplicitModelsAreDecided(t *testing.T) {
	rng := rand.New(rand.NewPCG(5, 56))
	for run := range 6 {
		n := 1 + run
		A := randDense(rng, n, n)
		for _, dt := range []float64{0, 0.1} {
			sys, err := New(A, randDense(rng, n, 1), randDense(rng, 1, n), mat.NewDense(1, 1, nil), dt)
			if err != nil {
				t.Fatal(err)
			}
			eye := mat.NewDense(n, n, nil)
			for i := range n {
				eye.Set(i, i, 1)
			}
			withI, err := NewDescriptor(A, sys.B, sys.C, sys.D, eye, dt)
			if err != nil {
				t.Fatal(err)
			}
			for _, s := range []*System{sys, withI} {
				res := checkPolesDetail(t, "explicit", s)
				if res.FiniteLo != n || res.FiniteHi != n || res.Accuracy != 0 {
					t.Errorf("n=%d dt=%v identity E=%v: bracket [%d, %d] Accuracy %d, want [%d, %d] 0",
						n, dt, s.E != nil, res.FiniteLo, res.FiniteHi, res.Accuracy, n, n)
				}
			}
		}
	}
	for _, dt := range []float64{0, 0.1} {
		sys, _ := internalDelayZerosFixture(t, dt, 2, mat.NewDense(1, 1, []float64{0.25}))
		res := checkPolesDetail(t, "internal delay", sys)
		if n, _, _ := sys.Dims(); res.FiniteLo != n || res.FiniteHi != n || res.Accuracy != 0 {
			t.Errorf("internal delay dt=%v: bracket [%d, %d] Accuracy %d, want [%d, %d] 0", dt, res.FiniteLo, res.FiniteHi, res.Accuracy, n, n)
		}
	}
	sys, _ := internalDelayZerosFixture(t, 0, 2, mat.NewDense(1, 1, []float64{0.25}))
	sys.E = mat.NewDense(3, 3, []float64{1, 2, 0, 0, 1, 1, 1, 3, 1})
	res := checkPolesDetail(t, "internal delay, singular E", sys)
	if res.FiniteLo != 2 || res.FiniteHi != 2 || res.Accuracy != 0 {
		t.Errorf("internal delay, singular E: bracket [%d, %d] Accuracy %d with %d poles, want [2, 2] 0", res.FiniteLo, res.FiniteHi, res.Accuracy, len(res.Poles))
	}
	gain, err := NewGain(mat.NewDense(1, 1, []float64{2}), 0)
	if err != nil {
		t.Fatal(err)
	}
	if res, err := gain.PolesDetail(); err != nil || res == nil || len(res.Poles) != 0 || res.FiniteLo != 0 || res.FiniteHi != 0 || res.Accuracy != 0 {
		t.Errorf("static gain: %+v, %v; want empty result, Accuracy 0", res, err)
	}
}

// TestPolesDetail_WellPosedDescriptorsAreDecided: descriptors whose finite
// count no rounding-size perturbation can change decide it, and Poles
// returns that many poles: a nonsingular E, and integer pencils
// M·(blkdiag(F, I), blkdiag(W, N))·P with unimodular M, P (exact in
// float64), N nilpotent of index 1, 2 or 3. Poles match eig(W⁻¹F).
func TestPolesDetail_WellPosedDescriptorsAreDecided(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 56))
	for run := range 4 {
		n := 3 + run
		E := randDense(rng, n, n)
		for i := range n {
			E.Set(i, i, E.At(i, i)+3)
		}
		sys, err := NewDescriptor(randDense(rng, n, n), randDense(rng, n, 1), randDense(rng, 1, n), mat.NewDense(1, 1, nil), E, 0)
		if err != nil {
			t.Fatal(err)
		}
		if res := checkPolesDetail(t, "nonsingular E", sys); res.FiniteLo != n || res.FiniteHi != n || res.Accuracy != 0 {
			t.Errorf("nonsingular E n=%d: bracket [%d, %d] Accuracy %d, want [%d, %d] 0", n, res.FiniteLo, res.FiniteHi, res.Accuracy, n, n)
		}
	}
	ints := func(lo, hi int) float64 { return float64(lo + rng.IntN(hi-lo+1)) }
	for run := range 9 {
		nf, k := 1+run%3, 1+run/3
		n := nf + k
		F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
		for i := range nf {
			for j := range nf {
				F.Set(i, j, ints(-4, 4))
				if j > i {
					W.Set(i, j, ints(-2, 2))
				}
			}
			W.Set(i, i, ints(1, 3))
		}
		for i := nf; i < n; i++ {
			F.Set(i, i, 1)
			if i+1 < n {
				W.Set(i, i+1, 1)
			}
			for j := range nf {
				F.Set(j, i, ints(-2, 2))
			}
		}
		unimodular := func() *mat.Dense {
			L, U := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
			for i := range n {
				L.Set(i, i, 1)
				U.Set(i, i, float64(1-2*rng.IntN(2)))
				for j := range i {
					L.Set(i, j, ints(-1, 1))
					U.Set(j, i, ints(-1, 1))
				}
			}
			var m mat.Dense
			m.Mul(L, U)
			return &m
		}
		M, P := unimodular(), unimodular()
		var T, A, E mat.Dense
		T.Mul(M, F)
		A.Mul(&T, P)
		T.Mul(M, W)
		E.Mul(&T, P)
		sys, err := NewDescriptor(&A, randDense(rng, n, 1), randDense(rng, 1, n), mat.NewDense(1, 1, nil), &E, 0)
		if err != nil {
			t.Fatal(err)
		}
		res := checkPolesDetail(t, "integer chain pencil", sys)
		if res.FiniteLo != nf || res.FiniteHi != nf || res.Accuracy != 0 {
			t.Errorf("run %d nf=%d chain %d: bracket [%d, %d] Accuracy %d with %d poles, want [%d, %d] 0",
				run, nf, k, res.FiniteLo, res.FiniteHi, res.Accuracy, len(res.Poles), nf, nf)
			continue
		}
		var winv, wf mat.Dense
		if err := winv.Inverse(W.Slice(0, nf, 0, nf)); err != nil {
			t.Fatal(err)
		}
		wf.Mul(&winv, F.Slice(0, nf, 0, nf))
		var eig mat.Eigen
		if !eig.Factorize(&wf, mat.EigenNone) {
			t.Fatal("eig failed")
		}
		for _, w := range eig.Values(nil) {
			best := math.Inf(1)
			for _, p := range res.Poles {
				best = min(best, cmplx.Abs(p-w))
			}
			if best > 1e-8*(1+cmplx.Abs(w)) {
				t.Errorf("run %d: pole %v of (F, W) missing from %v", run, w, res.Poles)
			}
		}
	}
}

// TestPolesDetail_InvalidInputMatchesPoles: PolesDetail fails exactly
// where Poles fails, with the same sentinel and a nil result.
func TestPolesDetail_InvalidInputMatchesPoles(t *testing.T) {
	nanA, err := NewDescriptor(mat.NewDense(2, 2, []float64{1, 0, 0, 2}), mat.NewDense(2, 1, nil), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil),
		mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	nanA.A.Set(0, 1, math.NaN())
	nanE, err := NewDescriptor(mat.NewDense(2, 2, []float64{1, 0, 0, 2}), mat.NewDense(2, 1, nil), mat.NewDense(1, 2, nil), mat.NewDense(1, 1, nil),
		mat.NewDense(2, 2, []float64{1, 0, 0, 0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	nanE.E.Set(1, 1, math.Inf(1))
	loop, _ := internalDelayZerosFixture(t, 0, 2, mat.NewDense(1, 1, []float64{1}))
	for _, tc := range []struct {
		name string
		sys  *System
		want error
	}{{"NaN A", nanA, ErrInvalidArgument}, {"Inf E", nanE, ErrInvalidArgument}, {"algebraic loop", loop, ErrAlgebraicLoop}} {
		_, perr := tc.sys.Poles()
		res, err := tc.sys.PolesDetail()
		if !errors.Is(perr, tc.want) || !errors.Is(err, tc.want) || res != nil {
			t.Errorf("%s: PolesDetail (%v, %v), Poles error %v; want nil and %v from both", tc.name, res, err, perr, tc.want)
		}
	}
}

// TestPolesDetail_StiffChainLiteralsFlagged: on the stiff-chain literals of
// ELT76F, IP4PC3 and F4OPPD and the in-repo resplit cases, no float64
// method decides the finite count (each has a level value in the band
// τ_E < s ≤ ρ_j), so Accuracy is 1, and the bracket contains the
// structural count of the generator.
func TestPolesDetail_StiffChainLiteralsFlagged(t *testing.T) {
	for _, l := range poleCountCases() {
		sys, err := NewDescriptor(mat.NewDense(l.n, l.n, l.a), mat.NewDense(l.n, 1, nil), mat.NewDense(1, l.n, nil), mat.NewDense(1, 1, nil),
			mat.NewDense(l.n, l.n, l.e), 0)
		if err != nil {
			t.Fatal(err)
		}
		res := checkPolesDetail(t, l.name, sys)
		if res.Accuracy != 1 || res.FiniteLo > l.want || res.FiniteHi < l.want {
			t.Errorf("%s: bracket [%d, %d] Accuracy %d, want Accuracy 1 and a bracket containing %d", l.name, res.FiniteLo, res.FiniteHi, res.Accuracy, l.want)
		}
	}
}

// TestPolesDetail_NormwiseAmbiguousChainsFlagged: Accuracy 1 on the
// stored rvb rows of L4BCTU and X7O5UZ, with the exact bracket.
//   - cpx 2531, cpx 7001, two 11 (L4BCTU F1): a level ≥ 1 value of the
//     exact staircase is at most τ_E, so a pencil within MATLAB's rank
//     tolerance of the data has one stiff pole fewer, and FiniteHi is one
//     below the structural count of the generator. Under the normwise
//     definition of "finite" (L4BCTU decision 1) the count is ambiguous,
//     not wrong. Poles returns 6 values on each.
//   - cpx 17 (X7O5UZ): float64 deflateStiffChain over-counts on amd64
//     only; the exact staircase keeps all 8 level-1 values (the smallest
//     3.6e-12, far above τ_E = 5.0e-16) for FiniteHi = 8 = len(Poles),
//     while FiniteLo = 0: the structural 3 lies in the bracket, but the
//     count is not decided.
func TestPolesDetail_NormwiseAmbiguousChainsFlagged(t *testing.T) {
	want := map[string][3]int{
		"rvb cpx:2531": {2, 3, 6},
		"rvb cpx:7001": {0, 1, 6},
		"rvb two:11":   {0, 1, 6},
		"rvb cpx:17":   {0, 8, 8},
	}
	for _, l := range rvbChainLiterals {
		sys, err := NewDescriptor(mat.NewDense(l.n, l.n, l.a), mat.NewDense(l.n, 1, nil), mat.NewDense(1, l.n, nil), mat.NewDense(1, 1, nil),
			mat.NewDense(l.n, l.n, l.e), 0)
		if err != nil {
			t.Fatal(err)
		}
		res := checkPolesDetail(t, l.name, sys)
		w := want[l.name]
		if res.Accuracy != 1 || res.FiniteLo != w[0] || res.FiniteHi != w[1] || len(res.Poles) != w[2] {
			t.Errorf("%s: bracket [%d, %d] Accuracy %d with %d poles, want [%d, %d] Accuracy 1 with %d (structural %d)",
				l.name, res.FiniteLo, res.FiniteHi, res.Accuracy, len(res.Poles), w[0], w[1], w[2], l.want)
		}
	}
}

// TestPolesDetail_ChainPencilOverCountFlagged: rvbBuild cpx 17 built on
// the running platform (its generator rounds differently on amd64) is
// flagged, Accuracy 1, with the structural count 3 inside the bracket
// (X7O5UZ).
func TestPolesDetail_ChainPencilOverCountFlagged(t *testing.T) {
	c := rvbBuild(t, "cpx", 17)
	res := checkPolesDetail(t, "cpx:17", c.sys)
	if res.Accuracy != 1 || res.FiniteLo > len(c.oracle) || res.FiniteHi < len(c.oracle) {
		t.Errorf("cpx:17: bracket [%d, %d] Accuracy %d with %d poles, want Accuracy 1 and a bracket containing %d",
			res.FiniteLo, res.FiniteHi, res.Accuracy, len(res.Poles), len(c.oracle))
	}
}

func BenchmarkPolesDetail_Index1_N10(b *testing.B) {
	benchPolesDetail(b, benchSingularDescriptor(b, 7, []int{1, 1, 1}))
}

func BenchmarkPolesDetail_Index2_N10(b *testing.B) {
	benchPolesDetail(b, benchSingularDescriptor(b, 6, []int{2, 2}))
}

func benchPolesDetail(b *testing.B, sys *System) {
	b.ReportAllocs()
	for b.Loop() {
		if _, err := sys.PolesDetail(); err != nil {
			b.Fatal(err)
		}
	}
}
