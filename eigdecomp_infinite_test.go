package controlsys

import (
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// weierstrassDescriptor is the SISO descriptor A = U·blkdiag(H, I_d)·V,
// E = U·blkdiag(J, N)·V, B = 1, C = 1ᵀ, with H, J random r×r and N = 0 or,
// with chain set, the d×d nilpotent shift (one infinite Jordan chain of
// length d). U, V are random orthogonal (orth) or random general. finite
// is the explicit model (J⁻¹H, J⁻¹·B_f, C_f) of its finite modes.
type weierstrassDescriptor struct {
	sys    *System
	finite *System
}

func newWeierstrassDescriptor(seed uint64, r, d int, chain, orth bool) weierstrassDescriptor {
	chains := []int{d}
	if !chain {
		chains = make([]int, d)
		for i := range chains {
			chains[i] = 1
		}
	}
	return newWeierstrassChains(seed, r, chains, orth)
}

// newWeierstrassChains is newWeierstrassDescriptor with N the block
// diagonal of nilpotent shifts of the sizes in chains (infinite Jordan
// chains of those lengths).
func newWeierstrassChains(seed uint64, r int, chains []int, orth bool) weierstrassDescriptor {
	rng := rand.New(rand.NewPCG(seed, 91))
	d := 0
	for _, k := range chains {
		d += k
	}
	n := r + d
	F := mat.NewDense(n, n, nil)
	W := mat.NewDense(n, n, nil)
	H := F.Slice(0, r, 0, r).(*mat.Dense)
	J := W.Slice(0, r, 0, r).(*mat.Dense)
	for i := range r {
		for j := range r {
			J.Set(i, j, rng.NormFloat64())
			H.Set(i, j, rng.NormFloat64())
		}
	}
	i := r
	for _, k := range chains {
		for j := range k {
			F.Set(i+j, i+j, 1)
			if j+1 < k {
				W.Set(i+j, i+j+1, 1)
			}
		}
		i += k
	}
	randMat := func() *mat.Dense {
		M := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				M.Set(i, j, rng.NormFloat64())
			}
		}
		if !orth {
			return M
		}
		var qr mat.QR
		qr.Factorize(M)
		var Q mat.Dense
		qr.QTo(&Q)
		return &Q
	}
	U, V := randMat(), randMat()
	var T, A, E mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B, C := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		B.Set(i, 0, 1)
		C.Set(0, i, 1)
	}
	sys, err := NewDescriptor(&A, B, C, mat.NewDense(1, 1, nil), &E, 0)
	if err != nil {
		panic(err)
	}

	var Ui, Vi mat.Dense
	if err := Ui.Inverse(U); err != nil {
		panic(err)
	}
	if err := Vi.Inverse(V); err != nil {
		panic(err)
	}
	var Bf, Cf, Af, JB mat.Dense
	Bf.Mul(Ui.Slice(0, r, 0, n), B)
	Cf.Mul(C, Vi.Slice(0, n, 0, r))
	var lu mat.LU
	lu.Factorize(J)
	if err := lu.SolveTo(&Af, false, H); err != nil {
		panic(err)
	}
	if err := lu.SolveTo(&JB, false, &Bf); err != nil {
		panic(err)
	}
	finite, err := New(&Af, &JB, &Cf, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		panic(err)
	}
	return weierstrassDescriptor{sys: sys, finite: finite}
}

func BenchmarkStabsep_SingularE_N20(b *testing.B) {
	sys := newWeierstrassDescriptor(1, 16, 4, false, true).sys
	b.ReportAllocs()
	for b.Loop() {
		if _, err := Stabsep(sys); err != nil {
			b.Fatal(err)
		}
	}
}

// assertStabsepMatchesFinite checks Stabsep of w.sys against oracles
// independent of the generalized Schur path: the stable poles against the
// eigenvalues of J⁻¹H, Stable against the real-Schur Stabsep of the
// explicit finite part, and Stable + Unstable against (sE − A)⁻¹ solves,
// the responses within relTol.
func assertStabsepMatchesFinite(t *testing.T, label string, w weierstrassDescriptor, res *StabsepResult, relTol float64) {
	t.Helper()
	var eig mat.Eigen
	if !eig.Factorize(w.finite.A, mat.EigenNone) {
		t.Fatal("eig failed")
	}
	var want []complex128
	for _, v := range eig.Values(nil) {
		if real(v) < 0 {
			want = append(want, v)
		}
	}
	poles, err := res.Stable.Poles()
	if err != nil {
		t.Fatal(err)
	}
	assertEigSetNear(t, label+" stable poles", poles, want, 1e-6)
	oracle, err := Stabsep(w.finite)
	if err != nil {
		t.Fatal(err)
	}
	for _, s := range []complex128{complex(0.3, 0.7), complex(-0.2, 2.5), complex(1.7, -0.4)} {
		g := pencilResponse(t, res.Stable, s)[0][0]
		o := pencilResponse(t, oracle.Stable, s)[0][0]
		if d := cmplx.Abs(g - o); !(d <= relTol*(1+cmplx.Abs(o))) {
			t.Errorf("%s: Stable(%v) = %v, want %v", label, s, g, o)
		}
		sum := g + pencilResponse(t, res.Unstable, s)[0][0]
		full := pencilResponse(t, w.sys, s)[0][0]
		if d := cmplx.Abs(sum - full); !(d <= relTol*(1+cmplx.Abs(full))) {
			t.Errorf("%s: Stable+Unstable(%v) = %v, want %v", label, s, sum, full)
		}
	}
}

// roundingBetaDescriptor returns the 4U3MDS descriptor, E = U·blkdiag(J, 0)·V
// with σ_min(E) = 9.3e-18, whose finite poles are eig(J⁻¹H) =
// {−0.744±1.438i, 2.063±0.851i}.
func roundingBetaDescriptor(t *testing.T) *System {
	t.Helper()
	A := mat.NewDense(5, 5, []float64{
		0.026574501752449863, -1.5203787643976214, 0.501363382935057, 1.002190557270059, -0.0485236295158988,
		1.3961142690519968, -0.6126382415079216, -1.0129535006412997, -0.006343432541636995, 0.10365788307260332,
		1.8194943931423508, 0.8429286020106258, -0.2742898744473912, 0.1898336298464324, 0.14347615509108338,
		-0.2965046557091565, -0.4619406835659692, -0.9499501564693674, 1.5950787568852622, 0.0891115693003487,
		-0.10504681012107242, 0.1928019652148416, 0.1356842134152691, -0.14042989215201176, 1.028992496152653,
	})
	E := mat.NewDense(5, 5, []float64{
		0.7919723729573327, -0.19001960978151022, -0.8930751741565348, 0.21279655749497506, 0.06068432080776817,
		-0.35844456760602333, 0.7591354261076573, 0.1536894779284962, -0.12986735927859178, 0.05763436284717543,
		0.06135786283835317, -0.29501178844337733, 1.0213076589543948, 0.10719078332765039, -0.12053511192478951,
		-0.02863567540575398, 0.15213887429866824, -0.048205241153679176, 1.1223408823803718, 0.05536892092211159,
		0.02121330825207401, -0.08970961932909507, 0.01753879918838696, -0.08642149521377758, -0.013116167810996004,
	})
	B := mat.NewDense(5, 1, []float64{1, 1, 1, 1, 1})
	C := mat.NewDense(1, 5, []float64{1, 1, 1, 1, 1})
	sys, err := NewDescriptor(A, B, C, mat.NewDense(1, 1, nil), E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// TestStabsep_SingularERoundingBetaInfinite: for a continuous descriptor
// with E = U·blkdiag(J, 0)·V (σ_min(E) = 9.3e-18), QZ returned the
// infinite eigenvalue with a rounding-level β instead of 0, and Stabsep
// kept it as the stable pole −2.67e32: Stable(0.3+0.7i) = −8.86e15, true
// G = −2.34−0.92i. MATLAB treats |β| ≤ tol·|α| as infinite. The oracle is
// eig(J⁻¹H) = {−0.744±1.438i, 2.063±0.851i} and (sE − A)⁻¹. 4U3MDS.
func TestStabsep_SingularERoundingBetaInfinite(t *testing.T) {
	sys := roundingBetaDescriptor(t)
	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}
	poles, err := res.Stable.Poles()
	if err != nil {
		t.Fatal(err)
	}
	assertEigSetNear(t, "stable poles", poles, []complex128{
		complex(-0.743589820135963, 1.4379014276730409), complex(-0.743589820135963, -1.4379014276730409),
	}, 1e-8)
	assertSplitSum(t, "rounding-level beta", sys, res.Stable, res.Unstable)
}

// TestStabsep_SingularEIndexOne: Stabsep of index-1 descriptors
// U·(blkdiag(H, I), blkdiag(J, 0))·V. Before 4U3MDS 7% of them kept an
// infinite mode with rounding-level β as a huge stable pole (silently
// wrong Stable), 2–9% more failed with "cannot be separated reliably"
// because that mode made the split ill-conditioned, and with general U, V
// an infinite β just above rounding also leaked.
func TestStabsep_SingularEIndexOne(t *testing.T) {
	for _, orth := range []bool{true, false} {
		for seed := range uint64(400) {
			rng := rand.New(rand.NewPCG(seed, 5))
			r, d := 2+rng.IntN(5), 1+rng.IntN(3)
			w := newWeierstrassDescriptor(seed, r, d, false, orth)
			if _, err := w.sys.ToExplicit(); err == nil {
				continue
			}
			res, err := Stabsep(w.sys)
			if err != nil {
				t.Fatalf("orth %v seed %d: %v", orth, seed, err)
			}
			assertStabsepMatchesFinite(t, "index 1", w, res, 1e-7)
			if t.Failed() {
				t.Fatalf("orth %v seed %d", orth, seed)
			}
		}
	}
}

// higherIndexChains are the infinite Jordan structures of the index ≥ 2
// tests: chain lengths of N.
var higherIndexChains = [][]int{{2}, {3}, {4}, {2, 1}, {3, 2}, {2, 2, 1}}

// higherIndexTol is the relative accuracy of the finite part of a
// descriptor whose longest infinite chain has length k and whose finite
// spectral radius is rho: a rounding-level perturbation splits the chain
// into finite eigenvalues of modulus ~ε^(−1/k), so separating a finite
// pole λ from it is conditioned like |λ|^(k−1).
func higherIndexTol(w weierstrassDescriptor, k int) float64 {
	var eig mat.Eigen
	if !eig.Factorize(w.finite.A, mat.EigenNone) {
		panic("eig failed")
	}
	rho := 0.0
	for _, v := range eig.Values(nil) {
		rho = math.Max(rho, cmplx.Abs(v))
	}
	return math.Max(1e-7, 1e6*eps()*math.Pow(math.Max(rho, 1), float64(k-1)))
}

// TestStabsep_SingularEHigherIndex: an infinite Jordan chain of length k
// is resolved by QZ only to |β|/|α| ~ ε^(1/k), so its members looked like
// finite poles of modulus ~1e5–1e7. Stabsep kept them in Stable (wrong by
// up to 1e8 relative, 34% of index-2 cases), and once that failed closed,
// returned ErrSchurFailed for 5–84% of these descriptors (index 2–4) and
// still kept a few wrong. The infinite structure is now deflated before
// QZ (staircase on E, then on the matching rows of A), as MATLAB stabsep
// splits any regular descriptor. The oracle is eig(J⁻¹H), the real-Schur
// Stabsep of the explicit finite part and (sE − A)⁻¹. 7YDREH.
func TestStabsep_SingularEHigherIndex(t *testing.T) {
	for _, chains := range higherIndexChains {
		k := slices.Max(chains)
		for _, orth := range []bool{true, false} {
			for seed := range uint64(120) {
				rng := rand.New(rand.NewPCG(seed, 6))
				w := newWeierstrassChains(seed, 2+rng.IntN(5), chains, orth)
				label := fmt.Sprintf("chains %v orth %v seed %d", chains, orth, seed)
				res, err := Stabsep(w.sys)
				if err != nil {
					t.Fatalf("%s: %v", label, err)
				}
				assertStabsepMatchesFinite(t, label, w, res, higherIndexTol(w, k))
				if t.Failed() {
					t.FailNow()
				}
			}
		}
	}
}

// TestModsep_SingularEHigherIndex: Modsep of index ≥ 2 descriptors keeps
// the finite modes below the cutoff in Slow and the infinite chains in
// Fast. With cutoff 1e10 every finite mode is slow, and the chain members
// QZ left at moduli ~1e5–1e8 were retained too, failing closed. Oracle:
// eig(J⁻¹H) and (sE − A)⁻¹. 7YDREH.
func TestModsep_SingularEHigherIndex(t *testing.T) {
	for _, chains := range higherIndexChains {
		for _, orth := range []bool{true, false} {
			for seed := range uint64(40) {
				rng := rand.New(rand.NewPCG(seed, 6))
				w := newWeierstrassChains(seed, 2+rng.IntN(5), chains, orth)
				var eig mat.Eigen
				if !eig.Factorize(w.finite.A, mat.EigenNone) {
					t.Fatal("eig failed")
				}
				tol := higherIndexTol(w, slices.Max(chains))
				for _, cutoff := range []float64{1, 1e10} {
					label := fmt.Sprintf("chains %v orth %v seed %d cutoff %g", chains, orth, seed, cutoff)
					var want []complex128
					for _, v := range eig.Values(nil) {
						if cmplx.Abs(v) < cutoff {
							want = append(want, v)
						}
					}
					res, err := Modsep(w.sys, cutoff)
					if err != nil {
						t.Fatalf("%s: %v", label, err)
					}
					poles, err := res.Slow.Poles()
					if err != nil {
						t.Fatal(err)
					}
					assertEigSetNear(t, label+" slow poles", poles, want, 1e-6)
					for _, s := range []complex128{complex(0.3, 0.7), complex(-0.2, 2.5)} {
						got := pencilResponse(t, res.Slow, s)[0][0] + pencilResponse(t, res.Fast, s)[0][0]
						full := pencilResponse(t, w.sys, s)[0][0]
						if d := cmplx.Abs(got - full); !(d <= tol*(1+cmplx.Abs(full))) {
							t.Fatalf("%s: Slow+Fast(%v) = %v, want %v", label, s, got, full)
						}
					}
				}
			}
		}
	}
}

// TestPoles_SingularEHigherIndex: Poles and IsStable of index ≥ 2
// descriptors. QZ leaves the members of an infinite chain of length k at
// |β|/|α| ~ ε^(1/k), and Poles listed them as finite poles such as
// 1.19±7.03e7i (chain 3, seed 952) or −2.56e5 (seed 549); MATLAB pole
// leaves infinite poles out. Oracle: eig(J⁻¹H). 7YDREH.
func TestPoles_SingularEHigherIndex(t *testing.T) {
	for _, chains := range higherIndexChains {
		for _, orth := range []bool{true, false} {
			for _, seed := range append([]uint64{549, 920, 952}, rangeSeeds(150)...) {
				rng := rand.New(rand.NewPCG(seed, 6))
				w := newWeierstrassChains(seed, 2+rng.IntN(5), chains, orth)
				label := fmt.Sprintf("chains %v orth %v seed %d", chains, orth, seed)
				var eig mat.Eigen
				if !eig.Factorize(w.finite.A, mat.EigenNone) {
					t.Fatal("eig failed")
				}
				want := eig.Values(nil)
				poles, err := w.sys.Poles()
				if err != nil {
					t.Fatalf("%s: %v", label, err)
				}
				assertEigSetNear(t, label, poles, want, 1e-6)
				stable := true
				for _, p := range want {
					stable = stable && real(p) < 0
				}
				got, err := w.sys.IsStable()
				if err != nil || got != stable {
					t.Fatalf("%s: IsStable = %v, %v, want %v", label, got, err, stable)
				}
			}
		}
	}
}

func rangeSeeds(n int) []uint64 {
	s := make([]uint64, n)
	for i := range s {
		s[i] = uint64(i)
	}
	return s
}

// TestStabsep_SingularEStiffFinitePole: a genuine finite pole of
// norm-scaled modulus 1e6 next to an infinite mode stays a stable pole
// (no fail-closed): E has index 1, so its one infinite eigenvalue is
// certified and the large retained one is finite.
func TestStabsep_SingularEStiffFinitePole(t *testing.T) {
	for seed := range uint64(20) {
		rng := rand.New(rand.NewPCG(seed, 7))
		n := 5
		F := mat.NewDiagDense(n, []float64{-1, -2, 0.5, 3, 1})
		W := mat.NewDiagDense(n, []float64{1e-6, 1, 1, 1, 0})
		U, V := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				U.Set(i, j, rng.NormFloat64())
				V.Set(i, j, rng.NormFloat64())
			}
		}
		var T, A, E mat.Dense
		T.Mul(U, F)
		A.Mul(&T, V)
		T.Mul(U, W)
		E.Mul(&T, V)
		B, C := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
		for i := range n {
			B.Set(i, 0, 1)
			C.Set(0, i, 1)
		}
		sys, err := NewDescriptor(&A, B, C, mat.NewDense(1, 1, nil), &E, 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := Stabsep(sys)
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		poles, err := res.Stable.Poles()
		if err != nil {
			t.Fatal(err)
		}
		assertEigSetNear(t, "stable poles", poles, []complex128{-1e6, -2}, 1e-7)
		pts := []complex128{complex(0.3, 0.7), complex(-0.2, 2.5)}
		for _, s := range pts {
			want := pencilResponse(t, sys, s)[0][0]
			got := pencilResponse(t, res.Stable, s)[0][0] + pencilResponse(t, res.Unstable, s)[0][0]
			if d := cmplx.Abs(got - want); !(d <= 1e-8*math.Max(1, cmplx.Abs(want))) {
				t.Fatalf("seed %d: sum(%v) = %v, want %v", seed, s, got, want)
			}
		}
	}
}

// TestPoles_SingularERoundingBetaInfinite: Poles and IsStable of
// singular-E descriptors whose infinite eigenvalues QZ returns with
// rounding-level β. Poles skipped only β == 0 and listed them as finite
// poles such as 3.94e15 (seed 5) or 1e12–1e13 (general U, V); MATLAB pole
// leaves infinite poles out. The oracle is eig(J⁻¹H) of the Weierstrass
// form. IOAGOI.
func TestPoles_SingularERoundingBetaInfinite(t *testing.T) {
	sys := roundingBetaDescriptor(t)
	poles, err := sys.Poles()
	if err != nil {
		t.Fatal(err)
	}
	assertEigSetNear(t, "4U3MDS", poles, []complex128{
		complex(-0.743589820135963, 1.4379014276730409), complex(-0.743589820135963, -1.4379014276730409),
		complex(2.063, 0.851), complex(2.063, -0.851),
	}, 2e-3)
	res, err := Stabsep(sys)
	if err != nil {
		t.Fatal(err)
	}
	poles, err = res.Unstable.Poles()
	if err != nil {
		t.Fatal(err)
	}
	assertEigSetNear(t, "4U3MDS unstable", poles, []complex128{complex(2.063, 0.851), complex(2.063, -0.851)}, 2e-3)

	for _, orth := range []bool{true, false} {
		for seed := range uint64(400) {
			rng := rand.New(rand.NewPCG(seed, 5))
			r, d := 2+rng.IntN(5), 1+rng.IntN(3)
			w := newWeierstrassDescriptor(seed, r, d, false, orth)
			if _, err := w.sys.ToExplicit(); err == nil {
				continue
			}
			var eig mat.Eigen
			if !eig.Factorize(w.finite.A, mat.EigenNone) {
				t.Fatal("eig failed")
			}
			want := eig.Values(nil)
			poles, err := w.sys.Poles()
			if err != nil {
				t.Fatalf("orth %v seed %d: %v", orth, seed, err)
			}
			assertEigSetNear(t, fmt.Sprintf("orth %v seed %d", orth, seed), poles, want, 1e-6)
			stable := true
			for _, p := range want {
				stable = stable && real(p) < 0
			}
			got, err := w.sys.IsStable()
			if err != nil || got != stable {
				t.Fatalf("orth %v seed %d: IsStable = %v, %v, want %v", orth, seed, got, err, stable)
			}
		}
	}
}

// TestPoles_SingularEStiffFinitePole: a genuine finite pole of norm-scaled
// modulus 1e6 beside an index-1 infinite mode stays a pole.
func TestPoles_SingularEStiffFinitePole(t *testing.T) {
	for seed := range uint64(20) {
		rng := rand.New(rand.NewPCG(seed, 7))
		n := 5
		F := mat.NewDiagDense(n, []float64{-1, -2, 0.5, 3, 1})
		W := mat.NewDiagDense(n, []float64{1e-6, 1, 1, 1, 0})
		U, V := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				U.Set(i, j, rng.NormFloat64())
				V.Set(i, j, rng.NormFloat64())
			}
		}
		var T, A, E mat.Dense
		T.Mul(U, F)
		A.Mul(&T, V)
		T.Mul(U, W)
		E.Mul(&T, V)
		sys, err := NewDescriptor(&A, mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil), mat.NewDense(1, 1, nil), &E, 0)
		if err != nil {
			t.Fatal(err)
		}
		poles, err := sys.Poles()
		if err != nil {
			t.Fatal(err)
		}
		assertEigSetNear(t, fmt.Sprintf("seed %d", seed), poles, []complex128{-1e6, -2, 0.5, 3}, 1e-6)
	}
}
