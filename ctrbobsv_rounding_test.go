package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// roundingCoupledRepros (4KYHPC) have exactly uncontrollable modes, an
// unstable pair and a ±i pair, under random orthogonal similarity.
var roundingCoupledRepros = []struct {
	name string
	n    int
	a, b []float64
}{
	{"unstable pair 0.908±1.662i", 6, []float64{0.7519137808354553, 0.12939949641140489, -0.9503608336118341, 0.9120474126346396, 0.12048307227555893, -0.7611451687010268, 1.814260680417558, -1.2435253220300282, -0.40496225128182517, 0.4076255041355587, -0.4590311150773289, -1.7381240156669302, -0.2392392363457023, 0.10573195595523201, -0.38836648888875464, -0.21652938698537055, 0.5585651185180622, 0.9737778541787145, -1.2551707651837514, -0.44836624444999246, -0.41485753488000277, -0.011274606950283855, 0.9390576119595544, 0.10986027054017995, 0.02495513035863211, -2.3786874912320317, -1.7908701101888822, 0.12457847764826456, -0.5369013237399257, -1.8748249655049873, 0.8687505448914252, 0.5165668514200726, 0.29354303095748474, 0.44790018519241237, -0.07601168525970689, 0.9313271492598416},
		[]float64{-0.21492145947508068, -1.0405945769662017, -0.1832331061129103, 0.5243543786434914, 0.5208590210547333, 1.024303779378484}},
	{"boundary pair ±i", 9, []float64{0.3033853960868275, -0.995286725866401, 1.068429592419651, -0.9144369234256374, 0.38593350707377627, -0.3485129644749626, -1.231257401890236, 0.6409396344103141, 1.3938098604604725, 0.2569604950091591, 0.5369629546514139, -0.6255566838448661, 1.300383311992172, 0.016536098902014823, -1.1272080506229285, -0.941672048299403, 0.4797225562248032, -0.17565443278733855, -1.2946334295866873, -0.831323337804918, 1.8394399368317809, -1.6745455349367344, -0.24417761569352797, -0.5381917426249178, -0.5187651573939237, -0.19716522780640328, 1.0963342959734474, -0.41526048353585376, -0.12620068869516116, -1.0781397450430281, -0.983415275657254, -1.0628766298953927, -0.14836226634704372, -0.131186976393843, 0.9165575405403497, 0.1753839848120577, -1.0503413781230013, -1.717403462370597, -0.06601624674410055, -0.22496431664715008, -1.0299561200420215, 0.002350440716269641, -0.18217137245801138, -0.5946311734209873, -0.05912030405070813, -1.8425899616592905, 1.7801808330279902, 0.1391772653205943, 0.6072047239015627, 0.6243952573286996, -1.6035598719772992, -0.7872463160882881, -1.1008558304202813, -1.226472778600163, -0.5848628447886609, -0.9502542719216095, -0.09123468701055523, 1.0687625306051969, 0.1751883755905236, -0.10825545123308168, 1.0314156703258948, 0.14855860617118422, 0.010208437412877436, 0.5707848608133327, -0.9270934819539919, -1.8564434648227073, -0.5165472376486263, 0.6691047407770093, 1.648488727863011, 0.5597010754208883, 0.4852075962702503, 0.6666816213948519, 0.3838181770589459, 0.8993387578110765, -0.26767843509018147, 0.6189333745773896, 0.45251366809347465, -1.139248894855972, -0.5126150179199848, -0.06703075278523302, -0.34239290804014505},
		[]float64{-1.5759817736364008, 0.7266140379513089, -1.0743519780971844, -0.9435837835012602, 0.15664743747447954, -0.37272357152938096, 0.9216939425001934, -0.3304004355879953, -0.05863582610265551}},
}

// TestIsStabilizable_RoundingCoupledUncontrollable: the staircase counted
// the repros' rounding-level coupling (5e-14, 4e-13) as rank, so base
// returned true; PBH σmin at the pair is 3e-16.
func TestIsStabilizable_RoundingCoupledUncontrollable(t *testing.T) {
	for _, r := range roundingCoupledRepros {
		A := mat.NewDense(r.n, r.n, r.a)
		B := mat.NewDense(r.n, 1, r.b)
		if ok, err := IsStabilizable(A, B, true); err != nil || ok {
			t.Errorf("%s: IsStabilizable = %v, %v, want false", r.name, ok, err)
		}
		if ok, err := IsDetectable(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T()), true); err != nil || ok {
			t.Errorf("%s: IsDetectable = %v, %v, want false", r.name, ok, err)
		}
	}
}

func stabSweepOrthogonal(rng *rand.Rand, n int) *mat.Dense {
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

// TestIsStabilizable_RandomUncontrollableSweep: A₀ = [A11 A12; 0 A22],
// B₀ = [B1; 0] under random orthogonal similarity, the verdict known from
// eig(A22) by construction. A22 has a mode resolvably unstable or exactly
// on the boundary (rotation ±iω, z = e^{±iθ}) or is resolvably stable; the
// controllable modes are random. Base missed 43 (arm64) and 8 (amd64) of
// the unstable cases.
func TestIsStabilizable_RandomUncontrollableSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(0x4b59, 0x4850))
	var misses [2]int
	for trial := range 6000 {
		n := 3 + rng.IntN(8)
		m := 1 + rng.IntN(3)
		nu := 1 + rng.IntN(min(3, n-1))
		nc := n - nu
		continuous := trial%2 == 0
		kind := rng.IntN(3)
		A0 := mat.NewDense(n, n, nil)
		B0 := mat.NewDense(n, m, nil)
		for i := range n {
			for j := range n {
				if i < nc || j >= nc {
					A0.Set(i, j, rng.NormFloat64())
				}
			}
		}
		for i := range nc {
			for j := range m {
				B0.Set(i, j, rng.NormFloat64())
			}
		}
		if kind == 2 && nu >= 2 {
			for i := nc; i < n; i++ {
				for j := nc; j < n; j++ {
					A0.Set(i, j, 0)
				}
				A0.Set(i, i, -0.5)
				if !continuous {
					A0.Set(i, i, 0.5)
				}
			}
			w := 0.5 + rng.Float64()
			re, im := 0.0, w
			if !continuous {
				re, im = math.Cos(w), math.Sin(w)
			}
			A0.Set(nc, nc, re)
			A0.Set(nc+1, nc+1, re)
			A0.Set(nc, nc+1, -im)
			A0.Set(nc+1, nc, im)
		}
		a22 := mat.DenseCopyOf(A0.Slice(nc, n, nc, n))
		var e mat.Eigen
		if !e.Factorize(a22, mat.EigenNone) {
			t.Fatalf("trial %d: eig(A22) failed", trial)
		}
		want, skip := true, false
		for _, v := range e.Values(nil) {
			d := real(v)
			if !continuous {
				d = cmplxAbs(v) - 1
			}
			switch {
			case kind == 2 && nu >= 2 && math.Abs(d) < 1e-12:
				want = false
			case d > 1e-3:
				want = false
			case d > -1e-3:
				skip = true
			}
		}
		if skip {
			continue
		}
		Q := stabSweepOrthogonal(rng, n)
		var A, B mat.Dense
		A.Product(Q, A0, Q.T())
		B.Mul(Q, B0)
		got, err := IsStabilizable(&A, &B, continuous)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if got != want {
			idx := 0
			if want {
				idx = 1
			}
			misses[idx]++
			if misses[idx] <= 5 {
				t.Errorf("trial %d (n=%d m=%d nu=%d cont=%v kind=%d): IsStabilizable = %v, want %v", trial, n, m, nu, continuous, kind, got, want)
			}
		}
	}
	if misses != [2]int{} {
		t.Errorf("misses: %d true want false, %d false want true", misses[0], misses[1])
	}
}

// TestIsStabilizable_WeakCouplingControllableJordan: a controllable unstable
// Jordan block (B reaches the chain end; eigenvalues split by √eps, slack at
// its cap) beside a stable mode coupled at 1e-12, so the rounding-coupling
// PBH pass runs: the unstable modes stay controllable.
func TestIsStabilizable_WeakCouplingControllableJordan(t *testing.T) {
	rng := rand.New(rand.NewPCG(4, 0x4b))
	for trial := range 200 {
		continuous := trial%2 == 0
		lam := 1.0
		if !continuous {
			lam = 1.5
		}
		A0 := mat.NewDense(4, 4, []float64{
			lam, 1, rng.NormFloat64(), rng.NormFloat64(),
			0, lam, rng.NormFloat64(), rng.NormFloat64(),
			0, 0, -0.5, 0,
			0, 0, 0, -0.25,
		})
		B0 := mat.NewDense(4, 1, []float64{rng.NormFloat64(), 1, 1e-12, 1})
		Q := stabSweepOrthogonal(rng, 4)
		var A, B mat.Dense
		A.Product(Q, A0, Q.T())
		B.Mul(Q, B0)
		st, err := controllabilityStaircase(&A, &B, nil, 0, false)
		if err != nil {
			t.Fatal(err)
		}
		if st.MinCoupling > 4*math.Sqrt(eps())*frobNorm(&A) {
			t.Fatalf("trial %d: MinCoupling %g, PBH pass not exercised", trial, st.MinCoupling)
		}
		ok, err := IsStabilizable(&A, &B, continuous)
		if err != nil || !ok {
			t.Errorf("trial %d (cont=%v): IsStabilizable = %v, %v, want true", trial, continuous, ok, err)
		}
	}
}

// weakChainEndSystem: an unstable 2×2 Jordan block whose chain end is
// reached only through c, in B (two inputs) or, single input, B = [0.3; c; 1; 1].
func weakChainEndSystem(c float64, continuous, single bool) (*mat.Dense, *mat.Dense) {
	l, s1, s2 := 1.0, -0.5, -0.25
	if !continuous {
		l, s1, s2 = 1.5, 0.5, 0.25
	}
	A := mat.NewDense(4, 4, []float64{l, 1, 0.3, -0.7, 0, l, 0, 0, 0, 0, s1, 0, 0, 0, 0, s2})
	if single {
		return A, mat.NewDense(4, 1, []float64{0.3, c, 1, 1})
	}
	return A, mat.NewDense(4, 2, []float64{0.3, 0, c, 0, 0, 1, 0, 1})
}

// TestIsStabilizable_WeakChainEndControllableJordan: the Jordan chain end is
// reached only through c, far above rounding (distance to uncontrollability
// 1e2–1e5 × n²·eps·‖[A, B]‖), so MATLAB's ctrbf counts nc = n; a PBH slack
// capped at n·√eps·‖A‖ called it unstabilizable. c = 0 is uncontrollable.
// Independent oracle: a stabilizing Riccati gain for two inputs, c = 1e-7, 1e-8.
func TestIsStabilizable_WeakChainEndControllableJordan(t *testing.T) {
	cases := []struct {
		c      float64
		single bool
	}{{1e-8, false}, {1e-9, false}, {1e-10, false}, {1e-7, true}, {0, false}, {0, true}}
	for _, continuous := range []bool{true, false} {
		for _, tc := range cases {
			A, B := weakChainEndSystem(tc.c, continuous, tc.single)
			want := tc.c > 0
			sf, err := CtrbF(A, B, mat.NewDense(1, 4, nil), 0)
			if err != nil {
				t.Fatal(err)
			}
			if nc := sumInts(sf.K); (nc == 4) != want {
				if want {
					t.Logf("cont=%v c=%g single=%v: CtrbF nc = %d on this platform; skipped", continuous, tc.c, tc.single, nc)
					continue
				}
				t.Fatalf("cont=%v c=%g single=%v: CtrbF nc = %d, want < 4", continuous, tc.c, tc.single, nc)
			}
			if ok, err := IsStabilizable(A, B, continuous); err != nil || ok != want {
				t.Errorf("cont=%v c=%g single=%v: IsStabilizable = %v, %v, want %v", continuous, tc.c, tc.single, ok, err, want)
			}
			if ok, err := IsDetectable(mat.DenseCopyOf(A.T()), mat.DenseCopyOf(B.T()), continuous); err != nil || ok != want {
				t.Errorf("cont=%v c=%g single=%v: IsDetectable = %v, %v, want %v", continuous, tc.c, tc.single, ok, err, want)
			}
		}
		for _, c := range []float64{1e-7, 1e-8} {
			A, B := weakChainEndSystem(c, continuous, false)
			m := 2
			var res *RiccatiResult
			var err error
			if continuous {
				res, err = Care(A, B, eye(4), eye(m), nil)
			} else {
				res, err = Dare(A, B, eye(4), eye(m), nil)
			}
			if err != nil {
				t.Fatalf("cont=%v c=%g: Riccati oracle: %v", continuous, c, err)
			}
			var cl mat.Dense
			cl.Mul(B, res.K)
			cl.Sub(A, &cl)
			var e mat.Eigen
			if !e.Factorize(&cl, mat.EigenNone) {
				t.Fatal("eig(A-BK) failed")
			}
			for _, v := range e.Values(nil) {
				if (continuous && real(v) >= 0) || (!continuous && cmplxAbs(v) >= 1) {
					t.Fatalf("cont=%v c=%g: oracle gain not stabilizing: %v", continuous, c, v)
				}
			}
		}
	}
}

// TestIsStabilizable_WeakChainEndJordanSweep: A₀ = [J C; 0 S], J an unstable
// Jordan block (size 2–3), S stable, under random orthogonal similarity. The
// chain end is reached only through c (a B entry or an A entry from a
// controllable stable state); every other state is driven directly.
// c ∈ 10^U(−10, −6) is controllable by construction; c = 0 leaves the
// unstable chain end uncontrollable (asserted for size 2; size 3 splits by
// eps^⅓, beyond the PBH pass's reach, a known precision limit). Controllable
// draws are asserted only when MATLAB's ctrbf (CtrbF, tol 0) agrees and the
// distance to uncontrollability of the exact (A₀, B₀), B₀ scaled to ‖A₀‖_F,
// exceeds 100·n²·eps·‖[A₀, B₀]‖_F. The capped Lipschitz slack missed 133 of
// 382 controllable draws; base missed an uncontrollable one.
func TestIsStabilizable_WeakChainEndJordanSweep(t *testing.T) {
	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(0x5eed, 22))
	var checked int
	var misses [2]int
	for trial := range 600 {
		continuous := trial%2 == 0
		js := 2 + rng.IntN(2)
		ns := 1 + rng.IntN(4)
		n := js + ns
		m := 1 + rng.IntN(3)
		viaA := rng.IntN(2) == 1
		c := 0.0
		if trial%6 != 5 {
			c = math.Pow(10, -10+4*rng.Float64())
		}
		l := 0.2 + rng.Float64()
		if !continuous {
			l = 1.2 + 0.5*rng.Float64()
			if rng.IntN(2) == 1 {
				l = -l
			}
		}
		A0 := mat.NewDense(n, n, nil)
		B0 := mat.NewDense(n, m, nil)
		end := js - 1
		for i := range js {
			A0.Set(i, i, l)
			if i < end {
				A0.Set(i, i+1, 1)
				for j := js; j < n; j++ {
					A0.Set(i, j, rng.NormFloat64())
				}
				for j := range m {
					B0.Set(i, j, rng.NormFloat64())
				}
			}
		}
		for i := js; i < n; i++ {
			d := -0.2 - rng.Float64()
			if !continuous {
				d = 0.9 * (2*rng.Float64() - 1)
			}
			A0.Set(i, i, d)
			for j := i + 1; j < n; j++ {
				A0.Set(i, j, rng.NormFloat64())
			}
			for j := range m {
				B0.Set(i, j, rng.NormFloat64())
			}
		}
		if viaA {
			A0.Set(end, js+rng.IntN(ns), c)
		} else {
			B0.Set(end, rng.IntN(m), c)
		}
		want := c > 0
		if !want && js == 3 {
			continue
		}
		if want {
			an := frobNorm(A0)
			bs := mat.DenseCopyOf(B0)
			bs.Scale(an/frobNorm(B0), bs)
			if pbhDistanceNear(A0, bs, complex(l, 0)) < 100*float64(n*n)*eps()*math.Sqrt2*an {
				continue
			}
		}
		Q := stabSweepOrthogonal(rng, n)
		var A, B mat.Dense
		A.Product(Q, A0, Q.T())
		B.Mul(Q, B0)
		if want {
			sf, err := CtrbF(&A, &B, mat.NewDense(1, n, nil), 0)
			if err != nil {
				t.Fatal(err)
			}
			if sumInts(sf.K) != n {
				continue
			}
		}
		checked++
		got, err := IsStabilizable(&A, &B, continuous)
		if err != nil {
			t.Fatalf("trial %d: %v", trial, err)
		}
		if got != want {
			idx := 0
			if want {
				idx = 1
			}
			misses[idx]++
			if misses[idx] <= 5 {
				t.Errorf("trial %d (n=%d m=%d js=%d viaA=%v cont=%v c=%g): IsStabilizable = %v, want %v", trial, n, m, js, viaA, continuous, c, got, want)
			}
		}
	}
	t.Logf("checked %d", checked)
	if misses != [2]int{} || checked < 300 {
		t.Errorf("of %d checked: %d true want false, %d false want true", checked, misses[0], misses[1])
	}
}

// pbhDistanceNear estimates min over λ near lam0 of σmin([A − λI, B]), the
// distance to uncontrollability, by a log-polar grid down to 1e-14 then
// compass refinement, with its own realified SVD.
func pbhDistanceNear(A, B *mat.Dense, lam0 complex128) float64 {
	sig := func(z complex128) float64 { return realifiedPBHSigma(A, B, z) }
	best, bz := sig(lam0), lam0
	for k := range 15 {
		r := math.Pow(10, -float64(k))
		for a := range 4 {
			z := lam0 + cmplx.Rect(r, float64(a)*math.Pi/2)
			if s := sig(z); s < best {
				best, bz = s, z
			}
		}
	}
	step := cmplx.Abs(bz-lam0) + 1e-14
	for range 120 {
		moved := false
		for _, d := range [4]complex128{1, -1, 1i, -1i} {
			if s := sig(bz + d*complex(step, 0)); s < best {
				best, bz, moved = s, bz+d*complex(step, 0), true
				break
			}
		}
		if !moved {
			step /= 2
			if step < 1e-16 {
				break
			}
		}
	}
	return best
}

// realifiedPBHSigma is σmin([A − λI, B]) through its own real embedding
// [[Re, −Im], [Im, Re]], independent of pbhSigmaMin.
func realifiedPBHSigma(A, B *mat.Dense, z complex128) float64 {
	n, _ := A.Dims()
	_, m := B.Dims()
	g := mat.NewDense(2*n, 2*(n+m), nil)
	var svd mat.SVD
	for i := range n {
		for j := range n {
			re := A.At(i, j)
			if i == j {
				re -= real(z)
			}
			g.Set(i, j, re)
			g.Set(n+i, n+m+j, re)
			if i == j {
				g.Set(i, n+m+j, imag(z))
				g.Set(n+i, j, -imag(z))
			}
		}
		for j := range m {
			g.Set(i, n+j, B.At(i, j))
			g.Set(n+i, 2*n+m+j, B.At(i, j))
		}
	}
	svd.Factorize(g, mat.SVDNone)
	v := svd.Values(nil)
	return v[len(v)-1]
}
