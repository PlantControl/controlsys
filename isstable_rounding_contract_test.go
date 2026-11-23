package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/lapack"
	"plantcontrol.org/v1/gonum/mat"
)

// IsStable's contract (GEFTJ6): for an explicit delay-free model its bool is
// the exact band verdict of a matrix within K·ε·‖A‖_F, K = 4(n+5), of A in
// the stored scaling or in its diagonally balanced form. Near the band edge
// either verdict is allowed; beyond it the verdict is exact.

type roundingLit struct {
	a, b, c, d *mat.Dense
}

func (l roundingLit) sys(t *testing.T) *System {
	t.Helper()
	sys, err := New(l.a, l.b, l.c, l.d, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

var (
	roundingClusterUnstable = roundingLit{
		// rotZ/62 (arm64 data)
		a: mat.NewDense(6, 6, []float64{0.5653997683149359, 0.33367611969060534, 0.40589531277193974, 0.8308014236484298, -0.41733046212245006, 0.05718953486617096, -0.4629978086129042, -0.7142450093351647, 0.22592827305196375, 0.23067020845866923, -1.1514618279647182, -0.3391414270171264, 0.2262725410392297, -0.013546982559455691, -0.08802099428468889, -0.45400954356985507, 0.21944777344461902, 0.31770213555274873, 0.3970585475443188, 0.694441075385292, 0.4994890445328307, -1.1125904810894833, -1.174521725369059, 0.5847347002370783, 0.8809751390657041, 0.6590511214588771, 0.46689329255392004, -0.498420426620728, -0.8309831660233953, 0.47260431264014646, -0.5206181361371552, 0.41094336820100597, 0.04968808789780175, 0.47046825383488394, 0.46567782098877397, 0.18043594850760242}),
		b: mat.NewDense(6, 1, []float64{2, 1, 0, 0.5, 0.5, 0.5}),
		c: mat.NewDense(1, 6, []float64{0, -1, 2, -1, -1, 1}),
		d: mat.NewDense(1, 1, []float64{0}),
	}
	roundingClusterStable = roundingLit{
		// rotZ/886 (arm64 data)
		a: mat.NewDense(6, 6, []float64{-7.450580596923828e-09, 0, 0, 0, 0, 2, -1, -3.725290298461914e-09, -2, 0, 0, 0, -1, 2, -3.725290298461914e-09, 0, 0, 1, 0.5, 0, 0, -1, 1, 1, 1, -0.5, 0, -1, -1, -0.5, -2, 0, 0, 0, 0, -7.450580596923828e-09}),
		b: mat.NewDense(6, 1, []float64{2, -1, 1, 0.5, -1, 2}),
		c: mat.NewDense(1, 6, []float64{0.5, -1, -1, -1, 0.5, 0.5}),
		d: mat.NewDense(1, 1, []float64{0}),
	}
	roundingSlowPoleFastPole = roundingLit{
		// rotZ/183 Balred MatchDC order 3 (arm64 data)
		a: mat.NewDense(3, 3, []float64{-8.63294138660643e-11, -0.5000000000984157, 0.1227439328468379, 0.500000000098416, -5.210109393493452e-10, 0.23958566524480146, -0.12274406363420412, 0.23958563573942324, -2.9881078629865026e+08}),
		b: mat.NewDense(3, 1, []float64{-1.1081326643583664, 2.7222982850164335, -6.97583816480629e+08}),
		c: mat.NewDense(1, 3, []float64{1.1081320854828347, 2.7222983089009714, -6.97583660746003e+08}),
		d: mat.NewDense(1, 1, []float64{-1.628532479506893e+09}),
	}
	roundingClear = roundingLit{
		// rotZ/30 (arm64 data)
		a: mat.NewDense(7, 7, []float64{-3.814697265625e-06, 0, 0, -0.5, -1, 0, -1, 0, -2.3283064365386963e-10, 0.5, 0.5, -1, 0, 1, 0, -0.5, -2.3283064365386963e-10, -1, 0, 0, -1, 0, 0, 0, -1, 1, -1, -0.5, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0, -2, 0.5, 0, 0, 0, 0, 0, -0.5, -2}),
		b: mat.NewDense(7, 1, []float64{0.5, 2, 2, 0, -1, -1, 0.5}),
		c: mat.NewDense(1, 7, []float64{0.5, 0.5, 2, 1, -1, 2, 0}),
		d: mat.NewDense(1, 1, []float64{0}),
	}
)

// bandEdgePoint is the point of IsStable's band edge nearest p.
func bandEdgePoint(p complex128, continuous bool) complex128 {
	tol := poleStabilityTolerance(p)
	if continuous {
		return complex(-tol, imag(p))
	}
	r := cmplx.Abs(p)
	if r == 0 {
		return complex(1-tol, 0)
	}
	return p * complex((1-tol)/r, 0)
}

// sigmaMinShift is σmin(M − zI), from the real 2n embedding of M − zI.
func sigmaMinShift(m *mat.Dense, z complex128) float64 {
	n, _ := m.Dims()
	x, y := real(z), imag(z)
	e := mat.NewDense(2*n, 2*n, nil)
	for i := range n {
		for j := range n {
			v := m.At(i, j)
			if i == j {
				v -= x
			}
			e.Set(i, j, v)
			e.Set(n+i, n+j, v)
		}
		e.Set(i, n+i, y)
		e.Set(n+i, i, -y)
	}
	var svd mat.SVD
	if !svd.Factorize(e, mat.SVDNone) {
		return math.Inf(1)
	}
	return svd.Values(nil)[2*n-1]
}

func balancedForm(a *mat.Dense) *mat.Dense {
	n, _ := a.Dims()
	data := slices.Clone(mat.DenseCopyOf(a).RawMatrix().Data)
	impl.Dgebal(lapack.PermuteScale, n, data, n, make([]float64, n))
	return mat.NewDense(n, n, data)
}

// bandEdgeDistance is the witness of the rounding contract: the least
// σmin(M − zI)/(ε‖M‖_F) over M the stored A and its balanced form and z the
// band-edge point nearest a computed pole of M. A value d proves a matrix
// within d·ε‖M‖_F of M with a pole on the band edge, which IsStable rejects;
// it bounds how far a "false" is from the exact verdict. For a "true" on a
// model that is not band-stable it is a necessary condition only.
func bandEdgeDistance(a *mat.Dense, continuous bool) float64 {
	best := math.Inf(1)
	for _, m := range []*mat.Dense{a, balancedForm(a)} {
		u := eps() * mat.Norm(m, 2)
		var eig mat.Eigen
		if !eig.Factorize(m, mat.EigenNone) {
			continue
		}
		for _, p := range eig.Values(nil) {
			best = math.Min(best, sigmaMinShift(m, bandEdgePoint(p, continuous))/u)
		}
	}
	return best
}

// exactBandStable decides the band on the exact float64 A by the integer
// Routh / Schur–Cohn test of pidTuningExactRootsStable: Re λ < −1e-10 or
// |λ| < 1−1e-10, IsStable's band for every pole with |λ| ≤ 1.
func exactBandStable(a *mat.Dense, continuous bool) bool {
	n, _ := a.Dims()
	type dy struct {
		m *big.Int
		e int
	}
	ent := make([]dy, n*n)
	emin := math.MaxInt
	for i := range n {
		for j := range n {
			d := newDyadic(a.At(i, j))
			ent[i*n+j] = dy{d.m, d.e}
			if d.m.Sign() != 0 {
				emin = min(emin, d.e)
			}
		}
	}
	if emin == math.MaxInt {
		emin = 0
	}
	m := make([][]*big.Int, n)
	for i := range n {
		m[i] = make([]*big.Int, n)
		for j := range n {
			d := ent[i*n+j]
			m[i][j] = new(big.Int)
			if d.m.Sign() != 0 {
				m[i][j].Lsh(d.m, uint(d.e-emin))
			}
		}
	}
	ln, ld := big.NewInt(1), big.NewInt(1)
	if emin >= 0 {
		ld.Lsh(ld, uint(emin))
	} else {
		ln.Lsh(ln, uint(-emin))
	}
	return pidTuningExactRootsStable(intCharPoly(m), ln, ld, continuous)
}

func TestIsStable_BandEdgeWitnessRejectsClearModel(t *testing.T) {
	a := roundingClear.a
	n, _ := a.Dims()
	if d := bandEdgeDistance(a, true); d <= float64(n+5) {
		t.Fatalf("band-edge witness %.3g·ε‖A‖_F on a model clear of the band, want > %d", d, n+5)
	}
}

// GEFTJ6: clustered lightly damped pairs (g3 rotZ/62, exactly unstable at
// +1.49e-9; rotZ/886, exactly stable at −3.7e-9). The eigensolver's verdict
// flips between arches; the contract allows either.
func TestIsStable_WithinRoundingOfBandEitherVerdict(t *testing.T) {
	for _, tc := range []struct {
		name        string
		lit         roundingLit
		exactStable bool
	}{
		{"exactly unstable cluster", roundingClusterUnstable, false},
		{"exactly stable cluster", roundingClusterStable, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.lit.sys(t)
			st, err := sys.IsStable()
			if err != nil {
				t.Fatal(err)
			}
			if got := exactBandStable(sys.A, true); got != tc.exactStable {
				t.Fatalf("exact band verdict %v, want %v", got, tc.exactStable)
			}
			n, _, _ := sys.Dims()
			if d := bandEdgeDistance(sys.A, true); d > float64(n+5) {
				t.Fatalf("IsStable = %v; band-edge witness %.3g·ε‖A‖_F, want ≤ %d", st, d, n+5)
			}
		})
	}
}

// GEFTJ6: a Balred MatchDC output (g3 rotZ/183, order 3) whose slow pair at
// −2.33e-10 ± 0.5i lies below ε‖A‖_F ≈ 6.6e-8 beside a fast pole −3e8. The
// eigensolver returns +1.99e-9 on arm64.
func TestIsStable_SlowPoleBelowNormRounding(t *testing.T) {
	sys := roundingSlowPoleFastPole.sys(t)
	if _, err := sys.IsStable(); err != nil {
		t.Fatal(err)
	}
	if !exactBandStable(sys.A, true) {
		t.Fatal("exact band verdict false, want true")
	}
	n, _, _ := sys.Dims()
	if d := bandEdgeDistance(sys.A, true); d > float64(n+5) {
		t.Fatalf("band-edge witness %.3g·ε‖A‖_F, want ≤ %d", d, n+5)
	}
}

// unimodularSimilar returns U·T·U⁻¹ for U = I + N, N the unit superdiagonal,
// computed exactly; ok is false if an entry does not round-trip float64.
func unimodularSimilar(T []float64, n int) (*mat.Dense, bool) {
	r := func(x float64) *big.Rat { return new(big.Rat).SetFloat64(x) }
	ut := make([]*big.Rat, n*n)
	for i := range n {
		for j := range n {
			v := r(T[i*n+j])
			if i+1 < n {
				v.Add(v, r(T[(i+1)*n+j]))
			}
			ut[i*n+j] = v
		}
	}
	out := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			s := new(big.Rat)
			for k := 0; k <= j; k++ {
				v := new(big.Rat).Set(ut[i*n+k])
				if (j-k)%2 == 1 {
					v.Neg(v)
				}
				s.Add(s, v)
			}
			f, exact := s.Float64()
			if !exact {
				return nil, false
			}
			out.Set(i, j, f)
		}
	}
	return out, true
}

// GEFTJ6: models whose exact poles clear the band by far more than 8K·ε‖A‖_F
// get the exact verdict, whatever the size or non-normality.
func TestIsStable_BeyondRoundingMatchesExact(t *testing.T) {
	type pole struct{ re, im float64 }
	build := func(n int, disc bool, special []pole) []float64 {
		T := make([]float64, n*n)
		i := 0
		put := func(p pole) {
			if p.im == 0 || i+1 >= n {
				T[i*n+i] = p.re
				i++
				return
			}
			T[i*n+i], T[i*n+i+1], T[(i+1)*n+i], T[(i+1)*n+i+1] = p.re, p.im, -p.im, p.re
			i += 2
		}
		for _, p := range special {
			put(p)
		}
		for k := 0; i < n; k++ {
			v := -0.25 - 0.125*float64(k%5)
			if disc {
				v = 0.5 - 0.0625*float64(k%7)
			}
			put(pole{v, 0})
		}
		for r := range n {
			for c := r + 1; c < n; c++ {
				if (r+2*c)%3 == 0 && T[r*n+c] == 0 {
					T[r*n+c] = 0.0625
				}
			}
		}
		return T
	}
	m := math.Ldexp(1, -20)
	for _, n := range []int{4, 8, 16, 24, 32} {
		for _, tc := range []struct {
			name   string
			disc   bool
			poles  []pole
			stable bool
		}{
			{"continuous slow stable", false, []pole{{-m, 0}}, true},
			{"continuous slow unstable", false, []pole{{m, 0}}, false},
			{"continuous lightly damped pair", false, []pole{{-m, 1}}, true},
			{"continuous growing pair", false, []pole{{m, 1}}, false},
			{"continuous slow stable beside fast", false, []pole{{-m, 0}, {-4096, 0}}, true},
			{"discrete slow stable", true, []pole{{1 - m, 0}}, true},
			{"discrete slow unstable", true, []pole{{1 + m, 0}}, false},
			{"discrete pair inside", true, []pole{{0.5, 0.5}}, true},
		} {
			t.Run(tc.name, func(t *testing.T) {
				A, ok := unimodularSimilar(build(n, tc.disc, tc.poles), n)
				if !ok {
					t.Fatalf("n=%d: similarity not exact in float64", n)
				}
				dt := 0.0
				if tc.disc {
					dt = 0.1
				}
				sys, err := New(A, mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil), mat.NewDense(1, 1, nil), dt)
				if err != nil {
					t.Fatal(err)
				}
				st, err := sys.IsStable()
				if err != nil {
					t.Fatal(err)
				}
				if st != tc.stable {
					t.Fatalf("n=%d: IsStable = %v, want %v", n, st, tc.stable)
				}
			})
		}
	}
	for _, tc := range []struct {
		name   string
		disc   bool
		p      float64
		stable bool
	}{
		{"continuous just clear of the band", false, -math.Ldexp(1, -33), true},
		{"continuous inside the band", false, -math.Ldexp(1, -36), false},
		{"discrete just clear of the band", true, 1 - math.Ldexp(1, -33), true},
		{"discrete inside the band", true, 1 - math.Ldexp(1, -36), false},
	} {
		t.Run(tc.name, func(t *testing.T) {
			o := -0.75
			if tc.disc {
				o = 0.25
			}
			T := []float64{
				o, 0.5, 0, 0,
				-0.5, o, 0, 0,
				0, 0, tc.p, 0.125,
				0, 0, 0, o - 0.25,
			}
			perm := []int{2, 0, 3, 1}
			A := mat.NewDense(4, 4, nil)
			for i := range 4 {
				for j := range 4 {
					A.Set(perm[i], perm[j], T[i*4+j])
				}
			}
			dt := 0.0
			if tc.disc {
				dt = 1
			}
			sys, err := New(A, mat.NewDense(4, 1, nil), mat.NewDense(1, 4, nil), mat.NewDense(1, 1, nil), dt)
			if err != nil {
				t.Fatal(err)
			}
			if got := exactBandStable(A, !tc.disc); got != tc.stable {
				t.Fatalf("exact band verdict %v, want %v", got, tc.stable)
			}
			st, err := sys.IsStable()
			if err != nil {
				t.Fatal(err)
			}
			if st != tc.stable {
				t.Fatalf("IsStable = %v, want %v", st, tc.stable)
			}
		})
	}
}

func gradedSimilar(a *mat.Dense, exps []int) *mat.Dense {
	n, _ := a.Dims()
	out := mat.NewDense(n, n, nil)
	for i := range n {
		for j := range n {
			out.Set(i, j, math.Ldexp(a.At(i, j), exps[i]-exps[j]))
		}
	}
	return out
}

// GEFTJ6: a power-of-two graded similarity keeps the poles exactly. A clear
// model keeps its exact verdict; a model within rounding of the band keeps
// a band-edge witness in the balanced form.
func TestIsStable_BadlyScaledMatchesBalancedForm(t *testing.T) {
	for _, tc := range []struct {
		name  string
		lit   roundingLit
		clear bool
	}{
		{"clear", roundingClear, true},
		{"within rounding", roundingClusterUnstable, false},
	} {
		n, _ := tc.lit.a.Dims()
		for _, g := range []int{10, 20, 40} {
			exps := make([]int, n)
			for i := range exps {
				exps[i] = (i*g)%(2*g+1) - g
			}
			A := gradedSimilar(tc.lit.a, exps)
			sys, err := New(A, mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil), mat.NewDense(1, 1, nil), 0)
			if err != nil {
				t.Fatal(err)
			}
			st, err := sys.IsStable()
			if err != nil {
				t.Fatal(err)
			}
			if tc.clear {
				if want := exactBandStable(A, true); st != want {
					t.Fatalf("%s, grading ±%d: IsStable = %v, exact %v", tc.name, g, st, want)
				}
				continue
			}
			if d := bandEdgeDistance(A, true); d > float64(n+5) {
				t.Fatalf("%s, grading ±%d: IsStable = %v; band-edge witness %.3g·ε‖M‖_F, want ≤ %d", tc.name, g, st, d, n+5)
			}
		}
	}
}

// GEFTJ6 (ruling 5): Norm(sys, 2) is +Inf with nil exactly when IsStable is
// false on a strictly proper model, including a model within rounding of
// the band.
func TestNorm_H2InfiniteFollowsIsStable(t *testing.T) {
	for _, tc := range []struct {
		name string
		lit  roundingLit
	}{
		{"clear", roundingClear},
		{"within rounding", roundingClusterStable},
		{"slow pole beside fast pole", roundingLit{roundingSlowPoleFastPole.a, roundingSlowPoleFastPole.b, roundingSlowPoleFastPole.c, mat.NewDense(1, 1, nil)}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys := tc.lit.sys(t)
			st, err := sys.IsStable()
			if err != nil {
				t.Fatal(err)
			}
			v, err := Norm(sys, 2)
			if !st {
				if err != nil || !math.IsInf(v, 1) {
					t.Fatalf("IsStable false: Norm(sys, 2) = %v, %v, want +Inf, nil", v, err)
				}
				return
			}
			if err == nil && math.IsInf(v, 0) {
				t.Fatalf("IsStable true: Norm(sys, 2) = %v with nil error, want finite or an error", v)
			}
		})
	}
}
