package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"slices"
	"testing"
)

// FuzzZeros: every invariant zero returned by Zeros / ZerosDetail / Pzmap
// makes the I/O-scaled Rosenbrock pencil [zI−A, −βB; γC, γβD] numerically
// rank deficient (backward-error residual, insensitive to zero
// conditioning). When the pencil is clearly of full normal rank,
// ZerosDetail.Rank is min(m, p), and the number of zeros inside a
// root-free annulus radius R equals the exact count of roots inside R
// (Schur–Cohn) of det P(s) for square models, of the gcd of the maximal
// minors of P(s) for non-square ones (both big.Rat interpolation); zeros
// near infinity are excluded by R. Non-square zeros are non-generic, so
// rounding-level near-structure gives zeros the exact pencil lacks (the
// backward-error check above bounds them): there the count is only a
// lower bound. I/O delays never change the result.
func FuzzZeros(f *testing.F) {
	for _, s := range fzZerosSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		zeros, err := sys.Zeros()
		fzErr(t, s, "Zeros", err)
		det, err := sys.ZerosDetail()
		fzContract(t, s, "ZerosDetail", det != nil, err)
		fzErr(t, s, "ZerosDetail", err)
		if !slices.Equal(det.Zeros, zeros) {
			t.Fatalf("ZerosDetail.Zeros = %v, Zeros = %v\n%s", det.Zeros, zeros, s)
		}
		dz, err := s.System(t, fzSysOpts{Delays: true}).Zeros()
		fzErr(t, s, "Zeros(delayed)", err)
		if !slices.Equal(dz, zeros) {
			t.Fatalf("Zeros with I/O delays = %v, delay-free %v\n%s", dz, zeros, s)
		}
		poles, err := sys.Poles()
		fzErr(t, s, "Poles", err)
		pz, err := Pzmap(sys)
		fzContract(t, s, "Pzmap", pz != nil, err)
		fzErr(t, s, "Pzmap", err)
		if !slices.Equal(pz.Poles, poles) || !slices.Equal(pz.Zeros, zeros) {
			t.Fatalf("Pzmap = %v / %v, Poles/Zeros = %v / %v\n%s", pz.Poles, pz.Zeros, poles, zeros, s)
		}
		if len(zeros) > s.N {
			t.Fatalf("%d zeros for order %d\n%s", len(zeros), s.N, s)
		}
		n, m, p := s.N, s.M, s.P
		nA := math.Max(fzFrob(s.A), 1)
		beta, gamma := 1.0, 1.0
		if nb := fzFrob(s.B); nb > 0 {
			beta = nA / nb
		}
		cd := fzFrob(s.C)
		cd = math.Hypot(cd, beta*fzFrob(s.D))
		if cd > 0 {
			gamma = nA / cd
		}
		if beta*gamma > 1e8 || beta*gamma < 1e-8 || beta > 1e8 || beta < 1e-8 || gamma > 1e8 || gamma < 1e-8 {
			beta, gamma = 1, 1
		}
		scale := fzFrob(fzComplexPencilFloat(s, 0, beta, gamma))
		const tol = 1e-8
		for _, z := range zeros {
			if cmplx.IsNaN(z) || cmplx.IsInf(z) {
				t.Fatalf("non-finite zero %v in %v\n%s", z, zeros, s)
			}
			sv := fzSingularValues(fzComplexPencil(s, z, beta, gamma), n+p, n+m)
			if res := sv[len(sv)-1]; res > tol*(scale+cmplx.Abs(z)*math.Sqrt(float64(n))) {
				t.Fatalf("zero %v: pencil σmin = %g, scale %g (not a zero)\nzeros=%v\n%s", z, res, scale, zeros, s)
			}
		}
		z0 := complex(0.31, 0.83) * complex(math.Max(1, s.NormJ), 0)
		sv := fzSingularValues(fzComplexPencil(s, z0, beta, gamma), n+p, n+m)
		if len(sv) == 0 || sv[len(sv)-1] <= 1e-6*sv[0] {
			fzSkip(t, "normal rank not clearly full")
		}
		if det.Rank != min(m, p) {
			t.Fatalf("ZerosDetail.Rank = %d, want %d (pencil σ = %v)\n%s", det.Rank, min(m, p), sv, s)
		}
		if n == 0 {
			return
		}
		var dp []*big.Rat
		if m == p {
			dp = fzTrimRat(fzPencilDetRat(n, m, s.A, s.B, s.C, s.D))
		} else {
			dp = fzZeroPolyRat(n, m, p, s.A, s.B, s.C, s.D)
		}
		if len(dp) == 0 {
			t.Fatalf("exact zero polynomial ≡ 0 but pencil σ = %v\n%s", sv, s)
		}
		deg := len(dp) - 1
		for k := 2; k >= -3; k-- {
			R := nA * math.Pow(10, float64(k))
			lo, ok1 := fzSchurCohnOutside(dp, R/2)
			hi, ok2 := fzSchurCohnOutside(dp, 2*R)
			if !ok1 || !ok2 || lo != hi {
				continue
			}
			inside, clear := 0, true
			for _, z := range zeros {
				a := cmplx.Abs(z)
				if a >= R/2 && a <= 2*R {
					clear = false
					break
				}
				if a < R {
					inside++
				}
			}
			if !clear {
				continue
			}
			if want := deg - hi; inside != want && (m == p || inside < want) {
				t.Fatalf("%d zeros inside |s| < %g, exact det P(s) has %d (degree %d)\nzeros=%v\n%s", inside, R, want, deg, zeros, s)
			}
			return
		}
		fzSkip(t, "no root-free annulus for the zero count")
	})
}

// fzZeroPolyRat returns the monic gcd of all maximal minors of the
// Rosenbrock pencil of a non-square model (exact, big.Rat): its roots are
// the invariant zeros when the normal rank is n+min(m, p). Empty when every
// maximal minor vanishes identically.
func fzZeroPolyRat(n, m, p int, a, b, c, d []float64) []*big.Rat {
	k := n + min(m, p)
	rows, cols := fzCombos(n+p, k), fzCombos(n+m, k)
	if p > m {
		cols = cols[:1]
	} else {
		rows = rows[:1]
	}
	var g []*big.Rat
	for _, rs := range rows {
		for _, cs := range cols {
			g = fzPolyGcdRat(g, fzMinorPolyRat(n, m, p, a, b, c, d, rs, cs))
			if len(g) == 1 {
				return g
			}
		}
	}
	return g
}

// fzMinorPolyRat returns the determinant of the rows rs, columns cs of
// [sI−A, −B; C, D], descending, by Newton interpolation at s = 0..n.
func fzMinorPolyRat(n, m, p int, a, b, c, d []float64, rs, cs []int) []*big.Rat {
	k, w := len(rs), n+m
	y := make([]*big.Rat, n+1)
	for x := range y {
		P := fzPencilRat(n, m, p, a, b, c, d, big.NewRat(int64(x), 1))
		S := make([]*big.Rat, k*k)
		for i, r := range rs {
			for j, cc := range cs {
				S[i*k+j] = P[r*w+cc]
			}
		}
		_, y[x] = fzRatElim(S, k, k)
	}
	for j := 1; j <= n; j++ {
		for x := n; x >= j; x-- {
			y[x] = new(big.Rat).Quo(new(big.Rat).Sub(y[x], y[x-1]), big.NewRat(int64(j), 1))
		}
	}
	poly := []*big.Rat{new(big.Rat).Set(y[n])}
	for x := n - 1; x >= 0; x-- {
		poly = fzRatPolyMul(poly, []*big.Rat{big.NewRat(1, 1), big.NewRat(-int64(x), 1)})
		poly[len(poly)-1].Add(poly[len(poly)-1], y[x])
	}
	return fzTrimRat(poly)
}

// fzPolyGcdRat returns the monic gcd of descending polynomials a and b.
func fzPolyGcdRat(a, b []*big.Rat) []*big.Rat {
	if len(a) == 0 {
		return b
	}
	var t big.Rat
	for len(b) > 0 {
		r := make([]*big.Rat, len(a))
		for i := range a {
			r[i] = new(big.Rat).Set(a[i])
		}
		for len(r) >= len(b) {
			f := new(big.Rat).Quo(r[0], b[0])
			for i := range b {
				r[i].Sub(r[i], t.Mul(f, b[i]))
			}
			r = fzTrimRat(r[1:])
		}
		a, b = b, r
	}
	out := make([]*big.Rat, len(a))
	for i := range a {
		out[i] = new(big.Rat).Quo(a[i], a[0])
	}
	return out
}

func fzCombos(n, k int) [][]int {
	var out [][]int
	cur := make([]int, 0, k)
	var rec func(s int)
	rec = func(s int) {
		if len(cur) == k {
			out = append(out, slices.Clone(cur))
			return
		}
		for i := s; i < n; i++ {
			cur = append(cur, i)
			rec(i + 1)
			cur = cur[:len(cur)-1]
		}
	}
	rec(0)
	return out
}

func fzFrob(x []float64) float64 {
	var s float64
	for _, v := range x {
		s = math.Hypot(s, v)
	}
	return s
}

func fzComplexPencilFloat(s *fzSpec, z complex128, beta, gamma float64) []float64 {
	P := fzComplexPencil(s, z, beta, gamma)
	out := make([]float64, len(P))
	for i, v := range P {
		out[i] = cmplx.Abs(v)
	}
	return out
}

// fzZerosSeeds: SISO, square and non-square MIMO, D = 0 (zeros at
// infinity), nearly singular D, discrete, static gains, integrators and
// I/O delays, and a pole at z = 2.7e43 whose ‖A‖ swamps B and C (residual
// falls back to the unscaled pencil).
func fzZerosSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37, DScal: 1}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Damp[0] = 2, 1, 1, 2.5 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Damp[0], s.DScal = 3, 1, 1, 1, 0 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag, s.DScal, s.Shear = 5, 1, 1, [3]float64{0.2, -1, 1}, 0.3, -0.5 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P, s.Mag, s.Damp, s.DScal = 4, 2, 2, [3]float64{0, 0.7}, [3]float64{1, -3}, 2
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag, s.DScal, s.Rot = 6, 2, 2, [3]float64{-0.5, 0.4, 1.2}, 0, 1.3 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal, s.Phase = 4, 2, 2, 1e-6, 0.4 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal = 3, 3, 3, 1 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal = 4, 2, 1, 0 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal = 4, 1, 3, 0.5 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.M, s.P, s.Mag, s.Damp, s.DScal = 4, 2, 2, [3]float64{0.5, 1}, [3]float64{2, 1}, 0.7
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -2
			s.N, s.M, s.P, s.Mag, s.DScal, s.Gain = 3, 1, 1, [3]float64{1, 2}, 0, 5
		}),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal = 0, 2, 2, 3 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag[0], s.Rot, s.DScal = 2, 1, 1, -40, 0, 0 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.In, s.Out, s.IO = 3, 2, 2, 0, 0.5, 0.2 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Gain, s.DScal = 4, 2, 2, -9, 300 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.M, s.P, s.DScal = true, 86, 3, 3, 1, 0
			s.Mag, s.Damp, s.Gain, s.Shear, s.Phase = [3]float64{60, 0, -44}, [3]float64{-21, -72, 0}, 98, -1.4285714285714286, 47
			s.IO = -9
		}),
	}
}
