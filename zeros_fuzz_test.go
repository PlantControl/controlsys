package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// FuzzZeros: every invariant zero returned by Zeros / ZerosDetail / Pzmap
// makes the I/O-scaled Rosenbrock pencil [zI−A, −βB; γC, γβD] numerically
// rank deficient (backward-error residual, insensitive to zero
// conditioning). When the pencil is clearly of full normal rank,
// ZerosDetail.Rank is min(m, p), and for square models the number of zeros
// inside a root-free annulus radius R equals the exact count of roots of
// det P(s) (big.Rat interpolation, Schur–Cohn) inside R; zeros near
// infinity are excluded by R. I/O delays never change the result.
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
		if m != p || n == 0 {
			return
		}
		dp := fzTrimRat(fzPencilDetRat(n, m, s.A, s.B, s.C, s.D))
		if len(dp) == 0 {
			t.Fatalf("exact det P(s) ≡ 0 but pencil σ = %v\n%s", sv, s)
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
			if want := deg - hi; inside != want {
				t.Fatalf("%d zeros inside |s| < %g, exact det P(s) has %d (degree %d)\nzeros=%v\n%s", inside, R, want, deg, zeros, s)
			}
			return
		}
		fzSkip(t, "no root-free annulus for the zero count")
	})
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
