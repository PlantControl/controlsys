package controlsys

import (
	"math"
	"testing"
)

// FuzzIsStableDelayFree: IsStable of a finite-dimensional model (no internal
// delays) matches the sign of its intended poles whenever they are clear of
// the boundary, I/O delays never change the verdict, and the exact Routh /
// Schur–Cohn count on the float64 A agrees with the intended count.
func FuzzIsStableDelayFree(f *testing.F) {
	for _, s := range fzIsStableSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		st, err := sys.IsStable()
		fzContract(t, s, "IsStable", st || err == nil, err)
		fzErr(t, s, "IsStable", err)
		dsys := s.System(t, fzSysOpts{Delays: true})
		dst, err := dsys.IsStable()
		fzContract(t, s, "IsStable(delayed)", dst || err == nil, err)
		fzErr(t, s, "IsStable(delayed)", err)
		if dst != st {
			t.Fatalf("IsStable with I/O delays = %v, delay-free %v\n%s", dst, st, s)
		}
		if !s.Decidable() {
			if s.N > 0 && fzSurelyNotStable(s) {
				if st {
					t.Fatalf("IsStable = true, but the exact float64 A has a root on or beyond the library's boundary band\n%s", s)
				}
				return
			}
			fzSkip(t, "verdict unresolvable: pole within error band of stability boundary")
		}
		want := s.Unstable() == 0
		if s.N > 0 {
			cp := fzCharPolyExact(s.A, s.N)
			var k int
			var ok bool
			if s.Discrete {
				k, ok = fzSchurCohnOutside(cp, 1)
			} else {
				k, ok = fzRouthRight(cp, 0)
			}
			if !ok {
				fzSkip(t, "degenerate Routh table")
			}
			if k != s.Unstable() {
				t.Fatalf("oracle: exact count %d, intended %d (EigErr bound unsound)\n%s", k, s.Unstable(), s)
			}
		}
		if st != want {
			t.Fatalf("IsStable = %v, want %v\n%s", st, want, s)
		}
	})
}

// fzSurelyNotStable reports whether the exact float64 A has a root that any
// backward-stable eigensolver must place on or outside the library's
// 1e-10·max(1,|p|) boundary band: a root right of (outside) the line
// (circle) shifted 1e-10 − r into the stable region, r = 2n·EigErr bounding
// computed-vs-exact distance within a Bauer–Fike disk cluster. A Hurwitz
// polynomial has a positive Routh column, so a degenerate table also means a
// root on or beyond the line.
func fzSurelyNotStable(s *fzSpec) bool {
	const band = 1e-10
	r := 2 * float64(s.N) * s.EigErr()
	if r >= band/2 {
		return false
	}
	cp := fzCharPolyExact(s.A, s.N)
	var k int
	var ok bool
	if s.Discrete {
		k, ok = fzSchurCohnOutside(cp, 1-band+r)
	} else {
		k, ok = fzRouthRight(cp, r-band)
	}
	return !ok || k > 0
}

// fzIsStableSeeds: regressions from ss_test.go / stabsep_test.go
// (boundary oscillator, -1e-12 pole, axis pair) and the clustered lightly
// damped pairs of TestOracleRHPRootsClusteredAxisPoles, plus MIMO,
// discrete, integrator and ill-conditioned realizations.
func fzIsStableSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0] = 2, 2.5 }),
		with(func(s *fzSeed) { s.N, s.Damp[0] = 2, -2.5 }),
		with(func(s *fzSeed) {
			s.N, s.Mag[0], s.Damp[0], s.Damp[1], s.Shear, s.Rot = 3, math.Log10(0.5), 13, 1, math.Log10(0.3), 0.4
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.Mag[0], s.Damp[0], s.Damp[1], s.Shear, s.Rot = 3, math.Log10(0.5), 13, 1, math.Log10(0.3), 0.4
		}),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0] = 1, -12, 1 }),
		with(func(s *fzSeed) { s.N, s.Damp[0] = 2, 13 }),
		with(func(s *fzSeed) { s.N, s.Damp[0], s.Rot = 2, 10, 0.3 }),
		with(func(s *fzSeed) { s.N, s.Damp[0], s.Rot = 2, -10, 0.3 }),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 5, 2, 2
			s.Mag = [3]float64{math.Log10(0.6914), math.Log10(0.6915), math.Log10(8174)}
			s.Damp = [3]float64{6, 6.3, 1}
			s.Shear, s.Rot, s.Phase = -1, 0.7, 0.2
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 6, 3, 3
			s.Mag, s.Damp = [3]float64{-1, 0.5, 2}, [3]float64{3.5, -4, 0}
			s.Shear, s.Rot, s.Gain, s.DScal = -0.3, 1.1, 8, 0.5
			s.In, s.Out, s.IO = 0, -0.5, 0.2
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.05)
			s.N, s.M, s.P = 6, 3, 2
			s.Mag, s.Damp = [3]float64{0, 1, -0.5}, [3]float64{4, 3.2, -0.5}
			s.Shear, s.Rot = -0.5, -2
			s.In, s.Out, s.IO = 0.5, 0, 0.7
		}),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0], s.Damp[1] = 3, -40, 1, 1 }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp = 4, [3]float64{3, -3}, [3]float64{0, -2} }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Shear, s.Rot = 6, [3]float64{0, 0.3, -0.2}, 0.5, 2.5 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal = 0, 3, 2, 4 }),
	}
}
