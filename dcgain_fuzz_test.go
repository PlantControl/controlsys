package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// FuzzDCGain: DCGain of a delay-free model equals the exact rational
// C(x0·I − A)⁻¹B + D (x0 = 0, or 1 for discrete) within a conditioning
// bound from the intended spectrum; when the float64 A has an exact pole at
// x0, every entry whose exact limit n_ij/det(sI−A) diverges is that signed
// infinity, as MATLAB dcgain; its sign is free when further intended
// poles lie within roundoff of x0 (a nearby model has them at x0 too). Poles within the eigenvalue error of x0, or
// an ill-conditioned solve, are skipped. I/O delays never change the gain.
func FuzzDCGain(f *testing.F) {
	for _, s := range fzDCGainSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		g, err := sys.DCGain()
		fzContract(t, s, "DCGain", g != nil, err)
		fzErr(t, s, "DCGain", err)
		n, m, p := s.N, s.M, s.P
		if r, c := g.Dims(); r != p || c != m {
			t.Fatalf("DCGain is %d×%d, want %d×%d\n%s", r, c, p, m, s)
		}
		dg, err := s.System(t, fzSysOpts{Delays: true}).DCGain()
		fzContract(t, s, "DCGain(delayed)", dg != nil, err)
		fzErr(t, s, "DCGain(delayed)", err)
		for i := range p {
			for j := range m {
				v := g.At(i, j)
				if math.IsNaN(v) {
					t.Fatalf("DCGain[%d,%d] = NaN\n%s", i, j, s)
				}
				if dv := dg.At(i, j); dv != v {
					t.Fatalf("DCGain[%d,%d] with I/O delays = %v, delay-free %v\n%s", i, j, dv, v, s)
				}
				if n == 0 && v != s.D[i*m+j] {
					t.Fatalf("static DCGain[%d,%d] = %v, want D = %v\n%s", i, j, v, s.D[i*m+j], s)
				}
			}
		}
		if n == 0 {
			return
		}
		x0f := 0.0
		if s.Discrete {
			x0f = 1
		}
		x0 := fzRat(x0f)
		cp := fzCharPolyExact(s.A, n)
		if sh := fzTaylorShift(cp, x0); sh[n].Sign() == 0 {
			ord := 0
			for k := n; k >= 0 && sh[k].Sign() == 0; k-- {
				ord++
			}
			band := math.Max(4*s.EigErr(), 1e-10*math.Max(1, s.NormJ))
			near := 0
			for _, e := range s.Eig {
				if cmplx.Abs(e-complex(x0f, 0)) <= band {
					near++
				}
			}
			signFree := near > ord
			finite := false
			for i := range p {
				for j := range m {
					bj := make([]float64, n)
					for k := range n {
						bj[k] = s.B[k*m+j]
					}
					num := fzPencilDetRat(n, 1, s.A, bj, s.C[i*n:(i+1)*n], []float64{s.D[i*m+j]})
					lim := fzLimitAt(num, cp, x0)
					if !math.IsInf(lim, 0) {
						finite = true
						continue
					}
					v := g.At(i, j)
					if signFree && math.IsInf(v, 0) {
						continue
					}
					if v != lim {
						t.Fatalf("DCGain[%d,%d] = %v, exact limit at the integrator %v\n%s", i, j, v, lim, s)
					}
				}
			}
			if finite {
				fzSkip(t, "integrator cancelled in an entry")
			}
			return
		}
		r := s.EigErr()
		minDist, maxDist := math.Inf(1), 0.0
		for _, e := range s.Eig {
			d := cmplx.Abs(e - complex(x0f, 0))
			minDist, maxDist = math.Min(minDist, d), math.Max(maxDist, d)
		}
		if minDist <= 4*r {
			fzSkip(t, "pole within eigenvalue error of DC")
		}
		const eps = 0x1p-52
		minv := s.CondT / (minDist - r)
		rel := 1e3 * float64(n) * eps * s.CondT * (maxDist + r) * minv
		if rel > 1e-3 {
			fzSkip(t, "ill-conditioned DC solve")
		}
		exact, ok := fzExactResponse(n, m, p, s.A, s.B, s.C, s.D, complex(x0f, 0))
		if !ok {
			t.Fatalf("oracle: det(x0·I − A) ≠ 0 exactly but the exact solve is singular\n%s", s)
		}
		var dmax float64
		for _, d := range s.D {
			dmax = math.Max(dmax, math.Abs(d))
		}
		bound := rel*fzFrob(s.C)*minv*fzFrob(s.B) + 1e3*eps*dmax
		for i := range p {
			for j := range m {
				want := real(exact[i*m+j])
				if v := g.At(i, j); !(math.Abs(v-want) <= bound+1e3*eps*math.Abs(want)) {
					t.Fatalf("DCGain[%d,%d] = %v, exact %v (bound %g)\n%s", i, j, v, want, bound, s)
				}
			}
		}
	})
}

// fzDCGainSeeds: exact integrators (unrotated J, single, double, discrete
// z = 1, mixed with stable modes), a rotated near-integrator, poles close
// to but clear of DC, MIMO, D = 0, large gain, static gains, and exact
// z = 1 integrators beside a pair 1.8e-15 from z = 1 (sign-free Inf).
func fzDCGainSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37, DScal: 0.5}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return []fzSeed{
		with(func(s *fzSeed) { s.N, s.Damp[0] = 2, 2.5 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag, s.Damp = 4, 2, 3, [3]float64{0.3, -0.5}, [3]float64{1, 2} }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Rot = 1, -40, 0 }),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Rot, s.Gain = 1, -40, 0, -3 }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp, s.Rot = 3, [3]float64{-40, 0.5}, [3]float64{1, 2}, 0 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag, s.Rot = 2, 2, 2, [3]float64{-40}, 0 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -1
			s.N, s.Mag, s.Damp, s.Rot = 3, [3]float64{-40, 0.2}, [3]float64{1, 1.5}, 0
		}),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Damp[0] = 2, -40, 1 }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp = 3, [3]float64{-6, 0}, [3]float64{1, 1} }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, -2
			s.N, s.M, s.P, s.Mag, s.Damp = 5, 2, 2, [3]float64{-1, 0.5, 1}, [3]float64{1, 2, -1}
		}),
		with(func(s *fzSeed) { s.N, s.Mag, s.DScal, s.Gain = 4, [3]float64{1, -1}, 0, 11 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.DScal = 0, 3, 2, 4 }),
		with(func(s *fzSeed) { s.N, s.M, s.P, s.Mag, s.In, s.Out, s.IO = 3, 2, 2, [3]float64{0.2, 0.4}, 0, 0.3, 0.1 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt, s.N, s.M, s.P = true, -44, 6, 3, 3
			s.Mag, s.Damp, s.Gain, s.Rot, s.Phase, s.IO = [3]float64{-40, -10.75}, [3]float64{34, 27.2}, 22, 0, 30.5, 88
		}),
	}
}
