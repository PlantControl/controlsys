package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// FuzzPoles: Poles of a delay-free model match the intended eigenvalues by
// Bauer–Fike cluster counting (fzMatchEig, radius EigErr), I/O delays leave
// them unchanged, and Damp lists the same poles in order with natural
// frequency, damping and time constant within the propagated error of the
// intended continuous-time values (log(z)/Dt for discrete models).
func FuzzPoles(f *testing.F) {
	for _, s := range fzPolesSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		poles, err := sys.Poles()
		fzErr(t, s, "Poles", err)
		if len(poles) != s.N {
			t.Fatalf("Poles returned %d poles, want %d\n%s", len(poles), s.N, s)
		}
		dpoles, err := s.System(t, fzSysOpts{Delays: true}).Poles()
		fzErr(t, s, "Poles(delayed)", err)
		if !slices.Equal(dpoles, poles) {
			t.Fatalf("Poles with I/O delays = %v, delay-free %v\n%s", dpoles, poles, s)
		}
		r := s.EigErr()
		comp, wantComp, msg := fzMatchEig(poles, s.Eig, r)
		if msg != "" {
			t.Fatalf("Poles = %v: %s\n%s", poles, msg, s)
		}
		damp, err := Damp(sys)
		fzErr(t, s, "Damp", err)
		if len(damp) != len(poles) {
			t.Fatalf("Damp returned %d entries for %d poles\n%s", len(damp), len(poles), s)
		}
		for i, d := range damp {
			if d.Pole != poles[i] {
				t.Fatalf("Damp[%d].Pole = %v, Poles[%d] = %v\n%s", i, d.Pole, i, poles[i], s)
			}
			if math.IsNaN(d.Wn) || math.IsNaN(d.Zeta) || math.IsNaN(d.Tau) {
				t.Fatalf("Damp[%d] = %+v has NaN\n%s", i, d, s)
			}
			fzCheckDamp(t, s, i, d, comp[i], wantComp, r)
		}
	})
}

// fzCheckDamp bounds Damp entry d by the intended values of its Bauer–Fike
// cluster: |δs| ≤ δ for continuous models, δ = r/((|z|−r)·Dt) after
// s = log(z)/Dt; ζ and τ are checked only when every |s| (|Re s|) in the
// cluster exceeds 4δ. Wn, ζ and τ are conjugation-invariant, so the log
// branch cut does not matter.
func fzCheckDamp(t *testing.T, s *fzSpec, i int, d DampInfo, comp int, wantComp []int, r float64) {
	t.Helper()
	var ss []complex128
	delta := r
	for j, w := range s.Eig {
		if wantComp[j] != comp {
			continue
		}
		if s.Discrete {
			if cmplx.Abs(w) <= 2*r {
				return
			}
			delta = math.Max(delta, r/((cmplx.Abs(w)-r)*s.Dt))
			w = cmplx.Log(w) / complex(s.Dt, 0)
		}
		ss = append(ss, w)
	}
	slack := func(x float64) float64 { return 1e-12 * math.Abs(x) }
	inRange := func(name string, got float64, f func(complex128) float64, e func(complex128) float64) {
		lo, hi := math.Inf(1), math.Inf(-1)
		for _, w := range ss {
			v, ev := f(w), e(w)
			lo, hi = math.Min(lo, v-ev), math.Max(hi, v+ev)
		}
		if got < lo-slack(lo) || got > hi+slack(hi) {
			t.Fatalf("Damp[%d].%s = %v outside intended [%v, %v] (pole %v, δ=%g)\n%s", i, name, got, lo, hi, d.Pole, delta, s)
		}
	}
	inRange("Wn", d.Wn, cmplx.Abs, func(complex128) float64 { return delta })
	for _, w := range ss {
		if cmplx.Abs(w) <= 4*delta {
			return
		}
	}
	inRange("Zeta", d.Zeta, func(w complex128) float64 { return -real(w) / cmplx.Abs(w) },
		func(w complex128) float64 { return 2 * delta / (cmplx.Abs(w) - delta) })
	for _, w := range ss {
		if math.Abs(real(w)) <= 4*delta {
			return
		}
	}
	inRange("Tau", d.Tau, func(w complex128) float64 { return -1 / real(w) },
		func(w complex128) float64 { return delta / ((math.Abs(real(w)) - delta) * math.Abs(real(w))) })
}

// fzPolesSeeds: the IsStable regressions plus integrators (exact and
// rotated), a discrete model with poles near z = 0 and near the branch cut,
// heavy damping, high Dt aliasing and repeated poles.
func fzPolesSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return append(fzIsStableSeeds(),
		with(func(s *fzSeed) { s.N, s.Mag[0], s.Rot = 1, -40, 0 }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp, s.Rot = 3, [3]float64{-40, 0.5}, [3]float64{1, 2}, 0 }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp, s.In, s.IO = 4, [3]float64{-40, 1}, [3]float64{-1, 0.5}, 0.3, 0 }),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, 0
			s.N, s.Mag, s.Damp = 5, [3]float64{0.49, 1.5, -1}, [3]float64{6, 2, 1}
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, 1.5
			s.N, s.Mag, s.Damp = 4, [3]float64{-1.2, 0.5}, [3]float64{3, 1.9}
		}),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp = 6, [3]float64{0, 0, 0}, [3]float64{4, 4, 4} }),
		with(func(s *fzSeed) { s.N, s.Mag, s.Damp, s.Shear = 4, [3]float64{3.5, -2}, [3]float64{-13, 0.3}, -1 }),
	)
}
