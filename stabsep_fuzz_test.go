package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// FuzzStabsep: on decidable delay-free models Stabsep's Stable part is
// exactly stable (fzRequireStable), its Unstable part has every exact root
// on or outside the boundary, the orders split as the intended poles do, and
// D goes to Stable.
func FuzzStabsep(f *testing.F) {
	for _, s := range fzStabsepSeeds() {
		fzAddSeed(f, s)
	}
	fzFuzz(f, func(t *testing.T, s *fzSpec) {
		sys := s.System(t, fzSysOpts{})
		res, err := Stabsep(sys)
		fzContract(t, s, "Stabsep", res != nil, err)
		fzErr(t, s, "Stabsep", err, ErrSchurFailed)
		if res.Stable == nil || res.Unstable == nil {
			t.Fatalf("Stabsep: nil part %+v\n%s", res, s)
		}
		if !s.Decidable() {
			fzSkip(t, "pole within error band of stability boundary")
		}
		nu := s.Unstable()
		ns, _, _ := res.Stable.Dims()
		nun, _, _ := res.Unstable.Dims()
		if ns != s.N-nu || nun != nu {
			t.Fatalf("Stabsep orders stable=%d unstable=%d, want %d/%d\n%s", ns, nun, s.N-nu, nu, s)
		}
		fzRequireStable(t, s, "Stabsep.Stable", res.Stable)
		if nun > 0 {
			A := mat.DenseCopyOf(res.Unstable.A).RawMatrix().Data
			cp := fzCharPolyExact(A, nun)
			var k int
			var ok bool
			if s.Discrete {
				k, ok = fzSchurCohnOutside(cp, 1)
			} else {
				k, ok = fzRouthRight(cp, 0)
			}
			if ok && k != nun {
				t.Fatalf("Stabsep.Unstable has %d of %d exact roots outside the boundary\n%s", k, nun, s)
			}
		}
		for i := range s.P {
			for j := range s.M {
				if g, w := res.Stable.D.At(i, j), s.D[i*s.M+j]; g != w {
					t.Fatalf("Stabsep.Stable.D[%d,%d] = %v, want %v\n%s", i, j, g, w, s)
				}
				if g := res.Unstable.D.At(i, j); g != 0 {
					t.Fatalf("Stabsep.Unstable.D[%d,%d] = %v, want 0\n%s", i, j, g, s)
				}
			}
		}
	})
}

func fzStabsepSeeds() []fzSeed {
	off := fzSeed{In: fzNoDelay, Out: fzNoDelay, IO: fzNoDelay, Tau: fzNoDelay, Shear: -10, Rot: 0.37}
	with := func(f func(*fzSeed)) fzSeed {
		s := off
		f(&s)
		return s
	}
	return append(fzIsStableSeeds(),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 5, 2, 3
			s.Mag, s.Damp = [3]float64{0, 1, -0.5}, [3]float64{-1.5, 2, -3}
			s.Shear, s.Rot, s.DScal = -1, 0.9, 2
		}),
		with(func(s *fzSeed) {
			s.Discrete, s.LogDt = true, math.Log10(0.2)
			s.N, s.M, s.P = 6, 2, 2
			s.Mag, s.Damp = [3]float64{0.5, -0.3, 0.2}, [3]float64{-2, 3, 1}
			s.Shear, s.Rot = -0.8, 1.7
		}),
		with(func(s *fzSeed) {
			s.N, s.M, s.P = 4, 3, 1
			s.Mag, s.Damp = [3]float64{1, 2}, [3]float64{-0.5, -1}
			s.Rot, s.DScal = -2.2, -1
		}),
	)
}
