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
		fzDescriptorSepTolSum(t, s, "Stabsep", sys, func(d *System) (*System, *System, error) {
			r, err := Stabsep(d)
			if err != nil {
				return nil, nil, err
			}
			return r.Stable, r.Unstable, nil
		})
		res, err := Stabsep(sys)
		fzContract(t, s, "Stabsep", res != nil, err)
		fzErr(t, s, "Stabsep", err, ErrSchurFailed)
		if res.Stable == nil || res.Unstable == nil {
			t.Fatalf("Stabsep: nil part %+v\n%s", res, s)
		}
		fzSepTolSum(t, s, "Stabsep", sys, res.Stable, res.Unstable)
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

// fzSepTolSum: a nil Stabsep/Modsep split of an explicit model has G1 + G2
// within SepTol = 10 of the normwise rounding accuracy of G (PZET7B): the
// exact difference at the test points is at most 10·sepTolUnit.
func fzSepTolSum(t *testing.T, s *fzSpec, op string, sys, g1, g2 *System) {
	t.Helper()
	if sys.E != nil || s.N == 0 {
		return
	}
	m1, m2 := fzSysMats(g1), fzSysMats(g2)
	n1, _, _ := g1.Dims()
	n2, _, _ := g2.Dims()
	for _, x := range fzTestPoints(s) {
		g0, ok0 := fzExactResponse(s.N, s.M, s.P, s.A, s.B, s.C, s.D, x)
		ga, ok1 := fzExactResponse(n1, s.M, s.P, m1[0], m1[1], m1[2], m1[3], x)
		gb, ok2 := fzExactResponse(n2, s.M, s.P, m2[0], m2[1], m2[2], m2[3], x)
		if !ok0 || !ok1 || !ok2 {
			continue
		}
		var d l2
		for k := range g0 {
			d.addC(g0[k] - ga[k] - gb[k])
		}
		unit, ok := sepTolUnitOK(sys, x, 1e8)
		if !ok {
			continue
		}
		if bar := 10 * unit; d.norm() > bar {
			t.Fatalf("%s: ‖G − G1 − G2‖_F at %v = %.3g > SepTol bar %.3g\n%s", op, x, d.norm(), bar, s)
		}
	}
}

// fzDescriptorSepTolSum: the model with one algebraic state appended
// (E = diag(I, 0), A = diag(A, 1), a row and column of ones in B and C,
// split by the generalized Schur form) either errors or splits within the
// descriptor SepTol bar at the test points (EUNKOP).
func fzDescriptorSepTolSum(t *testing.T, s *fzSpec, op string, sys *System, split func(*System) (*System, *System, error)) {
	t.Helper()
	if sys.E != nil || s.N == 0 || sys.HasDelay() {
		return
	}
	n, m, p := sys.Dims()
	A := mat.NewDense(n+1, n+1, nil)
	E := mat.NewDense(n+1, n+1, nil)
	B := mat.NewDense(n+1, m, nil)
	C := mat.NewDense(p, n+1, nil)
	A.Slice(0, n, 0, n).(*mat.Dense).Copy(sys.A)
	B.Slice(0, n, 0, m).(*mat.Dense).Copy(sys.B)
	C.Slice(0, p, 0, n).(*mat.Dense).Copy(sys.C)
	for i := range n {
		E.Set(i, i, 1)
	}
	A.Set(n, n, 1)
	for j := range m {
		B.Set(n, j, 1)
	}
	for i := range p {
		C.Set(i, n, 1)
	}
	d, err := NewDescriptor(A, B, C, mat.DenseCopyOf(sys.D), E, sys.Dt)
	if err != nil {
		return
	}
	g1, g2, err := split(d)
	if err != nil {
		return
	}
	for _, x := range fzTestPoints(s) {
		unit, ok := descTolUnitOK(d, x)
		if !ok {
			continue
		}
		g, a, b := sepEval320(t, d, x), sepEval320(t, g1, x), sepEval320(t, g2, x)
		var e l2
		for k := range g {
			e.addC(g[k] - a[k] - b[k])
		}
		if bar := decouplingSepTol * unit; e.norm() > bar {
			t.Fatalf("%s descriptor: ‖G − G1 − G2‖_F at %v = %.3g > SepTol bar %.3g\n%s", op, x, e.norm(), bar, s)
		}
	}
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
