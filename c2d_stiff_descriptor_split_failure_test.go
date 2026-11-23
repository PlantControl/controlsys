package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// The undetermined pencil of 7AAOBX (c = 3e10) fails its stiff split on
// rounding for some seeds; such a seed returns ErrSchurFailed with a nil
// model from every C2D method and the time responses, never the whole fold,
// while TransferFunction still returns its judged result (SCPYGG).
func TestC2DStiffDescriptorSplitFailureErrors(t *testing.T) {
	failed := 0
	for seed := range uint64(60) {
		sys := stiffRotatedDescriptor(t, rand.New(rand.NewPCG(seed, 777)),
			[]float64{-1, 3e10, 1, 0, -2, 1, 0, 0, -1},
			[]float64{1, 1, 1e-13}, []float64{1, 1.3, 1.6}, []float64{1, .8, .6})
		sp, ok := stiffExplicitSplit(sys)
		if !ok {
			t.Fatalf("seed %d: not on the stiff route", seed)
		}
		if sp.err == nil {
			continue
		}
		failed++
		if !errors.Is(sp.err, ErrSchurFailed) {
			t.Errorf("seed %d: split error %v does not wrap ErrSchurFailed", seed, sp.err)
		}
		for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodImpulse, C2DMethodMatched} {
			if d, err := sys.C2D(0.1, C2DOptions{Method: m}); !errors.Is(err, ErrSchurFailed) || d != nil {
				t.Errorf("seed %d: C2D %s = %v, %v; want nil, ErrSchurFailed", seed, m, d, err)
			}
		}
		if _, _, err := sys.C2DFit(0.1, C2DOptions{Method: C2DMethodLeastSquares}); !errors.Is(err, ErrSchurFailed) {
			t.Errorf("seed %d: C2DFit: %v; want ErrSchurFailed", seed, err)
		}
		if r, err := Step(sys, 1); !errors.Is(err, ErrSchurFailed) || r != nil {
			t.Errorf("seed %d: Step = %v, %v; want nil, ErrSchurFailed", seed, r, err)
		}
		if r, err := Impulse(sys, 1); !errors.Is(err, ErrSchurFailed) || r != nil {
			t.Errorf("seed %d: Impulse = %v, %v; want nil, ErrSchurFailed", seed, r, err)
		}
		if r, err := Initial(sys, mat.NewVecDense(3, []float64{1, 0.5, 0.25}), 1); !errors.Is(err, ErrSchurFailed) || r != nil {
			t.Errorf("seed %d: Initial = %v, %v; want nil, ErrSchurFailed", seed, r, err)
		}
		grid := []float64{0, 0.1, 0.2, 0.3}
		if r, err := Lsim(sys, mat.NewDense(4, 1, []float64{1, 1, 1, 1}), grid, nil); !errors.Is(err, ErrSchurFailed) || r != nil {
			t.Errorf("seed %d: Lsim = %v, %v; want nil, ErrSchurFailed", seed, r, err)
		}
		if _, err := sys.TransferFunction(nil); errors.Is(err, ErrSchurFailed) {
			t.Errorf("seed %d: TransferFunction returned the split error %v", seed, err)
		}
	}
	if failed == 0 {
		t.Fatal("no seed fails its split; the sweep no longer reaches the case")
	}
}

// stiffDownMoveDescriptor has moduli 1, 2.5, 2e3, 5e3 and 1.5e6: the widest
// gap (2.5, 2e3) lies below the nominal cut, so the cut moves down and the
// mode 2e3 goes to the fast part (SCPYGG).
func stiffDownMoveDescriptor(t *testing.T) *System {
	t.Helper()
	return stiffRotatedDescriptor(t, rand.New(rand.NewPCG(11, 0xd0)),
		[]float64{-1, 0.4, 0.2, -0.3, 0.1, 0, -2.5, 0.3, 0.2, -0.2, 0, 0, -1, 0.5, 0.3, 0, 0, 0, -1, 0.2, 0, 0, 0, 0, -1},
		[]float64{1, 1, 1 / 2e3, 1 / 5e3, 1 / 1.5e6}, []float64{1, -0.5, 0.8, 1.2, 0.6}, []float64{1, 0.7, -0.6, 0.9, 0.4})
}

// Initial takes each fast part's share of x0 through A_part⁻¹ (freePartShare),
// which needs the fast parts to hold only large poles; with the cut moved
// below the nominal one at cond(E) ≥ 1e6 the free response stays exact
// within 1e-7, as do Step and C2D ZOH DC (SCPYGG).
func TestInitialStiffDescriptorCutBelowNominal(t *testing.T) {
	sys := stiffDownMoveDescriptor(t)
	var lu mat.LU
	lu.Factorize(sys.E)
	if lu.Cond() < foldAccurateECond {
		t.Fatalf("cond(E) %g below foldAccurateECond", lu.Cond())
	}
	eq := equilibrateDescriptor(sys)
	n, _, _ := eq.Dims()
	ar, ai, be, ok := qzEigenvalues(n, eq.A.RawMatrix().Data, eq.A.RawMatrix().Stride, eq.E.RawMatrix().Data, eq.E.RawMatrix().Stride)
	if !ok {
		t.Fatal("qzEigenvalues failed")
	}
	var mods []float64
	for i := range ar {
		mods = append(mods, math.Hypot(ar[i], ai[i])/math.Abs(be[i]))
	}
	nominal := mat.Norm(eq.A, 2) / (mat.Norm(eq.E, 2) * qzLargeRatio)
	cut := stiffCut(mods, nominal)
	if !(cut < 2e3 && nominal > 2e3) {
		t.Fatalf("cut %g, nominal %g; want the cut moved below the mode 2e3", cut, nominal)
	}
	sp, ok := stiffExplicitSplit(sys)
	if !ok || sp.err != nil || len(sp.parts) < 2 {
		t.Fatalf("stiffExplicitSplit: ok %v, err %v, %d parts", ok, sp.err, len(sp.parts))
	}
	for _, l := range eigenvaluesOf(t, sp.parts[0].A) {
		if cmplx.Abs(l) >= cut {
			t.Errorf("slow part holds the mode %v above the cut %g", l, cut)
		}
	}
	x0 := []float64{1, 0.5, 1 / 3.0, 0.25, 0.2}
	r, err := Initial(sys, mat.NewVecDense(5, x0), 0)
	if err != nil {
		t.Fatalf("Initial: %v", err)
	}
	if w, ok := stiffSplitWithinRel(t, sys, "initial", x0, r, 1e-7); !ok {
		t.Errorf("Initial off the exact free response by %g relative", w)
	}
	s, err := Step(sys, 0)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if w, ok := stiffSplitWithinRel(t, sys, "step", nil, s, 1e-7); !ok {
		t.Errorf("Step off the exact response by %g relative", w)
	}
	d, err := sys.C2D(0.01, C2DOptions{})
	if err != nil {
		t.Fatalf("C2D: %v", err)
	}
	g0, bar := stiffSplitDC(t, sys)
	if dc := stiffSplitDiscreteDC(t, d); math.Abs(dc-g0) > 10*bar {
		t.Errorf("C2D ZOH DC %g, exact %g: %g bars", dc, g0, math.Abs(dc-g0)/bar)
	}
}

func eigenvaluesOf(t *testing.T, a *mat.Dense) []complex128 {
	t.Helper()
	var eig mat.Eigen
	if !eig.Factorize(a, mat.EigenNone) {
		t.Fatal("eigen factorization failed")
	}
	return eig.Values(nil)
}
