package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

var stiffClusterLiterals = []string{"cluster1e6s0", "cluster1e6s5", "cluster1e7s0", "cluster1e4s0"}

// stiffSplitDCWithin reports whether the discrete DC of d is within 10
// descriptor bars at s = 0 of the exact G(0) of sys, and the bars.
func stiffSplitDCWithin(t *testing.T, sys, d *System) (float64, bool) {
	t.Helper()
	g0, bar := stiffSplitDC(t, sys)
	r := math.Abs(stiffSplitDiscreteDC(t, d)-g0) / bar
	return r, r <= 10
}

// The pencils of the SCPYGG repro had their near-defective pair cut in two
// by the nominal stiff cut, and C2D returned nil with DC off by up to 6e-4;
// the cut now lies in the gap above the pair, and every method is nil with
// its DC within the descriptor bar (the pair's straddle used to cost
// 1e3–5e7 bars).
func TestC2DStiffDescriptorClusterAtSplitStaysAccurate(t *testing.T) {
	for _, name := range stiffClusterLiterals {
		sys := stiffSplitLiteral(t, name)
		for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodMatched} {
			d, err := sys.C2D(0.01, C2DOptions{Method: m})
			if err != nil {
				t.Errorf("%s: C2D %s: %v", name, m, err)
				continue
			}
			if r, ok := stiffSplitDCWithin(t, sys, d); !ok {
				t.Errorf("%s: C2D %s DC off by %g descriptor bars", name, m, r)
			}
		}
	}
}

// Step, Impulse, Initial and Lsim of the repro pencils follow the exact
// responses within 1e-7 relative (base: up to 1e-3).
func TestStepStiffDescriptorClusterAtSplitStaysAccurate(t *testing.T) {
	x0 := []float64{1, 0.5, 1 / 3.0, 0.25, 0.2}
	for _, name := range stiffClusterLiterals {
		sys := stiffSplitLiteral(t, name)
		s, err := Step(sys, 0)
		if err != nil {
			t.Fatalf("%s: Step: %v", name, err)
		}
		if w, ok := stiffSplitWithinRel(t, sys, "step", nil, s, 1e-7); !ok {
			t.Errorf("%s: Step off by %g relative", name, w)
		}
		r, err := Impulse(sys, 0)
		if err != nil {
			t.Fatalf("%s: Impulse: %v", name, err)
		}
		if w, ok := stiffSplitWithinRel(t, sys, "impulse", nil, r, 1e-7); !ok {
			t.Errorf("%s: Impulse off by %g relative", name, w)
		}
		in, err := Initial(sys, mat.NewVecDense(5, x0), 0)
		if err != nil {
			t.Fatalf("%s: Initial: %v", name, err)
		}
		if w, ok := stiffSplitWithinRel(t, sys, "initial", x0, in, 1e-7); !ok {
			t.Errorf("%s: Initial off by %g relative", name, w)
		}
		grid := make([]float64, 201)
		u := mat.NewDense(201, 1, nil)
		for k := range grid {
			grid[k] = 0.01 * float64(k)
			u.Set(k, 0, 1)
		}
		l, err := Lsim(sys, u, grid, nil)
		if err != nil {
			t.Fatalf("%s: Lsim: %v", name, err)
		}
		if w, ok := stiffSplitWithinRel(t, sys, "step", nil, l, 1e-7); !ok {
			t.Errorf("%s: Lsim of a unit step off by %g relative", name, w)
		}
	}
}

// strad1h/4 and /43 have no certified split at the moved cut (z2-dominated
// rejection): each C2D method returns ErrSchurFailed with a nil model, or,
// where rounding certifies a split, nil within the descriptor bar; never nil
// outside it.
func TestC2DStiffDescriptorUnseparableSplitErrors(t *testing.T) {
	for _, name := range []string{"unseparable4", "unseparable43"} {
		sys := stiffSplitLiteral(t, name)
		for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodImpulse, C2DMethodMatched} {
			d, err := sys.C2D(0.01, C2DOptions{Method: m})
			switch {
			case err != nil:
				if !errors.Is(err, ErrSchurFailed) || d != nil {
					t.Errorf("%s: C2D %s = %v, %v; want nil, ErrSchurFailed", name, m, d, err)
				}
			case m == C2DMethodImpulse:
			default:
				if r, ok := stiffSplitDCWithin(t, sys, d); !ok {
					t.Errorf("%s: C2D %s nil with DC off by %g descriptor bars", name, m, r)
				}
			}
		}
		if _, _, err := sys.C2DFit(0.01, C2DOptions{Method: C2DMethodLeastSquares}); err != nil && !errors.Is(err, ErrSchurFailed) && !errors.Is(err, ErrAccuracyUnverified) {
			t.Errorf("%s: C2DFit: %v", name, err)
		}
	}
}

// The time responses of the same pencils: ErrSchurFailed with no response,
// or nil within 1e-7 relative of the exact response.
func TestStepStiffDescriptorUnseparableSplitErrors(t *testing.T) {
	x0 := []float64{1, 0.5, 1 / 3.0, 0.25, 0.2}
	for _, name := range []string{"unseparable4", "unseparable43"} {
		sys := stiffSplitLiteral(t, name)
		grid := make([]float64, 201)
		u := mat.NewDense(201, 1, nil)
		for k := range grid {
			grid[k] = 0.01 * float64(k)
			u.Set(k, 0, 1)
		}
		for _, c := range []struct {
			api, kind string
			run       func() (*TimeResponse, error)
		}{
			{"Step", "step", func() (*TimeResponse, error) { return Step(sys, 0) }},
			{"Impulse", "impulse", func() (*TimeResponse, error) { return Impulse(sys, 0) }},
			{"Initial", "initial", func() (*TimeResponse, error) { return Initial(sys, mat.NewVecDense(5, x0), 0) }},
			{"Lsim", "step", func() (*TimeResponse, error) { return Lsim(sys, u, grid, nil) }},
		} {
			r, err := c.run()
			if err != nil {
				if !errors.Is(err, ErrSchurFailed) || r != nil {
					t.Errorf("%s: %s = %v, %v; want nil, ErrSchurFailed", name, c.api, r, err)
				}
				continue
			}
			if w, ok := stiffSplitWithinRel(t, sys, c.kind, x0, r, 1e-7); !ok {
				t.Errorf("%s: %s nil off by %g relative", name, c.api, w)
			}
		}
	}
}

// stfr/1350 splits at ζ″ 0.628 (arm64), certified, but its ZOH and FOH DC
// sit at 10.01 and 10.03 bars: ZOH and FOH take the parts only at ζ″ ≤ 0.5
// (ErrSchurFailed, or nil within the bar where rounding lowers ζ″), while
// Tustin and Step, which take them at ζ″ ≤ 1, stay nil and accurate.
func TestC2DStiffDescriptorMarginalSplitErrors(t *testing.T) {
	sys := stiffSplitLiteral(t, "marginal1350")
	sp, ok := stiffExplicitSplit(sys)
	if !ok || sp.err != nil {
		t.Fatalf("stiffExplicitSplit: ok %v, err %v; want a certified split", ok, sp.err)
	}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH} {
		d, err := sys.C2D(0.01, C2DOptions{Method: m})
		if err != nil {
			if !errors.Is(err, ErrSchurFailed) || d != nil || sp.zeta <= stiffHoldZeta {
				t.Errorf("C2D %s = %v, %v at ζ″ %g; want ErrSchurFailed only above %g", m, d, err, sp.zeta, stiffHoldZeta)
			}
			continue
		}
		if r, ok := stiffSplitDCWithin(t, sys, d); !ok || sp.zeta > stiffHoldZeta {
			t.Errorf("C2D %s nil at ζ″ %g with DC off by %g bars", m, sp.zeta, r)
		}
	}
	d, err := sys.C2D(0.01, C2DOptions{Method: C2DMethodTustin})
	if err != nil {
		t.Fatalf("C2D tustin: %v", err)
	}
	if r, ok := stiffSplitDCWithin(t, sys, d); !ok {
		t.Errorf("C2D tustin DC off by %g bars", r)
	}
	s, err := Step(sys, 0)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if w, ok := stiffSplitWithinRel(t, sys, "step", nil, s, 1e-7); !ok {
		t.Errorf("Step off by %g relative", w)
	}
}

// TransferFunction keeps judging the fold the split falls back to: on the
// unseparable and marginal pencils it returns nil, which the judge
// (descriptorTFVerified) certifies, never the split's error.
func TestTransferFunctionStiffDescriptorUnseparableSplitStaysJudged(t *testing.T) {
	for _, name := range []string{"unseparable4", "unseparable43", "marginal1350"} {
		sys := stiffSplitLiteral(t, name)
		res, err := sys.TransferFunction(nil)
		if err != nil {
			t.Errorf("%s: TransferFunction: %v", name, err)
			continue
		}
		if !descriptorTFVerified(sys, res.TF) {
			t.Errorf("%s: nil TransferFunction the judge does not verify", name)
		}
	}
}

// strad2/4 has its pair on the nominal cut of splitFastPencil; the cut moves
// below the pair and C2D, Step and Initial stay accurate.
func TestC2DStiffDescriptorInnerClusterAtSplitStaysAccurate(t *testing.T) {
	sys := stiffSplitLiteral(t, "innerStraddle4")
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodTustin} {
		d, err := sys.C2D(0.01, C2DOptions{Method: m})
		if err != nil {
			t.Fatalf("C2D %s: %v", m, err)
		}
		if r, ok := stiffSplitDCWithin(t, sys, d); !ok {
			t.Errorf("C2D %s DC off by %g bars", m, r)
		}
	}
	x0 := []float64{1, 0.5, 1 / 3.0, 0.25, 0.2}
	in, err := Initial(sys, mat.NewVecDense(5, x0), 0)
	if err != nil {
		t.Fatalf("Initial: %v", err)
	}
	if w, ok := stiffSplitWithinRel(t, sys, "initial", x0, in, 1e-7); !ok {
		t.Errorf("Initial off by %g relative", w)
	}
}

// The I/O-delay route of C2D (stiffDescriptorExplicit) and the internal-
// delay route of Step (stiffLFTSampledAuto, an inactive delay loop so the
// response is that of the pencil) take the moved cut too.
func TestC2DStiffDescriptorIODelayClusterAtSplit(t *testing.T) {
	sys := stiffSplitLiteral(t, "cluster1e6s0")
	sys.InputDelay = []float64{0.03}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodTustin} {
		d, err := sys.C2D(0.01, C2DOptions{Method: m})
		if err != nil {
			t.Fatalf("C2D %s: %v", m, err)
		}
		core := *d
		core.InputDelay, core.OutputDelay, core.Delay = nil, nil, nil
		if r, ok := stiffSplitDCWithin(t, stiffSplitLiteral(t, "cluster1e6s0"), &core); !ok {
			t.Errorf("C2D %s DC off by %g bars", m, r)
		}
	}
}

func TestStepStiffDescriptorInternalDelayClusterAtSplit(t *testing.T) {
	core := stiffSplitLiteral(t, "cluster1e6s0")
	sys := core.Copy()
	if err := sys.SetInternalDelay([]float64{0.05}, mat.NewDense(5, 1, []float64{0.3, -0.2, 0.5, 0.1, -0.4}),
		mat.NewDense(1, 5, nil), mat.NewDense(1, 1, []float64{0.3}), mat.NewDense(1, 1, nil), mat.NewDense(1, 1, nil)); err != nil {
		t.Fatal(err)
	}
	if _, ok, err := newStiffLFTSplitWithin(sys, nil, 1); !ok || err != nil {
		t.Fatalf("newStiffLFTSplitWithin: ok %v, err %v", ok, err)
	}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodTustin} {
		d, err := sys.C2D(0.01, C2DOptions{Method: m})
		if err != nil {
			t.Fatalf("C2D %s: %v", m, err)
		}
		plain := *d
		plain.LFT = nil
		if r, ok := stiffSplitDCWithin(t, core, &plain); !ok {
			t.Errorf("C2D %s DC off by %g bars", m, r)
		}
	}
	s, err := Step(sys, 0)
	if err != nil {
		t.Fatalf("Step: %v", err)
	}
	if w, ok := stiffSplitWithinRel(t, core, "step", nil, s, 1e-7); !ok {
		t.Errorf("Step off by %g relative", w)
	}
}

// C2DFit samples the slow part of the moved cut (leastSquaresPencilSlow):
// on the repro pencil it fits; on an unseparable one it errors rather than
// fall to the fold.
func TestC2DFitStiffDescriptorUnseparableSplitErrors(t *testing.T) {
	if slow, err := leastSquaresPencilSplit(stiffSplitLiteral(t, "cluster1e6s0")); err != nil || slow == nil {
		t.Fatalf("leastSquaresPencilSplit(cluster): %v, %v", slow, err)
	}
	for _, name := range []string{"unseparable4", "unseparable43"} {
		sys := stiffSplitLiteral(t, name)
		if sp, ok := stiffExplicitSplit(sys); !ok || sp.err == nil {
			continue
		}
		if d, _, err := sys.C2DFit(0.01, C2DOptions{Method: C2DMethodLeastSquares}); !errors.Is(err, ErrSchurFailed) || d != nil {
			t.Errorf("%s: C2DFit = %v, %v; want nil, ErrSchurFailed", name, d, err)
		}
		if leastSquaresPencilSlow(sys) != nil {
			t.Errorf("%s: leastSquaresPencilSlow returned a slow part", name)
		}
	}
}
