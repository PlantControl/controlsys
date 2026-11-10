package controlsys

import (
	"errors"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestLqiDescriptorJudgesPlantE: Lqi judges the nonsingularity of the plant
// E, as MATLAB lqi ("descriptor models with nonsingular E") and Lqr do, not
// of blkdiag(E, I): rank(blkdiag(E, I)) = rank(E) + p exactly, while the
// MATLAB rank tolerance of the augmented E, (n+p)·eps(max(σ₁(E), 1)), and
// its LU estimate rejected these full-rank plant E that Lqr admits
// (AHNSX7, RL6IT7 amendment 2). The design must match the 320-bit oracle;
// a singular plant E stays ErrDescriptorSingular.
func TestLqiDescriptorJudgesPlantE(t *testing.T) {
	for _, c := range []struct {
		seed uint64
		dt   float64
	}{{88, 1}, {857, 0}, {857, 1}} {
		sys := nearSingularTriangularE(c.seed, c.dt)
		n, m, p := sys.Dims()
		Q, R := eyeDense(n+p), eyeDense(m)
		res, err := Lqi(sys, Q, R, nil)
		if err != nil {
			t.Errorf("seed %d dt=%v: %v", c.seed, c.dt, err)
			continue
		}
		Aa, Ba := lqiAugmentation(sys)
		pr := nearSingularRiccati{sys.IsContinuous(), Aa, Ba, Q, R, nil, augmentedDescriptor(sys.E, p), false}
		if eX, eK, ok := pr.honest(t, res.X, res.K); !ok {
			t.Errorf("seed %d dt=%v: X off the oracle by %.3g, K by %.3g", c.seed, c.dt, eX, eK)
		}
	}
	for _, dt := range []float64{0, 0.1} {
		sys := obsTestPlant(t, dt, true)
		E := mat.DenseCopyOf(sys.E)
		for j := range 3 {
			E.Set(2, j, 0)
		}
		singular, err := NewDescriptor(sys.A, sys.B, sys.C, sys.D, E, dt)
		if err != nil {
			t.Fatal(err)
		}
		if _, err := Lqi(singular, eyeDense(5), eyeDense(2), nil); !errors.Is(err, ErrDescriptorSingular) {
			t.Errorf("dt=%v singular E: err = %v, want ErrDescriptorSingular", dt, err)
		}
	}
}
