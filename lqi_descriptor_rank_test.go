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
// its LU estimate rejected this E (σ₁ ~ 1e-3, cond ~ 1e13, full MATLAB
// rank) as ErrDescriptorSingular (AHNSX7, RL6IT7 amendment 2). The solve
// is at the edge of Care's convergence (it fails under -race), so a design,
// when returned, must match the 320-bit oracle, and an error must not be
// ErrDescriptorSingular. A singular plant E stays ErrDescriptorSingular.
func TestLqiDescriptorJudgesPlantE(t *testing.T) {
	plant := obsTestPlant(t, 0, true)
	E := mat.DenseCopyOf(plant.E)
	for i := range 3 {
		for j, d := range []float64{1e-3, 1e-4, 3e-16} {
			E.Set(i, j, E.At(i, j)*d)
		}
	}
	sys, err := NewDescriptor(plant.A, plant.B, plant.C, plant.D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	Q, R := eyeDense(5), eyeDense(2)
	res, err := acceptUnverified(Lqi(sys, Q, R, nil))
	switch {
	case errors.Is(err, ErrDescriptorSingular):
		t.Errorf("Lqi: %v", err)
	case err == nil && res.accuracy == 0:
		Aa, Ba := lqiAugmentation(sys)
		pr := nearSingularRiccati{true, Aa, Ba, Q, R, nil, augmentedDescriptor(E, 2), false}
		if eX, eK, ok := pr.honest(t, res.X, res.K); !ok {
			t.Errorf("Lqi: X off the oracle by %.3g, K by %.3g", eX, eK)
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
		if _, err := Lqi(singular, Q, R, nil); !errors.Is(err, ErrDescriptorSingular) {
			t.Errorf("dt=%v singular E: err = %v, want ErrDescriptorSingular", dt, err)
		}
	}
}
