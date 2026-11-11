package controlsys

import (
	"errors"
	"fmt"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestLqiDescriptorJudgesPlantE: Lqi judges the nonsingularity of the plant
// E, as MATLAB lqi ("descriptor models with nonsingular E") and Lqr do, not
// of blkdiag(E, I): rank(blkdiag(E, I)) = rank(E) + p exactly, while the
// MATLAB rank tolerance of the augmented E, (n+p)·eps(max(σ₁(E), 1)), and
// its LU estimate rejected these E as ErrDescriptorSingular (AHNSX7, RL6IT7
// amendment 2). nearSingularTriangularE seeds 857 (full MATLAB rank, LU
// estimate ≥ 1/ε) and 962 (LU estimate < 1/ε, MATLAB rank deficient), dt=1:
// Lqi must return the 320-bit oracle's design; both lie in the plant-row
// scale's dead band (bit-identical to blkdiag(E, I)) and do not exercise it
// (ZIUHCL ruling 3). So must the E with σ₁ ~ 1e-3 and σ_min ~ 3e-16 in
// every build (ZIUHCL): with blkdiag(E, I) the unit integrator rows set ‖J‖
// and the fast ±4.2e15 pair of the extended pencil sat at Dhgeqz's infinite
// threshold, where the FMA contraction choice in gonum Drot decided the
// sign of a numerically infinite eigenvalue and so whether Care found n
// stable ones. With the plant rows scaled to unit RMS the pair's β is ~300
// ulps above that threshold in every build: its sign is data, not FMA. Eig
// beyond the fast mode is not compared (unreliable for this E). A singular
// plant E stays ErrDescriptorSingular.
func TestLqiDescriptorJudgesPlantE(t *testing.T) {
	oracle := func(label string, sys *System, res *RiccatiResult) {
		t.Helper()
		n, m, p := sys.Dims()
		Aa, Ba := lqiAugmentation(sys)
		pr := nearSingularRiccati{sys.IsContinuous(), Aa, Ba, eyeDense(n + p), eyeDense(m), nil, augmentedDescriptor(sys.E, p), false}
		if eX, eK, ok := pr.honest(t, res.X, res.K); !ok {
			t.Errorf("%s: X off the oracle by %.3g, K by %.3g", label, eX, eK)
		}
	}
	for _, seed := range []uint64{857, 962} {
		sys := nearSingularTriangularE(seed, 1)
		n, m, p := sys.Dims()
		res, err := Lqi(sys, eyeDense(n+p), eyeDense(m), nil)
		if err != nil {
			t.Errorf("seed %d: %v", seed, err)
			continue
		}
		oracle(fmt.Sprintf("seed %d", seed), sys, res)
	}

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
	if res, err := Lqi(sys, eyeDense(5), eyeDense(2), nil); err != nil {
		t.Errorf("scaled E: %v", err)
	} else {
		oracle("scaled E", sys, res)
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
