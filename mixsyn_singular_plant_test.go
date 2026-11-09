package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// TestMixsynNumericallySingularStaticPlant is the square static G with
// singular values 258, 63 and 2.7e-13 (Z2C5YG). K = 0 gives γ = max|W1| =
// 100, the optimum since the near-null output direction of G cannot be
// controlled; Mixsyn returned ErrNotConverged: wellPosedQ counted the
// 2.7e-13 in the rank of D22 = −G, which shrank the Youla margin below
// what loopShiftIllPosed accepts.
func TestMixsynNumericallySingularStaticPlant(t *testing.T) {
	D := mat.NewDense(3, 3, []float64{
		95.76575281793012, 30.239088825335227, -40.061662395521715,
		-128.5980865880245, -71.08545020314747, -2.349984258481528,
		153.96097157258234, 107.80290777082504, 44.625135203013,
	})
	one := func(v float64) *mat.Dense { return mat.NewDense(1, 1, []float64{v}) }
	for _, dt := range []float64{0, 1, 0.1, 1e-2, 1e-4} {
		G, err := NewGain(D, dt)
		if err != nil {
			t.Fatal(err)
		}
		var W1 *System
		if dt == 0 {
			W1, err = New(one(-0.1), one(1), one(9.95), one(0.5), 0)
		} else {
			W1, err = New(one(0.9048374180359595), one(1), one(9.468676905422027), one(0.5), dt)
		}
		if err != nil {
			t.Fatal(err)
		}
		W2, err := NewGain(one(0.01), dt)
		if err != nil {
			t.Fatal(err)
		}
		res, err := Mixsyn(G, W1, W2, nil)
		if err != nil {
			t.Fatalf("dt=%g: %v", dt, err)
		}
		info := res.Info
		if !(info.GammaOpt <= 100*(1+2*hinfControllerBackoff)) {
			t.Errorf("dt=%g: GammaOpt = %.10g, want ≤ 100(1+2e-4) (K = 0 gives 100)", dt, info.GammaOpt)
		}
		if !(info.Gamma <= info.GammaOpt) {
			t.Errorf("dt=%g: Gamma = %.10g above GammaOpt = %.10g", dt, info.Gamma, info.GammaOpt)
		}
		P, err := Augw(G, W1, W2, nil)
		if err != nil {
			t.Fatal(err)
		}
		cl, err := LFT(P, res.K, LFTFeedback{Nu: 3, Ny: 3})
		if err != nil {
			t.Fatal(err)
		}
		if ok, err := cl.IsStable(); err != nil || !ok {
			t.Errorf("dt=%g: closed loop not stable (%v)", dt, err)
		}
		norm, _, err := HinfNorm(cl)
		if err != nil {
			t.Fatal(err)
		}
		if !(norm <= info.GammaOpt*(1+1e-9)) {
			t.Errorf("dt=%g: ‖CL‖∞ = %.10g above GammaOpt = %.10g", dt, norm, info.GammaOpt)
		}
	}
}
