package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Cheap control (R = 1e-12·I) with a cross term S and m > n: the inputs in
// the null space of B lower the cost by about xᵀSR⁻¹Sᵀx ≈ ‖Q‖, which
// R + BᵀXB (condition ≈ 1e28) loses in float64. The pole-placement Hewer
// fallback converged there to X ≈ Q and certified it (Accuracy 0, orel
// 6e-3 and 0.12), where the pencil (and base) returned ErrNoStabilizing.
// Draws seed31 k=1034 and seed99 class4 k=2476 of the cheapS_E review
// sweep; the doubling oracle moves only 1e-10 under eps perturbations.
func TestDare_CrossTermRankDeficientBIsErrorOrAccurate(t *testing.T) {
	cases := []struct{ a, b, q, s []float64 }{
		{
			[]float64{-160.16317185556957, 32.57715077392217, -160.90264665388742, 147.36258107952219},
			[]float64{-0.46943512585815883, 0.78425864045369575, 1.5178474246624851, -1.7749767000838221, 0.51375479371405053, -1.3216979308286421},
			[]float64{1501457333218732.5, 132642637393436.47, 132642637393436.47, 11717994821051.344},
			[]float64{-30.660856428607421, 10.644744120882223, -7.1744978968213493, -2.7086596278520543, 0.94038432084197143, -0.63381376343739981},
		},
		{
			[]float64{-48.041305338496379, -231.44376668522648, -302.15063650674261, -493.26040646523842},
			[]float64{-0.41420559473379848, 2.473199358455294, 0.3682003239634668, 0.040105097483021518, 0.83115945696284554, -0.59265437443322133},
			[]float64{1420564207548869.5, -82219070043302.672, -82219070043302.672, 4758655358809.5771},
			[]float64{9.7251634890181151, 14.203468472949853, 4.1889656787980316, -0.56287064944837639, -0.82206489719341413, -0.24244793774464996},
		},
	}
	R := mat.NewDense(3, 3, []float64{1e-12, 0, 0, 0, 1e-12, 0, 0, 0, 1e-12})
	for i, c := range cases {
		A, B, Q, S := mat.NewDense(2, 2, c.a), mat.NewDense(2, 3, c.b), mat.NewDense(2, 2, c.q), mat.NewDense(2, 3, c.s)
		want := dareDoublingBig(A, B, Q, R, S)
		for _, ns := range []bool{false, true} {
			res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, NoScaling: ns})
			if err != nil {
				continue
			}
			var d mat.Dense
			d.Sub(res.X, want)
			if rel := mat.Norm(&d, 2) / mat.Norm(want, 2); !(rel <= 1e-8) {
				t.Errorf("case %d NoScaling %v: Accuracy %d, ‖X − X*‖/‖X*‖ = %.3g", i, ns, res.Accuracy, rel)
			}
		}
	}
}
