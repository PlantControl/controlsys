package controlsys

import (
	"fmt"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// dareGainBig returns (R + BᵀXB)⁻¹(BᵀXA + Sᵀ) in 320-bit arithmetic.
func dareGainBig(A, B, R, S, X *mat.Dense) *mat.Dense {
	b, x := hbFrom(B), hbFrom(X)
	btx := hbMul(b.t(), x)
	return hbDense(hbSolve(hbAdd(hbFrom(R), hbMul(btx, b), 1), hbAdd(hbMul(btx, hbFrom(A)), hbFrom(S).t(), 1)))
}

// Cheap control (R ≈ 1e-12·I) with a cross term S and m > n: the inputs in
// the null space of B lower the cost by about xᵀSR⁻¹Sᵀx ≈ ‖Q‖, which
// R + BᵀXB (condition ≈ 1e28) loses in float64. The pole-placement Hewer
// fallback converged there to X ≈ Q and certified it (Accuracy 0, orel
// 6e-3 and 0.12), where the pencil returned ErrNoStabilizing: draws seed31
// k=1034 and seed99 class4 k=2476 of the cheapS_E review sweep. Reduced,
// their Q − S₂R₂₂⁻¹S₂ᵀ is singular (rank 1) and some variants stay an
// error or Accuracy 1 on amd64, so only a certified X is checked there.
// ZLHTCH: the pencil itself was certified at X = Q to 7 digits (Accuracy
// 0, rel 0.40); it must be solved. Dare now eliminates the null-space
// inputs first. The doubling
// oracle moves only 1e-10 (3e-16 for ZLHTCH) under eps perturbations, and
// the gain is checked against the 320-bit gain of the oracle X, since its
// null-space part is invisible to X, Eig and A − BK.
func TestDare_CrossTermRankDeficientBIsAccurate(t *testing.T) {
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
		{
			[]float64{137.49376849957935, -119.73780927231955, -226.41172927457049, 326.39255307674279},
			[]float64{-1.0892768441551315, -1.0711022897470619, 1.0405728611691283, 1.4792698090856553, -0.23313941332752872, 0.98955612351270239},
			[]float64{210978160131384, -123955056227369.69, -123955056227369.69, 477539722037894.5},
			[]float64{-5.308218111719408, -0.67678179911166603, 2.5863939290433362, -0.25612464867339946, -7.7288266704683011, -10.679115183263386},
		},
	}
	rs := map[string]*mat.Dense{
		"R=1e-12·I": mat.NewDense(3, 3, []float64{1e-12, 0, 0, 0, 1e-12, 0, 0, 0, 1e-12}),
		"coupled R": mat.NewDense(3, 3, []float64{1e-12, 0.3e-12, 0.1e-12, 0.3e-12, 2e-12, 0.2e-12, 0.1e-12, 0.2e-12, 1.5e-12}),
	}
	E := dareCheapDescriptor(2)
	for i, c := range cases {
		A, B, Q, S := mat.NewDense(2, 2, c.a), mat.NewDense(2, 3, c.b), mat.NewDense(2, 2, c.q), mat.NewDense(2, 3, c.s)
		var Ae, Be mat.Dense
		Ae.Mul(E, A)
		Be.Mul(E, B)
		for rname, R := range rs {
			want := dareDoublingBig(A, B, Q, R, S)
			wantK := dareGainBig(A, B, R, S, want)
			type variant struct {
				name string
				A, B *mat.Dense
				opts RiccatiOpts
			}
			variants := []variant{
				{"scaled", A, B, RiccatiOpts{S: S}},
				{"NoScaling", A, B, RiccatiOpts{S: S, NoScaling: true}},
				{"workspace", A, B, RiccatiOpts{S: S, Workspace: NewRiccatiWorkspace(2, 3)}},
			}
			variants = append(variants, variant{"descriptor", &Ae, &Be, RiccatiOpts{S: S, E: E}})
			for _, tc := range variants {
				name := fmt.Sprintf("case %d %s %s", i, rname, tc.name)
				res, err := Dare(tc.A, tc.B, Q, R, &tc.opts)
				if i < 2 && (err != nil || res.Accuracy != 0) {
					continue
				}
				if err != nil {
					t.Errorf("%s: %v", name, err)
					continue
				}
				X := res.X
				if tc.opts.E != nil {
					var ex mat.Dense
					ex.Mul(E.T(), res.X)
					ex.Mul(&ex, E)
					X = &ex
				}
				if rel := relErr(X, want); !(rel <= 1e-8) {
					t.Errorf("%s: Accuracy %d, ‖X − X*‖/‖X*‖ = %.3g", name, res.Accuracy, rel)
				}
				if r, c := res.K.Dims(); r != 3 || c != 2 {
					t.Fatalf("%s: K is %d×%d, want 3×2", name, r, c)
				}
				if rel := relErr(res.K, wantK); !(rel <= 1e-6) {
					t.Errorf("%s: ‖K − K*‖/‖K*‖ = %.3g", name, rel)
				}
				for _, e := range res.Eig {
					if !(cmplx.Abs(e) < 1) {
						t.Errorf("%s: Eig %v not stable", name, e)
					}
				}
			}
		}
	}
}
