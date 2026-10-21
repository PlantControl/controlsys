package controlsys

import (
	"errors"
	"fmt"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type careVerifyCase struct {
	name          string
	n, m          int
	mustSolve     [2]bool
	a, b, q, r, s []float64
}

var careVerifyCases = []careVerifyCase{
	{"w14#30", 5, 1, [2]bool{true, true},
		[]float64{0.04965432172693425, -0.03165948738350457, -0.055080741521631255, 0.03493333743143347, -0.02786075137964385, 0.018331847981550294, 0.018919148463571584, 0.03313301252483951, 0.004700068669567451, 0.04153142754623905, -0.044922592111098, -0.020697009370856726, -0.034810188415381686, -0.026944923215340765, -0.03617624106699076, -0.008710591648000199, -0.026384579354286845, -0.06060263379475673, -0.03254298068004032, -0.04619026611713184, 0.04754408806630633, 0.030064752707127884, 0.04034646545833218, -0.1379619074860704, -0.0028123309723317035},
		[]float64{-9.127121289458064, -10.208553047250046, -17.846076290937127, 2.3177472118403957, -10.375791688360813},
		[]float64{28134.69666313963, 5947.971155465895, 5180.144986968966, 4709.582803574447, 2539.5317511658595, 5947.971155465895, 7411.913954078084, 6547.219104279577, -2473.675245259564, -1565.7900310977147, 5180.144986968966, 6547.219104279577, 23467.986992056292, -9739.801952771291, -7604.131112275712, 4709.582803574447, -2473.675245259564, -9739.801952771291, 5953.3380407962795, 4255.435196396421, 2539.5317511658595, -1565.7900310977147, -7604.131112275712, 4255.435196396421, 3127.5950987577144},
		[]float64{3.648930774992671e-05}, nil},
	{"w14#32", 3, 2, [2]bool{true, true},
		[]float64{0.0435805386236684, -0.053555616615496696, 0.08150582081404058, -0.23623489379182055, -0.09511188269420727, 0.06261479497280781, 0.03695081562552784, -0.14606170488572906, 0.10804037977577631},
		[]float64{-4.547530266236957, 1.5774305751691902, 1.3729848496318586, 0.39725692493181264, 7.46289825563405, -0.9075375313881756},
		[]float64{1.3244588426399657, -0.13651978250417657, 0.33752267816286624, -0.13651978250417657, 0.30306632925549193, 0.5696502583893951, 0.33752267816286624, 0.5696502583893951, 1.7594570598263795},
		[]float64{2.2571446141393743e-13, 0, 0, 2.2571446141393743e-13}, nil},
	{"w14#23", 4, 1, [2]bool{true, true},
		[]float64{16.68754600898942, -5.407355835812163, 10.085305918970889, -4.191432188912107, 2.308620797559564, 1.6207105034687457, -19.977195561532778, 6.926931942817217, 14.077185132674408, -12.550763250913569, -2.3997049256765917, 16.634014383630976, -10.570833363621203, 2.3783445334207465, 8.308376722676933, 7.565295764099132},
		[]float64{-0.2071692269353604, 0.16122813422072046, 0.29551589856198396, -0.1827957261534804},
		[]float64{10.3351191293987, 2.70199057336242, 6.113191213360301, 32.266759509037264, 2.70199057336242, 1.818538310763379, 8.652240441985626, 9.18472916094571, 6.113191213360301, 8.652240441985626, 48.35796285307395, 23.836291777030464, 32.266759509037264, 9.18472916094571, 23.836291777030464, 101.2428468569077},
		[]float64{1.2796805656810533e-12}, nil},
	{"hamErr", 3, 1, [2]bool{true, true},
		[]float64{-0.12333753670867588, -0.047016477479840196, 0.05359901094214391, 0.04155233576437227, -0.2770399707716169, 0.06780522434231762, -0.3936733009750805, 0.08676801816197652, -0.1697317494701551},
		[]float64{0.5144891150464497, 7.258131134365759, -3.1424091797043046},
		[]float64{1.23710268179117e+06, 488573.2741385464, 227338.17660963573, 488573.2741385464, 193045.00442849798, 91901.81662944313, 227338.17660963573, 91901.81662944313, 91058.12954553009},
		[]float64{6.304128548020456e-09}, nil},
}

func (tc careVerifyCase) dense() (A, B, Q, R, S *mat.Dense) {
	A, B = mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, tc.m, tc.b)
	Q, R = mat.NewDense(tc.n, tc.n, tc.q), mat.NewDense(tc.m, tc.m, tc.r)
	if tc.s != nil {
		S = mat.NewDense(tc.n, tc.m, tc.s)
	}
	return
}

// TestCare_NewtonVerified: Care returned, with nil error, an inaccurate X
// (relative error 0.11 and 6.4e-2) or a K with A − BK unstable on cheap
// control problems, where the Hamiltonian forms G = BR⁻¹Bᵀ ≫ A. The
// oracle is 320-bit Newton–Kleinman from the returned K, which must be
// stabilizing.
//
//	"w14#30" base default rel err 0.11, NoScaling 3.4e-3 (1-ulp input
//	         sensitivity 9e-9); poles −6.2e5 and −0.009.
//	"w14#32" R = 2.3e-13·I: base default rel err 6.4e-2, NoScaling 3.3e-2
//	         (sensitivity 1.7e-16). Float64 Newton stalls near 3e-8.
//	"w14#23" R = 1.3e-12: base returned an unstable A − BK in both modes;
//	         the extended pencil (MATLAB icare) gives the stabilizing X.
//	"hamErr" random sweep, R = 6.3e-9: base default rel err 1.77, NoScaling
//	         ErrNoStabilizing from the Hamiltonian; the pencil solves both.
func TestCare_NewtonVerified(t *testing.T) {
	for _, tc := range careVerifyCases {
		A, B, Q, R, S := tc.dense()
		for k, noScaling := range []bool{false, true} {
			t.Run(fmt.Sprintf("%s/NoScaling=%v", tc.name, noScaling), func(t *testing.T) {
				res, err := Care(A, B, Q, R, &RiccatiOpts{S: S, NoScaling: noScaling})
				if errors.Is(err, ErrNoStabilizing) && !tc.mustSolve[k] {
					return
				}
				if err != nil {
					t.Fatal(err)
				}
				var bk, acl mat.Dense
				bk.Mul(B, res.K)
				acl.Sub(A, &bk)
				var eig mat.Eigen
				if !eig.Factorize(&acl, mat.EigenNone) {
					t.Fatal("Eigen failed")
				}
				for _, v := range eig.Values(nil) {
					if !(real(v) < 0) {
						t.Fatalf("A − BK eigenvalue %v", v)
					}
				}
				want := kleinmanCareBig(A, B, Q, R, S, res.K, 30)
				if e := relErr(kleinmanCareBig(A, B, Q, R, S, res.K, 20), want); !(e <= 1e-15) {
					t.Fatalf("oracle not converged: %g", e)
				}
				if e := relErr(res.X, want); !(e <= 1e-7) {
					t.Errorf("X rel err %g against 320-bit Newton–Kleinman", e)
				}
				var tr complex128
				for _, v := range res.Eig {
					tr += v
				}
				if d := math.Abs(real(tr) - mat.Trace(&acl)); !(d <= 1e-8*math.Max(1, mat.Norm(&acl, 2))) {
					t.Errorf("Σ Eig = %v, trace(A − BK) = %g", tr, mat.Trace(&acl))
				}
			})
		}
	}
}
