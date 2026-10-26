package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type dareInputsCase struct {
	name          string
	n, m          int
	a, b, q, r, s []float64
}

func (c dareInputsCase) matrices() (A, B, Q, R, S *mat.Dense) {
	A, B = mat.NewDense(c.n, c.n, c.a), mat.NewDense(c.n, c.m, c.b)
	Q, R = mat.NewDense(c.n, c.n, c.q), mat.NewDense(c.m, c.m, c.r)
	if c.s != nil {
		S = mat.NewDense(c.n, c.m, c.s)
	}
	return A, B, Q, R, S
}

// closedLoopRadiusBig returns ρ(A − BK), A − BK formed in 320 bits.
func closedLoopRadiusBig(t *testing.T, A, B, K *mat.Dense) float64 {
	t.Helper()
	return spectralRadius(t, hbDense(hbAdd(hbFrom(A), hbMul(hbFrom(B), hbFrom(K)), -1)))
}

// Cheap control (R ≈ 1e-12, Q ≈ 1e14) with a cross term and two nearly
// collinear actuators (σ₂/σ₁ ≈ 1e-8 to 3e-3): the doubling oracle moves
// by 3e-16 to 4e-10 under eps perturbations, yet R + BᵀXB loses its
// smallest eigenvalue to rounding, so the pencil's X stayed Accuracy 1 or
// Newton certified X off by 4e-6 to 0.73 (Accuracy 0), or, from a placed
// gain, by 5e-7 (r1, r2 of a riccati review; the rest draws 33,
// 82 and 55 of its near-collinear sweep). Dare now solves in the inputs of
// B's right singular vectors.
func TestDare_NearlyCollinearInputsAreCertified(t *testing.T) {
	for _, c := range []dareInputsCase{
		{"r1", 2, 2,
			[]float64{1.4620991642104912, -0.8784792442575562, -1.2071382595750249, 0.16268284639648423},
			[]float64{0.7873745903512255, -0.20096102363846285, -0.8592447854231653, 0.21930438011983636},
			[]float64{6.441166398562846e13, 4.786743416308757e13, 4.786743416308757e13, 1.614517464168949e14},
			[]float64{1.0193412194411262e-12, 2.999265069241783e-14, 2.999265069241783e-14, 6.403938765630009e-13},
			[]float64{2.9394250706403358, 0.8088194867064437, 2.625832947633263, 2.2414533619527712}},
		{"r2", 2, 2,
			[]float64{0.04477676362501934, -0.10997531085197076, 0.18278145381698863, -0.5939876437745422},
			[]float64{0.65530978279142, -0.017745498255182526, 0.3094120655804397, -0.00837873916352774},
			[]float64{3.347748065820429e13, 1.1009171512857527e13, 1.1009171512857527e13, 1.3393645199414516e14},
			[]float64{1.0225230629000013e-12, 0, 0, 9.302646545716362e-13},
			[]float64{-2.6647608525850623, -0.6622516055464818, 0.43266694227126656, -4.686341258998957}},
		{"draw 33", 2, 2,
			[]float64{0.5539224353225116, -0.7612888658172287, 2.2590982867083866, 0.9561959122895964},
			[]float64{-1.6516686180080142, -1.1015927527947798, -0.006347754064447764, -0.0042335798740499725},
			[]float64{2.466851128296175e+14, -1.2857241989643906e+13, -1.2857241989643906e+13, 1.2490895135755808e+14},
			[]float64{1.7230045410137294e-12, 1.9873192085068585e-13, 1.9873192085068585e-13, 1.212888093851943e-12},
			[]float64{5.815360706993781, -3.106215230894137, -3.530730310994973, -1.113435925337749}},
		{"draw 82", 2, 2,
			[]float64{-0.9913330658402236, 2.927339303016945, -0.5285063131927004, 0.6404614959201682},
			[]float64{-2.073633410697246, 2.0414253611870063, -0.37113783503156417, 0.3653733604116237},
			[]float64{6.547105145918588e+13, -3.841957554920592e+13, -3.841957554920592e+13, 3.76418578865797e+13},
			[]float64{1.0811116817062535e-12, 1.0634854736466282e-13, 1.0634854736466282e-13, 1.208668660792391e-12},
			[]float64{1.4367467487482455, -3.442251890125012, -2.545053691398813, 0.7329784421463956}},
		{"draw 55", 2, 2,
			[]float64{1.2743023357157424, -0.22626160756827257, -1.0657824600310555, -0.57699519625412},
			[]float64{-0.7387064205276983, -0.4105197620801335, -0.21413621959437412, -0.11900146984282577},
			[]float64{1.559570898931053e+14, 1.1567856160719672e+14, 1.1567856160719672e+14, 1.796570234220088e+14},
			[]float64{1.9329519471901176e-12, 1.1641462176967221e-14, 1.1641462176967221e-14, 1.8761990919460537e-12},
			[]float64{1.162613187811109, -5.952503893358973, 0.886048645191484, -4.860885644827148}},
	} {
		A, B, Q, R, S := c.matrices()
		want := dareDoublingBig(A, B, Q, R, S)
		for _, noScaling := range []bool{false, true} {
			res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, NoScaling: noScaling, Workspace: NewRiccatiWorkspace(c.n, c.m)})
			if err != nil {
				t.Errorf("%s NoScaling=%v: %v", c.name, noScaling, err)
				continue
			}
			if rel := relErr(res.X, want); res.Accuracy != 0 || !(rel <= 1e-8) {
				t.Errorf("%s NoScaling=%v: Accuracy %d, ‖X − X*‖/‖X*‖ = %.3g; want 0, ≤ 1e-8", c.name, noScaling, res.Accuracy, rel)
			}
			if rel := relErr(hewerDareBig(A, B, Q, R, S, res.K, 30), want); !(rel <= 1e-12) {
				t.Errorf("%s NoScaling=%v: Hewer from K reaches %.3g of X*", c.name, noScaling, rel)
			}
		}
	}
}

// With a cross term, cheap control and rank(B) = 1 < m = 2 by the rounding
// cutoff, yet ‖BV₂‖ ≈ 1e-16 ≠ 0, the K₂ ≈ R₂₂⁻¹S₂ᵀ ≈ 1e12 of the inputs B
// does not see moved A − BK by 1e-4: the reduced problem's K was returned
// with Eig of radius 0.2–0.6 where the 320-bit ρ(A − BK) was 1.27 and 1.45
// (draws 2008 and 300 of a riccati review's rank-deficient
// sweep; base returned ErrNoStabilizing). A returned K must stabilize the
// given plant, and Eig must be its closed loop.
func TestDare_ReducedInputsGainStabilizesGivenPlant(t *testing.T) {
	for _, c := range []dareInputsCase{
		{"draw 2008", 3, 2,
			[]float64{7.33999066730163, -7.930468323047169, -9.379124477622726, -7.676063924260655, 6.270748394198692, -9.520253427452053, 9.373707794221586, 7.969700911659494, 18.94295653412137},
			[]float64{0.03446671225835557, -0.0005715354869001516, 0.07337224803347592, -0.0012166766354280748, 0.48390615892354644, -0.008024250763496114},
			[]float64{1.0336659323447088e+14, -1.3675131277153653e+14, -1.6482652432946272e+14, -1.3675131277153653e+14, 3.198449046375242e+14, 2.5189475655900178e+14, -1.6482652432946272e+14, 2.5189475655900178e+14, 4.7103665996831275e+14},
			[]float64{1.5958230676198406e-12, 1.3254149444087839e-13, 1.3254149444087839e-13, 1.832787154117927e-12},
			[]float64{-4.072868604474522, -5.2798650466914925, 1.506223067782694, -3.1906432017664272, -3.1602666281643037, 4.3272883398748885}},
		{"draw 300", 3, 2,
			[]float64{0.7661559168999676, -16.05573255271991, -12.612692870198202, -6.703864361458822, 6.185797670805151, 9.142337960128401, -20.673685486874163, 5.852392359680119, 12.526805240901584},
			[]float64{0.40159277429521734, -0.28766229427099954, 0.398250130926652, -0.28526794725611915, 0.07737464139473818, -0.05542372344987503},
			[]float64{4.0007861432608144e+14, 3.712991235460045e+14, 5.05906104863623e+13, 3.712991235460045e+14, 1.0905247098673958e+15, 2.826659996785608e+14, 5.05906104863623e+13, 2.826659996785608e+14, 1.7708524256783622e+14},
			[]float64{1.163605868596306e-12, 4.834877394579096e-14, 4.834877394579096e-14, 1.356047755600614e-12},
			[]float64{-1.1376833923955496, -0.4243840134835313, 4.298291450895787, 7.0324833240386955, -2.79592573163756, -3.7196405540455557}},
	} {
		A, B, Q, R, S := c.matrices()
		for _, noScaling := range []bool{false, true} {
			res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, NoScaling: noScaling})
			if err != nil {
				continue
			}
			rho := closedLoopRadiusBig(t, A, B, res.K)
			if !(rho < 1) {
				t.Errorf("%s NoScaling=%v: Accuracy %d with ρ(A − BK) = %.3g", c.name, noScaling, res.Accuracy, rho)
				continue
			}
			eig := 0.0
			for _, v := range res.Eig {
				eig = math.Max(eig, cmplx.Abs(v))
			}
			if !(math.Abs(eig-rho) <= 1e-3*math.Max(rho, 1e-3)) {
				t.Errorf("%s NoScaling=%v: Eig radius %.6g, ρ(A − BK) = %.6g", c.name, noScaling, eig, rho)
			}
		}
	}
}

// Without a cross term and with rank(B) = 1 < m = 2, under cheap control,
// Newton's R + BᵀXB has lost its smallest eigenvalue, and the placed
// Hewer fallback certified X off by 2.1e-6 (oracle eps move 3e-8) over the
// pencil's Accuracy 1 X off by 1e-7 (draw 1429 of a riccati
// review's rank-deficient sweep). amd64 rounding leaves ErrNoStabilizing,
// as in base.
func TestDare_PoorPencilSolutionKeptOnRankDeficientInputs(t *testing.T) {
	c := dareInputsCase{"draw 1429", 2, 2,
		[]float64{0.2298123512449325, 0.9727822623913196, -1.9013112312358436, -3.484193360470578},
		[]float64{0.033829300778703136, 0.00348832191148471, -0.618356019029907, -0.06376202879246375},
		[]float64{2.890120015058661e+14, 1.4714160992009236e+13, 1.4714160992009236e+13, 2.1544560574160197e+14},
		[]float64{1.4472587904017713e-12, 1.9316508781847902e-13, 1.9316508781847902e-13, 1.915729608713395e-12},
		nil}
	A, B, Q, R, _ := c.matrices()
	want := dareDoublingBig(A, B, Q, R, nil)
	res, err := Dare(A, B, Q, R, nil)
	if err != nil {
		return
	}
	if rel := relErr(res.X, want); res.Accuracy == 0 && !(rel <= 1e-8) {
		t.Errorf("Accuracy 0 with ‖X − X*‖/‖X*‖ = %.3g", rel)
	}
	if rho := closedLoopRadiusBig(t, A, B, res.K); !(rho < 1) {
		t.Errorf("ρ(A − BK) = %.3g", rho)
	}
}

// TDHYKU: cheap control of strongly unstable plants whose rounded rank-1
// Q has eigenvalues at −6e-3 of ‖Q‖ = 5e14: the stabilizing X is
// indefinite and moves by 0.8 to 12 relative under eps perturbations of
// Q alone, so float64 does not determine it and the pencil's X off by 0.2
// to 1.4 must stay Accuracy 1 (precision limits, not wrong values). Plain
// and with E = 2I over 2A, 2B (the same X).
func TestDare_CheapUnstableRoundingBoundXIsNotCertified(t *testing.T) {
	for _, k := range []int{43, 49, 84, 109, 132} {
		A, B, Q, R := dareCheapUnstableCase(k)
		n, _ := A.Dims()
		want := dareDoublingBig(A, B, Q, R, nil)
		var A2, B2, E mat.Dense
		A2.Scale(2, A)
		B2.Scale(2, B)
		E.Scale(2, eye(n))
		for _, tc := range []struct {
			name string
			A, B *mat.Dense
			opts *RiccatiOpts
		}{{"plain", A, B, nil}, {"E=2I", &A2, &B2, &RiccatiOpts{E: &E}}} {
			res, err := Dare(tc.A, tc.B, Q, R, tc.opts)
			if err != nil {
				continue
			}
			if rel := relErr(res.X, want); res.Accuracy == 0 && !(rel <= 1e-8) {
				t.Errorf("k=%d %s: Accuracy 0 with ‖X − X*‖/‖X*‖ = %.3g", k, tc.name, rel)
			}
		}
	}
}
