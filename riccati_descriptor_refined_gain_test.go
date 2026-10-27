package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// descriptorRefinedCase is a descriptor plant (D = 0) with triangular or
// rotated E whose last pivot is 1e-10..1e-5, and a cross term S.
type descriptorRefinedCase struct {
	name          string
	n, m, p       int
	continuous    bool
	a, b, c, e, s []float64
}

func (dc descriptorRefinedCase) system(t *testing.T) *System {
	t.Helper()
	dt := 0.0
	if !dc.continuous {
		dt = 1
	}
	sys, err := NewDescriptor(mat.NewDense(dc.n, dc.n, dc.a), mat.NewDense(dc.n, dc.m, dc.b),
		mat.NewDense(dc.p, dc.n, dc.c), mat.NewDense(dc.p, dc.m, nil), mat.NewDense(dc.n, dc.n, dc.e), dt)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// checkRefinedGain fails when an Accuracy 0 K (and, with Y, the explicit
// solution EᵀXE) is off the 320-bit oracle of pr by more than
// descriptorGainTol; the oracle moves them by under 1e-14 under 1-ulp
// perturbations of A, B and E.
func checkRefinedGain(t *testing.T, label string, pr nearSingularRiccati, acc int, X, K *mat.Dense) {
	t.Helper()
	Y, _, Ko, ok := nearSingularOracle(pr.continuous, pr.A, pr.B, pr.Q, pr.R, pr.S, pr.E, K)
	if !ok {
		t.Fatalf("%s: oracle did not converge", label)
	}
	if acc != 0 {
		return
	}
	if e := relErr(K, Ko); e > descriptorGainTol {
		t.Errorf("%s: Accuracy 0, K off the oracle by %.3g", label, e)
	}
	if X != nil {
		if e := relErr(X, Y); e > descriptorGainTol {
			t.Errorf("%s: Accuracy 0, EᵀXE off the oracle by %.3g", label, e)
		}
	}
}

// TestDescriptorDiscreteGainRefined: for discrete E with a last pivot of
// 1e-10..1e-5, X ~ 1/σ_min(E)² and the gain of X rounded to float64 loses
// R against BᵀXB: Dare's K was off by up to 1.4e-5 (Kalman L), 3.5e-6
// (with S, rotated E), 2e-5 (Lqg L) and Dlqr's EᵀXE by 2.6e-7, each with
// Accuracy 0 against an eps-sensitivity of 1e-15 (5AFLTD). K and EᵀXE
// are now those of X refined in double-double.
func TestDescriptorDiscreteGainRefined(t *testing.T) {
	cases := []descriptorRefinedCase{
		{"triangular n4", 4, 2, 2, false,
			[]float64{-0.9731941564732303, 0.4657626505081749, -0.09184710038408159, -0.4161757559125549, 0.27760429819049975, -0.23116381912743822, 0.9372384962817603, -0.6501644249666848, -0.9543709090200962, -0.6112282449526714, -0.8577776445672457, 0.23271216646308338, -0.6592757094990589, -0.5921583001227954, -0.5232953435298945, 0.22314111263663494},
			[]float64{-1.64127180523892, -0.004915572680774027, 0.6316335330807503, 0.8427364635039963, 0.5132204390269803, 1.369223666634679, -1.320131067806111, 0.2908076627935746},
			[]float64{-1.146393452211413, -1.058533405246863, -0.3504507366079055, 0.3946576689542781, -1.7946530911247094, 0.5930610561839003, -0.07374041209750484, 0.6029329615415397},
			[]float64{1.4564697336061516, 0.2760847074978492, -0.6590461429988075, 1.4925385891780434, 0, 0.7211072318704417, 0.0475153268739302, -0.9787211290938023, 0, 0, -0.5461355672936248, -1.260044226378643, 0, 0, 0, 7.815672280813108e-06},
			[]float64{0.014936284508913895, 0.0488570691248716, -0.11019032023128385, 0.02170627745759503, 0.042599849712445084, 0.011744823964263986, 0.03398541452026597, 0.0953720004959054}},
		{"rotated n4", 4, 2, 1, false,
			[]float64{0.11213980800403327, -0.15836473259832975, -0.07589397369988791, 0.057998623799360854, 0.19943670912933256, 0.23399895960203126, 0.17537540687370368, 0.1439468342732289, 0.12618672756068286, -0.19098733475541071, 0.07125288620489378, -0.09817545021035261, 0.2120956663708031, -0.05697980135381442, -0.05559476418542871, -0.13583808721201657},
			[]float64{0.620720850226151, -0.9197212939888202, 1.5110824316966134, 0.3033370561683364, 0.5104788513336238, -0.8342549570709714, 1.3069501883758328, -1.4219410628527667},
			[]float64{0.13960187668831925, 0.4730727604014809, -0.848919248666553, -0.50324967574918},
			[]float64{-0.1714745956296777, 0.28599839473710836, -0.6543390439484057, 0.39771335450808504, 0.6642419409917744, -0.8035641979825435, 0.07619252870206897, 1.8300820098344448, 0.2648386438060673, 0.15700811095973957, 0.34894009342784477, -0.8378354834545735, -0.571261453583474, 0.7430341928012615, -1.1212181829483532, 0.04570180062774824},
			[]float64{-0.06312943946934738, 0.01962678591150422, -0.0034446675286937322, 0.09900838055424059, 0.11467311861922139, 0.0524258175953371, -0.06627217931021319, 0.11643967421034641}},
		{"triangular n2 pivot 3e-10", 2, 1, 1, false,
			[]float64{-0.041568325109354175, -0.3028268007792664, -0.6622730180664451, 0.8143562043962622},
			[]float64{-1.2184938442740778, -1.256106850451557},
			[]float64{-0.5982000052364733, -2.704946360204417},
			[]float64{-0.9894678344203465, -0.9030787745996771, 0, -3.095668172504194e-10},
			[]float64{0.03176216432810883, 0.000503288396157011}},
		{"triangular n2 pivot 1e-7", 2, 1, 2, false,
			[]float64{-0.026359001360911884, 0.6706041194970609, -0.23243726874396337, 0.0724658175198771},
			[]float64{-1.1317551833706636, 0.8299801231891104},
			[]float64{0.6887539800503351, 0.18987769515562825, 0.9534800304166203, 0.3746071832275493},
			[]float64{-0.665876116912555, -0.22407682368961446, 0, 1.0703925251677293e-07},
			[]float64{0.05053390274933036, -0.05333494429482365}},
	}
	for _, dc := range cases {
		checkDescriptorRefinedDesigns(t, dc)
	}
}

// TestDescriptorContinuousGainConverged: a certified continuous X within
// newtonTol (step 1.7e-9, state-scaled) left Lqr's K off by 9.2e-7, and
// Care with S's by 4.3e-7 and 3.8e-7, against an eps-sensitivity of
// 2e-15, with no cancellation in K or EᵀXE: the step of X, relative to
// ‖X‖ ~ 1/σ_min(E), does not bound K, which sees X in E's small
// directions (5AFLTD). K is now refined once its Newton step moves it.
func TestDescriptorContinuousGainConverged(t *testing.T) {
	cases := []descriptorRefinedCase{
		{"triangular n4 pivot 1e-9", 4, 2, 2, true,
			[]float64{0.2026037503263713, -0.5477606542908129, 0.15400289524404787, -0.7189761503434144, -0.023970271687737132, -0.3608856246388481, 0.19236101707892547, 0.6943506198427514, 1.0351531998402341, -0.2975730057830124, -0.5500341263817299, -0.9854197864255354, -0.2766267123134327, -0.34996892127828394, -0.1307737159564013, -0.1617902668540154},
			[]float64{-0.4492991670622115, 0.518738233246712, 1.2503542591990104, 0.8896926450498344, -1.7171892331749672, 1.7398785822348202, 0.3617309558155239, 1.1920631498420988},
			[]float64{0.04580097671809241, -1.6212230670186396, 0.3143199757806384, -1.7628012361594991, 1.1707190084659875, -0.4262752899307749, -0.10074253662966515, -0.02347416635236582},
			[]float64{1.0774579708967824, -0.8634845030042604, 0.28142013392212, -0.28141537008961715, 0, 0.7172934031404352, 0.09758992377750764, 0.39539621283782855, 0, 0, -0.6511041785466359, 0.4792380014852056, 0, 0, 0, 1.0937432695994265e-09},
			[]float64{-0.033944303601670105, -0.012575078029644502, -0.1543662233410482, 0.0584731866276314, -0.05649394448778982, 0.2559332835923504, 0.12696982782461788, -0.10291899977059647}},
		{"triangular n3 pivot 9e-11", 3, 1, 2, true,
			[]float64{1.3671851896230014, -1.1355801965968166, -0.8919521510639907, -0.7366840572618726, 0.26968849746132817, -0.6284196977050708, -0.7019594285746477, -0.14581074359215404, -0.055940508423691644},
			[]float64{-0.10425241400016327, 1.2117181790444769, 0.23181299530623978},
			[]float64{0.30192719218827707, -0.8547284685619629, 0.45158150158123356, 0.2518100556217165, -1.006929344718994, -1.652226526347806},
			[]float64{1.1244731351036548, -1.354683558787139, -0.8436815012365922, 0, -0.5186292472735337, -0.15478306252010832, 0, 0, -9.302182047149628e-11},
			[]float64{-0.3345130553818739, 0.17297884502095412, -0.09465859674319538}},
		{"triangular n3 pivot 7e-10", 3, 2, 2, true,
			[]float64{-0.5850691360562265, -0.5611008215345186, -0.008326950285431792, 0.627243047230733, -0.3518269086398168, -0.7689026545851658, -0.04995832522858949, -0.33038282133154545, -0.44091538394992846},
			[]float64{-0.7524905265140804, 0.3764252098190371, -1.123745281044931, 1.1294605450280533, 0.2494310783138156, 1.8924852020074245},
			[]float64{0.30693440799596794, 0.07387101771439603, 0.8926748749796538, -1.9222256541921625, 0.7637121824263071, 0.5560652606919038},
			[]float64{-0.6303472416410056, 0.5265960394690559, -1.0011934915929679, 0, 1.0719981749061422, -0.5095663089188236, 0, 0, -7.021271998313191e-10},
			[]float64{-0.03556312795469989, -0.04380392823287846, 0.11082364396971325, 0.13464513684848503, -0.07446463150817076, -0.05454859079572658}},
	}
	for _, dc := range cases {
		checkDescriptorRefinedDesigns(t, dc)
	}
}

// checkDescriptorRefinedDesigns runs Lqr (Dlqr), Care (Dare) with S,
// Kalman and Lqg's L on dc, each scaled and unscaled, with identity
// weights, and checks every certified gain (checkRefinedGain); Lqr and
// the Riccati solve with S must be certified.
func checkDescriptorRefinedDesigns(t *testing.T, dc descriptorRefinedCase) {
	t.Helper()
	sys := dc.system(t)
	n, m, p := dc.n, dc.m, dc.p
	A, B, E, S := sys.A, sys.B, sys.E, mat.NewDense(n, m, dc.s)
	In, Im := eyeDense(n), eyeDense(m)
	lqr, care := Lqr, Care
	if !dc.continuous {
		lqr, care = Dlqr, Dare
	}
	for _, ns := range []bool{false, true} {
		label := dc.name
		if ns {
			label += " NoScaling"
		}
		res, err := lqr(A, B, In, Im, &RiccatiOpts{E: E, NoScaling: ns})
		if err != nil || res.Accuracy != 0 {
			t.Fatalf("%s Lqr: not certified (%v)", label, err)
		}
		checkRefinedGain(t, label+" Lqr", nearSingularRiccati{dc.continuous, A, B, In, Im, nil, E, false}, res.Accuracy, res.X, res.K)
		res, err = care(A, B, In, Im, &RiccatiOpts{E: E, S: S, NoScaling: ns})
		if err != nil || res.Accuracy != 0 {
			t.Fatalf("%s Riccati with S: not certified (%v)", label, err)
		}
		checkRefinedGain(t, label+" Riccati with S", nearSingularRiccati{dc.continuous, A, B, In, Im, S, E, true}, res.Accuracy, nil, res.K)
		if res, err := Kalman(sys, Im, eyeDense(p), nil, &RiccatiOpts{NoScaling: ns}); err == nil {
			checkRefinedGain(t, label+" Kalman", kalmanDual(sys, sys.B, sys.D), res.Accuracy, nil, mat.DenseCopyOf(res.K.T()))
		}
	}
	if lq, err := Lqg(sys, eyeDense(n+m), eyeDense(n+p), nil); err == nil {
		checkRefinedGain(t, dc.name+" Lqg L", kalmanDual(sys, In, nil), lq.Accuracy, nil, mat.DenseCopyOf(lq.L.T()))
	}
}
