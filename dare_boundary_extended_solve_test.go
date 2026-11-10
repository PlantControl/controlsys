package controlsys

import (
	"errors"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Descriptor Dare/Dlqr with cond(E) 2e6-4e7 (below nearSingularDare's
// cond(E)·eps gate) whose float64 pencil reported an unstabilizable
// boundary mode or no stabilizing solution, while the 320-bit K of the
// explicit model gives a closed-loop radius 0.09-0.41 (IR5YOF; zw46Gen
// mid seeds 74, 85, 144, 279, 343, native literals). The returned K is
// checked against Hewer's iteration in 320 bits (nearSingularOracle) from
// it, that oracle against SDA doubling (riccatiExactReference), and the
// closed loop E⁻¹(A − BK) of the returned K by a 320-bit Stein test.
// ulpSensitive rows error on base natively only: the amd64 pencil (and
// the race build's) may solve them or keep the error. demoted rows are
// Accuracy 1 natively under gonum 30740af (HF5PTS; mid-279-cross Dlqr K
// and X exact, Dare X 1.67e-8 off): Dlqr need not certify them (owner
// 2ARMRV).
func TestDare_BoundaryVerdictExtendedSolve(t *testing.T) {
	native := nativeBoundaryPath()
	cases := []struct {
		name             string
		ulpSensitive     bool
		demoted          bool
		A, B, Q, R, S, E *mat.Dense
	}{
		{
			name: "mid-74-cross", ulpSensitive: false,
			A: mat.NewDense(5, 5, []float64{0.547016777182006, -1.4542979466635624, 0.6382572678493625, -0.7594531184774366, -0.03375180277010981, -1.0166738717390877, 0.06475899491666548, -1.175904883228698, 0.15731366478152736, -0.9346968160870506, 1.8835644239104998, 0.709495050618536, -0.23394958509984254, -0.9418998475798174, -0.017773183305342777, -0.0654842763696617, 0.4186568379368626, 1.2177975242547336, -1.685926441438724, -0.49678142568653993, 0.7346308491894282, 0.3402016907546498, 0.4963164048854082, 1.8947784963057637, 0.40457271092835145}),
			B: mat.NewDense(5, 2, []float64{-0.28865522920492603, 0.43190425987840675, -0.23269270702287564, 0.0022883782933258434, -0.48472313091925123, -0.4488781388645773, -0.14139755855569658, 0.6784695584464331, -1.417573325351182, -1.0336104572218072}),
			Q: mat.NewDense(5, 5, []float64{1.5783769752535581, -0.16310711458805957, -0.9103940695137615, 1.2988162589120837, -0.01887809405297218, -0.16310711458805957, 1.045997562087563, 0.2567386949512864, -0.36627698099208666, 0.005323779440826481, -0.9103940695137615, 0.2567386949512864, 2.433005457111216, -2.0444012643885707, 0.029715057142470767, 1.2988162589120837, -0.36627698099208666, -2.0444012643885707, 3.916650811825351, -0.042393069818387356, -0.01887809405297218, 0.005323779440826481, 0.029715057142470767, -0.042393069818387356, 1.0006161767330324}),
			R: mat.NewDense(2, 2, []float64{1.6465300369846037, 0.3180803234317257, 0.3180803234317257, 1.1564893916241057}),
			S: mat.NewDense(5, 2, []float64{-0.6115047728364237, -0.3008485681980551, 0.17244942887024495, 0.08484179693567763, 0.962538867363419, 0.473550580379074, -1.3732087813732545, -0.6755922669202953, 0.019959377896027985, 0.009819629427079202}),
			E: mat.NewDense(5, 5, []float64{0.29894104637061913, -0.028546944220571614, -0.6193496262831136, -0.14741392978921794, 0.7956014826582123, -0.3728899560163743, -0.28190645315503027, 0.4074199565757989, 0.6474765122631011, 0.27142624271905, 0.25238306781918146, 0.04337270479202771, -0.015851483042786236, -0.6492663756796475, -0.8533579028675375, 0.4903951242239574, -0.2560467216434383, 0.3376774790839015, 0.4485205940221198, -0.2502524286160035, -0.19518716461591987, -0.8290381058483789, 0.32445444774933135, 0.04515254474294859, -0.226987078868158}),
		},
		{
			name: "mid-85-cross", ulpSensitive: false,
			A: mat.NewDense(2, 2, []float64{1.0596625396702297, 0.4519569829385703, -0.6513319386306535, -1.9810332929168784}),
			B: mat.NewDense(2, 2, []float64{-0.1636986889048983, -0.7676224253271123, -2.2433395437117265, 2.575287134620651}),
			Q: mat.NewDense(2, 2, []float64{7.856481631560513, -3.2452493276788226, -3.2452493276788226, 8.113676225088883}),
			R: mat.NewDense(2, 2, []float64{3.247662912032643, -2.1654952719405163, -2.1654952719405163, 2.469682828283582}),
			S: mat.NewDense(2, 2, []float64{-2.3766118908897984, 1.8566378338922267, 2.861016817419372, -2.2350607968362817}),
			E: mat.NewDense(2, 2, []float64{0.7246350543397122, -0.7542339544330663, 0.6658573764177139, -0.6930552783575629}),
		},
		{
			name: "mid-144-nocross", ulpSensitive: false,
			A: mat.NewDense(4, 4, []float64{0.36867201830821594, 0.7641136720369576, -0.6711759916290065, 0.25608866677862874, 0.6808212536199475, -0.13577419480318714, -0.16326862398183706, 1.2898495072384715, 0.1506552235274485, 0.5386308192066966, 0.25708537773274975, 1.5609939664908143, -0.900974575546885, -0.2196397800320895, -0.22769119782260205, 0.11700133290996084}),
			B: mat.NewDense(4, 1, []float64{0.3158847948586132, -0.8333709431523175, 0.02656667910019006, -0.3891169097144075}),
			Q: mat.NewDense(4, 4, []float64{3.6506086347896467, -0.9200178639155153, -2.6181746693095054, 2.6099499169672646, -0.9200178639155153, 4.575608858342078, 1.203618967800732, 3.253910369209572, -2.6181746693095054, 1.203618967800732, 2.224384340097545, -2.0751391970468207, 2.6099499169672646, 3.253910369209572, -2.0751391970468207, 7.268443521517432}),
			R: mat.NewDense(1, 1, []float64{1.1476437393214118}),
			E: mat.NewDense(4, 4, []float64{-0.3065793535120014, 1.008827979321256, 0.42469758902170346, 0.20199722915539825, -0.45124961627813603, 0.03466015680448618, -0.29406111220434566, -0.43844788466632745, -0.22753349474013698, -0.32041489239108245, 0.27218078149727276, -0.3167005270475097, -0.5446942033177934, -0.17631103409741658, 0.19287181319735117, -0.5579606893154194}),
		},
		{
			name: "mid-144-cross", ulpSensitive: true,
			A: mat.NewDense(4, 4, []float64{0.36867201830821594, 0.7641136720369576, -0.6711759916290065, 0.25608866677862874, 0.6808212536199475, -0.13577419480318714, -0.16326862398183706, 1.2898495072384715, 0.1506552235274485, 0.5386308192066966, 0.25708537773274975, 1.5609939664908143, -0.900974575546885, -0.2196397800320895, -0.22769119782260205, 0.11700133290996084}),
			B: mat.NewDense(4, 1, []float64{0.3158847948586132, -0.8333709431523175, 0.02656667910019006, -0.3891169097144075}),
			Q: mat.NewDense(4, 4, []float64{3.7692825644778853, -1.2444285206585282, -2.350439198901379, 2.550882867138978, -1.2444285206585282, 5.462427677647284, 0.4717291583314188, 3.4153778497000724, -2.350439198901379, 0.4717291583314188, 2.828411545592239, -2.2083979878607884, 2.550882867138978, 3.4153778497000724, -2.2083979878607884, 7.297842702825263}),
			R: mat.NewDense(1, 1, []float64{1.3371449242511793}),
			S: mat.NewDense(4, 1, []float64{-0.14996282971521027, 0.4099429436840618, -0.33832509686847156, 0.07464033556863876}),
			E: mat.NewDense(4, 4, []float64{-0.3065793535120014, 1.008827979321256, 0.42469758902170346, 0.20199722915539825, -0.45124961627813603, 0.03466015680448618, -0.29406111220434566, -0.43844788466632745, -0.22753349474013698, -0.32041489239108245, 0.27218078149727276, -0.3167005270475097, -0.5446942033177934, -0.17631103409741658, 0.19287181319735117, -0.5579606893154194}),
		},
		{
			name: "mid-279-cross", ulpSensitive: true, demoted: true,
			A: mat.NewDense(5, 5, []float64{0.3345853680432942, -0.5444275521734381, 0.12170526714347651, 0.09496216282048281, 0.8698082007052628, -1.1832050021928828, -1.9769313652682223, 0.236925633588843, -1.0957790402457945, 0.4892996711553904, 0.630845571893818, -0.3195721156194128, -0.07693463896235497, -2.0497543274498478, -0.6806304034766936, 0.3851019339290179, -1.0385704600377679, 1.1774717763528888, 0.3633037140769124, -0.16830799934803908, -0.40538440193130365, -0.8487856726526775, 0.8766725432015005, 0.5629758043597742, 0.11977255729700625}),
			B: mat.NewDense(5, 2, []float64{0.4369270913557226, 1.320204430389531, -0.18344196673917534, -2.065044438827403, 1.9291060463927412, 1.2581787847157724, -0.12266320499603922, 1.9419573218872788, 0.17494976812979202, -0.5253548996252111}),
			Q: mat.NewDense(5, 5, []float64{1.4763330089923503, -0.7712926345378776, -0.20399808391151153, 0.07406185720162367, -0.7023591143098131, -0.7712926345378776, 3.817032944510636, -2.99160684332307, -1.2172015416451683, 2.8380768943850705, -0.20399808391151153, -2.99160684332307, 5.148645710885679, -0.17878575596093133, -2.5355952343338473, 0.07406185720162367, -1.2172015416451683, -0.17878575596093133, 2.776230392166753, -1.9418643175400545, -0.7023591143098131, 2.8380768943850705, -2.5355952343338473, -1.9418643175400545, 3.5334753523040856}),
			R: mat.NewDense(2, 2, []float64{0.9463892587710665, -0.30303258645510395, -0.30303258645510395, 1.0626175873291344}),
			S: mat.NewDense(5, 2, []float64{-0.11915549524967103, -0.4317607507250937, 0.12360690164069672, 0.44789045218069684, -0.0713712544301877, -0.2586142278071274, 0.03051731911910841, 0.11057971422446737, 0.0836407712006444, 0.3030729056108365}),
			E: mat.NewDense(5, 5, []float64{0.7834891481105034, 0.5114577525336104, 0.8428506922891332, -0.11132261421448142, -0.6411015997551006, 0.0, 0.6792452584555361, 0.38630006455952237, -1.6588834932557006, 0.34288125545743353, 0.0, 0.0, 0.8684111895432448, 0.8266077595095385, 1.1278545073622621, 0.0, 0.0, 0.0, 1.483960814652408, 0.11464828898632096, 0.0, 0.0, 0.0, 0.0, 4.2106157508275955e-07}),
		},
		{
			name: "mid-343-nocross", ulpSensitive: true,
			A: mat.NewDense(5, 5, []float64{-1.378777822275217, 1.4880382951775033, 0.5213760131976114, -2.19865029286236, 0.32701044103069066, 0.11036025987173637, 0.32855771991142824, -0.7717666423252667, 0.5930210280323902, 2.5959819352100872, -0.3007535867779332, -2.1810864641017, -0.5574470486533549, -0.5375814021922949, 1.4609324060556532, -0.4803034611056578, 0.6012080949423061, 0.5037630830321271, -0.6623969655244484, 0.4935019348213662, -0.8950981285692134, 0.49380023768730824, 0.2946287552743015, -0.16481322620301553, -2.2878024714284373}),
			B: mat.NewDense(5, 1, []float64{-0.6943564982944932, -1.4044386714256953, 0.7619783708849877, -0.2354482329757035, -0.8792132751773707}),
			Q: mat.NewDense(5, 5, []float64{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0}),
			R: mat.NewDense(1, 1, []float64{1.0}),
			E: mat.NewDense(5, 5, []float64{0.7962226854888659, 0.167667911419328, -0.10847652666971282, 0.216576900602753, -0.27798599933069124, 0.36083265397176073, -0.32497558480508676, 0.15639531694168213, -0.3807749414544276, -0.7535497346028412, 0.6131681585519182, -0.09124744758136169, 0.8325699754933359, -0.6250571979018031, 0.22546382890485436, -0.36365320337368123, 0.13783739995284633, 0.8096970397408122, 0.35215329558176633, -0.15519203031183093, -0.6813737673954084, -0.3533056826108905, 0.4715928563044987, -0.2726572589946339, -0.4790927640787879}),
		},
	}
	for _, c := range cases {
		n, m := c.B.Dims()
		ref := riccatiExactReference(false, c.A, c.B, c.Q, c.R, c.S, c.E, riccatiExactPrec)
		if !ref.OK {
			t.Fatalf("%s: doubling reference not certified", c.name)
		}
		md, ok := xbExplicitModel(c.A, c.B, c.Q, c.R, c.S, c.E, riccatiExactPrec)
		if !ok {
			t.Fatalf("%s: explicit model", c.name)
		}
		for _, ns := range []bool{false, true} {
			for _, api := range []string{"Dare", "Dlqr"} {
				o := &RiccatiOpts{E: c.E, S: c.S, NoScaling: ns}
				solve := Dare
				if api == "Dlqr" {
					solve = Dlqr
				}
				res, err := acceptUnverified(solve(c.A, c.B, c.Q, c.R, o))
				if err != nil {
					if c.ulpSensitive && !native && riccatiNoSolution(err) {
						t.Logf("%s %s NoScaling %v: %v, as on base off native", c.name, api, ns, err)
						continue
					}
					t.Fatalf("%s %s NoScaling %v: %v", c.name, api, ns, err)
				}
				Y, X, K, ok := nearSingularOracle(false, c.A, c.B, c.Q, c.R, c.S, c.E, res.K)
				if !ok {
					t.Fatalf("%s %s NoScaling %v: Hewer oracle does not converge from K", c.name, api, ns)
				}
				if d := relErr(K, ref.K); !(d <= 1e-12) {
					t.Fatalf("%s: Hewer and doubling K differ by %g", c.name, d)
				}
				acl := xbAxpy(md.a, -1, xbMul(md.b, xbFrom(res.K, m, n, riccatiExactPrec)))
				if !xbStable(false, acl) {
					t.Errorf("%s %s NoScaling %v: closed loop of K not Schur stable", c.name, api, ns)
				}
				want := X
				if api == "Dlqr" {
					want = Y
				}
				eK, eX := relErr(res.K, K), relErr(res.X, want)
				t.Logf("%s %s NoScaling %v: Accuracy %d, K %.3g X %.3g", c.name, api, ns, res.accuracy, eK, eX)
				if native && api == "Dlqr" && !c.demoted && res.accuracy != 0 {
					t.Errorf("%s Dlqr NoScaling %v: Accuracy %d, want 0", c.name, ns, res.accuracy)
				}
				if res.accuracy == 0 && !(eK <= newtonTol && eX <= newtonTol) {
					t.Errorf("%s %s NoScaling %v: Accuracy 0 with K %g X %g off", c.name, api, ns, eK, eX)
				}
			}
		}
	}
}

// The exact problem is stabilizable: at the fast mode λ = −2.31e12 the
// PBH matrix in μ = 1/λ has σmin 1.08e-6 against a first-order bound of
// 1.6e-14. The float64-rounded 320-bit K leaves the exact closed loop
// unstable (radius 14.5, cond(E) 5e12), so no float64 gain is known to
// stabilize: a solve failure, ErrNotConverged, not ErrNoStabilizing
// (r2b seed 7; IR5YOF pinned ErrNoStabilizing, reversed by AMDL7W).
func TestDare_BoundaryStabilizableWithoutFloatGainNotConverged(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.80066813905003553, 0.98643646743257241, -0.4737784785606195, -1.2835232191665173, 0.018523975266165071, 0.27232982656055116, -0.062319964509892009, 0.14571578944403996, 0.046527271400114353})
	B := mat.NewDense(3, 1, []float64{0.43999487561269301, 1.3437679189278851, -0.58207806446117882})
	Q := mat.NewDense(3, 3, []float64{1.1127759251430753, -2.0034657720335778, 2.0937075403107763, -2.0034657720335778, 4.2950150759232431, -3.9337439661370484, 2.0937075403107763, -3.9337439661370484, 3.978541771270097})
	R := mat.NewDense(1, 1, []float64{1.2949968976633907e-05})
	E := mat.NewDense(3, 3, []float64{0.78396477701977396, -0.19454394273016359, -0.10419933019111288, 0.8326616938748872, -0.20662746874186796, -0.11067284010627185, 0.83918960182650348, -0.20825103184184285, -0.11153577611361126})
	for _, ns := range []bool{false, true} {
		o := &RiccatiOpts{E: E, NoScaling: ns}
		if _, err := Dare(A, B, Q, R, o); !errors.Is(err, ErrNotConverged) {
			t.Errorf("Dare NoScaling %v: err %v, want ErrNotConverged", ns, err)
		}
		if _, err := Dlqr(A, B, Q, R, o); !errors.Is(err, ErrNotConverged) {
			t.Errorf("Dlqr NoScaling %v: err %v, want ErrNotConverged", ns, err)
		}
	}
}

// A descriptor Dare whose float64 pencil returns an uncertified X runs the
// extended solve on it (JGGEN2; IR5YOF mid-85-cross, cond(E) ~1e2): once
// the homogeneous PBH test no longer reports a boundary mode, the pencil
// leaves K 0.2 off the 320-bit gain at Accuracy 1, and Dlqr returns the
// extended solve's K and Y at Accuracy 0. With that solve capped the
// pencil's value stays, at Accuracy 1.
func TestDlqr_UncertifiedPencilRunsExtendedSolve(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{1.0596625396702297, 0.4519569829385703, -0.6513319386306535, -1.9810332929168784})
	B := mat.NewDense(2, 2, []float64{-0.1636986889048983, -0.7676224253271123, -2.2433395437117265, 2.575287134620651})
	Q := mat.NewDense(2, 2, []float64{7.856481631560513, -3.2452493276788226, -3.2452493276788226, 8.113676225088883})
	R := mat.NewDense(2, 2, []float64{3.247662912032643, -2.1654952719405163, -2.1654952719405163, 2.469682828283582})
	S := mat.NewDense(2, 2, []float64{-2.3766118908897984, 1.8566378338922267, 2.861016817419372, -2.2350607968362817})
	E := mat.NewDense(2, 2, []float64{0.7246350543397122, -0.7542339544330663, 0.6658573764177139, -0.6930552783575629})
	ref := riccatiExactReference(false, A, B, Q, R, S, E, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("doubling reference not certified")
	}
	for _, ns := range []bool{false, true} {
		o := &RiccatiOpts{E: E, S: S, NoScaling: ns}
		res, err := acceptUnverified(Dlqr(A, B, Q, R, o))
		if err != nil {
			t.Fatalf("Dlqr NoScaling %v: %v", ns, err)
		}
		eK, eY := relErr(res.K, ref.K), relErr(res.X, ref.Y)
		if res.accuracy != 0 || !(eK <= newtonTol && eY <= newtonTol) {
			t.Errorf("Dlqr NoScaling %v: Accuracy %d, K %.3g Y %.3g off", ns, res.accuracy, eK, eY)
		}
	}
	withIterationCap(t, &extendedDoublings, 2)
	for _, ns := range []bool{false, true} {
		o := &RiccatiOpts{E: E, S: S, NoScaling: ns}
		for _, api := range []string{"Dare", "Dlqr"} {
			solve := Dare
			if api == "Dlqr" {
				solve = Dlqr
			}
			res, err := acceptUnverified(solve(A, B, Q, R, o))
			if err != nil {
				t.Errorf("capped %s NoScaling %v: %v", api, ns, err)
				continue
			}
			if eK := relErr(res.K, ref.K); res.accuracy != 1 {
				t.Errorf("capped %s NoScaling %v: Accuracy %d (K %.3g off), want 1", api, ns, res.accuracy, eK)
			}
		}
	}
}

// JGGEN2 census rv46 edge 651 (native): the homogeneous PBH lets the
// pencil return X at Accuracy 0 with a loose refined EᵀXE (xeLoose), which
// Dlqr demoted to Accuracy 1 with X 3.9e-10 off. On base the vacuous PBH
// error sent it to the extended solve, whose certified Y Dlqr returned at
// Accuracy 0; a loose EᵀXE now runs that solve too. Off native it is an
// uncertified Accuracy 1, Y 0.11 off, as on base.
func TestDlqr_LooseExplicitSolutionRunsExtendedSolve(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{0.49064334409837296, -1.2630124986636051, 0.40991957396117545, 0.088641202660148924, 0.086357067712252156, -1.1146996452648377, 0.56944014566904322, 0.15795768461564391, -1.5277682013893639, -0.97213688341209525, 0.83126275735761923, -0.14821552239455368, 1.4912555327629049, -1.5542383807766351, -0.29372145618026019, 3.6433822518335832, -0.16421262927291874, 0.84432412335165097, -0.11802469634704317, 0.6984452915707684, -1.6998115128340563, 0.86924486592105088, 0.86372912944498093, -0.29478536619532253, 1.3430728636376159})
	B := mat.NewDense(5, 1, []float64{-1.1002279700680406, -1.133485809284837, -0.49701898768160857, -1.6850037544281669, 0.50344277858901432})
	Q := mat.NewDense(5, 5, []float64{3.6929075820887833, 0.56059583387044998, 0.35412136715499881, 2.8949887596086277, -2.8715519020834943, 0.56059583387044998, 6.9406760712811719, 3.5882402721815332, -3.1121356690817588, -1.1171793498728189, 0.35412136715499881, 3.5882402721815332, 2.8816821501968146, -0.46129779150349537, 0.51225179865283921, 2.8949887596086277, -3.1121356690817588, -0.46129779150349537, 7.0007396211055548, -1.689255443916053, -2.8715519020834943, -1.1171793498728189, 0.51225179865283921, -1.689255443916053, 4.176667572850091})
	R := mat.NewDense(1, 1, []float64{0.56614162451301342})
	E := mat.NewDense(5, 5, []float64{0.027640371106685199, -0.15835289668947367, 0.085081513234009545, 0.61239141396797292, -0.14122858328989185, -0.05415905770040004, -0.2987425768626355, 0.31636277775924954, -0.85743457835532799, -1.027338049485206, -0.31863286245762129, -0.073586027898234277, 0.41391183065911058, 0.12033842160115096, 0.75477942224586747, -0.2766137077530737, -0.11049355381654127, 0.78854145366571538, -0.096248754516484084, 0.031850224977045864, -0.14473648162328295, -1.1259999586917229, -0.96269119502210687, 0.30440820780531475, -0.074833515497742323})
	ref := riccatiExactReference(false, A, B, Q, R, nil, E, 320)
	if !ref.OK {
		t.Fatal("320-bit reference failed")
	}
	res, err := acceptUnverified(Dlqr(A, B, Q, R, &RiccatiOpts{E: E}))
	if err != nil {
		t.Fatal(err)
	}
	if res.accuracy != 0 && !nativeBoundaryPath() {
		t.Logf("Accuracy %d, as on base off native", res.accuracy)
		return
	}
	if res.accuracy != 0 {
		t.Errorf("Accuracy = %d, want 0", res.accuracy)
	}
	if d := relFrob(res.X, ref.Y); d > 1e-12 {
		t.Errorf("X %.3g off the 320-bit EᵀXE", d)
	}
	if d := relFrob(res.K, ref.K); d > 1e-12 {
		t.Errorf("K %.3g off the 320-bit gain", d)
	}
}

// The loose-EᵀXE path returns Y with the pencil K: Dlqr's K agrees with
// the 320-bit gain of the 320-bit Y, and Dare keeps the pencil X and K.
func TestDlqr_LooseExplicitSolutionGainPairAgrees(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{0.49064334409837296, -1.2630124986636051, 0.40991957396117545, 0.088641202660148924, 0.086357067712252156, -1.1146996452648377, 0.56944014566904322, 0.15795768461564391, -1.5277682013893639, -0.97213688341209525, 0.83126275735761923, -0.14821552239455368, 1.4912555327629049, -1.5542383807766351, -0.29372145618026019, 3.6433822518335832, -0.16421262927291874, 0.84432412335165097, -0.11802469634704317, 0.6984452915707684, -1.6998115128340563, 0.86924486592105088, 0.86372912944498093, -0.29478536619532253, 1.3430728636376159})
	B := mat.NewDense(5, 1, []float64{-1.1002279700680406, -1.133485809284837, -0.49701898768160857, -1.6850037544281669, 0.50344277858901432})
	Q := mat.NewDense(5, 5, []float64{3.6929075820887833, 0.56059583387044998, 0.35412136715499881, 2.8949887596086277, -2.8715519020834943, 0.56059583387044998, 6.9406760712811719, 3.5882402721815332, -3.1121356690817588, -1.1171793498728189, 0.35412136715499881, 3.5882402721815332, 2.8816821501968146, -0.46129779150349537, 0.51225179865283921, 2.8949887596086277, -3.1121356690817588, -0.46129779150349537, 7.0007396211055548, -1.689255443916053, -2.8715519020834943, -1.1171793498728189, 0.51225179865283921, -1.689255443916053, 4.176667572850091})
	R := mat.NewDense(1, 1, []float64{0.56614162451301342})
	E := mat.NewDense(5, 5, []float64{0.027640371106685199, -0.15835289668947367, 0.085081513234009545, 0.61239141396797292, -0.14122858328989185, -0.05415905770040004, -0.2987425768626355, 0.31636277775924954, -0.85743457835532799, -1.027338049485206, -0.31863286245762129, -0.073586027898234277, 0.41391183065911058, 0.12033842160115096, 0.75477942224586747, -0.2766137077530737, -0.11049355381654127, 0.78854145366571538, -0.096248754516484084, 0.031850224977045864, -0.14473648162328295, -1.1259999586917229, -0.96269119502210687, 0.30440820780531475, -0.074833515497742323})
	ref := riccatiExactReference(false, A, B, Q, R, nil, E, 320)
	if !ref.OK {
		t.Fatal("320-bit reference failed")
	}
	opts := &RiccatiOpts{E: E}
	dare, err := acceptUnverified(Dare(A, B, Q, R, opts))
	if err != nil {
		t.Fatal(err)
	}
	dlqr, err := acceptUnverified(Dlqr(A, B, Q, R, opts))
	if err != nil {
		t.Fatal(err)
	}
	if dlqr.accuracy == 0 || nativeBoundaryPath() {
		if d := relFrob(dlqr.K, ref.K); d > newtonTol {
			t.Errorf("Dlqr K %.3g off the gain of the 320-bit Y", d)
		}
		if d := relFrob(dlqr.X, ref.Y); d > newtonTol {
			t.Errorf("Dlqr Y %.3g off the 320-bit Y", d)
		}
	}
	if !mat.Equal(dare.K, dlqr.K) {
		t.Error("Dlqr K is not Dare's K")
	}
	if d := relFrob(dare.X, ref.X); d > newtonTol {
		t.Errorf("Dare X %.3g off the 320-bit X", d)
	}
}

// JGGEN2 census rv46 bnd 806 (cond(E) 2e12): the c = 1 PBH verdict sent it
// to nearSingularDare, which returned Y and K exact at Accuracy 1. The
// homogeneous verdict clears it, and the float64 pencil's uncertified Y
// came back 5.8e-3 off (bnd 461 on amd64: K 0.92 off). An uncertified
// result that the unscaled verdict marks runs the extended solve again.
// Off native the verdict does not fire and Y is 1.3e-2 off at Accuracy 1,
// as on base.
func TestDlqr_UncertifiedPencilAtIllConditionedERunsNearSingularSolve(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-0.56014363633412578, -0.77081107545778782, 1.2428497489887969, -0.6043037583739419})
	B := mat.NewDense(2, 1, []float64{0.018960481587139557, 0.096409661854527248})
	Q := mat.NewDense(2, 2, []float64{0.304740285002572, -0.17024434730499005, -0.17024434730499005, 2.6702839412058812})
	R := mat.NewDense(1, 1, []float64{5.2957243890862946})
	S := mat.NewDense(2, 1, []float64{-1.0855947835212159, 1.3867386019933887})
	E := mat.NewDense(2, 2, []float64{0.7746097390631389, 0.62768785119841863, -0.15233922970882519, -0.12344472180811875})
	ref := riccatiExactReference(false, A, B, Q, R, S, E, 320)
	if !ref.OK {
		t.Fatal("320-bit reference failed")
	}
	for _, ns := range []bool{false, true} {
		o := &RiccatiOpts{E: E, S: S, NoScaling: ns}
		res, err := acceptUnverified(Dlqr(A, B, Q, R, o))
		if err != nil {
			t.Fatalf("Dlqr NoScaling %v: %v", ns, err)
		}
		if res.accuracy != 0 && !nativeBoundaryPath() {
			t.Logf("Dlqr NoScaling %v: Accuracy %d, as on base off native", ns, res.accuracy)
			continue
		}
		if d := relFrob(res.X, ref.Y); d > 1e-12 {
			t.Errorf("Dlqr NoScaling %v: Y %.3g off the 320-bit Y", ns, d)
		}
		if d := relFrob(res.K, ref.K); d > 1e-12 {
			t.Errorf("Dlqr NoScaling %v: K %.3g off the 320-bit K", ns, d)
		}
	}
}

// nativeBoundaryPath reports the arm64 assembly backend, on which the
// float64 boundary cases above take the path the census recorded; other
// backends round them differently, as base did.
func nativeBoundaryPath() bool { return runtime.GOARCH == "arm64" && !raceEnabled }

// JGGEN2 census zw46mid 1914 (ρ 0.44, cond(E) 6e7): one scaling mode
// returns a poor X at Accuracy 1 and the other fails with the unscaled PBH
// test finding a deficient boundary mode. Base returned
// errBoundaryUnstabilizable there and solved in extended precision; with
// the verdict of the failed mode dropped, K was 2.4e-2 off the 320-bit
// gain at Accuracy 1.
func TestDare_FailedScalingModeUnscaledVerdictRunsNearSingularSolve(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{-0.54348738031290011, 0.73334636203808978, 0.6174557139527429, -0.37581976178648724, -0.41091170547645839, -0.34716981371260003, 0.4441338615012369, 0.9886260491643406, 0.33020600679380707, -0.92080378330119306, 1.4476779950508001, 0.76227906195847961, 0.53169648867224717, -0.7937527917268975, 0.31837611855391235, -0.48537929865183682})
	B := mat.NewDense(4, 2, []float64{-1.723307731978911, 1.6266153081824966, 0.070490066743343904, -0.45244914096442335, -0.87500718471444972, 0.31628519843089842, -0.27495354852120735, 0.36675946348392008})
	Q := mat.NewDense(4, 4, []float64{5.6143766518454985, 1.6624844435730595, -3.0566066369931542, 1.6877016024183351, 1.6624844435730595, 3.5854945506704872, -2.0298955813810555, 1.1124750510481676, -3.0566066369931542, -2.0298955813810555, 3.1534381916652143, -1.2751466505197664, 1.6877016024183351, 1.1124750510481676, -1.2751466505197664, 3.1121036670748712})
	R := mat.NewDense(2, 2, []float64{4.1431713041737277, -3.697266847566655, -3.697266847566655, 4.2987493708638453})
	E := mat.NewDense(4, 4, []float64{1.3115823857064357, -1.1282254549941944, 0.50437606518754907, 0.28861798113947756, 0, 0.79367692749015561, -0.53616079846115006, -0.14128752481883783, 0, 0, 1.4755648187283641, 1.1197288576143001, 0, 0, 0, 5.2266775494417855e-08})
	ref := riccatiExactReference(false, A, B, Q, R, nil, E, 320)
	if !ref.OK {
		t.Fatal("320-bit reference failed")
	}
	o := &RiccatiOpts{E: E}
	for _, api := range []string{"Dare", "Dlqr"} {
		var res *RiccatiResult
		var err error
		if api == "Dare" {
			res, err = acceptUnverified(Dare(A, B, Q, R, o))
		} else {
			res, err = acceptUnverified(Dlqr(A, B, Q, R, o))
		}
		if err != nil {
			t.Fatalf("%s: %v", api, err)
		}
		if res.accuracy != 0 {
			if nativeBoundaryPath() {
				t.Errorf("%s: Accuracy %d, want 0", api, res.accuracy)
			} else {
				t.Logf("%s: Accuracy %d off native", api, res.accuracy)
			}
			continue
		}
		if d := relFrob(res.K, ref.K); d > 1e-12 {
			t.Errorf("%s: K %.3g off the 320-bit K", api, d)
		}
	}
}
