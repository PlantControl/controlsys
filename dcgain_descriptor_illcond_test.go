package controlsys

import (
	"errors"
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func descriptorFromTriplets(t *testing.T, n int, anz, enz [][3]float64, b, c []float64, d, dt float64) *System {
	t.Helper()
	a, e := make([]float64, n*n), make([]float64, n*n)
	for _, v := range anz {
		a[int(v[0])*n+int(v[1])] = v[2]
	}
	for _, v := range enz {
		e[int(v[0])*n+int(v[1])] = v[2]
	}
	return descriptorFromSlices(t, n, 1, 1, a, b, c, []float64{d}, e, dt)
}

// KQVXXL: a descriptor whose pencil −A is ill-conditioned at DC fell
// through to the evaluation route and returned −22587.738920778414 with a
// nil error (OECG2I census band-series2/er 20.1, E = I + U/2 non-diagonal,
// A non-symmetric). The exact solve of the pencil decides it.
func TestDCGainDescriptorIllConditionedPencilExactGain(t *testing.T) {
	a := []float64{-0.062321557222830004, -33.040195914735044, 0.14584494763772893, -0.8645644279475685, 0.7856601555390705, -2.3836271301680236, 0.01052212111673384, -0.06237248809154187, 0.1469909311031274, 7.260349861110233, -0.032072380384298604, 0.18998193111713915, -0.07093960360766827, -222.716468841552, 0.9831418977233473, -5.827832769981412}
	b := []float64{1.1730845187417032, -0.10282942573771142, -0.06918628559469314, -0.06616129741221433}
	c := []float64{0, 38.21600200828384, 0, 1}
	e := []float64{1.1730845187417032, -0.062321557222830004, 0.18487822970748397, 0.14584494763772893, -0.10282942573771142, 0.7856601555390705, -0.0022196380548205608, 0.01052212111673384, -0.06918628559469314, 0.1469909311031274, 0.9098939661850487, -0.032072380384298604, -0.06616129741221433, -0.07093960360766827, -0.10829997596135021, 0.9831418977233473}
	sys := descriptorFromSlices(t, 4, 1, 1, a, b, c, []float64{0}, e, 0)
	if r := solveDCRegular(nil, sys.A, sys.B, sys.C, sys.D, 0, false); r.solveErr == nil {
		t.Fatal("−A is not ill-conditioned in double; the case misses the fall-through")
	}
	const want = -6577.782959800696
	if exact := dcGainExactFloat(t, sys); exact != want {
		t.Fatalf("exact gain %v, literal %v", exact, want)
	}
	g, err := sys.DCGain()
	if err != nil || g.At(0, 0) != want {
		t.Fatalf("DCGain = %v, %v; want %v, nil", g, err, want)
	}
}

// kqUnprovenPastExactCheck is an order-17 continuous descriptor, A
// non-symmetric and E non-diagonal: a nearly singular 2×2 block (cond ≈
// 2e14) coupled into a well-conditioned 15-state band. delta is the
// block's distance from singularity, 2⁻⁴⁴.
func kqUnprovenPastExactCheck(t *testing.T, delta float64) *System {
	anz := [][3]float64{{0, 0, -0.7}, {0, 1, -0.3}, {0, 2, 0.125}, {1, 0, -0.91}, {1, 1, -0.39 - delta}, {1, 3, 0.125}, {2, 2, -2}, {2, 3, 0.14116061834875884}, {3, 3, -3}, {3, 4, -0.19319112188899693}, {4, 2, 0.031113605876765864}, {4, 4, -4}, {4, 5, -0.17496867189838558}, {5, 3, 0.2498273471869794}, {5, 5, -2}, {5, 6, 0.16257196003927923}, {6, 4, 0.012605671951702806}, {6, 6, -3}, {6, 7, 0.20341843437677634}, {7, 5, -0.24762138022428912}, {7, 7, -4}, {7, 8, -0.1269741475976563}, {8, 6, -0.055938910046699104}, {8, 8, -2}, {8, 9, -0.22563865205254635}, {9, 7, 0.23783218469669568}, {9, 9, -3}, {9, 10, 0.0874878422391672}, {10, 8, 0.09755905882698654}, {10, 10, -4}, {10, 11, 0.24094884657102195}, {11, 9, -0.2207595477513547}, {11, 11, -2}, {11, 12, -0.0453222839674237}, {12, 10, -0.13619153085257626}, {12, 12, -3}, {12, 13, -0.24888015411962885}, {13, 11, 0.19692630674602946}, {13, 13, -4}, {13, 14, 0.0017687629999827343}, {14, 12, 0.17065323415810107}, {14, 14, -2}, {14, 15, 0.24918968404851644}, {15, 13, -0.1670623377270551}, {15, 15, -3}, {15, 16, 0.041838925075701736}, {16, 14, -0.19988880360236977}, {16, 16, -4}}
	enz := [][3]float64{{0, 0, 1}, {0, 1, 0.25}, {1, 0, -0.125}, {1, 1, 1}, {2, 2, 1}, {2, 3, 0.03529015458718971}, {3, 3, 1}, {3, 4, -0.04829778047224923}, {4, 2, 0.007778401469191466}, {4, 4, 1}, {4, 5, -0.043742167974596395}, {5, 3, 0.06245683679674485}, {5, 5, 1}, {5, 6, 0.04064299000981981}, {6, 4, 0.0031514179879257014}, {6, 6, 1}, {6, 7, 0.050854608594194085}, {7, 5, -0.06190534505607228}, {7, 7, 1}, {7, 8, -0.031743536899414074}, {8, 6, -0.013984727511674776}, {8, 8, 1}, {8, 9, -0.05640966301313659}, {9, 7, 0.05945804617417392}, {9, 9, 1}, {9, 10, 0.0218719605597918}, {10, 8, 0.024389764706746635}, {10, 10, 1}, {10, 11, 0.06023721164275549}, {11, 9, -0.055189886937838675}, {11, 11, 1}, {11, 12, -0.011330570991855925}, {12, 10, -0.034047882713144066}, {12, 12, 1}, {12, 13, -0.062220038529907214}, {13, 11, 0.049231576686507364}, {13, 13, 1}, {13, 14, 0.0004421907499956836}, {14, 12, 0.04266330853952527}, {14, 14, 1}, {14, 15, 0.06229742101212911}, {15, 13, -0.04176558443176377}, {15, 15, 1}, {15, 16, 0.010459731268925434}, {16, 14, -0.04997220090059244}, {16, 16, 1}}
	b := []float64{1, 0.3, 0.955336489125606, 0.8253356149096783, 0.6216099682706646, 0.3623577544766736, 0.0707372016677029, -0.2272020946930869, -0.5048461045998575, -0.7373937155412454, -0.904072142017061, -0.9899924966004454, -0.987479769908865, -0.8967584163341471, -0.7259323042001402, -0.4902608213406995, -0.21079579943077972}
	c := []float64{0.6, -0.9, 0.7833269096274833, 0.9320390859672264, 0.9974949866040544, 0.9738476308781953, 0.8632093666488737, 0.675463180551151, 0.42737988023383017, 0.1411200080598672, -0.1577456941432482, -0.4425204432948521, -0.6877661591839738, -0.8715757724135882, -0.977530117665097, -0.9961646088358407, -0.9258146823277323}
	return descriptorFromTriplets(t, 17, anz, enz, b, c, 0.05, 0)
}

// KQVXXL (OECG2I departure 4): above the exact check a descriptor solve
// the bound leaves unproven fell through to the evaluation route, which
// returned a gain off the exact one by more than its own rounding with a
// nil error. It is certified, by the exact solve up to order 20, or keeps
// the refined gain with ErrAccuracyUnverified.
func TestDCGainDescriptorUnprovenSolvePastExactCheck(t *testing.T) {
	sys := kqUnprovenPastExactCheck(t, 0x1p-44)
	r := solveDCRegular(nil, sys.A, sys.B, sys.C, sys.D, 0, true)
	if r.solveErr != nil || r.overflow || r.certified {
		t.Fatalf("solve %v, overflow %v, certified %v; want an unproven solve", r.solveErr, r.overflow, r.certified)
	}
	g, err := sys.DCGain()
	if g == nil {
		t.Fatalf("DCGain error %v without a value", err)
	}
	ex, _ := dcGainOwnRoundingExcess(sys, g)
	switch {
	case err == nil && ex > 0:
		t.Fatalf("DCGain = %v, nil; exceeds its own rounding of the exact %v by %g", g.At(0, 0), dcGainExactFloat(t, sys), ex)
	case err != nil && !errors.Is(err, ErrAccuracyUnverified):
		t.Fatalf("DCGain error %v; want nil or ErrAccuracyUnverified", err)
	case err != nil && !(math.Abs(g.At(0, 0)-dcGainExactFloat(t, sys)) <= 1e-6*math.Abs(g.At(0, 0))):
		t.Fatalf("DCGain = %v unverified; want the refined gain near %v", g.At(0, 0), dcGainExactFloat(t, sys))
	}
}

// Controls: the same order-17 model well away from singularity, and an
// order-4 pencil with the same non-diagonal E, keep a certified gain with
// a nil error.
func TestDCGainDescriptorCertifiedPastExactCheckStaysNil(t *testing.T) {
	sys := kqUnprovenPastExactCheck(t, 0.25)
	g, err := sys.DCGain()
	if err != nil {
		t.Fatalf("DCGain error %v; want a certified gain", err)
	}
	if ex, _ := dcGainOwnRoundingExcess(sys, g); ex > 0 {
		t.Fatalf("DCGain = %v exceeds its own rounding of %v by %g", g.At(0, 0), dcGainExactFloat(t, sys), ex)
	}
	for _, dt := range []float64{0, 0.1} {
		a := []float64{-0.7, -0.3, 0.2, -0.91, -1.4, 0.1, 0.05, 0.3, -2}
		e := []float64{1, 0.25, 0, -0.125, 1, 0.0625, 0, 0.5, 1}
		sys := descriptorFromSlices(t, 3, 1, 1, a, []float64{1, 0.3, -0.5}, []float64{0.6, -0.9, 0.4}, []float64{0.05}, e, dt)
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("dt=%v: DCGain error %v; want a certified gain", dt, err)
		}
		if ex, _ := dcGainOwnRoundingExcess(sys, g); ex > 0 {
			t.Fatalf("dt=%v: DCGain = %v exceeds its own rounding by %g", dt, g.At(0, 0), ex)
		}
	}
}

// Control: an exact integrator of a continuous descriptor (A singular, E
// non-diagonal, A non-symmetric) keeps its infinite gain with a nil error.
func TestDCGainDescriptorExactIntegratorStaysInf(t *testing.T) {
	a := []float64{0, 0.5, 0, -1}
	e := []float64{1, 0.25, -0.125, 1}
	sys := descriptorFromSlices(t, 2, 1, 1, a, []float64{1, 1}, []float64{1, 0.5}, []float64{0}, e, 0)
	g, err := sys.DCGain()
	if err != nil || !math.IsInf(g.At(0, 0), 0) {
		t.Fatalf("DCGain = %v, %v; want Inf, nil", g, err)
	}
}

// KQVXXL: OECG2I census ls-modal/e2 107-c0-utrue, order 17, continuous,
// E = 2I with A and B doubled. −A is ill-conditioned in double and its
// balanced form unproven; the evaluation route read the exact gain with a
// nil error. The exact pencil solve, up to order 20, keeps it nil.
func TestDCGainDescriptorPastExactCheckIllConditionedExactSolve(t *testing.T) {
	a := []float64{3.375891686544543, 0.2797266357632943, 0.15074589348639966, -4.309400727196526, -8.708744233719115, -14.317397743243731, -18.898561273821226, -18.088408115497636, 3.457075758613471, -2.017683188992892, -2.161974730482959, -10.321358679814505, -0.4735170136098262, 5.936206754824366, -23.299694665425253, -2.0464534561446097, -6.094745738409224, -2.3248485876130425, -1.7194093295307604, -6.177771697908094, -4.455665912083551, -2.863951024421969, -2.6589900395343964, -3.1798959775434588, 0.5789516738965023, 5.707933518953454, 2.2836517791356856, -5.588053026145164, 0.15529421401641944, 3.503384019012141, -0.1259084131774972, -7.2079152232565145, 1.68512555407755, 3.2344730560846986, -4.398870457938953, 5.209385911692087, -0.7668145428706824, 1.3549452753440976, -1.6452618194025814, 2.688916232535168, 4.616619990466083, -3.4821324038473467, 3.2408218911988986, 13.907495091306025, 4.825747143430652, -8.48309037937083, 6.698144201116262, -0.5898126286449088, 11.655917582113151, 1.362741960606103, -12.897600059123747, -0.014543583031041135, 5.465532307250883, -2.1020641083267875, -11.311677653279467, -10.95763614709597, -8.134241005392806, -24.394336028749393, -19.02688505805573, 6.024937314352305, 4.985732263588782, 3.887817209491794, -12.947131413383318, 10.666141637148499, 10.35919296512856, -16.55416325170206, 3.9597205856125646, 1.0342828673838729, 2.1209101504272825, 3.8158355893148794, -1.9683737442575993, 3.925203010740585, -4.605459155380346, -0.6545752972273583, 2.06793978246255, 0.8588587737079219, 2.7552626357776724, -4.8326357129223165, -5.031535684209436, 3.2774649969303704, 8.978972688179356, 0.06486232457155851, -8.68326871016871, -1.327495988366333, 5.713679820658567, 0.5394486902481311, -1.630903859918505, 1.1340224847792395, -4.907343254547805, 2.9377457510657554, -6.052748393170688, -2.9383396486858007, 4.248380473924194, -2.249028136740468, -2.099628345016205, 1.3919587330679317, 2.888400542717114, -7.363346968335284, -1.6817544627056087, 3.5172218517392575, -3.2298684872016685, -0.98756297702595, -1.0956017546098624, 0.0026999800085534737, 0.26025411825733025, 17.66714463418015, 1.1860248016237875, 13.185174355765128, 14.382466807735499, 4.121027052396576, -7.001594544768588, -8.175679278302455, -1.795143428061198, 13.577024386994411, 6.271677456480432, -3.1831315892721, 5.37525177155726, 1.7976457394022543, 3.3500538238187954, 5.478494816575462, -9.738841231680059, 4.328738183623923, -2.219210766794709, -5.638150398878862, -8.14019474852636, -2.5848917665172966, -10.678689827019317, 1.309525142347884, 10.289742031746957, 5.320058964014821, -15.045929411499442, -10.891476625178878, 1.7304257244390984, -1.5578590071567557, -0.7098236966083515, -17.906283886188483, 5.339149807658291, -4.359676950478962, -3.5543779092287573, 6.618071734920439, -4.525815468608235, -2.53449869921676, 2.014363705194306, -0.05949782801793133, -3.9914301257424336, -11.834893839378191, -6.497104219027554, 4.189961713276308, -3.6485544486918338, -2.038019303405882, -13.442486601896002, -0.04437605278045731, 1.0349029329542851, 7.24313477683845, -2.1882654693801453, -3.4356633482905807, -10.02042406513372, -12.280131380598968, -11.605256429341042, -19.227385093806937, -19.882527488723923, 8.614182570356636, 3.746380483330396, 8.637269022737538, -19.564266849466954, 5.401548249338359, 10.830058248216792, -21.749516880017126, 3.663612979593871, -3.5472597619722235, 5.75157235194726, 5.155107118178314, -0.8525207980258946, -1.4766427660849923, 1.5998731915085669, -10.60201603293616, -9.471982013887507, -0.7556542289669513, 1.7727075888848312, -19.321302204237323, -12.927761098305844, 9.990902391747102, -2.3109534720269798, -3.1543379755676733, -19.736446143977343, -7.158330275957464, 6.516842018422407, 4.730708893950467, -3.5459575665322074, -2.4645711521603135, -18.116427434639895, -2.0427261971525095, -18.778614522604872, -22.81157817978306, -5.686959855372125, 0.9415337786835887, 2.1512485900647587, -5.28445708655107, -13.72384054604575, -17.719994327151717, 1.2371396565568837, -12.540278138132468, -2.6962858741494253, -5.129194613459636, -0.560088577318188, -3.6900541631812036, -1.8490551046957266, -1.7393772018337856, 1.358076251788569, 2.0481762137371526, 3.5388456101471037, 4.498734460767577, -0.5503075405450069, 2.1346753451845486, -0.6828315902494702, 2.0872029152468086, -7.516319897026132, -2.3704630904525, 5.583292008130691, 1.5133057648223784, -6.8387857426722025, -3.821333473307315, -1.275869325685773, 6.587325508459479, 1.2848321739661503, 6.497955103177627, 8.073546192875584, 15.238998265626568, 8.008007437682634, -4.149337240252957, 11.548145287841697, 8.1185603102342, -0.36438401301806844, -6.897218077572224, -7.170978639809544, 25.898096564764344, -1.1283298286027885, -14.69775997462645, -0.026337254978675236, 3.433027592929706, -1.501472063886764, -4.38552322025636, 1.7542255660015864, -1.3170449347862008, -3.107723431918353, 0.4358200086657125, 5.565320961595097, 1.0865334783793803, 0.6022485541752167, -1.4658674030778018, 2.656900847454714, 0.16846486154412496, -5.867668990828923, 0.22635072445335883, 4.378693183183641, 2.6844034450869736, -5.173664671696381, 2.22557633302529, -7.573675234227212, 1.4344933162450613, -4.783461232771988, -7.718554042913947, -1.0071346857791792, -1.3949736711959202, 0.8391166635173589, 4.448523214152464, -4.690684447112606, -10.25624021039891, 0.8988224419169877, 0.32763768068546584, -4.356127391723996, -4.079258638961428, 4.699454388964629, 6.115357874429009, 5.803773803976968, 4.300218161215914, -5.497855778668295, 3.035814664203758, -0.10911067932981214, -3.3151445743346044, 7.4708728010797225, 0.26336430649984444, 6.489970107858231, -3.4937866993099016, 20.732439981176746, 10.935618136909886, -10.907557429422836, 2.557424001567516, 1.6690094162264564}
	b := []float64{-3.368110423243389, -5.831464370869425, 4.357081910186624, -5.548118058176307, -3.5958248398293886, -8.2706615503931, 5.788603164622224, -3.1659780126526935, -1.2919554074086816, -7.416043608693429, -3.998504028721028, -0.0885393969001872, 5.198303169774485, 3.7009494623232997, 3.9436361818928147, 2.7272521180572356, -1.1174395144195637}
	c := []float64{-3.4927948488615, 2.511104585979328, 3.9105348814153578, 2.466266467113616, 5.149920322253801, -6.278266645798197, -0.22014189503559725, 0.6779757822704034, 0.2991346682608668, -1.0364348751898609, 1.5961827975063052, 2.314242676619008, -3.3161643363659246, 3.0555396708874496, -4.323218301978748, -1.323686109220201, 1.0561210445910767}
	e := make([]float64, 17*17)
	for i := range 17 {
		e[i*17+i] = 2
	}
	sys := descriptorFromSlices(t, 17, 1, 1, a, b, c, []float64{-0.025108858496387082}, e, 0)
	if r := solveDCRegular(nil, sys.A, sys.B, sys.C, sys.D, 0, true); !finiteCondition(r.solveErr) {
		t.Fatalf("solve error %v; want −A ill-conditioned in double", r.solveErr)
	}
	if _, ok := sys.dcGainBalancedCertified(0); ok {
		t.Fatal("balanced pencil certifies; the case misses the exact solve")
	}
	const want = 2.3193154289526263
	if exact := dcGainExactFloat(t, sys); exact != want {
		t.Fatalf("exact gain %v, literal %v", exact, want)
	}
	if g, err := sys.DCGain(); err != nil || g.At(0, 0) != want {
		t.Fatalf("DCGain = %v, %v; want %v, nil", g, err, want)
	}
}

// KQVXXL: a graded continuous descriptor (OECG2I census cz-graded/e2 994,
// E = 2I, A and B doubled) whose −A has condition 9e76 in double. The
// bound on the radix-2 balanced pencil proves its gain without the exact
// solve, as cheaply as the evaluation it replaces.
func TestDCGainDescriptorGradedBalancedCertified(t *testing.T) {
	sys := descriptorFromSlices(t, 2, 1, 1, []float64{-2.9309225236693995e+196, -4.301089575684544e+157, 1.4244920749471379e+235, -5.611037196092018e+196}, []float64{-2.1386865577301936e-88, -3.502834653613593e+28}, []float64{1.3372539550948526e+18, 5.4377951661432195e+65}, []float64{0}, []float64{2, 0, 0, 2}, 0)
	if r := solveDCRegular(nil, sys.A, sys.B, sys.C, sys.D, 0, false); !finiteCondition(r.solveErr) {
		t.Fatalf("solve error %v; want −A ill-conditioned in double", r.solveErr)
	}
	bal, ok := sys.dcGainBalancedCertified(0)
	if !ok {
		t.Fatal("balanced pencil unproven")
	}
	g, err := sys.DCGain()
	if err != nil || g.At(0, 0) != bal.At(0, 0) {
		t.Fatalf("DCGain = %v, %v; want the balanced certificate's %v, nil", g, err, bal.At(0, 0))
	}
	if ex, _ := dcGainOwnRoundingExcess(sys, g); ex > 0 {
		t.Fatalf("DCGain = %v exceeds its own rounding of %v by %g", g.At(0, 0), dcGainExactFloat(t, sys), ex)
	}
}

// KQVXXL: a discrete descriptor with E = 2I, A and B doubled, rounds
// x0·E − A only on its diagonal, yet the inverse bound charged it
// u·‖fl(P)‖∞ for that rounding, where its explicit twin pays u·max|P_ii|,
// and lost certificates (OECG2I census ls-fz/e2 14409, a direct solve;
// band-F4nearDCTriangle/e2 1052, the balanced pencil). The exact rounding
// error of x0·E − A, from twoSum, now stands in.
func TestDCGainDescriptorPencilRoundingBoundCertifies(t *testing.T) {
	t.Run("solve", func(t *testing.T) {
		a, b := []float64{6.748663154854694e+14, 1.3934559095112258e+14, -1.0819047285612977e+14, -2.2339039643091055e+13}, []float64{1.0486812179460432, 0.9406378010758556, -1.9984358773710853, 1.9214771942108397, -1.449054007136373, -0.45837865249561577}
		c, d := []float64{-0.11578013396396815, 0.4700131730076014, 0.20593804395550136, 0.33486211162075735, 0.43080034193199807, 0.04222016677384402}, []float64{0.20934263980483694, 0.360747611757932, 0.4551984679350188, -0.100090597008547, -0.2747468646242268, -0.40602664345138584, -0.014975074964544782, 0.17278781135435245, 0.33327130151978984}
		sys := descriptorFromSlices(t, 2, 3, 3, a, b, c, d, []float64{2, 0, 0, 2}, 100)
		if r := solveDCRegular(sys.E, sys.A, sys.B, sys.C, sys.D, 1, false); !r.certified {
			t.Fatalf("descriptor solve uncertified (error %v)", r.solveErr)
		}
	})
	t.Run("balanced", func(t *testing.T) {
		sys := descriptorFromSlices(t, 3, 2, 2, []float64{1.948115543472991, 0, 0, -1180.9467536233915, 1.9999999973810703, 0, 0.011358749778637272, 2.224421151862535, 1.9957961574767922}, []float64{2.936374747237931, -2.4002672711129525, 0.4375002435886821, 0.8498365523536908, -1.8415963580306225, 1.5358082043584025}, []float64{0.3641635880323671, 0.47014994978249236, 0.89769459400313, 0.6920399416065166, 1.1878309698168406, -0.5081537413474582}, []float64{1.5747400294513838, 1.0614852930313083, -0.2227251827356369, -0.07441539925547325}, []float64{2, 0, 0, 0, 2, 0, 0, 0, 2}, 0.1)
		if r := solveDCRegular(sys.E, sys.A, sys.B, sys.C, sys.D, 1, false); !finiteCondition(r.solveErr) {
			t.Fatalf("solve error %v; want E − A ill-conditioned in double", r.solveErr)
		}
		bal, ok := sys.dcGainBalancedCertified(1)
		if !ok {
			t.Fatal("balanced pencil unproven")
		}
		g, err := sys.DCGain()
		if err != nil || !mat.Equal(g, bal) {
			t.Fatalf("DCGain = %v, %v; want the balanced certificate's %v, nil", g, err, bal)
		}
	})
}

// KQVXXL: exactDCGainAtNonsingular proves x0·E − A singular at once where
// a row or column of it is exactly zero, as the descriptor route's
// integrator beside a regular block (OECG2I census F1intBesideRegular,
// dpar0) has; A non-symmetric, E non-diagonal, and A alone has no zero line.
func TestExactDCGainPencilZeroLine(t *testing.T) {
	e := []float64{1, 0.25, 0.5, -0.125, 1, 0.375, 0.0625, -0.5, 1}
	tests := []struct {
		name string
		a    []float64
		zero bool
	}{
		{"zero column", []float64{-0.7, 0.25, 0.3, 0.2, 1, -0.4, -0.6, -0.5, 0.9}, true},
		{"zero row", []float64{-0.7, 0.4, 0.3, -0.125, 1, 0.375, -0.6, 0.8, 0.9}, true},
		{"none", []float64{-0.7, 0.4, 0.3, 0.2, -1.4, -0.4, -0.6, 0.8, -0.9}, false},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sys := descriptorFromSlices(t, 3, 1, 1, tc.a, []float64{1, 0.3, -0.5}, []float64{0.6, -0.9, 0.4}, []float64{0.05}, e, 0.1)
			if hasZeroLine(sys.A, 0) {
				t.Fatal("A alone has a zero line")
			}
			if got := pencilHasZeroLine(sys.E, sys.A, 1); got != tc.zero {
				t.Fatalf("pencilHasZeroLine = %v, want %v", got, tc.zero)
			}
			if _, _, ok := exactDCGainAtNonsingular(sys.E, sys.A, sys.B, sys.C, sys.D, 1); ok == tc.zero {
				t.Fatalf("exactDCGainAtNonsingular ok = %v; want %v", ok, !tc.zero)
			}
		})
	}
}
