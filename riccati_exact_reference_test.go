package controlsys

import (
	"math"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

type riccatiExactCase struct {
	name             string
	n, m             int
	a, b, q, r, s, e []float64
}

func (c riccatiExactCase) dense() (A, B, Q, R, S, E *mat.Dense) {
	n, m := c.n, c.m
	return mat.NewDense(n, n, c.a), mat.NewDense(n, m, c.b), mat.NewDense(n, n, c.q), mat.NewDense(m, m, c.r), mat.NewDense(n, m, c.s), mat.NewDense(n, n, c.e)
}

// riccatiExactReviewCases are Dare draws of the rv39 review sweep (seeds
// 13857, 37017, 48867, 41642 as generated natively by w39Gen, saved under
// scratchpad w47review/oracle): cheap control R ≈ 1e-12, m > n, cross
// term S, E = I + 0.15·N(0, 1). The sweep's oracle formed K from its
// float64-rounded Y and was off the 320-bit K by 2.7e-6, 2.1e-4, 7.4e-7
// and 3.7e-6, which read as certified-wrong rows (TDFWXM).
func riccatiExactReviewCases() []riccatiExactCase {
	return []riccatiExactCase{
		{"seed 13857", 3, 5, []float64{0.16512989717622614, -0.12691217974423774, -0.2625463172359406, -0.8743286070464587, 0.7519938135677892, 0.20108657681406852, -0.5805761169577841, -0.0882202476765931, 0.3784188768030361}, []float64{-1.1138365738913891, 0.33343041543455343, 0.6432335154959963, -1.1093050809476748, -1.031386719918751, -0.04093552261555988, -0.5570703531550396, -1.51147584566615, -1.670415473655939, -1.2440533806586151, 0.1014263938732878, -0.880569745044113, -0.2664934631444428, 0.7301539437474831, 0.3795621261446531}, []float64{1.258436206180033e+11, -7.620183351058147e+10, 3.161182331074342e+11, -7.620183351058147e+10, 9.839979165565472e+11, -1.0300894172230897e+12, 3.161182331074342e+11, -1.0300894172230897e+12, 1.544062468787748e+12}, []float64{1.0006048518127824e-11, -5.272352722285883e-13, -6.537006089345152e-13, 7.753345952230884e-13, -1.0834987382214139e-13, -5.272352722285883e-13, 9.874112315328928e-12, -9.691583544745983e-14, -2.7645868680294535e-13, 1.657409567862452e-13, -6.537006089345152e-13, -9.691583544745983e-14, 9.31632646110756e-12, -5.16955048371851e-13, -1.2899303371094128e-13, 7.753345952230884e-13, -2.7645868680294535e-13, -5.16955048371851e-13, 9.340401103997703e-12, 5.03129926842264e-13, -1.0834987382214139e-13, 1.657409567862452e-13, -1.2899303371094128e-13, 5.03129926842264e-13, 1.0001121368837716e-11}, []float64{-0.006016709407351466, 0.001323427924358537, 0.0029056073955610077, -0.012203720922295528, -0.01826452801999528, -0.05001425399901001, 0.05282724820612193, 0.01710699602118271, -0.018591708238488363, 0.039893454500025426, 0.03286896187091784, -0.044632589161792274, -0.009572306330212433, -0.0074219625679628285, -0.07166476234837728}, []float64{1.1858111144029029, -0.10985818610205345, -0.07221844004189239, -0.10358073796070186, 0.9912796253522691, -0.0016751547174243854, -0.12244927243585321, -0.1581051545198203, 1.1366520159218645}},
		{"seed 37017", 2, 4, []float64{-0.07702362175480322, 1.0534088341870036, -0.1778012497131539, -0.058345694748140574}, []float64{-1.6018566222630306, -1.6049000727699059, -1.6049000727699059, -1.6049000727699059, 1.637795523162474, -0.08122261886563509, -0.08122261886563509, -0.08122261886563509}, []float64{13.529416641760545, -22.023804111748408, -22.023804111748408, 35.851357112877565}, []float64{1.1701009113810691e-11, 1.0043820636875039e-13, 3.935159403679422e-14, -3.6162888583531543e-13, 1.0043820636875039e-13, 1.1615353014396924e-11, 2.1975486762130918e-13, -2.891765305955861e-13, 3.935159403679422e-14, 2.1975486762130918e-13, 1.1654744180904846e-11, 1.8719131862866346e-13, -3.6162888583531543e-13, -2.891765305955861e-13, 1.8719131862866346e-13, 1.184336895675327e-11}, []float64{1.1748222174587408e-07, 1.6759368580598408e-09, -3.5986693485769466e-08, 5.678235598236089e-08, -1.912429417213533e-07, -2.7281667822719936e-09, 5.85807879930132e-08, -9.243291993085882e-08}, []float64{0.8141861232558175, -0.25253770766318356, 0.03962040404906155, 1.2355691985124067}},
		{"seed 48867", 2, 3, []float64{-0.44875903833735376, 0.36294863144182665, 0.1320130445710161, -0.21974518822462966}, []float64{-0.08877739319037521, 0.29460110036930776, -0.38148670404918317, -0.8679962689159643, 0.34367824821551984, 0.5234835844319388}, []float64{1.0698753968511668e+11, -2.2377298707678735e+11, -2.2377298707678735e+11, 4.680390809307863e+11}, []float64{2.8439165920801418e-12, 2.535677281328547e-14, 8.07925208067241e-15, 2.535677281328547e-14, 2.8080929802360994e-12, 5.6956430351302944e-15, 8.07925208067241e-15, 5.6956430351302944e-15, 2.8209968483160937e-12}, []float64{0.00015975093963595464, 8.355472167148732e-05, 2.2212171419177535e-05, -0.0003341318536333638, -0.0001747613760240467, -4.6458531176247404e-05}, []float64{0.8550108414974151, -0.2824150856197495, -0.09174963417255322, 0.9781568666805839}},
		{"seed 41642", 5, 6, []float64{-0.19458880445283658, 0.25267073691891656, 0.0007420407159429843, 0.07591926959265387, -0.4179591160183193, -0.7190161737602189, -0.41180457046757896, -0.11205979102446487, 0.4284008041091133, -0.0782382950239907, 0.9917923212157532, 0.42866690879618663, -1.0119682066336437, 0.5607194457800456, -0.5839711966669757, -0.6713634453860062, -1.052851116761464, -0.10885196630768343, -0.34969765141219317, -1.1106770164735547, -0.42872880814663505, -0.11633665384491457, 0.0025725583038765047, 0.17515834374822492, 0.778494737927475}, []float64{0.22673753927014414, -0.9322808272915757, 0.17358954684902272, 0.4082221523543552, -0.8396406011157493, 0.4834304781101262, -1.7573200834707814, 0.741827031805596, -0.6599131969341082, -1.1422953824438569, 1.3937058015054904, -0.019872523129304764, -0.5635010320207794, 0.2766337317202135, -1.1903123262947957, -1.2432215633944446, -1.1885881995510101, 0.48784358924569615, 0.5653571583584016, -0.5235455593526663, -0.977801159357926, -0.2641746670894962, 0.5811697409926166, 0.38346853570405565, -0.6294092165004148, 0.3594262053926511, 0.29876114549501315, -0.5741882056861859, -0.9664839532536091, -0.4254997655081067}, []float64{8.017716777800822e+10, 8.119532140578935e+10, 1.693866917351191e+10, 1.0439636576548363e+11, 1.0768651064059856e+11, 8.119532140578935e+10, 1.771824926889628e+11, -2.1415999573274902e+10, 1.050197773223323e+11, 1.569734737390597e+11, 1.693866917351191e+10, -2.1415999573274902e+10, 4.5735142183673706e+10, -2.4572957084846474e+10, -2.1530833802708687e+10, 1.0439636576548363e+11, 1.050197773223323e+11, -2.4572957084846474e+10, 2.328024512820972e+11, 1.296503837297933e+11, 1.0768651064059856e+11, 1.569734737390597e+11, -2.1530833802708687e+10, 1.296503837297933e+11, 4.049044308021282e+11}, []float64{1.407424998942529e-12, 1.5722746671716373e-14, 6.319725655358001e-15, -6.9588916338866596e-15, 7.94134608759349e-15, -7.218138446327593e-15, 1.5722746671716373e-14, 1.3520231453137429e-12, -3.5203349657416993e-15, -4.27948233614394e-14, 1.117445372615073e-14, 9.096605322430567e-16, 6.319725655358001e-15, -3.5203349657416993e-15, 1.3573636501324225e-12, 4.476098695421101e-14, -4.0994733478734046e-14, 1.1431511419636604e-15, -6.9588916338866596e-15, -4.27948233614394e-14, 4.476098695421101e-14, 1.3788278945941605e-12, -5.437698176963212e-14, 7.419770280423131e-15, 7.94134608759349e-15, 1.117445372615073e-14, -4.0994733478734046e-14, -5.437698176963212e-14, 1.3922844713577454e-12, 2.330196116910424e-14, -7.218138446327593e-15, 9.096605322430567e-16, 1.1431511419636604e-15, 7.419770280423131e-15, 2.330196116910424e-14, 1.3343097075200805e-12}, []float64{-0.0036642083109287367, -0.02280843857617515, -0.006970705563982535, 0.01623180877697662, 0.009020987068625617, 0.03104122766349361, -0.0003744975725041917, -0.03228022895371933, 0.027260882456515107, 0.030666044236801176, -0.021313691533101035, 0.051879957326559645, -0.012235184292819716, -0.03092566254178332, -0.01950701169230804, 0.01512222551059633, 0.0008923901694569702, -0.010869786231092395, 0.02538174376837303, 0.023140279671246267, -0.013917120263967134, -0.020719231069647556, 0.04570906690400269, 0.04552155689523494, -0.042026915658412704, -0.007041663134492973, 0.05830994159818236, 0.055076310806829884, -0.030165996458424344, 0.10340019074770064}, []float64{1.0768716244946444, 0.11782478365596226, -0.023641213236542225, -0.07929088672654758, 0.0783188672374505, -0.08917403983922761, 0.6246632788911015, -0.07274113942927252, -0.1809369451002165, -0.028776855665361232, 0.1627270773734309, 0.052729552909762896, 0.9394616445320457, 0.21210326538637975, 0.06626266219727542, 0.03425590497997902, 0.08514428234173962, -0.04193742930899691, 0.9655583705579618, -0.057235017155546304, -0.09106770463174513, 0.13620404994193872, -0.13023468156240955, 0.12546286740549076, 1.1782377521235026}},
	}
}

// perturbGain returns K·(1 ± 1e-3) entrywise: a stabilizing start off the
// reference, so that Newton has to find the fixed point itself.
func perturbGain(K *mat.Dense) *mat.Dense {
	o := mat.DenseCopyOf(K)
	raw := o.RawMatrix().Data
	for i := range raw {
		raw[i] *= 1 + 1e-3*float64(1-2*(i%2))
	}
	return o
}

// TestRiccatiExactReference_DareDoublingMatchesHewer: doubling in 320 bits
// and Hewer's iteration in 448 bits, both on the unrounded E⁻¹A, E⁻¹B,
// agree far below float64; the rounded reference values are pinned.
func TestRiccatiExactReference_DareDoublingMatchesHewer(t *testing.T) {
	pins := map[string][2]float64{
		"seed 13857": {352998605.05961257, 125838069256.61671},
		"seed 37017": {3650.3677498615948, 13.528662373588036},
		"seed 48867": {29950687.857801881, 106987531063.55251},
		"seed 41642": {223635440.83509687, 80176359519.685623},
	}
	for _, c := range riccatiExactReviewCases() {
		A, B, Q, R, S, E := c.dense()
		ref := riccatiExactReference(false, A, B, Q, R, S, E, riccatiExactPrec)
		if !ref.OK {
			t.Fatalf("%s: doubling reference not certified (residual %g)", c.name, ref.Resid)
		}
		nw := riccatiExactNewtonReference(false, A, B, Q, R, S, E, perturbGain(ref.K), 448)
		if !nw.OK {
			t.Fatalf("%s: Hewer reference not certified (residual %g)", c.name, nw.Resid)
		}
		dY, dX, dK := xbRelDiff(ref.y, nw.y), xbRelDiff(ref.x, nw.x), xbRelDiff(ref.k, nw.k)
		t.Logf("%s: doubling %d steps, Hewer %d; differ Y %.2g X %.2g K %.2g", c.name, ref.Iter, nw.Iter, dY, dX, dK)
		if !(dY <= 1e-60 && dX <= 1e-60 && dK <= 1e-60) {
			t.Errorf("%s: doubling and Hewer differ: Y %g X %g K %g", c.name, dY, dX, dK)
		}
		p := pins[c.name]
		if k, y := ref.K.At(0, 0), ref.Y.At(0, 0); !(relDiffScalar(k, p[0]) <= 1e-15 && relDiffScalar(y, p[1]) <= 1e-15) {
			t.Errorf("%s: K₀₀ %.17g Y₀₀ %.17g, want %.17g %.17g", c.name, k, y, p[0], p[1])
		}
	}
}

func relDiffScalar(got, want float64) float64 { return math.Abs(got-want) / math.Abs(want) }

// TestRiccatiExactSensitivity_IncludesWeights: perturbing A, B and E alone
// moves K by about 1e-15 on the review draws, while componentwise eps in Q
// moves it by 1e-6–1e-3 (cheap control); a band without Q and R reads
// rounding-level K errors as certified-wrong. The gain of the float64
// rounded Y (what the rv39 oracle returned), a componentwise eps/2
// perturbation of Y, lies inside the first-order bound KofY.
func TestRiccatiExactSensitivity_IncludesWeights(t *testing.T) {
	skipSlowUnderRace(t)
	for _, c := range riccatiExactReviewCases() {
		A, B, Q, R, S, E := c.dense()
		ref := riccatiExactReference(false, A, B, Q, R, S, E, riccatiExactPrec)
		if !ref.OK {
			t.Fatalf("%s: reference not certified", c.name)
		}
		full, ok1 := riccatiExactSensitivity(false, A, B, Q, R, S, E, ref, "ABQRSE")
		abe, ok2 := riccatiExactSensitivity(false, A, B, Q, R, S, E, ref, "ABE")
		if !ok1 || !ok2 {
			t.Fatalf("%s: perturbed reference not certified", c.name)
		}
		md, _ := xbExplicitModel(A, B, Q, R, S, E, riccatiExactPrec)
		ky, _ := md.gain(false, xbFrom(ref.Y, c.n, c.n, riccatiExactPrec))
		kRounded := relErr(xbDense(ky), ref.K)
		t.Logf("%s: K sensitivity %.3g (A, B, E only %.3g), under Y(1±eps) %.3g; gain of rounded Y %.3g", c.name, full.K, abe.K, full.KofY, kRounded)
		if !(full.K >= 1e3*abe.K && full.K >= 1e-7) {
			t.Errorf("%s: K sensitivity %g, A, B, E only %g", c.name, full.K, abe.K)
		}
		if !(kRounded > 1e-7 && kRounded <= full.KofY) {
			t.Errorf("%s: gain of rounded Y off by %g, bound %g", c.name, kRounded, full.KofY)
		}
	}
}

// TestDare_DescriptorCheapControlNotCertifiedWrong: on the review draws a
// certified Dare result (Accuracy 0) is inside the eps band of the
// exact-E reference (Y6BJEX rule 1, K floor gainFloor).
func TestDare_DescriptorCheapControlNotCertifiedWrong(t *testing.T) {
	skipSlowUnderRace(t)
	for _, c := range riccatiExactReviewCases() {
		A, B, Q, R, S, E := c.dense()
		ref := riccatiExactReference(false, A, B, Q, R, S, E, riccatiExactPrec)
		sens, ok := riccatiExactSensitivity(false, A, B, Q, R, S, E, ref, "ABQRSE")
		if !ref.OK || !ok {
			t.Fatalf("%s: reference not certified", c.name)
		}
		for _, ns := range []bool{false, true} {
			res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, S: S, NoScaling: ns})
			if err != nil {
				t.Logf("%s NoScaling %v: %v", c.name, ns, err)
				continue
			}
			eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
			t.Logf("%s NoScaling %v: Accuracy %d, X err %.3g (sens %.3g), K err %.3g (sens %.3g)", c.name, ns, res.Accuracy, eX, sens.X, eK, sens.KBand())
			if riccatiCertifiedWrong(res.Accuracy, eX, sens.X, eK, sens.KBand(), gainFloor) {
				t.Errorf("%s NoScaling %v: certified-wrong: X err %g, K err %g", c.name, ns, eX, eK)
			}
		}
	}
}

// riccatiExactIllConditionedE: n 3, m 2, E upper triangular with last pivot
// 1e-7 (cond(E) ≈ 1e7), cross term S.
func riccatiExactIllConditionedE() (A, B, Q, R, S, E *mat.Dense) {
	A = mat.NewDense(3, 3, []float64{0.3, -1.2, 0.5, 0.8, -0.4, 1.1, -0.6, 0.9, -0.7})
	B = mat.NewDense(3, 2, []float64{1, 0, 0.5, -1, -0.3, 0.8})
	Q = mat.NewDense(3, 3, []float64{2, 0.3, -0.1, 0.3, 1, 0.2, -0.1, 0.2, 1.5})
	R = mat.NewDense(2, 2, []float64{1e-3, 1e-4, 1e-4, 2e-3})
	S = mat.NewDense(3, 2, []float64{1e-3, -2e-3, 0, 1e-3, 2e-3, 0})
	E = mat.NewDense(3, 3, []float64{1.2, 0.4, -0.3, 0, 0.9, 0.7, 0, 0, 1e-7})
	return
}

// TestRiccatiExactReference_IllConditionedEMethodsAgree: with cond(E) ≈ 1e7
// the matrix sign function and doubling (320 bits) agree with Kleinman and
// Hewer (448 bits). Rounding E⁻¹A, E⁻¹B to float64 alone moves K by
// cond(E)·eps ≈ 5e-10, 1e5 times the eps band, so a reference built on the
// rounded model would certify correct gains as wrong.
func TestRiccatiExactReference_IllConditionedEMethodsAgree(t *testing.T) {
	A, B, Q, R, S, E := riccatiExactIllConditionedE()
	for _, cont := range []bool{true, false} {
		ref := riccatiExactReference(cont, A, B, Q, R, S, E, riccatiExactPrec)
		if !ref.OK {
			t.Fatalf("continuous %v: reference not certified (residual %g)", cont, ref.Resid)
		}
		nw := riccatiExactNewtonReference(cont, A, B, Q, R, S, E, perturbGain(ref.K), 448)
		if !nw.OK {
			t.Fatalf("continuous %v: Newton reference not certified (residual %g)", cont, nw.Resid)
		}
		dY, dX, dK := xbRelDiff(ref.y, nw.y), xbRelDiff(ref.x, nw.x), xbRelDiff(ref.k, nw.k)
		sens, ok := riccatiExactSensitivity(cont, A, B, Q, R, S, E, ref, "ABQRSE")
		if !ok {
			t.Fatalf("continuous %v: perturbed reference not certified", cont)
		}
		md, _ := xbExplicitModel(A, B, Q, R, S, E, riccatiExactPrec)
		rd := riccatiExactReference(cont, xbDense(md.a), xbDense(md.b), Q, R, S, nil, riccatiExactPrec)
		kRounded := relErr(rd.K, ref.K)
		t.Logf("continuous %v: %d / %d steps; differ Y %.2g X %.2g K %.2g; K sensitivity %.3g; rounded model K %.3g", cont, ref.Iter, nw.Iter, dY, dX, dK, sens.K, kRounded)
		if !(dY <= 1e-50 && dX <= 1e-50 && dK <= 1e-50) {
			t.Errorf("continuous %v: methods differ: Y %g X %g K %g", cont, dY, dX, dK)
		}
		if !(kRounded > 1e3*sens.K) {
			t.Errorf("continuous %v: rounded model K off by %g, sensitivity %g", cont, kRounded, sens.K)
		}
	}
}
