package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// With rank(B) = m ≥ 2 and cond(B) above rotationCond, Dare solves in the
// inputs w = Vᵀu of B's right singular vectors (red.rotated): the given
// Dare under an orthogonal change of inputs, Q kept and nothing reduced
// away. A K that problem refines in double-double for a descriptor E is
// returned as terminal, VK_w, without gainMiss (LU3SUY). Unlike the
// reduced inputs of Y6BJEX am.4, rounding BV, VᵀRV and SV is an eps
// perturbation of B, R and S, inside the K band; it is the gain of the
// rounded X that is off. Over 4111 sweep rows reaching the path natively
// (scratchpad w48review/lu3suy) K was within 4x its band, never
// certified-wrong; routed through gainOfX, 767 rows fell to Accuracy 1
// and 6 became certified-wrong at Accuracy 0 (seed 653: K 8e-15 ->
// 3.2e-6, band 4.4e-13).

type dareRotatedInputsCase struct {
	name             string
	A, B, Q, R, S, E *mat.Dense
}

// TestDareDescriptorRotatedInputsRefinedGainTerminal: ill-conditioned
// full-rank B (cond 5e3..2e7), cheap R, E's last pivot 6e-8..2e-6. The
// refined K is within 1e-9 of the exact-E 320-bit gain (its band at most
// 9e-12) where the gain of the returned X is off by up to 8e-4.
func TestDareDescriptorRotatedInputsRefinedGainTerminal(t *testing.T) {
	cases := []dareRotatedInputsCase{
		{name: "n4 m4 S pivot 8.8e-7",
			A: mat.NewDense(4, 4, []float64{-0.17105539187954935, 0.4236580499533485, 0.4779464963055427, 0.19473532572317923, -0.5999155821946511, 0.37525106650881956, -0.31254970159410755, -0.4251195129754615, -0.8375347348728046, -0.18321547707165378, 0.8081906443985718, -0.4483006943356738, -0.9807861267670679, 0.22672464756634916, -0.5272054514862735, -0.07631190661968326}),
			B: mat.NewDense(4, 4, []float64{0.1102072476809986, -0.14697602658938835, -0.1133337576665614, 0.0942413423546121, 0.04878035725158111, -0.029688236185524634, -0.001194065523350237, 0.0442768913490374, 0.6784310221166552, -0.7791668158922569, -0.47940972385148894, 0.6038642847415594, -0.12421233822718214, 0.12504945383816632, 0.060353301295647595, -0.1124568984362274}),
			Q: mat.NewDense(4, 4, []float64{3216.4987467310407, 207.6445343325746, 6713.658887193858, -1702.0553696821912, 207.6445343325746, 13688.848492174917, -581.697160033564, -19937.684758165702, 6713.658887193858, -581.697160033564, 14125.808418271099, -2086.418978467895, -1702.0553696821912, -19937.684758165702, -2086.418978467895, 29671.486747802934}),
			R: mat.NewDense(4, 4, []float64{1.0382963484779979e-05, 5.049142592282745e-08, -4.0154141212898296e-08, -1.2387415301926645e-08, 5.049142592282745e-08, 1.0430328029654462e-05, -1.8814737064498644e-07, 5.1166323641222844e-08, -4.0154141212898296e-08, -1.8814737064498644e-07, 1.0479426186485558e-05, -1.1035648122608697e-07, -1.2387415301926645e-08, 5.1166323641222844e-08, -1.1035648122608697e-07, 1.0686338489568103e-05}),
			S: mat.NewDense(4, 4, []float64{-0.002018437858759671, 5.597281244273514e-05, -0.0007207753382849649, -0.0005787227781706179, -0.004025537033045007, -0.0012829881022012142, 0.0021397179431010822, 0.001975776043542021, 0.001881310699902766, -0.0032019128366809597, 0.004931564128622332, -0.0012067670299508355, -0.0038622970888051644, 0.001829688588548578, -0.0009663819410645553, -2.4671913023856035e-05}),
			E: mat.NewDense(4, 4, []float64{1.1256459208438887, 0.7136425873800321, -0.25254661497641706, 0.2296462706405676, 0, 1.2045344675400815, -0.27404766593463087, -0.22325180893549362, 0, 0, 1.1615061106826963, 0.372652492483558, 0, 0, 0, 8.805997926139464e-07})},
		{name: "n4 m4 pivot 1.9e-6",
			A: mat.NewDense(4, 4, []float64{0.1070603713668656, -0.22427932991811922, -0.45422429202552866, -0.3878478240239826, -0.5129026491989318, 0.27557546113669934, 0.27834678532561086, -0.21387898159866264, -0.31192254121709484, 0.4616641064453919, 0.24653872899105367, 0.1547167179268248, -0.2890008537862146, -0.44018435673606826, -0.25511849555877675, 0.41974955017691506}),
			B: mat.NewDense(4, 4, []float64{0.021299565589463745, -0.10956027942566268, -0.0012273034443738764, 0.08262682525871022, 0.17184804759989594, -0.6133851513538613, 0.08336510981022707, 0.4099030580749561, 0.17734600869632908, -0.645054388880928, 0.08205108850319552, 0.43457004595862475, 0.09540165237046483, -0.333887529758707, 0.04887333004042515, 0.22160257663571656}),
			Q: mat.NewDense(4, 4, []float64{9.533394312311549e+06, 2.013685444791075e+07, 4.143091491134956e+06, 6.130127989551839e+06, 2.013685444791075e+07, 4.2560307111543156e+07, 8.755650489684697e+06, 1.295488121572865e+07, 4.143091491134956e+06, 8.755650489684697e+06, 1.8062703696901686e+06, 2.665424149158419e+06, 6.130127989551839e+06, 1.295488121572865e+07, 2.665424149158419e+06, 3.9485920024101753e+06}),
			R: mat.NewDense(4, 4, []float64{2.5524497274887187e-07, 2.1423413790573216e-09, 3.198430044461344e-09, -1.6735439454378016e-09, 2.1423413790573216e-09, 2.561399103620291e-07, 6.724477935136384e-09, -6.035257870027763e-10, 3.198430044461344e-09, 6.724477935136384e-09, 2.676759677318339e-07, -8.209724649475446e-09, -1.6735439454378016e-09, -6.035257870027763e-10, -8.209724649475446e-09, 2.596280462162955e-07}),
			E: mat.NewDense(4, 4, []float64{1.071844362950793, 0.0697664059452094, 0.03326303232244504, -0.6530956867582552, 0, 1.0687150860207193, 0.04837085427586052, -0.011470359638865828, 0, 0, 1.0210601172798923, -0.019676516409887122, 0, 0, 0, 1.916774851053531e-06})},
		{name: "n2 m2 pivot 6.3e-8",
			A: mat.NewDense(2, 2, []float64{-0.18985711857963702, -0.0006962288916305807, 1.0112653194131973, 0.3544418528242476}),
			B: mat.NewDense(2, 2, []float64{0.011722258048125673, -0.028389953259824758, -0.30990415450999453, 0.7505575615186237}),
			Q: mat.NewDense(2, 2, []float64{13.500833739593526, -2.786279689444718, -2.786279689444718, 1.9413588898268752}),
			R: mat.NewDense(2, 2, []float64{7.068301824021857e-09, -3.886184433319728e-11, -3.886184433319728e-11, 7.70324633304546e-09}),
			E: mat.NewDense(2, 2, []float64{-1.0072779864930324, -0.002543726776920344, -0.47450134389198395, -0.001198211255127088})},
	}
	gainOff := 0
	for _, c := range cases {
		ref := riccatiExactReference(false, c.A, c.B, c.Q, c.R, c.S, c.E, riccatiExactPrec)
		if !ref.OK {
			t.Fatalf("%s: exact-E oracle did not certify", c.name)
		}
		for _, ns := range []bool{false, true} {
			opts := &RiccatiOpts{E: c.E, S: c.S, NoScaling: ns}
			red, _ := reduceInputs(c.A, c.B, c.Q, c.R, opts)
			if red == nil || !red.rotated {
				t.Fatalf("%s NoScaling %v: inputs not rotated", c.name, ns)
			}
			if r := red.solve(c.A, c.B, c.Q, opts); r == nil || r.xe == nil {
				t.Fatalf("%s NoScaling %v: rotated Dare K not refined", c.name, ns)
			}
			res, err := Dare(c.A, c.B, c.Q, c.R, opts)
			if err != nil {
				t.Fatalf("%s NoScaling %v: %v", c.name, ns, err)
			}
			eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
			kx, _, ok := ddGain(c.A, c.B, c.R, c.S, res.X)
			eKx := 1.0
			if ok {
				eKx = relErr(kx, ref.K)
			}
			t.Logf("%s NoScaling %v: Accuracy %d, X err %.3g, K err %.3g, gain of X err %.3g", c.name, ns, res.Accuracy, eX, eK, eKx)
			if res.Accuracy != 0 {
				t.Errorf("%s NoScaling %v: Accuracy %d", c.name, ns, res.Accuracy)
			}
			if eX > newtonTol || eK > 1e-9 {
				t.Errorf("%s NoScaling %v: X off by %.3g, K by %.3g", c.name, ns, eX, eK)
			}
			if dareRotatedInputsUnstable(c.A, c.B, c.E, res.K) {
				t.Errorf("%s NoScaling %v: E⁻¹(A − BK) unstable in %d bits", c.name, ns, riccatiExactPrec)
			}
			if eKx > 1e-7 {
				gainOff++
			}
		}
	}
	if gainOff == 0 {
		t.Error("no case where the gain of X is off the exact gain: the test no longer separates the refined K from it")
	}
}

// dareRotatedInputsUnstable reports whether E⁻¹(A − BK), formed from the
// float64 K and the unrounded E in riccatiExactPrec bits, is not Schur
// stable (xbStable).
func dareRotatedInputsUnstable(A, B, E, K *mat.Dense) bool {
	n, m := B.Dims()
	md, ok := xbExplicitModel(A, B, eye(n), eye(m), nil, E, riccatiExactPrec)
	if !ok {
		return true
	}
	return !xbStable(false, xbAxpy(md.a, -1, xbMul(md.b, xbFrom(K, m, n, md.prec))))
}

// TestDareRotatedInputsUnrefinedGainHeldToGainOfX: a rotated Dare whose K
// is not refined in double-double (xe nil) is held to the gain of its X
// (gainOfX), as the reduced inputs are, and not returned as terminal.
// Before, the pencil K was certified-wrong at Accuracy 0 (XD3P6I; rv48r
// seeds 1743 and 2270): 1.7e-6 and 2.2e-5 off against a band of 1.9e-9
// without E, 0.69 and 0.94 against 8e-11 with E's last pivot 4.8e-6, while
// X was within 1.5e-11.
func TestDareRotatedInputsUnrefinedGainHeldToGainOfX(t *testing.T) {
	cases := []dareRotatedInputsCase{
		{name: "n2 m2 no E",
			A: mat.NewDense(2, 2, []float64{1.1970987248830431, 0.21839387389975254, -0.5696974383317107, -0.325348970298359}),
			B: mat.NewDense(2, 2, []float64{0.4816730169796179, 0.35567159949345223, 0.6443107633354956, 0.4757651293207472}),
			Q: mat.NewDense(2, 2, []float64{1.952990385750042e+07, 8.1407794804044e+07, 8.1407794804044e+07, 3.3933751559724957e+08}),
			R: mat.NewDense(2, 2, []float64{2.197687831027214e-12, -2.8287591636902017e-14, -2.8287591636902017e-14, 2.0745224304428615e-12})},
		{name: "n2 m2 pivot 4.8e-6",
			A: mat.NewDense(2, 2, []float64{-0.3784656429752892, 0.1508584940756077, 0.3546425885971392, 1.06643209694223}),
			B: mat.NewDense(2, 2, []float64{-0.1835140494171699, 0.07209681722520986, 0.9124725198553958, -0.35849482998694643}),
			Q: mat.NewDense(2, 2, []float64{890.0283588313838, 1552.553172757777, 1552.553172757777, 2708.252304909865}),
			R: mat.NewDense(2, 2, []float64{8.106860178282055e-09, 3.6169833946935676e-11, 3.6169833946935676e-11, 8.144705169254964e-09}),
			E: mat.NewDense(2, 2, []float64{1.031897554859899, -0.18384676410337314, 0, 4.841874544440457e-06})},
	}
	for _, c := range cases {
		ref := riccatiExactReference(false, c.A, c.B, c.Q, c.R, c.S, c.E, riccatiExactPrec)
		if !ref.OK {
			t.Fatalf("%s: exact-E oracle did not certify", c.name)
		}
		sens, ok := riccatiExactSensitivity(false, c.A, c.B, c.Q, c.R, c.S, c.E, ref, "ABQRSE")
		if !ok {
			t.Fatalf("%s: no sensitivity", c.name)
		}
		kTol := max(newtonTol, 4*sens.KBand())
		for _, ns := range []bool{false, true} {
			opts := &RiccatiOpts{E: c.E, S: c.S, NoScaling: ns}
			red, _ := reduceInputs(c.A, c.B, c.Q, c.R, opts)
			if red == nil || !red.rotated {
				t.Fatalf("%s NoScaling %v: inputs not rotated", c.name, ns)
			}
			if r := red.solve(c.A, c.B, c.Q, opts); r == nil || r.xe != nil {
				t.Fatalf("%s NoScaling %v: rotated Dare K not the unrefined one", c.name, ns)
			}
			res, err := Dare(c.A, c.B, c.Q, c.R, opts)
			if err != nil {
				t.Fatalf("%s NoScaling %v: %v", c.name, ns, err)
			}
			eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
			t.Logf("%s NoScaling %v: Accuracy %d, X err %.3g, K err %.3g (tol %.3g)", c.name, ns, res.Accuracy, eX, eK, kTol)
			if res.Accuracy != 0 {
				t.Errorf("%s NoScaling %v: Accuracy %d", c.name, ns, res.Accuracy)
			}
			if eX > newtonTol || eK > kTol {
				t.Errorf("%s NoScaling %v: X off by %.3g, K by %.3g (tol %.3g)", c.name, ns, eX, eK, kTol)
			}
			if dareRotatedInputsUnstable(c.A, c.B, c.E, res.K) {
				t.Errorf("%s NoScaling %v: E⁻¹(A − BK) unstable in %d bits", c.name, ns, riccatiExactPrec)
			}
		}
	}
}

// TestDareRotatedInputsAccuracy1UnrefinedGainNotPromoted: rotated inputs
// (n 6, m 6, R ~6e-12, cond(E) 1.2e4, E's last pivot 2e-4) where the
// rotated Dare K is unrefined at Accuracy 1, 4.7 off the exact-E 320-bit
// gain with X 1.1e-7 off (rv49v seed 3 under -race; natively and on
// amd64 the K is refined). A Newton step took X to 4.3e-10 and its gain
// 0.44 off (band 2.1e-5): routed through the gain of X that was returned
// at Accuracy 0. An unrefined rotated K at Accuracy 1 is not promoted
// (Y6BJEX am.7, fail closed). Every platform must return Accuracy 1 or K
// within 4x its band. The bands are riccatiExactSensitivity(..., "ABQRSE") of this
// reference, pinned: computing them took 10s under -race.
func TestDareRotatedInputsAccuracy1UnrefinedGainNotPromoted(t *testing.T) {
	A := mat.NewDense(6, 6, []float64{-0.2297988526985603, 0.3731555690063577, 1.6005367049873171, -0.38756991713553657, 0.33510582389013127, 0.08504645344091989, -0.5878580140931389, -0.5782673939453522, -0.21762702892558278, 0.02143886398930735, -0.16506806977267724, -0.26020058732908286, -0.10214747110058674, -0.10461881409131268, -0.12375454705934234, -0.1895164312860169, 0.2176410185069525, -0.15170270590926807, 0.05256152317806577, -0.5961409450946975, -0.3110911237907826, -0.23886968259138613, -0.650335819081738, -0.2771955484055719, -0.4991114118076528, 0.12552433385164147, 0.30895872517593886, -0.46103270243213423, 0.38423116045407546, 0.33469547460146126, -0.14266517822780153, -1.1962252497363803, -0.5059055086723271, 0.3775313807989686, -1.7285400816033685, -1.072096871117419})
	B := mat.NewDense(6, 6, []float64{-0.03755141253403411, -0.1671847236581683, -0.0745767006288806, 0.10064276288144533, -0.026476638185591622, -0.09197327958982354, 0.008142554911597, 0.0882101871490937, 0.015611507229654868, -0.05897419012861597, 0.014800715999891426, 0.046813442769757616, -0.04977843096468789, -0.19870112420251826, -0.10197990462683068, 0.11493651765413067, -0.03220288738909322, -0.11013398325426542, 0.013777681823248376, 0.09953419898975424, 0.023545276835942593, -0.06613801753609716, 0.015196351606215186, 0.0534562892980566, -0.05805390167886049, -0.3971847164339445, -0.09694350401797225, 0.26587441931062733, -0.05881458626746961, -0.21357366466110572, -0.09697079422938103, -0.5418154905761314, -0.1837009545501403, 0.3436796252317284, -0.08468758313905844, -0.2942789716525973})
	Q := mat.NewDense(6, 6, []float64{34209.023096189616, 43830.27336253331, 56281.5702558246, 16813.585347948363, 37007.248768453566, 16356.893113504313, 43830.27336253331, 101339.14786141408, 89564.52624046597, 43714.59307887448, 7838.862131501163, 32030.92026215781, 56281.5702558246, 89564.52624046597, 118590.48099106591, 37255.15113282429, 96621.03691732543, 47619.76179116054, 16813.585347948363, 43714.59307887448, 37255.15113282429, 19199.344503041473, 1491.2994917198623, 14350.782232707803, 37007.248768453566, 7838.862131501163, 96621.03691732543, 1491.2994917198623, 209932.3461597246, 51542.88953784061, 16356.893113504313, 32030.92026215781, 47619.76179116054, 14350.782232707803, 51542.88953784061, 24558.635877973364})
	R := mat.NewDense(6, 6, []float64{6.745576406971377e-12, 3.9960975749636263e-13, 1.4657687012340858e-13, -2.4485969198500932e-14, 7.331254822661314e-14, 4.047859312449157e-14, 3.9960975749636263e-13, 7.023014342241732e-12, 3.383898378985037e-13, 2.2199292386079556e-13, 6.26408911258951e-14, 1.0594078624752675e-13, 1.4657687012340858e-13, 3.383898378985037e-13, 6.625820954166991e-12, -2.0136929327440082e-14, 1.0976703527844666e-13, -1.8377402355335295e-13, -2.4485969198500932e-14, 2.2199292386079556e-13, -2.0136929327440082e-14, 6.656289563680706e-12, 2.3825121706802e-14, 1.1119708844870987e-14, 7.331254822661314e-14, 6.26408911258951e-14, 1.0976703527844666e-13, 2.3825121706802e-14, 6.3548542455912406e-12, 3.2570514237403766e-14, 4.047859312449157e-14, 1.0594078624752675e-13, -1.8377402355335295e-13, 1.1119708844870987e-14, 3.2570514237403766e-14, 6.49880304377061e-12})
	E := mat.NewDense(6, 6, []float64{1.130290926636651, -0.07836538294209065, 0.015738385568786983, -0.649673140106426, -0.4536044749957713, 0.14334364029191377, 0, 1.0469912563915942, 0.3420650862788733, 0.1599513409111758, 0.11672951065862192, -0.3406444610170681, 0, 0, 1.1807478744239048, -0.2892797429357336, 0.09485219333518705, -0.42824085329432304, 0, 0, 0, 1.154953682326919, -0.1329712489844229, 0.21153069721367268, 0, 0, 0, 0, 1.1462579636908032, -0.08423225396399663, 0, 0, 0, 0, 0, 0.00019895149195974028})
	ref := riccatiExactReference(false, A, B, Q, R, nil, E, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("exact-E oracle did not certify")
	}
	const xBand, kBand = 1.5872480733037953e-14, 2.113731658368811e-05
	for _, ns := range []bool{false, true} {
		opts := &RiccatiOpts{E: E, NoScaling: ns}
		if red, _ := reduceInputs(A, B, Q, R, opts); red == nil || !red.rotated {
			t.Fatalf("NoScaling %v: inputs not rotated", ns)
		}
		res, err := Dare(A, B, Q, R, opts)
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
		t.Logf("NoScaling %v: Accuracy %d, X err %.3g (band %.3g), K err %.3g (band %.3g)", ns, res.Accuracy, eX, xBand, eK, kBand)
		if riccatiCertifiedWrong(res.Accuracy, eX, xBand, eK, kBand, newtonTol) {
			t.Errorf("NoScaling %v: certified-wrong: Accuracy 0 with X off by %.3g, K by %.3g", ns, eX, eK)
		}
		if dareRotatedInputsUnstable(A, B, E, res.K) {
			t.Errorf("NoScaling %v: E⁻¹(A − BK) unstable in %d bits", ns, riccatiExactPrec)
		}
	}
}

// TestDareRotatedInputsUnrefinedGainWithinFloorRefined: rotated inputs
// (n 5, m 4, R ~7e-13, E's last pivot 1.5e-5) whose unrefined K was 1.9e-6
// off the exact-E 320-bit gain (band 4.3e-9) at Accuracy 0 (M4LKAP; rv48r
// seed 4136): gainMissIn accepted it, as the gain of X moves 5e-7 per ulp
// of X (tolerance 4.2e-6), while the gain of X one Newton step on was
// within 8.7e-9. The bands are riccatiExactSensitivity(..., "ABQRSE") of
// this reference, pinned.
func TestDareRotatedInputsUnrefinedGainWithinFloorRefined(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{-0.5841825345794247, -0.3440717070890188, -0.12959541856493612, 0.11141751133232604, -0.1586275871651297, 0.46311937369386136, -0.14632175735824518, 0.15483445536350626, -0.22104662916609735, -0.467631899340577, -0.10246380740894212, 0.4015541025219238, 0.8951513817018157, 0.003777539065926909, 0.18929921159994959, -0.07786216129223318, 0.0842562962187307, -0.44944216418140237, -0.31428139223594176, 0.0025703588106172643, 0.4495800242080266, 0.0310598993147139, -0.6167122429772101, 0.056010556340855035, -0.14103439679386895})
	B := mat.NewDense(5, 4, []float64{0.44640911273616507, 0.11389976778377006, 0.6921131354055368, 0.24881244457857596, -0.02903583752690968, -0.007805549988942555, -0.04431957675702311, -0.0172693833096702, -0.10605462290468767, -0.026062127901601017, -0.16632809008527055, -0.05620179611019856, 0.1464000410372859, 0.03898489929420118, 0.22408138889475232, 0.08610063455980342, 0.178381164526886, 0.0452076010801929, 0.277102701945397, 0.09858250912347907})
	Q := mat.NewDense(5, 5, []float64{5456.288929824712, -3115.472305640989, 892.1993195844939, -3433.270642113965, -44.490135678558744, -3115.472305640989, 5070.191637230694, 624.407624662127, -916.8509703590398, -904.3785148814025, 892.1993195844939, 624.407624662127, 561.531496579672, -1827.3754908894102, -608.7137924316401, -3433.270642113965, -916.8509703590398, -1827.3754908894102, 7691.519372186855, 3926.426469901555, -44.490135678558744, -904.3785148814025, -608.7137924316401, 3926.426469901555, 3419.9074558460757})
	R := mat.NewDense(4, 4, []float64{7.030054309927255e-13, 6.406892585154254e-15, 1.9312905614061376e-14, -4.492769819792562e-14, 6.406892585154254e-15, 6.68858975302816e-13, 1.2208285217284012e-14, -3.6527421409940764e-16, 1.9312905614061376e-14, 1.2208285217284012e-14, 7.233888383383858e-13, -4.586209177303498e-15, -4.492769819792562e-14, -3.6527421409940764e-16, -4.586209177303498e-15, 7.070511943426042e-13})
	E := mat.NewDense(5, 5, []float64{1.1454397034701418, 0.2882709787792935, -0.06755473299413417, 0.3397107233057799, 0.11529351673408662, 0, 1.0128104894279837, 0.34128894414065536, 0.7443758454316614, -0.422857631869541, 0, 0, 1.0413214011330232, -0.16393032246388944, 0.4940298795946477, 0, 0, 0, 1.0281406818465948, -0.15316692585917466, 0, 0, 0, 0, 1.4664785409723689e-05})
	ref := riccatiExactReference(false, A, B, Q, R, nil, E, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("exact-E oracle did not certify")
	}
	const xBand, kBand = 1.4284004463093428e-11, 4.2889341364835306e-09
	for _, ns := range []bool{false, true} {
		opts := &RiccatiOpts{E: E, NoScaling: ns}
		if red, _ := reduceInputs(A, B, Q, R, opts); red == nil || !red.rotated {
			t.Fatalf("NoScaling %v: inputs not rotated", ns)
		}
		res, err := Dare(A, B, Q, R, opts)
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
		t.Logf("NoScaling %v: Accuracy %d, X err %.3g (band %.3g), K err %.3g (band %.3g)", ns, res.Accuracy, eX, xBand, eK, kBand)
		if res.Accuracy != 0 {
			t.Errorf("NoScaling %v: Accuracy %d", ns, res.Accuracy)
		}
		if riccatiCertifiedWrong(res.Accuracy, eX, xBand, eK, kBand, newtonTol) {
			t.Errorf("NoScaling %v: certified-wrong: Accuracy 0 with X off by %.3g, K by %.3g", ns, eX, eK)
		}
		if dareRotatedInputsUnstable(A, B, E, res.K) {
			t.Errorf("NoScaling %v: E⁻¹(A − BK) unstable in %d bits", ns, riccatiExactPrec)
		}
	}
}

// TestDareRotatedInputsRefinedGainBeyondFloorHewer: rotated inputs
// (n 3, m 3, R ~4e-10, E's last pivot 7.5e-5) whose unrefined K was off
// by O(1) and refined to within gainTol but beyond gainFloor of the gain
// of its X, 2.3e-6 off the exact-E 320-bit gain (band 1.1e-7) at
// Accuracy 0 natively, 5.4e-6 on amd64 (AT3HCA; rv49v seed 4403). A
// second refinement was within 5.3e-8 natively but 4.4e-6 on amd64
// (refinement cycles among X a few ulps apart); the Hewer gain of X was
// exact on both. The bands are
// riccatiExactSensitivity(..., "ABQRSE") of this reference, pinned.
func TestDareRotatedInputsRefinedGainBeyondFloorHewer(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{-0.1715478898601427, 0.16752989143887573, 0.07669486025546303, -0.3828044376881159, -0.2217838071345253, -0.1733414643177667, 0.11200079612909754, 0.29972133873324125, -0.01187752910411909})
	B := mat.NewDense(3, 3, []float64{-0.0044252317289711525, 0.008788066916732324, -0.00040508011592204794, 0.11827735565472396, -0.29633361973433536, -0.0056586607136435004, 0.35061243329098624, -0.880250962948512, -0.01726744763850466})
	Q := mat.NewDense(3, 3, []float64{6.488509821945091e+07, -1.5380440519737657e+07, 7.33986726722479e+07, -1.5380440519737657e+07, 1.2939955476131309e+07, -1.352656757673153e+06, 7.33986726722479e+07, -1.352656757673153e+06, 1.1073159268190524e+08})
	R := mat.NewDense(3, 3, []float64{4.259539859850293e-10, 1.0920974537143577e-12, 2.0713438824064635e-12, 1.0920974537143577e-12, 4.327462133628632e-10, 1.2108899146932686e-12, 2.0713438824064635e-12, 1.2108899146932686e-12, 4.30313930728902e-10})
	E := mat.NewDense(3, 3, []float64{1.1311141592119605, -0.24638606260717266, 0.16565715678543022, 0, 1.044040477615047, -0.4597924593730896, 0, 0, 7.525378985751742e-05})
	ref := riccatiExactReference(false, A, B, Q, R, nil, E, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("exact-E oracle did not certify")
	}
	const xBand, kBand = 8.620199149222536e-16, 1.1411321515005277e-07
	for _, ns := range []bool{false, true} {
		opts := &RiccatiOpts{E: E, NoScaling: ns}
		if red, _ := reduceInputs(A, B, Q, R, opts); red == nil || !red.rotated {
			t.Fatalf("NoScaling %v: inputs not rotated", ns)
		}
		res, err := Dare(A, B, Q, R, opts)
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
		t.Logf("NoScaling %v: Accuracy %d, X err %.3g (band %.3g), K err %.3g (band %.3g)", ns, res.Accuracy, eX, xBand, eK, kBand)
		if res.Accuracy != 0 {
			t.Errorf("NoScaling %v: Accuracy %d", ns, res.Accuracy)
		}
		if riccatiCertifiedWrong(res.Accuracy, eX, xBand, eK, kBand, newtonTol) {
			t.Errorf("NoScaling %v: certified-wrong: Accuracy 0 with X off by %.3g, K by %.3g", ns, eX, eK)
		}
		if dareRotatedInputsUnstable(A, B, E, res.K) {
			t.Errorf("NoScaling %v: E⁻¹(A − BK) unstable in %d bits", ns, riccatiExactPrec)
		}
	}
}
