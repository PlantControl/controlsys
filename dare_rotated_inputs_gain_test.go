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
