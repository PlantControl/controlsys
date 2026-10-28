package controlsys

import (
	"math"
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Under cheap control (m > n) Dare certifies in the reduced inputs (red.bw,
// red.rw, red.sw rounded to float64). There neither a K refined in
// double-double nor the gain of an X converged to eps is kept as terminal,
// nor is X stepped by such a refinement (Y6BJEX am.4): each left K or X
// off the 320-bit solution at Accuracy 0.

// dareReducedInputsOracle returns the errors of the X and K of res against
// the stabilizing solution of Hewer's iteration in 320 bits on the explicit
// model from the K of res, A − BK checked stable (Y need not be definite:
// Q − SR⁻¹Sᵀ is not).
func dareReducedInputsOracle(t *testing.T, A, B, Q, R, S, E *mat.Dense, res *RiccatiResult) (eX, eK float64) {
	t.Helper()
	n, _ := A.Dims()
	if E == nil {
		E = eye(n)
	}
	e := hbFrom(E)
	a, b := hbSolve(e, hbFrom(A)), hbSolve(e, hbFrom(B))
	q, r, s := hbFrom(Q), hbFrom(R), hbFrom(S)
	k := hbFrom(res.K)
	var y, prev hbMat
	conv := false
	for range 40 {
		y = hewerDareHB(a, b, q, r, s, k, 1)
		btx := hbMul(b.t(), y)
		rh := hbMul(btx, a)
		if s.r > 0 {
			rh = hbAdd(rh, s.t(), 1)
		}
		k = hbSolve(hbAdd(r, hbMul(btx, b), 1), rh)
		if prev.r > 0 {
			if d, yn := hbDense(hbAdd(y, prev, -1)), hbDense(y); mat.Norm(d, math.Inf(1)) <= 1e-60*mat.Norm(yn, math.Inf(1)) {
				conv = true
				break
			}
		}
		prev = y
	}
	var ev mat.Eigen
	if !conv || !ev.Factorize(hbDense(hbAdd(a, hbMul(b, k), -1)), mat.EigenNone) {
		t.Fatal("oracle did not converge")
	}
	for _, v := range ev.Values(nil) {
		if !(cmplx.Abs(v) < 1) {
			t.Fatalf("oracle closed-loop eigenvalue %v", v)
		}
	}
	z := hbSolve(e.t(), y)
	return relErr(res.X, hbDense(hbSolve(e.t(), z.t()).t())), relErr(res.K, hbDense(k))
}

// TestDareDescriptorReducedInputsRefinedGainNotTerminal: n 3, m 5, cheap R
// ~1e-11, cross term, E well conditioned (O2MVDD rv39 seed 13857). The
// double-double K refined in the reduced inputs is not terminal. Against
// the 320-bit solution of the given E (not of E⁻¹A, E⁻¹B rounded, which
// alone moves K by 2.7e-6) an eps perturbation of Q moves K by 2.6e-6, so
// K is held to 1e-5: base 2.7e-6, the refined K 9e-13 natively and 2.7e-6
// on amd64.
func TestDareDescriptorReducedInputsRefinedGainNotTerminal(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.16512989717622614, -0.12691217974423774, -0.2625463172359406, -0.8743286070464587, 0.7519938135677892, 0.20108657681406852, -0.5805761169577841, -0.0882202476765931, 0.3784188768030361})
	B := mat.NewDense(3, 5, []float64{-1.1138365738913891, 0.33343041543455343, 0.6432335154959963, -1.1093050809476748, -1.031386719918751, -0.04093552261555988, -0.5570703531550396, -1.51147584566615, -1.670415473655939, -1.2440533806586151, 0.1014263938732878, -0.880569745044113, -0.2664934631444428, 0.7301539437474831, 0.3795621261446531})
	Q := mat.NewDense(3, 3, []float64{1.258436206180033e+11, -7.620183351058147e+10, 3.161182331074342e+11, -7.620183351058147e+10, 9.839979165565472e+11, -1.0300894172230897e+12, 3.161182331074342e+11, -1.0300894172230897e+12, 1.544062468787748e+12})
	R := mat.NewDense(5, 5, []float64{1.0006048518127824e-11, -5.272352722285883e-13, -6.537006089345152e-13, 7.753345952230884e-13, -1.0834987382214139e-13, -5.272352722285883e-13, 9.874112315328928e-12, -9.691583544745983e-14, -2.7645868680294535e-13, 1.657409567862452e-13, -6.537006089345152e-13, -9.691583544745983e-14, 9.31632646110756e-12, -5.16955048371851e-13, -1.2899303371094128e-13, 7.753345952230884e-13, -2.7645868680294535e-13, -5.16955048371851e-13, 9.340401103997703e-12, 5.03129926842264e-13, -1.0834987382214139e-13, 1.657409567862452e-13, -1.2899303371094128e-13, 5.03129926842264e-13, 1.0001121368837716e-11})
	S := mat.NewDense(3, 5, []float64{-0.006016709407351466, 0.001323427924358537, 0.0029056073955610077, -0.012203720922295528, -0.01826452801999528, -0.05001425399901001, 0.05282724820612193, 0.01710699602118271, -0.018591708238488363, 0.039893454500025426, 0.03286896187091784, -0.044632589161792274, -0.009572306330212433, -0.0074219625679628285, -0.07166476234837728})
	E := mat.NewDense(3, 3, []float64{1.1858111144029029, -0.10985818610205345, -0.07221844004189239, -0.10358073796070186, 0.9912796253522691, -0.0016751547174243854, -0.12244927243585321, -0.1581051545198203, 1.1366520159218645})
	for _, ns := range []bool{false, true} {
		res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, S: S, NoScaling: ns})
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		eX, eK := dareReducedInputsOracle(t, A, B, Q, R, S, E, res)
		t.Logf("NoScaling %v: Accuracy %d, X err %.3g, K err %.3g", ns, res.Accuracy, eX, eK)
		if res.Accuracy == 0 && (eX > newtonTol || eK > 1e-5) {
			t.Errorf("NoScaling %v: Accuracy 0 with X off by %.3g, K by %.3g", ns, eX, eK)
		}
	}
}

// TestDareDescriptorReducedInputsUnsteppedXNotCertified: n 5, m 6, cheap R
// ~1.6e-12, E well conditioned (O2MVDD rv39 seed 34852). The reduced-input
// refinement kept K, but its X (Newton step 9.97e-9) was 1.25e-8 off the
// 320-bit X at Accuracy 0; base returned Accuracy 1.
func TestDareDescriptorReducedInputsUnsteppedXNotCertified(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{-0.6579921975889437, -0.21785485027929902, -0.16439376849858942, -0.21775233020019524, 0.09414096147840115, 0.020004752764177135, 0.33176945542239583, 0.4045351460587164, -0.7134576942848542, 0.3010520268853596, -0.18761654191918722, -0.06850121137806965, 0.20667134041682578, 0.5804050422020575, 0.44944285233812553, 0.5161921239325968, -0.4110794317302901, 0.20311987611002918, 0.8491470864365115, 0.15936396402768443, 0.29084454476336286, -0.20364762032562578, -0.1878548542736646, 0.2175248399099882, 0.40166582556243313})
	B := mat.NewDense(5, 6, []float64{-1.1677344463272226, -0.2959020588842308, -1.7531788262777144, -0.5272194186878238, 1.4028257508956532, 0.04136891008158422, 0.24694869132582659, 0.7391096877531709, -1.035819462500881, -0.6337533339033032, 1.9451326361410226, 0.3773549924481185, 0.22270002744853634, -0.8316812402982453, 0.34017418181744463, -0.7933060324329599, 0.22915811205823222, -2.2733335292243337, -0.3933755636413897, -0.1998957217925894, 0.9242449459980957, 0.9713275436422839, 0.4486884126389612, 0.29545988743476315, 2.034093867696889, 1.2071699817639607, 0.9817771856029418, -0.17132588562755607, 0.32132275907259, 0.06839190595772218})
	Q := mat.NewDense(5, 5, []float64{1.513060918796445e+12, -6.861490982295869e+11, -3.7403335287384467e+11, -1.1713186043321814e+12, 4.235486958073736e+10, -6.861490982295869e+11, 9.504040796698053e+11, 4.558549489263327e+10, 2.2326371353725168e+11, -1.044759361371651e+12, -3.7403335287384467e+11, 4.558549489263327e+10, 8.352621879188213e+11, 3.681768824994612e+11, 6.918000892064099e+11, -1.1713186043321814e+12, 2.2326371353725168e+11, 3.681768824994612e+11, 1.5416940879512139e+12, 1.1798306790998633e+11, 4.235486958073736e+10, -1.044759361371651e+12, 6.918000892064099e+11, 1.1798306790998633e+11, 2.2602544739168413e+12})
	R := mat.NewDense(6, 6, []float64{1.5779375686682818e-12, -2.0608127037435347e-14, -2.242592135634772e-14, 3.3198336954669306e-15, 1.923748387688546e-14, -6.661969088908642e-14, -2.0608127037435347e-14, 1.5829272205280108e-12, 1.9550332356234864e-14, -9.977955106021387e-15, 9.372403104814777e-14, -3.3057272514660906e-14, -2.242592135634772e-14, 1.9550332356234864e-14, 1.536340205649908e-12, 3.661360823363816e-14, 2.6568846005918142e-14, 1.7002444747737617e-14, 3.3198336954669306e-15, -9.977955106021387e-15, 3.661360823363816e-14, 1.5419810212094235e-12, 2.977512144435967e-14, 3.032639816787587e-14, 1.923748387688546e-14, 9.372403104814777e-14, 2.6568846005918142e-14, 2.977512144435967e-14, 1.6436843524501524e-12, -4.6404000482889246e-14, -6.661969088908642e-14, -3.3057272514660906e-14, 1.7002444747737617e-14, 3.032639816787587e-14, -4.6404000482889246e-14, 1.5738708882113034e-12})
	E := mat.NewDense(5, 5, []float64{0.9713189477084567, -0.3613259093210955, -0.23579723263809016, 0.05484436307734027, -0.36816299495050203, -0.037968895824093005, 1.108729701305503, -0.0023218160511450092, -0.004640677910160262, 0.08112140658431947, 0.07766527287837886, -0.04489345086778834, 1.108331987370999, -0.21353079059735428, -0.10039791025785894, 0.2403396276536591, -0.17038174413709858, -0.2801532407359234, 0.9348398616393371, 0.10664492279602827, 0.09306366501899203, -0.10655924753042285, -0.014422732822463934, 0.04699585439455927, 1.1732361492964607})
	for _, ns := range []bool{false, true} {
		res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, NoScaling: ns})
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		eX, eK := dareReducedInputsOracle(t, A, B, Q, R, nil, E, res)
		t.Logf("NoScaling %v: Accuracy %d, X err %.3g, K err %.3g", ns, res.Accuracy, eX, eK)
		if res.Accuracy == 0 && (eX > newtonTol || eK > gainFloor) {
			t.Errorf("NoScaling %v: Accuracy 0 with X off by %.3g, K by %.3g", ns, eX, eK)
		}
	}
}

// TestDareCrossTermReducedInputsGainNotSubstituted: n 3, m 4, cheap R
// ~1.2e-12, cross term, no E (O2MVDD rv39 seed 10831 as generated on
// amd64). On amd64 X is exact to 6.9e-17, yet its gain, substituted for
// K, was 4.7e-6 off the 320-bit gain (as was the 320-bit gain of that X;
// an eps perturbation of Q moves K by 8e-8; base 6.9e-7 and, with
// NoScaling, 1.39e-6). K is now the gain of X taken on by Hewer steps in
// extended precision (D6QQX7).
func TestDareCrossTermReducedInputsGainNotSubstituted(t *testing.T) {
	A := mat.NewDense(3, 3, []float64{0.11277767345321332, -0.12701086166535683, -0.3282418505397938, -0.6162985556828814, 0.20592104971302708, -0.12800032343860365, 0.0907954836845478, 0.10112538417271175, -0.4293440070156919})
	B := mat.NewDense(3, 4, []float64{-1.3460581287057858, -0.4303380677629103, -1.118400476700266, -0.4753542311719018, -1.0230348133871177, -1.5094495481218706, 0.8812892138062883, 0.544558646697797, 2.9391988593246956, -0.03745218545200002, -1.8306089580990381, 0.6069334486152766})
	Q := mat.NewDense(3, 3, []float64{2.384539871166807e+11, -4.376977359214548e+10, -1.987096945473849e+11, -4.376977359214548e+10, 1.6042916278752838e+10, 6.73731869398668e+10, -1.987096945473849e+11, 6.73731869398668e+10, 2.848017535679012e+11})
	R := mat.NewDense(4, 4, []float64{1.2020972856242958e-12, -1.1259540862215746e-14, 7.820449669239088e-15, -3.1770976235837157e-14, -1.1259540862215746e-14, 1.180783602283751e-12, 1.0477541448173887e-14, -2.1895375070522303e-14, 7.820449669239088e-15, 1.0477541448173887e-14, 1.172101703370729e-12, -5.005197143593813e-15, -3.1770976235837157e-14, -2.1895375070522303e-14, -5.005197143593813e-15, 1.1865001044155857e-12})
	S := mat.NewDense(3, 4, []float64{4.738256256751149e-08, 7.533326484301201e-07, 1.4390883235646677e-07, -4.664061119321964e-08, 7.61789153108352e-08, -1.1744669584943483e-07, 3.481002938461837e-08, -9.13013599018104e-08, 2.8798043731460986e-07, -5.473965105421443e-07, 1.1629407766604448e-07, -3.464179514743311e-07})
	for _, ns := range []bool{false, true} {
		res, err := Dare(A, B, Q, R, &RiccatiOpts{S: S, NoScaling: ns})
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		eX, eK := dareReducedInputsOracle(t, A, B, Q, R, S, nil, res)
		t.Logf("NoScaling %v: Accuracy %d, X err %.3g, K err %.3g", ns, res.Accuracy, eX, eK)
		if res.Accuracy == 0 && (eX > newtonTol || eK > 1e-7) {
			t.Errorf("NoScaling %v: Accuracy 0 with X off by %.3g, K by %.3g", ns, eX, eK)
		}
	}
}

// TestDareDescriptorDuplicateInputsGainOfConvergedX: n 2, m 4 with three
// duplicated input columns, cheap R ~1.9e-12, tiny cross term, E well
// conditioned (rv39 seed 1342, D6QQX7). X is exact to 2e-16, while the
// pencil K was 9e-5 off the exact-E 320-bit gain (its eps-sensitivity
// 4e-4); the gain of X taken on by Hewer steps in extended precision is
// exact. A K off the exact gain by more than 1e-7 must be the gain of its
// own X (320-bit, dareGainBig): on amd64 without NoScaling it is, 1.4e-6
// off with X exact, inside the band, and is kept.
func TestDareDescriptorDuplicateInputsGainOfConvergedX(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{-0.28864518346838963, -0.549870329431709, -0.6243674170543952, 0.8833863280696619})
	B := mat.NewDense(2, 4, []float64{0.047175451585948935, -0.5961903208497157, -0.5961903208497157, -0.5961903208497157, -0.06500131021983324, 0.9803468818573009, 0.9803468818573009, 0.9803468818573009})
	Q := mat.NewDense(2, 2, []float64{59.22482557898569, -117.240527973113, -117.240527973113, 232.08749481385476})
	R := mat.NewDense(4, 4, []float64{1.8689544268484782e-12, 1.6790511259354945e-14, 1.7683876130756696e-14, -5.764886519549634e-14, 1.6790511259354945e-14, 1.8593955819544937e-12, -1.1495510171279438e-14, -1.681607306145989e-14, 1.7683876130756696e-14, -1.1495510171279438e-14, 1.8630544642286726e-12, -4.4225454168101175e-14, -5.764886519549634e-14, -1.681607306145989e-14, -4.4225454168101175e-14, 1.9383488945488344e-12})
	S := mat.NewDense(2, 4, []float64{-1.3842151587849728e-16, 4.124675003603732e-17, -2.0137472633148114e-16, 3.9939457662832224e-16, 2.740170434573968e-16, -8.165141398265057e-17, 3.986382231562215e-16, -7.906351867774363e-16})
	E := mat.NewDense(2, 2, []float64{0.9141292581144288, -0.16189848878288213, -0.19642846953561233, 0.7158936931290532})
	ref := riccatiExactReference(false, A, B, Q, R, S, E, riccatiExactPrec)
	if !ref.OK {
		t.Fatal("reference not certified")
	}
	for _, ns := range []bool{false, true} {
		res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, S: S, NoScaling: ns})
		if err != nil {
			t.Fatalf("NoScaling %v: %v", ns, err)
		}
		eX, eK := relErr(res.X, ref.X), relErr(res.K, ref.K)
		eG := relErr(res.K, dareGainBig(A, B, R, S, res.X))
		t.Logf("NoScaling %v: Accuracy %d, X err %.3g, K err %.3g, off the gain of X by %.3g", ns, res.Accuracy, eX, eK, eG)
		if res.Accuracy == 0 && (eX > newtonTol || eK > 1e-7 && eG > 1e-9) {
			t.Errorf("NoScaling %v: Accuracy 0 with X off by %.3g, K by %.3g (%.3g off the gain of X)", ns, eX, eK, eG)
		}
	}
}
