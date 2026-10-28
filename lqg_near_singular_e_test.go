package controlsys

import (
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// descriptorNoiseGainBig is Lqg's current-estimator noise gain
// (R + BᵀXB)⁻¹BᵀX of the descriptor Dare solution X with cross term S
// (nil for none), all in hbPrec bits: Y by SDA doubling on E⁻¹A, E⁻¹B
// (never rounded), X = E⁻ᵀYE⁻¹.
func descriptorNoiseGainBig(A, B, Q, R, S, E *mat.Dense) (*mat.Dense, bool) {
	e, b := hbFrom(E), hbFrom(B)
	y, conv := dareDoublingHB(hbSolve(e, hbFrom(A)), hbSolve(e, b), hbFrom(Q), hbFrom(R), hbFrom(S), 120)
	if !conv {
		return nil, false
	}
	x := hbSolve(e.t(), hbSolve(e.t(), y).t()).t()
	btx := hbMul(b.t(), x)
	return hbDense(hbSolve(hbAdd(hbFrom(R), hbMul(btx, b), 1), btx)), true
}

// Lqg Current on a stabilizable near-singular-E plant (cond(E) 3.65e9,
// ‖X‖ ~ 1/σ_min(E)²): the regulator comes from Dare's extended-precision
// solve, and Kw = (R + BᵀXB)⁻¹BᵀX formed from the float64 X cancelled to
// 0.70 off at Accuracy 0 (O2YDYA review, edge seed 51; base
// errBoundaryUnstabilizable). Kw moves 8e-16 under eps perturbations.
func TestLqg_CurrentNearSingularENoiseGainMatchesOracle(t *testing.T) {
	A := mat.NewDense(5, 5, []float64{0.7123473565019564, 1.389425245054006, -1.620620132975544, -1.026736166016733, 0.9197761128014154, 1.0899289664113874, 1.8254623820931632, -0.66935705804177, -1.5991262264754011, 0.08346406091413627, 0.06316890114284349, 1.1994269961909554, -0.1117527353602423, -1.0586260295884946, 2.786824429733568, -0.6842661982961946, 1.7763648791316269, -2.286568656513844, 2.172175300864322, -0.6918711176717713, -1.0285502143496843, -0.058326516800719305, -0.8215246201882281, -1.9547682781643603, -1.2121967908963387})
	B := mat.NewDense(5, 2, []float64{-1.0685118297800806, 0.016166156715667394, 0.5977296206786947, 0.08499342043555719, 0.15628568598477816, 1.332258027258465, -0.17557417339153933, -1.3047420514265624, 1.128924057080265, -0.2797798523987997})
	E := mat.NewDense(5, 5, []float64{0.43507398531972813, -0.8742255980591255, -0.5106629553756165, -1.0479750759424344, -0.8116461662599351, 0.9168585049002512, 1.4206600474381792, 1.5071877040866621, 0.5064954095971331, -2.6001477148660745, 0.4560870077845191, -0.22231452138481386, -0.05940110933586429, 0.5034706732677925, 0.036852840156955405, -0.04587083533169269, 0.20627531512518937, -0.28918327644639924, -0.10996404510105914, 0.35762257511420315, 1.7770997371919333, 0.38604852200758505, -0.07277273966513896, 0.21208746551821284, -1.694120097767178})
	Q := mat.NewDense(5, 5, []float64{6.657549588624399, 4.410892534611584, -6.832387991043152, 2.171063965708072, -2.149201426120526, 4.410892534611584, 4.228668867735996, -3.6060248346286246, 1.030847264195174, -0.9998091956449603, -6.832387991043152, -3.6060248346286246, 14.357225640793184, -3.177614634793004, 4.0384796020452605, 2.171063965708072, 1.030847264195174, -3.177614634793004, 1.1118610974818954, -1.718371227613103, -2.149201426120526, -0.9998091956449603, 4.0384796020452605, -1.718371227613103, 4.020100655687023})
	R := mat.NewDense(2, 2, []float64{0.7306262818179861, 0.29938406008382973, 0.29938406008382973, 2.227625211673149})
	C := mat.NewDense(1, 5, []float64{1.6840207559434308, 0.6618344647930041, -0.4347790640815497, -0.5315900659866536, 0.26403849045331})
	checkLqgCurrentNoiseGain(t, A, B, C, E, Q, R, nil, mat.NewDense(1, 1, []float64{0.5004178462278779}))
}

// Lqg Current where the regulator Dare solves in float64 and only the
// Kalman dual needs the extended-precision solve (cond(E) 7.2e10, base
// ErrNoStabilizing): Kw of the float64 X was 4.1e-7 off at Accuracy 0
// (O2YDYA review 2, mixed seed 82; Kw moves 3e-16 under eps
// perturbations).
func TestLqg_CurrentNearSingularEFloatRegulatorNoiseGain(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{444562.99535538786, 0, -0.22594540719588263, -0.9016891657118802})
	B := mat.NewDense(2, 2, []float64{4.071401339571243e-11, -4.6407160979664015e-10, 0.5375947195865346, 0.3398117257077149})
	E := mat.NewDense(2, 2, []float64{1.0888022376440525, 0, 0, 1.5051327511388926e-11})
	Q := mat.NewDense(2, 2, []float64{1.3780019980278215, -1.8139079669120497, -1.8139079669120497, 6.479179684656806})
	R := mat.NewDense(2, 2, []float64{1.4774236832145244, -0.43040918076399104, -0.43040918076399104, 0.7298443780640265})
	C := mat.NewDense(1, 2, []float64{1.3221467858160478, 2.029922235601939})
	checkLqgCurrentNoiseGain(t, A, B, C, E, Q, R, nil, mat.NewDense(1, 1, []float64{0.7330290179542455}))
}

// checkLqgCurrentNoiseGain runs Lqg Current on (A, B, C, 0, E) with
// weights [Q S; Sᵀ R] (S nil for none), blkdiag(I, Rn) and requires Accuracy 0 with Xc,
// K and Kw within newtonTol of the 320-bit doubling oracles.
func checkLqgCurrentNoiseGain(t *testing.T, A, B, C, E, Q, R, S, Rn *mat.Dense) {
	t.Helper()
	n, m := B.Dims()
	p, _ := C.Dims()
	sys, err := NewDescriptor(A, B, C, mat.NewDense(p, m, nil), E, 1)
	if err != nil {
		t.Fatal(err)
	}
	qxu := mat.NewDense(n+m, n+m, nil)
	setBlock(qxu, 0, 0, Q)
	setBlock(qxu, n, n, R)
	if S != nil {
		setBlock(qxu, 0, n, S)
		setBlock(qxu, n, 0, mat.DenseCopyOf(S.T()))
	}
	qwv := eyeDense(n + p)
	setBlock(qwv, n, n, Rn)
	want, ok := descriptorNoiseGainBig(A, B, Q, R, S, E)
	if !ok {
		t.Fatal("doubling oracle does not converge")
	}
	_, wantY, wantK, rho, ok := descriptorDareDoublingBig(A, B, Q, R, S, E)
	if !ok {
		t.Fatalf("doubling oracle: no stabilizing solution (ρ %.3g)", rho)
	}
	res, err := Lqg(sys, qxu, qwv, &LqgOpts{Current: true})
	if err != nil {
		t.Fatal(err)
	}
	kw, _ := res.Kw()
	if res.Accuracy != 0 {
		t.Errorf("Accuracy %d, want 0", res.Accuracy)
	}
	for _, d := range []struct {
		name      string
		got, want *mat.Dense
	}{{"Xc", res.Xc, wantY}, {"K", res.K, wantK}, {"Kw", kw, want}} {
		if rel := relErr(d.got, d.want); !(rel <= newtonTol) {
			t.Errorf("Accuracy %d with %s off by %.3g", res.Accuracy, d.name, rel)
		}
	}
}

// lqgCurrent runs Lqg Current on (A, B, C, 0, E) with weights [Q S; Sᵀ R]
// and blkdiag(I, Rn).
func lqgCurrent(t *testing.T, A, B, C, E, Q, R, S, Rn *mat.Dense) *LqgResult {
	t.Helper()
	n, m := B.Dims()
	p, _ := C.Dims()
	sys, err := NewDescriptor(A, B, C, mat.NewDense(p, m, nil), E, 1)
	if err != nil {
		t.Fatal(err)
	}
	qxu := mat.NewDense(n+m, n+m, nil)
	setBlock(qxu, 0, 0, Q)
	setBlock(qxu, n, n, R)
	setBlock(qxu, 0, n, S)
	setBlock(qxu, n, 0, mat.DenseCopyOf(S.T()))
	qwv := eyeDense(n + p)
	setBlock(qwv, n, n, Rn)
	res, err := Lqg(sys, qxu, qwv, &LqgOpts{Current: true})
	if err != nil {
		t.Fatal(err)
	}
	return res
}

// floatKw is the float64 Kw = (R + BᵀXB)⁻¹BᵀX of Dare's descriptor X,
// what Lqg Current returned before the extended-precision Kw existed.
func floatKw(t *testing.T, A, B, E, Q, R, S *mat.Dense) *mat.Dense {
	t.Helper()
	res, err := Dare(A, B, Q, R, &RiccatiOpts{E: E, S: S})
	if err != nil {
		t.Fatal(err)
	}
	n, _ := A.Dims()
	kw, err := floatNoiseGain(B, res.X, R, n)
	if err != nil {
		t.Fatal(err)
	}
	return kw
}

func requireEqualDense(t *testing.T, name string, got, want *mat.Dense) {
	t.Helper()
	if !mat.Equal(got, want) {
		t.Errorf("%s %v, want %v", name, got.RawMatrix().Data, want.RawMatrix().Data)
	}
}

// mixed generator seed 1180 (O2YDYA round-2 sweep): cond(E) 2.2e10, a
// cross term, the regulator Dare solved in float64. The float64 Kw is
// 1.4e-7 off at Accuracy 0 (base); the extended-precision Kw matches the
// 320-bit oracle. When that solve hits an iteration cap, Lqg returns
// exactly the float64 Kw, at Accuracy 1.
func TestLqg_CurrentNoiseGainIterationCapKeepsFloatGain(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{99044.01351554065, 0, 1.0487692324354092, -0.6953941759371415})
	B := mat.NewDense(2, 2, []float64{-2.610506420650651e-07, -1.8170173245948552e-07, 0.6457960508325897, -0.08343661377617334})
	E := mat.NewDense(2, 2, []float64{0.9559215480825146, 0, 0, 4.4012666202884514e-11})
	Q := mat.NewDense(2, 2, []float64{12.203399920996048, -0.9582545390033572, -0.9582545390033572, 0.39170958224428754})
	R := mat.NewDense(2, 2, []float64{2.3653075853813097, 0.06401298917217467, 0.06401298917217467, 0.5240556077252992})
	S := mat.NewDense(2, 2, []float64{-1.9701701619311962, -0.2444165670281241, 0.06715750757897745, 0.008331466880266421})
	C := mat.NewDense(1, 2, []float64{-1.5857467997528196, -1.323027260137359})
	Rn := mat.NewDense(1, 1, []float64{0.5056739899770474})
	checkLqgCurrentNoiseGain(t, A, B, C, E, Q, R, S, Rn)
	want := floatKw(t, A, B, E, Q, R, S)
	for _, c := range []struct {
		name  string
		limit *int
		v     int
	}{{"stableLoop", &stableSquarings, 1}, {"smithStein", &steinDoublings, 1}} {
		t.Run(c.name, func(t *testing.T) {
			withIterationCap(t, c.limit, c.v)
			res := lqgCurrent(t, A, B, C, E, Q, R, S, Rn)
			kw, _ := res.Kw()
			requireEqualDense(t, "Kw", kw, want)
			if res.Accuracy != 1 {
				t.Errorf("Accuracy %d, want 1", res.Accuracy)
			}
		})
	}
}

// bnd generator seed 1195 (O2YDYA round-2 review): cond(E) 2.5e9 sends
// Kw to the extended-precision check, which took 27 s, but the float64
// Kw is within 1.3e-16 of the 320-bit oracle: Lqg keeps it, bit for bit,
// at Accuracy 0.
func TestLqg_CurrentNoiseGainKeepsAccurateFloatGain(t *testing.T) {
	A := mat.NewDense(4, 4, []float64{0.9183556219745826, 0, 0, 0, -2.2969114693285295, 0.08261525495786126, -1.3948025825870896, 0.8463873280314725, -2.403360915759004, 2.3204042332203, 0.9502594851010887, 0.5376189498275064, 0.5669414789724267, 0.0567055835576058, -1.2986117587904453, 0.9537360149593903})
	B := mat.NewDense(4, 1, []float64{0, 0.6055840707414784, -1.067654775309174, 0.2821496973777199})
	E := mat.NewDense(4, 4, []float64{0.9183556228929383, 0, 0, 0, 0, 0.9267457902072246, -0.22368097616560978, -0.12603908158672228, 0, 0, 1.2524916458443114, -1.2119825492005298, 0, 0, 0, 1.0014631606607184e-09})
	Q := mat.NewDense(4, 4, []float64{1.0256199901950613, -0.04413993039679562, 0.07006376489200947, 0.19176594480422388, -0.04413993039679562, 1.0760473927039023, -0.12071080754226411, -0.33038792722746024, 0.07006376489200947, -0.12071080754226411, 1.191605504665222, 0.5244281503919992, 0.19176594480422388, -0.33038792722746024, 0.5244281503919992, 2.4353704785470742})
	R := mat.NewDense(1, 1, []float64{1.0275889074156928})
	S := mat.NewDense(4, 1, []float64{-0.02658622834255557, 0.04580463378865494, -0.07270616568452702, -0.19899824933866586})
	C := mat.NewDense(1, 4, []float64{0.5092856705191796, -0.20603704357593627, -1.3369952145435433, -0.8249058573815162})
	Rn := mat.NewDense(1, 1, []float64{1.8324828128836743})
	checkLqgCurrentNoiseGain(t, A, B, C, E, Q, R, S, Rn)
	res := lqgCurrent(t, A, B, C, E, Q, R, S, Rn)
	kw, _ := res.Kw()
	requireEqualDense(t, "Kw", kw, floatKw(t, A, B, E, Q, R, S))
}
