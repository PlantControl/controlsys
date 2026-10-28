package controlsys

import (
	"errors"
	"runtime"
	"testing"
	"time"

	"plantcontrol.org/v1/gonum/mat"
)

// withIterationCap sets an extended-precision iteration cap for one test.
func withIterationCap(t *testing.T, limit *int, v int) {
	t.Helper()
	old := *limit
	*limit = v
	t.Cleanup(func() { *limit = old })
}

// boundaryError is the error base Dare returned (before the extended
// solve existed) on the stabilizable near-singular-E plants below.
const boundaryError = "Dare: controlsys: no stabilizing solution exists: a mode on or beyond the stability boundary is unstabilizable"

// A near-singular-E Dare, Dlqr or Kalman whose extended-precision solve
// hits an iteration cap (SDA doubling or its exponent spread, stableLoop
// squaring or the Smith doubling of the perturbed Hewer step) returns exactly the float64
// pencil's error, boundary verdict included, as before the extended solve
// existed (O2YDYA round 3).
func TestDare_NearSingularEIterationCapKeepsFloatError(t *testing.T) {
	A := mat.NewDense(2, 2, []float64{0.42832161169283334, 1.0362853424557894, 0.3431284077914542, -0.7926130901776904})
	B := mat.NewDense(2, 2, []float64{2.4606698336961226, 0.8073964829247321, -2.5367599785728485, -0.7294612311954138})
	E := mat.NewDense(2, 2, []float64{1.0831741096873624, -0.084731321391598, 0, 1.7497519700889574e-15})
	KA := mat.NewDense(2, 2, []float64{0.923010477433099, 2.0820636924475, -0.9200537881286719, -1.361812134364599})
	KB := mat.NewDense(2, 3, []float64{-0.7649133317207344, -0.0015390678071096064, 1.2371736568788594, 0.6699795793910823, 0.37405306155207996, 1.3833774115150517})
	KE := mat.NewDense(2, 2, []float64{0.5004371458498397, 2.921975405177471, 0, 8.520053948604121e-13})
	Rn := mat.NewDense(3, 3, []float64{4.7486349226800515, -1.1064545236921663, 1.512245701955511, -1.1064545236921663, 2.42778007790058, 0.6521422583478569, 1.512245701955511, 0.6521422583478569, 3.6215308801989963})
	sys, err := NewDescriptor(KA, KB, mat.DenseCopyOf(KB.T()), mat.NewDense(3, 3, nil), KE, 1)
	if err != nil {
		t.Fatal(err)
	}
	check := func(t *testing.T, api string, err error, want string) {
		t.Helper()
		if err == nil || err.Error() != want || !errors.Is(err, errBoundaryUnstabilizable) {
			t.Errorf("%s: %v, want %q", api, err, want)
		}
	}
	for _, c := range []struct {
		name  string
		limit *int
		v     int
	}{{"doubling", &extendedDoublings, 2}, {"spread", &extendedSpreadBits, 0}, {"stableLoop", &stableSquarings, 0}, {"smithStein", &steinDoublings, 1}} {
		t.Run(c.name, func(t *testing.T) {
			withIterationCap(t, c.limit, c.v)
			for _, ns := range []bool{false, true} {
				opts := &RiccatiOpts{E: E, NoScaling: ns}
				_, err := Dare(A, B, eyeDense(2), eyeDense(2), opts)
				check(t, "Dare", err, boundaryError)
				_, err = Dlqr(A, B, eyeDense(2), eyeDense(2), opts)
				check(t, "Dlqr", err, boundaryError)
			}
			_, err := Kalman(sys, eyeDense(3), Rn, nil, nil)
			check(t, "Kalman", err, "Kalman: "+boundaryError)
		})
	}
}

// bnd generator seed 82 (O2YDYA round-1 review): a stable uncontrollable
// mode at −(1 − 1e-9) beside fast modes, σ_min(E) 4.3e-14. Repeated
// squaring drove the fast entries to exponents near −2³⁰, and each
// big.Float sum then shifted its mantissa by that gap: 21.9 GB allocated,
// 4.8 GB resident. Base errBoundaryUnstabilizable; now Accuracy 1.
func TestDare_NearSingularENearUnitModeBoundedCost(t *testing.T) {
	skipSlowUnderRace(t)
	A := mat.NewDense(4, 4, []float64{-0.5665014768372381, 0, 0, 0, -1.5980210024765213, -0.06465621784083286, -0.0828969416783879, 0.9847150508953924, 0.7217324705211728, 0.53862200268466, -0.17110939638464817, 1.1214081883588474, -1.0400461037212216, -1.5146521616137787, 0.5536448758199233, 0.8495075925372486})
	B := mat.NewDense(4, 3, []float64{0, 0, 0, -0.32533712585683494, -0.3850771055627178, 0.819380445871456, -0.9629967805389414, 0.5603125968042595, 1.6017953055098246, -0.2470353648529452, -0.5959107444303582, -0.840210045986464})
	E := mat.NewDense(4, 4, []float64{0.5665014774037396, 0, 0, 0, 0, 0.749141452362936, 0.9986910131853461, -0.40126864570400955, 0, 0, 1.1127408060359611, 0.27638968643835526, 0, 0, 0, 4.331142713859568e-14})
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	t0 := time.Now()
	for _, ns := range []bool{false, true} {
		opts := &RiccatiOpts{E: E, NoScaling: ns}
		if _, err := Dare(A, B, eyeDense(4), eyeDense(3), opts); err != nil {
			t.Errorf("Dare NoScaling %v: %v", ns, err)
		}
		if _, err := Dlqr(A, B, eyeDense(4), eyeDense(3), opts); err != nil {
			t.Errorf("Dlqr NoScaling %v: %v", ns, err)
		}
	}
	elapsed := time.Since(t0)
	runtime.ReadMemStats(&after)
	if elapsed > 20*time.Second {
		t.Errorf("4 calls took %v", elapsed)
	}
	if raceEnabled {
		return
	}
	if alloc := after.TotalAlloc - before.TotalAlloc; alloc > 1<<28 {
		t.Errorf("4 calls allocated %d MB", alloc>>20)
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
	}{{"stableLoop", &stableSquarings, 0}, {"smithStein", &steinDoublings, 1}} {
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
