package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

var trimGateLowW = []float64{0.01, 0.3, 1, 3, 10, 30}

// trimGateErr is the worst relative error of TF and ZPK of sys against the
// exact rational TF of its float data at jω (e^{jω/100} in discrete time),
// ω ∈ trimGateLowW.
func trimGateErr(t *testing.T, sys *System) (tfErr, zpkErr float64, num []float64) {
	t.Helper()
	n, _, _ := sys.Dims()
	raw := func(m *mat.Dense) []float64 {
		r, c := m.Dims()
		out := make([]float64, 0, r*c)
		for i := range r {
			for j := range c {
				out = append(out, m.At(i, j))
			}
		}
		return out
	}
	en, ed := exactTFCoeffs(raw(sys.A), raw(sys.B), raw(sys.C), n)
	res, err := sys.TransferFunction(nil)
	if err != nil {
		t.Fatal(err)
	}
	zpk, err := res.TF.ZPK()
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range trimGateLowW {
		s := complex(0, w)
		if !sys.IsContinuous() {
			s = cmplx.Rect(1, w/100)
		}
		ex := Poly(en).Eval(s) / Poly(ed).Eval(s)
		g, _ := res.TF.Eval(s)
		z, _ := zpk.Eval(s)
		tfErr = max(tfErr, cmplx.Abs(g[0][0]-ex)/cmplx.Abs(ex))
		zpkErr = max(zpkErr, cmplx.Abs(z[0][0]-ex)/cmplx.Abs(ex))
	}
	return tfErr, zpkErr, res.TF.Num[0][0]
}

// A fast mode in a dense basis makes every Markov parameter C·A^k·B look
// negligible, so the trim offers dropping all but the constant; a check point
// where the full numerator is noise-dominated must not let that pass where
// the full numerator is accurate.
func TestTransferFunctionTrimGateRelativeDegreeFive(t *testing.T) {
	a := []float64{
		-300734892.70644146, -169603079.72644186, -102398657.39304641, 35043396.78175046, 24847604.160978448, 25638473.582307182,
		469891391.63049394, 265000933.37529042, 159995560.69834027, -54754506.75998623, -38823813.003935285, -40059528.622298926,
		128266829.57902764, 72337630.51247229, 43674183.827863105, -14946403.463409953, -10597783.868360376, -10935097.803372044,
		1332180095.5530968, 751299076.371342, 453600351.5940503, -155233454.90061352, -110068648.58390085, -113572002.34454507,
		1900332076.7233894, 1071715257.1759673, 647053128.504293, -221437863.732548, -157011042.43979317, -162008515.4137372,
		-3568546701.0851016, -2012525070.4448168, -1215071478.5461414, 415828037.053484, 294843855.88538533, 304228384.0178798,
	}
	b := []float64{0.02054774531485147, -0.03210538206432974, -0.008763845491471531, -0.09102135434842197, -0.12984040216310627, 0.24382135342581673}
	c := []float64{5.331537793077032, 16.79308562255658, -1.788144065900889, -1.4599873790650553, 1.5258232006053158, 1.9651675032660865}
	sys, err := NewFromSlices(6, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	tfErr, zpkErr, num := trimGateErr(t, sys)
	if len(num) < 2 || math.Abs(num[len(num)-2]-4.878425372003843) > 1e-5*4.878425372003843 {
		t.Errorf("num %v lost the s coefficient 4.878", num)
	}
	if tfErr > 1e-3 || zpkErr > 1e-3 {
		t.Errorf("low-frequency rel err TF %.3g ZPK %.3g vs exact, num %v", tfErr, zpkErr, num)
	}
}

// The s⁴…s² coefficients of the same plant, −4.3e-9, −5.4e-8 and 3.4e-8 in
// exact arithmetic on the float data, are below the rounding of its 1e9-sized
// A (the response above the fast pole is undetermined by the coefficients),
// so as in MATLAB tf(ss) the numerator is 4.878s − 2.094 with the single zero
// 0.4293, not three spurious zeros near |s| ≈ 1e3.
func TestTransferFunctionTrimRelativeDegreeFiveDegree(t *testing.T) {
	a := []float64{
		-300734892.70644146, -169603079.72644186, -102398657.39304641, 35043396.78175046, 24847604.160978448, 25638473.582307182,
		469891391.63049394, 265000933.37529042, 159995560.69834027, -54754506.75998623, -38823813.003935285, -40059528.622298926,
		128266829.57902764, 72337630.51247229, 43674183.827863105, -14946403.463409953, -10597783.868360376, -10935097.803372044,
		1332180095.5530968, 751299076.371342, 453600351.5940503, -155233454.90061352, -110068648.58390085, -113572002.34454507,
		1900332076.7233894, 1071715257.1759673, 647053128.504293, -221437863.732548, -157011042.43979317, -162008515.4137372,
		-3568546701.0851016, -2012525070.4448168, -1215071478.5461414, 415828037.053484, 294843855.88538533, 304228384.0178798,
	}
	b := []float64{0.02054774531485147, -0.03210538206432974, -0.008763845491471531, -0.09102135434842197, -0.12984040216310627, 0.24382135342581673}
	c := []float64{5.331537793077032, 16.79308562255658, -1.788144065900889, -1.4599873790650553, 1.5258232006053158, 1.9651675032660865}
	sys, err := NewFromSlices(6, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	tfErr, zpkErr, num := trimGateErr(t, sys)
	want := []float64{4.878425372003843, -2.0943814259188778}
	if len(num) != 2 || math.Abs(num[0]-want[0]) > 1e-5*want[0] || math.Abs(num[1]-want[1]) > 1e-5*-want[1] {
		t.Fatalf("num %v, want %v", num, want)
	}
	if tfErr > 1e-4 || zpkErr > 1e-4 {
		t.Errorf("low-frequency rel err TF %.3g ZPK %.3g vs exact", tfErr, zpkErr)
	}
	zpk, err := sys.ZPKModel(nil)
	if err != nil {
		t.Fatal(err)
	}
	z := zpk.ZPK.Zeros[0][0]
	if len(z) != 1 || cmplx.Abs(z[0]-complex(-want[1]/want[0], 0)) > 1e-5 {
		t.Errorf("zeros %v, want [%.6g]", z, -want[1]/want[0])
	}
}

// A constant-numerator 6-pole plant in a dense basis of A ≈ 1e7: the
// untrimmed leading coefficients, ≈ 3e-8, are noise, but they offset the
// rounding of the den, so at ω ≈ 10, where the given coefficients still
// determine the response to 2e-4, dropping any of them is 25× less accurate.
// The trim allowance for unresolved points must not reach resolved ones.
func TestTransferFunctionTrimKeepsAccuracyAtResolvedPoints(t *testing.T) {
	a := []float64{
		-2.410095930032569e+06, -3.1283743840901004e+06, -448060.38618058176, 1.3285480995570647e+06, -905449.7888760152, 478379.442494707,
		-1.5876786335107982e+06, -2.0608525582550706e+06, -295166.389696553, 875197.5435142154, -596475.3496942524, 315137.9508406461,
		-2.3063505986014386e+06, -2.993709288875023e+06, -428773.60553179646, 1.2713597798289459e+06, -866474.3538562271, 457787.5743546723,
		-8.1132952202379955e+06, -1.053129157189491e+07, -1.5083419362184438e+06, 4.472397661647691e+06, -3.0480881119112903e+06, 1.6104070144188069e+06,
		-1.142254652987063e+06, -1.4826799602193784e+06, -212356.18360042459, 629659.7439540701, -429135.07292833924, 226726.7783357066,
		-4.314418944442435e+06, -5.600241807330226e+06, -802096.5451621374, 2.3782941965085655e+06, -1.6208843489636483e+06, 856368.8262628533,
	}
	b := []float64{0.6504864429199331, 0.4285153832784697, 0.6224836118674104, 2.1897805224270908, 0.3082929067327497, 1.1644650259101244}
	c := []float64{-16.808674611375846, -21.41505434902472, -0.574388681570477, 8.475087985415275, -11.768443781197988, 4.7554780461989195}
	sys, err := NewFromSlices(6, 1, 1, a, b, c, []float64{0}, 0)
	if err != nil {
		t.Fatal(err)
	}
	tfErr, zpkErr, num := trimGateErr(t, sys)
	if tfErr > 1e-3 || zpkErr > 1e-3 {
		t.Errorf("low-frequency rel err TF %.3g ZPK %.3g vs exact, num %v", tfErr, zpkErr, num)
	}
}

// trimGatePlant is a random stable 5-pole plant with 0–2 zeros in series
// with a fast actuator F/(s+F), F ∈ [1e3, 1e5], in a random dense basis of
// condition at most 1e3.
func trimGatePlant(rng *rand.Rand) *System {
	const np, n = 5, 6
	den := Poly{1}
	for deg := 0; deg < np; {
		if np-deg >= 2 && rng.IntN(2) == 0 {
			re, im := 0.2+20*rng.Float64(), 0.2+10*rng.Float64()
			den = den.Mul(Poly{1, 2 * re, re*re + im*im})
			deg += 2
		} else {
			den = den.Mul(Poly{1, 0.2 + 30*rng.Float64()})
			deg++
		}
	}
	num := Poly{0.1 + 10*rng.Float64()}
	for range rng.IntN(3) {
		num = num.Mul(Poly{1, (1 - 2*rng.Float64()) * 30})
	}
	F := math.Pow(10, 3+2*rng.Float64())
	a, b, c := mat.NewDense(n, n, nil), mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range np - 1 {
		a.Set(i, i+1, 1)
	}
	for j := range np {
		a.Set(np-1, j, -den[np-j])
	}
	for j := range num {
		c.Set(0, j, num[len(num)-1-j])
	}
	a.Set(np, np, -F)
	a.Set(np-1, np, 1)
	b.Set(np, 0, F)
	tm := mat.NewDense(n, n, nil)
	for {
		for i := range n {
			for j := range n {
				tm.Set(i, j, rng.NormFloat64())
			}
		}
		if mat.Cond(tm, 2) <= 1e2 {
			break
		}
	}
	var ti, ta, at, tb, ct mat.Dense
	if err := ti.Inverse(tm); err != nil {
		panic(err)
	}
	ta.Mul(tm, a)
	at.Mul(&ta, &ti)
	tb.Mul(tm, b)
	ct.Mul(c, &ti)
	sys, err := New(&at, &tb, &ct, mat.NewDense(1, 1, nil), 0)
	if err != nil {
		panic(err)
	}
	return sys
}

func TestTransferFunctionTrimGateFastModeSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(17, 4))
	for trial := range 300 {
		sys := trimGatePlant(rng)
		tfErr, zpkErr, num := trimGateErr(t, sys)
		if tfErr > 0.1 || zpkErr > 0.1 {
			t.Errorf("trial %d: low-frequency rel err TF %.3g ZPK %.3g vs exact, num %v", trial, tfErr, zpkErr, num)
		}
	}
}
