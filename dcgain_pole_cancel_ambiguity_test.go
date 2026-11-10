package controlsys

import (
	"errors"
	"math"
	"testing"
)

// checkDCGainExactLimitEntries checks that every entry of g is the float
// model's exact limit: the same infinity, or within 1e-9 when finite.
func checkDCGainExactLimitEntries(t *testing.T, name string, n, m, p int, a, b, c, d, g []float64, dt float64) {
	t.Helper()
	x0 := 0.0
	if dt > 0 {
		x0 = 1
	}
	want := dcGainRationalLimit(n, m, p, a, b, c, d, x0)
	for k, v := range g {
		w := want[k]
		if math.IsInf(w, 0) || w == 0 {
			if v != w {
				t.Errorf("%s: entry %d = %v, exact limit %v", name, k, v, w)
			}
		} else if !(math.Abs(v-w) <= 1e-9*math.Abs(w)) {
			t.Errorf("%s: entry %d = %.17g, exact limit %.17g", name, k, v, w)
		}
	}
}

// INGGF2: where the exact limit of the float model and the deflated block
// expansion disagree on a pole at DC (Inf against finite, or the Inf sign),
// DCGain returns the exact limit with ErrAccuracyUnverified rather than
// either reading with a nil error. Over-cancelled discrete Series
// k(z−1)²/((z−a)(z−b))·1/(z−1) (intended 0), Parallel residue cancellation
// k/(z−1) − k(1−a)/((z−1)(z−a)) (intended k/(1−a)) and its zero-gain variant
// have an exact pole at z = 1 the given coefficients leave uncancelled; the
// fuzz rows 0xcc, 0xf7 and the w18#41 row have exact poles the deflation
// misreads or fails on.
func TestDCGainPoleCancelAmbiguityUnverified(t *testing.T) {
	type row struct {
		name       string
		n, m, p    int
		a, b, c, d []float64
		dt         float64
	}
	rows := []row{
		{"over-cancelled Series a=0.1 b=0.7", 3, 1, 1, []float64{1, 0, 0, 0.93000000000000005, 0, -0.069999999999999993, -1.2000000000000002, 1, 0.79999999999999993}, []float64{1, 0, 0}, []float64{1, 0, 1}, []float64{0}, 0.1},
		{"over-cancelled Series PCG row", 3, 1, 1, []float64{1, 0, 0, 43.146116550661567, 0, -0.47070270576561263, -51.179027658665589, 1, 1.3721585387766906}, []float64{1, 0, 0}, []float64{81.515845670571821, 0, 1}, []float64{0}, 0.1},
		{"over-cancelled Series, other order", 3, 1, 1, []float64{0, -0.45000000000000001, 0, 1, 1.3999999999999999, 0, 0, 1, 1}, []float64{0.055, -0.060000000000000026, 0.10000000000000001}, []float64{0, 0, 1}, []float64{0}, 0.1},
		{"Parallel residue cancellation", 3, 1, 1, []float64{1, 0, 0, 0, 0, -0.90000000000000002, 0, 1, 1.8999999999999999}, []float64{1, -0.099999999999999978, 0}, []float64{1, 0, 1}, []float64{0}, 0.1},
		{"Parallel zero-gain cancellation", 3, 1, 1, []float64{0, -0.90000000000000002, 0, 1, 1.8999999999999999, 0, 0, 0, 1}, []float64{-0.90000000000000002, 1, -1}, []float64{0, 1, 1}, []float64{0}, 0.1},
		{"integrators beside a tiny block", 6, 1, 3, []float64{
			-1e-13, 2e-13, -9899.499987499375, 19998.99997499875, -10099.499987499375, 0,
			-1e-13, 1e-13, -9899.499987499375, 19998.99997499875, -10099.499987499375, 0,
			0, 0, -9899.499987499375, 19998.99997499875, -10099.499987499375, 0,
			0, 0, -9999.499987499375, 10099.499987499375, -100, 0,
			0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0,
		}, []float64{-0.9999435261164459, -0.2572434700675131, 0.8623188722876839, 0.7185820465431654, -0.47787916084576904, -0.974246278043856}, []float64{
			704291.0931971872, 444099.1644246522, -991436.3462197612, 196942.88945958653, 864097.183359284, -755650.0977319883,
			996008.9570228793, -237538.8670471864, -842421.282141011, 782230.889887116, 336647.11098329315, -999899.8872783968,
			819288.2452914593, -807458.6576995418, -297202.3258757174, 999623.4801059181, -349133.35794434027, -773881.1359710619,
		}, []float64{331.1729782101074, -314.4549687182428, 279.47226068756777}, 0},
	}
	// Fuzz rows frozen as decoded on arm64: fzDecode uses math.Pow/cmplx.Exp,
	// whose amd64 results depend on runtime FMA dispatch.
	rows = append(rows,
		row{"fuzz 0xcc", 6, 3, 2, []float64{-472492.83825321327, 379320.5721724379, -165245.22874958202, 664010.4570270558, -1.702709059476704e+06, 24222.299594831577, -504494.9444288651, 405361.71528831, -176220.4279013586, 709191.4217311153, -1.8185657257637626e+06, 25870.446625764205, 189194.19733053423, -150332.48229049466, 67131.12284333729, -264959.7450528967, 679431.1046168241, -9665.411526662234, -0.00010023676431530897, 8.657360668988357e-05, -1.9478432279906744e-05, 0.0001638538610834163, -0.00040509104257800324, 5.752557217135122e-06, -1.0746109444725794e-05, 9.281319672167895e-06, -2.0882294687119006e-06, 1.7205553689891217e-05, -4.228784869679503e-05, 6.015760206690398e-07, 0, 0, 0, 0, 0, 0}, []float64{0.3635991279932228, -0.9876890479274097, 0.6336628088109636, -0.8003455990610225, -0.1134762573319723, 0.9149216920182602, -0.7917821484805551, 0.9269795161013984, -0.14418184701474382, 0.3767440045722096, 0.6094081267641936, -0.9920586423890088, 0.9933393083093572, -0.600947595973946, -0.38656720251717114, 0.15469019822681776, -0.9309136827397817, 0.7852460946729527}, []float64{-921578.9733711673, 665281.975433926, 491421.5300109033, -983024.8826031802, 144181.84701475085, 889799.9088612231, -454783.0165209421, 989803.6130434553, -185203.346160561, -870054.9939196184, 747762.7504239617, 386567.20251717116}, []float64{-22.17067205539593, -3.513219285071697, 15.698893561679771, 31.91424987894672, 15.044267297595244, -4.200874296610702}, 0},
		row{"fuzz 0xf7", 6, 3, 3, []float64{1, 0, 0.10959043172595173, 9.476079761276545e+127, -2.996599533505853e+128, 6.4794802277706936e+128, 0, 1, 0.10959043172595173, 9.476079761276545e+127, -2.996599533505853e+128, 6.4794802277706936e+128, 0, 0, 1.0346555374015185, 9.476079761276545e+127, -2.996599533505853e+128, 6.4794802277706936e+128, 0, 0, 0, 2.9965995335058526e+127, -9.476079761276545e+127, 2.0489915573781985e+128, 0, 0, 0, 0, 0.995012354809911, -3.146505341206915, 0, 0, 0, 0, 0, 1.390833510807106e-87}, []float64{-0.9667981925794609, 0.70866977429126, 0.25125984258225487, -0.01238866346289056, 0.8693974903498248, -0.8654352092411123, 0.9601702866503661, -0.24354415373579286, -0.7142656520271989, 0.5260775173811046, -0.9996930420352065, 0.4833047587530059, -0.6787200473200125, -0.29128928172134555, 0.9728325656974354, -0.8891911526253609, 0.8438539587324921, 0.037158384790824624}, []float64{-8.568887533689471e-07, -2.1079579943077972e-07, 9.931849187581927e-07, -4.3137684497061916e-07, -7.142656520272001e-07, 8.932061115093224e-07, -9.87479769908865e-07, 4.685166713003763e-07, 6.845466664428065e-07, -9.111302618846769e-07, -9.542885100095242e-08, 9.728325656974356e-07, -6.536436208636123e-07, 9.274784307440356e-07, 5.395542056265064e-08, -9.623648798313102e-07, 5.682896297679735e-07, 5.94920663309892e-07}, []float64{-0.4937398849544325, -0.424050015855204, -0.2874119732666346, 0.49827104851160875, 0.47511629597926475, 0.3769511271716523, -0.47386080106555595, -0.4985860780981892, -0.4445955763126809}, 1},
	)
	for _, tc := range rows {
		sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, tc.dt)
		if err != nil {
			t.Fatalf("%s: %v", tc.name, err)
		}
		g, err := sys.DCGain()
		if !errors.Is(err, ErrAccuracyUnverified) {
			t.Errorf("%s: DCGain error %v, want ErrAccuracyUnverified", tc.name, err)
		}
		if g == nil {
			t.Errorf("%s: no gain with %v", tc.name, err)
			continue
		}
		checkDCGainExactLimitEntries(t, tc.name, tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, g.RawMatrix().Data, tc.dt)
	}
}

// INGGF2: true integrators, where the exact limit and the deflated
// expansion agree or the exact route is not taken, keep their exact infinite
// gain with a nil error; so do the Series cancellations both routes read as
// finite, and the discrete over-cancelled Series both read as 0.
func TestDCGainPoleCancelRoutesAgreeVerified(t *testing.T) {
	checkDCGainAgainstRationalLimit(t, "tiny integrator", 2, 1, 1, []float64{0, 0, 1, -1.3258457598967608e-08}, []float64{7.29653485561532e-14, 1.0337733234010072e-13}, []float64{0, 1}, []float64{0}, 0)
	checkDCGainAgainstRationalLimit(t, "integrator beside stiff pole", 2, 1, 1, []float64{-1000, 10, 0, 0}, []float64{1, 1}, []float64{1, 1}, []float64{0}, 0)
	checkDCGainAgainstRationalLimit(t, "dense singular block", 3, 2, 2, []float64{0.38629023905802873, 3.851534645926291, -3.23782488498432, 0, 4, -3, 4, 1.6251019182011088, -4.625101918201109}, []float64{-0.5, 0, -1.75, 0, 0, 0}, []float64{-1, -2.176206339653932, 1, 0, -0.06848213510526557, 0}, []float64{5.737696031494993, 0, 0.75, 0}, 0.1)
	// fuzz 0xe2 frozen as decoded on arm64 (see above).
	checkDCGainAgainstRationalLimit(t, "fuzz 0xe2", 6, 2, 1, []float64{4526.2901245623725, 4665.395309237434, 3097.9945242905183, 19922.67557949887, 31330.35970950401, -41162.16126996112, -5707.193331573256, -5896.1859008290685, -3920.445128154872, -25152.018191917, -39554.013398665666, 51966.48533662049, 2073.659862438267, 2197.8132596776527, 1482.36412959357, 9267.502525106662, 14574.055941828843, -19147.5503785788, -2.1341524882764163e-05, -1.8432486263731137e-05, -4.147175440252878e-06, -6.138770077549167e-05, -0.00010693655668742842, 0.00013955575552906, -5.249529568299665e-06, -4.533972253167385e-06, -1.020110803615356e-06, -1.639171743345499e-05, -2.690110420264855e-05, 3.534295807348057e-05, 0, 0, 0, 0, 0, 0}, []float64{0.9502325919585295, -0.21079579943078058, 0.5543743361791615, -0.9982947757947531, -0.6536436208636123, -0.3232895668635029, -0.904072142017061, 0.8253356149096784, 0.169967142900241, 0.7648421872844884, 0.9950041652780257, -0.4161468365471428}, []float64{-438547.32757439034, -708669.7742912608, 896758.4163341465, 128844.49429552574, -980066.5778412418, 504846.10459985636}, []float64{-23.316770280000654, -25.496395227777086}, 0)

	tfd := func(num, den []float64) *System {
		res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: 0.1}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		return res.Sys
	}
	mul := func(a, b []float64) []float64 {
		o := make([]float64, len(a)+len(b)-1)
		for i, x := range a {
			for j, y := range b {
				o[i+j] += x * y
			}
		}
		return o
	}
	dz := []float64{1, -1}
	for _, q := range []struct{ a, b, k float64 }{{0.5, 0.5, 1}, {0.25, 0.75, 2}, {0.1, 0.1, 1}} {
		g2 := tfd(mul([]float64{q.k, -q.k}, dz), mul([]float64{1, -q.a}, []float64{1, -q.b}))
		sys, err := Series(tfd([]float64{1}, dz), g2)
		if err != nil {
			t.Fatal(err)
		}
		g, err := sys.DCGain()
		if err != nil {
			t.Fatalf("over-cancelled Series %v: %v", q, err)
		}
		if !(math.Abs(g.At(0, 0)) <= 1e-9) {
			t.Errorf("over-cancelled Series %v: DCGain = %v, want 0", q, g.At(0, 0))
		}
	}
}

// INGGF2, YRJCSJ: Bandwidth on a DC gain returned with ErrAccuracyUnverified
// is computed on that value and keeps the error. Series(7s/(s+1/3), 1/s,
// (s+1)/(s+1e-14)) has its integrator cancelled below rounding beside a pole
// 1e-14 from DC that deflation snaps onto it: the exact limit with the
// sub-rounding residue read as cancelled is the intended k/(a·d), the deflated
// expansion infinite. The bandwidth is that of the model with the
// integrator cancelled by hand. Where the exact DC gain is infinite the
// bandwidth stays undefined (ErrInvalidArgument, no value).
func TestBandwidthOnUnverifiedDCGain(t *testing.T) {
	tf := func(num, den []float64) *System {
		res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		return res.Sys
	}
	series := func(ss ...*System) *System {
		s := ss[0]
		for _, x := range ss[1:] {
			var err error
			if s, err = Series(s, x); err != nil {
				t.Fatal(err)
			}
		}
		return s
	}
	k, a, d := 7.0, 1.0/3, 1e-14
	sys := series(tf([]float64{k, 0}, []float64{1, a}), tf([]float64{1}, []float64{1, 0}), tf([]float64{1, 1}, []float64{1, d}))
	g, err := sys.DCGain()
	if !errors.Is(err, ErrAccuracyUnverified) || g == nil {
		t.Fatalf("DCGain = %v, %v; want a value with ErrAccuracyUnverified", g, err)
	}
	if w := k / a / d; !(math.Abs(g.At(0, 0)-w) <= 1e-12*w) {
		t.Fatalf("DCGain = %v, want %v", g.At(0, 0), w)
	}
	want, err := Bandwidth(series(tf([]float64{k}, []float64{1, a}), tf([]float64{1, 1}, []float64{1, d})), -3)
	if err != nil {
		t.Fatal(err)
	}
	bw, err := Bandwidth(sys, -3)
	if !errors.Is(err, ErrAccuracyUnverified) {
		t.Errorf("Bandwidth error %v, want ErrAccuracyUnverified", err)
	}
	if !(math.Abs(bw-want) <= 1e-4*want) {
		t.Errorf("Bandwidth = %v, want %v", bw, want)
	}

	over, err := NewFromSlices(3, 1, 1, []float64{1, 0, 0, 0.93000000000000005, 0, -0.069999999999999993, -1.2000000000000002, 1, 0.79999999999999993}, []float64{1, 0, 0}, []float64{1, 0, 1}, []float64{0}, 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if bw, err := Bandwidth(over, -3); !errors.Is(err, ErrInvalidArgument) || errors.Is(err, ErrAccuracyUnverified) {
		t.Errorf("Bandwidth on an infinite exact DC gain = %v, %v; want ErrInvalidArgument alone", bw, err)
	}
}
