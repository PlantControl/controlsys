package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Square models whose 2×2 response has rank 1 to the data's rounding:
// the staircase left Rank 1 with no zeros, while the rank-1 model within
// terms has one (sweep rows 0:6358 and 0:1302, GVD7AS).
func TestZerosRankDeficientSquareWithinTerms(t *testing.T) {
	cases := []struct {
		name    string
		a, b, c []float64
		n       int
		zero    float64
	}{
		{
			name: "large entries",
			n:    3,
			a: []float64{25116.36916343106, 80378.868582635376, 10427.918702622777,
				5170.1020009533604, 16545.619268398077, 2146.5422981486472,
				-100355.70120286789, -321157.64376667491, -41665.761828467344},
			b: []float64{-11219489.098965336, 5320416.4047933295,
				-2309483.8374922029, 1095184.9578187761,
				44828587.24522499, -21258254.175334886},
			c: []float64{-122095032.16728356, -39822606.582023717, -32608958.669835452,
				-79262.145640359915, -25852.200426619089, -21169.215367646677},
			zero: -7.644060282037268,
		},
		{
			name: "slow zero",
			n:    4,
			a: []float64{-729.79737760254397, 1897.8172906080549, 594.46633239037033, 2469.9477639503116,
				-341.80075068384031, 888.94604268086732, 278.47560751818293, 1156.8974397071825,
				-327.52056065651158, 851.45683579756212, 266.69180518260367, 1108.2082223310281,
				127.15698131110074, -330.71930500623148, -103.59880166835046, -430.44402421393659},
			b: []float64{1.0090049569581132, 1668.3168377489435,
				0.4726112740575521, 781.4286151747084,
				0.45271141192586734, 748.52563007614219,
				-0.17583867918994714, -290.73655901999854},
			c: []float64{228.06274255019278, -425.88593663621106, -169.38747916020648, -272.09998653210448,
				-4.3678261068591242, 8.1565085633151835, 3.2440855765302223, 5.211221313754816},
			zero: -0.0336598290365332,
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := New(mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, 2, tc.b), mat.NewDense(2, tc.n, tc.c), mat.NewDense(2, 2, nil), 0)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.ZerosDetail()
			if err != nil {
				t.Fatalf("ZerosDetail: %v", err)
			}
			if res.Rank != 1 || len(res.Zeros) != 1 {
				t.Fatalf("got Rank %d, zeros %v; want Rank 1 and one zero", res.Rank, res.Zeros)
			}
			if d := cmplx.Abs(res.Zeros[0]-complex(tc.zero, 0)) / math.Abs(tc.zero); d > 1e-3 {
				t.Errorf("zero %v, want near %g (rel %.2g)", res.Zeros[0], tc.zero, d)
			}
		})
	}
}

// A SISO model whose staircase zeros the within-terms check refutes: the
// zeros of the exact determinant of its system pencil replace them (sweep
// row 0:192, GVD7AS).
func TestZerosExactPencilReplacesRefutedZeros(t *testing.T) {
	a := []float64{-12484.232208234273, 5006.5907926646769, -4025.4217724202058, 14742.425428606221, 4596.0694923280671, 27527.887898283669, -1606.7215452556643,
		9516.3241986500725, -3815.7408811384457, 3068.1826925791752, -11238.389775398931, -3503.2000896483119, -20983.33773007337, 1224.4866433832706,
		4108.6002848815042, -1647.5134311704044, 1323.9271773827297, -4851.2707032325898, -1512.7151742167114, -9058.5681751653083, 528.14344126078731,
		12239.533054312118, -4907.3355633345545, 3946.1751528997297, -14453.527204729475, -4505.4715913720875, -26988.584714349701, 1574.8714551435146,
		-15443.346877241835, 6192.2172453854027, -4979.0518849280552, 18236.808421510254, 5685.8734670123313, 34053.678434066824, -1987.55368568171,
		-10286.917351260894, 4124.8999716050103, -3316.6080352278, 12146.908567656821, 3786.3408285639139, 22682.496308364, -1323.501769545494,
		7758.87139224771, -3111.243477615652, 2502.1928563971187, -9162.6794632114888, -2856.7194732978805, -17109.351304664662, 997.69428398373998}
	b := []float64{0.43660584500666627, -0.33280132902459747, -0.14367496280457337, -0.42805182752647647, 0.54009242350801745, 0.35974130133943921, -0.27135607547955842}
	c := []float64{0.0058735564537214487, 0.00047829367460046053, 0.0066780036251711423, 0.0026280410382666895, -0.00090786867855228667, 0.0026550742744932332, 0.0028953082949946187}
	sys, err := New(mat.NewDense(7, 7, a), mat.NewDense(7, 1, b), mat.NewDense(1, 7, c), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	want := []complex128{3.133463065543842, complex(10.26292203109801, 4.4396933133926275), complex(10.26292203109801, -4.4396933133926275), 22.79735415984172}
	res, err := sys.ZerosDetail()
	if err != nil {
		t.Fatalf("ZerosDetail: %v", err)
	}
	if res.Rank != 1 || len(res.Zeros) != len(want) {
		t.Fatalf("got Rank %d, zeros %v; want Rank 1 and %d zeros", res.Rank, res.Zeros, len(want))
	}
	for _, w := range want {
		best := math.Inf(1)
		for _, z := range res.Zeros {
			best = math.Min(best, cmplx.Abs(z-w)/cmplx.Abs(w))
		}
		if best > 1e-6 {
			t.Errorf("no zero near %v in %v", w, res.Zeros)
		}
	}
}

// A slow pole unresolved beside a 7e15 one: ZPK's dropped radius covers
// every slow probe, so the row is short of its slow count until those
// probes are judged in double-double and exactly (zz row nse-rot/133,
// GVD7AS).
func TestZerosDroppedProbesJudgedWhenShort(t *testing.T) {
	sys, err := New(
		mat.NewDense(2, 2, []float64{-6017014477400915, -2499994902374844.5, -2499994902374847, -1038716881166608.4}),
		mat.NewDense(2, 1, []float64{2555296713943104, 1061694098112320.9}),
		mat.NewDense(1, 2, []float64{0.21419672664507641, -0.32119311296812231}),
		mat.NewDense(1, 1, []float64{2.9705284616921785}), 0)
	if err != nil {
		t.Fatal(err)
	}
	res, err := sys.ZerosDetail()
	if err != nil {
		t.Fatalf("ZerosDetail: %v", err)
	}
	if res.Rank != 1 || len(res.Zeros) != 2 {
		t.Fatalf("got %+v, want Rank 1 with two zeros", res)
	}
}

// Textbook models must certify (nil) with their zeros: zeros at the
// origin, integrators, biproper models and discrete zeros at 0 and ±1.
func TestZerosTextbookCertified(t *testing.T) {
	type tc struct {
		name         string
		zeros, poles []complex128
		dt           float64
	}
	cases := []tc{
		{"origin zero", []complex128{0}, []complex128{-1, -2}, 0},
		{"double origin zero", []complex128{0, 0}, []complex128{-1, -2, -3}, 0},
		{"integrator", nil, []complex128{0}, 0},
		{"double integrator", nil, []complex128{0, 0}, 0},
		{"integrator with zero", []complex128{-1}, []complex128{0, -2}, 0},
		{"biproper", []complex128{-3}, []complex128{-1}, 0},
		{"biproper second order", []complex128{-2, -3}, []complex128{-1, -4}, 0},
		{"tiny zero beside unit zero", []complex128{-1e-9, -1}, []complex128{-2, -3, -4}, 0},
		{"discrete zero at 0", []complex128{0}, []complex128{0.5}, 0.1},
		{"discrete zero at 1", []complex128{1}, []complex128{0.5, 0.2}, 0.1},
		{"discrete zero at -1", []complex128{-1}, []complex128{0.5, 0.3}, 0.1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			z, err := NewZPK(c.zeros, c.poles, 2, c.dt)
			if err != nil {
				t.Fatal(err)
			}
			ss, err := z.StateSpace()
			if err != nil {
				t.Fatal(err)
			}
			got, err := ss.Sys.Zeros()
			if err != nil {
				t.Fatalf("Zeros: %v", err)
			}
			if len(got) != len(c.zeros) {
				t.Fatalf("zeros %v, want %v", got, c.zeros)
			}
			for _, w := range c.zeros {
				best := math.Inf(1)
				for _, g := range got {
					best = math.Min(best, cmplx.Abs(g-w))
				}
				if best > 1e-6*math.Max(1, cmplx.Abs(w)) {
					t.Errorf("no zero near %v in %v", w, got)
				}
			}
		})
	}
	t.Run("mimo diagonal with origin zero", func(t *testing.T) {
		A := mat.NewDense(2, 2, []float64{-1, 0, 0, -2})
		B := mat.NewDense(2, 2, []float64{1, 0, 0, 1})
		C := mat.NewDense(2, 2, []float64{1, 0, 0, -2})
		D := mat.NewDense(2, 2, []float64{0, 0, 0, 1})
		sys, err := New(A, B, C, D, 0)
		if err != nil {
			t.Fatal(err)
		}
		res, err := sys.ZerosDetail()
		if err != nil {
			t.Fatalf("ZerosDetail: %v", err)
		}
		if res.Rank != 2 || len(res.Zeros) != 1 || cmplx.Abs(res.Zeros[0]) > 1e-9 {
			t.Fatalf("got %+v, want Rank 2 and a zero at 0", res)
		}
	})
}

// matchedClusteredZerosModel is a matched discretization whose five zeros
// cluster at z = −1 (zz row matched/3, GVD7AS): neither the computed
// zeros nor those of the exact pencil certify within terms.
func matchedClusteredZerosModel(t *testing.T) *System {
	t.Helper()
	sys, err := New(
		mat.NewDense(6, 6, []float64{0.99987816825728837, 1, 0, 0, 0, 0,
			0, 0.99820636344796831, 33601.818837620704, 3.9989921406257212, 5147.3817946231757, 3.9996316118939212,
			0, 0, 0.99949607031286059, 0.00011898119385620552, 0, 0,
			0, 0, -0.00011898119385620552, 0.99949607031286059, 5147.3817946231757, 3.9996316118939212,
			0, 0, 0, 0, 0.99981580594696051, 0.00077695084872860963,
			0, 0, 0, 0, -0.00077695084872860963, 0.99981580594696051}),
		mat.NewDense(6, 1, []float64{0, 1, 0, 1, 0, 1}),
		mat.NewDense(1, 6, []float64{1.0621984559258416e-19, 5.3113158230606155e-20, 0, 0, 0, 0}),
		mat.NewDense(1, 1, nil), 0.001)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func TestZerosUnverifiedReturnsValue(t *testing.T) {
	sys := matchedClusteredZerosModel(t)
	res, err := sys.ZerosDetail()
	if !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("ZerosDetail error %v, want ErrAccuracyUnverified", err)
	}
	if res == nil || res.Rank != 1 || len(res.Zeros) != 5 {
		t.Fatalf("got %+v, want Rank 1 with five zeros", res)
	}
	z, err := sys.Zeros()
	if !errors.Is(err, ErrAccuracyUnverified) || len(z) != 5 {
		t.Fatalf("Zeros = %v, %v; want five zeros with ErrAccuracyUnverified", z, err)
	}
}

func TestPzmapRootLocusPropagateUnverifiedZeros(t *testing.T) {
	sys := matchedClusteredZerosModel(t)
	pz, err := Pzmap(sys)
	if !errors.Is(err, ErrAccuracyUnverified) || pz == nil || len(pz.Zeros) != 5 || len(pz.Poles) != 6 {
		t.Fatalf("Pzmap = %+v, %v; want poles, zeros and ErrAccuracyUnverified", pz, err)
	}
	rl, err := RootLocus(sys, []float64{0, 1, 10})
	if !errors.Is(err, ErrAccuracyUnverified) || rl == nil || len(rl.Branches) == 0 {
		t.Fatalf("RootLocus = %v, %v; want a result with ErrAccuracyUnverified", rl, err)
	}
}

// Models whose staircase reports Rank 0 while entries of G refute G ≡ 0 far
// beyond their terms (sweep row 0:270 and zz row fz/16422, GVD7AS round
// 4): Rank 0 is judged entrywise, so neither returns Rank 0 with nil.
func TestZerosRankZeroJudgedEntrywise(t *testing.T) {
	cases := []struct {
		name       string
		n, m       int
		a, b, c, d []float64
		dt         float64
	}{
		{
			name: "two outputs", n: 3, m: 2,
			a: []float64{324.84706121542285, -570.83234711780324, 444.1693810668728,
				105.29536365241503, -185.34811114861034, 143.58699949194386,
				-102.25723168335581, 179.28636711734288, -140.30466080497339},
			b: []float64{0.049628178864372333, 2972.7110100230016, 3.7768908399750596, 226234.47486019056, 4.8175893197624404, 288571.9593250716},
			c: []float64{-54557816757252424, 94906525371884912, -73872287723338528, 75427297770188400, -1.3121021285382562e+17, 1.0212994900190171e+17},
			d: make([]float64, 4),
		},
		{
			name: "three outputs, large A", n: 6, m: 3, dt: 0.01116318822923396,
			a: []float64{5.1928796145028383e+125, -9.2140329813774213e+125, 6.4029409955244212e+125, -3.8536908068614132e+124, -6.4721380361323185e+125, 2.1517632101801381e+126,
				-1.027931465283652e+126, 1.8239195064848373e+126, -1.26746333600193e+126, 7.6283879695263192e+124, 1.2811609027906373e+126, -4.2594191927857477e+126,
				6.3533596219601202e+125, -1.1273141194299899e+126, 7.8338397580304141e+125, -4.7148952768812226e+124, -7.9185006237530731e+125, 2.6326290055708558e+126,
				-9.3259815796778249e+124, 1.6547639891146723e+125, -1.1499151571558565e+125, 6.9209094272454316e+123, 1.1623423723810116e+125, -3.864388461061911e+125,
				-7.4731113292354579e+125, 1.3259982778875401e+126, -9.2145195818603273e+125, 5.5458748451813968e+124, 9.3141015530410617e+125, -3.0966182961221544e+126,
				2.2343826726899608e+126, -3.9645971344464863e+126, 2.755045654187706e+126, -1.6581589799824598e+125, -2.7848196293254893e+126, 9.2585670947015149e+126},
			b: []float64{0.99671175306480542, -0.43324114759325089, -0.55927154163515569, 0.34469527834745511, -0.98432533368626807, 0.64917034239348026,
				-0.81230058668406491, -0.093370599899914239, 0.90657615397131175, -0.77927418920555991, 0.93437228148386375, -0.16415422390086093,
				0.39539072110434026, 0.5932575814923482, -0.99439827918582879, 0.99080729869444351, -0.6169808652401465, -0.36784652583616878},
			c: []float64{1.0155784234312243e-06, 8.3686189046480903e-07, -1.5566758596171055e-06, 1.6965223834016256e-07, 1.4469822623162754e-06, -1.1052407760269633e-06,
				1.569988248385571e-06, -2.3549625905272348e-07, -1.4177212812113106e-06, 1.152165256924674e-06, 6.7275526747860017e-07, -1.5871547749812683e-06,
				1.3860080683811019e-06, -1.1970968382072083e-06, -6.1199023174574724e-07, 1.5927969520987632e-06, -4.1787904174529026e-07, -1.322605083284423e-06},
			d: []float64{1.3233384295877493, 1.3132454238928883, 1.0958198414132982, -1.2269275703303819, -1.3449973606000507, -1.250721641440697, 1.0592522553311163, 1.298626914865646, 1.3329769387589878},
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := New(mat.NewDense(tc.n, tc.n, tc.a), mat.NewDense(tc.n, tc.m, tc.b), mat.NewDense(tc.m, tc.n, tc.c), mat.NewDense(tc.m, tc.m, tc.d), tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			res, err := sys.ZerosDetail()
			if err == nil && res.Rank == 0 {
				t.Fatalf("Rank 0 with nil error; entries of G refute G ≡ 0")
			}
		})
	}
}

// s²/((s+1)(s+2)(s+3)) in realization forms whose rounding splits the
// double origin zero to ±5e-9j: the probe at the split zero is within
// √ε|ẑ| of it and is not judged, so the zeros certify (GVD7AS ruling 5).
func TestZerosSplitOriginPairCertified(t *testing.T) {
	den, num := []float64{1, 6, 11, 6}, []float64{1, 0, 0}
	for _, form := range []int{1, 7, 10} {
		a, b, c := rlpForm(den, num, form)
		sys := rlpSys(t, 3, a, b, c, 0, 0)
		z, err := sys.Zeros()
		if err != nil {
			t.Fatalf("form %d: Zeros: %v", form, err)
		}
		if len(z) != 2 || cmplx.Abs(z[0]) > 1e-7 || cmplx.Abs(z[1]) > 1e-7 {
			t.Fatalf("form %d: zeros %v, want a pair near 0", form, z)
		}
	}
}

// Wrong candidates the certificate must not verify: the split origin pair
// dropped or reduced to one zero, a lightly damped notch moved onto the
// axis, and any certified answer with one zero added in band (GVD7AS
// ruling 5).
func TestZerosCertifyRejectsWrongCandidates(t *testing.T) {
	den, num := []float64{1, 6, 11, 6}, []float64{1, 0, 0}
	a, b, c := rlpForm(den, num, 1)
	split := rlpSys(t, 3, a, b, c, 0, 0)
	zs, err := split.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	for name, cand := range map[string][]complex128{"pair dropped": nil, "one zero": zs[:1]} {
		if v := zerosCertify(split, cand, zerosModeCand).verdict; v == zerosCertVerified {
			t.Errorf("%s: verified", name)
		}
	}
	zeta := 1e-7
	nz, err := NewZPK([]complex128{complex(-zeta, math.Sqrt(1-zeta*zeta)), complex(-zeta, -math.Sqrt(1-zeta*zeta))}, []complex128{-1, -2, -3}, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	notch, err := nz.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	if v := zerosCertify(notch.Sys, []complex128{1i, -1i}, zerosModeCand).verdict; v == zerosCertVerified {
		t.Errorf("notch moved onto the axis: verified")
	}
	for _, sys := range []*System{split, notch.Sys} {
		z, err := sys.Zeros()
		if err != nil {
			t.Fatal(err)
		}
		r := zerosCertify(sys, z, zerosModeCand)
		if r.verdict != zerosCertVerified {
			t.Fatalf("zeros %v not verified", z)
		}
		lg, k := 0.0, 0
		for _, pt := range r.pts {
			if zerosCertDetermined(pt) {
				lg += math.Log(cmplx.Abs(pt.s))
				k++
			}
		}
		extra := complex(-math.Exp(lg/float64(k)), 0)
		if v := zerosCertify(sys, append(slices.Clone(z), extra), zerosModeCand).verdict; v == zerosCertVerified {
			t.Errorf("zeros %v plus in-band %v: verified", z, extra)
		}
	}
}

// G ≡ 0 exactly with no finite zero (A = diag(1, 2), B = e₁, C = e₂ᵀ):
// Rank 0 certifies only with the exact zero count, here none.
func TestZerosRankZeroExactCount(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{1, 0, 0, 2}), mat.NewDense(2, 1, []float64{1, 0}), mat.NewDense(1, 2, []float64{0, 1}), mat.NewDense(1, 1, nil), 0)
	if err != nil {
		t.Fatal(err)
	}
	if cnt, ok := zerosExactZeroCount(sys, newZerosBudget()); !ok || cnt != 0 {
		t.Fatalf("exact zero count %d, %v; want 0", cnt, ok)
	}
	if v := zerosVerdict(sys, nil, 0); v != zerosCertVerified {
		t.Errorf("Rank 0 without zeros: verdict %v, want verified", v)
	}
	if v := zerosVerdict(sys, []complex128{1}, 0); v == zerosCertVerified {
		t.Errorf("Rank 0 with a spurious zero: verified")
	}
}

// A certification that exceeds its work budget is capped and open.
func TestZerosCertifyWorkBudgetCaps(t *testing.T) {
	z, err := NewZPK([]complex128{-1}, []complex128{-2, -3}, 1, 0)
	if err != nil {
		t.Fatal(err)
	}
	ss, err := z.StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	r := zerosCertifyB(ss.Sys, []complex128{-1}, zerosModeCand, &zerosBudget{cap: 1})
	if !r.capped || r.verdict == zerosCertVerified {
		t.Fatalf("capped %v verdict %v, want capped and not verified", r.capped, r.verdict)
	}
}
