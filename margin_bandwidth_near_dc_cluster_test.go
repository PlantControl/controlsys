package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"testing"
)

// bandwidthNearDCClusterOracle is the first ω where |H| falls 3 dB below
// |H(DC)| by bandwidthDipGainSq (320-bit): the first of 30 log-spaced ν per
// decade from 1e-22 past which |H|² is below the level, bisected in ν.
func bandwidthNearDCClusterOracle(sys *System) float64 {
	target := bandwidthDipGainSq(sys, 0)
	target.Mul(target, bandwidthDipF(math.Pow(10, -3.0/10)))
	below := func(nu float64) bool { return bandwidthDipGainSq(sys, nu).Cmp(target) < 0 }
	prev := 1e-22
	for k := 1; k <= 30*40; k++ {
		nu := 1e-22 * math.Pow(10, float64(k)/30)
		if !below(nu) {
			prev = nu
			continue
		}
		lo, hi := prev, nu
		for range 200 {
			mid := math.Sqrt(lo * hi)
			if mid <= lo || mid >= hi {
				break
			}
			if below(mid) {
				hi = mid
			} else {
				lo = mid
			}
		}
		if sys.IsDiscrete() {
			return 2 * math.Atan(hi) / sys.Dt
		}
		return hi
	}
	return math.Inf(1)
}

// Poles within 1e-14 of DC in a permuted-triangular A with entries up to
// 1e4 leave the DC gain at 1e41–1e54, which DCGain certifies, while
// dcResolvent declines (condition ~1e40) and both double-double solves fail
// or misread |L| near the crossing: Bandwidth returned +Inf for the
// discrete rows and 3.3054e-17 against 3.3053e-17 for the continuous one,
// with nil (HKBAQT; F4nearDCTriangle census rows 922, 997, 1034, 1100).
func TestBandwidthNearDCPoleCluster(t *testing.T) {
	tests := []struct {
		name    string
		dt      float64
		a, b, c []float64
		d       float64
	}{
		{
			name: "discrete, DC gain 5e41", dt: 0.1,
			a: []float64{
				0.99678011478222672, 0, 0, -45.26813158249071, 0, 0,
				-863.37686321127012, 0.9915128236358105, 0, -4.0573557598410215, 0, 0,
				4.3760011700891468, -3.1016874875302647, 0.99994025732310465, 95.753951131603856, 0, 0,
				0, 0, 0, 0.99982577555596297, 0, 0,
				0.20604719089207479, -100.27582092184012, -207.68636158268677, 0.0064118962345007533, 0.99999999999999412, 0,
				171.05380766616148, -6.2451970783917945, 1.6196080745780161, 0.0019958243165173438, 0.078523163105456817, 0.99999999956490015,
			},
			b: []float64{-0.98285876360357571, 1.8445890281015149, 1.3118659717602663, -2.3458730367002358, 0.23730761268098352, 1.0091524876032296},
			c: []float64{0.28067944901824204, -1.7897216331376322, -1.0517126267614572, -0.40419366806333912, -0.96970086153424084, -0.085249091414129827},
			d: 0.16310354737820248,
		},
		{
			name: "discrete, DC gain -1e41", dt: 0.1,
			a: []float64{
				0.99977291584452932, 0, 0, 0, 0, 0,
				-0.09475175099034161, 0.99999999999788181, -23.735517036195368, 0.26789560319275918, -17.42064078232152, -0.015319364584223595,
				0.0035890430272879003, 0, 0.99999999999168399, 2.0103439950239972, 0.080250956906869689, 0,
				1.1898451184143455, 0, 0, 0.99999995307097911, 26.57034807785201, 0,
				0.00092853394464933569, 0, 0, 0, 0.9999999341824799, 0,
				-0.48650793036898082, 0, 0.014307175033470638, -28.256524943422804, -445.38292471840816, 0.99913464430271759,
			},
			b: []float64{0.12180169964000953, 0.01213159093594901, 0.29782628389861698, -0.64256344265867249, 1.5068349818695808, -0.32242934927242617},
			c: []float64{-0.89273574741623873, 2.4828044928374866, -0.65570121498864797, -0.85395304138798267, -0.50091416299857805, -0.047794765699443742},
			d: -1.151300817655569,
		},
		{
			name: "discrete, DC gain -2e54", dt: 0.1,
			a: []float64{
				0.9999999999921706, 0.035307708632898248, 0, 0, 0, 0,
				0, 0.96465745496628807, 0, 0, 0, 0,
				0.023409111903610582, 196.60689285594916, 0.99999999999998634, -0.046660154489228327, 0.055999150545907012, 0,
				0.016573258857612825, -36.821628121052782, 0, 0.99986547840860451, -791.80317325634985, 0,
				5.5246357888199276, -216.1292882970759, 0, 0, 0.99999999999964606, 0,
				-0.097195747008071112, 0.010124090813180197, 1651.0026006398309, 518.29566791800107, -0.68049114024298807, 0.99999999961286623,
			},
			b: []float64{0.21217728969679805, -0.18540136015632222, -0.97276994117861171, 0.87311011354507184, 0.1923608178930484, 0.56397799181370711},
			c: []float64{0.29908130815137174, -1.6140705352692826, 1.1278723993446615, -0.13450312028548858, -0.085497822494112752, -0.36700454355421808},
			d: 1.3701637036678158,
		},
		{
			name: "continuous, poles below 1e-13", dt: 0,
			a: []float64{
				-13379.752920032653, 0, 0, 0, 0, 0,
				0.01025617870686453, -3.3137462815326466e-17, -0.070113640164768148, 0, -0.0012098900659767444, 0,
				-0.0011186871581358171, 0, -3.5835598342136957e-14, 0, 0, 0,
				-15.364026646698418, 25.994492224708964, -0.01237813531439722, -4.9048117399650772, -0.004175072641834681, 0,
				3.6210896894727811, 0, 0.18682790619380044, 0, -2.4513461467801236e-15, 0,
				-157.3256776861769, 50.053290113193924, 69.057841415816668, -0.0030328746390441935, -0.033785502588466493, -0.058909708571066192,
			},
			b: []float64{0.066473276729444808, -0.35308250970675337, 0.16061025590543565, 0.75845527688306935, 0.82352971400440622, 0.72647962163551716},
			c: []float64{-0.57078460347528037, 0.98543472260796261, 0.37171752995072205, 0.66588966516551662, -0.50077294280365303, 0.59592136691677589},
			d: -0.59404963013156442,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewFromSlices(6, 1, 1, tc.a, tc.b, tc.c, []float64{tc.d}, tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			want := bandwidthNearDCClusterOracle(sys)
			if math.IsInf(want, 0) {
				t.Fatal("oracle found no crossing")
			}
			got, err := Bandwidth(sys, -3)
			if err != nil {
				t.Fatalf("Bandwidth: %v", err)
			}
			if rel := math.Abs(got/want - 1); !(rel <= 1e-9) {
				t.Errorf("Bandwidth = %.17g, oracle %.17g (rel %.3g)", got, want, rel)
			}
		})
	}
}

// ddFallbackSystem balances as well as permutes: in an upper-triangular A
// with entries from 1e-157 to 2e184, ddSISOSolve overflows forming states
// near 1e156 times entries near 1e184, while the balanced realization
// resolves the response of the same model (HKBAQT; F9extremeScale census
// row 2153).
func TestBandwidthFallbackRealizationExtremeScale(t *testing.T) {
	sys, err := NewFromSlices(3, 1, 1,
		[]float64{
			-9.9999999999999999e+51, -6.5422225499989852e+179, -2.3357749623052804e+184,
			0, -9.9999999999999998e-20, 2.7060499176688235e-62,
			0, 0, -9.9999999999999978e-157,
		},
		[]float64{-0.001895437674829914, -0.20972287088326069, -0.68308799019238853},
		[]float64{1.3592983885309462, 0.6216097497307882, 1.7560421609260812},
		[]float64{0.92995573354421601}, 0)
	if err != nil {
		t.Fatal(err)
	}
	for _, w := range []float64{0, 1e-157, 1e-156} {
		pt := newTimeDomain(0).frequencyPoint(w)
		if _, ok := ddSISOResponse(sys, pt); ok {
			t.Fatalf("ω=%g: ddSISOSolve resolved the unbalanced model", w)
		}
		fb := ddFallbackSystem(sys)
		if fb == nil {
			t.Fatal("ddFallbackSystem: nil")
		}
		h, ok := ddSISOResponse(fb, pt)
		if !ok {
			t.Fatalf("ω=%g: fallback not resolved", w)
		}
		want, _ := new(big.Float).Sqrt(bandwidthDipGainSq(sys, w)).Float64()
		if got := cmplx.Abs(h); !(math.Abs(got/want-1) <= 1e-12) {
			t.Errorf("ω=%g: |H| = %.17g, oracle %.17g", w, got, want)
		}
	}
}
