package controlsys

import (
	"errors"
	"math"
	"testing"
)

type scanFloorCase struct {
	name string
	dt   float64
	a    []float64
	b, c []float64
	d    float64
	want float64
}

// scanFloorBandwidthCases cross 3 dB below DC under endCrossings' 1e-300
// scan floor (CWMCXC, found with RLCKVA). Expected values are a 320-bit
// big.Float scan from 1e-320 at 30 points per decade, bisected
// geometrically; want 0 marks ErrAccuracyUnverified: a subnormal crossing,
// or one inside the dc rounding disk 2.2e-16 of the discrete rows' poles at
// 1 ± p (A2OX65), where only the stored data's double-double response,
// with no certificate, supported CWMCXC's values; the oracle crossings are
// 9.9762834511098376e-303, 9.976283451109837e-306 and 9.976283451109836e-308.
var scanFloorBandwidthCases = []scanFloorCase{
	{"continuous1e-302", 0, []float64{-1e-294, 9.9999999999999996e-303, 0, -9.9999999999999996e-303},
		[]float64{1.6699999999999999, -0.38}, []float64{5.8999999999999998e-301, -9.3999999999999993e-301}, 1.04, 9.9842592800998249e-303},
	{"continuous1e-305", 0, []float64{-1e-297, 1e-305, 0, -1e-305},
		[]float64{1.6699999999999999, -0.38}, []float64{5.8999999999999998e-301, -9.3999999999999993e-301}, 1.04, 9.9762834595622381e-306},
	{"continuous1e-307", 0, []float64{-9.9999999999999986e-300, 9.9999999999999991e-308, 0, -9.9999999999999991e-308},
		[]float64{1.6699999999999999, -0.38}, []float64{5.8999999999999998e-301, -9.3999999999999993e-301}, 1.04, 9.9762834511108518e-308},
	{"continuous3e-308", 0, []float64{-3.0000000000000002e-300, 3.0000000000000002e-308, 0, -3.0000000000000002e-308},
		[]float64{1.6699999999999999, -0.38}, []float64{5.8999999999999998e-301, -9.3999999999999993e-301}, 1.04, 2.9928850353329928e-308},
	{"continuousSubnormal", 0, []float64{-9.9999999999999698e-303, 9.9999999999999694e-311, 0, -9.9999999999999694e-311},
		[]float64{1.6699999999999999, -0.38}, []float64{5.8999999999999998e-301, -9.3999999999999993e-301}, 1.04, 0},
	{"discrete1e-302", 1, []float64{1, 1.9999999999999999e-302, 4.9999999999999998e-303, 1},
		[]float64{1, 0.5}, []float64{6.9999999999941985e-13, -1.9999999999771683e-13}, 0.01, 0},
	{"discrete1e-305", 1, []float64{1, 2e-305, 5e-306, 1},
		[]float64{1, 0.5}, []float64{6.9999999844311303e-16, -2.0000000167259936e-16}, 0.01, 0},
	{"discrete1e-307", 1, []float64{1, 1.9999999999999998e-307, 4.9999999999999995e-308, 1},
		[]float64{1, 0.5}, []float64{7.0000011207821165e-18, -2.0000024376476584e-18}, 0.01, 0},
}

func TestBandwidthBelowScanFloor(t *testing.T) {
	for _, tc := range scanFloorBandwidthCases {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewFromSlices(2, 1, 1, tc.a, tc.b, tc.c, []float64{tc.d}, tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			bw, err := Bandwidth(sys, -3)
			if tc.want == 0 {
				if !errors.Is(err, ErrAccuracyUnverified) {
					t.Fatalf("Bandwidth = %g, %v; want ErrAccuracyUnverified", bw, err)
				}
				return
			}
			if err != nil {
				t.Fatalf("Bandwidth: %v", err)
			}
			if rel := math.Abs(bw-tc.want) / tc.want; !(rel <= 1e-9) {
				t.Fatalf("Bandwidth = %.17g, want %.17g (rel %.3g)", bw, tc.want, rel)
			}
		})
	}
}

// TestAllMarginGainCrossingBelowScanFloor covers |L| = 1 crossings under the
// same floor, which AllMargin shares through endCrossings (CWMCXC).
func TestAllMarginGainCrossingBelowScanFloor(t *testing.T) {
	cases := []scanFloorCase{
		{"continuous1e-302", 0, []float64{-1e-294, 9.9999999999999996e-303, 0, -9.9999999999999996e-303},
			[]float64{1.6699999999999999, -0.38}, []float64{5.9000000000000001e-303, -4.9999999999999993e-302}, 0.5, 2.5192591765746032e-302},
		{"continuous1e-305", 0, []float64{-1e-297, 1e-305, 0, -1e-305},
			[]float64{1.6699999999999999, -0.38}, []float64{5.8999999999999995e-306, -4.9999999999999999e-305}, 0.5, 2.5192591765746038e-305},
		{"continuous1e-307", 0, []float64{-9.9999999999999986e-300, 9.9999999999999991e-308, 0, -9.9999999999999991e-308},
			[]float64{1.6699999999999999, -0.38}, []float64{5.8999999999999988e-308, -4.9999999999999993e-307}, 0.5, 2.5192591765746032e-307},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewFromSlices(2, 1, 1, tc.a, tc.b, tc.c, []float64{tc.d}, tc.dt)
			if err != nil {
				t.Fatal(err)
			}
			res, err := AllMargin(sys)
			if err != nil {
				t.Fatalf("AllMargin: %v", err)
			}
			if len(res.GainCrossFreqs) != 1 {
				t.Fatalf("GainCrossFreqs = %v, want [%.17g]", res.GainCrossFreqs, tc.want)
			}
			if w := res.GainCrossFreqs[0]; !(math.Abs(w-tc.want) <= 1e-9*tc.want) {
				t.Fatalf("GainCrossFreqs[0] = %.17g, want %.17g", w, tc.want)
			}
		})
	}
}
