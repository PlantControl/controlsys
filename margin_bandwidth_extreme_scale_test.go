package controlsys

import (
	"math"
	"testing"
)

// TestBandwidthExtremeScale covers continuous upper-triangular models whose
// first crossing sits at a pole of 1e-89 to 1e-240 rad/s, found by the scan
// from highFrequency in endCrossings: the bisection's geometric mean √(a·b)
// underflowed to 0 for 2111 and 2123 (RLCKVA, F9extremeScale census rows
// 2101, 2111, 2123, 2153, 2159). Expected values are a 320-bit big.Float
// scan from 1e-300 at 30 points per decade, bisected geometrically.
func TestBandwidthExtremeScale(t *testing.T) {
	cases := []struct {
		name string
		a    []float64
		b, c []float64
		d    float64
		want float64
	}{
		{"pole1e-89", []float64{-9.9999999999999969e-90, 1.3642433585072668e+87, 0, -1.0000000000000001e+99},
			[]float64{-0.53210845324314082, -0.11432951735421581}, []float64{0.21779249303263359, 0.35095330989716711},
			0.42362059497078897, 9.9762834511098329e-90},
		{"pole1e-176", []float64{-1e+43, -3.5936211455522251e+60, 0, -9.9999999999999965e-177},
			[]float64{1.6728038245134282, -0.38185826232608289}, []float64{0.59130487286604616, -0.9424725571225423},
			1.0432149909675568, 9.9762834511098331e-177},
		{"pole1e-240", []float64{-1.0000000000000001e+37, -1.1294687670937506e+85, 2.6095767011828084e+52, 0, -9.9999999999999957e-241, 7.8439187412915869e-52, 0, 0, -1e+18},
			[]float64{-0.67480888085189061, 1.6423711153465748, -0.32036928331799636}, []float64{0.69515726364241548, 0.88686480622568964, 1.5186779101733521},
			1.6883706370815272, 9.9762834511098351e-241},
		{"pole1e-156", []float64{-9.9999999999999999e+51, -6.5422225499989852e+179, -2.3357749623052804e+184, 0, -9.9999999999999998e-20, 2.7060499176688235e-62, 0, 0, -9.9999999999999978e-157},
			[]float64{-0.001895437674829914, -0.20972287088326069, -0.68308799019238853}, []float64{1.3592983885309462, 0.6216097497307882, 1.7560421609260812},
			0.92995573354421601, 9.9762834511098344e-157},
		{"pole1e-104", []float64{-1.0000000000000001e+82, -0.0011804374457915255, 0, -9.9999999999999977e-105},
			[]float64{-2.0864254058645102, 0.35338792784037132}, []float64{-1.4079501092545077, -0.56281911650895611},
			0.45882264010366891, 9.9762834511098338e-105},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			n := len(tc.b)
			sys, err := NewFromSlices(n, 1, 1, tc.a, tc.b, tc.c, []float64{tc.d}, 0)
			if err != nil {
				t.Fatal(err)
			}
			bw, err := Bandwidth(sys, -3)
			if err != nil {
				t.Fatalf("Bandwidth: %v", err)
			}
			if rel := math.Abs(bw-tc.want) / tc.want; !(rel <= 1e-9) {
				t.Fatalf("Bandwidth = %.17g, want %.17g (rel %.3g)", bw, tc.want, rel)
			}
		})
	}
}
