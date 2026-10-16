package controlsys

import (
	"math"
	"testing"
)

// TestHinfNormFinitePeakPastGridEnd checks one phase φ = ωh where sup |G∞|
// beats every sample on the Nyquist grid, yet the 1/ω correction lifts |G|
// above it at the first passage of φ through its maximiser, past the grid
// end: the peak is finite there, not sup |G∞| at ω = +Inf.
func TestHinfNormFinitePeakPastGridEnd(t *testing.T) {
	for _, tc := range []struct {
		name          string
		d22, d12, d21 []float64
		tau           []float64
		k             [2]float64
	}{
		{"commensurate",
			[]float64{-0.005378092169153308, -0.5666515205748464, 0.4187470077674682, -0.6227800395962164},
			[]float64{0.8344864558391585, -0.8626258581000215}, []float64{0.28231463864795225, -0.662828005575204},
			[]float64{0.1403112181094031, 0.07015560905470156}, [2]float64{2, 1}},
		{"equal-period-past-grid",
			[]float64{0.5812713491588295, 0.23820990617510865, -0.43566544333215473, 0.0917648133178673},
			[]float64{-0.7600175805671201, -0.7617770539213307}, []float64{0.9909186634420599, 0.5245113982552587},
			[]float64{0.03717229520194947, 0.03717229520194947}, [2]float64{1, 1}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, gain, gInf := feedthroughDifferenceLoop(t, tc.d22, tc.d12, tc.d21, tc.tau, tc.k)
			const k = 1 << 18
			lim := 0.0
			for i := range k {
				lim = max(lim, gInf(2*math.Pi*float64(i)/k))
			}
			h := tc.tau[0] / tc.k[0]
			want, wWant := oraclePeak(gain, 4*math.Pi/h, 1e-3)
			if !(want > lim*(1+1e-4)) {
				t.Fatalf("finite peak %.12g must exceed sup |G∞| %.12g", want, lim)
			}
			e, err := newDelayLFT(sys)
			if err != nil {
				t.Fatal(err)
			}
			_, res, err := e.stability()
			if err != nil {
				t.Fatal(err)
			}
			if end := res.w[len(res.w)-1]; !(wWant > end) {
				t.Fatalf("peak at %g must lie past the grid end %g", wWant, end)
			}
			if best, _, err := e.scan(res.w); err != nil || !(best < lim) {
				t.Fatalf("grid peak %.12g, %v; must stay below sup |G∞| %.12g", best, err, lim)
			}
			got, w, err := HinfNorm(sys)
			if err != nil {
				t.Fatalf("HinfNorm: %v", err)
			}
			if math.Abs(got-want) > 1e-9*want || math.Abs(w-wWant) > 1e-3 {
				t.Fatalf("HinfNorm = (%.15g, %.10g), want (%.15g, %.10g)", got, w, want, wWant)
			}
		})
	}
}
