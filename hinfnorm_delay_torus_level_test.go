package controlsys

import (
	"fmt"
	"math"
	"testing"
)

// TestHinfNormTorusLevelTail pins BIGRKC: G2 = G + kr/(s + p)² of
// torusClusterModel, alone and as U·diag(G, G2), with a finite peak barely
// above the torus sup of |G∞|. A grid sample exceeds that sup, so
// levelBelow certifies a level between them and torusPastLimit is skipped;
// tailGain then closes the tail only far past the delay span, which used to
// fail ("delay grid exceeds"), and torusTailBelow must close it instead.
// The first and last cases, from the report, pass without it. The oracle is a dense ω scan of the closed-form
// gain, refined by golden section, past the peak HinfNorm reports, and the
// torus sup of |G∞| from a grid search.
func TestHinfNormTorusLevelTail(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	for _, tc := range []struct {
		ci    int
		kr, p float64
	}{
		{0, -365, 100},
		{0, -200, 200},
		{0, -100, 200},
		{1, 200, 200},
		{1, 347, 200},
		{1, 347, 30},
	} {
		c := expansionFinite[tc.ci]
		sup := torusSupOf(func(p, q float64) float64 { return gInfTorus(c.d22, c.d12, c.d21, p, q) })
		_, _, sigma := torusClusterModel(t, c, tc.kr, tc.p, rotations[0][0])
		fin, wf := oraclePeak(sigma, 2500, 2e-3)
		if !(fin > sup*(1+1e-8)) {
			t.Fatalf("oracle peak %.15g at %g not above torus sup %.15g", fin, wf, sup)
		}
		for ri, rot := range append([][2][]float64{{}}, rotations...) {
			t.Run(fmt.Sprintf("%s/kr=%g,p=%g/%s", c.name, tc.kr, tc.p, [...]string{"siso", "diag", "rot"}[ri]), func(t *testing.T) {
				var sys *System
				if rot[0] == nil {
					sys = torusClusterModelSISO(t, c, tc.kr, tc.p)
				} else {
					sys, _, _ = torusClusterModel(t, c, tc.kr, tc.p, rot[0])
				}
				got, w, err := HinfNorm(sys)
				if err != nil || math.Abs(got-fin) > 1e-8*fin || math.Abs(w-wf) > 1e-3 {
					t.Fatalf("HinfNorm = %.15g @%g, %v; want %.15g @%g", got, w, err, fin, wf)
				}
			})
		}
	}
}
