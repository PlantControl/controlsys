package controlsys

import (
	"math"
	"testing"
)

// TestDelayNyquistDDReachNearDoublePair: den has a near-double lightly
// damped pair at |p| ~ 1e-4..1e-3 (coefficients down to 1e-15), whose
// companion realization makes the double solve fail over a band of ±8% to
// ±40% of the pair frequency, far from the poles. ddAllowed denied the
// double-double fallback across the whole band (the nearest pole is not a
// boundary one), so IsStable returned "singular complex matrix". The
// fallback is now denied only within ddDenyReach·|Re p| of the pair, where
// the failure still reaches clusterWinding. Oracles: argument principle on χ
// at two box sizes, and the parity of the real RHP roots from the sign of
// χ(0) (χ(x) > 0 for large real x).
func TestDelayNyquistDDReachNearDoublePair(t *testing.T) {
	cases := []struct {
		q     quasiLoop
		roots int
	}{
		{quasiLoop{num: []float64{0.11076720035335656, -6.130423072572409, -1.025734097506373, 2.740715007718803}, den: []float64{1, 2.511832216255788e-06, 6.992159985979351e-07, 8.781546547116625e-13, 1.2222520172891677e-13}, tau: 0.26598736421075958}, 2},
		{quasiLoop{num: []float64{0.7888666874022671, 3.8654855207558545, -0.12212921074024871, -0.1829508091600904}, den: []float64{1, 1.4334915553463322e-07, 6.39209078346261e-08, 4.581503711340512e-15, 1.0214704504120125e-15}, tau: 0.00039642299973125048}, 1},
		{quasiLoop{num: []float64{1.7881654623960874, 7.365322715526643, 0.7011603266679345, 1.4299568358509152}, den: []float64{1, 5.993352121789086e-07, 4.527910987579485e-07, 1.3568679771305505e-13, 5.1254924448136614e-14}, tau: 0.018411330449251636}, 0},
		{quasiLoop{num: []float64{2.193824372720179, 2.051696185825775, 0.29701318951193106, 0.24548913417721607}, den: []float64{1, 7.603731309054537e-08, 3.979193225034613e-08, 1.512835750545803e-15, 3.9584943929603884e-16}, tau: 0.005116032699002313}, 0},
		{quasiLoop{num: []float64{1.4696594522796829, 5.852358084584884, 1.474495068102936, 0.8124544370759264}, den: []float64{1, 2.627481221635727e-07, 2.61037031617441e-07, 3.4293492668914044e-14, 1.703508071627343e-14}, tau: 0.00014714369571039017}, 0},
	}
	for i, c := range cases {
		q := c.q
		for _, box := range []float64{200, 2000} {
			if got := oracleRHPRoots(t, q.chi, box, q.den, q.num); got != c.roots {
				t.Fatalf("case %d: oracle box %g: %d RHP roots, want %d", i, box, got, c.roots)
			}
		}
		if odd := real(q.chi(0)) < 0; odd != (c.roots%2 == 1) {
			t.Fatalf("case %d: χ(0) = %g, parity disagrees with %d roots", i, real(q.chi(0)), c.roots)
		}
		p := makePlant(t, q.num, q.den)
		if err := p.SetInputDelay([]float64{q.tau}); err != nil {
			t.Fatal(err)
		}
		cl, err := Feedback(p, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		if stable, err := cl.IsStable(); err != nil || stable != (c.roots == 0) {
			t.Errorf("case %d: IsStable = %v, %v; oracle %d RHP roots", i, stable, err, c.roots)
		}
		e, err := newDelayLFT(cl)
		if err != nil {
			t.Fatal(err)
		}
		if _, res, err := e.stability(); err != nil || res.axisRoot || res.roots != c.roots {
			t.Errorf("case %d: closed-loop roots = %d (axisRoot %v), %v; oracle %d", i, res.roots, res.axisRoot, err, c.roots)
		}
		l, err := delayLoopFromSystem(p, "test")
		if err != nil {
			t.Fatal(err)
		}
		if res, err := l.nyquist(0.5); err != nil || res.axisRoot || res.roots != c.roots {
			t.Errorf("case %d: open-loop Nyquist roots = %d (axisRoot %v), %v; oracle %d", i, res.roots, res.axisRoot, err, c.roots)
		}
		norm, _, err := HinfNorm(cl)
		if c.roots > 0 {
			if err != nil || !math.IsInf(norm, 1) {
				t.Errorf("case %d: HinfNorm = %g, %v; want +Inf for an unstable loop", i, norm, err)
			}
			continue
		}
		supT, _ := denseLoopPeaks(q)
		if err != nil || !(norm >= supT*(1-1e-6) && norm <= supT*(1+1e-3)) {
			t.Errorf("case %d: HinfNorm = %g, %v; dense sup|T| %g", i, norm, err, supT)
		}
	}
}
