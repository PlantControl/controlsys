package controlsys

import (
	"math"
	"testing"
)

// TestDelayNyquistDDTrustCrowdedIntegrators: den = s^m·q(s) with q holding
// tiny real roots and a lightly damped pair below 1e-4, plus one real pole
// at 0.01..0.02. The double solve fails over ω < 0.01, so the cluster walk
// needs the double-double fallback there, but the real pole denied it over
// [0, 10|q|] (ddDenyReach·|Re q|). The cluster circle around the origin
// failed its axis check, and the walk from the smaller circles ran out of
// its point budget, so IsStable and DiskMargin errored. The fallback is now
// denied only near poles with |Re q| < ddTrustRe·ρ, and dd circles try the
// outer radii first, where the model of S reaches far enough. Oracles: argument
// principle on χ at two box sizes, the sweep oracle, and the parity of the
// real RHP roots from the sign of χ(0).
func TestDelayNyquistDDTrustCrowdedIntegrators(t *testing.T) {
	skipSlowUnderRace(t)
	cases := []struct {
		q     quasiLoop
		roots int
	}{
		{quasiLoop{num: []float64{6.840024404727002, 17.378659843517983, 20.769470510544636, 12.205225117024748, 3.4300126522593968, 0.41430325974383475, 0.02126540360186989, 0.00038621074628372373}, den: []float64{1, -0.02234011552778956, 6.512873463748606e-08, -1.904256343145472e-10, 6.010100728674183e-16, -6.602226110472291e-23, 4.120430178343998e-31, 0, 0}, tau: 0.019739683857353685}, 0},
		{quasiLoop{num: []float64{0.2326413956659474, 0.16537057555545742, 0.11902246846704141, 0.18122730384117974, 0.1329130430827498, 0.25747425950551883, 0.00979214642849186}, den: []float64{1, -0.01241931283546442, 4.747864371577226e-08, 5.661584892535147e-13, 1.381355240952369e-19, 7.148030859397882e-25, 2.1271851320950975e-32, 0}, tau: 0.007061311110319203}, 2},
		// The walk from a mid-range dd circle (3e-4) exceeded the point
		// budget; from an outer one (4e-3) it takes ~6e3 points.
		{quasiLoop{num: []float64{11.804907045235103, 47.69278444420078, 78.93833711881778, 56.554249325501495, 20.364449939570378, 4.073141801943374, 0.4707595594277479, 0.0310525592479381, 0.0010784630364785225, 1.5231794863841214e-05}, den: []float64{1, -0.030506554427827935, 5.2038982663217025e-08, -8.555476402942147e-14, 1.4908513010377137e-19, -4.276994673844071e-28, 1.8994237953309433e-37, -2.30930559019089e-47, 0, 0, 0}, tau: 0.012199028614811678}, 0},
	}
	for i, c := range cases {
		q := c.q
		for _, box := range []float64{200, 2000} {
			if got := oracleRHPRoots(t, q.chi, box, q.den, q.num); got != c.roots {
				t.Fatalf("case %d: oracle box %g: %d RHP roots, want %d", i, box, got, c.roots)
			}
		}
		if got, ok := sweepOracle(q); !ok || got != c.roots {
			t.Fatalf("case %d: sweep oracle %d (ok %v), want %d", i, got, ok, c.roots)
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
		l, err := delayLoopFromSystem(p, "test")
		if err != nil {
			t.Fatal(err)
		}
		if res, err := l.nyquist(0.5); err != nil || res.axisRoot || res.roots != c.roots {
			t.Errorf("case %d: open-loop Nyquist roots = %d (axisRoot %v), %v; oracle %d", i, res.roots, res.axisRoot, err, c.roots)
		}
		dm, err := DiskMargin(p)
		norm, _, herr := HinfNorm(cl)
		if c.roots > 0 {
			if err != nil || dm.Alpha != 0 {
				t.Errorf("case %d: DiskMargin = %+v, %v; want alpha 0 for an unstable loop", i, dm, err)
			}
			if herr != nil || !math.IsInf(norm, 1) {
				t.Errorf("case %d: HinfNorm = %g, %v; want +Inf for an unstable loop", i, norm, herr)
			}
			continue
		}
		supT, supS := denseLoopPeaks(q)
		if err != nil || !(dm.Alpha <= 1/supS*(1+1e-6) && dm.Alpha >= 1/supS*(1-1e-3)) {
			t.Errorf("case %d: DiskMargin = %+v, %v; dense alpha %g", i, dm, err, 1/supS)
		}
		if herr != nil || !(norm >= supT*(1-1e-6) && norm <= supT*(1+1e-3)) {
			t.Errorf("case %d: HinfNorm = %g, %v; dense sup|T| %g", i, norm, herr, supT)
		}
	}
}
