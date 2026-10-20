package controlsys

import (
	"math"
	"testing"
)

// TestDelayNyquistMergedAxisPoleSpread: den holds two lightly damped pairs
// 1.4e-7 and 1.6e-7 apart, within addAxisPole's 1e-6 merge tolerance but
// beyond the 1e-7 indentation, so the merged double pole was indented as if
// both poles lay inside while the walk passed the second at ~1e-11. IsStable
// reported the unstable loop stable and the stable one unstable, with nil
// error. The indentation now encloses every merged pole. Oracles: argument
// principle on χ at two box sizes, the sweep oracle, and the parity of the
// real RHP roots from the sign of χ(0).
func TestDelayNyquistMergedAxisPoleSpread(t *testing.T) {
	cases := []struct {
		q     quasiLoop
		roots int
	}{
		// Re of the pairs +3e-11 and +8e-11; roots 0.43665±0.91543j.
		{quasiLoop{num: []float64{0.7194876744332475, 0.48307710020589417, 0.9006210999510125, 0.8689376494791426}, den: []float64{1, -2.2904812444629556e-10, 3.930218025243091e-06, -4.501249932868734e-16, 3.86165339109627e-12}, tau: 0.0005760929572876901}, 2},
		// Pairs at 1.6263e-5 and 1.6425e-5, Re -8e-12 and -6e-13.
		{quasiLoop{num: []float64{1.1516330598529883, 0.40152334289343644, 0.04128879625877322, 0.0012938485649076784}, den: []float64{1, 1.7245639736907134e-11, 5.342746792140293e-10, 4.646274483085076e-21, 7.135535394238958e-20}, tau: 0.00607265601266062}, 0},
		// Pairs at 1.3913e-6 and 1.6085e-6: the widened radius 2.2e-7 exceeds
		// indentLimit 2.9e-8 set by the conjugates, which once skipped the
		// first try and returned "L not finite" where 690c75a was correct.
		{quasiLoop{num: []float64{0.02627760588501345, 0.2207918587498755, 0.09123646473331047, 0.1267845430927261}, den: []float64{1, -4.604291177027741e-14, 4.522955059349833e-12, -1.0375621120906069e-25, 5.00821720153443e-24}, tau: 0.00018902756700667477}, 2},
		{quasiLoop{num: []float64{1, 1, 0.1, 0.01}, den: []float64{1, -4.604291177027741e-14, 4.522955059349833e-12, -1.0375621120906069e-25, 5.00821720153443e-24}, tau: 0.01}, 0},
	}
	for i, c := range cases {
		q := c.q
		for _, box := range []float64{50, 500} {
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
