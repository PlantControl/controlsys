package controlsys

import (
	"math"
	"os"
	"testing"
)

// TestDelayNyquistMergedAxisTriple: den holds three lightly damped pairs at
// 1.25e-5..1.28e-5, merged by addAxisPole into one triple axis pole whose
// widened indentation passes the growth check; the walk from it exceeded
// the point budget, and clusterWinding, which tried the origin only after
// an axis root, never formed the verified disk around all six poles.
// IsStable, HinfNorm and DiskMargin errored where 690c75a was correct.
// Oracles: argument principle on χ at two box sizes, the sweep oracle, and
// the parity of the real RHP roots from the sign of χ(0). The loop check
// cuts the point budget so that the failing walk ends early; the public
// calls, each walking 2e6 points first (~6 min in all), run with NYQ_SLOW set.
func TestDelayNyquistMergedAxisTriple(t *testing.T) {
	skipSlowUnderRace(t)
	cases := []struct {
		q     quasiLoop
		roots int
	}{
		// Re of the pairs +7.5e-13, -2.3e-15, -3.4e-13.
		{quasiLoop{num: []float64{9.117894554170206, 36.08603565127111, 64.80839610954266, 56.75361467660462, 21.787669122644516, 3.182415199292365, 0.12149412794757106}, den: []float64{1, 0.554582595918105, 4.762255748851759e-10, 2.6435956536511254e-10, 7.557582134506691e-20, 4.1995698627143264e-20, 3.996815122071488e-30, 2.2233016709337304e-30}, tau: 0.0025738849122849145}, 0},
		{quasiLoop{num: []float64{0.2730903911339774, 0.03595091146875167, 0.01755807551060303, 0.04919738575261187, 0.21343623784447122, 0.09271779567446202, 0.00034873192441782746}, den: []float64{1, 0.5499911852977044, 6.848907327003045e-10, 3.7649356287443065e-10, 1.5633386043608903e-19, 8.58984076998968e-20, 1.1893078947545774e-29, 6.531871092222672e-30}, tau: 0.010062852247457375}, 2},
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
		l, err := delayLoopFromSystem(p, "test")
		if err != nil {
			t.Fatal(err)
		}
		l.maxPoints = 1 << 16
		if res, err := l.nyquist(0.5); err != nil || res.axisRoot || res.roots != c.roots {
			t.Errorf("case %d: open-loop Nyquist roots = %d (axisRoot %v), %v; oracle %d", i, res.roots, res.axisRoot, err, c.roots)
		}
		if os.Getenv("NYQ_SLOW") == "" || raceEnabled {
			continue
		}
		cl, err := Feedback(p, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		if stable, err := cl.IsStable(); err != nil || stable != (c.roots == 0) {
			t.Errorf("case %d: IsStable = %v, %v; oracle %d RHP roots", i, stable, err, c.roots)
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
