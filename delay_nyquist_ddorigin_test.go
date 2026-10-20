package controlsys

import (
	"math"
	"testing"
)

// TestDelayNyquistNearDoublePairCluster: den is a near-double lightly damped
// pair at 1e-5..5e-4, some of it in the RHP.
//
// origin cases: with the double-double fallback denied only within
// ddDenyReach·|Re q| of the pair, the walk first fails at a non-pole ω
// inside that band, below the pair, where no cluster fits (hi <=
// w0/clusterMargin), so clusterWinding gave up and IsStable, HinfNorm and
// DiskMargin errored ("singular complex matrix"), though a failure at ω = 0
// found a verified origin cluster around all four poles. clusterWinding now
// retries with the origin as a candidate centre.
//
// inner cases: the walk's cluster at the pair frequency encloses RHP poles
// (inner > 0) and the fallback is denied at axisGap beside it.
//
// Oracles: argument principle on χ at two box sizes, the sweep oracle, the
// parity of the real RHP roots from χ(0), and dense sup|T| and sup|S-1/2|.
func TestDelayNyquistNearDoublePairCluster(t *testing.T) {
	cases := []struct {
		name   string
		q      quasiLoop
		roots  int
		origin bool
		inner  bool
	}{
		{"origin, stable", quasiLoop{num: []float64{3.2352121915287078, 2.9318462185373737, 0.9209310942430203, 0.09073512716732975}, den: []float64{1, 7.916822512875715e-07, 2.23193650784524e-09, 8.409041323629466e-16, 1.2424568960202456e-18}, tau: 0.001993944126195067}, 0, true, false},
		{"origin, 3 RHP roots", quasiLoop{num: []float64{0.0035412403357989973, 0.0009290551691596314, 0.01490447771022342, 0.00925057234567085, -0.03453728719690581}, den: []float64{1, 0.03877321282183848, 5.692903947267971e-07, 3.946154617055747e-09, 2.6374292206713098e-14, 1.003382448129943e-16}, tau: 0.007897857148837936}, 3, true, false},
		{"origin encloses RHP pair, stable", quasiLoop{num: []float64{3.791965960551473, 2.7528155057910957, 0.7364035725776352, 0.06745284845669158}, den: []float64{1, -3.0133005122967016e-09, 3.88622822121841e-10, -1.7582694396660176e-19, 3.763193317154558e-20}, tau: 0.005549795476667578}, 0, true, true},
		{"origin encloses 4 RHP poles, stable", quasiLoop{num: []float64{2.963050861212342, 2.8760128991676925, 1.0467592758999353, 0.10547358513039992}, den: []float64{1, -5.9612286734177074e-06, 2.8304071543184004e-08, -8.160314772109964e-14, 2.0006166209052638e-16}, tau: 0.2704082277437164}, 0, true, true},
		{"origin, RHP real pole, stable", quasiLoop{num: []float64{3.2399615534975843, 2.545764344023167, 0.9223801262678863, 0.11848131574343322, 0.004126254233369466}, den: []float64{1, -0.4487044762212055, -1.6249377792498235e-06, -5.768687146726817e-09, -1.0475259969808848e-14, -1.85406598768347e-17}, tau: 0.06078182083204582}, 0, true, false},
		{"origin, 2 RHP roots", quasiLoop{num: []float64{0.22166406412870573, 0.15330268277356618, -0.39299344020775157, 0.38861239317069174}, den: []float64{1, 2.3472754316657076e-06, 1.3993908959456621e-08, 1.6422374136774044e-14, 4.8949086593479687e-17}, tau: 0.0003861145262053144}, 2, true, false},
		{"pair cluster encloses 4 RHP poles, stable", quasiLoop{num: []float64{5.537510477677026, 11.083212862195024, 9.36873847264269, 2.747978146498711}, den: []float64{1, -4.841892472195739e-08, 4.211482521714141e-07, -1.0195787704592054e-14, 4.4341462519048443e-14}, tau: 0.00640788978531219}, 0, false, true},
		{"pair cluster encloses RHP pair, 4 RHP roots", quasiLoop{num: []float64{-0.10001631017511914, 0.17353666104041401, 0.20980088519890508, -0.10933698566813194, 0.01366866251438389}, den: []float64{1, 0.09448176700595785, 3.759758798052076e-08, 1.308643397253839e-08, -2.1963128386391888e-15, 4.53142795006062e-16}, tau: 0.0026622700140613922}, 4, false, true},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			q := c.q
			for _, box := range []float64{200, 2000} {
				if got := oracleRHPRoots(t, q.chi, box, q.den, q.num); got != c.roots {
					t.Fatalf("oracle box %g: %d RHP roots, want %d", box, got, c.roots)
				}
			}
			if got, ok := sweepOracle(q); !ok || got != c.roots {
				t.Fatalf("sweep oracle %d (ok %v), want %d", got, ok, c.roots)
			}
			if odd := real(q.chi(0)) < 0; odd != (c.roots%2 == 1) {
				t.Fatalf("χ(0) = %g, parity disagrees with %d roots", real(q.chi(0)), c.roots)
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
				t.Errorf("IsStable = %v, %v; oracle %d RHP roots", stable, err, c.roots)
			}
			l, err := delayLoopFromSystem(p, "test")
			if err != nil {
				t.Fatal(err)
			}
			res, err := l.nyquist(0.5)
			if err != nil || res.axisRoot || res.roots != c.roots {
				t.Fatalf("open-loop Nyquist roots = %d (axisRoot %v), %v; oracle %d", res.roots, res.axisRoot, err, c.roots)
			}
			origin, inner := false, false
			for _, a := range l.clusters {
				origin = origin || a.w == 0
				if a.inner > 0 && (a.w == 0 || !l.ddAllowed(a.w+l.axisGap(a.w))) {
					inner = true
				}
			}
			if origin != c.origin || inner != c.inner {
				t.Errorf("clusters %+v: origin %v, inner %v; want %v, %v", l.clusters, origin, inner, c.origin, c.inner)
			}
			dm, err := DiskMargin(p)
			norm, _, herr := HinfNorm(cl)
			if c.roots > 0 {
				if err != nil || dm.Alpha != 0 {
					t.Errorf("DiskMargin = %+v, %v; want alpha 0 for an unstable loop", dm, err)
				}
				if herr != nil || !math.IsInf(norm, 1) {
					t.Errorf("HinfNorm = %g, %v; want +Inf for an unstable loop", norm, herr)
				}
				return
			}
			supT, supS := denseLoopPeaks(q)
			if err != nil || !(dm.Alpha <= 1/supS*(1+1e-6) && dm.Alpha >= 1/supS*(1-1e-3)) {
				t.Errorf("DiskMargin = %+v, %v; dense alpha %g", dm, err, 1/supS)
			}
			if herr != nil || !(norm >= supT*(1-1e-6) && norm <= supT*(1+1e-3)) {
				t.Errorf("HinfNorm = %g, %v; dense sup|T| %g", norm, herr, supT)
			}
		})
	}
}
