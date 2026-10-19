package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
	"testing"
)

// TestDelayNyquistDDNearAxisCluster: den = (s²+w0²)(s²+2ζw1·s+w1²)(s+a) with
// w1 ≈ w0. Rounding moves the axis pair to Re ≈ 2.7e-10 (an RHP pole, not a
// boundary one) with the second pair ~3e-8 away. The double solve fails
// near the pair and clusterWinding counts the roots; a double-double
// fallback there evaluated L, hid the failure, and the uniform walk missed
// the phase turn (roots=2: IsStable false, HinfNorm +Inf, DiskMargin 0).
// Oracles: argument principle on χ and a dense sweep of |T| and |S-1/2|.
func TestDelayNyquistDDNearAxisCluster(t *testing.T) {
	cases := []quasiLoop{
		{num: []float64{1.9130262763840757, -4.530004355564204, -1.9893952709987275, -10.266928489786384, -4.27874143863459}, den: []float64{1, 0.4132314758797427, 6.439107723185883, 2.660821350331567, 10.365527067696277, 4.283295605039173}, tau: 0.15790851475545137},
		{num: []float64{9.384289704050971, -30.563643489443816, 31.131287966650696, -911.9375134454763, -105.33336767344285}, den: []float64{1, 0.11603118820977044, 61.02071071776695, 7.079607987456788, 930.8817841253477, 107.99003600494858}, tau: 0.023352110431757334},
		{num: []float64{0.4592576099537782, 1.1381175806259498, -16.863079593709763, -3.5802929938600148, -18.332386625492255}, den: []float64{1, 4.755979490535817, 3.928225751798537, 18.682695783580705, 3.8577393888266016, 18.34759393054576}, tau: 0.13754310283030566},
		{num: []float64{0.6910037283375035, 3.3233274187037765, 7.295987359497399, 2.181872269533212, 7.233652123163928}, den: []float64{1, 0.6651342877732155, 8.611431733742156, 5.727714919696599, 18.539189126215323, 12.330862655418429}, tau: 0.003648063502714841},
		{num: []float64{4.836088291165206, -11.045057037491008, -15.949934502741609, -121.23350400363579, -130.03271390666904}, den: []float64{1, 1.060903618065211, 22.148737927434308, 23.49766132392631, 122.64164794404581, 130.11080325492483}, tau: 0.1321113818341096},
	}
	for i, q := range cases {
		if want := oracleRHPRoots(t, q.chi, 200, q.den, q.num); want != 0 {
			t.Fatalf("case %d: oracle %d RHP roots, want 0", i, want)
		}
		supT, supS := denseLoopPeaks(q)
		p := makePlant(t, q.num, q.den)
		if err := p.SetInputDelay([]float64{q.tau}); err != nil {
			t.Fatal(err)
		}
		cl, err := Feedback(p, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		if stable, err := cl.IsStable(); err != nil || !stable {
			t.Errorf("case %d: IsStable = %v, %v; oracle 0 RHP roots", i, stable, err)
		}
		norm, _, err := HinfNorm(cl)
		if err != nil || !(norm >= supT*(1-1e-6) && norm <= supT*(1+1e-3)) {
			t.Errorf("case %d: HinfNorm = %g, %v; dense sup|T| %g", i, norm, err, supT)
		}
		dm, err := DiskMargin(p)
		if err != nil || !(dm.Alpha <= 1/supS*(1+1e-6) && dm.Alpha >= 1/supS*(1-1e-3)) {
			t.Errorf("case %d: DiskMargin = %+v, %v; dense alpha %g", i, dm, err, 1/supS)
		}
	}
}

// denseLoopPeaks returns sup|T| and sup|S-1/2| of L = q.L on a log grid
// with geometric clusters at the root frequencies of den, each peak refined
// by golden section between its neighbours.
func denseLoopPeaks(q quasiLoop) (supT, supS float64) {
	ws := make([]float64, 0, 400000)
	for i := range 300001 {
		ws = append(ws, math.Pow(10, -6+10*float64(i)/300000))
	}
	roots, _ := Poly(q.den).Roots()
	for _, r := range roots {
		w := math.Abs(imag(r))
		if w == 0 {
			continue
		}
		for d := 1e-12; d < 1e-1; d *= 1.05 {
			ws = append(ws, w*(1+d), w*(1-d))
		}
	}
	slices.Sort(ws)
	ws = slices.Compact(ws)
	tf := func(w float64) float64 { l := q.L(w); return cmplx.Abs(l / (1 + l)) }
	sf := func(w float64) float64 { l := q.L(w); return cmplx.Abs(1/(1+l) - 0.5) }
	peak := func(f func(float64) float64) float64 {
		k := 0
		for i := range ws {
			if f(ws[i]) > f(ws[k]) {
				k = i
			}
		}
		a, b := ws[max(k-1, 0)], ws[min(k+1, len(ws)-1)]
		g := (math.Sqrt(5) - 1) / 2
		for range 200 {
			c, d := b-g*(b-a), a+g*(b-a)
			if f(c) > f(d) {
				b = d
			} else {
				a = c
			}
		}
		return max(f(ws[k]), f((a+b)/2))
	}
	return peak(tf), peak(sf)
}
