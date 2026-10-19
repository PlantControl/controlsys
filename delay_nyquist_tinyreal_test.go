package controlsys

import (
	"math"
	"testing"
)

// TestDelayNyquistTinyRealPolesAtOrigin: den = s^m·q(s) with real roots of q
// within ~1e-6 of the origin, off the axis, and one more pole near 1e-2 (RHP
// in cases 0-4 but 3, where two are). Neither the solve nor the
// double-double fallback evaluates L within ~1e-5, so indent fails and the
// cluster at the origin must enclose the tiny poles; formCluster's three
// trial radii per pole count all fell outside the band where |L| dominates
// without enclosing a zero of L (IsStable, HinfNorm, DiskMargin errored).
// Oracles: argument principle on χ and a dense sweep of |T| and |S-1/2|.
func TestDelayNyquistTinyRealPolesAtOrigin(t *testing.T) {
	cases := []quasiLoop{
		{num: []float64{5.52461747222729, 10.25395216896589, 7.411393861456306, 2.08816698988852, 0.22553859559127404, 0.007420390418043597}, den: []float64{1, -0.008139247275845123, -7.936569881325126e-08, -2.288908801286239e-14, 0, 0, 0}, tau: 0.007702370618739055},
		{num: []float64{8.355629538965681, 22.563917082193022, 24.398355785727624, 9.639372238938034, 1.1305743755868467, 0.03782054040717314}, den: []float64{1, -0.0028639980109894134, -3.425788951502568e-05, -2.0789893203109462e-11, 0, 0, 0}, tau: 0.007443101012386161},
		{num: []float64{4.551658530169233, 3.0753000142052764, 0.9194674239822193, 0.11054865480512942, 0.003182926796710191}, den: []float64{1, -0.0048136878504125066, -1.6692283338125261e-06, -2.651188267756408e-13, 0, 0}, tau: 0.19337704386281568},
		{num: []float64{8.306255406460808, 22.680683272368427, 26.113994842611685, 12.223126925616675, 1.6176301437180853, 0.05728619385528791}, den: []float64{1, 0.007068687570826481, -2.5428999759620925e-06, 3.8407723669266984e-13, 0, 0, 0}, tau: 0.002128012440629736},
		{num: []float64{13.222200959921482, 63.741266377137514, 129.45539425818563, 104.87423932251659, 34.825187773001446, 3.8595098872161198}, den: []float64{1, 0.12864478027547946, 1.8146789689495635e-10, -4.594790547398718e-19, 0, 0, 0}, tau: 0.003604611258568011},
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

// TestDelayNyquistTinyRealPolesDDCircles: den = s^m·q(s), q with real roots
// between ~1e-9 and ~1e-6 and one near 1e-3..1e-2. The double solve is
// singular on every circle that encloses only the poles near the origin,
// and a circle past the root near 1e-2 encloses zeros of L too, so no
// cluster verified (IsStable, HinfNorm, DiskMargin errored); the circles
// now retry with the double-double fallback. The last case needs the
// double pass kept: there the fallback verifies a circle of ~2e-8 inside
// the RHP pole at 1.3e-7, beside which the walk then fails. Oracles:
// argument principle on χ and a dense sweep of |T| and |S-1/2|.
func TestDelayNyquistTinyRealPolesDDCircles(t *testing.T) {
	cases := []quasiLoop{
		{num: []float64{1.1248828988699036, 5.337591591109049, 7.966020292280976, 3.9251826905697533, 0.16739720620721268, 0.0017474023654776626}, den: []float64{1, 0.004586181042618237, -1.0902408236859952e-08, -2.02597355493661e-17, 0, 0, 0}, tau: 0.1456899273039475},
		{num: []float64{7.428774737994522, 27.535483143910717, 26.496350076689353, 9.510730545180923, 1.1303308497133764, 0.004373771149703272}, den: []float64{1, 0.006386411776610228, 3.0129825054035514e-08, 1.2921688426134479e-17, 0, 0, 0}, tau: 0.025221635624798323},
		{num: []float64{13.957650091429352, 48.99854842584783, 43.80141762536648, 4.501015527990205, 0.1490286842188432, 0.0013924016856337376}, den: []float64{1, -0.006475852535513324, 1.1087869874845573e-10, -4.719961128965432e-19, 0, 0, 0}, tau: 0.0459183954903688},
		{num: []float64{6.0081561026318875, 42.587588538843804, 100.0156287702273, 78.17092084395361, 0.8939171565437924, 0.0022150939168672614}, den: []float64{1, 0.009108165476837352, 9.754504803190256e-10, -8.018521765223733e-19, 0, 0, 0}, tau: 0.002451378336097036},
		{num: []float64{1.9188317375876882, 1.5083785090172463, 0.32156836056367194, 0.015658860719559062}, den: []float64{1, 0.46404121934763415, -6.209067747296065e-08, 0, 0}, tau: 0.001181949427158492},
	}
	for i, q := range cases {
		want := oracleRHPRoots(t, q.chi, 200, q.den, q.num)
		p := makePlant(t, q.num, q.den)
		if err := p.SetInputDelay([]float64{q.tau}); err != nil {
			t.Fatal(err)
		}
		l, err := delayLoopFromSystem(p, "delay loop")
		if err != nil {
			t.Fatal(err)
		}
		if res, err := l.nyquist(0.5); err != nil || res.axisRoot || res.roots != want {
			t.Errorf("case %d: nyquist roots %d, axis root %v, %v; oracle %d", i, res.roots, res.axisRoot, err, want)
		}
		cl, err := Feedback(p, nil, -1)
		if err != nil {
			t.Fatal(err)
		}
		if stable, err := cl.IsStable(); err != nil || stable != (want == 0) {
			t.Errorf("case %d: IsStable = %v, %v; oracle %d RHP roots", i, stable, err, want)
		}
		norm, _, normErr := HinfNorm(cl)
		dm, dmErr := DiskMargin(p)
		if want != 0 {
			if normErr != nil || !math.IsInf(norm, 1) {
				t.Errorf("case %d: HinfNorm = %g, %v; oracle unstable", i, norm, normErr)
			}
			if dmErr != nil || dm.Alpha != 0 {
				t.Errorf("case %d: DiskMargin = %+v, %v; oracle unstable", i, dm, dmErr)
			}
			continue
		}
		supT, supS := denseLoopPeaks(q)
		if normErr != nil || !(norm >= supT*(1-1e-6) && norm <= supT*(1+1e-3)) {
			t.Errorf("case %d: HinfNorm = %g, %v; dense sup|T| %g", i, norm, normErr, supT)
		}
		if dmErr != nil || !(dm.Alpha <= 1/supS*(1+1e-6) && dm.Alpha >= 1/supS*(1-1e-3)) {
			t.Errorf("case %d: DiskMargin = %+v, %v; dense alpha %g", i, dm, dmErr, 1/supS)
		}
	}
}
