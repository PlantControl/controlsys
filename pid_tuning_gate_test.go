package controlsys

import (
	"context"
	"slices"
	"strings"
	"testing"
)

// NML6QA: undelayed discrete plants with multiple poles at z = 1, where the
// back-off's deep gates (depths 20, 19, sometimes 16) cannot be evaluated,
// and designs sit only at depths the gates skip (3-5, 13-15, 18). TunePID
// read the unevaluated gates as failures and reported the target
// unattainable; explicit crossovers at those depths have designs, and
// MATLAB pidtune returns one. Each design must be Schur-stable by the exact
// test, meet its reported margin at its crossover on the raw num/den, keep
// sampled |S| <= 4 to Nyquist, and stay within the back-off's budget.
func TestTunePIDBackOffScansPastBlindGates(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	cases := []struct {
		name     string
		num, den []float64
		dt       float64
		fam      PidtuneType
	}{
		{"(z-0.95)/(z-1)^3", []float64{0.2376376495436092, -0.22623825681198792}, []float64{1, -3, 3, -1}, 0.0411986772067474, PidtunePIDF},
		{"(z-0.99)/((z-1)^3(z-0.72))", []float64{0.05561793864532625, -0.055150540391457045}, []float64{1, -3.7172691807634135, 5.151807542290241, -3.1518075422902405, 0.7172691807634135}, 0.022067595844186595, PidtunePIDF},
		{"k/((z-1)^3 lags) order 6", []float64{-0.26076155489343816}, []float64{1, -5.301065168243693, 11.569174727312362, -13.256141604617728, 8.364200100816014, -2.7313484558098238, 0.3551804005428702}, 0.14827014447740874, PidtunePID},
		{"k/((z-1)^3 lag) order 4", []float64{0.0406846146866439}, []float64{1, -3.8564046132602567, 5.569527491905905, -3.5698411440310416, 0.8567182653853925}, 0.060709355752838014, PidtunePID},
	}
	for _, c := range cases {
		name := c.name + " " + string(c.fam)
		r, err := TunePID(context.Background(), makeDiscretePlant(t, c.num, c.den, c.dt), c.fam, PIDTuningOptions{})
		if !pidCheckZPoleResult(t, name, c.num, c.den, c.dt, r, err) {
			t.Errorf("%s: no design: %v", name, err)
			continue
		}
		pidCheckUndelayedMarginAndS(t, name, c.num, c.den, c.dt, r)
		if !slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "lowered to") }) {
			t.Errorf("%s: no back-off in %q", name, r.Evidence.Warnings)
		}
		if r.Evidence.Evaluations > 6000 {
			t.Errorf("%s: %d evaluations, want <= 6000", name, r.Evidence.Evaluations)
		}
	}
}
