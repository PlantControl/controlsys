package controlsys

import (
	"context"
	"math"
	"strings"
	"testing"
)

func pidExplicitSWarned(r *PIDTuningResult) int {
	n := 0
	for _, w := range r.Evidence.Warnings {
		if strings.Contains(w, "explicit crossover does not bound") {
			n++
		}
	}
	return n
}

// 1/(s(s²+0.2s+1)) under PID at twice the default P crossover: the
// explicit design is kept, with |S| ≈ 40 at the resonance, and warned.
func TestTunePIDExplicitCrossoverSensitivityWarning(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, .2, 1, 0})
	r, err := TunePID(context.Background(), p, PidtunePID, PIDTuningOptions{CrossoverFrequency: .2316})
	if err != nil {
		t.Fatal(err)
	}
	c := *r.Controller
	if math.Abs(c.Kp-.18966)/.18966 > 1e-3 || math.Abs(c.Ki-.025563)/.025563 > 1e-3 || c.Kd != 0 {
		t.Errorf("gains %+v changed", c)
	}
	g := func(w float64) complex128 {
		s := complex(0, w)
		return 1 / (s * (s*s + .2*s + 1))
	}
	peak, at := pidDenseSensitivity(c, g, .002316, 23.16)
	if peak < 30 {
		t.Fatalf("dense |S| %g at ω=%g; reference no longer fragile", peak, at)
	}
	if pidExplicitSWarned(r) != 1 {
		t.Errorf("dense |S| %g at ω=%g, warnings %q", peak, at, r.Evidence.Warnings)
	}

	q := makePlant(t, []float64{1}, []float64{1, 1})
	r, err = TunePID(context.Background(), q, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1})
	if err != nil {
		t.Fatal(err)
	}
	g = func(w float64) complex128 { return 1 / complex(1, w) }
	if peak, at := pidDenseSensitivity(*r.Controller, g, .01, 100); peak > 2 || pidExplicitSWarned(r) != 0 {
		t.Errorf("1/(s+1) PI: dense |S| %g at ω=%g, warnings %q", peak, at, r.Evidence.Warnings)
	}

	d, err := NewFromSlices(1, 1, 1, []float64{.9}, []float64{1}, []float64{.1}, []float64{0}, .1)
	if err != nil {
		t.Fatal(err)
	}
	r, err = TunePID(context.Background(), d, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1})
	if err != nil {
		t.Fatal(err)
	}
	if pidExplicitSWarned(r) != 0 {
		t.Errorf("0.1/(z−0.9) PI: warnings %q", r.Evidence.Warnings)
	}
}
