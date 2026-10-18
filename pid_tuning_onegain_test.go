package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

// WDBBJB: a free P design on k/(s²+as+b) with ζ≈0.0014 took the plant's
// margin at the default crossover 0.137 (Kp=−1), crossing 0 dB again past
// the mode with PM 0.24° and |S| 241. MATLAB targets 60° by moving wc: the
// phase of 1/(b−w²+jaw) is −120° at w = a/(2√3) + √(a²/12+b), where
// Kp = −1/|G|. Oracle: that closed form, a dense |S| sweep, dense margins
// and the winding of 1+L.
func TestTunePIDFreePMovesCrossoverOffLightlyDampedMode(t *testing.T) {
	k, a, b := -1.054599885306892, 0.00299557580140128, 1.0733714787112956
	r, err := TunePID(context.Background(), makePlant(t, []float64{k}, []float64{1, a, b}), PidtuneP, PIDTuningOptions{})
	if err != nil {
		t.Fatal(err)
	}
	g := func(w float64) complex128 {
		s := complex(0, w)
		return complex(k, 0) / (s*s + complex(a, 0)*s + complex(b, 0))
	}
	wc := a/(2*math.Sqrt(3)) + math.Sqrt(a*a/12+b)
	kp := -1 / cmplx.Abs(g(wc))
	c := *r.Controller
	if c.Ki != 0 || c.Kd != 0 || math.Abs(c.Kp-kp) > 1e-6*math.Abs(kp) {
		t.Fatalf("gains %+v, want Kp %g", c, kp)
	}
	if math.Abs(r.Evidence.AchievedCrossover-wc) > 1e-9*wc {
		t.Fatalf("crossover %g, want %g", r.Evidence.AchievedCrossover, wc)
	}
	if !slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "moved to") }) {
		t.Fatalf("no move warning: %v", r.Evidence.Warnings)
	}
	L := func(w float64) complex128 { return complex(c.Kp, 0) * g(w) }
	margins := pidLoopMargins(L, 1e-3, 1e3)
	if len(margins) != 2 {
		t.Fatalf("dense margins %v", margins)
	}
	for _, m := range margins {
		if m < 59.9 || m > 180 {
			t.Fatalf("dense margins %v", margins)
		}
	}
	if peak, at := pidDenseSensitivity(c, g, 1e-2, 1e2); peak > 2 {
		t.Fatalf("|S| %g at ω=%g", peak, at)
	}
	if wind := pidWindingDeg(L, 1e-6, 1e4); math.Abs(wind) > 1 {
		t.Fatalf("arg(1+L) turns %g°", wind)
	}
}

// Free P and I designs on seeded lightly damped plants (ζ 0.001–0.2, with
// an extra pole, a zero or a delay) keep dense |S| ≤ 4 and a stable loop:
// the winding of 1+L is 0 under P and 90° under I, the integrator's
// indentation. Before WDBBJB a third of them returned PM 0.1–12° designs.
func TestTunePIDFreeOneGainLightlyDampedRobust(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 43))
	designs, moved := 0, 0
	for trial := range 60 {
		zeta := math.Pow(10, -3+2.3*rng.Float64())
		wn := math.Pow(10, -.5+1.5*rng.Float64())
		k := (2*rng.Float64() - 1) * 3
		num := []float64{k * wn * wn}
		den := []float64{1, 2 * zeta * wn, wn * wn}
		if trial%3 == 1 {
			p := .2 + 3*rng.Float64()
			den = []float64{1, den[1] + p, den[2] + p*den[1], p * den[2]}
			num = []float64{num[0] * p}
		}
		if trial%4 == 2 {
			z := .3 + 3*rng.Float64()
			num = []float64{num[0] / z, num[0]}
		}
		tau := 0.
		if trial%5 == 3 {
			tau = .3 * rng.Float64() / wn
		}
		plant := pidPlantDelayed(t, num, den, tau)
		g := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*complex(tau, 0))
		}
		for _, family := range []PidtuneType{PidtuneP, PidtuneI} {
			r, err := TunePID(context.Background(), plant, family, PIDTuningOptions{})
			if errors.Is(err, ErrPIDTuningTargetUnattainable) {
				continue
			}
			if err != nil {
				t.Fatalf("trial %d %s: %v", trial, family, err)
			}
			designs++
			if slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "moved to") }) {
				moved++
			}
			c := *r.Controller
			wc := r.Evidence.AchievedCrossover
			hi := math.Max(wc, wn) * 1e2
			if peak, at := pidDenseSensitivity(c, g, wc/1e3, hi); peak > 4 {
				t.Errorf("trial %d %s %v/%v τ=%g: %+v |S| %g at ω=%g", trial, family, num, den, tau, c, peak, at)
			}
			L := func(w float64) complex128 {
				s := complex(0, w)
				return (complex(c.Kp, 0) + complex(c.Ki, 0)/s) * g(w)
			}
			want := 0.
			if family == PidtuneI {
				want = 90
			}
			if wind := pidWindingDeg(L, 1e-6, hi*1e2); math.Abs(wind-want) > 1 {
				t.Errorf("trial %d %s: %+v arg(1+L) turns %g°", trial, family, c, wind)
			}
		}
	}
	if designs < 40 || moved < 15 {
		t.Fatalf("designs %d, moved %d", designs, moved)
	}
	t.Logf("designs %d, moved %d", designs, moved)
}
