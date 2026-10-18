package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"strings"
	"testing"
)

func pidHasWarning(r *PIDTuningResult, sub string) bool {
	return slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, sub) })
}

// pidSecondOrderPPM returns the margin of Kp·k/(s²+as+b) at its single
// crossover, closed form: |b−w²+jaw|² = (Kp·k)² is a quadratic in w².
func pidSecondOrderPPM(k, a, b, kp float64) (float64, bool) {
	q := (kp * k) * (kp * k)
	disc := (a*a-2*b)*(a*a-2*b) - 4*(b*b-q)
	if disc < 0 {
		return 0, false
	}
	w2 := (-(a*a - 2*b) + math.Sqrt(disc)) / 2
	if w2 <= 0 {
		return 0, false
	}
	w := math.Sqrt(w2)
	return 180 + cmplx.Phase(complex(kp*k, 0)/complex(b-w2, a*w))*180/math.Pi, true
}

// LV4BRD: free P on k/(s²+as−|b|) has PM capped near 20°, so the 60° point
// does not exist and wc0's design has |S| 7-9; base returned unattainable.
// Oracle: Routh (b+Kp·k > 0), the closed-form margin maximised over a dense
// Kp grid, and dense |S|. The design must be stable, within 0.5° of the
// best margin, and keep dense |S| <= 4.
func TestTunePIDFreePUnstablePlantBestMargin(t *testing.T) {
	for _, c := range [][3]float64{
		{-2.0739627429606573, 0.15845897409040383, -0.05017858948852898},
		{2.555032585496654, 0.23820043730636456, -0.06568831172000554},
	} {
		k, a, b := c[0], c[1], c[2]
		best := 0.
		for e := -4.; e <= 1; e += 1e-4 {
			kp := math.Copysign(math.Pow(10, e), k)
			if b+kp*k <= 0 {
				continue
			}
			if pm, ok := pidSecondOrderPPM(k, a, b, kp); ok {
				best = math.Max(best, pm)
			}
		}
		if best < 15 || best > 30 {
			t.Fatalf("%v: oracle best margin %g", c, best)
		}
		r, err := TunePID(context.Background(), makePlant(t, []float64{k}, []float64{1, a, b}), PidtuneP, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("%v: %v", c, err)
		}
		ctl := *r.Controller
		if ctl.Ki != 0 || ctl.Kd != 0 || b+ctl.Kp*k <= 0 {
			t.Fatalf("%v: gains %+v unstable or not P", c, ctl)
		}
		pm, ok := pidSecondOrderPPM(k, a, b, ctl.Kp)
		if !ok || pm < best-.5 || math.Abs(r.Evidence.AchievedPhaseMargin-pm) > 1e-6 {
			t.Fatalf("%v: margin %g (evidence %g), oracle best %g", c, pm, r.Evidence.AchievedPhaseMargin, best)
		}
		g := func(w float64) complex128 { s := complex(0, w); return complex(k, 0) / (s*s + complex(a, 0)*s + complex(b, 0)) }
		if peak, at := pidDenseSensitivity(ctl, g, 1e-3, 1e2); peak > 4 {
			t.Fatalf("%v: |S| %g at ω=%g", c, peak, at)
		}
		if !pidHasWarning(r, "nearest 60°") {
			t.Fatalf("%v: no scan warning: %v", c, r.Evidence.Warnings)
		}
	}
}

// LV4BRD: the unit loop's 60° point of 34.78/(s³+2.45s²+20.47s+9.94) sits on
// the resonance (Kp 1.02, |S| ~8), and wc0's design fails too. Oracle: Routh
// on s³+a2s²+a1s+(a0+Kp·k), and wc* = the largest crossover whose
// Kp = 1/|G(jwc)| keeps dense |S| <= 4, by bisection; the half-octave scan
// keeps the margin nearest 60°, the highest passing wc, in (wc*/√2, wc*].
func TestTunePIDFreePResonantSixtyPointScansDown(t *testing.T) {
	k, a2, a1, a0 := 34.78138169542219, 2.4505445973941598, 20.469628091638796, 9.936912119476766
	g := func(w float64) complex128 {
		s := complex(0, w)
		return complex(k, 0) / (((s+complex(a2, 0))*s+complex(a1, 0))*s + complex(a0, 0))
	}
	feasible := func(wc float64) bool {
		peak, _ := pidDenseSensitivity(PID2{Kp: 1 / cmplx.Abs(g(wc))}, g, .01, 100)
		return peak <= 4
	}
	lo, hi := .5, 3.37
	if !feasible(lo) || feasible(hi) {
		t.Fatal("oracle bracket")
	}
	for range 40 {
		if m := math.Sqrt(lo * hi); feasible(m) {
			lo = m
		} else {
			hi = m
		}
	}
	r, err := TunePID(context.Background(), makePlant(t, []float64{k}, []float64{1, a2, a1, a0}), PidtuneP, PIDTuningOptions{})
	if err != nil {
		t.Fatal(err)
	}
	ctl := *r.Controller
	wc := r.Evidence.AchievedCrossover
	if wc <= lo/math.Sqrt2 || wc > lo*(1+1e-6) {
		t.Fatalf("crossover %g, want in (%g, %g]", wc, lo/math.Sqrt2, lo)
	}
	if kp := 1 / cmplx.Abs(g(wc)); ctl.Ki != 0 || ctl.Kd != 0 || math.Abs(ctl.Kp-kp) > 1e-6*kp {
		t.Fatalf("gains %+v, want Kp %g", ctl, kp)
	}
	if c0 := a0 + ctl.Kp*k; c0 <= 0 || a2*a1 <= c0 {
		t.Fatalf("Kp %g unstable by Routh", ctl.Kp)
	}
	if peak, at := pidDenseSensitivity(ctl, g, 1e-3, 1e2); peak > 4 {
		t.Fatalf("|S| %g at ω=%g", peak, at)
	}
	if !pidHasWarning(r, "nearest 60°") {
		t.Fatalf("no scan warning: %v", r.Evidence.Warnings)
	}
}

// Seeded k/(s²+as+b) with b < 0: by Routh a stabilizing P exists iff
// k·Kp > −b, and its best margin is the closed-form one maximised over Kp.
// Whenever a stable design with margin >= 20° exists (so |S| <= 2.9 at the
// crossover), TunePID must return a Routh-stable P design with dense
// |S| <= 4; base returned unattainable on most of them.
func TestTunePIDFreePUnstableSecondOrderSweep(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	checked, scanned := 0, 0
	for trial := range 100 {
		k := (2*rng.Float64() - 1) * 5
		a := math.Pow(10, -1.5+1.5*rng.Float64())
		b := -math.Pow(10, -2+2*rng.Float64())
		best := 0.
		for e := -4.; e <= 2; e += 1e-3 {
			kp := math.Copysign(math.Pow(10, e), k)
			if b+kp*k <= 0 {
				continue
			}
			if pm, ok := pidSecondOrderPPM(k, a, b, kp); ok {
				best = math.Max(best, pm)
			}
		}
		if best < 20 {
			continue
		}
		checked++
		r, err := TunePID(context.Background(), makePlant(t, []float64{k}, []float64{1, a, b}), PidtuneP, PIDTuningOptions{})
		if err != nil {
			t.Errorf("trial %d k=%g a=%g b=%g best PM %g: %v", trial, k, a, b, best, err)
			continue
		}
		if pidHasWarning(r, "nearest 60°") {
			scanned++
		}
		ctl := *r.Controller
		if ctl.Ki != 0 || ctl.Kd != 0 || b+ctl.Kp*k <= 0 {
			t.Errorf("trial %d: gains %+v unstable or not P", trial, ctl)
			continue
		}
		g := func(w float64) complex128 { s := complex(0, w); return complex(k, 0) / (s*s + complex(a, 0)*s + complex(b, 0)) }
		wc := r.Evidence.AchievedCrossover
		if peak, at := pidDenseSensitivity(ctl, g, wc/1e3, wc*1e3); peak > 4 {
			t.Errorf("trial %d: %+v |S| %g at ω=%g", trial, ctl, peak, at)
		}
	}
	if checked < 30 || scanned < 10 {
		t.Fatalf("checked %d, scanned %d", checked, scanned)
	}
	t.Logf("checked %d, scanned %d", checked, scanned)
}
