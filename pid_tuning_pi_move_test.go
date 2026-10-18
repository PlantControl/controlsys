package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"strings"
	"testing"
)

// A default PI design needing lead at the plant crossover moves the
// crossover to where PI lag can meet the margin. Stability is checked by
// Routh-Hurwitz on the closed-form characteristic polynomial, margins by
// direct evaluation of L(jω) on a dense sweep.
func TestTunePIDPIMovedCrossover(t *testing.T) {
	g := func(num, den []float64) func(complex128) complex128 {
		return func(s complex128) complex128 {
			var n, d complex128
			for _, c := range num {
				n = n*s + complex(c, 0)
			}
			for _, c := range den {
				d = d*s + complex(c, 0)
			}
			return n / d
		}
	}
	for _, c := range []struct {
		name     string
		num, den []float64
		stable   func(kp, ki float64) bool
	}{
		// s³+s²+Kp·s+Ki
		{"1/(s(s+1))", []float64{1}, []float64{1, 1, 0}, func(kp, ki float64) bool { return ki > 0 && kp > ki }},
		{"-1/(s(s+1))", []float64{-1}, []float64{1, 1, 0}, func(kp, ki float64) bool { return -ki > 0 && -kp > -ki }},
		// s²+(Kp−1)s+Ki
		{"1/(s-1)", []float64{1}, []float64{1, -1}, func(kp, ki float64) bool { return kp > 1 && ki > 0 }},
		{"-1/(s-1)", []float64{-1}, []float64{1, -1}, func(kp, ki float64) bool { return -kp > 1 && -ki > 0 }},
	} {
		t.Run(c.name, func(t *testing.T) {
			p := makePlant(t, c.num, c.den)
			r, err := TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{})
			if err != nil {
				t.Fatal(err)
			}
			k := r.Controller
			if k.Kd != 0 || k.Ki == 0 {
				t.Fatalf("not a PI design: %+v", *k)
			}
			if !c.stable(k.Kp, k.Ki) {
				t.Fatalf("closed loop unstable: Kp=%g Ki=%g", k.Kp, k.Ki)
			}
			moved := false
			for _, w := range r.Evidence.Warnings {
				moved = moved || strings.Contains(w, "moved to")
			}
			if !moved {
				t.Fatalf("no moved-crossover warning: %q", r.Evidence.Warnings)
			}
			plant := g(c.num, c.den)
			loop := func(w float64) complex128 {
				s := complex(0, w)
				return plant(s) * (complex(k.Kp, 0) + complex(k.Ki, 0)/s)
			}
			wc := r.Evidence.AchievedCrossover
			if l := cmplx.Abs(loop(wc)); math.Abs(l-1) > 1e-9 {
				t.Fatalf("|L(j%g)| = %g", wc, l)
			}
			pm := 180 + cmplx.Phase(loop(wc))*180/math.Pi
			if math.Abs(pm-60) > 3+1e-9 {
				t.Fatalf("margin %g° at %g", pm, wc)
			}
			prev := cmplx.Abs(loop(1e-4*wc)) - 1
			for i := 1; i <= 20000; i++ {
				w := 1e-4 * wc * math.Pow(1e8, float64(i)/20000)
				d := cmplx.Abs(loop(w)) - 1
				if math.Signbit(d) != math.Signbit(prev) && math.Abs(math.Log(w/wc)) > 1e-3 {
					t.Fatalf("extra 0 dB crossing near %g", w)
				}
				prev = d
			}
		})
	}
}

// The I family stays unattainable on these plants: s³+s²+Ki and s²−s+Ki
// have no stabilizing Ki.
func TestTunePIDIUnstabilizable(t *testing.T) {
	for _, den := range [][]float64{{1, 1, 0}, {1, -1}} {
		_, err := TunePID(context.Background(), makePlant(t, []float64{1}, den), PidtuneI, PIDTuningOptions{})
		if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Fatalf("den %v: err %v", den, err)
		}
	}
}
