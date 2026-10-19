package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"strings"
	"testing"
)

// 26T5XY: on biproper pole-fallback plants a back-off step accepted PID Kp
// 2.67e7 (|L(j∞)| 2.67e4) and PDF Kp 5.3e8, Kd 1.7e13 (|L(j∞)| 8e5), whose
// huge level keeps |S| small and the margins met. A back-off design leveling
// off above max(10, 100·L0), L0 the first design's level, is rejected; PD and
// PIDF, leveling off near 3 and 9, still return. Oracles: |L(j∞)| =
// |C(∞)|·num[0]/den[0] from the coefficients, closed-loop roots, margins and
// |S| of the directly evaluated loop.
func TestPidtuneBackoffLevelCapped(t *testing.T) {
	for _, den := range []Poly{{1, 0.01, 1}, {1, 0.1, 1}} {
		num := Poly{0.001, 1e-9}.Mul(Poly{1, 1})
		p := pidtuneBackoffPlant(t, num, den, 0, 0)
		if _, src, err := autoCrossoverSource(p, 0, 60); err != nil || src != pidtuneCrossPole {
			t.Fatalf("%v/%v: crossover source %v, %v", num, den, src, err)
		}
		for _, f := range []PidtuneType{PidtunePID, PidtunePDF} {
			name := fmt.Sprintf("%s %v/%v", f, num, den)
			pid, err := Pidtune(p, f, 0, nil)
			if err == nil {
				t.Errorf("%s: %+v returned, |L(j∞)| %.3g", name, pid, pidtuneLevelOracle(pid, num, den))
				continue
			}
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) || !strings.Contains(err.Error(), "specify wc") {
				t.Errorf("%s: %v", name, err)
			}
		}
		for _, f := range []PidtuneType{PidtunePD, PidtunePIDF} {
			name := fmt.Sprintf("%s %v/%v", f, num, den)
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				t.Errorf("%s: %v", name, err)
				continue
			}
			if l := pidtuneLevelOracle(pid, num, den); !(l <= pidtuneLevelCap) {
				t.Errorf("%s: %+v |L(j∞)| %.3g", name, pid, l)
			}
			pidtuneCheckRobust(t, name, pid, num, den, 0, 4.5)
		}
	}
}

func pidtuneLevelOracle(pid *PID, num, den Poly) float64 {
	k := pid.Kp
	if pid.Kd != 0 {
		if !(pid.Tf > 0) {
			return math.Inf(1)
		}
		k += pid.Kd / pid.Tf
	}
	return math.Abs(k * num[0] / den[0])
}

// PBXZCH: the automatic PD placed phase and gain on the ideal derivative,
// so |L(wc)| ≠ 1 (0.49 on k·s/(s+a)), where MATLAB crosses over at wc. It
// is now placed on the Td/10 response (placedPD), the legacy pipeline
// rerun only when every placed step is unattainable. Over a seeded
// population of stable, unstable, discrete, zero-DC and delayed plants,
// every return is oracle-stable (closed-loop roots, or the Nyquist winding
// with a delay) and has |L| = 1 within 1e-6 at a back-off step
// wc·10^(−k/20), on its realized response (placed, |S| ≤ 4 as the gate)
// or on its ideal derivative (legacy); most are placed.
func TestPidtuneAutoPDPlaced(t *testing.T) {
	rng := rand.New(rand.NewPCG(11, 13))
	placed, legacy := 0, 0
	for trial := range 600 {
		num, den, dt, delay := pidtuneAutoPDPlant(rng, trial)
		p := pidtuneBackoffPlant(t, num, den, dt, delay)
		name := fmt.Sprintf("trial %d %v/%v dt=%g delay=%g", trial, num, den, dt, delay)
		pid, err := Pidtune(p, PidtunePD, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		if stable, ok := pidtuneOracleStable(t, pid, num, den, dt, delay); ok && !stable {
			t.Errorf("%s: %+v unstable", name, pid)
		}
		wc, _, err := autoCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		ideal := *pid
		ideal.Tf = 0
		L, Li := pidtuneOracleLoop(pid, num, den, dt, delay), pidtuneOracleLoop(&ideal, num, den, dt, delay)
		kind := ""
		for k := 0; k <= 20*pidtuneBackoffMaxDecade && kind == ""; k++ {
			w := wc * math.Pow(10, -float64(k)/pidtuneBackoffPerDecade)
			switch {
			case math.Abs(cmplx.Abs(L(w))-1) < 1e-6:
				kind = "placed"
			case math.Abs(cmplx.Abs(Li(w))-1) < 1e-6:
				kind = "legacy"
			}
		}
		switch kind {
		case "placed":
			placed++
			wMax := 1e6
			if dt > 0 {
				wMax = math.Pi / dt
			}
			if _, ms := pidtuneSignedMargins(L, 1e-7, wMax); ms > pidTuningMaxSensitivity+0.05 {
				t.Errorf("%s: placed %+v |S| peak %.3g", name, pid, ms)
			}
		case "legacy":
			legacy++
		default:
			t.Errorf("%s: %+v has |L| ≠ 1 at every back-off step", name, pid)
		}
	}
	t.Logf("placed %d, legacy %d", placed, legacy)
	if placed < 300 || legacy == 0 || placed < 4*legacy {
		t.Errorf("placed %d, legacy %d", placed, legacy)
	}
}

func pidtuneAutoPDPlant(rng *rand.Rand, trial int) (num, den Poly, dt, delay float64) {
	n := 1 + trial%4
	if trial%3 == 0 {
		dt = 0.05 + 0.2*rng.Float64()
	}
	den = Poly{1}
	for range n {
		if dt == 0 {
			p := 0.05 + 4*rng.Float64()
			if rng.Float64() < 0.1 {
				p = -p
			}
			den = den.Mul(Poly{1, p})
		} else {
			den = den.Mul(Poly{1, 0.95 - 1.9*rng.Float64()})
		}
	}
	num = Poly{2*rng.Float64() - 1}
	if trial%2 == 0 {
		zero := Poly{1, 0}
		if dt > 0 {
			zero = Poly{1, -1}
		}
		for range 1 + rng.IntN(n) {
			num = num.Mul(zero)
		}
	} else {
		for range rng.IntN(n) {
			num = num.Mul(Poly{1, 2*rng.Float64() - 1})
		}
	}
	if trial%5 == 1 {
		delay = rng.Float64()
		if dt > 0 {
			delay = float64(1 + rng.IntN(3))
		}
	}
	return num, den, dt, delay
}
