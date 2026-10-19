package controlsys

import (
	"errors"
	"fmt"
	"math"
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
