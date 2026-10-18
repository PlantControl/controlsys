package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// A discrete plant with |G| < 1 everywhere and gain rising with frequency
// has no gain crossover and an infinite bandwidth, so the crossover is half
// the slowest pole's natural frequency |ln z|/Dt, capped a decade below
// Nyquist; it scales with 1/Dt. Pidtune's designs there are stable by the
// independent oracle.
func TestPlantCrossoverFreqDiscretePoleFallback(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		pole     complex128
	}{
		{Poly{0.1, -0.095}, Poly{1, -0.8}, 0.8},
		{Poly{0.1, -0.09}, Poly{1, 0.5}, -0.5},
	} {
		for _, dt := range []float64{0.1, 0.01} {
			p := pidtuneBackoffPlant(t, c.num, c.den, dt, 0)
			want := math.Min(0.5*cmplx.Abs(cmplx.Log(c.pole))/dt, 0.1*math.Pi/dt)
			wc, err := plantCrossoverFreq(p)
			if err != nil || math.Abs(wc-want) > 1e-9*want {
				t.Fatalf("%v/%v dt=%g: crossover %g, %v; want %g", c.num, c.den, dt, wc, err, want)
			}
			for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePID, PidtunePIDF} {
				pid, err := Pidtune(p, f, 0, nil)
				if err != nil {
					if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
						t.Errorf("%v/%v dt=%g %s: %v", c.num, c.den, dt, f, err)
					}
					continue
				}
				if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, dt, 0); ok && !stable {
					t.Errorf("%v/%v dt=%g %s: %+v unstable", c.num, c.den, dt, f, pid)
				}
			}
		}
	}
}

// High-pass discrete plants (|G| < 1, gain rising to z = −1) take the
// crossover from their slowest pole near z = −1, about a decade below
// Nyquist, where no I design is stable; the first stable one lies more than
// three decades lower, so the back-off reaches six (TFE7I2 review). Oracle:
// closed-loop roots, signed margin at every crossover and |S| of the
// directly evaluated loop; Ki matches the design before TFE7I2.
func TestPidtunePoleFallbackBacksOffSixDecades(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		dt, ki   float64
	}{
		{Poly{-0.04189665061210819, 0.07677735137479592, -0.03510307300396654}, Poly{1, 0.9371514351204928, 0.13383773145948022}, 0.0757765515708101, -23.07},
		{Poly{-0.011302534772155698, 0.019672327226871213, -0.008550308672972825}, Poly{1, 1.3218336193087052, 0.4064581673333586}, 0.6967602637285988, -4.127},
		{Poly{0.02168291613348848, -0.040003288616014994, 0.018372574831365913}, Poly{1, 1.0961478669592766, 0.30038125589712455}, 0.06586027470121651, 56.0},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		if _, fromPole, err := autoCrossoverFreq(p, 0, 60); err != nil || !fromPole {
			t.Fatalf("%v/%v: pole fallback %v, %v", c.num, c.den, fromPole, err)
		}
		pid, err := Pidtune(p, PidtuneI, 0, nil)
		if err != nil {
			t.Errorf("%v/%v: %v", c.num, c.den, err)
			continue
		}
		if r := pid.Ki / c.ki; !(r > 1/1.5 && r < 1.5) {
			t.Errorf("%v/%v: Ki %g, want about %g", c.num, c.den, pid.Ki, c.ki)
		}
		if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, 0); !ok || !stable {
			t.Errorf("%v/%v: %+v unstable", c.num, c.den, pid)
		}
		pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0), 1e-8, math.Pi/c.dt)
		if len(pms) == 0 || ms > 4.5 {
			t.Errorf("%v/%v: %+v crossovers %v, |S| peak %.3g", c.num, c.den, pid, pms, ms)
		}
		for _, m := range pms {
			if m < 58.5 {
				t.Errorf("%v/%v: %+v phase margin %.3g°", c.num, c.den, pid, m)
			}
		}
	}
}
