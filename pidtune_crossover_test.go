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
