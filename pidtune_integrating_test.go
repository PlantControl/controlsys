package controlsys

import (
	"context"
	"errors"
	"math"
	"testing"
)

// Integrating plants whose gain stays above 1 have no gain crossover and an
// infinite DC gain, so Bandwidth is undefined; the automatic crossover comes
// from their other dynamics and every returned design is checked by the
// independent closed-loop oracles.
func TestPidtuneIntegratingNoCrossover(t *testing.T) {
	for _, c := range []struct {
		name     string
		num, den Poly
		dt, wc   float64
	}{
		// |G| = k/(2 sin(ωDt/2)) ≥ 1.42: no dynamics but the integrator, a
		// decade below Nyquist.
		{"discrete", Poly{2.834699423337496}, Poly{1, -1}, 0.02298944503089565, 0.1 * math.Pi / 0.02298944503089565},
		// |G| → 5.34 at high frequency: half the zero's 4.32 rad/s.
		{"continuous", Poly{5.340957844707375, 23.09187463052385}, Poly{1, 0}, 0, 0.5 * 23.09187463052385 / 5.340957844707375},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		wc, err := findCrossoverFreq(p, 0, 60)
		if err != nil || math.Abs(wc-c.wc) > 1e-12*c.wc {
			t.Fatalf("%s: crossover %g, %v; want %g", c.name, wc, err, c.wc)
		}
		wMax := 1e3 * wc
		if c.dt > 0 {
			wMax = math.Pi / c.dt
		}
		check := func(api string, f PidtuneType, pid *PID) {
			t.Helper()
			stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, 0)
			if !ok || !stable {
				t.Errorf("%s %s %s: %+v unstable (ok %v)", c.name, api, f, pid, ok)
			}
			if c.dt > 0 && !pidtuneExactSchurStable(t, pid, c.num, c.den, c.dt) {
				t.Errorf("%s %s %s: %+v not Schur stable", c.name, api, f, pid)
			}
			_, ms := pidtuneLoopMarginBand(pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0), wc*1e-4, wMax, 1.0002)
			if ms > pidTuningMaxSensitivity*1.01 {
				t.Errorf("%s %s %s: %+v peak |S| %.3g", c.name, api, f, pid, ms)
			}
		}
		for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF} {
			need := f == PidtuneP || f == PidtunePI || f == PidtunePID
			pid, err := Pidtune(p, f, 0, nil)
			switch {
			case err == nil:
				check("Pidtune", f, pid)
				if f == PidtuneP || f == PidtuneI || f == PidtunePD || f == PidtunePDF {
					break
				}
				pm, _ := pidtuneLoopMarginBand(pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0), wc*1e-4, wMax, 1.0002)
				if math.Abs(pm-60) > 0.5 {
					t.Errorf("%s Pidtune %s: phase margin %.4g°, want 60°", c.name, f, pm)
				}
			case need || !errors.Is(err, ErrPIDTuningTargetUnattainable):
				t.Errorf("%s Pidtune %s: %v", c.name, f, err)
			}
			res, err := TunePID(context.Background(), p, f, PIDTuningOptions{})
			switch {
			case err == nil:
				k := res.Controller
				check("TunePID", f, &PID{IFormula: k.IFormula, DFormula: k.DFormula, Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt})
			case need || !errors.Is(err, ErrPIDTuningTargetUnattainable):
				t.Errorf("%s TunePID %s: %v", c.name, f, err)
			}
		}
	}
}

// A triple integrator realized from its polynomial splits by about
// eps^(1/3) ≈ 6e-6 around z = 1; the split poles are integrators, so the
// crossover comes from the zero at z = 0.9.
func TestPidtuneIntegratingCrossoverSplitIntegrators(t *testing.T) {
	const dt = 0.05
	p := pidtuneBackoffPlant(t, Poly{9, -8.1}, Poly{1, -3, 3, -1}, dt, 0)
	poles, err := p.Poles()
	if err != nil {
		t.Fatal(err)
	}
	split := 0.0
	for _, z := range poles {
		split = math.Max(split, math.Abs(real(z)-1)+math.Abs(imag(z)))
	}
	if split == 0 {
		t.Fatalf("poles %v: expected rounding to split the integrators", poles)
	}
	wc, err := plantCrossoverFreq(p)
	if want := 0.5 * math.Log(1/0.9) / dt; err != nil || math.Abs(wc-want) > 1e-6*want {
		t.Fatalf("crossover %g, %v; want %g", wc, err, want)
	}
}
