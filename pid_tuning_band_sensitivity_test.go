package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// GMED3W: the |S| bound scanned [wc/100, max(wc·100, minHigh)] while a
// floor design reported a band up to 0.99·Nyquist, so PDF on u248 returned
// |S| 14.5 at 3.11 rad/s inside its band, PID on u198 |S| 14.2 and PDF on
// u96 |S| 6.6; with a sample of delay PDF on u248 kept |S| 7.0, and PD on
// a 125-sample delayed plant |S| 7.7. Every design now keeps |S| <= 4 over
// its reported band. The lag78 row's reported |S| 4.6 design had mixed-sign
// gains, which 6KBA3T removed; it stays as a guard. u96's band design kept
// |S(z = −1)| 4.16 past the band top; one-sign PDF designs crossing 0 dB
// with PM >= 45 all do on a Kp, Kd, Tf grid (best 4.38), so it now fails
// (2FQ6DJ). Oracle: the loop evaluated directly from the coefficients, the
// undelayed plant in 256-bit arithmetic, |S| at z = −1, and exact
// Schur–Cohn stability.
func TestTunePIDBandSensitivity(t *testing.T) {
	t.Parallel()
	skipSlowUnderRace(t)
	for _, c := range []struct {
		name     string
		num, den Poly
		dt, pm   float64
		lag      int
		fam      PidtuneType
		fails    bool
	}{
		{"u248", Poly{-1.6135533480124533, 4.580788382309392, -3.042109902683129}, Poly{1, -0.7430173634840683, -0.24120328804525268, -0.0146531797779481}, 1, 0, 0, PidtunePDF, false},
		{"u248 lag1", Poly{-1.6135533480124533, 4.580788382309392, -3.042109902683129}, Poly{1, -0.7430173634840683, -0.24120328804525268, -0.0146531797779481}, 1, 0, 1, PidtunePDF, false},
		{"u198", Poly{6.139952237001286, -8.24591813622576}, Poly{1, -1.7830459398889493, 0.7834493375074997}, 1, 30, 0, PidtunePID, false},
		{"u96", Poly{0.04789840298238053, -0.048996211490499125, 0.012223576551756373}, Poly{1, -0.09376161947657424, -0.7314444748472427, -0.13744625517591483}, .1, 45, 0, PidtunePDF, true},
		{"lag125", Poly{0.026460423781941782, -0.045975804457780156, 0.02267314440850615}, Poly{1, -0.8256026091672339, -0.17211118956233257}, .01, 43.5, 125, PidtunePD, false},
		{"lag78", Poly{0.001167543278169746, -0.002183061770214697, 0.0011554331334533623}, Poly{1, -1.8438882236394458, 0.8438882236394458}, .01, 0, 78, PidtunePD, false},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, float64(c.lag))
		r, err := TunePID(context.Background(), p, c.fam, PIDTuningOptions{PhaseMargin: c.pm})
		if c.fails {
			if _, ok := errors.AsType[*PIDTuningError](err); !ok {
				t.Errorf("%s: want a tuning error, got %v", c.name, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := r.Controller
		pid := &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt, IFormula: k.IFormula, DFormula: k.DFormula}
		if !pidtuneExactSchurStableLag(t, pid, c.num, c.den, c.dt, 1, c.lag) {
			t.Errorf("%s: %+v unstable", c.name, *k)
		}
		undelayed := pidtuneUndelayedLoop(k, c.num, c.den, c.dt)
		L := func(w float64) complex128 { return undelayed(w) * cmplx.Exp(complex(0, -w*c.dt*float64(c.lag))) }
		band := r.Evidence.FrequencyBand
		_, ms := pidtuneLoopMarginBand(L, band[0], band[1], 1.00005)
		ms = math.Max(ms, 1/cmplx.Abs(1+L(math.Pi/c.dt)))
		if ms > pidTuningMaxSensitivity*1.01 {
			t.Errorf("%s: %+v |S| %.4g over band %v", c.name, *k, ms, band)
		}
	}
}
