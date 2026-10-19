package controlsys

import (
	"context"
	"errors"
	"slices"
	"strings"
	"testing"
)

// D33HSG: G = 0.8296(s+1e-10)/(s²+0.999s+0.1257). Its zero sits on the
// IsStable tolerance, so every integral loop keeps a pole in (−1e-10, 0).
func TestTunePIDIntegralZeroOnBoundary(t *testing.T) {
	num, den := Poly{0.8296, 8.296e-11}, Poly{1, 0.999, 0.1257}
	plant := makePlant(t, num, den)
	z := -num[1] / num[0]
	for _, c := range []PID{{Kp: 8, Ki: 1}, {Kp: 8, Ki: 50}, {Kp: 1e3, Ki: 1e6}, {Kp: 5, Ki: 100, Kd: 1, Tf: .04}, {Ki: .5}} {
		dc, nc := Poly{1, 0}, Poly{c.Kd + c.Kp*c.Tf, c.Kp + c.Ki*c.Tf, c.Ki}
		if c.Tf != 0 {
			dc = Poly{c.Tf, 1, 0}
		}
		ch := dc.Mul(den).Add(nc.Mul(num))
		if real(ch.Eval(0)) <= 0 || real(ch.Eval(complex(z, 0))) >= 0 {
			t.Fatalf("%+v: characteristic %v has no sign change on [%g, 0]", c, ch, z)
		}
		cs, err := c.System()
		if err != nil {
			t.Fatal(err)
		}
		loop, err := Feedback(plant, cs, -1)
		if err != nil {
			t.Fatal(err)
		}
		if stable, err := loop.IsStable(); err != nil || stable {
			t.Fatalf("%+v: IsStable %v %v, want false", c, stable, err)
		}
	}
	for _, wc := range []float64{0, 1, 7.93} {
		for _, fam := range []PidtuneType{PidtuneI, PidtunePI, PidtunePIDF} {
			r, err := TunePID(context.Background(), plant, fam, PIDTuningOptions{CrossoverFrequency: wc})
			te, ok := errors.AsType[*PIDTuningError](err)
			if !ok {
				t.Errorf("wc %g %s: got %v %v, want *PIDTuningError", wc, fam, r, err)
				continue
			}
			if !slices.ContainsFunc(te.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "within the stability tolerance of s = 0") }) {
				t.Errorf("wc %g %s: warnings %q lack the boundary zero", wc, fam, te.Evidence.Warnings)
			}
			if wc == 0 && te.Evidence.Evaluations > 400 {
				t.Errorf("wc %g %s: %d evaluations, want no crossover ladder", wc, fam, te.Evidence.Evaluations)
			}
		}
	}
	if _, err := TunePID(context.Background(), plant, PidtunePI, PIDTuningOptions{MaxEvaluations: -1}); !errors.Is(err, ErrInvalidArgument) {
		t.Errorf("bad options: %v, want ErrInvalidArgument", err)
	}
}

func TestTunePIDIntegralZeroOffBoundary(t *testing.T) {
	plant := makePlant(t, []float64{0.8296, 8.296e-9}, []float64{1, 0.999, 0.1257})
	for _, fam := range []PidtuneType{PidtuneI, PidtunePI, PidtunePID, PidtunePIDF} {
		r, err := TunePID(context.Background(), plant, fam, PIDTuningOptions{})
		if err != nil {
			t.Errorf("%s: %v", fam, err)
			continue
		}
		c := r.Controller
		cs, err := (&PID{Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf}).System()
		if err != nil {
			if c.Kd != 0 && c.Tf == 0 {
				continue
			}
			t.Fatal(err)
		}
		loop, err := Feedback(plant, cs, -1)
		if err != nil {
			t.Fatal(err)
		}
		if stable, err := loop.IsStable(); err != nil || !stable {
			t.Errorf("%s: %+v closed loop IsStable %v %v", fam, *c, stable, err)
		}
	}
}
