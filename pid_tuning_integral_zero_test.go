package controlsys

import (
	"context"
	"errors"
	"math"
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
		for _, fam := range []PidtuneType{PidtuneI, PidtunePI, PidtunePID, PidtunePIDF} {
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

// OI7P4M: TunePID returned this ideal PID for the D33HSG plant; its slow
// closed-loop pole −a0/a1 sits inside the IsStable band.
func TestPIDTuningIdealStableBand(t *testing.T) {
	c := &PID2{Kp: 7.954440137905823, Ki: 121.18255736538495, Kd: 1.2450662060726023, B: 1, C: 1}
	for _, tc := range []struct {
		z    float64
		want bool
	}{{1e-10, false}, {1e-8, true}} {
		num, den := Poly{0.8296, 0.8296 * tc.z}, Poly{1, 0.999, 0.1257}
		ch := Poly{1, 0}.Mul(den).Add(Poly{c.Kd, c.Kp, c.Ki}.Mul(num))
		slow := -ch[3] / ch[2]
		if slow >= 0 || (slow > -1e-10) == tc.want || math.Abs(real(ch.Eval(complex(slow, 0)))) > 1e-6*ch[3] {
			t.Fatalf("z %g: characteristic %v slow root %g", tc.z, ch, slow)
		}
		plant := makePlant(t, num, den)
		tf, err := plant.rationalTransferFunction(nil)
		if err != nil {
			t.Fatal(err)
		}
		if got, err := pidTuningIdealStable(plant, func() (*TransferFuncResult, error) { return tf, nil }, c, 0); err != nil || got != tc.want {
			t.Errorf("z %g: pidTuningIdealStable %v %v, want %v (slow root %g)", tc.z, got, err, tc.want, slow)
		}
	}
}

func TestPIDTuningIntegralZeroSign(t *testing.T) {
	den := Poly{1, 2, 1}
	if _, ok := pidTuningIntegralZero(makePlant(t, Poly{1, 1e-10}, den), PidtunePID); !ok {
		t.Error("zero -1e-10: PID not blocked")
	}
	rhp := makePlant(t, Poly{-1, 1e-10}, den)
	for _, fam := range []PidtuneType{PidtuneI, PidtunePI, PidtunePID, PidtunePIDF} {
		if z, ok := pidTuningIntegralZero(rhp, fam); ok {
			t.Errorf("zero +1e-10 %s: blocked at %g", fam, z)
		}
	}
	ch := Poly{1, 0}.Mul(den).Add(Poly{-0.9, 0.9e-10})
	if slow := -ch[3] / ch[2]; math.Abs(slow+9e-10) > 1e-12 {
		t.Fatalf("I Ki 0.9: slow root %g, want -9e-10", slow)
	}
	cs, err := (&PID{Ki: 0.9}).System()
	if err != nil {
		t.Fatal(err)
	}
	loop, err := Feedback(rhp, cs, -1)
	if err != nil {
		t.Fatal(err)
	}
	if stable, err := loop.IsStable(); err != nil || !stable {
		t.Errorf("I Ki 0.9 on zero +1e-10: IsStable %v %v, want true", stable, err)
	}
}

func TestPIDTuningIntegralZeroRHPInsideBand(t *testing.T) {
	plant := makePlant(t, Poly{1, -5e-11}, Poly{1, 1, 0.25})
	if z, ok := pidTuningIntegralZero(plant, PidtunePI); ok {
		t.Errorf("zero +5e-11: blocked at %g", z)
	}
	for _, n := range []Poly{{1, 0}, {2, 3, 0}} {
		if _, ok := pidTuningIntegralZero(makePlant(t, n, Poly{1, 3, 3, 1}), PidtunePI); !ok {
			t.Errorf("num %v: zero at 0 not blocked", n)
		}
	}
	a := Poly{1, 1, 0.25, 0}.Add(Poly{-0.2499, 0.2499 * 5e-11})
	if len(a) != 4 || a[0] <= 0 || a[1] <= 0 || a[2] <= 0 || a[3] <= 0 || a[1]*a[2] <= a[0]*a[3] {
		t.Fatalf("I Ki -0.2499: characteristic %v not Hurwitz", a)
	}
	cs, err := (&PID{Ki: -0.2499}).System()
	if err != nil {
		t.Fatal(err)
	}
	loop, err := Feedback(plant, cs, -1)
	if err != nil {
		t.Fatal(err)
	}
	if stable, err := loop.IsStable(); err != nil || !stable {
		t.Errorf("I Ki -0.2499 on zero +5e-11: IsStable %v %v, want true", stable, err)
	}
}
