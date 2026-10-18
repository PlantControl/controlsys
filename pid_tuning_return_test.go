package controlsys

import (
	"context"
	"errors"
	"testing"
)

// returnOracleStable reports whether the loop C·num/den with C = Kp + Ki/s +
// Kd·s/(Tf·s+1) has a Hurwitz characteristic.
func returnOracleStable(t *testing.T, kp, ki, kd, tf float64, num, den Poly) bool {
	t.Helper()
	stable, _ := pidtuneOracleStable(t, &PID{Kp: kp, Ki: ki, Kd: kd, Tf: tf}, num, den, 0, 0)
	return stable
}

// M5IIG3: the ideal-PID screen accepted Pidtune(−2.03/s)'s former design,
// whose characteristic (1−2.03·Kd)s² − 0.101s − 0.041 is Hurwitz with
// 1+L(∞) = 1−2.03·Kd ≈ −0.009. Oracle: the same gains with a derivative
// filter Tf have a root near −(1+L(∞))/Tf > 0; an ideal PID with
// 1+L(∞) > 0 and the same filters stays stable.
func TestPIDScreenRejectsFlippedReturn(t *testing.T) {
	num, den := Poly{-2.03}, Poly{1, 0}
	screen, err := newPIDStabilityScreen(makePlant(t, num, den), nil, 1)
	if err != nil {
		t.Fatal(err)
	}
	for _, c := range []struct {
		kp, ki, kd float64
		want       bool
	}{
		{0.04998, 0.02029, 0.49706, false},
		{-0.8, -0.5, -0.2, true},
	} {
		if !returnOracleStable(t, c.kp, c.ki, c.kd, 0, num, den) {
			t.Fatalf("precondition: ideal %+v not Hurwitz", c)
		}
		for _, tf := range []float64{1e-3, 1e-6} {
			if got := returnOracleStable(t, c.kp, c.ki, c.kd, tf, num, den); got != c.want {
				t.Fatalf("oracle: %+v with Tf=%g stable=%v", c, tf, got)
			}
		}
		got, err := screen.closedLoopPoles(&PID2{Kp: c.kp, Ki: c.ki, Kd: c.kd})
		if err != nil {
			t.Fatal(err)
		}
		if got != c.want {
			t.Fatalf("screen(%+v) = %v, want %v", c, got, c.want)
		}
	}
}

// M5IIG3: TunePID returned ideal-derivative designs with 1+L(∞) = 1+Kd·g∞,
// g∞ = lim s·G(s), in (0, ¼) — |S(∞)| above the |S| ≤ 4 bound its delayed
// loops keep — or negative, so that every filtered realization is unstable:
// 2/s PD at PM 30° gave 0.161 (now its 42° floor band), and
// (s+0.5)/(s²−s+4) PID at wc=1 gave −6.6 at PM 30° and −68 at 45°. Oracle:
// g∞ = num[0]/den[0] and characteristic roots with a derivative filter.
func TestTunePIDBoundsIdealReturn(t *testing.T) {
	for _, c := range []struct {
		num, den Poly
		family   PidtuneType
		wc, pm   float64
		wantErr  bool
	}{
		{Poly{2}, Poly{1, 0}, PidtunePD, 1, 30, false},
		{Poly{2}, Poly{1, 0}, PidtunePD, 0, 30, false},
		{Poly{1}, Poly{1, -1}, PidtunePD, 30, 30, false},
		{Poly{1, 0.5}, Poly{1, -1, 4}, PidtunePID, 1, 30, true},
		{Poly{1, 0.5}, Poly{1, -1, 4}, PidtunePID, 1, 45, true},
	} {
		r, err := TunePID(context.Background(), makePlant(t, c.num, c.den), c.family, PIDTuningOptions{CrossoverFrequency: c.wc, PhaseMargin: c.pm})
		if c.wantErr {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Fatalf("%v/%v %s wc=%g pm=%g: got %v, want unattainable", c.num, c.den, c.family, c.wc, c.pm, err)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%v/%v %s wc=%g pm=%g: %v", c.num, c.den, c.family, c.wc, c.pm, err)
		}
		k := r.Controller
		ret := 1 + k.Kd*c.num[0]/c.den[0]
		if k.Tf != 0 || ret < 0.26 {
			t.Fatalf("%v/%v %s wc=%g pm=%g: %+v has 1+L(∞) = %g", c.num, c.den, c.family, c.wc, c.pm, *k, ret)
		}
		if r.Evidence.AchievedPhaseMargin < c.pm-3 {
			t.Fatalf("%v/%v: margin %g below floor %g", c.num, c.den, r.Evidence.AchievedPhaseMargin, c.pm)
		}
		for _, tf := range []float64{1e-3, 1e-6} {
			if !returnOracleStable(t, k.Kp, k.Ki, k.Kd, tf/r.Evidence.AchievedCrossover, c.num, c.den) {
				t.Fatalf("%v/%v: %+v unstable with Tf=%g/wc", c.num, c.den, *k, tf)
			}
		}
	}
}

// M5IIG3: Pidtune at wc=0.1 on this unstable third-order plant returned
// Kd ≈ 3168 with 1+L(∞) = 1−0.732·Kd ≈ −2318. Oracle: the filtered former
// design is unstable; any returned ideal PID has 1+L(∞) > 0.
func TestPidtuneRejectsFlippedReturn(t *testing.T) {
	num := Poly{-0.732}.Mul(Poly{1, 0.108}).Mul(Poly{1, 0.617})
	den := Poly{1, -0.284}.Mul(Poly{1, -1.46, 0.73*0.73 + 8.35*8.35})
	if returnOracleStable(t, 15.696764448878886, 0.3139352889775777, 3168.129855140306, 1e-4, num, den) {
		t.Fatal("oracle: former design stable with a derivative filter")
	}
	pid, err := Pidtune(makePlant(t, num, den), PidtunePID, 0.1, nil)
	if err == nil && 1+pid.Kd*num[0] <= 0 {
		t.Fatalf("%+v has 1+L(∞) = %g", *pid, 1+pid.Kd*num[0])
	}
	if err != nil && !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatal(err)
	}
}
