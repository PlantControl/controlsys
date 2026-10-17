package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

// NL37XL: a PDF whose crossover needs phase lag gets the least lead
// (0.05 rad) with a small Kd, as MATLAB pidtune returns, instead of
// ErrPIDTuningTargetUnattainable; an explicit margin still errors.
// Oracle: C and G evaluated from their coefficients, characteristic roots,
// dense margin sweep.
func TestPidtunePDFLagFloor(t *testing.T) {
	cases := []struct {
		num, den Poly
		dt       float64
	}{
		{Poly{0.1}, Poly{1, -0.9}, 0.1},
		{Poly{0.01, 0.009}, Poly{1, -1.7, 0.72}, 0.1},
		{Poly{1}, Poly{1, 1}, 0},
	}
	for _, c := range cases {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		auto, err := findCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		for _, wc := range []float64{0, 0.5, 2} {
			name := fmt.Sprintf("%v/%v dt=%g wc=%g", c.num, c.den, c.dt, wc)
			pid, err := Pidtune(p, PidtunePDF, wc, nil)
			if err != nil {
				t.Fatalf("%s: %v", name, err)
			}
			w := wc
			if w == 0 {
				w = auto
			}
			x := complex(0, w)
			if c.dt > 0 {
				x = cmplx.Exp(complex(0, w*c.dt))
			}
			cw := pidtuneControllerOracle(pid, w)
			l := cw * c.num.Eval(x) / c.den.Eval(x)
			if pid.Kp <= 0 || pid.Kd <= 0 || pid.Kd*w > 0.2*pid.Kp || pid.Ki != 0 {
				t.Errorf("%s: %+v, want small positive Kd", name, pid)
			}
			if math.Abs(cmplx.Abs(l)-1) > 1e-9 || math.Abs(cmplx.Phase(cw)-0.05) > 1e-9 {
				t.Errorf("%s: |L|=%.12g arg C=%.12g, want 1 and %g", name, cmplx.Abs(l), cmplx.Phase(cw), 0.05)
			}
			if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, 0); !ok || !stable {
				t.Errorf("%s: %+v unstable", name, pid)
			}
			hi := w * 1e3
			if c.dt > 0 {
				hi = math.Pi / c.dt
			}
			if pm, _ := pidtuneLoopMarginBand(pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0), w*1e-4, hi, 1.001); pm < 60 {
				t.Errorf("%s: margin %g, want ≥ 60", name, pm)
			}
			if wc == 0 {
				continue
			}
			if _, err := Pidtune(p, PidtunePDF, wc, &PidtuneOptions{PhaseMargin: 60}); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s explicit PM: err %v, want unattainable", name, err)
			}
		}
	}
}

// NL37XL: under an automatic crossover the lag-floor PDF passes the back-off
// gate. At this plant's automatic crossover the floored design has Ms > 9;
// Pidtune lowers the crossover to a design keeping |S| ≤ 4 and the margin.
func TestPidtunePDFLagFloorGated(t *testing.T) {
	num, den := Poly{0.20869270674360085, 0.5771080007526925}, Poly{1, 0.031493049125030315, -0.5971318727231876, -0.10709011948103661}
	dt := 0.05642002314620125
	p := pidtuneBackoffPlant(t, num, den, dt, 0)
	wc, err := findCrossoverFreq(p, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := Pidtune(p, PidtunePDF, wc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, ms := pidtuneLoopMarginBand(pidtuneOracleLoop(raw, num, den, dt, 0), wc*1e-6, math.Pi/dt, 1.001); ms < 8 {
		t.Fatalf("floored design at the automatic crossover has Ms %g; the gate is untested", ms)
	}
	pid, err := Pidtune(p, PidtunePDF, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if pid.Kp <= 0 || pid.Kd <= 0 {
		t.Fatalf("%+v, want positive PDF", pid)
	}
	if stable, ok := pidtuneOracleStable(t, pid, num, den, dt, 0); !ok || !stable {
		t.Fatalf("%+v unstable", pid)
	}
	pidtuneDenseRobust(t, "gated lag floor", pid, num, den, dt, 0, wc, 60)
}
