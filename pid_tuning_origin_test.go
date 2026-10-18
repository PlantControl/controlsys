package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"testing"
)

func originZeroPlant(t *testing.T, sign, eps, tau float64) *System {
	t.Helper()
	p := makePlant(t, []float64{sign, 0}, []float64{1, 1 + eps, eps})
	if tau > 0 {
		if err := p.SetInputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

// An integrator cancelling the plant zero at s = 0 leaves a closed-loop pole
// at the origin that the delay Nyquist test cannot see (G4YFOH).
func TestPIDTuningDelayOriginZeroRejected(t *testing.T) {
	for _, eps := range []float64{1e-8, 1e-7} {
		for _, sign := range []float64{1, -1} {
			p := originZeroPlant(t, sign, eps, .5)
			for _, tc := range []struct {
				fam PidtuneType
				wc  float64
			}{{PidtunePI, 1}, {PidtunePID, 1}, {PidtunePIDF, 1}, {PidtunePIDF, 3}, {PidtunePI, 3}, {PidtunePID, 3}} {
				t.Run(fmt.Sprintf("eps=%g/sign=%g/%s/wc=%g", eps, sign, tc.fam, tc.wc), func(t *testing.T) {
					r, err := TunePID(context.Background(), p, tc.fam, PIDTuningOptions{CrossoverFrequency: tc.wc})
					if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
						t.Errorf("TunePID = %+v, %v; want unattainable", r, err)
					}
					c, err := Pidtune(p, tc.fam, tc.wc, nil)
					if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
						t.Errorf("Pidtune = %+v, %v; want unattainable", c, err)
					}
				})
			}
		}
	}
}

// Independent oracle: Padé-approximated r→u loop of the controller the screen
// used to accept has a pole at the origin, for either plant sign.
func TestPIDTuningDelayOriginZeroOracle(t *testing.T) {
	for _, sign := range []float64{1, -1} {
		p := originZeroPlant(t, sign, 1e-8, .5)
		q, err := p.Pade(6)
		if err != nil {
			t.Fatal(err)
		}
		cs, err := mustPID(t, sign*1.0283391459757736, sign*0.9708339718272201, 0, 0, 0).System()
		if err != nil {
			t.Fatal(err)
		}
		ru, err := Feedback(cs, q, -1)
		if err != nil {
			t.Fatal(err)
		}
		poles, err := ru.Poles()
		if err != nil {
			t.Fatal(err)
		}
		least := 1.0
		for _, s := range poles {
			least = min(least, cmplx.Abs(s))
		}
		if least > 1e-9 {
			t.Fatalf("sign %g: r→u poles %v; want one at the origin", sign, poles)
		}
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		stable, ok, err := pidTuningDelayStability(p, eval, 1, false, 1)
		if err != nil || !ok {
			t.Fatal(ok, err)
		}
		if got, err := stable(&PID2{Kp: sign * 1.0283391459757736, Ki: sign * 0.9708339718272201, B: 1, C: 1}); got || err != nil {
			t.Fatalf("sign %g: screen = %v, %v; want rejected", sign, got, err)
		}
	}
}

func TestPIDTuningLeastZero(t *testing.T) {
	for _, tc := range []struct {
		num, den []float64
		lo, hi   float64
	}{
		{[]float64{1, 0}, []float64{1, 1 + 1e-8, 1e-8}, 0, 1e-12},
		{[]float64{1, 0}, []float64{1, 2, 1}, 0, 1e-12},
		{[]float64{1, 1e-6}, []float64{1, 2, 1}, .99e-6, 1.01e-6},
		{[]float64{1, 1e-4}, []float64{1, 1e6 + 1, 1e6}, .99e-4, 1.01e-4},
		{[]float64{1}, []float64{1, 1}, math.Inf(1), math.Inf(1)},
	} {
		if got := pidTuningLeastZero(makePlant(t, tc.num, tc.den)); !(got >= tc.lo && got <= tc.hi) {
			t.Errorf("%v/%v: got %g want [%g, %g]", tc.num, tc.den, got, tc.lo, tc.hi)
		}
	}
}

// A zero off the origin still tunes with Nyquist evidence.
func TestPIDTuningDelayNearOriginZeroTunes(t *testing.T) {
	p := makePlant(t, []float64{1, .1}, []float64{1, 1.01, .01})
	if err := p.SetInputDelay([]float64{.5}); err != nil {
		t.Fatal(err)
	}
	r, err := TunePID(context.Background(), p, PidtunePI, PIDTuningOptions{CrossoverFrequency: 1})
	if err != nil {
		t.Fatal(err)
	}
	if r.Evidence.Stability != PIDStabilityNyquist {
		t.Fatalf("stability evidence %q", r.Evidence.Stability)
	}
	if _, err := Pidtune(p, PidtunePI, 1, nil); err != nil {
		t.Fatal(err)
	}
}

// A slow zero under a fast pole is not cancelled; the spread must not trip
// the origin test.
func TestPIDTuningDelaySlowZeroFastPoleTunes(t *testing.T) {
	p := makePlant(t, []float64{1, 1e-4}, []float64{1, 1e6 + 1, 1e6})
	if err := p.SetInputDelay([]float64{.5}); err != nil {
		t.Fatal(err)
	}
	eval, err := newSISOEval(p)
	if err != nil {
		t.Fatal(err)
	}
	stable, ok, err := pidTuningDelayStability(p, eval, 1, false, 1)
	if err != nil || !ok {
		t.Fatal(ok, err)
	}
	if got, err := stable(&PID2{Kp: 1e5, Ki: 1e5, B: 1, C: 1}); !got || err != nil {
		t.Fatalf("screen = %v, %v; want stable", got, err)
	}
	q, err := p.Pade(6)
	if err != nil {
		t.Fatal(err)
	}
	cs, err := mustPID(t, 1e5, 1e5, 0, 0, 0).System()
	if err != nil {
		t.Fatal(err)
	}
	ru, err := Feedback(cs, q, -1)
	if err != nil {
		t.Fatal(err)
	}
	if ok, err := ru.IsStable(); !ok || err != nil {
		t.Fatalf("Padé r→u stable = %v, %v", ok, err)
	}
}
