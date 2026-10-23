package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"slices"
	"strings"
	"testing"
)

// pidOffbandLoop returns L(jω) of k on num/den with lag samples of input
// delay, the plant evaluated from its coefficients, a discrete one in
// 256-bit arithmetic, with k's integrator and derivative formulas.
func pidOffbandLoop(k *PID2, num, den Poly, dt float64, lag int) func(float64) complex128 {
	return func(w float64) complex128 {
		if dt > 0 {
			h := math.Sin(w * dt / 2)
			zm1 := complex(-2*h*h, math.Sin(w*dt))
			c := complex(k.Kp, 0) + complex(k.Ki*dt, 0)*(1/zm1+complex(pidFormulaWeight(k.IFormula), 0))
			if k.Kd != 0 {
				c += complex(k.Kd, 0) / (complex(k.Tf, 0) + complex(dt, 0)*(1/zm1+complex(pidFormulaWeight(k.DFormula), 0)))
			}
			return c * pidtuneBigEval(num, 1+zm1) / pidtuneBigEval(den, 1+zm1) * cmplx.Exp(complex(0, -w*dt*float64(lag)))
		}
		s := complex(0, w)
		c := complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(complex(k.Tf, 0)*s+1)
		return c * num.Eval(s) / den.Eval(s)
	}
}

func pidOffbandPeak(L func(float64) complex128, lo, hi float64) (peak, at float64) {
	for w := lo; w <= hi; w *= 1 + 1e-4 {
		if v := 1 / cmplx.Abs(1+L(w)); v > peak {
			peak, at = v, w
		}
	}
	if v := 1 / cmplx.Abs(1+L(hi)); v > peak {
		peak, at = v, hi
	}
	return peak, at
}

// SEFPKM: the |S| bound and the explicit-crossover warning scanned only
// [wc/100, wc·100], so |S| peaks below that floor or, discrete, between its
// top and π/Dt went unseen. Delayed discrete P (d40) claimed "sampled |S| <=
// 4" with |S| 12.4 at 9.58 rad/s (Nyquist 30); continuous PD (c42) kept a
// second crossover at 0.311 rad/s with PM 0.2° and |S| 383, its band
// starting at 4.08; explicit-crossover discrete PID (d43) peaked at |S| ~1e5
// at 2.475 rad/s above its band top 1.52, unwarned. Every default design now
// keeps |S| <= 4 densely over [1e-5, π/Dt] or 1e6 rad/s, else fails; the
// explicit design warns. Since AP6JSB its crossing at 2.69 rad/s, above
// wc·100, holds the 57° floor and |S| falls to ~9, still over 4 and warned.
// Oracle: the loop from the coefficients, a discrete plant in 256-bit
// arithmetic, on a 1e-4 log grid.
func TestTunePIDOffbandSensitivity(t *testing.T) {
	skipSlowUnderRace(t)
	for _, c := range []struct {
		name     string
		num, den Poly
		dt       float64
		lag      int
		fam      PidtuneType
		opts     PIDTuningOptions
	}{
		{"d40 P", Poly{0.019742253670921476}, Poly{1, -0.14539455905586565}, 0.10463990694240013, 2, PidtuneP, PIDTuningOptions{}},
		{"c42 PD", Poly{0.25735368672434666, -0.09959483477301487, 0.0011398317458989433}, Poly{1, 96.41485396485868, 3538.780004777385, 53732.01273661257, 13120.99411576446}, 0, 0, PidtunePD, PIDTuningOptions{PhaseMargin: 45}},
	} {
		r, err := TunePID(context.Background(), pidtuneBackoffPlant(t, c.num, c.den, c.dt, float64(c.lag)), c.fam, c.opts)
		if err != nil {
			if _, ok := errors.AsType[*PIDTuningError](err); !ok {
				t.Errorf("%s: %v", c.name, err)
			}
			continue
		}
		hi := 1e6
		if c.dt > 0 {
			hi = math.Pi / c.dt
		}
		if ms, at := pidOffbandPeak(pidOffbandLoop(r.Controller, c.num, c.den, c.dt, c.lag), 1e-5, hi); ms > pidTuningMaxSensitivity*1.01 {
			t.Errorf("%s: %+v dense |S| %.4g at ω=%.4g, warnings %q", c.name, *r.Controller, ms, at, r.Evidence.Warnings)
		}
	}

	num, den := Poly{3.813604642580321}, Poly{1, -0.39307405567494685, -0.517865771395293, -0.09524444711884916, 0.006184274189089005}
	r, err := TunePID(context.Background(), pidtuneBackoffPlant(t, num, den, 1, 0), PidtunePID, PIDTuningOptions{CrossoverFrequency: 0.015241268242045015, IFormula: Trapezoidal, DFormula: BackwardEuler})
	if err != nil {
		t.Fatal(err)
	}
	k := r.Controller
	if math.Abs(k.Kp-0.0060596)/0.0060596 > 1e-3 || math.Abs(k.Ki-7.7340e-5)/7.7340e-5 > 1e-3 || math.Abs(k.Kd-0.13954)/0.13954 > 1e-3 {
		t.Errorf("d43: gains %+v changed", *k)
	}
	ms, at := pidOffbandPeak(pidOffbandLoop(k, num, den, 1, 0), 1e-5, math.Pi)
	if ms <= pidTuningMaxSensitivity*1.01 || at < 100*0.015241268242045015 {
		t.Fatalf("d43: dense |S| %g at ω=%g, band %v; reference no longer above wc·100", ms, at, r.Evidence.FrequencyBand)
	}
	if pidExplicitSWarned(r) != 1 {
		t.Errorf("d43: dense |S| %g at ω=%g, warnings %q", ms, at, r.Evidence.Warnings)
	}
}

// d40's P loop has a gain crossover only for Kp in [1/|G(1)|, 1/|G(−1)|]
// ≈ [43.3, 58.0], and every such Kp keeps |S| > 4 somewhere below Nyquist,
// so its default design must fail rather than claim the bound.
func TestTunePIDOffbandP40Infeasible(t *testing.T) {
	skipSlowUnderRace(t)
	num, den, dt := Poly{0.019742253670921476}, Poly{1, -0.14539455905586565}, 0.10463990694240013
	lo, hi := 1/cmplx.Abs(num.Eval(1)/den.Eval(1)), 1/cmplx.Abs(num.Eval(-1)/den.Eval(-1))
	for kp := lo; kp <= hi; kp += (hi - lo) / 200 {
		L := pidOffbandLoop(&PID2{Kp: kp, Dt: dt}, num, den, dt, 2)
		peak := 0.
		for w := 1e-4; w <= math.Pi/dt; w *= 1.001 {
			peak = math.Max(peak, 1/cmplx.Abs(1+L(w)))
		}
		if peak <= pidTuningMaxSensitivity {
			t.Fatalf("Kp %g keeps |S| %g; d40 has a feasible P design", kp, peak)
		}
	}
	_, err := TunePID(context.Background(), pidtuneBackoffPlant(t, num, den, dt, 2), PidtuneP, PIDTuningOptions{})
	if te, ok := errors.AsType[*PIDTuningError](err); !ok || !slices.ContainsFunc(te.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "sensitivity peak") }) {
		t.Errorf("want an |S| tuning error, got %v", err)
	}
}
