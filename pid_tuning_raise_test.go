package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"strings"
	"testing"
)

// TestTunePIDRaisesCrossoverOnUnstablePlant: the default crossover of
// 1/(s(s−1)) (0.786 rad/s) lies below the unstable pole, where no D family
// meets the margin with a stabilising loop; as Pidtune, TunePID raises it.
// Stability is checked by the Routh array of the closed-loop polynomial
// built from the gains, the margin by L(jωc) evaluated directly.
func TestTunePIDRaisesCrossoverOnUnstablePlant(t *testing.T) {
	skipSlowUnderRace(t)
	cases := []struct {
		name string
		den  []float64
	}{
		{"s(s-1)", []float64{1, -1, 0}},
		{"s^2-0.5s+1", []float64{1, -0.5, 1}},
	}
	for _, tc := range cases {
		r, err := (&TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{tc.den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		plant := r.Sys
		g := func(s complex128) complex128 {
			return 1 / (complex(tc.den[0], 0)*s*s + complex(tc.den[1], 0)*s + complex(tc.den[2], 0))
		}
		for _, f := range []PidtuneType{PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF} {
			res, err := TunePID(context.Background(), plant, f, PIDTuningOptions{})
			if err != nil {
				t.Errorf("%s %s: %v", tc.name, f, err)
				continue
			}
			c := res.Controller
			ch := pidRaiseCharPoly(tc.den, c)
			if !routhStable(ch) {
				t.Errorf("%s %s: %+v closed loop %v unstable", tc.name, f, *c, ch)
			}
			wc := res.Evidence.AchievedCrossover
			s := complex(0, wc)
			k := complex(c.Kp, 0) + complex(c.Kd, 0)*s/(complex(c.Tf, 0)*s+1)
			if c.Ki != 0 {
				k += complex(c.Ki, 0) / s
			}
			l := k * g(s)
			if math.Abs(cmplx.Abs(l)-1) > 1e-5 {
				t.Errorf("%s %s: |L(j%.4g)| = %v, want 1", tc.name, f, wc, cmplx.Abs(l))
			}
			if pm := 180 - math.Abs(cmplx.Phase(l))*180/math.Pi; pm < 57-1e-3 {
				t.Errorf("%s %s: margin %.4g° at %.4g rad/s, want >= 57°", tc.name, f, pm, wc)
			}
			if !pidRaiseWarned(res.Evidence.Warnings, "raised to") {
				t.Errorf("%s %s: warnings %q lack the raise", tc.name, f, res.Evidence.Warnings)
			}
		}
	}
	r, err := (&TransferFunc{Num: [][][]float64{{{1}}}, Den: [][]float64{{1, -1, 0}}}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []PidtuneType{PidtuneP, PidtuneI, PidtunePI} {
		if _, err := TunePID(context.Background(), r.Sys, f, PIDTuningOptions{}); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Errorf("%s on 1/(s(s-1)): err %v, want ErrPIDTuningTargetUnattainable (no stabilising %s)", f, err, f)
		}
	}
}

// pidRaiseCharPoly returns, descending, the closed-loop characteristic
// polynomial of C = Kp + Ki/s + Kd·s/(Tf·s+1) on 1/D(s) for a quadratic D:
// s(Tf·s+1)·D(s) + Kp·s(Tf·s+1) + Ki(Tf·s+1) + Kd·s².
func pidRaiseCharPoly(den []float64, c *PID2) []float64 {
	mul := func(a, b []float64) []float64 {
		out := make([]float64, len(a)+len(b)-1)
		for i, x := range a {
			for j, y := range b {
				out[i+j] += x * y
			}
		}
		return out
	}
	add := func(a, b []float64) []float64 {
		if len(a) < len(b) {
			a, b = b, a
		}
		out := append([]float64(nil), a...)
		for i := range b {
			out[len(a)-len(b)+i] += b[i]
		}
		return out
	}
	filt := []float64{c.Tf, 1}
	p := mul(mul([]float64{1, 0}, filt), den)
	p = add(p, mul([]float64{c.Kp, 0}, filt))
	p = add(p, mul([]float64{c.Ki}, filt))
	p = add(p, []float64{c.Kd, 0, 0})
	for len(p) > 1 && p[0] == 0 {
		p = p[1:]
	}
	if c.Ki == 0 {
		// The s factor shared by s·D(s)… and Kp·s, Kd·s² cancels.
		p = p[:len(p)-1]
	}
	return p
}

func pidRaiseWarned(ws []string, sub string) bool {
	for _, w := range ws {
		if strings.Contains(w, sub) {
			return true
		}
	}
	return false
}
