package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"slices"
	"strings"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// pidLoopMargins sweeps |L| densely over (lo, hi) and returns the phase
// margin at every 0 dB crossing, each refined by bisection.
func pidLoopMargins(L func(float64) complex128, lo, hi float64) []float64 {
	var margins []float64
	const n = 200000
	step := math.Log(hi/lo) / n
	w := lo
	g := cmplx.Abs(L(w)) - 1
	for k := 1; k <= n; k++ {
		next := lo * math.Exp(float64(k)*step)
		ng := cmplx.Abs(L(next)) - 1
		if g == 0 || g*ng < 0 {
			a, b := w, next
			for range 60 {
				m := math.Sqrt(a * b)
				if (cmplx.Abs(L(m))-1)*g > 0 {
					a = m
				} else {
					b = m
				}
			}
			pm := 180 + cmplx.Phase(L(a))*180/math.Pi
			margins = append(margins, pm)
		}
		w, g = next, ng
	}
	return margins
}

// MATLAB pidtune treats PhaseMargin as a lower bound. On a first-order lag a
// same-sign PD only adds lead, so PM ≥ 135° at the default crossover: the
// design must exist rather than fail a ±3° band around 60°.
func TestTunePIDPhaseMarginIsAFloorContinuous(t *testing.T) {
	for _, a := range []float64{1, 10} {
		for _, family := range []PidtuneType{PidtunePD, PidtunePDF} {
			r, err := TunePID(context.Background(), makePlant(t, []float64{1}, []float64{1, a}), family, PIDTuningOptions{})
			if err != nil {
				t.Fatalf("1/(s+%g) %s: %v", a, family, err)
			}
			c := r.Controller
			if c.Kp <= 0 || c.Kd < 0 || c.Ki != 0 || c.Tf < 0 || family == PidtunePD && c.Tf != 0 {
				t.Fatalf("1/(s+%g) %s: gains %+v", a, family, c)
			}
			if r.Evidence.RequestedPhaseMargin != 60 || r.Evidence.AchievedPhaseMargin < 57 {
				t.Fatalf("1/(s+%g) %s: evidence %+v", a, family, r.Evidence)
			}
			if !slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "floor") }) {
				t.Errorf("1/(s+%g) %s: warnings %q do not report the floor", a, family, r.Evidence.Warnings)
			}
			// Closed loop Tf s² + (1 + a Tf + Kp Tf + Kd) s + (a + Kp): Routh.
			if c.Tf*(1+a*c.Tf+c.Kp*c.Tf+c.Kd) < 0 || 1+a*c.Tf+c.Kp*c.Tf+c.Kd <= 0 || a+c.Kp <= 0 {
				t.Fatalf("1/(s+%g) %s: %+v fails Routh", a, family, c)
			}
			if stable, _ := pidDelayLoopCheckPole(t, c, 0, a); !stable {
				t.Fatalf("1/(s+%g) %s: %+v fails Nyquist", a, family, c)
			}
			L := func(w float64) complex128 {
				s := complex(0, w)
				return (complex(c.Kp, 0) + complex(c.Kd, 0)*s/(1+complex(c.Tf, 0)*s)) / (s + complex(a, 0))
			}
			margins := pidLoopMargins(L, 1e-4*a, 1e5*a)
			if len(margins) == 0 {
				t.Fatalf("1/(s+%g) %s: no crossover", a, family)
			}
			for _, pm := range margins {
				if pm < 57 {
					t.Errorf("1/(s+%g) %s: independent margin %g below the 60° floor", a, family, pm)
				}
			}
		}
	}
}

// 0.1/(z−0.9), Ts = 0.1, with 0 or 3 samples of input delay: PDF's filter
// floor Tf ≥ 0.51 Ts leaves no same-sign design at PM 60°±3° anywhere on the
// crossover ladder, but PM ≥ 60° is met at the default crossover.
func TestTunePIDPhaseMarginIsAFloorDiscrete(t *testing.T) {
	const ts = .1
	for _, d := range []int{0, 3} {
		p := makePlant(t, []float64{.1}, []float64{1, -.9})
		p.Dt = ts
		if d > 0 {
			if err := p.SetInputDelay([]float64{float64(d)}); err != nil {
				t.Fatal(err)
			}
		}
		r, err := TunePID(context.Background(), p, PidtunePDF, PIDTuningOptions{})
		if err != nil {
			t.Fatalf("delay %d: %v", d, err)
		}
		c := r.Controller
		if c.Kp <= 0 || c.Kd < 0 || c.Ki != 0 || c.Tf < .51*ts || c.IFormula != ForwardEuler || c.DFormula != ForwardEuler {
			t.Fatalf("delay %d: gains %+v", d, c)
		}
		// C(z) = Kp + Kd(z−1)/(Tf(z−1)+Ts); characteristic polynomial
		// z^d (z−0.9)(Tf z + Ts−Tf) + 0.1[Kp(Tf z + Ts−Tf) + Kd(z−1)].
		den := []float64{c.Tf, ts - c.Tf}
		lhs := []float64{c.Tf, ts - c.Tf - .9*c.Tf, -.9 * (ts - c.Tf)}
		lhs = append(lhs, make([]float64, d)...)
		rhs := []float64{.1 * (c.Kp*den[0] + c.Kd), .1 * (c.Kp*den[1] - c.Kd)}
		poly := slices.Clone(lhs)
		for k := range rhs {
			poly[len(poly)-len(rhs)+k] += rhs[k]
		}
		n := len(poly) - 1
		comp := mat.NewDense(n, n, nil)
		for j := range n {
			comp.Set(0, j, -poly[j+1]/poly[0])
			if j+1 < n {
				comp.Set(j+1, j, 1)
			}
		}
		var eig mat.Eigen
		if !eig.Factorize(comp, mat.EigenNone) {
			t.Fatal("eigen failed")
		}
		for _, z := range eig.Values(nil) {
			if cmplx.Abs(z) >= 1 {
				t.Fatalf("delay %d: %+v closed-loop root %v outside the unit circle", d, c, z)
			}
		}
		L := func(w float64) complex128 {
			z := cmplx.Exp(complex(0, w*ts))
			k := complex(c.Kp, 0) + complex(c.Kd, 0)*(z-1)/(complex(c.Tf, 0)*(z-1)+ts)
			return k * .1 / (z - .9) * cmplx.Pow(z, complex(-float64(d), 0))
		}
		margins := pidLoopMargins(L, 1e-4, .9999*math.Pi/ts)
		if len(margins) == 0 {
			t.Fatalf("delay %d: no crossover", d)
		}
		for _, pm := range margins {
			if pm < 57 {
				t.Errorf("delay %d: independent margin %g below the 60° floor", d, pm)
			}
		}
		if r.Evidence.RequestedPhaseMargin != 60 || r.Evidence.AchievedPhaseMargin < 57 {
			t.Fatalf("delay %d: evidence %+v", d, r.Evidence)
		}
	}
}

// The floor is a fallback: a target reachable within ±3° (here by moving
// the PD crossover) keeps its design, and a one-gain family takes an
// explicit margin below its plant's as met.
func TestTunePIDPhaseMarginFloorIsAFallback(t *testing.T) {
	p := makePlant(t, []float64{1}, []float64{1, 3, 3, 1})
	r, err := TunePID(context.Background(), p, PidtunePD, PIDTuningOptions{})
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(r.Evidence.AchievedPhaseMargin-60) > 3 || !slices.ContainsFunc(r.Evidence.Warnings, func(w string) bool { return strings.Contains(w, "moved") }) {
		t.Fatalf("ladder design replaced: %+v", r.Evidence)
	}
	margin := 180 - 3*math.Atan(.185)*180/math.Pi
	r, err = TunePID(context.Background(), p, PidtuneP, PIDTuningOptions{CrossoverFrequency: .185, PhaseMargin: 42})
	if err != nil {
		t.Fatal(err)
	}
	if r.Controller.Kp <= 0 || math.Abs(r.Evidence.AchievedPhaseMargin-margin) > 1e-6 || r.Evidence.RequestedPhaseMargin != 42 {
		t.Fatalf("P %+v evidence %+v, want margin %g", r.Controller, r.Evidence, margin)
	}
}
