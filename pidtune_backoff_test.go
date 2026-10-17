package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

func pidtuneBackoffPlant(t *testing.T, num, den []float64, dt, delay float64) *System {
	t.Helper()
	res, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	if delay > 0 {
		if err := res.Sys.SetInputDelay([]float64{delay}); err != nil {
			t.Fatal(err)
		}
	}
	return res.Sys
}

// pidtuneLoopMargin returns the smallest phase margin over the gain
// crossovers of L = C·G (dense log sweep on (0, wMax)) and the sensitivity
// peak max 1/|1+L|, from C and G evaluated directly.
func pidtuneLoopMargin(L func(float64) complex128, wMax float64) (pm, ms float64) {
	pm = math.Inf(1)
	prev := cmplx.Abs(L(1e-4))
	for w := 1e-4; w < wMax; w *= 1.0005 {
		l := L(w)
		a := cmplx.Abs(l)
		ms = math.Max(ms, 1/cmplx.Abs(1+l))
		if (a-1)*(prev-1) <= 0 {
			pm = math.Min(pm, 180-math.Abs(cmplx.Phase(l))*180/math.Pi)
		}
		prev = a
	}
	return pm, ms
}

// Q7GUKL: at wc = 0 the automatic crossover can give an unstable closed loop;
// Pidtune must lower it to a stable design instead of erroring, as MATLAB
// pidtune returns a stable design for these stable plants. Oracle:
// closed-loop characteristic roots from num/den (delay-free), dense Nyquist
// winding of 1+L (continuous delays), and the phase margin at every
// crossover of the directly evaluated loop.
func TestPidtuneAutoCrossoverBacksOffToStable(t *testing.T) {
	cases := []struct {
		name      string
		num, den  Poly
		delay     float64
		typ       PidtuneType
		minMargin float64
	}{
		{"osc I", Poly{1}, Poly{1, 0.4, 1}, 0, PidtuneI, 60},
		{"lag4 I", Poly{1}, Poly{1, 4, 6, 4, 1}, 0, PidtuneI, 60},
		{"delay I", Poly{1}, Poly{1, 1}, 1, PidtuneI, 60},
		{"delay lag3 I", Poly{1}, Poly{1, 3, 3, 1}, 2, PidtuneI, 60},
		{"nmp PD", Poly{-1, 1}, Poly{1, 3, 2}, 0, PidtunePD, 60},
		{"nmp PDF", Poly{-1, 1}, Poly{1, 3, 2}, 0, PidtunePDF, 60},
		{"nmp PID", Poly{-1, 1}, Poly{1, 3, 2}, 0, PidtunePID, 60},
		{"nmp PIDF", Poly{-1, 1}, Poly{1, 3, 2}, 0, PidtunePIDF, 60},
	}
	for _, c := range cases {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, c.delay)
		pid, err := Pidtune(p, c.typ, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		cs := func(s complex128) complex128 {
			k := complex(pid.Kp, 0) + complex(pid.Kd, 0)*s/(1+complex(pid.Tf, 0)*s)
			if pid.Ki != 0 {
				k += complex(pid.Ki, 0) / s
			}
			return k
		}
		g := func(s complex128) complex128 {
			return c.num.Eval(s) / c.den.Eval(s) * cmplx.Exp(-s*complex(c.delay, 0))
		}
		if c.delay > 0 {
			stable, _ := pidDelayLoopCheckPlant(t, &PID2{Kp: pid.Kp, Ki: pid.Ki, Kd: pid.Kd, Tf: pid.Tf}, c.delay,
				func(s complex128) complex128 { return c.num.Eval(s) / c.den.Eval(s) }, len(c.den)-len(c.num))
			if !stable {
				t.Errorf("%s: %+v unstable by winding", c.name, pid)
			}
		} else {
			tf := pid.Tf
			numC := Poly{pid.Kd + pid.Kp*tf, pid.Kp + pid.Ki*tf, pid.Ki}
			denC := Poly{tf, 1, 0}
			if pid.Ki == 0 {
				numC, denC = numC[:2], denC[:2]
			}
			char := c.den.Mul(denC).Add(c.num.Mul(numC))
			for len(char) > 1 && math.Abs(char[0]) < 1e-14 {
				char = char[1:]
			}
			roots, err := char.Roots()
			if err != nil {
				t.Fatal(err)
			}
			for _, r := range roots {
				if real(r) >= -1e-9 {
					t.Errorf("%s: %+v closed-loop root %v", c.name, pid, r)
				}
			}
		}
		pm, ms := pidtuneLoopMargin(func(w float64) complex128 { s := complex(0, w); return cs(s) * g(s) }, 1e3)
		if pm < c.minMargin-1 {
			t.Errorf("%s: %+v phase margin %g < %g", c.name, pid, pm, c.minMargin)
		}
		t.Logf("%s: %+v PM=%.3g Ms=%.3g", c.name, pid, pm, ms)
	}
}

// Q7GUKL: a discrete PD with a forward-Euler derivative and Tf = Td/10 < Dt
// has the filter pole 1−Dt/Tf outside the unit circle. Pidtune then uses the
// backward-Euler derivative, pole Tf/(Tf+Dt), and keeps forward Euler when
// Tf > Dt/2.
// Oracle: roots of den·z^d·denC + num·numC in z with
// C = Kp + Kd(z−1)/((Tf+Dt)z − Tf).
func TestPidtuneDiscretePDStableFilter(t *testing.T) {
	const dt = 0.1
	cases := []struct {
		name     string
		num, den Poly
		delay    int
	}{
		{"first order", Poly{0.1}, Poly{1, -0.9}, 0},
		{"second order", Poly{0.01, 0.009}, Poly{1, -1.7, 0.72}, 0},
		{"second order delayed", Poly{0.01, 0.009}, Poly{1, -1.7, 0.72}, 3},
	}
	for _, c := range cases {
		p := pidtuneBackoffPlant(t, c.num, c.den, dt, float64(c.delay))
		pid, err := Pidtune(p, PidtunePD, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if pid.DFormula != BackwardEuler || pid.Tf <= 0 || pid.Kd <= 0 {
			t.Errorf("%s: %+v, want backward-Euler filtered PD", c.name, pid)
		}
		a := pid.Tf + dt
		numC := Poly{pid.Kp*a + pid.Kd, -pid.Kp*pid.Tf - pid.Kd}
		denC := Poly{a, -pid.Tf}
		den := append(Poly{}, c.den...)
		for range c.delay {
			den = append(den, 0)
		}
		roots, err := den.Mul(denC).Add(c.num.Mul(numC)).Roots()
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range roots {
			if cmplx.Abs(r) >= 1-1e-9 {
				t.Errorf("%s: %+v closed-loop root %v", c.name, pid, r)
			}
		}
	}
	p := pidtuneBackoffPlant(t, Poly{0.001}, Poly{1, -2.94, 2.8812, -0.941192}, 0.02, 0)
	pid, err := Pidtune(p, PidtunePD, 1, &PidtuneOptions{PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	if pid.DFormula != ForwardEuler || pid.Tf <= 0.01 {
		t.Fatalf("%+v: forward-Euler filter pole 1−Dt/Tf is stable, want it kept", pid)
	}
}

// pidtuneOracleStable decides closed-loop stability of pid on num/den
// (variable s, or z when dt > 0) with an input delay, from the
// characteristic polynomial (delay-free or discrete) or the winding of 1+L
// (continuous delay, stable plant only); ok is false when neither applies.
func pidtuneOracleStable(t *testing.T, pid *PID, num, den Poly, dt, delay float64) (stable, ok bool) {
	t.Helper()
	if dt == 0 && delay > 0 {
		roots, err := den.Roots()
		if err != nil {
			t.Fatal(err)
		}
		for _, r := range roots {
			if real(r) >= 0 {
				return false, false
			}
		}
		return pidtuneWindingStable(pid, num, den, delay)
	}
	var x, dd, ki Poly
	if dt == 0 {
		x, dd, ki = Poly{1, 0}, Poly{pid.Tf, 1}, Poly{pid.Ki}
	} else {
		w := pidFormulaWeight(pid.DFormula)
		x = Poly{1, -1}
		dd = Poly{pid.Tf + w*dt, dt - pid.Tf - w*dt}
		ki = Poly{pid.Ki * dt}
		den = append(Poly{}, den...)
		for range int(math.Round(delay)) {
			den = append(den, 0)
		}
	}
	numC, denC := Poly{pid.Kp}, Poly{1}
	switch {
	case pid.Ki != 0 && pid.Kd != 0:
		numC = Poly{pid.Kp}.Mul(x).Mul(dd).Add(ki.Mul(dd)).Add(Poly{pid.Kd}.Mul(x).Mul(x))
		denC = x.Mul(dd)
	case pid.Ki != 0:
		numC, denC = Poly{pid.Kp}.Mul(x).Add(ki), x
	case pid.Kd != 0:
		numC, denC = Poly{pid.Kp}.Mul(dd).Add(Poly{pid.Kd}.Mul(x)), dd
	}
	char := den.Mul(denC).Add(num.Mul(numC))
	for len(char) > 1 && math.Abs(char[0]) < 1e-14 {
		char = char[1:]
	}
	roots, err := char.Roots()
	if err != nil {
		t.Fatal(err)
	}
	for _, r := range roots {
		if dt == 0 && real(r) >= -1e-9 || dt > 0 && cmplx.Abs(r) >= 1-1e-9 {
			return false, true
		}
	}
	return true, true
}

// Every design Pidtune returns at wc = 0, including those found by lowering
// the crossover, is stable by the independent oracle; random continuous and
// discrete plants, some unstable or delayed.
func TestPidtuneAutoCrossoverRandomStable(t *testing.T) {
	// e^{−s}/(s+1) under P: phase −180° at ω ≈ 2.029, |G| ≈ 0.442, so the
	// loop is stable iff −1 < Kp < 2.26; Ki < 0 puts a real root in Re s > 0.
	for _, c := range []struct {
		kp, ki, delay float64
		want          bool
	}{
		{2.2, 0, 1, true}, {2.3, 0, 1, false}, {-0.5, 0, 1, true}, {-2, 0, 1, false}, {-2, 0, 0, false},
		{2.2, 0, 0, true}, {0.5, 0.3, 1, true}, {0.5, 3, 1, false}, {0.5, -0.3, 1, false},
	} {
		stable, ok := pidtuneOracleStable(t, &PID{Kp: c.kp, Ki: c.ki}, Poly{1}, Poly{1, 1}, 0, c.delay)
		if !ok || stable != c.want {
			t.Fatalf("oracle %+v: stable %v ok %v", c, stable, ok)
		}
	}
	rng := rand.New(rand.NewPCG(3, 5))
	fams := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	checked, rejected := 0, 0
	for trial := range 300 {
		n := 1 + trial%4
		dt := 0.0
		if trial%3 == 0 {
			dt = 0.05 + 0.2*rng.Float64()
		}
		den := Poly{1}
		for range n {
			if dt == 0 {
				p := 0.2 + 3*rng.Float64()
				if rng.Float64() < 0.1 {
					p = -p
				}
				den = den.Mul(Poly{1, p})
			} else {
				den = den.Mul(Poly{1, 0.95 - 1.9*rng.Float64()})
			}
		}
		num := make(Poly, rng.IntN(n)+1)
		for i := range num {
			num[i] = 2*rng.Float64() - 1
		}
		delay := 0.0
		if trial%5 == 1 {
			delay = rng.Float64()
			if dt > 0 {
				delay = float64(1 + rng.IntN(3))
			}
		}
		p := pidtuneBackoffPlant(t, num, den, dt, delay)
		for _, f := range fams {
			for _, opts := range []*PidtuneOptions{nil, {PhaseMargin: 45}} {
				pid, err := Pidtune(p, f, 0, opts)
				if err != nil {
					if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
						t.Errorf("trial %d %s: %v", trial, f, err)
					}
					rejected++
					continue
				}
				stable, ok := pidtuneOracleStable(t, pid, num, den, dt, delay)
				if !ok {
					continue
				}
				checked++
				if !stable {
					t.Errorf("trial %d %s %v/%v dt=%g delay=%g: %+v unstable", trial, f, num, den, dt, delay, pid)
				}
			}
		}
	}
	if checked < 2000 || rejected == 0 {
		t.Fatalf("checked %d, rejected %d", checked, rejected)
	}
	t.Logf("checked %d, rejected %d", checked, rejected)
}

// pidtuneWindingStable applies the Nyquist test to 1+L, L = C·e^{−τs}·num/den
// with den Hurwitz: the phase of 1+L, unwrapped along ω ∈ (0, ω_end] with
// steps refined until each turns less than π/8, must change by π/2 with
// integral action (the indentation at s = 0 adds −π) and by 0 without,
// within π/2, the tail beyond ω_end having |L| < 1/2. ok is false when the
// tail bound or refinement fails.
func pidtuneWindingStable(pid *PID, num, den Poly, tau float64) (stable, ok bool) {
	L := func(w float64) complex128 {
		s := complex(0, w)
		c := complex(pid.Kp, 0) + complex(pid.Kd, 0)*s/(1+complex(pid.Tf, 0)*s)
		if pid.Ki != 0 {
			c += complex(pid.Ki, 0) / s
		}
		return c * num.Eval(s) / den.Eval(s) * cmplx.Exp(-s*complex(tau, 0))
	}
	const wEnd = 1e5
	gEnd := 0.0
	for i, c := range num {
		gEnd += math.Abs(c) * math.Pow(wEnd, float64(len(num)-1-i))
	}
	dEnd := math.Pow(wEnd, float64(len(den)-1))
	for i, c := range den[1:] {
		dEnd -= math.Abs(c) * math.Pow(wEnd, float64(len(den)-2-i))
	}
	if dEnd <= 0 {
		return false, false
	}
	gEnd /= dEnd
	kd := math.Abs(pid.Kd) * wEnd
	if pid.Tf > 0 {
		kd = math.Abs(pid.Kd) / pid.Tf
	}
	if (math.Abs(pid.Kp)+math.Abs(pid.Ki)/wEnd+kd)*gEnd >= 0.5 {
		return false, false
	}
	var step func(a, b float64, fa complex128, depth int) (float64, bool)
	step = func(a, b float64, fa complex128, depth int) (float64, bool) {
		fb := 1 + L(b)
		if d := cmplx.Phase(fb / fa); math.Abs(d) < math.Pi/8 {
			return d, true
		}
		if depth == 0 {
			return 0, false
		}
		m := (a + b) / 2
		d1, ok1 := step(a, m, fa, depth-1)
		d2, ok2 := step(m, b, 1+L(m), depth-1)
		return d1 + d2, ok1 && ok2
	}
	w := 1e-7
	f0 := 1 + L(w)
	total, f := 0.0, f0
	for w < wEnd {
		next := math.Min(w*1.01, w+0.5/tau)
		d, ok := step(w, next, f, 30)
		if !ok {
			return false, false
		}
		total += d
		w, f = next, 1+L(next)
	}
	target := 0.0
	if pid.Ki != 0 {
		target = math.Pi / 2
	}
	return math.Abs(total-target) < math.Pi/2, true
}
