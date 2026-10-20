package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
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
	return pidtuneLoopMarginBand(L, 1e-4, wMax, 1.0002)
}

func pidtuneLoopMarginBand(L func(float64) complex128, wMin, wMax, ratio float64) (pm, ms float64) {
	pm = math.Inf(1)
	prevW, prev := wMin, cmplx.Abs(L(wMin))
	for w := wMin; w <= wMax; w *= ratio {
		l := L(w)
		a := cmplx.Abs(l)
		ms = math.Max(ms, 1/cmplx.Abs(1+l))
		if (a-1)*(prev-1) <= 0 {
			lo, hi := prevW, w
			for range 60 {
				mid := math.Sqrt(lo * hi)
				if (cmplx.Abs(L(lo))-1)*(cmplx.Abs(L(mid))-1) <= 0 {
					hi = mid
				} else {
					lo = mid
				}
			}
			x := l
			for _, v := range []complex128{L(lo), L(hi)} {
				if math.Abs(cmplx.Abs(v)-1) <= math.Abs(cmplx.Abs(x)-1) {
					x = v
				}
			}
			pm = math.Min(pm, 180-math.Abs(cmplx.Phase(x))*180/math.Pi)
		}
		prevW, prev = w, a
	}
	return pm, ms
}

// pidtuneOracleController returns the numerator and denominator of pid in s
// (z when dt > 0), from the PID formulas: Ki/s or Ki·Ts/(z−1), and
// Kd·s/(Tf·s+1) or Kd/(Tf + Ts·(w·z+1−w)/(z−1)) for derivative formula
// weight w.
func pidtuneOracleController(pid *PID, dt float64) (numC, denC Poly) {
	var x, dd, ki Poly
	if dt == 0 {
		x, dd, ki = Poly{1, 0}, Poly{pid.Tf, 1}, Poly{pid.Ki}
	} else {
		w := pidFormulaWeight(pid.DFormula)
		x = Poly{1, -1}
		dd = Poly{pid.Tf + w*dt, dt - pid.Tf - w*dt}
		ki = Poly{pid.Ki * dt}
	}
	if dd[0] == 0 {
		dd = dd[1:]
	}
	numC, denC = Poly{pid.Kp}, Poly{1}
	switch {
	case pid.Ki != 0 && pid.Kd != 0:
		numC = Poly{pid.Kp}.Mul(x).Mul(dd).Add(ki.Mul(dd)).Add(Poly{pid.Kd}.Mul(x).Mul(x))
		denC = x.Mul(dd)
	case pid.Ki != 0:
		numC, denC = Poly{pid.Kp}.Mul(x).Add(ki), x
	case pid.Kd != 0:
		numC, denC = Poly{pid.Kp}.Mul(dd).Add(Poly{pid.Kd}.Mul(x)), dd
	}
	return numC, denC
}

// pidtuneOracleLoop returns ω ↦ L(jω) (L(e^{jωdt})) of pid on num/den with
// an input delay in seconds (samples when dt > 0).
func pidtuneOracleLoop(pid *PID, num, den Poly, dt, delay float64) func(float64) complex128 {
	numC, denC := pidtuneOracleController(pid, dt)
	return func(w float64) complex128 {
		x := complex(0, w)
		d := cmplx.Exp(-x * complex(delay, 0))
		if dt > 0 {
			x = cmplx.Exp(complex(0, w*dt))
			d = cmplx.Exp(complex(0, -w*dt*delay))
		}
		return numC.Eval(x) / denC.Eval(x) * num.Eval(x) / den.Eval(x) * d
	}
}

// pidtuneBackedOff reports whether Pidtune(p, f, 0, opts), returning pid,
// had to lower the automatic crossover: the design at that crossover itself
// is rejected. A PD is compared with each first design the automatic
// pipeline can return at that crossover (placed, legacy-Td, legacy).
func pidtuneBackedOff(t *testing.T, p *System, f PidtuneType, opts *PidtuneOptions, pid *PID) bool {
	t.Helper()
	pm := 60.0
	if opts != nil && opts.PhaseMargin != 0 {
		pm = opts.PhaseMargin
	}
	wc, err := findCrossoverFreq(p, 0, pm)
	if err != nil {
		t.Fatal(err)
	}
	if f == PidtunePD && pid != nil {
		for _, sign := range []float64{1, -1} {
			for _, sf := range []bool{true, false} {
				for _, mode := range []pidtunePDMode{pidtunePDPlaced, pidtunePDLegacyTd, pidtunePDLegacy} {
					first, err := pidtuneGains(p, &pidtunePlantPoint{}, f, wc, pm, sign, sf, false, mode, pidtunePIDFIterate)
					if err == nil && *first == *pid {
						return false
					}
				}
			}
		}
	}
	_, err = Pidtune(p, f, wc, opts)
	return err != nil
}

// pidtuneDenseRobust checks a backed-off design by a dense sweep of the
// directly evaluated loop over the gate's band and four decades above it,
// which the gate scans while the loop gain has yet to roll off: every
// crossover keeps the phase margin and |S| stays near the gate's bound
// pidTuningMaxSensitivity (sampling slack only).
func pidtuneDenseRobust(t *testing.T, name string, pid *PID, num, den Poly, dt, delay, wc, pm float64) {
	t.Helper()
	hi := wc * 1e7
	if dt > 0 {
		hi = math.Pi / dt
	}
	m, ms := pidtuneLoopMarginBand(pidtuneOracleLoop(pid, num, den, dt, delay), wc*1e-6, hi, 1.001)
	if m < pm-1.5 || ms > 4.5 {
		t.Errorf("%s: %+v phase margin %.3g (want ≥ %g), Ms %.3g (want ≤ 4)", name, pid, m, pm, ms)
	}
}

// Q7GUKL: at wc = 0 the automatic crossover can give an unstable closed loop;
// Pidtune must lower it to a stable design instead of erroring, as MATLAB
// pidtune returns a stable design for these stable plants. Oracle:
// closed-loop characteristic roots from num/den (delay-free), dense Nyquist
// winding of 1+L (continuous delays), and the phase margin at every
// crossover of the directly evaluated loop. The back-off gate also bounds
// |S| by 4: every stable PDF design on (1−s)/((s+1)(s+2)) with the 60°
// margin has a larger peak (Ms ≥ 4.0 at wc = 1.12, 28 and PM 26° at
// another crossover for wc = 1.58), so that family may be unattainable.
func TestPidtuneAutoCrossoverBacksOffToStable(t *testing.T) {
	cases := []struct {
		name      string
		num, den  Poly
		delay     float64
		typ       PidtuneType
		minMargin float64
		mayFail   bool
	}{
		{"osc I", Poly{1}, Poly{1, 0.4, 1}, 0, PidtuneI, 60, false},
		{"lag4 I", Poly{1}, Poly{1, 4, 6, 4, 1}, 0, PidtuneI, 60, false},
		{"delay I", Poly{1}, Poly{1, 1}, 1, PidtuneI, 60, false},
		{"delay lag3 I", Poly{1}, Poly{1, 3, 3, 1}, 2, PidtuneI, 60, false},
		{"nmp PD", Poly{-1, 1}, Poly{1, 3, 2}, 0, PidtunePD, 60, false},
		{"nmp PDF", Poly{-1, 1}, Poly{1, 3, 2}, 0, PidtunePDF, 60, true},
		{"nmp PID", Poly{-1, 1}, Poly{1, 3, 2}, 0, PidtunePID, 60, false},
		{"nmp PIDF", Poly{-1, 1}, Poly{1, 3, 2}, 0, PidtunePIDF, 60, false},
	}
	for _, c := range cases {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, c.delay)
		pid, err := Pidtune(p, c.typ, 0, nil)
		if c.mayFail && errors.Is(err, ErrPIDTuningTargetUnattainable) {
			continue
		}
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
		if pm < c.minMargin-1.5 || ms > 4.5 {
			t.Errorf("%s: %+v phase margin %g (want ≥ %g), Ms %g", c.name, pid, pm, c.minMargin, ms)
		}
		wc, err := findCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		pidtuneDenseRobust(t, c.name, pid, c.num, c.den, 0, c.delay, wc, c.minMargin)
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

// pidtuneRobustOrUnattainable checks that Pidtune either reports the target
// unattainable or returns a design keeping pm at every crossover of the
// directly evaluated loop on (0, π/Dt] with |S| ≤ 4 (sampling slack only).
func pidtuneRobustOrUnattainable(t *testing.T, name string, pid *PID, err error, num, den Poly, dt, pm float64) {
	t.Helper()
	if errors.Is(err, ErrPIDTuningTargetUnattainable) {
		return
	}
	if err != nil {
		t.Fatalf("%s: %v", name, err)
	}
	pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, den, dt, 0), 1e-6, math.Pi/dt)
	if ms > 4.5 {
		t.Errorf("%s: %+v |S| peak %.3g", name, pid, ms)
	}
	for _, m := range pms {
		if m < pm-1.5 {
			t.Errorf("%s: %+v phase margin %.3g° at a crossover", name, pid, m)
		}
	}
	if stable, ok := pidtuneOracleStable(t, pid, num, den, dt, 0); !ok || !stable {
		t.Errorf("%s: %+v unstable", name, pid)
	}
}

// YXFRDR: when the backward-Euler redesign of a forward-Euler filter pole
// 1−Dt/Tf outside the unit circle (Tf < Dt/2) is unstable, the forward-Euler
// design is the fallback only if it keeps the margin at every crossover and
// |S| ≤ 4. These stable forward-Euler PIDFs cross over a second time with
// PM −34.5° (|S| 8.6, wc = 0 on an unstable plant) and −23.6° (explicit wc),
// which the ungated fallback returned. Oracle: characteristic roots in z and
// the signed margin at every crossover of the directly evaluated loop.
func TestPidtuneDiscreteFilterForwardEulerFallbackGated(t *testing.T) {
	cases := []struct {
		num, den Poly
		dt, wc   float64
	}{
		{Poly{0.24591}, Poly{1, -1.84612, 0.74449}, 0.274846, 0},
		{Poly{0.17032241632981948, 1.4143934743506665}, Poly{1, -1.8341193107935498, 0.756547032348602}, 0.3419630736464782, 1.763289606616569},
	}
	for _, c := range cases {
		name := fmt.Sprintf("%v/%v wc=%g", c.num, c.den, c.wc)
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		wc := c.wc
		if wc == 0 {
			var err error
			if wc, err = findCrossoverFreq(p, 0, 60); err != nil {
				t.Fatal(err)
			}
		}
		be, err := computePIDGains(p, PidtunePIDF, wc, 60, 1, true)
		if err != nil || be.DFormula != BackwardEuler {
			t.Fatalf("%s: backward-Euler redesign %+v, %v", name, be, err)
		}
		if stable, ok := pidtuneOracleStable(t, be, c.num, c.den, c.dt, 0); !ok || stable {
			t.Fatalf("%s: backward-Euler redesign %+v stable %v ok %v; the fallback is untested", name, be, stable, ok)
		}
		fe, err := computePIDGains(p, PidtunePIDF, wc, 60, 1, false)
		if err != nil || fe.DFormula != ForwardEuler || fe.Tf > c.dt/2 {
			t.Fatalf("%s: forward-Euler design %+v, %v", name, fe, err)
		}
		if stable, ok := pidtuneOracleStable(t, fe, c.num, c.den, c.dt, 0); !ok || !stable {
			t.Fatalf("%s: forward-Euler design %+v unstable; the fallback is untested", name, fe)
		}
		pms, _ := pidtuneSignedMargins(pidtuneOracleLoop(fe, c.num, c.den, c.dt, 0), 1e-6, math.Pi/c.dt)
		if len(pms) == 0 || slices.Min(pms) > 0 {
			t.Fatalf("%s: forward-Euler design %+v margins %v; the gate is untested", name, fe, pms)
		}
		pid, err := Pidtune(p, PidtunePIDF, c.wc, nil)
		pidtuneRobustOrUnattainable(t, name, pid, err, c.num, c.den, c.dt, 60)
	}
}

// YXFRDR: when the backward-Euler redesign of a filter pole 1−Dt/Tf outside
// the unit circle has no same-sign PDF gains, the forward-Euler design is
// the fallback only if it passes the back-off gate:
// (−0.7z+0.9)/(z²−1.6z+0.7), Dt = 1, PDF at wc = 0.5, whose stable
// forward-Euler design has PM −3.6° at a second crossover and |S| 23.
// Oracle: the gains solving C(e^{jωc})·G(e^{jωc}) = 1∠−120° with
// Tf = 0.1/ωc for each derivative formula, the characteristic roots in z and
// the signed margin at every crossover of the directly evaluated loop.
func TestPidtuneDiscreteFilterFallbackWhenBackwardEulerUnattainable(t *testing.T) {
	const dt, wc = 1.0, 0.5
	num, den := Poly{-0.7, 0.9}, Poly{1, -1.6, 0.7}
	z := cmplx.Exp(complex(0, wc*dt))
	target := cmplx.Rect(1, -120*math.Pi/180) * den.Eval(z) / num.Eval(z)
	tf := 0.1 / wc
	gains := func(w float64) (kp, kd float64) {
		d := 1 / (complex(tf, 0) + complex(dt, 0)*(complex(w, 0)*z+complex(1-w, 0))/(z-1))
		kd = imag(target) / imag(d)
		return real(target) - kd*real(d), kd
	}
	if kp, kd := gains(1); kp >= 0 && kd >= 0 {
		t.Fatalf("backward-Euler Kp %g Kd %g same-sign; the fallback is untested", kp, kd)
	}
	kp, kd := gains(0)
	if kp <= 0 || kd <= 0 || tf > dt/2 {
		t.Fatalf("forward-Euler Kp %g Kd %g Tf %g", kp, kd, tf)
	}
	fe := &PID{Kp: kp, Kd: kd, Tf: tf, Dt: dt, DFormula: ForwardEuler}
	if stable, ok := pidtuneOracleStable(t, fe, num, den, dt, 0); !ok || !stable {
		t.Fatalf("forward-Euler design %+v unstable; the fallback is untested", fe)
	}
	pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(fe, num, den, dt, 0), 1e-6, math.Pi/dt)
	if slices.Min(pms) > 0 || ms < 4.5 {
		t.Fatalf("forward-Euler design %+v margins %v |S| %.3g; the gate is untested", fe, pms, ms)
	}
	p := pidtuneBackoffPlant(t, num, den, dt, 0)
	pid, err := Pidtune(p, PidtunePDF, wc, nil)
	pidtuneRobustOrUnattainable(t, "PDF", pid, err, num, den, dt, 60)
}

// Q7GUKL back-off gate: a design found by lowering the crossover keeps the
// phase margin at every crossover and |S| ≤ 4. These plants have close
// crossover pairs (2.058 and 2.101 rad/s, PM 6.5° and 4.5°, Ms 532 for the
// first), or a near-unit |1+L| with the margin met (discrete, Ms 1253), at
// the first crossover a 100-per-decade phase scan accepted. Oracle: a dense
// sweep of the directly evaluated loop and the characteristic roots.
func TestPidtuneBackoffGateRobust(t *testing.T) {
	cases := []struct {
		num, den  Poly
		dt, delay float64
		typ       PidtuneType
	}{
		{Poly{-0.61483, 0.37602}, Poly{1, 9.3924, 29.375, 30.593}, 0, 0, PidtuneP},
		{Poly{-0.48212, 0.080496}, Poly{1, 3.3348, 1.0647}, 0, 0, PidtuneP},
		{Poly{0.77538}, Poly{1, -0.81441, 0.12105, -0.0040629}, 0.067056, 3, PidtunePID},
		{Poly{-1, 1}, Poly{1, 3, 2}, 0, 0, PidtunePDF},
	}
	designs := 0
	for _, c := range cases {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, c.delay)
		name := fmt.Sprintf("%v/%v %s", c.num, c.den, c.typ)
		if !pidtuneBackedOff(t, p, c.typ, nil, nil) {
			t.Fatalf("%s: automatic crossover accepted, the gate is untested", name)
		}
		pid, err := Pidtune(p, c.typ, 0, nil)
		if errors.Is(err, ErrPIDTuningTargetUnattainable) {
			continue
		}
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		designs++
		wc, err := findCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		pidtuneDenseRobust(t, name, pid, c.num, c.den, c.dt, c.delay, wc, 60)
		if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, c.delay); !ok || !stable {
			t.Errorf("%s: %+v unstable", name, pid)
		}
	}
	if designs == 0 {
		t.Fatal("no design returned")
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
	if dt > 0 {
		den = append(Poly{}, den...)
		for range int(math.Round(delay)) {
			den = append(den, 0)
		}
	}
	numC, denC := pidtuneOracleController(pid, dt)
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
	checked, rejected, backedOff := 0, 0, 0
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
				name := fmt.Sprintf("trial %d %s %v/%v dt=%g delay=%g", trial, f, num, den, dt, delay)
				if pidtuneBackedOff(t, p, f, opts, pid) {
					backedOff++
					pm := 60.0
					if opts != nil {
						pm = opts.PhaseMargin
					}
					wc, err := findCrossoverFreq(p, 0, pm)
					if err != nil {
						t.Fatal(err)
					}
					pidtuneDenseRobust(t, name, pid, num, den, dt, delay, wc, pm)
				}
				stable, ok := pidtuneOracleStable(t, pid, num, den, dt, delay)
				if !ok {
					continue
				}
				checked++
				if !stable {
					t.Errorf("%s: %+v unstable", name, pid)
				}
			}
		}
	}
	if checked < 2000 || rejected == 0 || backedOff < 50 {
		t.Fatalf("checked %d, rejected %d, backed off %d", checked, rejected, backedOff)
	}
	t.Logf("checked %d, rejected %d, backed off %d", checked, rejected, backedOff)
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

// The back-off gate's cached grids must reproduce the uncached loop bit for
// bit, or back-off would return different designs: controller response
// against pidTuningFeedback for every formula pair, and the crossings scan
// against pidTuningCrossings.
func TestPidtuneGateGridExact(t *testing.T) {
	formulas := []PIDFormula{ForwardEuler, BackwardEuler, Trapezoidal}
	for _, dt := range []float64{0, 0.1} {
		den := []float64{1, 0.6, 0.25}
		if dt > 0 {
			den = []float64{1, -1.6, 0.68}
		}
		p := pidtuneBackoffPlant(t, []float64{0.4, 0.3}, den, dt, 0)
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		gate := newPidtuneGate(p, p, eval, 2, 2e-3)
		checked := 0
		for _, fi := range formulas {
			for _, fd := range formulas {
				for _, c := range []PID2{
					{Kp: 1.3, Ki: 0.7, Kd: 0.4, Tf: 0.05},
					{Kp: -0.2, Ki: 3, Kd: 2, Tf: 0.3},
					{Kp: 0.9},
				} {
					c.Dt, c.IFormula, c.DFormula = dt, fi, fd
					k := gate.sens.feedback(c)
					for i, w := range gate.sens.w {
						if got, want := k(i), pidTuningFeedback(c, w); got != want {
							t.Fatalf("dt=%g %+v: w=%g cached %v, direct %v", dt, c, w, got, want)
						}
					}
					wc0, m0 := pidTuningCrossings(gate.p, c, gate.cross.w, 2)
					wc1, m1 := pidTuningCrossingsOn(gate.p, c, gate.cross.w, 2, gate.crossExcess(c))
					if fmt.Sprint(wc0, m0) != fmt.Sprint(wc1, m1) {
						t.Fatalf("dt=%g %+v: crossings %v %v, cached %v %v", dt, c, wc0, m0, wc1, m1)
					}
					checked += len(wc0) - 1
				}
				if dt == 0 {
					break
				}
			}
			if dt == 0 {
				break
			}
		}
		if checked == 0 {
			t.Fatalf("dt=%g: no crossing besides wc compared", dt)
		}
	}
}

// G = (0.1z+0.05)/((z−1)(z−0.999)²), Dt = 0.1: the double pole at ≈0.01 rad/s
// lags P/PI/PD past −180° at every crossover within three decades of the
// automatic 5.3 rad/s. Oracles: closed-loop roots of the polynomial loop,
// and the phase margin and peak |S| of L(e^{jωDt}) from the raw polynomials
// on a dense grid.
func TestPidtuneSlowPoleBackoff(t *testing.T) {
	const dt = .1
	num, den := Poly{0.1, 0.05}, Poly{1, -1}.Mul(Poly{1, -0.999}).Mul(Poly{1, -0.999})
	pos, neg := negatedPlant(t, num, den, dt)
	negNum := Poly{-num[0], -num[1]}
	for _, f := range []PidtuneType{PidtuneP, PidtunePI, PidtunePD} {
		cp, err := Pidtune(pos, f, 0, nil)
		if err != nil {
			t.Fatalf("%s: G: %v", f, err)
		}
		cn, err := Pidtune(neg, f, 0, nil)
		if err != nil {
			t.Fatalf("%s: −G: %v", f, err)
		}
		for _, g := range [][2]float64{{cp.Kp, cn.Kp}, {cp.Ki, cn.Ki}, {cp.Kd, cn.Kd}} {
			if math.Abs(g[0]+g[1]) > 1e-12*math.Abs(g[0]) {
				t.Errorf("%s: %+v, %+v not mirrored", f, cp, cn)
			}
		}
		for _, c := range []struct {
			pid *PID
			num Poly
		}{{cp, num}, {cn, negNum}} {
			if !pidtuneExactSchurStable(t, c.pid, c.num, den, dt) {
				t.Errorf("%s: %+v unstable", f, c.pid)
			}
			numC, denC := pidtuneOracleController(c.pid, dt)
			loop := func(w float64) complex128 {
				z := cmplx.Exp(complex(0, w*dt))
				return c.num.Eval(z) * numC.Eval(z) / (den.Eval(z) * denC.Eval(z))
			}
			var crosses []float64
			ms, prev := 0.0, math.NaN()
			for w := 1e-7; w < math.Pi/dt; w *= 1.001 {
				l := loop(w)
				ms = math.Max(ms, 1/cmplx.Abs(1+l))
				m := cmplx.Abs(l) - 1
				if prev > 0 && m <= 0 {
					crosses = append(crosses, w)
					pm := 180 + cmplx.Phase(l)*180/math.Pi
					if pm > 180 {
						pm -= 360
					}
					if pm < 59 {
						t.Errorf("%s: %+v phase margin %.3g° at %g", f, c.pid, pm, w)
					}
				}
				prev = m
			}
			if len(crosses) == 0 || crosses[0] < 1e-3 || crosses[0] > 1e-2 {
				t.Errorf("%s: %+v gain crossovers %v, want first in [1e-3, 1e-2]", f, c.pid, crosses)
			}
			if ms > pidTuningMaxSensitivity*1.01 {
				t.Errorf("%s: %+v peak |S| %.3g", f, c.pid, ms)
			}
		}
	}
}

// pidtuneExactSchurStable reports whether the discrete loop of a
// forward-Euler-integral pid and num/den has every closed-loop root strictly
// inside the unit circle, by the Schur–Cohn recursion in exact rationals:
// clustered roots near z = 1 defeat floating-point root finding.
func pidtuneExactSchurStable(t *testing.T, pid *PID, num, den Poly, dt float64) bool {
	t.Helper()
	return pidtuneExactSchurStableRadius(t, pid, num, den, dt, 1)
}

// pidtuneExactSchurStableRadius is pidtuneExactSchurStable for roots
// strictly inside |z| = rho.
func pidtuneExactSchurStableRadius(t *testing.T, pid *PID, num, den Poly, dt, rho float64) bool {
	t.Helper()
	if pid.IFormula != ForwardEuler {
		t.Fatalf("oracle needs a forward-Euler integral, got %v", pid.IFormula)
	}
	r := func(v float64) *big.Rat { return new(big.Rat).SetFloat64(v) }
	poly := func(p Poly) []*big.Rat {
		out := make([]*big.Rat, len(p))
		for i, v := range p {
			out[i] = r(v)
		}
		return out
	}
	mul := func(a, b []*big.Rat) []*big.Rat {
		out := make([]*big.Rat, len(a)+len(b)-1)
		for i := range out {
			out[i] = new(big.Rat)
		}
		for i, x := range a {
			for j, y := range b {
				out[i+j].Add(out[i+j], new(big.Rat).Mul(x, y))
			}
		}
		return out
	}
	add := func(a, b []*big.Rat) []*big.Rat {
		if len(a) < len(b) {
			a, b = b, a
		}
		out := make([]*big.Rat, len(a))
		for i := range a {
			out[i] = new(big.Rat).Set(a[i])
			if k := i - (len(a) - len(b)); k >= 0 {
				out[i].Add(out[i], b[k])
			}
		}
		return out
	}
	x := []*big.Rat{r(1), r(-1)}
	kp, kd, dtR := []*big.Rat{r(pid.Kp)}, []*big.Rat{r(pid.Kd)}, r(dt)
	ki := []*big.Rat{new(big.Rat).Mul(r(pid.Ki), dtR)}
	lead := new(big.Rat).Add(r(pid.Tf), new(big.Rat).Mul(r(pidFormulaWeight(pid.DFormula)), dtR))
	dd := []*big.Rat{lead, new(big.Rat).Sub(dtR, lead)}
	if lead.Sign() == 0 {
		dd = dd[1:]
	}
	numC, denC := kp, []*big.Rat{r(1)}
	switch {
	case pid.Ki != 0 && pid.Kd != 0:
		numC = add(add(mul(mul(kp, x), dd), mul(ki, dd)), mul(mul(kd, x), x))
		denC = mul(x, dd)
	case pid.Ki != 0:
		numC, denC = add(mul(kp, x), ki), x
	case pid.Kd != 0:
		numC, denC = add(mul(kp, dd), mul(kd, x)), dd
	}
	c := add(mul(poly(den), denC), mul(poly(num), numC))
	for len(c) > 1 && c[0].Sign() == 0 {
		c = c[1:]
	}
	scale := r(1)
	for i := len(c) - 2; i >= 0; i-- {
		scale.Mul(scale, r(rho))
		c[i].Mul(c[i], scale)
	}
	for n := len(c) - 1; n > 0; n-- {
		if new(big.Rat).Abs(c[n]).Cmp(new(big.Rat).Abs(c[0])) >= 0 {
			return false
		}
		q := make([]*big.Rat, n)
		for i := range q {
			q[i] = new(big.Rat).Sub(new(big.Rat).Mul(c[0], c[i]), new(big.Rat).Mul(c[n], c[n-i]))
		}
		c = q
	}
	return true
}
