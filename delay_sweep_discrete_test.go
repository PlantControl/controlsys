package controlsys_test

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
	"plantcontrol.org/v2/controlsys/v2"
)

const sweepDiscreteTs = 0.1

var sweepDiscreteRatios = []float64{0, 0.001, 0.5, 0.999999, 1.0, 1.000001, 2.5, 100.25}

var sweepDiscreteModelings = []controlsys.C2DDelayModeling{controlsys.C2DDelayModelingState, controlsys.C2DDelayModelingInternal}

var sweepDiscreteReps = []string{"input", "output", "iodelay", "internal"}

func sweepDiscreteMat(r, c int, data ...float64) *mat.Dense {
	return mat.NewDense(r, c, data)
}

// sweepDiscreteLag builds e^{-Ls}/(τs+1) with the delay in representation rep.
func sweepDiscreteLag(t *testing.T, tau, delay float64, rep string) *controlsys.System {
	t.Helper()
	b := 1 / tau
	if rep == "internal" {
		b = 0
	}
	sys, err := controlsys.New(sweepDiscreteMat(1, 1, -1/tau), sweepDiscreteMat(1, 1, b), sweepDiscreteMat(1, 1, 1), sweepDiscreteMat(1, 1, 0), 0)
	if err != nil {
		t.Fatal(err)
	}
	switch rep {
	case "input":
		err = sys.SetInputDelay([]float64{delay})
	case "output":
		err = sys.SetOutputDelay([]float64{delay})
	case "iodelay":
		err = sys.SetDelay(sweepDiscreteMat(1, 1, delay))
	case "internal":
		err = sys.SetInternalDelay([]float64{delay}, sweepDiscreteMat(1, 1, 1/tau), sweepDiscreteMat(1, 1, 0), sweepDiscreteMat(1, 1, 0), sweepDiscreteMat(1, 1, 1), sweepDiscreteMat(1, 1, 0))
	default:
		t.Fatalf("unknown rep %q", rep)
	}
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

func sweepDiscreteLagStep(tau, delay, tk float64) float64 {
	if tk <= delay {
		return 0
	}
	return 1 - math.Exp(-(tk-delay)/tau)
}

func sweepDiscreteRequireIntegerDelays(t *testing.T, label string, sys *controlsys.System) {
	t.Helper()
	check := func(field string, v float64) {
		if v != math.Trunc(v) || v < 0 {
			t.Fatalf("%s: %s delay %v is not a whole sample count", label, field, v)
		}
	}
	for _, v := range sys.InputDelay {
		check("input", v)
	}
	for _, v := range sys.OutputDelay {
		check("output", v)
	}
	if sys.Delay != nil {
		r, c := sys.Delay.Dims()
		for i := range r {
			for j := range c {
				check("io", sys.Delay.At(i, j))
			}
		}
	}
	if sys.LFT != nil {
		for _, v := range sys.LFT.Tau {
			check("internal", v)
		}
	}
}

func sweepDiscreteGrid(steps int, dt float64) []float64 {
	tv := make([]float64, steps)
	for k := range tv {
		tv[k] = float64(k) * dt
	}
	return tv
}

// sweepDiscreteLsim returns Y as p×steps regardless of the response layout.
func sweepDiscreteLsim(t *testing.T, sys *controlsys.System, u *mat.Dense, dt float64) *mat.Dense {
	t.Helper()
	steps, _ := u.Dims()
	resp, err := controlsys.Lsim(sys, u, sweepDiscreteGrid(steps, dt), nil)
	if err != nil {
		t.Fatal(err)
	}
	_, _, p := sys.Dims()
	r, c := resp.Y.Dims()
	if r == p && c == steps {
		return resp.Y
	}
	if r == steps && c == p {
		return mat.DenseCopyOf(resp.Y.T())
	}
	t.Fatalf("Lsim Y dims %dx%d, want %dx%d", r, c, p, steps)
	return nil
}

func sweepDiscreteOnes(steps, m, col int) *mat.Dense {
	u := mat.NewDense(steps, m, nil)
	for k := range steps {
		u.Set(k, col, 1)
	}
	return u
}

// TestSweepDiscreteZOHLagStepGrid covers C-01: exact ZOH C2D of e^{-Ls}/(τs+1)
// for input, output, I/O and internal delay representations under both
// DelayModeling options; the sampled step must equal the closed form to 1e-12
// (no silent rounding) and every discrete delay must be a whole sample count.
func TestSweepDiscreteZOHLagStepGrid(t *testing.T) {
	for _, tau := range []float64{1, 0.3} {
		for _, ratio := range sweepDiscreteRatios {
			delay := ratio * sweepDiscreteTs
			for _, rep := range sweepDiscreteReps {
				if rep == "internal" && delay == 0 {
					continue
				}
				for _, modeling := range sweepDiscreteModelings {
					label := fmt.Sprintf("tau=%g L/Ts=%g rep=%s modeling=%s", tau, ratio, rep, modeling)
					sys := sweepDiscreteLag(t, tau, delay, rep)
					disc, err := sys.C2D(sweepDiscreteTs, controlsys.C2DOptions{Method: controlsys.C2DMethodZOH, DelayModeling: modeling})
					if err != nil {
						t.Fatalf("%s: %v", label, err)
					}
					if disc.Dt != sweepDiscreteTs {
						t.Fatalf("%s: Dt=%g", label, disc.Dt)
					}
					sweepDiscreteRequireIntegerDelays(t, label, disc)
					if modeling == controlsys.C2DDelayModelingState && rep != "internal" && disc.HasInternalDelay() {
						t.Fatalf("%s: state modeling left internal delays %v", label, disc.LFT.Tau)
					}
					steps := int(ratio) + 40
					y := sweepDiscreteLsim(t, disc, sweepDiscreteOnes(steps, 1, 0), sweepDiscreteTs)
					for k := range steps {
						want := sweepDiscreteLagStep(tau, delay, float64(k)*sweepDiscreteTs)
						if got := y.At(0, k); math.Abs(got-want) > 1e-12 {
							t.Fatalf("%s: y[%d]=%.17g want %.17g", label, k, got, want)
						}
					}
				}
			}
		}
	}
}

// sweepDiscreteHeldLagResponse samples x' = (u(t-L)-x)/τ, y = x, for u held on
// [kTs,(k+1)Ts), integrating each constant segment of the delayed input exactly.
func sweepDiscreteHeldLagResponse(u []float64, tau, delay, dt float64) []float64 {
	steps := len(u)
	y := make([]float64, steps)
	x, now := 0.0, 0.0
	advance := func(to float64) {
		for now < to {
			value, next := 0.0, to
			if s := now - delay; s >= -1e-15 {
				k := int(math.Floor(s/dt + 1e-9))
				value = u[min(k, steps-1)]
				next = math.Min(to, float64(k+1)*dt+delay)
			} else {
				next = math.Min(to, delay)
			}
			x = value + (x-value)*math.Exp(-(next-now)/tau)
			now = next
		}
	}
	for k := range steps {
		advance(float64(k) * dt)
		y[k] = x
	}
	return y
}

// TestSweepDiscreteZOHHeldRandomInput covers C-02: a seeded random input held
// by ZOH and applied to the continuous delayed lag (exact piecewise analytic
// reference) matches Lsim of the C2D model to 1e-12.
func TestSweepDiscreteZOHHeldRandomInput(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 11))
	const steps = 80
	data := make([]float64, steps)
	for k := range data {
		data[k] = 2*rng.Float64() - 1
	}
	for _, tau := range []float64{1, 0.3} {
		for _, ratio := range []float64{0.001, 0.5, 0.999999, 1.000001, 2.5, 7.3} {
			delay := ratio * sweepDiscreteTs
			want := sweepDiscreteHeldLagResponse(data, tau, delay, sweepDiscreteTs)
			for _, rep := range []string{"input", "output", "internal"} {
				for _, modeling := range sweepDiscreteModelings {
					label := fmt.Sprintf("tau=%g L/Ts=%g rep=%s modeling=%s", tau, ratio, rep, modeling)
					disc, err := sweepDiscreteLag(t, tau, delay, rep).C2D(sweepDiscreteTs, controlsys.C2DOptions{DelayModeling: modeling})
					if err != nil {
						t.Fatalf("%s: %v", label, err)
					}
					y := sweepDiscreteLsim(t, disc, mat.NewDense(steps, 1, data), sweepDiscreteTs)
					for k := range steps {
						if got := y.At(0, k); math.Abs(got-want[k]) > 1e-12 {
							t.Fatalf("%s: y[%d]=%.17g want %.17g", label, k, got, want[k])
						}
					}
				}
			}
		}
	}
}

// TestSweepDiscreteZOHMIMOFractionalInputDelays covers C-03: a 2x2 model with
// non-symmetric A (distinct real eigenvalues -2±√2) and fractional per-channel
// input delays; C2D ZOH step equals the hand-diagonalised closed form to 1e-11.
func TestSweepDiscreteZOHMIMOFractionalInputDelays(t *testing.T) {
	a := []float64{-1, 2, 0.5, -3}
	b := []float64{1, 0.5, -0.3, 1}
	c := []float64{1, -0.4, 0.2, 1}
	d := []float64{0.1, 0, 0, -0.2}
	inputDelay := []float64{0.25, 0.37}
	l1, l2 := -2+math.Sqrt2, -2-math.Sqrt2
	integral := func(s float64) [4]float64 {
		if s <= 0 {
			return [4]float64{}
		}
		f1, f2 := (math.Exp(l1*s)-1)/l1, (math.Exp(l2*s)-1)/l2
		var out [4]float64
		for i := range 2 {
			for j := range 2 {
				eye := 0.0
				if i == j {
					eye = 1
				}
				out[i*2+j] = (f1*(a[i*2+j]-l2*eye) - f2*(a[i*2+j]-l1*eye)) / (l1 - l2)
			}
		}
		return out
	}
	stepRef := func(i, j int, tk float64) float64 {
		s := tk - inputDelay[j]
		if s <= 0 {
			return 0
		}
		phi := integral(s)
		v := d[i*2+j]
		for l := range 2 {
			v += c[i*2+l] * (phi[l*2]*b[j] + phi[l*2+1]*b[2+j])
		}
		return v
	}
	sys, err := controlsys.New(sweepDiscreteMat(2, 2, a...), sweepDiscreteMat(2, 2, b...), sweepDiscreteMat(2, 2, c...), sweepDiscreteMat(2, 2, d...), 0)
	if err != nil {
		t.Fatal(err)
	}
	if err := sys.SetInputDelay(inputDelay); err != nil {
		t.Fatal(err)
	}
	for _, modeling := range sweepDiscreteModelings {
		disc, err := sys.C2D(sweepDiscreteTs, controlsys.C2DOptions{DelayModeling: modeling})
		if err != nil {
			t.Fatal(err)
		}
		sweepDiscreteRequireIntegerDelays(t, string(modeling), disc)
		const steps = 60
		for j := range 2 {
			y := sweepDiscreteLsim(t, disc, sweepDiscreteOnes(steps, 2, j), sweepDiscreteTs)
			for i := range 2 {
				for k := range steps {
					want := stepRef(i, j, float64(k)*sweepDiscreteTs)
					if got := y.At(i, k); math.Abs(got-want) > 1e-11 {
						t.Fatalf("%s: u%d->y%d k=%d got %.17g want %.17g", modeling, j, i, k, got, want)
					}
				}
			}
		}
	}
}

func sweepDiscreteTotalDelays(sys *controlsys.System) (input, output, io, internal float64) {
	for _, v := range sys.InputDelay {
		input += v
	}
	for _, v := range sys.OutputDelay {
		output += v
	}
	if sys.Delay != nil {
		io = sys.Delay.At(0, 0)
	}
	if sys.LFT != nil {
		for _, v := range sys.LFT.Tau {
			internal += v
		}
	}
	return
}

// TestSweepDiscreteD2CRecoversDelays covers C-04: D2C(C2D(G)) for whole-sample
// delays recovers the delay in seconds, the pole, and e^{-jωL}/(jωτ+1) to
// 1e-9; for fractional delays (incl. a fractional internal loop delay) D2C
// can only yield whole-sample delays, keeps DC gain, and C2D(D2C(sysd))
// reproduces sysd's step to 1e-10.
func TestSweepDiscreteD2CRecoversDelays(t *testing.T) {
	const tau = 1.0
	for _, rep := range []string{"input", "output", "iodelay", "internal"} {
		sys := sweepDiscreteLag(t, tau, 0.3, rep)
		disc, err := sys.C2D(sweepDiscreteTs, controlsys.C2DOptions{})
		if err != nil {
			t.Fatal(err)
		}
		back, err := disc.D2C(controlsys.D2COptions{})
		if err != nil {
			t.Fatalf("%s: %v", rep, err)
		}
		in, out, io, internal := sweepDiscreteTotalDelays(back)
		if total := in + out + io + internal; math.Abs(total-0.3) > 1e-12 {
			t.Fatalf("%s: recovered delays in=%g out=%g io=%g internal=%g, want total 0.3", rep, in, out, io, internal)
		}
		if rep == "internal" && math.Abs(internal-0.3) > 1e-12 {
			t.Fatalf("internal delay not recovered as internal: %+v", back.LFT)
		}
		for _, w := range []float64{0.05, 0.5, 2, 10} {
			got, err := back.EvalFr(complex(0, w))
			if err != nil {
				t.Fatal(err)
			}
			want := cmplx.Exp(complex(0, -w*0.3)) / complex(1, w*tau)
			if cmplx.Abs(got[0][0]-want) > 1e-9 {
				t.Fatalf("%s w=%g: got %v want %v", rep, w, got[0][0], want)
			}
		}
	}

	loop, err := controlsys.Feedback(sweepDiscreteLag(t, tau, 0.25, "input"), nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	cases := map[string]*controlsys.System{
		"fractional-input":    sweepDiscreteLag(t, tau, 0.25, "input"),
		"fractional-internal": sweepDiscreteLag(t, tau, 0.25, "internal"),
		"fractional-loop":     loop,
	}
	for name, sys := range cases {
		dcWant := 1.0
		if name == "fractional-loop" {
			dcWant = 0.5
		}
		disc, err := sys.C2D(sweepDiscreteTs, controlsys.C2DOptions{})
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		back, err := disc.D2C(controlsys.D2COptions{})
		if err != nil {
			t.Fatalf("%s: D2C: %v", name, err)
		}
		for _, v := range append(append([]float64{}, back.InputDelay...), back.OutputDelay...) {
			if r := v / sweepDiscreteTs; math.Abs(r-math.Round(r)) > 1e-9 {
				t.Fatalf("%s: D2C external delay %g not a sample multiple", name, v)
			}
		}
		if back.LFT != nil {
			for _, v := range back.LFT.Tau {
				if r := v / sweepDiscreteTs; math.Abs(r-math.Round(r)) > 1e-9 {
					t.Fatalf("%s: D2C internal delay %g not a sample multiple", name, v)
				}
			}
		}
		dc, err := back.EvalFr(0)
		if err != nil {
			t.Fatal(err)
		}
		if cmplx.Abs(dc[0][0]-complex(dcWant, 0)) > 1e-10 {
			t.Fatalf("%s: D2C DC gain %v want %g", name, dc[0][0], dcWant)
		}
		again, err := back.C2D(sweepDiscreteTs, controlsys.C2DOptions{})
		if err != nil {
			t.Fatalf("%s: re-C2D: %v", name, err)
		}
		const steps = 50
		y0 := sweepDiscreteLsim(t, disc, sweepDiscreteOnes(steps, 1, 0), sweepDiscreteTs)
		y1 := sweepDiscreteLsim(t, again, sweepDiscreteOnes(steps, 1, 0), sweepDiscreteTs)
		for k := range steps {
			if math.Abs(y0.At(0, k)-y1.At(0, k)) > 1e-10 {
				t.Fatalf("%s: C2D(D2C) step k=%d got %.15g want %.15g", name, k, y1.At(0, k), y0.At(0, k))
			}
		}
	}
}

// TestSweepDiscreteD2DResamplingDelays covers C-05: D2D ZOH of the exact
// sampled lag with a 3-sample delay at Ts=0.1. Ts=0.05 must give 6 whole
// samples and the exact step; incompatible rates (0.07, 0.2) must either
// return an explicit error or a whole-sample model whose step is the exact
// sampled step of e^{-0.3s}/(s+1) (no silent rounding), to 1e-10.
func TestSweepDiscreteD2DResamplingDelays(t *testing.T) {
	a := math.Exp(-sweepDiscreteTs)
	for _, rep := range []string{"input", "output", "internal"} {
		sys, err := controlsys.New(sweepDiscreteMat(1, 1, a), sweepDiscreteMat(1, 1, 1-a), sweepDiscreteMat(1, 1, 1), sweepDiscreteMat(1, 1, 0), sweepDiscreteTs)
		if err != nil {
			t.Fatal(err)
		}
		switch rep {
		case "input":
			err = sys.SetInputDelay([]float64{3})
		case "output":
			err = sys.SetOutputDelay([]float64{3})
		case "internal":
			sys.B = sweepDiscreteMat(1, 1, 0)
			err = sys.SetInternalDelay([]float64{3}, sweepDiscreteMat(1, 1, 1-a), sweepDiscreteMat(1, 1, 0), sweepDiscreteMat(1, 1, 0), sweepDiscreteMat(1, 1, 1), sweepDiscreteMat(1, 1, 0))
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, dt := range []float64{0.05, 0.07, 0.2} {
			label := fmt.Sprintf("rep=%s dt=%g", rep, dt)
			out, err := sys.D2D(dt, controlsys.D2DOptions{})
			if err != nil {
				if dt == 0.05 {
					t.Fatalf("%s: compatible resampling failed: %v", label, err)
				}
				t.Logf("%s: explicit error %v", label, err)
				continue
			}
			sweepDiscreteRequireIntegerDelays(t, label, out)
			var internal []float64
			if out.LFT != nil {
				internal = out.LFT.Tau
			}
			t.Logf("%s: resampled exactly, delays in=%v out=%v internal=%v", label, out.InputDelay, out.OutputDelay, internal)
			if dt == 0.05 {
				in, o, io, internal := sweepDiscreteTotalDelays(out)
				if in+o+io+internal != 6 {
					t.Fatalf("%s: delays in=%g out=%g io=%g internal=%g, want 6 samples", label, in, o, io, internal)
				}
			}
			steps := int(1.5 / dt)
			y := sweepDiscreteLsim(t, out, sweepDiscreteOnes(steps, 1, 0), dt)
			for k := range steps {
				want := sweepDiscreteLagStep(1, 0.3, float64(k)*dt)
				if got := y.At(0, k); math.Abs(got-want) > 1e-10 {
					t.Fatalf("%s: y[%d]=%.15g want %.15g", label, k, got, want)
				}
			}
		}
	}
}

func sweepDiscreteLagPulse(tau, delay, dt float64, steps int) []float64 {
	g := make([]float64, steps)
	for k := 1; k < steps; k++ {
		g[k] = sweepDiscreteLagStep(tau, delay, float64(k)*dt) - sweepDiscreteLagStep(tau, delay, float64(k-1)*dt)
	}
	return g
}

// sweepDiscreteFineLoop integrates y' = -y + u(t-L) - y(t-L), u(t) = 1 for t ≥ 0,
// by an exponential integrator with linear history interpolation on step h.
func sweepDiscreteFineLoop(delay, h float64, steps int, sampleEvery int) []float64 {
	lag := int(math.Round(delay / h))
	total := steps * sampleEvery
	y := make([]float64, total+1)
	e := math.Exp(-h)
	w0, w1 := 1-e, (h-1+e)/h
	for n := range total {
		f0, f1 := 0.0, 0.0
		if n-lag >= 0 {
			f0 = 1 - y[n-lag]
		}
		if n+1-lag >= 0 {
			f1 = 1 - y[n+1-lag]
		}
		if n-lag == 0 {
			f0 = 1
		}
		y[n+1] = e*y[n] + f0*w0 + (f1-f0)*w1
	}
	out := make([]float64, steps)
	for k := range steps {
		out[k] = y[k*sampleEvery]
	}
	return out
}

// TestSweepDiscreteZOHInternalFeedbackLoop covers C-06: C2D ZOH of
// Feedback(e^{-0.25s}/(s+1), 1) at Ts ∈ {0.05, 0.1, 0.07}. The c2d contract is
// the sampled loop feedback(c2d(G),1) (MATLAB discretizes the LFT pieces; the
// fractional loop-delay residual is a held input delay), checked to 1e-12 by
// a hand convolution recursion with the exact ZOH pulse response of G. Versus
// the continuous loop driven by a held step, the result is only an O(Ts)
// approximation because the fed-back y is not piecewise constant; that
// deviation is bounded by 0.25·Ts.
func TestSweepDiscreteZOHInternalFeedbackLoop(t *testing.T) {
	const delay = 0.25
	plant := sweepDiscreteLag(t, 1, delay, "input")
	loop, err := controlsys.Feedback(plant, nil, -1)
	if err != nil {
		t.Fatal(err)
	}
	if !loop.HasInternalDelay() {
		t.Fatal("continuous delayed feedback must hold an internal delay")
	}
	for _, dt := range []float64{0.05, 0.1, 0.07} {
		const horizon = 8.0
		steps := int(horizon / dt)
		g := sweepDiscreteLagPulse(1, delay, dt, steps)
		want := make([]float64, steps)
		for k := range steps {
			for j := 1; j <= k; j++ {
				want[k] += g[j] * (1 - want[k-j])
			}
		}
		sampleEvery := int(math.Round(dt / 0.0005))
		fine := sweepDiscreteFineLoop(delay, 0.0005, steps, sampleEvery)
		for _, modeling := range sweepDiscreteModelings {
			label := fmt.Sprintf("dt=%g modeling=%s", dt, modeling)
			disc, err := loop.C2D(dt, controlsys.C2DOptions{DelayModeling: modeling})
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			sweepDiscreteRequireIntegerDelays(t, label, disc)
			y := sweepDiscreteLsim(t, disc, sweepDiscreteOnes(steps, 1, 0), dt)
			worst := 0.0
			for k := range steps {
				got := y.At(0, k)
				if math.Abs(got-want[k]) > 1e-12 {
					t.Fatalf("%s: y[%d]=%.17g want sampled-loop %.17g", label, k, got, want[k])
				}
				worst = math.Max(worst, math.Abs(got-fine[k]))
			}
			if worst > 0.25*dt {
				t.Fatalf("%s: deviation from continuous loop %g exceeds O(Ts) bound %g", label, worst, 0.25*dt)
			}
			t.Logf("%s: max |c2d loop - continuous loop| = %.3g", label, worst)
		}
	}
}

func sweepDiscreteRandom(seed uint64, steps int) []float64 {
	rng := rand.New(rand.NewPCG(seed, 3))
	u := make([]float64, steps)
	for k := range u {
		u[k] = 2*rng.Float64() - 1
	}
	return u
}

func sweepDiscretePlant(t *testing.T) *controlsys.System {
	t.Helper()
	sys, err := controlsys.New(sweepDiscreteMat(1, 1, 0.8), sweepDiscreteMat(1, 1, 0.5), sweepDiscreteMat(1, 1, 1), sweepDiscreteMat(1, 1, 0.3), sweepDiscreteTs)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// TestSweepDiscreteDelayedFeedbackRecursion covers D06/D07 discrete: negative
// Feedback of G(z) = z^{-k}(0.5/(z-0.8)+0.3) with a static or one-sample-delayed
// controller equals a hand-coded difference recursion on a random input, and
// the internal-delay closed loop equals its AbsorbDelay state augmentation, both
// to 1e-12.
func TestSweepDiscreteDelayedFeedbackRecursion(t *testing.T) {
	const steps = 120
	u := sweepDiscreteRandom(5, steps)
	for _, place := range []string{"input", "output", "iodelay"} {
		for _, ctrl := range []string{"static", "delayed"} {
			label := fmt.Sprintf("plant=%s controller=%s", place, ctrl)
			plant := sweepDiscretePlant(t)
			var err error
			switch place {
			case "input":
				err = plant.SetInputDelay([]float64{3})
			case "output":
				err = plant.SetOutputDelay([]float64{3})
			case "iodelay":
				err = plant.SetDelay(sweepDiscreteMat(1, 1, 3))
			}
			if err != nil {
				t.Fatal(err)
			}
			gain := 0.5
			ctrlDelay := 0
			if ctrl == "delayed" {
				gain, ctrlDelay = 0.4, 1
			}
			controller, err := controlsys.NewGain(sweepDiscreteMat(1, 1, gain), sweepDiscreteTs)
			if err != nil {
				t.Fatal(err)
			}
			if ctrlDelay > 0 {
				if err := controller.SetInputDelay([]float64{float64(ctrlDelay)}); err != nil {
					t.Fatal(err)
				}
			}
			cl, err := controlsys.Feedback(plant, controller, -1)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			want := make([]float64, steps)
			v := make([]float64, steps)
			q := make([]float64, steps)
			x := 0.0
			for k := range steps {
				vd := 0.0
				if k >= 3 {
					vd = v[k-3]
				}
				if place == "output" {
					if k >= 3 {
						want[k] = q[k-3]
					}
				} else {
					want[k] = x + 0.3*vd
				}
				fb := 0.0
				if k-ctrlDelay >= 0 {
					fb = gain * want[k-ctrlDelay]
				}
				v[k] = u[k] - fb
				if place == "output" {
					q[k] = x + 0.3*v[k]
					x = 0.8*x + 0.5*v[k]
				} else {
					x = 0.8*x + 0.5*vd
				}
			}
			resp, err := cl.Simulate(mat.NewDense(1, steps, u), nil, nil)
			if err != nil {
				t.Fatalf("%s: %v", label, err)
			}
			absorbed, err := cl.AbsorbDelay()
			if err != nil {
				t.Fatalf("%s: AbsorbDelay: %v", label, err)
			}
			if absorbed.HasDelay() {
				t.Fatalf("%s: AbsorbDelay left delays", label)
			}
			flat, err := absorbed.Simulate(mat.NewDense(1, steps, u), nil, nil)
			if err != nil {
				t.Fatal(err)
			}
			for k := range steps {
				if got := resp.Y.At(0, k); math.Abs(got-want[k]) > 1e-12 {
					t.Fatalf("%s: y[%d]=%.17g want recursion %.17g", label, k, got, want[k])
				}
				if got := flat.Y.At(0, k); math.Abs(got-want[k]) > 1e-12 {
					t.Fatalf("%s: absorbed y[%d]=%.17g want %.17g", label, k, got, want[k])
				}
			}
		}
	}
}

func sweepDiscreteThiran(d float64, order int) (num, den []float64) {
	den = make([]float64, order+1)
	for k := 0; k <= order; k++ {
		binom := 1.0
		for i := range k {
			binom = binom * float64(order-i) / float64(i+1)
		}
		prod := 1.0
		for i := 0; i <= order; i++ {
			prod *= (d - float64(order) + float64(i)) / (d - float64(order) + float64(k) + float64(i))
		}
		den[k] = math.Pow(-1, float64(k)) * binom * prod
	}
	num = make([]float64, order+1)
	for k := range num {
		num[k] = den[order-k]
	}
	return num, den
}

func sweepDiscretePolyInv(coef []float64, z complex128) complex128 {
	var v complex128
	for k, c := range coef {
		v += complex(c, 0) * cmplx.Pow(z, complex(-float64(k), 0))
	}
	return v
}

// TestSweepDiscreteThiranFractionalDelay covers C-07: Tustin C2D of
// e^{-0.35s}/(s+1) at Ts=0.1 with ThiranOrder 3 equals z^{-1}·Thiran(2.5, 3)·
// Tustin(1/(s+1)) from the closed-form Thiran coefficients, to 1e-10, under
// both DelayModeling options. ThiranOrder with ZOH is an explicit
// ErrInvalidConversionOptions (MATLAB ignores FractDelayApproxOrder for zoh,
// which already handles the fraction exactly).
func TestSweepDiscreteThiranFractionalDelay(t *testing.T) {
	const dt = 0.1
	num, den := sweepDiscreteThiran(2.5, 3)
	for _, rep := range []string{"input", "output"} {
		sys := sweepDiscreteLag(t, 1, 0.35, rep)
		if _, err := sys.C2D(dt, controlsys.C2DOptions{Method: controlsys.C2DMethodZOH, ThiranOrder: 3}); !errors.Is(err, controlsys.ErrInvalidConversionOptions) {
			t.Fatalf("%s: ZOH+Thiran err = %v, want ErrInvalidConversionOptions", rep, err)
		}
		for _, modeling := range sweepDiscreteModelings {
			disc, err := sys.C2D(dt, controlsys.C2DOptions{Method: controlsys.C2DMethodTustin, ThiranOrder: 3, DelayModeling: modeling})
			if err != nil {
				t.Fatalf("%s %s: %v", rep, modeling, err)
			}
			sweepDiscreteRequireIntegerDelays(t, string(modeling), disc)
			for _, theta := range []float64{0.05, 0.4, 1.3, 2.8} {
				z := cmplx.Exp(complex(0, theta))
				tustin := (z + 1) / (complex(2/dt, 0)*(z-1) + (z + 1))
				want := tustin / z * sweepDiscretePolyInv(num, z) / sweepDiscretePolyInv(den, z)
				got, err := disc.EvalFr(z)
				if err != nil {
					t.Fatal(err)
				}
				if cmplx.Abs(got[0][0]-want) > 1e-10 {
					t.Fatalf("%s %s theta=%g: got %v want %v", rep, modeling, theta, got[0][0], want)
				}
			}
		}
	}
}
