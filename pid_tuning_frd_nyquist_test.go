package controlsys

import (
	"context"
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// pidRandomStableTest returns a random stable plant num/den of order 1..3
// with poles (continuous: in the left half plane with magnitudes in
// [mLo, mHi]; discrete: inside |z| < 0.95) and zeros of either side.
func pidRandomStableTest(rng *rand.Rand, discrete bool, mLo, mHi float64) (num, den []float64) {
	root := func(stable bool) []float64 {
		if discrete {
			r := .95 * rng.Float64()
			if !stable {
				r = 2 * rng.Float64()
			}
			if rng.IntN(2) == 0 {
				return []float64{1, -r * math.Cos(math.Pi*rng.Float64())}
			}
			th := math.Pi * rng.Float64()
			return []float64{1, -2 * r * math.Cos(th), r * r}
		}
		m := math.Exp(math.Log(mLo) + rng.Float64()*math.Log(mHi/mLo))
		sign := 1.
		if !stable && rng.IntN(2) == 0 {
			sign = -1
		}
		if rng.IntN(2) == 0 {
			return []float64{1, sign * m}
		}
		z := .1 + .9*rng.Float64()
		return []float64{1, sign * 2 * z * m, m * m}
	}
	den = []float64{1}
	for len(den) < 2+rng.IntN(3) {
		den = polyMulTest(den, root(true))
	}
	num = []float64{math.Exp(4 * (rng.Float64() - .5))}
	if rng.IntN(2) == 0 {
		num[0] = -num[0]
	}
	if len(den) > 2 && rng.IntN(2) == 0 {
		num = polyMulTest(num, root(false)[:2])
	}
	return num, den
}

// pidRandomControllerTest returns a random PID2 of random gain signs and
// scales, filtered and discrete formulas as a tuning candidate may have
// them: a discrete forward-Euler filter pole stays inside the unit circle.
func pidRandomControllerTest(rng *rand.Rand, dt float64) PID2 {
	gain := func() float64 {
		if rng.IntN(3) == 0 {
			return 0
		}
		g := math.Exp(6 * (rng.Float64() - .5))
		if rng.IntN(2) == 0 {
			g = -g
		}
		return g
	}
	c := PID2{Kp: gain(), Ki: gain(), Kd: gain(), B: 1, C: 1, Dt: dt}
	if c.Kd != 0 && rng.IntN(2) == 0 {
		c.Tf = math.Exp(4 * (rng.Float64() - .5))
	}
	if dt > 0 {
		c.IFormula, c.DFormula = PIDFormula(rng.IntN(3)), PIDFormula(rng.IntN(3))
		if c.Kd != 0 && (c.DFormula == Trapezoidal || c.DFormula == ForwardEuler) {
			c.Tf = math.Max(c.Tf, .51*dt+rng.Float64()*dt)
		}
	}
	return c
}

// The sampled Nyquist count matches closed-loop roots of the exact loop on
// random stable plants and random PID2 controllers: continuous by
// companion eigenvalues (pidClosedLoopMaxRe), discrete by exact Schur–Cohn
// (pidtuneExactSchur). A loop within 1e-6 of the stability boundary, or
// one the samples do not resolve, is not compared: its dense |S| exceeds
// 1e3, or between two samples the interpolant's error in L exceeds half
// the distance of the exact 1 + L from 0. Where 1 + L ends unresolved, near
// 0 at the top sample, the count may reject a stable loop of |S| > 20.
func TestPIDTuningFRDNyquistMatchesClosedLoopRoots(t *testing.T) {
	rng := rand.New(rand.NewPCG(7, 13))
	counts := map[string]int{}
	for trial := range 3000 {
		discrete := trial%2 == 1
		dt, lo, hi := 0., 1e-3, 1e3
		if discrete {
			dt = []float64{.1, 1}[rng.IntN(2)]
			hi = .99 * math.Pi / dt
		}
		num, den := pidRandomStableTest(rng, discrete, 1e-2, 1e2)
		g := func(w float64) complex128 {
			x := complex(0, w)
			if discrete {
				x = cmplx.Exp(complex(0, w*dt))
			}
			return pidPolyAt(num, x) / pidPolyAt(den, x)
		}
		n := []int{100, 300}[rng.IntN(2)]
		omega := make([]float64, n)
		values := make([][][]complex128, n)
		for k := range omega {
			omega[k] = math.Exp(math.Log(lo) + float64(k)*math.Log(hi/lo)/float64(n-1))
			values[k] = [][]complex128{{g(omega[k])}}
		}
		frd, err := NewFRD(values, omega, dt)
		if err != nil {
			t.Fatal(err)
		}
		p := pidTuningFRDPlant(frd, nil)
		if p.dcGain == 0 {
			continue
		}
		for range 4 {
			c := pidRandomControllerTest(rng, dt)
			stable, known := pidTuningFRDNyquistCount(c, p)
			if !known {
				counts["unknown"]++
				continue
			}
			var truth bool
			if discrete {
				pid := &PID{IFormula: c.IFormula, DFormula: c.DFormula, Kp: c.Kp, Ki: c.Ki, Kd: c.Kd, Tf: c.Tf, Dt: dt}
				truth = pidtuneExactSchur(pid, num, den, dt)
				if pid.IFormula == ForwardEuler && (pidtuneExactSchurStableRadius(t, pid, num, den, dt, 1-1e-6) != truth || pidtuneExactSchurStableRadius(t, pid, num, den, dt, 1+1e-6) != truth) {
					counts["marginal"]++
					continue
				}
			} else {
				re := pidClosedLoopMaxRe(num, den, c)
				if math.Abs(re) < 1e-6 {
					counts["marginal"]++
					continue
				}
				truth = re < 0
			}
			if stable == truth {
				counts[fmt.Sprint("agree ", truth)]++
				continue
			}
			peak, coarse := 0., false
			for w := lo * 1e-3; w <= hi; w *= 1.001 {
				peak = math.Max(peak, 1/cmplx.Abs(1+g(w)*pidTuningFeedback(c, w)))
			}
			for k := 1; k < n; k++ {
				for j := 1; j < 256; j++ {
					w := omega[k-1] * math.Pow(omega[k]/omega[k-1], float64(j)/256)
					k := pidTuningFeedback(c, w)
					coarse = coarse || cmplx.Abs(k*(g(w)-p.at(w))) > .5*cmplx.Abs(1+k*g(w))
				}
			}
			if peak > 1e3 || coarse {
				counts["unresolved"]++
				continue
			}
			if !stable && peak > 20 {
				counts["conservative"]++
				continue
			}
			t.Errorf("trial %d: %v/%v dt %g %+v: count says stable=%v, roots say %v (|S| peak %g)", trial, num, den, dt, c, stable, truth, peak)
		}
	}
	if counts["agree true"] < 300 || counts["agree false"] < 300 {
		t.Errorf("too few decided loops: %v", counts)
	}
	t.Log(counts)
}

// TunePIDFRD on data of a stable plant (no unstable poles given) returned
// unstable loops whose characteristic polynomial has the same sign at both
// ends, so the DC test passed them: complex right-half-plane pairs and
// pairs of real unstable roots, continuous and discrete. The sampled
// Nyquist count rejects them; each case now errors or returns a stable
// loop. Oracles: companion roots (pidClosedLoopMaxRe), exact Schur–Cohn.
func TestTunePIDFRDNyquistRejectsUnstable(t *testing.T) {
	for i, c := range []struct {
		num, den []float64
		dt, top  float64
		n        int
		family   PidtuneType
		o        PIDTuningOptions
	}{
		{[]float64{-0.08453928824668518, 0.2574695851608255}, []float64{1, 3.55383559509827, 6.460274164969173}, 0, 1e3, 300, PidtunePD, PIDTuningOptions{PhaseMargin: 60}},
		{[]float64{-0.0656377000689616, -0.01066360857208608}, []float64{1, 10.283360851860468, 17.856421377229974}, 0, 1e3, 100, PidtunePD, PIDTuningOptions{PhaseMargin: 45}},
		{[]float64{4.3574105659571165, 0.9572857556693883}, []float64{1, 22.18446797058865, 51.17212458330976}, 0, 1e3, 600, PidtunePD, PIDTuningOptions{PhaseMargin: 30}},
		{[]float64{0.3355294200395295, 0.10987817852881804, -0.004844131095246884}, []float64{1, 7.862167723141873, 13.260352393551456, 0.7241889333781578}, 0, 1e3, 300, PidtunePID, PIDTuningOptions{}},
		{[]float64{11.96462207800562, -2.3659406182814235}, []float64{1, 1.7040182460022628, 0.09440729841430065}, 0, 1e3, 600, PidtunePID, PIDTuningOptions{}},
		{[]float64{0.21329576228258076, -0.022659778889879287}, []float64{1, 9.107333175618313, 0.5042560152064186}, 0, 1e3, 100, PidtunePID, PIDTuningOptions{}},
		{[]float64{1.169886169471443, -0.3351601299654014}, []float64{1, 8.583042549565267, 12.864155054613956}, 0, 1e3, 100, PidtunePDF, PIDTuningOptions{}},
		{[]float64{2}, []float64{1, -0.9}, 0.1, .99 * math.Pi / .1, 400, PidtunePD, PIDTuningOptions{}},
		{[]float64{0.3, 0.1}, []float64{1, -1.2, 0.5}, 0.1, .99 * math.Pi / .1, 400, PidtunePD, PIDTuningOptions{}},
		{[]float64{-0.4}, []float64{1, -1, 0.21}, 0.1, .99 * math.Pi / .1, 400, PidtunePDF, PIDTuningOptions{}},
		{[]float64{-0.2663672599168507}, []float64{1, 0.11429480673807586}, 1, .99 * math.Pi, 300, PidtunePD, PIDTuningOptions{PhaseMargin: 30}},
	} {
		name := fmt.Sprintf("case %d %s", i, c.family)
		frd := pidExactFRD(t, c.num, c.den, 1e-3, c.top, c.n)
		if c.dt > 0 {
			for k, w := range frd.Omega {
				z := cmplx.Exp(complex(0, w*c.dt))
				frd.Response[k][0][0] = pidPolyAt(c.num, z) / pidPolyAt(c.den, z)
			}
			frd.Dt = c.dt
		}
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", name, err)
			}
			continue
		}
		k := *r.Controller
		if c.dt > 0 {
			if !pidtuneExactSchur(&PID{IFormula: k.IFormula, DFormula: k.DFormula, Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt}, c.num, c.den, c.dt) {
				t.Errorf("%s: %+v unstable, reported margin %g", name, k, r.Evidence.AchievedPhaseMargin)
			}
		} else if re := pidClosedLoopMaxRe(c.num, c.den, k); re >= 0 {
			t.Errorf("%s: %+v closed-loop root at Re %g, reported margin %g", name, k, re, r.Evidence.AchievedPhaseMargin)
		}
	}
}
