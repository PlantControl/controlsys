package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// AP6JSB: a large Kd keeps |L| over 1 past the band's top wc·100, so an odd
// count of 0 dB crossings lies above it that the scan never looked at: PID
// PM 30 returned a 15.5° crossing at 5.25 rad/s (floor 27°), delayed PID PM
// 60 a 51.2° one at 110.7, discrete PIDF a 31.1° one at 9.14 (floor 57°),
// PD a 26.3° one at 45.3 and PD PM 70 a 43.5° one at 159 (floor 67°);
// evidence listed none. On a plant with ‖A‖ ≈ 100 and wc 1.8e-8 the tail
// bound has none below ~100 rad/s, so the grid stopped at 1e4·hi and PID
// returned a 14.6° crossing at 0.0417. On 1/(s+a)·wn²/(s²+2ζwn·s+wn²)
// plants with a light resonance above wc·100 the coarse upward walk skipped
// the 0 dB pair at the resonance, so PID chose Kd > 0 with a 19.6–50° pair
// (floors 57/42), evidence listing only wc. Oracle: a dense sweep of L from closed forms up to
// 1e8·wc (π/Dt discrete) for every crossing, its distance from −1 and the
// |S| peak; Routh on the undelayed characteristic, exact Schur–Cohn on the
// discrete one, and for the delayed loop the argument principle on the
// retarded quasi-polynomial Q = s·D + N·(Kd·s² + Kp·s + Ki)·e^{−sτ}: with
// no roots on jω, Δarg Q(jω) over [0, ∞) is π(n − 2Z)/2, n = deg s·D.
func TestTunePIDCrossingAboveBand(t *testing.T) {
	skipSlowUnderRace(t)
	for _, c := range []struct {
		num, den     Poly
		fam          PidtuneType
		pm, tau, dt  float64
		lag          int
		wantFloorMin float64
	}{
		{Poly{0.3475172890151685}, Poly{1, 68.97305246141737, 111.51081680451367, 314.62904029115396, -3.216614328748604}, PidtunePID, 30, 0, 0, 0, 27},
		{Poly{6.24013823353506}, Poly{1, 69.51808814752826, 3.5288398319050645}, PidtunePID, 60, 0.011180378342231274, 0, 0, 57},
		{Poly{-2.796109607219298, 5.593663885306331, -2.7975567338662084}, Poly{1, -2.918405067519246, 2.8369550245882103, -0.918549781024696}, PidtunePIDF, 0, 0, 0.05308576324401414, 2, 57},
		{Poly{3192.2331382806533, 5851.625992116486, -877.4244188893074, 333.241988470858}, Poly{1, 22.613219017484063, 300.08480842823417, 2335.3743882340327, 17335.04620231331, 777.8271144157021, 8.60408782147981}, PidtunePD, 0, 0, 0, 0, 57},
		{Poly{22325.58242167066, 124508.53366421786, -142162.3708506173, 56237.397576648174}, Poly{1, 100.95776592853754, 10305.077053015631, 92467.96236379766, 344493.371039096, 244772.35713592256, 21806.063124087068}, PidtunePD, 70, 0, 0, 0, 67},
		{Poly{0.24849499151690158, 0.0013485275346479704, 0.0002090434235708211}, Poly{1, 12.667867466748863, 2206.0447677301145, 11594.580723295541, 0}, PidtunePID, 0, 0, 0, 0, 57},
		{Poly{0.21467032679905076}, Poly{1, 0.03883985469722036, 8.813883266103884, 0.15326452402747637}, PidtunePID, 0, 0, 0, 0, 57},
		{Poly{4075438442.4872518}, Poly{1, 117.0569186253515, 1504139329.2863033, 9783782999.809948}, PidtunePID, 0, 0, 0, 0, 57},
		{Poly{2.1287074056182735}, Poly{1, 0.16558866802656774, 355.35147111975954, 17.797980704545058}, PidtunePID, 0, 0, 0, 0, 57},
		{Poly{59021969576.8529}, Poly{1, 231.91684178910205, 2139037576.7899263, 21319430144.2219}, PidtunePID, 45, 0, 0, 0, 42},
	} {
		tf := &TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: c.dt}
		res, err := tf.StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		plant := res.Sys
		if c.tau != 0 || c.lag != 0 {
			if err := plant.SetInputDelay([]float64{c.tau + float64(c.lag)}); err != nil {
				t.Fatal(err)
			}
		}
		r, err := TunePID(context.Background(), plant, c.fam, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			t.Fatalf("%v %s: %v", c.num, c.fam, err)
		}
		k := r.Controller
		if k.IFormula != ForwardEuler || k.DFormula != ForwardEuler {
			t.Fatalf("%v %s: oracle needs forward Euler, got %v", c.num, c.fam, k)
		}
		undelayed := pidtuneUndelayedLoop(k, c.num, c.den, c.dt)
		loop := func(w float64) complex128 {
			return undelayed(w) * cmplx.Exp(complex(0, -w*(c.tau+float64(c.lag)*c.dt)))
		}
		wc := r.Evidence.AchievedCrossover
		lo, hi, n := 1e-6*wc, 1e8*wc, 200000
		if c.dt > 0 {
			hi, n = math.Pi/c.dt, 40000
		}
		peak, crossings := 0., 0
		prev := cmplx.Abs(loop(lo)) - 1
		for i := 1; i <= n; i++ {
			w := lo * math.Pow(hi/lo, float64(i)/float64(n))
			l := loop(w)
			peak = math.Max(peak, 1/cmplx.Abs(1+l))
			d := cmplx.Abs(l) - 1
			if math.Signbit(d) != math.Signbit(prev) {
				crossings++
				if m := 180 - math.Abs(cmplx.Phase(l)*180/math.Pi); m < c.wantFloorMin-.1 {
					t.Errorf("%v %s: 0 dB crossing at %.4g rad/s %.3g° from −1, floor %g", c.num, c.fam, w, m, c.wantFloorMin)
				}
				found := false
				for _, x := range r.Evidence.GainCrossovers {
					found = found || math.Abs(math.Log(x/w)) < .01
				}
				if !found {
					t.Errorf("%v %s: crossing at %.4g rad/s not in evidence %v", c.num, c.fam, w, r.Evidence.GainCrossovers)
				}
			}
			prev = d
		}
		if crossings != len(r.Evidence.GainCrossovers) {
			t.Errorf("%v %s: %d crossings, evidence %v", c.num, c.fam, crossings, r.Evidence.GainCrossovers)
		}
		if peak > 4 {
			t.Errorf("%v %s: |S| peak %.3g", c.num, c.fam, peak)
		}
		switch {
		case c.dt > 0:
			pid := &PID{Kp: k.Kp, Ki: k.Ki, Kd: k.Kd, Tf: k.Tf, Dt: k.Dt, IFormula: k.IFormula, DFormula: k.DFormula}
			if !pidtuneExactSchurStableLag(t, pid, c.num, c.den, c.dt, 1, c.lag) {
				t.Errorf("%v %s: discrete closed loop unstable", c.num, c.fam)
			}
		case c.tau > 0:
			if z := pidtuneDelayedRHPRoots(c.num, c.den, k, c.tau); z != 0 {
				t.Errorf("%v %s: delayed closed loop has %d RHP roots", c.num, c.fam, z)
			}
		default:
			cn := Poly{k.Kd, k.Kp, k.Ki}
			cd := Poly{1, 0}
			if k.Ki == 0 {
				cn, cd = Poly{k.Kd, k.Kp}, Poly{1}
			}
			ch := polyMul(c.den, cd)
			nc := polyMul(c.num, cn)
			for i := range nc {
				ch[len(ch)-len(nc)+i] += nc[i]
			}
			if !routhStable(ch) {
				t.Errorf("%v %s: closed loop %v unstable", c.num, c.fam, ch)
			}
		}
	}
}

// pidtuneDelayedRHPRoots counts the open right-half-plane roots of
// s·D + N·(Kd·s² + Kp·s + Ki)·e^{−sτ}, deg N + 2 < deg D + 1, by the
// argument principle along the imaginary axis.
func pidtuneDelayedRHPRoots(num, den Poly, k *PID2, tau float64) int {
	cn := Poly{k.Kd, k.Kp, k.Ki}
	q := func(w float64) complex128 {
		s := complex(0, w)
		return s*den.Eval(s) + num.Eval(s)*cn.Eval(s)*cmplx.Exp(-s*complex(tau, 0))
	}
	deg := len(den)
	const top = 1e5
	arg, w, prev := 0., 0., q(0)
	for w < top {
		w += math.Min(.05/tau, math.Max(.005*w, 1e-6))
		v := q(w)
		arg += cmplx.Phase(v / prev)
		prev = v
	}
	lead := complex(den[0], 0) * cmplx.Pow(complex(0, w), complex(float64(deg), 0))
	arg -= cmplx.Phase(prev / lead)
	return int(math.Round((float64(deg) - 2*arg/math.Pi) / 2))
}

// The zero grid's top: π/Dt discrete; continuous, the first step where the
// loop tail bound drops under 1, or, without one, where |L| does; past
// 1e4·hi, while the tail bound is infinite, the first step it is finite.
func TestPIDTuningZeroGridTop(t *testing.T) {
	g := func(w float64) complex128 { return 1 / complex(1, w) }
	sample := func(w float64) (complex128, bool) { return g(w), true }
	step := math.Log(10) / 60
	c := PID2{Kp: .5, Kd: 2, B: 1, C: 1}
	tail := func(_ *PID2, w float64) float64 {
		if w < 100 {
			return 2
		}
		return .5
	}
	for _, x := range []struct {
		p        pidTuningPlant
		lo, high float64
	}{
		{pidTuningPlant{dt: .01}, pidTuningSensNyquist * math.Pi / .01, pidTuningSensNyquist * math.Pi / .01},
		{pidTuningPlant{loopTail: tail}, 100, 100 * math.Exp(step)},
		{pidTuningPlant{}, 1e4, 1e4},
		{pidTuningPlant{loopTail: func(*PID2, float64) float64 { return math.Inf(1) }}, 1e8, 1e8},
		{pidTuningPlant{loopTail: func(_ *PID2, w float64) float64 {
			if w < 1e6 {
				return math.Inf(1)
			}
			return 2
		}}, 1e6, 1e6 * math.Exp(step)},
	} {
		if top := pidTuningZeroGridTop(x.p, c, 1, step, sample); top < x.lo*(1-1e-12) || top > x.high*(1+1e-12) {
			t.Errorf("dt %g tail %v: top %g, want [%g, %g]", x.p.dt, x.p.loopTail != nil, top, x.lo, x.high)
		}
	}
	c.Kd = .2
	if top := pidTuningZeroGridTop(pidTuningPlant{}, c, 1, step, sample); top < 1 || top > 3 {
		t.Errorf("Kd .2 without tail: top %g, want first |L| < 1 above 1", top)
	}
}
