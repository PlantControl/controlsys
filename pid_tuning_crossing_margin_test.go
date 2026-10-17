package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"strings"
	"testing"
)

// The refined crossing scan finds a second 0 dB crossing where a lightly
// damped mode lifts |L| over 1 and ∠L lies in (0°, 60°). The wrapped margin
// 180° + ∠L − 360° ≈ −160° failed the floor, so designs base returned became
// unattainable, though the crossing is ≥ 120° from −1. Cases from a seeded
// base-vs-branch sweep of P/PD/PDF on lightly damped plants. Oracles: dense
// crossings of |L| = 1 over the evidence band, each at least 57° (P: 0°)
// from −1 measured either way round, |S| <= 4, and no net turn of
// arg(1+L(jω)), as the stable plant has no integrator.
func TestTunePIDKeepsCrossingFarFromMinusOne(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		tau      float64
		family   PidtuneType
	}{
		{"sweep 25 PD", []float64{0.8422342409714305}, []float64{1, 18.256125636185075, 81.17131833086688, 0.43675721515933075, 1.7478700537696203}, 0.0464207739550963, PidtunePD},
		{"sweep 92 PD", []float64{0.20537369176793527, 0.07644179822491082}, []float64{1, 1.7183399969090034, 0.37316542678414805, 0.6146203159254444}, 0.709792734590869, PidtunePD},
		{"sweep 280 PDF", []float64{-0.7181478491573415}, []float64{1, 15.36030373506239, 0.09459044794677574, 1.2021616305519631}, 1.0377200953917973, PidtunePDF},
		{"sweep 80 PD", []float64{-1.054599885306892}, []float64{1, 0.00299557580140128, 1.0733714787112956}, 0, PidtunePD},
		{"sweep 120 PDF", []float64{6.738053635107616}, []float64{1, 0.023101701101633985, 5.622166008185011}, 0, PidtunePDF},
		{"sweep 164 PDF", []float64{-0.37781717849161434, 1.4548937506677857}, []float64{1, 3.73663341903153, 52.126647355084614, 190.64242557388877, 9.501545598819801}, 0, PidtunePDF},
		{"sweep 70 P", []float64{-0.4839653090567671, 0.17578851538367593}, []float64{1, 0.5373380700910566, 29.303982053625848, 14.634713107039508, 0.7589946108096222}, 0.021518142814529132, PidtuneP},
	}
	for _, c := range cases {
		r, err := TunePID(context.Background(), pidPlantDelayed(t, c.num, c.den, c.tau), c.family, PIDTuningOptions{})
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := r.Controller
		g := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s) * cmplx.Exp(-s*complex(c.tau, 0))
		}
		L := func(w float64) complex128 {
			s := complex(0, w)
			return g(w) * (complex(k.Kp, 0) + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
		}
		floor := 57.
		if c.family == PidtuneP {
			floor = 0
		}
		band := r.Evidence.FrequencyBand
		margins := pidLoopMargins(L, band[0], band[1])
		far := false
		for _, m := range margins {
			if d := 180 - math.Abs(m-180); d < floor-1e-6 || d <= 0 {
				t.Errorf("%s: %+v crossing %g° from −1 (dense %v)", c.name, *k, d, margins)
			}
			far = far || m > 180 && m-360 <= -floor
		}
		if !far {
			t.Errorf("%s: dense margins %v lack a crossing with wrapped margin <= −%g°", c.name, margins, floor)
		}
		if len(margins) != len(r.Evidence.GainCrossovers) {
			t.Errorf("%s: evidence crossovers %v, dense margins %v", c.name, r.Evidence.GainCrossovers, margins)
		}
		if c.family != PidtuneP && r.Evidence.AchievedPhaseMargin < 57-1e-6 {
			t.Errorf("%s: achieved margin %g", c.name, r.Evidence.AchievedPhaseMargin)
		}
		wc := r.Evidence.AchievedCrossover
		if peak, at := pidDenseSensitivity(*k, g, wc/1e3, wc*1e2); peak > 4 {
			t.Errorf("%s: %+v |S| %g at ω=%g", c.name, *k, peak, at)
		}
		if wind := pidWindingDeg(L, 1e-6, wc*1e4); math.Abs(wind) > 1 {
			t.Errorf("%s: %+v arg(1+L) turns %g°", c.name, *k, wind)
		}
	}
}

// Without a stability certificate the sign of a wrapped margin is kept
// unless the loop phase, unwrapped from wc, shows the crossing one turn up.
// Oracle: the phase of the rational transfer function unwrapped on a dense
// grid from wc, less τω.
func TestPIDTuningCrossingMarginUnwrapsFromWc(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		tau      float64
		c        PID2
	}{
		{"sweep 25 PD", []float64{0.8422342409714305}, []float64{1, 18.256125636185075, 81.17131833086688, 0.43675721515933075, 1.7478700537696203}, 0.0464207739550963, PID2{Kp: 0.025106867451115664, Kd: 0.24450446435911616, B: 1, C: 1}},
		{"sweep 8 P", []float64{-0.7598912272633042}, []float64{1, 0.9911134835150913, 10.725723972363745, 10.38488812408643, 0.45166140242452574}, 0, PID2{Kp: -1, B: 1, C: 1}},
		{"sweep 70 P", []float64{-0.4839653090567671, 0.17578851538367593}, []float64{1, 0.5373380700910566, 29.303982053625848, 14.634713107039508, 0.7589946108096222}, 0.021518142814529132, PID2{Kp: 6.098853395484589, B: 1, C: 1}},
	}
	reflected, kept := 0, 0
	for _, c := range cases {
		plant := pidPlantDelayed(t, c.num, c.den, c.tau)
		eval, err := newSISOEval(plant)
		if err != nil {
			t.Fatal(err)
		}
		rational := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s) * pidTuningFeedback(c.c, w)
		}
		L := func(w float64) complex128 { return rational(w) * cmplx.Exp(complex(0, -c.tau*w)) }
		var crossings []float64
		lo, hi := 1e-4, 1e3
		prev := cmplx.Abs(L(lo)) - 1
		for w := lo * 1.0001; w < hi; w *= 1.0001 {
			next := cmplx.Abs(L(w)) - 1
			if prev*next < 0 {
				a, b := w/1.0001, w
				for range 60 {
					m := math.Sqrt(a * b)
					if (cmplx.Abs(L(m))-1)*prev > 0 {
						a = m
					} else {
						b = m
					}
				}
				crossings = append(crossings, math.Sqrt(a*b))
			}
			prev = next
		}
		if len(crossings) < 2 {
			t.Fatalf("%s: crossings %v", c.name, crossings)
		}
		wc := crossings[0]
		for _, x := range crossings {
			if m := 180 + cmplx.Phase(L(x))*180/math.Pi; m > 0 && m < 180 {
				wc = x
				break
			}
		}
		p := pidTuningPlant{at: eval.at, tau: c.tau, modes: pidTuningModes(plant)}
		checked := 0
		for _, w := range crossings {
			m := phaseMarginDeg(L(w))
			if m > -57 || w == wc {
				continue
			}
			phase, prev := cmplx.Phase(L(wc))+c.tau*wc, rational(wc)
			ratio := 1.0001
			if w < wc {
				ratio = 1 / ratio
			}
			for x := wc; x != w; {
				if x *= ratio; (x-w)*(x/ratio-w) <= 0 {
					x = w
				}
				next := rational(x)
				phase += cmplx.Phase(next / prev)
				prev = next
			}
			unwrapped := 180 + (phase-c.tau*w)*180/math.Pi
			want := m
			if math.Abs(unwrapped-(m+360)) < 1e-6 {
				want = -m
				reflected++
			} else {
				kept++
			}
			if got := pidTuningCrossingMargin(p, c.c, wc, w, m, 57, false); math.Abs(got-want) > 1e-9 {
				t.Errorf("%s: crossing %g wrapped %g° unwrapped %g°: got %g°, want %g°", c.name, w, m, unwrapped, got, want)
			}
			if got := pidTuningCrossingMargin(p, c.c, wc, w, m, 57, true); got != -m {
				t.Errorf("%s: certified crossing %g wrapped %g°: got %g°, want %g°", c.name, w, m, got, -m)
			}
			checked++
		}
		if checked == 0 {
			t.Errorf("%s: no crossing with wrapped margin <= −57° among %v", c.name, crossings)
		}
	}
	if reflected == 0 || kept == 0 {
		t.Errorf("%d crossings one turn up, %d not: want both", reflected, kept)
	}
}

// A design rejected for a crossing below the floor says where and by how
// much. Sweep 2's PD at the moved 2.714 rad/s crossover crosses 0 dB again
// near its 2.75 rad/s mode at −41°.
func TestTunePIDCrossingRejectionWarns(t *testing.T) {
	plant := pidPlantDelayed(t, []float64{-4.701604109784369, 7.875616525831797}, []float64{1, 0.5014895891419527, 7.5751678031609675, 3.376552705528892}, 0)
	_, err := TunePID(context.Background(), plant, PidtunePD, PIDTuningOptions{})
	var te *PIDTuningError
	if !errors.As(err, &te) {
		t.Fatalf("err %v, want unattainable", err)
	}
	for _, w := range te.Evidence.Warnings {
		if strings.Contains(w, "crossed 0 dB again") && strings.Contains(w, "57° floor") {
			return
		}
	}
	t.Errorf("warnings %q lack the crossing rejection", te.Evidence.Warnings)
}
