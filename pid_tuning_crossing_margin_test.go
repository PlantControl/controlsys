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

// A wrapped margin at or below −floor counts the far way round from −1
// only for a loop the caller certifies stable; the phase at one crossing,
// unwrapped or not, says nothing about encirclements.
func TestPIDTuningCrossingMarginNeedsCertificate(t *testing.T) {
	cases := []struct {
		wc, w, m, floor float64
		certified       bool
		want            float64
	}{
		{1, 2, -160, 57, true, 160},
		{1, 2, -57, 57, true, 57},
		{1, 2, -56, 57, true, -56},
		{1, 1, -160, 57, true, -160},
		{1, 2, 30, 57, true, 30},
		{1, 2, -160, 57, false, -160},
		{1, .5, -120, 57, false, -120},
		{1, 2, 30, 57, false, 30},
	}
	for _, c := range cases {
		if got := pidTuningCrossingMargin(c.wc, c.w, c.m, c.floor, c.certified); got != c.want {
			t.Errorf("%+v: got %g", c, got)
		}
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
