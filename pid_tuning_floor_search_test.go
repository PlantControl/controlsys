package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"testing"
	"time"
)

// pidWindingDeg returns the unwrapped change of arg(1+L(jω)) in degrees over
// a dense log grid on [lo, hi].
func pidWindingDeg(L func(float64) complex128, lo, hi float64) float64 {
	const n = 400000
	step := math.Log(hi/lo) / n
	prev := 1 + L(lo)
	total := 0.
	for k := 1; k <= n; k++ {
		next := 1 + L(lo*math.Exp(float64(k)*step))
		total += cmplx.Phase(next / prev)
		prev = next
	}
	return total * 180 / math.Pi
}

func pidPoly(k float64, factors ...[]float64) []float64 {
	p := []float64{k}
	for _, f := range factors {
		p = polyMulTest(p, f)
	}
	return p
}

func pidPolyAt(p []float64, s complex128) complex128 {
	var v complex128
	for _, c := range p {
		v = v*s + complex(c, 0)
	}
	return v
}

func pidPlantDelayed(t *testing.T, num, den []float64, tau float64) *System {
	t.Helper()
	p := makePlant(t, num, den)
	if tau > 0 {
		if err := p.SetInputDelay([]float64{tau}); err != nil {
			t.Fatal(err)
		}
	}
	return p
}

// The floor returns designs for plants whose default target fails. An
// independent dense sweep checks the margins, and the change of arg(1+L)
// over (0+, ∞) checks stability: with a stable plant and one integrator,
// 1+L turns from −90° to 0° when it does not encircle the origin.
func TestTunePIDFloorDesignsSound(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		tau      float64
		family   PidtuneType
		wc       float64
	}{
		{"-0.48e^-0.215s/((s+10.6)(s²+0.0338s+0.716))", pidPoly(-.48), pidPoly(1, []float64{1, 10.6}, []float64{1, .0338, .716}), .215, PidtunePID, 0},
		{"0.106/(s²+0.27s+45.6)", pidPoly(.106), pidPoly(1, []float64{1, .27, 45.6}), 0, PidtunePIDF, 3.378},
	}
	for _, c := range cases {
		plant := pidPlantDelayed(t, c.num, c.den, c.tau)
		r, err := TunePID(context.Background(), plant, c.family, PIDTuningOptions{CrossoverFrequency: c.wc})
		if err != nil {
			t.Fatalf("%s %s: %v", c.name, c.family, err)
		}
		k := r.Controller
		L := func(w float64) complex128 {
			s := complex(0, w)
			g := pidPolyAt(c.num, s) / pidPolyAt(c.den, s) * cmplx.Exp(-s*complex(c.tau, 0))
			return g * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
		}
		margins := pidLoopMargins(L, 1e-4, 1e4)
		if len(margins) == 0 {
			t.Fatalf("%s: no crossover", c.name)
		}
		for _, pm := range margins {
			if pm < 57 {
				t.Errorf("%s: %+v independent margin %g below the 60° floor", c.name, k, pm)
			}
		}
		if wind := pidWindingDeg(L, 1e-6, 1e4); math.Abs(wind-90) > 1 {
			t.Errorf("%s: %+v arg(1+L) changes by %g°, want 90° for a stable loop", c.name, k, wind)
		}
	}
}

// −1.09(1−s/1.81)e^{−0.437s}/((s+1.77)(s+4.47)(s+1.08)(s²+0.281s+0.0404))
// at wc = 7.927 has unwrapped phase near −675°: no same-sign PID, whose
// phase lies in [−90°, 90°], brings the loop above −180° there, so every
// floor band is skipped without a search instead of costing ~1 ms per
// Nyquist check across 1,400 evaluations.
func TestTunePIDFloorSkipsBandsBeyondUnwrappedPhase(t *testing.T) {
	num := pidPoly(-1.09, []float64{-1 / 1.81, 1})
	den := pidPoly(1, []float64{1, 1.77}, []float64{1, 4.47}, []float64{1, 1.08}, []float64{1, .281, .0404})
	plant := pidPlantDelayed(t, num, den, .437)
	const wc = 7.927
	_, ev, searches, err := pidTuneCounted(t, plant, PidtunePID, PIDTuningOptions{CrossoverFrequency: wc})
	if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Fatalf("err %v", err)
	}
	if searches != 1 || ev.Evaluations > 200 {
		t.Fatalf("%d searches, %d evaluations; want only the ±3° search", searches, ev.Evaluations)
	}
	eval, err := newSISOEval(plant)
	if err != nil {
		t.Fatal(err)
	}
	want := -(math.Atan(wc/1.77) + math.Atan(wc/4.47) + math.Atan(wc/1.08) + math.Atan2(.281*wc, .0404-wc*wc) + math.Atan(wc/1.81) + .437*wc) * 180 / math.Pi
	if got := pidTuningUnwrappedPhase(eval.at, -1, wc, .437); math.Abs(got-want) > 1e-6 {
		t.Fatalf("unwrapped phase %g, want %g", got, want)
	}
}

// With MaxEvaluations = 5 the ±3° PDF search fails, and each moved
// crossover gets its own five evaluations, as a cap per search: the first
// moved crossover meets PM 60°±3°.
func TestTunePIDMovedCrossoverGetsOwnBudget(t *testing.T) {
	num := pidPoly(-.52, []float64{1 / 1.24, 1})
	den := pidPoly(1, []float64{1, 10.9, 60.8}, []float64{1, .0371}, []float64{1, 1.62, 1.35})
	const tau = .108
	r, err := TunePID(context.Background(), pidPlantDelayed(t, num, den, tau), PidtunePDF, PIDTuningOptions{MaxEvaluations: 5})
	if err != nil {
		t.Fatal(err)
	}
	k := r.Controller
	if k.Kp >= 0 || k.Kd >= 0 || k.Ki != 0 || k.Tf <= 0 || r.Evidence.Evaluations <= 5 || r.Evidence.Evaluations > 25 {
		t.Fatalf("%+v, %d evaluations", k, r.Evidence.Evaluations)
	}
	L := func(w float64) complex128 {
		s := complex(0, w)
		g := pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*tau)
		return g * (complex(k.Kp, 0) + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
	}
	margins := pidLoopMargins(L, 1e-4, 1e4)
	if len(margins) == 0 {
		t.Fatal("no crossover")
	}
	for _, pm := range margins {
		if pm < 57 {
			t.Errorf("%+v independent margin %g below 57°", k, pm)
		}
	}
	if wind := pidWindingDeg(L, 1e-6, 1e4); math.Abs(wind) > 1 {
		t.Errorf("%+v arg(1+L) changes by %g°, want 0 for a stable loop", k, wind)
	}
}

// A floor band whose candidates meet the margins but all fail the stability
// check ends the floor: on delayed plants each check is a Nyquist sweep, and
// searching every band took 50 s on the PD plant (gain sign unknown, so no
// band is skipped) where the ±3° search fails in milliseconds.
func TestTunePIDFloorStopsAtUnstableBandDelayed(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		tau      float64
		family   PidtuneType
		o        PIDTuningOptions
	}{
		{"-31.29(s+0.09685)/(s(s²+14.84s+367.9)(s²+7.963s+36.18))e^-0.5042s", pidPoly(-31.29, []float64{1, .09685}), pidPoly(1, []float64{1, 0}, []float64{1, 14.84, 367.9}, []float64{1, 7.963, 36.18}), .5042, PidtunePD, PIDTuningOptions{PhaseMargin: 36.78}},
		{"-7.184/((s²+2.717s+114.2)s(s-0.1382))e^-0.07239s", pidPoly(-7.184), pidPoly(1, []float64{1, 2.717, 114.2}, []float64{1, 0}, []float64{1, -.1382}), .07239, PidtunePI, PIDTuningOptions{CrossoverFrequency: 15.15}},
	}
	for _, c := range cases {
		ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
		cc := &pidCountingContext{Context: ctx}
		_, err := TunePID(cc, pidPlantDelayed(t, c.num, c.den, c.tau), c.family, c.o)
		cancel()
		var te *PIDTuningError
		if !errors.As(err, &te) || !errors.Is(err, ErrPIDTuningTargetUnattainable) {
			t.Fatalf("%s: err %v", c.name, err)
		}
		if searches := int(cc.errs.Load()) - 1 - te.Evidence.Evaluations; searches != 2 {
			t.Errorf("%s: %d searches; want the ±3° search and one floor band", c.name, searches)
		}
	}
}

// pidDenseLoop returns the least margin over every 0 dB crossing of L on
// [lo, hi] and the peak of |1/(1+L)|, from a 4M-point log sweep.
func pidDenseLoop(L func(float64) complex128, lo, hi float64) (minMargin, peakS float64) {
	minMargin = math.Inf(1)
	for _, m := range pidLoopMargins(L, lo, hi) {
		minMargin = math.Min(minMargin, m)
	}
	const n = 4000000
	for k := 0; k <= n; k++ {
		peakS = math.Max(peakS, 1/cmplx.Abs(1+L(lo*math.Exp(float64(k)*math.Log(hi/lo)/n))))
	}
	return minMargin, peakS
}

// Ranking the incumbent by |S| excess first took the first seed within
// |S| ≤ 4 and stopped at a worse optimum, J +35% and integral action all but
// gone on the first plant, though the score-ranked path reaches a feasible
// design. The score-ranked pass runs first and keeps it.
func TestTunePIDSensitivitySearchKeepsScoreRankedOptimum(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		tau, pm  float64
		maxJ     float64
	}{
		{"89.42/(s+0.6103)e^-0.01376s", pidPoly(89.42), pidPoly(1, []float64{1, .6103}), .01376, 0, .30},
		{"3.452/((s+0.0549)(s+12.98))e^-0.4866s", pidPoly(3.452), pidPoly(1, []float64{1, .0549}, []float64{1, 12.98}), .4866, 31.17, .75},
	}
	for _, c := range cases {
		r, err := TunePID(context.Background(), pidPlantDelayed(t, c.num, c.den, c.tau), PidtunePIDF, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		k := r.Controller
		if r.Evidence.Objective > c.maxJ || k.Kp <= 0 || k.Ki <= 0 || k.Kd <= 0 || k.Tf <= 0 {
			t.Errorf("%s: %+v J=%g, want J ≤ %g with all gains positive", c.name, k, r.Evidence.Objective, c.maxJ)
		}
		pm := c.pm
		if pm == 0 {
			pm = 60
		}
		L := func(w float64) complex128 {
			s := complex(0, w)
			g := pidPolyAt(c.num, s) / pidPolyAt(c.den, s) * cmplx.Exp(-s*complex(c.tau, 0))
			return g * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s/(1+complex(k.Tf, 0)*s))
		}
		wc := r.Evidence.AchievedCrossover
		margin, peak := pidDenseLoop(L, wc/100, wc*100)
		if margin < pm-3-1e-6 || peak > 4*(1+1e-4) {
			t.Errorf("%s: %+v dense margin %g, max|S| %g", c.name, k, margin, peak)
		}
		if wind := pidWindingDeg(L, 1e-6, 1e6); math.Abs(wind-90) > 1 {
			t.Errorf("%s: %+v arg(1+L) changes by %g°, want 90° for a stable loop", c.name, k, wind)
		}
	}
}

// A floor design's high gain can cross 0 dB again near a lightly damped
// mode outside [wc/100, 100wc]: here the 9.2 rad/s resonance, where the
// unwidened floor returned a design with a 13.8° margin at 10.6 rad/s.
// The floor finds crossovers over the plant's dynamics and between grid
// samples, so every crossover in a dense sweep meets PM − 3°.
func TestTunePIDFloorMarginsCoverPlantModes(t *testing.T) {
	num := pidPoly(-6.691, []float64{1, .06333})
	den := pidPoly(1, []float64{1, .7119, 84.56}, []float64{1, 1.597}, []float64{1, .5175})
	r, err := TunePID(context.Background(), makePlant(t, num, den), PidtunePID, PIDTuningOptions{CrossoverFrequency: .03374})
	if err != nil {
		t.Fatal(err)
	}
	k := r.Controller
	L := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s) * (complex(k.Kp, 0) + complex(k.Ki, 0)/s + complex(k.Kd, 0)*s)
	}
	margins := pidLoopMargins(L, 1e-6, 1e4)
	if len(margins) == 0 {
		t.Fatalf("%+v: no crossover", k)
	}
	for _, pm := range margins {
		if pm < 57 {
			t.Errorf("%+v: independent margin %g below the 60° floor", k, pm)
		}
	}
	if wind := pidWindingDeg(L, 1e-6, 1e4); math.Abs(wind-90) > 1 {
		t.Errorf("%+v arg(1+L) changes by %g°, want 90° for a stable loop", k, wind)
	}
	if hi := r.Evidence.FrequencyBand[1]; hi < 10*9.19 {
		t.Errorf("evidence band %v does not cover the 9.2 rad/s mode", r.Evidence.FrequencyBand)
	}
}
