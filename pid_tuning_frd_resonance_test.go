package controlsys

import (
	"context"
	"math"
	"testing"
)

type pidFRDResonanceCase struct {
	name     string
	num, den []float64
	lo, hi   float64
	n        int
	o        PIDTuningOptions
	k        PID2
}

// Lightly damped plants whose resonance falls inside one step of their
// samples, 10-15% apart, so 1 + L loops round 0 between two samples while
// |1 + L| stays 0.6-0.8 at both; the chord the count followed missed it
// (N4MOKA residuals A, B, the 53.08 plant and the -6.007/-9.631 plant).
var pidFRDResonanceCases = []pidFRDResonanceCase{
	{"A", []float64{0.07240994924086397, 0.059495186577374334}, []float64{1, 0.06618095732149026, 1.1605932139252835, 0.06953950210523521}, 0.0013095419860959085, 9.526043738954273, 96, PIDTuningOptions{CrossoverFrequency: 0.9101415155439248}, PID2{Kp: 0.04993265431173388, Ki: 8.824601906317316, Kd: 6.996393449841795, Tf: 0.05481601240176187}},
	{"B", []float64{-0.9289845985534367, -0.39025102037202714}, []float64{1, 1.489420303960932, 1.6824967617504507, 1.7238582157494047, 0.596673736836853}, 0.0014246474627752684, 15.068890412978842, 70, PIDTuningOptions{}, PID2{Kp: 0.019400492042060763, Ki: -1.0443949524710314, Kd: -0.7664723286229123, Tf: 0.19891975242777188}},
	{"53.08 resonance", []float64{0.4306132124087867}, []float64{1, 0.13918092085589634, 53.07830316702831, 0.30219933603212895, 0.08522920474437591}, 1e-3, 1e3, 100, PIDTuningOptions{}, PID2{Kp: 13.327, Ki: 2.1872, Kd: 59.808, Tf: 0.22663}},
	{"-6.007/-9.631", []float64{-6.006874817954687, -9.631465973111077}, []float64{1, 0.6088284763961694, 5.978223314083145, 3.2080336945155454, 1.8717858124577227}, 1e-3, 1e3, 100, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: -0.04015450361054589, Ki: -0.07844207386734459, Kd: -0.005065080923915894, Tf: 0.3695158773619382}},
}

// N4MOKA: the strict count certified A's and B's unstable PIDFs (the
// other two cases' are guards). The samples about the resonance fix its
// pole by a Möbius fit, and the count now follows 1 + L along the fitted
// resonance across that step, which winds round 0, so none is certified.
// Oracle: companion roots.
func TestPIDTuningFRDCountResonanceBetweenSamples(t *testing.T) {
	for _, c := range pidFRDResonanceCases {
		k := c.k
		k.B, k.C = 1, 1
		if re := pidClosedLoopMaxRe(c.num, c.den, k); !(re > 1e-3) {
			t.Fatalf("%s: oracle max Re %g, want unstable", c.name, re)
		}
		p, _ := pidTuningFRDStrictPlant(pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, 0, c.lo, c.hi, c.n), c.num, c.den), nil)
		if stable, known, _ := pidTuningFRDNyquistVerdict(k, p); stable && known {
			t.Errorf("%s: strict count finds unstable %+v stable", c.name, k)
		}
	}
}

// N4MOKA: TunePIDFRD returned A's and B's unstable PIDFs as certified. Each
// PIDF it returns now is stable with dense |S| <= 4, or it fails. Oracles:
// companion roots and a dense |S| sweep of the exact plant.
func TestTunePIDFRDResonanceBetweenSamples(t *testing.T) {
	for _, c := range pidFRDResonanceCases {
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, 0, c.lo, c.hi, c.n), c.num, c.den)
		r, err := TunePIDFRD(context.Background(), frd, PidtunePIDF, c.o)
		if err != nil {
			continue
		}
		k := *r.Controller
		if re := pidClosedLoopMaxRe(c.num, c.den, k); !(re < 0) {
			t.Errorf("%s: %+v closed-loop root at Re %g (certified %v)", c.name, k, re, !pidHasWarning(r, "not certified"))
			continue
		}
		exact := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s)
		}
		if peak, at := pidDenseSensitivity(k, exact, 1e-3, 1e2); peak > 4 {
			t.Errorf("%s: %+v dense |S| %g at %g", c.name, k, peak, at)
		}
	}
}

// N4MOKA: a resonance at 3.344 rad/s (σ = 0.18) on samples 9% apart turns
// G by only 74° across its step, but its circle by 147°, and 1 + L of a
// PID at PM 45 looped round 0 there: TunePIDFRD certified it with a
// closed-loop root at Re s = +1.3e-4 and dense |S| 1444. A step whose
// fitted resonance circle turns past a quarter turn is now followed along
// it, and the PID returned is stable with dense |S| <= 4 and PM within 3°
// of 45. Oracles: companion roots and a dense |S| sweep of the exact plant.
func TestTunePIDFRDResonanceArcAcrossStep(t *testing.T) {
	num, den := []float64{224.784158985432, 4815.190127335998}, []float64{1, 64.52833861020099, 1749.1951108288692, 1746.042505195741, 19394.203200499687, 4506.091955648133, 172.42544507806994}
	frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, num, den, 0, 0.0023470463672586555, 946.2987972842172, 148), num, den)
	r, err := TunePIDFRD(context.Background(), frd, PidtunePID, PIDTuningOptions{PhaseMargin: 45})
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	if re := pidClosedLoopMaxRe(num, den, k); !(re < 0) {
		t.Fatalf("%+v closed-loop root at Re %g (certified %v)", k, re, !pidHasWarning(r, "not certified"))
	}
	exact := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s)
	}
	if peak, at := pidDenseSensitivity(k, exact, 1e-3, 1e2); peak > 4 {
		t.Errorf("%+v dense |S| %g at %g", k, peak, at)
	}
	if pm := r.Evidence.AchievedPhaseMargin; pm < 42 || pm > 48 {
		t.Errorf("%+v achieved PM %g", k, pm)
	}
}

// A resonance at 3.29 rad/s (σ = 0.073) on samples 5% apart turns G by 97°
// across one step, but a P controller's loop there is small and far from
// 0: TunePIDFRD still returns it certified, stable with dense |S| <= 4
// (a test of the arg G jump alone, or of 0 left of the chord, failed it).
// Oracles: companion roots and a dense |S| sweep of the exact plant.
func TestTunePIDFRDResolvedResonanceKeepsDesign(t *testing.T) {
	num, den := []float64{-0.027719450053264573}, []float64{1, 0.18043641442598132, 10.849680881786757, 0.37553236984532157, 0.00903269920139743}
	frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, num, den, 0, 0.0026341590584166813, 32.92992521922734, 187), num, den)
	r, err := TunePIDFRD(context.Background(), frd, PidtuneP, PIDTuningOptions{})
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	if pidHasWarning(r, "not certified") {
		t.Errorf("%+v not certified", k)
	}
	if re := pidClosedLoopMaxRe(num, den, k); !(re < 0) {
		t.Fatalf("%+v closed-loop root at Re %g", k, re)
	}
	exact := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s)
	}
	if peak, at := pidDenseSensitivity(k, exact, 1e-3, 1e2); peak > 4 {
		t.Errorf("%+v dense |S| %g at %g", k, peak, at)
	}
}

// 5G6YXU: ζ = 0.046 at 6.44 rad/s on samples 5% apart (σ = 0.30): the
// step above the resonance turns its circle by 78°, short of the count's
// quarter turn, and the chord between its samples cut inside the circle,
// so the sampled |S| of this PI read 3.57 against a dense 6.02 at 6.62
// rad/s. Along the fitted resonance it reads the dense peak. Oracle: a
// dense |S| sweep of the exact plant.
func TestPIDTuningFRDSensitivityPeakFollowsArc(t *testing.T) {
	num, den := []float64{87.79273299108002}, []float64{1, 0.5970851544204655, 41.47651141229077}
	frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, num, den, 0, 0.0011661855703621533, 452.1165827954031, 262), num, den)
	k := PID2{Kp: 0.02675078176886847, Ki: 0.24064168758398488, B: 1, C: 1}
	exact := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s)
	}
	dense, _ := pidDenseSensitivity(k, exact, 1, 20)
	for _, strict := range []bool{false, true} {
		p := pidTuningFRDPlant(frd, nil)
		if strict {
			p, _ = pidTuningFRDStrictPlant(frd, nil)
		}
		p.maxSensitivity, p.crossLow, p.crossHigh = pidTuningMaxSensitivity, p.low, p.high
		if peak := pidTuningSensitivityPeak(p, k); math.Abs(peak-dense) > .05*dense {
			t.Errorf("strict %v: sampled |S| %g, dense %g", strict, peak, dense)
		}
	}
}

// 5G6YXU: on these lightly damped plants (ζ 0.01-0.07) TunePIDFRD
// certified stable designs with sampled |S| <= 4 and dense |S| 4.4-6.0
// between samples 2.5-5% apart about the resonance. Each now returns a
// certified stable design with dense |S| <= 4 and the margin asked within
// 3°, or, for the PI, fails; 863's least margin is 59.3°, at a crossing
// pair its resonance lifts, which its achieved margin reports (CTA5F5).
// Oracles: companion roots and dense |S| and margin sweeps of the exact
// plant.
func TestTunePIDFRDArcSensitivity(t *testing.T) {
	cases := []struct {
		name     string
		num, den []float64
		lo, hi   float64
		n        int
		family   PidtuneType
		pm       float64
	}{
		{"82", []float64{490.901408513619}, []float64{1, 4.032335082820667, 535.4806740835618, 199.3019023956184, 4095.1958006460177, 31.245174001970312, 121.48384152077081}, 0.001463989011206276, 760.8521933269211, 326, PidtunePID, 0},
		{"446", []float64{0.03427386159776838, 0.014532818541378367}, []float64{1, 0.2636013737368447, 5.091192189068356, 0.038108756380640485, 0.024300556782702296}, 0.003781847655459871, 455.973506587706, 283, PidtunePDF, 45},
		{"863", []float64{0.00021970886469083835}, []float64{1, 0.9776769452222025, 0.1027180463208214, 0.018029569371016227, 0.0012495681804130616}, 0.00437729377687368, 257.2570683909091, 215, PidtunePID, 70},
		{"1139", []float64{87.79273299108002}, []float64{1, 0.5970851544204655, 41.47651141229077}, 0.0011661855703621533, 452.1165827954031, 262, PidtunePI, 70},
		{"1662", []float64{-0.0009820178513339885}, []float64{1, 0.08206323645579383, 0.641603795034597, 0.004614502651611619, 0.0014155991549359775}, 0.00857853825824086, 205.38058135561954, 252, PidtunePD, 45},
		{"1851", []float64{6.464532750828114}, []float64{1, 1.021751437068309, 52.08341188662163, 2.923264202150054}, 0.007945063099872534, 462.51586702986936, 338, PidtunePID, 0},
	}
	for _, c := range cases {
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, 0, c.lo, c.hi, c.n), c.num, c.den)
		r, err := TunePIDFRD(context.Background(), frd, c.family, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			if c.family != PidtunePI {
				t.Errorf("%s: %v", c.name, err)
			}
			continue
		}
		k := *r.Controller
		if pidHasWarning(r, "not certified") {
			t.Errorf("%s: %+v not certified", c.name, k)
		}
		if re := pidClosedLoopMaxRe(c.num, c.den, k); !(re < 0) {
			t.Errorf("%s: %+v closed-loop root at Re %g", c.name, k, re)
			continue
		}
		exact := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s)
		}
		if peak, at := pidDenseSensitivity(k, exact, c.lo, c.hi); peak > 4 {
			t.Errorf("%s: %+v dense |S| %g at %g", c.name, k, peak, at)
		}
		_, _, least := pidFRDDenseAngles(c.num, c.den, 0, k, c.lo, c.hi)
		if pm := r.Evidence.AchievedPhaseMargin; math.Abs(pm-least) > 1 {
			t.Errorf("%s: %+v achieved PM %g, dense %g", c.name, k, pm, least)
		}
		if want := c.pm; want > 0 && math.Abs(least-want) > 3 && !(c.name == "863" && least > 58.5) {
			t.Errorf("%s: %+v dense PM %g, want %g", c.name, k, least, want)
		}
	}
}
