package controlsys

import (
	"context"
	"math"
	"testing"
)

// N4MOKA: a PIDF on a lightly damped plant crosses 0 dB at 0.2387 and
// 0.2455 rad/s, 2.8% apart on data sampled 2.3% apart, so between the
// samples at 0.2421 and 0.2477 1 + L winds round 0 the other way from
// their chord, which passes 0.39 from it while the samples' deviation from
// their neighbours' chords is about 1. The strict count, with the DC
// sample anchoring its low end, certified the loop; it has a closed-loop
// root at Re s = +0.0024. The count is now unknown there, with or without
// the DC sample, and TunePIDFRD returns no unstable design as certified.
// Oracle: companion roots.
func TestPIDTuningFRDCountUnresolvedCrossings(t *testing.T) {
	num, den := []float64{0.0029356119296049623, 0.008229341811485028}, []float64{1, 0.16531968867170563, 0.059760631262763785, 0.00972338931651889}
	c := PID2{Kp: -0.021754, Ki: 0.298056, Kd: 4.884193, Tf: 0.171156, B: 1, C: 1}
	if re := pidClosedLoopMaxRe(num, den, c); !(re > 1e-3) {
		t.Fatalf("oracle max Re %g, want unstable", re)
	}
	frd := pidExactFRD(t, num, den, 1e-3, 1e3, 600)
	for _, d := range []*FRD{frd, pidFRDStrictWithDC(t, frd, num, den)} {
		p, _ := pidTuningFRDStrictPlant(d, nil)
		if stable, known, _ := pidTuningFRDNyquistVerdict(c, p); stable && known {
			t.Errorf("DC sample %v: strict count finds unstable %+v stable", d.Omega[0] == 0, c)
		}
		if pidTuningFRDNyquist(c, pidTuningFRDPlant(d, nil)) {
			t.Errorf("DC sample %v: looser count accepts unstable %+v", d.Omega[0] == 0, c)
		}
		for _, family := range []PidtuneType{PidtunePIDF, PidtunePID} {
			for _, o := range []PIDTuningOptions{{PhaseMargin: 45}, {PhaseMargin: 45, CrossoverFrequency: 0.1768}} {
				r, err := TunePIDFRD(context.Background(), d, family, o)
				if err != nil {
					continue
				}
				if !pidFRDStrictStable(num, den, *r.Controller) && !pidHasWarning(r, "not certified") {
					t.Errorf("DC sample %v %v %+v: unstable %+v returned as certified", d.Omega[0] == 0, family, o, *r.Controller)
				}
			}
		}
	}
}

// N4MOKA: on lightly damped plants sampled too sparsely to resolve their
// resonance, TunePIDFRD returned unstable loops, the first certified on
// data with a DC sample, as between two samples near the resonance 1 + L
// wound round 0 the other way from the count's chord. Each now comes back
// stable or fails. Oracles: companion roots and exact Schur–Cohn.
func TestTunePIDFRDUnresolvedResonanceUnstable(t *testing.T) {
	t.Parallel()
	for _, c := range []struct {
		name     string
		num, den []float64
		dt       float64
		n        int
		dc       bool
		family   PidtuneType
		pm       float64
	}{
		{"certified PIDF", []float64{-6.006874817954687, -9.631465973111077}, []float64{1, 0.6088284763961694, 5.978223314083145, 3.2080336945155454, 1.8717858124577227}, 0, 100, true, PidtunePIDF, 45},
		{"discrete PI", []float64{0.4891912612244529}, []float64{1, -1.7586444654047826, 2.041729217124253, -1.630513640922947, 0.6034105725149826}, 0.1, 300, false, PidtunePI, 60},
		{"PIDF PM60", []float64{0.001348299300117877, 0.0007419624806788853}, []float64{1, 0.08362777027875495, 0.013745941805154576, 0.00044306410898299256, 4.070900776672517e-05}, 0, 100, false, PidtunePIDF, 60},
		{"PIDF PM45", []float64{-0.004282306877400522}, []float64{1, 0.012825341410483975, 0.008421686701150677, 9.50863435000853e-05}, 0, 100, false, PidtunePIDF, 45},
	} {
		hi := 1e3
		if c.dt > 0 {
			hi = .99 * math.Pi / c.dt
		}
		frd := pidFRDStrictData(t, c.num, c.den, c.dt, 1e-3, hi, c.n)
		if c.dc {
			frd = pidFRDStrictWithDC(t, frd, c.num, c.den)
		}
		r, err := TunePIDFRD(context.Background(), frd, c.family, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			continue
		}
		if !pidFRDStrictStable(c.num, c.den, *r.Controller) {
			t.Errorf("%s: unstable %+v (certified %v)", c.name, *r.Controller, !pidHasWarning(r, "not certified"))
		}
	}
}

// N4MOKA #11: discrete PIDFs a Kd/Tf re-solving search found on the
// "derivative pole near −1" data, both unstable. The strict count rejects
// them (they could come back only from the looser count, which accepts
// every loop on data without a DC gain, uncertified and warned). Oracle:
// exact Schur–Cohn.
func TestPIDTuningFRDStrictCountRejectsResolvedPIDF(t *testing.T) {
	num, den := []float64{-1.3448981227532175}, []float64{1, -1.5811862552962634, 0.8405266355392914, -0.2593403802430281}
	p, _ := pidTuningFRDStrictPlant(pidFRDStrictData(t, num, den, .1, 0.5184154210153508, 17.34960121464623, 422), nil)
	for _, c := range []PID2{
		{IFormula: BackwardEuler, Kp: -1.257051414453467, Ki: 2.015450480743815, Kd: 0.7560127242993466, Tf: 0.5910346344538493, Dt: 0.1, B: 1, C: 1},
		{IFormula: BackwardEuler, Kp: -0.8903266789306609, Ki: 1.6807747216159954, Kd: 0.45742098332667525, Tf: 0.4823884097135857, Dt: 0.1, B: 1, C: 1},
	} {
		if pidFRDStrictStable(num, den, c) {
			t.Fatalf("oracle finds %+v stable", c)
		}
		if stable, known := pidTuningFRDNyquistCount(c, p); stable && known {
			t.Errorf("strict count certifies unstable %+v", c)
		}
	}
}

// N4MOKA #16: a PID whose resonance loop passes 0.0038 from 0 between
// samples 6.6% apart (dense crossings at 0.3962 and 0.4102 rad/s) was
// certified with a closed-loop root at Re s = +2.9e-5. Both arcs fitted
// to the samples round the resonance put 0 within 0.001 of their circle,
// on either side, so the lens test alone let the search walk to it; an
// arc sweeping most of its circle that passes within ¼ of 0 (|S| > 4) now
// leaves the count unknown. PID, PIDF and PI at PM 60 come back stable
// with dense |S| <= 4, where base's were unstable or |S| above 13.
// Oracles: companion roots and a dense |S| sweep of the exact plant.
func TestTunePIDFRDResonanceArcNearOrigin(t *testing.T) {
	num, den := []float64{1.0387992186759023e-05}, []float64{1, 0.13699372147773697, 0.1644482602582385, 0.02195020295201065, 0.00021753977446639744}
	frd := pidFRDStrictWithDC(t, pidExactFRD(t, num, den, 0.000590353197528667, 119.30244754043795, 191), num, den)
	p, _ := pidTuningFRDStrictPlant(frd, nil)
	c := PID2{Kp: 92.55344502177579, Ki: 1.5343353197175762, Kd: -32.6931943547603, B: 1, C: 1}
	if re := pidClosedLoopMaxRe(num, den, c); !(re > 1e-5) {
		t.Fatalf("oracle max Re %g, want unstable", re)
	}
	if stable, known := pidTuningFRDNyquistCount(c, p); stable && known {
		t.Errorf("strict count certifies unstable %+v", c)
	}
	exact := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s)
	}
	for _, family := range []PidtuneType{PidtunePID, PidtunePIDF, PidtunePI} {
		r, err := TunePIDFRD(context.Background(), frd, family, PIDTuningOptions{PhaseMargin: 60})
		if err != nil {
			t.Errorf("%v: %v", family, err)
			continue
		}
		k := *r.Controller
		if re := pidClosedLoopMaxRe(num, den, k); !(re < 0) {
			t.Errorf("%v: %+v closed-loop root at Re %g (certified %v)", family, k, re, !pidHasWarning(r, "not certified"))
		}
		if peak, at := pidDenseSensitivity(k, exact, 1e-4, 1e3); peak > 4 {
			t.Errorf("%v: %+v dense |S| %g at %g", family, k, peak, at)
		}
		if pm := r.Evidence.AchievedPhaseMargin; pm < 58.5 {
			t.Errorf("%v: achieved PM %g", family, pm)
		}
	}
}
