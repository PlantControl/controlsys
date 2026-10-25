package controlsys

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"
)

// IJCIOS: certified designs whose reported margin read 5-17° above the
// exact plant's, each design unchanged as only the reading is. A crossing
// just below the band, |L| = 1 + 1.3e-6 at the lowest sample over
// |Kp·G(0)| = 0.73, is now read along the count's low line; a crossing
// pair a mild curvature lifts between samples whose |L| stops 1.3e-6
// short of 1, and chord crossings 5-6° off across a curved step, along
// the steps' Möbius fits. Each now reports within 1° of the dense margin.
// 7WK6K5's crossover on a sample with a second crossing in the step is
// pinned too. Oracles: companion roots and dense margin sweeps of the
// exact plant.
func TestTunePIDFRDMarginReadsOffChord(t *testing.T) {
	res2622 := pidFRDResonanceSweepCase(rand.New(rand.NewPCG(2525, 2622)))
	res2622.name, res2622.k = "curved step PD", PID2{Kp: -0.0010673078157109148, Kd: 0.00374572587343424}
	for _, c := range []pidFRDStrictCase{
		{"below band PD", []float64{189082.69285889325}, []float64{1, 10.344396001890862, 1102.2926059493639, 2003.2145794532717, 247.3969662007516, 326.12067140980173}, 0, 0.07489266251998829, 353.8257925978028, 95, PidtunePD, PIDTuningOptions{}, PID2{Kp: -0.0012577041170745977, Kd: 0.014630773166788512}},
		{"lifted pair PIDF", []float64{-0.03855014297874039, 0.12503789679671823, 0.05216204572406545}, []float64{1, 0.8658537767296459, 0.8862316565660472, 0.2763260177233865, 0.006248543207942998}, 0, 0.0027932447333666147, 76.86776449520947, 181, PidtunePIDF, PIDTuningOptions{}, PID2{Kp: 1.7204626900640372, Ki: 0.4694574544256828, Kd: 2.1105117168537655, Tf: 0.002158123290689441}},
		{"lone fit PID", []float64{-259.35715939357885}, []float64{1, 7.913099316450911, 1984.9945613272291}, 0, 0.9763137013294461, 4449.408230909216, 34, PidtunePID, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: -6.816475320985241, Ki: -287.8045583667242, Kd: -0.1313441769899894}},
		res2622,
		{"crossover on sample PI", []float64{-0.6705834281487829}, []float64{1, 0.6532903586465822, 0.4992377215072653, 0.1815623484993936}, 0, 0.0466686439537967, 7.442890258481785, 51, PidtunePI, PIDTuningOptions{PhaseMargin: 60}, PID2{Kp: -0.14964961390937317, Ki: -0.00019857414660280298}},
	} {
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, 0, c.lo, c.hi, c.n), c.num, c.den)
		r, err := TunePIDFRD(context.Background(), frd, c.family, c.o)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		k := *r.Controller
		for _, g := range [][2]float64{{k.Kp, c.k.Kp}, {k.Ki, c.k.Ki}, {k.Kd, c.k.Kd}, {k.Tf, c.k.Tf}} {
			if math.Abs(g[0]-g[1]) > 1e-6*math.Abs(g[1]) {
				t.Errorf("%s: %+v, base gave %+v", c.name, k, c.k)
				break
			}
		}
		if !pidFRDStrictStable(c.num, c.den, k) {
			t.Errorf("%s: %+v unstable", c.name, k)
		}
		least := pidFRDDenseMargin(c.num, c.den, k, c.lo/1e3, c.hi*1e3)
		if pm := r.Evidence.AchievedPhaseMargin; pidHasWarning(r, "not certified") || math.Abs(pm-least) > 1 {
			t.Errorf("%s: certified %v, achieved margin %g, dense %g", c.name, !pidHasWarning(r, "not certified"), pm, least)
		}
	}
}
