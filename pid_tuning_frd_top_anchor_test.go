package controlsys

import (
	"context"
	"math"
	"math/rand/v2"
	"testing"
)

// M5M43J: continuous data stopping at 7.2 rad/s, below a ζ 0.0165 pole
// pair at 22.3 rad/s, look like a −1 slope at their top, but the phase
// turns back from −90° over the upper quarter decade as the pair nears.
// The top must not be taken as anchored there: the PIDF design, unstable
// on the exact plant, is returned unchanged but uncertified. A plant whose
// band runs past its poles (resonance sweep seed 446) keeps its top.
// Oracle: companion roots of the exact closed loop.
func TestTunePIDFRDTopUnanchoredBelowResonance(t *testing.T) {
	num := []float64{25.506210774400728, 829.8029209337119}
	den := []float64{1, 0.7357032104420134, 497.3597675085502, 248.16080700433128}
	frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, num, den, 0, 0.04708425628762577, 7.197059520389919, 71), num, den)
	r, err := TunePIDFRD(context.Background(), frd, PidtunePIDF, PIDTuningOptions{PhaseMargin: 70})
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	want := PID2{Kp: 0.14735832550618388, Ki: 0.5723998518542059, Kd: 1.040320161796169, Tf: 0.2664188586611643}
	for _, g := range [][2]float64{{k.Kp, want.Kp}, {k.Ki, want.Ki}, {k.Kd, want.Kd}, {k.Tf, want.Tf}} {
		if math.Abs(g[0]-g[1]) > 1e-6*math.Abs(g[1]) {
			t.Fatalf("%+v, base gave %+v", k, want)
		}
	}
	if re := pidClosedLoopMaxRe(num, den, k); !(re > 0) {
		t.Fatalf("oracle max Re %g, case no longer unstable", re)
	}
	if !pidHasWarning(r, "not certified") {
		t.Errorf("unstable %+v certified", k)
	}
	data, _, _ := pidTuningFRDSplitDC(frd)
	p := pidTuningFRDPlant(frd, nil)
	if n, ok := pidTuningFRDTopOrder(data, p.at); ok {
		t.Errorf("top anchored at order %d", n)
	}

	c := pidFRDResonanceSweepCase(rand.New(rand.NewPCG(2525, 446)))
	frd = pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, 0, c.lo, c.hi, c.n), c.num, c.den)
	data, _, _ = pidTuningFRDSplitDC(frd)
	p = pidTuningFRDPlant(frd, nil)
	if n, ok := pidTuningFRDTopOrder(data, p.at); !ok || n != -3 {
		t.Errorf("seed 446: top order %d %v, want -3 anchored", n, ok)
	}
}

// M5M43J: continuous data stopping 4.5-6 times below a ζ 0.01-0.03 pole
// pair, whose phase the pair turns under half a degree there, so only the
// bend it gives |G| over the top half decade shows it. Each design is
// returned unchanged, uncertified: the two PIDFs are unstable on the
// exact plant, the PID stable with dense |S| 14.6 at 48 rad/s, past the
// band, on the Markov end it took (Tf = 0). Oracles: companion roots and
// a dense exact |S| sweep.
func TestTunePIDFRDTopUnanchoredBelowFarResonance(t *testing.T) {
	for _, c := range []struct {
		pidFRDStrictCase
		unstable bool
	}{
		{pidFRDStrictCase{"pair 4.5x above PIDF", []float64{-8262.93623685102}, []float64{1, 0.7519020138192478, 1032.7699262257074, 358.01196203512444}, 0, 0.07926255008180683, 7.162852266317181, 117, PidtunePIDF, PIDTuningOptions{}, PID2{Kp: -0.04328738756302844, Ki: -0.15525853497588135, Kd: -0.17023952382149093, Tf: 0.2226680566048483}}, true},
		{pidFRDStrictCase{"pair 5x above PIDF", []float64{58.561483836812066}, []float64{1, 0.4948643611220235, 62.07187598018631, 5.326256369931688}, 0, 0.007795302566269365, 1.57856375781812, 31, PidtunePIDF, PIDTuningOptions{}, PID2{Kp: 0.03387099198725247, Ki: 0.037108072857484145, Kd: 1.8443099069106503, Tf: 1.3340800478048032}}, true},
		{pidFRDStrictCase{"pair 5.8x above PID", []float64{172.95458044720178, 243.93143446803765}, []float64{1, 2.869209679584981, 1419.286920970929, 2303.5392064029966, 880.1053701399039}, 0, 0.01493150412148546, 6.547781650196431, 159, PidtunePID, PIDTuningOptions{}, PID2{Kp: 0.336780194858159, Ki: 1.8053609921275728, Kd: 5.2798120749694615}}, false},
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
		re, peak := pidClosedLoopMaxRe(c.num, c.den, k), pidFRDStrictPeak(c.num, c.den, 0, k, c.lo)
		if c.unstable != (re > 0) || !c.unstable && !(peak > 4) {
			t.Errorf("%s: oracle max Re %g, dense |S| %g; case no longer pins its defect", c.name, re, peak)
		}
		if !pidHasWarning(r, "end above the band is established") {
			t.Errorf("%s: %+v certified, warnings %q", c.name, k, r.Evidence.Warnings)
		}
	}
}
