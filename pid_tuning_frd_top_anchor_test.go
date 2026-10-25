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
