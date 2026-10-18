package controlsys

import (
	"context"
	"math"
	"math/cmplx"
	"testing"
)

// EDMYXL: G(s) = (−184.15s+35.68)/(s³+4.559s²+3.272s+14.24)·e^{−1.0817s}
// (Hurwitz: 4.559·3.272 > 14.24; RHP zero 0.194) is the PI back-off plant.
// At depths 9 and 13 the floor's lowest admitted band center needs PI phase
// −92.3° resp. −92.0°, 2° past the same-sign cone [−90°, 0°]: the exact seed
// had Kp < 0, and the pattern steps from it skipped the sliver within the
// band (0 < Kp ≲ 0.007 at depth 13), so the probe failed while depths 10-12
// succeeded. The design is checked on L = (Kp+Ki/s)·G evaluated directly:
// unit gain and band phase at wc, the 57° floor over a dense sweep, and
// closed-loop stability by the winding of 1+L.
func TestTunePIDBackOffEdgeSeed(t *testing.T) {
	num := []float64{-184.15361912744427, 35.680543192821474}
	den := []float64{1, 4.558547394863393, 3.2717623475043296, 14.23672605433042}
	const tau = 1.0816929318008033
	plant := pidPlantDelayed(t, num, den, tau)
	eval, err := newSISOEval(plant)
	if err != nil {
		t.Fatal(err)
	}
	screen, err := newPIDStabilityScreen(plant, eval, pidTuningMaxTail)
	if err != nil {
		t.Fatal(err)
	}
	wc0, err := findCrossoverFreq(plant, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	lo, hi := pidTuningDynamicsSpan(screen.model)
	g := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*tau)
	}
	for _, j := range []int{9, 13} {
		wc := wc0 * math.Pow(2, -float64(j)/2)
		p := pidTuningPlant{low: wc / 100, high: wc0 * 100, at: eval.at, gainSign: 1, tau: tau,
			maxSensitivity: pidTuningMaxSensitivity, modes: pidTuningModes(screen.model), floorRungs: 1, minTfSeed: true}
		p.crossLow, p.crossHigh = math.Min(p.low, lo/10), math.Max(p.high, hi*10)
		stable, evidence, err := screen.at(wc, PidtunePI)
		if err != nil {
			t.Fatal(err)
		}
		p.stable, p.stability = stable, evidence
		p.returnSensitivity, p.maxReturnSensitivity = screen.returnSensitivity, 1/(1-pidTuningMaxTail)
		r, err := tunePIDFloor(context.Background(), "TunePID", p, PidtunePI, PIDTuningOptions{CrossoverFrequency: wc, PhaseMargin: 60})
		if err != nil {
			t.Fatalf("depth %d (wc %.4g): %v", j, wc, err)
		}
		c := r.Controller
		if c.Kp < 0 || c.Ki <= 0 {
			t.Fatalf("depth %d: Kp %g Ki %g, want same sign as DC gain", j, c.Kp, c.Ki)
		}
		L := func(w float64) complex128 { return complex(c.Kp, -c.Ki/w) * g(w) }
		at := L(wc)
		band := 180 + cmplx.Phase(at)*180/math.Pi
		if math.Abs(cmplx.Abs(at)-1) > 1e-9 || band < 57 || band > 93 {
			t.Fatalf("depth %d: |L(jwc)| %g, margin %g°", j, cmplx.Abs(at), band)
		}
		if pm, _ := pidtuneLoopMarginBand(L, wc/1e3, wc0*1e3, 1.0002); pm < 57-1e-6 {
			t.Errorf("depth %d: dense margin %g° below the 57° floor", j, pm)
		}
		ok, certain := pidtuneOracleStable(t, &PID{Kp: c.Kp, Ki: c.Ki}, num, den, 0, tau)
		if !certain || !ok {
			t.Errorf("depth %d: winding stable %v (decided %v)", j, ok, certain)
		}
	}
}
