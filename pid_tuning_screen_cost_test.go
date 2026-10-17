package controlsys

import (
	"context"
	"errors"
	"math"
	"testing"
)

// Past a stable incumbent, candidates over the |S| bound that would neither
// replace it nor lower the least excess skip the Nyquist check (QZIL4R).
// 1/(s+10)e^{-0.3s} PDF at the first moved crossover fails on |S| alone;
// the outcome must match the unscreened search.
func TestTunePIDSkipsStabilityForLosingCandidates(t *testing.T) {
	const tau = .3
	plant := pidDelayedLag(t, 10, tau)
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
	sign := pidTuningDCSign(plant, eval.at, wc0)
	wc, ok := pidTuningPhaseCrossing(eval.at, sign, -180+60-15, wc0*1e-6, wc0*1e4, tau)
	if !ok {
		t.Fatal("no moved crossover")
	}
	stable, _, err := screen.at(wc, PidtunePDF)
	if err != nil || stable == nil {
		t.Fatalf("no Nyquist test: %v", err)
	}
	calls := 0
	p := pidTuningPlant{low: wc / 100, high: wc * 100, at: eval.at, stability: PIDStabilityNyquist,
		gainSign: sign, tau: tau, maxSensitivity: pidTuningMaxSensitivity, modes: pidTuningModes(plant),
		stable: func(c *PID2) (bool, error) { calls++; return stable(c) }}
	_, err = tunePID(context.Background(), "TunePID", p, PidtunePDF, PIDTuningOptions{CrossoverFrequency: wc})
	var te *PIDTuningError
	if !errors.As(err, &te) {
		t.Fatalf("err = %v, want PIDTuningError", err)
	}
	// Without the pre-screen: 235 checks, least excess 5.11520280723083.
	if calls > 100 {
		t.Errorf("stability checks = %d, want <= 100", calls)
	}
	if te.unstable || math.Abs(te.excess-5.11520280723083) > 1e-6*5.11520280723083 {
		t.Errorf("unstable = %v, excess = %v; want false, 5.11520280723083", te.unstable, te.excess)
	}
}
