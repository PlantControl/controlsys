package controlsys

import (
	"fmt"
	"testing"
)

// TestDelayNyquistRefineModelCost: on the PID loop of
// BenchmarkDiskMargin_PIDExactDelay, a span certified by the model at one end
// over its whole width needs no model at the other, so refine evaluates
// about one descriptor sample per two nodes, where it took one per node
// outside the tail before. The root counts are pinned against the rectangle
// argument-principle oracle.
func TestDelayNyquistRefineModelCost(t *testing.T) {
	for _, k := range []float64{1, 2, 4, 8} {
		q := pidDelayQuasiLoop(fmt.Sprintf("k=%g", k), k)
		t.Run(q.name, func(t *testing.T) {
			want := oracleRHPRoots(t, q.chi, q.box, q.den, q.num)
			l, err := delayLoopFromSystem(q.system(t), "delay loop")
			if err != nil {
				t.Fatal(err)
			}
			res, err := l.nyquist(0.5)
			if err != nil || res.roots != want || res.stable != (want == 0) {
				t.Fatalf("roots=%d stable=%v err=%v, oracle %d", res.roots, res.stable, err, want)
			}
			c := l.cert.(*descriptorCertifier)
			evals := 0
			for _, r := range c.r {
				if r != nil {
					evals += r.evals
				}
			}
			if ratio := float64(evals) / float64(len(res.w)); ratio > 0.5 {
				t.Errorf("%d descriptor samples for %d nodes (%.2f per node), want <= 0.5", evals, len(res.w), ratio)
			}
			t.Logf("oracle %d, %d samples, %d nodes", want, evals, len(res.w))
		})
	}
}
