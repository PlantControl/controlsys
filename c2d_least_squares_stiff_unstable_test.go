package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
)

// Least-squares C2D of a stiff descriptor with an unstable slow pole keeps
// it. The fit of the LU fold E⁻¹A, whose cond(E)·ε error moves the slow
// poles, returned a stable model (max |z| 0.9998); the pencil target keeps
// the pole within 3% of e^{λ·dt}, λ = F_ii/W_ii the twin's unstable pole
// (the fit does not map poles exactly; e^{λ·dt} = 1.152, fitted 1.1366),
// and within 2e-3 of the unstable pole of the fit of the twin itself, an
// accurately folded descriptor with the same response (6QU2G3, ns1600
// seed 68).
func TestC2DLeastSquaresStiffDescriptorKeepsUnstablePole(t *testing.T) {
	const dt = 0.1
	rng := rand.New(rand.NewPCG(68, 9911))
	n := 2 + rng.IntN(5)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	if m != 1 || p != 1 {
		t.Fatal("draw is not SISO")
	}
	k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
	sys, twin := nearSingularEDraw(t, rng, n, 1, 1, k)
	if _, ok := foldInaccurateParts(sys); !ok {
		t.Fatal("fold is accurate; draw no longer exercises the pencil route")
	}
	var want []complex128
	for i := range n {
		if l := twin.A.At(i, i) / twin.E.At(i, i); l > 0 && l*dt < 10 {
			want = append(want, complex(math.Exp(l*dt), 0))
		}
	}
	if len(want) != 1 {
		t.Fatalf("draw has %d unstable slow poles, want 1", len(want))
	}
	opts := C2DOptions{Method: C2DMethodLeastSquares}
	dt0, err := twin.C2D(dt, opts)
	if err != nil {
		t.Fatal(err)
	}
	twinPoles, err := dt0.Poles()
	if err != nil {
		t.Fatal(err)
	}
	twinUnstable := complex(0, 0)
	for _, z := range twinPoles {
		if cmplx.Abs(z-want[0]) < cmplx.Abs(twinUnstable-want[0]) {
			twinUnstable = z
		}
	}
	d, err := sys.C2D(dt, opts)
	if err != nil {
		t.Fatal(err)
	}
	df, fit, err := sys.C2DFit(dt, opts)
	if err != nil {
		t.Fatal(err)
	}
	if fit.Stable {
		t.Error("C2DFit reports a stable fit of an unstable plant")
	}
	if fit.Accuracy != 0 {
		t.Errorf("Accuracy %d for a fit whose unstable pole is the plant's", fit.Accuracy)
	}
	for _, d := range []*System{d, df} {
		poles, err := d.Poles()
		if err != nil {
			t.Fatal(err)
		}
		best, bestTwin := math.Inf(1), math.Inf(1)
		for _, z := range poles {
			best = math.Min(best, cmplx.Abs(z-want[0]))
			bestTwin = math.Min(bestTwin, cmplx.Abs(z-twinUnstable))
		}
		if best > 0.03*cmplx.Abs(want[0]) {
			t.Errorf("poles %v: none within 3%% of e^{λ·dt} = %v", poles, want[0])
		}
		if bestTwin > 2e-3*cmplx.Abs(twinUnstable) {
			t.Errorf("poles %v: none within 2e-3 relative of the twin fit pole %v", poles, twinUnstable)
		}
		if got := lsPencilFitError(t, d, twin, dt); got > 0.01 {
			t.Errorf("fit error vs pencil %.4g > 0.01", got)
		}
	}
}
