package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// lsPencilFitError is the largest error of the discrete fit d against the
// pencil response of the continuous descriptor sys on a grid below Nyquist,
// relative to max(1, |G|), with G from a hand complex solve of (sE−A).
func lsPencilFitError(t *testing.T, d, sys *System, dt float64) float64 {
	t.Helper()
	worst := 0.0
	for q := 1; q < 200; q++ {
		theta := math.Pi * float64(q) / 200
		g, err := d.EvalFr(cmplx.Exp(complex(0, theta)))
		if err != nil {
			t.Fatal(err)
		}
		want := pencilResponse(t, sys, complex(0, theta/dt))[0][0]
		worst = math.Max(worst, cmplx.Abs(g[0][0]-want)/math.Max(1, cmplx.Abs(want)))
	}
	return worst
}

// Least-squares C2D of a descriptor with ill-conditioned E fits the pencil
// response (sE−A)⁻¹, as MATLAB c2d fits the continuous response, not that of
// the LU fold E⁻¹A, which moves the slow poles by cond(E)·ε (3PWI4Q).
// Seeds are nearSingularEContinuousK draws; base (fold target) errors were
// 116: 0.135, 171: 0.183, 190: 0.083, 237: 0.045, 355: 0.133, against
// 0.0095, 0.0195, 0.0085, 0.0020, 0.0105 when fitting the pencil.
func TestC2DLeastSquaresStiffDescriptorFitsPencil(t *testing.T) {
	const dt = 0.1
	for _, tc := range []struct {
		seed  uint64
		bound float64
	}{
		{116, 0.04},
		{171, 0.06},
		{190, 0.03},
		{237, 0.01},
		{355, 0.045},
	} {
		sys := nearSingularEContinuousK(t, tc.seed)
		if _, m, p := sys.Dims(); m != 1 || p != 1 {
			t.Fatalf("seed %d: draw is not SISO", tc.seed)
		}
		if _, ok := foldInaccurateParts(sys); !ok {
			t.Fatalf("seed %d: fold is accurate; draw no longer exercises the pencil target", tc.seed)
		}
		opts := C2DOptions{Method: C2DMethodLeastSquares}
		d, err := sys.C2D(dt, opts)
		if err != nil {
			t.Fatalf("seed %d: C2D: %v", tc.seed, err)
		}
		if got := lsPencilFitError(t, d, sys, dt); got > tc.bound {
			t.Errorf("seed %d: C2D fit error vs pencil %.4g > %.4g", tc.seed, got, tc.bound)
		}

		df, fit, err := sys.C2DFit(dt, opts)
		if err != nil {
			t.Fatalf("seed %d: C2DFit: %v", tc.seed, err)
		}
		if got := lsPencilFitError(t, df, sys, dt); got > tc.bound {
			t.Errorf("seed %d: C2DFit fit error vs pencil %.4g > %.4g", tc.seed, got, tc.bound)
		}
		n, _, _ := sys.Dims()
		samples := max(513, 32*n+1)
		maxErr, maxMag := 0.0, 0.0
		for k := range samples + 1 {
			w := math.Pi * (float64(k) + 0.5) / float64(samples+1) / dt
			g, err := df.EvalFr(cmplx.Exp(complex(0, w*dt)))
			if err != nil {
				t.Fatal(err)
			}
			want := pencilResponse(t, sys, complex(0, w))[0][0]
			maxErr = math.Max(maxErr, cmplx.Abs(g[0][0]-want))
			maxMag = math.Max(maxMag, cmplx.Abs(want))
		}
		if want := maxErr / maxMag; math.Abs(fit.MaxRelativeError-want) > 1e-6*want {
			t.Errorf("seed %d: C2DFit MaxRelativeError %.6g, want %.6g against the pencil", tc.seed, fit.MaxRelativeError, want)
		}
	}
}

// The pencil target also applies with an integer input delay, which the
// fit keeps as a discrete delay (3PWI4Q).
func TestC2DLeastSquaresStiffDescriptorDelayed(t *testing.T) {
	const dt = 0.1
	sys := nearSingularEContinuousK(t, 116)
	delayed := sys.Copy()
	delayed.InputDelay = []float64{0.2}
	d, err := delayed.C2D(dt, C2DOptions{Method: C2DMethodLeastSquares})
	if err != nil {
		t.Fatal(err)
	}
	if len(d.InputDelay) != 1 || d.InputDelay[0] != 2 {
		t.Fatalf("InputDelay = %v, want [2]", d.InputDelay)
	}
	d.InputDelay = nil
	if got := lsPencilFitError(t, d, sys, dt); got > 0.04 {
		t.Errorf("fit error vs pencil %.4g > 0.04", got)
	}
}
