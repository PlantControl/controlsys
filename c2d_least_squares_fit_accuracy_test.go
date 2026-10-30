package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"runtime"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// lsStiffDraw is the nearSingularEDraw of PCG(seed, stream) with the draw
// sequence of the reviewer census (6QU2G3): ns400 is stream 7704, ns1600
// stream 9911. scaled multiplies A and E of both by 7. It fails unless the
// draw is SISO and its fold is inaccurate (the pencil route).
func lsStiffDraw(t *testing.T, seed, stream uint64, scaled bool) (sys, twin *System) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, stream))
	n := 2 + rng.IntN(5)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	if m != 1 || p != 1 {
		t.Fatalf("seed %d: draw is not SISO", seed)
	}
	k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
	sys, twin = nearSingularEDraw(t, rng, n, 1, 1, k)
	if scaled {
		for _, s := range []*System{sys, twin} {
			s.A.Scale(7, s.A)
			s.E.Scale(7, s.E)
		}
	}
	if _, ok := foldInaccurateParts(sys); !ok {
		t.Fatalf("seed %d: fold is accurate; draw no longer exercises the pencil route", seed)
	}
	return sys, twin
}

// twinSlowNonStable counts the slow poles F_ii/W_ii of the triangular twin
// (all but the last, stiff, one) that the continuous classifier does not
// call stable.
func twinSlowNonStable(twin *System) int {
	n, _, _ := twin.Dims()
	count := 0
	for i := range n - 1 {
		p := complex(twin.A.At(i, i)/twin.E.At(i, i), 0)
		if poleOnOrOutsideStabilityBoundary(p, true, poleStabilityTolerance(p)) {
			count++
		}
	}
	return count
}

func discreteNonStable(t *testing.T, d *System) int {
	t.Helper()
	poles, err := d.Poles()
	if err != nil {
		t.Fatal(err)
	}
	count := 0
	for _, z := range poles {
		if poleOnOrOutsideStabilityBoundary(z, false, poleStabilityTolerance(z)) {
			count++
		}
	}
	return count
}

// C2DFit of a stable stiff descriptor at the auto order n, one above its
// minimal order, keeps a redundant near-cancelling pole whose side of the
// unit circle is set below float64 rounding of the target (6QU2G3); MATLAB
// returns such a fit as is. C2DFit returns the same model as C2D and reports
// it at Accuracy 1 exactly when it has more poles not stable than the plant
// slow poles (F_ii/W_ii of the exact twin). Where those land depends on the
// platform rounding, so each arch states the rows that fire there.
func TestC2DFitLeastSquaresStiffDescriptorExcessUnstableAccuracy(t *testing.T) {
	const dt = 0.1
	for _, tc := range []struct {
		name         string
		seed, stream uint64
		scaled       bool
		fires        map[string]bool
	}{
		{"ns400/54", 54, 7704, false, map[string]bool{"arm64": true, "amd64": false}},
		{"ns1600/103", 103, 9911, false, map[string]bool{"arm64": true, "amd64": false}},
		{"ns1600/1281", 1281, 9911, false, map[string]bool{"arm64": false, "amd64": true}},
		{"ns1600/123x7", 123, 9911, true, map[string]bool{"arm64": false, "amd64": true}},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, twin := lsStiffDraw(t, tc.seed, tc.stream, tc.scaled)
			if twinSlowNonStable(twin) != 0 {
				t.Fatal("plant slow part is not stable")
			}
			opts := C2DOptions{Method: C2DMethodLeastSquares}
			d, err := sys.C2D(dt, opts)
			if err != nil {
				t.Fatal(err)
			}
			df, fit, err := sys.C2DFit(dt, opts)
			if err != nil {
				t.Fatal(err)
			}
			if !mat.Equal(d.A, df.A) || !mat.Equal(d.B, df.B) || !mat.Equal(d.C, df.C) || !mat.Equal(d.D, df.D) {
				t.Fatal("C2DFit model differs from C2D")
			}
			ud := discreteNonStable(t, df)
			if fit.Stable != (ud == 0) {
				t.Errorf("Stable %v with %d poles not stable", fit.Stable, ud)
			}
			want := 0
			if ud > 0 {
				want = 1
			}
			if fit.Accuracy != want {
				t.Errorf("Accuracy %d with %d poles not stable against a stable plant, want %d", fit.Accuracy, ud, want)
			}
			if fires, ok := tc.fires[runtime.GOARCH]; ok && fires != (want == 1) {
				t.Errorf("%s: fit with %d poles not stable, want excess %v on this arch", runtime.GOARCH, ud, fires)
			}
		})
	}
}

// FitOrder 1, the minimal order and the MATLAB remedy for unstable poles
// and z = −1 cancellations, fits ns400 seed 54 and ns1600 seed 103 stable
// at Accuracy 0, about as well as the fit of the exact twin at that order
// (an accurately folded descriptor with the same response).
func TestC2DFitLeastSquaresStiffDescriptorReducedOrderStable(t *testing.T) {
	const dt = 0.1
	for _, tc := range []struct {
		name         string
		seed, stream uint64
	}{
		{"ns400/54", 54, 7704},
		{"ns1600/103", 103, 9911},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, twin := lsStiffDraw(t, tc.seed, tc.stream, false)
			opts := C2DOptions{Method: C2DMethodLeastSquares, FitOrder: 1}
			df, fit, err := sys.C2DFit(dt, opts)
			if err != nil {
				t.Fatal(err)
			}
			if !fit.Stable || fit.Accuracy != 0 || fit.FitOrder != 1 {
				t.Errorf("fit %+v, want stable order-1 fit at Accuracy 0", fit)
			}
			if stable, err := df.IsStable(); err != nil || !stable {
				t.Errorf("IsStable %v, %v", stable, err)
			}
			tw, err := twin.C2D(dt, opts)
			if err != nil {
				t.Fatal(err)
			}
			got, ref := lsPencilFitError(t, df, twin, dt), lsPencilFitError(t, tw, twin, dt)
			t.Logf("fit error vs twin pencil %.4g, twin fit %.4g", got, ref)
			if got > 1.01*ref+1e-6 {
				t.Errorf("fit error vs twin pencil %.4g above that of the twin fit %.4g", got, ref)
			}
		})
	}
}

// C2DFit of an explicit model, not sampled through a pencil, reports
// Accuracy 0 even for an unstable fit: the plant pole at s = 0.5 is kept.
func TestC2DFitLeastSquaresExplicitUnstableAccuracy(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{0.5, 1, 0, -2}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0.3}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	_, fit, err := sys.C2DFit(0.1, C2DOptions{Method: C2DMethodLeastSquares})
	if err != nil {
		t.Fatal(err)
	}
	if fit.Stable || fit.Accuracy != 0 {
		t.Errorf("fit %+v, want unstable at Accuracy 0", fit)
	}
}

// nonStablePoleCount counts the poles IsStable rejects, boundary included,
// and continuousNonStablePoleCount reads those of a quasi-triangular
// explicit part off its diagonal blocks as an eigensolve finds them.
func TestNonStablePoleCountMatchesIsStable(t *testing.T) {
	disc, err := New(mat.NewDense(4, 4, []float64{
		0.5, 1, 2, 3,
		0, 1, 4, 5,
		0, 0, 0.3, 2,
		0, 0, -0.5, 0.3,
	}), mat.NewDense(4, 1, []float64{1, 1, 1, 1}), mat.NewDense(1, 4, []float64{1, 1, 1, 1}), mat.NewDense(1, 1, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}
	if got, err := disc.nonStablePoleCount(); err != nil || got != 3 {
		t.Errorf("discrete count %d, %v, want 3 (z = 1 and 0.3 ± i)", got, err)
	}
	if stable, _ := disc.IsStable(); stable {
		t.Error("IsStable true with a pole at z = 1")
	}
	for _, tc := range []struct {
		a    []float64
		want int
	}{
		{[]float64{-1, 2, 3, 0, 0.5, 1, 0, -4, 0.5}, 2},
		{[]float64{-1, 2, 3, 0, -0.5, 1, 0, -4, -0.5}, 0},
		{[]float64{0, 2, 3, 0, -0.5, 1, 0, -4, -0.5}, 1},
		{[]float64{-1, 2, 3, 1, 0.5, 1, 0, -4, 0.5}, 2},
	} {
		sys, err := New(mat.NewDense(3, 3, tc.a), mat.NewDense(3, 1, []float64{1, 1, 1}), mat.NewDense(1, 3, []float64{1, 1, 1}), mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		got, err := continuousNonStablePoleCount(sys)
		if err != nil {
			t.Fatal(err)
		}
		poles, err := sys.Poles()
		if err != nil {
			t.Fatal(err)
		}
		oracle := 0
		for _, p := range poles {
			if real(p) >= -1e-10*math.Max(1, cmplx.Abs(p)) {
				oracle++
			}
		}
		if got != tc.want || oracle != tc.want {
			t.Errorf("A %v: count %d, eigensolve %d, want %d", tc.a, got, oracle, tc.want)
		}
	}
}
