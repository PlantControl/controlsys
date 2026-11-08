package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// lsContractBreach reports a least-squares C2D or C2DFit return outside its
// contract: a nil error with a model, ErrAccuracyUnverified with a model,
// any other error with a nil model, and no unexported sentinel in the chain.
func lsContractBreach(out *System, err error) string {
	unverified := errors.Is(err, ErrAccuracyUnverified)
	switch {
	case errors.Is(err, errPencilFitUncertified):
		return fmt.Sprintf("unexported sentinel escapes: %v", err)
	case err == nil && out == nil:
		return "nil model with nil error"
	case unverified && out == nil:
		return fmt.Sprintf("nil model with %v", err)
	case err != nil && !unverified && out != nil:
		return fmt.Sprintf("model with error %v", err)
	}
	return ""
}

// lsUnverified checks a least-squares C2D or C2DFit return against its
// contract and reports whether it is ErrAccuracyUnverified; any other error
// is fatal.
func lsUnverified(t *testing.T, op string, out *System, err error) bool {
	t.Helper()
	if msg := lsContractBreach(out, err); msg != "" {
		t.Fatalf("%s: %s", op, msg)
	}
	if err != nil && !errors.Is(err, ErrAccuracyUnverified) {
		t.Fatalf("%s: %v", op, err)
	}
	return err != nil
}

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
// returns such a fit as is. C2DFit returns the same model as C2D, both
// with ErrAccuracyUnverified, as every fit through the pencil. Which rows
// have excess
// poles not stable depends on the platform rounding.
func TestC2DFitLeastSquaresStiffDescriptorExcessUnstableAccuracy(t *testing.T) {
	const dt = 0.1
	for _, tc := range []struct {
		name         string
		seed, stream uint64
		scaled       bool
	}{
		{"ns400/54", 54, 7704, false},
		{"ns1600/103", 103, 9911, false},
		{"ns1600/1281", 1281, 9911, false},
		{"ns1600/123x7", 123, 9911, true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, twin := lsStiffDraw(t, tc.seed, tc.stream, tc.scaled)
			if twinSlowNonStable(twin) != 0 {
				t.Fatal("plant slow part is not stable")
			}
			opts := C2DOptions{Method: C2DMethodLeastSquares}
			d, err := sys.C2D(dt, opts)
			if !lsUnverified(t, "C2D", d, err) {
				t.Error("C2D: fit through the pencil certified")
			}
			df, fit, err := sys.C2DFit(dt, opts)
			if !lsUnverified(t, "C2DFit", df, err) {
				t.Errorf("C2DFit: fit through the pencil certified with %d poles not stable against a stable plant", discreteNonStable(t, df))
			}
			if !mat.Equal(d.A, df.A) || !mat.Equal(d.B, df.B) || !mat.Equal(d.C, df.C) || !mat.Equal(d.D, df.D) {
				t.Fatal("C2DFit model differs from C2D")
			}
			ud := discreteNonStable(t, df)
			if fit.Stable != (ud == 0) {
				t.Errorf("Stable %v with %d poles not stable", fit.Stable, ud)
			}
		})
	}
}

// FitOrder 1, the minimal order and the MATLAB remedy for unstable poles
// and z = −1 cancellations, fits ns400 seed 54 and ns1600 seed 103 stable
// with ErrAccuracyUnverified (not certified), about as well as the fit of
// the exact twin at that order
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
			if !lsUnverified(t, "C2DFit", df, err) || !fit.Stable || fit.FitOrder != 1 {
				t.Errorf("fit %+v, %v; want stable order-1 fit with ErrAccuracyUnverified", fit, err)
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

// A FitOrder below the minimal one can drop a real unstable slow pole: the
// order-1 fit keeps a stable slow pole of the plant and loses the unstable
// one, against the slow poles F_ii/W_ii of the exact twin. C2DFit returns it
// with ErrAccuracyUnverified though Stable is true (Ud < Uc, 6QU2G3).
func TestC2DFitLeastSquaresStiffDescriptorDroppedUnstablePoleAccuracy(t *testing.T) {
	const dt = 0.1
	for _, tc := range []struct {
		name         string
		seed, stream uint64
	}{
		{"ns1600/68", 68, 9911},
		{"ns1600/311", 311, 9911},
		{"ns400/59", 59, 7704},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, twin := lsStiffDraw(t, tc.seed, tc.stream, false)
			if twinSlowNonStable(twin) != 1 {
				t.Fatalf("plant has %d unstable slow poles, want 1", twinSlowNonStable(twin))
			}
			df, fit, err := sys.C2DFit(dt, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: 1})
			unverified := lsUnverified(t, "C2DFit", df, err)
			if ud := discreteNonStable(t, df); ud != 0 {
				t.Fatalf("order-1 fit has %d poles not stable, want 0", ud)
			}
			if !fit.Stable || !unverified {
				t.Errorf("fit %+v, %v; want stable with ErrAccuracyUnverified", fit, err)
			}
		})
	}
}

// The count is against the slow part only: a positive stiff pole (F_nn/W_nn
// of the twin near 1e15) has no sampled image and base never represented it,
// so a stable fit keeps stable slow poles, with ErrAccuracyUnverified like
// every fit through the pencil (decision LPEXWJ).
func TestC2DFitLeastSquaresStiffDescriptorPositiveStiffPoleNotCounted(t *testing.T) {
	sys, twin := lsStiffDraw(t, 345, 7704, false)
	n, _, _ := twin.Dims()
	if stiff := twin.A.At(n-1, n-1) / twin.E.At(n-1, n-1); stiff < 1e12 {
		t.Fatalf("stiff pole %g, want large positive", stiff)
	}
	if twinSlowNonStable(twin) != 0 {
		t.Fatal("plant slow part is not stable")
	}
	df, fit, err := sys.C2DFit(0.1, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: 1})
	if !lsUnverified(t, "C2DFit", df, err) || !fit.Stable {
		t.Errorf("fit %+v, %v; want stable with ErrAccuracyUnverified", fit, err)
	}
}

// C2DFit of an explicit model, not sampled through a pencil, returns a nil
// error even for an unstable fit: the plant pole at s = 0.5 is kept.
func TestC2DFitLeastSquaresExplicitUnstableAccuracy(t *testing.T) {
	sys, err := New(mat.NewDense(2, 2, []float64{0.5, 1, 0, -2}), mat.NewDense(2, 1, []float64{0, 1}),
		mat.NewDense(1, 2, []float64{1, 0.3}), mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}
	df, fit, err := sys.C2DFit(0.1, C2DOptions{Method: C2DMethodLeastSquares})
	if lsUnverified(t, "C2DFit", df, err) || fit.Stable {
		t.Errorf("fit %+v, %v; want unstable with a nil error", fit, err)
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
		got, err := continuousNonStablePoleCount(sys, 0)
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
