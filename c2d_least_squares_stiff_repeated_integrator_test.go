package controlsys

import (
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// A repeated integrator of a stiff descriptor spreads on the slow part of the
// pencil split to about ±1e-8, real or imaginary by platform rounding, so the
// tolerance of IsStable can call the negative pole stable. Accuracy counts the
// chain at its true multiplicity (6QU2G3 review): it is 1 exactly when the fit
// and the triangular twin, whose poles are F_ii/W_ii, disagree on the count of
// poles not stable.
func TestC2DFitAccuracyCountsRepeatedIntegrator(t *testing.T) {
	const stiffness = 1e-13
	for _, tc := range []struct {
		name     string
		f        []float64
		w        []float64
		b        []float64
		c        []float64
		plantNot int
		minEqual int
	}{
		{
			name:     "double",
			f:        []float64{0, 0.7, -0.4, 0, 0, 0.9, 0, 0, -0.8},
			w:        []float64{1, 0.8, stiffness},
			b:        []float64{0.6, -1.1, 0.5},
			c:        []float64{1.2, 0.4, -0.9},
			plantNot: 2,
			minEqual: 20,
		},
		{
			name: "double with unstable slow pole",
			f: []float64{
				0, 0.7, -0.4, 0.3,
				0, 0, 0.9, -0.2,
				0, 0, 0.2, 0.5,
				0, 0, 0, -0.8,
			},
			w:        []float64{1, 0.8, 1, stiffness},
			b:        []float64{0.6, -1.1, 0.7, 0.5},
			c:        []float64{1.2, 0.4, 0.8, -0.9},
			plantNot: 3,
			minEqual: 1,
		},
		{
			name: "double with hidden unstable slow pole",
			f: []float64{
				0, 0.7, -0.4, 0.3,
				0, 0, 0, -0.2,
				0, 0, 0.5, 0,
				0, 0, 0, -0.8,
			},
			w:        []float64{1, 0.8, 1, stiffness},
			b:        []float64{0.6, -1.1, 0, 0.5},
			c:        []float64{1.2, 0.4, 0.8, -0.9},
			plantNot: 3,
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			routed, equal, differ := 0, 0, 0
			for seed := range uint64(40) {
				sys := stiffRotatedDescriptor(t, rand.New(rand.NewPCG(seed, 60608)), tc.f, tc.w, tc.b, tc.c)
				if _, ok := foldInaccurateParts(sys); !ok {
					continue
				}
				routed++
				for _, dt := range []float64{0.01, 0.1, 1} {
					d, fit, err := sys.C2DFit(dt, C2DOptions{Method: C2DMethodLeastSquares})
					if err != nil {
						t.Fatalf("seed %d dt %g: %v", seed, dt, err)
					}
					poles, err := d.Poles()
					if err != nil {
						t.Fatal(err)
					}
					pinned := 0
					for _, z := range poles {
						if cmplx.Abs(z-1) < 1e-9 {
							pinned++
						}
					}
					if pinned != 2 {
						t.Errorf("seed %d dt %g: poles %v, %d at z = 1, want 2", seed, dt, poles, pinned)
					}
					ud := discreteNonStable(t, d)
					want := 0
					if ud != tc.plantNot {
						want = 1
						differ++
					} else {
						equal++
					}
					if fit.Accuracy != want {
						t.Errorf("seed %d dt %g: Ud %d, plant %d, Accuracy %d, want %d", seed, dt, ud, tc.plantNot, fit.Accuracy, want)
					}
				}
			}
			if routed == 0 {
				t.Fatal("no draw exercises the pencil route")
			}
			if equal < tc.minEqual {
				t.Errorf("Ud == plant on %d fits (%d differ), want at least %d", equal, differ, tc.minEqual)
			}
			t.Logf("routed %d, Ud == plant %d, differ %d", routed, equal, differ)
		})
	}
}

// Below the plant's count the fit keeps the chain and drops the unstable slow
// pole, so Accuracy stays 1 with the chain counted at its multiplicity: order 2
// holds only the two poles pinned at z = 1 against a plant count of 3.
func TestC2DFitAccuracyRepeatedIntegratorDroppedSlowPole(t *testing.T) {
	f := []float64{
		0, 0.7, -0.4, 0.3,
		0, 0, 0.9, -0.2,
		0, 0, 0.2, 0.5,
		0, 0, 0, -0.8,
	}
	w := []float64{1, 0.8, 1, 1e-13}
	b := []float64{0.6, -1.1, 0.7, 0.5}
	c := []float64{1.2, 0.4, 0.8, -0.9}
	for seed := range uint64(8) {
		sys := stiffRotatedDescriptor(t, rand.New(rand.NewPCG(seed, 60608)), f, w, b, c)
		if _, ok := foldInaccurateParts(sys); !ok {
			t.Fatalf("seed %d: fold is accurate; draw no longer exercises the pencil route", seed)
		}
		d, fit, err := sys.C2DFit(0.1, C2DOptions{Method: C2DMethodLeastSquares, FitOrder: 2})
		if err != nil {
			t.Fatalf("seed %d: %v", seed, err)
		}
		if ud := discreteNonStable(t, d); ud != 2 || fit.Accuracy != 1 {
			t.Errorf("seed %d: Ud %d, Accuracy %d, want 2 and 1", seed, ud, fit.Accuracy)
		}
	}
}

// The k poles nearest s = 0 count as not stable whatever side roundoff puts
// them on; every other pole is classified by the IsStable tolerance.
func TestContinuousNonStablePoleCountRepeatedIntegrator(t *testing.T) {
	const spread = 1e-8
	for _, tc := range []struct {
		name        string
		a           []float64
		integrators int
		want        int
	}{
		{"split chain unstable pole", []float64{0, 1, 0, spread * spread, 0, 0, 0, 0, 0.2}, 2, 3},
		{"split chain stable pole", []float64{0, 1, 0, spread * spread, 0, 0, 0, 0, -0.2}, 2, 2},
		{"split chain unranked", []float64{0, 1, 0, spread * spread, 0, 0, 0, 0, -0.2}, 0, 1},
		{"single, stable near pole", []float64{0, 0, 0, 0, -1e-3, 0, 0, 0, -0.2}, 1, 1},
		{"rank above order", []float64{0, 0, 0, 0, -1e-3, 0, 0, 0, -0.2}, 4, 3},
	} {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := New(mat.NewDense(3, 3, tc.a), mat.NewDense(3, 1, []float64{1, 1, 1}), mat.NewDense(1, 3, []float64{1, 1, 1}), mat.NewDense(1, 1, nil), 0)
			if err != nil {
				t.Fatal(err)
			}
			got, err := continuousNonStablePoleCount(sys, tc.integrators)
			if err != nil {
				t.Fatal(err)
			}
			if got != tc.want {
				t.Errorf("count %d, want %d", got, tc.want)
			}
		})
	}
}

// stiffRotatedDescriptor returns the SISO descriptor U(F, diag W, b, c)V with
// random orthogonal U and V.
func stiffRotatedDescriptor(t *testing.T, rng *rand.Rand, f, w, b, c []float64) *System {
	t.Helper()
	n := len(w)
	U, V := randomOrthogonal(rng, n), randomOrthogonal(rng, n)
	var T, A, E, B, C mat.Dense
	T.Mul(U, mat.NewDense(n, n, f))
	A.Mul(&T, V)
	T.Mul(U, mat.NewDiagDense(n, w))
	E.Mul(&T, V)
	B.Mul(U, mat.NewDense(n, 1, b))
	C.Mul(mat.NewDense(1, n, c), V)
	sys, err := NewDescriptor(&A, &B, &C, mat.NewDense(1, 1, []float64{0.3}), &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}
