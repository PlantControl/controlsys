package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// Least-squares C2D of a stiff descriptor counts integrators on the slow
// explicit part of the pencil. On the fold E⁻¹A, ‖A‖₁ ~ cond(E) inflated the
// integrator tolerance until a slow pole passed for one and the fit pinned a
// discrete pole at z = 1 (3PWI4Q). The draws are nearSingularEContinuousK
// seeds whose triangular twin (F, W) puts every pole, F_ii/W_ii, away from
// s = 0; bounds are the fit errors against the pencil at 60cf4cf.
func TestC2DLeastSquaresStiffDescriptorNoSpuriousIntegrator(t *testing.T) {
	const dt = 0.1
	for _, tc := range []struct {
		seed  uint64
		bound float64
	}{
		{43, 0.1913},
		{71, 0.491},
		{296, 0.222},
		{344, 0.9668},
	} {
		rng := rand.New(rand.NewPCG(tc.seed, 7704))
		n := 2 + rng.IntN(5)
		m, p := 1, 1
		if rng.IntN(3) == 0 {
			m, p = 1+rng.IntN(2), 1+rng.IntN(2)
		}
		if m != 1 || p != 1 {
			t.Fatalf("seed %d: draw is not SISO", tc.seed)
		}
		k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
		sys, twin := nearSingularEDraw(t, rng, n, 1, 1, k)
		if _, ok := foldInaccurateParts(sys); !ok {
			t.Fatalf("seed %d: fold is accurate; draw no longer exercises the pencil route", tc.seed)
		}
		for i := range n {
			if l := twin.A.At(i, i) / twin.E.At(i, i); math.Abs(l)*dt < 1e-3 {
				t.Fatalf("seed %d: pole %.3g near s = 0", tc.seed, l)
			}
		}
		opts := C2DOptions{Method: C2DMethodLeastSquares}
		d, err := sys.C2D(dt, opts)
		if err != nil {
			t.Fatalf("seed %d: C2D: %v", tc.seed, err)
		}
		poles, err := d.Poles()
		if err != nil {
			t.Fatal(err)
		}
		for _, z := range poles {
			if cmplx.Abs(z-1) < 1e-9 {
				t.Errorf("seed %d: discrete pole %v pinned at z = 1", tc.seed, z)
			}
		}
		if got := lsPencilFitError(t, d, twin, dt); got > tc.bound {
			t.Errorf("seed %d: C2D fit error vs pencil %.4g > base %.4g", tc.seed, got, tc.bound)
		}
		df, _, err := sys.C2DFit(dt, opts)
		if err != nil {
			t.Fatalf("seed %d: C2DFit: %v", tc.seed, err)
		}
		if got := lsPencilFitError(t, df, twin, dt); got > tc.bound {
			t.Errorf("seed %d: C2DFit fit error vs pencil %.4g > base %.4g", tc.seed, got, tc.bound)
		}
	}
}

// A stiff descriptor with a true pole at s = 0 keeps its integrator when it
// is counted on the slow part: the discrete fit pins one pole at z = 1 and
// matches the pencil response, r/s near s = 0, at low frequency.
func TestC2DLeastSquaresStiffDescriptorKeepsIntegrator(t *testing.T) {
	const dt = 0.1
	const n = 3
	rng := rand.New(rand.NewPCG(5, 7704))
	F := mat.NewDense(n, n, []float64{0, 0.7, -0.4, 0, -1.3, 0.9, 0, 0, -0.8})
	W := mat.NewDense(n, n, []float64{1, 0, 0, 0, 0.8, 0, 0, 0, 1e-13})
	b := mat.NewDense(n, 1, []float64{0.6, -1.1, 0.5})
	c := mat.NewDense(1, n, []float64{1.2, 0.4, -0.9})
	D := mat.NewDense(1, 1, []float64{0.3})
	U, V := randomOrthogonal(rng, n), randomOrthogonal(rng, n)
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	sys, err := NewDescriptor(&A, &B, &C, D, &E, 0)
	if err != nil {
		t.Fatal(err)
	}
	twin, err := NewDescriptor(F, b, c, D, W, 0)
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := foldInaccurateParts(sys); !ok {
		t.Fatal("fold is accurate; draw no longer exercises the pencil route")
	}
	opts := C2DOptions{Method: C2DMethodLeastSquares}
	d, err := sys.C2D(dt, opts)
	if err != nil {
		t.Fatal(err)
	}
	df, _, err := sys.C2DFit(dt, opts)
	if err != nil {
		t.Fatal(err)
	}
	for _, d := range []*System{d, df} {
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
		if pinned != 1 {
			t.Fatalf("poles %v: %d at z = 1, want 1", poles, pinned)
		}
		for _, w := range []float64{1e-4, 1e-3, 1e-2} {
			g, err := d.EvalFr(cmplx.Exp(complex(0, w*dt)))
			if err != nil {
				t.Fatal(err)
			}
			want := pencilResponse(t, twin, complex(0, w))[0][0]
			if e := cmplx.Abs(g[0][0]-want) / cmplx.Abs(want); e > 0.003 {
				t.Errorf("w=%g: fitted %v, pencil %v (rel %.3g)", w, g[0][0], want, e)
			}
		}
	}
}
