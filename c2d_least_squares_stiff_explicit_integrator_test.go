package controlsys

import (
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func stiffExplicitTwin(t *testing.T, seed, stream uint64) (sys, explicit *System, siso bool) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, stream))
	n := 2 + rng.IntN(5)
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(2), 1+rng.IntN(2)
	}
	k := math.Exp(math.Log(0.2) + rng.Float64()*(math.Log(2.5)-math.Log(0.2)))
	sys, twin := nearSingularEDraw(t, rng, n, m, p, k)
	explicit, err := twin.ToExplicit()
	if err != nil {
		return sys, nil, false
	}
	return sys, explicit, m == 1 && p == 1
}

// Least-squares C2D of the explicit fold of a stiff triangular twin (F, W)
// fits no integrator: its slow pencil poles, F_ii/W_ii = 0.88, 0.0166,
// {1.75, 0.45, −0.047} and −0.53, lie far below 1e-10·‖A‖₁ ≈ 1e5 beside
// the 1e15 pole but are located to ε of themselves by componentwise
// rounding (KEEWB4). Base pinned a pole at z = 1, fit errors against the
// pencil 0.191, 0.516, 0.541, 0.968; bounds are the largest error over 31
// fits of the twin with entries rounded by ±ε (PCG(seed,9002)) at a4c2e9b.
func TestC2DLeastSquaresStiffExplicitNoSpuriousIntegrator(t *testing.T) {
	const dt = 0.1
	for _, tc := range []struct {
		seed  uint64
		bound float64
	}{
		{43, 0.0112},
		{71, 0.0481},
		{296, 0.0952},
		{344, 0.0830},
	} {
		sys, explicit, siso := stiffExplicitTwin(t, tc.seed, 7704)
		if !siso {
			t.Fatalf("seed %d: draw is not SISO", tc.seed)
		}
		if k, _, err := leastSquaresIntegratorLimit(explicit, true, 0.1); err != nil || k != 0 {
			t.Errorf("seed %d: integrators %d (err %v), want 0", tc.seed, k, err)
		}
		opts := C2DOptions{Method: C2DMethodLeastSquares}
		d, err := explicit.C2D(dt, opts)
		if err != nil {
			t.Fatalf("seed %d: C2D: %v", tc.seed, err)
		}
		df, _, err := explicit.C2DFit(dt, opts)
		if err != nil {
			t.Fatalf("seed %d: C2DFit: %v", tc.seed, err)
		}
		for _, d := range []*System{d, df} {
			poles, err := d.Poles()
			if err != nil {
				t.Fatal(err)
			}
			for _, z := range poles {
				if cmplx.Abs(z-1) < 1e-9 {
					t.Errorf("seed %d: discrete pole %v pinned at z = 1", tc.seed, z)
				}
			}
			if got := lsPencilFitError(t, d, sys, dt); got > tc.bound {
				t.Errorf("seed %d: fit error vs pencil %.4g > rounding band %.4g", tc.seed, got, tc.bound)
			}
		}
	}
}

// The integrator count of the explicit fold of every SISO stiff triangular
// twin is zero: no pole F_ii/W_ii is zero, and each is located away from
// s = 0 by componentwise rounding however small against the 1e15 pole. At
// base 10 of the 82 tested draws, 5 per stream, counted one (KEEWB4).
func TestLeastSquaresIntegratorLimitStiffExplicitFamily(t *testing.T) {
	tested := 0
	for _, stream := range []uint64{7704, 9911} {
		for seed := range uint64(401) {
			_, explicit, siso := stiffExplicitTwin(t, seed, stream)
			if !siso {
				continue
			}
			tested++
			k, _, err := leastSquaresIntegratorLimit(explicit, true, 0.1)
			if err != nil {
				t.Fatalf("stream %d seed %d: %v", stream, seed, err)
			}
			if k != 0 {
				t.Errorf("stream %d seed %d: integrators %d, want 0", stream, seed, k)
			}
		}
	}
	if tested == 0 {
		t.Fatal("no SISO draw with an explicit fold")
	}
}

// Integrators within rounding of s = 0 or, at dt = 0.1, of z = 1 are
// counted, and only those: an exact zero pole and chain beside a stiff pole,
// where base also counted the slow −0.5 below 1e-10·‖A‖₁ = 100 (KEEWB4); a
// pole 1e-12, e^{λ·dt} within the IsStable band 1e-10 of z = 1, but not
// 3e-9 beside −49, which base counted below 1e-10·‖A‖₁ ≈ 5e-9 though
// fitted free it matches the response better; and the rounded zero,
// Jordan pair and Jordan triple of a dense similarity T·J·T⁻¹, whose
// computed poles spread to ε^{1/k}·‖A‖.
func TestLeastSquaresIntegratorLimitCountsRoundedIntegrators(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	T := mat.NewDense(3, 3, nil)
	for i := range T.RawMatrix().Data {
		T.RawMatrix().Data[i] = rng.NormFloat64()
	}
	var Ti mat.Dense
	if err := Ti.Inverse(T); err != nil {
		t.Fatal(err)
	}
	ones := mat.NewDense(3, 1, []float64{1, 1, 1})
	similar := func(j []float64) *mat.Dense {
		var tj, a mat.Dense
		tj.Mul(T, mat.NewDense(3, 3, j))
		a.Mul(&tj, &Ti)
		return &a
	}
	for _, tc := range []struct {
		name string
		a    *mat.Dense
		want int
	}{
		{"exact", mat.NewDense(3, 3, []float64{0, 1, 0, 0, -1, 2, 0, 0, -2}), 1},
		{"exact stiff", mat.NewDense(3, 3, []float64{0, 1, 0.3, 0, -0.5, 1, 0, 0, -1e12}), 1},
		{"exact chain stiff", mat.NewDense(3, 3, []float64{0, 1, 0.3, 0, 0, 1, 0, 0, -1e12}), 2},
		{"pole mapped to z = 1 within the boundary band", mat.NewDense(3, 3, []float64{1e-12, 1, 0, 0, -1, 1, 0, 0, -2}), 1},
		{"pole mapped outside the boundary band", mat.NewDense(3, 3, []float64{3e-9, 1, 0, 0, -49, 1, 0, 0, -2}), 0},
		{"dense zero", similar([]float64{0, 0, 0, 0, -1, 0, 0, 0, -2}), 1},
		{"dense Jordan pair", similar([]float64{0, 1, 0, 0, 0, 0, 0, 0, -2}), 2},
		{"dense Jordan triple", similar([]float64{0, 1, 0, 0, 0, 1, 0, 0, 0}), 3},
	} {
		var b, c mat.Dense
		b.Mul(T, ones)
		c.Mul(ones.T(), &Ti)
		sys, err := New(tc.a, &b, &c, mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		minimal, err := sys.MinimalRealization()
		if err != nil {
			t.Fatal(err)
		}
		if minimal.Order < tc.want {
			t.Fatalf("%s: minimal order %d < %d", tc.name, minimal.Order, tc.want)
		}
		if k, _, err := leastSquaresIntegratorLimit(sys, true, 0.1); err != nil || k != tc.want {
			t.Errorf("%s: integrators %d (err %v), want %d", tc.name, k, err, tc.want)
		}
	}
}
