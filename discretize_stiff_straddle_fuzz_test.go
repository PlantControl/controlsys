package controlsys

import (
	"errors"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// fzStiffStraddle is the 5×5 pencil of the SCPYGG straddle family from the
// spec's values: a near-defective pair (δ = 10^v0, v0 ∈ [−9, −1]) with E
// scale 10^v1 (v1 ∈ [−5.5, −4]) tuned to the nominal stiff cut, a fifth E
// entry 10^v2 (v2 ∈ [−11, −9]) and rotations seeded by v3.
func fzStiffStraddle(t *testing.T, s *fzSpec) *System {
	t.Helper()
	delta := math.Pow(10, fzClamp(s.Vals[0], -9, -1))
	ep := math.Pow(10, fzClamp(s.Vals[1], -5.5, -4))
	e5 := math.Pow(10, fzClamp(s.Vals[2], -11, -9))
	rng := rand.New(rand.NewPCG(math.Float64bits(s.Vals[3]), 0x5c99))
	u, v := randomOrthogonal(rng, 5), randomOrthogonal(rng, 5)
	build := func(kap float64) *System {
		A0 := mat.NewDense(5, 5, []float64{
			-1, 0.7, 0.3, -0.2, 0.1,
			0, -2.5, 0.4, 0.2, -0.3,
			0, 0, -kap, 1, 0.2,
			0, 0, 0, -kap * (1 + delta), 0.3,
			0, 0, 0, 0, -1})
		E0 := mat.NewDense(5, 5, nil)
		for i, d := range []float64{1, 1, ep, ep, e5} {
			E0.Set(i, i, d)
		}
		var a, e, b, c, tmp mat.Dense
		tmp.Mul(u, A0)
		a.Mul(&tmp, v.T())
		tmp.Reset()
		tmp.Mul(u, E0)
		e.Mul(&tmp, v.T())
		b.Mul(u, mat.NewDense(5, 1, []float64{1, -0.5, 0.8, 1.2, 0.6}))
		c.Mul(mat.NewDense(1, 5, []float64{1, 0.7, -0.6, 0.9, 0.4}), v.T())
		sys, err := NewDescriptor(&a, &b, &c, mat.NewDense(1, 1, nil), &e, 0)
		if err != nil {
			t.Fatal(err)
		}
		return sys
	}
	kap := 1.0
	for range 60 {
		eq := equilibrateDescriptor(build(kap))
		kap = 1 / qzLargeRatio * mat.Norm(eq.A, 2) / mat.Norm(eq.E, 2) * ep / (1 + delta/2)
	}
	return build(kap)
}

// fzStiffStraddleArm asserts on the straddle pencil of s that a nil C2D
// ZOH or Tustin has its DC within the descriptor bar (10 bars), else the
// error wraps ErrSchurFailed (SCPYGG).
func fzStiffStraddleArm(t *testing.T, s *fzSpec) {
	t.Helper()
	sys := fzStiffStraddle(t, s)
	if !sys.IsDescriptor() || !descriptorFoldInaccurate(sys.E) {
		return
	}
	g0, bar := stiffSplitDC(t, sys)
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodTustin} {
		d, err := sys.C2D(0.01, C2DOptions{Method: m})
		if err != nil {
			if !errors.Is(err, ErrSchurFailed) && !errors.Is(err, ErrOverflow) {
				t.Fatalf("straddle arm: C2D %s: %v\n%s", m, err, s)
			}
			continue
		}
		if dc := stiffSplitDiscreteDC(t, d); math.Abs(dc-g0) > 10*bar {
			t.Fatalf("straddle arm: C2D %s nil with DC %g, exact %g (%g bars)\n%s", m, dc, g0, math.Abs(dc-g0)/bar, s)
		}
	}
}
