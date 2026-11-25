package controlsys

import (
	"errors"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// innerStraddleDescriptor is a 12-state pencil whose stiff modes 1e6–1e12
// are spaced by 8, so no gap moves the cut of splitFastPencil, and whose
// near-defective pair 1e9·(1 ∓ δ/2) lies on that cut 1e-3·ρ (SCPYGG).
func innerStraddleDescriptor(t *testing.T, delta float64) *System {
	t.Helper()
	poles := []float64{1e6, 8e6, 6.4e7, 5.12e8, 1e9 * (1 - delta/2), 1e9 * (1 + delta/2), 4.096e9, 3.2768e10, 2.62144e11, 1e12}
	n := 2 + len(poles)
	A0 := mat.NewDense(n, n, nil)
	E0 := mat.NewDense(n, n, nil)
	A0.Set(0, 0, -1)
	A0.Set(1, 1, -2.5)
	A0.Set(0, 1, 0.7)
	E0.Set(0, 0, 1)
	E0.Set(1, 1, 1)
	for k, p := range poles {
		i := 2 + k
		A0.Set(i, i, -1)
		E0.Set(i, i, 1/p)
		if i+1 < n {
			A0.Set(i, i+1, 0.3)
		}
		A0.Set(0, i, 0.2)
	}
	A0.Set(6, 7, 1)
	rng := rand.New(rand.NewPCG(5, 0x1e9))
	B0, C0 := mat.NewDense(n, 1, nil), mat.NewDense(1, n, nil)
	for i := range n {
		B0.Set(i, 0, rng.NormFloat64())
		C0.Set(0, i, rng.NormFloat64())
	}
	u, v := randomOrthogonal(rng, n), randomOrthogonal(rng, n)
	var a, e, b, c, tmp mat.Dense
	tmp.Mul(u, A0)
	a.Mul(&tmp, v)
	tmp.Reset()
	tmp.Mul(u, E0)
	e.Mul(&tmp, v)
	b.Mul(u, B0)
	c.Mul(C0, v)
	sys, err := NewDescriptor(&a, &b, &c, mat.NewDense(1, 1, nil), &e, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// An inner split of the fast part that is not certified is an error for
// every consumer but TransferFunction, which judges the fold it falls back
// to. The pair of innerReject (δ = 1e-9, frozen from arm64) rejects there on
// arm64; where rounding certifies the split instead, every C2D method is
// nil within the descriptor bar. A well-separated pair splits (SCPYGG).
func TestC2DStiffDescriptorInnerSplitRejectErrors(t *testing.T) {
	sys := stiffSplitLiteral(t, "innerReject")
	sp, ok := stiffExplicitSplit(sys)
	if !ok {
		t.Fatal("not on the stiff route")
	}
	if _, slow, fast, _, _, err, ok := stiffPencilSplit(sys); ok && err == nil && slow != nil && fast != nil {
		folded, ok := svdExplicit(fast)
		if !ok {
			t.Fatal("svdExplicit of the fast part failed")
		}
		rho, mods := 0.0, []float64(nil)
		for _, l := range eigenvaluesOf(t, folded.A) {
			rho = math.Max(rho, cmplx.Abs(l))
			mods = append(mods, cmplx.Abs(l))
		}
		cut := stiffCut(mods, qzLargeRatio*rho)
		if _, _, _, err := decomposeGeneralizedQZZeta(fast, func(l complex128) bool { return cmplx.Abs(l) < cut }, nil); err != nil && splitsBothSides(mods, cut) && sp.err == nil {
			t.Fatalf("inner split at %g fails (%v) but stiffExplicitSplit reports no error", cut, err)
		}
	}
	if sp.err != nil {
		if !errors.Is(sp.err, ErrSchurFailed) {
			t.Fatalf("split error %v does not wrap ErrSchurFailed", sp.err)
		}
		if len(sp.parts) != 2 {
			t.Errorf("fallback has %d parts, want the slow part and the fold", len(sp.parts))
		}
	}
	for _, m := range []C2DMethod{C2DMethodZOH, C2DMethodFOH, C2DMethodTustin, C2DMethodImpulse, C2DMethodMatched} {
		d, err := sys.C2D(0.01, C2DOptions{Method: m})
		switch {
		case sp.err != nil:
			if !errors.Is(err, ErrSchurFailed) || d != nil {
				t.Errorf("C2D %s = %v, %v; want nil, ErrSchurFailed", m, d, err)
			}
		case err != nil:
			if !errors.Is(err, ErrSchurFailed) || m != C2DMethodZOH && m != C2DMethodFOH {
				t.Errorf("C2D %s of a certified split: %v", m, err)
			}
		case m != C2DMethodImpulse:
			if r, ok := stiffSplitDCWithin(t, sys, d); !ok {
				t.Errorf("C2D %s nil with DC off by %g bars", m, r)
			}
		}
	}
	if sp.err != nil {
		if r, err := Step(sys, 0); !errors.Is(err, ErrSchurFailed) || r != nil {
			t.Errorf("Step = %v, %v; want nil, ErrSchurFailed", r, err)
		}
		if r, err := Impulse(sys, 0); !errors.Is(err, ErrSchurFailed) || r != nil {
			t.Errorf("Impulse = %v, %v; want nil, ErrSchurFailed", r, err)
		}
	}
	if res, err := sys.TransferFunction(nil); errors.Is(err, ErrSchurFailed) || res == nil {
		t.Errorf("TransferFunction = %v, %v; want a judged result", res, err)
	}
	apart := innerStraddleDescriptor(t, 0.5)
	if sp, ok := stiffExplicitSplit(apart); !ok || sp.err != nil || len(sp.parts) < 3 {
		t.Fatalf("separated pair: ok %v, err %v, %d parts; want an inner split", ok, sp.err, len(sp.parts))
	}
	if _, err := apart.C2D(0.01, C2DOptions{Method: C2DMethodTustin}); err != nil {
		t.Errorf("separated pair: C2D tustin: %v", err)
	}
}
