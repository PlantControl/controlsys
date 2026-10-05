package controlsys

import (
	"math/cmplx"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

func TestModsep_ClearSeparation(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -10, 0, 0, 0, -100}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 5)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 1 {
		t.Errorf("slow order = %d, want 1", ns)
	}
	if nf != 2 {
		t.Errorf("fast order = %d, want 2", nf)
	}

	slowPoles, _ := res.Slow.Poles()
	for _, p := range slowPoles {
		if cmplx.Abs(p) >= 5 {
			t.Errorf("slow pole %v has |λ| >= 5", p)
		}
	}

	fastPoles, _ := res.Fast.Poles()
	for _, p := range fastPoles {
		if cmplx.Abs(p) < 5 {
			t.Errorf("fast pole %v has |λ| < 5", p)
		}
	}

	checkAdditiveDecomposition(t, sys, res.Slow, res.Fast)
}

func TestModsep_DifferentCutoff(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -10, 0, 0, 0, -100}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 50)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 2 {
		t.Errorf("slow order = %d, want 2", ns)
	}
	if nf != 1 {
		t.Errorf("fast order = %d, want 1", nf)
	}

	checkAdditiveDecomposition(t, sys, res.Slow, res.Fast)
}

func TestModsep_AllFast(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{-1, 0, 0, 0, -10, 0, 0, 0, -100}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 0.1)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 0 {
		t.Errorf("slow order = %d, want 0", ns)
	}
	if nf != 3 {
		t.Errorf("fast order = %d, want 3", nf)
	}
}

func TestModsep_Discrete(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{0.1, 0, 0, 0, 0.5, 0, 0, 0, 0.99}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1, 0, 0, 0, 1}),
		mat.NewDense(3, 3, nil), 0.1)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 0.3)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 1 {
		t.Errorf("slow order = %d, want 1", ns)
	}
	if nf != 2 {
		t.Errorf("fast order = %d, want 2", nf)
	}

	checkAdditiveDecomposition(t, sys, res.Slow, res.Fast)
}

func TestModsep_InvalidCutoff(t *testing.T) {
	sys, _ := New(
		mat.NewDense(2, 2, []float64{-1, 0, 0, -2}),
		mat.NewDense(2, 1, []float64{1, 1}),
		mat.NewDense(1, 2, []float64{1, 0}),
		mat.NewDense(1, 1, []float64{0}), 0)

	_, err := Modsep(sys, -1)
	if err == nil {
		t.Error("expected error for negative cutoff")
	}
}

func TestModsep_Empty(t *testing.T) {
	sys, _ := NewGain(mat.NewDense(1, 1, []float64{5}), 0)

	res, err := Modsep(sys, 1.0)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	if ns != 0 {
		t.Errorf("slow order = %d, want 0", ns)
	}
}

func TestModsep_NonDiagonal(t *testing.T) {
	sys, err := New(
		mat.NewDense(3, 3, []float64{
			-1, 0.5, 0,
			0, -10, 0,
			0, 0, -100,
		}),
		mat.NewDense(3, 1, []float64{1, 1, 1}),
		mat.NewDense(1, 3, []float64{1, 1, 1}),
		mat.NewDense(1, 1, []float64{0}), 0)
	if err != nil {
		t.Fatal(err)
	}

	res, err := Modsep(sys, 5)
	if err != nil {
		t.Fatal(err)
	}

	ns, _, _ := res.Slow.Dims()
	nf, _, _ := res.Fast.Dims()

	if ns != 1 {
		t.Errorf("slow order = %d, want 1", ns)
	}
	if nf != 2 {
		t.Errorf("fast order = %d, want 2", nf)
	}

	checkAdditiveDecomposition(t, sys, res.Slow, res.Fast)
}

func TestModsep_Descriptor(t *testing.T) {
	for _, dt := range []float64{0, 0.1} {
		sys := descriptorSplitFixture(t, dt)
		cutoff := 1.0
		wantSlow := 0
		for _, ev := range generalizedEigOracle(t, sys.A, sys.E) {
			if cmplx.Abs(ev) < cutoff {
				wantSlow++
			}
		}
		if wantSlow == 0 || wantSlow == 3 {
			t.Fatalf("dt=%v: fixture not mixed (%d slow)", dt, wantSlow)
		}
		res, err := Modsep(sys, cutoff)
		if err != nil {
			t.Fatal(err)
		}
		if ns, _, _ := res.Slow.Dims(); ns != wantSlow {
			t.Errorf("dt=%v: slow order %d, want %d", dt, ns, wantSlow)
		}
		assertSplitSum(t, "Modsep descriptor", sys, res.Slow, res.Fast)
	}
}

func TestModsep_SingularDescriptor(t *testing.T) {
	for _, tc := range []struct {
		dt, slow, fast float64
	}{
		{0, -0.6, -2.5},
		{0.1, 0.4, -1.7},
	} {
		sys, modal, poly := singularSplitFixture(t, tc.slow, tc.fast, tc.dt)
		res, err := Modsep(sys, 1)
		if err != nil {
			t.Fatalf("dt=%v: %v", tc.dt, err)
		}
		if ns, _, _ := res.Slow.Dims(); ns != 1 || res.Slow.IsDescriptor() {
			t.Fatalf("dt=%v: slow part order %d descriptor=%v, want explicit order 1", tc.dt, ns, res.Slow.IsDescriptor())
		}
		assertResponseParts(t, "slow part", res.Slow, nil, func(s complex128) [][]complex128 { return modal(s, 0) })
		assertResponseParts(t, "fast part", res.Fast, sys.D, func(s complex128) [][]complex128 { return modal(s, 1) }, poly)
		assertSplitSum(t, "Modsep singular E", sys, res.Slow, res.Fast)
	}
}

func TestModsep_BadlyScaledPair(t *testing.T) {
	sys := badlyScaledPairSystem(t, -1e8, 1, -2e8, 0)
	res, err := Modsep(sys, 1.5e8)
	if err != nil {
		t.Fatal(err)
	}
	assertPolesMatch(t, "slow", res.Slow, []complex128{complex(-1e8, 1), complex(-1e8, -1)}, 1e-6)
	assertPolesMatch(t, "fast", res.Fast, []complex128{-2e8}, 1e-6)
	assertSplitSum(t, "modsep", sys, res.Slow, res.Fast)
}
