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

func lqiScalingPlant(t *testing.T, E *mat.Dense) *System {
	t.Helper()
	A := mat.NewDense(4, 4, []float64{-0.8, 0.6, 0.1, 0.3, -0.4, -0.3, 0.9, -0.2, 0.2, -0.7, -1.1, 0.5, 0.1, 0.4, -0.6, -0.9})
	B := mat.NewDense(4, 2, []float64{1, 0.2, 0.3, -0.5, -0.1, 0.8, 0.4, 0.6})
	C := mat.NewDense(2, 4, []float64{1, 0.4, -0.2, 0.3, 0.1, -0.6, 0.9, -0.5})
	D := mat.NewDense(2, 2, []float64{0.3, -0.1, 0.2, 0.5})
	sys, err := NewDescriptor(A, B, C, D, E, 0)
	if err != nil {
		t.Fatal(err)
	}
	return sys
}

// TestScaleLqiPlantRows: rows < n of Ea, Aa, Ba are scaled by 2^−k, k =
// trunc(log2 RMS), RMS = ‖E‖_F/√n (ZIUHCL ruling 3): RMS in (1/2, 2) and
// E = 0 leave them bit-identical to the MATLAB blkdiag(E, I) augmentation;
// the integrator rows ≥ n are never touched. E = c·I₄ has RMS c exactly.
func TestScaleLqiPlantRows(t *testing.T) {
	const n, p = 4, 2
	for _, tc := range []struct {
		rms  float64
		want float64
	}{
		{0.5 * (1 + eps()), 1},
		{1, 1},
		{1.999, 1},
		{0.5, 2},
		{2, 0.5},
		{3, 0.5},
		{1.3 * 0x1p-10, 0x1p9},
		{0, 1},
	} {
		E := mat.NewDense(n, n, nil)
		for i := range n {
			E.Set(i, i, tc.rms)
		}
		sys := lqiScalingPlant(t, nil)
		Aa, Ba := lqiAugmentation(sys)
		Ea := mat.NewDense(n+p, n+p, nil)
		setBlock(Ea, 0, 0, E)
		for i := n; i < n+p; i++ {
			Ea.Set(i, i, 1)
		}
		Ea0, Aa0, Ba0 := mat.DenseCopyOf(Ea), mat.DenseCopyOf(Aa), mat.DenseCopyOf(Ba)
		scaleLqiPlantRows(E, n, Ea, Aa, Ba)
		for _, c := range []struct {
			name      string
			got, orig *mat.Dense
		}{{"Ea", Ea, Ea0}, {"Aa", Aa, Aa0}, {"Ba", Ba, Ba0}} {
			r, cols := c.orig.Dims()
			for i := range r {
				s := 1.0
				if i < n {
					s = tc.want
				}
				for j := range cols {
					if got, want := c.got.At(i, j), s*c.orig.At(i, j); got != want {
						t.Errorf("RMS %g: %s[%d,%d] = %g, want %g", tc.rms, c.name, i, j, got, want)
					}
				}
			}
		}
	}
}

// TestLqiEquationScaleInvariance: (2ʲE, 2ʲA, 2ʲB, C, D) is the same
// descriptor plant, so Lqi returns the same error class, K, explicit X and
// closed-loop eigenvalues to rounding (the plant-row scale is not
// equivariant in j, so not bit-identical). Plants have cond(E) ≤ 1e4: at
// cond(E) ~ 1/ε the outcome under exact equation scaling is rounding noise
// until the Dare boundary verdict is fixed (ZIUHCL rulings 2, 3).
func TestLqiEquationScaleInvariance(t *testing.T) {
	rng := rand.New(rand.NewPCG(41, 7))
	U, V := randomOrthogonal(rng, 3), randomOrthogonal(rng, 3)
	graded := mulDense(mulDense(U, mat.NewDense(3, 3, []float64{1, 0, 0, 0, 1e-2, 0, 0, 0, 1e-4})), V)
	for _, dt := range []float64{0, 0.1} {
		plant := obsTestPlant(t, dt, true)
		for _, E := range []*mat.Dense{plant.E, graded} {
			cond := mat.Cond(E, 2)
			ktol := math.Max(1e-10, 100*eps()*cond)
			base := lqiScaledRun(t, plant, E, 0)
			for _, j := range []int{-30, -9, 0, 9, 30} {
				label := fmt.Sprintf("dt=%v cond=%.1e j=%d", dt, cond, j)
				got := lqiScaledRun(t, plant, E, j)
				if lqiErrClass(got.err) != lqiErrClass(base.err) {
					t.Errorf("%s: err = %v, base %v", label, got.err, base.err)
					continue
				}
				if base.err != nil {
					continue
				}
				if e := relErr(got.res.K, base.res.K); e > ktol {
					t.Errorf("%s: K off by %.3g > %.3g", label, e, ktol)
				}
				if e := relErr(got.res.X, base.res.X); e > ktol {
					t.Errorf("%s: X off by %.3g > %.3g", label, e, ktol)
				}
				if e := eigSetRelErr(got.res.Eig, base.res.Eig); e > 1e-8 {
					t.Errorf("%s: Eig off by %.3g", label, e)
				}
			}
		}
	}
}

type lqiScaledResult struct {
	res *RiccatiResult
	err error
}

func lqiScaledRun(t *testing.T, plant *System, E *mat.Dense, j int) lqiScaledResult {
	t.Helper()
	pow := func(M *mat.Dense) *mat.Dense {
		o := mat.DenseCopyOf(M)
		o.Apply(func(_, _ int, v float64) float64 { return math.Ldexp(v, j) }, o)
		return o
	}
	sys, err := NewDescriptor(pow(plant.A), pow(plant.B), plant.C, plant.D, pow(E), plant.Dt)
	if err != nil {
		t.Fatal(err)
	}
	res, err := acceptUnverified(Lqi(sys, eyeDense(5), eyeDense(2), nil))
	return lqiScaledResult{res, err}
}

// eigSetRelErr is the largest |a−b|/|b| over a greedy nearest matching of
// got to want.
func eigSetRelErr(got, want []complex128) float64 {
	if len(got) != len(want) {
		return math.Inf(1)
	}
	used := make([]bool, len(want))
	worst := 0.0
	for _, a := range got {
		best, bi := math.Inf(1), -1
		for i, b := range want {
			if !used[i] {
				if d := cmplx.Abs(a-b) / cmplx.Abs(b); d < best {
					best, bi = d, i
				}
			}
		}
		used[bi] = true
		worst = math.Max(worst, best)
	}
	return worst
}

func lqiErrClass(err error) error {
	for _, e := range []error{ErrDescriptorSingular, ErrNoStabilizing, ErrNotConverged, ErrSchurFailed, ErrSingularEquation} {
		if errors.Is(err, e) {
			return e
		}
	}
	return err
}

// TestLqiUnitScaleDescriptorBitIdentical: for E with RMS in (1/2, 2) Lqi is
// bit-identical (K, X, Eig, error) to Lqr/Dlqr on the MATLAB augmentation
// blkdiag(E, I), [A 0; -hC 0 or I], [B; -hD] (ZIUHCL ruling 3).
func TestLqiUnitScaleDescriptorBitIdentical(t *testing.T) {
	for _, sys := range []*System{obsTestPlant(t, 0, true), obsTestPlant(t, 0.1, true), nearSingularTriangularE(857, 1), nearSingularTriangularE(962, 1)} {
		n, m, p := sys.Dims()
		Q, R := eyeDense(n+p), eyeDense(m)
		got, gerr := Lqi(sys, Q, R, nil)
		Aa, Ba := lqiAugmentation(sys)
		o := &RiccatiOpts{E: augmentedDescriptor(sys.E, p), plantE: sys.E, augmentedE: true}
		solve := Lqr
		if sys.IsDiscrete() {
			solve = Dlqr
		}
		want, werr := solve(Aa, Ba, Q, R, o)
		if fmt.Sprint(gerr) != fmt.Sprint(werr) {
			t.Errorf("dt=%v n=%d: err %v, want %v", sys.Dt, n, gerr, werr)
			continue
		}
		if want == nil {
			continue
		}
		if !mat.Equal(got.K, want.K) || !mat.Equal(got.X, want.X) || fmt.Sprint(got.Eig) != fmt.Sprint(want.Eig) {
			t.Errorf("dt=%v n=%d: Lqi differs from the blkdiag(E, I) augmentation", sys.Dt, n)
		}
	}
}
