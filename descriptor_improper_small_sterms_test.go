package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/rand/v2"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// smallSTermChainDescriptor: U·(sW − F)·V with F = diag(f, I), W =
// diag(w, N) for 1..3 finite poles f/w, the first large in A (|f| =
// 10^(2..6)), beside chains [2]..[6], [2 3], [3 2], [2 2] (N nilpotent),
// SISO or up to 3×3, Gaussian U, V, c and D. Each chain's kernel state
// has a Gaussian b, its other states Gaussian times bscale, so the model
// is improper with s-terms of size bscale for bscale > 0. oracle is the
// structured (F, W, b, c, D) model.
func smallSTermChainDescriptor(t *testing.T, seed uint64, bscale float64) (sys, oracle *System) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 3131))
	nf := 1 + rng.IntN(3)
	f, w := make([]float64, nf), make([]float64, nf)
	for i := range nf {
		f[i], w[i] = rng.NormFloat64(), 0.5+rng.Float64()
	}
	f[0] = math.Copysign(math.Pow(10, 2+4*rng.Float64()), rng.NormFloat64())
	m, p := 1, 1
	if rng.IntN(3) == 0 {
		m, p = 1+rng.IntN(3), 1+rng.IntN(3)
	}
	chains := [][]int{{2}, {3}, {4}, {5}, {6}, {2, 3}, {3, 2}, {2, 2}}[rng.IntN(8)]
	n := nf
	for _, k := range chains {
		n += k
	}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	b, c, D := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil)
	for i := range nf {
		F.Set(i, i, f[i])
		W.Set(i, i, w[i])
		for j := range m {
			b.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			c.Set(j, i, rng.NormFloat64())
		}
	}
	o := nf
	for _, k := range chains {
		for i := range k {
			F.Set(o+i, o+i, 1)
			if i+1 < k {
				W.Set(o+i, o+i+1, 1)
			}
			for j := range p {
				c.Set(j, o+i, rng.NormFloat64())
			}
			for j := range m {
				v := rng.NormFloat64()
				switch {
				case i == 0:
					b.Set(o+i, j, v)
				case bscale != 0:
					b.Set(o+i, j, v*bscale)
				}
			}
		}
		o += k
	}
	for i := range p {
		for j := range m {
			D.Set(i, j, rng.NormFloat64())
		}
	}
	gauss := func() *mat.Dense {
		g := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				g.Set(i, j, rng.NormFloat64())
			}
		}
		return g
	}
	U, V := gauss(), gauss()
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	var err error
	if sys, err = NewDescriptor(&A, &B, &C, D, &E, 0); err != nil {
		t.Fatal(err)
	}
	if oracle, err = NewDescriptor(F, b, c, D, W, 0); err != nil {
		t.Fatal(err)
	}
	return sys, oracle
}

// TestIsProperSmallSTermsBesideLargePole: KLVET5. Improper index-4..6
// chains with s-terms of 1e-8 beside a large-A pole (seed 40: [6] beside
// 4.8e5; 215 and 260: [5] beside 5.9e5 and 1.5e6; 85) read proper when a
// proper refit to direct (sE − A) solves at |s| ≤ 1..10, three decades
// under the pencil samples' range, was taken for them: there a residue on
// the pole mimics the own s-terms, so a match refutes nothing. They read
// improper again (MATLAB isproper false) or fail closed with
// ErrDescriptorSingular (215 on amd64), as on base. Oracle: the structured
// model, improper by construction.
func TestIsProperSmallSTermsBesideLargePole(t *testing.T) {
	for _, seed := range []uint64{40, 85, 215, 260} {
		label := fmt.Sprintf("seed %d", seed)
		sys, _ := smallSTermChainDescriptor(t, seed, 1e-8)
		pr, err := sys.IsProper()
		_, serr := Step(sys, 1)
		res, terr := sys.TransferFunction(nil)
		if errors.Is(err, ErrDescriptorSingular) {
			if !errors.Is(serr, ErrDescriptorSingular) || !errors.Is(terr, ErrDescriptorSingular) {
				t.Errorf("%s: IsProper %v, Step %v, TF %v disagree", label, err, serr, terr)
			}
			continue
		}
		if err != nil || pr {
			t.Fatalf("%s: IsProper = %v, %v; want false", label, pr, err)
		}
		if !errors.Is(serr, ErrImproperModel) {
			t.Errorf("%s: Step: %v, want ErrImproperModel", label, serr)
		}
		if terr != nil {
			t.Fatalf("%s: TransferFunction: %v", label, terr)
		}
		if !fzTFImproper(res.TF) {
			t.Errorf("%s: TF proper, IsProper false", label)
		}
	}
}

// firstTailChainDescriptor: U·(sW − F)·V with F = diag(f, I), W =
// diag(w, N) for 1..3 finite poles f/w, the first large in A (|f| =
// 10^(3..6)), beside chains (N nilpotent), SISO or (mimo) up to 3×3,
// Gaussian U, V, c and D. Each chain's kernel state has a Gaussian b, the
// next state Gaussian times bscale and the rest none, so the only s-term
// is s^1, of size bscale. oracle is the structured (F, W, b, c, D) model.
func firstTailChainDescriptor(t *testing.T, seed uint64, chains []int, bscale float64, mimo bool) (sys, oracle *System) {
	t.Helper()
	rng := rand.New(rand.NewPCG(seed, 2828))
	nf := 1 + rng.IntN(3)
	f, w := make([]float64, nf), make([]float64, nf)
	for i := range nf {
		f[i], w[i] = rng.NormFloat64(), 0.5+rng.Float64()
	}
	f[0] = math.Copysign(math.Pow(10, 3+3*rng.Float64()), rng.NormFloat64())
	m, p := 1, 1
	if mimo {
		m, p = 1+rng.IntN(3), 1+rng.IntN(3)
	}
	n := nf
	for _, k := range chains {
		n += k
	}
	F, W := mat.NewDense(n, n, nil), mat.NewDense(n, n, nil)
	b, c, D := mat.NewDense(n, m, nil), mat.NewDense(p, n, nil), mat.NewDense(p, m, nil)
	for i := range nf {
		F.Set(i, i, f[i])
		W.Set(i, i, w[i])
		for j := range m {
			b.Set(i, j, rng.NormFloat64())
		}
		for j := range p {
			c.Set(j, i, rng.NormFloat64())
		}
	}
	o := nf
	for _, k := range chains {
		for i := range k {
			F.Set(o+i, o+i, 1)
			if i+1 < k {
				W.Set(o+i, o+i+1, 1)
			}
			for j := range p {
				c.Set(j, o+i, rng.NormFloat64())
			}
			for j := range m {
				v := rng.NormFloat64()
				switch i {
				case 0:
					b.Set(o+i, j, v)
				case 1:
					b.Set(o+i, j, v*bscale)
				}
			}
		}
		o += k
	}
	for i := range p {
		for j := range m {
			D.Set(i, j, rng.NormFloat64())
		}
	}
	gauss := func() *mat.Dense {
		g := mat.NewDense(n, n, nil)
		for i := range n {
			for j := range n {
				g.Set(i, j, rng.NormFloat64())
			}
		}
		return g
	}
	U, V := gauss(), gauss()
	var T, A, E, B, C mat.Dense
	T.Mul(U, F)
	A.Mul(&T, V)
	T.Mul(U, W)
	E.Mul(&T, V)
	B.Mul(U, b)
	C.Mul(c, V)
	var err error
	if sys, err = NewDescriptor(&A, &B, &C, D, &E, 0); err != nil {
		t.Fatal(err)
	}
	if oracle, err = NewDescriptor(F, b, c, D, W, 0); err != nil {
		t.Fatal(err)
	}
	return sys, oracle
}

// TestIsProperFirstSTermBesideLargePole: KLVET5. Improper
// chains whose only s-term is s^1, of 1e-6 or 1e-8 (1e5 and more above
// rounding), beside a pole of 1e3..1e6 read proper when a proper refit
// matched direct (sE − A) solves under the pole: a residue change ΔR =
// −t·λ² on it mimics t·s up to t·s²/λ, below the samples' uncertainty, so
// the match refuted nothing (e.g. [2 3] seed 77 beside −1.7e4: TF off the
// oracle by 1.4e-4 at 1e4i). Base and now: IsProper false, Step
// ErrImproperModel, an improper TF that matches the structured oracle at
// low frequency on both arches.
func TestIsProperFirstSTermBesideLargePole(t *testing.T) {
	for _, tc := range []struct {
		chains []int
		bscale float64
		mimo   bool
		seeds  []uint64
	}{
		{[]int{2, 3}, 1e-6, false, []uint64{8, 45, 60, 75, 77}},
		{[]int{3}, 1e-6, true, []uint64{8, 93, 97, 105, 106}},
		{[]int{4}, 1e-8, false, []uint64{28, 58, 106, 133}},
		{[]int{6}, 1e-8, false, []uint64{4, 20, 39, 55}},
	} {
		for _, seed := range tc.seeds {
			label := fmt.Sprintf("%v %g seed %d", tc.chains, tc.bscale, seed)
			sys, oracle := firstTailChainDescriptor(t, seed, tc.chains, tc.bscale, tc.mimo)
			if pr, err := sys.IsProper(); err != nil || pr {
				t.Errorf("%s: IsProper = %v, %v; want false", label, pr, err)
				continue
			}
			if _, err := Step(sys, 1); !errors.Is(err, ErrImproperModel) {
				t.Errorf("%s: Step: %v, want ErrImproperModel", label, err)
			}
			res, err := sys.TransferFunction(nil)
			if err != nil {
				t.Fatalf("%s: TransferFunction: %v", label, err)
			}
			if !fzTFImproper(res.TF) {
				t.Errorf("%s: TF proper, IsProper false", label)
			}
			if d := chainOracleMismatch(t, res, oracle, chainLowFreq); !(d <= 1e-7) {
				t.Errorf("%s: TF off the oracle by %.3g relative", label, d)
			}
		}
	}
}
