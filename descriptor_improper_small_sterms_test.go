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
// 4.8e5; 215 and 260: [5] beside 5.9e5 and 1.5e6; 85) read proper on the
// pencil-sample refit: the split's s-terms (4.4e10, 9.2e4, 0.19, 3.9e-7,
// 9.4e-9 for seed 40) are a residue leaked from the pole, decaying by
// 1/λ, plus the model's own 1e-8 terms that surface at the tail. The
// direct (sE − A) solves are kept only to |s| ≤ 1..10, where the leak
// shows and the own terms do not, so a proper refit matching them refuted
// the leak alone. The own terms, what the leak's recurrence leaves of the
// split's terms, must now show at some sample before the samples may
// refute them; here none does, and the model reads improper (MATLAB
// isproper false) or fails closed with ErrDescriptorSingular (215 on
// amd64, as before). Oracle: the structured model, improper by
// construction.
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

// TestIsProperLeakResidueBesideLargePole: KLVET5. Proper index-3..6
// chains beside a large finite pole whose split's s-terms are only the
// residue leaked from the pole, each term the one before over λ
// (refutedChainDescriptor seed 46: 1.4e5, 0.91, 6.0e-6, 3.9e-11, 6.4e-16
// beside 1.53e5), read improper on base. The leak's recurrence leaves
// nothing above rounding, so the direct (sE − A) solves, which the terms
// reach, refute them and a proper refit stands: seed 46 on the confirm
// path, 60 on the improper split's refit, smallSTermChainDescriptor seed
// 31 (bscale 0) on the leak refit. Oracle: the structured model's pencil
// response.
func TestIsProperLeakResidueBesideLargePole(t *testing.T) {
	cases := []struct {
		label       string
		sys, oracle *System
	}{}
	for _, seed := range []uint64{46, 60} {
		sys, oracle := refutedChainDescriptor(t, seed)
		cases = append(cases, struct {
			label       string
			sys, oracle *System
		}{fmt.Sprintf("refuted seed %d", seed), sys, oracle})
	}
	sys, oracle := smallSTermChainDescriptor(t, 31, 0)
	cases = append(cases, struct {
		label       string
		sys, oracle *System
	}{"small s-term seed 31", sys, oracle})
	for _, c := range cases {
		pr, err := c.sys.IsProper()
		if err != nil || !pr {
			t.Fatalf("%s: IsProper = %v, %v; want true", c.label, pr, err)
		}
		res, err := c.sys.TransferFunction(nil)
		if err != nil {
			t.Fatalf("%s: TransferFunction: %v", c.label, err)
		}
		if fzTFImproper(res.TF) {
			t.Fatalf("%s: TF improper", c.label)
		}
		if d := chainOracleMismatch(t, res, c.oracle, chainLowFreq); !(d <= 1e-6) {
			t.Errorf("%s: TF off the oracle by %.3g relative", c.label, d)
		}
	}
}
