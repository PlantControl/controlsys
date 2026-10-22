package controlsys

import (
	"math"
	"math/big"
	"testing"
)

// TestZerosVanishingMarkovInfinite: h₁..h₄ = C·A^(k−1)·B of this model are
// rounding noise (exactly, below 2ε of |C|·|A|^(k−1)·|B|) and h₅ ≈ 2.4375,
// so its relative degree is 5 = n and it has no finite zeros (MATLAB tzero
// returns none). The balanced staircase kept one noise block at 1.4× its
// threshold, and the backward test, which every point that far out passes,
// preferred its zeros near ±1.07e7 (±7.04e6 with a 3-sample output delay,
// four near 3.6e3 with an input delay) over the unscaled reduction's none.
func TestZerosVanishingMarkovInfinite(t *testing.T) {
	a, b, c := vanishingMarkovModel()
	const n = 5

	v := make([]*big.Rat, n)
	vAbs := make([]float64, n)
	for i := range n {
		v[i] = new(big.Rat).SetFloat64(b[i])
		vAbs[i] = math.Abs(b[i])
	}
	for k := 1; k <= n; k++ {
		if k > 1 {
			w := make([]*big.Rat, n)
			wAbs := make([]float64, n)
			for i := range n {
				w[i] = new(big.Rat)
				for j := range n {
					w[i].Add(w[i], new(big.Rat).Mul(new(big.Rat).SetFloat64(a[i*n+j]), v[j]))
					wAbs[i] += math.Abs(a[i*n+j]) * vAbs[j]
				}
			}
			v, vAbs = w, wAbs
		}
		h, bound := new(big.Rat), 0.0
		for i := range n {
			h.Add(h, new(big.Rat).Mul(new(big.Rat).SetFloat64(c[i]), v[i]))
			bound += math.Abs(c[i]) * vAbs[i]
		}
		hf, _ := h.Float64()
		if noise := math.Abs(hf) <= 2*eps()*bound; noise != (k < n) {
			t.Fatalf("oracle: h%d = %g, bound %g", k, hf, bound)
		}
	}

	for _, delay := range []string{"none", "output", "input"} {
		sys, err := NewFromSlices(n, 1, 1, a, b, c, []float64{0}, 0.1)
		if err != nil {
			t.Fatal(err)
		}
		switch delay {
		case "output":
			err = sys.SetOutputDelay([]float64{3})
		case "input":
			err = sys.SetInputDelay([]float64{3})
		}
		if err != nil {
			t.Fatal(err)
		}
		for _, absorb := range []bool{false, true} {
			s := sys
			if absorb {
				if s, err = sys.AbsorbDelay(); err != nil {
					t.Fatal(err)
				}
			}
			res, err := s.ZerosDetail()
			if err != nil {
				t.Fatal(err)
			}
			if len(res.Zeros) != 0 || res.Rank != 1 {
				t.Errorf("delay=%s absorb=%v: zeros %v rank %d, want none, rank 1", delay, absorb, res.Zeros, res.Rank)
			}
		}
	}
}

func vanishingMarkovModel() (a, b, c []float64) {
	a = []float64{
		0.019797815752731235, 0.1279571753021788, 0.3811493659662884, 0.139536029114894, -0.24264502599501941,
		-0.024690736661808866, 0.3390417987388423, 0.07713472609337782, 0.02058320045505041, 0.16867929301626863,
		0.40147694836136044, 0.11464617555258745, 0.49445973150623634, -0.12279478026090808, 0.5966980275659939,
		0.1751612048549504, 0.04429700567025503, -0.11719819109801555, 0.3690601749614305, 0.18574256570034794,
		-0.21799075604346427, 0.06290279396295813, 0.6168424099007837, 0.2142588609269953, 0.2276404790407601,
	}
	return a,
		[]float64{-0.26141391328956, -0.5418493331123063, -1.6696096552907065, 0.5915593669800094, 0.7074766289020779},
		[]float64{-142.41775793762778, 47.204252982690086, -16.13755010077696, -144.21002232543742, 66.02760815553987}
}

func BenchmarkZerosVanishingMarkov(b *testing.B) {
	a, bm, c := vanishingMarkovModel()
	sys, err := NewFromSlices(5, 1, 1, a, bm, c, []float64{0}, 0.1)
	if err != nil {
		b.Fatal(err)
	}
	for b.Loop() {
		if _, err := sys.Zeros(); err != nil {
			b.Fatal(err)
		}
	}
}
