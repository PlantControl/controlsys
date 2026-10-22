package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
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

// TestZerosLargeOutputGainKeepsZeros: companion realizations of known
// zeros under an orthogonal similarity, where ‖C‖ (1.6e13 for a gain of
// 1e9; 4e7 from the companion coefficients alone) dominates ‖P‖ of the
// balanced model. Its staircase decision lands within rankMarginMin, so
// arbitration runs and must keep the true zeros, as MATLAB tzero, which is
// invariant to output scaling. Testing distinguishability at modulus
// max(|z|, ‖P‖) counted none of them and returned [] (gain 1e9) or
// {−45.08, −52.16±42.42i} (unscaled). JHFQWK.
func TestZerosLargeOutputGainKeepsZeros(t *testing.T) {
	for _, tc := range []struct {
		name    string
		n       int
		a, b, c []float64
		want    []complex128
		tol     float64
	}{
		{"gain1e9", 5, []float64{
			-176.43879948621068, -52.592456431717196, 68.56051650089319, -21.520142157024132, 33.219508092055285,
			696.0943390064521, 208.893899591836, -274.36083450914475, 85.04031968346678, -130.52292655268533,
			362.73004494873385, 108.31806554354489, -143.2003292877212, 44.40517611962239, -68.98131507567331,
			357.26789902248186, 108.01629993655823, -140.59622290456792, 43.81031858635301, -67.77994164275947,
			-344.43554026971776, -103.42774050820113, 135.65613212742383, -41.05052713123383, 65.00091059574271,
		}, []float64{0.1865872322789961, -0.7363496990107561, -0.38379963920809024, -0.37814491658418137, 0.3642507165710385},
			[]float64{-1.6375237119942805e+13, -1.9369463261305366e+12, 5.871082845628281e+11, -2.6946142223121226e+12, 2.293798110296307e+12},
			[]complex128{complex(-56.8195, 116.8170), complex(-56.8195, -116.8170)}, 1e-6},
		{"unscaled", 7, []float64{
			5.447249118388404, 18.1199731688176, -7.527868178710931, -35.61291038469344, -9.392880314721097, 19.761683789690952, -23.507823186995516,
			8.465759911249139, 28.030483339420396, -11.615895739833936, -56.207496819752606, -14.364661457581365, 32.673963060591305, -36.763684837075715,
			-12.314298900992148, -41.106165764369855, 15.890845903643251, 81.77939447611128, 20.38863679065462, -46.96318104873932, 54.31101426646289,
			-0.637739075984207, 0.060836990958524975, -0.21285822188433245, 1.1836154503145508, 0.45577034061537064, -0.3267514678813866, 0.16340318000623028,
			-1.313858564247498, -3.5228547481976467, 1.0336433056282497, 6.588996026577644, 1.9926741626366253, -3.710322221414347, 4.840732603125107,
			-5.004906914815527, -15.44694469103281, 6.027654020099627, 29.845594373960058, 6.9121089481001405, -16.970977512606307, 19.301559504329006,
			10.247363076727005, 33.7577440704105, -13.952764221969977, -68.68486391151626, -16.6233451019077, 38.70302012156812, -45.382890461796926,
		}, []float64{0.2737231092384947, 0.4365245747692002, -0.6307852121371015, -0.005044876434365073, -0.05639552104019392, -0.23077185072352172, 0.5293112681424227},
			[]float64{-4.133238156755118e+07, 1.0008380977551136e+07, 2.621813703869763e+07, -1.4362905359287597e+07, 590815.4189168043, -2.104105634829873e+07, 3.511716969395557e+07},
			[]complex128{-45.0797, -89.1304, complex(-119.754, 45.920), complex(-119.754, -45.920)}, 1e-3},
	} {
		sys, err := NewFromSlices(tc.n, 1, 1, tc.a, tc.b, tc.c, []float64{0}, 0)
		if err != nil {
			t.Fatal(err)
		}
		if m := reduceStaircase(zerosPrescale(sys)).margin; !(m < rankMarginMin) {
			t.Fatalf("%s: precondition: balanced staircase margin %g, want < %d (arbitration)", tc.name, m, rankMarginMin)
		}
		got, err := sys.Zeros()
		if err != nil {
			t.Fatal(err)
		}
		for _, w := range tc.want {
			best := math.Inf(1)
			for _, z := range got {
				best = math.Min(best, cmplx.Abs(z-w))
			}
			if !(best <= tc.tol*cmplx.Abs(w)) {
				t.Errorf("%s: zero %v missing from %v", tc.name, w, got)
			}
		}
	}
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
