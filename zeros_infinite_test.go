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

	h, bound := exactMarkov(a, b, c, n)
	for k := range n {
		if noise := math.Abs(h[k]) <= 2*eps()*bound[k]; noise != (k < n-1) {
			t.Fatalf("oracle: h%d = %g, bound %g", k+1, h[k], bound[k])
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

// exactMarkov returns h_k = C·A^(k−1)·B, k = 1..n, of the stored float64
// entries, computed exactly and then rounded, and |C|·|A|^(k−1)·|B|.
func exactMarkov(a, b, c []float64, n int) (h, bound []float64) {
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
		hk, bk := new(big.Rat), 0.0
		for i := range n {
			hk.Add(hk, new(big.Rat).Mul(new(big.Rat).SetFloat64(c[i]), v[i]))
			bk += math.Abs(c[i]) * vAbs[i]
		}
		hf, _ := hk.Float64()
		h, bound = append(h, hf), append(bound, bk)
	}
	return h, bound
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

// TestZerosSimilarCompanionRelativeDegree: companion realizations of
// k/den(s) ("none") and 5.8e-4·(s+2)(s+3)/den(s) ("s+2,s+3") under a dense
// similarity, so the vanishing Markov parameters are rounding noise (exactly,
// below 2ε of |C|·|A|^(k−1)·|B|) and the relative degree is 5 (3). The
// balanced staircase kept a noise block at margins 1.5 to 22, and the
// arbitration, when it ran, picked it or an unscaled reduction keeping one
// too, returning zeros far out (±6.57e5, triples at 4e4; 3.7e13 and −9.7e11
// beside −2, −3), which MATLAB tzero returns as infinite. MYNTN2.
func TestZerosSimilarCompanionRelativeDegree(t *testing.T) {
	capped := 0
	for _, tc := range similarCompanionModels() {
		const n = 5
		h, bound := exactMarkov(tc.a, tc.b, tc.c, n)
		for k := range n {
			if noise := math.Abs(h[k]) <= 2*eps()*bound[k]; noise != (k < n-1-len(tc.want)) {
				t.Fatalf("%s: oracle: h%d = %g, bound %g", tc.name, k+1, h[k], bound[k])
			}
		}
		for _, dt := range []float64{0, 0.1} {
			sys, err := NewFromSlices(n, 1, 1, tc.a, tc.b, tc.c, []float64{0}, dt)
			if err != nil {
				t.Fatal(err)
			}
			if st, _, err := prescaledStaircaseZeros(sys, zerosPrescale(sys)); err == nil && len(st) > len(tc.want) {
				capped++
			}
			res, err := sys.ZerosDetail()
			if err != nil {
				t.Fatal(err)
			}
			ok := len(res.Zeros) == len(tc.want) && res.Rank == 1
			for i := 0; ok && i < len(tc.want); i++ {
				ok = cmplx.Abs(res.Zeros[i]-tc.want[i]) <= 1e-6
			}
			if !ok {
				t.Errorf("%s dt=%g: zeros %v rank %d, want %v, rank 1", tc.name, dt, res.Zeros, res.Rank, tc.want)
			}
		}
	}
	if capped < 4 {
		t.Errorf("precondition: staircase kept noise zeros in %d of 10 cases, want ≥ 4", capped)
	}
}

type siso5Model struct {
	name    string
	a, b, c []float64
	want    []complex128
}

func similarCompanionModels() []siso5Model {
	return []siso5Model{
		{"none/2", []float64{
			2.444937294663357, 0.10513697286922953, 9.342873135806606, 4.441551374594324, 18.959821362312205,
			3.7073181732086193, -1.6344828135615028, 14.494888957755071, 6.639600320851898, 27.845215287137208,
			-2.7120883051379634, 3.1066403713044215, -11.88350838504976, -2.849793267082719, -26.26997184413238,
			4.001766105823747, -4.243626043580455, 16.6537374903842, 5.4476498573617915, 37.45163452814799,
			-9.009479793308829, 6.293223648148065, -35.61334908886938, -14.705388943881916, -75.61499139993215,
		}, []float64{-0.18450588794433898, -0.2780576379076768, 0.26847218133783696, -0.3727441804949622, 0.753752414815281},
			[]float64{-0.0008367074928394088, -9.550308436440604e-05, -0.0004925365350772862, -0.0006295762953850253, -0.000375947536245429},
			nil},
		{"none/11", []float64{
			-3.6010216759602174, -0.46512870837258147, 2.236497396208176, -2.4683005752946268, -1.3320590976227453,
			0.9079065707857181, -0.8064266897835732, 0.7124703524803059, 0.11372910652152624, 0.5889210456887256,
			-16.07081918413467, 3.1265972603201337, 12.951306306175812, -11.02416925826211, -3.9732029318673012,
			-15.17887249006612, 4.777787035905897, 12.099343472556473, -10.311537157193824, -3.277081188862873,
			-36.85271007226071, 3.1761629641702758, 29.670054058916094, -26.88463579862337, -9.257265515662914,
		}, []float64{0.03205381751478459, -0.031041354866690317, 0.2907356668029031, 0.32478132659586667, 0.5851337409813273},
			[]float64{0.012032755997848114, 0.006100507906573344, -0.00608097301619804, 0.0072055345075150165, -0.001313537387844313},
			nil},
		{"none/14", []float64{
			15.460970574216162, 16.291145717511764, -46.685209055739925, 2.594994933331953, 30.522987872368244,
			-7.742194407419967, -6.6551333118005775, 24.011201608392085, -0.5388297210952228, -16.070602587415618,
			-25.797729652219875, -23.678382903522774, 79.26698647244923, -2.0551292229724645, -52.76680205489684,
			7.777513040505962, 6.050311896001683, -23.789565341498744, 0.7744449908505388, 16.875772391225198,
			-44.59630215631924, -41.760266215165764, 136.29092171323356, -5.1503306028349565, -90.00846744305078,
		}, []float64{-0.26634080875693966, 0.12755678795496206, 0.44945677731769595, -0.12782497550518856, 0.7702280959616997},
			[]float64{-0.7030457700387364, 0.04572871831400207, 0.18352372862522875, -0.0254727266201958, -0.36200292245643845},
			nil},
		{"s+2,s+3/0", []float64{
			-1.0130237394008652, 2.3287914543598327, -1.5835460797730496, -1.0958531615427625, 1.2564304598093636,
			5.614078974924694, 1.7123995053793803, 6.561926338487748, 5.606258763128271, -6.377786886947062,
			2.600969332499258, 1.9504229132358917, 1.5794755475449676, 3.1599159530480097, -1.9077478891815942,
			-0.5389179648420596, -1.636388957369126, -0.525285541365883, -0.46399528901619136, 1.2018272444872915,
			25.881268850035568, 10.003678053851939, 26.675699414400068, 26.088798393314185, -26.86277671729723,
		}, []float64{-0.09117907860104175, 0.281932554509531, 0.07524613507732114, 0.0017994324710368268, 1.2211087083313872},
			[]float64{0.0065762466212722645, 0.012137062100383164, 0.003776120044725033, 0.004379284748623399, -0.002550334347387211},
			[]complex128{-2, -3}},
		{"s+2,s+3/182", []float64{
			-3.751590125568706, -0.23525455045190324, 1.270280920151462, 4.154738662579246, -3.326668767254115,
			4.923664987623986, 1.1899751094750468, -0.5934880511393412, -5.427260916572899, 5.260425489860091,
			4.306918135226285, -0.0281095702942773, -2.891917020765666, -2.48619756686081, 4.041708017354085,
			27.756391650013267, 8.70365434198883, -7.764774203936903, -32.17467796361963, 34.13733579345778,
			30.940932159501493, 8.832959704291078, -9.321977920468493, -35.40575293563956, 36.98470835536557,
		}, []float64{-0.11844914219805149, 0.08850987556274156, 0.41014895225342224, 0.5482328974640411, 0.7002416175514553},
			[]float64{0.0784628111668368, 0.028914817667556394, -0.01347153637344174, -0.09389612152148384, 0.09102127530997685},
			[]complex128{-2, -3}},
	}
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
