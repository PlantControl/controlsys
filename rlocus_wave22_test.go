package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"strconv"
	"testing"
)

// TestRootLocus_NoiseMarkovKeepsSign (wave 22 B2, reviewer case 11295
// class): k(s+12)(s+48)/((s+3.5)(s+12)(s+48)(s+72)((s+8)²+56²)) under a
// scaling/Givens/triangular similarity with entries ~1e13. Every exact hₖ
// is within 10·eps·sensₖ (h₄…h₆ at ρ 4…8), the rounding-level h₁…h₃ at
// ρ < 0.1; Zeros() returns two noise zeros. With one resolution threshold
// the relative degree is 4, the noise zeros are dropped and the gain sign
// is that of h₄; a noise h₂ had set relative degree 2 and the 0° rule,
// flipping every departure angle by π. Departure angles are checked at the
// poles away from the cancelling pairs, against the intended loop, to the
// 1e-2 the computed poles allow (they are 3e-3 off).
func TestRootLocus_NoiseMarkovKeepsSign(t *testing.T) {
	poles := []complex128{complex(-8, 56), complex(-8, -56), -72, -48, -12, -3.5}
	zeros := []complex128{-12, -48}
	for i, c := range []struct{ a, b, c []float64 }{
		{
			[]float64{-74963132.37725948, -9511.384684225211, 948165.0132592607, 1565098.0212562585, 460169014.6742613, -487.6001984110312, 37817023158.67332, 4798255.8547275225, -478325512.87315845, -789552716.5538807, -232143751619.16232, 245982.11324746237, -2260786023.649144, -286850.36756792275, 28595365.491656695, 47201226.099917084, 13878071165.65308, -14705.358237394117, -149613200.7230736, -18983.049116422902, 1892370.0755045891, 3123660.216335954, 918416291.7625471, -973.164012832899, -15626221.359052839, -1982.6680103696904, 197646.98175599411, 326247.9550113115, 95923192.05832767, -101.64128199636716, -8836528096590.037, -1121185844.503799, 111768098178.31207, 184491115049.68314, 54243951188997.086, -57477492.74378835},
			[]float64{-0.005564648973724355, 2.807226133546673, -0.16782223535867788, -0.011106058891801872, -0.0011599626905148934, -655.9514650395555},
			[]float64{-17520.069511167392, -0.27699507960125047, 54.46596491030627, 63.20878084119821, 90587.22980221029, -0.02775351409986515},
		},
		{
			[]float64{-72504523.04631251, -20283080.464146562, 61390825.59079451, 16313685.979787026, -0.6927383234371569, -16509649.086809227, 138700311.13803843, 38801301.26168309, -117439936.60339868, -31207891.73835765, 1.3252002143935115, 31582768.170062594, 67552301.43556345, 18897708.328706913, -57197693.131861374, -15199423.15534257, 0.6454226597029359, 15382003.164266203, -758622993.1246791, -212224183.48302314, 642339123.9318135, 170691932.0563096, -7.24800768021009, -172742324.45608312, 14973083.279266283, 4190336.6020686924, -12675319.674412185, -3366899.5722265346, 0.1430463646188881, 3409305.4792098426, -350414488.310972, -98028169.33862987, 296701974.82297105, 78844075.30230603, -3.3478143365102535, -79791168.78286518},
			[]float64{0.0010333238219211232, -0.001976736242270303, -0.0009627453641796911, 0.010811782136865036, -0.00021337525469430907, 0.004994054784714517},
			[]float64{106951.56459352338, 108451.20594420865, -65178.31847534501, 7066.97020602911, 0, -7066.97020602911},
		},
	} {
		name := "case " + strconv.Itoa(i)
		sys := rlpSys(t, 6, c.a, c.b, c.c, 0, 0)
		res, err := RootLocus(sys, []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		rlpAngles(t, name+" AsymptoteAngles", res.AsymptoteAngles, []float64{math.Pi / 4, 3 * math.Pi / 4, 5 * math.Pi / 4, 7 * math.Pi / 4})
		got, err := sys.Poles()
		if err != nil {
			t.Fatal(err)
		}
		for k, p := range got {
			j, _ := rvNearest(p, poles)
			if q := poles[j]; real(q) == -12 || real(q) == -48 {
				continue
			}
			want := math.Pi
			for l, q := range poles {
				if l != j {
					want -= cmplx.Phase(poles[j] - q)
				}
			}
			for _, z := range zeros {
				want += cmplx.Phase(poles[j] - z)
			}
			if d := math.Abs(math.Remainder(res.DepartureAngles[k]-want, 2*math.Pi)); d > 1e-2 {
				t.Errorf("%s: departure at %v = %g, want %g", name, poles[j], res.DepartureAngles[k], want)
			}
		}
	}
}

// TestRootLocus_SplitDoubleZeroNoFeedthrough (wave 22, reviewer case 10677
// class): 3(s+3)(s+2.5)²/(s(s+36)(s²+0.5625)), D = 0, under a scaling/
// Givens/triangular similarity whose rounding splits the double zero into
// two real zeros ~9e-5 apart (2.7e-5 in Zeros). The zeros, as eigenvalues
// of the zero-dynamics matrix A − B·C·A/h₁, are u·κ-linked with a pair mean
// far better conditioned than their split, so −2.5 (K = ∞) is a break point
// as in the intended loop; the system-pencil κ cap had left them apart.
func TestRootLocus_SplitDoubleZeroNoFeedthrough(t *testing.T) {
	oi := newRlswOracle(rlpRats(0, 3, 24, 63.75, 56.25), rlpRats(1, 36, 0.5625, 20.25, 0))
	for i, m := range []struct{ a, b, c []float64 }{
		{
			[]float64{163.81420983322218, 74.64311883533684, -18.657679132180508, 104.08724455861936, -472.64224358957824, -215.64402982158288, 64.8635449321895, -254.73496123334695, -141.16136208281264, -64.32148825943844, 16.077280882322196, -89.69348206133829, 60.12041991450314, 27.632977608583705, -16.15089553712487, -0.24746089396148463},
			[]float64{-0.16756204546463738, 0.009101100712632315, 0.14439063482537975, 0.341252489940445},
			[]float64{5160.933543272735, 208.4169900883478, 5390.671604349607, 256.45896951229673},
		},
		{
			[]float64{0.15390810036605915, -0.31384178289046555, -0.512144401732552, -0.3095796931570512, 16.57185812997175, 6.543536321209192, -22.181593872621626, 4.336700561913234, -8.454250686839965, 69.9742851310155, 70.01580658577112, 65.34358255525224, -2.9969293057060007, -122.63553821294721, -93.25032335157478, -112.71325100734637},
			[]float64{0, 0, -126.50252275153427, 209.27586745612555},
			[]float64{-5.825280442286043, 3.3362212538934637, 5.326103880181568, 3.2338443295263697},
		},
	} {
		res, err := RootLocus(rlpSys(t, 4, m.a, m.b, m.c, 0, 0), []float64{0, 1})
		if err != nil {
			t.Fatal(err)
		}
		if !oi.accepts(res.Breakaway) {
			t.Errorf("case %d: Breakaway = %v, want %v", i, res.Breakaway, oi.required)
		}
	}
}

func rlpRats(x ...float64) []*big.Rat {
	out := make([]*big.Rat, len(x))
	for i, v := range x {
		out[i] = fzRat(v)
	}
	return out
}

// TestRootLocus_ResolvedNonNormalPairKSign (wave 22 B1 class, reviewer
// cases 30987, 31290 and a minimal dyadic repro): a complex pole pair of
// relative split ~1e-4 realized by a non-normal A (σ_b/|λ| ~ 2e4) is
// u·κ-linked by eig and split by at most √u·σ_b, so only the converted D
// can tell it from a double; D needs 6e5·u…4.5e6·u to merge it, which the
// old √u tolerance admitted, listing the K < 0 point at the pair's real
// part. The exact oracle of the floats rejects it (robust to 1-ulp
// changes for 31290; rlswUlpSensitive flags the other two, pinned as
// given). Grid: G = 32/((s+3584)² + σ²) under exact I + v·e₀e₁ᵀ
// similarities of the companion [0 1; −q −7168] (the minimal repro is
// σ = 0.4375, v = −2), asserted only where the oracle is posed and robust
// to 1-ulp changes; every point is 1-ulp sensitive at entries ~1e7…1e10
// (eig splits most of them into two real roots), so they are logged as
// precision limits.
func TestRootLocus_ResolvedNonNormalPairKSign(t *testing.T) {
	type sysData struct {
		name    string
		n       int
		a, b, c []float64
	}
	cs := []sysData{
		{"minimal", 2, []float64{25690112.3828125, 51394561.765625, -12845056.19140625, -25697280.3828125}, []float64{-2, 1}, []float64{32, 64}},
		{"30987", 2, []float64{-2.56829453828125e7, -1.284147244140625e7, 5.1351556765625e7, 2.56757773828125e7}, []float64{32, -64}, []float64{2, 1}},
		{"31290", 3, []float64{45942.01278480432, -7700.285188740199, -46254.37380409317, 32263.2672003399, -5410.057865434646, -32487.91250383978, 40346.07038640868, -6761.950570224694, -40619.45491936967}, []float64{0.04153337941018198, 0.029169793742546878, 0.036473984259841205}, []float64{2898791.329147607, -505488.26534131414, -2896245.4694443424}},
	}
	const p = 3584.0
	for _, sg := range []float64{0.4375, 0.25, 0.125, 1.0 / 16, 1.0 / 64} {
		for _, v := range []float64{-2, -8, -64, -1024} {
			q := p*p + sg*sg
			cs = append(cs, sysData{
				"grid σ=" + strconv.FormatFloat(sg, 'g', -1, 64) + " v=" + strconv.FormatFloat(v, 'g', -1, 64), 2,
				[]float64{-v * q, v*v*q + 1 - 2*p*v, -q, v*q - 2*p},
				[]float64{v, 1},
				[]float64{32, -32 * v},
			})
		}
	}
	for i, c := range cs {
		res, err := RootLocus(rlpSys(t, c.n, c.a, c.b, c.c, 0, 0), nil)
		if err != nil {
			t.Fatal(err)
		}
		rc := rlswCase{n: c.n, a: c.a, b: c.b, c: c.c, num: rlpRats(1), den: rlpRats(1)}
		_, o := rc.oracles()
		if !o.posed || i >= 3 && rlswUlpSensitive(&rc, i) {
			t.Logf("%s: oracle unposed or 1-ulp sensitive (precision limit): Breakaway %v, oracle %v", c.name, res.Breakaway, o.required)
			continue
		}
		if !o.accepts(res.Breakaway) {
			t.Errorf("%s: Breakaway = %v, want %v", c.name, res.Breakaway, o.required)
		}
	}
}
