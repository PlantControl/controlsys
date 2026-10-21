package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"strconv"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

// zerosCheckExact matches got one-to-one with want (exact roots of the gcd
// of the maximal pencil minors, big.Rat plus 400-bit Newton): zeros below
// 1e6 in modulus to relative 1e-10, larger ones (near-infinite, condition
// up to 1e-6) to 1e-4.
func zerosCheckExact(t *testing.T, label string, got, want []complex128) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: %d zeros %v, want %d %v", label, len(got), got, len(want), want)
	}
	used := make([]bool, len(got))
	for _, w := range want {
		best := -1
		for i, z := range got {
			if !used[i] && (best < 0 || cmplx.Abs(z-w) < cmplx.Abs(got[best]-w)) {
				best = i
			}
		}
		used[best] = true
		tol := 1e-10
		if cmplx.Abs(w) >= 1e6 {
			tol = 1e-4
		}
		if e := cmplx.Abs(got[best]-w) / cmplx.Abs(w); !(e <= tol) {
			t.Errorf("%s: zero %v, exact %v: relative error %.2g > %.2g\nzeros=%v", label, got[best], w, e, tol, got)
		}
	}
}

// TestZerosIOScaledDecouplingZero: a strongly diagonally scaled SISO model
// with ‖B‖ ≈ 2e-5 and ‖C‖ ≈ 5e6 after state balancing keeps blocks below
// the staircase rank tolerance unless inputs and outputs are scaled too;
// the zeros (input-decoupling zero A[0][0] and a complex pair) must come
// out exact for every exactly zero-preserving B·2^e, C·2^-e.
func TestZerosIOScaledDecouplingZero(t *testing.T) {
	want := []complex128{1.968381171286278, complex(1.9775309995995483, 7.747629119246556), complex(1.9775309995995483, -7.747629119246556)}
	a := []float64{1.968381171286278, 0, 0, 0, -3.3056869914010242e+10, 0.2739524212751154, -7.452505416638818e+08, 31022.31595250236, -0.0036934119373775447, -1.3381648221105064e-10, -0.12073917927091736, -1.9491487586739942e-05, 13231.64100776368, 3.787172142112033e-07, -269842.69014969293, 5.551312281470159}
	b := []float64{0, 44.52009700657428, 2.665485955874114e-10, -8.261839357512965e-06}
	c := []float64{4.527433297075258e+06, 0.0006283026295836041, -2.1217336804057643e+07, 1063.924795326897}
	for e := -12; e <= 12; e += 3 {
		B := mat.NewDense(4, 1, append([]float64(nil), b...))
		C := mat.NewDense(1, 4, append([]float64(nil), c...))
		B.Scale(math.Ldexp(1, e), B)
		C.Scale(math.Ldexp(1, -e), C)
		sys, err := New(mat.NewDense(4, 4, a), B, C, mat.NewDense(1, 1, nil), 0)
		if err != nil {
			t.Fatal(err)
		}
		got, err := sys.Zeros()
		if err != nil {
			t.Fatal(err)
		}
		zerosCheckExact(t, "α=2^"+strconv.Itoa(e), got, want)
	}
}

// TestZerosDependentIONearlySingularD: non-square models whose extra output
// (input) is an exact multiple of another and whose remaining D is nearly
// singular. The staircase eliminates the dependent row through D, which
// amplifies its roundoff past the rank tolerance and drops every zero; the
// square model without that row has exactly the same zeros.
func TestZerosDependentIONearlySingularD(t *testing.T) {
	cases := []struct {
		name       string
		n, m, p    int
		a, b, c, d []float64
		want       []complex128
	}{
		{name: "dep237", n: 2, m: 3, p: 2,
			a:    []float64{-1.2326316313645662, 0.38029762288352276, -1.9598609736640504, -0.0929814348952096},
			b:    []float64{-0.0032522904008535124, -0.03109658281326586, 0.0032522904008535124, 0.011372912671598429, 0.011611300979258608, -0.011372912671598429},
			c:    []float64{0.017525448664808273, 0.02847137696492877, 0.0695738369017219, 0.1005212973178026},
			d:    []float64{0.32123640120210545, 0.024932039152632514, -0.32123640120210545, 0.3818797820220178, 0.029638782112319507, -0.3818797820220178},
			want: []complex128{6.311173038659335, 18041.64920104227}},
		{name: "dep1463", n: 3, m: 3, p: 2,
			a:    []float64{-0.7445203265675637, 1.2264829597571625, 0.8281999970282334, -0.3060272765269061, 0.29203427507682556, -0.5244076601849224, 0.8776211259942411, 1.1526944578944265, -1.1192158460759312},
			b:    []float64{-0.011043862810157162, -0.006518729113028201, -0.005521931405078581, -0.0057082511603708365, 0.013304279523584935, -0.0028541255801854183, -0.002870877699378295, 0.003933200432410207, -0.0014354388496891474},
			c:    []float64{-2.782860381897641, -0.4688592075077306, 2.331772257202833, 1.2962573920142897, -3.6948674208858, 0.1347522867768171},
			d:    []float64{-1.3014193286600861, 0.5749351027389665, -0.6507096643300431, 0.4666211236114008, -0.2061417544033756, 0.2333105618057004},
			want: []complex128{-0.0008597955371780876, -1.8620297735715663, 2.872651618630687e+09}},
		{name: "dep1706", n: 4, m: 2, p: 3,
			a:    []float64{-0.10171944932262955, -0.46483203613064833, -0.9150334508735103, 0.30394022723843517, 0.2732086519787197, -0.5874076109684606, 0.5858703899983629, -1.0592913414992355, 0.007336042809721083, -0.47997732929254305, 0.06193288692076665, 0.5442285753447715, -0.04247159589113925, 0.3060002839179923, 0.11639548768477781, -2.132963097671741},
			b:    []float64{633.258676855718, -599.3073572873805, 354.67083434071884, 506.9102438096829, -631.1076563848753, 968.5841186689194, -728.4164431557809, -14.606448929499354},
			c:    []float64{-0.007080630164653769, 0.04200501359700777, 0.023506143605080534, -0.022541443301544645, 0.04336314687378425, 0.01789562229767794, -0.0007144697204940661, -0.004449271766196648, -0.0035403150823268845, 0.021002506798503885, 0.011753071802540267, -0.011270721650772322},
			d:    []float64{0.027427707448869878, 0.024362882282919242, -1.8422014997446574, -1.6363575576855371, 0.013713853724434939, 0.012181441141459621},
			want: []complex128{28.934424192509535, -0.6364376715467547, -2.2202453277882492, 3.46113641039748e+08}},
	}
	for _, tc := range cases {
		sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, 0)
		if err != nil {
			t.Fatal(err)
		}
		got, err := sys.Zeros()
		if err != nil {
			t.Fatal(err)
		}
		zerosCheckExact(t, tc.name, got, tc.want)
	}
}

// TestZerosNonSquareN1463AllScales: N1463 (exactly dependent third output,
// nearly singular D) keeps all three zeros for every B·α, C/α.
func TestZerosNonSquareN1463AllScales(t *testing.T) {
	want := []complex128{-7.648048437606871, -21.744896691779353, -2.4645885181395255e7}
	for e := -12; e <= 12; e++ {
		al := math.Ldexp(1, e)
		b := mat.NewDense(3, 2, []float64{276.10128900489735, -78.84692454909796, 127.82976523308767, 187.88740870711626, 321.2963935868809, 54.17661547127937})
		c := mat.NewDense(3, 3, []float64{0.17250793681915383, -0.11797312052017801, 0.10637833423332513, 0.3356157358156948, 0.24897800377747303, -0.08465722682754247, 0.08625396840957691, -0.058986560260089006, 0.053189167116662564})
		b.Scale(al, b)
		c.Scale(1/al, c)
		sys, err := New(mat.NewDense(3, 3, []float64{-2.8687132210778583, 1.2506740496003697, 1.1714746261684497, 0.14562856868484902, 1.061271822834533, 2.476186851807767, 0.6797388996832177, -2.2334341698308453, 1.1065143740965773}),
			b, c, mat.NewDense(3, 2, []float64{0.6004904931149847, -1.1639933550034214, -0.25179754018380357, 0.48809541549347146, 0.30024524655749235, -0.5819966775017107}), 0)
		if err != nil {
			t.Fatal(err)
		}
		got, err := sys.Zeros()
		if err != nil {
			t.Fatal(err)
		}
		zerosCheckExact(t, "α=2^"+strconv.Itoa(e), got, want)
	}
}

// TestZerosNonSquareNoSpurious: the unscaled staircase returns zeros the
// exact pencil lacks (they fail the backward-error test on the balanced
// pencil); Zeros must return only the exact ones.
func TestZerosNonSquareNoSpurious(t *testing.T) {
	cases := []struct {
		name       string
		n, m, p    int
		a, b, c, d []float64
		want       []complex128
	}{
		{name: "scl2", n: 3, m: 1, p: 3,
			a: []float64{-0.9277211661509566, -0.0005722112137490791, -1.1078904881331456e-08, 2439.590808080833, -0.5564843325623235, 6.0435983141866196e-05, 1.9349200723321673e+08, -31901.172184349427, -0.5220380533930202},
			b: []float64{83.7505081670728, -1.5673606171956575e+06, -6.418982299788733e+10},
			c: []float64{0.000983092807777378, 2.408515765949644e-07, -8.086748578352003e-12, 0.0025064828883775555, 4.447960434756958e-07, -1.0267472015392317e-11, 0.0008529903884394659, -2.7313974052505543e-07, 2.5039427218279044e-11},
			d: []float64{-1.4845314381722889e-09, -6.15267994055074e-10, -2.1984188717365057e-09}},
		{name: "scl1054", n: 5, m: 2, p: 1,
			a:    []float64{-0.8299681070969, 0, 0, 0, 0, -9.781266668460804e-05, 1.0349975930677613, 4.845525357804767e-08, 1.5977221035365583e-08, -9.27759256408355, 86.4443091239945, 1.1783135221718898e+07, 1.4167476096962084, 0.4230727588122204, -1.2607179448493252e+07, 2366.535214113991, 1.1198272968386417e+07, 8.101733688008478, -0.7386110110834687, 2.2560122987884164e+08, -4.575894911665153e-08, 0.18958968280737784, 1.6922899377287384e-08, -2.7648271104457763e-09, -0.6171772074092041},
			b:    []float64{0, 0, -0.00021355372352163234, -8.765264470325985e-05, 572.2292901318705, -2838.751434278842, -2696.522372426019, -1018.2440478756572, 4.5946851906776646e-05, -6.689438705572675e-06},
			c:    []float64{0.27739590815468246, -24732.633573305244, 0.0003399947586760124, 6.574488244685939e-05, 8372.442626780909},
			d:    []float64{0, 0},
			want: []complex128{-0.8299681070969}},
		{name: "nearDepRow", n: 1, m: 1, p: 2,
			a: []float64{-0.701072291885586},
			b: []float64{0.6319681003953592},
			c: []float64{-1.037730676933334, -4.150922707734095},
			d: []float64{-0.030168131825319747, -0.12067252730127899}},
		{name: "nearDepCol", n: 1, m: 2, p: 1,
			a: []float64{-0.701072291885586},
			b: []float64{-1.037730676933334, -4.150922707734095},
			c: []float64{0.6319681003953592},
			d: []float64{-0.030168131825319747, -0.12067252730127899}},
		{name: "nearDepRowD0", n: 2, m: 1, p: 2,
			a: []float64{0.6540170680805857, 1.1125721293641633, -1.4644235834355248, 1.2708003694616787},
			b: []float64{0.7610064633928397, 2.2199490567922213},
			c: []float64{-0.15878141295721226, 0.560137695688259, -1.270251303657491, 4.481101565508152},
			d: []float64{0, 0}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			sys, err := NewFromSlices(tc.n, tc.m, tc.p, tc.a, tc.b, tc.c, tc.d, 0)
			if err != nil {
				t.Fatal(err)
			}
			got, err := sys.Zeros()
			if err != nil {
				t.Fatal(err)
			}
			zerosCheckExact(t, tc.name, got, tc.want)
		})
	}
}

// TestZerosDependentRowMixedScale: the third output is exactly
// row1/2 + 4·row2 of [C D] with ‖C‖ ≈ 1e4, ‖D‖ ≈ 1 and D nearly singular,
// so its Gram–Schmidt residual is roundoff of the large entries; the
// dependency must still be recognised and the zeros (exact gcd of the
// maximal minors) kept.
func TestZerosDependentRowMixedScale(t *testing.T) {
	sys, err := NewFromSlices(2, 2, 3,
		[]float64{-1.25, 0.75, -2.5, -0.375},
		[]float64{1.5, -0.5, 0.25, 2.0},
		[]float64{12288.0, -3072.5, 20480.25, 8192.75, 88065.0, 31234.75},
		[]float64{0.625, -1.125, 0.3125, -0.5624990463256836, 1.5625, -2.8124961853027344}, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	zerosCheckExact(t, "mixed", got, []complex128{-5.806574835274511e10, -14771.879890036555})
}

// TestZerosScaledSquareRefined: square model with nearly singular D (so the
// staircase path) whose I/O scaling shrinks ‖[A B; C D]‖ from 1.4e5 to 6;
// the refinement threshold must still cover the zeros, of entrywise 1-ulp
// sensitivity ≤ 4e-14, at every exactly zero-preserving B·2^e, C·2^-e.
// Exact roots of det P(s) (big.Rat), each bracketed by a sign change.
func TestZerosScaledSquareRefined(t *testing.T) {
	a := []float64{1.3554877670069594, -3.633764730386259, -2302.6971590529843, -0.20868148341205206, -0.14066075783141757, -1.2673850296477662, -1234.2093141436703, -0.01632021365675509, -0.00011986927238098787, 0.00010768199454390479, -0.22560427971723207, 4.588483639337091e-05, 0.015050991535433544, 16.752533544640333, 11093.670407761474, -1.1117141209668198}
	b := []float64{0.024221199282688677, 0.19352894549112065, 98564.16380067778, 0.006837941522563402, 0.28674456183228414, 23498.70590149646, 3.392786466531403e-06, 0.00028885695706667275, -2.534864893546679, 0.03870599754813482, -2.8035601160549164, -63726.73510443856}
	c := []float64{-1.3445716724182244, -1.9154487072989506, 307.5575508850005, 0.10284034441549594, -0.0003995087848939206, 0.004753020787533528, 1.316365399393171, 0.00013498582091764317, 0.00010978313572977657, -0.0019202902520464582, 0.25030842698889344, -0.00012338352903634283}
	d := []float64{-0.27158561407239074, -1.9458299598524795, -142389.0784307856, -1.9618303034088424e-05, -0.0010775723105780565, -10.285640861136152, 4.390734024995e-05, -0.0009008948355075159, 23.020091604965785}
	want := []float64{2.7155495183828733, -1.7103487314512633, -0.0051847513214708894}
	poly := fzZeroPolyRat(4, 3, 3, a, b, c, d)
	for _, w := range want {
		lo, hi := new(big.Rat).SetFloat64(w*(1-1e-15)), new(big.Rat).SetFloat64(w*(1+1e-15))
		if fzPolyEvalRat(poly, lo).Sign()*fzPolyEvalRat(poly, hi).Sign() >= 0 {
			t.Fatalf("oracle %v: no sign change of det P(s)", w)
		}
	}
	for e := -12; e <= 12; e++ {
		B := append([]float64(nil), b...)
		C := append([]float64(nil), c...)
		for i := range B {
			B[i] = math.Ldexp(B[i], e)
		}
		for i := range C {
			C[i] = math.Ldexp(C[i], -e)
		}
		sys, err := NewFromSlices(4, 3, 3, a, B, C, d, 0)
		if err != nil {
			t.Fatal(err)
		}
		got, err := sys.Zeros()
		if err != nil {
			t.Fatal(err)
		}
		if len(got) != len(want) {
			t.Fatalf("e=%d: zeros %v, want %v", e, got, want)
		}
		for _, w := range want {
			best := math.Inf(1)
			for _, z := range got {
				best = math.Min(best, cmplx.Abs(z-complex(w, 0))/math.Abs(w))
			}
			if !(best <= 1e-12) {
				t.Errorf("e=%d: zero %v relative error %.2g > 1e-12\nzeros=%v", e, w, best, got)
			}
		}
	}
}

// TestZerosScaledSquareMidRefined: I/O scaling shrinks ‖[A B; C D]‖ from
// 5.6e3 to 5.5 and leaves the staircase pair 0.53±1.22i, of modulus above
// any ε·‖·‖/zeroRefineTol threshold, with relative error 4e-11 (base 1e-15):
// every staircase zero of a square model must be refined. Oracle: the
// big.Float Newton step on det P(s) (big.Rat) from each zero.
func TestZerosScaledSquareMidRefined(t *testing.T) {
	a := []float64{-0.770029571783819, -0.0007913196561441765, -4.271644296139483, 0.050365734734622025, 2.2875498268309773, 3333.7313230971845, 1.4828545510423157, 4392.171398094697, 28.83916505824172, -395.56316516111474, -0.10690326002839659, 6.649202090416916e-05, -1.6674458548443705, -0.007412492824969434, -0.32456373549895345, 17.45792321363927, 0.0025650653316172806, -6.753768358592318, 0.48875756530873904, 95.29410635698503, 0.2566160121951959, -6.597623568260576e-05, 0.2205144167322186, 0.00940322729271148, -0.5170404777699744}
	b := []float64{0.00839698260522784, 1.8724596214771585, -3.9695732367354157, 595.2718630556319, 0.0028436129893132326, 0.6373811019360747, -0.0211193382739083, 73.71718792310234, -0.0009698812701394, -0.4101010704518105}
	c := []float64{-47.310375195142, 0.0832723506401385, 668.7647859103607, 6.923295611311023, -445.13706547053687, 0.1272016650146747, 7.487490645982214e-05, 0.32122882754708837, 0.0023412416884003324, -0.45619465544117066}
	d := []float64{-0.9532654819900053, 163.75406503270034, -0.23831637049750132, 40.938516258175085}
	poly := fzTrimRat(fzPencilDetRat(5, 2, a, b, c, d))
	sys, err := NewFromSlices(5, 2, 2, a, b, c, d, 0)
	if err != nil {
		t.Fatal(err)
	}
	got, err := sys.Zeros()
	if err != nil {
		t.Fatal(err)
	}
	if len(got) != len(poly)-1 {
		t.Fatalf("zeros %v, want %d", got, len(poly)-1)
	}
	for _, z := range got {
		if e := fzNewtonRelRat(poly, z); !(e <= 1e-13) {
			t.Errorf("zero %v: relative error %.2g > 1e-13\nzeros=%v", z, e, got)
		}
	}
}

// fzNewtonRelRat returns |p(z)/p'(z)|/|z| in 400-bit arithmetic for the
// descending exact polynomial p: the relative distance from z to the
// nearest root when that root is simple and z is close to it.
func fzNewtonRelRat(p []*big.Rat, z complex128) float64 {
	const prec = 400
	nf := func() *big.Float { return new(big.Float).SetPrec(prec) }
	cmul := func(ar, ai, br, bi *big.Float) (*big.Float, *big.Float) {
		return nf().Sub(nf().Mul(ar, br), nf().Mul(ai, bi)), nf().Add(nf().Mul(ar, bi), nf().Mul(ai, br))
	}
	zr, zi := nf().SetFloat64(real(z)), nf().SetFloat64(imag(z))
	vr, vi, dr, di := nf(), nf(), nf(), nf()
	for _, c := range p {
		dr, di = cmul(dr, di, zr, zi)
		dr.Add(dr, vr)
		di.Add(di, vi)
		vr, vi = cmul(vr, vi, zr, zi)
		vr.Add(vr, nf().SetRat(c))
	}
	num := nf().Add(nf().Mul(vr, vr), nf().Mul(vi, vi))
	den := nf().Add(nf().Mul(dr, dr), nf().Mul(di, di))
	if den.Sign() == 0 {
		return math.Inf(1)
	}
	q, _ := nf().Quo(num, den).Float64()
	return math.Sqrt(q) / math.Max(cmplx.Abs(z), math.SmallestNonzeroFloat64)
}

// fzPolyEvalRat evaluates the descending polynomial p at x.
func fzPolyEvalRat(p []*big.Rat, x *big.Rat) *big.Rat {
	v := new(big.Rat)
	for _, c := range p {
		v.Mul(v, x)
		v.Add(v, c)
	}
	return v
}
