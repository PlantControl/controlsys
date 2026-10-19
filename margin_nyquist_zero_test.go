package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"
)

// CMVMWK: π is a crossover iff L(−1) of the coefficients as given, exact in
// big floats, is negative, as MATLAB evaluates them; with an exact (z+1)
// factor L(−1) = 0 and π is none, and a k·(z+1)⁴ cluster rounded to
// L(−1) ~ −1e-19, below the coefficient-rounding bound, is one.
// 6OPXWW: beside a zero cluster at z = −1, as of num ≈ k·(z+1)⁴, the pencil
// roots of L − L~ fall up to ~1e-4 off a phase crossover within that of
// Nyquist, where |L| ~ 1e-18, beyond any polish bracket; it was dropped.
// Oracle: the expanded coefficients evaluated in big floats, bisected on a
// grid graded to Nyquist; the repro and a seeded sweep of perturbed
// k·(z+1)⁴ over stable fifth-order denominators.
func TestAllMarginPhaseCrossNyquistZeroCluster(t *testing.T) {
	type tc struct{ num, den []float64 }
	cases := []tc{{
		[]float64{-0.0031586889782527225, -0.012630121036594033, -0.01893823083182162, -0.01262085446680662, -0.0031540556933263115},
		[]float64{1, -3.839652412015958, 5.8306455706710025, -4.376107218517926, 1.623325246208784, -0.2381308105407497},
	}}
	rng := rand.New(rand.NewPCG(61, 67))
	for i := range 80 {
		num := []float64{math.Copysign(math.Pow(10, -3+2*rng.Float64()), rng.Float64()-0.5)}
		for j := range 4 {
			d := math.Copysign(math.Pow(10, -4+2*rng.Float64()), rng.Float64()-0.5)
			if i%2 == 0 && j == 0 {
				d = 0
			}
			num = polyMul(num, []float64{1, 1 - d})
		}
		den := []float64{1, -(0.9 - 1.8*rng.Float64())}
		for range 2 {
			rho, phi := 0.3+0.65*rng.Float64(), math.Pi*rng.Float64()/2
			den = polyMul(den, []float64{1, -2 * rho * math.Cos(phi), rho * rho})
		}
		cases = append(cases, tc{num, den})
	}
	loop := func(c tc, th float64) complex128 { return bigHorner(c.num, th) / bigHorner(c.den, th) }
	var nearNyquist, exactZero, clusterPi int
	for ci, c := range cases {
		phase := func(th float64) float64 { return phaseOffsetDeg(loop(c, th)) }
		var thp []float64
		const N = 20000
		for i := 2; i < N; i++ {
			a, b := math.Pi*(1-math.Pow(1-float64(i-1)/N, 3)), math.Pi*(1-math.Pow(1-float64(i)/N, 3))
			pa, pb := phase(a), phase(b)
			if (pa > 0) == (pb > 0) || math.Abs(pa) >= 90 || math.Abs(pb) >= 90 {
				continue
			}
			for range 200 {
				m := (a + b) / 2
				if m <= a || m >= b {
					break
				}
				if pm := phase(m); (pm > 0) == (pa > 0) {
					a, pa = m, pm
				} else {
					b = m
				}
			}
			thp = append(thp, (a+b)/2)
		}
		if slices.ContainsFunc(thp, func(th float64) bool { return th > math.Pi-1e-3 }) {
			nearNyquist++
		}
		r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: 1}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatalf("case %d: %v", ci, err)
		}
		got, gm := res.PhaseCrossFreqs, res.GainMargins
		gotPi := len(got) > 0 && got[len(got)-1] == math.Pi
		if gotPi {
			got, gm = got[:len(got)-1], gm[:len(gm)-1]
		}
		var scale float64
		for _, v := range c.num {
			scale += math.Abs(v)
		}
		nyq := polyAtMinusOne(c.num, c.den)
		if n, _ := polyAtMinusOne(c.num, []float64{1}).Float64(); math.Abs(n) <= 1e-13*scale {
			switch nyq.Sign() {
			case 0:
				exactZero++
			case -1:
				clusterPi++
			}
		}
		lm1, _ := nyq.Float64()
		ok := len(got) == len(thp) && gotPi == (nyq.Sign() < 0)
		if gotPi {
			ok = ok && math.Abs(res.GainMargins[len(got)]+20*math.Log10(-lm1)) <= 1e-5
		}
		for i := 0; ok && i < len(thp); i++ {
			th := thp[i]
			ok = math.Abs(got[i]-th) <= 1e-7*th && math.Abs(gm[i]+20*math.Log10(cmplx.Abs(loop(c, th)))) <= 1e-5
		}
		if !ok {
			t.Errorf("case %d num %v den %v: θpc %v gm %v, want θ %v", ci, c.num, c.den, res.PhaseCrossFreqs, res.GainMargins, thp)
		}
	}
	if nearNyquist < 10 {
		t.Errorf("only %d cases cross within 1e-3 of Nyquist", nearNyquist)
	}
	if exactZero < 3 || clusterPi < 10 {
		t.Errorf("only %d exact zeros and %d negative L(−1) at |num(−1)| ≤ 1e-13·‖num‖₁", exactZero, clusterPi)
	}
}

// 6OPXWW: beside an exact zero cluster at z = −1 the scan towards Nyquist
// took a sign change of rounding residue, |L| ~ 1e-41, for a crossover, and
// L(−1) of ~1 ulp of |D| for one at Nyquist. Oracle: the coefficients in big
// floats, whose phase at the reported crossover tends to 0°, not −180°, and
// the crossovers below 3 found as before.
func TestAllMarginNyquistZeroClusterNoise(t *testing.T) {
	cases := []struct {
		num, den, want []float64
	}{
		{
			[]float64{0.004446950175721477, 0.01778780070288591, 0.026681701054328863, 0.01778780070288591, 0.004446950175721477},
			[]float64{1, 0.5924323257687443, 0.17033971248001745, 0.08044852775749614, -0.13270444245004534, -0.27271221810899804, -0.1506786024130224, 0.012149078195984445, 0.007692746757699009},
			[]float64{0.55129690992, 1.61388467294, 2.67343541852},
		},
		{
			[]float64{-0.004387008771532378, -0.01735277121673983, -0.025736271196440508, -0.016962263828791034, -0.004191755077557978},
			[]float64{1, 0.04447989509354311, -0.9820837702600874, -0.033079139783047384, 0.2251477532922962},
			nil,
		},
		{
			[]float64{-0.6955939875900586, -1.3911879751801173, -0.6955939875900586},
			[]float64{1, -0.20632506540049872, -0.12389966592884397},
			nil,
		},
	}
	for i, c := range cases {
		for _, d := range []float64{1e-6, 1e-8} {
			if p := phaseOffsetDeg(bigHornerBeforeNyquist(c.num, d) / bigHornerBeforeNyquist(c.den, d)); math.Abs(math.Abs(p)-180) > 2 {
				t.Fatalf("case %d: oracle phase offset %g° at π−%g, want ±180°", i, p, d)
			}
		}
		r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: 1}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		var got []float64
		for _, w := range res.PhaseCrossFreqs {
			if w > 3 {
				t.Errorf("case %d: crossover %.17g (GM %g dB), want none above 3", i, w, res.GainMargins[len(got)])
			} else {
				got = append(got, w)
			}
		}
		ok := len(got) == len(c.want)
		for k := 0; ok && k < len(got); k++ {
			ok = math.Abs(got[k]-c.want[k]) <= 1e-9*c.want[k]
		}
		if !ok {
			t.Errorf("case %d: θpc %v, want %v", i, res.PhaseCrossFreqs, c.want)
		}
	}
}

// bigHornerBeforeNyquist is bigHorner at θ = π − d with z = −1 + 2sin²(d/2)
// + j·sin d, which keeps z + 1 to full relative precision.
func bigHornerBeforeNyquist(p []float64, d float64) complex128 {
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(300).SetFloat64(x) }
	sh := math.Sin(d / 2)
	zr := nf(-1)
	zr.Add(zr, nf(2*sh*sh))
	zi := nf(math.Sin(d))
	re, im, t1, t2 := nf(0), nf(0), nf(0), nf(0)
	for _, c := range p {
		nr := nf(0).Sub(t1.Mul(re, zr), t2.Mul(im, zi))
		im.Add(t1.Mul(re, zi), t2.Mul(im, zr))
		re.Add(nr, nf(c))
	}
	r, _ := re.Float64()
	i, _ := im.Float64()
	return complex(r, i)
}

// L real negative at Nyquist is a crossover unless L is real on the whole
// boundary, which has no isolated crossings.
func TestAllMarginDiscreteNyquistRealEnd(t *testing.T) {
	cases := []struct {
		num, den []float64
		want     []float64
	}{
		{[]float64{-3}, []float64{1}, nil},
		{[]float64{-2, 0}, []float64{1, 0}, nil},
		{[]float64{0.5}, []float64{1, 0.5}, []float64{math.Pi}},
	}
	for _, c := range cases {
		r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: 1}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatal(err)
		}
		if !slices.Equal(res.PhaseCrossFreqs, c.want) {
			t.Errorf("num %v den %v: θpc %v, want %v", c.num, c.den, res.PhaseCrossFreqs, c.want)
		}
	}
}

// 3FHD47: the scan towards Nyquist solved in double-double at each of ~50
// points, O(n³) each, most of AllMargin's time on high-order discrete loops.
// It keeps the double solve where |L| is far above the coefficient noise and
// the phase offset is not small; the crossings stay those of phaseCrossings.
func TestNyquistPhaseCrossingsScanCost(t *testing.T) {
	for kind := range sweepModelKinds {
		r, err := newRationalLoop(marginRandomLoop(t, 60, kind, 0.05, false, false))
		if err != nil {
			t.Fatal(err)
		}
		ws, err := r.phaseCrossings()
		if err != nil {
			t.Fatal(err)
		}
		r.eval.ddSolves = 0
		f := func(w float64) float64 { return phaseOffsetDeg(r.eval.at(w)) }
		got := r.nyquistPhaseCrossings(slices.Clone(ws), f, 1e-7)
		if !slices.Equal(got, ws) {
			t.Errorf("%v: nyquistPhaseCrossings = %v, want %v", kind, got, ws)
		}
		if r.eval.ddSolves > 30 {
			t.Errorf("%v: %d double-double solves, want ≤ 30", kind, r.eval.ddSolves)
		}
	}
}

// polyAtMinusOne is num(−1)/den(−1) of the coefficients as given, exactly.
func polyAtMinusOne(num, den []float64) *big.Float {
	at := func(p []float64) *big.Float {
		s := new(big.Float).SetPrec(3000)
		for i, c := range p {
			v := new(big.Float).SetPrec(3000).SetFloat64(c)
			if (len(p)-1-i)%2 == 1 {
				v.Neg(v)
			}
			s.Add(s, v)
		}
		return s
	}
	return new(big.Float).SetPrec(3000).Quo(at(num), at(den))
}

// bigNyquistPhaseCrossings returns the θ in (0, π) where num/den·e^{−jkθ}, in big
// floats, crosses −180°, bisected on a grid graded to Nyquist as π − θ
// from π to 1e-15, with z + 1 kept to full relative precision.
func bigNyquistPhaseCrossings(num, den []float64, k float64) []float64 {
	loop := func(d float64) complex128 {
		return bigHornerBeforeNyquist(num, d) / bigHornerBeforeNyquist(den, d) * cmplx.Exp(complex(0, -k*(math.Pi-d)))
	}
	phase := func(d float64) float64 { return phaseOffsetDeg(loop(d)) }
	var out []float64
	const N = 30000
	for i := 1; i <= N; i++ {
		a, b := math.Pi*math.Pow(1e-15/math.Pi, float64(i-1)/N), math.Pi*math.Pow(1e-15/math.Pi, float64(i)/N)
		pa, pb := phase(a), phase(b)
		if (pa > 0) == (pb > 0) || math.Abs(pa) >= 90 || math.Abs(pb) >= 90 {
			continue
		}
		for range 200 {
			m := (a + b) / 2
			if m >= a || m <= b {
				break
			}
			if pm := phase(m); (pm > 0) == (pa > 0) {
				a, pa = m, pm
			} else {
				b = m
			}
		}
		out = append(out, math.Pi-(a+b)/2)
	}
	return out
}

// CMVMWK, EOJHCC: π is a crossover iff L(−1) of the loop as given, with its
// delays, is negative, as MATLAB evaluates the coefficients; it was decided
// on the phase at frequencyPoint(π), z = −1 + 1.2e-16j, where L′(−1)·1.2e-16j
// outweighs L(−1) ~ 1e-17 beside a (z+1) cluster, and its GM taken there;
// for an odd input delay the sign flips. A crossing just below π where L
// runs from ~0° to −180° between scan points was taken for a wrap. Oracle:
// the coefficients in big floats, L(−1) exactly and the crossings bisected.
func TestAllMarginNyquistExactSign(t *testing.T) {
	c730n := []float64{-0.011204978782742445, -0.05984870582771428, -0.1273450360277268, -0.13499266094431892, -0.07132014320260238, -0.015028791241038416}
	c730d := []float64{1, -0.21579742457949225, -0.6185444651785603, 0.005936187798701448, -0.0008252966705985937, 0.09760593278156476, 0.0066096485407187225, -0.017048862169604856}
	cases := []struct {
		num, den []float64
		delay    float64
	}{
		{c730n, c730d, 0},
		{c730n, c730d, 1},
		{c730n, c730d, 2},
		{c730n, c730d, 3},
		{[]float64{0.44871625522376757, -1.5619278996205108, -1.538662685583291, 0.47198146926098755}, []float64{1, -1.7146062608388555, 1.2053818134652472, -0.37983840377876593, 0.026501847959910826, 0.009097592112874071}, 0},
		{[]float64{-0.006828289289983037, -0.010983952181621664, -0.004155662891638626}, []float64{1, -0.14997124900739556, -0.30071216944539947, -0.054229924006297454}, 0},
		{[]float64{-0.004265746045965482, -0.01706298427945588, -0.02559447656257475, -0.017062984470643788, -0.004265746141559436}, []float64{1, -0.7421381273163463, -0.514584142978884, 0.4577149265936352, -0.010332950949376953, -0.04897201276908954, 0.009974007600712643}, 0},
		{[]float64{0.22828544824082456, 0.9131419443948608, 1.369713143739327, 0.9131422472573697, 0.22828559967207898}, []float64{1, -3.4773867605081685, 5.089349418419434, -3.819062668520739, 1.4449385392353498, -0.2083616853249832}, 0},
		{[]float64{-0.04158478486918814, -0.1663391396743808, -0.24950870980801354, -0.1663391400696373, -0.041584785066816385}, []float64{1, -1.3795422707987064, 0.1409204839067461, 0.5307549918293859, -0.2951031384341074, 0.036975056986437035, -0.0009415329631018891}, 0},
		{[]float64{0.08755259620909053, 0.35021038683847777, 0.52531558326089, 0.350210390842709, 0.08755259821120616}, []float64{1, -1.2623671592125492, 0.03962272839964098, 0.5895720546592378, -0.3005780332327763, 0.04559429288019802}, 0},
	}
	for ci, c := range cases {
		r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: 1}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		sys := r.Sys
		if sys.D.At(0, 0) != 0 {
			t.Fatalf("case %d: D ≠ 0", ci)
		}
		if c.delay > 0 {
			sys.InputDelay = []float64{c.delay}
		}
		lm1, _ := polyAtMinusOne(c.num, c.den).Float64()
		lm1 *= math.Pow(-1, c.delay)
		if !(math.Abs(lm1) > 1e-19 && math.Abs(lm1) < 1e-16) {
			t.Fatalf("case %d: L(−1) %g, want a (z+1) cluster's", ci, lm1)
		}
		want := bigNyquistPhaseCrossings(c.num, c.den, c.delay)
		wantGM := make([]float64, len(want))
		for i, th := range want {
			wantGM[i] = -20 * math.Log10(cmplx.Abs(bigHornerBeforeNyquist(c.num, math.Pi-th)/bigHornerBeforeNyquist(c.den, math.Pi-th)))
		}
		if lm1 < 0 {
			want, wantGM = append(want, math.Pi), append(wantGM, -20*math.Log10(-lm1))
		}
		res, err := AllMargin(sys)
		if err != nil {
			t.Fatalf("case %d: %v", ci, err)
		}
		ok := len(res.PhaseCrossFreqs) == len(want)
		for i := 0; ok && i < len(want); i++ {
			ok = math.Abs(res.PhaseCrossFreqs[i]-want[i]) <= 1e-9*want[i] && math.Abs(res.GainMargins[i]-wantGM[i]) <= 1e-5
		}
		if !ok {
			t.Errorf("case %d (delay %g, L(−1) %.4g): θpc %.17g gm %v, want %.17g gm %v", ci, c.delay, lm1, res.PhaseCrossFreqs, res.GainMargins, want, wantGM)
		}
	}
}

// EOJHCC, D ≠ 0: beside a (z+1)³ cluster with num(−1) = 0 exactly, the
// biproper realization's L near π lies within its coefficient noise, of
// C = num − D·den rounded; the scan took a sign change there at π − 1.4e-6
// for a crossover, and π for another. Oracle: the coefficients in big
// floats, with no crossing in (3, π] and L(−1) = 0.
func TestAllMarginNyquistBiproperNoise(t *testing.T) {
	num := []float64{-0.011933620198413512, -0.035800758950438785, -0.03580065730565901, -0.011933518553633738}
	den := []float64{1, -0.1415166906468338, 0.16103257676435861, 0.6346090124581071}
	if polyAtMinusOne(num, den).Sign() != 0 || slices.ContainsFunc(bigNyquistPhaseCrossings(num, den, 0), func(th float64) bool { return th > 3 }) {
		t.Fatal("oracle: want L(−1) = 0 and no crossing above 3")
	}
	r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: 1}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	res, err := AllMargin(r.Sys)
	if err != nil {
		t.Fatal(err)
	}
	if slices.ContainsFunc(res.PhaseCrossFreqs, func(w float64) bool { return w > 3 }) {
		t.Errorf("θpc %.17g gm %v, want none above 3", res.PhaseCrossFreqs, res.GainMargins)
	}
}
