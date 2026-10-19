package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"

	"plantcontrol.org/v1/gonum/mat"
)

const bigPrec = 160

// bigLoopAt returns N(jω)·conj(D(jω)) in big.Float from float64 coefficients
// in descending powers.
func bigLoopAt(num, den []float64, w float64) (re, im *big.Float) {
	polyAt := func(p []float64) (*big.Float, *big.Float) {
		bw := new(big.Float).SetPrec(bigPrec).SetFloat64(w)
		pr, pi := new(big.Float).SetPrec(bigPrec), new(big.Float).SetPrec(bigPrec)
		for _, c := range p {
			nr := new(big.Float).SetPrec(bigPrec).Mul(pi, bw)
			nr.Neg(nr)
			nr.Add(nr, new(big.Float).SetFloat64(c))
			ni := new(big.Float).SetPrec(bigPrec).Mul(pr, bw)
			pr, pi = nr, ni
		}
		return pr, pi
	}
	nr, ni := polyAt(num)
	dr, di := polyAt(den)
	mul := func(a, b *big.Float) *big.Float { return new(big.Float).SetPrec(bigPrec).Mul(a, b) }
	re = new(big.Float).SetPrec(bigPrec).Add(mul(nr, dr), mul(ni, di))
	im = new(big.Float).SetPrec(bigPrec).Sub(mul(ni, dr), mul(nr, di))
	return re, im
}

// bigPhaseCrossings returns the float64 ω on grid where Im(N·conj D) changes
// sign with Re < 0, bisected in big.Float, with the gain margin there.
func bigPhaseCrossings(num, den, grid []float64) (ws, gm []float64) {
	sign := func(w float64) int {
		_, im := bigLoopAt(num, den, w)
		return im.Sign()
	}
	for i := 1; i < len(grid); i++ {
		a, b := grid[i-1], grid[i]
		sa, sb := sign(a), sign(b)
		if sa*sb > 0 || sa == 0 {
			continue
		}
		for {
			m := a + (b-a)/2
			if m <= a || m >= b {
				break
			}
			if sm := sign(m); sm == 0 {
				a, b = m, m
				break
			} else if sm == sa {
				a = m
			} else {
				b = m
			}
		}
		re, _ := bigLoopAt(num, den, b)
		if re.Sign() >= 0 {
			continue
		}
		ws = append(ws, b)
		bw := complex(0, b)
		var n, d complex128
		for _, c := range num {
			n = n*bw + complex(c, 0)
		}
		for _, c := range den {
			d = d*bw + complex(c, 0)
		}
		l := n / d
		gm = append(gm, -20*math.Log10(math.Hypot(real(l), imag(l))))
	}
	return ws, gm
}

// resonanceGrid covers [lo, hi] logarithmically and grades towards wn down
// to a relative distance of 1e-13.
func resonanceGrid(lo, hi, wn float64) []float64 {
	var g []float64
	for x := math.Log10(lo); x <= math.Log10(hi); x += 0.02 {
		g = append(g, math.Pow(10, x))
	}
	for k := -13.0; k <= -1; k += 0.05 {
		d := math.Pow(10, k)
		g = append(g, wn*(1-d), wn*(1+d))
	}
	slices.Sort(g)
	return slices.Compact(g)
}

// A phase crossover inside the resonance band of a pair with ζ ≲ 1e-7 moves f
// by more than 1e-7° per ulp of ω, so no float ω meets that tolerance; it is
// accepted on its bisected sign change. Oracle: float64 coefficients
// evaluated in big.Float and bisected for Im L = 0, Re L < 0.
func TestAllMargin_LightlyDampedPhaseCrossover(t *testing.T) {
	cases := []struct {
		num, den []float64
		wn       float64
	}{
		{[]float64{-4.998581848916508, 0.7194337044136604}, []float64{1, 3.4849223836704207e-07, 209.69770538172128}, math.Sqrt(209.69770538172128)},
		{[]float64{-94.7735203436477, -510.976787820926}, []float64{1, 3.3469538656795647e-08, 54.710049890106276}, math.Sqrt(54.710049890106276)},
		{[]float64{401.7319732939735}, []float64{1, 0.4333987836173656, 1.0446079475645176, 0.4527317905845433}, 1.0220606},
	}
	rng := rand.New(rand.NewPCG(42, 7))
	for range 200 {
		wn := math.Pow(10, -1+2*rng.Float64())
		zeta := math.Pow(10, -9+8*rng.Float64())
		den := []float64{1, 2 * zeta * wn, wn * wn}
		if rng.IntN(2) == 0 {
			p := math.Pow(10, -1+2*rng.Float64())
			den = []float64{1, den[1] + p, den[2] + p*den[1], p * den[2]}
		}
		num := []float64{10 * rng.NormFloat64()}
		if rng.IntN(2) == 0 {
			num = append(num, 10*rng.NormFloat64())
		}
		cases = append(cases, struct {
			num, den []float64
			wn       float64
		}{num, den, wn})
	}
	for i, c := range cases {
		res, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		all, err := AllMargin(res.Sys)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		want, wantGM := bigPhaseCrossings(c.num, c.den, resonanceGrid(1e-4, 1e4, c.wn))
		if len(all.PhaseCrossFreqs) != len(want) {
			t.Errorf("case %d num %v den %v: wpc %v, want %v", i, c.num, c.den, all.PhaseCrossFreqs, want)
			continue
		}
		for k, w := range want {
			if got := all.PhaseCrossFreqs[k]; math.Abs(got-w) > 1e-9*w {
				t.Errorf("case %d: wpc[%d] = %.17g, want %.17g", i, k, got, w)
			}
			if got := all.GainMargins[k]; math.Abs(got-wantGM[k]) > 1e-6*max(1, math.Abs(wantGM[k])) {
				t.Errorf("case %d: GM[%d] = %.17g, want %.17g", i, k, got, wantGM[k])
			}
		}
	}
}

// bigGainMarginDB returns −20·log10|N(jω)/D(jω)| from big.Float N·conj D and |D|².
func bigGainMarginDB(num, den []float64, w float64) float64 {
	re, im := bigLoopAt(num, den, w)
	d2, _ := bigLoopAt(den, den, w)
	r, _ := re.Float64()
	i, _ := im.Float64()
	d, _ := d2.Float64()
	return -20 * math.Log10(math.Hypot(r, i)/d)
}

// BYP4ZM: beside a pair with ζ ~ 1e-12 the phase crossover lies within ulps
// of the pole and f moves by ~0.1° per ulp; the scan held it to 1e-3° and to
// clearance points rounded past the |f| = 90 bracket end, and missed it.
// Oracle: as for TestAllMargin_LightlyDampedPhaseCrossover; GM from big.Float
// |L| at the reported ω, since |L| moves by ~1e-4 per ulp there.
func TestAllMarginPhaseCrossUltraLightResonance(t *testing.T) {
	type tc struct {
		num, den []float64
		wn       float64
	}
	cases := []tc{
		{[]float64{-14.58076267912999, 0.2780388291113343}, []float64{1, 1.7493356310572988e-11, 60.9821144190896}, math.Sqrt(60.9821144190896)},
		{[]float64{-0.47065815665752997, -78.56895717226088}, []float64{1, 4.978584544201923, 349.09061740501886, 1737.9771496708024, 0.23425954438849886, 1.1653612415035386}, 18.6839488959572},
		{[]float64{-10.117578176208571, -11.554653627263235}, []float64{1, 5.725929450921608e-13, 0.5811185290875198}, math.Sqrt(0.5811185290875198)},
	}
	rng := rand.New(rand.NewPCG(5, 11))
	for range 300 {
		wn := math.Pow(10, -1+2*rng.Float64())
		zeta := math.Pow(10, -12.5+2.5*rng.Float64())
		den := []float64{1, 2 * zeta * wn, wn * wn}
		if rng.IntN(2) == 0 {
			p := math.Pow(10, -1+2*rng.Float64())
			den = []float64{1, den[1] + p, den[2] + p*den[1], p * den[2]}
		}
		num := []float64{10 * rng.NormFloat64()}
		if rng.IntN(2) == 0 {
			num = append(num, 10*rng.NormFloat64())
		}
		cases = append(cases, tc{num, den, wn})
	}
	for i, c := range cases {
		res, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		all, err := AllMargin(res.Sys)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		want, _ := bigPhaseCrossings(c.num, c.den, resonanceGrid(1e-4, 1e4, c.wn))
		ok := len(all.PhaseCrossFreqs) == len(want)
		for k := 0; ok && k < len(want); k++ {
			w := all.PhaseCrossFreqs[k]
			gm := bigGainMarginDB(c.num, c.den, w)
			ok = math.Abs(w-want[k]) <= 1e-9*want[k] && math.Abs(all.GainMargins[k]-gm) <= 1e-6*max(1, math.Abs(gm))
		}
		if !ok {
			t.Errorf("case %d num %v den %v: wpc %v GM %v, want %v", i, c.num, c.den, all.PhaseCrossFreqs, all.GainMargins, want)
		}
	}
}

// bigUnitCosSin returns cos θ and sin θ in big.Float: Taylor series at
// θ/2¹⁰, then ten angle doublings.
func bigUnitCosSin(th float64) (c, s *big.Float) {
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(bigPrec).SetFloat64(x) }
	x := nf(th)
	x.SetMantExp(x, -10)
	c, s, term := nf(0), nf(0), nf(1)
	for k := range 40 {
		switch k % 4 {
		case 0:
			c.Add(c, term)
		case 1:
			s.Add(s, term)
		case 2:
			c.Sub(c, term)
		default:
			s.Sub(s, term)
		}
		term.Mul(term, x).Quo(term, nf(float64(k+1)))
	}
	for range 10 {
		c, s = nf(0).Sub(nf(0).Mul(c, c), nf(0).Mul(s, s)), nf(0).Mul(nf(2), nf(0).Mul(s, c))
	}
	return c, s
}

// bigUnitLoop returns N(e^{jθ})·conj(D(e^{jθ})) and |N/D| in big.Float from
// float64 coefficients in descending powers of z.
func bigUnitLoop(num, den []float64, th float64) (re, im *big.Float, mag float64) {
	nf := func(x float64) *big.Float { return new(big.Float).SetPrec(bigPrec).SetFloat64(x) }
	mul := func(a, b *big.Float) *big.Float { return nf(0).Mul(a, b) }
	c, s := bigUnitCosSin(th)
	at := func(p []float64) (re, im *big.Float) {
		re, im = nf(0), nf(0)
		for _, a := range p {
			re, im = nf(0).Sub(mul(re, c), mul(im, s)), nf(0).Add(mul(re, s), mul(im, c))
			re.Add(re, nf(a))
		}
		return re, im
	}
	nr, ni := at(num)
	dr, di := at(den)
	n2, _ := nf(0).Add(mul(nr, nr), mul(ni, ni)).Float64()
	d2, _ := nf(0).Add(mul(dr, dr), mul(di, di)).Float64()
	return nf(0).Add(mul(nr, dr), mul(ni, di)), nf(0).Sub(mul(ni, dr), mul(nr, di)), math.Sqrt(n2 / d2)
}

// unitPairAngle returns the smallest positive angle among the complex roots
// of den.
func unitPairAngle(t *testing.T, den []float64) float64 {
	n := len(den) - 1
	comp := mat.NewDense(n, n, nil)
	for j := range n {
		comp.Set(0, j, -den[j+1]/den[0])
		if j > 0 {
			comp.Set(j, j-1, 1)
		}
	}
	var eig mat.Eigen
	if !eig.Factorize(comp, mat.EigenNone) {
		t.Fatal("eigen failed")
	}
	th := math.Inf(1)
	for _, p := range eig.Values(nil) {
		if a := cmplx.Phase(p); imag(p) > 0 && a < th {
			th = a
		}
	}
	return th
}

// bigUnitPhaseCrossings returns the θ in (0, π) where Im L(e^{jθ}) changes
// sign with Re L < 0, on a log grid from 1e-10 graded towards thp, bisected
// in big.Float.
func bigUnitPhaseCrossings(num, den []float64, thp float64) []float64 {
	var g []float64
	for x := -10.0; x < math.Log10(math.Pi); x += 0.02 {
		g = append(g, math.Pow(10, x))
	}
	for k := -13.0; k <= -0.5; k += 0.05 {
		d := math.Pow(10, k)
		g = append(g, thp*(1-d), thp*(1+d))
	}
	g = append(g, math.Pi*(1-1e-12))
	slices.Sort(g)
	sign := func(th float64) int { _, im, _ := bigUnitLoop(num, den, th); return im.Sign() }
	var out []float64
	for i := 1; i < len(g); i++ {
		a, b := g[i-1], g[i]
		sa := sign(a)
		if sa == 0 || sa*sign(b) > 0 {
			continue
		}
		for {
			m := (a + b) / 2
			if m <= a || m >= b {
				break
			}
			if sign(m) == sa {
				a = m
			} else {
				b = m
			}
		}
		if re, _, _ := bigUnitLoop(num, den, b); re.Sign() < 0 {
			out = append(out, b)
		}
	}
	return out
}

// VT3OF7: crossovers beside a pair 1 ± jε, below it down to ~1e-2 of its
// frequency, where the phase stays within 1e-8° of −180°, or just past it,
// where the double solve's phase is off by 1e-5°. Oracle: float64
// coefficients at exact e^{jθ} in big.Float, Im L bisected, Re L < 0. Where
// the phase is that flat, arg L + 180° rounded to ulp(360) moved the
// crossing by up to 1e-5 relative (XRXQ4I); crossings must match to 1e-7.
func TestAllMarginPhaseCrossBesideUnitPolePair(t *testing.T) {
	type tc struct {
		num, den []float64
		dt       float64
	}
	cases := []tc{
		{[]float64{-0.0013196134305395058, -0.00023536903292695072}, []float64{1, -2.3854598907075455, 1.770919781465705, -0.3854598907270551}, 0.1},
		{[]float64{-4.737105371029762, -1.0653329101644098}, []float64{1, -0.4415335279470294, -1.5153667659780603, 0.355334116135632, 0.6015661780899885}, 1},
		{[]float64{-0.8297128408865633, -0.25644473472126117}, []float64{1, -1.7049260410106568, 0.40985208202314993, 0.29507395898988503}, 1},
		{[]float64{0.010321862812979532, -2.1987678119075276}, []float64{1, -2, 1.0000000000036688}, 1},
		{[]float64{0.00043269717096372506, -2.5205075284572747}, []float64{1, -2, 1.0000000032590524}, 1},
		{[]float64{0.4156266235836725, -0.4127608532382766}, []float64{1, -2, 1.0000000006278533}, 1},
		{[]float64{0.3541729069514521, -0.7616599197851204}, []float64{1, -2, 1.0000000000823044}, 1},
		{[]float64{-0.8619931462988315, -0.4791089481661279}, []float64{1, -1.4487438779878845, -0.10251224401424697, 0.5512561220176192}, 1},
		{[]float64{-0.0002929580685127985}, []float64{1, -2.1599186255777103, 2.317720058857019, -2.155684240978195, 0.9978828077015948}, 0.1},
		{[]float64{-0.012285664809521195, -0.013093860530681841, -0.016387643969841954}, []float64{1, -3.8905296631703514, 5.780997890779641, -3.890406792047608, 0.9999385644389388}, 0.01},
	}
	rng := rand.New(rand.NewPCG(21, 99))
	for range 30 {
		e := math.Pow(10, -6+4*rng.Float64())
		den := []float64{1, -2, 1 + e*e}
		for range rng.IntN(3) {
			den = polyMul(den, []float64{1, 0.9 - 1.8*rng.Float64()})
		}
		cases = append(cases, tc{[]float64{rng.NormFloat64(), rng.NormFloat64()}, den, 1})
	}
	for i, c := range cases {
		r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: c.dt}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		want := bigUnitPhaseCrossings(c.num, c.den, unitPairAngle(t, c.den))
		var got, gm []float64
		for k, w := range res.PhaseCrossFreqs {
			if w*c.dt < math.Pi*(1-1e-12) {
				got, gm = append(got, w*c.dt), append(gm, res.GainMargins[k])
			}
		}
		ok := len(got) == len(want)
		for k := 0; ok && k < len(want); k++ {
			re, _, mag := bigUnitLoop(c.num, c.den, got[k])
			ok = re.Sign() < 0 && math.Abs(got[k]-want[k]) <= 1e-7*want[k] &&
				math.Abs(gm[k]+20*math.Log10(mag)) <= 1e-5
		}
		if !ok {
			t.Errorf("case %d num %v den %v: θpc %v GM %v, want θ %v", i, c.num, c.den, got, gm, want)
		}
	}
}

// WSF2HP: crossovers beside a real pair straddling or next to z = 1, where
// the phase stays within ~1e-6 rad of −180° up to θ ~ ε and the double
// solve's phase error is of that order: crossings were missed, or reported
// a few percent off (case 1). Oracle: as for
// TestAllMarginPhaseCrossBesideUnitPolePair.
func TestAllMarginPhaseCrossBesideRealUnitPolePair(t *testing.T) {
	type tc struct {
		num, den []float64
		eps      float64
	}
	cases := []tc{
		{[]float64{0.6942811935961309, 4.422597857871295}, []float64{1, -2, 0.9999999998197421}, 1.34e-5},
		{[]float64{1e-3, 1}, []float64{1, -2, 1 - 2e-4*2e-4}, 2e-4},
		{[]float64{1e-4, 1}, []float64{1, -2, 1 - 2e-3*2e-3}, 2e-3},
		{[]float64{10, 1}, []float64{1, -2, 1 - 5e-4*5e-4}, 5e-4},
	}
	rng := rand.New(rand.NewPCG(7, 31))
	for k := range 45 {
		e := math.Pow(10, -6+3*rng.Float64())
		num := []float64{rng.NormFloat64(), rng.NormFloat64()}
		var den []float64
		switch k % 3 {
		case 0:
			den = []float64{1, -2, 1 - e*e}
			num[1] = math.Abs(num[1])
			num[0] = num[1] * math.Pow(10, -3*rng.Float64())
		case 1:
			den = polyMul([]float64{1, -1 + e}, []float64{1, -1 + e*(1+4*rng.Float64())})
		default:
			den = polyMul([]float64{1, -1}, []float64{1, -1 + e})
		}
		if k%3 != 0 && rng.IntN(2) == 1 {
			den = polyMul(den, []float64{1, 0.9 - 1.8*rng.Float64()})
		}
		cases = append(cases, tc{num, den, e})
	}
	for i, c := range cases {
		r, err := (&TransferFunc{Num: [][][]float64{{c.num}}, Den: [][]float64{c.den}, Dt: 1}).StateSpace()
		if err != nil {
			t.Fatal(err)
		}
		res, err := AllMargin(r.Sys)
		if err != nil {
			t.Fatalf("case %d: %v", i, err)
		}
		var want []float64
		for _, w := range bigUnitPhaseCrossings(c.num, c.den, c.eps) {
			if w < 10*c.eps {
				want = append(want, w)
			}
		}
		var got, gm []float64
		for k, w := range res.PhaseCrossFreqs {
			if w < 10*c.eps {
				got, gm = append(got, w), append(gm, res.GainMargins[k])
			}
		}
		ok := len(got) == len(want)
		for k := 0; ok && k < len(want); k++ {
			re, _, mag := bigUnitLoop(c.num, c.den, got[k])
			ok = re.Sign() < 0 && math.Abs(got[k]-want[k]) <= 1e-7*want[k] &&
				math.Abs(gm[k]+20*math.Log10(mag)) <= 1e-5
		}
		if !ok {
			t.Errorf("case %d num %v den %v: θpc %v GM %v, want θ %v", i, c.num, c.den, got, gm, want)
		}
	}
}

// WSF2HP scanned beside each real pole next to z = 1 in double-double, and
// slow modes at Δt = 0.05 put dozens there: AllMargin of order 150 took
// minutes. The scan shares one grid across them and keeps the double solve
// where the phase is ≥ 1e-2° from −180°.
func TestAllMarginRealUnitPoleScanCost(t *testing.T) {
	for _, n := range []int{60, 100} {
		r, err := newRationalLoop(marginRandomLoop(t, n, sweepWeakRows, 0.05, false, false))
		if err != nil {
			t.Fatal(err)
		}
		if _, reals := r.resonancePoles(); len(reals) < 5 {
			t.Fatalf("n=%d: %d real poles next to z = 1, want ≥ 5", n, len(reals))
		}
		if _, err := r.phaseCrossings(); err != nil {
			t.Fatal(err)
		}
		if r.eval.ddSolves > 400 {
			t.Errorf("n=%d: %d double-double solves, want ≤ 400", n, r.eval.ddSolves)
		}
	}
}
