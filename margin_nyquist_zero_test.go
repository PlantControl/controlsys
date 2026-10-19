package controlsys

import (
	"math"
	"math/big"
	"math/cmplx"
	"math/rand/v2"
	"slices"
	"testing"
)

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
	var nearNyquist int
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
		nyq := loop(c, math.Pi)
		ok := len(got) == len(thp)
		if cmplx.Abs(bigHorner(c.num, math.Pi)) > 1e-13*scale {
			ok = ok && gotPi == (real(nyq) < 0)
			if gotPi {
				ok = ok && math.Abs(res.GainMargins[len(got)]+20*math.Log10(cmplx.Abs(nyq))) <= 1e-5
			}
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
