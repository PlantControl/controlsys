package controlsys

import (
	"math"
	"math/big"
	"math/rand/v2"
	"slices"
	"testing"
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
