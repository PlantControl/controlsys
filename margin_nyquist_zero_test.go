package controlsys

import (
	"math"
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
