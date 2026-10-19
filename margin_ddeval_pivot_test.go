package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// TestDDSISOResponseCrowdedIntegrators: den = s³·q(s), q with real roots
// below ~1e-6 (TestDelayNyquistTinyRealPolesAtOrigin). At s = jω, ω ≤ 1e-6,
// det(sI−A) ~ 1e-31 fell below the fixed pivot floor n·scale·2⁻¹⁰⁰ though
// exact to dd precision, so ddSISOResponse failed where |L| ~ 1e29..1e41.
// Oracle: num and den evaluated exactly in big.Float at jω.
func TestDDSISOResponseCrowdedIntegrators(t *testing.T) {
	cases := []struct{ num, den []float64 }{
		{[]float64{5.52461747222729, 10.25395216896589, 7.411393861456306, 2.08816698988852, 0.22553859559127404, 0.007420390418043597}, []float64{1, -0.008139247275845123, -7.936569881325126e-08, -2.288908801286239e-14, 0, 0, 0}},
		{[]float64{8.355629538965681, 22.563917082193022, 24.398355785727624, 9.639372238938034, 1.1305743755868467, 0.03782054040717314}, []float64{1, -0.0028639980109894134, -3.425788951502568e-05, -2.0789893203109462e-11, 0, 0, 0}},
		{[]float64{4.551658530169233, 3.0753000142052764, 0.9194674239822193, 0.11054865480512942, 0.003182926796710191}, []float64{1, -0.0048136878504125066, -1.6692283338125261e-06, -2.651188267756408e-13, 0, 0}},
		{[]float64{8.306255406460808, 22.680683272368427, 26.113994842611685, 12.223126925616675, 1.6176301437180853, 0.05728619385528791}, []float64{1, 0.007068687570826481, -2.5428999759620925e-06, 3.8407723669266984e-13, 0, 0, 0}},
		{[]float64{13.222200959921482, 63.741266377137514, 129.45539425818563, 104.87423932251659, 34.825187773001446, 3.8595098872161198}, []float64{1, 0.12864478027547946, 1.8146789689495635e-10, -4.594790547398718e-19, 0, 0, 0}},
	}
	for i, c := range cases {
		sys := makePlant(t, c.num, c.den)
		for _, w := range []float64{1e-5, 1e-6, 1e-7, 1e-8, 1e-10} {
			want := bigHornerJ(c.num, w) / bigHornerJ(c.den, w)
			h, ok := ddSISOResponse(sys, pointAt(complex(0, w)))
			if !ok || !(cmplx.Abs(h-want) <= 1e-12*cmplx.Abs(want)) {
				t.Errorf("case %d ω=%g: ddSISOResponse = %v, %v; oracle %v", i, w, h, ok, want)
			}
		}
	}
}

// TestDDSISOResponseFailsClosedAtPole: den = (s−1)³ with exact coefficients.
// At s = 1 and s = 1+2⁻⁴⁰, where det = 2⁻¹²⁰ is below the dd rounding of
// its unit terms, ok must be false; at 1+2⁻²⁰ and off the cluster the solve
// matches (s+2)/(s−1)³ with s−1 exact.
func TestDDSISOResponseFailsClosedAtPole(t *testing.T) {
	sys := makePlant(t, []float64{1, 2}, []float64{1, -3, 3, -1})
	for _, s := range []float64{1, 1 + 0x1p-40} {
		if h, ok := ddSISOResponse(sys, pointAt(complex(s, 0))); ok {
			t.Errorf("s=1+%g: ddSISOResponse = %v, ok; want fail closed", s-1, h)
		}
	}
	for _, s := range []float64{1 + 0x1p-20, 0.5, 3} {
		d := s - 1
		want := (s + 2) / (d * d * d)
		h, ok := ddSISOResponse(sys, pointAt(complex(s, 0)))
		if !ok || !(cmplx.Abs(h-complex(want, 0)) <= 1e-12*math.Abs(want)) {
			t.Errorf("s=%v: ddSISOResponse = %v, %v; want %g", s, h, ok, want)
		}
	}
}

func BenchmarkDDSISOSolve(b *testing.B) {
	tf := &TransferFunc{
		Num: [][][]float64{{{5.52461747222729, 10.25395216896589, 7.411393861456306, 2.08816698988852, 0.22553859559127404, 0.007420390418043597}}},
		Den: [][]float64{{1, -0.008139247275845123, -7.936569881325126e-08, -2.288908801286239e-14, 0, 0, 0}},
	}
	res, err := tf.StateSpace()
	if err != nil {
		b.Fatal(err)
	}
	pt := pointAt(complex(0, 1e-3))
	for _, noise := range []bool{false, true} {
		name := "response"
		if noise {
			name = "noise"
		}
		b.Run(name, func(b *testing.B) {
			for b.Loop() {
				if _, _, _, ok := ddSISOSolve(res.Sys, pt, noise); !ok {
					b.Fatal("solve failed")
				}
			}
		})
	}
}
