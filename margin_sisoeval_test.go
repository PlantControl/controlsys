package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// IP3XIC: beside the real pole at 1 − 1.3e-7 the double solve fails at
// ω = 5.33e-5 off every pole, where FreqResponse takes the double-double
// response. sisoEval.at keeps that failure unless extended (forcing
// FreqResponse parity there adds spurious crossings beside origin and
// Nyquist clusters), so the extended evaluator must agree with FreqResponse
// and the margins must still match the loop. Oracle: the float64
// coefficients evaluated exactly in big.Float, phase offset bisected on a
// log grid.
func TestAllMarginBesideNearIntegratorIP3XIC(t *testing.T) {
	num := []float64{2.761969998249568e-16, 2.6332895798497864e-16, 6.196985260193668e-17}
	den := []float64{1, -5.388323209156541, 12.045405202918547, -14.292531743733242, 9.488502158799133, -3.339413745774066, 0.486361336946173}
	const dt = 0.01
	r, err := (&TransferFunc{Num: [][][]float64{{num}}, Den: [][]float64{den}, Dt: dt}).StateSpace()
	if err != nil {
		t.Fatal(err)
	}
	sys := r.Sys
	loop := func(th float64) complex128 { return bigHorner(num, th) / bigHorner(den, th) }

	w := 5.331792250825023e-05
	fr, err := sys.FreqResponse([]float64{w})
	if err != nil {
		t.Fatal(err)
	}
	want := loop(w * dt)
	if got := fr.At(0, 0, 0); cmplx.Abs(got-want) > 1e-6*cmplx.Abs(want) {
		t.Errorf("FreqResponse(%g) = %v, want %v", w, got, want)
	}
	e, err := newSISOEval(sys)
	if err != nil {
		t.Fatal(err)
	}
	e.extended = true
	if got, ok := e.try(w); !ok || cmplx.Abs(got-want) > 1e-6*cmplx.Abs(want) {
		t.Errorf("extended sisoEval(%g) = %v, %v, want %v", w, got, ok, want)
	}

	ph := func(th float64) float64 { return phaseOffsetDeg(loop(th)) }
	var thp []float64
	const N = 4000
	lo, hi := math.Log(1e-9), math.Log(math.Pi)
	for i := 1; i <= N; i++ {
		a, b := math.Exp(lo+(hi-lo)*float64(i-1)/N), math.Exp(lo+(hi-lo)*float64(i)/N)
		if i == N {
			b = math.Pi
		}
		pa, pb := ph(a), ph(b)
		if (pa > 0) == (pb > 0) || math.Abs(pa) >= 90 || math.Abs(pb) >= 90 {
			continue
		}
		for range 200 {
			m := (a + b) / 2
			if m <= a || m >= b {
				break
			}
			if pm := ph(m); (pm > 0) == (pa > 0) {
				a, pa = m, pm
			} else {
				b = m
			}
		}
		thp = append(thp, (a+b)/2)
	}
	if len(thp) != 2 {
		t.Fatalf("oracle phase crossings θ %v, want 2", thp)
	}

	res, err := AllMargin(sys)
	if err != nil {
		t.Fatal(err)
	}
	gmTol := []float64{1e-6, 1e-3}
	ok := len(res.PhaseCrossFreqs) == 2 && len(res.GainMargins) == 2 && len(res.GainCrossFreqs) == 0
	for i := 0; ok && i < 2; i++ {
		gm := -20 * math.Log10(cmplx.Abs(loop(thp[i])))
		ok = math.Abs(res.PhaseCrossFreqs[i]*dt-thp[i]) <= 1e-9*thp[i] && math.Abs(res.GainMargins[i]-gm) <= gmTol[i]
	}
	if !ok {
		t.Errorf("wpc %v gm %v wgc %v, want θ %v", res.PhaseCrossFreqs, res.GainMargins, res.GainCrossFreqs, thp)
	}

	m, err := Margin(sys)
	if err != nil {
		t.Fatal(err)
	}
	wpc, has := m.PhaseCrossover()
	gm := -20 * math.Log10(cmplx.Abs(loop(thp[0])))
	if !has || math.Abs(wpc*dt-thp[0]) > 1e-9*thp[0] || math.Abs(m.GainMargin-gm) > 1e-6 || !math.IsInf(m.PhaseMargin, 1) {
		t.Errorf("Margin: wpc %v gm %v pm %v, want θ %v gm %v", wpc, m.GainMargin, m.PhaseMargin, thp[0], gm)
	}
}
