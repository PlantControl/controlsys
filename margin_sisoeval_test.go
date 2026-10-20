package controlsys

import (
	"math"
	"math/cmplx"
	"slices"
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

// S2RNHI: beside a cluster of seven poles near z = 1 the double solve fails
// over decades, and the phase search, which evaluated in double, lost the
// crossings there: Margin reported 232.6 dB for a loop whose first phase
// crossover has 65.4 dB. Discrete delays are absorbed, so a 3-sample delay
// takes the same search. The k(z−c)/((z−1)·∏(z−pᵢ)) loops miss a lower
// crossover too. Oracle: the float64 coefficients evaluated exactly in
// big.Float; each crossing is bisected on the sign of the phase offset
// within 2% of the expected ω, and every reported crossing below Nyquist
// must change that sign across ω(1 ± 1e-3).
func TestAllMarginPhaseCrossBesideUnresolvedBand(t *testing.T) {
	clusterDen := []float64{1, -6.184746550403649, 16.242438259051475, -23.437999081676736, 20.022569140290482, -10.09289718715969, 2.766371287397997, -0.31573586749987936}
	cases := []struct {
		name     string
		num, den []float64
		delay    float64
		w        float64
	}{
		{"cluster", []float64{1.1974308853542987e-16}, clusterDen, 0, 0.18086},
		{"cluster delay 3", []float64{1.1974308853542987e-16}, clusterDen, 3, 0.18009},
		{"integrator zeros", []float64{4.172283466864154e-13, -1.0149106030767263e-13, -9.704345874936964e-14}, []float64{1, -8.658632958548061, 33.30994461652578, -74.7257981351094, 107.72944585819133, -103.5040112540225, 66.27293323893296, -27.269332855627272, 6.542921882949435, -0.6974703932922721}, 0, 3.4366},
		{"integrator lags", []float64{2.52350687932804e-16}, []float64{1, -7.767315112326138, 26.384037379634194, -51.19029458648767, 62.04694508267318, -48.109553101453315, 23.30299133761263, -6.446649260641789, 0.779838260988911}, 0, 3.3429},
	}
	const dt = 0.01
	for _, c := range cases {
		loop := func(w float64) complex128 {
			th := w * dt
			return bigHorner(c.num, th) / bigHorner(c.den, th) * cmplx.Exp(complex(0, -th*c.delay))
		}
		off := func(w float64) float64 { return phaseOffsetDeg(loop(w)) }
		a, b := c.w*0.98, c.w*1.02
		fa := off(a)
		if fa*off(b) >= 0 {
			t.Fatalf("%s: oracle has no crossing in [%g, %g]", c.name, a, b)
		}
		for range 100 {
			m := (a + b) / 2
			if fm := off(m); fm*fa > 0 {
				a, fa = m, fm
			} else {
				b = m
			}
		}
		wpc := (a + b) / 2
		gm := -20 * math.Log10(cmplx.Abs(loop(wpc)))

		sys := makeDiscretePlant(t, c.num, c.den, dt)
		if c.delay > 0 {
			if err := sys.SetInputDelay([]float64{c.delay}); err != nil {
				t.Fatal(err)
			}
		}
		all, err := AllMargin(sys)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		i := slices.IndexFunc(all.PhaseCrossFreqs, func(w float64) bool { return math.Abs(w-wpc) <= 1e-6*wpc })
		if i < 0 || math.Abs(all.GainMargins[i]-gm) > 1e-3 {
			t.Errorf("%s: wpc %v gm %v, want ω %v gm %v", c.name, all.PhaseCrossFreqs, all.GainMargins, wpc, gm)
		}
		for _, w := range all.PhaseCrossFreqs {
			if w < math.Pi/dt*(1-1e-3) && off(w*(1-1e-3))*off(w*(1+1e-3)) >= 0 {
				t.Errorf("%s: reported crossing %g is no oracle sign change", c.name, w)
			}
		}
		m, err := Margin(sys)
		if err != nil {
			t.Fatalf("%s: %v", c.name, err)
		}
		if w, ok := m.PhaseCrossover(); !ok || math.Abs(w-wpc) > 1e-6*wpc || math.Abs(m.GainMargin-gm) > 1e-3 {
			t.Errorf("%s: Margin wpc %v gm %v, want %v gm %v", c.name, w, m.GainMargin, wpc, gm)
		}
	}
}

// S2RNHI: sisoEval.at fails where the double solve is singular off every
// pole (w12 #52) and snaps to +Inf next to a computed pole tens of times
// slower than ω (#54); at keeps both for the searches tuned to them, and
// resolve returns FreqResponse's value. Oracle: the float64 coefficients
// evaluated exactly in big.Float.
func TestSISOEvalResolvedBesideClusterAtOne(t *testing.T) {
	const dt = 0.01
	cases := []struct {
		num, den []float64
		ws       []float64
	}{
		{[]float64{1.1974308853542987e-16}, []float64{1, -6.184746550403649, 16.242438259051475, -23.437999081676736, 20.022569140290482, -10.09289718715969, 2.766371287397997, -0.31573586749987936}, []float64{1e-6, 1e-4, 1e-3, 0.01, 0.1043}},
		{[]float64{6.110773630519141e-14, 3.4198170787824316e-15}, []float64{1, -5.71016988999306, 13.887907770588715, -18.645468986193702, 14.92072833819654, -7.1153754170879235, 1.8719233030765783, -0.20954511858714697}, []float64{1.9111864778082823e-05}},
	}
	for _, c := range cases {
		sys := makeDiscretePlant(t, c.num, c.den, dt)
		e, err := newSISOEval(sys)
		if err != nil {
			t.Fatal(err)
		}
		for _, w := range c.ws {
			want := bigHorner(c.num, w*dt) / bigHorner(c.den, w*dt)
			fr, err := sys.FreqResponse([]float64{w})
			if err != nil {
				t.Fatal(err)
			}
			got, ok := e.resolve(w)
			if !ok || cmplx.Abs(got-want) > 1e-6*cmplx.Abs(want) || cmplx.Abs(got-fr.At(0, 0, 0)) > 1e-12*cmplx.Abs(want) {
				t.Errorf("%v: resolve(%g) = %v, %v; FreqResponse %v, want %v", c.den, w, got, ok, fr.At(0, 0, 0), want)
			}
		}
		if e.err != nil {
			t.Errorf("%v: resolve left err %v", c.den, e.err)
		}
	}
}
