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
// crossover too. Beside five poles near z = 1, some just outside, the
// pencil root of L − L~ lies 3–39% above the crossing, past polish's 1%
// brackets (wideBracket; "root 25% above" reported 257.35 dB for −56.0 dB),
// or the double solve's phase noise leaves the bisected crossing at
// |f| ≈ 2e-7°, above tol (exactCrossing). Beside a ring of five to eight
// poles around z = 1 the pencil roots there are off the crossings by up to
// 6×, or missing, and the crossing below or between them was lost: "ring of
// 8 positive root" reported 320.7 dB for 14.8 dB (resonanceCrossings scans
// the ring). Oracle: the float64 coefficients evaluated exactly in
// big.Float; each crossing is bisected on the sign of the phase offset
// within 2% of the expected ω, and every reported crossing below Nyquist
// must change that sign across ω(1 ± 1e-3). Margin reports it, or one
// closer to 0 dB.
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
		{"root 25% above", []float64{5.487360777769021e-12, 3.729811085540435e-13, -8.118722465765552e-13}, []float64{1, -4.981971570006486, 9.92797413773678, -9.892092933713297, 4.928149734253077, -0.9820593682700745}, 0, 0.0352279},
		{"root 39% above", []float64{7.121922613723373e-09, -1.6095090078585384e-10, -2.424578746534387e-09}, []float64{1, -4.990857337380961, 9.96345770234538, -9.945229051503567, 4.963514345499375, -0.9908856589602281}, 0, 0.0401802},
		{"root 20% above", []float64{1.1573500080461832e-13, -1.1957101653994626e-14, 2.967072567576043e-16}, []float64{1, -4.998431428203281, 9.993728499751846, -9.990596926776579, 4.993734067111801, -0.9984342118837876}, 0, 0.0673995},
		{"root 3% above", []float64{1.7447480602492356e-08, 2.7085530777553863e-08, 9.913661534886364e-09}, []float64{1, -4.997272067895437, 9.989094639618557, -9.98365150105297, 4.989107354839763, -0.9972784255099109}, 0, 0.128026},
		{"phase noise 2e-7°", []float64{5.773354020211484e-11, 3.4781882066641815e-11, 2.9150543617647506e-12}, []float64{1, -4.991306713692669, 9.965259791780475, -9.947939025509188, 4.9653255305209445, -0.9913395830995323}, 0, 0.15503},
		{"ring of 5 below root", []float64{2.8120621156929326e-07, 3.52296574605969e-07, 8.824538827371326e-08}, []float64{1, -4.99774725235864, 9.990990156853018, -9.986486956145953, 4.990992451167435, -0.9977483995158617}, 0, 0.0146968231},
		{"ring of 8 positive root", []float64{2.8357027642397935e-15, -2.5512293296672896e-15, 1.7314681787761854e-16}, []float64{1, -7.991403327904192, 27.939842959360206, -55.81958786155226, 69.69941139400636, -55.69950967106022, 27.819764760250035, -7.93994123641546, 0.991422983315524}, 0, 1.15248109},
		{"ring of 6", []float64{3.086531715571932e-05, 4.748257100155147e-06, -1.6656262775276625e-06}, []float64{1, -5.999572410897895, 14.997861879983766, -19.995723410958078, 14.995723061950578, -5.997861356472517, 0.9995722363941447}, 0, 0.167802713},
		{"ring of 7 delay 1", []float64{2.27310135283274e-06, -1.3781887605808078e-06}, []float64{1, -6.995478220522176, 20.972874243896655, -34.93219791120347, 34.909613616290855, -20.932222512785664, 6.972893925162413, -0.9954831408386168}, 1, 0.387276661},
		{"ring of 8 delay 2", []float64{2.721976279115744e-13, -4.01726448639882e-14}, []float64{1, -8.025514241566164, 28.178884493191823, -56.53750970288973, 70.89727957862917, -56.89871269465865, 28.540089316110276, -8.180317616500474, 1.0258008676837358}, 2, 1.26678822},
		{"ring of 8 root 0.7° off", []float64{-1.2048083466934607e-12, -2.7995627145162057e-13}, []float64{1, -7.995926827451257, 27.971495050605192, -55.91450691976352, 69.85754780053844, -55.857584055828376, 27.914572179288236, -7.9715313058998305, 0.9959340785111253}, 0, 1.45607797},
		{"ring of 8 roots unevaluable", []float64{4.3420595630455874e-05}, []float64{1, -7.911922324338063, 27.386850254032066, -54.17065797954497, 66.96781846756932, -52.984417794578036, 26.200537374436056, -7.403450591516228, 0.9152425939395876}, 0, 0.43363441},
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
		w, ok := m.PhaseCrossover()
		if closer := ok && math.Abs(m.GainMargin) < math.Abs(gm)-1e-3; !closer && (!ok || math.Abs(w-wpc) > 1e-6*wpc || math.Abs(m.GainMargin-gm) > 1e-3) {
			t.Errorf("%s: Margin wpc %v gm %v, want %v gm %v", c.name, w, m.GainMargin, wpc, gm)
		}
	}
}

// S2RNHI: sisoEval.at fails where the double solve is singular off every
// pole (w12 #52) and snaps to +Inf next to a computed pole tens of times
// slower than ω (#54); at keeps both for the searches tuned to them, and
// resolve returns FreqResponse's value, leaving try's bookkeeping, which
// the screens sharing pidtune's eval read, as it was. Oracle: the float64
// coefficients evaluated exactly in big.Float.
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
		if e.err != nil || e.skipped != nil || e.resolved {
			t.Errorf("%v: resolve left err %v, skipped %v, resolved %v", c.den, e.err, e.skipped, e.resolved)
		}
	}
}
