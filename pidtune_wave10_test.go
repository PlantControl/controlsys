package controlsys

import (
	"errors"
	"fmt"
	"math"
	"math/cmplx"
	"math/rand/v2"
	"testing"
	"time"
)

// pidtuneCheckDelayRobust checks pid on num/den·e^{−delay·s} by the
// independent oracle: Nyquist winding of the directly evaluated loop, and on
// it the classical margin 180° − |∠L| ≥ 58.5° at every crossover and
// |S| ≤ 4.5.
func pidtuneCheckDelayRobust(t *testing.T, name string, pid *PID, num, den Poly, delay float64) {
	t.Helper()
	if stable, ok := pidtuneOracleStable(t, pid, num, den, 0, delay); !ok || !stable {
		t.Errorf("%s: %+v unstable (oracle ok %v)", name, pid, ok)
		return
	}
	pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, den, 0, delay), 1e-6, 1e4)
	if len(pms) == 0 || ms > 4.5 {
		t.Errorf("%s: %+v crossovers %v, |S| %.3g", name, pid, pms, ms)
	}
	for _, m := range pms {
		if math.Abs(m) < 58.5 {
			t.Errorf("%s: %+v classical margin %.3g°", name, pid, math.Abs(m))
		}
	}
}

// TFSAJU: a continuous-delay loop is gated on its signed margin 180° + ∠L
// before its Nyquist test; a design failing only at reflected crossings,
// signed margin ≤ −(pm − tol), keeps the classical margin, so with |S| ≤ 4
// it passes once the Nyquist test certifies it. Before, PD on this plant
// returned ErrPIDTuningTargetUnattainable. Oracle: Nyquist winding, classical
// margins and |S| of the directly evaluated loop.
func TestPidtuneDelayReflectedCrossingClassical(t *testing.T) {
	num, den, delay := Poly{0.7396645755398772, 0}, Poly{1, 1.819525996177783, 0.34428288351830255}, 0.5934124697362396
	p := pidtuneBackoffPlant(t, num, den, 0, delay)
	pid, err := Pidtune(p, PidtunePD, 0, nil)
	if err != nil {
		t.Fatalf("PD: %v", err)
	}
	for _, g := range []struct {
		name      string
		got, want float64
	}{{"Kp", pid.Kp, 0.294}, {"Kd", pid.Kd, 69.0}, {"Tf", pid.Tf, 23.5}} {
		if math.Abs(g.got-g.want) > 0.05*g.want {
			t.Errorf("PD %s %g, want about %g", g.name, g.got, g.want)
		}
	}
	pidtuneCheckDelayRobust(t, "PD", pid, num, den, delay)
	pms, _ := pidtuneSignedMargins(pidtuneOracleLoop(pid, num, den, 0, delay), 1e-6, 1e4)
	reflected := false
	for _, m := range pms {
		reflected = reflected || m <= -58.5
	}
	if !reflected {
		t.Errorf("PD %+v: signed margins %v, want a reflected crossing", pid, pms)
	}
}

// Every sampled winding verdict by which a gated design on a stable delayed
// plant skips its Nyquist test agrees with the test: over seeded stable
// delayed plants and designs (least-lead PDFs included) at crossovers down
// to the back-off floor, the high-gain PDs on a zero-DC plant whose tail
// bound is out of reach, and the gate-passing PDFs of the BIAA3G plants,
// each loop winds reports unstable is unstable by pidTuningDelayStability.
func TestPidtuneGateWindsAgreesWithNyquist(t *testing.T) {
	if raceEnabled {
		t.Skip("slow under race")
	}
	wound, checked, passed := 0, 0, 0
	onlyPassing := false
	run := func(num, den Poly, delay float64, fams []PidtuneType, k0, k1, dk int) {
		p := pidtuneBackoffPlant(t, num, den, 0, delay)
		wc, fromPole, err := autoCrossoverFreq(p, 0, 60)
		if err != nil {
			return
		}
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		screen, err := newPIDStabilityScreen(p, eval, 1)
		if err != nil {
			t.Fatal(err)
		}
		floor := pidtuneBackoffFloor(p, wc)
		if fromPole {
			floor = wc * math.Pow(10, -pidtuneBackoffMaxDecade)
		}
		gate := newPidtuneGate(p, screen.model, eval, wc, floor)
		for _, f := range fams {
			for k := k0; k <= k1; k += dk {
				w := wc * math.Pow(10, -float64(k)/pidtuneBackoffPerDecade)
				for _, sign := range []float64{1, -1} {
					for _, lagFloor := range []bool{false, true} {
						pid, err := pidtuneGains(p, &pidtunePlantPoint{}, f, w, 60, sign, true, lagFloor, pidtunePDLegacy, pidtunePIDFIterate)
						if err != nil {
							continue
						}
						c := pidtuneGateController(pid)
						if !gate.winds(c) {
							continue
						}
						wound++
						pass := gate.check(pid, f, w, 60, false) == nil
						if onlyPassing && !pass {
							continue
						}
						stable, _, err := screen.at(w, f)
						if err != nil || stable == nil {
							continue
						}
						ok, err := stable(&c)
						if err != nil {
							continue
						}
						checked++
						if pass {
							passed++
						}
						if ok {
							t.Errorf("%s %v/%v delay %g at wc=%g sign %g lag floor %v: %+v winds but Nyquist-stable", f, num, den, delay, w, sign, lagFloor, pid)
						}
					}
				}
			}
		}
	}
	run(Poly{0.012075199185333292, 0, 0}, Poly{1, 3.9182743506554067, 0.27016515117488543, 0.047907185397301336}, 0.11879022516031652, []PidtuneType{PidtunePD}, 60, 120, 15)
	if wound == 0 {
		t.Errorf("BIAA3G plant: no high-gain PD winds")
	}
	onlyPassing = true
	run(Poly{0.012075199185333292, 0, 0}, Poly{1, 3.9182743506554067, 0.27016515117488543, 0.047907185397301336}, 0.11879022516031652, []PidtuneType{PidtunePDF}, 0, 41, 3)
	run(Poly{0.1043562311628774, 0}, Poly{1, 2.157756223719102, 12.93308866944033}, 0.06098188011777296, []PidtuneType{PidtunePDF}, 72, 110, 4)
	t.Logf("BIAA3G plants: %d wound, %d checked, %d gate-passing", wound, checked, passed)
	if passed < 7 {
		t.Errorf("BIAA3G plants: %d gate-passing designs wind, want the back-off's", passed)
	}
	onlyPassing = false
	rng := rand.New(rand.NewPCG(10, 31))
	fams := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF, PidtunePID, PidtunePIDF}
	for trial := range 60 {
		n := 1 + rng.IntN(3)
		den := Poly{1}
		for range n {
			den = den.Mul(Poly{1, 0.05 + 3*rng.Float64()})
		}
		num := make(Poly, rng.IntN(n)+1)
		for i := range num {
			num[i] = 2*rng.Float64() - 1
		}
		if trial%3 == 0 {
			num = num.Mul(Poly{1, 0})
		}
		run(num, den, 0.05+2*rng.Float64(), fams, 0, 60, 6)
	}
	if checked < 50 {
		t.Fatalf("%d wound, %d checked by Nyquist", wound, checked)
	}
}

// AUOTFB: an automatic first design is gated like a back-off step (the
// classical margin on loops the pole test certifies); before, PI-family
// designs at the bandwidth crossover returned PM 3.2° with |S| 18 (I) and
// PM 6.4° with |S| 28 (PD). An explicit wc stays ungated (V7UW3M): |S| 24
// for the second PD; the first one's explicit-wc PD, placed at |L(wc)| = 1
// since E4I7RE, is unstable for every Td/10 filtered PD of either sign
// (oracle sweep), so it errored; MATLAB's ideal PD there (G6APMO) is stable,
// its fast pole near −2033 as Kd·N lead ≈ −0.996 nearly cancels the leading
// coefficient (|S| ≈ 230, ungated). Oracle: closed-loop roots, classical
// margin at every crossover and |S|.
func TestPidtuneAutoFirstDesignGated(t *testing.T) {
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
		ideal    bool
	}{
		{PidtuneI, Poly{0.8296, 0.008296}, Poly{1, 0.999, 0.1257}, false},
		{PidtunePD, Poly{0.9259305881240456, -0.25001768378306766, -14.09549687784304, -4.875339983688054}, Poly{1, 10.913817523403575, 51.69190733555752, 143.2440252325901, 183.43503385908775}, true},
		{PidtunePD, Poly{0.8001772693191576, -0.930259502360965}, Poly{1, 7.820477719021238, 22.734552161124903, 29.06460406628826, 13.753654621908568}, false},
	} {
		name := fmt.Sprintf("%s %v/%v", c.f, c.num, c.den)
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, 0)
		pid, err := Pidtune(p, c.f, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		pidtuneCheckRobust(t, name, pid, c.num, c.den, 0, 4.5)
		wc, err := findCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		first, err := Pidtune(p, c.f, wc, nil)
		if c.ideal {
			if err != nil || first.Tf != 0 {
				t.Errorf("%s at explicit wc=%g: %+v, %v; want the ideal PD", name, wc, first, err)
			} else if stable, ok := pidtuneOracleStable(t, first, c.num, c.den, 0, 0); !ok || !stable {
				t.Errorf("%s at explicit wc=%g: %+v unstable", name, wc, first)
			}
			continue
		}
		if err != nil {
			t.Fatalf("%s at explicit wc=%g: %v", name, wc, err)
		}
		if _, ms := pidtuneSignedMargins(pidtuneOracleLoop(first, c.num, c.den, 0, 0), 1e-6, 1e6); ms < 10 {
			t.Errorf("%s at explicit wc=%g: %+v |S| %.3g, want the ungated design (|S| > 10)", name, wc, first, ms)
		}
	}
}

// When no back-off crossover passes the gate, the automatic first design is
// returned only if it passed the stability screen, crosses 0 dB and keeps
// |S| ≤ pidTuningMaxSensitivity (IBWZGV): on this plant every PID from the
// bandwidth crossover down to the floor keeps under 60° at a crossover or
// |S| > 4, and the design at the bandwidth, returned for that explicit wc,
// has an ideal derivative with |S| 5.3 and −10.9° at a crossover, so the
// automatic design errors. An unstable first design still errors.
func TestPidtuneAutoFirstDesignFallback(t *testing.T) {
	num, den := Poly{-1.8326830463596728, -4.165603628320087e-05}, Poly{1, 0.7334709862442443, 0.25518924748370436, 0.03900616993539339}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	if pid, err := Pidtune(p, PidtunePID, 0, nil); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Errorf("PID: %+v, %v; want ErrPIDTuningTargetUnattainable", pid, err)
	}
	wc, err := findCrossoverFreq(p, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	first, err := Pidtune(p, PidtunePID, wc, nil)
	if err != nil {
		t.Fatalf("PID at wc=%g: %v", wc, err)
	}
	if stable, ok := pidtuneOracleStable(t, first, num, den, 0, 0); !ok || !stable {
		t.Errorf("PID %+v unstable", first)
	}
	if _, ms := pidtuneSignedMargins(pidtuneOracleLoop(first, num, den, 0, 0), 1e-9, 1e6); ms <= 4.5 {
		t.Errorf("PID at wc=%g: %+v |S| %.3g, want > 4.5 so the gate pins it", wc, first, ms)
	}
	for k := 1; k <= 60; k++ {
		w := wc * math.Pow(10, -float64(k)/pidtuneBackoffPerDecade)
		step, err := Pidtune(p, PidtunePID, w, nil)
		if err != nil {
			continue
		}
		stable, _ := pidtuneOracleStable(t, step, num, den, 0, 0)
		pms, ms := pidtuneSignedMargins(pidtuneOracleLoop(step, num, den, 0, 0), 1e-9, 1e6)
		robust := stable && len(pms) > 0 && ms <= 4
		for _, m := range pms {
			robust = robust && math.Abs(m) >= 59.5
		}
		if robust {
			t.Errorf("back-off wc=%g: %+v robust; the fallback should not be needed", w, step)
		}
	}

	q := pidtuneBackoffPlant(t, Poly{1}, Poly{1, -1, -2}, 0, 0)
	if pid, err := Pidtune(q, PidtuneI, 0, nil); !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Errorf("I on 1/(s²−s−2): %+v, %v; want ErrPIDTuningTargetUnattainable", pid, err)
	}
}

// BIAA3G: on stable zero-DC plants with a continuous delay, the back-off
// gate passes designs whose loop is unstable, each costing a Nyquist test
// that grew to over a second at low crossovers (4–10 s per call). The
// gate's sampled winding rejects them first (agreement with the Nyquist
// test: TestPidtuneGateWindsAgreesWithNyquist); the results are unchanged.
func TestPidtuneZeroDCDelayBackoffCost(t *testing.T) {
	if raceEnabled {
		t.Skip("timing under race")
	}
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
		delay    float64
		ok       bool
	}{
		{PidtunePDF, Poly{0.012075199185333292, 0, 0}, Poly{1, 3.9182743506554067, 0.27016515117488543, 0.047907185397301336}, 0.11879022516031652, false},
		{PidtunePDF, Poly{0.1043562311628774, 0}, Poly{1, 2.157756223719102, 12.93308866944033}, 0.06098188011777296, false},
		{PidtunePD, Poly{-0.014478772994582018, 0}, Poly{1, 17.775626209671092, 25.212313873525357}, 0.08062746075097056, true},
	} {
		name := fmt.Sprintf("%s %v/%v delay %g", c.f, c.num, c.den, c.delay)
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, c.delay)
		start := time.Now()
		pid, err := Pidtune(p, c.f, 0, nil)
		if d := time.Since(start); d > 2*time.Second {
			t.Errorf("%s: %v", name, d)
		}
		if !c.ok {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %+v, %v; want ErrPIDTuningTargetUnattainable", name, pid, err)
			}
			continue
		}
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		pidtuneCheckDelayRobust(t, name, pid, c.num, c.den, c.delay)
	}
}

// The gate's cached |S| scan above its band matches pidTuningSensitivityPeak
// bit for bit, over designs whose tail bound extends the scan within
// pidTuningTailReach times the band's top.
func TestPidtuneGateTailPeakMatchesScan(t *testing.T) {
	extended := 0
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
		delay    float64
	}{
		{PidtunePDF, Poly{0.012075199185333292, 0, 0}, Poly{1, 3.9182743506554067, 0.27016515117488543, 0.047907185397301336}, 0.11879022516031652},
		{PidtuneP, Poly{-0.7103680010513562, 0}, Poly{1, 7.438667224341966, 36.6377214599462}, 0.66426273141541514},
		{PidtunePD, Poly{-0.46375204436937656, -0.36621853843466834, 0}, Poly{1, 11.51148942688912, 43.55983162707241, 54.30021224234672}, 0.485905856269508},
		{PidtunePID, Poly{1, 2}, Poly{1, 0.4, 4, 1}, 0},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, 0, c.delay)
		wc, fromPole, err := autoCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		screen, err := newPIDStabilityScreen(p, eval, 1)
		if err != nil {
			t.Fatal(err)
		}
		floor := pidtuneBackoffFloor(p, wc)
		if fromPole {
			floor = wc * math.Pow(10, -pidtuneBackoffMaxDecade)
		}
		gate := newPidtuneGate(p, screen.model, eval, wc, floor)
		for k := 0; k <= 120; k += 3 {
			w := wc * math.Pow(10, -float64(k)/pidtuneBackoffPerDecade)
			for _, sign := range []float64{1, -1} {
				pid, err := computePIDGains(p, c.f, w, 60, sign, true)
				if err != nil {
					continue
				}
				ctrl := pidtuneGateController(pid)
				above, level := gate.beyond(&ctrl)
				if above == nil || level != 0 || above[len(above)-1] > gate.p.high*pidTuningTailReach {
					continue
				}
				extended++
				q := gate.p
				q.low, q.loopTail = q.high, gate.tail
				if got, want := gate.tailPeak(ctrl, above[len(above)-1]), pidTuningSensitivityPeak(q, ctrl); got != want && !(math.IsNaN(got) && math.IsNaN(want)) {
					t.Errorf("%s %v/%v at wc=%g: tailPeak %v, scan %v", c.f, c.num, c.den, w, got, want)
				}
			}
		}
	}
	if extended < 20 {
		t.Errorf("%d designs extend the scan", extended)
	}
}

// ZON57P: the gate's scans above its band end where the design's own tail
// bound, or leveling-off asymptote, rules out a crossover or |S| excess
// further up. Before, a fixed 1e4 times the band's top left PDF Kp 8.85e13,
// Kd 6.3e18 crossing at 1.04e7 rad/s (top 714) with PM 1e-4° and |S| 8.9e3
// unchecked, and a biproper plant's PD |S| → 15.6 as ω → ∞ unseen by a
// band below the plant's dynamics. Oracles: exact Routh test, and margins
// and |S| of the directly evaluated loop up to 1e9 rad/s; for the limit,
// 1/|1+L| and, with a delay, 1/(1−|L|) at 1e9 rad/s.
func TestPidtuneGateBandFromDesign(t *testing.T) {
	gateOf := func(p *System) (*pidtuneGate, float64) {
		t.Helper()
		wc, fromPole, err := autoCrossoverFreq(p, 0, 60)
		if err != nil {
			t.Fatal(err)
		}
		eval, err := newSISOEval(p)
		if err != nil {
			t.Fatal(err)
		}
		screen, err := newPIDStabilityScreen(p, eval, 1)
		if err != nil {
			t.Fatal(err)
		}
		floor := pidtuneBackoffFloor(p, wc)
		if fromPole {
			floor = wc * math.Pow(10, -pidtuneBackoffMaxDecade)
		}
		return newPidtuneGate(p, screen.model, eval, wc, floor), wc
	}
	robust := func(name string, f PidtuneType, pid *PID, num, den Poly) {
		t.Helper()
		if !pidtuneExactHurwitz(pid, num, den) {
			t.Errorf("%s: %+v closed loop not Hurwitz", name, pid)
			return
		}
		pm, ms := pidtuneLoopMarginBand(pidtuneOracleLoop(pid, num, den, 0, 0), 1e-10, 1e9, 1.001)
		if ms > 4.05 || f != PidtuneP && f != PidtunePD && pm < 58.5 {
			t.Errorf("%s: %+v phase margin %.3g, |S| %.3g", name, pid, pm, ms)
		}
	}

	num, den := Poly{0.8046831541982362, 2.6069650473739415e-06, 1.9527181787498443e-12}, Poly{1, 17.41109423258768, 107.8941773307406, 272.26498377608243, 215.3214628532836}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	gate, wc := gateOf(p)
	if err := gate.check(&PID{Kp: 8.851326453946484e+13, Kd: 6.297043913126736e+18, Tf: 140054.19513783758}, PidtunePDF, wc, 60, true); err == nil {
		t.Errorf("PDF crossing at 1.04e7 rad/s passes the gate")
	}
	for _, f := range []PidtuneType{PidtunePDF, PidtunePIDF} {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			if !errors.Is(err, ErrPIDTuningTargetUnattainable) {
				t.Errorf("%s: %v", f, err)
			}
			continue
		}
		robust(string(f), f, pid, num, den)
	}

	num = Poly{1.9355458998154456, 8.521119088466584, 10.888552438038644, 4.287024002086385, 7.201380557549947e-06}
	den = Poly{1, 4.131322890232889, 4.352348907079097, 0.6684842552460563, -0.0031624264514338862}
	p = pidtuneBackoffPlant(t, num, den, 0, 0)
	gate, wc = gateOf(p)
	old := &PID{Kp: -0.049979169270678435, Kd: -1336.3280290281982, Tf: 2673.7699896348404}
	if err := gate.sensitive(old, PidtunePD, wc); err == nil {
		t.Errorf("PD %+v: |S| → %.3g passes the gate", old, 1/cmplx.Abs(1+pidtuneOracleLoop(old, num, den, 0, 0)(1e9)))
	}
	for _, c := range []*PID{old, {Kp: 0.5}, {Kp: -0.45, Ki: 0.1}, {Kp: 0.1, Kd: 2, Tf: 0.5}} {
		ctrl := pidtuneGateController(c)
		_, level := gate.beyond(&ctrl)
		if want := 1 / cmplx.Abs(1+pidtuneOracleLoop(c, num, den, 0, 0)(1e9)); math.Abs(level-want) > 1e-6*want {
			t.Errorf("%+v: limit |S| %v, want %v", c, level, want)
		}
	}
	pid, err := Pidtune(p, PidtunePD, 0, nil)
	if err != nil {
		t.Fatalf("PD: %v", err)
	}
	robust("PD", PidtunePD, pid, num, den)

	num, den = Poly{0.9651311017405639, -0.3643799158344094}, Poly{1, -0.2215211283042633, 0.010177292879074453}
	gate, wc = gateOf(pidtuneBackoffPlant(t, num, den, 0, 0))
	ideal := &PID{Kp: -0.049979169270678414, Ki: -0.010151516877036266, Kd: -0.9932760277364697}
	want := func(c *PID) float64 { return 1 / cmplx.Abs(1+pidtuneOracleLoop(c, num, den, 0, 0)(1e9)) }
	c1 := pidtuneGateController(ideal)
	if _, level := gate.beyond(&c1); math.Abs(level-want(ideal)) > 1e-6*want(ideal) {
		t.Errorf("ideal-D PID %+v: limit |S| %v, want %v", ideal, level, want(ideal))
	}
	if err := gate.sensitive(ideal, PidtunePID, wc); err == nil {
		t.Errorf("ideal-D PID %+v passes the gate", ideal)
	}
	num = Poly{1.9355458998154456, 8.521119088466584, 10.888552438038644, 4.287024002086385, 7.201380557549947e-06}
	den = Poly{1, 4.131322890232889, 4.352348907079097, 0.6684842552460563, -0.0031624264514338862}

	const delay = 0.1
	pd := pidtuneBackoffPlant(t, num, den, 0, delay)
	gate, wc = gateOf(pd)
	ctrl := &PID{Kp: 0.8 / num[0]}
	l := pidtuneOracleLoop(ctrl, num, den, 0, delay)
	peak := 0.0
	for w := 1e3; w < 1e3+4*math.Pi/delay; w += 1e-3 / delay {
		peak = math.Max(peak, 1/cmplx.Abs(1+l(w)))
	}
	c2 := pidtuneGateController(ctrl)
	if _, level := gate.beyond(&c2); math.Abs(level-peak) > 1e-3*peak || peak <= pidTuningMaxSensitivity {
		t.Errorf("delayed P %+v: limit |S| %v, oracle %v", ctrl, level, peak)
	}
	if err := gate.sensitive(ctrl, PidtuneP, wc); err == nil {
		t.Errorf("delayed P %+v: |S| → %.3g passes the gate", ctrl, peak)
	}
}

// pidtuneOracleTopCrossing returns the highest ω in [1e-9, 1e9] where the
// directly evaluated |L| crosses 1, or 0 when it never does.
func pidtuneOracleTopCrossing(pid *PID, num, den Poly) float64 {
	L := pidtuneOracleLoop(pid, num, den, 0, 0)
	top, prev := 0.0, cmplx.Abs(L(1e-9))-1
	for w := 1e-9 * 1.005; w <= 1e9; w *= 1.005 {
		v := cmplx.Abs(L(w)) - 1
		if v*prev <= 0 && v != prev {
			top = w
		}
		prev = v
	}
	return top
}

// A back-off step on G = 0.001(s+1e-6)/(s²+0.01s+1) from the pole-fallback
// crossover 0.5 it took before the rolloff crossover (0a2620b) passes the
// gate with a PID Kp 2.67e7, Ki 844 crossing over again near 2.7e4, five
// decades above 0.5 (4G4ABV), robust by the oracle. The PI at 0.5, signed
// margin −93° at wc, passes on the classical margin, and the automatic
// designs, now at the rolloff crossover 10, cross over within a decade of
// it. Oracle: the directly evaluated loop and its closed-loop roots.
func TestPidtuneBackoffStaysInBand(t *testing.T) {
	num, den := Poly{0.001, 1e-9}, Poly{1, 0.01, 1}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	eval, err := newSISOEval(p)
	if err != nil {
		t.Fatal(err)
	}
	screen, err := newPIDStabilityScreen(p, eval, 1)
	if err != nil {
		t.Fatal(err)
	}
	const wc = 0.5
	gate := newPidtuneGate(p, screen.model, eval, wc, wc*math.Pow(10, -pidtuneBackoffMaxDecade))

	w := wc * math.Pow(10, -110.0/pidtuneBackoffPerDecade)
	pid, err := pidtuneGains(p, &pidtunePlantPoint{}, PidtunePID, w, 60, 1, true, false, pidtunePDLegacy, pidtunePIDFIterate)
	if err != nil {
		t.Fatal(err)
	}
	if top := pidtuneOracleTopCrossing(pid, num, den); !(pid.Kp > 1e7) || !(top > 1e3*wc) {
		t.Fatalf("PID at %g: %+v crosses over up to %g; want the 4G4ABV design", w, pid, top)
	}
	if err := gate.check(pid, PidtunePID, w, 60, true); err != nil {
		t.Errorf("PID at %g: %+v gate %v", w, pid, err)
	}
	pidtuneCheckRobust(t, "PID back-off", pid, num, den, 0, 4.5)

	pi, err := pidtuneGains(p, &pidtunePlantPoint{}, PidtunePI, wc, 60, 1, true, false, pidtunePDLegacy, pidtunePIDFIterate)
	if err != nil {
		t.Fatal(err)
	}
	if err := gate.check(pi, PidtunePI, wc, 60, true); err != nil {
		t.Errorf("PI at %g: %+v gate %v", wc, pi, err)
	}
	explicit, err := Pidtune(p, PidtunePI, wc, nil)
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(explicit.Kp/1498.1586818949-1) > 1e-6 || math.Abs(explicit.Ki/37.48520992983-1) > 1e-6 {
		t.Errorf("PI at explicit wc %g: %+v; want Kp 1498.16, Ki 37.485", wc, explicit)
	}
	pidtuneCheckRobust(t, "PI explicit", explicit, num, den, 0, 4.5)

	auto, _, err := autoCrossoverSource(p, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	for _, f := range []PidtuneType{PidtunePI, PidtunePID, PidtunePIDF} {
		pid, err := Pidtune(p, f, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", f, err)
			continue
		}
		pidtuneCheckRobust(t, string(f), pid, num, den, 0, 4.5)
		if top := pidtuneOracleTopCrossing(pid, num, den); !(top > auto/10 && top < 10*auto) {
			t.Errorf("%s: %+v crosses over up to %g, automatic wc %g", f, pid, top, auto)
		}
	}
}

// OBRXEO review: an ill-conditioned (z−1)·∏(z−pᵢ) realization can round
// DCGain to 0 instead of ∞ (∞ since 5YRXMC, so the sweep keeps both) while
// the integrator pole moves ~1e-9 off z = 1;
// the zero-DC pole fallback took that pole as the slowest, wc ~5e-8, where
// (zI−A) is singular, and every family failed "singular complex matrix".
// Plants whose clustered poles near z = 1 rounding scatters by up to 1e-2,
// some outside the unit circle, have a frequency response singular over
// decades on arm64 (not amd64, whose rounding differs) and return
// ErrInvalidArgument. Oracle: for a computed pole within
// 1.5e-6 of z = 1 the crossover is half the slowest generated pole's
// |ln p|/Dt, capped at 0.1π/Dt, to 5% (clustered poles round); designs stable by the exact Schur–Cohn
// test; failures only ErrPIDTuningTargetUnattainable or ErrInvalidArgument.
func TestPidtuneRoundedZeroDCIntegrator(t *testing.T) {
	num := Poly{-8.70643320022765e-16, 2.9186581655941266e-16, 1.3545850066477854e-16}
	den := Poly{1, -5.609886783435264, 13.099973168748164, -16.29841103348229, 11.394314088807377, -4.243853507030152, 0.6578640663921657}
	p := pidtuneBackoffPlant(t, num, den, 0.01, 0)
	wc, err := plantCrossoverFreq(p)
	if want := 0.5 * math.Log(1/0.9881300295446963) / 0.01; err != nil || math.Abs(wc-want) > 1e-3*want {
		t.Errorf("crossover %g, %v; want %g", wc, err, want)
	}
	pidtuneCheckDesigns(t, "repro", p, num, den, 0.01, []PidtuneType{PidtuneP, PidtunePI, PidtunePID, PidtunePIDF})

	scatteredNum := Poly{2.929235722501006e-16}
	scatteredDen := Poly{1, -6.606300338477126, 18.93640658504921, -30.736756689954774, 30.87235729206568, -19.6289794353736, 7.706575104770322, -1.7061947478678148, 0.16289222978811124}
	scattered := pidtuneBackoffPlant(t, scatteredNum, scatteredDen, 0.01, 0)
	for _, f := range pidtuneAllTypes {
		for _, w := range []float64{0, 1e-3} {
			pid, err := Pidtune(scattered, f, w, nil)
			switch {
			case err == nil:
				if !pidtuneExactSchur(pid, scatteredNum, scatteredDen, 0.01) {
					t.Errorf("scattered %s wc=%g: %+v unstable", f, w, pid)
				}
			case errors.Is(err, ErrSingularTransform):
				if !errors.Is(err, ErrInvalidArgument) {
					t.Errorf("scattered %s wc=%g: %v, want ErrInvalidArgument", f, w, err)
				}
			case !errors.Is(err, ErrPIDTuningTargetUnattainable):
				t.Errorf("scattered %s wc=%g: %v", f, w, err)
			}
		}
	}

	skipSlowUnderRace(t)
	rng := rand.New(rand.NewPCG(11, 3))
	kept, displaced := 0, 0
	for range 300 {
		dt := []float64{0.01, 0.1, 1}[rng.IntN(3)]
		num, den := Poly{math.Pow(10, -16+4*rng.Float64())}, Poly{1, -1}
		slow := math.Inf(1)
		for range 5 + rng.IntN(4) {
			a := math.Pow(10, -1+3*rng.Float64())
			slow = math.Min(slow, a)
			den = den.Mul(Poly{1, -math.Exp(-a * dt)})
		}
		for range rng.IntN(3) {
			num = num.Mul(Poly{1, -(rng.Float64()*1.8 - 0.9)})
		}
		p := pidtuneBackoffPlant(t, num, den, dt, 0)
		am, err := AllMargin(p)
		if err != nil || len(am.GainCrossFreqs) > 0 {
			continue
		}
		dc, err := p.DCGain()
		if err != nil || dc.At(0, 0) != 0 && !math.IsInf(dc.At(0, 0), 0) {
			continue
		}
		kept++
		poles, err := p.Poles()
		if err != nil {
			t.Fatal(err)
		}
		near := math.Inf(1)
		for _, r := range poles {
			near = math.Min(near, cmplx.Abs(r-1))
		}
		name := fmt.Sprintf("%v/%v dt=%g", num, den, dt)
		if near > 100*0x1p-52 && near <= pidtuneClusterRadius(2, 1) {
			displaced++
			want := math.Min(0.5*slow, 0.1*math.Pi/dt)
			if wc, err := plantCrossoverFreq(p); err != nil || math.Abs(wc-want) > 5e-2*want {
				t.Errorf("%s: crossover %g, %v; want %g", name, wc, err, want)
			}
		}
		for _, f := range pidtuneAllTypes {
			pid, err := Pidtune(p, f, 0, nil)
			if err != nil {
				if !errors.Is(err, ErrPIDTuningTargetUnattainable) && !errors.Is(err, ErrInvalidArgument) {
					t.Errorf("%s %s: %v", name, f, err)
				}
				continue
			}
			if !pidtuneExactSchur(pid, num, den, dt) {
				t.Errorf("%s %s: %+v unstable", name, f, pid)
			}
		}
	}
	if kept < 60 || displaced < 30 {
		t.Fatalf("sweep kept %d plants, %d displaced", kept, displaced)
	}
}

// On the zero-DC stable G = −s(0.5712s² + 0.1279s + 0.07166)/(s⁴ + …), its
// relative degree 1 makes |L| ≈ |Kp|·0.5712/ω at high ω, so a back-off P or
// PD crosses over again at about |Kp|·0.57, three decades above the first
// wc 0.526, with phase margin 90°. Rejecting every back-off design crossing
// over above the first wc's band (4G4ABV, reverted) made both error; they
// return the base designs. Oracle: closed-loop roots, margins at every
// crossover over [1e-6, 1e6] and |S|. The PD is MATLAB's ideal Kp + Kd·s
// (G6APMO), whose level Kd·CB ≈ 0.09 at high ω leaves no high crossover.
func TestPidtuneZeroDCBackoffCrossesAboveBand(t *testing.T) {
	num := Poly{-0.5711694594945204, -0.12792302726028815, -0.07166383720659307, 0}
	den := Poly{1, 6.286397495546808, 16.55207466253316, 23.123442417235935, 12.100970259959752}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	for _, c := range []struct {
		f      PidtuneType
		kp, kd float64
	}{{PidtuneP, -975.13, 0}, {PidtunePD, -7.6715, -0.16342}} {
		pid, err := Pidtune(p, c.f, 0, nil)
		if err != nil {
			t.Errorf("%s: %v", c.f, err)
			continue
		}
		if math.Abs(pid.Kp/c.kp-1) > 1e-3 || c.kd != 0 && math.Abs(pid.Kd/c.kd-1) > 1e-3 || pid.Tf != 0 {
			t.Errorf("%s: %+v; want Kp %g, Kd %g", c.f, pid, c.kp, c.kd)
		}
		pidtuneCheckRobust(t, string(c.f), pid, num, den, 0, 4.5)
		if top := pidtuneOracleTopCrossing(pid, num, den); c.f == PidtuneP && !(top > 300) {
			t.Errorf("%s: %+v crosses over up to %g; want the high crossover", c.f, pid, top)
		}
	}
}
