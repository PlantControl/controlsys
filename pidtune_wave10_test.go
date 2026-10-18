package controlsys

import (
	"errors"
	"fmt"
	"math"
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
						pid, err := pidtuneGains(p, &pidtunePlantPoint{}, f, w, 60, sign, true, lagFloor, pidtunePIDFIterate)
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
	if passed < 8 {
		t.Errorf("BIAA3G plants: %d gate-passing designs wind, want the back-off's", passed)
	}
	onlyPassing = false
	rng := rand.New(rand.NewPCG(10, 31))
	fams := []PidtuneType{PidtuneP, PidtuneI, PidtunePI, PidtunePD, PidtunePDF}
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
// PM 6.4° with |S| 28 (PD). An explicit wc stays ungated (V7UW3M). Oracle:
// closed-loop roots, classical margin at every crossover and |S|.
func TestPidtuneAutoFirstDesignGated(t *testing.T) {
	for _, c := range []struct {
		f        PidtuneType
		num, den Poly
	}{
		{PidtuneI, Poly{0.8296, 0.008296}, Poly{1, 0.999, 0.1257}},
		{PidtunePD, Poly{0.9259305881240456, -0.25001768378306766, -14.09549687784304, -4.875339983688054}, Poly{1, 10.913817523403575, 51.69190733555752, 143.2440252325901, 183.43503385908775}},
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
		if err != nil {
			t.Fatalf("%s at explicit wc=%g: %v", name, wc, err)
		}
		if _, ms := pidtuneSignedMargins(pidtuneOracleLoop(first, c.num, c.den, 0, 0), 1e-6, 1e6); ms < 10 {
			t.Errorf("%s at explicit wc=%g: %+v |S| %.3g, want the ungated design (|S| > 10)", name, wc, first, ms)
		}
	}
}

// When no back-off crossover passes the gate, the automatic first design is
// returned if it passed the stability screen: on this plant every PID from
// the bandwidth crossover down to the floor keeps under 60° at a crossover
// or |S| > 4, so Pidtune returns the design at the bandwidth, which is
// stable. An unstable first design still errors.
func TestPidtuneAutoFirstDesignFallback(t *testing.T) {
	num, den := Poly{-1.8326830463596728, -4.165603628320087e-05}, Poly{1, 0.7334709862442443, 0.25518924748370436, 0.03900616993539339}
	p := pidtuneBackoffPlant(t, num, den, 0, 0)
	pid, err := Pidtune(p, PidtunePID, 0, nil)
	if err != nil {
		t.Fatalf("PID: %v", err)
	}
	wc, err := findCrossoverFreq(p, 0, 60)
	if err != nil {
		t.Fatal(err)
	}
	first, err := Pidtune(p, PidtunePID, wc, nil)
	if err != nil {
		t.Fatalf("PID at wc=%g: %v", wc, err)
	}
	if *pid != *first {
		t.Errorf("PID %+v, want the first design %+v", pid, first)
	}
	if stable, ok := pidtuneOracleStable(t, pid, num, den, 0, 0); !ok || !stable {
		t.Errorf("PID %+v unstable", pid)
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
// bit for bit, over designs whose tail bound extends the scan.
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
				if gate.beyond(&ctrl) == nil {
					continue
				}
				extended++
				q := gate.p
				q.low, q.loopTail = q.high, gate.tail
				if got, want := gate.tailPeak(ctrl), pidTuningSensitivityPeak(q, ctrl); got != want && !(math.IsNaN(got) && math.IsNaN(want)) {
					t.Errorf("%s %v/%v at wc=%g: tailPeak %v, scan %v", c.f, c.num, c.den, w, got, want)
				}
			}
		}
	}
	if extended < 20 {
		t.Errorf("%d designs extend the scan", extended)
	}
}
