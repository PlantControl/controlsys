package controlsys

import (
	"context"
	"errors"
	"math"
	"math/cmplx"
	"testing"
)

// pidFRDDenseAngles returns the gain crossings of c on num/den·e^(−sτ) over
// [lo, hi], on a 2e5-point log sweep, each one's angle from −1 either way
// round, and the least of them.
func pidFRDDenseAngles(num, den []float64, tau float64, c PID2, lo, hi float64) (ws, angles []float64, least float64) {
	L := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*complex(tau, 0)) * pidTuningFeedback(c, w)
	}
	least = math.Inf(1)
	const n = 200000
	prev := cmplx.Abs(L(lo)) - 1
	for k := 1; k <= n; k++ {
		w := lo * math.Exp(float64(k)*math.Log(hi/lo)/n)
		v := cmplx.Abs(L(w)) - 1
		if prev*v < 0 {
			a := 180 - math.Abs(cmplx.Phase(L(w)))*180/math.Pi
			ws, angles, least = append(ws, w), append(angles, a), math.Min(least, a)
		}
		prev = v
	}
	return ws, angles, least
}

// N4MOKA regression: a resonance at 24.94 rad/s (σ = 0.05) sits on a
// sample of data 27% apart, so the step below it turns arg G by 111° and
// is marked, but both its fits skew through the far dynamics to σ < 0:
// with no fit the count left every loop unknown and TunePIDFRD failed,
// where base returned an I at 0.2266 rad/s whose |L| there is under 0.07.
// A marked step without a fit is now unknown only for |L| >= ¼ at its
// samples, and that design is back. Oracles: companion roots and dense |S|
// and crossing sweeps of the exact plant.
func TestTunePIDFRDMarkedStepWithoutFitSmallLoop(t *testing.T) {
	num, den := []float64{106261.00510779137}, []float64{1, 2.092741436561967, 1191.710129922301, 1541.774979994378, 354129.0555862611, 174487.41408022825}
	frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, num, den, 0, 0.0030865473837564826, 266.1138651806388, 49), num, den)
	p, _ := pidTuningFRDStrictPlant(frd, nil)
	if stable, known, _ := pidTuningFRDNyquistVerdict(PID2{Ki: 0.4099362096159351, B: 1, C: 1}, p); !stable || !known {
		t.Errorf("strict count on base's I: stable %v known %v", stable, known)
	}
	r, err := TunePIDFRD(context.Background(), frd, PidtuneI, PIDTuningOptions{PhaseMargin: 60})
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	if re := pidClosedLoopMaxRe(num, den, k); !(re < 0) {
		t.Fatalf("%+v closed-loop root at Re %g", k, re)
	}
	exact := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s)
	}
	if peak, at := pidDenseSensitivity(k, exact, 1e-3, 1e3); peak > 4 {
		t.Errorf("%+v dense |S| %g at %g", k, peak, at)
	}
	if _, _, least := pidFRDDenseAngles(num, den, 0, k, 1e-3, 1e3); least < 58.5 {
		t.Errorf("%+v dense margin %g", k, least)
	}
}

// N4MOKA regression: the resonance at 0.601 rad/s (ζ = 0.0106) on data
// 4.5% apart, behind a 0.399 s delay, is resolved: its fits agree. The
// count's fitted walk left a loop passing within ¼ of 0 unknown, which
// kept the search from ranking such loops by their |S| excess, and
// TunePIDFRD failed, where base returned a PIDF at 0.3364 rad/s; that |S|
// is now the gate's, along the same fits, and the design is back. Oracles:
// a dense Nyquist winding of the delayed exact loop (+90°, a stable one
// with the PIDF's integrator), and dense |S| and crossing sweeps.
func TestTunePIDFRDResolvedResonanceBehindDelay(t *testing.T) {
	num, den, tau := []float64{-0.13113919091951914}, []float64{1, 3.724214374357123, 0.4087683494656266, 1.3421413021487942}, 0.39917469426356833
	frd := pidFRDStrictData(t, num, den, 0, 0.004935767889998792, 183.37835802704285, 244)
	for k, w := range frd.Omega {
		frd.Response[k][0][0] *= cmplx.Exp(complex(0, -w*tau))
	}
	r, err := TunePIDFRD(context.Background(), frd, PidtunePIDF, PIDTuningOptions{PhaseMargin: 70})
	if err != nil {
		t.Fatal(err)
	}
	k := *r.Controller
	exact := func(w float64) complex128 {
		s := complex(0, w)
		return pidPolyAt(num, s) / pidPolyAt(den, s) * cmplx.Exp(-s*complex(tau, 0))
	}
	if wind := pidWindingDeg(func(w float64) complex128 { return exact(w) * pidTuningFeedback(k, w) }, 1e-7, 1e8); math.Abs(wind-90) > 1 {
		t.Errorf("%+v winding %g°, want 90°", k, wind)
	}
	if peak, at := pidDenseSensitivity(k, exact, 1e-3, 1e3); peak > 4 {
		t.Errorf("%+v dense |S| %g at %g", k, peak, at)
	}
	if _, _, least := pidFRDDenseAngles(num, den, tau, k, 1e-3, 1e3); least < 66.5 {
		t.Errorf("%+v dense margin %g", k, least)
	}
}

// 5G6YXU margins: a 0 dB crossing pair between samples, where a resonance
// lifts |L| over 1 though sampled |L| stays under it, or a crossing in a
// step the chord cuts far inside the mode's circle, was missed or read
// tens of degrees off. Along the step's agreeing fits the crossings of
// base's designs land within 1% and 3° of the dense ones: a PID's 16.5° at
// 8.43 rad/s between samples at 7.87 and 8.73 (sampled |L| <= 0.74) and
// 41.7° at 1.934 (the chord read 67°), and a PIDF's 27° at 2.351 rad/s.
// Oracle: a dense crossing sweep of the exact plant.
func TestPIDTuningFRDArcCrossingsReadHiddenPair(t *testing.T) {
	for _, c := range []struct {
		name     string
		num, den []float64
		lo, hi   float64
		n        int
		k        PID2
		band     [2]float64
	}{
		{"PID", []float64{-309.2560361793322, -9.876174638164844}, []float64{1, 0.34438666780163185, 70.49826361184176, 9.517500357530164, 216.6471488211492}, 0.011090121808060715, 508.14582362123843, 104, PID2{Kp: -0.017965938252623777, Ki: -0.17052256024967036, Kd: -0.015210989528842537, B: 1, C: 1}, [2]float64{1.9, 9}},
		{"PIDF", []float64{-6.006874817954687, -9.631465973111077}, []float64{1, 0.6088284763961694, 5.978223314083145, 3.2080336945155454, 1.8717858124577227}, 1e-3, 1e3, 100, PID2{Kp: -0.026062305420954092, Ki: -0.07877168570884527, Tf: 0.2, B: 1, C: 1}, [2]float64{2, 3}},
	} {
		p, _ := pidTuningFRDStrictPlant(pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, 0, c.lo, c.hi, c.n), c.num, c.den), nil)
		ws, ms := pidTuningArcCrossingsOf(p, c.k)
		dws, das, _ := pidFRDDenseAngles(c.num, c.den, 0, c.k, c.band[0], c.band[1])
		if len(dws) < 2 {
			t.Fatalf("%s: dense crossings %v", c.name, dws)
		}
		for i, w := range dws {
			found := false
			for j, x := range ws {
				found = found || math.Abs(x/w-1) < .01 && math.Abs(ms[j]-das[i]) < 3
			}
			if !found {
				t.Errorf("%s: dense crossing %g° at %g; along arcs %v at %v", c.name, das[i], w, ms, ws)
			}
		}
	}
}

// 5G6YXU margins: on those plants TunePIDFRD certified designs whose
// margin it reported at 42.7° (PIDF, PM 45) and 61.0° (PID, PM 60) against
// dense crossings 27° and 16.5° from −1. A design it returns now, as the
// PIDF case must, reports no margin over 1° above the dense least, and a
// certified one is stable with dense |S| <= 4. Oracles: companion roots
// and dense |S| and crossing sweeps of the exact plant.
func TestTunePIDFRDArcMarginNotOverstated(t *testing.T) {
	for _, c := range []struct {
		name     string
		num, den []float64
		lo, hi   float64
		n        int
		family   PidtuneType
		pm       float64
		need     bool
	}{
		{"PIDF", []float64{-6.006874817954687, -9.631465973111077}, []float64{1, 0.6088284763961694, 5.978223314083145, 3.2080336945155454, 1.8717858124577227}, 1e-3, 1e3, 100, PidtunePIDF, 45, true},
		{"PID", []float64{-309.2560361793322, -9.876174638164844}, []float64{1, 0.34438666780163185, 70.49826361184176, 9.517500357530164, 216.6471488211492}, 0.011090121808060715, 508.14582362123843, 104, PidtunePID, 0, false},
	} {
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, 0, c.lo, c.hi, c.n), c.num, c.den)
		r, err := TunePIDFRD(context.Background(), frd, c.family, PIDTuningOptions{PhaseMargin: c.pm})
		if err != nil {
			if c.need {
				t.Errorf("%s: %v", c.name, err)
			}
			continue
		}
		k := *r.Controller
		if _, _, least := pidFRDDenseAngles(c.num, c.den, 0, k, c.lo, c.hi); r.Evidence.AchievedPhaseMargin > least+1 {
			t.Errorf("%s: %+v reports PM %g, dense least %g", c.name, k, r.Evidence.AchievedPhaseMargin, least)
		}
		if pidHasWarning(r, "not certified") {
			continue
		}
		if re := pidClosedLoopMaxRe(c.num, c.den, k); !(re < 0) {
			t.Errorf("%s: %+v certified with a closed-loop root at Re %g", c.name, k, re)
			continue
		}
		exact := func(w float64) complex128 {
			s := complex(0, w)
			return pidPolyAt(c.num, s) / pidPolyAt(c.den, s)
		}
		if peak, at := pidDenseSensitivity(k, exact, c.lo, c.hi); peak > 4 {
			t.Errorf("%s: %+v dense |S| %g at %g", c.name, k, peak, at)
		}
	}
}

// N4MOKA residual: two lightly damped pairs inside one step (-0.51±20.50j
// and -0.136±21.62j between samples at 19.78 and 22.65 rad/s, 14.5%
// apart; and a pair on 0.37-0.68 rad/s data 36% apart) turn G by about
// -300°, which wraps to +57° and +84°, so neither the arg G test nor a fit
// marked the step and TunePIDFRD certified unstable PIDFs (closed-loop
// roots at Re +1.41 and +0.043). Both samples over 4 times above both
// outer neighbours now mark the step, which leaves the count unknown for a
// loop of |L| >= ¼ there; a design comes back uncertified or not at all.
// Oracle: companion roots.
func TestTunePIDFRDTwoModesInOneStep(t *testing.T) {
	for _, c := range []pidFRDResonanceCase{
		{"two modes", []float64{0.0365817544873904, -0.9777077015159896, -0.02427050361191629, 0.03754908906616891}, []float64{1, 1.300188527522685, 888.1037948487498, 594.8523897521477, 196508.90396520577}, 0.0034594738827932975, 338.1544394890859, 86, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: 3.852386970762351, Ki: 950.5039127911411, Tf: 0.005778530933574737}},
		{"sparse", []float64{0.28606681391540345, 0.15599887825791323, -0.025452865216768972}, []float64{1, 0.1105500555199797, 0.6839221974389588, 0.03590601787107231, 0.11403658785066209}, 0.0027586746784943974, 66.85733269443266, 34, PIDTuningOptions{PhaseMargin: 45}, PID2{Kp: -0.048876886610363414, Ki: -0.0385594382370572, Kd: -0.015272911030725799, Tf: 1.213286345993017}},
	} {
		k := c.k
		k.B, k.C = 1, 1
		if re := pidClosedLoopMaxRe(c.num, c.den, k); !(re > 1e-3) {
			t.Fatalf("%s: oracle max Re %g, want unstable", c.name, re)
		}
		frd := pidFRDStrictWithDC(t, pidFRDStrictData(t, c.num, c.den, 0, c.lo, c.hi, c.n), c.num, c.den)
		p, _ := pidTuningFRDStrictPlant(frd, nil)
		if stable, known, _ := pidTuningFRDNyquistVerdict(k, p); stable && known {
			t.Errorf("%s: strict count finds unstable %+v stable", c.name, k)
		}
		r, err := TunePIDFRD(context.Background(), frd, PidtunePIDF, c.o)
		if err != nil {
			continue
		}
		if got := *r.Controller; !pidHasWarning(r, "not certified") && !(pidClosedLoopMaxRe(c.num, c.den, got) < 0) {
			t.Errorf("%s: unstable %+v certified", c.name, got)
		}
	}
}

// N4MOKA #27: on the stable fourth-order plant of
// TestTunePIDFRDFailsProvenUnstableFallback, with data cut to the band's
// lower 40% (52 samples on [0.0032, 0.0283] rad/s), the strict count
// certifies no PD, and base returned the looser count's PD, which the
// strict count finds unstable on an end above the band the data do not
// reach, with a warning. Such a design now fails, whatever end it took.
// Oracle: companion roots.
func TestTunePIDFRDFailsUnstableFallbackShortData(t *testing.T) {
	num, den := []float64{1.9069932277433254e-05}, []float64{1, 0.025199971874419462, 0.0011256420978643122, 1.0718595210231957e-05, 1.8479606353999822e-07}
	lo := 0.0031957940883252326
	frd := pidFRDStrictData(t, num, den, 0, lo, lo*math.Pow(0.7111971289538478/lo, .4), 52)
	base := PID2{Kp: 0.0015308848988514487, Kd: -0.6617940789784681, B: 1, C: 1}
	if pidFRDStrictStable(num, den, base) {
		t.Fatalf("oracle finds %+v stable", base)
	}
	p, _ := pidTuningFRDStrictPlant(frd, nil)
	if stable, known, top := pidTuningFRDNyquistVerdict(base, p); stable || !known || top {
		t.Errorf("strict verdict on base's PD: %v %v %v, want unstable on an unreached end", stable, known, top)
	}
	r, err := TunePIDFRD(context.Background(), frd, PidtunePD, PIDTuningOptions{})
	if err == nil && !pidFRDStrictStable(num, den, *r.Controller) {
		t.Errorf("unstable %+v returned", *r.Controller)
	} else if err != nil && !errors.Is(err, ErrPIDTuningTargetUnattainable) {
		t.Error(err)
	}
}
