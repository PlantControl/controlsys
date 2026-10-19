package controlsys

import (
	"math"
	"math/cmplx"
	"testing"
)

// V7UW3M: an explicit wc is honoured as MATLAB pidtune(sys,type,wc) does:
// the design at wc is returned when the closed loop is stable, even though
// the loop crosses over elsewhere with a negative signed margin and |S| far
// above the back-off gate (PM −157°/−77°, |S| 103/9.9 here). Oracle:
// closed-loop roots from num/den, and the directly evaluated loop.
func TestPidtuneExplicitWcHonoured(t *testing.T) {
	num := Poly{0.17032241632981948, 1.4143934743506665}
	den := Poly{1, -1.8341193107935498, 0.756547032348602}
	dt := 0.3419630736464782
	p := pidtuneBackoffPlant(t, num, den, dt, 0)
	for _, wc := range []float64{0.612, 0.704} {
		pid, err := Pidtune(p, PidtunePIDF, wc, nil)
		if err != nil {
			t.Fatalf("wc=%g: %v", wc, err)
		}
		if stable, ok := pidtuneOracleStable(t, pid, num, den, dt, 0); !ok || !stable {
			t.Fatalf("wc=%g: %+v unstable", wc, pid)
		}
		L := pidtuneOracleLoop(pid, num, den, dt, 0)
		l := L(wc)
		if math.Abs(cmplx.Abs(l)-1) > 1e-6 {
			t.Errorf("wc=%g: |L(wc)| = %.10g, want 1", wc, cmplx.Abs(l))
		}
		if pm := 180 + cmplx.Phase(l)*180/math.Pi; math.Abs(pm-60) > 1e-3 {
			t.Errorf("wc=%g: PM at wc %.6g°, want 60°", wc, pm)
		}
		pms, ms := pidtuneSignedMargins(L, 1e-5, math.Pi/dt*(1-1e-9))
		worst := math.Inf(1)
		for _, pm := range pms {
			worst = math.Min(worst, pm)
		}
		if worst >= 0 || ms <= pidTuningMaxSensitivity {
			t.Errorf("wc=%g: crossovers %v, |S| %.3g: want a negative margin elsewhere and |S| > %g, so the test pins an ungated explicit wc", wc, pms, ms, float64(pidTuningMaxSensitivity))
		}
	}
}

// V7UW3M: an explicit wc also takes the PIDF filter fallbacks (Tf = 0.1/wc,
// else the off-target Td/10 design) ungated, as MATLAB returns its design at
// wc: these returned ErrPIDTuningTargetUnattainable because each stable
// alternative failed the back-off gate. The forward-Euler fallback for an
// unstable filter pole stays gated (YXFRDR). Oracle: closed-loop roots from
// num/den and the directly evaluated loop at wc.
func TestPidtuneExplicitWcFallbacksUngated(t *testing.T) {
	for _, c := range []struct {
		name     string
		num, den Poly
		dt, wc   float64
		form     PIDFormula
		onTarget bool
	}{
		{"default Tf", Poly{0.4655217158482053}, Poly{1, -1.0252230604287536, 0.27405286563857284, -0.010226466534870802}, 0.45679538738826414, 1.4802853629918067, BackwardEuler, true},
		{"missed Td/10", Poly{0.4315307186960532, 4.998130605056312}, Poly{1, -1.901994088838803, 1.0959292828093068, -0.20182982416756298, 0.007894630197059186}, 0.48124571872394317, 1.0569526043728357, ForwardEuler, false},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		pid, err := Pidtune(p, PidtunePIDF, c.wc, nil)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, 0); !ok || !stable {
			t.Errorf("%s: %+v unstable", c.name, pid)
		}
		if pid.DFormula != c.form {
			t.Errorf("%s: DFormula %v, want %v", c.name, pid.DFormula, c.form)
		}
		l := pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0)(c.wc)
		if math.Abs(cmplx.Abs(l)-1) > 1e-6 {
			t.Errorf("%s: |L(wc)| = %.10g, want 1", c.name, cmplx.Abs(l))
		}
		pm := 180 + cmplx.Phase(l)*180/math.Pi
		if c.onTarget != (math.Abs(pm-60) < 1e-3) {
			t.Errorf("%s: PM at wc %.6g°, on target %v", c.name, pm, c.onTarget)
		}
		if _, err := Pidtune(p, PidtunePIDF, c.wc, &PidtuneOptions{PhaseMargin: 60}); c.onTarget != (err == nil) {
			t.Errorf("%s: explicit 60° margin: %v", c.name, err)
		}
	}
}

// E4I7RE: an explicit-wc PD places |L(wc)| = 1, as MATLAB crosses over at
// the given wc. The Td/10 filter used to be left out of the phase and gain,
// so |L(wc)| was 0.49 at the upper phase clamp and 1.00025 at the lower one
// (the plants here). The phase is now placed on the realized response: the
// requested lead, else the lower clamp 0.05 rad, else the most lead of a
// Td/10 PD (atan(1.1x) − atan(0.1x) at x = wc·Td = 1/√0.11, 56.44°, for a
// continuous plant). When that design is unstable, the legacy Td with its
// gain placed is taken (zero-DC s²/(s²+…)). Oracle: closed-loop roots and
// L = C·num/den evaluated directly.
func TestPidtuneExplicitWcPDCrossesAtWc(t *testing.T) {
	x := 1 / math.Sqrt(0.11)
	most := math.Atan(1.1*x) - math.Atan(0.1*x)
	for _, c := range []struct {
		name     string
		num, den Poly
		dt, wc   float64
		form     PIDFormula
		lead, td float64
	}{
		{"most lead", Poly{3.421208948435586}, Poly{1, 4.795098433458527, 6.9583681203155, 3.1763365569137942}, 0, 5.616593566028197, ForwardEuler, most, x / 5.616593566028197},
		{"lower clamp", Poly{3.0280957501733172}, Poly{1, 4.64873917554455, 5.398081602694972}, 0, 1.0808388293067428, ForwardEuler, 0.05, 0},
		{"lower clamp discrete", Poly{0.34196211973925655}, Poly{1, -0.8966232705508059}, 0.11555016600854891, 2.836482715451361, BackwardEuler, 0.05, 0},
		{"legacy Td", Poly{0.9108330442863755, 0, 0}, Poly{1, 3.6385316848945637, 2.197552090147012}, 0, 0.3823341170934764, ForwardEuler, math.NaN(), 48.62872358299712 / 0.9303964644639571},
	} {
		p := pidtuneBackoffPlant(t, c.num, c.den, c.dt, 0)
		pid, err := Pidtune(p, PidtunePD, c.wc, nil)
		if err != nil {
			t.Errorf("%s: %v", c.name, err)
			continue
		}
		if stable, ok := pidtuneOracleStable(t, pid, c.num, c.den, c.dt, 0); !ok || !stable {
			t.Errorf("%s: %+v unstable", c.name, pid)
		}
		if pid.DFormula != c.form || math.Abs(pid.Tf-pid.Kd/pid.Kp/10) > 1e-12*pid.Tf {
			t.Errorf("%s: %+v, want %v and Tf = Td/10", c.name, pid, c.form)
		}
		l := pidtuneOracleLoop(pid, c.num, c.den, c.dt, 0)(c.wc)
		if math.Abs(cmplx.Abs(l)-1) > 1e-9 {
			t.Errorf("%s: |L(wc)| = %.12g, want 1", c.name, cmplx.Abs(l))
		}
		z := complex(0, c.wc)
		if c.dt > 0 {
			z = cmplx.Exp(complex(0, c.wc*c.dt))
		}
		if lead := cmplx.Phase(l / (c.num.Eval(z) / c.den.Eval(z))); !math.IsNaN(c.lead) && math.Abs(lead-c.lead) > 1e-6 {
			t.Errorf("%s: ∠C(wc) = %.9g rad, want %.9g", c.name, lead, c.lead)
		}
		if td := pid.Kd / pid.Kp; c.td != 0 && math.Abs(td-c.td) > 1e-6*c.td {
			t.Errorf("%s: Td %.9g, want %.9g", c.name, td, c.td)
		}
	}
}
